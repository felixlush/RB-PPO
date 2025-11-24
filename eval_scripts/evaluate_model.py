import argparse
import os
import sys
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


from config.patient_splits import patients_test
from train_utils import make_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained PPO controller across patients and seeds.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model_path", required=True, help="Path to the trained PPO .zip checkpoint."
    )
    parser.add_argument(
        "--vecnorm_path",
        help="Optional VecNormalize statistics to reuse (recommended).",
    )
    parser.add_argument(
        "--patients",
        nargs="+",
        default=patients_test,
        help="List of patient ids to evaluate.",
    )
    parser.add_argument(
        "--episodes_per_patient",
        type=int,
        default=30,
        help="Number of evaluation episodes (24h sims) per patient.",
    )
    parser.add_argument(
        "--seed_offset",
        type=int,
        default=10_000,
        help="Base seed offset; per-episode seeds increment from here.",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic actions during evaluation.",
    )
    parser.add_argument(
        "--debug_env",
        action="store_true",
        help="Enable debug flag when constructing environments.",
    )
    parser.add_argument(
        "--output_csv",
        default=os.path.join("results", "evaluation_metrics.csv"),
        help="Where to write the aggregated episode metrics.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device for loading the PPO model (e.g. cpu, cuda, auto).",
    )
    parser.add_argument(
        "--env_type",
        type=str,
        default="deterministic_bolus",
        choices=("deterministic_bolus", "no_bolus", "no_gate"),
        help="Which environment wrapper to use during evaluation.",
    )
    return parser.parse_args()


def build_eval_env(
    patient: str,
    seed: int,
    vecnorm_path: str | None,
    debug: bool,
    env_type: str,
) -> VecNormalize:
    env_fn = make_env(patient, env_type, seed=seed, debug=debug)
    vec_env = DummyVecEnv([env_fn])
    vec_env = VecMonitor(vec_env)
    if vecnorm_path:
        vec_env = VecNormalize.load(vecnorm_path, vec_env)
    else:
        vec_env = VecNormalize(
            vec_env,
            norm_obs=True,
            norm_reward=False,
            training=False,
        )
    vec_env.training = False
    vec_env.norm_reward = False
    return vec_env


def compute_episode_metrics(step_infos: List[Dict[str, Any]]) -> Dict[str, float]:
    bg_vals: List[float] = []
    risks: List[float] = []
    lbgis: List[float] = []
    hbgis: List[float] = []
    tir_streaks: List[int] = []
    tir_bonus: List[float] = []
    bolus_units: float = 0.0
    basal_units: float = 0.0

    for info in step_infos:
        bg = info.get("rbppo_bg_mgdl", info.get("bg"))
        if bg is not None:
            bg_val = float(bg)
            bg_vals.append(bg_val)
        if "risk" in info:
            risks.append(float(info["risk"]))
        if "lbgi" in info:
            lbgis.append(float(info["lbgi"]))
        if "hbgi" in info:
            hbgis.append(float(info["hbgi"]))
        if "rbppo_tir_streak" in info:
            tir_streaks.append(int(info["rbppo_tir_streak"]))
        if "rbppo_tir_bonus" in info:
            tir_bonus.append(float(info["rbppo_tir_bonus"]))

        bolus_units += float(info.get("rbppo_bolus_units", 0.0))
        basal_rate = float(info.get("rbppo_basal_u_per_min", 0.0))
        dt = float(info.get("sample_time", 5.0))
        basal_units += basal_rate * dt

    def safe_mean(values: Sequence[float]) -> float:
        return float(np.mean(values)) if values else float("nan")

    metrics: Dict[str, float] = {
        "bg_mean": safe_mean(bg_vals),
        "bg_min": float(np.min(bg_vals)) if bg_vals else float("nan"),
        "bg_max": float(np.max(bg_vals)) if bg_vals else float("nan"),
        "risk_mean": safe_mean(risks),
        "lbgi_mean": safe_mean(lbgis),
        "hbgi_mean": safe_mean(hbgis),
        "tir_bonus_mean": safe_mean(tir_bonus),
        "tir_streak_max": float(max(tir_streaks)) if tir_streaks else 0.0,
        "insulin_total_units": bolus_units + basal_units,
        "insulin_basal_units": basal_units,
        "insulin_bolus_units": bolus_units,
    }

    if bg_vals:
        bg_array = np.asarray(bg_vals)
        metrics["tir_70_180"] = float(np.mean((bg_array >= 70.0) & (bg_array <= 180.0)))
        metrics["percent_low"] = float(np.mean(bg_array < 70.0))
        metrics["percent_high"] = float(np.mean(bg_array > 180.0))
        metrics["severe_hypo_events"] = float(np.sum(bg_array < 54.0))
    else:
        metrics["tir_70_180"] = float("nan")
        metrics["percent_low"] = float("nan")
        metrics["percent_high"] = float("nan")
        metrics["severe_hypo_events"] = float("nan")

    return metrics


def run_episode(model: PPO, env: VecNormalize, deterministic: bool) -> Dict[str, Any]:
    obs = env.reset()
    step_infos: List[Dict[str, Any]] = []

    done = False
    ep_reward = 0.0
    ep_len = 0
    state = None

    while not done:
        action, state = model.predict(obs, state=state, deterministic=deterministic)
        obs, rewards, dones, infos = env.step(action)
        reward = float(rewards[0])
        info = infos[0]
        ep_reward += reward
        ep_len += 1
        step_infos.append(dict(info))
        done = bool(dones[0])

    metrics = compute_episode_metrics(step_infos)
    metrics["episode_reward"] = ep_reward
    metrics["episode_length"] = ep_len
    metrics["tir_steps_recorded"] = float(len(step_infos))
    return metrics


def main() -> None:
    args = parse_args()
    output_path = Path(args.output_csv)
    if output_path.parent != Path(""):
        output_path.parent.mkdir(parents=True, exist_ok=True)

    model = PPO.load(args.model_path, device=args.device)
    model.training = False

    results: List[Dict[str, Any]] = []

    for patient_idx, patient in enumerate(args.patients):
        for ep_idx in range(args.episodes_per_patient):
            seed = args.seed_offset + ep_idx + patient_idx * 1000
            env = build_eval_env(
                patient, seed, args.vecnorm_path, args.debug_env, args.env_type
            )
            try:
                metrics = run_episode(model, env, deterministic=args.deterministic)
            finally:
                env.close()

            metrics.update(
                {
                    "patient": patient,
                    "episode_index": ep_idx,
                    "seed": seed,
                    "deterministic": args.deterministic,
                }
            )
            results.append(metrics)
            print(
                f"[{patient}] episode {ep_idx + 1}/{args.episodes_per_patient} "
                f"reward={metrics['episode_reward']:.2f} tir={metrics['tir_70_180']:.3f} "
                f"ep_len={metrics['episode_length']}"
            )

    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    print(f"Saved evaluation metrics for {len(results)} episodes to {output_path}")


if __name__ == "__main__":
    main()
