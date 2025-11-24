#!/usr/bin/env python3
"""
Evaluate classical simglucose controllers (BB, PID) using the same metrics as PPO runs.

Example:
    python evaluate_classic_controllers.py \
        --controllers bb pid \
        --patients adult#009 adult#010 child#009 child#010 adolescent#009 adolescent#010 \
        --episodes_per_patient 100 \
        --output_csv results/classic/bb_pid_eval.csv
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from evaluate_model import compute_episode_metrics
from simglucose.controller.basal_bolus_ctrller import BBController
from simglucose.controller.pid_ctrller import PIDController
from simglucose.controller.base import Controller, Action
from simglucose.envs import T1DSimGymnaisumEnv


# Controller factory definitions
def _make_bb() -> Controller:
    return BBController()


def _make_pid() -> Controller:
    # Tuned parameters can be adjusted if needed
    return PIDController(P=0.001, I=0.00001, D=0.001, target=140)


CONTROLLER_FACTORIES: Dict[str, Controller] = {
    "bb": _make_bb,
    "pid": _make_pid,
}


@dataclass
class EpisodeResult:
    metrics: Dict[str, float]
    controller: str
    patient: str
    episode_index: int
    seed: int


def run_episode(
    controller: Controller, patient: str, seed: int, max_episode_steps: int
) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    gym_env = T1DSimGymnaisumEnv(patient_name=patient, seed=seed)
    try:
        try:
            gym_env.reset(seed=seed)
        except TypeError:
            gym_env.reset()

        # Access the underlying simglucose.simulation.env.T1DSimEnv instance
        core_env = getattr(gym_env, "env", None)
        if core_env is None:
            raise RuntimeError("Gymnasium wrapper did not expose inner env.")
        core_env = getattr(core_env, "env", core_env)

        step = core_env.reset()
        controller.reset()
        step_infos: List[Dict[str, float]] = []
        episode_reward = 0.0
        episode_length = 0

        while True:
            info_dict = getattr(step, "info", {})
            sample_time = info_dict.get(
                "sample_time", getattr(step, "sample_time", 5.0)
            )
            patient_name = info_dict.get(
                "patient_name", getattr(step, "patient_name", patient)
            )
            meal = info_dict.get("meal", getattr(step, "meal", 0.0))

            action: Action = controller.policy(
                step.observation,
                reward=step.reward,
                done=step.done,
                sample_time=sample_time,
                patient_name=patient_name,
                meal=meal,
            )

            info = {
                "sample_time": sample_time,
                "patient_name": patient_name,
                "meal": meal,
                "bg": info_dict.get("bg", getattr(step, "bg", np.nan)),
                "lbgi": info_dict.get("lbgi", getattr(step, "lbgi", np.nan)),
                "hbgi": info_dict.get("hbgi", getattr(step, "hbgi", np.nan)),
                "risk": info_dict.get("risk", getattr(step, "risk", np.nan)),
                "rbppo_basal_u_per_min": float(action.basal),
                "rbppo_bolus_units": float(action.bolus * sample_time),
                "rbppo_gate": 1.0,
                "rbppo_bolus_event": bool(meal > 0),
                "rbppo_correction": float(action.bolus),
            }
            step_infos.append(info)

            step = core_env.step(action)
            episode_reward += step.reward
            episode_length += 1
            if step.done or episode_length >= max_episode_steps:
                break

        metrics = compute_episode_metrics(step_infos)
        metrics.update(
            {
                "episode_reward": episode_reward,
                "episode_length": episode_length,
                "tir_steps_recorded": float(len(step_infos)),
            }
        )
        return metrics, step_infos
    finally:
        gym_env.close()


def evaluate_controllers(
    controllers: Iterable[str],
    patients: Iterable[str],
    episodes_per_patient: int,
    seed_offset: int,
    max_episode_steps: int,
) -> pd.DataFrame:
    results: List[Dict[str, float]] = []

    for controller_name in controllers:
        factory = CONTROLLER_FACTORIES.get(controller_name.lower())
        if factory is None:
            raise ValueError(
                f"Unknown controller '{controller_name}'. Options: {list(CONTROLLER_FACTORIES)}"
            )

        for patient_idx, patient in enumerate(patients):
            print(f"Evaluating controller '{controller_name}' on patient '{patient}'")
            for ep_idx in range(episodes_per_patient):
                print(f"  Episode {ep_idx + 1}/{episodes_per_patient}...")
                seed = seed_offset + ep_idx + patient_idx * 1000
                controller = factory()
                metrics, _ = run_episode(
                    controller, patient, seed, max_episode_steps=max_episode_steps
                )
                metrics.update(
                    {
                        "controller": controller_name,
                        "patient": patient,
                        "episode_index": ep_idx,
                        "seed": seed,
                        "deterministic": True,
                    }
                )
                results.append(metrics)
                print(
                    f"[{controller_name}] patient={patient} episode={ep_idx + 1}/{episodes_per_patient} "
                    f"TIR={metrics.get('tir_70_180', float('nan')):.3f}"
                )

    return pd.DataFrame(results)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate simglucose classical controllers using PPO metrics."
    )
    parser.add_argument(
        "--controllers",
        nargs="+",
        default=["bb", "pid"],
        help=f"Controllers to evaluate. Available: {list(CONTROLLER_FACTORIES)}",
    )
    parser.add_argument(
        "--patients",
        nargs="+",
        required=True,
        help="Patient IDs to evaluate (e.g. adult#009 adolescent#009).",
    )
    parser.add_argument(
        "--episodes_per_patient",
        type=int,
        default=30,
        help="Number of episodes per patient.",
    )
    parser.add_argument(
        "--seed_offset",
        type=int,
        default=10_000,
        help="Seed offset to ensure reproducibility.",
    )
    parser.add_argument(
        "--max_episode_steps",
        type=int,
        default=288,
        help="Maximum steps per episode (24h horizon at 5-minute sample time).",
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        default=Path("results") / "classic_controllers.csv",
        help="Path to save aggregated metrics.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_csv.parent, exist_ok=True)

    df = evaluate_controllers(
        controllers=args.controllers,
        patients=args.patients,
        episodes_per_patient=args.episodes_per_patient,
        seed_offset=args.seed_offset,
        max_episode_steps=args.max_episode_steps,
    )
    df.to_csv(args.output_csv, index=False)
    print(f"[DONE] Saved {len(df)} episodes to {args.output_csv}")


if __name__ == "__main__":
    main()
