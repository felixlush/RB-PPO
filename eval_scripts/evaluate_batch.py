#!/usr/bin/env python3
"""
Batch-evaluate multiple PPO checkpoints and store their metrics as CSV files.

Example:
    python evaluate_batch.py \
        runs/20251027-153502_RB-PPO_seed101 \
        runs_no_bolus/20251028-043728_RB-PPO-noBolus_seed101 \
        --episodes 100 \
        --deterministic \
        --output_dir results/batch
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd
from stable_baselines3 import PPO
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from config.patient_splits import patients_test
from evaluate_model import (
    build_eval_env,
    compute_episode_metrics,
    run_episode,
)


def infer_env_type(run_path: Path, default: str = "deterministic_bolus") -> str:
    """
    Infer environment type from the run path string.
    """
    name = run_path.as_posix().lower()
    if "nobolus" in name or "no_bolus" in name:
        return "no_bolus"
    if "nogate" in name or "no_gate" in name:
        return "no_gate"
    return default


def find_checkpoint(run_dir: Path) -> tuple[Path, Path | None]:
    """
    Locate best_model.zip and vecnorm.pkl under a run directory.
    """
    best_model = run_dir / "best" / "best_model.zip"
    if best_model.exists():
        vecnorm = run_dir / "best" / "vecnorm.pkl"
        return best_model, vecnorm if vecnorm.exists() else None

    final_dir = run_dir / "final"
    if final_dir.exists():
        zip_candidates = sorted(final_dir.glob("final_model_*_steps.zip"))
        if zip_candidates:
            model_path = zip_candidates[-1]
            vecnorm = final_dir / "final_model_vecnorm.pkl"
            return model_path, vecnorm if vecnorm.exists() else None

    raise FileNotFoundError(
        f"Could not locate a checkpoint in {run_dir}. "
        "Expected best/best_model.zip or final/final_model_*.zip."
    )


def evaluate_single_run(
    model_path: Path,
    vecnorm_path: Path | None,
    env_type: str,
    patients: list[str],
    episodes_per_patient: int,
    seed_offset: int,
    deterministic: bool,
    device: str,
    debug_env: bool,
) -> pd.DataFrame:
    """
    Evaluate a single checkpoint across multiple patients.
    """
    model = PPO.load(str(model_path), device=device)
    model.training = False

    results = []

    for patient_idx, patient in enumerate(patients):
        for ep_idx in range(episodes_per_patient):
            print(
                f"[{model_path}] Evaluating patient={patient} episode={ep_idx + 1}/{episodes_per_patient}"
            )
            seed = seed_offset + ep_idx + patient_idx * 1000
            env = build_eval_env(
                patient,
                seed,
                str(vecnorm_path) if vecnorm_path else None,
                debug_env,
                env_type,
            )
            try:
                metrics = run_episode(model, env, deterministic=deterministic)
            finally:
                env.close()

            metrics.update(
                {
                    "patient": patient,
                    "episode_index": ep_idx,
                    "seed": seed,
                    "deterministic": deterministic,
                }
            )
            results.append(metrics)

    return pd.DataFrame(results)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch evaluate PPO checkpoints.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "run_paths",
        nargs="+",
        help="Paths to run directories or explicit model.zip files.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=30,
        help="Episodes per patient to evaluate.",
    )
    parser.add_argument(
        "--patients",
        nargs="+",
        default=patients_test,
        help="Patients to evaluate.",
    )
    parser.add_argument(
        "--seed_offset",
        type=int,
        default=10_000,
        help="Base seed offset for evaluation episodes.",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic actions during evaluation.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device for loading the PPO model.",
    )
    parser.add_argument(
        "--debug_env",
        action="store_true",
        help="Enable debug logging in the environment wrapper.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("results") / "batch",
        help="Directory to store per-run CSV files.",
    )
    parser.add_argument(
        "--env_type",
        choices=("deterministic_bolus", "no_bolus", "no_gate"),
        help="Override environment type for all runs.",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip runs whose output CSV already exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    for run_input in args.run_paths:
        run_path = Path(run_input).resolve()
        if not run_path.exists():
            raise FileNotFoundError(f"Run path not found: {run_path}")

        if run_path.is_file() and run_path.suffix == ".zip":
            model_path = run_path
            vecnorm_candidate = run_path.with_name("vecnorm.pkl")
            vecnorm_path = vecnorm_candidate if vecnorm_candidate.exists() else None
            run_name = run_path.stem
            env_type = args.env_type or infer_env_type(run_path.parent)
        else:
            model_path, vecnorm_path = find_checkpoint(run_path)
            run_name = run_path.name
            env_type = args.env_type or infer_env_type(run_path)

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_csv = output_dir / f"{timestamp}_{run_name}.csv"

        if args.skip_existing and output_csv.exists():
            print(f"[SKIP] {run_name} -> {output_csv}")
            continue

        print(
            f"[EVAL] run={run_name} env_type={env_type} "
            f"model={model_path} vecnorm={vecnorm_path}"
        )

        df = evaluate_single_run(
            model_path=model_path,
            vecnorm_path=vecnorm_path,
            env_type=env_type,
            patients=list(args.patients),
            episodes_per_patient=args.episodes,
            seed_offset=args.seed_offset,
            deterministic=args.deterministic,
            device=args.device,
            debug_env=args.debug_env,
        )
        df.to_csv(output_csv, index=False)
        print(f"[DONE] Saved {len(df)} episodes to {output_csv}")


if __name__ == "__main__":
    main()
