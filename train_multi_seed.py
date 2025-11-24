#!/usr/bin/env python3
"""
Utility script to launch multiple RB-PPO training runs with different seeds.

Example:
    python train_multi_seed.py --steps 5_000_000
    python train_multi_seed.py --seeds 11 22 33 --steps 3_000_000 -- --debug
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable


# DEFAULT_SEEDS: tuple[int, ...] = (108, 193, 302, 22, 508, 221, 900)
DEFAULT_SEEDS: tuple[int, ...] = (108, 193, 202, 303, 508, 606, 707)


def launch_training(
    seeds: Iterable[int], steps: int, extra_args: list[str], script_path: Path
) -> None:
    """
    Launch train_ppo_v2.py once per seed with the requested number of steps.
    """
    for seed in seeds:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_name = f"RB-PPO_seed{seed}_safety_update"
        cmd = [
            sys.executable,
            str(script_path),
            "--type",
            "fresh",
            "--steps",
            str(steps),
            "--seed",
            str(seed),
            "--run_name",
            run_name,
        ]
        if extra_args:
            cmd.extend(extra_args)

        print("\n=== Launching training ===")
        print(f"Seed      : {seed}")
        print(f"Timesteps : {steps:,}")
        print(f"Command   : {' '.join(cmd)}")
        print(f"Started   : {timestamp}")

        subprocess.run(cmd, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run multiple RB-PPO training jobs with different seeds."
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=5_000_000,
        help="Number of training timesteps per seed.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(DEFAULT_SEEDS),
        help="Seeds to launch. Defaults to seven pre-defined seeds.",
    )
    parser.add_argument(
        "extra",
        nargs=argparse.REMAINDER,
        help=(
            "Additional arguments forwarded to train_ppo_v2.py "
            "(prefix with -- if you need to pass flags)."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    extra_args = args.extra
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]

    training_script = Path(__file__).resolve().parent / "train_ppo_v2.py"
    if not training_script.exists():
        raise FileNotFoundError(f"Expected training script at {training_script}")

    launch_training(args.seeds, args.steps, extra_args, training_script)


if __name__ == "__main__":
    main()
