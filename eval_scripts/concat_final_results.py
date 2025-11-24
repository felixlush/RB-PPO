#!/usr/bin/env python3
"""
Concatenate CSV evaluation results in results/final/ and tag each row with its seed.

Usage:
    python concat_final_results.py
    python concat_final_results.py --output results/final/all_results.csv
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Iterable

RESULTS_DIR = Path("results") / "no_gate"
SEED_PATTERN = re.compile(r"seed(\d+)", re.IGNORECASE)


def infer_seed(path: Path) -> int | None:
    match = SEED_PATTERN.search(path.stem)
    return int(match.group(1)) if match else None


def iter_csv_rows(path: Path) -> Iterable[dict[str, str]]:
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            yield row


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Concatenate evaluation CSVs under results/final/ with seed column."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results") / "final" / "all_seeds.csv",
        help="Destination CSV path.",
    )
    args = parser.parse_args()

    if not RESULTS_DIR.exists():
        raise FileNotFoundError(f"Results directory not found: {RESULTS_DIR}")

    csv_files = sorted(RESULTS_DIR.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {RESULTS_DIR}")

    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    header_written = False
    with output_path.open("w", newline="") as out_fh:
        writer = None

        for csv_path in csv_files:
            print(f"Processing file: {csv_path}")
            seed = infer_seed(csv_path)
            if seed is None:
                raise ValueError(f"Could not infer seed from filename: {csv_path.name}")

            for row in iter_csv_rows(csv_path):
                row["model_seed"] = str(seed)

                if not header_written:
                    fieldnames = list(row.keys())
                    writer = csv.DictWriter(out_fh, fieldnames=fieldnames)
                    writer.writeheader()
                    header_written = True

                assert writer is not None
                writer.writerow(row)

    print(f"Concatenated {len(csv_files)} files into {output_path}")


if __name__ == "__main__":
    main()
