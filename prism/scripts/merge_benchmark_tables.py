#!/usr/bin/env python3
"""Merge internal PRISM results and external baseline results into one table."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read_csv(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--internal", type=str, default="results/evaluation/final_results.csv")
    parser.add_argument("--external", type=str, default="results/external_baselines/final_results.csv")
    parser.add_argument("--out", type=str, default="results/benchmark/combined_results.csv")
    args = parser.parse_args()

    internal_rows = read_csv(Path(args.internal))
    external_rows = read_csv(Path(args.external))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    headers = [
        "source",
        "name",
        "n_seeds",
        "macro_f1_no_null_mean",
        "macro_f1_no_null_std",
        "accuracy_mean",
        "accuracy_std",
    ]

    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(headers)

        for r in internal_rows:
            w.writerow(
                [
                    "internal",
                    r.get("model", ""),
                    r.get("n_seeds", ""),
                    r.get("macro_f1_no_null_mean", ""),
                    r.get("macro_f1_no_null_std", ""),
                    r.get("accuracy_mean", ""),
                    r.get("accuracy_std", ""),
                ]
            )

        for r in external_rows:
            w.writerow(
                [
                    "external",
                    r.get("method", ""),
                    r.get("n_seeds", ""),
                    r.get("macro_f1_no_null_mean", ""),
                    r.get("macro_f1_no_null_std", ""),
                    r.get("accuracy_mean", ""),
                    r.get("accuracy_std", ""),
                ]
            )

    print(f"Saved combined table: {out_path}")


if __name__ == "__main__":
    main()
