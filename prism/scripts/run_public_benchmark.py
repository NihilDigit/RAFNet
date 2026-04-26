#!/usr/bin/env python3
"""Run public-benchmark protocol on pre-extracted features.

This orchestrator is designed for datasets like DAiSEE after feature extraction.
It runs:
1) selected internal PRISM models via config overrides
2) external baseline suite on the same train/test split
3) merged summary table for paper reporting
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import yaml

ROOT_DIR = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run public benchmark protocol")
    p.add_argument("--dataset", type=str, default="daisee")
    p.add_argument("--features-root", type=str, required=True, help="e.g. output/prism_features_daisee/latest")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--seeds", type=str, default="41,42,43,44,45")
    p.add_argument("--conda-cuda", type=str, default="11.1")
    p.add_argument(
        "--internal-models",
        type=str,
        default="convnext_only,grouprec_only,grouprec_convnext_mlp,grouprec_convnext_gated,grouprec_convnext_gated_loss_tuned",
    )
    p.add_argument(
        "--external-methods",
        type=str,
        default="convnext_logreg,grouprec_logreg,convnext_linear_svm,concat_logreg,concat_mlp,late_fusion_logreg,lmf_logreg",
    )
    p.add_argument("--include-xgboost", action="store_true")
    p.add_argument("--skip-internal", action="store_true")
    p.add_argument("--skip-external", action="store_true")
    return p.parse_args()


def run_cmd(cmd: List[str], env: Dict[str, str] | None = None) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT_DIR, check=True, env=env)


def make_override_config(base_cfg: Path, out_cfg: Path, features_root: Path, output_dir: str) -> None:
    with base_cfg.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg["training"]["output_dir"] = output_dir

    for split in ["train", "val", "test"]:
        cfg["datasets"][split]["params"]["features_path"] = str(features_root / f"{split}_features.pkl")

    out_cfg.parent.mkdir(parents=True, exist_ok=True)
    with out_cfg.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)


def aggregate_internal(dataset: str, models: List[str]) -> Path:
    out_dir = ROOT_DIR / "results" / "public_benchmarks" / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "internal_results.csv"

    headers = [
        "model",
        "n_seeds",
        "macro_f1_no_null_mean",
        "macro_f1_no_null_std",
        "accuracy_mean",
        "accuracy_std",
    ]

    rows: List[List[str]] = []
    for model in models:
        model_dir = ROOT_DIR / "results" / "public_benchmarks" / dataset / "training" / model
        seed_dirs = sorted(model_dir.glob("seed_*"))
        vals_f1: List[float] = []
        vals_acc: List[float] = []

        for sd in seed_dirs:
            rs = sd / "results.json"
            if not rs.exists():
                continue
            with rs.open("r", encoding="utf-8") as f:
                item = json.load(f)
            vals_f1.append(float(item.get("test_macro_f1_no_null", float("nan"))))
            vals_acc.append(float(item.get("test_accuracy", float("nan"))))

        if not vals_f1:
            continue

        import numpy as np

        rows.append(
            [
                model,
                str(len(vals_f1)),
                f"{np.nanmean(vals_f1):.4f}",
                f"{np.nanstd(vals_f1):.4f}",
                f"{np.nanmean(vals_acc):.4f}",
                f"{np.nanstd(vals_acc):.4f}",
            ]
        )

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(headers)
        w.writerows(rows)

    return out_csv


def merge_tables(dataset: str, internal_csv: Path | None, external_csv: Path | None) -> Path:
    out_path = ROOT_DIR / "results" / "public_benchmarks" / dataset / "combined_results.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows: List[List[str]] = [["source", "name", "n_seeds", "macro_f1_no_null_mean", "macro_f1_no_null_std", "accuracy_mean", "accuracy_std"]]

    if internal_csv and internal_csv.exists():
        with internal_csv.open("r", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                rows.append(["internal", r["model"], r["n_seeds"], r["macro_f1_no_null_mean"], r["macro_f1_no_null_std"], r["accuracy_mean"], r["accuracy_std"]])

    if external_csv and external_csv.exists():
        with external_csv.open("r", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                rows.append(["external", r["method"], r["n_seeds"], r["macro_f1_no_null_mean"], r["macro_f1_no_null_std"], r["accuracy_mean"], r["accuracy_std"]])

    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerows(rows)

    return out_path


def main() -> None:
    args = parse_args()

    dataset = args.dataset
    features_root = ROOT_DIR / args.features_root
    for split in ["train", "test"]:
        fp = features_root / f"{split}_features.pkl"
        if not fp.exists():
            raise FileNotFoundError(f"Missing feature file: {fp}")
    seeds = args.seeds
    model_list = [m.strip() for m in args.internal_models.split(",") if m.strip()]
    ext_methods = [m.strip() for m in args.external_methods.split(",") if m.strip()]
    if args.include_xgboost and "xgboost_concat" not in ext_methods:
        ext_methods.append("xgboost_concat")

    env = dict(os.environ)
    env.update({"CONDA_OVERRIDE_CUDA": args.conda_cuda})

    internal_csv: Path | None = None
    external_csv: Path | None = None

    if not args.skip_internal:
        print("=" * 80)
        print("Internal models")
        print("=" * 80)
        with tempfile.TemporaryDirectory(prefix=f"prism_{dataset}_") as td:
            tdp = Path(td)
            for model in model_list:
                base_cfg_name = "convnext_only_loss_tuned.yaml" if model == "convnext_only_loss_tuned" else f"{model}.yaml"
                base_cfg = ROOT_DIR / "prism" / "configs" / base_cfg_name
                if not base_cfg.exists():
                    raise FileNotFoundError(f"Config not found: {base_cfg}")

                override_cfg = tdp / f"{model}.yaml"
                out_dir = f"results/public_benchmarks/{dataset}/training/{model}"
                make_override_config(base_cfg, override_cfg, features_root, out_dir)

                for seed in [s.strip() for s in seeds.split(",") if s.strip()]:
                    run_cmd(
                        [
                            "pixi",
                            "run",
                            "python",
                            "prism/scripts/train.py",
                            "--model",
                            "convnext_only" if model == "convnext_only_loss_tuned" else model,
                            "--config-override",
                            str(override_cfg),
                            "--seed",
                            seed,
                            "--gpu",
                            str(args.gpu),
                        ],
                        env=env,
                    )

        internal_csv = aggregate_internal(dataset, model_list)
        print(f"Saved internal table: {internal_csv}")

    if not args.skip_external:
        print("=" * 80)
        print("External baselines")
        print("=" * 80)
        ext_out = f"results/public_benchmarks/{dataset}/external"
        run_cmd(
            [
                "pixi",
                "run",
                "python",
                "prism/scripts/run_external_baselines.py",
                "--methods",
                ",".join(ext_methods),
                "--seeds",
                seeds,
                "--train-path",
                str(features_root / "train_features.pkl"),
                "--test-path",
                str(features_root / "test_features.pkl"),
                "--output-dir",
                ext_out,
            ],
            env=env,
        )
        external_csv = ROOT_DIR / ext_out / "final_results.csv"
        print(f"Saved external table: {external_csv}")

    combined = merge_tables(dataset, internal_csv, external_csv)
    print("=" * 80)
    print("Done")
    print(f"Combined table: {combined}")
    print("=" * 80)


if __name__ == "__main__":
    main()
