#!/usr/bin/env python3
"""Run external baselines on pre-extracted PRISM features.

This script creates a unified benchmark table for non-PRISM baselines
(linear models, SVM, MLP, LMF-style fusion, optional XGBoost), with
multi-seed aggregation and the same key metrics used in the paper.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from prism.baselines import METHOD_REGISTRY, compute_metrics, load_feature_split, run_method


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run external baseline suite")
    parser.add_argument(
        "--methods",
        type=str,
        default=",".join(
            [
                "convnext_logreg",
                "grouprec_logreg",
                "convnext_linear_svm",
                "grouprec_linear_svm",
                "concat_logreg",
                "concat_linear_svm",
                "concat_mlp",
                "late_fusion_logreg",
                "lmf_logreg",
                "xgboost_concat",
            ]
        ),
        help="comma-separated method names",
    )
    parser.add_argument("--seeds", type=str, default="41,42,43,44,45")
    parser.add_argument(
        "--train-path",
        type=str,
        default="output/prism_features/latest/train_features.pkl",
    )
    parser.add_argument(
        "--test-path",
        type=str,
        default="output/prism_features/latest/test_features.pkl",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/external_baselines",
    )
    return parser.parse_args()


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _to_float(v: Any) -> float:
    try:
        return float(v)
    except Exception:
        return float("nan")


def aggregate(runs: List[Dict[str, Any]]) -> Dict[str, float]:
    keys = [
        "test_macro_f1_no_null",
        "test_macro_f1",
        "test_accuracy",
        "test_accuracy_no_null",
        "test_loss",
    ]
    out: Dict[str, float] = {}
    for k in keys:
        vals = np.array([_to_float(r["metrics"].get(k, np.nan)) for r in runs], dtype=np.float64)
        vals = vals[~np.isnan(vals)]
        out[f"{k}_mean"] = float(vals.mean()) if vals.size else float("nan")
        out[f"{k}_std"] = float(vals.std()) if vals.size else float("nan")
        out[f"{k}_min"] = float(vals.min()) if vals.size else float("nan")
        out[f"{k}_max"] = float(vals.max()) if vals.size else float("nan")
    return out


def export_csv(results: Dict[str, Any], out_csv: Path) -> None:
    import csv

    headers = [
        "method",
        "n_seeds",
        "macro_f1_no_null_mean",
        "macro_f1_no_null_std",
        "macro_f1_mean",
        "macro_f1_std",
        "accuracy_mean",
        "accuracy_std",
        "accuracy_no_null_mean",
        "accuracy_no_null_std",
        "loss_mean",
        "loss_std",
    ]

    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(headers)
        for method, item in results.items():
            if item.get("status") != "ok":
                continue
            m = item["metrics"]
            w.writerow(
                [
                    method,
                    item.get("n_seeds", 0),
                    f"{m.get('test_macro_f1_no_null_mean', np.nan):.4f}",
                    f"{m.get('test_macro_f1_no_null_std', np.nan):.4f}",
                    f"{m.get('test_macro_f1_mean', np.nan):.4f}",
                    f"{m.get('test_macro_f1_std', np.nan):.4f}",
                    f"{m.get('test_accuracy_mean', np.nan):.4f}",
                    f"{m.get('test_accuracy_std', np.nan):.4f}",
                    f"{m.get('test_accuracy_no_null_mean', np.nan):.4f}",
                    f"{m.get('test_accuracy_no_null_std', np.nan):.4f}",
                    f"{m.get('test_loss_mean', np.nan):.4f}",
                    f"{m.get('test_loss_std', np.nan):.4f}",
                ]
            )


def main() -> None:
    args = parse_args()

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    unknown = [m for m in methods if m not in METHOD_REGISTRY]
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}. Available: {sorted(METHOD_REGISTRY)}")

    out_dir = ROOT_DIR / args.output_dir
    _ensure_dir(out_dir)

    print("=" * 80)
    print("External Baseline Suite")
    print("=" * 80)
    print(f"Methods: {methods}")
    print(f"Seeds: {seeds}")

    train = load_feature_split(ROOT_DIR / args.train_path)
    test = load_feature_split(ROOT_DIR / args.test_path)

    x_train = {"grouprec": train["grouprec"], "convnext": train["convnext"]}
    y_train = train["labels"]
    x_test = {"grouprec": test["grouprec"], "convnext": test["convnext"]}
    y_test = test["labels"]

    all_results: Dict[str, Any] = {}

    for method in methods:
        print("\n" + "-" * 80)
        print(f"Method: {method} | {METHOD_REGISTRY[method].description}")

        method_runs: List[Dict[str, Any]] = []
        skipped: Dict[int, str] = {}

        for seed in seeds:
            print(f"  Seed {seed} ...", end="")
            try:
                y_pred, y_prob, skip_reason = run_method(method, x_train, y_train, x_test, seed)
                if skip_reason is not None:
                    skipped[seed] = skip_reason
                    print(f" skipped ({skip_reason})")
                    continue

                metrics = compute_metrics(y_test, y_pred, y_prob)
                run_item = {
                    "seed": seed,
                    "metrics": metrics,
                }
                method_runs.append(run_item)

                seed_dir = out_dir / method / f"seed_{seed}"
                _ensure_dir(seed_dir)
                with (seed_dir / "results.json").open("w") as f:
                    json.dump(
                        {
                            "method": method,
                            "seed": seed,
                            **metrics,
                        },
                        f,
                        indent=2,
                    )
                print(f" done (F1_no_null={metrics['test_macro_f1_no_null']:.2f})")
            except Exception as e:
                skipped[seed] = str(e)
                print(f" failed ({e})")

        if method_runs:
            agg = aggregate(method_runs)
            all_results[method] = {
                "status": "ok",
                "n_seeds": len(method_runs),
                "seeds": [r["seed"] for r in method_runs],
                "metrics": agg,
            }
        else:
            all_results[method] = {
                "status": "all_skipped",
                "n_seeds": 0,
                "seeds": [],
                "metrics": {},
            }

        if skipped:
            all_results[method]["skipped"] = skipped

    out_json = out_dir / "final_results.json"
    out_csv = out_dir / "final_results.csv"

    with out_json.open("w") as f:
        json.dump(all_results, f, indent=2)
    export_csv(all_results, out_csv)

    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    for method, item in all_results.items():
        if item.get("status") != "ok":
            print(f"- {method:24s}: {item.get('status')}")
            continue
        m = item["metrics"]
        print(
            f"- {method:24s}: F1_no_null={m['test_macro_f1_no_null_mean']:.2f} ± {m['test_macro_f1_no_null_std']:.2f}"
        )

    print(f"\nSaved:\n  - {out_json}\n  - {out_csv}")


if __name__ == "__main__":
    main()
