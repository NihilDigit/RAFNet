#!/usr/bin/env python3
"""Aggregate multi-seed training results and optionally run ensemble inference.

Examples:
  python prism/scripts/evaluate.py
  python prism/scripts/evaluate.py --models grouprec_only,resnet_only
  python prism/scripts/evaluate.py --ensemble grouprec_convnext_gated_loss_tuned,grouprec_convnext_transformer \
      --ensemble-weights 0.6,0.4 --device cuda:0 --split test
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# numpy>=2 renamed numpy.core -> numpy._core; our pkl features still reference the old path.
import numpy as _np  # type: ignore

sys.modules["numpy._core"] = _np.core
sys.modules["numpy._core._multiarray_umath"] = _np.core._multiarray_umath


OFFICIAL_MODELS = [
    "grouprec_only",
    "grouprec3d_only",
    "grouprec3d_convnext_gated_loss_tuned",
    "grouprec3d_convnext_gated_c_spatial_graph_loss_tuned",
    "resnet_only",
    "vit_only",
    "convnext_only",
    "convnext_only_loss_tuned",
    "grouprec_resnet_mlp",
    "grouprec_vit_mlp",
    "grouprec_convnext_mlp",
    "grouprec_resnet_gated",
    "grouprec_vit_gated",
    "grouprec_convnext_gated",
    "grouprec_convnext_gated_loss_tuned",
    "grouprec_convnext_gated_T0p5_E0p005",
    "grouprec_convnext_gated_T0p5_E0p01",
    "grouprec_convnext_gated_T0p5_E0p02",
    "grouprec_convnext_gated_T2p0_E0p005",
    "grouprec_convnext_gated_T2p0_E0p01",
    "grouprec_convnext_gated_T2p0_E0p02",
    "grouprec_convnext_gated_ae_loss_tuned",
    "grouprec_convnext_gated_b_disentangle_loss_tuned",
    "grouprec_convnext_gated_c_context_graph_loss_tuned",
    "grouprec_convnext_gated_c_spatial_graph_loss_tuned",
    "grouprec_convnext_gated_bc_spatial_loss_tuned",
    "grouprec_convnext_gated_ccsg_loss_tuned",
    "grouprec_convnext_gated_d_supcon_loss_tuned",
    "grouprec_resnet_transformer",
    "grouprec_vit_transformer",
    "grouprec_convnext_transformer",
    "grouprec_convnext_cross_attention_loss_tuned",
    "grouprec_resnet_loss_tuned",
    "grouprec_vit_loss_tuned",
    "grouprec_convnext_loss_tuned",
    "grouprec_convnext_gated_loss_tuned_effnum_gamma1p5",
    "grouprec_convnext_gated_loss_tuned_effnum_gamma2p0",
    "grouprec_convnext_gated_loss_tuned_effnum_gamma2p5",
    "grouprec_convnext_hybrid",
    "grouprec_convnext_hybrid_loss_tuned",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PRISM Unified Evaluation Script - Multi-seed Aggregation + Ensemble",
    )
    parser.add_argument("--models", type=str, default=None, help="Comma-separated model list (default: all)")
    parser.add_argument("--output-dir", type=str, default="results/evaluation", help="Output directory")
    parser.add_argument("--plot", action="store_true", help="Generate confusion matrix PNGs (optional)")
    parser.add_argument("--ensemble", type=str, default=None, help="Comma-separated model names for ensembling")
    parser.add_argument("--ensemble-weights", type=str, default=None, help="Comma-separated weights (optional)")
    parser.add_argument("--ensemble-name", type=str, default=None, help="Custom ensemble name")
    parser.add_argument("--device", type=str, default="cuda:0", help="Inference device (cuda:0 or cpu)")
    parser.add_argument(
        "--split", type=str, default="test", choices=["train", "val", "test"], help="Split to evaluate"
    )
    parser.add_argument("--batch-models", type=int, default=2, help="Number of models to load concurrently (default 2)")
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed-precision inference")
    parser.add_argument("--no-preload", action="store_true", help="Disable preloading data to GPU")
    return parser.parse_args()


def discover_seeds(model_dir: Path) -> Tuple[List[int], List[str]]:
    seed_dirs = []
    seed_labels = []
    for seed_dir in sorted(model_dir.glob("seed_*")):
        if seed_dir.is_dir():
            try:
                seed_num = int(seed_dir.name.split("_")[1])
                seed_dirs.append(seed_num)
                seed_labels.append(seed_dir.name)
            except (IndexError, ValueError):
                continue
    if not seed_dirs and (model_dir / "results.json").exists():
        seed_dirs.append(-1)
        seed_labels.append("seed_default")
    return seed_dirs, seed_labels


def load_seed_results(model_dir: Path, seed_label: str) -> Dict[str, Any] | None:
    results_path = (
        model_dir / "results.json" if seed_label == "seed_default" else model_dir / seed_label / "results.json"
    )
    if not results_path.exists():
        return None
    try:
        with open(results_path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"  [x] Failed to load {results_path}: {e}")
        return None


def aggregate_metrics(model_name: str, model_dir: Path) -> Dict[str, Any]:
    seeds, seed_labels = discover_seeds(model_dir)
    if not seeds:
        return {"status": "not_found", "seeds": [], "metrics": {}}

    metrics_by_seed = []
    for seed_label in seed_labels:
        result = load_seed_results(model_dir, seed_label)
        if result:
            metrics_by_seed.append(result)
    if not metrics_by_seed:
        return {"status": "no_results", "seeds": seed_labels, "metrics": {}}

    metrics_to_aggregate = [
        "test_macro_f1_no_null",
        "test_macro_f1",
        "test_accuracy",
        "test_accuracy_no_null",
        "test_loss",
    ]
    aggregated: Dict[str, Any] = {
        "status": "ok",
        "seeds": seed_labels,
        "n_seeds": len(metrics_by_seed),
        "metrics": {},
    }
    for metric_name in metrics_to_aggregate:
        values = [m.get(metric_name, np.nan) for m in metrics_by_seed]
        if "macro_f1" in metric_name:
            values = [v * 100.0 if isinstance(v, (int, float)) and not np.isnan(v) and v < 1.0 else v for v in values]
        values = [v for v in values if not np.isnan(v)]
        if values:
            aggregated["metrics"][f"{metric_name}_mean"] = float(np.mean(values))
            aggregated["metrics"][f"{metric_name}_std"] = float(np.std(values))
            aggregated["metrics"][f"{metric_name}_min"] = float(np.min(values))
            aggregated["metrics"][f"{metric_name}_max"] = float(np.max(values))
        else:
            aggregated["metrics"][f"{metric_name}_mean"] = None
            aggregated["metrics"][f"{metric_name}_std"] = None
    return aggregated


def export_to_csv(results: Dict[str, Dict[str, Any]], output_path: Path) -> None:
    import csv

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "model",
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
        )
        for model_name, data in results.items():
            if data.get("status") != "ok":
                continue
            metrics = data.get("metrics", {})
            writer.writerow(
                [
                    model_name,
                    data.get("n_seeds", 0),
                    f"{metrics.get('test_macro_f1_no_null_mean', 0):.4f}",
                    f"{metrics.get('test_macro_f1_no_null_std', 0):.4f}",
                    f"{metrics.get('test_macro_f1_mean', 0):.4f}",
                    f"{metrics.get('test_macro_f1_std', 0):.4f}",
                    f"{metrics.get('test_accuracy_mean', 0):.4f}",
                    f"{metrics.get('test_accuracy_std', 0):.4f}",
                    f"{metrics.get('test_accuracy_no_null_mean', 0):.4f}",
                    f"{metrics.get('test_accuracy_no_null_std', 0):.4f}",
                    f"{metrics.get('test_loss_mean', 0):.4f}",
                    f"{metrics.get('test_loss_std', 0):.4f}",
                ]
            )


def plot_confusion_matrix(model_name: str, model_dir: Path, output_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt  # noqa: F401
        import seaborn as sns  # noqa: F401
        from sklearn.metrics import confusion_matrix  # noqa: F401
    except ImportError:
        print("  [x] Skipping confusion matrix generation: matplotlib/seaborn not installed")
        return
    print("  [!] Confusion matrix generation requires extending train.py to save predictions (not yet implemented)")


def run_ensemble_evaluation(args: argparse.Namespace, results: Dict[str, Any]) -> None:
    import torch
    from prism.evaluation.ensemble import (
        validate_and_build_dataloader,
        infer_logits_for_model,
        average_logits,
        compute_metrics_from_logits,
        generate_ensemble_name,
    )
    from prism.utils import load_config

    print("\n" + "=" * 80)
    print("Ensemble Evaluation")
    print("=" * 80)

    ensemble_models = [m.strip() for m in args.ensemble.split(",")]
    print(f"Ensemble models: {ensemble_models}")
    for model_name in ensemble_models:
        model_dir = ROOT_DIR / "results" / "training" / model_name
        if not model_dir.exists():
            print(f"✗ Error: Model '{model_name}' not found")
            return

    weights = None
    if args.ensemble_weights:
        try:
            weights = [float(w.strip()) for w in args.ensemble_weights.split(",")]
            if len(weights) != len(ensemble_models):
                print("✗ Error: weight/model count mismatch")
                return
        except ValueError as e:
            print(f"✗ Error parsing weights: {e}")
            return

    ensemble_name = generate_ensemble_name(ensemble_models, args.ensemble_name)
    print(f"Ensemble name: {ensemble_name}")

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("  [!] CUDA not available, falling back to CPU")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    cfgs = [load_config(f"prism/configs/{m}.yaml") for m in ensemble_models]
    dataloader = validate_and_build_dataloader(cfgs, split=args.split)
    labels = dataloader.dataset.labels.numpy()

    # For each model, first mean-over-seeds; then mean across models (weighted if given).
    all_model_logits = []
    component_models: Dict[str, Any] = {}
    for idx, model_name in enumerate(ensemble_models):
        print(f"\nInferring {model_name}...")
        cfg = cfgs[idx]
        result = infer_logits_for_model(
            model_name, 
            cfg, 
            dataloader, 
            device,
            batch_models=args.batch_models,
            use_amp=not args.no_amp,
            preload_data=not args.no_preload,
        )
        model_avg_logits = average_logits(result["logits"])
        all_model_logits.append(model_avg_logits)
        component_models[model_name] = {
            "seeds": result["seeds"],
            "weight": (weights[idx] if weights else 1.0 / len(ensemble_models)),
        }

    print(f"\nAveraging across {len(all_model_logits)} models...")
    ensemble_logits = average_logits(all_model_logits, weights)
    ensemble_metrics = compute_metrics_from_logits(ensemble_logits, labels)

    # Fill mean/std keys (std=0) so CSV export keeps working.
    results[ensemble_name] = {
        "status": "ok",
        "seeds": [],
        "n_seeds": 0,
        "component_models": component_models,
        "metrics": {
            f"{args.split}_macro_f1_no_null_mean": ensemble_metrics["macro_f1_no_null"],
            f"{args.split}_macro_f1_no_null_std": 0.0,
            f"{args.split}_macro_f1_mean": ensemble_metrics["macro_f1"],
            f"{args.split}_macro_f1_std": 0.0,
            f"{args.split}_accuracy_mean": ensemble_metrics["accuracy"],
            f"{args.split}_accuracy_std": 0.0,
            f"{args.split}_accuracy_no_null_mean": ensemble_metrics["accuracy_no_null"],
            f"{args.split}_accuracy_no_null_std": 0.0,
            f"{args.split}_loss_mean": ensemble_metrics["loss"],
            f"{args.split}_loss_std": 0.0,
        },
    }
    print(f"  [OK] Ensemble computed on split: {args.split}")


def main(args: argparse.Namespace) -> None:
    if args.models:
        models = [m.strip() for m in args.models.split(",")]
        invalid_models = set(models) - set(OFFICIAL_MODELS)
        if invalid_models:
            print(f"Error: invalid model names: {invalid_models}")
            sys.exit(1)
    else:
        models = OFFICIAL_MODELS

    print("=" * 80)
    print("PRISM multi-seed aggregation evaluation")
    print("=" * 80)
    print(f"Number of models: {len(models)}")
    print(f"Output dir: {args.output_dir}")
    print("=" * 80)

    results: Dict[str, Any] = {}
    for model_name in models:
        print(f"\n[{model_name}]")
        model_dir = ROOT_DIR / "results" / "training" / model_name
        if not model_dir.exists():
            print(f"  [x] Directory not found: {model_dir}")
            results[model_name] = {"status": "not_found", "seeds": [], "metrics": {}}
            continue
        aggregated = aggregate_metrics(model_name, model_dir)
        results[model_name] = aggregated
        if aggregated["status"] == "ok":
            metrics = aggregated.get("metrics", {})
            print(
                f"  [OK] Seeds: {aggregated.get('seeds')}\n"
                f"  Macro F1(no-null): {metrics.get('test_macro_f1_no_null_mean', 0):.4f} +/- {metrics.get('test_macro_f1_no_null_std', 0):.4f}\n"
                f"  Macro F1: {metrics.get('test_macro_f1_mean', 0):.4f} +/- {metrics.get('test_macro_f1_std', 0):.4f}\n"
                f"  Accuracy: {metrics.get('test_accuracy_mean', 0):.2f} +/- {metrics.get('test_accuracy_std', 0):.2f}"
            )
        else:
            print(f"  [x] Status: {aggregated.get('status')}")

    if args.ensemble:
        run_ensemble_evaluation(args, results)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "final_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[OK] JSON results saved: {json_path}")

    csv_path = output_dir / "final_results.csv"
    export_to_csv(results, csv_path)
    print(f"[OK] CSV results saved: {csv_path}")

    if args.plot:
        print("\nGenerating confusion matrices...")
        for model_name in models:
            if results.get(model_name, {}).get("status") == "ok":
                png_path = output_dir / f"{model_name}_confusion_matrix.png"
                plot_confusion_matrix(model_name, ROOT_DIR / "results" / "training" / model_name, png_path)

    print("\n" + "=" * 80)
    print("Evaluation summary table")
    print("=" * 80)
    print(f"\n{'Model':<35} {'N':<5} {'F1(no-null)':<18} {'F1':<18} {'Acc':<18}")
    print("-" * 100)
    for model_name in models:
        data = results.get(model_name, {})
        if data.get("status") == "ok":
            metrics = data.get("metrics", {})
            f1_mean = metrics.get("test_macro_f1_no_null_mean", 0)
            f1_std = metrics.get("test_macro_f1_no_null_std", 0)
            f1_all_mean = metrics.get("test_macro_f1_mean", 0)
            f1_all_std = metrics.get("test_macro_f1_std", 0)
            acc_mean = metrics.get("test_accuracy_mean", 0)
            acc_std = metrics.get("test_accuracy_std", 0)
            print(
                f"{model_name:<35} "
                f"{data.get('n_seeds', 0):<5} "
                f"{f1_mean:.4f}+/-{f1_std:.4f}      "
                f"{f1_all_mean:.4f}+/-{f1_all_std:.4f}      "
                f"{acc_mean:.2f}+/-{acc_std:.2f}"
            )
        else:
            print(f"{model_name:<35} {'N/A':<5} {'N/A':<18} {'N/A':<18} {'N/A':<18}")
    print("-" * 100)
    print(f"\n[OK] Evaluation finished. Results saved to {args.output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    args = parse_args()
    main(args)
