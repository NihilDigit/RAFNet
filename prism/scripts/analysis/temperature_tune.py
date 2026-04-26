#!/usr/bin/env python3
"""Small tool for single-model temperature scaling.

Workflow:
  1) For a single model (first equal-weight averaging its multi-seed logits),
     sweep temperature T over the validation set using logits / T.
  2) Pick the T that maximizes macro_f1_no_null (call it T_val).
  3) Evaluate on the test set using T_val and write the result to
     results/evaluation/final_results.json.

Example usage:
  pixi run python prism/scripts/temperature_tune.py --model grouprec_convnext_gated_loss_tuned
  pixi run python prism/scripts/temperature_tune.py --model grouprec_convnext_mlp --t-grid 0.6,0.8,1.0,1.2,1.4,1.6
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single-model temperature scaling on val + test")
    parser.add_argument("--model", type=str, required=True, help="Model name (matches prism/configs/<model>.yaml)")
    parser.add_argument(
        "--t-grid",
        type=str,
        default=None,
        help="Comma-separated list of temperatures (e.g. 0.6,0.8,1.0,1.2). Defaults to [0.6..1.6] step 0.05.",
    )
    parser.add_argument("--device", type=str, default="cuda:0", help="Inference device (cuda:0 or cpu)")
    parser.add_argument("--batch-models", type=int, default=2, help="Number of models to load concurrently (default 2)")
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed-precision inference")
    parser.add_argument("--no-preload", action="store_true", help="Disable preloading data to GPU")
    parser.add_argument(
        "--output-json",
        type=str,
        default=str(ROOT_DIR / "results/evaluation/final_results.json"),
        help="JSON file to append/update results into",
    )
    return parser.parse_args()


def build_t_grid(arg: str | None) -> List[float]:
    if arg:
        vals: List[float] = []
        for t in arg.split(","):
            t = t.strip()
            if not t:
                continue
            vals.append(float(t))
        return vals
    # Default grid: 0.60 .. 1.60 (step 0.05)
    return [round(0.6 + i * 0.05, 2) for i in range(int((1.6 - 0.6) / 0.05) + 1)]


def main() -> None:
    import torch
    from prism.utils import load_config
    from prism.evaluation.ensemble import (
        validate_and_build_dataloader,
        infer_logits_for_model,
        average_logits,
        compute_metrics_from_logits,
    )

    args = parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("  [!] CUDA not available, falling back to CPU")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    cfg = load_config(f"prism/configs/{args.model}.yaml")

    val_loader = validate_and_build_dataloader([cfg], split="val")
    val_labels = val_loader.dataset.labels.numpy()
    val_inf = infer_logits_for_model(
        args.model,
        cfg,
        val_loader,
        device,
        batch_models=args.batch_models,
        use_amp=not args.no_amp,
        preload_data=not args.no_preload,
    )
    seeds = val_inf["seeds"]
    val_logits_avg = average_logits(val_inf["logits"])

    t_grid = build_t_grid(args.t_grid)
    best_t = None
    best_metric = -1.0
    for t in t_grid:
        scaled = val_logits_avg / t
        m = compute_metrics_from_logits(scaled, val_labels)
        f1nn = m["macro_f1_no_null"]
        if f1nn > best_metric:
            best_metric = f1nn
            best_t = t
    assert best_t is not None
    print(f"[OK] Best val temperature: T={best_t}  (macro_f1_no_null={best_metric:.4f})")

    test_loader = validate_and_build_dataloader([cfg], split="test")
    test_labels = test_loader.dataset.labels.numpy()
    test_inf = infer_logits_for_model(
        args.model,
        cfg,
        test_loader,
        device,
        batch_models=args.batch_models,
        use_amp=not args.no_amp,
        preload_data=not args.no_preload,
    )
    test_logits_avg = average_logits(test_inf["logits"])
    test_scaled = test_logits_avg / float(best_t)
    test_metrics = compute_metrics_from_logits(test_scaled, test_labels)

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results: Dict[str, Any] = {}
    if out_path.exists():
        try:
            with open(out_path, "r") as f:
                results = json.load(f)
        except Exception:
            results = {}

    key = f"temp_scaled({args.model})__T{str(best_t).replace('.', 'p')}"
    results[key] = {
        "status": "ok",
        "seeds": [],
        "n_seeds": 0,
        "component_models": {
            args.model: {
                "seeds": seeds,
                "weight": 1.0,
                "temperature": float(best_t),
            }
        },
        "metrics": {
            # Validation set (for traceability)
            "val_macro_f1_no_null_mean": float(best_metric),
            "val_macro_f1_no_null_std": 0.0,
            # Test set
            "test_macro_f1_no_null_mean": float(test_metrics["macro_f1_no_null"]),
            "test_macro_f1_no_null_std": 0.0,
            "test_macro_f1_mean": float(test_metrics["macro_f1"]),
            "test_macro_f1_std": 0.0,
            "test_accuracy_mean": float(test_metrics["accuracy"]),
            "test_accuracy_std": 0.0,
            "test_accuracy_no_null_mean": float(test_metrics["accuracy_no_null"]),
            "test_accuracy_no_null_std": 0.0,
            "test_loss_mean": float(test_metrics["loss"]),
            "test_loss_std": 0.0,
        },
    }

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[OK] Results written to: {out_path}")


if __name__ == "__main__":
    main()

