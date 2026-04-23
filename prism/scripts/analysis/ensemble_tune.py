#!/usr/bin/env python3
"""Weighted ensemble across models: search weights on val, then evaluate on test.

Highlights:
  - For each model we first average logits over 5 seeds with equal weights
    (consistent with the rest of the pipeline).
  - Search weights on the validation set over a simple grid (step 0.05, summing to 1.0).
  - Pick the weights with the best macro_f1_no_null and evaluate on test.
  - Write results to results/evaluation/final_results.json.

Example usage:
  pixi run python prism/scripts/ensemble_tune.py \
    --models grouprec_convnext_gated_loss_tuned,grouprec_convnext_mlp,grouprec_convnext_gated_T2p0_E0p02 \
    --device cpu
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from itertools import product
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
    parser = argparse.ArgumentParser(description="Ensemble weight tuning on val, evaluate on test")
    parser.add_argument("--models", type=str, required=True, help="Comma-separated list of model names")
    parser.add_argument("--device", type=str, default="cuda:0", help="Inference device (cuda:0 or cpu)")
    parser.add_argument("--step", type=float, default=0.05, help="Weight grid step (default 0.05)")
    parser.add_argument(
        "--min-weight", type=float, default=0.0, help="Minimum weight per model (default 0.0)"
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=str(ROOT_DIR / "results/evaluation/final_results.json"),
        help="JSON file to append/update results into",
    )
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed-precision inference")
    parser.add_argument("--no-preload", action="store_true", help="Disable preloading data to GPU")
    parser.add_argument("--batch-models", type=int, default=2, help="Number of models to load concurrently (default 2)")
    return parser.parse_args()


def generate_simplex_weights(n: int, step: float, min_w: float) -> List[List[float]]:
    """Generate an n-dimensional weight grid (step `step`, entries >= `min_w`, summing to 1)."""
    base = [min_w] * n
    rem = 1.0 - n * min_w
    if rem < -1e-9:
        raise ValueError("min_weight is too large to sum to 1.0")
    if rem <= 1e-9:
        return [base]
    k = int(round(rem / step))
    grids: List[List[float]] = []
    if n == 2:
        for i in range(k + 1):
            w0 = base[0] + i * step
            w1 = 1.0 - w0
            grids.append([w0, w1])
        return grids
    def rec(prefix: List[float], remaining: float, idx: int):
        if idx == n - 1:
            w_last = remaining
            grids.append(prefix + [w_last])
            return
        max_i = int(round(remaining / step))
        for i in range(max_i + 1):
            w = i * step
            rec(prefix + [w], remaining - w, idx + 1)

    rec([], rem, 0)
    final = []
    for rel in grids:
        ws = [b + r for b, r in zip(base, rel)]
        if all(w >= min_w - 1e-9 for w in ws) and abs(sum(ws) - 1.0) < 1e-6:
            final.append(ws)
    return final


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
    model_names = [m.strip() for m in args.models.split(",") if m.strip()]
    assert len(model_names) >= 2

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("  [!] CUDA not available, falling back to CPU")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    cfgs = [load_config(f"prism/configs/{m}.yaml") for m in model_names]

    print("Running validation-set inference (per-model seed averaging)...")
    val_loader = validate_and_build_dataloader(cfgs, split="val")
    val_labels = val_loader.dataset.labels.numpy()
    val_logits_list = []
    seeds_info: Dict[str, Any] = {}
    for m, cfg in zip(model_names, cfgs):
        print(f"  - {m}")
        r = infer_logits_for_model(
            m,
            cfg,
            val_loader,
            device,
            batch_models=args.batch_models,
            use_amp=not args.no_amp,
            preload_data=not args.no_preload,
        )
        val_logits_list.append(average_logits(r["logits"]))
        seeds_info[m] = {"seeds": r["seeds"]}

    print("Running test-set inference (per-model seed averaging)...")
    test_loader = validate_and_build_dataloader(cfgs, split="test")
    test_labels = test_loader.dataset.labels.numpy()
    test_logits_list = []
    for m, cfg in zip(model_names, cfgs):
        r = infer_logits_for_model(
            m,
            cfg,
            test_loader,
            device,
            batch_models=args.batch_models,
            use_amp=not args.no_amp,
            preload_data=not args.no_preload,
        )
        test_logits_list.append(average_logits(r["logits"]))

    weights_grid = generate_simplex_weights(len(model_names), args.step, args.min_weight)
    print(f"Weight search on val: {len(weights_grid)} candidates (step={args.step})")

    best_w: List[float] | None = None
    best_metric = -1.0
    for ws in weights_grid:
        logits = average_logits(val_logits_list, ws)
        m = compute_metrics_from_logits(logits, val_labels)
        f1nn = m["macro_f1_no_null"]
        if f1nn > best_metric:
            best_metric = f1nn
            best_w = ws
    assert best_w is not None
    print(f"[OK] Best val weights: {best_w} (macro_f1_no_null={best_metric:.4f})")

    test_logits = average_logits(test_logits_list, best_w)
    test_metrics = compute_metrics_from_logits(test_logits, test_labels)

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        results: Dict[str, Any] = json.loads(out_path.read_text())
    except Exception:
        results = {}

    joined = ",".join(model_names)
    key_base = f"ensemble_tuned({joined})"
    key = key_base
    if len(key) > 50:
        key = f"ensemble_tuned({joined[:30]})__{hashlib.md5(joined.encode()).hexdigest()[:6]}"

    results[key] = {
        "status": "ok",
        "seeds": [],
        "n_seeds": 0,
        "component_models": {
            m: {"seeds": seeds_info[m]["seeds"], "weight": float(w)}
            for m, w in zip(model_names, best_w)
        },
        "metrics": {
            "val_macro_f1_no_null_mean": float(best_metric),
            "val_macro_f1_no_null_std": 0.0,
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

    out_path.write_text(json.dumps(results, indent=2))
    print(f"[OK] Results written to: {out_path}")


if __name__ == "__main__":
    main()

