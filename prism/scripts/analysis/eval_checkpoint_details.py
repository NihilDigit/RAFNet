#!/usr/bin/env python3
"""Produce detailed test-set evaluation information for a specific checkpoint (model + seed).

Contents:
  - overall: loss, accuracy, macro_f1, macro_f1_no_null, accuracy_no_null
  - per_class: precision/recall/f1/support (via sklearn)
  - confusion_matrix (5x5)
  - predictions_count (counts per predicted class)
  - Metadata: model name, seed, checkpoint path, config file, feature path, etc.

Output: writes JSON to details_test.json in the seed directory and prints a summary to stdout.

Usage:
  pixi run python prism/scripts/eval_checkpoint_details.py \
    --model grouprec_convnext_gated_loss_tuned --seed 43 --device cpu
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# numpy>=2 renamed numpy.core -> numpy._core; our pkl features still reference the old path.
import numpy as _np  # type: ignore

sys.modules["numpy._core"] = _np.core
sys.modules["numpy._core._multiarray_umath"] = _np.core._multiarray_umath


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a specific checkpoint (model + seed) on test split")
    p.add_argument("--model", required=True, help="Model name (prism/configs/<model>.yaml)")
    p.add_argument("--seed", type=int, required=True, help="Seed number (e.g. 43)")
    p.add_argument("--device", type=str, default="cuda:0", help="cuda:0 or cpu")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--no-preload", action="store_true")
    return p.parse_args()


def main() -> None:
    import torch
    from sklearn.metrics import classification_report, confusion_matrix
    from prism.utils import load_config
    from prism.evaluation.ensemble import (
        validate_and_build_dataloader,
        build_model_from_config,
        compute_metrics_from_logits,
    )

    args = parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("  [!] CUDA not available, falling back to CPU")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    model_dir = ROOT_DIR / "results" / "training" / args.model
    ckpt_dir = model_dir / f"seed_{args.seed}"
    ckpt_path = ckpt_dir / "best_model.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    cfg = load_config(f"prism/configs/{args.model}.yaml")
    test_loader = validate_and_build_dataloader([cfg], split="test")
    labels = test_loader.dataset.labels.numpy()

    model = build_model_from_config(cfg).to(device)
    state = torch.load(ckpt_path, map_location=device)

    def _try_load(sd):
        try:
            model.load_state_dict(sd)
            return True
        except RuntimeError:
            return False

    def _remap_gated_keys(sd: dict) -> dict:
        # Legacy naming: gate_network.0 -> gate_fc1, gate_network.3 -> gate_fc2
        if any(k.startswith("gate_network.") for k in sd.keys()):
            new_sd = {}
            for k, v in sd.items():
                nk = k
                if k.startswith("gate_network.0."):
                    nk = k.replace("gate_network.0.", "gate_fc1.")
                elif k.startswith("gate_network.3."):
                    nk = k.replace("gate_network.3.", "gate_fc2.")
                new_sd[nk] = v
            return new_sd
        return sd

    sd = state if isinstance(state, dict) else state
    if not _try_load(sd):
        sd2 = _remap_gated_keys(sd)
        if not _try_load(sd2):
            raise RuntimeError("Failed to load state_dict (after gated key remap)")
    model.eval()

    # Inference
    logits_list = []
    with torch.no_grad():
        for features, _ in test_loader:
            features = features.to(device)
            outputs = model(features)
            logits = outputs[0] if isinstance(outputs, tuple) else outputs
            logits_list.append(logits.cpu().numpy())
    logits = np.concatenate(logits_list, axis=0)

    overall = compute_metrics_from_logits(logits, labels)

    preds = np.argmax(logits, axis=-1)
    report = classification_report(
        labels, preds, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(labels, preds)

    pred_counts = {str(i): int((preds == i).sum()) for i in range(cm.shape[0])}
    label_counts = {str(i): int((labels == i).sum()) for i in range(cm.shape[0])}

    out = {
        "model": args.model,
        "seed": args.seed,
        "checkpoint": str(ckpt_path.relative_to(ROOT_DIR)),
        "config": f"prism/configs/{args.model}.yaml",
        "features_path": cfg["datasets"]["test"]["params"]["features_path"],
        "modalities": cfg["datasets"]["test"]["params"]["modalities"],
        "overall": {
            "loss": float(overall["loss"]),
            "accuracy": float(overall["accuracy"]),
            "macro_f1": float(overall["macro_f1"]),
            "macro_f1_no_null": float(overall["macro_f1_no_null"]),
            "accuracy_no_null": float(overall["accuracy_no_null"]),
        },
        "per_class_report": report,
        "confusion_matrix": cm.tolist(),
        "predictions_count": pred_counts,
        "labels_count": label_counts,
    }
    out_path = ckpt_dir / "details_test.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print("\nDetailed evaluation summary (test set):")
    print(
        f"macro_f1_no_null={out['overall']['macro_f1_no_null']:.4f}, "
        f"macro_f1={out['overall']['macro_f1']:.4f}, "
        f"acc={out['overall']['accuracy']:.2f}, "
        f"acc_no_null={out['overall']['accuracy_no_null']:.2f}"
    )
    print(f"Confusion matrix shape: {cm.shape}. Details in {out_path.relative_to(ROOT_DIR)}")


if __name__ == "__main__":
    main()
