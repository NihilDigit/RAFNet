#!/usr/bin/env python3
"""Counterfactual modality ablation for a trained checkpoint.

Runs inference under:
  - baseline
  - zeroing each modality slice
  - (optional) permuting each modality slice across samples

Example:
  pixi run python prism/scripts/counterfactual_drop.py \
    --model grouprec_convnext_gated_loss_tuned --seed 43 --split test --include-permute
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as _np  # noqa: E402

sys.modules["numpy._core"] = _np.core
sys.modules["numpy._core._multiarray_umath"] = _np.core._multiarray_umath

from prism.evaluation.ensemble import (  # noqa: E402
    build_model_from_config,
    compute_metrics_from_logits,
    validate_and_build_dataloader,
)
from prism.utils import load_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Counterfactual modality drop analysis")
    p.add_argument("--model", required=True)
    p.add_argument("--seed", type=int, default=43)
    p.add_argument("--split", choices=["train", "val", "test"], default="test")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output-dir", default="results/analysis")
    p.add_argument("--include-permute", action="store_true")
    return p.parse_args()


def _resolve_checkpoint(model_name: str, seed: int) -> Path:
    model_dir = ROOT_DIR / "results" / "training" / model_name
    seed_ckpt = model_dir / f"seed_{seed}" / "best_model.pth"
    if seed_ckpt.exists():
        return seed_ckpt
    flat_ckpt = model_dir / "best_model.pth"
    if flat_ckpt.exists():
        return flat_ckpt
    raise FileNotFoundError(f"Checkpoint not found for {model_name} seed={seed}")


def _load_state_dict_compat(model: torch.nn.Module, state_dict: dict[str, Any]) -> None:
    try:
        model.load_state_dict(state_dict)
        return
    except RuntimeError:
        pass

    remapped = {}
    for k, v in state_dict.items():
        nk = k
        if k.startswith("gate_network.0."):
            nk = k.replace("gate_network.0.", "gate_fc1.")
        elif k.startswith("gate_network.3."):
            nk = k.replace("gate_network.3.", "gate_fc2.")
        remapped[nk] = v
    model.load_state_dict(remapped)


def _forward_logits(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    out = model(x)
    return out[0] if isinstance(out, tuple) else out


def _infer_logits(model: torch.nn.Module, features: torch.Tensor, batch_size: int = 512) -> np.ndarray:
    chunks = []
    with torch.no_grad():
        for i in range(0, features.shape[0], batch_size):
            x = features[i : i + batch_size]
            logits = _forward_logits(model, x)
            chunks.append(logits.cpu().numpy())
    return np.concatenate(chunks, axis=0)


def main() -> None:
    args = parse_args()
    out_dir = ROOT_DIR / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(f"prism/configs/{args.model}.yaml")
    dataloader = validate_and_build_dataloader([cfg], split=args.split)
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")

    model = build_model_from_config(cfg).to(device)
    ckpt = _resolve_checkpoint(args.model, args.seed)
    state = torch.load(ckpt, map_location=device)
    _load_state_dict_compat(model, state)
    model.eval()

    # Load all features once for deterministic counterfactual ops.
    f_list = []
    y_list = []
    for features, labels in dataloader:
        f_list.append(features)
        y_list.append(labels)
    features_all = torch.cat(f_list, dim=0).to(device)
    labels_all = torch.cat(y_list, dim=0).numpy()

    modalities = cfg["model"]["params"]["modalities"]
    input_dims = cfg["model"]["params"]["input_dims"]

    slices = []
    s = 0
    for mod, d in zip(modalities, input_dims):
        slices.append((mod, s, s + int(d)))
        s += int(d)

    results: list[dict[str, Any]] = []

    baseline_logits = _infer_logits(model, features_all)
    baseline_metrics = compute_metrics_from_logits(baseline_logits, labels_all)
    base_f1 = float(baseline_metrics["macro_f1_no_null"])
    base_acc = float(baseline_metrics["accuracy"])
    results.append(
        {
            "setting": "baseline",
            "macro_f1_no_null": base_f1,
            "accuracy": base_acc,
            "delta_f1_no_null": 0.0,
            "delta_accuracy": 0.0,
        }
    )

    for mod, st, ed in slices:
        x_zero = features_all.clone()
        x_zero[:, st:ed] = 0.0
        logits = _infer_logits(model, x_zero)
        m = compute_metrics_from_logits(logits, labels_all)
        f1 = float(m["macro_f1_no_null"])
        acc = float(m["accuracy"])
        results.append(
            {
                "setting": f"zero_{mod}",
                "macro_f1_no_null": f1,
                "accuracy": acc,
                "delta_f1_no_null": f1 - base_f1,
                "delta_accuracy": acc - base_acc,
            }
        )

        if args.include_permute:
            x_perm = features_all.clone()
            idx = torch.randperm(x_perm.shape[0], device=x_perm.device)
            x_perm[:, st:ed] = x_perm[idx, st:ed]
            logits_p = _infer_logits(model, x_perm)
            mp = compute_metrics_from_logits(logits_p, labels_all)
            f1p = float(mp["macro_f1_no_null"])
            accp = float(mp["accuracy"])
            results.append(
                {
                    "setting": f"permute_{mod}",
                    "macro_f1_no_null": f1p,
                    "accuracy": accp,
                    "delta_f1_no_null": f1p - base_f1,
                    "delta_accuracy": accp - base_acc,
                }
            )

    stem = f"{args.model}.seed_{args.seed}.{args.split}.counterfactual"
    json_path = out_dir / f"{stem}.json"
    csv_path = out_dir / f"{stem}.csv"

    with json_path.open("w") as f:
        json.dump(
            {
                "model": args.model,
                "seed": args.seed,
                "split": args.split,
                "checkpoint": str(ckpt.relative_to(ROOT_DIR)),
                "modalities": modalities,
                "results": results,
            },
            f,
            indent=2,
        )

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "setting",
                "macro_f1_no_null",
                "accuracy",
                "delta_f1_no_null",
                "delta_accuracy",
            ],
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"[OK] JSON: {json_path}")
    print(f"[OK] CSV:  {csv_path}")


if __name__ == "__main__":
    main()
