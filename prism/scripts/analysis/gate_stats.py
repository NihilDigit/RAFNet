#!/usr/bin/env python3
"""Analyze gate behaviors for a trained gated fusion checkpoint.

Outputs:
  - per-class gate statistics CSV
  - summary JSON (overall + per-class)
  - optional boxplot PNG (if matplotlib available)

Example:
  pixi run python prism/scripts/gate_stats.py \
    --model grouprec_convnext_gated_loss_tuned --seed 43 --split test
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
    p = argparse.ArgumentParser(description="Gate statistics for gated fusion models")
    p.add_argument("--model", required=True)
    p.add_argument("--seed", type=int, default=43)
    p.add_argument("--split", choices=["train", "val", "test"], default="test")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output-dir", default="results/analysis")
    p.add_argument("--plot", action="store_true")
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

    # Backward-compat for old gated naming.
    remapped = {}
    for k, v in state_dict.items():
        nk = k
        if k.startswith("gate_network.0."):
            nk = k.replace("gate_network.0.", "gate_fc1.")
        elif k.startswith("gate_network.3."):
            nk = k.replace("gate_network.3.", "gate_fc2.")
        remapped[nk] = v
    model.load_state_dict(remapped)


def _forward_with_gates(model: torch.nn.Module, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Forward for GatedFusionClassifier with explicit gate outputs.

    Returns:
      logits: [B, C]
      gate_probs: [B, M]
      entropy: [B]
    """
    if not hasattr(model, "modalities") or not hasattr(model, "slice_indices"):
        raise TypeError("Model is not a gated fusion classifier")

    projected = []
    for modality, (start, end) in zip(model.modalities, model.slice_indices):
        chunk = x[:, start:end]
        chunk = model.norms[modality](chunk)
        chunk = model.projections[modality](chunk)
        projected.append(chunk)

    concat = torch.cat(projected, dim=1)
    gate_hidden = model.gate_act(model.gate_fc1(concat))
    gate_hidden = model.gate_drop(gate_hidden)
    gate_logits = model.gate_fc2(gate_hidden)
    gate_probs = torch.softmax(gate_logits / max(float(model.gate_temperature), 1e-6), dim=1)
    entropy = -(gate_probs * torch.log(gate_probs + 1e-8)).sum(dim=1)

    gated = []
    for i, feat in enumerate(projected):
        gated.append(feat * gate_probs[:, i : i + 1])
    fused = torch.cat(gated, dim=1)
    logits = model.classifier(fused)
    return logits, gate_probs, entropy


def main() -> None:
    args = parse_args()
    out_dir = ROOT_DIR / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(f"prism/configs/{args.model}.yaml")
    labels_names = cfg.get("training", {}).get("label_names", [str(i) for i in range(5)])
    dataloader = validate_and_build_dataloader([cfg], split=args.split)
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")

    model = build_model_from_config(cfg).to(device)
    ckpt = _resolve_checkpoint(args.model, args.seed)
    state = torch.load(ckpt, map_location=device)
    _load_state_dict_compat(model, state)
    model.eval()

    all_logits = []
    all_labels = []
    all_gates = []
    all_entropy = []

    with torch.no_grad():
        for features, labels in dataloader:
            features = features.to(device)
            logits, gate_probs, entropy = _forward_with_gates(model, features)
            all_logits.append(logits.cpu().numpy())
            all_labels.append(labels.numpy())
            all_gates.append(gate_probs.cpu().numpy())
            all_entropy.append(entropy.cpu().numpy())

    logits_np = np.concatenate(all_logits, axis=0)
    labels_np = np.concatenate(all_labels, axis=0)
    gates_np = np.concatenate(all_gates, axis=0)
    entropy_np = np.concatenate(all_entropy, axis=0)
    preds_np = np.argmax(logits_np, axis=-1)

    metrics = compute_metrics_from_logits(logits_np, labels_np)

    modalities = list(getattr(model, "modalities", [f"m{i}" for i in range(gates_np.shape[1])]))
    per_class: dict[str, Any] = {}

    for class_id, class_name in enumerate(labels_names):
        mask = labels_np == class_id
        count = int(mask.sum())
        cls_dict: dict[str, Any] = {"count": count}
        if count > 0:
            cls_g = gates_np[mask]
            cls_e = entropy_np[mask]
            for m_idx, mod in enumerate(modalities):
                cls_dict[f"gate_{mod}_mean"] = float(cls_g[:, m_idx].mean())
                cls_dict[f"gate_{mod}_std"] = float(cls_g[:, m_idx].std())
            cls_dict["entropy_mean"] = float(cls_e.mean())
            cls_dict["entropy_std"] = float(cls_e.std())
            cls_dict["acc"] = float((preds_np[mask] == labels_np[mask]).mean() * 100.0)
        per_class[class_name] = cls_dict

    overall = {
        "count": int(labels_np.shape[0]),
        "macro_f1_no_null": float(metrics["macro_f1_no_null"]),
        "macro_f1": float(metrics["macro_f1"]),
        "accuracy": float(metrics["accuracy"]),
        "accuracy_no_null": float(metrics["accuracy_no_null"]),
        "entropy_mean": float(entropy_np.mean()),
        "entropy_std": float(entropy_np.std()),
    }
    for m_idx, mod in enumerate(modalities):
        overall[f"gate_{mod}_mean"] = float(gates_np[:, m_idx].mean())
        overall[f"gate_{mod}_std"] = float(gates_np[:, m_idx].std())

    stem = f"{args.model}.seed_{args.seed}.{args.split}.gate_stats"
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
                "overall": overall,
                "per_class": per_class,
            },
            f,
            indent=2,
        )

    fieldnames = ["class", "count", "acc", "entropy_mean", "entropy_std"]
    for mod in modalities:
        fieldnames.extend([f"gate_{mod}_mean", f"gate_{mod}_std"])

    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for class_name, row in per_class.items():
            out = {"class": class_name}
            out.update(row)
            w.writerow(out)

    if args.plot:
        try:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(8, 4))
            for m_idx, mod in enumerate(modalities):
                ax.boxplot(
                    [gates_np[labels_np == c, m_idx] for c in range(len(labels_names))],
                    positions=np.arange(len(labels_names)) + m_idx * 0.2,
                    widths=0.18,
                    patch_artist=True,
                    boxprops={"alpha": 0.5},
                )
            ax.set_xticks(np.arange(len(labels_names)) + 0.1)
            ax.set_xticklabels(labels_names, rotation=20, ha="right")
            ax.set_ylim(0.0, 1.0)
            ax.set_ylabel("Gate probability")
            ax.set_title(f"Gate distribution by class ({args.model}, seed={args.seed})")
            fig.tight_layout()
            fig.savefig(out_dir / f"{stem}.png", dpi=160)
            plt.close(fig)
        except Exception as e:
            print(f"[WARN] plot skipped: {e}")

    print(f"[OK] JSON: {json_path}")
    print(f"[OK] CSV:  {csv_path}")


if __name__ == "__main__":
    main()
