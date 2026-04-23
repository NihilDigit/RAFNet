#!/usr/bin/env python3
"""Aggregate per-class metrics across seeds and export a readable table.

Output CSV schema:
  Method,Class,Precision,Recall,F1(no-null),Accuracy

Notes
- Precision/Recall are per-class, averaged over seeds with mean±std (percentage).
- F1(no-null) and Accuracy are method-level no-null metrics (mean±std) repeated on each row for readability.
- Adds a summary row per method: Class = "Macro(no-null)" with macro precision/recall over classes {0,1,2,4}.
"""

from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


ROOT = Path(__file__).resolve().parents[2]

# Base models to include (4 + 3 + 3 + 3 = 13)
BASE_MODELS = [
    # single-modal
    "grouprec_only",
    "resnet_only",
    "vit_only",
    "convnext_only",
    # MLP fusion
    "grouprec_resnet_mlp",
    "grouprec_vit_mlp",
    "grouprec_convnext_mlp",
    # Gated fusion (use SOTA tuned variant for convnext)
    "grouprec_resnet_gated",
    "grouprec_vit_gated",
    "grouprec_convnext_gated_loss_tuned",
    # Transformer fusion
    "grouprec_resnet_transformer",
    "grouprec_vit_transformer",
    "grouprec_convnext_transformer",
]

# Class index → name mapping (consistent with dataset and configs)
CLASS_NAMES = {
    0: "Bowing Head",
    1: "Listening",
    2: "Reading/Writing",
    3: "Null",
    4: "Using Phone",
}

NO_NULL_INDICES = [0, 1, 2, 4]

# Optional display name overrides (use tuned result but display as baseline name)
DISPLAY_NAME = {
    "grouprec_convnext_gated_loss_tuned": "grouprec_convnext_gated",
}


def discover_seeds(model_dir: Path) -> List[int]:
    seeds: List[int] = []
    for sd in sorted(model_dir.glob("seed_*")):
        if not sd.is_dir():
            continue
        try:
            seeds.append(int(sd.name.split("_")[1]))
        except Exception:
            pass
    return seeds


def ensure_details(model: str, seed: int) -> Path:
    """Ensure `details_test.json` exists; if not, run per-checkpoint eval to generate it."""
    model_dir = ROOT / "results" / "training" / model
    seed_dir = model_dir / f"seed_{seed}"
    out_path = seed_dir / "details_test.json"
    if out_path.exists():
        return out_path

    ckpt = seed_dir / "best_model.pth"
    if not ckpt.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt}")

    # Run the details script on CPU to be safe
    cmd = [
        "python",
        str((ROOT / "prism" / "scripts" / "eval_checkpoint_details.py").resolve()),
        "--model",
        model,
        "--seed",
        str(seed),
        "--device",
        "cpu",
        "--no-amp",
        "--no-preload",
    ]
    subprocess.run(cmd, cwd=str(ROOT), check=True)
    if not out_path.exists():
        raise RuntimeError(f"Failed to produce details_test.json for {model} seed {seed}")
    return out_path


def mean_std_str(values: List[float], pct: bool = True) -> str:
    arr = np.array(values, dtype=float)
    m = np.mean(arr)
    s = np.std(arr, ddof=0)
    if pct:
        return f"{m*100:.2f}% ± {s*100:.2f}%"
    else:
        return f"{m:.4f} ± {s:.4f}"


def load_per_seed_metrics(details_path: Path) -> Tuple[Dict[int, Dict[str, float]], Dict[str, float]]:
    """Return per-class metrics and overall from a details_test.json.

    per_class: {idx: {"precision": p, "recall": r}}
    overall:   {"macro_f1_no_null": f1, "accuracy_no_null": acc_no_null}
    """
    with open(details_path, "r") as f:
        data = json.load(f)
    report = data["per_class_report"]
    per_class = {}
    for k in ["0", "1", "2", "3", "4"]:
        cls = int(k)
        m = report.get(k, {"precision": 0.0, "recall": 0.0})
        per_class[cls] = {
            "precision": float(m.get("precision", 0.0)),
            "recall": float(m.get("recall", 0.0)),
        }

    overall = data["overall"]
    out_overall = {
        "macro_f1_no_null": float(overall.get("macro_f1_no_null", 0.0)) / 100.0,
        "accuracy_no_null": float(overall.get("accuracy_no_null", 0.0)) / 100.0,
    }
    return per_class, out_overall


def main() -> None:
    output_dir = ROOT / "results" / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "per_class_mean_std.csv"

    rows: List[List[str]] = []

    for model in BASE_MODELS:
        model_dir = ROOT / "results" / "training" / model
        if not model_dir.exists():
            # skip absent models gracefully
            continue
        seeds = discover_seeds(model_dir)
        if not seeds:
            continue

        # Collect per-seed metrics
        per_seed_class_prec: Dict[int, List[float]] = {i: [] for i in CLASS_NAMES}
        per_seed_class_rec: Dict[int, List[float]] = {i: [] for i in CLASS_NAMES}
        per_seed_f1_no_null: List[float] = []
        per_seed_acc_no_null: List[float] = []

        # For macro(no-null) precision/recall per seed
        per_seed_macroP_no_null: List[float] = []
        per_seed_macroR_no_null: List[float] = []

        for seed in seeds:
            details = ensure_details(model, seed)
            per_class, overall = load_per_seed_metrics(details)

            # record per-class
            for idx in CLASS_NAMES:
                per_seed_class_prec[idx].append(per_class[idx]["precision"])  # 0..1
                per_seed_class_rec[idx].append(per_class[idx]["recall"])      # 0..1

            # macro(no-null) P/R for this seed
            macroP = float(np.mean([per_class[i]["precision"] for i in NO_NULL_INDICES]))
            macroR = float(np.mean([per_class[i]["recall"] for i in NO_NULL_INDICES]))
            per_seed_macroP_no_null.append(macroP)
            per_seed_macroR_no_null.append(macroR)

            # method-level no-null F1 / Acc (already percent in JSON overall; converted to 0..1)
            per_seed_f1_no_null.append(overall["macro_f1_no_null"])  # 0..1
            per_seed_acc_no_null.append(overall["accuracy_no_null"])  # 0..1

        # Compose rows: one per class + a Macro(no-null) row
        method_f1 = mean_std_str(per_seed_f1_no_null, pct=True)
        method_acc = mean_std_str(per_seed_acc_no_null, pct=True)

        for idx in CLASS_NAMES:
            cls_name = CLASS_NAMES[idx]
            prec_str = mean_std_str(per_seed_class_prec[idx], pct=True)
            rec_str = mean_std_str(per_seed_class_rec[idx], pct=True)
            rows.append([DISPLAY_NAME.get(model, model), cls_name, prec_str, rec_str, method_f1, method_acc])

        # Macro(no-null)
        macroP_str = mean_std_str(per_seed_macroP_no_null, pct=True)
        macroR_str = mean_std_str(per_seed_macroR_no_null, pct=True)
        rows.append([DISPLAY_NAME.get(model, model), "Macro(no-null)", macroP_str, macroR_str, method_f1, method_acc])

    # Write CSV
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Method", "Class", "Precision", "Recall", "F1(no-null)", "Accuracy"])
        writer.writerows(rows)

    print(f"\n✓ Exported: {csv_path}")


if __name__ == "__main__":
    main()
