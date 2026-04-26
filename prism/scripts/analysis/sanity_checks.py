#!/usr/bin/env python3
"""Lightweight sanity checks for feature-label signal.

Checks:
  1) linear probe on true labels
  2) linear probe on shuffled labels
  3) linear probe on shuffled features

This script is fast and useful to argue that performance is not from leakage/
accidental artifacts.

Example:
  pixi run python prism/scripts/sanity_checks.py --model grouprec_convnext_gated_loss_tuned
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from prism.utils import load_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sanity checks for multimodal features")
    p.add_argument("--model", required=True, help="config name under prism/configs/")
    p.add_argument("--output-dir", default="results/analysis")
    p.add_argument("--max-iter", type=int, default=300)
    p.add_argument("--c", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _load_xy(path: Path, modalities: list[str]) -> tuple[np.ndarray, np.ndarray]:
    with path.open("rb") as f:
        data = pickle.load(f)
    xs = [np.asarray(data[m], dtype=np.float32) for m in modalities]
    x = np.concatenate(xs, axis=1)
    y = np.asarray(data["labels"], dtype=np.int64)
    return x, y


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mask = y_true != 3
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0) * 100.0
    acc = accuracy_score(y_true, y_pred) * 100.0
    if mask.sum() > 0:
        f1_nn = f1_score(y_true[mask], y_pred[mask], average="macro", zero_division=0) * 100.0
        acc_nn = accuracy_score(y_true[mask], y_pred[mask]) * 100.0
    else:
        f1_nn = 0.0
        acc_nn = 0.0
    return {
        "macro_f1": float(f1),
        "macro_f1_no_null": float(f1_nn),
        "accuracy": float(acc),
        "accuracy_no_null": float(acc_nn),
    }


def _fit_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    max_iter: int,
    c: float,
    seed: int,
) -> np.ndarray:
    clf = LogisticRegression(
        max_iter=max_iter,
        C=c,
        random_state=seed,
        n_jobs=-1,
        multi_class="auto",
    )
    clf.fit(x_train, y_train)
    return clf.predict(x_test)


def main() -> None:
    args = parse_args()
    out_dir = ROOT_DIR / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(f"prism/configs/{args.model}.yaml")
    modalities = list(cfg["datasets"]["train"]["params"]["modalities"])

    train_path = ROOT_DIR / cfg["datasets"]["train"]["params"]["features_path"]
    test_path = ROOT_DIR / cfg["datasets"]["test"]["params"]["features_path"]

    x_train, y_train = _load_xy(train_path, modalities)
    x_test, y_test = _load_xy(test_path, modalities)

    rng = np.random.default_rng(args.seed)

    # 1) True labels baseline
    pred_true = _fit_predict(x_train, y_train, x_test, args.max_iter, args.c, args.seed)
    m_true = _metrics(y_test, pred_true)

    # 2) Shuffle train labels
    y_train_shuffle = y_train.copy()
    rng.shuffle(y_train_shuffle)
    pred_lbl = _fit_predict(x_train, y_train_shuffle, x_test, args.max_iter, args.c, args.seed)
    m_lbl = _metrics(y_test, pred_lbl)

    # 3) Shuffle train features row order (break X-Y alignment)
    perm = rng.permutation(x_train.shape[0])
    x_train_shuffle = x_train[perm]
    pred_feat = _fit_predict(x_train_shuffle, y_train, x_test, args.max_iter, args.c, args.seed)
    m_feat = _metrics(y_test, pred_feat)

    rows = [
        {"check": "true_labels", **m_true},
        {"check": "shuffle_train_labels", **m_lbl},
        {"check": "shuffle_train_features", **m_feat},
    ]

    stem = f"{args.model}.sanity"
    json_path = out_dir / f"{stem}.json"
    csv_path = out_dir / f"{stem}.csv"

    with json_path.open("w") as f:
        json.dump(
            {
                "model": args.model,
                "modalities": modalities,
                "train_path": str(train_path.relative_to(ROOT_DIR)),
                "test_path": str(test_path.relative_to(ROOT_DIR)),
                "rows": rows,
            },
            f,
            indent=2,
        )

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["check", "macro_f1_no_null", "macro_f1", "accuracy_no_null", "accuracy"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"[OK] JSON: {json_path}")
    print(f"[OK] CSV:  {csv_path}")


if __name__ == "__main__":
    main()
