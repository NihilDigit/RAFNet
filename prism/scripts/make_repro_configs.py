#!/usr/bin/env python3
"""Generate reproducible config overrides with pinned feature directories.

This avoids ambiguity from mutable `output/prism_features/latest`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]


def _load_yaml(path: Path):
    with path.open("r") as f:
        return yaml.safe_load(f)


def _dump_yaml(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


def _set_paths(cfg: dict, train: str, val: str, test: str) -> dict:
    out = dict(cfg)
    out["datasets"]["train"]["params"]["features_path"] = train
    out["datasets"]["val"]["params"]["features_path"] = val
    out["datasets"]["test"]["params"]["features_path"] = test
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Generate pinned reproducibility configs")
    p.add_argument("--out-dir", type=str, default="output/repro/configs")
    p.add_argument("--emb-features-dir", type=str, required=True, help="NCST grouprec+convnext features dir")
    p.add_argument(
        "--g3d-features-dir",
        type=str,
        required=True,
        help="NCST grouprec3d-only features dir",
    )
    p.add_argument(
        "--g3d-merged-features-dir",
        type=str,
        required=True,
        help="NCST merged convnext+grouprec3d features dir",
    )
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    emb = args.emb_features_dir.rstrip("/")
    g3d = args.g3d_features_dir.rstrip("/")
    g3dm = args.g3d_merged_features_dir.rstrip("/")

    plans = [
        (
            "convnext_only_loss_tuned",
            f"{emb}/train_features.pkl",
            f"{emb}/val_features.pkl",
            f"{emb}/test_features.pkl",
        ),
        (
            "grouprec_only",
            f"{emb}/train_features.pkl",
            f"{emb}/val_features.pkl",
            f"{emb}/test_features.pkl",
        ),
        (
            "grouprec_convnext_mlp",
            f"{emb}/train_features.pkl",
            f"{emb}/val_features.pkl",
            f"{emb}/test_features.pkl",
        ),
        (
            "grouprec_convnext_gated",
            f"{emb}/train_features.pkl",
            f"{emb}/val_features.pkl",
            f"{emb}/test_features.pkl",
        ),
        (
            "grouprec_convnext_gated_loss_tuned",
            f"{emb}/train_features.pkl",
            f"{emb}/val_features.pkl",
            f"{emb}/test_features.pkl",
        ),
        (
            "grouprec_convnext_cross_attention_loss_tuned",
            f"{emb}/train_features.pkl",
            f"{emb}/val_features.pkl",
            f"{emb}/test_features.pkl",
        ),
        (
            "grouprec_convnext_gated_c_spatial_graph_loss_tuned",
            f"{emb}/train_features_with_context.pkl",
            f"{emb}/val_features_with_context.pkl",
            f"{emb}/test_features_with_context.pkl",
        ),
        (
            "grouprec3d_only",
            f"{g3d}/train_features.pkl",
            f"{g3d}/val_features.pkl",
            f"{g3d}/test_features.pkl",
        ),
        (
            "grouprec3d_convnext_gated_loss_tuned",
            f"{g3dm}/train_features.pkl",
            f"{g3dm}/val_features.pkl",
            f"{g3dm}/test_features.pkl",
        ),
        (
            "grouprec3d_convnext_gated_c_spatial_graph_loss_tuned",
            f"{g3dm}/train_features_with_context.pkl",
            f"{g3dm}/val_features_with_context.pkl",
            f"{g3dm}/test_features_with_context.pkl",
        ),
    ]

    for model, tr, va, te in plans:
        src = ROOT / "prism" / "configs" / f"{model}.yaml"
        cfg = _load_yaml(src)
        cfg = _set_paths(cfg, tr, va, te)
        dst = ROOT / out_dir / f"{model}.yaml"
        _dump_yaml(dst, cfg)
        print(f"[OK] {dst}")


if __name__ == "__main__":
    main()

