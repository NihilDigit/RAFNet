#!/usr/bin/env python3
"""Merge GroupRec-3D features with ConvNeXt(+context) features for fusion experiments.

Input A (base):   output/prism_features/.../{split}_features(.with_context).pkl
Input B (3D):     output/prism_features_3d/.../{split}_features.pkl
Output:           output/prism_features_g3d_convnext/.../{split}_features(.with_context).pkl

The script enforces strict sample alignment using metadata.img_paths + metadata.bbox_ids + labels.
"""

from __future__ import annotations

import argparse
import pickle
import shutil
from pathlib import Path

import numpy as np


SPLITS = ("train", "val", "test")
NCST_EXPECTED_SAMPLES = {
    "train": 35259,
    "val": 8250,
    "test": 8035,
}


def _load_pkl(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def _dump_pkl(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(obj, f)


def _check_alignment(base: dict, g3d: dict, split: str) -> None:
    base_meta = base.get("metadata", {}) or {}
    g3d_meta = g3d.get("metadata", {}) or {}

    base_labels = np.asarray(base["labels"])
    g3d_labels = np.asarray(g3d["labels"])
    if base_labels.shape[0] != g3d_labels.shape[0]:
        raise ValueError(
            f"[{split}] sample count mismatch: base={base_labels.shape[0]} vs g3d={g3d_labels.shape[0]}"
        )
    if not np.array_equal(base_labels, g3d_labels):
        raise ValueError(f"[{split}] labels mismatch between base and g3d features")

    base_paths = base_meta.get("img_paths")
    g3d_paths = g3d_meta.get("img_paths")
    base_bbox_ids = base_meta.get("bbox_ids")
    g3d_bbox_ids = g3d_meta.get("bbox_ids")
    if base_paths is None or g3d_paths is None or base_bbox_ids is None or g3d_bbox_ids is None:
        raise ValueError(f"[{split}] missing metadata.img_paths/bbox_ids for alignment check")

    if len(base_paths) != len(g3d_paths) or len(base_bbox_ids) != len(g3d_bbox_ids):
        raise ValueError(f"[{split}] metadata length mismatch for paths/bbox_ids")

    for i, (bp, gp, bb, gb) in enumerate(zip(base_paths, g3d_paths, base_bbox_ids, g3d_bbox_ids)):
        if str(bp) != str(gp) or str(bb) != str(gb):
            raise ValueError(
                f"[{split}] sample order mismatch at idx={i}: "
                f"base=({bp},{bb}) vs g3d=({gp},{gb})"
            )


def _assert_ncst_signature(base: dict, g3d: dict, split: str) -> None:
    """Hard guard: ensure features are from NCST split layout."""
    expected = NCST_EXPECTED_SAMPLES[split]
    base_n = int(np.asarray(base["labels"]).shape[0])
    g3d_n = int(np.asarray(g3d["labels"]).shape[0])
    if base_n != expected or g3d_n != expected:
        raise ValueError(
            f"[{split}] NCST hard-check failed: expected {expected}, "
            f"got base={base_n}, g3d={g3d_n}. "
            "Please use NCST feature directories."
        )

    base_meta = base.get("metadata", {}) or {}
    g3d_meta = g3d.get("metadata", {}) or {}
    base_root = str(base_meta.get("data_root", "")).lower()
    g3d_root = str(g3d_meta.get("data_root", "")).lower()
    if base_root and "ncst_classroom" not in base_root:
        raise ValueError(f"[{split}] base metadata.data_root is not NCST: {base_root}")
    if g3d_root and "ncst_classroom" not in g3d_root:
        raise ValueError(f"[{split}] g3d metadata.data_root is not NCST: {g3d_root}")


def _merge_one(base: dict, g3d: dict, split: str) -> dict:
    _assert_ncst_signature(base, g3d, split)
    _check_alignment(base, g3d, split)

    merged = dict(base)
    merged["grouprec_3d"] = np.asarray(g3d["grouprec_3d"], dtype=np.float32)

    meta = dict(merged.get("metadata", {}) or {})
    base_mods = list(meta.get("modalities", []))
    if "grouprec_3d" not in base_mods:
        base_mods.append("grouprec_3d")
    meta["modalities"] = base_mods

    dims = dict(meta.get("modal_feature_dims", {}) or {})
    dims["grouprec_3d"] = int(merged["grouprec_3d"].shape[1])
    meta["modal_feature_dims"] = dims
    meta["merge_note"] = "Merged convnext(+context) base with grouprec_3d features by strict sample alignment."
    merged["metadata"] = meta
    return merged


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge GroupRec-3D and ConvNeXt feature PKLs")
    p.add_argument(
        "--base-features-dir",
        type=str,
        default="output/prism_features/latest",
        help="Directory containing base split files from extract_features/build_feature_context_sidecar",
    )
    p.add_argument(
        "--grouprec3d-features-dir",
        type=str,
        default="output/prism_features_3d/latest",
        help="Directory containing split files from extract_grouprec3d_features.py",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default="output/prism_features_g3d_convnext/latest",
        help="Output directory for merged split files",
    )
    p.add_argument(
        "--with-context",
        action="store_true",
        help="Merge base *_features_with_context.pkl (for C+Spatial).",
    )
    p.add_argument(
        "--clean-output",
        action="store_true",
        help="Remove output dir before writing merged files.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    base_dir = Path(args.base_features_dir)
    g3d_dir = Path(args.grouprec3d_features_dir)
    out_dir = Path(args.output_dir)

    if args.clean_output and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_suffix = "_features_with_context.pkl" if args.with_context else "_features.pkl"
    out_suffix = base_suffix

    for split in SPLITS:
        base_path = base_dir / f"{split}{base_suffix}"
        g3d_path = g3d_dir / f"{split}_features.pkl"
        out_path = out_dir / f"{split}{out_suffix}"

        if not base_path.exists():
            raise FileNotFoundError(f"Missing base features: {base_path}")
        if not g3d_path.exists():
            raise FileNotFoundError(f"Missing grouprec3d features: {g3d_path}")

        base = _load_pkl(base_path)
        g3d = _load_pkl(g3d_path)
        merged = _merge_one(base, g3d, split=split)
        _dump_pkl(out_path, merged)
        n = merged["labels"].shape[0]
        d3 = merged["grouprec_3d"].shape[1]
        print(f"[OK] {split}: {out_path} (n={n}, grouprec_3d_dim={d3})")

    print("Done.")
    print(f"Output dir: {out_dir}")


if __name__ == "__main__":
    main()
