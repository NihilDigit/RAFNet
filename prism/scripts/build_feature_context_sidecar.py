#!/usr/bin/env python3
"""Build context sidecar for extracted PRISM feature PKLs.

Purpose:
- Keep original feature extraction unchanged.
- Reconstruct per-sample spatial/context metadata aligned to feature sample order.

It can:
1) write `*_context.pkl` sidecar files (recommended), or
2) write merged `*_features_with_context.pkl` files.
"""

from __future__ import annotations

import argparse
import pickle
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Key:
    img_path: str
    bbox_id: str


def normalize_img_path(path: str) -> str:
    p = Path(path)
    if p.is_absolute():
        return p.as_posix()
    return p.as_posix()


def safe_date_from_iso(dt: str | None) -> str:
    if not dt:
        return "unknown"
    try:
        return datetime.fromisoformat(dt).date().isoformat()
    except Exception:
        return "unknown"


def build_raw_index(split_pkl: Path) -> dict[Key, dict[str, Any]]:
    with split_pkl.open("rb") as f:
        data = pickle.load(f)

    idx: dict[Key, dict[str, Any]] = {}
    for img in data["images"]:
        img_path = normalize_img_path(img["img_path"])
        h_w = img.get("h_w", (0, 0))
        img_h, img_w = int(h_w[0]), int(h_w[1])
        room_id = str(img.get("room_id", "unknown"))
        dt_iso = img.get("datetime_iso")
        date_str = safe_date_from_iso(dt_iso)
        for person_idx, b in enumerate(img["bboxes"]):
            x1, y1 = b["bbox"][0]
            x2, y2 = b["bbox"][1]
            key = Key(img_path=img_path, bbox_id=str(b["bbox_id"]))
            idx[key] = {
                "bbox_xyxy_abs": [float(x1), float(y1), float(x2), float(y2)],
                "image_hw": [img_h, img_w],
                "room_id": room_id,
                "datetime_iso": dt_iso,
                "date": date_str,
                "group_key": f"{room_id}_{date_str}",
                "person_idx": int(person_idx),
                "label": int(b["label"]),
            }
    return idx


def align_context(
    features_pkl: Path,
    raw_index: dict[Key, dict[str, Any]],
) -> dict[str, Any]:
    with features_pkl.open("rb") as f:
        feat = pickle.load(f)

    meta = feat.get("metadata", {}) or {}
    img_paths = meta.get("img_paths")
    bbox_ids = meta.get("bbox_ids")
    labels = feat.get("labels")

    if img_paths is None or bbox_ids is None:
        raise ValueError(f"{features_pkl} is missing metadata.img_paths / metadata.bbox_ids")

    n = len(img_paths)
    if len(bbox_ids) != n:
        raise ValueError(f"{features_pkl} img_paths and bbox_ids have different lengths")

    bbox_xyxy_abs = np.zeros((n, 4), dtype=np.float32)
    bbox_cxcywh_norm = np.zeros((n, 4), dtype=np.float32)
    image_hw = np.zeros((n, 2), dtype=np.int32)
    person_idx = np.full((n,), -1, dtype=np.int32)

    room_ids: list[str] = []
    datetime_isos: list[str | None] = []
    group_keys: list[str] = []

    missing = 0
    label_mismatch = 0

    for i, (img_path, bbox_id) in enumerate(zip(img_paths, bbox_ids)):
        key = Key(normalize_img_path(str(img_path)), str(bbox_id))
        row = raw_index.get(key)
        if row is None:
            missing += 1
            room_ids.append("unknown")
            datetime_isos.append(None)
            group_keys.append("unknown_unknown")
            continue

        x1, y1, x2, y2 = row["bbox_xyxy_abs"]
        h, w = row["image_hw"]

        bbox_xyxy_abs[i] = [x1, y1, x2, y2]
        image_hw[i] = [h, w]

        bw = max(x2 - x1, 1e-6)
        bh = max(y2 - y1, 1e-6)
        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5

        wn = max(float(w), 1e-6)
        hn = max(float(h), 1e-6)
        bbox_cxcywh_norm[i] = [cx / wn, cy / hn, bw / wn, bh / hn]

        person_idx[i] = int(row["person_idx"])
        room_ids.append(str(row["room_id"]))
        datetime_isos.append(row["datetime_iso"])
        group_keys.append(str(row["group_key"]))

        if labels is not None and int(labels[i]) != int(row["label"]):
            label_mismatch += 1

    return {
        "bbox_xyxy_abs": bbox_xyxy_abs,
        "bbox_cxcywh_norm": bbox_cxcywh_norm,
        "image_hw": image_hw,
        "person_idx": person_idx,
        "room_id": room_ids,
        "datetime_iso": datetime_isos,
        "group_key": group_keys,
        "metadata": {
            "source_features": str(features_pkl),
            "n_samples": n,
            "missing_samples": missing,
            "label_mismatch": label_mismatch,
        },
    }


def process_split(
    features_pkl: Path,
    split_pkl: Path,
    mode: str,
    output_path: Path | None,
) -> Path:
    raw_index = build_raw_index(split_pkl)
    context = align_context(features_pkl, raw_index)

    if mode == "sidecar":
        out = output_path or features_pkl.with_name(features_pkl.stem + "_context.pkl")
        with out.open("wb") as f:
            pickle.dump(context, f)
        print(f"[OK] sidecar: {out}")
        print(
            f"     missing={context['metadata']['missing_samples']} "
            f"label_mismatch={context['metadata']['label_mismatch']}"
        )
        return out

    # merged mode
    with features_pkl.open("rb") as f:
        feat = pickle.load(f)
    feat["context"] = {
        "bbox_xyxy_abs": context["bbox_xyxy_abs"],
        "bbox_cxcywh_norm": context["bbox_cxcywh_norm"],
        "image_hw": context["image_hw"],
        "person_idx": context["person_idx"],
        "room_id": context["room_id"],
        "datetime_iso": context["datetime_iso"],
        "group_key": context["group_key"],
    }
    feat.setdefault("metadata", {})["context_meta"] = context["metadata"]
    out = output_path or features_pkl.with_name(features_pkl.stem + "_with_context.pkl")
    with out.open("wb") as f:
        pickle.dump(feat, f)
    print(f"[OK] merged: {out}")
    print(
        f"     missing={context['metadata']['missing_samples']} "
        f"label_mismatch={context['metadata']['label_mismatch']}"
    )
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build feature context sidecar/merged PKL")
    p.add_argument("--features-path", type=str, default=None, help="single split feature pkl path")
    p.add_argument("--split-pkl", type=str, default=None, help="single split raw pkl path")
    p.add_argument(
        "--features-dir",
        type=str,
        default=None,
        help="directory containing train/val/test_features.pkl",
    )
    p.add_argument(
        "--data-root",
        type=str,
        default="data/datasets/ncst_classroom",
        help="dataset root containing train/val/test.pkl",
    )
    p.add_argument(
        "--mode",
        choices=["sidecar", "merged"],
        default="sidecar",
        help="output mode",
    )
    p.add_argument("--output-path", type=str, default=None, help="output path for single split")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.features_path and args.split_pkl:
        process_split(
            features_pkl=Path(args.features_path),
            split_pkl=Path(args.split_pkl),
            mode=args.mode,
            output_path=Path(args.output_path) if args.output_path else None,
        )
        return

    if args.features_dir:
        fdir = Path(args.features_dir)
        data_root = Path(args.data_root)
        for split in ["train", "val", "test"]:
            fpath = fdir / f"{split}_features.pkl"
            spath = data_root / f"{split}.pkl"
            if not fpath.exists() or not spath.exists():
                print(f"[SKIP] split={split} missing: {fpath if not fpath.exists() else spath}")
                continue
            process_split(features_pkl=fpath, split_pkl=spath, mode=args.mode, output_path=None)
        return

    raise SystemExit(
        "Provide either (--features-path and --split-pkl) or --features-dir."
    )


if __name__ == "__main__":
    main()
