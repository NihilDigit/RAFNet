#!/usr/bin/env python3
"""Extract GroupRec 3D-output features (pred_joints) for ablation baseline.

Output feature key: `grouprec_3d` with dim 78 (= 26 joints x 3).
"""

from __future__ import annotations

import argparse
import json
import pickle
import shutil
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# numpy compatibility
sys.modules["numpy._core"] = np.core
sys.modules["numpy._core._multiarray_umath"] = np.core._multiarray_umath

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from modules import ModelLoader, init  # noqa: E402
from utils.cliff_module import prepare_cliff  # noqa: E402


class NCSTDataset(Dataset):
    def __init__(self, data_path: Path, images_dir: Path):
        with data_path.open("rb") as f:
            data = pickle.load(f)
        self.images = data["images"]
        self.images_dir = images_dir

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        img_data = self.images[idx]
        rel_path = Path(img_data["img_path"])
        if (not rel_path.is_absolute()) and rel_path.parts and rel_path.parts[0] == self.images_dir.name:
            rel_path = Path(*rel_path.parts[1:])
        img_path = (self.images_dir / rel_path).resolve()
        return {
            "img_path": str(img_path),
            "img_path_rel": img_data["img_path"],
            "bboxes": img_data["bboxes"],
            "h_w": img_data["h_w"],
        }


def dynamic_collate_fn(batch):
    max_people = max(len(item["bboxes"]) for item in batch)
    max_people = max(max_people, 5)
    return {"batch": batch, "max_people": max_people}


class GroupRec3DExtractor:
    def __init__(self, model, device):
        self.model = model
        self.device = device

    @staticmethod
    def _sanitize_bbox_xyxy(bbox_xyxy, img_w: int, img_h: int):
        """Clamp and repair bbox to avoid invalid crop geometry.

        Returns a legal [x1, y1, x2, y2] with strictly positive width/height.
        """
        x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]

        # Reorder in case annotation has swapped corners.
        x1, x2 = sorted((x1, x2))
        y1, y2 = sorted((y1, y2))

        # Clamp inside image range.
        x1 = min(max(x1, 0.0), max(float(img_w - 1), 0.0))
        y1 = min(max(y1, 0.0), max(float(img_h - 1), 0.0))
        x2 = min(max(x2, 0.0), float(img_w))
        y2 = min(max(y2, 0.0), float(img_h))

        # Ensure strictly positive area.
        if x2 <= x1:
            x2 = min(float(img_w), x1 + 1.0)
            x1 = max(0.0, x2 - 1.0)
        if y2 <= y1:
            y2 = min(float(img_h), y1 + 1.0)
            y1 = max(0.0, y2 - 1.0)

        return [x1, y1, x2, y2]

    def extract_batch(self, batch_data, max_people):
        norm_imgs_list, centers_list, scales_list = [], [], []
        img_hs_list, img_ws_list, focal_lengthes_list, valid_list = [], [], [], []
        n_persons_list = []

        for img_data in batch_data:
            image = cv2.imread(img_data["img_path"])
            if image is None:
                return {"success": False, "error": f"Image load failed: {img_data['img_path']}"}
            image_rgb = image[:, :, ::-1].copy().astype(np.float32)
            h, w = image_rgb.shape[:2]

            boxes_xyxy = []
            for bbox_info in img_data["bboxes"]:
                (x1, y1), (x2, y2) = bbox_info["bbox"]
                boxes_xyxy.append(self._sanitize_bbox_xyxy([x1, y1, x2, y2], w, h))
            n_persons = len(boxes_xyxy)
            n_persons_list.append(n_persons)

            intris = [
                np.array([[5000.0, 0.0, w / 2.0], [0.0, 5000.0, h / 2.0], [0.0, 0.0, 1.0]], dtype=np.float32)
                for _ in range(n_persons)
            ]
            try:
                cliff_data = prepare_cliff(image_rgb, boxes_xyxy, intris=intris)
            except Exception:
                # Final fallback: enforce a known-safe tiny box for each person.
                safe_boxes = [[0.0, 0.0, 1.0, 1.0] for _ in range(n_persons)]
                cliff_data = prepare_cliff(image_rgb, safe_boxes, intris=intris)

            norm_imgs = torch.zeros((max_people, 3, 256, 192)).float()
            centers = torch.zeros((max_people, 2)).float()
            scales = torch.zeros((max_people,)).float()
            img_hs = torch.zeros((max_people,)).float()
            img_ws = torch.zeros((max_people,)).float()
            focal_lengthes = torch.ones((max_people,)).float() * 5000.0
            valid = torch.zeros((max_people,)).float()

            norm_imgs[:n_persons] = cliff_data["norm_img"]
            centers[:n_persons] = cliff_data["center"]
            scales[:n_persons] = cliff_data["scale"]
            img_hs[:n_persons] = cliff_data["img_h"]
            img_ws[:n_persons] = cliff_data["img_w"]
            focal_lengthes[:n_persons] = cliff_data["focal_length"]
            valid[:n_persons] = 1.0

            norm_imgs_list.append(norm_imgs)
            centers_list.append(centers)
            scales_list.append(scales)
            img_hs_list.append(img_hs)
            img_ws_list.append(img_ws)
            focal_lengthes_list.append(focal_lengthes)
            valid_list.append(valid)

        data = {
            "img": torch.stack(norm_imgs_list).to(self.device),
            "center": torch.stack(centers_list).to(self.device),
            "scale": torch.stack(scales_list).to(self.device),
            "img_h": torch.stack(img_hs_list).to(self.device),
            "img_w": torch.stack(img_ws_list).to(self.device),
            "focal_length": torch.stack(focal_lengthes_list).to(self.device),
            "valid": torch.stack(valid_list).to(self.device),
            "imgname": [item["img_path"] for item in batch_data],
        }

        self.model.model.eval()
        with torch.no_grad():
            pred = self.model.model(data)
            joints = pred["pred_joints"].detach().cpu().numpy()  # (num_valid, 26, 3)
            features = joints.reshape(joints.shape[0], -1)  # (num_valid, 78)

        features_per_image = []
        offset = 0
        for n_persons in n_persons_list:
            features_per_image.append(features[offset : offset + n_persons])
            offset += n_persons
        return {"success": True, "features": features_per_image}


def extract_split(
    extractor: GroupRec3DExtractor,
    data_path: Path,
    images_dir: Path,
    output_path: Path,
    split: str,
    batch_size: int,
):
    ds = NCSTDataset(data_path, images_dir)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=dynamic_collate_fn)

    features, labels, img_paths, bbox_ids = [], [], [], []
    label_dist = Counter()

    pbar = tqdm(dl, desc=f"Extracting {split}", ncols=100)
    for batch in pbar:
        batch_data = batch["batch"]
        max_people = batch["max_people"]
        out = extractor.extract_batch(batch_data, max_people)
        if not out["success"]:
            raise RuntimeError(out["error"])
        feats_per_img = out["features"]
        for img_data, img_feats in zip(batch_data, feats_per_img):
            bbs = img_data["bboxes"]
            for i, bb in enumerate(bbs):
                features.append(img_feats[i])
                labels.append(bb["label"])
                img_paths.append(img_data["img_path_rel"])
                bbox_ids.append(bb["bbox_id"])
                label_dist[int(bb["label"])] += 1

    feat_arr = np.asarray(features, dtype=np.float32)
    labels_arr = np.asarray(labels, dtype=np.int64)
    payload = {
        "grouprec_3d": feat_arr,
        "labels": labels_arr,
        "metadata": {
            "n_samples": int(len(labels_arr)),
            "modalities": ["grouprec_3d"],
            "modal_feature_dims": {"grouprec_3d": int(feat_arr.shape[1])},
            "label_distribution": dict(label_dist),
            "img_paths": img_paths,
            "bbox_ids": bbox_ids,
            "split": split,
        },
    }
    with output_path.open("wb") as f:
        pickle.dump(payload, f)
    print(f"[OK] {split}: {output_path} shape={feat_arr.shape}")


def main():
    p = argparse.ArgumentParser(description="Extract GroupRec 3D-output features (78D)")
    p.add_argument("--data-root", type=str, default="data/datasets/ncst_classroom")
    p.add_argument("--images-dir", type=str, default=None)
    p.add_argument("--grouprec-config", type=str, default="cfg_files/demo_smpl.yaml")
    p.add_argument("--output-root", type=str, default="output/prism_features_3d")
    p.add_argument("--run-tag", type=str, default=None)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--device", type=str, default="cuda:0")
    args = p.parse_args()

    data_root = (ROOT_DIR / args.data_root).resolve()
    images_dir = (ROOT_DIR / args.images_dir).resolve() if args.images_dir else (data_root / "images").resolve()

    run_tag = args.run_tag or datetime.now().strftime("%Y%m%d_%H%M%S_grouprec3d")
    output_root = (ROOT_DIR / args.output_root).resolve()
    run_dir = output_root / run_tag
    latest_tmp = output_root / "latest_tmp"
    latest = output_root / "latest"
    run_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if (not args.device.startswith("cuda") or torch.cuda.is_available()) else "cpu")

    cfg_path = (ROOT_DIR / args.grouprec_config).resolve()
    with cfg_path.open("r") as f:
        grouprec_args = yaml.safe_load(f)
    grouprec_args["config"] = str(cfg_path)
    grouprec_args["gpu_index"] = 0
    dtype = torch.float32
    out_dir, logger, smpl = init(dtype=dtype, **grouprec_args)
    model = ModelLoader(dtype=dtype, device=device, output=out_dir, **grouprec_args)

    extractor = GroupRec3DExtractor(model=model, device=device)
    for split in ("train", "val", "test"):
        extract_split(
            extractor=extractor,
            data_path=data_root / f"{split}.pkl",
            images_dir=images_dir,
            output_path=run_dir / f"{split}_features.pkl",
            split=split,
            batch_size=args.batch_size,
        )

    meta = {
        "run_tag": run_tag,
        "modalities": ["grouprec_3d"],
        "data_root": str(data_root),
        "images_dir": str(images_dir),
    }
    (run_dir / "run_metadata.json").write_text(json.dumps(meta, indent=2))

    if latest_tmp.exists():
        shutil.rmtree(latest_tmp)
    shutil.copytree(run_dir, latest_tmp)
    if latest.exists() or latest.is_symlink():
        if latest.is_symlink() or latest.is_file():
            latest.unlink()
        else:
            shutil.rmtree(latest)
    latest_tmp.replace(latest)
    print(f"[DONE] run_dir={run_dir}")
    print(f"[DONE] latest={latest}")


if __name__ == "__main__":
    main()
