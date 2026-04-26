#!/usr/bin/env python3
"""Extract per-modality features (GroupRec/ResNet/ViT/ConvNeXt) for each split.

Writes per-split pkl files under a timestamped run dir and mirrors the latest
run into `output/prism_features/latest/`.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path
import pickle
from collections import Counter
from typing import Any
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm
import yaml
import cv2

# numpy>=2 renamed numpy.core -> numpy._core; our pkl features still reference the old path.
sys.modules["numpy._core"] = np.core
sys.modules["numpy._core._multiarray_umath"] = np.core._multiarray_umath

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from modules import init, ModelLoader
from utils.cliff_module import prepare_cliff


class NCSTDataset(Dataset):
    """NCST classroom dataset (one sample = one image + its bboxes)."""

    def __init__(self, data_path, images_dir):
        with open(data_path, "rb") as f:
            data = pickle.load(f)

        self.images = data["images"]
        self.images_dir = Path(images_dir)
        self.split = data["metadata"]["split"]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_data = self.images[idx]
        rel_path = Path(img_data["img_path"])

        # Annotations occasionally prefix paths with images_dir's name; strip to avoid double-join.
        if (
            not rel_path.is_absolute()
            and rel_path.parts
            and rel_path.parts[0] == self.images_dir.name
        ):
            rel_path = Path(*rel_path.parts[1:])

        img_path = (self.images_dir / rel_path).resolve()

        return {
            "img_path": str(img_path),
            "img_path_rel": img_data["img_path"],
            "bboxes": img_data["bboxes"],
            "h_w": img_data["h_w"],
        }


def dynamic_collate_fn(batch):
    """Pad to per-batch max_people; GroupRec's hypergraph top-k needs >= 5."""
    max_people = max(len(item["bboxes"]) for item in batch)
    max_people = max(max_people, 5)

    return {"batch": batch, "max_people": max_people, "batch_size": len(batch)}


class GroupRecExtractor:
    """GroupRec 1024D feature extractor (hooks past_encoder output)."""

    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.fast_path = hasattr(self.model.model, "extract_relation_features")
        self.grouprec_features_captured = []

        # Older GroupRec builds lack extract_relation_features; fall back to a forward hook.
        if not self.fast_path:
            self.hook = self.model.model.past_encoder.register_forward_hook(self._hook_fn)
        else:
            self.hook = None

    def _hook_fn(self, module, input, output):
        self.grouprec_features_captured.append(output.detach().cpu())

    def extract_batch(self, batch_data, max_people):
        batch_size = len(batch_data)

        norm_imgs_list = []
        centers_list = []
        scales_list = []
        img_hs_list = []
        img_ws_list = []
        focal_lengthes_list = []
        valid_list = []
        n_persons_list = []

        for img_data in batch_data:
            try:
                image = cv2.imread(img_data["img_path"])
                if image is None:
                    raise ValueError(f"Image load failed: {img_data['img_path']}")

                image_rgb = image[:, :, ::-1].copy().astype(np.float32)
                h, w = image_rgb.shape[:2]

                boxes_xyxy = []
                for bbox_info in img_data["bboxes"]:
                    bbox = bbox_info["bbox"]
                    x1, y1 = bbox[0]
                    x2, y2 = bbox[1]
                    boxes_xyxy.append([x1, y1, x2, y2])

                n_persons = len(boxes_xyxy)
                n_persons_list.append(n_persons)

                intris = [
                    np.array(
                        [
                            [5000.0, 0.0, w / 2.0],
                            [0.0, 5000.0, h / 2.0],
                            [0.0, 0.0, 1.0],
                        ],
                        dtype=np.float32,
                    )
                    for _ in range(n_persons)
                ]

                cliff_data = prepare_cliff(image_rgb, boxes_xyxy, intris=intris)

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

            except Exception as e:
                return {
                    "success": False,
                    "error": f"Data preparation failed: {e}",
                    "img_path": img_data["img_path"],
                }

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

        try:
            with torch.no_grad():
                if self.fast_path:
                    # Skip SMPL regression heads; relation features alone suffice for downstream.
                    grouprec_features = (
                        self.model.model.extract_relation_features(data).detach().cpu().numpy()
                    )
                else:
                    self.grouprec_features_captured.clear()
                    _ = self.model.model(data)
                    if len(self.grouprec_features_captured) == 0:
                        return {"success": False, "error": "No grouprec features captured"}
                    grouprec_features = self.grouprec_features_captured[0].numpy()

            features_per_image = []
            offset = 0
            for n_persons in n_persons_list:
                img_features = grouprec_features[offset : offset + n_persons]
                features_per_image.append(img_features)
                offset += max_people

            return {
                "success": True,
                "features": features_per_image,
                "n_persons": n_persons_list,
            }

        except Exception as e:
            import traceback

            return {
                "success": False,
                "error": f"Forward failed: {e}\n{traceback.format_exc()}",
            }

    def __del__(self):
        if hasattr(self, "hook") and self.hook is not None:
            self.hook.remove()


class ResNetExtractor:
    """ResNet50 (2048D) feature extractor."""

    def __init__(self, device):
        self.device = device

        try:
            weights = models.ResNet50_Weights.DEFAULT  # type: ignore[attr-defined]
            resnet = models.resnet50(weights=weights)
            self.transform = weights.transforms()
        except AttributeError:
            resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)  # type: ignore[attr-defined]
            self.transform = transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )

        backbone = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*backbone, nn.Flatten())
        self.resnet.eval()
        self.resnet.to(device)

    def extract_batch(self, batch_data):
        all_crops = []
        crop_counts = []

        try:
            for img_data in batch_data:
                image = Image.open(img_data["img_path"]).convert("RGB")
                img_w, img_h = image.size

                crops = []
                for bbox_info in img_data["bboxes"]:
                    bbox = bbox_info["bbox"]
                    x1, y1 = bbox[0]
                    x2, y2 = bbox[1]
                    x1, y1 = max(0, int(x1)), max(0, int(y1))
                    x2, y2 = min(img_w, int(x2)), min(img_h, int(y2))

                    if x2 <= x1 or y2 <= y1:
                        crops.append(torch.zeros(3, 224, 224))
                    else:
                        crop = image.crop((x1, y1, x2, y2))
                        crop_tensor = self.transform(crop)
                        crops.append(crop_tensor)

                all_crops.extend(crops)
                crop_counts.append(len(crops))

            if len(all_crops) == 0:
                return {"success": False, "error": "No valid crops"}

            crops_batch = torch.stack(all_crops).to(self.device)

            with torch.no_grad():
                features = self.resnet(crops_batch)

            features_np = features.cpu().numpy()

            features_per_image = []
            offset = 0
            for count in crop_counts:
                img_features = features_np[offset : offset + count]
                features_per_image.append(img_features)
                offset += count

            return {
                "success": True,
                "features": features_per_image,
            }

        except Exception as e:
            import traceback

            return {
                "success": False,
                "error": f"ResNet extraction failed: {e}\n{traceback.format_exc()}",
            }


class VisionTransformerExtractor:
    """ViT-B/16 (768D) feature extractor."""

    def __init__(self, device):
        self.device = device

        try:
            weights = models.ViT_B_16_Weights.DEFAULT  # type: ignore[attr-defined]
            vit = models.vit_b_16(weights=weights)
            self.transform = weights.transforms()
            self.feature_dim = vit.hidden_dim
        except AttributeError:
            try:
                vit = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)  # type: ignore[attr-defined]
            except Exception:
                vit = models.vit_b_16(pretrained=True)  # type: ignore[call-arg]
            self.transform = transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
            self.feature_dim = 768

        vit.heads = nn.Identity()
        self.vit = vit.to(device)
        self.vit.eval()

    def extract_batch(self, batch_data):
        all_crops = []
        crop_counts = []

        try:
            for img_data in batch_data:
                image = Image.open(img_data["img_path"]).convert("RGB")
                img_w, img_h = image.size

                crops = []
                for bbox_info in img_data["bboxes"]:
                    bbox = bbox_info["bbox"]
                    x1, y1 = bbox[0]
                    x2, y2 = bbox[1]
                    x1, y1 = max(0, int(x1)), max(0, int(y1))
                    x2, y2 = min(img_w, int(x2)), min(img_h, int(y2))

                    if x2 <= x1 or y2 <= y1:
                        crops.append(torch.zeros(3, 224, 224))
                    else:
                        crop = image.crop((x1, y1, x2, y2))
                        crop_tensor = self.transform(crop)
                        crops.append(crop_tensor)

                all_crops.extend(crops)
                crop_counts.append(len(crops))

            if len(all_crops) == 0:
                return {"success": False, "error": "No valid crops"}

            crops_batch = torch.stack(all_crops).to(self.device)

            with torch.no_grad():
                features = self.vit(crops_batch)

            features_np = features.cpu().numpy()

            features_per_image = []
            offset = 0
            for count in crop_counts:
                img_features = features_np[offset : offset + count]
                features_per_image.append(img_features)
                offset += count

            return {
                "success": True,
                "features": features_per_image,
            }

        except Exception as e:
            import traceback

            return {
                "success": False,
                "error": f"ViT extraction failed: {e}\n{traceback.format_exc()}",
            }


class ConvNeXtExtractor:
    """ConvNeXt-Base (1024D) feature extractor via timm."""

    def __init__(self, device):
        self.device = device
        # Crowded classroom images can exceed 100 bboxes; chunk to avoid OOM.
        self.forward_chunk_size = 64

        try:
            import timm
        except ImportError:
            raise ImportError(
                "timm is required for ConvNeXt. Install with: pip install timm"
            )

        self.model = timm.create_model(
            "convnext_base",
            pretrained=True,
            num_classes=0,
            global_pool="avg",
        )
        self.model = self.model.to(device)
        self.model.eval()

        self.feature_dim = 1024

        data_config = timm.data.resolve_model_data_config(self.model)
        self.transform = timm.data.create_transform(**data_config, is_training=False)

    def extract_batch(self, batch_data):
        all_crops = []
        crop_counts = []

        try:
            for img_data in batch_data:
                image = Image.open(img_data["img_path"]).convert("RGB")
                img_w, img_h = image.size

                crops = []
                for bbox_info in img_data["bboxes"]:
                    bbox = bbox_info["bbox"]
                    x1, y1 = bbox[0]
                    x2, y2 = bbox[1]
                    x1, y1 = max(0, int(x1)), max(0, int(y1))
                    x2, y2 = min(img_w, int(x2)), min(img_h, int(y2))

                    if x2 <= x1 or y2 <= y1:
                        crops.append(torch.zeros(3, 224, 224))
                    else:
                        crop = image.crop((x1, y1, x2, y2))
                        crop_tensor = self.transform(crop)
                        crops.append(crop_tensor)

                all_crops.extend(crops)
                crop_counts.append(len(crops))

            if len(all_crops) == 0:
                return {"success": False, "error": "No valid crops"}

            with torch.no_grad():
                feat_chunks = []
                for i in range(0, len(all_crops), self.forward_chunk_size):
                    chunk = all_crops[i : i + self.forward_chunk_size]
                    crops_batch = torch.stack(chunk).to(self.device)
                    feat_chunks.append(self.model(crops_batch).cpu())
                features_np = torch.cat(feat_chunks, dim=0).numpy()

            features_per_image = []
            offset = 0
            for count in crop_counts:
                img_features = features_np[offset : offset + count]
                features_per_image.append(img_features)
                offset += count

            return {
                "success": True,
                "features": features_per_image,
            }

        except Exception as e:
            import traceback

            return {
                "success": False,
                "error": f"ConvNeXt extraction failed: {e}\n{traceback.format_exc()}",
            }


class FeatureExtractor:
    """Dispatch per-modality extractors and write flat-format pkl outputs."""

    SUPPORTED_MODALITIES = ("grouprec", "resnet", "vit", "convnext")

    def __init__(self, grouprec_model, device, run_dir, latest_dir, modalities):
        self.device = device
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.latest_dir = Path(latest_dir) if latest_dir is not None else None
        self.modalities = [m for m in modalities if m in self.SUPPORTED_MODALITIES]
        if not self.modalities:
            raise ValueError(f"No valid modality selected; supported: {self.SUPPORTED_MODALITIES}")

        self.grouprec_extractor = (
            GroupRecExtractor(grouprec_model, device)
            if "grouprec" in self.modalities
            else None
        )
        self.resnet_extractor = (
            ResNetExtractor(device) if "resnet" in self.modalities else None
        )
        self.vit_extractor = (
            VisionTransformerExtractor(device) if "vit" in self.modalities else None
        )
        self.convnext_extractor = (
            ConvNeXtExtractor(device) if "convnext" in self.modalities else None
        )

        self.features = {mod: [] for mod in self.modalities}
        self.features.update(
            {
                "labels": [],
                "img_paths": [],
                "bbox_ids": [],
            }
        )

        self.stats = {
            "total_images": 0,
            "total_samples": 0,
            "success_images": 0,
            "failed_images": [],
            "label_distribution": Counter(),
        }

    def process_batch(self, batch_dict):
        batch_data = batch_dict["batch"]
        max_people = batch_dict["max_people"]

        modal_results: dict[str, list[np.ndarray]] = {}

        if self.grouprec_extractor is not None:
            grouprec_result = self.grouprec_extractor.extract_batch(
                batch_data, max_people
            )
            if not grouprec_result["success"]:
                self.stats["failed_images"].append(
                    {
                        "modality": "grouprec",
                        "img_path": grouprec_result.get("img_path", "unknown"),
                        "error": grouprec_result["error"],
                    }
                )
                return
            modal_results["grouprec"] = grouprec_result["features"]

        if self.resnet_extractor is not None:
            resnet_result = self.resnet_extractor.extract_batch(batch_data)
            if not resnet_result["success"]:
                self.stats["failed_images"].append(
                    {
                        "modality": "resnet",
                        "img_path": batch_data[0]["img_path"]
                        if batch_data
                        else "unknown",
                        "error": resnet_result["error"],
                    }
                )
                return
            modal_results["resnet"] = resnet_result["features"]
        if self.vit_extractor is not None:
            vit_result = self.vit_extractor.extract_batch(batch_data)
            if not vit_result["success"]:
                self.stats["failed_images"].append(
                    {
                        "modality": "vit",
                        "img_path": batch_data[0]["img_path"]
                        if batch_data
                        else "unknown",
                        "error": vit_result["error"],
                    }
                )
                return
            modal_results["vit"] = vit_result["features"]

        if self.convnext_extractor is not None:
            convnext_result = self.convnext_extractor.extract_batch(batch_data)
            if not convnext_result["success"]:
                self.stats["failed_images"].append(
                    {
                        "modality": "convnext",
                        "img_path": batch_data[0]["img_path"]
                        if batch_data
                        else "unknown",
                        "error": convnext_result["error"],
                    }
                )
                return
            modal_results["convnext"] = convnext_result["features"]

        for idx, img_data in enumerate(batch_data):
            n_persons = len(img_data["bboxes"])

            per_modality_feats = {}
            for modality in self.modalities:
                feats_list = modal_results.get(modality)
                if feats_list is None:
                    continue
                modality_feats = feats_list[idx]
                if modality_feats.shape[0] != n_persons:
                    raise ValueError(
                        f"{modality} feature shape mismatch: {modality_feats.shape}, "
                        f"expected ({n_persons}, ?)"
                    )
                per_modality_feats[modality] = modality_feats

            for person_idx, bbox_info in enumerate(img_data["bboxes"]):
                for modality in self.modalities:
                    modality_feats = per_modality_feats.get(modality)
                    if modality_feats is not None:
                        self.features[modality].append(modality_feats[person_idx])
                self.features["labels"].append(bbox_info["label"])
                self.features["img_paths"].append(img_data["img_path_rel"])
                self.features["bbox_ids"].append(bbox_info["bbox_id"])

                self.stats["label_distribution"][bbox_info["label"]] += 1
                self.stats["total_samples"] += 1

            self.stats["total_images"] += 1
            self.stats["success_images"] += 1

    def save_features(self, split_name, latest_dir: Path = None):
        """Write the split pkl and return a stats snapshot."""
        print(f"\n{'=' * 80}")
        print(f"Saving features: {split_name}")
        print(f"{'=' * 80}")

        target_latest = Path(latest_dir) if latest_dir is not None else self.latest_dir

        modal_arrays = {mod: np.array(self.features[mod]) for mod in self.modalities}
        modal_dims = {
            mod: int(array.shape[1]) if array.ndim == 2 else None
            for mod, array in modal_arrays.items()
        }
        features_dict = {
            **modal_arrays,
            "labels": np.array(self.features["labels"]),
            "metadata": {
                "img_paths": self.features["img_paths"],
                "bbox_ids": self.features["bbox_ids"],
                "n_samples": len(self.features["labels"]),
                "modalities": self.modalities,
                "modal_feature_dims": modal_dims,
            },
        }

        output_file = self.run_dir / f"{split_name}_features.pkl"
        with open(output_file, "wb") as f:
            pickle.dump(features_dict, f)

        print("\nFeature dims:")
        for modality in self.modalities:
            print(f"  {modality.capitalize()}: {features_dict[modality].shape}")
        print(f"  Labels: {features_dict['labels'].shape}")
        print("\nLabel distribution:")
        total = len(features_dict["labels"])
        for label in sorted(self.stats["label_distribution"].keys()):
            count = self.stats["label_distribution"][label]
            pct = 100 * count / total
            print(f"  Class {label}: {count:5d} ({pct:5.2f}%)")
        print(f"\nSaved to: {output_file}")
        print(f"File size: {output_file.stat().st_size / 1024 / 1024:.2f} MB")

        print(f"\n{'=' * 80}")
        print(
            f"Successfully processed images: {self.stats['success_images']}/{self.stats['total_images']}"
        )
        print(f"Failed images: {len(self.stats['failed_images'])}")
        print(f"Total samples: {self.stats['total_samples']}")

        if self.stats["failed_images"]:
            failed_path = self.run_dir / f"{split_name}_failed.txt"
            with open(failed_path, "w") as f:
                for item in self.stats["failed_images"]:
                    f.write(f"{item.get('img_path', 'unknown')}\n")
                    if "modality" in item:
                        f.write(f"  Modality: {item['modality']}\n")
                    f.write(f"  Error: {item['error']}\n\n")
            print(f"\nFailure log saved to: {failed_path}")
        else:
            failed_path = None

        if target_latest is not None:
            target_latest.mkdir(parents=True, exist_ok=True)
            shutil.copy2(output_file, target_latest / output_file.name)
            if failed_path and failed_path.exists():
                shutil.copy2(failed_path, target_latest / failed_path.name)

        stats_snapshot = {
            "features_path": str(output_file.resolve()),
            "modalities": self.modalities,
            "n_modalities": len(self.modalities),
            "modal_feature_dims": modal_dims,
            "total_images": self.stats["total_images"],
            "success_images": self.stats["success_images"],
            "failed_images": len(self.stats["failed_images"]),
            "total_samples": self.stats["total_samples"],
            "label_distribution": {
                int(k): int(v) for k, v in self.stats["label_distribution"].items()
            },
        }
        if failed_path:
            stats_snapshot["failed_log"] = str(failed_path.resolve())
            stats_snapshot["failed_records"] = self.stats["failed_images"]

        return stats_snapshot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PRISM multi-modality feature extraction script")
    parser.add_argument(
        "--data-root",
        type=str,
        default="data/datasets/ncst_classroom",
        help="Directory containing the preprocessed train/val/test.pkl files",
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        default="data/datasets/ncst_classroom/images",
        help="Root directory of raw images",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="cfg_files/demo_smpl.yaml",
        help="GroupRec/CLIFF config file",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="output/prism_features",
        help="Root directory for saved features",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="DataLoader batch size (default 1 to avoid GPU OOM)",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Experiment tag appended to the output directory name",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        help="List of data splits to process",
    )
    parser.add_argument(
        "--modalities",
        nargs="+",
        choices=["grouprec", "resnet", "vit", "convnext"],
        default=["grouprec", "resnet", "vit", "convnext"],
        help="List of modalities to extract (default: all four)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 80)
    print("PRISM feature extraction (improved)")
    print("=" * 80)

    batch_size = args.batch_size
    data_root = Path(args.data_root)
    images_dir = Path(args.images_dir)
    output_root = Path(args.output_root)
    config_path = Path(args.config)
    splits = [split.lower() for split in args.splits]
    selected_modalities = [mod.lower() for mod in args.modalities]

    output_root.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{timestamp}_{args.tag}" if args.tag else timestamp
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=False)

    latest_tmp_dir = output_root / "latest_tmp"
    if latest_tmp_dir.exists():
        shutil.rmtree(latest_tmp_dir)
    latest_tmp_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nRun tag: {args.tag or 'default'}")
    print(f"Run output dir: {run_dir}")
    print(f"Temporary latest dir: {latest_tmp_dir}")

    dtype = torch.float32
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    use_grouprec = "grouprec" in selected_modalities
    grouprec_args = {}
    model = None
    if use_grouprec:
        with config_path.open("r") as f:
            grouprec_args = yaml.safe_load(f)

        grouprec_args["config"] = str(config_path)
        grouprec_args["gpu_index"] = 0

        print("\nInitializing model...")
        out_dir, logger, smpl = init(dtype=dtype, **grouprec_args)
        model = ModelLoader(dtype=dtype, device=device, output=out_dir, **grouprec_args)
    else:
        print("\nSkipping GroupRec/SMPL init (grouprec modality not selected)")

    run_metadata: dict[str, Any] = {
        "timestamp": timestamp,
        "tag": args.tag,
        "run_dir": str(run_dir.resolve()),
        "latest_dir": str((output_root / "latest").resolve()),
        "config_path": str(config_path.resolve()),
        "batch_size": batch_size,
        "device": str(device),
        "modalities": selected_modalities,
        "splits": {},
    }

    success = False
    try:
        for split in splits:
            print(f"\n{'=' * 80}")
            print(f"Processing split: {split}")
            print(f"{'=' * 80}")

            data_path = data_root / f"{split}.pkl"
            if not data_path.exists():
                print(f"Skipping: {data_path} does not exist")
                run_metadata["splits"][split] = {
                    "status": "skipped",
                    "reason": "missing_split_file",
                }
                continue

            dataset = NCSTDataset(data_path, images_dir)
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=dynamic_collate_fn,
                num_workers=0,
            )

            print(f"Dataset size: {len(dataset)} images")
            print(f"Batch size: {batch_size}")

            extractor = FeatureExtractor(
                model, device, run_dir, latest_tmp_dir, selected_modalities
            )
            run_metadata["modalities"] = extractor.modalities

            for batch_dict in tqdm(dataloader, desc=f"Extracting {split}"):
                extractor.process_batch(batch_dict)

            split_info = extractor.save_features(split, latest_tmp_dir)
            split_info["status"] = "completed"
            run_metadata["splits"][split] = split_info

        metadata_path = run_dir / "run_metadata.json"
        with metadata_path.open("w") as f:
            json.dump(run_metadata, f, indent=2, ensure_ascii=False)

        latest_metadata = latest_tmp_dir / "run_metadata.json"
        with latest_metadata.open("w") as f:
            json.dump(run_metadata, f, indent=2, ensure_ascii=False)

        success = True
    finally:
        if success:
            latest_dir = output_root / "latest"
            if latest_dir.exists():
                shutil.rmtree(latest_dir)
            latest_tmp_dir.rename(latest_dir)
        else:
            print("\nAn exception occurred; cleaning up incomplete output directories.")
            shutil.rmtree(run_dir, ignore_errors=True)
            shutil.rmtree(latest_tmp_dir, ignore_errors=True)

    print(f"\n{'=' * 80}")
    print("Feature extraction complete.")
    print(f"{'=' * 80}")
    print(f"\nRun output dir: {run_dir}")
    print(f"Latest snapshot: {output_root / 'latest'}")


if __name__ == "__main__":
    main()
