from __future__ import annotations

from pathlib import Path
import pickle

import torch
from torch.utils.data import Dataset


class MultiModalDataset(Dataset):
    """Load pre-extracted multimodal features from a pkl file.

    Each sample yields (concatenated_features, label). Feature file format
    follows `prism/scripts/extract_features.py`.
    """

    def __init__(
        self,
        features_path: str | Path,
        modalities: list[str] | None = None,
    ) -> None:
        self.features_path = Path(features_path)

        with self.features_path.open("rb") as f:
            data = pickle.load(f)

        # Merged context files expose spatial fields under data['context']; lift them
        # to pseudo-modalities (bbox_ctx, img_uid) so fusion classifiers can read them.
        context = data.get("context")
        if isinstance(context, dict):
            if "bbox_ctx" not in data and "bbox_cxcywh_norm" in context:
                data["bbox_ctx"] = context["bbox_cxcywh_norm"]
            if "img_uid" not in data:
                meta = data.get("metadata", {}) or {}
                img_paths = meta.get("img_paths")
                if img_paths is not None:
                    uid_map = {}
                    next_uid = 0
                    uids = []
                    for p in img_paths:
                        key = str(p)
                        if key not in uid_map:
                            uid_map[key] = next_uid
                            next_uid += 1
                        uids.append([float(uid_map[key])])
                    data["img_uid"] = uids

        metadata = data.get("metadata", {}) or {}
        available_modalities = metadata.get("modalities")
        if available_modalities is None:
            available_modalities = [
                k for k in data.keys() if k not in ("labels", "metadata")
            ]
        if "bbox_ctx" in data and "bbox_ctx" not in available_modalities:
            available_modalities = list(available_modalities) + ["bbox_ctx"]
        if "img_uid" in data and "img_uid" not in available_modalities:
            available_modalities = list(available_modalities) + ["img_uid"]
        if not available_modalities:
            raise ValueError(f"Feature file {self.features_path} contains no known modality keys")

        if modalities is not None:
            self.modalities = list(modalities)
        else:
            self.modalities = list(available_modalities)
        missing = [m for m in self.modalities if m not in data]
        if missing:
            raise KeyError(f"Feature file {self.features_path} is missing modalities: {missing}")

        self.features = {
            modality: torch.as_tensor(data[modality], dtype=torch.float32)
            for modality in self.modalities
        }
        modal_dims_meta = (
            metadata.get("modal_feature_dims", {}) if isinstance(metadata, dict) else {}
        )
        self.modal_dims = [
            int(modal_dims_meta.get(mod, self.features[mod].shape[1]))
            for mod in self.modalities
        ]

        for modality, tensor in self.features.items():
            setattr(self, modality, tensor)

        lengths = {tensor.shape[0] for tensor in self.features.values()}
        if len(lengths) != 1:
            raise ValueError(f"Sample counts differ between modalities: {lengths}")

        self.labels = torch.as_tensor(data["labels"], dtype=torch.long)
        if self.labels.shape[0] != next(iter(lengths)):
            raise ValueError("Number of labels does not match number of features")

    def __len__(self) -> int:  # pragma: no cover - simple delegation
        return self.labels.shape[0]

    def __getitem__(self, idx: int):
        feature_list = [self.features[modality][idx] for modality in self.modalities]
        features = torch.cat(feature_list)
        return features, self.labels[idx]


__all__ = ["MultiModalDataset"]
