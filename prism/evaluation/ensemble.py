from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

import pickle

from prism.data import DATASETS
from prism.models import MODELS
from prism.utils import load_config


def _resolve_path(p: str | Path) -> Path:
    return Path(p).expanduser().resolve()


def validate_and_build_dataloader(
    configs: Iterable[dict[str, Any]], split: str = "test"
) -> DataLoader:
    """Validate consistent features_path/modalities across configs and build one shared DataLoader."""
    configs = list(configs)
    if not configs:
        raise ValueError("No configs provided for dataloader validation")

    ref = configs[0]
    if "datasets" not in ref or split not in ref["datasets"]:
        raise ValueError(f"Config missing datasets.{split} section")
    ref_ds = ref["datasets"][split]
    ref_params = ref_ds.get("params", {})
    ref_path = _resolve_path(ref_params.get("features_path"))
    ref_modalities = tuple(ref_params.get("modalities", []))

    for cfg in configs[1:]:
        if "datasets" not in cfg or split not in cfg["datasets"]:
            raise ValueError(f"Config missing datasets.{split} section")
        ds = cfg["datasets"][split]
        params = ds.get("params", {})
        path = _resolve_path(params.get("features_path"))
        modalities = tuple(params.get("modalities", []))
        if path != ref_path:
            raise ValueError(
                f"features_path mismatch for {split}:\n  {ref_path}\n!= {path}"
            )
        if modalities != ref_modalities:
            raise ValueError(
                f"modalities order mismatch: {ref_modalities} != {modalities}"
            )

    try:
        with open(ref_path, "rb") as f:
            data = pickle.load(f)
        md = (data.get("metadata") or {})
        file_modalities = tuple(md.get("modalities", []))
        if file_modalities and file_modalities != ref_modalities:
            raise ValueError(
                f"Feature file modalities mismatch: {file_modalities} != {ref_modalities}"
            )
    except Exception:
        pass

    dataset = DATASETS.build(ref_ds["name"], **ref_params)
    loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=0, drop_last=False)
    return loader


def build_model_from_config(config: dict[str, Any]) -> torch.nn.Module:
    if "model" not in config:
        raise ValueError("Config missing 'model' section")

    model_cfg = config["model"]
    model = MODELS.build(model_cfg["name"], **model_cfg.get("params", {}))

    return model


def infer_logits_for_model(
    model_name: str,
    config: dict[str, Any],
    dataloader: DataLoader,
    device: torch.device,
    batch_models: int = 2,
    use_amp: bool = True,
    preload_data: bool = True,
) -> dict[str, Any]:
    """Run inference for every seed of one model and return {'seeds', 'logits'}."""
    model_base_dir = Path("results/training") / model_name

    if not model_base_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_base_dir}")

    seed_dirs = sorted(model_base_dir.glob("seed_*"))
    if not seed_dirs:
        raise FileNotFoundError(f"No seed_* directories found in {model_base_dir}")

    features_all = None
    if preload_data and device.type == "cuda":
        print(f"  Preloading dataset to GPU...")
        features_batches = []
        for features, _ in dataloader:
            features_batches.append(features)
        features_all = torch.cat(features_batches).to(device)
        print(f"  ✓ Loaded {features_all.shape[0]} samples to GPU ({features_all.element_size() * features_all.nelement() / 1024**2:.1f} MB)")

    seeds = []
    all_logits = []

    for i in range(0, len(seed_dirs), batch_models):
        batch_seed_dirs = seed_dirs[i : i + batch_models]
        models = []
        batch_seeds = []

        for seed_dir in batch_seed_dirs:
            checkpoint_path = seed_dir / "best_model.pth"
            if not checkpoint_path.exists():
                print(f"  ⚠ Warning: Skipping {seed_dir.name} (no best_model.pth)")
                continue

            seed = int(seed_dir.name.split("_")[1])
            batch_seeds.append(seed)

            model = build_model_from_config(config)
            state_dict = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(state_dict)
            model = model.to(device)
            model.eval()
            models.append(model)

        if not models:
            continue

        print(f"  ✓ Batch {i // batch_models + 1}: Loaded {len(models)} models (seeds: {batch_seeds})")

        batch_logits = [[] for _ in models]

        with torch.no_grad():
            if features_all is not None:
                for idx, model in enumerate(models):
                    if use_amp and device.type == "cuda":
                        with torch.cuda.amp.autocast():
                            outputs = model(features_all)
                    else:
                        outputs = model(features_all)

                    # Gated/hybrid models return (logits, entropy_loss).
                    logits = outputs[0] if isinstance(outputs, tuple) else outputs
                    batch_logits[idx] = logits.cpu().numpy()
            else:
                for features, _ in dataloader:
                    features = features.to(device)
                    for idx, model in enumerate(models):
                        if use_amp and device.type == "cuda":
                            with torch.cuda.amp.autocast():
                                outputs = model(features)
                        else:
                            outputs = model(features)

                        logits = outputs[0] if isinstance(outputs, tuple) else outputs
                        batch_logits[idx].append(logits.cpu().numpy())

                batch_logits = [np.concatenate(logits_list, axis=0) for logits_list in batch_logits]

        for seed, logits_array in zip(batch_seeds, batch_logits):
            seeds.append(seed)
            all_logits.append(logits_array)
            print(f"    ✓ Seed {seed}: logits shape {logits_array.shape}")

        del models
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return {"seeds": seeds, "logits": all_logits}


def average_logits(
    logits_list: list[np.ndarray], weights: list[float] | None = None
) -> np.ndarray:
    """Weighted average of logits arrays; weights normalized to sum to 1."""
    if not logits_list:
        raise ValueError("logits_list is empty")

    if weights is None:
        weights = [1.0] * len(logits_list)

    if len(weights) != len(logits_list):
        raise ValueError(
            f"weights length ({len(weights)}) != logits_list length ({len(logits_list)})"
        )

    total_weight = sum(weights)
    if total_weight <= 0:
        raise ValueError("Sum of weights must be > 0")
    weights = [w / total_weight for w in weights]

    averaged = sum(w * logits for w, logits in zip(weights, logits_list))
    return averaged


def compute_metrics_from_logits(
    logits: np.ndarray, labels: np.ndarray
) -> dict[str, float | np.ndarray]:
    """Compute loss / accuracy / macro_f1(/_no_null) from logits and labels."""
    probs = torch.softmax(torch.from_numpy(logits), dim=-1).numpy()
    predictions = np.argmax(logits, axis=-1)

    eps = 1e-8
    log_probs = np.log(probs + eps)
    ce_loss = -np.mean(
        [log_probs[i, labels[i]] for i in range(len(labels))]
    )

    accuracy = 100.0 * (predictions == labels).sum() / len(labels)
    macro_f1 = 100.0 * f1_score(labels, predictions, average="macro", zero_division=0)

    # Null class is index 3 in PRISM's 5-class schema.
    valid_mask = labels != 3
    if valid_mask.sum() > 0:
        macro_f1_no_null = 100.0 * f1_score(
            labels[valid_mask],
            predictions[valid_mask],
            average="macro",
            zero_division=0,
        )
        accuracy_no_null = (
            100.0
            * (predictions[valid_mask] == labels[valid_mask]).sum()
            / valid_mask.sum()
        )
    else:
        macro_f1_no_null = 0.0
        accuracy_no_null = 0.0

    return {
        "loss": float(ce_loss),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "macro_f1": macro_f1,
        "macro_f1_no_null": macro_f1_no_null,
        "accuracy_no_null": accuracy_no_null,
        "predictions": predictions,
        "labels": labels,
    }


def generate_ensemble_name(
    model_names: list[str], custom_name: str | None = None
) -> str:
    """Format ensemble name; truncate with hash suffix when > 50 chars."""
    if custom_name:
        return custom_name

    joined = ",".join(model_names)
    name = f"ensemble({joined})"

    if len(name) > 50:
        hash_suffix = hashlib.md5(joined.encode()).hexdigest()[:6]
        truncated = joined[:30]
        name = f"ensemble({truncated})__{hash_suffix}"

    return name


__all__ = [
    "build_dataset_from_config",
    "build_model_from_config",
    "infer_logits_for_model",
    "average_logits",
    "compute_metrics_from_logits",
    "generate_ensemble_name",
]
