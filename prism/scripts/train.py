#!/usr/bin/env python3
"""PRISM unified training driver with multi-seed support.

Example:
    python prism/scripts/train.py --model grouprec_only --gpu 0
    python prism/scripts/train.py --model grouprec_resnet_mlp --gpu 0 --seeds 41,42,43
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from prism.data import DATASETS  # noqa: E402
from prism.losses import LOSSES, compute_class_weights  # noqa: E402
from prism.models import MODELS  # noqa: E402
from prism.training import OPTIMIZERS, SCHEDULERS, evaluate, train_one_epoch  # noqa: E402
from prism.utils import load_config  # noqa: E402

# numpy>=2 renamed numpy.core -> numpy._core; our pkl features still reference the old path.
sys.modules["numpy._core"] = np.core
sys.modules["numpy._core._multiarray_umath"] = np.core._multiarray_umath


def apply_mixup_loss(
    criterion: torch.nn.Module,
    logits: torch.Tensor,
    labels: torch.Tensor,
    mixup_alpha: float = 0.0,
) -> torch.Tensor:
    """Mixup loss (https://arxiv.org/abs/1710.09412); mixup_alpha<=0 disables it."""
    if mixup_alpha <= 0:
        return criterion(logits, labels)

    batch_size = labels.size(0)
    lam = np.random.beta(mixup_alpha, mixup_alpha)

    perm = torch.randperm(batch_size, device=labels.device)
    labels_perm = labels[perm]

    loss_1 = criterion(logits, labels)
    loss_2 = criterion(logits, labels_perm)
    return lam * loss_1 + (1 - lam) * loss_2


def supervised_contrastive_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.2,
) -> torch.Tensor:
    """Within-batch supervised contrastive loss."""
    if features.size(0) < 2:
        return features.new_tensor(0.0)

    feats = torch.nn.functional.normalize(features, dim=1)
    logits = torch.matmul(feats, feats.t()) / max(temperature, 1e-6)

    logits_mask = torch.ones_like(logits, device=features.device)
    logits_mask.fill_diagonal_(0.0)

    label_eq = labels.unsqueeze(0) == labels.unsqueeze(1)
    pos_mask = label_eq.float() * logits_mask

    exp_logits = torch.exp(logits - logits.max(dim=1, keepdim=True).values) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)

    pos_count = pos_mask.sum(dim=1)
    valid = pos_count > 0
    if valid.sum() == 0:
        return features.new_tensor(0.0)

    mean_log_prob_pos = (pos_mask * log_prob).sum(dim=1) / (pos_count + 1e-8)
    loss = -mean_log_prob_pos[valid].mean()
    return loss


OFFICIAL_MODELS = {
    "grouprec_only",
    "grouprec3d_only",
    "grouprec3d_convnext_gated_loss_tuned",
    "grouprec3d_convnext_gated_c_spatial_graph_loss_tuned",
    "resnet_only",
    "resnet_only_loss_tuned",
    "vit_only",
    "vit_only_loss_tuned",
    "convnext_only",
    "grouprec_resnet_mlp",
    "grouprec_vit_mlp",
    "grouprec_convnext_mlp",
    "grouprec_resnet_gated",
    "grouprec_vit_gated",
    "grouprec_convnext_gated",
    "grouprec_convnext_gated_loss_tuned",
    "grouprec_convnext_gated_T0p5_E0p005",
    "grouprec_convnext_gated_T0p5_E0p01",
    "grouprec_convnext_gated_T0p5_E0p02",
    "grouprec_convnext_gated_T2p0_E0p005",
    "grouprec_convnext_gated_T2p0_E0p01",
    "grouprec_convnext_gated_T2p0_E0p02",
    "grouprec_convnext_gated_ae_loss_tuned",
    "grouprec_convnext_gated_b_disentangle_loss_tuned",
    "grouprec_convnext_gated_c_context_graph_loss_tuned",
    "grouprec_convnext_gated_c_spatial_graph_loss_tuned",
    "grouprec_convnext_gated_bc_spatial_loss_tuned",
    "grouprec_convnext_gated_ccsg_loss_tuned",
    "grouprec_convnext_gated_d_supcon_loss_tuned",
    "grouprec_resnet_transformer",
    "grouprec_vit_transformer",
    "grouprec_convnext_transformer",
    "grouprec_convnext_cross_attention_loss_tuned",
    "grouprec_resnet_loss_tuned",
    "grouprec_vit_loss_tuned",
    "grouprec_convnext_loss_tuned",
    "grouprec_convnext_gated_loss_tuned_effnum_gamma1p5",
    "grouprec_convnext_gated_loss_tuned_effnum_gamma2p0",
    "grouprec_convnext_gated_loss_tuned_effnum_gamma2p5",
    "grouprec_convnext_hybrid",
    "grouprec_convnext_hybrid_loss_tuned",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PRISM Unified Training Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python prism/scripts/train.py --model grouprec_only --gpu 0 --seed 43
  python prism/scripts/train.py --model grouprec_resnet_mlp --gpu 0 --seeds 41,42,43,44,45
        """,
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=sorted(OFFICIAL_MODELS),
        help="Model name to train",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU device index (default: 0)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Single random seed (default: None, use config seeds)",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Multiple random seeds (comma-separated, e.g., '41,42,43')",
    )
    parser.add_argument(
        "--config-override",
        type=str,
        default=None,
        help="Override config file path (optional)",
    )
    return parser.parse_args()


def get_git_commit_hash() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT_DIR,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "N/A"


def get_pixi_lock_hash() -> str:
    import hashlib

    pixi_lock = ROOT_DIR / "pixi.lock"
    if not pixi_lock.exists():
        return "N/A"

    try:
        with open(pixi_lock, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()[:8]
    except Exception:
        return "N/A"


def get_feature_metadata() -> Dict[str, Any]:
    metadata_path = ROOT_DIR / "output" / "prism_features" / "latest" / "run_metadata.json"
    if not metadata_path.exists():
        return {}

    try:
        with open(metadata_path, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def infer_null_class_idx(label_names: list[str]) -> int | None:
    """Return the index of the Null class in label_names, or None."""
    for idx, name in enumerate(label_names):
        if str(name).strip().lower() == "null":
            return idx
    return None


def build_dataloader(
    entry: Dict[str, Any], batch_size: int, shuffle: bool, num_workers: int = 4
) -> Tuple[Any, DataLoader]:
    dataset = DATASETS.build(entry["name"], **entry.get("params", {}))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )

    print(f"  Loaded {len(dataset)} samples from {entry['params']['features_path']}")
    if hasattr(dataset, "modalities"):
        for modality in dataset.modalities:
            tensor = getattr(dataset, modality, None)
            if tensor is not None:
                print(f"  {modality.capitalize()} shape: {tuple(tensor.shape)}")

    return dataset, loader


def prepare_loss(
    cfg: Dict[str, Any], labels: torch.Tensor, label_names: list[str]
) -> Tuple[torch.nn.Module, np.ndarray | None, Dict[str, Any]]:
    """Build loss function with class weights; returns (loss, weights, metadata)."""
    loss_cfg = cfg.get("loss", {"name": "focal", "params": {}})
    loss_params = dict(loss_cfg.get("params", {}))

    class_weights = None
    loss_metadata = {
        "method": None,
        "beta": None,
        "null_weight_factor": None,
        "focal_gamma": loss_params.get("gamma", 2.0),
    }

    cw_cfg = cfg.get("class_weights")
    if cw_cfg is not None:
        num_classes = (
            cw_cfg.get("num_classes")
            or loss_params.get("num_classes")
            or len(label_names)
        )
        method = cw_cfg.get("method", "log_balanced")
        null_weight_factor = cw_cfg.get("null_weight_factor", 1.0)
        beta = cw_cfg.get("params", {}).get("beta", 0.999) if method == "effective_num" else None

        loss_metadata["method"] = method
        loss_metadata["null_weight_factor"] = null_weight_factor
        loss_metadata["beta"] = beta

        if method == "effective_num":
            class_weights = compute_class_weights(
                labels.cpu().numpy(),
                num_classes=num_classes,
                method=method,
                null_weight_factor=null_weight_factor,
                beta=beta,
            )
        else:
            class_weights = compute_class_weights(
                labels.cpu().numpy(),
                num_classes=num_classes,
                method=method,
                null_weight_factor=null_weight_factor,
            )

        loss_params["class_weights"] = class_weights

    loss = LOSSES.build(loss_cfg["name"], **loss_params)
    return loss, class_weights, loss_metadata


def validate_config(cfg: Dict[str, Any]) -> None:
    """Ensure model/dataset modalities and input_dims agree."""
    model_modalities = cfg["model"]["params"].get("modalities", [])
    model_input_dims = cfg["model"]["params"].get("input_dims", [])

    if len(model_modalities) != len(model_input_dims):
        raise ValueError(
            f"In the model config, modalities ({len(model_modalities)}) and "
            f"input_dims ({len(model_input_dims)}) have different lengths"
        )

    for split in ["train", "val", "test"]:
        dataset_modalities = cfg["datasets"][split]["params"].get("modalities", [])
        if dataset_modalities != model_modalities:
            raise ValueError(
                f"{split} dataset modalities {dataset_modalities} "
                f"do not match model modalities {model_modalities}"
            )

    print("[OK] Configuration validated")


def train_single_seed(
    args: argparse.Namespace,
    cfg: Dict[str, Any],
    seed: int,
    base_output_dir: Path,
    config_path_used: str,
) -> Dict[str, Any]:
    """Train one seed end-to-end and return the final results dict."""
    set_seed(seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    output_dir = base_output_dir / f"seed_{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*80}")
    print(f"Training: {args.model} | Seed: {seed} | Device: {device}")
    print(f"Output: {output_dir}")
    print(f"{'='*80}\n")

    with open(output_dir / "config.yaml", "w") as f:
        import yaml

        yaml.dump(cfg, f, default_flow_style=False)

    print("=== Loading Datasets ===")
    train_dataset, train_loader = build_dataloader(
        cfg["datasets"]["train"],
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["training"].get("num_workers", 4),
    )
    val_dataset, val_loader = build_dataloader(
        cfg["datasets"]["val"],
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["training"].get("num_workers", 4),
    )
    test_dataset, test_loader = build_dataloader(
        cfg["datasets"]["test"],
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["training"].get("num_workers", 4),
    )

    print("\n=== Preparing Loss Function ===")
    train_labels = train_dataset.labels
    loss_fn, class_weights, loss_metadata = prepare_loss(
        cfg, train_labels, cfg["training"]["label_names"]
    )
    null_class_idx = infer_null_class_idx(cfg["training"]["label_names"])
    primary_metric_key = "macro_f1_no_null" if null_class_idx is not None else "macro_f1"
    primary_metric_name = "Macro F1(no-null)" if null_class_idx is not None else "Macro F1"
    print(f"Loss function: {cfg.get('loss', {}).get('name', 'focal')}")
    print(f"Focal gamma: {loss_metadata['focal_gamma']}")
    if class_weights is not None:
        print(f"Class weighting method: {loss_metadata['method']}")
        if loss_metadata['beta'] is not None:
            print(f"  Beta (effective_num): {loss_metadata['beta']}")
        print(f"  Null weight factor: {loss_metadata['null_weight_factor']}")
        print(f"Class weights: {class_weights}")
        label_names = cfg["training"]["label_names"]
        for i, (name, weight) in enumerate(zip(label_names, class_weights)):
            print(f"  [{i}] {name}: {weight:.4f}")

    print("\n=== Building Model ===")
    model = MODELS.build(cfg["model"]["name"], **cfg["model"]["params"]).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {cfg['model']['name']}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Gated/hybrid models return (logits, aux_loss) from forward; plain models return logits.
    is_gated = cfg["model"]["name"] in [
        "gated_fusion_classifier",
        "hybrid_fusion_classifier",
        "disentangled_gated_fusion_classifier",
        "context_graph_fusion_classifier",
        "disentangled_spatial_graph_fusion_classifier",
        "class_conditioned_spatial_gated_fusion_classifier",
    ]

    contrastive_cfg = cfg["training"].get("contrastive", {})
    supcon_enabled = bool(contrastive_cfg.get("enabled", False))
    supcon_weight = float(contrastive_cfg.get("weight", 0.0)) if supcon_enabled else 0.0
    supcon_temperature = float(contrastive_cfg.get("temperature", 0.2))
    if supcon_enabled and supcon_weight > 0:
        print(
            f"SupCon enabled: weight={supcon_weight:.4f}, "
            f"temperature={supcon_temperature:.3f}"
        )

    print("\n=== Building Optimizer & Scheduler ===")
    optimizer = OPTIMIZERS.build(
        cfg["optimizer"]["name"],
        params=model.parameters(),
        **cfg["optimizer"]["params"],
    )
    print(f"Optimizer: {cfg['optimizer']['name']}")
    print(f"LR: {cfg['optimizer']['params']['lr']}")

    scheduler = None
    if "scheduler" in cfg:
        scheduler = SCHEDULERS.build(
            cfg["scheduler"]["name"], optimizer=optimizer, **cfg["scheduler"]["params"]
        )
        print(f"Scheduler: {cfg['scheduler']['name']}")

    mixup_alpha = cfg["training"].get("mixup_alpha", 0.0)
    if mixup_alpha > 0:
        print(f"\n[OK] Mixup enabled: alpha={mixup_alpha}")
        if mixup_alpha > 1.0:
            print(f"  [!] Warning: mixup_alpha={mixup_alpha} > 1.0 is uncommon (recommended: 0.1-0.4)")

    print("\n=== Training ===")
    best_val_macro_f1_no_null = 0.0
    best_epoch = 0
    training_history = {"train": [], "val": []}

    for epoch in range(cfg["training"]["num_epochs"]):
        if is_gated:
            train_metrics = train_one_epoch_gated(
                model=model,
                train_loader=train_loader,
                criterion=loss_fn,
                optimizer=optimizer,
                device=device,
                grad_clip=cfg["training"].get("grad_clip", 1.0),
                mixup_alpha=mixup_alpha,
                supcon_weight=supcon_weight,
                supcon_temperature=supcon_temperature,
            )
        else:
            train_metrics = train_one_epoch_with_mixup(
                model=model,
                train_loader=train_loader,
                criterion=loss_fn,
                optimizer=optimizer,
                device=device,
                grad_clip=cfg["training"].get("grad_clip", 1.0),
                mixup_alpha=mixup_alpha,
                supcon_weight=supcon_weight,
                supcon_temperature=supcon_temperature,
            )

        if is_gated:
            val_metrics = evaluate_gated(
                model=model,
                data_loader=val_loader,
                criterion=loss_fn,
                device=device,
                null_class_idx=null_class_idx,
            )
        else:
            val_metrics = evaluate(
                model=model,
                data_loader=val_loader,
                criterion=loss_fn,
                device=device,
                null_class_idx=null_class_idx,
            )

        val_macro_f1 = val_metrics.get("macro_f1", 0.0)
        val_macro_f1_no_null = val_metrics.get("macro_f1_no_null", 0.0)
        val_accuracy = val_metrics.get("accuracy", 0.0)
        val_accuracy_no_null = val_metrics.get("accuracy_no_null", 0.0)

        if scheduler is not None:
            if hasattr(scheduler, "step"):
                scheduler.step(val_metrics.get(primary_metric_key, 0.0))

        train_record = {
            "epoch": epoch + 1,
            "loss": train_metrics.get("loss", 0.0),
            "accuracy": train_metrics.get("accuracy", 0.0),
        }
        val_record = {
            "epoch": epoch + 1,
            "loss": val_metrics.get("loss", 0.0),
            "macro_f1": val_macro_f1,
            "macro_f1_no_null": val_macro_f1_no_null,
            "accuracy": val_accuracy,
            "accuracy_no_null": val_accuracy_no_null,
        }
        training_history["train"].append(train_record)
        training_history["val"].append(val_record)

        print(
            f"Epoch {epoch + 1:3d}/{cfg['training']['num_epochs']} | "
            f"Train Loss: {train_metrics.get('loss', 0.0):.4f} | "
            f"Val {primary_metric_name}: {val_metrics.get(primary_metric_key, 0.0):.4f} | "
            f"Val Macro F1: {val_macro_f1:.4f} | "
            f"Val Acc: {val_accuracy:.2f}%"
        )

        current_metric = val_metrics.get(primary_metric_key, 0.0)
        if current_metric > best_val_macro_f1_no_null:
            best_val_macro_f1_no_null = current_metric
            best_epoch = epoch + 1
            model_path = output_dir / "best_model.pth"
            torch.save(model.state_dict(), model_path)
            print(
                "  [OK] Best model updated: "
                f"Macro F1(no-null)={best_val_macro_f1_no_null:.4f}"
            )

    best_model_path = output_dir / "best_model.pth"
    if best_epoch > 0 and best_model_path.exists():
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        print(
            f"\nLoaded best model checkpoint from epoch {best_epoch} "
            "for final test evaluation."
        )
    else:
        print(
            "\nBest checkpoint not found; using final epoch weights for test evaluation."
        )

    print("\n=== Final Evaluation on Test Set ===")
    if is_gated:
        test_metrics = evaluate_gated(
            model=model,
            data_loader=test_loader,
            criterion=loss_fn,
            device=device,
            null_class_idx=null_class_idx,
        )
    else:
        test_metrics = evaluate(
            model=model,
            data_loader=test_loader,
            criterion=loss_fn,
            device=device,
            null_class_idx=null_class_idx,
        )

    git_hash = get_git_commit_hash()
    pixi_hash = get_pixi_lock_hash()
    feature_metadata = get_feature_metadata()

    final_results = {
        "model": args.model,
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_macro_f1_no_null": best_val_macro_f1_no_null,
        "best_val_macro_f1": max(record["macro_f1"] for record in training_history["val"])
        if training_history["val"]
        else 0.0,
        "test_macro_f1": test_metrics.get("macro_f1", 0.0),
        "test_macro_f1_no_null": test_metrics.get("macro_f1_no_null", 0.0),
        "test_accuracy": test_metrics.get("accuracy", 0.0),
        "test_accuracy_no_null": test_metrics.get("accuracy_no_null", 0.0),
        "test_loss": test_metrics.get("loss", 0.0),
        "timestamp": datetime.now().isoformat(),
        "device": str(device),
        "git_commit": git_hash,
        "pixi_lock_hash": pixi_hash,
        "config_path": str(Path(config_path_used).resolve()),
        "feature_tag": feature_metadata.get("tag", "N/A"),
        "feature_modalities": feature_metadata.get("modalities", []),
        "dataset_features_path": {
            split: str(cfg["datasets"][split]["params"].get("features_path", ""))
            for split in ["train", "val", "test"]
        },
        "dataset_modalities": {
            split: list(cfg["datasets"][split]["params"].get("modalities", []))
            for split in ["train", "val", "test"]
        },
        "class_weighting": {
            "method": loss_metadata["method"],
            "beta": loss_metadata["beta"],
            "null_weight_factor": loss_metadata["null_weight_factor"],
        },
        "focal_gamma": loss_metadata["focal_gamma"],
        "null_class_idx": null_class_idx,
        "primary_metric_key": primary_metric_key,
        "contrastive": {
            "enabled": supcon_enabled,
            "weight": supcon_weight,
            "temperature": supcon_temperature,
        },
    }

    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(final_results, f, indent=2)

    history_path = output_dir / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(training_history, f, indent=2)

    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Seed: {seed}")
    print(f"Best Epoch: {best_epoch}")
    print(f"Best Val Macro F1(no-null): {best_val_macro_f1_no_null:.4f}")
    print(f"Test Macro F1 (no-null): {test_metrics.get('macro_f1_no_null', 0.0):.4f}")
    print(f"Test Macro F1: {test_metrics.get('macro_f1', 0.0):.4f}")
    print(f"Test Accuracy: {test_metrics.get('accuracy', 0.0):.2f}%")
    print(f"Test Accuracy (no-null): {test_metrics.get('accuracy_no_null', 0.0):.2f}%")
    print(f"Git Commit: {git_hash}")
    print(f"Results saved to: {results_path}")
    print("=" * 80)

    return final_results


def train_one_epoch_with_mixup(
    model,
    train_loader,
    criterion,
    optimizer,
    device,
    grad_clip=1.0,
    mixup_alpha=0.0,
    supcon_weight=0.0,
    supcon_temperature=0.2,
):
    """One training epoch for non-gated models with optional Mixup and SupCon."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for features, labels in train_loader:
        features = features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(features)
        aux_loss = 0.0
        if isinstance(outputs, tuple):
            logits, extra_loss = outputs
            aux_loss = extra_loss
        else:
            logits = outputs

        loss = apply_mixup_loss(criterion, logits, labels, mixup_alpha)
        if supcon_weight > 0:
            supcon = supervised_contrastive_loss(logits, labels, temperature=supcon_temperature)
            loss = loss + supcon_weight * supcon
        if isinstance(aux_loss, torch.Tensor):
            loss = loss + aux_loss

        loss.backward()

        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

        optimizer.step()

        total_loss += loss.item()
        _, predicted = logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    avg_loss = total_loss / max(len(train_loader), 1)
    accuracy = 100.0 * correct / max(total, 1)
    return {"loss": avg_loss, "accuracy": accuracy}


def train_one_epoch_gated(
    model,
    train_loader,
    criterion,
    optimizer,
    device,
    grad_clip=1.0,
    mixup_alpha=0.0,
    supcon_weight=0.0,
    supcon_temperature=0.2,
):
    """One training epoch for gated models: L = L_task + lambda * H(gate) + optional SupCon."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for features, labels in train_loader:
        features = features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits, entropy_loss = model(features)

        ce_loss = apply_mixup_loss(criterion, logits, labels, mixup_alpha)
        loss = ce_loss + entropy_loss
        if supcon_weight > 0:
            supcon = supervised_contrastive_loss(logits, labels, temperature=supcon_temperature)
            loss = loss + supcon_weight * supcon

        loss.backward()

        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        total_loss += loss.item() * features.size(0)
        _, predicted = logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    avg_loss = total_loss / total
    accuracy = 100.0 * correct / total

    return {"loss": avg_loss, "accuracy": accuracy}


def evaluate_gated(model, data_loader, criterion, device, null_class_idx: int | None = 3):
    """Evaluate a gated model; includes entropy loss in the reported loss."""
    from sklearn.metrics import f1_score

    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for features, labels in data_loader:
            features = features.to(device)
            labels = labels.to(device)

            logits, entropy_loss = model(features)
            ce_loss = criterion(logits, labels)
            loss = ce_loss + entropy_loss

            total_loss += loss.item() * features.size(0)
            _, predicted = logits.max(1)

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)

    avg_loss = total_loss / len(all_labels)
    accuracy = 100.0 * (all_predictions == all_labels).sum() / len(all_labels)

    macro_f1 = (
        f1_score(all_labels, all_predictions, average="macro", zero_division=0) * 100.0
        if len(all_labels) > 0
        else 0.0
    )

    if null_class_idx is None:
        macro_f1_no_null = macro_f1
        accuracy_no_null = accuracy
    else:
        valid_mask = all_labels != null_class_idx
        if valid_mask.sum() > 0:
            macro_f1_no_null = (
                f1_score(
                    all_labels[valid_mask],
                    all_predictions[valid_mask],
                    average="macro",
                    zero_division=0,
                )
                * 100.0
            )
            accuracy_no_null = (
                100.0
                * (all_predictions[valid_mask] == all_labels[valid_mask]).sum()
                / valid_mask.sum()
            )
        else:
            macro_f1_no_null = 0.0
            accuracy_no_null = 0.0

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "macro_f1_no_null": macro_f1_no_null,
        "accuracy_no_null": accuracy_no_null,
        "predictions": all_predictions,
        "labels": all_labels,
    }


def main(args: argparse.Namespace) -> None:
    if args.config_override:
        config_path = args.config_override
    else:
        config_path = f"prism/configs/{args.model}.yaml"

    print(f"\nLoading config from: {config_path}")
    cfg = load_config(config_path)

    print("\n=== Validating Configuration ===")
    validate_config(cfg)

    if args.seeds is not None:
        seeds = [int(s.strip()) for s in args.seeds.split(",")]
    elif args.seed is not None:
        seeds = [args.seed]
    elif "seeds" in cfg.get("training", {}):
        seeds = cfg["training"]["seeds"]
    else:
        seeds = [43]

    print(f"\nSeeds to train: {seeds}")

    base_output_dir = Path(cfg["training"]["output_dir"])

    all_results = []
    for seed in seeds:
        result = train_single_seed(args, cfg, seed, base_output_dir, config_path)
        all_results.append(result)

    if len(seeds) > 1:
        summary_path = base_output_dir / "all_seeds_summary.json"
        with open(summary_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nAll seeds summary saved to: {summary_path}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
