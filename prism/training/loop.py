from __future__ import annotations

from typing import Dict

import numpy as np
import torch

from torch.utils.data import DataLoader


def train_one_epoch(
    model: torch.nn.Module,
    train_loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float | None = 1.0,
) -> Dict[str, float]:
    """Run a single training epoch.

    Args:
        model: model to train
        train_loader: training dataloader
        criterion: loss function
        optimizer: optimizer
        device: training device (CPU or GPU)
        grad_clip: max norm for gradient clipping; no clipping if None

    Returns:
        Training metrics dict containing 'loss' and 'accuracy'.
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for features, labels in train_loader:
        features = features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()

        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    avg_loss = total_loss / max(len(train_loader), 1)
    accuracy = 100.0 * correct / max(total, 1)
    return {"loss": avg_loss, "accuracy": accuracy}


def evaluate(
    model: torch.nn.Module,
    data_loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    null_class_idx: int | None = 3,
) -> Dict[str, float | np.ndarray]:
    """Evaluate model performance on a validation or test set.

    Args:
        model: model to evaluate
        data_loader: validation or test dataloader
        criterion: loss function
        device: evaluation device (CPU or GPU)

    Returns:
        Metrics dict containing:
            - loss: average loss
            - accuracy: overall accuracy (%)
            - macro_f1: overall macro F1 (%)
            - accuracy_no_null: accuracy excluding the Null class (%)
            - macro_f1_no_null: macro F1 excluding the Null class (%)
            - predictions: array of predicted labels
            - labels: array of ground-truth labels
    """
    model.eval()

    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for features, labels in data_loader:
            features = features.to(device)
            labels = labels.to(device)

            outputs = model(features)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)

            all_preds.append(predicted.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    if all_preds:
        preds = np.concatenate(all_preds)
        labels = np.concatenate(all_labels)
    else:  # pragma: no cover - empty dataloader
        preds = np.array([])
        labels = np.array([])

    avg_loss = total_loss / max(len(data_loader), 1)

    accuracy = 100.0 * (preds == labels).sum() / max(len(labels), 1)
    macro_f1 = _safe_macro_f1(labels, preds)
    if null_class_idx is None:
        accuracy_no_null = accuracy
        macro_f1_no_null = macro_f1
    else:
        mask_no_null = labels != null_class_idx
        if mask_no_null.sum() > 0:
            accuracy_no_null = (
                100.0
                * (preds[mask_no_null] == labels[mask_no_null]).sum()
                / mask_no_null.sum()
            )
            macro_f1_no_null = _safe_macro_f1(labels[mask_no_null], preds[mask_no_null])
        else:
            accuracy_no_null = 0.0
            macro_f1_no_null = 0.0

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "accuracy_no_null": accuracy_no_null,
        "macro_f1_no_null": macro_f1_no_null,
        "predictions": preds,
        "labels": labels,
    }


def _safe_macro_f1(labels: np.ndarray, preds: np.ndarray) -> float:
    from sklearn.metrics import f1_score

    if labels.size == 0:
        return 0.0
    return float(f1_score(labels, preds, average="macro") * 100.0)


__all__ = ["train_one_epoch", "evaluate"]
