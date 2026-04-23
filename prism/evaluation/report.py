from __future__ import annotations

from typing import Iterable, Mapping

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix


def build_classification_report(
    labels: np.ndarray,
    predictions: np.ndarray,
    target_names: Iterable[str],
) -> str:
    return classification_report(labels, predictions, target_names=tuple(target_names), digits=4)


def build_confusion_matrix(labels: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    return confusion_matrix(labels, predictions)


def per_class_accuracy(labels: np.ndarray, predictions: np.ndarray) -> Mapping[int, float]:
    result: dict[int, float] = {}
    for cls in np.unique(labels):
        mask = labels == cls
        if mask.sum():
            result[int(cls)] = 100.0 * (predictions[mask] == cls).sum() / mask.sum()
        else:  # pragma: no cover - empty class
            result[int(cls)] = 0.0
    return result


__all__ = [
    "build_classification_report",
    "build_confusion_matrix",
    "per_class_accuracy",
]
