from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, log_loss


NULL_LABEL = 3


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray | None = None) -> Dict[str, float]:
    mask = y_true != NULL_LABEL

    metrics: Dict[str, float] = {
        "test_accuracy": float(accuracy_score(y_true, y_pred) * 100.0),
        "test_macro_f1": float(f1_score(y_true, y_pred, average="macro") * 100.0),
        "test_accuracy_no_null": float(accuracy_score(y_true[mask], y_pred[mask]) * 100.0) if mask.any() else 0.0,
        "test_macro_f1_no_null": float(f1_score(y_true[mask], y_pred[mask], average="macro") * 100.0) if mask.any() else 0.0,
    }

    if y_prob is not None:
        try:
            metrics["test_loss"] = float(log_loss(y_true, y_prob, labels=np.arange(y_prob.shape[1])))
        except Exception:
            metrics["test_loss"] = float("nan")
    else:
        metrics["test_loss"] = float("nan")

    return metrics
