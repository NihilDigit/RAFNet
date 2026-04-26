from __future__ import annotations

from typing import Iterable

import numpy as np
import torch
import torch.nn as nn


class FocalLoss(nn.Module):
    """Focal Loss with class weights and label smoothing."""

    def __init__(
        self,
        num_classes: int,
        gamma: float = 2.0,
        class_weights: Iterable[float] | None = None,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.num_classes = num_classes
        self.label_smoothing = label_smoothing

        if class_weights is not None:
            self.register_buffer(
                "class_weights",
                torch.as_tensor(class_weights, dtype=torch.float32),
                persistent=False,
            )
        else:
            self.register_buffer(
                "class_weights",
                torch.ones(num_classes, dtype=torch.float32),
                persistent=False,
            )

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.class_weights.device != inputs.device:
            self.class_weights = self.class_weights.to(inputs.device)

        ce_loss = nn.functional.cross_entropy(
            inputs,
            targets,
            weight=self.class_weights,
            label_smoothing=self.label_smoothing,
            reduction="none",
        )

        pt = torch.exp(-ce_loss)
        loss = (1 - pt) ** self.gamma * ce_loss
        return loss.mean()


def compute_class_weights(
    labels: np.ndarray,
    num_classes: int = 5,
    method: str = "log_balanced",
    null_weight_factor: float = 0.3,
    beta: float = 0.999,
) -> np.ndarray:
    """Compute class weights via log_balanced, inverse, or effective_num."""

    counts = np.bincount(labels, minlength=num_classes)

    if method == "log_balanced":
        log_counts = np.log(counts + 1.0)
        weights = np.zeros_like(log_counts, dtype=np.float32)
        valid_mask = log_counts > 0

        if valid_mask.any():
            weights[valid_mask] = 1.0 / log_counts[valid_mask]
            total = weights[valid_mask].sum()
            if total > 0:
                weights = weights / total * valid_mask.sum()
        else:
            weights = np.ones(num_classes, dtype=np.float32)

        if len(weights) > 3:
            weights[3] *= null_weight_factor

    elif method == "inverse":
        weights = np.zeros(num_classes, dtype=np.float32)
        valid_mask = counts > 0

        if valid_mask.any():
            weights[valid_mask] = 1.0 / counts[valid_mask]
            total = weights[valid_mask].sum()
            if total > 0:
                weights = weights / total * valid_mask.sum()
        else:
            weights = np.ones(num_classes, dtype=np.float32)

    elif method == "effective_num":
        # Effective Number of Samples (https://arxiv.org/abs/1901.05555)
        weights = np.zeros(num_classes, dtype=np.float32)
        valid_mask = counts > 0

        if valid_mask.any():
            for c in range(num_classes):
                if counts[c] > 0:
                    beta_nc = beta ** counts[c]
                    if beta_nc < 1.0:
                        weights[c] = (1.0 - beta) / (1.0 - beta_nc)
                    else:
                        weights[c] = 1.0

            total = weights[valid_mask].sum()
            if total > 0:
                weights = weights / total * valid_mask.sum()
        else:
            weights = np.ones(num_classes, dtype=np.float32)

        if len(weights) > 3:
            weights[3] *= null_weight_factor

    else:
        weights = np.ones(num_classes, dtype=np.float32)

    return weights.astype(np.float32)


__all__ = ["FocalLoss", "compute_class_weights"]
