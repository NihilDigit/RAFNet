from __future__ import annotations

from functools import partial
from typing import Any

import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

from prism.utils import Registry


OPTIMIZERS = Registry("optimizer")
SCHEDULERS = Registry("scheduler")


@OPTIMIZERS.register("adamw")
def build_adamw(params, lr: float = 1e-3, weight_decay: float = 1e-3, **kwargs: Any):
    return optim.AdamW(params, lr=lr, weight_decay=weight_decay, **kwargs)


@SCHEDULERS.register("reduce_on_plateau")
def build_reduce_lr_on_plateau(
    optimizer, mode: str = "max", factor: float = 0.5, patience: int = 10, **kwargs: Any
):
    return ReduceLROnPlateau(
        optimizer, mode=mode, factor=factor, patience=patience, **kwargs
    )


@SCHEDULERS.register("cosine_annealing")
def build_cosine_annealing(
    optimizer, T_max: int = 100, eta_min: float = 1e-6, **kwargs: Any
):
    return CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min, **kwargs)


__all__ = ["OPTIMIZERS", "SCHEDULERS"]
