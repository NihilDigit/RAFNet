from .focal import FocalLoss, compute_class_weights
from prism.utils import Registry

LOSSES = Registry("loss")


@LOSSES.register("focal")
def build_focal_loss(**kwargs):
    return FocalLoss(**kwargs)


__all__ = ["LOSSES", "FocalLoss", "compute_class_weights"]
