from .datasets import MultiModalDataset
from prism.utils import Registry

DATASETS = Registry("dataset")


@DATASETS.register("multimodal_features")
def build_multimodal_dataset(**kwargs):
    return MultiModalDataset(**kwargs)


__all__ = ["DATASETS", "MultiModalDataset"]
