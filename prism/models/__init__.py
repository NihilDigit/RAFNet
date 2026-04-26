from .multimodal import MultiModalClassifier
from .gated import GatedFusionClassifier
from .disentangled import DisentangledGatedFusionClassifier
from .context_graph import ContextGraphFusionClassifier
from .bc_spatial import DisentangledSpatialGraphFusionClassifier
from .ccsg import ClassConditionedSpatialGatedFusionClassifier
from .cross_attention import CrossAttentionFusionClassifier
from prism.utils import Registry

MODELS = Registry("model")


@MODELS.register("multimodal_classifier")
def build_multimodal_classifier(**kwargs):
    return MultiModalClassifier(**kwargs)


@MODELS.register("gated_fusion_classifier")
def build_gated_fusion_classifier(**kwargs):
    return GatedFusionClassifier(**kwargs)


@MODELS.register("disentangled_gated_fusion_classifier")
def build_disentangled_gated_fusion_classifier(**kwargs):
    return DisentangledGatedFusionClassifier(**kwargs)


@MODELS.register("context_graph_fusion_classifier")
def build_context_graph_fusion_classifier(**kwargs):
    return ContextGraphFusionClassifier(**kwargs)

@MODELS.register("disentangled_spatial_graph_fusion_classifier")
def build_disentangled_spatial_graph_fusion_classifier(**kwargs):
    return DisentangledSpatialGraphFusionClassifier(**kwargs)

@MODELS.register("class_conditioned_spatial_gated_fusion_classifier")
def build_class_conditioned_spatial_gated_fusion_classifier(**kwargs):
    return ClassConditionedSpatialGatedFusionClassifier(**kwargs)

@MODELS.register("cross_attention_fusion_classifier")
def build_cross_attention_fusion_classifier(**kwargs):
    return CrossAttentionFusionClassifier(**kwargs)


__all__ = [
    "MODELS",
    "MultiModalClassifier",
    "GatedFusionClassifier",
    "DisentangledGatedFusionClassifier",
    "ContextGraphFusionClassifier",
    "DisentangledSpatialGraphFusionClassifier",
    "ClassConditionedSpatialGatedFusionClassifier",
    "CrossAttentionFusionClassifier",
]
