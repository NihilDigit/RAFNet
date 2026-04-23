"""CCSG: class-conditioned + spatial-conditioned gated fusion (extends Route C+Spatial)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassConditionedSpatialGatedFusionClassifier(nn.Module):

    def __init__(
        self,
        modalities: list[str],
        input_dims: list[int],
        hidden_dim: int = 256,
        num_classes: int = 5,
        dropout: float = 0.3,
        gate_dropout: float = 0.1,
        entropy_weight: float = 0.01,
        gate_temperature: float = 1.0,
        gate_hidden_dim: int = 128,
        graph_k: int = 8,
        graph_alpha: float = 0.5,
        graph_dropout: float = 0.1,
        spatial_modality: str = "bbox_ctx",
        image_id_modality: str = "img_uid",
        class_prior_detach: bool = True,
        context_detach: bool = True,
    ) -> None:
        super().__init__()

        if len(modalities) != len(input_dims):
            raise ValueError(
                f"modalities and input_dims have different lengths: {len(modalities)} != {len(input_dims)}"
            )

        self.modalities = modalities
        self.input_dims = input_dims
        self.hidden_dim = hidden_dim
        self.entropy_weight = entropy_weight
        self.gate_temperature = gate_temperature

        self.graph_k = graph_k
        self.graph_alpha = graph_alpha
        self.spatial_modality = spatial_modality
        self.image_id_modality = image_id_modality
        self.class_prior_detach = class_prior_detach
        self.context_detach = context_detach

        self.slice_indices: dict[str, tuple[int, int]] = {}
        offset = 0
        for modality, dim in zip(self.modalities, self.input_dims):
            self.slice_indices[modality] = (offset, offset + dim)
            offset += dim

        self.fusion_modalities = [
            m for m in self.modalities if m not in {self.spatial_modality, self.image_id_modality}
        ]
        self.num_modalities = len(self.fusion_modalities)

        dim_map = dict(zip(self.modalities, self.input_dims))
        self.norms = nn.ModuleDict()
        self.projections = nn.ModuleDict()
        for modality in self.fusion_modalities:
            dim = dim_map[modality]
            self.norms[modality] = nn.LayerNorm(dim)
            self.projections[modality] = nn.Linear(dim, hidden_dim)

        fused_dim = hidden_dim * self.num_modalities

        self.class_prior_head = nn.Linear(fused_dim, num_classes)

        self.context_token_head = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(graph_dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Gate input: fused features || class prior || spatial context token
        gate_input_dim = fused_dim + num_classes + hidden_dim
        self.gate_fc1 = nn.Linear(gate_input_dim, gate_hidden_dim)
        self.gate_act = nn.ReLU(inplace=True)
        self.gate_drop = nn.Dropout(gate_dropout)
        self.gate_fc2 = nn.Linear(gate_hidden_dim, self.num_modalities)

        self.graph_update = nn.Sequential(
            nn.Linear(fused_dim, fused_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(graph_dropout),
            nn.Linear(fused_dim, fused_dim),
        )

        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def _knn_aggregate(
        self,
        features: torch.Tensor,
        spatial_ctx: torch.Tensor | None,
        image_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        batch_size = features.size(0)
        if batch_size < 2 or self.graph_k <= 0:
            return features

        k = min(self.graph_k, batch_size - 1)

        if spatial_ctx is not None and spatial_ctx.dim() == 2 and spatial_ctx.size(1) >= 2:
            coords = spatial_ctx[:, :2]
            sim = -torch.cdist(coords, coords, p=2)
        else:
            normed = F.normalize(features, dim=1)
            sim = normed @ normed.t()

        if image_ids is not None and image_ids.dim() == 2 and image_ids.size(1) >= 1:
            img = image_ids[:, 0]
            same_img = img.unsqueeze(0) == img.unsqueeze(1)
            same_img.fill_diagonal_(False)
            if same_img.any():
                sim_same = sim.masked_fill(~same_img, -1e9)
                has_neighbor = same_img.any(dim=1)
                sim = torch.where(has_neighbor.unsqueeze(1), sim_same, sim)

        sim.fill_diagonal_(-1e9)
        neigh_vals, neigh_idx = torch.topk(sim, k=k, dim=1)
        neigh_w = F.softmax(neigh_vals, dim=1)

        neigh_feats = features[neigh_idx]
        agg = (neigh_feats * neigh_w.unsqueeze(-1)).sum(dim=1)
        return agg

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        expected_dim = sum(self.input_dims)
        if x.shape[1] != expected_dim:
            raise ValueError(f"Input feature dim {x.shape[1]} does not match expected total dim {expected_dim}")

        raw_chunks: dict[str, torch.Tensor] = {}
        for modality in self.modalities:
            start, end = self.slice_indices[modality]
            raw_chunks[modality] = x[:, start:end]

        projected_features = []
        for modality in self.fusion_modalities:
            chunk = self.norms[modality](raw_chunks[modality])
            chunk = self.projections[modality](chunk)
            projected_features.append(chunk)

        base_fused = torch.cat(projected_features, dim=1)

        class_prior_logits = self.class_prior_head(base_fused)
        class_prior = F.softmax(class_prior_logits, dim=1)
        if self.class_prior_detach:
            class_prior = class_prior.detach()

        neigh_agg = self._knn_aggregate(
            base_fused,
            spatial_ctx=raw_chunks.get(self.spatial_modality),
            image_ids=raw_chunks.get(self.image_id_modality),
        )
        spatial_token = self.context_token_head(neigh_agg)
        if self.context_detach:
            spatial_token = spatial_token.detach()

        gate_input = torch.cat([base_fused, class_prior, spatial_token], dim=1)

        gate_hidden = self.gate_act(self.gate_fc1(gate_input))
        gate_hidden = self.gate_drop(gate_hidden)
        gate_logits = self.gate_fc2(gate_hidden)
        gate_probs = F.softmax(gate_logits / max(self.gate_temperature, 1e-6), dim=1)

        entropy = -(gate_probs * torch.log(gate_probs + 1e-8)).sum(dim=1)
        entropy_loss = entropy.mean() * self.entropy_weight

        gated_features = []
        for i, feat in enumerate(projected_features):
            gated_features.append(feat * gate_probs[:, i : i + 1])
        fused_features = torch.cat(gated_features, dim=1)

        spatial_update = self._knn_aggregate(
            fused_features,
            spatial_ctx=raw_chunks.get(self.spatial_modality),
            image_ids=raw_chunks.get(self.image_id_modality),
        )
        fused_features = fused_features + self.graph_alpha * self.graph_update(spatial_update)

        logits = self.classifier(fused_features)
        return logits, entropy_loss


__all__ = ["ClassConditionedSpatialGatedFusionClassifier"]
