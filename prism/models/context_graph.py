"""Route C: gated fusion + one-hop kNN context graph over fused features.

If `bbox_ctx` (normalized cx, cy, w, h) is present, spatial distance is used
to pick neighbors; otherwise we fall back to feature cosine similarity.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContextGraphFusionClassifier(nn.Module):

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
        include_spatial_in_fusion: bool = False,
        image_id_modality: str = "img_uid",
        include_image_id_in_fusion: bool = False,
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
        self.include_spatial_in_fusion = include_spatial_in_fusion
        self.image_id_modality = image_id_modality
        self.include_image_id_in_fusion = include_image_id_in_fusion

        self.slice_indices: dict[str, tuple[int, int]] = {}
        offset = 0
        for modality, dim in zip(self.modalities, self.input_dims):
            self.slice_indices[modality] = (offset, offset + dim)
            offset += dim

        self.fusion_modalities = [
            m
            for m in self.modalities
            if (
                (m != self.spatial_modality or self.include_spatial_in_fusion)
                and (m != self.image_id_modality or self.include_image_id_in_fusion)
            )
        ]
        self.num_modalities = len(self.fusion_modalities)

        self.norms = nn.ModuleDict()
        self.projections = nn.ModuleDict()
        dim_map = dict(zip(self.modalities, self.input_dims))
        for modality in self.fusion_modalities:
            dim = dim_map[modality]
            self.norms[modality] = nn.LayerNorm(dim)
            self.projections[modality] = nn.Linear(dim, hidden_dim)

        gate_input_dim = hidden_dim * self.num_modalities
        self.gate_fc1 = nn.Linear(gate_input_dim, gate_hidden_dim)
        self.gate_act = nn.ReLU(inplace=True)
        self.gate_drop = nn.Dropout(gate_dropout)
        self.gate_fc2 = nn.Linear(gate_hidden_dim, self.num_modalities)

        fused_dim = hidden_dim * self.num_modalities
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

    def _context_aggregate(
        self,
        fused_features: torch.Tensor,
        spatial_ctx: torch.Tensor | None = None,
        image_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size = fused_features.size(0)
        if batch_size < 2 or self.graph_k <= 0:
            return fused_features

        k = min(self.graph_k, batch_size - 1)
        use_spatial = spatial_ctx is not None and spatial_ctx.dim() == 2 and spatial_ctx.size(1) >= 2
        if use_spatial:
            coords = spatial_ctx[:, :2]
            dist = torch.cdist(coords, coords, p=2)
            sim = -dist
        else:
            normed = F.normalize(fused_features, dim=1)
            sim = normed @ normed.t()

        # Image-identity masking: concentrate aggregation on same-image peers when they share the mini-batch.
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

        neigh_feats = fused_features[neigh_idx]
        agg = (neigh_feats * neigh_w.unsqueeze(-1)).sum(dim=1)

        updated = fused_features + self.graph_alpha * self.graph_update(agg)
        return updated

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
            chunk = raw_chunks[modality]
            chunk = self.norms[modality](chunk)
            chunk = self.projections[modality](chunk)
            projected_features.append(chunk)

        concat_features = torch.cat(projected_features, dim=1)
        gate_hidden = self.gate_act(self.gate_fc1(concat_features))
        gate_hidden = self.gate_drop(gate_hidden)
        gate_logits = self.gate_fc2(gate_hidden)
        gate_probs = F.softmax(gate_logits / max(self.gate_temperature, 1e-6), dim=1)

        entropy = -(gate_probs * torch.log(gate_probs + 1e-8)).sum(dim=1)
        entropy_loss = entropy.mean() * self.entropy_weight

        gated_features = []
        for i, feat in enumerate(projected_features):
            gated_features.append(feat * gate_probs[:, i : i + 1])
        fused_features = torch.cat(gated_features, dim=1)

        spatial_ctx = raw_chunks.get(self.spatial_modality)
        image_ids = raw_chunks.get(self.image_id_modality)
        context_features = self._context_aggregate(
            fused_features, spatial_ctx=spatial_ctx, image_ids=image_ids
        )
        logits = self.classifier(context_features)
        return logits, entropy_loss


__all__ = ["ContextGraphFusionClassifier"]
