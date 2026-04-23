"""Route B+C: relation disentanglement + spatial context graph, both gated."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DisentangledSpatialGraphFusionClassifier(nn.Module):

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
        relation_modality: str = "grouprec",
        appearance_modality: str = "convnext",
        spatial_modality: str = "bbox_ctx",
        image_id_modality: str = "img_uid",
        graph_k: int = 8,
        graph_alpha: float = 0.5,
        graph_dropout: float = 0.1,
        disentangle_ortho_weight: float = 0.02,
        disentangle_align_weight: float = 0.01,
        align_detach_appearance: bool = True,
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

        self.relation_modality = relation_modality
        self.appearance_modality = appearance_modality
        self.spatial_modality = spatial_modality
        self.image_id_modality = image_id_modality
        self.graph_k = graph_k
        self.graph_alpha = graph_alpha

        self.disentangle_ortho_weight = disentangle_ortho_weight
        self.disentangle_align_weight = disentangle_align_weight
        self.align_detach_appearance = align_detach_appearance

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

        self.relation_private_head = nn.Linear(hidden_dim, hidden_dim)
        self.relation_shared_head = nn.Linear(hidden_dim, hidden_dim)

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
        spatial_ctx: torch.Tensor | None,
        image_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        batch_size = fused_features.size(0)
        if batch_size < 2 or self.graph_k <= 0:
            return fused_features

        k = min(self.graph_k, batch_size - 1)

        if spatial_ctx is not None and spatial_ctx.dim() == 2 and spatial_ctx.size(1) >= 2:
            coords = spatial_ctx[:, :2]
            sim = -torch.cdist(coords, coords, p=2)
        else:
            normed = F.normalize(fused_features, dim=1)
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

        neigh_feats = fused_features[neigh_idx]
        agg = (neigh_feats * neigh_w.unsqueeze(-1)).sum(dim=1)
        return fused_features + self.graph_alpha * self.graph_update(agg)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        expected_dim = sum(self.input_dims)
        if x.shape[1] != expected_dim:
            raise ValueError(f"Input feature dim {x.shape[1]} does not match expected total dim {expected_dim}")

        raw_chunks: dict[str, torch.Tensor] = {}
        for modality in self.modalities:
            start, end = self.slice_indices[modality]
            raw_chunks[modality] = x[:, start:end]

        projected: dict[str, torch.Tensor] = {}
        for modality in self.fusion_modalities:
            chunk = self.norms[modality](raw_chunks[modality])
            projected[modality] = self.projections[modality](chunk)

        rel_base = projected[self.relation_modality]
        rel_private = self.relation_private_head(rel_base)
        rel_shared = self.relation_shared_head(rel_base)
        projected[self.relation_modality] = rel_private

        fusion_features = [projected[m] for m in self.fusion_modalities]

        concat_features = torch.cat(fusion_features, dim=1)
        gate_hidden = self.gate_act(self.gate_fc1(concat_features))
        gate_hidden = self.gate_drop(gate_hidden)
        gate_logits = self.gate_fc2(gate_hidden)
        gate_probs = F.softmax(gate_logits / max(self.gate_temperature, 1e-6), dim=1)

        entropy = -(gate_probs * torch.log(gate_probs + 1e-8)).sum(dim=1)
        entropy_loss = entropy.mean() * self.entropy_weight

        gated_features = []
        for i, feat in enumerate(fusion_features):
            gated_features.append(feat * gate_probs[:, i : i + 1])
        fused_features = torch.cat(gated_features, dim=1)

        context_features = self._context_aggregate(
            fused_features,
            spatial_ctx=raw_chunks.get(self.spatial_modality),
            image_ids=raw_chunks.get(self.image_id_modality),
        )
        logits = self.classifier(context_features)

        rel_private_norm = F.normalize(rel_private, dim=1)
        rel_shared_norm = F.normalize(rel_shared, dim=1)
        ortho = (rel_private_norm * rel_shared_norm).sum(dim=1).abs().mean()

        app_feat = projected[self.appearance_modality]
        if self.align_detach_appearance:
            app_feat = app_feat.detach()
        align = F.mse_loss(rel_shared, app_feat)
        disentangle_loss = self.disentangle_ortho_weight * ortho + self.disentangle_align_weight * align

        return logits, entropy_loss + disentangle_loss


__all__ = ["DisentangledSpatialGraphFusionClassifier"]
