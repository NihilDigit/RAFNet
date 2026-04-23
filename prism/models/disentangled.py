"""Route B: disentangled gated fusion — split relation into shared/private, fuse private only."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DisentangledGatedFusionClassifier(nn.Module):

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
        disentangle_ortho_weight: float = 0.01,
        disentangle_align_weight: float = 0.01,
        align_detach_appearance: bool = True,
    ) -> None:
        super().__init__()

        if len(modalities) != len(input_dims):
            raise ValueError(
                f"modalities and input_dims have different lengths: {len(modalities)} != {len(input_dims)}"
            )
        if relation_modality not in modalities:
            raise ValueError(f"relation_modality={relation_modality} is not in modalities={modalities}")
        if appearance_modality not in modalities:
            raise ValueError(f"appearance_modality={appearance_modality} is not in modalities={modalities}")

        self.modalities = modalities
        self.input_dims = input_dims
        self.hidden_dim = hidden_dim
        self.num_modalities = len(modalities)
        self.entropy_weight = entropy_weight
        self.gate_temperature = gate_temperature

        self.relation_modality = relation_modality
        self.appearance_modality = appearance_modality
        self.disentangle_ortho_weight = disentangle_ortho_weight
        self.disentangle_align_weight = disentangle_align_weight
        self.align_detach_appearance = align_detach_appearance

        self.slice_indices = []
        offset = 0
        for dim in self.input_dims:
            self.slice_indices.append((offset, offset + dim))
            offset += dim

        self.norms = nn.ModuleDict()
        self.projections = nn.ModuleDict()
        for modality, dim in zip(self.modalities, self.input_dims):
            self.norms[modality] = nn.LayerNorm(dim)
            self.projections[modality] = nn.Linear(dim, hidden_dim)

        self.relation_private_head = nn.Linear(hidden_dim, hidden_dim)
        self.relation_shared_head = nn.Linear(hidden_dim, hidden_dim)

        gate_input_dim = hidden_dim * self.num_modalities
        self.gate_fc1 = nn.Linear(gate_input_dim, gate_hidden_dim)
        self.gate_act = nn.ReLU(inplace=True)
        self.gate_drop = nn.Dropout(gate_dropout)
        self.gate_fc2 = nn.Linear(gate_hidden_dim, self.num_modalities)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * self.num_modalities, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        expected_dim = self.slice_indices[-1][1]
        if x.shape[1] != expected_dim:
            raise ValueError(f"Input feature dim {x.shape[1]} does not match expected total dim {expected_dim}")

        projected_features: dict[str, torch.Tensor] = {}
        for modality, (start, end) in zip(self.modalities, self.slice_indices):
            chunk = x[:, start:end]
            chunk = self.norms[modality](chunk)
            chunk = self.projections[modality](chunk)
            projected_features[modality] = chunk

        rel_base = projected_features[self.relation_modality]
        rel_private = self.relation_private_head(rel_base)
        rel_shared = self.relation_shared_head(rel_base)

        fusion_features: list[torch.Tensor] = []
        for modality in self.modalities:
            if modality == self.relation_modality:
                fusion_features.append(rel_private)
            else:
                fusion_features.append(projected_features[modality])

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

        logits = self.classifier(fused_features)

        rel_private_norm = F.normalize(rel_private, dim=1)
        rel_shared_norm = F.normalize(rel_shared, dim=1)
        ortho = (rel_private_norm * rel_shared_norm).sum(dim=1).abs().mean()

        app_feat = projected_features[self.appearance_modality]
        if self.align_detach_appearance:
            app_feat = app_feat.detach()
        align = F.mse_loss(rel_shared, app_feat)

        disentangle_loss = self.disentangle_ortho_weight * ortho + self.disentangle_align_weight * align
        aux_loss = entropy_loss + disentangle_loss

        return logits, aux_loss


__all__ = ["DisentangledGatedFusionClassifier"]
