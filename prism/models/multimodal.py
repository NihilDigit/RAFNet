from __future__ import annotations

import torch
import torch.nn as nn


class MultiModalClassifier(nn.Module):
    """MLP fusion classifier over one or more modalities."""

    def __init__(
        self,
        grouprec_dim: int = 1024,
        mobilenet_dim: int = 960,
        hidden_dim: int = 256,
        num_classes: int = 5,
        dropout: float = 0.5,
        modalities: list[str] | None = None,
        input_dims: list[int] | None = None,
    ) -> None:
        super().__init__()

        if modalities is None or input_dims is None:
            modalities = ["grouprec", "mobilenet"]
            input_dims = [grouprec_dim, mobilenet_dim]

        if len(modalities) != len(input_dims):
            raise ValueError("modalities and input_dims have different lengths")

        self.modalities = modalities
        self.input_dims = input_dims
        self.hidden_dim = hidden_dim

        self.norms = nn.ModuleDict()
        self.projections = nn.ModuleDict()
        for modality, dim in zip(self.modalities, self.input_dims):
            self.norms[modality] = nn.LayerNorm(dim)
            self.projections[modality] = nn.Linear(dim, hidden_dim)

        self.slice_indices = []
        offset = 0
        for dim in self.input_dims:
            self.slice_indices.append((offset, offset + dim))
            offset += dim

        fusion_input_dim = hidden_dim * len(self.modalities)
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != self.slice_indices[-1][1]:
            raise ValueError(
                f"Input feature dim {x.shape[1]} does not match expected total dim {self.slice_indices[-1][1]}"
            )

        fused_features = []
        for modality, (start, end) in zip(self.modalities, self.slice_indices):
            chunk = x[:, start:end]
            chunk = self.norms[modality](chunk)
            chunk = self.projections[modality](chunk)
            fused_features.append(chunk)

        fused = (
            torch.cat(fused_features, dim=1)
            if len(fused_features) > 1
            else fused_features[0]
        )

        return self.fusion(fused)
