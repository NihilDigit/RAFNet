from __future__ import annotations

import torch
import torch.nn as nn


class CrossAttentionFusionClassifier(nn.Module):
    """Cross-attention fusion baseline for two modalities."""

    def __init__(
        self,
        modalities: list[str],
        input_dims: list[int],
        d_model: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1,
        num_classes: int = 5,
        ffn_ratio: int = 2,
    ) -> None:
        super().__init__()
        if len(modalities) != 2 or len(input_dims) != 2:
            raise ValueError("CrossAttentionFusionClassifier expects exactly 2 modalities")
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")

        self.modalities = modalities
        self.input_dims = input_dims
        self.d_model = d_model

        self.norm_a = nn.LayerNorm(input_dims[0])
        self.norm_b = nn.LayerNorm(input_dims[1])
        self.proj_a = nn.Linear(input_dims[0], d_model)
        self.proj_b = nn.Linear(input_dims[1], d_model)

        self.attn_a_to_b = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_b_to_a = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        hidden = d_model * ffn_ratio
        self.ffn_a = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
        )
        self.ffn_b = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dim_a, dim_b = self.input_dims
        if x.shape[1] != dim_a + dim_b:
            raise ValueError(f"Expected input dim {dim_a + dim_b}, got {x.shape[1]}")

        feat_a = self.proj_a(self.norm_a(x[:, :dim_a])).unsqueeze(1)
        feat_b = self.proj_b(self.norm_b(x[:, dim_a:])).unsqueeze(1)

        a_delta, _ = self.attn_a_to_b(query=feat_a, key=feat_b, value=feat_b, need_weights=False)
        b_delta, _ = self.attn_b_to_a(query=feat_b, key=feat_a, value=feat_a, need_weights=False)

        feat_a = feat_a + a_delta
        feat_b = feat_b + b_delta

        feat_a = feat_a + self.ffn_a(feat_a)
        feat_b = feat_b + self.ffn_b(feat_b)

        fused = torch.cat([feat_a.squeeze(1), feat_b.squeeze(1)], dim=1)
        logits = self.classifier(fused)
        return logits

