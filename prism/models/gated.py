from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedFusionClassifier(nn.Module):
    """Multimodal gated fusion classifier with entropy regularization.

    Per-modality LayerNorm+projection, a softmax gate over modalities, gated
    weighting, concat, MLP head. Forward returns (logits, entropy_loss).
    """

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
        class_aware_gate: bool = False,
        class_prior_detach: bool = True,
        uncertainty_aware: bool = False,
        uncertainty_floor: float = 1e-4,
    ) -> None:
        super().__init__()

        if len(modalities) != len(input_dims):
            raise ValueError(
                f"modalities and input_dims have different lengths: {len(modalities)} != {len(input_dims)}"
            )

        self.modalities = modalities
        self.input_dims = input_dims
        self.hidden_dim = hidden_dim
        self.num_modalities = len(modalities)
        self.entropy_weight = entropy_weight
        self.gate_temperature = gate_temperature
        self.class_aware_gate = class_aware_gate
        self.class_prior_detach = class_prior_detach
        self.uncertainty_aware = uncertainty_aware
        self.uncertainty_floor = uncertainty_floor

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

        gate_input_dim = hidden_dim * self.num_modalities
        if self.class_aware_gate:
            self.class_prior_head = nn.Linear(gate_input_dim, num_classes)
            gate_input_dim += num_classes
        self.gate_fc1 = nn.Linear(gate_input_dim, gate_hidden_dim)
        self.gate_act = nn.ReLU(inplace=True)
        self.gate_drop = nn.Dropout(gate_dropout)
        self.gate_fc2 = nn.Linear(gate_hidden_dim, self.num_modalities)

        if self.uncertainty_aware:
            self.uncertainty_heads = nn.ModuleDict()
            for modality in self.modalities:
                self.uncertainty_heads[modality] = nn.Linear(hidden_dim, 1)

        classifier_input_dim = hidden_dim * self.num_modalities
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.shape[1] != self.slice_indices[-1][1]:
            raise ValueError(
                f"Input feature dim {x.shape[1]} does not match expected total dim "
                f"{self.slice_indices[-1][1]}"
            )

        batch_size = x.size(0)

        projected_features = []
        for modality, (start, end) in zip(self.modalities, self.slice_indices):
            chunk = x[:, start:end]
            chunk = self.norms[modality](chunk)
            chunk = self.projections[modality](chunk)
            projected_features.append(chunk)

        concat_features = torch.cat(projected_features, dim=1)

        gate_input = concat_features
        if self.class_aware_gate:
            class_prior_logits = self.class_prior_head(concat_features)
            class_prior = F.softmax(class_prior_logits, dim=1)
            if self.class_prior_detach:
                class_prior = class_prior.detach()
            gate_input = torch.cat([concat_features, class_prior], dim=1)

        gate_hidden = self.gate_act(self.gate_fc1(gate_input))
        gate_hidden = self.gate_drop(gate_hidden)
        gate_logits = self.gate_fc2(gate_hidden)
        gate_probs = F.softmax(gate_logits / max(self.gate_temperature, 1e-6), dim=1)

        if self.uncertainty_aware:
            precisions = []
            for modality, feat in zip(self.modalities, projected_features):
                log_var = self.uncertainty_heads[modality](feat).squeeze(1)
                precision = torch.exp(-log_var).clamp(min=self.uncertainty_floor)
                precisions.append(precision)
            precision_stack = torch.stack(precisions, dim=1)
            gate_probs = gate_probs * precision_stack
            gate_probs = gate_probs / (gate_probs.sum(dim=1, keepdim=True) + 1e-8)

        entropy = -(gate_probs * torch.log(gate_probs + 1e-8)).sum(dim=1)
        entropy_loss = entropy.mean() * self.entropy_weight

        gated_features = []
        for i, feat in enumerate(projected_features):
            gate_value = gate_probs[:, i : i + 1]
            gated_feat = feat * gate_value
            gated_features.append(gated_feat)

        fused_features = torch.cat(gated_features, dim=1)
        logits = self.classifier(fused_features)

        return logits, entropy_loss


if __name__ == "__main__":
    print("=" * 60)
    print("GatedFusionClassifier self-test")
    print("=" * 60)

    modalities = ["grouprec", "resnet"]
    input_dims = [1024, 2048]
    batch_size = 8

    model = GatedFusionClassifier(
        modalities=modalities,
        input_dims=input_dims,
        hidden_dim=256,
        num_classes=5,
        dropout=0.3,
        gate_dropout=0.1,
        entropy_weight=0.01,
    )

    print(f"\n[OK] Model built")
    print(f"  - Modalities: {modalities}")
    print(f"  - Input dims: {input_dims}")
    print(f"  - Total params: {sum(p.numel() for p in model.parameters()):,}")

    total_dim = sum(input_dims)
    x = torch.randn(batch_size, total_dim)
    logits, entropy_loss = model(x)

    print(f"\n[OK] Forward pass")
    print(f"  - Input shape: {x.shape}")
    print(f"  - Output logits shape: {logits.shape}")
    print(f"  - Expected shape: ({batch_size}, 5)")
    print(f"  - Entropy loss: {entropy_loss.item():.6f}")

    assert logits.shape == (
        batch_size,
        5,
    ), f"Shape mismatch: {logits.shape} != ({batch_size}, 5)"
    assert entropy_loss.dim() == 0, "Entropy loss should be a scalar"

    assert not torch.isnan(logits).any(), "logits contains NaN"
    assert not torch.isinf(logits).any(), "logits contains Inf"
    assert not torch.isnan(entropy_loss), "entropy_loss is NaN"
    assert entropy_loss.item() >= 0, "Entropy loss should be non-negative"

    print(f"\n[OK] Numeric checks passed")
    print(f"  - logits range: [{logits.min():.3f}, {logits.max():.3f}]")
    print(f"  - No NaN/Inf values")

    print(f"\n[Variable modality count test]")
    for test_modalities, test_dims in [
        (["grouprec"], [1024]),
        (["grouprec", "resnet", "vit"], [1024, 2048, 768]),
    ]:
        model_test = GatedFusionClassifier(
            modalities=test_modalities, input_dims=test_dims, hidden_dim=128
        )
        x_test = torch.randn(4, sum(test_dims))
        logits_test, ent_test = model_test(x_test)
        assert logits_test.shape == (4, 5)
        print(f"  [OK] {len(test_modalities)} modalities: output shape {logits_test.shape}")

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    targets = torch.randint(0, 5, (batch_size,))

    logits, entropy_loss = model(x)
    ce_loss = F.cross_entropy(logits, targets)
    total_loss = ce_loss + entropy_loss

    total_loss.backward()
    optimizer.step()

    print(f"\n[OK] Backpropagation succeeded")
    print(f"  - CE Loss: {ce_loss.item():.4f}")
    print(f"  - Entropy Loss: {entropy_loss.item():.6f}")
    print(f"  - Total Loss: {total_loss.item():.4f}")

    print(f"\n{'=' * 60}")
    print("All checks passed. GatedFusionClassifier is working correctly.")
    print("=" * 60)
