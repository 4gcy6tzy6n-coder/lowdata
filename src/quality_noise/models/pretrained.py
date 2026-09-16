"""Frozen pretrained encoders for the representation-dependence ablation (Phase 9).

These encoders are NOT part of the noisy-label training path. Their embeddings
are precomputed once and used to answer: does the detectability/quality
conclusion survive a change of representation?
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

WEIGHTS = {
    "resnet18": (lambda: torchvision_resnet18_weights()),
    "resnet50": (lambda: torchvision_resnet50_weights()),
}


def torchvision_resnet18_weights():
    import torchvision

    return torchvision.models.ResNet18_Weights.DEFAULT


def torchvision_resnet50_weights():
    import torchvision

    return torchvision.models.ResNet50_Weights.DEFAULT


class FrozenEncoder(nn.Module):
    """Penultimate-feature extractor with frozen weights."""

    def __init__(self, name: str = "resnet18") -> None:
        super().__init__()
        import torchvision

        if name == "resnet18":
            self.backbone = torchvision.models.resnet18(weights=WEIGHTS["resnet18"]())
            self.dim = 512
        elif name == "resnet50":
            self.backbone = torchvision.models.resnet50(weights=WEIGHTS["resnet50"]())
            self.dim = 2048
        else:
            raise ValueError(f"Unsupported pretrained encoder: {name}")
        self.backbone.fc = nn.Identity()
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.backbone.eval()

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def build_pretrained_encoder(name: str = "resnet18") -> FrozenEncoder:
    return FrozenEncoder(name)


def extract_pretrained_embeddings(
    encoder: FrozenEncoder,
    loader,
    device: torch.device,
    transform,
    sample_ids: np.ndarray | None = None,
) -> np.ndarray:
    """Precompute penultimate embeddings with the frozen encoder.

    ``loader`` yields raw (uint8) tensors; ``transform`` converts them. Returns
    a (N, dim) float32 array in loader order (or sample_id ascending order if
    ``sample_ids`` is given).
    """
    encoder = encoder.to(device)
    encoder.eval()
    out = []
    with torch.no_grad():
        for x, *_rest in loader:
            x = torch.stack([transform(t) for t in x])
            out.append(encoder(x.to(device)).cpu().numpy())
    arr = np.concatenate(out, axis=0).astype(np.float32)
    if sample_ids is not None:
        order = np.argsort(np.asarray(sample_ids, dtype=np.int64))
        arr = arr[order]
    return arr
