"""Model construction. Default: ResNet-18 from torchvision with a replaced FC
head. Pretrained encoders are reserved for the representation ablation (Phase 9)
and are NOT used in the noisy-label training path by default.
"""
from __future__ import annotations

import numpy as np
import torch.nn as nn

ARCHS = {
    "resnet18": ("resnet18", 512),
    "resnet34": ("resnet34", 512),
    "resnet50": ("resnet50", 2048),
}


def build_model(cfg_model: dict, num_classes: int, pretrained: bool | None = None) -> nn.Module:
    """Build the model. ``pretrained`` defaults to the config value.

    ``arch="mlp"`` builds a small fully-connected net (used for CPU smoke tests);
    the input dimension is derived from ``cfg_dataset`` if present.
    """
    arch = cfg_model.get("arch", "resnet18")
    if arch == "mlp":
        cfg_dataset = cfg_model.get("_dataset_cfg", {})
        size = cfg_dataset.get("input_size", [3, 32, 32])
        input_dim = int(np.prod(size))
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    import torchvision

    if arch not in ARCHS:
        raise ValueError(f"Unsupported arch: {arch}")
    fn_name, _feat_dim = ARCHS[arch]
    use_pretrained = cfg_model.get("pretrained", False) if pretrained is None else pretrained
    fn = getattr(torchvision.models, fn_name)
    model = fn(weights=torchvision.models.ResNet18_Weights.DEFAULT if use_pretrained else None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def penultimate_feature_dim(model: nn.Module) -> int:
    """Return the penultimate (pre-FC) feature dimensionality."""
    return getattr(model.fc, "in_features", None) or 512
