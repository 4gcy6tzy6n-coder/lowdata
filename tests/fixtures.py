"""Test fixtures: synthetic datasets, toy models, and planted-array scenarios.

Everything runs in seconds on CPU and never downloads CIFAR.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


@dataclass
class SimpleRaw:
    """Minimal stand-in for a torchvision CIFAR object (CHW uint8 + targets)."""

    data: np.ndarray
    targets: np.ndarray


def make_blob_dataset(
    num_classes: int = 4,
    per_class: int = 50,
    dim: int = 8,
    seed: int = 0,
    sep: float = 2.0,
) -> SimpleRaw:
    """Anisotropic Gaussian blobs in (3, dim, dim) uint8 images with clean labels."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(0, 1, size=(num_classes, 3))
    imgs, labels = [], []
    for c in range(num_classes):
        for _ in range(per_class):
            x = centers[c] + rng.normal(0, 0.5, size=3)
            imgs.append(np.clip(0, 255, (x * 80 + 128)).astype(np.uint8))
            labels.append(c)
    images = np.asarray(imgs, dtype=np.uint8)  # (N, 3)
    images = np.repeat(images[:, :, None, None], dim, axis=2)  # (N,3,dim)
    images = np.repeat(images, dim, axis=3)  # (N,3,dim,dim)
    return SimpleRaw(data=images, targets=np.asarray(labels))


def make_toy_model(num_classes: int = 4, input_dim: int = 3 * 8 * 8) -> nn.Module:
    """Tiny MLP fast enough for CPU tests."""
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(input_dim, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, num_classes),
    )


def toy_cfg(
    num_classes: int = 4,
    noise_type: str = "symmetric",
    rate: float = 0.3,
    seed: int = 0,
    epochs: int = 3,
    batch_size: int = 16,
) -> dict:
    """A minimal resolved-style config for CPU tests."""
    return {
        "dataset": {"name": "blob", "num_classes": num_classes, "input_size": [3, 8, 8], "val_frac": 0.0},
        "noise": {"type": noise_type, "rate": rate, "seed": seed, "num_classes": num_classes},
        "model": {
            "arch": "mlp",
            "pretrained": False,
            "optimizer": {"name": "adam", "lr": 0.01, "weight_decay": 0.0},
            "scheduler": {"name": "none"},
            "epochs": epochs,
            "batch_size": batch_size,
            "num_workers": 0,
        },
        "seed": seed,
    }


def make_score_mask_quality(
    n: int = 500,
    seed: int = 0,
    mode: str = "clean",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Plant AUC scenarios.

    - mode="clean": noisy scores ~ N(0.8, .1), clean ~ N(0.2, .1), quality
      independent of noise -> global AUC ~ 0.998 (well separated).
    - mode="confounded": noisy/clean scores differ ONLY at high quality;
      at low quality the two are indistinguishable -> high global AUC but low
      matched/within-quality AUC.
    Returns (score, mask, quality) where quality in [0,1].
    """
    rng = np.random.default_rng(seed)
    mask = np.zeros(n, dtype=bool)
    n_noisy = n // 2
    mask[rng.permutation(n)[:n_noisy]] = True
    quality = rng.uniform(0, 1, size=n)

    score = np.empty(n, dtype=np.float64)
    if mode == "clean":
        score[~mask] = rng.normal(0.2, 0.1, size=n - n_noisy)
        score[mask] = rng.normal(0.8, 0.1, size=n_noisy)
    elif mode == "confounded":
        hi = quality > 0.7
        score[hi & mask] = rng.normal(0.9, 0.05, size=(hi & mask).sum())
        score[hi & ~mask] = rng.normal(0.1, 0.05, size=(hi & ~mask).sum())
        # Low quality: noise vs clean indistinguishable.
        lo = ~hi
        score[lo] = rng.normal(0.5, 0.05, size=lo.sum())
    else:
        raise ValueError(mode)
    return score, mask, quality
