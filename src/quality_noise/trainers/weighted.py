"""Reliability-aware weighted trainer: soft-weighted CE over observed labels.

A drop-in replacement for the standard trainer that uses per-sample weights
w_i = 1 - eta_hat_i (see governance/weighting.py). Everything else — optimizer,
scheduler, seeding, trace callbacks — is identical so weighted vs standard
results stay comparable.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset

from .standard import History, train_standard


def train_weighted(
    cfg: dict,
    model: nn.Module,
    train_view: Dataset,
    weights: dict[int, float] | np.ndarray,
    device,
    callbacks: list | None = None,
    progress: bool = False,
) -> History:
    """Train with soft weights. ``weights`` may be a {sample_id: w} dict or an
    array aligned to ``train_view.sample_ids``."""
    if isinstance(weights, np.ndarray):
        weights = {int(sid): float(w) for sid, w in zip(train_view.sample_ids, weights)}
    return train_standard(
        cfg,
        model,
        train_view,
        device,
        callbacks=callbacks,
        progress=progress,
        sample_weights=weights,
    )
