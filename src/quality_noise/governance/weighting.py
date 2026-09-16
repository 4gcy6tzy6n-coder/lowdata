"""Soft governance: continuous training weights and weighted CE (Phase 7).

The training weight is the estimated clean probability w_i = 1 - eta_hat_i.
The fixed margin-threshold action rules (protect / normal / quarantine) are
replaced by the posterior itself; an interpretation layer may map eta to
actions for reporting, but the training algorithm is the continuous weight.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def soft_weights(eta_shrunk: np.ndarray) -> np.ndarray:
    """w_i = 1 - eta_hat_i, clipped to [0, 1]."""
    return np.clip(1.0 - np.asarray(eta_shrunk, dtype=np.float64), 0.0, 1.0).astype(np.float32)


def calibrated_soft_weights(eta: np.ndarray, noise_rate: float) -> np.ndarray:
    """Noise-rate-calibrated soft weights.

    The diffuse posterior from an unsupervised mixture barely downweights the
    noisy tail (w ~ 0.7-0.9), so accuracy stays near CE. Instead, rank the
    samples by noise posterior and smoothly downweight the top ``noise_rate``
    fraction to ~0 while keeping the rest at ~1 — a soft analogue of small-loss
    selection that stays continuous (no hard threshold). ``noise_rate`` is a
    known experimental hyperparameter (the injected noise rate).

        w_i = clip((1 - rank_i) / noise_rate, 0, 1)
    """
    eta = np.asarray(eta, dtype=np.float64)
    ranks = np.argsort(np.argsort(eta))
    r = ranks / max(len(ranks) - 1, 1)  # [0, 1], higher eta -> higher r
    rate = max(float(noise_rate), 1e-3)
    return np.clip((1.0 - r) / rate, 0.0, 1.0).astype(np.float32)


def action_from_posterior(eta: np.ndarray, tau1: float = 0.3, tau2: float = 0.7) -> np.ndarray:
    """Interpretation layer (reporting only): retain / reweight / exclude."""
    eta = np.asarray(eta)
    action = np.full(len(eta), "reweight", dtype=object)
    action[eta < tau1] = "retain"
    action[eta >= tau2] = "exclude"
    return action


def weighted_ce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Weighted cross-entropy: sum_i w_i * l_i / sum_i w_i.

    With uniform weights this equals the standard mean CE.
    """
    logits = logits.float()
    per_sample = F.cross_entropy(logits, targets, reduction="none")
    weights = weights.float().to(logits.device)
    denom = weights.sum().clamp(min=1e-8)
    return (per_sample * weights).sum() / denom
