"""Per-sample training dynamics computed from a (n_samples, n_epochs) trace.

All functions are pure: they take a 2D array indexed [sample, epoch] and return
a per-sample 1D array. They operate on the trace data only — never on the
noise mask or clean labels.
"""
from __future__ import annotations

import numpy as np


def compute_ema_loss(loss: np.ndarray, decay: float = 0.9) -> np.ndarray:
    """Exponential moving average of per-sample loss over epochs."""
    loss = np.asarray(loss, dtype=np.float64)
    out = np.zeros(loss.shape[0], dtype=np.float64)
    ema = np.zeros(loss.shape[0], dtype=np.float64)
    for t in range(loss.shape[1]):
        ema = decay * ema + (1.0 - decay) * loss[:, t]
        out = ema.copy()
    return out.astype(np.float32)


def compute_loss_variance(loss: np.ndarray) -> np.ndarray:
    """Per-sample variance of loss across epochs."""
    return np.asarray(loss, dtype=np.float64).var(axis=1).astype(np.float32)


def compute_prediction_consistency(preds: np.ndarray) -> np.ndarray:
    """Fraction of epochs where the prediction equals the per-sample mode."""
    preds = np.asarray(preds)
    modal = np.apply_along_axis(lambda row: np.bincount(row).argmax(), axis=1, arr=preds)
    return (preds == modal[:, None]).mean(axis=1).astype(np.float32)


def compute_forgetting_count(correct: np.ndarray) -> np.ndarray:
    """Per-sample count of forgetting events (Toneva et al.).

    A forgetting event occurs at epoch t when the sample is classified
    incorrectly at t but was classified correctly at some earlier epoch.
    """
    correct = np.asarray(correct, dtype=bool)
    ever_correct = np.zeros(correct.shape[0], dtype=bool)
    forgets = np.zeros(correct.shape[0], dtype=np.int64)
    for t in range(correct.shape[1]):
        now_wrong = ~correct[:, t]
        forgets += ever_correct & now_wrong
        ever_correct |= correct[:, t]
    return forgets.astype(np.int64)


def compute_confidence_variance(conf: np.ndarray) -> np.ndarray:
    """Per-sample variance of confidence across epochs."""
    return np.asarray(conf, dtype=np.float64).var(axis=1).astype(np.float32)


def compute_margin_trend(margin: np.ndarray) -> np.ndarray:
    """OLS slope of margin over epochs (per-sample trend)."""
    margin = np.asarray(margin, dtype=np.float64)
    t = np.arange(margin.shape[1], dtype=np.float64)
    t_centered = t - t.mean()
    denom = (t_centered**2).sum()
    slope = ((margin - margin.mean(axis=1, keepdims=True)) * t_centered[None, :]).sum(axis=1) / denom
    return slope.astype(np.float32)


def compute_all_dynamics(
    loss: np.ndarray,
    conf: np.ndarray,
    margin: np.ndarray,
    preds: np.ndarray,
    correct: np.ndarray,
    sample_ids: np.ndarray,
    ema_decay: float = 0.9,
) -> dict[str, np.ndarray]:
    """Compute the full dynamics set aligned to ``sample_ids``."""
    return {
        "sample_id": np.asarray(sample_ids, dtype=np.int64),
        "ema_loss": compute_ema_loss(loss, decay=ema_decay),
        "loss_variance": compute_loss_variance(loss),
        "prediction_consistency": compute_prediction_consistency(preds),
        "forgetting_count": compute_forgetting_count(correct),
        "confidence_variance": compute_confidence_variance(conf),
        "margin_trend": compute_margin_trend(margin),
    }
