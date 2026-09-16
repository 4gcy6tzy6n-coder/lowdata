"""Hard-clean preservation metrics (Gate B, dimension 2).

These measure whether soft weighting suppresses *noisy* samples without
damaging *hard-clean* samples — samples that are clean but hard (low quality).
All functions take the evaluation-only mask (or clean indicator) as the last
argument and are used strictly for reporting.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def hard_clean_false_suppression(
    weights: np.ndarray,
    quality: np.ndarray,
    is_clean: np.ndarray,
    hard_quantile: float = 0.1,
    cutoff: float = 0.5,
) -> float:
    """Fraction of clean samples in the hardest quality decile with w < cutoff.

    This is the primary "hard-clean damage" rate: how often the method would
    have dropped a clean-but-ambiguous sample.
    """
    weights = np.asarray(weights, dtype=np.float64)
    quality = np.asarray(quality, dtype=np.float64)
    is_clean = np.asarray(is_clean, dtype=bool)
    if is_clean.sum() == 0:
        return float("nan")
    thr = np.quantile(quality[is_clean], hard_quantile)
    hard_clean = is_clean & (quality <= thr)
    if hard_clean.sum() == 0:
        return float("nan")
    return float(np.mean(weights[hard_clean] < cutoff))


def false_exclusion_rate(weights: np.ndarray, is_clean: np.ndarray, cutoff: float = 0.5) -> float:
    """Fraction of clean samples with weight below the cutoff."""
    weights = np.asarray(weights, dtype=np.float64)
    is_clean = np.asarray(is_clean, dtype=bool)
    if is_clean.sum() == 0:
        return float("nan")
    return float(np.mean(weights[is_clean] < cutoff))


def quality_stratified_fpr(
    weights: np.ndarray,
    is_clean: np.ndarray,
    quality: np.ndarray,
    cutoff: float = 0.5,
    n_bins: int = 5,
) -> pd.DataFrame:
    """Clean-sample false-suppression rate per quality bin."""
    weights = np.asarray(weights, dtype=np.float64)
    is_clean = np.asarray(is_clean, dtype=bool)
    quality = np.asarray(quality, dtype=np.float64)

    from ..detectability.stratified_auc import quality_bins

    bin_id = quality_bins(quality, n_bins)
    rows = []
    for b in range(n_bins):
        sel = (bin_id == b) & is_clean
        rows.append(
            {
                "quality_bin": b,
                "false_suppression_rate": float(np.mean(weights[sel] < cutoff)) if sel.sum() else float("nan"),
                "n_clean": int(sel.sum()),
            }
        )
    return pd.DataFrame(rows)
