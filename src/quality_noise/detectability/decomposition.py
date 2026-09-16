"""AUC decomposition (Phase 5 / Sprint 4).

Global AUC is a quality-weighted average over cross-quality pairs of noisy
samples in one quality bin and clean samples in another:

    AUC_global = sum_{q,r} P(Q_N=q) P(Q_C=r) P(S_{N,q} > S_{C,r})

The (q, r) pair matrix exposes which part of global AUROC comes from
within-quality separation (diagonal) versus cross-quality separation
(off-diagonal, i.e. quality confounding).
"""
from __future__ import annotations

import numpy as np

from .stratified_auc import quality_bins


def cross_quality_pair_matrix(
    score: np.ndarray,
    mask: np.ndarray,
    quality: np.ndarray,
    n_bins: int = 5,
) -> np.ndarray:
    """(n_bins, n_bins) matrix of P(S_noisy > S_clean | Q_noisy=a, Q_clean=b)."""
    score = np.asarray(score, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    bin_id = quality_bins(np.asarray(quality, dtype=np.float64), n_bins)

    noisy_idx = np.where(mask)[0]
    clean_idx = np.where(~mask)[0]
    if len(noisy_idx) == 0 or len(clean_idx) == 0:
        return np.full((n_bins, n_bins), np.nan)

    mat = np.full((n_bins, n_bins), np.nan)
    for a in range(n_bins):
        n_a = noisy_idx[bin_id[noisy_idx] == a]
        if len(n_a) == 0:
            continue
        for b in range(n_bins):
            c_b = clean_idx[bin_id[clean_idx] == b]
            if len(c_b) == 0:
                continue
            s_n = score[n_a][:, None]
            s_c = score[c_b][None, :]
            wins = (s_n > s_c).mean()
            ties = (s_n == s_c).mean()
            mat[a, b] = wins + 0.5 * ties
    return mat


def confounding_gap(
    score: np.ndarray,
    mask: np.ndarray,
    quality: np.ndarray,
    n_bins: int = 5,
) -> float:
    """Mean off-diagonal minus mean diagonal of the pair matrix.

    A large positive value means global AUC is substantially inflated by
    cross-quality comparisons — quality confounding.
    """
    mat = cross_quality_pair_matrix(score, mask, quality, n_bins)
    diag = np.diag(mat)
    mask_valid = ~np.isnan(mat)
    off = mat[mask_valid & ~np.eye(n_bins, dtype=bool)]
    if len(off) == 0:
        return float("nan")
    return float(np.nanmean(off) - np.nanmean(diag))
