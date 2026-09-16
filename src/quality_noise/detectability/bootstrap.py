"""Bootstrap confidence intervals for detectability metrics.

Resampling is done at the *sample* level (noisy and clean drawn with
replacement separately), and the metric is recomputed inside each resample —
for matched AUC this means re-matching within the resample. This keeps the
bootstrap honest about the dependent-pair structure of matching.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from .matched_auc import match_quality_pairs, matched_auc


def _resample_indices(mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Sample-level resample: draw noisy and clean indices separately."""
    noisy = np.where(mask)[0]
    clean = np.where(~mask)[0]
    n_noisy, n_clean = len(noisy), len(clean)
    if n_noisy == 0 or n_clean == 0:
        return np.arange(len(mask))
    b_noisy = noisy[rng.integers(0, n_noisy, size=n_noisy)]
    b_clean = clean[rng.integers(0, n_clean, size=n_clean)]
    return np.concatenate([b_noisy, b_clean])


def bootstrap_ci(
    score: np.ndarray,
    mask: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float] | None = None,
    n_resamples: int = 1000,
    ci: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    """Bootstrap (lo, hi) CI for a metric that does not need quality.

    ``metric_fn(score, mask)`` defaults to global AUROC.
    """
    if metric_fn is None:
        from .global_auc import global_auc

        metric_fn = global_auc

    score = np.asarray(score, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    rng = np.random.default_rng(seed)
    alphas = np.asarray([(1 - ci) / 2, 1 - (1 - ci) / 2])

    stats = np.empty(n_resamples)
    for r in range(n_resamples):
        b_idx = _resample_indices(mask, rng)
        stats[r] = metric_fn(score[b_idx], mask[b_idx])

    valid = stats[~np.isnan(stats)]
    if len(valid) == 0:
        return float("nan"), float("nan")
    return tuple(np.quantile(valid, alphas))


def bootstrap_indexed_ci(
    score: np.ndarray,
    mask: np.ndarray,
    quality: np.ndarray,
    metric: str = "global",
    n_resamples: int = 1000,
    ci: float = 0.95,
    seed: int = 0,
    matched_eps: float | None = None,
    max_clean_per_noisy: int = 5,
    matched_subsample: int | None = 5000,
) -> tuple[float, float]:
    """Bootstrap CI with quality-aware resampling.

    ``metric`` in {"global", "matched"}. For "matched", noisy-clean pairs are
    re-matched within each resample using the resampled quality values. To keep
    runtime bounded, each resample matches a subsample of ``matched_subsample``
    noisy samples (the matched AUC is estimated on a budget); pass None for the
    full set.
    """
    from .global_auc import global_auc

    score = np.asarray(score, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    quality = np.asarray(quality, dtype=np.float64)
    if metric == "matched" and matched_eps is None:
        raise ValueError("bootstrap_indexed_ci(metric='matched') requires matched_eps")
    rng = np.random.default_rng(seed)
    alphas = np.asarray([(1 - ci) / 2, 1 - (1 - ci) / 2])

    noisy = np.where(mask)[0]
    n_noisy = len(noisy)

    stats = np.empty(n_resamples)
    for r in range(n_resamples):
        b_idx = _resample_indices(mask, rng)
        if metric == "global":
            stats[r] = global_auc(score[b_idx], mask[b_idx])
        elif metric == "matched":
            b_mask = mask[b_idx]
            if matched_subsample is not None and n_noisy > matched_subsample:
                sub = rng.choice(n_noisy, size=matched_subsample, replace=False)
                # Keep the subsampled noisy indices plus all resampled clean.
                clean_idx = np.where(~b_mask)[0]
                keep = np.concatenate([sub, clean_idx])
                keep = np.sort(keep)
                b_idx = b_idx[keep]
                b_mask = mask[b_idx]
            pairs = match_quality_pairs(
                score[b_idx], b_mask, quality[b_idx], eps=matched_eps, max_clean_per_noisy=max_clean_per_noisy
            )
            stats[r] = matched_auc(pairs) if len(pairs.s_noisy) else float("nan")
        else:
            raise ValueError(f"Unknown metric: {metric}")

    valid = stats[~np.isnan(stats)]
    if len(valid) == 0:
        return float("nan"), float("nan")
    return tuple(np.quantile(valid, alphas))
