"""Matched-quality AUROC: compare noisy samples only against clean samples of
similar quality.

Matching controls for quality confounding. Each noisy sample is matched (greedy,
with replacement capped) to clean samples within ``|Q_i - Q_j| <= eps``.
The matched AUC is P(S_noisy > S_clean | Q_noisy ≈ Q_clean). The confounding
gap is Delta_conf = AUC_global - AUC_matched.

This is a *descriptive* quantity (matching creates dependent pairs), so
bootstrap CIs are computed by resampling samples and re-matching within each
resample — see ``bootstrap.py``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MatchedPairs:
    s_noisy: np.ndarray  # (P,) noisy scores
    s_clean: np.ndarray  # (P,) matched clean scores
    q_noisy: np.ndarray  # (P,) noisy sample quality
    q_clean: np.ndarray  # (P,) matched clean sample quality


def match_quality_pairs(
    score: np.ndarray,
    mask: np.ndarray,
    quality: np.ndarray,
    eps: float = 0.1,
    max_clean_per_noisy: int = 5,
    candidate_cap: int = 512,
) -> MatchedPairs:
    """Greedy |Q_i - Q_j| <= eps matching of noisy samples to clean samples.

    Every noisy sample with at least one qualifying clean partner contributes
    ``min(max_clean_per_noisy, n_matches)`` pairs; clean samples may be reused
    up to ``max_clean_per_noisy`` times. Returns the matched pair set.
    """
    score = np.asarray(score, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    quality = np.asarray(quality, dtype=np.float64)

    noisy_idx = np.where(mask)[0]
    clean_idx = np.where(~mask)[0]
    if len(noisy_idx) == 0 or len(clean_idx) == 0:
        return MatchedPairs(
            s_noisy=np.array([]), s_clean=np.array([]), q_noisy=np.array([]), q_clean=np.array([])
        )

    q_noisy = quality[noisy_idx]
    q_clean = quality[clean_idx]

    # Clean samples sorted by quality once, then nearest-neighbor search per noisy.
    order = np.argsort(q_clean)
    q_sorted = q_clean[order]
    clean_sorted = clean_idx[order]

    # Enforce the reuse cap: each clean sample may be matched to at most
    # ``max_clean_per_noisy`` noisy samples (prevents one clean sample from
    # dominating the pair set).
    reuse = np.zeros(len(clean_sorted), dtype=np.int32)

    s_noisy_list, s_clean_list, qn_list, qc_list = [], [], [], []

    for n_i, n_q in zip(noisy_idx, q_noisy):
        # Binary search window in sorted clean qualities.
        lo = int(np.searchsorted(q_sorted, n_q - eps, side="left"))
        hi = int(np.searchsorted(q_sorted, n_q + eps, side="right"))
        window = np.arange(lo, hi)
        if len(window) == 0:
            continue
        # Cap the candidate window to the ``candidate_cap`` nearest clean samples
        # in quality. Matching only ever uses the nearest few, so this barely
        # changes results while bounding the runtime (important inside bootstrap).
        if len(window) > candidate_cap:
            wq = q_sorted[window]
            nearest = window[np.argpartition(np.abs(wq - n_q), candidate_cap - 1)[:candidate_cap]]
            window = nearest
        # Skip clean samples that have reached their reuse cap.
        available = window[reuse[window] < max_clean_per_noisy]
        if len(available) == 0:
            continue
        cand_q = q_sorted[available]
        dist = np.abs(cand_q - n_q)
        # argpartition finds the nearest few in O(n) instead of O(n log n).
        k = min(max_clean_per_noisy, len(available))
        part_idx = np.argpartition(dist, k - 1)[:k]
        cand_order = available[part_idx[np.argsort(dist[part_idx])]]
        chosen = cand_order
        reuse[chosen] += 1
        for c_pos in chosen:
            c = clean_sorted[c_pos]
            s_noisy_list.append(score[n_i])
            s_clean_list.append(score[c])
            qn_list.append(n_q)
            qc_list.append(quality[c])

    return MatchedPairs(
        s_noisy=np.asarray(s_noisy_list),
        s_clean=np.asarray(s_clean_list),
        q_noisy=np.asarray(qn_list),
        q_clean=np.asarray(qc_list),
    )


def matched_auc(pairs: MatchedPairs) -> float:
    """P(S_noisy > S_clean) over the matched pairs (with ties -> 0.5)."""
    if len(pairs.s_noisy) == 0:
        return float("nan")
    wins = (pairs.s_noisy > pairs.s_clean).mean()
    ties = (pairs.s_noisy == pairs.s_clean).mean()
    return float(wins + 0.5 * ties)
