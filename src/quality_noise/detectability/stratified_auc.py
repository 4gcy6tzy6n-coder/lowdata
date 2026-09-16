"""Quality-stratified AUROC: detector performance within quality bins.

Bins samples by quality quintile and computes AUROC among the noisy/clean
samples that fall *inside* each bin. This is the within-quality view: it asks
"given two samples of comparable quality, can the detector tell noisy from
clean?" — the confounding-controlled comparison.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def quality_bins(quality: np.ndarray, n_bins: int = 5) -> np.ndarray:
    """Map quality to a bin id in [0, n_bins) via quantiles (ties kept together)."""
    q = np.asarray(quality, dtype=np.float64)
    bin_id = np.full(len(q), -1, dtype=np.int32)
    edges = np.quantile(q, np.linspace(0, 1, n_bins + 1))
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        sel = (q >= lo) & (q <= hi) if b == n_bins - 1 else (q >= lo) & (q < hi)
        bin_id[sel] = b
    return bin_id


def quality_stratified_auc(
    score: np.ndarray,
    mask: np.ndarray,
    quality: np.ndarray,
    n_bins: int = 5,
) -> pd.DataFrame:
    """AUROC per quality bin. Rows: quality_bin, q_low, q_high, auc, n_noisy, n_clean."""
    score = np.asarray(score, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    quality = np.asarray(quality, dtype=np.float64)
    bin_id = quality_bins(quality, n_bins)
    edges = np.quantile(quality, np.linspace(0, 1, n_bins + 1))

    rows = []
    for b in range(n_bins):
        sel = bin_id == b
        s, m = score[sel], mask[sel]
        auc = float(roc_auc_score(m, s)) if m.sum() > 0 and (~m).sum() > 0 else float("nan")
        rows.append(
            {
                "quality_bin": b,
                "q_low": float(edges[b]),
                "q_high": float(edges[b + 1]),
                "auc": auc,
                "n_noisy": int(m.sum()),
                "n_clean": int((~m).sum()),
            }
        )
    return pd.DataFrame(rows)
