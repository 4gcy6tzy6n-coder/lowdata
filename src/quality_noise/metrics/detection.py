"""Noise-detection metrics: AUROC, AUPRC, precision/recall at thresholds."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def detection_report(
    score: np.ndarray,
    mask: np.ndarray,
    thresholds: tuple[float, ...] = (0.5, 0.8, 0.9),
) -> dict:
    """Detection metrics as a flat dict (used by Gate B evaluation)."""
    score = np.asarray(score, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    report: dict = {}
    if mask.sum() > 0 and (~mask).sum() > 0:
        report["auroc"] = float(roc_auc_score(mask, score))
        report["auprc"] = float(average_precision_score(mask, score))
    else:
        report["auroc"] = report["auprc"] = float("nan")
    for t in thresholds:
        pred = score >= t
        tp = (pred & mask).sum()
        fp = (pred & ~mask).sum()
        fn = (~pred & mask).sum()
        report[f"precision@{t}"] = float(tp / max(tp + fp, 1))
        report[f"recall@{t}"] = float(tp / max(tp + fn, 1))
    return report
