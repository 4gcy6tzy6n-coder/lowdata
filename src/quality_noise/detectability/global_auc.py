"""Global noise-detection AUROC: P(S_noisy > S_clean) over all samples."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def global_auc(score: np.ndarray, mask: np.ndarray) -> float:
    """AUROC of ``score`` against the boolean noise ``mask`` (1 = noisy).

    Requires at least one clean and one noisy sample. Returns a float in [0, 1].
    """
    score = np.asarray(score, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    if mask.sum() == 0 or (~mask).sum() == 0:
        return float("nan")
    return float(roc_auc_score(mask, score))
