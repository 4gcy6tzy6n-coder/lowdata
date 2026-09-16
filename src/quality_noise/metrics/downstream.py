"""Downstream learning metrics: test accuracy and balanced accuracy."""
from __future__ import annotations

import numpy as np


def accuracy(preds: np.ndarray, targets: np.ndarray) -> float:
    preds = np.asarray(preds)
    targets = np.asarray(targets)
    if preds.shape[0] == 0:
        return 0.0
    return float(np.mean(preds == targets))


def balanced_accuracy(preds: np.ndarray, targets: np.ndarray, num_classes: int | None = None) -> float:
    """Mean per-class recall (balanced accuracy)."""
    preds = np.asarray(preds)
    targets = np.asarray(targets)
    if num_classes is None:
        num_classes = int(max(targets.max(), preds.max())) + 1
    recalls = []
    for c in range(num_classes):
        idx = np.where(targets == c)[0]
        if idx.shape[0] == 0:
            continue
        recalls.append(np.mean(preds[idx] == c))
    return float(np.mean(recalls)) if recalls else 0.0
