"""Evidence from forgetting dynamics (Toneva et al.)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..traces.dynamics import compute_forgetting_count


def evidence_forgetting(traces: pd.DataFrame, normalize: bool = True) -> np.ndarray:
    """Per-sample forgetting count, optionally normalized by epoch count."""
    pivot = traces.pivot(index="sample_id", columns="epoch", values="correct_to_observed").sort_index()
    counts = compute_forgetting_count(pivot.values.astype(bool))
    if normalize:
        n_epochs = pivot.shape[1]
        return (counts / max(n_epochs, 1)).astype(np.float32)
    return counts.astype(np.float32)
