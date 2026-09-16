"""Evidence from training loss dynamics."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..traces.dynamics import compute_ema_loss


def evidence_ema_loss(traces: pd.DataFrame, decay: float = 0.9) -> np.ndarray:
    """Per-sample EMA loss from the trace (sample_id x epoch pivot)."""
    pivot = traces.pivot(index="sample_id", columns="epoch", values="loss").sort_index()
    return compute_ema_loss(pivot.values, decay=decay)


def evidence_mean_loss(traces: pd.DataFrame) -> np.ndarray:
    """Per-sample mean loss over epochs."""
    pivot = traces.pivot(index="sample_id", columns="epoch", values="loss").sort_index()
    return pivot.values.mean(axis=1).astype(np.float32)
