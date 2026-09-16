"""Assemble a single scalar noise-evidence score E_i per sample.

The evidence is a fixed-weight combination of a small set of signals (EMA loss,
forgetting, neighborhood conflict). It is NOT a trained classifier — weights
are constants from config. Signals are rank-normalized to [0, 1].
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .conflict import evidence_neighborhood_conflict
from .forgetting import evidence_forgetting
from .loss import evidence_ema_loss

RECIPES: dict[str, dict] = {
    "ema_loss": {"ema_loss": 1.0},
    "forgetting": {"forgetting": 1.0},
    "conflict": {"conflict": 1.0},
    "ema_loss_forgetting_conflict": {"ema_loss": 0.5, "forgetting": 0.3, "conflict": 0.2},
}


def _robust_normalize(x: np.ndarray) -> np.ndarray:
    """Robust min-max normalization to [0, 1], clipping 1%/99% outliers.

    Unlike rank normalization (which maps to a uniform distribution), this
    preserves the *bimodal* shape of the evidence so the 2-component Beta
    mixture (governance/mixture.py) can separate clean from noisy samples.
    """
    x = np.asarray(x, dtype=np.float64)
    lo, hi = np.quantile(x, [0.01, 0.99])
    span = (hi - lo) or 1.0
    return np.clip((x - lo) / span, 0.0, 1.0)


def assemble_evidence(
    recipe: str,
    traces: pd.DataFrame,
    embeddings: np.ndarray,
    observed_labels: np.ndarray,
    conflict_k: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the scalar evidence score E and its per-signal breakdown.

    Returns (E, signal_matrix) where E has shape (N,) in trace sample order and
    signal_matrix is (N, n_signals) of the rank-normalized raw signals.
    """
    if recipe not in RECIPES:
        raise ValueError(f"Unknown evidence recipe: {recipe}")
    weights = RECIPES[recipe]

    signals: list[np.ndarray] = []
    names: list[str] = []
    for name, w in weights.items():
        if name == "ema_loss":
            s = evidence_ema_loss(traces)
        elif name == "forgetting":
            s = evidence_forgetting(traces)
        elif name == "conflict":
            s = evidence_neighborhood_conflict(embeddings, observed_labels, k=conflict_k)
        else:
            raise ValueError(f"Unknown signal: {name}")
        # All signals are in trace sample_id ascending order (embeddings are
        # extracted aligned to the view, which is ordered by sample_id).
        signals.append(_robust_normalize(s) * w)
        names.append(name)

    signal_matrix = np.stack(signals, axis=1)
    E = signal_matrix.sum(axis=1)
    E = _robust_normalize(E).astype(np.float32)
    return E, signal_matrix
