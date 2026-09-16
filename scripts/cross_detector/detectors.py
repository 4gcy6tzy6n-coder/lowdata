"""Detector implementations for cross-detector detectability analysis.

All detectors follow the convention: E_i ↑ ⇒ more likely noisy.

D1 — EMA Loss            (per-sample EMA over the loss trajectory, decay 0.9)
D2 — 1 - mean(observed-label confidence over last 20 epochs)
D3 — -AUM                (negative average margin over all epochs)
D4 — Forgetting Events   (Toneva et al. ever-correct count / n_epochs)
D5 — Neighborhood Conflict (1 - mean(KNN agreement with observed labels))
D6 — Combined Evidence    (0.5·D1 + 0.3·D4 + 0.2·D5, robust-normalized;
                          the original "ours" detector)

IMPORTANT: D1, D4 and D5 reuse the exact signal functions the paper pipeline
uses (quality_noise.evidence.loss/forgetting/conflict) so D6 *reproduces* the
paper's combined evidence (AUC_global etc. match 04_detectability.py output).
D2 and D3 are independent detectors computed from the saved confidence /
margin trajectories.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from quality_noise.evidence.conflict import evidence_neighborhood_conflict
from quality_noise.evidence.forgetting import evidence_forgetting
from quality_noise.evidence.loss import evidence_ema_loss
from quality_noise.evidence.signals import _robust_normalize


def detector_d1_ema_loss(traces: pd.DataFrame) -> np.ndarray:
    """D1: EMA loss per sample (identical to the paper's ema_loss signal)."""
    return _robust_normalize(evidence_ema_loss(traces))


def detector_d2_low_confidence(traces: pd.DataFrame, last_k: int = 20) -> np.ndarray:
    """D2: 1 - mean(observed-label confidence) over last ``last_k`` epochs.

    The confidence column is the softmax probability of the OBSERVED label
    (see 02_collect_traces.py), so this is the low-observed-confidence signal.
    """
    pivot = traces.pivot(index="sample_id", columns="epoch", values="confidence").sort_index()
    tail = pivot.values[:, -last_k:]
    mean_conf = tail.mean(axis=1)
    return _robust_normalize(1.0 - mean_conf)


def detector_d3_aum(traces: pd.DataFrame) -> np.ndarray:
    """D3: -AUM where AUM_i = mean(margin_i over epochs).

    Higher mean margin ⇒ easier ⇒ less likely noisy ⇒ negative score.
    """
    pivot = traces.pivot(index="sample_id", columns="epoch", values="margin").sort_index()
    aum = pivot.values.mean(axis=1)
    return _robust_normalize(-aum)


def detector_d4_forgetting(traces: pd.DataFrame) -> np.ndarray:
    """D4: forgetting events (identical to the paper's forgetting signal).

    Toneva et al. ever-correct counting, normalized by number of epochs.
    """
    return _robust_normalize(evidence_forgetting(traces, normalize=True))


def detector_d5_neighborhood_conflict(
    embeddings: np.ndarray,
    observed_labels: np.ndarray,
    k: int = 20,
) -> np.ndarray:
    """D5: 1 - KNN observed-label agreement (identical to the paper signal)."""
    return _robust_normalize(
        evidence_neighborhood_conflict(embeddings, observed_labels, k=k)
    )


def detector_d6_combined(
    traces: pd.DataFrame,
    embeddings: np.ndarray,
    observed_labels: np.ndarray,
    conflict_k: int = 20,
) -> np.ndarray:
    """D6: combined evidence = 0.5·loss + 0.3·forget + 0.2·conflict.

    Uses the paper's assemble_evidence recipe so output *reproduces* the
    combined detector used in the paper (04_detectability.py).
    """
    from quality_noise.evidence.signals import assemble_evidence

    E, _ = assemble_evidence(
        "ema_loss_forgetting_conflict",
        traces,
        embeddings,
        observed_labels,
        conflict_k=conflict_k,
    )
    return np.asarray(E, dtype=np.float32)


# Detector registry (label for figures/tables).
DETECTORS: dict[str, str] = {
    "ema_loss": "EMA Loss",
    "confidence": "Confidence",
    "aum": "AUM",
    "forgetting": "Forgetting",
    "neighbor": "Neighbor",
    "combined": "Combined",
}


def compute_all_detectors(
    traces: pd.DataFrame,
    embeddings: np.ndarray,
    observed_labels: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compute all 6 detectors, each in sample_id ascending order (N,)."""
    d1 = detector_d1_ema_loss(traces)
    d2 = detector_d2_low_confidence(traces, last_k=20)
    d3 = detector_d3_aum(traces)
    d4 = detector_d4_forgetting(traces)
    d5 = detector_d5_neighborhood_conflict(embeddings, observed_labels, k=20)
    d6 = detector_d6_combined(traces, embeddings, observed_labels, conflict_k=20)
    n = len(d1)
    return {
        "ema_loss": d1[:n],
        "confidence": d2[:n],
        "aum": d3[:n],
        "forgetting": d4[:n],
        "neighbor": d5[:n],
        "combined": d6[:n],
    }
