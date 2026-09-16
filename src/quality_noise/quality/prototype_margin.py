"""Q1 — prototype margin: classification ambiguity / representational separability.

The margin is measured in representation space against per-class prototypes,
but prototypes are built from *high-confidence anchors* (selected by observed
softmax confidence) so noisy samples do not directly pollute the prototypes.
Anchor selection and margin computation use observed labels only — never the
noise mask.
"""
from __future__ import annotations

import numpy as np


def select_high_confidence_anchors(
    probs: np.ndarray,
    observed_labels: np.ndarray,
    quantile: float = 0.8,
) -> np.ndarray:
    """Boolean mask of anchor samples: observed-label confidence >= class quantile.

    ``probs`` is the softmax probability of the observed label per sample.
    The quantile is a per-class threshold computed from observed labels only.
    """
    probs = np.asarray(probs, dtype=np.float64)
    y = np.asarray(observed_labels)
    anchors = np.zeros(len(y), dtype=bool)
    for c in np.unique(y):
        cls = np.where(y == c)[0]
        if cls.shape[0] == 0:
            continue
        thr = np.quantile(probs[cls], quantile)
        anchors[cls] = probs[cls] >= thr
    return anchors


def compute_prototypes(
    embeddings: np.ndarray,
    observed_labels: np.ndarray,
    anchor_ids: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    """Per-class prototype = mean of anchor embeddings per observed class."""
    emb = np.asarray(embeddings, dtype=np.float64)
    y = np.asarray(observed_labels)
    prototypes = np.zeros((num_classes, emb.shape[1]), dtype=np.float64)
    for c in range(num_classes):
        sel = np.where((y == c) & anchor_ids)[0]
        if sel.shape[0] > 0:
            prototypes[c] = emb[sel].mean(axis=0)
    return prototypes


def _similarity(embeddings: np.ndarray, prototypes: np.ndarray) -> np.ndarray:
    emb = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
    proto = prototypes / (np.linalg.norm(prototypes, axis=1, keepdims=True) + 1e-8)
    return emb @ proto.T  # (N, C)


def compute_margin(
    embeddings: np.ndarray,
    observed_labels: np.ndarray,
    anchors: np.ndarray,
    num_classes: int,
    use_leave_one_out: bool = True,
) -> np.ndarray:
    """Q1 margin = sim(f_i, mu_yi) - max_{c!=yi} sim(f_i, mu_c).

    ``anchors`` is the boolean anchor mask. With ``use_leave_one_out``, an
    anchor sample's own embedding is excluded from its class prototype so the
    margin is not self-referential.
    """
    emb = np.asarray(embeddings, dtype=np.float64)
    y = np.asarray(observed_labels)
    n = emb.shape[0]
    anchors = np.asarray(anchors, dtype=bool)

    prototypes = compute_prototypes(emb, y, anchors, num_classes)
    sim = _similarity(emb, prototypes)

    if use_leave_one_out:
        # Per-class anchor sums/counts for leave-one-out own-class prototype.
        sums = np.zeros((num_classes, emb.shape[1]))
        counts = np.zeros(num_classes)
        for c in range(num_classes):
            sel = np.where((y == c) & anchors)[0]
            if sel.shape[0] > 0:
                sums[c] = emb[sel].sum(axis=0)
                counts[c] = sel.shape[0]
        for i in range(n):
            c = y[i]
            if anchors[i] and counts[c] > 1:
                loo = (sums[c] - emb[i]) / (counts[c] - 1)
                loo_n = loo / (np.linalg.norm(loo) + 1e-8)
                own_n = emb[i] / (np.linalg.norm(emb[i]) + 1e-8)
                sim[i, c] = float(own_n @ loo_n)

    own = sim[np.arange(n), y]
    masked = sim.copy()
    masked[np.arange(n), y] = -np.inf
    second = masked.max(axis=1)
    return (own - second).astype(np.float32)
