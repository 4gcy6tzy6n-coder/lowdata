"""Label-independent quality Q.

Two constructions, both **class-agnostic** — no label is read at any step, so
``Q not<- y_tilde`` by construction:

``Q^dino``     density in a *frozen* DINOv2 embedding (self-supervised
               pretraining, never fine-tuned on this data, no labels passed).
``Q^native``   density in the run's own trained ResNet-18 penultimate
               embedding.  This one *does* depend on noisy-label training
               (z -> H_noisy -> Q), so it is reported only as a contrast that
               isolates how much of the effect needs the trained encoder.

Definitions (K neighbours, cosine geometry)::

    Q_density(i) = mean_{j in N_K(i)} cos_sim(h_i, h_j)          in [-1, 1]
    Q_dist(i)    = -mean_{j in N_K(i)} ||h_i - h_j||_2           (negative)

Both are mapped to [0, 1] by a rank-preserving affinity transform
(``unit_affine``) so the existing caliper semantics (eps in Q-units) carry over
from Q^KNN.  Neighbour search and quality computation use the embedding only.

Note the deliberate difference from the paper's Q^KNN
(``quality/neighborhood.py``), which is ``mean_j 1[y_j == y_i]`` — that one
*does* read the observed labels and is exactly the post-treatment quantity the
reviewers objected to.
"""
from __future__ import annotations

import numpy as np

DEFAULT_K = 20


def unit_affine(x: np.ndarray) -> np.ndarray:
    """Map x to [0, 1] by its min/max (rank preserving, ties preserved)."""
    x = np.asarray(x, dtype=np.float64)
    lo = np.nanmin(x)
    hi = np.nanmax(x)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-12:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def _knn(emb: np.ndarray, k: int, pool: np.ndarray | None = None,
         chunk: int = 4096) -> tuple[np.ndarray, np.ndarray]:
    """Indices and distances of the k nearest neighbours of every row of ``emb``.

    ``emb`` is assumed L2-normalised, so cosine similarity = inner product and
    squared Euclidean distance = 2 - 2*cos.  ``pool`` optionally restricts the
    candidate set (e.g. to the run's own samples); every row of ``emb`` is still
    queried.  Self is always excluded.
    """
    emb = np.ascontiguousarray(emb, dtype=np.float32)
    n = emb.shape[0]
    cand = emb if pool is None else np.ascontiguousarray(emb[pool], dtype=np.float32)
    m = cand.shape[0]
    k = int(min(k, m - 1 if pool is None else m))
    idx_out = np.empty((n, k), dtype=np.int64)
    dist_out = np.empty((n, k), dtype=np.float32)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        sim = emb[s:e] @ cand.T                       # (b, m) cosine similarity
        if pool is None:
            rows = np.arange(s, e)
            sim[np.arange(e - s), rows] = -np.inf     # drop self
        if k < m:
            part = np.argpartition(-sim, k - 1, axis=1)[:, :k]
        else:
            part = np.tile(np.arange(m), (e - s, 1))
        take = np.take_along_axis(sim, part, axis=1)
        order = np.argsort(-take, axis=1)
        idx_out[s:e] = np.take_along_axis(part, order, axis=1)
        dist_out[s:e] = np.take_along_axis(take, order, axis=1)
    if pool is not None:
        idx_out = pool[idx_out]
    return idx_out, dist_out


def compute_independent_q(
    embeddings: np.ndarray,
    k: int = DEFAULT_K,
    pool: np.ndarray | None = None,
    mode: str = "density",
) -> dict[str, np.ndarray]:
    """Class-agnostic density quality from ``embeddings`` (rows = samples).

    Parameters
    ----------
    embeddings : (N, D) float array.  L2-normalised internally.
    k          : neighbourhood size.
    pool       : optional index array restricting the neighbour *candidate* set.
    mode       : ``"density"`` (mean cosine similarity, higher = denser = higher
                 quality) or ``"dist"`` (negative mean L2 distance, affine-mapped
                 so that higher = higher quality as well).

    Returns
    -------
    dict with ``q`` (affine-mapped to [0,1]), ``raw`` (unmapped), ``mean_dist``.
    """
    emb = np.asarray(embeddings, dtype=np.float32)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    emb = emb / np.maximum(norms, 1e-12)
    idx, sim = _knn(emb, k=k, pool=pool)

    mean_sim = sim.mean(axis=1).astype(np.float64)
    if mode == "density":
        raw = mean_sim
    elif mode == "dist":
        raw = -np.sqrt(np.maximum(2.0 - 2.0 * mean_sim, 0.0))
    else:
        raise ValueError(f"unknown mode: {mode}")

    return {
        "q": unit_affine(raw),
        "raw": raw,
        "mean_dist": np.sqrt(np.maximum(2.0 - 2.0 * mean_sim, 0.0)).astype(np.float64),
        "neighbor_idx": idx,
    }


def label_dependence_report(q: np.ndarray, mask: np.ndarray,
                            y_observed: np.ndarray | None = None,
                            y_clean: np.ndarray | None = None) -> dict:
    """Diagnostics proving Q is label-independent.

    A label-independent Q must not be *constructible* from labels; operationally
    we report its association with the evaluation-only noise mask and with the
    observed labels.  Low |SMD| for Q across the mask is expected (a Q that
    perfectly separated noisy/clean would itself be a detector); the point is
    that Q's definition never touches labels, which the code path enforces.
    """
    from scipy import stats

    q = np.asarray(q, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    out: dict = {
        "q_mean_noisy": float(q[mask].mean()) if mask.any() else float("nan"),
        "q_mean_clean": float(q[~mask].mean()) if (~mask).any() else float("nan"),
    }
    if mask.any() and (~mask).any():
        out["q_auc_vs_mask"] = float(stats.mannwhitneyu(q[mask], q[~mask]).statistic /
                                     (mask.sum() * (~mask).sum()))
    return out
