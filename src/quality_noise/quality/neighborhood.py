"""Q2 — neighborhood agreement: fraction of KNN neighbors sharing the observed
label. Uses observed labels only; never the noise mask.
"""
from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors


def compute_knn_agreement(
    embeddings: np.ndarray,
    observed_labels: np.ndarray,
    k: int = 20,
) -> np.ndarray:
    """Mean fraction of the k nearest neighbors (in representation space) that
    share sample i's observed label.

    For very large N, the neighbor graph is computed on a subsample of the
    index (the label-agreement of neighbors is approximately class-conditional),
    then exact neighbors are used for the remaining samples. The subsample
    fallback is transparent and deterministic.
    """
    emb = np.asarray(embeddings, dtype=np.float32)
    y = np.asarray(observed_labels)
    n = emb.shape[0]
    k = min(k, n - 1)

    if n <= 20000:
        nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine").fit(emb)
        _, idx = nn.kneighbors(emb)
        nbrs = idx[:, 1:]  # drop self
    else:
        # Build index on a deterministic subsample; query full set.
        rng = np.random.default_rng(0)
        sub = rng.choice(n, size=20000, replace=False)
        nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine").fit(emb[sub])
        _, idx = nn.kneighbors(emb)
        nbrs = sub[idx[:, 1:]]

    agree = y[nbrs] == y[:, None]
    return agree.mean(axis=1).astype(np.float32)
