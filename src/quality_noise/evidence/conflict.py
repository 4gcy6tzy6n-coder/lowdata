"""Evidence from neighborhood label conflict (observed labels only).

Uses a GPU-accelerated KNN agreement (torch matmul + topk) when CUDA is
available, falling back to the CPU sklearn NearestNeighbors otherwise. The
result is identical in spirit (cosine KNN agreement on a deterministic 20k
subsample index) but much faster, which matters because the evidence is
recomputed per run in the estimator.
"""
from __future__ import annotations

import numpy as np

from ..quality.neighborhood import compute_knn_agreement


def _gpu_knn_agreement(emb: np.ndarray, y: np.ndarray, k: int) -> np.ndarray:
    """GPU KNN label-agreement mirroring compute_knn_agreement's subsampling."""
    import torch
    import torch.nn.functional as F

    emb = np.asarray(emb, dtype=np.float32)
    n = emb.shape[0]
    k = min(k, n - 1)
    rng = np.random.default_rng(0)
    sub = rng.choice(n, size=min(20000, n), replace=False) if n > 20000 else np.arange(n)

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        E = torch.from_numpy(emb).to(device)
        S = torch.from_numpy(emb[sub]).to(device)
        E = F.normalize(E, dim=1)
        S = F.normalize(S, dim=1)
        with torch.no_grad():
            sims = E @ S.T
            _, idx = sims.topk(k + 1, dim=1)
        nbrs = sub[idx[:, 1:].cpu().numpy()]
        agree = y[nbrs] == y[:, None]
        return agree.mean(axis=1).astype(np.float32)
    except (RuntimeError, ValueError):
        # Out of memory / device issues: fall back to CPU.
        return compute_knn_agreement(emb, y, k=k)


def evidence_neighborhood_conflict(
    embeddings: np.ndarray,
    observed_labels: np.ndarray,
    k: int = 20,
) -> np.ndarray:
    """Fraction of KNN neighbors with a *different* observed label = 1 - agreement."""
    try:
        import torch

        if torch.cuda.is_available():
            agreement = _gpu_knn_agreement(embeddings, observed_labels, k=k)
        else:
            agreement = compute_knn_agreement(embeddings, observed_labels, k=k)
    except Exception:
        agreement = compute_knn_agreement(embeddings, observed_labels, k=k)
    return (1.0 - agreement).astype(np.float32)
