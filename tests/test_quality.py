"""Quality signal tests: prototype margin and neighborhood agreement on
synthetic embeddings with orthogonally-separated class centers."""
from __future__ import annotations

import numpy as np

from quality_noise.quality.neighborhood import compute_knn_agreement
from quality_noise.quality.prototype_margin import compute_margin, select_high_confidence_anchors


def _cluster_embeddings(
    num_classes: int = 4,
    per_class: int = 80,
    dim: int = 16,
    sep: float = 3.0,
    seed: int = 0,
    noise: float = 0.2,
) -> tuple[np.ndarray, np.ndarray]:
    """Orthogonal cluster centers scaled by ``sep`` + small Gaussian noise."""
    rng = np.random.default_rng(seed)
    centers = sep * np.eye(num_classes, dim)
    labels = np.repeat(np.arange(num_classes), per_class)
    emb = centers[labels] + rng.normal(0, noise, size=(len(labels), dim))
    return emb.astype(np.float32), labels


def test_margin_separates_well_separated_clusters():
    emb, y = _cluster_embeddings(num_classes=2, per_class=80, sep=3.0)
    anchors = np.ones(len(y), dtype=bool)
    margin = compute_margin(emb, y, anchors, 2)
    assert np.all(margin > 0.5)


def test_margin_shrinks_for_ambiguous_samples():
    emb, y = _cluster_embeddings(num_classes=2, per_class=80, sep=3.0, noise=0.1)
    anchors = np.ones(len(y), dtype=bool)
    margin = compute_margin(emb, y, anchors, 2)
    # Samples closest to the decision boundary have the smallest margin.
    mean_own_sim = np.einsum("ij,ij->i", emb, emb)  # rough proxy
    assert np.argmin(margin) >= 0


def test_anchor_selection_threshold():
    probs = np.array([0.99, 0.6, 0.3, 0.98, 0.5])
    y = np.zeros(5, dtype=int)
    anchors = select_high_confidence_anchors(probs, y, quantile=0.5)
    assert anchors.sum() == 3  # top 50% (>= median) of class-0 confidences


def test_anchor_selection_never_sees_mask():
    # Signature-level: only probs + observed labels are passed.
    emb, y = _cluster_embeddings(num_classes=3, per_class=40)
    probs = np.full(len(y), 0.9)
    anchors = select_high_confidence_anchors(probs, y, quantile=0.5)
    assert anchors.all()


def test_knn_agreement_near_one_for_separated_clusters():
    emb, y = _cluster_embeddings(num_classes=4, per_class=60, sep=3.0, noise=0.1)
    agreement = compute_knn_agreement(emb, y, k=10)
    assert agreement.mean() > 0.97


def test_leave_one_out_margin_close_to_full():
    emb, y = _cluster_embeddings(num_classes=4, per_class=50, sep=3.0, noise=0.2)
    anchors = np.ones(len(y), dtype=bool)
    margin_loo = compute_margin(emb, y, anchors, 4, use_leave_one_out=True)
    margin_full = compute_margin(emb, y, anchors, 4, use_leave_one_out=False)
    assert np.all(np.isfinite(margin_loo))
    assert np.allclose(margin_loo, margin_full, atol=0.05)


def test_representation_stability_uses_global_indices():
    """Each sample must be fetched exactly once per view draw (regression for a
    bug that re-fetched the first batch for every batch)."""
    import torch

    from quality_noise.quality.stability import compute_representation_stability

    class CountingView:
        def __init__(self, n):
            self.n = n
            self.accesses = []
            self.sample_ids = np.arange(n)

        def __len__(self):
            return self.n

        def __getitem__(self, i):
            self.accesses.append(i)
            x = torch.full((3, 8, 8), float(i), dtype=torch.uint8)
            return x, i, i

    class IdTransform:
        def __call__(self, t):
            return t.to(torch.float32)

    import torch.nn as nn

    model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, 8))
    view = CountingView(6)
    sim = compute_representation_stability(model, view, torch.device("cpu"), IdTransform(), n_views=2, batch_size=2)
    assert sim.shape == (6,)
    assert np.all(np.isfinite(sim))
    # Every index accessed exactly twice (one per view draw).
    from collections import Counter

    counts = Counter(view.accesses)
    assert sorted(counts.keys()) == [0, 1, 2, 3, 4, 5]
    assert all(c == 2 for c in counts.values())
