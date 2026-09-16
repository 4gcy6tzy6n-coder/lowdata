"""Q3 — representation stability: cosine similarity between two augmented views
of the same input. Measures how robust the representation is to augmentation.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from ..models.features import feature_hook


def compute_representation_stability(
    model: torch.nn.Module,
    view,
    device: torch.device,
    transform,
    n_views: int = 2,
    batch_size: int = 256,
) -> np.ndarray:
    """Per-sample stability = mean cosine similarity across augmented views.

    The view is iterated directly (no DataLoader) so every sample gets the
    correct global index; ``transform`` is applied twice (independent random
    draws) per sample. ``n_views`` pairs are averaged. Returns a (N,) float32
    array in view sample order.
    """
    model.eval()
    model = model.to(device)
    n = len(view)
    all_sim: list[np.ndarray] = []

    with torch.no_grad():
        for start in range(0, n, batch_size):
            idx = list(range(start, min(start + batch_size, n)))
            x1 = torch.stack([transform(view[i][0]) for i in idx])
            x2 = torch.stack([transform(view[i][0]) for i in idx])
            sims = []
            for _ in range(n_views):
                with feature_hook(model, device) as feats:
                    model(x1.to(device))
                    model(x2.to(device))
                e1 = feats[0].reshape(feats[0].shape[0], -1)
                e2 = feats[1].reshape(feats[1].shape[0], -1)
                sims.append(F.cosine_similarity(e1, e2).cpu().numpy())
            all_sim.append(np.mean(sims, axis=0))
    return np.concatenate(all_sim).astype(np.float32)
