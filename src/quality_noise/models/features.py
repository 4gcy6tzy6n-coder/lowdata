"""Embedding extraction at the penultimate layer (post-avgpool, pre-FC).

Embeddings are written once per run (default: final epoch) to a separate .npy
— they are too large to store per-epoch in the trace parquet.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..utils import to_device


def _penultimate_module(model: torch.nn.Module) -> torch.nn.Module:
    """The module whose *input* is the penultimate representation."""
    if hasattr(model, "fc"):
        return model.fc
    # Fall back to the last Linear layer for models without `.fc`.
    linear = [m for m in model.modules() if isinstance(m, torch.nn.Linear)]
    if linear:
        return linear[-1]
    raise AttributeError("No .fc or Linear layer found to hook penultimate features")


@contextmanager
def feature_hook(model: torch.nn.Module, device: torch.device) -> Iterator[list[torch.Tensor]]:
    """Context manager that collects penultimate activations during forward.

    A pre-hook on the final Linear/FC module captures its *input*, which is the
    penultimate representation. Batches are appended in forward order.
    """
    out: list[torch.Tensor] = []

    def _hook(_module, inp) -> None:
        x = inp[0].detach().to("cpu")
        out.append(x)

    handle = _penultimate_module(model).register_forward_pre_hook(_hook)
    try:
        yield out
    finally:
        handle.remove()


def extract_embeddings(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    view: object | None = None,
) -> np.ndarray:
    """Extract penultimate embeddings for all samples in ``loader``.

    If ``view`` is provided it is used for indexing bookkeeping; otherwise the
    loader order is used directly. Returns a (N, D) float32 array in the same
    order as ``view`` when provided.
    """
    model.eval()
    model = model.to(device)
    embeddings: list[np.ndarray] = []
    with torch.no_grad():
        with feature_hook(model, device) as feats:
            for batch in loader:
                if isinstance(batch, (tuple, list)):
                    x = batch[0]
                else:
                    x = batch
                x = to_device(x, device)
                model(x)
                for f in feats:
                    embeddings.append(f.numpy())
                feats.clear()
    if not embeddings:
        raise RuntimeError("No embeddings collected (empty loader?)")
    arr = np.concatenate([e.reshape(e.shape[0], -1) for e in embeddings], axis=0)
    if view is not None and hasattr(view, "sample_ids"):
        # Reorder to sample_id ascending so embeddings align with view order.
        order = np.argsort(np.asarray(view.sample_ids, dtype=int))
        arr = arr[order]
    return arr.astype(np.float32)
