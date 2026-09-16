"""Held-out test-set evaluation shared by weighted training and baselines."""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data.datasets import load_raw_dataset, to_numpy_arrays
from .metrics.downstream import accuracy, balanced_accuracy
from .trainers.standard import default_eval_transform


def evaluate_test(model, cfg, device) -> dict:
    """Evaluate a trained model on the held-out test set.

    Returns test accuracy and balanced accuracy.
    """
    _raw_train, raw_test = load_raw_dataset(cfg["dataset"])
    imgs, labels = to_numpy_arrays(raw_test)
    transform = default_eval_transform(cfg["dataset"])
    loader = DataLoader(imgs, batch_size=int(cfg["model"]["batch_size"]), shuffle=False, num_workers=0)
    model.eval()
    model = model.to(device)
    preds = []
    with torch.no_grad():
        for x in loader:
            x = torch.stack([transform(t) for t in x])
            preds.append(model(x.to(device)).argmax(dim=-1).cpu().numpy())
    preds = np.concatenate(preds)
    return {
        "test_accuracy": accuracy(preds, labels),
        "test_balanced_accuracy": balanced_accuracy(preds, labels, num_classes=int(cfg["dataset"]["num_classes"])),
    }
