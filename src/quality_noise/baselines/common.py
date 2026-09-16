"""Shared harness for strong noisy-label baselines.

Every baseline subclasses ``BaseTrainer``, implements ``train_epoch_step``
(which returns the loss and the main net's (logits, observed_labels,
sample_ids) for trace recording), and calls ``run()``. This keeps the training
loop, seeding, scheduler, callbacks, and validation identical across baselines
and comparable with the standard / weighted trainers.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from ..trainers.standard import (
    History,
    _ImageWrapper,
    _accuracy,
    _make_optimizer,
    _make_scheduler,
    default_train_transform,
    default_eval_transform,
)
from ..utils import set_seed, to_device


def build_train_loader(cfg: dict, train_view: Dataset):
    ds = _ImageWrapper(train_view, default_train_transform(cfg["dataset"]))
    return DataLoader(
        ds,
        batch_size=int(cfg["model"]["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["model"]["num_workers"]),
        pin_memory=True,
        drop_last=False,
    )


def build_val_loader(cfg: dict, val_view: Dataset | None):
    if val_view is None:
        return None
    ds = _ImageWrapper(val_view, default_eval_transform(cfg["dataset"]))
    return DataLoader(ds, batch_size=int(cfg["model"]["batch_size"]), shuffle=False, num_workers=int(cfg["model"]["num_workers"]))


class BaseTrainer:
    """Loop skeleton shared by all baselines."""

    def __init__(self, cfg, train_view, device, callbacks=None, val_view=None, progress=False):
        self.cfg = cfg
        self.device = device
        self.callbacks = callbacks or []
        self.epochs = int(cfg["model"]["epochs"])
        set_seed(int(cfg.get("seed", 0)), device)
        self.train_loader = build_train_loader(cfg, train_view)
        self.val_loader = build_val_loader(cfg, val_view)
        self.history = History()
        self.progress = progress
        # Subclasses build models + optimizers in init_model().
        self.model = None
        self.optimizer = None
        self.cfg["model"]["_dataset_cfg"] = self.cfg["dataset"]  # for mlp arch
        self.init_model()
        # Move all networks to the device (single- and two-net baselines).
        self.model = self.model.to(self.device)
        if getattr(self, "model2", None) is not None:
            self.model2 = self.model2.to(self.device)
        self.scheduler = _make_scheduler(cfg["model"], self.optimizer, self.epochs) if self.optimizer else None

    def init_model(self) -> None:
        """Build self.model and self.optimizer (also self.model2 for two-net)."""
        raise NotImplementedError

    def train_epoch_step(self, x, y, sample_ids, *extra) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (loss, logits, observed_labels, sample_ids) for the main net.
        ``extra`` carries additional views for multi-view baselines (DivideMix)."""
        raise NotImplementedError

    def on_epoch_begin(self, epoch: int) -> None:
        """Hook for baselines that need a per-epoch schedule."""

    def run(self) -> History:
        for epoch in range(self.epochs):
            self.on_epoch_begin(epoch)
            self.model.train()
            running_loss, running_acc, n_seen = 0.0, 0.0, 0
            for batch in self.train_loader:
                x, y, sids = batch[0], batch[1], batch[2]
                extra = tuple(
                    b.to(self.device, non_blocking=True) if torch.is_tensor(b) else b for b in batch[3:]
                )
                x, y, sids = to_device((x, y, sids), self.device)
                loss, logits, y_obs, sids_ret = self.train_epoch_step(x, y, sids, *extra)
                running_loss += float(loss.item()) * x.size(0)
                running_acc += _accuracy(logits, y_obs) * x.size(0)
                n_seen += x.size(0)
                for cb in self.callbacks:
                    cb.on_train_batch(logits, y_obs, sids_ret, epoch)
            for cb in self.callbacks:
                cb.on_epoch_end()
            self.history.train_loss.append(running_loss / max(n_seen, 1))
            self.history.train_acc.append(running_acc / max(n_seen, 1))
            if self.val_loader is not None:
                self._validate(epoch)
            if self.scheduler is not None:
                self.scheduler.step()
        return self.history

    def _validate(self, epoch: int) -> None:
        import torch.nn.functional as F

        self.model.eval()
        v_loss, v_acc, v_n = 0.0, 0.0, 0
        with torch.no_grad():
            for x, y, _sids in self.val_loader:
                x, y = to_device((x, y), self.device)
                logits = self.model(x)
                v_loss += F.cross_entropy(logits, y).item() * x.size(0)
                v_acc += _accuracy(logits, y) * x.size(0)
                v_n += x.size(0)
        self.history.val_loss.append(v_loss / max(v_n, 1))
        self.history.val_acc.append(v_acc / max(v_n, 1))
