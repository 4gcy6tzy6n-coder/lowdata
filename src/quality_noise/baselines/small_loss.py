"""Small-loss selection baseline: per-batch keep the smallest-loss samples."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from ..models.resnet import build_model
from ..trainers.standard import _make_optimizer
from .common import BaseTrainer


class SmallLossTrainer(BaseTrainer):
    def __init__(self, cfg, train_view, device, callbacks=None, val_view=None, progress=False):
        self.keep_rate = float(cfg.get("baseline", {}).get("small_loss_keep_rate", 1.0 - cfg["noise"].get("rate", 0.4)))
        super().__init__(cfg, train_view, device, callbacks=callbacks, val_view=val_view, progress=progress)

    def init_model(self) -> None:
        num_classes = int(self.cfg["dataset"]["num_classes"])
        self.model = build_model(self.cfg["model"], num_classes)
        self.optimizer = _make_optimizer(self.cfg["model"], self.model)

    def train_epoch_step(self, x, y, sample_ids):
        self.optimizer.zero_grad()
        logits = self.model(x)
        per_sample = F.cross_entropy(logits, y, reduction="none")
        keep = int(self.keep_rate * x.size(0))
        if keep < x.size(0) and keep > 0:
            _, sel = torch.topk(per_sample, k=keep, largest=False)
            loss = per_sample[sel].mean()
        else:
            loss = per_sample.mean()
        loss.backward()
        self.optimizer.step()
        return loss, logits.detach(), y, sample_ids
