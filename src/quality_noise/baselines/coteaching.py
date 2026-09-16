"""Co-teaching (Han et al., 2018): two networks, each learns from the other's
small-loss selection."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from ..models.resnet import build_model
from ..trainers.standard import _make_optimizer
from .common import BaseTrainer


class CoTeachingTrainer(BaseTrainer):
    def __init__(self, cfg, train_view, device, callbacks=None, val_view=None, progress=False):
        self.keep_rate = float(cfg.get("baseline", {}).get("coteaching_keep_rate", 1.0 - cfg["noise"].get("rate", 0.4)))
        super().__init__(cfg, train_view, device, callbacks=callbacks, val_view=val_view, progress=progress)

    def init_model(self) -> None:
        num_classes = int(self.cfg["dataset"]["num_classes"])
        self.model = build_model(self.cfg["model"], num_classes)
        self.model2 = build_model(self.cfg["model"], num_classes)
        self.optimizer = _make_optimizer(self.cfg["model"], self.model)
        self.optimizer2 = _make_optimizer(self.cfg["model"], self.model2)
        self.model2.to(self.device)

    def train_epoch_step(self, x, y, sample_ids):
        keep = max(int(self.keep_rate * x.size(0)), 1)
        self.optimizer.zero_grad()
        self.optimizer2.zero_grad()

        logits1 = self.model(x)
        logits2 = self.model2(x)
        loss1_all = F.cross_entropy(logits1, y, reduction="none")
        loss2_all = F.cross_entropy(logits2, y, reduction="none")

        # Each net selects small-loss samples; the other net is updated on them.
        _, sel1 = torch.topk(loss1_all, k=keep, largest=False)
        _, sel2 = torch.topk(loss2_all, k=keep, largest=False)

        loss1 = F.cross_entropy(logits1[sel2], y[sel2])
        loss2 = F.cross_entropy(logits2[sel1], y[sel1])
        loss1.backward()
        loss2.backward()
        self.optimizer.step()
        self.optimizer2.step()
        return (loss1 + loss2) / 2, logits1.detach(), y, sample_ids
