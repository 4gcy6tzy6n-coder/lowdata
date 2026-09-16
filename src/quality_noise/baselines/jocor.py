"""JoCoR (Wei et al., 2020): joint training with co-regularization. Two networks
minimize a joint loss (CE + agreement), and each is updated on the other's
small joint-loss selection."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from ..models.resnet import build_model
from ..trainers.standard import _make_optimizer
from .common import BaseTrainer


class JoCoRTrainer(BaseTrainer):
    def __init__(self, cfg, train_view, device, callbacks=None, val_view=None, progress=False):
        self.keep_rate = float(cfg.get("baseline", {}).get("jocor_keep_rate", 1.0 - cfg["noise"].get("rate", 0.4)))
        self.lam = float(cfg.get("baseline", {}).get("jocor_lambda", 6.0))
        super().__init__(cfg, train_view, device, callbacks=callbacks, val_view=val_view, progress=progress)

    def init_model(self) -> None:
        num_classes = int(self.cfg["dataset"]["num_classes"])
        self.model = build_model(self.cfg["model"], num_classes)
        self.model2 = build_model(self.cfg["model"], num_classes)
        self.optimizer = _make_optimizer(self.cfg["model"], self.model)
        self.optimizer2 = _make_optimizer(self.cfg["model"], self.model2)
        self.model2.to(self.device)

    @staticmethod
    def _agreement_loss(logits1, logits2) -> torch.Tensor:
        p1 = F.softmax(logits1, dim=-1)
        p2 = F.softmax(logits2, dim=-1)
        return F.mse_loss(p1, p2)

    def train_epoch_step(self, x, y, sample_ids):
        keep = max(int(self.keep_rate * x.size(0)), 1)
        self.optimizer.zero_grad()
        self.optimizer2.zero_grad()

        logits1 = self.model(x)
        logits2 = self.model2(x)
        ce1 = F.cross_entropy(logits1, y, reduction="none")
        ce2 = F.cross_entropy(logits2, y, reduction="none")
        agree = F.mse_loss(F.softmax(logits1, -1), F.softmax(logits2, -1), reduction="none").mean(dim=-1)
        # Joint loss: CE + co-regularization agreement (per sample).
        joint = ce1 + ce2 + self.lam * agree

        _, sel = torch.topk(joint, k=keep, largest=False)

        # Single combined loss so one backward covers both networks (no
        # double-backward through the shared agreement term).
        loss = ce1[sel].mean() + ce2[sel].mean() + self.lam * agree[sel].mean()
        loss.backward()
        self.optimizer.step()
        self.optimizer2.step()
        return loss, logits1.detach(), y, sample_ids
