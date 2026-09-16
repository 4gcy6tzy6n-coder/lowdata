"""Co-teaching+ (Yu et al., 2019): adds disagreement-based sample discard with a
linearly decaying drop rate before small-loss selection."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from ..models.resnet import build_model
from ..trainers.standard import _make_optimizer
from .common import BaseTrainer


class CoTeachingPlusTrainer(BaseTrainer):
    def __init__(self, cfg, train_view, device, callbacks=None, val_view=None, progress=False):
        self.keep_rate = float(cfg.get("baseline", {}).get("coteaching_keep_rate", 1.0 - cfg["noise"].get("rate", 0.4)))
        self.forget_rate = float(cfg.get("baseline", {}).get("coteaching_forget_rate", cfg["noise"].get("rate", 0.4)))
        self.num_gradual = int(cfg.get("baseline", {}).get("coteaching_num_gradual", 10))
        super().__init__(cfg, train_view, device, callbacks=callbacks, val_view=val_view, progress=progress)
        self.exponent = 1.0
        self.rate_schedule = None  # set on first epoch in run()

    def init_model(self) -> None:
        num_classes = int(self.cfg["dataset"]["num_classes"])
        self.model = build_model(self.cfg["model"], num_classes)
        self.model2 = build_model(self.cfg["model"], num_classes)
        self.optimizer = _make_optimizer(self.cfg["model"], self.model)
        self.optimizer2 = _make_optimizer(self.cfg["model"], self.model2)
        self.model2.to(self.device)

    def _forget_rate_at(self, epoch: int) -> float:
        """Linearly decay the discard rate to the target over num_gradual epochs."""
        if epoch >= self.num_gradual:
            return self.forget_rate
        return self.forget_rate * (1.0 - epoch / self.num_gradual)

    def on_epoch_begin(self, epoch: int) -> None:
        self._current_forget_rate = self._forget_rate_at(epoch)

    def train_epoch_step(self, x, y, sample_ids):
        rate_sched = getattr(self, "_current_forget_rate", self.forget_rate)
        keep = max(int(self.keep_rate * x.size(0)), 1)
        self.optimizer.zero_grad()
        self.optimizer2.zero_grad()

        logits1 = self.model(x)
        logits2 = self.model2(x)
        # Disagreement mask: drop samples the two nets predict differently,
        # with a fraction growing to the target forget rate.
        pred1 = logits1.argmax(dim=-1)
        pred2 = logits2.argmax(dim=-1)
        disagree = pred1 != pred2
        n_drop = int(rate_sched * x.size(0))
        if n_drop > 0 and disagree.any():
            drop_idx = torch.where(disagree)[0]
            perm = torch.randperm(drop_idx.size(0), device=x.device)[:n_drop]
            mask_keep = torch.ones(x.size(0), dtype=torch.bool, device=x.device)
            mask_keep[drop_idx[perm]] = False
            keep = min(keep, int(mask_keep.sum()))
            l1_all = F.cross_entropy(logits1[mask_keep], y[mask_keep], reduction="none")
            l2_all = F.cross_entropy(logits2[mask_keep], y[mask_keep], reduction="none")
        else:
            mask_keep = torch.ones(x.size(0), dtype=torch.bool, device=x.device)
            l1_all = F.cross_entropy(logits1, y, reduction="none")
            l2_all = F.cross_entropy(logits2, y, reduction="none")

        # sel1/sel2 index into the *kept* subset; apply them to the kept logits.
        _, sel1 = torch.topk(l1_all, k=min(keep, len(l1_all)), largest=False)
        _, sel2 = torch.topk(l2_all, k=min(keep, len(l2_all)), largest=False)
        loss1 = F.cross_entropy(logits1[mask_keep][sel2], y[mask_keep][sel2])
        loss2 = F.cross_entropy(logits2[mask_keep][sel1], y[mask_keep][sel1])
        loss1.backward()
        loss2.backward()
        self.optimizer.step()
        self.optimizer2.step()
        return (loss1 + loss2) / 2, logits1.detach(), y, sample_ids
