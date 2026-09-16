"""ELR (Liu et al., 2020): Early-Learning Regularization.

A temporal-ensembling target z_i per sample regularizes against memorization:
    L = CE - lambda * sum_i log(1 - exp(-z_i^T softmax(x_i)))

The regularizer is the SCALAR dot product z_i^T f_i (not an elementwise sum).
Since log(1-exp(-z^T f)) <= 0, subtracting lambda*reg adds a penalty that grows
as the prediction diverges from the ensemble target z_i; minimizing therefore
*encourages* agreement. We use log1p(-exp(-a)) for stability and initialize
z_i to the uniform distribution so z_i^T f_i is never exactly 0.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from ..models.resnet import build_model
from ..trainers.standard import _make_optimizer
from .common import BaseTrainer


class ELRTrainer(BaseTrainer):
    def __init__(self, cfg, train_view, device, callbacks=None, val_view=None, progress=False):
        self.lam = float(cfg.get("baseline", {}).get("elr_lambda", 3.0))
        self.beta = float(cfg.get("baseline", {}).get("elr_beta", 0.9))
        super().__init__(cfg, train_view, device, callbacks=callbacks, val_view=val_view, progress=progress)
        self.num_classes = int(cfg["dataset"]["num_classes"])
        # Per-sample temporal ensemble targets as a dense (N, C) tensor,
        # indexed by a sample_id -> row map (vectorized update per batch).
        self._sid2row = {int(s): i for i, s in enumerate(train_view.sample_ids)}
        self._targets = torch.full(
            (len(train_view), self.num_classes), 1.0 / self.num_classes, dtype=torch.float32
        )
        self._targets_dev: torch.Tensor | None = None

    def init_model(self) -> None:
        self.num_classes = int(self.cfg["dataset"]["num_classes"])
        self.model = build_model(self.cfg["model"], self.num_classes)
        self.optimizer = _make_optimizer(self.cfg["model"], self.model)

    def train_epoch_step(self, x, y, sample_ids):
        self.optimizer.zero_grad()
        logits = self.model(x)
        probs = F.softmax(logits, dim=-1)

        # Vectorized temporal ensembling + ELR regularization for the batch.
        if self._targets_dev is None or self._targets_dev.device != self.device:
            self._targets_dev = self._targets.to(self.device)
        rows = torch.tensor(
            [self._sid2row[int(s)] for s in sample_ids.cpu().numpy()], dtype=torch.long
        )
        z = self._targets_dev[rows]
        a = (z * probs).sum(dim=1).clamp(max=30)  # scalar dot product z^T f
        reg = torch.log1p(-torch.exp(-a)).clamp_min(-100.0)
        self._targets_dev[rows] = self.beta * z + (1 - self.beta) * probs.detach()

        ce = F.cross_entropy(logits, y)
        loss = ce - self.lam * reg.mean()
        loss.backward()
        self.optimizer.step()
        return loss, logits.detach(), y, sample_ids
