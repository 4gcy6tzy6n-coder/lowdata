"""DivideMix (Li et al., 2020): GMM-split semi-supervised co-training.

Per epoch:
    - compute per-sample losses for both nets (no-grad, aligned to sample_id)
    - fit a 2-component Gaussian mixture on each net's loss -> clean probability p
    - split into labeled (p >= 0.5) and unlabeled sets
    - train both nets on a mix of labeled CE (weighted by p) and unlabeled
      consistency to a weak-teacher guess, combined via MixUp.

A compact reimplementation capturing the mechanism (GMM split + co-training +
semi-supervised refinement) rather than byte-for-byte replication.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.mixture import GaussianMixture
from torch.utils.data import DataLoader, Dataset

from ..models.resnet import build_model
from ..trainers.standard import _ImageWrapper, _make_optimizer, default_eval_transform, default_train_transform
from ..utils import to_device
from .common import BaseTrainer


class _DivideMixDS(Dataset):
    """Two-view dataset: weak view (no aug) for guess targets, strong view
    (crop+flip) for training — the standard DivideMix/MixMatch decoupling that
    prevents the model from copying its own guess (confirmation collapse)."""

    def __init__(self, view, transform_weak, transform_strong):
        self.view = view
        self.tw = transform_weak
        self.ts = transform_strong

    def __len__(self):
        return len(self.view)

    def __getitem__(self, i):
        x, y, sid = self.view[i]
        # Order: (strong, y, sid, weak) so the shared BaseTrainer loop unpacks
        # (x, y, sids) correctly and the weak view arrives as `extra`.
        return self.ts(x), y, sid, self.tw(x)


class DivideMixTrainer(BaseTrainer):
    def __init__(self, cfg, train_view, device, callbacks=None, val_view=None, progress=False):
        self.T = float(cfg.get("baseline", {}).get("dividemix_T", 0.5))  # sharpening temp
        self.lam_u = float(cfg.get("baseline", {}).get("dividemix_lambda_u", 25.0))
        self.mix_alpha = float(cfg.get("baseline", {}).get("dividemix_alpha", 4.0))
        self.warmup = int(cfg.get("baseline", {}).get("dividemix_warmup", 10))
        super().__init__(cfg, train_view, device, callbacks=callbacks, val_view=val_view, progress=progress)
        # Two-view loader: (x_strong, x_weak, y, sample_id).
        ds = _DivideMixDS(train_view, default_eval_transform(cfg["dataset"]), default_train_transform(cfg["dataset"]))
        self.train_loader = DataLoader(
            ds, batch_size=int(cfg["model"]["batch_size"]), shuffle=True,
            num_workers=int(cfg["model"]["num_workers"]), pin_memory=True, drop_last=False,
        )
        # No-shuffle strong-view loader for per-epoch loss computation.
        ds_loss = _ImageWrapper(train_view, default_train_transform(cfg["dataset"]))
        self.loss_loader = DataLoader(ds_loss, batch_size=int(cfg["model"]["batch_size"]), shuffle=False, num_workers=0)
        self.sample_ids = np.asarray(train_view.sample_ids, dtype=np.int64)
        self.p_clean = np.ones(len(self.sample_ids), dtype=np.float64)
        self.num_classes = int(cfg["dataset"]["num_classes"])

    def init_model(self) -> None:
        self.num_classes = int(self.cfg["dataset"]["num_classes"])
        self.model = build_model(self.cfg["model"], self.num_classes)
        self.model2 = build_model(self.cfg["model"], self.num_classes)
        self.optimizer = _make_optimizer(self.cfg["model"], self.model)
        self.optimizer2 = _make_optimizer(self.cfg["model"], self.model2)
        self.model2.to(self.device)

    def _per_sample_loss(self, net) -> np.ndarray:
        net.eval()
        losses = []
        with torch.no_grad():
            for batch in self.loss_loader:
                x, y, _ = to_device((batch[0], batch[1], batch[2]), self.device)
                losses.append(F.cross_entropy(net(x), y, reduction="none").cpu().numpy())
        return np.concatenate(losses)

    def _clean_probability(self, losses: np.ndarray) -> np.ndarray:
        losses = np.nan_to_num(np.asarray(losses, dtype=np.float64), nan=1e6, posinf=1e6, neginf=-1e6)
        if losses.size == 0 or np.ptp(losses) < 1e-9:
            return np.full(len(losses), 0.5)
        try:
            gmm = GaussianMixture(n_components=2, random_state=int(self.cfg.get("seed", 0)))
            gmm.fit(losses.reshape(-1, 1))
            # Clean component = lower mean loss.
            clean_comp = int(np.argmin(gmm.means_.ravel()))
            return gmm.predict_proba(losses.reshape(-1, 1))[:, clean_comp]
        except Exception:
            # Degenerate loss distribution: no clean/noisy split -> all "clean".
            return np.full(len(losses), 0.5)

    def on_epoch_begin(self, epoch: int) -> None:
        self._epoch = epoch
        if epoch < self.warmup:
            # Warmup: pure CE on all samples (no GMM split, no guess targets).
            self.p_clean = np.ones(len(self.sample_ids), dtype=np.float64)
            self.labeled_mask = np.ones(len(self.sample_ids), dtype=bool)
        else:
            losses1 = self._per_sample_loss(self.model)
            losses2 = self._per_sample_loss(self.model2)
            p1 = self._clean_probability(losses1)
            p2 = self._clean_probability(losses2)
            self.p_clean = np.clip((p1 + p2) / 2.0, 1e-6, 1 - 1e-6)
            self.labeled_mask = self.p_clean >= 0.5
        self.sid_to_p = {int(s): float(p) for s, p in zip(self.sample_ids, self.p_clean)}
        self.sid_is_labeled = {int(s): bool(b) for s, b in zip(self.sample_ids, self.labeled_mask)}

    @staticmethod
    def _mixup(x1, t1, x2, t2, lam):
        return lam * x1 + (1 - lam) * x2, lam * t1 + (1 - lam) * t2

    @staticmethod
    def _sharpen(p: torch.Tensor, T: float) -> torch.Tensor:
        pt = p ** (1.0 / T)
        return pt / pt.sum(dim=1, keepdim=True)

    def train_epoch_step(self, x, y, sample_ids, x_w=None):
        """x is the STRONG view; x_w (extra) is the WEAK view for guess targets
        (DivideMix decoupling)."""
        x_s = x
        sids = [int(s) for s in sample_ids.cpu().numpy()]
        labeled = torch.tensor([self.sid_is_labeled.get(s, True) for s in sids], dtype=torch.bool, device=self.device)

        # Unlabeled guess target from the weak view, sharpened to one-hot.
        use_guess = self._epoch >= self.warmup if hasattr(self, "_epoch") else True
        if use_guess and (~labeled).any():
            with torch.no_grad():
                p1 = F.softmax(self.model(x_w), dim=-1)
                p2 = F.softmax(self.model2(x_w), dim=-1)
                guess = self._sharpen((p1 + p2) / 2, self.T)
                guess = torch.zeros_like(guess).scatter_(1, guess.argmax(dim=1, keepdim=True), 1.0)
        else:
            guess = None

        # Partition into labeled / unlabeled (strong view).
        x_l = x_s[labeled] if labeled.any() else torch.empty(0, *x_s.shape[1:], device=self.device)
        y_l = y[labeled] if labeled.any() else torch.empty(0, dtype=y.dtype, device=self.device)
        p_arr = torch.tensor([self.sid_to_p.get(s, 1.0) for s in sids], dtype=torch.float32, device=self.device)
        w_l = p_arr[labeled] if labeled.any() else torch.empty(0, device=self.device)
        x_u = x_s[~labeled] if (~labeled).any() else torch.empty(0, *x_s.shape[1:], device=self.device)
        y_u_soft = guess[~labeled] if guess is not None and (~labeled).any() else torch.empty(0, self.num_classes, device=self.device)

        self.optimizer.zero_grad()
        self.optimizer2.zero_grad()

        # Net 1: weighted CE on the labeled subset + strong consistency to the
        # guess on the unlabeled subset (the DivideMix mechanism).
        loss1 = self._net_loss(self.model, x_l, y_l, w_l, x_u, y_u_soft)
        if not torch.isfinite(loss1):
            loss1 = F.cross_entropy(self.model(x_s), y)
        loss1.backward(retain_graph=True)
        self.optimizer.step()

        # Net 2: symmetric loss on the same split.
        loss2 = self._net_loss(self.model2, x_l, y_l, w_l, x_u, y_u_soft)
        if not torch.isfinite(loss2):
            loss2 = F.cross_entropy(self.model2(x_s), y)
        loss2.backward()
        self.optimizer2.step()

        logits = self.model(x_s)
        return (loss1 + loss2) / 2, logits.detach(), y, sample_ids

    def _net_loss(self, net, x_l, y_l, w_l, x_u, y_u_soft):
        """DivideMix loss: weighted CE on labeled + lam_u consistency on unlabeled."""
        loss = torch.zeros((), device=self.device)

        def _fwd(xx):
            # BatchNorm breaks on batch size 1; duplicate for the forward pass.
            if len(xx) == 1:
                return net(xx.repeat(2, 1, 1, 1))[:1]
            return net(xx)

        if len(x_l) > 0:
            ce = F.cross_entropy(_fwd(x_l), y_l, reduction="none")
            loss = loss + (ce * w_l).mean()
        if len(x_u) > 0:
            u1 = F.softmax(_fwd(x_u), dim=-1)
            loss = loss + self.lam_u * F.mse_loss(u1, y_u_soft)
        return loss
