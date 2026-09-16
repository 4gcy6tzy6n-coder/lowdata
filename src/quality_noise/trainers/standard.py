"""Standard cross-entropy trainer.

The standard trainer is the backbone of the whole project: every training path
(standard, weighted, baselines) emits per-sample traces through the same
callback hooks, so results stay comparable. It trains on the *observed* labels
only and never touches the evaluation view.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from ..utils import set_seed, to_device


def default_train_transform(cfg_dataset: dict):
    import torchvision.transforms as T

    size = cfg_dataset.get("input_size", [3, 32, 32])[-1]
    return T.Compose(
        [
            T.ToPILImage(),
            T.RandomCrop(size, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )


def default_eval_transform(cfg_dataset: dict):
    import torchvision.transforms as T

    return T.Compose(
        [
            T.ToPILImage(),
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )


class _ImageWrapper(Dataset):
    """Wraps TrainingView (returns uint8 CHW tensors) with a torch transform."""

    def __init__(self, view: Dataset, transform) -> None:
        self.view = view
        self.transform = transform

    def __len__(self) -> int:
        return len(self.view)

    def __getitem__(self, i: int):
        x, y, sid = self.view[i]
        return self.transform(x), y, sid


@dataclass
class History:
    train_loss: list[float] = field(default_factory=list)
    train_acc: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_acc: list[float] = field(default_factory=list)

    def summarize(self) -> dict[str, float]:
        if self.val_acc:
            best = max(self.val_acc)
            best_epoch = self.val_acc.index(best)
        else:
            best = self.train_acc[-1] if self.train_acc else 0.0
            best_epoch = len(self.train_acc) - 1
        return {
            "best_val_acc": float(best),
            "best_epoch": int(best_epoch),
            "final_train_acc": float(self.train_acc[-1]) if self.train_acc else 0.0,
        }


def _make_optimizer(cfg_model: dict, model: nn.Module) -> torch.optim.Optimizer:
    opt_cfg = cfg_model.get("optimizer", {})
    name = opt_cfg.get("name", "sgd")
    lr = float(opt_cfg.get("lr", 0.1))
    if name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=float(opt_cfg.get("momentum", 0.9)),
            weight_decay=float(opt_cfg.get("weight_decay", 5.0e-4)),
        )
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=float(opt_cfg.get("weight_decay", 0.0)))
    raise ValueError(f"Unknown optimizer: {name}")


def _make_scheduler(cfg_model: dict, optimizer: torch.optim.Optimizer, epochs: int):
    sch = cfg_model.get("scheduler", {})
    name = sch.get("name", "cosine")
    if name == "cosine":
        t_max = int(sch.get("t_max", epochs))
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max)
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(sch.get("step_size", 30)), gamma=float(sch.get("gamma", 0.1)))
    if name == "none":
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _e: 1.0)
    raise ValueError(f"Unknown scheduler: {name}")


def _accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=-1)
    return float((preds == targets).float().mean().item())


def train_standard(
    cfg: dict,
    model: nn.Module,
    train_view: Dataset,
    device: torch.device,
    callbacks: list | None = None,
    val_view: Dataset | None = None,
    progress: bool = False,
    sample_weights: dict[int, float] | None = None,
) -> History:
    """Train ``model`` with (weighted) CE on the observed labels.

    ``callbacks`` receive ``on_train_batch(logits, observed_labels, sample_ids,
    epoch)`` and ``on_epoch_end()``. When ``sample_weights`` (dict sample_id ->
    weight) is given, the loss is the soft-weighted CE sum_i w_i*l_i / sum_i w_i
    (the reliability-aware governance training objective). Returns a History.
    """
    from tqdm import tqdm

    cfg_model = cfg["model"]
    cfg_dataset = cfg["dataset"]
    seed = int(cfg.get("seed", 0))
    set_seed(seed, device)

    epochs = int(cfg_model.get("epochs", 200))
    batch_size = int(cfg_model.get("batch_size", 128))
    num_workers = int(cfg_model.get("num_workers", 4))

    train_ds = _ImageWrapper(train_view, default_train_transform(cfg_dataset))
    # drop_last=False so every sample is traced each epoch (trace completeness).
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=False)

    val_loader = None
    if val_view is not None:
        val_ds = _ImageWrapper(val_view, default_eval_transform(cfg_dataset))
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = model.to(device)
    model.train()
    optimizer = _make_optimizer(cfg_model, model)
    scheduler = _make_scheduler(cfg_model, optimizer, epochs)
    history = History()
    use_weights = sample_weights is not None
    if not use_weights:
        criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        running_loss, running_acc, n_seen = 0.0, 0.0, 0
        it = tqdm(train_loader, desc=f"epoch {epoch + 1}/{epochs}") if progress else train_loader
        for x, y, sample_ids in it:
            x, y, sample_ids = to_device((x, y, sample_ids), device)
            optimizer.zero_grad()
            logits = model(x)
            if use_weights:
                w = torch.tensor([sample_weights[int(s)] for s in sample_ids.cpu().numpy()], dtype=torch.float32)
                from ..governance.weighting import weighted_ce_loss

                loss = weighted_ce_loss(logits, y, w)
            else:
                loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)
            running_acc += _accuracy(logits, y) * x.size(0)
            n_seen += x.size(0)

            for cb in callbacks or []:
                cb.on_train_batch(logits, y, sample_ids, epoch)

        for cb in callbacks or []:
            cb.on_epoch_end()

        history.train_loss.append(running_loss / max(n_seen, 1))
        history.train_acc.append(running_acc / max(n_seen, 1))

        if val_loader is not None:
            model.eval()
            v_loss, v_acc, v_n = 0.0, 0.0, 0
            with torch.no_grad():
                for x, y, _sid in val_loader:
                    x, y = to_device((x, y), device)
                    logits = model(x)
                    v_loss += criterion(logits, y).item() * x.size(0)
                    v_acc += _accuracy(logits, y) * x.size(0)
                    v_n += x.size(0)
            history.val_loss.append(v_loss / max(v_n, 1))
            history.val_acc.append(v_acc / max(v_n, 1))

        scheduler.step()

    return history
