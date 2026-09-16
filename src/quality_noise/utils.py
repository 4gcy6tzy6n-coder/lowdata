"""Shared utilities: device detection, RNG seeding, path helpers, atomic writes."""
from __future__ import annotations

import json
import os
import random
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch


def get_device(prefer: str = "auto") -> torch.device:
    """Resolve the compute device. ``auto`` -> cuda > mps > cpu."""
    if prefer != "auto":
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int, device: torch.device | None = None) -> None:
    """Seed Python, NumPy, and PyTorch deterministically."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if device is not None and device.type == "cuda":
        torch.cuda.manual_seed(seed)
    # Keep convolution algorithms deterministic on CUDA where supported.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def to_device(batch: Any, device: torch.device) -> Any:
    """Move a tuple/list of tensors to ``device``, preserving structure."""
    if isinstance(batch, (tuple, list)):
        return tuple(t.to(device, non_blocking=True) if torch.is_tensor(t) else t for t in batch)
    if torch.is_tensor(batch):
        return batch.to(device, non_blocking=True)
    return batch


def make_results_dirs(cfg: dict[str, Any]) -> dict[str, Path]:
    """Create and return the standard results directories for a run."""
    from .config import PROJECT_ROOT

    root = Path(cfg.get("results_root") or PROJECT_ROOT / "results")
    dirs = {
        "data": root / "data",
        "traces": root / "traces",
        "quality": root / "quality",
        "detectability": root / "detectability",
        "estimator": root / "estimator",
        "baselines": root / "baselines",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def write_json_atomic(path: Path, data: Any) -> None:
    """Write JSON atomically (tmp file + rename) so partial writes never appear."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=True, indent=2, sort_keys=True)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
