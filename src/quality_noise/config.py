"""Configuration loading, merging, and hashing.

Configs are plain YAML. An experiment file (configs/experiments/*.yaml) is the
entry point and may reference dataset / noise / model configs by file path.
`load_config` merges user overrides over canonical `DEFAULTS`.

Every resolved run config is hashed (`hash_config`) so that results can be
traced back to the exact configuration that produced them.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULTS: dict[str, Any] = {
    "dataset": {
        "name": "cifar10",
        "root": None,  # None -> runtime default; set to /data/... on a server
        "download": True,
        "num_classes": 10,
        "val_frac": 0.1,
        "input_size": [3, 32, 32],
    },
    "noise": {
        "type": "symmetric",
        "rate": 0.4,
        "seed": 0,
        "pair_map": None,
        "base_rate": 0.05,
        "max_rate": 0.8,
    },
    "model": {
        "arch": "resnet18",
        "pretrained": False,
        "optimizer": {"name": "sgd", "lr": 0.1, "momentum": 0.9, "weight_decay": 5.0e-4},
        "scheduler": {"name": "cosine", "t_max": 200},
        "epochs": 200,
        "batch_size": 128,
        "num_workers": 4,
        "seed": 0,
        "device": "auto",
    },
    "evidence": {
        "recipe": "ema_loss_forgetting_conflict",
        "conflict_k": 20,
    },
    "quality": {
        "anchor_quantile": 0.8,
        "knn_k": 20,
        "stability_views": 2,
    },
    "matched_auc": {
        "eps": 0.1,
        "max_clean_per_noisy": 5,
    },
    "bootstrap": {"n_resamples": 1000, "ci": 0.95},
    "reliability": {"method": "knn_agreement"},
    "mixture": {"n_restarts": 5, "max_iter": 1000, "n_bins": 5},
}


def load_config(path: str | Path | None) -> dict[str, Any]:
    """Load a YAML config file, or return an empty dict if ``path`` is None."""
    if path is None:
        return {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")
    with open(p, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into ``base`` returning a new dict."""
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def resolve_config(user: dict[str, Any] | None, defaults: dict[str, Any] | None = None) -> dict[str, Any]:
    """Deep-merge user config over defaults (defaults = module DEFAULTS)."""
    base = dict(DEFAULTS) if defaults is None else defaults
    return _deep_merge(base, user or {})


def _canonical_json(obj: Any) -> str:
    """Serialize an object to a canonical, order-stable JSON string for hashing."""
    return json.dumps(obj, sort_keys=True, ensure_ascii=True, separators=(",", ":"))


def hash_config(cfg: dict[str, Any]) -> str:
    """Return a stable sha256 of a resolved config for result provenance."""
    return hashlib.sha256(_canonical_json(cfg).encode("utf-8")).hexdigest()[:16]
