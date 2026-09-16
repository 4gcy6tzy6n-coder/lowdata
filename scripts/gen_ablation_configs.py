#!/usr/bin/env python
"""gen_ablation_configs.py — generate ablation experiment configs for the final
method (hard50) on CIFAR-10 sym0.2 (3 seeds, the most sensitive setting).

Ablations:
  hardfrac025 / hardfrac075  — exclusion fraction sensitivity (vs 0.5 final)
  soft                        — drop hard exclusion entirely (diffuse weights)
  rel_const                   — reliability method: constant R=0.9
  ev_loss / ev_conflict / ev_forget — single-signal evidence recipes

Outputs configs/experiments/abl_*.yaml.
"""
from __future__ import annotations

from pathlib import Path

import yaml

OUT = Path(__file__).resolve().parents[1] / "configs" / "experiments"

BASE = {
    "dataset": {"file": "configs/datasets/cifar10.yaml", "name": "cifar10"},
    "noise": [{"file": "configs/noise/symmetric.yaml", "rate": 0.2}],
    "model": {"file": "configs/models/resnet18.yaml"},
    "seeds": [0, 1, 2],
    "noise_seed": "run",
    "stage": "weighted",
    "evidence": {"recipe": "ema_loss_forgetting_conflict", "conflict_k": 20},
    "quality": {"anchor_quantile": 0.8, "knn_k": 20, "stability_views": 2},
    "matched_auc": {"eps": 0.1, "max_clean_per_noisy": 5},
    "bootstrap": {"n_resamples": 1000, "ci": 0.95},
    "reliability": {
        "method": "knn_agreement",
        "weight_mode": "calibrated",
        "hard_threshold": "auto",
        "hard_frac": 0.5,
    },
    "mixture": {"n_restarts": 5, "max_iter": 1000, "n_bins": 5},
}

VARIANTS = {
    "abl_hardfrac025": {"reliability": {"hard_frac": 0.25}},
    "abl_hardfrac075": {"reliability": {"hard_frac": 0.75}},
    "abl_soft": {"reliability": {"weight_mode": "soft", "hard_threshold": None}},
    "abl_rel_const": {"reliability": {"method": "constant"}},
    "abl_ev_loss": {"evidence": {"recipe": "ema_loss"}},
    "abl_ev_conflict": {"evidence": {"recipe": "conflict"}},
    "abl_ev_forget": {"evidence": {"recipe": "forgetting"}},
}


def _merge(base: dict, patch: dict) -> dict:
    import copy

    out = copy.deepcopy(base)
    for k, v in patch.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, patch in VARIANTS.items():
        cfg = _merge(BASE, patch)
        path = OUT / f"{name}.yaml"
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        print(f"wrote {path}")
    print(f"{len(VARIANTS)} ablation configs")


if __name__ == "__main__":
    main()
