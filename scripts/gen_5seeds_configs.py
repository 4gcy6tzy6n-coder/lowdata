#!/usr/bin/env python
"""gen_5seeds_configs.py — extend the four core settings to 5 seeds.

Core settings:
  CIFAR-10 sym20, CIFAR-10 asym40 (governance strength),
  CIFAR-100 sym20 (boundary case), CIFAR-10N Aggregate (real annotation noise).
Each config uses seeds [0..4]; cached artifacts for seeds 0-2 are reused
(same config hash), so only seeds 3-4 need new runs.
"""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "configs" / "experiments"


def load(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _pick(cfg: dict, keep: callable) -> dict:
    c = copy.deepcopy(cfg)
    c["noise"] = [spec for spec in c["noise"] if keep(spec)]
    c["seeds"] = [0, 1, 2, 3, 4]
    return c


def _base(spec: dict) -> str:
    return str(spec.get("file", "")).split("/")[-1]


def main() -> None:
    c10 = load(ROOT / "configs" / "experiments" / "gate_a_final.yaml")
    c100 = load(ROOT / "configs" / "experiments" / "gate_a_final_cifar100.yaml")
    c10n = load(ROOT / "configs" / "experiments" / "gate_a_cifar10n.yaml")

    variants = {
        "gate_a_final_sym20_5seeds.yaml": _pick(
            c10, lambda s: _base(s) == "symmetric.yaml" and s.get("rate") == 0.2
        ),
        "gate_a_final_asym40_5seeds.yaml": _pick(
            c10, lambda s: _base(s) == "asymmetric.yaml" and s.get("rate") == 0.4
        ),
        "gate_a_final_cifar100_sym20_5seeds.yaml": _pick(
            c100, lambda s: _base(s) == "symmetric.yaml" and s.get("rate") == 0.2
        ),
        "gate_a_cifar10n_aggre_5seeds.yaml": _pick(
            c10n, lambda s: _base(s) == "cifar10n_aggre.yaml"
        ),
    }
    for name, cfg in variants.items():
        p = OUT / name
        with open(p, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        print(f"wrote {p}  files={[_base(s) for s in cfg['noise']]}")


if __name__ == "__main__":
    main()
