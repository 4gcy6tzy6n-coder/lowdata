#!/usr/bin/env python
"""gen_oracle_configs.py — generate the P0-② oracle-vs-estimated noise-rate
experiment configs from gate_a_final.yaml.

Variants (all CIFAR-10, 4 noise settings x 3 seeds):
  gate_a_final_oracle.yaml    — rate_source=config (uses the oracle rho)
  gate_a_final_estimated.yaml — rate_source=estimated (Otsu rho_hat, no oracle)
  gate_a_final_mis_plus.yaml  — oracle rho + 0.10 (misspecification check)
  gate_a_final_mis_minus.yaml — oracle rho - 0.10
"""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "configs" / "experiments" / "gate_a_final.yaml"
OUT = ROOT / "configs" / "experiments"


def load(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _bump_rate(noise_list: list[dict], delta: float) -> list[dict]:
    out = []
    for spec in noise_list:
        s = dict(spec)
        if "rate" in s:
            s["rate"] = round(float(s["rate"]) + delta, 3)
        out.append(s)
    return out


def _set_reliability(cfg: dict, **kwargs) -> dict:
    c = copy.deepcopy(cfg)
    c.setdefault("reliability", {})
    c["reliability"].update(kwargs)
    return c


def main() -> None:
    base = load(BASE)
    base.setdefault("reliability", {})
    # Existing final uses rate_source default (config) + hard50. Explicitly tag.
    variants = {
        "gate_a_final_oracle.yaml": _set_reliability(base, rate_source="config"),
        "gate_a_final_estimated.yaml": _set_reliability(base, rate_source="estimated"),
        "gate_a_final_mis_plus.yaml": _set_reliability(
            {**base, "noise": _bump_rate(base["noise"], +0.10)}, rate_source="config"
        ),
        "gate_a_final_mis_minus.yaml": _set_reliability(
            {**base, "noise": _bump_rate(base["noise"], -0.10)}, rate_source="config"
        ),
    }
    for name, cfg in variants.items():
        p = OUT / name
        with open(p, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
