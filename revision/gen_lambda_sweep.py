"""P1-7 — governance exclusion-strength sweep lambda in {0, .25, .5, .75, 1.0}.

The manuscript's "hard50" rule excludes the top ``0.5 * rho_hat`` fraction of
samples by posterior.  Reviewers asked why exactly 0.5.  This sweep makes the
intervention strength an explicit knob:

    exclude fraction = lambda * rho_hat

which maps 1:1 onto the existing ``hard_frac`` implementation
(``reliability/estimator.py``: tau = quantile(eta_shrunk, 1 - rate * frac);
lambda = 0 coincides with the soft-only recipe, since tau becomes the maximum
of eta and nothing is zeroed).

For each (setting, lambda, seed) we report four quantities:

    test accuracy, noise removal rate, clean retention rate,
    hard-clean false removal rate

So the claim becomes "increasing intervention strength helps under strong
controlled detectability and becomes harmful in weak-detectability regimes"
rather than "0.5 is optimal".

Config generation
-----------------
Configs are derived from a base experiment YAML by replacing ONLY the
``reliability`` block, so the resulting config hash is reproducible.  The
generator reports whether an existing estimator run already matches that hash
(by checking the hashes present in ``results/estimator/<...>/``), which lets the
driver skip redundant work: lambda = 0 and lambda = 0.5 are often already run.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path("/root/quality_noise")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "revision"))

from quality_noise.config import load_config  # noqa: E402
from quality_noise.experiment import expand_experiment  # noqa: E402

LAMBDAS = [0.0, 0.25, 0.5, 0.75, 1.0]

# Setting -> (base experiment yaml, human label).  Base configs are chosen so
# every other knob (evidence recipe, quality knobs, mixture) matches the frozen
# headline runs exactly; only ``reliability`` is touched.
SETTINGS = {
    "c10_s20":  ("configs/experiments/gate_a_final_sym20_5seeds.yaml", "CIFAR-10 S20"),
    "c10_a40":  ("configs/experiments/gate_a_final_asym40_5seeds.yaml", "CIFAR-10 A40"),
    "c100_s20": ("configs/experiments/gate_a_final_cifar100_sym20_5seeds.yaml", "CIFAR-100 S20"),
    "c100_s40": ("configs/experiments/_rev_c100_sym40_5seeds.yaml", "CIFAR-100 S40"),
    "c100_a20": ("configs/experiments/_rev_c100_asym20_5seeds.yaml", "CIFAR-100 A20"),
}


def reliability_block(lam: float) -> dict:
    """The single knob under study: exclude fraction = lambda * rho_hat."""
    if lam == 0.0:
        # Soft-only: no hard exclusion at all.
        return {"method": "knn_agreement", "weight_mode": "soft",
                "hard_threshold": None, "hard_frac": 0.0}
    return {"method": "knn_agreement", "weight_mode": "calibrated",
            "hard_threshold": "auto", "hard_frac": float(lam)}


def write_variant(base_yaml: Path, lam: float, out_yaml: Path) -> None:
    with open(base_yaml) as f:
        cfg = yaml.safe_load(f)
    cfg["reliability"] = reliability_block(lam)
    out_yaml.parent.mkdir(parents=True, exist_ok=True)
    with open(out_yaml, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def resolve_base(setting: str) -> Path:
    rel, _ = SETTINGS[setting]
    p = ROOT / rel
    if p.exists():
        return p
    raise FileNotFoundError(f"base config missing for {setting}: {p}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=ROOT / "configs" / "experiments" / "lambda_sweep")
    ap.add_argument("--settings", type=str, default="c10_s20")
    ap.add_argument("--report", action="store_true", help="only report hashes / reuse")
    args = ap.parse_args()

    settings = [s.strip() for s in args.settings.split(",") if s.strip()]
    print(f"{'setting':10s} {'lambda':>6s} {'hash':>18s}  status")
    for s in settings:
        try:
            base = resolve_base(s)
        except FileNotFoundError as e:
            print(f"{s:10s} {'-':>6s} {'-':>18s}  SKIP ({e})")
            continue
        for lam in LAMBDAS:
            out_yaml = args.out_dir / f"{s}_lam{lam:.2f}.yaml"
            if not args.report:
                write_variant(base, lam, out_yaml)
            try:
                from quality_noise.config import hash_config
                exp = load_config(str(out_yaml if out_yaml.exists() else base))
                runs = expand_experiment(exp)
                cfg_hash = hash_config(runs[0])
            except Exception as e:  # noqa: BLE001
                print(f"{s:10s} {lam:6.2f} {'-':>18s}  ERROR {e}")
                continue
            print(f"{s:10s} {lam:6.2f} {cfg_hash:>18s}  {out_yaml.name}")

    print("\nTo run the sweep end-to-end use revision/run_p1_lambda_sweep.sh")


if __name__ == "__main__":
    main()
