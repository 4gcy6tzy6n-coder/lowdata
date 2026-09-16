"""Backfill the support-coverage columns of the E1/E4 audit CSVs.

`run_round2.evaluate_with_bootstrap` wrote

    "noisy_coverage": float(common_support(q[mask], q[~mask])[0]
                            == common_support(q[mask], q[~mask])[0])

which compares a value with itself and is therefore always 1.0.  The bug is
confined to that column: no AUROC and no bootstrap replicate in the row ever read
it, and the frozen `p0_batch` audit (which uses `evaluate_cell`) computes the
column correctly.  The function is fixed in place; this script recomputes the
correct values for the CSVs that were produced before the fix, so the delivered
tables carry a meaningful number instead of a constant.

The correct quantity is the fraction of each stratum that lies inside the common
support of Q, matching `evaluate_cell`'s `noisy_coverage` / `clean_coverage`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "revision"))

from run_p0_batch import CACHE, common_support  # noqa: E402

OUT = ROOT / "results" / "revision" / "round2"
SETTINGS = {"C100-S20": ("cifar100", "symmetric0.2"),
            "C100-A40": ("cifar100", "asymmetric0.4")}
TARGETS = {"C100-S20": "e1_combined_noN.csv", "C100-A40": "e4_a40_audit.csv"}


def support_cov(q: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    lo, hi = common_support(q[mask], q[~mask])
    keep = (q >= lo) & (q <= hi) if np.isfinite(lo) else np.zeros(len(q), bool)
    return (float((mask & keep).sum() / max(int(mask.sum()), 1)),
            float(((~mask) & keep).sum() / max(int((~mask).sum()), 1)))


def main() -> None:
    for name, fname in TARGETS.items():
        path = OUT / fname
        if not path.exists():
            print(f"[{name}] missing {path}")
            continue
        df = pd.read_csv(path)
        dataset, noise = SETTINGS[name]
        cache: dict[int, tuple[float, float]] = {}
        for seed in sorted(df.seed.unique()):
            with np.load(CACHE / dataset / noise / f"seed{seed}.npz",
                         allow_pickle=False) as z:
                mask = np.asarray(z["mask"], bool)
                qs = {p: np.asarray(z[f"sig__{p}"], np.float64) for p in df.proxy.unique()}
            for px, q in qs.items():
                cache[(int(seed), px)] = support_cov(q, mask)
        before = df.noisy_coverage.mean()
        df["noisy_coverage"] = [cache[(int(s), p)][0] for s, p in zip(df.seed, df.proxy)]
        df["clean_coverage"] = [cache[(int(s), p)][1] for s, p in zip(df.seed, df.proxy)]
        df.to_csv(path, index=False)
        print(f"[{name}] {fname}: noisy_coverage {before:.4f} -> "
              f"{df.noisy_coverage.mean():.4f}; clean_coverage "
              f"{df.clean_coverage.mean():.4f}")
        g = df.groupby("proxy")[["noisy_coverage", "clean_coverage",
                                 "matched_noisy_coverage"]].mean().round(4)
        print(g.to_string())


if __name__ == "__main__":
    main()
