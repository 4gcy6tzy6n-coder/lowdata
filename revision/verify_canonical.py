#!/usr/bin/env python3
"""Independent check of the canonical round-2 statistics.

Deliberately does NOT import `run_final_analysis_set`: it re-derives the headline
multiplicity table from the audit CSVs with its own implementation, so a bug shared
between producer and checker cannot hide.  Exits non-zero on any mismatch.

Usage
-----
    python revision/verify_canonical.py [--root .]

What it checks
--------------
1. coverage schema   -- no ambiguous name, all four canonical columns, in every audit
2. detector tiers    -- final_analysis_set.csv tiers match the frozen registry
3. audit integrity   -- delta_total == global_auc - conditioned_auc where finite
4. multiplicity      -- recomputed 28/40, 18/40, 3/40 for C100-S20 core-5 (the
                        paper's universe) and 34/48, 21/48, 5/48 for primary-6
5. cross-audit core-5 -- recomputed per-audit BH counts
6. analysis pool     -- 1,440 cells and 84 interval reversals after tier + AUC gates
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PRIMARY6 = ["ema_loss", "confidence", "aum", "confident_learning",
            "combined", "combined_noN"]
CORE5 = ["ema_loss", "confidence", "aum", "confident_learning", "combined"]
COARSE = ["forgetting"]
EXCLUDED = ["neighbor"]
CANON = ["common_noisy_coverage", "common_clean_coverage",
         "matched_noisy_coverage", "matched_clean_coverage"]
AMBIGUOUS = ["noisy_coverage", "clean_coverage"]

AUDITS = {
    "C100-S20": ("results/revision/round2/e1_combined_noN.csv", 10),
    "C100-A40": ("results/revision/round2/e4_a40_audit.csv", 5),
    "C100N-human": ("results/revision/round2/e9_c100n_audit.csv", 5),
    "C100-S20-frozen": ("results/revision/p0_batch/j1_bootstrap_all_proxies.csv", 10),
}
EXPECT_PRIMARY = {"signflip": 28, "wilcoxon": 28, "pairedt": 28}
EXPECT_PRIMARY_EFFECT = 18
EXPECT_REVERSAL = 3
EXPECT_CORE5 = {"C100-S20": 28, "C100-S20-frozen": 28, "C100N-human": 26, "C100-A40": 0}

_fail = 0


def check(cond: bool, label: str) -> None:
    global _fail
    if not cond:
        _fail += 1
    print(f"  {'OK  ' if cond else 'FAIL'} {label}")


def sign_flip_less(d: np.ndarray) -> float:
    """Exact one-sided p for H0 E[d]=0 vs H1 E[d]<0, from scratch."""
    d = np.asarray(d, float)
    n = len(d)
    s = np.array(list(itertools.product([1.0, -1.0], repeat=n)))
    return float(((s * d).mean(axis=1) <= d.mean()).mean())


def bh(p: np.ndarray, q: float = 0.05) -> np.ndarray:
    m = len(p)
    order = np.argsort(p)
    passed = p[order] <= q * np.arange(1, m + 1) / m
    k = int(np.max(np.arange(1, m + 1)[passed])) if passed.any() else 0
    sig = np.zeros(m, bool)
    if k:
        sig[order[:k]] = True
    return sig


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    args = ap.parse_args()
    root = Path(args.root).resolve()
    print(f"verifying canonical round-2 assets under {root}\n")

    print("1. coverage schema")
    loaded = {}
    for name, (rel, _) in AUDITS.items():
        df = pd.read_csv(root / rel)
        loaded[name] = df
        cols = set(df.columns)
        check(not (set(AMBIGUOUS) & cols), f"{name}: no ambiguous coverage name")
        check(set(CANON) <= cols, f"{name}: all four canonical coverage columns")

    print("\n2. detector tiers in final_analysis_set.csv")
    fas = pd.read_csv(root / "results/revision/round2/final_analysis_set.csv")
    tiers = dict(zip(fas.detector, fas.detector_tier))
    check(all(tiers.get(d) == "primary" for d in PRIMARY6), "primary-6 tagged primary")
    check(all(tiers.get(d) == "coarse_secondary" for d in COARSE), "forgetting tagged coarse_secondary")
    check(all(tiers.get(d) == "excluded" for d in EXCLUDED), "neighbor tagged excluded")

    print("\n3. audit integrity: delta_total == global_auc - conditioned_auc")
    for name, df in loaded.items():
        d = df.dropna(subset=["delta_total", "global_auc", "conditioned_auc"])
        resid = (d.delta_total - (d.global_auc - d.conditioned_auc)).abs().max()
        check(resid < 1e-12, f"{name}: max identity residual {resid:.2e}")

    print("\n4. multiplicity, recomputed from the audit CSVs (C100-S20, core-5)")
    s20 = loaded["C100-S20"]
    s20 = s20[(s20.global_auc >= 0.5) & s20.detector.isin(CORE5)]
    rows = []
    for (det, px), g in s20.groupby(["detector", "proxy"]):
        g = g.sort_values("seed")
        delta = (g.global_auc - g.conditioned_auc).to_numpy(float)
        ac = g.conditioned_auc.to_numpy(float)
        rows.append({"detector": det, "proxy": px,
                     "delta": delta.mean(), "ac": ac.mean(),
                     "p": sign_flip_less(-delta),
                     "p_rev": sign_flip_less(ac - 0.5)})
    r = pd.DataFrame(rows)
    check(len(r) == 40, f"40 families (got {len(r)})")
    atten_dir = (r.delta > 0).to_numpy()
    p = r.p.to_numpy().copy(); p[~atten_dir] = 1.0
    sig = bh(p)
    check(int(sig.sum()) == EXPECT_PRIMARY["signflip"],
          f"sign-flip BH-significant = {int(sig.sum())} (expect {EXPECT_PRIMARY['signflip']})")
    gated = sig & (r.delta.to_numpy() >= 0.01)
    check(int(gated.sum()) == EXPECT_PRIMARY_EFFECT,
          f"BH AND delta>=0.01 = {int(gated.sum())} (expect {EXPECT_PRIMARY_EFFECT})")
    # reversal claims need A_cond below chance, so the direction guard uses ac
    p_rev = r.p_rev.to_numpy().copy()
    p_rev[(r.ac.to_numpy() >= 0.5)] = 1.0
    pr = bh(p_rev)
    check(int(pr.sum()) == EXPECT_REVERSAL,
          f"reversal BH-significant = {int(pr.sum())} (expect {EXPECT_REVERSAL})")
    rev_fam = sorted(zip(r.detector[pr], r.proxy[pr]))
    check(rev_fam == [("aum", "native_density"), ("combined", "knn_agreement"),
                      ("combined", "proto_margin")],
          f"reversal families = {rev_fam}")

    print("\n4a. sensitivity: primary-6 (Combined-noN gets its own families)")
    s6 = loaded["C100-S20"]
    s6 = s6[(s6.global_auc >= 0.5) & s6.detector.isin(PRIMARY6)]
    r6 = []
    for (det, px), g in s6.groupby(["detector", "proxy"]):
        g = g.sort_values("seed")
        delta = (g.global_auc - g.conditioned_auc).to_numpy(float)
        r6.append({"delta": delta.mean(), "p": sign_flip_less(-delta),
                   "p_rev": sign_flip_less(g.conditioned_auc.to_numpy(float) - 0.5)})
    r6 = pd.DataFrame(r6)
    check(len(r6) == 48, f"48 families (got {len(r6)})")
    p6 = r6.p.to_numpy().copy(); p6[r6.delta.to_numpy() <= 0] = 1.0
    s6sig = bh(p6)
    check(int(s6sig.sum()) == 34, f"sign-flip BH-significant = {int(s6sig.sum())} (expect 34)")
    check(int((s6sig & (r6.delta.to_numpy() >= 0.01)).sum()) == 21,
          f"BH AND delta>=0.01 = {int((s6sig & (r6.delta.to_numpy() >= 0.01)).sum())} (expect 21)")
    p6r = r6.p_rev.to_numpy().copy()
    check(int(bh(p6r).sum()) == 5, f"reversal BH-significant = {int(bh(p6r).sum())} (expect 5)")

    print("\n5. cross-audit core-5 (recomputed)")
    for name, (_, n_seeds) in AUDITS.items():
        df = loaded[name]
        if "combined_variant" in df.columns:
            df = df[df.combined_variant == "paper"]
        df = df[(df.global_auc >= 0.5) & df.detector.isin(CORE5)]
        rr = []
        for (det, px), g in df.groupby(["detector", "proxy"]):
            g = g.sort_values("seed")
            delta = (g.global_auc - g.conditioned_auc).to_numpy(float)
            rr.append({"delta": delta.mean(), "p": sign_flip_less(-delta)})
        rr = pd.DataFrame(rr)
        pp = rr.p.to_numpy().copy(); pp[rr.delta.to_numpy() <= 0] = 1.0
        n = int(bh(pp).sum())
        check(n == EXPECT_CORE5[name], f"{name} (n={n_seeds}): {n}/40 (expect {EXPECT_CORE5[name]})")

    print("\n6. analysis pool")
    pool = fas[(fas.detector_tier != "excluded") & (fas.global_auc >= 0.5)]
    check(len(pool) == 1440, f"1,440 cells (got {len(pool)})")
    check(int(pool.interval_reversal.sum()) == 84,
          f"84 interval reversals (got {int(pool.interval_reversal.sum())})")

    print(f"\n{'ALL CHECKS PASSED' if _fail == 0 else str(_fail) + ' CHECK(S) FAILED'}")
    return 1 if _fail else 0


if __name__ == "__main__":
    sys.exit(main())
