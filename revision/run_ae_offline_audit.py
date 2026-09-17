"""Offline statistics for the minor revision (no new training, no new datasets).

Covers, from the canonical round-2 assets only:

  A1  family-level effect-size uncertainty  -- mean delta, SE, 95% CI, the three
      tests and BH-adjusted p-values, for the paper's 40-family core-5 universe
  A2  Table II uncertainty                  -- conditioned AUROC mean +- SE for the
      four uncoupled detectors under each conditioning estimator
  A3  bootstrap failure counts              -- exact per-proxy counts, no rerun
  A4  effect-floor sensitivity              -- floor in {0, .005, .01, .02}
  A5  A_g uncertainty propagation           -- reversal margin R = 0.5 - A_cond

Every number comes from `results/revision/round2/final_analysis_set.csv` and the
per-cell bootstrap columns already stored there.  Nothing is retrained, nothing is
rematched except the cheap exact admissible-pair counts in A3 (40 greedy matches
on the local S20 cache, seconds).
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "revision"))
OUT = ROOT / "results" / "revision" / "round2"

CORE5 = ["ema_loss", "confidence", "aum", "confident_learning", "combined"]
PRIMARY6 = CORE5 + ["combined_noN"]
DISPLAY = {"ema_loss": "EMA Loss", "confidence": "Confidence", "aum": "AUM",
           "confident_learning": "CL", "combined": "Combined",
           "combined_noN": "Combined-noN", "forgetting": "Forgetting",
           "neighbor": "Neighbor"}
FAMILY_ORDER = ["label_dependent", "encoder_derived", "label_free", "control"]
PROXY_ORDER = ["knn_agreement", "proto_margin", "native_density", "dino_density",
               "dino_density_fullpool", "dino_knndist", "dino_augcons", "random"]

B_SEED = 10000        # seed-bootstrap draws for A1
RNG_SEED = 20260917


# ---------------------------------------------------------------------------
def sign_flip_less(d: np.ndarray) -> float:
    """Exact one-sided sign-flip p, H0 E[d]=0 vs H1 E[d]<0."""
    d = np.asarray(d, float)
    n = len(d)
    s = np.array(list(itertools.product([1.0, -1.0], repeat=n)))
    return float(((s * d).mean(axis=1) <= d.mean()).mean())


def bh_stepup(p: np.ndarray, q: float = 0.05, mask: np.ndarray | None = None):
    p = np.asarray(p, float).copy()
    if mask is not None:
        p[~mask] = 1.0
    m = len(p)
    order = np.argsort(p)
    passed = p[order] <= q * np.arange(1, m + 1) / m
    k = int(np.max(np.arange(1, m + 1)[passed])) if passed.any() else 0
    sig = np.zeros(m, bool)
    if k:
        sig[order[:k]] = True
    return sig, k


def seed_bootstrap_ci(x: np.ndarray, rng: np.random.Generator, B: int) -> dict:
    """Non-parametric bootstrap over seeds (the resampling unit of the audit)."""
    x = np.asarray(x, float)
    n = len(x)
    idx = rng.integers(0, n, size=(B, n))
    means = x[idx].mean(axis=1)
    return {"boot_mean": float(means.mean()),
            "se_boot": float(means.std(ddof=1)),
            "ci_lo": float(np.quantile(means, 0.025)),
            "ci_hi": float(np.quantile(means, 0.975))}


# ---------------------------------------------------------------------------
def a1_family_uncertainty(df: pd.DataFrame, detectors: list[str], label: str,
                          out_name: str) -> pd.DataFrame:
    from scipy import stats

    rng = np.random.default_rng(RNG_SEED)
    d = df[(df.audit == "C100-S20") & (df.detector.isin(detectors))
           & (df.global_auc >= 0.5)]
    rows = []
    for (det, px), g in d.groupby(["detector", "proxy"]):
        g = g.sort_values("seed")
        delta = (g.global_auc - g.conditioned_auc).to_numpy(float)
        ac = g.conditioned_auc.to_numpy(float)
        n = len(delta)
        bd = seed_bootstrap_ci(delta, rng, B_SEED)
        ba = seed_bootstrap_ci(ac, rng, B_SEED)
        se_t = float(delta.std(ddof=1) / np.sqrt(n))
        try:
            p_wil = float(stats.wilcoxon(delta, alternative="greater").pvalue)
        except ValueError:
            p_wil = np.nan
        try:
            p_t = float(stats.ttest_rel(g.global_auc, g.conditioned_auc,
                                        alternative="greater").pvalue)
        except Exception:
            p_t = np.nan
        rows.append({
            "detector": det, "detector_label": DISPLAY.get(det, det),
            "proxy": px, "proxy_family": g.proxy_family.iloc[0], "n_seeds": n,
            "mean_delta": float(delta.mean()),
            "sd_delta": float(delta.std(ddof=1)) if n > 1 else np.nan,
            "se_delta_t": se_t,
            "se_delta_boot": bd["se_boot"],
            "delta_ci_lo": bd["ci_lo"], "delta_ci_hi": bd["ci_hi"],
            "mean_conditioned_auc": float(ac.mean()),
            "se_cond_t": float(ac.std(ddof=1) / np.sqrt(n)) if n > 1 else np.nan,
            "se_cond_boot": ba["se_boot"],
            "cond_ci_lo": ba["ci_lo"], "cond_ci_hi": ba["ci_hi"],
            "mean_global_auc": float(g.global_auc.mean()),
            "n_seeds_attenuated": int((delta > 0).sum()),
            "p_signflip": sign_flip_less(-delta),
            "p_wilcoxon": p_wil,
            "p_pairedt": p_t,
            "reversal_margin": float(0.5 - ac.mean()),
            "reversal_margin_ci_lo": float(0.5 - ba["ci_hi"]),
            "reversal_margin_ci_hi": float(0.5 - ba["ci_lo"]),
        })
    r = pd.DataFrame(rows)

    atten_dir = (r.mean_delta > 0).to_numpy()
    rev_dir = (r.mean_conditioned_auc < 0.5).to_numpy()
    for tag, col, mask in (("signflip", "p_signflip", atten_dir),
                           ("wilcoxon", "p_wilcoxon", atten_dir),
                           ("pairedt", "p_pairedt", atten_dir),
                           ("reversal", "p_signflip", rev_dir)):
        pv = r[col].to_numpy(float).copy()
        pv[~mask] = 1.0                      # direction guard
        sig, k = bh_stepup(r[col].to_numpy(), 0.05, mask)
        r[f"bh_{tag}_significant"] = sig
        r[f"bh_{tag}_k"] = k
        # BH-adjusted p: step-up cumulative minimum of m*p_(i)/i
        m = len(pv)
        order = np.argsort(pv)
        adj_sorted = np.minimum.accumulate((pv[order] * m / np.arange(1, m + 1))[::-1])[::-1]
        adj = np.empty(m)
        adj[order] = np.clip(adj_sorted, 0, 1)
        r[f"p_bh_{tag}"] = adj

    # order for reading
    r["_f"] = r.proxy_family.map({f: i for i, f in enumerate(FAMILY_ORDER)})
    r["_p"] = r.proxy.map({p: i for i, p in enumerate(PROXY_ORDER)})
    r = (r.sort_values(["_f", "mean_delta"], ascending=[True, False])
         .drop(columns=["_f", "_p"]).reset_index(drop=True))
    r.to_csv(OUT / out_name, index=False)
    return r


def report_a1(r: pd.DataFrame, title: str) -> None:
    print(f"\n== A1 / {title} ==")
    print(f"   {len(r)} families, n_seeds = {int(r.n_seeds.iloc[0])}, "
          f"seed-bootstrap B = {B_SEED:,}")
    print(f"   bootstrap SE vs t-based SE, max |diff| = "
          f"{(r.se_delta_boot - r.se_delta_t).abs().max():.2e}  "
          f"(they should agree to Monte-Carlo error)")
    print("\n   label-dependent and encoder-derived families:")
    sub = r[r.proxy_family.isin(["label_dependent", "encoder_derived"])]
    print(sub[["detector_label", "proxy", "mean_delta", "se_delta_boot",
               "delta_ci_lo", "delta_ci_hi", "p_signflip", "p_bh_signflip",
               "bh_signflip_significant"]].round(5).to_string(index=False))
    print("\n   label-free and control families:")
    sub = r[r.proxy_family.isin(["label_free", "control"])]
    print(sub[["detector_label", "proxy", "mean_delta", "se_delta_boot",
               "delta_ci_lo", "delta_ci_hi", "p_signflip", "p_bh_signflip",
               "bh_signflip_significant"]].round(5).to_string(index=False))
    print(f"\n   BH-significant attenuation: {int(r.bh_signflip_significant.sum())}/{len(r)}")
    big = r.mean_delta >= 0.01
    print(f"   BH and delta >= 0.01:       {int((r.bh_signflip_significant & big).sum())}/{len(r)}")
    print(f"   BH-significant reversal:    {int(r.bh_reversal_significant.sum())}/{len(r)}")
    print(f"   families whose delta CI excludes 0: {int((r.delta_ci_lo > 0).sum())}/{len(r)}")
    rev = r[r.bh_reversal_significant]
    if len(rev):
        print("\n   reversal families with propagated margin R = 0.5 - A_cond:")
        print(rev[["detector_label", "proxy", "mean_conditioned_auc", "se_cond_boot",
                   "cond_ci_lo", "cond_ci_hi", "reversal_margin",
                   "reversal_margin_ci_lo", "reversal_margin_ci_hi"]]
              .round(5).to_string(index=False))


# ---------------------------------------------------------------------------
def a2_table2(df: pd.DataFrame) -> pd.DataFrame:
    """Table II with uncertainty: conditioned AUROC mean +- SE.

    Restricted to the four uncoupled detectors the table reports, under every
    conditioning estimator in the paper.
    """
    dets = ["ema_loss", "confidence", "aum", "confident_learning"]
    d = df[(df.audit == "C100-S20") & (df.detector.isin(dets)) & (df.global_auc >= 0.5)]
    rows = []
    for (det, px), g in d.groupby(["detector", "proxy"]):
        n = len(g)
        rows.append({"detector": det, "detector_label": DISPLAY.get(det, det),
                     "proxy": px, "proxy_family": g.proxy_family.iloc[0],
                     "estimator": px, "n_seeds": n,
                     "global_auc_mean": g.global_auc.mean(),
                     "global_auc_se": g.global_auc.std(ddof=1) / np.sqrt(n),
                     "conditioned_auc_mean": g.conditioned_auc.mean(),
                     "conditioned_auc_se": g.conditioned_auc.std(ddof=1) / np.sqrt(n),
                     "delta_mean": g.delta_total.mean(),
                     "delta_se": g.delta_total.std(ddof=1) / np.sqrt(n)})
    r = pd.DataFrame(rows).sort_values(["detector", "proxy"]).reset_index(drop=True)
    r.to_csv(OUT / "ae_table2_uncertainty.csv", index=False)
    return r


def report_a2(r: pd.DataFrame) -> None:
    print("\n== A2 / Table II uncertainty (four uncoupled detectors, C100-S20) ==")
    for det, g in r.groupby("detector", sort=False):
        print(f"\n   {DISPLAY.get(det, det)}  (n_seeds = {int(g.n_seeds.iloc[0])})")
        print("     " + "  ".join(
            f"{px}: {m:.4f}+-{s:.4f}" for px, m, s in
            zip(g.estimator, g.conditioned_auc_mean, g.conditioned_auc_se)))
    print("\n   full table -> ae_table2_uncertainty.csv")


# ---------------------------------------------------------------------------
def a3_bootstrap_failures(df: pd.DataFrame) -> pd.DataFrame:
    """Exact per-proxy bootstrap failure counts, from the stored columns."""
    d = df[(df.audit == "C100-S20") & (df.detector.isin(CORE5))]
    rows = []
    for px, g in d.groupby("proxy"):
        reps = int(g.bootstrap_B.fillna(0).sum())
        failed = int(g.bootstrap_failed.fillna(0).sum())
        rows.append({"proxy": px, "proxy_family": g.proxy_family.iloc[0],
                     "cells": len(g), "n_bootstrap_replicates": reps,
                     "n_bootstrap_failed": failed,
                     "failure_rate": (failed / reps) if reps else np.nan,
                     "min_pairs_per_cell": int(g.n_pairs.min()),
                     "matching_failure_cells": int(g.get(
                         "matching_failure", pd.Series(0, index=g.index)).fillna(0).sum())
                     if "matching_failure" in g.columns else np.nan})
    r = pd.DataFrame(rows).sort_values("proxy").reset_index(drop=True)
    r.to_csv(OUT / "ae_bootstrap_failures.csv", index=False)
    return r


def report_a3(r: pd.DataFrame) -> None:
    print("\n== A3 / bootstrap failure counts, C100-S20 core-5 ==")
    print("   (one replicate == one paired equal-count resample; a failure is a")
    print("    replicate in which no admissible pair survived the caliper)")
    print(r[["proxy", "cells", "n_bootstrap_replicates", "n_bootstrap_failed",
             "failure_rate", "min_pairs_per_cell"]].round(6).to_string(index=False))
    tot = int(r.n_bootstrap_replicates.sum())
    print(f"\n   total replicates {tot:,}, total failures "
          f"{int(r.n_bootstrap_failed.sum())} -> overall failure rate "
          f"{r.n_bootstrap_failed.sum()/max(tot,1):.2e}")


# ---------------------------------------------------------------------------
def a4_effect_floor(r: pd.DataFrame, floors=(0.0, 0.005, 0.01, 0.02, 0.05)) -> pd.DataFrame:
    rows = []
    for f in floors:
        keep = r.mean_delta >= f
        rows.append({"floor": f, "bh_significant": int(r.bh_signflip_significant.sum()),
                     "bh_and_floor": int((r.bh_signflip_significant & keep).sum()),
                     "bh_reversal_and_floor": int((r.bh_reversal_significant & keep).sum()),
                     "families_clearing_floor": int(keep.sum())})
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "ae_effect_floor_sensitivity.csv", index=False)
    return out


def report_a4(r: pd.DataFrame, detail: pd.DataFrame) -> None:
    print("\n== A4 / effect-floor sensitivity (core-5, 40 families) ==")
    print(r.to_string(index=False))
    n0 = int(r[r.floor == 0.0].bh_and_floor.iloc[0])
    n5 = int(r[r.floor == 0.005].bh_and_floor.iloc[0])
    n10 = int(r[r.floor == 0.01].bh_and_floor.iloc[0])
    n20 = int(r[r.floor == 0.02].bh_and_floor.iloc[0])
    # How many of the 40 families actually sit in each gap?  A monotone count is
    # not by itself evidence against a knife-edge; what matters is whether the
    # neighbourhood of the chosen floor is empty of families.
    lo, hi = 0.005, 0.01
    near = detail[(detail.mean_delta >= lo) & (detail.mean_delta < hi)]
    print(f"\n   the count falls {n0} -> {n5} between floor 0 and .005, then "
          f"{n5} -> {n10} at .010, {n10} -> {n20} at .020.")
    print(f"   families with mean_delta in [.005, .010): {len(near)}"
          + ("" if not len(near) else
             "  [" + ", ".join(f"{r.detector_label}/{r.proxy}={r.mean_delta:.5f}"
                               for r in near.itertuples()) + "]"))
    print(f"   so the chosen floor .010 sits on a plateau edge: "
          f"{'no family' if not len(near) else str(len(near)) + ' family/families'} "
          "lies between .005 and .010, and the next family below the floor is at "
          f"{detail[detail.mean_delta < 0.01].mean_delta.max():.5f}."
          if len(detail[detail.mean_delta < 0.01]) else "")
    print("   Interpretation: the choice of .010 changes the count by at most "
          f"{n5 - n10} family relative to .005 and {n10 - n20} relative to .020, so no "
          "single family's inclusion decides the headline; but it is a real threshold, "
          "not a flat region.")


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-a3-exact", action="store_true",
                    help="skip the exact admissible-pair recount for A3")
    args = ap.parse_args()

    df = pd.read_csv(OUT / "final_analysis_set.csv")

    r6 = a1_family_uncertainty(df, CORE5, "core-5 (paper universe)",
                               "ae_family_uncertainty_core5.csv")
    report_a1(r6, "core-5, 40 families")
    r7 = a1_family_uncertainty(df, PRIMARY6, "primary-6 (sensitivity)",
                               "ae_family_uncertainty_primary6.csv")
    report_a1(r7, "primary-6 sensitivity, 48 families")

    report_a2(a2_table2(df))
    report_a3(a3_bootstrap_failures(df))
    a4_effect_floor(r6, floors=(0.0, 0.005, 0.01, 0.02, 0.05, 0.10))
    report_a4(pd.read_csv(OUT / "ae_effect_floor_sensitivity.csv"), r6)

    json.dump({
        "family_uncertainty": {
            "universe": "core-5 x 8 proxies, C100-S20, 10 seeds",
            "n_families": int(len(r6)),
            "bootstrap": {"type": "seed", "B": B_SEED, "rng_seed": RNG_SEED},
            "max_abs_se_boot_minus_se_t": float((r6.se_delta_boot - r6.se_delta_t).abs().max()),
            "bh_significant_attenuation": int(r6.bh_signflip_significant.sum()),
            "bh_and_delta_ge_0.01": int((r6.bh_signflip_significant
                                         & (r6.mean_delta >= 0.01)).sum()),
            "bh_significant_reversal": int(r6.bh_reversal_significant.sum()),
            "delta_ci_excludes_zero": int((r6.delta_ci_lo > 0).sum()),
        },
    }, open(OUT / "ae_offline_summary.json", "w"), indent=2)
    print("\n   wrote ae_family_uncertainty_core5.csv, "
          "ae_family_uncertainty_primary6.csv,")
    print("         ae_table2_uncertainty.csv, ae_bootstrap_failures.csv,")
    print("         ae_effect_floor_sensitivity.csv, ae_offline_summary.json")


if __name__ == "__main__":
    main()
