"""E10: is reversal just attenuation applied to a weak detector?

By definition A_cond = A_global - delta, so a reversal (A_cond < 0.5) needs

    delta > A_global - 0.5.

That is arithmetic, not a hypothesis.  The empirical question is whether the same
delta is available to every detector in a family: if attenuation is a property of
the *quality proxy* and not of the detector, then a family reverses exactly for
the detectors whose global AUROC is low enough to be pushed under chance, and the
whole phenomenon collapses to one number per (family, detector strength) pair.

This script tests that directly.  For every audit CSV it pairs the realised
attenuation with the realised reversal count and reports

  * delta as a function of the detector's global AUROC (is delta strength-free?);
  * the *predicted* reversal from delta and A_global versus the observed one;
  * the family-level residual, i.e. how much of the effect the "same delta"
    account leaves unexplained.

The prediction uses a leave-one-detector-out delta so a family cannot trivially
predict itself: the delta is estimated from the other detectors in the same
(dataset, proxy) family.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "revision" / "round2"

# The canonical pooled table written by run_final_analysis_set.py.  Analysis reads
# it rather than the per-audit CSVs so the detector tiers and the coverage schema
# are applied in exactly one place.
ANALYSIS_SET = OUT / "final_analysis_set.csv"
EXCLUDED_TIERS = ("excluded",)     # neighbor: saturating score resolution


def load() -> pd.DataFrame:
    if not ANALYSIS_SET.exists():
        raise SystemExit(f"missing {ANALYSIS_SET}; run run_final_analysis_set.py first")
    d = pd.read_csv(ANALYSIS_SET)
    d = d[~d.detector_tier.isin(EXCLUDED_TIERS)].copy()
    d = d[d.global_auc >= 0.5].copy()
    d["dataset"] = d.audit
    return d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="e10_weak_detector_account.csv")
    args = ap.parse_args()

    df = load()
    # Rows without a bootstrap CI carry no interval-reversal verdict.
    has_boot = df.interval_reversal.notna()
    print(f"== e10: {len(df)} cells from {df.dataset.nunique()} audits, "
          f"{int(has_boot.sum())} with a bootstrap verdict ==")

    rows = []
    for (ds, px), g in df.groupby(["dataset", "proxy"]):
        fam = g[g.eligible] if "eligible" in g.columns else g
        if not len(fam):
            continue
        for det, gg in fam.groupby("detector"):
            other = fam[fam.detector != det]
            # Leave-one-detector-out delta: the attenuation this family shows on
            # the *other* detectors is the only thing used to predict this one.
            delta_loo = float(other.delta_total.mean()) if len(other) else float("nan")
            a_g = float(gg.global_auc.mean())
            rows.append({
                "dataset": ds, "proxy": px,
                "proxy_family": gg.proxy_family.iloc[0],
                "detector": det, "n_seeds": len(gg),
                "A_global": a_g,
                "A_cond_observed": float(gg.conditioned_auc.mean()),
                "delta_observed": float(gg.delta_total.mean()),
                "delta_loo": delta_loo,
                "A_cond_predicted": a_g - delta_loo,
                "predicted_reversal": bool(a_g - delta_loo < 0.5),
                "point_rev_frac": float(gg.point_reversal.mean())
                                  if "point_reversal" in gg.columns else np.nan,
                "interval_rev_frac": float(gg.interval_reversal.mean())
                                     if "interval_reversal" in gg.columns else np.nan,
            })
    r = pd.DataFrame(rows)
    r["observed_reversal"] = r.interval_rev_frac.fillna(0) > 0.5
    r.to_csv(OUT / args.out, index=False)

    ok = r.dropna(subset=["delta_loo"])
    agree = (ok.predicted_reversal == ok.observed_reversal).mean()
    tp = int((ok.predicted_reversal & ok.observed_reversal).sum())
    fp = int((ok.predicted_reversal & ~ok.observed_reversal).sum())
    fn = int((~ok.predicted_reversal & ok.observed_reversal).sum())
    tn = int((~ok.predicted_reversal & ~ok.observed_reversal).sum())
    print(f"\n   leave-one-detector-out prediction of interval reversal:")
    print(f"     agreement {agree:.3f}   TP={tp} FP={fp} FN={fn} TN={tn}")

    print("\n== delta_total vs detector strength, by proxy family (all audits) ==")
    df["strength_bin"] = pd.cut(df.global_auc, [0.5, 0.6, 0.7, 0.8, 1.01],
                                labels=["[0.50,0.60)", "[0.60,0.70)",
                                        "[0.70,0.80)", "[0.80,1.00]"], right=False)
    t = (df.groupby(["proxy_family", "strength_bin"], observed=True)
         .agg(n_cells=("delta_total", "size"), A_global=("global_auc", "mean"),
              delta=("delta_total", "mean"), delta_sd=("delta_total", "std"),
              rev=("interval_reversal", "sum"))
         .reset_index())
    print(t.round(4).to_string(index=False))
    t.to_csv(OUT / args.out.replace(".csv", "_by_strength.csv"), index=False)

    print("\n== the deciding inequality: delta > A_global - 0.5 ==")
    print("   cells where observed interval reversal disagrees with the prediction:")
    bad = ok[ok.predicted_reversal != ok.observed_reversal]
    cols = ["dataset", "proxy", "detector", "A_global", "delta_observed",
            "delta_loo", "A_cond_predicted", "A_cond_observed",
            "predicted_reversal", "interval_rev_frac"]
    print(bad[cols].round(4).to_string(index=False) if len(bad) else "     none")


if __name__ == "__main__":
    main()
