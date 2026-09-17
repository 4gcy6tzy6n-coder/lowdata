"""A5 (minor revision, optional): does A_g uncertainty threaten the reversal claim?

The reversal criterion is

    A_cond < 0.5   <=>   Delta_Q > A_g - 0.5.

The reviewer notes that A_g is itself an estimator, so the margin

    R = 0.5 - A_cond = Delta_Q - (A_g - 0.5)

carries uncertainty.  This script propagates it for the three BH-significant
reversal families of the core-5 universe, and reports the three quantities a
referee would ask for:

  * Delta_Q          -- the attenuation, with a seed-bootstrap CI;
  * M = A_g - 0.5    -- the threshold the attenuation must exceed, with its CI;
  * R = Delta_Q - M  -- the margin, with its CI (this is algebraically identical to
                        0.5 - A_cond, which is checked explicitly).

A reversal is robust only if R > 0 with the CI excluding 0, and the crucial point
for the paper is that M is the *small* number: A_g is 0.70-0.71, so the threshold is
0.20-0.21 while Delta_Q is 0.22-0.31.  The uncertainty in A_g would have to move M
by more than R to overturn the claim.

Resampling: seed bootstrap, B = 10000, the same unit used elsewhere in this
revision.  The paired structure is respected by resampling whole seeds rather than
the two AUCs independently.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "revision"))
OUT = ROOT / "results" / "revision" / "round2"
CORE5 = ["ema_loss", "confidence", "aum", "confident_learning", "combined"]
DISPLAY = {"ema_loss": "EMA Loss", "confidence": "Confidence", "aum": "AUM",
           "confident_learning": "CL", "combined": "Combined"}
B_SEED = 10000
RNG_SEED = 20260917
TOL = 1e-12


def main() -> None:
    df = pd.read_csv(OUT / "final_analysis_set.csv")
    fam = pd.read_csv(OUT / "ae_family_uncertainty_core5.csv")
    rev = fam[fam.bh_reversal_significant][["detector", "proxy"]].to_numpy()

    rng = np.random.default_rng(RNG_SEED)
    rows = []
    for det, proxy in rev:
        g = df[(df.audit == "C100-S20") & (df.detector == det) & (df.proxy == proxy)]
        g = g.sort_values("seed")
        ag = g.global_auc.to_numpy(float)
        ac = g.conditioned_auc.to_numpy(float)
        delta = ag - ac                      # >0 = attenuation
        n = len(ag)
        idx = rng.integers(0, n, size=(B_SEED, n))
        # resample whole seeds: the paired structure is preserved
        d_b = delta[idx].mean(axis=1)
        m_b = ag[idx].mean(axis=1) - 0.5
        r_b = d_b - m_b

        def ci(x):
            return float(np.quantile(x, 0.025)), float(np.quantile(x, 0.975))

        d_lo, d_hi = ci(d_b)
        m_lo, m_hi = ci(m_b)
        r_lo, r_hi = ci(r_b)
        # identity check: R must equal 0.5 - A_cond exactly
        identity_gap = float(np.max(np.abs(r_b - (0.5 - ac[idx].mean(axis=1)))))
        rows.append({
            "detector": det, "detector_label": DISPLAY.get(det, det), "proxy": proxy,
            "n_seeds": n,
            "A_global": float(ag.mean()), "A_global_se": float(ag.std(ddof=1) / np.sqrt(n)),
            "A_cond": float(ac.mean()), "A_cond_se": float(ac.std(ddof=1) / np.sqrt(n)),
            "delta_Q": float(delta.mean()),
            "delta_Q_ci_lo": d_lo, "delta_Q_ci_hi": d_hi,
            "M_threshold": float(ag.mean() - 0.5),
            "M_ci_lo": m_lo, "M_ci_hi": m_hi,
            "R_margin": float(delta.mean() - (ag.mean() - 0.5)),
            "R_ci_lo": r_lo, "R_ci_hi": r_hi,
            "R_equals_half_minus_Acond": bool(identity_gap < 1e-9),
            "identity_max_gap": identity_gap,
            "R_ci_excludes_zero": bool(r_lo > 0),
            "M_shift_needed_to_overturn": float(delta.mean() - (ag.mean() - 0.5)),
        })
    r = pd.DataFrame(rows).sort_values("R_margin").reset_index(drop=True)
    r.to_csv(OUT / "ae_ag_margin_propagation.csv", index=False)

    print("== A5 / A_g uncertainty propagated onto the reversal margin ==")
    print(f"   {len(r)} BH-significant reversal families; seed bootstrap B = {B_SEED:,}")
    print(f"   identity R == 0.5 - A_cond holds for all rows: "
          f"{bool(r.R_equals_half_minus_Acond.all())} "
          f"(max gap {r.identity_max_gap.max():.2e})\n")
    print(r[["detector_label", "proxy", "A_global", "A_global_se",
             "M_threshold", "M_ci_lo", "M_ci_hi",
             "delta_Q", "delta_Q_ci_lo", "delta_Q_ci_hi",
             "R_margin", "R_ci_lo", "R_ci_hi"]].round(5).to_string(index=False))
    print("\n   reading:")
    for row in r.itertuples():
        print(f"     {row.detector_label} x {row.proxy}: threshold M = {row.M_threshold:.4f} "
              f"[{row.M_ci_lo:.4f}, {row.M_ci_hi:.4f}], attenuation {row.delta_Q:.4f}, "
              f"margin R = {row.R_margin:+.4f} [{row.R_ci_lo:+.4f}, {row.R_ci_hi:+.4f}]")
    tight = r.loc[r.R_margin.idxmin()]
    print(f"\n   tightest case: {tight.detector_label} x {tight.proxy}, R = "
          f"{tight.R_margin:.4f} with CI [{tight.R_ci_lo:.4f}, {tight.R_ci_hi:.4f}]. "
          f"A_g would have to fall by {tight.M_shift_needed_to_overturn:.4f} AUROC "
          f"({tight.M_shift_needed_to_overturn / tight.A_global_se:.1f} SE) to remove "
          "the reversal.")
    print("\n   wrote ae_ag_margin_propagation.csv")


if __name__ == "__main__":
    main()
