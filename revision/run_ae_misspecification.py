"""A3 (minor revision): Gaussian misspecification sensitivity.

The reviewer's one explicitly requested new simulation.  The baseline Gaussian
model is

    Z ~ Bernoulli(rho=0.2)
    D ~ N(0, 1)
    E = beta*Z + gamma*D + eps_E,     eps_E ~ N(0, sigma_E^2=1)
    Q = D + lambda*Z + eps_Q,         eps_Q ~ N(0, sigma_Q^2=0.09)

with N = 20000 and beta calibrated so the *global* AUROC is 0.70.  Two
misspecification settings are added, each re-calibrated to the same 0.70:

  M1  nonlinear difficulty:  Q = f(D) + lambda*Z + eps_Q  with  f(D) = D + 0.3 D^3
      (E unchanged), i.e. the quality proxy is a curved function of difficulty, so
      the rank statistic the audit uses sees a distorted Q.

  M2  correlated difficulty: D | Z=z ~ N(kappa*z, 1), kappa in {0.25, 0.5}
      (so D and Z are correlated), E and Q unchanged.  This is exactly the
      conditional-mean formulation the reviewer proposed; the calibration absorbs
      gamma*kappa into E's conditional mean, which leaves the residual structure
      identical to shifting D by -kappa*z, so the model is internally consistent.

Three questions, answered from the same grid as the main text (lambda in [0,5] step
0.1, 100 repeats per cell, gamma in {0.25, 0.5, 1.0, 1.5}):

  1. does the conditioned AUROC still fall as lambda grows?
  2. is there still a crossing / reversal?
  3. does a larger gamma bring the crossing earlier?

Calibration is analytic.  Both E|Z and Q|Z are Gaussian, so with a 1:1 variance
ratio the global AUROC is exactly Phi(delta / sqrt(2 sigma^2)) where delta is the
noisy-minus-clean mean gap and sigma^2 the common conditional variance; beta is
found by bisection.  Without this the settings would not be comparable, because
the cubic term alone already changes the global AUROC.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "revision"))

from run_p0_batch import MAX_CLEAN, matched_pairs, pairwise_auc, rank_auc  # noqa: E402

OUT = ROOT / "results" / "revision" / "round2"

RHO, SIG_E, SIG_Q, N = 0.2, 1.0, 0.3, 20000
CUBIC = 0.3
TARGET_AUC = 0.70
GAMMAS = [0.25, 0.5, 1.0, 1.5]
KAPPAS = [0.25, 0.5]
LAMS = np.round(np.linspace(0.0, 5.0, 51), 4)


def analytic_global_auc(beta: float, gamma: float, kappa: float) -> float:
    """Global AUROC of the baseline (and M2) model, in closed form.

    Generative structure -- E6's, unchanged:

        Z ~ Bernoulli(rho)
        d ~ N(0, 1)                         shared latent "difficulty"
        D = d + kappa*Z                     M2 only; kappa = 0 in the baseline
        E = beta*Z + gamma*D + eps_E,       eps_E ~ N(0, sigma_E^2)
        Q = f(D) + lambda*Z + eps_Q,        eps_Q ~ N(0, sigma_Q^2)

    Z is binary outside both E and Q's noise, so E | Z is Gaussian and

        Var(E | Z) = gamma^2 + sigma_E^2      (the shared d contributes)
        E[E | Z=1] - E[E | Z=0] = beta + gamma*kappa
        AUROC = Phi(gap / sqrt(2 Var(E|Z)))

    Equivalently beta = 2 Phi^-1(AUC) sqrt(gamma^2 + sigma_E^2) - gamma*kappa, which
    reproduces E6's main-text calibration (gamma=0.25 -> 0.7644) when kappa = 0.
    """
    return float(norm.cdf((beta + gamma * kappa) / np.sqrt(2.0 * (gamma ** 2 + SIG_E ** 2))))


def calibrate_beta(gamma: float, kappa: float, target: float = TARGET_AUC) -> float:
    b = float(norm.ppf(target) * np.sqrt(2.0 * (gamma ** 2 + SIG_E ** 2)))
    return b - gamma * kappa


def q_residual_sd(sigma_q: float, nonlinear: bool) -> float:
    """sd of (Q - E[Q|Z]) per unit of the D-signal.

    The residual of Q given Z is deterministic in D plus the independent noise:

        baseline:  Q - l*Z = D + eps_Q                     -> sd = sqrt(1 + s^2)
        M1:        Q - l*Z = D + 0.3 D^3 + eps_Q           -> sd = sqrt(1 + 6a + 15a^2 + s^2),
                                                              a = 0.3

    The cubic inflates this (1.0440 -> 2.0591) more than it inflates the marginal
    spread of f(D) (1.0000 -> 2.0378), because f' = 1 + 0.9 D^2 is 1.18 at D = 0 but
    5.4 at |D| = 2.2, so the curve adds curvature everywhere and spread mostly in the
    tails.  Both
    the caliper (0.1 sd(Q)) and the match quality are governed by this residual, so
    it -- not the marginal sd of Q -- is what must be held fixed across settings.
    """
    if not nonlinear:
        return float(np.sqrt(1.0 + sigma_q ** 2))
    a = CUBIC
    return float(np.sqrt(1.0 + 6.0 * a + 15.0 * a ** 2 + sigma_q ** 2))


def make_data(rng: np.random.Generator, beta: float, gamma: float, kappa: float,
              nonlinear: bool, sigma_q: float, n: int = N):
    z = rng.random(n) < RHO
    d_lat = rng.normal(size=n)
    d = d_lat + kappa * z                     # shared latent drives both E and Q
    e = beta * z + gamma * d + SIG_E * rng.normal(size=n)
    f = d + CUBIC * d ** 3 if nonlinear else d
    return z, d, e, f, sigma_q * rng.normal(size=n)


def run_setting(name: str, gamma: float, kappa: float, nonlinear: bool,
                reps: int, rng_seed0: int) -> list[dict]:
    beta = calibrate_beta(gamma, kappa)
    realised = analytic_global_auc(beta, gamma, kappa)
    # Keep Q's within-stratum spread at the baseline's level in every setting, so
    # the three settings differ in *structure* and not in signal-to-noise.
    # Normalise so the residual spread of Q matches the baseline's, i.e. the
    # caliper admits the same number of candidates and the settings differ in
    # *structure* rather than in signal-to-noise.
    sigma_eff = SIG_Q / q_residual_sd(SIG_Q, nonlinear)
    rows = []
    for rep in range(reps):
        rng = np.random.default_rng(rng_seed0 + 10_000 * rep + int(gamma * 100))
        z, d, e, f, eps_q = make_data(rng, beta, gamma, kappa, nonlinear, sigma_eff)
        for lam in LAMS:
            q = f + lam * z + eps_q
            ni, ci = matched_pairs(e, z, q, 0.10 * float(q.std()), MAX_CLEAN)
            a_audit = pairwise_auc(e, ni, ci)
            rows.append({"setting": name, "gamma": gamma, "kappa": kappa,
                         "beta": beta, "target_auc": TARGET_AUC,
                         "analytic_global_auc": realised, "sigma_eff": sigma_eff,
                         "lam": lam, "rep": rep,
                         "global_auc": rank_auc(z, e), "audit_auc": a_audit,
                         "reversal": bool(np.isfinite(a_audit) and a_audit < 0.5)})
    return rows


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (setting, gamma, kappa), g in df.groupby(["setting", "gamma", "kappa"]):
        curve = g.groupby("lam").audit_auc.mean()
        below = curve[curve < 0.5]
        lam_cross = float(below.index[0]) if len(below) else float("nan")
        per_rep = []
        for _, gr in g.groupby("rep"):
            gr = gr.sort_values("lam")
            b = gr[gr.audit_auc < 0.5]
            per_rep.append(float(b.lam.iloc[0]) if len(b) else np.nan)
        per_rep = np.asarray(per_rep, float)
        ok = per_rep[np.isfinite(per_rep)]
        out.append({
            "setting": setting, "gamma": gamma, "kappa": kappa,
            "beta": g.beta.iloc[0], "sigma_eff": g.sigma_eff.iloc[0],
            "analytic_global_auc": g.analytic_global_auc.iloc[0],
            "realised_global_auc": g[g.lam == 0.0].global_auc.mean(),
            "audit_auc_at_lam0": curve.iloc[0],
            "audit_auc_at_lam5": curve.loc[5.0],
            "delta_at_lam5": curve.iloc[0] - curve.loc[5.0],
            "crossing_mean_curve": lam_cross,
            "crossing_mean": float(ok.mean()) if ok.size else np.nan,
            "crossing_ci_lo": float(np.quantile(ok, 0.025)) if ok.size else np.nan,
            "crossing_ci_hi": float(np.quantile(ok, 0.975)) if ok.size else np.nan,
            "n_reps": len(per_rep), "n_reps_crossing": int(ok.size),
            "reversal_rate_at_lam_ge_cross": float(
                g[g.lam >= (lam_cross if np.isfinite(lam_cross) else 99)].reversal.mean())
            if np.isfinite(lam_cross) else np.nan,
        })
    return pd.DataFrame(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--workers", type=int, default=1,
                    help="accepted for interface parity; the grid is cheap enough to run serially")
    ap.add_argument("--settings", default="baseline,M1,M2")
    ap.add_argument("--out", default="ae_misspecification.csv")
    args = ap.parse_args()

    want = {s.strip() for s in args.settings.split(",")}
    print("== A3 / Gaussian misspecification sensitivity ==")
    print(f"   N={N}, rho={RHO}, sigma_E={SIG_E}, sigma_Q={SIG_Q}, "
          f"reps={args.reps}, lambda in [0,5] step 0.1")
    print("\n   calibration to a global AUROC of 0.70:")
    calib = []
    for gamma in GAMMAS:
        b0 = calibrate_beta(gamma, 0.0)
        calib.append({"setting": "baseline (D indep Z)", "gamma": gamma, "kappa": 0.0,
                      "beta": b0, "analytic_global_auc": analytic_global_auc(b0, gamma, 0.0)})
        print(f"     baseline gamma={gamma:<4}: beta={b0:.4f}  -> AUC={calib[-1]['analytic_global_auc']:.4f}")
    # M1 changes only Q (f enters Q, not E), and the global AUROC is a function of
    # E alone, so M1 must keep the baseline beta.  Recalibrating here would silently
    # make M1 a different global-AUC point rather than a misspecification control.
    for gamma in GAMMAS:
        b1 = calibrate_beta(gamma, 0.0)
        calib.append({"setting": "M1 cubic f(D)=D+0.3D^3", "gamma": gamma, "kappa": 0.0,
                      "beta": b1, "analytic_global_auc": analytic_global_auc(b1, gamma, 0.0),
                      "note": "beta identical to baseline (f enters Q only)"})
        print(f"     M1       gamma={gamma:<4}: beta={b1:.4f} "
              f"(= baseline, f enters Q only)")
    for kappa in KAPPAS:
        for gamma in GAMMAS:
            bk = calibrate_beta(gamma, kappa)
            calib.append({"setting": f"M2 correlated D|Z, kappa={kappa}", "gamma": gamma,
                          "kappa": kappa, "beta": bk,
                          "analytic_global_auc": analytic_global_auc(bk, gamma, kappa)})
            print(f"     M2 kappa={kappa} gamma={gamma:<4}: beta={bk:.4f}  "
                  f"-> AUC={calib[-1]['analytic_global_auc']:.4f}")
    pd.DataFrame(calib).to_csv(OUT / "ae_misspecification_calibration.csv", index=False)

    t0 = time.time()
    rows = []
    if "baseline" in want:
        for gamma in GAMMAS:
            rows += run_setting("baseline", gamma, 0.0, False, args.reps, 900_000)
            print(f"   baseline gamma={gamma} done ({time.time()-t0:.0f}s)", flush=True)
    if "M1" in want:
        for gamma in GAMMAS:
            rows += run_setting("M1 cubic f(D)=D+0.3D^3", gamma, 0.0, True,
                                args.reps, 900_000)
            print(f"   M1 gamma={gamma} done ({time.time()-t0:.0f}s)", flush=True)
    if "M2" in want:
        for kappa in KAPPAS:
            for gamma in GAMMAS:
                rows += run_setting(f"M2 correlated D|Z (kappa={kappa})", gamma, kappa,
                                    False, args.reps, 700_000)
                print(f"   M2 kappa={kappa} gamma={gamma} done ({time.time()-t0:.0f}s)",
                      flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / args.out, index=False)
    s = summarise(df)
    s.to_csv(OUT / args.out.replace(".csv", "_crossing.csv"), index=False)

    print("\n== summary ==")
    print(s[["setting", "gamma", "kappa", "beta", "realised_global_auc",
             "audit_auc_at_lam0", "audit_auc_at_lam5", "delta_at_lam5",
             "crossing_mean", "crossing_ci_lo", "crossing_ci_hi",
             "n_reps_crossing"]].round(4).to_string(index=False))

    print("\n== the three questions ==")
    for setting, g in s.groupby("setting"):
        g = g.sort_values(["kappa", "gamma"])
        mono = bool((g.delta_at_lam5 > 0).all())
        print(f"\n   {setting}")
        print(f"     1. conditioned AUROC falls with lambda in every cell: {mono} "
              f"(delta at lam=5 ranges {g.delta_at_lam5.min():+.4f} .. {g.delta_at_lam5.max():+.4f})")
        crossed = g[g.n_reps_crossing > 0]
        print(f"     2. crossing occurs in {len(g[g.n_reps_crossing > 0])}/{len(g)} cells; "
              f"full crossing (100/100 reps) in {int((g.n_reps_crossing == g.n_reps).sum())}/{len(g)}")
        for kappa, gk in g.groupby("kappa"):
            mono_g = gk.sort_values("gamma")
            print(f"     3. kappa={kappa}: crossing vs gamma -> "
                  + ", ".join(f"{r.gamma}:{r.crossing_mean:.2f}" for r in mono_g.itertuples()))
    print(f"\n   wrote {args.out} and {args.out.replace('.csv','_crossing.csv')} "
          f"({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
