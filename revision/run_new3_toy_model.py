"""NEW-3 — synthetic model of the post-treatment conditioning artifact.

The revised paper's claim is mechanistic: conditioning on a quality covariate Q
that is *itself* a function of the corruption indicator Z manufactures a gap
between global and quality-matched detectability, up to a full ranking reversal,
even when the detector's true conditional skill is fixed and positive.

This script derives that result in closed form for a Gaussian model and verifies
the derivation by Monte Carlo.

Model
-----
    Z ~ Bernoulli(rho)                      corruption indicator (z_i)
    D ~ N(0, 1)                             latent sample difficulty
    eps_Y ~ N(0, s_Y^2)                     label noise: Y = beta*Z + eps_Y
    eps_E ~ N(0, s_E^2)                     detector noise
    eps_Q ~ N(0, s_Q^2)                     quality-proxy noise
    E = beta*Z + gamma*D + eps_E            detector evidence
    Q = D + lam*Z + eps_Q                   quality proxy  (lam >= 0)

`lam` is exactly what we control in the real experiments (NEW-1): the amount of
corruption information the quality proxy carries.  With lam = 0, Q is a pure
difficulty measure and is independent of Z given D.

Derivation
----------
Take Z = 1 (all quantities below are conditional on Z and on the infinite-sample
limit, so only the linear-Gaussian structure matters):

    Q | Z=1 ~ N(lam, 1 + s_Q^2)
    E | Z=1 ~ N(beta, gamma^2 + s_E^2)

so the *global* comparison against Z = 0 is a two-Gaussian AUC with

    AUC_global = Phi( (mu_E1 - mu_E0) / sqrt(s_E1^2 + s_E0^2) ),  mu_E1 - mu_E0 = beta.

Matching holds Q fixed, hence also D_hat = Q - lam*Z.  Since E, Q, Z are jointly
Gaussian, E | Q, Z is Gaussian and the matched comparison is again a two-Gaussian
AUC with

    mu_E1|q - mu_E0|q = beta - lam * Cov(E,D)/Var(Q)
    Var(E | Q, Z=z)   = g^2 + s_E^2 - Cov(E,D)^2 / Var(Q)

where Cov(E,D) = gamma, Var(Q) = 1 + s_Q^2, and g^2 + s_E^2 = gamma^2 + s_E^2.
Note the matched *variances* are equal across Z, so

    AUC_matched(lam) = Phi( (beta - lam*gamma/(1+s_Q^2)) / sqrt(gamma^2 + s_E^2 - gamma^2/(1+s_Q^2)) ).

The numerator carries the whole effect: as lam grows, the effective separation
shrinks linearly and changes sign at lam* = beta*(1+s_Q^2)/gamma, making
AUC_matched drop below 0.5 — a conditional ranking reversal — while AUC_global is
untouched.  The gap Delta(lam) = AUC_global - AUC_matched(lam) is therefore a
pure artifact of conditioning on a post-treatment covariate.

Outputs -> results/revision/toy/{curves.csv, verification.csv, summary.json}
           results/revision/toy/toy_conditioning.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys

import numpy as np
import pandas as pd
from scipy.stats import norm

OUT = Path("/root/quality_noise/results/revision/toy")


def auc_matched_closed(lam: float, beta: float, gamma: float, s_e: float,
                       s_q: float) -> float:
    """Closed-form AUC of E between Z=1 and Z=0 at matched Q (see module docstring)."""
    vq = 1.0 + s_q * s_q
    num = beta - lam * gamma / vq
    var = gamma * gamma + s_e * s_e - (gamma * gamma) / vq
    return float(norm.cdf(num / np.sqrt(var)))


def auc_global_closed(beta: float, gamma: float, s_e: float) -> float:
    """Global AUC of E between Z=1 and Z=0 (D marginalised, not conditioned)."""
    var = 2.0 * (gamma * gamma + s_e * s_e)
    return float(norm.cdf(beta / np.sqrt(var)))


def lam_star(beta: float, gamma: float, s_q: float) -> float:
    """Blend weight at which the matched AUC crosses 0.5 (conditional reversal)."""
    return beta * (1.0 + s_q * s_q) / gamma


def simulate(n: int, lam: float, beta: float, gamma: float, s_e: float, s_q: float,
             rho: float, seed: int) -> dict:
    """Monte Carlo analogue: an i.i.d. sample, matched with a caliper on Q."""
    rng = np.random.default_rng(seed)
    z = (rng.random(n) < rho).astype(int)
    d = rng.normal(size=n)
    e = beta * z + gamma * d + s_e * rng.normal(size=n)
    q = d + lam * z + s_q * rng.normal(size=n)

    from sklearn.metrics import roc_auc_score
    auc_g = float(roc_auc_score(z, e))

    # Quality-matched AUC: the same caliper-matching estimator the real
    # experiments use (vectorised, so a 400k-sample Monte Carlo stays cheap).
    sys.path.insert(0, "/root/quality_noise")
    sys.path.insert(0, "/root/quality_noise/revision")
    from revq.matching import matched_auc_fast
    thr = 0.10 * float(q.std())
    auc_m = matched_auc_fast(e, z.astype(bool), q, thr, clean_budget=256)
    total = -1
    return {"auc_global": auc_g, "auc_matched": auc_m, "delta": auc_g - auc_m,
            "n_pairs": total,
            # realised information about Z carried by Q
            "info_auc_q_vs_z": float(roc_auc_score(z, q))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--beta", type=float, default=0.5, help="true detector signal")
    ap.add_argument("--gamma", type=float, default=0.5, help="difficulty loading of E")
    ap.add_argument("--s-e", type=float, default=1.0)
    ap.add_argument("--s-q", type=float, default=0.3)
    ap.add_argument("--rho", type=float, default=0.2)
    ap.add_argument("--n", type=int, default=400_000)
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    lams = np.round(np.linspace(0.0, 1.0, 21), 4)
    g0 = auc_global_closed(args.beta, args.gamma, args.s_e)
    ls = lam_star(args.beta, args.gamma, args.s_q)

    rows = []
    for lam in lams:
        m0 = auc_matched_closed(lam, args.beta, args.gamma, args.s_e, args.s_q)
        reps = [simulate(args.n, lam, args.beta, args.gamma, args.s_e, args.s_q,
                         args.rho, seed=1000 + r) for r in range(args.reps)]
        rows.append({
            "lam": float(lam),
            "auc_global_closed": g0,
            "auc_matched_closed": m0,
            "delta_closed": g0 - m0,
            "auc_global_mc": float(np.mean([x["auc_global"] for x in reps])),
            "auc_matched_mc": float(np.mean([x["auc_matched"] for x in reps])),
            "delta_mc": float(np.mean([x["delta"] for x in reps])),
            "delta_mc_sd": float(np.std([x["delta"] for x in reps], ddof=1)),
            "info_auc_q_vs_z_mc": float(np.mean([x["info_auc_q_vs_z"] for x in reps])),
            "matched_auc_below_half": bool(m0 < 0.5),
        })
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "curves.csv", index=False)

    # Verification: closed form vs Monte Carlo
    ver = pd.DataFrame({
        "quantity": ["auc_global", "auc_matched@lam=0", "auc_matched@lam=0.5",
                     "auc_matched@lam=1", "delta@lam=1"],
        "closed_form": [g0,
                        auc_matched_closed(0.0, args.beta, args.gamma, args.s_e, args.s_q),
                        auc_matched_closed(0.5, args.beta, args.gamma, args.s_e, args.s_q),
                        auc_matched_closed(1.0, args.beta, args.gamma, args.s_e, args.s_q),
                        g0 - auc_matched_closed(1.0, args.beta, args.gamma, args.s_e, args.s_q)],
        "monte_carlo": [df.loc[np.isclose(df.lam, 0.0), "auc_global_mc"].iloc[0],
                        df.loc[np.isclose(df.lam, 0.0), "auc_matched_mc"].iloc[0],
                        df.loc[np.isclose(df.lam, 0.5), "auc_matched_mc"].iloc[0],
                        df.loc[np.isclose(df.lam, 1.0), "auc_matched_mc"].iloc[0],
                        df.loc[np.isclose(df.lam, 1.0), "delta_mc"].iloc[0]],
    })
    ver["abs_diff"] = (ver["closed_form"] - ver["monte_carlo"]).abs()
    ver.to_csv(OUT / "verification.csv", index=False)

    summary = {
        "params": vars(args),
        "auc_global": g0,
        "auc_matched_at_lam0": auc_matched_closed(0.0, args.beta, args.gamma, args.s_e, args.s_q),
        "auc_matched_at_lam1": auc_matched_closed(1.0, args.beta, args.gamma, args.s_e, args.s_q),
        "delta_at_lam0": g0 - auc_matched_closed(0.0, args.beta, args.gamma, args.s_e, args.s_q),
        "delta_at_lam1": g0 - auc_matched_closed(1.0, args.beta, args.gamma, args.s_e, args.s_q),
        "lam_star_reversal": float(ls),
        "max_abs_verification_diff": float(ver["abs_diff"].max()),
        "delta_monotone_in_lam": bool(np.all(np.diff(df["delta_closed"].to_numpy()) >= -1e-12)),
    }
    with open(OUT / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ax[0].plot(df.lam, df.auc_global_closed, label="AUC_global (closed form)", lw=2)
        ax[0].plot(df.lam, df.auc_matched_closed, label="AUC_matched (closed form)", lw=2)
        ax[0].plot(df.lam, df.auc_matched_mc, "o", ms=3, label="AUC_matched (Monte Carlo)")
        ax[0].axhline(0.5, color="k", ls=":", lw=1)
        ax[0].axvline(ls, color="r", ls="--", lw=1, label=f"lam* = {ls:.2f}")
        ax[0].set_xlabel(r"$\lambda$  (corruption information in $Q$)")
        ax[0].set_ylabel("AUROC")
        ax[0].set_title("Conditioning artifact")
        ax[0].legend(fontsize=8)
        ax[1].plot(df.info_auc_q_vs_z_mc, df.delta_closed, "-o", ms=3)
        ax[1].set_xlabel("AUC(Q, Z)")
        ax[1].set_ylabel(r"$\Delta_Q$")
        ax[1].set_title(r"$\Delta_Q$ vs information in $Q$")
        fig.tight_layout()
        fig.savefig(OUT / "toy_conditioning.png", dpi=160)
        print(f"figure -> {OUT/'toy_conditioning.png'}")
    except Exception as e:  # noqa: BLE001
        print(f"figure skipped: {e}")

    print("\n== toy model ==")
    print(f"  AUC_global                 = {g0:.4f}  (constant in lam)")
    print(f"  AUC_matched  lam=0         = {summary['auc_matched_at_lam0']:.4f}")
    print(f"  AUC_matched  lam=1         = {summary['auc_matched_at_lam1']:.4f}")
    print(f"  Delta_Q      lam=0 -> lam=1: {summary['delta_at_lam0']:+.4f} -> {summary['delta_at_lam1']:+.4f}")
    print(f"  reversal threshold lam*    = {ls:.3f}")
    print(f"  Delta monotone in lam      = {summary['delta_monotone_in_lam']}")
    print(f"  max |closed form - MC|     = {summary['max_abs_verification_diff']:.4f}")
    print(f"\nwrote {OUT}/curves.csv, verification.csv, summary.json")


if __name__ == "__main__":
    main()
