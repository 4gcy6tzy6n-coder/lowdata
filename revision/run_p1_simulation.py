#!/usr/bin/env python3
"""Finite-sample Gaussian simulation for the conditioning artifact.

The manuscript's Proposition 1 states the mechanism analytically.  This script
checks it through the *same* estimator chain the experiments use, so the
proposition is not left floating outside the evaluation:

    Z ~ Bernoulli(rho)            corruption indicator
    D ~ N(0, 1)                   latent difficulty
    Q = D + lambda*Z + eps_Q      conditioning covariate
    E = beta*Z + gamma*D + eps_E  detector evidence

For each (lambda, gamma, delta) cell the pipeline is:
    standardise Q  ->  common support  ->  caliper = delta  ->
    nn_wo matching (max_matches = 5, without replacement)  ->  A_audit
and the three AUROCs A_g, A_exact, A_audit are reported along with the matched
coverage c0 (clean) and c1 (noisy), so the reversal boundary A_audit < 0.5 can be
located directly.

beta is chosen so that A_g ~= 0.70, matching the C100-S20 headline.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_p0_batch import (MAX_CLEAN, common_support, matched_pairs,  # noqa: E402
                          pairwise_auc, rank_auc)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "revision" / "p1_simulation"

RHO = 0.2
SIG_E = 1.0
SIG_Q = 0.3


def beta_for_target(target: float, gamma: float, s_e: float = SIG_E) -> float:
    """beta such that the global two-Gaussian AUC equals ``target``.

    A_g = Phi(beta / sqrt(2 (gamma^2 + s_e^2))) for Z ~ Bernoulli(0.2) at these
    scales, so invert it.
    """
    from scipy.stats import norm
    return float(norm.ppf(target) * np.sqrt(2.0 * (gamma * gamma + s_e * s_e)))


def simulate(n: int, lam: float, gamma: float, beta: float, caliper_sd: float,
             max_clean: int, rho: float, seed: int,
             s_e: float = SIG_E, s_q: float = SIG_Q) -> dict:
    rng = np.random.default_rng(seed)
    z = (rng.random(n) < rho).astype(bool)
    d = rng.normal(size=n)
    e = beta * z + gamma * d + s_e * rng.normal(size=n)
    q = d + lam * z + s_q * rng.normal(size=n)

    mask = z
    a_g = rank_auc(mask, e)
    # exact / oracle conditioning: compare only at equal D (the true difficulty),
    # which is the "population" target A_exact
    lo, hi = common_support(q[mask], q[~mask])
    keep = (q >= lo) & (q <= hi) if np.isfinite(lo) else np.zeros(n, bool)
    nk, ck = np.where(mask & keep)[0], np.where(~mask & keep)[0]
    a_exact = float("nan")
    if len(nk) and len(ck):
        sel = np.concatenate([nk, ck])
        lab = np.concatenate([np.ones(len(nk), bool), np.zeros(len(ck), bool)])
        a_exact = rank_auc(lab, e[sel])

    cal = caliper_sd * float(q.std())
    ni, ci = matched_pairs(e, mask, q, cal, max_clean)
    a_audit = pairwise_auc(e, ni, ci)

    return {
        "n": n, "lambda": lam, "gamma": gamma, "beta": beta,
        "caliper_sd": caliper_sd, "max_matches": max_clean, "rho": rho,
        "global_auc": a_g, "exact_auc": a_exact, "audit_auc": a_audit,
        "delta_total": a_g - a_audit if np.isfinite(a_audit) else np.nan,
        "delta_population": a_g - a_exact if np.isfinite(a_exact) else np.nan,
        "delta_matching": a_exact - a_audit if np.isfinite(a_audit) and np.isfinite(a_exact) else np.nan,
        "c0_clean_coverage": len(np.unique(ci)) / max(int((~mask).sum()), 1) if len(ci) else 0.0,
        "c1_noisy_coverage": len(np.unique(ni)) / max(int(mask.sum()), 1) if len(ni) else 0.0,
        "n_pairs": int(len(ni)),
        "reversal": bool(np.isfinite(a_audit) and a_audit < 0.5),
    }


def _worker(task: dict) -> list[dict]:
    out = []
    for lam in task["lambdas"]:
        for rep in range(task["reps"]):
            out.append(simulate(task["n"], lam, task["gamma"], task["beta"],
                                task["caliper_sd"], task["max_clean"], RHO,
                                seed=task["seed0"] + rep))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    ap.add_argument("--target-ag", type=float, default=0.70)
    ap.add_argument("--caliper", type=float, default=0.10)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    lambdas = np.round(np.linspace(0.0, 3.0, 31), 4)
    gammas = [0.25, 0.5, 1.0, 1.5]
    tasks = []
    for gi, gamma in enumerate(gammas):
        tasks.append({"lambdas": lambdas, "gamma": float(gamma),
                      "beta": beta_for_target(args.target_ag, gamma),
                      "caliper_sd": args.caliper, "max_clean": MAX_CLEAN,
                      "n": args.n, "reps": args.reps, "seed0": 1000 * (gi + 1)})

    print(f"== finite-sample Gaussian simulation ==  n={args.n} reps={args.reps} "
          f"target A_g={args.target_ag}")
    for t in tasks:
        print(f"   gamma={t['gamma']:.2f}  beta={t['beta']:.4f}")
    t0 = time.time()
    rows: list[dict] = []
    with mp.Pool(processes=min(args.workers, len(tasks))) as pool:
        for i, res in enumerate(pool.imap_unordered(_worker, tasks)):
            rows.extend(res)
            print(f"   [{i+1}/{len(tasks)}] gamma done ({time.time()-t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "simulation.csv", index=False)

    agg = (df.groupby(["gamma", "lambda"])
           .agg(n=("global_auc", "size"), a_g=("global_auc", "mean"),
                a_exact=("exact_auc", "mean"), a_audit=("audit_auc", "mean"),
                d_total=("delta_total", "mean"), d_pop=("delta_population", "mean"),
                d_match=("delta_matching", "mean"),
                c0=("c0_clean_coverage", "mean"), c1=("c1_noisy_coverage", "mean"),
                n_rev=("reversal", "sum"))
           .reset_index())
    agg.to_csv(OUT / "simulation_curves.csv", index=False)

    print("\n== reversal boundary (lambda at which A_audit first drops below 0.5) ==")
    bnd = []
    for gamma, g in agg.groupby("gamma"):
        g = g.sort_values("lambda")
        below = g[g.a_audit < 0.5]
        lam_star = float(below["lambda"].iloc[0]) if len(below) else float("nan")
        bnd.append({"gamma": gamma,
                    "beta": float(df[df.gamma == gamma].beta.iloc[0]),
                    "a_g": float(g.a_g.mean()),
                    "a_audit_at_lambda0": float(g[g["lambda"] == 0].a_audit.iloc[0]),
                    "a_audit_at_lambda3": float(g[g["lambda"] == 3.0].a_audit.iloc[0]),
                    "lambda_star_reversal": lam_star})
    b = pd.DataFrame(bnd)
    b.to_csv(OUT / "reversal_boundary.csv", index=False)
    print(b.round(4).to_string(index=False))
    (OUT / "summary.json").write_text(json.dumps(
        {"n": args.n, "reps": args.reps, "target_ag": args.target_ag,
         "caliper_sd": args.caliper, "max_matches": MAX_CLEAN, "rho": RHO,
         "boundary": b.to_dict(orient="records")}, indent=2, default=str))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
