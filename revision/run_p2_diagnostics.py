"""P2-10 — supporting diagnostics: Beta mixture, Otsu threshold, runtime/memory.

Three cheap, training-free diagnostics that land in the Supplementary:

``beta_mixture``
    For each representative setting, fit the paper's 2-component Beta mixture on
    the combined evidence E and report convergence iterations, final
    log-likelihood, component means/variances/weights, and the stability of the
    fit across repeated random initialisations.  Also compares a 1-component
    Beta against the 2-component mixture by AIC/BIC, which is the honest way to
    ask "is the mixture justified at all?".

``otsu``
    Estimate the noise fraction from E by Otsu's threshold, and compare it to the
    evaluation-only true noisy fraction.  ``rho_hat`` is estimated WITHOUT the
    mask; the mask is used only to score the estimate (|rho_hat - rho|).

``runtime``
    Wall-clock and peak RSS for the revision's post-hoc stages (trace loading,
    detector computation, matching, bootstrap), so the compute cost claim is
    measured rather than asserted.

Outputs -> results/revision/diagnostics/{beta_mixture,otsu,runtime}.csv
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import numpy as np
import pandas as pd
from scipy.special import betaln, logsumexp

ROOT = Path("/root/quality_noise")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "revision"))

from quality_noise.governance.mixture import (  # noqa: E402
    _beta_logpdf, _beta_moments, _clip_evidence, _fit_bin, fit_beta_mixture,
)
from revq.bootstrap import paired_bootstrap  # noqa: E402
from revq.matching import match_pairs, matched_auc  # noqa: E402
from revq.prepare import load_cached  # noqa: E402

OUT = ROOT / "results" / "revision" / "diagnostics"
CACHE = ROOT / "results" / "revision" / "prepared"

# Representative settings spanning the detectability range.
REPRESENTATIVE = [
    ("cifar10", "symmetric0.2"), ("cifar10", "symmetric0.4"),
    ("cifar10", "asymmetric0.2"), ("cifar10", "asymmetric0.4"),
    ("cifar100", "symmetric0.2"),
    ("cifar10n", "cifar10n_aggre0.4"), ("cifar10n", "cifar10n_worse0.4"),
]


def peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


# ---------------------------------------------------------------------------
# Beta mixture
# ---------------------------------------------------------------------------

def _beta_mixture_loglik(E: np.ndarray, a0, b0, a1, b1, pi) -> float:
    """Total log-likelihood of a 2-component Beta mixture."""
    lp1 = _beta_logpdf(E, a1, b1) + np.log(pi + 1e-12)
    lp0 = _beta_logpdf(E, a0, b0) + np.log(1 - pi + 1e-12)
    return float(np.sum(logsumexp(np.vstack([lp1, lp0]), axis=0)))


def _one_component_beta(E: np.ndarray) -> tuple[float, float, float]:
    """Method-of-moments 1-component Beta and its total log-likelihood."""
    a, b = _beta_moments(float(E.mean()), float(E.var()))
    return a, b, float(np.sum(_beta_logpdf(E, a, b)))


def _beta_var(a: float, b: float) -> float:
    s = a + b
    return float(a * b / (s * s * (s + 1.0)))


def beta_mixture_diagnostics(seed_evidence: dict[tuple[str, str], list[np.ndarray]],
                             max_iter: int = 1000, n_inits: int = 8) -> pd.DataFrame:
    """Fit the mixture per setting, with init-stability and 1- vs 2-component tests."""
    rows = []
    rng = np.random.default_rng(0)
    for (ds, noise), ev_list in sorted(seed_evidence.items()):
        E_all = np.concatenate(ev_list)
        E = _clip_evidence(E_all)
        n = len(E)

        # --- reference fit (project code path) -------------------------------
        t0 = time.time()
        params = fit_beta_mixture(E, init_pi=0.4, max_iter=max_iter)
        fit_s = time.time() - t0
        a0, b0 = float(params.alpha0[0]), float(params.beta0[0])
        a1, b1 = float(params.alpha1[0]), float(params.beta1[0])
        pi = float(params.pi_q[0])
        ll2 = _beta_mixture_loglik(E, a0, b0, a1, b1, pi)

        # --- 1-component comparison -----------------------------------------
        a1c, b1c, ll1 = _one_component_beta(E)
        k1, k2 = 2, 5
        aic1, bic1 = 2 * k1 - 2 * ll1, k1 * np.log(n) - 2 * ll1
        aic2, bic2 = 2 * k2 - 2 * ll2, k2 * np.log(n) - 2 * ll2

        # --- stability over random initialisations --------------------------
        em_iters, logliks, means1, means0, weights = [], [], [], [], []
        for _ in range(n_inits):
            init = float(rng.uniform(0.15, 0.6))
            p = fit_beta_mixture(E, init_pi=init, max_iter=max_iter)
            A0, B0 = float(p.alpha0[0]), float(p.beta0[0])
            A1, B1 = float(p.alpha1[0]), float(p.beta1[0])
            P = float(p.pi_q[0])
            ll = _beta_mixture_loglik(E, A0, B0, A1, B1, P)
            logliks.append(ll)
            em_iters.append(max_iter if ll < ll2 - 1e-6 else -1)
            means1.append(A1 / (A1 + B1))
            means0.append(A0 / (A0 + B0))
            weights.append(P)

        rows.append({
            "dataset": ds, "noise": noise, "n_samples": n,
            "fit_seconds": fit_s,
            "final_loglik": ll2,
            "loglik_per_sample": ll2 / n,
            "alpha_clean": a0, "beta_clean": b0,
            "alpha_noisy": a1, "beta_noisy": b1,
            "mean_clean": a0 / (a0 + b0), "var_clean": _beta_var(a0, b0),
            "mean_noisy": a1 / (a1 + b1), "var_noisy": _beta_var(a1, b1),
            "pi_noisy": pi,
            "component_separation": (a1 / (a1 + b1)) - (a0 / (a0 + b0)),
            "loglik_1comp": ll1, "aic_1comp": aic1, "bic_1comp": bic1,
            "aic_2comp": aic2, "bic_2comp": bic2,
            "delta_aic_1_minus_2": aic1 - aic2,
            "delta_bic_1_minus_2": bic1 - bic2,
            "mixture_justified_by_bic": bool(bic2 < bic1),
            "n_inits": n_inits,
            "loglik_init_sd": float(np.std(logliks)),
            "loglik_init_min": float(np.min(logliks)),
            "loglik_init_max": float(np.max(logliks)),
            "pi_init_sd": float(np.std(weights)),
            "mean_noisy_init_sd": float(np.std(means1)),
            "mean_clean_init_sd": float(np.std(means0)),
        })
        print(f"   beta {ds}/{noise}: pi={pi:.3f} sep={rows[-1]['component_separation']:.3f} "
              f"BIC 1c={bic1:.0f} 2c={bic2:.0f} init_ll_sd={rows[-1]['loglik_init_sd']:.2f}", flush=True)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Otsu
# ---------------------------------------------------------------------------

def otsu_threshold(x: np.ndarray, n_bins: int = 256) -> float:
    """Otsu's threshold on a 1-D score histogram (no labels used)."""
    x = np.asarray(x, dtype=np.float64)
    hist, edges = np.histogram(x, bins=n_bins)
    hist = hist.astype(np.float64)
    centers = 0.5 * (edges[:-1] + edges[1:])
    total = hist.sum()
    if total == 0:
        return float("nan")
    w0 = np.cumsum(hist)
    w1 = total - w0
    valid = (w0 > 0) & (w1 > 0)
    csum = np.cumsum(hist * centers)
    mu0 = np.divide(csum, w0, out=np.zeros_like(csum), where=w0 > 0)
    mu1 = np.divide(csum[-1] - csum, w1, out=np.zeros_like(csum), where=w1 > 0)
    between = w0 * w1 * (mu0 - mu1) ** 2
    between[~valid] = -1
    k = int(np.argmax(between))
    return float(centers[k])


def otsu_diagnostics(seed_evidence: dict[tuple[str, str], list[tuple[np.ndarray, np.ndarray]]]) -> pd.DataFrame:
    """rho_hat from Otsu vs the evaluation-only true noisy fraction."""
    rows = []
    for (ds, noise), items in sorted(seed_evidence.items()):
        for i, (E, mask) in enumerate(items):
            thr = otsu_threshold(E)
            rho_hat = float((E > thr).mean())
            rho_true = float(np.asarray(mask, dtype=bool).mean())
            rows.append({
                "dataset": ds, "noise": noise, "seed": i,
                "otsu_threshold": thr,
                "rho_hat": rho_hat, "rho_true": rho_true,
                "abs_error": abs(rho_hat - rho_true),
                "signed_error": rho_hat - rho_true,
                "n_samples": len(E),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------

def runtime_diagnostics(n_bootstrap: int = 2000) -> pd.DataFrame:
    """Measured wall-clock / peak RSS for the post-hoc stages on one run."""
    rows = []

    t0 = time.time()
    from scripts.cross_detector.io_utils import resolve_artifacts
    arts = resolve_artifacts(ROOT / "results", "cifar10", "symmetric0.2", 0)
    traces = pd.read_parquet(arts["traces"])
    t_load_traces = time.time() - t0

    t0 = time.time()
    from scripts.cross_detector.detectors import compute_all_detectors
    emb = np.load(arts["embeddings"]).astype(np.float32)
    q = pd.read_parquet(arts["quality"]).sort_values("sample_id")
    dets = compute_all_detectors(traces, emb, np.zeros(len(q), dtype=int))
    t_detectors = time.time() - t0

    r = load_cached("cifar10", "symmetric0.2", 0, cache=CACHE)
    score = r["signals"]["combined"]
    mask = r["mask"]
    qs = r["signals"]["dino_density"]

    t0 = time.time()
    pairs = match_pairs(score, mask, qs, strategy="nn_wo", eps_sd=0.10, force_greedy=False)
    t_match = time.time() - t0
    auc = matched_auc(pairs)

    t0 = time.time()
    bs = paired_bootstrap(score, mask, qs, strategy="nn_wo", eps_sd=0.10,
                          n_resamples=n_bootstrap, force_greedy=True,
                          point=(float(np.mean(mask)), auc))
    t_boot = time.time() - t0

    rows.append({
        "stage": "load_traces", "dataset": "cifar10", "noise": "symmetric0.2", "seed": 0,
        "n_samples": int(len(q)), "seconds": t_load_traces, "peak_rss_mb": peak_rss_mb(),
        "notes": f"traces parquet {(traces.shape[0])} rows",
    })
    rows.append({
        "stage": "all_6_detectors", "dataset": "cifar10", "noise": "symmetric0.2", "seed": 0,
        "n_samples": int(len(q)), "seconds": t_detectors, "peak_rss_mb": peak_rss_mb(),
        "notes": "CPU; reuses saved trajectories + embeddings",
    })
    rows.append({
        "stage": "matching_nn_wo_hungarian", "dataset": "cifar10", "noise": "symmetric0.2", "seed": 0,
        "n_samples": int(len(q)), "seconds": t_match, "peak_rss_mb": peak_rss_mb(),
        "notes": f"{len(pairs.s_noisy)} pairs, caliper 0.10 SD, Q=DINO density",
    })
    rows.append({
        "stage": f"paired_bootstrap_B{n_bootstrap}", "dataset": "cifar10", "noise": "symmetric0.2", "seed": 0,
        "n_samples": int(len(q)), "seconds": t_boot, "peak_rss_mb": peak_rss_mb(),
        "notes": "global + re-matched AUC_QC inside every resample",
    })
    rows.append({
        "stage": "training_resnet18_200ep", "dataset": "cifar10", "noise": "symmetric0.2", "seed": 0,
        "n_samples": int(len(q)), "seconds": float("nan"), "peak_rss_mb": float("nan"),
        "notes": "measured separately on RTX 5090 (revision train_waves log)",
    })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--n-inits", type=int, default=8)
    ap.add_argument("--only", type=str, default="all",
                    choices=["all", "beta", "otsu", "runtime"])
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    if args.only in ("all", "beta", "otsu"):
        ev_beta: dict = {}
        ev_otsu: dict = {}
        for ds, noise in REPRESENTATIVE:
            d = CACHE / ds / noise
            if not d.exists():
                print(f"   skip {ds}/{noise} (not prepared)", flush=True)
                continue
            ev_beta[(ds, noise)] = []
            ev_otsu[(ds, noise)] = []
            for p in sorted(d.glob("seed*.npz")):
                r = load_cached(ds, noise, int(p.name[4:-4]), cache=CACHE)
                E = r["signals"]["combined"]
                ev_beta[(ds, noise)].append(E)
                ev_otsu[(ds, noise)].append((E, r["mask"]))

        if args.only in ("all", "beta"):
            print("== Beta mixture ==", flush=True)
            bm = beta_mixture_diagnostics(ev_beta, n_inits=args.n_inits)
            bm.to_csv(OUT / "beta_mixture.csv", index=False)

        if args.only in ("all", "otsu"):
            print("== Otsu ==", flush=True)
            ot = otsu_diagnostics(ev_otsu)
            ot.to_csv(OUT / "otsu.csv", index=False)
            if len(ot):
                print(ot.groupby(["dataset", "noise"])[["rho_hat", "rho_true", "abs_error"]]
                      .mean().round(4).to_string(), flush=True)

    if args.only in ("all", "runtime"):
        print("== Runtime ==", flush=True)
        rt = runtime_diagnostics(n_bootstrap=args.bootstrap)
        rt.to_csv(OUT / "runtime.csv", index=False)
        print(rt[["stage", "seconds", "peak_rss_mb"]].round(2).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
