"""NEW-1 — controlled contamination / leakage sweep.

The revision's core observation is a *natural* gradient: Delta_Q rises with how
much information the quality covariate Q carries about the noise label Z.

    random   ->  DINO density  ->  native density  ->  prototype margin  ->  KNN agreement
    I(Q;Z) :    none              none                 some                 much

That is correlational — five covariates differ in many ways at once.  This script
turns it into a controlled experiment: hold the *label-free* part fixed at
Q_DINO and blend in a noise-informative component with weight alpha,

    Q_alpha = (1 - alpha) * Q_DINO + alpha * Q_noise,   alpha in {0,.1,.2,.4,.6,.8,1}

then trace two curves as alpha goes 0 -> 1:

    AUC(Q_alpha, Z)          how much information about Z the covariate carries
    Delta_Q(alpha)           the global-minus-matched gap it induces

Two flavours of Q_noise are swept, and the distinction matters:

``leak``   Q_noise = Z, the evaluation-only corruption indicator itself.
           This is NOT a deployable quality metric; it is a synthetic control
           that pins the mechanism at the extreme.  Declared as such.
``obs``    Q_noise = the paper's observed-label KNN agreement (a real,
           label-only, deployable covariate).  This is the realistic case: a
           practitioner could actually condition on this.

Also blended against the paper's prototype margin as ``proto``.

Because the mixture is computed post hoc, the whole sweep is CPU-only: no
network is retrained.

Outputs -> results/revision/contamination/{sweep.csv, curves.csv, summary.json}
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import zlib
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "3")

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path("/root/quality_noise")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "revision"))

from revq.bootstrap import paired_bootstrap  # noqa: E402
from revq.matching import match_pairs, matched_auc  # noqa: E402
from revq.prepare import load_cached  # noqa: E402

OUT = ROOT / "results" / "revision" / "contamination"
CACHE = ROOT / "results" / "revision" / "prepared"

ALPHAS = [0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
BLENDS = ["leak", "obs", "proto"]
DETECTORS = ["combined", "ema_loss", "confidence", "aum", "forgetting",
             "neighbor", "confident_learning"]

SETTINGS = [
    ("cifar10", "symmetric0.2"), ("cifar10", "symmetric0.4"),
    ("cifar10", "asymmetric0.2"), ("cifar10", "asymmetric0.4"),
    ("cifar100", "symmetric0.2"),
    ("cifar10n", "cifar10n_aggre0.4"), ("cifar10n", "cifar10n_worse0.4"),
]


def mutual_information(q: np.ndarray, mask: np.ndarray, n_bins: int = 5) -> float:
    """Empirical MI (nats) between a quality covariate and the noise indicator.

    Bins Q into quantiles and computes I(Q;Z) from the 2 x n_bins contingency
    table, so the number is comparable across covariates with different scales
    and does not depend on the arbitrary direction of the association.
    """
    q = np.asarray(q, dtype=np.float64)
    z = np.asarray(mask, dtype=bool)
    edges = np.quantile(q, np.linspace(0, 1, n_bins + 1))
    edges = np.unique(edges)
    if len(edges) < 3:
        return 0.0
    b = np.clip(np.digitize(q, edges[1:-1]), 0, len(edges) - 2)
    joint = np.zeros((2, len(edges) - 1), dtype=np.float64)
    for zi in (0, 1):
        sel = z == bool(zi)
        joint[zi] = np.bincount(b[sel], minlength=joint.shape[1])
    n = joint.sum()
    if n <= 0:
        return 0.0
    pxy = joint / n
    px = pxy.sum(axis=1, keepdims=True)
    py = pxy.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(pxy > 0, pxy / (px @ py), 1.0)
        mi = float(np.sum(np.where(pxy > 0, pxy * np.log(ratio), 0.0)))
    return mi


def _unit(x: np.ndarray) -> np.ndarray:
    """Min-max to [0,1] (rank preserving) so blends are on a common scale."""
    x = np.asarray(x, dtype=np.float64)
    lo, hi = np.nanmin(x), np.nanmax(x)
    if not np.isfinite(lo) or hi - lo < 1e-12:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def _mixture(q0: np.ndarray, qn: np.ndarray, alpha: float) -> np.ndarray:
    """Q_alpha = (1-alpha) Q0 + alpha Qnoise, with both mapped to [0,1] first."""
    return (1.0 - alpha) * _unit(q0) + alpha * _unit(qn)


def _blend_source(name: str, r: dict, rng: np.random.Generator) -> np.ndarray:
    """The noise-informative component for one blend flavour."""
    if name == "leak":
        # Evaluation-only synthetic control: the corruption indicator itself.
        return r["mask"].astype(np.float64)
    if name == "obs":
        return np.asarray(r["signals"]["knn_agreement"], dtype=np.float64)
    if name == "proto":
        return np.asarray(r["signals"]["proto_margin"], dtype=np.float64)
    raise ValueError(f"unknown blend: {name}")


def _seed_worker(task: dict) -> list[dict]:
    """All (blend, alpha, detector) cells for one run.  Module-level for mp.Pool."""
    ds, noise, seed = task["dataset"], task["noise"], task["seed"]
    detectors = task["detectors"]
    n_bs = task["bootstrap"]
    max_noisy_bs = task["max_noisy_bs"]

    r = load_cached(ds, noise, seed, cache=CACHE, retries=2)
    mask = r["mask"]
    q0 = np.asarray(r["signals"]["dino_density"], dtype=np.float64)
    out: list[dict] = []
    for blend in BLENDS:
        qn = _blend_source(blend, r, np.random.default_rng(seed))
        for alpha in ALPHAS:
            qa = _mixture(q0, qn, alpha)
            # Signed AUC is direction-dependent (a *low* quality score can be
            # strongly informative about Z), so information content is
            # |AUC - 0.5|; mutual_information() below is the scale-free check.
            auc_qz = float(roc_auc_score(mask, qa))
            info = abs(auc_qz - 0.5)
            for det in detectors:
                score = np.asarray(r["signals"][det], dtype=np.float64)
                auc_g = float(roc_auc_score(mask, score))
                pairs = match_pairs(score, mask, qa, strategy="nn_wo",
                                    eps_sd=0.10, force_greedy=False)
                auc_qc = matched_auc(pairs)
                bs = paired_bootstrap(
                    score, mask, qa, n_resamples=n_bs,
                    seed=seed * 977 + abs(zlib.crc32(f"{det}|{blend}".encode())) % 99991,
                    eps_sd=0.10, strategy="nn_wo", point=(auc_g, auc_qc),
                    force_greedy=True, max_noisy_per_resample=max_noisy_bs)
                smd_after = float("nan")
                if len(pairs.q_noisy):
                    denom = max(np.sqrt((pairs.q_noisy.var() + pairs.q_clean.var()) / 2), 1e-12)
                    smd_after = float((pairs.q_noisy.mean() - pairs.q_clean.mean()) / denom)
                out.append({
                    "dataset": ds, "noise": noise, "seed": seed,
                    "blend": blend, "alpha": alpha, "detector": det,
                    "auc_q_vs_z_signed": auc_qz,
                    "info_auc_q_vs_z": info,
                    "mutual_information": mutual_information(qa, mask),
                    "auc_global": auc_g, "auc_qc": auc_qc,
                    "delta_q": auc_g - auc_qc,
                    "ci_delta_lo": bs["ci_delta"][0], "ci_delta_hi": bs["ci_delta"][1],
                    "ci_qc_lo": bs["ci_qc"][0], "ci_qc_hi": bs["ci_qc"][1],
                    "ci_qc_hi_lt_0p5": bool(bs["ci_qc"][1] < 0.5),
                    "n_pairs": int(len(pairs.s_noisy)),
                    "coverage_noisy": float(len(np.unique(pairs.n_idx)) / max(mask.sum(), 1)),
                    "smd_before": float((qa[mask].mean() - qa[~mask].mean()) / max(qa.std(), 1e-12)),
                    "smd_after": smd_after,
                })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bootstrap", type=int, default=500)
    ap.add_argument("--max-noisy-bs", type=int, default=1200)
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--detectors", type=str, default=",".join(DETECTORS))
    ap.add_argument("--settings", type=str, default="all")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    detectors = [d.strip() for d in args.detectors.split(",") if d.strip()]
    settings = SETTINGS if args.settings == "all" else \
        [tuple(s.split("/")) for s in args.settings.split(",")]

    tasks: list[dict] = []
    for ds, noise in settings:
        d = CACHE / ds / noise
        if not d.exists():
            print(f"skip {ds}/{noise}: not prepared", flush=True)
            continue
        seeds = sorted(int(p.name[4:-4]) for p in d.glob("seed*.npz"))
        print(f"== {ds}/{noise}  seeds={seeds}", flush=True)
        for seed in seeds:
            tasks.append({"dataset": ds, "noise": noise, "seed": seed,
                          "detectors": detectors, "bootstrap": args.bootstrap,
                          "max_noisy_bs": args.max_noisy_bs})

    print(f"\n{len(tasks)} run-tasks across {args.workers} workers "
          f"({len(ALPHAS)} alphas x {len(BLENDS)} blends x {len(detectors)} detectors each)",
          flush=True)
    t0 = time.time()
    rows: list[dict] = []
    partial = OUT / "sweep.partial.csv"
    with mp.Pool(processes=args.workers) as pool:
        for i, res in enumerate(pool.imap_unordered(_seed_worker, tasks)):
            rows.extend(res)
            if (i + 1) % 4 == 0 or (i + 1) == len(tasks):
                pd.DataFrame(rows).to_csv(partial, index=False)
                print(f"   {i+1}/{len(tasks)} runs done ({time.time()-t0:.0f}s, "
                      f"{len(rows)} cells)", flush=True)

    if not rows:
        print("no rows produced")
        return
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "sweep.csv", index=False)

    key = ["dataset", "noise", "blend", "detector", "alpha"]
    curves = (df.groupby(key)
              .agg(n_seeds=("seed", "nunique"),
                   info_auc_q_vs_z=("info_auc_q_vs_z", "mean"),
                   info_auc_sd=("info_auc_q_vs_z", "std"),
                   auc_global=("auc_global", "mean"),
                   auc_qc=("auc_qc", "mean"),
                   delta_q=("delta_q", "mean"),
                   delta_q_sd=("delta_q", "std"),
                   n_ci_qc_below_half=("ci_qc_hi_lt_0p5", "sum"),
                   coverage=("coverage_noisy", "mean"),
                   smd_after=("smd_after", "mean"))
              .reset_index())
    curves.to_csv(OUT / "curves.csv", index=False)

    summary: dict = {"n_rows": len(df), "alphas": ALPHAS, "blends": BLENDS}
    mono = []
    for (ds, noise, blend, det), g in curves.groupby(["dataset", "noise", "blend", "detector"]):
        g = g.sort_values("alpha")
        d = g["delta_q"].to_numpy(float)
        i = g["info_auc_q_vs_z"].to_numpy(float)
        mono.append({
            "dataset": ds, "noise": noise, "blend": blend, "detector": det,
            "delta_at_alpha0": float(d[0]), "delta_at_alpha1": float(d[-1]),
            "delta_gain": float(d[-1] - d[0]),
            "spearman_alpha_vs_delta": float(pd.Series(d).corr(pd.Series(g["alpha"].to_numpy()), method="spearman")),
            "spearman_info_vs_delta": float(pd.Series(d).corr(pd.Series(i), method="spearman")),
            "monotone_increasing": bool(np.all(np.diff(d) >= -0.005)),
        })
    mono_df = pd.DataFrame(mono)
    mono_df.to_csv(OUT / "monotonicity.csv", index=False)

    summary["mean_delta_at_alpha0"] = float(mono_df["delta_at_alpha0"].mean())
    summary["mean_delta_at_alpha1"] = float(mono_df["delta_at_alpha1"].mean())
    summary["frac_curves_gain_positive"] = float((mono_df["delta_gain"] > 0).mean())
    summary["frac_curves_monotone"] = float(mono_df["monotone_increasing"].mean())
    summary["mean_spearman_info_vs_delta"] = float(mono_df["spearman_info_vs_delta"].mean())
    summary["mean_spearman_alpha_vs_delta"] = float(mono_df["spearman_alpha_vs_delta"].mean())

    pool_tbl = (curves.groupby(["blend", "alpha"])
                .agg(info=("info_auc_q_vs_z", "mean"), delta=("delta_q", "mean"),
                     auc_qc=("auc_qc", "mean"), n=("delta_q", "size")).reset_index())
    summary["pooled_curve"] = pool_tbl.to_dict(orient="records")
    with open(OUT / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print("\n== pooled curve (mean over settings x detectors x seeds) ==")
    print(f"{'blend':8s} {'alpha':>6s} {'AUC(Q,Z)':>9s} {'AUC_QC':>8s} {'Delta_Q':>9s} {'n':>4s}")
    for _, r in pool_tbl.iterrows():
        print(f"{r['blend']:8s} {r['alpha']:6.1f} {r['info']:9.3f} {r['auc_qc']:8.3f} "
              f"{r['delta']:+9.3f} {int(r['n']):4d}")
    print(f"\ncurves gaining Delta_Q with alpha: {summary['frac_curves_gain_positive']:.2f}")
    print(f"monotone curves                  : {summary['frac_curves_monotone']:.2f}")
    print(f"mean Spearman(I(Q;Z), Delta_Q)   : {summary['mean_spearman_info_vs_delta']:+.3f}")
    print(f"\nwrote {OUT}/sweep.csv, curves.csv, monotonicity.csv, summary.json")


if __name__ == "__main__":
    main()
