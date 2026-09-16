"""Item 3 — recompute the headline C100-S20 CIs with the PRIMARY estimator.

The main bootstrap used a vectorised caliper-band comparison (``matched_auc_fast``)
for speed.  A unit test showed that it is NOT numerically the same statistic as
the primary 1:1-without-replacement matching: mean |diff| ~ 0.011 with some
replicates differing by 0.03 at these sample sizes.  The point estimates are
unaffected (they use the primary estimator), but the *CIs* would then carry a
different estimand.

This script recomputes the CIs for the cells the paper actually leans on, with
the exact primary estimator re-run inside every replicate:

    4 detectors x 3 qualities x 10 seeds = 120 cells, B = 500

B = 500 is recorded explicitly; a percentile CI at B = 500 has a Monte-Carlo
error of ~0.002 on the 2.5/97.5 quantiles, well below the effects reported.
Cost: 500 x 152 ms ~ 76 s per cell (single core).
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
from sklearn.metrics import roc_auc_score

ROOT = Path("/root/quality_noise")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "revision"))

from revq.matching import match_pairs, matched_auc  # noqa: E402
from revq.prepare import load_cached  # noqa: E402

OUT = ROOT / "results" / "revision" / "c100_s20_reversal"
DETECTORS = ["combined", "confident_learning", "ema_loss", "neighbor"]
QUALITIES = ["knn_agreement", "proto_margin", "dino_density"]
# Kept small: the container caps memory at 2 GB, and the exact matching path
# holds several 14k-element arrays per replicate.
NOISY_BUDGET = 600
CLEAN_BUDGET = 4000


def _ci(v: np.ndarray) -> tuple[float, float]:
    v = v[np.isfinite(v)]
    if v.size == 0:
        return (float("nan"), float("nan"))
    return (float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975)))


def _worker(task: dict) -> list[dict]:
    """All detector x quality cells for ONE run, so the cache is loaded once."""
    seed, B, strategy = task["seed"], task["B"], task["strategy"]
    r = load_cached("cifar100", "symmetric0.2", seed)
    out = []
    for det in task["detectors"]:
        for qname in task["qualities"]:
            out.append(_cell(r, seed, det, qname, B, strategy))
            import gc
            gc.collect()
    return out


def _cell(r: dict, seed: int, det: str, qname: str, B: int, strategy: str) -> dict:
    mask = r["mask"]
    score = np.asarray(r["signals"][det], dtype=np.float64)
    q = np.asarray(r["signals"][qname], dtype=np.float64)

    auc_g = float(roc_auc_score(mask, score))
    pairs = match_pairs(score, mask, q, strategy=strategy, eps_sd=0.10,
                        force_greedy=False)
    auc_qc = matched_auc(pairs)

    n_idx = np.where(mask)[0]
    c_idx = np.where(~mask)[0]
    rng = np.random.default_rng(seed * 977 + abs(hash((det, qname))) % 99991)
    pool_n = n_idx if len(n_idx) <= NOISY_BUDGET else \
        np.random.default_rng(seed + 991).choice(n_idx, NOISY_BUDGET, replace=False)
    pool_c = c_idx if len(c_idx) <= CLEAN_BUDGET else \
        np.random.default_rng(seed + 992).choice(c_idx, CLEAN_BUDGET, replace=False)

    g = np.empty(B)
    m = np.empty(B)
    for b in range(B):
        bn = pool_n[rng.integers(0, len(pool_n), size=len(pool_n))]
        bc = pool_c[rng.integers(0, len(pool_c), size=len(pool_c))]
        ridx = np.concatenate([bn, bc])
        rm = mask[ridx]
        g[b] = roc_auc_score(rm, score[ridx])
        # EXACT primary estimator, re-run from scratch inside the replicate
        p = match_pairs(score[ridx], rm, q[ridx], strategy=strategy, eps_sd=0.10,
                        force_greedy=True, max_clean_per_noisy=5, candidate_cap=256)
        m[b] = matched_auc(p) if len(p.s_noisy) else np.nan
        del p, ridx, rm
    d = g - m
    return {
        "seed": seed, "detector": det, "quality": qname, "strategy": strategy,
        "B": B, "estimator": "primary_1to1_without_replacement",
        "auc_global": auc_g, "auc_qc": auc_qc, "delta_q": auc_g - auc_qc,
        "ci_qc_lo": _ci(m)[0], "ci_qc_hi": _ci(m)[1],
        "ci_delta_lo": _ci(d)[0], "ci_delta_hi": _ci(d)[1],
        "ci_qc_hi_lt_0p5": bool(_ci(m)[1] < 0.5),
        "ci_qc_lo_gt_0p5": bool(_ci(m)[0] > 0.5),
        "n_pairs": int(len(pairs.s_noisy)),
        "coverage_noisy": float(len(np.unique(pairs.n_idx)) / max(mask.sum(), 1)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=500)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--strategy", default="nn_wo")
    ap.add_argument("--seeds", type=str, default="all")
    ap.add_argument("--detectors-filter", type=str, default="",
                    help="comma list to restrict detectors (default: all four)")
    args = ap.parse_args()

    seeds = sorted(int(p.name[4:-4]) for p in
                   (ROOT / "results" / "revision" / "prepared" / "cifar100" /
                    "symmetric0.2").glob("seed*.npz"))
    if args.seeds != "all":
        want = {int(s) for s in args.seeds.split(",")}
        seeds = [s for s in seeds if s in want]

    dets = [d.strip() for d in args.detectors_filter.split(",") if d.strip()] or DETECTORS
    tasks = [{"seed": s, "detectors": dets, "qualities": QUALITIES,
              "B": args.B, "strategy": args.strategy} for s in seeds]
    print(f"== exact-estimator bootstrap: {len(tasks)} cells, B={args.B}, "
          f"{args.workers} workers ==", flush=True)

    t0 = time.time()
    rows = []
    partial = OUT / "bootstrap_exact_estimator.partial.csv"
    with mp.Pool(processes=args.workers) as pool:
        for i, res in enumerate(pool.imap_unordered(_worker, tasks)):
            rows.extend(res)
            if (i + 1) % 4 == 0 or (i + 1) == len(tasks):
                pd.DataFrame(rows).to_csv(partial, index=False)
                print(f"   {i+1}/{len(tasks)} ({time.time()-t0:.0f}s)", flush=True)

    df = pd.DataFrame(rows).sort_values(["quality", "detector", "seed"])
    df.to_csv(OUT / "bootstrap_exact_estimator.csv", index=False)

    agg = (df.groupby(["quality", "detector"])
           .agg(n_seeds=("seed", "nunique"),
                auc_global=("auc_global", "mean"), auc_qc=("auc_qc", "mean"),
                delta_q=("delta_q", "mean"),
                n_ci_qc_below_half=("ci_qc_hi_lt_0p5", "sum"),
                n_ci_qc_above_half=("ci_qc_lo_gt_0p5", "sum"),
                coverage=("coverage_noisy", "mean"))
           .reset_index())
    agg.to_csv(OUT / "bootstrap_exact_estimator_summary.csv", index=False)

    print("\n== C100-S20, CIs from the PRIMARY estimator (1:1 without replacement) ==")
    print(agg.round(4).to_string(index=False))
    print(f"\nwall {time.time()-t0:.0f}s")
    print(f"wrote {OUT/'bootstrap_exact_estimator.csv'}")


if __name__ == "__main__":
    main()
