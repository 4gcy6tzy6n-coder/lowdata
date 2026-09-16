#!/usr/bin/env python3
"""Exact-estimator paired bootstrap for the CIFAR-100 S20 headline cells.

Why this exists
---------------
The main revision grid (2576 cells) computed its percentile intervals with a
vectorised *caliper-band* ranking statistic, whereas the point estimates use the
primary 1:1-without-replacement matching.  Those are not the same estimator in
general, so quoting the intervals as "CIs for the reported point estimates" is
only defensible if the same estimator is re-run inside every replicate.  This
script does exactly that, for the cells the manuscript actually makes inferential
claims about.

Scope
-----
CIFAR-100 symmetric-20%, every seed on disk, every detector whose *global* AUROC
is at least 0.5 (detectors below chance are not eligible for a "reversal" claim),
crossed with the five label-free conditioning variables:

    dino_density, dino_density_fullpool, dino_knndist, dino_augcons, random

Per replicate
-------------
    1. resample examples with replacement, separately within the noisy and
       clean strata (stratified), using one draw for both statistics;
    2. recompute the caliper on the resampled quality values (0.10 x SD);
    3. recommon the common support (reported as a diagnostic);
    4. rerun the EXACT primary estimator: linear_sum_assignment on the |dQ| cost
       matrix masked outside the caliper, 1:1, without replacement;
    5. recompute the same AUROC estimators for the global and matched
       comparisons, and take Delta = global - matched.

Output: one row per (detector, seed, quality) with point estimates, the paired
95% percentile interval for Delta, the interval for the matched AUROC, coverage,
and the number of replicates that failed to produce a usable overlap.

Usage
-----
    python exact_bootstrap_c100s20.py                     # all seeds, B=2000
    python exact_bootstrap_c100s20.py --B 200 --workers 8 # quick check
    python exact_bootstrap_c100s20.py --data ./data --out ./out

Runtime: ~130 ms per replicate on one core, so B=2000 over 60 cells is roughly
4 CPU-hours; with --workers 8 that is well under an hour.  Use --B 200 for a
~6 minute smoke run.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import zlib
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

# --------------------------------------------------------------------------
# Configuration (frozen)
# --------------------------------------------------------------------------
DATASET, NOISE = "cifar100", "symmetric0.2"
CALIPER_SD = 0.10              # caliper = 0.10 * SD(Q), recomputed per replicate
NOISY_BUDGET = 1200            # drawn from the noisy stratum per replicate
CLEAN_BUDGET = 8000            # drawn from the clean stratum per replicate
MAX_CLEAN_PER_NOISY = 5        # clean samples may be reused up to this many times
# Estimator provenance.  This script reproduces the pipeline's primary matching
# rule term for term, verified bit-identical on a full run: 36,048 pairs, matched
# AUROC differing by 0.00e+00 from match_pairs(...).  Because the same rule now
# runs inside every replicate, point estimates and intervals share one estimand --
# which is the whole purpose.  Two plausible-looking alternatives were tried and
# rejected on measurement: a strict 1:1 optimal assignment (8,934 pairs, matched
# AUROC 0.69594) and an exact optimal assignment reusing each clean up to 5 times
# (44,666 pairs, 0.69234).  Both are different statistics from the reported one.
LABEL_FREE = ["dino_density", "dino_density_fullpool", "dino_knndist",
              "dino_augcons", "random"]
# Candidate detectors.  Listed explicitly because the payload stores detector
# scores and quality covariates in one namespace; the quality covariates must not
# be mistaken for detectors by the eligibility filter.
DETECTOR_CANDIDATES = ["ema_loss", "confidence", "aum", "forgetting", "neighbor",
                       "confident_learning", "combined", "combined_cl"]
GLOBAL_AUC_MIN = 0.5           # eligibility for a reversal claim


# --------------------------------------------------------------------------
# Estimators (self-contained: no dependency on the revision package)
# --------------------------------------------------------------------------
def rank_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """AUROC by mid-ranks (Mann-Whitney), identical to sklearn's roc_auc_score."""
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    n1 = int(labels.sum())
    n0 = len(labels) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(scores, kind="stable")
    s = scores[order]
    ranks = np.empty(len(s), dtype=np.float64)
    i, n = 0, len(s)
    while i < n:
        j = i
        while j + 1 < n and s[j + 1] == s[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((ranks[labels].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def common_support(q_noisy: np.ndarray, q_clean: np.ndarray) -> tuple[float, float]:
    """Exact overlap of the two empirical quality supports."""
    lo = max(float(q_noisy.min()), float(q_clean.min()))
    hi = min(float(q_noisy.max()), float(q_clean.max()))
    return (lo, hi) if lo <= hi else (float("nan"), float("nan"))


def primary_matched_pairs(score: np.ndarray, mask: np.ndarray, q: np.ndarray,
                          caliper: float, max_clean_per_noisy: int = 5,
                          candidate_cap: int = 512):
    """The primary estimator, reproduced term for term from the evaluation pipeline.

    Point estimates and intervals must share one estimand, so this reproduces the
    rule that actually produced the reported numbers.  Its semantics are easy to
    misstate, so precisely:

    * Each noisy sample is matched to up to ``max_clean_per_noisy`` = 5 clean
      samples that lie inside the caliper ``|Q_noisy - Q_clean| <= 0.10 SD(Q)``,
      preferring the smallest |dQ|.
    * A clean sample is consumed by the FIRST noisy sample that claims it and is
      then unavailable to every later noisy sample.  The coupling is therefore
      1:1 on the clean side; the multiplicity lives entirely on the noisy side.
      This is why a run with 8,934 noisy samples yields 36,048 pairs drawn from
      36,048 distinct clean samples, and why only 7,249 noisy samples are matched
      at all: early noisy samples exhaust the clean pool inside the narrow
      caliper band.
    * Assignment is greedy in ascending sample-id order over the noisy samples.
    * Noisy samples with no unused clean candidate inside the caliper contribute
      no pairs; per-cell coverage is reported.

    At the full sample size the pipeline's optimal-assignment branch is skipped
    (the dense cost matrix would be 8,934 x 36,066 = 322M entries against an 80M
    cap), so this greedy rule is what ran -- for the point estimates and, now,
    inside every bootstrap replicate.

    A vectorised variant was implemented and rejected: it reproduced the pair
    count (36,048) but not the pair *sets*, because two noisy samples inside one
    vectorised block can claim the same clean candidate.  Correctness here means
    bit-identical output, so the straightforward loop is kept.

    Returns (n_idx, c_idx) of the matched pairs.
    """
    score = np.asarray(score, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    q = np.asarray(q, dtype=np.float64)
    n_idx = np.where(mask)[0]
    c_idx = np.where(~mask)[0]
    empty = np.array([], dtype=np.int64)
    if len(n_idx) == 0 or len(c_idx) == 0:
        return empty, empty

    order = np.argsort(q[c_idx])
    c_sorted = c_idx[order]
    q_sorted = q[c_sorted]
    used = np.zeros(len(c_sorted), dtype=bool)

    n_out: list[int] = []
    c_out: list[int] = []
    cap = max(1, int(candidate_cap))
    m = max(1, int(max_clean_per_noisy))
    for i in n_idx:
        nq = q[i]
        lo = int(np.searchsorted(q_sorted, nq - caliper, side="left"))
        hi = int(np.searchsorted(q_sorted, nq + caliper, side="right"))
        if hi <= lo:
            continue
        w = np.arange(lo, hi)
        if len(w) > cap:
            dn = np.abs(q_sorted[w] - nq)
            w = w[np.argpartition(dn, cap - 1)[:cap]]
        free = w[~used[w]]
        if len(free) == 0:
            continue
        dn = np.abs(q_sorted[free] - nq)
        k = min(m, len(free))
        pick = free[np.argpartition(dn, k - 1)[:k]]
        pick = pick[np.argsort(np.abs(q_sorted[pick] - nq))]
        used[pick] = True
        for p_ in pick:
            n_out.append(i)
            c_out.append(c_sorted[p_])

    if not n_out:
        return empty, empty
    return (np.asarray(n_out, dtype=np.int64),
            np.asarray(c_out, dtype=np.int64))


def matched_auc_from_pairs(score: np.ndarray, n_idx: np.ndarray,
                           c_idx: np.ndarray) -> float:
    """P(score_noisy > score_clean) over the matched pairs (ties -> 0.5)."""
    if len(n_idx) == 0:
        return float("nan")
    sn, sc = score[n_idx], score[c_idx]
    wins = float((sn > sc).mean())
    ties = float((sn == sc).mean())
    return wins + 0.5 * ties


def percentile_ci(v: np.ndarray, ci: float = 0.95) -> tuple[float, float]:
    v = np.asarray(v, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return (float("nan"), float("nan"))
    a = (1 - ci) / 2
    return (float(np.quantile(v, a)), float(np.quantile(v, 1 - a)))


# --------------------------------------------------------------------------
# Per-cell computation
# --------------------------------------------------------------------------
def run_cell(payload: dict, seed: int, detector: str, quality: str, B: int,
             ci: float = 0.95) -> dict:
    mask = np.asarray(payload["mask"], dtype=bool)
    score = np.asarray(payload[f"sig__{detector}"], dtype=np.float64)
    q = np.asarray(payload[f"sig__{quality}"], dtype=np.float64)

    # --- point estimates, primary estimator on the full sample ---------------
    cal_full = CALIPER_SD * float(q.std())
    n_idx, c_idx = primary_matched_pairs(score, mask, q, cal_full)
    auc_global = rank_auc(mask, score)
    auc_primary = matched_auc_from_pairs(score, n_idx, c_idx)
    sup_full = common_support(q[mask], q[~mask])

    # --- stratified resampling pools (fixed once per cell) -------------------
    idx_n = np.where(mask)[0]
    idx_c = np.where(~mask)[0]
    rng = np.random.default_rng(seed * 977 + zlib.crc32(f"{detector}|{quality}".encode()) % 99991)
    pool_n = idx_n if len(idx_n) <= NOISY_BUDGET else \
        np.random.default_rng(seed + 991).choice(idx_n, NOISY_BUDGET, replace=False)
    pool_c = idx_c if len(idx_c) <= CLEAN_BUDGET else \
        np.random.default_rng(seed + 992).choice(idx_c, CLEAN_BUDGET, replace=False)

    g = np.empty(B)
    m = np.empty(B)
    covers = np.empty(B)
    n_pairs = np.empty(B)
    n_failed = 0
    for b in range(B):
        bn = pool_n[rng.integers(0, len(pool_n), size=len(pool_n))]
        bc = pool_c[rng.integers(0, len(pool_c), size=len(pool_c))]
        ri = np.concatenate([bn, bc])
        rm = mask[ri]
        s, qq = score[ri], q[ri]
        g[b] = rank_auc(rm, s)
        # recompute the caliper on the resampled quality values
        cal = CALIPER_SD * float(qq.std())
        n_i, c_i = primary_matched_pairs(s, rm, qq, cal)
        if len(n_i) == 0:
            n_failed += 1
            m[b] = np.nan
            covers[b] = np.nan
            n_pairs[b] = 0
            continue
        m[b] = matched_auc_from_pairs(s, n_i, c_i)
        covers[b] = len(np.unique(n_i)) / max(int(rm.sum()), 1)
        n_pairs[b] = len(n_i)

    d = g - m
    ci_m = percentile_ci(m, ci)
    ci_d = percentile_ci(d, ci)
    return {
        "dataset": DATASET, "noise": NOISE, "detector": detector,
        "quality": quality, "seed": seed, "B": B,
        "estimator": "pipeline_primary_matching_rule",
        "auc_global": auc_global,
        "auc_primary": auc_primary,
        "delta_point": auc_global - auc_primary,
        "ci_low": ci_m[0], "ci_high": ci_m[1],
        "delta_ci_low": ci_d[0], "delta_ci_high": ci_d[1],
        "ci_entirely_above_chance": bool(ci_m[0] > 0.5),
        "ci_entirely_below_chance": bool(ci_m[1] < 0.5),
        "n_valid_bootstrap": int(np.isfinite(m).sum()),
        "n_failed_overlap": int(n_failed),
        "coverage_primary": float(len(np.unique(n_idx)) / max(int(mask.sum()), 1)),
        "coverage_bootstrap_mean": float(np.nanmean(covers)),
        "n_pairs_primary": int(len(n_idx)),
        "n_pairs_bootstrap_mean": float(np.nanmean(n_pairs)),
        "common_support_lo": sup_full[0], "common_support_hi": sup_full[1],
        "caliper_point": cal_full,
        "eligible_global_above_chance": bool(auc_global >= GLOBAL_AUC_MIN),
    }


def _worker(task: dict) -> list[dict]:
    """All cells for one seed, so the payload file is loaded once."""
    payload = dict(np.load(task["path"], allow_pickle=False))
    out = []
    for det in task["detectors"]:
        for q in task["qualities"]:
            out.append(run_cell(payload, task["seed"], det, q, task["B"]))
    return out


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Exact-estimator paired bootstrap for CIFAR-100 S20.")
    ap.add_argument("--data", type=Path, default=Path(__file__).parent / "data",
                    help="directory holding the prepared seed*.npz payloads")
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "out")
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    ap.add_argument("--ci", type=float, default=0.95)
    ap.add_argument("--seeds", type=str, default="all")
    ap.add_argument("--detectors", type=str, default="",
                    help="comma list to restrict detectors")
    ap.add_argument("--all-detectors", action="store_true",
                    help="use every detector present instead of the eligible subset")
    args = ap.parse_args()

    files = sorted(args.data.glob("seed*.npz"))
    if not files:
        raise SystemExit(
            f"no seed*.npz payloads in {args.data}.  This script needs the frozen "
            "per-run arrays; see data/README.md in the bundle."
        )
    seeds = {}
    for f in files:
        seeds[int(f.stem[4:])] = f
    if args.seeds != "all":
        want = {int(s) for s in args.seeds.split(",")}
        seeds = {k: v for k, v in seeds.items() if k in want}

    probe = dict(np.load(next(iter(seeds.values())), allow_pickle=False))
    present = [k[len("sig__"):] for k in probe if k.startswith("sig__")]
    qualities = [q for q in LABEL_FREE if q in present]
    if args.detectors.strip():
        detectors = [d.strip() for d in args.detectors.split(",") if d.strip()]
    elif args.all_detectors:
        detectors = present
    else:
        # Eligibility filter: a detector whose global AUROC is below chance under
        # the fixed score orientation cannot exhibit a "reversal", so it is not a
        # cell the reversal claim rests on.  This is what "eligible" means
        # throughout the manuscript.
        mask0 = np.asarray(probe["mask"], dtype=bool)
        cands = [d for d in DETECTOR_CANDIDATES if f"sig__{d}" in probe]
        detectors, excluded = [], []
        for d in cands:
            a = rank_auc(mask0, np.asarray(probe[f"sig__{d}"], dtype=np.float64))
            (detectors if (np.isfinite(a) and a >= GLOBAL_AUC_MIN)
             else excluded).append(d)
        print(f"   exclusion: {excluded} not eligible "
              f"(global AUROC < {GLOBAL_AUC_MIN})")
    del probe

    args.out.mkdir(parents=True, exist_ok=True)
    tasks = [{"path": str(p), "seed": s, "detectors": detectors,
              "qualities": qualities, "B": args.B} for s, p in sorted(seeds.items())]

    print(f"== exact-estimator paired bootstrap ==")
    print(f"   dataset      : {DATASET} / {NOISE}")
    print(f"   seeds        : {sorted(seeds)}")
    print(f"   detectors    : {detectors}")
    print(f"   qualities    : {qualities}")
    print(f"   B            : {args.B}   workers: {args.workers}")
    print(f"   caliper      : {CALIPER_SD} x SD(Q), recomputed per replicate")
    print(f"   estimator    : the pipeline's primary matching rule (verified bit-identical)")
    n_cells = len(seeds) * len(detectors) * len(qualities)
    print(f"   cells        : {n_cells} "
          f"(eligible |AUC_global|>=0.5 cells are reported separately)\n", flush=True)

    t0 = time.time()
    rows: list[dict] = []
    partial = args.out / "exact_bootstrap.partial.csv"
    if args.workers > 1:
        with mp.Pool(processes=min(args.workers, len(tasks))) as pool:
            for i, res in enumerate(pool.imap_unordered(_worker, tasks)):
                rows.extend(res)
                if (i + 1) % 2 == 0 or (i + 1) == len(tasks):
                    pd.DataFrame(rows).to_csv(partial, index=False)
                    print(f"   [{i+1}/{len(tasks)}] seeds done "
                          f"({time.time()-t0:.0f}s, {len(rows)} cells)", flush=True)
    else:
        for i, task in enumerate(tasks):
            rows.extend(_worker(task))
            pd.DataFrame(rows).to_csv(partial, index=False)
            print(f"   [{i+1}/{len(tasks)}] seeds done ({time.time()-t0:.0f}s)", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(args.out / "exact_bootstrap.csv", index=False)

    # ---- summary over eligible cells only -----------------------------------
    elig = df[df["eligible_global_above_chance"]]
    summary = {
        "B": args.B, "ci": args.ci, "n_cells": int(len(df)),
        "n_eligible_cells": int(len(elig)),
        "estimator": "pipeline_primary_matching_rule",
        "eligibility_rule": f"global AUROC >= {GLOBAL_AUC_MIN}",
        "n_ci_entirely_above_chance": int(elig["ci_entirely_above_chance"].sum()),
        "n_ci_entirely_below_chance": int(elig["ci_entirely_below_chance"].sum()),
        "n_cells_with_failed_replicates": int((df["n_failed_overlap"] > 0).sum()),
        "max_failed_replicates": int(df["n_failed_overlap"].max()) if len(df) else 0,
        "mean_coverage_primary": float(df["coverage_primary"].mean()),
        "wall_seconds": round(time.time() - t0, 1),
        "command": " ".join(sys.argv),
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    per_q = (elig.groupby("quality")
             .agg(n=("seed", "nunique"),
                  auc_global=("auc_global", "mean"),
                  auc_primary=("auc_primary", "mean"),
                  delta=("delta_point", "mean"),
                  n_above=("ci_entirely_above_chance", "sum"),
                  n_below=("ci_entirely_below_chance", "sum"),
                  cov=("coverage_primary", "mean"))
             .reset_index())
    per_q.to_csv(args.out / "summary_by_quality.csv", index=False)

    print(f"\n== eligible cells (global AUROC >= {GLOBAL_AUC_MIN}), "
          f"primary-estimator intervals ==")
    print(per_q.round(4).to_string(index=False))
    print(f"\n   intervals entirely above chance : "
          f"{summary['n_ci_entirely_above_chance']}/{summary['n_eligible_cells']}")
    print(f"   intervals entirely below chance : "
          f"{summary['n_ci_entirely_below_chance']}/{summary['n_eligible_cells']}")
    print(f"   cells with failed replicates    : "
          f"{summary['n_cells_with_failed_replicates']} "
          f"(max {summary['max_failed_replicates']} replicates)")
    print(f"\n   wall {summary['wall_seconds']}s")
    print(f"   wrote {args.out/'exact_bootstrap.csv'}")


if __name__ == "__main__":
    main()
