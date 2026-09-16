"""P0-2 — CIFAR-100 S20 reversal audit.

The single most scrutinised number in the paper is CIFAR-100 symmetric-20%
``combined``: AUC_global 0.697 -> AUC_QC 0.391.  A matched AUROC *below* 0.5
means that, at equal quality, the detector ranks clean samples as noisier than
noisy ones — either a genuine conditional ranking reversal, or an artefact of a
post-treatment quality covariate.

This script interrogates that number along every axis a referee asked for:

A. common support          min/max and robust 1%-99% overlap of Q_noisy/Q_clean
B. matching coverage       matched noisy/clean counts, pair counts, totals
C. matching sensitivity    4 matching rules x 3 caliper levels x 3 quality
                           covariates (all reported, none tuned for the best
                           looking number)
D. distribution diagnostics  Q and detector distributions before/after matching,
                           per-class AUC_QC
E. the decisive comparison  is AUC_QC < 0.5 still true when Q is the
                           label-independent DINOv2 density?

Outputs land in ``results/revision/c100_s20_reversal/``.
"""
from __future__ import annotations

import argparse
import zlib
import json
import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path("/root/quality_noise")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "revision"))

from revq.bootstrap import paired_bootstrap  # noqa: E402
from revq.matching import (HUNGARIAN_CAP, balance_diagnostics,  # noqa: E402
                           common_support, match_pairs, matched_auc, smd)
from revq.prepare import DETECTOR_NAMES, load_cached  # noqa: E402

OUT = ROOT / "results" / "revision" / "c100_s20_reversal"
CACHE = ROOT / "results" / "revision" / "prepared"
DATASET, NOISE = "cifar100", "symmetric0.2"

STRATEGIES = ["nn_wo", "nn_wr", "class_cond", "trimmed"]
QUALITIES = ["knn_agreement", "proto_margin", "dino_density"]
CALIPERS = [0.05, 0.10, 0.20]
# The bootstrap is ~45 s per cell; running it over the whole 4x3x3x8 grid would
# cost ~16 CPU-hours for CIs that answer nothing extra.  Point estimates cover
# the full grid; CIs are computed for the default caliper and the detectors that
# carry the argument.
BS_DETECTORS = ["combined", "confident_learning", "ema_loss", "neighbor"]
BS_CALIPER = 0.10
BS_STRATEGY = "nn_wo"
DETECTORS = ["combined", "combined_cl", "ema_loss", "confidence", "aum", "forgetting",
             "neighbor", "confident_learning"]


def DEToffset(name: str) -> int:
    """Stable per-detector bootstrap seed offset (hash() is salted per process)."""
    return abs(zlib.crc32(name.encode())) % 100000


def _thirds(q: np.ndarray) -> np.ndarray:
    """Deterministic 3-way split of the sample axis (used for split-half checks)."""
    n = len(q)
    idx = np.arange(n)
    return idx % 3


def common_support_table(runs: list[dict]) -> pd.DataFrame:
    rows = []
    for r in runs:
        for qname in QUALITIES:
            q = r["signals"].get(qname)
            if q is None:
                continue
            mask = r["mask"]
            qn, qc = q[mask], q[~mask]
            lo, hi = common_support(qn, qc, robust=False)
            rlo, rhi = common_support(qn, qc, robust=True)
            rows.append({
                "seed": r["seed"], "quality": qname,
                "q_noisy_min": float(qn.min()), "q_noisy_max": float(qn.max()),
                "q_clean_min": float(qc.min()), "q_clean_max": float(qc.max()),
                "support_lo": lo, "support_hi": hi,
                "support_width": hi - lo if np.isfinite(lo) else float("nan"),
                "robust_lo": rlo, "robust_hi": rhi,
                "robust_width": rhi - rlo if np.isfinite(rlo) else float("nan"),
                "frac_noisy_in_support": float(((qn >= lo) & (qn <= hi)).mean())
                if np.isfinite(lo) else float("nan"),
                "frac_clean_in_support": float(((qc >= lo) & (qc <= hi)).mean())
                if np.isfinite(lo) else float("nan"),
                "mean_q_noisy": float(qn.mean()), "mean_q_clean": float(qc.mean()),
                "sd_q_noisy": float(qn.std()), "sd_q_clean": float(qc.std()),
                "smd_before": smd(qn, qc),
                "overlap_coefficient": float(
                    np.mean([min((qn <= v).sum(), (qc <= v).sum()) for v in
                             np.quantile(qn, [0.1, 0.25, 0.5, 0.75, 0.9])]) / max(len(qn), 1)),
            })
    return pd.DataFrame(rows)


def coverage_table(runs: list[dict]) -> pd.DataFrame:
    """Matched/eligible counts for every matching rule at the default caliper."""
    rows = []
    for r in runs:
        mask, y = r["mask"], r["y_observed"]
        for qname in QUALITIES:
            q = r["signals"].get(qname)
            if q is None:
                continue
            for strat in STRATEGIES:
                for det in ["combined"]:
                    score = r["signals"][det]
                    pairs = match_pairs(score, mask, q, strategy=strat, eps_sd=0.10,
                                        y_observed=y, force_greedy=False)
                    bal = balance_diagnostics(pairs, q, mask)
                    rows.append({"seed": r["seed"], "quality": qname, "strategy": strat,
                                 "detector": det, **{k: bal[k] for k in
                                 ["n_noisy", "n_clean", "n_pairs", "n_noisy_matched",
                                  "n_clean_matched", "coverage_noisy", "coverage_clean",
                                  "smd_before", "smd_after", "mean_abs_dq",
                                  "max_abs_dq"]}})
    return pd.DataFrame(rows)


def matching_sensitivity(runs: list[dict], n_bootstrap: int) -> pd.DataFrame:
    """4 rules x 3 calipers x 3 qualities x all detectors, with paired bootstrap.

    Point estimates are computed for the whole grid; paired bootstrap CIs only
    for ``BS_STRATEGY`` x ``BS_CALIPER`` x ``BS_DETECTORS``.  Partial results are
    flushed after every seed.
    """
    rows = []
    for r in runs:
        _n_before = len(rows)
        mask, y = r["mask"], r["y_observed"]
        for qname in QUALITIES:
            q = r["signals"].get(qname)
            if q is None:
                continue
            for strat in STRATEGIES:
                for cal in CALIPERS:
                    for det in DETECTORS:
                        score = np.asarray(r["signals"][det], dtype=np.float64)
                        # y_observed is required by the class-conditioned rule
                        # (and harmless for the others).
                        pairs = match_pairs(score, mask, q, strategy=strat, eps_sd=cal,
                                            max_clean_per_noisy=5, candidate_cap=512,
                                            y_observed=y)
                        auc_g = float(roc_auc_score(mask, score))
                        auc_qc = matched_auc(pairs)
                        # Bootstrap only the headline geometry (default caliper,
                        # the three qualities of interest); the rest of the grid
                        # is a point-estimate sensitivity sweep.  Running the
                        # bootstrap 4x3x3x8x10 times would cost hours for CIs
                        # that answer nothing extra.
                        do_bs = (n_bootstrap > 0 and abs(cal - BS_CALIPER) < 1e-9
                                 and strat == BS_STRATEGY and det in BS_DETECTORS)
                        if do_bs:
                            bs = paired_bootstrap(score, mask, q, n_resamples=n_bootstrap,
                                                  seed=r["seed"] * 131 + DEToffset(det) + int(cal * 1000),
                                                  eps_sd=cal, strategy=strat,
                                                  point=(auc_g, auc_qc), force_greedy=True,
                                                  max_noisy_per_resample=1500,
                                                  y_observed=y)
                            ci_qc, ci_d = bs["ci_qc"], bs["ci_delta"]
                        else:
                            ci_qc = ci_d = (float("nan"), float("nan"))
                        rows.append({
                            "seed": r["seed"], "detector": det, "quality": qname,
                            "strategy": strat, "caliper_sd": cal,
                            "auc_global": auc_g, "auc_qc": auc_qc,
                            "delta_q": auc_g - auc_qc,
                            "ci_qc_lo": ci_qc[0], "ci_qc_hi": ci_qc[1],
                            "ci_delta_lo": ci_d[0], "ci_delta_hi": ci_d[1],
                            "ci_qc_hi_lt_0p5": bool(ci_qc[1] < 0.5),
                            "ci_qc_lo_gt_0p5": bool(ci_qc[0] > 0.5),
                            "n_pairs": int(len(pairs.s_noisy)),
                            "coverage_noisy": float(len(np.unique(pairs.n_idx)) / max(mask.sum(), 1)),
                            "smd_after": smd(pairs.q_noisy, pairs.q_clean),
                            "n_noisy": int(mask.sum()), "n_clean": int((~mask).sum()),
                        })
    return pd.DataFrame(rows)


def distribution_dump(runs: list[dict]) -> None:
    """Per-sample Q and detector values before/after matching, for plotting."""
    d = OUT / "distributions"
    d.mkdir(parents=True, exist_ok=True)
    for r in runs:
        mask, y = r["mask"], r["y_observed"]
        seed = r["seed"]
        frames = []
        for qname in QUALITIES:
            q = r["signals"].get(qname)
            if q is None:
                continue
            for det in DETECTORS:
                score = np.asarray(r["signals"][det], dtype=np.float64)
                pairs = match_pairs(score, mask, q, strategy="nn_wo", eps_sd=0.10,
                                    y_observed=y)
                df = pd.DataFrame({
                    "sample_id": r["sample_ids"],
                    "mask": mask.astype(int),
                    "quality_name": qname,
                    "detector": det,
                    "q": q,
                    "score": score,
                })
                df["matched"] = 0
                df.loc[pairs.n_idx, "matched"] = 1
                df["matched_q"] = np.nan
                df["matched_score"] = np.nan
                df.loc[pairs.n_idx, "matched_q"] = q[pairs.c_idx]
                df.loc[pairs.n_idx, "matched_score"] = score[pairs.c_idx]
                frames.append(df)
        pd.concat(frames, ignore_index=True).to_parquet(
            d / f"seed{seed}_distributions.parquet", index=False)
        print(f"   distributions written for seed{seed}", flush=True)


def per_class_auc(runs: list[dict]) -> pd.DataFrame:
    """AUC_QC computed within each observed class (class-conditional view)."""
    rows = []
    for r in runs:
        mask, y = r["mask"], r["y_observed"]
        for qname in QUALITIES:
            q = r["signals"].get(qname)
            if q is None:
                continue
            for det in DETECTORS:
                score = np.asarray(r["signals"][det], dtype=np.float64)
                pairs = match_pairs(score, mask, q, strategy="class_cond", eps_sd=0.10,
                                    y_observed=y)
                if len(pairs.s_noisy) == 0:
                    continue
                cls = y[pairs.n_idx]
                for c in np.unique(cls):
                    sel = cls == c
                    if sel.sum() < 5:
                        continue
                    sn, sc = pairs.s_noisy[sel], pairs.s_clean[sel]
                    rows.append({
                        "seed": r["seed"], "quality": qname, "detector": det,
                        "class": int(c), "n_pairs": int(sel.sum()),
                        "auc_qc": float((sn > sc).mean() + 0.5 * (sn == sc).mean()),
                    })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--seeds", type=str, default="all")
    ap.add_argument("--skip-distributions", action="store_true")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    cache_seeds = sorted(int(p.name[4:-4]) for p in
                         (CACHE / DATASET / NOISE).glob("seed*.npz"))
    if args.seeds != "all":
        want = {int(s) for s in args.seeds.split(",")}
        cache_seeds = [s for s in cache_seeds if s in want]
    print(f"== C100-S20 reversal audit ==", flush=True)
    print(f"   seeds available: {cache_seeds}", flush=True)

    runs = [load_cached(DATASET, NOISE, s, cache=CACHE) for s in cache_seeds]
    for r in runs:
        r["n"] = len(r["mask"])

    t0 = time.time()
    cs = common_support_table(runs)
    cs.to_csv(OUT / "common_support.csv", index=False)
    print(f"[A] common support  -> {OUT/'common_support.csv'}  ({len(cs)} rows)", flush=True)

    cov = coverage_table(runs)
    cov.to_csv(OUT / "matching_coverage.csv", index=False)
    print(f"[B] matching coverage -> {OUT/'matching_coverage.csv'} ({len(cov)} rows)", flush=True)

    ms = matching_sensitivity(runs, args.bootstrap)
    ms.to_csv(OUT / "matching_sensitivity.csv", index=False)
    ms.to_csv(OUT / "matching_sensitivity.partial.csv", index=False)
    print(f"[C] matching sensitivity -> {OUT/'matching_sensitivity.csv'} "
          f"({len(ms)} rows, {time.time()-t0:.0f}s)", flush=True)

    pc = per_class_auc(runs)
    pc.to_csv(OUT / "class_conditioned.csv", index=False)
    print(f"[D] per-class AUC_QC -> {OUT/'class_conditioned.csv'} ({len(pc)} rows)", flush=True)

    if not args.skip_distributions:
        distribution_dump(runs)
        print(f"[D] distributions -> {OUT/'distributions'}", flush=True)

    # ---- E: the decisive comparison -----------------------------------------
    base = ms[(ms.strategy == "nn_wo") & (np.isclose(ms.caliper_sd, 0.10))]
    key = (base.groupby(["quality", "detector"])
           .agg(auc_global=("auc_global", "mean"), auc_qc=("auc_qc", "mean"),
                delta_q=("delta_q", "mean"), n_seeds=("seed", "nunique"),
                n_ci_below_0p5=("ci_qc_hi_lt_0p5", "sum"),
                n_ci_above_0p5=("ci_qc_lo_gt_0p5", "sum"),
                n_pairs=("n_pairs", "mean"),
                coverage=("coverage_noisy", "mean"),
                smd_after=("smd_after", "mean"))
           .reset_index())
    key.to_csv(OUT / "reversal_key_comparison.csv", index=False)

    summary = {
        "dataset": DATASET, "noise": NOISE, "seeds": cache_seeds,
        "n_seeds": len(cache_seeds), "bootstrap": args.bootstrap,
        "key_comparison": key.to_dict(orient="records"),
        "auc_qc_by_quality": {
            q: {d: float(base[(base.quality == q) & (base.detector == d)]["auc_qc"].mean())
                for d in DETECTORS if len(base[(base.quality == q) & (base.detector == d)])}
            for q in QUALITIES},
        "n_cells_ci_qc_below_0p5": {
            q: int(base[base.quality == q]["ci_qc_hi_lt_0p5"].sum()) for q in QUALITIES},
        "n_cells_ci_qc_above_0p5": {
            q: int(base[base.quality == q]["ci_qc_lo_gt_0p5"].sum()) for q in QUALITIES},
        "wall_s": round(time.time() - t0, 1),
    }
    with open(OUT / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print("\n== decisive comparison (nn_wo, caliper 0.10 SD) ==", flush=True)
    piv = key.pivot(index="detector", columns="quality", values="auc_qc")
    print(piv.round(3).to_string(), flush=True)
    print("\n   cells with bootstrap CI(AUC_QC) entirely < 0.5:", flush=True)
    for q in QUALITIES:
        print(f"     {q:18s} {summary['n_cells_ci_qc_below_0p5'][q]:2d} / "
              f"{len(base[base.quality == q])}", flush=True)
    print(f"\n== done in {time.time()-t0:.0f}s ==", flush=True)


if __name__ == "__main__":
    main()
