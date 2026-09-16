#!/usr/bin/env python3
"""P1 experiment batch — alternative conditioning and retention diagnostics.

Two questions, both on C100-S20 with the local ``prepared`` cache:

J6  Alternative conditioning estimators
    Is the phenomenon specific to the pipeline's hard-caliper ``nn_wo`` rule?
    Three estimators are computed on the same cells:
      * hard caliper      the pipeline rule (reference)
      * stratified        Q split into K quantile strata, AUROC computed within
                          each stratum, then aggregated by pair-share weighting
                          (and by sample-size weighting, reported separately)
      * kernel            every noisy-clean pair is kept but weighted by a kernel
                          on |q_i - q_j|, with Hajek normalisation so an exact
                          match counts once and distant pairs fade continuously

J7  Retained vs excluded diagnostics
    The primary rule matches only a fraction of noisy samples.  This job reports
    what the excluded ones look like: detector-score distribution, quality
    distribution and the detector's own per-subset AUROC, for noisy and clean
    samples separately.  That is the quantity behind the j2 finding that the gap
    is within-support *weighting* rather than support restriction.

Usage
-----
    python run_p1_alternatives.py --job j6 --workers 8
    python run_p1_alternatives.py --job j7 --workers 8
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
from run_p0_batch import (CACHE, DATASET, MAX_CLEAN, NOISE, PROXY_FAMILY,  # noqa: E402
                          PROXIES, DETECTORS, matched_pairs, pairwise_auc,
                          rank_auc)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "revision" / "p1_alternatives"
CALIPER_SD = 0.10
KERNEL_BW = 0.10          # bandwidth in SD(Q) units, matching the caliper
STRATA = [5, 10]
UNCOUPLED = ["ema_loss", "confidence", "aum", "confident_learning"]


def load_payload(seed: int) -> dict:
    return dict(np.load(CACHE / DATASET / NOISE / f"seed{seed}.npz", allow_pickle=False))


def available_seeds() -> list[int]:
    return sorted(int(p.stem[4:]) for p in (CACHE / DATASET / NOISE).glob("seed*.npz"))


# ---------------------------------------------------------------------------
# estimator A: quantile-stratified
# ---------------------------------------------------------------------------
def stratified_auc(score: np.ndarray, mask: np.ndarray, q: np.ndarray,
                   n_strata: int) -> dict:
    """AUROC within Q quantile strata, aggregated two ways.

    ``pair_weighted`` weights each stratum by its share of comparable pairs,
    which is the analogue of what the matched estimator does.  ``sample_weighted``
    weights by the stratum's share of samples.  Both are reported because they
    answer different questions and can disagree.
    """
    edges = np.quantile(q, np.linspace(0, 1, n_strata + 1))
    edges = np.unique(edges)
    if len(edges) < 3:
        return {"pair_weighted": float("nan"), "sample_weighted": float("nan"),
                "strata_used": 0, "min_stratum_pairs": 0}
    b = np.clip(np.digitize(q, edges[1:-1]), 0, len(edges) - 2)
    aucs, p_pairs, n_samp = [], [], []
    for s in range(len(edges) - 1):
        sel = b == s
        m = mask & sel
        c = (~mask) & sel
        n1, n0 = int(m.sum()), int(c.sum())
        if n1 == 0 or n0 == 0:
            continue
        lab = np.concatenate([np.ones(n1, bool), np.zeros(n0, bool)])
        idx = np.concatenate([np.where(m)[0], np.where(c)[0]])
        a = rank_auc(lab, score[idx])
        if not np.isfinite(a):
            continue
        aucs.append(a); p_pairs.append(n1 * n0); n_samp.append(n1 + n0)
    if not aucs:
        return {"pair_weighted": float("nan"), "sample_weighted": float("nan"),
                "strata_used": 0, "min_stratum_pairs": 0}
    aucs = np.asarray(aucs); p_pairs = np.asarray(p_pairs, float); n_samp = np.asarray(n_samp, float)
    return {
        "pair_weighted": float((aucs * p_pairs).sum() / p_pairs.sum()),
        "sample_weighted": float((aucs * n_samp).sum() / n_samp.sum()),
        "strata_used": int(len(aucs)),
        "min_stratum_pairs": int(p_pairs.min()),
    }


# ---------------------------------------------------------------------------
# estimator B: kernel-weighted pairwise AUROC
# ---------------------------------------------------------------------------
def kernel_auc(score: np.ndarray, mask: np.ndarray, q: np.ndarray,
               bw: float, chunk: int = 512) -> dict:
    """P(score_noisy > score_clean) with KC(|q_i - q_j|) weights, Hajek-normalised.

    ``KC`` is the Epanechnikov kernel on |dq|/bw.  Dividing by the summed weights
    makes the result invariant to the overall density of comparable pairs, so it
    is comparable across bandwidths and across covariates instead of being
    dominated by whichever covariate happens to produce more near-ties.
    """
    n_idx = np.where(mask)[0]
    c_idx = np.where(~mask)[0]
    if len(n_idx) == 0 or len(c_idx) == 0:
        return {"auc": float("nan"), "n_eff": float("nan"), "w_sum": 0.0}
    sn_all, qn_all = score[n_idx], q[n_idx]
    sc_all, qc_all = score[c_idx], q[c_idx]
    wins = 0.0; ties = 0.0; wsum = 0.0; w2 = 0.0
    for s in range(0, len(n_idx), chunk):
        e = min(s + chunk, len(n_idx))
        dq = np.abs(qn_all[s:e][:, None] - qc_all[None, :])
        w = 0.75 * np.clip(1.0 - (dq / bw) ** 2, 0.0, None) / bw   # Epanechnikov
        sn = sn_all[s:e][:, None]; sc = sc_all[None, :]
        wins += float((w * (sn > sc)).sum())
        ties += float((w * (sn == sc)).sum())
        wsum += float(w.sum()); w2 += float((w ** 2).sum())
    if wsum <= 0:
        return {"auc": float("nan"), "n_eff": float("nan"), "w_sum": 0.0}
    return {"auc": (wins + 0.5 * ties) / wsum,
            "n_eff": wsum ** 2 / w2 if w2 > 0 else float("nan"),
            "w_sum": wsum}


# ---------------------------------------------------------------------------
# workers
# ---------------------------------------------------------------------------
def _worker_j6(task: dict) -> list[dict]:
    p = load_payload(task["seed"])
    mask = np.asarray(p["mask"], dtype=bool)
    out = []
    for det in task["detectors"]:
        score = np.asarray(p[f"sig__{det}"], dtype=np.float64)
        for px in task["proxies"]:
            q = np.asarray(p[f"sig__{px}"], dtype=np.float64)
            sd = float(q.std())
            ni, ci = matched_pairs(score, mask, q, CALIPER_SD * sd, MAX_CLEAN)
            row = {
                "dataset": DATASET, "seed": task["seed"], "detector": det,
                "proxy": px, "proxy_family": PROXY_FAMILY.get(px, "?"),
                "global_auc": rank_auc(mask, score),
                "hard_caliper_auc": pairwise_auc(score, ni, ci),
                "hard_caliper_coverage": len(np.unique(ni)) / max(int(mask.sum()), 1),
                "hard_caliper_n_pairs": int(len(ni)),
            }
            for K in STRATA:
                r = stratified_auc(score, mask, q, K)
                row[f"stratified{K}_pair_weighted"] = r["pair_weighted"]
                row[f"stratified{K}_sample_weighted"] = r["sample_weighted"]
                row[f"stratified{K}_strata_used"] = r["strata_used"]
                row[f"stratified{K}_min_stratum_pairs"] = r["min_stratum_pairs"]
            for bw in (KERNEL_BW,):
                r = kernel_auc(score, mask, q, bw * sd)
                row[f"kernel_bw{bw}_auc"] = r["auc"]
                row[f"kernel_bw{bw}_n_eff"] = r["n_eff"]
            out.append(row)
    return out


def _worker_j7(task: dict) -> list[dict]:
    p = load_payload(task["seed"])
    mask = np.asarray(p["mask"], dtype=bool)
    out = []
    for det in task["detectors"]:
        score = np.asarray(p[f"sig__{det}"], dtype=np.float64)
        for px in task["proxies"]:
            q = np.asarray(p[f"sig__{px}"], dtype=np.float64)
            ni, ci = matched_pairs(score, mask, q, CALIPER_SD * float(q.std()), MAX_CLEAN)
            rn = np.zeros(len(mask), dtype=bool); rn[np.unique(ni)] = True
            rc = np.zeros(len(mask), dtype=bool); rc[np.unique(ci)] = True
            # noisy samples split into matched / unmatched; clean likewise
            for group, sel in (("noisy_retained", mask & rn),
                               ("noisy_excluded", mask & ~rn),
                               ("clean_retained", (~mask) & rc),
                               ("clean_excluded", (~mask) & ~rc)):
                if sel.sum() == 0:
                    continue
                out.append({
                    "dataset": DATASET, "seed": task["seed"], "detector": det,
                    "proxy": px, "proxy_family": PROXY_FAMILY.get(px, "?"),
                    "group": group, "n_samples": int(sel.sum()),
                    "score_mean": float(score[sel].mean()),
                    "score_median": float(np.median(score[sel])),
                    "score_std": float(score[sel].std()),
                    "q_mean": float(q[sel].mean()),
                    "q_median": float(np.median(q[sel])),
                })
            # detector AUC restricted to the retained population
            n_r, c_r = np.where(mask & rn)[0], np.where((~mask) & rc)[0]
            a_ret = float("nan")
            if len(n_r) and len(c_r):
                lab = np.concatenate([np.ones(len(n_r), bool), np.zeros(len(c_r), bool)])
                a_ret = rank_auc(lab, score[np.concatenate([n_r, c_r])])
            out.append({
                "dataset": DATASET, "seed": task["seed"], "detector": det,
                "proxy": px, "proxy_family": PROXY_FAMILY.get(px, "?"),
                "group": "auc_retained_population", "n_samples": len(n_r) + len(c_r),
                "score_mean": a_ret, "score_median": float("nan"),
                "score_std": float("nan"), "q_mean": float("nan"),
                "q_median": float("nan"),
            })
    return out


def run_pooled(tasks, worker, workers, out_csv, label):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time(); rows = []
    with mp.Pool(processes=min(workers, len(tasks))) as pool:
        for i, res in enumerate(pool.imap_unordered(worker, tasks)):
            rows.extend(res)
            pd.DataFrame(rows).to_csv(out_csv.with_suffix(".partial.csv"), index=False)
            print(f"   [{i+1}/{len(tasks)}] {label} ({time.time()-t0:.0f}s, {len(rows)} rows)",
                  flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"\nwrote {out_csv} ({len(df)} rows, {time.time()-t0:.0f}s)")
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True, choices=["j6", "j7"])
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    ap.add_argument("--seeds", type=str, default="all")
    ap.add_argument("--detectors", type=str, default=",".join(UNCOUPLED))
    ap.add_argument("--proxies", type=str, default=",".join(PROXIES))
    args = ap.parse_args()

    seeds = available_seeds()
    if args.seeds != "all":
        want = {int(s) for s in args.seeds.split(",")}
        seeds = [s for s in seeds if s in want]
    dets = [d.strip() for d in args.detectors.split(",") if d.strip()]
    pxs = [x.strip() for x in args.proxies.split(",") if x.strip()]
    print(f"== P1 alternatives: {args.job} ==  seeds={seeds}  detectors={dets}")

    tasks = [{"seed": s, "detectors": dets, "proxies": pxs} for s in seeds]

    if args.job == "j6":
        df = run_pooled(tasks, _worker_j6, args.workers,
                        OUT / "j6_alternative_conditioning.csv", "j6")
        _summarise_j6(df)
    else:
        df = run_pooled(tasks, _worker_j7, args.workers,
                        OUT / "j7_retention_diagnostics.csv", "j7")
        _summarise_j7(df)


def _summarise_j6(df: pd.DataFrame) -> None:
    print("\n== j6: estimator comparison (mean over 4 detectors x 10 seeds) ==")
    g = (df.groupby("proxy")
         .agg(hard=("hard_caliper_auc", "mean"),
              strat5_pair=("stratified5_pair_weighted", "mean"),
              strat5_samp=("stratified5_sample_weighted", "mean"),
              strat10_pair=("stratified10_pair_weighted", "mean"),
              strat10_samp=("stratified10_sample_weighted", "mean"),
              kernel=("kernel_bw0.1_auc", "mean"),
              cover=("hard_caliper_coverage", "mean"),
              min_pairs=("stratified10_min_stratum_pairs", "min"))
         .reset_index())
    print(g.round(4).to_string(index=False))
    g.to_csv(OUT / "j6_summary.csv", index=False)
    print("\n  spread across estimators (max - min of the five non-reference columns):")
    cols = ["strat5_pair", "strat5_samp", "strat10_pair", "strat10_samp", "kernel"]
    for _, r in g.iterrows():
        v = np.array([r[c] for c in cols], dtype=float)
        print(f"    {r['proxy']:24s} spread = {np.nanmax(v)-np.nanmin(v):.4f}")


def _summarise_j7(df: pd.DataFrame) -> None:
    print("\n== j7: retained vs excluded (mean over 4 detectors x 10 seeds) ==")
    det = df[df.group != "auc_retained_population"]
    piv = det.pivot_table(index=["proxy", "group"], values="score_mean",
                          aggfunc="mean").reset_index()
    print(piv.round(4).to_string(index=False))
    auc = df[df.group == "auc_retained_population"][["proxy", "score_mean"]]
    print("\n  detector AUC within the retained population:")
    print(auc.groupby("proxy").score_mean.mean().round(4).to_string())
    df.to_csv(OUT / "j7_full.csv", index=False)


if __name__ == "__main__":
    main()
