#!/usr/bin/env python3
"""Second-round revision experiments.

Closes the remaining alternative explanations raised in the second review round:
whether the real-data reversal is an artefact of construction coupling, of the
matching implementation, of multiplicity, or of monotone-dependence-only
conditioning.

  e1  combined_noN     Combined rebuilt without the neighbourhood term, so the
                       documented coupling E_neighbor = 1 - Q_knn cannot be the
                       source of the reversal
  e2  order            randomised processing order of the corrupted examples
  e3  replacement      clean pool reusable instead of consumed
  e4  a40              full audit on C100-A40 (39.8% asymmetric noise, 5 seeds)
  e6  sim              Gaussian simulation, 100 repeats, empirical vs analytic
                       crossing threshold
  e7  strength         reversal rate vs detector strength (no new computation)
  e8  mi               kNN mutual information I(Q;Z) + permutation test

Usage
-----
    python run_round2.py --job e1 --B 1000 --workers 24
    python run_round2.py --job e2 --reps 100 --workers 24
    # e5 (multiplicity) moved to run_final_analysis_set.py; the version that
    # lived here had two one-sided direction bugs, see ROUND2_RESULTS.md 5.3.
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "revision"))

from run_p0_batch import (CACHE, DETECTORS, GLOBAL_AUC_MIN, MAX_CLEAN,  # noqa: E402
                          NOISY_BUDGET, CLEAN_BUDGET, PROXY_FAMILY, PROXIES,
                          common_support, matched_pairs, pairwise_auc, rank_auc,
                          _pct)

OUT = ROOT / "results" / "revision" / "round2"
S20 = ("cifar100", "symmetric0.2")
A40 = ("cifar100", "asymmetric0.4")
# Weights of the paper's Combined, renormalised over the two non-neighbourhood
# terms: 0.5 : 0.3 -> 0.625 : 0.375.
W_LOSS_NO_N, W_FORGET_NO_N = 0.625, 0.375
UNCOUPLED = ["ema_loss", "confidence", "aum", "confident_learning"]


# ---------------------------------------------------------------------------
# payload / derived detector
# ---------------------------------------------------------------------------
def load_payload(dataset: str, noise: str, seed: int) -> dict:
    return dict(np.load(CACHE / dataset / noise / f"seed{seed}.npz", allow_pickle=False))


def available_seeds(dataset: str, noise: str) -> list[int]:
    return sorted(int(p.stem[4:]) for p in (CACHE / dataset / noise).glob("seed*.npz"))


def robust_normalize(x: np.ndarray) -> np.ndarray:
    """Same definition as the pipeline: 1%/99% quantile min-max clipped to [0, 1]."""
    x = np.asarray(x, dtype=np.float64)
    lo, hi = np.quantile(x, [0.01, 0.99])
    span = (hi - lo) or 1.0
    return np.clip((x - lo) / span, 0.0, 1.0)


def get_detector(p: dict, name: str) -> np.ndarray:
    """Return a detector score vector, deriving combined_noN when asked."""
    if name == "combined_noN":
        loss = np.asarray(p["sig__ema_loss"], dtype=np.float64)
        forget = np.asarray(p["sig__forgetting"], dtype=np.float64)
        return robust_normalize(W_LOSS_NO_N * loss + W_FORGET_NO_N * forget)
    return np.asarray(p[f"sig__{name}"], dtype=np.float64)


def detector_names(include_noN: bool = True) -> list[str]:
    base = list(DETECTORS)          # 7 paper detectors, incl. combined
    return base + (["combined_noN"] if include_noN else [])


# ---------------------------------------------------------------------------
# estimators used by the ablations
# ---------------------------------------------------------------------------
def matched_pairs_random_order(score, mask, q, caliper, max_clean, rng,
                               candidate_cap: int = 512):
    """The nn_wo rule with the corrupted examples visited in a random order.

    Everything else is unchanged: greedy nearest-first within the caliper, up to
    ``max_clean`` partners, and a clean sample is consumed by whoever claims it
    first.  Only the visiting order is randomised, which isolates how much of the
    result depends on pipeline order.
    """
    n_idx = np.where(mask)[0]
    c_idx = np.where(~mask)[0]
    empty = np.array([], dtype=np.int64)
    if len(n_idx) == 0 or len(c_idx) == 0:
        return empty, empty
    order = np.argsort(q[c_idx])
    c_sorted = c_idx[order]
    q_sorted = q[c_sorted]
    used = np.zeros(len(c_sorted), dtype=bool)
    visit = n_idx[rng.permutation(len(n_idx))]
    ni_out, ci_out = [], []
    for i in visit:
        nq = q[i]
        lo = int(np.searchsorted(q_sorted, nq - caliper, side="left"))
        hi = int(np.searchsorted(q_sorted, nq + caliper, side="right"))
        if hi <= lo:
            continue
        w = np.arange(lo, hi)
        if len(w) > candidate_cap:
            dn = np.abs(q_sorted[w] - nq)
            w = w[np.argpartition(dn, candidate_cap - 1)[:candidate_cap]]
        free = w[~used[w]]
        if len(free) == 0:
            continue
        dn = np.abs(q_sorted[free] - nq)
        k = min(max_clean, len(free))
        pick = free[np.argpartition(dn, k - 1)[:k]]
        pick = pick[np.argsort(np.abs(q_sorted[pick] - nq))]
        used[pick] = True
        for p_ in pick:
            ni_out.append(i); ci_out.append(c_sorted[p_])
    if not ni_out:
        return empty, empty
    return np.asarray(ni_out, dtype=np.int64), np.asarray(ci_out, dtype=np.int64)


def matched_pairs_with_replacement(score, mask, q, caliper, max_clean,
                                   candidate_cap: int = 512):
    """Clean candidates may be reused by any number of corrupted examples.

    This isolates pool consumption: if the coverage collapse under the primary
    rule drives the extra attenuation, allowing reuse should recover most of it.
    """
    n_idx = np.where(mask)[0]
    c_idx = np.where(~mask)[0]
    empty = np.array([], dtype=np.int64)
    if len(n_idx) == 0 or len(c_idx) == 0:
        return empty, empty
    order = np.argsort(q[c_idx])
    c_sorted = c_idx[order]
    q_sorted = q[c_sorted]
    ni_out, ci_out = [], []
    for i in n_idx:
        nq = q[i]
        lo = int(np.searchsorted(q_sorted, nq - caliper, side="left"))
        hi = int(np.searchsorted(q_sorted, nq + caliper, side="right"))
        if hi <= lo:
            continue
        w = np.arange(lo, hi)
        if len(w) > candidate_cap:
            dn = np.abs(q_sorted[w] - nq)
            w = w[np.argpartition(dn, candidate_cap - 1)[:candidate_cap]]
        dn = np.abs(q_sorted[w] - nq)
        k = min(max_clean, len(w))
        pick = w[np.argpartition(dn, k - 1)[:k]]
        pick = pick[np.argsort(np.abs(q_sorted[pick] - nq))]
        for p_ in pick:
            ni_out.append(i); ci_out.append(c_sorted[p_])
    return np.asarray(ni_out, dtype=np.int64), np.asarray(ci_out, dtype=np.int64)


# ---------------------------------------------------------------------------
# e1: Combined without the neighbourhood term
# ---------------------------------------------------------------------------
def _worker_e1(task: dict) -> list[dict]:
    p = load_payload(*task["setting"], task["seed"])
    mask = np.asarray(p["mask"], dtype=bool)
    out = []
    for det in task["detectors"]:
        score = get_detector(p, det)
        for px in task["proxies"]:
            q = np.asarray(p[f"sig__{px}"], dtype=np.float64)
            row = evaluate_with_bootstrap(score, mask, q, task["B"], task["seed"], det, px,
                                          use_noN=det == "combined_noN")
            row["setting"] = "/".join(task["setting"])
            out.append(row)
    return out


def evaluate_with_bootstrap(score, mask, q, B, seed, det, px, use_noN=False):
    """Point estimates plus paired bootstrap CI (primary rule, bounded draw)."""
    cal = 0.10 * float(q.std())
    ni, ci = matched_pairs(score, mask, q, cal, MAX_CLEAN)
    a_g = rank_auc(mask, score)
    a_audit = pairwise_auc(score, ni, ci)
    # Coverage inside the common support of Q.  An earlier version of this row
    # compared common_support(...)[0] with itself, so ``noisy_coverage`` was 1.0
    # by construction; the retained-fraction definition below is the one the
    # frozen p0_batch audit uses, and run_backfill_coverage.py recomputes it for
    # the E1/E4 CSVs that were written before this fix.  No AUC in this row and
    # no bootstrap replicate ever read this column.
    sup_lo, sup_hi = common_support(q[mask], q[~mask])
    keep = (q >= sup_lo) & (q <= sup_hi) if np.isfinite(sup_lo) else np.zeros(len(q), bool)
    row = {
        "seed": seed, "detector": det, "proxy": px,
        "proxy_family": PROXY_FAMILY.get(px, "?"),
        "global_auc": a_g, "conditioned_auc": a_audit,
        "delta_total": a_g - a_audit,
        "matched_clean_coverage": len(np.unique(ci)) / max(int((~mask).sum()), 1) if len(ci) else 0.0,
        "matched_noisy_coverage": len(np.unique(ni)) / max(int(mask.sum()), 1) if len(ni) else 0.0,
        "noisy_coverage": float((mask & keep).sum() / max(int(mask.sum()), 1)),
        "clean_coverage": float(((~mask) & keep).sum() / max(int((~mask).sum()), 1)),
        "n_pairs": int(len(ni)),
        "rq": float(q[mask].mean() - q[~mask].mean()),
        "eligible": bool(np.isfinite(a_g) and a_g >= GLOBAL_AUC_MIN),
        "combined_variant": "no_neighbor" if use_noN else "paper",
    }
    if B > 0:
        i_n, i_c = np.where(mask)[0], np.where(~mask)[0]
        tag = seed * 977 + abs(hash(det + px)) % 9973
        if len(i_n) > NOISY_BUDGET:
            i_n = np.random.default_rng(tag + 991).choice(i_n, NOISY_BUDGET, replace=False)
        if len(i_c) > CLEAN_BUDGET:
            i_c = np.random.default_rng(tag + 992).choice(i_c, CLEAN_BUDGET, replace=False)
        rng = np.random.default_rng(tag)
        g = np.empty(B); au = np.empty(B)
        for b in range(B):
            bn = i_n[rng.integers(0, len(i_n), size=len(i_n))]
            bc = i_c[rng.integers(0, len(i_c), size=len(i_c))]
            ri = np.concatenate([bn, bc]); rm = mask[ri]; s = score[ri]; qq = q[ri]
            g[b] = rank_auc(rm, s)
            nb, cb = matched_pairs(s, rm, qq, 0.10 * float(qq.std()), MAX_CLEAN)
            au[b] = pairwise_auc(s, nb, cb) if len(nb) else np.nan
        lo, hi = _pct(au, 0.025), _pct(au, 0.975)
        row.update({
            "bootstrap_B": B, "ci_low": lo, "ci_high": hi,
            "bootstrap_valid": int(np.isfinite(au).sum()),
            "bootstrap_failed": int(B - np.isfinite(au).sum()),
            "ci_entirely_above_chance": bool(np.isfinite(lo) and lo > 0.5),
            "ci_straddles_chance": bool(np.isfinite(lo) and np.isfinite(hi) and lo <= 0.5 <= hi),
            "point_reversal": bool(np.isfinite(a_audit) and a_audit < 0.5 and row["eligible"]),
            "interval_reversal": bool(np.isfinite(hi) and hi < 0.5 and row["eligible"]),
        })
    return row


# ---------------------------------------------------------------------------
# e2 / e3: order and replacement ablations
# ---------------------------------------------------------------------------
def _worker_e23(task: dict) -> list[dict]:
    p = load_payload(*task["setting"], task["seed"])
    mask = np.asarray(p["mask"], dtype=bool)
    out = []
    for det in task["detectors"]:
        score = get_detector(p, det)
        for px in task["proxies"]:
            q = np.asarray(p[f"sig__{px}"], dtype=np.float64)
            cal = 0.10 * float(q.std())
            a_g = rank_auc(mask, score)
            base = {"setting": "/".join(task["setting"]), "seed": task["seed"],
                    "detector": det, "proxy": px, "proxy_family": PROXY_FAMILY.get(px, "?"),
                    "global_auc": a_g}
            # --- baseline (pipeline order) ---
            ni, ci = matched_pairs(score, mask, q, cal, MAX_CLEAN)
            a_base = pairwise_auc(score, ni, ci)
            c1_base = len(np.unique(ni)) / max(int(mask.sum()), 1)
            # --- e2: randomised visiting order ---
            rng = np.random.default_rng(task["seed"] * 7919 + abs(hash(px)) % 9973)
            a_ord = np.empty(task["reps"]); c_ord = np.empty(task["reps"])
            for r in range(task["reps"]):
                nr, cr = matched_pairs_random_order(score, mask, q, cal, MAX_CLEAN, rng)
                a_ord[r] = pairwise_auc(score, nr, cr)
                c_ord[r] = len(np.unique(nr)) / max(int(mask.sum()), 1)
            # --- e3: with replacement ---
            nw, cw = matched_pairs_with_replacement(score, mask, q, cal, MAX_CLEAN)
            a_wr = pairwise_auc(score, nw, cw)
            c1_wr = len(np.unique(nw)) / max(int(mask.sum()), 1)
            out.append({**base,
                        "auc_pipeline_order": a_base,
                        "coverage_pipeline_order": c1_base,
                        "auc_random_order_mean": float(np.nanmean(a_ord)),
                        "auc_random_order_sd": float(np.nanstd(a_ord, ddof=1)),
                        "auc_random_order_min": float(np.nanmin(a_ord)),
                        "auc_random_order_max": float(np.nanmax(a_ord)),
                        "coverage_random_order_mean": float(np.nanmean(c_ord)),
                        "coverage_random_order_sd": float(np.nanstd(c_ord, ddof=1)),
                        "p_auc_below_half_random_order": float(np.mean(a_ord < 0.5)),
                        "auc_with_replacement": a_wr,
                        "coverage_with_replacement": c1_wr,
                        "n_pairs_pipeline": int(len(ni)),
                        "n_pairs_with_replacement": int(len(nw)),
                        "reps": task["reps"]})
    return out


# ---------------------------------------------------------------------------
# e4: C100-A40 full audit
# ---------------------------------------------------------------------------
def _worker_e4(task: dict) -> list[dict]:
    return _worker_e1(task)


# ---------------------------------------------------------------------------
# main drivers
# ---------------------------------------------------------------------------
def run_pooled(tasks, worker, workers, out_csv, label):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time(); rows = []
    if workers > 1:
        with mp.Pool(processes=min(workers, len(tasks))) as pool:
            for i, res in enumerate(pool.imap_unordered(worker, tasks)):
                rows.extend(res)
                pd.DataFrame(rows).to_csv(out_csv.with_suffix(".partial.csv"), index=False)
                print(f"   [{i+1}/{len(tasks)}] {label} ({time.time()-t0:.0f}s, {len(rows)} rows)",
                      flush=True)
    else:
        for i, t in enumerate(tasks):
            rows.extend(worker(t))
            pd.DataFrame(rows).to_csv(out_csv.with_suffix(".partial.csv"), index=False)
            print(f"   [{i+1}/{len(tasks)}] {label} ({time.time()-t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    df.to_csv(out_csv.with_suffix(".partial.csv"), index=False)
    print(f"\nwrote {out_csv} ({len(df)} rows, {time.time()-t0:.0f}s)")
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True,
                    choices=["e1", "e2", "e3", "e4", "e6", "e7", "e8", "all"])
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    ap.add_argument("--B", type=int, default=1000)
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--seeds", type=str, default="all")
    ap.add_argument("--lam_max", type=float, default=3.0,
                    help="e6: upper end of the lambda grid")
    ap.add_argument("--lam_n", type=int, default=31,
                    help="e6: number of lambda grid points")
    ap.add_argument("--gammas", type=str, default="0.25,0.5,1.0,1.5",
                    help="e6: comma-separated gamma values")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    jobs = ["e1", "e2", "e3", "e4", "e6", "e7", "e8"] if args.job == "all" else [args.job]

    for job in jobs:
        if job == "e1":
            seeds = available_seeds(*S20)
            if args.seeds != "all":
                want = {int(s) for s in args.seeds.split(",")}
                seeds = [s for s in seeds if s in want]
            tasks = [{"setting": S20, "seed": s, "detectors": detector_names(),
                      "proxies": PROXIES, "B": args.B} for s in seeds]
            print(f"== e1: Combined-noNeighbor, {len(tasks)} seeds x "
                  f"{len(detector_names())} detectors x {len(PROXIES)} proxies ==")
            df = run_pooled(tasks, _worker_e1, args.workers,
                            OUT / "e1_combined_noN.csv", "e1")
            summarise_e1(df)

        elif job in ("e2", "e3"):
            seeds = available_seeds(*S20)
            if args.seeds != "all":
                want = {int(s) for s in args.seeds.split(",")}
                seeds = [s for s in seeds if s in want]
            dets = UNCOUPLED + ["combined", "combined_noN"]
            pxs = ["random", "dino_density", "native_density", "proto_margin", "knn_agreement"]
            tasks = [{"setting": S20, "seed": s, "detectors": dets, "proxies": pxs,
                      "reps": args.reps} for s in seeds]
            print(f"== e2/e3: order ({args.reps} reps) + with-replacement ==")
            df = run_pooled(tasks, _worker_e23, args.workers,
                            OUT / "e2e3_order_replacement.csv", "e2/e3")
            summarise_e23(df)

        elif job == "e4":
            seeds = available_seeds(*A40)
            if args.seeds != "all":
                want = {int(s) for s in args.seeds.split(",")}
                seeds = [s for s in seeds if s in want]
            all_dets = detector_names()
            tasks = [{"setting": A40, "seed": s, "detectors": all_dets,
                      "proxies": PROXIES, "B": args.B} for s in seeds]
            print(f"== e4: C100-A40 audit, {len(tasks)} seeds x {len(all_dets)} x {len(PROXIES)} ==")
            df = run_pooled(tasks, _worker_e4, args.workers,
                            OUT / "e4_a40_audit.csv", "e4")
            summarise_e1(df, setting="C100-A40")

        elif job == "e6":
            run_simulation(args)
        elif job == "e7":
            analyse_strength()
        elif job == "e8":
            run_mi(seeds_arg=args.seeds)


# ---------------------------------------------------------------------------
# summaries and analyses
# ---------------------------------------------------------------------------
def summarise_e1(df: pd.DataFrame, setting: str = "C100-S20") -> None:
    print(f"\n== {setting}: combined vs combined-noNeighbor ==")
    g = (df[df.eligible].groupby(["detector", "proxy"])
         .agg(a_g=("global_auc", "mean"), a_cond=("conditioned_auc", "mean"),
              delta=("delta_total", "mean"), ci_low=("ci_low", "mean"),
              ci_high=("ci_high", "mean"),
              point_rev=("point_reversal", "sum"), int_rev=("interval_reversal", "sum"),
              m_noisy_cov=("matched_noisy_coverage", "mean"), n=("seed", "size"))
         .reset_index())
    key = g[g.detector.isin(["combined", "combined_noN"])]
    print(key.round(4).to_string(index=False))
    g.to_csv(OUT / f"{'e4' if 'A40' in setting else 'e1'}_summary.csv", index=False)
    print("\n  interval reversals by detector (all proxies pooled):")
    print(df[df.eligible].groupby("detector").interval_reversal.sum().to_string())


def summarise_e23(df: pd.DataFrame) -> None:
    print("\n== e2/e3: order and replacement ablations ==")
    g = (df.groupby(["detector", "proxy"])
         .agg(a_pipe=("auc_pipeline_order", "mean"),
              a_rand=("auc_random_order_mean", "mean"),
              rand_sd=("auc_random_order_sd", "mean"),
              c_rand=("coverage_random_order_mean", "mean"),
              p_below_half=("p_auc_below_half_random_order", "mean"),
              a_wr=("auc_with_replacement", "mean"),
              c_pipe=("coverage_pipeline_order", "mean"),
              c_wr=("coverage_with_replacement", "mean"))
         .reset_index())
    print(g.round(4).to_string(index=False))
    g.to_csv(OUT / "e2e3_summary.csv", index=False)
    print("\n  pooled by proxy:")
    p = (df.groupby("proxy")
         .agg(a_pipe=("auc_pipeline_order", "mean"),
              a_rand=("auc_random_order_mean", "mean"),
              a_wr=("auc_with_replacement", "mean"),
              c_pipe=("coverage_pipeline_order", "mean"),
              c_wr=("coverage_with_replacement", "mean")).reset_index())
    print(p.round(4).to_string(index=False))


def analyse_strength() -> None:
    """Reversal rate as a function of detector strength (no new computation)."""
    src = ROOT / "results" / "revision" / "p0_batch" / "j1_bootstrap_all_proxies.csv"
    df = pd.read_csv(src)
    df = df[df.global_auc >= GLOBAL_AUC_MIN].copy()
    bins = [0.50, 0.60, 0.70, 0.80, 1.01]
    labels = ["[0.50,0.60)", "[0.60,0.70)", "[0.70,0.80)", "[0.80,1.00]"]
    df["strength_bin"] = pd.cut(df.global_auc, bins=bins, labels=labels, right=False)
    fam = df.proxy_family
    df["family"] = np.where(fam == "label_dependent", "label_dependent", fam)
    g = (df.groupby(["strength_bin", "family"], observed=True)
         .agg(n_cells=("seed", "size"), a_g=("global_auc", "mean"),
              a_cond=("conditioned_auc", "mean"), delta=("delta_total", "mean"),
              point_rev=("point_reversal", "sum"),
              int_rev=("interval_reversal", "sum"))
         .reset_index())
    g["p_point_rev"] = g.point_rev / g.n_cells
    g["p_int_rev"] = g.int_rev / g.n_cells
    g.to_csv(OUT / "e7_detector_strength.csv", index=False)
    print("\n== e7: reversal rate vs detector strength bin ==")
    print(g.round(4).to_string(index=False))


def run_simulation(args) -> None:
    """Gaussian model, R repeats, empirical vs analytic crossing threshold."""
    from scipy.stats import norm
    rng_global = np.random.default_rng(0)
    RHO, SIG_E, SIG_Q, N = 0.2, 1.0, 0.3, 20000
    lams = np.round(np.linspace(0.0, args.lam_max, args.lam_n), 4)
    rows = []
    for gamma in [float(x) for x in str(args.gammas).split(",")]:
        beta = float(norm.ppf(0.70) * np.sqrt(2 * (gamma ** 2 + SIG_E ** 2)))
        for rep in range(args.reps):
            rng = np.random.default_rng(10_000 * rep + int(gamma * 100))
            z = rng.random(N) < RHO
            d = rng.normal(size=N)
            e = beta * z + gamma * d + SIG_E * rng.normal(size=N)
            eps_q = SIG_Q * rng.normal(size=N)
            for lam in lams:
                q = d + lam * z + eps_q
                ni, ci = matched_pairs(e, z, q, 0.10 * float(q.std()), MAX_CLEAN)
                a_audit = pairwise_auc(e, ni, ci)
                rows.append({"gamma": gamma, "beta": beta, "lambda": lam, "rep": rep,
                             "global_auc": rank_auc(z, e), "audit_auc": a_audit,
                             "reversal": bool(np.isfinite(a_audit) and a_audit < 0.5)})
        print(f"   gamma={gamma} done", flush=True)
    df = pd.DataFrame(rows)
    suffix = "" if (args.lam_max == 3.0 and args.lam_n == 31
                    and args.gammas == "0.25,0.5,1.0,1.5") else (
        f"_lam{args.lam_max:g}_n{args.lam_n}_g{args.gammas.replace(',', '-')}")
    df.to_csv(OUT / f"e6_simulation{suffix}.csv", index=False)

    # empirical crossing: smallest lambda at which the mean audit AUC < 0.5
    out = []
    for gamma, g in df.groupby("gamma"):
        beta = float(g.beta.iloc[0])
        curve = g.groupby("lambda").audit_auc.agg(["mean", "std", "size"]).reset_index()
        below = curve[curve["mean"] < 0.5]
        lam_cross = float(below["lambda"].iloc[0]) if len(below) else float("nan")
        # per-repetition crossing, then a CI over repetitions
        per_rep = []
        for rep, gr in g.groupby("rep"):
            gr = gr.sort_values("lambda")
            b = gr[gr.audit_auc < 0.5]
            per_rep.append(float(b["lambda"].iloc[0]) if len(b) else np.nan)
        per_rep = np.asarray(per_rep, dtype=float)
        ok = per_rep[np.isfinite(per_rep)]
        out.append({
            "gamma": gamma, "beta": beta,
            "analytic_lambda_star": beta * (1 + SIG_Q ** 2) / gamma,
            "empirical_crossing_mean_curve": lam_cross,
            "empirical_crossing_mean": float(ok.mean()) if ok.size else float("nan"),
            "empirical_crossing_ci_lo": float(np.quantile(ok, 0.025)) if ok.size else float("nan"),
            "empirical_crossing_ci_hi": float(np.quantile(ok, 0.975)) if ok.size else float("nan"),
            "n_reps": int(args.reps), "n_reps_crossing": int(ok.size),
            "n_reps_no_crossing": int((~np.isfinite(per_rep)).sum()),
            "lam_max": float(args.lam_max), "lam_n": int(args.lam_n),
        })
    b = pd.DataFrame(out)
    b.to_csv(OUT / f"e6_crossing{suffix}.csv", index=False)
    print("\n== e6: analytic vs empirical reversal threshold ==")
    print(b.round(4).to_string(index=False))


def run_mi(seeds_arg: str = "all") -> None:
    """kNN mutual information I(Q;Z) with a label-permutation null."""
    from sklearn.feature_selection import mutual_info_classif
    rows = []
    for setting in (S20, A40):
        seeds = available_seeds(*setting)
        if seeds_arg != "all":
            want = {int(s) for s in seeds_arg.split(",")}
            seeds = [s for s in seeds if s in want]
        for seed in seeds:
            p = load_payload(*setting, seed)
            mask = np.asarray(p["mask"], dtype=bool)
            z = mask.astype(int)
            rng = np.random.default_rng(seed)
            for px in PROXIES:
                q = np.asarray(p[f"sig__{px}"], dtype=np.float64).reshape(-1, 1)
                mi = float(mutual_info_classif(q, z, discrete_features=False,
                                               random_state=seed)[0])
                n_perm = 1000
                null = np.empty(n_perm)
                for i in range(n_perm):
                    null[i] = mutual_info_classif(q, rng.permutation(z),
                                                  discrete_features=False,
                                                  random_state=seed)[0]
                p_perm = float((null >= mi).mean())
                # rank association for reference
                a_qz = rank_auc(mask, q.ravel())
                rows.append({"setting": "/".join(setting), "seed": seed, "proxy": px,
                             "proxy_family": PROXY_FAMILY.get(px, "?"),
                             "mutual_information": mi,
                             "mi_null_mean": float(null.mean()),
                             "mi_perm_p": p_perm,
                             "rq_abs_auc_minus_half": abs(a_qz - 0.5),
                             "auc_q_vs_z": a_qz})
            print(f"   {setting[0]}/{setting[1]} seed{seed} done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "e8_mutual_information.csv", index=False)
    print("\n== e8: I(Q;Z) and permutation test (mean over seeds) ==")
    g = (df.groupby(["setting", "proxy"])
         .agg(mi=("mutual_information", "mean"), mi_null=("mi_null_mean", "mean"),
              p=("mi_perm_p", "mean"), rq=("rq_abs_auc_minus_half", "mean"))
         .reset_index())
    print(g.round(5).to_string(index=False))


if __name__ == "__main__":
    main()
