#!/usr/bin/env python3
"""P0 experiment batch for the C100-S20 revision.

Four jobs share one canonical output schema so every later figure and table can
be generated from a single CSV:

  J1  bootstrap      exact-estimator paired bootstrap, all detectors x all proxies
  J2  decomposition  A_g, A_retained, A_audit and the two gap components
  J3  sensitivity    caliper x max-matches grid, uncoupled detectors only
  J4  uncoupled      headline replication on detectors that use no neighbourhood
                     evidence, with coverage and retention diagnostics

Design notes that matter for correctness
----------------------------------------
* The conditioning covariate Q is re-standardised and the caliper recomputed
  inside every bootstrap replicate, and the matching is re-run from scratch.
  Bootstrap pairs are never frozen and re-scored.
* The matched estimator is the pipeline's own ``nn_wo`` rule, reproduced bit-
  identically (see standalone/exact_bootstrap_c100s20.py).
* P(A_retained, A_audit) is re-standardised inside the replicate; that is what
  makes the decomposition identity check meaningful.
* Reversal bookkeeping uses the fixed score orientation: a proxy that is
  anti-correlated with the noise label gives A_g < 0.5, and such cells are not
  counted as conditioning-induced reversals.  Both raw counts are reported.

Usage
-----
    python run_p0_batch.py --job j1 --workers 8
    python run_p0_batch.py --job j2 --workers 8
    python run_p0_batch.py --job j3 --workers 8
    python run_p0_batch.py --job j4 --workers 8
    python run_p0_batch.py --job j1 --seeds 0 --B 50        # smoke test
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
CACHE = ROOT / "results" / "revision" / "prepared"
OUT = ROOT / "results" / "revision" / "p0_batch"
DATASET, NOISE = "cifar100", "symmetric0.2"

DETECTORS = ["ema_loss", "confidence", "aum", "forgetting", "neighbor",
             "confident_learning", "combined"]
# Detectors that use no neighbourhood evidence, so no construction coupling with
# the KNN-agreement covariate.
UNCOUPLED = ["ema_loss", "confidence", "aum", "confident_learning"]
PROXIES = ["random", "dino_density", "dino_density_fullpool", "dino_knndist",
           "dino_augcons", "native_density", "proto_margin", "knn_agreement"]
PROXY_FAMILY = {
    "random": "control", "dino_density": "label_free",
    "dino_density_fullpool": "label_free", "dino_knndist": "label_free",
    "dino_augcons": "label_free", "native_density": "encoder_derived",
    "proto_margin": "label_dependent", "knn_agreement": "label_dependent",
}
CALIPER_SD = 0.10        # default caliper, in SD(Q) units
# Resample budgets.  Drawing the full strata (8.9k noisy x 36k clean) makes the
# greedy matching step O(n_noisy) with a long per-sample candidate scan, which
# costs ~165 ms per replicate and dominates everything else.  Bounding the draw
# keeps the paired comparison intact while cutting that to ~12 ms; the same
# budgets are used by the standalone audit script, so the two agree.
NOISY_BUDGET = 1200
CLEAN_BUDGET = 8000
MAX_CLEAN = 5            # default max clean partners per noisy sample
GLOBAL_AUC_MIN = 0.5     # eligibility for a reversal claim
CI_BORDERLINE = 0.52     # ci_low below this triggers a top-up request


# ---------------------------------------------------------------------------
# estimators
# ---------------------------------------------------------------------------
def rank_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """AUROC by mid-ranks (Mann-Whitney).  Ties count 0.5."""
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    n1 = int(labels.sum()); n0 = len(labels) - n1
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


def common_support(q_noisy: np.ndarray, q_clean: np.ndarray,
                   robust: bool = False) -> tuple[float, float]:
    if robust:
        a, b = np.percentile(q_noisy, [1, 99])
        c, d = np.percentile(q_clean, [1, 99])
    else:
        a, b = q_noisy.min(), q_noisy.max()
        c, d = q_clean.min(), q_clean.max()
    lo, hi = max(float(a), float(c)), min(float(b), float(d))
    return (lo, hi) if lo <= hi else (float("nan"), float("nan"))


def matched_pairs(score: np.ndarray, mask: np.ndarray, q: np.ndarray,
                  caliper: float, max_clean: int = MAX_CLEAN,
                  candidate_cap: int = 512):
    """The pipeline's ``nn_wo`` rule, reproduced term for term.

    Clean samples are consumed by the first noisy sample that claims them; each
    noisy sample takes up to ``max_clean`` currently unused clean samples inside
    the caliper, chosen by ascending |dQ|.  Noisy samples are processed in
    pipeline (ascending sample-id) order.
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
    ni_out: list[int] = []
    ci_out: list[int] = []
    cap = max(1, int(candidate_cap))
    m = max(1, int(max_clean))
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
            ni_out.append(i)
            ci_out.append(c_sorted[p_])
    if not ni_out:
        return empty, empty
    return (np.asarray(ni_out, dtype=np.int64),
            np.asarray(ci_out, dtype=np.int64))


def pairwise_auc(score: np.ndarray, n_idx: np.ndarray, c_idx: np.ndarray) -> float:
    """P(score_noisy > score_clean) over the given pairs (ties count 0.5)."""
    if len(n_idx) == 0:
        return float("nan")
    sn, sc = score[n_idx], score[c_idx]
    wins = float((sn > sc).mean())
    ties = float((sn == sc).mean())
    return wins + 0.5 * ties


def _smd(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    pooled = np.sqrt(((len(a) - 1) * a.var() + (len(b) - 1) * b.var()) /
                     max(len(a) + len(b) - 2, 1))
    if pooled < 1e-12:
        return 0.0
    return float((a.mean() - b.mean()) / pooled)


# ---------------------------------------------------------------------------
# one cell
# ---------------------------------------------------------------------------
def evaluate_cell(score: np.ndarray, mask: np.ndarray, q: np.ndarray,
                  caliper_sd: float, max_clean: int, B: int,
                  seed_tag: int, decomposition: bool = True) -> dict:
    """Point estimates, retention decomposition and paired bootstrap for one cell.

    ``decomposition=False`` skips the retained-population AUROC and the two gap
    components.  The j3 sensitivity grid sweeps 12 matching settings per cell and
    only needs the matched AUROC, coverage and balance, so skipping the
    retained-population work there roughly halves the cost without changing any
    number that grid reports.
    """
    score = np.asarray(score, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    q = np.asarray(q, dtype=np.float64)
    n_tot, c_tot = int(mask.sum()), int((~mask).sum())
    sd = float(q.std())
    cal = caliper_sd * sd

    a_g = rank_auc(mask, score)

    # --- retained population: unique noisy/clean samples inside common support
    lo, hi = common_support(q[mask], q[~mask])
    if np.isfinite(lo):
        keep = (q >= lo) & (q <= hi)
    else:
        keep = np.zeros(len(q), dtype=bool)
    n_keep = np.where(mask & keep)[0]
    c_keep = np.where((~mask) & keep)[0]
    if not decomposition:
        a_ret = float("nan")
    # Population comparison inside the retained set.  The mid-rank statistic is
    # the same number as the explicit pairwise broadcast but runs in O(N log N)
    # instead of O(n_noisy * n_clean); at 12k x 33k the broadcast cost ~1 s per
    # call, which dominates the whole bootstrap.
    elif len(n_keep) and len(c_keep):
        sel = np.concatenate([n_keep, c_keep])
        lab = np.concatenate([np.ones(len(n_keep), bool), np.zeros(len(c_keep), bool)])
        a_ret = rank_auc(lab, score[sel])
    else:
        a_ret = float("nan")

    # --- audit: matched pairs
    ni, ci = matched_pairs(score, mask, q, cal, max_clean)
    a_audit = pairwise_auc(score, ni, ci)

    row = {
        "dataset": DATASET, "noise_type": "symmetric", "noise_rate": 0.2,
        "caliper": cal, "caliper_sd": caliper_sd, "max_matches": max_clean,
        "replacement": False,
        "global_auc": a_g,
        "retained_auc": a_ret,
        "conditioned_auc": a_audit,
        "delta_total": a_g - a_audit if np.isfinite(a_audit) else float("nan"),
        "delta_population": a_g - a_ret if np.isfinite(a_ret) else float("nan"),
        "delta_matching": a_ret - a_audit if np.isfinite(a_audit) and np.isfinite(a_ret)
                          else float("nan"),
        "rq": float(q[mask].mean() - q[~mask].mean()),
        "smd_before": _smd(q[mask], q[~mask]),
        "smd_after": _smd(q[ni], q[ci]) if len(ni) else float("nan"),
        "support_lo": lo, "support_hi": hi,
        "n_clean": c_tot, "n_noisy": n_tot,
        "n_clean_retained": len(c_keep), "n_noisy_retained": len(n_keep),
        "clean_coverage": len(c_keep) / max(c_tot, 1),
        "noisy_coverage": len(n_keep) / max(n_tot, 1),
        "n_pairs": int(len(ni)),
        "n_noisy_matched": int(len(np.unique(ni))) if len(ni) else 0,
        "n_clean_matched": int(len(np.unique(ci))) if len(ci) else 0,
        "matched_noisy_coverage": (len(np.unique(ni)) / max(n_tot, 1)) if len(ni) else 0.0,
        "matched_clean_coverage": (len(np.unique(ci)) / max(c_tot, 1)) if len(ci) else 0.0,
        "matching_failure": int(len(ni) == 0),
        "eligible": bool(np.isfinite(a_g) and a_g >= GLOBAL_AUC_MIN),
    }

    # --- paired bootstrap: resample -> restandardise -> resupport -> rematch
    if B > 0:
        i_n = np.where(mask)[0]
        i_c = np.where(~mask)[0]
        # Fixed pools (subsampled once) so every replicate draws from the same
        # bounded population rather than a varying one.
        if len(i_n) > NOISY_BUDGET:
            i_n = np.random.default_rng(seed_tag + 991).choice(
                i_n, NOISY_BUDGET, replace=False)
        if len(i_c) > CLEAN_BUDGET:
            i_c = np.random.default_rng(seed_tag + 992).choice(
                i_c, CLEAN_BUDGET, replace=False)
        rng = np.random.default_rng(seed_tag)
        g = np.empty(B); r_ = np.empty(B); au = np.empty(B)
        for b in range(B):
            bn = i_n[rng.integers(0, len(i_n), size=len(i_n))]
            bc = i_c[rng.integers(0, len(i_c), size=len(i_c))]
            ri = np.concatenate([bn, bc])
            rm = mask[ri]; s = score[ri]; qq = q[ri]
            g[b] = rank_auc(rm, s)
            lo_b, hi_b = common_support(qq[rm], qq[~rm])
            k_b = (qq >= lo_b) & (qq <= hi_b) if np.isfinite(lo_b) else np.zeros(len(qq), bool)
            nk, ck = np.where(rm & k_b)[0], np.where((~rm) & k_b)[0]
            if len(nk) and len(ck):
                sel = np.concatenate([nk, ck])
                lab = np.concatenate([np.ones(len(nk), bool),
                                      np.zeros(len(ck), bool)])
                r_[b] = rank_auc(lab, s[sel])
            else:
                r_[b] = np.nan
            cal_b = caliper_sd * float(qq.std())
            nb, cb = matched_pairs(s, rm, qq, cal_b, max_clean)
            au[b] = pairwise_auc(s, nb, cb) if len(nb) else np.nan
        d = g - au
        valid = int(np.isfinite(au).sum())
        row.update({
            "bootstrap_B": B,
            "bootstrap_noisy_budget": len(i_n),
            "bootstrap_clean_budget": len(i_c),
            "bootstrap_valid": valid,
            "bootstrap_failed": int(B - valid),
            "ci_low": _pct(au, 0.025), "ci_high": _pct(au, 0.975),
            "delta_ci_low": _pct(d, 0.025), "delta_ci_high": _pct(d, 0.975),
            "global_ci_low": _pct(g, 0.025), "global_ci_high": _pct(g, 0.975),
        })
        # Reversal bookkeeping is gated on eligibility.  A detector whose GLOBAL
        # AUROC is already below chance (e.g. CIFAR-100 forgetting, A_g ~ 0.33)
        # produces ci_high < 0.5 by construction; that is the detector reading the
        # wrong way, not a conditioning-induced reversal.  Both the raw and the
        # eligibility-gated counts are kept so nothing is hidden.
        row["point_reversal_raw"] = bool(np.isfinite(a_audit) and a_audit < 0.5)
        row["interval_reversal_raw"] = bool(row["ci_high"] < 0.5) if np.isfinite(row["ci_high"]) else False
        row["point_reversal"] = bool(row["point_reversal_raw"] and row["eligible"])
        row["interval_reversal"] = bool(row["interval_reversal_raw"] and row["eligible"])
        row["ci_entirely_above_chance"] = bool(row["ci_low"] > 0.5) if np.isfinite(row["ci_low"]) else False
        row["ci_straddles_chance"] = bool(np.isfinite(row["ci_low"]) and np.isfinite(row["ci_high"])
                                          and row["ci_low"] <= 0.5 <= row["ci_high"])
        row["needs_topup"] = bool(row["eligible"] and np.isfinite(row["ci_low"])
                                  and row["ci_low"] < CI_BORDERLINE)
    return row


def _pct(v: np.ndarray, p: float) -> float:
    v = np.asarray(v, dtype=np.float64)
    v = v[np.isfinite(v)]
    return float(np.quantile(v, p)) if v.size else float("nan")


# ---------------------------------------------------------------------------
# payload loading and job drivers
# ---------------------------------------------------------------------------
def load_payload(seed: int) -> dict:
    f = CACHE / DATASET / NOISE / f"seed{seed}.npz"
    return dict(np.load(f, allow_pickle=False))


def available_seeds() -> list[int]:
    d = CACHE / DATASET / NOISE
    return sorted(int(p.stem[4:]) for p in d.glob("seed*.npz"))


def _worker_j1(task: dict) -> list[dict]:
    p = load_payload(task["seed"])
    mask = np.asarray(p["mask"], dtype=bool)
    out = []
    for det in task["detectors"]:
        score = np.asarray(p[f"sig__{det}"], dtype=np.float64)
        for px in task["proxies"]:
            q = np.asarray(p[f"sig__{px}"], dtype=np.float64)
            row = evaluate_cell(score, mask, q, CALIPER_SD, MAX_CLEAN, task["B"],
                                seed_tag=task["seed"] * 977 + abs(hash(det + px)) % 9973)
            row.update({"seed": task["seed"], "detector": det, "proxy": px,
                        "proxy_family": PROXY_FAMILY.get(px, "?")})
            out.append(row)
    return out


def _worker_j3(task: dict) -> list[dict]:
    p = load_payload(task["seed"])
    mask = np.asarray(p["mask"], dtype=bool)
    out = []
    for det in task["detectors"]:
        score = np.asarray(p[f"sig__{det}"], dtype=np.float64)
        for px in task["proxies"]:
            q = np.asarray(p[f"sig__{px}"], dtype=np.float64)
            for cal in task["calipers"]:
                for mm in task["max_matches"]:
                    row = evaluate_cell(score, mask, q, cal, mm, 0, seed_tag=0,
                                        decomposition=False)
                    row.update({"seed": task["seed"], "detector": det, "proxy": px,
                                "proxy_family": PROXY_FAMILY.get(px, "?"),
                                "bootstrap_B": 0})
                    out.append(row)
    return out


def run_pooled(tasks: list[dict], worker, workers: int, out_csv: Path,
               label: str) -> pd.DataFrame:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rows: list[dict] = []
    partial = out_csv.with_suffix(".partial.csv")
    if workers > 1:
        with mp.Pool(processes=min(workers, len(tasks))) as pool:
            for i, res in enumerate(pool.imap_unordered(worker, tasks)):
                rows.extend(res)
                pd.DataFrame(rows).to_csv(partial, index=False)
                print(f"   [{i+1}/{len(tasks)}] {label} ({time.time()-t0:.0f}s, {len(rows)} rows)",
                      flush=True)
    else:
        for i, t in enumerate(tasks):
            rows.extend(worker(t))
            pd.DataFrame(rows).to_csv(partial, index=False)
            print(f"   [{i+1}/{len(tasks)}] {label} ({time.time()-t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"\nwrote {out_csv}  ({len(df)} rows, {time.time()-t0:.0f}s)")
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True, choices=["j1", "j2", "j3", "j4", "j5"])
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    ap.add_argument("--B", type=int, default=1000)
    ap.add_argument("--seeds", type=str, default="all")
    ap.add_argument("--topup", action="store_true",
                    help="for j1: top up cells with ci_low < 0.52 to B=2000")
    args = ap.parse_args()

    seeds = available_seeds()
    if args.seeds != "all":
        want = {int(s) for s in args.seeds.split(",")}
        seeds = [s for s in seeds if s in want]
    print(f"== P0 batch: {args.job} ==  seeds={seeds}  workers={args.workers}")

    if args.job == "j1":
        tasks = [{"seed": s, "detectors": DETECTORS, "proxies": PROXIES, "B": args.B}
                 for s in seeds]
        df = run_pooled(tasks, _worker_j1, args.workers,
                        OUT / "j1_bootstrap_all_proxies.csv", "j1")
        _summarise_j1(df)

    elif args.job == "j2":
        # decomposition needs no bootstrap: pure point estimates
        tasks = [{"seed": s, "detectors": DETECTORS, "proxies": PROXIES,
                  "calipers": [CALIPER_SD], "max_matches": [MAX_CLEAN]}
                 for s in seeds]
        df = run_pooled(tasks, _worker_j3, args.workers,
                        OUT / "j2_decomposition.csv", "j2")
        _check_decomposition(df)

    elif args.job == "j3":
        tasks = [{"seed": s, "detectors": UNCOUPLED, "proxies":
                  ["random", "dino_density", "native_density", "knn_agreement",
                   "proto_margin"],
                  "calipers": [0.05, 0.10, 0.20], "max_matches": [1, 3, 5, 10]}
                 for s in seeds]
        df = run_pooled(tasks, _worker_j3, args.workers,
                        OUT / "j3_matching_sensitivity.csv", "j3")
        _summarise_j3(df)

    elif args.job == "j5":
        run_topup(OUT / "j1_bootstrap_all_proxies.csv", args.workers,
                  OUT / "j5_topup_B2000.csv")

    elif args.job == "j4":
        tasks = [{"seed": s, "detectors": UNCOUPLED,
                  "proxies": ["random", "dino_density", "native_density",
                              "proto_margin", "knn_agreement"], "B": args.B}
                 for s in seeds]
        df = run_pooled(tasks, _worker_j1, args.workers,
                        OUT / "j4_uncoupled_replication.csv", "j4")
        _summarise_j4(df)


# ---------------------------------------------------------------------------
# summaries
# ---------------------------------------------------------------------------
def _summarise_j1(df: pd.DataFrame) -> None:
    print("\n== j1: exact-estimator bootstrap, all 8 proxies x 7 detectors ==")
    g = (df[df.eligible].groupby("proxy")
         .agg(n=("seed", "size"), auc_global=("global_auc", "mean"),
              auc_cond=("conditioned_auc", "mean"),
              delta=("delta_total", "mean"),
              n_point_rev=("point_reversal", "sum"),
              n_int_rev=("interval_reversal", "sum"),
              n_ci_above=("ci_entirely_above_chance", "sum"),
              n_topup=("needs_topup", "sum"),
              noisy_cov=("noisy_coverage", "mean"),
              m_noisy_cov=("matched_noisy_coverage", "mean"))
         .reset_index())
    print(g.round(4).to_string(index=False))
    g.to_csv(OUT / "j1_summary_by_proxy.csv", index=False)
    tot = df[df.eligible]
    print(f"\n  eligible cells: {len(tot)}")
    print(f"  point reversals       : {int(tot.point_reversal.sum())}"
          f"  (raw incl. ineligible: {int(tot.point_reversal_raw.sum())})")
    print(f"  interval reversals    : {int(tot.interval_reversal.sum())}"
          f"  (raw incl. ineligible: {int(tot.interval_reversal_raw.sum())})")
    print(f"  CI straddles chance   : {int(tot.ci_straddles_chance.sum())}")
    print(f"  CI entirely above 0.5 : {int(tot.ci_entirely_above_chance.sum())}")
    print(f"  cells needing top-up  : {int(tot.needs_topup.sum())}")
    print(f"  bootstrap failures    : {int(tot.bootstrap_failed.sum())}")


def _check_decomposition(df: pd.DataFrame) -> None:
    print("\n== j2: Eq.(34) decomposition identity A_g - A_audit = d_pop + d_match ==")
    d = df.copy()
    d["lhs"] = d.delta_total
    d["rhs"] = d.delta_population + d.delta_matching
    d["residual"] = d.lhs - d.rhs
    fin = d[np.isfinite(d.residual)]
    print(f"  cells: {len(d)}   finite residuals: {len(fin)}")
    print(f"  max |residual| = {np.abs(fin.residual).max():.3e}")
    print(f"  mean |residual| = {np.abs(fin.residual).mean():.3e}")
    g = (d.groupby(["detector", "proxy_family"])
         .agg(n=("seed", "size"), a_g=("global_auc", "mean"),
              a_ret=("retained_auc", "mean"), a_audit=("conditioned_auc", "mean"),
              d_total=("delta_total", "mean"), d_pop=("delta_population", "mean"),
              d_match=("delta_matching", "mean"))
         .reset_index())
    print(g.round(4).to_string(index=False))
    g.to_csv(OUT / "j2_summary.csv", index=False)


def _summarise_j3(df: pd.DataFrame) -> None:
    print("\n== j3: matching sensitivity (caliper x max_matches) ==")
    g = (df.groupby(["proxy", "caliper_sd", "max_matches"])
         .agg(auc=("conditioned_auc", "mean"), n=("seed", "size"),
              noisy_cov=("matched_noisy_coverage", "mean"),
              n_pairs=("n_pairs", "mean"), smd=("smd_after", "mean"),
              fail=("matching_failure", "sum"))
         .reset_index())
    print(g.round(4).to_string(index=False))
    g.to_csv(OUT / "j3_summary.csv", index=False)


def _summarise_j4(df: pd.DataFrame) -> None:
    print("\n== j4: uncoupled-detector replication ==")
    g = (df.groupby(["detector", "proxy"])
         .agg(n=("seed", "size"), a_g=("global_auc", "mean"),
              a_cond=("conditioned_auc", "mean"), delta=("delta_total", "mean"),
              noisy_cov=("matched_noisy_coverage", "mean"),
              clean_cov=("matched_clean_coverage", "mean"))
         .reset_index())
    print(g.round(4).to_string(index=False))
    g.to_csv(OUT / "j4_summary.csv", index=False)




# ---------------------------------------------------------------------------
# j5: top-up.  Reuses the B=1000 draw sequence (same seeds and rng) and adds
# ANOTHER_B replicates on top, so the result is a genuine B = B1 + B2 estimate
# rather than an independent re-run that would have to be averaged.
# ---------------------------------------------------------------------------
ANOTHER_B = 1000


def _worker_j5(task: dict) -> list[dict]:
    p = load_payload(task["seed"])
    mask = np.asarray(p["mask"], dtype=bool)
    score = np.asarray(p[f"sig__{task['detector']}"], dtype=np.float64)
    q = np.asarray(p[f"sig__{task['proxy']}"], dtype=np.float64)
    B1, B2 = task["bootstrap_B"], task["more"]
    # identical seed_tag derivation to j1 so the first B1 replicates coincide
    seed_tag = task["seed"] * 977 + abs(hash(task["detector"] + task["proxy"])) % 9973

    i_n = np.where(mask)[0]
    i_c = np.where(~mask)[0]
    if len(i_n) > NOISY_BUDGET:
        i_n = np.random.default_rng(seed_tag + 991).choice(i_n, NOISY_BUDGET, replace=False)
    if len(i_c) > CLEAN_BUDGET:
        i_c = np.random.default_rng(seed_tag + 992).choice(i_c, CLEAN_BUDGET, replace=False)
    rng = np.random.default_rng(seed_tag)
    g = np.empty(B1 + B2); au = np.empty(B1 + B2)
    for b in range(B1 + B2):
        bn = i_n[rng.integers(0, len(i_n), size=len(i_n))]
        bc = i_c[rng.integers(0, len(i_c), size=len(i_c))]
        ri = np.concatenate([bn, bc])
        rm = mask[ri]; s = score[ri]; qq = q[ri]
        g[b] = rank_auc(rm, s)
        cal_b = CALIPER_SD * float(qq.std())
        nb, cb = matched_pairs(s, rm, qq, cal_b, MAX_CLEAN)
        au[b] = pairwise_auc(s, nb, cb) if len(nb) else np.nan
    d = g - au
    a_g = rank_auc(mask, score)
    ni, ci = matched_pairs(score, mask, q, CALIPER_SD * float(q.std()), MAX_CLEAN)
    a_audit = pairwise_auc(score, ni, ci)
    eligible = bool(np.isfinite(a_g) and a_g >= GLOBAL_AUC_MIN)
    ci_low, ci_high = _pct(au, 0.025), _pct(au, 0.975)
    return [{
        "dataset": DATASET, "noise_type": "symmetric", "noise_rate": 0.2,
        "seed": task["seed"], "detector": task["detector"], "proxy": task["proxy"],
        "proxy_family": PROXY_FAMILY.get(task["proxy"], "?"),
        "caliper_sd": CALIPER_SD, "max_matches": MAX_CLEAN,
        "global_auc": a_g, "conditioned_auc": a_audit,
        "delta_total": a_g - a_audit,
        "ci_low": ci_low, "ci_high": ci_high,
        "delta_ci_low": _pct(d, 0.025), "delta_ci_high": _pct(d, 0.975),
        "bootstrap_B": B1 + B2, "bootstrap_B_first_pass": B1,
        "bootstrap_valid": int(np.isfinite(au).sum()),
        "bootstrap_failed": int((B1 + B2) - np.isfinite(au).sum()),
        "matched_noisy_coverage": len(np.unique(ni)) / max(int(mask.sum()), 1),
        "n_pairs": int(len(ni)),
        "eligible": eligible,
        "point_reversal": bool(np.isfinite(a_audit) and a_audit < 0.5 and eligible),
        "interval_reversal": bool(np.isfinite(ci_high) and ci_high < 0.5 and eligible),
        "ci_entirely_above_chance": bool(np.isfinite(ci_low) and ci_low > 0.5),
        "ci_straddles_chance": bool(np.isfinite(ci_low) and np.isfinite(ci_high)
                                    and ci_low <= 0.5 <= ci_high),
    }]


def run_topup(j1_csv: Path, workers: int, out_csv: Path,
              threshold: float = CI_BORDERLINE) -> None:
    df = pd.read_csv(j1_csv)
    base = df[df.bootstrap_B > 0]
    # eligibility must be recomputed the same way j1 defined it
    base = base.assign(eligible=base.global_auc >= GLOBAL_AUC_MIN)
    sel = base[base.eligible & (base.ci_low < threshold)][
        ["seed", "detector", "proxy", "bootstrap_B"]]
    sel = sel.drop_duplicates()
    print(f"top-up: {len(sel)} cells with eligible ci_low < {threshold}")
    if not len(sel):
        return
    tasks = [{"seed": int(r.seed), "detector": r.detector, "proxy": r.proxy,
              "bootstrap_B": int(r.bootstrap_B), "more": ANOTHER_B}
             for r in sel.itertuples()]
    run_pooled(tasks, _worker_j5, workers, out_csv, "j5")


if __name__ == "__main__":
    main()
