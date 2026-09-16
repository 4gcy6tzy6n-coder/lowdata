"""E9: CIFAR-100N human noise -- does the conditioning effect survive real noise?

Gate 4 of the revision asks whether the quality-conditioning phenomenon is an
artefact of *synthetic* label noise.  CIFAR-100N carries real, human-annotated
label noise (`noisy_label`: 20,100 / 50,000 = 0.4022 realised error rate), so it
is the natural test: the noise is not a uniform flip of a known rate, and the
quality of a sample's label error is not exchangeable with anything the training
run was told.

This driver mirrors the primary pipeline exactly -- same constrained 1:many
quality match, same rank AUROC, same paired equal-count stratified bootstrap --
so the numbers are directly comparable to the C100-S20 / C100-A40 audits.  It is
kept standalone because `run_round2.py` hard-codes the two synthetic settings.

Reported per cell:
  * ``global_auc``  -- noisy-vs-clean detectability with no conditioning;
  * ``conditioned_auc`` -- detectability among quality-matched pairs;
  * ``delta_total`` -- the drop, i.e. the paper's main effect;
  * paired bootstrap CI on ``conditioned_auc``, which is what decides whether a
    "reversal" (conditioned detectability at or below chance) is
    interval-supported rather than a point estimate.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "revision"))

from run_p0_batch import (CACHE, CLEAN_BUDGET, GLOBAL_AUC_MIN,  # noqa: E402
                          MAX_CLEAN, NOISY_BUDGET, PROXIES, PROXY_FAMILY,
                          _pct, matched_pairs, pairwise_auc, rank_auc)
from run_round2 import detector_names, get_detector  # noqa: E402

OUT = ROOT / "results" / "revision" / "round2"
SETTING = ("cifar100n", "cifar100n_human0.4")


def evaluate(score: np.ndarray, mask: np.ndarray, q: np.ndarray,
             B: int, tag: int) -> dict:
    """Point estimates plus a paired equal-count stratified bootstrap."""
    score = np.asarray(score, np.float64)
    mask = np.asarray(mask, bool)
    q = np.asarray(q, np.float64)

    a_g = rank_auc(mask, score)
    ni, ci = matched_pairs(score, mask, q, 0.10 * float(q.std()), MAX_CLEAN)
    a_c = pairwise_auc(score, ni, ci)

    row = {
        "global_auc": a_g,
        "conditioned_auc": a_c,
        "delta_total": a_g - a_c if np.isfinite(a_c) else np.nan,
        "n_noisy": int(mask.sum()), "n_clean": int((~mask).sum()),
        "noisy_coverage": len(np.unique(ni)) / max(int(mask.sum()), 1) if len(ni) else 0.0,
        "clean_coverage": len(np.unique(ci)) / max(int((~mask).sum()), 1) if len(ci) else 0.0,
        "n_pairs": int(len(ni)),
        "rq": float(q[mask].mean() - q[~mask].mean()),
        "eligible": bool(np.isfinite(a_g) and a_g >= GLOBAL_AUC_MIN),
    }
    if B <= 0:
        return row

    i_n, i_c = np.where(mask)[0], np.where(~mask)[0]
    if len(i_n) > NOISY_BUDGET:
        i_n = np.random.default_rng(tag + 991).choice(i_n, NOISY_BUDGET, replace=False)
    if len(i_c) > CLEAN_BUDGET:
        i_c = np.random.default_rng(tag + 992).choice(i_c, CLEAN_BUDGET, replace=False)
    rng = np.random.default_rng(tag)
    g = np.empty(B); au = np.empty(B)
    for b in range(B):
        bn = i_n[rng.integers(0, len(i_n), size=len(i_n))]
        bc = i_c[rng.integers(0, len(i_c), size=len(i_c))]
        idx = np.concatenate([bn, bc])
        rm, s, qq = mask[idx], score[idx], q[idx]
        g[b] = rank_auc(rm, s)
        nb, cb = matched_pairs(s, rm, qq, 0.10 * float(qq.std()), MAX_CLEAN)
        au[b] = pairwise_auc(s, nb, cb) if len(nb) else np.nan
    lo, hi = _pct(au, 0.025), _pct(au, 0.975)
    row.update({
        "bootstrap_B": B,
        "global_ci_low": _pct(g, 0.025), "global_ci_high": _pct(g, 0.975),
        "ci_low": lo, "ci_high": hi,
        "bootstrap_valid": int(np.isfinite(au).sum()),
        "bootstrap_failed": int(B - np.isfinite(au).sum()),
        "ci_entirely_above_chance": bool(np.isfinite(lo) and lo > 0.5),
        "ci_straddles_chance": bool(np.isfinite(lo) and np.isfinite(hi) and lo <= 0.5 <= hi),
        "point_reversal": bool(np.isfinite(a_c) and a_c < 0.5 and row["eligible"]),
        "interval_reversal": bool(np.isfinite(hi) and hi < 0.5 and row["eligible"]),
    })
    return row


def _worker(task: dict) -> list[dict]:
    dataset, noise = SETTING
    with np.load(CACHE / dataset / noise / f"seed{task['seed']}.npz",
                 allow_pickle=False) as z:
        # combined_noN is derived on the fly, so route every detector through
        # get_detector rather than reading sig__<name> directly.
        p = {k: np.asarray(z[k]).copy() for k in z.files}
    mask = np.asarray(p["mask"], bool)
    out = []
    for det in task["detectors"]:
        score = get_detector(p, det)
        for px in task["proxies"]:
            q = np.asarray(p[f"sig__{px}"], np.float64)
            row = evaluate(score, mask, q, task["B"],
                           task["seed"] * 977 + abs(hash(det + px)) % 9973)
            row.update({"seed": task["seed"], "detector": det, "proxy": px,
                        "proxy_family": PROXY_FAMILY.get(px, "?"),
                        "setting": f"{dataset}/{noise}", "B": task["B"]})
            out.append(row)
    return out


def summarise(df: pd.DataFrame) -> None:
    print("\n== e9: CIFAR-100N (human noise, realised rate %.4f) =="
          % (df.n_noisy.iloc[0] / (df.n_noisy.iloc[0] + df.n_clean.iloc[0])))
    # B=0 runs carry point estimates only, so the bootstrap-derived columns are
    # absent; fill them with zero so the same summary works for both.
    df = df.copy()
    for c in ("point_reversal", "interval_reversal", "ci_entirely_above_chance"):
        if c not in df.columns:
            df[c] = False
    g = (df.groupby(["detector", "proxy", "proxy_family"], as_index=False)
         .agg(Aglobal=("global_auc", "mean"), Acond=("conditioned_auc", "mean"),
              delta=("delta_total", "mean"), ncov=("noisy_coverage", "mean"),
              point_rev=("point_reversal", "sum"),
              int_rev=("interval_reversal", "sum"),
              ci_above=("ci_entirely_above_chance", "sum"),
              n=("seed", "size")))
    g["ci_above_frac"] = g.ci_above / g.n
    piv = g.pivot_table(index="detector", columns="proxy", values="delta", aggfunc="mean")
    print("\ndelta_total (detector x proxy):")
    print(piv.round(4).to_string())
    print("\ninterval reversals (detector x proxy):")
    print(g.pivot_table(index="detector", columns="proxy",
                        values="int_rev", aggfunc="sum").fillna(0).astype(int).to_string())
    print("\nby proxy (all eligible detectors pooled):")
    p = (g.groupby("proxy", as_index=False)
         .agg(delta=("delta", "mean"), Aglobal=("Aglobal", "mean"),
              Acond=("Acond", "mean"), point_rev=("point_rev", "sum"),
              int_rev=("int_rev", "sum"), cells=("n", "sum"),
              ci_above=("ci_above", "sum")))
    print(p.round(4).to_string(index=False))
    g.to_csv(OUT / "e9_c100n_summary.csv", index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    ap.add_argument("--seeds", default="all")
    args = ap.parse_args()

    dataset, noise = SETTING
    seeds = sorted(int(p.stem[4:]) for p in (CACHE / dataset / noise).glob("seed*.npz"))
    if args.seeds != "all":
        want = {int(s) for s in args.seeds.split(",")}
        seeds = [s for s in seeds if s in want]
    dets = detector_names()
    tasks = [{"seed": s, "detectors": dets, "proxies": PROXIES, "B": args.B}
             for s in seeds]
    print(f"== e9: CIFAR-100N, {len(tasks)} seeds x {len(dets)} detectors "
          f"x {len(PROXIES)} proxies, B={args.B} (max_clean={MAX_CLEAN}) ==", flush=True)

    t0 = time.time(); rows = []
    if args.workers > 1:
        with mp.Pool(processes=min(args.workers, len(tasks))) as pool:
            for i, res in enumerate(pool.imap_unordered(_worker, tasks), 1):
                rows.extend(res)
                pd.DataFrame(rows).to_csv(OUT / "e9_c100n_audit.partial.csv", index=False)
                print(f"   [{i}/{len(tasks)}] {time.time()-t0:.0f}s, {len(rows)} rows", flush=True)
    else:
        for t in tasks:
            rows.extend(_worker(t))
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "e9_c100n_audit.csv", index=False)
    print(f"\nwrote e9_c100n_audit.csv ({len(df)} rows, {time.time()-t0:.0f}s)")
    summarise(df)


if __name__ == "__main__":
    main()
