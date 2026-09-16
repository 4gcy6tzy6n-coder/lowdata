"""P0 pipeline — label-independent Q, cross-detector Delta_Q, paired bootstrap.

Two phases so the expensive per-run work is done once and cached:

Phase 1 (``--phase prepare``)
    For every run with traces on disk: load it, compute the 7 detectors and all
    quality covariates, cache to ``results/revision/prepared/``.  Parallel over
    runs (CPU-heavy: Confident Learning + kNN densities).

Phase 2 (``--phase evaluate``)
    For every (run x detector x quality) cell: point estimates of AUC_global,
    AUC_QC and Delta_Q, matching balance diagnostics, and a paired bootstrap CI.
    Parallel over cells.

Outputs (under ``results/revision/``):

    tables/independent_q_table_all_settings.csv    per cell, per seed
    tables/cross_detector.csv                      per (setting, detector, q)
    tables/matching_balance.csv                    matching audit trail
    tables/delta_q_bootstrap_ci.csv                paired bootstrap CIs
    tables/auc_qc_vs_05.csv                        is AUC_QC < 0.5 (CI)?
    summary.json                                   roll-up + gate evaluation
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

# Bound BLAS/OMP threads *before* numpy is imported: the pipeline parallelises
# over runs, so per-process threading only oversubscribes the machine (and made
# the CUDA caching allocator fail under 24 concurrent workers).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "4")

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path("/root/quality_noise")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "revision"))

from revq.bootstrap import paired_bootstrap, seed_bootstrap  # noqa: E402
from revq.matching import (balance_diagnostics, match_pairs,  # noqa: E402
                           matched_auc, smd)
from revq.prepare import (DETECTOR_NAMES, QUALITY_NAMES,  # noqa: E402
                          add_extra_quality, load_cached, prepare_and_cache)

CACHE = ROOT / "results" / "revision" / "prepared"
OUT = ROOT / "results" / "revision"

# Detector -> family (for the roll-up; dynamics vs loss-family behaviour differs).
FAMILY = {
    "ema_loss": "loss", "confidence": "confidence", "aum": "dynamics",
    "forgetting": "dynamics", "neighbor": "representation",
    "confident_learning": "label_auditing", "combined": "combined",
    "combined_cl": "combined+CL",
}

# Quality -> whether it is label-free (independent of y_tilde).
LABEL_FREE = {
    "knn_agreement": False, "proto_margin": False,
    "dino_density": True, "dino_density_fullpool": True,
    "dino_knndist": True, "dino_augcons": True,
    "native_density": "trained-encoder (z -> H_noisy -> Q; labels unused)",
    "random": True,
}


def all_settings(results_root: Path) -> list[tuple[str, str, int]]:
    """Every (dataset, noise, seed) that has saved traces."""
    from scripts.cross_detector.io_utils import list_seeds
    out = []
    for ds_dir in sorted((results_root / "traces").glob("*")):
        ds = ds_dir.name
        for noise_dir in sorted(ds_dir.glob("*")):
            noise = noise_dir.name
            for s in list_seeds(results_root, ds, noise):
                out.append((ds, noise, s))
    return out


# ---------------------------------------------------------------------------
# Phase 1
# ---------------------------------------------------------------------------

def _prep_worker(args) -> str:
    ds, noise, seed, overwrite, extra = args
    try:
        if extra:
            p = add_extra_quality(ds, noise, seed, cache=CACHE)
        else:
            p = prepare_and_cache(ds, noise, seed, cache=CACHE, overwrite=overwrite)
        return f"OK   {ds}/{noise}/seed{seed} -> {p.name}"
    except Exception as e:  # noqa: BLE001 - report and continue
        import traceback
        return f"FAIL {ds}/{noise}/seed{seed}: {type(e).__name__}: {e}\n{traceback.format_exc()}"


# ---------------------------------------------------------------------------
# Phase 2
# ---------------------------------------------------------------------------

def _cell_worker(args) -> dict:
    (ds, noise, seed, detector, qname, strategy, eps_sd, n_bs, force_greedy,
     max_noisy_bs) = args
    try:
        r = load_cached(ds, noise, seed, cache=CACHE, retries=1)
        # A quality covariate can be absent for runs cached before it existed
        # (e.g. augmentation views are per-dataset).  Emit NaN rather than
        # killing the batch.
        if detector not in r["signals"] or qname not in r["signals"]:
            return {"dataset": ds, "noise": noise, "seed": seed, "detector": detector,
                    "detector_family": FAMILY.get(detector, "?"), "quality": qname,
                    "strategy": strategy, "eps_sd": eps_sd,
                    "error": f"missing signal: det={detector in r['signals']} "
                             f"q={qname in r['signals']}"}
    except FileNotFoundError as e:
        # A run can appear on disk (training in progress) before its cache entry
        # exists.  Emit a NaN row instead of killing the whole batch.
        return {"dataset": ds, "noise": noise, "seed": seed, "detector": detector,
                "detector_family": FAMILY.get(detector, "?"), "quality": qname,
                "strategy": strategy, "eps_sd": eps_sd, "error": str(e)}
    sig = r["signals"]
    score = np.asarray(sig[detector], dtype=np.float64)
    q = np.asarray(sig[qname], dtype=np.float64)
    mask = r["mask"]
    y = r["y_observed"]

    if not np.isfinite(score).all() or not np.isfinite(q).all():
        return {"dataset": ds, "noise": noise, "seed": seed, "detector": detector,
                "quality": qname, "strategy": strategy, "eps_sd": eps_sd,
                "auc_global": float("nan"), "auc_qc": float("nan"),
                "delta_q": float("nan"), "error": "non-finite inputs"}

    auc_g = float(roc_auc_score(mask, score))
    pairs = match_pairs(score, mask, q, strategy=strategy, eps_sd=eps_sd,
                        y_observed=y, force_greedy=force_greedy)
    auc_qc = matched_auc(pairs)
    bal = balance_diagnostics(pairs, q, mask)

    bs = paired_bootstrap(score, mask, q, strategy=strategy, eps_sd=eps_sd,
                          n_resamples=n_bs,
                          seed=seed * 7919 + abs(hash((detector, qname))) % 100000,
                          point=(auc_g, auc_qc), force_greedy=True,
                          max_noisy_per_resample=max_noisy_bs)
    return {
        "dataset": ds, "noise": noise, "seed": seed,
        "detector": detector, "detector_family": FAMILY.get(detector, "?"),
        "quality": qname, "label_free": LABEL_FREE.get(qname, "?"),
        "strategy": strategy, "eps_sd": eps_sd,
        "auc_global": auc_g, "auc_qc": auc_qc, "delta_q": auc_g - auc_qc,
        "ci_global_lo": bs["ci_global"][0], "ci_global_hi": bs["ci_global"][1],
        "ci_qc_lo": bs["ci_qc"][0], "ci_qc_hi": bs["ci_qc"][1],
        "ci_delta_lo": bs["ci_delta"][0], "ci_delta_hi": bs["ci_delta"][1],
        "ci_qc_hi_lt_0p5": bool(bs["ci_qc"][1] < 0.5),
        "ci_delta_lo_gt_0": bool(bs["ci_delta"][0] > 0),
        "p_delta_le_0": bs["p_delta_le_0"],
        "n_noisy": bal["n_noisy"], "n_clean": bal["n_clean"],
        "n_pairs": bal["n_pairs"], "coverage_noisy": bal["coverage_noisy"],
        "coverage_clean": bal["coverage_clean"],
        "smd_before": bal["smd_before"], "smd_after": bal["smd_after"],
        "mean_abs_dq": bal["mean_abs_dq"], "max_abs_dq": bal["max_abs_dq"],
        "common_support_lo": bal["common_support_lo"],
        "common_support_hi": bal["common_support_hi"],
        "q_sd": float(np.std(q)),
        "q_auc_vs_mask": float(roc_auc_score(mask, q)),
    }


def _aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Per (setting, detector, quality, strategy) roll-up over seeds."""
    keys = ["dataset", "noise", "detector", "detector_family", "quality",
            "label_free", "strategy", "eps_sd"]
    rows = []
    for k, g in df.groupby(keys, dropna=False):
        d = dict(zip(keys, k))
        g = g.sort_values("seed")
        d["n_seeds"] = int(len(g))
        for col in ["auc_global", "auc_qc", "delta_q"]:
            d[f"{col}_mean"] = float(g[col].mean())
            d[f"{col}_std"] = float(g[col].std(ddof=1)) if len(g) > 1 else float("nan")
        sb = seed_bootstrap(g["delta_q"].to_numpy())
        d["delta_q_seed_ci_lo"], d["delta_q_seed_ci_hi"] = sb["ci"]
        d["delta_q_sign_consistency"] = sb["sign_consistency"]
        d["delta_q_n_positive_seeds"] = int((g["delta_q"] > 0).sum())
        d["auc_qc_mean_ci_lo"] = float(g["ci_qc_lo"].min())
        d["auc_qc_mean_ci_hi"] = float(g["ci_qc_hi"].max())
        d["n_seeds_ci_qc_hi_lt_0p5"] = int(g["ci_qc_hi_lt_0p5"].sum())
        d["n_seeds_ci_delta_lo_gt_0"] = int(g["ci_delta_lo_gt_0"].sum())
        for col in ["smd_before", "smd_after", "coverage_noisy", "coverage_clean",
                    "n_pairs", "mean_abs_dq", "q_auc_vs_mask"]:
            d[f"{col}_mean"] = float(g[col].mean())
        rows.append(d)
    return pd.DataFrame(rows).sort_values(keys).reset_index(drop=True)


def _json_safe_groupmean(df: pd.DataFrame, keys: list[str]) -> dict:
    """groupby(...).mean().to_dict() produces tuple keys, which json rejects."""
    if not len(df):
        return {}
    g = df.groupby(keys)["delta_q"].mean()
    return {"/".join(str(x) for x in k): float(v) for k, v in g.items()}


def _gate_summary(cells: pd.DataFrame) -> dict:
    """The four decisive questions, evaluated on the label-free quality."""
    lf = cells[cells["quality"] == "dino_density"]
    main = lf[lf["strategy"] == "nn_wo"]
    g1 = {
        "question": "Label-independent Q => Delta_Q > 0 still appears?",
        "n_cells": int(len(main)),
        "n_positive": int((main["delta_q"] > 0).sum()),
        "frac_positive": float((main["delta_q"] > 0).mean()) if len(main) else float("nan"),
        "n_ci_delta_gt_0": int(main["ci_delta_lo_gt_0"].sum()),
        "mean_delta_q": float(main["delta_q"].mean()) if len(main) else float("nan"),
        "c100_s20_delta_q": main[(main.dataset == "cifar100")]["delta_q"].tolist(),
        "c100_s20_auc_qc": main[(main.dataset == "cifar100")]["auc_qc"].tolist(),
        "c100_s20_auc_global": main[(main.dataset == "cifar100")]["auc_global"].tolist(),
        "c100_s20_detectors": main[(main.dataset == "cifar100")]["detector"].tolist(),
        "n_ci_qc_hi_lt_0p5": int(main["ci_qc_hi_lt_0p5"].sum()),
    }
    by_det = main.groupby("detector")["delta_q"].agg(["mean", "count",
                                                      lambda s: (s > 0).mean()])
    by_det.columns = ["mean_delta_q", "n", "frac_positive"]
    g2 = by_det.reset_index().to_dict(orient="records")
    cl = main[main.detector == "confident_learning"]
    g3 = {
        "question": "Does Confident Learning also show quality sensitivity?",
        "n_cells": int(len(cl)),
        "n_positive": int((cl["delta_q"] > 0).sum()),
        "mean_delta_q": float(cl["delta_q"].mean()) if len(cl) else float("nan"),
        "per_setting": _json_safe_groupmean(cl, ["dataset", "noise"]),
    }
    real = main[main.dataset.isin(["cifar10n"])]
    g4 = {
        "question": "Does the phenomenon survive human/real annotation noise?",
        "n_cells": int(len(real)),
        "n_positive": int((real["delta_q"] > 0).sum()),
        "mean_delta_q": float(real["delta_q"].mean()) if len(real) else float("nan"),
        "per_setting": _json_safe_groupmean(real, ["dataset", "noise"]),
    }
    return {"G1_label_independent_delta_q": g1, "G1_by_detector": g2,
            "G3_confident_learning": g3, "G4_real_noise": g4}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["prepare", "evaluate", "all"], default="all")
    ap.add_argument("--results-root", type=Path, default=ROOT / "results")
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--datasets", type=str, default="cifar10,cifar100,cifar10n,cifar100n")
    ap.add_argument("--strategy", type=str, default="nn_wo")
    ap.add_argument("--eps-sd", type=float, default=0.1)
    ap.add_argument("--quality", type=str, default="all")
    ap.add_argument("--detectors", type=str, default="all")
    ap.add_argument("--overwrite-prep", action="store_true")
    ap.add_argument("--extra-quality", action="store_true",
                    help="only merge the extra label-free covariates into the cache")
    ap.add_argument("--force-greedy", action="store_true",
                    help="use greedy matching instead of optimal assignment")
    ap.add_argument("--max-noisy-bs", type=int, default=1500,
                    help="noisy budget inside each bootstrap resample (bounds runtime)")
    args = ap.parse_args()

    datasets = {d.strip() for d in args.datasets.split(",") if d.strip()}
    runs = [r for r in all_settings(args.results_root) if r[0] in datasets]
    if args.phase == "evaluate":
        # Only evaluate runs whose cached signals exist; runs mid-training are
        # skipped.  The prepare phase must NOT filter this way -- its whole job is
        # to create those cache entries.
        prepared = {p.parent.parent.name + "/" + p.parent.name + "/" + p.stem
                    for p in CACHE.glob("*/*/seed*.npz")}
        missing = [r for r in runs if f"{r[0]}/{r[1]}/seed{r[2]}" not in prepared]
        if missing:
            print(f"   skipping {len(missing)} unprepared runs: {missing[:6]}"
                  f"{' ...' if len(missing) > 6 else ''}", flush=True)
        runs = [r for r in runs if f"{r[0]}/{r[1]}/seed{r[2]}" in prepared]
    print(f"== P0 label-independent Q pipeline ==", flush=True)
    print(f"   runs: {len(runs)}  bootstrap: {args.bootstrap}  workers: {args.workers}", flush=True)
    print(f"   strategy: {args.strategy}  eps_sd: {args.eps_sd}", flush=True)

    if args.phase in ("prepare", "all"):
        t0 = time.time()
        print(f"\nPhase 1: prepare {len(runs)} runs", flush=True)
        with mp.Pool(processes=min(args.workers, len(runs))) as pool:
            for i, msg in enumerate(pool.imap_unordered(
                    _prep_worker, [(d, n, s, args.overwrite_prep, args.extra_quality) for d, n, s in runs])):
                if msg.startswith("FAIL") or i % 5 == 0:
                    print(f"  [{i+1}/{len(runs)}] {msg}", flush=True)
        print(f"   prepare done in {time.time()-t0:.1f}s", flush=True)

    if args.phase in ("evaluate", "all"):
        (OUT / "tables").mkdir(parents=True, exist_ok=True)
        dets = DETECTOR_NAMES if args.detectors == "all" else args.detectors.split(",")
        # Only qualities that were actually cached.
        probe = load_cached(*runs[0][:3], cache=CACHE, retries=1)
        qs = [q for q in QUALITY_NAMES if q in probe["signals"]]
        qs += [q for q in probe["signals"] if q.startswith("dino_density") and q not in qs]
        if args.quality != "all":
            want = set(args.quality.split(","))
            qs = [q for q in qs if q in want]
        print(f"\nPhase 2: evaluate {len(runs)} runs x {len(dets)} detectors x {len(qs)} qualities"
              f" = {len(runs)*len(dets)*len(qs)} cells", flush=True)
        print(f"   qualities: {qs}", flush=True)

        work = [(d, n, s, det, q, args.strategy, args.eps_sd, args.bootstrap,
                 args.force_greedy, args.max_noisy_bs)
                for d, n, s in runs for det in dets for q in qs
                if det in probe["signals"] and q in probe["signals"]]
        print(f"   planned cells: {len(work)}", flush=True)
        t1 = time.time()
        rows = []
        with mp.Pool(processes=args.workers) as pool:
            for i, row in enumerate(pool.imap_unordered(_cell_worker, work, chunksize=4)):
                rows.append(row)
                if (i + 1) % 100 == 0:
                    print(f"   {i+1}/{len(work)} cells ({time.time()-t1:.0f}s)", flush=True)
        cells = pd.DataFrame(rows)
        # Per-dataset shards let a later batch (e.g. newly trained settings) be
        # merged with pandas instead of re-running every cell.
        shard_dir = OUT / "tables" / "shards"
        shard_dir.mkdir(parents=True, exist_ok=True)
        for ds, g in cells.groupby("dataset"):
            g.to_csv(shard_dir / f"{ds}.csv", index=False)
        n_err = int(cells.get("error", pd.Series(dtype=object)).notna().sum()) if "error" in cells else 0
        if n_err:
            print(f"   {n_err} cells had no cached run (recorded, excluded from aggregates)", flush=True)
        cells.to_csv(OUT / "tables" / "independent_q_table_all_settings.csv", index=False)
        cells = cells[cells["auc_global"].notna()].reset_index(drop=True)
        agg = _aggregate(cells)
        agg.to_csv(OUT / "tables" / "cross_detector.csv", index=False)

        bal_cols = ["dataset", "noise", "seed", "detector", "quality", "strategy",
                    "eps_sd", "n_noisy", "n_clean", "n_pairs", "coverage_noisy",
                    "coverage_clean", "smd_before", "smd_after", "mean_abs_dq",
                    "max_abs_dq", "common_support_lo", "common_support_hi"]
        cells[bal_cols].to_csv(OUT / "tables" / "matching_balance.csv", index=False)

        ci_cols = ["dataset", "noise", "seed", "detector", "quality", "strategy",
                   "eps_sd", "auc_global", "ci_global_lo", "ci_global_hi",
                   "auc_qc", "ci_qc_lo", "ci_qc_hi", "delta_q",
                   "ci_delta_lo", "ci_delta_hi", "p_delta_le_0"]
        cells[ci_cols].to_csv(OUT / "tables" / "delta_q_bootstrap_ci.csv", index=False)

        v = cells.copy()
        v["ci_excludes_0p5_below"] = v["ci_qc_hi_lt_0p5"]
        v[["dataset", "noise", "seed", "detector", "quality", "auc_qc",
           "ci_qc_lo", "ci_qc_hi", "ci_excludes_0p5_below", "n_pairs",
           "coverage_noisy"]].to_csv(OUT / "tables" / "auc_qc_vs_05.csv", index=False)

        summary = _gate_summary(cells)
        summary["meta"] = {
            "n_runs": len(runs), "n_cells": len(cells),
            "bootstrap_resamples": args.bootstrap,
            "strategy": args.strategy, "eps_sd": args.eps_sd,
            "qualities": qs, "detectors": dets,
            "wall_s": round(time.time() - t1, 1),
        }
        with open(OUT / "summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)

        print(f"\n== done in {time.time()-t1:.0f}s ==", flush=True)
        print(f"   cells: {len(cells)}  -> {OUT/'tables'}", flush=True)
        g = summary["G1_label_independent_delta_q"]
        print(f"   G1: {g['n_positive']}/{g['n_cells']} cells Delta_Q>0 with Q^DINO, "
              f"mean {g['mean_delta_q']:+.3f}", flush=True)


if __name__ == "__main__":
    main()
