"""11_cross_detector.py — P0 Cross-Detector Detectability Matrix (v2, per-cell parallel).

Computes 6 detectors (D1-D6) post-hoc per (dataset, noise, seed), then for
each (detector, setting) cell evaluates:
    AUC_global, AUC_matched, Δ_conf
    bootstrap 95% CI for global / matched / Δ_conf (joint resample)
    matching-quality diagnostics (SMD before/after, n_pairs, etc.)

Pipeline:
    Phase 1 (parallel): load each run once, compute all 6 detectors
    Phase 2 (parallel, per-cell): matching + bootstrap for every cell
    Phase 3: write JSONs

No retraining; all post-hoc on saved traces / embeddings / quality / mask.

Outputs (one JSON per cell):
    results/cross_detector/<dataset>/<noise>/seed<seed>/<detector>.json

Usage:
    python scripts/11_cross_detector.py --bootstrap 200 --workers 64
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.cross_detector.detectors import (  # noqa: E402
    compute_all_detectors,
    DETECTORS,
)
from scripts.cross_detector.io_utils import (  # noqa: E402
    all_settings,
    list_seeds,
    load_run,
    write_json_atomic,
)
from sklearn.metrics import roc_auc_score  # noqa: E402

from quality_noise.detectability.matched_auc import (  # noqa: E402
    match_quality_pairs,
    matched_auc,
)


# ---------------------------------------------------------------------------
# Phase 1 worker: load run, compute all 6 detectors once.
# ---------------------------------------------------------------------------

def _prepare_run(args) -> dict:
    dataset, noise, seed, results_root = args
    t0 = time.time()
    run = load_run(results_root, dataset, noise, seed)
    t1 = time.time()
    detectors = compute_all_detectors(
        run["traces"], run["embeddings"], run["y_observed"]
    )
    t2 = time.time()
    mask = run["mask"]
    q_knn = run["quality"]["knn_agreement"].to_numpy(dtype=np.float64)
    q_proto = run["quality"]["margin"].to_numpy(dtype=np.float64)
    n = len(detectors[next(iter(detectors))])
    return {
        "dataset": dataset,
        "noise": noise,
        "seed": seed,
        "detectors": {k: v[:n] for k, v in detectors.items()},
        "mask": mask[:n],
        "q_knn": q_knn[:n],
        "q_proto": q_proto[:n],
        "load_s": round(t1 - t0, 2),
        "detect_s": round(t2 - t1, 2),
    }


# ---------------------------------------------------------------------------
# Phase 2 worker: evaluate ONE cell (detector x q-kind). Picklable.
# ---------------------------------------------------------------------------

def _eval_cell_worker(args) -> dict:
    (dataset, noise, seed, det_name, score, mask, q,
     eps, max_clean, n_bs, q_kind, seed_tag) = args

    # --- point estimates ---
    pairs = match_quality_pairs(score, mask, q, eps=eps, max_clean_per_noisy=max_clean)
    auc_g = float(roc_auc_score(mask, score)) if mask.sum() > 0 and (~mask).sum() > 0 else float("nan")
    auc_m = matched_auc(pairs)
    delta = auc_g - auc_m if not np.isnan(auc_m) else float("nan")

    # --- matching diagnostics ---
    noisy_q = q[mask]
    clean_q = q[~mask]
    smd_before = _smd(noisy_q, clean_q)
    smd_after = _smd(np.asarray(pairs.q_noisy), np.asarray(pairs.q_clean))
    diffs = np.abs(np.asarray(pairs.q_noisy) - np.asarray(pairs.q_clean))
    diag = {
        "n_noisy": int(mask.sum()),
        "n_clean": int((~mask).sum()),
        "n_pairs": int(len(pairs.s_noisy)),
        "match_rate": float(len(pairs.s_noisy) / max(mask.sum(), 1)),
        "smd_before": float(smd_before),
        "smd_after": float(smd_after),
        "smd_reduction": float(abs(smd_before) - abs(smd_after)),
        "mean_abs_q_diff": float(diffs.mean()) if diffs.size else 0.0,
        "median_abs_q_diff": float(np.median(diffs)) if diffs.size else 0.0,
        "max_abs_q_diff": float(diffs.max()) if diffs.size else 0.0,
    }

    # --- joint bootstrap (sample-level resample; re-match inside each resample) ---
    rng = np.random.default_rng(seed_tag * 1000 + (hash(det_name) % 9999))
    noisy_idx = np.where(mask)[0]
    clean_idx = np.where(~mask)[0]
    n_noisy, n_clean = len(noisy_idx), len(clean_idx)
    sub_noisy = min(2000, n_noisy)
    g_stats = np.empty(n_bs)
    m_stats = np.empty(n_bs)
    for r in range(n_bs):
        bn = noisy_idx[rng.integers(0, n_noisy, size=n_noisy)]
        bc = clean_idx[rng.integers(0, n_clean, size=n_clean)]
        ridx = np.concatenate([bn, bc])
        rmask = mask[ridx]
        rscore = score[ridx]
        rq = q[ridx]
        try:
            g_stats[r] = roc_auc_score(rmask, rscore)
        except ValueError:
            g_stats[r] = float("nan")
        # Subsample noisy (without replacing within resample) to bound runtime.
        sub = rng.choice(n_noisy, size=sub_noisy, replace=False)
        m_ridx = np.concatenate([bn[sub], bc])
        m_mask = mask[m_ridx]
        m_pairs = match_quality_pairs(
            score[m_ridx], m_mask, q[m_ridx], eps=eps, max_clean_per_noisy=max_clean
        )
        m_stats[r] = matched_auc(m_pairs) if len(m_pairs.s_noisy) else float("nan")
    d_stats = g_stats - m_stats
    ci_g = _pct_ci(g_stats)
    ci_m = _pct_ci(m_stats)
    ci_d = _pct_ci(d_stats)

    return {
        "dataset": dataset,
        "noise": noise,
        "seed": seed,
        "detector": det_name,
        "quality_col": q_kind,
        "auc_global": auc_g,
        "auc_matched": auc_m,
        "delta_conf": delta,
        "ci_global": list(ci_g),
        "ci_matched": list(ci_m),
        "ci_delta": list(ci_d),
        "n_noisy": int(mask.sum()),
        "n_clean": int((~mask).sum()),
        "matching": diag,
    }


def _smd(x_noisy: np.ndarray, x_clean: np.ndarray) -> float:
    """Standardized Mean Difference: (mu_noisy - mu_clean) / pooled_sd."""
    if len(x_noisy) == 0 or len(x_clean) == 0:
        return float("nan")
    mu_n, mu_c = float(x_noisy.mean()), float(x_clean.mean())
    var_n, var_c = float(x_noisy.var()), float(x_clean.var())
    pooled = np.sqrt(((len(x_noisy) - 1) * var_n + (len(x_clean) - 1) * var_c) /
                     max(len(x_noisy) + len(x_clean) - 2, 1))
    if pooled < 1e-12:
        return 0.0
    return (mu_n - mu_c) / pooled


def _pct_ci(arr: np.ndarray) -> tuple[float, float]:
    v = arr[~np.isnan(arr)]
    if len(v) == 0:
        return (float("nan"), float("nan"))
    return (float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975)))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", type=Path, default=Path("/root/quality_noise/results"))
    ap.add_argument("--out-root", type=Path, default=Path("/root/quality_noise/results/cross_detector"))
    ap.add_argument("--bootstrap", type=int, default=200)
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--datasets", type=str, default="cifar10,cifar100,cifar10n")
    ap.add_argument("--noises", type=str, default="all")
    ap.add_argument("--detectors", type=str, default=",".join(DETECTORS.keys()))
    ap.add_argument("--eps", type=float, default=0.1)
    ap.add_argument("--max-clean", type=int, default=5)
    ap.add_argument("--quality-cols", type=str, default="knn_agreement",
                    help="comma list: knn_agreement, margin")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    datasets = set(args.datasets.split(","))
    noise_filter = None if args.noises == "all" else set(args.noises.split(","))
    selected_detectors = args.detectors.split(",")
    quality_cols = [q.strip() for q in args.quality_cols.split(",") if q.strip()]

    # Build run list
    runs = []
    skipped_settings = []
    for dataset, noise in all_settings():
        if dataset not in datasets:
            continue
        if noise_filter and noise not in noise_filter:
            continue
        seeds = list_seeds(args.results_root, dataset, noise)
        if not seeds:
            skipped_settings.append((dataset, noise))
            continue
        for seed in seeds:
            runs.append((dataset, noise, seed))

    print(f"== Cross-detector pass v2 ==", flush=True)
    print(f"   detectors: {selected_detectors}", flush=True)
    print(f"   quality cols: {quality_cols}", flush=True)
    print(f"   bootstrap: {args.bootstrap}, workers: {args.workers}", flush=True)
    print(f"   eps={args.eps} max_clean={args.max_clean}", flush=True)
    print(f"   runs: {len(runs)}  skipped: {skipped_settings}", flush=True)
    print(flush=True)

    if not runs:
        print("Nothing to do.", flush=True)
        return

    t0 = time.time()
    # Phase 1: prepare runs
    print("Phase 1: loading runs + computing detectors...", flush=True)
    with mp.Pool(processes=min(args.workers, len(runs))) as pool:
        prepared = pool.map(
            _prepare_run,
            [(d, n, s, args.results_root) for d, n, s in runs],
        )
    print(f"   prepared {len(prepared)} runs in {time.time()-t0:.1f}s", flush=True)

    # Phase 2: per-cell evaluation
    t1 = time.time()
    work = []
    for p in prepared:
        q_map = {"knn_agreement": p["q_knn"], "margin": p["q_proto"]}
        for q_kind in quality_cols:
            q = q_map[q_kind]
            for det in selected_detectors:
                score = p["detectors"][det]
                seed_tag = p["seed"]
                work.append((
                    p["dataset"], p["noise"], p["seed"], det, score,
                    p["mask"], q, args.eps, args.max_clean, args.bootstrap,
                    q_kind, seed_tag,
                ))
    print(f"Phase 2: evaluating {len(work)} cells...", flush=True)
    with mp.Pool(processes=args.workers) as pool:
        cells = pool.map(_eval_cell_worker, work)
    print(f"   evaluated {len(cells)} cells in {time.time()-t1:.1f}s", flush=True)

    # Phase 3: write
    n_written = 0
    for cell in cells:
        # write into per-quality-col subdir so margin results don't collide
        qsub = cell["quality_col"]
        out_path = (
            args.out_root / qsub / cell["dataset"] / cell["noise"]
            / f"seed{cell['seed']}" / f"{cell['detector']}.json"
        )
        write_json_atomic(out_path, cell)
        n_written += 1

    print(flush=True)
    print(f"== Summary ==", flush=True)
    print(f"   runs: {len(prepared)}  cells written: {n_written}", flush=True)
    print(f"   output root: {args.out_root}", flush=True)
    print(f"   total wall: {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
