"""10_matched_pair_analysis.py — P1: direct controlled evidence for confounding.

For each run, match every noisy sample to clean samples of similar quality and
compare the raw evidence:
    diff = E_noisy - E_matched_clean
Report mean / median / effect size (Cohen's d) / fraction where E_noisy > clean,
plus a histogram. This makes AUC_matched tangible: after quality matching, the
separation attributable to the noise label itself shrinks sharply.

Usage:
    python scripts/10_matched_pair_analysis.py --exp configs/experiments/gate_a.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from quality_noise.config import hash_config, load_config
from quality_noise.data.datasets import load_bundle
from quality_noise.detectability.matched_auc import match_quality_pairs
from quality_noise.evidence.signals import assemble_evidence
from quality_noise.experiment import expand_experiment, run_key
from quality_noise.utils import make_results_dirs, write_json_atomic


def _find(run_dir: Path, prefix: str, cfg_hash: str, suffix: str = "parquet") -> Path:
    exact = run_dir / f"{prefix}_{cfg_hash}.{suffix}"
    if exact.exists():
        return exact
    cands = sorted(run_dir.glob(f"{prefix}_*.{suffix}"))
    if len(cands) == 1:
        return cands[0]
    raise FileNotFoundError(f"cannot resolve {prefix} in {run_dir}: {cands}")


def _effect_size(diff: np.ndarray) -> float:
    if diff.size == 0:
        return 0.0
    sd = diff.std()
    if sd < 1e-9:
        return 0.0
    return float(diff.mean() / sd)


def analyze_run(cfg: dict, results: dict) -> dict:
    bundle = load_bundle(cfg)
    key = run_key(cfg)
    cfg_hash = hash_config(cfg)
    base = results["traces"] / cfg["dataset"]["name"] / f"{cfg['noise']['type']}{cfg['noise']['rate']}" / f"seed{cfg['seed']}"
    q_dir = results["quality"] / cfg["dataset"]["name"] / f"{cfg['noise']['type']}{cfg['noise']['rate']}" / f"seed{cfg['seed']}"
    out_dir = results["detectability"] / cfg["dataset"]["name"] / f"{cfg['noise']['type']}{cfg['noise']['rate']}" / f"seed{cfg['seed']}"
    out_dir.mkdir(parents=True, exist_ok=True)

    traces = pd.read_parquet(_find(base, "traces", cfg_hash))
    embeddings = np.load(_find(base, "embeddings", cfg_hash, suffix="npy"))
    quality = pd.read_parquet(_find(q_dir, "quality", cfg_hash))
    y_observed = bundle.train_view.y_observed.astype(int)

    E, _ = assemble_evidence(
        cfg.get("evidence", {}).get("recipe", "ema_loss_forgetting_conflict"),
        traces, embeddings, y_observed,
        conflict_k=int(cfg.get("evidence", {}).get("conflict_k", 20)),
    )
    id2mask = {int(sid): bool(m) for sid, m in zip(bundle.eval_view.sample_ids, bundle.eval_view.mask)}
    mask = np.array([id2mask[int(sid)] for sid in quality["sample_id"].values], dtype=bool)
    mask = mask[: len(E)]

    q = quality["knn_agreement"].to_numpy(dtype=np.float64)
    ma = cfg.get("matched_auc", {})
    pairs = match_quality_pairs(E, mask, q, eps=float(ma.get("eps", 0.1)), max_clean_per_noisy=int(ma.get("max_clean_per_noisy", 5)))

    if len(pairs.s_noisy) == 0:
        return {"run_key": key, "n_pairs": 0}

    diff = pairs.s_noisy - pairs.s_clean
    from quality_noise.detectability.global_auc import global_auc

    res = {
        "run_key": key,
        "config_hash": cfg_hash,
        "n_pairs": int(len(diff)),
        "mean_diff": float(diff.mean()),
        "median_diff": float(np.median(diff)),
        "std_diff": float(diff.std()),
        "effect_size_cohens_d": _effect_size(diff),
        "frac_noisy_gt_clean": float((diff > 0).mean()),
        "auc_global_full": float(global_auc(E, mask)),
    }
    write_json_atomic(out_dir / f"matched_pair_{cfg_hash}.json", res)

    # Histogram of the evidence difference (saved for the paper).
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(diff, bins=40, alpha=0.7)
    ax.axvline(0, color="k", lw=1)
    ax.set_xlabel("E_noisy − E_matched_clean")
    ax.set_ylabel("count")
    ax.set_title(f"{key}\nmean diff={diff.mean():+.3f}, frac>0={(diff > 0).mean():.3f}")
    fig.tight_layout()
    fig.savefig(results["figures_root"] / "detectability" / f"{cfg['dataset']['name']}_{cfg['noise']['type']}{cfg['noise']['rate']}_seed{cfg['seed']}_matched_diff.png", dpi=150)
    plt.close(fig)
    print(f"[10] {key}: n_pairs={res['n_pairs']} mean_diff={diff.mean():+.3f} frac>0={(diff > 0).mean():.3f}")
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="configs/experiments/gate_a.yaml")
    ap.add_argument("--device", default="auto", help="accepted for uniform CLI; analysis is CPU-only")
    ap.add_argument("--runs", type=int, nargs="*", default=None)
    args = ap.parse_args()

    exp = load_config(args.exp)
    runs = expand_experiment(exp)
    if args.runs is not None:
        runs = [runs[i] for i in args.runs]
    results = make_results_dirs({"results_root": exp.get("results_root")})
    results["figures_root"] = Path("figures")
    for cfg in runs:
        analyze_run(cfg, results)
    print(f"[10] done: {len(runs)} runs")


if __name__ == "__main__":
    main()
