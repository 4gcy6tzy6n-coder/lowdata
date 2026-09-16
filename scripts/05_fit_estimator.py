"""05_fit_estimator.py — fit the reliability-aware noise estimator.

For each run:
    load quality + traces + embeddings
    -> assemble evidence E
    -> estimate_noise_posterior(E, Q)   (no mask in the signature)
    -> save weights / eta / reliability / pi_q
    -> report retention metrics vs the evaluation-only mask (reporting only)

Usage:
    python scripts/05_fit_estimator.py --exp configs/experiments/gate_a.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from quality_noise.config import hash_config, load_config
from quality_noise.data.datasets import load_bundle
from quality_noise.evidence.signals import assemble_evidence
from quality_noise.experiment import expand_experiment, run_key
from quality_noise.metrics.detection import detection_report
from quality_noise.metrics.retention import false_exclusion_rate, hard_clean_false_suppression
from quality_noise.reliability.estimator import estimate_noise_posterior
from quality_noise.utils import make_results_dirs, write_json_atomic


def _find_run_artifact(run_dir: Path, prefix: str, cfg_hash: str, suffix: str = "parquet") -> Path:
    """Resolve a 02/03-produced artifact for this run.

    Prefers the exact hash-named file; falls back to scanning the run directory
    (variant configs change the hash but reuse the same traces/quality data).
    """
    exact = run_dir / f"{prefix}_{cfg_hash}.{suffix}"
    if exact.exists():
        return exact
    cands = sorted(run_dir.glob(f"{prefix}_*.{suffix}"))
    if len(cands) == 1:
        return cands[0]
    raise FileNotFoundError(f"cannot resolve {prefix} in {run_dir}: {cands}")


def fit_estimator(cfg: dict, results: dict, tag: str = "") -> None:
    bundle = load_bundle(cfg)
    key = run_key(cfg)
    cfg_hash = hash_config(cfg)
    tr_dir = results["traces"] / cfg["dataset"]["name"] / f"{cfg['noise']['type']}{cfg['noise']['rate']}" / f"seed{cfg['seed']}"
    q_dir = results["quality"] / cfg["dataset"]["name"] / f"{cfg['noise']['type']}{cfg['noise']['rate']}" / f"seed{cfg['seed']}"
    out_dir = results["estimator"] / cfg["dataset"]["name"] / f"{cfg['noise']['type']}{cfg['noise']['rate']}" / f"seed{cfg['seed']}"
    out_dir.mkdir(parents=True, exist_ok=True)

    traces = pd.read_parquet(_find_run_artifact(tr_dir, "traces", cfg_hash))
    embeddings = np.load(_find_run_artifact(tr_dir, "embeddings", cfg_hash, suffix="npy"))
    quality = pd.read_parquet(_find_run_artifact(q_dir, "quality", cfg_hash))
    y_observed = bundle.train_view.y_observed.astype(int)

    ev_cfg = cfg.get("evidence", {})
    E, _ = assemble_evidence(
        ev_cfg.get("recipe", "ema_loss_forgetting_conflict"),
        traces,
        embeddings,
        y_observed,
        conflict_k=int(ev_cfg.get("conflict_k", 20)),
    )
    Q = quality["knn_agreement"].to_numpy(dtype=np.float64)

    res = estimate_noise_posterior(E, Q, cfg)

    # Report (evaluation-only mask read at the very end, for reporting).
    id2mask = {int(sid): bool(m) for sid, m in zip(bundle.eval_view.sample_ids, bundle.eval_view.mask)}
    mask = np.array([id2mask[int(sid)] for sid in quality["sample_id"].values], dtype=bool)
    is_clean = ~mask

    det = detection_report(res.eta_shrunk, mask)
    retention = {
        "hard_clean_false_suppression": hard_clean_false_suppression(res.weights, Q, is_clean),
        "false_exclusion_rate": false_exclusion_rate(res.weights, is_clean),
        "noise_rate_estimate": float(np.mean(res.weights < 0.5)),
        "true_noise_rate": float(mask.mean()),
    }
    summary = {
        **det,
        **retention,
        "run_key": key,
        "config_hash": cfg_hash,
        "rate_used": float(res.rate_used),
        "weight_mode": cfg.get("reliability", {}).get("weight_mode", "calibrated"),
        "weight_power": cfg.get("reliability", {}).get("weight_power", 1.0),
        "hard_threshold": cfg.get("reliability", {}).get("hard_threshold", None),
        "rate_source": cfg.get("reliability", {}).get("rate_source", "config"),
    }
    out_name = f"estimator_summary_{tag}.json" if tag else "estimator_summary.json"
    write_json_atomic(out_dir / out_name, summary)

    pd.DataFrame(
        {
            "sample_id": quality["sample_id"].values,
            "eta_raw": res.eta_raw,
            "eta_shrunk": res.eta_shrunk,
            "weight": res.weights,
            "reliability": res.reliability,
            "quality_bin": res.bin_id,
        }
    ).to_parquet(out_dir / f"posterior_{cfg_hash}.parquet", index=False)

    print(f"[05] {key}: AUROC={det['auroc']:.3f} FER={retention['false_exclusion_rate']:.3f} "
          f"noise_est={retention['noise_rate_estimate']:.2f} (true {retention['true_noise_rate']:.2f})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="configs/experiments/gate_a.yaml")
    ap.add_argument("--device", default="auto", help="accepted for uniform CLI; estimator is CPU-only")
    ap.add_argument("--runs", type=int, nargs="*", default=None, help="restrict to these run indices")
    ap.add_argument("--tag", default="", help="write estimator_summary_<tag>.json (keeps variants side by side)")
    args = ap.parse_args()

    exp = load_config(args.exp)
    runs = expand_experiment(exp)
    if args.runs is not None:
        runs = [runs[i] for i in args.runs]
    results = make_results_dirs({"results_root": exp.get("results_root")})
    for cfg in runs:
        fit_estimator(cfg, results, tag=args.tag)
    print(f"[05] done: {len(runs)} estimators (tag={args.tag or 'default'})")


if __name__ == "__main__":
    main()
