"""06_train_weighted.py — train with reliability-aware soft weights (Phase 7).

For each run:
    load the estimator weights (from 05)
    -> train with weighted CE (same scheduler/seeding as standard)
    -> evaluate on the held-out test set
    -> write downstream metrics

Gate B evaluation compares standard CE, the weighted method, and small-loss on
three dimensions: noise detection, hard-clean damage, downstream accuracy.

Usage:
    python scripts/06_train_weighted.py --exp configs/experiments/gate_a.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import torch

from quality_noise.config import hash_config, load_config
from quality_noise.data.datasets import load_bundle
from quality_noise.evaluate import evaluate_test
from quality_noise.experiment import expand_experiment, run_key
from quality_noise.models.resnet import build_model
from quality_noise.trainers.weighted import train_weighted
from quality_noise.utils import get_device, make_results_dirs, write_json_atomic


def _find_run_artifact(run_dir: Path, prefix: str, cfg_hash: str, suffix: str = "parquet") -> Path:
    """Resolve a 02-produced artifact for this run (exact hash first, then scan)."""
    exact = run_dir / f"{prefix}_{cfg_hash}.{suffix}"
    if exact.exists():
        return exact
    cands = sorted(run_dir.glob(f"{prefix}_*.{suffix}"))
    if len(cands) == 1:
        return cands[0]
    raise FileNotFoundError(f"cannot resolve {prefix} in {run_dir}: {cands}")


def train_weighted_run(cfg: dict, results: dict, device, tag: str = "", warm_start: bool = False) -> None:
    bundle = load_bundle(cfg)
    key = run_key(cfg)
    cfg_hash = hash_config(cfg)
    base = Path(cfg["dataset"]["name"]) / f"{cfg['noise']['type']}{cfg['noise']['rate']}" / f"seed{cfg['seed']}"
    est_dir = results["estimator"] / base
    out_dir = results["estimator"] / base

    posterior = pd.read_parquet(est_dir / f"posterior_{cfg_hash}.parquet")
    weights = dict(zip(posterior["sample_id"].astype(int), posterior["weight"].astype(float)))

    model = build_model(cfg["model"], bundle.num_classes)
    if warm_start:
        ckpt_path = _find_run_artifact(results["traces"] / base, "model", cfg_hash, suffix="pt")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        print(f"[06] {key}: warm-started from {ckpt_path.name}")
    history = train_weighted(cfg, model, bundle.train_view, weights, device)

    downstream = evaluate_test(model, cfg, device)
    rel = cfg.get("reliability", {})
    summary = {
        **downstream,
        **history.summarize(),
        "run_key": key,
        "config_hash": cfg_hash,
        # Variant provenance: which weight recipe produced this result.
        "weight_mode": rel.get("weight_mode", "calibrated"),
        "weight_power": rel.get("weight_power", 1.0),
        "hard_threshold": rel.get("hard_threshold", None),
        "rate_source": rel.get("rate_source", "config"),
    }
    out_name = f"weighted_summary_{tag}.json" if tag else "weighted_summary.json"
    write_json_atomic(out_dir / out_name, summary)
    print(f"[06] {key} [{tag or 'default'}]: test_acc={downstream['test_accuracy']:.3f} bal_acc={downstream['test_balanced_accuracy']:.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="configs/experiments/gate_a.yaml")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--runs", type=int, nargs="*", default=None, help="restrict to these run indices")
    ap.add_argument("--tag", default="", help="write weighted_summary_<tag>.json (keeps variants side by side)")
    ap.add_argument("--warm-start", action="store_true", help="initialize from the 02 standard-model checkpoint")
    args = ap.parse_args()

    exp = load_config(args.exp)
    runs = expand_experiment(exp)
    if args.runs is not None:
        runs = [runs[i] for i in args.runs]
    device = get_device(args.device)
    results = make_results_dirs({"results_root": exp.get("results_root")})
    for cfg in runs:
        train_weighted_run(cfg, results, device, tag=args.tag, warm_start=args.warm_start)
    print(f"[06] done: {len(runs)} weighted runs (tag={args.tag or 'default'}, warm_start={args.warm_start})")


if __name__ == "__main__":
    main()
