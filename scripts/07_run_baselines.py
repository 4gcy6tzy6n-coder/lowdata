"""07_run_baselines.py — run strong noisy-label baselines (Phase 8 / Sprint 7).

Baselines (configured in the experiment file under ``baselines``):
    ce, small_loss, coteaching, coteaching_plus, jocor, elr, dividemix
Each trains on the observed labels and is evaluated on the held-out test set.
Results are written per run to results/baselines/.

Usage:
    python scripts/07_run_baselines.py --exp configs/experiments/gate_a.yaml --methods elr,jocor
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from quality_noise.baselines.coteaching import CoTeachingTrainer
from quality_noise.baselines.coteaching_plus import CoTeachingPlusTrainer
from quality_noise.baselines.dividemix import DivideMixTrainer
from quality_noise.baselines.elr import ELRTrainer
from quality_noise.baselines.jocor import JoCoRTrainer
from quality_noise.baselines.small_loss import SmallLossTrainer
from quality_noise.config import hash_config, load_config
from quality_noise.data.datasets import load_bundle
from quality_noise.evaluate import evaluate_test
from quality_noise.experiment import expand_experiment, run_key
from quality_noise.models.resnet import build_model
from quality_noise.trainers.standard import train_standard
from quality_noise.utils import get_device, make_results_dirs, write_json_atomic

TRAINERS = {
    "small_loss": SmallLossTrainer,
    "coteaching": CoTeachingTrainer,
    "coteaching_plus": CoTeachingPlusTrainer,
    "jocor": JoCoRTrainer,
    "elr": ELRTrainer,
    "dividemix": DivideMixTrainer,
}


def _methods_from_cfg(exp: dict, only: str | None) -> list[str]:
    # "ce" is trained as part of 02_collect_traces (standard CE training); it is
    # NOT re-run here. Only the dedicated noisy-label baselines are run.
    default = ["small_loss", "coteaching", "coteaching_plus", "jocor", "elr", "dividemix"]
    methods = exp.get("baselines", {}).get("methods", default)
    methods = [m for m in methods if m != "ce"]
    if only:
        methods = [m for m in methods if m in only.split(",")]
    return methods


def run_baseline(cfg: dict, method: str, results: dict, device) -> None:
    bundle = load_bundle(cfg)
    key = run_key(cfg)
    cfg_hash = hash_config(cfg)
    run_dir = results["baselines"] / cfg["dataset"]["name"] / f"{cfg['noise']['type']}{cfg['noise']['rate']}" / f"seed{cfg['seed']}" / method
    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_dir / f"summary_{cfg_hash}.json"
    if out_path.exists():
        print(f"[07] {key} {method}: cached")
        return

    if method == "ce":
        model = build_model(cfg["model"], bundle.num_classes)
        history = train_standard(cfg, model, bundle.train_view, device)
    else:
        trainer_cls = TRAINERS[method]
        trainer = trainer_cls(cfg, bundle.train_view, device)
        history = trainer.run()
        model = trainer.model

    downstream = evaluate_test(model, cfg, device)
    summary = {**downstream, **history.summarize(), "method": method, "run_key": key, "config_hash": cfg_hash}
    write_json_atomic(out_path, summary)
    print(f"[07] {key} {method}: test_acc={downstream['test_accuracy']:.3f} bal_acc={downstream['test_balanced_accuracy']:.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="configs/experiments/gate_a.yaml")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--methods", default=None, help="comma-separated subset, e.g. elr,jocor")
    ap.add_argument("--runs", type=int, nargs="*", default=None, help="restrict to these run indices")
    args = ap.parse_args()

    exp = load_config(args.exp)
    methods = _methods_from_cfg(exp, args.methods)
    runs = expand_experiment(exp)
    if args.runs is not None:
        runs = [runs[i] for i in args.runs]
    device = get_device(args.device)
    results = make_results_dirs({"results_root": exp.get("results_root")})
    for cfg in runs:
        for method in methods:
            run_baseline(cfg, method, results, device)
    print(f"[07] done: {len(runs)} runs x {len(methods)} methods")


if __name__ == "__main__":
    main()
