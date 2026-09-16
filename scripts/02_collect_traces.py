"""02_collect_traces.py — train ResNet-18 with standard CE and record traces.

Usage:
    python scripts/02_collect_traces.py --exp configs/experiments/gate_a.yaml

Heavy training step: intended to run on the GPU server. For each run, trains
the standard model and writes per-sample x per-epoch traces to
    results/traces/<dataset>/<noise><rate>/seed<s>/traces_<hash>.parquet
plus a dynamics.parquet of per-sample aggregates.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from quality_noise.config import hash_config, load_config
from quality_noise.data.datasets import load_bundle
from quality_noise.experiment import expand_experiment, run_key
from quality_noise.models.features import extract_embeddings
from quality_noise.models.resnet import build_model
from quality_noise.traces.dynamics import compute_all_dynamics
from quality_noise.traces.recorder import TraceRecorder
from quality_noise.trainers.standard import _ImageWrapper, default_eval_transform, train_standard
from quality_noise.utils import get_device, make_results_dirs, write_json_atomic


def noise_dirname(noise_type: str, noise_rate) -> str:
    """Directory name for a noise setting, e.g. ``symmetric0.2``, ``cifar10n_aggre0.4``.

    Real human-annotation datasets already carry the setting in the type
    (``cifar10n_aggre``, ``cifar100n_human``), so the nominal rate is simply
    appended, which keeps the pre-existing result layout intact.
    """
    return f"{noise_type}{noise_rate}"


def collect_trace(cfg: dict, results: dict, device) -> None:
    bundle = load_bundle(cfg)
    key = run_key(cfg)
    cfg_hash = hash_config(cfg)
    run_dir = results["traces"] / cfg["dataset"]["name"] / noise_dirname(
        cfg["noise"]["type"], cfg["noise"]["rate"]) / f"seed{cfg['seed']}"
    run_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(cfg["model"], bundle.num_classes)
    recorder = TraceRecorder(run_dir, cfg_hash)
    history = train_standard(cfg, model, bundle.train_view, device, callbacks=[recorder], progress=False)

    trace_path = recorder.finalize()

    # Held-out test evaluation: the standard-trained model IS the CE baseline,
    # so evaluate it here and save the numbers (Gate B needs CE accuracy).
    from quality_noise.evaluate import evaluate_test

    downstream = evaluate_test(model, cfg, device)
    write_json_atomic(run_dir / "ce_summary.json", {"method": "ce", **downstream})

    # Save the trained model and final-epoch embeddings for quality/detectability.
    torch.save({"state_dict": model.state_dict(), "cfg_hash": cfg_hash}, run_dir / f"model_{cfg_hash}.pt")
    eval_ds = _ImageWrapper(bundle.train_view, default_eval_transform(cfg["dataset"]))
    emb_loader = DataLoader(eval_ds, batch_size=cfg["model"]["batch_size"], shuffle=False, num_workers=0)
    embeddings = extract_embeddings(model, emb_loader, device, view=bundle.train_view)
    np.save(run_dir / f"embeddings_{cfg_hash}.npy", embeddings)
    print(f"[02] saved embeddings {embeddings.shape} for {key}")

    # Per-sample dynamics + final embeddings (written once per run).
    df = pd.read_parquet(trace_path)
    losses = df.pivot(index="sample_id", columns="epoch", values="loss").sort_index().values
    conf = df.pivot(index="sample_id", columns="epoch", values="confidence").sort_index().values
    margin = df.pivot(index="sample_id", columns="epoch", values="margin").sort_index().values
    preds = df.pivot(index="sample_id", columns="epoch", values="prediction").sort_index().values
    correct = df.pivot(index="sample_id", columns="epoch", values="correct_to_observed").sort_index().values.astype(bool)
    sample_ids = df["sample_id"].unique()
    sample_ids.sort()

    dyn = compute_all_dynamics(losses, conf, margin, preds, correct, sample_ids)
    pd.DataFrame(dyn).to_parquet(run_dir / f"dynamics_{cfg_hash}.parquet", index=False)

    write_json_atomic(run_dir / "history.json", {"summary": history.summarize()})
    print(f"[02] {key}: {history.summarize()}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="configs/experiments/gate_a.yaml")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--runs", type=int, nargs="*", default=None, help="restrict to these run indices")
    args = ap.parse_args()

    exp = load_config(args.exp)
    runs = expand_experiment(exp)
    if args.runs is not None:
        runs = [runs[i] for i in args.runs]
    device = get_device(args.device)
    results = make_results_dirs({"results_root": exp.get("results_root")})
    for cfg in runs:
        collect_trace(cfg, results, device)
    print(f"[02] done: {len(runs)} traces collected")


if __name__ == "__main__":
    main()
