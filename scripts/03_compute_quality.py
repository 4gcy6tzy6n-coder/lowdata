"""03_compute_quality.py — compute Q1/Q2/Q3 quality signals per run.

Usage:
    python scripts/03_compute_quality.py --exp configs/experiments/gate_a.yaml

Reads traces + final-epoch embeddings written by 02, computes:
    Q1 prototype margin (high-confidence anchors, leave-one-out)
    Q2 KNN neighborhood agreement
    Q3 representation stability (optional; requires model + augment forward)
and writes results/quality/<dataset>/<noise><rate>/seed<s>/quality_<hash>.parquet.

None of the quality signals use the noise mask or clean labels.
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
from quality_noise.quality.neighborhood import compute_knn_agreement
from quality_noise.quality.prototype_margin import compute_margin, select_high_confidence_anchors
from quality_noise.quality.stability import compute_representation_stability
from quality_noise.trainers.standard import _ImageWrapper, default_train_transform, default_eval_transform
from quality_noise.utils import get_device, make_results_dirs


def compute_quality(cfg: dict, results: dict, device) -> None:
    bundle = load_bundle(cfg)
    key = run_key(cfg)
    cfg_hash = hash_config(cfg)
    run_dir = results["traces"] / cfg["dataset"]["name"] / f"{cfg['noise']['type']}{cfg['noise']['rate']}" / f"seed{cfg['seed']}"
    out_dir = results["quality"] / cfg["dataset"]["name"] / f"{cfg['noise']['type']}{cfg['noise']['rate']}" / f"seed{cfg['seed']}"
    out_dir.mkdir(parents=True, exist_ok=True)

    trace_path = run_dir / f"traces_{cfg_hash}.parquet"
    emb_path = run_dir / f"embeddings_{cfg_hash}.npy"
    if not trace_path.exists() or not emb_path.exists():
        raise FileNotFoundError(f"Run {key} has no traces/embeddings yet (run scripts/02 first).")

    traces = pd.read_parquet(trace_path)
    embeddings = np.load(emb_path)
    y_observed = bundle.train_view.y_observed.astype(int)

    # Anchor selection uses final-epoch observed-label confidence from the trace.
    final = traces[traces["epoch"] == traces["epoch"].max()]
    final = final.sort_values("sample_id")
    final_confidence = final["confidence"].to_numpy()

    q_cfg = cfg.get("quality", {})
    anchors = select_high_confidence_anchors(final_confidence, y_observed, float(q_cfg.get("anchor_quantile", 0.8)))
    q1 = compute_margin(embeddings, y_observed, anchors, bundle.num_classes, use_leave_one_out=True)
    q2 = compute_knn_agreement(embeddings, y_observed, k=int(q_cfg.get("knn_k", 20)))

    df = pd.DataFrame(
        {
            "sample_id": bundle.train_view.sample_ids.astype(int),
            "margin": q1,
            "knn_agreement": q2,
        }
    )

    # Q3 stability (optional; needs a model forward pass with augmentations).
    # Disable with --no-stability or stability_views: 0 in config.
    if cfg["quality"].get("stability_views", 2) > 0:
        model = build_model(cfg["model"], bundle.num_classes)
        ckpt = torch.load(run_dir / f"model_{cfg_hash}.pt", map_location="cpu")
        model.load_state_dict(ckpt["state_dict"])
        transform = default_train_transform(cfg["dataset"])
        stability = compute_representation_stability(
            model,
            bundle.train_view,
            device,
            transform,
            n_views=int(q_cfg.get("stability_views", 2)),
        )
        df["stability"] = stability

    df.to_parquet(out_dir / f"quality_{cfg_hash}.parquet", index=False)
    print(f"[03] {key}: Q1/Q2 computed over {len(df)} samples")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="configs/experiments/gate_a.yaml")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--no-stability", action="store_true", help="skip Q3 (faster)")
    ap.add_argument("--runs", type=int, nargs="*", default=None, help="restrict to these run indices")
    args = ap.parse_args()

    exp = load_config(args.exp)
    if args.no_stability:
        exp["quality"] = {**exp.get("quality", {}), "stability_views": 0}
    runs = expand_experiment(exp)
    if args.runs is not None:
        runs = [runs[i] for i in args.runs]
    device = get_device(args.device)
    results = make_results_dirs({"results_root": exp.get("results_root")})
    for cfg in runs:
        compute_quality(cfg, results, device)
    print(f"[03] done: {len(runs)} quality runs")


if __name__ == "__main__":
    main()
