"""09_representation_dependence.py — Phase-9 ablation: does the detectability
conclusion survive a change of representation?

For each run: recompute Q2 (KNN agreement) and the neighborhood-conflict
evidence signal on embeddings from a FROZEN pretrained encoder (resnet18
ImageNet, never trained on the noisy labels), then re-run the global /
matched-quality AUC benchmark. Compares Δ_conf(native) vs Δ_conf(pretrained):
if the confounding gap persists, the Gate-A finding is representation-robust.

Usage:
    python scripts/09_representation_dependence.py --exp configs/experiments/gate_a.yaml
        [--encoder resnet18] [--runs 0 1 2] [--device cuda]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from quality_noise.config import hash_config, load_config
from quality_noise.data.datasets import load_bundle
from quality_noise.detectability.global_auc import global_auc
from quality_noise.detectability.matched_auc import match_quality_pairs, matched_auc
from quality_noise.evidence.forgetting import evidence_forgetting
from quality_noise.evidence.loss import evidence_ema_loss
from quality_noise.experiment import expand_experiment, run_key
from quality_noise.models.pretrained import build_pretrained_encoder, extract_pretrained_embeddings
from quality_noise.quality.neighborhood import compute_knn_agreement
from quality_noise.trainers.standard import _ImageWrapper
from quality_noise.utils import get_device, make_results_dirs, write_json_atomic


def _run_artifacts(cfg: dict, results: dict) -> dict[str, Path]:
    base = results["traces"] / cfg["dataset"]["name"] / f"{cfg['noise']['type']}{cfg['noise']['rate']}" / f"seed{cfg['seed']}"
    return {
        "traces": base / f"traces_{hash_config(cfg)}.parquet",
        "quality": results["quality"] / cfg["dataset"]["name"] / f"{cfg['noise']['type']}{cfg['noise']['rate']}" / f"seed{cfg['seed']}",
        "out": results["detectability"] / cfg["dataset"]["name"] / f"{cfg['noise']['type']}{cfg['noise']['rate']}" / f"seed{cfg['seed']}",
    }


def _find(run_dir: Path, prefix: str, cfg_hash: str, suffix: str = "parquet") -> Path:
    exact = run_dir / f"{prefix}_{cfg_hash}.{suffix}"
    if exact.exists():
        return exact
    cands = sorted(run_dir.glob(f"{prefix}_*.{suffix}"))
    if len(cands) == 1:
        return cands[0]
    raise FileNotFoundError(f"cannot resolve {prefix} in {run_dir}: {cands}")


def _knn_agreement(emb: np.ndarray, y: np.ndarray, k: int, device) -> np.ndarray:
    """GPU KNN label-agreement, mirroring compute_knn_agreement's subsampling
    (20k deterministic index, query all) but with torch matmul + topk."""
    import torch
    import torch.nn.functional as F

    emb = np.asarray(emb, dtype=np.float32)
    n = emb.shape[0]
    k = min(k, n - 1)
    rng = np.random.default_rng(0)
    sub = rng.choice(n, size=min(20000, n), replace=False) if n > 20000 else np.arange(n)

    E = torch.from_numpy(emb).to(device)
    S = torch.from_numpy(emb[sub]).to(device)
    E = F.normalize(E, dim=1)
    S = F.normalize(S, dim=1)
    with torch.no_grad():
        sims = E @ S.T  # (n, n_sub)
        vals, idx = sims.topk(k + 1, dim=1)
    nbrs = sub[idx[:, 1:].cpu().numpy()]
    agree = y[nbrs] == y[:, None]
    return agree.mean(axis=1).astype(np.float32)


def representation_run(cfg: dict, results: dict, device, encoder_name: str) -> dict:
    bundle = load_bundle(cfg)
    key = run_key(cfg)
    cfg_hash = hash_config(cfg)
    arts = _run_artifacts(cfg, results)
    tr_dir = arts["traces"].parent

    traces = pd.read_parquet(_find(tr_dir, "traces", cfg_hash))
    native_emb = np.load(_find(tr_dir, "embeddings", cfg_hash, suffix="npy"))
    quality = pd.read_parquet(_find(arts["quality"], "quality", cfg_hash))
    y_observed = bundle.train_view.y_observed.astype(int)

    # --- frozen pretrained embeddings (vectorized resize + batched forward) ---
    import torch
    import torch.nn.functional as F

    IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    encoder = build_pretrained_encoder(encoder_name).to(device)
    view = bundle.train_view
    n = len(view)
    pre_parts = []
    bs = 256
    with torch.no_grad():
        for i in range(0, n, bs):
            xs = torch.stack(
                [torch.as_tensor(view[j][0], dtype=torch.float32) for j in range(i, min(i + bs, n))]
            ) / 255.0  # (B, 3, 32, 32) in [0, 1]
            xs = xs.to(device)
            xs = F.interpolate(xs, size=(224, 224), mode="bilinear", align_corners=False)
            xs = (xs - IMAGENET_MEAN) / IMAGENET_STD
            pre_parts.append(encoder(xs.to(device)).cpu().numpy())
    pre_emb = np.concatenate(pre_parts, axis=0).astype(np.float32)
    # Align to sample_id ascending order (view order is already sample_id sorted).
    order = np.argsort(np.asarray(bundle.train_view.sample_ids, dtype=np.int64))
    pre_emb = pre_emb[order]
    print(f"[09] {key}: pretrained embeddings {pre_emb.shape}")

    # --- recompute Q2 (KNN agreement) and conflict evidence on pretrained space ---
    q_pre = _knn_agreement(pre_emb, y_observed, 20, device)
    mask = np.array(
        [bool(bundle.eval_view.mask[bundle.eval_view.sample_ids.tolist().index(int(s))]) for s in quality["sample_id"].values]
    )
    mask = mask[: len(q_pre)]

    # Evidence: same EMA-loss + forgetting (trace-based, representation-free),
    # but conflict recomputed in the pretrained space.
    from quality_noise.evidence.signals import _robust_normalize

    conflict_pre = 1.0 - _knn_agreement(pre_emb, y_observed, 20, device)
    E_pre = (
        _robust_normalize(evidence_ema_loss(traces)) * 0.5
        + _robust_normalize(evidence_forgetting(traces)) * 0.3
        + _robust_normalize(conflict_pre) * 0.2
    )
    E_pre = _robust_normalize(E_pre)

    # Native-baseline numbers (from the standard pipeline artifacts).
    native_sum = json.load(open(arts["out"] / "global_summary.json"))
    q_native = quality["knn_agreement"].to_numpy(dtype=np.float64)

    def _bench(E, q):
        auc_g = global_auc(E, mask)
        pairs = match_quality_pairs(E, mask, q, eps=0.1, max_clean_per_noisy=5)
        auc_m = matched_auc(pairs)
        return auc_g, auc_m, auc_g - auc_m

    g_pre, m_pre, d_pre = _bench(E_pre, q_pre)
    g_nat, m_nat, d_nat = native_sum["auc_global"], native_sum["auc_matched"], native_sum["delta_conf"]

    out = {
        "run_key": key,
        "config_hash": cfg_hash,
        "encoder": encoder_name,
        "auc_global_native": g_nat,
        "auc_matched_native": m_nat,
        "delta_conf_native": d_nat,
        "auc_global_pretrained": float(g_pre),
        "auc_matched_pretrained": float(m_pre),
        "delta_conf_pretrained": float(d_pre),
        "delta_conf_gap": float(d_nat - d_pre),  # >0: native gap larger
    }
    out_dir = results["detectability"] / cfg["dataset"]["name"] / f"{cfg['noise']['type']}{cfg['noise']['rate']}" / f"seed{cfg['seed']}"
    write_json_atomic(out_dir / f"representation_{encoder_name}.json", out)
    print(f"[09] {key}: native Δ={d_nat:+.3f} pretrained Δ={d_pre:+.3f} (gap {d_nat - d_pre:+.3f})")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="configs/experiments/gate_a.yaml")
    ap.add_argument("--encoder", default="resnet18")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--runs", type=int, nargs="*", default=None)
    args = ap.parse_args()

    exp = load_config(args.exp)
    runs = expand_experiment(exp)
    if args.runs is not None:
        runs = [runs[i] for i in args.runs]
    device = get_device(args.device)
    results = make_results_dirs({"results_root": exp.get("results_root")})
    for cfg in runs:
        representation_run(cfg, results, device, args.encoder)
    print(f"[09] done: {len(runs)} runs (encoder={args.encoder})")


if __name__ == "__main__":
    main()
