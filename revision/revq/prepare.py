"""Assemble every per-run signal the revision needs, in one place.

Produces, for one (dataset, noise, seed) run, a dictionary containing:

* the evaluation-only noise ``mask`` (z_i) and observed labels,
* 7 detector scores (the paper's 6 + Confident Learning), all oriented
  higher = more suspicious,
* several quality covariates, all on the same run subset and sample order:
    ``knn_agreement``  the paper's label-dependent Q (post-treatment; reference)
    ``proto_margin``   the paper's prototype-margin Q (label-dependent)
    ``dino_density``   class-agnostic density in frozen DINOv2  (label-free)
    ``native_density`` class-agnostic density in the run's own ResNet features
                       (noisy-label trained; contrast only)
    ``random``         uniform control (should give Delta_Q ~ 0)

All heavy per-run work is cached to ``<cache>/prepared/<dataset>/<noise>/seed<seed>.npz``
so re-running the pipeline (e.g. with more bootstrap resamples) is cheap.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/root/quality_noise")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "revision"))

from revq.confident_learning import compute_cl  # noqa: E402
from revq.independent_q import compute_independent_q, unit_affine  # noqa: E402

NUM_CLASSES = {"cifar10": 10, "cifar100": 100, "cifar10n": 10, "cifar100n": 100}
# Human-annotation datasets reuse the base CIFAR image pool for DINO features.
IMAGE_DATASET = {"cifar10": "cifar10", "cifar100": "cifar100",
                 "cifar10n": "cifar10", "cifar100n": "cifar100"}
DINO_K = 20
NATIVE_K = 20

DETECTOR_NAMES = ["ema_loss", "confidence", "aum", "forgetting",
                  "neighbor", "confident_learning", "combined", "combined_cl"]

QUALITY_NAMES = ["knn_agreement", "proto_margin", "dino_density",
                 "native_density", "random",
                 # second/third label-free families (NEW-2)
                 "dino_knndist", "dino_augcons"]


def _build_cfg(dataset: str, noise: str, seed: int) -> dict:
    """Minimal config dict accepted by ``quality_noise.data.datasets.load_bundle``."""
    cfg = {
        "dataset": {"name": dataset, "root": None, "val_frac": 0.1,
                    "num_classes": NUM_CLASSES[dataset], "input_size": [3, 32, 32]},
        "noise": {"type": None, "rate": None, "seed": seed},
        "seed": seed,
    }
    if dataset in ("cifar10n", "cifar100n"):
        m = re.match(rf"{dataset}_(\w+?)([0-9.]+)$", noise)
        cfg["noise"] = {"type": m.group(1), "rate": float(m.group(2)), "seed": seed}
    else:
        m = re.match(r"(symmetric|asymmetric)([0-9.]+)$", noise)
        cfg["noise"] = {"type": m.group(1), "rate": float(m.group(2)), "seed": seed}
    return cfg


def _robust_normalize(x: np.ndarray) -> np.ndarray:
    from quality_noise.evidence.signals import _robust_normalize as rn
    return np.asarray(rn(x), dtype=np.float64)


def detectors_for_run(traces: pd.DataFrame, embeddings: np.ndarray,
                      y_observed: np.ndarray, native_features: np.ndarray | None,
                      num_classes: int, seed: int) -> tuple[dict, dict]:
    """All 7 detectors for one run. Returns (detectors, extras)."""
    from scripts.cross_detector.detectors import (  # noqa: E402
        detector_d1_ema_loss, detector_d2_low_confidence, detector_d3_aum,
        detector_d4_forgetting, detector_d5_neighborhood_conflict,
        detector_d6_combined,
    )

    d = {}
    d["ema_loss"] = np.asarray(detector_d1_ema_loss(traces), dtype=np.float64)
    d["confidence"] = np.asarray(detector_d2_low_confidence(traces, last_k=20), dtype=np.float64)
    d["aum"] = np.asarray(detector_d3_aum(traces), dtype=np.float64)
    d["forgetting"] = np.asarray(detector_d4_forgetting(traces), dtype=np.float64)
    d["neighbor"] = np.asarray(detector_d5_neighborhood_conflict(
        embeddings, y_observed, k=20), dtype=np.float64)
    combined = np.asarray(detector_d6_combined(traces, embeddings, y_observed,
                                               conflict_k=20), dtype=np.float64)

    extras: dict = {}
    feats = embeddings if native_features is None else native_features
    if feats is not None:
        cl = compute_cl(feats, y_observed, num_classes, n_folds=5, seed=seed)
        selfconf = np.asarray(cl["cl_selfconf"], dtype=np.float64)
        d["confident_learning"] = _robust_normalize(selfconf)
        extras["cl_rate"] = np.asarray(cl["cl_rate"], dtype=np.float64)
        extras["cl_oof_probs"] = cl["oof_probs"]
        extras["cl_fold_acc"] = float((cl["oof_probs"].argmax(axis=1) == y_observed).mean())
    else:
        d["confident_learning"] = np.full(len(combined), np.nan)

    # ``combined`` MUST stay the paper's 3-signal recipe
    # (0.5 loss + 0.3 forgetting + 0.2 conflict, robust-normalised).  Verified to
    # reproduce the archived cross-detector JSONs to 4 decimals
    # (C100-S20 seed0: 0.7030).  Available as ``assemble_evidence`` directly.
    d["combined"] = combined

    # ``combined_cl`` is the revision's extension: the same recipe with the
    # Confident-Learning family given a share, so the CL signal is visible in a
    # single fused detector as well.  Reported alongside, never in place of,
    # ``combined``.
    if np.isfinite(d["confident_learning"]).all():
        d["combined_cl"] = _robust_normalize(
            0.40 * d["ema_loss"] + 0.24 * d["forgetting"] + 0.16 * d["neighbor"]
            + 0.20 * d["confident_learning"])
        extras["combined_cl_includes_cl"] = True
    else:
        d["combined_cl"] = combined
        extras["combined_cl_includes_cl"] = False
    return d, extras


def prepare_run(dataset: str, noise: str, seed: int,
                results_root: Path = ROOT / "results",
                emb_root: Path = ROOT / "results" / "revision" / "embeddings") -> dict:
    """Load one run and compute detectors + all quality covariates."""
    from quality_noise.data.datasets import load_bundle
    from scripts.cross_detector.io_utils import resolve_artifacts

    arts = resolve_artifacts(results_root, dataset, noise, seed)
    traces = pd.read_parquet(arts["traces"])
    embeddings = np.load(arts["embeddings"]).astype(np.float32)
    quality = pd.read_parquet(arts["quality"]).sort_values("sample_id").reset_index(drop=True)
    sids = quality["sample_id"].to_numpy().astype(int)
    n = len(sids)

    bundle = load_bundle(_build_cfg(dataset, noise, seed))
    id2mask = {int(s): bool(m) for s, m in zip(bundle.eval_view.sample_ids, bundle.eval_view.mask)}
    id2y = {int(s): int(v) for s, v in zip(bundle.eval_view.sample_ids, bundle.eval_view.y_observed)}
    mask = np.array([id2mask[int(s)] for s in sids], dtype=bool)
    y_observed = np.array([id2y[int(s)] for s in sids], dtype=int)

    # --- native ResNet features for this run (recomputed; used for CL + Q^native)
    native_path = emb_root / "native" / dataset / noise / f"seed{seed}.npy"
    native = np.load(native_path).astype(np.float32) if native_path.exists() else None
    if native is not None and len(native) != n:
        native = None

    det, extras = detectors_for_run(traces, embeddings, y_observed, native,
                                    NUM_CLASSES[dataset], seed)

    # --- quality covariates -------------------------------------------------
    q: dict[str, np.ndarray] = {}
    q["knn_agreement"] = quality["knn_agreement"].to_numpy(dtype=np.float64)
    if "margin" in quality.columns:
        q["proto_margin"] = quality["margin"].to_numpy(dtype=np.float64)

    dino_all = np.load(emb_root / "dino" / dataset / f"seed{seed}.npy").astype(np.float32)
    q["dino_density"] = compute_independent_q(dino_all[sids], k=DINO_K)["q"]

    # --- second and third label-free families (see run_new2_label_free_q.py) ---
    # Typicality: negative mean distance to the K nearest neighbours.  Same
    # embedding as dino_density but a different functional (a distance rather
    # than a similarity), so it is not a monotone rescaling of it.
    q["dino_knndist"] = compute_independent_q(dino_all[sids], k=DINO_K,
                                              mode="dist")["q"]
    # Stability: mean cosine agreement with deterministic augmented views.
    # Structurally different from any density measure -- it asks whether the
    # representation of a sample survives a label-preserving perturbation.
    aug_dir = emb_root / "dino" / dataset
    aug_files = sorted(aug_dir.glob("aug*.npy"))
    if aug_files:
        base = dino_all[sids]
        base = base / np.maximum(np.linalg.norm(base, axis=1, keepdims=True), 1e-12)
        sims = []
        for af in aug_files:
            v = np.load(af).astype(np.float32)[sids]
            v = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)
            sims.append((base * v).sum(axis=1))
        q["dino_augcons"] = unit_affine(np.mean(sims, axis=0))
        q["dino_augcons_sd"] = unit_affine(np.std(sims, axis=0))

    # Full-corpus density: neighbours may be *any* of the 50k CIFAR train images,
    # not only this run's 45k subset (precomputed once per dataset).
    fp = emb_root / "dino" / dataset / "density_fullpool.npy"
    if fp.exists():
        q["dino_density_fullpool"] = np.load(fp).astype(np.float64)[sids]

    if native is not None:
        q["native_density"] = compute_independent_q(native, k=NATIVE_K)["q"]

    rng = np.random.default_rng(seed + 12345)
    q["random"] = rng.random(n)

    out = {k: v[:n] for k, v in det.items()}
    out.update({k: v[:n] for k, v in q.items()})
    return {
        "dataset": dataset, "noise": noise, "seed": seed, "n": n,
        "mask": mask, "y_observed": y_observed, "sample_ids": sids,
        "signals": out, "extras": {k: v for k, v in extras.items() if k != "cl_oof_probs"},
    }


def extra_quality(r: dict, emb_root: Path) -> dict[str, np.ndarray]:
    """Label-free quality covariates computed from an already-prepared run.

    Kept separate from ``prepare_run`` so adding a new label-free family never
    requires recomputing detectors (the expensive part).
    """
    dataset, seed = r["dataset"], r["seed"]
    sids = r["sample_ids"]
    dino_all = np.load(emb_root / "dino" / dataset / f"seed{seed}.npy").astype(np.float32)
    out: dict[str, np.ndarray] = {}
    out["dino_knndist"] = compute_independent_q(dino_all[sids], k=DINO_K, mode="dist")["q"]
    base = dino_all[sids]
    base = base / np.maximum(np.linalg.norm(base, axis=1, keepdims=True), 1e-12)
    aug_files = sorted((emb_root / "dino" / dataset).glob("aug*.npy"))
    if aug_files:
        sims = []
        for af in aug_files:
            v = np.load(af).astype(np.float32)[sids]
            v = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)
            sims.append((base * v).sum(axis=1))
        out["dino_augcons"] = unit_affine(np.mean(sims, axis=0))
        out["dino_augcons_sd"] = unit_affine(np.std(sims, axis=0))
    return out


def add_extra_quality(dataset: str, noise: str, seed: int,
                      cache: Path = ROOT / "results" / "revision" / "prepared",
                      emb_root: Path = ROOT / "results" / "revision" / "embeddings") -> Path:
    """Merge the extra label-free covariates into an existing cache entry."""
    out = cache / dataset / noise / f"seed{seed}.npz"
    if not out.exists():
        raise FileNotFoundError(out)
    z = np.load(out, allow_pickle=False)
    flat = {k: z[k] for k in z.files}
    have = {k for k in z.files if k.startswith("sig__")}
    r = {"dataset": dataset, "seed": seed,
         "sample_ids": z["sample_ids"].astype(int)}
    extra = extra_quality(r, emb_root)
    added = []
    for k, v in extra.items():
        key = f"sig__{k}"
        if key not in have:
            flat[key] = np.asarray(v, dtype=np.float64)
            added.append(k)
    if not added:
        return out
    tmp = out.with_name(out.name + ".part.npz")
    np.savez_compressed(tmp, **flat)
    tmp.replace(out)
    return out


def prepare_and_cache(dataset: str, noise: str, seed: int,
                      cache: Path = ROOT / "results" / "revision" / "prepared",
                      overwrite: bool = False) -> Path:
    """Compute a run once and cache it to ``.npz``."""
    out = cache / dataset / noise / f"seed{seed}.npz"
    if out.exists() and not overwrite:
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    r = prepare_run(dataset, noise, seed)
    flat = {f"sig__{k}": v for k, v in r["signals"].items()}
    flat.update({"mask": r["mask"], "y_observed": r["y_observed"],
                 "sample_ids": r["sample_ids"]})
    for k, v in r["extras"].items():
        if isinstance(v, (int, float, bool, np.floating, np.integer, np.bool_)):
            flat[f"extra__{k}"] = np.asarray(v)
    # np.savez_compressed appends '.npz' to a path that lacks it, so give the
    # temp path the final extension explicitly and then atomically swap it in.
    tmp = out.with_name(out.name + ".part.npz")
    np.savez_compressed(tmp, **flat)
    tmp.replace(out)
    return out


def load_cached(dataset: str, noise: str, seed: int,
                cache: Path = ROOT / "results" / "revision" / "prepared",
                retries: int = 40, retry_s: float = 5.0) -> dict:
    """Load a cached run.

    Retries while the file is missing: cache entries are written atomically via
    a temp file + rename, so a concurrent reader can briefly observe no file at
    all.  Blocking here is far cheaper than failing a whole evaluation batch.
    """
    import time

    path = cache / dataset / noise / f"seed{seed}.npz"
    for attempt in range(retries):
        if path.exists():
            try:
                z = np.load(path, allow_pickle=False)
                signals = {k[len("sig__"):]: z[k].astype(np.float64)
                           for k in z.files if k.startswith("sig__")}
                extras = {k[len("extra__"):]: float(z[k])
                          for k in z.files if k.startswith("extra__")}
                return {"dataset": dataset, "noise": noise, "seed": seed,
                        "mask": z["mask"].astype(bool),
                        "y_observed": z["y_observed"].astype(int),
                        "sample_ids": z["sample_ids"].astype(int),
                        "signals": signals, "extras": extras,
                        "n": int(len(z["mask"]))}
            except (EOFError, OSError, ValueError):
                pass  # mid-write or truncated; retry
        if attempt < retries - 1:
            time.sleep(retry_s)
    raise FileNotFoundError(f"cached run unavailable after {retries} attempts: {path}")
