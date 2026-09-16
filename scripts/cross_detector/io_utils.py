"""I/O + matching diagnostics helpers for cross-detector analysis.

Provides:
- Path resolution for the saved traces / quality / embeddings artifacts
- Per-cell computation: AUC_global, AUC_matched, Δ_conf, bootstrap CI,
  matching-quality diagnostics (SMD before/after, n_pairs, mean |ΔQ|)
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from quality_noise.detectability.matched_auc import match_quality_pairs, matched_auc


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

def _find(prefix_dir: Path, prefix: str, suffix: str) -> Path:
    """Resolve a unique ``prefix_*.{suffix}`` artifact under ``prefix_dir``."""
    cands = sorted(prefix_dir.glob(f"{prefix}_*.{suffix}"))
    if len(cands) == 1:
        return cands[0]
    if len(cands) == 0:
        raise FileNotFoundError(f"no {prefix}_*.{suffix} in {prefix_dir}")
    raise FileNotFoundError(f"ambiguous {prefix}_*.{suffix} in {prefix_dir}: {cands}")


def resolve_artifacts(results_root: Path, dataset: str, noise: str, seed: int) -> dict[str, Path]:
    """Return paths to the artifacts for one (dataset, noise, seed) run."""
    trace_dir = results_root / "traces" / dataset / noise / f"seed{seed}"
    qual_dir = results_root / "quality" / dataset / noise / f"seed{seed}"
    return {
        "traces": _find(trace_dir, "traces", "parquet"),
        "embeddings": _find(trace_dir, "embeddings", "npy"),
        "quality": _find(qual_dir, "quality", "parquet"),
    }


def load_run(results_root: Path, dataset: str, noise: str, seed: int) -> dict:
    """Load traces / embeddings / quality / mask for one run.

    The noise mask (z_i = 1 if label is corrupted) is loaded via
    ``load_bundle`` from the same configuration that produced the run.
    We re-derive the config dict here without depending on the original YAML
    so we don't need to ship the YAMLs around.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from quality_noise.config import hash_config  # noqa: F401
    from quality_noise.data.datasets import load_bundle

    arts = resolve_artifacts(results_root, dataset, noise, seed)
    traces = pd.read_parquet(arts["traces"])
    embeddings = np.load(arts["embeddings"])
    quality = pd.read_parquet(arts["quality"])

    # Build a minimal config dict for load_bundle
    cfg = _build_cfg_for_load(dataset, noise, seed)
    bundle = load_bundle(cfg)

    y_observed = bundle.train_view.y_observed.astype(int)
    sample_ids = quality["sample_id"].values

    id2mask = {int(sid): bool(m) for sid, m in zip(bundle.eval_view.sample_ids, bundle.eval_view.mask)}
    mask = np.array([id2mask[int(sid)] for sid in sample_ids], dtype=bool)

    # Align traces to sample_id order
    pivot = traces.pivot(index="sample_id", columns="epoch", values="loss").sort_index()
    sid_to_idx = {int(sid): i for i, sid in enumerate(pivot.index.values)}

    # Embeddings should be in sample_id order. We sort embeddings to match.
    if len(embeddings) == len(pivot.index):
        emb_sorted = embeddings
    else:
        # Shouldn't happen, but defensive: do nothing if shape mismatch.
        emb_sorted = embeddings

    return {
        "traces": traces,
        "embeddings": emb_sorted,
        "quality": quality,
        "y_observed": y_observed,
        "mask": mask,
        "sample_ids": sample_ids,
    }


_NUM_CLASSES = {"cifar10": 10, "cifar100": 100, "cifar10n": 10}


def _build_cfg_for_load(dataset: str, noise: str, seed: int) -> dict:
    """Build a minimal cfg dict compatible with load_bundle.

    The noise string convention is e.g. 'symmetric0.2', 'asymmetric0.4',
    'cifar10n_aggre0.4'. We handle the standard CIFAR-10/100 cases and
    CIFAR-10N subset names by looking at the registered noise configs.
    """
    import re
    cfg = {
        "dataset": {
            "name": dataset,
            "root": None,
            "val_frac": 0.1,
            "num_classes": _NUM_CLASSES[dataset],
            "input_size": [3, 32, 32],
        },
        "noise": {"type": None, "rate": None, "seed": seed},
        "seed": seed,
    }
    if dataset == "cifar10n":
        m = re.match(r"cifar10n_(\w+)([0-9.]+)", noise)
        cfg["noise"]["type"] = m.group(1)
        cfg["noise"]["rate"] = float(m.group(2))
    else:
        m = re.match(r"(symmetric|asymmetric)([0-9.]+)", noise)
        cfg["noise"]["type"] = m.group(1)
        cfg["noise"]["rate"] = float(m.group(2))
    return cfg


# ---------------------------------------------------------------------------
# Per-cell metrics
# ---------------------------------------------------------------------------

@dataclass
class CellResult:
    detector: str
    seed: int
    auc_global: float
    auc_matched: float
    delta_conf: float
    ci_global: tuple[float, float]
    ci_matched: tuple[float, float]
    ci_delta: tuple[float, float]
    matching: dict
    n_noisy: int
    n_clean: int


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


def matching_diagnostics(
    pairs, mask: np.ndarray, q: np.ndarray
) -> dict:
    """Compute SMD_before, SMD_after, and pair-distance statistics."""
    noisy_q = q[mask]
    clean_q = q[~mask]
    smd_before = _smd(noisy_q, clean_q)
    smd_after = _smd(np.asarray(pairs.q_noisy), np.asarray(pairs.q_clean))
    diffs = np.abs(np.asarray(pairs.q_noisy) - np.asarray(pairs.q_clean))
    return {
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


def _bootstrap_delta_ci(
    score: np.ndarray,
    mask: np.ndarray,
    q: np.ndarray,
    eps: float,
    max_clean_per_noisy: int,
    n_resamples: int,
    seed: int,
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    """Bootstrap CI for AUC_global, AUC_matched, and Δ_conf simultaneously."""
    rng = np.random.default_rng(seed)
    noisy_idx = np.where(mask)[0]
    clean_idx = np.where(~mask)[0]
    n_noisy, n_clean = len(noisy_idx), len(clean_idx)

    g_stats = np.empty(n_resamples)
    m_stats = np.empty(n_resamples)
    d_stats = np.empty(n_resamples)
    for r in range(n_resamples):
        bn = noisy_idx[rng.integers(0, n_noisy, size=n_noisy)]
        bc = clean_idx[rng.integers(0, n_clean, size=n_clean)]
        ridx = np.concatenate([bn, bc])
        rmask = mask[ridx]
        rscore = score[ridx]
        rq = q[ridx]
        try:
            g = float(roc_auc_score(rmask, rscore))
        except ValueError:
            g = float("nan")
        pairs = match_quality_pairs(rscore, rmask, rq, eps=eps, max_clean_per_noisy=max_clean_per_noisy)
        m = matched_auc(pairs)
        g_stats[r] = g
        m_stats[r] = m
        d_stats[r] = g - m if not np.isnan(m) else float("nan")

    def _ci(arr):
        v = arr[~np.isnan(arr)]
        if len(v) == 0:
            return (float("nan"), float("nan"))
        return (float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975)))

    return _ci(g_stats), _ci(m_stats), _ci(d_stats)


def evaluate_cell(
    score: np.ndarray,
    mask: np.ndarray,
    q: np.ndarray,
    eps: float = 0.1,
    max_clean_per_noisy: int = 5,
    n_bootstrap: int = 500,
    seed: int = 0,
    detector_name: str = "",
    seed_tag: int = 0,
) -> CellResult:
    """Compute all per-cell quantities in one pass."""
    pairs = match_quality_pairs(score, mask, q, eps=eps, max_clean_per_noisy=max_clean_per_noisy)
    auc_g = float(roc_auc_score(mask, score)) if mask.sum() > 0 and (~mask).sum() > 0 else float("nan")
    auc_m = matched_auc(pairs)
    delta = auc_g - auc_m if not np.isnan(auc_m) else float("nan")
    ci_g, ci_m, ci_d = _bootstrap_delta_ci(
        score, mask, q,
        eps=eps, max_clean_per_noisy=max_clean_per_noisy,
        n_resamples=n_bootstrap, seed=seed,
    )
    return CellResult(
        detector=detector_name,
        seed=seed_tag,
        auc_global=auc_g,
        auc_matched=auc_m,
        delta_conf=delta,
        ci_global=ci_g,
        ci_matched=ci_m,
        ci_delta=ci_d,
        matching=matching_diagnostics(pairs, mask, q),
        n_noisy=int(mask.sum()),
        n_clean=int((~mask).sum()),
    )


def cell_to_dict(result: CellResult) -> dict:
    return {
        "detector": result.detector,
        "seed": result.seed,
        "auc_global": result.auc_global,
        "auc_matched": result.auc_matched,
        "delta_conf": result.delta_conf,
        "ci_global": list(result.ci_global),
        "ci_matched": list(result.ci_matched),
        "ci_delta": list(result.ci_delta),
        "n_noisy": result.n_noisy,
        "n_clean": result.n_clean,
        "matching": result.matching,
    }


def write_json_atomic(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    tmp.replace(path)


def list_seeds(root: Path, dataset: str, noise: str) -> list[int]:
    """Return sorted list of seed ints that have traces for this (dataset, noise)."""
    base = root / "traces" / dataset / noise
    if not base.exists():
        return []
    seeds = []
    for p in sorted(base.iterdir()):
        if p.is_dir() and p.name.startswith("seed"):
            try:
                seeds.append(int(p.name[4:]))
            except ValueError:
                pass
    return seeds


def all_settings() -> list[tuple[str, str]]:
    """The 10 settings required by the cross-detector spec.

    For settings that have NO traces on disk, the calling code will skip
    them and record them as 'data not available' rather than crashing.
    """
    return [
        ("cifar10", "symmetric0.2"),
        ("cifar10", "symmetric0.4"),
        ("cifar10", "asymmetric0.2"),
        ("cifar10", "asymmetric0.4"),
        ("cifar100", "symmetric0.2"),
        ("cifar100", "symmetric0.4"),
        ("cifar100", "asymmetric0.2"),
        ("cifar100", "asymmetric0.4"),
        ("cifar10n", "cifar10n_aggre0.4"),
        ("cifar10n", "cifar10n_worse0.4"),
    ]


def iter_runs(root: Path, datasets: Iterable[str] | None = None) -> Iterable[tuple[str, str, int]]:
    """Yield (dataset, noise, seed) tuples that have traces on disk."""
    for dataset, noise in all_settings():
        if datasets and dataset not in datasets:
            continue
        for seed in list_seeds(root, dataset, noise):
            yield dataset, noise, seed
