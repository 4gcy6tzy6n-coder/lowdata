"""Experiment config expansion: load an experiment file and expand the
noise x seeds sweep into a list of fully-resolved per-run configs."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import PROJECT_ROOT, load_config, resolve_config


def _load_referenced(entry: dict) -> dict:
    """Load a config referenced by `file`, merged with inline overrides."""
    ref = load_config(entry["file"])
    overrides = {k: v for k, v in entry.items() if k != "file"}
    merged = {**ref, **overrides}
    return merged


def expand_experiment(exp_cfg: dict, base_dir: Path = PROJECT_ROOT) -> list[dict[str, Any]]:
    """Expand an experiment config into per-run resolved configs.

    Each run config is the canonical defaults merged with the dataset, model,
    top-level analysis settings, one noise spec, and one seed. ``base_dir`` is
    used to resolve relative `file:` paths.
    """
    dataset = _load_referenced(exp_cfg["dataset"]) if isinstance(exp_cfg["dataset"], dict) and "file" in exp_cfg["dataset"] else exp_cfg["dataset"]
    model = _load_referenced(exp_cfg["model"]) if isinstance(exp_cfg["model"], dict) and "file" in exp_cfg["model"] else exp_cfg["model"]
    seeds = [int(s) for s in exp_cfg.get("seeds", [0])]

    top_keys = [
        "evidence",
        "quality",
        "matched_auc",
        "bootstrap",
        "reliability",
        "mixture",
        "results_root",
    ]
    top = {k: exp_cfg[k] for k in top_keys if k in exp_cfg}
    # "run" (default): each seed gets a fresh noise realization AND fresh
    # training -> a full replication. "fixed": noise seeded only by the noise
    # config (same mask across seeds, only training stochasticity varies).
    noise_seed_mode = exp_cfg.get("noise_seed", "run")

    runs: list[dict[str, Any]] = []
    for noise_spec in exp_cfg.get("noise", []):
        noise = _load_referenced(noise_spec) if isinstance(noise_spec, dict) and "file" in noise_spec else noise_spec
        for seed in seeds:
            noise_run = dict(noise)
            if noise_seed_mode == "run":
                noise_run["seed"] = seed
            run = resolve_config(
                {
                    "dataset": dataset,
                    "noise": noise_run,
                    "model": model,
                    "seed": seed,
                    "stage": exp_cfg.get("stage", "trace"),
                    **top,
                }
            )
            runs.append(run)
    return runs


def run_key(cfg: dict) -> str:
    """A short human-readable key for a run, e.g. cifar10_sym40_seed0."""
    ds = cfg["dataset"]["name"]
    nt = cfg["noise"]["type"]
    rate = str(cfg["noise"]["rate"]).replace(".", "")
    seed = cfg["seed"]
    return f"{ds}_{nt}{rate}_seed{seed}"
