"""01_prepare_data.py — build views + registry + sample manifests for all runs.

Usage:
    python scripts/01_prepare_data.py --exp configs/experiments/gate_a.yaml
    python scripts/01_prepare_data.py --exp ... --runs 0 2   # only selected run indices

Writes (per run):
    results/data/<dataset>/<noise><rate>/seed<s>/data_registry.json
    results/data/<dataset>/<noise><rate>/seed<s>/manifest.parquet
And a global results/registry.json index.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from quality_noise.data.datasets import load_bundle
from quality_noise.data.registry import build_registry_entry, update_registry_index
from quality_noise.experiment import expand_experiment, run_key
from quality_noise.utils import make_results_dirs, write_json_atomic


def prepare_run(cfg: dict, results: dict) -> None:
    bundle = load_bundle(cfg)
    entry = build_registry_entry(cfg, bundle.registry_entry)
    entry["run_key"] = run_key(cfg)
    key = run_key(cfg)
    run_dir = results["data"] / cfg["dataset"]["name"] / f"{cfg['noise']['type']}{cfg['noise']['rate']}" / f"seed{cfg['seed']}"
    run_dir.mkdir(parents=True, exist_ok=True)

    write_json_atomic(run_dir / "data_registry.json", entry)
    update_registry_index(results["data"].parent / "registry.json", entry)

    manifest = pd.DataFrame(
        {
            "sample_id": bundle.eval_view.sample_ids,
            "y_observed": bundle.eval_view.y_observed,
            "y_clean": bundle.eval_view.clean_labels,
            "mask": bundle.eval_view.mask.astype(int),
        }
    )
    manifest.to_parquet(run_dir / "manifest.parquet", index=False)
    print(f"[01] prepared {key}: {bundle.num_noisy}/{bundle.num_train} noisy")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="configs/experiments/gate_a.yaml")
    ap.add_argument("--runs", type=int, nargs="*", default=None, help="restrict to these run indices")
    ap.add_argument("--device", default="auto", help="accepted for uniform CLI; not used here")
    args = ap.parse_args()

    from quality_noise.config import load_config

    exp = load_config(args.exp)
    runs = expand_experiment(exp)
    if args.runs is not None:
        runs = [runs[i] for i in args.runs]
    results = make_results_dirs({"results_root": exp.get("results_root")})
    for cfg in runs:
        prepare_run(cfg, results)
    print(f"[01] done: {len(runs)} runs prepared")


if __name__ == "__main__":
    main()
