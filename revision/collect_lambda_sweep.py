"""P1-7 collector — turn the lambda sweep into the four requested metrics.

For every (setting, lambda, seed) it reads the fitted posterior (weights) and the
downstream test accuracy, then reports:

    test_accuracy
    noise_removal_rate        = P(w = 0 | sample is genuinely noisy)
    clean_retention_rate      = P(w > 0 | sample is genuinely clean)
    hard_clean_false_removal  = P(w = 0 | clean AND in the hardest quality decile)

The mask (z_i) is evaluation-only: it is read here for reporting, never used to
build the weights.

Output -> results/revision/governance_sensitivity/lambda_sweep.csv
          results/revision/governance_sensitivity/lambda_sweep_summary.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path("/root/quality_noise")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "revision"))

from quality_noise.config import hash_config, load_config  # noqa: E402
from quality_noise.data.datasets import load_bundle  # noqa: E402
from quality_noise.experiment import expand_experiment  # noqa: E402

OUT = ROOT / "results" / "revision" / "governance_sensitivity"
LAMBDAS = [0.0, 0.25, 0.5, 0.75, 1.0]
QUALITY_FOR_HARD = "knn_agreement"


def _build_cfg_for_load(dataset: str, noise_type: str, noise_rate, seed: int) -> dict:
    import re
    nc = {"cifar10": 10, "cifar100": 100, "cifar10n": 10, "cifar100n": 100}[dataset]
    cfg = {"dataset": {"name": dataset, "root": None, "val_frac": 0.1,
                       "num_classes": nc, "input_size": [3, 32, 32]},
           "noise": {"type": noise_type, "rate": noise_rate, "seed": seed},
           "seed": seed}
    if dataset in ("cifar10n", "cifar100n"):
        m = re.match(rf"{dataset}_(\w+?)([0-9.]+)$", f"{noise_type}{noise_rate}")
        cfg["noise"] = {"type": m.group(1), "rate": float(m.group(2)), "seed": seed}
    return cfg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--settings", type=str, default="c10_s20,c10_a40,c100_s20,c100_s40,c100_a20")
    ap.add_argument("--lambdas", type=str, default=",".join(str(x) for x in LAMBDAS))
    ap.add_argument("--cutoff", type=float, default=0.5,
                    help="weight below this counts as 'removed'")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    settings = [s.strip() for s in args.settings.split(",") if s.strip()]
    lambdas = [float(x) for x in args.lambdas.split(",")]
    rows = []

    for s in settings:
        for lam in lambdas:
            cfg_path = ROOT / "configs" / "experiments" / "lambda_sweep" / f"{s}_lam{lam:.2f}.yaml"
            if not cfg_path.exists():
                print(f"skip {s} lam={lam}: no config")
                continue
            exp = load_config(str(cfg_path))
            base = Path(str(cfg_path))
            base_dir = ROOT / "results" / "estimator"
            for cfg in expand_experiment(exp):
                key = f"{cfg['dataset']['name']}/{cfg['noise']['type']}{cfg['noise']['rate']}/seed{cfg['seed']}"
                run_dir = base_dir / cfg["dataset"]["name"] / \
                    f"{cfg['noise']['type']}{cfg['noise']['rate']}" / f"seed{cfg['seed']}"
                cfg_hash = hash_config(cfg)
                post_path = run_dir / f"posterior_{cfg_hash}.parquet"
                tag = f"{s}_lam{lam:.2f}"
                # The frozen sweep points already exist under archival tags whose
                # config hash matches ours exactly (verified: our generated
                # lambda=0.50 config reproduces the paper's hard50 hash).  Accept
                # those, and record which tag supplied the number so provenance
                # stays explicit.
                wsum_path = run_dir / f"weighted_summary_{tag}.json"
                wsum_tag = tag
                if not wsum_path.exists():
                    for legacy in ("final_hard50",):
                        cand = run_dir / f"weighted_summary_{legacy}.json"
                        if cand.exists():
                            wsum_path, wsum_tag = cand, legacy
                            break
                if not post_path.exists() or not wsum_path.exists():
                    continue

                post = pd.read_parquet(post_path).sort_values("sample_id")
                w = post["weight"].to_numpy(dtype=np.float64)
                sids = post["sample_id"].to_numpy().astype(int)
                wsum = json.load(open(wsum_path))

                # evaluation-only mask + quality (never an input to the weights)
                b = load_bundle(_build_cfg_for_load(
                    cfg["dataset"]["name"], cfg["noise"]["type"],
                    cfg["noise"]["rate"], cfg["seed"]))
                id2m = {int(x): bool(m) for x, m in
                        zip(b.eval_view.sample_ids, b.eval_view.mask)}
                mask = np.array([id2m[int(x)] for x in sids], dtype=bool)

                q_path = ROOT / "results" / "quality" / cfg["dataset"]["name"] / \
                    f"{cfg['noise']['type']}{cfg['noise']['rate']}" / f"seed{cfg['seed']}"
                qs = sorted(q_path.glob("quality_*.parquet"))
                removed = w < args.cutoff
                hard_clean = np.zeros(len(w), dtype=bool)
                if qs:
                    qdf = pd.read_parquet(qs[0]).sort_values("sample_id")
                    qv = qdf[QUALITY_FOR_HARD].to_numpy(dtype=np.float64) \
                        if QUALITY_FOR_HARD in qdf.columns else None
                    if qv is not None and len(qv) == len(w):
                        thr = np.quantile(qv[~mask], 0.1) if (~mask).any() else np.nan
                        hard_clean = (~mask) & (qv <= thr)

                rows.append({
                    "setting": s,
                    "dataset": cfg["dataset"]["name"],
                    "noise": f"{cfg['noise']['type']}{cfg['noise']['rate']}",
                    "seed": cfg["seed"],
                    "lambda": lam,
                    "exclude_fraction_target": lam,
                    "exclude_fraction_realized": float(removed.mean()),
                    "test_accuracy": float(wsum.get("test_accuracy", float("nan"))),
                    "test_balanced_accuracy": float(wsum.get("test_balanced_accuracy", float("nan"))),
                    "noise_removal_rate": float(removed[mask].mean()) if mask.any() else float("nan"),
                    "clean_retention_rate": float((~removed[~mask]).mean()) if (~mask).any() else float("nan"),
                    "clean_false_removal_rate": float(removed[~mask].mean()) if (~mask).any() else float("nan"),
                    "hard_clean_false_removal_rate": float(removed[hard_clean].mean())
                    if hard_clean.any() else float("nan"),
                    "n_noisy": int(mask.sum()), "n_clean": int((~mask).sum()),
                    "n_hard_clean": int(hard_clean.sum()),
                    "weight_mode": wsum.get("weight_mode"),
                    "hard_threshold": str(wsum.get("hard_threshold")),
                    "config_hash": cfg_hash,
                    "weighted_summary_tag": wsum_tag,
                    "reused_from_archive": bool(wsum_tag != tag),
                })

    if not rows:
        print("no lambda-sweep results found yet — run revision/run_p1_lambda_sweep.sh first")
        return

    df = pd.DataFrame(rows).sort_values(["setting", "lambda", "seed"])
    df.to_csv(OUT / "lambda_sweep.csv", index=False)

    agg = (df.groupby(["setting", "dataset", "noise", "lambda"])
           .agg(n_seeds=("seed", "nunique"),
                exclude_fraction_realized=("exclude_fraction_realized", "mean"),
                test_accuracy_mean=("test_accuracy", "mean"),
                test_accuracy_std=("test_accuracy", "std"),
                noise_removal_rate=("noise_removal_rate", "mean"),
                clean_retention_rate=("clean_retention_rate", "mean"),
                clean_false_removal_rate=("clean_false_removal_rate", "mean"),
                hard_clean_false_removal_rate=("hard_clean_false_removal_rate", "mean"))
           .reset_index())
    agg.to_csv(OUT / "lambda_sweep_summary.csv", index=False)

    print(agg.round(4).to_string(index=False))
    print(f"\nwrote {OUT/'lambda_sweep.csv'} and {OUT/'lambda_sweep_summary.csv'}")


if __name__ == "__main__":
    main()
