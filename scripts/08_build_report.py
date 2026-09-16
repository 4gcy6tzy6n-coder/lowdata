"""08_build_report.py — final aggregation, Gate B evaluation, paper figures.

Reads per-seed artifacts (detectability, estimator, weighted, baselines) and
produces:
    results/summary.csv                     one row per (run, method)
    results/gate_b.csv                      three-way evaluation per noise setting
    figures/gate_b_pareto.png               clean-retention vs noise-suppression
    figures/summary_*.png                   per-metric bar charts across methods

Usage:
    python scripts/08_build_report.py --exp configs/experiments/gate_a.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from quality_noise.config import load_config
from quality_noise.experiment import expand_experiment, run_key
from quality_noise.utils import make_results_dirs


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _collect_rows(exp: dict, results: dict, tag: str = "") -> pd.DataFrame:
    runs = expand_experiment(exp)
    rows = []
    for cfg in runs:
        base = Path(cfg["dataset"]["name"]) / f"{cfg['noise']['type']}{cfg['noise']['rate']}" / f"seed{cfg['seed']}"
        est_name = f"estimator_summary_{tag}.json" if tag else "estimator_summary.json"
        wgt_name = f"weighted_summary_{tag}.json" if tag else "weighted_summary.json"
        det = _load_json(results["detectability"] / base / "global_summary.json")
        est = _load_json(results["estimator"] / base / est_name)
        wgt = _load_json(results["estimator"] / base / wgt_name)
        # CE baseline: standard training is done in 02; its test eval is saved there.
        ce = _load_json(results["traces"] / base / "ce_summary.json")
        row = {
            "dataset": cfg["dataset"]["name"],
            "noise_type": cfg["noise"]["type"],
            "noise_rate": cfg["noise"]["rate"],
            "seed": cfg["seed"],
            "method": "weighted",
        }
        if ce:
            rows.append(
                {
                    "dataset": cfg["dataset"]["name"],
                    "noise_type": cfg["noise"]["type"],
                    "noise_rate": cfg["noise"]["rate"],
                    "seed": cfg["seed"],
                    "method": "ce",
                    "test_accuracy": ce.get("test_accuracy"),
                    "test_balanced_accuracy": ce.get("test_balanced_accuracy"),
                }
            )
        if det:
            row.update({"auc_global": det["auc_global"], "auc_matched": det["auc_matched"], "delta_conf": det["delta_conf"]})
        if est:
            row.update(
                {
                    "detection_auroc": est.get("auroc"),
                    "hard_clean_false_suppression": est.get("hard_clean_false_suppression"),
                    "false_exclusion_rate": est.get("false_exclusion_rate"),
                    "noise_rate_estimate": est.get("noise_rate_estimate"),
                }
            )
        if wgt:
            row.update({"test_accuracy": wgt.get("test_accuracy"), "test_balanced_accuracy": wgt.get("test_balanced_accuracy")})
        rows.append(row)

        # Baseline rows (CE + strong baselines).
        for method_dir in sorted((results["baselines"] / base).glob("*")) if (results["baselines"] / base).exists() else []:
            if not method_dir.is_dir():
                continue
            files = list(method_dir.glob("summary_*.json"))
            if not files:
                continue
            s = _load_json(files[0])
            if not s:
                continue
            rows.append(
                {
                    "dataset": cfg["dataset"]["name"],
                    "noise_type": cfg["noise"]["type"],
                    "noise_rate": cfg["noise"]["rate"],
                    "seed": cfg["seed"],
                    "method": method_dir.name,
                    "test_accuracy": s.get("test_accuracy"),
                    "test_balanced_accuracy": s.get("test_balanced_accuracy"),
                }
            )
    return pd.DataFrame(rows)


def _gate_b(exp: dict, results: dict, summary: pd.DataFrame, tag: str = "") -> pd.DataFrame:
    """Three-way evaluation: noise detection up, hard-clean damage down,
    downstream accuracy not degraded (compared against standard CE)."""
    rows = []
    for cfg in expand_experiment(exp):
        base = Path(cfg["dataset"]["name"]) / f"{cfg['noise']['type']}{cfg['noise']['rate']}" / f"seed{cfg['seed']}"
        est_name = f"estimator_summary_{tag}.json" if tag else "estimator_summary.json"
        wgt_name = f"weighted_summary_{tag}.json" if tag else "weighted_summary.json"
        est = _load_json(results["estimator"] / base / est_name)
        wgt = _load_json(results["estimator"] / base / wgt_name)
        if est is None or wgt is None:
            continue
        # CE baseline accuracy: standard training eval saved by 02.
        ce_sum = _load_json(results["traces"] / base / "ce_summary.json")
        rows.append(
            {
                "noise_type": cfg["noise"]["type"],
                "noise_rate": cfg["noise"]["rate"],
                "seed": cfg["seed"],
                "detection_auroc": est.get("auroc"),
                "hard_clean_false_suppression": est.get("hard_clean_false_suppression"),
                "false_exclusion_rate": est.get("false_exclusion_rate"),
                "weighted_acc": wgt.get("test_accuracy"),
                "ce_acc": ce_sum.get("test_accuracy") if ce_sum else None,
                "acc_delta_vs_ce": (wgt.get("test_accuracy") - ce_sum.get("test_accuracy")) if ce_sum and ce_sum.get("test_accuracy") is not None else None,
            }
        )
    df = pd.DataFrame(rows)
    return df


def _pareto_figure(exp: dict, results: dict, fig_root: Path, tag: str = "") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = _collect_rows(exp, results, tag=tag)
    methods = []
    pts = []
    for m in df["method"].unique():
        sub = df[df["method"] == m]
        if sub.empty:
            continue
        # noise suppression = fraction of noisy downweighted (only for methods
        # with per-sample weights); clean retention = 1 - FER.
        if "noise_rate_estimate" in sub.columns and sub["noise_rate_estimate"].notna().any():
            noise_supp = sub["noise_rate_estimate"].mean()
            clean_ret = (1 - sub["false_exclusion_rate"]).mean() if sub["false_exclusion_rate"].notna().any() else 1.0
            acc = sub["test_accuracy"].mean() if sub["test_accuracy"].notna().any() else 0.0
        else:
            # Baselines without weights: retain everything, suppress nothing.
            noise_supp = 0.0
            clean_ret = 1.0
            acc = sub["test_accuracy"].mean() if sub["test_accuracy"].notna().any() else 0.0
        methods.append(m)
        pts.append((noise_supp, clean_ret, acc, m))
    if not pts:
        return
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for x, y, acc, m in pts:
        ax.scatter(x, y, s=80 + 800 * acc, alpha=0.7, label=f"{m} (acc={acc:.3f})")
        ax.annotate(m, (x, y), textcoords="offset points", xytext=(6, 6), fontsize=9)
    ax.set_xlabel("Noisy-label suppression (est. noise rate)")
    ax.set_ylabel("Hard-clean retention (1 - false exclusion)")
    ax.set_title("Clean retention vs noise suppression (marker size = test accuracy)")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_root / "gate_b_pareto.png", dpi=150)
    fig.savefig(fig_root / "gate_b_pareto.svg")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="configs/experiments/gate_a.yaml")
    ap.add_argument("--device", default="auto", help="accepted for uniform CLI; reporting is CPU-only")
    ap.add_argument("--tag", default="", help="report a variant tag (weighted_summary_<tag>.json / estimator_summary_<tag>.json)")
    args = ap.parse_args()

    exp = load_config(args.exp)
    results = make_results_dirs({"results_root": exp.get("results_root")})
    fig_root = Path("figures")
    fig_root.mkdir(exist_ok=True)

    suffix = f"_{args.tag}" if args.tag else ""
    summary = _collect_rows(exp, results, tag=args.tag)
    summary.to_csv(results["detectability"].parent / f"summary{suffix}.csv", index=False)
    print(f"[08] summary{suffix}.csv: {len(summary)} rows")

    gate_b = _gate_b(exp, results, summary, tag=args.tag)
    if not gate_b.empty:
        gate_b.to_csv(results["detectability"].parent / f"gate_b{suffix}.csv", index=False)
        g = gate_b.groupby(["noise_type", "noise_rate"]).agg(
            detection_auroc=("detection_auroc", "mean"),
            hard_clean_fss=("hard_clean_false_suppression", "mean"),
            weighted_acc=("weighted_acc", "mean"),
            ce_acc=("ce_acc", "mean"),
            acc_delta=("acc_delta_vs_ce", "mean"),
        )
        print("[08] Gate B (mean over seeds):")
        print(g.round(3).to_string())

    _pareto_figure(exp, results, fig_root, tag=args.tag)
    print("[08] done")


if __name__ == "__main__":
    main()
