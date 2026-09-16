#!/usr/bin/env python
"""paper_figures.py — paper-level summary figures.

    1. delta_conf_by_dataset.png   — Δ_conf bars per dataset x noise setting
    2. gate_b_delta.png            — Ours-vs-CE accuracy delta per setting
    3. method_accuracy.png          — accuracy bars: CE / Ours / baselines

Usage:  python scripts/paper_figures.py --root results --tag final_hard50 --tag-cifar10 gate_a_v3_hard50
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SETTINGS = ["symmetric0.2", "symmetric0.4", "asymmetric0.2", "asymmetric0.4"]
SEEDS = [0, 1, 2]
BASELINES = ["small_loss", "coteaching", "coteaching_plus", "jocor", "elr", "dividemix"]


def _json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _mean(paths: list[Path], key: str) -> float | None:
    vals = []
    for p in paths:
        d = _json(p)
        if d and d.get(key) is not None:
            vals.append(float(d[key]))
    return float(np.mean(vals)) if vals else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results")
    ap.add_argument("--tag", default="final_hard50")
    ap.add_argument("--tag-cifar10", default=None)
    ap.add_argument("--out", default="figures/paper")
    args = ap.parse_args()
    root = Path(args.root)
    tag10 = args.tag_cifar10 or args.tag
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # ---- 1. Δ_conf by dataset ----
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(SETTINGS))
    w = 0.36
    for i, ds in enumerate(["cifar10", "cifar100"]):
        vals = []
        for s in SETTINGS:
            v = _mean([root / "detectability" / ds / s / f"seed{seed}" / "global_summary.json" for seed in SEEDS], "delta_conf")
            vals.append(v if v is not None else 0.0)
        ax.bar(x + (i - 0.5) * w, vals, w, label=ds)
    ax.set_xticks(x)
    ax.set_xticklabels(SETTINGS, rotation=20)
    ax.set_ylabel("Δ_conf = AUC_global − AUC_matched")
    ax.set_title("Quality-confounding gap by dataset & noise setting")
    ax.axhline(0, color="k", lw=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "delta_conf_by_dataset.png", dpi=150)
    fig.savefig(out / "delta_conf_by_dataset.svg")
    plt.close(fig)
    print("[paper_figures] delta_conf_by_dataset")

    # ---- 2. Gate B delta (ours vs CE) ----
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, ds in enumerate(["cifar10", "cifar100"]):
        tag = tag10 if ds == "cifar10" else args.tag
        vals = []
        for s in SETTINGS:
            ours = _mean([root / "estimator" / ds / s / f"seed{seed}" /
                          (f"weighted_summary_{tag}.json" if tag else "weighted_summary.json") for seed in SEEDS], "test_accuracy")
            ce = _mean([root / "traces" / ds / s / f"seed{seed}" / "ce_summary.json" for seed in SEEDS], "test_accuracy")
            vals.append((ours - ce) if ours is not None and ce is not None else 0.0)
        ax.bar(x + (i - 0.5) * w, vals, w, label=ds)
    ax.set_xticks(x)
    ax.set_xticklabels(SETTINGS, rotation=20)
    ax.set_ylabel("Δ accuracy vs CE")
    ax.set_title("Ours vs standard CE")
    ax.axhline(0, color="k", lw=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "gate_b_delta.png", dpi=150)
    fig.savefig(out / "gate_b_delta.svg")
    plt.close(fig)
    print("[paper_figures] gate_b_delta")

    # ---- 3. method accuracy bars (CIFAR-10, per setting) ----
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for a, setting in zip(axes, SETTINGS):
        methods = ["CE", "Ours"] + BASELINES
        accs = []
        ce = _mean([root / "traces" / "cifar10" / setting / f"seed{s}" / "ce_summary.json" for s in SEEDS], "test_accuracy")
        ours = _mean([root / "estimator" / "cifar10" / setting / f"seed{s}" /
                      (f"weighted_summary_{tag10}.json" if tag10 else "weighted_summary.json") for s in SEEDS], "test_accuracy")
        accs = [ce, ours]
        for m in BASELINES:
            paths = []
            for s in SEEDS:
                mdir = root / "baselines" / "cifar10" / setting / f"seed{s}" / m
                if mdir.exists():
                    fs = list(mdir.glob("summary_*.json"))
                    if fs:
                        paths.append(fs[0])
            v = _mean(paths, "test_accuracy") if paths else None
            accs.append(v)
        colors = ["#bbbbbb", "#d62728"] + ["#1f77b4"] * len(BASELINES)
        keep = [(m, v, c) for (m, v, c) in zip(methods, accs, colors) if v is not None]
        if not keep:
            continue
        names = [k[0] for k in keep]
        vals = [k[1] for k in keep]
        cols = [k[2] for k in keep]
        a.bar(range(len(vals)), vals, color=cols)
        a.set_xticks(range(len(vals)))
        a.set_xticklabels(names, rotation=45, fontsize=8)
        a.set_title(setting)
        a.set_ylim(0.5, 0.9)
        for i, v in enumerate(vals):
            a.text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=7)
    fig.suptitle("CIFAR-10 accuracy by method")
    fig.tight_layout()
    fig.savefig(out / "method_accuracy.png", dpi=150)
    fig.savefig(out / "method_accuracy.svg")
    plt.close(fig)
    print("[paper_figures] method_accuracy")

    print("[paper_figures] done ->", out)


if __name__ == "__main__":
    main()
