#!/usr/bin/env python
"""gen_final_figures.py — emit the 3 core paper figures.

  Figure 1: conceptual framework (E, Q, R -> observed detectability -> governance)
  Figure 2: core Δ_conf bar chart (per dataset x noise setting, with CI)
  Figure 3: detectability-governance boundary (matched AUC vs soft/hard ΔAcc)

Reads server_results/results; data-missing points are skipped.
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


def _json(p: Path) -> dict | None:
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def _mean_key(root: Path, sub: str, ds: str, setting: str, fname: str, key: str) -> float | None:
    vals = []
    for s in [0, 1, 2, 3, 4]:
        d = _json(root / sub / ds / setting / f"seed{s}" / fname)
        if d and d.get(key) is not None:
            vals.append(float(d[key]))
    return float(np.mean(vals)) if vals else None


def _fig1(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 2.6))
    ax.axis("off")
    boxes = [
        (0.02, 0.35, 0.14, 0.3, "Evidence\nE"),
        (0.20, 0.35, 0.14, 0.3, "Quality\nQ"),
        (0.38, 0.35, 0.14, 0.3, "Reliability\nR"),
        (0.60, 0.35, 0.26, 0.3, "Observed\ndetectability\n(global AUC)"),
        (0.92, 0.35, 0.20, 0.3, "Governance\n(hard/soft)"),
    ]
    for x, y, w, h, label in boxes:
        ax.add_patch(plt.Rectangle((x, y), w, h, fill=False, lw=1.5, edgecolor="k"))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=9)
    for x1, x2 in [(0.16, 0.20), (0.34, 0.38), (0.52, 0.60), (0.86, 0.92)]:
        ax.annotate("", xy=(x2, 0.5), xytext=(x1, 0.5), arrowprops=dict(arrowstyle="->", lw=1.2))
    ax.text(0.5, 0.85, "E, Q, R  →  observed detectability  →  governance", ha="center", fontsize=10, weight="bold")
    ax.text(0.5, 0.12, "global detection can exploit quality differences (confounding)", ha="center", fontsize=8, style="italic")
    fig.tight_layout()
    fig.savefig(out / "fig1_framework.png", dpi=150)
    fig.savefig(out / "fig1_framework.svg")
    plt.close(fig)
    print("[fig] fig1_framework")


def _fig2(root: Path, out: Path) -> None:
    labels, deltas, ci_lo, ci_hi = [], [], [], []
    for ds, sset in [("cifar10", SETTINGS), ("cifar100", SETTINGS)]:
        for setting in sset:
            vals = []
            for s in [0, 1, 2, 3, 4]:
                d = _json(root / "detectability" / ds / setting / f"seed{s}" / "global_summary.json")
                if d and d.get("delta_conf") is not None:
                    vals.append(float(d["delta_conf"]))
            if vals:
                labels.append(f"{ds}\n{setting}")
                deltas.append(float(np.mean(vals)))
                ci_lo.append(float(np.mean(vals) - 1.96 * np.std(vals) / np.sqrt(len(vals))))
                ci_hi.append(float(np.mean(vals) + 1.96 * np.std(vals) / np.sqrt(len(vals))))
    if not deltas:
        print("[fig] fig2: no data")
        return
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(deltas))
    yerr = np.array([np.array(deltas) - np.array(ci_lo), np.array(ci_hi) - np.array(deltas)])
    ax.bar(x, deltas, 0.6, yerr=yerr, capsize=3, color="#1f77b4", edgecolor="k")
    ax.axhline(0, color="k", lw=1)
    for i, v in enumerate(deltas):
        ax.text(i, v + 0.01, f"{v:+.3f}", ha="center", fontsize=8, rotation=0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7, rotation=30, ha="right")
    ax.set_ylabel("Δ_conf = AUC_global − AUC_matched")
    ax.set_title("Quality-confounding gap (mean ± 95% CI over seeds)")
    fig.tight_layout()
    fig.savefig(out / "fig2_delta_conf.png", dpi=150)
    fig.savefig(out / "fig2_delta_conf.svg")
    plt.close(fig)
    print("[fig] fig2_delta_conf")


def _fig3(root: Path, out: Path) -> None:
    pts = {"soft": ([], []), "hard": ([], [])}
    for ds, sset in [("cifar10", SETTINGS), ("cifar100", SETTINGS)]:
        for setting in sset:
            matched = _mean_key(root, "detectability", ds, setting, "global_summary.json", "auc_matched")
            soft = _mean_key(root, "traces", ds, setting, "ce_summary.json", "test_accuracy")
            soft_m = _mean_key(root, "estimator", ds, setting, "weighted_summary_v2_soft.json", "test_accuracy")
            hard_m = _mean_key(root, "estimator", ds, setting, "weighted_summary_gate_a_v3_hard50.json", "test_accuracy")
            if hard_m is None:
                hard_m = _mean_key(root, "estimator", ds, setting, "weighted_summary_final_hard50.json", "test_accuracy")
            if soft_m is None:
                soft_m = _mean_key(root, "estimator", ds, setting, "weighted_summary_v2_soft_asym.json", "test_accuracy")
            if matched is None or soft is None:
                continue
            if soft_m is not None:
                pts["soft"][0].append(matched); pts["soft"][1].append(soft_m - soft)
            if hard_m is not None:
                pts["hard"][0].append(matched); pts["hard"][1].append(hard_m - soft)
    if not pts["hard"][0]:
        print("[fig] fig3: no data")
        return
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for kind, color in [("soft", "#2ca02c"), ("hard", "#d62728")]:
        xs, ys = pts[kind]
        if not xs:
            continue
        ax.scatter(xs, ys, label=f"{kind} governance", color=color, s=60, alpha=0.8)
        z = np.polyfit(xs, ys, 1)
        xl = np.linspace(min(xs), max(xs), 50)
        ax.plot(xl, np.polyval(z, xl), color=color, ls="--", lw=1.2)
    ax.axhline(0, color="k", lw=1)
    ax.set_xlabel("Matched-quality AUC (detectability)")
    ax.set_ylabel("Δ accuracy vs CE")
    ax.set_title("Detectability → governance")
    # Border note
    xlim = ax.get_xlim(); ylim = ax.get_ylim()
    ax.text(xlim[1] - 0.02, ylim[1] - 0.02, "higher detectability\n→ harder governance safer", ha="right", va="top", fontsize=8, color="#666")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "fig3_detectability_governance.png", dpi=150)
    fig.savefig(out / "fig3_detectability_governance.svg")
    plt.close(fig)
    print("[fig] fig3_detectability_governance")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="server_results/results")
    ap.add_argument("--out", default="final_figures")
    args = ap.parse_args()
    root = Path(args.root)
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    _fig1(out)
    _fig2(root, out)
    _fig3(root, out)


if __name__ == "__main__":
    main()
