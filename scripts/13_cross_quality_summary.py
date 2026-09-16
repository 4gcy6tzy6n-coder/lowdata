"""13_cross_quality_summary.py — Cross-Q matching robustness (P0.5).

Compares Δ_conf computed with KNN-agreement quality vs prototype-margin quality
on the 4 representative settings:
    A: CIFAR-10 sym20
    B: CIFAR-10 asym40
    C: CIFAR-100 sym20
    D: CIFAR-10N Aggregate

Both quality-col runs are produced by scripts/11_cross_detector.py with
--quality-cols knn_agreement,margin (only on the 4 settings + their seeds).

Reads:
    results/cross_detector/knn_agreement/<dataset>/<noise>/...
    results/cross_detector/margin/<dataset>/<noise>/...

Writes (under results/cross_detector_summary/):
    cross_quality_matching.md
    fig_cross_quality_robustness.pdf/.png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.cross_detector.detectors import DETECTORS  # noqa: E402

# Representative settings: (dataset, noise, label, n_seeds_to_use)
REP_SETTINGS = [
    ("cifar10", "symmetric0.2", "C10-S20"),
    ("cifar10", "asymmetric0.4", "C10-A40"),
    ("cifar100", "symmetric0.2", "C100-S20"),
    ("cifar10n", "cifar10n_aggre0.4", "C10N-Agg"),
]

DETECTOR_ORDER = [
    ("ema_loss", "EMA Loss"),
    ("confidence", "Confidence"),
    ("aum", "AUM"),
    ("forgetting", "Forgetting"),
    ("neighbor", "Neighbor"),
    ("combined", "Combined"),
]


def _mean_delta(q_root: Path, det: str, ds: str, ns: str) -> dict:
    """Return {mean, std, n} of Δ_conf over seeds for one quality col."""
    base = q_root / ds / ns
    if not base.exists():
        return {"mean": None, "n": 0}
    vals = []
    for seed_dir in sorted(base.iterdir()):
        if not seed_dir.is_dir():
            continue
        f = seed_dir / f"{det}.json"
        if not f.exists():
            continue
        with open(f) as fh:
            d = json.load(fh)
        if d["delta_conf"] is not None and not np.isnan(d["delta_conf"]):
            vals.append(d["delta_conf"])
    if not vals:
        return {"mean": None, "n": 0}
    arr = np.array(vals)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "n": len(arr),
        "positive": int((arr > 0).sum()),
    }


def write_report(knn_root: Path, proto_root: Path, out_root: Path) -> None:
    lines = [
        "# Cross-Quality Matching Robustness (P0.5)\n",
        "Δ_conf computed with two quality metrics on 4 representative settings:",
        "- **Q^KNN**: KNN observed-label agreement (the paper's primary Q)",
        "- **Q^proto**: prototype margin (Q1)",
        "\nRequirement: main direction consistent (majority of detectors have Δ_conf > 0 under both Q).\n",
    ]
    cols = ["Detector", "Setting", "KNN-Q Δ_conf", "Proto-Q Δ_conf", "Direction-consistent?"]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join(["---"] * len(cols)) + "|")

    matrix = {}  # det -> list of (setting_label, knn_mean, proto_mean)
    for det, det_lbl in DETECTOR_ORDER:
        matrix[det] = []
        for ds, ns, lbl in REP_SETTINGS:
            k = _mean_delta(knn_root, det, ds, ns)
            p = _mean_delta(proto_root, det, ds, ns)
            if k["mean"] is None and p["mean"] is None:
                continue
            consistent = (k["mean"] is not None and p["mean"] is not None
                          and (k["mean"] > 0) == (p["mean"] > 0))
            k_str = f"{k['mean']:+.3f}" if k["mean"] is not None else "N/A"
            p_str = f"{p['mean']:+.3f}" if p["mean"] is not None else "N/A"
            lines.append(
                f"| {det_lbl} | {lbl} | {k_str} | {p_str} | {'✓' if consistent else ('N/A' if k['mean'] is None or p['mean'] is None else '✗')} |"
            )
            matrix[det].append((lbl, k["mean"], p["mean"]))

    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "cross_quality_matching.md").write_text("\n".join(lines) + "\n")

    # Consistency summary
    n_both = 0
    n_consistent = 0
    n_knn_pos = 0
    n_proto_pos = 0
    for det, rows in matrix.items():
        for lbl, km, pm in rows:
            if km is None or pm is None:
                continue
            n_both += 1
            if (km > 0) == (pm > 0):
                n_consistent += 1
            if km > 0:
                n_knn_pos += 1
            if pm > 0:
                n_proto_pos += 1
    stats = {
        "cells_with_both": n_both,
        "direction_consistent": n_consistent,
        "knn_q_positive": n_knn_pos,
        "proto_q_positive": n_proto_pos,
    }
    (out_root / "cross_quality_summary.json").write_text(
        json.dumps(stats, indent=2) + "\n"
    )
    print(f"  cross-Q cells with both: {n_both}, consistent: {n_consistent}, "
          f"KNN+ : {n_knn_pos}, Proto+: {n_proto_pos}", flush=True)

    # Figure: grouped bars per detector per setting (KNN-Q vs Proto-Q)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharey=True)
    axes = np.array(axes).ravel()
    for i, (ds, ns, lbl) in enumerate(REP_SETTINGS):
        ax = axes[i]
        dets = []
        kvals = []
        pvals = []
        for det, det_lbl in DETECTOR_ORDER:
            k = _mean_delta(knn_root, det, ds, ns)
            p = _mean_delta(proto_root, det, ds, ns)
            if k["mean"] is None and p["mean"] is None:
                continue
            dets.append(det_lbl)
            kvals.append(k["mean"] if k["mean"] is not None else np.nan)
            pvals.append(p["mean"] if p["mean"] is not None else np.nan)
        x = np.arange(len(dets))
        w = 0.38
        ax.bar(x - w / 2, kvals, w, label="KNN-Q", color="#4C72B0")
        ax.bar(x + w / 2, pvals, w, label="Proto-Q", color="#DD8452")
        ax.axhline(0, color="black", lw=0.6)
        ax.set_title(f"{lbl} (matched AUC regimes)")
        ax.set_xticks(x)
        ax.set_xticklabels(dets, rotation=25, ha="right", fontsize=8)
        ax.legend(fontsize=8)
    fig.suptitle("Cross-Quality Robustness of Δ_conf (KNN-Q vs Prototype-Q)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_root / "fig_cross_quality_robustness.pdf", bbox_inches="tight")
    fig.savefig(out_root / "fig_cross_quality_robustness.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("  wrote cross_quality_matching.md + fig_cross_quality_robustness", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cross-root", type=Path,
                    default=Path("/root/quality_noise/results/cross_detector"))
    ap.add_argument("--out-root", type=Path,
                    default=Path("/root/quality_noise/results/cross_detector_summary"))
    args = ap.parse_args()
    knn_root = args.cross_root / "knn_agreement"
    proto_root = args.cross_root / "margin"
    write_report(knn_root, proto_root, args.out_root)


if __name__ == "__main__":
    main()
