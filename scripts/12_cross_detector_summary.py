"""12_cross_detector_summary.py — aggregate per-cell JSONs into tables + figures.

Reads:
    results/cross_detector/<dataset>/<noise>/seed<seed>/<detector>.json

Writes (under /root/quality_noise/results/cross_detector_summary/):
    table_cross_detector.md        (Δ_conf matrix, 6 detectors × 10 settings)
    table_cross_detector_full.md   (full matrix: AUC_g / AUC_m / Δ / CI per cell)
    matching_balance.md            (SMD before/after, n_pairs, etc.)
    cross_detector_summary.json    (machine-readable summary stats)
    fig_cross_detector_heatmap.pdf/.png

The heatmap is the centerpiece figure: rows = detectors, cols = settings,
cell = mean Δ_conf over seeds (with bootstrap CI band as a secondary panel
or text annotation).
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


SETTING_ORDER = [
    ("cifar10", "symmetric0.2", "C10-S20"),
    ("cifar10", "symmetric0.4", "C10-S40"),
    ("cifar10", "asymmetric0.2", "C10-A20"),
    ("cifar10", "asymmetric0.4", "C10-A40"),
    ("cifar100", "symmetric0.2", "C100-S20"),
    ("cifar100", "symmetric0.4", "C100-S40"),
    ("cifar100", "asymmetric0.2", "C100-A20"),
    ("cifar100", "asymmetric0.4", "C100-A40"),
    ("cifar10n", "cifar10n_aggre0.4", "C10N-Agg"),
    ("cifar10n", "cifar10n_worse0.4", "C10N-Worse"),
]

DETECTOR_ORDER = [
    ("ema_loss", "EMA Loss"),
    ("confidence", "Confidence"),
    ("aum", "AUM"),
    ("forgetting", "Forgetting"),
    ("neighbor", "Neighbor"),
    ("combined", "Combined"),
]


def collect_cells(in_root: Path) -> dict:
    """Return nested dict cells[det][(dataset, noise)] = list[CellResult-dicts].

    New layout: in_root/<quality_col>/<dataset>/<noise>/seed<seed>/<detector>.json
    """
    cells: dict = {d: {} for d, _ in DETECTOR_ORDER}
    for det, _ in DETECTOR_ORDER:
        for ds, ns, _ in SETTING_ORDER:
            base = in_root / ds / ns
            if not base.exists():
                cells[det][(ds, ns)] = []
                continue
            per_seed = []
            for seed_dir in sorted(base.iterdir()):
                if not seed_dir.is_dir() or not seed_dir.name.startswith("seed"):
                    continue
                f = seed_dir / f"{det}.json"
                if f.exists():
                    with open(f) as fh:
                        per_seed.append(json.load(fh))
            cells[det][(ds, ns)] = per_seed
    return cells


def aggregate(cells: dict) -> dict:
    """Aggregate per-seed into mean/std/CI for the heatmap."""
    agg = {}
    for det, _ in DETECTOR_ORDER:
        agg[det] = {}
        for ds, ns, _ in SETTING_ORDER:
            entries = cells[det][(ds, ns)]
            if not entries:
                agg[det][(ds, ns)] = {
                    "n_seeds": 0,
                    "delta_mean": None,
                    "delta_std": None,
                    "auc_g_mean": None,
                    "auc_m_mean": None,
                    "positive_count": 0,
                    "ci_delta_overall": None,
                }
                continue
            deltas = np.array([e["delta_conf"] for e in entries], dtype=float)
            deltas = deltas[~np.isnan(deltas)]
            auc_g = np.array([e["auc_global"] for e in entries], dtype=float)
            auc_m = np.array([e["auc_matched"] for e in entries], dtype=float)
            n_pos = int((deltas > 0).sum())

            # Aggregate bootstrap Δ_conf CIs across seeds by pooling means
            ci_lo_means = []
            ci_hi_means = []
            for e in entries:
                if "ci_delta" in e and e["ci_delta"][0] is not None and not np.isnan(e["ci_delta"][0]):
                    ci_lo_means.append(e["ci_delta"][0])
                    ci_hi_means.append(e["ci_delta"][1])
            agg[det][(ds, ns)] = {
                "n_seeds": len(entries),
                "delta_mean": float(deltas.mean()) if len(deltas) else None,
                "delta_std": float(deltas.std(ddof=1)) if len(deltas) > 1 else 0.0,
                "auc_g_mean": float(auc_g.mean()) if len(auc_g) else None,
                "auc_m_mean": float(auc_m.mean()) if len(auc_m) else None,
                "positive_count": n_pos,
                "n_total": len(entries),
                "ci_delta_overall": (
                    float(np.mean(ci_lo_means)) if ci_lo_means else None,
                    float(np.mean(ci_hi_means)) if ci_hi_means else None,
                ),
            }
    return agg


def write_matrix_table(agg: dict, out_path: Path) -> None:
    """table_cross_detector.md — Δ_conf matrix."""
    lines = []
    header = ["Detector"] + [lbl for _, _, lbl in SETTING_ORDER]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for det, det_lbl in DETECTOR_ORDER:
        row = [det_lbl]
        for ds, ns, _ in SETTING_ORDER:
            a = agg[det][(ds, ns)]
            if a["n_seeds"] == 0:
                row.append("N/A")
            else:
                mean = a["delta_mean"]
                std = a["delta_std"]
                row.append(f"{mean:+.3f} ± {std:.3f}")
        lines.append("| " + " | ".join(row) + " |")
    out_path.write_text("# Cross-Detector Δ_conf Matrix\n\n")
    out_path.write_text("Values are mean ± std Δ_conf over seeds. Higher = more confounding.\n\n")
    out_path.write_text("\n".join(lines) + "\n")


def write_full_table(cells: dict, agg: dict, out_path: Path) -> None:
    """table_cross_detector_full.md — full per-cell info."""
    out = ["# Cross-Detector Full Table\n",
           "For each (detector, setting): mean Δ_conf, AUC_g, AUC_m, and bootstrap CI.\n"]
    for det, det_lbl in DETECTOR_ORDER:
        out.append(f"\n## {det_lbl}\n")
        cols = ["Setting", "n_seeds", "AUC_g (mean)", "AUC_m (mean)", "Δ_conf (mean±std)", "n_pos", "Δ CI (lo, hi)"]
        out.append("| " + " | ".join(cols) + " |")
        out.append("|" + "|".join(["---"] * len(cols)) + "|")
        for ds, ns, lbl in SETTING_ORDER:
            a = agg[det][(ds, ns)]
            if a["n_seeds"] == 0:
                out.append(f"| {lbl} | 0 | N/A | N/A | N/A | N/A | N/A |")
            else:
                ci = a["ci_delta_overall"]
                ci_str = f"({ci[0]:+.3f}, {ci[1]:+.3f})" if ci[0] is not None else "N/A"
                out.append(
                    f"| {lbl} | {a['n_seeds']} | {a['auc_g_mean']:.3f} | "
                    f"{a['auc_m_mean']:.3f} | {a['delta_mean']:+.3f} ± {a['delta_std']:.3f} | "
                    f"{a['positive_count']}/{a['n_total']} | {ci_str} |"
                )
    out_path.write_text("\n".join(out) + "\n")


def write_matching_balance(cells: dict, out_path: Path) -> None:
    """matching_balance.md — SMD before/after per cell."""
    out = ["# Matching Quality Diagnostics\n",
           "Per-cell SMD_before, SMD_after, n_pairs. Goal: |SMD_after| < 0.1.\n"]
    cols = ["Detector", "Setting", "n_pairs", "SMD_before", "SMD_after", "mean|ΔQ|", "median|ΔQ|"]
    out.append("| " + " | ".join(cols) + " |")
    out.append("|" + "|".join(["---"] * len(cols)) + "|")
    for det, det_lbl in DETECTOR_ORDER:
        for ds, ns, lbl in SETTING_ORDER:
            entries = cells[det][(ds, ns)]
            if not entries:
                continue
            n_pairs = int(np.mean([e["matching"]["n_pairs"] for e in entries]))
            smd_before = np.mean([e["matching"]["smd_before"] for e in entries])
            smd_after = np.mean([e["matching"]["smd_after"] for e in entries])
            mean_diff = np.mean([e["matching"]["mean_abs_q_diff"] for e in entries])
            median_diff = np.mean([e["matching"]["median_abs_q_diff"] for e in entries])
            out.append(
                f"| {det_lbl} | {lbl} | {n_pairs} | {smd_before:+.3f} | "
                f"{smd_after:+.4f} | {mean_diff:.4f} | {median_diff:.4f} |"
            )
    out_path.write_text("\n".join(out) + "\n")


def write_global_summary(agg: dict, out_path: Path) -> None:
    """cross_detector_summary.json — overall stats."""
    all_deltas = []
    n_pos = 0
    n_total = 0
    n_zero_ci = 0
    per_setting = {}
    per_detector = {}
    for det, _ in DETECTOR_ORDER:
        det_deltas = []
        for ds, ns, _ in SETTING_ORDER:
            a = agg[det][(ds, ns)]
            if a["n_seeds"] == 0:
                continue
            det_deltas.append(a["delta_mean"])
            all_deltas.append(a["delta_mean"])
            if a["delta_mean"] is not None:
                n_total += 1
                if a["delta_mean"] > 0:
                    n_pos += 1
                ci = a["ci_delta_overall"]
                if ci[0] is not None and ci[1] is not None and ci[0] <= 0 <= ci[1]:
                    n_zero_ci += 1
                key = f"{ds}__{ns}"
                per_setting.setdefault(key, {"deltas": [], "positives": 0, "n_detectors": 0})
                per_setting[key]["deltas"].append(a["delta_mean"])
                per_setting[key]["n_detectors"] += 1
                if a["delta_mean"] > 0:
                    per_setting[key]["positives"] += 1
        per_detector[det] = {
            "mean_delta": float(np.mean(det_deltas)) if det_deltas else None,
            "min_delta": float(np.min(det_deltas)) if det_deltas else None,
            "max_delta": float(np.max(det_deltas)) if det_deltas else None,
            "n_settings_evaluated": len(det_deltas),
            "n_settings_positive": int(sum(1 for d in det_deltas if d > 0)),
        }
    arr = np.array([d for d in all_deltas if d is not None])
    summary = {
        "detectors_tested": [d for d, _ in DETECTOR_ORDER],
        "settings_tested": [lbl for _, _, lbl in SETTING_ORDER],
        "total_cells": int(n_total),
        "positive_delta_cells": int(n_pos),
        "negative_delta_cells": int(n_total - n_pos),
        "zero_crossing_CI_cells": int(n_zero_ci),
        "median_delta_conf": float(np.median(arr)) if len(arr) else None,
        "mean_delta_conf": float(np.mean(arr)) if len(arr) else None,
        "min_delta_conf": float(np.min(arr)) if len(arr) else None,
        "max_delta_conf": float(np.max(arr)) if len(arr) else None,
        "positive_rate": float(n_pos / n_total) if n_total else None,
        "per_setting": {
            k: {
                "n_detectors": v["n_detectors"],
                "positives": v["positives"],
                "mean_delta": float(np.mean(v["deltas"])) if v["deltas"] else None,
                "min_delta": float(np.min(v["deltas"])) if v["deltas"] else None,
                "max_delta": float(np.max(v["deltas"])) if v["deltas"] else None,
            }
            for k, v in per_setting.items()
        },
        "per_detector": per_detector,
    }
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    return summary


def make_heatmap(agg: dict, out_path: Path, summary: dict | None = None) -> None:
    """Render the cross-detector confounding heatmap."""
    det_labels = [lbl for _, lbl in DETECTOR_ORDER]
    set_labels = [lbl for _, _, lbl in SETTING_ORDER]
    M = np.full((len(DETECTOR_ORDER), len(SETTING_ORDER)), np.nan)
    n_seeds_grid = np.zeros_like(M, dtype=int)
    for i, (det, _) in enumerate(DETECTOR_ORDER):
        for j, (ds, ns, _) in enumerate(SETTING_ORDER):
            a = agg[det][(ds, ns)]
            if a["n_seeds"] > 0:
                M[i, j] = a["delta_mean"]
                n_seeds_grid[i, j] = a["n_seeds"]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    vmax = float(np.nanmax(np.abs(M))) if not np.isnan(M).all() else 0.1
    vmax = max(vmax, 0.05)
    im = ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")

    # Annotate each cell with value and n_seeds
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isnan(M[i, j]):
                ax.text(j, i, "N/A", ha="center", va="center", color="black", fontsize=8)
            else:
                ns = n_seeds_grid[i, j]
                color = "white" if abs(M[i, j]) > 0.5 * vmax else "black"
                ax.text(j, i, f"{M[i, j]:+.3f}\n(n={ns})", ha="center", va="center",
                        color=color, fontsize=8)

    ax.set_xticks(range(len(set_labels)))
    ax.set_xticklabels(set_labels, rotation=30, ha="right", fontsize=10)
    ax.set_yticks(range(len(det_labels)))
    ax.set_yticklabels(det_labels, fontsize=10)
    ax.set_title("Cross-Detector Confounding Heatmap\n"
                 "Δ_conf = AUC_global − AUC_matched (mean over seeds)",
                 fontsize=12, pad=12)
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
    cbar.set_label("Δ_conf  (positive ⇒ matching reduces apparent detectability)", fontsize=10)

    # Caption with overall summary
    if summary:
        cap = (
            f"Cells: {summary['total_cells']}; positive: {summary['positive_delta_cells']}; "
            f"mean Δ_conf: {summary['mean_delta_conf']:+.3f}; "
            f"median Δ_conf: {summary['median_delta_conf']:+.3f}"
        )
        fig.text(0.5, -0.02, cap, ha="center", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    png_path = out_path.with_suffix(".png")
    fig.savefig(png_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-root", type=Path,
                    default=Path("/root/quality_noise/results/cross_detector/knn_agreement"))
    ap.add_argument("--out-root", type=Path,
                    default=Path("/root/quality_noise/results/cross_detector_summary/knn_agreement"))
    ap.add_argument("--quality-col", type=str, default="knn_agreement")
    args = ap.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)

    print(f"== Cross-detector summary ==", flush=True)
    print(f"   in: {args.in_root}", flush=True)
    print(f"   out: {args.out_root}", flush=True)
    print(f"   quality_col: {args.quality_col}", flush=True)

    cells = collect_cells(args.in_root)
    agg = aggregate(cells)

    write_matrix_table(agg, args.out_root / "table_cross_detector.md")
    write_full_table(cells, agg, args.out_root / "table_cross_detector_full.md")
    write_matching_balance(cells, args.out_root / "matching_balance.md")
    summary = write_global_summary(agg, args.out_root / "cross_detector_summary.json")
    make_heatmap(agg, args.out_root / "fig_cross_detector_heatmap.pdf", summary=summary)

    print("\n== Overall ==")
    print(f"  total_cells: {summary['total_cells']}")
    print(f"  positive_delta_cells: {summary['positive_delta_cells']}")
    if summary['mean_delta_conf'] is not None:
        print(f"  mean Δ_conf: {summary['mean_delta_conf']:+.4f}")
        print(f"  median Δ_conf: {summary['median_delta_conf']:+.4f}")
    for det, info in summary["per_detector"].items():
        if info["mean_delta"] is not None:
            print(f"  {det:12s}: mean={info['mean_delta']:+.3f}  "
                  f"min={info['min_delta']:+.3f}  max={info['max_delta']:+.3f}  "
                  f"positive={info['n_settings_positive']}/{info['n_settings_evaluated']}")

    # Gate checks
    n_pos = summary["positive_delta_cells"]
    n_total = summary["total_cells"]
    print("\n== Gate checks ==")
    if n_total:
        pct = n_pos / n_total
        print(f"  Gate 1 (>=80% positive): {n_pos}/{n_total} = {pct*100:.1f}%  "
              f"{'PASS' if pct >= 0.8 else 'FAIL'}")
    # Gate 2: CIFAR-100 sym20 has AUC_g > AUC_m for multiple detectors
    c100_s20_deltas = []
    for det, _ in DETECTOR_ORDER:
        a = agg[det][("cifar100", "symmetric0.2")]
        if a["n_seeds"] > 0 and a["delta_mean"] is not None:
            c100_s20_deltas.append(a["delta_mean"])
    n_pos_c100 = sum(1 for d in c100_s20_deltas if d > 0)
    print(f"  Gate 2 (C100-S20 multi-detector Δ>0): {n_pos_c100}/{len(c100_s20_deltas)}  "
          f"{'PASS' if n_pos_c100 >= 3 else 'FAIL'}")
    # Gate 3: CIFAR-10N majority positive
    c10n_pos = 0
    c10n_total = 0
    for det, _ in DETECTOR_ORDER:
        for ns in ("cifar10n_aggre0.4", "cifar10n_worse0.4"):
            a = agg[det][("cifar10n", ns)]
            if a["n_seeds"] > 0 and a["delta_mean"] is not None:
                c10n_total += 1
                if a["delta_mean"] > 0:
                    c10n_pos += 1
    if c10n_total:
        print(f"  Gate 3 (C10N majority Δ>0): {c10n_pos}/{c10n_total}  "
              f"{'PASS' if c10n_pos / c10n_total >= 0.5 else 'FAIL'}")
    # Gate 4: prototype Q (this is reported in cross_quality script)


if __name__ == "__main__":
    main()
