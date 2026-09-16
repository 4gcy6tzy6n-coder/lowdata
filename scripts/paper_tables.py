#!/usr/bin/env python
"""paper_tables.py — generate paper-level result tables from all artifacts.

Reads results/{traces,estimator,baselines,detectability}/... and emits:
    paper_tables/main_table.md      — accuracy matrix (datasets x noise x methods)
    paper_tables/gate_b.md          — three-way Gate B summary
    paper_tables/detectability.md   — Gate A: global/matched AUC + Delta_conf
    paper_tables/ablation.md        — ablation rows for the final method
    paper_tables/representation.md  — native vs pretrained Delta_conf

Usage:  python scripts/paper_tables.py [--root results]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

SETTINGS = ["symmetric0.2", "symmetric0.4", "asymmetric0.2", "asymmetric0.4"]
METHOD_ORDER = ["ce", "ours", "small_loss", "coteaching", "coteaching_plus", "jocor", "elr", "dividemix"]
SEEDS = [0, 1, 2]


def _json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _accs(root: Path, sub: str, setting: str, fname: str) -> list[float]:
    out = []
    for seed in SEEDS:
        d = _json(root / sub / setting / f"seed{seed}" / fname)
        if d and d.get("test_accuracy") is not None:
            out.append(float(d["test_accuracy"]))
    return out


def _fmt(accs: list[float]) -> str:
    if not accs:
        return "—"
    m, s = float(np.mean(accs)), float(np.std(accs))
    return f"{m:.3f}±{s:.3f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results")
    ap.add_argument("--tag", default="final_hard50", help="weighted/estimator variant tag")
    ap.add_argument("--tag-cifar10", default=None, help="CIFAR-10 tag (defaults to --tag)")
    args = ap.parse_args()
    root = Path(args.root)
    out_dir = Path("paper_tables")
    out_dir.mkdir(exist_ok=True)
    tag10 = args.tag_cifar10 or args.tag

    # ---------- main accuracy table ----------
    rows = []
    for ds in ["cifar10", "cifar100"]:
        tag_ds = tag10 if ds == "cifar10" else args.tag
        for setting in SETTINGS:
            wgt_name = f"weighted_summary_{tag_ds}.json" if tag_ds else "weighted_summary.json"
            ours = _accs(root, "estimator", f"{ds}/{setting}", wgt_name)
            ce = _accs(root, "traces", f"{ds}/{setting}", "ce_summary.json")
            methods = {"ce": ce, "ours": ours}
            for m in METHOD_ORDER[2:]:
                files = [
                    _json(root / "baselines" / setting / f"seed{s}" / m / f"summary_{h}.json")
                    for s in SEEDS
                    for h in [""]
                ]
                accs = _accs(root / "baselines", setting, m, "x")  # placeholder
                # proper: baselines live under results/baselines/<setting>/seed<s>/<method>/summary_*.json
                accs = []
                for s in SEEDS:
                    mdir = root / "baselines" / ds / setting / f"seed{s}" / m
                    if mdir.exists():
                        fs = list(mdir.glob("summary_*.json"))
                        if fs:
                            d = _json(fs[0])
                            if d and d.get("test_accuracy") is not None:
                                accs.append(float(d["test_accuracy"]))
                methods[m] = accs
            for m in METHOD_ORDER:
                rows.append({"dataset": ds, "setting": setting, "method": m, "acc": _fmt(methods.get(m, []))})
    df = pd.DataFrame(rows)
    main_tbl = df.pivot_table(index=["method"], columns=["dataset", "setting"], values="acc", aggfunc="first")
    main_tbl = main_tbl.reindex([m for m in METHOD_ORDER if m in main_tbl.index])
    with open(out_dir / "main_table.md", "w") as f:
        f.write("# Test accuracy (mean ± std over seeds)\n\n")
        f.write(main_tbl.to_string() + "\n")
    print(f"[paper_tables] main_table.md ({len(df)} rows)")

    # ---------- Gate B ----------
    gb_rows = []
    for setting in SETTINGS:
        est_name = f"estimator_summary_{tag10}.json" if tag10 else "estimator_summary.json"
        wgt_name = f"weighted_summary_{tag10}.json" if tag10 else "weighted_summary.json"
        for seed in SEEDS:
            est = _json(root / "estimator" / "cifar10" / setting / f"seed{seed}" / est_name)
            wgt = _json(root / "estimator" / "cifar10" / setting / f"seed{seed}" / wgt_name)
            ce = _json(root / "traces" / "cifar10" / setting / f"seed{seed}" / "ce_summary.json")
            if est and wgt and ce:
                gb_rows.append(
                    {
                        "setting": setting,
                        "det_auroc": est.get("auroc"),
                        "hc_fss": est.get("hard_clean_false_suppression"),
                        "fer": est.get("false_exclusion_rate"),
                        "ours": wgt.get("test_accuracy"),
                        "ce": ce.get("test_accuracy"),
                        "delta": wgt.get("test_accuracy") - ce.get("test_accuracy"),
                    }
                )
    gb = pd.DataFrame(gb_rows)
    if gb.empty:
        print("[paper_tables] gate_b: no data yet")
    else:
        gb = gb.groupby("setting").agg(
            det_auroc=("det_auroc", "mean"), hc_fss=("hc_fss", "mean"), fer=("fer", "mean"),
            ours=("ours", "mean"), ce=("ce", "mean"), delta=("delta", "mean"),
        )
        with open(out_dir / "gate_b.md", "w") as f:
            f.write("# Gate B (three-way, mean over seeds)\n\n")
            f.write(gb.round(3).to_string() + "\n")
        print("[paper_tables] gate_b.md")

    # ---------- detectability (Gate A) ----------
    det_rows = []
    for setting in SETTINGS:
        for seed in SEEDS:
            d = _json(root / "detectability" / "cifar10" / setting / f"seed{seed}" / "global_summary.json")
            if d:
                det_rows.append(
                    {"setting": setting, "auc_global": d["auc_global"], "auc_matched": d["auc_matched"], "delta": d["delta_conf"]}
                )
    det = pd.DataFrame(det_rows)
    if det.empty:
        print("[paper_tables] detectability: no data yet")
    else:
        det = det.groupby("setting").agg(
            auc_global=("auc_global", "mean"), auc_matched=("auc_matched", "mean"), delta=("delta", "mean")
        )
        with open(out_dir / "detectability.md", "w") as f:
            f.write("# Gate A: global vs matched-quality detectability\n\n")
            f.write(det.round(3).to_string() + "\n")
        print("[paper_tables] detectability.md")

    # ---------- ablation ----------
    abl_rows = []
    for abl in ["abl_hardfrac025", "abl_hardfrac075", "abl_soft", "abl_rel_const", "abl_ev_loss", "abl_ev_conflict", "abl_ev_forget"]:
        accs = _accs(root, "estimator", "cifar10/symmetric0.2", f"weighted_summary_{abl}.json")
        if accs:
            abl_rows.append({"ablation": abl, "acc_sym02": _fmt(accs)})
    if abl_rows:
        abl_df = pd.DataFrame(abl_rows)
        with open(out_dir / "ablation.md", "w") as f:
            f.write("# Ablations (CIFAR-10 sym0.2)\n\n")
            f.write(abl_df.to_string(index=False) + "\n")
        print("[paper_tables] ablation.md")

    # ---------- representation ----------
    rep_rows = []
    for setting in SETTINGS:
        for seed in SEEDS:
            d = _json(root / "detectability" / "cifar10" / setting / f"seed{seed}" / "representation_resnet18.json")
            if d:
                rep_rows.append(
                    {
                        "setting": setting,
                        "d_native": d["delta_conf_native"],
                        "d_pretrained": d["delta_conf_pretrained"],
                        "gap": d["delta_conf_gap"],
                    }
                )
    if rep_rows:
        rep = pd.DataFrame(rep_rows).groupby("setting").agg(
            d_native=("d_native", "mean"), d_pretrained=("d_pretrained", "mean"), gap=("gap", "mean")
        )
        with open(out_dir / "representation.md", "w") as f:
            f.write("# Representation dependence (native vs frozen pretrained)\n\n")
            f.write(rep.round(3).to_string() + "\n")
        print("[paper_tables] representation.md")

    print("[paper_tables] done")


if __name__ == "__main__":
    main()
