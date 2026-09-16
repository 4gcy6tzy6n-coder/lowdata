#!/usr/bin/env python
"""gen_final_tables.py — emit the 5 final tables required by the study plan.

  Table 1  main detectability finding   (dataset x noise: global/matched/delta + CI)
  Table 2  representation robustness    (native vs frozen pretrained)
  Table 3  governance performance       (CE / small-loss / Co-teaching / Ours)
  Table 4  detectability vs governance  (matched AUC vs soft/hard delta-acc)
  Table 5  ablation / oracle check      (estimated-vs-oracle, no calibration,
                                         soft only, hard25/50, constant R)

Reads from a results root (default server_results/results) and writes into
final_tables/*.md. Missing data is marked '—' so the script is safe to run
before the CIFAR-10N run completes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

SETTINGS = ["symmetric0.2", "symmetric0.4", "asymmetric0.2", "asymmetric0.4"]
SEEDS = [0, 1, 2, 3, 4]


def _json(p: Path) -> dict | None:
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def _mean_key(root: Path, sub: str, ds: str, setting: str, fname: str, key: str, seeds: list[int] = SEEDS) -> float | None:
    vals = [_json(root / sub / ds / setting / f"seed{s}" / fname) for s in seeds]
    k = [v[key] for v in vals if v and v.get(key) is not None]
    return float(np.mean(k)) if k else None


def _fmt(v: float | None) -> str:
    return f"{v:.3f}" if v is not None else "—"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="server_results/results")
    ap.add_argument("--out", default="final_tables")
    ap.add_argument("--tag-hard", default="gate_a_v3_hard50")
    ap.add_argument("--tag-soft", default="v2_soft")
    args = ap.parse_args()
    root = Path(args.root)
    out = Path(args.out)
    out.mkdir(exist_ok=True)

    # ---------- Table 1: main detectability ----------
    lines1 = ["# Table 1 — Main detectability finding\n",
              "| Dataset | Noise | Rate | Global AUC | Matched AUC | Δ_conf |",
              "|---|---|---|---|---|---|"]
    det_rows = []
    for ds in ["cifar10", "cifar100", "cifar10n"]:
        settings_for_ds = ["symmetric0.2", "symmetric0.4", "asymmetric0.2", "asymmetric0.4"] if ds != "cifar10n" else ["aggregate", "worse"]
        for setting in settings_for_ds:
            dset = ds
            fname = "global_summary.json"
            g = _mean_key(root, "detectability", dset, setting, fname, "auc_global")
            m = _mean_key(root, "detectability", dset, setting, fname, "auc_matched")
            d = _mean_key(root, "detectability", dset, setting, fname, "delta_conf")
            if g is None:
                continue
            lines1.append(f"| {dset} | {setting} | — | {_fmt(g)} | {_fmt(m)} | {_fmt(d)} |")
            det_rows.append((dset, setting, g, m, d))
    with open(out / "table1_detectability.md", "w") as f:
        f.write("\n".join(lines1) + "\n")
    print("[gen_final] table1_detectability.md")

    # ---------- Table 2: representation robustness ----------
    lines2 = ["# Table 2 — Representation robustness (Δ_conf)",
              "\n| Representation | Global | Matched | Δ_conf |",
              "|---|---|---|---|"]
    for ds in ["cifar10"]:
        for setting in SETTINGS:
            native = _mean_key(root, "detectability", ds, setting, "global_summary.json", "delta_conf")
            rep = _mean_key(root, "detectability", ds, setting, "representation_resnet18.json", "delta_conf_pretrained")
            rep_g = _mean_key(root, "detectability", ds, setting, "representation_resnet18.json", "auc_global_pretrained")
            rep_m = _mean_key(root, "detectability", ds, setting, "representation_resnet18.json", "auc_matched_pretrained")
            if rep is None:
                continue
            lines2.append(f"| {setting} (native) | {_fmt(_mean_key(root, 'detectability', ds, setting, 'global_summary.json', 'auc_global'))} | {_fmt(_mean_key(root, 'detectability', ds, setting, 'global_summary.json', 'auc_matched'))} | {_fmt(native)} |")
            lines2.append(f"| {setting} (pretrained) | {_fmt(rep_g)} | {_fmt(rep_m)} | {_fmt(rep)} |")
    with open(out / "table2_representation.md", "w") as f:
        f.write("\n".join(lines2) + "\n")
    print("[gen_final] table2_representation.md")

    # ---------- Table 3: governance performance (CIFAR-10) ----------
    lines3 = ["# Table 3 — Governance performance (CIFAR-10, mean accuracy)",
              "\n| Method | sym20 | sym40 | asym20 | asym40 |",
              "|---|---|---|---|---|"]
    def acc_for(method: str) -> list[str]:
        row = []
        for setting in SETTINGS:
            if method == "ce":
                for s in SEEDS:
                    if (root / "traces" / "cifar10" / setting / f"seed{s}" / "ce_summary.json").exists():
                        row.append(_fmt(_mean_key(root, "traces", "cifar10", setting, "ce_summary.json", "test_accuracy")))
                        break
                else:
                    row.append("—")
            elif method == "ours":
                fname = f"weighted_summary_{args.tag_hard}.json"
                row.append(_fmt(_mean_key(root, "estimator", "cifar10", setting, fname, "test_accuracy")))
            else:  # baseline dir
                row.append(_fmt(_mean_key(root, "baselines", "cifar10", setting, "x", "test_accuracy")))
        return row
    # baseline under results/baselines/cifar10/<setting>/seed<s>/<method>/summary_*.json
    def baseline_acc(method: str) -> list[str]:
        row = []
        for setting in SETTINGS:
            vals = []
            for s in SEEDS:
                mdir = root / "baselines" / "cifar10" / setting / f"seed{s}" / method
                if mdir.exists():
                    fs = list(mdir.glob("summary_*.json"))
                    if fs:
                        d = _json(fs[0])
                        if d and d.get("test_accuracy") is not None:
                            vals.append(float(d["test_accuracy"]))
            row.append(f"{np.mean(vals):.3f}" if vals else "—")
        return row
    lines3.append("| CE | " + " | ".join(acc_for("ce")) + " |")
    lines3.append("| Ours (hard50) | " + " | ".join(acc_for("ours")) + " |")
    for m in ["small_loss", "coteaching"]:
        lines3.append(f"| {m} | " + " | ".join(baseline_acc(m)) + " |")
    with open(out / "table3_governance.md", "w") as f:
        f.write("\n".join(lines3) + "\n")
    print("[gen_final] table3_governance.md")

    # ---------- Table 4: detectability vs governance ----------
    lines4 = ["# Table 4 — Detectability vs governance",
              "\n| Setting | Matched AUC | Soft ΔAcc | Hard ΔAcc | Preferred |",
              "|---|---|---|---|---|"]
    for ds, settings in [("cifar10", SETTINGS), ("cifar100", SETTINGS)]:
        for setting in settings:
            matched = _mean_key(root, "detectability", ds, setting, "global_summary.json", "auc_matched")
            soft = _mean_key(root, "traces", ds, setting, "ce_summary.json", "test_accuracy")
            soft_m = _mean_key(root, "estimator", ds, setting, f"weighted_summary_{args.tag_soft}.json", "test_accuracy")
            hard_m = _mean_key(root, "estimator", ds, setting, f"weighted_summary_{args.tag_hard}.json", "test_accuracy")
            # fall back per-dataset tag for cifar100 (final_hard50)
            if hard_m is None and ds == "cifar100":
                hard_m = _mean_key(root, "estimator", ds, setting, "weighted_summary_final_hard50.json", "test_accuracy")
            if matched is None:
                continue
            soft_d = (soft_m - soft) if (soft_m is not None and soft is not None) else None
            hard_d = (hard_m - soft) if (hard_m is not None and soft is not None) else None
            pref = "hard" if (hard_d is not None and soft_d is not None and hard_d > soft_d) else ("soft" if hard_d is not None and soft_d is not None and hard_d < soft_d else "—")
            lines4.append(f"| {ds} {setting} | {_fmt(matched)} | {_fmt(soft_d)} | {_fmt(hard_d)} | {pref} |")
    with open(out / "table4_detectability_governance.md", "w") as f:
        f.write("\n".join(lines4) + "\n")
    print("[gen_final] table4_detectability_governance.md")

    # ---------- Table 5: ablation / oracle ----------
    lines5 = ["# Table 5 — Ablation / oracle check (CIFAR-10 sym20)", ""]
    ablations = {
        "oracle (rate=config, hard50)": f"weighted_summary_{args.tag_hard}.json",
        "estimated (Otsu ρ̂)": "weighted_summary_final_estimated.json",
        "no calibration (soft)": f"weighted_summary_{args.tag_soft}.json",
        "hard25": "weighted_summary_abl_hardfrac025.json",
        "hard50": "weighted_summary_abl_hardfrac025.json",
        "constant R": "weighted_summary_abl_rel_const.json",
    }
    lines5.append("| Variant | acc |")
    lines5.append("|---|---|")
    seen = {}
    for name, fname in ablations.items():
        acc = _mean_key(root, "estimator", "cifar10", "symmetric0.2", fname, "test_accuracy")
        if acc is not None:
            lines5.append(f"| {name} | {acc:.3f} |")
    with open(out / "table5_ablation_oracle.md", "w") as f:
        f.write("\n".join(lines5) + "\n")
    print("[gen_final] table5_ablation_oracle.md")

    print("[gen_final] done ->", out)


if __name__ == "__main__":
    main()
