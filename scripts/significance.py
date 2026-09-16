#!/usr/bin/env python
"""significance.py — paired significance tests for the paper.

For each dataset x noise setting, tests Ours (final method) against CE and
against each available baseline across the 3 seeds (paired by seed):

    - paired t-test (param, n=3 seeds -> 2 df, weak but standard)
    - Wilcoxon signed-rank (non-param; needs >=5 pairs for exact p, so it is
      reported only as a sanity signal at n=3)

Outputs paper_tables/significance.md.
Usage:  python scripts/significance.py --root results --tag final_hard50 --tag-cifar10 gate_a_v3_hard50
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

SETTINGS = ["symmetric0.2", "symmetric0.4", "asymmetric0.2", "asymmetric0.4"]
SEEDS = [0, 1, 2]
BASELINES = ["small_loss", "coteaching", "coteaching_plus", "jocor", "elr", "dividemix"]


def _load(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _accs(root: Path, sub: str, ds: str, setting: str, fname: str) -> list[float]:
    out = []
    for s in SEEDS:
        d = _load(root / sub / ds / setting / f"seed{s}" / fname)
        if d and d.get("test_accuracy") is not None:
            out.append(float(d["test_accuracy"]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results")
    ap.add_argument("--tag", default="final_hard50")
    ap.add_argument("--tag-cifar10", default=None)
    args = ap.parse_args()
    root = Path(args.root)
    tag10 = args.tag_cifar10 or args.tag
    out_dir = Path("paper_tables")
    out_dir.mkdir(exist_ok=True)

    lines = ["# Significance tests (paired across seeds, n=3)\n",
             "p_t = paired t-test two-sided; p_w = Wilcoxon signed-rank.",
             "Bold p<0.05.\n"]
    rows = []
    for ds in ["cifar10", "cifar100"]:
        tag = tag10 if ds == "cifar10" else args.tag
        for setting in SETTINGS:
            ours = np.array(
                _accs(root, "estimator", ds, setting,
                      f"weighted_summary_{tag}.json" if tag else "weighted_summary.json")
            )
            ce = np.array(_accs(root, "traces", ds, setting, "ce_summary.json"))
            if len(ours) < 3 or len(ce) < 3:
                continue
            for name, ref in [("CE", ce)] + [(m, None) for m in BASELINES]:
                if name != "CE":
                    vals = []
                    for s in SEEDS:
                        mdir = root / "baselines" / ds / setting / f"seed{s}" / name
                        if mdir.exists():
                            fs = list(mdir.glob("summary_*.json"))
                            if fs:
                                d = _load(fs[0])
                                if d and d.get("test_accuracy") is not None:
                                    vals.append(float(d["test_accuracy"]))
                    ref = np.array(vals) if len(vals) == 3 else None
                if ref is None:
                    continue
                diff = ours - ref
                if diff.std() == 0:
                    t_stat, p_t = 0.0, 1.0
                else:
                    t_stat, p_t = stats.ttest_rel(ours, ref)
                try:
                    _, p_w = stats.wilcoxon(ours, ref)
                except ValueError:
                    p_w = float("nan")
                rows.append({
                    "dataset": ds, "setting": setting, "vs": name,
                    "ours": round(float(ours.mean()), 3),
                    "ref": round(float(ref.mean()), 3),
                    "delta": round(float(diff.mean()), 3),
                    "p_t": round(float(p_t), 4),
                    "p_w": round(float(p_w), 4) if not np.isnan(p_w) else "n/a",
                })
                mark = " **" if p_t < 0.05 else ""
                lines.append(f"| {ds} | {setting} | {name} | {ours.mean():.3f} | {ref.mean():.3f} "
                             f"| {diff.mean():+.3f} | {p_t:.3f}{mark} | {p_w if np.isnan(p_w) else round(float(p_w), 3)} |")

    df = pd.DataFrame(rows)
    if df.empty:
        print("[significance] no paired data yet")
        return
    with open(out_dir / "significance.md", "w") as f:
        f.write("# Significance\n\n")
        f.write(df.to_string(index=False) + "\n")
    print("[significance] wrote", len(df), "rows")


if __name__ == "__main__":
    main()
