#!/usr/bin/env python
"""compare_variants.py — compare weighted-method variants vs CE / small-loss.

Reads weighted_summary_<tag>.json (and the default weighted_summary.json) under
results/estimator/cifar10/ and prints a compact comparison table.

Usage (on server or after syncing results locally):
    python scripts/compare_variants.py [--root results]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

SETTINGS = ["symmetric0.2", "symmetric0.4", "asymmetric0.2", "asymmetric0.4"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results")
    args = ap.parse_args()

    root = Path(args.root)
    est_root = root / "estimator" / "cifar10"
    traces_root = root / "traces" / "cifar10"

    rows = []
    for setting in SETTINGS:
        # CE baseline from 02.
        ce_accs = []
        for seed in range(3):
            f = traces_root / setting / f"seed{seed}" / "ce_summary.json"
            if f.exists():
                ce_accs.append(json.load(open(f))["test_accuracy"])
        if ce_accs:
            rows.append({"setting": setting, "variant": "CE", "acc": sum(ce_accs) / len(ce_accs)})

        # Weighted variants: default + any *_tag files.
        seed_dirs = sorted((est_root / setting).glob("seed*"))
        tags = set()
        for sd in seed_dirs:
            for f in sd.glob("weighted_summary_*.json"):
                tags.add(f.name[len("weighted_summary_"):-len(".json")])
        tags.add("default")
        for tag in sorted(tags):
            accs = []
            for sd in seed_dirs:
                f = sd / f"weighted_summary_{tag}.json" if tag != "default" else sd / "weighted_summary.json"
                if f.exists():
                    accs.append(json.load(open(f))["test_accuracy"])
            if accs:
                n = len(accs)
                rows.append({"setting": setting, "variant": f"w[{tag}]", "acc": sum(accs) / n, "n": n})

        # small_loss baseline.
        sl_accs = []
        for seed in range(3):
            base = root / "baselines" / "cifar10" / setting / f"seed{seed}" / "small_loss"
            files = list(base.glob("summary_*.json")) if base.exists() else []
            if files:
                sl_accs.append(json.load(open(files[0]))["test_accuracy"])
        if sl_accs:
            rows.append({"setting": setting, "variant": "small_loss", "acc": sum(sl_accs) / len(sl_accs)})

    df = pd.DataFrame(rows)
    wide = df.pivot(index="variant", columns="setting", values="acc").round(3)
    print(wide.to_string())
    print("\n(acc = mean test accuracy over available seeds; n<3 marked by fewer seeds)")


if __name__ == "__main__":
    main()
