"""governance_baselines_final.py — restore existing governance baseline numbers.

Reads baseline JSONs from the local/offline result tree
    server_results/results/baselines/<dataset>/<noise>/seed<seed>/<method>/*.json
and produces
    governance_baselines_final.md

Only reports what exists on disk; missing (dataset, noise, method, seed)
combinations are marked, never fabricated.

Usage:
    python scripts/governance_baselines_final.py [--root server_results/results/baselines]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SETTINGS = [
    ("cifar10", "symmetric0.2", "C10-S20"),
    ("cifar10", "symmetric0.4", "C10-S40"),
    ("cifar10", "asymmetric0.2", "C10-A20"),
    ("cifar10", "asymmetric0.4", "C10-A40"),
    ("cifar100", "symmetric0.2", "C100-S20"),
    ("cifar100", "symmetric0.4", "C100-S40"),
    ("cifar100", "asymmetric0.2", "C100-A20"),
    ("cifar100", "asymmetric0.4", "C100-A40"),
]

METHODS = ["ce", "small_loss", "coteaching", "coteaching_plus", "jocor", "elr"]


def _acc_from_json(p: Path) -> float | None:
    try:
        with open(p) as fh:
            d = json.load(fh)
        acc = d.get("test_accuracy")
        if acc is None:
            acc = d.get("best_val_acc")
        return float(acc) if acc is not None else None
    except Exception:
        return None


def _ce_accs(root: Path, ds: str, ns: str) -> list[float]:
    """CE summaries live under results/traces/<ds>/<ns>/seed<seed>/ce_summary.json."""
    base = root / "results" / "traces" / ds / ns
    if not base.exists():
        return []
    accs = []
    for sd in sorted(base.iterdir()):
        if not sd.is_dir() or not sd.name.startswith("seed"):
            continue
        for jf in sd.glob("ce_summary*.json"):
            a = _acc_from_json(jf)
            if a is not None:
                accs.append(a)
    return accs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path,
                    default=Path("server_results"))
    args = ap.parse_args()

    root = args.root
    if not root.exists():
        print(f"ERROR: {root} not found.", flush=True)
        sys.exit(1)

    baselines_root = root / "results" / "baselines"
    if not baselines_root.exists():
        print(f"ERROR: {baselines_root} not found.", flush=True)
        sys.exit(1)

    lines = [
        "# Governance Baselines — Final Summary (deliverable G)\n",
        "Recovered from on-disk baseline runs. Missing cells are marked, never "
        "fabricated.\n",
    ]

    # Coverage table
    lines.append("## Coverage check\n")
    header = ["Dataset", "Noise", "CE", "small_loss", "Co-teaching", "Co-teaching+", "JoCoR", "ELR"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    coverage = {}
    for ds, ns, lbl in SETTINGS:
        base = baselines_root / ds / ns
        row = [lbl]
        for m in METHODS:
            cnt = 0
            accs = []
            if m == "ce":
                accs = _ce_accs(root, ds, ns)
                cnt = len(accs)
            elif base.exists():
                for sd in sorted(base.iterdir()):
                    if not sd.is_dir() or not sd.name.startswith("seed"):
                        continue
                    mdir = sd / m
                    if mdir.exists():
                        for jf in mdir.glob("*.json"):
                            a = _acc_from_json(jf)
                            if a is not None:
                                cnt += 1
                                accs.append(a)
            coverage[(ds, ns, m)] = (cnt, accs)
            if cnt > 0:
                mu = sum(accs) / len(accs)
                row.append(f"{cnt}s ({mu:.3f})")
            else:
                row.append("—")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # Accuracy matrix by method (mean over available seeds)
    lines.append("## Test accuracy (mean over available seeds)\n")
    header2 = ["Method"] + [lbl for _, _, lbl in SETTINGS]
    lines.append("| " + " | ".join(header2) + " |")
    lines.append("|" + "|".join(["---"] * len(header2)) + "|")
    for m in METHODS:
        row = [m]
        for ds, ns, _ in SETTINGS:
            cnt, accs = coverage[(ds, ns, m)]
            if cnt > 0:
                row.append(f"{sum(accs)/len(accs):.3f} (n={cnt})")
            else:
                row.append("—")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # Notes on gaps (esp. Co-teaching+/JoCoR/ELR on CIFAR-100)
    lines.append("## Notes\n")
    for m in ["coteaching_plus", "jocor", "elr"]:
        present = [(ds, ns, lbl) for ds, ns, lbl in SETTINGS
                   if coverage[(ds, ns, m)][0] > 0]
        absent = [(ds, ns, lbl) for ds, ns, lbl in SETTINGS
                  if coverage[(ds, ns, m)][0] == 0]
        if absent:
            lines.append(f"- **{m}**: present on "
                         f"{', '.join(lbl for _, _, lbl in present) or '—'}; "
                         f"**absent on {', '.join(lbl for _, _, lbl in absent)}** "
                         f"(not recovered / not run — do not fabricate).")
        else:
            lines.append(f"- **{m}**: complete across all settings.")

    out = Path("paper_tables/governance_baselines_final.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out}", flush=True)

    # Gate check print
    print("\n== Baseline coverage ==", flush=True)
    for m in METHODS:
        total = sum(1 for ds, ns, _ in SETTINGS if coverage[(ds, ns, m)][0] > 0)
        print(f"  {m:14s}: {total}/{len(SETTINGS)} settings with data", flush=True)


if __name__ == "__main__":
    main()
