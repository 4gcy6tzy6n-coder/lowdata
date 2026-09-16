"""Reproduction gate for the CIFAR-100 revision reruns.

Nothing from the new CIFAR-100 runs is folded into the revised paper until the
pipeline is shown to reproduce the frozen headline numbers.  The gate compares
the *re-run* C100-S20 results against the archived ones:

    AUC_global ~ 0.697,  AUC_matched ~ 0.391,  Delta_conf ~ 0.306

Two comparisons are meaningful and both are reported:

1. **Archive cross-check (shape only).** The archived per-cell JSONs in
   ``results/cross_detector/knn_agreement/cifar100/symmetric0.2`` were produced
   by the original pipeline.  They are compared against the fresh numbers for
   the same seeds.

2. **Old-vs-new seed agreement (the real gate).** Seeds 0-4 are the archived
   ones; seeds 5-9 come from the revision rerun.  If the pipeline is
   reproducible, both halves must agree on the headline quantities: the
   seed-mean gap must have the same sign and lie within tolerance.

Tolerance: |Delta| <= TOL in AUC on the seed mean, and identical sign.
A FAIL means: stop, do not spend GPU on C100 S40/A20/A40, and re-examine the
main claim first.

Usage:
    python revision/check_reproduction_gate.py --tol 0.01
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/root/quality_noise")
CACHE = ROOT / "results" / "revision" / "prepared"
CROSS = ROOT / "results" / "cross_detector" / "knn_agreement" / "cifar100" / "symmetric0.2"

ARCHIVED = {"auc_global": 0.697, "auc_matched": 0.391, "delta_conf": 0.306}


def archived_per_seed() -> dict[int, dict]:
    out = {}
    for p in sorted(CROSS.glob("seed*/combined.json")):
        seed = int(p.parent.name[4:])
        d = json.load(open(p))
        out[seed] = {"auc_global": d["auc_global"], "auc_matched": d["auc_matched"],
                     "delta_conf": d["delta_conf"]}
    return out


def fresh_per_seed() -> dict[int, dict]:
    """Recompute the paper's combined detector + KNN-Q matched AUC from the cache."""
    from sklearn.metrics import roc_auc_score
    from revq.matching import match_pairs, matched_auc
    from revq.prepare import load_cached

    out = {}
    if not (CACHE / "cifar100" / "symmetric0.2").exists():
        return out
    for p in sorted((CACHE / "cifar100" / "symmetric0.2").glob("seed*.npz")):
        seed = int(p.name[4:-4])
        r = load_cached("cifar100", "symmetric0.2", seed, cache=CACHE)
        score = r["signals"]["combined"]
        q = r["signals"]["knn_agreement"]
        mask = r["mask"]
        auc_g = float(roc_auc_score(mask, score))
        # Archive used eps=0.1 on raw Q with the legacy greedy-with-replacement rule.
        pairs = match_pairs(score, mask, q, strategy="nn_wr", eps=0.1, max_clean_per_noisy=5)
        auc_m = matched_auc(pairs)
        out[seed] = {"auc_global": auc_g, "auc_matched": auc_m,
                     "delta_conf": auc_g - auc_m, "n_pairs": int(len(pairs.s_noisy))}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tol", type=float, default=0.01,
                    help="max |Delta AUC| on the seed mean for the gate to pass")
    ap.add_argument("--json-out", type=Path,
                    default=ROOT / "results" / "revision" / "reproduction_gate.json")
    args = ap.parse_args()

    arch = archived_per_seed()
    fresh = fresh_per_seed()

    print("== CIFAR-100 S20 reproduction gate ==")
    print(f"archived seeds: {sorted(arch)}   fresh seeds: {sorted(fresh)}")

    report: dict = {"archived_headline": ARCHIVED, "tol": args.tol}
    if not fresh:
        print("\nFAIL: no fresh C100-S20 runs prepared yet.")
        report["status"] = "FAIL"
        report["reason"] = "no fresh runs"
        _write(args.json_out, report)
        sys.exit(1)

    keys = ["auc_global", "auc_matched", "delta_conf"]

    # ---- 1. archive cross-check on the overlapping seeds --------------------
    common = sorted(set(arch) & set(fresh))
    print(f"\n[A] per-seed archive vs fresh on seeds {common}")
    print(f"{'seed':>5s} {'metric':14s} {'archive':>9s} {'fresh':>9s} {'diff':>9s}")
    per_seed_diffs = {k: [] for k in keys}
    for s in common:
        for k in keys:
            d = fresh[s][k] - arch[s][k]
            per_seed_diffs[k].append(d)
            print(f"{s:>5d} {k:14s} {arch[s][k]:9.4f} {fresh[s][k]:9.4f} {d:+9.4f}")
    report["archive_cross_check"] = {
        "seeds": common,
        "mean_diff": {k: float(np.mean(v)) for k, v in per_seed_diffs.items()},
        "max_abs_diff": {k: float(np.max(np.abs(v))) for k, v in per_seed_diffs.items()},
    }

    # ---- 2. the actual gate: old half vs new half ---------------------------
    old = [s for s in sorted(fresh) if s <= 4]
    new = [s for s in sorted(fresh) if s >= 5]
    print(f"\n[B] headline comparison: seeds<=4 (n={len(old)}) vs seeds>=5 (n={len(new)})")
    gate: dict = {}
    ok = True
    for k in keys:
        if not old or not new:
            gate[k] = {"old_mean": float(np.mean([fresh[s][k] for s in old])) if old else None,
                       "new_mean": float(np.mean([fresh[s][k] for s in new])) if new else None,
                       "abs_diff": None, "sign_match": None, "pass": None}
            continue
        mo = float(np.mean([fresh[s][k] for s in old]))
        mn = float(np.mean([fresh[s][k] for s in new]))
        diff = abs(mo - mn)
        sign_match = (np.sign(mo) == np.sign(mn))
        passed = bool(diff <= args.tol and sign_match)
        gate[k] = {"old_mean": mo, "new_mean": mn, "abs_diff": diff,
                   "sign_match": bool(sign_match), "pass": passed}
        ok = ok and passed
        print(f"   {k:14s} old={mo:8.4f} new={mn:8.4f} |diff|={diff:.4f} "
              f"sign_match={sign_match} -> {'PASS' if passed else 'FAIL'}")

    # ---- 3. distance from the archived headline ----------------------------
    headline = {}
    for k in keys:
        m = float(np.mean([fresh[s][k] for s in sorted(fresh)]))
        headline[k] = {"fresh_mean": m, "archived": ARCHIVED[k], "diff": m - ARCHIVED[k]}
        print(f"   headline {k:14s} fresh={m:8.4f} archived={ARCHIVED[k]:8.4f} "
              f"diff={m - ARCHIVED[k]:+.4f}")
    report["headline"] = headline

    report["gate"] = gate
    report["status"] = "PASS" if ok else "FAIL"
    report["n_fresh_seeds"] = len(fresh)
    report["frozen_seeds"] = {"old": old, "new": new}
    _write(args.json_out, report)

    print(f"\nGATE: {report['status']}")
    if report["status"] == "PASS":
        print("  -> pipeline reproduces the archived headline; clearing C100 S40/A20/A40 reruns.")
    else:
        print("  -> DO NOT spend GPU on C100 S40/A20/A40. Re-examine the main claim first.")


def _write(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
