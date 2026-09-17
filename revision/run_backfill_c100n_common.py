"""Backfill the common-support coverage columns for the C100N human-noise audit.

E9's `evaluate()` wrote `noisy_coverage` / `clean_coverage` with *matched* values
(the fraction of each stratum actually paired after the caliper).  The unified
schema needs those under `matched_*` and additionally wants the common-support
fractions under `common_*`.  The matched pair can be renamed offline; the
common-support pair needs the prepared cache, so it is computed here.

Common-support coverage for a proxy Q is

    fraction of the noisy stratum with  q in [max(min q_noisy, min q_clean),
                                             min(max q_noisy, max q_clean)]

and likewise for the clean stratum.  It depends only on (Q, mask), not on the
detector, so each (seed, proxy) cell is computed once.

This runs on a 0.5-core container, so the npz for each seed is opened exactly once
and every proxy is handled in that pass -- note ``np.load`` on a compressed archive
decompresses every member on access, and the audit npz holds 22 arrays.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "revision"))

from run_p0_batch import CACHE, common_support  # noqa: E402

OUT = ROOT / "results" / "revision" / "round2"
AUDIT = OUT / "e9_c100n_audit.csv"
SETTING = ("cifar100n", "cifar100n_human0.4")
CANON = ["common_noisy_coverage", "common_clean_coverage",
         "matched_noisy_coverage", "matched_clean_coverage"]
AMBIGUOUS = ["noisy_coverage", "clean_coverage"]


def support_cov(q: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    lo, hi = common_support(q[mask], q[~mask])
    keep = (q >= lo) & (q <= hi) if np.isfinite(lo) else np.zeros(len(q), bool)
    return (float((mask & keep).sum() / max(int(mask.sum()), 1)),
            float(((~mask) & keep).sum() / max(int((~mask).sum()), 1)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", default=str(AUDIT))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    path = Path(args.audit)
    df = pd.read_csv(path)
    dataset, noise = SETTING
    proxies = sorted(df.proxy.unique())
    seeds = sorted(int(s) for s in df.seed.unique())
    print(f"== C100N common-support backfill ==")
    print(f"   {path}")
    print(f"   {len(df)} rows, seeds={seeds}, {len(proxies)} proxies")

    if {"noisy_coverage", "clean_coverage"} <= set(df.columns):
        # E9 wrote matched coverage under the short names; move it to the
        # canonical matched pair before adding the common pair.
        df["matched_noisy_coverage"] = df["noisy_coverage"]
        df["matched_clean_coverage"] = df["clean_coverage"]
        print("   renamed noisy_coverage/clean_coverage -> matched_*")
    if not {"matched_noisy_coverage", "matched_clean_coverage"} <= set(df.columns):
        raise SystemExit("audit has no matched coverage columns to anchor on")

    common: dict[tuple[int, str], tuple[float, float]] = {}
    for seed in seeds:
        f = CACHE / dataset / noise / f"seed{seed}.npz"
        if not f.exists():
            raise SystemExit(f"missing prepared cache: {f}")
        with np.load(f, allow_pickle=False) as z:
            mask = np.asarray(z["mask"], bool)
            for px in proxies:
                q = np.asarray(z[f"sig__{px}"], np.float64)
                common[(seed, px)] = support_cov(q, mask)
        c = common[(seed, proxies[0])]
        print(f"   seed{seed}: loaded, {len(proxies)} proxies computed "
              f"(first common cov {c[0]:.4f}/{c[1]:.4f})", flush=True)

    df["common_noisy_coverage"] = [common[(int(s), p)][0] for s, p in zip(df.seed, df.proxy)]
    df["common_clean_coverage"] = [common[(int(s), p)][1] for s, p in zip(df.seed, df.proxy)]
    df = df.drop(columns=[c for c in AMBIGUOUS if c in df.columns])
    df = df[[c for c in CANON if c in df.columns] + [c for c in df.columns if c not in CANON]]

    print("\n   per-proxy means:")
    print(df.groupby("proxy")[CANON].mean().round(4).to_string())

    if args.dry_run:
        print("\n   --dry-run: not written")
        return
    df.to_csv(path, index=False)
    print(f"\n   wrote {path} ({len(df)} rows, {len(df.columns)} cols)")


if __name__ == "__main__":
    main()
