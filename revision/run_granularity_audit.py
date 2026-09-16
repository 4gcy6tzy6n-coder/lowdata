"""Score-granularity audit for the conditioning audit.

The audit compares detector scores *within* a quality-matched pair, so it needs a
score whose resolution is finer than the between-sample differences it is asked to
resolve.  Two of the paper's detectors are not continuous:

* ``neighbor`` is a kNN agreement ratio, so it takes only ~11-18 distinct values;
* ``forgetting`` is a cumulative event count, so it takes only ~110-130 distinct
  values.

With 4.4k pairs drawn from a handful of score levels, the pairwise AUROC of such a
detector is dominated by ties and can land on exactly 0.5 by construction -- which
looks like a "reversal" but is an artefact of resolution, not of conditioning.

This script measures, per setting and detector: the number of distinct values, the
tie rate inside the matched noisy/clean sets, and the pairwise AUROC of the
matched sets.  Its output is what lets the results document say which detectors
the audit can and cannot speak for.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "revision"))

from run_p0_batch import CACHE, MAX_CLEAN, matched_pairs, pairwise_auc, rank_auc  # noqa: E402
from run_round2 import detector_names, get_detector  # noqa: E402

SETTINGS = [("cifar100", "symmetric0.2"),
            ("cifar100", "asymmetric0.4"),
            ("cifar100n", "cifar100n_human0.4")]
OUT = ROOT / "results" / "revision" / "round2"


def tie_rate(a: np.ndarray, b: np.ndarray) -> float:
    """Fraction of cross-pairs (a_i, b_j) that are exactly tied."""
    if not len(a) or not len(b):
        return float("nan")
    # Count ties without materialising the n_a x n_b matrix.
    vals, inv = np.unique(np.concatenate([a, b]), return_inverse=True)
    ca = np.bincount(inv[:len(a)], minlength=len(vals)).astype(np.float64)
    cb = np.bincount(inv[len(a):], minlength=len(vals)).astype(np.float64)
    return float((ca * cb).sum() / (len(a) * len(b)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=1,
                    help="unused; kept so the driver matches the others")
    ap.add_argument("--out", default="granularity_audit.csv")
    args = ap.parse_args()

    rows = []
    for dataset, noise in SETTINGS:
        for path in sorted((CACHE / dataset / noise).glob("seed*.npz")):
            seed = int(path.stem[4:])
            with np.load(path, allow_pickle=False) as z:
                p = {k: np.asarray(z[k]).copy() for k in z.files}
            mask = np.asarray(p["mask"], bool)
            zz = mask.astype(int)
            for det in detector_names():
                score = get_detector(p, det)
                uniq_all = int(len(np.unique(score)))
                row = {
                    "setting": f"{dataset}/{noise}", "seed": seed, "detector": det,
                    "n_total": int(len(score)),
                    "n_unique_all": uniq_all,
                    "n_unique_noisy": int(len(np.unique(score[mask]))),
                    "n_unique_clean": int(len(np.unique(score[~mask]))),
                    "global_auc": rank_auc(mask, score),
                }
                for px in ("knn_agreement", "proto_margin", "native_density"):
                    q = np.asarray(p[f"sig__{px}"], np.float64)
                    ni, ci = matched_pairs(score, mask, q, 0.10 * float(q.std()), MAX_CLEAN)
                    row[f"n_pairs__{px}"] = int(len(ni))
                    row[f"tie_rate__{px}"] = tie_rate(score[ni], score[ci]) if len(ni) else np.nan
                    row[f"audit_auc__{px}"] = pairwise_auc(score, ni, ci) if len(ni) else np.nan
                rows.append(row)
            print(f"   {dataset}/{noise} seed{seed} done", flush=True)

    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / args.out, index=False)

    print("\n== granularity audit: mean over seeds ==")
    g = (df.groupby(["setting", "detector"], as_index=False)
         .agg(n_unique=("n_unique_all", "mean"),
              tie_knn=("tie_rate__knn_agreement", "mean"),
              tie_proto=("tie_rate__proto_margin", "mean"),
              tie_native=("tie_rate__native_density", "mean"),
              auc_knn=("audit_auc__knn_agreement", "mean"),
              auc_proto=("audit_auc__proto_margin", "mean"),
              Ag=("global_auc", "mean")))
    print(g.round(4).to_string(index=False))
    g.to_csv(OUT / args.out.replace(".csv", "_summary.csv"), index=False)


if __name__ == "__main__":
    main()
