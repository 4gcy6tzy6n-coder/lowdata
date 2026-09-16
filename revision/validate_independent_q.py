"""Validation checks for the label-independent-Q result.

Before reporting that the C100-S20 reversal collapses under a label-free
quality covariate, rule out the obvious failure modes:

V1. The random-Q control must give Delta_Q ~ 0 (matching on noise cannot move
    the estimate).  A non-zero value means the matching machinery itself is
    biased.
V2. Q^DINO must be a usable covariate: finite overlap, adequate coverage, SMD
    removed, and real spread.
V3. The collapse must be attributable to the *covariate*, not to a pipeline
    difference: Q^KNN and Q^DINO go through identical code, identical caliper,
    identical matching rule.
V4. Report how much information the two covariates carry about the noise mask;
    a covariate that already ranks noise cannot be a clean control.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path("/root/quality_noise")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "revision"))

from revq.matching import match_pairs, matched_auc  # noqa: E402
from revq.prepare import load_cached  # noqa: E402

DATASET, NOISE = "cifar100", "symmetric0.2"


def main() -> None:
    seeds = sorted(int(p.name[4:-4]) for p in
                   (ROOT / "results" / "revision" / "prepared" / DATASET / NOISE).glob("seed*.npz"))
    print(f"== validation on {DATASET}/{NOISE}, seeds {seeds} ==\n")

    rows = []
    for seed in seeds:
        r = load_cached(DATASET, NOISE, seed)
        mask = r["mask"]
        score = r["signals"]["combined"]
        auc_g = float(roc_auc_score(mask, score))

        row = {"seed": seed, "auc_global": auc_g,
               "noise_rate": float(mask.mean())}
        for qname in ["knn_agreement", "dino_density", "proto_margin",
                      "native_density", "random"]:
            q = r["signals"][qname]
            cal = 0.10 * np.std(q)
            pairs = match_pairs(score, mask, q, strategy="nn_wo", eps=cal)
            row[f"auc_qc_{qname}"] = matched_auc(pairs)
            row[f"dq_{qname}"] = auc_g - matched_auc(pairs)
            row[f"cov_{qname}"] = len(np.unique(pairs.n_idx)) / max(mask.sum(), 1)
            row[f"smd_after_{qname}"] = (
                (pairs.q_noisy.mean() - pairs.q_clean.mean()) /
                max(np.sqrt((pairs.q_noisy.var() + pairs.q_clean.var()) / 2), 1e-12)
                if len(pairs.q_noisy) else np.nan)
            # V4: information the covariate itself carries about the noise mask
            row[f"qauc_{qname}"] = float(roc_auc_score(mask, q))
            # V2: usable spread
            row[f"qsd_{qname}"] = float(np.std(q))
            row[f"qrange_{qname}"] = float(np.ptp(q))
        rows.append(row)

    df = pd.DataFrame(rows)
    pd.set_option("display.width", 250)

    print("V1/V3 — Delta_Q by covariate (per seed)")
    cols = ["seed", "auc_global"] + [f"dq_{q}" for q in
                                     ["knn_agreement", "dino_density", "proto_margin",
                                      "native_density", "random"]]
    print(df[cols].round(4).to_string(index=False))
    print()
    print("   seed-mean Delta_Q:")
    for q in ["knn_agreement", "dino_density", "proto_margin", "native_density", "random"]:
        c = df[f"dq_{q}"]
        print(f"     {q:18s} {c.mean():+.4f}  (sd {c.std(ddof=1):.4f}, "
              f"{int((c > 0).sum())}/{len(c)} positive)")
    print()
    print("V2 — covariate usability (seed-mean)")
    for q in ["knn_agreement", "dino_density", "proto_margin", "native_density", "random"]:
        print(f"     {q:18s} coverage={df[f'cov_{q}'].mean():.3f}  "
              f"SMD_after={df[f'smd_after_{q}'].mean():+.5f}  "
              f"SD={df[f'qsd_{q}'].mean():.4f}  range={df[f'qrange_{q}'].mean():.4f}")
    print()
    print("V4 — information each covariate carries about the noise mask (AUC, 0.5 = none)")
    for q in ["knn_agreement", "dino_density", "proto_margin", "native_density", "random"]:
        print(f"     {q:18s} AUC(Q vs mask) = {df[f'qauc_{q}'].mean():.4f}")

    out = ROOT / "results" / "revision" / "validation_c100s20.csv"
    df.to_csv(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
