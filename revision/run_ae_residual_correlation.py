"""A2 (minor revision): residual dependence between Combined-noN and Q_agree.

The reviewers accept that dropping the explicit `neighbor` term removes the *exact*
construction coupling E_neighbor = 1 - Q_knn, but ask whether residual statistical
dependence remains, since loss and forgetting are themselves produced by
noisy-label training.

This script measures that dependence directly on the ten C100-S20 seeds, for the
six primary detectors against `Q_knn_agreement`:

  * Spearman rho  -- monotone dependence
  * Pearson r     -- linear dependence
  * kNN mutual information I(E; Q) -- full dependence, with the Pearson/Spearman gap
    separating "straight line" from "monotone but curved"

A per-seed value is reported alongside the mean +- SD and a seed-bootstrap 95% CI,
because ten seeds is the resampling unit throughout this revision.

No new training: the detector scores and the proxy are read from the prepared
C100-S20 cache.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "revision"))

from run_p0_batch import CACHE  # noqa: E402
from run_round2 import get_detector  # noqa: E402

OUT = ROOT / "results" / "revision" / "round2"
SETTING = ("cifar100", "symmetric0.2")
DETS = ["combined_noN", "combined", "ema_loss", "confidence", "aum",
        "confident_learning"]
DISPLAY = {"combined_noN": "Combined-noN", "combined": "Combined",
           "ema_loss": "EMA Loss", "confidence": "Confidence", "aum": "AUM",
           "confident_learning": "CL"}
PROXY = "knn_agreement"
B_SEED = 10000
RNG_SEED = 20260917


def main() -> None:
    from scipy.stats import spearmanr, pearsonr
    from sklearn.feature_selection import mutual_info_classif

    ap = argparse.ArgumentParser()
    ap.add_argument("--proxy", default=PROXY)
    ap.add_argument("--n_perm", type=int, default=200,
                    help="permutation draws for the MI null (cheap sanity check)")
    args = ap.parse_args()

    dataset, noise = SETTING
    seeds = sorted(int(p.stem[4:]) for p in (CACHE / dataset / noise).glob("seed*.npz"))
    print(f"== A2 / residual dependence of detector score E on {args.proxy} ==")
    print(f"   C100-S20, seeds {seeds}, {len(DETS)} detectors\n")

    rows = []
    for seed in seeds:
        with np.load(CACHE / dataset / noise / f"seed{seed}.npz",
                     allow_pickle=False) as z:
            p = {k: np.asarray(z[k]).copy() for k in z.files}
        q = np.asarray(p[f"sig__{args.proxy}"], np.float64)
        for det in DETS:
            e = get_detector(p, det)
            rho = float(spearmanr(e, q).statistic)
            r = float(pearsonr(e, q).statistic)
            mi = float(mutual_info_classif(e.reshape(-1, 1), (q > np.median(q)).astype(int),
                                           discrete_features=False, random_state=seed)[0])
            rows.append({"seed": seed, "detector": det,
                         "detector_label": DISPLAY.get(det, det),
                         "proxy": args.proxy,
                         "spearman": rho, "pearson": r,
                         "abs_spearman": abs(rho), "abs_pearson": abs(r),
                         "mi_median_split": mi})
        print(f"   seed{seed}: done", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "ae_residual_correlation_perseed.csv", index=False)

    rng = np.random.default_rng(RNG_SEED)
    summary = []
    for det, g in df.groupby("detector"):
        s = g.spearman.to_numpy()
        r = g.pearson.to_numpy()
        m = g.mi_median_split.to_numpy()
        def boot(x):
            idx = rng.integers(0, len(x), size=(B_SEED, len(x)))
            mm = x[idx].mean(axis=1)
            return float(mm.std(ddof=1)), float(np.quantile(mm, 0.025)), float(np.quantile(mm, 0.975))
        sp_se, sp_lo, sp_hi = boot(s)
        pe_se, pe_lo, pe_hi = boot(r)
        mi_se, mi_lo, mi_hi = boot(m)
        summary.append({
            "detector": det, "detector_label": DISPLAY.get(det, det),
            "proxy": args.proxy, "n_seeds": len(g),
            "spearman_mean": s.mean(), "spearman_sd": s.std(ddof=1),
            "spearman_se": sp_se, "spearman_ci_lo": sp_lo, "spearman_ci_hi": sp_hi,
            "pearson_mean": r.mean(), "pearson_sd": r.std(ddof=1),
            "pearson_se": pe_se, "pearson_ci_lo": pe_lo, "pearson_ci_hi": pe_hi,
            "mi_mean": m.mean(), "mi_sd": m.std(ddof=1),
            "mi_se": mi_se, "mi_ci_lo": mi_lo, "mi_ci_hi": mi_hi,
            "abs_spearman_mean": np.abs(s).mean(), "abs_pearson_mean": np.abs(r).mean(),
        })
    sr = pd.DataFrame(summary).sort_values("abs_spearman_mean", ascending=False)
    sr.to_csv(OUT / "ae_residual_correlation_summary.csv", index=False)

    print("\n== summary (mean +- SD over 10 seeds; seed-bootstrap 95% CI) ==")
    print(sr[["detector_label", "spearman_mean", "spearman_sd", "spearman_ci_lo",
              "spearman_ci_hi", "pearson_mean", "pearson_sd",
              "mi_mean", "mi_sd"]].round(4).to_string(index=False))

    print("\n== does removing the neighbour term remove the dependence? ==")
    a = sr[sr.detector == "combined_noN"].iloc[0]
    b = sr[sr.detector == "combined"].iloc[0]
    print(f"   Combined      |rho| = {abs(b.spearman_mean):.4f}  (r = {b.pearson_mean:+.4f})")
    print(f"   Combined-noN  |rho| = {abs(a.spearman_mean):.4f}  (r = {a.pearson_mean:+.4f})")
    print(f"   Spearman drop: {abs(b.spearman_mean) - abs(a.spearman_mean):+.4f}; "
          f"Pearson drop: {abs(b.pearson_mean) - abs(a.pearson_mean):+.4f}")
    print(f"   Combined-noN residual is {'significant' if a.spearman_ci_lo > 0 or a.spearman_ci_hi < 0 else 'NOT significant'}"
          f" at the 95% level (CI [{a.spearman_ci_lo:+.4f}, {a.spearman_ci_hi:+.4f}])")
    print("\n   wrote ae_residual_correlation_perseed.csv, ae_residual_correlation_summary.csv")


if __name__ == "__main__":
    main()
