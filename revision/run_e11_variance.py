"""E11: is reversal a family-level or a detector-level phenomenon?

E10 showed the arithmetic of reversal -- a reversal needs delta > A_global - 0.5 --
explains ~94% of direction, but that its disagreements are exactly the cases where
one detector's own delta departs from its family's.  So the useful question is
which level owns the effect:

  * if reversal is a property of the *quality proxy* (family), then knowing the
    family and the detector's strength should predict reversal and adding the
    detector's identity should add little;
  * if it is a property of the *(proxy, detector)* pair, the detector identity must
    add significantly once family and strength are accounted for.

Both models are binomial GLMs fitted by IRLS in numpy (no statsmodels dependency),
and the likelihood-ratio statistic is calibrated against a permutation null that
shuffles the detector label *within each proxy*, which preserves the family-level
structure while destroying any detector-specific effect.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "revision" / "round2"

ANALYSIS_SET = OUT / "final_analysis_set.csv"
EXCLUDED_TIERS = ("excluded",)     # neighbor: saturating score resolution


def load() -> pd.DataFrame:
    """Canonical pooled table, excluded detectors removed.

    Reading final_analysis_set.csv instead of the per-audit CSVs keeps the
    detector-tier decision and the coverage schema in one place.
    """
    if not ANALYSIS_SET.exists():
        raise SystemExit(f"missing {ANALYSIS_SET}; run run_final_analysis_set.py first")
    d = pd.read_csv(ANALYSIS_SET)
    d = d[~d.detector_tier.isin(EXCLUDED_TIERS)].copy()
    d = d[(d.global_auc >= 0.5) & d.interval_reversal.notna()].copy()
    d["dataset"] = d.audit
    return d


def design(df: pd.DataFrame, terms: list[str]) -> np.ndarray:
    """Intercept + numeric strength + one-hot dummies for each categorical term."""
    cols = [np.ones(len(df))]
    if "strength" in terms:
        cols.append(df.global_auc.to_numpy(float))
    for t in terms:
        if t == "strength":
            continue
        dummies = pd.get_dummies(df[t].astype(str), drop_first=True).to_numpy(float)
        cols.extend(dummies[:, j] for j in range(dummies.shape[1]))
    return np.column_stack(cols)


def irls_loglik(X: np.ndarray, y: np.ndarray, ridge: float = 1e-8,
                iters: int = 200) -> tuple[float, np.ndarray]:
    """Binomial GLM by IRLS; a tiny ridge keeps separable cells from diverging."""
    n, k = X.shape
    beta = np.zeros(k)
    for _ in range(iters):
        eta = np.clip(X @ beta, -30, 30)
        mu = 1.0 / (1.0 + np.exp(-eta))
        w = np.maximum(mu * (1 - mu), 1e-10)
        z = eta + (y - mu) / w
        XtW = X.T * w
        A = XtW @ X + ridge * np.eye(k)
        b = XtW @ z
        try:
            new = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            break
        if np.max(np.abs(new - beta)) < 1e-10:
            beta = new
            break
        beta = new
    eta = np.clip(X @ beta, -30, 30)
    mu = 1.0 / (1.0 + np.exp(-eta))
    mu = np.clip(mu, 1e-12, 1 - 1e-12)
    return float((y * np.log(mu) + (1 - y) * np.log(1 - mu)).sum()), beta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_perm", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="e11_variance_decomposition.csv")
    args = ap.parse_args()

    df = load()
    df["rev"] = df.interval_reversal.astype(int)
    y = df.rev.to_numpy(float)
    print(f"== e11: {len(df)} cells, {int(y.sum())} interval reversals "
          f"({y.mean():.3%}), {df.dataset.nunique()} audits ==")

    X_fam = design(df, ["strength", "proxy_family"])
    X_prox = design(df, ["strength", "proxy"])
    X_full = design(df, ["strength", "proxy", "detector"])

    ll_fam, _ = irls_loglik(X_fam, y)
    ll_prox, _ = irls_loglik(X_prox, y)
    ll_full, _ = irls_loglik(X_full, y)

    lr_det = 2 * (ll_full - ll_prox)
    lr_prox = 2 * (ll_prox - ll_fam)
    print(f"   loglik family       {ll_fam:10.2f}  (k={X_fam.shape[1]})")
    print(f"   loglik proxy        {ll_prox:10.2f}  (k={X_prox.shape[1]})")
    print(f"   loglik proxy+det    {ll_full:10.2f}  (k={X_full.shape[1]})")
    print(f"   LR proxy | family   {lr_prox:8.2f} on {X_prox.shape[1]-X_fam.shape[1]} df"
          "   (uncalibrated)" if False else
          f"   LR proxy | family   {lr_prox:8.2f} on {X_prox.shape[1]-X_fam.shape[1]} df")

    # Permutation null for the detector term: shuffle detector labels within proxy,
    # which leaves the proxy-level structure intact but removes detector identity.
    rng = np.random.default_rng(args.seed)
    null = np.empty(args.n_perm)
    for i in range(args.n_perm):
        d2 = df.copy()
        d2["detector"] = (d2.groupby("proxy", group_keys=False)["detector"]
                          .transform(lambda s: rng.permutation(s.to_numpy())))
        Xp = design(d2, ["strength", "proxy"])
        Xf = design(d2, ["strength", "proxy", "detector"])
        null[i] = 2 * (irls_loglik(Xf, y)[0] - irls_loglik(Xp, y)[0])
    p_perm = float((null >= lr_det).mean())
    print(f"   LR detector | proxy {lr_det:8.2f} on {X_full.shape[1]-X_prox.shape[1]} df"
          f"   permutation p = {p_perm:.4f}  (null mean {null.mean():.1f})")

    rows = [{"model": "strength+family", "loglik": ll_fam, "k": X_fam.shape[1]},
            {"model": "strength+proxy", "loglik": ll_prox, "k": X_prox.shape[1]},
            {"model": "strength+proxy+detector", "loglik": ll_full, "k": X_full.shape[1]},
            {"model": "LR_proxy_given_family", "loglik": lr_prox,
             "k": X_prox.shape[1] - X_fam.shape[1]},
            {"model": "LR_detector_given_proxy", "loglik": lr_det,
             "k": X_full.shape[1] - X_prox.shape[1]}]
    pd.DataFrame(rows).to_csv(OUT / args.out, index=False)
    pd.DataFrame({"perm_LR_detector_given_proxy": null}).to_csv(
        OUT / args.out.replace(".csv", "_null.csv"), index=False)

    print("\n== reversal rate by proxy x detector (pooled audits) ==")
    t = df.pivot_table(index="proxy", columns="detector", values="rev", aggfunc="mean").round(3)
    print(t.to_string())
    t.to_csv(OUT / args.out.replace(".csv", "_by_cell.csv"))

    print("\n== reversal rate by proxy x strength bin ==")
    df["strength_bin"] = pd.cut(df.global_auc, [0.5, 0.6, 0.7, 0.8, 1.01],
                                labels=["[0.50,0.60)", "[0.60,0.70)",
                                        "[0.70,0.80)", "[0.80,1.00]"], right=False)
    t2 = df.pivot_table(index="proxy", columns="strength_bin", values="rev",
                        aggfunc="mean", observed=True).round(3)
    print(t2.to_string())
    t2.to_csv(OUT / args.out.replace(".csv", "_by_strength.csv"))


if __name__ == "__main__":
    main()
