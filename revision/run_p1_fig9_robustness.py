"""P1-8 — Fig. 9 robustness: association between quality-controlled detectability
and governance outcome.

The manuscript reported a single Pearson r (0.798, p = 0.018, n = 8) and called
it predictive.  A referee correctly noted that (a) n = 8 with 2 datasets means
the correlation can be driven by a between-dataset shift, and (b) a single
statistic with no CI is not evidence of prediction.

This script reports, on the same 8 settings:
  * Pearson r and Spearman rho with bootstrap 95% CIs,
  * within-dataset correlations (n = 4 each; direction only, no significance),
  * leave-one-out r and its full range,
  * a dataset-cluster check: correlation after removing the between-dataset
    mean shift (i.e. on within-dataset centred values).

It then states the honest conclusion: association, not prediction.

Input : revision/results/fig9_source.csv  (manually curated from the frozen
        headline tables; provenance is stated in the file)
Output: revision/results/fig9_correlation_analysis.csv
        revision/results/fig9_loo.csv
        revision/results/fig9_summary.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
SRC = HERE / "results" / "fig9_source.csv"
OUT = HERE / "results"


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(stats.pearsonr(x, y).statistic)


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(stats.spearmanr(x, y).statistic)


def boot_ci(x: np.ndarray, y: np.ndarray, fn, n: int = 10000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    idx = np.arange(len(x))
    stats_ = []
    for _ in range(n):
        b = rng.choice(idx, size=len(idx), replace=True)
        if len(np.unique(b)) < 3:
            continue
        v = fn(x[b], y[b])
        if np.isfinite(v):
            stats_.append(v)
    if not stats_:
        return (float("nan"), float("nan"))
    return (float(np.quantile(stats_, 0.025)), float(np.quantile(stats_, 0.975)))


def main() -> None:
    df = pd.read_csv(SRC)
    df = df[df["hard_delta_acc"].notna()].reset_index(drop=True)
    x = df["auc_matched"].to_numpy(float)
    y = df["hard_delta_acc"].to_numpy(float)

    rows = []

    def add(scope: str, n: int, xs: np.ndarray, ys: np.ndarray, with_ci: bool) -> None:
        r = pearson(xs, ys)
        rho = spearman(xs, ys)
        p_r = float(stats.pearsonr(xs, ys).pvalue) if n >= 3 and np.std(xs) > 0 else float("nan")
        p_s = float(stats.spearmanr(xs, ys).pvalue) if n >= 3 and np.std(xs) > 0 else float("nan")
        r_ci = boot_ci(xs, ys, pearson) if with_ci else (float("nan"), float("nan"))
        rho_ci = boot_ci(xs, ys, spearman) if with_ci else (float("nan"), float("nan"))
        rows.append({
            "scope": scope, "n": n,
            "pearson_r": r, "pearson_p": p_r, "pearson_ci_lo": r_ci[0], "pearson_ci_hi": r_ci[1],
            "spearman_rho": rho, "spearman_p": p_s,
            "spearman_ci_lo": rho_ci[0], "spearman_ci_hi": rho_ci[1],
        })

    add("all_settings", len(x), x, y, with_ci=True)
    for ds, g in df.groupby("dataset"):
        add(f"within_{ds}", len(g), g["auc_matched"].to_numpy(float),
            g["hard_delta_acc"].to_numpy(float), with_ci=False)

    # Dataset-cluster check: centre each variable within its dataset, then
    # correlate.  If the association survives centring it is not merely a
    # between-dataset level shift.
    xc = df.groupby("dataset")["auc_matched"].transform(lambda s: s - s.mean()).to_numpy(float)
    yc = df.groupby("dataset")["hard_delta_acc"].transform(lambda s: s - s.mean()).to_numpy(float)
    add("within_dataset_centred", len(xc), xc, yc, with_ci=False)

    res = pd.DataFrame(rows)
    res.to_csv(OUT / "fig9_correlation_analysis.csv", index=False)

    # Leave-one-out
    loo = []
    for i in range(len(df)):
        keep = np.arange(len(df)) != i
        loo.append({
            "removed": df.loc[i, "setting"],
            "removed_dataset": df.loc[i, "dataset"],
            "pearson_r": pearson(x[keep], y[keep]),
            "spearman_rho": spearman(x[keep], y[keep]),
            "n": int(keep.sum()),
        })
    loo_df = pd.DataFrame(loo)
    loo_df.to_csv(OUT / "fig9_loo.csv", index=False)

    r_all = pearson(x, y)
    r_loo = loo_df["pearson_r"].to_numpy(float)
    loo_range = [float(np.nanmin(r_loo)), float(np.nanmax(r_loo))]
    # Does removing any single point destroy the association (CI crossing 0 / r<0.5)?
    fragile = [s for s, v in zip(loo_df["removed"], r_loo) if not np.isfinite(v) or v < 0.5]

    summary = {
        "n_settings": int(len(df)),
        "pearson_r": r_all,
        "spearman_rho": spearman(x, y),
        "pearson_ci": list(boot_ci(x, y, pearson)),
        "spearman_ci": list(boot_ci(x, y, spearman)),
        "within_dataset": res[res.scope.str.startswith("within_") & ~res.scope.str.contains("centred")]
                            .set_index("scope")[["n", "pearson_r", "spearman_rho"]].to_dict(orient="index"),
        "within_dataset_centred_r": float(res.loc[res.scope == "within_dataset_centred", "pearson_r"].iloc[0]),
        "loo_r_range": loo_range,
        "loo_min_setting": str(loo_df.loc[loo_df["pearson_r"].idxmin(), "removed"]),
        "fragile_settings_r_below_0.5": fragile,
        "conclusion": (
            "ASSOCIATION, NOT PREDICTION. The headline correlation is carried by a "
            "between-dataset level shift: all four CIFAR-10 settings cluster at "
            "AUC_QC 0.86-0.93 with positive hard-governance gains, all four CIFAR-100 "
            "settings at 0.39-0.60 with near-zero or negative gains. Within either "
            "dataset (n=4) the correlation is not interpretable, and the "
            "leave-one-out range shows how few points carry the fit."
            if abs(float(res.loc[res.scope == "within_dataset_centred", "pearson_r"].iloc[0])) < 0.5
            else "Association survives within-dataset centring; still descriptive only."
        ),
    }
    with open(OUT / "fig9_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(res.round(3).to_string(index=False))
    print("\nLeave-one-out:")
    print(loo_df.round(3).to_string(index=False))
    print(f"\nLOO Pearson range: [{loo_range[0]:.3f}, {loo_range[1]:.3f}] "
          f"(most influential: {summary['loo_min_setting']})")
    print(f"\n{summary['conclusion']}")


if __name__ == "__main__":
    main()
