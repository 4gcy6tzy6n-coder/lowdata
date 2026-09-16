"""Answer the four decisive revision questions from the produced tables.

Run after ``run_p0_independent_q.py --phase evaluate``.  Prints the gate verdicts
and the numbers they rest on, so the headline claims can be checked at a glance.
Also writes ``results/revision/gate_summary.json``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/root/quality_noise")
REV = ROOT / "results" / "revision"
CELLS = REV / "tables" / "independent_q_table_all_settings.csv"


def _p(v, nd=3):
    return "n/a" if v is None or not np.isfinite(float(v)) else f"{float(v):+.{nd}f}"


def main() -> None:
    cells = pd.read_csv(CELLS)
    if "error" in cells.columns:
        n_err = int(cells["error"].notna().sum())
        cells = cells[cells["auc_global"].notna()]
    else:
        n_err = 0
    base = cells[(cells.strategy == "nn_wo") & (np.isclose(cells.eps_sd, 0.10))]

    out: dict = {"n_cells": int(len(cells)), "n_skipped": n_err,
                 "datasets": sorted(cells.dataset.unique().tolist()),
                 "settings": sorted((cells.dataset + "/" + cells.noise).unique().tolist())}

    print(f"== P0 gate report ==  ({len(cells)} cells over {cells.dataset.nunique()} datasets, "
          f"{n_err} skipped)")
    print(f"   settings: {', '.join(out['settings'])}")
    print()

    # ---------------- Gate 1 ----------------
    dino = base[base.quality == "dino_density"]
    knn = base[base.quality == "knn_agreement"]
    n, pos = len(dino), int((dino.delta_q > 0).sum())
    print("GATE 1 — label-independent Q (DINOv2 density)")
    print(f"  Delta_Q > 0 : {pos}/{n}  ({100*pos/max(n,1):.1f}%)   mean Delta_Q = {_p(dino.delta_q.mean())}")
    print(f"  paired CI(Delta_Q) entirely > 0 : {int(dino.ci_delta_lo_gt_0.sum())}/{n}")
    print(f"  [reference] Q^KNN : {int((knn.delta_q > 0).sum())}/{len(knn)} positive, "
          f"mean {_p(knn.delta_q.mean())}")
    out["gate1"] = {
        "n": n, "n_positive": pos, "mean_delta_q": float(dino.delta_q.mean()),
        "n_ci_above_0": int(dino.ci_delta_lo_gt_0.sum()),
        "knn_reference": {"n": len(knn), "n_positive": int((knn.delta_q > 0).sum()),
                          "mean_delta_q": float(knn.delta_q.mean())},
    }

    print("\n  per-detector (DINO-Q):")
    g = dino.groupby("detector").agg(n=("delta_q", "size"), mean_d=("delta_q", "mean"),
                                     frac_pos=("delta_q", lambda s: float((s > 0).mean())),
                                     n_ci=("ci_delta_lo_gt_0", "sum"))
    for det, r in g.sort_values("mean_d", ascending=False).iterrows():
        print(f"    {det:20s} n={int(r['n']):3d}  mean={_p(r['mean_d'])}  "
              f"pos={r['frac_pos']:.2f}  CI>0={int(r['n_ci'])}")
    out["gate1_by_detector"] = g.reset_index().to_dict(orient="records")

    print("\n  per-setting (DINO-Q, mean over detectors):")
    g2 = dino.groupby(["dataset", "noise"]).agg(n=("delta_q", "size"),
                                                auc_g=("auc_global", "mean"),
                                                auc_qc=("auc_qc", "mean"),
                                                mean_d=("delta_q", "mean"))
    for (ds, noise), r in g2.iterrows():
        print(f"    {ds}/{noise:22s} n={int(r['n']):3d}  AUC_g={r['auc_g']:.3f}  "
              f"AUC_QC={r['auc_qc']:.3f}  Delta_Q={_p(r['mean_d'])}")
    out["gate1_by_setting"] = [{"dataset": k[0], "noise": k[1], **v}
                              for k, v in g2.to_dict(orient="index").items()]
    out["gate1_by_setting"] = [
        {"dataset": ds, "noise": noise, "n": int(r["n"]), "auc_global": float(r["auc_g"]),
         "auc_qc": float(r["auc_qc"]), "delta_q": float(r["mean_d"])}
        for (ds, noise), r in g2.iterrows()]

    # ---------------- Gate 2 ----------------
    print("\nGATE 2 — C100-S20 reversal under matching rigour")
    c100 = base[(base.dataset == "cifar100") & (base.noise == "symmetric0.2")]
    for q in ["knn_agreement", "dino_density"]:
        s = c100[c100.quality == q]
        if not len(s):
            continue
        print(f"  Q={q:16s} n_cells={len(s)}  AUC_QC mean={s.auc_qc.mean():.3f}  "
              f"cells with CI(AUC_QC) fully <0.5: {int(s.ci_qc_hi_lt_0p5.sum())}  "
              f"fully >0.5: {int(s.ci_qc_lo_gt_0p5.sum()) if 'ci_qc_lo_gt_0p5' in s else 0}")
        for _, r in s.sort_values("detector").iterrows():
            print(f"      {r.detector:20s} AUC_g={r.auc_global:.3f} AUC_QC={r.auc_qc:.3f} "
                  f"CI_QC=[{r.ci_qc_lo:.3f},{r.ci_qc_hi:.3f}] Delta={_p(r.delta_q)}")
    out["gate2_c100s20"] = {
        q: c100[c100.quality == q][["detector", "auc_global", "auc_qc", "ci_qc_lo",
                                    "ci_qc_hi", "delta_q", "ci_delta_lo", "ci_delta_hi",
                                    "ci_qc_hi_lt_0p5"]].to_dict(orient="records")
        for q in ["knn_agreement", "dino_density"]}

    # ---------------- Gate 3 ----------------
    print("\nGATE 3 — Confident Learning quality sensitivity")
    cl = dino[dino.detector == "confident_learning"]
    print(f"  Delta_Q > 0 : {int((cl.delta_q > 0).sum())}/{len(cl)}   "
          f"mean Delta_Q = {_p(cl.delta_q.mean())}")
    g3 = cl.groupby(["dataset", "noise"]).delta_q.agg(["mean", "size"])
    for (ds, noise), r in g3.iterrows():
        print(f"    {ds}/{noise:22s} n={int(r['size']):3d}  mean Delta_Q={_p(r['mean'])}")
    out["gate3"] = {"n": int(len(cl)), "n_positive": int((cl.delta_q > 0).sum()),
                    "mean_delta_q": float(cl.delta_q.mean()) if len(cl) else None,
                    "per_setting": [{"dataset": k[0], "noise": k[1], "n": int(v["size"]),
                                     "mean_delta_q": float(v["mean"])}
                                    for k, v in g3.iterrows()]}

    # ---------------- Gate 4 ----------------
    print("\nGATE 4 — real human annotation noise")
    real = dino[dino.dataset.isin(["cifar10n", "cifar100n"])]
    if len(real):
        print(f"  Delta_Q > 0 : {int((real.delta_q > 0).sum())}/{len(real)}   "
              f"mean Delta_Q = {_p(real.delta_q.mean())}")
        g4 = real.groupby(["dataset", "noise"]).agg(n=("delta_q", "size"),
                                                    auc_g=("auc_global", "mean"),
                                                    auc_qc=("auc_qc", "mean"),
                                                    mean_d=("delta_q", "mean"))
        for (ds, noise), r in g4.iterrows():
            print(f"    {ds}/{noise:22s} n={int(r['n']):3d}  AUC_g={r['auc_g']:.3f}  "
                  f"AUC_QC={r['auc_qc']:.3f}  Delta_Q={_p(r['mean_d'])}")
        out["gate4"] = [
            {"dataset": ds, "noise": noise, "n": int(r["n"]), "auc_global": float(r["auc_g"]),
             "auc_qc": float(r["auc_qc"]), "delta_q": float(r["mean_d"])}
            for (ds, noise), r in g4.iterrows()]
    else:
        print("  no real-noise cells yet")

    # ---------------- matching sanity ----------------
    print("\nMATCHING BALANCE (DINO-Q, nn_wo, caliper 0.10 SD)")
    print(f"  |SMD| after matching: mean {abs(dino.smd_after).mean():.4f}, "
          f"max {abs(dino.smd_after).max():.4f}   (target < 0.1)")
    print(f"  noisy coverage: mean {dino.coverage_noisy.mean():.3f}, "
          f"min {dino.coverage_noisy.min():.3f}")
    print(f"  mean |dQ| within pairs: {dino.mean_abs_dq.mean():.5f}  "
          f"(Q SD = {dino.q_sd.mean():.4f} <-> caliper {0.10*dino.q_sd.mean():.5f})")
    out["matching"] = {
        "abs_smd_after_mean": float(abs(dino.smd_after).mean()),
        "abs_smd_after_max": float(abs(dino.smd_after).max()),
        "coverage_noisy_mean": float(dino.coverage_noisy.mean()),
        "coverage_noisy_min": float(dino.coverage_noisy.min()),
        "mean_abs_dq": float(dino.mean_abs_dq.mean()),
    }

    (REV / "gate_summary.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"\nwrote {REV/'gate_summary.json'}")


if __name__ == "__main__":
    main()
