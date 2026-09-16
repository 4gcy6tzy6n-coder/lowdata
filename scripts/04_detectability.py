"""04_detectability.py — the Phase-4 core benchmark.

Computes, per run:
    global AUROC, quality-stratified AUROC (quintiles),
    matched-quality AUROC, bootstrap 95% CIs, and Delta_conf = AUC_global - AUC_matched.
Writes per-run artifacts and cross-seed aggregation + figures.

Usage:
    python scripts/04_detectability.py --exp configs/experiments/gate_a.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from quality_noise.config import hash_config, load_config
from quality_noise.data.datasets import load_bundle
from quality_noise.detectability.bootstrap import bootstrap_indexed_ci
from quality_noise.detectability.decomposition import cross_quality_pair_matrix
from quality_noise.detectability.global_auc import global_auc
from quality_noise.detectability.matched_auc import match_quality_pairs, matched_auc
from quality_noise.detectability.stratified_auc import quality_stratified_auc
from quality_noise.evidence.signals import assemble_evidence
from quality_noise.experiment import expand_experiment, run_key
from quality_noise.utils import make_results_dirs, write_json_atomic


def _run_artifacts(results: dict, cfg: dict, cfg_hash: str) -> dict[str, Path]:
    base = (
        results["detectability"]
        / cfg["dataset"]["name"]
        / f"{cfg['noise']['type']}{cfg['noise']['rate']}"
        / f"seed{cfg['seed']}"
    )
    base.mkdir(parents=True, exist_ok=True)
    return {
        "traces": results["traces"] / cfg["dataset"]["name"] / f"{cfg['noise']['type']}{cfg['noise']['rate']}" / f"seed{cfg['seed']}",
        "quality": results["quality"] / cfg["dataset"]["name"] / f"{cfg['noise']['type']}{cfg['noise']['rate']}" / f"seed{cfg['seed']}",
        "out": base,
    }


def detectability_run(cfg: dict, results: dict, exp: dict) -> dict:
    bundle = load_bundle(cfg)
    key = run_key(cfg)
    cfg_hash = hash_config(cfg)
    arts = _run_artifacts(results, cfg, cfg_hash)
    out = arts["out"]

    q_path = arts["quality"] / f"quality_{cfg_hash}.parquet"
    tr_path = arts["traces"] / f"traces_{cfg_hash}.parquet"
    emb_path = arts["traces"] / f"embeddings_{cfg_hash}.npy"
    if not q_path.exists() or not tr_path.exists() or not emb_path.exists():
        raise FileNotFoundError(f"Run {key} missing quality/traces (run 02, 03 first).")

    quality = pd.read_parquet(q_path)
    traces = pd.read_parquet(tr_path)
    embeddings = np.load(emb_path)
    y_observed = bundle.train_view.y_observed.astype(int)
    mask = bundle.eval_view.mask
    id2mask = {int(sid): bool(m) for sid, m in zip(bundle.eval_view.sample_ids, mask)}
    mask_aligned = np.array([id2mask[int(sid)] for sid in quality["sample_id"].values], dtype=bool)

    quality_val = quality["knn_agreement"].to_numpy(dtype=np.float64)
    q = quality_val

    # Evidence E (fixed-weight combination of loss/forgetting/conflict signals).
    ev_cfg = cfg.get("evidence", {})
    E, _signals = assemble_evidence(
        ev_cfg.get("recipe", "ema_loss_forgetting_conflict"),
        traces,
        embeddings,
        y_observed,
        conflict_k=int(ev_cfg.get("conflict_k", 20)),
    )
    mask_aligned = mask_aligned[: len(E)]

    ma = cfg.get("matched_auc", {})
    eps = float(ma.get("eps", 0.1))
    max_clean = int(ma.get("max_clean_per_noisy", 5))
    bs = cfg.get("bootstrap", {})

    auc_global = global_auc(E, mask_aligned)
    pairs = match_quality_pairs(E, mask_aligned, q, eps=eps, max_clean_per_noisy=max_clean)
    auc_matched = matched_auc(pairs)
    ci_global = bootstrap_indexed_ci(E, mask_aligned, q, metric="global", n_resamples=bs.get("n_resamples", 1000), ci=bs.get("ci", 0.95))
    ci_matched = bootstrap_indexed_ci(
        E, mask_aligned, q,
        metric="matched",
        n_resamples=bs.get("matched_n_resamples", 300),
        ci=bs.get("ci", 0.95),
        matched_eps=eps,
        matched_subsample=bs.get("matched_subsample", 5000),
    )

    strat = quality_stratified_auc(E, mask_aligned, q, n_bins=5)
    strat.to_csv(out / "detectability.csv", index=False)

    # Quality-pair decomposition matrix (Phase 5 / Sprint 4): saved for the
    # cross-seed 5x5 heatmap produced in the aggregate step.
    pair_mat = cross_quality_pair_matrix(E, mask_aligned, q, n_bins=5)
    np.save(out / "pair_matrix.npy", pair_mat)

    summary = {
        "run_key": key,
        "config_hash": cfg_hash,
        "auc_global": auc_global,
        "auc_matched": auc_matched,
        "delta_conf": auc_global - auc_matched,
        "ci_global": list(ci_global),
        "ci_matched": list(ci_matched),
        "n_pairs": int(len(pairs.s_noisy)),
    }
    write_json_atomic(out / "global_summary.json", summary)

    pd.DataFrame(
        {
            "s_noisy": pairs.s_noisy,
            "s_clean": pairs.s_clean,
            "q_noisy": pairs.q_noisy,
            "q_clean": pairs.q_clean,
        }
    ).to_parquet(out / "matched_pairs.parquet", index=False)

    print(f"[04] {key}: AUC_global={auc_global:.3f} AUC_matched={auc_matched:.3f} delta={auc_global - auc_matched:+.3f}")
    return summary


def _aggregate(exp_cfg: dict, results: dict) -> None:
    """Aggregate per-seed metrics across seeds -> aggregated.csv + figures."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = expand_experiment(exp_cfg)
    rows = []
    for cfg in runs:
        key = run_key(cfg)
        if not key.endswith("seed0"):
            continue
        noise = cfg["noise"]
        root = (
            results["detectability"]
            / cfg["dataset"]["name"]
            / f"{noise['type']}{noise['rate']}"
        )
        summaries = []
        strat_frames = []
        for seed in exp_cfg.get("seeds", [0]):
            sdir = root / f"seed{seed}"
            if not (sdir / "global_summary.json").exists():
                continue
            with open(sdir / "global_summary.json") as f:
                summaries.append(json.load(f))
            strat_frames.append(pd.read_csv(sdir / "detectability.csv").assign(seed=seed))
        if not summaries:
            continue
        agg = {
            "dataset": cfg["dataset"]["name"],
            "noise_type": noise["type"],
            "noise_rate": noise["rate"],
        }
        for m in ("auc_global", "auc_matched", "delta_conf"):
            vals = [s[m] for s in summaries]
            agg[f"{m}_mean"] = float(np.mean(vals))
            agg[f"{m}_std"] = float(np.std(vals))
        rows.append(agg)

        # Figure 1: AUC by quality quintile, per-seed lines + mean.
        if strat_frames:
            fig, ax = plt.subplots(figsize=(6, 4.5))
            bins = sorted(strat_frames[0]["quality_bin"].unique())
            for sf in strat_frames:
                ax.plot(bins, sf.sort_values("quality_bin")["auc"], alpha=0.4, lw=1)
            mean_auc = np.nanmean(np.stack([sf.sort_values("quality_bin")["auc"].values for sf in strat_frames]), axis=0)
            ax.plot(bins, mean_auc, "k-o", lw=2, label="mean")
            ax.axhline(agg["auc_global_mean"], ls="--", color="gray", label=f"AUC_global={agg['auc_global_mean']:.2f}")
            ax.set_xticks(bins)
            ax.set_xlabel("Quality quintile")
            ax.set_ylabel("Within-quality AUROC")
            ax.set_ylim(0.4, 1.0)
            ax.set_title(f"{noise['type']} {noise['rate']}")
            ax.legend()
            fig.tight_layout()
            fig.savefig(results["figures_root"] / "detectability" / f"{cfg['dataset']['name']}_{noise['type']}{noise['rate']}_quintile_auc.png", dpi=150)
            fig.savefig(results["figures_root"] / "detectability" / f"{cfg['dataset']['name']}_{noise['type']}{noise['rate']}_quintile_auc.svg")
            plt.close(fig)

        # Figure: 5x5 quality-pair heatmap (mean across seeds) + confounding gap.
        mats = [np.load(sdir / "pair_matrix.npy") for sdir in (root / f"seed{s}" for s in exp_cfg.get("seeds", [0])) if (sdir / "pair_matrix.npy").exists()]
        if mats:
            mean_mat = np.nanmean(np.stack(mats), axis=0)
            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.imshow(mean_mat, cmap="viridis", vmin=0.4, vmax=1.0)
            for i in range(mean_mat.shape[0]):
                for j in range(mean_mat.shape[1]):
                    ax.text(j, i, f"{mean_mat[i, j]:.2f}", ha="center", va="center", color="w", fontsize=9)
            ax.set_xlabel("Clean quality bin (r)")
            ax.set_ylabel("Noisy quality bin (q)")
            ax.set_title(f"{noise['type']} {noise['rate']} quality-pair AUC\n(off-diag = confounding)")
            fig.colorbar(im, ax=ax)
            fig.tight_layout()
            fig.savefig(results["figures_root"] / "detectability" / f"{cfg['dataset']['name']}_{noise['type']}{noise['rate']}_quality_pair_heatmap.png", dpi=150)
            fig.savefig(results["figures_root"] / "detectability" / f"{cfg['dataset']['name']}_{noise['type']}{noise['rate']}_quality_pair_heatmap.svg")
            plt.close(fig)

    df = pd.DataFrame(rows)
    df.to_csv(results["detectability"].parent / "aggregated_detectability.csv", index=False)

    # Figure 2: AUC_global vs AUC_matched grouped bars across noise settings.
    if not df.empty:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        labels = df.apply(lambda r: f"{r['noise_type']}\n{r['noise_rate']}", axis=1)
        x = np.arange(len(df))
        w = 0.36
        ax.bar(x - w / 2, df["auc_global_mean"], w, yerr=df["auc_global_std"], label="AUC_global", capsize=3)
        ax.bar(x + w / 2, df["auc_matched_mean"], w, yerr=df["auc_matched_std"], label="AUC_matched", capsize=3)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 1)
        ax.set_ylabel("AUROC")
        ax.set_title("Global vs matched-quality detectability")
        ax.legend()
        fig.tight_layout()
        fig.savefig(results["figures_root"] / "detectability" / "global_vs_matched.png", dpi=150)
        fig.savefig(results["figures_root"] / "detectability" / "global_vs_matched.svg")
        plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="configs/experiments/gate_a.yaml")
    ap.add_argument("--aggregate-only", action="store_true")
    ap.add_argument("--no-aggregate", action="store_true", help="skip cross-seed aggregation (for parallel per-run invocations)")
    ap.add_argument("--device", default="auto", help="accepted for uniform CLI; detectability is CPU-only")
    ap.add_argument("--runs", type=int, nargs="*", default=None, help="restrict compute to these run indices")
    args = ap.parse_args()

    exp = load_config(args.exp)
    results = make_results_dirs({"results_root": exp.get("results_root")})
    results["figures_root"] = Path("figures")
    (results["figures_root"] / "detectability").mkdir(parents=True, exist_ok=True)

    if not args.aggregate_only:
        runs = expand_experiment(exp)
        if args.runs is not None:
            runs = [runs[i] for i in args.runs]
        for cfg in runs:
            detectability_run(cfg, results, exp)
    if not args.no_aggregate:
        _aggregate(exp, results)
    print("[04] done")


if __name__ == "__main__":
    main()
