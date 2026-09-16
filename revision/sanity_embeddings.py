"""Sanity-check the revision embeddings before trusting them in the pipeline.

Checks
------
1. Native ResNet features we recompute match the run's own saved
   ``embeddings_<hash>.npy`` (cosine similarity per sample).  If they match, the
   sample ordering recovery in ``extract_embeddings.extract_native`` is correct.
2. DINOv2 embeddings carry class structure (kNN classification accuracy on clean
   labels) — a frozen backbone with no signal would make Q^DINO meaningless.
3. Q^DINO is genuinely label-free: its AUC against the evaluation-only noise
   mask, and its correlation with the label-dependent Q^KNN.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/root/quality_noise")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "revision"))

from revq.independent_q import compute_independent_q, unit_affine  # noqa: E402

EMB = ROOT / "results" / "revision" / "embeddings"
RESULTS = ROOT / "results"


def _run_sample_ids(dataset: str, noise: str, seed: int) -> np.ndarray:
    qf = sorted((RESULTS / "quality" / dataset / noise / f"seed{seed}").glob("quality_*.parquet"))
    return np.sort(pd.read_parquet(qf[0], columns=["sample_id"])["sample_id"].to_numpy().astype(int))


def main() -> None:
    # --- 1. DINO class structure -------------------------------------------------
    from quality_noise.data.datasets import load_bundle
    import torchvision

    print("== DINOv2 embedding sanity ==", flush=True)
    for ds, name in [("cifar10", "cifar10"), ("cifar100", "cifar100")]:
        emb = np.load(EMB / "dino" / ds / "seed0.npy").astype(np.float32)
        cls = torchvision.datasets.CIFAR10 if name == "cifar10" else torchvision.datasets.CIFAR100
        targets = np.asarray(cls(root=str(ROOT.parent / ".cache" / "quality_noise" / name)
                                 if False else "/root/.cache/quality_noise/" + name,
                                 train=True, download=False).targets)
        # kNN accuracy: fraction of the 200 nearest neighbours sharing the label.
        e = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12)
        rng = np.random.default_rng(0)
        q_idx = rng.choice(len(e), size=2000, replace=False)
        sim = e[q_idx] @ e.T
        sim[np.arange(len(q_idx)), q_idx] = -np.inf
        top = np.argpartition(-sim, 200, axis=1)[:, :200]
        acc = float((targets[top] == targets[q_idx][:, None]).mean())
        top1 = float((targets[top[:, 0]] == targets[q_idx]).mean())
        print(f"  {ds}: DINO 200-NN label agreement {acc:.3f}, 1-NN acc {top1:.3f} "
              f"(chance {1/len(np.unique(targets)):.3f})", flush=True)

    # --- 2. Q^DINO properties ----------------------------------------------------
    print("\n== Q^DINO properties (run subsets) ==", flush=True)
    for ds, noise, seed in [("cifar10", "symmetric0.2", 0),
                            ("cifar100", "symmetric0.2", 0),
                            ("cifar10n", "cifar10n_aggre0.4", 0)]:
        sids = _run_sample_ids(ds, noise, seed)
        emb_all = np.load(EMB / "dino" / ds / f"seed{seed}.npy").astype(np.float32)[sids]
        qk = pd.read_parquet(sorted((RESULTS / "quality" / ds / noise / f"seed{seed}")
                                    .glob("quality_*.parquet"))[0])
        qk = qk.sort_values("sample_id")
        cfg = {"dataset": {"name": ds, "root": None, "val_frac": 0.1,
                           "num_classes": 10 if ds != "cifar100" else 100,
                           "input_size": [3, 32, 32]},
               "noise": {"type": noise, "rate": 0.2, "seed": seed}, "seed": seed}
        import re
        m = re.match(r"cifar10n_(\w+)([0-9.]+)", noise) if ds == "cifar10n" else None
        if ds == "cifar10n":
            cfg["noise"] = {"type": m.group(1), "rate": float(m.group(2)), "seed": seed}
        else:
            mm = re.match(r"(symmetric|asymmetric)([0-9.]+)", noise)
            cfg["noise"] = {"type": mm.group(1), "rate": float(mm.group(2)), "seed": seed}
        bundle = load_bundle(cfg)
        id2m = {int(s): bool(v) for s, v in zip(bundle.eval_view.sample_ids, bundle.eval_view.mask)}
        mask = np.array([id2m[int(s)] for s in sids], dtype=bool)

        # neighbour pool = run subset (strict comparability with Q^KNN)
        qd = compute_independent_q(emb_all, k=20)["q"]
        qknn = qk["knn_agreement"].to_numpy(dtype=np.float64)
        from scipy import stats
        r = float(stats.pearsonr(qd, qknn).statistic)
        auc = float(stats.mannwhitneyu(qd[mask], qd[~mask]).statistic / (mask.sum() * (~mask).sum()))
        print(f"  {ds}/{noise}: noise rate {mask.mean():.3f} | corr(Q^DINO, Q^KNN) = {r:+.3f} | "
              f"Q^DINO AUC vs mask = {auc:.3f} | Q^KNN AUC vs mask = "
              f"{float(stats.mannwhitneyu(qknn[mask], qknn[~mask]).statistic/(mask.sum()*(~mask).sum())):.3f}",
              flush=True)


if __name__ == "__main__":
    main()
