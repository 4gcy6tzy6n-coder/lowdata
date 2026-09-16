"""Precompute DINOv2 density quality over the full CIFAR train split.

The per-run covariate uses the run's own 45k sample pool (strict comparability
with the paper's Q^KNN).  The full-pool variant lets every sample draw its
neighbours from all 50k train images, which is the more natural density estimate
and a useful robustness check against the train/val split boundary.

Output: ``<emb>/dino/<dataset>/density_fullpool.npy`` (50000,) float32.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path("/root/quality_noise")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "revision"))

from revq.independent_q import compute_independent_q  # noqa: E402

EMB = ROOT / "results" / "revision" / "embeddings" / "dino"


def main() -> None:
    for ds in ["cifar10", "cifar100", "cifar10n", "cifar100n"]:
        out = EMB / ds / "density_fullpool.npy"
        if out.exists():
            print(f"{ds}: cached", flush=True)
            continue
        emb = np.load(EMB / ds / "seed0.npy").astype(np.float32)
        r = compute_independent_q(emb, k=20)
        np.save(out, r["q"].astype(np.float32))
        np.save(EMB / ds / "density_fullpool_raw.npy", r["raw"].astype(np.float32))
        print(f"{ds}: density computed, q in [{r['q'].min():.3f}, {r['q'].max():.3f}], "
              f"mean_dist mean {r['mean_dist'].mean():.3f}", flush=True)


if __name__ == "__main__":
    main()
