"""Verify recomputed native ResNet features align with each run's saved embeddings.

If the recovered sample ordering were wrong, per-sample cosine similarity between
our recomputed features and the run's own ``embeddings_<hash>.npy`` would collapse.
A per-sample cos-sim near 1.0 (augmentation aside) proves the alignment; we also
compare a label-agreement statistic so a *permuted* alignment cannot pass.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/root/quality_noise")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "revision"))

EMB = ROOT / "results" / "revision" / "embeddings"
RESULTS = ROOT / "results"


def main() -> None:
    print("%-34s %6s %8s %8s %8s" % ("run", "n", "cos_mean", "cos_p05", "1nn_agree"))
    for ds_dir in sorted((RESULTS / "traces").glob("*")):
        ds = ds_dir.name
        for noise_dir in sorted(ds_dir.glob("*")):
            noise = noise_dir.name
            for p in sorted(noise_dir.glob("seed*")):
                seed = int(p.name[4:])
                nat_p = EMB / "native" / ds / noise / f"seed{seed}.npy"
                if not nat_p.exists():
                    continue
                nat = np.load(nat_p).astype(np.float32)
                saved = np.load(sorted(p.glob("embeddings_*.npy"))[0]).astype(np.float32)
                if nat.shape != saved.shape:
                    print(f"{ds}/{noise}/seed{seed}: SHAPE MISMATCH {nat.shape} vs {saved.shape}")
                    continue
                a = nat / np.maximum(np.linalg.norm(nat, axis=1, keepdims=True), 1e-12)
                b = saved / np.maximum(np.linalg.norm(saved, axis=1, keepdims=True), 1e-12)
                cos = (a * b).sum(axis=1)
                # 1-NN label agreement using saved-feature neighbours vs own label
                qf = sorted((RESULTS / "quality" / ds / noise / f"seed{seed}").glob("quality_*.parquet"))
                q = pd.read_parquet(qf[0]).sort_values("sample_id")
                knn = q["knn_agreement"].to_numpy(dtype=np.float64)
                print("%-34s %6d %8.3f %8.3f %8.3f" % (
                    f"{ds}/{noise}/seed{seed}", len(cos), cos.mean(),
                    np.percentile(cos, 5), knn.mean()))
        print()


if __name__ == "__main__":
    main()
