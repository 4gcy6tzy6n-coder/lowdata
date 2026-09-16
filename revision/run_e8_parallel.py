"""E8 (parallel): mutual information I(Q; Z) plus a label-permutation null.

The sequential implementation in `run_round2.run_mi` costs ~30 min per setting
(2 settings x 10 seeds x 8 proxies x 1001 MI fits at ~11 ms each).  This driver
parallelises over (seed, proxy) cells so the same numbers come out in a couple of
minutes on a many-core box.

The estimator is `sklearn.feature_selection.mutual_info_classif` with
`discrete_features=False` (kNN-based Kozachenko-Leonenko estimator), applied to
the one-dimensional proxy score Q and the binary noisy/clean indicator Z.  The
null is a label permutation of Z (1000 draws), which breaks any Q-Z association
while preserving both marginals.

Output schema is identical to `run_round2.run_mi` so downstream analysis does
not care which driver produced the CSV.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "revision"))

from run_p0_batch import CACHE, PROXIES, PROXY_FAMILY, rank_auc  # noqa: E402

OUT = ROOT / "results" / "revision" / "round2"
SETTINGS = {"s20": ("cifar100", "symmetric0.2"),
            "a40": ("cifar100", "asymmetric0.4")}
N_PERM = 1000


def _mi(q: np.ndarray, z: np.ndarray, seed: int) -> float:
    from sklearn.feature_selection import mutual_info_classif
    return float(mutual_info_classif(q, z, discrete_features=False,
                                     random_state=seed)[0])


def _cell(task):
    """One (setting, seed, proxy) cell: point MI, permutation null, rank AUC."""
    setting_key, seed, px, n_perm = task
    dataset, noise = SETTINGS[setting_key]
    with np.load(CACHE / dataset / noise / f"seed{seed}.npz",
                 allow_pickle=False) as p:
        mask = np.asarray(p["mask"], dtype=bool)
        q = np.asarray(p[f"sig__{px}"], dtype=np.float64).reshape(-1, 1)
    z = mask.astype(int)
    # The MI estimator needs a float64 C-contiguous column; `z` is int64.
    mi = _mi(q, z.astype(np.float64), seed)
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm, dtype=np.float64)
    for i in range(n_perm):
        null[i] = _mi(q, rng.permutation(z).astype(np.float64), seed)
    a_qz = rank_auc(mask, q.ravel())
    return {
        "setting": f"{dataset}/{noise}",
        "seed": int(seed),
        "proxy": px,
        "proxy_family": PROXY_FAMILY.get(px, "?"),
        "mutual_information": mi,
        "mi_null_mean": float(null.mean()),
        "mi_null_sd": float(null.std(ddof=1)),
        "mi_null_q975": float(np.quantile(null, 0.975)),
        "mi_perm_p": float((null >= mi).mean()),
        "mi_z": float((mi - null.mean()) / (null.std(ddof=1) or np.nan)),
        "rq_abs_auc_minus_half": abs(a_qz - 0.5),
        "auc_q_vs_z": a_qz,
        "n_perm": int(n_perm),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--setting", default="all", choices=["all", "s20", "a40"])
    ap.add_argument("--seeds", default="all")
    ap.add_argument("--n_perm", type=int, default=N_PERM)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    ap.add_argument("--out", default="e8_mutual_information.csv")
    args = ap.parse_args()

    keys = list(SETTINGS) if args.setting == "all" else [args.setting]
    tasks = []
    for key in keys:
        dataset, noise = SETTINGS[key]
        seeds = sorted(int(p.stem[4:]) for p in (CACHE / dataset / noise).glob("seed*.npz"))
        if args.seeds != "all":
            want = {int(s) for s in args.seeds.split(",")}
            seeds = [s for s in seeds if s in want]
        for seed in seeds:
            for px in PROXIES:
                tasks.append((key, seed, px, args.n_perm))
    print(f"== e8: {len(tasks)} cells "
          f"({len(set(t[:2] for t in tasks))} seed-settings) x {args.n_perm} permutations",
          flush=True)

    rows = []
    if args.workers > 1:
        with mp.Pool(processes=min(args.workers, len(tasks))) as pool:
            for i, res in enumerate(pool.imap_unordered(_cell, tasks), 1):
                rows.append(res)
                if i % 10 == 0 or i == len(tasks):
                    print(f"   {i}/{len(tasks)} cells done", flush=True)
    else:
        for i, t in enumerate(tasks, 1):
            rows.append(_cell(t))
            print(f"   {i}/{len(tasks)} cells done", flush=True)

    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / args.out, index=False)

    print("\n== e8: I(Q;Z), permutation null and rank AUC (mean over seeds) ==")
    g = (df.groupby(["setting", "proxy", "proxy_family"], as_index=False)
         .agg(mi=("mutual_information", "mean"), mi_null=("mi_null_mean", "mean"),
              mi_z=("mi_z", "mean"), p_min=("mi_perm_p", "min"),
              p_max=("mi_perm_p", "max"), p_mean=("mi_perm_p", "mean"),
              rq_auc=("auc_q_vs_z", "mean"), n_seeds=("seed", "nunique")))
    g = g.sort_values(["setting", "mi"], ascending=[True, False])
    print(g.round(5).to_string(index=False))
    g.to_csv(OUT / args.out.replace(".csv", "_summary.csv"), index=False)


if __name__ == "__main__":
    main()
