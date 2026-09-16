"""Bootstrap confidence intervals for the revision.

Two uncertainty notions are reported, and they answer different questions:

**Sample-level paired bootstrap** (``paired_bootstrap``)
    Resample the *samples* (noisy and clean separately, with replacement) and
    recompute AUC_global, AUC_QC and Delta_Q = AUC_global - AUC_QC inside each
    resample.  Global and matched are recomputed on the *same* resample, so the
    resulting Delta_Q CI is paired.  This answers: "given these runs, how well
    is the global-to-controlled gap pinned down?"  It does NOT account for
    variation across training seeds.

**Seed-level bootstrap** (``seed_bootstrap``)
    Resample *seeds* with replacement and average the per-seed statistic.  This
    answers the headline question: "is the average gap across runs positive?"

Report both; never quote only the first.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

from .matching import (MatchedPairs, match_pairs, matched_auc,
                       matched_auc_fast, resolve_caliper)


def _ci(samples: np.ndarray, ci: float = 0.95) -> tuple[float, float]:
    v = np.asarray(samples, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return (float("nan"), float("nan"))
    a = (1 - ci) / 2
    return (float(np.quantile(v, a)), float(np.quantile(v, 1 - a)))


def rank_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """AUROC via mid-ranks (Mann-Whitney U), much faster than sklearn on 45k rows.

    Identical value to ``roc_auc_score`` with correct tie handling; ~10x cheaper
    because it avoids building the ROC curve.  Used inside every bootstrap
    resample, where it is called thousands of times.
    """
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    n1 = int(labels.sum())
    n0 = len(labels) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(scores, kind="stable")
    s = scores[order]
    ranks = np.empty(len(s), dtype=np.float64)
    i, n = 0, len(s)
    while i < n:
        j = i
        while j + 1 < n and s[j + 1] == s[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((ranks[labels].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def _rank_auc_from_draws(score_n: np.ndarray, draw_n: np.ndarray,
                         score_c: np.ndarray, draw_c: np.ndarray) -> np.ndarray:
    """AUROC per resample from pre-drawn index matrices.

    ``draw_n`` / ``draw_c`` are (R, n) and (R, c) index arrays into the noisy and
    clean pools.  Passing the *same* draws to both the global and the matched
    statistic is what makes the Delta_Q bootstrap paired.

    Per row the value equals the Mann-Whitney mid-rank statistic computed by
    ``rank_auc``: with ``L_c`` = number of noisy scores strictly below clean
    score c and ``E_c`` = number equal to it,

        AUC = sum_c (n - R_c) / (n * c) + 0.5 * sum_c E_c / (n * c)

    where ``R_c = L_c + E_c`` is the count of noisy scores <= c, so ``n - R_c``
    is the count strictly greater (higher score = more likely noisy).  Only the
    noisy scores need sorting; the two counts are then one ``searchsorted`` per
    row.
    """
    bn = np.take(score_n, draw_n, axis=0)
    bc = np.take(score_c, draw_c, axis=0)
    bn.sort(axis=1)                                     # only the noisy side
    n, c = bn.shape[1], bc.shape[1]
    left = np.stack([np.searchsorted(bn[r], bc[r], side="left") for r in range(bn.shape[0])])
    right = np.stack([np.searchsorted(bn[r], bc[r], side="right") for r in range(bn.shape[0])])
    wins = (n - right).sum(axis=1) + 0.5 * (right - left).sum(axis=1)
    return wins / (n * c)


def paired_bootstrap(
    score: np.ndarray,
    mask: np.ndarray,
    q: np.ndarray,
    *,
    strategy: str = "nn_wo",
    eps: float | None = None,
    eps_sd: float | None = None,
    n_resamples: int = 2000,
    ci: float = 0.95,
    seed: int = 0,
    force_greedy: bool = True,
    max_noisy_per_resample: int | None = 3000,
    max_clean_per_resample: int | None = 12000,
    point: tuple[float, float] | None = None,
    y_observed: np.ndarray | None = None,
) -> dict:
    """Paired bootstrap CI for AUC_global, AUC_QC and Delta_Q.

    Both statistics are recomputed on the *same* resample, so the Delta_Q CI is
    paired.  Resample sizes for the global statistic are bounded
    (``max_noisy_per_resample`` / ``max_clean_per_resample``) because the AUC
    variance is already negligible at those sizes and every extra sample costs
    linearly inside the loop; this bounds the runtime without changing the
    inference.

    ``point`` optionally supplies the already-computed full-data
    ``(auc_global, auc_qc)`` so the caller does not recompute the expensive
    optimal assignment.
    """
    score = np.asarray(score, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    q = np.asarray(q, dtype=np.float64)

    n_idx_all = np.where(mask)[0]
    c_idx_all = np.where(~mask)[0]
    n_n, n_c = len(n_idx_all), len(c_idx_all)
    if n_n == 0 or n_c == 0:
        nan = float("nan")
        return {"auc_global": nan, "auc_qc": nan, "delta_q": nan,
                "ci_global": (nan, nan), "ci_qc": (nan, nan), "ci_delta": (nan, nan),
                "n_resamples": 0, "delta_samples": np.array([])}

    if point is None:
        g0 = rank_auc(mask, score)
        m0 = matched_auc(match_pairs(score, mask, q, strategy=strategy, eps=eps,
                                     eps_sd=eps_sd, force_greedy=force_greedy))
    else:
        g0, m0 = float(point[0]), float(point[1])

    n_draw = n_n if max_noisy_per_resample is None else min(n_n, max_noisy_per_resample)
    c_draw = n_c if max_clean_per_resample is None else min(n_c, max_clean_per_resample)
    # Subsample the *pool* once (fixed indices reused across resamples) so the
    # bounded draw is a genuine bootstrap of a bounded sample, not a varying one.
    if n_draw < n_n:
        pool_n = np.random.default_rng(seed + 991).choice(n_idx_all, size=n_draw, replace=False)
    else:
        pool_n = n_idx_all
    if c_draw < n_c:
        pool_c = np.random.default_rng(seed + 992).choice(c_idx_all, size=c_draw, replace=False)
    else:
        pool_c = c_idx_all

    # Matched AUC on the quality caliper is computed by the vectorised helper:
    # identical in value to re-matching inside every resample, but with one
    # searchsorted instead of a Python matching loop (thousands of calls).
    caliper = resolve_caliper(q, eps, eps_sd)

    rng = np.random.default_rng(seed)
    # ONE set of draws drives both statistics, so Delta_Q^(b) = AUC_g^(b) - AUC_QC^(b)
    # is genuinely paired.
    draw_n = rng.integers(0, len(pool_n), size=(n_resamples, len(pool_n)))
    draw_c = rng.integers(0, len(pool_c), size=(n_resamples, len(pool_c)))

    # --- global AUROC for all resamples at once ------------------------------
    g_stats = _rank_auc_from_draws(score[pool_n], draw_n, score[pool_c], draw_c)

    # --- matched AUROC, one resample at a time -------------------------------
    # Needs the joint (score, quality) structure, so it cannot be merged as
    # cheaply as the global one.
    m_stats = np.empty(n_resamples)
    for r in range(n_resamples):
        bn = pool_n[draw_n[r]]
        bc = pool_c[draw_c[r]]
        ridx = np.concatenate([bn, bc])
        rm = mask[ridx]
        if rm.sum() == 0 or (~rm).sum() == 0:
            m_stats[r] = np.nan
            continue
        if strategy in ("nn_wo", "nn_wr"):
            m_stats[r] = matched_auc_fast(score[ridx], rm, q[ridx], caliper)
        else:
            # class-conditioned / trimmed rules need the observed labels and the
            # full matching bookkeeping, so they take the exact path.
            pairs = match_pairs(score[ridx], rm, q[ridx], strategy=strategy,
                                eps=eps, eps_sd=eps_sd, force_greedy=True,
                                max_clean_per_noisy=5,
                                y_observed=None if y_observed is None else y_observed[ridx])
            m_stats[r] = matched_auc(pairs) if len(pairs.s_noisy) else np.nan

    d_stats = g_stats - m_stats
    return {
        "auc_global": g0,
        "auc_qc": m0,
        "delta_q": g0 - m0,
        "ci_global": _ci(g_stats, ci),
        "ci_qc": _ci(m_stats, ci),
        "ci_delta": _ci(d_stats, ci),
        "n_resamples": int(n_resamples),
        "delta_samples": d_stats,
        "p_delta_le_0": float(np.mean(d_stats <= 0)) if np.isfinite(d_stats).any() else float("nan"),
    }


def seed_bootstrap(per_seed_values: np.ndarray, n_resamples: int = 5000,
                   ci: float = 0.95, seed: int = 0) -> dict:
    """Bootstrap CI over *seeds* for a per-seed statistic (e.g. per-seed Delta_Q)."""
    v = np.asarray(per_seed_values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        nan = float("nan")
        return {"mean": nan, "sd": nan, "ci": (nan, nan), "n_seeds": 0,
                "p_le_0": nan, "sign_consistency": nan}
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, v.size, size=(n_resamples, v.size))].mean(axis=1)
    return {
        "mean": float(v.mean()),
        "sd": float(v.std(ddof=1)) if v.size > 1 else float("nan"),
        "ci": _ci(means, ci),
        "n_seeds": int(v.size),
        "p_le_0": float(np.mean(means <= 0)),
        "sign_consistency": float(np.mean(v > 0)),
    }
