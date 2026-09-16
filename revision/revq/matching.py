"""Matching variants, caliper sweep, balance diagnostics and paired bootstrap.

Matching is the identification device of the paper: a detector is scored by
comparing noisy samples against clean samples of *comparable quality*, so that
apparent skill cannot be explained by the quality gap between the two groups.

Variants (the SAME quality covariate Q; only the matching rule changes):

``nn_wo``     1:1 nearest neighbour **without** replacement (default; optimal
              min-sum |dQ| assignment via Hungarian inside the caliper)
``nn_wr``     1:1 nearest neighbour **with** replacement (legacy behaviour)
``class_cond`` class-conditioned matching (noisy and clean share y_observed)
``trimmed``   common-support trimmed matching (restricted to the Q overlap
              interval, optionally the robust 1%-99% overlap)

The caliper is expressed in **standardised Q units** (``eps_sd`` x SD(Q));
legacy raw-Q ``eps`` is still accepted.

All helpers return, per matched pair, both the *noisy index* and the *clean
index*, so every downstream statistic can be audited back to individual samples.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

# Above this many (noisy x clean) candidate entries we fall back to greedy local
# matching, which on these sample sizes reproduces the optimal assignment.
HUNGARIAN_CAP = 80_000_000


@dataclass
class MatchedPairs:
    """Matched noisy/clean pairs plus everything needed to audit the match."""

    n_idx: np.ndarray          # (P,) noisy sample indices
    c_idx: np.ndarray          # (P,) matched clean sample indices
    s_noisy: np.ndarray        # (P,) detector scores (same order as n_idx)
    s_clean: np.ndarray        # (P,) detector scores
    q_noisy: np.ndarray        # (P,)
    q_clean: np.ndarray        # (P,)
    n_noisy_total: int = 0
    n_clean_total: int = 0
    strategy: str = "nn_wo"
    caliper: float = 0.0
    common_support: tuple[float, float] = (float("nan"), float("nan"))
    extra: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Caliper / common support
# ---------------------------------------------------------------------------

def resolve_caliper(q: np.ndarray, eps: float | None, eps_sd: float | None) -> float:
    """Caliper in raw Q units, from an SD multiple when ``eps_sd`` is given."""
    if eps_sd is not None:
        return float(eps_sd * np.std(q))
    if eps is not None:
        return float(eps)
    raise ValueError("one of eps / eps_sd is required")


def common_support(q_noisy: np.ndarray, q_clean: np.ndarray,
                   robust: bool = False) -> tuple[float, float]:
    """Overlap interval of two quality distributions (1%-99% when ``robust``)."""
    if robust:
        a, b = np.percentile(q_noisy, [1, 99])
        c, d = np.percentile(q_clean, [1, 99])
    else:
        a, b = np.min(q_noisy), np.max(q_noisy)
        c, d = np.min(q_clean), np.max(q_clean)
    lo, hi = max(float(a), float(c)), min(float(b), float(d))
    return (lo, hi) if lo <= hi else (float("nan"), float("nan"))


# ---------------------------------------------------------------------------
# Matching primitives (each returns noisy_idx, clean_idx)
# ---------------------------------------------------------------------------

def _hungarian_wo(n_idx: np.ndarray, c_idx: np.ndarray, q: np.ndarray,
                  eps: float, max_pairs_per_noisy: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Optimal 1:1 min-sum |dQ| assignment inside the caliper."""
    cost = np.abs(q[n_idx][:, None] - q[c_idx][None, :])
    big = np.where(cost <= eps, cost, 1e6)
    ri, ci = linear_sum_assignment(big)
    keep = big[ri, ci] <= eps
    ri, ci = ri[keep], ci[keep]
    return n_idx[ri], c_idx[ci]


def _greedy_wo(n_idx: np.ndarray, c_idx: np.ndarray, q: np.ndarray, eps: float,
               max_clean_per_noisy: int, candidate_cap: int) -> tuple[np.ndarray, np.ndarray]:
    """Greedy nearest-neighbour without replacement (used inside bootstrap)."""
    order = np.argsort(q[c_idx])
    q_sorted = q[c_idx][order]
    used = np.zeros(len(c_idx), dtype=bool)
    ni_out, ci_out = [], []
    for i in n_idx:
        nq = q[i]
        lo = int(np.searchsorted(q_sorted, nq - eps, side="left"))
        hi = int(np.searchsorted(q_sorted, nq + eps, side="right"))
        if hi <= lo:
            continue
        w = np.arange(lo, hi)
        if len(w) > candidate_cap:
            dn = np.abs(q_sorted[w] - nq)
            w = w[np.argpartition(dn, candidate_cap - 1)[:candidate_cap]]
        free = w[~used[w]]
        if len(free) == 0:
            continue
        dn = np.abs(q_sorted[free] - nq)
        k = min(max_clean_per_noisy, len(free))
        pick = free[np.argpartition(dn, k - 1)[:k]]
        pick = pick[np.argsort(np.abs(q_sorted[pick] - nq))]
        used[pick] = True
        for p in pick:
            ni_out.append(i)
            ci_out.append(c_idx[order[p]])
    return np.asarray(ni_out, dtype=np.int64), np.asarray(ci_out, dtype=np.int64)


def _greedy_wr(n_idx: np.ndarray, c_idx: np.ndarray, q: np.ndarray, eps: float,
               max_clean_per_noisy: int, candidate_cap: int) -> tuple[np.ndarray, np.ndarray]:
    """Greedy nearest-neighbour with replacement (legacy matching rule)."""
    order = np.argsort(q[c_idx])
    q_sorted = q[c_idx][order]
    ni_out, ci_out = [], []
    for i in n_idx:
        nq = q[i]
        lo = int(np.searchsorted(q_sorted, nq - eps, side="left"))
        hi = int(np.searchsorted(q_sorted, nq + eps, side="right"))
        if hi <= lo:
            continue
        w = np.arange(lo, hi)
        if len(w) > candidate_cap:
            dn = np.abs(q_sorted[w] - nq)
            w = w[np.argpartition(dn, candidate_cap - 1)[:candidate_cap]]
        dn = np.abs(q_sorted[w] - nq)
        k = min(max_clean_per_noisy, len(w))
        pick = w[np.argpartition(dn, k - 1)[:k]]
        pick = pick[np.argsort(np.abs(q_sorted[pick] - nq))]
        for p in pick:
            ni_out.append(i)
            ci_out.append(c_idx[order[p]])
    return np.asarray(ni_out, dtype=np.int64), np.asarray(ci_out, dtype=np.int64)


def _match_one(score, n_idx, c_idx, q, caliper, max_clean_per_noisy,
               candidate_cap, with_replacement, force_greedy):
    """Dispatch to the optimal assignment or to greedy local matching."""
    n, m = len(n_idx), len(c_idx)
    if n == 0 or m == 0:
        e = np.array([], dtype=np.int64)
        return e, e
    if not with_replacement and not force_greedy and n * m <= HUNGARIAN_CAP:
        ni, ci = _hungarian_wo(n_idx, c_idx, q, caliper)
    elif with_replacement:
        ni, ci = _greedy_wr(n_idx, c_idx, q, caliper, max_clean_per_noisy, candidate_cap)
    else:
        ni, ci = _greedy_wo(n_idx, c_idx, q, caliper, max_clean_per_noisy, candidate_cap)
    return ni, ci


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def match_pairs(
    score: np.ndarray,
    mask: np.ndarray,
    q: np.ndarray,
    *,
    strategy: str = "nn_wo",
    eps: float | None = None,
    eps_sd: float | None = None,
    max_clean_per_noisy: int = 5,
    candidate_cap: int = 512,
    y_observed: np.ndarray | None = None,
    robust_support: bool = False,
    force_greedy: bool = False,
) -> MatchedPairs:
    """Match noisy samples to clean samples of comparable quality.

    ``score`` is detector evidence (higher = more likely noisy).
    ``strategy`` in {"nn_wo", "nn_wr", "class_cond", "trimmed"}.
    """
    score = np.asarray(score, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    q = np.asarray(q, dtype=np.float64)
    caliper = resolve_caliper(q, eps, eps_sd)

    n_all = np.where(mask)[0]
    c_all = np.where(~mask)[0]
    support = common_support(q[n_all], q[c_all], robust=robust_support) if (len(n_all) and len(c_all)) \
        else (float("nan"), float("nan"))

    if len(n_all) == 0 or len(c_all) == 0:
        e = np.array([])
        return MatchedPairs(np.array([], dtype=np.int64), np.array([], dtype=np.int64),
                            e, e, e, e, len(n_all), len(c_all), strategy, caliper, support)

    if strategy == "trimmed":
        if not np.isfinite(support[0]):
            e = np.array([])
            return MatchedPairs(np.array([], dtype=np.int64), np.array([], dtype=np.int64),
                                e, e, e, e, len(n_all), len(c_all), strategy, caliper,
                                support, {"reason": "disjoint common support"})
        lo, hi = support
        n_idx = n_all[(q[n_all] >= lo) & (q[n_all] <= hi)]
        c_idx = c_all[(q[c_all] >= lo) & (q[c_all] <= hi)]
        ni, ci = _match_one(score, n_idx, c_idx, q, caliper, max_clean_per_noisy,
                            candidate_cap, False, force_greedy)
    elif strategy == "class_cond":
        if y_observed is None:
            raise ValueError("class_cond matching requires y_observed")
        y = np.asarray(y_observed)
        ni_l, ci_l = [], []
        for cls in np.unique(y):
            nk = n_all[y[n_all] == cls]
            ck = c_all[y[c_all] == cls]
            if len(nk) == 0 or len(ck) == 0:
                continue
            a, b = _match_one(score, nk, ck, q, caliper, max_clean_per_noisy,
                              candidate_cap, False, force_greedy)
            if len(a):
                ni_l.append(a); ci_l.append(b)
        ni = np.concatenate(ni_l) if ni_l else np.array([], dtype=np.int64)
        ci = np.concatenate(ci_l) if ci_l else np.array([], dtype=np.int64)
    elif strategy in ("nn_wo", "nn_wr"):
        ni, ci = _match_one(score, n_all, c_all, q, caliper, max_clean_per_noisy,
                            candidate_cap, strategy == "nn_wr", force_greedy)
    else:
        raise ValueError(f"unknown strategy: {strategy}")

    ni = np.asarray(ni, dtype=np.int64)
    ci = np.asarray(ci, dtype=np.int64)
    return MatchedPairs(
        n_idx=ni, c_idx=ci,
        s_noisy=score[ni] if len(ni) else np.array([]),
        s_clean=score[ci] if len(ci) else np.array([]),
        q_noisy=q[ni] if len(ni) else np.array([]),
        q_clean=q[ci] if len(ci) else np.array([]),
        n_noisy_total=len(n_all), n_clean_total=len(c_all),
        strategy=strategy, caliper=caliper, common_support=support,
    )


def matched_auc_fast(score: np.ndarray, mask: np.ndarray, q: np.ndarray,
                     eps: float, clean_budget: int = 400) -> float:
    """P(score_noisy > score_clean | |Q_i - Q_j| <= eps), vectorised.

    Equivalent in value to ``matched_auc(match_pairs(..., max_clean_per_noisy=m))``
    for large ``m`` (the caller always used m = 5 with a 512-wide candidate
    window, i.e. far more pairs than the caliper band contains), but computed
    with one ``searchsorted`` per noisy sample instead of a Python loop over
    matching bookkeeping.  Built for the bootstrap inner loop, where it runs
    thousands of times.

    For each noisy sample the clean candidates inside the quality caliper are
    the contiguous block found by ``searchsorted`` on the sorted clean
    qualities; ties in Q make the paired-index arithmetic ambiguous, so draws
    are capped at ``clean_budget`` candidates per noisy sample.
    """
    score = np.asarray(score, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    q = np.asarray(q, dtype=np.float64)

    n_idx = np.where(mask)[0]
    c_idx = np.where(~mask)[0]
    if len(n_idx) == 0 or len(c_idx) == 0:
        return float("nan")

    order = np.argsort(q[c_idx], kind="stable")
    c_sorted = c_idx[order]
    q_sorted = q[c_sorted]

    lo = np.searchsorted(q_sorted, q[n_idx] - eps, side="left")
    hi = np.searchsorted(q_sorted, q[n_idx] + eps, side="right")
    lo = np.minimum(lo, len(c_sorted))
    hi = np.minimum(hi, len(c_sorted))
    width = np.maximum(hi - lo, 0)
    keep = width > 0
    if not keep.any():
        return float("nan")
    lo, width = lo[keep], np.minimum(width[keep], clean_budget)
    n_kept = n_idx[keep]
    total_pairs = int(width.sum())

    offs = np.repeat(lo, width)
    offs += np.arange(total_pairs, dtype=np.int64) - np.repeat(
        np.concatenate([[0], np.cumsum(width)[:-1]]), width)
    c_sel = c_sorted[offs]
    n_sel = np.repeat(n_kept, width)
    sn = score[n_sel]
    sc = score[c_sel]
    wins = float((sn > sc).sum())
    ties = float((sn == sc).sum())
    return (wins + 0.5 * ties) / max(total_pairs, 1)


def matched_auc(pairs: MatchedPairs) -> float:
    """P(s_noisy > s_clean) over matched pairs (ties -> 0.5)."""
    if len(pairs.s_noisy) == 0:
        return float("nan")
    wins = float((pairs.s_noisy > pairs.s_clean).mean())
    ties = float((pairs.s_noisy == pairs.s_clean).mean())
    return wins + 0.5 * ties


def smd(x_noisy: np.ndarray, x_clean: np.ndarray) -> float:
    """Standardised mean difference (mu_noisy - mu_clean) / pooled SD."""
    x_noisy = np.asarray(x_noisy, dtype=np.float64)
    x_clean = np.asarray(x_clean, dtype=np.float64)
    if len(x_noisy) == 0 or len(x_clean) == 0:
        return float("nan")
    vn, vc = float(x_noisy.var()), float(x_clean.var())
    pooled = np.sqrt(((len(x_noisy) - 1) * vn + (len(x_clean) - 1) * vc) /
                     max(len(x_noisy) + len(x_clean) - 2, 1))
    if pooled < 1e-12:
        return 0.0
    return float((x_noisy.mean() - x_clean.mean()) / pooled)


def balance_diagnostics(pairs: MatchedPairs, q: np.ndarray, mask: np.ndarray) -> dict:
    """SMD before/after, coverage, pair distances — the matching audit trail."""
    mask = np.asarray(mask, dtype=bool)
    q = np.asarray(q, dtype=np.float64)
    n_tot, c_tot = int(mask.sum()), int((~mask).sum())
    p = len(pairs.s_noisy)
    diffs = np.abs(pairs.q_noisy - pairs.q_clean) if p else np.array([])
    sd = float(np.std(q))
    return {
        "strategy": pairs.strategy,
        "caliper": float(pairs.caliper),
        "caliper_sd_units": float(pairs.caliper / sd) if sd > 0 else float("nan"),
        "n_noisy": n_tot,
        "n_clean": c_tot,
        "n_pairs": int(p),
        "n_noisy_matched": int(len(np.unique(pairs.n_idx))) if p else 0,
        "n_clean_matched": int(len(np.unique(pairs.c_idx))) if p else 0,
        "coverage_noisy": float(len(np.unique(pairs.n_idx)) / n_tot) if (p and n_tot) else float("nan"),
        "coverage_clean": float(len(np.unique(pairs.c_idx)) / c_tot) if (p and c_tot) else float("nan"),
        "smd_before": smd(q[mask], q[~mask]),
        "smd_after": smd(pairs.q_noisy, pairs.q_clean),
        "mean_abs_dq": float(diffs.mean()) if diffs.size else float("nan"),
        "max_abs_dq": float(diffs.max()) if diffs.size else float("nan"),
        "common_support_lo": pairs.common_support[0],
        "common_support_hi": pairs.common_support[1],
    }
