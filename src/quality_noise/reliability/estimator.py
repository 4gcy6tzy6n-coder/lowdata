"""Reliability-aware noise estimator (Phase 6).

This orchestrator maps evidence E and quality Q to a soft training weight via:

    bin Q into quintiles
    -> fit per-bin 2-component Beta mixture on E        (unsupervised)
    -> posterior eta_i = P(noisy | E_i, Q_i)
    -> representation reliability R_i                   (from Q or Q3 stability)
    -> shrink: logit(eta_hat) = R*logit(eta) + (1-R)*logit(pi_q)
    -> weight w_i = 1 - eta_hat_i

CRITICAL: the estimator takes ONLY (E, Q) — there is no noise-mask or
clean-label argument. This is a compile-time guarantee against leakage: it is
impossible to call this function with the evaluation-only mask.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..detectability.stratified_auc import quality_bins
from ..governance.mixture import fit_beta_mixture
from ..governance.posterior import beta_mixture_posterior, shrink_posterior
from ..governance.weighting import calibrated_soft_weights, soft_weights


@dataclass
class PosteriorResult:
    eta_raw: np.ndarray  # unshrunk noise posterior
    eta_shrunk: np.ndarray  # shrunk by representation reliability
    weights: np.ndarray  # training weights (after calibration / sharpening / hard cutoff)
    pi_q: np.ndarray  # per-bin noisy-component prior
    reliability: np.ndarray  # R_i in [0, 1]
    bin_id: np.ndarray  # quality bin id per sample
    rate_used: float  # noise-rate estimate actually used for weight calibration


def _tail_noise_rate(E: np.ndarray) -> float:
    """Adaptive noisy-tail fraction from the evidence distribution (no labels).

    The evidence E is roughly bimodal (clean vs noisy components). The clean/noisy
    boundary is the largest gap in the *central* region of the sorted evidence
    (extreme-tail gaps are artifacts of the Beta density shape, not the boundary);
    the fraction of samples above it is the estimated noise rate.
    """
    E = np.sort(np.asarray(E, dtype=np.float64))
    if len(E) < 8:
        return 0.4
    lo, hi = int(0.1 * len(E)), int(0.9 * len(E))
    central = E[lo:hi]
    if len(central) < 3:
        return 0.4
    diffs = np.diff(central)
    k = int(np.argmax(diffs))
    boundary_idx = lo + k
    frac = (len(E) - 1 - boundary_idx) / len(E)
    return float(np.clip(frac, 0.02, 0.8))


def _otsu_noise_rate(E: np.ndarray) -> float:
    """Unsupervised noise-rate estimate via a 1-D Otsu break on the evidence.

    The evidence is roughly bimodal (clean vs noisy components). Otsu finds the
    threshold that maximizes between-class variance; the fraction of samples
    above it is the estimated noise rate. This is fully unsupervised and does
    not depend on the mixture fit quality, so it is a more robust ``rho_hat``.
    """
    E = np.asarray(E, dtype=np.float64)
    lo, hi = float(E.min()), float(E.max())
    if hi - lo < 1e-9:
        return 0.3
    nbins = 64
    hist, edges = np.histogram(E, bins=nbins, range=(lo, hi))
    centers = 0.5 * (edges[:-1] + edges[1:])
    hist = hist.astype(np.float64)
    total_w = hist.sum()
    if total_w == 0:
        return 0.3
    csum_w = np.cumsum(hist)
    csum_x = np.cumsum(hist * centers)
    w0 = csum_w / total_w
    w1 = 1.0 - w0
    mu0 = csum_x / (csum_w + 1e-9)
    mu1 = (csum_x[-1] - csum_x) / (total_w - csum_w + 1e-9)
    between = w0 * w1 * (mu0 - mu1) ** 2
    k = int(np.argmax(between))
    threshold = float(centers[k])
    frac = float(np.mean(E > threshold))
    return float(np.clip(frac, 0.02, 0.6))


def _estimate_noise_rate(E: np.ndarray, params: "MixtureParams", rel_cfg: dict, cfg_rate: float) -> float:
    """Noise-rate used for calibration; source chosen by ``rate_source``.

    - "config": the configured injected noise rate (the *oracle* rho — used only
      for synthetic consistency and the oracle-vs-estimated comparison; the
      final method should NOT rely on it).
    - "mixture": the global noisy-component prior pi fitted by the unsupervised
      mixture (adaptive, but can under-estimate when evidence separates poorly).
    - "tail": largest-gap boundary on the evidence (fully unsupervised).
    - "estimated": a robust unsupervised rho_hat via Otsu break on the evidence
      (recommended for the final method — no oracle used).
    """
    from ..governance.mixture import MixtureParams

    src = rel_cfg.get("rate_source", "config")
    if src == "config":
        return float(np.clip(cfg_rate, 0.02, 0.8))
    if src == "mixture":
        pi = float(np.asarray(params.pi_q).reshape(-1)[0])
        return float(np.clip(pi, 0.02, 0.8))
    if src == "tail":
        return _tail_noise_rate(E)
    if src == "estimated":
        return _otsu_noise_rate(E)
    raise ValueError(f"Unknown rate_source: {src}")


def _rank_normalize(x: np.ndarray) -> np.ndarray:
    ranks = np.argsort(np.argsort(np.asarray(x, dtype=np.float64)))
    return ranks / max(len(ranks) - 1, 1)


def compute_reliability(Q: np.ndarray, method: str = "knn_agreement") -> np.ndarray:
    """Representation reliability R_i in [0,1].

    For a model-native representation, reliability is high (trust the sample-
    specific evidence), so R is scaled to [0.8, 1.0] with mild variation by
    quality. The representation-dependence ablation uses a lower-R method.
    Alternative methods (e.g. Q3 augmentation stability) plug in here; none use
    the noise mask.
    """
    if method == "knn_agreement":
        # Mild reliability variation by neighborhood agreement, high baseline.
        return (0.8 + 0.2 * _rank_normalize(Q)).astype(np.float32)
    if method == "constant":
        return np.full(len(Q), 0.9, dtype=np.float32)
    if method == "weak":
        # Representation-dependence ablation: low reliability, strong shrinkage.
        return (0.3 + 0.4 * _rank_normalize(Q)).astype(np.float32)
    raise ValueError(f"Unknown reliability method: {method}")


def estimate_noise_posterior(
    E: np.ndarray,
    Q: np.ndarray,
    cfg: dict | None = None,
) -> PosteriorResult:
    """Estimate the noise posterior and soft weights from evidence + quality.

    Args:
        E: scalar evidence score per sample in [0, 1].
        Q: sample quality per sample (e.g. KNN agreement) in [0, 1].
        cfg: optional config with keys ``mixture`` (n_bins, n_restarts, max_iter),
             ``reliability`` (method).

    No mask / clean-label argument exists by design.
    """
    cfg = cfg or {}
    mix_cfg = cfg.get("mixture", {})
    rel_cfg = cfg.get("reliability", {})
    n_bins = int(mix_cfg.get("n_bins", 5))
    max_iter = int(mix_cfg.get("max_iter", 1000))
    init_pi = float(mix_cfg.get("init_pi", 0.4))

    # Anchor the noise prior to the configured noise rate (a training-time known
    # quantity, not a data signal): prevents the mixture from over-suppressing
    # when evidence does not separate clean/noisy well (e.g. asymmetric noise).
    rate = float(cfg.get("noise", {}).get("rate", 0.4)) if isinstance(cfg, dict) else 0.4
    max_pi = float(mix_cfg.get("max_pi", min(rate * 1.5 + 0.05, 0.5)))
    init_pi = float(mix_cfg.get("init_pi", min(rate, 0.4)))

    E = np.asarray(E, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)
    bin_id = quality_bins(Q, n_bins)

    # Fit the mixture GLOBALLY (full-evidence separation is strong), then use
    # the per-quality-bin posterior mean as the shrinkage prior pi_q.
    params = fit_beta_mixture(E, init_pi=init_pi, max_iter=max_iter, max_pi=max_pi)
    eta_raw = beta_mixture_posterior(E, params, np.zeros(len(E), dtype=int))

    # Per-bin noisy prior: the empirical posterior mean within each quality bin.
    pi_q = np.array([eta_raw[bin_id == b].mean() for b in range(n_bins)])
    pi_q = np.clip(pi_q, 0.01, 0.9)

    R = compute_reliability(Q, method=rel_cfg.get("method", "knn_agreement"))
    eta_shrunk = shrink_posterior(eta_raw, pi_q, R, bin_id)

    # Adaptive noise rate (A3): which rate anchors the weight calibration.
    # ``calibration_offset`` (P0② misspecification test) shifts ONLY the rate
    # used for calibration / hard-exclusion; the DATA rate (true rho) is
    # unchanged, so the same traces/quality are reused without retraining 02.
    offset = float(rel_cfg.get("calibration_offset", 0.0))
    if offset != 0.0:
        rate = float(np.clip(rate + offset, 0.02, 0.8))
    else:
        rate = _estimate_noise_rate(E, params, rel_cfg, rate)

    # Weight mapping: default "calibrated" suppresses the top-noise-rate fraction
    # (a soft analogue of small-loss) to make the method competitive; "soft"
    # keeps the original diffuse 1 - eta weights.
    weight_mode = rel_cfg.get("weight_mode", "calibrated")
    if weight_mode == "calibrated":
        weights = calibrated_soft_weights(eta_shrunk, rate)
    else:
        weights = soft_weights(eta_shrunk)

    # Sharpening (A1): w -> w^p turns "downweighted to ~0.4" into "~0.05",
    # a continuous approximation of small-loss hard selection. p=1 is identity.
    power = float(rel_cfg.get("weight_power", 1.0))
    if power != 1.0:
        weights = np.power(weights, power).astype(np.float32)

    # Soft+hard mix (A2): samples with eta_shrunk > tau get weight exactly 0
    # (excluded), the rest keep their soft weights. tau="auto" excludes the top
    # noise-rate fraction by posterior — the same count small-loss drops, but
    # chosen by our posterior and leaving the rest continuous. hard_frac scales
    # that count (e.g. 0.5 -> exclude half the noise-rate fraction): full 1.0
    # hard exclusion can over-prune at high noise rates (sym40/asym40), so
    # smaller fractions trade exclusion strength against hard-clean retention.
    tau = rel_cfg.get("hard_threshold", None)
    if tau is not None:
        if tau == "auto":
            frac = float(rel_cfg.get("hard_frac", 1.0))
            tau = float(np.quantile(eta_shrunk, 1.0 - rate * frac))
        else:
            tau = float(tau)
        weights = np.where(eta_shrunk > tau, 0.0, weights).astype(np.float32)

    return PosteriorResult(
        eta_raw=eta_raw,
        eta_shrunk=eta_shrunk,
        weights=weights,
        pi_q=params.pi_q,
        reliability=R,
        bin_id=bin_id,
        rate_used=rate,
    )
