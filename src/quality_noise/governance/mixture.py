"""Quality-conditioned empirical mixture (Phase 6).

Within each quality bin, the evidence E is modeled as a 2-component Beta
mixture:
    p(E | Q=q) = pi_q * Beta(alpha1, beta1) + (1 - pi_q) * Beta(alpha0, beta0)
where component 1 (higher mean) corresponds to noisy samples and component 0
to clean samples. The mixture is UNSUPERVISED: it operates only on E and the
quality bin id — never on the noise mask or clean labels.

Beta parameters are fit per bin by EM, with a weighted MLE step solved via
scipy numerical optimization.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


@dataclass
class MixtureParams:
    alpha0: np.ndarray  # (n_bins,) clean component shape a
    beta0: np.ndarray  # (n_bins,) clean component shape b
    alpha1: np.ndarray  # (n_bins,) noisy component shape a
    beta1: np.ndarray  # (n_bins,) noisy component shape b
    pi_q: np.ndarray  # (n_bins,) mixing weight of the noisy component

    @property
    def n_bins(self) -> int:
        return len(self.pi_q)


def _clip_evidence(E: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(E, dtype=np.float64), 1e-6, 1 - 1e-6)


def _beta_logpdf(x: np.ndarray, a: float, b: float) -> np.ndarray:
    from scipy.special import betaln

    return (a - 1) * np.log(x) + (b - 1) * np.log1p(-x) - betaln(a, b)


def _beta_moments(mean: float, var: float) -> tuple[float, float]:
    """Method-of-moments estimates of (a, b) for a Beta."""
    m = float(np.clip(mean, 1e-3, 1 - 1e-3))
    v = float(np.clip(var, 1e-6, m * (1 - m) - 1e-6))
    scale = m * (1 - m) / v - 1.0
    scale = max(scale, 1e-3)
    a = m * scale
    b = (1 - m) * scale
    return max(a, 1e-2), max(b, 1e-2)


def _fit_weighted_beta(x: np.ndarray, w: np.ndarray, init_a: float, init_b: float) -> tuple[float, float]:
    """Weighted MLE of Beta(a, b) by numerical minimization of -weighted NLL."""
    x = _clip_evidence(x)
    w = np.asarray(w, dtype=np.float64)
    total = w.sum()
    if total < 5.0:
        return init_a, init_b

    def nll(params):
        a, b = np.exp(params)  # log-parametrization keeps a,b > 0
        return -float((w * _beta_logpdf(x, a, b)).sum()) / total

    res = minimize(nll, x0=[np.log(init_a), np.log(init_b)], method="Nelder-Mead")
    a, b = np.exp(res.x)
    return max(a, 1e-2), max(b, 1e-2)


def _fit_bin(
    E: np.ndarray,
    init_pi: float,
    max_iter: int,
) -> tuple[float, float, float, float, float]:
    """EM for a 2-component Beta mixture on one bin. Returns (a0,b0,a1,b1,pi)."""
    # Initialization: split at the mean; noisy component has the higher mean.
    m = E.mean()
    comp = np.where(E >= m, 1, 0)
    a1, b1 = _beta_moments(E[comp == 1].mean() if (comp == 1).any() else m + 0.1, E[comp == 1].var() if (comp == 1).sum() > 1 else 0.01)
    a0, b0 = _beta_moments(E[comp == 0].mean() if (comp == 0).any() else m - 0.1, E[comp == 0].var() if (comp == 0).sum() > 1 else 0.01)
    pi = float(init_pi)

    for _ in range(max_iter):
        logp1 = _beta_logpdf(E, a1, b1)
        logp0 = _beta_logpdf(E, a0, b0)
        log_numer = np.log(pi + 1e-12) + logp1
        log_denom = np.logaddexp(log_numer, np.log(1 - pi + 1e-12) + logp0)
        r = np.exp(log_numer - log_denom)  # responsibility of noisy component
        r = np.clip(r, 1e-6, 1 - 1e-6)
        pi_new = float(r.mean())
        a1, b1 = _fit_weighted_beta(E, r, a1, b1)
        a0, b0 = _fit_weighted_beta(E, 1 - r, a0, b0)
        if abs(pi_new - pi) < 1e-5:
            pi = pi_new
            break
        pi = pi_new

    # Resolve label switching: the noisy component must have the higher mean.
    if (a1 / (a1 + b1)) < (a0 / (a0 + b0)):
        a1, b1, a0, b0 = a0, b0, a1, b1
        pi = 1.0 - pi
    return a0, b0, a1, b1, pi


def fit_beta_mixture(
    E: np.ndarray,
    init_pi: float = 0.4,
    max_iter: int = 1000,
    max_pi: float = 1.0,
) -> MixtureParams:
    """Fit a single global 2-component Beta mixture over evidence E.

    The global fit uses the *full* evidence separation, which is much stronger
    than within-quality-bin separation (per-bin fitting degrades because
    within-quality detectability is weak — the Gate A finding).
    """
    bin_id = np.zeros(len(E), dtype=int)
    return fit_beta_mixture_per_bin(E, bin_id, 1, init_pi=init_pi, max_iter=max_iter, max_pi=max_pi)


def fit_beta_mixture_per_bin(
    E: np.ndarray,
    bin_id: np.ndarray,
    n_bins: int,
    init_pi: float = 0.4,
    max_iter: int = 1000,
    max_pi: float = 1.0,
) -> MixtureParams:
    """Fit a per-bin 2-component Beta mixture over evidence E.

    ``max_pi`` caps the fitted noisy-component prior (prevents over-suppression
    when the evidence does not separate clean/noisy well, e.g. asymmetric noise).
    """
    E = _clip_evidence(E)
    bin_id = np.asarray(bin_id, dtype=int)
    a0s, b0s, a1s, b1s, pis = [], [], [], [], []
    for b in range(n_bins):
        xb = E[bin_id == b]
        if len(xb) < 10:
            a0s.append(2.0); b0s.append(6.0); a1s.append(6.0); b1s.append(2.0); pis.append(min(init_pi, max_pi))
            continue
        a0, b0, a1, b1, pi = _fit_bin(xb, init_pi, max_iter)
        pi = min(pi, max_pi)
        a0s.append(a0); b0s.append(b0); a1s.append(a1); b1s.append(b1); pis.append(pi)
    return MixtureParams(
        alpha0=np.asarray(a0s), beta0=np.asarray(b0s),
        alpha1=np.asarray(a1s), beta1=np.asarray(b1s),
        pi_q=np.asarray(pis),
    )
