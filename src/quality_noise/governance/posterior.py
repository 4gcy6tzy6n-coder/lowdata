"""Posterior computation and representation-reliability shrinkage (Phase 6.5).

Given a fitted per-bin mixture, the noise posterior is
    eta_i = P(noisy | E_i, Q_i) = pi_q * p1(E) / (pi_q p1 + (1-pi_q) p0).

Representation reliability R shrinks the posterior toward the quality-region
prior pi_q in logit space:
    logit(eta_hat) = R * logit(eta) + (1 - R) * logit(pi_q).
Reliable representations trust the sample-specific evidence; unreliable ones
fall back to the prior — the mechanism that prevents weak/domain-mismatched
embeddings from producing extreme posteriors.
"""
from __future__ import annotations

import numpy as np

from .mixture import MixtureParams, _beta_logpdf, _clip_evidence


def beta_mixture_posterior(
    E: np.ndarray,
    params: MixtureParams,
    bin_id: np.ndarray,
) -> np.ndarray:
    """Noise posterior eta_i from the fitted mixture (no shrinkage yet)."""
    E = _clip_evidence(E)
    bin_id = np.asarray(bin_id, dtype=int)
    eta = np.zeros(len(E), dtype=np.float64)
    for b in range(params.n_bins):
        sel = bin_id == b
        if not sel.any():
            continue
        pi = params.pi_q[b]
        logp1 = _beta_logpdf(E[sel], params.alpha1[b], params.beta1[b])
        logp0 = _beta_logpdf(E[sel], params.alpha0[b], params.beta0[b])
        log_numer = np.log(pi + 1e-12) + logp1
        log_denom = np.logaddexp(log_numer, np.log(1 - pi + 1e-12) + logp0)
        eta[sel] = np.exp(log_numer - log_denom)
    return np.clip(eta, 1e-6, 1 - 1e-6).astype(np.float32)


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def shrink_posterior(
    eta: np.ndarray,
    pi_q: np.ndarray,
    reliability: np.ndarray,
    bin_id: np.ndarray | None = None,
) -> np.ndarray:
    """Shrink the noise posterior toward the per-bin prior by reliability R.

    If ``bin_id`` is None, ``pi_q`` is taken as a scalar broadcast to all
    samples; otherwise it is indexed per sample by its quality bin.
    """
    eta = np.clip(np.asarray(eta, dtype=np.float64), 1e-6, 1 - 1e-6)
    R = np.clip(np.asarray(reliability, dtype=np.float64), 0.0, 1.0)
    pi = np.asarray(pi_q, dtype=np.float64)
    if bin_id is not None:
        pi = pi[np.asarray(bin_id, dtype=int)]
    logit_eta = _logit(eta)
    logit_pi = _logit(pi)
    logit_hat = R * logit_eta + (1 - R) * logit_pi
    return _sigmoid(logit_hat).astype(np.float32)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))
