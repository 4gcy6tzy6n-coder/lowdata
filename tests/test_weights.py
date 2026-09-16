"""Governance + estimator tests: weighted CE, mixture EM, shrinkage, and the
no-leakage signature of the estimator."""
from __future__ import annotations

import inspect

import numpy as np
import pytest
import torch

from quality_noise.governance.mixture import fit_beta_mixture_per_bin
from quality_noise.governance.posterior import beta_mixture_posterior, shrink_posterior
from quality_noise.governance.weighting import weighted_ce_loss
from quality_noise.reliability.estimator import estimate_noise_posterior


def test_weighted_ce_exact_reduction():
    torch.manual_seed(0)
    logits = torch.randn(8, 5)
    targets = torch.randint(0, 5, (8,))
    weights = torch.linspace(0.1, 1.0, 8)
    loss = weighted_ce_loss(logits, targets, weights)
    import torch.nn.functional as F

    per_sample = F.cross_entropy(logits, targets, reduction="none")
    expected = (per_sample * weights).sum() / weights.sum()
    assert torch.allclose(loss, expected, atol=1e-6)


def test_weighted_ce_uniform_equals_ce():
    torch.manual_seed(1)
    logits = torch.randn(16, 10)
    targets = torch.randint(0, 10, (16,))
    import torch.nn.functional as F

    loss_w = weighted_ce_loss(logits, targets, torch.ones(16))
    loss_plain = F.cross_entropy(logits, targets)
    assert torch.allclose(loss_w, loss_plain, atol=1e-6)


def _two_beta_data(n=2000, seed=0, pi=0.4):
    rng = np.random.default_rng(seed)
    from scipy.stats import beta

    n1 = int(pi * n)
    clean = beta.rvs(2, 8, size=n - n1, random_state=rng)  # mean 0.2
    noisy = beta.rvs(8, 2, size=n1, random_state=rng)  # mean 0.8
    E = np.concatenate([clean, noisy])
    mask = np.concatenate([np.zeros(n - n1), np.ones(n1)]).astype(bool)
    order = rng.permutation(n)
    return E[order], mask[order]


def test_mixture_recovers_component_order():
    E, mask = _two_beta_data()
    bin_id = np.zeros(len(E), dtype=int)  # single bin
    params = fit_beta_mixture_per_bin(E, bin_id, 1, init_pi=0.4, max_iter=200)
    mean_noisy = params.alpha1[0] / (params.alpha1[0] + params.beta1[0])
    mean_clean = params.alpha0[0] / (params.alpha0[0] + params.beta0[0])
    assert mean_noisy > mean_clean + 0.2
    assert 0.15 < params.pi_q[0] < 0.7


def test_posterior_auroc_and_calibration():
    E, mask = _two_beta_data(n=3000)
    Q = np.random.default_rng(0).uniform(0, 1, size=len(E))
    res = estimate_noise_posterior(E, Q, {"mixture": {"n_bins": 1}, "noise": {"rate": 0.4}})
    from sklearn.metrics import roc_auc_score

    auroc = roc_auc_score(mask, res.eta_shrunk)
    assert auroc > 0.9
    # Calibrated weights: the top noise-rate fraction (by posterior) is
    # smoothly downweighted toward 0 while the clean majority stays ~1.
    rate = 0.4
    n_top = int(rate * len(E))
    top = np.argsort(res.eta_shrunk)[-n_top:]
    rest = np.setdiff1d(np.arange(len(E)), top)
    assert res.weights[top].mean() < 0.6  # linear 1->0 ramp over the top fraction
    assert res.weights[rest].mean() > 0.9  # clean majority kept at ~1
    assert res.weights.min() >= 0.0 and res.weights.max() <= 1.0


def test_shrink_posterior_extremes():
    eta = np.array([0.9, 0.3])
    pi = np.array([0.4, 0.4])
    # R=1 -> keep eta; R=0 -> collapse to pi.
    s1 = shrink_posterior(eta, pi, np.array([1.0, 1.0]))
    assert np.allclose(s1, eta, atol=1e-5)
    s0 = shrink_posterior(eta, pi, np.array([0.0, 0.0]))
    assert np.allclose(s0, pi, atol=1e-5)
    s_mid = shrink_posterior(eta, pi, np.array([0.5, 0.5]))
    assert np.all((s_mid > np.minimum(eta, pi)) & (s_mid < np.maximum(eta, pi)))


def test_estimator_signature_has_no_mask():
    sig = inspect.signature(estimate_noise_posterior)
    params = list(sig.parameters)
    assert "mask" not in params and "clean" not in params
    assert "E" in params and "Q" in params


def test_weight_sharpening_power():
    """A1: w -> w^p sharpens weights toward 0 without breaking ordering."""
    E, mask = _two_beta_data(n=3000)
    Q = np.random.default_rng(0).uniform(0, 1, size=len(E))
    base = dict({"mixture": {"n_bins": 1}, "noise": {"rate": 0.4}})
    res_flat = estimate_noise_posterior(E, Q, {**base, "reliability": {"weight_power": 1.0}})
    res_sharp = estimate_noise_posterior(E, Q, {**base, "reliability": {"weight_power": 3.0}})
    # Sharpening never increases a weight and preserves the same ordering.
    assert (res_sharp.weights <= res_flat.weights + 1e-6).all()
    assert (res_sharp.weights > 0).any()
    # The noisy tail gets crushed much harder.
    rate = 0.4
    n_top = int(rate * len(E))
    top = np.argsort(res_flat.eta_shrunk)[-n_top:]
    assert res_sharp.weights[top].mean() < res_flat.weights[top].mean() - 0.05


def test_hard_threshold_excludes_noisy_tail():
    """A2: eta > tau -> w = 0 exactly; the rest keeps soft weights."""
    E, mask = _two_beta_data(n=3000)
    Q = np.random.default_rng(0).uniform(0, 1, size=len(E))
    base = {"mixture": {"n_bins": 1}, "noise": {"rate": 0.4}}
    res = estimate_noise_posterior(E, Q, {**base, "reliability": {"hard_threshold": 0.5}})
    dropped = res.eta_shrunk > 0.5
    assert (res.weights[dropped] == 0.0).all()
    assert (res.weights[~dropped] > 0.0).all()
    # Auto threshold: drop exactly the top noise-rate fraction by posterior.
    res_auto = estimate_noise_posterior(E, Q, {**base, "reliability": {"hard_threshold": "auto"}})
    n_top = int(0.4 * len(E))
    assert abs((res_auto.weights == 0.0).sum() - n_top) <= max(2, int(0.01 * len(E)))
    # Dropped set is the top-posterior set.
    assert (res_auto.eta_shrunk[res_auto.weights == 0.0] >= res_auto.eta_shrunk[res_auto.weights > 0.0].max() - 1e-6).all()
    # hard_frac scales the excluded count.
    res_half = estimate_noise_posterior(E, Q, {**base, "reliability": {"hard_threshold": "auto", "hard_frac": 0.5}})
    n_half = int(0.4 * 0.5 * len(E))
    assert abs((res_half.weights == 0.0).sum() - n_half) <= max(2, int(0.01 * len(E)))


def test_rate_source_mixture_and_tail():
    """A3: adaptive noise-rate sources recover the true rate on separable data."""
    E, mask = _two_beta_data(n=4000, pi=0.3)
    Q = np.random.default_rng(0).uniform(0, 1, size=len(E))
    base = {"mixture": {"n_bins": 1}, "noise": {"rate": 0.4}}
    res_mix = estimate_noise_posterior(E, Q, {**base, "reliability": {"rate_source": "mixture"}})
    assert 0.2 < res_mix.rate_used < 0.45  # close to true 0.3, not pinned at 0.4
    res_tail = estimate_noise_posterior(E, Q, {**base, "reliability": {"rate_source": "tail"}})
    assert 0.2 < res_tail.rate_used < 0.45
    # Both still detect noise well.
    from sklearn.metrics import roc_auc_score

    assert roc_auc_score(mask, res_mix.eta_shrunk) > 0.9
    assert roc_auc_score(mask, res_tail.eta_shrunk) > 0.9


def test_rate_source_estimated_otsu():
    """Otsu-based rho_hat: robust unsupervised estimate on a bimodal evidence."""
    from sklearn.metrics import roc_auc_score
    from quality_noise.reliability.estimator import _otsu_noise_rate

    E, mask = _two_beta_data(n=4000, pi=0.3)
    rate_hat = _otsu_noise_rate(E)
    assert 0.15 < rate_hat < 0.6  # recover ~true 0.3 (Otsu can over-weight the noisy arm)

    Q = np.random.default_rng(0).uniform(0, 1, size=len(E))
    res = estimate_noise_posterior(
        E, Q, {"mixture": {"n_bins": 1}, "noise": {"rate": 0.4}, "reliability": {"rate_source": "estimated"}}
    )
    assert 0.15 < res.rate_used < 0.6
    # E and mask are aligned (from _two_beta_data); posterior ranks noise well.
    assert roc_auc_score(mask, res.eta_shrunk) > 0.9


def test_default_matches_v2_behavior():
    """Backward compatibility: no new reliability keys -> v2 calibrated weights."""
    E, mask = _two_beta_data(n=3000)
    Q = np.random.default_rng(0).uniform(0, 1, size=len(E))
    cfg = {"mixture": {"n_bins": 1}, "noise": {"rate": 0.4}}
    res = estimate_noise_posterior(E, Q, cfg)
    assert res.rate_used == pytest.approx(0.4)
    assert res.weights.min() >= 0.0 and res.weights.max() <= 1.0
    # Calibrated ramp: top fraction smoothly downweighted; the very last rank
    # reaches exactly 0 (the v2 behavior), the rest stays positive.
    n_top = int(0.4 * len(E))
    top = np.argsort(res.eta_shrunk)[-n_top:]
    assert res.weights[top].mean() < 0.6
    assert (res.weights[top] >= 0.0).all()
    assert (res.weights[top] > 0.0).sum() >= n_top - 1
