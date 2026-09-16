"""Detectability metric tests on planted AUC scenarios.

- mode="clean": noise and quality independent -> high global AND high matched AUC.
- mode="confounded": noisy/clean separable only at high quality -> high global
  AUC but low matched/within-quality AUC. The gap IS the phenomenon Phase 4
  is designed to measure.
"""
from __future__ import annotations

import numpy as np

from quality_noise.detectability.bootstrap import bootstrap_ci
from quality_noise.detectability.decomposition import confounding_gap, cross_quality_pair_matrix
from quality_noise.detectability.global_auc import global_auc
from quality_noise.detectability.matched_auc import match_quality_pairs, matched_auc
from quality_noise.detectability.stratified_auc import quality_stratified_auc
from tests.fixtures import make_score_mask_quality


def test_global_auc_clean_scenario_high():
    score, mask, _ = make_score_mask_quality(mode="clean")
    assert global_auc(score, mask) > 0.98


def test_global_auc_confounded_still_high():
    score, mask, _ = make_score_mask_quality(mode="confounded")
    assert global_auc(score, mask) > 0.7


def test_matched_auc_reveals_confounding():
    score, mask, q = make_score_mask_quality(mode="confounded")
    pairs = match_quality_pairs(score, mask, q, eps=0.05, max_clean_per_noisy=3)
    auc_matched = matched_auc(pairs)
    auc_global = global_auc(score, mask)
    # Global AUC is inflated by cross-quality comparisons; matching shrinks it.
    assert auc_matched < auc_global - 0.1
    assert auc_matched < 0.7  # within-quality separation is poor


def test_matched_auc_preserves_true_separability():
    score, mask, q = make_score_mask_quality(mode="clean")
    pairs = match_quality_pairs(score, mask, q, eps=0.05, max_clean_per_noisy=3)
    assert matched_auc(pairs) > 0.9


def test_stratified_auc_flat_when_confounded():
    score, mask, q = make_score_mask_quality(mode="confounded")
    strat = quality_stratified_auc(score, mask, q, n_bins=5)
    assert strat.loc[0, "auc"] < 0.65  # low-quality bin near random


def test_decomposition_matrix_and_gap():
    score, mask, q = make_score_mask_quality(mode="confounded")
    mat = cross_quality_pair_matrix(score, mask, q, n_bins=4)
    assert mat.shape == (4, 4)
    gap = confounding_gap(score, mask, q, n_bins=4)
    assert gap > 0.05  # off-diagonal (cross-quality) comparisons inflate AUC


def test_bootstrap_perfect_separation_tight_ci():
    score, mask, _ = make_score_mask_quality(mode="clean")
    lo, hi = bootstrap_ci(score, mask, n_resamples=200, seed=3)
    assert lo > 0.95


def test_bootstrap_random_covers_half():
    rng = np.random.default_rng(1)
    score = rng.random(400)
    mask = rng.random(400) > 0.5
    lo, hi = bootstrap_ci(score, mask, n_resamples=200, seed=4)
    assert lo <= 0.5 <= hi


def test_matched_reuse_cap_enforced():
    """No clean sample may appear in more than max_clean_per_noisy pairs."""
    n_noisy, n_clean, cap = 100, 5, 2
    score = np.random.default_rng(0).random(n_noisy + n_clean)
    mask = np.concatenate([np.ones(n_noisy, dtype=bool), np.zeros(n_clean, dtype=bool)])
    quality = np.concatenate([np.zeros(n_noisy), np.zeros(n_clean)])  # all same quality
    pairs = match_quality_pairs(score, mask, quality, eps=0.5, max_clean_per_noisy=cap)
    # With 5 clean samples each reusable at most `cap` times, total pairs cap out.
    assert len(pairs.s_noisy) <= n_clean * cap
