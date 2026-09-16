"""Noise engine tests: rate accuracy, determinism, transition correctness,
immutability."""
from __future__ import annotations

import numpy as np
import pytest

from quality_noise.data import noise as noise_mod
from quality_noise.data.noise import CIFAR10_ASYMMETRIC_MAP, apply_noise
from tests.fixtures import make_blob_dataset


def _y(n: int = 1000, num_classes: int = 10) -> np.ndarray:
    return np.arange(n) % num_classes


def test_symmetric_rate_and_determinism():
    y = _y()
    rng = np.random.default_rng(0)
    y1, m1 = noise_mod.SymmetricNoise.flip(y, 10, 0.4, rng)
    rng2 = np.random.default_rng(0)
    y2, m2 = noise_mod.SymmetricNoise.flip(y, 10, 0.4, rng2)
    assert np.array_equal(y1, y2)
    assert np.array_equal(m1, m2)
    # Rate within tolerance; every flip goes to a different class.
    assert abs(m1.mean() - 0.4) < 0.05
    flips = np.where(m1)[0]
    assert np.all(y1[flips] != y[flips])


def _apply(**kw):
    base = {"type": "symmetric", "rate": 0.4, "seed": 0, "num_classes": 10}
    base.update(kw)
    return apply_noise(base, _y())


def test_symmetric_different_seed_different_mask():
    m1 = _apply(seed=0)["mask"]
    m2 = _apply(seed=1)["mask"]
    assert not np.array_equal(m1, m2)


def test_asymmetric_respects_pair_map():
    cfg = {"type": "asymmetric", "rate": 0.4, "seed": 0, "num_classes": 10, "pair_map": {2: 7, 4: 9}}
    res = apply_noise(cfg, _y())
    y_noisy, mask = res["y_noisy"], res["mask"]
    y = _y()
    assert np.all(mask[y == 2] == (y_noisy[y == 2] == 7))
    assert np.all(mask[y == 4] == (y_noisy[y == 4] == 9))
    # Classes not in the map are never flipped.
    assert np.all(y_noisy[y == 0] == 0)
    assert np.all(mask[y == 0] == False)  # noqa: E712


def test_asymmetric_rate_within_tolerance():
    res = _apply(type="asymmetric")
    y = _y()
    # Only the mapped source classes receive flips; rate is per-class there.
    mapped = np.isin(y, list(CIFAR10_ASYMMETRIC_MAP.keys()))
    assert abs(res["mask"][mapped].mean() - 0.4) < 0.05


def test_instance_dependent_rate_and_no_mask_leak():
    raw = make_blob_dataset(num_classes=4, per_class=100)
    y = np.asarray(raw.targets)
    feats = raw.data.reshape(raw.data.shape[0], -1).astype(np.float32)
    feats = (feats - feats.mean(0)) / (feats.std(0) + 1e-8)
    cfg = {"type": "instance_dependent", "rate": 0.3, "seed": 0, "num_classes": 4, "max_rate": 0.9}
    res = apply_noise(cfg, y, features=feats)
    y_noisy, mask = res["y_noisy"], res["mask"]
    assert abs(mask.mean() - 0.3) < 0.06
    # Flips never target the observed class.
    assert np.all(y_noisy[mask] != y[mask])


def test_immutability_of_clean_labels():
    y = _y()
    y0 = y.copy()
    apply_noise({"type": "symmetric", "rate": 0.4, "seed": 0, "num_classes": 10}, y)
    assert np.array_equal(y, y0)


def test_unknown_noise_type_raises():
    with pytest.raises(ValueError):
        apply_noise({"type": "nope", "rate": 0.3, "seed": 0, "num_classes": 10}, _y())


def test_instance_dependent_requires_features():
    with pytest.raises(ValueError):
        apply_noise({"type": "instance_dependent", "rate": 0.3, "seed": 0, "num_classes": 4}, _y())
