"""CIFAR-10N real-annotation-noise view construction tests.

Uses a synthetic npz (aggre_label / worse_label / clean_label) and a mock raw
CIFAR object to verify the iron rule holds for real annotation noise: the
training view exposes NO clean labels / mask, and the noisy labels come from
the annotation array (not a synthetic flip).
"""
from __future__ import annotations

import numpy as np
import pytest

from quality_noise.data.datasets import build_cifar10n_views
from tests.fixtures import SimpleRaw


def _make_npz(tmp_path, n=100, num_classes=10, seed=0):
    rng = np.random.default_rng(seed)
    clean = rng.integers(0, num_classes, size=n)
    noisy = clean.copy()
    # Flip ~40% to a random other class.
    n_flip = int(0.4 * n)
    idx = rng.choice(n, size=n_flip, replace=False)
    for i in idx:
        noisy[i] = (noisy[i] + rng.integers(1, num_classes)) % num_classes
    # Encode as one-hot (like the .npz) to exercise the argmax path.
    def one_hot(labels):
        m = np.zeros((n, num_classes), dtype=np.uint8)
        m[np.arange(n), labels] = 1
        return m

    npz = tmp_path / "CIFAR-10_human_anno.npz"
    assert not npz.exists()
    np.savez(npz, aggre_label=one_hot(noisy), worse_label=one_hot(noisy), clean_label=one_hot(clean))
    return noisy, clean


def test_cifar10n_views_iron_rule(tmp_path, monkeypatch):
    cfg = {
        "dataset": {
            "name": "cifar10n", "root": str(tmp_path), "download": False,
            "num_classes": 10, "val_frac": 0.1, "input_size": [3, 32, 32],
        },
        "noise": {"type": "real_annotation", "subset": "aggregate"},
        "seed": 0,
    }
    noisy, clean = _make_npz(tmp_path, n=100)
    # Mock raw CIFAR: images + clean labels per sample id.
    imgs = np.random.default_rng(1).integers(0, 255, size=(100, 3, 32, 32), dtype=np.uint8)
    mock_raw = SimpleRaw(data=imgs, targets=clean)

    import quality_noise.data.datasets as ds_mod

    monkeypatch.setattr(ds_mod, "load_raw_dataset", lambda _cfg: (mock_raw, None))

    bundle = build_cifar10n_views(cfg)
    # Training view: no clean labels, no mask.
    assert not hasattr(bundle.train_view, "clean_labels")
    assert not hasattr(bundle.train_view, "mask")
    # Noisy labels follow the annotation array, not a synthetic flip.
    assert bundle.eval_view.y_observed.tolist() == noisy.tolist()
    assert bundle.eval_view.clean_labels.tolist() == clean.tolist()
    # Mask = where annotation disagrees with clean.
    expected_mask = noisy != clean
    assert np.array_equal(bundle.eval_view.mask.astype(bool), expected_mask)
    # Registry records the realized rate (reporting only), never used in training.
    assert "realized_noise_rate" in bundle.registry_entry
    assert abs(bundle.registry_entry["realized_noise_rate"] - expected_mask.mean()) < 1e-9


def test_cifar10n_views_subset(tmp_path, monkeypatch):
    cfg = {
        "dataset": {
            "name": "cifar10n", "root": str(tmp_path), "download": False,
            "num_classes": 10, "val_frac": 0.1, "input_size": [3, 32, 32],
        },
        "noise": {"type": "real_annotation", "subset": "worse"},
        "seed": 0,
    }
    noisy, clean = _make_npz(tmp_path, seed=7)
    imgs = np.random.default_rng(1).integers(0, 255, size=(100, 3, 32, 32), dtype=np.uint8)
    mock_raw = SimpleRaw(data=imgs, targets=clean)

    import quality_noise.data.datasets as ds_mod

    monkeypatch.setattr(ds_mod, "load_raw_dataset", lambda _cfg: (mock_raw, None))
    bundle = build_cifar10n_views(cfg)
    assert bundle.registry_entry["subset"] == "worse"
    assert bundle.eval_view.mask.astype(bool).mean() == pytest.approx(0.4, abs=0.1)
