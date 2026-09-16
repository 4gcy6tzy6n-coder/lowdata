"""Baseline smoke tests: each strong baseline runs 1-2 epochs on the toy dataset
on CPU and produces a History with recorded loss/accuracy."""
from __future__ import annotations

import pytest

from quality_noise.baselines.coteaching import CoTeachingTrainer
from quality_noise.baselines.coteaching_plus import CoTeachingPlusTrainer
from quality_noise.baselines.dividemix import DivideMixTrainer
from quality_noise.baselines.elr import ELRTrainer
from quality_noise.baselines.jocor import JoCoRTrainer
from quality_noise.baselines.small_loss import SmallLossTrainer
from quality_noise.data.datasets import build_views
from quality_noise.utils import get_device
from tests.fixtures import make_blob_dataset, toy_cfg

ALL = [SmallLossTrainer, CoTeachingTrainer, CoTeachingPlusTrainer, JoCoRTrainer, ELRTrainer, DivideMixTrainer]


@pytest.mark.parametrize("cls", ALL, ids=lambda c: c.__name__)
def test_baseline_smoke(cls):
    cfg = toy_cfg(epochs=1, batch_size=32, noise_type="symmetric", rate=0.4)
    bundle = build_views(cfg, make_blob_dataset())
    trainer = cls(cfg, bundle.train_view, get_device("cpu"))
    history = trainer.run()
    assert len(history.train_loss) == 1
    assert len(history.train_acc) == 1
    assert 0.0 <= history.train_acc[0] <= 1.0
