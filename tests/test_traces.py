"""Trace recorder + dynamics tests on a tiny toy training run."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quality_noise.data.datasets import build_views
from quality_noise.traces.dynamics import (
    compute_forgetting_count,
    compute_prediction_consistency,
)
from quality_noise.traces.recorder import TRACE_COLUMNS, TraceRecorder
from quality_noise.trainers.standard import train_standard
from quality_noise.utils import get_device
from tests.fixtures import make_blob_dataset, make_toy_model, toy_cfg


def _run_toy(cfg: dict, tmp_path) -> tuple[TraceRecorder, int]:
    bundle = build_views(cfg, make_blob_dataset())
    model = make_toy_model(bundle.num_classes)
    device = get_device("cpu")
    recorder = TraceRecorder(tmp_path, cfg_hash="testhash")
    train_standard(cfg, model, bundle.train_view, device, callbacks=[recorder])
    return recorder, len(bundle.train_view)


def test_recorder_writes_parquet_with_correct_schema(tmp_path):
    cfg = toy_cfg(epochs=2, batch_size=16)
    recorder, n_train = _run_toy(cfg, tmp_path)
    path = recorder.finalize()
    assert path.exists()
    df = pd.read_parquet(path)
    assert list(df.columns) == list(TRACE_COLUMNS)
    assert df["sample_id"].nunique() == n_train
    assert df["epoch"].max() == 1  # 2 epochs -> epoch 0,1
    assert len(df) == n_train * 2


def test_recorder_finalize_idempotent(tmp_path):
    cfg = toy_cfg(epochs=1)
    recorder, _ = _run_toy(cfg, tmp_path)
    p1 = recorder.finalize()
    p2 = recorder.finalize()
    assert p1 == p2


def test_dynamics_on_crafted_arrays():
    correct = np.array([[1, 1, 0, 0, 1]], dtype=bool)  # two forgetting events (t2, t3)
    forgets = compute_forgetting_count(correct)
    assert int(forgets[0]) == 2
    preds = np.array([[0, 0, 1, 0, 0]])
    assert compute_prediction_consistency(preds)[0] == pytest.approx(0.8)


def test_trace_values_in_range(tmp_path):
    cfg = toy_cfg(epochs=1)
    recorder, _ = _run_toy(cfg, tmp_path)
    path = recorder.finalize()
    df = pd.read_parquet(path)
    assert (df["confidence"] >= 0).all() and (df["confidence"] <= 1).all()
    assert (df["entropy"] >= 0).all()
    assert df["correct_to_observed"].isin([True, False]).all()
