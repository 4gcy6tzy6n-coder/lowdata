"""Integration test: the full toy pipeline from noisy data to soft weights.

Runs on the synthetic blob dataset on CPU and exercises data -> train -> traces
-> quality -> evidence -> detectability -> estimator -> weighted training. This
is the end-to-end guard that the pieces wire together and stay leakage-free.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quality_noise.data.datasets import build_views, load_bundle
from quality_noise.data.registry import build_registry_entry, read_registry_index, update_registry_index
from quality_noise.detectability.global_auc import global_auc
from quality_noise.detectability.matched_auc import match_quality_pairs, matched_auc
from quality_noise.evidence.loss import evidence_ema_loss, evidence_mean_loss
from quality_noise.evidence.signals import assemble_evidence
from quality_noise.metrics.downstream import accuracy, balanced_accuracy
from quality_noise.models.features import extract_embeddings
from quality_noise.quality.neighborhood import compute_knn_agreement
from quality_noise.quality.prototype_margin import compute_margin, select_high_confidence_anchors
from quality_noise.reliability.estimator import estimate_noise_posterior
from quality_noise.traces.dynamics import compute_all_dynamics
from quality_noise.traces.recorder import TraceRecorder
from quality_noise.trainers.standard import train_standard
from quality_noise.trainers.weighted import train_weighted
from quality_noise.utils import get_device
from tests.fixtures import SimpleRaw, make_blob_dataset, make_toy_model, toy_cfg


def _run_pipeline(tmp_path, epochs=2):
    cfg = toy_cfg(epochs=epochs, batch_size=32, noise_type="symmetric", rate=0.4, seed=0)
    bundle = build_views(cfg, make_blob_dataset())
    device = get_device("cpu")
    model = make_toy_model(bundle.num_classes)
    recorder = TraceRecorder(tmp_path, cfg_hash="pipe")
    train_standard(cfg, model, bundle.train_view, device, callbacks=[recorder])
    trace_path = recorder.finalize()
    traces = pd.read_parquet(trace_path)
    return cfg, bundle, model, traces, device


def test_pipeline_to_estimator(tmp_path):
    cfg, bundle, model, traces, device = _run_pipeline(tmp_path)
    n_train = len(bundle.train_view)
    y_obs = bundle.train_view.y_observed.astype(int)
    sample_ids = bundle.train_view.sample_ids.astype(int)

    # Dynamics + embeddings.
    losses = traces.pivot(index="sample_id", columns="epoch", values="loss").sort_index().values
    correct = traces.pivot(index="sample_id", columns="epoch", values="correct_to_observed").sort_index().values.astype(bool)
    dyn = compute_all_dynamics(losses, losses, losses, losses.astype(int) * 0, correct, sample_ids)
    assert dyn["forgetting_count"].shape == (n_train,)

    # Embeddings from the toy model (wrapped with the eval transform).
    from torch.utils.data import DataLoader

    from quality_noise.trainers.standard import _ImageWrapper, default_eval_transform

    wrapped = _ImageWrapper(bundle.train_view, default_eval_transform(cfg["dataset"]))
    loader = DataLoader(wrapped, batch_size=32, shuffle=False)
    embeddings = extract_embeddings(model, loader, device, view=bundle.train_view)
    assert embeddings.shape == (n_train, 32)

    # Quality Q1 + Q2.
    anchors = np.ones(n_train, dtype=bool)
    q1 = compute_margin(embeddings, y_obs, anchors, bundle.num_classes)
    q2 = compute_knn_agreement(embeddings, y_obs, k=10)
    assert q1.shape == (n_train,) and q2.shape == (n_train,)

    # Evidence + detectability.
    E, _ = assemble_evidence("ema_loss_forgetting_conflict", traces, embeddings, y_obs, conflict_k=10)
    mask = np.array([bool(bundle.eval_view.mask[bundle.eval_view.sample_ids.tolist().index(int(s))]) for s in sample_ids])
    auc = global_auc(E, mask)
    assert 0.0 <= auc <= 1.0
    pairs = match_quality_pairs(E, mask, q2, eps=0.2)
    assert matched_auc(pairs) >= 0.0

    # Estimator (no mask in signature) -> weights.
    res = estimate_noise_posterior(E, q2, {"mixture": {"n_bins": 2}, "reliability": {"method": "knn_agreement"}})
    assert res.weights.shape == (n_train,)
    assert res.weights.min() >= 0.0 and res.weights.max() <= 1.0

    # Weighted training runs.
    hist = train_weighted(cfg, make_toy_model(bundle.num_classes), bundle.train_view, res.weights, device)
    assert len(hist.train_loss) == cfg["model"]["epochs"]


def test_evidence_functions_agree(tmp_path):
    _cfg, _bundle, _model, traces, _device = _run_pipeline(tmp_path)
    ema = evidence_ema_loss(traces)
    mean = evidence_mean_loss(traces)
    assert ema.shape == mean.shape
    assert np.all(np.isfinite(ema))


def test_registry_roundtrip(tmp_path):
    cfg = toy_cfg()
    bundle = build_views(cfg, make_blob_dataset())
    entry = build_registry_entry(cfg, bundle.registry_entry)
    entry["config_hash"] = "abc123"
    index_path = tmp_path / "registry.json"
    update_registry_index(index_path, entry)
    index = read_registry_index(index_path)
    assert index["abc123"]["dataset"] == "blob"
    assert index["abc123"]["num_noisy"] == bundle.num_noisy


def test_metrics_and_accuracy():
    preds = np.array([0, 1, 2, 3])
    targets = np.array([0, 1, 3, 2])
    assert accuracy(preds, targets) == 0.5
    bal = balanced_accuracy(preds, targets, num_classes=4)
    assert 0.0 < bal <= 1.0


def test_load_bundle_and_evaluate(tmp_path, monkeypatch):
    from quality_noise import evaluate as evaluate_mod

    cfg = toy_cfg()
    raw = make_blob_dataset()
    bundle = load_bundle(cfg, raw=raw)
    assert bundle.num_noisy > 0

    # Patch the raw loader: returns (train, test) — the test set is a held-out
    # slice that must NOT be the training set (evaluate_test reads the 2nd).
    test_raw = SimpleRaw(data=raw.data[-40:].copy(), targets=raw.targets[-40:].copy())
    monkeypatch.setattr(evaluate_mod, "load_raw_dataset", lambda _d: (raw, test_raw))
    model = make_toy_model(bundle.num_classes)
    from quality_noise.trainers.standard import train_standard

    train_standard(cfg, model, bundle.train_view, get_device("cpu"), progress=False)
    out = evaluate_mod.evaluate_test(model, cfg, get_device("cpu"))
    assert 0.0 <= out["test_accuracy"] <= 1.0
