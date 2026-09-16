"""Experiment expansion tests: noise x seed sweep and noise-seed modes."""
from __future__ import annotations

from quality_noise.config import load_config
from quality_noise.experiment import expand_experiment, run_key


def test_gate_a_expands_to_12_runs():
    runs = expand_experiment(load_config("configs/experiments/gate_a.yaml"))
    assert len(runs) == 12  # 4 noise settings x 3 seeds
    keys = [run_key(r) for r in runs]
    assert len(set(keys)) == 12


def test_noise_seed_follows_run_seed_by_default():
    runs = expand_experiment(load_config("configs/experiments/gate_a.yaml"))
    assert all(r["noise"]["seed"] == r["seed"] for r in runs)


def test_fixed_noise_seed_mode():
    exp = load_config("configs/experiments/gate_a.yaml")
    exp["noise_seed"] = "fixed"
    runs = expand_experiment(exp)
    assert all(r["noise"]["seed"] == 0 for r in runs)


def test_run_key_unique_and_readable():
    runs = expand_experiment(load_config("configs/experiments/gate_a.yaml"))
    k = run_key(runs[0])
    assert k.startswith("cifar10_") and k.endswith("_seed0")
