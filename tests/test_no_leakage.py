"""The iron-rule gate: clean labels and the noise mask must never be reachable
from training inputs or from non-evaluation code paths."""
from __future__ import annotations

from pathlib import Path

import torch

from quality_noise.data.datasets import build_views
from tests.fixtures import make_blob_dataset, toy_cfg

SRC = Path(__file__).resolve().parents[1] / "src" / "quality_noise"

# Modules allowed to touch evaluation-only fields (metrics/eval consumers).
EVAL_ALLOWLIST = {
    "metrics",
    "detectability",
    "data/datasets.py",  # constructs the views
    "data/registry.py",  # records counts only
    "scripts",  # orchestrators may read manifest for reporting
}


def test_training_view_has_no_clean_labels_or_mask():
    cfg = toy_cfg()
    bundle = build_views(cfg, make_blob_dataset())
    tv = bundle.train_view
    assert not hasattr(tv, "clean_labels")
    assert not hasattr(tv, "mask")
    # __getitem__ yields only (x, y_observed, sample_id).
    item = tv[0]
    assert len(item) == 3
    assert torch.is_tensor(item[0])
    assert isinstance(item[1], int)
    assert isinstance(item[2], int)


def test_training_inputs_never_contain_clean_label():
    cfg = toy_cfg()
    bundle = build_views(cfg, make_blob_dataset())
    tv = bundle.train_view
    for i in range(len(tv)):
        item = tv[i]
        assert len(item) == 3, "training input must be (x, y_observed, sample_id)"
    # No attribute anywhere on the training view that leaks clean labels.
    banned = [a for a in dir(tv) if "clean" in a.lower() or "mask" in a.lower()]
    assert banned == [], f"leaky attributes on TrainingView: {banned}"


def test_eval_view_provides_eval_only_fields():
    cfg = toy_cfg()
    bundle = build_views(cfg, make_blob_dataset())
    ev = bundle.eval_view
    assert hasattr(ev, "clean_labels")
    assert hasattr(ev, "mask")
    item = ev[0]
    assert len(item) == 5  # (x, y_observed, y_clean, mask, sample_id)
    assert item[2] in (0, 1)
    assert item[3] in (0, 1)


def test_views_share_sample_ordering():
    cfg = toy_cfg()
    bundle = build_views(cfg, make_blob_dataset())
    tv_ids = set(bundle.train_view.sample_ids.tolist())
    ev_ids = set(bundle.eval_view.sample_ids.tolist())
    assert tv_ids.issubset(ev_ids)


def _source_files():
    return [p for p in SRC.rglob("*.py") if p.name != "__init__.py"]


def test_no_eval_only_reference_outside_allowlist():
    """Grep guard: eval-only fields must not appear outside the allowlist."""
    patterns = ["clean_labels", ".mask", "noise_mask"]
    offenders = []
    for path in _source_files():
        rel = path.relative_to(SRC).as_posix()
        if any(rel.startswith(a) for a in EVAL_ALLOWLIST):
            continue
        if path.name == "noise.py":
            continue  # constructs the mask; consumption elsewhere is the risk
        text = path.read_text(encoding="utf-8")
        for pat in patterns:
            if pat in text:
                offenders.append((rel, pat))
    assert not offenders, f"eval-only field referenced outside allowlist: {offenders}"
