"""Sample-level trace recorder.

Records per-sample × per-epoch signals during training so that downstream
quality estimation, evidence, and detectability analysis can reason about
individual samples. Composes with any trainer via callback hooks.

Frozen schema (add-only policy): adding a column is fine; renaming/removing a
column requires a schema-version bump so old parquet traces stay valid.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

TRACE_COLUMNS: tuple[str, ...] = (
    "sample_id",
    "epoch",
    "loss",
    "confidence",
    "margin",
    "entropy",
    "prediction",
    "correct_to_observed",
)
TRACE_SCHEMA_VERSION = 1

# Dynamics columns (per-sample aggregates) — computed in dynamics.py.
DYNAMICS_COLUMNS: tuple[str, ...] = (
    "sample_id",
    "ema_loss",
    "loss_variance",
    "prediction_consistency",
    "forgetting_count",
    "confidence_variance",
    "margin_trend",
)


def sample_signals(
    logits: torch.Tensor, observed_labels: torch.Tensor
) -> dict[str, np.ndarray]:
    """Compute per-sample signals from a batch of logits.

    Returns arrays for loss (CE, reduction none), softmax confidence, prob
    margin (top1 - top2), entropy, argmax prediction, and correctness to the
    observed label. All signals are functions of the *observed* label only.
    """
    logits = logits.detach()
    labels = observed_labels.detach()
    loss = F.cross_entropy(logits, labels, reduction="none")
    probs = F.softmax(logits, dim=-1)
    top2 = torch.topk(probs, k=2, dim=-1).values
    confidence = probs.gather(1, labels[:, None]).squeeze(1)
    margin = (top2[:, 0] - top2[:, 1]).clamp(min=0.0)
    entropy = -(probs * (probs + 1e-12).log()).sum(dim=-1)
    prediction = probs.argmax(dim=-1)
    correct = (prediction == labels).float()
    return {
        "loss": loss.cpu().numpy().astype(np.float32),
        "confidence": confidence.cpu().numpy().astype(np.float32),
        "margin": margin.cpu().numpy().astype(np.float32),
        "entropy": entropy.cpu().numpy().astype(np.float32),
        "prediction": prediction.cpu().numpy().astype(np.int16),
        "correct_to_observed": correct.cpu().numpy().astype(bool),
    }


class TraceRecorder:
    """Callback-based recorder. Attach to any trainer that calls the hooks."""

    def __init__(
        self,
        save_dir: Path,
        cfg_hash: str,
        ema_decay: float = 0.9,
        collect_embeddings: bool = False,
    ) -> None:
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.cfg_hash = cfg_hash
        self.ema_decay = ema_decay
        self.collect_embeddings = collect_embeddings
        self._rows: list[dict[str, np.ndarray]] = []
        self._current_epoch: list[dict[str, np.ndarray]] = []
        self._finalized = False

    # -- hooks called by trainers -------------------------------------------
    def on_train_batch(
        self,
        logits: torch.Tensor,
        observed_labels: torch.Tensor,
        sample_ids: torch.Tensor,
        epoch: int,
    ) -> None:
        signals = sample_signals(logits, observed_labels)
        batch = {
            "sample_id": sample_ids.detach().cpu().numpy().astype(np.int64),
            "epoch": np.full(len(sample_ids), epoch, dtype=np.int32),
            **signals,
        }
        self._current_epoch.append(batch)

    def on_epoch_end(self) -> None:
        """Flush the completed epoch into the row store."""
        if not self._current_epoch:
            return
        epoch_rows = {k: np.concatenate([b[k] for b in self._current_epoch], axis=0) for k in self._current_epoch[0]}
        self._rows.append(epoch_rows)
        self._current_epoch = []

    # -- persistence ----------------------------------------------------------
    def finalize(self) -> Path:
        """Concatenate all epochs, write the trace parquet, and return its path.

        Idempotent: calling twice returns the same path without rewriting.
        """
        if self._finalized:
            return self._trace_path()
        if self._current_epoch:
            self.on_epoch_end()
        if not self._rows:
            raise RuntimeError("TraceRecorder.finalize() with no recorded data")
        data = {k: np.concatenate([r[k] for r in self._rows], axis=0) for k in self._rows[0]}
        df = pd.DataFrame(data)
        df["sample_id"] = df["sample_id"].astype("int64")
        df["epoch"] = df["epoch"].astype("int32")
        path = self._trace_path()
        table = _to_arrow_table(df)
        _write_arrow(table, path, self.cfg_hash)
        self._finalized = True
        return path

    def _trace_path(self) -> Path:
        return self.save_dir / f"traces_{self.cfg_hash}.parquet"


def _to_arrow_table(df: pd.DataFrame):
    import pyarrow as pa

    schema = pa.schema(
        [
            pa.field("sample_id", pa.int64()),
            pa.field("epoch", pa.int32()),
            pa.field("loss", pa.float32()),
            pa.field("confidence", pa.float32()),
            pa.field("margin", pa.float32()),
            pa.field("entropy", pa.float32()),
            pa.field("prediction", pa.int16()),
            pa.field("correct_to_observed", pa.bool_()),
        ]
    )
    return pa.Table.from_pandas(df, schema=schema, preserve_index=False)


def _write_arrow(table, path: Path, cfg_hash: str) -> None:
    import pyarrow.parquet as pq

    table = table.replace_schema_metadata(
        {
            "schema_version": str(TRACE_SCHEMA_VERSION),
            "config_hash": cfg_hash,
        }
    )
    pq.write_table(table, path)
