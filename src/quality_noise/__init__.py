"""quality_noise — quality-conditioned label-noise detectability & reliability-aware soft governance.

Two-layer research project:
- Layer 1: Quality-Conditioned Detectability — when is label noise truly detectable.
- Layer 2: Reliability-Aware Soft Governance / reweighting for risk control.

Iron rule: clean labels and the noise mask are evaluation-only and must never
enter training, threshold selection, prototype construction, or model selection.
"""

__version__ = "0.1.0"
