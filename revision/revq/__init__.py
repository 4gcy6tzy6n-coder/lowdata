"""Revision experiment package (Major Revision, label-independent Q).

Modules
-------
dinov2        : frozen DINOv2 ViT-S/14 backbone (no external deps).
embeddings    : extract frozen DINOv2 (+ native ResNet) embeddings for all settings.
independent_q : class-agnostic density quality Q^DINO (no labels at any step).
matching      : matching variants (with/without replacement, class-conditioned,
                common-support trimmed) + caliper sweep + balance diagnostics.
bootstrap     : paired bootstrap CIs for AUC_g, AUC_QC and Delta_Q.
gates         : Gate evaluation helpers.
"""
