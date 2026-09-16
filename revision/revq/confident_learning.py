"""Confident Learning detector (Northcutt et al., JAIR 2021).

Two scores are produced, both oriented so that **higher = more suspicious**:

``cl_selfconf``
    ``1 - p_hat(y_obs | x)`` from **out-of-fold** predicted probabilities.
    This is Confident Learning's per-sample label-quality criterion (the paper's
    ``self_confidence``), and the quantity ``cleanlab.filter.find_label_issues``
    thresholds per class.

``cl_rate``
    ``P(y_obs != y_true | x)`` estimated by the full CL pipeline: out-of-fold
    probabilities -> per-class confidence thresholds -> confident joint ->
    ``estimate_joint`` / ``calibrate_confident_joint`` -> normalised rate matrix
    -> row-normalised to a per-sample suspiciousness.

Why out-of-fold matters
-----------------------
The reviewers' objection to the paper's other detectors is that a score built
from the *same* fitted model that produced the sample's own label can be
optimistic.  CL is defined on held-out probabilities, so we fit a regularised
multinomial logistic regression on the run's frozen penultimate features with
5-fold cross-validation and use only held-out predictions.  The features
themselves come from noisy-label training (that dependence is inherent to CL and
is reported explicitly), but no *label* is ever predicted from a model that saw
it.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold


def oof_probabilities(
    features: np.ndarray,
    y_observed: np.ndarray,
    num_classes: int,
    n_folds: int = 5,
    seed: int = 0,
    max_iter: int = 1000,
    C: float = 1.0,
) -> np.ndarray:
    """Out-of-fold predicted class probabilities (N, num_classes).

    ``features`` are L2-normalised first (cosine geometry, matching how the
    quality signals treat the same embeddings).  Each fold's model is fit on the
    other folds only, so every row is predicted by a model that never saw it.
    """
    x = np.asarray(features, dtype=np.float32)
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    y = np.asarray(y_observed).astype(int)
    n = len(y)

    # Ensure every class is present in every fold's *training* part.
    counts = np.bincount(y, minlength=num_classes)
    n_splits = int(min(n_folds, max(2, counts[counts > 0].min())))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    probs = np.full((n, num_classes), np.nan, dtype=np.float64)
    for tr, te in skf.split(x, y):
        # n_jobs is deliberately 1: callers parallelise over runs, and letting
        # every worker fan out internally oversubscribes the machine.
        clf = LogisticRegression(C=C, max_iter=max_iter, n_jobs=1)
        clf.fit(x[tr], y[tr])
        p = clf.predict_proba(x[te])
        # Map the fitted classes back onto the full class index space.
        full = np.zeros((len(te), num_classes), dtype=np.float64)
        full[:, clf.classes_] = p
        probs[te] = full

    # Any row left unpredicted (pathological folds) falls back to the class prior.
    bad = np.isnan(probs).any(axis=1)
    if bad.any():
        prior = counts / max(counts.sum(), 1)
        probs[bad] = prior
    return probs


def cl_scores(probs: np.ndarray, y_observed: np.ndarray) -> dict[str, np.ndarray]:
    """Confident-Learning suspiciousness scores from OOF probabilities."""
    p = np.asarray(probs, dtype=np.float64)
    y = np.asarray(y_observed).astype(int)
    n, k = p.shape
    p = p / np.maximum(p.sum(axis=1, keepdims=True), 1e-12)

    p_obs = p[np.arange(n), y]
    self_conf = 1.0 - p_obs                       # CL label-quality criterion

    # --- CL pipeline: per-class thresholds -> confident joint -> rate matrix ---
    # Threshold per class: average predicted probability of that class among the
    # samples whose observed label is that class (Northcutt et al., Alg. 1).
    t = np.empty(k)
    for c in range(k):
        sel = y == c
        t[c] = p_obs[sel].mean() if sel.any() else 1.0

    # Confident joint: with p(y_obs|x) >= t[y_obs] the whole mass goes to the
    # diagonal, otherwise it is spread over the classes clearing that sample's
    # threshold.  Built with flat bincount rather than np.add.at((y,y)) because
    # numpy 2.5.2 mis-validates paired fancy indexing on large arrays.
    conf = p_obs >= t[y]
    yc = y[conf]
    joint = np.bincount(yc * k + yc, minlength=k * k).reshape(k, k).astype(np.float64)
    for c in range(k):
        sel = (~conf) & (y == c)
        if not sel.any():
            continue
        joint[c, :] += (p[sel] >= t[c]).sum(axis=0)
    confident_joint = joint

    # Calibrate so the joint is consistent with the observed marginals.
    joint = _calibrate(confident_joint, p, y, k)
    row_sums = joint.sum(axis=1, keepdims=True)
    rate = np.divide(joint, np.maximum(row_sums, 1e-12))   # P(y_true | y_obs)
    # Diagonal lookup by flat index (avoids the same paired-indexing issue).
    cl_rate = 1.0 - rate.ravel()[y * k + y]                # 1 - P(y_true = y_obs | y_obs)
    return {"cl_selfconf": self_conf, "cl_rate": cl_rate, "thresholds": t,
            "confident_joint": joint}


def _calibrate(confident_joint: np.ndarray, p: np.ndarray, y: np.ndarray,
               k: int) -> np.ndarray:
    """Scale the confident joint so its row sums match observed label counts."""
    obs = np.bincount(y, minlength=k).astype(np.float64)
    row = confident_joint.sum(axis=1)
    scale = np.divide(obs, np.maximum(row, 1e-12))
    out = confident_joint * scale[:, None]
    # Samples out of scope of the confident joint get the prior in their own row.
    prior = obs / max(obs.sum(), 1)
    bad = row <= 0
    if bad.any():
        out[bad] = prior[None, :]
    return out


def compute_cl(
    features: np.ndarray,
    y_observed: np.ndarray,
    num_classes: int,
    n_folds: int = 5,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Convenience wrapper: OOF probabilities -> CL scores."""
    probs = oof_probabilities(features, y_observed, num_classes,
                              n_folds=n_folds, seed=seed)
    out = cl_scores(probs, y_observed)
    out["oof_probs"] = probs
    return out
