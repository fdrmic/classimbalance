"""Evaluation metrics for the AML benchmark.

Primary metric
--------------
``pr_auc`` (area under the Precision-Recall curve) – the most informative
metric under extreme class imbalance, where ROC-AUC can be misleadingly
optimistic.

Secondary metrics
-----------------
``roc_auc``, ``precision``, ``recall``, ``f1``, ``f2``,
``weighted_accuracy``, ``tp``, ``fp``, ``tn``, ``fn``.

Weighted accuracy
-----------------
Standard accuracy treats every sample equally, so it is dominated by the
majority class under severe imbalance.  Weighted accuracy corrects for this
by up-weighting each positive (illicit) sample:

    weight_positive = n_negative / n_positive
    weight_negative = 1.0

These weights are passed to ``sklearn.metrics.accuracy_score`` via the
``sample_weight`` argument.  The result reflects per-class accuracy averaged
across both classes proportionally.

Threshold optimisation
----------------------
At the default threshold of 0.5 virtually no positive predictions are made
under extreme imbalance, so threshold-based metrics (precision, recall, F1)
are near zero even when PR-AUC is reasonable.  ``find_optimal_threshold``
searches over all unique score values and returns the threshold that
maximises F1 on a validation set.  This threshold is then applied to the
test set to produce operationally meaningful results.

Usage
-----
    metrics = compute_all_metrics(y_true, y_score)
    save_metrics(metrics, output_dir=Path("outputs/runs/run_001"), split="test")

    # Threshold optimisation
    threshold = find_optimal_threshold(y_val_true, y_val_score)
    metrics_opt = compute_all_metrics(y_test_true, y_test_score, threshold=threshold)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from aml_benchmark.utils.logging_utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Threshold optimisation
# ---------------------------------------------------------------------------

def find_optimal_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    criterion: str = "f1",
) -> float:
    """Find the decision threshold that maximises *criterion* on a labelled set.

    This function must be called on the **validation set only**.  The
    resulting threshold is then applied to the test set without any further
    tuning, preserving the integrity of the held-out test evaluation.

    Strategy
    --------
    The full Precision-Recall curve is computed via
    ``sklearn.metrics.precision_recall_curve``, which returns one
    (precision, recall) pair per unique predicted score.  For each pair the
    chosen criterion is evaluated and the threshold with the highest value is
    returned.

    Parameters
    ----------
    y_true:
        Ground-truth binary labels from the *validation* set.
    y_score:
        Predicted positive-class probabilities from the *validation* set.
    criterion:
        Optimisation target.  Supported values:

        ``"f1"``        – maximise F1 (harmonic mean of precision and recall).
                          Good default: balances detection rate and alert rate.
        ``"recall"``    – maximise recall subject to precision > 0.
                          Use when catching all illicit transactions is paramount.
        ``"precision"`` – maximise precision subject to recall > 0.
                          Use when minimising false alerts is paramount.

    Returns
    -------
    Optimal threshold in [0, 1].  If all scores are identical or the
    criterion is always zero, falls back to the median score.
    """
    from sklearn.metrics import precision_recall_curve

    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)

    precisions, recalls, thresholds = precision_recall_curve(y_true, y_score)
    # precision_recall_curve returns N+1 values for precisions/recalls but N
    # for thresholds; drop the last precision/recall (appended sentinel at 1.0)
    precisions = precisions[:-1]
    recalls = recalls[:-1]

    if len(thresholds) == 0:
        logger.warning("find_optimal_threshold: no threshold candidates found; using 0.5")
        return 0.5

    if criterion == "f1":
        denom = precisions + recalls
        f1_scores = np.where(denom > 0, 2 * precisions * recalls / denom, 0.0)
        best_idx = int(np.argmax(f1_scores))
        best_score = float(f1_scores[best_idx])
    elif criterion == "recall":
        # Recall as high as possible while precision > 0
        valid = precisions > 0
        if not valid.any():
            best_idx = int(np.argmax(recalls))
        else:
            best_idx = int(np.argmax(np.where(valid, recalls, -1.0)))
        best_score = float(recalls[best_idx])
    elif criterion == "precision":
        valid = recalls > 0
        if not valid.any():
            best_idx = int(np.argmax(precisions))
        else:
            best_idx = int(np.argmax(np.where(valid, precisions, -1.0)))
        best_score = float(precisions[best_idx])
    else:
        raise ValueError(f"Unknown criterion '{criterion}'. Use 'f1', 'recall', or 'precision'.")

    optimal_threshold = float(thresholds[best_idx])

    logger.info(
        f"Optimal threshold ({criterion}): {optimal_threshold:.6f} "
        f"| best_{criterion}={best_score:.6f} "
        f"| precision={precisions[best_idx]:.6f} "
        f"| recall={recalls[best_idx]:.6f}"
    )
    return optimal_threshold


# ---------------------------------------------------------------------------
# Core metric computation
# ---------------------------------------------------------------------------

def compute_all_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.5,
    split: str = "test",
) -> dict[str, float]:
    """Compute the full set of benchmark metrics.

    Parameters
    ----------
    y_true:
        Ground-truth binary labels (0 = normal, 1 = illicit).
    y_score:
        Predicted probability scores for the positive class.
    threshold:
        Decision threshold applied to *y_score* to produce binary
        predictions used by precision, recall, F1, confusion matrix, and
        weighted accuracy.  Defaults to 0.5.
    split:
        Label for logging (e.g. ``"val"`` or ``"test"``).

    Returns
    -------
    Ordered dict with keys (in order of importance):

    ``pr_auc``            – PRIMARY: area under Precision-Recall curve
    ``roc_auc``           – area under ROC curve
    ``precision``         – at *threshold*
    ``recall``            – at *threshold*
    ``f1``                – at *threshold*
    ``f2``                – at *threshold* (recall-weighted F-beta with beta=2)
    ``weighted_accuracy`` – class-imbalance-corrected accuracy (see module docstring)
    ``tp``                – true positives
    ``fp``                – false positives
    ``tn``                – true negatives
    ``fn``                – false negatives
    ``threshold``         – the threshold value used for binary metrics
    """
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    y_pred = (y_score >= threshold).astype(int)

    # --- Threshold-independent metrics ------------------------------------
    pr_auc = float(average_precision_score(y_true, y_score))
    roc_auc = float(roc_auc_score(y_true, y_score))

    # --- Threshold-dependent metrics -------------------------------------
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    f2 = float(fbeta_score(y_true, y_pred, beta=2.0, zero_division=0))

    # --- Confusion matrix counts -----------------------------------------
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    # --- Weighted accuracy -----------------------------------------------
    weighted_acc = _weighted_accuracy(y_true, y_pred, split=split)

    metrics: dict[str, float] = {
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "f2": f2,
        "weighted_accuracy": weighted_acc,
        "tp": float(tp),
        "fp": float(fp),
        "tn": float(tn),
        "fn": float(fn),
        "threshold": threshold,
    }

    _log_metrics(metrics, split=split)
    return metrics


# ---------------------------------------------------------------------------
# Weighted accuracy
# ---------------------------------------------------------------------------

def _weighted_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    split: str = "test",
) -> float:
    """Compute class-imbalance-corrected accuracy.

    Each positive sample receives weight ``n_negative / n_positive``; each
    negative sample receives weight ``1.0``.  This ensures that the two
    classes contribute equally regardless of how imbalanced they are.

    Parameters
    ----------
    y_true:
        Ground-truth binary labels (already validated as ndarray[int]).
    y_pred:
        Binary predictions at the chosen threshold.
    split:
        Label used for transparency logging.

    Returns
    -------
    Scalar weighted accuracy in [0, 1].
    """
    n_positive = int(y_true.sum())
    n_negative = int((1 - y_true).sum())

    if n_positive == 0:
        logger.warning(
            f"[{split}] weighted_accuracy: no positive samples found; "
            "returning standard accuracy."
        )
        return float(accuracy_score(y_true, y_pred))

    weight_positive = n_negative / n_positive
    weight_negative = 1.0

    logger.info(
        f"[{split}] weighted_accuracy class weights: "
        f"weight_positive={weight_positive:.4f} "
        f"(n_neg={n_negative:,} / n_pos={n_positive:,}), "
        f"weight_negative={weight_negative:.4f}"
    )

    sample_weights = np.where(y_true == 1, weight_positive, weight_negative)
    return float(accuracy_score(y_true, y_pred, sample_weight=sample_weights))


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_metrics(
    metrics: dict[str, float],
    output_dir: Path,
    split: str = "test",
) -> None:
    """Persist a metrics dict to both JSON and a single-row CSV.

    Files are named ``metrics_<split>.json`` and ``metrics_<split>.csv``
    so that validation and test results can coexist in the same directory.

    Parameters
    ----------
    metrics:
        Dict returned by :func:`compute_all_metrics`.
    output_dir:
        Directory where files are written (created if absent).
    split:
        ``"val"`` or ``"test"`` – used as filename suffix.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON
    json_path = output_dir / f"metrics_{split}.json"
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    logger.info(f"Metrics saved -> {json_path}")

    # CSV (single row, one column per metric)
    csv_path = output_dir / f"metrics_{split}.csv"
    pd.DataFrame([metrics]).to_csv(csv_path, index=False)
    logger.info(f"Metrics saved -> {csv_path}")


# ---------------------------------------------------------------------------
# Internal logging helper
# ---------------------------------------------------------------------------

def _log_metrics(metrics: dict[str, float], split: str) -> None:
    """Emit a formatted metrics table to the logger."""
    logger.info(f"[{split}] Evaluation metrics:")
    logger.info(f"  [PRIMARY]  pr_auc            : {metrics['pr_auc']:.6f}")
    logger.info(f"             roc_auc           : {metrics['roc_auc']:.6f}")
    logger.info(f"             precision         : {metrics['precision']:.6f}")
    logger.info(f"             recall            : {metrics['recall']:.6f}")
    logger.info(f"             f1                : {metrics['f1']:.6f}")
    logger.info(f"             f2                : {metrics['f2']:.6f}")
    logger.info(f"             weighted_accuracy : {metrics['weighted_accuracy']:.6f}")
    logger.info(
        f"             confusion: "
        f"TP={int(metrics['tp'])} FP={int(metrics['fp'])} "
        f"TN={int(metrics['tn'])} FN={int(metrics['fn'])}"
    )
