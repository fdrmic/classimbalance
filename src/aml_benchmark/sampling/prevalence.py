"""Prevalence control utilities for the AML benchmark.

All functions operate on the TRAINING split only.  Validation and test
splits must never be passed to these routines.

Key concepts
------------
Natural prevalence
    The positive-class ratio in the raw training split (~0.046% for the
    LI-Small dataset).  This is lower than all three target levels
    (0.1%, 0.5%, 1.0%), so reaching a target always requires either
    undersampling the majority class or oversampling the minority class.

Target prevalence
    The desired fraction of positive samples in the post-processed
    training split.  Controlled per benchmark condition via
    ``configs/benchmark.yaml``.

Sampling-strategy ratio
    The imbalanced-learn ``sampling_strategy`` parameter accepts the
    ratio ``n_minority / n_majority`` after resampling, NOT the fraction
    ``n_minority / n_total``.  ``prevalence_to_ratio`` converts between
    the two representations.
"""
from __future__ import annotations

import numpy as np

from aml_benchmark.utils.logging_utils import get_logger

logger = get_logger(__name__)


def prevalence_to_ratio(prevalence: float) -> float:
    """Convert a target positive fraction to a minority/majority ratio.

    ``imbalanced-learn`` sampling methods accept ``sampling_strategy``
    as ``n_minority / n_majority``, not ``n_minority / n_total``.

    Parameters
    ----------
    prevalence:
        Desired positive-class fraction, e.g. ``0.01`` for 1%.

    Returns
    -------
    Minority-to-majority ratio ``p / (1 - p)``.
    """
    if not (0.0 < prevalence < 1.0):
        raise ValueError(f"prevalence must be in (0, 1); got {prevalence}")
    return prevalence / (1.0 - prevalence)


def compute_achieved_prevalence(y: np.ndarray) -> float:
    """Return the positive-class fraction of a label array."""
    n = len(y)
    return float(y.sum()) / n if n > 0 else 0.0


def compute_class_weights(target_prevalence: float) -> dict[int, float]:
    """Derive class weights from a target positive-class prevalence.

    Used by the class-weighting strategy to set model-internal costs
    without resampling the training data.

    The negative class always receives weight 1.0.  The positive class
    receives a weight proportional to the inverse prevalence ratio,
    simulating the desired class balance.

    Parameters
    ----------
    target_prevalence:
        Desired positive-class fraction (e.g. ``0.01`` for 1%).

    Returns
    -------
    Dict ``{0: 1.0, 1: weight_positive}`` ready for sklearn's
    ``class_weight`` parameter or XGBoost's ``scale_pos_weight``.
    """
    weight_positive = (1.0 - target_prevalence) / target_prevalence
    weights = {0: 1.0, 1: weight_positive}
    logger.info(
        f"Class weights from target_prevalence={target_prevalence:.4%}: "
        f"w0={weights[0]:.4f}, w1={weights[1]:.4f}"
    )
    return weights


def log_class_counts(y: np.ndarray, label: str = "") -> None:
    """Log positive/negative counts and prevalence for a label array."""
    n_pos = int(y.sum())
    n_neg = int((1 - y).sum())
    n_total = len(y)
    prevalence = n_pos / n_total if n_total else 0.0
    tag = f"[{label}] " if label else ""
    logger.info(
        f"{tag}n_total={n_total:,}  n_pos={n_pos:,}  "
        f"n_neg={n_neg:,}  prevalence={prevalence:.6%}"
    )


def compute_class_weights_from_data(y: np.ndarray) -> dict[int, float]:
    """Derive class weights from the actual class distribution in the data.

    Unlike ``compute_class_weights`` which uses a target prevalence,
    this function uses the real minority/majority ratio in the training
    split. For XGBoost this translates to scale_pos_weight = n_neg / n_pos.

    Parameters
    ----------
    y:
        Binary label array for the training split.

    Returns
    -------
    Dict ``{0: 1.0, 1: n_negative / n_positive}``.
    """
    n_pos = int(y.sum())
    n_neg = int((1 - y).sum())
    if n_pos == 0:
        raise ValueError("No positive samples found in training data.")
    weight_positive = n_neg / n_pos
    weights = {0: 1.0, 1: weight_positive}
    logger.info(
        f"Class weights from data ratio: "
        f"n_neg={n_neg:,} / n_pos={n_pos:,} = w1={weight_positive:.2f}"
    )
    return weights
