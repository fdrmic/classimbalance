"""Imbalance-mitigation strategies for the AML benchmark.

All strategies are applied to the TRAINING split only.
Validation and test data must never be passed to these functions.

Strategy semantics
------------------
``baseline``
    Training data is used unchanged at its natural prevalence (~0.046%).
    The ``target_prevalence`` parameter is recorded but not enforced.
    This serves as the unmodified reference condition.

``random_undersampling``
    The majority class (normal transactions) is randomly undersampled
    until the positive fraction reaches ``target_prevalence``.
    All original minority samples are retained.

``smote``
    Synthetic Minority Over-sampling TEchnique.  Synthetic positive
    samples are generated via linear interpolation between real minority
    neighbours until ``target_prevalence`` is reached.  The majority
    class is not modified.  Requires ``imbalanced-learn``.

``adasyn``
    Adaptive Synthetic Sampling.  Similar to SMOTE but generates more
    synthetic samples in regions where the minority class is harder to
    learn (i.e., close to the decision boundary).  Requires
    ``imbalanced-learn``.

``class_weighting``
    No resampling is performed.  The full training data is used
    unchanged.  Class weights derived from ``target_prevalence`` are
    returned as a dict and must be passed to the model constructor.

Return contract
---------------
Every strategy returns a :class:`SamplingResult` with:
  - ``X``, ``y``            – (resampled) feature matrix and labels
  - ``class_weight``        – ``{0: w0, 1: w1}`` or ``None``
  - ``achieved_prevalence`` – actual positive fraction after sampling
  - ``n_positive``          – positive sample count after sampling
  - ``n_negative``          – negative sample count after sampling
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from aml_benchmark.sampling.prevalence import (
    compute_achieved_prevalence,
    compute_class_weights,
    log_class_counts,
    prevalence_to_ratio,
)
from aml_benchmark.utils.logging_utils import get_logger

logger = get_logger(__name__)

SUPPORTED_STRATEGIES = (
    "baseline",
    "random_undersampling",
    "smote",
    "smote_class_weighting",
    "adasyn",
    "class_weighting",
    "true_cost_weighting",
)


@dataclass
class SamplingResult:
    """Container for the output of :func:`apply_strategy`."""

    X: np.ndarray
    y: np.ndarray
    class_weight: dict[int, float] | None
    achieved_prevalence: float
    n_positive: int
    n_negative: int
    strategy: str
    target_prevalence: float
    # Synthetic sample count (SMOTE / ADASYN only)
    n_synthetic: int = field(default=0)


# ---------------------------------------------------------------------------
# Public dispatch API
# ---------------------------------------------------------------------------

def apply_strategy(
    X_train: np.ndarray,
    y_train: np.ndarray,
    strategy: str,
    target_prevalence: float,
    random_state: int = 42,
) -> SamplingResult:
    """Apply an imbalance-mitigation strategy to the training split.

    Parameters
    ----------
    X_train:
        Feature matrix for the training split (numpy float array).
        Must be the output of ``FeaturePipeline.fit_transform``.
    y_train:
        Binary label array for the training split (0/1 integers).
    strategy:
        One of :data:`SUPPORTED_STRATEGIES`.
    target_prevalence:
        Desired positive-class fraction (e.g. ``0.01`` for 1.0%).
        Interpretation is strategy-specific; see module docstring.
    random_state:
        Integer seed for reproducibility of random operations.

    Returns
    -------
    :class:`SamplingResult` with resampled arrays, class weights (if
    applicable), and diagnostic counts.
    """
    strategy = strategy.lower().strip()
    if strategy not in SUPPORTED_STRATEGIES:
        raise ValueError(
            f"Unknown strategy '{strategy}'. "
            f"Supported: {SUPPORTED_STRATEGIES}"
        )

    logger.info("-" * 62)
    logger.info(
        f"Sampling strategy: {strategy} | "
        f"target_prevalence={target_prevalence:.4%} | "
        f"random_state={random_state}"
    )
    log_class_counts(y_train, label="before")

    if strategy == "baseline":
        result = _baseline(X_train, y_train, target_prevalence)
    elif strategy == "random_undersampling":
        result = _random_undersampling(X_train, y_train, target_prevalence, random_state)
    elif strategy == "smote":
        result = _smote(X_train, y_train, target_prevalence, random_state)
    elif strategy == "smote_class_weighting":
        result = _smote_class_weighting(X_train, y_train, target_prevalence, random_state)
    elif strategy == "adasyn":
        result = _adasyn(X_train, y_train, target_prevalence, random_state)
    elif strategy == "class_weighting":
        result = _class_weighting(X_train, y_train, target_prevalence)
    else:  # true_cost_weighting
        result = _true_cost_weighting(X_train, y_train, target_prevalence)

    log_class_counts(result.y, label="after")
    logger.info(
        f"Achieved prevalence: {result.achieved_prevalence:.6%} "
        f"(target: {target_prevalence:.4%})"
    )
    if result.n_synthetic:
        logger.info(f"Synthetic samples generated: {result.n_synthetic:,}")
    logger.info("-" * 62)

    return result


# ---------------------------------------------------------------------------
# Strategy implementations
# ---------------------------------------------------------------------------

def _baseline(
    X: np.ndarray,
    y: np.ndarray,
    target_prevalence: float,
) -> SamplingResult:
    """Return training data completely unchanged."""
    achieved = compute_achieved_prevalence(y)
    n_pos = int(y.sum())
    n_neg = int((1 - y).sum())
    logger.info(
        f"Baseline: no resampling applied. "
        f"Natural prevalence={achieved:.6%} "
        f"(target {target_prevalence:.4%} is recorded but not enforced)."
    )
    return SamplingResult(
        X=X, y=y,
        class_weight=None,
        achieved_prevalence=achieved,
        n_positive=n_pos,
        n_negative=n_neg,
        strategy="baseline",
        target_prevalence=target_prevalence,
    )


def _random_undersampling(
    X: np.ndarray,
    y: np.ndarray,
    target_prevalence: float,
    random_state: int,
) -> SamplingResult:
    """Undersample the majority class to reach ``target_prevalence``."""
    from imblearn.under_sampling import RandomUnderSampler

    natural = compute_achieved_prevalence(y)
    if target_prevalence <= natural:
        logger.warning(
            f"RUS: target_prevalence={target_prevalence:.4%} <= "
            f"natural={natural:.6%}. No undersampling needed; "
            "returning training data unchanged."
        )
        return _baseline(X, y, target_prevalence)

    ratio = prevalence_to_ratio(target_prevalence)
    rus = RandomUnderSampler(sampling_strategy=ratio, random_state=random_state)
    X_res, y_res = rus.fit_resample(X, y)

    achieved = compute_achieved_prevalence(y_res)
    n_pos = int(y_res.sum())
    n_neg = int((1 - y_res).sum())

    return SamplingResult(
        X=X_res, y=y_res,
        class_weight=None,
        achieved_prevalence=achieved,
        n_positive=n_pos,
        n_negative=n_neg,
        strategy="random_undersampling",
        target_prevalence=target_prevalence,
    )


def _smote(
    X: np.ndarray,
    y: np.ndarray,
    target_prevalence: float,
    random_state: int,
) -> SamplingResult:
    """Oversample the minority class with SMOTE to reach ``target_prevalence``."""
    from imblearn.over_sampling import SMOTE

    natural = compute_achieved_prevalence(y)
    n_pos_before = int(y.sum())

    if target_prevalence <= natural:
        logger.warning(
            f"SMOTE: target_prevalence={target_prevalence:.4%} <= "
            f"natural={natural:.6%}. Natural prevalence already meets target; "
            "returning data unchanged."
        )
        return _baseline(X, y, target_prevalence)

    # k_neighbors must be < n_minority; default 5 is safe with 2231+ positives
    k_neighbors = min(5, n_pos_before - 1)
    if k_neighbors < 1:
        raise ValueError(
            f"SMOTE requires at least 2 positive samples; got {n_pos_before}."
        )

    ratio = prevalence_to_ratio(target_prevalence)
    smote = SMOTE(
        sampling_strategy=ratio,
        k_neighbors=k_neighbors,
        random_state=random_state,
    )
    logger.info(f"SMOTE fit_resample starting (ratio={ratio:.4f}, k={k_neighbors}) ...")
    X_res, y_res = smote.fit_resample(X, y)
    logger.info("SMOTE fit_resample done.")

    n_pos_after = int(y_res.sum())
    n_synthetic = n_pos_after - n_pos_before
    achieved = compute_achieved_prevalence(y_res)

    return SamplingResult(
        X=X_res, y=y_res,
        class_weight=None,
        achieved_prevalence=achieved,
        n_positive=n_pos_after,
        n_negative=int((1 - y_res).sum()),
        strategy="smote",
        target_prevalence=target_prevalence,
        n_synthetic=n_synthetic,
    )


def _smote_class_weighting(
    X: np.ndarray,
    y: np.ndarray,
    target_prevalence: float,
    random_state: int,
) -> SamplingResult:
    """SMOTE oversampling plus cost-sensitive class weights from original labels."""
    from imblearn.over_sampling import SMOTE

    from aml_benchmark.sampling.prevalence import compute_class_weights_from_data

    logger.info(
        "Strategy: smote_class_weighting | SMOTE oversampling + true cost-sensitive weighting"
    )

    y_original = y

    natural = compute_achieved_prevalence(y)
    n_pos_before = int(y.sum())

    if target_prevalence <= natural:
        logger.warning(
            f"SMOTE: target_prevalence={target_prevalence:.4%} <= "
            f"natural={natural:.6%}. Natural prevalence already meets target; "
            "returning data unchanged."
        )
        class_weight = compute_class_weights_from_data(y_original)
        achieved = compute_achieved_prevalence(y)
        return SamplingResult(
            X=X, y=y,
            class_weight=class_weight,
            achieved_prevalence=achieved,
            n_positive=int(y.sum()),
            n_negative=int((1 - y).sum()),
            strategy="smote_class_weighting",
            target_prevalence=target_prevalence,
            n_synthetic=0,
        )

    k_neighbors = min(5, n_pos_before - 1)
    if k_neighbors < 1:
        raise ValueError(
            f"SMOTE requires at least 2 positive samples; got {n_pos_before}."
        )

    ratio = prevalence_to_ratio(target_prevalence)
    smote = SMOTE(
        sampling_strategy=ratio,
        k_neighbors=k_neighbors,
        random_state=random_state,
    )
    logger.info(f"SMOTE fit_resample starting (ratio={ratio:.4f}, k={k_neighbors}) ...")
    X_res, y_res = smote.fit_resample(X, y)
    logger.info("SMOTE fit_resample done.")

    class_weight = compute_class_weights_from_data(y_original)

    n_pos_after = int(y_res.sum())
    n_synthetic = n_pos_after - n_pos_before
    achieved = compute_achieved_prevalence(y_res)

    return SamplingResult(
        X=X_res, y=y_res,
        class_weight=class_weight,
        achieved_prevalence=achieved,
        n_positive=n_pos_after,
        n_negative=int((1 - y_res).sum()),
        strategy="smote_class_weighting",
        target_prevalence=target_prevalence,
        n_synthetic=n_synthetic,
    )


def _adasyn(
    X: np.ndarray,
    y: np.ndarray,
    target_prevalence: float,
    random_state: int,
) -> SamplingResult:
    """Oversample the minority class with ADASYN to reach ``target_prevalence``."""
    from imblearn.over_sampling import ADASYN

    natural = compute_achieved_prevalence(y)
    n_pos_before = int(y.sum())

    if target_prevalence <= natural:
        logger.warning(
            f"ADASYN: target_prevalence={target_prevalence:.4%} <= "
            f"natural={natural:.6%}. Natural prevalence already meets target; "
            "returning data unchanged."
        )
        return _baseline(X, y, target_prevalence)

    n_neighbors = min(5, n_pos_before - 1)
    if n_neighbors < 1:
        raise ValueError(
            f"ADASYN requires at least 2 positive samples; got {n_pos_before}."
        )

    ratio = prevalence_to_ratio(target_prevalence)
    try:
        MAX_MAJORITY_FOR_KNN = 500_000
        n_majority = int((y == 0).sum())

        if n_majority > MAX_MAJORITY_FOR_KNN:
            logger.info(
                f"ADASYN: majority class ({n_majority:,} samples) exceeds "
                f"{MAX_MAJORITY_FOR_KNN:,} — subsampling for KNN density estimation."
            )
            rng = np.random.default_rng(random_state)
            maj_idx = np.where(y == 0)[0]
            min_idx = np.where(y == 1)[0]
            sub_maj_idx = rng.choice(maj_idx, size=MAX_MAJORITY_FOR_KNN, replace=False)
            idx_sub = np.concatenate([sub_maj_idx, min_idx])
            X_sub = X[idx_sub]
            y_sub = y[idx_sub]

            # Compute ratio for the subsample
            n_pos_sub = int(y_sub.sum())
            n_neg_sub = int((y_sub == 0).sum())
            # Target: enough synthetic samples to reach target_prevalence on full data
            n_pos_needed = int(round(target_prevalence * n_majority / (1 - target_prevalence)))
            n_synthetic_needed = max(0, n_pos_needed - int(y.sum()))
            ratio_sub = (n_pos_sub + n_synthetic_needed) / n_neg_sub

            adasyn = ADASYN(
                sampling_strategy=min(ratio_sub, 1.0),
                n_neighbors=n_neighbors,
                random_state=random_state,
            )
            logger.info(f"ADASYN fit_resample starting (ratio={ratio_sub:.4f}, n_neighbors={n_neighbors}) ...")
            X_res_sub, y_res_sub = adasyn.fit_resample(X_sub, y_sub)
            logger.info("ADASYN fit_resample done.")

            # Extract only the synthetic minority samples
            n_synthetic = len(y_res_sub) - len(y_sub)
            X_synthetic = X_res_sub[len(y_sub):]
            y_synthetic = y_res_sub[len(y_sub):]

            # Append synthetic samples to full original data
            X_res = np.concatenate([X, X_synthetic])
            y_res = np.concatenate([y, y_synthetic])
        else:
            adasyn = ADASYN(
                sampling_strategy=ratio,
                n_neighbors=n_neighbors,
                random_state=random_state,
            )
            logger.info(f"ADASYN fit_resample starting (ratio={ratio:.4f}, n_neighbors={n_neighbors}) ...")
            X_res, y_res = adasyn.fit_resample(X, y)
            logger.info("ADASYN fit_resample done.")
    except RuntimeError as exc:
        # ADASYN can fail if density estimation produces an all-zero sample
        # distribution; fall back to SMOTE in that case.
        logger.warning(
            f"ADASYN failed ({exc}). Falling back to SMOTE for this condition."
        )
        return _smote(X, y, target_prevalence, random_state)

    n_pos_after = int(y_res.sum())
    n_synthetic = n_pos_after - n_pos_before
    achieved = compute_achieved_prevalence(y_res)

    return SamplingResult(
        X=X_res, y=y_res,
        class_weight=None,
        achieved_prevalence=achieved,
        n_positive=n_pos_after,
        n_negative=int((1 - y_res).sum()),
        strategy="adasyn",
        target_prevalence=target_prevalence,
        n_synthetic=n_synthetic,
    )


def _class_weighting(
    X: np.ndarray,
    y: np.ndarray,
    target_prevalence: float,
) -> SamplingResult:
    """Return full training data with class weights derived from target prevalence."""
    achieved = compute_achieved_prevalence(y)
    class_weight = compute_class_weights(target_prevalence)
    logger.info(
        f"Class weighting: no resampling. "
        f"Natural prevalence={achieved:.6%}. "
        f"Model weights: {class_weight}."
    )
    return SamplingResult(
        X=X, y=y,
        class_weight=class_weight,
        achieved_prevalence=achieved,
        n_positive=int(y.sum()),
        n_negative=int((1 - y).sum()),
        strategy="class_weighting",
        target_prevalence=target_prevalence,
    )


def _true_cost_weighting(
    X: np.ndarray,
    y: np.ndarray,
    target_prevalence: float,
) -> SamplingResult:
    """Class weighting based on the actual class ratio in the training data.

    Part B strategy: uses scale_pos_weight = n_negatives / n_positives,
    i.e. the true imbalance ratio (~1930x for the Large dataset).
    This differs from class_weighting which derives weights from
    target_prevalence. No resampling is performed.

    Parameters
    ----------
    X:
        Feature matrix for the training split.
    y:
        Binary label array for the training split.
    target_prevalence:
        Recorded for metadata but not used to derive weights.
    """
    from aml_benchmark.sampling.prevalence import compute_class_weights_from_data

    achieved = compute_achieved_prevalence(y)
    class_weight = compute_class_weights_from_data(y)

    logger.info(
        f"True cost weighting: natural prevalence={achieved:.6%}. "
        f"Model weights: {class_weight}."
    )
    return SamplingResult(
        X=X, y=y,
        class_weight=class_weight,
        achieved_prevalence=achieved,
        n_positive=int(y.sum()),
        n_negative=int((1 - y).sum()),
        strategy="true_cost_weighting",
        target_prevalence=target_prevalence,
    )
