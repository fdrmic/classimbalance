"""Model factory for the AML benchmark.

Returns a freshly instantiated, unfitted estimator ready for training.
All models use ``random_state`` for reproducibility.

Class-weight integration
------------------------
The optional ``class_weight`` parameter accepts a dict
``{0: w_negative, 1: w_positive}`` returned by the class-weighting
sampling strategy.  Each model family handles this differently:

* **RandomForest** – passed directly as the ``class_weight`` constructor
  argument (sklearn native support).
* **XGBoost** – the positive-class weight ``w1`` is extracted and passed
  as ``scale_pos_weight``, which multiplies the gradient contribution of
  every positive sample.  The ratio ``w1 / w0`` is used so that the
  relative weighting is preserved even if ``w0 != 1.0``.

Supported model names
---------------------
``"random_forest"``  –  sklearn RandomForestClassifier
``"xgboost"``        –  XGBClassifier (requires the ``xgboost`` package)

Usage
-----
    # Unweighted
    model = get_model("random_forest", random_state=42)

    # Class-weighted (from sampling strategy)
    model = get_model("xgboost", random_state=42, class_weight={0: 1.0, 1: 999.0})
"""
from __future__ import annotations

from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier

from aml_benchmark.utils.logging_utils import get_logger

logger = get_logger(__name__)

_SUPPORTED = ("random_forest", "xgboost")


def get_model(
    name: str,
    random_state: int = 42,
    class_weight: dict[int, float] | None = None,
) -> BaseEstimator:
    """Instantiate a benchmark model by name.

    Parameters
    ----------
    name:
        One of ``"random_forest"`` or ``"xgboost"``.
    random_state:
        Integer seed for reproducibility.
    class_weight:
        Optional dict ``{0: w0, 1: w1}`` from the class-weighting
        strategy.  If ``None``, no class weighting is applied.

    Returns
    -------
    An unfitted sklearn-compatible estimator.
    """
    name = name.lower().strip()

    if name == "random_forest":
        return _random_forest(random_state, class_weight)

    if name == "xgboost":
        return _xgboost(random_state, class_weight)

    raise ValueError(f"Unknown model '{name}'. Supported: {_SUPPORTED}")


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

def _random_forest(
    random_state: int,
    class_weight: dict[int, float] | None,
) -> RandomForestClassifier:
    """RandomForest configured for large, imbalanced transaction data.

    ``max_samples=200_000`` caps how many rows each tree sees, keeping
    memory and runtime manageable on training splits up to ~5 M rows.
    """
    if class_weight is not None:
        logger.info(
            f"RandomForest class_weight: {class_weight}"
        )

    model = RandomForestClassifier(
        n_estimators=100,
        max_features="sqrt",
        max_samples=200_000,
        min_samples_leaf=5,
        n_jobs=4,
        class_weight=class_weight,
        random_state=random_state,
        verbose=2,
    )
    logger.info(
        f"Model: RandomForestClassifier | "
        f"n_estimators=100, max_samples=200_000, "
        f"class_weight={class_weight}, random_state={random_state}"
    )
    return model


def _detect_xgb_device() -> str:
    """Return ``"cuda"`` if an NVIDIA GPU is available, else ``"cpu"``."""
    import subprocess
    try:
        subprocess.check_output(["nvidia-smi"], stderr=subprocess.DEVNULL)
        return "cuda"
    except Exception:
        return "cpu"


def _xgboost(
    random_state: int,
    class_weight: dict[int, float] | None,
) -> "XGBClassifier":
    """XGBClassifier configured for the AML benchmark.

    When ``class_weight`` is provided, ``scale_pos_weight`` is set to
    ``w1 / w0``, which instructs XGBoost to up-weight positive-class
    gradient updates proportionally.

    GPU acceleration is enabled automatically when an NVIDIA GPU is
    detected (``nvidia-smi`` available).  Falls back to CPU silently.
    ``n_jobs=-1`` ensures all CPU cores are used regardless of device.

    Requires the ``xgboost`` package (``pip install xgboost``).
    """
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise ImportError(
            "xgboost is not installed. Run: pip install xgboost"
        ) from exc

    # Auto-detect GPU
    device = _detect_xgb_device()
    logger.info(f"XGBoost device: {device}")

    # Derive scale_pos_weight from class_weight dict if provided
    scale_pos_weight: float | None = None
    if class_weight is not None:
        w0 = float(class_weight.get(0, 1.0))
        w1 = float(class_weight.get(1, 1.0))
        scale_pos_weight = w1 / w0
        logger.info(
            f"XGBoost scale_pos_weight={scale_pos_weight:.4f} "
            f"(derived from class_weight {class_weight})"
        )

    kwargs = dict(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="aucpr",
        tree_method="hist",
        device=device,
        n_jobs=-1,
        random_state=random_state,
        verbosity=0,
    )
    if scale_pos_weight is not None:
        kwargs["scale_pos_weight"] = scale_pos_weight

    model = XGBClassifier(**kwargs)
    logger.info(
        f"Model: XGBClassifier | n_estimators=200, max_depth=6, "
        f"lr=0.05, device={device}, n_jobs=-1, "
        f"scale_pos_weight={scale_pos_weight}, random_state={random_state}"
    )
    return model
