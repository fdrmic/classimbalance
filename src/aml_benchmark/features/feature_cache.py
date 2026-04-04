"""Feature caching for AML benchmark v2.

Saves precomputed feature matrices to parquet so that account-level
features are only computed once and reused across all 30 runs.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from aml_benchmark.utils.logging_utils import get_logger

logger = get_logger(__name__)


def _cache_path(splits_dir: Path, split_name: str) -> Path:
    return splits_dir / f"{split_name}_features_v2.parquet"


def cache_exists(splits_dir: Path) -> bool:
    """Return True if all three split caches exist."""
    return all(
        _cache_path(splits_dir, s).exists()
        for s in ("train", "val", "test")
    )


def save_features(
    X: np.ndarray,
    feature_names: list[str],
    splits_dir: Path,
    split_name: str,
) -> None:
    p = _cache_path(splits_dir, split_name)
    pd.DataFrame(X, columns=feature_names).to_parquet(p, index=False)
    logger.info(f"Feature cache saved -> {p}  shape={X.shape}")


def load_features(splits_dir: Path, split_name: str) -> np.ndarray:
    p = _cache_path(splits_dir, split_name)
    logger.info(f"Loading feature cache <- {p}")
    return pd.read_parquet(p).to_numpy(dtype=np.float64)
