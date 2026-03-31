"""File I/O helpers used across the pipeline."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from aml_benchmark.utils.logging_utils import get_logger

logger = get_logger(__name__)


def save_parquet(df: pd.DataFrame, path: Path) -> None:
    """Persist a DataFrame as Parquet, creating parent directories as needed.

    Parameters
    ----------
    df:
        DataFrame to save.
    path:
        Destination file path (will be created if absent).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving {len(df):,} rows -> {path} ...")
    df.to_parquet(path, index=False, engine="pyarrow")
    logger.info(f"Saved {len(df):,} rows -> {path}")


def load_parquet(path: Path) -> pd.DataFrame:
    """Load a Parquet file into a DataFrame.

    Parameters
    ----------
    path:
        Source file path.
    """
    path = Path(path)
    df = pd.read_parquet(path, engine="pyarrow")
    logger.info(f"Loaded {len(df):,} rows <- {path}")
    return df
