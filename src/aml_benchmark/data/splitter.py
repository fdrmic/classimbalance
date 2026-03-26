"""Chronological train / validation / test splitter.

The labeled dataset is split **strictly by time** to prevent any form of
temporal leakage into validation or test sets.  Rows are sorted by
``timestamp`` ascending and then divided at quantile boundaries.

Split ratios are read from ``configs/split.yaml``:

    train_ratio : float  (default 0.70)
    val_ratio   : float  (default 0.15)
    test_ratio  : implicitly 1 - train_ratio - val_ratio

Outputs
-------
    data/splits/train.parquet
    data/splits/val.parquet
    data/splits/test.parquet
    data/splits/split_manifest.json

Usage
-----
    python -m aml_benchmark.data.splitter
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from aml_benchmark.config import PathConfig, load_yaml
from aml_benchmark.utils.io import load_parquet, save_parquet
from aml_benchmark.utils.logging_utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Core split logic
# ---------------------------------------------------------------------------

def split_chronological(
    df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a time-sorted DataFrame into train / val / test.

    Parameters
    ----------
    df:
        Labeled transaction DataFrame with a ``timestamp`` column.
        Need not be pre-sorted – this function sorts it.
    train_ratio:
        Fraction of rows assigned to the training split.
    val_ratio:
        Fraction of rows assigned to the validation split.
        The test split receives the remainder.

    Returns
    -------
    (train, val, test) DataFrames, each sorted by timestamp,
    with disjoint, non-overlapping time intervals.
    """
    if not (0.0 < train_ratio < 1.0):
        raise ValueError(f"train_ratio must be in (0, 1); got {train_ratio}")
    if not (0.0 < val_ratio < 1.0):
        raise ValueError(f"val_ratio must be in (0, 1); got {val_ratio}")
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be < 1.0")

    df_sorted = df.sort_values("timestamp").reset_index(drop=True)
    n = len(df_sorted)

    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train = df_sorted.iloc[:train_end].copy()
    val = df_sorted.iloc[train_end:val_end].copy()
    test = df_sorted.iloc[val_end:].copy()

    return train, val, test


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def _split_stats(name: str, df: pd.DataFrame) -> dict:
    """Build a stats dict for one split."""
    n = len(df)
    n_pos = int(df["label"].sum())
    n_neg = n - n_pos
    return {
        "split": name,
        "n_rows": n,
        "n_positive": n_pos,
        "n_negative": n_neg,
        "illicit_ratio": round(n_pos / n, 8) if n else 0.0,
        "date_start": str(df["timestamp"].min()),
        "date_end": str(df["timestamp"].max()),
    }


def save_split_manifest(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    manifest_path: Path,
) -> None:
    """Write a JSON manifest with counts, ratios, and date ranges."""
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": round(1.0 - train_ratio - val_ratio, 6),
        "splits": [
            _split_stats("train", train),
            _split_stats("val", val),
            _split_stats("test", test),
        ],
    }
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    logger.info(f"Split manifest saved -> {manifest_path}")


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_split(paths: PathConfig | None = None) -> None:
    """Load the labeled dataset, split it chronologically, and persist."""
    t0 = time.perf_counter()

    if paths is None:
        paths = PathConfig()

    if not paths.output_transactions_labeled.exists():
        raise FileNotFoundError(
            f"Labeled dataset not found: {paths.output_transactions_labeled}\n"
            "Run: python -m aml_benchmark.data.make_dataset"
        )

    split_cfg = load_yaml("split")
    train_ratio: float = float(split_cfg.get("train_ratio", 0.70))
    val_ratio: float = float(split_cfg.get("val_ratio", 0.15))

    logger.info(
        f"Split ratios: train={train_ratio:.0%}  "
        f"val={val_ratio:.0%}  "
        f"test={1 - train_ratio - val_ratio:.0%}"
    )

    # Load
    df = load_parquet(paths.output_transactions_labeled)

    # Split
    train, val, test = split_chronological(df, train_ratio, val_ratio)

    # Save splits
    paths.splits_dir.mkdir(parents=True, exist_ok=True)
    save_parquet(train, paths.train_split)
    save_parquet(val, paths.val_split)
    save_parquet(test, paths.test_split)

    # Save manifest
    save_split_manifest(
        train, val, test, train_ratio, val_ratio, paths.split_manifest
    )

    elapsed = time.perf_counter() - t0
    _print_summary(train, val, test, elapsed)


def _print_summary(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    elapsed_sec: float,
) -> None:
    print()
    print("=" * 62)
    print("  TEMPORAL SPLIT SUMMARY")
    print("=" * 62)
    for name, df in [("train", train), ("val", val), ("test", test)]:
        n = len(df)
        n_pos = int(df["label"].sum())
        ratio = n_pos / n if n else 0.0
        ts_min = df["timestamp"].min()
        ts_max = df["timestamp"].max()
        print(f"  {name:<6} : {n:>10,} rows | "
              f"{n_pos:>5,} illicit ({ratio:.4%}) | "
              f"{ts_min}  to  {ts_max}")
    print(f"  Elapsed  : {elapsed_sec:.1f}s")
    print("=" * 62)
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    try:
        run_split()
    except FileNotFoundError as exc:
        logger.error(str(exc))
        sys.exit(1)
    except Exception as exc:
        logger.exception(f"Splitter failed: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
