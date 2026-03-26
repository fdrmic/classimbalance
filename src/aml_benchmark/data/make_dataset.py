"""CLI entrypoint: build the labeled transaction dataset.

Usage
-----
From the project root (after ``pip install -e .``):

    python -m aml_benchmark.data.make_dataset

Steps executed
--------------
1. Validate that raw files exist.
2. Load transactions CSV  →  ``DataFrame``
3. Load accounts CSV      →  ``DataFrame``
4. Parse patterns TXT     →  ``DataFrame``
5. Generate labels        →  join patterns onto transactions
6. Save labeled dataset   →  ``data/processed/transactions_labeled.parquet``
7. Print summary.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

from aml_benchmark.config import PathConfig
from aml_benchmark.data.ingest import load_accounts, load_transactions
from aml_benchmark.data.labeler import create_labels
from aml_benchmark.data.pattern_parser import parse_patterns
from aml_benchmark.utils.io import save_parquet
from aml_benchmark.utils.logging_utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(paths: PathConfig | None = None) -> pd.DataFrame:
    """Execute the full data ingestion and labeling pipeline.

    Parameters
    ----------
    paths:
        Optional pre-constructed :class:`~aml_benchmark.config.PathConfig`.
        If *None*, a fresh instance is built from ``configs/paths.yaml``.

    Returns
    -------
    The labeled transaction DataFrame that was persisted to disk.
    """
    t0 = time.perf_counter()

    if paths is None:
        paths = PathConfig()

    # Validate raw inputs before doing any heavy I/O
    paths.validate()

    # 1. Load raw data
    transactions = load_transactions(paths.transactions_path)
    accounts = load_accounts(paths.accounts_path)

    # 2. Parse laundering patterns
    patterns = parse_patterns(paths.patterns_path)

    # 3. Generate labels
    labeled = create_labels(transactions, patterns)

    # 4. Persist
    paths.processed_dir.mkdir(parents=True, exist_ok=True)
    save_parquet(labeled, paths.output_transactions_labeled)

    elapsed = time.perf_counter() - t0

    # 5. Final summary
    _print_final_summary(labeled, accounts, paths.output_transactions_labeled, elapsed)

    return labeled


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def _print_final_summary(
    df: pd.DataFrame,
    accounts: pd.DataFrame,
    output_path: Path,
    elapsed_sec: float,
) -> None:
    total = len(df)
    n_illicit = int(df["label"].sum())
    ratio = n_illicit / total if total else 0.0
    n_mismatch = int(df["mismatch_flag"].sum())
    ts_min = df["timestamp"].min()
    ts_max = df["timestamp"].max()

    print()
    print("=" * 62)
    print("  AML BENCHMARK - LABELED DATASET READY")
    print("=" * 62)
    print(f"  Accounts loaded              : {len(accounts):>10,}")
    print(f"  Total transactions           : {total:>10,}")
    print(f"  Illicit transactions (label) : {n_illicit:>10,}")
    print(f"  Illicit ratio                : {ratio:>10.4%}")
    print(f"  Mismatch flags               : {n_mismatch:>10,}")
    print(f"  Date range                   : {ts_min}  to  {ts_max}")
    print(f"  Output columns               : {list(df.columns)}")
    print(f"  Saved to                     : {output_path}")
    print(f"  Elapsed                      : {elapsed_sec:.1f}s")
    print("=" * 62)
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Main entry point; exits with code 1 on failure."""
    try:
        run_pipeline()
    except FileNotFoundError as exc:
        logger.error(str(exc))
        sys.exit(1)
    except Exception as exc:
        logger.exception(f"Pipeline failed: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
