"""Raw data loaders for the three IBM AML source files.

Each loader:
* reads the file with ``dtype=str`` to preserve leading zeros in IDs,
* renames / reorders columns to the project schema,
* applies correct dtypes afterwards,
* logs a brief summary.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from aml_benchmark.data.schema import (
    ACCOUNTS_COLUMNS,
    RAW_TRANS_COLUMNS,
    apply_transaction_dtypes,
)
from aml_benchmark.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Timestamp format used consistently in both the CSV and the patterns file.
_TS_FORMAT = "%Y/%m/%d %H:%M"


def load_transactions(path: Path, chunk_size: int = 10_000_000) -> pd.DataFrame:
    """Load the transaction CSV memory-efficiently via chunked reading.

    The raw file has 11 columns but the header row contains a duplicate
    "Account" name for columns 2 (sender) and 4 (receiver).  We skip the
    original header (``header=0``) and supply unambiguous names via
    ``names=RAW_TRANS_COLUMNS``.

    Chunked reading keeps peak RAM usage well below the full file size
    (~15-20 GB instead of ~50 GB for the LI-Large dataset) by applying
    dtype conversions per chunk before concatenating.

    Parameters
    ----------
    path:
        Absolute path to the transactions CSV (Small or Large variant).
    chunk_size:
        Number of rows per chunk.  Default 10 M rows works well for Colab
        (12-16 GB RAM).  Reduce if memory is limited.

    Returns
    -------
    DataFrame with columns defined in ``RAW_TRANS_COLUMNS``, a parsed
    ``timestamp`` column, and ``is_laundering_csv`` cast to int.
    """
    path = Path(path)
    logger.info(f"Loading transactions from {path} (chunk_size={chunk_size:,}) ...")

    chunks = []
    total_read = 0
    total_bad_ts = 0

    for chunk in pd.read_csv(
        path,
        header=0,           # skip the original (duplicate) header row
        names=RAW_TRANS_COLUMNS,
        dtype=str,          # read all as str first; avoids leading-zero loss
        low_memory=False,
        chunksize=chunk_size,
    ):
        # Parse timestamp per chunk
        chunk["timestamp"] = pd.to_datetime(
            chunk["timestamp"], format=_TS_FORMAT, errors="coerce"
        )

        # Apply numeric / string dtypes
        chunk = apply_transaction_dtypes(chunk)

        # Preserve original label as int
        chunk["is_laundering_csv"] = (
            pd.to_numeric(chunk["is_laundering_csv"], errors="coerce")
            .fillna(0)
            .astype(int)
        )

        # Count and drop rows with unparseable timestamps
        n_bad = int(chunk["timestamp"].isna().sum())
        if n_bad:
            total_bad_ts += n_bad
            chunk = chunk.dropna(subset=["timestamp"])

        chunks.append(chunk)
        total_read += len(chunk)
        logger.info(f"  Read so far: {total_read:,} rows ...")

    if total_bad_ts:
        logger.warning(
            f"{total_bad_ts:,} rows with unparseable timestamps were dropped."
        )

    df = pd.concat(chunks, ignore_index=True)

    logger.info(
        f"Loaded {len(df):,} transactions | "
        f"date range: {df['timestamp'].min()} -> {df['timestamp'].max()}"
    )
    return df


def load_accounts(path: Path) -> pd.DataFrame:
    """Load the accounts CSV.

    Parameters
    ----------
    path:
        Absolute path to ``LI-Small_accounts.csv``.

    Returns
    -------
    DataFrame with columns defined in ``ACCOUNTS_COLUMNS``.
    """
    path = Path(path)
    logger.info(f"Loading accounts from {path} ...")

    df = pd.read_csv(
        path,
        header=0,
        names=ACCOUNTS_COLUMNS,
        dtype=str,
        low_memory=False,
    )
    df = df.dropna(how="all")

    logger.info(f"Loaded {len(df):,} accounts")
    return df
