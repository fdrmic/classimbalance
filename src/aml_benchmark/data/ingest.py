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


def load_transactions(path: Path) -> pd.DataFrame:
    """Load the transaction CSV and return a cleaned DataFrame.

    The raw file has 11 columns but the header row contains a duplicate
    "Account" name for columns 2 (sender) and 4 (receiver).  We skip the
    original header (``header=0``) and supply unambiguous names via
    ``names=RAW_TRANS_COLUMNS``.

    Parameters
    ----------
    path:
        Absolute path to ``LI-Small_Trans.csv``.

    Returns
    -------
    DataFrame with columns defined in ``RAW_TRANS_COLUMNS``, a parsed
    ``timestamp`` column, and ``is_laundering_csv`` cast to int.
    """
    path = Path(path)
    logger.info(f"Loading transactions from {path} ...")

    df = pd.read_csv(
        path,
        header=0,           # skip the original (duplicate) header row
        names=RAW_TRANS_COLUMNS,
        dtype=str,          # read all as str first; avoids leading-zero loss
        low_memory=False,
    )

    # Parse timestamp – errors produce NaT, logged and dropped below
    df["timestamp"] = pd.to_datetime(
        df["timestamp"], format=_TS_FORMAT, errors="coerce"
    )

    # Apply numeric / string dtypes
    df = apply_transaction_dtypes(df)

    # Preserve original label as int
    df["is_laundering_csv"] = (
        pd.to_numeric(df["is_laundering_csv"], errors="coerce")
        .fillna(0)
        .astype(int)
    )

    n_bad_ts = int(df["timestamp"].isna().sum())
    if n_bad_ts:
        logger.warning(
            f"{n_bad_ts:,} rows with unparseable timestamps will be dropped."
        )
        df = df.dropna(subset=["timestamp"])

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
