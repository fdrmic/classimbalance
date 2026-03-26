"""Schema constants, column naming, and dtype enforcement.

The IBM AML transaction CSV has 11 columns but ships with a duplicate
column header ("Account" appears for both sender and receiver).
This module maps those 11 positional columns to unambiguous names and
defines the canonical subset used by every downstream module.
"""
from __future__ import annotations

import pandas as pd

# ---------------------------------------------------------------------------
# Column name definitions
# ---------------------------------------------------------------------------

# The raw CSV has these 11 columns in order.
# The original header row is skipped; we supply names explicitly.
RAW_TRANS_COLUMNS: list[str] = [
    "timestamp",            # 0  – "Timestamp"
    "from_bank",            # 1  – "From Bank"
    "from_account",         # 2  – "Account"  (sender)
    "to_bank",              # 3  – "To Bank"
    "to_account",           # 4  – "Account"  (receiver – duplicate in raw CSV)
    "amount_received",      # 5  – "Amount Received"
    "receiving_currency",   # 6  – "Receiving Currency"
    "amount_paid",          # 7  – "Amount Paid"
    "payment_currency",     # 8  – "Payment Currency"
    "payment_format",       # 9  – "Payment Format"
    "is_laundering_csv",    # 10 – "Is Laundering" (original CSV ground-truth label)
]

# Canonical columns exposed to feature engineering and modelling stages.
# Excludes raw / redundant fields that are not part of the core schema.
CANONICAL_TRANS_COLUMNS: list[str] = [
    "timestamp",
    "from_bank",
    "from_account",
    "to_bank",
    "to_account",
    "amount_paid",
    "payment_currency",
    "payment_format",
]

# Accounts CSV – the original header is already unambiguous.
ACCOUNTS_COLUMNS: list[str] = [
    "bank_name",
    "bank_id",
    "account_number",
    "entity_id",
    "entity_name",
]

# ---------------------------------------------------------------------------
# Dtype mappings
# ---------------------------------------------------------------------------

# Applied *after* initial string read so that leading zeros in IDs are
# preserved during the str→str cast.
TRANSACTION_DTYPES: dict[str, type] = {
    "from_bank": str,
    "from_account": str,
    "to_bank": str,
    "to_account": str,
    "receiving_currency": str,
    "amount_received": float,
    "amount_paid": float,
    "payment_currency": str,
    "payment_format": str,
}


def apply_transaction_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Cast each transaction column to its canonical dtype in-place.

    String columns are also stripped of surrounding whitespace.

    Parameters
    ----------
    df:
        Transaction DataFrame with column names matching RAW_TRANS_COLUMNS
        or any canonical subset thereof.

    Returns
    -------
    The same DataFrame with corrected dtypes (operates on a copy of values,
    not the original memory).
    """
    for col, dtype in TRANSACTION_DTYPES.items():
        if col not in df.columns:
            continue
        if dtype is str:
            df[col] = df[col].astype(str).str.strip()
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df
