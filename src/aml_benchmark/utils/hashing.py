"""Deterministic match-key generation for transaction-to-pattern linking.

A match key is a pipe-delimited string built from the eight fields that
uniquely identify a transaction in both the transactions CSV and the
patterns text file:

    <timestamp>|<from_bank>|<from_account>|<to_bank>|<to_account>
    |<amount_paid>|<payment_currency>|<payment_format>

Normalisation rules applied to every field before concatenation:

* **timestamp** – parsed to ``datetime`` then formatted as
  ``YYYY-MM-DD HH:MM`` (minute precision, no seconds).
* **bank / account IDs** – stripped of leading/trailing whitespace,
  lowercased.  Leading zeros are intentionally preserved because
  ``"011"`` and ``"11"`` identify different banks in the IBM dataset.
* **amount_paid** – cast to ``float``, rounded to 2 decimal places,
  zero-padded to exactly 2 decimals (e.g. ``"10154.74"``).
* **currency / format** – stripped, lowercased.
"""
from __future__ import annotations

import pandas as pd

_AMOUNT_DECIMALS = 2
_SEP = "|"


# ---------------------------------------------------------------------------
# Internal normalisation helpers
# ---------------------------------------------------------------------------

def _norm_str(series: pd.Series) -> pd.Series:
    """Strip whitespace and lowercase a string Series.

    Leading zeros (e.g. bank IDs ``"011"``) are preserved because
    ``str.lower()`` does not modify digits.
    """
    return series.astype(str).str.strip().str.lower()


def _norm_timestamp(series: pd.Series) -> pd.Series:
    """Coerce to datetime then format at minute precision.

    Minute-level granularity is used because both the transaction CSV and
    the patterns file record timestamps without seconds.
    """
    return pd.to_datetime(series, errors="coerce").dt.strftime("%Y-%m-%d %H:%M")


def _norm_amount(series: pd.Series) -> pd.Series:
    """Round to fixed decimal precision and format as zero-padded string.

    Using an explicit format string avoids floating-point string
    representation variance (e.g. ``"5326.7"`` vs ``"5326.70"``).
    """
    return series.astype(float).round(_AMOUNT_DECIMALS).map(
        lambda x: f"{x:.{_AMOUNT_DECIMALS}f}"
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def make_match_key(df: pd.DataFrame) -> pd.Series:
    """Build a deterministic, pipe-delimited match key for each row.

    Parameters
    ----------
    df:
        DataFrame that must contain the following columns:
        ``timestamp``, ``from_bank``, ``from_account``, ``to_bank``,
        ``to_account``, ``amount_paid``, ``payment_currency``,
        ``payment_format``.

    Returns
    -------
    pd.Series of string keys with the same index as *df*.
    Rows where ``timestamp`` or ``amount_paid`` cannot be parsed will
    produce keys containing ``"NaT"`` or ``"nan"`` – these are excluded
    from set-based lookups via ``dropna()`` at the call site.
    """
    required = {
        "timestamp", "from_bank", "from_account",
        "to_bank", "to_account",
        "amount_paid", "payment_currency", "payment_format",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"make_match_key: missing columns {missing}")

    key = (
        _norm_timestamp(df["timestamp"])
        + _SEP + _norm_str(df["from_bank"])
        + _SEP + _norm_str(df["from_account"])
        + _SEP + _norm_str(df["to_bank"])
        + _SEP + _norm_str(df["to_account"])
        + _SEP + _norm_amount(df["amount_paid"])
        + _SEP + _norm_str(df["payment_currency"])
        + _SEP + _norm_str(df["payment_format"])
    )
    return key
