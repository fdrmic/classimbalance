"""Parser for ``LI-Small_Patterns.txt``.

The patterns file encodes individual laundering scenarios ("attempts") as
blocks delimited by ``BEGIN LAUNDERING ATTEMPT`` / ``END LAUNDERING ATTEMPT``
markers.  Each block contains one or more transaction data lines in the same
11-field CSV format used by the transaction CSV.

This module extracts only the data lines, discards all markers and blank
lines, and returns a structured DataFrame enriched with the laundering
pattern type and a sequential block index.

Laundering pattern types observed in the IBM AML dataset include:
    FAN-IN, FAN-OUT, GATHER-SCATTER, SCATTER-GATHER, RANDOM, STACK
"""
from __future__ import annotations

import io
import re
from pathlib import Path

import pandas as pd

from aml_benchmark.data.schema import RAW_TRANS_COLUMNS, apply_transaction_dtypes
from aml_benchmark.utils.logging_utils import get_logger

logger = get_logger(__name__)

_BEGIN_MARKER = "BEGIN LAUNDERING ATTEMPT"
_END_MARKER = "END LAUNDERING ATTEMPT"
_TS_FORMAT = "%Y/%m/%d %H:%M"

# Pattern type is the text between " - " and ":" on the BEGIN line,
# e.g. "BEGIN LAUNDERING ATTEMPT - FAN-IN:  Max 3-degree Fan-In" → "FAN-IN"
_TYPE_RE = re.compile(r"BEGIN LAUNDERING ATTEMPT\s*-\s*([^:]+)")


def _extract_pattern_type(line: str) -> str:
    """Return the laundering type from a BEGIN marker line."""
    match = _TYPE_RE.search(line)
    return match.group(1).strip() if match else "UNKNOWN"


def parse_patterns(path: Path) -> pd.DataFrame:
    """Parse the patterns text file into a structured transaction DataFrame.

    Only transaction data lines are extracted.  Block header/footer lines
    and blank lines are silently discarded.  Each transaction is annotated
    with the laundering pattern type and a sequential block index (1-based).

    Parameters
    ----------
    path:
        Absolute path to ``LI-Small_Patterns.txt``.

    Returns
    -------
    DataFrame with the columns from ``RAW_TRANS_COLUMNS`` plus:

    * ``pattern_type``       – string label such as ``"FAN-IN"``.
    * ``pattern_block_idx``  – integer block counter (1 = first block).

    ``is_laundering_csv`` is set to ``1`` for every row (all pattern
    transactions are by definition illicit).
    """
    path = Path(path)
    logger.info(f"Parsing patterns from {path} ...")

    data_lines: list[str] = []
    meta_rows: list[dict] = []

    current_type: str = "UNKNOWN"
    block_idx: int = 0

    with path.open(encoding="utf-8", errors="replace") as fh:
        for raw_line in fh:
            line = raw_line.strip()

            if not line:
                continue

            if line.startswith(_BEGIN_MARKER):
                current_type = _extract_pattern_type(line)
                block_idx += 1
                continue

            if line.startswith(_END_MARKER):
                continue

            # Data line – expected to be a comma-separated transaction row
            data_lines.append(line)
            meta_rows.append(
                {"pattern_type": current_type, "pattern_block_id": block_idx}
            )

    if not data_lines:
        logger.warning("No transaction data lines found in patterns file.")
        return pd.DataFrame(
            columns=RAW_TRANS_COLUMNS + ["pattern_type", "pattern_block_id"]
        )

    logger.info(
        f"Extracted {len(data_lines):,} data lines from {block_idx} blocks."
    )

    # Parse data lines as CSV (same 11-field format as transactions CSV)
    df_data = pd.read_csv(
        io.StringIO("\n".join(data_lines)),
        header=None,
        names=RAW_TRANS_COLUMNS,
        dtype=str,
        on_bad_lines="warn",
    )

    df_meta = pd.DataFrame(meta_rows)

    # on_bad_lines="warn" may drop malformed rows; align lengths defensively
    min_len = min(len(df_data), len(df_meta))
    if len(df_data) != len(df_meta):
        logger.warning(
            f"Row count mismatch after CSV parse (data={len(df_data)}, "
            f"meta={len(df_meta)}); truncating to {min_len}."
        )

    df = pd.concat(
        [
            df_data.iloc[:min_len].reset_index(drop=True),
            df_meta.iloc[:min_len].reset_index(drop=True),
        ],
        axis=1,
    )

    # Parse timestamp
    df["timestamp"] = pd.to_datetime(
        df["timestamp"], format=_TS_FORMAT, errors="coerce"
    )

    df = apply_transaction_dtypes(df)

    n_bad_ts = int(df["timestamp"].isna().sum())
    if n_bad_ts:
        logger.warning(
            f"{n_bad_ts:,} pattern rows with unparseable timestamps will be dropped."
        )
        df = df.dropna(subset=["timestamp"])

    # All pattern transactions are illicit by definition
    df["is_laundering_csv"] = 1

    logger.info(
        f"Parsed {len(df):,} pattern transactions | "
        f"{df['pattern_block_id'].nunique()} blocks | "
        f"pattern types: {sorted(df['pattern_type'].unique())}"
    )
    return df
