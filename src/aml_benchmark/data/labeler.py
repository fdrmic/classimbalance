"""Label generation: match pattern transactions to the transaction table.

Matching is done via a deterministic key (see ``utils.hashing.make_match_key``).
Four label columns and two pattern-metadata columns are produced:

``label_from_patterns``
    1 if the transaction's match key is found in the patterns file, else 0.
    The patterns file is a subset of all illicit transactions – it records
    the seed transactions of each scheme but does NOT contain every illicit
    transaction present in the dataset.

``label_existing_csv``
    The original ``Is Laundering`` column from the transactions CSV.
    This is the **authoritative ground truth**: it was assigned by the IBM
    AML dataset generator and covers ALL illicit transactions, including
    layering-step transactions not listed in the patterns file.

``mismatch_flag``
    1 where ``label_existing_csv == 1`` but ``label_from_patterns == 0``,
    i.e. illicit transactions confirmed by the CSV that are absent from the
    patterns file (expected – these are the full chain beyond seed rows).

``label``
    Canonical binary target used for modelling (= ``label_existing_csv``).

``pattern_type``
    Laundering scheme type for matched transactions (e.g. ``"FAN-IN"``).
    ``"NONE"`` for transactions not found in the patterns file.

``pattern_block_id``
    Sequential block index from the patterns file (1-based).
    ``-1`` for transactions not found in the patterns file.
"""
from __future__ import annotations

import pandas as pd

from aml_benchmark.utils.hashing import make_match_key
from aml_benchmark.utils.logging_utils import get_logger

logger = get_logger(__name__)


def create_labels(
    transactions: pd.DataFrame,
    patterns: pd.DataFrame,
) -> pd.DataFrame:
    """Join pattern labels and metadata onto the transaction table.

    Parameters
    ----------
    transactions:
        Output of :func:`aml_benchmark.data.ingest.load_transactions`.
        Must contain ``is_laundering_csv`` and the eight key fields
        consumed by ``make_match_key``.
    patterns:
        Output of :func:`aml_benchmark.data.pattern_parser.parse_patterns`.
        Must contain ``pattern_type`` and ``pattern_block_id``.

    Returns
    -------
    A copy of *transactions* with six additional columns:
    ``label_from_patterns``, ``label_existing_csv``, ``mismatch_flag``,
    ``label``, ``pattern_type``, ``pattern_block_id``.
    Rows are sorted chronologically by ``timestamp``.
    """
    df = transactions.copy()
    pat = patterns.copy()

    # --- Build match keys -------------------------------------------------
    logger.info("Building match keys for transactions ...")
    df["_match_key"] = make_match_key(df)

    logger.info("Building match keys for patterns ...")
    pat["_match_key"] = make_match_key(pat)

    illicit_keys: set[str] = set(pat["_match_key"].dropna())
    trans_keys: set[str] = set(df["_match_key"].dropna())
    logger.info(f"Unique pattern match keys : {len(illicit_keys):,}")

    # --- Label assignment -------------------------------------------------
    df["label_from_patterns"] = df["_match_key"].isin(illicit_keys).astype(int)
    df = df.rename(columns={"is_laundering_csv": "label_existing_csv"})
    df["mismatch_flag"] = (
        df["label_from_patterns"] != df["label_existing_csv"]
    ).astype(int)
    # Authoritative label = CSV ground truth (more complete than patterns file)
    df["label"] = df["label_existing_csv"]

    # --- Attach pattern metadata ------------------------------------------
    logger.info("Attaching pattern metadata ...")
    _attach_pattern_metadata(df, pat)

    # --- Cleanup ----------------------------------------------------------
    df = df.drop(columns=["_match_key"])
    logger.info("Sorting by timestamp ...")
    df = df.sort_values("timestamp").reset_index(drop=True)

    # --- Audit log --------------------------------------------------------
    _log_summary(df, illicit_keys, trans_keys, n_pattern_rows=len(pat))

    return df


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _attach_pattern_metadata(df: pd.DataFrame, pat: pd.DataFrame) -> None:
    """Attach ``pattern_type`` and ``pattern_block_id`` to *df* in-place.

    Uses a dictionary lookup on the pre-computed ``_match_key`` column.
    In the rare case of duplicate keys in the patterns file the first
    occurrence wins.

    Unmatched rows receive:
        ``pattern_type``    = ``"NONE"``
        ``pattern_block_id`` = ``-1``
    """
    pat_dedup = pat.drop_duplicates(subset="_match_key", keep="first")
    key_to_type: dict[str, str] = (
        pat_dedup.set_index("_match_key")["pattern_type"].to_dict()
    )
    key_to_block_id: dict[str, int] = (
        pat_dedup.set_index("_match_key")["pattern_block_id"].to_dict()
    )

    df["pattern_type"] = df["_match_key"].map(key_to_type).fillna("NONE")
    df["pattern_block_id"] = (
        df["_match_key"].map(key_to_block_id).fillna(-1).astype(int)
    )


def _log_summary(
    df: pd.DataFrame,
    illicit_keys: set[str],
    trans_keys: set[str],
    n_pattern_rows: int,
) -> None:
    """Emit a structured labeling summary to the logger."""
    total = len(df)
    n_illicit_csv = int(df["label_existing_csv"].sum())
    n_illicit_patterns = int(df["label_from_patterns"].sum())
    n_mismatch = int(df["mismatch_flag"].sum())
    ratio = n_illicit_csv / total if total else 0.0
    n_unmatched_pattern_keys = len(illicit_keys - trans_keys)

    logger.info("=" * 62)
    logger.info("LABELING SUMMARY")
    logger.info(f"  Total transactions             : {total:>10,}")
    logger.info(f"  Illicit (label / csv)          : {n_illicit_csv:>10,}")
    logger.info(f"  Illicit (pattern-file subset)  : {n_illicit_patterns:>10,}")
    logger.info(f"  Illicit ratio                  : {ratio:>10.4%}")
    logger.info(f"  Pattern file rows              : {n_pattern_rows:>10,}")
    logger.info(f"  Unique pattern keys            : {len(illicit_keys):>10,}")
    logger.info(f"  Pattern keys unmatched in CSV  : {n_unmatched_pattern_keys:>10,}")
    logger.info(f"  CSV-illicit not in patterns    : {n_mismatch:>10,}")

    # Pattern metadata summary (only for pattern-matched transactions)
    matched = df[df["label_from_patterns"] == 1]
    if not matched.empty and "pattern_block_id" in df.columns:
        n_blocks = int(matched["pattern_block_id"].nunique())
        logger.info(f"  Unique laundering blocks       : {n_blocks:>10,}")
        logger.info("  Pattern type distribution (matched transactions):")
        type_dist = matched["pattern_type"].value_counts()
        for ptype, count in type_dist.items():
            logger.info(f"    {ptype:<28}: {count:>5,}")

    logger.info("=" * 62)
    logger.info(
        "NOTE: 'label' uses the CSV 'Is Laundering' column as ground truth. "
        "The patterns file is a subset (seed transactions only)."
    )

    if n_unmatched_pattern_keys > 0:
        logger.warning(
            f"{n_unmatched_pattern_keys:,} pattern keys could not be matched "
            "to any transaction."
        )
