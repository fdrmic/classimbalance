"""Account-level feature aggregation for AML benchmark.

Vectorised implementation using numpy.searchsorted + prefix sums.
All features are computed without temporal leakage.

Performance
-----------
The key bottleneck of a naive row-by-row approach (Python while-loop) is
replaced by three techniques:

1. ``numpy.searchsorted`` — finds the left window boundary for every row of
   an account group in one vectorised call instead of an inner Python loop.
2. Prefix sums — compute ``avg_amount`` and ``cross_bank_ratio`` in O(k)
   per account group (no per-row loops for numerical aggregations).
3. Pre-allocated output arrays — avoids ``pd.concat`` of thousands of small
   DataFrames (which is O(N^2) in allocations).

The only remaining Python-level inner loop is for ``unique_counterparties``,
which requires set operations (inherently hard to vectorise exactly).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from aml_benchmark.utils.logging_utils import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Feature name constants
# ---------------------------------------------------------------------------

SENDER_FEATURES: list[str] = [
    "sender_tx_count_1d",
    "sender_tx_count_7d",
    "sender_tx_count_30d",
    "sender_avg_amount_7d",
    "sender_avg_amount_30d",
    "sender_unique_counterparties_7d",
    "sender_unique_counterparties_30d",
    "sender_cross_bank_ratio_30d",
    "sender_entity_type",
]

RECEIVER_FEATURES: list[str] = [
    "receiver_tx_count_1d",
    "receiver_tx_count_7d",
    "receiver_tx_count_30d",
    "receiver_avg_amount_7d",
    "receiver_avg_amount_30d",
    "receiver_unique_counterparties_7d",
    "receiver_unique_counterparties_30d",
    "receiver_cross_bank_ratio_30d",
    "receiver_entity_type",
]

ACCOUNT_FEATURE_NAMES: list[str] = SENDER_FEATURES + RECEIVER_FEATURES

# Nanosecond constants for window comparisons
_NS_1D  = int(1  * 24 * 3600 * 1e9)
_NS_7D  = int(7  * 24 * 3600 * 1e9)
_NS_30D = int(30 * 24 * 3600 * 1e9)


# ---------------------------------------------------------------------------
# Entity-type mapping
# ---------------------------------------------------------------------------

def load_entity_type_map(accounts_path: str) -> dict[str, int]:
    """Load accounts CSV and return account -> entity_type mapping.

    Entity type encoding
    --------------------
    0 = Corporation
    1 = Partnership
    2 = Unknown / other

    Parameters
    ----------
    accounts_path:
        Path to the accounts CSV (``LI-Small_accounts.csv`` or
        ``LI-Large_accounts.csv``).
    """
    logger.info(f"Loading accounts from {accounts_path} ...")
    df = pd.read_csv(accounts_path, dtype=str, low_memory=False)
    df.columns = [c.strip() for c in df.columns]

    acct_col = next(
        (c for c in df.columns if "account" in c.lower() and "number" in c.lower()),
        next((c for c in df.columns if c.lower() == "account"), None),
    )
    entity_col = next(
        (c for c in df.columns if "entity" in c.lower() and "name" in c.lower()), None
    )

    if acct_col is None:
        logger.warning("No account-number column found; returning empty map.")
        return {}

    mapping: dict[str, int] = {}
    if entity_col is None:
        logger.warning("No 'Entity Name' column found; mapping all to 2 (Unknown).")
        for acct in df[acct_col].dropna():
            mapping[str(acct).strip()] = 2
    else:
        for acct, name in zip(df[acct_col], df[entity_col]):
            key = str(acct).strip()
            val = str(name).strip() if pd.notna(name) else ""
            if "Corporation" in val:
                mapping[key] = 0
            elif "Partnership" in val:
                mapping[key] = 1
            else:
                mapping[key] = 2

    logger.info(f"Loaded entity types for {len(mapping):,} accounts.")
    return mapping


# ---------------------------------------------------------------------------
# Vectorised rolling helper
# ---------------------------------------------------------------------------

def _rolling_agg(
    df: pd.DataFrame,
    account_col: str,
    counterparty_col: str,
    cross_bank_col: str,
    amount_col: str,
    prefix: str,
) -> pd.DataFrame:
    """Compute rolling aggregations for one account role (sender or receiver).

    Optimisations vs. naive approach
    ---------------------------------
    * ``np.searchsorted`` replaces the inner ``while left < i`` Python loop.
    * Prefix sums replace per-row numerical accumulations.
    * Results written directly into pre-allocated numpy arrays (no concat).

    Parameters
    ----------
    df:
        Transaction DataFrame with a ``timestamp`` column.
    account_col, counterparty_col, cross_bank_col, amount_col:
        Column names for the relevant role.
    prefix:
        ``"sender"`` or ``"receiver"`` — used for output column names.

    Returns
    -------
    DataFrame (same length as *df*, original row order) with 8 feature
    columns (tx_count ×3, avg_amount ×2, unique_counterparties ×2,
    cross_bank_ratio ×1).
    """
    logger.info(f"Computing {prefix} rolling features ...")

    # ── Sort by account + timestamp ────────────────────────────────────
    sort_keys = np.lexsort([
        df["timestamp"].values.astype("datetime64[ns]").astype(np.int64),
        df[account_col].values,
    ])
    inv_sort = np.argsort(sort_keys)   # maps sorted_pos -> original_pos

    accounts    = df[account_col].values[sort_keys]
    ts_ns       = df["timestamp"].values.astype("datetime64[ns]").astype(np.int64)[sort_keys]
    amounts     = df[amount_col].values.astype(np.float64)[sort_keys]
    counterparts = df[counterparty_col].values[sort_keys]
    cross_arr   = df[cross_bank_col].values.astype(np.float64)[sort_keys]

    n = len(accounts)

    # ── Pre-allocate output arrays ─────────────────────────────────────
    cnt_1d  = np.zeros(n, dtype=np.float32)
    cnt_7d  = np.zeros(n, dtype=np.float32)
    cnt_30d = np.zeros(n, dtype=np.float32)
    amt_7d  = np.zeros(n, dtype=np.float32)
    amt_30d = np.zeros(n, dtype=np.float32)
    ucp_7d  = np.zeros(n, dtype=np.float32)
    ucp_30d = np.zeros(n, dtype=np.float32)
    crs_30d = np.zeros(n, dtype=np.float32)

    # ── Account group boundaries ───────────────────────────────────────
    # accounts is sorted, so boundaries are where the value changes
    boundaries = np.concatenate(
        [[0], np.where(accounts[:-1] != accounts[1:])[0] + 1, [n]]
    )
    n_accounts = len(boundaries) - 1

    log_every = max(1, n_accounts // 10)

    # ── Main loop: one iteration per unique account ────────────────────
    for gi in range(n_accounts):
        if gi % log_every == 0:
            logger.info(f"  {prefix}: {gi:,}/{n_accounts:,} accounts ...")

        s = boundaries[gi]
        e = boundaries[gi + 1]
        k = e - s
        if k == 0:
            continue

        ts_g   = ts_ns[s:e]        # already sorted within group
        amt_g  = amounts[s:e]
        cp_g   = counterparts[s:e]
        crs_g  = cross_arr[s:e]

        # Prefix sums for O(1) range queries
        cs_amt = np.empty(k + 1, dtype=np.float64)
        cs_amt[0] = 0.0
        np.cumsum(amt_g, out=cs_amt[1:])

        cs_crs = np.empty(k + 1, dtype=np.float64)
        cs_crs[0] = 0.0
        np.cumsum(crs_g, out=cs_crs[1:])

        pos = np.arange(k, dtype=np.int64)  # positions 0…k-1 within group

        # ── Window 1d ─────────────────────────────────────────────────
        lefts_1d = np.searchsorted(ts_g, ts_g - _NS_1D, side="left")
        c1 = pos - lefts_1d            # tx_count (= #past txns in window)
        cnt_1d[s:e] = c1

        # ── Window 7d ─────────────────────────────────────────────────
        lefts_7d = np.searchsorted(ts_g, ts_g - _NS_7D, side="left")
        c7 = pos - lefts_7d
        cnt_7d[s:e] = c7

        mask7 = c7 > 0
        if mask7.any():
            sums7  = cs_amt[pos[mask7]] - cs_amt[lefts_7d[mask7]]
            amt_7d[s + np.where(mask7)[0]] = (sums7 / c7[mask7]).astype(np.float32)

        # unique_counterparties_7d — Python inner loop (unavoidable for sets)
        for i in range(k):
            if c7[i] > 0:
                ucp_7d[s + i] = len(set(cp_g[lefts_7d[i]:i]))

        # ── Window 30d ────────────────────────────────────────────────
        lefts_30d = np.searchsorted(ts_g, ts_g - _NS_30D, side="left")
        c30 = pos - lefts_30d
        cnt_30d[s:e] = c30

        mask30 = c30 > 0
        if mask30.any():
            idx30 = np.where(mask30)[0]
            sums30  = cs_amt[pos[mask30]] - cs_amt[lefts_30d[mask30]]
            sums_crs = cs_crs[pos[mask30]] - cs_crs[lefts_30d[mask30]]
            amt_30d[s + idx30] = (sums30  / c30[mask30]).astype(np.float32)
            crs_30d[s + idx30] = (sums_crs / c30[mask30]).astype(np.float32)

        # unique_counterparties_30d
        for i in range(k):
            if c30[i] > 0:
                ucp_30d[s + i] = len(set(cp_g[lefts_30d[i]:i]))

    logger.info(f"  {prefix}: done.")

    # ── Re-index to original row order ────────────────────────────────
    feat = pd.DataFrame({
        f"{prefix}_tx_count_1d":               cnt_1d[inv_sort],
        f"{prefix}_tx_count_7d":               cnt_7d[inv_sort],
        f"{prefix}_tx_count_30d":              cnt_30d[inv_sort],
        f"{prefix}_avg_amount_7d":             amt_7d[inv_sort],
        f"{prefix}_avg_amount_30d":            amt_30d[inv_sort],
        f"{prefix}_unique_counterparties_7d":  ucp_7d[inv_sort],
        f"{prefix}_unique_counterparties_30d": ucp_30d[inv_sort],
        f"{prefix}_cross_bank_ratio_30d":      crs_30d[inv_sort],
    })
    return feat


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_account_features(
    df: pd.DataFrame,
    entity_type_map: dict[str, int],
) -> pd.DataFrame:
    """Compute all 18 account-level features for a transaction DataFrame.

    Parameters
    ----------
    df:
        Transaction DataFrame with columns:
        ``timestamp``, ``from_account``, ``to_account``,
        ``from_bank``, ``to_bank``, ``amount_paid``.
    entity_type_map:
        Account -> entity_type mapping from :func:`load_entity_type_map`.

    Returns
    -------
    DataFrame with ``ACCOUNT_FEATURE_NAMES`` columns (18 features),
    same length and index as *df*.
    """
    logger.info(f"Computing account features for {len(df):,} rows ...")

    # Cross-bank indicator (shared by sender and receiver perspectives)
    cross = (df["from_bank"] != df["to_bank"]).astype(np.float32)
    df = df.copy()
    df["_cross_bank"] = cross

    # ── Sender features ────────────────────────────────────────────────
    sender_df = _rolling_agg(
        df=df,
        account_col="from_account",
        counterparty_col="to_account",
        cross_bank_col="_cross_bank",
        amount_col="amount_paid",
        prefix="sender",
    )

    # ── Receiver features ──────────────────────────────────────────────
    receiver_df = _rolling_agg(
        df=df,
        account_col="to_account",
        counterparty_col="from_account",
        cross_bank_col="_cross_bank",
        amount_col="amount_paid",
        prefix="receiver",
    )

    # ── Entity types (static lookup) ───────────────────────────────────
    sender_df["sender_entity_type"] = (
        df["from_account"].map(lambda x: entity_type_map.get(str(x), 2))
        .values.astype(np.float32)
    )
    receiver_df["receiver_entity_type"] = (
        df["to_account"].map(lambda x: entity_type_map.get(str(x), 2))
        .values.astype(np.float32)
    )

    out = pd.concat([sender_df, receiver_df], axis=1)[ACCOUNT_FEATURE_NAMES]

    logger.info("Account features done.")
    return out
