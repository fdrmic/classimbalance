"""Account-level feature aggregation for AML benchmark.

Uses a vectorised pandas approach for performance on 176M rows.
All features are computed without temporal leakage.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from aml_benchmark.utils.logging_utils import get_logger

logger = get_logger(__name__)

SENDER_FEATURES = [
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

RECEIVER_FEATURES = [
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

ACCOUNT_FEATURE_NAMES = SENDER_FEATURES + RECEIVER_FEATURES


def load_entity_type_map(accounts_path: str) -> dict[str, int]:
    """Load accounts.csv and return account -> entity_type mapping.

    0 = Corporation, 1 = Partnership, 2 = Unknown
    """
    logger.info(f"Loading accounts from {accounts_path} ...")
    df = pd.read_csv(accounts_path)
    mapping = {}
    for _, row in df.iterrows():
        account = str(row.get("Account Number", "")).strip()
        entity_name = str(row.get("Entity Name", "")).strip()
        if "Corporation" in entity_name:
            mapping[account] = 0
        elif "Partnership" in entity_name:
            mapping[account] = 1
        else:
            mapping[account] = 2
    logger.info(f"Loaded entity types for {len(mapping):,} accounts.")
    return mapping


def _rolling_agg(
    df: pd.DataFrame,
    account_col: str,
    counterparty_col: str,
    cross_bank_col: str,
    amount_col: str,
    prefix: str,
) -> pd.DataFrame:
    """Compute rolling aggregations for one account role (sender or receiver).

    Uses merge_asof for leakage-free lookups.
    """
    logger.info(f"Computing {prefix} rolling features ...")

    df_sorted = df[["timestamp", account_col, counterparty_col,
                     cross_bank_col, amount_col]].copy()
    df_sorted = df_sorted.sort_values(["timestamp"]).reset_index(drop=True)

    results = {}

    for window, label in [(1, "1d"), (7, "7d"), (30, "30d")]:
        td = pd.Timedelta(days=window)
        logger.info(f"  {prefix} window={label} ...")

        grp = df_sorted.groupby(account_col, sort=False)

        tx_counts = []

        for acct, group in grp:
            group = group.sort_values("timestamp")
            ts = group["timestamp"].values
            amt = group[amount_col].values
            cp = group[counterparty_col].values
            cross = group[cross_bank_col].values

            n = len(group)
            cnt = np.zeros(n, dtype=np.int32)
            avg_amt = np.zeros(n, dtype=np.float32)
            uniq_cp = np.zeros(n, dtype=np.int32)
            cross_r = np.zeros(n, dtype=np.float32)

            left = 0
            for i in range(n):
                cutoff = ts[i] - td.value
                while left < i and ts[left] < cutoff:
                    left += 1
                window_slice = slice(left, i)
                c = i - left
                cnt[i] = c
                if c > 0:
                    avg_amt[i] = amt[window_slice].mean()
                    uniq_cp[i] = len(set(cp[window_slice]))
                    cross_r[i] = cross[window_slice].mean()

            group = group.copy()
            group[f"_cnt_{label}"] = cnt
            group[f"_amt_{label}"] = avg_amt
            group[f"_ucp_{label}"] = uniq_cp
            group[f"_crs_{label}"] = cross_r
            tx_counts.append(group)

        merged = pd.concat(tx_counts).sort_index()
        results[f"{prefix}_tx_count_{label}"] = merged[f"_cnt_{label}"].values
        results[f"{prefix}_avg_amount_{label}"] = merged[f"_amt_{label}"].values
        if label in ("7d", "30d"):
            results[f"{prefix}_unique_counterparties_{label}"] = merged[f"_ucp_{label}"].values
        if label == "30d":
            results[f"{prefix}_cross_bank_ratio_{label}"] = merged[f"_crs_{label}"].values

    return pd.DataFrame(results, index=df_sorted.index)


def compute_account_features(
    df: pd.DataFrame,
    entity_type_map: dict[str, int],
) -> pd.DataFrame:
    """Compute all account-level features for a transaction DataFrame.

    Parameters
    ----------
    df:
        Transaction DataFrame sorted by timestamp with columns:
        timestamp, from_account, to_account, from_bank, to_bank, amount_paid
    entity_type_map:
        Mapping from account number to entity type integer.

    Returns
    -------
    DataFrame with ACCOUNT_FEATURE_NAMES columns, same length as df.
    """
    logger.info(f"Computing account features for {len(df):,} rows ...")

    df = df.copy()
    df["_cross_bank"] = (df["from_bank"] != df["to_bank"]).astype(np.float32)

    # Sender features
    sender_df = _rolling_agg(
        df=df,
        account_col="from_account",
        counterparty_col="to_account",
        cross_bank_col="_cross_bank",
        amount_col="amount_paid",
        prefix="sender",
    )

    # Receiver features
    receiver_df = _rolling_agg(
        df=df,
        account_col="to_account",
        counterparty_col="from_account",
        cross_bank_col="_cross_bank",
        amount_col="amount_paid",
        prefix="receiver",
    )

    # Entity types
    entity_sender = df["from_account"].map(
        lambda x: entity_type_map.get(str(x), 2)
    ).values.astype(np.float32)
    entity_receiver = df["to_account"].map(
        lambda x: entity_type_map.get(str(x), 2)
    ).values.astype(np.float32)

    # Assemble
    out = pd.concat([sender_df, receiver_df], axis=1)
    out["sender_entity_type"] = entity_sender
    out["receiver_entity_type"] = entity_receiver

    # Ensure correct column order
    out = out[ACCOUNT_FEATURE_NAMES]

    logger.info("Account features done.")
    return out
