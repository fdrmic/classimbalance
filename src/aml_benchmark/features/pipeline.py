"""Leakage-safe MVP feature pipeline for the AML benchmark.

Feature set
-----------
Numeric (2)
    ``amount_paid``, ``amount_received``
    Applied log1p transform for scale stability (trees benefit modestly;
    critical for future linear or neural models).

Categorical (2) – OrdinalEncoder fit on training data only
    ``payment_format``, ``payment_currency``
    Unknown categories at inference time map to ``-1``.

Temporal (2) – derived from timestamp, no fitting required
    ``hour``        – hour of day  (0–23)
    ``day_of_week`` – day of week  (0 = Monday … 6 = Sunday)

Boolean (2) – derived from raw columns, no fitting required
    ``same_bank_flag``     – 1 if from_bank == to_bank
    ``self_transfer_flag`` – 1 if from_account == to_account

Design constraints
------------------
* ``fit_transform`` must be called on training data ONLY.
* ``transform`` applies the same encodings to val / test without refitting.
* The fitted pipeline can be serialised with ``joblib.dump`` for
  reproducibility across runs.

Usage
-----
    pipeline = FeaturePipeline()
    X_train = pipeline.fit_transform(train_df)
    X_val   = pipeline.transform(val_df)
    X_test  = pipeline.transform(test_df)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

from aml_benchmark.features.aggregator import (
    ACCOUNT_FEATURE_NAMES,
    compute_account_features,
    load_entity_type_map,
)
from aml_benchmark.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Column groups
_NUMERIC: list[str] = ["amount_paid", "amount_received"]
_CATEGORICAL: list[str] = ["payment_format", "payment_currency"]
_DERIVED: list[str] = [
    "hour",
    "day_of_week",
    "same_bank_flag",
    "self_transfer_flag",
    "currency_mismatch",
    "amount_ratio",
    "fan_in_score",
    "fan_out_score",
]

FEATURE_NAMES: list[str] = _NUMERIC + _CATEGORICAL + _DERIVED + ACCOUNT_FEATURE_NAMES


class FeaturePipeline:
    """Stateful feature transformer for AML transaction data.

    Attributes
    ----------
    feature_names:
        Ordered list of feature column names matching the output array.
    encoder:
        Fitted ``OrdinalEncoder`` for categorical columns.
    """

    def __init__(self, accounts_path: str | None = None) -> None:
        self.encoder: OrdinalEncoder = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=np.nan,
            dtype=np.float64,
        )
        self.feature_names: list[str] = list(FEATURE_NAMES)
        self._fitted: bool = False
        self._entity_type_map: dict[str, int] = (
            load_entity_type_map(accounts_path) if accounts_path else {}
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """Fit on *df* and return the transformed feature matrix.

        Must be called on training data only.

        Parameters
        ----------
        df:
            Labeled transaction split with at minimum the columns required
            by ``_derive`` and the raw numeric / categorical fields.

        Returns
        -------
        2-D float array of shape ``(n_rows, n_features)``.
        """
        logger.info(f"FeaturePipeline: deriving features for {len(df):,} rows ...")
        derived = self._derive(df)
        logger.info("FeaturePipeline: fitting encoder ...")
        self.encoder.fit(derived[_CATEGORICAL])
        self._fitted = True
        logger.info(
            f"FeaturePipeline fitted on {len(df):,} rows | "
            f"{len(self.feature_names)} features: {self.feature_names}"
        )
        logger.info("FeaturePipeline: computing account-level features ...")
        account_feats = compute_account_features(derived, self._entity_type_map)
        self._fill_fan_scores(derived, account_feats)
        logger.info("FeaturePipeline: assembling feature matrix ...")
        return self._assemble(derived, account_feats)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Apply the fitted pipeline to *df*.

        Parameters
        ----------
        df:
            Validation or test split.

        Returns
        -------
        2-D float array of shape ``(n_rows, n_features)``.
        """
        if not self._fitted:
            raise RuntimeError(
                "FeaturePipeline has not been fitted yet. "
                "Call fit_transform(train_df) first."
            )
        logger.info(f"FeaturePipeline: transforming {len(df):,} rows ...")
        derived = self._derive(df)
        logger.info("FeaturePipeline: computing account-level features ...")
        account_feats = compute_account_features(derived, self._entity_type_map)
        self._fill_fan_scores(derived, account_feats)
        return self._assemble(derived, account_feats)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _derive(df: pd.DataFrame) -> pd.DataFrame:
        """Return a copy of *df* with derived feature columns added."""
        logger.info(f"Deriving features for {len(df):,} rows ...")
        d = df.copy()
        logger.info("Copy done, computing features ...")

        # Temporal
        d["hour"] = d["timestamp"].dt.hour.astype(np.float64)
        d["day_of_week"] = d["timestamp"].dt.dayofweek.astype(np.float64)
        logger.info("Temporal features done ...")

        # Boolean flags
        d["same_bank_flag"] = (d["from_bank"] == d["to_bank"]).astype(np.float64)
        d["self_transfer_flag"] = (
            d["from_account"] == d["to_account"]
        ).astype(np.float64)
        logger.info("Boolean flags done ...")

        # Currency mismatch — different payment vs receiving currency
        d["currency_mismatch"] = (
            d["payment_currency"] != d["receiving_currency"]
        ).astype(np.float64)

        # Amount ratio — received / paid using RAW amounts (before log1p)
        # Clip to [0, 10] to suppress extreme FX-conversion outliers
        raw_paid = d["amount_paid"].astype(np.float64)
        raw_recv = d["amount_received"].astype(np.float64)
        d["amount_ratio"] = np.clip(
            np.where(raw_paid > 0, raw_recv / raw_paid, 1.0),
            0.0, 10.0,
        )

        # Fan-in / fan-out placeholders — filled after account features are
        # computed in fit_transform() / transform()
        d["fan_in_score"]  = 0.0
        d["fan_out_score"] = 0.0
        logger.info("Ratio / fan features initialised ...")

        # Log1p on amounts (preserves 0-handling, reduces skew)
        d["amount_paid"]     = np.log1p(raw_paid)
        d["amount_received"] = np.log1p(raw_recv)
        logger.info("Amount transforms done ...")

        return d

    @staticmethod
    def _fill_fan_scores(
        derived: pd.DataFrame,
        account_feats: pd.DataFrame,
    ) -> None:
        """Write fan-in and fan-out scores into *derived* in-place.

        Both scores are computed from the 7-day account-level rolling counts
        and therefore can only be set after :func:`compute_account_features`
        has run.

        Fan-in score
            ``receiver_tx_count_7d / (sender_tx_count_7d + eps)``
            High values indicate the account receives far more transactions
            than it sends — characteristic of collector accounts in Fan-In
            laundering patterns.

        Fan-out score
            ``sender_unique_counterparties_7d /
             (receiver_unique_counterparties_7d + eps)``
            High values indicate the account disperses funds to many unique
            recipients while receiving from few — characteristic of disperser
            accounts in Fan-Out / Scatter patterns.
        """
        eps = 1e-6

        fan_in = np.where(
            account_feats["sender_tx_count_7d"].values > 0,
            account_feats["receiver_tx_count_7d"].values
            / (account_feats["sender_tx_count_7d"].values + eps),
            0.0,
        )
        derived["fan_in_score"] = np.clip(fan_in, 0.0, 100.0).astype(np.float64)

        fan_out = np.where(
            account_feats["receiver_unique_counterparties_7d"].values > 0,
            account_feats["sender_unique_counterparties_7d"].values
            / (account_feats["receiver_unique_counterparties_7d"].values + eps),
            0.0,
        )
        derived["fan_out_score"] = np.clip(fan_out, 0.0, 100.0).astype(np.float64)

    def _assemble(self, d: pd.DataFrame, account_feats: pd.DataFrame) -> np.ndarray:
        """Stack numeric, encoded categorical, derived, and account-level arrays."""
        X_numeric     = d[_NUMERIC].to_numpy(dtype=np.float64)
        X_categorical = self.encoder.transform(d[_CATEGORICAL])
        X_derived     = d[_DERIVED].to_numpy(dtype=np.float64)
        X_account     = account_feats[ACCOUNT_FEATURE_NAMES].to_numpy(dtype=np.float64)
        return np.hstack([X_numeric, X_categorical, X_derived, X_account])
