"""Aggregate feature importances from all completed benchmark runs.

Each run folder under ``outputs_dir`` (see ``configs/paths.yaml``) may contain
``model.pkl`` and ``run_config.json``.  This script loads fitted Random Forest
or XGBoost models, reads ``feature_importances_``, aligns values with
``feature_names`` from the run config, and writes summary CSVs under
``outputs/feature_importance/``.

Usage
-----
    python analysis/feature_importance.py
    python analysis/feature_importance.py --paths configs/paths_large_v2.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from aml_benchmark.config import PathConfig
from aml_benchmark.utils.logging_utils import get_logger

logger = get_logger(__name__)


def _load_run_records(outputs_dir: Path) -> list[dict]:
    """Load one flat record list: run metadata + per-feature importance rows."""
    if not outputs_dir.is_dir():
        logger.warning(f"Outputs directory does not exist: {outputs_dir}")
        return []

    rows: list[dict] = []
    for run_dir in sorted(outputs_dir.iterdir()):
        if not run_dir.is_dir():
            continue

        model_path = run_dir / "model.pkl"
        cfg_path = run_dir / "run_config.json"
        if not model_path.exists() or not cfg_path.exists():
            logger.debug(f"Skip {run_dir.name}: missing model.pkl or run_config.json")
            continue

        try:
            with cfg_path.open(encoding="utf-8") as fh:
                cfg = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(f"Skip {run_dir.name}: could not read run_config.json ({exc})")
            continue

        feature_names = cfg.get("feature_names")
        if not isinstance(feature_names, list) or not feature_names:
            logger.warning(f"Skip {run_dir.name}: feature_names missing or invalid")
            continue

        try:
            model = joblib.load(model_path)
        except Exception as exc:
            logger.warning(f"Skip {run_dir.name}: failed to load model.pkl ({exc})")
            continue

        importances = getattr(model, "feature_importances_", None)
        if importances is None:
            logger.warning(f"Skip {run_dir.name}: model has no feature_importances_")
            continue

        imp = np.asarray(importances, dtype=np.float64).ravel()
        names = [str(n) for n in feature_names]
        if len(imp) != len(names):
            logger.warning(
                f"Skip {run_dir.name}: len(feature_importances_)={len(imp)} "
                f"!= len(feature_names)={len(names)}"
            )
            continue

        run_id = str(cfg.get("run_id", run_dir.name))
        model_name = str(cfg.get("model", ""))
        strategy = str(cfg.get("strategy", ""))
        prev = cfg.get("target_prevalence")
        try:
            target_prevalence = float(prev) if prev is not None else float("nan")
        except (TypeError, ValueError):
            target_prevalence = float("nan")

        for feat, importance in zip(names, imp):
            rows.append(
                {
                    "run_id": run_id,
                    "model": model_name,
                    "strategy": strategy,
                    "target_prevalence": target_prevalence,
                    "feature": feat,
                    "importance": float(importance),
                }
            )

        logger.info(f"Loaded importances: {run_dir.name} ({model_name})")

    return rows


def _mean_by_feature(df: pd.DataFrame, model_filter: str) -> pd.DataFrame:
    sub = df[df["model"] == model_filter].copy()
    if sub.empty:
        return pd.DataFrame(columns=["feature", "importance", "n_runs"])
    g = (
        sub.groupby("feature", as_index=False)
        .agg(importance=("importance", "mean"), n_runs=("run_id", "nunique"))
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    return g


def _per_strategy_xgboost(df: pd.DataFrame) -> pd.DataFrame:
    sub = df[df["model"] == "xgboost"].copy()
    if sub.empty:
        return pd.DataFrame(columns=["strategy", "feature", "importance", "n_runs"])
    g = (
        sub.groupby(["strategy", "feature"], as_index=False)
        .agg(importance=("importance", "mean"), n_runs=("run_id", "nunique"))
        .sort_values(["strategy", "importance"], ascending=[True, False])
        .reset_index(drop=True)
    )
    return g


def _print_top_n(mean_df: pd.DataFrame, title: str, n: int = 15) -> None:
    print()
    print("=" * 62)
    print(f"  {title}")
    print("=" * 62)
    if mean_df.empty:
        print("  (no runs)")
        print("=" * 62)
        return
    top = mean_df.head(n)
    for i, row in enumerate(top.itertuples(index=False), start=1):
        feat = row.feature
        imp = row.importance
        print(f"  {i:2d}. {feat:<32}  {imp:.6f}")
    print("=" * 62)
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract and aggregate feature importances from saved benchmark runs."
    )
    parser.add_argument(
        "--paths",
        type=str,
        default=None,
        help="Path to a custom paths.yaml (e.g. for Colab or large-run configs)",
    )
    args = parser.parse_args()

    paths = PathConfig(args.paths) if args.paths else PathConfig()
    out_dir = paths.outputs_dir.parent / "feature_importance"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Scanning runs under: {paths.outputs_dir}")
    logger.info(f"Writing CSVs under: {out_dir}")

    records = _load_run_records(paths.outputs_dir)
    if not records:
        logger.warning("No valid runs found; writing empty CSVs with headers.")
        df_all = pd.DataFrame(
            columns=["run_id", "model", "strategy", "target_prevalence", "feature", "importance"],
        )
    else:
        df_all = pd.DataFrame(records)
    path_all = out_dir / "feature_importance_all_runs.csv"
    df_all.to_csv(path_all, index=False)
    logger.info(f"Saved -> {path_all}")

    xgb_mean = _mean_by_feature(df_all, "xgboost")
    path_xgb = out_dir / "feature_importance_xgboost_mean.csv"
    xgb_mean.to_csv(path_xgb, index=False)
    logger.info(f"Saved -> {path_xgb}")

    rf_mean = _mean_by_feature(df_all, "random_forest")
    path_rf = out_dir / "feature_importance_rf_mean.csv"
    rf_mean.to_csv(path_rf, index=False)
    logger.info(f"Saved -> {path_rf}")

    per_strat = _per_strategy_xgboost(df_all)
    path_ps = out_dir / "feature_importance_per_strategy.csv"
    per_strat.to_csv(path_ps, index=False)
    logger.info(f"Saved -> {path_ps}")

    _print_top_n(xgb_mean, "Top 15 features — XGBoost (mean importance across runs)")
    _print_top_n(rf_mean, "Top 15 features — Random Forest (mean importance across runs)")


if __name__ == "__main__":
    main()
