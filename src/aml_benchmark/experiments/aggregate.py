"""Part A result aggregator.

Scans all run folders under ``outputs/runs/``, collects the per-run
metrics and configuration, and writes a single leaderboard CSV to
``outputs/leaderboard/part_a_summary.csv``.

Only runs that contain ``run_config.json`` with both a ``strategy`` and
a ``target_prevalence`` field are included (i.e. Part A runs created by
the updated :func:`aml_benchmark.experiments.runner.run_experiment`).
Older smoke-test runs without those fields are silently skipped.

Two metric sets are collected per run
--------------------------------------
``*_val`` / ``*_test``
    Metrics at the default threshold of 0.5 (as produced by the main runner).
    These are always present.

``*_val_thresh`` / ``*_test_thresh``
    Metrics at the validation-optimal threshold (produced by
    :mod:`aml_benchmark.experiments.re_evaluate`).  Only present after
    ``re_evaluate.py`` has been run.  Columns are left as ``None`` when the
    files do not exist.

Output columns
--------------
    run_id, model, strategy, target_prevalence, achieved_train_prevalence
    train_rows_after_sampling, train_positives_after_sampling, n_synthetic_samples
    val_rows, val_positives, test_rows, test_positives
    --- default threshold (0.5) ---
    pr_auc_val, roc_auc_val, precision_val, recall_val, f1_val, f2_val,
    weighted_accuracy_val, tp_val, fp_val, tn_val, fn_val
    pr_auc_test, roc_auc_test, precision_test, recall_test, f1_test, f2_test,
    weighted_accuracy_test, tp_test, fp_test, tn_test, fn_test
    --- optimal threshold ---
    optimal_threshold, threshold_criterion
    precision_val_thresh, recall_val_thresh, f1_val_thresh, f2_val_thresh,
    tp_val_thresh, fp_val_thresh, tn_val_thresh, fn_val_thresh
    precision_test_thresh, recall_test_thresh, f1_test_thresh, f2_test_thresh,
    weighted_accuracy_test_thresh,
    tp_test_thresh, fp_test_thresh, tn_test_thresh, fn_test_thresh
    --- run metadata ---
    train_time_sec, created_at

The table is sorted by ``pr_auc_test`` DESC, then ``recall_test_thresh`` DESC
(falling back to ``recall_test`` when thresh metrics are absent).

Usage
-----
    python -m aml_benchmark.experiments.aggregate
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

from aml_benchmark.config import PathConfig
from aml_benchmark.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Metrics to extract from the default-threshold splits
_METRIC_KEYS = (
    "pr_auc",
    "roc_auc",
    "precision",
    "recall",
    "f1",
    "f2",
    "weighted_accuracy",
    "tp",
    "fp",
    "tn",
    "fn",
)

# Metrics to extract from the optimal-threshold splits
# (pr_auc and roc_auc are threshold-independent, kept only in the default set)
_THRESH_METRIC_KEYS = (
    "precision",
    "recall",
    "f1",
    "f2",
    "weighted_accuracy",
    "tp",
    "fp",
    "tn",
    "fn",
)

# run_config.json fields to include as metadata columns
_CONFIG_KEYS = (
    "run_id",
    "model",
    "strategy",
    "target_prevalence",
    "achieved_train_prevalence",
    "train_rows_after_sampling",
    "train_positives_after_sampling",
    "n_synthetic_samples",
    "val_rows",
    "val_positives",
    "test_rows",
    "test_positives",
    "train_time_sec",
    "created_at",
)


def aggregate_part_a(paths: PathConfig | None = None) -> pd.DataFrame:
    """Collect all Part A run results and return a summary DataFrame.

    Parameters
    ----------
    paths:
        Optional pre-built :class:`~aml_benchmark.config.PathConfig`.

    Returns
    -------
    DataFrame with one row per Part A run, sorted by ``pr_auc_test``
    descending and ``recall_test_thresh`` (or ``recall_test``) descending.
    """
    if paths is None:
        paths = PathConfig()

    runs_dir = paths.outputs_dir
    if not runs_dir.exists():
        raise FileNotFoundError(
            f"Runs directory not found: {runs_dir}\n"
            "Run the benchmark first with:\n"
            "  python -m aml_benchmark.experiments.grid_runner"
        )

    rows: list[dict] = []

    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue

        config_path = run_dir / "run_config.json"
        if not config_path.exists():
            logger.debug(f"Skipping {run_dir.name}: no run_config.json")
            continue

        with config_path.open(encoding="utf-8") as fh:
            cfg = json.load(fh)

        # Only include Part A runs
        if "strategy" not in cfg or "target_prevalence" not in cfg:
            logger.debug(f"Skipping {run_dir.name}: not a Part A run")
            continue

        row: dict = {}

        # Config metadata
        for key in _CONFIG_KEYS:
            row[key] = cfg.get(key)

        # --- Default-threshold metrics (val + test) ---
        for split in ("val", "test"):
            metrics_path = run_dir / f"metrics_{split}.json"
            if not metrics_path.exists():
                logger.warning(f"{run_dir.name}: missing metrics_{split}.json")
                for key in _METRIC_KEYS:
                    row[f"{key}_{split}"] = None
                continue

            with metrics_path.open(encoding="utf-8") as fh:
                m = json.load(fh)

            for key in _METRIC_KEYS:
                row[f"{key}_{split}"] = m.get(key)

        # --- Optimal-threshold metadata ---
        thresh_info_path = run_dir / "threshold_info.json"
        if thresh_info_path.exists():
            with thresh_info_path.open(encoding="utf-8") as fh:
                ti = json.load(fh)
            row["optimal_threshold"] = ti.get("optimal_threshold")
            row["threshold_criterion"] = ti.get("criterion")
        else:
            row["optimal_threshold"] = None
            row["threshold_criterion"] = None

        # --- Optimal-threshold metrics (val_thresh + test_thresh) ---
        for split_tag in ("val_thresh", "test_thresh"):
            metrics_path = run_dir / f"metrics_{split_tag}.json"
            if not metrics_path.exists():
                for key in _THRESH_METRIC_KEYS:
                    row[f"{key}_{split_tag}"] = None
                continue

            with metrics_path.open(encoding="utf-8") as fh:
                m = json.load(fh)

            for key in _THRESH_METRIC_KEYS:
                row[f"{key}_{split_tag}"] = m.get(key)

        rows.append(row)

    if not rows:
        logger.warning("No Part A runs found. Is the grid finished?")
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Sort: pr_auc_test DESC, then best available recall DESC
    recall_col = (
        "recall_test_thresh"
        if "recall_test_thresh" in df.columns and df["recall_test_thresh"].notna().any()
        else "recall_test"
    )
    sort_cols = [c for c in ("pr_auc_test", recall_col) if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols, ascending=False).reset_index(drop=True)

    logger.info(f"Aggregated {len(df)} Part A runs.")
    return df


def save_leaderboard(
    df: pd.DataFrame,
    paths: PathConfig | None = None,
) -> Path:
    """Save the Part A leaderboard to CSV and return the output path."""
    if paths is None:
        paths = PathConfig()

    paths.leaderboard_dir.mkdir(parents=True, exist_ok=True)
    out = paths.part_a_summary
    df.to_csv(out, index=False)
    logger.info(f"Leaderboard saved -> {out}  ({len(df)} rows)")
    return out


def _print_leaderboard(df: pd.DataFrame, top_n: int = 10) -> None:
    """Print the top-N rows of the leaderboard to stdout."""
    # Prefer thresh columns when available
    has_thresh = (
        "f1_test_thresh" in df.columns
        and df["f1_test_thresh"].notna().any()
    )

    if has_thresh:
        display_cols = [
            "model", "strategy", "target_prevalence",
            "pr_auc_test",
            "optimal_threshold",
            "recall_test_thresh", "precision_test_thresh", "f1_test_thresh",
            "tp_test_thresh", "fp_test_thresh", "fn_test_thresh",
        ]
        if "f2_test_thresh" in df.columns:
            display_cols.insert(
                display_cols.index("f1_test_thresh") + 1,
                "f2_test_thresh",
            )
    else:
        display_cols = [
            "model", "strategy", "target_prevalence",
            "pr_auc_val", "pr_auc_test",
            "roc_auc_test", "recall_test", "f1_test",
            "tp_test", "fn_test",
        ]

    cols_present = [c for c in display_cols if c in df.columns]
    top = df[cols_present].head(top_n)

    header = "optimal-threshold metrics" if has_thresh else "default-threshold (0.5) metrics"
    print()
    print("=" * 80)
    print(
        f"  PART A LEADERBOARD  (top {min(top_n, len(df))} of {len(df)} runs)"
    )
    print(f"  Showing: {header}")
    print("  sorted by pr_auc_test DESC, recall DESC")
    print("=" * 80)
    with pd.option_context(
        "display.max_columns", None,
        "display.width", 160,
        "display.float_format", "{:.6f}".format,
    ):
        print(top.to_string(index=True))
    print()


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", type=str, default=None,
                        help="Path to a custom paths.yaml (e.g. for Colab)")
    args = parser.parse_args()
    try:
        paths = PathConfig(args.paths) if args.paths else PathConfig()
        df = aggregate_part_a(paths)
        if df.empty:
            print("No Part A runs found. Run the benchmark first.")
            sys.exit(0)
        out = save_leaderboard(df, paths)
        _print_leaderboard(df)
        print(f"Leaderboard saved to: {out}")
    except FileNotFoundError as exc:
        logger.error(str(exc))
        sys.exit(1)
    except Exception as exc:
        logger.exception(f"Aggregation failed: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
