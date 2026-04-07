"""Post-hoc threshold optimisation for all completed Part A runs.

Why this script exists
----------------------
The main experiment runner saves metrics at the default threshold of 0.5.
Under extreme class imbalance (~0.046% positives) this threshold almost
never triggers a positive prediction, so precision/recall/F1 are near zero
even when PR-AUC is reasonable.

This script performs a **scientifically correct** post-hoc threshold search:

1. For each run folder, load the saved model and feature pipeline.
2. Re-run predictions on the validation split.
3. Find the threshold that maximises F1 on the validation split.
4. Apply that threshold to the test split.
5. Save supplementary metrics files:
   - ``metrics_val_thresh.json`` / ``metrics_val_thresh.csv``
   - ``metrics_test_thresh.json`` / ``metrics_test_thresh.csv``
   - ``threshold_info.json``  – documents the chosen threshold and criterion

This approach is methodologically sound because:
- The threshold is selected purely on the validation set.
- The test set is evaluated exactly once with the frozen threshold.
- No re-training occurs; the model weights are unchanged.

Usage
-----
    python -m aml_benchmark.experiments.re_evaluate
"""
from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

import joblib
import numpy as np

from aml_benchmark.config import PathConfig
from aml_benchmark.features.feature_cache import load_features
from aml_benchmark.evaluation.metrics import (
    compute_all_metrics,
    find_optimal_threshold,
    save_metrics,
)
from aml_benchmark.utils.io import load_parquet
from aml_benchmark.utils.logging_utils import get_logger

logger = get_logger(__name__)


def re_evaluate_run(
    run_dir: Path,
    val_df,
    test_df,
    paths: PathConfig,
    criterion: str = "f1",
    overwrite: bool = False,
) -> dict | None:
    """Re-evaluate one run with an optimal threshold.

    Parameters
    ----------
    run_dir:
        Path to a single run output folder.
    val_df / test_df:
        Pre-loaded validation and test DataFrames (avoids re-loading per run).
    paths:
        :class:`~aml_benchmark.config.PathConfig` (used for ``splits_dir`` feature cache).
    criterion:
        Threshold optimisation criterion (``"f1"``, ``"recall"``, ``"precision"``).
    overwrite:
        If ``False`` and ``metrics_test_thresh.json`` already exists, skip.

    Returns
    -------
    Dict with val and test metrics at the optimal threshold, or ``None`` if
    the run was skipped or failed.
    """
    thresh_path = run_dir / "metrics_test_thresh.json"
    if thresh_path.exists() and not overwrite:
        logger.info(f"Skipping {run_dir.name} (already re-evaluated)")
        return None

    model_path = run_dir / "model.pkl"

    if not model_path.exists():
        logger.warning(f"Skipping {run_dir.name}: model.pkl missing")
        return None

    config_path = run_dir / "run_config.json"
    if not config_path.exists():
        logger.warning(f"Skipping {run_dir.name}: run_config.json missing")
        return None

    with config_path.open(encoding="utf-8") as fh:
        run_cfg = json.load(fh)

    # Only re-evaluate Part A runs
    if "strategy" not in run_cfg or "target_prevalence" not in run_cfg:
        logger.debug(f"Skipping {run_dir.name}: not a Part A run")
        return None

    logger.info(f"Re-evaluating {run_dir.name} ...")

    model = joblib.load(model_path)

    y_val = val_df["label"].to_numpy(dtype=int)
    y_test = test_df["label"].to_numpy(dtype=int)

    splits_dir = paths.splits_dir
    X_val = load_features(splits_dir, "val")
    X_test = load_features(splits_dir, "test")

    y_score_val: np.ndarray = model.predict_proba(X_val)[:, 1]
    y_score_test: np.ndarray = model.predict_proba(X_test)[:, 1]

    # Find optimal threshold on the validation set
    optimal_threshold = find_optimal_threshold(y_val, y_score_val, criterion=criterion)

    # Compute and save metrics at optimal threshold
    m_val = compute_all_metrics(y_val, y_score_val, threshold=optimal_threshold, split="val_thresh")
    m_test = compute_all_metrics(y_test, y_score_test, threshold=optimal_threshold, split="test_thresh")

    save_metrics(m_val, output_dir=run_dir, split="val_thresh")
    save_metrics(m_test, output_dir=run_dir, split="test_thresh")

    # Save threshold metadata
    threshold_info = {
        "criterion": criterion,
        "optimal_threshold": optimal_threshold,
        "val_f1_at_threshold": m_val["f1"],
        "val_recall_at_threshold": m_val["recall"],
        "val_precision_at_threshold": m_val["precision"],
    }
    with (run_dir / "threshold_info.json").open("w", encoding="utf-8") as fh:
        json.dump(threshold_info, fh, indent=2)

    return {"val": m_val, "test": m_test, "threshold": optimal_threshold}


def re_evaluate_all(
    criterion: str = "f1",
    overwrite: bool = False,
    paths: PathConfig | None = None,
) -> None:
    """Re-evaluate all Part A runs in ``outputs/runs/``.

    Parameters
    ----------
    criterion:
        Threshold optimisation criterion for every run.
    overwrite:
        If ``True``, re-evaluate even runs that already have thresh files.
    paths:
        Optional pre-built :class:`~aml_benchmark.config.PathConfig`.
    """
    if paths is None:
        paths = PathConfig()

    paths.validate_splits()

    logger.info(f"Loading val and test splits once ...")
    val_df = load_parquet(paths.val_split)
    test_df = load_parquet(paths.test_split)

    run_dirs = sorted(
        d for d in paths.outputs_dir.iterdir() if d.is_dir()
    )
    logger.info(f"Found {len(run_dirs)} run folders to check.")

    passed: list[str] = []
    skipped: list[str] = []
    failed: list[tuple[str, str]] = []

    t_start = time.perf_counter()

    for run_dir in run_dirs:
        try:
            result = re_evaluate_run(
                run_dir=run_dir,
                val_df=val_df,
                test_df=test_df,
                paths=paths,
                criterion=criterion,
                overwrite=overwrite,
            )
            if result is None:
                skipped.append(run_dir.name)
            else:
                passed.append(run_dir.name)
                logger.info(
                    f"  threshold={result['threshold']:.4f} | "
                    f"val_f1={result['val']['f1']:.4f} | "
                    f"test_f1={result['test']['f1']:.4f} | "
                    f"test_recall={result['test']['recall']:.4f} | "
                    f"test_tp={int(result['test']['tp'])}"
                )
        except Exception as exc:
            logger.error(f"Failed {run_dir.name}: {traceback.format_exc()}")
            failed.append((run_dir.name, f"{type(exc).__name__}: {exc}"))

    elapsed = time.perf_counter() - t_start
    _print_summary(passed, skipped, failed, elapsed, criterion)


def _print_summary(
    passed: list[str],
    skipped: list[str],
    failed: list[tuple[str, str]],
    elapsed: float,
    criterion: str,
) -> None:
    print()
    print("=" * 62)
    print("  RE-EVALUATION COMPLETE")
    print(f"  Threshold criterion : {criterion}")
    print("=" * 62)
    print(f"  Re-evaluated : {len(passed)}")
    print(f"  Skipped      : {len(skipped)}")
    print(f"  Failed       : {len(failed)}")
    print(f"  Elapsed      : {elapsed:.1f}s")
    if failed:
        print("\n  FAILED RUNS:")
        for name, err in failed:
            print(f"    {name}  ->  {err}")
    print("=" * 62)
    print()


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", type=str, default=None,
                        help="Path to a custom paths.yaml (e.g. for Colab)")
    args = parser.parse_args()
    try:
        paths = PathConfig(args.paths) if args.paths else PathConfig()
        re_evaluate_all(criterion="f1", overwrite=False, paths=paths)
    except FileNotFoundError as exc:
        logger.error(str(exc))
        sys.exit(1)
    except Exception as exc:
        logger.exception(f"Re-evaluation failed: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
