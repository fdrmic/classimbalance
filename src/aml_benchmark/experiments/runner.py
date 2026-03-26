"""Single-experiment runner for the AML benchmark.

Executes ONE complete AML classification experiment:

    Load splits -> Build features -> Apply sampling strategy
    -> Train model -> Evaluate -> Save all artefacts

This module supports both the baseline smoke test and all Part A
benchmark conditions.  The grid loop is in ``grid_runner.py``.

Output structure
----------------
    outputs/runs/<run_id>/
        run_config.json          – all parameters, counts, timestamps
        feature_pipeline.pkl     – serialised fitted FeaturePipeline
        model.pkl                – serialised fitted model
        metrics_val.json/csv     – validation-set metrics
        metrics_test.json/csv    – test-set metrics

Usage
-----
    # Default: baseline RandomForest at 1% target prevalence
    python -m aml_benchmark.experiments.runner

    # Programmatic (used by grid_runner):
    from aml_benchmark.experiments.runner import run_experiment
    result = run_experiment(
        model_name="xgboost",
        strategy="smote",
        target_prevalence=0.005,
    )
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np

from aml_benchmark.config import PathConfig, load_yaml
from aml_benchmark.evaluation.metrics import compute_all_metrics, save_metrics
from aml_benchmark.features.pipeline import FeaturePipeline
from aml_benchmark.models.factory import get_model
from aml_benchmark.sampling.strategies import SamplingResult, apply_strategy
from aml_benchmark.utils.io import load_parquet
from aml_benchmark.utils.logging_utils import get_logger

logger = get_logger(__name__)


def _prevalence_tag(prevalence: float) -> str:
    """Format prevalence as a zero-padded permille string, e.g. 'p010'."""
    return f"p{int(round(prevalence * 1000)):03d}"


# ---------------------------------------------------------------------------
# Core single-experiment function
# ---------------------------------------------------------------------------

def run_experiment(
    model_name: str = "random_forest",
    strategy: str = "baseline",
    target_prevalence: float = 0.01,
    run_id: str | None = None,
    paths: PathConfig | None = None,
) -> dict[str, dict]:
    """Execute one complete AML benchmark experiment.

    Parameters
    ----------
    model_name:
        Model family.  One of ``"random_forest"`` or ``"xgboost"``.
    strategy:
        Imbalance-mitigation strategy applied to the training split.
        One of ``"baseline"``, ``"random_undersampling"``, ``"smote"``,
        ``"adasyn"``, ``"class_weighting"``.
    target_prevalence:
        Desired positive fraction in the processed training split (e.g.
        ``0.01`` for 1.0%).  Interpretation is strategy-specific; for
        ``baseline`` it is recorded but not enforced.
    run_id:
        Output sub-directory name under ``outputs/runs/``.  Auto-generated
        from model, strategy, prevalence, and timestamp if ``None``.
    paths:
        Optional pre-built :class:`~aml_benchmark.config.PathConfig`.

    Returns
    -------
    Dict with keys ``"val"`` and ``"test"``, each containing the metrics
    dict from :func:`~aml_benchmark.evaluation.metrics.compute_all_metrics`.
    """
    t_start = time.perf_counter()

    if paths is None:
        paths = PathConfig()

    paths.validate_splits()

    exp_cfg = load_yaml("experiment")
    random_seed: int = int(exp_cfg.get("random_seed", 42))

    if run_id is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        ptag = _prevalence_tag(target_prevalence)
        run_id = f"{model_name}__{strategy}__{ptag}__{ts}"

    output_dir = paths.outputs_dir / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 62)
    logger.info(f"EXPERIMENT: {run_id}")
    logger.info(f"  model              : {model_name}")
    logger.info(f"  strategy           : {strategy}")
    logger.info(f"  target_prevalence  : {target_prevalence:.4%}")
    logger.info(f"  random_seed        : {random_seed}")
    logger.info(f"  output_dir         : {output_dir}")
    logger.info("=" * 62)

    # ------------------------------------------------------------------
    # 1. Load splits
    # ------------------------------------------------------------------
    train = load_parquet(paths.train_split)
    val = load_parquet(paths.val_split)
    test = load_parquet(paths.test_split)

    y_val = val["label"].to_numpy(dtype=int)
    y_test = test["label"].to_numpy(dtype=int)

    # ------------------------------------------------------------------
    # 2. Build features  (encoder fit on original training data only)
    # ------------------------------------------------------------------
    logger.info("Building features ...")
    pipeline = FeaturePipeline()
    X_train_raw = pipeline.fit_transform(train)
    X_val = pipeline.transform(val)
    X_test = pipeline.transform(test)

    y_train_raw = train["label"].to_numpy(dtype=int)

    logger.info(
        f"Feature matrix shapes: "
        f"train={X_train_raw.shape}, val={X_val.shape}, test={X_test.shape}"
    )

    # ------------------------------------------------------------------
    # 3. Apply sampling strategy  (train only)
    # ------------------------------------------------------------------
    sampling_result: SamplingResult = apply_strategy(
        X_train=X_train_raw,
        y_train=y_train_raw,
        strategy=strategy,
        target_prevalence=target_prevalence,
        random_state=random_seed,
    )
    X_train = sampling_result.X
    y_train = sampling_result.y

    # ------------------------------------------------------------------
    # 4. Train
    # ------------------------------------------------------------------
    logger.info(f"Training {model_name} on {len(y_train):,} samples ...")
    t_train = time.perf_counter()
    model = get_model(
        model_name,
        random_state=random_seed,
        class_weight=sampling_result.class_weight,
    )
    model.fit(X_train, y_train)
    train_sec = time.perf_counter() - t_train
    logger.info(f"Training complete in {train_sec:.1f}s")

    # ------------------------------------------------------------------
    # 5. Evaluate  (val and test remain untouched)
    # ------------------------------------------------------------------
    results: dict[str, dict] = {}
    for split_name, X, y in [("val", X_val, y_val), ("test", X_test, y_test)]:
        logger.info(f"Evaluating on {split_name} ...")
        y_score: np.ndarray = model.predict_proba(X)[:, 1]
        m = compute_all_metrics(y, y_score, threshold=0.5, split=split_name)
        save_metrics(m, output_dir=output_dir, split=split_name)
        results[split_name] = m

    # ------------------------------------------------------------------
    # 6. Persist artefacts
    # ------------------------------------------------------------------
    joblib.dump(pipeline, output_dir / "feature_pipeline.pkl")
    joblib.dump(model, output_dir / "model.pkl")

    elapsed = time.perf_counter() - t_start

    run_config = {
        "run_id": run_id,
        "model": model_name,
        "strategy": strategy,
        "target_prevalence": target_prevalence,
        "achieved_train_prevalence": round(sampling_result.achieved_prevalence, 8),
        "random_seed": random_seed,
        "feature_names": pipeline.feature_names,
        "train_rows_original": int(len(train)),
        "train_rows_after_sampling": int(len(y_train)),
        "train_positives_after_sampling": int(sampling_result.n_positive),
        "train_negatives_after_sampling": int(sampling_result.n_negative),
        "n_synthetic_samples": int(sampling_result.n_synthetic),
        "val_rows": int(len(val)),
        "val_positives": int(y_val.sum()),
        "test_rows": int(len(test)),
        "test_positives": int(y_test.sum()),
        "class_weight_used": sampling_result.class_weight,
        "train_time_sec": round(train_sec, 2),
        "total_time_sec": round(elapsed, 2),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    config_path = output_dir / "run_config.json"
    with config_path.open("w", encoding="utf-8") as fh:
        json.dump(run_config, fh, indent=2)
    logger.info(f"Run config saved -> {config_path}")

    _print_final_summary(run_id, results, elapsed, output_dir)
    return results


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def _print_final_summary(
    run_id: str,
    results: dict[str, dict],
    elapsed: float,
    output_dir: Path,
) -> None:
    print()
    print("=" * 62)
    print(f"  EXPERIMENT COMPLETE: {run_id}")
    print("=" * 62)
    for split_name, m in results.items():
        print(f"  [{split_name}]")
        print(f"    pr_auc (PRIMARY)  : {m['pr_auc']:.6f}")
        print(f"    roc_auc           : {m['roc_auc']:.6f}")
        print(f"    precision         : {m['precision']:.6f}")
        print(f"    recall            : {m['recall']:.6f}")
        print(f"    f1                : {m['f1']:.6f}")
        print(f"    weighted_accuracy : {m['weighted_accuracy']:.6f}")
        print(
            f"    confusion: TP={int(m['tp'])} FP={int(m['fp'])} "
            f"TN={int(m['tn'])} FN={int(m['fn'])}"
        )
    print(f"  Elapsed  : {elapsed:.1f}s")
    print(f"  Results  : {output_dir}")
    print("=" * 62)
    print()


# ---------------------------------------------------------------------------
# Entry point (single baseline experiment)
# ---------------------------------------------------------------------------

def main() -> None:
    """Run one baseline experiment (quick smoke test)."""
    try:
        run_experiment(
            model_name="random_forest",
            strategy="baseline",
            target_prevalence=0.01,
        )
    except FileNotFoundError as exc:
        logger.error(str(exc))
        sys.exit(1)
    except Exception as exc:
        logger.exception(f"Runner failed: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
