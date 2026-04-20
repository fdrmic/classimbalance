"""Strategy 6: precision-constrained threshold optimization (no retraining).

This module implements a *post-hoc* operating-point optimization for the
**XGBoost baseline** model trained in Part A. The trained model stays fixed;
only the decision threshold is selected using the validation split.

Important: This strategy must not claim improvements in PR-AUC because PR-AUC
is threshold-independent. Any improvements are limited to threshold-dependent
operational metrics (precision/recall/F1, confusion counts, and utility).

Entry point
-----------
    python -m aml_benchmark.experiments.threshold_optimizer --paths configs/paths_large_v2.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from aml_benchmark.config import PathConfig
from aml_benchmark.evaluation.metrics import compute_all_metrics
from aml_benchmark.features.feature_cache import load_features
from aml_benchmark.utils.logging_utils import get_logger

logger = get_logger(__name__)


def _iter_completed_baseline_runs(outputs_dir: Path) -> list[Path]:
    """Return all completed xgboost baseline run dirs under outputs_dir."""
    if not outputs_dir.exists():
        return []
    runs: list[Path] = []
    prefix = "xgboost__baseline__"
    for d in sorted(outputs_dir.iterdir()):
        if not d.is_dir():
            continue
        if not d.name.startswith(prefix):
            continue
        if (d / "model.pkl").exists() and (d / "run_config.json").exists() and (d / "metrics_val.json").exists():
            runs.append(d)
    return runs


def _load_pr_auc_val(run_dir: Path) -> float | None:
    p = run_dir / "metrics_val.json"
    try:
        with p.open(encoding="utf-8") as fh:
            m = json.load(fh)
        v = m.get("pr_auc")
        return float(v) if v is not None else None
    except Exception as exc:
        logger.warning(f"Could not read {p}: {exc}")
        return None


def _select_reference_run(outputs_dir: Path) -> Path:
    """Select baseline run with highest PR-AUC on validation."""
    runs = _iter_completed_baseline_runs(outputs_dir)
    if not runs:
        # Common local setup: Part A runs may still be under outputs/runs
        fallback_dir = outputs_dir.parent / "runs"
        if fallback_dir != outputs_dir and fallback_dir.exists():
            logger.warning(
                f"No completed baseline runs found under {outputs_dir}. "
                f"Falling back to {fallback_dir}."
            )
            runs = _iter_completed_baseline_runs(fallback_dir)

    if not runs:
        raise FileNotFoundError(
            f"No completed baseline runs found under {outputs_dir}. "
            "Expected folders like xgboost__baseline__* with model.pkl, run_config.json, metrics_val.json."
        )

    scored: list[tuple[float, Path]] = []
    for r in runs:
        s = _load_pr_auc_val(r)
        if s is None:
            continue
        scored.append((s, r))

    if not scored:
        chosen = runs[0]
        logger.warning(
            "All baseline runs missing readable pr_auc_val in metrics_val.json; "
            f"selecting arbitrary run: {chosen.name}"
        )
        return chosen

    scored.sort(key=lambda t: t[0], reverse=True)
    best_score, best_run = scored[0]
    # If identical (within float equality), any run is acceptable — log explicitly.
    if all(s == best_score for s, _ in scored):
        logger.info(
            f"All baseline runs have identical pr_auc_val={best_score:.6f}; "
            f"selecting run: {best_run.name}"
        )
    else:
        logger.info(
            f"Selected baseline run by highest pr_auc_val: {best_run.name} (pr_auc_val={best_score:.6f})"
        )
    return best_run


def _load_labels(paths: PathConfig) -> tuple[np.ndarray, np.ndarray]:
    """Load y_val and y_test from split parquet files."""
    val_df = pd.read_parquet(paths.val_split, columns=["label"])
    test_df = pd.read_parquet(paths.test_split, columns=["label"])
    y_val = val_df["label"].to_numpy(dtype=int)
    y_test = test_df["label"].to_numpy(dtype=int)
    return y_val, y_test


def _threshold_grid(n_thresholds: int) -> np.ndarray:
    return np.linspace(0.0, 0.15, int(n_thresholds), dtype=np.float64)


def _confusion_at_threshold(y_true: np.ndarray, y_score: np.ndarray, thr: float) -> tuple[int, int, int, int]:
    y_pred = (y_score >= thr)
    y_true_b = (y_true == 1)
    tp = int(np.sum(y_pred & y_true_b))
    fp = int(np.sum(y_pred & (~y_true_b)))
    fn = int(np.sum((~y_pred) & y_true_b))
    tn = int(np.sum((~y_pred) & (~y_true_b)))
    return tp, fp, fn, tn


def _precision_recall_f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    denom = precision + recall
    f1 = (2 * precision * recall / denom) if denom > 0 else 0.0
    return precision, recall, f1


def _utility(tp: int, fp: int, lambda_fp: float) -> float:
    return float(tp) - float(lambda_fp) * float(fp)


def _select_threshold(
    y_val: np.ndarray,
    y_score_val: np.ndarray,
    precision_constraint: float,
    lambda_fp: float,
    n_thresholds: int,
) -> tuple[float, dict[str, float], str]:
    """Select tau* on validation set using the required rule."""
    grid = _threshold_grid(n_thresholds)

    best_thr: float | None = None
    best_f1: float = -1.0
    best_u: float = -1.0
    best_pr: float = 0.0
    best_rc: float = 0.0
    best_counts: tuple[int, int, int, int] = (0, 0, 0, 0)

    best_fallback_thr: float | None = None
    best_fallback_prec: float = -1.0
    best_fallback_u: float = -1.0
    best_fallback_counts: tuple[int, int, int, int] = (0, 0, 0, 0)
    best_fallback_rc: float = 0.0
    best_fallback_f1: float = 0.0

    for thr in grid:
        tp, fp, fn, tn = _confusion_at_threshold(y_val, y_score_val, float(thr))
        prec, rec, f1 = _precision_recall_f1(tp, fp, fn)
        u = _utility(tp, fp, lambda_fp)

        if prec >= precision_constraint:
            if (f1 > best_f1) or (f1 == best_f1 and u > best_u):
                best_thr = float(thr)
                best_f1 = float(f1)
                best_u = float(u)
                best_pr = float(prec)
                best_rc = float(rec)
                best_counts = (tp, fp, fn, tn)

        # Fallback pool: highest precision with non-zero recall
        if rec > 0:
            if (prec > best_fallback_prec) or (prec == best_fallback_prec and u > best_fallback_u):
                best_fallback_thr = float(thr)
                best_fallback_prec = float(prec)
                best_fallback_u = float(u)
                best_fallback_counts = (tp, fp, fn, tn)
                best_fallback_rc = float(rec)
                best_fallback_f1 = float(f1)

    if best_thr is not None:
        tp, fp, fn, tn = best_counts
        details = {
            "tp": float(tp),
            "fp": float(fp),
            "fn": float(fn),
            "tn": float(tn),
            "precision": best_pr,
            "recall": best_rc,
            "f1": best_f1,
            "utility": best_u,
        }
        return best_thr, details, "max_f1_subject_to_precision_constraint_tiebreak_utility"

    if best_fallback_thr is not None:
        tp, fp, fn, tn = best_fallback_counts
        details = {
            "tp": float(tp),
            "fp": float(fp),
            "fn": float(fn),
            "tn": float(tn),
            "precision": best_fallback_prec,
            "recall": best_fallback_rc,
            "f1": best_fallback_f1,
            "utility": best_fallback_u,
        }
        logger.warning(
            "No threshold satisfies the precision constraint "
            f"(precision >= {precision_constraint:.3f}). "
            "Falling back to highest precision with non-zero recall."
        )
        return best_fallback_thr, details, "fallback_max_precision_with_nonzero_recall"

    # Degenerate case: never predicts positives at any threshold in grid (or no positives in y_val)
    logger.warning(
        "No threshold yields non-zero recall on validation within the grid; "
        "falling back to threshold=0.5."
    )
    thr = 0.5
    tp, fp, fn, tn = _confusion_at_threshold(y_val, y_score_val, thr)
    prec, rec, f1 = _precision_recall_f1(tp, fp, fn)
    details = {
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "utility": _utility(tp, fp, lambda_fp),
    }
    return thr, details, "fallback_threshold_0_5"


def _save_metrics_files(metrics: dict[str, float], out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{stem}.json"
    csv_path = out_dir / f"{stem}.csv"
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    pd.DataFrame([metrics]).to_csv(csv_path, index=False)
    logger.info(f"Saved -> {json_path}")
    logger.info(f"Saved -> {csv_path}")


def run_threshold_optimization(
    paths: PathConfig,
    precision_constraint: float = 0.10,
    lambda_fp: float = 0.05,
    n_thresholds: int = 1000,
) -> dict:
    """Run Strategy 6 threshold optimization for all three baseline prevalences."""
    # 1) Locate baseline runs (p001/p005/p010)
    runs = _iter_completed_baseline_runs(paths.outputs_dir)
    if not runs:
        fallback_dir = paths.outputs_dir.parent / "runs"
        if fallback_dir != paths.outputs_dir and fallback_dir.exists():
            logger.warning(
                f"No completed baseline runs found under {paths.outputs_dir}. "
                f"Falling back to {fallback_dir}."
            )
            runs = _iter_completed_baseline_runs(fallback_dir)

    wanted_tags = ("__p001__", "__p005__", "__p010__")
    runs = [r for r in runs if any(tag in r.name for tag in wanted_tags)]
    if not runs:
        raise FileNotFoundError(
            "No completed xgboost baseline runs found for p001/p005/p010. "
            "Expected folders like xgboost__baseline__p001__* with model.pkl, run_config.json, metrics_val.json."
        )

    # If there are multiple runs per prevalence tag, keep the one with highest pr_auc_val
    chosen_by_tag: dict[str, Path] = {}
    for tag in wanted_tags:
        candidates = [r for r in runs if tag in r.name]
        if not candidates:
            continue
        scored: list[tuple[float, Path]] = []
        for c in candidates:
            s = _load_pr_auc_val(c)
            if s is not None:
                scored.append((s, c))
        if scored:
            scored.sort(key=lambda t: t[0], reverse=True)
            chosen_by_tag[tag] = scored[0][1]
        else:
            chosen_by_tag[tag] = sorted(candidates)[-1]

    selected_runs = [chosen_by_tag[t] for t in wanted_tags if t in chosen_by_tag]
    logger.info(f"Selected {len(selected_runs)} baseline runs for Strategy 6: {[r.name for r in selected_runs]}")

    # 2) Load shared data once (features from cache, labels from split parquet)
    X_val = load_features(paths.splits_dir, "val")
    X_test = load_features(paths.splits_dir, "test")
    y_val, y_test = _load_labels(paths)

    all_infos: dict[str, dict] = {}
    summary_rows: list[dict[str, object]] = []

    for ref_run in selected_runs:
        run_id = ref_run.name
        logger.info("-" * 62)
        logger.info(f"STRATEGY 6 — processing baseline run: {run_id}")

        model = joblib.load(ref_run / "model.pkl")
        y_score_val: np.ndarray = model.predict_proba(X_val)[:, 1]
        y_score_test: np.ndarray = model.predict_proba(X_test)[:, 1]

        # 3) Threshold search on validation
        tau_star, val_details, selection_criterion = _select_threshold(
            y_val=y_val,
            y_score_val=y_score_val,
            precision_constraint=precision_constraint,
            lambda_fp=lambda_fp,
            n_thresholds=n_thresholds,
        )
        logger.info(
            f"Selected tau*={tau_star:.6f} "
            f"| val_precision={val_details['precision']:.4f} "
            f"| val_recall={val_details['recall']:.4f} "
            f"| val_f1={val_details['f1']:.4f} "
            f"| val_U={val_details['utility']:.2f}"
        )

        # 4) Evaluate tau* on validation and test
        m_val = compute_all_metrics(y_val, y_score_val, threshold=tau_star, split="val_strategy6")
        m_test = compute_all_metrics(y_test, y_score_test, threshold=tau_star, split="test_strategy6")

        # Baseline operating point: Part A optimized threshold if available
        thresh_info_path = ref_run / "threshold_info.json"
        if thresh_info_path.exists():
            with thresh_info_path.open(encoding="utf-8") as fh:
                thresh_info = json.load(fh)
            baseline_thr = float(thresh_info.get("optimal_threshold", 0.5))
            logger.info(f"Loaded Part A optimized threshold: {baseline_thr:.6f}")
        else:
            baseline_thr = 0.5
            logger.warning("threshold_info.json not found; using baseline_thr=0.5")

        m_test_baseline = compute_all_metrics(
            y_test, y_score_test, threshold=baseline_thr, split="test_baseline_part_a_thresh"
        )

        # 5) Persist outputs per run
        out_dir = paths.outputs_dir.parent / "strategy6" / run_id
        _save_metrics_files(m_val, out_dir, "metrics_strategy6_val")
        _save_metrics_files(m_test, out_dir, "metrics_strategy6_test")

        tp_b = int(m_test_baseline["tp"])
        fp_b = int(m_test_baseline["fp"])
        tp_s = int(m_test["tp"])
        fp_s = int(m_test["fp"])
        delta_tp = tp_s - tp_b
        delta_fp = fp_s - fp_b

        fp_per_tp_b = (fp_b / tp_b) if tp_b > 0 else float("inf")
        fp_per_tp_s = (fp_s / tp_s) if tp_s > 0 else float("inf")

        info = {
            "selected_baseline_run_id": run_id,
            "optimal_threshold": float(tau_star),
            "selection_criterion": selection_criterion,
            "lambda_fp": float(lambda_fp),
            "precision_constraint": float(precision_constraint),
            "threshold_grid": {
                "min": 0.0,
                "max": 0.15,
                "n_thresholds": int(n_thresholds),
                "spacing": "linear",
            },
            "val_metrics_at_tau_star": m_val,
            "test_metrics_at_tau_star": m_test,
            "baseline_comparison_test": {
                "baseline_threshold": float(baseline_thr),
                "baseline": {
                    "precision": float(m_test_baseline["precision"]),
                    "recall": float(m_test_baseline["recall"]),
                    "f1": float(m_test_baseline["f1"]),
                    "tp": tp_b,
                    "fp": fp_b,
                },
                "strategy6": {
                    "precision": float(m_test["precision"]),
                    "recall": float(m_test["recall"]),
                    "f1": float(m_test["f1"]),
                    "tp": tp_s,
                    "fp": fp_s,
                },
                "delta": {
                    "tp": int(delta_tp),
                    "fp": int(delta_fp),
                },
                "fp_per_tp": {
                    "baseline": float(fp_per_tp_b),
                    "strategy6": float(fp_per_tp_s),
                },
            },
        }

        info_path = out_dir / "threshold_info_strategy6.json"
        with info_path.open("w", encoding="utf-8") as fh:
            json.dump(info, fh, indent=2)
        logger.info(f"Saved -> {info_path}")

        all_infos[run_id] = info

        summary_rows.append(
            {
                "run_id": run_id,
                "baseline_thr": float(baseline_thr),
                "tau_star": float(tau_star),
                "baseline_precision": float(m_test_baseline["precision"]),
                "baseline_recall": float(m_test_baseline["recall"]),
                "baseline_f1": float(m_test_baseline["f1"]),
                "baseline_tp": tp_b,
                "baseline_fp": fp_b,
                "strategy6_precision": float(m_test["precision"]),
                "strategy6_recall": float(m_test["recall"]),
                "strategy6_f1": float(m_test["f1"]),
                "strategy6_tp": tp_s,
                "strategy6_fp": fp_s,
                "delta_tp": int(delta_tp),
                "delta_fp": int(delta_fp),
                "baseline_fp_per_tp": float(fp_per_tp_b),
                "strategy6_fp_per_tp": float(fp_per_tp_s),
            }
        )

    # Final summary table
    summary_df = pd.DataFrame(summary_rows)
    logger.info("=" * 62)
    logger.info("STRATEGY 6 — SUMMARY ACROSS BASELINE RUNS (test)")
    if not summary_df.empty:
        with pd.option_context("display.max_colwidth", 80, "display.width", 200):
            print(summary_df.to_string(index=False))
    logger.info("=" * 62)

    return {"runs": all_infos, "summary": summary_rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--paths",
        type=str,
        default=None,
        help="Path to a custom paths.yaml (e.g. for Colab)",
    )
    args = parser.parse_args()
    paths = PathConfig(args.paths) if args.paths else PathConfig()
    run_threshold_optimization(paths=paths)


if __name__ == "__main__":
    main()

