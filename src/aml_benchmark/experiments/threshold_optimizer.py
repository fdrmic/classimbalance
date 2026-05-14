"""Part B — multi-strategy threshold optimisation (no retraining).

This module implements *post-hoc* operating-point selection for the
**XGBoost Baseline** model trained in Part A.  The trained model stays
fixed; only the decision threshold is selected, and only on the validation
split.  The chosen threshold is then applied to the test split for the
final operational metrics.

Three strategies are evaluated in parallel on the SAME pre-computed
probability scores:

* ``precision_constrained`` — operationally motivated.  Selects the
  threshold that maximises F1 subject to ``precision >= min_precision``
  (default 0.10), with ``U(τ) = TP - λ·FP`` as a tiebreaker.  This
  strategy is the recommended choice for AML compliance teams: it
  enforces a minimum alert quality consistent with realistic
  investigative capacity (≈50–200 alerts per 10,000 transactions).
* ``f1_max`` — methodologically standard reference.  Maximises F1 on
  validation; balanced trade-off between precision and recall.  Serves
  as a neutral scientific benchmark, not an operational recommendation.
* ``f2_max`` — recall-weighted reference.  Maximises F2 on validation
  (β = 2 weights recall twice as heavily as precision).  Aligned with
  regulatory recall expectations (FATF / FinCEN), where false negatives
  carry direct supervisory consequences.

PR-AUC invariance
-----------------
PR-AUC equals ``average_precision_score(y_true, y_score)`` and is a
property of the **entire** Precision-Recall curve, integrated over all
thresholds.  Threshold selection chooses one operating point on that
curve; it cannot move the curve itself.  Therefore PR-AUC must be
identical (within float tolerance) across all three strategies and
identical to the Part A XGBoost Baseline PR-AUC.  This module asserts
that invariance at runtime; any violation indicates a bug (e.g. mixed-up
scores or accidental retraining).

Reference operating point
-------------------------
The comparison reference is the Part A XGBoost Baseline at its
**F1-optimal threshold** (loaded from each run's ``threshold_info.json``,
written by ``re_evaluate.py``), NOT the default 0.5 threshold.  Comparing
against 0.5 would inflate apparent improvements because no practitioner
would deploy at 0.5 under such extreme imbalance.

Outputs
-------
Per (run_id, strategy):
    outputs/part_b_thresholds/<run_id>/<strategy>/
        metrics_val.json  / metrics_val.csv
        metrics_test.json / metrics_test.csv
        threshold_info.json   (per-strategy schema; see below)

CLI
---
    python -m aml_benchmark.experiments.threshold_optimizer \
        --paths configs/paths_large_v2.yaml

    # Custom subset of strategies:
    python -m aml_benchmark.experiments.threshold_optimizer \
        --paths configs/paths_large_v2.yaml \
        --strategies precision_constrained f1_max

    # Self-test with mock scores (no model / no I/O of real splits):
    python -m aml_benchmark.experiments.threshold_optimizer --dry-run
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from aml_benchmark.config import PathConfig
from aml_benchmark.evaluation.metrics import compute_all_metrics
from aml_benchmark.features.feature_cache import load_features
from aml_benchmark.utils.logging_utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

VALID_STRATEGIES: tuple[str, ...] = (
    "precision_constrained",
    "f1_max",
    "f2_max",
)

# Preferred representative baseline run — Baseline is invariant to
# target_prevalence so all three (p001/p005/p010) yield bit-identical models.
PREFERRED_RUN_NAME = "xgboost__baseline__p001__20260404_143052"


# ---------------------------------------------------------------------------
# Run discovery
# ---------------------------------------------------------------------------

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
        if (
            (d / "model.pkl").exists()
            and (d / "run_config.json").exists()
            and (d / "metrics_val.json").exists()
        ):
            runs.append(d)
    return runs


def _resolve_runs_dir(outputs_dir: Path) -> Path:
    """Use outputs_dir if it has baseline runs, else fall back to outputs/runs."""
    if _iter_completed_baseline_runs(outputs_dir):
        return outputs_dir
    fallback = outputs_dir.parent / "runs"
    if fallback != outputs_dir and fallback.exists():
        if _iter_completed_baseline_runs(fallback):
            logger.warning(
                f"No baseline runs in {outputs_dir}; falling back to {fallback}."
            )
            return fallback
    return outputs_dir


def _select_single_baseline_run(outputs_dir: Path) -> Path:
    """Select exactly ONE representative baseline run.

    Baseline training is invariant to ``target_prevalence`` (no resampling and
    no class weighting), so ``xgboost__baseline__p001/p005/p010`` produce
    bit-identical models.  We therefore pick a single representative run.
    """
    runs_dir = _resolve_runs_dir(outputs_dir)
    runs = _iter_completed_baseline_runs(runs_dir)
    if not runs:
        raise FileNotFoundError(
            f"No completed xgboost baseline runs found under {runs_dir}. "
            "Expected folders like xgboost__baseline__p001__* with "
            "model.pkl, run_config.json, metrics_val.json."
        )

    preferred = runs_dir / PREFERRED_RUN_NAME
    if preferred in runs:
        chosen = preferred
        logger.info(f"Selected representative baseline run: {chosen.name}")
    else:
        chosen = runs[0]
        logger.info(
            f"Preferred run '{PREFERRED_RUN_NAME}' not found; using first "
            f"available baseline run: {chosen.name}"
        )
    return chosen


# ---------------------------------------------------------------------------
# Label loading
# ---------------------------------------------------------------------------

def _load_labels(paths: PathConfig) -> tuple[np.ndarray, np.ndarray]:
    """Load y_val and y_test from split parquet files."""
    val_df = pd.read_parquet(paths.val_split, columns=["label"])
    test_df = pd.read_parquet(paths.test_split, columns=["label"])
    y_val = val_df["label"].to_numpy(dtype=int)
    y_test = test_df["label"].to_numpy(dtype=int)
    return y_val, y_test


# ---------------------------------------------------------------------------
# Threshold grid + confusion helpers
# ---------------------------------------------------------------------------

def _build_threshold_grid(
    y_score_val: np.ndarray, n_dense: int = 1000
) -> np.ndarray:
    """Score-quantile-based grid built from VALIDATION scores only.

    Anti-leakage marker: this function takes only validation scores; it never
    sees ``y_test`` or ``y_score_test``.  The grid is therefore selected on
    the validation distribution alone.
    """
    y_score_val = np.asarray(y_score_val, dtype=np.float64)
    uniq = np.unique(y_score_val)
    if uniq.size <= n_dense:
        grid = uniq
    else:
        qs = np.linspace(0.0, 1.0, int(n_dense), dtype=np.float64)
        grid = np.unique(np.quantile(y_score_val, qs))
    grid = grid.astype(np.float64)
    logger.info(
        f"Threshold grid: n={grid.size} (val-quantile based, "
        f"min={grid.min():.6f}, max={grid.max():.6f})"
    )
    return grid


def _confusion_at_threshold(
    y_true: np.ndarray, y_score: np.ndarray, thr: float
) -> tuple[int, int, int, int]:
    y_pred = y_score >= thr
    y_true_b = y_true == 1
    tp = int(np.sum(y_pred & y_true_b))
    fp = int(np.sum(y_pred & (~y_true_b)))
    fn = int(np.sum((~y_pred) & y_true_b))
    tn = int(np.sum((~y_pred) & (~y_true_b)))
    return tp, fp, fn, tn


def _precision_recall_f1_f2(
    tp: int, fp: int, fn: int
) -> tuple[float, float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    denom_f1 = precision + recall
    f1 = (2 * precision * recall / denom_f1) if denom_f1 > 0 else 0.0
    denom_f2 = 4.0 * precision + recall
    f2 = (5.0 * precision * recall / denom_f2) if denom_f2 > 0 else 0.0
    return precision, recall, f1, f2


def _utility(tp: int, fp: int, lambda_fp: float) -> float:
    return float(tp) - float(lambda_fp) * float(fp)


# ---------------------------------------------------------------------------
# Strategy: per-strategy threshold selection on VALIDATION ONLY
# ---------------------------------------------------------------------------

def _select_threshold_by_strategy(
    strategy_type: str,
    y_val: np.ndarray,
    y_score_val: np.ndarray,
    grid: np.ndarray,
    *,
    min_precision: float = 0.10,
    lambda_fp: float = 0.05,
) -> tuple[float, dict[str, float], str]:
    """Select tau* on the VALIDATION set only.

    Anti-leakage marker: this function intentionally accepts ONLY the
    validation labels and validation scores.  Test data is not in scope here
    and is never read by callers when invoking this function.
    """
    if strategy_type not in VALID_STRATEGIES:
        raise ValueError(
            f"Unknown strategy_type '{strategy_type}'. "
            f"Expected one of {VALID_STRATEGIES}."
        )

    if strategy_type == "precision_constrained":
        return _select_precision_constrained(
            y_val=y_val,
            y_score_val=y_score_val,
            grid=grid,
            min_precision=min_precision,
            lambda_fp=lambda_fp,
        )

    # f1_max / f2_max — argmax of the chosen objective on validation.
    best_thr: float | None = None
    best_obj: float = -np.inf
    best_details: dict[str, float] = {}

    for thr in grid:
        tp, fp, fn, tn = _confusion_at_threshold(y_val, y_score_val, float(thr))
        precision, recall, f1, f2 = _precision_recall_f1_f2(tp, fp, fn)
        obj = f1 if strategy_type == "f1_max" else f2
        if obj > best_obj:
            best_obj = float(obj)
            best_thr = float(thr)
            best_details = {
                "tp": int(tp),
                "fp": int(fp),
                "fn": int(fn),
                "tn": int(tn),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "f2": float(f2),
                "objective": strategy_type,
                "objective_value": best_obj,
                "utility": _utility(tp, fp, lambda_fp),
            }

    if best_thr is None:
        # Degenerate: empty grid — fall back to 0.5
        logger.warning(
            f"{strategy_type}: empty threshold grid; falling back to 0.5."
        )
        return 0.5, {"objective": strategy_type}, f"fallback_threshold_0_5"

    return best_thr, best_details, f"argmax_{strategy_type}_on_validation"


def _select_precision_constrained(
    y_val: np.ndarray,
    y_score_val: np.ndarray,
    grid: np.ndarray,
    *,
    min_precision: float,
    lambda_fp: float,
) -> tuple[float, dict[str, float], str]:
    """Max F1 subject to ``precision >= min_precision`` on validation.

    Tiebreak by utility ``U(τ) = TP − λ·FP``.  Two fallbacks: highest
    precision with non-zero recall; finally threshold = 0.5.
    """
    best_thr: float | None = None
    best_f1 = -1.0
    best_u = -np.inf
    best_details: dict[str, float] = {}

    fb_thr: float | None = None
    fb_prec = -1.0
    fb_u = -np.inf
    fb_details: dict[str, float] = {}

    for thr in grid:
        tp, fp, fn, tn = _confusion_at_threshold(y_val, y_score_val, float(thr))
        precision, recall, f1, f2 = _precision_recall_f1_f2(tp, fp, fn)
        u = _utility(tp, fp, lambda_fp)

        if precision >= min_precision:
            if (f1 > best_f1) or (f1 == best_f1 and u > best_u):
                best_thr = float(thr)
                best_f1 = float(f1)
                best_u = float(u)
                best_details = {
                    "tp": int(tp),
                    "fp": int(fp),
                    "fn": int(fn),
                    "tn": int(tn),
                    "precision": float(precision),
                    "recall": float(recall),
                    "f1": float(f1),
                    "f2": float(f2),
                    "utility": float(u),
                    "objective": "precision_constrained",
                    "objective_value": float(f1),
                }

        if recall > 0:
            if (precision > fb_prec) or (precision == fb_prec and u > fb_u):
                fb_thr = float(thr)
                fb_prec = float(precision)
                fb_u = float(u)
                fb_details = {
                    "tp": int(tp),
                    "fp": int(fp),
                    "fn": int(fn),
                    "tn": int(tn),
                    "precision": float(precision),
                    "recall": float(recall),
                    "f1": float(f1),
                    "f2": float(f2),
                    "utility": float(u),
                    "objective": "precision_constrained",
                    "objective_value": float(f1),
                }

    if best_thr is not None:
        return (
            best_thr,
            best_details,
            "max_f1_subject_to_precision_constraint_tiebreak_utility",
        )
    if fb_thr is not None:
        logger.warning(
            "No threshold satisfies the precision constraint "
            f"(precision >= {min_precision:.3f}); "
            "falling back to highest precision with non-zero recall."
        )
        return fb_thr, fb_details, "fallback_max_precision_with_nonzero_recall"

    logger.warning(
        "No threshold yields non-zero recall on validation within the grid; "
        "falling back to threshold=0.5."
    )
    thr = 0.5
    tp, fp, fn, tn = _confusion_at_threshold(y_val, y_score_val, thr)
    precision, recall, f1, f2 = _precision_recall_f1_f2(tp, fp, fn)
    return (
        thr,
        {
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "f2": float(f2),
            "utility": _utility(tp, fp, lambda_fp),
            "objective": "precision_constrained",
            "objective_value": float(f1),
        },
        "fallback_threshold_0_5",
    )


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def _save_metrics_files(
    metrics: dict[str, float], out_dir: Path, stem: str
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{stem}.json"
    csv_path = out_dir / f"{stem}.csv"
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    pd.DataFrame([metrics]).to_csv(csv_path, index=False)
    logger.info(f"Saved -> {json_path}")
    logger.info(f"Saved -> {csv_path}")


def _load_part_a_reference_threshold(run_dir: Path) -> float:
    """Return the F1-optimal threshold for the Part A baseline run.

    Reads from ``threshold_info.json`` written by ``re_evaluate.py``.  Logs a
    warning (but does not raise) if it happens to equal 0.5, since 0.5 is
    almost certainly NOT the F1-optimal operating point under extreme
    imbalance and would suggest the Part A re-evaluation was skipped.
    """
    p = run_dir / "threshold_info.json"
    if not p.exists():
        raise FileNotFoundError(
            f"Part A threshold_info.json not found at {p}. "
            "Run `python -m aml_benchmark.experiments.re_evaluate` first."
        )
    info = json.loads(p.read_text(encoding="utf-8"))
    thr = float(info.get("optimal_threshold", 0.5))
    crit = str(info.get("criterion", "?"))
    if abs(thr - 0.5) < 1e-9:
        logger.warning(
            "Part A reference threshold equals 0.5 — verify Part A "
            "F1-optimal re-evaluation actually produced this value."
        )
    logger.info(
        f"Part A reference operating point: threshold={thr:.6f} "
        f"(criterion={crit})"
    )
    return thr


# ---------------------------------------------------------------------------
# Core entry point
# ---------------------------------------------------------------------------

def run_threshold_optimization(
    paths: PathConfig,
    strategies: Iterable[str] = VALID_STRATEGIES,
    *,
    precision_constraint: float = 0.10,
    lambda_fp: float = 0.05,
    n_dense: int = 1000,
    write_summary: bool = True,
) -> dict:
    """Run multi-strategy threshold optimisation on ONE Part A baseline run.

    Parameters
    ----------
    paths:
        Resolved :class:`PathConfig` (controls splits + outputs locations).
    strategies:
        Iterable of strategy names from :data:`VALID_STRATEGIES`.
    precision_constraint:
        Used by ``precision_constrained`` only (default 0.10).
    lambda_fp:
        Tiebreak utility coefficient for ``precision_constrained`` (default
        0.05).
    n_dense:
        Maximum size of the validation-quantile threshold grid.
    write_summary:
        If True, also write ``results/part_b_multi_threshold_summary.json``.

    Returns
    -------
    A dict with keys ``selected_run_id``, ``part_a_reference``, ``records``,
    and ``pr_auc_invariance``.
    """
    strategies = tuple(strategies)
    for s in strategies:
        if s not in VALID_STRATEGIES:
            raise ValueError(
                f"Unknown strategy '{s}'. Expected subset of {VALID_STRATEGIES}."
            )
    if not strategies:
        raise ValueError("At least one strategy must be selected.")

    # 1) Select ONE representative baseline run (baseline is prevalence-invariant)
    ref_run = _select_single_baseline_run(paths.outputs_dir)
    run_id = ref_run.name

    # 2) Load shared inputs once (features from cache, labels from split parquet)
    X_val = load_features(paths.splits_dir, "val")
    X_test = load_features(paths.splits_dir, "test")
    y_val, y_test = _load_labels(paths)

    # 3) Compute scores ONCE per run.  Identical scores across all strategies
    #    is what makes PR-AUC invariant.
    model = joblib.load(ref_run / "model.pkl")
    y_score_val: np.ndarray = model.predict_proba(X_val)[:, 1]
    y_score_test: np.ndarray = model.predict_proba(X_test)[:, 1]

    # 4) Build val-quantile grid (anti-leakage: validation only)
    grid = _build_threshold_grid(y_score_val, n_dense=n_dense)

    # 5) Part A reference operating point (F1-optimal, NOT 0.5)
    ref_threshold = _load_part_a_reference_threshold(ref_run)
    m_test_partA = compute_all_metrics(
        y_test,
        y_score_test,
        threshold=ref_threshold,
        split="test_part_a_f1opt",
    )
    partA_pr_auc = float(m_test_partA["pr_auc"])
    part_a_reference = {
        "selected_baseline_run_id": run_id,
        "threshold": float(ref_threshold),
        "threshold_criterion": "f1",
        "precision": float(m_test_partA["precision"]),
        "recall": float(m_test_partA["recall"]),
        "f1": float(m_test_partA["f1"]),
        "f2": float(m_test_partA["f2"]),
        "tp": int(m_test_partA["tp"]),
        "fp": int(m_test_partA["fp"]),
        "tn": int(m_test_partA["tn"]),
        "fn": int(m_test_partA["fn"]),
        "pr_auc": partA_pr_auc,
    }

    out_root = paths.outputs_dir.parent / "part_b_thresholds" / run_id
    records: list[dict] = []

    for strategy in strategies:
        logger.info("-" * 62)
        logger.info(f"STRATEGY: {strategy}  (run: {run_id})")

        # ------------------------------------------------------------------
        # ANTI-LEAKAGE: tau* is selected from y_val + y_score_val only.
        # Test scores are NEVER passed into the selector below.
        # ------------------------------------------------------------------
        tau, val_details, criterion = _select_threshold_by_strategy(
            strategy_type=strategy,
            y_val=y_val,
            y_score_val=y_score_val,
            grid=grid,
            min_precision=precision_constraint,
            lambda_fp=lambda_fp,
        )
        logger.info(
            f"Selected tau*={tau:.6f} | criterion={criterion} | "
            f"val_P={val_details.get('precision', 0.0):.4f} "
            f"val_R={val_details.get('recall', 0.0):.4f} "
            f"val_F1={val_details.get('f1', 0.0):.4f} "
            f"val_F2={val_details.get('f2', 0.0):.4f}"
        )

        # ------------------------------------------------------------------
        # Apply tau* (selected on VAL) to TEST.  No further tuning happens.
        # ------------------------------------------------------------------
        m_val = compute_all_metrics(
            y_val, y_score_val, threshold=tau, split=f"val_{strategy}"
        )
        m_test = compute_all_metrics(
            y_test, y_score_test, threshold=tau, split=f"test_{strategy}"
        )

        # PR-AUC invariance: same scores -> identical PR-AUC across strategies
        if abs(m_test["pr_auc"] - partA_pr_auc) > 1e-9:
            raise RuntimeError(
                f"PR-AUC invariance violated for strategy '{strategy}': "
                f"got {m_test['pr_auc']} vs Part A {partA_pr_auc} "
                f"(diff > 1e-9). Likely cause: different scores or model."
            )

        record = {
            "selected_baseline_run_id": run_id,
            "strategy_type": strategy,
            "selection_criterion": criterion,
            "threshold_value": float(tau),
            "threshold_grid": {
                "source": "val_score_quantiles",
                "n": int(grid.size),
                "n_dense_requested": int(n_dense),
            },
            "precision_constraint": float(precision_constraint)
            if strategy == "precision_constrained"
            else None,
            "lambda_fp": float(lambda_fp)
            if strategy == "precision_constrained"
            else None,
            # validation
            "val_precision": float(m_val["precision"]),
            "val_recall": float(m_val["recall"]),
            "val_f1": float(m_val["f1"]),
            "val_f2": float(m_val["f2"]),
            # test
            "test_precision": float(m_test["precision"]),
            "test_recall": float(m_test["recall"]),
            "test_f1": float(m_test["f1"]),
            "test_f2": float(m_test["f2"]),
            "test_pr_auc": float(m_test["pr_auc"]),
            "test_tp": int(m_test["tp"]),
            "test_fp": int(m_test["fp"]),
            "test_tn": int(m_test["tn"]),
            "test_fn": int(m_test["fn"]),
            # Part A reference (F1-optimal threshold, NOT 0.5)
            "part_a_reference": part_a_reference,
            # Deltas vs Part A F1-optimal operating point on TEST
            "delta_precision_vs_part_a_f1opt": float(
                m_test["precision"] - m_test_partA["precision"]
            ),
            "delta_recall_vs_part_a_f1opt": float(
                m_test["recall"] - m_test_partA["recall"]
            ),
            "delta_f1_vs_part_a_f1opt": float(
                m_test["f1"] - m_test_partA["f1"]
            ),
            "delta_f2_vs_part_a_f1opt": float(
                m_test["f2"] - m_test_partA["f2"]
            ),
            "delta_fp_vs_part_a_f1opt": int(m_test["fp"]) - int(m_test_partA["fp"]),
        }

        # Persist per (run_id, strategy)
        out_dir = out_root / strategy
        _save_metrics_files(m_val, out_dir, "metrics_val")
        _save_metrics_files(m_test, out_dir, "metrics_test")
        info_path = out_dir / "threshold_info.json"
        with info_path.open("w", encoding="utf-8") as fh:
            json.dump(record, fh, indent=2)
        logger.info(f"Saved -> {info_path}")

        records.append(record)

    # 6) PR-AUC invariance summary
    pr_aucs = [r["test_pr_auc"] for r in records]
    pr_auc_diff = float(max(pr_aucs) - min(pr_aucs)) if pr_aucs else 0.0
    invariance = {
        "max_minus_min_test_pr_auc": pr_auc_diff,
        "tolerance": 1e-9,
        "passed": bool(pr_auc_diff < 1e-9),
        "part_a_pr_auc": partA_pr_auc,
    }
    logger.info(
        f"PR-AUC invariance check: max-min={pr_auc_diff:.3e} "
        f"(tolerance 1e-9) -> {'PASS' if invariance['passed'] else 'FAIL'}"
    )

    summary = {
        "selected_run_id": run_id,
        "note": (
            "Baseline strategy is invariant to target_prevalence; all three "
            "baseline runs (p001, p005, p010) produce bit-identical models. "
            "One run selected as representative."
        ),
        "strategies": list(strategies),
        "part_a_reference": part_a_reference,
        "pr_auc_invariance": invariance,
        "records": records,
    }

    if write_summary:
        results_dir = paths.outputs_dir.parent.parent / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        summary_path = results_dir / "part_b_multi_threshold_summary.json"
        with summary_path.open("w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)
        logger.info(f"Saved -> {summary_path}")

    # 7) Console summary
    logger.info("=" * 62)
    logger.info("PART B — MULTI-THRESHOLD SUMMARY (test set)")
    df = pd.DataFrame(
        [
            {
                "strategy": r["strategy_type"],
                "tau": r["threshold_value"],
                "precision": r["test_precision"],
                "recall": r["test_recall"],
                "f1": r["test_f1"],
                "f2": r["test_f2"],
                "fp": r["test_fp"],
                "delta_fp_vs_partA_f1opt": r["delta_fp_vs_part_a_f1opt"],
                "delta_f1_vs_partA_f1opt": r["delta_f1_vs_part_a_f1opt"],
                "pr_auc": r["test_pr_auc"],
            }
            for r in records
        ]
    )
    if not df.empty:
        with pd.option_context("display.width", 200, "display.max_colwidth", 80):
            print(df.to_string(index=False))
    logger.info("=" * 62)

    return summary


# ---------------------------------------------------------------------------
# Dry-run self-test (mock scores)
# ---------------------------------------------------------------------------

def dry_run() -> dict:
    """Run all three strategies on mock scores; assert PR-AUC invariance.

    Useful as a CI / smoke test that does not touch any real model or data.
    """
    rng = np.random.default_rng(0)
    y_val = rng.binomial(1, 0.001, 1000)
    y_test = rng.binomial(1, 0.001, 1000)
    s_val = rng.random(1000)
    s_test = rng.random(1000)

    grid = _build_threshold_grid(s_val, 1000)

    out: dict[str, dict[str, float]] = {}
    for s in VALID_STRATEGIES:
        tau, _details, _crit = _select_threshold_by_strategy(
            strategy_type=s, y_val=y_val, y_score_val=s_val, grid=grid,
        )
        m_test = compute_all_metrics(
            y_test, s_test, threshold=tau, split=f"dry_{s}"
        )
        required = {
            "pr_auc",
            "roc_auc",
            "precision",
            "recall",
            "f1",
            "f2",
            "tp",
            "fp",
            "tn",
            "fn",
            "threshold",
        }
        missing = required - set(m_test.keys())
        if missing:
            raise RuntimeError(f"Dry-run schema missing keys for {s}: {missing}")
        out[s] = m_test

    pr_aucs = [m["pr_auc"] for m in out.values()]
    spread = float(max(pr_aucs) - min(pr_aucs))
    if spread > 1e-9:
        raise RuntimeError(
            f"Dry-run PR-AUC invariance violated: spread={spread:.3e} > 1e-9"
        )
    logger.info(
        f"Dry-run OK: 3 strategies executed; PR-AUC spread = {spread:.3e}"
    )
    return {"results": out, "pr_auc_spread": spread}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--paths",
        type=str,
        default=None,
        help="Path to a custom paths.yaml (e.g. for Colab).",
    )
    parser.add_argument(
        "--strategies",
        type=str,
        nargs="+",
        default=list(VALID_STRATEGIES),
        choices=list(VALID_STRATEGIES),
        help="Subset of threshold strategies to run.",
    )
    parser.add_argument(
        "--precision-constraint",
        type=float,
        default=0.10,
        help="Minimum precision for the precision_constrained strategy.",
    )
    parser.add_argument(
        "--lambda-fp",
        type=float,
        default=0.05,
        help="Tiebreak utility coefficient (precision_constrained).",
    )
    parser.add_argument(
        "--n-dense",
        type=int,
        default=1000,
        help="Maximum size of the validation-quantile threshold grid.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run a self-test with mock scores (no real model/data needed).",
    )
    args = parser.parse_args()

    if args.dry_run:
        dry_run()
        return

    paths = PathConfig(args.paths) if args.paths else PathConfig()
    run_threshold_optimization(
        paths=paths,
        strategies=tuple(args.strategies),
        precision_constraint=args.precision_constraint,
        lambda_fp=args.lambda_fp,
        n_dense=args.n_dense,
    )


if __name__ == "__main__":
    main()
