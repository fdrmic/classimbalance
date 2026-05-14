"""Part B Strategy 6 runner: Part-A-Informed Hard-Negative Undersampling.

Orchestrates one or more PAI-HNU experiments:

    Load splits -> Load feature cache -> Load cached baseline scores
    -> PAI-HNU index selection -> Train XGBoost -> Evaluate on val/test
    -> Persist artefacts (model, metrics, run_config, sampling_manifest)

Run modes
---------
* **Full**: writes to ``paths.outputs_dir`` (configured per YAML).
* **Smoke** (``--sample-n-train K``): mini-end-to-end on a deterministic
  subsample of K training rows, persisted under
  ``outputs/runs_part_b_pai_hnu_smoke/`` so production artefacts are
  never touched.

Anti-leakage
------------
The validation/test feature caches and labels are loaded once, but are
NEVER passed to the sampler.  The sampler operates exclusively on
``y_train`` and the cached baseline training scores.

Usage
-----
    # Full (all three target prevalences):
    python -m aml_benchmark.experiments.run_part_b_pai_hnu \\
        --paths configs/paths_large_part_b_pai_hnu.yaml

    # Single target prevalence:
    python -m aml_benchmark.experiments.run_part_b_pai_hnu \\
        --paths configs/paths_large_part_b_pai_hnu.yaml \\
        --target-prevalences 0.005

    # Mini smoke test (deterministic subsample by row_idx):
    python -m aml_benchmark.experiments.run_part_b_pai_hnu \\
        --paths configs/paths_large_part_b_pai_hnu.yaml \\
        --target-prevalences 0.01 \\
        --sample-n-train 200000
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

from aml_benchmark.config import PathConfig, load_yaml
from aml_benchmark.evaluation.metrics import (
    compute_all_metrics,
    find_optimal_threshold,
    save_metrics,
)
from aml_benchmark.features.feature_cache import load_features
from aml_benchmark.models.factory import get_model
from aml_benchmark.sampling.hard_negative_undersampling import (
    PaiHnuSelection,
    build_pai_hnu_training_indices,
    load_baseline_score_cache,
    save_sampling_manifest,
    validate_no_overlap,
)
from aml_benchmark.utils.io import load_parquet
from aml_benchmark.utils.logging_utils import get_logger

logger = get_logger(__name__)


_DEFAULT_BENCHMARK_CFG = "benchmark_part_b_pai_hnu"
_SMOKE_OUTPUTS_DIR_NAME = "outputs/runs_part_b_pai_hnu_smoke"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _prevalence_tag(prevalence: float) -> str:
    return f"p{int(round(prevalence * 1000)):03d}"


def _file_sha256(path: Path, chunk: int = 4 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            data = fh.read(chunk)
            if not data:
                break
            h.update(data)
    return h.hexdigest()


def _load_benchmark_cfg(name: str) -> dict:
    project_root = Path(__file__).resolve().parents[3]
    p = project_root / "configs" / f"{name}.yaml"
    if not p.exists():
        raise FileNotFoundError(f"benchmark config not found: {p}")
    with p.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _resolve_baseline_model_path(
    paths: PathConfig,
    benchmark_cfg: dict,
    cli_path: str | None,
) -> Path | None:
    """Best-effort discovery for the manifest only.

    The runner does NOT need the model itself (scores are cached). It only
    records the resolved path in the manifest if available, so the user can
    audit later. Returns None silently if not found (the score cache is the
    source of truth).
    """
    if cli_path:
        p = Path(cli_path)
        return p if p.exists() else None
    if paths.baseline_model_path is not None and paths.baseline_model_path.exists():
        return paths.baseline_model_path
    preferred = benchmark_cfg.get("baseline", {}).get("preferred_run_id", "")
    if preferred:
        auto = paths.outputs_dir / preferred / "model.pkl"
        if auto.exists():
            return auto
    return None


# ---------------------------------------------------------------------------
# Subsample alignment (smoke mode)
# ---------------------------------------------------------------------------

_ROW_INDEX_MODE_INTERNAL = "internal_0_based_with_orig_row_idx_mapping"
_ROW_INDEX_MODE_FULL = "full_train_row_order"


def _subsample_aligned(
    X_train: np.ndarray,
    y_train: np.ndarray,
    score_df: pd.DataFrame,
    n_sample: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame, bool]:
    """Deterministic subsample on **sorted original ``row_idx``**.

    Returns
    -------
    X_sub, y_sub, scores_sub, mapping_df, did_subsample

    ``scores_sub`` has columns ``row_idx`` (internal 0..n-1), ``score``,
    ``orig_row_idx``.  ``mapping_df`` has ``internal_row_idx``, ``orig_row_idx``.

    ``did_subsample`` is False when ``n_sample >= n_train`` (full train, no draw).
    """
    n_total = int(len(y_train))
    if n_sample >= n_total:
        logger.info(
            f"--sample-n-train ({n_sample:,}) >= total ({n_total:,}); "
            "using full training split (no random subsample)."
        )
        sub = score_df.sort_values("row_idx", kind="stable").reset_index(drop=True)
        orig_row_idx_sorted = sub["row_idx"].to_numpy(dtype=np.int64)
        n = len(orig_row_idx_sorted)
        internal = np.arange(n, dtype=np.int64)
        mapping_df = pd.DataFrame(
            {
                "internal_row_idx": internal,
                "orig_row_idx": orig_row_idx_sorted,
            }
        )
        scores_sub = pd.DataFrame(
            {
                "row_idx": internal,
                "score": sub["score"].to_numpy(dtype=np.float32),
                "orig_row_idx": orig_row_idx_sorted,
            }
        )
        assert len(scores_sub) == len(mapping_df) == n_total
        assert np.array_equal(scores_sub["orig_row_idx"].to_numpy(), orig_row_idx_sorted)
        assert np.array_equal(X_train, X_train[orig_row_idx_sorted])
        assert np.array_equal(y_train, y_train[orig_row_idx_sorted])
        return X_train, y_train, scores_sub, mapping_df, False

    rng = np.random.default_rng(seed)
    all_row_idx = score_df["row_idx"].to_numpy(dtype=np.int64)
    if len(all_row_idx) != n_total:
        raise ValueError(
            f"Score cache rows ({len(all_row_idx):,}) != y_train rows "
            f"({n_total:,}). Re-run score_baseline_train --overwrite."
        )

    picked = rng.choice(all_row_idx, size=int(n_sample), replace=False)
    orig_row_idx_sorted = np.sort(picked)

    X_sub = X_train[orig_row_idx_sorted]
    y_sub = y_train[orig_row_idx_sorted]

    score_by_idx = score_df.set_index("row_idx")["score"]
    scores_ordered = score_by_idx.loc[orig_row_idx_sorted].to_numpy(dtype=np.float32)

    n = len(orig_row_idx_sorted)
    internal = np.arange(n, dtype=np.int64)
    scores_sub = pd.DataFrame(
        {
            "row_idx": internal,
            "score": scores_ordered,
            "orig_row_idx": orig_row_idx_sorted,
        }
    )
    mapping_df = pd.DataFrame(
        {
            "internal_row_idx": internal,
            "orig_row_idx": orig_row_idx_sorted,
        }
    )

    if not (
        len(X_sub) == len(y_sub) == len(scores_sub) == len(orig_row_idx_sorted)
    ):
        raise AssertionError(
            "Subsample alignment failed: length mismatch "
            f"(X={len(X_sub)}, y={len(y_sub)}, scores={len(scores_sub)})."
        )
    if not np.array_equal(scores_sub["orig_row_idx"].to_numpy(), orig_row_idx_sorted):
        raise AssertionError(
            "Subsample alignment failed: scores_sub.orig_row_idx != sorted pick list."
        )
    if not np.array_equal(X_sub, X_train[orig_row_idx_sorted]):
        raise AssertionError(
            "Subsample alignment failed: X_sub != X_train[orig_row_idx_sorted]."
        )
    if not np.array_equal(y_sub, y_train[orig_row_idx_sorted]):
        raise AssertionError(
            "Subsample alignment failed: y_sub != y_train[orig_row_idx_sorted]."
        )

    n_pos_sub = int(y_sub.sum())
    if n_pos_sub < 1:
        raise RuntimeError(
            f"Smoke subsample contains zero positives (n_sample={n_sample:,}). "
            "Either increase --sample-n-train or use a different seed; consider "
            "stratified subsampling for very small smoke runs."
        )
    if np.isnan(scores_ordered).any():
        raise RuntimeError("Smoke subsample baseline scores contain NaN.")

    logger.info(
        f"Subsample (orig_row_idx-sorted, seed={seed}): "
        f"n={len(y_sub):,}, n_pos={n_pos_sub:,}, "
        f"prevalence={n_pos_sub / len(y_sub):.4%}"
    )
    return X_sub, y_sub, scores_sub, mapping_df, True


# ---------------------------------------------------------------------------
# Single experiment
# ---------------------------------------------------------------------------

def _run_one_pai_hnu_experiment(
    *,
    target_prevalence: float,
    X_train: np.ndarray,
    y_train: np.ndarray,
    baseline_scores: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    paths: PathConfig,
    benchmark_cfg: dict,
    score_cache_path: Path,
    score_cache_sha: str,
    score_meta: dict,
    baseline_model_path_str: str,
    random_seed: int,
    output_root: Path,
    threshold_criterion: str,
    is_smoke: bool,
    extra_manifest: dict | None = None,
    smoke_alignment_meta: dict | None = None,
    subsample_mapping_df: pd.DataFrame | None = None,
) -> dict:
    """Run one PAI-HNU experiment for a single target prevalence."""
    t_start = time.perf_counter()

    sampling_cfg = benchmark_cfg["sampling"]
    baseline_cfg = benchmark_cfg.get("baseline", {})

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    smoke_tag = "_SMOKE" if is_smoke else ""
    run_id = f"xgboost__pai_hnu__{_prevalence_tag(target_prevalence)}{smoke_tag}__{ts}"
    output_dir = output_root / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    align_meta: dict = dict(smoke_alignment_meta or {})
    map_path = output_dir / "subsample_row_mapping.parquet"
    if subsample_mapping_df is not None:
        subsample_mapping_df.to_parquet(map_path, index=False)
        align_meta["subsample_row_mapping_parquet"] = str(map_path.resolve())

    logger.info("=" * 62)
    logger.info(f"PAI-HNU EXPERIMENT: {run_id}")
    logger.info(f"  target_prevalence : {target_prevalence:.4%}")
    logger.info(f"  output_dir        : {output_dir}")
    logger.info(f"  is_smoke          : {is_smoke}")
    logger.info("=" * 62)

    selection: PaiHnuSelection = build_pai_hnu_training_indices(
        y_train=y_train,
        baseline_scores=baseline_scores,
        target_prevalence=target_prevalence,
        hard_negative_share=float(sampling_cfg["hard_negative_share"]),
        temporal_random_share=float(sampling_cfg["temporal_random_share"]),
        global_random_share=float(sampling_cfg["global_random_share"]),
        n_temporal_blocks=int(sampling_cfg["n_temporal_blocks"]),
        hard_negative_cap_multiplier=int(
            sampling_cfg["hard_negative_cap_multiplier"]
        ),
        fill_shortfall_from_global=bool(
            sampling_cfg.get("fill_shortfall_from_global", True)
        ),
        random_state=random_seed,
    )

    # Paranoid no-overlap check (covers the hot path)
    validate_no_overlap(
        selection.pos_idx,
        selection.hard_neg_idx,
        selection.temporal_neg_idx,
        selection.global_neg_idx,
    )

    train_idx = selection.all_idx
    rng_shuffle = np.random.default_rng(random_seed + 1)
    rng_shuffle.shuffle(train_idx)
    X_train_sub = X_train[train_idx]
    y_train_sub = y_train[train_idx]
    logger.info(
        f"PAI-HNU training set: rows={len(y_train_sub):,}, "
        f"prevalence={selection.achieved_prevalence:.6%}"
    )

    manifest_extra: dict = {**(extra_manifest or {}), **align_meta}
    save_sampling_manifest(
        selection=selection,
        output_path=output_dir / "sampling_manifest.json",
        target_prevalence=target_prevalence,
        sampling_shares={
            "hard_negative_share": float(sampling_cfg["hard_negative_share"]),
            "temporal_random_share": float(sampling_cfg["temporal_random_share"]),
            "global_random_share": float(sampling_cfg["global_random_share"]),
            "n_temporal_blocks": int(sampling_cfg["n_temporal_blocks"]),
            "hard_negative_cap_multiplier": int(
                sampling_cfg["hard_negative_cap_multiplier"]
            ),
        },
        random_seed=random_seed,
        baseline_run_id=baseline_cfg.get("preferred_run_id", ""),
        baseline_model_path=baseline_model_path_str,
        score_cache_path=str(score_cache_path),
        score_cache_sha256=score_cache_sha,
        n_train_total=int(len(y_train)),
        score_meta=score_meta,
        extra=manifest_extra,
    )

    use_factory_defaults = bool(
        benchmark_cfg.get("xgboost", {}).get("use_factory_defaults", True)
    )
    class_weight_cfg = benchmark_cfg.get("xgboost", {}).get("class_weight", None)
    if not use_factory_defaults:
        logger.warning(
            "use_factory_defaults=False is not implemented; falling back to defaults."
        )
    model = get_model("xgboost", random_state=random_seed, class_weight=class_weight_cfg)
    logger.info(f"Training XGBoost on {len(y_train_sub):,} PAI-HNU rows ...")
    t_train = time.perf_counter()
    model.fit(X_train_sub, y_train_sub)
    train_sec = time.perf_counter() - t_train
    logger.info(f"Training done in {train_sec:.1f}s")

    val_score = model.predict_proba(X_val)[:, 1]
    test_score = model.predict_proba(X_test)[:, 1]

    default_th = float(benchmark_cfg["evaluation"]["default_threshold"])
    m_val_default = compute_all_metrics(y_val, val_score, threshold=default_th, split="val")
    m_test_default = compute_all_metrics(y_test, test_score, threshold=default_th, split="test")

    optimal_th = find_optimal_threshold(
        y_val, val_score, criterion=threshold_criterion
    )
    m_val_opt = compute_all_metrics(y_val, val_score, threshold=optimal_th, split="val_opt")
    m_test_opt = compute_all_metrics(y_test, test_score, threshold=optimal_th, split="test_opt")

    save_metrics(m_val_default, output_dir=output_dir, split="val")
    save_metrics(m_test_default, output_dir=output_dir, split="test")
    save_metrics(m_val_opt, output_dir=output_dir, split="val_opt")
    save_metrics(m_test_opt, output_dir=output_dir, split="test_opt")

    joblib.dump(model, output_dir / "model.pkl")
    elapsed = time.perf_counter() - t_start

    run_config = {
        "run_id": run_id,
        "model": "xgboost",
        "strategy": "pai_hnu",
        "target_prevalence": target_prevalence,
        "achieved_train_prevalence": round(selection.achieved_prevalence, 8),
        "random_seed": random_seed,
        "is_smoke": is_smoke,
        "smoke_subsample_used": align_meta.get("smoke_subsample_used", False),
        "sample_n_train": align_meta.get("sample_n_train"),
        "row_index_mode": align_meta.get(
            "row_index_mode", _ROW_INDEX_MODE_FULL
        ),
        "subsample_row_mapping_parquet": align_meta.get(
            "subsample_row_mapping_parquet"
        ),
        "train_rows_after_sampling": int(len(y_train_sub)),
        "train_positives_after_sampling": int(selection.counts["n_pos"]),
        "train_negatives_after_sampling": int(selection.counts["n_total_neg_actual"]),
        "selection_counts": selection.counts,
        "val_rows": int(len(y_val)),
        "val_positives": int(y_val.sum()),
        "test_rows": int(len(y_test)),
        "test_positives": int(y_test.sum()),
        "train_time_sec": round(train_sec, 2),
        "total_time_sec": round(elapsed, 2),
        "default_threshold": default_th,
        "optimal_threshold_val": float(optimal_th),
        "threshold_criterion": threshold_criterion,
        "baseline_score_cache_sha256": score_cache_sha,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    with (output_dir / "run_config.json").open("w", encoding="utf-8") as fh:
        json.dump(run_config, fh, indent=2)
    logger.info(f"Run config saved -> {output_dir / 'run_config.json'}")

    print()
    print("=" * 62)
    print(f"  PAI-HNU COMPLETE: {run_id}")
    print("=" * 62)
    for label, m in [
        ("val  @0.5", m_val_default),
        ("test @0.5", m_test_default),
        (f"val  @opt({optimal_th:.4f})", m_val_opt),
        (f"test @opt({optimal_th:.4f})", m_test_opt),
    ]:
        print(
            f"  [{label}] pr_auc={m['pr_auc']:.6f} "
            f"P={m['precision']:.4f} R={m['recall']:.4f} "
            f"F1={m['f1']:.4f} F2={m['f2']:.4f} "
            f"TP={int(m['tp'])} FP={int(m['fp'])} FN={int(m['fn'])}"
        )
    print(f"  Elapsed: {elapsed:.1f}s    Output: {output_dir}")
    print("=" * 62)
    print()

    return {
        "run_id": run_id,
        "output_dir": str(output_dir),
        "val_default": m_val_default,
        "test_default": m_test_default,
        "val_opt": m_val_opt,
        "test_opt": m_test_opt,
        "selection_counts": selection.counts,
        "optimal_threshold_val": float(optimal_th),
    }


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------

def run(
    paths: PathConfig,
    benchmark_cfg: dict,
    *,
    target_prevalences: list[float] | None = None,
    sample_n_train: int | None = None,
    baseline_model_path_cli: str | None = None,
) -> list[dict]:
    """Run all configured PAI-HNU experiments. Returns one dict per run."""
    paths.validate_splits()

    exp_cfg = load_yaml("experiment")
    random_seed = int(exp_cfg.get("random_seed", 42))
    threshold_criterion = str(
        benchmark_cfg.get("evaluation", {}).get("optimal_threshold_criterion", "f1")
    )

    targets = target_prevalences or list(benchmark_cfg["target_prevalences"])
    is_smoke = sample_n_train is not None and sample_n_train > 0

    score_cache_name = benchmark_cfg.get("baseline", {}).get(
        "score_cache_filename", "baseline_train_scores.parquet"
    )
    score_meta_name = benchmark_cfg.get("baseline", {}).get(
        "score_meta_filename", "baseline_train_scores_meta.json"
    )
    score_cache_path = paths.splits_dir / score_cache_name
    score_meta_path = paths.splits_dir / score_meta_name
    if not score_cache_path.exists():
        raise FileNotFoundError(
            f"Baseline score cache missing: {score_cache_path}\n"
            "Run first: python -m aml_benchmark.experiments.score_baseline_train "
            f"--paths configs/paths_large_part_b_pai_hnu.yaml"
        )
    score_cache_sha = _file_sha256(score_cache_path)
    score_meta: dict = {}
    if score_meta_path.exists():
        with score_meta_path.open(encoding="utf-8") as fh:
            score_meta = json.load(fh)

    score_df = load_baseline_score_cache(score_cache_path)

    logger.info("Loading splits and feature caches ...")
    train_df = load_parquet(paths.train_split)
    y_train_full = train_df["label"].to_numpy(dtype=np.int8)
    del train_df
    X_train_full = load_features(paths.splits_dir, "train")

    val_df = load_parquet(paths.val_split)
    y_val = val_df["label"].to_numpy(dtype=np.int8)
    del val_df
    X_val = load_features(paths.splits_dir, "val")

    test_df = load_parquet(paths.test_split)
    y_test = test_df["label"].to_numpy(dtype=np.int8)
    del test_df
    X_test = load_features(paths.splits_dir, "test")

    if len(score_df) != len(y_train_full):
        raise ValueError(
            f"Score cache rows ({len(score_df):,}) != train rows "
            f"({len(y_train_full):,}). Run score_baseline_train --overwrite."
        )

    # Smoke subsample (orig_row_idx–aligned) or full train
    extra_manifest_base: dict | None = None
    subsample_mapping_df: pd.DataFrame | None = None
    if is_smoke:
        (
            X_train,
            y_train,
            score_df,
            subsample_mapping_df,
            did_subsample,
        ) = _subsample_aligned(
            X_train_full,
            y_train_full,
            score_df,
            n_sample=int(sample_n_train),
            seed=random_seed,
        )
        extra_manifest_base = {
            "smoke": True,
            "smoke_subsample_seed": random_seed,
            "smoke_sample_n_train": int(sample_n_train),
        }
        smoke_alignment_meta = {
            "smoke_subsample_used": bool(did_subsample),
            "sample_n_train": int(sample_n_train),
            "row_index_mode": (
                _ROW_INDEX_MODE_INTERNAL
                if did_subsample
                else _ROW_INDEX_MODE_FULL
            ),
        }
        output_root = Path(__file__).resolve().parents[3] / _SMOKE_OUTPUTS_DIR_NAME
    else:
        X_train, y_train = X_train_full, y_train_full
        smoke_alignment_meta = {
            "smoke_subsample_used": False,
            "sample_n_train": None,
            "row_index_mode": _ROW_INDEX_MODE_FULL,
            "subsample_row_mapping_parquet": None,
        }
        output_root = paths.outputs_dir

    output_root.mkdir(parents=True, exist_ok=True)
    baseline_scores = score_df["score"].to_numpy(dtype=np.float32)

    baseline_model_path = _resolve_baseline_model_path(
        paths, benchmark_cfg, baseline_model_path_cli
    )
    baseline_model_path_str = str(baseline_model_path) if baseline_model_path else ""

    runs = []
    for tp in targets:
        result = _run_one_pai_hnu_experiment(
            target_prevalence=float(tp),
            X_train=X_train,
            y_train=y_train,
            baseline_scores=baseline_scores,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            paths=paths,
            benchmark_cfg=benchmark_cfg,
            score_cache_path=score_cache_path,
            score_cache_sha=score_cache_sha,
            score_meta=score_meta,
            baseline_model_path_str=baseline_model_path_str,
            random_seed=random_seed,
            output_root=output_root,
            threshold_criterion=threshold_criterion,
            is_smoke=is_smoke,
            extra_manifest=extra_manifest_base,
            smoke_alignment_meta=smoke_alignment_meta,
            subsample_mapping_df=subsample_mapping_df if is_smoke else None,
        )
        runs.append(result)

    return runs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--paths", type=str, default=None,
        help="Optional paths YAML (defaults to configs/paths.yaml).",
    )
    parser.add_argument(
        "--benchmark", type=str, default=_DEFAULT_BENCHMARK_CFG,
        help="Benchmark config stem under configs/.",
    )
    parser.add_argument(
        "--target-prevalences", type=str, default=None,
        help="Comma-separated overrides, e.g. '0.001,0.005' (default: from YAML).",
    )
    parser.add_argument(
        "--sample-n-train", type=int, default=None,
        help=(
            "Mini-end-to-end smoke: deterministically subsample K training rows "
            "(seed=42, row_idx-aligned). Output goes to outputs/runs_part_b_pai_hnu_smoke/."
        ),
    )
    parser.add_argument(
        "--baseline-model-path", type=str, default=None,
        help="Optional override for the baseline model path (recorded in manifest).",
    )
    args = parser.parse_args()

    try:
        paths = PathConfig(args.paths) if args.paths else PathConfig()
        benchmark_cfg = _load_benchmark_cfg(args.benchmark)
        targets = (
            [float(x) for x in args.target_prevalences.split(",")]
            if args.target_prevalences
            else None
        )

        runs = run(
            paths=paths,
            benchmark_cfg=benchmark_cfg,
            target_prevalences=targets,
            sample_n_train=args.sample_n_train,
            baseline_model_path_cli=args.baseline_model_path,
        )
        print(f"\nFinished {len(runs)} PAI-HNU run(s).")
        for r in runs:
            print(f"  - {r['run_id']} -> {r['output_dir']}")
    except FileNotFoundError as exc:
        logger.error(str(exc))
        sys.exit(1)
    except Exception as exc:
        logger.exception(f"run_part_b_pai_hnu failed: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
