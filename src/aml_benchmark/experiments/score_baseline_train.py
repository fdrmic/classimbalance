"""One-shot scorer that produces baseline_train_scores.parquet for PAI-HNU.

Why this script exists
----------------------
Strategy 6 (PAI-HNU) selects "hard negatives" from the training split using
risk scores from the Part-A XGBoost Baseline. Computing predict_proba on
~123 M training rows is the dominant cost (~10–15 min on GPU,
~60+ min on CPU), so we cache the scores once and reuse them for all three
target prevalence runs.

Resolution order for the baseline model
---------------------------------------
1. CLI argument --baseline-model-path (highest priority; ad-hoc)
2. Field ``baseline_model_path`` in the active paths YAML
3. Auto-discovery: ``paths.outputs_dir / preferred_run_id / "model.pkl"``
4. Hard error with explicit guidance (see _missing_model_error_message)

Anti-leakage
------------
This script reads ONLY the training feature cache (``train_features_v2.parquet``)
and writes ``baseline_train_scores.parquet`` containing one score per training
row. Validation and test data are never touched.  This is enforced by
construction: only ``load_features(splits_dir, "train")`` is called.

Usage
-----
    python -m aml_benchmark.experiments.score_baseline_train \\
        --paths configs/paths_large_part_b_pai_hnu.yaml

    # Override model path explicitly (no need to edit YAML):
    python -m aml_benchmark.experiments.score_baseline_train \\
        --paths configs/paths_large_part_b_pai_hnu.yaml \\
        --baseline-model-path /content/drive/.../model.pkl

    # Force rebuild (overwrite cache):
    python -m aml_benchmark.experiments.score_baseline_train \\
        --paths configs/paths_large_part_b_pai_hnu.yaml --overwrite
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

from aml_benchmark.config import PathConfig
from aml_benchmark.features.feature_cache import load_features
from aml_benchmark.utils.logging_utils import get_logger

logger = get_logger(__name__)


_DEFAULT_BENCHMARK_CFG = "benchmark_part_b_pai_hnu"
_CHUNK_SIZE = 5_000_000


# ---------------------------------------------------------------------------
# Device detection (mirrors models/factory.py:_detect_xgb_device)
# ---------------------------------------------------------------------------

def _detect_runtime_device() -> str:
    """Return 'cuda' if nvidia-smi succeeds, else 'cpu'."""
    import subprocess
    try:
        subprocess.check_output(["nvidia-smi"], stderr=subprocess.DEVNULL)
        return "cuda"
    except Exception:
        return "cpu"


# ---------------------------------------------------------------------------
# Baseline model resolution
# ---------------------------------------------------------------------------

def _missing_model_error_message(
    cli_path: str | None,
    yaml_path: Path | None,
    auto_discovery_path: Path,
) -> str:
    return (
        "ERROR: Part-A baseline model not found.\n"
        "\n"
        "Searched in this order:\n"
        f"  1. --baseline-model-path argument: {cli_path or '<not provided>'}\n"
        f"  2. configs YAML baseline_model_path: {yaml_path or '<empty>'}\n"
        f"  3. Auto-discovery: {auto_discovery_path}  (NOT FOUND)\n"
        "\n"
        "To resolve, choose ONE of:\n"
        "  (a) Mount Drive and ensure the model file exists at the expected path.\n"
        "  (b) Set baseline_model_path in your paths YAML to the absolute path\n"
        "      of an existing baseline model.\n"
        "  (c) Pass --baseline-model-path /path/to/model.pkl on the CLI.\n"
        "  (d) As an explicit fallback, retrain the baseline locally with:\n"
        "        python -m aml_benchmark.experiments.score_baseline_train --retrain-baseline\n"
        "      WARNING: Retraining is NOT the default. Only use this if no original\n"
        "      Part-A baseline model is available.\n"
    )


def _resolve_baseline_model_path(
    paths: PathConfig,
    benchmark_cfg: dict,
    cli_path: str | None,
) -> Path:
    """Apply the documented resolution order; raise FileNotFoundError if all miss."""
    if cli_path:
        p = Path(cli_path)
        if not p.exists():
            raise FileNotFoundError(
                f"--baseline-model-path does not exist: {p}"
            )
        logger.info(f"Baseline model resolved from CLI override: {p}")
        return p

    if paths.baseline_model_path is not None:
        if paths.baseline_model_path.exists():
            logger.info(
                f"Baseline model resolved from paths YAML: {paths.baseline_model_path}"
            )
            return paths.baseline_model_path
        logger.warning(
            f"baseline_model_path in YAML does not exist: {paths.baseline_model_path}; "
            "falling back to auto-discovery."
        )

    preferred = benchmark_cfg.get("baseline", {}).get("preferred_run_id", "")
    auto = paths.outputs_dir / preferred / "model.pkl"
    if auto.exists():
        logger.info(f"Baseline model resolved by auto-discovery: {auto}")
        return auto

    raise FileNotFoundError(
        _missing_model_error_message(
            cli_path=cli_path,
            yaml_path=paths.baseline_model_path,
            auto_discovery_path=auto,
        )
    )


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _cache_paths(paths: PathConfig, benchmark_cfg: dict) -> tuple[Path, Path]:
    cache_name = benchmark_cfg.get("baseline", {}).get(
        "score_cache_filename", "baseline_train_scores.parquet"
    )
    meta_name = benchmark_cfg.get("baseline", {}).get(
        "score_meta_filename", "baseline_train_scores_meta.json"
    )
    return paths.splits_dir / cache_name, paths.splits_dir / meta_name


def _file_sha256(path: Path, chunk: int = 4 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            data = fh.read(chunk)
            if not data:
                break
            h.update(data)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Core scoring
# ---------------------------------------------------------------------------

def _predict_in_chunks(model, X: np.ndarray) -> np.ndarray:
    """Call predict_proba in chunks of _CHUNK_SIZE rows (progress logging)."""
    n = X.shape[0]
    out = np.empty(n, dtype=np.float32)
    for start in range(0, n, _CHUNK_SIZE):
        end = min(n, start + _CHUNK_SIZE)
        t0 = time.perf_counter()
        out[start:end] = model.predict_proba(X[start:end])[:, 1].astype(
            np.float32, copy=False
        )
        logger.info(
            f"  scored rows {start:>11,} .. {end:>11,} of {n:,} "
            f"in {time.perf_counter() - t0:.1f}s"
        )
    return out


def score_baseline_on_train(
    paths: PathConfig,
    benchmark_cfg: dict,
    cli_baseline_path: str | None = None,
    overwrite: bool = False,
) -> Path:
    """Compute (or load) baseline_train_scores.parquet. Returns its path."""
    cache_path, meta_path = _cache_paths(paths, benchmark_cfg)

    if cache_path.exists() and not overwrite:
        logger.info(f"Cache exists, skipping recomputation: {cache_path}")
        return cache_path

    model_path = _resolve_baseline_model_path(paths, benchmark_cfg, cli_baseline_path)

    runtime = _detect_runtime_device()
    logger.info(f"Runtime device detected: {runtime}")

    logger.info(f"Loading baseline model: {model_path}")
    model = joblib.load(model_path)

    # ANTI-LEAKAGE (AL1): only the training feature cache is read here.
    #                     val_features_v2.parquet and test_features_v2.parquet
    #                     are NEVER opened in this script.
    logger.info(f"Loading training features from cache: {paths.splits_dir}")
    X_train = load_features(paths.splits_dir, "train")
    logger.info(f"X_train shape: {X_train.shape}")

    try:
        model_device = str(model.get_params().get("device", "unknown"))
    except Exception:
        model_device = "unknown"
    logger.info(f"Model device parameter: {model_device}")

    if model_device == "cuda" and runtime == "cpu":
        logger.warning(
            "Loaded model was trained with device='cuda' but no GPU detected "
            "in this runtime. predict_proba may fail or fall back silently to "
            "CPU. Scoring 123M rows on CPU may take 60+ minutes."
        )
    elif runtime == "cpu":
        logger.warning(
            "No GPU detected. Scoring on CPU may take 60+ minutes."
        )
    else:
        logger.info("GPU detected; scoring on cuda.")

    # The model is used as-is; we never modify trained weights or hyperparameters.
    t_score = time.perf_counter()
    scores = _predict_in_chunks(model, X_train)
    score_sec = time.perf_counter() - t_score
    logger.info(f"Scored {len(scores):,} rows in {score_sec / 60:.2f} min")

    df = pd.DataFrame(
        {
            "row_idx": np.arange(len(scores), dtype=np.int64),
            "score": scores,
        }
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, index=False)
    logger.info(f"Wrote baseline scores -> {cache_path}")

    sha = _file_sha256(cache_path)
    meta = {
        "source_run_id": benchmark_cfg.get("baseline", {}).get(
            "preferred_run_id", ""
        ),
        "model_path": str(model_path),
        "model_device_param": model_device,
        "runtime_device": runtime,
        "n_rows": int(len(scores)),
        "score_min": float(scores.min()),
        "score_max": float(scores.max()),
        "score_mean": float(scores.mean()),
        "score_dtype": str(scores.dtype),
        "scoring_time_sec": round(score_sec, 2),
        "sha256_score_file": sha,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    with meta_path.open("w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    logger.info(f"Wrote baseline score meta -> {meta_path}")

    return cache_path


# ---------------------------------------------------------------------------
# Retrain fallback (explicit only)
# ---------------------------------------------------------------------------

def _retrain_baseline_locally(paths: PathConfig) -> Path:
    """Explicit fallback: train xgboost baseline locally, save into outputs_dir.

    NOT a default code path. Only used when --retrain-baseline is passed.
    Bit-identical reproduction is only guaranteed when XGBoost-hist + same
    seed + same library versions match the original Part-A run.
    """
    from aml_benchmark.config import load_yaml
    from aml_benchmark.models.factory import get_model
    from aml_benchmark.utils.io import load_parquet

    logger.warning(
        "Retraining baseline locally. This is an explicit fallback, NOT the "
        "default. Bit-identical reproduction depends on identical library "
        "versions and runtime conditions to the original Part-A run."
    )
    seed = int(load_yaml("experiment").get("random_seed", 42))

    X_train = load_features(paths.splits_dir, "train")
    train_df = load_parquet(paths.train_split)
    y_train = train_df["label"].to_numpy(dtype=int)
    del train_df

    model = get_model("xgboost", random_state=seed, class_weight=None)
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    logger.info(
        f"Local baseline retrain done in {(time.perf_counter() - t0) / 60:.2f} min"
    )

    out_dir = paths.outputs_dir / f"xgboost__baseline__retrain__{datetime.now():%Y%m%d_%H%M%S}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "model.pkl"
    joblib.dump(model, out_path)
    logger.info(f"Local baseline saved -> {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _load_benchmark_cfg(name: str) -> dict:
    project_root = Path(__file__).resolve().parents[3]
    p = project_root / "configs" / f"{name}.yaml"
    if not p.exists():
        raise FileNotFoundError(f"benchmark config not found: {p}")
    with p.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--paths", type=str, default=None,
        help="Path to a paths YAML (e.g. configs/paths_large_part_b_pai_hnu.yaml).",
    )
    parser.add_argument(
        "--benchmark", type=str, default=_DEFAULT_BENCHMARK_CFG,
        help="Benchmark config stem under configs/ (default: benchmark_part_b_pai_hnu).",
    )
    parser.add_argument(
        "--baseline-model-path", type=str, default=None,
        help="Explicit override for the baseline model path (highest priority).",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Recompute baseline_train_scores.parquet even if cache exists.",
    )
    parser.add_argument(
        "--retrain-baseline", action="store_true",
        help="EXPLICIT FALLBACK: train baseline locally (not the default).",
    )
    args = parser.parse_args()

    try:
        paths = PathConfig(args.paths) if args.paths else PathConfig()
        benchmark_cfg = _load_benchmark_cfg(args.benchmark)

        if args.retrain_baseline:
            new_model_path = _retrain_baseline_locally(paths)
            score_baseline_on_train(
                paths=paths,
                benchmark_cfg=benchmark_cfg,
                cli_baseline_path=str(new_model_path),
                overwrite=True,
            )
        else:
            score_baseline_on_train(
                paths=paths,
                benchmark_cfg=benchmark_cfg,
                cli_baseline_path=args.baseline_model_path,
                overwrite=args.overwrite,
            )
    except FileNotFoundError as exc:
        logger.error(str(exc))
        sys.exit(1)
    except Exception as exc:
        logger.exception(f"score_baseline_train failed: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
