"""Part A benchmark grid runner.

Executes the full 5 x 2 x 3 = 30-condition benchmark:

    strategies x models x target_prevalences = 30 runs

Each condition is an independent call to
:func:`aml_benchmark.experiments.runner.run_experiment`.
The grid configuration is read from ``configs/benchmark.yaml``.

Run ordering
------------
Conditions are iterated in the order defined in ``benchmark.yaml``
(outermost: strategy; middle: model; innermost: prevalence) so that
slower strategies (SMOTE, ADASYN) are grouped together and the user
can monitor progress predictably.

Failure handling
----------------
If one condition raises an exception (e.g. an edge-case ADASYN failure),
it is logged and the grid continues with the remaining conditions.  At
the end a summary of passed and failed runs is printed.

Usage
-----
    python -m aml_benchmark.experiments.grid_runner
"""
from __future__ import annotations

import sys
import time
import traceback
from datetime import datetime

from aml_benchmark.config import PathConfig, load_yaml
from aml_benchmark.experiments.runner import _prevalence_tag, run_experiment
from aml_benchmark.utils.logging_utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Grid runner
# ---------------------------------------------------------------------------

def run_grid(paths: PathConfig | None = None) -> None:
    """Execute the full Part A benchmark grid.

    Reads ``configs/benchmark.yaml`` for the list of models, strategies,
    and target prevalence levels.  Calls :func:`run_experiment` for each
    combination and collects pass/fail status.

    Parameters
    ----------
    paths:
        Optional pre-built :class:`~aml_benchmark.config.PathConfig`.
        If ``None``, a fresh instance is built from configs.
    """
    if paths is None:
        paths = PathConfig()

    paths.validate_splits()

    cfg = load_yaml("benchmark")
    models: list[str] = cfg["models"]
    strategies: list[str] = cfg["strategies"]
    prevalences: list[float] = [float(p) for p in cfg["target_prevalences"]]

    total = len(models) * len(strategies) * len(prevalences)
    logger.info("=" * 62)
    logger.info("PART A BENCHMARK GRID")
    logger.info(f"  Models      : {models}")
    logger.info(f"  Strategies  : {strategies}")
    logger.info(f"  Prevalences : {[f'{p:.3%}' for p in prevalences]}")
    logger.info(f"  Total runs  : {total}")
    logger.info("=" * 62)

    passed: list[str] = []
    failed: list[tuple[str, str]] = []  # (run_id, error_summary)

    t_grid_start = time.perf_counter()
    run_count = 0

    for strategy in strategies:
        for model_name in models:
            for target_prevalence in prevalences:
                run_count += 1
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                ptag = _prevalence_tag(target_prevalence)
                run_id = f"{model_name}__{strategy}__{ptag}__{ts}"

                logger.info(
                    f"[{run_count}/{total}] "
                    f"model={model_name}  strategy={strategy}  "
                    f"prevalence={target_prevalence:.3%}"
                )

                try:
                    run_experiment(
                        model_name=model_name,
                        strategy=strategy,
                        target_prevalence=target_prevalence,
                        run_id=run_id,
                        paths=paths,
                    )
                    passed.append(run_id)

                except Exception as exc:
                    error_msg = f"{type(exc).__name__}: {exc}"
                    logger.error(
                        f"Run FAILED: {run_id}\n{traceback.format_exc()}"
                    )
                    failed.append((run_id, error_msg))

    elapsed = time.perf_counter() - t_grid_start
    _print_grid_summary(passed, failed, elapsed)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def _print_grid_summary(
    passed: list[str],
    failed: list[tuple[str, str]],
    elapsed: float,
) -> None:
    total = len(passed) + len(failed)
    print()
    print("=" * 62)
    print("  PART A BENCHMARK GRID - COMPLETE")
    print("=" * 62)
    print(f"  Total runs   : {total}")
    print(f"  Passed       : {len(passed)}")
    print(f"  Failed       : {len(failed)}")
    print(f"  Elapsed      : {elapsed / 60:.1f} min")
    if failed:
        print()
        print("  FAILED RUNS:")
        for run_id, err in failed:
            print(f"    {run_id}")
            print(f"      {err}")
    print("=" * 62)
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", type=str, default=None,
                        help="Path to a custom paths.yaml (e.g. for Colab)")
    args = parser.parse_args()
    try:
        paths = PathConfig(args.paths) if args.paths else PathConfig()
        run_grid(paths=paths)
    except FileNotFoundError as exc:
        logger.error(str(exc))
        sys.exit(1)
    except Exception as exc:
        logger.exception(f"Grid runner failed: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
