"""
Extract F2-scores from per-run metrics JSONs and add them to the Part A summary CSV.

Google Colab (Drive mounted at /content/drive), matching aml_results layout::

    python extract_f2_from_runs.py \\
        --summary-csv /content/drive/MyDrive/aml_results/part_a_summary_v2.csv \\
        --aml-results /content/drive/MyDrive/aml_results \\
        --output-csv /content/drive/MyDrive/aml_results/part_a_summary_v2_with_f2.csv

Or pass each ``.../large_run_v2_<batch>/runs`` folder explicitly::

    python extract_f2_from_runs.py \\
        --summary-csv /path/to/part_a_summary_v2.csv \\
        --runs-roots /path/to/large_run_v2_ongoing/runs /path/to/large_run_v2_20260404_1637/runs \\
        --output-csv /path/to/part_a_summary_v2_with_f2.csv

When combining ``--aml-results`` and ``--runs-roots``, explicit roots are appended
after the auto-discovered paths (duplicates removed).

Colab / shell tip: if you break lines with ``\\``, do not indent the next line — leading
spaces become part of the path. This script strips whitespace on all path arguments to
avoid that pitfall.

Notes:
    - Tries multiple JSON filenames (metrics_test_thresh.json, metrics_test_opt.json,
      metrics_test.json) because outputs may differ between pipelines.
    - F2 is read from JSON; verification compares to F2 from precision/recall at the
      F1-optimal threshold (should match thresh JSON).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

# Possible filenames where F2 at the F1-optimal threshold might be stored.
_CANDIDATE_FILENAMES_TEST_THRESH = [
    "metrics_test_thresh.json",
    "metrics_test_opt.json",
]
_CANDIDATE_FILENAMES_TEST = [
    "metrics_test.json",
]


def _normalize_path(raw: Path | str) -> Path:
    """Strip whitespace (notebook line-wrap), expand user home."""
    return Path(str(raw).strip()).expanduser()


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    seen: set[str] = set()
    out: list[Path] = []
    for p in paths:
        key = str(p.resolve())
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def _discover_runs_roots(aml_results: Path) -> list[Path]:
    """
    Under aml_results, find .../large_run_v2_*/runs (Colab batch folders).

    Sorted by path name so order is stable (e.g. ongoing vs dated batches).
    """
    if not aml_results.is_dir():
        return []
    found = sorted(aml_results.glob("large_run_v2_*/runs"))
    return [p for p in found if p.is_dir()]


def _load_json_safely(path: Path) -> dict | None:
    """Return parsed JSON dict, or None if the file does not exist or is invalid."""
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError):
        return None


def _read_f2_from_run_dir(run_dir: Path) -> tuple[float | None, str | None]:
    """
    Read F2-score from a single run directory.

    Returns (f2_value, source_filename) or (None, None) if not found.
    Prefers metrics_test_thresh.json (at F1-optimal threshold) over metrics_test.json.
    """
    for fname in _CANDIDATE_FILENAMES_TEST_THRESH:
        data = _load_json_safely(run_dir / fname)
        if data is not None and "f2" in data:
            return float(data["f2"]), fname

    for fname in _CANDIDATE_FILENAMES_TEST:
        data = _load_json_safely(run_dir / fname)
        if data is not None and "f2" in data:
            return float(data["f2"]), fname

    return None, None


def _find_run_dir(run_id: str, runs_roots: list[Path]) -> Path | None:
    for root in runs_roots:
        candidate = root / str(run_id).strip()
        if candidate.is_dir():
            return candidate
    return None


def _f2_from_pr(precision: float, recall: float, beta: float = 2.0) -> float:
    """Standard F-beta formula. Returns 0 for degenerate inputs."""
    if precision + recall == 0:
        return 0.0
    denom = (beta**2) * precision + recall
    if denom == 0:
        return 0.0
    return (1 + beta**2) * precision * recall / denom


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Add F2-scores from per-run JSONs to the Part A summary CSV."
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        required=True,
        help="Path to the existing part_a_summary_v2.csv",
    )
    parser.add_argument(
        "--aml-results",
        type=Path,
        default=None,
        help=(
            "Parent folder (e.g. .../aml_results). Collects each "
            "large_run_v2_*/runs subdirectory automatically."
        ),
    )
    parser.add_argument(
        "--runs-roots",
        type=Path,
        nargs="*",
        default=[],
        help=(
            "Directories that contain per-run subfolders named by run_id "
            "(e.g. .../large_run_v2_20260404_1637/runs). Optional if --aml-results is set."
        ),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        required=True,
        help="Path to write the augmented CSV.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail (exit code 1) if any run cannot be matched. Default: warn and continue.",
    )
    args = parser.parse_args()

    summary_csv = _normalize_path(args.summary_csv)
    output_csv = _normalize_path(args.output_csv)

    runs_roots: list[Path] = []
    if args.aml_results is not None:
        ar = _normalize_path(args.aml_results)
        discovered = _discover_runs_roots(ar)
        if not discovered:
            print(
                f"WARNING: No large_run_v2_*/runs directories under {ar}",
                file=sys.stderr,
            )
        runs_roots.extend(discovered)
    runs_roots.extend(_normalize_path(p) for p in args.runs_roots)
    runs_roots = _dedupe_paths(runs_roots)

    if not runs_roots:
        print(
            "ERROR: No runs roots. Use --aml-results (parent of large_run_v2_* folders) "
            "and/or --runs-roots with one or more .../runs paths.",
            file=sys.stderr,
        )
        return 1

    if not summary_csv.is_file():
        print(f"ERROR: Summary CSV not found: {summary_csv}", file=sys.stderr)
        return 1

    for root in runs_roots:
        if not root.is_dir():
            print(f"ERROR: Runs root not found: {root}", file=sys.stderr)
            return 1

    print("Using runs roots (in search order):")
    for r in runs_roots:
        print(f"  - {r}")

    df = pd.read_csv(summary_csv)
    print(f"Loaded summary CSV: {len(df)} rows, {len(df.columns)} columns")

    if "run_id" not in df.columns:
        print("ERROR: Column 'run_id' not in summary CSV.", file=sys.stderr)
        return 1

    f2_values: list[float | None] = []
    f2_sources: list[str | None] = []
    matched_dirs: list[str | None] = []

    unmatched_runs: list[str] = []
    missing_f2_runs: list[str] = []

    for run_id in df["run_id"]:
        run_dir = _find_run_dir(str(run_id), runs_roots)
        if run_dir is None:
            unmatched_runs.append(str(run_id))
            f2_values.append(None)
            f2_sources.append(None)
            matched_dirs.append(None)
            continue

        matched_dirs.append(str(run_dir))
        f2, source = _read_f2_from_run_dir(run_dir)
        if f2 is None:
            missing_f2_runs.append(str(run_id))
        f2_values.append(f2)
        f2_sources.append(source)

    df["f2_test_thresh"] = f2_values
    df["_f2_source_file"] = f2_sources
    df["_run_dir_matched"] = matched_dirs

    n_total = len(df)
    n_matched = sum(1 for d in matched_dirs if d is not None)
    n_with_f2 = sum(1 for v in f2_values if v is not None)
    print(f"\nMatched run folders: {n_matched}/{n_total}")
    print(f"F2-values read:      {n_with_f2}/{n_total}")

    if unmatched_runs:
        print(f"\nWARNING: {len(unmatched_runs)} run_ids could not be located:")
        for rid in unmatched_runs:
            print(f"  - {rid}")

    if missing_f2_runs:
        print(f"\nWARNING: {len(missing_f2_runs)} run folders contained no F2 in their JSONs:")
        for rid in missing_f2_runs:
            print(f"  - {rid}")

    print("\nVerification (JSON-F2 vs F2 from precision/recall at F1-optimal threshold):")
    print("Differences > 0.001 indicate unexpected mismatch.\n")
    mismatches = 0
    samples_ok = 0
    for _, row in df.iterrows():
        f2_json = row["f2_test_thresh"]
        if pd.isna(f2_json):
            continue
        p = row.get("precision_test_thresh")
        r = row.get("recall_test_thresh")
        if pd.isna(p) or pd.isna(r):
            continue
        f2_computed = _f2_from_pr(float(p), float(r))
        diff = abs(float(f2_json) - f2_computed)
        rid = row["run_id"]
        if diff > 0.001:
            mismatches += 1
            print(
                f"  {rid}: JSON={float(f2_json):.6f}, computed={f2_computed:.6f}, "
                f"diff={diff:.2e} (!)"
            )
        elif samples_ok < 2:
            print(
                f"  {rid}: JSON={float(f2_json):.6f}, computed={f2_computed:.6f}, "
                f"diff={diff:.2e} (sample OK)"
            )
            samples_ok += 1

    if mismatches == 0:
        print("All checked rows: JSON-F2 == F2-from-PR within tolerance.")
    else:
        print(f"\n{mismatches} row(s) showed a difference greater than 0.001.")

    if args.strict and (unmatched_runs or missing_f2_runs):
        print(
            "\nSTRICT mode: exiting with code 1 because of unmatched runs or missing F2.",
            file=sys.stderr,
        )
        return 1

    df_to_save = df.drop(columns=["_f2_source_file", "_run_dir_matched"])
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_to_save.to_csv(output_csv, index=False)
    print(f"\nWrote augmented CSV: {output_csv}")
    print(f"  Rows: {len(df_to_save)}, columns: {len(df_to_save.columns)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
