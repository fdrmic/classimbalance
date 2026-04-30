"""Generate thesis result tables from precomputed CSV/JSON artefacts.

Inputs (expected under the repository `results/` directory):
  - results/part_a_summary_v2.csv
  - results/threshold_info_strategy6.json                 (legacy, optional)
  - results/feature_importance_xgboost_mean.csv
  - results/feature_importance_rf_mean.csv
  - results/part_b_multi_threshold_summary.json           (Part B multi)

Plus per-run output folders for the Part B multi-strategy threshold runs:
  - outputs/part_b_thresholds/<run_id>/<strategy>/threshold_info.json

Outputs (written to results/tables/ as both .csv and .md):
  - table1a_main_results
  - table1b_appendix_results
  - table2_strategy6_comparison        (legacy, only if strategy6 input exists)
  - table3_feature_importance_xgboost
  - table4_feature_importance_rf
  - table5_part_b_multi_threshold      (Part A reference + 3 strategies = 4 rows)

CLI:
    python -m aml_benchmark.analysis.results_tables

No path arguments: paths are hardcoded relative to the project root.
Uses only pandas and json (plus stdlib).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def _project_root() -> Path:
    # src/aml_benchmark/analysis/results_tables.py -> parents:
    # [0]=analysis, [1]=aml_benchmark, [2]=src, [3]=repo root
    return Path(__file__).resolve().parents[3]


def _read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def _write_csv_and_md(df: pd.DataFrame, out_stem: Path) -> None:
    out_csv = out_stem.with_suffix(".csv")
    out_md = out_stem.with_suffix(".md")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"Saved {out_csv}")

    out_md.write_text(_to_markdown_table(df), encoding="utf-8")
    print(f"Saved {out_md}")


def _to_markdown_table(df: pd.DataFrame) -> str:
    """Minimal markdown table renderer without optional dependencies."""
    if df.empty:
        return "_(no rows)_\n"

    d = df.copy()
    for c in d.columns:
        d[c] = d[c].map(lambda x: "" if pd.isna(x) else str(x))

    cols = [str(c) for c in d.columns]
    rows = d.values.tolist()

    # Escape pipe characters
    def esc(s: str) -> str:
        return s.replace("|", "\\|")

    cols_esc = [esc(c) for c in cols]
    rows_esc = [[esc(str(v)) for v in r] for r in rows]

    lines: list[str] = []
    lines.append("| " + " | ".join(cols_esc) + " |")
    lines.append("|" + "|".join(["---"] * len(cols_esc)) + "|")
    for r in rows_esc:
        lines.append("| " + " | ".join(r) + " |")
    return "\n".join(lines) + "\n"


def _table1_base(part_a: pd.DataFrame) -> pd.DataFrame:
    """Return the 24 Part A conditions (no ADASYN), sorted for table output."""
    df = part_a[part_a["strategy"].astype(str) != "adasyn"].copy()
    df["pr_auc_test"] = df["pr_auc_test"].astype(float)
    df["target_prevalence"] = df["target_prevalence"].astype(float)
    df["_model_order"] = df["model"].map({"xgboost": 0, "random_forest": 1}).fillna(99).astype(int)
    df = df.sort_values(
        ["_model_order", "strategy", "target_prevalence"],
        ascending=[True, True, True],
    ).drop(columns=["_model_order"])
    return df.reset_index(drop=True)


def _table1a_main(part_a: pd.DataFrame) -> pd.DataFrame:
    """Table 1a — Main text (compact)."""
    df = _table1_base(part_a)

    # fp_per_tp = fp_test_thresh / tp_test_thresh
    tp = df["tp_test_thresh"].astype(float)
    fp = df["fp_test_thresh"].astype(float)
    fp_per_tp = fp / tp.replace({0.0: pd.NA})

    out = pd.DataFrame(
        {
            "model": df["model"].astype(str),
            "strategy": df["strategy"].astype(str),
            "target_prevalence": df["target_prevalence"].map(lambda v: f"{float(v):.3%}"),
            "pr_auc_test": df["pr_auc_test"].map(lambda v: f"{float(v):.4f}"),
            "precision_thresh": df["precision_test_thresh"].map(lambda v: f"{float(v):.4f}"),
            "recall_thresh": df["recall_test_thresh"].map(lambda v: f"{float(v):.4f}"),
            "f1_thresh": df["f1_test_thresh"].map(lambda v: f"{float(v):.4f}"),
            "fp_per_tp": fp_per_tp.map(lambda v: "" if pd.isna(v) else f"{float(v):.2f}"),
        }
    )
    return out


def _table1b_appendix(part_a: pd.DataFrame) -> pd.DataFrame:
    """Table 1b — Appendix (complete)."""
    df = _table1_base(part_a)

    cols = [
        "model",
        "strategy",
        "target_prevalence",
        "pr_auc_test",
        # Default threshold (0.5)
        "precision_test",
        "recall_test",
        "f1_test",
        "weighted_accuracy_test",
        "tp_test",
        "fp_test",
        # Optimised threshold (selected on val, applied to test)
        "precision_test_thresh",
        "recall_test_thresh",
        "f1_test_thresh",
        "weighted_accuracy_test_thresh",
        "tp_test_thresh",
        "fp_test_thresh",
    ]
    out = df[cols].copy()

    out["target_prevalence"] = out["target_prevalence"].map(lambda v: f"{float(v):.3%}")
    out = out.rename(
        columns={
            # Default threshold (0.5)
            "precision_test": "precision_default",
            "recall_test": "recall_default",
            "f1_test": "f1_default",
            "weighted_accuracy_test": "weighted_acc_default",
            "tp_test": "tp_default",
            "fp_test": "fp_default",
            # Optimised threshold
            "precision_test_thresh": "precision_thresh",
            "recall_test_thresh": "recall_thresh",
            "f1_test_thresh": "f1_thresh",
            "weighted_accuracy_test_thresh": "weighted_acc_thresh",
            "tp_test_thresh": "tp_thresh",
            "fp_test_thresh": "fp_thresh",
        }
    )

    for c in [
        "pr_auc_test",
        "precision_default",
        "recall_default",
        "f1_default",
        "weighted_acc_default",
        "precision_thresh",
        "recall_thresh",
        "f1_thresh",
        "weighted_acc_thresh",
    ]:
        out[c] = out[c].map(lambda v: f"{float(v):.4f}")
    for c in ["tp_default", "fp_default", "tp_thresh", "fp_thresh"]:
        out[c] = out[c].map(lambda v: f"{int(float(v)):,}")

    return out.reset_index(drop=True)


def _table2_strategy6_comparison(info: dict) -> pd.DataFrame:
    # Pull strategy6 test metrics
    m_s6 = info.get("test_metrics_at_tau_star", {}) or {}
    # Baseline comparison (test)
    base_cmp = info.get("baseline_comparison_test", {}) or {}
    base = base_cmp.get("baseline", {}) or {}
    s6 = base_cmp.get("strategy6", {}) or {}
    fppt = base_cmp.get("fp_per_tp", {}) or {}

    pr_auc = float(m_s6.get("pr_auc", 0.0))
    rows = [
        ("PR-AUC", pr_auc, pr_auc, 0.0),  # threshold-independent
        ("Precision", float(base.get("precision", 0.0)), float(s6.get("precision", 0.0)), None),
        ("Recall", float(base.get("recall", 0.0)), float(s6.get("recall", 0.0)), None),
        ("F1", float(base.get("f1", 0.0)), float(s6.get("f1", 0.0)), None),
        ("TP", int(base.get("tp", 0)), int(s6.get("tp", 0)), None),
        ("FP", int(base.get("fp", 0)), int(s6.get("fp", 0)), None),
        ("FP per TP", float(fppt.get("baseline", 0.0)), float(fppt.get("strategy6", 0.0)), None),
    ]

    out_rows: list[dict] = []
    for metric, b, s, d in rows:
        if d is None:
            try:
                d = s - b  # works for float and int
            except Exception:
                d = ""
        out_rows.append(
            {
                "metric": metric,
                "part_a_baseline": b,
                "strategy6": s,
                "delta": d,
            }
        )

    df = pd.DataFrame(out_rows)

    # Format
    def fmt(metric: str, v) -> str:
        if metric in {"TP", "FP"}:
            return f"{int(v):,}"
        if metric == "PR-AUC":
            return f"{float(v):.4f}"
        if metric == "FP per TP":
            return f"{float(v):.2f}"
        return f"{float(v):.4f}"

    df["part_a_baseline"] = [fmt(m, v) for m, v in zip(df["metric"], df["part_a_baseline"])]
    df["strategy6"] = [fmt(m, v) for m, v in zip(df["metric"], df["strategy6"])]
    df["delta"] = [fmt(m, v) if m != "PR-AUC" else "0.0000 (threshold-independent)" for m, v in zip(df["metric"], df["delta"])]

    return df


def _load_part_b_multi_records(
    summary_path: Path | None,
    part_b_outputs_dir: Path | None,
) -> tuple[dict, list[dict]]:
    """Return (part_a_reference, strategy_records) for Part B multi run.

    Preference order:
    1. ``results/part_b_multi_threshold_summary.json`` (canonical, written by
       the threshold optimizer).
    2. Fallback: scan ``outputs/part_b_thresholds/<run_id>/<strategy>/threshold_info.json``
       and reconstruct the structure.
    """
    if summary_path is not None and summary_path.exists():
        summary = _read_json(summary_path)
        return summary.get("part_a_reference", {}) or {}, list(
            summary.get("records", []) or []
        )

    if part_b_outputs_dir is None or not part_b_outputs_dir.exists():
        raise FileNotFoundError(
            "Neither part_b_multi_threshold_summary.json nor "
            f"outputs/part_b_thresholds/ found ({part_b_outputs_dir}). "
            "Run `python -m aml_benchmark.experiments.threshold_optimizer` first."
        )

    run_dirs = [d for d in sorted(part_b_outputs_dir.iterdir()) if d.is_dir()]
    if not run_dirs:
        raise FileNotFoundError(
            f"No run subdirectories found under {part_b_outputs_dir}."
        )

    run_dir = run_dirs[0]
    records: list[dict] = []
    part_a_ref: dict = {}
    for strategy_dir in sorted(run_dir.iterdir()):
        info_path = strategy_dir / "threshold_info.json"
        if not info_path.exists():
            continue
        info = _read_json(info_path)
        records.append(info)
        if not part_a_ref:
            part_a_ref = info.get("part_a_reference", {}) or {}
    return part_a_ref, records


def _table5_part_b_multi_threshold(
    part_a_reference: dict,
    records: list[dict],
) -> tuple[pd.DataFrame, str]:
    """Build the Part B multi-threshold comparison table.

    Returns (dataframe, header_note).  4 rows: 1 Part A reference + 3 strategies.
    """
    if not part_a_reference:
        raise ValueError("Empty part_a_reference passed to Table 5 builder.")
    if not records:
        raise ValueError("No strategy records passed to Table 5 builder.")

    rows: list[dict] = []
    rows.append(
        {
            "Strategy": "Part A Baseline (F1-optimal)",
            "Threshold": float(part_a_reference["threshold"]),
            "Precision": float(part_a_reference["precision"]),
            "Recall": float(part_a_reference["recall"]),
            "F1": float(part_a_reference["f1"]),
            "F2": float(part_a_reference["f2"]),
            "FP": int(part_a_reference["fp"]),
            "FP_delta": 0,
            "F1_delta": 0.0,
            "PR-AUC": float(part_a_reference["pr_auc"]),
        }
    )

    for r in records:
        rows.append(
            {
                "Strategy": str(r["strategy_type"]),
                "Threshold": float(r["threshold_value"]),
                "Precision": float(r["test_precision"]),
                "Recall": float(r["test_recall"]),
                "F1": float(r["test_f1"]),
                "F2": float(r["test_f2"]),
                "FP": int(r["test_fp"]),
                "FP_delta": int(r["delta_fp_vs_part_a_f1opt"]),
                "F1_delta": float(r["delta_f1_vs_part_a_f1opt"]),
                "PR-AUC": float(r["test_pr_auc"]),
            }
        )

    # Highlights only over the strategy rows (skip reference row at index 0)
    strategy_slice = pd.DataFrame(rows[1:])
    highlight_idx: dict[int, list[str]] = {i: [] for i in range(1, len(rows))}
    if not strategy_slice.empty:
        min_fp_pos = int(strategy_slice["FP"].astype(int).idxmin())
        max_recall_pos = int(strategy_slice["Recall"].astype(float).idxmax())
        max_f1_pos = int(strategy_slice["F1"].astype(float).idxmax())
        # idxmin/idxmax above operate on a 0-based slice — translate to row index
        highlight_idx[min_fp_pos + 1].append("min-FP")
        highlight_idx[max_recall_pos + 1].append("max-Recall")
        highlight_idx[max_f1_pos + 1].append("max-F1")

    fmt_rows: list[dict] = []
    for i, r in enumerate(rows):
        suffix = (
            "  *" + " *".join(highlight_idx[i])
            if highlight_idx.get(i)
            else ""
        )
        fmt_rows.append(
            {
                "Strategy": r["Strategy"] + suffix,
                "Threshold": f"{r['Threshold']:.6f}",
                "Precision": f"{r['Precision']:.4f}",
                "Recall": f"{r['Recall']:.4f}",
                "F1": f"{r['F1']:.4f}",
                "F2": f"{r['F2']:.4f}",
                "FP": f"{int(r['FP']):,}",
                "FP_delta": ("0" if i == 0 else f"{int(r['FP_delta']):+,}"),
                "F1_delta": ("0.0000" if i == 0 else f"{float(r['F1_delta']):+.4f}"),
                "PR-AUC": f"{r['PR-AUC']:.6f}",
            }
        )

    df = pd.DataFrame(fmt_rows)

    header_note = (
        f"Part A reference threshold (F1-optimal): "
        f"{float(part_a_reference['threshold']):.6f} | "
        f"Part A reference run: {part_a_reference.get('selected_baseline_run_id', '?')}\n"
        "PR-AUC is identical across all rows by construction "
        "(threshold-independent).\n"
    )
    return df, header_note


def _write_table5(df: pd.DataFrame, header_note: str, out_stem: Path) -> None:
    out_csv = out_stem.with_suffix(".csv")
    out_md = out_stem.with_suffix(".md")
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(out_csv, index=False)
    print(f"Saved {out_csv}")

    md = header_note + "\n" + _to_markdown_table(df)
    out_md.write_text(md, encoding="utf-8")
    print(f"Saved {out_md}")


def _table3_feature_importance_top15(path: Path, title: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Accept either column name from existing pipelines: importance or mean_importance
    if "mean_importance" in df.columns:
        imp_col = "mean_importance"
    elif "importance" in df.columns:
        imp_col = "importance"
    else:
        raise ValueError(f"{title}: expected column 'mean_importance' or 'importance' in {path}")

    out = (
        df[["feature", imp_col]]
        .rename(columns={imp_col: "mean_importance"})
        .copy()
    )
    out["mean_importance"] = out["mean_importance"].astype(float)
    out = out.sort_values("mean_importance", ascending=False).head(15).reset_index(drop=True)
    out.insert(0, "rank", range(1, len(out) + 1))
    out["mean_importance"] = out["mean_importance"].map(lambda v: f"{float(v):.6f}")
    return out


def main() -> None:
    root = _project_root()
    results_dir = root / "results"
    tables_dir = results_dir / "tables"

    part_a_path = results_dir / "part_a_summary_v2.csv"
    xgb_fi_path = results_dir / "feature_importance_xgboost_mean.csv"
    rf_fi_path = results_dir / "feature_importance_rf_mean.csv"

    # Required inputs
    for p in [part_a_path, xgb_fi_path, rf_fi_path]:
        if not p.exists():
            raise FileNotFoundError(
                f"Missing required input file: {p}\n"
                "Expected all inputs under the repo `results/` directory."
            )

    # Optional inputs
    strategy6_path = results_dir / "threshold_info_strategy6.json"
    part_b_multi_summary = results_dir / "part_b_multi_threshold_summary.json"
    part_b_outputs_dir = root / "outputs" / "part_b_thresholds"

    part_a = pd.read_csv(part_a_path)

    t1a = _table1a_main(part_a)
    _write_csv_and_md(t1a, tables_dir / "table1a_main_results")

    t1b = _table1b_appendix(part_a)
    _write_csv_and_md(t1b, tables_dir / "table1b_appendix_results")

    if strategy6_path.exists():
        info = _read_json(strategy6_path)
        t2 = _table2_strategy6_comparison(info)
        _write_csv_and_md(t2, tables_dir / "table2_strategy6_comparison")
    else:
        print(
            f"[skip] {strategy6_path.name} not found — "
            "table2_strategy6_comparison not generated."
        )

    t3 = _table3_feature_importance_top15(xgb_fi_path, title="XGBoost feature importance")
    _write_csv_and_md(t3, tables_dir / "table3_feature_importance_xgboost")

    t4 = _table3_feature_importance_top15(rf_fi_path, title="Random Forest feature importance")
    _write_csv_and_md(t4, tables_dir / "table4_feature_importance_rf")

    # Table 5 — Part B multi-threshold comparison (optional but preferred)
    try:
        part_a_ref, records = _load_part_b_multi_records(
            summary_path=part_b_multi_summary,
            part_b_outputs_dir=part_b_outputs_dir,
        )
        t5, header = _table5_part_b_multi_threshold(part_a_ref, records)
        _write_table5(t5, header, tables_dir / "table5_part_b_multi_threshold")
    except FileNotFoundError as exc:
        print(f"[skip] table5_part_b_multi_threshold: {exc}")


if __name__ == "__main__":
    main()

