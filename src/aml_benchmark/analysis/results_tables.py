"""Generate thesis result tables from precomputed CSV/JSON artefacts.

Inputs (expected under the repository `results/` directory):
  - results/part_a_summary_v2.csv
  - results/threshold_info_strategy6.json
  - results/feature_importance_xgboost_mean.csv
  - results/feature_importance_rf_mean.csv

Outputs (written to results/tables/ as both .csv and .md):
  - table1_part_a_summary
  - table2_strategy6_comparison
  - table3_feature_importance_xgboost
  - table4_feature_importance_rf

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
    strategy6_path = results_dir / "threshold_info_strategy6.json"
    xgb_fi_path = results_dir / "feature_importance_xgboost_mean.csv"
    rf_fi_path = results_dir / "feature_importance_rf_mean.csv"

    for p in [part_a_path, strategy6_path, xgb_fi_path, rf_fi_path]:
        if not p.exists():
            raise FileNotFoundError(
                f"Missing required input file: {p}\n"
                "Expected all inputs under the repo `results/` directory."
            )

    part_a = pd.read_csv(part_a_path)
    info = _read_json(strategy6_path)

    t1a = _table1a_main(part_a)
    _write_csv_and_md(t1a, tables_dir / "table1a_main_results")

    t1b = _table1b_appendix(part_a)
    _write_csv_and_md(t1b, tables_dir / "table1b_appendix_results")

    t2 = _table2_strategy6_comparison(info)
    _write_csv_and_md(t2, tables_dir / "table2_strategy6_comparison")

    t3 = _table3_feature_importance_top15(xgb_fi_path, title="XGBoost feature importance")
    _write_csv_and_md(t3, tables_dir / "table3_feature_importance_xgboost")

    t4 = _table3_feature_importance_top15(rf_fi_path, title="Random Forest feature importance")
    _write_csv_and_md(t4, tables_dir / "table4_feature_importance_rf")


if __name__ == "__main__":
    main()

