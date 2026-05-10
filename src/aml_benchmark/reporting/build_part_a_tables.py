"""Build Part A main-result tables (Markdown + LaTeX) from leaderboard CSV.

Reads ``results/part_a_summary_v2.csv`` (path configurable) and writes
``docs/part_a_results_tables.{md,tex}``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from aml_benchmark.config import get_project_root

STRATEGY_ORDER: tuple[str, ...] = (
    "baseline",
    "random_undersampling",
    "smote",
    "class_weighting",
)

STRATEGY_DISPLAY: dict[str, str] = {
    "baseline": "Baseline",
    "random_undersampling": "Random Undersampling",
    "smote": "SMOTE",
    "class_weighting": "Class Weighting",
}


def _f2_from_precision_recall(p: float, r: float) -> float:
    """F_beta with beta=2 from precision and recall (no y_true/y_pred needed)."""
    if p <= 0.0 and r <= 0.0:
        return 0.0
    b2 = 4.0  # beta**2 for beta=2
    denom = b2 * p + r
    if denom <= 0.0:
        return 0.0
    return (1.0 + b2) * p * r / denom


def _format_pct_target(row: pd.Series) -> str:
    if row["strategy"] == "baseline":
        return "—"
    p = float(row["target_prevalence"])
    return f"{p * 100:.1f}%"


def _fmt_float(x: float | None, decimals: int) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{float(x):.{decimals}f}"


def _fmt_fp_tp_ratio(tp: float, fp: float) -> str:
    tp_i = int(tp) if not pd.isna(tp) else 0
    fp_f = float(fp) if not pd.isna(fp) else 0.0
    if tp_i <= 0:
        return "—"
    return f"{fp_f / tp_i:.1f}"


def _prepare_subset(df: pd.DataFrame, model: str) -> pd.DataFrame:
    d = df[df["model"] == model].copy()
    d = d[d["strategy"] != "adasyn"]

    is_bl = d["strategy"] == "baseline"
    baseline = (
        d[is_bl]
        .sort_values("target_prevalence", ascending=True)
        .drop_duplicates(subset=["model"], keep="first")
    )
    others = d[~is_bl]
    out = pd.concat([baseline, others], ignore_index=True)

    rank_map = {s: i for i, s in enumerate(STRATEGY_ORDER)}
    out["_srank"] = out["strategy"].map(lambda s: rank_map.get(str(s), 999))
    out = out.sort_values(by=["_srank", "target_prevalence"], kind="mergesort")
    return out.drop(columns=["_srank"])


def _row_to_cells(row: pd.Series) -> list[str]:
    p = row.get("precision_test_thresh")
    r = row.get("recall_test_thresh")
    p_f = float(p) if p is not None and not pd.isna(p) else None
    r_f = float(r) if r is not None and not pd.isna(r) else None
    if p_f is not None and r_f is not None:
        f2 = _f2_from_precision_recall(p_f, r_f)
        f2_s = _fmt_float(f2, 3)
    else:
        f2_s = "—"

    tp = row.get("tp_test_thresh")
    fp = row.get("fp_test_thresh")
    tp_i = int(tp) if tp is not None and not pd.isna(tp) else None
    fp_i = int(fp) if fp is not None and not pd.isna(fp) else None
    fp_str = f"{fp_i:,}" if fp_i is not None else "—"

    ratio = (
        _fmt_fp_tp_ratio(float(tp), float(fp))
        if tp is not None and fp is not None and not pd.isna(tp) and not pd.isna(fp)
        else "—"
    )

    strat = STRATEGY_DISPLAY.get(str(row["strategy"]), str(row["strategy"]))

    return [
        strat,
        _format_pct_target(row),
        _fmt_float(row.get("pr_auc_test"), 3),
        _fmt_float(row.get("roc_auc_test"), 3),
        _fmt_float(p_f, 3),
        _fmt_float(r_f, 3),
        _fmt_float(
            float(row["f1_test_thresh"])
            if row.get("f1_test_thresh") is not None
            and not pd.isna(row.get("f1_test_thresh"))
            else None,
            3,
        ),
        f2_s,
        _fmt_float(
            float(row["weighted_accuracy_test_thresh"])
            if row.get("weighted_accuracy_test_thresh") is not None
            and not pd.isna(row.get("weighted_accuracy_test_thresh"))
            else None,
            3,
        ),
        str(tp_i) if tp_i is not None else "—",
        fp_str,
        ratio,
        _fmt_float(
            float(row["optimal_threshold"])
            if row.get("optimal_threshold") is not None
            and not pd.isna(row.get("optimal_threshold"))
            else None,
            4,
        ),
    ]


_HEADER_MD = (
    "| Strategy | Target Prev. | PR-AUC | ROC-AUC | Precision | Recall | F1 | F2 | "
    "C-W Acc | TP | FP | FP/TP | Threshold |"
)
_SEP_MD = (
    "|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
)


def _latex_escape(s: str) -> str:
    # Apply backslash escapes first; em dash last so "\textemdash{}" is not corrupted.
    t = (
        s.replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("_", "\\_")
        .replace("#", "\\#")
    )
    return t.replace("—", r"\textemdash{}")


def _build_markdown_table_body(df: pd.DataFrame, model: str) -> str:
    sub = _prepare_subset(df, model)
    lines = [_HEADER_MD, _SEP_MD]
    for _, row in sub.iterrows():
        cells = _row_to_cells(row)
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _build_latex_table(caption: str, label: str, df: pd.DataFrame, model: str) -> str:
    sub = _prepare_subset(df, model)
    cols = (
        "l r r r r r r r r r r r r"
    )  # Strategy l; numeric r; FP r with commas as text
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{{_latex_escape(caption)}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{cols}}}",
        "\\toprule",
        "\\textbf{Strategy} & \\textbf{Prev.} & \\textbf{PR-AUC} & \\textbf{ROC-AUC} & "
        "\\textbf{Prec.} & \\textbf{Rec.} & \\textbf{F1} & \\textbf{F2} & \\textbf{C-W Acc} "
        "& \\textbf{TP} & \\textbf{FP} & \\textbf{FP/TP} & \\textbf{Thr.} \\\\",
        "\\midrule",
    ]
    for _, row in sub.iterrows():
        cells = _row_to_cells(row)
        escaped = [_latex_escape(c) for c in cells]
        lines.append(" & ".join(escaped) + " \\\\")
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def build_part_a_tables(
    csv_path: Path,
    md_path: Path,
    tex_path: Path,
) -> None:
    df = pd.read_csv(csv_path)
    required = [
        "model",
        "strategy",
        "target_prevalence",
        "pr_auc_test",
        "roc_auc_test",
        "precision_test_thresh",
        "recall_test_thresh",
        "f1_test_thresh",
        "weighted_accuracy_test_thresh",
        "tp_test_thresh",
        "fp_test_thresh",
        "optimal_threshold",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")

    md_parts = [
        "# Part A — Main analysis tables",
        "",
        "Generated by "
        "`python -m aml_benchmark.reporting.build_part_a_tables`.",
        "",
        "## Table 5: XGBoost Part A Main Analysis Results",
        "",
        _build_markdown_table_body(df, "xgboost"),
        "",
        "## Table 6: Random Forest Part A Main Analysis Results",
        "",
        _build_markdown_table_body(df, "random_forest"),
        "",
    ]
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(md_parts).strip() + "\n", encoding="utf-8")

    tex_parts = [
        "% Auto-generated by aml_benchmark.reporting.build_part_a_tables",
        "% \\usepackage{booktabs}",
        "",
        _build_latex_table(
            "XGBoost Part A Main Analysis Results",
            "tab:part-a-xgboost-main",
            df,
            "xgboost",
        ),
        _build_latex_table(
            "Random Forest Part A Main Analysis Results",
            "tab:part-a-rf-main",
            df,
            "random_forest",
        ),
    ]
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    tex_path.write_text("\n".join(tex_parts), encoding="utf-8")


def main() -> None:
    root = get_project_root()
    parser = argparse.ArgumentParser(description="Build Part A Markdown/LaTeX tables.")
    parser.add_argument(
        "--csv",
        type=Path,
        default=root / "results" / "part_a_summary_v2.csv",
        help="Input leaderboard CSV",
    )
    parser.add_argument(
        "--md-out",
        type=Path,
        default=root / "docs" / "part_a_results_tables.md",
        help="Output Markdown path",
    )
    parser.add_argument(
        "--tex-out",
        type=Path,
        default=root / "docs" / "part_a_results_tables.tex",
        help="Output LaTeX path",
    )
    args = parser.parse_args()
    build_part_a_tables(args.csv, args.md_out, args.tex_out)


if __name__ == "__main__":
    main()
