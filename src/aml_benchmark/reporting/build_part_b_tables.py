"""Build Part B Section 7 tables (Markdown + LaTeX) from PAI-HNU and Part A CSVs.

Generates:
  - docs/part_b_results_tables.{md,tex} — Tables 7 & 8
  - docs/part_b_cap_analysis.md — Table 9 (cap / sampling stats)

Run::

    python -m aml_benchmark.reporting.build_part_b_tables \\
        --part-b-summary path/to/part_b_pai_hnu_summary.csv \\
        --part-a-summary path/to/part_a_summary_v2.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from aml_benchmark.config import get_project_root
from aml_benchmark.reporting.build_part_a_tables import (
    _f2_from_precision_recall,
    _fmt_float,
    _fmt_fp_tp_ratio,
    _latex_escape,
)

# ---------------------------------------------------------------------------
# Strategy labels for Configuration column (Table 8)
# ---------------------------------------------------------------------------

_CFG_STRATEGY: dict[str, str] = {
    "baseline": "baseline",
    "random_undersampling": "random undersampling",
    "smote": "smote",
    "class_weighting": "class weighting",
}

# (strategy, target_prevalence, marker suffix or None)
_PART_A_ROWS: list[tuple[str, float, str | None]] = [
    ("baseline", 0.001, "(best PR-AUC Part A)"),
    ("random_undersampling", 0.005, "(Pareto-optimal Part A)"),
    ("random_undersampling", 0.01, None),
    ("class_weighting", 0.01, "(best F1 Part A)"),
    ("smote", 0.001, "(best Recall Part A)"),
]


def _resolve_path(raw: str | Path, repo_root: Path) -> Path:
    p = Path(raw).expanduser()
    return p.resolve() if p.is_absolute() else (repo_root / p).resolve()


def _require_file(path: Path, hint: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(
            f"Required input file not found:\n  {path}\n{hint}"
        )


def _fmt_target_prev_frac(p: float) -> str:
    return f"{float(p) * 100:.1f}%"


def _eq_prevalence(a, b, *, tol: float = 1e-9) -> bool:
    return abs(float(a) - float(b)) < tol


def _f2_cell(row: pd.Series) -> str:
    if "f2_test_thresh" in row.index:
        v = row["f2_test_thresh"]
        if v is not None and not pd.isna(v):
            return _fmt_float(float(v), 3)
    p = row.get("precision_test_thresh")
    r = row.get("recall_test_thresh")
    if p is None or r is None or pd.isna(p) or pd.isna(r):
        return "—"
    return _fmt_float(_f2_from_precision_recall(float(p), float(r)), 3)


def _part_a_config_string(strategy: str, prev: float) -> str:
    slug = _CFG_STRATEGY.get(strategy, strategy.replace("_", " "))
    return f"xgboost {slug} @{_fmt_target_prev_frac(prev)}"


def _pick_xgboost_row(part_a: pd.DataFrame, strategy: str, prev: float) -> pd.Series:
    m = part_a[
        (part_a["model"] == "xgboost")
        & (part_a["strategy"] == strategy)
        & (part_a["target_prevalence"].apply(lambda x: _eq_prevalence(x, prev)))
    ]
    if m.empty:
        raise ValueError(
            f"No Part A row for xgboost / {strategy} / target_prevalence={prev}. "
            "Check part_a_summary_v2.csv."
        )
    return m.sort_values("run_id").iloc[0]


def _table7_row_cells(row: pd.Series) -> list[str]:
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
    p = row.get("precision_test_thresh")
    r = row.get("recall_test_thresh")
    p_f = float(p) if p is not None and not pd.isna(p) else None
    r_f = float(r) if r is not None and not pd.isna(r) else None
    return [
        _fmt_target_prev_frac(row["target_prevalence"]),
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
        _f2_cell(row),
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


_T7_HEADER_MD = (
    "| Target Prev. | PR-AUC | ROC-AUC | Precision | Recall | F1 | F2 | "
    "C-W Acc | TP | FP | FP/TP | Threshold |"
)
_T7_SEP_MD = "|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"


def _markdown_table7(df_b: pd.DataFrame) -> str:
    sub = df_b.sort_values("target_prevalence", ascending=True, kind="mergesort")
    lines = [_T7_HEADER_MD, _T7_SEP_MD]
    for _, row in sub.iterrows():
        lines.append("| " + " | ".join(_table7_row_cells(row)) + " |")
    return "\n".join(lines)


def _latex_table7(df_b: pd.DataFrame) -> str:
    sub = df_b.sort_values("target_prevalence", ascending=True, kind="mergesort")
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{PAI-HNU Results across Target Prevalence Levels}",
        "\\label{tab:pai-hnu-prevalence}",
        "\\begin{tabular}{l r r r r r r r r r r r}",
        "\\toprule",
        "\\textbf{Prev.} & \\textbf{PR-AUC} & \\textbf{ROC-AUC} & \\textbf{Prec.} & "
        "\\textbf{Rec.} & \\textbf{F1} & \\textbf{F2} & \\textbf{C-W Acc} & "
        "\\textbf{TP} & \\textbf{FP} & \\textbf{FP/TP} & \\textbf{Thr.} \\\\",
        "\\midrule",
    ]
    for _, row in sub.iterrows():
        cells = _table7_row_cells(row)
        lines.append(" & ".join(_latex_escape(c) for c in cells) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    return "\n".join(lines)


def _table8_row_cells(
    group: str,
    configuration: str,
    row: pd.Series,
) -> list[str]:
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
    p = row.get("precision_test_thresh")
    r = row.get("recall_test_thresh")
    p_f = float(p) if p is not None and not pd.isna(p) else None
    r_f = float(r) if r is not None and not pd.isna(r) else None
    return [
        group,
        configuration,
        _fmt_float(row.get("pr_auc_test"), 3),
        _fmt_float(
            float(row["f1_test_thresh"])
            if row.get("f1_test_thresh") is not None
            and not pd.isna(row.get("f1_test_thresh"))
            else None,
            3,
        ),
        _f2_cell(row),
        _fmt_float(p_f, 3),
        _fmt_float(r_f, 3),
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


_T8_HEADER_MD = (
    "| Group | "  # continues
    "Configuration | PR-AUC | F1 | F2 | Precision | Recall | TP | FP | FP/TP | Threshold |"
)
_T8_SEP_MD = "|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"


def _build_table8_rows(part_a: pd.DataFrame, df_b: pd.DataFrame) -> list[list[str]]:
    rows: list[list[str]] = []
    for strategy, prev, marker in _PART_A_ROWS:
        row = _pick_xgboost_row(part_a, strategy, prev)
        cfg = _part_a_config_string(strategy, prev)
        if marker:
            cfg = f"{cfg} {marker}"
        rows.append(_table8_row_cells("Part A", cfg, row))

    b_sub = df_b.sort_values("target_prevalence", ascending=True, kind="mergesort")
    for _, row in b_sub.iterrows():
        prev = float(row["target_prevalence"])
        cfg = f"PAI-HNU @{_fmt_target_prev_frac(prev)}"
        if _eq_prevalence(prev, 0.005):
            cfg = f"{cfg} (Pareto-optimal Part B)"
        rows.append(_table8_row_cells("Part B PAI-HNU", cfg, row))
    return rows


def _markdown_table8(part_a: pd.DataFrame, df_b: pd.DataFrame) -> str:
    lines = [_T8_HEADER_MD, _T8_SEP_MD]
    for cells in _build_table8_rows(part_a, df_b):
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _latex_escape_config(s: str) -> str:
    """Like _latex_escape but makes @ safe in tabular (pdfLaTeX)."""
    placeholder = "\uE000"
    t = s.replace("@", placeholder)
    t = _latex_escape(t)
    return t.replace(placeholder, r"\symbol{64}")


def _latex_table8(part_a: pd.DataFrame, df_b: pd.DataFrame) -> str:
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{PAI-HNU vs.\\ Strongest Part A Configurations}",
        "\\label{tab:pai-hnu-vs-part-a}",
        "\\begin{tabular}{l p{5.8cm} r r r r r r r r r}",
        "\\toprule",
        "\\textbf{Group} & \\textbf{Configuration} & \\textbf{PR-AUC} & \\textbf{F1} & "
        "\\textbf{F2} & \\textbf{Prec.} & \\textbf{Rec.} & \\textbf{TP} & \\textbf{FP} "
        "& \\textbf{FP/TP} & \\textbf{Thr.} \\\\",
        "\\midrule",
    ]
    for cells in _build_table8_rows(part_a, df_b):
        g0 = _latex_escape(cells[0])
        g1 = _latex_escape_config(cells[1])
        rest = [_latex_escape(c) for c in cells[2:]]
        lines.append(" & ".join([g0, g1, *rest]) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    return "\n".join(lines)


def _cap_utilization(actual: float, planned: float) -> str:
    if planned <= 0 or np.isnan(planned):
        return "—"
    return f"{100.0 * float(actual) / float(planned):.1f}%"


def _effective_hard_share(row: pd.Series) -> str:
    h = float(row["n_hard_actual"])
    t = float(row["n_temporal_actual"])
    g = float(row["n_global_actual"])
    den = h + t + g
    if den <= 0:
        return "—"
    return f"{100.0 * h / den:.1f}%"


def _markdown_table9(df_b: pd.DataFrame) -> str:
    sub = df_b.sort_values("target_prevalence", ascending=True, kind="mergesort")
    header = (
        "| Target Prev. | n_hard_planned | n_hard_cap | n_hard_actual | "
        "Cap utilization | n_temporal_actual | n_global_actual | "
        "Effective hard share |"
    )
    sep = "|:---|---:|---:|---:|---:|---:|---:|---:|"
    lines = [header, sep]
    for _, row in sub.iterrows():
        planned = float(row["n_hard_planned"])
        actual = float(row["n_hard_actual"])
        cap = float(row["n_hard_cap"])
        lines.append(
            "| "
            + " | ".join(
                [
                    _fmt_target_prev_frac(row["target_prevalence"]),
                    f"{int(round(planned)):,}",
                    f"{int(round(cap)):,}",
                    f"{int(round(actual)):,}",
                    _cap_utilization(actual, planned),
                    f"{int(round(float(row['n_temporal_actual']))):,}",
                    f"{int(round(float(row['n_global_actual']))):,}",
                    _effective_hard_share(row),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def _validate_part_a(df: pd.DataFrame) -> None:
    required = [
        "model",
        "strategy",
        "target_prevalence",
        "pr_auc_test",
        "precision_test_thresh",
        "recall_test_thresh",
        "f1_test_thresh",
        "tp_test_thresh",
        "fp_test_thresh",
        "optimal_threshold",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"part_a_summary CSV missing columns: {missing}")


def _validate_part_b(df: pd.DataFrame) -> None:
    required = [
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
        "n_hard_planned",
        "n_hard_cap",
        "n_hard_actual",
        "n_temporal_actual",
        "n_global_actual",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"part_b_pai_hnu_summary CSV missing columns: {missing}")
    if len(df) < 1:
        raise ValueError("part_b_pai_hnu_summary.csv has no rows.")


def build_part_b_tables(
    part_b_csv: Path,
    part_a_csv: Path,
    md_out: Path,
    tex_out: Path,
    cap_md_out: Path,
) -> None:
    hint_b = (
        "Pass --part-b-summary with a path to part_b_pai_hnu_summary.csv "
        "(e.g. from your Colab run folder or strategie6/)."
    )
    hint_a = (
        "Pass --part-a-summary with a path to part_a_summary_v2.csv "
        "(same run bundle or repo results/)."
    )
    _require_file(part_b_csv, hint_b)
    _require_file(part_a_csv, hint_a)

    df_b = pd.read_csv(part_b_csv)
    df_a = pd.read_csv(part_a_csv)
    _validate_part_b(df_b)
    _validate_part_a(df_a)

    md_parts = [
        "# Part B — Section 7 results tables",
        "",
        "Generated by `python -m aml_benchmark.reporting.build_part_b_tables`.",
        "",
        "## Table 7: PAI-HNU Results across Target Prevalence Levels",
        "",
        _markdown_table7(df_b),
        "",
        "## Table 8: PAI-HNU vs. Strongest Part A Configurations",
        "",
        _markdown_table8(df_a, df_b),
        "",
    ]
    md_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.write_text("\n".join(md_parts).strip() + "\n", encoding="utf-8")

    tex_parts = [
        "% Auto-generated by aml_benchmark.reporting.build_part_b_tables",
        "% \\usepackage{booktabs}",
        "",
        _latex_table7(df_b),
        _latex_table8(df_a, df_b),
    ]
    tex_out.parent.mkdir(parents=True, exist_ok=True)
    tex_out.write_text("\n".join(tex_parts), encoding="utf-8")

    cap_body = [
        "# Table 9: PAI-HNU Sampling Cap Analysis",
        "",
        "Defense-relevant sampling statistics (from `part_b_pai_hnu_summary.csv`).",
        "",
        _markdown_table9(df_b),
        "",
    ]
    cap_md_out.parent.mkdir(parents=True, exist_ok=True)
    cap_md_out.write_text("\n".join(cap_body).strip() + "\n", encoding="utf-8")


def main() -> None:
    root = get_project_root()
    parser = argparse.ArgumentParser(
        description="Build Part B Section 7 tables (Markdown + LaTeX)."
    )
    parser.add_argument(
        "--part-b-summary",
        type=Path,
        default=root / "strategie6" / "part_b_pai_hnu_summary.csv",
        help="part_b_pai_hnu_summary.csv (absolute or relative to repo root)",
    )
    parser.add_argument(
        "--part-a-summary",
        type=Path,
        default=root / "results" / "part_a_summary_v2.csv",
        help="part_a_summary_v2.csv (absolute or relative to repo root)",
    )
    parser.add_argument(
        "--md-out",
        type=Path,
        default=root / "docs" / "part_b_results_tables.md",
    )
    parser.add_argument(
        "--tex-out",
        type=Path,
        default=root / "docs" / "part_b_results_tables.tex",
    )
    parser.add_argument(
        "--cap-md-out",
        type=Path,
        default=root / "docs" / "part_b_cap_analysis.md",
    )
    args = parser.parse_args()

    build_part_b_tables(
        _resolve_path(args.part_b_summary, root),
        _resolve_path(args.part_a_summary, root),
        _resolve_path(args.md_out, root),
        _resolve_path(args.tex_out, root),
        _resolve_path(args.cap_md_out, root),
    )


if __name__ == "__main__":
    main()
