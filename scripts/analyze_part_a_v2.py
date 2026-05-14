"""Analyze Part A v2 results and generate a clean overview.

This script evaluates `results/part_a_summary_v2.csv` and prints:
  - Aggregated performance tables per (model, strategy, target_prevalence)
  - Robustness summaries across prevalence levels
  - Recall–precision trade-off views (operational workload proxy via FP counts)

It also answers the research sub-questions directly in the output:
1) Comparison of Baseline, Random Undersampling, SMOTE, ADASYN, Class Weighting
   for RandomForest and XGBoost (PR-AUC, Recall, Precision).
2) Robustness across prevalence levels (1%, 0.5%, 0.1% targets used in grid).
3) Trade-offs between recall and precision and implications for workload.

Notes on metrics used
---------------------
PR-AUC is threshold-free -> we use `pr_auc_test` (and `pr_auc_val` for context).
Recall/Precision are threshold-dependent -> we use the post-hoc optimized
threshold metrics on the TEST split:
  - `recall_test_thresh`
  - `precision_test_thresh`
  - `f1_test_thresh`
This aligns with the intended evaluation setup (threshold picked on val, applied to test).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = REPO_ROOT / "results" / "part_a_summary_v2.csv"
DEFAULT_OUT = REPO_ROOT / "results" / "part_a_report_v2.md"

CORE_STRATEGIES_ORDER = [
    "baseline",
    "random_undersampling",
    "smote",
    "adasyn",
    "class_weighting",
]
MODELS_ORDER = ["random_forest", "xgboost"]


def _format_percent(x: float, digits: int = 2) -> str:
    return f"{100 * float(x):.{digits}f}%"


def _markdown_table(df: pd.DataFrame) -> str:
    """Compact Markdown table with stable column order."""
    if df.empty:
        return "_(no rows)_"
    try:
        return df.to_markdown(index=False)
    except ImportError:
        # pandas.DataFrame.to_markdown requires optional dependency `tabulate`.
        return "```text\n" + df.to_string(index=False) + "\n```"


def _load(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Normalize types that matter for grouping/sorting
    df["model"] = df["model"].astype(str)
    df["strategy"] = df["strategy"].astype(str)
    df["target_prevalence"] = df["target_prevalence"].astype(float)

    return df


def _select_best_per_condition(df: pd.DataFrame) -> pd.DataFrame:
    """If multiple runs exist for same condition, keep the best by val F1@thresh."""
    key = ["model", "strategy", "target_prevalence"]
    # The summary file should already contain one row per condition, but we guard
    # against duplicates (e.g., reruns) deterministically.
    if df.duplicated(key).any():
        df = (
            df.sort_values(
                by=["f1_val_thresh", "created_at"],
                ascending=[False, False],
            )
            .drop_duplicates(key, keep="first")
            .reset_index(drop=True)
        )
    return df


def _overview_table(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "model",
        "strategy",
        "target_prevalence",
        "pr_auc_test",
        "precision_test_thresh",
        "recall_test_thresh",
        "f1_test_thresh",
        "tp_test_thresh",
        "fp_test_thresh",
        "fn_test_thresh",
    ]
    out = df[cols].copy()
    out["target_prevalence"] = out["target_prevalence"].map(lambda v: _format_percent(v, 3))
    out["pr_auc_test"] = out["pr_auc_test"].map(lambda v: f"{float(v):.4f}")
    out["precision_test_thresh"] = out["precision_test_thresh"].map(lambda v: f"{float(v):.4f}")
    out["recall_test_thresh"] = out["recall_test_thresh"].map(lambda v: f"{float(v):.4f}")
    out["f1_test_thresh"] = out["f1_test_thresh"].map(lambda v: f"{float(v):.4f}")
    return out


def _robustness_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate across prevalence levels (mean/std) per model-strategy."""
    grp = df.groupby(["model", "strategy"], as_index=False, observed=True).agg(
        pr_auc_test_mean=("pr_auc_test", "mean"),
        pr_auc_test_std=("pr_auc_test", "std"),
        precision_test_thresh_mean=("precision_test_thresh", "mean"),
        precision_test_thresh_std=("precision_test_thresh", "std"),
        recall_test_thresh_mean=("recall_test_thresh", "mean"),
        recall_test_thresh_std=("recall_test_thresh", "std"),
        f1_test_thresh_mean=("f1_test_thresh", "mean"),
        f1_test_thresh_std=("f1_test_thresh", "std"),
        fp_test_thresh_mean=("fp_test_thresh", "mean"),
        fp_test_thresh_std=("fp_test_thresh", "std"),
    )

    def fms(mean: float, std: float) -> str:
        if pd.isna(std):
            return f"{mean:.4f}"
        return f"{mean:.4f} ± {std:.4f}"

    out = pd.DataFrame(
        {
            "model": grp["model"],
            "strategy": grp["strategy"],
            "pr_auc_test": [fms(m, s) for m, s in zip(grp["pr_auc_test_mean"], grp["pr_auc_test_std"])],
            "precision@thresh (test)": [
                fms(m, s)
                for m, s in zip(grp["precision_test_thresh_mean"], grp["precision_test_thresh_std"])
            ],
            "recall@thresh (test)": [
                fms(m, s)
                for m, s in zip(grp["recall_test_thresh_mean"], grp["recall_test_thresh_std"])
            ],
            "f1@thresh (test)": [fms(m, s) for m, s in zip(grp["f1_test_thresh_mean"], grp["f1_test_thresh_std"])],
            "FP@thresh (test)": [
                f"{m:,.0f} ± {s:,.0f}" if not pd.isna(s) else f"{m:,.0f}"
                for m, s in zip(grp["fp_test_thresh_mean"], grp["fp_test_thresh_std"])
            ],
        }
    )
    return out


def _rankings(df: pd.DataFrame) -> pd.DataFrame:
    """Rank strategies per model & prevalence by PR-AUC and by F1@thresh."""
    rows = []
    for (model, prev), sub in df.groupby(["model", "target_prevalence"], observed=True):
        sub = sub.copy()
        sub["rank_pr_auc"] = sub["pr_auc_test"].rank(ascending=False, method="min").astype(int)
        sub["rank_f1"] = sub["f1_test_thresh"].rank(ascending=False, method="min").astype(int)
        for _, r in sub.iterrows():
            rows.append(
                {
                    "model": model,
                    "target_prevalence": _format_percent(prev, 3),
                    "strategy": r["strategy"],
                    "pr_auc_test": float(r["pr_auc_test"]),
                    "f1_test_thresh": float(r["f1_test_thresh"]),
                    "rank_pr_auc": int(r["rank_pr_auc"]),
                    "rank_f1": int(r["rank_f1"]),
                }
            )
    out = pd.DataFrame(rows)
    out["pr_auc_test"] = out["pr_auc_test"].map(lambda v: f"{v:.4f}")
    out["f1_test_thresh"] = out["f1_test_thresh"].map(lambda v: f"{v:.4f}")
    return out.sort_values(["model", "target_prevalence", "rank_f1", "rank_pr_auc", "strategy"])


def _answer_subquestions(df: pd.DataFrame) -> str:
    """Return a concise narrative based on computed summaries."""
    lines: list[str] = []

    lines.append(
        "## Context / How to read this report\n"
        "- **PR-AUC (test)** is threshold-free and captures ranking quality under extreme imbalance.\n"
        "- **Precision/Recall/F1 (test, `*_thresh`)** are computed at a threshold optimized on the "
        "validation split and then applied to test.\n"
        "- **FP (false positives)** is used as a simple **workload proxy** (alerts to investigate).\n"
    )

    lines.append("## Key findings (quick takeaways)")
    # Best F1@thresh per model x prevalence
    best_rows = (
        df.sort_values(["model", "target_prevalence", "f1_test_thresh"], ascending=[True, True, False])
        .groupby(["model", "target_prevalence"], as_index=False, observed=True)
        .first()
    )
    for _, r in best_rows.iterrows():
        lines.append(
            f"- **Best F1 (test@thresh)** for `{r['model']}` at prevalence "
            f"**{_format_percent(float(r['target_prevalence']), 3)}**: "
            f"`{r['strategy']}` "
            f"(F1={float(r['f1_test_thresh']):.4f}, "
            f"Prec={float(r['precision_test_thresh']):.4f}, "
            f"Rec={float(r['recall_test_thresh']):.4f}, "
            f"FP={int(r['fp_test_thresh']):,})."
        )
    lines.append("")

    # Q1: Compare strategies for both models in PR-AUC, recall, precision
    lines.append("## Q1 — Strategy comparison (PR-AUC, Recall, Precision)")
    for model in MODELS_ORDER:
        sub = df[df["model"] == model]
        if sub.empty:
            continue
        # Average across prevalence for a stable summary
        avg = (
            sub.groupby("strategy", as_index=False, observed=True)
            .agg(
                pr_auc=("pr_auc_test", "mean"),
                precision=("precision_test_thresh", "mean"),
                recall=("recall_test_thresh", "mean"),
                fp=("fp_test_thresh", "mean"),
            )
            .sort_values("pr_auc", ascending=False)
        )
        lines.append(f"### Model: `{model}` (means over prevalences)")
        lines.append(
            _markdown_table(
                avg.assign(
                    pr_auc=lambda d: d["pr_auc"].map(lambda v: f"{v:.4f}"),
                    precision=lambda d: d["precision"].map(lambda v: f"{v:.4f}"),
                    recall=lambda d: d["recall"].map(lambda v: f"{v:.4f}"),
                    fp=lambda d: d["fp"].map(lambda v: f"{v:,.0f}"),
                )[["strategy", "pr_auc", "precision", "recall", "fp"]]
            )
        )

    # Q2: Robustness
    lines.append("\n## Q2 — Robustness across prevalence levels")
    rob = _robustness_summary(df)
    # Keep only core strategies and stable ordering if possible
    rob = rob[rob["strategy"].isin(CORE_STRATEGIES_ORDER)]
    lines.append(_markdown_table(rob.sort_values(["model", "strategy"])))
    lines.append(
        "\nInterpretation guide: small std (±) indicates robustness across prevalence targets; "
        "large std suggests sensitivity to imbalance level."
    )

    # Q3: Trade-offs and workload
    lines.append("\n## Q3 — Recall/Precision trade-offs and operational workload proxy")
    lines.append(
        "We approximate *operational workload* with the number of false positives (FP) at the "
        "chosen threshold (selected on val, applied to test). Higher recall typically increases FP."
    )
    trade = (
        df.groupby(["model", "strategy"], as_index=False, observed=True)
        .agg(
            precision=("precision_test_thresh", "mean"),
            recall=("recall_test_thresh", "mean"),
            f1=("f1_test_thresh", "mean"),
            fp=("fp_test_thresh", "mean"),
        )
        .sort_values(["model", "f1"], ascending=[True, False])
    )
    trade["precision"] = trade["precision"].map(lambda v: f"{v:.4f}")
    trade["recall"] = trade["recall"].map(lambda v: f"{v:.4f}")
    trade["f1"] = trade["f1"].map(lambda v: f"{v:.4f}")
    trade["fp"] = trade["fp"].map(lambda v: f"{v:,.0f}")
    lines.append(_markdown_table(trade[["model", "strategy", "precision", "recall", "f1", "fp"]]))
    lines.append(
        "\nOperational implication: strategies that boost recall by lowering the threshold can "
        "produce very large FP volumes; this increases alert review workload for AML teams. "
        "A practically useful strategy should improve recall while keeping precision (and FP) at "
        "manageable levels."
    )

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default=str(DEFAULT_CSV), help="Input CSV (Part A summary)")
    parser.add_argument("--out", type=str, default=str(DEFAULT_OUT), help="Output Markdown report path")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = _load(csv_path)
    df = _select_best_per_condition(df)

    # Restrict to the five standard strategies explicitly referenced in the questions.
    df = df[df["strategy"].isin(CORE_STRATEGIES_ORDER)].copy()

    # Stable ordering for readability
    df["strategy"] = pd.Categorical(df["strategy"], categories=CORE_STRATEGIES_ORDER, ordered=True)
    df["model"] = pd.Categorical(df["model"], categories=MODELS_ORDER, ordered=True)
    df = df.sort_values(["model", "strategy", "target_prevalence"]).reset_index(drop=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    report_lines: list[str] = []
    report_lines.append("# Part A v2 — Results Overview\n")
    try:
        report_lines.append(f"**Input:** `{csv_path.relative_to(REPO_ROOT)}`\n")
    except ValueError:
        report_lines.append(f"**Input:** `{csv_path}`\n")

    report_lines.append("## Table — Per condition (test metrics)\n")
    report_lines.append(_markdown_table(_overview_table(df)))
    report_lines.append("")

    report_lines.append("## Rankings — per model & prevalence\n")
    report_lines.append(_markdown_table(_rankings(df)))
    report_lines.append("")

    report_lines.append(_answer_subquestions(df))
    report = "\n".join(report_lines).rstrip() + "\n"

    out_path.write_text(report, encoding="utf-8")

    print(f"Report written: {out_path}")


if __name__ == "__main__":
    main()

