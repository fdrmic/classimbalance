"""Analyze Part B v2 results (Strategy 6) and compare to Part A v2.

This script evaluates:
  - `results/part_b_summary_v2.csv` (Strategy 6: `smote_class_weighting`)
  - `results/part_a_summary_v2.csv` (standard strategies)

and prints a compact, research-question oriented overview.

Main research question
----------------------
How do standard and tailored class imbalance mitigation strategies compare
in detection performance under extreme class imbalance for AML monitoring?

Sub-question answered here (Q4)
-------------------------------
4) Can a purpose-designed sixth strategy (smote_class_weighting) achieve a
   superior balance of PR-AUC, Recall, and Precision compared to the
   standard approaches?

Metric conventions (same as Part A analyzer)
--------------------------------------------
PR-AUC: `pr_auc_test` (threshold-free).
Precision/Recall/F1: test metrics at the post-hoc optimized threshold:
  `precision_test_thresh`, `recall_test_thresh`, `f1_test_thresh`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
PART_A_CSV = REPO_ROOT / "results" / "part_a_summary_v2.csv"
PART_B_CSV = REPO_ROOT / "results" / "part_b_summary_v2.csv"
DEFAULT_OUT = REPO_ROOT / "results" / "part_b_report_v2.md"

STANDARD_STRATEGIES = [
    "baseline",
    "random_undersampling",
    "smote",
    "adasyn",
    "class_weighting",
]
STRATEGY_6 = "smote_class_weighting"


def _format_percent(x: float, digits: int = 2) -> str:
    return f"{100 * float(x):.{digits}f}%"


def _markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_(no rows)_"
    try:
        return df.to_markdown(index=False)
    except ImportError:
        return "```text\n" + df.to_string(index=False) + "\n```"


def _load(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["model"] = df["model"].astype(str)
    df["strategy"] = df["strategy"].astype(str)
    df["target_prevalence"] = df["target_prevalence"].astype(float)
    return df


def _select_best_per_condition(df: pd.DataFrame) -> pd.DataFrame:
    key = ["model", "strategy", "target_prevalence"]
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


def _core_view(df: pd.DataFrame) -> pd.DataFrame:
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
    out["fp_test_thresh"] = out["fp_test_thresh"].map(lambda v: f"{float(v):,.0f}")
    return out


def _compare_strategy6_to_best_standard(part_a: pd.DataFrame, part_b: pd.DataFrame) -> pd.DataFrame:
    """Compare Strategy 6 vs best standard strategy per prevalence (XGBoost only)."""
    a = part_a[(part_a["model"] == "xgboost") & (part_a["strategy"].isin(STANDARD_STRATEGIES))].copy()
    b = part_b[(part_b["model"] == "xgboost") & (part_b["strategy"] == STRATEGY_6)].copy()

    if a.empty or b.empty:
        return pd.DataFrame()

    # Choose "best standard" per prevalence by f1_test_thresh (primary operational metric here)
    best = (
        a.sort_values(["target_prevalence", "f1_test_thresh"], ascending=[True, False])
        .groupby(["target_prevalence"], as_index=False)
        .first()
        .rename(columns={"strategy": "best_standard_strategy"})
    )

    merged = best.merge(
        b,
        on=["model", "target_prevalence"],
        how="inner",
        suffixes=("_best_standard", "_s6"),
    )

    out = pd.DataFrame(
        {
            "target_prevalence": merged["target_prevalence"].map(lambda v: _format_percent(v, 3)),
            "best_standard_strategy": merged["best_standard_strategy"],
            "PR-AUC test (best std)": merged["pr_auc_test_best_standard"].map(lambda v: f"{float(v):.4f}"),
            "PR-AUC test (S6)": merged["pr_auc_test_s6"].map(lambda v: f"{float(v):.4f}"),
            "Precision@thresh test (best std)": merged["precision_test_thresh_best_standard"].map(
                lambda v: f"{float(v):.4f}"
            ),
            "Precision@thresh test (S6)": merged["precision_test_thresh_s6"].map(lambda v: f"{float(v):.4f}"),
            "Recall@thresh test (best std)": merged["recall_test_thresh_best_standard"].map(
                lambda v: f"{float(v):.4f}"
            ),
            "Recall@thresh test (S6)": merged["recall_test_thresh_s6"].map(lambda v: f"{float(v):.4f}"),
            "F1@thresh test (best std)": merged["f1_test_thresh_best_standard"].map(lambda v: f"{float(v):.4f}"),
            "F1@thresh test (S6)": merged["f1_test_thresh_s6"].map(lambda v: f"{float(v):.4f}"),
            "FP@thresh test (best std)": merged["fp_test_thresh_best_standard"].map(lambda v: f"{float(v):,.0f}"),
            "FP@thresh test (S6)": merged["fp_test_thresh_s6"].map(lambda v: f"{float(v):,.0f}"),
        }
    )
    return out


def _strategy6_conclusion(comp: pd.DataFrame) -> str:
    if comp.empty:
        return "No comparison rows available (missing inputs or filters returned empty)."

    lines: list[str] = []
    wins_f1 = 0
    wins_pr = 0
    loses_fp = 0

    for _, r in comp.iterrows():
        prev = str(r["target_prevalence"])
        f1_std = float(r["F1@thresh test (best std)"])
        f1_s6 = float(r["F1@thresh test (S6)"])
        pr_std = float(r["PR-AUC test (best std)"])
        pr_s6 = float(r["PR-AUC test (S6)"])
        fp_std = float(str(r["FP@thresh test (best std)"]).replace(",", ""))
        fp_s6 = float(str(r["FP@thresh test (S6)"]).replace(",", ""))

        if f1_s6 > f1_std:
            wins_f1 += 1
        if pr_s6 > pr_std:
            wins_pr += 1
        if fp_s6 > fp_std:
            loses_fp += 1

        lines.append(
            f"- **{prev}**: S6 vs best standard -> "
            f"ΔPR-AUC={pr_s6 - pr_std:+.4f}, "
            f"ΔF1={f1_s6 - f1_std:+.4f}, "
            f"ΔFP={fp_s6 - fp_std:+,.0f}"
        )

    lines.append("")
    lines.append(
        f"Summary across prevalences: S6 improves **F1** in {wins_f1}/{len(comp)} cases, "
        f"improves **PR-AUC** in {wins_pr}/{len(comp)} cases, and increases **FP** in {loses_fp}/{len(comp)} cases."
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--part_a_csv", type=str, default=str(PART_A_CSV), help="Input CSV (Part A summary)")
    parser.add_argument("--part_b_csv", type=str, default=str(PART_B_CSV), help="Input CSV (Part B summary)")
    parser.add_argument("--out", type=str, default=str(DEFAULT_OUT), help="Output Markdown report path")
    args = parser.parse_args()

    part_a_csv = Path(args.part_a_csv)
    part_b_csv = Path(args.part_b_csv)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not part_a_csv.exists():
        raise FileNotFoundError(f"CSV not found: {part_a_csv}")
    if not part_b_csv.exists():
        raise FileNotFoundError(f"CSV not found: {part_b_csv}")

    part_a = _select_best_per_condition(_load(part_a_csv))
    part_b = _select_best_per_condition(_load(part_b_csv))

    report_lines: list[str] = []
    report_lines.append("# Part B v2 — Strategy 6 (`smote_class_weighting`) Overview\n")
    try:
        report_lines.append(
            f"**Inputs:** `{part_b_csv.relative_to(REPO_ROOT)}` and `{part_a_csv.relative_to(REPO_ROOT)}`\n"
        )
    except ValueError:
        report_lines.append(f"**Inputs:** `{part_b_csv}` and `{part_a_csv}`\n")

    report_lines.append(
        "## Context / How to read this report\n"
        "- **PR-AUC (test)** is threshold-free and captures ranking quality.\n"
        "- **Precision/Recall/F1 (test, `*_thresh`)** use a threshold optimized on validation and applied to test.\n"
        "- **FP (false positives)** is used as a workload proxy.\n"
    )

    report_lines.append("## Strategy 6 results (per prevalence)\n")
    b_view = part_b[part_b["strategy"] == STRATEGY_6].copy()
    report_lines.append(_markdown_table(_core_view(b_view)))
    report_lines.append("")

    report_lines.append("## Q4 — Does Strategy 6 improve the balance vs standard strategies?\n")
    report_lines.append(
        "Comparison setup: **XGBoost only**; for each prevalence level we compare Strategy 6 against "
        "the **best standard strategy** (baseline, random_undersampling, smote, adasyn, class_weighting) "
        "chosen by highest `f1_test_thresh`."
    )
    report_lines.append("")

    comp = _compare_strategy6_to_best_standard(part_a, part_b)
    report_lines.append(_markdown_table(comp))
    report_lines.append("")

    report_lines.append(
        "### Interpretation\n"
        "- If Strategy 6 increases **F1** and/or **Recall** at comparable **PR-AUC** *without* exploding **FP**, "
        "it achieves a superior operational balance.\n"
        "- If Strategy 6 mainly increases recall but FP rises sharply, it may be unsuitable in practice due to "
        "alert overload.\n"
    )
    report_lines.append("")
    report_lines.append("### Quantified conclusion (from the table above)\n")
    report_lines.append(_strategy6_conclusion(comp))

    report = "\n".join(report_lines).rstrip() + "\n"
    out_path.write_text(report, encoding="utf-8")

    print(f"Report written: {out_path}")


if __name__ == "__main__":
    main()

