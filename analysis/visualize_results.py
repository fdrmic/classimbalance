"""Visualisation script for the AML benchmark thesis results.

Generates five publication-quality figures from Part A and Part B leaderboard
CSVs.  All figures are saved as 300-DPI PNGs in ``analysis/figures/``.

Usage
-----
    python analysis/visualize_results.py

Input files (relative to project root)
---------------------------------------
    results/part_a_summary.csv
    results/part_b_summary.csv

Output figures
--------------
    analysis/figures/fig1_pr_auc_by_strategy.png
    analysis/figures/fig2_recall_precision_scatter.png
    analysis/figures/fig3_pr_auc_by_prevalence.png
    analysis/figures/fig4_fp_tp_ratio.png
    analysis/figures/fig5_part_b_comparison.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parents[1]
_RESULTS_DIR = _ROOT / "results"
_FIGURES_DIR = _ROOT / "analysis" / "figures"
_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    plt.style.use("ggplot")

TITLE_SIZE = 14
LABEL_SIZE = 12
TICK_SIZE = 10
DPI = 300

# Consistent strategy colour palette
STRATEGY_COLORS: dict[str, str] = {
    "baseline":             "#808080",   # grey
    "random_undersampling": "#2196F3",   # blue
    "smote":                "#4CAF50",   # green
    "adasyn":               "#FF9800",   # orange
    "class_weighting":      "#9C27B0",   # purple
    "true_cost_weighting":  "#F44336",   # red
}

STRATEGY_ORDER = list(STRATEGY_COLORS.keys())

MODEL_COLORS: dict[str, str] = {
    "xgboost":       "#1565C0",
    "random_forest": "#2E7D32",
}

MODEL_MARKERS: dict[str, str] = {
    "xgboost":       "o",
    "random_forest": "^",
}

MODEL_LABELS: dict[str, str] = {
    "xgboost":       "XGBoost",
    "random_forest": "Random Forest",
}

PREVALENCE_LABELS = {0.001: "0.1%", 0.005: "0.5%", 0.010: "1.0%"}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load Part A and Part B CSVs and return (part_a, part_b, combined)."""
    path_a = _RESULTS_DIR / "part_a_summary.csv"
    path_b = _RESULTS_DIR / "part_b_summary.csv"

    df_a = pd.read_csv(path_a)
    df_b = pd.read_csv(path_b)

    df_a["part"] = "A"
    df_b["part"] = "B"

    combined = pd.concat([df_a, df_b], ignore_index=True)

    # Use optimal-threshold metrics where available, fall back to default
    for col in ("recall", "precision", "f1", "tp", "fp", "fn"):
        thresh_col = f"{col}_test_thresh"
        base_col = f"{col}_test"
        if thresh_col in combined.columns:
            combined[col] = combined[thresh_col].fillna(combined.get(base_col, 0))
        elif base_col in combined.columns:
            combined[col] = combined[base_col]

    if "pr_auc_test" in combined.columns:
        combined["pr_auc"] = combined["pr_auc_test"]

    # Safe FP/TP ratio
    combined["tp"] = combined["tp"].fillna(0)
    combined["fp"] = combined["fp"].fillna(0)
    combined["fp_tp_ratio"] = combined.apply(
        lambda r: r["fp"] / r["tp"] if r["tp"] > 0 else np.nan, axis=1
    )

    print(f"Loaded {len(df_a)} Part A rows, {len(df_b)} Part B rows.")
    return df_a, df_b, combined


# ---------------------------------------------------------------------------
# Fig 1 — Grouped bar: PR-AUC by strategy and model
# ---------------------------------------------------------------------------

def fig1_pr_auc_by_strategy(combined: pd.DataFrame) -> None:
    strategies = [s for s in STRATEGY_ORDER if s in combined["strategy"].values]
    models = ["xgboost", "random_forest"]

    # Mean PR-AUC over prevalence levels per (strategy, model)
    pivot = (
        combined.groupby(["strategy", "model"])["pr_auc"]
        .mean()
        .unstack(fill_value=0)
    )

    part_a_mean = combined[combined["part"] == "A"]["pr_auc"].mean()

    x = np.arange(len(strategies))
    width = 0.35
    fig, ax = plt.subplots(figsize=(12, 6))

    for i, model in enumerate(models):
        vals = [pivot.loc[s, model] if (s in pivot.index and model in pivot.columns) else 0
                for s in strategies]
        offset = (i - 0.5) * width
        bars = ax.bar(
            x + offset, vals, width,
            label=MODEL_LABELS[model],
            color=MODEL_COLORS[model],
            alpha=0.85,
            edgecolor="white",
            linewidth=0.5,
        )
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.0005,
                    f"{val:.4f}",
                    ha="center", va="bottom", fontsize=8, rotation=45,
                )

    ax.axhline(part_a_mean, color="grey", linestyle="--", linewidth=1.2,
               label=f"Part A mean ({part_a_mean:.4f})")

    ax.set_xticks(x)
    ax.set_xticklabels(
        [s.replace("_", "\n") for s in strategies],
        fontsize=TICK_SIZE,
    )
    ax.set_xlabel("Strategy", fontsize=LABEL_SIZE)
    ax.set_ylabel("PR-AUC (Test)", fontsize=LABEL_SIZE)
    ax.set_title("PR-AUC by Strategy and Model (mean over prevalence levels)",
                 fontsize=TITLE_SIZE)
    ax.legend(fontsize=TICK_SIZE)
    ax.set_ylim(0, max(pivot.values.max() * 1.25, 0.01))

    fig.tight_layout()
    _save(fig, "fig1_pr_auc_by_strategy.png")


# ---------------------------------------------------------------------------
# Fig 2 — Scatter: Recall vs Precision
# ---------------------------------------------------------------------------

def fig2_recall_precision_scatter(combined: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 7))

    strategies = combined["strategy"].unique()

    for _, row in combined.iterrows():
        strategy = row["strategy"]
        model = row["model"]
        part = row["part"]

        color = STRATEGY_COLORS.get(strategy, "#333333")
        marker = MODEL_MARKERS.get(model, "o")
        lw = 2.5 if part == "B" else 0.5
        size = 120 if part == "B" else 70

        ax.scatter(
            row["recall"], row["precision"],
            c=color, marker=marker,
            s=size,
            edgecolors="black", linewidths=lw,
            zorder=3 if part == "B" else 2,
        )

    # Legend: strategies (colours)
    strategy_patches = [
        mpatches.Patch(color=STRATEGY_COLORS.get(s, "#333"), label=s.replace("_", " "))
        for s in STRATEGY_ORDER if s in combined["strategy"].values
    ]
    # Legend: models (shapes)
    model_handles = [
        plt.Line2D([0], [0], marker=MODEL_MARKERS[m], color="w",
                   markerfacecolor="#555", markersize=10,
                   label=MODEL_LABELS[m])
        for m in ["xgboost", "random_forest"]
    ]
    # Part B indicator
    part_b_handle = plt.Line2D(
        [0], [0], marker="o", color="w", markerfacecolor="#555",
        markersize=10, markeredgecolor="black", markeredgewidth=2.5,
        label="Part B (thick border)",
    )

    legend1 = ax.legend(
        handles=strategy_patches,
        title="Strategy", fontsize=9, title_fontsize=10,
        loc="upper right",
    )
    ax.add_artist(legend1)
    ax.legend(
        handles=model_handles + [part_b_handle],
        title="Model / Part", fontsize=9, title_fontsize=10,
        loc="lower right",
    )

    ax.set_xlabel("Recall (at optimal threshold)", fontsize=LABEL_SIZE)
    ax.set_ylabel("Precision (at optimal threshold)", fontsize=LABEL_SIZE)
    ax.set_title("Recall vs. Precision — All 36 Runs (Part A + Part B)",
                 fontsize=TITLE_SIZE)

    fig.tight_layout()
    _save(fig, "fig2_recall_precision_scatter.png")


# ---------------------------------------------------------------------------
# Fig 3 — Line chart: PR-AUC over prevalence levels
# ---------------------------------------------------------------------------

def fig3_pr_auc_by_prevalence(combined: pd.DataFrame) -> None:
    prevalences = sorted(combined["target_prevalence"].unique())
    strategies = [s for s in STRATEGY_ORDER if s in combined["strategy"].values]

    fig, ax = plt.subplots(figsize=(10, 6))

    for strategy in strategies:
        sub = combined[combined["strategy"] == strategy]
        vals = []
        for p in prevalences:
            mean_val = sub[sub["target_prevalence"] == p]["pr_auc"].mean()
            vals.append(mean_val)

        color = STRATEGY_COLORS.get(strategy, "#333")
        linestyle = "--" if strategy == "true_cost_weighting" else "-"
        lw = 2.2 if strategy == "true_cost_weighting" else 1.5

        ax.plot(
            [PREVALENCE_LABELS.get(p, str(p)) for p in prevalences],
            vals,
            marker="o", color=color,
            linestyle=linestyle, linewidth=lw,
            label=strategy.replace("_", " "),
        )

    ax.set_xlabel("Target Training Prevalence", fontsize=LABEL_SIZE)
    ax.set_ylabel("PR-AUC (Test, mean over models)", fontsize=LABEL_SIZE)
    ax.set_title("PR-AUC Across Prevalence Levels by Strategy",
                 fontsize=TITLE_SIZE)
    ax.legend(fontsize=TICK_SIZE, loc="best")

    fig.tight_layout()
    _save(fig, "fig3_pr_auc_by_prevalence.png")


# ---------------------------------------------------------------------------
# Fig 4 — Horizontal bar: FP/TP ratio
# ---------------------------------------------------------------------------

def fig4_fp_tp_ratio(combined: pd.DataFrame) -> None:
    sub = combined.dropna(subset=["fp_tp_ratio"]).copy()
    sub = sub[sub["fp_tp_ratio"] < 1e6]  # remove extreme outliers

    sub["label"] = (
        sub["model"].map({"xgboost": "XGB", "random_forest": "RF"})
        + " | "
        + sub["strategy"].str.replace("_", " ")
        + " | "
        + sub["target_prevalence"].map(PREVALENCE_LABELS).fillna(sub["target_prevalence"].astype(str))
    )

    sub = sub.sort_values("fp_tp_ratio")

    fig, ax = plt.subplots(figsize=(12, max(6, len(sub) * 0.35)))

    colors = [MODEL_COLORS.get(m, "#555") for m in sub["model"]]
    bars = ax.barh(
        sub["label"], sub["fp_tp_ratio"],
        color=colors, alpha=0.85, edgecolor="white",
    )

    for bar, val in zip(bars, sub["fp_tp_ratio"]):
        ax.text(
            val + sub["fp_tp_ratio"].max() * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val:,.0f}",
            va="center", fontsize=8,
        )

    model_patches = [
        mpatches.Patch(color=MODEL_COLORS[m], label=MODEL_LABELS[m])
        for m in ["xgboost", "random_forest"]
    ]
    ax.legend(handles=model_patches, fontsize=TICK_SIZE)

    ax.set_xlabel("False Positives per True Positive", fontsize=LABEL_SIZE)
    ax.set_title("Alert Load: FP/TP Ratio per Run (lower = better for operations)",
                 fontsize=TITLE_SIZE)
    ax.tick_params(axis="y", labelsize=8)

    fig.tight_layout()
    _save(fig, "fig4_fp_tp_ratio.png")


# ---------------------------------------------------------------------------
# Fig 5 — Part B vs best Part A comparison
# ---------------------------------------------------------------------------

def fig5_part_b_comparison(combined: pd.DataFrame) -> None:
    metrics = ["pr_auc", "recall", "precision"]
    metric_labels = ["PR-AUC", "Recall", "Precision"]

    part_a = combined[combined["part"] == "A"]
    part_b = combined[combined["part"] == "B"]

    def best_by_model(df: pd.DataFrame, model: str, metric: str) -> float:
        sub = df[df["model"] == model]
        if sub.empty:
            return 0.0
        return float(sub[metric].max())

    groups = {
        "Part B\nXGBoost":       ("B", "xgboost",       "#F44336"),
        "Part B\nRandom Forest": ("B", "random_forest",  "#E57373"),
        "Part A best\nXGBoost":  ("A", "xgboost",        "#1565C0"),
        "Part A best\nRF":       ("A", "random_forest",  "#2E7D32"),
    }

    x = np.arange(len(metrics))
    width = 0.18
    fig, ax = plt.subplots(figsize=(11, 6))

    for i, (group_label, (part, model, color)) in enumerate(groups.items()):
        df_src = part_b if part == "B" else part_a
        vals = [best_by_model(df_src, model, m) for m in metrics]
        offset = (i - 1.5) * width
        bars = ax.bar(
            x + offset, vals, width,
            label=group_label, color=color, alpha=0.85,
            edgecolor="white", linewidth=0.5,
        )
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.002,
                f"{val:.3f}",
                ha="center", va="bottom", fontsize=8, rotation=45,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=LABEL_SIZE)
    ax.set_ylabel("Score (at optimal threshold)", fontsize=LABEL_SIZE)
    ax.set_title(
        "Part B (true_cost_weighting) vs. Best Part A Results",
        fontsize=TITLE_SIZE,
    )
    ax.legend(fontsize=TICK_SIZE, ncol=2)
    ax.set_ylim(0, 1.05)

    fig.tight_layout()
    _save(fig, "fig5_part_b_comparison.png")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save(fig: plt.Figure, filename: str) -> None:
    out = _FIGURES_DIR / filename
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Loading data ...")
    df_a, df_b, combined = load_data()

    print("Generating figures ...")
    fig1_pr_auc_by_strategy(combined)
    fig2_recall_precision_scatter(combined)
    fig3_pr_auc_by_prevalence(combined)
    fig4_fp_tp_ratio(combined)
    fig5_part_b_comparison(combined)

    print(f"\nAll figures saved to: {_FIGURES_DIR}")


if __name__ == "__main__":
    main()
