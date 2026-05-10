from pathlib import Path
import json
import pandas as pd

pd.set_option("display.max_columns", 80)
pd.set_option("display.width", 220)
pd.set_option("display.max_colwidth", 80)

ROOT = Path(__file__).resolve().parent
TABLES = ROOT / "tables"

print("=" * 100)
print("STRATEGIE 6 RESULT INSPECTION")
print("=" * 100)
print("Root:", ROOT)
print("Tables:", TABLES)

def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f"\nMISSING: {path}")
        return pd.DataFrame()
    df = pd.read_csv(path)
    print(f"\nLoaded: {path}")
    print(f"Shape: {df.shape}")
    return df

def show(df: pd.DataFrame, title: str, cols=None, sort_by=None, ascending=False):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)

    if df.empty:
        print("EMPTY / MISSING")
        return

    out = df.copy()

    if sort_by and sort_by in out.columns:
        out = out.sort_values(sort_by, ascending=ascending)

    if cols:
        existing = [c for c in cols if c in out.columns]
        missing = [c for c in cols if c not in out.columns]
        if missing:
            print("Missing columns:", missing)
        out = out[existing]

    print(out.to_string(index=False))

# ---------------------------------------------------------------------
# 1) Load relevant tables
# ---------------------------------------------------------------------

table1a = load_csv(TABLES / "table1a_main_results.csv")
table1b = load_csv(TABLES / "table1b_appendix_results.csv")
table2 = load_csv(TABLES / "table2_strategy6_comparison.csv")
table6 = load_csv(TABLES / "table6_pai_hnu_vs_part_a.csv")
part_a = load_csv(ROOT / "part_a_summary_v2.csv")

# ---------------------------------------------------------------------
# 2) Main Part-B / Strategy-6 comparison
# ---------------------------------------------------------------------

preferred_cols = [
    "model",
    "strategy",
    "target_prevalence",
    "pr_auc_test",
    "roc_auc_test",
    "precision_test_opt",
    "recall_test_opt",
    "f1_test_opt",
    "f2_test_opt",
    "tp_test_opt",
    "fp_test_opt",
    "optimal_threshold_val",
]

show(
    table2,
    "TABLE 2 — Strategy 6 comparison",
    cols=preferred_cols,
    sort_by="target_prevalence",
    ascending=True,
)

show(
    table6,
    "TABLE 6 — PAI-HNU vs Part A",
    sort_by="f1_test_opt" if "f1_test_opt" in table6.columns else None,
    ascending=False,
)

# ---------------------------------------------------------------------
# 3) Best rows by important metrics
# ---------------------------------------------------------------------

for df_name, df in [
    ("table1a_main_results", table1a),
    ("table2_strategy6_comparison", table2),
    ("table6_pai_hnu_vs_part_a", table6),
]:
    if df.empty:
        continue

    for metric in ["pr_auc_test", "f1_test_opt", "precision_test_opt", "recall_test_opt"]:
        if metric in df.columns:
            show(
                df.nlargest(min(10, len(df)), metric),
                f"TOP by {metric} — {df_name}",
                cols=[
                    "model",
                    "strategy",
                    "target_prevalence",
                    "pr_auc_test",
                    "roc_auc_test",
                    "precision_test_opt",
                    "recall_test_opt",
                    "f1_test_opt",
                    "tp_test_opt",
                    "fp_test_opt",
                ],
            )

# ---------------------------------------------------------------------
# 4) Part-A XGBoost baseline reference
# ---------------------------------------------------------------------

if not part_a.empty:
    print("\n" + "=" * 100)
    print("PART-A XGBOOST BASELINE REFERENCE")
    print("=" * 100)

    mask = (
        part_a.get("model", "").astype(str).str.lower().eq("xgboost")
        & part_a.get("strategy", "").astype(str).str.lower().eq("baseline")
    )

    baseline = part_a[mask].copy()

    baseline_cols = [
        "model",
        "strategy",
        "target_prevalence",
        "pr_auc_test",
        "roc_auc_test",
        "precision_test_thresh",
        "recall_test_thresh",
        "f1_test_thresh",
        "tp_test_thresh",
        "fp_test_thresh",
        "optimal_threshold",
    ]

    existing = [c for c in baseline_cols if c in baseline.columns]
    print(baseline[existing].to_string(index=False))

# ---------------------------------------------------------------------
# 5) Threshold info JSON, if available
# ---------------------------------------------------------------------

threshold_path = ROOT / "threshold_info_strategy6.json"
if threshold_path.exists():
    print("\n" + "=" * 100)
    print("THRESHOLD INFO STRATEGY 6")
    print("=" * 100)
    with open(threshold_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(json.dumps(data, indent=2)[:4000])
else:
    print("\nNo threshold_info_strategy6.json found.")