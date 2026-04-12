# Part B v2 — Strategy 6 (`smote_class_weighting`) Overview

**Inputs:** `results\part_b_summary_v2.csv` and `results\part_a_summary_v2.csv`

## Context / How to read this report
- **PR-AUC (test)** is threshold-free and captures ranking quality.
- **Precision/Recall/F1 (test, `*_thresh`)** use a threshold optimized on validation and applied to test.
- **FP (false positives)** is used as a workload proxy.

## Strategy 6 results (per prevalence)

| model   | strategy              | target_prevalence   |   pr_auc_test |   precision_test_thresh |   recall_test_thresh |   f1_test_thresh |   tp_test_thresh |   fp_test_thresh |   fn_test_thresh |
|:--------|:----------------------|:--------------------|--------------:|------------------------:|---------------------:|-----------------:|-----------------:|-----------------:|-----------------:|
| xgboost | smote_class_weighting | 0.100%              |        0.05   |                  0.0679 |               0.286  |           0.1098 |             5630 |           77,232 |            14056 |
| xgboost | smote_class_weighting | 0.500%              |        0.045  |                  0.0554 |               0.3089 |           0.0939 |             6081 |          103,750 |            13605 |
| xgboost | smote_class_weighting | 1.000%              |        0.0429 |                  0.0465 |               0.3245 |           0.0813 |             6389 |          131,025 |            13297 |

## Q4 — Does Strategy 6 improve the balance vs standard strategies?

Comparison setup: **XGBoost only**; for each prevalence level we compare Strategy 6 against the **best standard strategy** (baseline, random_undersampling, smote, adasyn, class_weighting) chosen by highest `f1_test_thresh`.

| target_prevalence   | best_standard_strategy   |   PR-AUC test (best std) |   PR-AUC test (S6) |   Precision@thresh test (best std) |   Precision@thresh test (S6) |   Recall@thresh test (best std) |   Recall@thresh test (S6) |   F1@thresh test (best std) |   F1@thresh test (S6) |   FP@thresh test (best std) |   FP@thresh test (S6) |
|:--------------------|:-------------------------|-------------------------:|-------------------:|-----------------------------------:|-----------------------------:|--------------------------------:|--------------------------:|----------------------------:|----------------------:|----------------------------:|----------------------:|
| 0.100%              | baseline                 |                   0.1079 |             0.05   |                             0.0909 |                       0.0679 |                          0.2393 |                    0.286  |                      0.1318 |                0.1098 |                      47,091 |                77,232 |
| 0.500%              | random_undersampling     |                   0.0976 |             0.045  |                             0.097  |                       0.0554 |                          0.2139 |                    0.3089 |                      0.1334 |                0.0939 |                      39,209 |               103,750 |
| 1.000%              | class_weighting          |                   0.0752 |             0.0429 |                             0.0937 |                       0.0465 |                          0.2407 |                    0.3245 |                      0.1349 |                0.0813 |                      45,840 |               131,025 |

### Interpretation
- If Strategy 6 increases **F1** and/or **Recall** at comparable **PR-AUC** *without* exploding **FP**, it achieves a superior operational balance.
- If Strategy 6 mainly increases recall but FP rises sharply, it may be unsuitable in practice due to alert overload.


### Quantified conclusion (from the table above)

- **0.100%**: S6 vs best standard -> ΔPR-AUC=-0.0579, ΔF1=-0.0220, ΔFP=+30,141
- **0.500%**: S6 vs best standard -> ΔPR-AUC=-0.0526, ΔF1=-0.0395, ΔFP=+64,541
- **1.000%**: S6 vs best standard -> ΔPR-AUC=-0.0323, ΔF1=-0.0536, ΔFP=+85,185

Summary across prevalences: S6 improves **F1** in 0/3 cases, improves **PR-AUC** in 0/3 cases, and increases **FP** in 3/3 cases.
