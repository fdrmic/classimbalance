# Part A v2 — Results Overview

**Input:** `results\part_a_summary_v2.csv`

## Table — Per condition (test metrics)

| model         | strategy             | target_prevalence   |   pr_auc_test |   precision_test_thresh |   recall_test_thresh |   f1_test_thresh |   tp_test_thresh |   fp_test_thresh |   fn_test_thresh |
|:--------------|:---------------------|:--------------------|--------------:|------------------------:|---------------------:|-----------------:|-----------------:|-----------------:|-----------------:|
| random_forest | baseline             | 0.100%              |        0.026  |                  0.0366 |               0.1992 |           0.0619 |             3922 |           103128 |            15764 |
| random_forest | baseline             | 0.500%              |        0.026  |                  0.0366 |               0.1992 |           0.0619 |             3922 |           103128 |            15764 |
| random_forest | baseline             | 1.000%              |        0.026  |                  0.0366 |               0.1992 |           0.0619 |             3922 |           103128 |            15764 |
| random_forest | random_undersampling | 0.100%              |        0.0294 |                  0.0577 |               0.1904 |           0.0885 |             3749 |            61244 |            15937 |
| random_forest | random_undersampling | 0.500%              |        0.0566 |                  0.074  |               0.2436 |           0.1136 |             4795 |            59968 |            14891 |
| random_forest | random_undersampling | 1.000%              |        0.0582 |                  0.0795 |               0.2336 |           0.1186 |             4599 |            53272 |            15087 |
| random_forest | smote                | 0.100%              |        0.0238 |                  0.0513 |               0.1872 |           0.0805 |             3685 |            68153 |            16001 |
| random_forest | smote                | 0.500%              |        0.045  |                  0.0622 |               0.2614 |           0.1005 |             5145 |            77538 |            14541 |
| random_forest | smote                | 1.000%              |        0.0484 |                  0.0629 |               0.2678 |           0.1018 |             5272 |            78575 |            14414 |
| random_forest | adasyn               | 0.100%              |        0.0283 |                  0.0527 |               0.1948 |           0.0829 |             3835 |            68950 |            15851 |
| random_forest | adasyn               | 0.500%              |        0.0398 |                  0.0568 |               0.2178 |           0.0901 |             4288 |            71212 |            15398 |
| random_forest | adasyn               | 1.000%              |        0.0398 |                  0.0568 |               0.2178 |           0.0901 |             4288 |            71212 |            15398 |
| random_forest | class_weighting      | 0.100%              |        0.0349 |                  0.064  |               0.2157 |           0.0987 |             4247 |            62163 |            15439 |
| random_forest | class_weighting      | 0.500%              |        0.0365 |                  0.0627 |               0.2183 |           0.0975 |             4297 |            64198 |            15389 |
| random_forest | class_weighting      | 1.000%              |        0.0377 |                  0.0685 |               0.1861 |           0.1002 |             3664 |            49787 |            16022 |
| xgboost       | baseline             | 0.100%              |        0.1079 |                  0.0909 |               0.2393 |           0.1318 |             4710 |            47091 |            14976 |
| xgboost       | baseline             | 0.500%              |        0.1079 |                  0.0909 |               0.2393 |           0.1318 |             4710 |            47091 |            14976 |
| xgboost       | baseline             | 1.000%              |        0.1079 |                  0.0909 |               0.2393 |           0.1318 |             4710 |            47091 |            14976 |
| xgboost       | random_undersampling | 0.100%              |        0.0924 |                  0.0863 |               0.2207 |           0.124  |             4344 |            46019 |            15342 |
| xgboost       | random_undersampling | 0.500%              |        0.0976 |                  0.097  |               0.2139 |           0.1334 |             4210 |            39209 |            15476 |
| xgboost       | random_undersampling | 1.000%              |        0.0882 |                  0.0917 |               0.2318 |           0.1314 |             4563 |            45211 |            15123 |
| xgboost       | smote                | 0.100%              |        0.0719 |                  0.0385 |               0.3667 |           0.0697 |             7218 |           180184 |            12468 |
| xgboost       | smote                | 0.500%              |        0.0219 |                  0.0325 |               0.4068 |           0.0603 |             8009 |           238107 |            11677 |
| xgboost       | smote                | 1.000%              |        0.0552 |                  0.0618 |               0.2823 |           0.1014 |             5558 |            84358 |            14128 |
| xgboost       | adasyn               | 0.100%              |        0.0989 |                  0.0833 |               0.2846 |           0.1289 |             5603 |            61670 |            14083 |
| xgboost       | adasyn               | 0.500%              |        0.0316 |                  0.0325 |               0.4145 |           0.0603 |             8159 |           242574 |            11527 |
| xgboost       | adasyn               | 1.000%              |        0.0316 |                  0.0325 |               0.4145 |           0.0603 |             8159 |           242574 |            11527 |
| xgboost       | class_weighting      | 0.100%              |        0.0558 |                  0.0746 |               0.2772 |           0.1175 |             5456 |            67704 |            14230 |
| xgboost       | class_weighting      | 0.500%              |        0.0706 |                  0.0854 |               0.2542 |           0.1279 |             5004 |            53585 |            14682 |
| xgboost       | class_weighting      | 1.000%              |        0.0752 |                  0.0937 |               0.2407 |           0.1349 |             4738 |            45840 |            14948 |

## Rankings — per model & prevalence

| model         | target_prevalence   | strategy             |   pr_auc_test |   f1_test_thresh |   rank_pr_auc |   rank_f1 |
|:--------------|:--------------------|:---------------------|--------------:|-----------------:|--------------:|----------:|
| random_forest | 0.100%              | class_weighting      |        0.0349 |           0.0987 |             1 |         1 |
| random_forest | 0.100%              | random_undersampling |        0.0294 |           0.0885 |             2 |         2 |
| random_forest | 0.100%              | adasyn               |        0.0283 |           0.0829 |             3 |         3 |
| random_forest | 0.100%              | smote                |        0.0238 |           0.0805 |             5 |         4 |
| random_forest | 0.100%              | baseline             |        0.026  |           0.0619 |             4 |         5 |
| random_forest | 0.500%              | random_undersampling |        0.0566 |           0.1136 |             1 |         1 |
| random_forest | 0.500%              | smote                |        0.045  |           0.1005 |             2 |         2 |
| random_forest | 0.500%              | class_weighting      |        0.0365 |           0.0975 |             4 |         3 |
| random_forest | 0.500%              | adasyn               |        0.0398 |           0.0901 |             3 |         4 |
| random_forest | 0.500%              | baseline             |        0.026  |           0.0619 |             5 |         5 |
| random_forest | 1.000%              | random_undersampling |        0.0582 |           0.1186 |             1 |         1 |
| random_forest | 1.000%              | smote                |        0.0484 |           0.1018 |             2 |         2 |
| random_forest | 1.000%              | class_weighting      |        0.0377 |           0.1002 |             4 |         3 |
| random_forest | 1.000%              | adasyn               |        0.0398 |           0.0901 |             3 |         4 |
| random_forest | 1.000%              | baseline             |        0.026  |           0.0619 |             5 |         5 |
| xgboost       | 0.100%              | baseline             |        0.1079 |           0.1318 |             1 |         1 |
| xgboost       | 0.100%              | adasyn               |        0.0989 |           0.1289 |             2 |         2 |
| xgboost       | 0.100%              | random_undersampling |        0.0924 |           0.124  |             3 |         3 |
| xgboost       | 0.100%              | class_weighting      |        0.0558 |           0.1175 |             5 |         4 |
| xgboost       | 0.100%              | smote                |        0.0719 |           0.0697 |             4 |         5 |
| xgboost       | 0.500%              | random_undersampling |        0.0976 |           0.1334 |             2 |         1 |
| xgboost       | 0.500%              | baseline             |        0.1079 |           0.1318 |             1 |         2 |
| xgboost       | 0.500%              | class_weighting      |        0.0706 |           0.1279 |             3 |         3 |
| xgboost       | 0.500%              | adasyn               |        0.0316 |           0.0603 |             4 |         4 |
| xgboost       | 0.500%              | smote                |        0.0219 |           0.0603 |             5 |         5 |
| xgboost       | 1.000%              | class_weighting      |        0.0752 |           0.1349 |             3 |         1 |
| xgboost       | 1.000%              | baseline             |        0.1079 |           0.1318 |             1 |         2 |
| xgboost       | 1.000%              | random_undersampling |        0.0882 |           0.1314 |             2 |         3 |
| xgboost       | 1.000%              | smote                |        0.0552 |           0.1014 |             4 |         4 |
| xgboost       | 1.000%              | adasyn               |        0.0316 |           0.0603 |             5 |         5 |

## Context / How to read this report
- **PR-AUC (test)** is threshold-free and captures ranking quality under extreme imbalance.
- **Precision/Recall/F1 (test, `*_thresh`)** are computed at a threshold optimized on the validation split and then applied to test.
- **FP (false positives)** is used as a simple **workload proxy** (alerts to investigate).

## Key findings (quick takeaways)
- **Best F1 (test@thresh)** for `random_forest` at prevalence **0.100%**: `class_weighting` (F1=0.0987, Prec=0.0640, Rec=0.2157, FP=62,163).
- **Best F1 (test@thresh)** for `random_forest` at prevalence **0.500%**: `random_undersampling` (F1=0.1136, Prec=0.0740, Rec=0.2436, FP=59,968).
- **Best F1 (test@thresh)** for `random_forest` at prevalence **1.000%**: `random_undersampling` (F1=0.1186, Prec=0.0795, Rec=0.2336, FP=53,272).
- **Best F1 (test@thresh)** for `xgboost` at prevalence **0.100%**: `baseline` (F1=0.1318, Prec=0.0909, Rec=0.2393, FP=47,091).
- **Best F1 (test@thresh)** for `xgboost` at prevalence **0.500%**: `random_undersampling` (F1=0.1334, Prec=0.0970, Rec=0.2139, FP=39,209).
- **Best F1 (test@thresh)** for `xgboost` at prevalence **1.000%**: `class_weighting` (F1=0.1349, Prec=0.0937, Rec=0.2407, FP=45,840).

## Q1 — Strategy comparison (PR-AUC, Recall, Precision)
### Model: `random_forest` (means over prevalences)
| strategy             |   pr_auc |   precision |   recall |      fp |
|:---------------------|---------:|------------:|---------:|--------:|
| random_undersampling |   0.0481 |      0.0704 |   0.2225 |  58,161 |
| smote                |   0.0391 |      0.0588 |   0.2388 |  74,755 |
| class_weighting      |   0.0364 |      0.0651 |   0.2067 |  58,716 |
| adasyn               |   0.036  |      0.0554 |   0.2101 |  70,458 |
| baseline             |   0.026  |      0.0366 |   0.1992 | 103,128 |
### Model: `xgboost` (means over prevalences)
| strategy             |   pr_auc |   precision |   recall |      fp |
|:---------------------|---------:|------------:|---------:|--------:|
| baseline             |   0.1079 |      0.0909 |   0.2393 |  47,091 |
| random_undersampling |   0.0927 |      0.0916 |   0.2221 |  43,480 |
| class_weighting      |   0.0672 |      0.0846 |   0.2573 |  55,710 |
| adasyn               |   0.054  |      0.0495 |   0.3712 | 182,273 |
| smote                |   0.0497 |      0.0443 |   0.3519 | 167,550 |

## Q2 — Robustness across prevalence levels
| model         | strategy             | pr_auc_test     | precision@thresh (test)   | recall@thresh (test)   | f1@thresh (test)   | FP@thresh (test)   |
|:--------------|:---------------------|:----------------|:--------------------------|:-----------------------|:-------------------|:-------------------|
| random_forest | baseline             | 0.0260 ± 0.0000 | 0.0366 ± 0.0000           | 0.1992 ± 0.0000        | 0.0619 ± 0.0000    | 103,128 ± 0        |
| random_forest | random_undersampling | 0.0481 ± 0.0162 | 0.0704 ± 0.0113           | 0.2225 ± 0.0282        | 0.1069 ± 0.0161    | 58,161 ± 4,282     |
| random_forest | smote                | 0.0391 ± 0.0134 | 0.0588 ± 0.0065           | 0.2388 ± 0.0448        | 0.0943 ± 0.0119    | 74,755 ± 5,741     |
| random_forest | adasyn               | 0.0360 ± 0.0066 | 0.0554 ± 0.0024           | 0.2101 ± 0.0133        | 0.0877 ± 0.0041    | 70,458 ± 1,306     |
| random_forest | class_weighting      | 0.0364 ± 0.0014 | 0.0651 ± 0.0031           | 0.2067 ± 0.0179        | 0.0988 ± 0.0014    | 58,716 ± 7,799     |
| xgboost       | baseline             | 0.1079 ± 0.0000 | 0.0909 ± 0.0000           | 0.2393 ± 0.0000        | 0.1318 ± 0.0000    | 47,091 ± 0         |
| xgboost       | random_undersampling | 0.0927 ± 0.0047 | 0.0916 ± 0.0054           | 0.2221 ± 0.0091        | 0.1296 ± 0.0049    | 43,480 ± 3,721     |
| xgboost       | smote                | 0.0497 ± 0.0254 | 0.0443 ± 0.0155           | 0.3519 ± 0.0635        | 0.0771 ± 0.0216    | 167,550 ± 77,649   |
| xgboost       | adasyn               | 0.0540 ± 0.0389 | 0.0495 ± 0.0293           | 0.3712 ± 0.0750        | 0.0832 ± 0.0396    | 182,273 ± 104,445  |
| xgboost       | class_weighting      | 0.0672 ± 0.0101 | 0.0846 ± 0.0096           | 0.2573 ± 0.0184        | 0.1267 ± 0.0087    | 55,710 ± 11,086    |

Interpretation guide: small std (±) indicates robustness across prevalence targets; large std suggests sensitivity to imbalance level.

## Q3 — Recall/Precision trade-offs and operational workload proxy
We approximate *operational workload* with the number of false positives (FP) at the chosen threshold (selected on val, applied to test). Higher recall typically increases FP.
| model         | strategy             |   precision |   recall |     f1 |      fp |
|:--------------|:---------------------|------------:|---------:|-------:|--------:|
| random_forest | random_undersampling |      0.0704 |   0.2225 | 0.1069 |  58,161 |
| random_forest | class_weighting      |      0.0651 |   0.2067 | 0.0988 |  58,716 |
| random_forest | smote                |      0.0588 |   0.2388 | 0.0943 |  74,755 |
| random_forest | adasyn               |      0.0554 |   0.2101 | 0.0877 |  70,458 |
| random_forest | baseline             |      0.0366 |   0.1992 | 0.0619 | 103,128 |
| xgboost       | baseline             |      0.0909 |   0.2393 | 0.1318 |  47,091 |
| xgboost       | random_undersampling |      0.0916 |   0.2221 | 0.1296 |  43,480 |
| xgboost       | class_weighting      |      0.0846 |   0.2573 | 0.1267 |  55,710 |
| xgboost       | adasyn               |      0.0495 |   0.3712 | 0.0832 | 182,273 |
| xgboost       | smote                |      0.0443 |   0.3519 | 0.0771 | 167,550 |

Operational implication: strategies that boost recall by lowering the threshold can produce very large FP volumes; this increases alert review workload for AML teams. A practically useful strategy should improve recall while keeping precision (and FP) at manageable levels.
