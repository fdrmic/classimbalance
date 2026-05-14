| Group | Configuration | PR-AUC | Threshold | Precision | Recall | F1 | TP | FP | FP_per_TP |
|---|---|---|---|---|---|---|---|---|---|
| Part A | xgboost baseline @0.1% (best PR-AUC; identical across baseline p001/p005/p010) | 0.1079 | 0.0405 | 0.0909 | 0.2393 | 0.1318 | 4,710 | 47,091 | 10.00 |
| Part A | xgboost random_undersampling @0.5% (F1-opt)  *Pareto | 0.0976 | 0.2952 | 0.0970 | 0.2139 | 0.1334 | 4,210 | 39,209 | 9.31 |
| Part A | xgboost class_weighting @1.0% (F1-opt)  *Pareto | 0.0752 | 0.7366 | 0.0937 | 0.2407 | 0.1349 | 4,738 | 45,840 | 9.67 |
| Part A | xgboost adasyn @0.1% (best PR-AUC)  *Pareto | 0.0989 | 0.0310 | 0.0833 | 0.2846 | 0.1289 | 5,603 | 61,670 | 11.01 |
| Part A | xgboost smote @0.1% (best PR-AUC)  *Pareto | 0.0719 | 0.0228 | 0.0385 | 0.3667 | 0.0697 | 7,218 | 180,184 | 24.96 |
| Part A | xgboost smote @1.0% (best F1) | 0.0552 | 0.0545 | 0.0618 | 0.2823 | 0.1014 | 5,558 | 84,358 | 15.18 |
| Part B PAI-HNU | PAI-HNU @0.1% (F1-opt) | 0.1080 | 0.0407 | 0.0926 | 0.2345 | 0.1328 | 4,617 | 45,240 | 9.80 |
| Part B PAI-HNU | PAI-HNU @0.5% (F1-opt)  *Pareto | 0.1050 | 0.0568 | 0.1134 | 0.1945 | 0.1433 | 3,829 | 29,941 | 7.82 |
| Part B PAI-HNU | PAI-HNU @1.0% (F1-opt) | 0.0893 | 0.0801 | 0.1079 | 0.1438 | 0.1233 | 2,831 | 23,416 | 8.27 |
