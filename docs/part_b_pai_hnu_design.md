# Part B — Strategy 6: Part-A-Informed Hard-Negative Undersampling (PAI-HNU)

> Purpose: define a single, defendable, Part-A-informed mitigation
> strategy that addresses the empirical weaknesses observed in Part A
> without changing the model family, the splits, or the feature set.

---

## 1. Sub-Question 4 (revised)

**SQ-4 (Part B)** — *Does a sampling strategy that is **explicitly informed
by the failure modes observed in Part A** — namely (a) the predictability
ceiling of the XGBoost Baseline at the natural prevalence of ~0.05 %,
(b) the operational efficiency of Random Undersampling, and (c) the
false-positive inflation produced by SMOTE/ADASYN — recover or surpass
the operational characteristics of the best Part A configurations on
the held-out test split, without modifying the model family,
hyperparameters, splits, or features?*

The phrase **"informed by"** has a precise meaning in this work: each
design decision in PAI-HNU is traceable to one specific Part A finding
(see §3 — Mapping Table). It is **not** a hyperparameter sweep, a new
model, or a re-engineering of features.

---

## 2. Motivation from Part A

The Part A grid (XGBoost × {Baseline, RUS, SMOTE, ADASYN, Class-Weighting}
× {0.1 %, 0.5 %, 1.0 %} target prevalences) produced four findings that
directly motivate PAI-HNU.

| # | Finding (Part A)                                                                                  | Source row(s) in `results/part_a_summary_v2.csv`                              |
|---|---------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------|
| 1 | XGBoost dominates RF on PR-AUC at every prevalence.                                               | All `random_forest__*` rows have `pr_auc_test` ≤ 0.06; all `xgboost__*` rows ≥ 0.07. |
| 2 | XGBoost Baseline already reaches PR-AUC ≈ 0.108 with no resampling — a **ceiling** for the rest. | `xgboost__baseline__p001` → `pr_auc_test = 0.10785`.                         |
| 3 | RUS is operationally efficient: best F1 / lowest FP-per-TP among XGBoost rows.                    | `xgboost__random_undersampling__p010` → `f1_test_thresh = 0.131`, `fp_per_tp ≈ 9.9`. |
| 4 | SMOTE / ADASYN inflate FP/TP by 3–10× relative to RUS — they hurt operations.                    | `xgboost__smote__p001` → `fp_per_tp ≈ 24.96`; `xgboost__adasyn__p010` → `fp_per_tp ≈ 29.74`. |
| 5 | Class Weighting alone does not lift PR-AUC above the Baseline.                                   | `xgboost__class_weighting__p005` → `pr_auc_test = 0.07060`.                  |

---

## 3. Mapping table — Part A finding → observed weakness → design decision

| Part A finding                                                  | Observed weakness                                                              | PAI-HNU design decision                                                                                  |
|-----------------------------------------------------------------|---------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------|
| XGBoost > RF on PR-AUC at every prevalence                      | RF is structurally limited (`max_samples=200_000` for RAM)                     | Use **XGBoost only**; same hyperparameters as `models/factory.py`. No RF in Part B.                     |
| Baseline PR-AUC ceiling                                         | Random data does not expose the baseline's hardest cases                       | **Hard-Negative pool (50 %)**: top-scored negatives from the Part-A baseline are kept in training.       |
| RUS is operationally strong                                     | RUS discards information uniformly                                             | **Random pools (25 % temporal + 25 % global)** preserve coverage; baseline-hard examples replace blind discards. |
| SMOTE/ADASYN inflate FP/TP                                      | Synthetic minorities pull the decision boundary into low-density majority area | **No oversampling**; we work strictly with real negatives. Synthetic minorities are not introduced.       |
| Class Weighting alone is insufficient                           | Re-weighting cannot re-shape the training distribution                         | PAI-HNU **constructs the training set itself**; XGBoost runs with `scale_pos_weight=None`.              |

---

## 4. Method

### 4.1 Inputs (immutable)

* Same temporal train / val / test split as Part A
  (`splits_v2/{train,val,test}.parquet`, manifest in `splits_v2/split_manifest.json`).
* Same 30-feature pipeline (`features/pipeline.py`), cache reused.
* Same XGBoost factory (`models/factory.py:_xgboost`) — no hyperparameter tuning.
* `seed = 42` (from `configs/experiment.yaml`).

### 4.2 Outputs (per target prevalence)

```
outputs/runs_part_b_pai_hnu/<run_id>/
    run_config.json               # parameters, counts, timestamps, sha256 of score cache
    sampling_manifest.json        # how this training set was constructed
    model.pkl                     # trained XGBoost model
    metrics_val.json   / .csv     # @ default threshold 0.5
    metrics_test.json  / .csv
    metrics_val_opt.json   / .csv # @ F1-optimal threshold from val
    metrics_test_opt.json  / .csv
```

### 4.3 Algorithm

```
Inputs : y_train (full),
         baseline_train_scores (cached),
         target_prevalence p ∈ {0.001, 0.005, 0.010},
         shares (h, t, g) = (0.50, 0.25, 0.25),
         hard_cap_multiplier = 20

Output : index list selected_idx ⊂ {0, …, n_train − 1}

1.  pos_idx ← positions with y_train == 1
    neg_idx ← positions with y_train == 0
    n_pos = |pos_idx|

2.  n_neg_target = round(n_pos · (1 − p) / p)
    n_hard_planned    = round(h · n_neg_target)
    n_temporal_planned = round(t · n_neg_target)
    n_global_planned   = n_neg_target − n_hard_planned − n_temporal_planned

3.  HARD CAP
    n_hard_cap     = hard_cap_multiplier · n_pos
    n_hard_actual  = min(n_hard_planned, n_hard_cap, |neg_idx|)
    leftover       = n_hard_planned − n_hard_actual
    n_temporal_planned += leftover // 2
    n_global_planned   += leftover − leftover // 2

4.  hard_neg_idx     = top-k(baseline_scores[neg_idx], k = n_hard_actual)        # np.argpartition
    remaining_neg    = neg_idx \ hard_neg_idx
    temporal_neg_idx = stratified_sample(remaining_neg, k = n_temporal_planned, n_blocks = 20, rng)
    rest_pool        = remaining_neg \ temporal_neg_idx
    global_neg_idx   = uniform_sample(rest_pool, k = n_global_planned + shortfall_from_temporal, rng)

5.  selected_idx = pos_idx ∪ hard_neg_idx ∪ temporal_neg_idx ∪ global_neg_idx
    assert disjoint(four pools)
```

`np.argpartition` is preferred over a full sort for the hard-negative
selection: O(N) instead of O(N log N), with deterministic tie-breaking
on a given input. Memory footprint (negative-only): ≈ `4 N` bytes for the
score view + `8 N` bytes for the index permutation; for `N ≈ 1.23 ⋅ 10⁸`
this is ≈ 1.4 GB.

### 4.4 Why this construction?

1. **All positives are kept** — there is no information loss on the
   minority class, which is in line with RUS-style strategies.
2. **Hard negatives** force the model to revisit the cases the baseline
   already finds confusing. They are the empirical equivalent of
   "hard-example mining" without the need for online retraining.
3. **Temporal-stratified random** preserves coverage across the training
   period; it keeps the model exposed to negatives outside the
   baseline-hard cluster (an explicit guard against over-fitting to the
   baseline's mistakes).
4. **Global random** is a safety pool for diversity. It is also the
   target for any quota leftover if the hard cap binds.
5. **No SMOTE / ADASYN** — Part A showed these expand the decision
   region into low-density majority area and cost FP/TP.

---

## 5. Anti-leakage guarantees

| ID  | Rule                                                                                                    | Where enforced                                                                                          |
|-----|---------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------|
| AL1 | Validation and test feature caches are loaded but **never passed to the sampler**.                      | `experiments/run_part_b_pai_hnu.py:run` — `X_val/X_test` are only used in `model.predict_proba`.        |
| AL2 | The Part-A baseline scores used for hard mining are computed strictly on the training split.            | `experiments/score_baseline_train.py` — only `load_features(splits_dir, "train")` is called.            |
| AL3 | The optimal F1-threshold is derived from the validation split only and applied verbatim to test.        | `find_optimal_threshold(y_val, val_score, criterion="f1")` (mirror of Part A).                          |
| AL4 | Score cache is content-hashed (sha256) and recorded in every run's manifest.                            | `_file_sha256` in `experiments/run_part_b_pai_hnu.py`.                                                  |
| AL5 | Smoke runs are written to a separate `outputs/runs_part_b_pai_hnu_smoke/` tree and tagged `_SMOKE`.     | `_SMOKE_OUTPUTS_DIR_NAME` constant.                                                                     |

---

## 6. Configuration surface

### `configs/benchmark_part_b_pai_hnu.yaml`

| Field                                | Default        | Meaning                                                  |
|--------------------------------------|----------------|----------------------------------------------------------|
| `target_prevalences`                 | [0.001, 0.005, 0.010] | One run per value.                                |
| `sampling.hard_negative_share`       | 0.50           | Fraction of `n_neg_target` drawn from baseline-hard.     |
| `sampling.temporal_random_share`     | 0.25           | Fraction drawn temporally-stratified.                    |
| `sampling.global_random_share`       | 0.25           | Fraction drawn globally at random.                       |
| `sampling.n_temporal_blocks`         | 20             | Number of equal-size temporal blocks for stratification. |
| `sampling.fill_shortfall_from_global`| true           | Redirect temporal shortfall to global pool.              |
| `sampling.hard_negative_cap_multiplier` | 20          | `n_hard_actual ≤ multiplier × n_pos`.                     |
| `evaluation.default_threshold`       | 0.5            | Fixed threshold for `metrics_test.json`.                 |
| `evaluation.optimal_threshold_criterion` | "f1"       | Identical to Part A re-evaluation.                       |
| `xgboost.use_factory_defaults`       | true           | Same hyperparameters as Part A.                          |
| `xgboost.class_weight`               | null           | No additional `scale_pos_weight`.                        |

### `configs/paths_large_part_b_pai_hnu.yaml`

Reuses `processed_dir` and `splits_dir` from Part A; only the
`outputs_dir`, `leaderboard_dir`, and `part_a_summary` filename change so
that Part-A artefacts are never overwritten. Optional
`baseline_model_path` provides a Drive / Colab override.

---

## 7. Success criteria

A PAI-HNU run is considered **successful** if **at least one** of the
three target-prevalence runs satisfies the following on the test split:

1. `pr_auc_test ≥ pr_auc_test(xgboost__baseline__p001)`
   ( ≥ 0.108 ) — the strategy must not destroy ranking quality.
2. `f1_test_thresh ≥ 0.95 × max F1` across the **six XGBoost anchor rows**
   used in Table 6 (Baseline best PR-AUC, RUS @0.5 %, Class Weighting @1.0 %,
   ADASYN best PR-AUC, SMOTE best PR-AUC, SMOTE best F1 —
   see `_select_part_a_xgboost_rows_for_table6` in `results_tables.py`).
   Example: ≥ 0.95 × 0.131 ≈ 0.124 at F1-optimal threshold when RUS @1 % sets the max among those six.
3. `fp_per_tp(test_thresh) ≤ fp_per_tp(xgboost__random_undersampling__p010)`

The thesis discussion will report all three across all three target
prevalences. **Even partial success is a defendable scientific result**
because it locates the failure mode of PAI-HNU on the same axes used to
critique Part A.

---

## 8. Smoke-test protocol

Four levels, executed in order, before any full run:

| # | Level                  | Command                                                                                                          | Run-time target |
|---|------------------------|------------------------------------------------------------------------------------------------------------------|------------------|
| 1 | Unit tests              | `pytest tests/test_pai_hnu_sampler.py -v`                                                                        | < 5 s            |
| 2 | Sampler smoke           | (Stub script) shape/no-overlap check on tiny synthetic data                                                      | < 5 s            |
| 3 | Mini end-to-end smoke   | `python -m aml_benchmark.experiments.run_part_b_pai_hnu --paths configs/paths_large_part_b_pai_hnu.yaml --target-prevalences 0.01 --sample-n-train 200000` | minutes          |
| 4 | Full runs               | `python -m aml_benchmark.experiments.run_part_b_pai_hnu --paths configs/paths_large_part_b_pai_hnu.yaml`         | ≈ 30–60 min      |

Mini-smoke (level 3) writes to `outputs/runs_part_b_pai_hnu_smoke/`,
tags artefacts with `_SMOKE`, and uses an **orig_row_idx–aligned** subsample:
after drawing row indices, rows are ordered by ascending `orig_row_idx`;
`X_sub`, `y_sub`, and baseline scores are built from the same sorted list
(asserted in code). Each smoke run directory includes
`subsample_row_mapping.parquet` (`internal_row_idx`, `orig_row_idx`) and
records `smoke_subsample_used`, `sample_n_train`, and
`row_index_mode: "internal_0_based_with_orig_row_idx_mapping"` (or full-train
mode when `--sample-n-train` ≥ train size) in `run_config.json` and
`sampling_manifest.json`. Full runs use `row_index_mode: "full_train_row_order"`.
