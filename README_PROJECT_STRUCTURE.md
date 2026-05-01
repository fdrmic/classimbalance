# AML Benchmark — Project Structure and Technical Documentation

**Internal project documentation · Bachelor Thesis · FS26**

---

## Table of Contents

1. [Project & Research Question](#1-project--research-question)
2. [Implementation Status](#2-implementation-status)
3. [Repository Structure and Module Mapping](#3-repository-structure-and-module-mapping)
4. [End-to-End Pipeline Walk-Through](#4-end-to-end-pipeline-walk-through)
5. [Dataset (LI-Large)](#5-dataset-li-large)
6. [Labeling Logic](#6-labeling-logic)
7. [Temporal Split](#7-temporal-split)
8. [Feature Engineering](#8-feature-engineering)
9. [Imbalance Mitigation Strategies](#9-imbalance-mitigation-strategies)
10. [Models and Hyperparameters](#10-models-and-hyperparameters)
11. [Evaluation Metrics and Threshold Policy](#11-evaluation-metrics-and-threshold-policy)
12. [Result Aggregation and Leaderboard](#12-result-aggregation-and-leaderboard)
13. [Part B — Custom Strategy and Multi-Threshold Analysis](#13-part-b--custom-strategy-and-multi-threshold-analysis)
14. [Reproducibility](#14-reproducibility)
15. [Hardware Context and Performance](#15-hardware-context-and-performance)
16. [Known Limitations](#16-known-limitations)
17. [Open Questions / TBD](#17-open-questions--tbd)
18. [Appendix — CLI Cheat-Sheet](#18-appendix--cli-cheat-sheet)

---

## 1. Project & Research Question

### Research question

> *How do different class-imbalance mitigation strategies affect detection performance in AML transaction monitoring under extreme class imbalance?*

### Project framing

The project implements a reproducible academic benchmark for **binary transaction-level AML classification** on the **IBM AMLworld synthetic dataset** (Altman et al., 2023, NeurIPS — [arXiv:2306.16424](https://arxiv.org/abs/2306.16424)). The active production variant is **Low-Illicit Large (LI-Large)** with a natural training-set prevalence of **0.052 %** (≈ 1 illicit per 1,930 transactions).

This is **not** a production fraud-detection system. It is a controlled experiment that isolates the effect of imbalance handling under fixed splits, fixed features, and fixed hyperparameters.

### Two-part design

| Part | Scope | Grid | Status |
|---|---|---|---|
| **Part A** | Five canonical mitigation strategies × two model families × three target prevalences | 30 runs | Complete |
| **Part B** | (i) Purpose-designed sixth strategy `true_cost_weighting`. (ii) Multi-strategy threshold optimisation on the Part A XGBoost Baseline. | 6 + 3 evaluations | Complete |

**Five canonical strategies (Part A):**

| Strategy | Mechanism |
|---|---|
| `baseline` | No imbalance handling (reference condition) |
| `random_undersampling` | Reduce majority class |
| `smote` | Synthetic minority oversampling (linear interpolation between k-NN) |
| `adasyn` | Adaptive synthetic oversampling (denser near decision boundary) |
| `class_weighting` | Cost-sensitive learning via `class_weight = {0: 1, 1: (1−p)/p}` |

**Three target training prevalences:** 1.0 %, 0.5 %, 0.1 %.

**Two model families:** Random Forest (`sklearn`), XGBoost (`xgboost`).

---

## 2. Implementation Status

### Implemented (production-ready)

| Component | Module |
|---|---|
| Raw data ingestion (chunked, leading-zero safe) | `data/ingest.py` |
| Schema normalisation and dtypes | `data/schema.py` |
| Pattern parsing (LI-Small only — disabled for Large) | `data/pattern_parser.py` |
| Transaction labelling (CSV ground truth) | `data/labeler.py` |
| Labeled-dataset export | `data/make_dataset.py` |
| Chronological train/val/test split + manifest | `data/splitter.py` |
| Leakage-safe feature pipeline (30 features incl. account aggregates) | `features/pipeline.py`, `features/aggregator.py`, `features/feature_cache.py` |
| Imbalance-mitigation strategy module | `sampling/strategies.py`, `sampling/prevalence.py` |
| Model factory (RF + XGBoost) with class-weight integration | `models/factory.py` |
| Evaluation metrics (PR-AUC primary, F1, F2, weighted accuracy, etc.) | `evaluation/metrics.py` |
| Single-experiment runner (with feature cache + full artefact persistence) | `experiments/runner.py` |
| Grid runner (resume + auto-backup to Drive) | `experiments/grid_runner.py` |
| Post-hoc F1-optimal threshold re-evaluation | `experiments/re_evaluate.py` |
| Result aggregator → leaderboard CSV | `experiments/aggregate.py` |
| Part B multi-strategy threshold optimisation (3 strategies, no retraining) | `experiments/threshold_optimizer.py` |
| Thesis-table generator (CSV + MD output) | `analysis/results_tables.py` |
| Audit notebook (data integrity check) | `notebooks/01_data_check.ipynb` |
| Production run notebooks | `aml_large_run.ipynb`, `aml_part_b_multi_threshold_run.ipynb` |

### Not yet implemented (deliberately deferred)

| Item | Reason |
|---|---|
| Hyperparameter tuning | Out of scope — the thesis question concerns strategies, not hyperparameters; defaults documented in §10 |
| Cross-validation (temporal folds) | Single fixed split; CV would change the variance baseline and complicate the strategy comparison |
| Graph-topology features (degree centrality, betweenness) | Listed in §8.4 as a planned extension; not required for the current research question |
| Pattern metadata stratification on LI-Large | Pattern matching disabled in Large for performance; would require a separate Large-scale pattern parser |

---

## 3. Repository Structure and Module Mapping

### Folder layout

```
classimbalance/
├── configs/                           # YAML configuration files (multiple environments)
│   ├── paths.yaml                     # Default (LI-Small smoke-test)
│   ├── paths_large_v2.yaml            # Production: LI-Large Part A (active)
│   ├── paths_large_part_b_v2.yaml     # Production: LI-Large Part B Strategy 6
│   ├── paths_large_part_b_v3.yaml     # Production: LI-Large Part B v3 (newer outputs)
│   ├── paths_part_b.yaml              # Legacy Part B paths (LI-Large, old layout)
│   ├── split.yaml                     # Train/val/test ratios (70/15/15)
│   ├── experiment.yaml                # Global random_seed = 42
│   ├── benchmark.yaml                 # Part A grid: 5×2×3 = 30 runs
│   ├── benchmark_part_b.yaml          # Part B Strategy 6 grid (true_cost_weighting)
│   └── benchmark_part_b_multi.yaml    # Part B Multi-Threshold (precision/F1/F2)
│
├── data/
│   ├── raw/                           # Original IBM AML files (immutable)
│   │   ├── LI-Large_Trans.csv         # ~ 176 M transactions
│   │   ├── LI-Large_accounts.csv      # Account → entity_type mapping
│   │   └── LI-Large_Patterns.txt      # (parsed only for Small; skipped for Large)
│   ├── processed_v2/
│   │   └── transactions_labeled.parquet
│   └── splits_v2/
│       ├── train.parquet              # 70 % (~ 123 M rows)
│       ├── val.parquet                # 15 % (~ 26 M rows)
│       ├── test.parquet               # 15 % (~ 26 M rows)
│       ├── split_manifest.json        # row counts, date ranges, class ratios
│       ├── train_features_v2.parquet  # cached feature matrix (X_train)
│       ├── val_features_v2.parquet
│       ├── test_features_v2.parquet
│       └── feature_pipeline_v2.pkl    # serialised fitted FeaturePipeline
│
├── outputs/                           # Run artefacts (lives on Drive in Colab)
│   ├── runs_v2/                       # Part A: one folder per run
│   │   └── <model>__<strategy>__pXXX__<timestamp>/
│   │       ├── run_config.json
│   │       ├── feature_pipeline.pkl
│   │       ├── model.pkl
│   │       ├── metrics_val.json/.csv          # @ threshold = 0.5
│   │       ├── metrics_test.json/.csv         # @ threshold = 0.5
│   │       ├── metrics_val_thresh.json/.csv   # @ F1-optimal threshold
│   │       ├── metrics_test_thresh.json/.csv  # @ F1-optimal threshold
│   │       └── threshold_info.json            # F1-optimal threshold metadata
│   ├── runs_part_b_v3/                # Part B Strategy 6 runs
│   ├── leaderboard_v2/
│   │   └── part_a_summary_v2.csv      # 30-row leaderboard
│   └── part_b_thresholds/             # Part B Multi-Threshold per-strategy outputs
│       └── <run_id>/<strategy>/{metrics_*.json,csv, threshold_info.json}
│
├── results/                           # Inputs for thesis table generator
│   ├── part_a_summary_v2.csv
│   ├── part_b_multi_threshold_summary.json
│   ├── feature_importance_xgboost_mean.csv
│   ├── feature_importance_rf_mean.csv
│   └── tables/                        # Auto-generated: table1a, table1b, table5, ...
│
├── notebooks/
│   └── 01_data_check.ipynb
│
├── aml_large_run.ipynb                # Part A end-to-end runbook (Colab)
├── aml_part_b_multi_threshold_run.ipynb  # Part B Multi-Threshold runbook
│
├── src/aml_benchmark/                 # Main Python package (see mapping below)
├── pyproject.toml                     # Package definition
└── README_PROJECT_STRUCTURE.md        # ← this file
```

### Doc-section ↔ module mapping

| Section | Module(s) |
|---|---|
| §5 Dataset | `data/ingest.py`, `data/schema.py`, `configs/paths_large_v2.yaml` |
| §6 Labeling | `data/labeler.py`, `data/make_dataset.py`, `utils/hashing.py` |
| §7 Split | `data/splitter.py`, `configs/split.yaml` |
| §8 Features | `features/pipeline.py`, `features/aggregator.py`, `features/feature_cache.py` |
| §9 Strategies | `sampling/strategies.py`, `sampling/prevalence.py` |
| §10 Models | `models/factory.py` |
| §11 Evaluation | `evaluation/metrics.py`, `experiments/re_evaluate.py` |
| §12 Aggregation | `experiments/aggregate.py` |
| §13 Part B | `experiments/threshold_optimizer.py`, `configs/benchmark_part_b*.yaml`, `analysis/results_tables.py` |

### Configuration architecture

`PathConfig` (`config.py`) accepts an optional explicit YAML path so the same code can be invoked against different environments without touching code:

```python
from aml_benchmark.config import PathConfig
paths = PathConfig("configs/paths_large_v2.yaml")  # Part A
paths = PathConfig("configs/paths_large_part_b_v3.yaml")  # Part B
```

Relative paths in the YAML resolve against the repository root; absolute paths (e.g. Drive mount points) are used as-is. This is what enables seamless Colab ↔ local execution.

---

## 4. End-to-End Pipeline Walk-Through

The complete benchmark consists of **four stages**, each callable independently:

```
[Raw CSV/TXT] -> Stage 1: Labeling     -> transactions_labeled.parquet
              -> Stage 2: Splitting    -> {train,val,test}.parquet + manifest
              -> Stage 3: Experiment   -> outputs/runs_v2/<run_id>/...
                  (includes Stage 3a: feature build + cache,
                              3b: imbalance strategy,
                              3c: train,
                              3d: evaluate @ threshold 0.5)
              -> Stage 4a: Re-evaluate -> metrics_*_thresh + threshold_info
              -> Stage 4b: Aggregate   -> part_a_summary_v2.csv (leaderboard)
```

### Stage 1 — Labelling

```
python -m aml_benchmark.data.make_dataset --paths configs/paths_large_v2.yaml
```

`ingest.py` chunked-reads the transactions CSV (`dtype=str` to preserve leading IDs), parses timestamps, drops unparseable rows, and casts numeric columns. `labeler.py` assigns the canonical binary `label` column directly from `is_laundering_csv` (the IBM-generator ground truth — see §6) and sorts the dataset chronologically. Output: `data/processed_v2/transactions_labeled.parquet`.

### Stage 2 — Splitting

```
python -m aml_benchmark.data.splitter --paths configs/paths_large_v2.yaml
```

`splitter.py` sorts by `timestamp`, splits at row-index quantiles (70/15/15), persists three parquet files plus `split_manifest.json` recording row counts, positives, achieved prevalence, and date ranges per split.

### Stage 3 — Single experiment

```
python -m aml_benchmark.experiments.runner --paths configs/paths_large_v2.yaml
```

`runner.py:run_experiment(model_name, strategy, target_prevalence, ...)` performs **one** complete condition. The grid runner (`grid_runner.py`) iterates this over the full 30-condition grid defined in `benchmark.yaml`. On the first invocation it builds and caches the feature matrices; all subsequent runs load from the cache, which removes the dominant computational cost (account-level rolling aggregates).

Per-run outputs (in `outputs/runs_v2/<run_id>/`):

| File | Content |
|---|---|
| `run_config.json` | All inputs + achieved prevalences + class weights + timings |
| `feature_pipeline.pkl` | The fitted `FeaturePipeline` (for downstream re-use) |
| `model.pkl` | The fitted estimator (joblib) |
| `metrics_{val,test}.json/.csv` | Metrics at threshold = 0.5 |

### Stage 4a — F1-optimal threshold re-evaluation

```
python -m aml_benchmark.experiments.re_evaluate --paths configs/paths_large_v2.yaml
```

`re_evaluate.py` walks every run folder, loads the model + cached features, recomputes scores, finds the F1-optimal threshold **on the validation set only**, and applies it once to the test set. Outputs: `metrics_{val,test}_thresh.json/.csv` and `threshold_info.json`. See §11.2.

### Stage 4b — Aggregation

```
python -m aml_benchmark.experiments.aggregate --paths configs/paths_large_v2.yaml
```

`aggregate.py` collects every run's `run_config.json`, `metrics_*.json`, `metrics_*_thresh.json`, and `threshold_info.json` into a single tabular leaderboard at `outputs/leaderboard_v2/part_a_summary_v2.csv` (one row per run). Sorted by `pr_auc_test` desc, then `recall_test_thresh` desc.

---

## 5. Dataset (LI-Large)

**Source.** IBM AMLworld Synthetic Dataset — Variant **Low-Illicit Large**.
*Altman et al., 2023, NeurIPS — [arXiv:2306.16424](https://arxiv.org/abs/2306.16424).*

**Raw files** (resolved by `configs/paths_large_v2.yaml`):

| File | Path | Notes |
|---|---|---|
| Transactions | `data/raw/LI-Large_Trans.csv` | The 11-column transaction log |
| Accounts | `data/raw/LI-Large_accounts.csv` | Account → entity_type mapping |
| Patterns | `data/raw/LI-Large_Patterns.txt` | **Not parsed** in the Large pipeline (see below) |

**Raw-column schema** (from `data/schema.py:RAW_TRANS_COLUMNS`): `timestamp`, `from_bank`, `from_account`, `to_bank`, `to_account`, `amount_received`, `receiving_currency`, `amount_paid`, `payment_currency`, `payment_format`, `is_laundering_csv`. The original CSV header has a duplicated `Account` column for sender and receiver; `ingest.py` overrides the header with these unambiguous names.

**Total transactions** (post drop of unparseable timestamps): rows are partitioned 70/15/15 into the splits below; the manifest recorded by `splitter.py` is the authoritative source. **TBD: read exact total + global natural prevalence from `data/splits_v2/split_manifest.json` (lives on Drive).**

**Pattern matching disabled in Large.** `labeler.py` keeps the API for `patterns` but skips the join (`label_from_patterns` = 0, `pattern_type` = `"NONE"` for all rows) — performance reasons on > 100 M rows. The label column `label` is taken **verbatim** from the IBM-generator field `is_laundering_csv`, which is the authoritative ground truth covering both seed and layering-step transactions. See §6.

---

## 6. Labeling Logic

| Output column | Meaning | Source |
|---|---|---|
| `label_existing_csv` | Original `Is Laundering` from the transactions CSV (IBM ground truth) | CSV column 10 |
| `label` | **Canonical binary target** for all training and evaluation | = `label_existing_csv` |
| `label_from_patterns` | 1 if the transaction matched the patterns file (Small only) | Pattern join — Large: always 0 |
| `mismatch_flag` | CSV-illicit but absent from patterns | Large: always 0 |
| `pattern_type` | Laundering scheme (e.g. `FAN-IN`); `"NONE"` if unmatched | Pattern join — Large: `"NONE"` |
| `pattern_block_id` | 1-based pattern block; `-1` if unmatched | Pattern join — Large: `-1` |

**Why use `label_existing_csv` rather than `label_from_patterns` as the target?** The patterns file is an incomplete subset (seed transactions only). The CSV column was assigned by the IBM generator and covers **all** illicit transactions including layering steps. Using only the patterns label would silently misclassify thousands of genuine positives as legitimate.

---

## 7. Temporal Split

**Method.** Strictly **temporal** and **deterministic** — no random seed for the split itself.

```python
df_sorted = df.sort_values("timestamp").reset_index(drop=True)
train_end = int(n * train_ratio)
val_end   = int(n * (train_ratio + val_ratio))
train, val, test = df_sorted.iloc[:train_end], df_sorted.iloc[train_end:val_end], df_sorted.iloc[val_end:]
```

**Ratios** (`configs/split.yaml`): 70 % train, 15 % val, 15 % test.

**Recorded sizes for LI-Large** (Part A baseline; from `outputs/leaderboard_v2/part_a_summary_v2.csv`):

| Split | Rows | Positives | Achieved prevalence | Date range |
|---|---:|---:|---:|---|
| Train (baseline, pre-sampling) | 123,246,589 | 63,811 | 0.051775 % | TBD (manifest) |
| Validation | 26,409,984 | 17,107 | 0.064775 % | TBD (manifest) |
| Test | 26,409,984 | 19,686 | 0.074549 % | TBD (manifest) |

Date ranges are written by `_split_stats(...)` in `splitter.py` to `split_manifest.json` but the file currently lives on Drive only (see §17).

The mild upward drift in test prevalence (0.052 % → 0.075 %) reflects natural concentration of generator-burst patterns in the later time window. This is preserved as part of the realistic evaluation condition.

**Look-ahead-bias defences.**
1. **Split** is purely chronological — no random permutation.
2. **Encoders** in `FeaturePipeline.fit_transform` are fit on training rows only; `transform` re-applies them to val/test without refitting.
3. **Account-level rolling aggregates** in `aggregator._rolling_agg` use only past-of-event observations within each rolling window (1 d / 7 d / 30 d) — no future leakage into past features.

---

## 8. Feature Engineering

The final feature matrix has **30 columns** (assembled in this order in `features/pipeline.py:_assemble`):

### 8.1 Numeric (2)

| Feature | Source | Transform |
|---|---|---|
| `amount_paid` | `amount_paid` (raw) | `np.log1p` |
| `amount_received` | `amount_received` (raw) | `np.log1p` |

### 8.2 Categorical (2)

`OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=NaN)` — fit on training data only:

| Feature | Source |
|---|---|
| `payment_format` | `payment_format` |
| `payment_currency` | `payment_currency` |

### 8.3 Derived (8)

| Feature | Formula |
|---|---|
| `hour` | `timestamp.dt.hour` (0–23) |
| `day_of_week` | `timestamp.dt.dayofweek` (0 = Monday) |
| `same_bank_flag` | `from_bank == to_bank` |
| `self_transfer_flag` | `from_account == to_account` |
| `currency_mismatch` | `payment_currency != receiving_currency` |
| `amount_ratio` | `clip(raw_received / raw_paid, 0, 10)` |
| `fan_in_score` | `clip(receiver_tx_count_7d / (sender_tx_count_7d + ε), 0, 100)` |
| `fan_out_score` | `clip(sender_unique_counterparties_7d / (receiver_unique_counterparties_7d + ε), 0, 100)` |

### 8.4 Account-level (18)

Computed by `features/aggregator.py:compute_account_features` using integer-encoded account IDs and per-group numpy operations (~20–50× faster than naive Python set loops). Rolling windows: 1 d / 7 d / 30 d.

| Sender features | Receiver features |
|---|---|
| `sender_tx_count_1d/7d/30d` | `receiver_tx_count_1d/7d/30d` |
| `sender_avg_amount_7d/30d` | `receiver_avg_amount_7d/30d` |
| `sender_unique_counterparties_7d/30d` | `receiver_unique_counterparties_7d/30d` |
| `sender_cross_bank_ratio_30d` | `receiver_cross_bank_ratio_30d` |
| `sender_entity_type` | `receiver_entity_type` |

`entity_type` is loaded from `LI-Large_accounts.csv` (mapping: 0 = Corporation, 1 = Partnership, 2 = Unknown).

### 8.5 Cache mechanism

`features/feature_cache.py` persists the assembled matrices after the first computation:

- `data/splits_v2/{train,val,test}_features_v2.parquet`
- `data/splits_v2/feature_pipeline_v2.pkl`

Subsequent runs (the remaining 29 of the 30-condition grid, plus all Part B work) load from cache, avoiding the expensive aggregation step entirely.

### 8.6 Planned but not implemented

- Graph-topology features (in/out degree, betweenness centrality proxy)
- Round-amount indicator (suspicious-round flag)
- Multi-currency exposure metric per account

---

## 9. Imbalance Mitigation Strategies

All strategies are implemented in `sampling/strategies.py` and applied **only to the training split**. Validation and test data remain untouched at every step. Each strategy returns a `SamplingResult` carrying the resampled `(X, y)`, the optional `class_weight` dict, achieved prevalence, and synthetic-sample count.

The conversion *target prevalence p → imblearn `sampling_strategy` ratio* is `r = p / (1 − p)` (`prevalence_to_ratio` in `sampling/prevalence.py`).

### 9.1 `baseline`

Returns training data unchanged. `target_prevalence` is recorded but not enforced. Reference condition.

### 9.2 `random_undersampling`

`imblearn.under_sampling.RandomUnderSampler(sampling_strategy=r, random_state=42)`. Applied only when `target_prevalence > natural_prevalence`; otherwise falls back to `baseline`.

### 9.3 `smote`

`imblearn.over_sampling.SMOTE(sampling_strategy=r, k_neighbors=k, random_state=42)` with `k = min(5, n_pos − 1)`. Synthetic positives are generated by linear interpolation between minority k-NN. The majority class is unchanged.

### 9.4 `adasyn` — with mandatory subsampling for KNN density estimation

`imblearn.over_sampling.ADASYN(sampling_strategy=ratio, n_neighbors=min(5, n_pos − 1), random_state=42)`.

When the majority class exceeds **`MAX_MAJORITY_FOR_KNN = 500,000`** rows (always the case on LI-Large), the implementation in `strategies.py:_adasyn` does the following:

1. Select a reproducible random subsample of **500,000 majority rows** with `np.random.default_rng(random_state)` and concatenate with **all** minority rows.
2. Run `ADASYN.fit_resample` on this subsample.
3. Compute the number of synthetic samples required so that the *full* training set reaches `target_prevalence`, not just the subsample (`n_pos_needed = round(target_prevalence × n_majority / (1 − target_prevalence))`).
4. Extract only the new synthetic minority samples from the result and append them to the **complete original training data**.

Subsample reproducibility:
- The subsample seed is `random_seed = 42` from `configs/experiment.yaml`, so identical Drive runs produce identical subsamples.
- `n_neighbors` resolves to **5** in practice on LI-Large (`min(5, 63810)`).
- A `RuntimeError` from ADASYN (e.g. degenerate density estimate) triggers an automatic fallback to `smote` for that condition.

### 9.5 `class_weighting`

No resampling. Returns the unchanged training data plus a class-weight dict derived from the target prevalence:

```
w0 = 1.0
w1 = (1 − target_prevalence) / target_prevalence
class_weight = {0: w0, 1: w1}
```

Per model family (`models/factory.py`):
- **Random Forest** — passed directly as the `class_weight=` constructor argument.
- **XGBoost** — converted to `scale_pos_weight = w1 / w0`, which multiplies the gradient contribution of every positive sample.

### 9.6 Additional strategies implemented (used in Part B)

| Strategy | Purpose |
|---|---|
| `smote_class_weighting` | SMOTE oversampling + cost-sensitive class weights derived from the *original* class ratio (not from `target_prevalence`). Combined effect of resampling and weighting. |
| `true_cost_weighting` | No resampling; class weights derived from the *actual* training imbalance (`w1 = n_neg / n_pos`, ≈ 1,930 on LI-Large). Defines the Part B "Strategy 6" baseline. |

### 9.7 Class weighting × `max_samples` interaction (Random Forest)

Defense-relevant for §10.2. Given how `sklearn._forest._parallel_build_trees` handles `class_weight=dict` (verified against sklearn ≥ 1.3 source plus issue #24037):

1. `compute_sample_weight(class_weight, y)` is called **once on the full y** before any tree training, producing per-sample weights (length = `n_train_samples`).
2. Per tree: indices are bootstrapped to size `n_samples_bootstrap = max_samples`; `sample_counts = bincount(indices)`; the final `sample_weight = expanded_class_weight × sample_counts` is passed to the tree fit.

→ **Per-sample class weights survive bootstrap.** They are *not* mathematically diluted by `max_samples`. Each drawn positive retains its `w1` factor in the impurity computation.

→ **However**, the *effective* signal is structurally weak under extreme imbalance combined with small bootstrap caps: the expected number of positives per tree is `200,000 × 63,811 / 123,246,589 ≈ 103`, with many duplicates. Each tree therefore sees only ~100 unique positive observations, which limits ensemble diversity and explains why varying `target_prevalence` does not move RF-Class-Weighting results much in our Part A tables.

This is a *result*, not a bug. It motivates Part B's discussion of why cost-sensitive trees under extreme imbalance benefit from a stratified-bootstrap variant (`class_weight="balanced_subsample"`), which is not used here.

---

## 10. Models and Hyperparameters

Both models are instantiated in `models/factory.py:get_model(name, random_state, class_weight)` and integrate cleanly with the strategy module:

- **Random Forest** receives the `class_weight` dict directly.
- **XGBoost** computes `scale_pos_weight = w1 / w0` and passes it as a constructor argument.

### 10.1 XGBoost (`xgboost.XGBClassifier`)

| Parameter | Value | Source / Rationale |
|---|---|---|
| `objective` | `binary:logistic` | xgboost default |
| `eval_metric` | `"aucpr"` | Aligns internal eval with the thesis primary metric |
| `learning_rate` | `0.05` | Conservative; better generalisation than the default 0.3 |
| `max_depth` | `6` | Standard default; avoids extreme depth on imbalanced data |
| `n_estimators` | `200` | More trees compensate for the lower learning rate |
| `subsample` | `0.8` | Row subsampling per tree; regularisation |
| `colsample_bytree` | `0.8` | Feature subsampling per tree; regularisation |
| `tree_method` | `"hist"` | Histogram-based; significantly faster on large datasets |
| `scale_pos_weight` | `w1 / w0` (or unset) | Derived from `class_weight` dict by `factory.py` |
| `device` | `"cuda"` if `nvidia-smi` available, else `"cpu"` | Auto-detect via `_detect_xgb_device()`; silent CPU fallback |
| `n_jobs` | `-1` | All CPU cores |
| `random_state` | `42` | `configs/experiment.yaml` |
| Early stopping | not set | — |

### 10.2 Random Forest (`sklearn.ensemble.RandomForestClassifier`)

| Parameter | Value | Source / Rationale |
|---|---|---|
| `n_estimators` | `100` | Stable estimate baseline |
| `max_features` | `"sqrt"` | Standard for classification |
| `min_samples_leaf` | `5` | Prevents overfitting to isolated noise; matters under extreme imbalance |
| `max_samples` | `200_000` | **RAM-bound** — see below |
| `bootstrap` | `True` | sklearn default |
| `class_weight` | `{0: w0, 1: w1}` (or `None`) | Strategy-supplied |
| `n_jobs` | `4` | Conservative parallelism (memory-bounded, see below) |
| `random_state` | `42` | `configs/experiment.yaml` |
| `verbose` | `2` | Per-tree progress logging |

#### Why `max_samples = 200_000` ?

The compute environment used for Part A had **179 GB of RAM**. Despite this, attempting `max_samples=None` (sklearn default — bootstrap of full training-set size) reproducibly hit OOM. Memory budget:

| Item | Calculation | Size |
|---|---|---|
| Train matrix `X_train` (float64) | 123,246,589 × 30 × 8 B | ~30 GB |
| Per-tree bootstrap sample (default = `n_samples`) | identical | ~30 GB |
| Parallel trees (`n_jobs=4`) | 4 × 30 GB just for bootstraps | ~120 GB |
| Plus feature cache, tree structures, NumPy/Python overhead | | ~30–50 GB |
| **Total** | | **> 179 GB → OOM** |

`max_samples = 200_000` (≈ 0.0016 × `n_train`) was the only tested value that produced stable training. Intermediate values were not systematically swept. **No methodological motivation exists** beyond the RAM constraint — the choice is purely pragmatic.

**Quantitative consequence (defense-relevant).** Under bootstrap with replacement at `max_samples = 200_000`, expected positives per tree = `200_000 × (63,811 / 123,246,589) ≈ 103`. This is a **structural ceiling** for what any imbalance strategy can do on RF in this setup, and explains the limited variation observed for RF + class weighting across `target_prevalence` levels (cf. §9.7).

#### Hyperparameter rationale beyond inline comments

No commit messages or pilot-tuning notebooks document additional rationale beyond the inline comments above. **TBD: confirm with the author whether pilot tuning was performed on a sub-sample, or whether all values are deliberate informed defaults.**

---

## 11. Evaluation Metrics and Threshold Policy

### 11.1 Metrics (`evaluation/metrics.py:compute_all_metrics`)

In order of importance:

| Metric | Notes |
|---|---|
| `pr_auc` | **Primary**. `sklearn.metrics.average_precision_score`. Threshold-independent. |
| `roc_auc` | Threshold-independent. Reported but **not** primary under extreme imbalance. |
| `precision` / `recall` / `f1` / `f2` | Threshold-dependent. F2 (β = 2) added for recall-weighted reference. |
| `weighted_accuracy` | Class-imbalance-corrected. Each positive carries `sample_weight = n_neg / n_pos`. |
| `tp`, `fp`, `tn`, `fn`, `threshold` | Confusion-matrix counts and the threshold that produced them. |

Metrics are persisted as both JSON (per-split, human-readable) and single-row CSV (per-split, easy to aggregate).

### 11.2 Threshold policy

Two threshold conditions are systematically reported per run:

| Condition | Threshold | Where |
|---|---|---|
| **Default** | 0.5 | `metrics_{val,test}.json/.csv` (written by `runner.py`) |
| **F1-optimal** | argmax F1 over all unique val scores | `metrics_{val,test}_thresh.json/.csv` + `threshold_info.json` (written by `re_evaluate.py`) |

**No-leakage guarantees** in `re_evaluate.py`:
- The threshold is selected *only* on the validation set, via `find_optimal_threshold(y_val, y_score_val, criterion="f1")`.
- The model is **not** retrained.
- The selected threshold is applied **once** to the test set.

This separation lets the leaderboard (§12) compare strategies both under the naive threshold and at an operationally meaningful operating point.

---

## 12. Result Aggregation and Leaderboard

`experiments/aggregate.py:aggregate_part_a` walks `outputs/runs_v2/`, picks up every run that has both a `run_config.json` (with `strategy` + `target_prevalence`) and `metrics_{val,test}.json`, and assembles `outputs/leaderboard_v2/part_a_summary_v2.csv`.

Per-run columns include:
- **Configuration**: `model`, `strategy`, `target_prevalence`, `achieved_train_prevalence`, `train_rows_after_sampling`, `train_positives_after_sampling`, `n_synthetic_samples`, `val_rows`, `val_positives`, `test_rows`, `test_positives`, `train_time_sec`, `created_at`
- **Default-threshold metrics** (val + test): `pr_auc`, `roc_auc`, `precision`, `recall`, `f1`, `weighted_accuracy`, `tp`, `fp`, `tn`, `fn`
- **Optimal-threshold metrics** (val_thresh + test_thresh): `precision`, `recall`, `f1`, `weighted_accuracy`, `tp`, `fp`, `tn`, `fn`
- **Threshold metadata**: `optimal_threshold`, `threshold_criterion`

Sort: `pr_auc_test` desc, then `recall_test_thresh` desc.

The thesis-table generator (`analysis/results_tables.py`) consumes this CSV plus per-strategy JSONs from Part B and produces:

| Output | Content |
|---|---|
| `results/tables/table1a_main_results.{csv,md}` | Primary Part A leaderboard |
| `results/tables/table1b_appendix_results.{csv,md}` | Extended Part A metrics |
| `results/tables/table3_feature_importance_xgboost.{csv,md}` | XGBoost feature importance |
| `results/tables/table4_feature_importance_rf.{csv,md}` | Random Forest feature importance |
| `results/tables/table5_part_b_multi_threshold.{csv,md}` | Part B Multi-Threshold (4 rows: Part A reference + 3 strategies) |

---

## 13. Part B — Custom Strategy and Multi-Threshold Analysis

Part B has two independent components.

### 13.1 Custom strategy (`true_cost_weighting`)

**Idea.** Whereas `class_weighting` derives weights from a chosen `target_prevalence`, `true_cost_weighting` uses the **actual** observed training imbalance: `w1 = n_neg / n_pos ≈ 1930` for LI-Large. No resampling.

**Grid** (`configs/benchmark_part_b.yaml`): 1 strategy × 2 models × 3 prevalences = 6 runs (target_prevalence is recorded as metadata only; the weights themselves are computed from the data).

**Run command.**

```
python -m aml_benchmark.experiments.grid_runner \
    --paths configs/paths_large_part_b_v3.yaml \
    --benchmark configs/benchmark_part_b.yaml
```

Or programmatically: `grid_runner.run_part_b_grid(paths)`.

### 13.2 Multi-strategy threshold optimisation (no retraining)

**Premise.** The Part A XGBoost Baseline produces a fixed score function. PR-AUC is then a property of the *entire* PR curve and is invariant under threshold selection. Three operating-point strategies are evaluated against the **same** scores:

| Strategy | Objective on validation |
|---|---|
| `precision_constrained` | argmax F1 subject to `precision ≥ 0.10`; tiebreak by utility `U = TP − 0.05 · FP`. **Operationally recommended** for AML compliance (capped alert volume). |
| `f1_max` | argmax F1. Methodologically standard reference. |
| `f2_max` | argmax F2 (β = 2). Recall-weighted; aligns with FATF/FinCEN expectations. |

**Anti-leakage guarantees** (`experiments/threshold_optimizer.py`):
1. The threshold grid is built from **validation-score quantiles only** (max 1,000 points). Test scores are never inspected for grid construction.
2. The selector functions accept only `y_val` and `y_score_val`; `y_test` is never in scope at selection time.
3. **PR-AUC invariance** is asserted at runtime (`abs(test_pr_auc − part_a_pr_auc) > 1e-9` raises `RuntimeError`). A failure means scores differ across strategies — i.e. an accidental retrain or score mix-up.
4. The Part A reference operating point is loaded from `threshold_info.json` (i.e. **F1-optimal**, not 0.5). Using 0.5 as the comparison baseline would inflate apparent improvements.

**Selected representative run.** `xgboost__baseline__p001__20260404_143052`. The `baseline` strategy is mathematically invariant to `target_prevalence` (no resampling, no class weighting), so all three baseline runs (p001/p005/p010) produce bit-identical models — one is sufficient.

**Run command.**

```
python -m aml_benchmark.experiments.threshold_optimizer \
    --paths configs/paths_large_v2.yaml
```

**Outputs.**

| Path | Content |
|---|---|
| `outputs/part_b_thresholds/<run_id>/<strategy>/metrics_{val,test}.{json,csv}` | Per-strategy metrics |
| `outputs/part_b_thresholds/<run_id>/<strategy>/threshold_info.json` | Per-strategy chosen threshold + Part A reference + deltas |
| `results/part_b_multi_threshold_summary.json` | Consolidated record (all strategies + invariance check) |

A self-test (`--dry-run`) executes all three strategies on mock scores and asserts the invariance — useful as a CI-style smoke check.

---

## 14. Reproducibility

| Mechanism | Implementation |
|---|---|
| Single random seed | `configs/experiment.yaml: random_seed: 42` |
| Seed propagation | All models, samplers, and the ADASYN subsample RNG receive `random_state=42` |
| Deterministic split | Pure chronological partition, no random state involved |
| Deterministic path resolution | `PathConfig` resolves all paths from the project root via `__file__`; absolute paths in the YAML pass through as-is (Drive support) |
| Immutable raw data | `data/raw/` is never written to |
| Frozen splits + manifest | `data/splits_v2/{train,val,test}.parquet` + `split_manifest.json` |
| Frozen feature cache | `*_features_v2.parquet` + `feature_pipeline_v2.pkl` |
| Per-run `run_config.json` | Records every parameter, achieved prevalence, class weights, timings |
| Serialised artefacts | `feature_pipeline.pkl`, `model.pkl` per run — re-evaluation never retrains |
| PR-AUC invariance assertion (Part B) | Runtime check: `abs(test_pr_auc − part_a_pr_auc) > 1e-9` raises |
| Resume support | `grid_runner._find_completed_run` skips runs that already have `metrics_test.json` + `run_config.json` |
| Auto-backup to Drive | `grid_runner._auto_backup` copies splits + per-run outputs after each successful run |

### Library versions

`pyproject.toml` declares lower bounds:

```
"pandas>=2.0", "pyarrow>=12.0", "numpy>=1.24", "PyYAML>=6.0",
"scikit-learn>=1.3", "imbalanced-learn>=0.11", "xgboost>=2.0",
"matplotlib>=3.7", "seaborn>=0.12", "jupyter>=1.0"
# requires-python = ">=3.10"
```

**Installed versions used in the LI-Large Colab production run — TBD.** Run the following in the production environment and paste the result into this section:

```bash
python --version
pip freeze | grep -E "^(scikit-learn|xgboost|imbalanced-learn|numpy|pandas|pyarrow|joblib|PyYAML)="
```

---

## 15. Hardware Context and Performance

| Item | Value |
|---|---|
| Production environment | Colab Pro+ high-memory instance, **179 GB RAM** |
| GPU | NVIDIA (auto-detected by `factory.py:_detect_xgb_device`); used by XGBoost |
| Random Forest parallelism | `n_jobs = 4` (memory-bounded — see §10.2) |
| XGBoost parallelism | `n_jobs = -1`, `device = cuda` |
| RF max_samples | `200_000` per tree — full bootstrap is OOM even at 179 GB (§10.2) |
| Approximate per-run training time | XGBoost (GPU) baseline / class_weighting: ~ 2–3 min; RUS @ 1 %: ~ 1–2 min; SMOTE / ADASYN @ 1 %: ~ 5–10 min; RF generally longer than XGBoost-GPU at the same `n_estimators`. **Exact numbers: TBD from Colab logs.** |

**Smaller-RAM environments are not supported** for Random Forest with the current setup — XGBoost is significantly more memory-friendly and runs in much smaller environments.

---

## 16. Known Limitations

1. **No hyperparameter tuning.** Defaults are documented in §10; the comparison isolates strategies, not hyperparameters.
2. **Feature set is MVP.** No graph-topology features; no embeddings; no external data.
3. **No cross-validation.** A single fixed temporal split is used for all 30 conditions.
4. **Pattern matching disabled in Large.** Per-pattern-type stratification (e.g. "which strategies catch FAN-IN best?") is therefore not available for LI-Large. The CSV ground truth still labels every illicit transaction correctly.
5. **RF + `max_samples=200_000` structurally limits any imbalance strategy.** ~103 expected positives per tree at LI-Large prevalence is a structural cap on RF responsiveness to `target_prevalence` (§9.7, §10.2). This is treated as a result, not a defect.
6. **Class weighting on RF does not vary much across `target_prevalence`.** Per the analysis in §9.7, this follows from (5), not from a `class_weight` bug in sklearn.

---

## 17. Open Questions / TBD

| # | Item | Where to find it |
|---|---|---|
| 1 | Total transaction count + global natural prevalence (pre-split) | `data/splits_v2/split_manifest.json` (Drive) or directly from `transactions_labeled.parquet` |
| 2 | Train/Val/Test exact `date_start` and `date_end` | `split_manifest.json` (Drive) — written by `splitter.py:_split_stats` |
| 3 | Exact installed library versions | Run the `pip freeze` snippet in §14 in the Colab environment |
| 4 | Hyperparameter rationale beyond inline comments | Author confirmation: were defaults pilot-tuned, or are they all informed defaults? |
| 5 | Per-condition runtimes (XGB vs RF, baseline vs SMOTE vs ADASYN) | Aggregate from `run_config.json:train_time_sec` across `outputs/runs_v2/` |

---

## 18. Appendix — CLI Cheat-Sheet

All commands assume the package is installed (`pip install -e .`) and the working directory is the repository root.

### Fresh end-to-end Part A on LI-Large

```bash
# 1. Build labeled dataset
python -m aml_benchmark.data.make_dataset --paths configs/paths_large_v2.yaml

# 2. Build splits
python -m aml_benchmark.data.splitter --paths configs/paths_large_v2.yaml

# 3. Run the full 30-condition Part A grid (resume-capable)
python -m aml_benchmark.experiments.grid_runner --paths configs/paths_large_v2.yaml

# 4. F1-optimal threshold post-hoc (writes metrics_*_thresh + threshold_info)
python -m aml_benchmark.experiments.re_evaluate --paths configs/paths_large_v2.yaml

# 5. Aggregate the leaderboard
python -m aml_benchmark.experiments.aggregate --paths configs/paths_large_v2.yaml
```

### Part B — Strategy 6 (`true_cost_weighting`)

```bash
python -m aml_benchmark.experiments.grid_runner \
    --paths configs/paths_large_part_b_v3.yaml \
    --benchmark configs/benchmark_part_b.yaml
```

### Part B — Multi-Threshold (no retraining)

```bash
# All three strategies on the Part A XGBoost Baseline
python -m aml_benchmark.experiments.threshold_optimizer \
    --paths configs/paths_large_v2.yaml

# Subset of strategies
python -m aml_benchmark.experiments.threshold_optimizer \
    --paths configs/paths_large_v2.yaml \
    --strategies precision_constrained f1_max

# Self-test with mock scores (CI-style smoke check)
python -m aml_benchmark.experiments.threshold_optimizer --dry-run
```

### Generate thesis tables

```bash
python -m aml_benchmark.analysis.results_tables
# Outputs to results/tables/{table1a, table1b, table3, table4, table5}.{csv,md}
```

### Single ad-hoc experiment (for debugging)

```bash
python -m aml_benchmark.experiments.runner --paths configs/paths_large_v2.yaml
# Defaults: random_forest, baseline, target_prevalence=0.01
```

---

*Document last updated: 2026-05-01 · Project: AML Benchmark · Thesis: Bachelor FS26*
