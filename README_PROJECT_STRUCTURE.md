# AML Benchmark — Project Structure and Technical Documentation

**Internal project documentation · Bachelor Thesis · FS26**

---

## Table of Contents

1. [Project Purpose](#1-project-purpose)
2. [Implementation Status](#2-implementation-status)
3. [Folder and File Structure](#3-folder-and-file-structure)
4. [End-to-End Data Flow](#4-end-to-end-data-flow)
5. [Labeling Logic](#5-labeling-logic)
6. [Pattern Metadata](#6-pattern-metadata)
7. [Temporal Split Logic](#7-temporal-split-logic)
8. [Feature Engineering](#8-feature-engineering)
9. [Models](#9-models)
10. [Evaluation Metrics](#10-evaluation-metrics)
11. [Baseline Smoke Test Results](#11-baseline-smoke-test-results)
12. [Reproducibility](#12-reproducibility)
13. [Next Development Steps](#13-next-development-steps)

---

## 1. Project Purpose

This project implements a reproducible, academically structured benchmark for comparing **class imbalance mitigation strategies in machine learning-based anti-money laundering (AML) transaction monitoring**.

It supports a Bachelor thesis with the following research objective:

> *How do different class imbalance mitigation strategies affect detection performance in AML transaction monitoring under extreme class imbalance?*

The benchmark is based on the **IBM AML synthetic dataset**, specifically the *Low-Illicit (LI-Small)* variant, which provides a realistic transaction graph with a very low proportion of labeled illicit transactions (approximately 0.05% of all transactions). The task is framed as **binary transaction-level classification**: given a financial transaction, predict whether it is illicit (label = 1) or legitimate (label = 0).

The project is not a production fraud detection system. It is a **controlled academic experiment** designed to isolate the effect of imbalance handling strategies on model performance under identical conditions.

The planned benchmark (Part A) evaluates five strategies:

| Strategy | Description |
|---|---|
| Baseline | No imbalance handling |
| Random Undersampling (RUS) | Reduce majority class in training data |
| SMOTE | Synthetic oversampling of minority class |
| ADASYN | Adaptive synthetic oversampling |
| Class Weighting | Cost-sensitive learning via model weights |

These are evaluated across two model families (Random Forest, XGBoost) and three training prevalence levels (~1.0%, ~0.5%, ~0.1%), yielding a **30-condition benchmark grid**. A Part B subsequently designs and evaluates a tailored sixth strategy based on weaknesses identified in Part A.

---

## 2. Implementation Status

### Implemented

| Component | Status | Entry Point |
|---|---|---|
| Raw data ingestion | Complete | `data/ingest.py` |
| Schema normalization | Complete | `data/schema.py` |
| Laundering pattern parsing | Complete | `data/pattern_parser.py` |
| Transaction labeling | Complete | `data/labeler.py` |
| Pattern metadata retention | Complete | `data/labeler.py` |
| Labeled dataset export | Complete | `data/make_dataset.py` |
| Chronological data split | Complete | `data/splitter.py` |
| Leakage-safe feature pipeline | Complete (MVP) | `features/pipeline.py` |
| Model factory (RF + XGBoost) | Complete | `models/factory.py` |
| Evaluation metrics | Complete | `evaluation/metrics.py` |
| Baseline experiment runner | Complete | `experiments/runner.py` |
| Audit notebook | Complete | `notebooks/01_data_check.ipynb` |

### Not Yet Implemented

| Component | Description |
|---|---|
| Imbalance strategies | RUS, SMOTE, ADASYN, class weighting — not yet applied |
| Prevalence control | Training set adjustment to ~1.0%, ~0.5%, ~0.1% positive rates |
| Full benchmark grid | 30-condition loop (5 strategies × 2 models × 3 prevalences) |
| Result aggregation | Comparative summary tables and plots across all conditions |
| Part B custom strategy | Tailored sixth strategy based on Part A findings |
| Account-level features | Rolling aggregates per sender/receiver account |
| Graph-based features | Network topology features per transaction node |

---

## 3. Folder and File Structure

```
classimbalance/
|
|-- configs/                        # YAML configuration files
|   |-- paths.yaml                  # All file and directory paths
|   |-- split.yaml                  # Train/val/test split ratios
|   `-- experiment.yaml             # Global experiment settings (e.g. random seed)
|
|-- data/
|   |-- raw/                        # Original, immutable IBM AML source files
|   |   |-- LI-Small_Trans.csv      # 6.9M transaction rows (raw)
|   |   |-- LI-Small_accounts.csv   # 712K account records
|   |   `-- LI-Small_Patterns.txt   # Laundering pattern definitions (117 blocks)
|   |
|   |-- processed/
|   |   `-- transactions_labeled.parquet   # Full labeled dataset (16 columns)
|   |
|   `-- splits/
|       |-- train.parquet           # Chronological training split (70%)
|       |-- val.parquet             # Validation split (15%)
|       |-- test.parquet            # Test split (15%)
|       `-- split_manifest.json     # Row counts, date ranges, class ratios
|
|-- notebooks/
|   `-- 01_data_check.ipynb         # Data audit and label validation notebook
|
|-- outputs/
|   `-- runs/
|       `-- <run_id>/               # One directory per experiment run
|           |-- run_config.json     # Model, seed, feature names, row counts
|           |-- metrics_val.json    # Validation metrics
|           |-- metrics_val.csv     # Validation metrics (CSV format)
|           |-- metrics_test.json   # Test metrics
|           |-- metrics_test.csv    # Test metrics (CSV format)
|           |-- feature_pipeline.pkl  # Serialized fitted FeaturePipeline
|           `-- model.pkl           # Serialized fitted model
|
|-- src/
|   `-- aml_benchmark/              # Main Python package
|       |-- __init__.py
|       |-- config.py               # PathConfig, load_yaml, project root resolution
|       |
|       |-- data/                   # Data ingestion, labeling, and splitting
|       |   |-- schema.py           # Column names, dtype definitions
|       |   |-- ingest.py           # load_transactions(), load_accounts()
|       |   |-- pattern_parser.py   # parse_patterns() for patterns TXT file
|       |   |-- labeler.py          # create_labels() — joins patterns to transactions
|       |   |-- make_dataset.py     # CLI: produces transactions_labeled.parquet
|       |   `-- splitter.py         # CLI: produces train/val/test splits
|       |
|       |-- features/
|       |   `-- pipeline.py         # FeaturePipeline class (fit/transform)
|       |
|       |-- models/
|       |   `-- factory.py          # get_model("random_forest" | "xgboost")
|       |
|       |-- evaluation/
|       |   `-- metrics.py          # compute_all_metrics(), save_metrics()
|       |
|       |-- experiments/
|       |   `-- runner.py           # CLI: end-to-end baseline experiment
|       |
|       `-- utils/
|           |-- hashing.py          # make_match_key() — deterministic join key
|           |-- io.py               # save_parquet(), load_parquet()
|           `-- logging_utils.py    # get_logger() — consistent log format
|
|-- pyproject.toml                  # Package definition; install with pip install -e .
|-- requirements.txt                # Dependency list
`-- .gitignore
```

### File roles in detail

#### `configs/paths.yaml`
Single source of truth for every file path used in the pipeline. All other modules resolve paths through `PathConfig`, which reads this file. Changing a directory location requires editing only this file.

#### `configs/split.yaml`
Specifies the train/val/test split ratios (currently 70%/15%/15%). The splitter reads these at runtime and applies them to the sorted timestamp sequence.

#### `configs/experiment.yaml`
Stores the global random seed (currently `42`) and any future experiment-level defaults.

#### `src/aml_benchmark/config.py`
Central configuration module. `PathConfig` resolves all paths relative to the project root (detected via `__file__`), so the pipeline can be invoked from any working directory. Also provides `load_yaml()` for reading any config file by name.

#### `src/aml_benchmark/data/schema.py`
Defines `RAW_TRANS_COLUMNS` — the unambiguous column names used to override the duplicate `Account` header in the raw transactions CSV — as well as dtype mappings applied after initial string-safe loading.

#### `src/aml_benchmark/data/ingest.py`
Provides `load_transactions()` and `load_accounts()`. Both load raw files with `dtype=str` to preserve leading zeros in bank and account IDs, then apply correct dtypes after the initial read.

#### `src/aml_benchmark/data/pattern_parser.py`
Parses the structured text file containing laundering block definitions. Each block is delimited by `BEGIN LAUNDERING ATTEMPT` / `END LAUNDERING ATTEMPT` markers. Data rows within blocks share the same 11-field CSV format as the transactions file. Each extracted transaction is tagged with the block's pattern type (e.g., `FAN-IN`) and a sequential block ID.

#### `src/aml_benchmark/data/labeler.py`
Joins the parsed patterns onto the transaction table using a deterministic match key. Produces the final labeled dataset. See [Section 5](#5-labeling-logic) for full details.

#### `src/aml_benchmark/utils/hashing.py`
Implements `make_match_key()`, which constructs a pipe-delimited string from eight transaction fields after careful normalisation (timestamp to minute precision, amounts rounded to two decimal places, strings lowercased with leading zeros preserved). This key is used for deterministic pattern-to-transaction matching.

#### `src/aml_benchmark/data/make_dataset.py`
CLI entry point that orchestrates the full ingestion and labeling pipeline: load raw files → parse patterns → generate labels → save labeled parquet.

```
python -m aml_benchmark.data.make_dataset
```

#### `src/aml_benchmark/data/splitter.py`
CLI entry point for the chronological split. Reads `configs/split.yaml`, sorts the labeled dataset by timestamp, divides by row-index quantiles, saves three parquet files, and writes a JSON manifest.

```
python -m aml_benchmark.data.splitter
```

#### `src/aml_benchmark/features/pipeline.py`
Stateful `FeaturePipeline` class. `fit_transform(train_df)` fits encoders and returns the training feature matrix. `transform(df)` applies the fitted encoders to validation or test data without refitting. See [Section 8](#8-feature-engineering) for details.

#### `src/aml_benchmark/models/factory.py`
`get_model(name, random_state)` factory function that instantiates a freshly configured, unfitted estimator. Currently supports `"random_forest"` and `"xgboost"`.

#### `src/aml_benchmark/evaluation/metrics.py`
`compute_all_metrics(y_true, y_score)` computes the full metric set and returns an ordered dictionary. `save_metrics(metrics, output_dir, split)` persists results as both JSON and CSV. See [Section 10](#10-evaluation-metrics).

#### `src/aml_benchmark/experiments/runner.py`
CLI entry point for a complete end-to-end experiment run. Loads splits, builds features, trains a model, evaluates on validation and test, and saves all artefacts to a timestamped run directory.

```
python -m aml_benchmark.experiments.runner
```

---

## 4. End-to-End Data Flow

The pipeline is divided into three independent stages, each callable via its own CLI command and each reading/writing deterministic intermediate files.

### Stage 1: Labeling

```
python -m aml_benchmark.data.make_dataset
```

```
LI-Small_Trans.csv          ─┐
LI-Small_accounts.csv        ├─> [ingest.py]        -> DataFrames
LI-Small_Patterns.txt       ─┘
                               [pattern_parser.py]  -> patterns DataFrame
                                                       (pattern_type, pattern_block_id)
                               [utils/hashing.py]   -> match keys for both tables
                               [labeler.py]         -> label columns + pattern metadata
                               [utils/io.py]        -> save Parquet

OUTPUT: data/processed/transactions_labeled.parquet
        (6,924,049 rows, 16 columns)
```

**Step-by-step:**

1. `ingest.py` loads `LI-Small_Trans.csv` with `dtype=str` to preserve leading zeros, renames duplicate `Account` columns to `from_account` / `to_account`, parses the timestamp, and casts numeric columns.
2. `ingest.py` loads `LI-Small_accounts.csv` as a reference table (not yet joined to transactions — reserved for account-level feature engineering).
3. `pattern_parser.py` reads `LI-Small_Patterns.txt` line by line, extracts transaction data rows from within `BEGIN`/`END` block markers, assigns `pattern_type` and `pattern_block_id` to each row, and returns a patterns DataFrame in the same 11-field schema as the transactions.
4. `hashing.py` builds a deterministic match key from eight normalised fields for both DataFrames.
5. `labeler.py` performs a set-based lookup of pattern keys against transaction keys, assigns label columns, attaches pattern metadata via dictionary mapping, and sorts the result chronologically.
6. The labeled DataFrame (16 columns) is saved as `data/processed/transactions_labeled.parquet`.

### Stage 2: Splitting

```
python -m aml_benchmark.data.splitter
```

```
transactions_labeled.parquet
        -> [splitter.py] sort by timestamp
                         split at row-index quantiles (70/15/15)
                         save three parquets
                         write split_manifest.json

OUTPUT: data/splits/train.parquet
        data/splits/val.parquet
        data/splits/test.parquet
        data/splits/split_manifest.json
```

### Stage 3: Experiment

```
python -m aml_benchmark.experiments.runner
```

```
train.parquet / val.parquet / test.parquet
        -> [features/pipeline.py]    fit_transform(train) / transform(val, test)
                                     -> X_train, X_val, X_test  (numpy arrays)
        -> [models/factory.py]       get_model("random_forest")
        -> model.fit(X_train, y_train)
        -> model.predict_proba(X_val)[:, 1]   -> y_score_val
        -> model.predict_proba(X_test)[:, 1]  -> y_score_test
        -> [evaluation/metrics.py]   compute_all_metrics(y_true, y_score)
                                     save_metrics(metrics, output_dir, split)

OUTPUT: outputs/runs/<run_id>/
            run_config.json
            feature_pipeline.pkl
            model.pkl
            metrics_val.json / metrics_val.csv
            metrics_test.json / metrics_test.csv
```

---

## 5. Labeling Logic

The IBM AML dataset provides two independent sources of labeling information that must be reconciled:

| Source | Column | Description |
|---|---|---|
| `LI-Small_Trans.csv` | `Is Laundering` | Binary label (0/1) assigned at data-generation time by the IBM simulator for every transaction |
| `LI-Small_Patterns.txt` | *(implicit)* | Explicit listing of laundering transactions used as seeds for the patterns |

The labeler produces four label columns:

### `label_from_patterns`
Set to `1` if the transaction's match key is found in the parsed patterns file, `0` otherwise. This reflects only the **seed/key transactions** explicitly listed in the patterns file — not the full chain of derived illicit transactions that the IBM simulator generated from those seeds.

In the current dataset: **1,023** transactions are matched from patterns (across 117 laundering blocks).

### `label_existing_csv`
The original `Is Laundering` value from the transactions CSV, preserved verbatim and renamed for clarity. This column represents the **authoritative ground truth** produced by the IBM AML data generator. It labels every illicit transaction in the dataset, including:
- seed transactions (also present in the patterns file)
- layering-step transactions (derived from patterns but absent from the patterns file)

In the current dataset: **3,565** transactions are labeled illicit in the CSV.

### `mismatch_flag`
Set to `1` where `label_existing_csv == 1` but `label_from_patterns == 0`, i.e., transactions that the CSV labels illicit but that do not appear in the patterns file. These 2,542 rows are **expected** — they represent the full illicit transaction chain beyond the seed rows. They are not labeling errors.

### `label`
The **canonical binary target** used for all model training and evaluation. Set equal to `label_existing_csv`.

The decision to use `label_existing_csv` rather than `label_from_patterns` as the final target is deliberate and methodologically important: the CSV label covers all illicit transactions in the dataset (including layering steps), whereas the patterns file is an incomplete subset. Using only the patterns-derived label would artificially reduce the positive class and misclassify 2,542 genuinely illicit transactions as legitimate.

A cross-check confirms the integrity of the patterns file: **0 pattern keys are unmatched** in the transactions CSV, confirming that all 1,023 pattern-file transactions are present and correctly labeled in the source data.

---

## 6. Pattern Metadata

Two additional columns are attached to every row in the labeled dataset:

### `pattern_type`
The laundering scheme category of the block that contains this transaction, as extracted from the `BEGIN LAUNDERING ATTEMPT - <TYPE>` header line of the patterns file. For unmatched transactions, this is set to `"NONE"`.

The eight pattern types present in `LI-Small_Patterns.txt` are:

| Pattern type | Matched transactions |
|---|---|
| SCATTER-GATHER | 182 |
| STACK | 180 |
| GATHER-SCATTER | 150 |
| FAN-OUT | 149 |
| BIPARTITE | 129 |
| CYCLE | 83 |
| RANDOM | 77 |
| FAN-IN | 73 |

### `pattern_block_id`
A 1-based integer that identifies which block in the patterns file the transaction belongs to. For unmatched transactions, this is set to `-1`. The 117 blocks correspond to distinct laundering episodes.

These two fields are retained in the processed and split datasets for use in later analysis, specifically to investigate **which laundering pattern types are hardest to detect** under different imbalance mitigation strategies — a secondary research question in the thesis.

---

## 7. Temporal Split Logic

### Why chronological splitting matters

A standard random train/test split on time-series financial data introduces **temporal leakage**: the model can implicitly learn patterns from future observations (e.g., account-level behaviour in the test period leaks into training aggregates). For a thesis benchmark, this would invalidate the evaluation by producing unrealistically high performance estimates.

The splitter enforces a strict chronological partition: the dataset is sorted by `timestamp` and divided at fixed row-index quantile boundaries. At no point is any test or validation row visible during training.

### Current split configuration

Ratios from `configs/split.yaml`: **70% train, 15% val, 15% test**.

| Split | Rows | Positives | Illicit ratio | Start date | End date |
|---|---|---|---|---|---|
| **train** | 4,846,834 | 2,231 | 0.0460% | 2022-09-01 00:00 | 2022-09-07 14:48 |
| **val** | 1,038,607 | 583 | 0.0561% | 2022-09-07 14:48 | 2022-09-09 03:13 |
| **test** | 1,038,608 | 751 | 0.0723% | 2022-09-09 03:13 | 2022-09-17 15:28 |

**Total dataset span:** 2022-09-01 to 2022-09-17 (17 days).

The slight increase in illicit ratio from train to test (0.046% → 0.072%) reflects a natural concentration of labeled laundering activity in the latter portion of the dataset, consistent with how the IBM simulator generates burst patterns. This temporal variation is realistic and should not be equalized — it is part of the evaluation condition.

The split files and their manifest are saved to `data/splits/` and remain fixed for all benchmark conditions to ensure comparability across strategies.

---

## 8. Feature Engineering

The current feature set is an **MVP (minimum viable product)** sufficient for the baseline smoke test and Part A benchmark. It is implemented in `src/aml_benchmark/features/pipeline.py` as the `FeaturePipeline` class.

### Feature table

| Feature | Type | Raw column(s) | Transformation |
|---|---|---|---|
| `amount_paid` | Numeric | `amount_paid` | `log1p(x)` — reduces right skew |
| `amount_received` | Numeric | `amount_received` | `log1p(x)` — reduces right skew |
| `payment_format` | Categorical | `payment_format` | OrdinalEncoder (fit on train) |
| `payment_currency` | Categorical | `payment_currency` | OrdinalEncoder (fit on train) |
| `hour` | Temporal | `timestamp` | `timestamp.dt.hour` |
| `day_of_week` | Temporal | `timestamp` | `timestamp.dt.dayofweek` (0=Mon) |
| `same_bank_flag` | Boolean | `from_bank`, `to_bank` | 1 if `from_bank == to_bank` |
| `self_transfer_flag` | Boolean | `from_account`, `to_account` | 1 if `from_account == to_account` |

### Leakage safety

- `FeaturePipeline.fit_transform(train_df)` fits the `OrdinalEncoder` on the training split only.
- `FeaturePipeline.transform(val_df)` and `transform(test_df)` apply the frozen training-set encodings. Unknown categories (currency or format values not seen during training) are mapped to `NaN`.
- All derived features (`hour`, `day_of_week`, `same_bank_flag`, `self_transfer_flag`) are computed from within-row information and require no fitting.
- The pipeline can be serialised with `joblib.dump` to guarantee identical transformations across replications.

### Planned extensions (not yet implemented)

- Account-level rolling aggregates (transaction count, total volume, in/out ratio over 1h/24h/7d windows) computed from past-only observations using a point-in-time approach.
- Graph-topology features (in/out degree, betweenness centrality proxy) derived from the transaction network.
- Currency change indicator (flag when paying currency differs from receiving currency).
- Round-amount indicator (flag for suspiciously round transaction amounts).

---

## 9. Models

Two model families are supported via the `get_model()` factory in `src/aml_benchmark/models/factory.py`. Both return unfitted, sklearn-compatible estimators configured for the AML benchmark context.

### Random Forest (`"random_forest"`)

Implemented with `sklearn.ensemble.RandomForestClassifier`.

| Hyperparameter | Value | Rationale |
|---|---|---|
| `n_estimators` | 100 | Sufficient for stable estimates in baseline runs |
| `max_features` | `"sqrt"` | Standard for classification; controls overfitting |
| `max_samples` | 200,000 | Caps rows per tree; makes training tractable on 4.8M-row splits without OOM |
| `min_samples_leaf` | 5 | Prevents overfitting to isolated noise; important with extreme imbalance |
| `n_jobs` | -1 | Uses all available CPU cores |
| `random_state` | 42 (configurable) | Reproducibility |

### XGBoost (`"xgboost"`)

Implemented with `xgboost.XGBClassifier`. Requires `pip install xgboost`.

| Hyperparameter | Value | Rationale |
|---|---|---|
| `n_estimators` | 200 | More trees compensate for lower learning rate |
| `max_depth` | 6 | Standard default; avoids extreme depth on imbalanced data |
| `learning_rate` | 0.05 | Conservative; better generalisation than default 0.3 |
| `subsample` | 0.8 | Row subsampling per tree; regularisation |
| `colsample_bytree` | 0.8 | Feature subsampling per tree; regularisation |
| `eval_metric` | `"aucpr"` | Aligns internal XGBoost evaluation with the thesis primary metric |
| `tree_method` | `"hist"` | Histogram-based algorithm; significantly faster on large datasets |
| `random_state` | 42 (configurable) | Reproducibility |

### Current scope

The smoke test uses Random Forest with the baseline strategy (no imbalance handling). The full Part A benchmark will iterate both models across all five strategies and three prevalence levels.

---

## 10. Evaluation Metrics

All metrics are computed in `src/aml_benchmark/evaluation/metrics.py` by `compute_all_metrics(y_true, y_score, threshold, split)`. Results are returned as an ordered dictionary and persisted as both JSON and CSV.

### PR-AUC (Primary metric)

The **area under the Precision-Recall curve** is the primary benchmark metric. Under extreme class imbalance, ROC-AUC can be misleadingly optimistic because it is dominated by the large number of true negatives. PR-AUC, by contrast, focuses on the model's ability to correctly identify the minority positive class and is directly sensitive to the ratio of positives to negatives. A random classifier achieves PR-AUC approximately equal to the positive class prevalence (≈0.0005 at 0.05% rate).

### Secondary metrics

| Metric | Description |
|---|---|
| `roc_auc` | Area under the ROC curve (threshold-independent) |
| `precision` | TP / (TP + FP) at the decision threshold |
| `recall` | TP / (TP + FN) at the decision threshold |
| `f1` | Harmonic mean of precision and recall |
| `weighted_accuracy` | Class-imbalance-corrected accuracy (see below) |
| `tp`, `fp`, `tn`, `fn` | Confusion matrix counts at the decision threshold |
| `threshold` | The decision threshold used for all binary metrics (default: 0.5) |

### Weighted accuracy

Standard accuracy is dominated by the majority class under severe imbalance — a model that never predicts positive achieves ~99.95% standard accuracy. To correct for this, weighted accuracy assigns each positive sample a weight equal to `n_negative / n_positive` and each negative sample weight `1.0`:

```
weight_positive = n_negative / n_positive
weight_negative = 1.0
sample_weight[i] = weight_positive  if y_true[i] == 1  else  weight_negative
weighted_accuracy = accuracy_score(y_true, y_pred, sample_weight=sample_weight)
```

This ensures both classes contribute equally to the final metric regardless of their absolute frequencies. The weights used are logged for every evaluation call for transparency and traceability.

### Output format

For each experiment run and each split (`val`, `test`), two files are saved:
- `metrics_<split>.json` — human-readable, suitable for version control
- `metrics_<split>.csv` — single-row tabular format, suitable for aggregation across runs

---

## 11. Baseline Smoke Test Results

The first end-to-end experiment was executed using Random Forest with the baseline strategy (no imbalance handling, default decision threshold 0.5). The run identifier is `random_forest__baseline__20260323_153031`.

### Run configuration

| Parameter | Value |
|---|---|
| Model | RandomForestClassifier |
| Strategy | Baseline (no resampling) |
| Random seed | 42 |
| Training rows | 4,846,834 |
| Training positives | 2,231 (0.046%) |
| Features | 8 (see Section 8) |
| Training time | 53.9 s |
| Total pipeline time | 73.2 s |

### Metrics

| Metric | Validation | Test |
|---|---|---|
| **PR-AUC (primary)** | **0.0054** | **0.0229** |
| ROC-AUC | 0.7855 | 0.8449 |
| Precision | 0.0000 | 0.0000 |
| Recall | 0.0000 | 0.0000 |
| F1 | 0.0000 | 0.0000 |
| Weighted accuracy | 0.5000 | 0.5000 |
| TP / FP / TN / FN | 0 / 0 / 1,038,024 / 583 | 0 / 0 / 1,037,857 / 751 |

### Interpretation

These results are **correct and scientifically expected**. They are not indicative of a software defect.

- **ROC-AUC of 0.78–0.84** demonstrates that the model assigns, on average, a higher predicted probability to genuinely illicit transactions than to legitimate ones — i.e., it has learned a meaningful ranking signal from the eight MVP features.

- **PR-AUC of 0.005–0.023** is low in absolute terms, but should be compared against the random-classifier baseline of approximately 0.0005 (equal to the positive rate). The baseline model achieves roughly 10–46× the random baseline on PR-AUC.

- **TP = 0** at the default threshold of 0.5 is the central finding motivating Part A. With only approximately 9 positive samples per tree bootstrap sample (200,000 rows × 0.046%), the Random Forest never accumulates sufficient evidence to push any prediction above 0.5. The class probabilities are overwhelmingly negative-dominated and the threshold is inappropriate for this prevalence level.

This result precisely demonstrates the core problem the thesis investigates: **without imbalance mitigation, a standard classifier predicts no illicit transactions** even when it possesses some latent discriminative power. Part A will measure how much each mitigation strategy recovers this latent signal.

---

## 12. Reproducibility

The following mechanisms ensure that all results can be reproduced deterministically:

| Mechanism | Implementation |
|---|---|
| Fixed random seed | `configs/experiment.yaml → random_seed: 42`; passed to all model constructors |
| Deterministic path resolution | `PathConfig` resolves all paths from the project root via `__file__`; no hardcoded absolute paths |
| Immutable raw data | Raw files in `data/raw/` are never modified; all derived data is written to separate directories |
| Frozen split files | `data/splits/` parquet files are generated once and reused for all benchmark conditions |
| Frozen split manifest | `split_manifest.json` records exact row counts, date boundaries, and class ratios |
| Config-driven structure | All tunable parameters live in `configs/*.yaml`; no magic numbers in code |
| Serialized artefacts | `feature_pipeline.pkl` and `model.pkl` saved per run; evaluation can be re-run without retraining |
| Structured outputs | Every run produces a `run_config.json` recording all parameters, counts, and timestamps |
| Parquet format | Column types and values are preserved exactly (no CSV float rounding) |

---

## 13. Next Development Steps

The following components must be implemented to complete the Part A benchmark grid.

### Priority 1 — Imbalance strategy module

Create `src/aml_benchmark/sampling/strategies.py` implementing:

- `random_undersampling(X_train, y_train, target_ratio)` — randomly remove majority-class samples
- `smote(X_train, y_train, target_ratio)` — synthetic minority oversampling (via `imbalanced-learn`)
- `adasyn(X_train, y_train, target_ratio)` — adaptive synthetic oversampling (via `imbalanced-learn`)
- `class_weight_dict(y_train)` — compute `{0: w0, 1: w1}` dict for model constructors

All strategies must be applied **after** the train/val/test split and **only to training data**.

### Priority 2 — Prevalence control

Create `src/aml_benchmark/sampling/prevalence.py` implementing:

- `adjust_prevalence(X_train, y_train, target_ratio)` — subsample the negative class to achieve an approximately specified positive rate (~1.0%, ~0.5%, ~0.1%)
- Verify and log achieved ratio vs. target ratio per run

### Priority 3 — Full benchmark grid runner

Extend `src/aml_benchmark/experiments/runner.py` (or create `grid_runner.py`) to iterate over all 30 conditions:

```
for strategy in [baseline, RUS, SMOTE, ADASYN, class_weight]:
    for model in [random_forest, xgboost]:
        for prevalence in [0.010, 0.005, 0.001]:
            run_experiment(strategy, model, prevalence)
```

Each condition writes results to its own subdirectory under `outputs/runs/`.

### Priority 4 — Result aggregation

Create `src/aml_benchmark/experiments/aggregate.py` to:

- Collect all `metrics_test.json` files from completed runs
- Build a summary DataFrame (one row per condition)
- Export `outputs/part_a_summary.csv` for thesis table generation
- Generate comparison plots (PR-AUC by strategy, by model, by prevalence level)

### Priority 5 — Part B custom strategy

Based on weaknesses identified in the Part A results (expected: underperformance at ~0.1% prevalence; poor recall under SMOTE for rare pattern types), design and implement a sixth strategy. Evaluate under the identical 30-condition protocol to ensure comparability.

### Priority 6 — Extended feature set

Implement account-level rolling aggregate features in `features/`:

- Transaction counts per sender/receiver over 1h, 24h, 7d rolling windows
- Volume sums and averages over same windows
- In/out flow ratio per account
- All computed using past-only (`.shift(1)`) logic to prevent leakage

---

*Document generated: 2026-03-23 | Project: AML Benchmark | Thesis: Bachelor FS26*
