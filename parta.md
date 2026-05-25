# Part A — Benchmark setup, pipeline, and files

This document describes **end-to-end** how the **Part A benchmark** is structured, which **configuration files** and **code modules** matter, and **where artefacts are written** — mirroring the structure of `partB.md`.

**Relevant modules (selection):**

- `src/aml_benchmark/config.py` — `PathConfig`, `load_yaml`
- `src/aml_benchmark/data/make_dataset.py` — Labelling / processed parquet
- `src/aml_benchmark/data/splitter.py` — temporal splits
- `src/aml_benchmark/experiments/grid_runner.py` — 30-run grid (Part A)
- `src/aml_benchmark/experiments/runner.py` — single experiment (`run_experiment`)
- `src/aml_benchmark/features/feature_cache.py` — feature caches under `splits_dir`
- `src/aml_benchmark/sampling/strategies.py` — imbalance strategies
- `src/aml_benchmark/experiments/re_evaluate.py` — F1-optimal threshold (val → test)
- `src/aml_benchmark/experiments/aggregate.py` — leaderboard CSV

---

## 1. Configuration files

| File | Role |
|------|------|
| `configs/benchmark.yaml` | **Part A grid:** `models`, `strategies`, `target_prevalences` (5 × 2 × 3 = **30** runs) |
| `configs/experiment.yaml` | Global experiment defaults, including **`random_seed`** (default **42**) |
| `configs/paths.yaml` | Default paths; **LI-Small** (`data/processed`, `data/splits`, `outputs/runs`, …) |
| `configs/paths_large_v2.yaml` | **LI-Large + v2 features:** `processed_v2`, `splits_v2`, `runs_v2`, `leaderboard_v2` |

Paths in YAML are relative to the **repository root**; absolute paths (e.g. Colab/Drive) are used as given (`PathConfig`).

**Typical choice for the large Part A run:** `--paths configs/paths_large_v2.yaml`.

Core contents of `paths_large_v2.yaml`:

- `raw_dir`, `processed_dir`, `splits_dir`, `outputs_dir`, `leaderboard_dir`
- Raw files: `LI-Large_Trans.csv`, `LI-Large_accounts.csv`, `LI-Large_Patterns.txt`
- Outputs: `transactions_labeled.parquet`, `split_manifest.json`, **`part_a_summary_v2.csv`**

---

## 2. Benchmark grid (`configs/benchmark.yaml`)

| Dimension | Values |
|-----------|--------|
| **Models** | `random_forest`, `xgboost` |
| **Strategies** | `baseline`, `random_undersampling`, `smote`, `adasyn`, `class_weighting` |
| **target_prevalences** | `0.010` (1 %), `0.005` (0.5 %), `0.001` (0.1 %) |

**Interpretation of `target_prevalence`** (as documented in code):

- **baseline:** natural training prevalence; parameter is logged for protocol only
- **random_undersampling:** majority class is **undersampled** until the target rate is reached
- **smote / adasyn:** minority class is **oversampled** (synthetically) until the target rate
- **class_weighting:** **no** resampling; weights derived from the target rate for the model

**Grid iteration order** (`grid_runner.run_grid`): outer loop **strategy** → **model** → **target_prevalence**.

`run_id` pattern per condition:

`{model_name}__{strategy}__p{permille}__{YYYYMMDD_HHMMSS}`  
(e.g. `xgboost__smote__p010__20260115_143022`)

Optional: `--benchmark /path/to/custom.yaml` copies that file to `configs/benchmark.yaml` and then runs `run_grid` (helper for Colab / one-off changes).

---

## 3. Pipeline stages (data → leaderboard)

### Stage 1 — Labelling

```text
python -m aml_benchmark.data.make_dataset --paths configs/paths_large_v2.yaml
```

- Input: `paths.raw_dir` (CSV/TXT per YAML)
- Output: **`{processed_dir}/transactions_labeled.parquet`**

### Stage 2 — Splitting

```text
python -m aml_benchmark.data.splitter --paths configs/paths_large_v2.yaml
```

- Sorted by time; roughly **70 / 15 / 15** row fractions
- Output under **`paths.splits_dir`:**
  - **`train.parquet`**, **`val.parquet`**, **`test.parquet`**
  - **`split_manifest.json`** (counts, prevalence, date ranges)

### Stage 3 — Benchmark grid (30 runs)

```text
python -m aml_benchmark.experiments.grid_runner --paths configs/paths_large_v2.yaml
```

Invokes `runner.run_experiment(...)` for each combination (resume: an already completed matching combination may be skipped — see `_find_completed_run` in `grid_runner.py`).

**Note:** `python -m aml_benchmark.experiments.runner` with no arguments runs only a **single smoke** experiment (default: RF, baseline, 1 %) with **`PathConfig()`** → default `configs/paths.yaml`. For Large you must set paths programmatically or use the grid with `--paths`.

### Stage 4a — Threshold optimisation (post-hoc)

```text
python -m aml_benchmark.experiments.re_evaluate --paths configs/paths_large_v2.yaml
```

- Loads **`model.pkl`** per run, uses **cached feature matrices** `load_features(splits_dir, "val"|"test")`
- Optimises threshold on **val** (default criterion **F1**), evaluates **test** **once** at that threshold
- No retraining

### Stage 4b — Aggregation

```text
python -m aml_benchmark.experiments.aggregate --paths configs/paths_large_v2.yaml
```

- Scans **`paths.outputs_dir`**, collects metrics + `run_config.json`
- Writes **`{leaderboard_dir}/{part_a_summary}`** — for v2 e.g. `outputs/leaderboard_v2/part_a_summary_v2.csv`

---

## 4. Single-experiment flow (`run_experiment`)

File: `src/aml_benchmark/experiments/runner.py`.

1. **Load splits:** `train` / `val` / `test` parquet
2. **Features (train-only encoder fit on the “no cache” pipeline path):**
   - On the **first** run: `FeaturePipeline` **fit_transform** on **raw train**, val/test **transform** only
   - **Caches under `paths.splits_dir`:**
     - **`train_features_v2.parquet`**, **`val_features_v2.parquet`**, **`test_features_v2.parquet`**
     - **`feature_pipeline_v2.pkl`** (global fit for all runs)
   - Once all three parquet caches exist, each further run only loads matrices + pipeline
3. **Sampling:** `apply_strategy(...)` only on **`X_train_raw` / `y_train_raw`** → `X_train`, `y_train`
4. **Training:** `get_model(model_name, random_state, class_weight=sampling_result.class_weight)` → `fit`
5. **Evaluation:** val + test with **`predict_proba(...)[:, 1]`**, metrics at **fixed threshold 0.5** (`compute_all_metrics`)
6. **Artefacts per run** under **`paths.outputs_dir / run_id/`:**

| File | Contents |
|------|----------|
| `run_config.json` | Model, strategy, prevalences, row counts, sampling stats, timings, … |
| `feature_pipeline.pkl` | Serialised pipeline (copy in run folder) |
| `model.pkl` | Fitted model |
| `metrics_val.json` / `.csv` | Val @ 0.5 |
| `metrics_test.json` / `.csv` | Test @ 0.5 |

After **`re_evaluate`**, additionally among others: **`metrics_*_thresh`**, **`threshold_info.json`**.

---

## 5. Part B dependency (brief)

The **Part A XGBoost baseline** (typically a `xgboost__baseline__…` condition under the same `splits_dir` / feature setup) can serve as the **reference model** for **`score_baseline_train`** and **PAI-HNU** (Part B) — see `partB.md` and `benchmark_part_b_pai_hnu.yaml`.

---

## 6. At a glance

1. **`paths*.yaml`** defines all directories and file names; **Large/v2** isolates outputs in `*_v2` folders.
2. **`benchmark.yaml`** defines the **30**-run grid; **`grid_runner`** orchestrates **`runner.run_experiment`**.
3. **Feature computation** happens **once**; afterwards **`train|val|test_features_v2.parquet`** dominate runtime.
4. **`re_evaluate`** + **`aggregate`** yield **threshold-adjusted** metrics and the **leaderboard CSV**.

---

