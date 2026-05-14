# AML Benchmark — Bachelor Thesis (FHNW FS26)

**Technical documentation and replication guide** for the empirical benchmark in *class imbalance mitigation for AML transaction classification* on **IBM AMLworld Low-Illicit Large (LI-Large)**.

The companion thesis defines the full methodology (Chapter 5). This document maps that design to **code, configs, and CLI entry points** so examiners can install, locate artefacts, and reproduce the pipeline structure.

**Where the reported runs actually live:** Full-scale Part A / Part B jobs for this thesis were run in **Google Colab**, orchestrated mainly from **`notebooks/`** (clone → `pip install -e .` → cells invoke the same `aml_benchmark` CLI/modules as below). `PathConfig` YAML files typically set **`outputs_dir`**, **`leaderboard_dir`**, and data paths to **absolute locations on the mounted Google Drive**. The directories **`outputs/`** and **`results/`** at the **repository root** may therefore be **empty, incomplete, or only hold copies** you added for submission or local smoke tests — that does **not** contradict the code layout described here.

**Example paths (Google Colab + Drive — replace with your layout):**

| Role | Example path |
|------|----------------|
| Drive root after `drive.mount` | `/content/drive/MyDrive/` |
| Aggregated experiment / backup parent | `/content/drive/MyDrive/aml_results/` |
| One Part A backup folder (timestamp in name varies) | `/content/drive/MyDrive/aml_results/large_run_v2_20260407_1904/` |
| Per-run artefacts | `.../large_run_v2_<batch>/runs/<model>__<strategy>__pXXX__<timestamp>/` (`model.pkl`, `run_config.json`, `metrics_*.json`, …) |
| Leaderboard CSV inside that backup | `.../large_run_v2_<batch>/leaderboard/part_a_summary_v2.csv` |
| Feature-importance sweep over all `runs/` trees (notebook) | e.g. pass several `--runs-roots` under `aml_results/large_run_v2_*/runs` or `--aml-results /content/drive/MyDrive/aml_results` (see `extract_f2_from_runs.py` / `notebooks/feature_importance.ipynb`) |
| Large raw / processed data (typical) | Often a separate folder such as `/content/drive/MyDrive/aml_data/` — must match the paths in `configs/paths_large_v2.yaml` |

In Colab, **`/content/classimbalance`** (or similar) is the **cloned repo**; **artefacts** usually live under **`/content/drive/...`**, not under the repo root.

---

## Table of contents

1. [Quick start (replication overview)](#1-quick-start-replication-overview)
2. [Research design (summary)](#2-research-design-summary)
3. [Installation and dependencies](#3-installation-and-dependencies)
4. [Data acquisition](#4-data-acquisition)
5. [Repository layout](#5-repository-layout)
6. [End-to-end pipeline](#6-end-to-end-pipeline)
7. [Dataset (LI-Large)](#7-dataset-li-large)
8. [Labelling](#8-labelling)
9. [Temporal split](#9-temporal-split)
10. [Feature engineering (30 features)](#10-feature-engineering-30-features)
11. [Imbalance mitigation strategies (Part A)](#11-imbalance-mitigation-strategies-part-a)
12. [Models and hyperparameters](#12-models-and-hyperparameters)
13. [Evaluation metrics and threshold policy](#13-evaluation-metrics-and-threshold-policy)
14. [Results aggregation and thesis tables](#14-results-aggregation-and-thesis-tables)
15. [Part B — PAI-HNU (primary contribution)](#15-part-b--pai-hnu-primary-contribution)
16. [Legacy / exploratory Part B code (optional)](#16-legacy--exploratory-part-b-code-optional)
17. [Feature importance](#17-feature-importance)
18. [Reproducibility](#18-reproducibility)
19. [Hardware and runtime](#19-hardware-and-runtime)
20. [Known limitations](#20-known-limitations)
21. [Appendix — CLI cheat sheet](#21-appendix--cli-cheat-sheet)

---

## 1. Quick start (replication overview)

**Prerequisites:** Python ≥ 3.10, the LI-Large raw files under `data/raw/` (see §4), and sufficient RAM/GPU for full runs (§19). Paths in `configs/*.yaml` may use absolute locations (e.g. Google Drive mount on Colab).

**Primary replication path for examiners:** Follow **`notebooks/`** in order (Part A large run → PAI-HNU → optional feature importance), after mounting Drive and adjusting path variables/YAML to match your Drive tree. Local CLI-only replication is equivalent if you point the same YAML keys to writable directories on your machine.

**Minimal install:**

```bash
pip install -e .
# Optional: pinned stack for examination (maintain requirements.txt alongside pyproject.toml)
pip install -r requirements.txt
```

**Typical order of execution (Part A v2 on LI-Large):**

1. `python -m aml_benchmark.data.make_dataset --paths configs/paths_large_v2.yaml`
2. `python -m aml_benchmark.data.splitter --paths configs/paths_large_v2.yaml`
3. `python -m aml_benchmark.experiments.grid_runner --paths configs/paths_large_v2.yaml` (30 runs; resume-capable)
4. `python -m aml_benchmark.experiments.re_evaluate --paths configs/paths_large_v2.yaml` (F1-optimal threshold on validation → test)
5. `python -m aml_benchmark.experiments.aggregate --paths configs/paths_large_v2.yaml`

**Part B (PAI-HNU)** — after Part A baseline artefacts and feature cache exist:

1. `pytest tests/test_pai_hnu_sampler.py -v`
2. `python -m aml_benchmark.experiments.score_baseline_train --paths configs/paths_large_part_b_pai_hnu.yaml`
3. `python -m aml_benchmark.experiments.run_part_b_pai_hnu --paths configs/paths_large_part_b_pai_hnu.yaml`

**Colab notebooks** (orchestrations with Drive paths) live under `notebooks/` and/or the repository root; adjust paths to your environment. See §5.

**Thesis tables (CSV + Markdown):**

```bash
python -m aml_benchmark.analysis.results_tables
```

---

## 2. Research design (summary)

- **Part A:** Controlled factorial benchmark — **five** training-time strategies (`baseline`, `random_undersampling`, `smote`, `adasyn`, `class_weighting`) × **two** model families (Random Forest, XGBoost) × **three** target training prevalences (0.1 %, 0.5 %, 1.0 %) → **30 executed runs**. The **main comparative analysis** uses **20 conditions** (baseline is prevalence-invariant → one reference per model; ADASYN is reported separately in the thesis appendix due to structural constraints of the subsampled density step).
- **Part B (thesis “sixth strategy”):** **Part-A-informed Hard-Negative Undersampling (PAI-HNU)** — XGBoost only; same splits, metrics, and hyperparameters as Part A; constructs training data from **all positives** plus **50 % / 25 % / 25 %** hard-negative, temporal-stratified, and global-random negatives (details in `docs/part_b_pai_hnu_design.md` and §15).
- **Isolation principle:** Feature engineering, hyperparameters, temporal split, metric definitions, and **random seed 42** are fixed across conditions unless noted; only strategy, model class, and target prevalence vary (Part A), or sampling construction varies (Part B).

---

## 3. Installation and dependencies

- **Package definition:** `pyproject.toml` (installable package `aml_benchmark` under `src/`).
- **Version pins:** If you submit `requirements.txt`, treat it as the **examiner-facing frozen stack** for the environment used to produce the reported numbers; `pyproject.toml` may retain **lower bounds** only.

Declared core libraries (see `pyproject.toml` for current minimums) include **pandas**, **pyarrow**, **numpy**, **PyYAML**, **scikit-learn**, **imbalanced-learn**, **xgboost**, **joblib**, and optional plotting/notebook dependencies.

To record exact versions from your run environment:

```bash
python --version
pip freeze
```

---

## 4. Data acquisition

**Source:** IBM AMLworld synthetic AML dataset (Low-Illicit Large). Public download (e.g. Kaggle) — use the transaction CSV, accounts CSV, and optional patterns file as required by your ingestion path.

**Expected raw layout** (typical; exact names resolved in `configs/paths_large_v2.yaml`):

| File | Role |
|------|------|
| `LI-Large_Trans.csv` | Transaction log |
| `LI-Large_accounts.csv` | Account → entity metadata (for entity-type encoding) |
| `LI-Large_Patterns.txt` | Pattern metadata; **not used** for LI-Large labelling in this pipeline (performance); labels come from the CSV ground-truth column |

Place files under `data/raw/` or set **absolute paths** in YAML for Colab/Drive.

---

## 5. Repository layout

```
classimbalance/
├── configs/                  # paths*.yaml, split.yaml, experiment.yaml, benchmark*.yaml
├── data/
│   ├── raw/                  # Immutable IBM exports (not committed if large)
│   ├── processed_v2/         # Labelled parquet(s)
│   └── splits_v2/          # train/val/test parquet + feature cache + manifest
├── docs/
│   └── part_b_pai_hnu_design.md
├── notebooks/                # Primary Colab runbooks for thesis-scale experiments (edit Drive paths)
├── outputs/                  # *Logical* artefact tree when outputs_dir is inside the repo (local/tests)
│   ├── runs_v2/              # Part A runs — on Colab often under Drive, see paths_large_v2.yaml
│   ├── runs_part_b_pai_hnu/  # PAI-HNU runs — see paths_large_part_b_pai_hnu.yaml
│   └── leaderboard_v2/       # part_a_summary_v2.csv — may live on Drive after Colab aggregate
├── results/                  # inputs for results_tables + generated tables; may be copied from Drive or empty in git
├── tests/
│   └── test_pai_hnu_sampler.py
├── scripts/                  # Optional local analytics (not imported as package)
│   ├── analyze_part_a_v2.py  # Summaries / reports from part_a_summary_v2.csv
│   └── analyze_part_b_v2.py  # Part B v2 comparison helpers
├── analysis/                 # Standalone repo-root utilities (run as scripts)
│   ├── feature_importance.py # CLI: aggregate feature_importances_ from run folders
│   ├── visualize_results.py  # Figures from leaderboard CSVs → analysis/figures/
│   └── figures/              # Generated PNGs (e.g. PR-AUC, recall–precision)
├── src/aml_benchmark/        # Installable package (pip install -e .)
│   ├── config.py             # PathConfig, YAML loading
│   ├── data/                 # ingest, schema, labeler, make_dataset, splitter, pattern_parser
│   ├── features/             # pipeline, aggregator, feature_cache
│   ├── sampling/             # strategies, prevalence, hard_negative_undersampling
│   ├── models/               # factory (RF, XGBoost)
│   ├── evaluation/           # metrics
│   ├── experiments/          # runner, grid_runner, re_evaluate, aggregate,
│   │                         # score_baseline_train, run_part_b_pai_hnu, threshold_optimizer
│   ├── analysis/             # results_tables.py — thesis tables (python -m aml_benchmark.analysis.results_tables)
│   ├── reporting/            # build_part_a_tables, build_part_b_tables (Markdown/LaTeX)
│   └── utils/                # io, logging, hashing
├── pyproject.toml
└── README_PROJECT_STRUCTURE.md   # This file
```

**Two different `analysis` locations:**

| Path | Role |
|------|------|
| **`analysis/`** (repo root) | Helper **scripts** executed as `python analysis/<script>.py`; not part of the `aml_benchmark` import path unless you add it manually. |
| **`src/aml_benchmark/analysis/`** | Package module **`results_tables`** — run as `python -m aml_benchmark.analysis.results_tables` after `pip install -e .`. |

**Configuration:** `PathConfig` (`src/aml_benchmark/config.py`) loads YAML; relative paths resolve from the repository root, absolute paths are used as-is (e.g. Colab `/content/drive/...`).

**Repo vs. Drive layout:** Backup folders on Drive (e.g. `aml_results/large_run_v2_<timestamp>/`) may bundle `runs/`, `leaderboard/`, `processed/`, and `splits/` in one tree that **differs from the flat `outputs/` sketch above**. That is expected: the **code** writes to whatever `outputs_dir` / `leaderboard_dir` / `splits_dir` the active YAML defines. The tree in this section shows the **package’s conventional directory names** (`runs_v2`, `runs_part_b_pai_hnu`, etc.); your Colab config may redirect all of that under a single Drive parent.

---

## 6. End-to-end pipeline

```
Raw CSV  →  make_dataset  →  transactions_labeled.parquet
         →  splitter      →  train/val/test.parquet + split_manifest.json
         →  grid_runner / runner  →  outputs/.../<run_id>/ (model, metrics @0.5)
         →  re_evaluate   →  metrics_*_thresh + threshold_info (F1 on val)
         →  aggregate     →  part_a_summary_v2.csv
Part B   →  score_baseline_train  →  baseline scores on train only
         →  run_part_b_pai_hnu    →  PAI-HNU runs + metrics
```

Artefact folders in the diagram are **where the pipeline writes for the currently loaded `paths_*.yaml`** — typically **Google Drive** for thesis-scale Colab runs, not necessarily `./outputs` in a local git clone.

---

## 7. Dataset (LI-Large)

**Scale (thesis):** ~176 M transactions; **splits** approximately **123.25M / 26.41M / 26.41M** train/val/test (70/15/15). Natural training prevalence ≈ **0.052 %**; validation and test are slightly higher — preserved as realistic drift.

**Authoritative row counts and date ranges:** `data/splits_v2/split_manifest.json` (produced by `data/splitter.py`).

**Schema helpers:** `src/aml_benchmark/data/schema.py` (`RAW_TRANS_COLUMNS`, canonical columns for modelling).

---

## 8. Labelling

**Canonical binary target:** `label` = IBM generator column `is_laundering_csv` (CSV column mapped in schema). **Pattern joins are disabled for LI-Large** in the labelling path used at scale; pattern-based labels would under-cover layering-stage illicit events.

See `src/aml_benchmark/data/labeler.py`, `make_dataset.py`.

---

## 9. Temporal split

**Method:** Sort by `timestamp`, partition by **row index quantiles** (70/15/15). No random permutation — **purely temporal**.

**Leakage controls relevant to features:**  
(1) `FeaturePipeline.fit_transform` only on **train**; val/test **transform** only.  
(2) Account-level rolling features in `features/aggregator.py` use **strictly past** transactions in each window relative to the current row’s timestamp.

`configs/split.yaml` stores the ratio; `data/splitter.py` implements the split and manifest.

---

## 10. Feature engineering (30 features)

**Implementation:** `src/aml_benchmark/features/pipeline.py` + `src/aml_benchmark/features/aggregator.py` (+ optional `feature_cache.py` for persisted matrices).

**Assembly order** (column index matches `FEATURE_NAMES` and per-run `run_config.json:feature_names`):

| Block | Count | Content |
|-------|------:|---------|
| Numeric | 2 | `amount_paid`, `amount_received` — **log1p** |
| Categorical | 2 | `payment_format`, `payment_currency` — **OrdinalEncoder** fit on train; unknown → NaN |
| Derived | 8 | `hour`, `day_of_week`, `same_bank_flag`, `self_transfer_flag`, `currency_mismatch`, `amount_ratio` (raw ratio clipped to [0,10]; paid=0 → 1.0), `fan_in_score`, `fan_out_score` (from 7d account rolls, clipped; see code) |
| Account-level | 18 | Sender/receiver: tx counts (1d/7d/30d), avg amounts (7d/30d), unique counterparties (7d/30d), cross-bank ratio (30d), entity type |

**Cache (Part A v2 grid):** After the first full feature build, matrices may be stored as `data/splits_v2/*_features_v2.parquet` and `feature_pipeline_v2.pkl` to avoid recomputing rolling aggregates.

---

## 11. Imbalance mitigation strategies (Part A)

**Module:** `src/aml_benchmark/sampling/strategies.py`, `prevalence.py`.

**Target prevalence → imblearn ratio:** \(r = p / (1-p)\) with `prevalence_to_ratio`.

| Strategy | Mechanism |
|----------|-----------|
| `baseline` | No change; prevalence recorded only |
| `random_undersampling` | `RandomUnderSampler` when imposed prevalence > natural |
| `smote` | `SMOTE`, `k_neighbors = min(5, n_pos-1)` |
| `adasyn` | `ADASYN` with **majority subsample cap (500,000)** for density estimation; synthetic counts scaled to full-train target; subsample uses `ratio_sub` bounded by `min(ratio_sub, 1.0)` per thesis — see code and thesis §5.5.1 |
| `class_weighting` | No resampling; `w0=1`, `w1=(1-p)/p` → RF `class_weight`; XGB `scale_pos_weight = w1/w0` |

**Additional implemented helpers** (used in **legacy / exploratory** Part B grids, **not** the thesis PAI-HNU definition):

| Key | Purpose |
|-----|---------|
| `smote_class_weighting` | Combined SMOTE + cost weighting (experimental grid) |
| `true_cost_weighting` | Weights from **observed** class counts (`n_neg/n_pos`) without resampling |

These are **not** PAI-HNU; PAI-HNU is implemented only via `sampling/hard_negative_undersampling.py` and `run_part_b_pai_hnu.py`.

---

## 12. Models and hyperparameters

**Factory:** `src/aml_benchmark/models/factory.py`.

**XGBoost:** `n_estimators=200`, `max_depth=6`, `learning_rate=0.05`, `subsample=0.8`, `colsample_bytree=0.8`, `tree_method="hist"`, `eval_metric="aucpr"`, `random_state` from config. **`device`:** `"cuda"` if `nvidia-smi` succeeds, else `"cpu"`.

**Random Forest:** `n_estimators=100`, `max_features="sqrt"`, `min_samples_leaf=5`, **`max_samples=200_000`** (memory cap on LI-Large), `n_jobs=4`, `random_state` from config.

**Thesis alignment:** RF bootstrap cap is **pragmatic** (OOM with full bootstrap at ~123M rows × parallelism); interpret RF results under that structural constraint (see thesis §8).

---

## 13. Evaluation metrics and threshold policy

**Module:** `src/aml_benchmark/evaluation/metrics.py`.

**Primary:** PR-AUC (`average_precision_score`). **Secondary:** ROC-AUC.

**Threshold-dependent (at chosen threshold):** precision, recall, F1, **F2** (β=2), weighted accuracy (positive weight `n_neg/n_pos`), TP/FP/TN/FN.

**Threshold selection (thesis):** F1 maximised on **validation** scores; that threshold is applied **once** to **test** without further tuning. Implemented in `experiments/re_evaluate.py` (writes `metrics_*_thresh.json` and `threshold_info.json` per Part A run). Part B PAI-HNU uses the same policy for its packaged `*_opt` metrics (see `run_part_b_pai_hnu.py`).

**Part A leaderboard:** `aggregate.py` can include `f2_*` columns when present in per-run JSONs; otherwise F2 may be derived post hoc from reported precision and recall at the F1-optimal threshold (thesis §5.6).

---

## 14. Results aggregation and thesis tables

**Part A leaderboard:** `python -m aml_benchmark.experiments.aggregate --paths configs/paths_large_v2.yaml` → path given by your YAML’s `leaderboard_dir` / `part_a_summary` (e.g. `outputs/leaderboard_v2/part_a_summary_v2.csv` locally, or a CSV on Drive after Colab).

**Thesis tables:** `python -m aml_benchmark.analysis.results_tables` expects inputs under the repo’s **`results/`** directory by default (`results/part_a_summary_v2.csv`, feature-importance CSVs, etc.) and scans **`outputs/runs_part_b_pai_hnu/`** relative to the project root for PAI-HNU runs. If your truth files live only on **Drive**, **copy or symlink** the needed CSVs and ensure PAI-HNU run folders are visible at the paths documented in `results_tables.py`, or extend paths in a small fork — the thesis tables are generated from **files on disk**, not from Colab implicitly.

---

## 15. Part B — PAI-HNU (primary contribution)

**Name in thesis:** Part-A-informed Hard-Negative Undersampling (**PAI-HNU**) — the **sixth** imbalance strategy; **XGBoost only**; same hyperparameters as Part A; **no** extra `scale_pos_weight` (distribution handles imbalance).

**Design reference:** `docs/part_b_pai_hnu_design.md` (mapping table, anti-leakage, smoke levels).

| Component | Path |
|-----------|------|
| Sampler | `src/aml_benchmark/sampling/hard_negative_undersampling.py` |
| Baseline scores on **train only** | `src/aml_benchmark/experiments/score_baseline_train.py` |
| Orchestrator | `src/aml_benchmark/experiments/run_part_b_pai_hnu.py` |
| Configs | `configs/benchmark_part_b_pai_hnu.yaml`, `configs/paths_large_part_b_pai_hnu.yaml` |
| Tests | `tests/test_pai_hnu_sampler.py` |

**Anti-leakage (summary):** Baseline that ranks hard negatives is trained on **train** only; scoring cache covers **training rows only**; sampler uses **training indices**; val/test untouched; final shuffle uses seed **`random_seed + 1`** (42 → 43) per thesis §5.7.

**Outputs:** One directory per run as configured by **`paths_large_part_b_pai_hnu.yaml`** (conventionally `.../runs_part_b_pai_hnu/<run_id>/` when outputs sit next to the repo; on Colab often under **`aml_results`/analogous Drive folders**). Each run includes `run_config.json`, `sampling_manifest.json`, `model.pkl`, metrics at 0.5 and F1-opt-derived `*_opt` artefacts as implemented.

---

## 16. Legacy / exploratory Part B code (optional)

The repository may still contain **auxiliary** experiments **not** central to the final thesis narrative:

| Component | Description |
|-----------|-------------|
| `experiments/threshold_optimizer.py` | Multi **operating point** selection on a **fixed** Part A XGBoost baseline (F1/F2/precision-constrained) — **no retraining** |
| `configs/benchmark_part_b.yaml`, `paths_large_part_b_v3.yaml` | Grid for `true_cost_weighting` / older Part B layout |
| `configs/benchmark_part_b_multi.yaml` | Multi-threshold config hook |

You may delete or archive these **after** confirming your submitted thesis does not depend on their tables. `results_tables.py` skips optional inputs if files are missing.

---

## 17. Feature importance

**Script:** `analysis/feature_importance.py` — walks completed run directories, reads `model.feature_importances_` and `run_config.json:feature_names`, writes aggregate CSVs (RF: impurity-based MDI; XGBoost: library default importances — state explicitly in thesis).

**Notebook:** `notebooks/feature_importance.ipynb` (or equivalent) for Google Colab + Drive paths.

---

## 18. Reproducibility

| Mechanism | Detail |
|-----------|--------|
| Global seed | `configs/experiment.yaml` — `random_seed: 42` |
| PAI-HNU shuffle | Derived seed **43** (`random_seed + 1`) for final training-set shuffle |
| Split | Deterministic temporal indices — **no** RNG |
| Artefacts | Per-run `run_config.json` + serialised `model.pkl` and metrics JSON/CSV |
| Paths | YAML-driven `PathConfig` for local vs Colab |
| Immutability | `data/raw/` treated as read-only |

---

## 19. Hardware and runtime

**Reference environment (thesis):** Google Colab Pro+ with **~179 GB RAM** and **NVIDIA A100-class GPU** for XGBoost. Random Forest uses **`n_jobs=4`** to stay within memory limits.

**Order-of-magnitude runtimes:** XGBoost Part A conditions often **a few minutes** with GPU; Random Forest **longer**; exact `train_time_sec` per run is stored in `run_config.json`.

Smaller machines may run **subsampled** PAI-HNU (`--sample-n-train`) for smoke tests only.

---

## 20. Known limitations

1. **No hyperparameter tuning** — isolates strategies, not tuned models.  
2. **Single temporal split** — no cross-validation folds.  
3. **No global graph features** (PageRank, betweenness) — out of scope.  
4. **LI-Large:** pattern-based stratification disabled; labels from CSV ground truth.  
5. **RF `max_samples=200_000`** — structural cap on positives per tree; interpret RF under thesis §5.4 / §8.

---

## 21. Appendix — CLI cheat sheet

All commands assume `pip install -e .` and cwd = repository root.

### Part A (LI-Large v2)

```bash
python -m aml_benchmark.data.make_dataset --paths configs/paths_large_v2.yaml
python -m aml_benchmark.data.splitter --paths configs/paths_large_v2.yaml
python -m aml_benchmark.experiments.grid_runner --paths configs/paths_large_v2.yaml
python -m aml_benchmark.experiments.re_evaluate --paths configs/paths_large_v2.yaml
python -m aml_benchmark.experiments.aggregate --paths configs/paths_large_v2.yaml
```

### Part B — PAI-HNU

```bash
pytest tests/test_pai_hnu_sampler.py -v

python -m aml_benchmark.experiments.score_baseline_train \
    --paths configs/paths_large_part_b_pai_hnu.yaml

# Optional: explicit baseline model path or retrain fallback
python -m aml_benchmark.experiments.score_baseline_train \
    --paths configs/paths_large_part_b_pai_hnu.yaml \
    --baseline-model-path /path/to/model.pkl

python -m aml_benchmark.experiments.score_baseline_train \
    --paths configs/paths_large_part_b_pai_hnu.yaml --retrain-baseline

# Smoke: subsampled train, reduced prevalence
python -m aml_benchmark.experiments.run_part_b_pai_hnu \
    --paths configs/paths_large_part_b_pai_hnu.yaml \
    --target-prevalences 0.01 \
    --sample-n-train 200000

# Full PAI-HNU grid (confirm resources)
python -m aml_benchmark.experiments.run_part_b_pai_hnu \
    --paths configs/paths_large_part_b_pai_hnu.yaml
```

### Legacy — `true_cost_weighting` grid (optional)

```bash
python -m aml_benchmark.experiments.grid_runner \
    --paths configs/paths_large_part_b_v3.yaml \
    --benchmark configs/benchmark_part_b.yaml
```

### Legacy — multi-threshold on fixed baseline scores (optional)

```bash
python -m aml_benchmark.experiments.threshold_optimizer --paths configs/paths_large_v2.yaml
python -m aml_benchmark.experiments.threshold_optimizer --paths configs/paths_large_v2.yaml --dry-run
```

### Thesis tables and feature importance

```bash
python -m aml_benchmark.analysis.results_tables
python analysis/feature_importance.py --paths configs/paths_large_v2.yaml
```

### Single debug run (Part A runner defaults)

```bash
python -m aml_benchmark.experiments.runner --paths configs/paths_large_v2.yaml
```

---

*Module mapping (thesis Ch. 5 ↔ code): dataset §7–8 → `data/*`; split §9 → `splitter.py`; features §10 → `features/pipeline.py`, `aggregator.py`; strategies §11 → `sampling/strategies.py`; models §12 → `models/factory.py`; metrics §13 → `metrics.py`, `re_evaluate.py`; PAI-HNU §15 → `hard_negative_undersampling.py`, `score_baseline_train.py`, `run_part_b_pai_hnu.py`.*

*Document version: 2026-05 — aligned with bachelor thesis methodology (PAI-HNU as Part B strategy; legacy Part B tools documented separately).*
