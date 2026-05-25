# Part B (PAI-HNU) — Hard negatives, selection, and retraining

This document describes **at code level** how *hard negatives* are defined and selected, how **PAI-HNU training** works, and which **files** are involved — including the optional **baseline retrain fallback**.

**Relevant modules:**

- `src/aml_benchmark/sampling/hard_negative_undersampling.py` — sampler, hard-negative selection
- `src/aml_benchmark/experiments/score_baseline_train.py` — baseline scoring & optional baseline retrain
- `src/aml_benchmark/experiments/run_part_b_pai_hnu.py` — PAI-HNU orchestration, new XGBoost training
- `configs/benchmark_part_b_pai_hnu.yaml` — shares, cap, prevalences

---

## 1. What counts as a “hard negative” in code

**Hard negatives** are **not** separately labelled rows; they are **true negatives** (`y == 0`) from the **training** split, ordered by a **risk score**.

| Aspect | Definition in code |
|--------|-------------------|
| **Score source** | Positive-class probability of the **Part A XGBoost baseline** model: `predict_proba(X)[:, 1]` for each training row (train feature cache only). |
| **Semantics** | Higher score ⇒ the model treats the row as more “positive-like” ⇒ among negatives, a **harder** negative. |

Top-k selection is implemented in `select_hard_negatives()` in `hard_negative_undersampling.py`:

- Input: `negative_indices` (all rows with `y == 0`), `negative_scores` (baseline scores at those positions), `n_target`.
- Output: indices of the **`n_target` largest** scores.
- Algorithm: **`np.argpartition`** with `kth = n_neg - n_target` — O(N), no full sort.
- **Tied scores:** Determinism via NumPy partition ordering.

---

## 2. Index selection flow (`build_pai_hnu_training_indices`)

File: `src/aml_benchmark/sampling/hard_negative_undersampling.py`, function `build_pai_hnu_training_indices`.

1. **All positives** are kept: `pos_idx = np.where(y == 1)`.
2. **Target negative count** `n_neg_target` from `target_prevalence` and `n_pos`:
   - `compute_target_negative_count`:  
     \(n_{\text{neg}} = \text{round}(n_{\text{pos}} \cdot (1-p) / p)\).
3. **Shares** (default from YAML, typically **50 % / 25 % / 25 %**), after normalisation:
   - `n_hard_planned ≈ hard_share × n_neg_target`
   - `n_temporal_planned ≈ temporal_share × n_neg_target`
   - `n_global_planned` = remainder (rounding via `round` / difference)
4. **Hard cap (“variant B”):**  
   `n_hard_actual = min(n_hard_planned, hard_negative_cap_multiplier × n_pos, n_neg_total)`  
   Default multiplier often **20** (see `configs/benchmark_part_b_pai_hnu.yaml`).  
   Shortfall in the hard quota due to the cap is **split half-and-half** across the planned temporal and global budgets (`leftover_from_cap`).

5. **Hard pool:** `select_hard_negatives(neg_idx_all, baseline_scores[neg_idx_all], n_hard_actual)`.

6. **Temporal pool:** Negatives **not** in `hard_neg_idx`; stratified across **`n_temporal_blocks`** equal index blocks (time order = row order after splitter).

7. **Global pool:** Random sample from the remainder; optional shortfall from temporal moved into the global budget (`fill_shortfall_from_global`).

### 2.1 `n_pos` and hard cap (manifest)

- **`n_pos`** in the sampler is the number of **training positives** (`y_train == 1`), identical to **`n_positive`** of the train split in **`split_manifest.json`** (path: `paths.splits_dir` / `split_manifest.json`, field `"splits"` → entry `"split": "train"`).
- **Hard cap:** `n_hard_cap = hard_negative_cap_multiplier × n_pos` (default multiplier **20** in `configs/benchmark_part_b_pai_hnu.yaml`).
- **Reference LI-Large v2 run:** `n_pos = 63_811` ⇒ `n_hard_cap = 1_276_220` — consistent with `docs/part_b_cap_analysis.md`. Re-check the manifest after any re-split.

**Disjointness:** The runner calls `validate_no_overlap(pos_idx, hard_neg_idx, temporal_neg_idx, global_neg_idx)`.

**Edge cases:** If `n_neg_target` ≥ available negatives, all negatives are taken (degenerate path with log warning).

---

## 3. Baseline scores (prerequisite for hard negatives)

File: `src/aml_benchmark/experiments/score_baseline_train.py`

**Purpose:** Score all training rows once with the **Part A baseline XGBoost** and cache them (hard-negative mining uses only these scores).

**Anti-leakage:** Only `load_features(paths.splits_dir, "train")` is read — no val/test features.

**Model resolution (priority):**

1. CLI `--baseline-model-path`
2. YAML field `baseline_model_path` (paths config)
3. Auto-discovery: `paths.outputs_dir / <preferred_run_id> / model.pkl` (see `benchmark_part_b_pai_hnu.yaml`)

**Written files (default under `paths.splits_dir`):**

| File | Contents |
|------|----------|
| `baseline_train_scores.parquet` | Columns `row_idx` (0 … N−1), `score` (class-1 probability) |
| `baseline_train_scores_meta.json` | e.g. `model_path`, `sha256_score_file`, runtime, device notes |

Scoring uses chunks (`_predict_in_chunks`), default chunk size 5_000_000 rows.

---

## 4. “Retraining” — two distinct meanings

### 4A. Training the PAI-HNU model (normal case)

File: `src/aml_benchmark/experiments/run_part_b_pai_hnu.py`

- The **baseline model** is **not** retrained.
- After `build_pai_hnu_training_indices`:  
  `train_idx = selection.all_idx`, then **shuffle** with `random_seed + 1`.  
  `X_train_sub = X_train[train_idx]`, `y_train_sub = y_train[train_idx]`.
- New **`get_model("xgboost", random_state=..., class_weight=...)`** — per benchmark typically **`class_weight: null`** (no extra `scale_pos_weight` on top of the constructed training set).
- **`model.fit(X_train_sub, y_train_sub)`**.

**Typical artefacts per run** (path: `paths.outputs_dir` or for smoke `outputs/runs_part_b_pai_hnu_smoke/`):

| Artefact | Meaning |
|----------|---------|
| `run_config.json` | Parameters, timings, `optimal_threshold_val`, smoke fields (`smoke_subsample_used`, `sample_n_train`, `row_index_mode`, optionally `subsample_row_mapping_parquet`) |
| `sampling_manifest.json` | Shares, cap, baseline path, score-cache path/hash, counts |
| `model.pkl` | **New PAI-HNU XGBoost** |
| `metrics_{val,test}.json`/`.csv` | Metrics @ default threshold (e.g. 0.5) |
| `metrics_{val,test}_opt.json`/`.csv` | Metrics @ F1-optimal threshold (chosen on val only) |
| `subsample_row_mapping.parquet` (smoke only) | `internal_row_idx`, `orig_row_idx` |

The **score cache** `baseline_train_scores.parquet` is **not** modified (unless you run `score_baseline_train` separately with `--overwrite`).

---

### 4B. Retraining the baseline locally (explicit fallback)

Only when running:  
`python -m aml_benchmark.experiments.score_baseline_train --paths … --retrain-baseline`

File: `score_baseline_train.py`, function `_retrain_baseline_locally`:

- Loads **full** `X_train` from the feature cache and **`y_train`** from `train.parquet`.
- Trains **`get_model("xgboost", random_state=seed, class_weight=None)`** on **all** training rows (no PAI-HNU sampling).
- Saves:  
  **`{paths.outputs_dir}/xgboost__baseline__retrain__<YYYYMMDD_HHMMSS>/model.pkl`**

`main()` then calls **`score_baseline_on_train(..., cli_baseline_path=<new model>, overwrite=True)`** — so **`baseline_train_scores.parquet`** and **`baseline_train_scores_meta.json`** are **regenerated**. All subsequent hard-negative selection uses **this** baseline.

**Code note:** Bit-identical replay of the original Part A run is only realistic with the same software/runtime stack.

**Affected files (overview):**

- New/overwrite: `data/splits_v2/baseline_train_scores.parquet`, `baseline_train_scores_meta.json` (concrete path = `paths.splits_dir` from the active YAML)
- New: `outputs/.../xgboost__baseline__retrain__*/model.pkl` (concretely: `paths.outputs_dir` from the paths YAML)

---

## 5. At a glance

1. **Hard negative** = true negatives with the **highest** Part A baseline **scores** (top-k via `argpartition`), optionally **capped**; remainder allocated temporally/globally.
2. **Scores** come from **`score_baseline_train`** → parquet + meta under **`splits_dir`**.
3. **PAI-HNU training** = **new** XGBoost on the **constructed** training set → run folder with **`model.pkl`**, metrics, manifests.
4. **Baseline retrain** = optional fallback → new **`xgboost__baseline__retrain__*`** + rescored **`baseline_train_scores.*`**.

---

