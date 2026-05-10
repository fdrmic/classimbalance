# Part B (PAI-HNU) — Hard Negatives, Selektion & Retraining

Dieses Dokument beschreibt **codegenau**, wie *Hard Negatives* definiert und selektiert werden, wie das **PAI-HNU-Training** abläuft und welche **Dateien** betroffen sind — inklusive des optionalen **Baseline-Retrain-Fallbacks**.

**Relevante Module:**

- `src/aml_benchmark/sampling/hard_negative_undersampling.py` — Sampler, Hard-Negative-Auswahl
- `src/aml_benchmark/experiments/score_baseline_train.py` — Baseline-Scoring & optional Baseline-Retrain
- `src/aml_benchmark/experiments/run_part_b_pai_hnu.py` — PAI-HNU-Orchestrierung, neues XGBoost-Training
- `configs/benchmark_part_b_pai_hnu.yaml` — Shares, Cap, Prevalences

---

## 1. Was im Code als „Hard Negative“ gilt

**Hard Negatives** sind **keine** separat gelabelten Zeilen, sondern **echte Negative** (`y == 0`) aus dem Trainingssplit, die nach einem **Risiko-Score** sortiert sind.

| Aspekt | Definition im Code |
|--------|-------------------|
| **Score-Quelle** | Positive-Klassen-Wahrscheinlichkeit des **Part-A-XGBoost-Baseline-Modells**: `predict_proba(X)[:, 1]` für jede Trainingszeile (nur Train-Feature-Cache). |
| **Semantik** | Höherer Score ⇒ Modell hält die Zeile für eher „positiv“ ⇒ unter den Negativen ein **schwereres** Negativ. |

Implementierung der Top-k-Auswahl: `select_hard_negatives()` in `hard_negative_undersampling.py`:

- Eingabe: `negative_indices` (alle Zeilen mit `y == 0`), `negative_scores` (Baseline-Scores an denselben Positionen), `n_target`.
- Ausgabe: Indizes der **`n_target` höchsten** Scores.
- Algorithmus: **`np.argpartition`** mit `kth = n_neg - n_target` —komplexität O(N), keine vollständige Sortierung.
- **Gleiche Scores:** Determinismus über die NumPy-Partition-Reihenfolge.

---

## 2. Ablauf der Indexwahl (`build_pai_hnu_training_indices`)

Datei: `src/aml_benchmark/sampling/hard_negative_undersampling.py`, Funktion `build_pai_hnu_training_indices`.

1. **Alle Positiven** bleiben enthalten: `pos_idx = np.where(y == 1)`.
2. **Zielanzahl Negativer** `n_neg_target` aus `target_prevalence` und `n_pos`:
   - `compute_target_negative_count`:  
     \(n_{\text{neg}} = \text{round}(n_{\text{pos}} \cdot (1-p) / p)\).
3. **Anteile** (Standard aus YAML, typisch **50 % / 25 % / 25 %**), nach Summennormalisierung:
   - `n_hard_planned ≈ hard_share × n_neg_target`
   - `n_temporal_planned ≈ temporal_share × n_neg_target`
   - `n_global_planned` = Rest (abrunden/Aufrunden über `round` / Differenz)
4. **Hard-Cap („Variant B“):**  
   `n_hard_actual = min(n_hard_planned, hard_negative_cap_multiplier × n_pos, n_neg_total)`  
   Standard-Multiplier oft **20** (siehe `configs/benchmark_part_b_pai_hnu.yaml`).  
   Fehlende Hard-Quote durch den Cap wird **hälftig** auf die geplanten temporal- und global-Kontingente **aufgeschlagen** (`leftover_from_cap`).

5. **Hard-Pool:** `select_hard_negatives(neg_idx_all, baseline_scores[neg_idx_all], n_hard_actual)`.

6. **Temporal-Pool:** Nur Negativen, die **nicht** in `hard_neg_idx` liegen; stratifiziert über **`n_temporal_blocks`** gleich große Index-Blöcke (Zeitreihenfolge = Row-Order nach Splitter).

7. **Global-Pool:** Zufallsstichprobe aus dem Rest; optional Shortfall vom Temporal-Pool zum globalen Budget (`fill_shortfall_from_global`).

**Disjunktheit:** Der Runner ruft `validate_no_overlap(pos_idx, hard_neg_idx, temporal_neg_idx, global_neg_idx)` auf.

**Randfälle:** Wenn `n_neg_target ≥` verfügbare Negativen, werden alle Negativen genommen (degenerierter Pfad mit Log-Warnung).

---

## 3. Baseline-Scores (Voraussetzung für Hard Negatives)

Datei: `src/aml_benchmark/experiments/score_baseline_train.py`

**Zweck:** Einmalig alle Trainingszeilen mit dem **Part-A-Baseline-XGBoost** scoren und cachen (Hard-Negative-Mining nutzt nur diese Scores).

**Anti-Leakage:** Es wird nur `load_features(paths.splits_dir, "train")` gelesen — keine Val/Test-Features.

**Modell-Auflösung (Priorität):**

1. CLI `--baseline-model-path`
2. YAML-Feld `baseline_model_path` (Paths-Config)
3. Auto-Discovery: `paths.outputs_dir / <preferred_run_id> / model.pkl` (siehe `benchmark_part_b_pai_hnu.yaml`)

**Geschriebene Dateien (Standard unter `paths.splits_dir`):**

| Datei | Inhalt |
|-------|--------|
| `baseline_train_scores.parquet` | Spalten `row_idx` (0 … N−1), `score` (Klasse-1-Wahrscheinlichkeit) |
| `baseline_train_scores_meta.json` | u. a. `model_path`, `sha256_score_file`, Laufzeit, Device-Hinweise |

Scoring erfolgt in Chunks (`_predict_in_chunks`), Default-Chunkgröße 5_000_000 Zeilen.

---

## 4. „Retraining“ — zwei getrennte Bedeutungen

### 4A. PAI-HNU-Modell trainieren (Normalfall)

Datei: `src/aml_benchmark/experiments/run_part_b_pai_hnu.py`

- Das **Baseline-Modell** wird **nicht** neu trainiert.
- Nach `build_pai_hnu_training_indices`:  
  `train_idx = selection.all_idx`, dann **Shuffle** mit `random_seed + 1`.  
  `X_train_sub = X_train[train_idx]`, `y_train_sub = y_train[train_idx]`.
- Neues **`get_model("xgboost", random_state=..., class_weight=...)`** — laut Benchmark typisch **`class_weight: null`** (kein zusätzliches `scale_pos_weight` neben dem konstruierten Trainingsset).
- **`model.fit(X_train_sub, y_train_sub)`**.

**Typische Artefakte pro Run** (Pfad: `paths.outputs_dir` oder bei Smoke `outputs/runs_part_b_pai_hnu_smoke/`):

| Artefakt | Bedeutung |
|----------|-----------|
| `run_config.json` | Parameter, Zeiten, `optimal_threshold_val`, Smoke-Felder (`smoke_subsample_used`, `sample_n_train`, `row_index_mode`, ggf. `subsample_row_mapping_parquet`) |
| `sampling_manifest.json` | Shares, Cap, Baseline-Pfad, Score-Cache-Pfad/-SHA, Zählungen |
| `model.pkl` | **Neues PAI-HNU-XGBoost** |
| `metrics_{val,test}.json`/`.csv` | Metriken @ Default-Threshold (z. B. 0.5) |
| `metrics_{val,test}_opt.json`/`.csv` | Metriken @ F1-optimaler Schwelle (nur auf Val gewählt) |
| `subsample_row_mapping.parquet` (nur Smoke) | `internal_row_idx`, `orig_row_idx` |

Der **Score-Cache** `baseline_train_scores.parquet` wird dabei **nicht** geändert (außer du startest `score_baseline_train` separat mit `--overwrite`).

---

### 4B. Baseline lokal neu trainieren (expliziter Fallback)

Nur bei:  
`python -m aml_benchmark.experiments.score_baseline_train --paths … --retrain-baseline`

Datei: `score_baseline_train.py`, Funktion `_retrain_baseline_locally`:

- Lädt **vollen** `X_train` aus dem Feature-Cache und **`y_train`** aus `train.parquet`.
- Trainiert **`get_model("xgboost", random_state=seed, class_weight=None)`** auf **allen** Trainingszeilen (kein PAI-HNU-Sampling).
- Speichert:  
  **`{paths.outputs_dir}/xgboost__baseline__retrain__<YYYYMMDD_HHMMSS>/model.pkl`**

`main()` ruft danach **`score_baseline_on_train(..., cli_baseline_path=<neues Modell>, overwrite=True)`** auf — damit werden **`baseline_train_scores.parquet`** und **`baseline_train_scores_meta.json`** **neu** erzeugt. Alle späteren Hard-Negative-Selektionen beziehen sich dann auf **diese** Baseline.

**Hinweis im Code:** Bit-identisches Replay zum Original-Part-A-Lauf ist nur bei gleicher Software-/Runtime-Umgebung realistisch.

**Betroffene Dateien (überblick):**

- Neu/überschrieben: `data/splits_v2/baseline_train_scores.parquet`, `baseline_train_scores_meta.json` (konkreter Pfad = `paths.splits_dir` aus dem verwendeten YAML)
- Neu: `outputs/.../xgboost__baseline__retrain__*/model.pkl` (konkret: `paths.outputs_dir` aus dem Paths-YAML)

---

## 5. Kurz-Faden

1. **Hard Negative** = echte Negativen mit den **höchsten** Part-A-Baseline-**Scores** (Top-k via `argpartition`), optional **gecappt**; Restquote wird temporal/global verteilt bzw. umgerechnet.
2. **Scores** entstehen durch **`score_baseline_train`** → Parquet + Meta unter **`splits_dir`**.
3. **PAI-HNU-Training** = **neues** XGBoost auf der **konstruierten** Trainingsmenge → Run-Ordner mit **`model.pkl`**, Metriken, Manifests.
4. **Baseline-Retrain** = optionaler Fallback → neues **`xgboost__baseline__retrain__*`** + **neu** gescorte **`baseline_train_scores.*`**.

---

*Dokumentationsstand: konsistent zu `aml_benchmark` Part B PAI-HNU (Strategy 6). Bei Code-Änderungen dieses File ggf. anpassen.*
