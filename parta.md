# Part A — Benchmark-Setup, Pipeline & Dateien

Dieses Dokument beschreibt **End-to-End**, wie der **Part-A-Benchmark** aufgebaut ist, welche **Konfigurationsdateien** und **Code-Module** eine Rolle spielen und **welche Artefakte** wo landen — analog zum Aufbau von `partb.md`.

**Relevante Module (Auswahl):**

- `src/aml_benchmark/config.py` — `PathConfig`, `load_yaml`
- `src/aml_benchmark/data/make_dataset.py` — Labelling / processed parquet
- `src/aml_benchmark/data/splitter.py` — zeitliche Splits
- `src/aml_benchmark/experiments/grid_runner.py` — 30-Run-Grid (Part A)
- `src/aml_benchmark/experiments/runner.py` — einzelnes Experiment (`run_experiment`)
- `src/aml_benchmark/features/feature_cache.py` — Feature-Caches unter `splits_dir`
- `src/aml_benchmark/sampling/strategies.py` — Imbalance-Strategien
- `src/aml_benchmark/experiments/re_evaluate.py` — F1-optimale Schwelle (Val → Test)
- `src/aml_benchmark/experiments/aggregate.py` — Leaderboard-CSV

---

## 1. Konfigurationsdateien

| Datei | Rolle |
|-------|--------|
| `configs/benchmark.yaml` | **Part-A-Grid:** `models`, `strategies`, `target_prevalences` (5 × 2 × 3 = **30** Läufe) |
| `configs/experiment.yaml` | Globale Experiment-Defaults, u. a. **`random_seed`** (Standard **42**) |
| `configs/paths.yaml` | Standard-Pfade; **LI-Small** (`data/processed`, `data/splits`, `outputs/runs`, …) |
| `configs/paths_large_v2.yaml` | **LI-Large + v2-Features:** `processed_v2`, `splits_v2`, `runs_v2`, `leaderboard_v2` |

Pfade in den YAML-Dateien sind relativ zum **Repository-Root**; absolute Pfade (z. B. Colab/Drive) werden unverändert übernommen (`PathConfig`).

**Typische Wahl für den großen Part-A-Lauf:** `--paths configs/paths_large_v2.yaml`.

Inhalt `paths_large_v2.yaml` (Kern):

- `raw_dir`, `processed_dir`, `splits_dir`, `outputs_dir`, `leaderboard_dir`
- Rohdateien: `LI-Large_Trans.csv`, `LI-Large_accounts.csv`, `LI-Large_Patterns.txt`
- Ausgaben: `transactions_labeled.parquet`, `split_manifest.json`, **`part_a_summary_v2.csv`**

---

## 2. Benchmark-Raster (`configs/benchmark.yaml`)

| Dimension | Werte |
|-----------|--------|
| **Models** | `random_forest`, `xgboost` |
| **Strategies** | `baseline`, `random_undersampling`, `smote`, `adasyn`, `class_weighting` |
| **target_prevalences** | `0.010` (1 %), `0.005` (0.5 %), `0.001` (0.1 %) |

**Interpretation von `target_prevalence`** (im Code dokumentiert):

- **baseline:** natürliche Train-Prävalenz; Parameter wird nur protokolliert
- **random_undersampling:** Majorität wird **unterstichprobt**, bis die Zielquote erreicht ist
- **smote / adasyn:** Minorität wird **übersampelt** (synthetisch), bis zur Zielquote
- **class_weighting:** **kein** Resampling; Gewichte aus der Zielquote für das Modell

**Grid-Reihenfolge** (`grid_runner.run_grid`): äußerer Loop **strategy** → **model** → **target_prevalence**.

`run_id`-Schema pro Bedingung:

`{model_name}__{strategy}__p{permille}__{YYYYMMDD_HHMMSS}`  
(z. B. `xgboost__smote__p010__20260115_143022`)

Optional: `--benchmark /pfad/zur/alten.yaml` kopiert die Datei nach `configs/benchmark.yaml` und startet dann `run_grid` (Hilfe für Colab/Abweichungen).

---

## 3. Pipeline-Stufen (Daten → Leaderboard)

### Stufe 1 — Labelling

```text
python -m aml_benchmark.data.make_dataset --paths configs/paths_large_v2.yaml
```

- Eingabe: `paths.raw_dir` (CSV/TXT laut YAML)
- Ausgabe: **`{processed_dir}/transactions_labeled.parquet`**

### Stufe 2 — Splitting

```text
python -m aml_benchmark.data.splitter --paths configs/paths_large_v2.yaml
```

- Sortierung nach Zeit; ca. **70 / 15 / 15** Zeilenanteile
- Ausgabe unter **`paths.splits_dir`:**
  - **`train.parquet`**, **`val.parquet`**, **`test.parquet`**
  - **`split_manifest.json`** (Zählungen, Prävalenz, Datumsbereiche)

### Stufe 3 — Benchmark-Grid (30 Runs)

```text
python -m aml_benchmark.experiments.grid_runner --paths configs/paths_large_v2.yaml
```

Ruft für jede Kombination `runner.run_experiment(...)` auf (Resume: bereits abgeschlossene gleiche Kombination kann übersprungen werden — siehe `_find_completed_run` in `grid_runner.py`).

**Hinweis:** `python -m aml_benchmark.experiments.runner` ohne Argumente startet nur **einen** Smoke-Lauf (Default: RF, baseline, 1 %) mit **`PathConfig()`** → Standard `configs/paths.yaml`. Für Large musst du die Pfade programmatisch setzen oder über das Grid mit `--paths` arbeiten.

### Stufe 4a — Schwellen-Optimierung (post-hoc)

```text
python -m aml_benchmark.experiments.re_evaluate --paths configs/paths_large_v2.yaml
```

- Lädt pro Run **`model.pkl`**, nutzt **gespeicherte Feature-Matrizen** `load_features(splits_dir, "val"|"test")`
- Optimiert Schwelle auf **Val** (Default-Kriterium **F1**), wertet **Test** **einmal** mit dieser Schwelle aus
- Kein Retraining

### Stufe 4b — Aggregation

```text
python -m aml_benchmark.experiments.aggregate --paths configs/paths_large_v2.yaml
```

- Scannt **`paths.outputs_dir`**, sammelt Metriken + `run_config.json`
- Schreibt **`{leaderboard_dir}/{part_a_summary}`** — bei v2 z. B. `outputs/leaderboard_v2/part_a_summary_v2.csv`

---

## 4. Ablauf eines einzelnen Experiments (`run_experiment`)

Datei: `src/aml_benchmark/experiments/runner.py`.

1. **Splits laden:** `train` / `val` / `test` parquet
2. **Features (Train-only fit des Encoders im Pipeline-Pfad „kein Cache“):**
   - Beim **ersten** Lauf: `FeaturePipeline` auf **Original-Train** `fit_transform`, Val/Test nur `transform`
   - **Cache-Pfade unter `paths.splits_dir`:**
     - **`train_features_v2.parquet`**, **`val_features_v2.parquet`**, **`test_features_v2.parquet`**
     - **`feature_pipeline_v2.pkl`** (globaler Fit für alle Runs)
   - Sobald alle drei Parquet-Caches existieren, lädt jeder weitere Run nur noch die Matrizen + Pipeline
3. **Sampling:** `apply_strategy(...)` nur auf **`X_train_raw` / `y_train_raw`** → `X_train`, `y_train`
4. **Training:** `get_model(model_name, random_state, class_weight=sampling_result.class_weight)` → `fit`
5. **Evaluation:** Val + Test mit **`predict_proba(...)[:, 1]`**, Metriken bei **fester Schwelle 0.5** (`compute_all_metrics`)
6. **Artefakte pro Run** unter **`paths.outputs_dir / run_id/`:**

| Datei | Inhalt |
|-------|--------|
| `run_config.json` | Modell, Strategie, Prävalenzen, Zeilen-Zähler, Sampling-Stats, Zeiten, … |
| `feature_pipeline.pkl` | Serialisierte Pipeline (Kopie im Run-Ordner) |
| `model.pkl` | Gefittetes Modell |
| `metrics_val.json` / `.csv` | Val @ 0.5 |
| `metrics_test.json` / `.csv` | Test @ 0.5 |

Nach **`re_evaluate`** zusätzlich u. a.: **`metrics_*_thresh`**, **`threshold_info.json`**.

---

## 5. Abhängigkeit Part B (kurz)

Die **Part-A-XGBoost-Baseline** (typischerweise eine Bedingung `xgboost__baseline__…` unter demselben `splits_dir` / Feature-Setup) kann als **Referenzmodell** für **`score_baseline_train`** und **PAI-HNU** (Part B) dienen — siehe `partb.md` und `benchmark_part_b_pai_hnu.yaml`.

---

## 6. Kurz-Faden

1. **`paths*.yaml`** definiert alle Verzeichnisse und Dateinamen; **Large/v2** isoliert Outputs in `*_v2`-Ordnern.
2. **`benchmark.yaml`** definiert das **30**-Läufe-Raster; **`grid_runner`** orchestriert **`runner.run_experiment`**.
3. **Feature-Berechnung** passiert **einmal**; danach dominieren **`train|val|test_features_v2.parquet`** die Laufzeit.
4. **`re_evaluate`** + **`aggregate`** liefern **schwellenangepasste** Metriken und die **Leaderboard-CSV**.

---

*Dokumentationsstand: konsistent zu `aml_benchmark` Part A (`grid_runner` / `runner`). Bei Code-Änderungen dieses File ggf. anpassen.*
