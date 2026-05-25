# Code submission — artefacts and consistency checks

Short checklist for examiners and for tidying before hand-in.

## PAI-HNU: `n_pos` and hard-negative cap

1. Open **`split_manifest.json`** next to your training splits (path from `configs/paths_large_v2.yaml` / `paths_large_part_b_pai_hnu.yaml`: `splits_dir` + `split_manifest.json`).
2. Find the `"splits"` array entry with **`"split": "train"`** and read **`n_positive`**. That value is **`n_pos`** in PAI-HNU (same as `sum(y_train)` in code).
3. With `hard_negative_cap_multiplier` **M** from `configs/benchmark_part_b_pai_hnu.yaml`, the **hard-pool cap** is **`M × n_pos`** (before pooling / availability limits; the sampler uses `min(planned, cap, n_neg_total)` — see `hard_negative_undersampling.py`).
4. **Reported LI-Large v2 thesis run:** `n_pos = 63,811`, `M = 20` → cap **1,276,220**, consistent with `docs/part_b_cap_analysis.md`. After any **re-split**, re-read the manifest; do not rely on this number alone.

## Smoke vs. production outputs

- **Production PAI-HNU** runs write under the `outputs_dir` in your paths YAML (e.g. `outputs/runs_part_b_pai_hnu/`).
- **`run_part_b_pai_hnu --sample-n-train …`** (smoke) writes under **`outputs/runs_part_b_pai_hnu_smoke/`**. That tree is **not** used for thesis tables and is listed in `.gitignore`. Delete it locally before submission if you do not want stray folders; or leave it only on your machine, not in the handed-in zip/repo snapshot.

## Legacy Part B configs

Optional YAMLs for older `true_cost_weighting` / multi-threshold layouts are **not included** in this repository snapshot. Restore from git history only if required.
