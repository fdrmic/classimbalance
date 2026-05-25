# Table 9: PAI-HNU Sampling Cap Analysis

Defense-relevant sampling statistics (from `part_b_pai_hnu_summary.csv`).

**Ground truth for `n_pos`:** Train positives = `n_positive` in `split_manifest.json` under the `"split": "train"` entry (same value the sampler uses). For the reported LI-Large v2 thesis run, **`n_pos = 63,811`**, so with `hard_negative_cap_multiplier: 20` in `configs/benchmark_part_b_pai_hnu.yaml`, **`n_hard_cap = 1,276,220`** — matching the `n_hard_cap` column below.

| Target Prev. | n_hard_planned | n_hard_cap | n_hard_actual | Cap utilization | n_temporal_actual | n_global_actual | Effective hard share |
|:---|---:|---:|---:|---:|---:|---:|---:|
| 0.1% | 31,873,594 | 1,276,220 | 1,276,220 | 4.0% | 31,235,484 | 31,235,485 | 2.0% |
| 0.5% | 6,349,194 | 1,276,220 | 1,276,220 | 20.1% | 5,711,084 | 5,711,085 | 10.1% |
| 1.0% | 3,158,644 | 1,276,220 | 1,276,220 | 40.4% | 2,520,534 | 2,520,535 | 20.2% |
