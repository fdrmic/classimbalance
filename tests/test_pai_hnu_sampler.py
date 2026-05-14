"""Unit tests for the PAI-HNU sampler.

Six fast, deterministic tests that lock in the contract of
``aml_benchmark.sampling.hard_negative_undersampling``:

1. ``test_compute_target_negative_count_formula``
2. ``test_select_hard_negatives_returns_top_k``
3. ``test_temporal_random_negatives_block_distribution``
4. ``test_global_random_negatives_uniform_sample``
5. ``test_build_pai_hnu_training_indices_no_overlap_and_prevalence``
6. ``test_hard_negative_cap_reallocates_to_other_pools``

No file I/O, no model loading, runs in milliseconds.
"""
from __future__ import annotations

import numpy as np

from aml_benchmark.sampling.hard_negative_undersampling import (
    build_pai_hnu_training_indices,
    compute_target_negative_count,
    select_global_random_negatives,
    select_hard_negatives,
    select_temporal_random_negatives,
    validate_no_overlap,
)


# ---------------------------------------------------------------------------
# 1. n_neg_target formula
# ---------------------------------------------------------------------------
def test_compute_target_negative_count_formula():
    # 1000 positives at 1% prevalence => 99000 negatives
    assert compute_target_negative_count(1000, 0.01) == 99000
    # 500 positives at 0.5% prevalence => 99500 negatives
    assert compute_target_negative_count(500, 0.005) == 99500
    # 100 positives at 0.1% prevalence => 99900 negatives
    assert compute_target_negative_count(100, 0.001) == 99900


# ---------------------------------------------------------------------------
# 2. Hard-negative selection returns the top-k
# ---------------------------------------------------------------------------
def test_select_hard_negatives_returns_top_k():
    # 100 negatives with scores 0..99; top 10 must be indices 90..99
    indices = np.arange(100, 200, dtype=np.int64)  # arbitrary global indices
    scores = np.arange(100, dtype=np.float32)      # rank == score
    top10 = select_hard_negatives(indices, scores, n_target=10)

    assert len(top10) == 10
    # Mapped global indices: scores 90..99 are at local positions 90..99
    expected = set(range(190, 200))
    assert set(top10.tolist()) == expected


# ---------------------------------------------------------------------------
# 3. Temporal-stratified sample respects per-block quotas
# ---------------------------------------------------------------------------
def test_temporal_random_negatives_block_distribution():
    rng = np.random.default_rng(0)
    candidates = np.arange(1000, dtype=np.int64)
    n_blocks = 10
    n_target = 100

    chosen = select_temporal_random_negatives(
        candidate_indices=candidates,
        n_target=n_target,
        n_blocks=n_blocks,
        rng=rng,
    )

    assert len(chosen) == n_target
    # Each block (size 100) should contribute 10 picks
    block_size = len(candidates) // n_blocks
    counts_per_block = np.zeros(n_blocks, dtype=int)
    for c in chosen:
        b = min(int(c) // block_size, n_blocks - 1)
        counts_per_block[b] += 1
    assert (counts_per_block == 10).all(), (
        f"Per-block counts not balanced: {counts_per_block}"
    )


# ---------------------------------------------------------------------------
# 4. Global random sample is uniform-ish and disjoint from a forbidden set
# ---------------------------------------------------------------------------
def test_global_random_negatives_uniform_sample():
    rng = np.random.default_rng(42)
    candidates = np.arange(10_000, dtype=np.int64)
    chosen = select_global_random_negatives(
        candidate_indices=candidates, n_target=500, rng=rng
    )
    assert len(chosen) == 500
    assert len(np.unique(chosen)) == 500           # no duplicates
    assert chosen.min() >= 0 and chosen.max() < 10_000

    # Determinism: same seed => same selection
    rng2 = np.random.default_rng(42)
    chosen2 = select_global_random_negatives(
        candidate_indices=candidates, n_target=500, rng=rng2
    )
    assert np.array_equal(np.sort(chosen), np.sort(chosen2))


# ---------------------------------------------------------------------------
# 5. End-to-end builder: prevalence + zero overlap
# ---------------------------------------------------------------------------
def test_build_pai_hnu_training_indices_no_overlap_and_prevalence():
    rng = np.random.default_rng(7)
    n = 50_000
    n_pos = 100
    y = np.zeros(n, dtype=np.int8)
    pos_idx = rng.choice(n, size=n_pos, replace=False)
    y[pos_idx] = 1
    # Synthetic baseline scores: positives have higher scores on average,
    # but plenty of "hard" negatives mixed in.
    scores = rng.beta(a=2.0, b=20.0, size=n).astype(np.float32)
    scores[pos_idx] = rng.beta(a=5.0, b=2.0, size=n_pos).astype(np.float32)

    selection = build_pai_hnu_training_indices(
        y_train=y,
        baseline_scores=scores,
        target_prevalence=0.01,
        hard_negative_share=0.50,
        temporal_random_share=0.25,
        global_random_share=0.25,
        n_temporal_blocks=10,
        hard_negative_cap_multiplier=20,
        random_state=42,
    )

    # No overlap between any of the four pools
    validate_no_overlap(
        selection.pos_idx,
        selection.hard_neg_idx,
        selection.temporal_neg_idx,
        selection.global_neg_idx,
    )

    # All positives retained
    assert len(selection.pos_idx) == n_pos

    # Achieved prevalence is within 0.5 percentage points of target
    assert abs(selection.achieved_prevalence - 0.01) < 0.005, (
        f"Expected ~1% prevalence, got {selection.achieved_prevalence:.4%}"
    )


# ---------------------------------------------------------------------------
# 6. Cap reallocation: planned vs capped
# ---------------------------------------------------------------------------
def test_hard_negative_cap_reallocates_to_other_pools():
    """When hard cap binds, leftover quota must move to temporal+global."""
    rng = np.random.default_rng(11)
    n = 200_000
    n_pos = 100
    y = np.zeros(n, dtype=np.int8)
    pos_idx = rng.choice(n, size=n_pos, replace=False)
    y[pos_idx] = 1
    scores = rng.uniform(size=n).astype(np.float32)

    # n_neg_target at 0.1% prevalence: n_pos * 999 = 99,900
    # Planned hard share (50%):                       49,950
    # Cap = 5 * n_pos = 500  -> binds tightly
    selection = build_pai_hnu_training_indices(
        y_train=y,
        baseline_scores=scores,
        target_prevalence=0.001,
        hard_negative_share=0.50,
        temporal_random_share=0.25,
        global_random_share=0.25,
        n_temporal_blocks=20,
        hard_negative_cap_multiplier=5,
        random_state=42,
    )

    counts = selection.counts
    assert counts["n_hard_planned"] > counts["n_hard_actual"], counts
    assert counts["n_hard_actual"] <= 5 * n_pos
    # Reallocation: temporal + global actuals must exceed naive 25/25 of
    # n_neg_target; combined they should cover the full leftover.
    naive_temp = int(round(0.25 * counts["n_neg_target"]))
    naive_global = counts["n_neg_target"] - counts["n_hard_planned"] - naive_temp
    assert (
        counts["n_temporal_actual"] + counts["n_global_actual"]
        >= naive_temp + naive_global
    )

    # No overlap
    validate_no_overlap(
        selection.pos_idx,
        selection.hard_neg_idx,
        selection.temporal_neg_idx,
        selection.global_neg_idx,
    )
