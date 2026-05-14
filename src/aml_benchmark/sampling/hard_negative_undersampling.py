"""Part-A-Informed Hard-Negative Undersampling (PAI-HNU).

This module implements Strategy 6 of the AML benchmark.  It constructs a
training set by:

1. retaining ALL positives (no information loss on the minority class), and
2. selecting a target number of negatives from three disjoint pools:

   * **Hard negatives**   – top-k risk scores from the Part-A XGBoost
     baseline; these are the negatives the baseline already finds confusing.
   * **Temporal-stratified random negatives** – uniform random sample
     drawn within ``n_temporal_blocks`` equal-size blocks of row-index
     (rows are already sorted chronologically by the splitter), preserving
     coverage across the training period.
   * **Global random negatives** – uniform random sample from the
     remaining negatives.

The shares (default 50/25/25) and the optional cap on hard negatives
(``hard_negative_cap_multiplier * n_pos``) are configurable from
``configs/benchmark_part_b_pai_hnu.yaml``.

Anti-leakage
------------
This module reads only training-split arrays.  Validation and test
arrays are never accepted as inputs.  Hard-negative scores must originate
from the Part-A baseline trained on the same training split (this is
enforced at the orchestrator level, not here).

Public API
----------
* :func:`compute_target_negative_count`
* :func:`select_hard_negatives`
* :func:`select_temporal_random_negatives`
* :func:`select_global_random_negatives`
* :func:`build_pai_hnu_training_indices`
* :func:`validate_no_overlap`
* :func:`save_sampling_manifest`
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from aml_benchmark.utils.logging_utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class PaiHnuSelection:
    """Indices selected by :func:`build_pai_hnu_training_indices`."""

    pos_idx: np.ndarray
    hard_neg_idx: np.ndarray
    temporal_neg_idx: np.ndarray
    global_neg_idx: np.ndarray
    target_prevalence: float
    n_neg_target: int
    achieved_prevalence: float
    counts: dict[str, int] = field(default_factory=dict)

    @property
    def all_neg_idx(self) -> np.ndarray:
        return np.concatenate(
            [self.hard_neg_idx, self.temporal_neg_idx, self.global_neg_idx]
        )

    @property
    def all_idx(self) -> np.ndarray:
        return np.concatenate([self.pos_idx, self.all_neg_idx])


# ---------------------------------------------------------------------------
# Negative-count budget
# ---------------------------------------------------------------------------

def compute_target_negative_count(
    n_positive: int, target_prevalence: float
) -> int:
    """Return ``n_neg_target`` such that ``n_pos / (n_pos + n_neg_target) == p``.

    Derivation:  ``p = n_pos / (n_pos + n_neg)``  =>  ``n_neg = n_pos * (1 - p) / p``.

    Parameters
    ----------
    n_positive:
        Number of positive (illicit) training rows; must be > 0.
    target_prevalence:
        Desired post-sampling positive fraction in (0, 1).
    """
    if n_positive <= 0:
        raise ValueError(f"n_positive must be > 0; got {n_positive}")
    if not (0.0 < target_prevalence < 1.0):
        raise ValueError(
            f"target_prevalence must be in (0, 1); got {target_prevalence}"
        )
    n_neg_target = int(round(n_positive * (1.0 - target_prevalence) / target_prevalence))
    return max(0, n_neg_target)


# ---------------------------------------------------------------------------
# Selectors
# ---------------------------------------------------------------------------

def _log_argpartition_memory(n_neg: int, k: int) -> None:
    """Estimate memory used by argpartition over the negative score array."""
    bytes_scores = n_neg * 4  # float32 view
    bytes_idx = n_neg * 8     # int64 indices
    total_mb = (bytes_scores + bytes_idx) / (1024 * 1024)
    logger.info(
        f"  argpartition: n_neg={n_neg:,} k={k:,} "
        f"(~{total_mb:.1f} MB temp arrays)"
    )


def select_hard_negatives(
    negative_indices: np.ndarray,
    negative_scores: np.ndarray,
    n_target: int,
) -> np.ndarray:
    """Return the indices of the ``n_target`` highest-scoring negatives.

    Uses ``np.argpartition`` (O(N)) instead of full sort (O(N log N)).
    Ties are broken by NumPy's internal partition ordering, which is
    deterministic for a given input.

    Parameters
    ----------
    negative_indices:
        Row indices of the negative training rows (length N_neg).
    negative_scores:
        Risk scores from the Part-A baseline at those indices (length N_neg).
        Higher score = more likely positive (= harder negative).
    n_target:
        How many indices to return.  Clipped to ``[0, len(negative_indices)]``.
    """
    if len(negative_indices) != len(negative_scores):
        raise ValueError(
            "negative_indices and negative_scores must have equal length; "
            f"got {len(negative_indices)} vs {len(negative_scores)}"
        )
    n_neg = len(negative_indices)
    n_target = max(0, min(n_target, n_neg))
    if n_target == 0:
        return np.empty(0, dtype=np.int64)
    if n_target == n_neg:
        return negative_indices.astype(np.int64, copy=False)

    _log_argpartition_memory(n_neg, n_target)
    # argpartition: indices of the (n_target) largest entries are placed at
    # the end of the returned permutation.
    part = np.argpartition(negative_scores, kth=n_neg - n_target)
    top_local = part[-n_target:]
    return negative_indices[top_local].astype(np.int64, copy=False)


def select_temporal_random_negatives(
    candidate_indices: np.ndarray,
    n_target: int,
    n_blocks: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Stratified random sample over ``n_blocks`` equal slices of the index range.

    The training parquet is already sorted chronologically by the splitter,
    so equal-size index blocks are equivalent to equal-size temporal blocks.

    Parameters
    ----------
    candidate_indices:
        Indices to draw from (typically the negatives left after hard
        negatives have been removed). Must be sorted ascending.
    n_target:
        Total number of indices to return.
    n_blocks:
        Number of temporal blocks.  Quotas are split as evenly as possible;
        any rounding remainder is added to the first blocks.
    rng:
        Seeded :class:`numpy.random.Generator` for reproducibility.
    """
    n_target = max(0, min(int(n_target), len(candidate_indices)))
    if n_target == 0 or len(candidate_indices) == 0:
        return np.empty(0, dtype=np.int64)
    n_blocks = max(1, int(n_blocks))

    # Per-block quotas (integer with leftover distributed to first blocks)
    base = n_target // n_blocks
    leftover = n_target - base * n_blocks
    quotas = np.full(n_blocks, base, dtype=np.int64)
    quotas[:leftover] += 1

    # Equal-size index slicing of the candidate array
    edges = np.linspace(0, len(candidate_indices), n_blocks + 1, dtype=np.int64)
    selected: list[np.ndarray] = []
    shortfall = 0
    for b in range(n_blocks):
        start, end = int(edges[b]), int(edges[b + 1])
        block = candidate_indices[start:end]
        q = int(quotas[b])
        if len(block) == 0 or q == 0:
            shortfall += q
            continue
        if q >= len(block):
            selected.append(block)
            shortfall += q - len(block)
        else:
            picks = rng.choice(block, size=q, replace=False)
            selected.append(picks)

    chosen = np.concatenate(selected) if selected else np.empty(0, dtype=np.int64)

    # If small blocks couldn't fill their quota, top-up uniformly from the
    # remaining (non-selected) candidates so the temporal share is honoured.
    if shortfall > 0:
        chosen_set = np.zeros(candidate_indices.max() + 1, dtype=bool)
        chosen_set[chosen] = True
        remaining_mask = ~chosen_set[candidate_indices]
        remaining = candidate_indices[remaining_mask]
        if len(remaining) > 0:
            extra_n = min(shortfall, len(remaining))
            extra = rng.choice(remaining, size=extra_n, replace=False)
            chosen = np.concatenate([chosen, extra])

    return chosen.astype(np.int64, copy=False)


def select_global_random_negatives(
    candidate_indices: np.ndarray,
    n_target: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Uniform random sample from ``candidate_indices`` (no stratification)."""
    n_target = max(0, min(int(n_target), len(candidate_indices)))
    if n_target == 0 or len(candidate_indices) == 0:
        return np.empty(0, dtype=np.int64)
    if n_target == len(candidate_indices):
        return candidate_indices.astype(np.int64, copy=False)
    return rng.choice(candidate_indices, size=n_target, replace=False).astype(
        np.int64, copy=False
    )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def _normalise_shares(
    hard: float, temporal: float, global_: float
) -> tuple[float, float, float]:
    s = float(hard) + float(temporal) + float(global_)
    if s <= 0:
        raise ValueError("Sampling shares must sum to a positive value.")
    return float(hard) / s, float(temporal) / s, float(global_) / s


def build_pai_hnu_training_indices(
    y_train: np.ndarray,
    baseline_scores: np.ndarray,
    target_prevalence: float,
    hard_negative_share: float = 0.50,
    temporal_random_share: float = 0.25,
    global_random_share: float = 0.25,
    n_temporal_blocks: int = 20,
    hard_negative_cap_multiplier: int | None = 20,
    fill_shortfall_from_global: bool = True,
    random_state: int = 42,
) -> PaiHnuSelection:
    """Build the PAI-HNU training-row selection.

    Parameters
    ----------
    y_train:
        Binary label array for the full training split.
    baseline_scores:
        Part-A baseline scores aligned 1:1 with ``y_train``
        (``len(baseline_scores) == len(y_train)`` is asserted).
    target_prevalence:
        Desired positive fraction after sampling, in (0, 1).
    hard_negative_share, temporal_random_share, global_random_share:
        Pool shares; will be re-normalised to sum to 1.
    n_temporal_blocks:
        Number of equal-size temporal blocks used by the temporal-random pool.
    hard_negative_cap_multiplier:
        If not ``None``, the actual hard-negative count is capped at
        ``hard_negative_cap_multiplier * n_positive``.  Any quota leftover
        from this cap is reallocated 50/50 to the temporal and global pools.
    fill_shortfall_from_global:
        When the temporal-random pool can't deliver its full quota (e.g.
        empty blocks after hard-negative removal), redirect the shortfall
        to the global random pool so the overall ``n_neg_target`` is
        honoured.
    random_state:
        Seed for the RNG used by both random selectors.
    """
    if len(y_train) != len(baseline_scores):
        raise ValueError(
            f"y_train and baseline_scores length mismatch: "
            f"{len(y_train)} vs {len(baseline_scores)}"
        )
    if np.isnan(baseline_scores).any():
        raise ValueError("baseline_scores contains NaN entries.")

    y = np.asarray(y_train, dtype=np.int8)
    pos_idx = np.where(y == 1)[0].astype(np.int64)
    neg_idx_all = np.where(y == 0)[0].astype(np.int64)
    n_pos = int(len(pos_idx))
    n_neg_total = int(len(neg_idx_all))
    if n_pos == 0:
        raise ValueError("No positives in y_train; cannot sample.")
    if n_neg_total == 0:
        raise ValueError("No negatives in y_train; cannot sample.")

    n_neg_target = compute_target_negative_count(n_pos, target_prevalence)

    if n_neg_target >= n_neg_total:
        logger.warning(
            f"n_neg_target ({n_neg_target:,}) >= available negatives "
            f"({n_neg_total:,}). Returning all negatives unchanged "
            "(degenerate case for very low target_prevalence)."
        )
        return PaiHnuSelection(
            pos_idx=pos_idx,
            hard_neg_idx=neg_idx_all,
            temporal_neg_idx=np.empty(0, dtype=np.int64),
            global_neg_idx=np.empty(0, dtype=np.int64),
            target_prevalence=target_prevalence,
            n_neg_target=n_neg_total,
            achieved_prevalence=n_pos / float(n_pos + n_neg_total),
            counts={
                "n_pos": n_pos,
                "n_neg_total_available": n_neg_total,
                "n_neg_target": n_neg_total,
                "n_hard": n_neg_total,
                "n_temporal": 0,
                "n_global": 0,
            },
        )

    # Normalise shares
    hs, ts, gs = _normalise_shares(
        hard_negative_share, temporal_random_share, global_random_share
    )
    n_hard_planned = int(round(hs * n_neg_target))
    n_temporal_planned = int(round(ts * n_neg_target))
    n_global_planned = n_neg_target - n_hard_planned - n_temporal_planned

    # Hard cap (Variant B)
    n_hard_cap = (
        hard_negative_cap_multiplier * n_pos
        if hard_negative_cap_multiplier is not None
        else n_hard_planned
    )
    n_hard_actual = min(n_hard_planned, n_hard_cap, n_neg_total)
    leftover_from_cap = n_hard_planned - n_hard_actual
    if leftover_from_cap > 0:
        add_temporal = leftover_from_cap // 2
        add_global = leftover_from_cap - add_temporal
        n_temporal_planned += add_temporal
        n_global_planned += add_global
        logger.info(
            f"Hard-negative cap binds: planned={n_hard_planned:,} "
            f"capped={n_hard_actual:,} "
            f"(cap = {hard_negative_cap_multiplier} * n_pos = {n_hard_cap:,}). "
            f"Reallocated {leftover_from_cap:,} -> temporal+{add_temporal:,}, "
            f"global+{add_global:,}."
        )

    # ------- Hard negatives ------------------------------------------------
    rng = np.random.default_rng(random_state)
    neg_scores = baseline_scores[neg_idx_all]
    hard_neg_idx = select_hard_negatives(
        negative_indices=neg_idx_all,
        negative_scores=neg_scores,
        n_target=n_hard_actual,
    )

    # Remaining pool (negatives that are NOT hard-selected)
    hard_mask = np.zeros(len(y), dtype=bool)
    hard_mask[hard_neg_idx] = True
    remaining_neg_idx = neg_idx_all[~hard_mask[neg_idx_all]]

    # ------- Temporal random ----------------------------------------------
    temporal_neg_idx = select_temporal_random_negatives(
        candidate_indices=remaining_neg_idx,
        n_target=n_temporal_planned,
        n_blocks=n_temporal_blocks,
        rng=rng,
    )
    temp_mask = np.zeros(len(y), dtype=bool)
    temp_mask[temporal_neg_idx] = True

    # ------- Global random -------------------------------------------------
    candidate_for_global = remaining_neg_idx[~temp_mask[remaining_neg_idx]]
    actual_temporal = len(temporal_neg_idx)
    shortfall_temp = max(0, n_temporal_planned - actual_temporal)
    n_global_actual_target = n_global_planned + (
        shortfall_temp if fill_shortfall_from_global else 0
    )
    global_neg_idx = select_global_random_negatives(
        candidate_indices=candidate_for_global,
        n_target=n_global_actual_target,
        rng=rng,
    )

    n_total_neg_actual = (
        len(hard_neg_idx) + len(temporal_neg_idx) + len(global_neg_idx)
    )
    achieved = n_pos / float(n_pos + n_total_neg_actual)

    counts = {
        "n_pos": n_pos,
        "n_neg_total_available": n_neg_total,
        "n_neg_target": n_neg_target,
        "n_hard_planned": n_hard_planned,
        "n_hard_cap": int(n_hard_cap),
        "n_hard_actual": int(len(hard_neg_idx)),
        "n_temporal_planned": n_temporal_planned,
        "n_temporal_actual": int(len(temporal_neg_idx)),
        "n_global_planned": n_global_planned,
        "n_global_actual": int(len(global_neg_idx)),
        "n_total_neg_actual": int(n_total_neg_actual),
    }
    logger.info("PAI-HNU selection summary: " + json.dumps(counts))

    return PaiHnuSelection(
        pos_idx=pos_idx,
        hard_neg_idx=hard_neg_idx,
        temporal_neg_idx=temporal_neg_idx,
        global_neg_idx=global_neg_idx,
        target_prevalence=target_prevalence,
        n_neg_target=n_neg_target,
        achieved_prevalence=achieved,
        counts=counts,
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_no_overlap(*index_arrays: Iterable[int]) -> None:
    """Assert that each index appears in at most one of the provided arrays.

    Used as a paranoid check that hard / temporal / global pools, plus
    positives, never reuse the same row.
    """
    seen = set()
    for arr in index_arrays:
        a = np.asarray(arr, dtype=np.int64)
        s = set(a.tolist())
        overlap = seen & s
        if overlap:
            raise AssertionError(
                f"Index overlap detected: {len(overlap)} duplicate indices, "
                f"e.g. {sorted(list(overlap))[:5]}"
            )
        seen.update(s)


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def save_sampling_manifest(
    selection: PaiHnuSelection,
    output_path: Path,
    *,
    target_prevalence: float,
    sampling_shares: dict[str, float],
    random_seed: int,
    baseline_run_id: str,
    baseline_model_path: str,
    score_cache_path: str,
    score_cache_sha256: str,
    n_train_total: int,
    score_meta: dict | None = None,
    extra: dict | None = None,
) -> Path:
    """Persist a JSON manifest describing how the training set was constructed.

    The manifest is required for thesis-grade reproducibility: it allows a
    reviewer to reconstruct the exact training rows used by every
    PAI-HNU run from {y_train, baseline_train_scores, seed}.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    manifest = {
        "strategy": "pai_hnu",
        "target_prevalence": target_prevalence,
        "sampling_shares": sampling_shares,
        "random_seed": random_seed,
        "baseline": {
            "preferred_run_id": baseline_run_id,
            "model_path": baseline_model_path,
            "score_cache_path": score_cache_path,
            "score_cache_sha256": score_cache_sha256,
            "score_meta": score_meta or {},
        },
        "n_train_total_original": int(n_train_total),
        "selection_counts": selection.counts,
        "achieved_prevalence": selection.achieved_prevalence,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    if extra:
        manifest.update(extra)

    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    logger.info(f"Sampling manifest saved -> {output_path}")
    return output_path


def load_baseline_score_cache(score_cache_path: Path) -> pd.DataFrame:
    """Load the parquet score cache and validate (row_idx, score) schema."""
    df = pd.read_parquet(score_cache_path)
    if not {"row_idx", "score"}.issubset(df.columns):
        raise ValueError(
            f"score cache must contain (row_idx, score); got {df.columns.tolist()}"
        )
    df = df.sort_values("row_idx", kind="stable").reset_index(drop=True)
    expected = np.arange(len(df), dtype=np.int64)
    if not np.array_equal(df["row_idx"].to_numpy(dtype=np.int64), expected):
        raise ValueError(
            "score cache row_idx must be 0..N-1 contiguous and unique."
        )
    return df
