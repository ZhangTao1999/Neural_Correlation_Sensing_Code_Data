"""Calibration sampling and grouped training targets."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GroupedCalibration:
    """Unique patterns, posterior targets, and their occurrence weights."""

    patterns: np.ndarray
    targets: np.ndarray
    weights: np.ndarray
    counts: np.ndarray


def sample_counts(
    probabilities: np.ndarray,
    shots_per_bin: int,
    seed: int,
) -> np.ndarray:
    """Sample calibration counts from ``p(pattern | parameter_bin)``.

    Parameters
    ----------
    probabilities:
        Array with shape ``(n_bins, n_patterns)``.
    shots_per_bin:
        Equal number of calibration measurements collected in every bin.
    seed:
        NumPy random seed.

    Returns
    -------
    np.ndarray
        Integer counts with shape ``(n_patterns, n_bins)``.
    """
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 2:
        raise ValueError("probabilities must have shape (n_bins, n_patterns)")
    if shots_per_bin < 1:
        raise ValueError("shots_per_bin must be positive")
    row_sums = probabilities.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0):
        raise ValueError("every parameter bin must have nonzero probability")
    normalized = probabilities / row_sums
    rng = np.random.default_rng(seed)
    counts_by_bin = np.stack(
        [rng.multinomial(shots_per_bin, row) for row in normalized], axis=0
    )
    return counts_by_bin.T


def grouped_calibration(
    patterns: np.ndarray,
    counts: np.ndarray,
    pseudocount: float = 1.0,
    include_unobserved: bool = True,
) -> GroupedCalibration:
    """Group snapshot-level cross entropy by unique measurement pattern.

    With equal calibration shots in every parameter bin, the normalized row of
    ``counts`` estimates the single-shot posterior ``P(parameter | pattern)``.
    Weighting each row loss by its total occurrence count is algebraically
    equivalent to evaluating cross entropy over all labelled snapshots.
    """
    patterns = np.asarray(patterns, dtype=np.float32)
    counts = np.asarray(counts)
    if counts.ndim != 2 or patterns.ndim != 2:
        raise ValueError("patterns and counts must both be two-dimensional")
    if counts.shape[0] != patterns.shape[0]:
        raise ValueError("patterns and counts disagree on n_patterns")
    if pseudocount < 0:
        raise ValueError("pseudocount must be nonnegative")

    raw_weights = counts.sum(axis=1).astype(np.float64)
    smoothed = counts.astype(np.float64) + pseudocount
    denominators = smoothed.sum(axis=1, keepdims=True)
    targets = np.divide(
        smoothed,
        denominators,
        out=np.full_like(smoothed, 1.0 / counts.shape[1]),
        where=denominators > 0,
    )

    if include_unobserved:
        keep = np.ones(counts.shape[0], dtype=bool)
        weights = raw_weights + pseudocount * counts.shape[1]
    else:
        keep = raw_weights > 0
        weights = raw_weights

    return GroupedCalibration(
        patterns=patterns[keep],
        targets=targets[keep].astype(np.float32),
        weights=weights[keep].astype(np.float32),
        counts=counts[keep],
    )

