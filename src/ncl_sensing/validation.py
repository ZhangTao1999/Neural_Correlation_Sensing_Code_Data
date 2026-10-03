"""Diagnostics for comparing neural and tabulated posterior estimators."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class PosteriorAgreement:
    """Weighted discrepancy between two single-shot posterior maps."""

    weighted_kl: float
    weighted_total_variation: float
    max_total_variation: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def normalize_log_probabilities(log_probabilities: np.ndarray) -> np.ndarray:
    """Normalize rows of an array represented in log space."""
    values = np.asarray(log_probabilities, dtype=np.float64)
    shifted = values - np.max(values, axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def posterior_agreement(
    reference: np.ndarray,
    candidate_log_probabilities: np.ndarray,
    weights: np.ndarray | None = None,
) -> PosteriorAgreement:
    """Compare a candidate posterior map with normalized reference rows."""
    reference = np.asarray(reference, dtype=np.float64)
    if reference.ndim != 2:
        raise ValueError("reference must have shape (n_patterns, n_bins)")
    row_sums = reference.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0):
        raise ValueError("every reference row must have nonzero probability")
    reference = reference / row_sums
    candidate = normalize_log_probabilities(candidate_log_probabilities)
    if candidate.shape != reference.shape:
        raise ValueError("reference and candidate shapes differ")

    if weights is None:
        normalized_weights = np.full(reference.shape[0], 1.0 / reference.shape[0])
    else:
        normalized_weights = np.asarray(weights, dtype=np.float64)
        if normalized_weights.shape != (reference.shape[0],):
            raise ValueError("weights must have shape (n_patterns,)")
        if np.any(normalized_weights < 0) or normalized_weights.sum() <= 0:
            raise ValueError("weights must be nonnegative with a positive sum")
        normalized_weights = normalized_weights / normalized_weights.sum()

    tiny = np.finfo(np.float64).tiny
    per_pattern_kl = np.sum(
        reference
        * (
            np.log(np.clip(reference, tiny, None))
            - np.log(np.clip(candidate, tiny, None))
        ),
        axis=1,
    )
    total_variation = 0.5 * np.abs(reference - candidate).sum(axis=1)
    return PosteriorAgreement(
        weighted_kl=float(normalized_weights @ per_pattern_kl),
        weighted_total_variation=float(normalized_weights @ total_variation),
        max_total_variation=float(np.max(total_variation)),
    )

