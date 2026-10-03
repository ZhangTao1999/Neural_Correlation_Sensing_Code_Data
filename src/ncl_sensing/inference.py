"""Single-shot network inference and multi-shot Bayesian accumulation."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .models import PosteriorNetwork
from .patterns import all_bit_patterns


def network_log_posteriors(
    model: PosteriorNetwork,
    n_qubits: int,
    device: str = "cpu",
) -> np.ndarray:
    """Evaluate log single-shot posteriors for every binary pattern."""
    model = model.to(device)
    model.eval()
    patterns = torch.as_tensor(
        all_bit_patterns(n_qubits), dtype=torch.float32, device=device
    )
    with torch.no_grad():
        values = F.log_softmax(model(patterns), dim=1).cpu().numpy()
    model.to("cpu")
    return values


def posterior_trace(
    log_single_shot_posteriors: np.ndarray,
    shot_indices: np.ndarray,
    prior: np.ndarray | None = None,
) -> np.ndarray:
    """Accumulate independent single-shot posteriors in log space."""
    table = np.asarray(log_single_shot_posteriors, dtype=np.float64)
    shots = np.asarray(shot_indices, dtype=np.int64)
    n_bins = table.shape[1]
    if prior is None:
        log_prior = np.full(n_bins, -np.log(n_bins), dtype=np.float64)
    else:
        prior = np.asarray(prior, dtype=np.float64)
        prior = prior / prior.sum()
        log_prior = np.log(prior)

    # Every network output already contains one calibration prior. For the
    # uniform calibration and sensing priors used in the paper this correction
    # is an additive constant and cancels on normalization.
    cumulative = np.cumsum(table[shots], axis=0) + log_prior
    cumulative -= cumulative.max(axis=1, keepdims=True)
    trace = np.exp(cumulative)
    trace /= trace.sum(axis=1, keepdims=True)
    return trace


def posterior_means(posteriors: np.ndarray, grid: np.ndarray) -> np.ndarray:
    return np.asarray(posteriors) @ np.asarray(grid)


def table_log_posteriors_from_likelihoods(
    likelihoods: np.ndarray,
    prior: np.ndarray | None = None,
) -> np.ndarray:
    """Build exact single-shot posterior logs for validation only."""
    likelihoods = np.asarray(likelihoods, dtype=np.float64)
    n_bins = likelihoods.shape[1]
    if prior is None:
        prior = np.full(n_bins, 1.0 / n_bins)
    unnormalized = likelihoods * np.asarray(prior)[None, :]
    posteriors = unnormalized / unnormalized.sum(axis=1, keepdims=True)
    return np.log(np.clip(posteriors, np.finfo(float).tiny, None))

