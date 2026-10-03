"""Deterministic full-batch training for posterior networks."""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .models import PosteriorNetwork


@dataclass(frozen=True)
class TrainingConfig:
    learning_rate: float = 1e-3
    max_epochs: int = 5000
    patience: int = 500
    min_delta: float = 1e-9
    target_kl: float = 1e-6
    seed: int = 0
    device: str = "auto"
    log_every: int = 0


@dataclass(frozen=True)
class TrainingResult:
    best_loss: float
    best_epoch: int
    epochs_run: int
    converged: bool
    history: tuple[float, ...]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def grouped_kl_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Weighted KL whose minimizer equals snapshot-level cross entropy."""
    per_pattern = F.kl_div(
        F.log_softmax(logits, dim=1), targets, reduction="none"
    ).sum(dim=1)
    normalized_weights = weights / weights.sum()
    return torch.sum(normalized_weights * per_pattern)


def train_network(
    model: PosteriorNetwork,
    patterns: np.ndarray,
    targets: np.ndarray,
    weights: np.ndarray,
    config: TrainingConfig,
) -> TrainingResult:
    seed_everything(config.seed)
    device = resolve_device(config.device)
    model.to(device)

    x = torch.as_tensor(patterns, dtype=torch.float32, device=device)
    y = torch.as_tensor(targets, dtype=torch.float32, device=device)
    w = torch.as_tensor(weights, dtype=torch.float32, device=device)
    if x.shape[0] < 2 and model.config.batch_norm:
        raise ValueError("batch-normalized training requires at least two patterns")

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    best_loss = float("inf")
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    stale_epochs = 0
    history: list[float] = []

    model.train()
    for epoch in range(1, config.max_epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        loss = grouped_kl_loss(model(x), y, w)
        value = float(loss.detach().cpu())
        history.append(value)
        if value < best_loss - config.min_delta:
            best_loss = value
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1

        loss.backward()
        optimizer.step()

        if config.log_every and epoch % config.log_every == 0:
            print(f"epoch={epoch} kl={value:.8e} best={best_loss:.8e}")
        if best_loss <= config.target_kl or stale_epochs >= config.patience:
            break

    model.load_state_dict(best_state)
    model.to("cpu")
    return TrainingResult(
        best_loss=best_loss,
        best_epoch=best_epoch,
        epochs_run=epoch,
        converged=best_loss <= config.target_kl,
        history=tuple(history),
    )
