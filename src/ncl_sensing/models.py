"""Neural estimators for single-shot parameter posteriors."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class NetworkConfig:
    input_size: int
    output_size: int
    hidden_size: int = 256
    hidden_blocks: int = 3
    batch_norm: bool = False


class DenseBlock(nn.Module):
    """Two affine layers matching the network used in the original scripts."""

    def __init__(self, hidden_size: int, batch_norm: bool = True):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(hidden_size, hidden_size)]
        if batch_norm:
            layers.append(nn.BatchNorm1d(hidden_size))
        layers.extend([nn.ReLU(), nn.Linear(hidden_size, hidden_size)])
        if batch_norm:
            layers.append(nn.BatchNorm1d(hidden_size))
        self.layers = nn.Sequential(*layers)
        self.activation = nn.ReLU()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.activation(self.layers(inputs))


class PosteriorNetwork(nn.Module):
    """Map a binary spatial pattern to logits over parameter bins."""

    def __init__(self, config: NetworkConfig):
        super().__init__()
        self.config = config
        input_layers: list[nn.Module] = [
            nn.Linear(config.input_size, config.hidden_size)
        ]
        if config.batch_norm:
            input_layers.append(nn.BatchNorm1d(config.hidden_size))
        input_layers.append(nn.ReLU())
        self.input_layer = nn.Sequential(*input_layers)
        self.hidden_layers = nn.Sequential(
            *[
                DenseBlock(config.hidden_size, config.batch_norm)
                for _ in range(config.hidden_blocks)
            ]
        )
        self.output_layer = nn.Linear(config.hidden_size, config.output_size)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.input_layer(inputs)
        hidden = self.hidden_layers(hidden)
        return self.output_layer(hidden)

    def checkpoint(self, metadata: dict | None = None) -> dict:
        return {
            "network_config": asdict(self.config),
            "state_dict": self.state_dict(),
            "metadata": metadata or {},
        }


def network_from_checkpoint(checkpoint: dict) -> PosteriorNetwork:
    config = NetworkConfig(**checkpoint["network_config"])
    model = PosteriorNetwork(config)
    model.load_state_dict(checkpoint["state_dict"])
    return model
