"""Original five-input bounded MLP parameterization."""
from __future__ import annotations
from typing import Iterable
import torch
from torch import nn

class ResidualMLP(nn.Module):
    """A smooth 5-16-16-1 MLP with normalization and bounded output."""

    def __init__(
        self,
        input_mean: Iterable[float],
        input_std: Iterable[float],
        hidden_dims: tuple[int, int] = (16, 16),
        output_limit_mps2: float = 0.25,
    ) -> None:
        super().__init__()
        mean = torch.as_tensor(tuple(input_mean), dtype=torch.float64)
        std = torch.as_tensor(tuple(input_std), dtype=torch.float64)
        if mean.shape != (5,) or std.shape != (5,) or torch.any(std <= 0.0):
            raise ValueError("ResidualMLP requires five finite means and positive standard deviations.")
        if len(hidden_dims) != 2 or min(hidden_dims) <= 0:
            raise ValueError("ResidualMLP requires exactly two positive hidden dimensions.")
        self.register_buffer("input_mean", mean)
        self.register_buffer("input_std", std)
        self.output_limit_mps2 = float(output_limit_mps2)
        self.hidden_dims = tuple(int(dim) for dim in hidden_dims)
        self.layers = nn.Sequential(
            nn.Linear(5, self.hidden_dims[0], dtype=torch.float64),
            nn.Tanh(),
            nn.Linear(self.hidden_dims[0], self.hidden_dims[1], dtype=torch.float64),
            nn.Tanh(),
            nn.Linear(self.hidden_dims[1], 1, dtype=torch.float64),
        )
        final = self.layers[-1]
        assert isinstance(final, nn.Linear)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        normalized = (features - self.input_mean) / self.input_std
        return self.forward_normalized(normalized)

    def forward_normalized(self, normalized_features: torch.Tensor) -> torch.Tensor:
        raw = self.layers(normalized_features).squeeze(-1)
        return self.output_limit_mps2 * torch.tanh(raw)

    def normalized_residual(self, normalized_features: torch.Tensor) -> torch.Tensor:
        return self.forward_normalized(normalized_features) / self.output_limit_mps2


    def project(self):
        pass

    def regularization(self):
        return 1.0e-5 * sum(p.square().sum() for p in self.parameters())
