"""Small bounded five-input MLP used for acceleration residual prediction."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch
from torch import nn

from .config import ModelConfig


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


def create_model(
    config: ModelConfig,
    normalization: Mapping[str, Any],
) -> ResidualMLP:
    return ResidualMLP(
        input_mean=normalization["mean"],
        input_std=normalization["std"],
        hidden_dims=(config.hidden_dims[0], config.hidden_dims[1]),
        output_limit_mps2=config.output_limit_mps2,
    )


def save_portable_weights(
    model: ResidualMLP,
    path: Path,
    metadata: Mapping[str, Any],
) -> None:
    """Export model arrays in an implementation-neutral NumPy archive."""

    linear_layers = [layer for layer in model.layers if isinstance(layer, nn.Linear)]
    arrays: dict[str, np.ndarray] = {
        "input_mean": model.input_mean.detach().cpu().numpy(),
        "input_std": model.input_std.detach().cpu().numpy(),
        "output_limit_mps2": np.asarray(model.output_limit_mps2, dtype=np.float64),
        "metadata_json": np.asarray(json.dumps(dict(metadata), sort_keys=True)),
    }
    for index, layer in enumerate(linear_layers):
        arrays[f"layer_{index}_weight"] = layer.weight.detach().cpu().numpy()
        arrays[f"layer_{index}_bias"] = layer.bias.detach().cpu().numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)


def load_portable_weights(path: Path) -> tuple[ResidualMLP, dict[str, Any]]:
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata_json"].item()))
        hidden_dims = (
            int(archive["layer_0_weight"].shape[0]),
            int(archive["layer_1_weight"].shape[0]),
        )
        model = ResidualMLP(
            archive["input_mean"],
            archive["input_std"],
            hidden_dims=hidden_dims,
            output_limit_mps2=float(archive["output_limit_mps2"]),
        )
        linear_layers = [layer for layer in model.layers if isinstance(layer, nn.Linear)]
        with torch.no_grad():
            for index, layer in enumerate(linear_layers):
                layer.weight.copy_(torch.from_numpy(archive[f"layer_{index}_weight"]))
                layer.bias.copy_(torch.from_numpy(archive[f"layer_{index}_bias"]))
    return model, metadata


def load_checkpoint(path: Path, map_location: str | torch.device = "cpu") -> tuple[ResidualMLP, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    model_info = checkpoint["model"]
    model = ResidualMLP(
        model_info["input_mean"],
        model_info["input_std"],
        hidden_dims=tuple(model_info["hidden_dims"]),
        output_limit_mps2=float(model_info["output_limit_mps2"]),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device=map_location, dtype=torch.float64)
    return model, checkpoint


def checkpoint_model_metadata(model: ResidualMLP) -> dict[str, Any]:
    return {
        "input_mean": model.input_mean.detach().cpu().tolist(),
        "input_std": model.input_std.detach().cpu().tolist(),
        "hidden_dims": list(model.hidden_dims),
        "output_limit_mps2": model.output_limit_mps2,
        "activation": "tanh",
        "input_names": ["pb", "vb", "theta", "omega", "a_nom"],
        "dtype": "float64",
    }
