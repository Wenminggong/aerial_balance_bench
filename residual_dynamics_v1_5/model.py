"""Dynamic residual MLP and differentiable one-parameter smooth deadzone."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, NamedTuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .config import ModelConfig


class ModelOutput(NamedTuple):
    mlp_residual: torch.Tensor
    drive_acceleration: torch.Tensor
    sliding_weight: torch.Tensor
    deadzone_acceleration: torch.Tensor
    deadzone_suppression: torch.Tensor
    total_acceleration: torch.Tensor
    total_residual: torch.Tensor


class DynamicResidualMLP(nn.Module):
    def __init__(self, input_mean: Iterable[float], input_std: Iterable[float],
                 hidden_dims: tuple[int, int] = (16, 16),
                 output_limit_mps2: float = 0.20) -> None:
        super().__init__()
        mean = torch.as_tensor(tuple(input_mean), dtype=torch.float64)
        std = torch.as_tensor(tuple(input_std), dtype=torch.float64)
        if mean.shape != (5,) or std.shape != (5,) or not torch.isfinite(mean).all() or torch.any(std <= 0):
            raise ValueError("DynamicResidualMLP requires five finite means and positive stds")
        self.register_buffer("input_mean", mean)
        self.register_buffer("input_std", std)
        self.hidden_dims = tuple(int(value) for value in hidden_dims)
        self.output_limit_mps2 = float(output_limit_mps2)
        self.layers = nn.Sequential(
            nn.Linear(5, hidden_dims[0], dtype=torch.float64), nn.Tanh(),
            nn.Linear(hidden_dims[0], hidden_dims[1], dtype=torch.float64), nn.Tanh(),
            nn.Linear(hidden_dims[1], 1, dtype=torch.float64),
        )
        final = self.layers[-1]
        assert isinstance(final, nn.Linear)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.forward_normalized((features - self.input_mean) / self.input_std)

    def forward_normalized(self, features: torch.Tensor) -> torch.Tensor:
        return self.output_limit_mps2 * torch.tanh(self.layers(features).squeeze(-1))

    def normalized_residual(self, features: torch.Tensor) -> torch.Tensor:
        return self.forward_normalized(features) / self.output_limit_mps2


class SmoothDeadzoneResidualModel(nn.Module):
    def __init__(self, mlp: DynamicResidualMLP, config: ModelConfig) -> None:
        super().__init__()
        self.mlp = mlp
        self.deadzone_min_mps2 = float(config.deadzone_min_mps2)
        self.deadzone_max_mps2 = float(config.deadzone_max_mps2)
        self.deadzone_smoothing_mps2 = float(config.deadzone_smoothing_mps2)
        self.velocity_switch_mps = float(config.velocity_switch_mps)
        fraction = ((config.deadzone_initial_mps2 - config.deadzone_min_mps2)
                    / (config.deadzone_max_mps2 - config.deadzone_min_mps2))
        raw = math.log(fraction / (1.0 - fraction))
        self.raw_deadzone_threshold = nn.Parameter(torch.tensor(raw, dtype=torch.float64))

    @property
    def deadzone_threshold(self) -> torch.Tensor:
        return self.deadzone_min_mps2 + (
            self.deadzone_max_mps2 - self.deadzone_min_mps2
        ) * torch.sigmoid(self.raw_deadzone_threshold)

    def forward(self, features: torch.Tensor) -> ModelOutput:
        residual = self.mlp(features)
        a_nom = features[..., 4]
        velocity = features[..., 1]
        drive = a_nom + residual
        deadzone = smooth_deadzone(
            drive, self.deadzone_threshold, self.deadzone_smoothing_mps2
        )
        sliding_weight = 1.0 - torch.exp(-(velocity / self.velocity_switch_mps).square())
        total = (1.0 - sliding_weight) * deadzone + sliding_weight * drive
        return ModelOutput(
            residual, drive, sliding_weight, deadzone, drive - total,
            total, total - a_nom,
        )

    def set_mlp_trainable(self, trainable: bool) -> None:
        for parameter in self.mlp.parameters():
            parameter.requires_grad_(trainable)


def smooth_deadzone(drive: torch.Tensor, threshold: torch.Tensor,
                    smoothing_mps2: float) -> torch.Tensor:
    tau = float(smoothing_mps2)
    return tau * (
        F.softplus((drive - threshold) / tau)
        - F.softplus((-drive - threshold) / tau)
    )


def create_model(config: ModelConfig, normalization: Mapping[str, Any]) -> SmoothDeadzoneResidualModel:
    mlp = DynamicResidualMLP(normalization["mean"], normalization["std"],
                             config.hidden_dims, config.output_limit_mps2)
    return SmoothDeadzoneResidualModel(mlp, config)


def checkpoint_model_metadata(model: SmoothDeadzoneResidualModel) -> dict[str, Any]:
    return {
        "schema_version": 1, "model_version": "1.5.0",
        "input_names": ["pb", "vb", "theta", "omega", "a_nom"],
        "input_mean": model.mlp.input_mean.detach().cpu().tolist(),
        "input_std": model.mlp.input_std.detach().cpu().tolist(),
        "hidden_dims": list(model.mlp.hidden_dims),
        "output_limit_mps2": model.mlp.output_limit_mps2,
        "deadzone_min_mps2": model.deadzone_min_mps2,
        "deadzone_max_mps2": model.deadzone_max_mps2,
        "deadzone_threshold_mps2": float(model.deadzone_threshold.detach().cpu()),
        "deadzone_smoothing_mps2": model.deadzone_smoothing_mps2,
        "velocity_switch_mps": model.velocity_switch_mps,
        "activation": "tanh", "dtype": "float64",
        "formula": "(1-w)*D(a_nom+r)+w*(a_nom+r)",
    }


def save_portable_weights(model: SmoothDeadzoneResidualModel, path: Path,
                          metadata: Mapping[str, Any]) -> None:
    linear = [layer for layer in model.mlp.layers if isinstance(layer, nn.Linear)]
    arrays: dict[str, np.ndarray] = {
        "input_mean": model.mlp.input_mean.detach().cpu().numpy(),
        "input_std": model.mlp.input_std.detach().cpu().numpy(),
        "output_limit_mps2": np.asarray(model.mlp.output_limit_mps2),
        "deadzone_min_mps2": np.asarray(model.deadzone_min_mps2),
        "deadzone_max_mps2": np.asarray(model.deadzone_max_mps2),
        "deadzone_threshold_mps2": np.asarray(float(model.deadzone_threshold.detach().cpu())),
        "deadzone_smoothing_mps2": np.asarray(model.deadzone_smoothing_mps2),
        "velocity_switch_mps": np.asarray(model.velocity_switch_mps),
        "metadata_json": np.asarray(json.dumps(dict(metadata), sort_keys=True)),
    }
    for index, layer in enumerate(linear):
        arrays[f"layer_{index}_weight"] = layer.weight.detach().cpu().numpy()
        arrays[f"layer_{index}_bias"] = layer.bias.detach().cpu().numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)


def load_portable_weights(path: Path) -> tuple[SmoothDeadzoneResidualModel, dict[str, Any]]:
    with np.load(path, allow_pickle=False) as archive:
        config = ModelConfig(
            input_names=("pb", "vb", "theta", "omega", "a_nom"),
            hidden_dims=(int(archive["layer_0_weight"].shape[0]),
                         int(archive["layer_1_weight"].shape[0])),
            output_limit_mps2=float(archive["output_limit_mps2"]), dtype="float64",
            deadzone_min_mps2=float(archive["deadzone_min_mps2"]),
            deadzone_max_mps2=float(archive["deadzone_max_mps2"]),
            deadzone_initial_mps2=float(archive["deadzone_threshold_mps2"]),
            deadzone_smoothing_mps2=float(archive["deadzone_smoothing_mps2"]),
            velocity_switch_mps=float(archive["velocity_switch_mps"]),
        )
        model = create_model(config, {"mean": archive["input_mean"], "std": archive["input_std"]})
        linear = [layer for layer in model.mlp.layers if isinstance(layer, nn.Linear)]
        with torch.no_grad():
            for index, layer in enumerate(linear):
                layer.weight.copy_(torch.from_numpy(archive[f"layer_{index}_weight"]))
                layer.bias.copy_(torch.from_numpy(archive[f"layer_{index}_bias"]))
            fraction = ((float(archive["deadzone_threshold_mps2"]) - model.deadzone_min_mps2)
                        / (model.deadzone_max_mps2 - model.deadzone_min_mps2))
            model.raw_deadzone_threshold.copy_(torch.tensor(
                math.log(fraction / (1 - fraction)), dtype=torch.float64
            ))
        metadata = json.loads(str(archive["metadata_json"].item()))
    return model, metadata


def load_checkpoint(path: Path, map_location: str | torch.device = "cpu") -> tuple[SmoothDeadzoneResidualModel, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    info = checkpoint["model"]
    config = ModelConfig(
        tuple(info["input_names"]), tuple(info["hidden_dims"]), info["output_limit_mps2"], "float64",
        info["deadzone_min_mps2"], info["deadzone_max_mps2"], info["deadzone_threshold_mps2"],
        info["deadzone_smoothing_mps2"], info["velocity_switch_mps"],
    )
    model = create_model(config, {"mean": info["input_mean"], "std": info["input_std"]})
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device=map_location, dtype=torch.float64)
    return model, checkpoint
