"""Smooth bounded two-input Gaussian RBF residual model."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn

from .config import Config


MODEL_ID = "acceleration_residual_rbf2_pv_v1"


class ResidualRBF2D(nn.Module):
    """Normalized 9x9 Gaussian RBF with a train-only two-dimensional OOD gate."""

    def __init__(
        self,
        pb_centers_m: tuple[float, ...] | list[float],
        vb_centers_mps: tuple[float, ...] | list[float],
        pb_width_m: float,
        vb_width_mps: float,
        denominator_epsilon: float,
        output_limit_mps2: float,
        ood: Mapping[str, Any],
    ) -> None:
        super().__init__()
        pb = torch.as_tensor(pb_centers_m, dtype=torch.float64)
        vb = torch.as_tensor(vb_centers_mps, dtype=torch.float64)
        if pb.shape != (9,) or vb.shape != (9,):
            raise ValueError("ResidualRBF2D requires the approved 9x9 center grid.")
        grid_pb, grid_vb = torch.meshgrid(pb, vb, indexing="ij")
        self.register_buffer("centers", torch.stack((grid_pb.reshape(-1), grid_vb.reshape(-1)), -1))
        self.register_buffer("pb_centers", pb)
        self.register_buffer("vb_centers", vb)
        self.pb_width_m = float(pb_width_m)
        self.vb_width_mps = float(vb_width_mps)
        self.denominator_epsilon = float(denominator_epsilon)
        self.output_limit_mps2 = float(output_limit_mps2)
        self.coefficients = nn.Parameter(torch.zeros(81, dtype=torch.float64))
        for name in (
            "mean",
            "std",
            "lower_quantile",
            "upper_quantile",
            "inverse_correlation",
        ):
            self.register_buffer(f"ood_{name}", torch.as_tensor(ood[name], dtype=torch.float64))
        self.ood_mahalanobis_threshold = float(ood["mahalanobis_threshold"])
        self.ood_decay_width_std = float(ood["decay_width_std"])
        self.ood_min_weight = float(ood["min_weight"])

    def _pv(self, features: torch.Tensor) -> torch.Tensor:
        if features.shape[-1] < 2:
            raise ValueError("RBF input must end with [pb, vb].")
        return features[..., :2]

    def basis(self, features: torch.Tensor) -> torch.Tensor:
        pv = self._pv(features)
        delta_p = (pv[..., None, 0] - self.centers[:, 0]) / self.pb_width_m
        delta_v = (pv[..., None, 1] - self.centers[:, 1]) / self.vb_width_mps
        phi = torch.exp(-0.5 * (delta_p.square() + delta_v.square()))
        return phi / (self.denominator_epsilon + phi.sum(dim=-1, keepdim=True))

    def base_forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.sum(self.basis(features) * self.coefficients, dim=-1)

    def ood_components(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        pv = self._pv(features)
        below = (self.ood_lower_quantile - pv) / self.ood_std
        above = (pv - self.ood_upper_quantile) / self.ood_std
        marginal_components = torch.maximum(
            torch.zeros_like(pv), torch.maximum(below, above)
        )
        marginal = torch.linalg.vector_norm(marginal_components, dim=-1)
        z = (pv - self.ood_mean) / self.ood_std
        mahalanobis_sq = torch.einsum(
            "...i,ij,...j->...", z, self.ood_inverse_correlation, z
        ).clamp_min(0.0)
        mahalanobis = torch.sqrt(mahalanobis_sq + 1.0e-30)
        joint_excess = torch.relu(mahalanobis - self.ood_mahalanobis_threshold)
        distance = torch.sqrt(marginal.square() + joint_excess.square() + 1.0e-30)
        weight = self.ood_min_weight + (1.0 - self.ood_min_weight) * torch.exp(
            -0.5 * (distance / self.ood_decay_width_std).square()
        )
        return {
            "marginal": marginal,
            "mahalanobis": mahalanobis,
            "joint_excess": joint_excess,
            "distance": distance,
            "weight": weight,
        }

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.ood_components(features)["weight"] * self.base_forward(features)

    @torch.no_grad()
    def project_coefficients(
        self,
        reference: torch.Tensor | None = None,
        radius_mps2: float | None = None,
    ) -> None:
        lower = torch.full_like(self.coefficients, -self.output_limit_mps2)
        upper = torch.full_like(self.coefficients, self.output_limit_mps2)
        if reference is not None:
            if radius_mps2 is None:
                raise ValueError("A trust reference requires a radius.")
            lower = torch.maximum(lower, reference - float(radius_mps2))
            upper = torch.minimum(upper, reference + float(radius_mps2))
        self.coefficients.copy_(torch.minimum(upper, torch.maximum(lower, self.coefficients)))

    def coefficient_grid(self) -> torch.Tensor:
        return self.coefficients.reshape(9, 9)


def create_model(config: Config, manifest: Mapping[str, Any]) -> ResidualRBF2D:
    return ResidualRBF2D(
        config.rbf.pb_centers_m,
        config.rbf.vb_centers_mps,
        config.rbf.pb_width_m,
        config.rbf.vb_width_mps,
        config.rbf.denominator_epsilon,
        float(manifest["normalization"]["effective_cap_mps2"]),
        manifest["normalization"],
    )


def checkpoint_payload(
    model: ResidualRBF2D,
    config: Config,
    manifest: Mapping[str, Any],
    *,
    stage: str,
    epoch: int,
    validation: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "model_id": MODEL_ID,
        "state_dict": model.state_dict(),
        "model": model_metadata(model),
        "config_hash": config.config_hash,
        "dataset_hash": manifest["dataset_hash"],
        "stage": stage,
        "epoch": int(epoch),
        "validation": dict(validation),
    }


def model_metadata(model: ResidualRBF2D) -> dict[str, Any]:
    return {
        "model_id": MODEL_ID,
        "input_names": ["pb", "vb"],
        "pb_centers_m": model.pb_centers.detach().cpu().tolist(),
        "vb_centers_mps": model.vb_centers.detach().cpu().tolist(),
        "pb_width_m": model.pb_width_m,
        "vb_width_mps": model.vb_width_mps,
        "denominator_epsilon": model.denominator_epsilon,
        "output_limit_mps2": model.output_limit_mps2,
        "dtype": "float64",
        "ood": {
            "mean": model.ood_mean.detach().cpu().tolist(),
            "std": model.ood_std.detach().cpu().tolist(),
            "lower_quantile": model.ood_lower_quantile.detach().cpu().tolist(),
            "upper_quantile": model.ood_upper_quantile.detach().cpu().tolist(),
            "inverse_correlation": model.ood_inverse_correlation.detach().cpu().tolist(),
            "mahalanobis_threshold": model.ood_mahalanobis_threshold,
            "decay_width_std": model.ood_decay_width_std,
            "min_weight": model.ood_min_weight,
        },
    }


def load_checkpoint(path: Path, device: str | torch.device = "cpu") -> tuple[ResidualRBF2D, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    info = checkpoint["model"]
    model = ResidualRBF2D(
        info["pb_centers_m"],
        info["vb_centers_mps"],
        info["pb_width_m"],
        info["vb_width_mps"],
        info["denominator_epsilon"],
        info["output_limit_mps2"],
        info["ood"],
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device=device, dtype=torch.float64)
    return model, checkpoint


def numpy_predict(features: np.ndarray, payload: Mapping[str, Any]) -> np.ndarray:
    pv = np.asarray(features, dtype=np.float64)[..., :2]
    centers = np.asarray(payload["centers"], dtype=np.float64)
    delta_p = (pv[..., None, 0] - centers[:, 0]) / payload["widths"][0]
    delta_v = (pv[..., None, 1] - centers[:, 1]) / payload["widths"][1]
    phi = np.exp(-0.5 * (delta_p**2 + delta_v**2))
    alpha = phi / (payload["denominator_epsilon"] + phi.sum(axis=-1, keepdims=True))
    base = np.sum(alpha * np.asarray(payload["coefficients"]), axis=-1)
    ood = payload["ood"]
    mean, std = np.asarray(ood["mean"]), np.asarray(ood["std"])
    lower, upper = np.asarray(ood["lower_quantile"]), np.asarray(ood["upper_quantile"])
    marginal_components = np.maximum(0.0, np.maximum((lower - pv) / std, (pv - upper) / std))
    marginal = np.linalg.norm(marginal_components, axis=-1)
    z = (pv - mean) / std
    inverse = np.asarray(ood["inverse_correlation"])
    mahalanobis = np.sqrt(np.maximum(0.0, np.einsum("...i,ij,...j->...", z, inverse, z)))
    joint = np.maximum(0.0, mahalanobis - ood["mahalanobis_threshold"])
    distance = np.sqrt(marginal**2 + joint**2)
    weight = ood["min_weight"] + (1.0 - ood["min_weight"]) * np.exp(
        -0.5 * (distance / ood["decay_width_std"]) ** 2
    )
    return weight * base


def portable_payload(
    model: ResidualRBF2D,
    metadata: Mapping[str, Any],
    golden_features: np.ndarray,
) -> dict[str, Any]:
    payload = {
        "schema_version": 1,
        "version": "rbf2_pv_v1",
        "model_id": MODEL_ID,
        "input_names": ["pb", "vb"],
        "centers": model.centers.detach().cpu().tolist(),
        "widths": [model.pb_width_m, model.vb_width_mps],
        "denominator_epsilon": model.denominator_epsilon,
        "coefficients": model.coefficients.detach().cpu().tolist(),
        "output_limit_mps2": model.output_limit_mps2,
        "ood": model_metadata(model)["ood"],
        "metadata": dict(metadata),
    }
    payload["golden_examples"] = [
        {"input": row.tolist(), "residual_mps2": float(value)}
        for row, value in zip(golden_features, numpy_predict(golden_features, payload))
    ]
    return payload


def save_portable_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(dict(payload), stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
