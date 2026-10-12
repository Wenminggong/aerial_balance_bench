"""Differentiable nominal, free-drive, and composite v1.5 ball dynamics."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .model import ModelOutput, SmoothDeadzoneResidualModel


@dataclass(frozen=True)
class TorchPhysicalConstants:
    plank_length_m: torch.Tensor
    ball_position_offset_m: torch.Tensor
    gravity_mps2: torch.Tensor
    acceleration_gain: torch.Tensor

    def to(self, device: torch.device) -> "TorchPhysicalConstants":
        return TorchPhysicalConstants(*[
            value.to(device=device, dtype=torch.float64)
            for value in (self.plank_length_m, self.ball_position_offset_m,
                          self.gravity_mps2, self.acceleration_gain)
        ])


def nominal_acceleration_torch(pb: torch.Tensor, theta: torch.Tensor, omega: torch.Tensor,
                               constants: TorchPhysicalConstants) -> torch.Tensor:
    return constants.acceleration_gain * (
        (pb + constants.ball_position_offset_m - constants.plank_length_m) * omega.square()
        - constants.gravity_mps2 * torch.sin(theta)
    )


def acceleration_torch(model: SmoothDeadzoneResidualModel | None, pb: torch.Tensor,
                       vb: torch.Tensor, theta: torch.Tensor, omega: torch.Tensor,
                       constants: TorchPhysicalConstants,
                       mode: str = "composite") -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    a_nom = nominal_acceleration_torch(pb, theta, omega, constants)
    if model is None:
        return a_nom, a_nom, torch.zeros_like(a_nom)
    output = model(torch.stack((pb, vb, theta, omega, a_nom), dim=-1))
    acceleration = output.drive_acceleration if mode == "dynamic" else output.total_acceleration
    return acceleration, a_nom, acceleration - a_nom


def rk4_step(model: SmoothDeadzoneResidualModel | None, pb: torch.Tensor, vb: torch.Tensor,
             theta_0: torch.Tensor, theta_1: torch.Tensor, omega_0: torch.Tensor,
             omega_1: torch.Tensor, dt: torch.Tensor, constants: TorchPhysicalConstants,
             mode: str = "composite") -> tuple[torch.Tensor, torch.Tensor]:
    theta_half, omega_half = 0.5 * (theta_0 + theta_1), 0.5 * (omega_0 + omega_1)
    a1, _, _ = acceleration_torch(model, pb, vb, theta_0, omega_0, constants, mode)
    p2, v2 = pb + 0.5 * dt * vb, vb + 0.5 * dt * a1
    a2, _, _ = acceleration_torch(model, p2, v2, theta_half, omega_half, constants, mode)
    p3, v3 = pb + 0.5 * dt * v2, vb + 0.5 * dt * a2
    a3, _, _ = acceleration_torch(model, p3, v3, theta_half, omega_half, constants, mode)
    p4, v4 = pb + dt * v3, vb + dt * a3
    a4, _, _ = acceleration_torch(model, p4, v4, theta_1, omega_1, constants, mode)
    return (
        pb + dt * (vb + 2 * v2 + 2 * v3 + v4) / 6,
        vb + dt * (a1 + 2 * a2 + 2 * a3 + a4) / 6,
    )


def rollout_ball_dynamics(model: SmoothDeadzoneResidualModel | None, pb_0: torch.Tensor,
                          vb_0: torch.Tensor, theta: torch.Tensor, omega: torch.Tensor,
                          dt: torch.Tensor, constants: TorchPhysicalConstants,
                          mode: str = "composite") -> tuple[torch.Tensor, torch.Tensor]:
    if theta.ndim != 2 or omega.shape != theta.shape or dt.shape != (theta.shape[0], theta.shape[1] - 1):
        raise ValueError("theta/omega must be [batch,horizon+1] and dt [batch,horizon]")
    pb_values, vb_values = [pb_0], [vb_0]
    pb, vb = pb_0, vb_0
    for step in range(dt.shape[1]):
        pb, vb = rk4_step(model, pb, vb, theta[:, step], theta[:, step + 1],
                          omega[:, step], omega[:, step + 1], dt[:, step], constants, mode)
        pb_values.append(pb); vb_values.append(vb)
    return torch.stack(pb_values, dim=1), torch.stack(vb_values, dim=1)
