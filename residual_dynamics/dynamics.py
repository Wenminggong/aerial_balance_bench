"""Differentiable nominal and residual ball dynamics."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .model import ResidualMLP


@dataclass(frozen=True)
class TorchPhysicalConstants:
    plank_length_m: torch.Tensor
    ball_position_offset_m: torch.Tensor
    gravity_mps2: torch.Tensor
    acceleration_gain: torch.Tensor

    def to(self, device: torch.device) -> "TorchPhysicalConstants":
        return TorchPhysicalConstants(
            plank_length_m=self.plank_length_m.to(device=device, dtype=torch.float64),
            ball_position_offset_m=self.ball_position_offset_m.to(
                device=device, dtype=torch.float64
            ),
            gravity_mps2=self.gravity_mps2.to(device=device, dtype=torch.float64),
            acceleration_gain=self.acceleration_gain.to(device=device, dtype=torch.float64),
        )

    def repeat_interleave(self, repeats: int) -> "TorchPhysicalConstants":
        return TorchPhysicalConstants(
            plank_length_m=self.plank_length_m.repeat_interleave(repeats),
            ball_position_offset_m=self.ball_position_offset_m.repeat_interleave(repeats),
            gravity_mps2=self.gravity_mps2.repeat_interleave(repeats),
            acceleration_gain=self.acceleration_gain.repeat_interleave(repeats),
        )


@dataclass(frozen=True)
class TorchGeometryConstants:
    plank_length_m: torch.Tensor
    rope_length_m: torch.Tensor

    def repeat_interleave(self, repeats: int) -> "TorchGeometryConstants":
        return TorchGeometryConstants(
            plank_length_m=self.plank_length_m.repeat_interleave(repeats),
            rope_length_m=self.rope_length_m.repeat_interleave(repeats),
        )


@dataclass(frozen=True)
class TorchResponseConstants:
    enabled: torch.Tensor
    tau_s: torch.Tensor
    gain: torch.Tensor
    bias: torch.Tensor
    max_velocity_mps: torch.Tensor

    def repeat_interleave(self, repeats: int) -> "TorchResponseConstants":
        return TorchResponseConstants(
            enabled=self.enabled.repeat_interleave(repeats),
            tau_s=self.tau_s.repeat_interleave(repeats),
            gain=self.gain.repeat_interleave(repeats),
            bias=self.bias.repeat_interleave(repeats),
            max_velocity_mps=self.max_velocity_mps.repeat_interleave(repeats),
        )


@dataclass(frozen=True)
class CausalRollout:
    states: torch.Tensor
    stage_states: torch.Tensor


def nominal_acceleration_torch(
    pb: torch.Tensor,
    theta: torch.Tensor,
    omega: torch.Tensor,
    constants: TorchPhysicalConstants,
) -> torch.Tensor:
    return constants.acceleration_gain * (
        (pb + constants.ball_position_offset_m - constants.plank_length_m) * omega.square()
        - constants.gravity_mps2 * torch.sin(theta)
    )


def acceleration_torch(
    model: ResidualMLP | None,
    pb: torch.Tensor,
    vb: torch.Tensor,
    theta: torch.Tensor,
    omega: torch.Tensor,
    constants: TorchPhysicalConstants,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    a_nom = nominal_acceleration_torch(pb, theta, omega, constants)
    if model is None:
        residual = torch.zeros_like(a_nom)
    else:
        features = torch.stack((pb, vb, theta, omega, a_nom), dim=-1)
        residual = model(features)
    return a_nom + residual, a_nom, residual


def geometry_torch(
    theta: torch.Tensor,
    vrz: torch.Tensor,
    geometry: TorchGeometryConstants,
    epsilon: float = 1.0e-9,
) -> tuple[torch.Tensor, torch.Tensor]:
    sin_beta = (geometry.plank_length_m / geometry.rope_length_m) * (
        1.0 - torch.cos(theta)
    )
    sin_beta = torch.clamp(sin_beta, -1.0 + epsilon, 1.0 - epsilon)
    beta = torch.asin(sin_beta)
    denominator = geometry.plank_length_m * torch.cos(beta - theta)
    sign = torch.where(denominator >= 0.0, 1.0, -1.0)
    denominator = torch.where(
        denominator.abs() < epsilon, sign * epsilon, denominator
    )
    omega = -vrz * torch.cos(beta) / denominator
    return beta, omega


def causal_residual_torch(
    model: ResidualMLP | None,
    state: torch.Tensor,
    constants: TorchPhysicalConstants,
    geometry: TorchGeometryConstants,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    features, a_nom, omega = causal_features_torch(state, constants, geometry)
    residual = torch.zeros_like(a_nom) if model is None else model(features)
    return a_nom + residual, a_nom, residual, omega


def causal_features_torch(
    state: torch.Tensor,
    constants: TorchPhysicalConstants,
    geometry: TorchGeometryConstants,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if state.shape[-1] != 4:
        raise ValueError("causal state must end with [pb, vb, theta, vrz]")
    pb, vb, theta, vrz = state.unbind(dim=-1)
    _, omega = geometry_torch(theta, vrz, geometry)
    a_nom = nominal_acceleration_torch(pb, theta, omega, constants)
    features = torch.stack((pb, vb, theta, omega, a_nom), dim=-1)
    return features, a_nom, omega


def response_stages_torch(
    previous_vrz: torch.Tensor,
    command: torch.Tensor,
    dt: torch.Tensor,
    response: TorchResponseConstants,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    modeled_target = response.gain * command + response.bias
    target = torch.where(response.enabled, modeled_target, command)
    safe_tau = torch.clamp(response.tau_s, min=1.0e-12)
    half_decay = torch.exp(-0.5 * dt / safe_tau)
    full_decay = torch.exp(-dt / safe_tau)
    dynamic_response = response.enabled & (response.tau_s > 0.0)
    start = torch.where(dynamic_response, previous_vrz, target)
    half = torch.where(
        dynamic_response,
        target + (previous_vrz - target) * half_decay,
        target,
    )
    end = torch.where(
        dynamic_response,
        target + (previous_vrz - target) * full_decay,
        target,
    )
    limit = response.max_velocity_mps
    return tuple(
        torch.where(limit > 0.0, torch.clamp(value, -limit, limit), value)
        for value in (start, half, end)
    )


def _causal_derivatives(
    model: ResidualMLP | None,
    physical: torch.Tensor,
    vrz: torch.Tensor,
    constants: TorchPhysicalConstants,
    geometry: TorchGeometryConstants,
) -> torch.Tensor:
    state = torch.cat((physical, vrz.unsqueeze(-1)), dim=-1)
    acceleration, _, _, omega = causal_residual_torch(model, state, constants, geometry)
    return torch.stack((physical[:, 1], acceleration, omega), dim=-1)


def causal_rk4_step(
    model: ResidualMLP | None,
    state: torch.Tensor,
    command: torch.Tensor,
    dt: torch.Tensor,
    constants: TorchPhysicalConstants,
    geometry: TorchGeometryConstants,
    response: TorchResponseConstants,
) -> tuple[torch.Tensor, torch.Tensor]:
    if state.ndim != 2 or state.shape[1] != 4:
        raise ValueError("state must have shape [batch, 4]")
    physical, previous_vrz = state[:, :3], state[:, 3]
    vrz_start, vrz_half, vrz_end = response_stages_torch(
        previous_vrz, command, dt, response
    )
    half_dt = 0.5 * dt.unsqueeze(-1)
    full_dt = dt.unsqueeze(-1)
    k1_state = torch.cat((physical, vrz_start.unsqueeze(-1)), dim=-1)
    k1 = _causal_derivatives(model, physical, vrz_start, constants, geometry)
    p2 = physical + half_dt * k1
    k2_state = torch.cat((p2, vrz_half.unsqueeze(-1)), dim=-1)
    k2 = _causal_derivatives(model, p2, vrz_half, constants, geometry)
    p3 = physical + half_dt * k2
    k3_state = torch.cat((p3, vrz_half.unsqueeze(-1)), dim=-1)
    k3 = _causal_derivatives(model, p3, vrz_half, constants, geometry)
    p4 = physical + full_dt * k3
    k4_state = torch.cat((p4, vrz_end.unsqueeze(-1)), dim=-1)
    k4 = _causal_derivatives(model, p4, vrz_end, constants, geometry)
    next_physical = physical + (full_dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    next_state = torch.cat((next_physical, vrz_end.unsqueeze(-1)), dim=-1)
    stages = torch.stack((k1_state, k2_state, k3_state, k4_state), dim=1)
    return next_state, stages


def rollout_causal_dynamics(
    model: ResidualMLP | None,
    initial_state: torch.Tensor,
    commands: torch.Tensor,
    dt: torch.Tensor,
    constants: TorchPhysicalConstants,
    geometry: TorchGeometryConstants,
    response: TorchResponseConstants,
) -> CausalRollout:
    if commands.ndim != 2 or dt.shape != commands.shape:
        raise ValueError("commands and dt must have shape [batch, horizon]")
    state = initial_state
    states = [state]
    stages = []
    for step in range(commands.shape[1]):
        state, step_stages = causal_rk4_step(
            model,
            state,
            commands[:, step],
            dt[:, step],
            constants,
            geometry,
            response,
        )
        states.append(state)
        stages.append(step_stages)
    return CausalRollout(
        states=torch.stack(states, dim=1),
        stage_states=torch.stack(stages, dim=1),
    )


def rk4_step(
    model: ResidualMLP | None,
    pb: torch.Tensor,
    vb: torch.Tensor,
    theta_0: torch.Tensor,
    theta_1: torch.Tensor,
    omega_0: torch.Tensor,
    omega_1: torch.Tensor,
    dt: torch.Tensor,
    constants: TorchPhysicalConstants,
) -> tuple[torch.Tensor, torch.Tensor]:
    theta_half = 0.5 * (theta_0 + theta_1)
    omega_half = 0.5 * (omega_0 + omega_1)

    a1, _, _ = acceleration_torch(model, pb, vb, theta_0, omega_0, constants)
    k1_p, k1_v = vb, a1

    p2 = pb + 0.5 * dt * k1_p
    v2 = vb + 0.5 * dt * k1_v
    a2, _, _ = acceleration_torch(model, p2, v2, theta_half, omega_half, constants)
    k2_p, k2_v = v2, a2

    p3 = pb + 0.5 * dt * k2_p
    v3 = vb + 0.5 * dt * k2_v
    a3, _, _ = acceleration_torch(model, p3, v3, theta_half, omega_half, constants)
    k3_p, k3_v = v3, a3

    p4 = pb + dt * k3_p
    v4 = vb + dt * k3_v
    a4, _, _ = acceleration_torch(model, p4, v4, theta_1, omega_1, constants)
    k4_p, k4_v = v4, a4

    pb_next = pb + (dt / 6.0) * (k1_p + 2.0 * k2_p + 2.0 * k3_p + k4_p)
    vb_next = vb + (dt / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v)
    return pb_next, vb_next


def rollout_ball_dynamics(
    model: ResidualMLP | None,
    pb_0: torch.Tensor,
    vb_0: torch.Tensor,
    theta: torch.Tensor,
    omega: torch.Tensor,
    dt: torch.Tensor,
    constants: TorchPhysicalConstants,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Roll out batched states; theta/omega have shape [batch, horizon + 1]."""

    if theta.ndim != 2 or omega.shape != theta.shape:
        raise ValueError("theta and omega must have shape [batch, horizon + 1].")
    if dt.shape != (theta.shape[0], theta.shape[1] - 1):
        raise ValueError("dt must have shape [batch, horizon].")
    pb_values = [pb_0]
    vb_values = [vb_0]
    pb, vb = pb_0, vb_0
    for step in range(dt.shape[1]):
        pb, vb = rk4_step(
            model,
            pb,
            vb,
            theta[:, step],
            theta[:, step + 1],
            omega[:, step],
            omega[:, step + 1],
            dt[:, step],
            constants,
        )
        pb_values.append(pb)
        vb_values.append(vb)
    return torch.stack(pb_values, dim=1), torch.stack(vb_values, dim=1)
