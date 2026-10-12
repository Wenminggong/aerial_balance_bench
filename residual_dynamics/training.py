"""Training loop, balanced window sampling, and runtime device selection."""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import os
import random
import resource
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

# Required by PyTorch deterministic algorithms for CUDA/cuBLAS >= 10.2.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import torch.nn.functional as F

from .config import AppConfig
from .data import (
    PhysicalConstants,
    TrialData,
    load_manifest,
    load_processed_trials,
    position_bin_indices,
)
from .dynamics import (
    TorchGeometryConstants,
    TorchPhysicalConstants,
    TorchResponseConstants,
    causal_residual_torch,
    rollout_ball_dynamics,
    rollout_causal_dynamics,
)
from .model import (
    ResidualMLP,
    checkpoint_model_metadata,
    create_model,
    load_checkpoint,
    save_portable_weights,
)
from .wandb_monitor import WandbMonitor


@dataclass(frozen=True)
class WindowReference:
    trial_index: int
    start: int
    regime: int
    position_bin: int = -1


@dataclass
class WindowBatch:
    features: torch.Tensor
    residual_target: torch.Tensor
    pb: torch.Tensor
    vb: torch.Tensor
    theta: torch.Tensor
    omega: torch.Tensor
    dt: torch.Tensor
    constants: TorchPhysicalConstants
    causal_state: torch.Tensor | None = None
    smooth_pb: torch.Tensor | None = None
    smooth_vb: torch.Tensor | None = None
    commands: torch.Tensor | None = None
    geometry: TorchGeometryConstants | None = None
    response: TorchResponseConstants | None = None
    state_scales: torch.Tensor | None = None


class WindowStore:
    """Keep trial sequences resident on the selected training device."""

    def __init__(
        self,
        trials: Sequence[TrialData],
        horizon: int,
        config: AppConfig | None = None,
        normalization: dict[str, Any] | None = None,
    ) -> None:
        self.trials = list(trials)
        self.horizon = int(horizon)
        self.config = config
        self.causal = bool(config is not None and config.causal_rollout.enabled)
        self.state_scales = None
        if self.causal:
            if normalization is None or "state_scales" not in normalization:
                raise ValueError("Causal WindowStore requires manifest state_scales.")
            self.state_scales = np.asarray(normalization["state_scales"], dtype=np.float64)
            if self.state_scales.shape != (4,) or not np.all(self.state_scales > 0.0):
                raise ValueError("state_scales must contain four positive values.")
        self.references = _build_window_references(
            self.trials, self.horizon, stride=1, config=config
        )
        if not self.references:
            raise RuntimeError("No valid rollout windows were found.")
        self.by_trial_regime: dict[int, dict[int, list[WindowReference]]] = {}
        for ref in self.references:
            self.by_trial_regime.setdefault(ref.trial_index, {}).setdefault(ref.regime, []).append(ref)
        self.by_coverage: dict[
            int, dict[int, dict[str, dict[str, list[WindowReference]]]]
        ] = {}
        for ref in self.references:
            if ref.position_bin < 0:
                continue
            trial = self.trials[ref.trial_index]
            self.by_coverage.setdefault(ref.position_bin, {}).setdefault(
                ref.regime, {}
            ).setdefault(trial.session, {}).setdefault(trial.group, []).append(ref)
        self._cache: dict[str, list[dict[str, torch.Tensor]]] = {}

    def tensors(self, device: torch.device) -> list[dict[str, torch.Tensor]]:
        key = str(device)
        if key not in self._cache:
            names = ["features", "residual_target", "pb", "vb", "theta", "omega", "dt_s"]
            if self.causal:
                names.extend(
                    (
                        "causal_state",
                        "smooth_pb",
                        "smooth_vb",
                        "command_z",
                        "pending_commands",
                        "response_command",
                    )
                )
            self._cache[key] = [
                {
                    name: torch.as_tensor(
                        trial.arrays[name], device=device, dtype=torch.float64
                    )
                    for name in names
                }
                for trial in self.trials
            ]
        return self._cache[key]

    def balanced_references(
        self,
        rng: np.random.Generator,
        batch_size: int,
    ) -> list[WindowReference]:
        if (
            self.config is not None
            and self.config.data.split_assignments
            and self.config.data.empirical_sampling_fraction < 1.0
        ):
            return self._mixed_references(rng, batch_size)
        trial_ids = tuple(sorted(self.by_trial_regime))
        result: list[WindowReference] = []
        for _ in range(batch_size):
            trial_index = int(trial_ids[int(rng.integers(0, len(trial_ids)))])
            regimes = tuple(sorted(self.by_trial_regime[trial_index]))
            regime = int(regimes[int(rng.integers(0, len(regimes)))])
            candidates = self.by_trial_regime[trial_index][regime]
            result.append(candidates[int(rng.integers(0, len(candidates)))])
        return result

    def _mixed_references(
        self,
        rng: np.random.Generator,
        batch_size: int,
    ) -> list[WindowReference]:
        assert self.config is not None
        empirical_count = int(round(batch_size * self.config.data.empirical_sampling_fraction))
        empirical_count = min(batch_size, max(0, empirical_count))
        result = [
            self.references[int(rng.integers(0, len(self.references)))]
            for _ in range(empirical_count)
        ]
        position_bins = tuple(sorted(self.by_coverage))
        if not position_bins and empirical_count < batch_size:
            raise RuntimeError("Coverage stream has no in-range position windows.")
        for _ in range(batch_size - empirical_count):
            position_bin = position_bins[int(rng.integers(0, len(position_bins)))]
            by_regime = self.by_coverage[position_bin]
            regimes = tuple(sorted(by_regime))
            regime = regimes[int(rng.integers(0, len(regimes)))]
            by_session = by_regime[regime]
            sessions = tuple(sorted(by_session))
            session = sessions[int(rng.integers(0, len(sessions)))]
            by_group = by_session[session]
            groups = tuple(sorted(by_group))
            group = groups[int(rng.integers(0, len(groups)))]
            candidates = by_group[group]
            weights = np.asarray(
                [
                    self.config.data.failure_sampling_weight
                    if self.trials[ref.trial_index].is_failure_prefix
                    else 1.0
                    for ref in candidates
                ],
                dtype=np.float64,
            )
            weights /= weights.sum()
            result.append(candidates[int(rng.choice(len(candidates), p=weights))])
        rng.shuffle(result)
        return result

    def make_batch(
        self,
        references: Sequence[WindowReference],
        device: torch.device,
    ) -> WindowBatch:
        tensors = self.tensors(device)
        horizon = self.horizon

        def stack(name: str, include_endpoint: bool = True) -> torch.Tensor:
            length = horizon + 1 if include_endpoint else horizon
            return torch.stack(
                [
                    tensors[ref.trial_index][name][ref.start : ref.start + length]
                    for ref in references
                ],
                dim=0,
            )

        constants = [self.trials[ref.trial_index].constants for ref in references]
        torch_constants = TorchPhysicalConstants(
            plank_length_m=torch.tensor(
                [item.plank_length_m for item in constants], device=device, dtype=torch.float64
            ),
            ball_position_offset_m=torch.tensor(
                [item.ball_position_offset_m for item in constants],
                device=device,
                dtype=torch.float64,
            ),
            gravity_mps2=torch.tensor(
                [item.gravity_mps2 for item in constants], device=device, dtype=torch.float64
            ),
            acceleration_gain=torch.tensor(
                [item.acceleration_gain for item in constants],
                device=device,
                dtype=torch.float64,
            ),
        )
        batch = WindowBatch(
            features=stack("features"),
            residual_target=stack("residual_target"),
            pb=stack("pb"),
            vb=stack("vb"),
            theta=stack("theta"),
            omega=stack("omega"),
            dt=stack("dt_s", include_endpoint=False),
            constants=torch_constants,
        )
        if not self.causal:
            return batch

        pending = stack("pending_commands")[:, 0, :]
        delay_steps = pending.shape[1]
        committed = stack("command_z")
        commands = reconstruct_deployment_commands(pending, committed, horizon)
        expected_commands = stack("response_command", include_endpoint=False)
        if not torch.allclose(commands, expected_commands, rtol=0.0, atol=1.0e-9):
            error = torch.max(torch.abs(commands - expected_commands)).item()
            raise RuntimeError(
                f"Pending-command reconstruction disagrees with logged queue (max={error:.3e})."
            )
        responses = [self.trials[ref.trial_index].response for ref in references]
        if any(item is None for item in responses):
            raise RuntimeError("Causal rollout trial is missing response metadata.")
        response_items = [item for item in responses if item is not None]
        if any(item.delay_steps != delay_steps for item in response_items):
            raise RuntimeError("Window pending queue length disagrees with response metadata.")
        batch.causal_state = stack("causal_state")
        batch.smooth_pb = stack("smooth_pb")
        batch.smooth_vb = stack("smooth_vb")
        batch.commands = commands
        batch.geometry = TorchGeometryConstants(
            plank_length_m=torch.tensor(
                [item.plank_length_m for item in constants], device=device, dtype=torch.float64
            ),
            rope_length_m=torch.tensor(
                [item.rope_length_m for item in constants], device=device, dtype=torch.float64
            ),
        )
        batch.response = TorchResponseConstants(
            enabled=torch.tensor(
                [item.enabled for item in response_items], device=device, dtype=torch.bool
            ),
            tau_s=torch.tensor(
                [item.tau_s for item in response_items], device=device, dtype=torch.float64
            ),
            gain=torch.tensor(
                [item.gain for item in response_items], device=device, dtype=torch.float64
            ),
            bias=torch.tensor(
                [item.bias for item in response_items], device=device, dtype=torch.float64
            ),
            max_velocity_mps=torch.tensor(
                [item.max_velocity_mps for item in response_items],
                device=device,
                dtype=torch.float64,
            ),
        )
        batch.state_scales = torch.as_tensor(
            self.state_scales, device=device, dtype=torch.float64
        )
        return batch


def reconstruct_deployment_commands(
    initial_pending: torch.Tensor,
    committed: torch.Tensor,
    horizon: int,
) -> torch.Tensor:
    """Consume the initial ring buffer, then future post-window commits."""

    if initial_pending.ndim != 2 or committed.ndim != 2:
        raise ValueError("initial_pending and committed must be batched matrices.")
    if initial_pending.shape[0] != committed.shape[0] or horizon < 1:
        raise ValueError("Invalid batch size or rollout horizon.")
    pending_count = min(horizon, initial_pending.shape[1])
    remaining = horizon - pending_count
    if committed.shape[1] < remaining + 1:
        raise ValueError("Committed command sequence is too short for this horizon.")
    parts = [initial_pending[:, :pending_count]]
    if remaining > 0:
        parts.append(committed[:, 1 : remaining + 1])
    return torch.cat(parts, dim=1)


def set_deterministic_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def compute_batch_loss(
    model: ResidualMLP,
    batch: WindowBatch,
    config: AppConfig,
    include_jacobian: bool,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if config.causal_rollout.enabled:
        return _compute_causal_batch_loss(model, batch, config, include_jacobian)

    pred_pb, pred_vb = rollout_ball_dynamics(
        model,
        batch.pb[:, 0],
        batch.vb[:, 0],
        batch.theta,
        batch.omega,
        batch.dt,
        batch.constants,
    )
    position_loss = torch.zeros((), device=pred_pb.device, dtype=torch.float64)
    velocity_loss = torch.zeros_like(position_loss)
    details: dict[str, torch.Tensor] = {}
    for step, weight in zip(config.data.rollout_steps, config.loss.horizon_weights):
        p_error = pred_pb[:, step] - batch.pb[:, step]
        v_error = pred_vb[:, step] - batch.vb[:, step]
        p_mse = p_error.square().mean()
        v_mse = v_error.square().mean()
        position_loss = position_loss + weight * p_mse / config.loss.position_scale_m**2
        velocity_loss = velocity_loss + weight * v_mse / config.loss.velocity_scale_mps**2
        details[f"position_rmse_h{step}"] = torch.sqrt(p_mse)
        details[f"velocity_rmse_h{step}"] = torch.sqrt(v_mse)

    flat_features = batch.features.reshape(-1, batch.features.shape[-1])
    flat_target = batch.residual_target.reshape(-1)
    point_prediction = model(flat_features)
    point_loss = F.smooth_l1_loss(
        point_prediction,
        flat_target,
        beta=config.loss.huber_beta_mps2,
    )

    jacobian_loss = torch.zeros_like(position_loss)
    if include_jacobian and config.loss.jacobian_weight > 0.0:
        normalized = (
            (batch.features[:, 0, :] - model.input_mean) / model.input_std
        ).detach().requires_grad_(True)
        normalized_output = model.normalized_residual(normalized)
        gradient = torch.autograd.grad(
            normalized_output.sum(), normalized, create_graph=True
        )[0]
        jacobian_loss = gradient.square().sum(dim=-1).mean()

    total = (
        position_loss
        + velocity_loss
        + config.loss.point_weight * point_loss
        + config.loss.jacobian_weight * jacobian_loss
    )
    residual_error = point_prediction - flat_target
    target_centered = flat_target - flat_target.mean()
    r2 = 1.0 - residual_error.square().sum() / target_centered.square().sum().clamp_min(1e-15)
    details.update(
        {
            "total_loss": total,
            "position_rollout_loss": position_loss,
            "velocity_rollout_loss": velocity_loss,
            "point_residual_loss": point_loss,
            "jacobian_loss": jacobian_loss,
            "residual_rmse": torch.sqrt(residual_error.square().mean()),
            "residual_mae": residual_error.abs().mean(),
            "residual_bias": residual_error.mean(),
            "residual_r2": r2,
            "residual_mean": point_prediction.mean(),
            "residual_std": point_prediction.std(unbiased=False),
            "residual_saturation_rate": (
                point_prediction.abs() >= 0.95 * model.output_limit_mps2
            ).to(torch.float64).mean(),
        }
    )
    return total, details


def _compute_causal_batch_loss(
    model: ResidualMLP,
    batch: WindowBatch,
    config: AppConfig,
    include_jacobian: bool,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if any(
        item is None
        for item in (
            batch.causal_state,
            batch.smooth_pb,
            batch.smooth_vb,
            batch.commands,
            batch.geometry,
            batch.response,
            batch.state_scales,
        )
    ):
        raise ValueError("Causal rollout batch is incomplete.")
    causal_state = batch.causal_state
    smooth_pb = batch.smooth_pb
    smooth_vb = batch.smooth_vb
    commands = batch.commands
    geometry = batch.geometry
    response = batch.response
    state_scales = batch.state_scales
    assert causal_state is not None
    assert smooth_pb is not None and smooth_vb is not None
    assert commands is not None and geometry is not None and response is not None
    assert state_scales is not None

    rollout = rollout_causal_dynamics(
        model,
        causal_state[:, 0],
        commands,
        batch.dt,
        batch.constants,
        geometry,
        response,
    )
    primary_position = torch.zeros((), device=causal_state.device, dtype=torch.float64)
    primary_velocity = torch.zeros_like(primary_position)
    smooth_position = torch.zeros_like(primary_position)
    smooth_velocity = torch.zeros_like(primary_position)
    auxiliary_weight = config.causal_rollout.smooth_rollout_weight
    details: dict[str, torch.Tensor] = {}
    for step, weight in zip(config.data.rollout_steps, config.loss.horizon_weights):
        causal_p_mse = (rollout.states[:, step, 0] - causal_state[:, step, 0]).square().mean()
        causal_v_mse = (rollout.states[:, step, 1] - causal_state[:, step, 1]).square().mean()
        smooth_p_mse = (rollout.states[:, step, 0] - smooth_pb[:, step]).square().mean()
        smooth_v_mse = (rollout.states[:, step, 1] - smooth_vb[:, step]).square().mean()
        weighted_causal_position = (
            weight * causal_p_mse / config.loss.position_scale_m**2
        )
        weighted_causal_velocity = (
            weight * causal_v_mse / config.loss.velocity_scale_mps**2
        )
        weighted_smooth_position = (
            auxiliary_weight
            * weight
            * smooth_p_mse
            / config.loss.position_scale_m**2
        )
        weighted_smooth_velocity = (
            auxiliary_weight
            * weight
            * smooth_v_mse
            / config.loss.velocity_scale_mps**2
        )
        primary_position = (
            primary_position + weighted_causal_position
        )
        primary_velocity = (
            primary_velocity + weighted_causal_velocity
        )
        smooth_position = (
            smooth_position + weight * smooth_p_mse / config.loss.position_scale_m**2
        )
        smooth_velocity = (
            smooth_velocity + weight * smooth_v_mse / config.loss.velocity_scale_mps**2
        )
        theta_mse = (
            rollout.states[:, step, 2] - causal_state[:, step, 2]
        ).square().mean()
        vrz_mse = (
            rollout.states[:, step, 3] - causal_state[:, step, 3]
        ).square().mean()
        details.update(
            {
                f"causal_position_rmse_h{step}": torch.sqrt(causal_p_mse),
                f"causal_velocity_rmse_h{step}": torch.sqrt(causal_v_mse),
                f"smooth_position_rmse_h{step}": torch.sqrt(smooth_p_mse),
                f"smooth_velocity_rmse_h{step}": torch.sqrt(smooth_v_mse),
                f"theta_response_rmse_h{step}": torch.sqrt(theta_mse),
                f"vrz_response_rmse_h{step}": torch.sqrt(vrz_mse),
                f"causal_position_mse_h{step}": causal_p_mse,
                f"causal_velocity_mse_h{step}": causal_v_mse,
                f"smooth_position_mse_h{step}": smooth_p_mse,
                f"smooth_velocity_mse_h{step}": smooth_v_mse,
                f"weighted_causal_position_loss_h{step}": weighted_causal_position,
                f"weighted_causal_velocity_loss_h{step}": weighted_causal_velocity,
                f"weighted_smooth_position_loss_h{step}": weighted_smooth_position,
                f"weighted_smooth_velocity_loss_h{step}": weighted_smooth_velocity,
                f"weighted_rollout_loss_h{step}": (
                    weighted_causal_position
                    + weighted_causal_velocity
                    + weighted_smooth_position
                    + weighted_smooth_velocity
                ),
            }
        )

    flat_features = batch.features.reshape(-1, batch.features.shape[-1])
    flat_target = batch.residual_target.reshape(-1)
    point_prediction = model(flat_features)
    point_loss = F.smooth_l1_loss(
        point_prediction,
        flat_target,
        beta=config.loss.huber_beta_mps2,
    )

    jacobian_loss = torch.zeros_like(primary_position)
    theta_channel_loss = torch.zeros_like(primary_position)
    anti_damping_loss = torch.zeros_like(primary_position)
    jacobian_norm = torch.zeros(1, device=causal_state.device, dtype=torch.float64)
    theta_ratio = torch.ones(1, device=causal_state.device, dtype=torch.float64)
    anti_damping = torch.zeros(1, device=causal_state.device, dtype=torch.float64)
    derivative_losses_enabled = include_jacobian and (
        config.causal_rollout.jacobian_weight > 0.0
        or config.control_channel.enabled
    )
    if derivative_losses_enabled:
        # Re-evaluate the network at detached endpoint and actual RK4 stage states.
        # This isolates the local residual Jacobian from backpropagation through rollout history.
        all_states = torch.cat(
            (
                rollout.states,
                rollout.stage_states.reshape(
                    rollout.stage_states.shape[0], -1, rollout.stage_states.shape[-1]
                ),
            ),
            dim=1,
        )
        flat_states = all_states.detach().reshape(-1, 4).requires_grad_(True)
        states_per_window = all_states.shape[1]
        repeated_constants = batch.constants.repeat_interleave(states_per_window)
        repeated_geometry = geometry.repeat_interleave(states_per_window)
        total, nominal, _, _ = causal_residual_torch(
            model,
            flat_states,
            repeated_constants,
            repeated_geometry,
        )
        total_jacobian = torch.autograd.grad(
            total.sum(), flat_states, create_graph=True, retain_graph=True
        )[0]
        nominal_jacobian = torch.autograd.grad(
            nominal.sum(), flat_states, create_graph=True
        )[0]

        if config.control_channel.enabled and config.control_channel.collocation_fraction > 0.0:
            collocation_states, collocation_constants, collocation_geometry = (
                _deterministic_collocation_batch(batch, config)
            )
            collocation_states = collocation_states.requires_grad_(True)
            collocation_total, collocation_nominal, _, _ = causal_residual_torch(
                model,
                collocation_states,
                collocation_constants,
                collocation_geometry,
            )
            collocation_total_jacobian = torch.autograd.grad(
                collocation_total.sum(),
                collocation_states,
                create_graph=True,
                retain_graph=True,
            )[0]
            collocation_nominal_jacobian = torch.autograd.grad(
                collocation_nominal.sum(), collocation_states, create_graph=True
            )[0]
            total_jacobian = torch.cat(
                (total_jacobian, collocation_total_jacobian), dim=0
            )
            nominal_jacobian = torch.cat(
                (nominal_jacobian, collocation_nominal_jacobian), dim=0
            )

        residual_jacobian = total_jacobian - nominal_jacobian
        jacobian_norm = torch.linalg.vector_norm(
            residual_jacobian * state_scales, dim=-1
        )
        limit = config.causal_rollout.jacobian_norm_limit_mps2
        jacobian_loss = torch.relu(jacobian_norm / limit - 1.0).square().mean()
        if config.control_channel.enabled:
            (
                theta_channel_loss,
                anti_damping_loss,
                theta_ratio,
                anti_damping,
            ) = control_channel_losses(total_jacobian, nominal_jacobian, config)

    position_loss = primary_position + auxiliary_weight * smooth_position
    velocity_loss = primary_velocity + auxiliary_weight * smooth_velocity
    rollout_loss = position_loss + velocity_loss
    weighted_point_loss = config.loss.point_weight * point_loss
    weighted_jacobian_loss = config.causal_rollout.jacobian_weight * jacobian_loss
    weighted_theta_channel_loss = (
        config.control_channel.theta_weight * theta_channel_loss
    )
    weighted_anti_damping_loss = (
        config.control_channel.anti_damping_weight * anti_damping_loss
    )
    total = (
        rollout_loss
        + weighted_point_loss
        + weighted_jacobian_loss
        + weighted_theta_channel_loss
        + weighted_anti_damping_loss
    )
    fraction_denominator = total.detach().clamp_min(torch.finfo(total.dtype).eps)
    residual_error = point_prediction - flat_target
    target_centered = flat_target - flat_target.mean()
    r2 = 1.0 - residual_error.square().sum() / target_centered.square().sum().clamp_min(1e-15)
    details.update(
        {
            "total_loss": total,
            "position_rollout_loss": position_loss,
            "velocity_rollout_loss": velocity_loss,
            "rollout_loss": rollout_loss,
            "causal_position_rollout_loss": primary_position,
            "causal_velocity_rollout_loss": primary_velocity,
            "smooth_position_rollout_loss": smooth_position,
            "smooth_velocity_rollout_loss": smooth_velocity,
            "point_residual_loss": point_loss,
            "weighted_point_residual_loss": weighted_point_loss,
            "jacobian_loss": jacobian_loss,
            "weighted_jacobian_loss": weighted_jacobian_loss,
            "weighted_theta_channel_loss": weighted_theta_channel_loss,
            "weighted_anti_damping_loss": weighted_anti_damping_loss,
            "rollout_loss_fraction": rollout_loss.detach() / fraction_denominator,
            "point_loss_fraction": weighted_point_loss.detach() / fraction_denominator,
            "jacobian_loss_fraction": (
                weighted_jacobian_loss.detach() / fraction_denominator
            ),
            "control_channel_loss_fraction": (
                (
                    weighted_theta_channel_loss.detach()
                    + weighted_anti_damping_loss.detach()
                )
                / fraction_denominator
            ),
            "jacobian_norm_mean": jacobian_norm.mean(),
            "jacobian_norm_max": jacobian_norm.max(),
            "theta_channel_loss": theta_channel_loss,
            "theta_ratio_min": theta_ratio.min(),
            "theta_ratio_max": theta_ratio.max(),
            "anti_damping_loss": anti_damping_loss,
            "anti_damping_max_sinv": anti_damping.max(),
            "residual_rmse": torch.sqrt(residual_error.square().mean()),
            "residual_mae": residual_error.abs().mean(),
            "residual_bias": residual_error.mean(),
            "residual_r2": r2,
            "residual_mean": point_prediction.mean(),
            "residual_std": point_prediction.std(unbiased=False),
            "residual_saturation_rate": (
                point_prediction.abs() >= 0.95 * model.output_limit_mps2
            ).to(torch.float64).mean(),
        }
    )
    return total, details


def _deterministic_collocation_batch(
    batch: WindowBatch,
    config: AppConfig,
) -> tuple[torch.Tensor, TorchPhysicalConstants, TorchGeometryConstants]:
    """Create a deterministic Latin-hypercube batch tied to trial metadata."""

    assert batch.causal_state is not None and batch.geometry is not None
    count = max(
        1,
        int(round(batch.causal_state.shape[0] * config.control_channel.collocation_fraction)),
    )
    count = min(count, batch.causal_state.shape[0])
    device = batch.causal_state.device
    base = (torch.arange(count, device=device, dtype=torch.float64) + 0.5) / count

    def dimension(low: float, high: float, shift: int) -> torch.Tensor:
        unit = torch.roll(base, shifts=shift % count)
        return low + (high - low) * unit

    channel = config.control_channel
    states = torch.stack(
        (
            dimension(
                config.data.ball_position_min_m,
                config.data.ball_position_max_m,
                0,
            ),
            dimension(
                channel.collocation_vb_min_mps,
                channel.collocation_vb_max_mps,
                1,
            ),
            dimension(
                channel.collocation_theta_min_rad,
                channel.collocation_theta_max_rad,
                2,
            ),
            dimension(
                channel.collocation_vrz_min_mps,
                channel.collocation_vrz_max_mps,
                3,
            ),
        ),
        dim=-1,
    )
    constants = TorchPhysicalConstants(
        plank_length_m=batch.constants.plank_length_m[:count],
        ball_position_offset_m=batch.constants.ball_position_offset_m[:count],
        gravity_mps2=batch.constants.gravity_mps2[:count],
        acceleration_gain=batch.constants.acceleration_gain[:count],
    )
    geometry = TorchGeometryConstants(
        plank_length_m=batch.geometry.plank_length_m[:count],
        rope_length_m=batch.geometry.rope_length_m[:count],
    )
    return states, constants, geometry


def control_channel_losses(
    total_jacobian: torch.Tensor,
    nominal_jacobian: torch.Tensor,
    config: AppConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute differentiable theta-sector and anti-damping penalties."""

    channel = config.control_channel
    theta_mask = nominal_jacobian[:, 2].abs() >= channel.nominal_theta_gain_floor
    if bool(theta_mask.any()):
        theta_ratio = (
            total_jacobian[theta_mask, 2] / nominal_jacobian[theta_mask, 2]
        )
        theta_loss = (
            torch.relu(channel.theta_ratio_soft_min - theta_ratio).square()
            + torch.relu(theta_ratio - channel.theta_ratio_soft_max).square()
        ).mean()
    else:
        theta_ratio = torch.ones(
            1, device=total_jacobian.device, dtype=total_jacobian.dtype
        )
        theta_loss = torch.zeros(
            (), device=total_jacobian.device, dtype=total_jacobian.dtype
        )
    anti_damping = total_jacobian[:, 1]
    anti_damping_loss = torch.relu(
        anti_damping - channel.anti_damping_soft_max_sinv
    ).square().mean()
    return theta_loss, anti_damping_loss, theta_ratio, anti_damping


def train_model(config: AppConfig) -> Path:
    """Train one deterministic MLP and return its run directory."""

    manifest = load_manifest(config)
    train_trials = load_processed_trials(config, split="train")
    validation_trials = load_processed_trials(config, split="validation")
    normalization = manifest["normalization"]
    max_horizon = max(config.data.rollout_steps)
    train_store = WindowStore(train_trials, max_horizon, config, normalization)
    validation_store = WindowStore(validation_trials, max_horizon, config, normalization)
    validation_refs = _validation_references(
        validation_store.references,
        config.data.validation_stride,
        config.training.validation_max_windows,
    )

    set_deterministic_seed(config.training.seed)
    benchmark_rng = np.random.default_rng(config.training.seed)
    benchmark_refs = train_store.balanced_references(
        benchmark_rng, config.training.batch_size
    )
    device, device_benchmark = select_training_device(
        config,
        train_store,
        benchmark_refs,
        normalization,
    )
    set_deterministic_seed(config.training.seed)
    rng = np.random.default_rng(config.training.seed)
    model = create_model(config.model, normalization).to(device=device, dtype=torch.float64)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.training.lr_factor,
        patience=config.training.lr_patience,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"mlp5_seed{config.training.seed}_{config.config_hash[:8]}_{timestamp}"
    run_dir = config.paths.training_runs_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_path = run_dir / "best_checkpoint.pt"
    history_path = run_dir / "history.csv"
    monitor = WandbMonitor(config.wandb, config.paths.wandb_dir)
    monitor.start(
        run_name,
        {
            "config": _safe_config_for_logging(config.raw),
            "config_hash": config.config_hash,
            "dataset_hash": manifest["dataset_hash"],
            "split_counts": manifest["split_counts"],
            "seed": config.training.seed,
            "device": str(device),
            "device_benchmark": device_benchmark,
        },
    )

    history: list[dict[str, Any]] = []
    best_validation = math.inf
    best_epoch = 0
    epochs_without_improvement = 0
    global_step = 0
    stop_reason = "max_epochs"
    started = time.perf_counter()
    try:
        for epoch in range(1, config.training.max_epochs + 1):
            epoch_started = time.perf_counter()
            model.train()
            train_accumulator: dict[str, float] = {}
            for _ in range(config.training.steps_per_epoch):
                refs = train_store.balanced_references(rng, config.training.batch_size)
                batch = train_store.make_batch(refs, device)
                optimizer.zero_grad(set_to_none=True)
                loss, details = compute_batch_loss(model, batch, config, include_jacobian=True)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite training loss.")
                loss.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.training.gradient_clip_norm
                )
                optimizer.step()
                global_step += 1
                _accumulate(train_accumulator, details)
                if global_step % config.wandb.log_batch_interval == 0:
                    monitor.log(
                        {
                            **{
                                f"train/{key}": float(value.detach().cpu())
                                for key, value in details.items()
                                if value.numel() == 1
                            },
                            "train/gradient_norm": float(gradient_norm.detach().cpu()),
                            "optimizer/learning_rate": optimizer.param_groups[0]["lr"],
                            "epoch": epoch,
                        },
                        step=global_step,
                    )

            train_metrics = {
                key: value / config.training.steps_per_epoch
                for key, value in train_accumulator.items()
            }
            validation_metrics = validate_model(
                model,
                validation_store,
                validation_refs,
                config,
                device,
                config.training.batch_size,
            )
            validation_loss = validation_metrics["total_loss"]
            scheduler.step(validation_loss)
            improved = validation_loss < best_validation - 1e-12
            if improved:
                best_validation = validation_loss
                best_epoch = epoch
                epochs_without_improvement = 0
                _save_checkpoint(
                    checkpoint_path,
                    model,
                    config,
                    manifest,
                    epoch,
                    validation_metrics,
                    str(device),
                    device_benchmark,
                    monitor.state(),
                )
            else:
                epochs_without_improvement += 1

            epoch_row: dict[str, Any] = {
                "epoch": epoch,
                "global_step": global_step,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "epoch_time_s": time.perf_counter() - epoch_started,
                "best_epoch": best_epoch,
                "early_stopping_counter": epochs_without_improvement,
            }
            epoch_row.update({f"train_{key}": value for key, value in train_metrics.items()})
            epoch_row.update(
                {f"validation_{key}": value for key, value in validation_metrics.items()}
            )
            history.append(epoch_row)
            _write_history(history_path, history)

            epoch_log = {
                "epoch": epoch,
                "epoch/time_s": epoch_row["epoch_time_s"],
                "epoch/best_epoch": best_epoch,
                "epoch/early_stopping_counter": epochs_without_improvement,
                "system/cpu_max_rss_mb": _cpu_max_rss_mb(),
                **{f"train_epoch/{key}": value for key, value in train_metrics.items()},
                **{
                    f"validation/{key}": value
                    for key, value in validation_metrics.items()
                },
            }
            if device.type == "cuda":
                epoch_log.update(
                    {
                        "system/gpu_allocated_mb": torch.cuda.memory_allocated(device)
                        / 1024**2,
                        "system/gpu_peak_mb": torch.cuda.max_memory_allocated(device)
                        / 1024**2,
                    }
                )
            monitor.log(epoch_log, step=global_step)
            if epoch % config.wandb.log_histogram_interval_epochs == 0:
                monitor.log_histograms(model, epoch)

            if epochs_without_improvement >= config.training.early_stopping_patience:
                stop_reason = "early_stopping"
                break

        model, checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
        portable_metadata = {
            "config_hash": config.config_hash,
            "dataset_hash": manifest["dataset_hash"],
            "best_epoch": checkpoint["epoch"],
            "training_device": checkpoint["training_device"],
            "wandb": checkpoint["wandb"],
        }
        portable_path = run_dir / "weights.npz"
        save_portable_weights(model, portable_path, portable_metadata)
        total_time = time.perf_counter() - started
        checkpoint_hash = _sha256(checkpoint_path)
        summary = {
            "run_name": run_name,
            "run_dir": run_dir.relative_to(config.paths.training_runs_dir.parent.parent).as_posix(),
            "config_hash": config.config_hash,
            "dataset_hash": manifest["dataset_hash"],
            "seed": config.training.seed,
            "training_device": str(device),
            "device_benchmark": device_benchmark,
            "best_epoch": best_epoch,
            "epochs_completed": len(history),
            "stop_reason": stop_reason,
            "best_validation_loss": best_validation,
            "best_validation_metrics": checkpoint["validation_metrics"],
            "total_training_time_s": total_time,
            "checkpoint_sha256": checkpoint_hash,
            "weights_sha256": _sha256(portable_path),
            "wandb": monitor.state(),
        }
        _write_json(run_dir / "training_summary.json", summary)
        _write_json(config.paths.training_runs_dir / "latest.json", {"run_dir": str(run_dir)})
        monitor.summary(
            {
                "best_epoch": best_epoch,
                "best_validation_loss": best_validation,
                "stop_reason": stop_reason,
                "total_training_time_s": total_time,
                "checkpoint_sha256": checkpoint_hash,
            }
        )
        monitor.finish(exit_code=0)
        return run_dir
    except Exception:
        monitor.finish(exit_code=1)
        raise


def validate_model(
    model: ResidualMLP,
    store: WindowStore,
    references: Sequence[WindowReference],
    config: AppConfig,
    device: torch.device,
    batch_size: int,
) -> dict[str, float]:
    model.eval()
    sums: dict[str, float] = {}
    count = 0
    include_jacobian = (
        config.causal_rollout.enabled
        and (
            config.causal_rollout.jacobian_weight > 0.0
            or config.control_channel.enabled
        )
    )
    gradient_context = torch.enable_grad() if include_jacobian else torch.no_grad()
    with gradient_context:
        for offset in range(0, len(references), batch_size):
            refs = references[offset : offset + batch_size]
            batch = store.make_batch(refs, device)
            _, details = compute_batch_loss(
                model, batch, config, include_jacobian=include_jacobian
            )
            weight = len(refs)
            for key, value in details.items():
                if value.numel() == 1:
                    sums[key] = sums.get(key, 0.0) + float(value.detach().cpu()) * weight
            count += weight
    if count == 0:
        raise RuntimeError("Validation contains no windows.")
    return {key: value / count for key, value in sums.items()}


def select_training_device(
    config: AppConfig,
    store: WindowStore,
    references: Sequence[WindowReference],
    normalization: dict[str, Any],
) -> tuple[torch.device, dict[str, Any]]:
    requested = config.runtime.device
    info: dict[str, Any] = {
        "requested": requested,
        "cuda_available": torch.cuda.is_available(),
        "cpu_median_step_s": None,
        "cuda_median_step_s": None,
        "speedup": None,
        "reason": None,
    }
    if requested == "cpu" or not torch.cuda.is_available():
        info["reason"] = "cpu requested" if requested == "cpu" else "CUDA unavailable"
        return torch.device("cpu"), info
    if requested == "cuda":
        info["reason"] = "cuda requested"
        return torch.device("cuda"), info

    try:
        cpu_time = _benchmark_device(
            torch.device("cpu"), config, store, references, normalization
        )
        cuda_time = _benchmark_device(
            torch.device("cuda"), config, store, references, normalization
        )
        speedup = cpu_time / cuda_time
        info.update(
            {
                "cpu_median_step_s": cpu_time,
                "cuda_median_step_s": cuda_time,
                "speedup": speedup,
                "gpu_name": torch.cuda.get_device_name(0),
                "torch_version": torch.__version__,
                "cuda_version": torch.version.cuda,
            }
        )
        if speedup >= config.runtime.gpu_min_speedup:
            info["reason"] = "CUDA met the configured speedup threshold"
            return torch.device("cuda"), info
        info["reason"] = "CUDA did not meet the configured speedup threshold"
    except (RuntimeError, FloatingPointError) as exc:
        info["reason"] = f"CUDA benchmark failed: {type(exc).__name__}: {exc}"
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return torch.device("cpu"), info


def _benchmark_device(
    device: torch.device,
    config: AppConfig,
    store: WindowStore,
    references: Sequence[WindowReference],
    normalization: dict[str, Any],
) -> float:
    set_deterministic_seed(config.training.seed)
    model = create_model(config.model, normalization).to(device=device, dtype=torch.float64)
    batch = store.make_batch(references, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )

    def step() -> None:
        optimizer.zero_grad(set_to_none=True)
        loss, _ = compute_batch_loss(model, batch, config, include_jacobian=True)
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite benchmark loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.gradient_clip_norm)
        optimizer.step()

    for _ in range(config.runtime.benchmark_warmup_steps):
        step()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    durations: list[float] = []
    for _ in range(config.runtime.benchmark_steps):
        started = time.perf_counter()
        step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        durations.append(time.perf_counter() - started)
    return float(np.median(np.asarray(durations)))


def _build_window_references(
    trials: Sequence[TrialData],
    horizon: int,
    stride: int,
    config: AppConfig | None = None,
) -> list[WindowReference]:
    refs: list[WindowReference] = []
    for trial_index, trial in enumerate(trials):
        segment_ids = trial.arrays["segment_id"]
        for segment_id in np.unique(segment_ids):
            indices = np.flatnonzero(segment_ids == segment_id)
            if indices.size <= horizon:
                continue
            start_min, start_max = int(indices[0]), int(indices[-1] - horizon)
            for start in range(start_min, start_max + 1, stride):
                if segment_ids[start + horizon] != segment_id:
                    continue
                midpoint = start + horizon // 2
                if config is not None and config.data.split_assignments:
                    regime = int(trial.arrays["speed_regime"][midpoint])
                    position_bin = int(
                        position_bin_indices(
                            np.asarray([trial.arrays["pb"][midpoint]]),
                            config.data.position_bin_edges_m,
                        )[0]
                    )
                else:
                    regime = int(trial.arrays["speed_regime"][start])
                    position_bin = -1
                refs.append(
                    WindowReference(
                        trial_index=trial_index,
                        start=start,
                        regime=regime,
                        position_bin=position_bin,
                    )
                )
    return refs


def _validation_references(
    references: Sequence[WindowReference],
    stride: int,
    maximum: int,
) -> list[WindowReference]:
    selected = list(references[:: max(1, stride)])
    if len(selected) <= maximum:
        return selected
    indices = np.linspace(0, len(selected) - 1, maximum, dtype=np.int64)
    return [selected[int(index)] for index in indices]


def _save_checkpoint(
    path: Path,
    model: ResidualMLP,
    config: AppConfig,
    manifest: dict[str, Any],
    epoch: int,
    validation_metrics: dict[str, float],
    device: str,
    device_benchmark: dict[str, Any],
    wandb_state: dict[str, Any],
) -> None:
    checkpoint = {
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "model": checkpoint_model_metadata(model),
        "config_hash": config.config_hash,
        "dataset_hash": manifest["dataset_hash"],
        "seed": config.training.seed,
        "epoch": epoch,
        "validation_metrics": validation_metrics,
        "training_device": device,
        "device_benchmark": device_benchmark,
        "wandb": wandb_state,
    }
    temporary = path.with_suffix(".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def resolve_run_dir(config: AppConfig, run_dir: str | Path | None) -> Path:
    if run_dir is not None:
        path = Path(run_dir).expanduser().resolve()
    else:
        latest_path = config.paths.training_runs_dir / "latest.json"
        if not latest_path.is_file():
            raise FileNotFoundError("No latest training run is available.")
        with latest_path.open("r", encoding="utf-8") as stream:
            path = Path(json.load(stream)["run_dir"]).expanduser().resolve()
    if not (path / "best_checkpoint.pt").is_file():
        raise FileNotFoundError(f"Training checkpoint not found below {path}")
    return path


def _accumulate(accumulator: dict[str, float], details: dict[str, torch.Tensor]) -> None:
    for key, value in details.items():
        if value.numel() == 1:
            accumulator[key] = accumulator.get(key, 0.0) + float(value.detach().cpu())


def _write_history(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _safe_config_for_logging(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _safe_config_for_logging(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_config_for_logging(item) for item in value]
    if isinstance(value, Path):
        return value.name
    return value


def _cpu_max_rss_mb() -> float:
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
