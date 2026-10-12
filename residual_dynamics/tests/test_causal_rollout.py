from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch

from residual_dynamics.config import load_config
from residual_dynamics.data import (
    PhysicalConstants,
    ResponseConstants,
    TrialData,
    _response_constants,
    _validate_pending_queue,
    _zero_order_hold,
)
from residual_dynamics.dynamics import (
    TorchGeometryConstants,
    TorchPhysicalConstants,
    TorchResponseConstants,
    causal_residual_torch,
    response_stages_torch,
    rollout_causal_dynamics,
)
from residual_dynamics.model import ResidualMLP
from residual_dynamics.training import (
    _build_window_references,
    reconstruct_deployment_commands,
)


CONFIG = Path("residual_dynamics/configs/mlp_5input_causal_rollout.yaml")


def _constants(batch: int = 1) -> tuple[TorchPhysicalConstants, TorchGeometryConstants]:
    physical = TorchPhysicalConstants(
        plank_length_m=torch.ones(batch, dtype=torch.float64),
        ball_position_offset_m=torch.zeros(batch, dtype=torch.float64),
        gravity_mps2=torch.zeros(batch, dtype=torch.float64),
        acceleration_gain=torch.ones(batch, dtype=torch.float64),
    )
    geometry = TorchGeometryConstants(
        plank_length_m=torch.ones(batch, dtype=torch.float64),
        rope_length_m=torch.ones(batch, dtype=torch.float64),
    )
    return physical, geometry


def _response(batch: int = 1, enabled: bool = False) -> TorchResponseConstants:
    return TorchResponseConstants(
        enabled=torch.full((batch,), enabled, dtype=torch.bool),
        tau_s=torch.full((batch,), 0.2, dtype=torch.float64),
        gain=torch.ones(batch, dtype=torch.float64),
        bias=torch.zeros(batch, dtype=torch.float64),
        max_velocity_mps=torch.full((batch,), 3.0, dtype=torch.float64),
    )


def test_zero_order_hold_never_interpolates_future_command() -> None:
    source_time = np.asarray([0.0, 0.1, 0.3])
    commands = np.asarray([1.0, 2.0, 4.0])
    indices, sampled = _zero_order_hold(
        source_time, commands, np.asarray([0.0, 0.05, 0.1, 0.29, 0.3])
    )
    assert np.array_equal(indices, [0, 0, 1, 1, 2])
    assert np.array_equal(sampled, [1.0, 1.0, 2.0, 2.0, 4.0])


def test_pending_queue_is_consumed_before_future_commits() -> None:
    pending = torch.tensor([[10.0, 11.0, 12.0]], dtype=torch.float64)
    committed = torch.tensor([[12.0, 13.0, 14.0, 15.0, 16.0]], dtype=torch.float64)
    result = reconstruct_deployment_commands(pending, committed, horizon=5)
    assert torch.equal(
        result, torch.tensor([[10.0, 11.0, 12.0, 13.0, 14.0]], dtype=torch.float64)
    )


def test_sparse_pending_diagnostic_detects_queue_reset() -> None:
    command = np.arange(8, dtype=np.float64)
    pending = np.full((8, 3), np.nan)
    pending[4] = [2.0, 3.0, 4.0]
    _validate_pending_queue(command, pending, 3)
    pending[4] = [0.0, 0.0, 0.0]
    with pytest.raises(ValueError, match="mismatch"):
        _validate_pending_queue(command, pending, 3)


def test_metadata_response_parameters_are_used() -> None:
    config = load_config(CONFIG)
    metadata = {
        "parameters": {
            "controller.response.enabled": True,
            "controller.response.tau_s": 0.187,
            "controller.response.gain": 1.03,
            "controller.response.bias": -0.004,
            "controller.constraints.max_velocity": 3.0,
            "predictor.delay_steps": 3,
        }
    }
    response = _response_constants(metadata, config)
    assert response == ResponseConstants(True, 0.187, 1.03, -0.004, 3.0, 3)


def test_exponential_velocity_response_matches_closed_form() -> None:
    previous = torch.tensor([0.4], dtype=torch.float64)
    command = torch.tensor([0.2], dtype=torch.float64)
    dt = torch.tensor([0.1], dtype=torch.float64)
    response = TorchResponseConstants(
        enabled=torch.tensor([True]),
        tau_s=torch.tensor([0.2], dtype=torch.float64),
        gain=torch.tensor([2.0], dtype=torch.float64),
        bias=torch.tensor([-0.1], dtype=torch.float64),
        max_velocity_mps=torch.tensor([3.0], dtype=torch.float64),
    )
    start, half, end = response_stages_torch(previous, command, dt, response)
    target = 0.3
    assert start.item() == pytest.approx(0.4)
    assert half.item() == pytest.approx(target + 0.1 * math.exp(-0.25))
    assert end.item() == pytest.approx(target + 0.1 * math.exp(-0.5))


def test_causal_rk4_constant_acceleration_and_stage_shape() -> None:
    model = ResidualMLP(np.zeros(5), np.ones(5))
    desired_acceleration = 0.1
    with torch.no_grad():
        model.layers[-1].bias.fill_(math.atanh(desired_acceleration / 0.25))
    physical, geometry = _constants(batch=2)
    initial = torch.tensor([[0.1, -0.1, 0.0, 0.0], [0.3, 0.2, 0.0, 0.0]])
    initial = initial.to(torch.float64)
    horizon = 8
    dt = torch.full((2, horizon), 0.02, dtype=torch.float64)
    rollout = rollout_causal_dynamics(
        model,
        initial,
        torch.zeros_like(dt),
        dt,
        physical,
        geometry,
        _response(batch=2),
    )
    duration = horizon * 0.02
    assert rollout.states.shape == (2, horizon + 1, 4)
    assert rollout.stage_states.shape == (2, horizon, 4, 4)
    assert torch.allclose(
        rollout.states[:, -1, 1],
        initial[:, 1] + desired_acceleration * duration,
        atol=1.0e-12,
    )
    assert torch.allclose(
        rollout.states[:, -1, 0],
        initial[:, 0]
        + initial[:, 1] * duration
        + 0.5 * desired_acceleration * duration**2,
        atol=1.0e-12,
    )


def test_physical_jacobian_has_analytic_shape_and_value() -> None:
    class LinearResidual(torch.nn.Module):
        def forward(self, features: torch.Tensor) -> torch.Tensor:
            return 2.0 * features[:, 0] + 3.0 * features[:, 1]

    physical, geometry = _constants(batch=3)
    state = torch.zeros((3, 4), dtype=torch.float64, requires_grad=True)
    _, _, residual, _ = causal_residual_torch(
        LinearResidual(), state, physical, geometry  # type: ignore[arg-type]
    )
    jacobian = torch.autograd.grad(residual.sum(), state)[0]
    expected = torch.tensor([2.0, 3.0, 0.0, 0.0], dtype=torch.float64).repeat(3, 1)
    assert jacobian.shape == (3, 4)
    assert torch.allclose(jacobian, expected)
    assert torch.isfinite(jacobian).all()


def test_all_rollout_stage_jacobians_have_finite_model_gradient() -> None:
    torch.manual_seed(4)
    model = ResidualMLP(np.zeros(5), np.ones(5))
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(std=0.03)
    physical, geometry = _constants(batch=2)
    initial = torch.tensor(
        [[0.2, -0.1, 0.02, 0.1], [0.4, 0.05, -0.01, -0.08]],
        dtype=torch.float64,
    )
    dt = torch.full((2, 3), 1.0 / 60.0, dtype=torch.float64)
    rollout = rollout_causal_dynamics(
        model,
        initial,
        torch.zeros_like(dt),
        dt,
        physical,
        geometry,
        _response(batch=2),
    )
    stage_state = rollout.stage_states.detach().reshape(-1, 4).requires_grad_(True)
    states_per_batch = 3 * 4
    _, _, residual, _ = causal_residual_torch(
        model,
        stage_state,
        physical.repeat_interleave(states_per_batch),
        geometry.repeat_interleave(states_per_batch),
    )
    jacobian = torch.autograd.grad(residual.sum(), stage_state, create_graph=True)[0]
    assert jacobian.shape == stage_state.shape
    loss = jacobian.square().mean()
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_window_references_never_cross_segment_boundary() -> None:
    arrays = {
        "time_s": np.arange(10, dtype=np.float64),
        "segment_id": np.asarray([0] * 5 + [1] * 5),
        "speed_regime": np.zeros(10, dtype=np.int8),
    }
    trial = TrialData(
        "trial", "group", "session", "train", "constant", "source", arrays,
        PhysicalConstants(1.0, 1.0, 0.0, 9.81, 1.0),
    )
    references = _build_window_references([trial], horizon=3, stride=1)
    assert [ref.start for ref in references] == [0, 1, 5, 6]
