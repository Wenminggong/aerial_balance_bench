from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch

from residual_dynamics.dynamics import TorchPhysicalConstants, rollout_ball_dynamics
from residual_dynamics.model import ResidualMLP, load_portable_weights, save_portable_weights


def test_model_starts_at_zero_and_is_bounded() -> None:
    model = ResidualMLP(np.zeros(5), np.ones(5))
    features = torch.randn(128, 5, dtype=torch.float64) * 100.0
    output = model(features)
    assert torch.equal(output, torch.zeros_like(output))
    with torch.no_grad():
        model.layers[-1].bias.fill_(100.0)
    assert torch.all(model(features) <= 0.25)
    assert torch.all(model(features) >= -0.25)


def test_portable_weights_round_trip(tmp_path: Path) -> None:
    torch.manual_seed(3)
    model = ResidualMLP(np.arange(5), np.arange(1, 6), hidden_dims=(16, 16))
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_()
    features = torch.randn(32, 5, dtype=torch.float64)
    expected = model(features)
    path = tmp_path / "weights.npz"
    save_portable_weights(model, path, {"dataset_hash": "abc"})
    restored, metadata = load_portable_weights(path)
    assert metadata["dataset_hash"] == "abc"
    assert torch.allclose(restored(features), expected, atol=0.0, rtol=0.0)


def test_rk4_constant_residual_matches_analytic_solution() -> None:
    model = ResidualMLP(np.zeros(5), np.ones(5))
    desired_acceleration = 0.1
    with torch.no_grad():
        model.layers[-1].bias.fill_(math.atanh(desired_acceleration / 0.25))
    batch, horizon = 3, 30
    pb0 = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64)
    vb0 = torch.tensor([-0.1, 0.0, 0.2], dtype=torch.float64)
    theta = torch.zeros(batch, horizon + 1, dtype=torch.float64)
    omega = torch.zeros_like(theta)
    dt = torch.full((batch, horizon), 1.0 / 60.0, dtype=torch.float64)
    constants = TorchPhysicalConstants(
        plank_length_m=torch.ones(batch, dtype=torch.float64),
        ball_position_offset_m=torch.zeros(batch, dtype=torch.float64),
        gravity_mps2=torch.zeros(batch, dtype=torch.float64),
        acceleration_gain=torch.ones(batch, dtype=torch.float64),
    )
    pb, vb = rollout_ball_dynamics(model, pb0, vb0, theta, omega, dt, constants)
    duration = horizon / 60.0
    assert torch.allclose(vb[:, -1], vb0 + desired_acceleration * duration, atol=1e-12)
    assert torch.allclose(
        pb[:, -1],
        pb0 + vb0 * duration + 0.5 * desired_acceleration * duration**2,
        atol=1e-12,
    )


def test_rollout_gradient_is_finite() -> None:
    model = ResidualMLP(np.zeros(5), np.ones(5))
    with torch.no_grad():
        model.layers[-1].weight.fill_(0.01)
    batch, horizon = 2, 4
    theta = torch.zeros(batch, horizon + 1, dtype=torch.float64)
    omega = torch.zeros_like(theta)
    dt = torch.full((batch, horizon), 0.02, dtype=torch.float64)
    constants = TorchPhysicalConstants(
        *(torch.ones(batch, dtype=torch.float64) for _ in range(4))
    )
    pb, vb = rollout_ball_dynamics(
        model,
        torch.zeros(batch, dtype=torch.float64),
        torch.zeros(batch, dtype=torch.float64),
        theta,
        omega,
        dt,
        constants,
    )
    loss = pb.square().mean() + vb.square().mean()
    loss.backward()
    assert all(parameter.grad is not None for parameter in model.parameters())
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters())
