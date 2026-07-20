"""Tests for the shared velocity-interface nonlinear model."""

from __future__ import annotations

import math

import pytest
import torch

from baselines.velocity_interface_model import VelocityInterfaceModel, VelocityInterfaceModelCfg


def make_model(**overrides) -> VelocityInterfaceModel:
    cfg = VelocityInterfaceModelCfg(**overrides)
    return VelocityInterfaceModel(cfg)


def test_benchmark_coordinate_uses_slide_and_block_offset():
    model = make_model(ball_position_offset=0.33)
    benchmark_pb = torch.tensor([[0.10], [0.35], [0.60]])

    model_pb = model.model_ball_position(benchmark_pb)

    torch.testing.assert_close(model_pb, benchmark_pb + 0.33)


def test_ball_dynamics_and_inverse_recover_beam_angle():
    model = make_model()
    pb = torch.tensor([[0.10], [0.35], [0.60]], dtype=torch.float32)
    theta = torch.tensor([[-0.25], [0.0], [0.30]], dtype=torch.float32)
    omega = torch.tensor([[0.20], [-0.10], [0.15]], dtype=torch.float32)

    acceleration = model.ball_acceleration(pb, theta, omega)
    recovered = model.desired_beam_angle(pb, omega, acceleration, theta_max=0.5)

    torch.testing.assert_close(recovered, theta, atol=1.0e-6, rtol=1.0e-6)


def test_velocity_geometry_round_trip():
    model = make_model()
    theta = torch.tensor([[-0.3], [0.0], [0.35]], dtype=torch.float32)
    omega = torch.tensor([[-0.4], [0.2], [0.5]], dtype=torch.float32)

    velocity = model.vertical_velocity(theta, omega)
    recovered = model.beam_angular_velocity(theta, velocity)

    torch.testing.assert_close(recovered, omega, atol=1.0e-6, rtol=1.0e-6)


def test_feasible_acceleration_interval_matches_angle_endpoints():
    model = make_model()
    pb = torch.tensor([[0.2], [0.5]])
    omega = torch.tensor([[0.1], [0.3]])
    theta_max = math.radians(40.0)

    acceleration_min, acceleration_max = model.feasible_ball_acceleration(pb, omega, theta_max)

    expected_min = model.ball_acceleration(pb, torch.full_like(pb, theta_max), omega)
    expected_max = model.ball_acceleration(pb, torch.full_like(pb, -theta_max), omega)
    torch.testing.assert_close(acceleration_min, expected_min)
    torch.testing.assert_close(acceleration_max, expected_max)


def test_geometry_protection_remains_finite_near_domain_limit():
    model = make_model(rope_length=0.2, epsilon=1.0e-5)
    theta = torch.tensor([[1.4], [-1.4]])
    omega = torch.tensor([[10.0], [-10.0]])

    beta = model.rope_angle(theta)
    velocity = model.vertical_velocity(theta, omega)

    assert torch.isfinite(beta).all()
    assert torch.isfinite(velocity).all()


@pytest.mark.parametrize(
    "overrides",
    (
        {"plank_length": 0.0},
        {"rope_length": -1.0},
        {"ball_mass": 0.0},
        {"ball_radius": -0.1},
        {"ball_inertia_ratio": -0.1},
        {"epsilon": 1.0},
    ),
)
def test_invalid_model_parameters_fail_fast(overrides):
    with pytest.raises(ValueError):
        make_model(**overrides)
