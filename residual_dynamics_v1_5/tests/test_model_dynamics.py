from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from residual_dynamics_v1_5.config import load_config
from residual_dynamics_v1_5.dynamics import TorchPhysicalConstants, rollout_ball_dynamics
from residual_dynamics_v1_5.model import (
    create_model,
    load_portable_weights,
    save_portable_weights,
    smooth_deadzone,
)


CONFIG = Path("residual_dynamics_v1_5/configs/mlp_deadzone.yaml")


def _model():
    config = load_config(CONFIG)
    return create_model(config.model, {"mean": np.zeros(5), "std": np.ones(5)})


def test_deadzone_symmetry_direction_and_threshold_bounds() -> None:
    model = _model()
    threshold = model.deadzone_threshold
    drive = torch.tensor([-0.20, -0.01, 0.0, 0.01, 0.20], dtype=torch.float64)
    output = smooth_deadzone(drive, threshold, model.deadzone_smoothing_mps2)
    assert output[2].item() == 0.0
    assert torch.allclose(output, -torch.flip(output, dims=(0,)), atol=1e-15)
    assert abs(output[1]) < 1e-5
    assert output[0] < 0 < output[-1]
    assert model.deadzone_min_mps2 <= float(threshold.detach()) <= model.deadzone_max_mps2


def test_velocity_fusion_uses_deadzone_at_zero_and_drive_at_high_speed() -> None:
    model = _model()
    with torch.no_grad():
        model.mlp.layers[-1].bias.fill_(0.5)
    features = torch.zeros(2, 5, dtype=torch.float64)
    features[:, 4] = 0.08
    features[1, 1] = 0.20
    output = model(features)
    assert torch.allclose(output.total_acceleration[0], output.deadzone_acceleration[0])
    assert abs(output.total_acceleration[1] - output.drive_acceleration[1]) < 1e-12
    assert torch.all(output.mlp_residual.abs() <= 0.20)


def test_composite_rk4_is_continuous_and_has_finite_gradients() -> None:
    model = _model()
    with torch.no_grad():
        model.mlp.layers[-1].weight.fill_(0.01)
    batch, horizon = 3, 8
    theta = torch.linspace(-0.01, 0.01, horizon + 1, dtype=torch.float64).repeat(batch, 1)
    omega = torch.linspace(0.02, -0.02, horizon + 1, dtype=torch.float64).repeat(batch, 1)
    dt = torch.full((batch, horizon), 1.0 / 60.0, dtype=torch.float64)
    constants = TorchPhysicalConstants(
        torch.full((batch,), 1.06, dtype=torch.float64),
        torch.full((batch,), 0.33, dtype=torch.float64),
        torch.full((batch,), 9.81, dtype=torch.float64),
        torch.full((batch,), 1.0 / 1.4, dtype=torch.float64),
    )
    pb, vb = rollout_ball_dynamics(
        model, torch.full((batch,), 0.35, dtype=torch.float64),
        torch.tensor([-0.03, 0.0, 0.03], dtype=torch.float64),
        theta, omega, dt, constants,
    )
    (pb.square().mean() + vb.square().mean()).backward()
    assert torch.isfinite(pb).all() and torch.isfinite(vb).all()
    assert model.raw_deadzone_threshold.grad is not None
    assert torch.isfinite(model.raw_deadzone_threshold.grad)
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in model.mlp.parameters())


def test_portable_export_is_bit_exact_for_all_regimes(tmp_path: Path) -> None:
    torch.manual_seed(7)
    model = _model()
    with torch.no_grad():
        for parameter in model.mlp.parameters():
            parameter.normal_(std=0.05)
        model.raw_deadzone_threshold.add_(0.3)
    features = torch.randn(128, 5, dtype=torch.float64)
    features[:, 1] = torch.linspace(-0.1, 0.1, 128)
    expected = model(features)
    path = tmp_path / "weights_v1_5.npz"
    save_portable_weights(model, path, {"fit_dataset_hash": "abc", "best_epoch": 3})
    restored, metadata = load_portable_weights(path)
    actual = restored(features)
    assert metadata["fit_dataset_hash"] == "abc"
    for left, right in zip(expected, actual):
        assert torch.allclose(left, right, atol=1e-12, rtol=0.0)
