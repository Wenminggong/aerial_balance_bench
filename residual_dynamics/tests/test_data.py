from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from residual_dynamics.config import load_config
from residual_dynamics.data import (
    PhysicalConstants,
    _process_segment,
    calculate_geometry,
    nominal_acceleration,
)


CONFIG = Path("residual_dynamics/configs/mlp_5input.yaml")


def test_config_locks_five_input_contract() -> None:
    config = load_config(CONFIG)
    assert config.model.input_names == ("pb", "vb", "theta", "omega", "a_nom")
    assert config.model.hidden_dims == (16, 16)
    assert config.training.seed == 7


def test_geometry_and_nominal_acceleration_are_finite() -> None:
    constants = PhysicalConstants(1.06, 0.9, 0.33, 9.81, 1.0 / 1.4)
    theta = np.asarray([-0.02, 0.0, 0.03])
    vrz = np.asarray([0.1, -0.2, 0.05])
    _, omega = calculate_geometry(theta, vrz, constants)
    acceleration = nominal_acceleration(
        np.asarray([0.2, 0.35, 0.5]), theta, omega, constants
    )
    assert np.isfinite(omega).all()
    assert np.isfinite(acceleration).all()
    assert abs(omega[1] - 0.2 / 1.06) < 1e-12


def test_savgol_segment_recovers_smooth_derivatives() -> None:
    config = load_config(CONFIG)
    time_s = np.arange(180, dtype=np.float64) / 60.0
    frequency = 0.5
    pb = 0.35 + 0.04 * np.sin(frequency * time_s)
    theta = 0.01 * np.sin(0.3 * time_s)
    vrz = 0.05 * np.cos(0.3 * time_s)
    constants = PhysicalConstants(1.06, 0.9, 0.33, 9.81, 1.0 / 1.4)
    piece = _process_segment(time_s, pb, theta, vrz, 0, constants, config)
    assert piece is not None
    expected_v = 0.04 * frequency * np.cos(frequency * piece["time_s"])
    expected_a = -0.04 * frequency**2 * np.sin(frequency * piece["time_s"])
    assert np.sqrt(np.mean((piece["vb"] - expected_v) ** 2)) < 2e-5
    assert np.sqrt(np.mean((piece["ab"] - expected_a) ** 2)) < 2e-5
    assert piece["features"].shape[1] == 5
