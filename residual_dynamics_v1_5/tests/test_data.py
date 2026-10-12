from __future__ import annotations

from pathlib import Path

import numpy as np

from residual_dynamics.data import (
    PhysicalConstants as V1Constants,
    calculate_geometry as v1_geometry,
    nominal_acceleration as v1_nominal,
)
from residual_dynamics_v1_5.config import load_config
from residual_dynamics_v1_5.data import (
    PhysicalConstants,
    _process_segment,
    calculate_geometry,
    load_manifest,
    nominal_acceleration,
    prepare_datasets,
)


CONFIG = Path("residual_dynamics_v1_5/configs/mlp_deadzone.yaml")


def test_real_manifests_have_locked_split_and_normalization_contract() -> None:
    config = load_config(CONFIG)
    prepare_datasets(config)
    fit, test = load_manifest(config, "fit"), load_manifest(config, "test")
    assert fit["split_counts"] == {"train": 20, "validation": 9, "excluded": 1}
    assert test["split_counts"] == {"test": 14, "stress": 5}
    assert fit["normalization"]["source_split"] == "train_dynamic_only"
    extra = [entry for entry in fit["entries"] if entry["group"] == "constant_035_2"]
    assert [entry["split"] for entry in extra].count("train") == 2
    assert [entry["split"] for entry in extra].count("excluded") == 1
    assert test["fit_dataset_hash"] == fit["dataset_hash"]


def test_v1_5_nominal_formulas_equal_v1_to_machine_precision() -> None:
    constants = PhysicalConstants(1.06, 0.9, 0.33, 9.81, 1.0 / 1.4)
    old_constants = V1Constants(1.06, 0.9, 0.33, 9.81, 1.0 / 1.4)
    theta = np.linspace(-0.08, 0.08, 101)
    vrz = np.linspace(-0.3, 0.3, 101)
    pb = np.linspace(0.1, 0.6, 101)
    beta, omega = calculate_geometry(theta, vrz, constants)
    old_beta, old_omega = v1_geometry(theta, vrz, old_constants)
    assert np.max(np.abs(beta - old_beta)) < 1e-12
    assert np.max(np.abs(omega - old_omega)) < 1e-12
    assert np.max(np.abs(
        nominal_acceleration(pb, theta, omega, constants)
        - v1_nominal(pb, theta, old_omega, old_constants)
    )) < 1e-12


def test_masks_expand_laser_jump_and_separate_modes() -> None:
    config = load_config(CONFIG)
    time_s = np.arange(300, dtype=np.float64) / 60.0
    pb = np.full_like(time_s, 0.35)
    moving = time_s >= 1.25
    pb[moving] += 0.06 * np.sin(np.pi * (time_s[moving] - 1.25))
    pb[180:] += 0.03
    theta = 0.005 * np.sin(0.5 * time_s)
    vrz = 0.02 * np.cos(0.5 * time_s)
    constants = PhysicalConstants(1.06, 0.9, 0.33, 9.81, 1.0 / 1.4)
    piece = _process_segment(time_s, pb, theta, vrz, 0, constants, False, config)
    assert piece is not None
    jump_ids = np.flatnonzero(piece["jump_mask"])
    assert jump_ids.size >= 2 * config.data.jump_dilation_samples + 1
    assert np.all(piece["invalid_mask"][jump_ids])
    assert piece["static_mask"].any()
    assert piece["dynamic_mask"].any()
    assert not np.any(piece["static_mask"] & piece["dynamic_mask"])
    assert np.array_equal(
        piece["transition_mask"],
        piece["valid_mask"] & ~piece["static_mask"] & ~piece["dynamic_mask"],
    )
