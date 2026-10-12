from __future__ import annotations

from pathlib import Path

import numpy as np

from residual_dynamics_v2.config import load_config
from residual_dynamics_v2.data import (
    _window_starts,
    actuation_regimes,
    ood_numpy,
    position_bins,
    velocity_bins,
)


CONFIG = Path(__file__).resolve().parents[1] / "configs" / "rbf_2input_causal_success.yaml"


def test_config_and_cross_session_selectors() -> None:
    config = load_config(CONFIG)
    assert config.rbf.pb_centers_m[0] == 0.0
    assert config.rbf.pb_centers_m[-1] == 0.7
    for split in ("validation", "test", "stress"):
        assignments = [item for item in config.legacy.data.split_assignments if item.split == split]
        assert len(assignments) in {5, 6}
    assert len(
        {item.session for item in config.legacy.data.split_assignments if item.split == "validation"}
    ) == 6


def test_position_velocity_and_actuation_boundaries() -> None:
    pedges = np.arange(0.0, 0.8, 0.1)
    assert position_bins(np.asarray([0.0, 0.1, 0.699, 0.7]), pedges).tolist() == [0, 1, 6, 6]
    vedges = [-1e100, -0.3, -0.15, -0.06, -0.02, 0.02, 0.06, 0.15, 0.3, 1e100]
    assert velocity_bins(np.asarray([-0.31, -0.02, 0.0, 0.02, 0.3]), vedges).tolist() == [0, 4, 4, 4, 8]
    assert actuation_regimes(np.asarray([-0.2, -0.1, 0.0, 0.1, 0.2]), 0.1).tolist() == [-1, 0, 0, 0, 1]


def test_joint_ood_detects_box_inside_outlier() -> None:
    stats = {
        "mean": [0.0, 0.0],
        "std": [1.0, 1.0],
        "lower_quantile": [-3.0, -3.0],
        "upper_quantile": [3.0, 3.0],
        "inverse_correlation": [[5.0, -4.0], [-4.0, 5.0]],
        "mahalanobis_threshold": 1.0,
        "decay_width_std": 1.0,
        "min_weight": 0.0,
    }
    value = ood_numpy(np.asarray([[1.0, -1.0]]), stats)
    assert value["marginal"][0] == 0.0
    assert value["joint_excess"][0] > 0.0
    assert value["weight"][0] < 1.0


def test_invalid_sample_and_segment_boundary_break_windows() -> None:
    arrays = {
        "rbf_valid": np.asarray([True, True, False, True, True, True]),
        "segment_id": np.asarray([0, 0, 0, 1, 1, 1]),
    }
    assert _window_starts(arrays, 2).tolist() == [3]
