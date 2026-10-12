from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from residual_dynamics_v2.evaluation import (
    command_queue_alignment,
    shadow_full_state_ood,
)


def test_command_queue_alignment_is_measured() -> None:
    aligned = SimpleNamespace(
        arrays={
            "response_command": np.asarray([0.1, -0.2]),
            "pending_commands": np.asarray([[0.1, 0.0], [-0.2, 0.1]]),
        }
    )
    shifted = SimpleNamespace(
        arrays={
            "response_command": np.asarray([0.1, -0.1]),
            "pending_commands": np.asarray([[0.1, 0.0], [-0.2, 0.1]]),
        }
    )
    assert command_queue_alignment([aligned])["passed"]
    assert not command_queue_alignment([shifted])["passed"]


def test_shadow_full_state_ood_is_diagnostic_and_finite() -> None:
    normalization = {
        "mean": [0.0] * 5,
        "std": [1.0] * 5,
        "ood": {
            "lower": [-2.0] * 5,
            "upper": [2.0] * 5,
            "inverse_correlation": np.eye(5).tolist(),
            "mahalanobis_radius": 3.0,
        },
    }
    result = shadow_full_state_ood(
        np.asarray([[0.0] * 5, [4.0, 0.0, 0.0, 0.0, 0.0]]), normalization
    )
    assert result["diagnostic_only"]
    assert result["all_finite"]
    assert result["fraction_outside"] == 0.5
