from __future__ import annotations

import numpy as np

from residual_dynamics.closed_loop_validation import (
    _cluster_bootstrap_ratio,
    _interpolate_trial_array,
)
from residual_dynamics.data import PhysicalConstants, TrialData


def test_cluster_bootstrap_ratio_reports_exact_pooled_estimate() -> None:
    result = _cluster_bootstrap_ratio(
        [(4.0, 1.0, 10), (9.0, 4.0, 20)],
        bootstrap_samples=1000,
        seed=7,
    )

    assert result["estimate"] == 1.0 - 5.0 / 13.0
    assert result["bootstrap_probability_improvement"] == 1.0
    assert result["ci95_low"] > 0.0


def test_interpolate_trial_array_never_crosses_segment_boundary() -> None:
    trial = TrialData(
        trial_id="trial",
        group="group",
        session="session",
        split="heldout",
        reference_type="constant",
        source_file="samples.csv",
        arrays={
            "time_s": np.asarray([0.0, 1.0, 3.0, 4.0]),
            "segment_id": np.asarray([0, 0, 1, 1]),
            "smooth_ab": np.asarray([0.0, 1.0, 30.0, 40.0]),
        },
        constants=PhysicalConstants(1.06, 0.9, 0.33, 9.81, 1.0 / 1.4),
    )

    values, valid = _interpolate_trial_array(
        trial, np.asarray([0.5, 2.0, 3.5]), "smooth_ab"
    )

    np.testing.assert_allclose(values[valid], np.asarray([0.5, 35.0]))
    np.testing.assert_array_equal(valid, np.asarray([True, False, True]))
