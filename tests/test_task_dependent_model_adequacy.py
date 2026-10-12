from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from baselines.acados_nmpc_core import discrete_dynamics_numpy
from scripts.analyze_task_dependent_model_adequacy import (
    ResponseParameters,
    _resolved_controller_cfg,
    central_difference_jacobian,
    critical_alpha_rows,
    dc_offset_envelope,
    exact_response_error_step,
    fopdt_log_sensitivity,
    fopdt_transfer,
    linearity_check_rows,
    load_identification_statistics,
    nonlinear_step,
    paired_metrics,
    periodic_sensitivity_solution,
    predictor_boundary_envelope,
    reference_energy_bandwidth,
    scaled_bounds,
    simulate_reduced_closed_loop,
    uncertainty_vertices,
    worst_case_periodic_rms,
)


IDENTIFICATION_DIR = Path(
    "logs/velocity_response_identification/prbs_velocity_response_fit"
)


@pytest.fixture(scope="module")
def statistics():
    result, _ = load_identification_statistics(
        IDENTIFICATION_DIR / "fopdt_trial_parameters.csv",
        sample_period_s=1.0 / 60.0,
        overall_csv=IDENTIFICATION_DIR / "overall_fits.csv",
    )
    return result


def test_identification_conversion_uses_total_delay_and_up_positive_bias(statistics):
    assert statistics.count == 26
    assert statistics.mean.delay_s == pytest.approx(0.07157692273262654)
    assert statistics.mean.tau_s == pytest.approx(0.1896950666264215)
    assert statistics.mean.gain == pytest.approx(1.0192137321912966)
    assert statistics.mean.bias_mps == pytest.approx(-0.004490963957057612)
    assert statistics.minimum.bias_mps == pytest.approx(-0.007976743738899403)
    assert statistics.maximum.bias_mps == pytest.approx(-0.0007149764069081427)


def test_scaled_uncertainty_box_contract(statistics):
    lower_zero, upper_zero = scaled_bounds(statistics, 0.0)
    np.testing.assert_allclose(lower_zero, statistics.mean.vector())
    np.testing.assert_allclose(upper_zero, statistics.mean.vector())

    lower_one, upper_one = scaled_bounds(statistics, 1.0)
    np.testing.assert_allclose(lower_one, statistics.minimum.vector())
    np.testing.assert_allclose(upper_one, statistics.maximum.vector())
    assert uncertainty_vertices(statistics, 0.0).shape == (1, 4)
    assert uncertainty_vertices(statistics, 1.0).shape == (16, 4)


def test_fopdt_log_sensitivity_matches_central_difference(statistics):
    parameters = statistics.mean
    omega = 2.0 * math.pi / 4.0
    base = fopdt_transfer(parameters, omega)
    analytic = fopdt_log_sensitivity(parameters, omega)
    steps = [1.0e-6, 1.0e-6, 1.0e-6]
    values = parameters.vector()
    for index, epsilon in enumerate(steps):
        plus = values.copy()
        minus = values.copy()
        plus[index] += epsilon
        minus[index] -= epsilon
        numerical = (
            fopdt_transfer(plus, omega) - fopdt_transfer(minus, omega)
        ) / (2.0 * epsilon * base)
        assert numerical == pytest.approx(analytic[index], rel=2.0e-6, abs=2.0e-6)


def test_exact_response_error_recurrence_is_identity(statistics):
    true_parameters = statistics.maximum
    nominal_parameters = statistics.mean
    next_true, next_nominal, recurrence = exact_response_error_step(
        true_velocity=0.12,
        nominal_velocity=0.10,
        true_command=0.3,
        nominal_command=0.25,
        true_parameters=true_parameters,
        nominal_parameters=nominal_parameters,
        sample_period_s=1.0 / 60.0,
    )
    assert next_true - next_nominal == pytest.approx(recurrence, abs=1.0e-14)


def test_fast_scalar_dynamics_matches_acados_numpy_helper(statistics):
    parameters = statistics.mean
    cfg = _resolved_controller_cfg(1.0 / 60.0)
    state = np.asarray([0.31, 0.04, -0.07, 0.11])
    command = -0.2
    fast = nonlinear_step(state, command, parameters, cfg)
    full_state = np.asarray([[*state, command]])
    stage_parameters = np.asarray(
        [[0.0, 0.0, parameters.tau_s, parameters.gain, parameters.bias_mps]]
    )
    expected = discrete_dynamics_numpy(
        full_state,
        np.zeros((1, 1)),
        stage_parameters,
        cfg,
    )[0, :4]
    np.testing.assert_allclose(fast, expected, rtol=1.0e-12, atol=1.0e-12)


def test_matched_composite_rollout_has_zero_paired_mismatch(statistics):
    rollout = simulate_reduced_closed_loop(
        statistics.mean,
        statistics.mean,
        response_aware=True,
        period_s=4.0,
        sample_period_s=1.0 / 60.0,
        center_m=0.35,
        amplitude_m=0.15,
        burn_periods=1,
        score_periods=1,
    )
    metrics = paired_metrics(rollout, rollout, amplitude_m=0.15)
    assert metrics.mismatch_rmse_m == 0.0
    assert metrics.relative_degradation == 0.0
    assert math.isfinite(metrics.tracking_rmse_m)


def test_critical_alpha_interpolation_recovers_two_thresholds():
    rows = [
        {
            "period_s": 4.0,
            "alpha": 0.0,
            "mean_mismatch_rmse_m": 0.0,
            "mean_relative_degradation": 0.0,
        },
        {
            "period_s": 4.0,
            "alpha": 1.0,
            "mean_mismatch_rmse_m": 0.02,
            "mean_relative_degradation": 0.2,
        },
    ]
    result = critical_alpha_rows(rows)[0]
    assert result["alpha_crit_10mm"] == pytest.approx(0.5)
    assert result["alpha_crit_10percent"] == pytest.approx(0.5)


def test_central_difference_reports_step_halving_convergence():
    jacobian, discrepancy = central_difference_jacobian(
        lambda value: np.asarray([value[0] ** 2 + 3.0 * value[1]]),
        np.asarray([2.0, -1.0]),
        np.asarray([1.0e-3, 1.0e-3]),
    )
    np.testing.assert_allclose(jacobian, [[4.0, 3.0]], atol=1.0e-10)
    assert discrepancy < 1.0e-10


def test_dc_offset_vanishes_for_matched_parameters(statistics):
    rows = dc_offset_envelope(statistics, [0.0, 1.0])
    assert rows[0]["velocity_offset_abs_max_mps"] == pytest.approx(0.0, abs=1.0e-15)
    assert rows[1]["velocity_offset_abs_max_mps"] == pytest.approx(0.0044398, rel=2.0e-4)


def test_periodic_sensitivity_and_vertex_rms_have_closed_form():
    sensitivities, spectral_radius = periodic_sensitivity_solution(
        [np.asarray([[0.5]])],
        [np.asarray([[2.0]])],
    )
    assert spectral_radius == pytest.approx(0.5)
    np.testing.assert_allclose(sensitivities, [[[4.0]]])
    assert worst_case_periodic_rms(sensitivities[:, 0, :], [-0.2], [0.1]) == pytest.approx(0.8)


def test_nested_linear_uncertainty_envelopes_are_nondecreasing(statistics):
    center = statistics.mean.vector()
    output_sensitivity = np.asarray(
        [[1.0, -0.5, 0.2, 2.0], [-0.3, 0.7, 1.1, -0.4]],
        dtype=np.float64,
    )
    values = []
    for alpha in (0.0, 0.25, 0.5, 1.0, 2.0):
        lower, upper = scaled_bounds(statistics, alpha)
        values.append(
            worst_case_periodic_rms(
                output_sensitivity,
                lower - center,
                upper - center,
            )
        )
    assert np.all(np.diff(values) >= -1.0e-12)


def test_reference_energy_bandwidth_recovers_sine_frequency():
    dt = 1.0 / 60.0
    period = 4.0
    time_s = np.arange(4 * int(period / dt)) * dt
    reference = np.sin(2.0 * math.pi * time_s / period)
    assert reference_energy_bandwidth(reference, dt) == pytest.approx(2.0 * math.pi / period)


def test_predictor_boundary_envelope_is_zero_without_quantization_or_parameter_error(statistics):
    aligned_mean = ResponseParameters(
        delay_s=4.0 / 60.0,
        tau_s=statistics.mean.tau_s,
        gain=statistics.mean.gain,
        bias_mps=statistics.mean.bias_mps,
    )
    aligned_statistics = type(statistics)(
        count=statistics.count,
        mean=aligned_mean,
        std=statistics.std,
        minimum=aligned_mean,
        p05=aligned_mean,
        median=aligned_mean,
        p95=aligned_mean,
        maximum=aligned_mean,
        pooled_fit=statistics.pooled_fit,
        pooled_rmse_mps=statistics.pooled_rmse_mps,
        pooled_r2=statistics.pooled_r2,
    )
    row = predictor_boundary_envelope(
        aligned_statistics,
        [4.0],
        [0.0],
        sample_period_s=1.0 / 60.0,
        phase_count=8,
    )[0]
    assert row["pb_error_max_m"] == pytest.approx(0.0, abs=1.0e-15)
    assert row["vrz_error_max_mps"] == pytest.approx(0.0, abs=1.0e-15)


def test_linearity_check_requires_numerical_agreement_and_validity():
    periodic = [{
        "period_s": 4.0,
        "alpha": 0.5,
        "linear_paired_rms_m": 0.0011,
        "local_ltv_valid": 1.0,
    }]
    nonlinear = [{
        "period_s": 4.0,
        "alpha": 0.5,
        "mean_mismatch_rmse_m": 0.0010,
        "local_mean_valid": 1.0,
    }]
    row = linearity_check_rows(periodic, nonlinear)[0]
    assert row["relative_difference"] == pytest.approx(0.1)
    assert row["local_linearity_accepted"] == 1.0

    nonlinear[0]["local_mean_valid"] = 0.0
    assert linearity_check_rows(periodic, nonlinear)[0]["local_linearity_accepted"] == 0.0
