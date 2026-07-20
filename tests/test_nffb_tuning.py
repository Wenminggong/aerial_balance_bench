"""Tests for NFFB sine-tuning rollout analysis."""

from __future__ import annotations

import math

import numpy as np
import yaml

from utils.nffb_tuning import (
    SATURATION_FIELDS,
    aggregate_episode_metrics,
    analyze_rollout,
    evaluate_acceptance,
    tuning_rank_key,
)
from scripts.tune_nffb_sine import (
    Candidate,
    _best_stress,
    _effective_policy_config,
)


def _make_sine_run(
    tmp_path,
    *,
    amplitude_gain: float = 1.0,
    phase_error_deg: float = 0.0,
    terminate: bool = False,
    boundary_violation: bool = False,
):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    step_dt = 1.0 / 60.0
    samples = 1200
    time_s = np.arange(samples) * step_dt
    period = 4.0
    amplitude = 0.1
    center = 0.35
    angular_frequency = 2.0 * math.pi / period
    pg = center + amplitude * np.sin(angular_frequency * time_s)
    pb = center + amplitude * amplitude_gain * np.sin(
        angular_frequency * time_s + math.radians(phase_error_deg)
    )
    if boundary_violation:
        pb[100] = 0.0

    fields = np.asarray(
        ["pb", "vb", "ab", "theta", "omega", "alpha", "drz", "vrz", "arz", "pg", "a_prev"]
    )
    observations = np.zeros((samples, 1, fields.size), dtype=np.float32)
    observations[:, 0, 0] = pb
    observations[:, 0, 9] = pg
    terminated = np.zeros((samples, 1), dtype=bool)
    truncated = np.zeros_like(terminated)
    if terminate:
        terminated[-1, 0] = True
    else:
        truncated[-1, 0] = True

    payload = {
        "observations": observations,
        "observation_fields": fields,
        "actions": np.zeros((samples, 1, 1), dtype=np.float32),
        "terminated": terminated,
        "truncated": truncated,
        "step_trajectory_center": np.full((samples, 1), center, dtype=np.float32),
        "step_trajectory_amplitude": np.full((samples, 1), amplitude, dtype=np.float32),
        "step_trajectory_period": np.full((samples, 1), period, dtype=np.float32),
        "step_trajectory_phase": np.zeros((samples, 1), dtype=np.float32),
        "policy_theta_d": np.zeros((samples, 1), dtype=np.float32),
        "policy_theta_star": np.zeros((samples, 1), dtype=np.float32),
        "policy_velocity_command": np.zeros((samples, 1), dtype=np.float32),
        "policy_omega_d": np.zeros((samples, 1), dtype=np.float32),
    }
    for field in SATURATION_FIELDS:
        payload[field] = np.zeros((samples, 1), dtype=bool)
    np.savez_compressed(run_dir / "rollout.npz", **payload)
    with open(run_dir / "resolved_run.yaml", "w", encoding="utf-8") as stream:
        yaml.safe_dump(
            {
                "step_dt": step_dt,
                "constraints": {"omega_max": 0.5},
            },
            stream,
        )
    return run_dir


def test_analyze_rollout_recovers_sine_gain_and_phase(tmp_path):
    run_dir = _make_sine_run(
        tmp_path,
        amplitude_gain=1.05,
        phase_error_deg=8.0,
    )

    rows, metrics = analyze_rollout(run_dir)

    assert len(rows) == 1
    assert rows[0]["amplitude_gain"] == pytest.approx(1.05, abs=1.0e-4)
    assert rows[0]["phase_error_deg"] == pytest.approx(8.0, abs=1.0e-3)
    assert not rows[0]["boundary_violation"]
    assert metrics["episode_count"] == 1
    assert metrics["p95_abs_phase_error_deg"] == pytest.approx(8.0, abs=1.0e-3)
    assert not metrics["hard_failure"]


def test_termination_and_boundary_are_hard_failures(tmp_path):
    run_dir = _make_sine_run(
        tmp_path,
        terminate=True,
        boundary_violation=True,
    )

    _, metrics = analyze_rollout(run_dir)

    assert metrics["termination_count"] == 1
    assert metrics["boundary_violation_count"] == 1
    assert metrics["hard_failure"]


def test_rank_prefers_feasible_then_normalized_error():
    feasible = {
        "hard_failure": False,
        "p95_nrmse": 0.3,
        "mean_rmse": 0.02,
        "p95_abs_phase_error_deg": 10.0,
        "mean_abs_gain_error": 0.1,
        "mean_action_rms": 0.01,
        "bandwidth_cost": 20.0,
    }
    infeasible = {**feasible, "hard_failure": True, "p95_nrmse": 0.01}
    better = {**feasible, "p95_nrmse": 0.2}

    assert tuning_rank_key(feasible) < tuning_rank_key(infeasible)
    assert tuning_rank_key(better) < tuning_rank_key(feasible)


def test_acceptance_reports_each_gate():
    selected = {
        "episode_count": 500,
        "nonfinite_count": 0,
        "termination_count": 0,
        "boundary_violation_count": 0,
        "mean_mae": 0.01,
        "mean_rmse": 0.015,
        "p95_rmse": 0.025,
        "p95_nrmse": 0.25,
        "p95_maxe": 0.05,
        "max_full_saturation_rate": 0.0,
        "max_steady_saturation_rate": 0.0,
    }
    baseline = {"mean_rmse": 0.04}
    stress = {
        "min_amplitude_gain": 0.95,
        "max_amplitude_gain": 1.05,
        "p95_abs_phase_error_deg": 8.0,
        "p95_steady_nrmse": 0.15,
    }
    thresholds = {
        "minimum_episodes": 500,
        "mean_mae": 0.015,
        "mean_rmse": 0.020,
        "p95_rmse": 0.030,
        "p95_nrmse": 0.30,
        "p95_maxe": 0.060,
        "rmse_improvement": 0.50,
        "full_saturation_rate": 0.05,
        "steady_saturation_rate": 0.01,
        "stress_gain_min": 0.90,
        "stress_gain_max": 1.10,
        "stress_phase_error_deg": 10.0,
        "stress_steady_nrmse": 0.20,
        "feedforward_rmse_improvement": 0.10,
        "feedforward_phase_improvement_deg": 5.0,
    }

    result = evaluate_acceptance(
        selected,
        baseline,
        stress,
        thresholds,
        feedforward_rmse_improvement=0.12,
        feedforward_phase_improvement_deg=2.0,
    )

    assert result["passed"]
    assert all(result["checks"].values())


def test_effective_policy_config_updates_top_level_policy():
    base = {
        "policy_name": "nffb",
        "outer_loop": {"natural_frequency": 1.0},
        "command_filter": {"natural_frequency": 2.5},
        "inner_loop": {"k_theta": 4.0},
        "constraints": {"omega_max": 0.5},
    }
    candidate = Candidate(1.35, 1.0, 0.0, 16.0, 1.0, 6.0, 0.5)

    effective = _effective_policy_config(base, candidate)

    assert "nffb_policy" not in effective
    assert effective["policy_name"] == "nffb_sine_phase0_acc5"
    assert effective["outer_loop"]["natural_frequency"] == pytest.approx(1.35)
    assert effective["command_filter"]["natural_frequency"] == pytest.approx(16.0)
    assert effective["inner_loop"]["k_theta"] == pytest.approx(6.0)


def test_stress_selection_prefers_gate_passing_candidate():
    thresholds = {
        "stress_gain_min": 0.9,
        "stress_gain_max": 1.1,
        "stress_phase_error_deg": 10.0,
        "stress_steady_nrmse": 0.2,
    }
    lower_error_but_failed_gate = {
        "hard_failure": False,
        "termination_count": 0,
        "boundary_violation_count": 0,
        "nonfinite_count": 0,
        "min_amplitude_gain": 1.0,
        "max_amplitude_gain": 1.11,
        "p95_abs_phase_error_deg": 1.0,
        "p95_steady_nrmse": 0.05,
        "p95_nrmse": 0.05,
        "mean_rmse": 0.01,
    }
    passing = {
        **lower_error_but_failed_gate,
        "max_amplitude_gain": 1.09,
        "p95_nrmse": 0.08,
        "mean_rmse": 0.015,
    }
    failed_candidate = Candidate(1.4, 1.0, 0.0, 14.0, 1.0, 8.0, 0.5)
    passing_candidate = Candidate(1.35, 1.0, 0.0, 16.0, 1.0, 6.0, 0.5)

    selected = _best_stress(
        [
            (failed_candidate, lower_error_but_failed_gate),
            (passing_candidate, passing),
        ],
        thresholds,
    )

    assert selected[0][0] == passing_candidate


# Imported late so the synthetic-data helpers above remain visually compact.
import pytest  # noqa: E402
