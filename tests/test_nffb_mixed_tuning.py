"""Tests for NFFB mixed-reference coarse tuning."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import yaml

import scripts.tune_nffb_mixed as mixed_tuning_script
from scripts.tune_nffb_mixed import (
    Candidate,
    TuningSession,
    _grid_candidates,
    _run_grid_search,
    _validate_grid_policy_contract,
)
from utils.nffb_tuning import (
    SATURATION_FIELDS,
    aggregate_mixed_episode_metrics,
    analyze_rollout,
    mixed_tuning_rank_key,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GRID_CONFIG_PATH = (
    PROJECT_ROOT / "baselines/configs/nffb_mixed_grid_tuning.yaml"
)


def _load_grid_config():
    with open(GRID_CONFIG_PATH, encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def _make_mixed_run(tmp_path):
    run_dir = tmp_path / "mixed_run"
    run_dir.mkdir()
    samples = 60
    type_ids = np.asarray([3, 4, 5], dtype=np.float32)
    errors = np.asarray([0.01, 0.02, 0.03], dtype=np.float32)
    fields = np.asarray(
        [
            "pb",
            "vb",
            "ab",
            "theta",
            "omega",
            "alpha",
            "drz",
            "vrz",
            "arz",
            "pg",
            "a_prev",
        ]
    )
    observations = np.zeros((samples, 3, fields.size), dtype=np.float32)
    observations[:, :, 0] = 0.35 + errors
    observations[:, :, 9] = 0.35
    terminated = np.zeros((samples, 3), dtype=bool)
    truncated = np.zeros_like(terminated)
    truncated[-1] = True
    payload = {
        "observations": observations,
        "observation_fields": fields,
        "actions": np.zeros((samples, 3, 1), dtype=np.float32),
        "terminated": terminated,
        "truncated": truncated,
        "step_trajectory_type_id": np.broadcast_to(type_ids, (samples, 3)),
        "step_trajectory_center": np.full((samples, 3), 0.35, dtype=np.float32),
        "step_trajectory_amplitude": np.zeros((samples, 3), dtype=np.float32),
        "step_trajectory_period": np.ones((samples, 3), dtype=np.float32),
        "step_trajectory_phase": np.zeros((samples, 3), dtype=np.float32),
        "policy_theta_d": np.zeros((samples, 3), dtype=np.float32),
        "policy_theta_star": np.zeros((samples, 3), dtype=np.float32),
        "policy_velocity_command": np.zeros((samples, 3), dtype=np.float32),
        "policy_omega_d": np.zeros((samples, 3), dtype=np.float32),
    }
    for field in SATURATION_FIELDS:
        payload[field] = np.zeros((samples, 3), dtype=bool)
    np.savez_compressed(run_dir / "rollout.npz", **payload)
    with open(run_dir / "resolved_run.yaml", "w", encoding="utf-8") as stream:
        yaml.safe_dump(
            {
                "step_dt": 1.0 / 60.0,
                "constraints": {"omega_max": 0.5},
                "trajectory_type_to_id": {
                    "sine": 0,
                    "triangle": 1,
                    "trapezoid": 2,
                    "constant": 3,
                    "random_b_spline": 4,
                    "random_ramp_dwell": 5,
                },
            },
            stream,
        )
    return run_dir


def test_mixed_analysis_maps_types_and_uses_absolute_errors(tmp_path):
    rows, _ = analyze_rollout(_make_mixed_run(tmp_path))

    metrics = aggregate_mixed_episode_metrics(
        rows,
        required_types=("constant", "random_b_spline", "random_ramp_dwell"),
        min_episodes_per_type=1,
    )

    assert [row["trajectory_type"] for row in rows] == [
        "constant",
        "random_b_spline",
        "random_ramp_dwell",
    ]
    assert metrics["constant_episode_count"] == 1
    assert metrics["random_b_spline_episode_count"] == 1
    assert metrics["random_ramp_dwell_episode_count"] == 1
    assert metrics["worst_type_p95_rmse"] == pytest.approx(0.03, abs=1.0e-6)
    assert metrics["max_tail_saturation_rate"] == 0.0
    assert not metrics["hard_failure"]


def test_missing_required_type_is_a_hard_failure(tmp_path):
    rows, _ = analyze_rollout(_make_mixed_run(tmp_path))

    metrics = aggregate_mixed_episode_metrics(
        rows[:-1],
        required_types=("constant", "random_b_spline", "random_ramp_dwell"),
        min_episodes_per_type=1,
    )

    assert metrics["coverage_shortfall"] == 1
    assert metrics["coverage_failure"]
    assert metrics["hard_failure"]


def test_mixed_rank_prefers_feasibility_then_worst_family_error():
    feasible = {
        "hard_failure": False,
        "coverage_shortfall": 0,
        "worst_type_p95_rmse": 0.04,
        "mean_rmse": 0.02,
        "p95_maxe": 0.08,
        "max_full_saturation_rate": 0.0,
        "max_tail_saturation_rate": 0.0,
        "mean_action_rms": 0.01,
        "bandwidth_cost": 20.0,
    }
    infeasible = {**feasible, "hard_failure": True, "worst_type_p95_rmse": 0.001}
    better = {**feasible, "worst_type_p95_rmse": 0.03}

    assert mixed_tuning_rank_key(feasible) < mixed_tuning_rank_key(infeasible)
    assert mixed_tuning_rank_key(better) < mixed_tuning_rank_key(feasible)


def test_mixed_rank_prefers_less_severe_candidate_when_all_fail():
    base = {
        "hard_failure": True,
        "coverage_shortfall": 0,
        "nonfinite_count": 0,
        "worst_type_p95_rmse": 0.2,
        "mean_rmse": 0.1,
        "p95_maxe": 0.3,
        "max_full_saturation_rate": 0.06,
        "max_tail_saturation_rate": 0.06,
        "mean_action_rms": 0.01,
        "bandwidth_cost": 10.0,
    }
    one_boundary = {
        **base,
        "termination_count": 0,
        "boundary_violation_count": 1,
    }
    many_terminations = {
        **base,
        "termination_count": 20,
        "boundary_violation_count": 10,
        "worst_type_p95_rmse": 0.01,
    }

    assert mixed_tuning_rank_key(one_boundary) < mixed_tuning_rank_key(
        many_terminations
    )


def test_run_config_preserves_explicit_response_compensation(tmp_path):
    config = {
        "base_run_config": "baselines/configs/nffb_unified_tracking_response_compensation_eval.yaml",
        "base_env_config": "environments/configs/unified_tracking_mixed.yaml",
        "base_policy_config": "baselines/configs/nffb.yaml",
        "sine_policy_config": "baselines/configs/nffb_sine_phase0_acc5.yaml",
        "output_root": str(tmp_path / "logs"),
        "session_name": "test",
        "output_policy_config": str(tmp_path / "selected.yaml"),
        "required_types": ["constant", "random_b_spline", "random_ramp_dwell"],
        "policy_defaults": {
            "state_predictor": {
                "enabled": False,
                "delay_step": 0,
                "velocity_response_enabled": True,
                "velocity_response_tau_s": 0.155,
                "velocity_response_gain": 0.86,
                "velocity_response_bias": -0.0035,
            },
            "velocity_response_compensation": {
                "enabled": True,
                "parameter_source": "state_predictor",
            },
        },
        "acceptance": {"full_saturation_rate": 0.05, "tail_saturation_rate": 0.01},
    }
    session = TuningSession(
        config,
        python="python3",
        resume=True,
        dry_run=True,
    )
    candidate = Candidate(1.2, 1.0, 0.0, 10.0, 1.0, 8.0, 0.5)

    run_config_path = session._run_config(  # noqa: SLF001
        candidate,
        trial_id="merge_test",
        episodes=3,
    )
    with open(run_config_path, encoding="utf-8") as stream:
        run_config = yaml.safe_load(stream)

    predictor = run_config["policy_overrides"]["state_predictor"]
    compensation = run_config["policy_overrides"]["velocity_response_compensation"]
    assert Path(run_config["env_config"]) == (
        PROJECT_ROOT / "environments/configs/unified_tracking_mixed.yaml"
    )
    assert predictor["velocity_response_tau_s"] == pytest.approx(0.155)
    assert predictor["velocity_response_gain"] == pytest.approx(0.86)
    assert predictor["velocity_response_bias"] == pytest.approx(-0.0035)
    assert compensation == {"enabled": True, "parameter_source": "state_predictor"}
    assert run_config["policy_overrides"]["outer_loop"]["natural_frequency"] == 1.2

    effective = session.effective_policy_config(candidate)
    assert effective["policy_name"] == "nffb_unified_mixed_coarse"
    assert effective["velocity_response_compensation"]["enabled"]
    assert effective["state_predictor"]["velocity_response_tau_s"] == pytest.approx(0.155)


def test_grid_config_expands_to_expected_27_point_neighborhood():
    config = _load_grid_config()

    center, candidates = _grid_candidates(config)

    assert len(candidates) == 27
    assert len(set(candidates)) == 27
    assert center in candidates
    assert center == Candidate(0.68, 1.0, 0.0, 14.0, 1.0, 6.0, 0.5, True)
    assert {candidate.outer_omega for candidate in candidates} == {0.55, 0.68, 0.8}
    assert {candidate.k_theta for candidate in candidates} == {4.5, 5.25, 6.0}
    assert {candidate.integral_pole for candidate in candidates} == {0.0, 0.05, 0.1}
    assert {candidate.filter_omega for candidate in candidates} == {14.0}
    assert {candidate.omega_max for candidate in candidates} == {0.5}
    assert all(candidate.feedforward for candidate in candidates)


def test_grid_rejects_excessive_count_missing_center_and_compensation():
    config = _load_grid_config()
    too_many = deepcopy(config)
    too_many["search"]["parameters"]["outer_omega"] = [
        0.50 + 0.01 * index for index in range(11)
    ]
    with pytest.raises(ValueError, match="99 combinations"):
        _grid_candidates(too_many)

    missing_center = deepcopy(config)
    missing_center["search"]["parameters"]["outer_omega"] = [0.55, 0.80]
    with pytest.raises(ValueError, match="include initial_candidate"):
        _grid_candidates(missing_center)

    compensation_enabled = deepcopy(config)
    compensation_enabled["policy_defaults"]["velocity_response_compensation"][
        "enabled"
    ] = True
    with pytest.raises(ValueError, match="compensation disabled"):
        _validate_grid_policy_contract(compensation_enabled)


def test_grid_uses_one_budget_and_reports_best_even_when_all_fail(
    tmp_path,
    monkeypatch,
):
    config = _load_grid_config()
    config["output_root"] = str(tmp_path / "logs")
    config["session_name"] = "grid_test"
    session = TuningSession(
        config,
        python="python3",
        resume=True,
        dry_run=False,
    )
    calls = []

    def fake_run_trial(candidate, **kwargs):
        calls.append((candidate, kwargs))
        type_metrics = {
            name: {"mean_rmse": 0.1, "p95_rmse": 0.2}
            for name in config["required_types"]
        }
        metrics = {
            "candidate_id": candidate.identifier,
            "bandwidth_cost": candidate.bandwidth_cost,
            "hard_failure": True,
            "coverage_failure": False,
            "coverage_shortfall": 0,
            "termination_count": 0,
            "boundary_violation_count": 0,
            "nonfinite_count": 0,
            "max_full_saturation_rate": 0.01,
            "max_tail_saturation_rate": 0.02,
            "worst_type_p95_rmse": 0.2,
            "mean_rmse": 0.1,
            "p95_maxe": 0.3,
            "mean_action_rms": 0.01,
            "type_metrics": type_metrics,
        }
        session.results.append(metrics)
        session.write_leaderboard()
        return metrics

    monkeypatch.setattr(session, "run_trial", fake_run_trial)

    _run_grid_search(session, "all")

    assert len(calls) == 27
    assert {
        (
            kwargs["seed"],
            kwargs["num_envs"],
            kwargs["episodes"],
            kwargs["min_episodes_per_type"],
        )
        for _, kwargs in calls
    } == {(668, 500, 500, 130)}
    with open(session.session_root / "best_selection.yaml", encoding="utf-8") as stream:
        selection = yaml.safe_load(stream)
    assert not selection["acceptance"]["passed"]
    assert selection["candidate"]["outer_omega"] == pytest.approx(0.55)
    assert selection["candidate"]["k_theta"] == pytest.approx(4.5)
    assert (session.session_root / "best_observed_policy.yaml").is_file()
    assert (session.session_root / "grid_manifest.yaml").is_file()
    assert (session.session_root / "grid_summary.csv").is_file()
    assert (session.session_root / "leaderboard.csv").is_file()


def test_coarse_strategy_remains_the_default(monkeypatch):
    session = type(
        "DrySession",
        (),
        {"config": {"search": {}}, "dry_run": True},
    )()
    calls = []
    monkeypatch.setattr(
        mixed_tuning_script,
        "_dry_run_candidates",
        lambda value: calls.append(value),
    )

    mixed_tuning_script.run_search(session, "all")

    assert calls == [session]
