"""Tests for acados NMPC first-episode analysis and sequential objective search."""

from __future__ import annotations

import numpy as np
import pytest
import yaml

from scripts.tune_acados_nmpc_objective import (
    DEFAULT_CONFIG,
    _build_run_config,
    _initialize_or_load_session,
    _launch_or_reuse,
    _promote_policy,
    _propose_next,
    _validate_tuning_config,
)
from utils.acados_nmpc_tuning import (
    ObjectiveCandidate,
    analyze_acados_rollout,
    compare_screen_candidate,
    evaluate_validation,
)


BASELINE = ObjectiveCandidate(
    (5.0, 0.5, 0.1, 0.05, 0.05, 0.5, 1.0),
    (25.0, 2.5, 0.5, 0.25, 0.25, 0.5),
)


def _make_mixed_run(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    steps = 8
    num_envs = 3
    fields = np.asarray(
        ["pb", "vb", "ab", "theta", "omega", "alpha", "drz", "vrz", "arz", "pg", "a_prev"]
    )
    observations = np.zeros((steps, num_envs, fields.size), dtype=np.float32)
    observations[..., 0] = 0.35
    observations[..., 9] = 0.35
    observations[..., 1] = 0.02
    observations[..., 3] = 0.01
    observations[..., 7] = 0.03
    # This belongs to the post-autoreset second episode and must be ignored.
    observations[4:, 0, 0] = 2.0
    actions = np.full((steps, num_envs, 1), 0.01, dtype=np.float32)
    terminated = np.zeros((steps, num_envs), dtype=bool)
    truncated = np.zeros_like(terminated)
    truncated[3, :] = True
    terminated[7, 0] = True
    trajectory_type = np.tile(np.asarray([[3, 4, 5]], dtype=np.float32), (steps, 1))
    solver_success = np.ones((steps, num_envs), dtype=bool)
    fallback = np.zeros((steps, num_envs), dtype=bool)
    slack = np.zeros((steps, num_envs), dtype=np.float32)
    np.savez_compressed(
        run_dir / "rollout.npz",
        observations=observations,
        observation_fields=fields,
        actions=actions,
        terminated=terminated,
        truncated=truncated,
        step_trajectory_type_id=trajectory_type,
        step_command_z=np.zeros((steps, num_envs), dtype=np.float32),
        acados_nmpc_solver_success=solver_success,
        acados_nmpc_fallback=fallback,
        acados_nmpc_max_slack=slack,
        policy_act_wall_time=np.full(steps, 0.001, dtype=np.float64),
    )
    with open(run_dir / "resolved_run.yaml", "w", encoding="utf-8") as stream:
        yaml.safe_dump(
            {
                "control_step_dt": 0.1,
                "resolved_policy_config": {"constraints": {"max_acc": 5.0}},
            },
            stream,
        )
    return run_dir


def test_analysis_uses_only_first_episode_and_balances_types(tmp_path):
    rows, metrics = analyze_acados_rollout(
        _make_mixed_run(tmp_path), minimum_per_type=1
    )

    assert len(rows) == 3
    assert {row["trajectory_type"] for row in rows} == {
        "constant",
        "random_b_spline",
        "random_ramp_dwell",
    }
    assert metrics["episode_count"] == 3
    assert metrics["coverage_ok"]
    assert metrics["boundary_violation_count"] == 0
    assert metrics["balanced_mean_rmse"] == pytest.approx(0.0)
    assert metrics["objective_score"] == pytest.approx(0.0)


def test_candidate_is_full_strict_and_scales_stage_terminal_pair():
    velocity = BASELINE.scale_group("velocity", 8.0)

    assert velocity.stage_weights[1] == pytest.approx(4.0)
    assert velocity.terminal_weights[1] == pytest.approx(20.0)
    with pytest.raises(ValueError, match="unknown fields"):
        ObjectiveCandidate.from_mapping(
            {**BASELINE.as_dict(), "solver": {"n_horizon": 10}}
        )
    with pytest.raises(ValueError, match="finite"):
        ObjectiveCandidate((5, np.nan, 1, 1, 1, 1, 1), (1, 1, 1, 1, 1, 1))
    with pytest.raises(ValueError, match=r"\[0.001, 1000\]"):
        BASELINE.scale_group("velocity", 5000.0)


def _screen_metrics(score=1.0, action_rms=0.2):
    return {
        "coverage_ok": True,
        "nonfinite_count": 0,
        "termination_count": 0,
        "boundary_violation_count": 0,
        "solver_success_rate": 1.0,
        "fallback_rate": 0.0,
        "action_saturation_rate": 0.0,
        "tail_action_saturation_rate": 0.0,
        "objective_score": score,
        "action_rms": action_rms,
        "per_type": {
            name: {"p90_rmse": 1.0, "mean_tail_rmse": 1.0}
            for name in ("constant", "random_b_spline", "random_ramp_dwell")
        },
    }


def test_screen_acceptance_requires_improvement_or_lower_action_tiebreak():
    incumbent = _screen_metrics()

    assert compare_screen_candidate(_screen_metrics(0.96), incumbent)["accepted"]
    assert compare_screen_candidate(_screen_metrics(0.99, 0.1), incumbent)["accepted"]
    saturated = _screen_metrics(0.5)
    saturated["action_saturation_rate"] = 0.21
    decision = compare_screen_candidate(saturated, incumbent)
    assert not decision["accepted"]
    assert decision["hard_rejection"]


def test_first_automatic_recommendation_is_velocity_damping_x8():
    metrics = _screen_metrics()
    state = {
        "protocol_cursor": 0,
        "incumbent_candidate": BASELINE.as_dict(),
        "incumbent_candidate_id": BASELINE.identifier,
        "screen_trials": [
            {
                "label": "baseline",
                "candidate_id": BASELINE.identifier,
                "candidate": BASELINE.as_dict(),
                "accepted": True,
                "metrics": metrics,
            }
        ],
    }

    proposal = _propose_next(state)

    assert proposal["label"] == "velocity_x8"
    candidate = ObjectiveCandidate.from_mapping(proposal["candidate"])
    assert candidate.stage_weights[1] == pytest.approx(4.0)
    assert candidate.terminal_weights[1] == pytest.approx(20.0)
    assert state["protocol_cursor"] == 1


def test_session_refuses_environment_hash_change(tmp_path):
    env_path = tmp_path / "env.yaml"
    env_path.write_text("env:\n  seed: 666\n", encoding="utf-8")
    policy_path = tmp_path / "policy.yaml"
    with open(policy_path, "w", encoding="utf-8") as stream:
        yaml.safe_dump({"acados_nmpc_policy": {"objective": BASELINE.as_dict()}}, stream)
    config = {
        "name": "test",
        "baseline": BASELINE.as_dict(),
    }
    output = tmp_path / "session"
    _initialize_or_load_session(
        config, tmp_path / "tuning.yaml", env_path, policy_path, output
    )
    env_path.write_text("env:\n  seed: 667\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="changed"):
        _initialize_or_load_session(
            config, tmp_path / "tuning.yaml", env_path, policy_path, output
        )


def test_validation_includes_statistical_and_timing_gates():
    baseline = {"balanced_mean_rmse": 1.0, "worst_type_p90_rmse": 1.0}
    validation = {
        "episode_count": 96,
        "trajectory_coverage": {
            "constant": 32,
            "random_b_spline": 32,
            "random_ramp_dwell": 32,
        },
        "nonfinite_count": 0,
        "termination_count": 0,
        "boundary_violation_count": 0,
        "solver_success_rate": 1.0,
        "fallback_rate": 0.0,
        "action_saturation_rate": 0.05,
        "tail_action_saturation_rate": 0.01,
        "slack_active_rate": 0.0,
        "max_slack": 0.0,
        "balanced_mean_rmse": 0.6,
        "worst_type_p90_rmse": 0.75,
        "constant_steady_mae": 0.04,
    }

    passed = evaluate_validation(validation, baseline, timing_p99_s=0.01)
    failed_timing = evaluate_validation(validation, baseline, timing_p99_s=0.02)

    assert passed["passed"]
    assert not failed_timing["passed"]
    assert not failed_timing["checks"]["single_env_timing_p99"]


def test_generated_screen_run_is_fixed_length_and_objective_only(tmp_path):
    run_config = _build_run_config(
        env_path=tmp_path / "env.yaml",
        policy_path=tmp_path / "policy.yaml",
        candidate=BASELINE,
        output_root=tmp_path / "output",
        run_name="trial",
        seed=666,
        target_episodes=48,
        max_steps=600,
        batch_threads=8,
    )

    assert run_config["runner"]["max_steps"] == 600
    assert run_config["runner"]["stop_on_target_episodes"] is False
    assert set(run_config["policy_overrides"]) == {"objective", "solver"}
    assert run_config["policy_overrides"]["solver"] == {
        "num_threads_in_batch_solve": 8
    }


def test_resume_reuses_rollout_without_subprocess(tmp_path, monkeypatch):
    run_dir = tmp_path / "runs" / "done"
    run_dir.mkdir(parents=True)
    (run_dir / "rollout.npz").write_bytes(b"complete")

    def unexpected_run(*args, **kwargs):
        raise AssertionError("resume must not launch an already completed candidate")

    monkeypatch.setattr("scripts.tune_acados_nmpc_objective.subprocess.run", unexpected_run)
    available = _launch_or_reuse(
        run_config={"runner": {}},
        generated_config_path=tmp_path / "generated.yaml",
        run_dir=run_dir,
        stdout_path=tmp_path / "stdout.log",
        python="python",
        num_envs=48,
        resume=True,
        dry_run=False,
    )

    assert available


def test_protocol_config_and_evaluator_default_are_explicit():
    config = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    _validate_tuning_config(config)
    changed = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    changed["fixed"]["n_horizon"] = 29
    with pytest.raises(ValueError, match="n_horizon=30"):
        _validate_tuning_config(changed)

    project_root = DEFAULT_CONFIG.parents[2]
    eval_config = yaml.safe_load(
        (project_root / "baselines/configs/acados_nmpc_unified_tracking_eval.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert eval_config["runner"]["stop_on_target_episodes"] is True


def test_promotion_changes_only_two_weight_lines(tmp_path):
    policy_path = tmp_path / "policy.yaml"
    policy_path.write_text(
        "acados_nmpc_policy:\n"
        "  name: keep_me\n"
        "  objective:\n"
        "    stage_weights: [5, 0.5, 0.1, 0.05, 0.05, 0.5, 1]\n"
        "    terminal_weights: [25, 2.5, 0.5, 0.25, 0.25, 0.5]\n"
        "  solver:\n"
        "    n_horizon: 30\n",
        encoding="utf-8",
    )
    candidate = BASELINE.scale_group("velocity", 8.0)

    _promote_policy(policy_path, candidate)

    promoted = policy_path.read_text(encoding="utf-8")
    assert "name: keep_me" in promoted
    assert "stage_weights: [5, 4, 0.1, 0.05, 0.05, 0.5, 1]" in promoted
    assert "terminal_weights: [25, 20, 0.5, 0.25, 0.25, 0.5]" in promoted
    assert "n_horizon: 30" in promoted
