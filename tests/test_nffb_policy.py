"""CPU tests for the nonlinear feedforward--feedback policy."""

from __future__ import annotations

import math

import pytest
import torch

from baselines.nffb_policy import (
    LEGACY_OBSERVATION_FIELDS,
    NFFBPolicy,
    NFFBPolicyCfg,
    validate_nffb_environment_contract,
)


PREVIEW_FIELDS = (*LEGACY_OBSERVATION_FIELDS, "vg_0", "pg_1", "vg_1")


def make_policy(
    *,
    num_envs: int = 2,
    step_dt: float = 0.1,
    cfg: NFFBPolicyCfg | None = None,
    fields=PREVIEW_FIELDS,
) -> NFFBPolicy:
    return NFFBPolicy(
        cfg or NFFBPolicyCfg(),
        num_envs,
        "cpu",
        step_dt,
        raw_observation_dim=len(fields),
        raw_observation_fields=fields,
    )


def make_observation(num_envs: int = 2) -> torch.Tensor:
    observation = torch.zeros((num_envs, len(PREVIEW_FIELDS)), dtype=torch.float32)
    observation[:, 0] = 0.35
    observation[:, 9] = 0.35
    observation[:, 12] = 0.35
    return observation


def test_policy_returns_vectorized_physical_increment_and_tracks_command():
    policy = make_policy()
    observation = make_observation()
    observation[:, 9] = 0.45
    observation[:, 12] = 0.45

    first = policy.act(observation)
    second = policy.act(observation)

    assert first.shape == (2, 1)
    assert first.dtype == torch.float32
    assert torch.count_nonzero(first) == 0
    assert torch.all(second > 0.0)
    assert torch.all(torch.abs(second) <= policy.max_velocity_change)
    torch.testing.assert_close(policy.velocity_command, first + second)


def test_reference_acceleration_uses_preview_velocity_and_clips_corner_spike():
    policy = make_policy(step_dt=0.01)
    observation = make_observation()
    observation[:, 11] = -1.0
    observation[:, 13] = 1.0

    policy.act(observation)

    assert torch.allclose(policy.a_ref_raw, torch.full((2, 1), 200.0))
    assert torch.all(policy.a_ref_limited == policy.max_reference_acceleration)
    assert torch.all(policy.reference_acceleration_saturated)


def test_disabling_feedforward_keeps_reference_diagnostics_but_uses_zero():
    cfg = NFFBPolicyCfg()
    cfg.outer_loop.acceleration_feedforward_enabled = False
    policy = make_policy(cfg=cfg)
    observation = make_observation()
    observation[:, 13] = 1.0

    policy.act(observation)

    assert torch.count_nonzero(policy.a_ref_raw) > 0
    assert torch.count_nonzero(policy.a_ref_used) == 0


def test_command_filter_and_commands_stay_within_constraints():
    cfg = NFFBPolicyCfg()
    cfg.constraints.theta_max = 0.1
    cfg.constraints.omega_max = 0.05
    cfg.constraints.max_velocity = 0.01
    cfg.command_filter.natural_frequency = 3.0
    policy = make_policy(num_envs=1, cfg=cfg)
    observation = make_observation(1)
    observation[:, 0] = 0.1
    observation[:, 9] = 0.6
    observation[:, 12] = 0.6

    actions = [policy.act(observation) for _ in range(100)]

    assert torch.abs(policy.theta_d).max() <= cfg.constraints.theta_max + 1.0e-7
    assert torch.abs(policy.omega_d).max() <= cfg.constraints.omega_max + 1.0e-7
    assert torch.abs(policy.velocity_command).max() <= cfg.constraints.max_velocity + 1.0e-7
    assert max(float(torch.abs(action).max()) for action in actions) <= policy.max_velocity_change + 1.0e-7


def test_command_filter_reports_angular_rate_saturation():
    cfg = NFFBPolicyCfg()
    cfg.constraints.omega_max = 0.01
    cfg.command_filter.natural_frequency = 20.0
    policy = make_policy(num_envs=1, step_dt=0.01, cfg=cfg)
    observation = make_observation(1)
    observation[:, 0] = 0.1
    observation[:, 9] = 0.6
    observation[:, 12] = 0.6

    policy.act(observation)
    policy.act(observation)

    assert policy.omega_command_saturated.item()
    assert "policy_omega_command_saturated" in policy.get_state()


def test_conditional_anti_windup_freezes_integral_when_outer_loop_saturates():
    cfg = NFFBPolicyCfg()
    cfg.outer_loop.integral_pole = 0.2
    cfg.constraints.theta_max = math.radians(5.0)
    policy = make_policy(num_envs=1, cfg=cfg)
    observation = make_observation(1)
    observation[:, 0] = -10.0
    observation[:, 9] = 10.0
    observation[:, 12] = 10.0

    policy.act(observation)

    assert policy.ki > 0.0
    assert policy.ball_acceleration_saturated.item()
    assert policy.anti_windup_frozen.item()
    assert policy.integral_error.item() == 0.0


def test_partial_reset_only_clears_selected_environment():
    policy = make_policy()
    observation = make_observation()
    observation[:, 9] = torch.tensor([0.55, 0.15])
    observation[:, 12] = observation[:, 9]
    policy.act(observation)
    policy.act(observation)
    command_before = policy.velocity_command.clone()

    policy.reset(torch.tensor([0]))

    assert policy.velocity_command[0].item() == 0.0
    assert policy.filter_needs_init[0].item()
    torch.testing.assert_close(policy.velocity_command[1], command_before[1])
    assert not policy.filter_needs_init[1].item()


def test_policy_state_contains_finite_diagnostics():
    policy = make_policy()
    observation = make_observation()

    policy.act(observation)
    state = policy.get_state()

    assert "policy_theta_star" in state
    assert "policy_acceleration_saturated" in state
    for value in state.values():
        if value.dtype != torch.bool:
            assert torch.isfinite(value).all()


def test_preview_horizon_and_legacy_prefix_are_validated():
    with pytest.raises(ValueError, match="horizon"):
        make_policy(fields=LEGACY_OBSERVATION_FIELDS)
    invalid_fields = ("wrong", *PREVIEW_FIELDS[1:])
    with pytest.raises(ValueError, match="legacy"):
        make_policy(fields=invalid_fields)


def test_non_finite_observation_fails_fast():
    policy = make_policy()
    observation = make_observation()
    observation[0, 0] = float("nan")

    with pytest.raises(ValueError, match="non-finite"):
        policy.act(observation)


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        ({"task_name": "target_position"}, "unified_tracking"),
        ({"interface_name": "position"}, "velocity"),
        ({"reference_preview_enabled": False}, "reference_preview"),
        ({"reference_preview_future_steps": 0}, "future_steps"),
        (
            {
                "robustness_enabled": True,
                "action_delay_enabled": True,
                "delay_step": 1,
            },
            "action delay",
        ),
    ),
)
def test_environment_contract_rejects_unsupported_combinations(overrides, message):
    values = {
        "task_name": "unified_tracking",
        "interface_name": "velocity",
        "reference_preview_enabled": True,
        "reference_preview_future_steps": 1,
        "robustness_enabled": False,
        "action_delay_enabled": False,
        "delay_step": 0,
    }
    values.update(overrides)

    with pytest.raises(ValueError, match=message):
        validate_nffb_environment_contract(**values)


def test_environment_contract_accepts_delay_free_unified_velocity():
    validate_nffb_environment_contract(
        task_name="unified_tracking",
        interface_name="velocity",
        reference_preview_enabled=True,
        reference_preview_future_steps=5,
        robustness_enabled=False,
        action_delay_enabled=False,
        delay_step=0,
    )
