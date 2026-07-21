"""CPU tests for the nonlinear feedforward--feedback policy."""

from __future__ import annotations

import math
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from baselines.base_policy import ObservationIndex
from baselines.nffb_policy import (
    LEGACY_OBSERVATION_FIELDS,
    NFFBPolicy,
    NFFBPolicyCfg,
    validate_nffb_environment_contract,
)
from baselines.velocity_response_compensator import (
    NFFBVelocityResponseCompensationCfg,
)
from environments.robustness.velocity_response import VelocityResponseModel


PREVIEW_FIELDS = (*LEGACY_OBSERVATION_FIELDS, "vg_0", "pg_1", "vg_1")


def preview_fields(future_steps: int) -> tuple[str, ...]:
    fields = [*LEGACY_OBSERVATION_FIELDS, "vg_0"]
    for offset in range(1, future_steps + 1):
        fields.extend((f"pg_{offset}", f"vg_{offset}"))
    return tuple(fields)


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


def make_observation(
    num_envs: int = 2,
    *,
    fields: tuple[str, ...] = PREVIEW_FIELDS,
) -> torch.Tensor:
    observation = torch.zeros((num_envs, len(fields)), dtype=torch.float32)
    observation[:, 0] = 0.35
    observation[:, 9] = 0.35
    for name in fields:
        if name.startswith("pg_"):
            observation[:, fields.index(name)] = 0.35
    return observation


def make_predictor_cfg(delay_step: int = 2) -> NFFBPolicyCfg:
    cfg = NFFBPolicyCfg()
    cfg.state_predictor.enabled = True
    cfg.state_predictor.delay_step = delay_step
    cfg.state_predictor.solver = "euler"
    cfg.state_predictor.step_dt = 0.1
    cfg.state_predictor.max_acc = 0.5
    cfg.state_predictor.max_velocity = 0.0
    return cfg


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


def test_active_predictor_uses_shifted_position_velocity_and_acceleration_reference():
    fields = preview_fields(3)
    policy = make_policy(
        num_envs=1,
        cfg=make_predictor_cfg(),
        fields=fields,
    )
    observation = make_observation(1, fields=fields)
    observation[:, fields.index("pg_1")] = 0.4
    observation[:, fields.index("pg_2")] = 0.5
    observation[:, fields.index("vg_2")] = 0.2
    observation[:, fields.index("vg_3")] = 0.5

    policy.act(observation)
    state = policy.get_state()

    assert policy.reference_offset == 2
    assert policy.p_ref.item() == pytest.approx(0.5)
    assert policy.v_ref.item() == pytest.approx(0.2)
    assert policy.a_ref_raw.item() == pytest.approx(3.0)
    assert state["policy_predicted_pg"].item() == pytest.approx(0.5)
    assert state["policy_predictor_reference_preview_used"].item() == 1.0
    assert state["policy_reference_offset"].item() == 2.0


def test_predictor_initializes_command_filter_from_predicted_beam_state():
    fields = preview_fields(3)
    policy = make_policy(
        num_envs=1,
        cfg=make_predictor_cfg(),
        fields=fields,
    )
    observation = make_observation(1, fields=fields)
    policy.state_predictor.command_queue[:, 0, 0] = 0.2

    policy.act(observation)
    predicted_theta = policy.get_state()["policy_predicted_theta"]

    torch.testing.assert_close(policy.theta_d[:, 0], predicted_theta)
    assert predicted_theta.item() != pytest.approx(
        observation[0, ObservationIndex.THETA].item()
    )


def test_active_predictor_advances_command_filter_only_once_per_policy_cycle():
    fields = preview_fields(3)
    cfg = make_predictor_cfg()
    policy = make_policy(num_envs=1, cfg=cfg, fields=fields)
    observation = make_observation(1, fields=fields)
    observation[:, fields.index("pg_2")] = 0.5
    policy.act(observation)

    theta_d_before = policy.theta_d.clone()
    omega_d_before = policy.omega_d.clone()
    policy.act(observation)

    omega_f = float(cfg.command_filter.natural_frequency)
    zeta_f = float(cfg.command_filter.damping_ratio)
    expected_alpha = (
        omega_f**2 * (policy.theta_star - theta_d_before)
        - 2.0 * zeta_f * omega_f * omega_d_before
    )
    expected_omega = torch.clamp(
        omega_d_before + policy.step_dt * expected_alpha,
        min=-policy.omega_max,
        max=policy.omega_max,
    )
    expected_theta = torch.clamp(
        theta_d_before + policy.step_dt * expected_omega,
        min=-policy.theta_max,
        max=policy.theta_max,
    )
    outward = (
        ((expected_theta >= policy.theta_max) & (expected_omega > 0.0))
        | ((expected_theta <= -policy.theta_max) & (expected_omega < 0.0))
    )
    expected_omega = torch.where(outward, torch.zeros_like(expected_omega), expected_omega)

    torch.testing.assert_close(policy.filter_alpha, expected_alpha)
    torch.testing.assert_close(policy.omega_d, expected_omega)
    torch.testing.assert_close(policy.theta_d, expected_theta)


def test_predictor_command_and_queue_follow_nffb_incremental_command():
    fields = preview_fields(3)
    policy = make_policy(
        num_envs=1,
        cfg=make_predictor_cfg(),
        fields=fields,
    )
    observation = make_observation(1, fields=fields)
    observation[:, ObservationIndex.PG] = 0.5
    observation[:, fields.index("pg_1")] = 0.5
    observation[:, fields.index("pg_2")] = 0.5

    first_action = policy.act(observation)
    second_action = policy.act(observation)

    torch.testing.assert_close(policy.state_predictor.command_z, policy.velocity_command)
    torch.testing.assert_close(policy.predictor_command_sync_error, torch.zeros((1, 1)))
    torch.testing.assert_close(
        policy.state_predictor.get_pending_commands()[:, 0, 0],
        torch.stack((first_action[0, 0], first_action[0, 0] + second_action[0, 0])),
    )


def test_active_predictor_requires_one_more_velocity_preview_step():
    with pytest.raises(ValueError, match="future_steps >= 3"):
        make_policy(
            num_envs=1,
            cfg=make_predictor_cfg(),
            fields=preview_fields(2),
        )


def test_partial_reset_clears_predictor_and_controller_command_together():
    fields = preview_fields(3)
    policy = make_policy(
        cfg=make_predictor_cfg(),
        fields=fields,
    )
    observation = make_observation(fields=fields)
    observation[:, ObservationIndex.PG] = torch.tensor([0.5, 0.2])
    observation[:, fields.index("pg_1")] = observation[:, ObservationIndex.PG]
    observation[:, fields.index("pg_2")] = observation[:, ObservationIndex.PG]
    policy.act(observation)
    policy.act(observation)
    command_before = policy.velocity_command.clone()

    policy.reset(torch.tensor([0]))

    assert policy.velocity_command[0].item() == 0.0
    assert policy.state_predictor.command_z[0].item() == 0.0
    torch.testing.assert_close(policy.velocity_command[1], command_before[1])
    torch.testing.assert_close(
        policy.state_predictor.command_z[1],
        command_before[1],
    )


def test_active_predictor_rejects_nffb_command_model_mismatch():
    cfg = make_predictor_cfg()
    cfg.state_predictor.max_acc = 0.6

    with pytest.raises(ValueError, match="state_predictor.max_acc"):
        make_policy(
            num_envs=1,
            cfg=cfg,
            fields=preview_fields(3),
        )


def test_active_predictor_rejects_nffb_geometry_mismatch():
    cfg = make_predictor_cfg()
    cfg.state_predictor.ball_position_offset = 0.34

    with pytest.raises(ValueError, match="state_predictor.ball_position_offset"):
        make_policy(
            num_envs=1,
            cfg=cfg,
            fields=preview_fields(3),
        )


def test_active_predictor_combines_first_order_response_and_shifted_reference():
    cfg = make_predictor_cfg()
    cfg.state_predictor.velocity_response_enabled = True
    cfg.state_predictor.velocity_response_tau_s = 0.2
    fields = preview_fields(3)
    policy = make_policy(num_envs=1, cfg=cfg, fields=fields)
    observation = make_observation(1, fields=fields)
    observation[:, fields.index("pg_2")] = 0.45
    policy.state_predictor.command_queue[:, 0, 0] = 1.0

    policy.act(observation)
    state = policy.get_state()

    assert state["policy_predicted_pg"].item() == pytest.approx(0.45)
    assert state["policy_predicted_vrz"].item() > 0.0
    assert state["policy_predictor_velocity_response_enabled"].item() == 1.0


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


def test_environment_contract_accepts_aligned_delay_predictor_and_preview():
    validate_nffb_environment_contract(
        task_name="unified_tracking",
        interface_name="velocity",
        reference_preview_enabled=True,
        reference_preview_future_steps=9,
        robustness_enabled=True,
        action_delay_enabled=True,
        delay_step=8,
        state_predictor_enabled=True,
        state_predictor_delay_step=8,
    )


def test_environment_contract_requires_d_plus_one_preview_for_nffb():
    with pytest.raises(ValueError, match="future_steps >= 9"):
        validate_nffb_environment_contract(
            task_name="unified_tracking",
            interface_name="velocity",
            reference_preview_enabled=True,
            reference_preview_future_steps=8,
            robustness_enabled=True,
            action_delay_enabled=True,
            delay_step=8,
            state_predictor_enabled=True,
            state_predictor_delay_step=8,
        )


@pytest.mark.parametrize(
    ("environment_delay", "predictor_delay", "message"),
    [
        (8, 7, "must match"),
        (0, 8, "must not be active"),
    ],
)
def test_environment_contract_rejects_delay_alignment_errors(
    environment_delay: int,
    predictor_delay: int,
    message: str,
):
    with pytest.raises(ValueError, match=message):
        validate_nffb_environment_contract(
            task_name="unified_tracking",
            interface_name="velocity",
            reference_preview_enabled=True,
            reference_preview_future_steps=10,
            robustness_enabled=environment_delay > 0,
            action_delay_enabled=environment_delay > 0,
            delay_step=environment_delay,
            state_predictor_enabled=True,
            state_predictor_delay_step=predictor_delay,
        )


def test_policy_config_parses_state_predictor_section():
    cfg = NFFBPolicyCfg.from_dict(
        {
            "state_predictor": {
                "enabled": True,
                "delay_step": 3,
                "ball_position_offset": 0.33,
            }
        }
    )

    assert cfg.state_predictor.enabled
    assert cfg.state_predictor.delay_step == 3
    assert cfg.state_predictor.ball_position_offset == pytest.approx(0.33)


def test_policy_config_parses_velocity_response_compensation_section():
    cfg = NFFBPolicyCfg.from_dict(
        {
            "velocity_response_compensation": {
                "enabled": True,
                "parameter_source": "explicit",
                "tau_s": 0.2,
                "gain": 0.9,
                "bias": -0.01,
                "max_abs_velocity": 0.7,
            }
        }
    )

    assert cfg.velocity_response_compensation.enabled
    assert cfg.velocity_response_compensation.parameter_source == "explicit"
    assert cfg.velocity_response_compensation.tau_s == pytest.approx(0.2)
    assert cfg.velocity_response_compensation.gain == pytest.approx(0.9)
    assert cfg.velocity_response_compensation.bias == pytest.approx(-0.01)
    assert cfg.velocity_response_compensation.max_abs_velocity == pytest.approx(0.7)


def test_disabled_compensation_preserves_legacy_action_path_exactly():
    legacy = make_policy(num_envs=1)
    cfg = NFFBPolicyCfg()
    cfg.velocity_response_compensation = NFFBVelocityResponseCompensationCfg(
        enabled=False,
        parameter_source="explicit",
        tau_s=0.3,
        gain=0.7,
        bias=0.1,
        max_abs_velocity=0.2,
    )
    disabled = make_policy(num_envs=1, cfg=cfg)
    observation = make_observation(1)
    observation[:, ObservationIndex.PG] = 0.5
    observation[:, PREVIEW_FIELDS.index("pg_1")] = 0.5

    for _ in range(4):
        legacy_action = legacy.act(observation)
        disabled_action = disabled.act(observation)
        assert torch.equal(legacy_action, disabled_action)
        assert torch.equal(legacy.velocity_command, disabled.velocity_command)
        assert torch.equal(legacy.velocity_desired, disabled.velocity_desired)


def test_delay_free_compensation_uses_predictor_parameters_without_active_predictor():
    cfg = NFFBPolicyCfg()
    cfg.constraints.max_acc = 100.0
    cfg.state_predictor.velocity_response_tau_s = 0.0
    cfg.state_predictor.velocity_response_gain = 2.0
    cfg.state_predictor.velocity_response_bias = -0.1
    cfg.state_predictor.velocity_response_max_abs_velocity = 0.0
    cfg.velocity_response_compensation.enabled = True
    cfg.velocity_response_compensation.parameter_source = "state_predictor"
    policy = make_policy(num_envs=1, cfg=cfg)
    observation = make_observation(1)

    policy.act(observation)

    assert not policy.state_predictor.active
    assert policy.velocity_response_compensator.gain == pytest.approx(2.0)
    assert policy.velocity_response_compensator.bias == pytest.approx(-0.1)
    assert policy.velocity_command.item() == pytest.approx(0.05)
    assert policy.velocity_response_compensator.predicted_output_z.item() == pytest.approx(
        policy.velocity_desired.item()
    )


def test_delayed_compensation_forecasts_queue_and_advances_persistent_state_once():
    cfg = make_predictor_cfg(delay_step=2)
    cfg.state_predictor.velocity_response_enabled = True
    cfg.state_predictor.velocity_response_tau_s = 0.2
    cfg.state_predictor.velocity_response_gain = 1.0
    cfg.state_predictor.velocity_response_bias = 0.0
    cfg.velocity_response_compensation.enabled = True
    cfg.velocity_response_compensation.parameter_source = "state_predictor"
    fields = preview_fields(3)
    policy = make_policy(num_envs=1, cfg=cfg, fields=fields)
    observation = make_observation(1, fields=fields)
    policy.velocity_response_compensator.nominal_z.fill_(0.1)
    policy.state_predictor.command_queue[:, 0, 0] = torch.tensor([0.2, 0.4])

    decay = math.exp(-0.5)
    expected_after_oldest = decay * 0.1 + (1.0 - decay) * 0.2
    expected_execution = (
        decay * expected_after_oldest + (1.0 - decay) * 0.4
    )

    policy.act(observation)

    assert policy.velocity_response_compensator.execution_nominal_z.item() == pytest.approx(
        expected_execution
    )
    assert policy.velocity_response_compensator.nominal_z.item() == pytest.approx(
        expected_after_oldest
    )
    torch.testing.assert_close(
        policy.state_predictor.command_z,
        policy.velocity_command,
    )


def test_delayed_compensator_state_stays_aligned_with_simulator_queue():
    cfg = make_predictor_cfg(delay_step=2)
    cfg.constraints.max_acc = 5.0
    cfg.state_predictor.max_acc = 5.0
    cfg.state_predictor.velocity_response_enabled = True
    cfg.state_predictor.velocity_response_tau_s = 0.2
    cfg.state_predictor.velocity_response_gain = 0.9
    cfg.state_predictor.velocity_response_bias = -0.01
    cfg.velocity_response_compensation.enabled = True
    cfg.velocity_response_compensation.parameter_source = "state_predictor"
    fields = preview_fields(3)
    policy = make_policy(num_envs=1, cfg=cfg, fields=fields)
    observation = make_observation(1, fields=fields)
    observation[:, ObservationIndex.PG] = 0.5
    observation[:, fields.index("pg_1")] = 0.5
    observation[:, fields.index("pg_2")] = 0.5

    simulator = VelocityResponseModel(1, "cpu")
    zeros = torch.zeros(1)
    simulator.reset(
        torch.tensor([0]),
        zeros,
        tau_s=torch.tensor([0.2]),
        gain=torch.tensor([0.9]),
        bias=torch.tensor([-0.01]),
        noise_std=zeros,
        ou_theta=zeros,
        ou_mu=zeros,
    )
    environment_queue = torch.zeros((2, 1))
    environment_read_index = 0
    environment_command = torch.zeros(1)

    for _ in range(6):
        action = policy.act(observation)
        environment_command += action[:, 0]
        executed_input = environment_queue[environment_read_index].clone()
        environment_queue[environment_read_index] = environment_command
        environment_read_index = (environment_read_index + 1) % 2
        simulator.step(executed_input, 0.1, "none")

        torch.testing.assert_close(
            policy.velocity_response_compensator.nominal_z[:, 0],
            simulator.nominal_z,
        )


def test_partial_reset_clears_compensator_state_with_controller_state():
    cfg = NFFBPolicyCfg()
    cfg.velocity_response_compensation.enabled = True
    policy = make_policy(cfg=cfg)
    policy.velocity_response_compensator.nominal_z[:, 0] = torch.tensor([0.2, 0.4])

    policy.reset(torch.tensor([0]))

    assert policy.velocity_response_compensator.nominal_z[0].item() == 0.0
    assert policy.velocity_response_compensator.nominal_z[1].item() == pytest.approx(0.4)


def test_explicit_compensation_rejects_active_predictor_response_mismatch():
    cfg = make_predictor_cfg()
    cfg.state_predictor.velocity_response_enabled = True
    cfg.state_predictor.velocity_response_tau_s = 0.2
    cfg.velocity_response_compensation = NFFBVelocityResponseCompensationCfg(
        enabled=True,
        parameter_source="explicit",
        tau_s=0.3,
        gain=1.0,
        bias=0.0,
        max_abs_velocity=0.0,
    )

    with pytest.raises(
        ValueError,
        match="velocity_response_compensation.tau_s",
    ):
        make_policy(
            num_envs=1,
            cfg=cfg,
            fields=preview_fields(3),
        )


def test_compensation_diagnostics_are_exposed_and_finite():
    cfg = NFFBPolicyCfg()
    cfg.velocity_response_compensation.enabled = True
    policy = make_policy(num_envs=1, cfg=cfg)

    policy.act(make_observation(1))
    state = policy.get_state()

    assert state["policy_velocity_response_compensation_enabled"].item() == 1.0
    assert "policy_velocity_response_compensation_raw_input_z" in state
    assert "policy_velocity_response_compensation_tracking_error_z" in state
    for name, value in state.items():
        if "velocity_response_compensation" in name and value.dtype != torch.bool:
            assert torch.isfinite(value).all()


@pytest.mark.parametrize(
    ("run_config_name", "expected_delay"),
    (
        ("nffb_unified_tracking_response_compensation_eval.yaml", 0),
        (
            "nffb_unified_tracking_predictor_response_compensation_eval.yaml",
            8,
        ),
    ),
)
def test_compensation_evaluation_configs_resolve_and_construct(
    run_config_name: str,
    expected_delay: int,
):
    project_root = Path(__file__).resolve().parents[1]
    with open(
        project_root / "baselines" / "configs" / run_config_name,
        encoding="utf-8",
    ) as stream:
        run_config = yaml.safe_load(stream)
    with open(project_root / run_config["policy_config"], encoding="utf-8") as stream:
        effective_policy = yaml.safe_load(stream)
    effective_policy = deepcopy(effective_policy)
    _deep_update_for_test(effective_policy, run_config["policy_overrides"])
    cfg = NFFBPolicyCfg.from_dict(effective_policy)

    with open(project_root / run_config["env_config"], encoding="utf-8") as stream:
        env_config = yaml.safe_load(stream)
    robustness = SimpleNamespace(**env_config["robustness"])
    cfg.constraints.max_acc = float(env_config["velocity_interface"]["max_acc"])
    cfg.constraints.max_velocity = float(
        env_config["velocity_interface"]["max_velocity"]
    )
    if cfg.state_predictor.delay_step == "auto":
        cfg.state_predictor.delay_step = expected_delay
    cfg.state_predictor.resolve_velocity_response_from_robustness(robustness)
    cfg.velocity_response_compensation.resolve_from_robustness(robustness)

    future_steps = int(env_config["reference_preview"]["future_steps"])
    policy = make_policy(
        num_envs=1,
        step_dt=1.0 / 60.0,
        cfg=cfg,
        fields=preview_fields(future_steps),
    )

    assert policy.reference_offset == expected_delay
    assert policy.velocity_response_compensator.enabled
    assert policy.velocity_response_compensator.tau_s == pytest.approx(0.155)
    assert policy.velocity_response_compensator.gain == pytest.approx(0.86)
    assert policy.velocity_response_compensator.bias == pytest.approx(-0.0035)


def _deep_update_for_test(target: dict, values: dict) -> None:
    for key, value in values.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update_for_test(target[key], value)
        else:
            target[key] = value
