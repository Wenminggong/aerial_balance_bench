from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from baselines.base_policy import ObservationIndex
from baselines.model_state_predictor import (
    VelocityModelStatePredictor,
    VelocityModelStatePredictorCfg,
    extract_reference_positions,
    validate_reference_preview_horizon,
)
from environments.robustness.velocity_response import VelocityResponseModel


def make_cfg(**values) -> VelocityModelStatePredictorCfg:
    cfg = VelocityModelStatePredictorCfg(
        enabled=True,
        delay_step=2,
        solver="euler",
        step_dt=0.1,
        max_acc=10.0,
    )
    for key, value in values.items():
        setattr(cfg, key, value)
    return cfg


def make_observation(num_envs: int, vrz: float | torch.Tensor = 0.0) -> torch.Tensor:
    observation = torch.zeros((num_envs, 11), dtype=torch.float32)
    observation[:, ObservationIndex.PB] = 0.35
    observation[:, ObservationIndex.PG] = 0.35
    observation[:, ObservationIndex.VRZ] = torch.as_tensor(vrz, dtype=torch.float32)
    return observation


def set_pending_commands(
    predictor: VelocityModelStatePredictor,
    commands: torch.Tensor,
    *,
    read_index: int = 0,
) -> None:
    commands = torch.as_tensor(commands, dtype=torch.float32)
    if commands.ndim == 1:
        commands = commands[:, None]
    predictor.command_queue[:, :, 0] = commands
    predictor.read_index = read_index


def reset_response_model(
    model: VelocityResponseModel,
    initial_z: torch.Tensor,
    *,
    tau_s: float,
    gain: float,
    bias: float,
) -> None:
    num_envs = model.num_envs
    zeros = torch.zeros(num_envs)
    model.reset(
        torch.arange(num_envs),
        initial_z,
        tau_s=torch.full((num_envs,), tau_s),
        gain=torch.full((num_envs,), gain),
        bias=torch.full((num_envs,), bias),
        noise_std=zeros,
        ou_theta=zeros,
        ou_mu=zeros,
    )


def test_disabled_response_preserves_legacy_prediction():
    commands = torch.tensor([0.25, 0.5])
    legacy = VelocityModelStatePredictor(make_cfg(), 1, "cpu")
    identity_response = VelocityModelStatePredictor(
        make_cfg(
            velocity_response_enabled=True,
            velocity_response_tau_s=0.0,
            velocity_response_gain=1.0,
            velocity_response_bias=0.0,
        ),
        1,
        "cpu",
    )
    set_pending_commands(legacy, commands)
    set_pending_commands(identity_response, commands)
    observation = make_observation(1, vrz=-0.1)

    legacy_prediction = legacy.predict(observation)
    identity_prediction = identity_response.predict(observation)

    assert torch.equal(legacy_prediction, identity_prediction)
    assert legacy_prediction[0, ObservationIndex.VRZ].item() == pytest.approx(0.5)


def test_legacy_dict_configuration_keeps_response_disabled():
    cfg = VelocityModelStatePredictorCfg.from_dict(
        {
            "enabled": True,
            "delay_step": 1,
            "step_dt": 0.1,
        }
    )

    predictor = VelocityModelStatePredictor(cfg, 1, "cpu")

    assert predictor.velocity_response_enabled is False
    assert predictor.velocity_response_tau_s == 0.0
    assert predictor.velocity_response_gain == 1.0
    assert predictor.velocity_response_bias == 0.0
    assert predictor.velocity_response_max_abs_velocity == 0.0


def test_missing_reference_preview_preserves_constant_goal_prediction():
    implicit_hold = VelocityModelStatePredictor(make_cfg(), 2, "cpu")
    explicit_hold = VelocityModelStatePredictor(make_cfg(), 2, "cpu")
    commands = torch.tensor([[0.1, -0.2], [0.3, 0.4]])
    set_pending_commands(implicit_hold, commands)
    set_pending_commands(explicit_hold, commands)
    observation = make_observation(2)
    held_references = observation[:, ObservationIndex.PG : ObservationIndex.PG + 1].expand(-1, 3)

    implicit_prediction = implicit_hold.predict(observation)
    explicit_prediction = explicit_hold.predict(
        observation,
        reference_positions=held_references,
    )

    torch.testing.assert_close(implicit_prediction, explicit_prediction, rtol=0.0, atol=0.0)
    assert torch.equal(
        implicit_hold.get_state()["policy_predictor_reference_preview_used"],
        torch.zeros(2),
    )
    assert torch.equal(
        explicit_hold.get_state()["policy_predictor_reference_preview_used"],
        torch.ones(2),
    )


def test_dynamic_reference_is_advanced_at_each_prediction_step():
    predictor = VelocityModelStatePredictor(make_cfg(), 1, "cpu")
    observation = make_observation(1)
    reference_positions = torch.tensor([[0.35, 0.40, 0.50]])

    prediction = predictor.predict(
        observation,
        reference_positions=reference_positions,
    )
    error, error_dot, error_ddot = predictor.get_error_prediction()

    assert prediction[0, ObservationIndex.PG].item() == pytest.approx(0.50)
    assert error.item() == pytest.approx(-0.15)
    assert error_dot.item() == pytest.approx(-0.10)
    assert error_ddot.item() == pytest.approx(-0.05)
    assert predictor.get_state()["policy_predictor_reference_horizon"].item() == 2


def test_dynamic_reference_supports_vectorized_environments():
    predictor = VelocityModelStatePredictor(make_cfg(), 2, "cpu")
    observation = make_observation(2)
    references = torch.tensor(
        [
            [0.35, 0.40, 0.45],
            [0.35, 0.30, 0.20],
        ]
    )

    prediction = predictor.predict(observation, reference_positions=references)
    error, _, _ = predictor.get_error_prediction()

    torch.testing.assert_close(prediction[:, ObservationIndex.PG], references[:, -1])
    torch.testing.assert_close(error[:, 0], torch.tensor([-0.10, 0.15]))


@pytest.mark.parametrize(
    ("references", "message"),
    [
        (torch.zeros((1, 2)), "reference_positions shape"),
        (torch.tensor([[0.35, 0.40, float("nan")]]), "must be finite"),
        (torch.tensor([[0.36, 0.40, 0.50]]), "must match the observation PG"),
    ],
)
def test_invalid_future_reference_sequence_is_rejected(
    references: torch.Tensor,
    message: str,
):
    predictor = VelocityModelStatePredictor(make_cfg(), 1, "cpu")

    with pytest.raises(ValueError, match=message):
        predictor.predict(make_observation(1), reference_positions=references)


def test_reference_extraction_uses_named_fields_and_rejects_short_preview():
    fields = (
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
        "vg_0",
        "pg_1",
        "vg_1",
        "pg_2",
        "vg_2",
    )
    observation = torch.arange(32, dtype=torch.float32).reshape(2, 16)

    references = extract_reference_positions(observation, fields, future_steps=2)

    torch.testing.assert_close(references, observation[:, [9, 12, 14]])
    with pytest.raises(ValueError, match="future_steps >= 3"):
        extract_reference_positions(observation, fields, future_steps=3)


def test_reference_extraction_accepts_pg_zero_alias():
    fields = ("pb", "pg_0", "pg_1")
    observation = torch.tensor([[0.0, 0.2, 0.3]])

    references = extract_reference_positions(observation, fields, future_steps=1)

    torch.testing.assert_close(references, torch.tensor([[0.2, 0.3]]))


def test_reference_preview_horizon_validation_fails_fast():
    validate_reference_preview_horizon(
        delay_step=2,
        preview_enabled=True,
        preview_future_steps=2,
        context="test",
    )
    with pytest.raises(ValueError, match="enabled=true"):
        validate_reference_preview_horizon(
            delay_step=2,
            preview_enabled=False,
            preview_future_steps=0,
            context="test",
        )
    with pytest.raises(ValueError, match="got 1 < 2"):
        validate_reference_preview_horizon(
            delay_step=2,
            preview_enabled=True,
            preview_future_steps=1,
            context="test",
        )


def test_exact_first_order_response_updates_vrz_drz_and_arz():
    predictor = VelocityModelStatePredictor(
        make_cfg(
            velocity_response_enabled=True,
            velocity_response_tau_s=0.2,
        ),
        1,
        "cpu",
    )
    set_pending_commands(predictor, torch.tensor([1.0, 1.0]))

    prediction = predictor.predict(make_observation(1))

    decay = torch.exp(torch.tensor(-0.5)).item()
    first_vrz = 1.0 - decay
    second_vrz = 1.0 - decay**2
    assert prediction[0, ObservationIndex.VRZ].item() == pytest.approx(second_vrz)
    assert prediction[0, ObservationIndex.DRZ].item() == pytest.approx(
        0.1 * (first_vrz + second_vrz)
    )
    assert prediction[0, ObservationIndex.ARZ].item() == pytest.approx(
        (second_vrz - first_vrz) / 0.1
    )


def test_zero_tau_gain_bias_and_response_output_limit():
    predictor = VelocityModelStatePredictor(
        make_cfg(
            delay_step=1,
            velocity_response_enabled=True,
            velocity_response_tau_s=0.0,
            velocity_response_gain=2.0,
            velocity_response_bias=-0.1,
            velocity_response_max_abs_velocity=0.5,
        ),
        1,
        "cpu",
    )
    set_pending_commands(predictor, torch.tensor([0.4]))

    prediction = predictor.predict(make_observation(1, vrz=-0.2))

    assert prediction[0, ObservationIndex.VRZ].item() == pytest.approx(0.5)
    assert prediction[0, ObservationIndex.ARZ].item() == pytest.approx(7.0)


def test_response_is_applied_to_commands_in_delay_queue_order():
    predictor = VelocityModelStatePredictor(
        make_cfg(
            velocity_response_enabled=True,
            velocity_response_tau_s=0.1,
        ),
        1,
        "cpu",
    )
    predictor.update_after_action(torch.tensor([[0.4]]))
    predictor.update_after_action(torch.tensor([[0.1]]))

    assert torch.allclose(predictor._pending_commands()[:, 0, 0], torch.tensor([0.4, 0.5]))

    prediction = predictor.predict(make_observation(1))
    decay = torch.exp(torch.tensor(-1.0)).item()
    first_vrz = (1.0 - decay) * 0.4
    expected_vrz = decay * first_vrz + (1.0 - decay) * 0.5
    assert prediction[0, ObservationIndex.VRZ].item() == pytest.approx(expected_vrz)


def test_velocity_response_and_future_reference_are_applied_together():
    predictor = VelocityModelStatePredictor(
        make_cfg(
            velocity_response_enabled=True,
            velocity_response_tau_s=0.2,
        ),
        1,
        "cpu",
    )
    set_pending_commands(predictor, torch.tensor([1.0, 1.0]))

    prediction = predictor.predict(
        make_observation(1),
        reference_positions=torch.tensor([[0.35, 0.40, 0.50]]),
    )

    assert prediction[0, ObservationIndex.PG].item() == pytest.approx(0.50)
    assert prediction[0, ObservationIndex.VRZ].item() > 0.0
    assert predictor.get_state()["policy_predictor_reference_preview_used"].item() == 1.0


def test_vectorized_response_matches_simulator_nominal_model_with_output_clipping():
    initial_z = torch.tensor([0.2, -0.1])
    commands = torch.tensor([[1.2, -1.0], [1.2, -1.0], [0.1, 0.6]])
    tau_s = 0.25
    gain = 0.9
    bias = -0.02
    max_abs_velocity = 0.4
    predictor = VelocityModelStatePredictor(
        make_cfg(
            delay_step=3,
            velocity_response_enabled=True,
            velocity_response_tau_s=tau_s,
            velocity_response_gain=gain,
            velocity_response_bias=bias,
            velocity_response_max_abs_velocity=max_abs_velocity,
        ),
        2,
        "cpu",
    )
    set_pending_commands(predictor, commands)

    response_model = VelocityResponseModel(2, "cpu")
    reset_response_model(
        response_model,
        initial_z,
        tau_s=tau_s,
        gain=gain,
        bias=bias,
    )
    expected_vrz = initial_z
    for command in commands:
        expected_vrz = response_model.step(
            command,
            0.1,
            "none",
            max_abs_velocity=max_abs_velocity,
        ).clone()

    prediction = predictor.predict(make_observation(2, vrz=initial_z))

    assert torch.allclose(
        prediction[:, ObservationIndex.VRZ],
        expected_vrz,
        atol=1.0e-6,
        rtol=1.0e-6,
    )


def test_auto_response_parameters_require_fixed_environment_ranges():
    predictor_cfg = make_cfg(
        velocity_response_tau_s="auto",
        velocity_response_gain="auto",
        velocity_response_bias="auto",
        velocity_response_max_abs_velocity="auto",
    )
    robustness_cfg = SimpleNamespace(
        velocity_response_tau_s_range=(0.2, 0.2),
        velocity_response_gain_range=(0.9, 0.9),
        velocity_response_bias_range=(-0.01, -0.01),
        velocity_response_max_abs_velocity=0.75,
    )

    predictor_cfg.resolve_velocity_response_from_robustness(robustness_cfg)

    assert predictor_cfg.velocity_response_tau_s == pytest.approx(0.2)
    assert predictor_cfg.velocity_response_gain == pytest.approx(0.9)
    assert predictor_cfg.velocity_response_bias == pytest.approx(-0.01)
    assert predictor_cfg.velocity_response_max_abs_velocity == pytest.approx(0.75)

    randomized_cfg = make_cfg(velocity_response_tau_s="auto")
    robustness_cfg.velocity_response_tau_s_range = (0.1, 0.3)
    with pytest.raises(ValueError, match="explicit nominal predictor value"):
        randomized_cfg.resolve_velocity_response_from_robustness(robustness_cfg)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("velocity_response_tau_s", -0.1, "tau_s >= 0"),
        ("velocity_response_gain", 0.0, "gain > 0"),
        ("velocity_response_bias", float("nan"), "finite velocity_response_bias"),
        ("velocity_response_max_abs_velocity", -1.0, "max_abs_velocity >= 0"),
        ("ball_position_offset", float("nan"), "finite ball_position_offset"),
    ],
)
def test_invalid_velocity_response_configuration(field: str, value: float, message: str):
    with pytest.raises(ValueError, match=message):
        VelocityModelStatePredictor(make_cfg(**{field: value}), 1, "cpu")
