"""CPU tests for NFFB's deterministic velocity-response inverse."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from baselines.velocity_response_compensator import (
    FirstOrderVelocityResponseCompensator,
    NFFBVelocityResponseCompensationCfg,
)
from environments.robustness.velocity_response import VelocityResponseModel


def make_cfg(**values) -> NFFBVelocityResponseCompensationCfg:
    cfg = NFFBVelocityResponseCompensationCfg(
        enabled=True,
        parameter_source="explicit",
        tau_s=0.2,
        gain=0.9,
        bias=-0.01,
        max_abs_velocity=0.0,
    )
    for key, value in values.items():
        setattr(cfg, key, value)
    return cfg


def make_compensator(
    *,
    num_envs: int = 1,
    step_dt: float = 0.1,
    **values,
) -> FirstOrderVelocityResponseCompensator:
    return FirstOrderVelocityResponseCompensator(
        make_cfg(**values),
        num_envs,
        "cpu",
        step_dt,
    )


def reset_simulator(
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


def test_exact_inverse_matches_simulator_nominal_step():
    compensator = make_compensator()
    compensator.reset(initial_z=torch.tensor([[0.2]]))
    command = compensator.compensate(
        torch.tensor([[0.4]]),
        torch.tensor([[0.0]]),
        pending_commands=None,
        max_input_change=10.0,
        max_abs_input=0.0,
    )

    simulator = VelocityResponseModel(1, "cpu")
    reset_simulator(
        simulator,
        torch.tensor([0.2]),
        tau_s=0.2,
        gain=0.9,
        bias=-0.01,
    )
    output = simulator.step(command[:, 0], 0.1, "none")

    torch.testing.assert_close(output, torch.tensor([0.4]))
    torch.testing.assert_close(
        compensator.predicted_output_z[:, 0],
        output,
    )
    torch.testing.assert_close(
        compensator.tracking_error_z,
        torch.zeros((1, 1)),
        atol=1.0e-6,
        rtol=0.0,
    )


def test_zero_tau_inverts_static_gain_and_bias():
    compensator = make_compensator(tau_s=0.0, gain=2.0, bias=-0.1)
    command = compensator.compensate(
        torch.tensor([[0.5]]),
        torch.tensor([[0.0]]),
        pending_commands=None,
        max_input_change=10.0,
        max_abs_input=0.0,
    )

    torch.testing.assert_close(command, torch.tensor([[0.3]]))
    torch.testing.assert_close(
        compensator.predicted_output_z,
        torch.tensor([[0.5]]),
    )


def test_pending_commands_are_applied_before_inverse():
    compensator = make_compensator(tau_s=0.2, gain=1.0, bias=0.0)
    pending = torch.tensor([[[0.2]], [[0.6]]])
    desired = torch.tensor([[0.5]])

    command = compensator.compensate(
        desired,
        torch.tensor([[0.0]]),
        pending_commands=pending,
        max_input_change=10.0,
        max_abs_input=0.0,
    )

    simulator = VelocityResponseModel(1, "cpu")
    reset_simulator(
        simulator,
        torch.tensor([0.0]),
        tau_s=0.2,
        gain=1.0,
        bias=0.0,
    )
    simulator.step(pending[0, :, 0], 0.1, "none")
    simulator.step(pending[1, :, 0], 0.1, "none")
    output = simulator.step(command[:, 0], 0.1, "none")

    torch.testing.assert_close(
        compensator.execution_nominal_z[:, 0],
        simulator.nominal_z.new_tensor(
            [
                (1.0 - torch.exp(torch.tensor(-0.5))) * 0.6
                + torch.exp(torch.tensor(-0.5))
                * (1.0 - torch.exp(torch.tensor(-0.5)))
                * 0.2
            ]
        ),
    )
    torch.testing.assert_close(output, desired[:, 0])


def test_vectorized_output_and_input_constraints_are_reported():
    compensator = make_compensator(
        num_envs=2,
        tau_s=0.0,
        gain=1.0,
        bias=0.0,
        max_abs_velocity=0.5,
    )
    command = compensator.compensate(
        torch.tensor([[1.0], [-0.4]]),
        torch.tensor([[0.0], [0.45]]),
        pending_commands=None,
        max_input_change=0.1,
        max_abs_input=0.5,
    )

    torch.testing.assert_close(command, torch.tensor([[0.1], [0.35]]))
    assert compensator.output_saturated[:, 0].tolist() == [True, False]
    assert compensator.input_velocity_saturated[:, 0].tolist() == [False, False]
    assert compensator.input_acceleration_saturated[:, 0].tolist() == [True, True]
    torch.testing.assert_close(
        compensator.predicted_output_z,
        command,
    )


def test_absolute_input_limit_is_distinct_from_response_output_limit():
    compensator = make_compensator(
        tau_s=0.0,
        gain=0.5,
        bias=0.0,
        max_abs_velocity=0.0,
    )
    command = compensator.compensate(
        torch.tensor([[0.4]]),
        torch.tensor([[0.0]]),
        pending_commands=None,
        max_input_change=10.0,
        max_abs_input=0.5,
    )

    torch.testing.assert_close(command, torch.tensor([[0.5]]))
    assert compensator.input_velocity_saturated.item()
    assert not compensator.output_saturated.item()
    assert compensator.predicted_output_z.item() == pytest.approx(0.25)


def test_advance_and_partial_reset_only_change_selected_state():
    compensator = make_compensator(
        num_envs=2,
        tau_s=0.0,
        gain=1.0,
        bias=0.0,
    )
    compensator.advance(torch.tensor([[0.2], [0.4]]))
    compensator.reset(torch.tensor([0]))

    torch.testing.assert_close(
        compensator.nominal_z,
        torch.tensor([[0.0], [0.4]]),
    )


def test_explicit_auto_resolution_accepts_fixed_ranges_and_rejects_random_ranges():
    cfg = make_cfg(
        tau_s="auto",
        gain="auto",
        bias="auto",
        max_abs_velocity="auto",
    )
    robustness = SimpleNamespace(
        velocity_response_tau_s_range=(0.2, 0.2),
        velocity_response_gain_range=(0.9, 0.9),
        velocity_response_bias_range=(-0.01, -0.01),
        velocity_response_max_abs_velocity=0.7,
    )

    cfg.resolve_from_robustness(robustness)

    assert cfg.tau_s == pytest.approx(0.2)
    assert cfg.gain == pytest.approx(0.9)
    assert cfg.bias == pytest.approx(-0.01)
    assert cfg.max_abs_velocity == pytest.approx(0.7)

    randomized = make_cfg(tau_s="auto")
    robustness.velocity_response_tau_s_range = (0.1, 0.3)
    with pytest.raises(ValueError, match="explicit nominal"):
        randomized.resolve_from_robustness(robustness)


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        ({"parameter_source": "unknown"}, "parameter_source"),
        ({"tau_s": -0.1}, "tau_s"),
        ({"gain": 0.0}, "gain"),
        ({"bias": float("nan")}, "bias"),
        ({"max_abs_velocity": -1.0}, "max_abs_velocity"),
        ({"tau_s": 1.0e8}, "near-zero"),
    ),
)
def test_invalid_configuration_fails_fast(overrides, message):
    with pytest.raises(ValueError, match=message):
        make_compensator(**overrides)


def test_non_finite_and_invalid_shapes_fail_fast():
    compensator = make_compensator()
    with pytest.raises(ValueError, match="shape"):
        compensator.compensate(
            torch.zeros(1),
            torch.zeros((1, 1)),
            pending_commands=None,
            max_input_change=0.1,
            max_abs_input=0.0,
        )
    with pytest.raises(ValueError, match="finite"):
        compensator.compensate(
            torch.tensor([[float("nan")]]),
            torch.zeros((1, 1)),
            pending_commands=None,
            max_input_change=0.1,
            max_abs_input=0.0,
        )
