from __future__ import annotations

import math

import pytest
import torch

from environments.robustness.velocity_response import VelocityResponseModel


def reset_model(
    model: VelocityResponseModel,
    *,
    initial_z: torch.Tensor | None = None,
    initial_input_z: torch.Tensor | None = None,
    tau_s: torch.Tensor | None = None,
    gain: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    noise_std: torch.Tensor | None = None,
    ou_theta: torch.Tensor | None = None,
    ou_mu: torch.Tensor | None = None,
    sim_tau_s: torch.Tensor | None = None,
    sim_gain: torch.Tensor | None = None,
    sim_bias: torch.Tensor | None = None,
):
    num_envs = model.num_envs
    zeros = torch.zeros(num_envs)
    model.reset(
        torch.arange(num_envs),
        zeros if initial_z is None else initial_z,
        tau_s=zeros if tau_s is None else tau_s,
        gain=torch.ones(num_envs) if gain is None else gain,
        bias=zeros if bias is None else bias,
        noise_std=zeros if noise_std is None else noise_std,
        ou_theta=zeros if ou_theta is None else ou_theta,
        ou_mu=zeros if ou_mu is None else ou_mu,
        initial_input_z=initial_input_z,
        sim_tau_s=sim_tau_s,
        sim_gain=sim_gain,
        sim_bias=sim_bias,
    )


def sim_plant_step(
    previous_z: torch.Tensor,
    command_z: torch.Tensor,
    step_dt: float,
    tau_s: torch.Tensor,
    gain: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    decay = torch.exp(-float(step_dt) / tau_s)
    return decay * previous_z + (1.0 - decay) * (gain * command_z + bias)


def test_zero_target_tau_is_realized_through_sim_plant():
    model = VelocityResponseModel(2, "cpu")
    sim_tau_s = torch.tensor([0.12, 0.20])
    sim_gain = torch.tensor([0.8, 1.1])
    sim_bias = torch.tensor([0.03, -0.04])
    reset_model(
        model,
        tau_s=torch.zeros(2),
        gain=torch.tensor([1.0, 2.0]),
        bias=torch.tensor([0.0, -0.5]),
        sim_tau_s=sim_tau_s,
        sim_gain=sim_gain,
        sim_bias=sim_bias,
    )

    compensated = model.step(torch.tensor([0.25, 0.75]), 0.1, "none")
    actual = sim_plant_step(
        torch.zeros(2),
        compensated,
        0.1,
        sim_tau_s,
        sim_gain,
        sim_bias,
    )

    torch.testing.assert_close(model.executed_z, torch.tensor([0.25, 1.0]))
    torch.testing.assert_close(actual, model.executed_z, rtol=1.0e-6, atol=1.0e-6)
    torch.testing.assert_close(model.target_z, model.executed_z)
    assert torch.equal(model.noise_z, torch.zeros(2))


def test_exact_first_order_target_recurrence_matches_cascade():
    model = VelocityResponseModel(1, "cpu")
    sim_tau_s = torch.tensor([0.139])
    sim_gain = torch.tensor([1.0])
    sim_bias = torch.tensor([-0.00055])
    reset_model(
        model,
        tau_s=torch.tensor([0.2]),
        sim_tau_s=sim_tau_s,
        sim_gain=sim_gain,
        sim_bias=sim_bias,
    )

    actual = torch.zeros(1)
    desired_history = []
    for _ in range(2):
        compensated = model.step(torch.tensor([1.0]), 0.1, "none").clone()
        actual = sim_plant_step(
            actual,
            compensated,
            0.1,
            sim_tau_s,
            sim_gain,
            sim_bias,
        )
        desired_history.append(model.executed_z.item())
        torch.testing.assert_close(actual, model.executed_z, rtol=1.0e-6, atol=1.0e-6)

    decay = math.exp(-0.5)
    assert desired_history[0] == pytest.approx(1.0 - decay)
    assert desired_history[1] == pytest.approx(1.0 - decay**2)


def test_vectorized_random_and_sine_inputs_match_target_response_step_by_step():
    torch.manual_seed(17)
    num_envs = 3
    model = VelocityResponseModel(num_envs, "cpu")
    target_tau = torch.tensor([0.15, 0.22, 0.0])
    target_gain = torch.tensor([0.84, 0.88, 1.05])
    target_bias = torch.tensor([-0.004, -0.003, 0.01])
    sim_tau = torch.tensor([0.139, 0.139, 0.139])
    sim_gain = torch.tensor([1.0, 0.97, 1.04])
    sim_bias = torch.tensor([-0.00055, 0.002, -0.001])
    initial = torch.tensor([0.1, -0.2, 0.05])
    reset_model(
        model,
        initial_z=initial,
        initial_input_z=torch.zeros(num_envs),
        tau_s=target_tau,
        gain=target_gain,
        bias=target_bias,
        sim_tau_s=sim_tau,
        sim_gain=sim_gain,
        sim_bias=sim_bias,
    )

    actual = initial.clone()
    time = torch.arange(24, dtype=torch.float32) * 0.05
    commands = torch.stack(
        (
            0.4 * torch.sin(2.0 * math.pi * time),
            0.3 * torch.cos(math.pi * time),
            0.2 * torch.randn_like(time),
        ),
        dim=1,
    )
    for command in commands:
        compensated = model.step(command, 0.05, "none").clone()
        actual = sim_plant_step(actual, compensated, 0.05, sim_tau, sim_gain, sim_bias)
        torch.testing.assert_close(actual, model.executed_z, rtol=2.0e-6, atol=2.0e-6)


def test_gaussian_noise_is_target_output_noise_and_not_nominal_feedback():
    model = VelocityResponseModel(1, "cpu")
    sim_tau = torch.tensor([0.139])
    sim_gain = torch.tensor([1.0])
    sim_bias = torch.tensor([-0.00055])
    reset_model(
        model,
        tau_s=torch.tensor([0.2]),
        noise_std=torch.tensor([0.1]),
        sim_tau_s=sim_tau,
        sim_gain=sim_gain,
        sim_bias=sim_bias,
    )

    compensated = model.step(
        torch.tensor([1.0]),
        0.1,
        "white",
        normal_samples=torch.tensor([2.0]),
    ).clone()
    nominal_after_first = model.nominal_z.clone()
    actual = sim_plant_step(
        torch.zeros(1), compensated, 0.1, sim_tau, sim_gain, sim_bias
    )
    torch.testing.assert_close(actual, model.executed_z, rtol=1.0e-6, atol=1.0e-6)
    assert model.executed_z.item() == pytest.approx(nominal_after_first.item() + 0.2)

    compensated = model.step(
        torch.tensor([1.0]),
        0.1,
        "gaussian",
        normal_samples=torch.tensor([0.0]),
    ).clone()
    actual = sim_plant_step(actual, compensated, 0.1, sim_tau, sim_gain, sim_bias)
    torch.testing.assert_close(actual, model.executed_z, rtol=1.0e-6, atol=1.0e-6)
    assert model.nominal_z.item() == pytest.approx(1.0 - math.exp(-1.0))


def test_ou_noise_and_target_output_clipping_are_inverted_after_application():
    model = VelocityResponseModel(1, "cpu")
    sim_tau = torch.tensor([0.139])
    sim_gain = torch.tensor([1.0])
    sim_bias = torch.tensor([-0.00055])
    reset_model(
        model,
        noise_std=torch.tensor([0.5]),
        ou_theta=torch.tensor([2.0]),
        ou_mu=torch.tensor([0.2]),
        sim_tau_s=sim_tau,
        sim_gain=sim_gain,
        sim_bias=sim_bias,
    )

    compensated = model.step(
        torch.tensor([0.9]),
        0.1,
        "ou",
        noise_clip=0.3,
        max_abs_velocity=1.0,
        normal_samples=torch.tensor([1.0]),
    ).clone()
    actual = sim_plant_step(
        torch.zeros(1), compensated, 0.1, sim_tau, sim_gain, sim_bias
    )

    unconstrained_noise = 0.2 + 0.5 * math.sqrt(1.0 - math.exp(-0.4))
    assert unconstrained_noise > 0.3
    assert model.noise_z.item() == pytest.approx(0.3)
    assert model.nominal_z.item() == pytest.approx(0.9)
    assert model.executed_z.item() == pytest.approx(1.0)
    torch.testing.assert_close(actual, model.executed_z, rtol=1.0e-6, atol=1.0e-6)


def test_partial_reset_only_changes_selected_environment_and_all_response_state():
    model = VelocityResponseModel(3, "cpu")
    reset_model(model, initial_z=torch.tensor([1.0, 2.0, 3.0]))
    model.nominal_z[:] = torch.tensor([10.0, 20.0, 30.0])
    model.previous_executed_z[:] = torch.tensor([11.0, 21.0, 31.0])

    model.reset(
        torch.tensor([1]),
        torch.tensor([7.0]),
        tau_s=torch.tensor([0.4]),
        gain=torch.tensor([0.8]),
        bias=torch.tensor([-0.1]),
        noise_std=torch.tensor([0.2]),
        ou_theta=torch.tensor([3.0]),
        ou_mu=torch.tensor([0.05]),
        initial_input_z=torch.tensor([0.6]),
        sim_tau_s=torch.tensor([0.12]),
        sim_gain=torch.tensor([0.9]),
        sim_bias=torch.tensor([-0.02]),
    )

    assert torch.equal(model.nominal_z, torch.tensor([10.0, 7.0, 30.0]))
    assert torch.equal(model.previous_executed_z, torch.tensor([11.0, 7.0, 31.0]))
    assert model.input_z[1].item() == pytest.approx(0.6)
    assert model.tau_s[1].item() == pytest.approx(0.4)
    assert model.gain[1].item() == pytest.approx(0.8)
    assert model.noise_z[1].item() == pytest.approx(0.05)
    assert model.sim_tau_s[1].item() == pytest.approx(0.12)
    assert model.sim_gain[1].item() == pytest.approx(0.9)
    assert model.sim_bias[1].item() == pytest.approx(-0.02)
    assert model.tau_s[0].item() == 0.0
    assert model.gain[2].item() == 1.0


def test_shape_dtype_mode_and_sim_parameter_validation():
    model = VelocityResponseModel(2, "cpu")
    reset_model(model)

    output = model.step(torch.tensor([1.0, 2.0], dtype=torch.float64), 0.1, "none")
    assert output.shape == (2,)
    assert output.dtype == torch.float32
    assert output.device.type == "cpu"

    with pytest.raises(ValueError, match="noise mode"):
        model.step(torch.ones(2), 0.1, "invalid")
    with pytest.raises(ValueError, match="step_dt"):
        model.step(torch.ones(2), 0.0, "none")
    with pytest.raises(ValueError, match="shape"):
        model.step(torch.ones(1), 0.1, "none")

    with pytest.raises(ValueError, match="sim_tau_s"):
        reset_model(model, sim_tau_s=torch.tensor([0.0, 0.1]))
    with pytest.raises(ValueError, match="sim_gain"):
        reset_model(model, sim_gain=torch.tensor([1.0, 0.0]))
    with pytest.raises(ValueError, match="sim_bias"):
        reset_model(model, sim_bias=torch.tensor([0.0, float("nan")]))

    reset_model(model, sim_tau_s=torch.full((2,), 1.0e8))
    with pytest.raises(ValueError, match="near-zero"):
        model.step(torch.ones(2), 0.1, "none")
