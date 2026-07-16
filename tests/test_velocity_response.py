from __future__ import annotations

import math

import pytest
import torch

from environments.robustness.velocity_response import VelocityResponseModel


def reset_model(
    model: VelocityResponseModel,
    *,
    initial_z: torch.Tensor | None = None,
    tau_s: torch.Tensor | None = None,
    gain: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    noise_std: torch.Tensor | None = None,
    ou_theta: torch.Tensor | None = None,
    ou_mu: torch.Tensor | None = None,
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
    )


def test_identity_and_zero_tau_static_response():
    model = VelocityResponseModel(2, "cpu")
    reset_model(
        model,
        tau_s=torch.zeros(2),
        gain=torch.tensor([1.0, 2.0]),
        bias=torch.tensor([0.0, -0.5]),
    )

    executed = model.step(torch.tensor([0.25, 0.75]), 0.1, "none")

    assert torch.allclose(executed, torch.tensor([0.25, 1.0]))
    assert torch.allclose(model.target_z, executed)
    assert torch.allclose(model.nominal_z, executed)
    assert torch.equal(model.noise_z, torch.zeros(2))


def test_exact_first_order_recurrence():
    model = VelocityResponseModel(1, "cpu")
    reset_model(model, tau_s=torch.tensor([0.2]))

    first = model.step(torch.tensor([1.0]), 0.1, "none").clone()
    second = model.step(torch.tensor([1.0]), 0.1, "none").clone()

    decay = math.exp(-0.5)
    assert first.item() == pytest.approx(1.0 - decay)
    assert second.item() == pytest.approx(1.0 - decay**2)


def test_gaussian_noise_is_additive_and_does_not_feed_back_nominal_state():
    model = VelocityResponseModel(1, "cpu")
    reset_model(model, tau_s=torch.tensor([0.2]), noise_std=torch.tensor([0.1]))

    noisy = model.step(
        torch.tensor([1.0]),
        0.1,
        "white",
        normal_samples=torch.tensor([2.0]),
    ).clone()
    nominal_after_first = model.nominal_z.clone()
    quiet = model.step(
        torch.tensor([1.0]),
        0.1,
        "gaussian",
        normal_samples=torch.tensor([0.0]),
    ).clone()

    assert noisy.item() == pytest.approx(nominal_after_first.item() + 0.2)
    assert quiet.item() == pytest.approx(1.0 - math.exp(-1.0))
    assert model.nominal_z.item() == pytest.approx(1.0 - math.exp(-1.0))


def test_ou_exact_discretization_and_clipping():
    model = VelocityResponseModel(1, "cpu")
    reset_model(
        model,
        noise_std=torch.tensor([0.5]),
        ou_theta=torch.tensor([2.0]),
        ou_mu=torch.tensor([0.2]),
    )

    executed = model.step(
        torch.tensor([0.9]),
        0.1,
        "ou",
        noise_clip=0.3,
        max_abs_velocity=1.0,
        normal_samples=torch.tensor([1.0]),
    )

    unconstrained_noise = 0.2 + 0.5 * math.sqrt(1.0 - math.exp(-0.4))
    assert unconstrained_noise > 0.3
    assert model.noise_z.item() == pytest.approx(0.3)
    assert model.nominal_z.item() == pytest.approx(0.9)
    assert executed.item() == pytest.approx(1.0)
    assert model.nominal_z.item() != executed.item()


def test_partial_reset_only_changes_selected_environment():
    model = VelocityResponseModel(3, "cpu")
    reset_model(model, initial_z=torch.tensor([1.0, 2.0, 3.0]))
    model.nominal_z[:] = torch.tensor([10.0, 20.0, 30.0])

    model.reset(
        torch.tensor([1]),
        torch.tensor([7.0]),
        tau_s=torch.tensor([0.4]),
        gain=torch.tensor([0.8]),
        bias=torch.tensor([-0.1]),
        noise_std=torch.tensor([0.2]),
        ou_theta=torch.tensor([3.0]),
        ou_mu=torch.tensor([0.05]),
    )

    assert torch.equal(model.nominal_z, torch.tensor([10.0, 7.0, 30.0]))
    assert model.tau_s[1].item() == pytest.approx(0.4)
    assert model.gain[1].item() == pytest.approx(0.8)
    assert model.noise_z[1].item() == pytest.approx(0.05)
    assert model.tau_s[0].item() == 0.0
    assert model.gain[2].item() == 1.0


def test_shape_dtype_mode_and_step_validation():
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

