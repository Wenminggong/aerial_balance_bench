from __future__ import annotations

import json

import numpy as np
import torch

from residual_dynamics_v2.model import (
    ResidualRBF2D,
    numpy_predict,
    portable_payload,
    save_portable_json,
)
from residual_dynamics_v2.profile_friction_probe import (
    ProbeSettings,
    ProfileFrictionResidual,
)


def make_model() -> ResidualRBF2D:
    return ResidualRBF2D(
        np.linspace(0.0, 0.7, 9).tolist(),
        [-0.5, -0.3, -0.15, -0.06, 0.0, 0.06, 0.15, 0.3, 0.5],
        0.1,
        0.1,
        1.0e-12,
        0.25,
        {
            "mean": [0.35, 0.0],
            "std": [0.15, 0.15],
            "lower_quantile": [0.0, -0.5],
            "upper_quantile": [0.7, 0.5],
            "inverse_correlation": [[1.0, 0.0], [0.0, 1.0]],
            "mahalanobis_threshold": 4.0,
            "decay_width_std": 1.0,
            "min_weight": 0.0,
        },
    )


def test_zero_coefficients_and_global_cap() -> None:
    model = make_model()
    points = torch.randn(2048, 5, dtype=torch.float64)
    assert torch.equal(model(points), torch.zeros(2048, dtype=torch.float64))
    with torch.no_grad():
        model.coefficients.copy_(torch.linspace(-1.0, 1.0, 81))
    model.project_coefficients()
    values = model(points)
    assert float(values.abs().max().detach()) <= 0.25 + 1.0e-12
    assert float(model.coefficients.abs().max().detach()) == 0.25


def test_ignored_inputs_have_exact_zero_gradient() -> None:
    model = make_model()
    with torch.no_grad():
        model.coefficients.copy_(torch.linspace(-0.2, 0.2, 81))
    features = torch.tensor(
        [[0.3, 0.1, 0.2, -0.3, 1.0]], dtype=torch.float64, requires_grad=True
    )
    gradient = torch.autograd.grad(model(features).sum(), features)[0]
    assert torch.equal(gradient[:, 2:], torch.zeros_like(gradient[:, 2:]))
    assert torch.isfinite(gradient[:, :2]).all()


def test_stage2_coefficient_trust_projection() -> None:
    model = make_model()
    reference = torch.linspace(-0.1, 0.1, 81, dtype=torch.float64)
    with torch.no_grad():
        model.coefficients.fill_(0.25)
    model.project_coefficients(reference, radius_mps2=0.03)
    assert torch.all(model.coefficients <= reference + 0.03 + 1.0e-15)
    assert torch.all(model.coefficients >= reference - 0.03 - 1.0e-15)


def test_ood_decays_and_numpy_parity(tmp_path) -> None:
    model = make_model()
    with torch.no_grad():
        model.coefficients.fill_(0.1)
    near = torch.tensor([[0.35, 0.0]], dtype=torch.float64)
    far = torch.tensor([[0.35, 3.0]], dtype=torch.float64)
    assert float(model.ood_components(far)["weight"]) < float(
        model.ood_components(near)["weight"]
    )
    features = np.asarray([[0.1, -0.2], [0.35, 0.0], [0.7, 0.3]], dtype=np.float64)
    payload = portable_payload(model, {"eligible": False}, features)
    expected = model(torch.from_numpy(features)).detach().numpy()
    np.testing.assert_allclose(numpy_predict(features, payload), expected, atol=1.0e-12)
    differentiable = torch.tensor([[0.35, 0.08]], dtype=torch.float64, requires_grad=True)
    torch_gradient = torch.autograd.grad(model(differentiable).sum(), differentiable)[0]
    epsilon = 1.0e-6
    finite_difference = []
    for axis in range(2):
        plus = differentiable.detach().numpy().copy()
        minus = plus.copy()
        plus[:, axis] += epsilon
        minus[:, axis] -= epsilon
        difference = (numpy_predict(plus, payload) - numpy_predict(minus, payload)) / (
            2.0 * epsilon
        )
        finite_difference.append(float(difference.item()))
    np.testing.assert_allclose(
        np.asarray(finite_difference), torch_gradient.detach().numpy()[0], atol=1.0e-8
    )
    path = tmp_path / "model.json"
    save_portable_json(path, payload)
    loaded = json.loads(path.read_text())
    assert loaded["model_id"] == "acceleration_residual_rbf2_pv_v1"
    assert len(loaded["golden_examples"]) == 3


def test_profile_friction_structure_and_cap() -> None:
    model = ProfileFrictionResidual(
        profile_enabled=True,
        friction_enabled=True,
        acceleration_gain=5.0 / 7.0,
        gravity_mps2=9.81,
        settings=ProbeSettings(),
    )
    with torch.no_grad():
        model.delta_coefficients.fill_(0.02)
        model.coulomb_coefficients.fill_(0.1)
        model.viscous_coefficients.fill_(0.2)
    features = torch.tensor(
        [[0.35, 0.2, 0.0, 0.4, 0.5], [0.35, -0.2, 0.0, -0.4, -0.5]],
        dtype=torch.float64,
        requires_grad=True,
    )
    values = model(features)
    # Positive local slope and positive velocity both produce negative acceleration.
    assert values[0] < 0.0
    # Friction reverses with velocity; ignored omega/a_nom inputs remain disconnected.
    components = model.components(features)
    assert components["friction"][0] < 0.0 < components["friction"][1]
    gradient = torch.autograd.grad(values.sum(), features)[0]
    assert torch.equal(gradient[:, 3:], torch.zeros_like(gradient[:, 3:]))
    assert float(values.abs().max().detach()) <= model.output_limit_mps2


def test_profile_blackbox_ignores_unapproved_inputs() -> None:
    model = ProfileFrictionResidual(
        profile_enabled=True,
        friction_enabled=False,
        blackbox_enabled=True,
        acceleration_gain=5.0 / 7.0,
        gravity_mps2=9.81,
        settings=ProbeSettings(),
    )
    with torch.no_grad():
        model.delta_coefficients.fill_(0.01)
        model.blackbox_coefficients.copy_(torch.linspace(-0.2, 0.2, 81))
    features = torch.tensor(
        [[0.35, 0.1, 0.05, 0.4, 0.5]], dtype=torch.float64, requires_grad=True
    )
    value = model(features)
    gradient = torch.autograd.grad(value.sum(), features)[0]
    assert torch.isfinite(gradient).all()
    assert torch.equal(gradient[:, 3:], torch.zeros_like(gradient[:, 3:]))
    assert float(value.abs().max().detach()) <= model.output_limit_mps2


def test_bounded_kappa_scales_nominal_acceleration() -> None:
    settings = ProbeSettings(kappa_limit=0.10)
    model = ProfileFrictionResidual(
        profile_enabled=False,
        friction_enabled=False,
        blackbox_enabled=False,
        kappa_enabled=True,
        acceleration_gain=5.0 / 7.0,
        gravity_mps2=9.81,
        settings=settings,
    )
    with torch.no_grad():
        model.kappa_coefficients.fill_(0.25)
    model.project()
    assert float(model.kappa_coefficients.max().detach()) == settings.kappa_limit
    features = torch.tensor(
        [[0.35, 0.05, 0.0, 0.4, 0.2], [0.35, 0.05, 0.0, -0.4, -0.2]],
        dtype=torch.float64,
        requires_grad=True,
    )
    components = model.components(features)
    assert torch.all(components["kappa"].abs() <= settings.kappa_limit + 1.0e-12)
    assert components["gain_residual"][0] > 0.0
    assert components["gain_residual"][1] < 0.0
    gradient = torch.autograd.grad(model(features).sum(), features)[0]
    assert torch.equal(gradient[:, 2:4], torch.zeros_like(gradient[:, 2:4]))
    assert torch.all(gradient[:, 4].abs() <= settings.kappa_limit + 1.0e-12)
