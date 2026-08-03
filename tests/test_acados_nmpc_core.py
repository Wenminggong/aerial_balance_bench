from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from baselines.acados_nmpc_core import (
    AcadosNMPCController,
    AcadosNMPCControllerCfg,
    acados_available,
    build_acados_ocp,
    discrete_dynamics_numpy,
    geometry_numpy,
    require_acados,
    response_profile_numpy,
)
import baselines.acados_nmpc_core as core_module
from baselines.model_state_predictor import (
    VelocityModelStatePredictor,
    VelocityModelStatePredictorCfg,
)
from baselines.velocity_interface_model import VelocityInterfaceModel, VelocityInterfaceModelCfg


def make_cfg() -> AcadosNMPCControllerCfg:
    cfg = AcadosNMPCControllerCfg()
    cfg.model.plank_length = 1.06
    cfg.model.rope_length = 0.9
    cfg.model.ball_position_offset = 0.33
    cfg.model.gravity = 9.81
    cfg.model.ball_mass = 0.0005
    cfg.model.ball_radius = 0.023
    cfg.constraints.max_acc = 5.0
    cfg.constraints.max_velocity = 0.0
    cfg.solver.step_dt = 1.0 / 60.0
    return cfg


def test_optional_dependency_boundary_is_actionable():
    if acados_available():
        require_acados()
    else:
        with pytest.raises(ImportError, match="acados v0.5.4"):
            require_acados()


def test_release_gate_uses_source_tag_not_template_metadata(monkeypatch, tmp_path):
    source_dir = tmp_path / "acados"
    module_path = (
        source_dir
        / "interfaces"
        / "acados_template"
        / "acados_template"
        / "__init__.py"
    )
    module_path.parent.mkdir(parents=True)
    module_path.touch()

    monkeypatch.setattr(core_module, "_ACADOS_IMPORT_ERROR", None)
    monkeypatch.setattr(core_module, "acados_source_directory", lambda: source_dir)
    monkeypatch.setattr(core_module, "acados_template_module_path", lambda: module_path)
    monkeypatch.setattr(core_module, "acados_source_version", lambda: "0.5.4")
    monkeypatch.setattr(core_module, "acados_template_distribution_version", lambda: "0.5.1")
    monkeypatch.setattr(core_module, "_validate_acados_installation", lambda source: None)

    require_acados()
    assert core_module.installed_acados_version() == "0.5.4"


def test_release_gate_rejects_an_actual_old_source_checkout(monkeypatch, tmp_path):
    source_dir = tmp_path / "acados"
    module_path = (
        source_dir
        / "interfaces"
        / "acados_template"
        / "acados_template"
        / "__init__.py"
    )
    module_path.parent.mkdir(parents=True)
    module_path.touch()

    monkeypatch.setattr(core_module, "_ACADOS_IMPORT_ERROR", None)
    monkeypatch.setattr(core_module, "acados_source_directory", lambda: source_dir)
    monkeypatch.setattr(core_module, "acados_template_module_path", lambda: module_path)
    monkeypatch.setattr(core_module, "acados_source_version", lambda: "0.5.1")
    monkeypatch.setattr(core_module, "acados_template_distribution_version", lambda: "0.5.1")
    monkeypatch.setattr(core_module, "_validate_acados_installation", lambda source: None)

    with pytest.raises(RuntimeError, match="detected source release 0.5.1"):
        require_acados()


def test_release_gate_rejects_template_from_another_checkout(monkeypatch, tmp_path):
    source_dir = tmp_path / "acados"
    source_dir.mkdir()
    module_path = (
        tmp_path
        / "old_acados"
        / "interfaces"
        / "acados_template"
        / "acados_template.py"
    )
    module_path.parent.mkdir(parents=True)
    module_path.touch()

    monkeypatch.setattr(core_module, "_ACADOS_IMPORT_ERROR", None)
    monkeypatch.setattr(core_module, "acados_source_directory", lambda: source_dir)
    monkeypatch.setattr(core_module, "acados_template_module_path", lambda: module_path)
    monkeypatch.setattr(core_module, "acados_template_distribution_version", lambda: "0.5.1")

    with pytest.raises(RuntimeError, match="source/Python interface mismatch"):
        require_acados()


def test_installation_preflight_reports_missing_install_target(tmp_path):
    with pytest.raises(RuntimeError, match="--target install"):
        core_module._validate_acados_installation(tmp_path / "acados")


def test_geometry_matches_shared_torch_model():
    cfg = make_cfg()
    theta = np.linspace(-0.69, 0.69, 31)
    vrz = np.linspace(-0.8, 0.8, 31)
    beta, omega = geometry_numpy(theta, vrz, cfg.model)
    model = VelocityInterfaceModel(
        VelocityInterfaceModelCfg(
            plank_length=float(cfg.model.plank_length),
            rope_length=float(cfg.model.rope_length),
            ball_position_offset=float(cfg.model.ball_position_offset),
            epsilon=float(cfg.model.epsilon),
        )
    )
    theta_t = torch.as_tensor(theta, dtype=torch.float64)
    vrz_t = torch.as_tensor(vrz, dtype=torch.float64)
    assert np.allclose(beta, model.rope_angle(theta_t).numpy(), atol=1.0e-12)
    assert np.allclose(omega, model.beam_angular_velocity(theta_t, vrz_t).numpy(), atol=1.0e-12)


@pytest.mark.parametrize(
    ("tau_s", "gain", "bias"),
    [(0.0, 0.73, -0.02), (0.155, 0.86, -0.0035), (2.0, 1.3, 0.04)],
)
def test_exact_response_endpoint_and_discrete_dynamics_are_finite(tau_s, gain, bias):
    cfg = make_cfg()
    rng = np.random.default_rng(7)
    states = np.column_stack(
        (
            rng.uniform(0.0, 0.7, 64),
            rng.uniform(-0.5, 0.5, 64),
            rng.uniform(-0.69, 0.69, 64),
            rng.uniform(-1.0, 1.0, 64),
            rng.uniform(-1.0, 1.0, 64),
        )
    )
    actions = rng.uniform(-5.0 / 60.0, 5.0 / 60.0, (64, 1))
    parameters = np.tile([0.35, 0.0, tau_s, gain, bias], (64, 1))
    result = discrete_dynamics_numpy(states, actions, parameters, cfg)
    expected_vrz = response_profile_numpy(
        states[:, 3],
        states[:, 4] + actions[:, 0],
        parameters[:, 2],
        parameters[:, 3],
        parameters[:, 4],
        float(cfg.solver.step_dt),
        float(cfg.model.epsilon),
    )
    assert result.shape == states.shape
    assert np.all(np.isfinite(result))
    assert np.allclose(result[:, 3], expected_vrz, atol=1.0e-12)
    assert np.allclose(result[:, 4], states[:, 4] + actions[:, 0], atol=1.0e-12)


def test_tau_zero_one_step_matches_existing_predictor_equations():
    cfg = make_cfg()
    cfg.model.epsilon = 1.0e-9
    action = 0.04
    gain = 0.82
    bias = -0.006
    state = np.array([[0.31, -0.08, 0.13, 0.22, 0.0]])
    parameters = np.array([[0.35, 0.0, 0.0, gain, bias]])
    expected = discrete_dynamics_numpy(state, np.array([[action]]), parameters, cfg)[0]

    predictor_cfg = VelocityModelStatePredictorCfg(
        enabled=True,
        delay_step=1,
        solver="rk4",
        step_dt=float(cfg.solver.step_dt),
        max_acc=5.0,
        max_velocity=0.0,
        velocity_response_enabled=True,
        velocity_response_tau_s=0.0,
        velocity_response_gain=gain,
        velocity_response_bias=bias,
        velocity_response_max_abs_velocity=0.0,
        plank_length=float(cfg.model.plank_length),
        rope_length=float(cfg.model.rope_length),
        ball_position_offset=float(cfg.model.ball_position_offset),
        gravity=float(cfg.model.gravity),
        ball_mass=float(cfg.model.ball_mass),
        ball_radius=float(cfg.model.ball_radius),
        ball_inertia_ratio=float(cfg.model.ball_inertia_ratio),
        epsilon=float(cfg.model.epsilon),
    )
    predictor = VelocityModelStatePredictor(predictor_cfg, 1, "cpu")
    predictor.update_after_action(torch.tensor([[action]]))
    observation = torch.zeros((1, 11), dtype=torch.float32)
    observation[0, 0] = state[0, 0]
    observation[0, 1] = state[0, 1]
    observation[0, 3] = state[0, 2]
    observation[0, 7] = state[0, 3]
    observation[0, 9] = parameters[0, 0]
    actual = predictor.predict(
        observation, reference_positions=torch.tensor([[0.35, 0.35]])
    )[0]
    assert actual[0].item() == pytest.approx(expected[0], abs=2.0e-7)
    assert actual[1].item() == pytest.approx(expected[1], abs=2.0e-7)
    assert actual[3].item() == pytest.approx(expected[2], abs=2.0e-7)
    assert actual[7].item() == pytest.approx(expected[3], abs=2.0e-7)


@pytest.mark.skipif(not acados_available(), reason="external acados installation is optional")
def test_acados_ocp_dimensions_and_explicit_solver_mapping():
    cfg = make_cfg()
    ocp = build_acados_ocp(cfg, "test_acados_unified_nmpc")
    assert int(ocp.model.x.rows()) == 5
    assert int(ocp.model.u.rows()) == 1
    assert int(ocp.model.p.rows()) == 5
    assert int(ocp.model.cost_y_expr.rows()) == 7
    assert int(ocp.model.cost_y_expr_e.rows()) == 6
    assert int(ocp.model.con_h_expr.rows()) == 3
    assert int(ocp.model.con_h_expr_0.rows()) == 3
    assert int(ocp.model.con_h_expr_e.rows()) == 3
    assert ocp.solver_options.nlp_solver_type == "SQP_RTI"
    assert ocp.solver_options.qp_solver == "PARTIAL_CONDENSING_HPIPM"
    assert ocp.solver_options.hessian_approx == "GAUSS_NEWTON"
    assert ocp.solver_options.integrator_type == "DISCRETE"
    assert np.array_equal(ocp.constraints.idxbu, np.array([0]))
    assert math.isclose(ocp.constraints.ubu[0], 5.0 / 60.0)


class _FakeOcpSolver:
    last_kwargs = {}

    def __init__(self, ocp=None, **kwargs):
        del ocp
        type(self).last_kwargs = kwargs.copy()
        self.values = {}
        self.constraints = {}
        self.reset_count = 0
        self.solve_count = 0
        self.status = 5

    def reset(self, **kwargs):
        del kwargs
        self.reset_count += 1

    def set(self, stage, field, value):
        self.values[(stage, field)] = np.asarray(value).copy()

    def constraints_set(self, stage, field, value):
        self.constraints[(stage, field)] = np.asarray(value).copy()

    def solve(self):
        self.solve_count += 1
        self.status = 0
        return 0

    def get(self, stage, field):
        if field == "u":
            return np.array([0.02 + 0.001 * stage])
        if field == "x":
            return np.full(5, 0.1 * stage)
        if field in {"sl", "su"}:
            return np.zeros(3)
        raise KeyError(field)

    def get_stats(self, field):
        return {
            "time_tot": 0.001,
            "time_qp": 0.0002,
            "sqp_iter": 1,
            "qp_iter": np.array([2]),
        }[field]

    def get_cost(self):
        return 3.0


class _FakeBatchSolver:
    solve_calls = 0
    last_kwargs = {}

    def __init__(self, ocp, N_batch_init, **kwargs):
        del ocp
        type(self).last_kwargs = kwargs.copy()
        self.ocp_solvers = [_FakeOcpSolver() for _ in range(N_batch_init)]
        self.status = np.full(N_batch_init, 5, dtype=np.int64)

    def constraints_set(self, stage, field, values):
        for solver, value in zip(self.ocp_solvers, values, strict=True):
            solver.constraints_set(stage, field, value)

    def set(self, stage, field, values):
        for solver, value in zip(self.ocp_solvers, values, strict=True):
            solver.set(stage, field, value)

    def solve(self):
        type(self).solve_calls += 1
        self.status = np.asarray([solver.solve() for solver in self.ocp_solvers])


def test_single_and_native_batch_lifecycle_are_consistent(monkeypatch, tmp_path):
    cfg = make_cfg()
    cfg.solver.n_horizon = 3
    cfg.solver.qp_solver_cond_N = 2
    fake_ocp = SimpleNamespace(
        solver_options=SimpleNamespace(), code_gen_opts=SimpleNamespace()
    )
    monkeypatch.setattr(core_module, "require_acados", lambda: None)
    monkeypatch.setattr(core_module, "build_acados_ocp", lambda cfg, name: fake_ocp)
    monkeypatch.setattr(core_module, "controller_fingerprint", lambda cfg: "fake")
    monkeypatch.setattr(core_module, "AcadosOcpSolver", _FakeOcpSolver)
    monkeypatch.setattr(core_module, "AcadosOcpBatchSolver", _FakeBatchSolver)
    _FakeBatchSolver.solve_calls = 0

    states = np.array([[0.3, 0.0, 0.1, 0.0, 0.0]])
    parameters = np.tile([0.35, 0.0, 0.155, 0.86, -0.0035], (1, 4, 1))
    single = AcadosNMPCController(cfg, 1, build_root=tmp_path / "single")
    assert _FakeOcpSolver.last_kwargs["generate"] is True
    assert _FakeOcpSolver.last_kwargs["build"] is True
    assert fake_ocp.code_gen_opts.code_export_directory == str(
        tmp_path / "single" / "fake" / "c_generated_code"
    )
    single_result = single.solve(states, parameters)

    batch = AcadosNMPCController(cfg, 2, build_root=tmp_path / "batch")
    assert _FakeBatchSolver.last_kwargs["generate"] is True
    assert _FakeBatchSolver.last_kwargs["build"] is True
    batch_result = batch.solve(
        np.repeat(states, 2, axis=0), np.repeat(parameters, 2, axis=0)
    )
    assert _FakeBatchSolver.solve_calls == 1
    assert batch_result.action[:, 0] == pytest.approx(
        [single_result.action[0, 0], single_result.action[0, 0]]
    )
    assert np.all(batch_result.status == 0)
    assert np.all(~batch_result.fallback)
    assert batch._solvers[0].constraints[(0, "lbx")] == pytest.approx(states[0])
    before = batch._solvers[0].reset_count
    batch.reset([1])
    assert batch._solvers[0].reset_count == before
