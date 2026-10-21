"""Human motion model, solver schema, and current-measurement lifecycle contracts."""
from __future__ import annotations

import ast
import copy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

import baselines.acados_nmpc_core as core
import baselines.acados_nmpc_policy as policy_module
from baselines.model_state_predictor import VelocityModelStatePredictor, VelocityModelStatePredictorCfg
from test_acados_nmpc_core import make_cfg, _FakeOcpSolver, _FakeBatchSolver
from test_acados_nmpc_policy import FakeController, make_cfg as policy_cfg, observation, fields


def predictor(cfg, *, tau=0.187, delay=4, solver="rk4", enabled=True):
    return VelocityModelStatePredictor(VelocityModelStatePredictorCfg(
        enabled=enabled, delay_step=delay, solver=solver, step_dt=cfg.solver.step_dt,
        max_acc=5.0, max_velocity=0.0, velocity_response_enabled=True,
        velocity_response_tau_s=tau, velocity_response_gain=0.86,
        velocity_response_bias=-0.0035, velocity_response_max_abs_velocity=0.0,
        **vars(cfg.model),
    ), 2, "cpu")


def test_schema_zero_degeneracy_and_geometry():
    cfg = make_cfg()
    x = np.array([[.3, -.02, .1, .04, .02], [.4, .1, -.2, -.04, -.03]])
    p = np.tile([.35, 0, .187, .86, -.0035], (2, 1))
    old = core.discrete_dynamics_numpy(x, np.zeros((2, 1)), p, cfg)
    old_hash = core.controller_fingerprint(cfg)
    assert core.parameter_fields(cfg) == core.PARAMETER_FIELDS
    cfg.human_motion.enabled = True
    assert core.NP == 5 and core.parameter_dimension(cfg) == 6
    assert core.parameter_fields(cfg)[-1] == "human_vel_z"
    assert core.controller_fingerprint(cfg) != old_hash
    np.testing.assert_array_equal(old, core.discrete_dynamics_numpy(x, np.zeros((2, 1)), np.c_[p, [0, 0]], cfg))
    with pytest.raises(ValueError, match="dimensions"):
        core.discrete_dynamics_numpy(x, np.zeros((2, 1)), p, cfg)
    theta = np.array([0., .3, -.3, 1.57, -1.57, 4.])
    assert np.all(core.geometry_numpy(theta, .1, cfg.model, .1)[1] == 0)
    assert np.all(np.isfinite(core.geometry_numpy(theta, -.1, cfg.model, .1)[1]))
    assert core.geometry_numpy(0., 0., cfg.model, .1)[1] > 0
    assert core.geometry_numpy(0., 0., cfg.model, -.1)[1] < 0


@pytest.mark.parametrize("tau", [0., .187])
def test_numpy_torch_multistep_with_human_and_absolute_drone_position(tau):
    cfg = make_cfg()
    cfg.human_motion.enabled = True
    pred = predictor(cfg, tau=tau)
    raw = torch.zeros((2, 11))
    raw[:, 0] = torch.tensor([.3, .4])
    raw[:, 1] = torch.tensor([.1, -.02])
    raw[:, 3] = torch.tensor([.13, -.2])
    raw[:, 7] = torch.tensor([.08, -.04])
    raw[:, 9] = .35
    human = torch.tensor([.07, -.06])
    for a in [.03, -.01, .02, -.005]:
        pred.update_after_action(torch.tensor([[a], [-a]]))
    expected = raw[:, [0, 1, 3, 7]].numpy().astype(float)
    displacement = np.zeros(2)
    for command in pred.get_pending_commands().numpy()[..., 0]:
        target = .86 * command - .0035
        dt = cfg.solver.step_dt
        displacement += target * dt + ((expected[:, 3] - target) * tau * (1-np.exp(-dt/tau)) if tau else 0)
        x = np.c_[expected, command]
        p = np.column_stack([np.full(2, v) for v in [.35, 0, tau, .86, -.0035]] + [human.numpy()])
        expected = core.discrete_dynamics_numpy(x, np.zeros((2, 1)), p, cfg)[:, :4]
    result = pred.predict(raw, human_velocity_z=human)
    np.testing.assert_allclose(result[:, [0, 1, 3, 7]], expected, atol=1e-7)
    np.testing.assert_allclose(result[:, 6], displacement, atol=1e-7)
    np.testing.assert_allclose(result[:, 4], core.geometry_numpy(expected[:, 2], expected[:, 3], cfg.model, human.numpy())[1], atol=1e-7)
    baseline = predictor(cfg, tau=tau)
    zero = baseline.predict(raw, human_velocity_z=torch.zeros(2)).clone()
    np.testing.assert_array_equal(zero, baseline.predict(raw))


@pytest.mark.parametrize("solver,nodes", [("euler", 2), ("rk4", 5)])
def test_hold_every_node_and_reset(monkeypatch, solver, nodes):
    pred = predictor(make_cfg(), solver=solver)
    seen = []
    original = pred._derivatives
    def record(pb, vb, theta, vrz, human_velocity_z=0.):
        seen.append(human_velocity_z.clone())
        return original(pb, vb, theta, vrz, human_velocity_z)
    monkeypatch.setattr(pred, "_derivatives", record)
    raw = torch.zeros(2, 11)
    pred.predict(raw, human_velocity_z=torch.tensor([[.05], [-.02]]))
    assert len(seen) == 4 * nodes
    assert all(torch.equal(v, torch.tensor([.05, -.02])) for v in seen)
    pred.reset([0])
    pred.to("cpu")
    assert pred.human_velocity_z_used.tolist() == pytest.approx([0, -.02])
    pred.predict(raw)  # Never silently reuse previous measurement.
    assert torch.count_nonzero(pred.human_velocity_z_used) == 0


@pytest.mark.skipif(not core.acados_available(), reason="optional acados installation")
@pytest.mark.parametrize("tau", [0., .187])
def test_casadi_dynamics_and_both_costs(tau):
    cfg = make_cfg()
    cfg.human_motion.enabled = True
    ocp = core.build_acados_ocp(cfg, "human_test")
    model = ocp.model
    fn = core.ca.Function("evaluate_human", [model.x, model.u, model.p],
                          [model.disc_dyn_expr, model.cost_y_expr, model.cost_y_expr_e])
    old_cfg = copy.deepcopy(cfg)
    old_cfg.human_motion.enabled = False
    old = core.build_acados_ocp(old_cfg, "reactive_test").model
    oldfn = core.ca.Function("evaluate_old", [old.x, old.u, old.p],
                             [old.disc_dyn_expr, old.cost_y_expr, old.cost_y_expr_e])
    for vh in [0., .09, -.04]:
        x = np.array([.3, .02, -.13, .08, -.04])
        u = np.array([.01])
        p = np.array([.35, .1, tau, .86, -.0035, vh])
        result, stage, terminal = [np.asarray(v).ravel() for v in fn(x, u, p)]
        np.testing.assert_allclose(result, core.discrete_dynamics_numpy(x, u, p, cfg), atol=1e-12)
        omega = core.geometry_numpy(x[2], x[3], cfg.model, vh)[1]
        assert stage[3] == pytest.approx(omega)
        assert terminal[3] == pytest.approx(omega)
        if vh == 0:
            for new, legacy in zip([result, stage, terminal], oldfn(x, u, p[:5])):
                np.testing.assert_allclose(new, np.asarray(legacy).ravel(), atol=1e-12)
    assert ocp.parameter_values.shape == (6,)


@pytest.mark.parametrize("num_envs", [1, 2])
def test_solver_writes_all_six_parameters_through_terminal(monkeypatch, tmp_path, num_envs):
    cfg = make_cfg()
    cfg.human_motion.enabled = True
    cfg.solver.n_horizon = 3
    cfg.solver.qp_solver_cond_N = 2
    monkeypatch.setattr(core, "require_acados", lambda: None)
    monkeypatch.setattr(core, "build_acados_ocp", lambda *args: SimpleNamespace(solver_options=SimpleNamespace(), code_gen_opts=SimpleNamespace()))
    monkeypatch.setattr(core, "AcadosOcpSolver", _FakeOcpSolver)
    monkeypatch.setattr(core, "AcadosOcpBatchSolver", _FakeBatchSolver)
    ctrl = core.AcadosNMPCController(cfg, num_envs, build_root=tmp_path)
    p = np.tile([.35, 0, .187, .86, -.0035, .04], (num_envs, 4, 1))
    result = ctrl.solve(np.zeros((num_envs, 5)), p)
    assert not result.fallback.any()
    for i, solver in enumerate(ctrl._solvers):
        for stage in range(4):
            np.testing.assert_array_equal(solver.values[stage, "p"], p[i, stage])
    with pytest.raises(ValueError, match="stage_parameters"):
        ctrl.solve(np.zeros((num_envs, 5)), p[..., :5])


@pytest.mark.parametrize("active,delay", [(False, 0), (True, 0), (True, 4)])
def test_policy_same_fresh_measurement_and_partial_reset(monkeypatch, active, delay):
    monkeypatch.setattr(policy_module, "AcadosNMPCController", FakeController)
    FakeController.raise_on_solve = False
    FakeController.next_action = None
    FakeController.next_status = None
    cfg = policy_cfg(predictor=active)
    cfg.state_predictor.delay_step = delay
    cfg.human_motion.enabled = True
    policy = policy_module.AcadosNMPCPolicy(cfg, 2, "cpu", 1/60, raw_observation_fields=fields(delay+2))
    raw = observation(2, delay+2)
    for hv in [torch.tensor([.04, -.02]), torch.tensor([[.08], [-.03]])]:
        policy.act(raw, {"human_motion": {"velocity_z": hv.double()}})
        np.testing.assert_allclose(policy.controller.last_parameters[..., 5], hv.reshape(2, 1).expand(-1, 3))
        np.testing.assert_allclose(policy.human_velocity_z_used, hv.reshape(2))
        np.testing.assert_allclose(policy.state_predictor.human_velocity_z_used, hv.reshape(2) if active and delay else [0, 0])
    policy.reset([0])
    policy.to("cpu")
    assert policy.human_velocity_z_used.tolist() == pytest.approx([0, -.03])
    assert policy.get_resolved_metadata()["parameter_dimension"] == 6
    for invalid in [None, {}, {"human_motion": {"velocity_z": torch.zeros(2, 2)}},
                    {"human_motion": {"velocity_z": torch.tensor([float("nan"), 0])}},
                    {"human_motion": {"velocity_z": torch.tensor([float("inf"), 0])}}]:
        FakeController.raise_on_solve = True
        with pytest.raises(ValueError, match="human_motion"):
            policy.act(raw, invalid)
    # Even solver fallback records the fresh input and advances the existing queue.
    policy.act(raw, {"human_motion": {"velocity_z": torch.tensor([.01, .02])}})
    assert torch.all(policy.fallback == 1)
    assert policy.human_velocity_z_used.tolist() == pytest.approx([.01, .02])
    FakeController.raise_on_solve = False


def test_runner_uses_current_post_reset_measurement_without_mutating_terminal_snapshot():
    # Load only the pure helper, avoiding the script's Isaac startup side effects.
    path = Path(__file__).parents[1] / "scripts/acados_nmpc_unified_eval.py"
    tree = ast.parse(path.read_text())
    helper = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_policy_extras")
    scope = {"Any": Any}
    exec(compile(ast.Module(body=[helper], type_ignores=[]), str(path), "exec"), scope)
    current = torch.tensor([0., .02])
    terminal = torch.tensor([.07, .02])
    env = SimpleNamespace(robustness=SimpleNamespace(get_state=lambda: {"human_vel_z": current}))
    infos = {"step": {"human_vel_z": terminal}}
    extras = scope["_policy_extras"](infos, env, True)
    assert torch.equal(extras["human_motion"]["velocity_z"], current)
    current.fill_(5)
    assert extras["human_motion"]["velocity_z"][0] == 0
    assert "human_motion" not in infos and torch.equal(infos["step"]["human_vel_z"], terminal)
    assert "human_motion" not in scope["_policy_extras"](infos, None, False)


def test_disabled_ignores_extras_and_strict_flag(monkeypatch):
    monkeypatch.setattr(policy_module, "AcadosNMPCController", FakeController)
    cfg = policy_cfg()
    policy = policy_module.AcadosNMPCPolicy(cfg, 2, "cpu", 1/60, raw_observation_fields=fields(2))
    policy.act(observation(2, 2), {"human_motion": {"velocity_z": torch.tensor([float("nan")])}})
    assert policy.controller.last_parameters.shape[-1] == 5
    assert torch.count_nonzero(policy.human_velocity_z_used) == 0
    cfg.human_motion.enabled = "false"
    with pytest.raises(ValueError, match="boolean"):
        core.validate_controller_cfg(policy.controller_cfg)


def test_offline_alignment_excludes_terminal_transition_but_allows_new_episode():
    from scripts.analyze_human_motion_rollouts import uninterrupted_pairs, error_metrics
    done = np.array([[False], [True], [False], [False], [False]])
    np.testing.assert_array_equal(uninterrupted_pairs(done, 2)[:, 0], [False, False, True])
    np.testing.assert_array_equal(uninterrupted_pairs(done, 0), np.ones_like(done))
    metrics = error_metrics(np.array([[1.], [np.nan], [3.]]), np.ones((3, 1), dtype=bool))
    assert metrics == {"samples": 2, "mae": 2., "rmse": pytest.approx(np.sqrt(5))}


def test_runner_cpu_log_snapshot_survives_reset():
    path = Path(__file__).parents[1] / "scripts/acados_nmpc_unified_eval.py"
    tree = ast.parse(path.read_text())
    helper = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_tensor_to_numpy")
    scope = {"torch": torch, "np": np}
    exec(compile(ast.Module(body=[helper], type_ignores=[]), str(path), "exec"), scope)
    velocity = torch.tensor([.03, -.02])
    snapshot = scope["_tensor_to_numpy"](velocity)
    velocity.zero_()
    np.testing.assert_allclose(snapshot, [.03, -.02])
