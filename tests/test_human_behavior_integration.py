"""Exercise manager hooks, real environment step ordering, and runner schemas."""

from __future__ import annotations

import ast
from pathlib import Path
from types import MethodType, SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
import yaml

# Reuse the existing lightweight Isaac configclass stub and fake drone interface.
from test_robustness_velocity_response import FakeVelocityInterface, make_cfg
from environments.robustness.human_behavior import HUMAN_STEP_EXTRA_FIELDS
from environments.robustness.robustness_manager import RobustnessManager


ROOT = Path(__file__).resolve().parents[1]
RUNNERS = (
    "zero_action_policy_eval", "cpid_policy_eval", "nmpc_policy_eval", "rl_policy_eval",
    "cpid_unified_tracking_eval", "acados_nmpc_unified_eval", "nffb_policy_eval",
)


def load_definitions(path, names, namespace, class_name=None):
    """Load actual methods without starting the runners' top-level Isaac app."""
    tree = ast.parse(path.read_text())
    body = tree.body
    if class_name:
        body = next(node.body for node in body if isinstance(node, ast.ClassDef) and node.name == class_name)
    nodes = [node for node in body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert len(nodes) == len(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


class FakeArticulation:
    def __init__(self, n):
        self.data = SimpleNamespace(**{name: torch.zeros(n, 1) for name in
                                      ("joint_pos", "joint_vel", "joint_acc", "default_joint_pos")})
        self.target = torch.zeros(n, 1)

    def set_joint_velocity_target(self, target, joint_ids, env_ids=None):
        assert joint_ids == [0]
        self.target[slice(None) if env_ids is None else env_ids] = target


def fake_env(mode="cooperative", enabled=True, interface="velocity"):
    env = SimpleNamespace(
        num_envs=2, device="cpu", cfg=SimpleNamespace(decimation=3, interface_name=interface,
                                                      task_name="target_position", sim=SimpleNamespace(render_interval=3)),
        step_dt=0.03, physics_dt=0.01, pb=torch.tensor([0.45, 0.25]), theta=torch.zeros(2),
        episode_length_buf=torch.zeros(2, dtype=torch.long), slider_holder_joint_ids=[0],
        drone_rope_plank=FakeArticulation(2), control_interface=FakeVelocityInterface(2),
    )
    env.task = SimpleNamespace(get_reference=lambda step: (torch.full((2,), 0.35) + step * 0.001, torch.zeros(2)))
    env.robustness = RobustnessManager(make_cfg(enabled=enabled, external_disturbance_enabled=True,
                                               human_behavior_type=mode, human_ou_enabled=False), 2, "cpu")
    env.robustness._reset_external_disturbance(env, torch.arange(2))
    env.robustness.after_state_reset(env, torch.arange(2))
    return env


@pytest.mark.parametrize("interface", ["thrust", "velocity", "position"])
@pytest.mark.parametrize("mode", ["cooperative", "random-walk", "disturbed"])
def test_manager_hooks_use_task_error_and_do_not_change_drone_command(interface, mode):
    env = fake_env(mode, interface=interface)
    env.control_interface.set_command_z(0.12)
    env.episode_length_buf[:] = 5
    env.robustness.prepare_human_step(env)
    model = env.robustness.human_behavior
    torch.testing.assert_close(model.error_prev1[:, 0], torch.tensor([0.095, -0.105]))
    for _ in range(3):
        env.robustness.before_physics_step(env)
        torch.testing.assert_close(env.drone_rope_plank.target, model.vel_cmd)
        env.drone_rope_plank.data.joint_vel[:] = 0.27
        env.drone_rope_plank.data.joint_acc[:] = 0.6
        env.robustness.after_physics_step(env)
    state = env.robustness.get_state()
    torch.testing.assert_close(state["human_vel_z"], torch.full((2,), 0.27))
    torch.testing.assert_close(state["human_acc_z"], torch.full((2,), 0.6))
    assert not torch.equal(state["human_vel_z"], state["human_target_vel_z"])
    assert torch.equal(state["human_target_vel_z"], state["external_disturbance_vel_z"])
    assert torch.equal(state["human_ou_theta"], state["external_disturbance_ou_theta"])
    torch.testing.assert_close(env.control_interface.command_velocity[:, 2], torch.full((2,), 0.12))


def test_real_environment_step_snapshots_terminal_values_before_partial_autoreset():
    env = fake_env()
    namespace = load_definitions(ROOT / "environments/aerial_balance_env.py", {"step", "_step_extras"},
                                 {"torch": torch}, "AerialBalanceEnv")
    env._step_extras = MethodType(namespace["_step_extras"], env)
    env._sim_step_counter = env.common_step_counter = 0
    env.episode_physics_length_buf = torch.zeros(2, dtype=torch.long)
    env.reset_terminated = torch.zeros(2, dtype=torch.bool)
    env.reset_time_outs = torch.zeros(2, dtype=torch.bool)
    env._pre_physics_step = lambda action: env.control_interface.get_command_state()
    env._apply_action = lambda: None
    env._update_state_from_sim = lambda: None
    env._get_dones = lambda: (torch.tensor([True, False]), torch.zeros(2, dtype=torch.bool))
    env._get_rewards = lambda: torch.zeros(2)
    env._state_dict = lambda: {}
    env._get_observations = lambda: {"policy": torch.zeros(2, 11)}
    env.evaluator = SimpleNamespace(update=lambda *args, **kwargs: {})
    submitted = []
    env.scene = SimpleNamespace(write_data_to_sim=lambda: submitted.append(env.drone_rope_plank.target.clone()),
                                update=lambda dt: None)

    def physics_step(render=False):
        # Deliberately different from the commanded velocity.
        env.drone_rope_plank.data.joint_vel[:] = 0.42
        env.drone_rope_plank.data.joint_pos[:] = 0.07
        env.drone_rope_plank.data.joint_acc[:] = len(submitted)

    env.sim = SimpleNamespace(step=physics_step, has_gui=lambda: False, has_rtx_sensors=lambda: False)

    def reset_selected(ids):
        env.drone_rope_plank.data.joint_vel[ids] = 0
        env.drone_rope_plank.data.joint_pos[ids] = 0
        env.robustness._reset_external_disturbance(env, ids)
        env.robustness.after_state_reset(env, ids)

    env._reset_idx = reset_selected
    observation, _, terminated, _, infos = namespace["step"](env, torch.zeros(2, 1))
    assert observation["policy"].shape == (2, 11)
    assert terminated.tolist() == [True, False]
    assert len(submitted) == 3
    assert submitted[0][0].item() != 0  # First substep already carries the new command.
    assert infos["step"]["human_vel_z"].tolist() == pytest.approx([0.42, 0.42])
    assert infos["step"]["human_pos_z"].tolist() == pytest.approx([0.07, 0.07])
    assert infos["step"]["human_acc_z"].tolist() == pytest.approx([2, 2])
    assert infos["step"]["human_target_vel_z"][0].item() != 0
    assert env.robustness.human_behavior.vel_cmd[0].item() == 0
    assert env.robustness.human_behavior.vel[0].item() == 0
    assert env.robustness.human_behavior.vel[1].item() == pytest.approx(0.42)
    assert torch.equal(env.drone_rope_plank.target[1], submitted[-1][1])


def test_disabled_manager_holds_zero_target_but_reports_real_motion():
    env = fake_env(enabled=False)
    before = torch.random.get_rng_state().clone()
    action = torch.tensor([[0.01], [-0.01]])
    assert env.robustness.before_action(env, action) is action
    env.robustness.prepare_human_step(env)
    env.robustness.before_physics_step(env)
    env.drone_rope_plank.data.joint_vel[:] = 0.07
    env.robustness.after_physics_step(env)
    assert torch.equal(before, torch.random.get_rng_state())
    assert torch.count_nonzero(env.drone_rope_plank.target) == 0
    assert env.robustness.get_state()["human_vel_z"].tolist() == pytest.approx([0.07, 0.07])


@pytest.mark.parametrize("runner", RUNNERS)
def test_runner_whitelist_collects_all_human_fields_and_npz_roundtrip(runner, tmp_path):
    path = ROOT / "scripts" / (runner + ".py")
    tree = ast.parse(path.read_text())
    assignment = next(node for node in tree.body if isinstance(node, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == "STEP_EXTRA_FIELDS" for t in node.targets))
    namespace = {"HUMAN_STEP_EXTRA_FIELDS": HUMAN_STEP_EXTRA_FIELDS, "torch": torch, "np": np, "Any": Any}
    exec(compile(ast.Module(body=[assignment], type_ignores=[]), str(path), "exec"), namespace)
    fields = namespace["STEP_EXTRA_FIELDS"]
    assert len(set(fields)) == len(fields)
    assert set(HUMAN_STEP_EXTRA_FIELDS) <= set(fields)
    names = {"_tensor_to_numpy", "_collect_step_extras"} if runner == "zero_action_policy_eval" else {"_tensor_to_numpy", "_collect_tensor_fields"}
    namespace = load_definitions(path, names, namespace)
    state = fake_env().robustness.get_state()
    if runner == "zero_action_policy_eval":
        collected = namespace["_collect_step_extras"]({"step": state})
    else:
        collected = namespace["_collect_tensor_fields"](state, fields)
    assert set(HUMAN_STEP_EXTRA_FIELDS) <= set(collected)
    payload = {"step_" + key: np.stack([value, value]) for key, value in collected.items()}
    np.savez_compressed(tmp_path / "rollout.npz", **payload)
    with np.load(tmp_path / "rollout.npz") as rollout:
        for key in HUMAN_STEP_EXTRA_FIELDS:
            assert rollout["step_" + key].shape == (2, 2)
            assert np.isfinite(rollout["step_" + key]).all()


@pytest.mark.parametrize("runner", ["cpid_unified_tracking_eval", "acados_nmpc_unified_eval"])
def test_new_and_legacy_yaml_supported_by_strict_config_loaders(runner):
    path = ROOT / "scripts" / (runner + ".py")
    namespace = load_definitions(path, {"_set_known_attrs"}, {"Any": Any})
    paths = [ROOT / "environments/configs/template_eval_disturbance.yaml",
             *sorted((ROOT / "environments/configs").glob("human_*_eval.yaml"))]
    assert len(paths) == 4
    for config_path in paths:
        values = yaml.safe_load(config_path.read_text())["robustness"]
        cfg = make_cfg()
        namespace["_set_known_attrs"](cfg, values, "robustness")
        manager = RobustnessManager(cfg, 2, "cpu")
        assert manager.human_behavior.active
    # Historical YAML without any human_* fields selects the new random-walk.
    cfg = make_cfg()
    namespace["_set_known_attrs"](cfg, {"enabled": True, "external_disturbance_enabled": True}, "robustness")
    assert RobustnessManager(cfg, 1, "cpu").human_behavior.mode == "random-walk"
    # Deprecated scalar inputs still load, but shared ranges are authoritative.
    namespace["_set_known_attrs"](cfg, {
        "human_behavior_type": "disturbed", "human_ou_theta": 9.0, "human_ou_sigma": 8.0,
        "external_disturbance_ou_theta_range": [0.8, 0.8],
        "external_disturbance_ou_sigma_range": [0.01, 0.01],
    }, "robustness")
    manager = RobustnessManager(cfg, 2, "cpu")
    manager.human_behavior.reset(torch.arange(2), torch.zeros(2))
    state = manager.get_state()
    torch.testing.assert_close(state["human_ou_theta"], torch.full((2,), 0.8))
    torch.testing.assert_close(state["human_ou_sigma"], torch.full((2,), 0.01))
    assert torch.equal(state["human_ou_theta"], state["external_disturbance_ou_theta"])
    assert torch.equal(state["human_ou_sigma"], state["external_disturbance_ou_sigma"])
