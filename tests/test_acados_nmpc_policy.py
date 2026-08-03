from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

import baselines.acados_nmpc_policy as policy_module
from baselines.acados_nmpc_core import AcadosNMPCStepResult
from baselines.acados_nmpc_policy import AcadosNMPCPolicy, AcadosNMPCPolicyCfg


LEGACY = ("pb", "vb", "ab", "theta", "omega", "alpha", "drz", "vrz", "arz", "pg", "a_prev")


def fields(horizon: int) -> tuple[str, ...]:
    result = [*LEGACY, "vg_0"]
    for offset in range(1, horizon + 1):
        result.extend((f"pg_{offset}", f"vg_{offset}"))
    return tuple(result)


class FakeController:
    next_action: np.ndarray | None = None
    next_status: np.ndarray | None = None
    raise_on_solve = False

    def __init__(self, cfg, num_envs, *, build_root):
        self.cfg = cfg
        self.num_envs = num_envs
        self.n_horizon = cfg.solver.n_horizon
        self.build_dir = Path(build_root) / "fake"
        self.fingerprint = "fake"
        self.reset_calls: list[list[int] | None] = []
        self.last_states = None
        self.last_parameters = None
        self.closed = False

    def reset(self, env_ids=None):
        self.reset_calls.append(None if env_ids is None else list(env_ids))

    def solve(self, initial_states, stage_parameters):
        if self.raise_on_solve:
            raise RuntimeError("synthetic solver failure")
        self.last_states = np.array(initial_states, copy=True)
        self.last_parameters = np.array(stage_parameters, copy=True)
        action = (
            np.full((self.num_envs, 1), 0.02)
            if self.next_action is None
            else np.array(self.next_action, copy=True)
        )
        status = (
            np.zeros(self.num_envs, dtype=np.int64)
            if self.next_status is None
            else np.array(self.next_status, copy=True)
        )
        fallback = status != 0
        values = np.arange(self.num_envs, dtype=np.float64) + 1.0
        return AcadosNMPCStepResult(
            action=action,
            status=status,
            solver_time=values * 1.0e-3,
            qp_time=values * 1.0e-4,
            sqp_iterations=values,
            qp_iterations=values + 1.0,
            cost=values * 2.0,
            max_slack=values * 0.01,
            fallback=fallback,
            wall_time=0.003,
        )

    def close(self):
        self.closed = True


@pytest.fixture(autouse=True)
def fake_controller(monkeypatch):
    FakeController.next_action = None
    FakeController.next_status = None
    FakeController.raise_on_solve = False
    monkeypatch.setattr(policy_module, "AcadosNMPCController", FakeController)


def make_cfg(*, predictor: bool = False) -> AcadosNMPCPolicyCfg:
    data = {
        "solver": {"n_horizon": 2, "qp_solver_cond_N": 2},
        "constraints": {"max_acc": 5.0, "max_velocity": 0.0},
        "state_predictor": {
            "enabled": predictor,
            "delay_step": 1 if predictor else 0,
            "max_acc": "auto",
            "max_velocity": "auto",
            "velocity_response_enabled": True,
            "velocity_response_tau_s": 0.155,
            "velocity_response_gain": 0.86,
            "velocity_response_bias": -0.0035,
            "velocity_response_max_abs_velocity": 0.0,
        },
    }
    return AcadosNMPCPolicyCfg.from_dict(data)


def observation(num_envs: int, horizon: int) -> torch.Tensor:
    names = fields(horizon)
    value = torch.zeros((num_envs, len(names)), dtype=torch.float32)
    index = {name: idx for idx, name in enumerate(names)}
    value[:, index["pb"]] = torch.arange(num_envs) * 0.01 + 0.3
    value[:, index["vb"]] = 0.02
    value[:, index["theta"]] = 0.03
    value[:, index["vrz"]] = 0.04
    value[:, index["pg"]] = 0.4
    value[:, index["vg_0"]] = 0.1
    for offset in range(1, horizon + 1):
        value[:, index[f"pg_{offset}"]] = 0.4 + offset * 0.01
        value[:, index[f"vg_{offset}"]] = 0.1 + offset * 0.02
    return value


def test_config_is_strict_and_preview_length_is_validated():
    with pytest.raises(ValueError, match="Unknown field"):
        AcadosNMPCPolicyCfg.from_dict({"solver": {"unknown": 1}})
    with pytest.raises(ValueError, match="preview is too short"):
        AcadosNMPCPolicy(make_cfg(), 1, "cpu", 1.0 / 60.0, raw_observation_fields=fields(1))


def test_delay_free_policy_uses_named_preview_and_returns_device_dtype_shape():
    policy = AcadosNMPCPolicy(
        make_cfg(), 2, "cpu", 1.0 / 60.0, raw_observation_fields=fields(2)
    )
    action = policy.act({"policy": observation(2, 2)})
    assert action.shape == (2, 1)
    assert action.dtype == torch.float32
    assert action.device.type == "cpu"
    assert torch.allclose(action, torch.full((2, 1), 0.02))
    assert np.allclose(
        policy.controller.last_states[:, :4],
        [[0.30, 0.02, 0.03, 0.04], [0.31, 0.02, 0.03, 0.04]],
    )
    assert policy.controller.last_states[:, 4] == pytest.approx([0.0, 0.0])
    assert policy.controller.last_parameters[0, :, 0] == pytest.approx([0.40, 0.41, 0.42])
    assert policy.controller.last_parameters[0, :, 1] == pytest.approx([0.10, 0.12, 0.14])
    assert torch.all(policy.get_state()["acados_nmpc_solver_success"] == 1.0)


def test_predictor_rebases_ocp_reference_to_d_through_d_plus_n():
    policy = AcadosNMPCPolicy(
        make_cfg(predictor=True),
        1,
        "cpu",
        1.0 / 60.0,
        raw_observation_fields=fields(3),
    )
    raw = observation(1, 3)
    policy.act(raw)
    assert policy.reference_base_offset == 1
    assert policy.controller.last_parameters[0, :, 0] == pytest.approx([0.41, 0.42, 0.43])
    assert policy.controller.last_parameters[0, :, 1] == pytest.approx([0.12, 0.14, 0.16])
    assert policy.controller.last_states[0, 4] == pytest.approx(0.0)
    assert policy.state_predictor.command_z[0, 0].item() == pytest.approx(0.02)


def test_partial_reset_preserves_other_command_and_resets_only_selected_solver():
    policy = AcadosNMPCPolicy(
        make_cfg(), 2, "cpu", 1.0 / 60.0, raw_observation_fields=fields(2)
    )
    policy.act(observation(2, 2))
    policy.reset(torch.tensor([1]))
    assert policy.controller.reset_calls[-1] == [1]
    assert policy.state_predictor.command_z[:, 0].tolist() == pytest.approx([0.02, 0.0])
    assert policy.action[:, 0].tolist() == pytest.approx([0.02, 0.0])


def test_nonzero_status_nan_and_exception_fall_back_to_zero_increment():
    policy = AcadosNMPCPolicy(
        make_cfg(), 2, "cpu", 1.0 / 60.0, raw_observation_fields=fields(2)
    )
    FakeController.next_action = np.array([[np.nan], [0.02]])
    FakeController.next_status = np.array([0, 3])
    action = policy.act(observation(2, 2))
    assert action[:, 0].tolist() == [0.0, 0.0]
    assert policy.fallback.tolist() == [1.0, 1.0]
    assert policy.state_predictor.command_z[:, 0].tolist() == [0.0, 0.0]

    FakeController.raise_on_solve = True
    action = policy.act(observation(2, 2))
    assert torch.equal(action, torch.zeros_like(action))
    assert policy.solver_status.tolist() == [-1.0, -1.0]


def test_active_predictor_requires_identical_response_parameters():
    cfg = make_cfg(predictor=True)
    cfg.state_predictor.velocity_response_gain = 0.9
    with pytest.raises(ValueError, match="mismatched field.*gain"):
        AcadosNMPCPolicy(cfg, 1, "cpu", 1.0 / 60.0, raw_observation_fields=fields(3))
