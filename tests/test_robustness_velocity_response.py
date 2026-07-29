from __future__ import annotations

import dataclasses
import sys
import types
from types import SimpleNamespace

import pytest
import torch


def _install_configclass_stub_if_needed():
    try:
        from omni.isaac.lab.utils import configclass  # noqa: F401

        return
    except ModuleNotFoundError:
        pass

    omni = sys.modules.setdefault("omni", types.ModuleType("omni"))
    isaac = sys.modules.setdefault("omni.isaac", types.ModuleType("omni.isaac"))
    lab = sys.modules.setdefault("omni.isaac.lab", types.ModuleType("omni.isaac.lab"))
    utils = types.ModuleType("omni.isaac.lab.utils")
    utils.configclass = dataclasses.dataclass
    sys.modules["omni.isaac.lab.utils"] = utils
    omni.isaac = isaac
    isaac.lab = lab
    lab.utils = utils


_install_configclass_stub_if_needed()

from environments.robustness.robustness_manager import (  # noqa: E402
    CommandDelayQueue,
    PerEnvCommandDelayQueue,
    RobustnessManager,
    RobustnessManagerCfg,
)


class FakeVelocityInterface:
    def __init__(self, num_envs: int):
        self.command_velocity = torch.zeros((num_envs, 3))
        self.executed_velocity = torch.zeros((num_envs, 3))
        self.last_action = torch.zeros((num_envs, 1))

    def set_command_z(self, value: float):
        self.command_velocity[:, 2] = value
        self.executed_velocity.copy_(self.command_velocity)

    def get_delay_command(self) -> torch.Tensor:
        return self.command_velocity

    def set_executed_delay_command(self, command: torch.Tensor):
        self.executed_velocity = command.to(dtype=torch.float32).clone()

    def get_command_state(self) -> dict[str, torch.Tensor]:
        return {
            "command_z": self.command_velocity[:, 2],
            "executed_command_z": self.executed_velocity[:, 2],
            "target_velocity": self.command_velocity,
            "executed_velocity": self.executed_velocity,
            "vrz_cmd": self.command_velocity[:, 2],
            "executed_vrz_cmd": self.executed_velocity[:, 2],
            "last_action": self.last_action,
        }


def make_cfg(**values) -> RobustnessManagerCfg:
    cfg = RobustnessManagerCfg()
    for key, value in values.items():
        setattr(cfg, key, value)
    return cfg


def make_env(num_envs: int = 1, interface_name: str = "velocity"):
    return SimpleNamespace(
        cfg=SimpleNamespace(interface_name=interface_name),
        step_dt=0.1,
        vrz=torch.zeros(num_envs),
        control_interface=FakeVelocityInterface(num_envs),
    )


def initialize_command_models(manager: RobustnessManager, env, env_ids: torch.Tensor):
    manager._reset_action_delay(env, env_ids)
    manager._reset_velocity_response(env, env_ids)


def test_fixed_delay_queue_preserves_legacy_sequence():
    queue = CommandDelayQueue(2, 1, 1, torch.device("cpu"))
    queue.reset(torch.tensor([0]), torch.zeros((1, 1)))

    outputs = [queue.step(torch.tensor([[value]])).item() for value in (1.0, 2.0, 3.0)]

    assert outputs == pytest.approx([0.0, 0.0, 1.0])


def test_per_environment_delay_queue_supports_zero_and_mixed_delays():
    queue = PerEnvCommandDelayQueue(2, 3, 1, torch.device("cpu"))
    queue.reset(torch.arange(3), torch.zeros((3, 1)))
    delay_steps = torch.tensor([0, 1, 2])

    first = queue.step(torch.tensor([[1.0], [10.0], [100.0]]), delay_steps)
    second = queue.step(torch.tensor([[2.0], [20.0], [200.0]]), delay_steps)
    third = queue.step(torch.tensor([[3.0], [30.0], [300.0]]), delay_steps)

    assert torch.equal(first[:, 0], torch.tensor([1.0, 0.0, 0.0]))
    assert torch.equal(second[:, 0], torch.tensor([2.0, 10.0, 0.0]))
    assert torch.equal(third[:, 0], torch.tensor([3.0, 20.0, 100.0]))


@pytest.mark.parametrize("choices", ([True], [-1], [1.5], 1, "1"))
def test_random_delay_choices_reject_invalid_values(choices):
    with pytest.raises(ValueError, match="non-negative integers"):
        RobustnessManager(make_cfg(delay_step_choices=choices), 1, "cpu")


def test_random_delay_reset_samples_selected_envs_and_preserves_other_histories():
    torch.manual_seed(7)
    cfg = make_cfg(
        enabled=True,
        action_delay_enabled=True,
        delay_step=99,
        delay_step_choices=(0, 1, 2),
    )
    manager = RobustnessManager(cfg, 3, "cpu")
    env = make_env(3)
    all_env_ids = torch.arange(3)
    manager._reset_action_delay(env, all_env_ids)
    original_delays = manager.sampled_delay_step.clone()
    manager.per_env_delay_queue.buffer[:, 0] = 11.0
    manager.per_env_delay_queue.buffer[:, 2] = 33.0

    manager._reset_action_delay(env, torch.tensor([1]))

    assert manager.sampled_delay_step[0] == original_delays[0]
    assert manager.sampled_delay_step[2] == original_delays[2]
    assert manager.sampled_delay_step[1].item() in cfg.delay_step_choices
    assert torch.all(manager.per_env_delay_queue.buffer[:, 0] == 11.0)
    assert torch.all(manager.per_env_delay_queue.buffer[:, 2] == 33.0)


def test_random_delay_state_reports_each_environment_actual_delay():
    cfg = make_cfg(
        enabled=True,
        action_delay_enabled=True,
        delay_step=99,
        delay_step_choices=(0, 1, 2),
    )
    manager = RobustnessManager(cfg, 3, "cpu")
    env = make_env(3)
    manager._reset_action_delay(env, torch.arange(3))
    manager.sampled_delay_step[:] = torch.tensor([0, 1, 2])
    env.control_interface.command_velocity[:, 2] = torch.tensor([1.0, 10.0, 100.0])

    state = manager.after_command_update(env, env.control_interface.get_command_state())
    diagnostics = manager.get_state()

    assert torch.equal(state["executed_command_z"], torch.tensor([1.0, 0.0, 0.0]))
    assert torch.equal(diagnostics["delay_step"], torch.tensor([0.0, 1.0, 2.0]))
    assert torch.equal(diagnostics["action_delay_enabled"], torch.tensor([0.0, 1.0, 1.0]))
    assert torch.equal(diagnostics["delayed_command_z"], torch.tensor([1.0, 0.0, 0.0]))


def test_zero_only_random_delay_choice_is_an_exact_passthrough():
    cfg = make_cfg(
        enabled=True,
        action_delay_enabled=True,
        delay_step=99,
        delay_step_choices=(0,),
    )
    manager = RobustnessManager(cfg, 1, "cpu")
    env = make_env()
    manager._reset_action_delay(env, torch.tensor([0]))
    env.control_interface.set_command_z(0.75)

    state = manager.after_command_update(env, env.control_interface.get_command_state())

    assert manager.per_env_delay_queue is None
    assert state["executed_command_z"].item() == pytest.approx(0.75)
    assert manager.delay_step.item() == 0.0
    assert manager.action_delay_enabled.item() == 0.0


def test_disabled_response_is_exact_passthrough():
    cfg = make_cfg(enabled=False, velocity_response_enabled=True)
    manager = RobustnessManager(cfg, 1, "cpu")
    env = make_env()
    initialize_command_models(manager, env, torch.tensor([0]))
    env.control_interface.set_command_z(0.75)

    state = manager.after_command_update(env, env.control_interface.get_command_state())

    assert state["executed_command_z"].item() == pytest.approx(0.75)
    assert manager.delayed_command_z.item() == pytest.approx(0.75)
    assert manager.velocity_response_executed_z.item() == pytest.approx(0.75)
    assert manager.velocity_response_enabled.item() == 0.0


def test_delay_precedes_lead_lag_response():
    cfg = make_cfg(
        enabled=True,
        action_delay_enabled=True,
        delay_step=1,
        velocity_response_enabled=True,
        velocity_response_tau_s_range=(0.139, 0.139),
        velocity_response_gain_range=(1.0, 1.0),
        velocity_response_bias_range=(-0.00055, -0.00055),
    )
    manager = RobustnessManager(cfg, 1, "cpu")
    env = make_env()
    initialize_command_models(manager, env, torch.tensor([0]))

    env.control_interface.set_command_z(1.0)
    manager.after_command_update(env, env.control_interface.get_command_state())
    first_delayed_z = manager.delayed_command_z.item()
    first_response_input_z = manager.velocity_response_input_z.item()
    env.control_interface.set_command_z(1.0)
    second = manager.after_command_update(env, env.control_interface.get_command_state())

    assert first_delayed_z == 0.0
    assert first_response_input_z == 0.0
    assert manager.delayed_command_z.item() == 1.0
    assert torch.isfinite(second["executed_command_z"]).all()
    assert manager.velocity_response_input_z.item() == 1.0
    assert manager.velocity_response_executed_z.item() != pytest.approx(1.0)


def test_random_delay_precedes_lead_lag_response():
    cfg = make_cfg(
        enabled=True,
        action_delay_enabled=True,
        delay_step=99,
        delay_step_choices=(1,),
        velocity_response_enabled=True,
        velocity_response_tau_s_range=(0.139, 0.139),
        velocity_response_gain_range=(1.0, 1.0),
        velocity_response_bias_range=(-0.00055, -0.00055),
    )
    manager = RobustnessManager(cfg, 1, "cpu")
    env = make_env()
    initialize_command_models(manager, env, torch.tensor([0]))

    env.control_interface.set_command_z(1.0)
    first = manager.after_command_update(env, env.control_interface.get_command_state())

    assert manager.delay_queue is None
    assert manager.per_env_delay_queue is not None
    assert manager.delayed_command_z.item() == 0.0
    assert manager.velocity_response_input_z.item() == 0.0
    assert first["executed_command_z"].item() != pytest.approx(1.0)


@pytest.mark.parametrize(
    ("action_delay_enabled", "velocity_response_enabled", "expected_first"),
    [
        (False, False, 1.0),
        (True, False, 0.0),
        (False, True, 1.0),
        (True, True, 0.0),
    ],
)
def test_delay_and_response_flags_are_independent(
    action_delay_enabled: bool,
    velocity_response_enabled: bool,
    expected_first: float,
):
    cfg = make_cfg(
        enabled=True,
        action_delay_enabled=action_delay_enabled,
        delay_step=1,
        velocity_response_enabled=velocity_response_enabled,
        velocity_response_tau_s_range=(0.139, 0.139),
        velocity_response_gain_range=(1.0, 1.0),
        velocity_response_bias_range=(-0.00055, -0.00055),
    )
    manager = RobustnessManager(cfg, 1, "cpu")
    env = make_env()
    initialize_command_models(manager, env, torch.tensor([0]))
    env.control_interface.set_command_z(1.0)

    state = manager.after_command_update(env, env.control_interface.get_command_state())

    assert manager.delayed_command_z.item() == pytest.approx(expected_first)
    if velocity_response_enabled:
        assert manager.velocity_response_input_z.item() == pytest.approx(expected_first)
    else:
        assert state["executed_command_z"].item() == pytest.approx(expected_first)


def test_reset_samples_each_environment_and_preserves_unselected_state():
    torch.manual_seed(7)
    cfg = make_cfg(
        enabled=True,
        velocity_response_enabled=True,
        velocity_response_tau_s_range=(0.1, 0.3),
        velocity_response_gain_range=(0.8, 1.2),
        velocity_response_bias_range=(-0.1, 0.1),
        velocity_response_noise_std_range=(0.01, 0.03),
        velocity_response_ou_theta_range=(2.0, 4.0),
        velocity_response_ou_mu_range=(-0.02, 0.02),
    )
    manager = RobustnessManager(cfg, 3, "cpu")
    env = make_env(3)
    manager.velocity_response_nominal_z[:] = torch.tensor([10.0, 20.0, 30.0])
    env.control_interface.executed_velocity[:, 2] = torch.tensor([1.0, 2.0, 3.0])
    env.vrz[:] = torch.tensor([4.0, 5.0, 6.0])

    manager._reset_velocity_response(env, torch.tensor([1]))

    assert torch.equal(manager.velocity_response_nominal_z, torch.tensor([10.0, 5.0, 30.0]))
    assert manager.velocity_response_input_z[1].item() == pytest.approx(2.0)
    assert manager.velocity_response_compensated_command_z[1].item() == pytest.approx(2.0)
    assert 0.1 <= manager.velocity_response_tau_s[1].item() <= 0.3
    assert 0.8 <= manager.velocity_response_gain[1].item() <= 1.2
    assert manager.velocity_response_tau_s[0].item() == 0.0
    assert manager.velocity_response_gain[2].item() == 1.0


def test_invalid_configuration_and_interface_fail_fast():
    with pytest.raises(ValueError, match="ordered"):
        RobustnessManager(
            make_cfg(velocity_response_tau_s_range=(0.2, 0.1)),
            1,
            "cpu",
        )
    with pytest.raises(ValueError, match="noise mode"):
        RobustnessManager(
            make_cfg(velocity_response_noise_mode="pink"),
            1,
            "cpu",
        )
    with pytest.raises(ValueError, match="sim_tau_s"):
        RobustnessManager(
            make_cfg(velocity_response_sim_tau_s=0.0),
            1,
            "cpu",
        )
    with pytest.raises(ValueError, match="sim_gain"):
        RobustnessManager(
            make_cfg(velocity_response_sim_gain=0.0),
            1,
            "cpu",
        )
    with pytest.raises(ValueError, match="sim_bias"):
        RobustnessManager(
            make_cfg(velocity_response_sim_bias=float("nan")),
            1,
            "cpu",
        )
    with pytest.raises(ValueError, match="sim_max_abs_velocity"):
        RobustnessManager(
            make_cfg(velocity_response_sim_max_abs_velocity=0.1),
            1,
            "cpu",
        )

    manager = RobustnessManager(
        make_cfg(enabled=True, velocity_response_enabled=True),
        1,
        "cpu",
    )
    with pytest.raises(ValueError, match="interface_name='velocity'"):
        manager._reset_velocity_response(make_env(interface_name="position"), torch.tensor([0]))


def test_state_exposes_sim_parameters_and_compensated_command():
    cfg = make_cfg(
        enabled=True,
        velocity_response_enabled=True,
        velocity_response_tau_s_range=(0.155, 0.155),
        velocity_response_gain_range=(0.86, 0.86),
        velocity_response_bias_range=(-0.0035, -0.0035),
    )
    manager = RobustnessManager(cfg, 1, "cpu")
    env = make_env()
    initialize_command_models(manager, env, torch.tensor([0]))
    env.control_interface.set_command_z(0.4)

    manager.after_command_update(env, env.control_interface.get_command_state())
    state = manager.get_state()

    assert state["velocity_response_sim_tau_s"].item() == pytest.approx(
        cfg.velocity_response_sim_tau_s
    )
    assert state["velocity_response_sim_gain"].item() == pytest.approx(
        cfg.velocity_response_sim_gain
    )
    assert state["velocity_response_sim_bias"].item() == pytest.approx(
        cfg.velocity_response_sim_bias
    )
    assert state["velocity_response_compensated_command_z"].item() == pytest.approx(
        env.control_interface.executed_velocity[0, 2].item()
    )


def test_near_zero_simulator_step_response_fails_at_runtime():
    cfg = make_cfg(
        enabled=True,
        velocity_response_enabled=True,
        velocity_response_sim_tau_s=1.0e8,
    )
    manager = RobustnessManager(cfg, 1, "cpu")
    env = make_env()
    initialize_command_models(manager, env, torch.tensor([0]))
    env.control_interface.set_command_z(0.4)

    with pytest.raises(ValueError, match="near-zero"):
        manager.after_command_update(env, env.control_interface.get_command_state())
