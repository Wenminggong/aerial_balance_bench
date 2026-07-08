"""Robustness manager with benchmark randomization hooks."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from omni.isaac.lab.utils import configclass


OBS_PB_INDEX = 0
OBS_THETA_INDEX = 3


class CommandDelayQueue:
    """Fixed-step per-environment command delay queue."""

    def __init__(
        self,
        delay_step: int,
        num_envs: int,
        command_dim: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ):
        if delay_step <= 0:
            raise ValueError("CommandDelayQueue requires delay_step > 0.")
        self.delay_step = int(delay_step)
        self.num_envs = int(num_envs)
        self.command_dim = int(command_dim)
        self.device = device
        self.dtype = dtype
        self.buffer = torch.zeros((self.delay_step, self.num_envs, self.command_dim), device=device, dtype=dtype)
        self.read_index = 0

    def reset(self, env_ids: Sequence[int] | torch.Tensor, initial_command: torch.Tensor):
        """Fill selected environment queues with the current neutral command."""
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        initial_command = initial_command.to(device=self.device, dtype=self.dtype)
        self.buffer[:, env_ids] = initial_command[env_ids].unsqueeze(0)

    def step(self, command: torch.Tensor) -> torch.Tensor:
        """Return the delayed command and enqueue the current command."""
        command = command.to(device=self.device, dtype=self.dtype)
        delayed_command = self.buffer[self.read_index].clone()
        self.buffer[self.read_index] = command.clone()
        self.read_index = (self.read_index + 1) % self.delay_step
        return delayed_command


class PerEnvCommandDelayQueue:
    """Command delay queue with independently sampled delays for each environment."""

    def __init__(
        self,
        max_delay_step: int,
        num_envs: int,
        command_dim: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ):
        if max_delay_step < 0:
            raise ValueError("PerEnvCommandDelayQueue requires max_delay_step >= 0.")
        self.max_delay_step = int(max_delay_step)
        self.num_envs = int(num_envs)
        self.command_dim = int(command_dim)
        self.device = device
        self.dtype = dtype
        self.buffer_length = self.max_delay_step + 1
        self.buffer = torch.zeros(
            (self.buffer_length, self.num_envs, self.command_dim),
            device=device,
            dtype=dtype,
        )
        self.write_index = -1
        self.env_indices = torch.arange(self.num_envs, dtype=torch.long, device=device)

    def reset(self, env_ids: Sequence[int] | torch.Tensor, initial_command: torch.Tensor):
        """Fill selected environment histories with the current neutral command."""
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        initial_command = initial_command.to(device=self.device, dtype=self.dtype)
        self.buffer[:, env_ids] = initial_command[env_ids].unsqueeze(0)

    def step(self, command: torch.Tensor, delay_steps: torch.Tensor) -> torch.Tensor:
        """Enqueue the current command and return each environment's delayed command."""
        command = command.to(device=self.device, dtype=self.dtype)
        delay_steps = delay_steps.to(device=self.device, dtype=torch.long)
        if command.shape != (self.num_envs, self.command_dim):
            raise ValueError(
                "PerEnvCommandDelayQueue expected command shape "
                f"({self.num_envs}, {self.command_dim}), got {tuple(command.shape)}."
            )
        if delay_steps.shape != (self.num_envs,):
            raise ValueError(
                f"PerEnvCommandDelayQueue expected delay_steps shape ({self.num_envs},), "
                f"got {tuple(delay_steps.shape)}."
            )
        if torch.any(delay_steps < 0) or torch.any(delay_steps > self.max_delay_step):
            raise ValueError("PerEnvCommandDelayQueue delay_steps must be within [0, max_delay_step].")

        self.write_index = (self.write_index + 1) % self.buffer_length
        self.buffer[self.write_index] = command.clone()
        read_indices = (self.write_index - delay_steps) % self.buffer_length
        return self.buffer[read_indices, self.env_indices].clone()


@configclass
class RobustnessManagerCfg:
    """Configuration for benchmark robustness tests."""

    enabled: bool = False
    ball_mass_variation_enabled: bool = False
    ball_mass_range: tuple[float, float] = (0.0005, 0.0005)
    recompute_ball_inertia: bool = True
    controller_gain_variation_enabled: bool = False
    controller_gain_range: tuple[float, float] = (10.0, 10.0)
    action_delay_enabled: bool = False
    delay_step: int = 0
    delay_step_choices: tuple[int, ...] = ()
    acceleration_response_enabled: bool = False
    acceleration_response_tau_s_range: tuple[float, float] = (0.0, 0.0)
    acceleration_response_gain_range: tuple[float, float] = (1.0, 1.0)
    acceleration_response_bias_range: tuple[float, float] = (0.0, 0.0)
    acceleration_response_noise_mode: str = "none"
    acceleration_response_noise_std_range: tuple[float, float] = (0.0, 0.0)
    acceleration_response_ou_theta_range: tuple[float, float] = (8.0, 16.0)
    acceleration_response_noise_clip: float = 0.0
    acceleration_response_max_abs_acc: float = 0.0
    external_disturbance_enabled: bool = False
    external_disturbance_ou_mu: float = 0.0
    external_disturbance_ou_theta_range: tuple[float, float] = (0.1, 0.3)
    external_disturbance_ou_sigma_range: tuple[float, float] = (0.01, 0.05)
    external_disturbance_ou_clip: float = 0.05
    observation_degradation_enabled: bool = False
    obs_theta_noise_std: float = 0.0
    obs_theta_outlier_prob: float = 0.0
    obs_theta_outlier_std: float = 0.0
    obs_theta_quantization: float = 0.0
    obs_theta_delay_step: int = 0
    obs_theta_hold_prob: float = 0.0
    obs_pb_noise_std: float = 0.0
    obs_pb_outlier_prob: float = 0.0
    obs_pb_outlier_std: float = 0.0
    obs_pb_quantization: float = 0.0
    obs_pb_delay_step: int = 0
    obs_pb_hold_prob: float = 0.0
    observation_filter_enabled: bool = False
    obs_theta_filter_cutoff_hz: float = 0.0
    obs_pb_filter_cutoff_hz: float = 0.0


class RobustnessManager:
    """Hook object for reset-time randomization and step-time perturbations.

    This manager currently implements reset-time ball-mass variation,
    low-level controller gain variation, command-level action delay,
    OU-process fixed-end external disturbances, and policy-observation
    degradation for selected benchmark observation fields.
    """

    def __init__(self, cfg: RobustnessManagerCfg, num_envs: int, device: str | torch.device):
        self.cfg = cfg
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.ball_mass = torch.full((num_envs,), 0.0005, device=self.device)
        self.low_level_controller_gain = torch.full((num_envs,), float("nan"), device=self.device)
        self.position_gain = torch.full((num_envs,), float("nan"), device=self.device)
        self.velocity_gain = torch.full((num_envs,), float("nan"), device=self.device)
        self.attitude_gain = torch.full((num_envs,), float("nan"), device=self.device)
        self.delay_queue: CommandDelayQueue | None = None
        self.per_env_delay_queue: PerEnvCommandDelayQueue | None = None
        self.delay_step_choices = self._validate_delay_step_choices(cfg.delay_step_choices)
        initial_delay = self._initial_delay_step()
        self.sampled_delay_step = torch.full((num_envs,), initial_delay, dtype=torch.long, device=self.device)
        self.action_delay_enabled = torch.full(
            (num_envs,),
            float(self._action_delay_active() and initial_delay > 0),
            device=self.device,
        )
        self.delay_step = self.sampled_delay_step.to(dtype=torch.float32)
        self.delayed_command_z = torch.full((num_envs,), float("nan"), device=self.device)
        self.acceleration_response_noise_mode = self._validate_acceleration_response_noise_mode(
            cfg.acceleration_response_noise_mode
        )
        self.acceleration_response_enabled = torch.full(
            (num_envs,),
            float(self._acceleration_response_active()),
            device=self.device,
        )
        self.acceleration_response_tau_s = torch.zeros(num_envs, device=self.device)
        self.acceleration_response_gain = torch.ones(num_envs, device=self.device)
        self.acceleration_response_bias = torch.zeros(num_envs, device=self.device)
        self.acceleration_response_noise_std = torch.zeros(num_envs, device=self.device)
        self.acceleration_response_ou_theta = torch.zeros(num_envs, device=self.device)
        self.acceleration_response_noise_z = torch.zeros(num_envs, device=self.device)
        self.acceleration_response_target_z = torch.zeros(num_envs, device=self.device)
        self.acceleration_response_executed_z = torch.zeros(num_envs, device=self.device)
        self.acceleration_response_error_z = torch.zeros(num_envs, device=self.device)
        self.external_disturbance_enabled = torch.full(
            (num_envs,),
            float(self._external_disturbance_active()),
            device=self.device,
        )
        self.external_disturbance_vel_z = torch.zeros((num_envs, 1), device=self.device)
        self.external_disturbance_ou_theta = torch.zeros((num_envs, 1), device=self.device)
        self.external_disturbance_ou_sigma = torch.zeros((num_envs, 1), device=self.device)
        self.observation_degradation_enabled = torch.full(
            (num_envs,),
            float(self._observation_degradation_active()),
            device=self.device,
        )
        self.obs_pb_clean = torch.zeros(num_envs, device=self.device)
        self.obs_pb_corrupted = torch.zeros(num_envs, device=self.device)
        self.obs_pb_noise = torch.zeros(num_envs, device=self.device)
        self.obs_pb_filtered = torch.zeros(num_envs, device=self.device)
        self.obs_pb_filter_noise = torch.zeros(num_envs, device=self.device)
        self.obs_theta_clean = torch.zeros(num_envs, device=self.device)
        self.obs_theta_corrupted = torch.zeros(num_envs, device=self.device)
        self.obs_theta_noise = torch.zeros(num_envs, device=self.device)
        self.obs_theta_filtered = torch.zeros(num_envs, device=self.device)
        self.obs_theta_filter_noise = torch.zeros(num_envs, device=self.device)
        self.observation_filter_enabled = torch.full(
            (num_envs,),
            float(self._observation_filter_active()),
            device=self.device,
        )
        self.obs_pb_initialized = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self.obs_theta_initialized = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self.obs_pb_last_held = torch.zeros(num_envs, device=self.device)
        self.obs_theta_last_held = torch.zeros(num_envs, device=self.device)
        self.obs_pb_delay_queue: CommandDelayQueue | None = None
        self.obs_theta_delay_queue: CommandDelayQueue | None = None
        self.obs_pb_filter_initialized = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self.obs_theta_filter_initialized = torch.zeros(num_envs, dtype=torch.bool, device=self.device)

    def reset(self, env, env_ids: Sequence[int] | torch.Tensor):
        """Apply reset-time randomization for selected environments."""
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        if env_ids.numel() == 0:
            return None

        if self.cfg.enabled and self.cfg.ball_mass_variation_enabled:
            sampled_mass = self._sample_ball_mass(env_ids.numel())
            self._write_ball_mass_to_sim(env, env_ids, sampled_mass)
        else:
            self._sync_ball_mass_from_sim(env, env_ids)

        if self.cfg.enabled and self.cfg.controller_gain_variation_enabled:
            sampled_gain = self._sample_controller_gain(env_ids.numel())
            self._write_controller_gain(env, env_ids, sampled_gain)
        else:
            self._sync_controller_gain(env, env_ids)

        self._reset_action_delay(env, env_ids)
        self._reset_acceleration_response(env, env_ids)
        self._reset_external_disturbance(env, env_ids)
        self._reset_observation_degradation(env_ids)
        self._reset_observation_filter(env_ids)
        return None

    def before_action(self, env, action: torch.Tensor) -> torch.Tensor:
        """Modify high-level actions before interface processing. No-op."""
        return action

    def after_command_update(self, env, command: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Apply command-level action delay and optional acceleration-response degradation."""
        if self._action_delay_active():
            current_command = env.control_interface.get_delay_command()
            if self._random_action_delay_enabled():
                self._ensure_per_env_delay_queue(current_command)
                delayed_command = self.per_env_delay_queue.step(current_command, self.sampled_delay_step)
            else:
                self._ensure_delay_queue(current_command)
                delayed_command = self.delay_queue.step(current_command)
            env.control_interface.set_executed_delay_command(delayed_command)
            command_state = env.control_interface.get_command_state()
        else:
            command_state = command

        self._sync_delayed_command_z(command_state)
        if not self._acceleration_response_active():
            return command_state
        return self._apply_acceleration_response(env, command_state)

    def after_physics_step(self, env):
        """Apply step-time perturbations after each physics step."""
        if not self._external_disturbance_active():
            self.external_disturbance_enabled[:] = 0.0
            self.external_disturbance_vel_z.zero_()
            self._set_external_disturbance_target(env)
            return None

        dt = float(getattr(env, "physics_dt", 0.0))
        if dt <= 0.0:
            raise ValueError("env.physics_dt must be positive for external disturbance integration.")

        mu = float(self.cfg.external_disturbance_ou_mu)
        noise_delta = self.external_disturbance_ou_theta * (mu - self.external_disturbance_vel_z) * dt
        noise_delta += self.external_disturbance_ou_sigma * math.sqrt(dt) * torch.randn_like(
            self.external_disturbance_vel_z
        )
        self.external_disturbance_vel_z += noise_delta

        clip = float(self.cfg.external_disturbance_ou_clip)
        if clip > 0.0:
            self.external_disturbance_vel_z.clamp_(min=-clip, max=clip)

        self.external_disturbance_enabled[:] = 1.0
        self._set_external_disturbance_target(env)
        return None

    def corrupt_observation(self, env, obs: torch.Tensor) -> torch.Tensor:
        """Return the policy observation after optional measurement degradation."""
        obs = obs.to(device=self.device, dtype=torch.float32)
        degradation_active = self._observation_degradation_active()
        filter_active = self._observation_filter_active()
        if not degradation_active and not filter_active:
            self._sync_observation_diagnostics(obs, obs, obs)
            return obs

        corrupted = obs.clone()
        if degradation_active:
            corrupted[:, OBS_PB_INDEX] = self._corrupt_observation_signal(
                "pb",
                obs[:, OBS_PB_INDEX],
                noise_std=self.cfg.obs_pb_noise_std,
                outlier_prob=self.cfg.obs_pb_outlier_prob,
                outlier_std=self.cfg.obs_pb_outlier_std,
                quantization=self.cfg.obs_pb_quantization,
                hold_prob=self.cfg.obs_pb_hold_prob,
                delay_step=self.cfg.obs_pb_delay_step,
            )
            corrupted[:, OBS_THETA_INDEX] = self._corrupt_observation_signal(
                "theta",
                obs[:, OBS_THETA_INDEX],
                noise_std=self.cfg.obs_theta_noise_std,
                outlier_prob=self.cfg.obs_theta_outlier_prob,
                outlier_std=self.cfg.obs_theta_outlier_std,
                quantization=self.cfg.obs_theta_quantization,
                hold_prob=self.cfg.obs_theta_hold_prob,
                delay_step=self.cfg.obs_theta_delay_step,
            )

        filtered = corrupted.clone()
        if filter_active:
            step_dt = float(getattr(env, "step_dt", 0.0))
            if step_dt <= 0.0:
                raise ValueError("env.step_dt must be positive for observation filtering.")
            filtered[:, OBS_PB_INDEX] = self._filter_observation_signal(
                "pb",
                corrupted[:, OBS_PB_INDEX],
                cutoff_hz=self.cfg.obs_pb_filter_cutoff_hz,
                step_dt=step_dt,
            )
            filtered[:, OBS_THETA_INDEX] = self._filter_observation_signal(
                "theta",
                corrupted[:, OBS_THETA_INDEX],
                cutoff_hz=self.cfg.obs_theta_filter_cutoff_hz,
                step_dt=step_dt,
            )

        self._sync_observation_diagnostics(obs, corrupted, filtered)
        return filtered

    def get_state(self) -> dict[str, torch.Tensor]:
        """Return robustness state for logging and diagnostics."""
        return {
            "ball_mass": self.ball_mass,
            "low_level_controller_gain": self.low_level_controller_gain,
            "position_gain": self.position_gain,
            "velocity_gain": self.velocity_gain,
            "attitude_gain": self.attitude_gain,
            "action_delay_enabled": self.action_delay_enabled,
            "delay_step": self.delay_step,
            "delayed_command_z": self.delayed_command_z,
            "acceleration_response_enabled": self.acceleration_response_enabled,
            "acceleration_response_tau_s": self.acceleration_response_tau_s,
            "acceleration_response_gain": self.acceleration_response_gain,
            "acceleration_response_bias": self.acceleration_response_bias,
            "acceleration_response_noise_std": self.acceleration_response_noise_std,
            "acceleration_response_ou_theta": self.acceleration_response_ou_theta,
            "acceleration_response_noise_z": self.acceleration_response_noise_z,
            "acceleration_response_target_z": self.acceleration_response_target_z,
            "acceleration_response_executed_z": self.acceleration_response_executed_z,
            "acceleration_response_error_z": self.acceleration_response_error_z,
            "external_disturbance_enabled": self.external_disturbance_enabled,
            "external_disturbance_vel_z": self.external_disturbance_vel_z[:, 0],
            "external_disturbance_ou_theta": self.external_disturbance_ou_theta[:, 0],
            "external_disturbance_ou_sigma": self.external_disturbance_ou_sigma[:, 0],
            "observation_degradation_enabled": self.observation_degradation_enabled,
            "obs_pb_clean": self.obs_pb_clean,
            "obs_pb_corrupted": self.obs_pb_corrupted,
            "obs_pb_noise": self.obs_pb_noise,
            "obs_pb_filtered": self.obs_pb_filtered,
            "obs_pb_filter_noise": self.obs_pb_filter_noise,
            "obs_theta_clean": self.obs_theta_clean,
            "obs_theta_corrupted": self.obs_theta_corrupted,
            "obs_theta_noise": self.obs_theta_noise,
            "obs_theta_filtered": self.obs_theta_filtered,
            "obs_theta_filter_noise": self.obs_theta_filter_noise,
            "observation_filter_enabled": self.observation_filter_enabled,
        }

    def _sample_ball_mass(self, count: int) -> torch.Tensor:
        min_mass, max_mass = self._ball_mass_bounds()
        unit = torch.rand(count, device=self.device)
        return min_mass + (max_mass - min_mass) * unit

    def _ball_mass_bounds(self) -> tuple[float, float]:
        min_mass, max_mass = (float(value) for value in self.cfg.ball_mass_range)
        if min_mass <= 0.0 or max_mass <= 0.0:
            raise ValueError("robustness.ball_mass_range values must be positive.")
        if min_mass > max_mass:
            raise ValueError("robustness.ball_mass_range must be ordered as [min_mass, max_mass].")
        return min_mass, max_mass

    def _write_ball_mass_to_sim(self, env, env_ids: torch.Tensor, sampled_mass: torch.Tensor):
        env_ids_cpu = env_ids.detach().to(device="cpu", dtype=torch.int)
        sampled_mass_cpu = sampled_mass.detach().to(device="cpu", dtype=torch.float32)

        masses = env.ball.root_physx_view.get_masses()
        masses[env_ids_cpu, 0] = sampled_mass_cpu
        env.ball.root_physx_view.set_masses(masses, env_ids_cpu)

        if self.cfg.recompute_ball_inertia:
            default_mass = env.ball.data.default_mass.detach().to(device="cpu", dtype=torch.float32)
            default_inertia = env.ball.data.default_inertia.detach().to(device="cpu", dtype=torch.float32)
            ratios = sampled_mass_cpu / default_mass[env_ids_cpu, 0].clamp_min(1e-12)
            inertias = env.ball.root_physx_view.get_inertias()
            inertias[env_ids_cpu] = default_inertia[env_ids_cpu] * ratios.unsqueeze(-1)
            env.ball.root_physx_view.set_inertias(inertias, env_ids_cpu)

        self.ball_mass[env_ids] = sampled_mass

    def _sync_ball_mass_from_sim(self, env, env_ids: torch.Tensor):
        env_ids_cpu = env_ids.detach().to(device="cpu", dtype=torch.int)
        masses = env.ball.root_physx_view.get_masses()
        self.ball_mass[env_ids] = masses[env_ids_cpu, 0].to(device=self.device, dtype=torch.float32)

    def _sample_controller_gain(self, count: int) -> torch.Tensor:
        min_gain, max_gain = self._controller_gain_bounds()
        unit = torch.rand(count, device=self.device)
        return min_gain + (max_gain - min_gain) * unit

    def _controller_gain_bounds(self) -> tuple[float, float]:
        min_gain, max_gain = (float(value) for value in self.cfg.controller_gain_range)
        if min_gain <= 0.0 or max_gain <= 0.0:
            raise ValueError("robustness.controller_gain_range values must be positive.")
        if min_gain > max_gain:
            raise ValueError("robustness.controller_gain_range must be ordered as [min_gain, max_gain].")
        return min_gain, max_gain

    def _write_controller_gain(self, env, env_ids: torch.Tensor, sampled_gain: torch.Tensor):
        controller, gain_name, log_name = self._active_controller_gain(env)
        controller.set_gain(gain_name, env_ids, sampled_gain)
        self._record_controller_gain(env_ids, log_name, sampled_gain)

    def _sync_controller_gain(self, env, env_ids: torch.Tensor):
        controller, gain_name, log_name = self._active_controller_gain(env)
        gains = controller.get_gain(gain_name).to(device=self.device, dtype=torch.float32)
        if gains.ndim == 1:
            values = gains[0].repeat(env_ids.numel())
        else:
            values = gains[env_ids, 0]
        self._record_controller_gain(env_ids, log_name, values)

    def _active_controller_gain(self, env):
        interface_name = env.cfg.interface_name
        if interface_name == "acceleration":
            return env.control_interface.attitude_controller, "attitude_gain", "attitude_gain"
        if interface_name == "position":
            return env.control_interface.position_controller, "pos_gain", "position_gain"
        if interface_name == "velocity":
            return env.control_interface.velocity_controller, "vel_gain", "velocity_gain"
        if interface_name == "thrust":
            return env.control_interface.attitude_controller, "attitude_gain", "attitude_gain"
        raise ValueError(f"Unsupported interface_name '{interface_name}' for controller gain variation.")

    def _record_controller_gain(self, env_ids: torch.Tensor, log_name: str, values: torch.Tensor):
        values = values.to(device=self.device, dtype=torch.float32)
        self.low_level_controller_gain[env_ids] = values
        self.position_gain[env_ids] = float("nan")
        self.velocity_gain[env_ids] = float("nan")
        self.attitude_gain[env_ids] = float("nan")
        getattr(self, log_name)[env_ids] = values

    def _action_delay_active(self) -> bool:
        return bool(
            self.cfg.enabled
            and self.cfg.action_delay_enabled
            and self._max_configured_action_delay_step() > 0
        )

    def _reset_action_delay(self, env, env_ids: torch.Tensor):
        sampled_delay_step = self._sample_action_delay_steps(env_ids)
        self.sampled_delay_step[env_ids] = sampled_delay_step
        if self._action_delay_active():
            self.action_delay_enabled[env_ids] = (sampled_delay_step > 0).to(dtype=torch.float32)
        else:
            self.action_delay_enabled[env_ids] = 0.0
        self.delay_step[env_ids] = sampled_delay_step.to(dtype=torch.float32)

        command = env.control_interface.get_command_state()
        if not self._action_delay_active():
            self.delayed_command_z[env_ids] = command["executed_command_z"][env_ids].to(
                device=self.device,
                dtype=torch.float32,
            )
            return

        current_command = env.control_interface.get_delay_command()
        if self._random_action_delay_enabled():
            self._ensure_per_env_delay_queue(current_command)
            self.per_env_delay_queue.reset(env_ids, current_command)
        else:
            self._ensure_delay_queue(current_command)
            self.delay_queue.reset(env_ids, current_command)
        env.control_interface.set_executed_delay_command(current_command)
        command = env.control_interface.get_command_state()
        self.delayed_command_z[env_ids] = command["executed_command_z"][env_ids].to(
            device=self.device,
            dtype=torch.float32,
        )

    def _ensure_delay_queue(self, command: torch.Tensor):
        delay_step = int(self.cfg.delay_step)
        command_dim = int(command.shape[-1])
        if (
            self.delay_queue is None
            or self.delay_queue.delay_step != delay_step
            or self.delay_queue.command_dim != command_dim
        ):
            self.delay_queue = CommandDelayQueue(
                delay_step=delay_step,
                num_envs=self.num_envs,
                command_dim=command_dim,
                device=self.device,
                dtype=command.dtype,
            )

    def _ensure_per_env_delay_queue(self, command: torch.Tensor):
        max_delay_step = self._max_configured_action_delay_step()
        command_dim = int(command.shape[-1])
        if (
            self.per_env_delay_queue is None
            or self.per_env_delay_queue.max_delay_step != max_delay_step
            or self.per_env_delay_queue.command_dim != command_dim
        ):
            self.per_env_delay_queue = PerEnvCommandDelayQueue(
                max_delay_step=max_delay_step,
                num_envs=self.num_envs,
                command_dim=command_dim,
                device=self.device,
                dtype=command.dtype,
            )
            env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
            self.per_env_delay_queue.reset(env_ids, command)

    def _sync_delayed_command_z(self, command: dict[str, torch.Tensor]):
        if self._random_action_delay_enabled():
            self.action_delay_enabled[:] = (self.sampled_delay_step > 0).to(dtype=torch.float32)
            self.delay_step[:] = self.sampled_delay_step.to(dtype=torch.float32)
        else:
            self.action_delay_enabled[:] = float(self._action_delay_active())
            self.delay_step[:] = float(max(int(self.cfg.delay_step), 0))
        self.delayed_command_z[:] = command["executed_command_z"].to(device=self.device, dtype=torch.float32)

    def _acceleration_response_active(self) -> bool:
        return bool(self.cfg.enabled and self.cfg.acceleration_response_enabled)

    def _reset_acceleration_response(self, env, env_ids: torch.Tensor):
        active = self._acceleration_response_active()
        if active:
            self._require_acceleration_interface(env)
            self._validate_acceleration_response_scalar_limits()

        self.acceleration_response_enabled[env_ids] = float(active)
        command = env.control_interface.get_command_state()
        executed_z = command["executed_command_z"][env_ids].to(device=self.device, dtype=torch.float32)

        if not active:
            self.acceleration_response_tau_s[env_ids] = 0.0
            self.acceleration_response_gain[env_ids] = 1.0
            self.acceleration_response_bias[env_ids] = 0.0
            self.acceleration_response_noise_std[env_ids] = 0.0
            self.acceleration_response_ou_theta[env_ids] = 0.0
            self.acceleration_response_noise_z[env_ids] = 0.0
            self.acceleration_response_target_z[env_ids] = executed_z
            self.acceleration_response_executed_z[env_ids] = executed_z
            self.acceleration_response_error_z[env_ids] = 0.0
            return

        count = int(env_ids.numel())
        self.acceleration_response_tau_s[env_ids] = self._sample_acceleration_response_parameter(
            count,
            self.cfg.acceleration_response_tau_s_range,
            "acceleration_response_tau_s_range",
            lower_bound=0.0,
        )
        self.acceleration_response_gain[env_ids] = self._sample_acceleration_response_parameter(
            count,
            self.cfg.acceleration_response_gain_range,
            "acceleration_response_gain_range",
            lower_bound=0.0,
            strict_lower=True,
        )
        self.acceleration_response_bias[env_ids] = self._sample_acceleration_response_parameter(
            count,
            self.cfg.acceleration_response_bias_range,
            "acceleration_response_bias_range",
        )
        self.acceleration_response_noise_std[env_ids] = self._sample_acceleration_response_parameter(
            count,
            self.cfg.acceleration_response_noise_std_range,
            "acceleration_response_noise_std_range",
            lower_bound=0.0,
        )
        self.acceleration_response_ou_theta[env_ids] = self._sample_acceleration_response_parameter(
            count,
            self.cfg.acceleration_response_ou_theta_range,
            "acceleration_response_ou_theta_range",
            lower_bound=0.0,
        )
        self.acceleration_response_noise_z[env_ids] = 0.0
        self.acceleration_response_target_z[env_ids] = executed_z
        self.acceleration_response_executed_z[env_ids] = executed_z
        self.acceleration_response_error_z[env_ids] = 0.0

    def _apply_acceleration_response(self, env, command: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        self._require_acceleration_interface(env)
        step_dt = float(getattr(env, "step_dt", 0.0))
        if step_dt <= 0.0:
            raise ValueError("env.step_dt must be positive for acceleration response integration.")
        if "executed_acceleration" not in command:
            raise ValueError("Acceleration response requires an 'executed_acceleration' command state.")

        response_command = command["executed_acceleration"].to(device=self.device, dtype=torch.float32).clone()
        reference_z = response_command[:, 2]
        noise_z = self._acceleration_response_noise(step_dt)
        target_z = self.acceleration_response_gain * reference_z + self.acceleration_response_bias + noise_z

        tau_s = self.acceleration_response_tau_s
        filtered_mask = tau_s > 0.0
        alpha = torch.zeros_like(tau_s)
        alpha[filtered_mask] = 1.0 - torch.exp(-step_dt / tau_s[filtered_mask])
        filtered_z = self.acceleration_response_executed_z + alpha * (
            target_z - self.acceleration_response_executed_z
        )
        executed_z = torch.where(filtered_mask, filtered_z, target_z)

        max_abs_acc = float(self.cfg.acceleration_response_max_abs_acc)
        if max_abs_acc > 0.0:
            executed_z = torch.clamp(executed_z, min=-max_abs_acc, max=max_abs_acc)

        active_mask = self.acceleration_response_enabled > 0.5
        target_z = torch.where(active_mask, target_z, reference_z)
        executed_z = torch.where(active_mask, executed_z, reference_z)
        response_command[:, 2] = executed_z

        self.acceleration_response_target_z[:] = target_z
        self.acceleration_response_executed_z[:] = executed_z
        self.acceleration_response_error_z[:] = executed_z - reference_z

        env.control_interface.set_executed_delay_command(response_command)
        return env.control_interface.get_command_state()

    def _acceleration_response_noise(self, step_dt: float) -> torch.Tensor:
        mode = self.acceleration_response_noise_mode
        if mode == "none":
            self.acceleration_response_noise_z.zero_()
            return self.acceleration_response_noise_z

        if mode == "white":
            noise = self.acceleration_response_noise_std * torch.randn_like(self.acceleration_response_noise_z)
        elif mode == "ou":
            theta = self.acceleration_response_ou_theta.clamp_min(0.0)
            std = self.acceleration_response_noise_std.clamp_min(0.0)
            noise_delta = theta * (0.0 - self.acceleration_response_noise_z) * step_dt
            noise_delta += torch.sqrt(2.0 * theta) * std * math.sqrt(step_dt) * torch.randn_like(
                self.acceleration_response_noise_z
            )
            noise = self.acceleration_response_noise_z + noise_delta
        else:  # pragma: no cover - mode is validated at construction.
            raise ValueError(f"Unsupported acceleration_response_noise_mode '{mode}'.")

        clip = float(self.cfg.acceleration_response_noise_clip)
        if clip > 0.0:
            noise = torch.clamp(noise, min=-clip, max=clip)
        self.acceleration_response_noise_z[:] = noise
        return self.acceleration_response_noise_z

    def _require_acceleration_interface(self, env):
        if getattr(env.cfg, "interface_name", None) != "acceleration":
            raise ValueError("robustness.acceleration_response_enabled requires interface_name='acceleration'.")

    def _validate_acceleration_response_scalar_limits(self):
        noise_clip = float(self.cfg.acceleration_response_noise_clip)
        if noise_clip < 0.0:
            raise ValueError("robustness.acceleration_response_noise_clip must be non-negative.")
        max_abs_acc = float(self.cfg.acceleration_response_max_abs_acc)
        if max_abs_acc < 0.0:
            raise ValueError("robustness.acceleration_response_max_abs_acc must be non-negative.")

    def _sample_acceleration_response_parameter(
        self,
        count: int,
        bounds: Sequence[float],
        field_name: str,
        *,
        lower_bound: float | None = None,
        strict_lower: bool = False,
    ) -> torch.Tensor:
        min_value, max_value = (float(value) for value in bounds)
        if min_value > max_value:
            raise ValueError(f"robustness.{field_name} must be ordered as [min, max].")
        if lower_bound is not None:
            if strict_lower:
                invalid = min_value <= lower_bound or max_value <= lower_bound
                relation = f"> {lower_bound:g}"
            else:
                invalid = min_value < lower_bound or max_value < lower_bound
                relation = f">= {lower_bound:g}"
            if invalid:
                raise ValueError(f"robustness.{field_name} values must be {relation}.")
        unit = torch.rand(count, device=self.device)
        return min_value + (max_value - min_value) * unit

    @staticmethod
    def _validate_acceleration_response_noise_mode(value: str) -> str:
        mode = str(value).lower()
        if mode not in {"none", "white", "ou"}:
            raise ValueError("robustness.acceleration_response_noise_mode must be 'none', 'white', or 'ou'.")
        return mode

    def _random_action_delay_enabled(self) -> bool:
        return len(self.delay_step_choices) > 0

    def _initial_delay_step(self) -> int:
        if self._random_action_delay_enabled():
            if not self.cfg.enabled or not self.cfg.action_delay_enabled:
                return 0
            return max(self.delay_step_choices)
        return max(int(self.cfg.delay_step), 0)

    def _max_configured_action_delay_step(self) -> int:
        if self._random_action_delay_enabled():
            return max(self.delay_step_choices)
        return max(int(self.cfg.delay_step), 0)

    def _sample_action_delay_steps(self, env_ids: torch.Tensor) -> torch.Tensor:
        if not self._random_action_delay_enabled():
            fixed_delay = max(int(self.cfg.delay_step), 0)
            return torch.full((env_ids.numel(),), fixed_delay, dtype=torch.long, device=self.device)
        if not self.cfg.enabled or not self.cfg.action_delay_enabled:
            return torch.zeros((env_ids.numel(),), dtype=torch.long, device=self.device)

        choices = torch.as_tensor(self.delay_step_choices, dtype=torch.long, device=self.device)
        choice_ids = torch.randint(0, choices.numel(), (env_ids.numel(),), device=self.device)
        return choices[choice_ids]

    @staticmethod
    def _validate_delay_step_choices(values: Sequence[int] | None) -> tuple[int, ...]:
        if values in (None, ""):
            return ()
        choices: list[int] = []
        for value in values:
            if isinstance(value, bool):
                raise ValueError("robustness.delay_step_choices values must be non-negative integers.")
            if isinstance(value, float) and not value.is_integer():
                raise ValueError("robustness.delay_step_choices values must be non-negative integers.")
            try:
                step = int(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "robustness.delay_step_choices values must be non-negative integers."
                ) from exc
            if step < 0:
                raise ValueError("robustness.delay_step_choices values must be non-negative integers.")
            choices.append(step)
        return tuple(choices)

    def _external_disturbance_active(self) -> bool:
        return bool(self.cfg.enabled and self.cfg.external_disturbance_enabled)

    def _reset_external_disturbance(self, env, env_ids: torch.Tensor):
        active = self._external_disturbance_active()
        self.external_disturbance_enabled[env_ids] = float(active)
        self.external_disturbance_vel_z[env_ids] = 0.0

        if not active:
            self.external_disturbance_ou_theta[env_ids] = 0.0
            self.external_disturbance_ou_sigma[env_ids] = 0.0
            self._set_external_disturbance_target(env)
            return

        count = int(env_ids.numel())
        self.external_disturbance_ou_theta[env_ids, 0] = self._sample_ou_parameter(
            count,
            self.cfg.external_disturbance_ou_theta_range,
            "external_disturbance_ou_theta_range",
        )
        self.external_disturbance_ou_sigma[env_ids, 0] = self._sample_ou_parameter(
            count,
            self.cfg.external_disturbance_ou_sigma_range,
            "external_disturbance_ou_sigma_range",
        )
        self._set_external_disturbance_target(env)

    def _sample_ou_parameter(self, count: int, bounds: Sequence[float], field_name: str) -> torch.Tensor:
        min_value, max_value = (float(value) for value in bounds)
        if min_value < 0.0 or max_value < 0.0:
            raise ValueError(f"robustness.{field_name} values must be non-negative.")
        if min_value > max_value:
            raise ValueError(f"robustness.{field_name} must be ordered as [min, max].")
        unit = torch.rand(count, device=self.device)
        return min_value + (max_value - min_value) * unit

    def _set_external_disturbance_target(self, env):
        env.drone_rope_plank.set_joint_velocity_target(
            target=self.external_disturbance_vel_z,
            joint_ids=env.slider_holder_joint_ids,
        )

    def _observation_degradation_active(self) -> bool:
        return bool(self.cfg.enabled and self.cfg.observation_degradation_enabled)

    def _observation_filter_active(self) -> bool:
        return bool(self.cfg.enabled and self.cfg.observation_filter_enabled)

    def _reset_observation_degradation(self, env_ids: torch.Tensor):
        active = self._observation_degradation_active()
        self.observation_degradation_enabled[env_ids] = float(active)
        self.obs_pb_initialized[env_ids] = False
        self.obs_theta_initialized[env_ids] = False
        self.obs_pb_last_held[env_ids] = 0.0
        self.obs_theta_last_held[env_ids] = 0.0
        for buffer in (
            self.obs_pb_clean,
            self.obs_pb_corrupted,
            self.obs_pb_noise,
            self.obs_theta_clean,
            self.obs_theta_corrupted,
            self.obs_theta_noise,
        ):
            buffer[env_ids] = 0.0

    def _reset_observation_filter(self, env_ids: torch.Tensor):
        active = self._observation_filter_active()
        self.observation_filter_enabled[env_ids] = float(active)
        self.obs_pb_filter_initialized[env_ids] = False
        self.obs_theta_filter_initialized[env_ids] = False
        for buffer in (
            self.obs_pb_filtered,
            self.obs_pb_filter_noise,
            self.obs_theta_filtered,
            self.obs_theta_filter_noise,
        ):
            buffer[env_ids] = 0.0

    def _corrupt_observation_signal(
        self,
        signal_name: str,
        clean_value: torch.Tensor,
        *,
        noise_std: float,
        outlier_prob: float,
        outlier_std: float,
        quantization: float,
        hold_prob: float,
        delay_step: int,
    ) -> torch.Tensor:
        clean_value = clean_value.to(device=self.device, dtype=torch.float32)
        initialized, last_held = self._observation_signal_state(signal_name)
        was_initialized = initialized.clone()

        uninitialized_env_ids = (~initialized).nonzero(as_tuple=False).squeeze(-1)
        if uninitialized_env_ids.numel() > 0:
            last_held[uninitialized_env_ids] = clean_value[uninitialized_env_ids]
            initialized[uninitialized_env_ids] = True

        corrupted = clean_value.clone()
        if float(noise_std) > 0.0:
            corrupted += torch.randn_like(corrupted) * float(noise_std)

        prob = self._probability(outlier_prob)
        if prob > 0.0 and float(outlier_std) > 0.0:
            outlier_mask = torch.rand(self.num_envs, device=self.device) < prob
            corrupted += outlier_mask.to(dtype=corrupted.dtype) * torch.randn_like(corrupted) * float(outlier_std)

        quantization = float(quantization)
        if quantization > 0.0:
            corrupted = torch.round(corrupted / quantization) * quantization

        prob = self._probability(hold_prob)
        if prob > 0.0:
            hold_mask = (torch.rand(self.num_envs, device=self.device) < prob) & was_initialized
            corrupted = torch.where(hold_mask, last_held, corrupted)
        last_held.copy_(corrupted)

        delay_step = int(delay_step)
        if delay_step > 0:
            queue = self._ensure_observation_delay_queue(signal_name, delay_step, clean_value)
            if uninitialized_env_ids.numel() > 0:
                queue.reset(uninitialized_env_ids, clean_value.unsqueeze(-1))
            return queue.step(corrupted.unsqueeze(-1))[:, 0]
        return corrupted

    def _filter_observation_signal(
        self,
        signal_name: str,
        measurement: torch.Tensor,
        *,
        cutoff_hz: float,
        step_dt: float,
    ) -> torch.Tensor:
        measurement = measurement.to(device=self.device, dtype=torch.float32)
        cutoff_hz = float(cutoff_hz)
        if cutoff_hz <= 0.0:
            return measurement

        initialized, filtered = self._observation_filter_state(signal_name)
        uninitialized_env_ids = (~initialized).nonzero(as_tuple=False).squeeze(-1)
        if uninitialized_env_ids.numel() > 0:
            filtered[uninitialized_env_ids] = measurement[uninitialized_env_ids]
            initialized[uninitialized_env_ids] = True

        alpha = 1.0 - math.exp(-2.0 * math.pi * cutoff_hz * float(step_dt))
        alpha = min(max(alpha, 0.0), 1.0)
        filtered += alpha * (measurement - filtered)
        return filtered.clone()

    def _observation_signal_state(self, signal_name: str) -> tuple[torch.Tensor, torch.Tensor]:
        if signal_name == "pb":
            return self.obs_pb_initialized, self.obs_pb_last_held
        if signal_name == "theta":
            return self.obs_theta_initialized, self.obs_theta_last_held
        raise ValueError(f"Unsupported observation degradation signal '{signal_name}'.")

    def _observation_filter_state(self, signal_name: str) -> tuple[torch.Tensor, torch.Tensor]:
        if signal_name == "pb":
            return self.obs_pb_filter_initialized, self.obs_pb_filtered
        if signal_name == "theta":
            return self.obs_theta_filter_initialized, self.obs_theta_filtered
        raise ValueError(f"Unsupported observation filter signal '{signal_name}'.")

    def _ensure_observation_delay_queue(
        self,
        signal_name: str,
        delay_step: int,
        clean_value: torch.Tensor,
    ) -> CommandDelayQueue:
        attr_name = f"obs_{signal_name}_delay_queue"
        queue = getattr(self, attr_name)
        if queue is None or queue.delay_step != delay_step:
            queue = CommandDelayQueue(
                delay_step=delay_step,
                num_envs=self.num_envs,
                command_dim=1,
                device=self.device,
                dtype=clean_value.dtype,
            )
            env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
            queue.reset(env_ids, clean_value.unsqueeze(-1))
            setattr(self, attr_name, queue)
        return queue

    def _sync_observation_diagnostics(
        self,
        clean_obs: torch.Tensor,
        corrupted_obs: torch.Tensor,
        filtered_obs: torch.Tensor,
    ):
        self.observation_degradation_enabled[:] = float(self._observation_degradation_active())
        self.observation_filter_enabled[:] = float(self._observation_filter_active())
        self.obs_pb_clean[:] = clean_obs[:, OBS_PB_INDEX].to(device=self.device, dtype=torch.float32)
        self.obs_pb_corrupted[:] = corrupted_obs[:, OBS_PB_INDEX].to(device=self.device, dtype=torch.float32)
        self.obs_pb_noise[:] = self.obs_pb_corrupted - self.obs_pb_clean
        self.obs_pb_filtered[:] = filtered_obs[:, OBS_PB_INDEX].to(device=self.device, dtype=torch.float32)
        self.obs_pb_filter_noise[:] = self.obs_pb_filtered - self.obs_pb_corrupted
        self.obs_theta_clean[:] = clean_obs[:, OBS_THETA_INDEX].to(device=self.device, dtype=torch.float32)
        self.obs_theta_corrupted[:] = corrupted_obs[:, OBS_THETA_INDEX].to(device=self.device, dtype=torch.float32)
        self.obs_theta_noise[:] = self.obs_theta_corrupted - self.obs_theta_clean
        self.obs_theta_filtered[:] = filtered_obs[:, OBS_THETA_INDEX].to(device=self.device, dtype=torch.float32)
        self.obs_theta_filter_noise[:] = self.obs_theta_filtered - self.obs_theta_corrupted

    @staticmethod
    def _probability(value: float) -> float:
        return min(max(float(value), 0.0), 1.0)
