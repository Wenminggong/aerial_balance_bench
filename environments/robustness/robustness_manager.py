"""Robustness manager with benchmark randomization hooks."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from omni.isaac.lab.utils import configclass

from .velocity_response import VelocityResponseModel


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
    velocity_response_enabled: bool = False
    velocity_response_tau_s_range: tuple[float, float] = (0.0, 0.0)
    velocity_response_gain_range: tuple[float, float] = (1.0, 1.0)
    velocity_response_bias_range: tuple[float, float] = (0.0, 0.0)
    velocity_response_noise_mode: str = "none"
    velocity_response_noise_std_range: tuple[float, float] = (0.0, 0.0)
    velocity_response_ou_theta_range: tuple[float, float] = (0.0, 0.0)
    velocity_response_ou_mu_range: tuple[float, float] = (0.0, 0.0)
    velocity_response_noise_clip: float = 0.0
    velocity_response_max_abs_velocity: float = 0.0
    velocity_response_sim_tau_s: float = 0.100
    velocity_response_sim_gain: float = 1.013
    velocity_response_sim_bias: float = -0.00056
    velocity_response_sim_max_abs_velocity: float = 0.0
    external_disturbance_enabled: bool = False
    external_disturbance_ou_mu: float = 0.0
    external_disturbance_ou_theta_range: tuple[float, float] = (0.1, 0.3)
    external_disturbance_ou_sigma_range: tuple[float, float] = (0.01, 0.05)
    external_disturbance_ou_clip: float = 0.05


class RobustnessManager:
    """Hook object for reset-time randomization and step-time perturbations.

    This manager currently implements reset-time ball-mass variation,
    low-level controller gain variation, command-level action delay,
    plant-matched velocity response, and OU-process fixed-end external
    disturbances.
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
        self.action_delay_enabled = torch.full((num_envs,), float(self._action_delay_active()), device=self.device)
        self.delay_step = torch.full((num_envs,), float(max(int(cfg.delay_step), 0)), device=self.device)
        self.delayed_command_z = torch.full((num_envs,), float("nan"), device=self.device)
        self.velocity_response_noise_mode = VelocityResponseModel.normalize_noise_mode(
            cfg.velocity_response_noise_mode
        )
        self._validate_velocity_response_config()
        self.velocity_response_model = VelocityResponseModel(num_envs, self.device)
        self.velocity_response_enabled = torch.full(
            (num_envs,),
            float(self._velocity_response_active()),
            device=self.device,
        )
        self.velocity_response_noise_mode_id = torch.full(
            (num_envs,),
            float(VelocityResponseModel.NOISE_MODE_IDS[self.velocity_response_noise_mode]),
            device=self.device,
        )
        self.velocity_response_tau_s = self.velocity_response_model.tau_s
        self.velocity_response_gain = self.velocity_response_model.gain
        self.velocity_response_bias = self.velocity_response_model.bias
        self.velocity_response_noise_std = self.velocity_response_model.noise_std
        self.velocity_response_ou_theta = self.velocity_response_model.ou_theta
        self.velocity_response_ou_mu = self.velocity_response_model.ou_mu
        self.velocity_response_input_z = self.velocity_response_model.input_z
        self.velocity_response_target_z = self.velocity_response_model.target_z
        self.velocity_response_nominal_z = self.velocity_response_model.nominal_z
        self.velocity_response_noise_z = self.velocity_response_model.noise_z
        self.velocity_response_executed_z = self.velocity_response_model.executed_z
        self.velocity_response_compensated_command_z = (
            self.velocity_response_model.compensated_command_z
        )
        self.velocity_response_error_z = self.velocity_response_model.error_z
        self.velocity_response_sim_tau_s = self.velocity_response_model.sim_tau_s
        self.velocity_response_sim_gain = self.velocity_response_model.sim_gain
        self.velocity_response_sim_bias = self.velocity_response_model.sim_bias
        self.external_disturbance_enabled = torch.full(
            (num_envs,),
            float(self._external_disturbance_active()),
            device=self.device,
        )
        self.external_disturbance_vel_z = torch.zeros((num_envs, 1), device=self.device)
        self.external_disturbance_ou_theta = torch.zeros((num_envs, 1), device=self.device)
        self.external_disturbance_ou_sigma = torch.zeros((num_envs, 1), device=self.device)

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
        self._reset_velocity_response(env, env_ids)
        self._reset_external_disturbance(env, env_ids)
        return None

    def before_action(self, env, action: torch.Tensor) -> torch.Tensor:
        """Modify high-level actions before interface processing. No-op."""
        return action

    def after_command_update(self, env, command: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Apply command-level action delay followed by optional velocity response."""
        command_state = command
        if self._action_delay_active():
            current_command = env.control_interface.get_delay_command()
            self._ensure_delay_queue(current_command)
            delayed_command = self.delay_queue.step(current_command)
            env.control_interface.set_executed_delay_command(delayed_command)
            command_state = env.control_interface.get_command_state()

        self._sync_delayed_command_z(command_state)
        if not self._velocity_response_active():
            self._sync_velocity_response_passthrough(command_state)
            return command_state
        return self._apply_velocity_response(env, command_state)

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
            "velocity_response_enabled": self.velocity_response_enabled,
            "velocity_response_tau_s": self.velocity_response_tau_s,
            "velocity_response_gain": self.velocity_response_gain,
            "velocity_response_bias": self.velocity_response_bias,
            "velocity_response_noise_mode_id": self.velocity_response_noise_mode_id,
            "velocity_response_noise_std": self.velocity_response_noise_std,
            "velocity_response_ou_theta": self.velocity_response_ou_theta,
            "velocity_response_ou_mu": self.velocity_response_ou_mu,
            "velocity_response_input_z": self.velocity_response_input_z,
            "velocity_response_target_z": self.velocity_response_target_z,
            "velocity_response_nominal_z": self.velocity_response_nominal_z,
            "velocity_response_noise_z": self.velocity_response_noise_z,
            "velocity_response_executed_z": self.velocity_response_executed_z,
            "velocity_response_compensated_command_z": (
                self.velocity_response_compensated_command_z
            ),
            "velocity_response_error_z": self.velocity_response_error_z,
            "velocity_response_sim_tau_s": self.velocity_response_sim_tau_s,
            "velocity_response_sim_gain": self.velocity_response_sim_gain,
            "velocity_response_sim_bias": self.velocity_response_sim_bias,
            "external_disturbance_enabled": self.external_disturbance_enabled,
            "external_disturbance_vel_z": self.external_disturbance_vel_z[:, 0],
            "external_disturbance_ou_theta": self.external_disturbance_ou_theta[:, 0],
            "external_disturbance_ou_sigma": self.external_disturbance_ou_sigma[:, 0],
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
        return bool(self.cfg.enabled and self.cfg.action_delay_enabled and int(self.cfg.delay_step) > 0)

    def _reset_action_delay(self, env, env_ids: torch.Tensor):
        self.action_delay_enabled[env_ids] = float(self._action_delay_active())
        self.delay_step[env_ids] = float(max(int(self.cfg.delay_step), 0))

        command = env.control_interface.get_command_state()
        if not self._action_delay_active():
            self.delayed_command_z[env_ids] = command["executed_command_z"][env_ids].to(
                device=self.device,
                dtype=torch.float32,
            )
            return

        current_command = env.control_interface.get_delay_command()
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

    def _sync_delayed_command_z(self, command: dict[str, torch.Tensor]):
        self.action_delay_enabled[:] = float(self._action_delay_active())
        self.delay_step[:] = float(max(int(self.cfg.delay_step), 0))
        self.delayed_command_z[:] = command["executed_command_z"].to(device=self.device, dtype=torch.float32)

    def _velocity_response_active(self) -> bool:
        return bool(self.cfg.enabled and self.cfg.velocity_response_enabled)

    def _reset_velocity_response(self, env, env_ids: torch.Tensor):
        active = self._velocity_response_active()
        if active:
            self._require_velocity_interface(env)

        self.velocity_response_enabled[env_ids] = float(active)
        mode_id = VelocityResponseModel.NOISE_MODE_IDS[self.velocity_response_noise_mode] if active else 0
        self.velocity_response_noise_mode_id[env_ids] = float(mode_id)

        command = env.control_interface.get_command_state()
        initial_input_z = command["executed_command_z"][env_ids].to(
            device=self.device,
            dtype=torch.float32,
        )
        if not hasattr(env, "vrz"):
            raise ValueError("Velocity response reset requires env.vrz actual velocity state.")
        initial_z = env.vrz[env_ids].to(device=self.device, dtype=torch.float32)
        count = int(env_ids.numel())
        if active:
            tau_s = self._sample_velocity_response_parameter(
                count,
                self.cfg.velocity_response_tau_s_range,
                "velocity_response_tau_s_range",
                lower_bound=0.0,
            )
            gain = self._sample_velocity_response_parameter(
                count,
                self.cfg.velocity_response_gain_range,
                "velocity_response_gain_range",
                lower_bound=0.0,
                strict_lower=True,
            )
            bias = self._sample_velocity_response_parameter(
                count,
                self.cfg.velocity_response_bias_range,
                "velocity_response_bias_range",
            )
            noise_std = self._sample_velocity_response_parameter(
                count,
                self.cfg.velocity_response_noise_std_range,
                "velocity_response_noise_std_range",
                lower_bound=0.0,
            )
            ou_theta = self._sample_velocity_response_parameter(
                count,
                self.cfg.velocity_response_ou_theta_range,
                "velocity_response_ou_theta_range",
                lower_bound=0.0,
            )
            ou_mu = self._sample_velocity_response_parameter(
                count,
                self.cfg.velocity_response_ou_mu_range,
                "velocity_response_ou_mu_range",
            )
        else:
            tau_s = torch.zeros(count, device=self.device)
            gain = torch.ones(count, device=self.device)
            bias = torch.zeros(count, device=self.device)
            noise_std = torch.zeros(count, device=self.device)
            ou_theta = torch.zeros(count, device=self.device)
            ou_mu = torch.zeros(count, device=self.device)

        self.velocity_response_model.reset(
            env_ids,
            initial_z,
            tau_s=tau_s,
            gain=gain,
            bias=bias,
            noise_std=noise_std,
            ou_theta=ou_theta,
            ou_mu=ou_mu,
            initial_input_z=initial_input_z,
            sim_tau_s=torch.full(
                (count,),
                float(self.cfg.velocity_response_sim_tau_s),
                device=self.device,
            ),
            sim_gain=torch.full(
                (count,),
                float(self.cfg.velocity_response_sim_gain),
                device=self.device,
            ),
            sim_bias=torch.full(
                (count,),
                float(self.cfg.velocity_response_sim_bias),
                device=self.device,
            ),
        )

    def _apply_velocity_response(
        self,
        env,
        command: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        self._require_velocity_interface(env)
        step_dt = float(getattr(env, "step_dt", 0.0))
        if step_dt <= 0.0:
            raise ValueError("env.step_dt must be positive for velocity response integration.")
        if "executed_velocity" not in command:
            raise ValueError("Velocity response requires an 'executed_velocity' command state.")

        response_command = command["executed_velocity"].to(device=self.device, dtype=torch.float32).clone()
        compensated_z = self.velocity_response_model.step(
            response_command[:, 2],
            step_dt,
            self.velocity_response_noise_mode,
            noise_clip=float(self.cfg.velocity_response_noise_clip),
            max_abs_velocity=float(self.cfg.velocity_response_max_abs_velocity),
        )
        response_command[:, 2] = compensated_z
        self.velocity_response_enabled[:] = 1.0
        self.velocity_response_noise_mode_id[:] = float(
            VelocityResponseModel.NOISE_MODE_IDS[self.velocity_response_noise_mode]
        )
        env.control_interface.set_executed_delay_command(response_command)
        return env.control_interface.get_command_state()

    def _sync_velocity_response_passthrough(self, command: dict[str, torch.Tensor]):
        input_z = command["executed_command_z"].to(device=self.device, dtype=torch.float32)
        self.velocity_response_enabled[:] = 0.0
        self.velocity_response_noise_mode_id[:] = 0.0
        self.velocity_response_model.sync_passthrough(input_z)

    def _require_velocity_interface(self, env):
        if getattr(env.cfg, "interface_name", None) != "velocity":
            raise ValueError("robustness.velocity_response_enabled requires interface_name='velocity'.")

    def _validate_velocity_response_config(self):
        self._velocity_response_bounds(
            self.cfg.velocity_response_tau_s_range,
            "velocity_response_tau_s_range",
            lower_bound=0.0,
        )
        self._velocity_response_bounds(
            self.cfg.velocity_response_gain_range,
            "velocity_response_gain_range",
            lower_bound=0.0,
            strict_lower=True,
        )
        self._velocity_response_bounds(
            self.cfg.velocity_response_bias_range,
            "velocity_response_bias_range",
        )
        self._velocity_response_bounds(
            self.cfg.velocity_response_noise_std_range,
            "velocity_response_noise_std_range",
            lower_bound=0.0,
        )
        self._velocity_response_bounds(
            self.cfg.velocity_response_ou_theta_range,
            "velocity_response_ou_theta_range",
            lower_bound=0.0,
        )
        self._velocity_response_bounds(
            self.cfg.velocity_response_ou_mu_range,
            "velocity_response_ou_mu_range",
        )
        for field_name in ("velocity_response_noise_clip", "velocity_response_max_abs_velocity"):
            value = float(getattr(self.cfg, field_name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"robustness.{field_name} must be finite and non-negative.")

        sim_tau_s = float(self.cfg.velocity_response_sim_tau_s)
        if not math.isfinite(sim_tau_s) or sim_tau_s <= 0.0:
            raise ValueError("robustness.velocity_response_sim_tau_s must be finite and positive.")
        sim_gain = float(self.cfg.velocity_response_sim_gain)
        if not math.isfinite(sim_gain) or sim_gain <= 0.0:
            raise ValueError("robustness.velocity_response_sim_gain must be finite and positive.")
        sim_bias = float(self.cfg.velocity_response_sim_bias)
        if not math.isfinite(sim_bias):
            raise ValueError("robustness.velocity_response_sim_bias must be finite.")
        sim_max_abs_velocity = float(self.cfg.velocity_response_sim_max_abs_velocity)
        if not math.isfinite(sim_max_abs_velocity) or sim_max_abs_velocity != 0.0:
            raise ValueError(
                "robustness.velocity_response_sim_max_abs_velocity must be 0.0; "
                "plant saturation inversion is not supported."
            )

    def _sample_velocity_response_parameter(
        self,
        count: int,
        bounds: Sequence[float],
        field_name: str,
        *,
        lower_bound: float | None = None,
        strict_lower: bool = False,
    ) -> torch.Tensor:
        min_value, max_value = self._velocity_response_bounds(
            bounds,
            field_name,
            lower_bound=lower_bound,
            strict_lower=strict_lower,
        )
        unit = torch.rand(count, device=self.device)
        return min_value + (max_value - min_value) * unit

    @staticmethod
    def _velocity_response_bounds(
        bounds: Sequence[float],
        field_name: str,
        *,
        lower_bound: float | None = None,
        strict_lower: bool = False,
    ) -> tuple[float, float]:
        if not isinstance(bounds, Sequence) or isinstance(bounds, (str, bytes)) or len(bounds) != 2:
            raise ValueError(f"robustness.{field_name} must contain exactly [min, max].")
        min_value, max_value = (float(value) for value in bounds)
        if not math.isfinite(min_value) or not math.isfinite(max_value):
            raise ValueError(f"robustness.{field_name} values must be finite.")
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
        return min_value, max_value

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
