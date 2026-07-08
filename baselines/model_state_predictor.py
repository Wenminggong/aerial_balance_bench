"""Policy-side model state predictors for action-delay compensation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch

from .base_policy import ObservationIndex


@dataclass
class VelocityModelStatePredictorCfg:
    """Configuration for the velocity-interface model state predictor."""

    type: str = "nominal"
    enabled: bool = False
    delay_step: int | str = 0
    solver: str = "rk4"
    step_dt: float | str = 0.0
    max_acc: float | str = 0.5
    max_delta_acc: float | str = "auto"
    max_velocity: float | str = 0.0
    plank_length: float | str = 1.06
    rope_length: float | str = 0.9
    gravity: float | str = 9.81
    ball_mass: float | str = 0.0005
    ball_radius: float | str = 0.023
    ball_inertia_ratio: float = 0.4
    ball_position_offset: float | str = 0.33
    epsilon: float = 1e-6
    lambda_j: float | str = 1.0
    ukf_alpha: float | str = 1e-3
    ukf_beta: float | str = 2.0
    ukf_kappa: float | str = 0.0
    process_covariance_diag: tuple[float, float, float, float, float, float] = (
        1e-5,
        1e-3,
        1e-5,
        1e-4,
        1e-3,
        1e-2,
    )
    measurement_covariance_diag: tuple[float, float, float, float, float] = (
        1e-6,
        1e-4,
        1e-6,
        1e-3,
        1e-2,
    )
    initial_covariance_diag: tuple[float, float, float, float, float, float] = (
        1e-5,
        1e-3,
        1e-5,
        1e-2,
        1e-1,
        1e-1,
    )
    human_velocity_key: str = "external_disturbance_vel_z"
    human_velocity_sign: float | str = 1.0
    missing_human_velocity_policy: str = "zero"
    acceleration_response_model: str = "disabled"
    acceleration_response_source: str = "extras_or_config"
    acceleration_response_tau_s: float | str = 0.0
    acceleration_response_gain: float | str = 1.0
    acceleration_response_bias: float | str = 0.0
    acceleration_response_noise_mode: str = "none"
    acceleration_response_noise_std: float | str = 0.0
    acceleration_response_ou_theta: float | str = 0.0
    acceleration_response_noise_clip: float | str = 0.0
    acceleration_response_max_abs_acc: float | str = 0.0

    @classmethod
    def from_dict(cls, data: Mapping | None) -> "VelocityModelStatePredictorCfg":
        """Build a predictor config from a YAML dictionary."""
        cfg = cls()
        if not data:
            return cfg
        aliases = {
            "alpha": "ukf_alpha",
            "beta": "ukf_beta",
            "kappa": "ukf_kappa",
        }
        for key, value in data.items():
            if key == "ukf_random_jerk":
                continue
            if key in {"predictor_type", "state_predictor_type"}:
                cfg.type = str(value)
                continue
            normalized_key = aliases.get(key, key)
            if hasattr(cfg, normalized_key):
                setattr(cfg, normalized_key, value)
        ukf_data = data.get("ukf_random_jerk", {})
        if isinstance(ukf_data, Mapping):
            for key, value in ukf_data.items():
                normalized_key = aliases.get(key, key)
                if hasattr(cfg, normalized_key):
                    setattr(cfg, normalized_key, value)
        return cfg


class VelocityModelStatePredictor:
    """Predict the future 11-D benchmark observation under velocity commands."""

    def __init__(self, cfg: VelocityModelStatePredictorCfg, num_envs: int, device: str | torch.device):
        self.cfg = cfg
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.enabled = bool(cfg.enabled)

        self.delay_step = self._as_int(cfg.delay_step, "delay_step")
        self.step_dt = self._as_float(cfg.step_dt, "step_dt")
        self.max_acc = self._as_float(cfg.max_acc, "max_acc")
        self.max_velocity = self._as_float(cfg.max_velocity, "max_velocity")
        self.plank_length = self._as_float(cfg.plank_length, "plank_length")
        self.rope_length = self._as_float(cfg.rope_length, "rope_length")
        self.gravity = abs(self._as_float(cfg.gravity, "gravity"))
        self.ball_mass = self._as_float(cfg.ball_mass, "ball_mass")
        self.ball_radius = self._as_float(cfg.ball_radius, "ball_radius")
        self.ball_inertia_ratio = float(cfg.ball_inertia_ratio)
        self.ball_position_offset = self._as_float(cfg.ball_position_offset, "ball_position_offset")
        self.epsilon = float(cfg.epsilon)

        if self.step_dt <= 0.0:
            raise ValueError("VelocityModelStatePredictor requires step_dt > 0.")
        if self.max_acc <= 0.0:
            raise ValueError("VelocityModelStatePredictor requires max_acc > 0.")
        if self.plank_length <= 0.0 or self.rope_length <= 0.0:
            raise ValueError("VelocityModelStatePredictor requires positive plank_length and rope_length.")
        if self.ball_mass <= 0.0 or self.ball_radius <= 0.0:
            raise ValueError("VelocityModelStatePredictor requires positive ball_mass and ball_radius.")
        if self.ball_inertia_ratio < 0.0:
            raise ValueError("VelocityModelStatePredictor requires non-negative ball_inertia_ratio.")
        if self.solver not in {"euler", "rk4"}:
            raise ValueError("VelocityModelStatePredictor solver must be 'euler' or 'rk4'.")

        self.command_z = torch.zeros((self.num_envs, 1), device=self.device)
        self.last_action = torch.zeros((self.num_envs, 1), device=self.device)
        self.predicted_observation = torch.zeros((self.num_envs, 11), device=self.device)
        self.predicted_error = torch.zeros((self.num_envs, 1), device=self.device)
        self.predicted_error_dot = torch.zeros((self.num_envs, 1), device=self.device)
        self.predicted_error_ddot = torch.zeros((self.num_envs, 1), device=self.device)
        self.read_index = 0
        if self.delay_step > 0:
            self.command_queue = torch.zeros((self.delay_step, self.num_envs, 1), device=self.device)
        else:
            self.command_queue = torch.zeros((0, self.num_envs, 1), device=self.device)

    @property
    def solver(self) -> str:
        return str(self.cfg.solver).lower()

    @property
    def active(self) -> bool:
        return self.enabled and self.delay_step > 0

    def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None):
        """Reset command and queue state for all envs or a selected subset."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        else:
            env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        if env_ids.numel() == 0:
            return

        self.command_z[env_ids] = 0.0
        self.last_action[env_ids] = 0.0
        self.predicted_observation[env_ids] = 0.0
        self.predicted_error[env_ids] = 0.0
        self.predicted_error_dot[env_ids] = 0.0
        self.predicted_error_ddot[env_ids] = 0.0
        if self.delay_step > 0:
            self.command_queue[:, env_ids] = 0.0

    def predict(
        self,
        observation: torch.Tensor,
        error_prev1: torch.Tensor | None = None,
        extras: Mapping | None = None,
    ) -> torch.Tensor:
        """Return a future 11-D observation after the configured delay horizon."""
        del extras
        observation = observation.to(device=self.device, dtype=torch.float32)
        if observation.shape != (self.num_envs, 11):
            raise ValueError(
                "VelocityModelStatePredictor expects observation shape "
                f"({self.num_envs}, 11), got {tuple(observation.shape)}."
            )
        current_error = (
            observation[:, ObservationIndex.PB : ObservationIndex.PB + 1]
            - observation[:, ObservationIndex.PG : ObservationIndex.PG + 1]
        )
        if not self.active:
            self.predicted_observation.copy_(observation)
            self.predicted_error.copy_(current_error)
            self.predicted_error_dot.zero_()
            self.predicted_error_ddot.zero_()
            return observation

        predicted = observation.clone()
        commands = self._pending_commands()
        pb = predicted[:, ObservationIndex.PB].clone()
        vb = predicted[:, ObservationIndex.VB].clone()
        theta = predicted[:, ObservationIndex.THETA].clone()
        drz = predicted[:, ObservationIndex.DRZ].clone()
        omega_prev = predicted[:, ObservationIndex.OMEGA].clone()
        vrz_prev = predicted[:, ObservationIndex.VRZ].clone()

        error_curr = current_error[:, 0].clone()
        error_prev = self._prepare_error_prev1(error_prev1, current_error)[:, 0].clone()
        predicted_error_dot = torch.zeros_like(error_curr)
        predicted_error_ddot = torch.zeros_like(error_curr)
        pg = observation[:, ObservationIndex.PG].clone()

        ab = predicted[:, ObservationIndex.AB].clone()
        omega = omega_prev.clone()
        alpha = predicted[:, ObservationIndex.ALPHA].clone()
        vrz = vrz_prev.clone()
        arz = predicted[:, ObservationIndex.ARZ].clone()

        for step_id in range(commands.shape[0]):
            vrz = commands[step_id, :, 0]
            pb, vb, theta, ab, omega = self._integrate_one_step(pb, vb, theta, vrz)
            alpha = (omega - omega_prev) / self.step_dt
            arz = (vrz - vrz_prev) / self.step_dt
            drz = drz + vrz * self.step_dt
            omega_prev = omega
            vrz_prev = vrz
            new_error = pb - pg
            predicted_error_dot = new_error - error_curr
            predicted_error_ddot = predicted_error_dot - (error_curr - error_prev)
            error_prev = error_curr
            error_curr = new_error

        predicted[:, ObservationIndex.PB] = pb
        predicted[:, ObservationIndex.VB] = vb
        predicted[:, ObservationIndex.AB] = ab
        predicted[:, ObservationIndex.THETA] = theta
        predicted[:, ObservationIndex.OMEGA] = omega
        predicted[:, ObservationIndex.ALPHA] = alpha
        predicted[:, ObservationIndex.DRZ] = drz
        predicted[:, ObservationIndex.VRZ] = vrz
        predicted[:, ObservationIndex.ARZ] = arz
        predicted[:, ObservationIndex.PG] = observation[:, ObservationIndex.PG]
        predicted[:, ObservationIndex.A_PREV] = observation[:, ObservationIndex.A_PREV]

        self.predicted_observation.copy_(predicted)
        self.predicted_error.copy_(error_curr.unsqueeze(-1))
        self.predicted_error_dot.copy_(predicted_error_dot.unsqueeze(-1))
        self.predicted_error_ddot.copy_(predicted_error_ddot.unsqueeze(-1))
        return predicted

    def get_error_prediction(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return predicted error and its horizon-local finite differences."""
        return self.predicted_error, self.predicted_error_dot, self.predicted_error_ddot

    def update_after_action(self, action: torch.Tensor):
        """Mirror VelocityInterface command accumulation and enqueue the command."""
        action = action.to(device=self.device, dtype=torch.float32)
        clipped_action = torch.clamp(action, -self.max_acc * self.step_dt, self.max_acc * self.step_dt)
        self.last_action.copy_(clipped_action)
        self.command_z += clipped_action
        if self.max_velocity > 0.0:
            self.command_z.clamp_(min=-self.max_velocity, max=self.max_velocity)

        if self.delay_step <= 0:
            return
        self.command_queue[self.read_index] = self.command_z
        self.read_index = (self.read_index + 1) % self.delay_step

    def get_state(self) -> dict[str, torch.Tensor]:
        """Return predictor state for rollout diagnostics."""
        state = {
            "policy_predictor_enabled": torch.full(
                (self.num_envs,),
                float(self.active),
                device=self.device,
            ),
            "policy_predictor_command_z": self.command_z[:, 0],
            "policy_predictor_error": self.predicted_error[:, 0],
            "policy_predictor_error_dot": self.predicted_error_dot[:, 0],
            "policy_predictor_error_ddot": self.predicted_error_ddot[:, 0],
        }
        names = (
            "pb",
            "vb",
            "ab",
            "theta",
            "omega",
            "alpha",
            "drz",
            "vrz",
            "arz",
            "pg",
            "a_prev",
        )
        for index, name in enumerate(names):
            state[f"policy_predicted_{name}"] = self.predicted_observation[:, index]
        return state

    def to(self, device: str | torch.device):
        """Move predictor buffers to a device and return self."""
        device = torch.device(device)
        for name, value in vars(self).items():
            if isinstance(value, torch.Tensor):
                setattr(self, name, value.to(device=device))
        self.device = device
        return self

    def _pending_commands(self) -> torch.Tensor:
        if self.delay_step <= 0:
            return self.command_queue
        ordered_ids = (torch.arange(self.delay_step, device=self.device) + self.read_index) % self.delay_step
        return self.command_queue[ordered_ids].clone()

    def _prepare_error_prev1(self, error_prev1: torch.Tensor | None, current_error: torch.Tensor) -> torch.Tensor:
        if error_prev1 is None:
            return current_error.clone()
        error_prev1 = error_prev1.to(device=self.device, dtype=torch.float32)
        if error_prev1.shape == (self.num_envs,):
            error_prev1 = error_prev1.unsqueeze(-1)
        if error_prev1.shape != (self.num_envs, 1):
            raise ValueError(
                "VelocityModelStatePredictor expects error_prev1 shape "
                f"({self.num_envs}, 1), got {tuple(error_prev1.shape)}."
            )
        return error_prev1

    def _integrate_one_step(
        self,
        pb: torch.Tensor,
        vb: torch.Tensor,
        theta: torch.Tensor,
        vrz: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.solver == "euler":
            derivatives = self._derivatives(pb, vb, theta, vrz)
            pb_next = pb + derivatives[:, 0] * self.step_dt
            vb_next = vb + derivatives[:, 1] * self.step_dt
            theta_next = theta + derivatives[:, 2] * self.step_dt
        else:
            state = torch.stack((pb, vb, theta), dim=-1)
            k1 = self._derivatives_from_state(state, vrz)
            k2 = self._derivatives_from_state(state + 0.5 * self.step_dt * k1, vrz)
            k3 = self._derivatives_from_state(state + 0.5 * self.step_dt * k2, vrz)
            k4 = self._derivatives_from_state(state + self.step_dt * k3, vrz)
            next_state = state + (self.step_dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            pb_next = next_state[:, 0]
            vb_next = next_state[:, 1]
            theta_next = next_state[:, 2]

        final_derivatives = self._derivatives(pb_next, vb_next, theta_next, vrz)
        ab_next = final_derivatives[:, 1]
        omega_next = final_derivatives[:, 2]
        return pb_next, vb_next, theta_next, ab_next, omega_next

    def _derivatives_from_state(self, state: torch.Tensor, vrz: torch.Tensor) -> torch.Tensor:
        return self._derivatives(state[:, 0], state[:, 1], state[:, 2], vrz)

    def _derivatives(
        self,
        pb: torch.Tensor,
        vb: torch.Tensor,
        theta: torch.Tensor,
        vrz: torch.Tensor,
    ) -> torch.Tensor:
        beta = self._beta(theta)
        denominator = self.plank_length * self._safe_denominator(torch.cos(beta - theta))
        omega = -vrz * torch.cos(beta) / denominator
        ball_inertia = self.ball_inertia_ratio * self.ball_mass * self.ball_radius**2
        effective_mass = ball_inertia / self.ball_radius**2 + self.ball_mass
        ab = self.ball_mass * (pb + self.ball_position_offset - self.plank_length) * omega.square()
        ab -= self.ball_mass * self.gravity * torch.sin(theta)
        ab = ab / effective_mass
        return torch.stack((vb, ab, omega), dim=-1)

    def _beta(self, theta: torch.Tensor) -> torch.Tensor:
        sin_beta = (self.plank_length / self.rope_length) * (1.0 - torch.cos(theta))
        sin_beta = torch.clamp(sin_beta, min=-1.0 + self.epsilon, max=1.0 - self.epsilon)
        return torch.asin(sin_beta)

    def _safe_denominator(self, value: torch.Tensor) -> torch.Tensor:
        sign = torch.where(value >= 0.0, torch.ones_like(value), -torch.ones_like(value))
        return torch.where(torch.abs(value) < self.epsilon, sign * self.epsilon, value)

    @staticmethod
    def _as_float(value, field_name: str) -> float:
        if isinstance(value, str):
            if value.lower() == "auto":
                raise ValueError(f"VelocityModelStatePredictorCfg.{field_name} must be resolved before use.")
            return float(value)
        return float(value)

    @staticmethod
    def _as_int(value, field_name: str) -> int:
        if isinstance(value, str):
            if value.lower() == "auto":
                raise ValueError(f"VelocityModelStatePredictorCfg.{field_name} must be resolved before use.")
            return int(value)
        return int(value)


class AccelerationModelStatePredictor(VelocityModelStatePredictor):
    """Predict the future 11-D benchmark observation under acceleration commands."""

    def __init__(self, cfg: VelocityModelStatePredictorCfg, num_envs: int, device: str | torch.device):
        super().__init__(cfg, num_envs, device)
        self.max_delta_acc = self._as_float(cfg.max_delta_acc, "max_delta_acc")
        self.acceleration_response_model = str(cfg.acceleration_response_model).lower().replace("-", "_")
        self.acceleration_response_source = str(cfg.acceleration_response_source).lower().replace("-", "_")
        self.acceleration_response_tau_s = self._as_float(
            cfg.acceleration_response_tau_s,
            "acceleration_response_tau_s",
        )
        self.acceleration_response_gain = self._as_float(
            cfg.acceleration_response_gain,
            "acceleration_response_gain",
        )
        self.acceleration_response_bias = self._as_float(
            cfg.acceleration_response_bias,
            "acceleration_response_bias",
        )
        self.acceleration_response_noise_mode = str(cfg.acceleration_response_noise_mode).lower()
        self.acceleration_response_noise_std = self._as_float(
            cfg.acceleration_response_noise_std,
            "acceleration_response_noise_std",
        )
        self.acceleration_response_ou_theta = self._as_float(
            cfg.acceleration_response_ou_theta,
            "acceleration_response_ou_theta",
        )
        self.acceleration_response_noise_clip = self._as_float(
            cfg.acceleration_response_noise_clip,
            "acceleration_response_noise_clip",
        )
        self.acceleration_response_max_abs_acc = self._as_float(
            cfg.acceleration_response_max_abs_acc,
            "acceleration_response_max_abs_acc",
        )
        if self.max_delta_acc <= 0.0:
            raise ValueError("AccelerationModelStatePredictor requires max_delta_acc > 0.")
        if self.acceleration_response_model in {"none", "off"}:
            self.acceleration_response_model = "disabled"
        if self.acceleration_response_model not in {"disabled", "deterministic"}:
            raise ValueError(
                "AccelerationModelStatePredictor acceleration_response_model must be "
                "'disabled' or 'deterministic'."
            )
        if self.acceleration_response_source not in {"extras_or_config", "config"}:
            raise ValueError(
                "AccelerationModelStatePredictor acceleration_response_source must be "
                "'extras_or_config' or 'config'."
            )
        if self.acceleration_response_noise_mode not in {"none", "white", "ou"}:
            raise ValueError(
                "AccelerationModelStatePredictor acceleration_response_noise_mode must be "
                "'none', 'white', or 'ou'."
            )
        for field_name, value in (
            ("acceleration_response_tau_s", self.acceleration_response_tau_s),
            ("acceleration_response_noise_std", self.acceleration_response_noise_std),
            ("acceleration_response_ou_theta", self.acceleration_response_ou_theta),
            ("acceleration_response_noise_clip", self.acceleration_response_noise_clip),
            ("acceleration_response_max_abs_acc", self.acceleration_response_max_abs_acc),
        ):
            if value < 0.0:
                raise ValueError(f"AccelerationModelStatePredictor {field_name} must be non-negative.")

        self.predicted_response_active = torch.zeros((self.num_envs,), device=self.device)
        self.predicted_response_reference_z = torch.zeros((self.num_envs,), device=self.device)
        self.predicted_response_target_z = torch.zeros((self.num_envs,), device=self.device)
        self.predicted_response_executed_z = torch.zeros((self.num_envs,), device=self.device)
        self.predicted_response_error_z = torch.zeros((self.num_envs,), device=self.device)
        self.predicted_response_noise_z = torch.zeros((self.num_envs,), device=self.device)

    def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None):
        """Reset command, queue, and response diagnostics for all envs or a selected subset."""
        super().reset(env_ids)
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        else:
            env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        if env_ids.numel() == 0:
            return
        self.predicted_response_active[env_ids] = 0.0
        self.predicted_response_reference_z[env_ids] = 0.0
        self.predicted_response_target_z[env_ids] = 0.0
        self.predicted_response_executed_z[env_ids] = 0.0
        self.predicted_response_error_z[env_ids] = 0.0
        self.predicted_response_noise_z[env_ids] = 0.0

    def predict(
        self,
        observation: torch.Tensor,
        error_prev1: torch.Tensor | None = None,
        extras: Mapping | None = None,
    ) -> torch.Tensor:
        """Return a future 11-D observation after the configured delay horizon."""
        observation = observation.to(device=self.device, dtype=torch.float32)
        if observation.shape != (self.num_envs, 11):
            raise ValueError(
                "AccelerationModelStatePredictor expects observation shape "
                f"({self.num_envs}, 11), got {tuple(observation.shape)}."
            )
        current_error = (
            observation[:, ObservationIndex.PB : ObservationIndex.PB + 1]
            - observation[:, ObservationIndex.PG : ObservationIndex.PG + 1]
        )
        if not self.active:
            self.predicted_observation.copy_(observation)
            self.predicted_error.copy_(current_error)
            self.predicted_error_dot.zero_()
            self.predicted_error_ddot.zero_()
            self._set_inactive_response_diagnostics(observation)
            return observation

        predicted = observation.clone()
        commands = self._pending_commands()
        pb = predicted[:, ObservationIndex.PB].clone()
        vb = predicted[:, ObservationIndex.VB].clone()
        theta = predicted[:, ObservationIndex.THETA].clone()
        drz = predicted[:, ObservationIndex.DRZ].clone()
        omega_prev = predicted[:, ObservationIndex.OMEGA].clone()
        vrz = predicted[:, ObservationIndex.VRZ].clone()

        error_curr = current_error[:, 0].clone()
        error_prev = self._prepare_error_prev1(error_prev1, current_error)[:, 0].clone()
        predicted_error_dot = torch.zeros_like(error_curr)
        predicted_error_ddot = torch.zeros_like(error_curr)
        pg = observation[:, ObservationIndex.PG].clone()

        ab = predicted[:, ObservationIndex.AB].clone()
        omega = omega_prev.clone()
        alpha = predicted[:, ObservationIndex.ALPHA].clone()
        arz = predicted[:, ObservationIndex.ARZ].clone()
        response_state = self._acceleration_response_state(observation, extras)

        for step_id in range(commands.shape[0]):
            reference_arz = commands[step_id, :, 0]
            arz = self._response_executed_acceleration(reference_arz, response_state)
            pb, vb, theta, drz, vrz, ab, omega = self._integrate_acceleration_one_step(
                pb,
                vb,
                theta,
                drz,
                vrz,
                arz,
            )
            alpha = (omega - omega_prev) / self.step_dt
            omega_prev = omega
            new_error = pb - pg
            predicted_error_dot = new_error - error_curr
            predicted_error_ddot = predicted_error_dot - (error_curr - error_prev)
            error_prev = error_curr
            error_curr = new_error

        predicted[:, ObservationIndex.PB] = pb
        predicted[:, ObservationIndex.VB] = vb
        predicted[:, ObservationIndex.AB] = ab
        predicted[:, ObservationIndex.THETA] = theta
        predicted[:, ObservationIndex.OMEGA] = omega
        predicted[:, ObservationIndex.ALPHA] = alpha
        predicted[:, ObservationIndex.DRZ] = drz
        predicted[:, ObservationIndex.VRZ] = vrz
        predicted[:, ObservationIndex.ARZ] = arz
        predicted[:, ObservationIndex.PG] = observation[:, ObservationIndex.PG]
        predicted[:, ObservationIndex.A_PREV] = observation[:, ObservationIndex.A_PREV]

        self.predicted_observation.copy_(predicted)
        self.predicted_error.copy_(error_curr.unsqueeze(-1))
        self.predicted_error_dot.copy_(predicted_error_dot.unsqueeze(-1))
        self.predicted_error_ddot.copy_(predicted_error_ddot.unsqueeze(-1))
        return predicted

    def get_state(self) -> dict[str, torch.Tensor]:
        """Return predictor state and acceleration-response diagnostics."""
        state = super().get_state()
        state.update(
            {
                "policy_predictor_response_active": self.predicted_response_active,
                "policy_predictor_response_reference_z": self.predicted_response_reference_z,
                "policy_predictor_response_target_z": self.predicted_response_target_z,
                "policy_predictor_response_executed_z": self.predicted_response_executed_z,
                "policy_predictor_response_error_z": self.predicted_response_error_z,
                "policy_predictor_response_noise_z": self.predicted_response_noise_z,
            }
        )
        return state

    def update_after_action(self, action: torch.Tensor):
        """Mirror AccelerationInterface command accumulation and enqueue the command."""
        action = action.to(device=self.device, dtype=torch.float32)
        if action.ndim == 1:
            action = action.unsqueeze(-1)
        if action.shape != (self.num_envs, 1):
            raise ValueError(
                "AccelerationModelStatePredictor expects action shape "
                f"({self.num_envs}, 1), got {tuple(action.shape)}."
            )

        clipped_action = torch.clamp(action, -self.max_delta_acc, self.max_delta_acc)
        next_command = self.command_z + clipped_action
        next_command = torch.clamp(next_command, -self.max_acc, self.max_acc)
        self.last_action.copy_(next_command - self.command_z)
        self.command_z.copy_(next_command)

        if self.delay_step <= 0:
            return
        self.command_queue[self.read_index] = self.command_z
        self.read_index = (self.read_index + 1) % self.delay_step

    def _integrate_acceleration_one_step(
        self,
        pb: torch.Tensor,
        vb: torch.Tensor,
        theta: torch.Tensor,
        drz: torch.Tensor,
        vrz: torch.Tensor,
        arz: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.solver == "euler":
            derivatives = self._acceleration_derivatives(pb, vb, theta, drz, vrz, arz)
            next_state = torch.stack((pb, vb, theta, drz, vrz), dim=-1) + derivatives * self.step_dt
        else:
            state = torch.stack((pb, vb, theta, drz, vrz), dim=-1)
            k1 = self._acceleration_derivatives_from_state(state, arz)
            k2 = self._acceleration_derivatives_from_state(state + 0.5 * self.step_dt * k1, arz)
            k3 = self._acceleration_derivatives_from_state(state + 0.5 * self.step_dt * k2, arz)
            k4 = self._acceleration_derivatives_from_state(state + self.step_dt * k3, arz)
            next_state = state + (self.step_dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        pb_next = next_state[:, 0]
        vb_next = next_state[:, 1]
        theta_next = next_state[:, 2]
        drz_next = next_state[:, 3]
        vrz_next = next_state[:, 4]
        final_derivatives = self._acceleration_derivatives(pb_next, vb_next, theta_next, drz_next, vrz_next, arz)
        ab_next = final_derivatives[:, 1]
        omega_next = final_derivatives[:, 2]
        return pb_next, vb_next, theta_next, drz_next, vrz_next, ab_next, omega_next

    def _acceleration_derivatives_from_state(self, state: torch.Tensor, arz: torch.Tensor) -> torch.Tensor:
        return self._acceleration_derivatives(state[:, 0], state[:, 1], state[:, 2], state[:, 3], state[:, 4], arz)

    def _acceleration_derivatives(
        self,
        pb: torch.Tensor,
        vb: torch.Tensor,
        theta: torch.Tensor,
        drz: torch.Tensor,
        vrz: torch.Tensor,
        arz: torch.Tensor,
    ) -> torch.Tensor:
        del drz
        ball_derivatives = self._derivatives(pb, vb, theta, vrz)
        return torch.stack(
            (
                ball_derivatives[:, 0],
                ball_derivatives[:, 1],
                ball_derivatives[:, 2],
                vrz,
                arz,
            ),
            dim=-1,
        )

    def _acceleration_response_state(self, observation: torch.Tensor, extras: Mapping | None) -> dict[str, torch.Tensor]:
        if self.acceleration_response_model != "deterministic":
            active = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        elif self.acceleration_response_source == "extras_or_config" and self._has_extras_value(
            extras,
            "acceleration_response_enabled",
        ):
            active = self._extras_or_default(
                extras,
                "acceleration_response_enabled",
                1.0,
            ) > 0.5
        else:
            active = torch.ones((self.num_envs,), dtype=torch.bool, device=self.device)

        source_allows_extras = self.acceleration_response_source == "extras_or_config"
        tau_s = self._response_parameter(
            extras,
            "acceleration_response_tau_s",
            self.acceleration_response_tau_s,
            source_allows_extras,
        ).clamp_min(0.0)
        gain = self._response_parameter(
            extras,
            "acceleration_response_gain",
            self.acceleration_response_gain,
            source_allows_extras,
        )
        bias = self._response_parameter(
            extras,
            "acceleration_response_bias",
            self.acceleration_response_bias,
            source_allows_extras,
        )
        noise_std = self._response_parameter(
            extras,
            "acceleration_response_noise_std",
            self.acceleration_response_noise_std,
            source_allows_extras,
        ).clamp_min(0.0)
        ou_theta = self._response_parameter(
            extras,
            "acceleration_response_ou_theta",
            self.acceleration_response_ou_theta,
            source_allows_extras,
        ).clamp_min(0.0)
        noise_z = self._response_state_value(
            extras,
            "acceleration_response_noise_z",
            torch.zeros((self.num_envs,), device=self.device),
        )
        executed_z = self._response_state_value(
            extras,
            "acceleration_response_executed_z",
            observation[:, ObservationIndex.ARZ],
        )
        return {
            "active": active,
            "tau_s": tau_s,
            "gain": gain,
            "bias": bias,
            "noise_std": noise_std,
            "ou_theta": ou_theta,
            "noise_z": noise_z,
            "executed_z": executed_z,
        }

    def _response_executed_acceleration(
        self,
        reference_z: torch.Tensor,
        response_state: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        active = response_state["active"]
        if self.acceleration_response_model != "deterministic":
            self._set_response_diagnostics(
                active,
                reference_z,
                reference_z,
                reference_z,
                torch.zeros_like(reference_z),
            )
            return reference_z

        noise_z = self._predict_response_noise(response_state)
        target_z = response_state["gain"] * reference_z + response_state["bias"] + noise_z

        tau_s = response_state["tau_s"]
        filtered_mask = tau_s > 0.0
        alpha = torch.zeros_like(tau_s)
        alpha[filtered_mask] = 1.0 - torch.exp(-self.step_dt / tau_s[filtered_mask])
        filtered_z = response_state["executed_z"] + alpha * (target_z - response_state["executed_z"])
        executed_z = torch.where(filtered_mask, filtered_z, target_z)

        max_abs_acc = float(self.acceleration_response_max_abs_acc)
        if max_abs_acc > 0.0:
            executed_z = torch.clamp(executed_z, min=-max_abs_acc, max=max_abs_acc)

        target_z = torch.where(active, target_z, reference_z)
        executed_z = torch.where(active, executed_z, reference_z)
        noise_z = torch.where(active, noise_z, torch.zeros_like(noise_z))
        response_state["noise_z"] = noise_z
        response_state["executed_z"] = executed_z
        self._set_response_diagnostics(active, reference_z, target_z, executed_z, noise_z)
        return executed_z

    def _predict_response_noise(self, response_state: dict[str, torch.Tensor]) -> torch.Tensor:
        mode = self.acceleration_response_noise_mode
        if mode == "none":
            return torch.zeros_like(response_state["noise_z"])
        if mode == "white":
            return torch.zeros_like(response_state["noise_z"])

        noise_z = response_state["noise_z"]
        ou_theta = response_state["ou_theta"]
        noise_z = noise_z + ou_theta * (0.0 - noise_z) * self.step_dt
        clip = float(self.acceleration_response_noise_clip)
        if clip > 0.0:
            noise_z = torch.clamp(noise_z, min=-clip, max=clip)
        return noise_z

    def _response_parameter(
        self,
        extras: Mapping | None,
        key: str,
        fallback: float,
        source_allows_extras: bool,
    ) -> torch.Tensor:
        if source_allows_extras and self._has_extras_value(extras, key):
            return self._extras_or_default(extras, key, fallback)
        return torch.full((self.num_envs,), float(fallback), device=self.device, dtype=torch.float32)

    def _response_state_value(
        self,
        extras: Mapping | None,
        key: str,
        fallback: torch.Tensor,
    ) -> torch.Tensor:
        if self._has_extras_value(extras, key):
            return self._extras_or_default(extras, key, fallback)
        fallback = fallback.to(device=self.device, dtype=torch.float32)
        if fallback.shape != (self.num_envs,):
            raise ValueError(
                f"AccelerationModelStatePredictor expected fallback shape ({self.num_envs},), "
                f"got {tuple(fallback.shape)}."
            )
        return fallback.clone()

    def _has_extras_value(self, extras: Mapping | None, key: str) -> bool:
        if not isinstance(extras, Mapping):
            return False
        step_info = extras.get("step", {})
        return (isinstance(step_info, Mapping) and key in step_info) or key in extras

    def _extras_or_default(self, extras: Mapping | None, key: str, default) -> torch.Tensor:
        value = None
        if isinstance(extras, Mapping):
            step_info = extras.get("step", {})
            if isinstance(step_info, Mapping) and key in step_info:
                value = step_info[key]
            elif key in extras:
                value = extras[key]

        if value is None:
            value = default
        tensor = torch.as_tensor(value, device=self.device, dtype=torch.float32)
        if tensor.ndim == 0:
            tensor = tensor.expand(self.num_envs)
        if tensor.ndim == 2 and tensor.shape[1] == 1:
            tensor = tensor[:, 0]
        if tensor.shape != (self.num_envs,):
            raise ValueError(
                f"AccelerationModelStatePredictor expected extras['{key}'] shape ({self.num_envs},), "
                f"got {tuple(tensor.shape)}."
            )
        return tensor.clone()

    def _set_inactive_response_diagnostics(self, observation: torch.Tensor):
        arz = observation[:, ObservationIndex.ARZ].to(device=self.device, dtype=torch.float32)
        inactive = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        self._set_response_diagnostics(inactive, arz, arz, arz, torch.zeros_like(arz))

    def _set_response_diagnostics(
        self,
        active: torch.Tensor,
        reference_z: torch.Tensor,
        target_z: torch.Tensor,
        executed_z: torch.Tensor,
        noise_z: torch.Tensor,
    ):
        self.predicted_response_active.copy_(active.to(device=self.device, dtype=torch.float32))
        self.predicted_response_reference_z.copy_(reference_z.to(device=self.device, dtype=torch.float32))
        self.predicted_response_target_z.copy_(target_z.to(device=self.device, dtype=torch.float32))
        self.predicted_response_executed_z.copy_(executed_z.to(device=self.device, dtype=torch.float32))
        self.predicted_response_error_z.copy_(
            (executed_z - reference_z).to(device=self.device, dtype=torch.float32)
        )
        self.predicted_response_noise_z.copy_(noise_z.to(device=self.device, dtype=torch.float32))


class UKFJerkHumanVelocityStatePredictor(VelocityModelStatePredictor):
    """UKF delay compensator with a random-jerk human endpoint model."""

    state_names = ("p_b", "v_b", "theta", "v_hz", "a_hz", "j_hz")
    measurement_names = ("p_b", "v_b", "theta", "omega", "v_hz")

    IDX_PB = 0
    IDX_VB = 1
    IDX_THETA = 2
    IDX_V_HZ = 3
    IDX_A_HZ = 4
    IDX_J_HZ = 5
    STATE_DIM = 6
    MEASUREMENT_DIM = 5

    def __init__(self, cfg: VelocityModelStatePredictorCfg, num_envs: int, device: str | torch.device):
        super().__init__(cfg, num_envs, device)
        self.ukf_dtype = torch.float64
        self.lambda_j = self._as_float(cfg.lambda_j, "lambda_j")
        self.ukf_alpha = self._as_float(cfg.ukf_alpha, "ukf_alpha")
        self.ukf_beta = self._as_float(cfg.ukf_beta, "ukf_beta")
        self.ukf_kappa = self._as_float(cfg.ukf_kappa, "ukf_kappa")
        self.human_velocity_key = str(cfg.human_velocity_key)
        self.human_velocity_sign = self._as_float(cfg.human_velocity_sign, "human_velocity_sign")
        self.missing_human_velocity_policy = str(cfg.missing_human_velocity_policy).lower()
        self.ukf_jitter = 1e-10

        if self.lambda_j < 0.0:
            raise ValueError("UKFJerkHumanVelocityStatePredictor requires lambda_j >= 0.")
        if self.ukf_alpha <= 0.0:
            raise ValueError("UKFJerkHumanVelocityStatePredictor requires ukf_alpha > 0.")
        if self.missing_human_velocity_policy not in {"zero", "last", "error"}:
            raise ValueError(
                "UKFJerkHumanVelocityStatePredictor missing_human_velocity_policy must be "
                "'zero', 'last', or 'error'."
            )

        self.ukf_lambda = self.ukf_alpha**2 * (self.STATE_DIM + self.ukf_kappa) - self.STATE_DIM
        self.ukf_weight_scale = self.STATE_DIM + self.ukf_lambda
        if self.ukf_weight_scale <= 0.0:
            raise ValueError("UKFJerkHumanVelocityStatePredictor requires STATE_DIM + lambda > 0.")

        sigma_count = 2 * self.STATE_DIM + 1
        self.mean_weights = torch.full(
            (sigma_count,),
            1.0 / (2.0 * self.ukf_weight_scale),
            device=self.device,
            dtype=self.ukf_dtype,
        )
        self.covariance_weights = self.mean_weights.clone()
        self.mean_weights[0] = self.ukf_lambda / self.ukf_weight_scale
        self.covariance_weights[0] = (
            self.ukf_lambda / self.ukf_weight_scale + 1.0 - self.ukf_alpha**2 + self.ukf_beta
        )

        self.process_covariance = self._diag_from_cfg(
            cfg.process_covariance_diag,
            self.STATE_DIM,
            "process_covariance_diag",
        )
        self.measurement_covariance = self._diag_from_cfg(
            cfg.measurement_covariance_diag,
            self.MEASUREMENT_DIM,
            "measurement_covariance_diag",
        )
        self.initial_covariance = self._diag_from_cfg(
            cfg.initial_covariance_diag,
            self.STATE_DIM,
            "initial_covariance_diag",
        )
        self.ukf_identity = torch.eye(self.STATE_DIM, device=self.device, dtype=self.ukf_dtype)
        self.measurement_identity = torch.eye(self.MEASUREMENT_DIM, device=self.device, dtype=self.ukf_dtype)

        self.ukf_state = torch.zeros((self.num_envs, self.STATE_DIM), device=self.device, dtype=self.ukf_dtype)
        self.ukf_covariance = self.initial_covariance.unsqueeze(0).repeat(self.num_envs, 1, 1)
        self.ukf_initialized = torch.zeros((self.num_envs,), device=self.device, dtype=torch.bool)

        self.measured_human_velocity = torch.zeros((self.num_envs,), device=self.device)
        self.last_human_velocity = torch.zeros((self.num_envs,), device=self.device)
        self.human_velocity_valid = torch.zeros((self.num_envs,), device=self.device)
        self.ukf_covariance_trace = torch.zeros((self.num_envs,), device=self.device)
        self.ukf_innovation_norm = torch.zeros((self.num_envs,), device=self.device)
        self.ukf_omega_residual = torch.zeros((self.num_envs,), device=self.device)

    def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None):
        """Reset command queue and UKF state for all envs or a selected subset."""
        super().reset(env_ids)
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        else:
            env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        if env_ids.numel() == 0:
            return
        self.ukf_state[env_ids] = 0.0
        self.ukf_covariance[env_ids] = self.initial_covariance
        self.ukf_initialized[env_ids] = False
        self.measured_human_velocity[env_ids] = 0.0
        self.last_human_velocity[env_ids] = 0.0
        self.human_velocity_valid[env_ids] = 0.0
        self.ukf_covariance_trace[env_ids] = 0.0
        self.ukf_innovation_norm[env_ids] = 0.0
        self.ukf_omega_residual[env_ids] = 0.0

    def predict(
        self,
        observation: torch.Tensor,
        error_prev1: torch.Tensor | None = None,
        extras: Mapping | None = None,
    ) -> torch.Tensor:
        """Return a delay-compensated 11-D observation using UKF human-state estimates."""
        observation = observation.to(device=self.device, dtype=torch.float32)
        if observation.shape != (self.num_envs, 11):
            raise ValueError(
                "UKFJerkHumanVelocityStatePredictor expects observation shape "
                f"({self.num_envs}, 11), got {tuple(observation.shape)}."
            )

        current_error = (
            observation[:, ObservationIndex.PB : ObservationIndex.PB + 1]
            - observation[:, ObservationIndex.PG : ObservationIndex.PG + 1]
        )
        if not self.active:
            self.predicted_observation.copy_(observation)
            self.predicted_error.copy_(current_error)
            self.predicted_error_dot.zero_()
            self.predicted_error_ddot.zero_()
            return observation

        human_velocity, human_valid = self._human_velocity_from_extras(extras)
        measurement = torch.stack(
            (
                observation[:, ObservationIndex.PB],
                observation[:, ObservationIndex.VB],
                observation[:, ObservationIndex.THETA],
                observation[:, ObservationIndex.OMEGA],
                human_velocity,
            ),
            dim=-1,
        ).to(dtype=self.ukf_dtype)
        current_drone_velocity = observation[:, ObservationIndex.VRZ].to(dtype=self.ukf_dtype)

        was_initialized = self.ukf_initialized.clone()
        needs_init = ~was_initialized
        if torch.any(needs_init):
            self._initialize_ukf_state(needs_init, observation, human_velocity)
        if torch.any(was_initialized):
            env_ids = was_initialized.nonzero(as_tuple=False).squeeze(-1)
            self._ukf_step(
                env_ids,
                measurement[env_ids],
                current_drone_velocity[env_ids],
                human_valid[env_ids],
            )
        self._refresh_ukf_diagnostics(observation, human_velocity, human_valid)

        predicted = observation.clone()
        commands = self._pending_commands().to(dtype=self.ukf_dtype)
        rollout_state = self.ukf_state.clone()
        drz = observation[:, ObservationIndex.DRZ].to(dtype=self.ukf_dtype)
        omega_prev = observation[:, ObservationIndex.OMEGA].to(dtype=self.ukf_dtype)
        vrz_prev = observation[:, ObservationIndex.VRZ].to(dtype=self.ukf_dtype)
        pg = observation[:, ObservationIndex.PG].to(dtype=self.ukf_dtype)

        error_curr = current_error[:, 0].to(dtype=self.ukf_dtype).clone()
        error_prev = self._prepare_error_prev1(error_prev1, current_error)[:, 0].to(dtype=self.ukf_dtype).clone()
        predicted_error_dot = torch.zeros_like(error_curr)
        predicted_error_ddot = torch.zeros_like(error_curr)

        ab = observation[:, ObservationIndex.AB].to(dtype=self.ukf_dtype)
        omega = omega_prev.clone()
        alpha = observation[:, ObservationIndex.ALPHA].to(dtype=self.ukf_dtype)
        vrz = vrz_prev.clone()
        arz = observation[:, ObservationIndex.ARZ].to(dtype=self.ukf_dtype)

        for step_id in range(commands.shape[0]):
            vrz = commands[step_id, :, 0]
            rollout_state = self._propagate_ukf_state(rollout_state, vrz)
            omega = self._omega_from_state(rollout_state, vrz)
            ab = self._ball_acceleration_from_state(rollout_state, omega)
            alpha = (omega - omega_prev) / self.step_dt
            arz = (vrz - vrz_prev) / self.step_dt
            drz = drz + vrz * self.step_dt
            omega_prev = omega
            vrz_prev = vrz
            new_error = rollout_state[:, self.IDX_PB] - pg
            predicted_error_dot = new_error - error_curr
            predicted_error_ddot = predicted_error_dot - (error_curr - error_prev)
            error_prev = error_curr
            error_curr = new_error

        predicted[:, ObservationIndex.PB] = rollout_state[:, self.IDX_PB].to(dtype=torch.float32)
        predicted[:, ObservationIndex.VB] = rollout_state[:, self.IDX_VB].to(dtype=torch.float32)
        predicted[:, ObservationIndex.AB] = ab.to(dtype=torch.float32)
        predicted[:, ObservationIndex.THETA] = rollout_state[:, self.IDX_THETA].to(dtype=torch.float32)
        predicted[:, ObservationIndex.OMEGA] = omega.to(dtype=torch.float32)
        predicted[:, ObservationIndex.ALPHA] = alpha.to(dtype=torch.float32)
        predicted[:, ObservationIndex.DRZ] = drz.to(dtype=torch.float32)
        predicted[:, ObservationIndex.VRZ] = vrz.to(dtype=torch.float32)
        predicted[:, ObservationIndex.ARZ] = arz.to(dtype=torch.float32)
        predicted[:, ObservationIndex.PG] = observation[:, ObservationIndex.PG]
        predicted[:, ObservationIndex.A_PREV] = observation[:, ObservationIndex.A_PREV]

        self.predicted_observation.copy_(predicted)
        self.predicted_error.copy_(error_curr.unsqueeze(-1).to(dtype=torch.float32))
        self.predicted_error_dot.copy_(predicted_error_dot.unsqueeze(-1).to(dtype=torch.float32))
        self.predicted_error_ddot.copy_(predicted_error_ddot.unsqueeze(-1).to(dtype=torch.float32))
        return predicted

    def get_state(self) -> dict[str, torch.Tensor]:
        """Return predictor state and UKF diagnostics for rollout logging."""
        state = super().get_state()
        state.update(
            {
                "policy_predictor_type_id": torch.full((self.num_envs,), 1.0, device=self.device),
                "policy_ukf_estimated_v_hz": self.ukf_state[:, self.IDX_V_HZ].to(dtype=torch.float32),
                "policy_ukf_estimated_a_hz": self.ukf_state[:, self.IDX_A_HZ].to(dtype=torch.float32),
                "policy_ukf_estimated_j_hz": self.ukf_state[:, self.IDX_J_HZ].to(dtype=torch.float32),
                "policy_ukf_measured_v_hz": self.measured_human_velocity,
                "policy_ukf_human_velocity_valid": self.human_velocity_valid,
                "policy_ukf_covariance_trace": self.ukf_covariance_trace,
                "policy_ukf_innovation_norm": self.ukf_innovation_norm,
                "policy_ukf_omega_residual": self.ukf_omega_residual,
            }
        )
        return state

    def _initialize_ukf_state(self, mask: torch.Tensor, observation: torch.Tensor, human_velocity: torch.Tensor):
        env_ids = mask.nonzero(as_tuple=False).squeeze(-1)
        state = torch.zeros((env_ids.numel(), self.STATE_DIM), device=self.device, dtype=self.ukf_dtype)
        state[:, self.IDX_PB] = observation[env_ids, ObservationIndex.PB].to(dtype=self.ukf_dtype)
        state[:, self.IDX_VB] = observation[env_ids, ObservationIndex.VB].to(dtype=self.ukf_dtype)
        state[:, self.IDX_THETA] = observation[env_ids, ObservationIndex.THETA].to(dtype=self.ukf_dtype)
        state[:, self.IDX_V_HZ] = human_velocity[env_ids].to(dtype=self.ukf_dtype)
        self.ukf_state[env_ids] = state
        self.ukf_covariance[env_ids] = self.initial_covariance
        self.ukf_initialized[env_ids] = True

    def _ukf_step(
        self,
        env_ids: torch.Tensor,
        measurement: torch.Tensor,
        drone_velocity: torch.Tensor,
        human_valid: torch.Tensor,
    ):
        state = self.ukf_state[env_ids]
        covariance = self.ukf_covariance[env_ids]
        sigma_points = self._sigma_points(state, covariance)
        propagated_sigma_points = self._propagate_ukf_state(sigma_points, drone_velocity.unsqueeze(-1))

        predicted_state = torch.sum(self.mean_weights.view(1, -1, 1) * propagated_sigma_points, dim=1)
        state_diff = propagated_sigma_points - predicted_state.unsqueeze(1)
        predicted_covariance = torch.einsum(
            "s,nsi,nsj->nij",
            self.covariance_weights,
            state_diff,
            state_diff,
        ) + self.process_covariance.unsqueeze(0)
        predicted_covariance = self._symmetrize(predicted_covariance)

        predicted_measurements = self._measurement_model(propagated_sigma_points, drone_velocity.unsqueeze(-1))
        predicted_measurement_mean = torch.sum(
            self.mean_weights.view(1, -1, 1) * predicted_measurements,
            dim=1,
        )

        measurement_for_update = measurement.clone()
        measurement_noise = self.measurement_covariance.unsqueeze(0).repeat(env_ids.numel(), 1, 1)
        invalid_human = ~human_valid.to(device=self.device, dtype=torch.bool)
        if torch.any(invalid_human):
            measurement_for_update[invalid_human, 4] = predicted_measurement_mean[invalid_human, 4]
            measurement_noise[invalid_human, 4, 4] = 1e6

        measurement_diff = predicted_measurements - predicted_measurement_mean.unsqueeze(1)
        state_measurement_covariance = torch.einsum(
            "s,nsi,nsj->nij",
            self.covariance_weights,
            state_diff,
            measurement_diff,
        )
        innovation_covariance = torch.einsum(
            "s,nsi,nsj->nij",
            self.covariance_weights,
            measurement_diff,
            measurement_diff,
        ) + measurement_noise
        innovation_covariance = self._symmetrize(innovation_covariance)
        stable_innovation_covariance = innovation_covariance + self.ukf_jitter * self.measurement_identity.unsqueeze(0)

        try:
            kalman_gain = torch.linalg.solve(
                stable_innovation_covariance,
                state_measurement_covariance.transpose(-1, -2),
            ).transpose(-1, -2)
        except RuntimeError:
            kalman_gain = state_measurement_covariance @ torch.linalg.pinv(stable_innovation_covariance)

        innovation = measurement_for_update - predicted_measurement_mean
        updated_state = predicted_state + torch.einsum("nij,nj->ni", kalman_gain, innovation)
        updated_covariance = (
            predicted_covariance
            - kalman_gain @ stable_innovation_covariance @ kalman_gain.transpose(-1, -2)
        )
        updated_covariance = self._symmetrize(updated_covariance)
        updated_covariance = updated_covariance + self.ukf_jitter * self.ukf_identity.unsqueeze(0)

        self.ukf_state[env_ids] = updated_state
        self.ukf_covariance[env_ids] = updated_covariance
        self.ukf_innovation_norm[env_ids] = torch.linalg.norm(innovation.to(dtype=torch.float32), dim=-1)

    def _human_velocity_from_extras(self, extras: Mapping | None) -> tuple[torch.Tensor, torch.Tensor]:
        value = None
        if isinstance(extras, Mapping):
            step_info = extras.get("step", {})
            if isinstance(step_info, Mapping) and self.human_velocity_key in step_info:
                value = step_info[self.human_velocity_key]
            elif self.human_velocity_key in extras:
                value = extras[self.human_velocity_key]

        valid = torch.zeros((self.num_envs,), device=self.device, dtype=torch.bool)
        if value is None:
            if self.missing_human_velocity_policy == "error":
                raise ValueError(
                    f"UKFJerkHumanVelocityStatePredictor requires extras['step']['{self.human_velocity_key}']."
                )
            human_velocity = self._missing_human_velocity_fallback()
        else:
            raw = torch.as_tensor(value, device=self.device, dtype=torch.float32)
            if raw.ndim == 0:
                raw = raw.expand(self.num_envs)
            if raw.ndim == 2 and raw.shape[1] == 1:
                raw = raw[:, 0]
            if raw.shape != (self.num_envs,):
                raise ValueError(
                    f"UKFJerkHumanVelocityStatePredictor expected human velocity shape ({self.num_envs},), "
                    f"got {tuple(raw.shape)}."
                )
            measured = raw * float(self.human_velocity_sign)
            valid = torch.isfinite(measured)
            fallback = self._missing_human_velocity_fallback()
            human_velocity = torch.where(valid, measured, fallback)

        self.measured_human_velocity.copy_(human_velocity)
        self.human_velocity_valid.copy_(valid.to(dtype=torch.float32))
        if torch.any(valid):
            self.last_human_velocity[valid] = human_velocity[valid]
        return human_velocity, valid

    def _missing_human_velocity_fallback(self) -> torch.Tensor:
        if self.missing_human_velocity_policy == "last":
            return self.last_human_velocity.clone()
        return torch.zeros((self.num_envs,), device=self.device, dtype=torch.float32)

    def _refresh_ukf_diagnostics(
        self,
        observation: torch.Tensor,
        human_velocity: torch.Tensor,
        human_valid: torch.Tensor,
    ):
        del human_velocity
        omega_model = self._omega_from_state(
            self.ukf_state,
            observation[:, ObservationIndex.VRZ].to(dtype=self.ukf_dtype),
        )
        self.ukf_omega_residual.copy_(
            (observation[:, ObservationIndex.OMEGA].to(dtype=self.ukf_dtype) - omega_model).to(dtype=torch.float32)
        )
        self.ukf_covariance_trace.copy_(
            torch.diagonal(self.ukf_covariance, dim1=-2, dim2=-1).sum(dim=-1).to(dtype=torch.float32)
        )
        self.human_velocity_valid.copy_(human_valid.to(dtype=torch.float32))

    def _sigma_points(self, state: torch.Tensor, covariance: torch.Tensor) -> torch.Tensor:
        covariance = self._symmetrize(covariance)
        cholesky = self._cholesky_factor(covariance)
        offsets = cholesky.transpose(-1, -2)
        sigma_points = torch.zeros(
            (state.shape[0], 2 * self.STATE_DIM + 1, self.STATE_DIM),
            device=self.device,
            dtype=self.ukf_dtype,
        )
        sigma_points[:, 0, :] = state
        sigma_points[:, 1 : self.STATE_DIM + 1, :] = state.unsqueeze(1) + offsets
        sigma_points[:, self.STATE_DIM + 1 :, :] = state.unsqueeze(1) - offsets
        return sigma_points

    def _cholesky_factor(self, covariance: torch.Tensor) -> torch.Tensor:
        for jitter in (self.ukf_jitter, 1e-9, 1e-7, 1e-5):
            matrix = self.ukf_weight_scale * (covariance + jitter * self.ukf_identity.unsqueeze(0))
            cholesky, info = torch.linalg.cholesky_ex(matrix)
            if bool(torch.all(info == 0)):
                return cholesky

        eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
        eigenvalues = torch.clamp(eigenvalues, min=1e-8)
        covariance_psd = eigenvectors @ torch.diag_embed(eigenvalues) @ eigenvectors.transpose(-1, -2)
        matrix = self.ukf_weight_scale * (covariance_psd + 1e-8 * self.ukf_identity.unsqueeze(0))
        return torch.linalg.cholesky(matrix)

    def _propagate_ukf_state(self, state: torch.Tensor, drone_velocity: torch.Tensor) -> torch.Tensor:
        dt = self.step_dt
        if self.solver == "euler":
            return state + dt * self._ukf_dynamics(state, drone_velocity)

        k1 = self._ukf_dynamics(state, drone_velocity)
        k2 = self._ukf_dynamics(state + 0.5 * dt * k1, drone_velocity)
        k3 = self._ukf_dynamics(state + 0.5 * dt * k2, drone_velocity)
        k4 = self._ukf_dynamics(state + dt * k3, drone_velocity)
        return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def _ukf_dynamics(self, state: torch.Tensor, drone_velocity: torch.Tensor) -> torch.Tensor:
        omega = self._omega_from_state(state, drone_velocity)
        ball_acceleration = self._ball_acceleration_from_state(state, omega)
        return torch.stack(
            (
                state[..., self.IDX_VB],
                ball_acceleration,
                omega,
                state[..., self.IDX_A_HZ],
                state[..., self.IDX_J_HZ],
                -self.lambda_j * state[..., self.IDX_J_HZ],
            ),
            dim=-1,
        )

    def _measurement_model(self, state: torch.Tensor, drone_velocity: torch.Tensor) -> torch.Tensor:
        omega = self._omega_from_state(state, drone_velocity)
        return torch.stack(
            (
                state[..., self.IDX_PB],
                state[..., self.IDX_VB],
                state[..., self.IDX_THETA],
                omega,
                state[..., self.IDX_V_HZ],
            ),
            dim=-1,
        )

    def _omega_from_state(self, state: torch.Tensor, drone_velocity: torch.Tensor) -> torch.Tensor:
        theta = state[..., self.IDX_THETA]
        human_velocity = state[..., self.IDX_V_HZ]
        beta = self._beta(theta)
        denominator = self.plank_length * self._safe_denominator(torch.cos(beta - theta))
        return (human_velocity - drone_velocity) * torch.cos(beta) / denominator

    def _ball_acceleration_from_state(self, state: torch.Tensor, omega: torch.Tensor) -> torch.Tensor:
        ball_inertia = self.ball_inertia_ratio * self.ball_mass * self.ball_radius**2
        effective_mass = ball_inertia / self.ball_radius**2 + self.ball_mass
        ball_position = state[..., self.IDX_PB]
        theta = state[..., self.IDX_THETA]
        ab = self.ball_mass * (ball_position + self.ball_position_offset - self.plank_length) * omega.square()
        ab -= self.ball_mass * self.gravity * torch.sin(theta)
        return ab / effective_mass

    def _diag_from_cfg(self, values: Sequence[float], expected_dim: int, field_name: str) -> torch.Tensor:
        diagonal = torch.as_tensor(values, device=self.device, dtype=self.ukf_dtype)
        if diagonal.shape != (expected_dim,):
            raise ValueError(
                f"VelocityModelStatePredictorCfg.{field_name} must have shape ({expected_dim},), "
                f"got {tuple(diagonal.shape)}."
            )
        if torch.any(diagonal < 0.0):
            raise ValueError(f"VelocityModelStatePredictorCfg.{field_name} values must be non-negative.")
        return torch.diag(diagonal)

    @staticmethod
    def _symmetrize(matrix: torch.Tensor) -> torch.Tensor:
        return 0.5 * (matrix + matrix.transpose(-1, -2))


class UKFJerkHumanAccelerationStatePredictor(UKFJerkHumanVelocityStatePredictor, AccelerationModelStatePredictor):
    """Acceleration-interface delay compensator with a random-jerk human endpoint UKF."""

    IDX_ROLLOUT_DRZ = 6
    IDX_ROLLOUT_VRZ = 7
    ROLLOUT_STATE_DIM = 8

    def predict(
        self,
        observation: torch.Tensor,
        error_prev1: torch.Tensor | None = None,
        extras: Mapping | None = None,
    ) -> torch.Tensor:
        """Return a delay-compensated 11-D observation using acceleration commands and UKF human estimates."""
        observation = observation.to(device=self.device, dtype=torch.float32)
        if observation.shape != (self.num_envs, 11):
            raise ValueError(
                "UKFJerkHumanAccelerationStatePredictor expects observation shape "
                f"({self.num_envs}, 11), got {tuple(observation.shape)}."
            )

        current_error = (
            observation[:, ObservationIndex.PB : ObservationIndex.PB + 1]
            - observation[:, ObservationIndex.PG : ObservationIndex.PG + 1]
        )
        if not self.active:
            self.predicted_observation.copy_(observation)
            self.predicted_error.copy_(current_error)
            self.predicted_error_dot.zero_()
            self.predicted_error_ddot.zero_()
            self._set_inactive_response_diagnostics(observation)
            return observation

        human_velocity, human_valid = self._human_velocity_from_extras(extras)
        measurement = torch.stack(
            (
                observation[:, ObservationIndex.PB],
                observation[:, ObservationIndex.VB],
                observation[:, ObservationIndex.THETA],
                observation[:, ObservationIndex.OMEGA],
                human_velocity,
            ),
            dim=-1,
        ).to(dtype=self.ukf_dtype)
        current_drone_velocity = observation[:, ObservationIndex.VRZ].to(dtype=self.ukf_dtype)

        was_initialized = self.ukf_initialized.clone()
        needs_init = ~was_initialized
        if torch.any(needs_init):
            self._initialize_ukf_state(needs_init, observation, human_velocity)
        if torch.any(was_initialized):
            env_ids = was_initialized.nonzero(as_tuple=False).squeeze(-1)
            self._ukf_step(
                env_ids,
                measurement[env_ids],
                current_drone_velocity[env_ids],
                human_valid[env_ids],
            )
        self._refresh_ukf_diagnostics(observation, human_velocity, human_valid)

        predicted = observation.clone()
        commands = self._pending_commands()
        response_state = self._acceleration_response_state(observation, extras)
        rollout_state = torch.cat(
            (
                self.ukf_state.clone(),
                observation[:, ObservationIndex.DRZ : ObservationIndex.DRZ + 1].to(dtype=self.ukf_dtype),
                observation[:, ObservationIndex.VRZ : ObservationIndex.VRZ + 1].to(dtype=self.ukf_dtype),
            ),
            dim=-1,
        )
        omega_prev = observation[:, ObservationIndex.OMEGA].to(dtype=self.ukf_dtype)
        pg = observation[:, ObservationIndex.PG].to(dtype=self.ukf_dtype)

        error_curr = current_error[:, 0].to(dtype=self.ukf_dtype).clone()
        error_prev = self._prepare_error_prev1(error_prev1, current_error)[:, 0].to(dtype=self.ukf_dtype).clone()
        predicted_error_dot = torch.zeros_like(error_curr)
        predicted_error_ddot = torch.zeros_like(error_curr)

        human_state = rollout_state[:, : self.STATE_DIM]
        drz = rollout_state[:, self.IDX_ROLLOUT_DRZ]
        vrz = rollout_state[:, self.IDX_ROLLOUT_VRZ]
        omega = omega_prev.clone()
        ab = observation[:, ObservationIndex.AB].to(dtype=self.ukf_dtype)
        alpha = observation[:, ObservationIndex.ALPHA].to(dtype=self.ukf_dtype)
        arz = observation[:, ObservationIndex.ARZ].to(dtype=self.ukf_dtype)

        for step_id in range(commands.shape[0]):
            reference_arz = commands[step_id, :, 0].to(dtype=torch.float32)
            arz = self._response_executed_acceleration(reference_arz, response_state).to(dtype=self.ukf_dtype)
            rollout_state = self._propagate_acceleration_human_rollout_state(rollout_state, arz)
            human_state = rollout_state[:, : self.STATE_DIM]
            drz = rollout_state[:, self.IDX_ROLLOUT_DRZ]
            vrz = rollout_state[:, self.IDX_ROLLOUT_VRZ]
            omega = self._omega_from_human_state_and_vrz(human_state, vrz)
            ab = self._ball_acceleration_from_state(human_state, omega)
            alpha = (omega - omega_prev) / self.step_dt
            omega_prev = omega
            new_error = human_state[:, self.IDX_PB] - pg
            predicted_error_dot = new_error - error_curr
            predicted_error_ddot = predicted_error_dot - (error_curr - error_prev)
            error_prev = error_curr
            error_curr = new_error

        predicted[:, ObservationIndex.PB] = human_state[:, self.IDX_PB].to(dtype=torch.float32)
        predicted[:, ObservationIndex.VB] = human_state[:, self.IDX_VB].to(dtype=torch.float32)
        predicted[:, ObservationIndex.AB] = ab.to(dtype=torch.float32)
        predicted[:, ObservationIndex.THETA] = human_state[:, self.IDX_THETA].to(dtype=torch.float32)
        predicted[:, ObservationIndex.OMEGA] = omega.to(dtype=torch.float32)
        predicted[:, ObservationIndex.ALPHA] = alpha.to(dtype=torch.float32)
        predicted[:, ObservationIndex.DRZ] = drz.to(dtype=torch.float32)
        predicted[:, ObservationIndex.VRZ] = vrz.to(dtype=torch.float32)
        predicted[:, ObservationIndex.ARZ] = arz.to(dtype=torch.float32)
        predicted[:, ObservationIndex.PG] = observation[:, ObservationIndex.PG]
        predicted[:, ObservationIndex.A_PREV] = observation[:, ObservationIndex.A_PREV]

        self.predicted_observation.copy_(predicted)
        self.predicted_error.copy_(error_curr.unsqueeze(-1).to(dtype=torch.float32))
        self.predicted_error_dot.copy_(predicted_error_dot.unsqueeze(-1).to(dtype=torch.float32))
        self.predicted_error_ddot.copy_(predicted_error_ddot.unsqueeze(-1).to(dtype=torch.float32))
        return predicted

    def get_state(self) -> dict[str, torch.Tensor]:
        """Return response-aware acceleration predictor state and UKF diagnostics."""
        state = super().get_state()
        state["policy_predictor_type_id"] = torch.full((self.num_envs,), 2.0, device=self.device)
        return state

    def _propagate_acceleration_human_rollout_state(
        self,
        state: torch.Tensor,
        drone_acceleration: torch.Tensor,
    ) -> torch.Tensor:
        dt = self.step_dt
        if self.solver == "euler":
            return state + dt * self._acceleration_human_rollout_dynamics(state, drone_acceleration)

        k1 = self._acceleration_human_rollout_dynamics(state, drone_acceleration)
        k2 = self._acceleration_human_rollout_dynamics(state + 0.5 * dt * k1, drone_acceleration)
        k3 = self._acceleration_human_rollout_dynamics(state + 0.5 * dt * k2, drone_acceleration)
        k4 = self._acceleration_human_rollout_dynamics(state + dt * k3, drone_acceleration)
        return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def _acceleration_human_rollout_dynamics(
        self,
        state: torch.Tensor,
        drone_acceleration: torch.Tensor,
    ) -> torch.Tensor:
        drone_acceleration = drone_acceleration.to(device=self.device, dtype=self.ukf_dtype)
        human_state = state[..., : self.STATE_DIM]
        drone_velocity = state[..., self.IDX_ROLLOUT_VRZ]
        omega = self._omega_from_human_state_and_vrz(human_state, drone_velocity)
        ball_acceleration = self._ball_acceleration_from_state(human_state, omega)
        return torch.stack(
            (
                human_state[..., self.IDX_VB],
                ball_acceleration,
                omega,
                human_state[..., self.IDX_A_HZ],
                human_state[..., self.IDX_J_HZ],
                -self.lambda_j * human_state[..., self.IDX_J_HZ],
                drone_velocity,
                drone_acceleration,
            ),
            dim=-1,
        )

    def _omega_from_human_state_and_vrz(self, state: torch.Tensor, drone_velocity: torch.Tensor) -> torch.Tensor:
        theta = state[..., self.IDX_THETA]
        human_velocity = state[..., self.IDX_V_HZ]
        drone_velocity = drone_velocity.to(device=self.device, dtype=state.dtype)
        beta = self._beta(theta)
        denominator = self.plank_length * self._safe_denominator(torch.cos(beta - theta))
        return (human_velocity - drone_velocity) * torch.cos(beta) / denominator


def make_velocity_state_predictor(
    cfg: VelocityModelStatePredictorCfg,
    num_envs: int,
    device: str | torch.device,
) -> VelocityModelStatePredictor:
    """Create the configured velocity-interface state predictor."""
    predictor_type = str(getattr(cfg, "type", "nominal")).lower().replace("-", "_")
    if predictor_type in {"nominal", "model", "velocity"}:
        return VelocityModelStatePredictor(cfg, num_envs, device)
    if predictor_type in {"ukf_random_jerk", "ukf_jerk_human"}:
        return UKFJerkHumanVelocityStatePredictor(cfg, num_envs, device)
    raise ValueError(
        "VelocityModelStatePredictorCfg.type must be 'nominal' or 'ukf_random_jerk', "
        f"got '{predictor_type}'."
    )


def make_acceleration_state_predictor(
    cfg: VelocityModelStatePredictorCfg,
    num_envs: int,
    device: str | torch.device,
) -> AccelerationModelStatePredictor:
    """Create the configured acceleration-interface state predictor."""
    predictor_type = str(getattr(cfg, "type", "nominal")).lower().replace("-", "_")
    if predictor_type in {"nominal", "model", "acceleration"}:
        return AccelerationModelStatePredictor(cfg, num_envs, device)
    if predictor_type in {"ukf_random_jerk", "ukf_jerk_human", "ukf_jerk_human_acceleration"}:
        return UKFJerkHumanAccelerationStatePredictor(cfg, num_envs, device)
    raise ValueError(
        "VelocityModelStatePredictorCfg.type must be 'nominal' or 'ukf_random_jerk' "
        "for acceleration-interface prediction, "
        f"got '{predictor_type}'."
    )
