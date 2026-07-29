"""Policy-side model state predictors for action-delay compensation."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch

from .base_policy import ObservationIndex


def extract_reference_positions(
    observation: torch.Tensor,
    observation_fields: Sequence[str],
    future_steps: int,
) -> torch.Tensor:
    """Extract ``pg[0:future_steps + 1]`` from a named raw observation."""
    if future_steps < 0:
        raise ValueError(f"future_steps must be non-negative, got {future_steps}.")
    if observation.ndim != 2:
        raise ValueError(
            "Reference extraction expects a 2-D observation, "
            f"got shape {tuple(observation.shape)}."
        )
    if isinstance(observation_fields, (str, bytes)):
        raise TypeError("observation_fields must be a sequence of field names, not a string.")

    fields = tuple(str(name) for name in observation_fields)
    if len(fields) != observation.shape[1]:
        raise ValueError(
            "observation_fields length must match the raw observation dimension: "
            f"got {len(fields)} fields for shape {tuple(observation.shape)}."
        )

    current_names = [name for name in ("pg", "pg_0") if name in fields]
    if len(current_names) != 1:
        raise ValueError(
            "Reference extraction requires exactly one current-reference field named "
            f"'pg' or 'pg_0'; got {current_names or 'none'}."
        )
    required_fields = [current_names[0], *(f"pg_{offset}" for offset in range(1, future_steps + 1))]
    missing_fields = [name for name in required_fields if name not in fields]
    if missing_fields:
        raise ValueError(
            "Reference preview is shorter than the state-predictor horizon. "
            f"Missing field(s): {', '.join(missing_fields)}. Configure "
            f"reference_preview.future_steps >= {future_steps}."
        )
    duplicate_fields = [name for name in required_fields if fields.count(name) != 1]
    if duplicate_fields:
        raise ValueError(
            "Reference preview fields must be unique; duplicate field(s): "
            f"{', '.join(duplicate_fields)}."
        )

    indices = [fields.index(name) for name in required_fields]
    return observation[:, indices]


def validate_reference_preview_horizon(
    *,
    delay_step: int,
    preview_enabled: bool,
    preview_future_steps: int,
    context: str,
) -> None:
    """Validate the preview contract for an active moving-reference predictor."""
    delay_step = int(delay_step)
    preview_future_steps = int(preview_future_steps)
    if delay_step <= 0:
        return
    if not preview_enabled:
        raise ValueError(
            f"{context} requires reference_preview.enabled=true when the state predictor "
            f"is active with delay_step={delay_step}."
        )
    if preview_future_steps < delay_step:
        raise ValueError(
            f"{context} requires reference_preview.future_steps >= state predictor "
            f"delay_step; got {preview_future_steps} < {delay_step}."
        )


@dataclass
class VelocityModelStatePredictorCfg:
    """Configuration for the velocity-interface model state predictor."""

    enabled: bool = False
    delay_step: int | str = 0
    solver: str = "rk4"
    step_dt: float | str = 0.0
    max_acc: float | str = 0.5
    max_velocity: float | str = 0.0
    velocity_response_enabled: bool = True
    velocity_response_tau_s: float | str = 0.139
    velocity_response_gain: float | str = 1.0
    velocity_response_bias: float | str = -0.00055
    velocity_response_max_abs_velocity: float | str = 0.0
    plank_length: float | str = 1.06
    rope_length: float | str = 0.9
    ball_position_offset: float | str = 0.33
    gravity: float | str = 9.81
    ball_mass: float | str = 0.0005
    ball_radius: float | str = 0.023
    ball_inertia_ratio: float | str = 0.4
    epsilon: float | str = 1e-6

    @classmethod
    def from_dict(cls, data: Mapping | None) -> "VelocityModelStatePredictorCfg":
        """Build a predictor config from a YAML dictionary."""
        cfg = cls()
        if not data:
            return cfg
        for key, value in data.items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)
        return cfg

    def resolve_delay_step_from_robustness(self, robustness_cfg) -> None:
        """Resolve an ``auto`` delay to a fixed nominal predictor horizon."""
        if not _is_auto(self.delay_step):
            return
        if not self.enabled:
            self.delay_step = 0
            return

        delay_active = bool(
            getattr(robustness_cfg, "enabled", False)
            and getattr(robustness_cfg, "action_delay_enabled", False)
        )
        if not delay_active:
            self.delay_step = 0
            return

        raw_choices = getattr(robustness_cfg, "delay_step_choices", ())
        choices = tuple(int(value) for value in (raw_choices or ()))
        unique_choices = set(choices)
        if len(unique_choices) > 1:
            raise ValueError(
                "VelocityModelStatePredictorCfg.delay_step='auto' cannot resolve a per-environment "
                "random action delay. Configure an explicit fixed nominal delay_step for the predictor."
            )
        if unique_choices:
            self.delay_step = unique_choices.pop()
            return
        self.delay_step = int(getattr(robustness_cfg, "delay_step", 0))

    def resolve_velocity_response_from_robustness(self, robustness_cfg) -> None:
        """Resolve ``auto`` as the final response or nominal simulator model."""
        response_active = bool(
            getattr(robustness_cfg, "enabled", False)
            and getattr(robustness_cfg, "velocity_response_enabled", False)
        )
        target_or_sim_fields = (
            (
                "velocity_response_tau_s",
                "velocity_response_tau_s_range",
                "velocity_response_sim_tau_s",
            ),
            (
                "velocity_response_gain",
                "velocity_response_gain_range",
                "velocity_response_sim_gain",
            ),
            (
                "velocity_response_bias",
                "velocity_response_bias_range",
                "velocity_response_sim_bias",
            ),
        )
        for predictor_field, target_field, sim_field in target_or_sim_fields:
            if not _is_auto(getattr(self, predictor_field)):
                continue
            if not response_active:
                setattr(self, predictor_field, float(getattr(robustness_cfg, sim_field)))
                continue

            bounds = getattr(robustness_cfg, target_field)
            if len(bounds) != 2:
                raise ValueError(f"robustness.{target_field} must contain exactly two values.")
            lower, upper = (float(value) for value in bounds)
            if lower != upper:
                raise ValueError(
                    f"VelocityModelStatePredictorCfg.{predictor_field}='auto' requires "
                    f"robustness.{target_field} to have equal bounds; got [{lower}, {upper}]. "
                    "Set an explicit nominal predictor value for randomized response parameters."
                )
            setattr(self, predictor_field, lower)

        if _is_auto(self.velocity_response_max_abs_velocity):
            max_field = (
                "velocity_response_max_abs_velocity"
                if response_active
                else "velocity_response_sim_max_abs_velocity"
            )
            self.velocity_response_max_abs_velocity = float(getattr(robustness_cfg, max_field))


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
        self.velocity_response_enabled = bool(cfg.velocity_response_enabled)
        self.velocity_response_tau_s = self._as_float(
            cfg.velocity_response_tau_s,
            "velocity_response_tau_s",
        )
        self.velocity_response_gain = self._as_float(
            cfg.velocity_response_gain,
            "velocity_response_gain",
        )
        self.velocity_response_bias = self._as_float(
            cfg.velocity_response_bias,
            "velocity_response_bias",
        )
        self.velocity_response_max_abs_velocity = self._as_float(
            cfg.velocity_response_max_abs_velocity,
            "velocity_response_max_abs_velocity",
        )
        self.plank_length = self._as_float(cfg.plank_length, "plank_length")
        self.rope_length = self._as_float(cfg.rope_length, "rope_length")
        self.ball_position_offset = self._as_float(
            cfg.ball_position_offset,
            "ball_position_offset",
        )
        self.gravity = abs(self._as_float(cfg.gravity, "gravity"))
        self.ball_mass = self._as_float(cfg.ball_mass, "ball_mass")
        self.ball_radius = self._as_float(cfg.ball_radius, "ball_radius")
        self.ball_inertia_ratio = float(cfg.ball_inertia_ratio)
        self.epsilon = float(cfg.epsilon)

        if self.step_dt <= 0.0:
            raise ValueError("VelocityModelStatePredictor requires step_dt > 0.")
        if self.delay_step < 0:
            raise ValueError("VelocityModelStatePredictor requires delay_step >= 0.")
        if self.max_acc <= 0.0:
            raise ValueError("VelocityModelStatePredictor requires max_acc > 0.")
        if not math.isfinite(self.velocity_response_tau_s) or self.velocity_response_tau_s < 0.0:
            raise ValueError(
                "VelocityModelStatePredictor requires finite velocity_response_tau_s >= 0."
            )
        if not math.isfinite(self.velocity_response_gain) or self.velocity_response_gain <= 0.0:
            raise ValueError(
                "VelocityModelStatePredictor requires finite velocity_response_gain > 0."
            )
        if not math.isfinite(self.velocity_response_bias):
            raise ValueError("VelocityModelStatePredictor requires finite velocity_response_bias.")
        if (
            not math.isfinite(self.velocity_response_max_abs_velocity)
            or self.velocity_response_max_abs_velocity < 0.0
        ):
            raise ValueError(
                "VelocityModelStatePredictor requires finite "
                "velocity_response_max_abs_velocity >= 0."
            )
        if self.plank_length <= 0.0 or self.rope_length <= 0.0:
            raise ValueError("VelocityModelStatePredictor requires positive plank_length and rope_length.")
        if not math.isfinite(self.ball_position_offset):
            raise ValueError("VelocityModelStatePredictor requires finite ball_position_offset.")
        if self.ball_mass <= 0.0 or self.ball_radius <= 0.0:
            raise ValueError("VelocityModelStatePredictor requires positive ball_mass and ball_radius.")
        if self.ball_inertia_ratio < 0.0:
            raise ValueError("VelocityModelStatePredictor requires non-negative ball_inertia_ratio.")
        if self.solver not in {"euler", "rk4"}:
            raise ValueError("VelocityModelStatePredictor solver must be 'euler' or 'rk4'.")

        response_decay = 0.0
        if self.velocity_response_tau_s > 0.0:
            response_decay = torch.exp(
                torch.tensor(
                    -self.step_dt / self.velocity_response_tau_s,
                    device=self.device,
                    dtype=torch.float32,
                )
            ).item()
        self.velocity_response_decay = torch.tensor(
            response_decay,
            device=self.device,
            dtype=torch.float32,
        )
        self.command_z = torch.zeros((self.num_envs, 1), device=self.device)
        self.last_action = torch.zeros((self.num_envs, 1), device=self.device)
        self.predicted_observation = torch.zeros((self.num_envs, 11), device=self.device)
        self.predicted_error = torch.zeros((self.num_envs, 1), device=self.device)
        self.predicted_error_dot = torch.zeros((self.num_envs, 1), device=self.device)
        self.predicted_error_ddot = torch.zeros((self.num_envs, 1), device=self.device)
        self.reference_preview_used = torch.zeros((self.num_envs,), device=self.device)
        self.reference_horizon = torch.zeros((self.num_envs,), device=self.device)
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
        self.reference_preview_used[env_ids] = 0.0
        self.reference_horizon[env_ids] = 0.0
        if self.delay_step > 0:
            self.command_queue[:, env_ids] = 0.0

    def predict(
        self,
        observation: torch.Tensor,
        error_prev1: torch.Tensor | None = None,
        *,
        reference_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return a future 11-D observation after the configured delay horizon."""
        observation = observation.to(device=self.device, dtype=torch.float32)
        if observation.shape != (self.num_envs, 11):
            raise ValueError(
                "VelocityModelStatePredictor expects observation shape "
                f"({self.num_envs}, 11), got {tuple(observation.shape)}."
            )
        current_pg = observation[:, ObservationIndex.PG : ObservationIndex.PG + 1]
        current_error = (
            observation[:, ObservationIndex.PB : ObservationIndex.PB + 1]
            - current_pg
        )
        if not self.active:
            self.predicted_observation.copy_(observation)
            self.predicted_error.copy_(current_error)
            self.predicted_error_dot.zero_()
            self.predicted_error_ddot.zero_()
            self.reference_preview_used.zero_()
            self.reference_horizon.zero_()
            return observation

        reference_positions, preview_used = self._prepare_reference_positions(
            reference_positions,
            current_pg,
        )
        self.reference_preview_used.fill_(float(preview_used))
        self.reference_horizon.fill_(float(self.delay_step if preview_used else 0))

        predicted = observation.clone()
        commands = self._pending_commands()
        pb = predicted[:, ObservationIndex.PB].clone()
        vb = predicted[:, ObservationIndex.VB].clone()
        theta = predicted[:, ObservationIndex.THETA].clone()
        drz = predicted[:, ObservationIndex.DRZ].clone()
        omega_prev = predicted[:, ObservationIndex.OMEGA].clone()
        vrz_prev = predicted[:, ObservationIndex.VRZ].clone()
        response_nominal_z = vrz_prev.clone()

        error_curr = current_error[:, 0].clone()
        error_prev = self._prepare_error_prev1(error_prev1, current_error)[:, 0].clone()
        predicted_error_dot = torch.zeros_like(error_curr)
        predicted_error_ddot = torch.zeros_like(error_curr)
        ab = predicted[:, ObservationIndex.AB].clone()
        omega = omega_prev.clone()
        alpha = predicted[:, ObservationIndex.ALPHA].clone()
        vrz = vrz_prev.clone()
        arz = predicted[:, ObservationIndex.ARZ].clone()

        for step_id in range(commands.shape[0]):
            vrz, response_nominal_z = self._velocity_response_step(
                commands[step_id, :, 0],
                response_nominal_z,
            )
            pb, vb, theta, ab, omega = self._integrate_one_step(pb, vb, theta, vrz)
            alpha = (omega - omega_prev) / self.step_dt
            arz = (vrz - vrz_prev) / self.step_dt
            drz = drz + vrz * self.step_dt
            omega_prev = omega
            vrz_prev = vrz
            new_error = pb - reference_positions[:, step_id + 1]
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
        predicted[:, ObservationIndex.PG] = reference_positions[:, -1]
        predicted[:, ObservationIndex.A_PREV] = observation[:, ObservationIndex.A_PREV]

        self.predicted_observation.copy_(predicted)
        self.predicted_error.copy_(error_curr.unsqueeze(-1))
        self.predicted_error_dot.copy_(predicted_error_dot.unsqueeze(-1))
        self.predicted_error_ddot.copy_(predicted_error_ddot.unsqueeze(-1))
        return predicted

    def get_error_prediction(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return predicted error and its horizon-local finite differences."""
        return self.predicted_error, self.predicted_error_dot, self.predicted_error_ddot

    def get_pending_commands(self) -> torch.Tensor:
        """Return a read-only ordered copy of commands awaiting execution."""
        return self._pending_commands()

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
            "policy_predictor_reference_preview_used": self.reference_preview_used,
            "policy_predictor_reference_horizon": self.reference_horizon,
            "policy_predictor_velocity_response_enabled": torch.full(
                (self.num_envs,),
                float(self.active and self.velocity_response_enabled),
                device=self.device,
            ),
            "policy_predictor_velocity_response_tau_s": torch.full(
                (self.num_envs,),
                self.velocity_response_tau_s,
                device=self.device,
            ),
            "policy_predictor_velocity_response_gain": torch.full(
                (self.num_envs,),
                self.velocity_response_gain,
                device=self.device,
            ),
            "policy_predictor_velocity_response_bias": torch.full(
                (self.num_envs,),
                self.velocity_response_bias,
                device=self.device,
            ),
            "policy_predictor_velocity_response_max_abs_velocity": torch.full(
                (self.num_envs,),
                self.velocity_response_max_abs_velocity,
                device=self.device,
            ),
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

    def _prepare_reference_positions(
        self,
        reference_positions: torch.Tensor | None,
        current_pg: torch.Tensor,
    ) -> tuple[torch.Tensor, bool]:
        if reference_positions is None:
            return current_pg.expand(-1, self.delay_step + 1), False

        reference_positions = reference_positions.to(device=self.device, dtype=torch.float32)
        expected_shape = (self.num_envs, self.delay_step + 1)
        if reference_positions.shape != expected_shape:
            raise ValueError(
                "VelocityModelStatePredictor expects reference_positions shape "
                f"{expected_shape}, got {tuple(reference_positions.shape)}."
            )
        if not torch.all(torch.isfinite(reference_positions)):
            raise ValueError("VelocityModelStatePredictor reference_positions must be finite.")

        tolerance = max(abs(self.epsilon), 1.0e-6)
        if not torch.allclose(
            reference_positions[:, :1],
            current_pg,
            rtol=0.0,
            atol=tolerance,
        ):
            max_error = torch.max(torch.abs(reference_positions[:, :1] - current_pg)).item()
            raise ValueError(
                "VelocityModelStatePredictor reference_positions[:, 0] must match the "
                f"observation PG (max error {max_error:.6g}, tolerance {tolerance:.6g})."
            )
        return reference_positions, True

    def _pending_commands(self) -> torch.Tensor:
        if self.delay_step <= 0:
            return self.command_queue
        ordered_ids = (torch.arange(self.delay_step, device=self.device) + self.read_index) % self.delay_step
        return self.command_queue[ordered_ids].clone()

    def _velocity_response_step(
        self,
        input_z: torch.Tensor,
        previous_nominal_z: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply the simulator's deterministic first-order response for one step."""
        if not self.velocity_response_enabled:
            return input_z, input_z

        target_z = self.velocity_response_gain * input_z + self.velocity_response_bias
        if self.velocity_response_tau_s > 0.0:
            nominal_z = self.velocity_response_decay * previous_nominal_z
            nominal_z += (1.0 - self.velocity_response_decay) * target_z
        else:
            nominal_z = target_z

        executed_z = nominal_z
        if self.velocity_response_max_abs_velocity > 0.0:
            executed_z = torch.clamp(
                executed_z,
                min=-self.velocity_response_max_abs_velocity,
                max=self.velocity_response_max_abs_velocity,
            )
        return executed_z, nominal_z

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
        ab = self.ball_mass * (
            pb + self.ball_position_offset - self.plank_length
        ) * omega.square()
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


def _is_auto(value) -> bool:
    return isinstance(value, str) and value.lower() == "auto"
