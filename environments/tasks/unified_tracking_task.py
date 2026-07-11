"""Unified set-point and trajectory-tracking task."""

from __future__ import annotations

from collections.abc import Sequence
import math
import operator

import torch
from omni.isaac.lab.utils import configclass


TRAJECTORY_TYPE_TO_ID = {
    "sine": 0,
    "triangle": 1,
    "trapezoid": 2,
    "constant": 3,
}
ID_TO_TRAJECTORY_TYPE = {value: key for key, value in TRAJECTORY_TYPE_TO_ID.items()}
INITIALIZATION_MODES = {"fixed", "on_reference", "independent_uniform"}


@configclass
class UnifiedTrackingTaskCfg:
    """Configuration for mixed set-point and dynamic-reference tracking."""

    trajectory_types: tuple[str, ...] = ("constant", "sine", "triangle", "trapezoid")
    trajectory_type_weights: tuple[float, ...] = ()

    constant_goal_range: tuple[float, float] = (0.10, 0.60)
    dynamic_center_range: tuple[float, float] = (0.25, 0.45)
    amplitude_range: tuple[float, float] = (0.05, 0.20)
    period_range: tuple[float, float] = (4.0, 10.0)
    phase_range: tuple[float, float] = (0.0, 2.0 * math.pi)

    constant_initialization_mode: str = "independent_uniform"
    dynamic_initialization_mode: str = "on_reference"
    fixed_initial_ball_position: float = 0.35
    initial_ball_position_range: tuple[float, float] = (0.10, 0.60)
    min_initial_reference_distance: float = 0.05

    position_weight: float = 5.0
    velocity_weight: float = 0.5
    command_weight: float = 0.5
    action_weight: float = 1.0
    failure_penalty: float = 500.0
    progress_weight: float = 1.0
    max_error_for_failure: float = 0.5


class UnifiedTrackingTask:
    """Vectorized task covering constant and dynamic position references."""

    def __init__(
        self,
        cfg: UnifiedTrackingTaskCfg,
        num_envs: int,
        device: str | torch.device,
        step_dt: float,
        beam_position_min: float,
        beam_position_max: float,
    ):
        self.cfg = cfg
        try:
            self.num_envs = operator.index(num_envs)
        except TypeError as exc:
            raise TypeError("num_envs must be an integer.") from exc
        if isinstance(num_envs, bool):
            raise TypeError("num_envs must be an integer, not bool.")
        self.device = torch.device(device)
        self.step_dt = float(step_dt)
        self.beam_position_min = float(beam_position_min)
        self.beam_position_max = float(beam_position_max)

        self._validate_config()

        configured_types = tuple(cfg.trajectory_types)
        configured_weights = tuple(cfg.trajectory_type_weights)
        if configured_weights:
            weight_tensor = torch.tensor(configured_weights, dtype=torch.float32, device=self.device)
        else:
            weight_tensor = torch.ones(len(configured_types), dtype=torch.float32, device=self.device)
        self._configured_trajectory_types = configured_types
        self._configured_trajectory_ids = torch.tensor(
            [TRAJECTORY_TYPE_TO_ID[name] for name in configured_types],
            dtype=torch.long,
            device=self.device,
        )
        self._normalized_type_weights = weight_tensor / weight_tensor.sum()

        self.trajectory_type_id = torch.full(
            (self.num_envs,),
            int(self._configured_trajectory_ids[0].item()),
            dtype=torch.long,
            device=self.device,
        )
        self.center = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.amplitude = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.period = torch.ones(self.num_envs, dtype=torch.float32, device=self.device)
        self.phase = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.initial_ball_position = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.previous_abs_error = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

    def sample_reset(self, env, env_ids: Sequence[int] | torch.Tensor):
        """Sample per-environment references and initialize the ball state."""
        env_ids = self._as_env_ids(env_ids)
        if env_ids.numel() == 0:
            return

        sampled_ids = self._sample_trajectory_types(env_ids.numel())
        self.trajectory_type_id[env_ids] = sampled_ids

        constant_mask = sampled_ids == TRAJECTORY_TYPE_TO_ID["constant"]
        dynamic_mask = ~constant_mask
        count = env_ids.numel()

        sampled_center = self._sample_uniform(*self.cfg.dynamic_center_range, count)
        if torch.any(constant_mask):
            sampled_center[constant_mask] = self._sample_uniform(
                *self.cfg.constant_goal_range,
                int(constant_mask.sum().item()),
            )
        sampled_amplitude = self._sample_uniform(*self.cfg.amplitude_range, count)
        sampled_period = self._sample_uniform(*self.cfg.period_range, count)
        sampled_phase = self._sample_uniform(*self.cfg.phase_range, count)
        sampled_amplitude[constant_mask] = 0.0
        sampled_period[constant_mask] = 1.0
        sampled_phase[constant_mask] = 0.0

        self.center[env_ids] = sampled_center
        self.amplitude[env_ids] = sampled_amplitude
        self.period[env_ids] = sampled_period
        self.phase[env_ids] = sampled_phase

        initial_reference = self.get_reference(torch.zeros(self.num_envs, device=self.device))[0][env_ids]
        initial_position = torch.empty(count, dtype=torch.float32, device=self.device)
        self._sample_initial_positions(
            initial_position,
            constant_mask,
            dynamic_mask,
            initial_reference,
        )

        self.initial_ball_position[env_ids] = initial_position
        self.previous_abs_error[env_ids] = torch.abs(initial_position - initial_reference)
        env.set_ball_position_along_beam(env_ids, initial_position)

    def get_reference(self, step: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Return desired position and forward finite-difference velocity."""
        pg, vg = self.get_reference_preview(step=step, future_steps=0)
        return pg[:, 0], vg[:, 0]

    def get_reference_preview(
        self,
        step: torch.Tensor | None = None,
        future_steps: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return current plus future position/velocity references.

        The returned tensors have shape ``(num_envs, future_steps + 1)``.  Each
        velocity is a one-control-step forward finite difference of position.
        """
        if isinstance(future_steps, bool):
            raise TypeError("future_steps must be an integer, not bool.")
        try:
            future_steps = operator.index(future_steps)
        except TypeError as exc:
            raise TypeError("future_steps must be an integer.") from exc
        if future_steps < 0:
            raise ValueError("future_steps must be non-negative.")

        base_step = self._normalize_step(step)
        offsets = torch.arange(future_steps + 1, dtype=torch.float32, device=self.device)
        preview_step = base_step.unsqueeze(-1) + offsets.unsqueeze(0)
        t = preview_step * self.step_dt
        pg = self._reference_position(t)
        pg_next = self._reference_position(t + self.step_dt)
        vg = (pg_next - pg) / self.step_dt
        return pg, vg

    def compute_reward(
        self,
        state: dict[str, torch.Tensor],
        command: dict[str, torch.Tensor],
        action: torch.Tensor,
        terminated: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the common tracking reward for every reference type."""
        error = state["pb"] - state["pg"]
        velocity_error = state["vb"] - state["vg"]
        command_z = command["command_z"]
        action_z = action.squeeze(-1)
        abs_error = torch.abs(error)

        reward = -self.cfg.position_weight * error.square()
        reward -= self.cfg.velocity_weight * velocity_error.square()
        reward -= self.cfg.command_weight * command_z.square()
        reward -= self.cfg.action_weight * action_z.square()
        reward += self.cfg.progress_weight * (self.previous_abs_error - abs_error)

        if terminated is None:
            failure_mask = (torch.abs(state["theta"]) > state["theta_limit"]) | (
                abs_error > self.cfg.max_error_for_failure
            )
        else:
            failure_mask = terminated.to(device=self.device, dtype=torch.bool)
        reward -= self.cfg.failure_penalty * failure_mask.float()

        self.previous_abs_error.copy_(abs_error.detach())
        return reward

    def compute_task_dones(self, state: dict[str, torch.Tensor]) -> torch.Tensor:
        """Return task-specific tracking-error failure flags."""
        return torch.abs(state["pb"] - state["pg"]) > self.cfg.max_error_for_failure

    def get_task_info(self) -> dict[str, torch.Tensor]:
        """Return per-environment reference parameters for logging."""
        return {
            "trajectory_type_id": self.trajectory_type_id,
            "trajectory_center": self.center,
            "trajectory_amplitude": self.amplitude,
            "trajectory_period": self.period,
            "trajectory_phase": self.phase,
            "initial_ball_position": self.initial_ball_position,
        }

    def get_config_info(self) -> dict[str, object]:
        """Return JSON-serializable sampling metadata."""
        return {
            "trajectory_type_to_id": dict(TRAJECTORY_TYPE_TO_ID),
            "trajectory_types": list(self._configured_trajectory_types),
            "normalized_trajectory_type_weights": self._normalized_type_weights.cpu().tolist(),
        }

    def _reference_position(self, t: torch.Tensor) -> torch.Tensor:
        parameter_shape = (self.num_envs,) + (1,) * (t.ndim - 1)
        center = self.center.reshape(parameter_shape)
        amplitude = self.amplitude.reshape(parameter_shape)
        period = self.period.reshape(parameter_shape)
        phase_offset = self.phase.reshape(parameter_shape)
        type_id = self.trajectory_type_id.reshape(parameter_shape)

        phase = 2.0 * math.pi * t / period + phase_offset
        triangle = self._triangle_wave(phase)
        waveform = torch.zeros_like(phase)
        waveform = torch.where(type_id == TRAJECTORY_TYPE_TO_ID["sine"], torch.sin(phase), waveform)
        waveform = torch.where(type_id == TRAJECTORY_TYPE_TO_ID["triangle"], triangle, waveform)
        waveform = torch.where(
            type_id == TRAJECTORY_TYPE_TO_ID["trapezoid"],
            torch.clamp(2.0 * triangle, -1.0, 1.0),
            waveform,
        )
        return center + amplitude * waveform

    def _sample_initial_positions(
        self,
        output: torch.Tensor,
        constant_mask: torch.Tensor,
        dynamic_mask: torch.Tensor,
        initial_reference: torch.Tensor,
    ):
        for mask, mode in (
            (constant_mask, self.cfg.constant_initialization_mode),
            (dynamic_mask, self.cfg.dynamic_initialization_mode),
        ):
            count = int(mask.sum().item())
            if count == 0:
                continue
            if mode == "fixed":
                output[mask] = self.cfg.fixed_initial_ball_position
            elif mode == "on_reference":
                output[mask] = initial_reference[mask]
            else:
                output[mask] = self._sample_independent_initial_position(initial_reference[mask])

    def _sample_independent_initial_position(self, reference: torch.Tensor) -> torch.Tensor:
        low, high = self.cfg.initial_ball_position_range
        distance = self.cfg.min_initial_reference_distance
        if distance == 0.0:
            return self._sample_uniform(low, high, reference.numel())

        left_high = torch.minimum(torch.full_like(reference, high), reference - distance)
        left_length = torch.clamp(left_high - low, min=0.0)
        right_low = torch.maximum(torch.full_like(reference, low), reference + distance)
        right_length = torch.clamp(high - right_low, min=0.0)
        total_length = left_length + right_length

        sample = torch.empty_like(reference)
        non_degenerate = total_length > 0.0
        if torch.any(non_degenerate):
            unit = torch.rand(int(non_degenerate.sum().item()), device=self.device)
            left_probability = left_length[non_degenerate] / total_length[non_degenerate]
            choose_left = unit < left_probability
            interval_low = torch.where(
                choose_left,
                torch.full_like(unit, low),
                right_low[non_degenerate],
            )
            interval_high = torch.where(
                choose_left,
                left_high[non_degenerate],
                torch.full_like(unit, high),
            )
            sample[non_degenerate] = interval_low + torch.rand_like(unit) * (interval_high - interval_low)

        if torch.any(~non_degenerate):
            degenerate_reference = reference[~non_degenerate]
            low_distance = torch.abs(degenerate_reference - low)
            high_distance = torch.abs(degenerate_reference - high)
            endpoint = torch.where(
                low_distance >= high_distance,
                torch.full_like(degenerate_reference, low),
                torch.full_like(degenerate_reference, high),
            )
            if torch.any(torch.abs(endpoint - degenerate_reference) + 1.0e-7 < distance):
                raise RuntimeError(
                    "No initial ball position satisfies min_initial_reference_distance "
                    "for a sampled reference."
                )
            sample[~non_degenerate] = endpoint
        return sample

    def _sample_trajectory_types(self, size: int) -> torch.Tensor:
        sampled_indices = torch.multinomial(self._normalized_type_weights, size, replacement=True)
        return self._configured_trajectory_ids[sampled_indices]

    def _sample_uniform(self, low: float, high: float, size: int) -> torch.Tensor:
        return torch.rand(size, dtype=torch.float32, device=self.device) * (high - low) + low

    def _normalize_step(self, step: torch.Tensor | None) -> torch.Tensor:
        if step is None:
            return torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        step_tensor = torch.as_tensor(step, dtype=torch.float32, device=self.device)
        if step_tensor.ndim == 0:
            return step_tensor.repeat(self.num_envs)
        if step_tensor.ndim != 1 or step_tensor.shape[0] != self.num_envs:
            raise ValueError(f"step must be scalar or have shape ({self.num_envs},).")
        return step_tensor

    def _as_env_ids(self, env_ids: Sequence[int] | torch.Tensor) -> torch.Tensor:
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        if env_ids.ndim != 1:
            raise ValueError("env_ids must be one-dimensional.")
        if torch.any((env_ids < 0) | (env_ids >= self.num_envs)):
            raise IndexError("env_ids contains an out-of-range environment index.")
        return env_ids

    def _validate_config(self):
        if self.num_envs <= 0:
            raise ValueError("num_envs must be positive.")
        if not math.isfinite(self.step_dt) or self.step_dt <= 0.0:
            raise ValueError("step_dt must be finite and positive.")
        if not (
            math.isfinite(self.beam_position_min)
            and math.isfinite(self.beam_position_max)
            and self.beam_position_min < self.beam_position_max
        ):
            raise ValueError("Beam position bounds must be finite and strictly increasing.")

        try:
            trajectory_types = tuple(self.cfg.trajectory_types)
        except TypeError as exc:
            raise ValueError("trajectory_types must be a sequence of type names.") from exc
        if not trajectory_types:
            raise ValueError("trajectory_types must contain at least one type.")
        if len(set(trajectory_types)) != len(trajectory_types):
            raise ValueError("trajectory_types must not contain duplicates.")
        unsupported = [name for name in trajectory_types if name not in TRAJECTORY_TYPE_TO_ID]
        if unsupported:
            raise ValueError(
                f"Unsupported trajectory_types {unsupported}; expected members of "
                f"{list(TRAJECTORY_TYPE_TO_ID)}."
            )

        try:
            weights = tuple(self.cfg.trajectory_type_weights)
        except TypeError as exc:
            raise ValueError("trajectory_type_weights must be a sequence.") from exc
        if weights and len(weights) != len(trajectory_types):
            raise ValueError("trajectory_type_weights must be empty or match trajectory_types in length.")
        if weights:
            if any(not math.isfinite(float(weight)) or float(weight) < 0.0 for weight in weights):
                raise ValueError("trajectory_type_weights must be finite and non-negative.")
            if sum(float(weight) for weight in weights) <= 0.0:
                raise ValueError("trajectory_type_weights must have a positive sum.")

        ranges = {
            "constant_goal_range": self.cfg.constant_goal_range,
            "dynamic_center_range": self.cfg.dynamic_center_range,
            "amplitude_range": self.cfg.amplitude_range,
            "period_range": self.cfg.period_range,
            "phase_range": self.cfg.phase_range,
            "initial_ball_position_range": self.cfg.initial_ball_position_range,
        }
        validated_ranges = {name: self._validate_range(name, value) for name, value in ranges.items()}
        goal_low, goal_high = validated_ranges["constant_goal_range"]
        center_low, center_high = validated_ranges["dynamic_center_range"]
        amplitude_low, amplitude_high = validated_ranges["amplitude_range"]
        period_low, _ = validated_ranges["period_range"]
        initial_low, initial_high = validated_ranges["initial_ball_position_range"]

        if amplitude_low < 0.0:
            raise ValueError("amplitude_range must be non-negative.")
        if period_low <= 0.0:
            raise ValueError("period_range must be strictly positive.")
        if goal_low < self.beam_position_min or goal_high > self.beam_position_max:
            raise ValueError("constant_goal_range must lie within the beam position bounds.")
        if center_low - amplitude_high < self.beam_position_min or center_high + amplitude_high > self.beam_position_max:
            raise ValueError(
                "Every dynamic reference must lie within the beam position bounds; "
                "adjust dynamic_center_range or amplitude_range."
            )
        if initial_low < self.beam_position_min or initial_high > self.beam_position_max:
            raise ValueError("initial_ball_position_range must lie within the beam position bounds.")
        if not math.isfinite(self.cfg.fixed_initial_ball_position) or not (
            self.beam_position_min <= self.cfg.fixed_initial_ball_position <= self.beam_position_max
        ):
            raise ValueError("fixed_initial_ball_position must lie within the beam position bounds.")

        for name in ("constant_initialization_mode", "dynamic_initialization_mode"):
            mode = getattr(self.cfg, name)
            if mode not in INITIALIZATION_MODES:
                raise ValueError(f"{name} must be one of {sorted(INITIALIZATION_MODES)}.")

        distance = float(self.cfg.min_initial_reference_distance)
        if not math.isfinite(distance) or distance < 0.0:
            raise ValueError("min_initial_reference_distance must be finite and non-negative.")
        if "constant" in trajectory_types and self.cfg.constant_initialization_mode == "independent_uniform":
            self._validate_initial_distance_feasibility(goal_low, goal_high, initial_low, initial_high, distance)
        if any(name != "constant" for name in trajectory_types) and self.cfg.dynamic_initialization_mode == "independent_uniform":
            self._validate_initial_distance_feasibility(
                center_low - amplitude_high,
                center_high + amplitude_high,
                initial_low,
                initial_high,
                distance,
            )

        for name in (
            "position_weight",
            "velocity_weight",
            "command_weight",
            "action_weight",
            "failure_penalty",
            "progress_weight",
            "max_error_for_failure",
        ):
            value = float(getattr(self.cfg, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative.")
        if self.cfg.max_error_for_failure == 0.0:
            raise ValueError("max_error_for_failure must be positive.")

    @staticmethod
    def _validate_range(name: str, value: Sequence[float]) -> tuple[float, float]:
        if isinstance(value, (str, bytes)):
            raise ValueError(f"{name} must contain exactly two numeric values.")
        try:
            values = tuple(value)
        except TypeError as exc:
            raise ValueError(f"{name} must contain exactly two numeric values.") from exc
        if len(values) != 2:
            raise ValueError(f"{name} must contain exactly two values.")
        try:
            low, high = float(values[0]), float(values[1])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must contain numeric bounds.") from exc
        if not math.isfinite(low) or not math.isfinite(high) or low > high:
            raise ValueError(f"{name} must contain finite, non-decreasing bounds.")
        return low, high

    @staticmethod
    def _validate_initial_distance_feasibility(
        reference_low: float,
        reference_high: float,
        initial_low: float,
        initial_high: float,
        distance: float,
    ):
        midpoint = 0.5 * (initial_low + initial_high)
        closest_reference = min(max(midpoint, reference_low), reference_high)
        maximum_available_distance = max(
            abs(closest_reference - initial_low),
            abs(initial_high - closest_reference),
        )
        if distance > maximum_available_distance + 1.0e-7:
            raise ValueError(
                "min_initial_reference_distance is infeasible for the configured "
                "reference and initial-position ranges."
            )

    @staticmethod
    def _triangle_wave(phase: torch.Tensor) -> torch.Tensor:
        shifted_phase = torch.remainder(phase + math.pi / 2.0, 2.0 * math.pi)
        rising = shifted_phase < math.pi
        return torch.where(
            rising,
            2.0 * shifted_phase / math.pi - 1.0,
            3.0 - 2.0 * shifted_phase / math.pi,
        )
