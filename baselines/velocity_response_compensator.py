"""Deterministic first-order velocity-response inverse compensation."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch


@dataclass
class NFFBVelocityResponseCompensationCfg:
    """Configuration for NFFB's optional velocity-response inverse."""

    enabled: bool = False
    parameter_source: str = "state_predictor"
    tau_s: float | str = 0.0
    gain: float | str = 1.0
    bias: float | str = 0.0
    max_abs_velocity: float | str = 0.0

    @classmethod
    def from_dict(
        cls,
        data: Mapping | None,
    ) -> "NFFBVelocityResponseCompensationCfg":
        """Build a compensation configuration from a YAML dictionary."""
        cfg = cls()
        if not data:
            return cfg
        unknown = sorted(key for key in data if not hasattr(cfg, key))
        if unknown:
            raise ValueError(
                "Unknown NFFB velocity-response compensation field(s): "
                f"{', '.join(unknown)}"
            )
        for key, value in data.items():
            setattr(cfg, key, value)
        return cfg

    @property
    def normalized_parameter_source(self) -> str:
        """Return and validate the configured parameter source."""
        source = str(self.parameter_source).strip().lower().replace("-", "_")
        if source not in {"state_predictor", "explicit"}:
            raise ValueError(
                "velocity_response_compensation.parameter_source must be "
                "'state_predictor' or 'explicit'."
            )
        return source

    def resolve_from_robustness(self, robustness_cfg) -> None:
        """Resolve explicit ``auto`` parameters from fixed robustness ranges."""
        if self.normalized_parameter_source != "explicit":
            return

        range_fields = (
            ("tau_s", "velocity_response_tau_s_range"),
            ("gain", "velocity_response_gain_range"),
            ("bias", "velocity_response_bias_range"),
        )
        for compensation_field, robustness_field in range_fields:
            if not _is_auto(getattr(self, compensation_field)):
                continue
            bounds = getattr(robustness_cfg, robustness_field)
            if len(bounds) != 2:
                raise ValueError(
                    f"robustness.{robustness_field} must contain exactly two values."
                )
            lower, upper = (float(value) for value in bounds)
            if lower != upper:
                raise ValueError(
                    "NFFB velocity_response_compensation."
                    f"{compensation_field}='auto' requires robustness."
                    f"{robustness_field} to have equal bounds; got [{lower}, {upper}]. "
                    "Set an explicit nominal compensation value for randomized "
                    "response parameters."
                )
            setattr(self, compensation_field, lower)

        if _is_auto(self.max_abs_velocity):
            self.max_abs_velocity = float(
                robustness_cfg.velocity_response_max_abs_velocity
            )


class FirstOrderVelocityResponseCompensator:
    """Vectorized exact inverse of the simulator's nominal response model."""

    PARAMETER_SOURCE_IDS = {"explicit": 0, "state_predictor": 1}

    def __init__(
        self,
        cfg: NFFBVelocityResponseCompensationCfg,
        num_envs: int,
        device: str | torch.device,
        step_dt: float,
        *,
        epsilon: float = 1.0e-6,
    ):
        self.cfg = cfg
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.enabled = bool(cfg.enabled)
        self.parameter_source = cfg.normalized_parameter_source
        self.step_dt = float(step_dt)
        self.tau_s = self._as_float(cfg.tau_s, "tau_s")
        self.gain = self._as_float(cfg.gain, "gain")
        self.bias = self._as_float(cfg.bias, "bias")
        self.max_abs_velocity = self._as_float(
            cfg.max_abs_velocity,
            "max_abs_velocity",
        )
        self.epsilon = max(abs(float(epsilon)), 1.0e-7)

        if not math.isfinite(self.step_dt) or self.step_dt <= 0.0:
            raise ValueError(
                "FirstOrderVelocityResponseCompensator requires finite step_dt > 0."
            )
        if not math.isfinite(self.tau_s) or self.tau_s < 0.0:
            raise ValueError(
                "velocity_response_compensation.tau_s must be finite and non-negative."
            )
        if not math.isfinite(self.gain) or self.gain <= 0.0:
            raise ValueError(
                "velocity_response_compensation.gain must be finite and positive."
            )
        if not math.isfinite(self.bias):
            raise ValueError(
                "velocity_response_compensation.bias must be finite."
            )
        if (
            not math.isfinite(self.max_abs_velocity)
            or self.max_abs_velocity < 0.0
        ):
            raise ValueError(
                "velocity_response_compensation.max_abs_velocity must be finite "
                "and non-negative."
            )

        decay = math.exp(-self.step_dt / self.tau_s) if self.tau_s > 0.0 else 0.0
        response_fraction = 1.0 - decay
        if self.enabled and self.tau_s > 0.0 and response_fraction <= self.epsilon:
            raise ValueError(
                "velocity_response_compensation produces a near-zero one-step "
                "response fraction; reduce tau_s or increase step_dt."
            )
        self.decay = torch.tensor(
            decay,
            dtype=torch.float32,
            device=self.device,
        )
        self.response_fraction = torch.tensor(
            response_fraction,
            dtype=torch.float32,
            device=self.device,
        )

        shape = (self.num_envs, 1)
        float_buffers = (
            "nominal_z",
            "execution_nominal_z",
            "desired_output_z",
            "limited_output_z",
            "raw_input_z",
            "limited_input_z",
            "predicted_nominal_z",
            "predicted_output_z",
            "tracking_error_z",
        )
        for name in float_buffers:
            setattr(
                self,
                name,
                torch.zeros(shape, dtype=torch.float32, device=self.device),
            )
        bool_buffers = (
            "output_saturated",
            "predicted_output_clipped",
            "input_velocity_saturated",
            "input_acceleration_saturated",
        )
        for name in bool_buffers:
            setattr(
                self,
                name,
                torch.zeros(shape, dtype=torch.bool, device=self.device),
            )

    def reset(
        self,
        env_ids: Sequence[int] | torch.Tensor | None = None,
        initial_z: torch.Tensor | None = None,
    ) -> None:
        """Reset all or selected nominal response states."""
        env_ids = self._env_ids_tensor(env_ids)
        if env_ids.numel() == 0:
            return
        if initial_z is None:
            initial = torch.zeros(
                (env_ids.numel(), 1),
                dtype=torch.float32,
                device=self.device,
            )
        else:
            initial = torch.as_tensor(
                initial_z,
                dtype=torch.float32,
                device=self.device,
            )
            if initial.shape == (self.num_envs, 1):
                initial = initial[env_ids]
            elif initial.shape != (env_ids.numel(), 1):
                raise ValueError(
                    "Velocity-response compensation initial_z must have shape "
                    f"({env_ids.numel()}, 1) or ({self.num_envs}, 1), got "
                    f"{tuple(initial.shape)}."
                )
            self._require_finite(initial, "initial_z")

        for name in (
            "nominal_z",
            "execution_nominal_z",
            "desired_output_z",
            "limited_output_z",
            "raw_input_z",
            "limited_input_z",
            "predicted_nominal_z",
            "predicted_output_z",
            "tracking_error_z",
        ):
            getattr(self, name)[env_ids] = 0.0
        self.nominal_z[env_ids] = initial
        self.execution_nominal_z[env_ids] = initial
        for name in (
            "output_saturated",
            "predicted_output_clipped",
            "input_velocity_saturated",
            "input_acceleration_saturated",
        ):
            getattr(self, name)[env_ids] = False

    def compensate(
        self,
        desired_output_z: torch.Tensor,
        current_input_z: torch.Tensor,
        *,
        pending_commands: torch.Tensor | None,
        max_input_change: float,
        max_abs_input: float,
    ) -> torch.Tensor:
        """Return the reachable absolute command for a desired response output."""
        if not self.enabled:
            raise RuntimeError(
                "FirstOrderVelocityResponseCompensator.compensate() requires "
                "enabled=true."
            )
        desired_output_z = self._full_values(desired_output_z, "desired_output_z")
        current_input_z = self._full_values(current_input_z, "current_input_z")
        max_input_change = float(max_input_change)
        max_abs_input = float(max_abs_input)
        if not math.isfinite(max_input_change) or max_input_change <= 0.0:
            raise ValueError("max_input_change must be finite and positive.")
        if not math.isfinite(max_abs_input) or max_abs_input < 0.0:
            raise ValueError("max_abs_input must be finite and non-negative.")

        self.desired_output_z.copy_(desired_output_z)
        if self.max_abs_velocity > 0.0:
            self.limited_output_z.copy_(
                torch.clamp(
                    desired_output_z,
                    min=-self.max_abs_velocity,
                    max=self.max_abs_velocity,
                )
            )
        else:
            self.limited_output_z.copy_(desired_output_z)
        self.output_saturated.copy_(
            torch.abs(self.desired_output_z - self.limited_output_z)
            > self.epsilon
        )

        execution_nominal = self.forecast_execution_state(pending_commands)
        target_z = self.limited_output_z
        if self.tau_s > 0.0:
            raw_input = (
                (target_z - self.decay * execution_nominal)
                / self.response_fraction
                - self.bias
            ) / self.gain
        else:
            raw_input = (target_z - self.bias) / self.gain
        self._require_finite(raw_input, "inverse input")
        self.raw_input_z.copy_(raw_input)

        if max_abs_input > 0.0:
            absolute_lower = -max_abs_input
            absolute_upper = max_abs_input
            self.input_velocity_saturated.copy_(
                (raw_input < absolute_lower - self.epsilon)
                | (raw_input > absolute_upper + self.epsilon)
            )
        else:
            absolute_lower = -math.inf
            absolute_upper = math.inf
            self.input_velocity_saturated.zero_()

        rate_lower = current_input_z - max_input_change
        rate_upper = current_input_z + max_input_change
        self.input_acceleration_saturated.copy_(
            (raw_input < rate_lower - self.epsilon)
            | (raw_input > rate_upper + self.epsilon)
        )
        feasible_lower = torch.clamp(rate_lower, min=absolute_lower)
        feasible_upper = torch.clamp(rate_upper, max=absolute_upper)
        if bool(torch.any(feasible_lower > feasible_upper)):
            raise RuntimeError(
                "Velocity-response compensation input constraints have an empty "
                "reachable interval."
            )
        limited_input = torch.maximum(
            torch.minimum(raw_input, feasible_upper),
            feasible_lower,
        )
        self.limited_input_z.copy_(limited_input)

        predicted_nominal = self._nominal_step(limited_input, execution_nominal)
        self.predicted_nominal_z.copy_(predicted_nominal)
        predicted_output = self._clip_output(predicted_nominal)
        self.predicted_output_z.copy_(predicted_output)
        self.predicted_output_clipped.copy_(
            torch.abs(predicted_nominal - predicted_output) > self.epsilon
        )
        self.tracking_error_z.copy_(predicted_output - self.limited_output_z)
        return self.limited_input_z

    def forecast_execution_state(
        self,
        pending_commands: torch.Tensor | None,
    ) -> torch.Tensor:
        """Forecast the nominal state immediately before the new command executes."""
        if pending_commands is None:
            pending = torch.empty(
                (0, self.num_envs, 1),
                dtype=torch.float32,
                device=self.device,
            )
        else:
            pending = torch.as_tensor(
                pending_commands,
                dtype=torch.float32,
                device=self.device,
            )
            if pending.ndim == 2 and pending.shape[1] == self.num_envs:
                pending = pending.unsqueeze(-1)
            if pending.ndim != 3 or pending.shape[1:] != (self.num_envs, 1):
                raise ValueError(
                    "Velocity-response pending_commands must have shape "
                    f"(D, {self.num_envs}, 1), got {tuple(pending.shape)}."
                )
            self._require_finite(pending, "pending_commands")

        nominal = self.nominal_z.clone()
        for step_id in range(pending.shape[0]):
            nominal = self._nominal_step(pending[step_id], nominal)
        self.execution_nominal_z.copy_(nominal)
        return self.execution_nominal_z

    def advance(self, executed_input_z: torch.Tensor) -> torch.Tensor:
        """Advance the persistent nominal state by one actually executed input."""
        executed_input_z = self._full_values(
            executed_input_z,
            "executed_input_z",
        )
        self.nominal_z.copy_(self._nominal_step(executed_input_z, self.nominal_z))
        return self.nominal_z

    def get_state(self) -> dict[str, torch.Tensor]:
        """Return response-compensation diagnostics for rollout logging."""
        prefix = "policy_velocity_response_compensation"
        state = {
            f"{prefix}_enabled": torch.full(
                (self.num_envs,),
                float(self.enabled),
                dtype=torch.float32,
                device=self.device,
            ),
            f"{prefix}_parameter_source_id": torch.full(
                (self.num_envs,),
                float(self.PARAMETER_SOURCE_IDS[self.parameter_source]),
                dtype=torch.float32,
                device=self.device,
            ),
            f"{prefix}_tau_s": torch.full(
                (self.num_envs,),
                self.tau_s,
                dtype=torch.float32,
                device=self.device,
            ),
            f"{prefix}_gain": torch.full(
                (self.num_envs,),
                self.gain,
                dtype=torch.float32,
                device=self.device,
            ),
            f"{prefix}_bias": torch.full(
                (self.num_envs,),
                self.bias,
                dtype=torch.float32,
                device=self.device,
            ),
            f"{prefix}_max_abs_velocity": torch.full(
                (self.num_envs,),
                self.max_abs_velocity,
                dtype=torch.float32,
                device=self.device,
            ),
        }
        names = (
            "nominal_z",
            "execution_nominal_z",
            "desired_output_z",
            "limited_output_z",
            "raw_input_z",
            "limited_input_z",
            "predicted_nominal_z",
            "predicted_output_z",
            "tracking_error_z",
            "output_saturated",
            "predicted_output_clipped",
            "input_velocity_saturated",
            "input_acceleration_saturated",
        )
        state.update(
            {
                f"{prefix}_{name}": getattr(self, name)[:, 0]
                for name in names
            }
        )
        return state

    def to(self, device: str | torch.device):
        """Move compensation buffers to a device and return self."""
        device = torch.device(device)
        for name, value in vars(self).items():
            if isinstance(value, torch.Tensor):
                setattr(self, name, value.to(device=device))
        self.device = device
        return self

    def _nominal_step(
        self,
        input_z: torch.Tensor,
        previous_nominal_z: torch.Tensor,
    ) -> torch.Tensor:
        target_z = self.gain * input_z + self.bias
        if self.tau_s <= 0.0:
            return target_z
        return (
            self.decay * previous_nominal_z
            + self.response_fraction * target_z
        )

    def _clip_output(self, nominal_z: torch.Tensor) -> torch.Tensor:
        if self.max_abs_velocity <= 0.0:
            return nominal_z
        return torch.clamp(
            nominal_z,
            min=-self.max_abs_velocity,
            max=self.max_abs_velocity,
        )

    def _full_values(self, values: torch.Tensor, field_name: str) -> torch.Tensor:
        values = torch.as_tensor(
            values,
            dtype=torch.float32,
            device=self.device,
        )
        if values.shape != (self.num_envs, 1):
            raise ValueError(
                f"{field_name} must have shape ({self.num_envs}, 1), got "
                f"{tuple(values.shape)}."
            )
        self._require_finite(values, field_name)
        return values

    def _env_ids_tensor(
        self,
        env_ids: Sequence[int] | torch.Tensor | None,
    ) -> torch.Tensor:
        if env_ids is None:
            return torch.arange(self.num_envs, device=self.device)
        return torch.as_tensor(env_ids, dtype=torch.long, device=self.device)

    @staticmethod
    def _require_finite(values: torch.Tensor, field_name: str) -> None:
        if not bool(torch.isfinite(values).all()):
            raise ValueError(
                f"Velocity-response compensation {field_name} must be finite."
            )

    @staticmethod
    def _as_float(value: float | str, field_name: str) -> float:
        if _is_auto(value):
            raise ValueError(
                "NFFB velocity_response_compensation."
                f"{field_name} must be resolved before use."
            )
        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "NFFB velocity_response_compensation."
                f"{field_name} must be numeric."
            ) from exc


def _is_auto(value) -> bool:
    return isinstance(value, str) and value.lower() == "auto"
