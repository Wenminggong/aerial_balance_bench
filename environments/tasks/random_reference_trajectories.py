"""Vectorized random reference trajectories shared by tracking tasks."""

from __future__ import annotations

from collections.abc import Sequence
import math
import operator
from typing import Any

import torch


TRAJECTORY_TYPE_TO_ID = {
    "sine": 0,
    "triangle": 1,
    "trapezoid": 2,
    "constant": 3,
    "random_b_spline": 4,
    "random_ramp_dwell": 5,
}
PERIODIC_TRAJECTORY_TYPES = ("sine", "triangle", "trapezoid")
RANDOM_TRAJECTORY_TYPES = ("random_b_spline", "random_ramp_dwell")


class RandomReferenceTrajectories:
    """Sample and evaluate finite-duration B-spline and ramp-dwell references."""

    def __init__(
        self,
        cfg: Any,
        num_envs: int,
        device: str | torch.device,
        beam_position_min: float,
        beam_position_max: float,
    ):
        self.cfg = cfg
        self.num_envs = self._validate_positive_integer("num_envs", num_envs)
        self.device = torch.device(device)
        self.beam_position_min = float(beam_position_min)
        self.beam_position_max = float(beam_position_max)
        if not (
            math.isfinite(self.beam_position_min)
            and math.isfinite(self.beam_position_max)
            and self.beam_position_min < self.beam_position_max
        ):
            raise ValueError("Beam position bounds must be finite and strictly increasing.")

        self.b_spline_degree = self._validate_positive_integer(
            "random_b_spline_degree",
            cfg.random_b_spline_degree,
        )
        self.b_spline_num_control_points = self._validate_positive_integer(
            "random_b_spline_num_control_points",
            cfg.random_b_spline_num_control_points,
        )
        if self.b_spline_degree >= self.b_spline_num_control_points:
            raise ValueError(
                "random_b_spline_degree must be smaller than "
                "random_b_spline_num_control_points."
            )
        self.b_spline_duration_s = self._validate_positive_float(
            "random_b_spline_duration_s",
            cfg.random_b_spline_duration_s,
        )
        self.b_spline_position_range = self._validate_position_range(
            "random_b_spline_position_range",
            cfg.random_b_spline_position_range,
        )
        self.b_spline_start_position = self._validate_position(
            "random_b_spline_start_position",
            cfg.random_b_spline_start_position,
        )
        self.b_spline_end_position = self._validate_position(
            "random_b_spline_end_position",
            cfg.random_b_spline_end_position,
        )

        self.ramp_dwell_num_segments = self._validate_positive_integer(
            "random_ramp_dwell_num_segments",
            cfg.random_ramp_dwell_num_segments,
        )
        self.ramp_dwell_duration_s = self._validate_positive_float(
            "random_ramp_dwell_duration_s",
            cfg.random_ramp_dwell_duration_s,
        )
        self.ramp_dwell_start_position = self._validate_position(
            "random_ramp_dwell_start_position",
            cfg.random_ramp_dwell_start_position,
        )
        self.ramp_dwell_target_ranges = self._validate_target_ranges(
            cfg.random_ramp_dwell_target_ranges
        )
        self.ramp_duration_range = self._validate_duration_range(
            "random_ramp_duration_range",
            cfg.random_ramp_duration_range,
            strictly_positive=True,
        )
        self.dwell_duration_range = self._validate_duration_range(
            "random_dwell_duration_range",
            cfg.random_dwell_duration_range,
            strictly_positive=False,
        )

        self.b_spline_control_positions = torch.full(
            (self.num_envs, self.b_spline_num_control_points),
            self.b_spline_start_position,
            dtype=torch.float32,
            device=self.device,
        )
        self.b_spline_control_positions[:, -1] = self.b_spline_end_position
        self._b_spline_knots = self._build_open_uniform_knots()

        ramp_shape = (self.num_envs, self.ramp_dwell_num_segments)
        self.ramp_start_positions = torch.full(
            ramp_shape,
            self.ramp_dwell_start_position,
            dtype=torch.float32,
            device=self.device,
        )
        self.ramp_end_positions = self.ramp_start_positions.clone()
        self.ramp_start_times = torch.zeros(ramp_shape, dtype=torch.float32, device=self.device)
        self.ramp_end_times = torch.zeros_like(self.ramp_start_times)
        self.dwell_end_times = torch.zeros_like(self.ramp_start_times)
        self.ramp_dwell_final_position = torch.full(
            (self.num_envs,),
            self.ramp_dwell_start_position,
            dtype=torch.float32,
            device=self.device,
        )

    def sample_reset(
        self,
        env_ids: Sequence[int] | torch.Tensor,
        trajectory_type_id: torch.Tensor,
    ):
        """Sample only the random-reference buffers selected by ``env_ids``."""
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        if env_ids.ndim != 1:
            raise ValueError("env_ids must be one-dimensional.")
        if env_ids.numel() == 0:
            return
        if torch.any((env_ids < 0) | (env_ids >= self.num_envs)):
            raise IndexError("env_ids contains an out-of-range environment index.")

        type_id = torch.as_tensor(trajectory_type_id, dtype=torch.long, device=self.device)
        if type_id.shape == (self.num_envs,):
            type_id = type_id[env_ids]
        elif type_id.shape != (env_ids.numel(),):
            raise ValueError(
                "trajectory_type_id must have shape "
                f"({self.num_envs},) or ({env_ids.numel()},)."
            )

        b_spline_ids = env_ids[type_id == TRAJECTORY_TYPE_TO_ID["random_b_spline"]]
        if b_spline_ids.numel() > 0:
            self._sample_b_spline(b_spline_ids)

        ramp_dwell_ids = env_ids[type_id == TRAJECTORY_TYPE_TO_ID["random_ramp_dwell"]]
        if ramp_dwell_ids.numel() > 0:
            self._sample_ramp_dwell(ramp_dwell_ids)

    def get_position(self, t: torch.Tensor, trajectory_type_id: torch.Tensor) -> torch.Tensor:
        """Return random-reference positions, with zero for non-random types."""
        t = torch.as_tensor(t, dtype=torch.float32, device=self.device)
        if t.ndim == 0 or t.shape[0] != self.num_envs:
            raise ValueError(f"t must have first dimension {self.num_envs}.")
        type_id = torch.as_tensor(trajectory_type_id, dtype=torch.long, device=self.device)
        if type_id.shape != (self.num_envs,):
            raise ValueError(f"trajectory_type_id must have shape ({self.num_envs},).")

        parameter_shape = (self.num_envs,) + (1,) * (t.ndim - 1)
        expanded_type_id = type_id.reshape(parameter_shape)
        output = torch.zeros_like(t)

        b_spline_mask = expanded_type_id == TRAJECTORY_TYPE_TO_ID["random_b_spline"]
        if torch.any(b_spline_mask):
            output = torch.where(b_spline_mask, self._evaluate_b_spline(t), output)

        ramp_dwell_mask = expanded_type_id == TRAJECTORY_TYPE_TO_ID["random_ramp_dwell"]
        if torch.any(ramp_dwell_mask):
            output = torch.where(ramp_dwell_mask, self._evaluate_ramp_dwell(t), output)
        return output

    def get_config_info(self) -> dict[str, object]:
        """Return JSON-serializable random-reference configuration metadata."""
        return {
            "random_b_spline": {
                "degree": self.b_spline_degree,
                "num_control_points": self.b_spline_num_control_points,
                "duration_s": self.b_spline_duration_s,
                "position_range": list(self.b_spline_position_range),
                "start_position": self.b_spline_start_position,
                "end_position": self.b_spline_end_position,
            },
            "random_ramp_dwell": {
                "num_segments": self.ramp_dwell_num_segments,
                "duration_s": self.ramp_dwell_duration_s,
                "start_position": self.ramp_dwell_start_position,
                "target_ranges": [list(value_range) for value_range in self.ramp_dwell_target_ranges],
                "ramp_duration_range": list(self.ramp_duration_range),
                "dwell_duration_range": list(self.dwell_duration_range),
            },
        }

    def _sample_b_spline(self, env_ids: torch.Tensor):
        count = env_ids.numel()
        control_positions = torch.empty(
            (count, self.b_spline_num_control_points),
            dtype=torch.float32,
            device=self.device,
        )
        control_positions[:, 0] = self.b_spline_start_position
        control_positions[:, -1] = self.b_spline_end_position
        if self.b_spline_num_control_points > 2:
            low, high = self.b_spline_position_range
            control_positions[:, 1:-1] = (
                torch.rand(
                    (count, self.b_spline_num_control_points - 2),
                    dtype=torch.float32,
                    device=self.device,
                )
                * (high - low)
                + low
            )
        self.b_spline_control_positions[env_ids] = control_positions

    def _sample_ramp_dwell(self, env_ids: torch.Tensor):
        count = env_ids.numel()
        num_segments = self.ramp_dwell_num_segments
        target_ranges = torch.tensor(
            self.ramp_dwell_target_ranges,
            dtype=torch.float32,
            device=self.device,
        )
        range_ids = torch.randint(
            0,
            target_ranges.shape[0],
            (count, num_segments),
            device=self.device,
        )
        selected_ranges = target_ranges[range_ids]
        target_positions = selected_ranges[..., 0] + torch.rand(
            (count, num_segments),
            dtype=torch.float32,
            device=self.device,
        ) * (selected_ranges[..., 1] - selected_ranges[..., 0])
        requested_ramp_durations = self._sample_matrix_range(
            self.ramp_duration_range,
            count,
            num_segments,
        )
        requested_dwell_durations = self._sample_matrix_range(
            self.dwell_duration_range,
            count,
            num_segments,
        )

        current_time = torch.zeros(count, dtype=torch.float32, device=self.device)
        current_position = torch.full(
            (count,),
            self.ramp_dwell_start_position,
            dtype=torch.float32,
            device=self.device,
        )
        for segment in range(num_segments):
            remaining_time = torch.clamp(self.ramp_dwell_duration_s - current_time, min=0.0)
            requested_ramp = requested_ramp_durations[:, segment]
            actual_ramp = torch.minimum(requested_ramp, remaining_time)
            fraction = actual_ramp / requested_ramp
            end_position = current_position + fraction * (
                target_positions[:, segment] - current_position
            )

            self.ramp_start_times[env_ids, segment] = current_time
            self.ramp_start_positions[env_ids, segment] = current_position
            current_time = current_time + actual_ramp
            self.ramp_end_times[env_ids, segment] = current_time
            self.ramp_end_positions[env_ids, segment] = end_position

            remaining_time = torch.clamp(self.ramp_dwell_duration_s - current_time, min=0.0)
            actual_dwell = torch.minimum(requested_dwell_durations[:, segment], remaining_time)
            current_time = current_time + actual_dwell
            self.dwell_end_times[env_ids, segment] = current_time
            current_position = end_position

        self.ramp_dwell_final_position[env_ids] = current_position

    def _evaluate_b_spline(self, t: torch.Tensor) -> torch.Tensor:
        duration = self.b_spline_duration_s
        clamped_t = torch.clamp(t, min=0.0, max=duration)
        endpoint_mask = clamped_t >= duration
        evaluation_t = torch.where(
            endpoint_mask,
            torch.full_like(clamped_t, duration - self._endpoint_epsilon(duration)),
            clamped_t,
        )

        knots = self._b_spline_knots
        basis = (
            (evaluation_t.unsqueeze(-1) >= knots[:-1])
            & (evaluation_t.unsqueeze(-1) < knots[1:])
        ).to(dtype=torch.float32)
        for degree in range(1, self.b_spline_degree + 1):
            next_basis = torch.zeros(
                (*evaluation_t.shape, basis.shape[-1] - 1),
                dtype=torch.float32,
                device=self.device,
            )
            for index in range(next_basis.shape[-1]):
                left_denominator = knots[index + degree] - knots[index]
                if left_denominator > 0.0:
                    next_basis[..., index] += (
                        (evaluation_t - knots[index]) / left_denominator
                    ) * basis[..., index]
                right_denominator = knots[index + degree + 1] - knots[index + 1]
                if right_denominator > 0.0:
                    next_basis[..., index] += (
                        (knots[index + degree + 1] - evaluation_t) / right_denominator
                    ) * basis[..., index + 1]
            basis = next_basis

        control_shape = (self.num_envs,) + (1,) * (t.ndim - 1) + (
            self.b_spline_num_control_points,
        )
        positions = (basis * self.b_spline_control_positions.reshape(control_shape)).sum(dim=-1)
        end_position_shape = (self.num_envs,) + (1,) * (t.ndim - 1)
        end_position = self.b_spline_control_positions[:, -1].reshape(end_position_shape)
        return torch.where(endpoint_mask, end_position, positions)

    def _evaluate_ramp_dwell(self, t: torch.Tensor) -> torch.Tensor:
        clamped_t = torch.clamp(t, min=0.0, max=self.ramp_dwell_duration_s)
        parameter_shape = (self.num_envs,) + (1,) * (t.ndim - 1)
        segment_shape = (self.num_envs,) + (1,) * (t.ndim - 1) + (
            self.ramp_dwell_num_segments,
        )
        expanded_t = clamped_t.unsqueeze(-1)
        start_times = self.ramp_start_times.reshape(segment_shape)
        ramp_end_times = self.ramp_end_times.reshape(segment_shape)
        dwell_end_times = self.dwell_end_times.reshape(segment_shape)
        start_positions = self.ramp_start_positions.reshape(segment_shape)
        end_positions = self.ramp_end_positions.reshape(segment_shape)

        output = self.ramp_dwell_final_position.reshape(parameter_shape).expand_as(clamped_t).clone()
        for segment in range(self.ramp_dwell_num_segments - 1, -1, -1):
            ramp_duration = ramp_end_times[..., segment] - start_times[..., segment]
            ramp_mask = (
                (expanded_t[..., 0] >= start_times[..., segment])
                & (expanded_t[..., 0] < ramp_end_times[..., segment])
                & (ramp_duration > 0.0)
            )
            ramp_fraction = torch.where(
                ramp_duration > 0.0,
                (expanded_t[..., 0] - start_times[..., segment])
                / torch.clamp(ramp_duration, min=torch.finfo(torch.float32).eps),
                torch.zeros_like(ramp_duration),
            )
            ramp_position = start_positions[..., segment] + ramp_fraction * (
                end_positions[..., segment] - start_positions[..., segment]
            )
            output = torch.where(ramp_mask, ramp_position, output)

            dwell_mask = (
                (expanded_t[..., 0] >= ramp_end_times[..., segment])
                & (expanded_t[..., 0] < dwell_end_times[..., segment])
            )
            output = torch.where(dwell_mask, end_positions[..., segment], output)
        return output

    def _build_open_uniform_knots(self) -> torch.Tensor:
        degree = self.b_spline_degree
        num_control_points = self.b_spline_num_control_points
        interior_count = num_control_points - degree - 1
        knots = [0.0] * (degree + 1)
        if interior_count > 0:
            knots.extend(
                self.b_spline_duration_s * index / (interior_count + 1)
                for index in range(1, interior_count + 1)
            )
        knots.extend([self.b_spline_duration_s] * (degree + 1))
        return torch.tensor(knots, dtype=torch.float32, device=self.device)

    def _sample_matrix_range(
        self,
        value_range: tuple[float, float],
        rows: int,
        columns: int,
    ) -> torch.Tensor:
        low, high = value_range
        return (
            torch.rand((rows, columns), dtype=torch.float32, device=self.device)
            * (high - low)
            + low
        )

    def _validate_position_range(
        self,
        name: str,
        value: Sequence[float],
    ) -> tuple[float, float]:
        low, high = self._validate_range(name, value)
        if low < self.beam_position_min or high > self.beam_position_max:
            raise ValueError(f"{name} must lie within the beam position bounds.")
        return low, high

    def _validate_position(self, name: str, value: float) -> float:
        value = float(value)
        if not math.isfinite(value) or not (
            self.beam_position_min <= value <= self.beam_position_max
        ):
            raise ValueError(f"{name} must lie within the beam position bounds.")
        return value

    def _validate_target_ranges(
        self,
        value: Sequence[Sequence[float]],
    ) -> tuple[tuple[float, float], ...]:
        if isinstance(value, (str, bytes)):
            raise ValueError("random_ramp_dwell_target_ranges must be a sequence of ranges.")
        try:
            ranges = tuple(value)
        except TypeError as exc:
            raise ValueError(
                "random_ramp_dwell_target_ranges must be a sequence of ranges."
            ) from exc
        if not ranges:
            raise ValueError("random_ramp_dwell_target_ranges must not be empty.")
        return tuple(
            self._validate_position_range(
                f"random_ramp_dwell_target_ranges[{index}]",
                value_range,
            )
            for index, value_range in enumerate(ranges)
        )

    @classmethod
    def _validate_duration_range(
        cls,
        name: str,
        value: Sequence[float],
        *,
        strictly_positive: bool,
    ) -> tuple[float, float]:
        low, high = cls._validate_range(name, value)
        if strictly_positive and low <= 0.0:
            raise ValueError(f"{name} must be strictly positive.")
        if not strictly_positive and low < 0.0:
            raise ValueError(f"{name} must be non-negative.")
        return low, high

    @staticmethod
    def _validate_positive_integer(name: str, value: int) -> int:
        if isinstance(value, bool):
            raise ValueError(f"{name} must be a positive integer.")
        try:
            value = operator.index(value)
        except TypeError as exc:
            raise ValueError(f"{name} must be a positive integer.") from exc
        if value <= 0:
            raise ValueError(f"{name} must be a positive integer.")
        return value

    @staticmethod
    def _validate_positive_float(name: str, value: float) -> float:
        value = float(value)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive.")
        return value

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
    def _endpoint_epsilon(duration: float) -> float:
        return max(1.0, duration) * torch.finfo(torch.float32).eps * 8.0
