"""Metrics for mixed constant and dynamic reference tracking."""

from __future__ import annotations

from collections.abc import Sequence
import math
import operator

import torch
from omni.isaac.lab.utils import configclass

from ..tasks.unified_tracking_task import TRAJECTORY_TYPE_TO_ID


@configclass
class UnifiedTrackingEvaluatorCfg:
    """Configuration for unified-tracking benchmark evaluation."""

    episode_length_s: float = 10.0
    num_eval_episodes: int = 1000
    target_zone: float = 0.01
    steady_state_window_s: float = 1.0


class UnifiedTrackingEvaluator:
    """Accumulate global, per-reference-type, and constant-reference metrics."""

    def __init__(
        self,
        cfg: UnifiedTrackingEvaluatorCfg,
        num_envs: int,
        device: str | torch.device,
        step_dt: float,
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
        self._validate_config()

        self.num_trajectory_types = len(TRAJECTORY_TYPE_TO_ID)
        self.constant_type_id = TRAJECTORY_TYPE_TO_ID["constant"]
        self.timeout_value = cfg.episode_length_s + self.step_dt
        self.steady_state_window_steps = max(1, int(math.ceil(cfg.steady_state_window_s / self.step_dt)))
        self.env_indices = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        self.reset_all()

    def reset_all(self):
        """Clear aggregate and per-episode metric buffers."""
        self.completed_episodes = torch.zeros((), dtype=torch.float32, device=self.device)
        self.mae_sum = torch.zeros((), dtype=torch.float32, device=self.device)
        self.rmse_sum = torch.zeros((), dtype=torch.float32, device=self.device)
        self.maxe_sum = torch.zeros((), dtype=torch.float32, device=self.device)

        self.type_completed_episodes = torch.zeros(
            self.num_trajectory_types,
            dtype=torch.float32,
            device=self.device,
        )
        self.type_mae_sum = torch.zeros_like(self.type_completed_episodes)
        self.type_rmse_sum = torch.zeros_like(self.type_completed_episodes)
        self.type_maxe_sum = torch.zeros_like(self.type_completed_episodes)

        self.constant_success_episodes = torch.zeros((), dtype=torch.float32, device=self.device)
        self.constant_steady_state_error_sum = torch.zeros((), dtype=torch.float32, device=self.device)
        self.constant_steady_state_error_sum_sq = torch.zeros((), dtype=torch.float32, device=self.device)
        self.constant_convergence_time_sum = torch.zeros((), dtype=torch.float32, device=self.device)
        self.constant_convergence_time_sum_sq = torch.zeros((), dtype=torch.float32, device=self.device)
        self.constant_climbing_time_sum = torch.zeros((), dtype=torch.float32, device=self.device)
        self.constant_climbing_time_sum_sq = torch.zeros((), dtype=torch.float32, device=self.device)

        self.episode_abs_error_sum = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.episode_squared_error_sum = torch.zeros_like(self.episode_abs_error_sum)
        self.episode_max_abs_error = torch.zeros_like(self.episode_abs_error_sum)
        self.episode_samples = torch.zeros_like(self.episode_abs_error_sum)

        self.climbing_time = torch.full(
            (self.num_envs,),
            self.timeout_value,
            dtype=torch.float32,
            device=self.device,
        )
        self.convergence_time = torch.full_like(self.climbing_time, self.timeout_value)
        self.steady_state_error_buffer = torch.zeros(
            (self.num_envs, self.steady_state_window_steps),
            dtype=torch.float32,
            device=self.device,
        )
        self.steady_state_error_count = torch.zeros(
            self.num_envs,
            dtype=torch.long,
            device=self.device,
        )
        self.steady_state_error_index = torch.zeros_like(self.steady_state_error_count)

    def reset_episode(self, env_ids: Sequence[int] | torch.Tensor):
        """Reset per-episode buffers for selected environments."""
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        if env_ids.ndim != 1:
            raise ValueError("env_ids must be one-dimensional.")
        if env_ids.numel() == 0:
            return
        if torch.any((env_ids < 0) | (env_ids >= self.num_envs)):
            raise IndexError("env_ids contains an out-of-range environment index.")

        self.episode_abs_error_sum[env_ids] = 0.0
        self.episode_squared_error_sum[env_ids] = 0.0
        self.episode_max_abs_error[env_ids] = 0.0
        self.episode_samples[env_ids] = 0.0
        self.climbing_time[env_ids] = self.timeout_value
        self.convergence_time[env_ids] = self.timeout_value
        self.steady_state_error_buffer[env_ids] = 0.0
        self.steady_state_error_count[env_ids] = 0
        self.steady_state_error_index[env_ids] = 0

    def update(
        self,
        state: dict[str, torch.Tensor],
        terminated: torch.Tensor,
        time_outs: torch.Tensor,
        episode_length_buf: torch.Tensor,
        trajectory_type_id: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Update per-step statistics and aggregate completed episodes by type."""
        type_id = torch.as_tensor(trajectory_type_id, dtype=torch.long, device=self.device)
        if type_id.shape != (self.num_envs,):
            raise ValueError(f"trajectory_type_id must have shape ({self.num_envs},).")
        if torch.any((type_id < 0) | (type_id >= self.num_trajectory_types)):
            raise ValueError("trajectory_type_id contains an unsupported type ID.")

        terminated = torch.as_tensor(terminated, dtype=torch.bool, device=self.device)
        time_outs = torch.as_tensor(time_outs, dtype=torch.bool, device=self.device)
        episode_length_buf = torch.as_tensor(episode_length_buf, device=self.device)
        for name, value in (("terminated", terminated), ("time_outs", time_outs), ("episode_length_buf", episode_length_buf)):
            if value.shape != (self.num_envs,):
                raise ValueError(f"{name} must have shape ({self.num_envs},).")

        error = torch.abs(state["pb"] - state["pg"])
        if error.shape != (self.num_envs,):
            raise ValueError(f"state errors must have shape ({self.num_envs},).")
        self.episode_abs_error_sum += error
        self.episode_squared_error_sum += error.square()
        self.episode_max_abs_error = torch.maximum(self.episode_max_abs_error, error)
        self.episode_samples += 1.0

        self._update_constant_episode_metrics(error, episode_length_buf)

        done = terminated | time_outs
        if torch.any(done):
            sample_count = torch.clamp(self.episode_samples[done], min=1.0)
            episode_mae = self.episode_abs_error_sum[done] / sample_count
            episode_rmse = torch.sqrt(self.episode_squared_error_sum[done] / sample_count)
            episode_maxe = self.episode_max_abs_error[done]
            done_type_id = type_id[done]

            self.completed_episodes += done.sum().float()
            self.mae_sum += episode_mae.sum()
            self.rmse_sum += episode_rmse.sum()
            self.maxe_sum += episode_maxe.sum()

            ones = torch.ones_like(episode_mae)
            self.type_completed_episodes.scatter_add_(0, done_type_id, ones)
            self.type_mae_sum.scatter_add_(0, done_type_id, episode_mae)
            self.type_rmse_sum.scatter_add_(0, done_type_id, episode_rmse)
            self.type_maxe_sum.scatter_add_(0, done_type_id, episode_maxe)

            constant_done = done & (type_id == self.constant_type_id)
            if torch.any(constant_done):
                steady_state_error = self._get_steady_state_error(constant_done)
                convergence_time = self.convergence_time[constant_done]
                climbing_time = self.climbing_time[constant_done]
                self.constant_success_episodes += (
                    convergence_time < self.cfg.episode_length_s
                ).sum().float()
                self.constant_steady_state_error_sum += steady_state_error.sum()
                self.constant_steady_state_error_sum_sq += steady_state_error.square().sum()
                self.constant_convergence_time_sum += convergence_time.sum()
                self.constant_convergence_time_sum_sq += convergence_time.square().sum()
                self.constant_climbing_time_sum += climbing_time.sum()
                self.constant_climbing_time_sum_sq += climbing_time.square().sum()

        return self.get_metrics()

    def get_metrics(self) -> dict[str, torch.Tensor]:
        """Return a stable metric schema, using NaN for unsampled categories."""
        metrics = {
            "completed_episodes": self.completed_episodes,
            "evaluation_complete": self.completed_episodes >= self.cfg.num_eval_episodes,
            "mean_absolute_error": self._mean_or_nan(self.mae_sum, self.completed_episodes),
            "root_mean_square_error": self._mean_or_nan(self.rmse_sum, self.completed_episodes),
            "maximum_absolute_error": self._mean_or_nan(self.maxe_sum, self.completed_episodes),
        }
        for trajectory_name, trajectory_id in TRAJECTORY_TYPE_TO_ID.items():
            count = self.type_completed_episodes[trajectory_id]
            metrics[f"{trajectory_name}_completed_episodes"] = count
            metrics[f"{trajectory_name}_mean_absolute_error"] = self._mean_or_nan(
                self.type_mae_sum[trajectory_id],
                count,
            )
            metrics[f"{trajectory_name}_root_mean_square_error"] = self._mean_or_nan(
                self.type_rmse_sum[trajectory_id],
                count,
            )
            metrics[f"{trajectory_name}_maximum_absolute_error"] = self._mean_or_nan(
                self.type_maxe_sum[trajectory_id],
                count,
            )

        constant_count = self.type_completed_episodes[self.constant_type_id]
        constant_steady_state_error = self._mean_or_nan(
            self.constant_steady_state_error_sum,
            constant_count,
        )
        constant_convergence_time = self._mean_or_nan(
            self.constant_convergence_time_sum,
            constant_count,
        )
        constant_climbing_time = self._mean_or_nan(
            self.constant_climbing_time_sum,
            constant_count,
        )
        metrics.update(
            {
                "constant_success_rate": self._mean_or_nan(
                    self.constant_success_episodes,
                    constant_count,
                ),
                "constant_steady_state_error": constant_steady_state_error,
                "constant_steady_state_error_std": self._population_std_or_nan(
                    self.constant_steady_state_error_sum_sq,
                    constant_steady_state_error,
                    constant_count,
                ),
                "constant_convergence_time": constant_convergence_time,
                "constant_convergence_time_std": self._population_std_or_nan(
                    self.constant_convergence_time_sum_sq,
                    constant_convergence_time,
                    constant_count,
                ),
                "constant_climbing_time": constant_climbing_time,
                "constant_climbing_time_std": self._population_std_or_nan(
                    self.constant_climbing_time_sum_sq,
                    constant_climbing_time,
                    constant_count,
                ),
            }
        )
        return metrics

    def _update_constant_episode_metrics(self, error: torch.Tensor, episode_length_buf: torch.Tensor):
        in_target_zone = error <= self.cfg.target_zone
        current_time = episode_length_buf.to(dtype=torch.float32) * self.step_dt
        self._record_steady_state_error(error)

        first_entry = in_target_zone & (self.climbing_time >= self.timeout_value)
        self.climbing_time[first_entry] = current_time[first_entry]

        new_convergence = in_target_zone & (self.convergence_time >= self.timeout_value)
        self.convergence_time[new_convergence] = current_time[new_convergence]
        self.convergence_time[~in_target_zone] = self.timeout_value

    def _record_steady_state_error(self, error: torch.Tensor):
        self.steady_state_error_buffer[self.env_indices, self.steady_state_error_index] = error
        self.steady_state_error_index = (self.steady_state_error_index + 1) % self.steady_state_window_steps
        self.steady_state_error_count = torch.clamp(
            self.steady_state_error_count + 1,
            max=self.steady_state_window_steps,
        )

    def _get_steady_state_error(self, done: torch.Tensor) -> torch.Tensor:
        sample_count = torch.clamp(self.steady_state_error_count[done], min=1).to(dtype=torch.float32)
        return self.steady_state_error_buffer[done].sum(dim=-1) / sample_count

    def _mean_or_nan(self, metric_sum: torch.Tensor, count: torch.Tensor) -> torch.Tensor:
        mean = metric_sum / torch.clamp(count, min=1.0)
        return torch.where(count > 0.0, mean, torch.full_like(mean, float("nan")))

    def _population_std_or_nan(
        self,
        metric_sum_sq: torch.Tensor,
        metric_mean: torch.Tensor,
        count: torch.Tensor,
    ) -> torch.Tensor:
        variance = metric_sum_sq / torch.clamp(count, min=1.0) - metric_mean.square()
        std = torch.sqrt(torch.clamp(variance, min=0.0))
        return torch.where(
            count > 1.0,
            std,
            torch.where(count > 0.0, torch.zeros_like(std), torch.full_like(std, float("nan"))),
        )

    def _validate_config(self):
        if self.num_envs <= 0:
            raise ValueError("num_envs must be positive.")
        if not math.isfinite(self.step_dt) or self.step_dt <= 0.0:
            raise ValueError("step_dt must be finite and positive.")
        if not math.isfinite(self.cfg.episode_length_s) or self.cfg.episode_length_s <= 0.0:
            raise ValueError("episode_length_s must be finite and positive.")
        if isinstance(self.cfg.num_eval_episodes, bool):
            raise TypeError("num_eval_episodes must be an integer, not bool.")
        try:
            num_eval_episodes = operator.index(self.cfg.num_eval_episodes)
        except TypeError as exc:
            raise TypeError("num_eval_episodes must be an integer.") from exc
        if num_eval_episodes <= 0:
            raise ValueError("num_eval_episodes must be positive.")
        if not math.isfinite(self.cfg.target_zone) or self.cfg.target_zone < 0.0:
            raise ValueError("target_zone must be finite and non-negative.")
        if not math.isfinite(self.cfg.steady_state_window_s) or self.cfg.steady_state_window_s <= 0.0:
            raise ValueError("steady_state_window_s must be finite and positive.")
