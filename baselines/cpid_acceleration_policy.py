"""Cascaded incremental PID baseline for the acceleration-command interface."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import torch

from .base_policy import BasePolicy, BasePolicyCfg, ObservationIndex


@dataclass
class AccelerationAnglePIDCfg:
    """Outer-loop angle PID configuration."""

    kp: float = 0.5
    ti: float = 1.5
    td: float = 0.6
    max_theta_change: float = 0.1 / 180.0 * math.pi
    max_theta_ref: float = 0.12


@dataclass
class AccelerationPIDCfg:
    """Inner-loop vertical acceleration PID configuration."""

    kp: float = 10.0
    ti: float = 1000.0
    td: float = 0.0
    max_delta_acc: float | str = "auto"
    max_acc: float | str = "auto"
    action_sign: float = -1.0


@dataclass
class AccelerationCPIDPolicyCfg(BasePolicyCfg):
    """Configuration for the acceleration-interface CPID high-level baseline."""

    name: str = "cpid_acceleration"
    angle_pid: AccelerationAnglePIDCfg = field(default_factory=AccelerationAnglePIDCfg)
    acceleration_pid: AccelerationPIDCfg = field(default_factory=AccelerationPIDCfg)

    @classmethod
    def from_dict(cls, data: Mapping | None) -> "AccelerationCPIDPolicyCfg":
        """Build an acceleration CPID config from a YAML dictionary."""
        cfg = cls()
        if not data:
            return cfg

        policy_data = data.get("cpid_acceleration_policy", data.get("policy", data))
        if "name" in policy_data:
            cfg.name = str(policy_data["name"])
        if "policy_name" in policy_data:
            cfg.name = str(policy_data["policy_name"])

        angle_data = policy_data.get("angle_pid", policy_data.get("angle_pid_config", {}))
        acceleration_data = policy_data.get(
            "acceleration_pid",
            policy_data.get("acc_pid_config", policy_data.get("acceleration_pid_config", {})),
        )
        _update_cfg(cfg.angle_pid, angle_data)
        _update_cfg(cfg.acceleration_pid, acceleration_data)
        return cfg


class AccelerationCPIDPolicy(BasePolicy):
    """Cascaded incremental PID policy for the benchmark acceleration interface."""

    cfg: AccelerationCPIDPolicyCfg

    def __init__(
        self,
        cfg: AccelerationCPIDPolicyCfg,
        num_envs: int,
        device: str | torch.device,
        step_dt: float,
    ):
        super().__init__(cfg, num_envs, device)
        if step_dt <= 0.0:
            raise ValueError("AccelerationCPIDPolicy requires a positive step_dt.")
        self.step_dt = float(step_dt)
        self.max_delta_acc = self._resolve_positive_limit(cfg.acceleration_pid.max_delta_acc, default=0.5)
        self.max_acc = self._resolve_nonnegative_limit(cfg.acceleration_pid.max_acc, default=5.0)
        self.action_sign = float(cfg.acceleration_pid.action_sign)

        self.theta_ref = torch.zeros((self.num_envs, 1), device=self.device)
        self.acceleration_cmd = torch.zeros((self.num_envs, 1), device=self.device)
        self.error_prev1 = torch.zeros((self.num_envs, 1), device=self.device)
        self.error_prev2 = torch.zeros((self.num_envs, 1), device=self.device)
        self.error_theta_prev1 = torch.zeros((self.num_envs, 1), device=self.device)
        self.error_theta_prev2 = torch.zeros((self.num_envs, 1), device=self.device)
        self.history_needs_init = torch.ones((self.num_envs,), dtype=torch.bool, device=self.device)

        self.raw_error = torch.zeros((self.num_envs, 1), device=self.device)
        self.error = torch.zeros((self.num_envs, 1), device=self.device)
        self.error_dot = torch.zeros((self.num_envs, 1), device=self.device)
        self.error_ddot = torch.zeros((self.num_envs, 1), device=self.device)
        self.error_theta = torch.zeros((self.num_envs, 1), device=self.device)
        self.error_theta_dot = torch.zeros((self.num_envs, 1), device=self.device)
        self.error_theta_ddot = torch.zeros((self.num_envs, 1), device=self.device)
        self.delta_theta = torch.zeros((self.num_envs, 1), device=self.device)
        self.raw_delta_acc = torch.zeros((self.num_envs, 1), device=self.device)
        self.last_action = torch.zeros((self.num_envs, 1), device=self.device)
        self.saturated = torch.zeros((self.num_envs, 1), dtype=torch.bool, device=self.device)

    def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None):
        """Reset CPID internal state for all envs or selected envs."""
        env_ids = self._env_ids_tensor(env_ids)
        if env_ids.numel() == 0:
            return

        for buffer in (
            self.theta_ref,
            self.acceleration_cmd,
            self.error_prev1,
            self.error_prev2,
            self.error_theta_prev1,
            self.error_theta_prev2,
            self.raw_error,
            self.error,
            self.error_dot,
            self.error_ddot,
            self.error_theta,
            self.error_theta_dot,
            self.error_theta_ddot,
            self.delta_theta,
            self.raw_delta_acc,
            self.last_action,
        ):
            buffer[env_ids] = 0.0
        self.saturated[env_ids] = False
        self.history_needs_init[env_ids] = True

    def act(self, observations: dict[str, torch.Tensor] | torch.Tensor, extras: dict | None = None) -> torch.Tensor:
        """Compute the acceleration increment action ``delta_arz``."""
        del extras
        raw_obs = self._extract_policy_observation(observations)
        if raw_obs.shape[-1] < 11:
            raise ValueError(
                f"AccelerationCPIDPolicy expects an 11-D observation, got shape {tuple(raw_obs.shape)}."
            )
        if raw_obs.shape[0] != self.num_envs:
            raise ValueError(f"AccelerationCPIDPolicy expected {self.num_envs} envs, got {raw_obs.shape[0]}.")
        obs = raw_obs[:, :11]

        raw_pb = obs[:, ObservationIndex.PB : ObservationIndex.PB + 1]
        raw_pg = obs[:, ObservationIndex.PG : ObservationIndex.PG + 1]
        current_theta = obs[:, ObservationIndex.THETA : ObservationIndex.THETA + 1]
        self.raw_error.copy_(raw_pb - raw_pg)

        init_envs = self.history_needs_init.clone()
        if torch.any(init_envs):
            self.error_prev1[init_envs] = self.raw_error[init_envs]
            self.error_prev2[init_envs] = self.raw_error[init_envs]

        self.error.copy_(self.raw_error)
        self.error_dot.copy_(self.error - self.error_prev1)
        self.error_ddot.copy_(self.error - 2.0 * self.error_prev1 + self.error_prev2)

        self.delta_theta.copy_(self._angle_increment(self.error, self.error_dot, self.error_ddot))
        self.theta_ref += self.delta_theta
        max_theta_ref = float(self.cfg.angle_pid.max_theta_ref)
        if max_theta_ref > 0.0:
            self.theta_ref.clamp_(min=-max_theta_ref, max=max_theta_ref)

        self.error_theta.copy_(self.theta_ref - current_theta)
        if torch.any(init_envs):
            self.error_theta_prev1[init_envs] = self.error_theta[init_envs]
            self.error_theta_prev2[init_envs] = self.error_theta[init_envs]
        self.error_theta_dot.copy_(self.error_theta - self.error_theta_prev1)
        self.error_theta_ddot.copy_(self.error_theta - 2.0 * self.error_theta_prev1 + self.error_theta_prev2)

        self.raw_delta_acc.copy_(
            self.action_sign
            * self._acceleration_increment(
                self.error_theta,
                self.error_theta_dot,
                self.error_theta_ddot,
            )
        )
        clipped_delta_acc = torch.clamp(self.raw_delta_acc, -self.max_delta_acc, self.max_delta_acc)
        next_acc = self.acceleration_cmd + clipped_delta_acc
        if self.max_acc > 0.0:
            next_acc = torch.clamp(next_acc, -self.max_acc, self.max_acc)
        self.last_action.copy_(next_acc - self.acceleration_cmd)
        self.acceleration_cmd.copy_(next_acc)
        self.saturated.copy_(torch.abs(self.raw_delta_acc - self.last_action) > 1e-6)

        self.error_prev2.copy_(self.error_prev1)
        self.error_prev1.copy_(self.raw_error)
        self.error_theta_prev2.copy_(self.error_theta_prev1)
        self.error_theta_prev1.copy_(self.error_theta)
        if torch.any(init_envs):
            self.history_needs_init[init_envs] = False
        return self.last_action.clone()

    def get_state(self) -> dict[str, torch.Tensor]:
        """Return CPID state useful for rollout diagnostics."""
        return {
            "policy_raw_error": self.raw_error[:, 0],
            "policy_error": self.error[:, 0],
            "policy_error_dot": self.error_dot[:, 0],
            "policy_error_ddot": self.error_ddot[:, 0],
            "policy_theta_ref": self.theta_ref[:, 0],
            "policy_error_theta": self.error_theta[:, 0],
            "policy_error_theta_dot": self.error_theta_dot[:, 0],
            "policy_error_theta_ddot": self.error_theta_ddot[:, 0],
            "policy_delta_theta": self.delta_theta[:, 0],
            "policy_raw_delta_arz": self.raw_delta_acc[:, 0],
            "policy_delta_arz": self.last_action[:, 0],
            "policy_arz_cmd": self.acceleration_cmd[:, 0],
            "policy_acceleration_saturated": self.saturated[:, 0].to(dtype=torch.float32),
        }

    def to(self, device: str | torch.device):
        device = torch.device(device)
        for name, value in vars(self).items():
            if isinstance(value, torch.Tensor):
                setattr(self, name, value.to(device=device))
        self.device = device
        return self

    def _angle_increment(self, error: torch.Tensor, error_dot: torch.Tensor, error_ddot: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg.angle_pid
        delta_theta = float(cfg.kp) * error_dot
        delta_theta += self._safe_incremental_i_term(cfg.kp, cfg.ti, error)
        delta_theta += float(cfg.kp) * float(cfg.td) / self.step_dt * error_ddot
        return torch.clamp(delta_theta, -float(cfg.max_theta_change), float(cfg.max_theta_change))

    def _acceleration_increment(
        self,
        error_theta: torch.Tensor,
        error_theta_dot: torch.Tensor,
        error_theta_ddot: torch.Tensor,
    ) -> torch.Tensor:
        cfg = self.cfg.acceleration_pid
        delta_acc = float(cfg.kp) * error_theta_dot
        delta_acc += self._safe_incremental_i_term(cfg.kp, cfg.ti, error_theta)
        delta_acc += float(cfg.kp) * float(cfg.td) / self.step_dt * error_theta_ddot
        return delta_acc

    def _safe_incremental_i_term(self, kp: float, ti: float, error: torch.Tensor) -> torch.Tensor:
        if float(ti) == 0.0:
            return torch.zeros_like(error)
        return float(kp) * self.step_dt / float(ti) * error

    @staticmethod
    def _resolve_positive_limit(value: float | str, default: float) -> float:
        if _is_auto(value):
            return float(default)
        limit = float(value)
        if limit <= 0.0:
            raise ValueError("AccelerationCPIDPolicy requires positive acceleration limits.")
        return limit

    @staticmethod
    def _resolve_nonnegative_limit(value: float | str, default: float) -> float:
        if _is_auto(value):
            return float(default)
        limit = float(value)
        if limit < 0.0:
            raise ValueError("AccelerationCPIDPolicy requires max_acc >= 0.")
        return limit


def _update_cfg(target, values: Mapping | None):
    if not values:
        return
    aliases = {
        "Kp": "kp",
        "Ti": "ti",
        "Td": "td",
    }
    valid_keys = set(target.__dataclass_fields__)
    for key, value in values.items():
        normalized_key = aliases.get(key, key)
        if normalized_key not in valid_keys:
            continue
        if _is_auto(value):
            setattr(target, normalized_key, "auto")
        else:
            setattr(target, normalized_key, float(value))


def _is_auto(value) -> bool:
    return isinstance(value, str) and value.lower() == "auto"
