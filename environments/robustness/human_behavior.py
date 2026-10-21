"""Independent, vectorized human endpoint dynamics (no Isaac runtime dependency)."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch


# Shared by the model and evaluation runners. Values name (N, 1) model buffers.
_LOG_BUFFERS = {
    "human_enabled": "enabled",
    "human_pos_z": "pos",
    "human_vel_z": "vel",
    "human_acc_z": "acc_mean",
    "human_pos_ref_z": "pos_ref",
    "human_pos_feedback_raw_vel_z": "pos_feedback_raw",
    "human_pos_feedback_vel_z": "pos_feedback",
    "human_pos_k_base": "pos_k_base",
    "human_pos_deadzone": "pos_deadzone",
    "human_pos_corr_tau": "pos_corr_tau",
    "human_ou_theta": "ou_theta",
    "human_ou_sigma": "ou_sigma",
    "human_behavior_sigma_d": "behavior_sigma_d",
    "human_behavior_e_scale": "behavior_e_scale",
    "human_behavior_wc": "behavior_wc",
    "human_behavior_wd": "behavior_wd",
    "human_behavior_gain": "behavior_gain",
    "human_cooperative_term_z": "cooperative_term",
    "human_disturbed_term_z": "disturbed_term",
    "human_behavior_target_vel_z": "behavior_term",
    "human_desired_vel_z": "vel_des",
    "human_cpid_action_z": "cpid_action",
    "human_pid_target_vel_z": "cpid_vel",
    "human_cpid_sign_z": "cpid_sign",
    "human_ou_drift_vel_z": "ou_state",
    "human_target_vel_z": "vel_cmd",
    "human_target_acc_z": "acc_cmd",
    "human_angle_kp": "angle_kp",
    "human_angle_ti": "angle_ti",
    "human_angle_td": "angle_td",
    "human_angle_max_theta_change": "angle_max_theta_change",
    "human_vel_kp": "vel_kp",
    "human_vel_ti": "vel_ti",
    "human_vel_td": "vel_td",
    "human_vel_max_acc": "vel_max_acc",
}
HUMAN_STEP_EXTRA_FIELDS = (*_LOG_BUFFERS, "human_behavior_type_id")


class HumanBehaviorModel:
    """Cascaded PID + error-dependent disruption + recentering + OU velocity.

    Inputs to reset/history initialization are full-batch tensors; ``env_ids``
    selects rows. All state is local to this model, including error differences.
    Configuration is fixed for the lifetime of the model.
    """

    MODE_IDS = {"cooperative": 0, "random-walk": 1, "disturbed": 2}
    _INTERNAL_BUFFERS = (
        "error_prev1", "error_prev2", "theta_target", "theta_error_prev1",
        "theta_error_prev2", "vel_target", "vel_increment", "acc_sum", "acc_count",
    )

    def __init__(self, cfg: Any, num_envs: int, device: str | torch.device):
        self.cfg = cfg
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.mode = str(cfg.human_behavior_type).lower()
        if self.mode == "random":
            self.mode = "random-walk"
        if self.mode not in self.MODE_IDS:
            raise ValueError(f"Unsupported human_behavior_type: {cfg.human_behavior_type!r}")
        self.active = bool(cfg.enabled and cfg.external_disturbance_enabled)
        self._validate_config()
        self._buffer_names = (*_LOG_BUFFERS.values(), *self._INTERNAL_BUFFERS)
        for name in self._buffer_names:
            setattr(self, name, torch.zeros((self.num_envs, 1), device=self.device, dtype=torch.float32))
        self.behavior_type_id = torch.full(
            (self.num_envs,), self.MODE_IDS[self.mode], device=self.device, dtype=torch.long
        )
        self.ramp_counter = torch.zeros((self.num_envs, 1), device=self.device, dtype=torch.long)
        self.decimation = 1

    def _validate_config(self):
        cfg = self.cfg
        nonnegative = (
            "human_angle_kp_min", "human_angle_kp_max", "human_angle_ti_min", "human_angle_ti_max",
            "human_angle_td_min", "human_angle_td_max", "human_angle_max_theta_change_deg",
            "human_vel_kp_min", "human_vel_kp_max", "human_vel_ti", "human_vel_td", "human_vel_max_acc",
            "human_pos_k_base", "human_pos_deadzone", "human_pos_corr_max", "human_pos_corr_tau",
            "human_pos_k_base_min", "human_pos_k_base_max", "human_pos_deadzone_min",
            "human_pos_deadzone_max", "human_pos_corr_tau_min", "human_pos_corr_tau_max",
            "external_disturbance_ou_clip",
        )
        positive = (
            "human_behavior_sigma_d", "human_behavior_e_scale", "human_disturbed_sigma_d_min",
            "human_disturbed_sigma_d_max", "human_disturbed_e_scale_min", "human_disturbed_e_scale_max",
        )
        unrestricted = (
            "external_disturbance_ou_mu", "human_disturbed_wd_min", "human_disturbed_wd_max",
            *(f"human_behavior_{mode}_{weight}" for mode in ("cooperative", "random_walk", "disturbed")
              for weight in ("wc", "wd")),
        )
        for name in (*nonnegative, *positive, *unrestricted):
            value = float(getattr(cfg, name))
            if not math.isfinite(value) or (name in nonnegative and value < 0) or (name in positive and value <= 0):
                raise ValueError(f"Invalid robustness.{name}: {value}")
        for stem in ("human_angle_kp", "human_angle_ti", "human_angle_td", "human_vel_kp",
                     "human_pos_k_base", "human_pos_deadzone", "human_pos_corr_tau",
                     "human_disturbed_wd", "human_disturbed_sigma_d", "human_disturbed_e_scale"):
            if getattr(cfg, stem + "_min") > getattr(cfg, stem + "_max"):
                raise ValueError(f"robustness.{stem}_min must not exceed {stem}_max")
        for name in ("external_disturbance_ou_theta_range", "external_disturbance_ou_sigma_range"):
            bounds = getattr(cfg, name)
            if len(bounds) != 2:
                raise ValueError(f"robustness.{name} must contain two bounds")
            lo, hi = map(float, bounds)
            if not math.isfinite(lo) or not math.isfinite(hi) or lo < 0 or lo > hi:
                raise ValueError(f"robustness.{name} must be finite, nonnegative and ordered")

    def _ids(self, env_ids: Sequence[int] | torch.Tensor) -> torch.Tensor:
        return torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

    def _column(self, value: torch.Tensor) -> torch.Tensor:
        if value.shape not in ((self.num_envs,), (self.num_envs, 1)):
            raise ValueError(f"Expected ({self.num_envs},) or ({self.num_envs}, 1), got {tuple(value.shape)}")
        return value.to(device=self.device, dtype=torch.float32).reshape(self.num_envs, 1)

    def _sample(self, ids: torch.Tensor, name: str, low: float, high: float):
        getattr(self, name)[ids] = low + (high - low) * torch.rand(
            (ids.numel(), 1), device=self.device, dtype=torch.float32
        )

    def reset(self, env_ids: Sequence[int] | torch.Tensor, position_reference: torch.Tensor):
        """Clear and resample only selected rows; disabled models consume no RNG."""
        ids = self._ids(env_ids)
        if ids.numel() == 0:
            return
        for name in self._buffer_names:
            getattr(self, name)[ids] = 0.0
        self.ramp_counter[ids] = 0
        self.pos_ref[ids] = self._column(position_reference)[ids]
        self.enabled[ids] = float(self.active)
        if not self.active:
            return
        cfg = self.cfg
        for name in ("pos_k_base", "pos_deadzone", "pos_corr_tau"):
            stem = "human_" + name
            if cfg.human_pos_feedback_sample:
                self._sample(ids, name, getattr(cfg, stem + "_min"), getattr(cfg, stem + "_max"))
            else:
                getattr(self, name)[ids] = getattr(cfg, stem)
        mode_key = self.mode.replace("-", "_")
        self.behavior_wc[ids] = getattr(cfg, f"human_behavior_{mode_key}_wc")
        self.behavior_wd[ids] = getattr(cfg, f"human_behavior_{mode_key}_wd")
        self.behavior_sigma_d[ids] = cfg.human_behavior_sigma_d
        self.behavior_e_scale[ids] = cfg.human_behavior_e_scale
        if self.mode == "disturbed":
            for suffix in ("wd", "sigma_d", "e_scale"):
                self._sample(ids, "behavior_" + suffix, getattr(cfg, f"human_disturbed_{suffix}_min"),
                             getattr(cfg, f"human_disturbed_{suffix}_max"))
        # All modes use the same OU configuration and reset-time sampling rule.
        self._sample(ids, "ou_theta", *cfg.external_disturbance_ou_theta_range)
        self._sample(ids, "ou_sigma", *cfg.external_disturbance_ou_sigma_range)
        for name, default in (("angle_kp", 0.5), ("angle_ti", 1.5), ("angle_td", 0.6), ("vel_kp", 10.0)):
            if self.mode == "cooperative":
                self._sample(ids, name, getattr(cfg, f"human_{name}_min"), getattr(cfg, f"human_{name}_max"))
            else:
                getattr(self, name)[ids] = default
        self.angle_max_theta_change[ids] = math.radians(cfg.human_angle_max_theta_change_deg)
        for name in ("vel_ti", "vel_td", "vel_max_acc"):
            getattr(self, name)[ids] = getattr(cfg, "human_" + name)

    def initialize_history(self, env_ids: Sequence[int] | torch.Tensor, error: torch.Tensor):
        """Initialize after the task and physical state have finished resetting."""
        ids = self._ids(env_ids)
        values = self._column(error)[ids]
        self.error_prev1[ids] = values
        self.error_prev2[ids] = values

    @staticmethod
    def _integral(kp: torch.Tensor, ti: torch.Tensor, error: torch.Tensor, dt: float) -> torch.Tensor:
        safe_ti = torch.where(ti != 0, ti, torch.ones_like(ti))
        return torch.where(ti != 0, kp * dt / safe_ti * error, torch.zeros_like(error))

    def prepare_step(
        self, error: torch.Tensor, theta: torch.Tensor, position: torch.Tensor,
        step_dt: float, decimation: int, *, noise: torch.Tensor | None = None,
    ):
        """Update once per control step. Optional standard-normal samples aid replay."""
        dt = float(step_dt)
        if not math.isfinite(dt) or dt <= 0 or isinstance(decimation, bool) or int(decimation) != decimation or decimation < 1:
            raise ValueError("Human dynamics require positive step_dt and integer decimation >= 1")
        self.decimation = int(decimation)
        self.acc_sum.zero_()
        self.acc_count.zero_()
        self.acc_mean.zero_()
        self.ramp_counter.zero_()
        if not self.active:
            return
        error, theta, position = (self._column(x) for x in (error, theta, position))
        d1 = error - self.error_prev1
        d2 = error - 2 * self.error_prev1 + self.error_prev2
        delta_theta = self.angle_kp * d1 + self._integral(self.angle_kp, self.angle_ti, error, dt)
        delta_theta += self.angle_kp * self.angle_td / dt * d2
        self.theta_target.add_(torch.clamp(delta_theta, -self.angle_max_theta_change, self.angle_max_theta_change))
        theta_error = self.theta_target - theta
        delta_vel = self.vel_kp * (theta_error - self.theta_error_prev1)
        delta_vel += self._integral(self.vel_kp, self.vel_ti, theta_error, dt)
        delta_vel += self.vel_kp * self.vel_td / dt * (theta_error - 2 * self.theta_error_prev1 + self.theta_error_prev2)
        max_change = self.vel_max_acc * dt
        delta_vel = torch.clamp(delta_vel, -max_change, max_change)
        safe_change = torch.where(max_change > 0, max_change, torch.ones_like(max_change))
        self.cpid_action.copy_(-delta_vel / safe_change)
        # Old human endpoint sign: -normalized_action * max_change == delta_vel.
        self.cpid_vel.add_(delta_vel)
        self.error_prev2.copy_(self.error_prev1)
        self.error_prev1.copy_(error)
        self.theta_error_prev2.copy_(self.theta_error_prev1)
        self.theta_error_prev1.copy_(theta_error)
        self.cpid_sign.copy_(torch.tanh(error / self.behavior_e_scale))
        self.behavior_gain.copy_(torch.exp(-torch.square(error / self.behavior_sigma_d)))
        self.cooperative_term.copy_(self.behavior_wc * self.cpid_vel)
        self.disturbed_term.copy_(self.behavior_wd * self.behavior_gain * self.cpid_sign)
        self.behavior_term.copy_(self.cooperative_term + self.disturbed_term)
        pos_error = position - self.pos_ref
        effective = torch.sign(pos_error) * torch.clamp(pos_error.abs() - self.pos_deadzone, min=0)
        cap = float(self.cfg.human_pos_corr_max)
        self.pos_feedback_raw.copy_(cap * torch.tanh(self.pos_k_base * effective / max(cap, 1e-6)))
        self.pos_feedback.add_(dt / (self.pos_corr_tau + dt) * (self.pos_feedback_raw - self.pos_feedback))
        if self.cfg.human_ou_enabled:
            xi = torch.randn_like(self.ou_state) if noise is None else self._column(noise)
            self.ou_state.add_(self.ou_theta * (self.cfg.external_disturbance_ou_mu - self.ou_state) * dt
                               + self.ou_sigma * math.sqrt(dt) * xi)
            clip = float(self.cfg.external_disturbance_ou_clip)
            if clip > 0:
                self.ou_state.clamp_(-clip, clip)
        else:
            self.ou_state.zero_()
        self.vel_des.copy_(self.behavior_term - self.pos_feedback + self.ou_state)
        delta = torch.clamp(self.vel_des - self.vel_cmd, -max_change, max_change)
        self.vel_target.copy_(self.vel_cmd + delta)
        self.acc_cmd.copy_(delta / dt)
        self.vel_increment.copy_(delta / self.decimation)

    def advance_substep(self) -> torch.Tensor:
        """Return this substep's command; never overshoot the prepared target."""
        if self.active:
            self.ramp_counter.add_(1)
            self.vel_cmd.copy_(torch.where(self.ramp_counter >= self.decimation,
                                           self.vel_target, self.vel_cmd + self.vel_increment))
        return self.vel_cmd

    def record_measurement(
        self, position: torch.Tensor, velocity: torch.Tensor, acceleration: torch.Tensor,
        env_ids: Sequence[int] | torch.Tensor | None = None, *, accumulate: bool = True,
    ):
        """Read actual joint motion, also when human commands are disabled."""
        ids = slice(None) if env_ids is None else self._ids(env_ids)
        self.pos[ids] = self._column(position)[ids]
        self.vel[ids] = self._column(velocity)[ids]
        if accumulate:
            self.acc_sum[ids] += self._column(acceleration)[ids]
            self.acc_count[ids] += 1
            self.acc_mean[ids] = self.acc_sum[ids] / self.acc_count[ids].clamp_min(1)

    def get_state(self) -> dict[str, torch.Tensor]:
        """Return live per-environment diagnostics; caller snapshots before reset."""
        state = {key: getattr(self, name)[:, 0] for key, name in _LOG_BUFFERS.items()}
        state["human_behavior_type_id"] = self.behavior_type_id
        return state
