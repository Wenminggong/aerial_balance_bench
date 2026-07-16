"""Vectorized first-order velocity-response dynamics."""

from __future__ import annotations

from collections.abc import Sequence

import torch


class VelocityResponseModel:
    """Per-environment first-order response with optional additive noise."""

    NOISE_MODE_IDS = {"none": 0, "gaussian": 1, "ou": 2}

    def __init__(self, num_envs: int, device: str | torch.device):
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.tau_s = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self.gain = torch.ones(self.num_envs, device=self.device, dtype=torch.float32)
        self.bias = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self.noise_std = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self.ou_theta = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self.ou_mu = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self.input_z = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self.target_z = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self.nominal_z = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self.noise_z = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self.executed_z = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self.error_z = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)

    def reset(
        self,
        env_ids: Sequence[int] | torch.Tensor,
        initial_z: torch.Tensor,
        *,
        tau_s: torch.Tensor,
        gain: torch.Tensor,
        bias: torch.Tensor,
        noise_std: torch.Tensor,
        ou_theta: torch.Tensor,
        ou_mu: torch.Tensor,
    ):
        """Reset selected environments and install their sampled parameters."""
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        initial_z = self._selected_values(initial_z, env_ids, "initial_z")
        self.tau_s[env_ids] = self._selected_values(tau_s, env_ids, "tau_s")
        self.gain[env_ids] = self._selected_values(gain, env_ids, "gain")
        self.bias[env_ids] = self._selected_values(bias, env_ids, "bias")
        self.noise_std[env_ids] = self._selected_values(noise_std, env_ids, "noise_std")
        self.ou_theta[env_ids] = self._selected_values(ou_theta, env_ids, "ou_theta")
        self.ou_mu[env_ids] = self._selected_values(ou_mu, env_ids, "ou_mu")
        self.input_z[env_ids] = initial_z
        self.target_z[env_ids] = initial_z
        self.nominal_z[env_ids] = initial_z
        self.noise_z[env_ids] = self.ou_mu[env_ids]
        self.executed_z[env_ids] = initial_z
        self.error_z[env_ids] = 0.0

    def sync_passthrough(self, input_z: torch.Tensor):
        """Synchronize diagnostics when the response model is disabled."""
        input_z = self._full_values(input_z, "input_z")
        self.input_z.copy_(input_z)
        self.target_z.copy_(input_z)
        self.nominal_z.copy_(input_z)
        self.noise_z.zero_()
        self.executed_z.copy_(input_z)
        self.error_z.zero_()

    def step(
        self,
        input_z: torch.Tensor,
        step_dt: float,
        noise_mode: str,
        *,
        noise_clip: float = 0.0,
        max_abs_velocity: float = 0.0,
        normal_samples: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Advance every environment by one control step and return executed Z velocity."""
        if step_dt <= 0.0:
            raise ValueError("Velocity response requires step_dt > 0.")
        if noise_clip < 0.0:
            raise ValueError("Velocity response noise_clip must be non-negative.")
        if max_abs_velocity < 0.0:
            raise ValueError("Velocity response max_abs_velocity must be non-negative.")

        mode = self.normalize_noise_mode(noise_mode)
        input_z = self._full_values(input_z, "input_z")
        self.input_z.copy_(input_z)
        self.target_z.copy_(self.gain * input_z + self.bias)

        filtered_mask = self.tau_s > 0.0
        decay = torch.zeros_like(self.tau_s)
        decay[filtered_mask] = torch.exp(-float(step_dt) / self.tau_s[filtered_mask])
        filtered = decay * self.nominal_z + (1.0 - decay) * self.target_z
        nominal = torch.where(filtered_mask, filtered, self.target_z)
        self.nominal_z.copy_(nominal)

        noise = self._next_noise(mode, float(step_dt), normal_samples)
        if noise_clip > 0.0:
            noise = torch.clamp(noise, min=-noise_clip, max=noise_clip)
        self.noise_z.copy_(noise)

        executed = self.nominal_z + self.noise_z
        if max_abs_velocity > 0.0:
            executed = torch.clamp(executed, min=-max_abs_velocity, max=max_abs_velocity)
        self.executed_z.copy_(executed)
        self.error_z.copy_(self.executed_z - self.input_z)
        return self.executed_z

    def _next_noise(
        self,
        mode: str,
        step_dt: float,
        normal_samples: torch.Tensor | None,
    ) -> torch.Tensor:
        if mode == "none":
            return torch.zeros_like(self.noise_z)

        if normal_samples is None:
            normal_samples = torch.randn_like(self.noise_z)
        else:
            normal_samples = self._full_values(normal_samples, "normal_samples")

        if mode == "gaussian":
            return self.noise_std * normal_samples

        decay = torch.exp(-self.ou_theta * step_dt)
        innovation_scale = self.noise_std * torch.sqrt(torch.clamp(1.0 - decay.square(), min=0.0))
        return self.ou_mu + decay * (self.noise_z - self.ou_mu) + innovation_scale * normal_samples

    def _selected_values(self, values: torch.Tensor, env_ids: torch.Tensor, field_name: str) -> torch.Tensor:
        values = torch.as_tensor(values, device=self.device, dtype=torch.float32)
        if values.shape == (self.num_envs,):
            return values[env_ids]
        if values.shape != (env_ids.numel(),):
            raise ValueError(
                f"Velocity response {field_name} must have shape ({env_ids.numel()},) "
                f"or ({self.num_envs},), got {tuple(values.shape)}."
            )
        return values

    def _full_values(self, values: torch.Tensor, field_name: str) -> torch.Tensor:
        values = torch.as_tensor(values, device=self.device, dtype=torch.float32)
        if values.shape != (self.num_envs,):
            raise ValueError(
                f"Velocity response {field_name} must have shape ({self.num_envs},), "
                f"got {tuple(values.shape)}."
            )
        return values

    @classmethod
    def normalize_noise_mode(cls, value: str) -> str:
        """Normalize and validate the configured noise mode."""
        mode = str(value).lower().replace("-", "_")
        if mode == "white":
            mode = "gaussian"
        if mode not in cls.NOISE_MODE_IDS:
            raise ValueError("Velocity response noise mode must be 'none', 'gaussian'/'white', or 'ou'.")
        return mode

