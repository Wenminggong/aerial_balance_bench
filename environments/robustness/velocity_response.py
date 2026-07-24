"""Vectorized plant-matched velocity-response dynamics."""

from __future__ import annotations

from collections.abc import Sequence

import torch


class VelocityResponseModel:
    """Per-environment target response with a feed-forward plant inverse."""

    NOISE_MODE_IDS = {"none": 0, "gaussian": 1, "ou": 2}
    DEFAULT_SIM_TAU_S = 0.139
    DEFAULT_SIM_GAIN = 1.0
    DEFAULT_SIM_BIAS = -0.00055
    RESPONSE_FRACTION_EPSILON = 1.0e-6

    def __init__(self, num_envs: int, device: str | torch.device):
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.tau_s = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self.gain = torch.ones(self.num_envs, device=self.device, dtype=torch.float32)
        self.bias = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self.sim_tau_s = torch.full(
            (self.num_envs,),
            self.DEFAULT_SIM_TAU_S,
            device=self.device,
            dtype=torch.float32,
        )
        self.sim_gain = torch.full(
            (self.num_envs,),
            self.DEFAULT_SIM_GAIN,
            device=self.device,
            dtype=torch.float32,
        )
        self.sim_bias = torch.full(
            (self.num_envs,),
            self.DEFAULT_SIM_BIAS,
            device=self.device,
            dtype=torch.float32,
        )
        self.noise_std = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self.ou_theta = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self.ou_mu = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self.input_z = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self.target_z = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self.nominal_z = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self.noise_z = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self.executed_z = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self.previous_executed_z = torch.zeros(
            self.num_envs,
            device=self.device,
            dtype=torch.float32,
        )
        self.compensated_command_z = torch.zeros(
            self.num_envs,
            device=self.device,
            dtype=torch.float32,
        )
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
        initial_input_z: torch.Tensor | None = None,
        sim_tau_s: torch.Tensor | None = None,
        sim_gain: torch.Tensor | None = None,
        sim_bias: torch.Tensor | None = None,
    ):
        """Reset selected environments and install target and plant parameters."""
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        initial_z = self._selected_values(initial_z, env_ids, "initial_z")
        if initial_input_z is None:
            initial_input_z = initial_z
        else:
            initial_input_z = self._selected_values(
                initial_input_z,
                env_ids,
                "initial_input_z",
            )
        if sim_tau_s is None:
            sim_tau_s = torch.full_like(initial_z, self.DEFAULT_SIM_TAU_S)
        if sim_gain is None:
            sim_gain = torch.full_like(initial_z, self.DEFAULT_SIM_GAIN)
        if sim_bias is None:
            sim_bias = torch.full_like(initial_z, self.DEFAULT_SIM_BIAS)

        self.tau_s[env_ids] = self._selected_values(tau_s, env_ids, "tau_s")
        self.gain[env_ids] = self._selected_values(gain, env_ids, "gain")
        self.bias[env_ids] = self._selected_values(bias, env_ids, "bias")
        self.sim_tau_s[env_ids] = self._selected_values(
            sim_tau_s,
            env_ids,
            "sim_tau_s",
        )
        self.sim_gain[env_ids] = self._selected_values(
            sim_gain,
            env_ids,
            "sim_gain",
        )
        self.sim_bias[env_ids] = self._selected_values(
            sim_bias,
            env_ids,
            "sim_bias",
        )
        if torch.any(~torch.isfinite(self.sim_tau_s[env_ids])) or torch.any(
            self.sim_tau_s[env_ids] <= 0.0
        ):
            raise ValueError("Velocity response sim_tau_s must be finite and positive.")
        if torch.any(~torch.isfinite(self.sim_gain[env_ids])) or torch.any(
            self.sim_gain[env_ids] <= 0.0
        ):
            raise ValueError("Velocity response sim_gain must be finite and positive.")
        if torch.any(~torch.isfinite(self.sim_bias[env_ids])):
            raise ValueError("Velocity response sim_bias must be finite.")
        self.noise_std[env_ids] = self._selected_values(noise_std, env_ids, "noise_std")
        self.ou_theta[env_ids] = self._selected_values(ou_theta, env_ids, "ou_theta")
        self.ou_mu[env_ids] = self._selected_values(ou_mu, env_ids, "ou_mu")
        self.input_z[env_ids] = initial_input_z
        self.target_z[env_ids] = self.gain[env_ids] * initial_input_z + self.bias[env_ids]
        self.nominal_z[env_ids] = initial_z
        self.noise_z[env_ids] = self.ou_mu[env_ids]
        self.executed_z[env_ids] = initial_z
        self.previous_executed_z[env_ids] = initial_z
        self.compensated_command_z[env_ids] = initial_input_z
        self.error_z[env_ids] = initial_z - initial_input_z

    def sync_passthrough(self, input_z: torch.Tensor):
        """Synchronize diagnostics when the response model is disabled."""
        input_z = self._full_values(input_z, "input_z")
        self.input_z.copy_(input_z)
        self.target_z.copy_(input_z)
        self.nominal_z.copy_(input_z)
        self.noise_z.zero_()
        self.executed_z.copy_(input_z)
        self.previous_executed_z.copy_(input_z)
        self.compensated_command_z.copy_(input_z)
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
        """Advance the target response and return the plant-compensated command."""
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

        target_filtered_mask = self.tau_s > 0.0
        target_decay = torch.zeros_like(self.tau_s)
        target_decay[target_filtered_mask] = torch.exp(
            -float(step_dt) / self.tau_s[target_filtered_mask]
        )
        filtered = target_decay * self.nominal_z + (1.0 - target_decay) * self.target_z
        nominal = torch.where(target_filtered_mask, filtered, self.target_z)
        self.nominal_z.copy_(nominal)

        noise = self._next_noise(mode, float(step_dt), normal_samples)
        if noise_clip > 0.0:
            noise = torch.clamp(noise, min=-noise_clip, max=noise_clip)
        self.noise_z.copy_(noise)

        executed = self.nominal_z + self.noise_z
        if max_abs_velocity > 0.0:
            executed = torch.clamp(executed, min=-max_abs_velocity, max=max_abs_velocity)
        self.executed_z.copy_(executed)

        sim_decay = torch.exp(-float(step_dt) / self.sim_tau_s)
        sim_response_fraction = 1.0 - sim_decay
        if torch.any(sim_response_fraction <= self.RESPONSE_FRACTION_EPSILON):
            min_fraction = float(torch.min(sim_response_fraction).item())
            raise ValueError(
                "Velocity response simulator model has a near-zero one-step "
                f"response fraction ({min_fraction:.6g}); reduce sim_tau_s or "
                "increase step_dt."
            )
        denominator = sim_response_fraction * self.sim_gain
        compensated = (
            self.executed_z - sim_decay * self.previous_executed_z
        ) / denominator
        compensated -= self.sim_bias / self.sim_gain
        self.compensated_command_z.copy_(compensated)
        self.previous_executed_z.copy_(self.executed_z)
        self.error_z.copy_(self.executed_z - self.input_z)
        return self.compensated_command_z

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
