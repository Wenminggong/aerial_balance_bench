"""Observation adapters used by RL baselines."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch

from .base_policy import ObservationIndex


@dataclass
class RLObservationAdapterCfg:
    """Configuration for mapping benchmark observations to RL inputs."""

    observation_mode: str = "legacy8"
    command_history_length: int | str = 0

    @classmethod
    def from_dict(cls, data: Mapping | None) -> "RLObservationAdapterCfg":
        cfg = cls()
        if not data:
            return cfg
        policy_data = data.get("rl_policy", data.get("policy", data))
        if "observation_mode" in policy_data:
            cfg.observation_mode = str(policy_data["observation_mode"])
        if "command_history_length" in policy_data:
            cfg.command_history_length = policy_data["command_history_length"]
        elif "action_history_length" in policy_data:
            cfg.command_history_length = policy_data["action_history_length"]
        return cfg


class RLObservationAdapter:
    """Convert the 11-D benchmark observation to an RL policy input."""

    LEGACY8_FIELDS = (
        "error",
        "error_d1",
        "error_d2",
        "vb",
        "theta",
        "omega",
        "vrz",
        "a_prev",
    )
    FULL11_FIELDS = (
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
    ERROR10_FIELDS = (
        "error",
        "vb",
        "ab",
        "theta",
        "omega",
        "alpha",
        "drz",
        "vrz",
        "arz",
        "a_prev",
    )
    ERROR9_FIELDS = (
        "error",
        "vb",
        "ab",
        "theta",
        "omega",
        "alpha",
        "vrz",
        "arz",
        "a_prev",
    )
    SUPPORTED_MODES = {"legacy8", "full11", "error10", "error9", "error9_acc_history"}

    def __init__(self, cfg: RLObservationAdapterCfg, num_envs: int, device: str | torch.device):
        self.cfg = cfg
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.observation_mode = str(cfg.observation_mode).lower()
        if self.observation_mode not in self.SUPPORTED_MODES:
            supported = "', '".join(sorted(self.SUPPORTED_MODES))
            raise ValueError(f"RL observation_mode must be one of '{supported}'.")
        self.command_history_length = self._parse_command_history_length(cfg.command_history_length)
        if self.observation_mode == "error9_acc_history" and self.command_history_length <= 0:
            raise ValueError(
                "RL observation_mode='error9_acc_history' requires command_history_length > 0 "
                "after resolving any 'auto' value."
            )
        if self.observation_mode != "error9_acc_history" and self.command_history_length != 0:
            raise ValueError("command_history_length is only supported with observation_mode='error9_acc_history'.")

        self.error_prev1 = torch.zeros((self.num_envs, 1), device=self.device)
        self.error_prev2 = torch.zeros((self.num_envs, 1), device=self.device)
        self.history_needs_init = torch.ones((self.num_envs,), dtype=torch.bool, device=self.device)
        self.command_history = torch.zeros((self.num_envs, self.command_history_length), device=self.device)
        self.last_policy_input = torch.zeros((self.num_envs, self.input_dim), device=self.device)
        self.error = torch.zeros((self.num_envs, 1), device=self.device)
        self.error_d1 = torch.zeros((self.num_envs, 1), device=self.device)
        self.error_d2 = torch.zeros((self.num_envs, 1), device=self.device)

    @property
    def input_dim(self) -> int:
        if self.observation_mode == "legacy8":
            return 8
        if self.observation_mode in {"error9", "error9_acc_history"}:
            return 9 + self.command_history_length
        if self.observation_mode == "error10":
            return 10
        return 11

    @property
    def field_names(self) -> tuple[str, ...]:
        if self.observation_mode == "legacy8":
            return self.LEGACY8_FIELDS
        if self.observation_mode == "error9":
            return self.ERROR9_FIELDS
        if self.observation_mode == "error9_acc_history":
            history_fields = tuple(
                f"arz_cmd_prev{history_index}" for history_index in range(1, self.command_history_length + 1)
            )
            return self.ERROR9_FIELDS + history_fields
        if self.observation_mode == "error10":
            return self.ERROR10_FIELDS
        return self.FULL11_FIELDS

    def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None):
        """Reset history for all envs or a selected subset."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        else:
            env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        if env_ids.numel() == 0:
            return
        self.error_prev1[env_ids] = 0.0
        self.error_prev2[env_ids] = 0.0
        self.error[env_ids] = 0.0
        self.error_d1[env_ids] = 0.0
        self.error_d2[env_ids] = 0.0
        self.command_history[env_ids] = 0.0
        self.last_policy_input[env_ids] = 0.0
        self.history_needs_init[env_ids] = True

    def update_command_history(self, command_z: torch.Tensor):
        """Record the latest absolute Z-acceleration command for delay-aware observations."""
        if self.command_history_length <= 0:
            return
        command_z = command_z.to(device=self.device, dtype=torch.float32)
        if command_z.ndim == 1:
            command_z = command_z.unsqueeze(-1)
        elif command_z.ndim != 2:
            raise ValueError(
                "RLObservationAdapter.update_command_history expects shape "
                f"({self.num_envs},), ({self.num_envs}, 1), or ({self.num_envs}, 3); "
                f"got {tuple(command_z.shape)}."
            )
        if command_z.shape == (self.num_envs, 3):
            command_z = command_z[:, 2:3]
        if command_z.shape != (self.num_envs, 1):
            raise ValueError(
                "RLObservationAdapter.update_command_history expects command shape "
                f"({self.num_envs}, 1), got {tuple(command_z.shape)}."
            )
        if self.command_history_length > 1:
            self.command_history[:, 1:] = self.command_history[:, :-1].clone()
        self.command_history[:, 0:1] = command_z

    def transform(self, observation: torch.Tensor, update_history: bool = True) -> torch.Tensor:
        """Map a benchmark observation tensor to the configured RL input."""
        observation = observation.to(device=self.device, dtype=torch.float32)
        if observation.shape != (self.num_envs, 11):
            raise ValueError(
                f"RLObservationAdapter expects observation shape ({self.num_envs}, 11), "
                f"got {tuple(observation.shape)}."
            )

        if self.observation_mode == "full11":
            self.last_policy_input.copy_(observation)
            return observation

        pb = observation[:, ObservationIndex.PB : ObservationIndex.PB + 1]
        pg = observation[:, ObservationIndex.PG : ObservationIndex.PG + 1]
        error = pb - pg

        if update_history:
            init_envs = self.history_needs_init.clone()
            if torch.any(init_envs):
                self.error_prev1[init_envs] = error[init_envs]
                self.error_prev2[init_envs] = error[init_envs]

        error_d1 = error - self.error_prev1
        error_d2 = error - 2.0 * self.error_prev1 + self.error_prev2
        if self.observation_mode in {"error9", "error9_acc_history"}:
            error9_input = torch.cat(
                [
                    error,
                    observation[:, ObservationIndex.VB : ObservationIndex.DRZ],
                    observation[:, ObservationIndex.VRZ : ObservationIndex.PG],
                    observation[:, ObservationIndex.A_PREV : ObservationIndex.A_PREV + 1],
                ],
                dim=-1,
            )
            if self.observation_mode == "error9_acc_history":
                policy_input = torch.cat([error9_input, self.command_history], dim=-1)
            else:
                policy_input = error9_input
        elif self.observation_mode == "error10":
            policy_input = torch.cat(
                [
                    error,
                    observation[:, ObservationIndex.VB : ObservationIndex.PG],
                    observation[:, ObservationIndex.A_PREV : ObservationIndex.A_PREV + 1],
                ],
                dim=-1,
            )
        else:
            policy_input = torch.cat(
                [
                    error,
                    error_d1,
                    error_d2,
                    observation[:, ObservationIndex.VB : ObservationIndex.VB + 1],
                    observation[:, ObservationIndex.THETA : ObservationIndex.THETA + 1],
                    observation[:, ObservationIndex.OMEGA : ObservationIndex.OMEGA + 1],
                    observation[:, ObservationIndex.VRZ : ObservationIndex.VRZ + 1],
                    observation[:, ObservationIndex.A_PREV : ObservationIndex.A_PREV + 1],
                ],
                dim=-1,
            )

        self.error.copy_(error)
        self.error_d1.copy_(error_d1)
        self.error_d2.copy_(error_d2)
        self.last_policy_input.copy_(policy_input)

        if update_history:
            self.error_prev2.copy_(self.error_prev1)
            self.error_prev1.copy_(error)
            if torch.any(self.history_needs_init):
                self.history_needs_init[:] = False
        return policy_input

    def get_state(self) -> dict[str, torch.Tensor]:
        """Return adapter diagnostics."""
        state = {
            "rl_adapter_error": self.error[:, 0],
            "rl_adapter_error_d1": self.error_d1[:, 0],
            "rl_adapter_error_d2": self.error_d2[:, 0],
            "rl_policy_input": self.last_policy_input,
        }
        if self.command_history_length > 0:
            state["rl_adapter_command_history"] = self.command_history
        return state

    def to(self, device: str | torch.device):
        """Move adapter buffers to a device and return self."""
        device = torch.device(device)
        for name, value in vars(self).items():
            if isinstance(value, torch.Tensor):
                setattr(self, name, value.to(device=device))
        self.device = device
        return self

    @staticmethod
    def _parse_command_history_length(value: int | str) -> int:
        if isinstance(value, str):
            if value.lower() == "auto":
                raise ValueError("command_history_length='auto' must be resolved before constructing RLObservationAdapter.")
            value = int(value)
        length = int(value)
        if length < 0:
            raise ValueError("command_history_length must be non-negative.")
        return length
