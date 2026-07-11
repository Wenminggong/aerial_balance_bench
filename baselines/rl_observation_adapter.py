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

    @classmethod
    def from_dict(cls, data: Mapping | None) -> "RLObservationAdapterCfg":
        cfg = cls()
        if not data:
            return cfg
        policy_data = data.get("rl_policy", data.get("policy", data))
        if "observation_mode" in policy_data:
            cfg.observation_mode = str(policy_data["observation_mode"])
        return cfg


class RLObservationAdapter:
    """Convert raw benchmark observations to an RL policy input.

    The first 11 raw fields retain the original benchmark layout.  Optional
    reference-preview fields are appended as ``vg_0, pg_1, vg_1, ...``.
    """

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
    REFERENCE_PREVIEW_PLANT_FIELDS = FULL11_FIELDS[:9]

    def __init__(
        self,
        cfg: RLObservationAdapterCfg,
        num_envs: int,
        device: str | torch.device,
        raw_observation_dim: int | None = None,
        raw_observation_fields: Sequence[str] | None = None,
    ):
        self.cfg = cfg
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.observation_mode = str(cfg.observation_mode).lower()
        if self.observation_mode not in {"legacy8", "full11", "reference_preview"}:
            raise ValueError(
                "RL observation_mode must be 'legacy8', 'full11', or 'reference_preview'."
            )

        self.raw_observation_dim, self.raw_observation_fields = self._resolve_raw_observation_spec(
            raw_observation_dim,
            raw_observation_fields,
        )
        self.reference_preview_future_steps: int | None = None
        if self.observation_mode == "reference_preview":
            self.reference_preview_future_steps = self._validate_reference_preview_spec()

        self.error_prev1 = torch.zeros((self.num_envs, 1), device=self.device)
        self.error_prev2 = torch.zeros((self.num_envs, 1), device=self.device)
        self.history_needs_init = torch.ones((self.num_envs,), dtype=torch.bool, device=self.device)
        self.last_policy_input = torch.zeros((self.num_envs, self.input_dim), device=self.device)
        self.error = torch.zeros((self.num_envs, 1), device=self.device)
        self.error_d1 = torch.zeros((self.num_envs, 1), device=self.device)
        self.error_d2 = torch.zeros((self.num_envs, 1), device=self.device)

    @property
    def input_dim(self) -> int:
        if self.observation_mode == "legacy8":
            return 8
        if self.observation_mode == "full11":
            return 11
        return self.raw_observation_dim

    @property
    def field_names(self) -> tuple[str, ...]:
        if self.observation_mode == "legacy8":
            return self.LEGACY8_FIELDS
        if self.observation_mode == "full11":
            return self.FULL11_FIELDS
        assert self.reference_preview_future_steps is not None
        fields = [*self.REFERENCE_PREVIEW_PLANT_FIELDS, "a_prev", "pg_0", "vg_0"]
        for offset in range(1, self.reference_preview_future_steps + 1):
            fields.extend((f"pg_{offset}", f"vg_{offset}"))
        return tuple(fields)

    @property
    def preview_offsets(self) -> tuple[int, ...]:
        """Return reference offsets consumed by the preview policy input."""
        if self.reference_preview_future_steps is None:
            return ()
        return tuple(range(self.reference_preview_future_steps + 1))

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
        self.last_policy_input[env_ids] = 0.0
        self.history_needs_init[env_ids] = True

    def transform(self, observation: torch.Tensor, update_history: bool = True) -> torch.Tensor:
        """Map a benchmark observation tensor to the configured RL input."""
        observation = observation.to(device=self.device, dtype=torch.float32)
        if observation.ndim != 2 or observation.shape[0] != self.num_envs or observation.shape[1] < 11:
            raise ValueError(
                f"RLObservationAdapter expects observation shape ({self.num_envs}, D) with D >= 11, "
                f"got {tuple(observation.shape)}."
            )

        if self.observation_mode == "reference_preview":
            if observation.shape[1] != self.raw_observation_dim:
                raise ValueError(
                    "RLObservationAdapter reference_preview input dimension does not match its raw "
                    f"observation spec: expected {self.raw_observation_dim}, got {observation.shape[1]}."
                )
            policy_input = torch.cat(
                (
                    observation[:, :9],
                    observation[:, ObservationIndex.A_PREV : ObservationIndex.A_PREV + 1],
                    observation[:, ObservationIndex.PG : ObservationIndex.PG + 1],
                    observation[:, 11:],
                ),
                dim=-1,
            )
            self.last_policy_input.copy_(policy_input)
            return policy_input

        if self.observation_mode == "full11":
            legacy_observation = observation[:, :11]
            self.last_policy_input.copy_(legacy_observation)
            return legacy_observation

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

    def _resolve_raw_observation_spec(
        self,
        raw_observation_dim: int | None,
        raw_observation_fields: Sequence[str] | None,
    ) -> tuple[int, tuple[str, ...]]:
        if isinstance(raw_observation_fields, (str, bytes)):
            raise TypeError("raw_observation_fields must be a sequence of field names, not a string.")
        fields = None if raw_observation_fields is None else tuple(str(name) for name in raw_observation_fields)

        if raw_observation_dim is None:
            raw_observation_dim = len(fields) if fields is not None else 11
        raw_observation_dim = int(raw_observation_dim)
        if raw_observation_dim < 11:
            raise ValueError(f"raw_observation_dim must be at least 11, got {raw_observation_dim}.")
        if fields is not None and len(fields) != raw_observation_dim:
            raise ValueError(
                "raw_observation_fields length must match raw_observation_dim: "
                f"got {len(fields)} fields for dimension {raw_observation_dim}."
            )
        if fields is None:
            fields = self._default_raw_observation_fields(raw_observation_dim)
        return raw_observation_dim, fields

    def _validate_reference_preview_spec(self) -> int:
        extra_dim = self.raw_observation_dim - 12
        if extra_dim < 0 or extra_dim % 2 != 0:
            raise ValueError(
                "reference_preview requires raw layout [legacy11, vg_0, pg_1, vg_1, ...] "
                "with dimension 12 + 2 * future_steps; "
                f"got dimension {self.raw_observation_dim}."
            )

        future_steps = extra_dim // 2
        expected_fields = self._default_raw_observation_fields(self.raw_observation_dim)
        normalized_actual = list(self.raw_observation_fields)
        # The legacy environment calls the current target field ``pg``.  Also
        # accept the explicit preview spelling ``pg_0`` when a caller provides
        # its own observation spec.
        if normalized_actual[ObservationIndex.PG] == "pg_0":
            normalized_actual[ObservationIndex.PG] = "pg"
        if tuple(normalized_actual) != expected_fields:
            raise ValueError(
                "reference_preview raw_observation_fields must follow "
                "[pb, vb, ab, theta, omega, alpha, drz, vrz, arz, pg, a_prev, "
                "vg_0, pg_1, vg_1, ...]; "
                f"got {self.raw_observation_fields}."
            )
        return future_steps

    @classmethod
    def _default_raw_observation_fields(cls, raw_observation_dim: int) -> tuple[str, ...]:
        fields = list(cls.FULL11_FIELDS)
        extra_dim = raw_observation_dim - 12
        if extra_dim >= 0 and extra_dim % 2 == 0:
            fields.append("vg_0")
            for offset in range(1, extra_dim // 2 + 1):
                fields.extend((f"pg_{offset}", f"vg_{offset}"))
        else:
            fields.extend(f"extra_{index}" for index in range(raw_observation_dim - 11))
        return tuple(fields)

    def get_state(self) -> dict[str, torch.Tensor]:
        """Return adapter diagnostics."""
        state = {
            "rl_adapter_error": self.error[:, 0],
            "rl_adapter_error_d1": self.error_d1[:, 0],
            "rl_adapter_error_d2": self.error_d2[:, 0],
            "rl_policy_input": self.last_policy_input,
        }
        return state

    def to(self, device: str | torch.device):
        """Move adapter buffers to a device and return self."""
        device = torch.device(device)
        for name, value in vars(self).items():
            if isinstance(value, torch.Tensor):
                setattr(self, name, value.to(device=device))
        self.device = device
        return self
