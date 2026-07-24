"""Observation adapters used by RL baselines."""

from __future__ import annotations

import operator
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch

from .base_policy import ObservationIndex


@dataclass
class RLObservationAdapterCfg:
    """Configuration for mapping benchmark observations to RL inputs."""

    observation_mode: str = "legacy8"
    reference_preview_samples: int | None = None

    @classmethod
    def from_dict(cls, data: Mapping | None) -> "RLObservationAdapterCfg":
        cfg = cls()
        if not data:
            return cfg
        policy_data = data.get("rl_policy", data.get("policy", data))
        if "observation_mode" in policy_data:
            cfg.observation_mode = str(policy_data["observation_mode"])
        if "reference_preview_samples" in policy_data:
            cfg.reference_preview_samples = policy_data["reference_preview_samples"]
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
    RELATIVE_REFERENCE_PREVIEW_BASE_FIELDS = (
        "e_b",
        "e_vb",
        "ab",
        "theta",
        "omega",
        "alpha",
        "vrz",
        "arz",
        "a_prev",
    )
    PREVIEW_OBSERVATION_MODES = {"reference_preview", "relative_reference_preview"}

    def __init__(
        self,
        cfg: RLObservationAdapterCfg,
        num_envs: int,
        device: str | torch.device,
        raw_observation_dim: int | None = None,
        raw_observation_fields: Sequence[str] | None = None,
        reference_preview_base_offset: int = 0,
    ):
        self.cfg = cfg
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.observation_mode = str(cfg.observation_mode).lower()
        if self.observation_mode not in {
            "legacy8",
            "full11",
            *self.PREVIEW_OBSERVATION_MODES,
        }:
            raise ValueError(
                "RL observation_mode must be 'legacy8', 'full11', 'reference_preview', "
                "or 'relative_reference_preview'."
            )

        self.raw_observation_dim, self.raw_observation_fields = self._resolve_raw_observation_spec(
            raw_observation_dim,
            raw_observation_fields,
        )
        self.reference_preview_base_offset = self._resolve_reference_preview_base_offset(
            reference_preview_base_offset
        )
        if self.observation_mode != "relative_reference_preview" and self.reference_preview_base_offset != 0:
            raise ValueError(
                "reference_preview_base_offset is only supported for "
                "observation_mode='relative_reference_preview'."
            )
        self.reference_preview_future_steps: int | None = None
        self.effective_reference_preview_future_steps: int | None = None
        self.reference_preview_samples: int | None = None
        self.sampled_reference_preview_offsets: tuple[int, ...] = ()
        self.sampled_reference_preview_source_offsets: tuple[int, ...] = ()
        self._vg_base_index: int | None = None
        self._sampled_pg_indices: tuple[int, ...] = ()
        self._sampled_vg_indices: tuple[int, ...] = ()
        if self.observation_mode in self.PREVIEW_OBSERVATION_MODES:
            self.reference_preview_future_steps = self._validate_reference_preview_spec()
        if self.observation_mode == "relative_reference_preview":
            (
                self.reference_preview_samples,
                self.sampled_reference_preview_offsets,
                self.sampled_reference_preview_source_offsets,
            ) = self._resolve_relative_reference_preview_sampling()
            self._vg_base_index = self.raw_observation_fields.index(
                f"vg_{self.reference_preview_base_offset}"
            )
            self._sampled_pg_indices = tuple(
                self.raw_observation_fields.index(f"pg_{offset}")
                for offset in self.sampled_reference_preview_source_offsets
            )
            self._sampled_vg_indices = tuple(
                self.raw_observation_fields.index(f"vg_{offset}")
                for offset in self.sampled_reference_preview_source_offsets
            )

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
        if self.observation_mode == "relative_reference_preview":
            assert self.reference_preview_samples is not None
            return len(self.RELATIVE_REFERENCE_PREVIEW_BASE_FIELDS) + 2 * self.reference_preview_samples
        return self.raw_observation_dim

    @property
    def field_names(self) -> tuple[str, ...]:
        if self.observation_mode == "legacy8":
            return self.LEGACY8_FIELDS
        if self.observation_mode == "full11":
            return self.FULL11_FIELDS
        if self.observation_mode == "relative_reference_preview":
            fields = list(self.RELATIVE_REFERENCE_PREVIEW_BASE_FIELDS)
            fields.extend(f"delta_pg_{offset}" for offset in self.sampled_reference_preview_offsets)
            fields.extend(f"delta_vg_{offset}" for offset in self.sampled_reference_preview_offsets)
            return tuple(fields)
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
        if self.observation_mode == "relative_reference_preview":
            return self.sampled_reference_preview_offsets
        return tuple(range(self.reference_preview_future_steps + 1))

    @property
    def preview_source_offsets(self) -> tuple[int, ...]:
        """Return raw reference offsets backing the preview policy input."""
        if self.reference_preview_future_steps is None:
            return ()
        if self.observation_mode == "relative_reference_preview":
            return self.sampled_reference_preview_source_offsets
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

        if self.observation_mode in self.PREVIEW_OBSERVATION_MODES:
            if observation.shape[1] != self.raw_observation_dim:
                raise ValueError(
                    f"RLObservationAdapter {self.observation_mode} input dimension does not match "
                    "its raw observation spec: "
                    f"expected {self.raw_observation_dim}, got {observation.shape[1]}."
                )

        if self.observation_mode == "relative_reference_preview":
            pb = observation[:, ObservationIndex.PB : ObservationIndex.PB + 1]
            vb = observation[:, ObservationIndex.VB : ObservationIndex.VB + 1]
            pg_base = observation[:, ObservationIndex.PG : ObservationIndex.PG + 1]
            assert self._vg_base_index is not None
            vg_base = observation[:, self._vg_base_index : self._vg_base_index + 1]
            error = pb - pg_base
            velocity_error = vb - vg_base
            sampled_pg = observation[:, self._sampled_pg_indices]
            sampled_vg = observation[:, self._sampled_vg_indices]
            policy_input = torch.cat(
                (
                    error,
                    velocity_error,
                    observation[:, ObservationIndex.AB : ObservationIndex.AB + 1],
                    observation[:, ObservationIndex.THETA : ObservationIndex.THETA + 1],
                    observation[:, ObservationIndex.OMEGA : ObservationIndex.OMEGA + 1],
                    observation[:, ObservationIndex.ALPHA : ObservationIndex.ALPHA + 1],
                    observation[:, ObservationIndex.VRZ : ObservationIndex.VRZ + 1],
                    observation[:, ObservationIndex.ARZ : ObservationIndex.ARZ + 1],
                    observation[:, ObservationIndex.A_PREV : ObservationIndex.A_PREV + 1],
                    sampled_pg - pg_base,
                    sampled_vg - vg_base,
                ),
                dim=-1,
            )
            self.error.copy_(error)
            self.last_policy_input.copy_(policy_input)
            return policy_input

        if self.observation_mode == "reference_preview":
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

    def _resolve_relative_reference_preview_sampling(
        self,
    ) -> tuple[int, tuple[int, ...], tuple[int, ...]]:
        raw_future_steps = self.reference_preview_future_steps
        assert raw_future_steps is not None
        base_offset = self.reference_preview_base_offset
        if base_offset > raw_future_steps:
            raise ValueError(
                "relative_reference_preview requires reference_preview_base_offset <= raw "
                f"reference_preview.future_steps; got D={base_offset}, H_raw={raw_future_steps}."
            )
        effective_future_steps = raw_future_steps - base_offset
        self.effective_reference_preview_future_steps = effective_future_steps
        configured_samples = self.cfg.reference_preview_samples
        if configured_samples is None:
            raise ValueError(
                "relative_reference_preview requires policy configuration "
                "reference_preview_samples=K with a positive integer value."
            )
        if isinstance(configured_samples, bool):
            raise TypeError("reference_preview_samples must be a positive integer, not bool.")
        try:
            samples = operator.index(configured_samples)
        except TypeError as exc:
            raise TypeError("reference_preview_samples must be a positive integer.") from exc
        if samples <= 0:
            raise ValueError(
                "relative_reference_preview requires reference_preview_samples > 0; "
                f"got K={samples}."
            )
        if effective_future_steps < samples:
            raise ValueError(
                "relative_reference_preview requires effective preview horizon "
                "H_effective=H_raw-D >= reference_preview_samples; "
                f"got H_raw={raw_future_steps}, D={base_offset}, "
                f"H_effective={effective_future_steps}, K={samples}."
            )
        if effective_future_steps % samples != 0:
            raise ValueError(
                "relative_reference_preview requires effective preview horizon "
                "H_effective=H_raw-D to be an integer multiple of reference_preview_samples "
                "for uniform sampling; "
                f"got H_raw={raw_future_steps}, D={base_offset}, "
                f"H_effective={effective_future_steps}, K={samples}."
            )
        stride = effective_future_steps // samples
        offsets = tuple(stride * sample for sample in range(1, samples + 1))
        source_offsets = tuple(base_offset + offset for offset in offsets)
        return samples, offsets, source_offsets

    @staticmethod
    def _resolve_reference_preview_base_offset(value: int) -> int:
        if isinstance(value, bool):
            raise TypeError("reference_preview_base_offset must be a non-negative integer, not bool.")
        try:
            offset = operator.index(value)
        except TypeError as exc:
            raise TypeError("reference_preview_base_offset must be a non-negative integer.") from exc
        if offset < 0:
            raise ValueError(
                "reference_preview_base_offset must be non-negative; "
                f"got D={offset}."
            )
        return offset

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
