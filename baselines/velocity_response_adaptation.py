"""History encoding utilities for online velocity-response adaptation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


ADAPTIVE_CHECKPOINT_FORMAT_VERSION = 2
ACTOR_CRITIC_COMPONENT = "adaptive_actor_critic_heads"
ENCODER_COMPONENT = "velocity_response_encoder"
SUPPORTED_ENCODER_ACTIVATIONS = {"elu", "relu", "silu", "tanh"}
PRIVILEGED_RESPONSE_FIELDS = (
    "velocity_response_tau_s",
    "velocity_response_gain",
    "velocity_response_bias",
)


@dataclass
class VelocityResponseEncoderCfg:
    """Configurable MLP used to encode measured and commanded velocity histories."""

    hidden_dims: tuple[int, ...] = (64, 32)
    activation: str = "relu"

    @classmethod
    def from_dict(cls, data: Mapping | None) -> "VelocityResponseEncoderCfg":
        cfg = cls()
        if not data:
            return cfg
        if "hidden_dims" in data:
            hidden_dims = data["hidden_dims"]
            if isinstance(hidden_dims, (str, bytes)) or not isinstance(hidden_dims, Sequence):
                raise TypeError("velocity_response_adaptation.encoder.hidden_dims must be a sequence.")
            cfg.hidden_dims = tuple(int(value) for value in hidden_dims)
        if "activation" in data:
            cfg.activation = str(data["activation"]).lower()
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if any(isinstance(value, bool) or int(value) <= 0 for value in self.hidden_dims):
            raise ValueError(
                "velocity_response_adaptation.encoder.hidden_dims must contain positive integers."
            )
        self.hidden_dims = tuple(int(value) for value in self.hidden_dims)
        self.activation = str(self.activation).lower()
        if self.activation not in SUPPORTED_ENCODER_ACTIVATIONS:
            supported = ", ".join(sorted(SUPPORTED_ENCODER_ACTIVATIONS))
            raise ValueError(
                "velocity_response_adaptation.encoder.activation must be one of "
                f"{supported}; got '{self.activation}'."
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "hidden_dims": list(self.hidden_dims),
            "activation": self.activation,
        }


@dataclass
class VelocityResponseAdaptationCfg:
    """Configuration for history-conditioned RL velocity-response adaptation."""

    enabled: bool = False
    history_length: int = 30
    latent_dim: int = 8
    encoder: VelocityResponseEncoderCfg = field(default_factory=VelocityResponseEncoderCfg)

    @classmethod
    def from_dict(cls, data: Mapping | None) -> "VelocityResponseAdaptationCfg":
        cfg = cls()
        if not data:
            return cfg
        if "enabled" in data:
            cfg.enabled = bool(data["enabled"])
        if "history_length" in data:
            cfg.history_length = data["history_length"]
        if "latent_dim" in data:
            cfg.latent_dim = data["latent_dim"]
        cfg.encoder = VelocityResponseEncoderCfg.from_dict(data.get("encoder", {}))
        cfg.validate()
        return cfg

    def validate(self) -> None:
        self.history_length = _positive_integer(self.history_length, "history_length")
        self.latent_dim = _positive_integer(self.latent_dim, "latent_dim")
        self.encoder.validate()

    @property
    def actual_history_field_names(self) -> tuple[str, ...]:
        return tuple(
            f"velocity_response_vrz_t_minus_{offset}"
            for offset in range(self.history_length, 0, -1)
        )

    @property
    def command_history_field_names(self) -> tuple[str, ...]:
        return tuple(
            f"velocity_response_command_z_t_minus_{offset}"
            for offset in range(self.history_length, 0, -1)
        )

    @property
    def history_field_names(self) -> tuple[str, ...]:
        """Return encoder fields in channel-major, oldest-to-newest order."""
        return (*self.actual_history_field_names, *self.command_history_field_names)

    @property
    def encoder_input_dim(self) -> int:
        return 2 * self.history_length

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "history_length": self.history_length,
            "latent_dim": self.latent_dim,
            "encoder": self.encoder.to_dict(),
        }


class VelocityResponseHistoryBuffer:
    """Vectorized measured and commanded histories in oldest-to-newest order."""

    def __init__(
        self,
        num_envs: int,
        history_length: int,
        device: str | torch.device,
    ):
        self.num_envs = int(num_envs)
        self.history_length = _positive_integer(history_length, "history_length")
        self.device = torch.device(device)
        self.actual_velocity_history = torch.zeros(
            (self.num_envs, self.history_length),
            device=self.device,
            dtype=torch.float32,
        )
        self.commanded_velocity_history = torch.zeros(
            (self.num_envs, self.history_length),
            device=self.device,
            dtype=torch.float32,
        )
        # Deployment receives reset infos before it receives a real transition.
        # This mask prevents those reset infos (and terminal infos after an
        # autoreset) from being paired with the new episode's observation.
        self.needs_transition = torch.ones(
            (self.num_envs,),
            device=self.device,
            dtype=torch.bool,
        )

    @property
    def encoder_input(self) -> torch.Tensor:
        return torch.cat(
            (self.actual_velocity_history, self.commanded_velocity_history),
            dim=-1,
        )

    @property
    def error_history(self) -> torch.Tensor:
        return self.actual_velocity_history - self.commanded_velocity_history

    @property
    def latest_actual_velocity(self) -> torch.Tensor:
        return self.actual_velocity_history[:, -1]

    @property
    def latest_commanded_velocity(self) -> torch.Tensor:
        return self.commanded_velocity_history[:, -1]

    @property
    def latest_error(self) -> torch.Tensor:
        return self.latest_actual_velocity - self.latest_commanded_velocity

    def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> None:
        env_ids = self._env_ids(env_ids)
        if env_ids.numel() == 0:
            return
        self.actual_velocity_history[env_ids] = 0.0
        self.commanded_velocity_history[env_ids] = 0.0
        self.needs_transition[env_ids] = True

    def mark_transition_boundary(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> None:
        """Mark reset observations as consumed without adding a history sample."""
        env_ids = self._env_ids(env_ids)
        if env_ids.numel() > 0:
            self.needs_transition[env_ids] = False

    def update(
        self,
        actual_velocity: torch.Tensor,
        commanded_velocity: torch.Tensor,
        env_ids: Sequence[int] | torch.Tensor | None = None,
    ) -> None:
        env_ids = self._env_ids(env_ids)
        if env_ids.numel() == 0:
            return
        actual_values = self._selected_values(actual_velocity, env_ids, "actual_velocity")
        command_values = self._selected_values(
            commanded_velocity,
            env_ids,
            "commanded_velocity",
        )
        if self.history_length > 1:
            self.actual_velocity_history[env_ids, :-1] = self.actual_velocity_history[
                env_ids, 1:
            ].clone()
            self.commanded_velocity_history[env_ids, :-1] = self.commanded_velocity_history[
                env_ids, 1:
            ].clone()
        self.actual_velocity_history[env_ids, -1] = actual_values
        self.commanded_velocity_history[env_ids, -1] = command_values
        self.needs_transition[env_ids] = False

    def _selected_values(
        self,
        values: torch.Tensor,
        env_ids: torch.Tensor,
        field_name: str,
    ) -> torch.Tensor:
        values = torch.as_tensor(values, device=self.device, dtype=torch.float32)
        if values.ndim == 2 and values.shape[-1] == 1:
            values = values[:, 0]
        if values.shape == (self.num_envs,):
            values = values[env_ids]
        elif values.shape != (env_ids.numel(),):
            raise ValueError(
                f"VelocityResponseHistoryBuffer {field_name} must have shape "
                f"({self.num_envs},), ({self.num_envs}, 1), ({env_ids.numel()},), or "
                f"({env_ids.numel()}, 1); got {tuple(values.shape)}."
            )
        if not torch.isfinite(values).all():
            raise ValueError(f"Velocity-response history {field_name} values must be finite.")
        return values

    def to(self, device: str | torch.device) -> "VelocityResponseHistoryBuffer":
        self.device = torch.device(device)
        self.actual_velocity_history = self.actual_velocity_history.to(device=self.device)
        self.commanded_velocity_history = self.commanded_velocity_history.to(device=self.device)
        self.needs_transition = self.needs_transition.to(device=self.device)
        return self

    def _env_ids(self, env_ids: Sequence[int] | torch.Tensor | None) -> torch.Tensor:
        if env_ids is None:
            return torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        return torch.as_tensor(env_ids, device=self.device, dtype=torch.long)


class VelocityResponseEncoder(nn.Module):
    """MLP mapping normalized H-step measured/command histories to a latent code."""

    def __init__(
        self,
        history_length: int,
        latent_dim: int,
        cfg: VelocityResponseEncoderCfg,
    ):
        super().__init__()
        self.history_length = _positive_integer(history_length, "history_length")
        self.input_dim = 2 * self.history_length
        self.latent_dim = _positive_integer(latent_dim, "latent_dim")
        cfg.validate()
        self.cfg = cfg

        layers: list[nn.Module] = []
        input_dim = self.input_dim
        activation_cls = _activation_class(cfg.activation)
        for hidden_dim in cfg.hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(activation_cls())
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, self.latent_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, normalized_velocity_history: torch.Tensor) -> torch.Tensor:
        if normalized_velocity_history.shape[-1] != self.input_dim:
            raise ValueError(
                "VelocityResponseEncoder expected history dimension "
                f"{self.input_dim} (2H), got {normalized_velocity_history.shape[-1]}."
            )
        return self.net(normalized_velocity_history)


class ReusableVelocityResponseEncoder(nn.Module):
    """Standalone raw-history encoder restored from a component checkpoint."""

    def __init__(
        self,
        encoder: VelocityResponseEncoder,
        running_mean: torch.Tensor,
        running_variance: torch.Tensor,
        *,
        epsilon: float = 1.0e-8,
        clip_threshold: float = 5.0,
        metadata: Mapping[str, Any] | None = None,
    ):
        super().__init__()
        self.encoder = encoder
        self.epsilon = float(epsilon)
        self.clip_threshold = float(clip_threshold)
        self.metadata = dict(metadata or {})
        self.register_buffer("running_mean", running_mean.to(dtype=torch.float32))
        self.register_buffer("running_variance", running_variance.to(dtype=torch.float32))

    def forward(self, raw_velocity_history: torch.Tensor) -> torch.Tensor:
        if raw_velocity_history.ndim != 2:
            raise ValueError(
                "ReusableVelocityResponseEncoder expects raw history shape (batch, 2H); "
                f"got {tuple(raw_velocity_history.shape)}."
            )
        raw_velocity_history = raw_velocity_history.to(
            device=self.running_mean.device,
            dtype=torch.float32,
        )
        normalized = (raw_velocity_history - self.running_mean) / (
            torch.sqrt(self.running_variance) + self.epsilon
        )
        normalized = torch.clamp(
            normalized,
            min=-self.clip_threshold,
            max=self.clip_threshold,
        )
        return self.encoder(normalized)


def load_velocity_response_encoder(
    path: str | Path,
    device: str | torch.device = "cpu",
    expected_metadata: Mapping[str, Any] | None = None,
) -> ReusableVelocityResponseEncoder:
    """Load an independently reusable encoder from an adaptive checkpoint."""
    payload = _torch_load(path, device)
    _validate_component_payload(payload, ENCODER_COMPONENT)
    metadata = payload["metadata"]
    validate_adaptive_checkpoint_metadata_structure(metadata)
    if expected_metadata is not None:
        validate_adaptive_checkpoint_metadata(metadata, expected_metadata)
    adaptation = metadata.get("velocity_response_adaptation", {})
    encoder_cfg = VelocityResponseEncoderCfg.from_dict(adaptation.get("encoder", {}))
    history_length = _positive_integer(adaptation.get("history_length"), "history_length")
    latent_dim = _positive_integer(adaptation.get("latent_dim"), "latent_dim")
    encoder = VelocityResponseEncoder(history_length, latent_dim, encoder_cfg).to(device=device)
    encoder.load_state_dict(payload["encoder_state_dict"])
    normalization = payload.get("history_normalization", {})
    running_mean_data = normalization.get("running_mean")
    running_variance_data = normalization.get("running_variance")
    if running_mean_data is None or running_variance_data is None:
        raise ValueError(
            "Velocity-response encoder checkpoint is missing history normalization statistics."
        )
    running_mean = torch.as_tensor(running_mean_data, device=device)
    running_variance = torch.as_tensor(running_variance_data, device=device)
    encoder_input_dim = 2 * history_length
    if running_mean.shape != (encoder_input_dim,) or running_variance.shape != (
        encoder_input_dim,
    ):
        raise ValueError(
            "Velocity-response encoder checkpoint has invalid history normalization shape."
        )
    model = ReusableVelocityResponseEncoder(
        encoder,
        running_mean,
        running_variance,
        epsilon=float(normalization.get("epsilon", 1.0e-8)),
        clip_threshold=float(normalization.get("clip_threshold", 5.0)),
        metadata=metadata,
    ).to(device=device)
    model.eval()
    return model


def adaptive_checkpoint_metadata(
    *,
    algorithm: str,
    observation_mode: str,
    base_input_dim: int,
    total_input_dim: int,
    input_fields: Sequence[str],
    adaptation_cfg: VelocityResponseAdaptationCfg,
) -> dict[str, Any]:
    """Build the stable metadata shared by adaptive checkpoint components."""
    metadata = {
        "algorithm": str(algorithm).lower(),
        "observation_mode": str(observation_mode).lower(),
        "base_input_dim": int(base_input_dim),
        "total_input_dim": int(total_input_dim),
        "input_fields": list(input_fields),
        "actor_observation_dim": int(base_input_dim) + adaptation_cfg.encoder_input_dim,
        "actor_head_input_dim": int(base_input_dim) + int(adaptation_cfg.latent_dim),
        "critic_input_dim": int(base_input_dim) + len(PRIVILEGED_RESPONSE_FIELDS),
        "encoder_input_layout": "vrz_history_then_command_z_history",
        "history_order": "oldest_to_newest",
        "critic_privileged_fields": list(PRIVILEGED_RESPONSE_FIELDS),
        "velocity_response_adaptation": adaptation_cfg.to_dict(),
    }
    validate_adaptive_checkpoint_metadata_structure(metadata)
    return metadata


def validate_adaptive_checkpoint_metadata_structure(metadata: Mapping[str, Any]) -> None:
    """Validate the self-contained policy and history contract in adaptive metadata."""
    if not isinstance(metadata, Mapping):
        raise ValueError("Adaptive checkpoint metadata must be a mapping.")
    algorithm = str(metadata.get("algorithm", "")).lower()
    if algorithm not in {"ppo", "rpo"}:
        raise ValueError(f"Adaptive checkpoint has unsupported algorithm {algorithm!r}.")
    observation_mode = str(metadata.get("observation_mode", "")).lower()
    if observation_mode not in {
        "legacy8",
        "full11",
        "reference_preview",
        "relative_reference_preview",
    }:
        raise ValueError(
            f"Adaptive checkpoint has unsupported observation_mode {observation_mode!r}."
        )
    try:
        base_input_dim = int(metadata["base_input_dim"])
        total_input_dim = int(metadata["total_input_dim"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Adaptive checkpoint has invalid input dimensions.") from exc
    if base_input_dim <= 0 or total_input_dim <= 0:
        raise ValueError("Adaptive checkpoint input dimensions must be positive.")
    input_fields = metadata.get("input_fields")
    if isinstance(input_fields, (str, bytes)) or not isinstance(input_fields, Sequence):
        raise ValueError("Adaptive checkpoint input_fields must be an ordered sequence.")
    if len(input_fields) != total_input_dim:
        raise ValueError(
            "Adaptive checkpoint input_fields length does not match total_input_dim: "
            f"{len(input_fields)} != {total_input_dim}."
        )
    adaptation_data = metadata.get("velocity_response_adaptation")
    if not isinstance(adaptation_data, Mapping):
        raise ValueError("Adaptive checkpoint is missing velocity_response_adaptation metadata.")
    adaptation_cfg = VelocityResponseAdaptationCfg.from_dict(adaptation_data)
    if not adaptation_cfg.enabled:
        raise ValueError("Adaptive checkpoint metadata must have adaptation enabled.")
    expected_total_dim = (
        base_input_dim
        + adaptation_cfg.encoder_input_dim
        + len(PRIVILEGED_RESPONSE_FIELDS)
    )
    if total_input_dim != expected_total_dim:
        raise ValueError(
            "Adaptive checkpoint total_input_dim must equal base_input_dim + 2H + 3."
        )
    history_end = base_input_dim + adaptation_cfg.encoder_input_dim
    if tuple(input_fields[base_input_dim:history_end]) != adaptation_cfg.history_field_names:
        raise ValueError(
            "Adaptive checkpoint history fields must be vrz history followed by command_z "
            "history, with each channel ordered oldest-to-newest."
        )
    if tuple(input_fields[history_end:]) != PRIVILEGED_RESPONSE_FIELDS:
        raise ValueError("Adaptive checkpoint has invalid critic privileged fields.")
    expected_dimensions = {
        "actor_observation_dim": base_input_dim + adaptation_cfg.encoder_input_dim,
        "actor_head_input_dim": base_input_dim + adaptation_cfg.latent_dim,
        "critic_input_dim": base_input_dim + len(PRIVILEGED_RESPONSE_FIELDS),
    }
    for key, expected_value in expected_dimensions.items():
        if metadata.get(key) != expected_value:
            raise ValueError(
                f"Adaptive checkpoint {key}={metadata.get(key)!r}, expected {expected_value}."
            )
    if metadata.get("encoder_input_layout") != "vrz_history_then_command_z_history":
        raise ValueError("Adaptive checkpoint has an unsupported encoder input layout.")
    if metadata.get("history_order") != "oldest_to_newest":
        raise ValueError("Adaptive checkpoint history_order must be 'oldest_to_newest'.")
    if tuple(metadata.get("critic_privileged_fields", ())) != PRIVILEGED_RESPONSE_FIELDS:
        raise ValueError("Adaptive checkpoint critic privileged fields do not match v2.")


def validate_adaptive_checkpoint_metadata(
    actual: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> None:
    """Reject component checkpoints built for a different policy input contract."""
    required_keys = (
        "algorithm",
        "observation_mode",
        "base_input_dim",
        "total_input_dim",
        "input_fields",
        "actor_observation_dim",
        "actor_head_input_dim",
        "critic_input_dim",
        "encoder_input_layout",
        "history_order",
        "critic_privileged_fields",
        "velocity_response_adaptation",
    )
    mismatches = []
    for key in required_keys:
        if actual.get(key) != expected.get(key):
            mismatches.append(f"{key}: checkpoint={actual.get(key)!r}, expected={expected.get(key)!r}")
    if mismatches:
        raise ValueError(
            "Adaptive checkpoint metadata does not match the configured policy: "
            + "; ".join(mismatches)
        )


def validate_velocity_response_adaptation_environment(
    cfg: VelocityResponseAdaptationCfg,
    *,
    interface_name: str,
    robustness_enabled: bool,
    action_delay_enabled: bool,
    delay_step: int,
    delay_step_choices: Sequence[int] = (),
    predictor_active: bool = False,
    context: str = "RL velocity-response adaptation",
) -> None:
    """Validate the v2 no-delay environment contract."""
    if not cfg.enabled:
        return
    cfg.validate()
    if str(interface_name).lower() != "velocity":
        raise ValueError(f"{context} requires interface_name='velocity'.")
    effective_delay = 0
    if robustness_enabled and action_delay_enabled:
        choices = tuple(int(value) for value in (delay_step_choices or ()))
        effective_delay = max(choices) if choices else max(int(delay_step), 0)
    if effective_delay > 0:
        raise ValueError(
            f"{context} currently requires effective action delay D=0; got D={effective_delay}."
        )
    if predictor_active:
        raise ValueError(f"{context} cannot be combined with an active state predictor.")


def _positive_integer(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"velocity_response_adaptation.{field_name} must be a positive integer.")
    try:
        integer = int(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"velocity_response_adaptation.{field_name} must be a positive integer."
        ) from exc
    if integer <= 0 or (isinstance(value, float) and not value.is_integer()):
        raise ValueError(f"velocity_response_adaptation.{field_name} must be a positive integer.")
    return integer


def _activation_class(name: str) -> type[nn.Module]:
    classes = {
        "elu": nn.ELU,
        "relu": nn.ReLU,
        "silu": nn.SiLU,
        "tanh": nn.Tanh,
    }
    try:
        return classes[str(name).lower()]
    except KeyError as exc:  # Defensive; config validation normally catches this.
        raise ValueError(f"Unsupported velocity-response encoder activation: {name}") from exc


def _validate_component_payload(payload: Any, component: str) -> None:
    if not isinstance(payload, Mapping):
        raise ValueError(f"{component} checkpoint must contain a dictionary payload.")
    if payload.get("format_version") != ADAPTIVE_CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            f"Unsupported {component} checkpoint format_version={payload.get('format_version')!r}; "
            f"expected {ADAPTIVE_CHECKPOINT_FORMAT_VERSION}."
        )
    if payload.get("component") != component:
        raise ValueError(
            f"Expected checkpoint component '{component}', got {payload.get('component')!r}."
        )
    if not isinstance(payload.get("pair_id"), str) or not payload["pair_id"]:
        raise ValueError(f"{component} checkpoint is missing its component pair identifier.")
    if not isinstance(payload.get("metadata"), Mapping):
        raise ValueError(f"{component} checkpoint is missing metadata.")


def _torch_load(path: str | Path, device: str | torch.device):
    path = str(Path(path).expanduser())
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # PyTorch before the weights_only argument was introduced.
        return torch.load(path, map_location=device)


__all__ = [
    "ACTOR_CRITIC_COMPONENT",
    "ADAPTIVE_CHECKPOINT_FORMAT_VERSION",
    "ENCODER_COMPONENT",
    "PRIVILEGED_RESPONSE_FIELDS",
    "ReusableVelocityResponseEncoder",
    "VelocityResponseAdaptationCfg",
    "VelocityResponseEncoder",
    "VelocityResponseEncoderCfg",
    "VelocityResponseHistoryBuffer",
    "adaptive_checkpoint_metadata",
    "load_velocity_response_encoder",
    "validate_adaptive_checkpoint_metadata",
    "validate_adaptive_checkpoint_metadata_structure",
    "validate_velocity_response_adaptation_environment",
]
