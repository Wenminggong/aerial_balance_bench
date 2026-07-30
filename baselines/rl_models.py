"""skrl-compatible MLP models used by RL baselines."""

from __future__ import annotations

import copy
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces

from .velocity_response_adaptation import (
    ACTOR_CRITIC_COMPONENT,
    ADAPTIVE_CHECKPOINT_FORMAT_VERSION,
    ENCODER_COMPONENT,
    PRIVILEGED_RESPONSE_FIELDS,
    VelocityResponseAdaptationCfg,
    VelocityResponseEncoder,
    validate_adaptive_checkpoint_metadata,
    validate_adaptive_checkpoint_metadata_structure,
)

try:  # skrl is an optional dependency unless RL training/evaluation is used.
    from skrl.models.torch import DeterministicMixin, GaussianMixin, Model

    _SKRL_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - depends on local Isaac/skrl environment.
    _SKRL_IMPORT_ERROR = exc

    class Model:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            require_skrl()

    class GaussianMixin:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            require_skrl()

    class DeterministicMixin:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            require_skrl()


@dataclass
class MLPNetworkCfg:
    """Configuration for actor and critic MLPs."""

    layer_num: int = 3
    hidden_dim: int = 256
    clip_actions: bool = True
    clip_log_std: bool = True
    min_log_std: float = -20.0
    max_log_std: float = 2.0
    reduction: str = "sum"
    pid_obs: bool = False

    @classmethod
    def from_dict(cls, data: Mapping | None) -> "MLPNetworkCfg":
        cfg = cls()
        if not data:
            return cfg
        aliases = {
            "inner_dimention": "hidden_dim",
            "inner_dimension": "hidden_dim",
            "model_inner_dimention": "hidden_dim",
            "model_inner_dimension": "hidden_dim",
            "model_layer_num": "layer_num",
        }
        for key, value in data.items():
            normalized_key = aliases.get(key, key)
            if hasattr(cfg, normalized_key):
                setattr(cfg, normalized_key, value)
        cfg.layer_num = int(cfg.layer_num)
        cfg.hidden_dim = int(cfg.hidden_dim)
        cfg.clip_actions = bool(cfg.clip_actions)
        cfg.clip_log_std = bool(cfg.clip_log_std)
        cfg.min_log_std = float(cfg.min_log_std)
        cfg.max_log_std = float(cfg.max_log_std)
        cfg.pid_obs = bool(cfg.pid_obs)
        return cfg


class MLPActor(GaussianMixin, Model):
    """Stochastic MLP actor with a tanh-normalized mean action."""

    def __init__(
        self,
        observation_space,
        action_space,
        pid_obs: bool = False,
        device: str | torch.device = "cuda:0",
        layer_num: int = 3,
        hidden_dim: int = 256,
        clip_actions: bool = True,
        clip_log_std: bool = True,
        min_log_std: float = -20.0,
        max_log_std: float = 2.0,
        reduction: str = "sum",
    ):
        require_skrl()
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)

        self.pid_obs = bool(pid_obs)
        input_dim = 3 if self.pid_obs else self.num_observations
        self.net = _make_mlp(input_dim, self.num_actions, int(layer_num), int(hidden_dim), output_tanh=True)
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role):
        states = inputs["states"]
        if self.pid_obs:
            states = states[:, :3]
        actor_action = self.net(states)
        return actor_action, self.log_std_parameter, {}


class RPOMLPActor(GaussianMixin, Model):
    """RPO actor: MLP mean plus the training-time uniform mean perturbation."""

    def __init__(
        self,
        observation_space,
        action_space,
        pid_obs: bool = False,
        device: str | torch.device = "cuda:0",
        layer_num: int = 3,
        hidden_dim: int = 256,
        clip_actions: bool = True,
        clip_log_std: bool = True,
        min_log_std: float = -20.0,
        max_log_std: float = 2.0,
        reduction: str = "sum",
    ):
        require_skrl()
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)

        self.pid_obs = bool(pid_obs)
        input_dim = 3 if self.pid_obs else self.num_observations
        self.net = _make_mlp(input_dim, self.num_actions, int(layer_num), int(hidden_dim), output_tanh=True)
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role):
        states = inputs["states"]
        if self.pid_obs:
            states = states[:, :3]
        mean_action = self.net(states)
        rpo_alpha = inputs.get("alpha", 0.0)
        if not torch.is_tensor(rpo_alpha):
            rpo_alpha = torch.as_tensor(rpo_alpha, device=mean_action.device, dtype=mean_action.dtype)
        perturbation = torch.zeros_like(mean_action).uniform_(-float(rpo_alpha), float(rpo_alpha))
        actor_action = mean_action + perturbation
        return actor_action, self.log_std_parameter, {"no_pert_mean_actions": mean_action}


class MLPCritic(DeterministicMixin, Model):
    """Deterministic MLP critic."""

    def __init__(
        self,
        observation_space,
        action_space,
        device: str | torch.device = "cuda:0",
        layer_num: int = 3,
        hidden_dim: int = 256,
        clip_actions: bool = False,
    ):
        require_skrl()
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.net = _make_mlp(
            self.num_observations,
            1,
            int(layer_num),
            int(hidden_dim),
            output_tanh=False,
        )

    def compute(self, inputs, role):
        return self.net(inputs["states"]), {}


class AdaptiveVelocityActor(GaussianMixin, Model):
    """History-conditioned actor that owns the velocity-response encoder."""

    def __init__(
        self,
        observation_space,
        action_space,
        algorithm: str,
        adaptation_cfg: VelocityResponseAdaptationCfg,
        base_observation_dim: int,
        device: str | torch.device = "cuda:0",
        layer_num: int = 3,
        hidden_dim: int = 256,
        clip_actions: bool = True,
        clip_log_std: bool = True,
        min_log_std: float = -20.0,
        max_log_std: float = 2.0,
        reduction: str = "sum",
    ):
        require_skrl()
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(
            self,
            clip_actions,
            clip_log_std,
            min_log_std,
            max_log_std,
            reduction,
        )

        adaptation_cfg.validate()
        self.algorithm = str(algorithm).lower()
        self.adaptation_cfg = adaptation_cfg
        self.base_observation_dim = int(base_observation_dim)
        self.history_length = int(adaptation_cfg.history_length)
        self.latent_dim = int(adaptation_cfg.latent_dim)
        self.encoder_input_dim = adaptation_cfg.encoder_input_dim
        self.privileged_dim = len(PRIVILEGED_RESPONSE_FIELDS)
        expected_observations = (
            self.base_observation_dim + self.encoder_input_dim + self.privileged_dim
        )
        if self.num_observations != expected_observations:
            raise ValueError(
                "AdaptiveVelocityActor transport dimension must equal base_observation_dim "
                f"+ 2H + 3 ({self.base_observation_dim} + {self.encoder_input_dim} + "
                f"{self.privileged_dim} = "
                f"{expected_observations}), got {self.num_observations}."
            )

        self.response_encoder = VelocityResponseEncoder(
            self.history_length,
            self.latent_dim,
            adaptation_cfg.encoder,
        )
        fused_dim = self.base_observation_dim + self.latent_dim
        self.actor_observation_dim = self.base_observation_dim + self.encoder_input_dim
        self.actor_input_dim = fused_dim
        self.actor_head = _make_mlp(
            fused_dim,
            self.num_actions,
            int(layer_num),
            int(hidden_dim),
            output_tanh=True,
        )
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))
        self.last_latent = torch.zeros((1, self.latent_dim), device=self.device)

    def compute(self, inputs, role):
        states = inputs["states"]
        expected_dim = self.base_observation_dim + self.encoder_input_dim + self.privileged_dim
        if states.shape[-1] != expected_dim:
            raise ValueError(
                "AdaptiveVelocityActor received state dimension "
                f"{states.shape[-1]}, expected {expected_dim}."
            )
        base_observation = states[:, : self.base_observation_dim]
        velocity_history = states[
            :,
            self.base_observation_dim : self.base_observation_dim + self.encoder_input_dim,
        ]
        latent = self.response_encoder(velocity_history)
        self.last_latent = latent.detach()
        fused = torch.cat((base_observation, latent), dim=-1)

        if role != "policy":
            raise ValueError(f"AdaptiveVelocityActor received unsupported role '{role}'.")
        mean_action = self.actor_head(fused)
        if self.algorithm == "rpo":
            rpo_alpha = inputs.get("alpha", 0.0)
            if not torch.is_tensor(rpo_alpha):
                rpo_alpha = torch.as_tensor(
                    rpo_alpha,
                    device=mean_action.device,
                    dtype=mean_action.dtype,
                )
            perturbation = torch.zeros_like(mean_action).uniform_(
                -float(rpo_alpha),
                float(rpo_alpha),
            )
            actor_action = mean_action + perturbation
            outputs = {"no_pert_mean_actions": mean_action, "velocity_response_latent": latent}
        else:
            actor_action = mean_action
            outputs = {"velocity_response_latent": latent}
        return actor_action, self.log_std_parameter, outputs

    def load_head_state_dict(self, state_dict: Mapping[str, torch.Tensor]) -> None:
        """Load the actor head and log standard deviation without the encoder."""
        incompatible = self.load_state_dict(state_dict, strict=False)
        expected_missing = {
            key for key in self.state_dict() if key.startswith("response_encoder.")
        }
        if set(incompatible.missing_keys) != expected_missing or incompatible.unexpected_keys:
            raise RuntimeError(
                "Adaptive actor head checkpoint keys do not match the model: "
                f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}."
            )


class PrivilegedVelocityCritic(DeterministicMixin, Model):
    """Asymmetric critic using base observations and true response parameters."""

    def __init__(
        self,
        observation_space,
        action_space,
        adaptation_cfg: VelocityResponseAdaptationCfg,
        base_observation_dim: int,
        device: str | torch.device = "cuda:0",
        layer_num: int = 3,
        hidden_dim: int = 256,
    ):
        require_skrl()
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions=False)

        adaptation_cfg.validate()
        self.base_observation_dim = int(base_observation_dim)
        self.encoder_input_dim = adaptation_cfg.encoder_input_dim
        self.privileged_dim = len(PRIVILEGED_RESPONSE_FIELDS)
        expected_observations = (
            self.base_observation_dim + self.encoder_input_dim + self.privileged_dim
        )
        if self.num_observations != expected_observations:
            raise ValueError(
                "PrivilegedVelocityCritic transport dimension must equal base_observation_dim "
                f"+ 2H + 3; expected {expected_observations}, got {self.num_observations}."
            )
        self.privileged_start = self.base_observation_dim + self.encoder_input_dim
        self.critic_input_dim = self.base_observation_dim + self.privileged_dim
        self.critic_head = _make_mlp(
            self.critic_input_dim,
            1,
            int(layer_num),
            int(hidden_dim),
            output_tanh=False,
        )

    def compute(self, inputs, role):
        if role != "value":
            raise ValueError(f"PrivilegedVelocityCritic received unsupported role '{role}'.")
        states = inputs["states"]
        expected_dim = self.privileged_start + self.privileged_dim
        if states.shape[-1] != expected_dim:
            raise ValueError(
                "PrivilegedVelocityCritic received state dimension "
                f"{states.shape[-1]}, expected {expected_dim}."
            )
        critic_input = torch.cat(
            (
                states[:, : self.base_observation_dim],
                states[:, self.privileged_start :],
            ),
            dim=-1,
        )
        return self.critic_head(critic_input), {}


def require_skrl():
    """Raise a clear error if skrl is unavailable."""
    if _SKRL_IMPORT_ERROR is not None:
        raise RuntimeError(
            "RL baselines require the optional 'skrl' package. Run RL scripts inside the Isaac/skrl "
            "environment used for training, or install skrl>=1.3.0."
        ) from _SKRL_IMPORT_ERROR


def make_models(
    algorithm: str,
    network_cfg: MLPNetworkCfg,
    observation_space,
    action_space,
    device: str | torch.device,
    adaptation_cfg: VelocityResponseAdaptationCfg | None = None,
    base_observation_dim: int | None = None,
) -> dict[str, Model]:
    """Build skrl policy/value models for PPO or RPO."""
    require_skrl()
    algorithm = str(algorithm).lower()
    adaptation_cfg = adaptation_cfg or VelocityResponseAdaptationCfg()
    if adaptation_cfg.enabled:
        adaptation_cfg.validate()
        if network_cfg.pid_obs:
            raise ValueError("velocity_response_adaptation is incompatible with network.pid_obs=True.")
        if base_observation_dim is None:
            raise ValueError("Adaptive RL models require base_observation_dim.")
        return {
            "policy": AdaptiveVelocityActor(
                observation_space=observation_space,
                action_space=action_space,
                algorithm=algorithm,
                adaptation_cfg=adaptation_cfg,
                base_observation_dim=base_observation_dim,
                device=device,
                layer_num=network_cfg.layer_num,
                hidden_dim=network_cfg.hidden_dim,
                clip_actions=network_cfg.clip_actions,
                clip_log_std=network_cfg.clip_log_std,
                min_log_std=network_cfg.min_log_std,
                max_log_std=network_cfg.max_log_std,
                reduction=network_cfg.reduction,
            ),
            "value": PrivilegedVelocityCritic(
                observation_space=observation_space,
                action_space=action_space,
                adaptation_cfg=adaptation_cfg,
                base_observation_dim=base_observation_dim,
                device=device,
                layer_num=network_cfg.layer_num,
                hidden_dim=network_cfg.hidden_dim,
            ),
        }

    actor_cls = RPOMLPActor if algorithm == "rpo" else MLPActor
    actor_observation_space = observation_space
    if network_cfg.pid_obs:
        actor_observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)

    return {
        "policy": actor_cls(
            observation_space=actor_observation_space,
            action_space=action_space,
            device=device,
            pid_obs=network_cfg.pid_obs,
            layer_num=network_cfg.layer_num,
            hidden_dim=network_cfg.hidden_dim,
            clip_actions=network_cfg.clip_actions,
            clip_log_std=network_cfg.clip_log_std,
            min_log_std=network_cfg.min_log_std,
            max_log_std=network_cfg.max_log_std,
            reduction=network_cfg.reduction,
        ),
        "value": MLPCritic(
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            layer_num=network_cfg.layer_num,
            hidden_dim=network_cfg.hidden_dim,
        ),
    }


def make_agent_class_and_cfg(
    algorithm: str,
    overrides: Mapping | None,
    observation_space,
    device: str | torch.device,
    adaptive_checkpointing: bool = False,
):
    """Return the selected skrl agent class and a configured agent dictionary."""
    require_skrl()
    algorithm = str(algorithm).lower()
    if algorithm == "rpo":
        from skrl.agents.torch.rpo import RPO, RPO_DEFAULT_CONFIG

        agent_cls = RPO
        cfg = copy.deepcopy(RPO_DEFAULT_CONFIG)
    elif algorithm == "ppo":
        from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG

        agent_cls = PPO
        cfg = copy.deepcopy(PPO_DEFAULT_CONFIG)
    else:
        raise ValueError("RL algorithm must be 'rpo' or 'ppo'.")

    from skrl.resources.preprocessors.torch import RunningStandardScaler
    from skrl.resources.schedulers.torch import KLAdaptiveRL

    cfg.setdefault("experiment", {})
    cfg["state_preprocessor"] = RunningStandardScaler
    cfg["state_preprocessor_kwargs"] = {"size": observation_space, "device": device}
    cfg["value_preprocessor"] = RunningStandardScaler
    cfg["value_preprocessor_kwargs"] = {"size": 1, "device": device}
    cfg["learning_rate_scheduler"] = KLAdaptiveRL
    cfg["learning_rate_scheduler_kwargs"] = {"kl_threshold": 0.008}

    _deep_update(cfg, overrides or {})
    if adaptive_checkpointing:
        agent_cls = _adaptive_checkpoint_agent_class(agent_cls)
    return agent_cls, cfg


def configure_adaptive_agent_checkpointing(
    agent,
    metadata: Mapping[str, Any],
) -> None:
    """Validate asymmetric models and enable adaptive component export."""
    policy = agent.models.get("policy")
    value = agent.models.get("value")
    if (
        policy is value
        or not isinstance(policy, AdaptiveVelocityActor)
        or not isinstance(value, PrivilegedVelocityCritic)
    ):
        raise ValueError(
            "Adaptive checkpointing requires distinct AdaptiveVelocityActor and "
            "PrivilegedVelocityCritic models."
        )
    metadata = dict(metadata)
    validate_adaptive_checkpoint_metadata_structure(metadata)
    agent.adaptive_checkpoint_metadata = metadata
    agent.checkpoint_modules["adaptive_metadata"] = _AdaptiveCheckpointMetadata(metadata)


def validate_adaptive_agent_checkpoint(
    path: str | Path,
    expected_metadata: Mapping[str, Any],
    device: str | torch.device,
) -> None:
    """Validate a complete asymmetric adaptive checkpoint before skrl loads it."""
    payload = _load_checkpoint_payload(path, device)
    if not isinstance(payload, Mapping):
        raise ValueError("Adaptive agent checkpoint must contain a dictionary payload.")
    if "shared_actor_critic" in payload:
        raise ValueError(
            "Adaptive checkpoint format v1 used a shared actor-critic and cannot be loaded "
            "by the asymmetric v2 policy. Retraining is required."
        )
    required_modules = {"policy", "value", "state_preprocessor", "adaptive_metadata"}
    missing_modules = sorted(required_modules.difference(payload))
    if missing_modules:
        raise ValueError(
            "Adaptive agent checkpoint is missing v2 modules: " + ", ".join(missing_modules)
        )
    metadata_payload = payload["adaptive_metadata"]
    if not isinstance(metadata_payload, Mapping):
        raise ValueError("Adaptive agent checkpoint has invalid metadata state.")
    if metadata_payload.get("format_version") != ADAPTIVE_CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            "Unsupported adaptive agent checkpoint format_version="
            f"{metadata_payload.get('format_version')!r}; expected "
            f"{ADAPTIVE_CHECKPOINT_FORMAT_VERSION}."
        )
    actual_metadata = metadata_payload.get("metadata")
    validate_adaptive_checkpoint_metadata_structure(actual_metadata)
    validate_adaptive_checkpoint_metadata(actual_metadata, expected_metadata)


def export_adaptive_agent_components(
    agent,
    checkpoint_dir: str | Path,
    tag: str,
    *,
    use_best: bool = False,
) -> tuple[Path, Path]:
    """Export actor/critic heads and the response encoder into separate files."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metadata = getattr(agent, "adaptive_checkpoint_metadata", None)
    if not isinstance(metadata, Mapping):
        raise ValueError("Adaptive checkpoint export requires checkpoint metadata.")

    actor = agent.models.get("policy")
    critic = agent.models.get("value")
    if not isinstance(actor, AdaptiveVelocityActor) or not isinstance(
        critic,
        PrivilegedVelocityCritic,
    ):
        raise ValueError("Adaptive checkpoint export requires asymmetric adaptive models.")

    if use_best:
        modules = agent.checkpoint_best_modules.get("modules", {})
        actor_state = modules.get("policy")
        critic_state = modules.get("value")
        preprocessor_state = modules.get("state_preprocessor")
        if actor_state is None or critic_state is None or preprocessor_state is None:
            raise ValueError("No best adaptive checkpoint snapshot is available for component export.")
    else:
        actor_state = actor.state_dict()
        critic_state = critic.state_dict()
        preprocessor_state = agent._state_preprocessor.state_dict()

    actor_state = _cpu_state_dict(actor_state)
    critic_state = _cpu_state_dict(critic_state)
    preprocessor_state = _cpu_state_dict(preprocessor_state)
    encoder_prefix = "response_encoder."
    encoder_state = {
        key[len(encoder_prefix) :]: value
        for key, value in actor_state.items()
        if key.startswith(encoder_prefix)
    }
    actor_head_state = {
        key: value for key, value in actor_state.items() if not key.startswith(encoder_prefix)
    }
    if not encoder_state or not actor_head_state or not critic_state:
        raise RuntimeError("Unable to split asymmetric adaptive checkpoint components.")

    history_start = actor.base_observation_dim
    history_end = history_start + actor.encoder_input_dim
    running_mean = preprocessor_state.get("running_mean")
    running_variance = preprocessor_state.get("running_variance")
    if running_mean is None or running_variance is None:
        raise RuntimeError("Adaptive component export requires RunningStandardScaler state.")
    if running_mean.shape[-1] < history_end or running_variance.shape[-1] < history_end:
        raise RuntimeError("State preprocessor is smaller than the configured velocity histories.")

    actor_critic_path = checkpoint_dir / f"adaptive_actor_critic_heads_{tag}.pt"
    encoder_path = checkpoint_dir / f"velocity_response_encoder_{tag}.pt"
    pair_id = uuid.uuid4().hex
    torch.save(
        {
            "format_version": ADAPTIVE_CHECKPOINT_FORMAT_VERSION,
            "component": ACTOR_CRITIC_COMPONENT,
            "pair_id": pair_id,
            "metadata": dict(metadata),
            "actor_head_state_dict": actor_head_state,
            "critic_state_dict": critic_state,
            "state_preprocessor_state_dict": preprocessor_state,
        },
        actor_critic_path,
    )
    torch.save(
        {
            "format_version": ADAPTIVE_CHECKPOINT_FORMAT_VERSION,
            "component": ENCODER_COMPONENT,
            "pair_id": pair_id,
            "metadata": dict(metadata),
            "encoder_state_dict": encoder_state,
            "history_normalization": {
                "running_mean": running_mean[history_start:history_end].clone(),
                "running_variance": running_variance[history_start:history_end].clone(),
                "current_count": preprocessor_state.get("current_count"),
                "epsilon": float(getattr(agent._state_preprocessor, "epsilon", 1.0e-8)),
                "clip_threshold": float(
                    getattr(agent._state_preprocessor, "clip_threshold", 5.0)
                ),
            },
        },
        encoder_path,
    )
    return actor_critic_path, encoder_path


def load_adaptive_agent_components(
    agent,
    actor_critic_path: str | Path,
    encoder_path: str | Path,
    expected_metadata: Mapping[str, Any],
) -> None:
    """Load separately saved heads, encoder, and policy input normalizer."""
    actor_payload = _load_checkpoint_payload(actor_critic_path, agent.device)
    encoder_payload = _load_checkpoint_payload(encoder_path, agent.device)
    _validate_component(actor_payload, ACTOR_CRITIC_COMPONENT)
    _validate_component(encoder_payload, ENCODER_COMPONENT)
    if actor_payload["pair_id"] != encoder_payload["pair_id"]:
        raise ValueError("Adaptive actor-critic and encoder checkpoints are not a saved pair.")
    validate_adaptive_checkpoint_metadata(actor_payload["metadata"], expected_metadata)
    validate_adaptive_checkpoint_metadata(encoder_payload["metadata"], expected_metadata)
    if actor_payload["metadata"] != encoder_payload["metadata"]:
        raise ValueError("Adaptive actor-critic and encoder checkpoint metadata differ.")

    actor = agent.models.get("policy")
    critic = agent.models.get("value")
    if not isinstance(actor, AdaptiveVelocityActor) or not isinstance(
        critic,
        PrivilegedVelocityCritic,
    ):
        raise ValueError("Adaptive component loading requires asymmetric adaptive models.")
    actor.load_head_state_dict(actor_payload["actor_head_state_dict"])
    actor.response_encoder.load_state_dict(encoder_payload["encoder_state_dict"])
    critic.load_state_dict(actor_payload["critic_state_dict"])
    agent._state_preprocessor.load_state_dict(
        actor_payload["state_preprocessor_state_dict"]
    )
    actor.eval()
    critic.eval()


def _make_mlp(input_dim: int, output_dim: int, layer_num: int, hidden_dim: int, output_tanh: bool) -> nn.Sequential:
    net = nn.Sequential()
    current_dim = int(input_dim)
    for _ in range(1, int(layer_num)):
        net.append(nn.Linear(current_dim, int(hidden_dim)))
        net.append(nn.ReLU())
        current_dim = int(hidden_dim)
    net.append(nn.Linear(current_dim, int(output_dim)))
    if output_tanh:
        net.append(nn.Tanh())
    return net


def _deep_update(target: dict, values: Mapping):
    for key, value in values.items():
        if isinstance(value, Mapping) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = value


class _AdaptiveCheckpointMetadata:
    """Small skrl checkpoint module carrying the v2 input contract."""

    def __init__(self, metadata: Mapping[str, Any]):
        self.metadata = dict(metadata)

    def state_dict(self) -> dict[str, Any]:
        return {
            "format_version": ADAPTIVE_CHECKPOINT_FORMAT_VERSION,
            "metadata": copy.deepcopy(self.metadata),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("format_version") != ADAPTIVE_CHECKPOINT_FORMAT_VERSION:
            raise ValueError(
                "Unsupported adaptive checkpoint metadata format_version="
                f"{state.get('format_version')!r}."
            )
        actual = state.get("metadata")
        validate_adaptive_checkpoint_metadata_structure(actual)
        validate_adaptive_checkpoint_metadata(actual, self.metadata)


def _adaptive_checkpoint_agent_class(base_cls):
    class AdaptiveCheckpointAgent(base_cls):
        def record_transition(
            self,
            states,
            actions,
            rewards,
            next_states,
            terminated,
            truncated,
            infos,
            timestep,
            timesteps,
        ) -> None:
            super().record_transition(
                states,
                actions,
                rewards,
                next_states,
                terminated,
                truncated,
                infos,
                timestep,
                timesteps,
            )
            actor = self.models.get("policy")
            if not isinstance(actor, AdaptiveVelocityActor):
                return
            base = actor.base_observation_dim
            history_length = actor.history_length
            actual_latest = states[:, base + history_length - 1]
            command_latest = states[:, base + 2 * history_length - 1]
            self.track_data(
                "Adaptation / Actual velocity history latest mean",
                actual_latest.float().mean().item(),
            )
            self.track_data(
                "Adaptation / Command velocity history latest mean",
                command_latest.float().mean().item(),
            )
            self.track_data(
                "Adaptation / Velocity history error latest mean",
                (actual_latest - command_latest).float().mean().item(),
            )
            for index, field_name in enumerate(PRIVILEGED_RESPONSE_FIELDS):
                self.track_data(
                    f"Critic / {field_name} mean",
                    states[:, -len(PRIVILEGED_RESPONSE_FIELDS) + index].float().mean().item(),
                )

        def _update(self, timestep: int, timesteps: int) -> None:
            super()._update(timestep, timesteps)
            model = self.models.get("policy")
            if not isinstance(model, AdaptiveVelocityActor):
                return
            squared_gradient_norm = torch.zeros((), device=self.device)
            for parameter in model.response_encoder.parameters():
                if parameter.grad is not None:
                    squared_gradient_norm += parameter.grad.detach().float().square().sum()
            self.track_data(
                "Learning / Velocity response encoder gradient norm",
                torch.sqrt(squared_gradient_norm).item(),
            )
            self.track_data(
                "Policy / Velocity response latent norm",
                torch.linalg.vector_norm(model.last_latent.float(), dim=-1).mean().item(),
            )
            for index in range(model.latent_dim):
                self.track_data(
                    f"Policy / Velocity response latent z_{index} mean",
                    model.last_latent[:, index].float().mean().item(),
                )

        def write_checkpoint(self, timestep: int, timesteps: int) -> None:
            super().write_checkpoint(timestep, timesteps)
            if not hasattr(self, "adaptive_checkpoint_metadata"):
                return
            tag = str(timestep)
            checkpoint_dir = Path(self.experiment_dir) / "checkpoints"
            export_adaptive_agent_components(self, checkpoint_dir, tag)
            if self.checkpoint_best_modules.get("modules"):
                export_adaptive_agent_components(
                    self,
                    checkpoint_dir,
                    "best",
                    use_best=True,
                )

    AdaptiveCheckpointAgent.__name__ = f"AdaptiveCheckpoint{base_cls.__name__}"
    return AdaptiveCheckpointAgent


def _cpu_state_dict(state_dict: Mapping[str, Any]) -> dict[str, Any]:
    values = {}
    for key, value in state_dict.items():
        if torch.is_tensor(value):
            values[key] = value.detach().cpu().clone()
        else:
            values[key] = copy.deepcopy(value)
    return values


def _load_checkpoint_payload(path: str | Path, device: str | torch.device):
    path = str(Path(path).expanduser())
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # PyTorch before the weights_only argument was introduced.
        return torch.load(path, map_location=device)


def _validate_component(payload: Any, expected_component: str) -> None:
    if not isinstance(payload, Mapping):
        raise ValueError(f"{expected_component} checkpoint must contain a dictionary payload.")
    if payload.get("format_version") != ADAPTIVE_CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            f"Unsupported adaptive checkpoint format_version={payload.get('format_version')!r}."
        )
    if payload.get("component") != expected_component:
        raise ValueError(
            f"Expected adaptive component '{expected_component}', got "
            f"{payload.get('component')!r}."
        )
    if not isinstance(payload.get("pair_id"), str) or not payload["pair_id"]:
        raise ValueError(f"{expected_component} checkpoint is missing its component pair identifier.")
    validate_adaptive_checkpoint_metadata_structure(payload.get("metadata"))
