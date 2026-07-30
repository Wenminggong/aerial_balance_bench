"""RL baseline policy wrappers for Aerial-Balance-Bench."""

from __future__ import annotations

import operator
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from gymnasium import spaces

from .base_policy import BasePolicy, BasePolicyCfg, ObservationIndex
from .model_state_predictor import (
    VelocityModelStatePredictor,
    VelocityModelStatePredictorCfg,
    extract_reference_positions,
    validate_reference_preview_horizon,
)
from .rl_models import (
    AdaptiveVelocityActor,
    MLPNetworkCfg,
    configure_adaptive_agent_checkpointing,
    load_adaptive_agent_components,
    make_agent_class_and_cfg,
    make_models,
    validate_adaptive_agent_checkpoint,
)
from .rl_observation_adapter import RLObservationAdapter, RLObservationAdapterCfg
from .velocity_response_adaptation import (
    PRIVILEGED_RESPONSE_FIELDS,
    VelocityResponseAdaptationCfg,
    VelocityResponseHistoryBuffer,
    adaptive_checkpoint_metadata,
)


@dataclass
class RLPolicyCfg(BasePolicyCfg):
    """Configuration for RL policy deployment/evaluation."""

    name: str = "rl"
    algorithm: str = "rpo"
    observation_mode: str = "legacy8"
    reference_preview_samples: int | None = None
    checkpoint_path: str | None = None
    actor_critic_checkpoint_path: str | None = None
    encoder_checkpoint_path: str | None = None
    load_checkpoint: bool = True
    deterministic: bool = True
    physical_action_limit: float | str = "auto"
    network: MLPNetworkCfg = field(default_factory=MLPNetworkCfg)
    agent: dict[str, Any] = field(default_factory=dict)
    state_predictor: VelocityModelStatePredictorCfg = field(default_factory=VelocityModelStatePredictorCfg)
    velocity_response_adaptation: VelocityResponseAdaptationCfg = field(
        default_factory=VelocityResponseAdaptationCfg
    )

    @classmethod
    def from_dict(cls, data: Mapping | None) -> "RLPolicyCfg":
        """Build an RL policy config from a YAML dictionary."""
        cfg = cls()
        if not data:
            return cfg

        policy_data = data.get("rl_policy", data.get("policy", data))
        if "name" in policy_data:
            cfg.name = str(policy_data["name"])
        if "policy_name" in policy_data:
            cfg.name = str(policy_data["policy_name"])
        if "algorithm" in policy_data:
            cfg.algorithm = str(policy_data["algorithm"]).lower()
        if "observation_mode" in policy_data:
            cfg.observation_mode = str(policy_data["observation_mode"]).lower()
        if "reference_preview_samples" in policy_data:
            cfg.reference_preview_samples = policy_data["reference_preview_samples"]
        if "checkpoint_path" in policy_data:
            checkpoint_path = policy_data["checkpoint_path"]
            cfg.checkpoint_path = None if checkpoint_path in (None, "null", "") else str(checkpoint_path)
        if "actor_critic_checkpoint_path" in policy_data:
            checkpoint_path = policy_data["actor_critic_checkpoint_path"]
            cfg.actor_critic_checkpoint_path = (
                None if checkpoint_path in (None, "null", "") else str(checkpoint_path)
            )
        if "encoder_checkpoint_path" in policy_data:
            checkpoint_path = policy_data["encoder_checkpoint_path"]
            cfg.encoder_checkpoint_path = (
                None if checkpoint_path in (None, "null", "") else str(checkpoint_path)
            )
        if "load_checkpoint" in policy_data:
            cfg.load_checkpoint = bool(policy_data["load_checkpoint"])
        if "deterministic" in policy_data:
            cfg.deterministic = bool(policy_data["deterministic"])
        if "physical_action_limit" in policy_data:
            cfg.physical_action_limit = policy_data["physical_action_limit"]

        cfg.network = MLPNetworkCfg.from_dict(policy_data.get("network", policy_data.get("model", {})))
        cfg.agent = dict(policy_data.get("agent", {}))
        cfg.state_predictor = VelocityModelStatePredictorCfg.from_dict(policy_data.get("state_predictor", {}))
        cfg.velocity_response_adaptation = VelocityResponseAdaptationCfg.from_dict(
            policy_data.get("velocity_response_adaptation", {})
        )
        return cfg


def validate_rl_predictor_environment_contract(
    *,
    observation_mode: str,
    reference_preview_samples: int | None,
    predictor_enabled: bool,
    predictor_delay_step: int,
    preview_enabled: bool,
    preview_future_steps: int,
    robustness_enabled: bool,
    action_delay_enabled: bool,
    environment_delay_step: int,
    environment_delay_step_choices: Sequence[int] = (),
    moving_reference: bool = True,
    context: str = "RL unified tracking",
) -> None:
    """Validate delayed-environment and preview alignment for RL evaluation."""
    delay_step = int(predictor_delay_step)
    if not predictor_enabled or delay_step <= 0:
        return

    delay_step_choices = tuple(int(value) for value in (environment_delay_step_choices or ()))
    random_delay_enabled = len(delay_step_choices) > 0
    environment_delay_step = (
        max(delay_step_choices) if random_delay_enabled else int(environment_delay_step)
    )
    environment_delay_active = (
        bool(robustness_enabled)
        and bool(action_delay_enabled)
        and environment_delay_step > 0
    )
    if not environment_delay_active:
        raise ValueError(
            f"{context} requires an active environment action delay when the state predictor "
            f"is active with delay_step={delay_step}. Enable robustness.enabled and "
            "robustness.action_delay_enabled with a positive delay_step."
        )
    if not random_delay_enabled and delay_step != environment_delay_step:
        raise ValueError(
            f"{context} state predictor delay_step must match robustness.delay_step; "
            f"got predictor D={delay_step}, environment D={environment_delay_step}."
        )

    mode = str(observation_mode).lower()
    if mode == "reference_preview":
        raise ValueError(
            f"{context} observation_mode='reference_preview' cannot be combined with an active "
            "state predictor. Use observation_mode='relative_reference_preview' for aligned "
            "preview compensation."
        )
    if not moving_reference and mode != "relative_reference_preview":
        return

    validate_reference_preview_horizon(
        delay_step=delay_step,
        preview_enabled=preview_enabled,
        preview_future_steps=preview_future_steps,
        context=context,
    )
    if mode != "relative_reference_preview":
        return

    configured_samples = reference_preview_samples
    if configured_samples is None or isinstance(configured_samples, bool):
        raise ValueError(
            f"{context} relative_reference_preview requires a positive integer "
            "reference_preview_samples=K."
        )
    try:
        samples = operator.index(configured_samples)
    except TypeError as exc:
        raise ValueError(
            f"{context} relative_reference_preview requires a positive integer "
            "reference_preview_samples=K."
        ) from exc
    if samples <= 0:
        raise ValueError(
            f"{context} relative_reference_preview requires a positive integer "
            f"reference_preview_samples=K; got K={configured_samples}."
        )

    raw_horizon = int(preview_future_steps)
    effective_horizon = raw_horizon - delay_step
    if effective_horizon < samples:
        raise ValueError(
            f"{context} requires H_effective=H_raw-D >= K; got "
            f"H_raw={raw_horizon}, D={delay_step}, H_effective={effective_horizon}, K={samples}."
        )
    if effective_horizon % samples != 0:
        raise ValueError(
            f"{context} requires H_effective=H_raw-D to be an integer multiple of K; got "
            f"H_raw={raw_horizon}, D={delay_step}, H_effective={effective_horizon}, K={samples}."
        )


class RLPolicy(BasePolicy):
    """skrl-backed high-level RL policy."""

    cfg: RLPolicyCfg

    def __init__(
        self,
        cfg: RLPolicyCfg,
        num_envs: int,
        device: str | torch.device,
        step_dt: float,
        physical_action_limit: float,
        raw_observation_dim: int | None = None,
        raw_observation_fields: Sequence[str] | None = None,
    ):
        super().__init__(cfg, num_envs, device)
        self.step_dt = float(step_dt)
        self.physical_action_limit = float(physical_action_limit)
        if self.step_dt <= 0.0:
            raise ValueError("RLPolicy requires a positive step_dt.")
        if self.physical_action_limit <= 0.0:
            raise ValueError("RLPolicy requires a positive physical_action_limit.")

        self._resolve_predictor_cfg_defaults()
        self.state_predictor = VelocityModelStatePredictor(cfg.state_predictor, num_envs, self.device)
        cfg.velocity_response_adaptation.validate()
        if cfg.velocity_response_adaptation.enabled and self.state_predictor.active:
            raise ValueError(
                "velocity_response_adaptation cannot be combined with an active state predictor."
            )
        observation_mode = str(cfg.observation_mode).lower()
        reference_preview_base_offset = (
            self.state_predictor.delay_step
            if observation_mode == "relative_reference_preview" and self.state_predictor.active
            else 0
        )
        adapter_cfg = RLObservationAdapterCfg(
            observation_mode=cfg.observation_mode,
            reference_preview_samples=cfg.reference_preview_samples,
        )
        self.observation_adapter = RLObservationAdapter(
            adapter_cfg,
            num_envs,
            self.device,
            raw_observation_dim=raw_observation_dim,
            raw_observation_fields=raw_observation_fields,
            reference_preview_base_offset=reference_preview_base_offset,
        )
        if self.observation_mode == "reference_preview" and self.state_predictor.active:
            raise ValueError(
                f"RL observation_mode='{self.observation_mode}' cannot be combined with an active "
                "state predictor. Use observation_mode='relative_reference_preview' for aligned "
                "preview compensation, or disable the state predictor."
            )

        self.base_policy_input_dim = self.observation_adapter.input_dim
        self.response_history: VelocityResponseHistoryBuffer | None = None
        if cfg.velocity_response_adaptation.enabled:
            self.response_history = VelocityResponseHistoryBuffer(
                num_envs,
                cfg.velocity_response_adaptation.history_length,
                self.device,
            )
        adaptation_dim = (
            cfg.velocity_response_adaptation.encoder_input_dim
            + len(PRIVILEGED_RESPONSE_FIELDS)
            if cfg.velocity_response_adaptation.enabled
            else 0
        )
        self.model_input_dim = self.base_policy_input_dim + adaptation_dim
        self.policy_input_fields = (
            *self.observation_adapter.field_names,
            *(
                cfg.velocity_response_adaptation.history_field_names
                if cfg.velocity_response_adaptation.enabled
                else ()
            ),
            *(PRIVILEGED_RESPONSE_FIELDS if cfg.velocity_response_adaptation.enabled else ()),
        )
        # Privileged dynamics are training-only. Keep zero placeholders during
        # deployment so full asymmetric agent checkpoints retain one state shape.
        self.privileged_response_parameters = torch.zeros(
            (self.num_envs, len(PRIVILEGED_RESPONSE_FIELDS)),
            device=self.device,
            dtype=torch.float32,
        )

        self.normalized_action_space = spaces.Box(
            low=np.array([-1.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.policy_observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.model_input_dim,),
            dtype=np.float32,
        )
        self.models = make_models(
            cfg.algorithm,
            cfg.network,
            self.policy_observation_space,
            self.normalized_action_space,
            self.device,
            adaptation_cfg=cfg.velocity_response_adaptation,
            base_observation_dim=self.base_policy_input_dim,
        )
        agent_cls, agent_cfg = make_agent_class_and_cfg(
            cfg.algorithm,
            cfg.agent,
            self.policy_observation_space,
            self.device,
            adaptive_checkpointing=cfg.velocity_response_adaptation.enabled,
        )
        agent_cfg.setdefault("experiment", {})
        agent_cfg["experiment"]["wandb"] = False
        self.agent_cfg = agent_cfg
        self.agent = agent_cls(
            models=self.models,
            memory=None,
            cfg=agent_cfg,
            observation_space=self.policy_observation_space,
            action_space=self.normalized_action_space,
            device=self.device,
        )
        self.adaptive_checkpoint_metadata: dict[str, Any] | None = None
        if cfg.velocity_response_adaptation.enabled:
            self.adaptive_checkpoint_metadata = adaptive_checkpoint_metadata(
                algorithm=cfg.algorithm,
                observation_mode=self.observation_mode,
                base_input_dim=self.base_policy_input_dim,
                total_input_dim=self.model_input_dim,
                input_fields=self.policy_input_fields,
                adaptation_cfg=cfg.velocity_response_adaptation,
            )
            configure_adaptive_agent_checkpointing(
                self.agent,
                self.adaptive_checkpoint_metadata,
            )
        if cfg.load_checkpoint:
            component_paths = (
                cfg.actor_critic_checkpoint_path,
                cfg.encoder_checkpoint_path,
            )
            component_loading = any(component_paths)
            if component_loading and not all(component_paths):
                raise ValueError(
                    "Adaptive component loading requires both actor_critic_checkpoint_path "
                    "and encoder_checkpoint_path."
                )
            if component_loading and cfg.checkpoint_path:
                raise ValueError(
                    "checkpoint_path is mutually exclusive with adaptive component checkpoint paths."
                )
            if component_loading and not cfg.velocity_response_adaptation.enabled:
                raise ValueError(
                    "Adaptive component checkpoints require velocity_response_adaptation.enabled=True."
                )
            if not component_loading and not cfg.checkpoint_path:
                raise ValueError(
                    "RLPolicyCfg.load_checkpoint=True requires checkpoint_path or both adaptive "
                    "component checkpoint paths."
                )
            try:
                if component_loading:
                    assert self.adaptive_checkpoint_metadata is not None
                    load_adaptive_agent_components(
                        self.agent,
                        str(Path(cfg.actor_critic_checkpoint_path).expanduser()),
                        str(Path(cfg.encoder_checkpoint_path).expanduser()),
                        self.adaptive_checkpoint_metadata,
                    )
                else:
                    if cfg.velocity_response_adaptation.enabled:
                        assert self.adaptive_checkpoint_metadata is not None
                        validate_adaptive_agent_checkpoint(
                            str(Path(cfg.checkpoint_path).expanduser()),
                            self.adaptive_checkpoint_metadata,
                            self.device,
                        )
                    self.agent.load(str(Path(cfg.checkpoint_path).expanduser()))
            except Exception as exc:
                preview_horizon = self.observation_adapter.reference_preview_future_steps
                raise RuntimeError(
                    f"Failed to load RL checkpoint for observation_mode='{self.observation_mode}', "
                    f"raw_observation_dim={self.observation_adapter.raw_observation_dim}, "
                    f"base_policy_input_dim={self.observation_adapter.input_dim}, "
                    f"model_input_dim={self.model_input_dim}, "
                    "velocity_response_history_length="
                    f"{cfg.velocity_response_adaptation.history_length}, "
                    f"velocity_response_latent_dim={cfg.velocity_response_adaptation.latent_dim}, "
                    f"raw_preview_future_steps={preview_horizon}, "
                    "reference_preview_base_offset="
                    f"{self.observation_adapter.reference_preview_base_offset}, "
                    "effective_preview_future_steps="
                    f"{self.observation_adapter.effective_reference_preview_future_steps}, "
                    f"reference_preview_samples={self.observation_adapter.reference_preview_samples}, "
                    f"policy_preview_offsets={self.observation_adapter.preview_offsets}, "
                    "policy_preview_source_offsets="
                    f"{self.observation_adapter.preview_source_offsets}. Check that "
                    "the checkpoint was trained with the same observation layout."
                ) from exc

        self.timestep = 0
        self.policy_input = torch.zeros((self.num_envs, self.model_input_dim), device=self.device)
        self.normalized_action = torch.zeros((self.num_envs, 1), device=self.device)
        self.physical_action = torch.zeros((self.num_envs, 1), device=self.device)
        self.velocity_response_latent = torch.zeros(
            (self.num_envs, cfg.velocity_response_adaptation.latent_dim),
            device=self.device,
        ) if cfg.velocity_response_adaptation.enabled else None

    @property
    def observation_mode(self) -> str:
        return self.observation_adapter.observation_mode

    def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None):
        """Reset RL policy-side history for all envs or selected envs."""
        env_ids = self._env_ids_tensor(env_ids)
        if env_ids.numel() == 0:
            return
        self.observation_adapter.reset(env_ids)
        self.state_predictor.reset(env_ids)
        if self.response_history is not None:
            self.response_history.reset(env_ids)
        self.normalized_action[env_ids] = 0.0
        self.physical_action[env_ids] = 0.0
        self.policy_input[env_ids] = 0.0
        self.privileged_response_parameters[env_ids] = 0.0
        if self.velocity_response_latent is not None:
            self.velocity_response_latent[env_ids] = 0.0
        if env_ids.numel() == self.num_envs:
            self.timestep = 0

    def act(self, observations: dict[str, torch.Tensor] | torch.Tensor, extras: dict | None = None) -> torch.Tensor:
        """Compute a physical velocity-increment action from environment observations."""
        raw_obs = self._extract_policy_observation(observations)
        if raw_obs.ndim != 2 or raw_obs.shape[0] != self.num_envs or raw_obs.shape[1] < 11:
            raise ValueError(
                f"RLPolicy expects observation shape ({self.num_envs}, D) with D >= 11, "
                f"got {tuple(raw_obs.shape)}."
            )

        if self.response_history is not None:
            self._update_response_history(raw_obs, extras)

        reference_positions = self._extract_reference_positions(raw_obs)
        model_legacy_obs = self.state_predictor.predict(
            raw_obs[:, :11],
            reference_positions=reference_positions,
        )
        # The predictor replaces only the legacy plant-state prefix. Appended
        # raw references stay intact so a relative-preview adapter can rebase
        # them from reference offset D after predicting the plant to k + D.
        if raw_obs.shape[1] > 11:
            model_obs = torch.cat((model_legacy_obs, raw_obs[:, 11:]), dim=-1)
        else:
            model_obs = model_legacy_obs
        base_policy_input = self.observation_adapter.transform(model_obs, update_history=True)
        if self.response_history is not None:
            model_input = torch.cat(
                (
                    base_policy_input,
                    self.response_history.encoder_input,
                    self.privileged_response_parameters,
                ),
                dim=-1,
            )
        else:
            model_input = base_policy_input
        self.policy_input.copy_(model_input)

        outputs = self.agent.act(self.policy_input, timestep=self.timestep, timesteps=self.timestep)
        action, info = self._select_normalized_action(outputs)
        del info
        self.normalized_action.copy_(torch.clamp(action, -1.0, 1.0))
        self.physical_action.copy_(self.normalized_action * self.physical_action_limit)
        adaptive_model = self.models.get("policy")
        if isinstance(adaptive_model, AdaptiveVelocityActor):
            if adaptive_model.last_latent.shape != self.velocity_response_latent.shape:
                raise RuntimeError(
                    "Adaptive actor returned an unexpected latent shape: "
                    f"{tuple(adaptive_model.last_latent.shape)}."
                )
            self.velocity_response_latent.copy_(adaptive_model.last_latent)
        self.state_predictor.update_after_action(self.physical_action)
        self.timestep += 1
        return self.physical_action.clone()

    def get_state(self) -> dict[str, torch.Tensor]:
        """Return RL policy diagnostics."""
        state = {
            "rl_normalized_action": self.normalized_action[:, 0],
            "rl_physical_action": self.physical_action[:, 0],
            **self.observation_adapter.get_state(),
            **self.state_predictor.get_state(),
        }
        if self.response_history is not None:
            state.update(
                {
                    "policy_velocity_response_error": self.response_history.latest_error,
                    "policy_velocity_response_actual_history": (
                        self.response_history.actual_velocity_history
                    ),
                    "policy_velocity_response_command_history": (
                        self.response_history.commanded_velocity_history
                    ),
                    "policy_velocity_response_error_history": (
                        self.response_history.error_history
                    ),
                    "policy_velocity_response_latent": self.velocity_response_latent,
                }
            )
        return state

    def to(self, device: str | torch.device):
        device = torch.device(device)
        for name, value in vars(self).items():
            if isinstance(value, torch.Tensor):
                setattr(self, name, value.to(device=device))
        self.observation_adapter.to(device)
        self.state_predictor.to(device)
        if self.response_history is not None:
            self.response_history.to(device)
        self.device = device
        return self

    def _update_response_history(
        self,
        raw_obs: torch.Tensor,
        extras: dict | None,
    ) -> None:
        assert self.response_history is not None
        boundary_mask = self.response_history.needs_transition.clone()
        update_env_ids = (~boundary_mask).nonzero(as_tuple=False).squeeze(-1)
        boundary_env_ids = boundary_mask.nonzero(as_tuple=False).squeeze(-1)
        if update_env_ids.numel() > 0:
            if not isinstance(extras, Mapping):
                raise ValueError(
                    "velocity_response_adaptation requires extras['step']['command_z'] "
                    "after the reset observation."
                )
            step_extras = extras.get("step")
            if not isinstance(step_extras, Mapping) or "command_z" not in step_extras:
                raise ValueError(
                    "velocity_response_adaptation requires extras['step']['command_z']."
                )
            command_z = torch.as_tensor(
                step_extras["command_z"],
                device=self.device,
                dtype=torch.float32,
            )
            if command_z.ndim == 2 and command_z.shape[-1] == 1:
                command_z = command_z[:, 0]
            if command_z.shape != (self.num_envs,):
                raise ValueError(
                    "extras['step']['command_z'] must have shape "
                    f"({self.num_envs},), got {tuple(command_z.shape)}."
                )
            measured_vrz = raw_obs[:, ObservationIndex.VRZ]
            self.response_history.update(
                measured_vrz[update_env_ids],
                command_z[update_env_ids],
                update_env_ids,
            )
        if boundary_env_ids.numel() > 0:
            self.response_history.mark_transition_boundary(boundary_env_ids)

    def _extract_reference_positions(self, raw_obs: torch.Tensor) -> torch.Tensor | None:
        if not self.state_predictor.active:
            return None
        fields = self.observation_adapter.raw_observation_fields
        if not any(name.startswith("pg_") and name != "pg_0" for name in fields):
            return None
        return extract_reference_positions(
            raw_obs,
            fields,
            self.state_predictor.delay_step,
        )

    def _select_normalized_action(self, outputs) -> tuple[torch.Tensor, dict]:
        info = outputs[-1] if isinstance(outputs, tuple) and isinstance(outputs[-1], dict) else {}
        if self.cfg.deterministic and "no_pert_mean_actions" in info:
            action = info["no_pert_mean_actions"]
        elif isinstance(outputs, tuple):
            action = outputs[0]
        else:
            action = outputs
        action = action.to(device=self.device, dtype=torch.float32)
        if action.ndim == 1:
            action = action.unsqueeze(-1)
        if action.shape != (self.num_envs, 1):
            raise ValueError(f"RLPolicy expected normalized action shape ({self.num_envs}, 1), got {tuple(action.shape)}.")
        return action, info

    def _resolve_predictor_cfg_defaults(self):
        predictor_cfg = self.cfg.state_predictor
        if _is_auto(predictor_cfg.delay_step):
            predictor_cfg.delay_step = 0
        if _is_auto(predictor_cfg.step_dt) or float(predictor_cfg.step_dt) <= 0.0:
            predictor_cfg.step_dt = self.step_dt
        if _is_auto(predictor_cfg.max_acc) or float(predictor_cfg.max_acc) <= 0.0:
            predictor_cfg.max_acc = self.physical_action_limit / self.step_dt
        if _is_auto(predictor_cfg.max_velocity):
            predictor_cfg.max_velocity = 0.0
        if _is_auto(predictor_cfg.plank_length):
            predictor_cfg.plank_length = 1.06
        if _is_auto(predictor_cfg.rope_length):
            predictor_cfg.rope_length = 0.9
        if _is_auto(predictor_cfg.gravity):
            predictor_cfg.gravity = 9.81
        if _is_auto(predictor_cfg.ball_mass):
            predictor_cfg.ball_mass = 0.0005
        if _is_auto(predictor_cfg.ball_radius):
            predictor_cfg.ball_radius = 0.023


def _is_auto(value) -> bool:
    return isinstance(value, str) and value.lower() == "auto"
