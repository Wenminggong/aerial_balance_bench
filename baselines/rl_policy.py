"""RL baseline policy wrappers for Aerial-Balance-Bench."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from gymnasium import spaces

from .base_policy import BasePolicy, BasePolicyCfg
from .model_state_predictor import (
    VelocityModelStatePredictorCfg,
    make_acceleration_state_predictor,
    make_velocity_state_predictor,
)
from .rl_models import MLPNetworkCfg, make_agent_class_and_cfg, make_models
from .rl_observation_adapter import RLObservationAdapter, RLObservationAdapterCfg


@dataclass
class RLPolicyCfg(BasePolicyCfg):
    """Configuration for RL policy deployment/evaluation."""

    name: str = "rl"
    algorithm: str = "rpo"
    observation_mode: str = "legacy8"
    checkpoint_path: str | None = None
    load_checkpoint: bool = True
    deterministic: bool = True
    physical_action_limit: float | str = "auto"
    command_history_length: int | str = 0
    network: MLPNetworkCfg = field(default_factory=MLPNetworkCfg)
    agent: dict[str, Any] = field(default_factory=dict)
    state_predictor: VelocityModelStatePredictorCfg = field(default_factory=VelocityModelStatePredictorCfg)

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
        if "checkpoint_path" in policy_data:
            checkpoint_path = policy_data["checkpoint_path"]
            cfg.checkpoint_path = None if checkpoint_path in (None, "null", "") else str(checkpoint_path)
        if "load_checkpoint" in policy_data:
            cfg.load_checkpoint = bool(policy_data["load_checkpoint"])
        if "deterministic" in policy_data:
            cfg.deterministic = bool(policy_data["deterministic"])
        if "physical_action_limit" in policy_data:
            cfg.physical_action_limit = policy_data["physical_action_limit"]
        if "command_history_length" in policy_data:
            cfg.command_history_length = policy_data["command_history_length"]
        elif "action_history_length" in policy_data:
            cfg.command_history_length = policy_data["action_history_length"]

        cfg.network = MLPNetworkCfg.from_dict(policy_data.get("network", policy_data.get("model", {})))
        cfg.agent = dict(policy_data.get("agent", {}))
        cfg.state_predictor = VelocityModelStatePredictorCfg.from_dict(policy_data.get("state_predictor", {}))
        return cfg


class _IdentityStatePredictor:
    """No-op state predictor for interfaces without policy-side compensation."""

    def __init__(self, num_envs: int, device: str | torch.device):
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.predicted_observation = torch.zeros((self.num_envs, 11), device=self.device)
        self.last_action = torch.zeros((self.num_envs, 1), device=self.device)

    @property
    def active(self) -> bool:
        return False

    def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        else:
            env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        if env_ids.numel() == 0:
            return
        self.predicted_observation[env_ids] = 0.0
        self.last_action[env_ids] = 0.0

    def predict(
        self,
        observation: torch.Tensor,
        error_prev1: torch.Tensor | None = None,
        extras: Mapping | None = None,
    ) -> torch.Tensor:
        del error_prev1, extras
        observation = observation.to(device=self.device, dtype=torch.float32)
        if observation.shape != (self.num_envs, 11):
            raise ValueError(
                f"IdentityStatePredictor expects observation shape ({self.num_envs}, 11), "
                f"got {tuple(observation.shape)}."
            )
        self.predicted_observation.copy_(observation)
        return observation

    def update_after_action(self, action: torch.Tensor):
        action = action.to(device=self.device, dtype=torch.float32)
        if action.ndim == 1:
            action = action.unsqueeze(-1)
        if action.shape == (self.num_envs, 1):
            self.last_action.copy_(action)

    def get_state(self) -> dict[str, torch.Tensor]:
        return {
            "policy_predictor_enabled": torch.zeros((self.num_envs,), device=self.device),
        }

    def to(self, device: str | torch.device):
        device = torch.device(device)
        for name, value in vars(self).items():
            if isinstance(value, torch.Tensor):
                setattr(self, name, value.to(device=device))
        self.device = device
        return self


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
        interface_name: str = "velocity",
    ):
        super().__init__(cfg, num_envs, device)
        self.interface_name = str(interface_name).lower()
        self.step_dt = float(step_dt)
        self.physical_action_limit = float(physical_action_limit)
        if self.interface_name not in {"velocity", "acceleration"}:
            raise ValueError("RLPolicy supports only interface_name='velocity' or 'acceleration'.")
        if self.step_dt <= 0.0:
            raise ValueError("RLPolicy requires a positive step_dt.")
        if self.physical_action_limit <= 0.0:
            raise ValueError("RLPolicy requires a positive physical_action_limit.")

        self._resolve_predictor_cfg_defaults()
        adapter_cfg = RLObservationAdapterCfg(
            observation_mode=cfg.observation_mode,
            command_history_length=cfg.command_history_length,
        )
        self.observation_adapter = RLObservationAdapter(adapter_cfg, num_envs, self.device)
        if self.observation_adapter.command_history_length > 0 and self.interface_name != "acceleration":
            raise ValueError(
                f"observation_mode='{self.observation_adapter.observation_mode}' is supported only with "
                "interface_name='acceleration'."
            )
        self.state_predictor = self._make_state_predictor()

        self.normalized_action_space = spaces.Box(
            low=np.array([-1.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.policy_observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.observation_adapter.input_dim,),
            dtype=np.float32,
        )
        self.models = make_models(
            cfg.algorithm,
            cfg.network,
            self.policy_observation_space,
            self.normalized_action_space,
            self.device,
        )
        agent_cls, agent_cfg = make_agent_class_and_cfg(
            cfg.algorithm,
            cfg.agent,
            self.policy_observation_space,
            self.device,
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
        if cfg.load_checkpoint:
            if not cfg.checkpoint_path:
                raise ValueError("RLPolicyCfg.load_checkpoint=True requires checkpoint_path.")
            self.agent.load(str(Path(cfg.checkpoint_path).expanduser()))

        self.timestep = 0
        self.policy_input = torch.zeros((self.num_envs, self.observation_adapter.input_dim), device=self.device)
        self.normalized_action = torch.zeros((self.num_envs, 1), device=self.device)
        self.physical_action = torch.zeros((self.num_envs, 1), device=self.device)
        self.command_z = torch.zeros((self.num_envs, 1), device=self.device)
        self.human_velocity_needs_reset = torch.ones(
            (self.num_envs,), dtype=torch.bool, device=self.device
        )

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
        self.normalized_action[env_ids] = 0.0
        self.physical_action[env_ids] = 0.0
        self.command_z[env_ids] = 0.0
        self.policy_input[env_ids] = 0.0
        self.human_velocity_needs_reset[env_ids] = True
        if env_ids.numel() == self.num_envs:
            self.timestep = 0

    def act(self, observations: dict[str, torch.Tensor] | torch.Tensor, extras: dict | None = None) -> torch.Tensor:
        """Compute a physical high-level action from environment observations."""
        raw_obs = self._extract_policy_observation(observations)
        if raw_obs.shape != (self.num_envs, 11):
            raise ValueError(f"RLPolicy expects observation shape ({self.num_envs}, 11), got {tuple(raw_obs.shape)}.")

        model_obs = self.state_predictor.predict(raw_obs, extras=extras)
        self._update_human_velocity_history(extras)
        self.policy_input.copy_(self.observation_adapter.transform(model_obs, update_history=True))

        outputs = self.agent.act(self.policy_input, timestep=self.timestep, timesteps=self.timestep)
        action, info = self._select_normalized_action(outputs)
        del info
        self.normalized_action.copy_(torch.clamp(action, -1.0, 1.0))
        self.physical_action.copy_(self.normalized_action * self.physical_action_limit)
        self._update_command_history_after_action()
        self.state_predictor.update_after_action(self.physical_action)
        self.timestep += 1
        return self.physical_action.clone()

    def get_state(self) -> dict[str, torch.Tensor]:
        """Return RL policy diagnostics."""
        state = {
            "rl_normalized_action": self.normalized_action[:, 0],
            "rl_physical_action": self.physical_action[:, 0],
            "rl_policy_command_z": self.command_z[:, 0],
            **self.observation_adapter.get_state(),
            **self.state_predictor.get_state(),
        }
        return state

    def to(self, device: str | torch.device):
        device = torch.device(device)
        for name, value in vars(self).items():
            if isinstance(value, torch.Tensor):
                setattr(self, name, value.to(device=device))
        self.observation_adapter.to(device)
        self.state_predictor.to(device)
        self.device = device
        return self

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

    def _update_command_history_after_action(self):
        if self.observation_adapter.command_history_length <= 0:
            return
        max_acc = float(self.cfg.state_predictor.max_acc)
        if max_acc <= 0.0:
            raise ValueError(
                f"{self.observation_adapter.observation_mode} requires a positive acceleration command limit."
            )
        next_command_z = self.command_z + self.physical_action
        self.command_z.copy_(torch.clamp(next_command_z, min=-max_acc, max=max_acc))
        self.observation_adapter.update_command_history(self.command_z)

    def _update_human_velocity_history(self, extras: Mapping | None):
        if self.observation_adapter.observation_mode != "error9_acc_vhz_history":
            return
        human_velocity_z = None
        if isinstance(extras, Mapping):
            step_info = extras.get("step", {})
            if isinstance(step_info, Mapping):
                human_velocity_z = step_info.get("external_disturbance_vel_z")
            if human_velocity_z is None:
                human_velocity_z = extras.get("external_disturbance_vel_z")
        if human_velocity_z is None:
            raise ValueError(
                "observation_mode='error9_acc_vhz_history' requires extras['step'] to include "
                "'external_disturbance_vel_z'."
            )

        human_velocity_z = torch.as_tensor(human_velocity_z, device=self.device, dtype=torch.float32)
        if human_velocity_z.ndim == 1:
            human_velocity_z = human_velocity_z.unsqueeze(-1)
        if human_velocity_z.shape != (self.num_envs, 1):
            raise ValueError(
                "RLPolicy expected human-side Z velocity shape "
                f"({self.num_envs},) or ({self.num_envs}, 1), got {tuple(human_velocity_z.shape)}."
            )
        if torch.any(self.human_velocity_needs_reset):
            human_velocity_z = human_velocity_z.clone()
            human_velocity_z[self.human_velocity_needs_reset] = 0.0
            self.human_velocity_needs_reset[:] = False
        self.observation_adapter.update_human_velocity_history(human_velocity_z)

    def _resolve_predictor_cfg_defaults(self):
        predictor_cfg = self.cfg.state_predictor
        if self.interface_name == "acceleration":
            if _is_auto(predictor_cfg.delay_step):
                predictor_cfg.delay_step = 0
            if _is_auto(predictor_cfg.step_dt) or float(predictor_cfg.step_dt) <= 0.0:
                predictor_cfg.step_dt = self.step_dt
            if _is_auto(predictor_cfg.max_delta_acc) or float(predictor_cfg.max_delta_acc) <= 0.0:
                predictor_cfg.max_delta_acc = self.physical_action_limit
            if _is_auto(predictor_cfg.max_acc) or float(predictor_cfg.max_acc) <= 0.0:
                predictor_cfg.max_acc = 5.0
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
            if _is_auto(predictor_cfg.ball_position_offset):
                predictor_cfg.ball_position_offset = 0.33
            self._resolve_acceleration_response_cfg_defaults(predictor_cfg)
            return

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
        if _is_auto(predictor_cfg.ball_position_offset):
            predictor_cfg.ball_position_offset = 0.33

    def _resolve_acceleration_response_cfg_defaults(self, predictor_cfg):
        if _is_auto(predictor_cfg.acceleration_response_tau_s):
            predictor_cfg.acceleration_response_tau_s = 0.0
        if _is_auto(predictor_cfg.acceleration_response_gain):
            predictor_cfg.acceleration_response_gain = 1.0
        if _is_auto(predictor_cfg.acceleration_response_bias):
            predictor_cfg.acceleration_response_bias = 0.0
        if _is_auto(predictor_cfg.acceleration_response_noise_mode):
            predictor_cfg.acceleration_response_noise_mode = "none"
        if _is_auto(predictor_cfg.acceleration_response_noise_std):
            predictor_cfg.acceleration_response_noise_std = 0.0
        if _is_auto(predictor_cfg.acceleration_response_ou_theta):
            predictor_cfg.acceleration_response_ou_theta = 0.0
        if _is_auto(predictor_cfg.acceleration_response_noise_clip):
            predictor_cfg.acceleration_response_noise_clip = 0.0
        if _is_auto(predictor_cfg.acceleration_response_max_abs_acc):
            predictor_cfg.acceleration_response_max_abs_acc = 0.0

    def _make_state_predictor(self):
        if self.interface_name == "velocity":
            return make_velocity_state_predictor(self.cfg.state_predictor, self.num_envs, self.device)
        if bool(self.cfg.state_predictor.enabled):
            return make_acceleration_state_predictor(self.cfg.state_predictor, self.num_envs, self.device)
        return _IdentityStatePredictor(self.num_envs, self.device)


def _is_auto(value) -> bool:
    return isinstance(value, str) and value.lower() == "auto"
