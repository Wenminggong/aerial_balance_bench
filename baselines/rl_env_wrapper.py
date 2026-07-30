"""Environment wrappers used for RL training."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import torch
from gymnasium import Wrapper, spaces

from .base_policy import ObservationIndex
from .rl_observation_adapter import RLObservationAdapter, RLObservationAdapterCfg
from .velocity_response_adaptation import (
    PRIVILEGED_RESPONSE_FIELDS,
    VelocityResponseAdaptationCfg,
    VelocityResponseHistoryBuffer,
)


class NormalizedRLTrainingWrapper(Wrapper):
    """Expose normalized actions and adapted observations to skrl trainers."""

    def __init__(
        self,
        env,
        adapter_cfg: RLObservationAdapterCfg,
        physical_action_limit: float,
        raw_observation_dim: int | None = None,
        raw_observation_fields: Sequence[str] | None = None,
        adaptation_cfg: VelocityResponseAdaptationCfg | None = None,
    ):
        super().__init__(env)
        self.physical_action_limit = float(physical_action_limit)
        if self.physical_action_limit <= 0.0:
            raise ValueError("NormalizedRLTrainingWrapper requires a positive physical_action_limit.")
        base_env = env.unwrapped
        raw_observation_dim, raw_observation_fields = self._resolve_raw_observation_spec(
            base_env,
            raw_observation_dim,
            raw_observation_fields,
        )
        self.adapter = RLObservationAdapter(
            adapter_cfg,
            base_env.num_envs,
            base_env.device,
            raw_observation_dim=raw_observation_dim,
            raw_observation_fields=raw_observation_fields,
        )
        self.raw_observation_dim = self.adapter.raw_observation_dim
        self.raw_observation_fields = self.adapter.raw_observation_fields
        self.adaptation_cfg = adaptation_cfg or VelocityResponseAdaptationCfg()
        self.adaptation_cfg.validate()
        self.base_policy_observation_dim = self.adapter.input_dim
        self.base_policy_observation_fields = self.adapter.field_names
        self.response_history: VelocityResponseHistoryBuffer | None = None
        self.privileged_response_parameters = torch.zeros(
            (base_env.num_envs, len(PRIVILEGED_RESPONSE_FIELDS)),
            device=base_env.device,
            dtype=torch.float32,
        )
        if self.adaptation_cfg.enabled:
            self.response_history = VelocityResponseHistoryBuffer(
                base_env.num_envs,
                self.adaptation_cfg.history_length,
                base_env.device,
            )
        adaptation_dim = (
            self.adaptation_cfg.encoder_input_dim + len(PRIVILEGED_RESPONSE_FIELDS)
            if self.adaptation_cfg.enabled
            else 0
        )
        self.policy_observation_dim = self.base_policy_observation_dim + adaptation_dim
        self.policy_observation_fields = (
            *self.base_policy_observation_fields,
            *(self.adaptation_cfg.history_field_names if self.adaptation_cfg.enabled else ()),
            *(PRIVILEGED_RESPONSE_FIELDS if self.adaptation_cfg.enabled else ()),
        )

        self.action_space = spaces.Box(
            low=np.array([-1.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32,
        )
        policy_observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.policy_observation_dim,),
            dtype=np.float32,
        )
        # skrl's IsaacLab wrapper reads observation spaces from
        # env.unwrapped.single_observation_space["policy"].  Keep the exposed
        # space aligned with the adapted observation returned by reset/step.
        self.observation_space = {"policy": policy_observation_space}
        self.single_action_space = self.action_space
        self.single_observation_space = {"policy": policy_observation_space}
        self.last_normalized_action = torch.zeros((self.num_envs, 1), device=self.device)
        self.last_physical_action = torch.zeros((self.num_envs, 1), device=self.device)

    @property
    def unwrapped(self):
        """Expose this wrapper to skrl so adapted spaces are used."""
        return self

    @property
    def num_envs(self) -> int:
        return self.env.unwrapped.num_envs

    @property
    def device(self):
        return self.env.unwrapped.device

    @property
    def max_episode_length(self) -> int:
        return self.env.unwrapped.max_episode_length

    def reset(self, **kwargs):
        observations, infos = self.env.reset(**kwargs)
        self.adapter.reset()
        if self.response_history is not None:
            self.response_history.reset()
            self._refresh_privileged_response_parameters()
        self.last_normalized_action.zero_()
        self.last_physical_action.zero_()
        self._copy_benchmark_metrics_to_episode_info(infos)
        return {"policy": self._adapt_observation(observations["policy"])}, infos

    def step(self, action):
        if torch.is_tensor(action):
            normalized_action = action.to(device=self.device, dtype=torch.float32)
        else:
            normalized_action = torch.as_tensor(action, device=self.device, dtype=torch.float32)
        if normalized_action.ndim == 1:
            normalized_action = normalized_action.unsqueeze(-1)
        normalized_action = torch.clamp(normalized_action, -1.0, 1.0)
        physical_action = normalized_action * self.physical_action_limit
        observations, rewards, terminated, truncated, infos = self.env.step(physical_action)

        done_env_ids = (terminated | truncated).nonzero(as_tuple=False).squeeze(-1)
        if self.response_history is not None:
            active_env_ids = (~(terminated | truncated)).nonzero(as_tuple=False).squeeze(-1)
            if active_env_ids.numel() > 0:
                command_z = self._command_z_from_infos(infos)
                actual_vrz = observations["policy"][:, ObservationIndex.VRZ]
                self.response_history.update(
                    actual_vrz[active_env_ids],
                    command_z[active_env_ids],
                    active_env_ids,
                )
            if done_env_ids.numel() > 0:
                self.response_history.reset(done_env_ids)
            # AerialBalanceEnv refreshes robustness parameters during autoreset,
            # after terminal extras are built. Read the manager's current state
            # so done environments receive the new episode's privileged values.
            self._refresh_privileged_response_parameters()
        if done_env_ids.numel() > 0:
            self.adapter.reset(done_env_ids)

        self.last_normalized_action.copy_(normalized_action)
        self.last_physical_action.copy_(physical_action)
        adapted_observation = self._adapt_observation(observations["policy"])
        infos.setdefault("rl", {})
        infos["rl"]["normalized_action"] = self.last_normalized_action
        infos["rl"]["physical_action"] = self.last_physical_action
        infos["rl"]["policy_input"] = adapted_observation
        if self.response_history is not None:
            infos["rl"]["velocity_response_actual_history"] = (
                self.response_history.actual_velocity_history
            )
            infos["rl"]["velocity_response_command_history"] = (
                self.response_history.commanded_velocity_history
            )
            infos["rl"]["velocity_response_error_history"] = (
                self.response_history.error_history
            )
            infos["rl"]["velocity_response_error"] = self.response_history.latest_error
            infos["rl"]["velocity_response_privileged_parameters"] = (
                self.privileged_response_parameters
            )
        self._copy_benchmark_metrics_to_episode_info(infos)
        return {"policy": adapted_observation}, rewards, terminated, truncated, infos

    def _adapt_observation(self, observation: torch.Tensor) -> torch.Tensor:
        base_observation = self.adapter.transform(observation, update_history=True)
        if self.response_history is None:
            return base_observation
        return torch.cat(
            (
                base_observation,
                self.response_history.encoder_input,
                self.privileged_response_parameters,
            ),
            dim=-1,
        )

    def _refresh_privileged_response_parameters(self) -> None:
        robustness = getattr(self.env.unwrapped, "robustness", None)
        if robustness is None:
            raise ValueError(
                "velocity_response_adaptation requires env.unwrapped.robustness "
                "to expose current response parameters."
            )
        values = []
        for field_name in PRIVILEGED_RESPONSE_FIELDS:
            value = getattr(robustness, field_name, None)
            if value is None:
                raise ValueError(
                    "velocity_response_adaptation requires robustness."
                    f"{field_name}."
                )
            value = torch.as_tensor(value, device=self.device, dtype=torch.float32)
            if value.ndim == 2 and value.shape[-1] == 1:
                value = value[:, 0]
            if value.shape != (self.num_envs,):
                raise ValueError(
                    f"robustness.{field_name} must have shape ({self.num_envs},), "
                    f"got {tuple(value.shape)}."
                )
            if not torch.isfinite(value).all():
                raise ValueError(f"robustness.{field_name} must contain finite values.")
            values.append(value)
        self.privileged_response_parameters.copy_(torch.stack(values, dim=-1))

    def _command_z_from_infos(self, infos: Mapping) -> torch.Tensor:
        step_infos = infos.get("step")
        if not isinstance(step_infos, Mapping) or "command_z" not in step_infos:
            raise ValueError(
                "velocity_response_adaptation requires infos['step']['command_z'] from the environment."
            )
        command_z = torch.as_tensor(
            step_infos["command_z"],
            device=self.device,
            dtype=torch.float32,
        )
        if command_z.ndim == 2 and command_z.shape[-1] == 1:
            command_z = command_z[:, 0]
        if command_z.shape != (self.num_envs,):
            raise ValueError(
                "infos['step']['command_z'] must have shape "
                f"({self.num_envs},), got {tuple(command_z.shape)}."
            )
        return command_z

    def _copy_benchmark_metrics_to_episode_info(self, infos: dict):
        """Expose benchmark metrics through skrl's default environment-info hook."""
        benchmark = infos.get("benchmark")
        if not isinstance(benchmark, dict):
            return
        episode_info = infos.setdefault("episode", {})
        for key, value in benchmark.items():
            if torch.is_tensor(value):
                is_scalar = value.numel() == 1
            else:
                is_scalar = isinstance(value, (bool, int, float, np.number))
            if is_scalar:
                episode_info[key] = value

    @staticmethod
    def _resolve_raw_observation_spec(
        base_env,
        raw_observation_dim: int | None,
        raw_observation_fields: Sequence[str] | None,
    ) -> tuple[int | None, Sequence[str] | None]:
        """Resolve the raw observation layout without assuming a fixed dimension."""
        if raw_observation_fields is None:
            raw_observation_fields = getattr(base_env, "observation_fields", None)
        if raw_observation_dim is None:
            raw_observation_dim = getattr(base_env, "raw_observation_dim", None)
        if raw_observation_dim is None and raw_observation_fields is not None:
            raw_observation_dim = len(raw_observation_fields)
        if raw_observation_dim is None:
            raw_observation_dim = NormalizedRLTrainingWrapper._observation_space_dimension(base_env)
        return raw_observation_dim, raw_observation_fields

    @staticmethod
    def _observation_space_dimension(base_env) -> int | None:
        for attribute in ("single_observation_space", "observation_space"):
            observation_space = getattr(base_env, attribute, None)
            if isinstance(observation_space, spaces.Dict):
                observation_space = observation_space.spaces.get("policy")
            elif isinstance(observation_space, Mapping):
                observation_space = observation_space.get("policy")
            shape = getattr(observation_space, "shape", None)
            if shape is not None and len(shape) == 1:
                return int(shape[0])
        return None
