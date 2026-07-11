"""Environment wrappers used for RL training."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import torch
from gymnasium import Wrapper, spaces

from .rl_observation_adapter import RLObservationAdapter, RLObservationAdapterCfg


class NormalizedRLTrainingWrapper(Wrapper):
    """Expose normalized actions and adapted observations to skrl trainers."""

    def __init__(
        self,
        env,
        adapter_cfg: RLObservationAdapterCfg,
        physical_action_limit: float,
        raw_observation_dim: int | None = None,
        raw_observation_fields: Sequence[str] | None = None,
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
        self.policy_observation_dim = self.adapter.input_dim
        self.policy_observation_fields = self.adapter.field_names

        self.action_space = spaces.Box(
            low=np.array([-1.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32,
        )
        policy_observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.adapter.input_dim,),
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
        self.last_normalized_action.zero_()
        self.last_physical_action.zero_()
        self._copy_benchmark_metrics_to_episode_info(infos)
        return {"policy": self.adapter.transform(observations["policy"], update_history=True)}, infos

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
        if done_env_ids.numel() > 0:
            self.adapter.reset(done_env_ids)

        self.last_normalized_action.copy_(normalized_action)
        self.last_physical_action.copy_(physical_action)
        adapted_observation = self.adapter.transform(observations["policy"], update_history=True)
        infos.setdefault("rl", {})
        infos["rl"]["normalized_action"] = self.last_normalized_action
        infos["rl"]["physical_action"] = self.last_physical_action
        infos["rl"]["policy_input"] = self.adapter.last_policy_input
        self._copy_benchmark_metrics_to_episode_info(infos)
        return {"policy": adapted_observation}, rewards, terminated, truncated, infos

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
