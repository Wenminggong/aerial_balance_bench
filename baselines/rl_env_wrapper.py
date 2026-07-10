"""Environment wrappers used for RL training."""

from __future__ import annotations

import numpy as np
import torch
from gymnasium import Wrapper, spaces

from .rl_observation_adapter import RLObservationAdapter, RLObservationAdapterCfg


TARGET_POSITION_BENCHMARK_METRICS = (
    "success_rate",
    "steady_state_error",
    "convergence_time",
    "climbing_time",
)


class NormalizedRLTrainingWrapper(Wrapper):
    """Expose normalized actions and adapted observations to skrl trainers."""

    def __init__(
        self,
        env,
        adapter_cfg: RLObservationAdapterCfg,
        physical_action_limit: float,
    ):
        super().__init__(env)
        self.physical_action_limit = float(physical_action_limit)
        if self.physical_action_limit <= 0.0:
            raise ValueError("NormalizedRLTrainingWrapper requires a positive physical_action_limit.")
        self.adapter = RLObservationAdapter(adapter_cfg, env.unwrapped.num_envs, env.unwrapped.device)
        interface_name = getattr(getattr(env.unwrapped, "cfg", None), "interface_name", None)
        if self.adapter.command_history_length > 0 and interface_name != "acceleration":
            raise ValueError(
                f"observation_mode='{self.adapter.observation_mode}' is supported only with "
                "interface_name='acceleration'."
            )

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
        self._update_adapter_human_velocity_history(infos)
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

        self.last_normalized_action.copy_(normalized_action)
        self.last_physical_action.copy_(physical_action)
        self._update_adapter_command_history(infos)
        self._update_adapter_human_velocity_history(infos)

        done_env_ids = (terminated | truncated).nonzero(as_tuple=False).squeeze(-1)
        if done_env_ids.numel() > 0:
            self.adapter.reset(done_env_ids)

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
        for key in TARGET_POSITION_BENCHMARK_METRICS:
            value = benchmark.get(key)
            if value is not None:
                episode_info[key] = value

    def _update_adapter_command_history(self, infos: dict):
        """Feed the latest absolute acceleration command into delay-aware observations."""
        if self.adapter.command_history_length <= 0:
            return
        step_info = infos.get("step", {})
        command_z = step_info.get("command_z")
        if command_z is None:
            command_z = step_info.get("arz_cmd")
        if command_z is None:
            raise ValueError(
                f"observation_mode='{self.adapter.observation_mode}' requires step extras to include "
                "'command_z' or 'arz_cmd'."
            )
        self.adapter.update_command_history(command_z)

    def _update_adapter_human_velocity_history(self, infos: dict):
        """Feed observed human-side Z velocity into the VHZ history observation."""
        if self.adapter.observation_mode != "error9_acc_vhz_history":
            return
        step_info = infos.get("step", {})
        human_velocity_z = step_info.get("external_disturbance_vel_z")
        if human_velocity_z is None:
            raise ValueError(
                "observation_mode='error9_acc_vhz_history' requires step extras to include "
                "'external_disturbance_vel_z'."
            )
        self.adapter.update_human_velocity_history(human_velocity_z)
