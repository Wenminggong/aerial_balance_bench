#!/usr/bin/env python3
"""Run the acceleration-interface CPID baseline and log rollout data."""

from __future__ import annotations

import argparse
import math
import random
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from omni.isaac.lab.app import AppLauncher


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_PARENT = PROJECT_ROOT.parent
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))


DEFAULT_RUN_CONFIG = PROJECT_ROOT / "baselines" / "configs" / "cpid_target_position_acceleration_eval_delay_free.yaml"
DEFAULT_ENV_CONFIG = PROJECT_ROOT / "environments" / "configs" / "template_eval_acceleration_delay_free.yaml"
DEFAULT_POLICY_CONFIG = PROJECT_ROOT / "baselines" / "configs" / "cpid_acceleration.yaml"


def _parse_args():
    parser = argparse.ArgumentParser(description="Acceleration CPID baseline runner for Aerial-Balance-Bench.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_RUN_CONFIG),
        help="Path to an acceleration CPID run YAML config.",
    )
    parser.add_argument("--env_config", type=str, default=None, help="Override environment YAML config path.")
    parser.add_argument("--policy_config", type=str, default=None, help="Override acceleration CPID policy YAML config path.")
    parser.add_argument("--episodes", type=int, default=None, help="Override target completed episodes.")
    parser.add_argument("--num_envs", type=int, default=None, help="Override number of parallel environments.")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed. Use -1 for a random seed.")
    parser.add_argument("--run_name", type=str, default=None, help="Override log run name.")
    parser.add_argument("--log_root", type=str, default=None, help="Override log root directory.")
    parser.add_argument("--video", action="store_true", default=False, help="Record rollout video.")
    parser.add_argument("--no_save_rollout", action="store_true", default=False, help="Skip rollout NPZ logging.")
    parser.add_argument(
        "--action_delay_enabled",
        action="store_true",
        default=None,
        help="Enable command-level action delay. Passing --delay_step > 0 also enables it.",
    )
    parser.add_argument("--delay_step", type=int, default=None, help="Override command-level action delay steps.")
    parser.add_argument(
        "--external_disturbance_enabled",
        action="store_true",
        default=None,
        help="Enable OU-process external disturbance.",
    )
    parser.add_argument(
        "--external_disturbance_ou_clip",
        type=float,
        default=None,
        help="Override the OU external-disturbance velocity clip.",
    )
    parser.add_argument("--angle_kp", type=float, default=None, help="Override outer angle PID Kp.")
    parser.add_argument("--angle_ti", type=float, default=None, help="Override outer angle PID Ti.")
    parser.add_argument("--angle_td", type=float, default=None, help="Override outer angle PID Td.")
    parser.add_argument("--max_theta_change", type=float, default=None, help="Override per-step theta reference limit.")
    parser.add_argument("--max_theta_ref", type=float, default=None, help="Override absolute theta reference limit.")
    parser.add_argument("--acc_kp", type=float, default=None, help="Override inner acceleration PID Kp.")
    parser.add_argument("--acc_ti", type=float, default=None, help="Override inner acceleration PID Ti.")
    parser.add_argument("--acc_td", type=float, default=None, help="Override inner acceleration PID Td.")
    parser.add_argument(
        "--max_delta_acc",
        type=float,
        default=None,
        help="Override both policy and acceleration-interface acceleration increment limit.",
    )
    parser.add_argument("--max_acc", type=float, default=None, help="Override acceleration command limit.")
    parser.add_argument("--action_sign", type=float, default=None, help="Override acceleration action sign.")
    AppLauncher.add_app_launcher_args(parser)
    args_cli = parser.parse_args()
    if args_cli.video:
        args_cli.enable_cameras = True
    return args_cli


def _cli_option_present(option: str) -> bool:
    return any(arg == option or arg.startswith(f"{option}=") for arg in sys.argv[1:])


def _read_yaml_safely(path: str | Path) -> dict[str, Any]:
    try:
        with open(Path(path).expanduser().resolve(), encoding="utf-8") as stream:
            return yaml.safe_load(stream) or {}
    except FileNotFoundError:
        return {}


def _resolve_config_path(path: str | Path | None, base_dir: Path, default_path: Path) -> Path:
    if path is None:
        return default_path.resolve()

    raw_path = Path(path).expanduser()
    if raw_path.is_absolute():
        return raw_path.resolve()

    for root in (PROJECT_ROOT, base_dir):
        candidate = (root / raw_path).resolve()
        if candidate.exists():
            return candidate
    return (PROJECT_ROOT / raw_path).resolve()


args_cli = _parse_args()
_run_config_path = Path(args_cli.config).expanduser().resolve()
_launch_run_config = _read_yaml_safely(_run_config_path)
_launch_env_config_path = _resolve_config_path(
    args_cli.env_config or _launch_run_config.get("env_config"),
    _run_config_path.parent,
    DEFAULT_ENV_CONFIG,
)
_launch_env_config = _read_yaml_safely(_launch_env_config_path)
_yaml_device = _launch_env_config.get("env", {}).get("device")
if _yaml_device is not None and not _cli_option_present("--device"):
    args_cli.device = _yaml_device
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import gymnasium as gym
import numpy as np
import torch

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - tqdm is optional for this smoke runner.
    tqdm = None

from aerial_balance_bench.baselines import AccelerationCPIDPolicy, AccelerationCPIDPolicyCfg
from aerial_balance_bench.environments.aerial_balance_env import AerialBalanceEnv, AerialBalanceEnvCfg
from aerial_balance_bench.utils.io import append_csv_row, ensure_dir, save_yaml


OBSERVATION_FIELDS = [
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
]

STEP_EXTRA_FIELDS = (
    "pb",
    "pg",
    "vg",
    "command_z",
    "executed_command_z",
    "arz_cmd",
    "executed_arz_cmd",
    "vrz_cmd",
    "executed_vrz_cmd",
    "drz_cmd",
    "executed_drz_cmd",
    "target_position_z",
    "frz_cmd",
    "executed_frz_cmd",
    "delta_frz_cmd",
    "hover_force",
    "target_pitch",
    "target_height_acc",
    "beta",
    "ball_mass",
    "low_level_controller_gain",
    "position_gain",
    "velocity_gain",
    "attitude_gain",
    "action_delay_enabled",
    "delay_step",
    "delayed_command_z",
    "external_disturbance_enabled",
    "external_disturbance_vel_z",
    "external_disturbance_ou_theta",
    "external_disturbance_ou_sigma",
    "last_action",
    "trajectory_type_id",
    "trajectory_amplitude",
    "trajectory_period",
)

POLICY_EXTRA_FIELDS = (
    "policy_raw_error",
    "policy_error",
    "policy_error_dot",
    "policy_error_ddot",
    "policy_theta_ref",
    "policy_error_theta",
    "policy_error_theta_dot",
    "policy_error_theta_ddot",
    "policy_delta_theta",
    "policy_raw_delta_arz",
    "policy_delta_arz",
    "policy_arz_cmd",
    "policy_acceleration_saturated",
)


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as stream:
        return yaml.safe_load(stream) or {}


def _maybe_set_attrs(target: Any, values: dict[str, Any]):
    for key, value in values.items():
        if value is not None:
            setattr(target, key, value)


def _resolve_seed(run_config: dict[str, Any], env_config: dict[str, Any]) -> int:
    if args_cli.seed is not None:
        seed = args_cli.seed
    else:
        seed = int(run_config.get("seed", env_config.get("env", {}).get("seed", 666)))
    if seed == -1:
        seed = random.randint(0, 10000)
    return seed


def _build_env_cfg(config: dict[str, Any], seed: int) -> AerialBalanceEnvCfg:
    env_cfg = AerialBalanceEnvCfg()

    env_values = config.get("env", {})
    env_cfg.seed = seed
    env_cfg.task_name = config.get("task_name", env_values.get("task_name", env_cfg.task_name))
    env_cfg.interface_name = config.get("interface_name", env_values.get("interface_name", env_cfg.interface_name))
    env_cfg.episode_length_s = float(env_values.get("episode_length_s", env_cfg.episode_length_s))
    env_cfg.scene.num_envs = int(
        args_cli.num_envs if args_cli.num_envs is not None else env_values.get("num_envs", env_cfg.scene.num_envs)
    )

    if _cli_option_present("--device"):
        env_cfg.sim.device = args_cli.device
    else:
        env_cfg.sim.device = env_values.get("device", env_cfg.sim.device)

    sim_values = config.get("sim", {})
    if "dt" in sim_values:
        env_cfg.sim.dt = float(sim_values["dt"])
    if "decimation" in sim_values:
        env_cfg.decimation = int(sim_values["decimation"])
        env_cfg.sim.render_interval = env_cfg.decimation
    if "rope_length" in env_values:
        env_cfg.rope_length = float(env_values["rope_length"])

    target_position_task_cfg = config.get("target_position_task", config.get("task", {}))
    _maybe_set_attrs(env_cfg.target_position_task, target_position_task_cfg)
    _maybe_set_attrs(env_cfg.trajectory_tracking_task, config.get("trajectory_tracking_task", {}))
    _maybe_set_attrs(env_cfg.acceleration_interface, config.get("acceleration_interface", {}))
    _maybe_set_attrs(env_cfg.velocity_interface, config.get("velocity_interface", {}))
    _maybe_set_attrs(env_cfg.position_interface, config.get("position_interface", {}))
    _maybe_set_attrs(env_cfg.thrust_interface, config.get("thrust_interface", {}))
    _maybe_set_attrs(env_cfg.robustness, config.get("robustness", {}))

    target_position_evaluator_cfg = config.get("target_position_evaluator", config.get("evaluator", {}))
    _maybe_set_attrs(env_cfg.target_position_evaluator, target_position_evaluator_cfg)
    _maybe_set_attrs(env_cfg.trajectory_tracking_evaluator, config.get("trajectory_tracking_evaluator", {}))
    env_cfg.target_position_evaluator.episode_length_s = env_cfg.episode_length_s
    env_cfg.trajectory_tracking_evaluator.episode_length_s = env_cfg.episode_length_s

    return env_cfg


def _validate_env_cfg(env_cfg: AerialBalanceEnvCfg):
    if env_cfg.task_name != "target_position":
        raise ValueError("AccelerationCPIDPolicy runner supports only task_name='target_position'.")
    if env_cfg.interface_name != "acceleration":
        raise ValueError("AccelerationCPIDPolicy runner supports only interface_name='acceleration'.")


def _apply_cli_overrides(env_cfg: AerialBalanceEnvCfg, policy_cfg: AccelerationCPIDPolicyCfg):
    robustness_override_requested = False

    if args_cli.delay_step is not None:
        env_cfg.robustness.delay_step = int(args_cli.delay_step)
        env_cfg.robustness.action_delay_enabled = int(args_cli.delay_step) > 0
        robustness_override_requested = True
    if args_cli.action_delay_enabled is not None:
        env_cfg.robustness.action_delay_enabled = bool(args_cli.action_delay_enabled)
        robustness_override_requested = True
    if args_cli.external_disturbance_enabled is not None:
        env_cfg.robustness.external_disturbance_enabled = bool(args_cli.external_disturbance_enabled)
        robustness_override_requested = True
    if args_cli.external_disturbance_ou_clip is not None:
        env_cfg.robustness.external_disturbance_ou_clip = float(args_cli.external_disturbance_ou_clip)
        robustness_override_requested = True
    if robustness_override_requested:
        env_cfg.robustness.enabled = bool(
            env_cfg.robustness.ball_mass_variation_enabled
            or env_cfg.robustness.controller_gain_variation_enabled
            or (env_cfg.robustness.action_delay_enabled and int(env_cfg.robustness.delay_step) > 0)
            or env_cfg.robustness.external_disturbance_enabled
        )

    angle_cfg = policy_cfg.angle_pid
    acc_cfg = policy_cfg.acceleration_pid
    for cli_name, target, attr in (
        ("angle_kp", angle_cfg, "kp"),
        ("angle_ti", angle_cfg, "ti"),
        ("angle_td", angle_cfg, "td"),
        ("max_theta_change", angle_cfg, "max_theta_change"),
        ("max_theta_ref", angle_cfg, "max_theta_ref"),
        ("acc_kp", acc_cfg, "kp"),
        ("acc_ti", acc_cfg, "ti"),
        ("acc_td", acc_cfg, "td"),
        ("action_sign", acc_cfg, "action_sign"),
    ):
        value = getattr(args_cli, cli_name)
        if value is not None:
            setattr(target, attr, float(value))

    if args_cli.max_delta_acc is not None:
        env_cfg.acceleration_interface.max_delta_acc = float(args_cli.max_delta_acc)
        acc_cfg.max_delta_acc = float(args_cli.max_delta_acc)
    if args_cli.max_acc is not None:
        env_cfg.acceleration_interface.max_acc = float(args_cli.max_acc)
        acc_cfg.max_acc = float(args_cli.max_acc)


def _resolve_policy_limits_from_env(policy_cfg: AccelerationCPIDPolicyCfg, env_cfg: AerialBalanceEnvCfg):
    acc_cfg = policy_cfg.acceleration_pid
    if _is_auto(acc_cfg.max_delta_acc):
        acc_cfg.max_delta_acc = float(env_cfg.acceleration_interface.max_delta_acc)
    if _is_auto(acc_cfg.max_acc):
        acc_cfg.max_acc = float(env_cfg.acceleration_interface.max_acc)


def _is_auto(value) -> bool:
    return isinstance(value, str) and value.lower() == "auto"


def _make_output_dir(run_config: dict[str, Any], seed: int, target_episodes: int) -> Path:
    logging_cfg = run_config.get("logging", {})
    root_dir = ensure_dir(PROJECT_ROOT / (args_cli.log_root or logging_cfg.get("root_dir", "logs/cpid_acceleration")))

    if args_cli.run_name is not None:
        run_name = args_cli.run_name
    else:
        run_name = logging_cfg.get("run_name")
        if not run_name:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"cpid_acceleration_ep_{target_episodes}_seed_{seed}_{timestamp}"

    return ensure_dir(root_dir / run_name)


def _tensor_to_numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy()


def _scalar_metric(value: Any) -> float | bool:
    if isinstance(value, torch.Tensor):
        if value.dtype == torch.bool:
            return bool(value.detach().cpu().item())
        return float(value.detach().cpu().item())
    if isinstance(value, np.generic):
        return value.item()
    return value


def _collect_tensor_fields(source: dict[str, Any], field_names: tuple[str, ...]) -> dict[str, np.ndarray]:
    collected = {}
    for key in field_names:
        value = source.get(key)
        if isinstance(value, torch.Tensor):
            collected[key] = _tensor_to_numpy(value)
    return collected


def _collect_step_extras(infos: dict[str, Any]) -> dict[str, np.ndarray]:
    return _collect_tensor_fields(infos.get("step", {}), STEP_EXTRA_FIELDS)


def _collect_policy_extras(policy: AccelerationCPIDPolicy) -> dict[str, np.ndarray]:
    return _collect_tensor_fields(policy.get_state(), POLICY_EXTRA_FIELDS)


def _collect_benchmark_metrics(infos: dict[str, Any]) -> dict[str, float | bool]:
    benchmark = infos.get("benchmark", {})
    return {key: _scalar_metric(value) for key, value in benchmark.items()}


def _stack_or_empty(records: list[np.ndarray], shape: tuple[int, ...], dtype=np.float32) -> np.ndarray:
    if records:
        return np.stack(records, axis=0)
    return np.empty(shape, dtype=dtype)


def main():
    run_config_path = Path(args_cli.config).expanduser().resolve()
    run_config = _load_yaml(run_config_path)
    env_config_path = _resolve_config_path(
        args_cli.env_config or run_config.get("env_config"),
        run_config_path.parent,
        DEFAULT_ENV_CONFIG,
    )
    policy_config_path = _resolve_config_path(
        args_cli.policy_config or run_config.get("policy_config"),
        run_config_path.parent,
        DEFAULT_POLICY_CONFIG,
    )
    env_config = _load_yaml(env_config_path)
    policy_config = _load_yaml(policy_config_path)

    seed = _resolve_seed(run_config, env_config)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env_cfg = _build_env_cfg(env_config, seed)
    policy_cfg = AccelerationCPIDPolicyCfg.from_dict(policy_config)
    _apply_cli_overrides(env_cfg, policy_cfg)
    _validate_env_cfg(env_cfg)
    _resolve_policy_limits_from_env(policy_cfg, env_cfg)

    runner_cfg = run_config.get("runner", {})
    target_episodes = int(args_cli.episodes if args_cli.episodes is not None else runner_cfg.get("target_episodes", 1))
    target_episodes = max(target_episodes, 1)
    env_cfg.target_position_evaluator.num_eval_episodes = target_episodes
    env_cfg.trajectory_tracking_evaluator.num_eval_episodes = target_episodes

    output_dir = _make_output_dir(run_config, seed, target_episodes)
    save_yaml(run_config, output_dir / "input_run_config.yaml")
    save_yaml(env_config, output_dir / "input_env_config.yaml")
    save_yaml(policy_config, output_dir / "input_policy_config.yaml")

    env = None
    try:
        env = AerialBalanceEnv(cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
        if args_cli.video:
            video_kwargs = {
                "video_folder": str(output_dir / "videos"),
                "step_trigger": lambda step: step == 0,
                "video_length": env.max_episode_length - 1,
                "disable_logger": True,
            }
            env = gym.wrappers.RecordVideo(env, **video_kwargs)

        base_env = env.unwrapped
        policy = AccelerationCPIDPolicy(policy_cfg, base_env.num_envs, base_env.device, base_env.step_dt)
        num_envs = base_env.num_envs

        save_yaml(
            {
                "seed": seed,
                "num_envs": num_envs,
                "episode_length_s": env_cfg.episode_length_s,
                "task_name": env_cfg.task_name,
                "interface_name": env_cfg.interface_name,
                "target_episodes": target_episodes,
                "sim_device": env_cfg.sim.device,
                "run_config_path": str(run_config_path),
                "env_config_path": str(env_config_path),
                "policy_config_path": str(policy_config_path),
                "policy_name": policy_cfg.name,
                "policy_step_dt": base_env.step_dt,
                "angle_pid": vars(policy_cfg.angle_pid),
                "acceleration_pid": vars(policy_cfg.acceleration_pid),
                "robustness": {
                    "enabled": bool(env_cfg.robustness.enabled),
                    "action_delay_enabled": bool(env_cfg.robustness.action_delay_enabled),
                    "delay_step": int(env_cfg.robustness.delay_step),
                    "external_disturbance_enabled": bool(env_cfg.robustness.external_disturbance_enabled),
                    "external_disturbance_ou_clip": float(env_cfg.robustness.external_disturbance_ou_clip),
                },
                "observation_fields": OBSERVATION_FIELDS,
            },
            output_dir / "resolved_run.yaml",
        )

        observations, infos = env.reset()
        policy.reset()

        configured_max_steps = runner_cfg.get("max_steps")
        if configured_max_steps is None:
            max_steps = math.ceil(target_episodes / num_envs) * base_env.max_episode_length
        else:
            max_steps = int(configured_max_steps)

        save_rollout_enabled = bool(runner_cfg.get("save_rollout", True)) and not args_cli.no_save_rollout
        obs_records: list[np.ndarray] = []
        action_records: list[np.ndarray] = []
        reward_records: list[np.ndarray] = []
        terminated_records: list[np.ndarray] = []
        truncated_records: list[np.ndarray] = []
        policy_compute_time_records: list[float] = []
        benchmark_records: list[dict[str, float | bool]] = []
        step_extra_records: dict[str, list[np.ndarray]] = {key: [] for key in STEP_EXTRA_FIELDS}
        policy_extra_records: dict[str, list[np.ndarray]] = {key: [] for key in POLICY_EXTRA_FIELDS}
        reward_sum = 0.0
        reward_count = 0

        iterator = range(max_steps)
        if tqdm is not None:
            iterator = tqdm(iterator, desc="Acceleration CPID rollout")

        final_metrics = _collect_benchmark_metrics(infos)
        for _ in iterator:
            if save_rollout_enabled:
                obs_records.append(_tensor_to_numpy(observations["policy"]))

            start_time = time.perf_counter()
            # CPID is stateful. Use no_grad instead of inference_mode so its
            # internal buffers remain normal mutable tensors across resets.
            with torch.no_grad():
                actions = policy.act(observations, infos)
            policy_compute_time = time.perf_counter() - start_time
            if save_rollout_enabled:
                policy_extras = _collect_policy_extras(policy)

            next_observations, reward, terminated, truncated, infos = env.step(actions)

            reward_sum += float(reward.detach().sum().cpu().item())
            reward_count += int(reward.numel())
            policy_compute_time_records.append(policy_compute_time)

            if save_rollout_enabled:
                action_records.append(_tensor_to_numpy(actions))
                reward_records.append(_tensor_to_numpy(reward))
                terminated_records.append(_tensor_to_numpy(terminated))
                truncated_records.append(_tensor_to_numpy(truncated))

                step_extras = _collect_step_extras(infos)
                for key, value in step_extras.items():
                    step_extra_records[key].append(value)
                for key, value in policy_extras.items():
                    policy_extra_records[key].append(value)

            final_metrics = _collect_benchmark_metrics(infos)
            if save_rollout_enabled:
                benchmark_records.append(final_metrics)

            done_env_ids = (terminated | truncated).nonzero(as_tuple=False).squeeze(-1)
            if done_env_ids.numel() > 0:
                policy.reset(done_env_ids)

            observations = next_observations

            if bool(runner_cfg.get("render", True)) and not args_cli.headless:
                env.render()

            completed_episodes = int(final_metrics.get("completed_episodes", 0))
            if completed_episodes >= target_episodes:
                break

        if save_rollout_enabled:
            rollout_steps = len(obs_records)
            observations_np = _stack_or_empty(obs_records, (0, num_envs, len(OBSERVATION_FIELDS)))
            rewards_np = _stack_or_empty(reward_records, (0, num_envs))
        else:
            rollout_steps = int(final_metrics.get("rollout_steps", 0)) or len(policy_compute_time_records)
            observations_np = np.empty((0, num_envs, len(OBSERVATION_FIELDS)), dtype=np.float32)
            rewards_np = np.empty((0, num_envs), dtype=np.float32)

        if save_rollout_enabled:
            rollout_payload = {
                "observations": observations_np,
                "actions": _stack_or_empty(action_records, (0, num_envs, 1)),
                "rewards": rewards_np,
                "terminated": _stack_or_empty(terminated_records, (0, num_envs), dtype=bool),
                "truncated": _stack_or_empty(truncated_records, (0, num_envs), dtype=bool),
                "policy_compute_time": np.asarray(policy_compute_time_records, dtype=np.float64),
                "observation_fields": np.asarray(OBSERVATION_FIELDS),
            }
            for key, records in step_extra_records.items():
                if records:
                    rollout_payload[f"step_{key}"] = _stack_or_empty(records, (0, num_envs))
                else:
                    rollout_payload[f"step_{key}"] = np.full((rollout_steps, num_envs), np.nan, dtype=np.float32)
            for key, records in policy_extra_records.items():
                if records:
                    rollout_payload[key] = _stack_or_empty(records, (0, num_envs))
                else:
                    rollout_payload[key] = np.full((rollout_steps, num_envs), np.nan, dtype=np.float32)
            if benchmark_records:
                for key in benchmark_records[-1].keys():
                    rollout_payload[f"benchmark_{key}"] = np.asarray(
                        [record.get(key, np.nan) for record in benchmark_records]
                    )
            np.savez_compressed(output_dir / "rollout.npz", **rollout_payload)

        completed_episodes = int(final_metrics.get("completed_episodes", 0))
        mean_reward = reward_sum / reward_count if reward_count > 0 else float("nan")
        summary_row = {
            "run_name": output_dir.name,
            "run_config_path": str(run_config_path),
            "env_config_path": str(env_config_path),
            "policy_config_path": str(policy_config_path),
            "seed": seed,
            "num_envs": num_envs,
            "target_episodes": target_episodes,
            "completed_episodes": completed_episodes,
            "rollout_steps": rollout_steps,
            "episode_length_s": env_cfg.episode_length_s,
            "task_name": env_cfg.task_name,
            "interface_name": env_cfg.interface_name,
            "policy_name": policy_cfg.name,
            "angle_kp": policy_cfg.angle_pid.kp,
            "angle_ti": policy_cfg.angle_pid.ti,
            "angle_td": policy_cfg.angle_pid.td,
            "max_theta_change": policy_cfg.angle_pid.max_theta_change,
            "max_theta_ref": policy_cfg.angle_pid.max_theta_ref,
            "acc_kp": policy_cfg.acceleration_pid.kp,
            "acc_ti": policy_cfg.acceleration_pid.ti,
            "acc_td": policy_cfg.acceleration_pid.td,
            "max_delta_acc": policy_cfg.acceleration_pid.max_delta_acc,
            "max_acc": policy_cfg.acceleration_pid.max_acc,
            "action_sign": policy_cfg.acceleration_pid.action_sign,
            "robustness_enabled": bool(env_cfg.robustness.enabled),
            "action_delay_enabled": bool(env_cfg.robustness.action_delay_enabled),
            "delay_step": int(env_cfg.robustness.delay_step),
            "external_disturbance_enabled": bool(env_cfg.robustness.external_disturbance_enabled),
            "external_disturbance_ou_clip": float(env_cfg.robustness.external_disturbance_ou_clip),
            "mean_reward": mean_reward,
            "policy_compute_time_mean": float(np.mean(policy_compute_time_records))
            if policy_compute_time_records
            else float("nan"),
        }
        for key, value in final_metrics.items():
            summary_row[f"benchmark_{key}"] = value
        append_csv_row(output_dir / "summary.csv", summary_row)
        append_csv_row(
            PROJECT_ROOT
            / (args_cli.log_root or run_config.get("logging", {}).get("root_dir", "logs/cpid_acceleration"))
            / "summary.csv",
            summary_row,
        )

        print(f"[INFO] Acceleration CPID rollout finished. Logs saved to: {output_dir}")
        print(f"[INFO] Summary: {summary_row}")
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
