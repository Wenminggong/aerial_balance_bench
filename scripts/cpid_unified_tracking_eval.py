#!/usr/bin/env python3
"""Evaluate the unchanged CPID baseline on unified reference tracking."""

from __future__ import annotations

import argparse
from dataclasses import asdict
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


DEFAULT_RUN_CONFIG = PROJECT_ROOT / "baselines" / "configs" / "cpid_unified_tracking_eval.yaml"
DEFAULT_ENV_CONFIG = PROJECT_ROOT / "environments" / "configs" / "unified_tracking_mixed.yaml"
DEFAULT_POLICY_CONFIG = PROJECT_ROOT / "baselines" / "configs" / "cpid.yaml"


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate the unchanged CPID baseline on Aerial-Balance-Bench unified tracking."
    )
    parser.add_argument("--config", type=str, default=str(DEFAULT_RUN_CONFIG), help="Path to a CPID run YAML.")
    parser.add_argument("--env_config", type=str, default=None, help="Override environment YAML config path.")
    parser.add_argument("--policy_config", type=str, default=None, help="Override CPID policy YAML config path.")
    parser.add_argument("--episodes", type=int, default=None, help="Override target completed episodes.")
    parser.add_argument("--num_envs", type=int, default=None, help="Override number of parallel environments.")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed. Use -1 for a random seed.")
    parser.add_argument("--run_name", type=str, default=None, help="Override log run name.")
    parser.add_argument("--video", action="store_true", default=False, help="Record rollout video.")
    AppLauncher.add_app_launcher_args(parser)
    parsed = parser.parse_args()
    if parsed.video:
        parsed.enable_cameras = True
    return parsed


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
except Exception:  # pragma: no cover
    tqdm = None

from aerial_balance_bench.baselines import (
    CPIDPolicy,
    CPIDPolicyCfg,
    validate_reference_preview_horizon,
)
from aerial_balance_bench.environments.aerial_balance_env import AerialBalanceEnv, AerialBalanceEnvCfg
from aerial_balance_bench.utils.io import append_csv_row, ensure_dir, save_yaml


LEGACY_OBSERVATION_FIELDS = (
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

STEP_EXTRA_FIELDS = (
    "pb",
    "pg",
    "vg",
    "command_z",
    "executed_command_z",
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
    "velocity_response_enabled",
    "velocity_response_tau_s",
    "velocity_response_gain",
    "velocity_response_bias",
    "velocity_response_sim_tau_s",
    "velocity_response_sim_gain",
    "velocity_response_sim_bias",
    "velocity_response_noise_mode_id",
    "velocity_response_noise_std",
    "velocity_response_ou_theta",
    "velocity_response_ou_mu",
    "velocity_response_input_z",
    "velocity_response_target_z",
    "velocity_response_nominal_z",
    "velocity_response_noise_z",
    "velocity_response_executed_z",
    "velocity_response_compensated_command_z",
    "velocity_response_error_z",
    "external_disturbance_enabled",
    "external_disturbance_vel_z",
    "external_disturbance_ou_theta",
    "external_disturbance_ou_sigma",
    "last_action",
    "trajectory_type_id",
    "trajectory_center",
    "trajectory_amplitude",
    "trajectory_period",
    "trajectory_phase",
    "initial_ball_position",
)


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as stream:
        return yaml.safe_load(stream) or {}


def _set_known_attrs(target: Any, values: dict[str, Any], section: str):
    unknown = sorted(key for key in values if not hasattr(target, key))
    if unknown:
        raise ValueError(f"Unknown field(s) in {section}: {', '.join(unknown)}")
    for key, value in values.items():
        if value is not None:
            setattr(target, key, value)


def _validate_known_keys(values: dict[str, Any], allowed: set[str], section: str):
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"Unknown field(s) in {section}: {', '.join(unknown)}")


def _resolve_seed(run_config: dict[str, Any], env_config: dict[str, Any]) -> int:
    if args_cli.seed is not None:
        seed = args_cli.seed
    else:
        seed = int(run_config.get("seed", env_config.get("env", {}).get("seed", 666)))
    return random.randint(0, 10000) if seed == -1 else seed


def _build_env_cfg(config: dict[str, Any], seed: int) -> AerialBalanceEnvCfg:
    _validate_known_keys(
        config,
        {
            "task_name",
            "interface_name",
            "env",
            "sim",
            "unified_tracking_task",
            "reference_preview",
            "velocity_interface",
            "robustness",
            "unified_tracking_evaluator",
            "runner",
            "logging",
        },
        "environment config",
    )
    env_cfg = AerialBalanceEnvCfg()
    env_values = config.get("env", {})
    _validate_known_keys(
        env_values,
        {
            "seed",
            "num_envs",
            "episode_length_s",
            "device",
            "task_name",
            "interface_name",
            "plank_length",
            "plank_slide_length",
            "beam_block_offset",
            "r_holder",
            "rope_length",
            "beam_position_min",
            "beam_position_max",
            "max_theta",
        },
        "env",
    )
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
    _validate_known_keys(sim_values, {"dt", "decimation"}, "sim")
    if "dt" in sim_values:
        env_cfg.sim.dt = float(sim_values["dt"])
    if "decimation" in sim_values:
        env_cfg.decimation = int(sim_values["decimation"])
        env_cfg.sim.render_interval = env_cfg.decimation
    for name in (
        "plank_length",
        "plank_slide_length",
        "beam_block_offset",
        "r_holder",
        "rope_length",
        "beam_position_min",
        "beam_position_max",
        "max_theta",
    ):
        if name in env_values:
            setattr(env_cfg, name, env_values[name])

    _set_known_attrs(
        env_cfg.unified_tracking_task,
        config.get("unified_tracking_task", {}),
        "unified_tracking_task",
    )
    _set_known_attrs(env_cfg.reference_preview, config.get("reference_preview", {}), "reference_preview")
    _set_known_attrs(env_cfg.velocity_interface, config.get("velocity_interface", {}), "velocity_interface")
    _set_known_attrs(env_cfg.robustness, config.get("robustness", {}), "robustness")
    _set_known_attrs(
        env_cfg.unified_tracking_evaluator,
        config.get("unified_tracking_evaluator", {}),
        "unified_tracking_evaluator",
    )
    env_cfg.unified_tracking_evaluator.episode_length_s = env_cfg.episode_length_s
    return env_cfg


def _validate_env_cfg(env_cfg: AerialBalanceEnvCfg):
    if env_cfg.task_name != "unified_tracking":
        raise ValueError("CPID unified-tracking runner requires task_name='unified_tracking'.")
    if env_cfg.interface_name != "velocity":
        raise ValueError("CPID unified-tracking runner requires interface_name='velocity'.")


def _resolve_predictor_cfg_from_env(policy_cfg: CPIDPolicyCfg, env_cfg: AerialBalanceEnvCfg, step_dt: float):
    predictor_cfg = policy_cfg.state_predictor
    predictor_cfg.resolve_delay_step_from_robustness(env_cfg.robustness)
    if _is_auto(predictor_cfg.step_dt) or float(predictor_cfg.step_dt) <= 0.0:
        predictor_cfg.step_dt = float(step_dt)
    if _is_auto(predictor_cfg.max_acc):
        predictor_cfg.max_acc = float(env_cfg.velocity_interface.max_acc)
    if _is_auto(predictor_cfg.max_velocity):
        predictor_cfg.max_velocity = float(env_cfg.velocity_interface.max_velocity)
    if _is_auto(predictor_cfg.plank_length):
        predictor_cfg.plank_length = float(env_cfg.plank_length)
    if _is_auto(predictor_cfg.rope_length):
        predictor_cfg.rope_length = float(env_cfg.rope_length)
    if _is_auto(predictor_cfg.gravity):
        predictor_cfg.gravity = float(abs(env_cfg.sim.gravity[-1]))
    if _is_auto(predictor_cfg.ball_radius):
        predictor_cfg.ball_radius = float(env_cfg.ball_cfg.spawn.radius)
    if _is_auto(predictor_cfg.ball_mass):
        predictor_cfg.ball_mass = float(env_cfg.ball_cfg.spawn.mass_props.mass)
    predictor_cfg.resolve_velocity_response_from_robustness(env_cfg.robustness)


def _validate_predictor_reference_preview(policy_cfg: CPIDPolicyCfg, env_cfg: AerialBalanceEnvCfg):
    predictor_cfg = policy_cfg.state_predictor
    if not predictor_cfg.enabled:
        return
    validate_reference_preview_horizon(
        delay_step=int(predictor_cfg.delay_step),
        preview_enabled=bool(env_cfg.reference_preview.enabled),
        preview_future_steps=int(env_cfg.reference_preview.future_steps),
        context="CPID unified tracking",
    )


def _is_auto(value) -> bool:
    return isinstance(value, str) and value.lower() == "auto"


def _make_output_dir(run_config: dict[str, Any], seed: int, target_episodes: int) -> Path:
    logging_cfg = run_config.get("logging", {})
    root_dir = ensure_dir(PROJECT_ROOT / logging_cfg.get("root_dir", "logs/cpid/unified_tracking"))
    if args_cli.run_name is not None:
        run_name = args_cli.run_name
    else:
        run_name = logging_cfg.get("run_name")
        if not run_name:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"cpid_unified_ep_{target_episodes}_seed_{seed}_{timestamp}"
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


def _collect_tensor_fields(
    source: dict[str, Any],
    field_names: tuple[str, ...] | None = None,
) -> dict[str, np.ndarray]:
    items = source.items() if field_names is None else ((key, source.get(key)) for key in field_names)
    return {
        key: _tensor_to_numpy(value)
        for key, value in items
        if isinstance(value, torch.Tensor)
    }


def _collect_benchmark_metrics(infos: dict[str, Any]) -> dict[str, float | bool]:
    return {key: _scalar_metric(value) for key, value in infos.get("benchmark", {}).items()}


def _stack_or_empty(records: list[np.ndarray], shape: tuple[int, ...], dtype=np.float32) -> np.ndarray:
    return np.stack(records, axis=0) if records else np.empty(shape, dtype=dtype)


def main():
    run_config_path = Path(args_cli.config).expanduser().resolve()
    run_config = _load_yaml(run_config_path)
    _validate_known_keys(
        run_config,
        {"env_config", "policy_config", "seed", "runner", "logging"},
        "run config",
    )
    _validate_known_keys(
        run_config.get("runner", {}),
        {"target_episodes", "max_steps", "render", "save_rollout"},
        "runner",
    )
    _validate_known_keys(
        run_config.get("logging", {}),
        {"root_dir", "run_name"},
        "logging",
    )
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
    policy_cfg = CPIDPolicyCfg.from_dict(policy_config)

    seed = _resolve_seed(run_config, env_config)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env_cfg = _build_env_cfg(env_config, seed)
    _validate_env_cfg(env_cfg)
    runner_cfg = run_config.get("runner", {})
    target_episodes = max(
        int(args_cli.episodes if args_cli.episodes is not None else runner_cfg.get("target_episodes", 1)),
        1,
    )
    env_cfg.unified_tracking_evaluator.num_eval_episodes = target_episodes

    output_dir = _make_output_dir(run_config, seed, target_episodes)
    save_yaml(run_config, output_dir / "input_run_config.yaml")
    save_yaml(env_config, output_dir / "input_env_config.yaml")
    save_yaml(policy_config, output_dir / "input_policy_config.yaml")

    env = None
    try:
        env = AerialBalanceEnv(cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
        if args_cli.video:
            env = gym.wrappers.RecordVideo(
                env,
                video_folder=str(output_dir / "videos"),
                step_trigger=lambda step: step == 0,
                video_length=env.max_episode_length - 1,
                disable_logger=True,
            )

        base_env = env.unwrapped
        _resolve_predictor_cfg_from_env(policy_cfg, env_cfg, base_env.step_dt)
        _validate_predictor_reference_preview(policy_cfg, env_cfg)
        observation_fields = tuple(base_env.observation_fields)
        if observation_fields[: len(LEGACY_OBSERVATION_FIELDS)] != LEGACY_OBSERVATION_FIELDS:
            raise ValueError(
                "Unified observation legacy prefix does not match the 11-D CPID observation contract."
            )
        policy = CPIDPolicy(
            policy_cfg,
            base_env.num_envs,
            base_env.device,
            base_env.step_dt,
            raw_observation_fields=observation_fields,
        )
        num_envs = base_env.num_envs
        predictor_active = policy.state_predictor.active
        reference_fields = (
            ["pg", *(f"pg_{offset}" for offset in range(1, policy.state_predictor.delay_step + 1))]
            if predictor_active
            else []
        )
        policy_consumed_fields = [
            *LEGACY_OBSERVATION_FIELDS,
            *(field for field in reference_fields if field != "pg"),
        ]
        task_metadata = base_env.task.get_config_info()
        save_yaml(
            {
                "seed": seed,
                "num_envs": num_envs,
                "episode_length_s": env_cfg.episode_length_s,
                "task_name": env_cfg.task_name,
                "interface_name": env_cfg.interface_name,
                "target_episodes": target_episodes,
                "sim_device": env_cfg.sim.device,
                "environment_velocity_max_acc": env_cfg.velocity_interface.max_acc,
                "raw_observation_dim": base_env.raw_observation_dim,
                "observation_fields": list(observation_fields),
                "policy_consumed_observation_dim": len(policy_consumed_fields),
                "policy_consumed_observation_fields": policy_consumed_fields,
                "reference_preview_consumed_by_policy": predictor_active,
                "reference_preview_enabled": env_cfg.reference_preview.enabled,
                "reference_preview_future_steps": env_cfg.reference_preview.future_steps,
                "reference_preview_offsets": list(base_env.reference_preview_offsets),
                "state_predictor_reference_mode": (
                    "future_preview" if predictor_active else "inactive"
                ),
                "state_predictor_reference_horizon": (
                    policy.state_predictor.delay_step if predictor_active else 0
                ),
                "state_predictor_reference_fields": reference_fields,
                "run_config_path": str(run_config_path),
                "env_config_path": str(env_config_path),
                "policy_config_path": str(policy_config_path),
                "resolved_policy_config": asdict(policy_cfg),
                **task_metadata,
            },
            output_dir / "resolved_run.yaml",
        )

        observations, infos = env.reset()
        policy.reset()
        configured_max_steps = runner_cfg.get("max_steps")
        max_steps = (
            math.ceil(target_episodes / num_envs) * base_env.max_episode_length
            if configured_max_steps is None
            else int(configured_max_steps)
        )
        if max_steps <= 0:
            raise ValueError("runner.max_steps must be positive when specified.")

        obs_records: list[np.ndarray] = []
        action_records: list[np.ndarray] = []
        reward_records: list[np.ndarray] = []
        terminated_records: list[np.ndarray] = []
        truncated_records: list[np.ndarray] = []
        compute_time_records: list[float] = []
        benchmark_records: list[dict[str, float | bool]] = []
        step_records: dict[str, list[np.ndarray]] = {key: [] for key in STEP_EXTRA_FIELDS}
        policy_records: dict[str, list[np.ndarray]] = {}

        iterator = tqdm(range(max_steps), desc="CPID unified rollout") if tqdm is not None else range(max_steps)
        final_metrics = _collect_benchmark_metrics(infos)
        for _ in iterator:
            obs_records.append(_tensor_to_numpy(observations["policy"]))
            start_time = time.perf_counter()
            # CPID owns mutable history buffers, so use no_grad rather than inference_mode.
            with torch.no_grad():
                actions = policy.act(observations, infos)
            compute_time_records.append(time.perf_counter() - start_time)
            policy_state = _collect_tensor_fields(policy.get_state())

            next_observations, reward, terminated, truncated, infos = env.step(actions)
            action_records.append(_tensor_to_numpy(actions))
            reward_records.append(_tensor_to_numpy(reward))
            terminated_records.append(_tensor_to_numpy(terminated))
            truncated_records.append(_tensor_to_numpy(truncated))
            for key, value in _collect_tensor_fields(infos.get("step", {}), STEP_EXTRA_FIELDS).items():
                step_records[key].append(value)
            for key, value in policy_state.items():
                policy_records.setdefault(key, []).append(value)

            final_metrics = _collect_benchmark_metrics(infos)
            benchmark_records.append(final_metrics)
            done_env_ids = (terminated | truncated).nonzero(as_tuple=False).squeeze(-1)
            if done_env_ids.numel() > 0:
                policy.reset(done_env_ids)
            observations = next_observations

            if bool(runner_cfg.get("render", True)) and not args_cli.headless:
                env.render()
            if int(final_metrics.get("completed_episodes", 0)) >= target_episodes:
                break

        rollout_steps = len(obs_records)
        observations_np = _stack_or_empty(obs_records, (0, num_envs, len(observation_fields)))
        rewards_np = _stack_or_empty(reward_records, (0, num_envs))
        if bool(runner_cfg.get("save_rollout", True)):
            rollout_payload = {
                "observations": observations_np,
                "actions": _stack_or_empty(action_records, (0, num_envs, 1)),
                "rewards": rewards_np,
                "terminated": _stack_or_empty(terminated_records, (0, num_envs), dtype=bool),
                "truncated": _stack_or_empty(truncated_records, (0, num_envs), dtype=bool),
                "policy_compute_time": np.asarray(compute_time_records, dtype=np.float64),
                "observation_fields": np.asarray(observation_fields),
                "policy_consumed_observation_fields": np.asarray(policy_consumed_fields),
            }
            for key, records in step_records.items():
                rollout_payload[f"step_{key}"] = (
                    _stack_or_empty(records, (0, num_envs))
                    if records
                    else np.full((rollout_steps, num_envs), np.nan, dtype=np.float32)
                )
            for key, records in policy_records.items():
                rollout_payload[key] = _stack_or_empty(records, (0, *records[0].shape))
            if benchmark_records:
                for key in benchmark_records[-1]:
                    rollout_payload[f"benchmark_{key}"] = np.asarray(
                        [record.get(key, np.nan) for record in benchmark_records]
                    )
            np.savez_compressed(output_dir / "rollout.npz", **rollout_payload)

        completed_episodes = int(final_metrics.get("completed_episodes", 0))
        evaluation_complete = bool(final_metrics.get("evaluation_complete", False))
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
            "mean_reward": float(np.mean(rewards_np)) if rewards_np.size else float("nan"),
            "policy_compute_time_mean": (
                float(np.mean(compute_time_records)) if compute_time_records else float("nan")
            ),
            "evaluation_complete": evaluation_complete,
        }
        for key, value in final_metrics.items():
            summary_row[f"benchmark_{key}"] = value
        append_csv_row(output_dir / "summary.csv", summary_row)
        append_csv_row(
            PROJECT_ROOT
            / run_config.get("logging", {}).get("root_dir", "logs/cpid/unified_tracking")
            / "summary.csv",
            summary_row,
        )
        if not evaluation_complete:
            print(
                "[WARN] Evaluation stopped before reaching the requested episode count: "
                f"completed={completed_episodes}, target={target_episodes}, max_steps={max_steps}."
            )
        print(f"[INFO] CPID unified rollout finished. Logs saved to: {output_dir}")
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
