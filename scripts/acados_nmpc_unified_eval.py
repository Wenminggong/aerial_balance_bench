#!/usr/bin/env python3
"""Evaluate the independent acados NMPC on unified reference tracking."""

from __future__ import annotations

import argparse
from copy import deepcopy
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

DEFAULT_RUN_CONFIG = (
    PROJECT_ROOT / "baselines" / "configs" / "acados_nmpc_unified_tracking_eval.yaml"
)
DEFAULT_ENV_CONFIG = PROJECT_ROOT / "environments" / "configs" / "unified_tracking_mixed.yaml"
DEFAULT_POLICY_CONFIG = PROJECT_ROOT / "baselines" / "configs" / "acados_nmpc_unified.yaml"


def _parse_args():
    parser = argparse.ArgumentParser(description="Evaluate acados NMPC on unified tracking.")
    parser.add_argument("--config", default=str(DEFAULT_RUN_CONFIG), help="Run YAML path.")
    parser.add_argument("--env_config", default=None, help="Override environment YAML.")
    parser.add_argument("--policy_config", default=None, help="Override policy YAML.")
    parser.add_argument("--episodes", type=int, default=None, help="Override completed episodes.")
    parser.add_argument("--num_envs", type=int, default=None, help="Override parallel environments.")
    parser.add_argument("--seed", type=int, default=None, help="Override seed; -1 samples one.")
    parser.add_argument("--run_name", default=None, help="Override output run name.")
    parser.add_argument("--video", action="store_true", help="Record one rollout video.")
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
    raw = Path(path).expanduser()
    if raw.is_absolute():
        return raw.resolve()
    for root in (PROJECT_ROOT, base_dir):
        candidate = (root / raw).resolve()
        if candidate.exists():
            return candidate
    return (PROJECT_ROOT / raw).resolve()


args_cli = _parse_args()
_launch_run_path = Path(args_cli.config).expanduser().resolve()
_launch_run = _read_yaml_safely(_launch_run_path)
_launch_env_path = _resolve_config_path(
    args_cli.env_config or _launch_run.get("env_config"),
    _launch_run_path.parent,
    DEFAULT_ENV_CONFIG,
)
_launch_env = _read_yaml_safely(_launch_env_path)
_yaml_device = _launch_env.get("env", {}).get("device")
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

from aerial_balance_bench.baselines import AcadosNMPCPolicy, AcadosNMPCPolicyCfg
from aerial_balance_bench.environments.aerial_balance_env import AerialBalanceEnv, AerialBalanceEnvCfg
from aerial_balance_bench.environments.observation_schema import LEGACY_OBSERVATION_FIELDS
from aerial_balance_bench.utils.io import append_csv_row, ensure_dir, save_yaml


STEP_EXTRA_FIELDS = (
    "pb",
    "pg",
    "vg",
    "command_z",
    "executed_command_z",
    "vrz_cmd",
    "executed_vrz_cmd",
    "action_delay_enabled",
    "delay_step",
    "delayed_command_z",
    "velocity_response_enabled",
    "velocity_response_tau_s",
    "velocity_response_gain",
    "velocity_response_bias",
    "velocity_response_input_z",
    "velocity_response_target_z",
    "velocity_response_nominal_z",
    "velocity_response_executed_z",
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


def _validate_known_keys(values: dict[str, Any], allowed: set[str], section: str) -> None:
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"Unknown field(s) in {section}: {', '.join(unknown)}")


def _set_known_attrs(target: Any, values: dict[str, Any], section: str) -> None:
    unknown = sorted(key for key in values if not hasattr(target, key))
    if unknown:
        raise ValueError(f"Unknown field(s) in {section}: {', '.join(unknown)}")
    for key, value in values.items():
        if value is not None:
            setattr(target, key, value)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


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
    env_cfg.interface_name = config.get(
        "interface_name", env_values.get("interface_name", env_cfg.interface_name)
    )
    env_cfg.episode_length_s = float(env_values.get("episode_length_s", env_cfg.episode_length_s))
    env_cfg.scene.num_envs = int(
        args_cli.num_envs
        if args_cli.num_envs is not None
        else env_values.get("num_envs", env_cfg.scene.num_envs)
    )
    env_cfg.sim.device = (
        args_cli.device
        if _cli_option_present("--device")
        else env_values.get("device", env_cfg.sim.device)
    )
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
    _set_known_attrs(
        env_cfg.reference_preview, config.get("reference_preview", {}), "reference_preview"
    )
    _set_known_attrs(
        env_cfg.velocity_interface, config.get("velocity_interface", {}), "velocity_interface"
    )
    _set_known_attrs(env_cfg.robustness, config.get("robustness", {}), "robustness")
    _set_known_attrs(
        env_cfg.unified_tracking_evaluator,
        config.get("unified_tracking_evaluator", {}),
        "unified_tracking_evaluator",
    )
    env_cfg.unified_tracking_evaluator.episode_length_s = env_cfg.episode_length_s
    return env_cfg


def _is_auto(value: object) -> bool:
    return isinstance(value, str) and value.lower() == "auto"


def _resolve_policy_cfg_from_env(
    cfg: AcadosNMPCPolicyCfg, env_cfg: AerialBalanceEnvCfg, step_dt: float
) -> None:
    if env_cfg.task_name != "unified_tracking":
        raise ValueError("acados NMPC unified runner requires task_name='unified_tracking'.")
    if env_cfg.interface_name != "velocity":
        raise ValueError("acados NMPC unified runner requires interface_name='velocity'.")
    model_values = {
        "plank_length": float(env_cfg.plank_length),
        "rope_length": float(env_cfg.rope_length),
        "ball_position_offset": float(env_cfg.plank_slide_length + env_cfg.beam_block_offset),
        "gravity": float(abs(env_cfg.sim.gravity[-1])),
        "ball_mass": float(env_cfg.ball_cfg.spawn.mass_props.mass),
        "ball_radius": float(env_cfg.ball_cfg.spawn.radius),
    }
    for name, value in model_values.items():
        if _is_auto(getattr(cfg.model, name)):
            setattr(cfg.model, name, value)
    if _is_auto(cfg.solver.step_dt):
        cfg.solver.step_dt = float(step_dt)
    if _is_auto(cfg.constraints.max_acc):
        cfg.constraints.max_acc = float(env_cfg.velocity_interface.max_acc)
    if _is_auto(cfg.constraints.max_velocity):
        cfg.constraints.max_velocity = float(env_cfg.velocity_interface.max_velocity)

    response = cfg.response
    if response.enabled:
        response_fields = (
            ("tau_s", "velocity_response_tau_s_range", "velocity_response_sim_tau_s"),
            ("gain", "velocity_response_gain_range", "velocity_response_sim_gain"),
            ("bias", "velocity_response_bias_range", "velocity_response_sim_bias"),
        )
        response_active = bool(
            env_cfg.robustness.enabled and env_cfg.robustness.velocity_response_enabled
        )
        for policy_name, range_name, sim_name in response_fields:
            if not _is_auto(getattr(response, policy_name)):
                continue
            if not response_active:
                setattr(response, policy_name, float(getattr(env_cfg.robustness, sim_name)))
                continue
            lower, upper = (float(value) for value in getattr(env_cfg.robustness, range_name))
            if lower != upper:
                raise ValueError(
                    f"response.{policy_name}='auto' cannot resolve randomized robustness.{range_name} "
                    f"[{lower}, {upper}]; configure an explicit nominal NMPC value."
                )
            setattr(response, policy_name, lower)

    predictor = cfg.state_predictor
    predictor.resolve_delay_step_from_robustness(env_cfg.robustness)
    if _is_auto(predictor.step_dt) or float(predictor.step_dt) <= 0.0:
        predictor.step_dt = float(step_dt)
    if _is_auto(predictor.max_acc):
        predictor.max_acc = float(env_cfg.velocity_interface.max_acc)
    if _is_auto(predictor.max_velocity):
        predictor.max_velocity = float(env_cfg.velocity_interface.max_velocity)
    for name, value in model_values.items():
        if _is_auto(getattr(predictor, name)):
            setattr(predictor, name, value)
    if _is_auto(predictor.ball_inertia_ratio):
        predictor.ball_inertia_ratio = float(cfg.model.ball_inertia_ratio)
    if _is_auto(predictor.epsilon):
        predictor.epsilon = float(cfg.model.epsilon)
    if predictor.enabled:
        predictor.resolve_velocity_response_from_robustness(env_cfg.robustness)
    else:
        for predictor_name, response_name in (
            ("velocity_response_tau_s", "tau_s"),
            ("velocity_response_gain", "gain"),
            ("velocity_response_bias", "bias"),
        ):
            if _is_auto(getattr(predictor, predictor_name)):
                setattr(predictor, predictor_name, getattr(response, response_name))
        if _is_auto(predictor.velocity_response_max_abs_velocity):
            predictor.velocity_response_max_abs_velocity = 0.0

    base_offset = int(predictor.delay_step) if predictor.enabled and int(predictor.delay_step) > 0 else 0
    required = base_offset + int(cfg.solver.n_horizon)
    if not env_cfg.reference_preview.enabled or int(env_cfg.reference_preview.future_steps) < required:
        raise ValueError(
            "acados NMPC requires reference_preview.enabled=true and "
            f"future_steps >= D+N={required}; got {env_cfg.reference_preview.future_steps}."
        )


def _resolve_seed(run_cfg: dict[str, Any], env_cfg: dict[str, Any]) -> int:
    raw = (
        args_cli.seed
        if args_cli.seed is not None
        else int(run_cfg.get("seed", env_cfg.get("env", {}).get("seed", 666)))
    )
    return random.randint(0, 10000) if raw == -1 else int(raw)


def _make_output_dir(run_cfg: dict[str, Any], seed: int, target_episodes: int) -> Path:
    logging_cfg = run_cfg.get("logging", {})
    root = ensure_dir(
        PROJECT_ROOT / logging_cfg.get("root_dir", "logs/acados_nmpc/unified_tracking")
    )
    run_name = args_cli.run_name or logging_cfg.get("run_name")
    if not run_name:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"acados_nmpc_ep_{target_episodes}_seed_{seed}_{timestamp}"
    return ensure_dir(root / run_name)


def _tensor_to_numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy()


def _collect_tensor_fields(
    source: dict[str, Any], names: tuple[str, ...] | None = None
) -> dict[str, np.ndarray]:
    items = source.items() if names is None else ((name, source.get(name)) for name in names)
    return {
        name: _tensor_to_numpy(value)
        for name, value in items
        if isinstance(value, torch.Tensor)
    }


def _scalar(value: Any) -> float | bool:
    if isinstance(value, torch.Tensor):
        return bool(value.item()) if value.dtype == torch.bool else float(value.item())
    if isinstance(value, np.generic):
        return value.item()
    return value


def _benchmark_metrics(infos: dict[str, Any]) -> dict[str, float | bool]:
    return {name: _scalar(value) for name, value in infos.get("benchmark", {}).items()}


def _stack(records: list[np.ndarray], empty_shape: tuple[int, ...], dtype=np.float32) -> np.ndarray:
    return np.stack(records, axis=0) if records else np.empty(empty_shape, dtype=dtype)


def _timing_summary(values: np.ndarray, deadline_s: float) -> dict[str, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return {
            "controller_time_mean": float("nan"),
            "controller_time_p50": float("nan"),
            "controller_time_p95": float("nan"),
            "controller_time_p99": float("nan"),
            "controller_time_max": float("nan"),
            "deadline_miss_rate": float("nan"),
        }
    return {
        "controller_time_mean": float(np.mean(finite)),
        "controller_time_p50": float(np.percentile(finite, 50)),
        "controller_time_p95": float(np.percentile(finite, 95)),
        "controller_time_p99": float(np.percentile(finite, 99)),
        "controller_time_max": float(np.max(finite)),
        "deadline_miss_rate": float(np.mean(finite > deadline_s)),
    }


def main():
    run_path = Path(args_cli.config).expanduser().resolve()
    run_cfg = _load_yaml(run_path)
    _validate_known_keys(
        run_cfg,
        {"env_config", "policy_config", "policy_overrides", "seed", "runner", "logging"},
        "run config",
    )
    _validate_known_keys(
        run_cfg.get("runner", {}),
        {
            "target_episodes",
            "max_steps",
            "render",
            "save_rollout",
            "stop_on_target_episodes",
        },
        "runner",
    )
    _validate_known_keys(run_cfg.get("logging", {}), {"root_dir", "run_name"}, "logging")
    env_path = _resolve_config_path(
        args_cli.env_config or run_cfg.get("env_config"), run_path.parent, DEFAULT_ENV_CONFIG
    )
    policy_path = _resolve_config_path(
        args_cli.policy_config or run_cfg.get("policy_config"), run_path.parent, DEFAULT_POLICY_CONFIG
    )
    env_data = _load_yaml(env_path)
    policy_data = _load_yaml(policy_path)
    policy_section_name = "acados_nmpc_policy" if "acados_nmpc_policy" in policy_data else "policy"
    if run_cfg.get("policy_overrides"):
        policy_data = deepcopy(policy_data)
        policy_data[policy_section_name] = _deep_merge(
            policy_data.get(policy_section_name, {}), run_cfg["policy_overrides"]
        )
    policy_cfg = AcadosNMPCPolicyCfg.from_dict(policy_data)

    seed = _resolve_seed(run_cfg, env_data)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    env_cfg = _build_env_cfg(env_data, seed)
    runner_cfg = run_cfg.get("runner", {})
    stop_on_target_episodes = bool(runner_cfg.get("stop_on_target_episodes", True))
    target_episodes = max(
        int(
            args_cli.episodes
            if args_cli.episodes is not None
            else runner_cfg.get("target_episodes", 1)
        ),
        1,
    )
    env_cfg.unified_tracking_evaluator.num_eval_episodes = target_episodes
    output_dir = _make_output_dir(run_cfg, seed, target_episodes)
    save_yaml(run_cfg, output_dir / "input_run_config.yaml")
    save_yaml(env_data, output_dir / "input_env_config.yaml")
    save_yaml(policy_data, output_dir / "input_policy_config.yaml")

    env = None
    policy = None
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
        _resolve_policy_cfg_from_env(policy_cfg, env_cfg, base_env.step_dt)
        fields = tuple(base_env.observation_fields)
        if fields[: len(LEGACY_OBSERVATION_FIELDS)] != LEGACY_OBSERVATION_FIELDS:
            raise ValueError("Unified raw observation changed its legacy 11-D prefix.")
        policy = AcadosNMPCPolicy(
            policy_cfg,
            base_env.num_envs,
            base_env.device,
            base_env.step_dt,
            raw_observation_fields=fields,
        )
        metadata = policy.get_resolved_metadata()
        save_yaml(
            {
                "seed": seed,
                "num_envs": base_env.num_envs,
                "episode_length_s": env_cfg.episode_length_s,
                "task_name": env_cfg.task_name,
                "interface_name": env_cfg.interface_name,
                "target_episodes": target_episodes,
                "stop_on_target_episodes": stop_on_target_episodes,
                "sim_device": env_cfg.sim.device,
                "control_step_dt": base_env.step_dt,
                "raw_observation_dim": base_env.raw_observation_dim,
                "observation_fields": list(fields),
                "reference_preview_enabled": env_cfg.reference_preview.enabled,
                "reference_preview_future_steps": env_cfg.reference_preview.future_steps,
                "reference_preview_offsets": list(base_env.reference_preview_offsets),
                "policy_consumed_observation_dim": len(policy.consumed_observation_fields),
                "run_config_path": str(run_path),
                "env_config_path": str(env_path),
                "policy_config_path": str(policy_path),
                **metadata,
                **base_env.task.get_config_info(),
            },
            output_dir / "resolved_run.yaml",
        )

        observations, infos = env.reset()
        policy.reset()
        num_envs = base_env.num_envs
        configured_steps = runner_cfg.get("max_steps")
        max_steps = (
            math.ceil(target_episodes / num_envs) * base_env.max_episode_length
            if configured_steps is None
            else int(configured_steps)
        )
        if max_steps <= 0:
            raise ValueError("runner.max_steps must be positive when specified.")

        obs_records: list[np.ndarray] = []
        action_records: list[np.ndarray] = []
        reward_records: list[np.ndarray] = []
        terminated_records: list[np.ndarray] = []
        truncated_records: list[np.ndarray] = []
        act_time_records: list[float] = []
        step_records: dict[str, list[np.ndarray]] = {name: [] for name in STEP_EXTRA_FIELDS}
        policy_records: dict[str, list[np.ndarray]] = {}
        final_metrics = _benchmark_metrics(infos)
        iterator = (
            tqdm(range(max_steps), desc="acados NMPC unified rollout")
            if tqdm is not None
            else range(max_steps)
        )
        for _ in iterator:
            obs_records.append(_tensor_to_numpy(observations["policy"]))
            started = time.perf_counter()
            with torch.no_grad():
                actions = policy.act(observations, infos)
            act_time_records.append(time.perf_counter() - started)
            state = _collect_tensor_fields(policy.get_state())
            next_observations, reward, terminated, truncated, infos = env.step(actions)
            action_records.append(_tensor_to_numpy(actions))
            reward_records.append(_tensor_to_numpy(reward))
            terminated_records.append(_tensor_to_numpy(terminated))
            truncated_records.append(_tensor_to_numpy(truncated))
            for name, value in _collect_tensor_fields(infos.get("step", {}), STEP_EXTRA_FIELDS).items():
                step_records[name].append(value)
            for name, value in state.items():
                policy_records.setdefault(name, []).append(value)
            final_metrics = _benchmark_metrics(infos)
            done_ids = (terminated | truncated).nonzero(as_tuple=False).squeeze(-1)
            if done_ids.numel() > 0:
                policy.reset(done_ids)
            observations = next_observations
            if bool(runner_cfg.get("render", False)) and not args_cli.headless:
                env.render()
            if (
                stop_on_target_episodes
                and int(final_metrics.get("completed_episodes", 0)) >= target_episodes
            ):
                break

        rollout_steps = len(obs_records)
        rewards_np = _stack(reward_records, (0, num_envs))
        if bool(runner_cfg.get("save_rollout", True)):
            payload: dict[str, Any] = {
                "observations": _stack(obs_records, (0, num_envs, len(fields))),
                "actions": _stack(action_records, (0, num_envs, 1)),
                "rewards": rewards_np,
                "terminated": _stack(terminated_records, (0, num_envs), dtype=bool),
                "truncated": _stack(truncated_records, (0, num_envs), dtype=bool),
                "policy_act_wall_time": np.asarray(act_time_records, dtype=np.float64),
                "observation_fields": np.asarray(fields),
                "policy_consumed_observation_fields": np.asarray(
                    policy.consumed_observation_fields
                ),
            }
            for name, records in step_records.items():
                payload[f"step_{name}"] = (
                    _stack(records, (0, num_envs))
                    if records
                    else np.full((rollout_steps, num_envs), np.nan, dtype=np.float32)
                )
            for name, records in policy_records.items():
                payload[name] = _stack(records, (0, *records[0].shape))
            np.savez_compressed(output_dir / "rollout.npz", **payload)

        act_times = np.asarray(act_time_records, dtype=np.float64)
        successes = np.asarray(policy_records.get("acados_nmpc_solver_success", []))
        fallbacks = np.asarray(policy_records.get("acados_nmpc_fallback", []))
        summary = {
            "run_name": output_dir.name,
            "run_config_path": str(run_path),
            "env_config_path": str(env_path),
            "policy_config_path": str(policy_path),
            "seed": seed,
            "num_envs": num_envs,
            "target_episodes": target_episodes,
            "stop_on_target_episodes": stop_on_target_episodes,
            "completed_episodes": int(final_metrics.get("completed_episodes", 0)),
            "rollout_steps": rollout_steps,
            "task_name": env_cfg.task_name,
            "interface_name": env_cfg.interface_name,
            "policy_name": policy_cfg.name,
            "mean_reward": float(np.mean(rewards_np)) if rewards_np.size else float("nan"),
            **_timing_summary(act_times, float(base_env.step_dt)),
            "solver_success_rate": float(np.mean(successes)) if successes.size else float("nan"),
            "fallback_rate": float(np.mean(fallbacks)) if fallbacks.size else float("nan"),
            "evaluation_complete": bool(final_metrics.get("evaluation_complete", False)),
            "acados_build_fingerprint": metadata["acados_build_fingerprint"],
        }
        for name, value in final_metrics.items():
            summary[f"benchmark_{name}"] = value
        append_csv_row(output_dir / "summary.csv", summary)
        append_csv_row(
            PROJECT_ROOT
            / run_cfg.get("logging", {}).get("root_dir", "logs/acados_nmpc/unified_tracking")
            / "summary.csv",
            summary,
        )
        save_yaml(summary, output_dir / "timing_summary.yaml")
        print(f"[INFO] acados NMPC unified rollout finished: {output_dir}")
        print(f"[INFO] Summary: {summary}")
    finally:
        if policy is not None:
            policy.close()
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
