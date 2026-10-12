from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from fit_real_velocity_response import delayed_command
from fit_sim_velocity_response_delay import (
    fit_trajectory_delays,
    load_simulation_trajectories,
    resolve_step_dt,
    summarize_delay_fits,
)


def _write_synthetic_rollout(path: Path, step_dt: float, delay_samples: list[int]) -> None:
    rng = np.random.default_rng(42)
    num_steps = 180
    num_envs = len(delay_samples)
    command = rng.normal(size=(num_steps, num_envs)).astype(np.float32)
    observations = np.zeros((num_steps, num_envs, 11), dtype=np.float32)
    for env_id, samples in enumerate(delay_samples):
        response = delayed_command(
            command[:-1, env_id].astype(np.float64),
            samples * step_dt,
            step_dt,
        )
        observations[1:, env_id, 7] = response.astype(np.float32)

    terminated = np.zeros((num_steps, num_envs), dtype=bool)
    truncated = np.zeros_like(terminated)
    truncated[-1] = True
    fields = np.asarray(
        ["pb", "vb", "ab", "theta", "omega", "alpha", "drz", "vrz", "arz", "pg", "a_prev"]
    )
    np.savez_compressed(
        path,
        observations=observations,
        observation_fields=fields,
        step_command_z=command,
        terminated=terminated,
        truncated=truncated,
    )


def test_parallel_pure_delay_fit_recovers_per_environment_delays(tmp_path: Path):
    step_dt = 0.02
    expected_samples = [2, 5, 9]
    rollout_path = tmp_path / "rollout.npz"
    _write_synthetic_rollout(rollout_path, step_dt, expected_samples)

    trajectories, load_info = load_simulation_trajectories(
        rollout_path,
        min_samples=30,
        min_command_std=1.0e-6,
    )
    rows = fit_trajectory_delays(
        trajectories,
        step_dt=step_dt,
        max_delay_s=0.3,
    )

    assert load_info["num_envs"] == 3
    assert load_info["skipped_trajectory_count"] == 0
    assert len(rows) == 3
    np.testing.assert_allclose(
        [row["delay_s"] for row in rows],
        np.asarray(expected_samples) * step_dt,
        atol=2.0e-5,
    )

    summary = summarize_delay_fits(
        rows,
        rollout_path=rollout_path,
        step_dt=step_dt,
        step_dt_source="test",
        max_delay_s=0.3,
        load_info=load_info,
    )
    expected_delays = np.asarray(expected_samples, dtype=np.float64) * step_dt
    assert np.isclose(summary["delay_s"]["mean"], np.mean(expected_delays))
    assert np.isclose(summary["delay_s"]["std"], np.std(expected_delays))
    assert np.isclose(summary["delay_s"]["lower"], np.min(expected_delays), atol=2.0e-5)
    assert np.isclose(summary["delay_s"]["upper"], np.max(expected_delays), atol=2.0e-5)


def test_step_dt_is_loaded_from_resolved_run(tmp_path: Path):
    rollout_path = tmp_path / "rollout.npz"
    rollout_path.touch()
    with (tmp_path / "resolved_run.yaml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump({"control_step_dt": 0.025}, stream)

    step_dt, source = resolve_step_dt(rollout_path, None)

    assert np.isclose(step_dt, 0.025)
    assert source == str(tmp_path / "resolved_run.yaml")


def test_terminal_transition_splits_parallel_episode_segments(tmp_path: Path):
    rollout_path = tmp_path / "rollout.npz"
    num_steps = 20
    observations = np.zeros((num_steps, 1, 11), dtype=np.float32)
    command = np.arange(num_steps, dtype=np.float32)[:, None]
    observations[1:, 0, 7] = command[:-1, 0]
    terminated = np.zeros((num_steps, 1), dtype=bool)
    terminated[9, 0] = True
    fields = np.asarray(
        ["pb", "vb", "ab", "theta", "omega", "alpha", "drz", "vrz", "arz", "pg", "a_prev"]
    )
    np.savez_compressed(
        rollout_path,
        observations=observations,
        observation_fields=fields,
        step_command_z=command,
        terminated=terminated,
    )

    trajectories, load_info = load_simulation_trajectories(
        rollout_path,
        min_samples=3,
        min_command_std=0.0,
    )

    assert load_info["candidate_trajectory_count"] == 2
    assert [(item.start_step, item.end_step) for item in trajectories] == [(0, 8), (10, 18)]
