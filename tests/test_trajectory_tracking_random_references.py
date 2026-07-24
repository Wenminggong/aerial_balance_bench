from __future__ import annotations

import pytest
import torch

from environments.tasks.trajectory_tracking_task import (
    LEGACY_RANDOM_TRAJECTORY_TYPES,
    TRAJECTORY_TYPE_TO_ID,
    TrajectoryTrackingTask,
    TrajectoryTrackingTaskCfg,
)


class FakeEnv:
    def __init__(self, num_envs: int):
        self.episode_length_buf = torch.zeros(num_envs, dtype=torch.long)
        self.ball_position = torch.zeros(num_envs)

    def set_ball_position_along_beam(self, env_ids: torch.Tensor, ball_position: torch.Tensor):
        self.ball_position[env_ids] = ball_position


def test_legacy_random_pool_excludes_held_out_references():
    task = TrajectoryTrackingTask(
        TrajectoryTrackingTaskCfg(trajectory_type="random"),
        4096,
        "cpu",
        0.1,
    )
    task.sample_reset(FakeEnv(4096), torch.arange(4096))

    expected_ids = {TRAJECTORY_TYPE_TO_ID[name] for name in LEGACY_RANDOM_TRAJECTORY_TYPES}
    assert set(task.trajectory_type_id.tolist()) == expected_ids
    assert TRAJECTORY_TYPE_TO_ID["random_b_spline"] not in expected_ids
    assert TRAJECTORY_TYPE_TO_ID["random_ramp_dwell"] not in expected_ids


def test_legacy_task_explicit_random_references_extend_through_preview_horizon():
    for trajectory_type in ("random_b_spline", "random_ramp_dwell"):
        torch.manual_seed(12)
        random_overrides = (
            {"random_b_spline_extrema_ranges": ((0.10, 0.10), (0.60, 0.60))}
            if trajectory_type == "random_b_spline"
            else {
                "random_ramp_dwell_continuous_ramp_probability": 1.0,
                "random_ramp_dwell_half_cycle_duration_range": (4.5, 4.5),
            }
        )
        cfg = TrajectoryTrackingTaskCfg(
            trajectory_type=trajectory_type,
            **random_overrides,
        )
        task = TrajectoryTrackingTask(cfg, 4, "cpu", 0.1, reference_horizon_s=20.6)
        env = FakeEnv(4)
        task.sample_reset(env, torch.arange(4))

        pg_zero, _ = task.get_reference(torch.zeros(4))
        preview_positions = []
        preview_velocities = []
        base_step = torch.full((4,), 198)
        for offset in range(6):
            pg, vg = task.get_reference(base_step + offset)
            preview_positions.append(pg)
            preview_velocities.append(vg)
        pg_preview = torch.stack(preview_positions, dim=-1)
        vg_preview = torch.stack(preview_velocities, dim=-1)

        assert torch.allclose(pg_zero, torch.full((4,), 0.35), atol=1.0e-6)
        assert not torch.allclose(pg_preview[:, 2], pg_preview[:, 3])
        assert torch.count_nonzero(vg_preview[:, 2]) > 0
        assert task.random_references.b_spline_duration_s == pytest.approx(20.6)
        assert task.random_references.ramp_dwell_duration_s == pytest.approx(20.6)
