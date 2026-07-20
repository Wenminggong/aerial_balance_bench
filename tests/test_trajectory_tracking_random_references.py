from __future__ import annotations

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


def test_legacy_task_explicit_random_references_start_and_hold_at_endpoint():
    for trajectory_type in ("random_b_spline", "random_ramp_dwell"):
        torch.manual_seed(12)
        cfg = TrajectoryTrackingTaskCfg(trajectory_type=trajectory_type)
        task = TrajectoryTrackingTask(cfg, 4, "cpu", 0.1)
        env = FakeEnv(4)
        task.sample_reset(env, torch.arange(4))

        pg_zero, _ = task.get_reference(torch.zeros(4))
        pg_end, vg_end = task.get_reference(torch.full((4,), 200))
        pg_later, vg_later = task.get_reference(torch.full((4,), 300))

        assert torch.allclose(pg_zero, torch.full((4,), 0.35), atol=1.0e-6)
        assert torch.allclose(pg_end, pg_later)
        assert torch.count_nonzero(vg_end) == 0
        assert torch.count_nonzero(vg_later) == 0
