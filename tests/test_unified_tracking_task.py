from __future__ import annotations

import math

import pytest
import torch

from environments.tasks.unified_tracking_task import (
    TRAJECTORY_TYPE_TO_ID,
    UnifiedTrackingTask,
    UnifiedTrackingTaskCfg,
)


class FakeEnv:
    def __init__(self, num_envs: int):
        self.episode_length_buf = torch.zeros(num_envs, dtype=torch.long)
        self.ball_position = torch.full((num_envs,), float("nan"))

    def set_ball_position_along_beam(self, env_ids: torch.Tensor, ball_position: torch.Tensor):
        self.ball_position[env_ids.cpu()] = ball_position.cpu()


def make_task(num_envs: int = 4, **cfg_overrides) -> UnifiedTrackingTask:
    cfg = UnifiedTrackingTaskCfg(**cfg_overrides)
    return UnifiedTrackingTask(cfg, num_envs, "cpu", 0.1, 0.0, 0.7)


@pytest.mark.parametrize(
    ("trajectory_type", "expected_id"),
    list(TRAJECTORY_TYPE_TO_ID.items()),
)
def test_singleton_sampling_selects_requested_type(trajectory_type: str, expected_id: int):
    task = make_task(32, trajectory_types=(trajectory_type,), trajectory_type_weights=())
    env = FakeEnv(32)

    task.sample_reset(env, torch.arange(32))

    assert torch.all(task.trajectory_type_id == expected_id)


def test_mixed_sampling_honors_weights_and_is_seed_reproducible():
    overrides = {
        "trajectory_types": ("constant", "sine", "triangle", "trapezoid"),
        "trajectory_type_weights": (0.0, 0.0, 1.0, 0.0),
    }
    torch.manual_seed(123)
    first = make_task(16, **overrides)
    first.sample_reset(FakeEnv(16), torch.arange(16))

    torch.manual_seed(123)
    second = make_task(16, **overrides)
    second.sample_reset(FakeEnv(16), torch.arange(16))

    assert torch.all(first.trajectory_type_id == TRAJECTORY_TYPE_TO_ID["triangle"])
    for name in ("trajectory_type_id", "center", "amplitude", "period", "phase", "initial_ball_position"):
        assert torch.equal(getattr(first, name), getattr(second, name))
    assert first.get_config_info() == {
        "trajectory_type_to_id": TRAJECTORY_TYPE_TO_ID,
        "trajectory_types": ["constant", "sine", "triangle", "trapezoid"],
        "normalized_trajectory_type_weights": [0.0, 0.0, 1.0, 0.0],
    }


def test_constant_reference_and_independent_initial_position():
    task = make_task(
        64,
        trajectory_types=("constant",),
        constant_goal_range=(0.25, 0.45),
        initial_ball_position_range=(0.1, 0.6),
        min_initial_reference_distance=0.1,
    )
    env = FakeEnv(64)
    task.sample_reset(env, torch.arange(64))

    pg, vg = task.get_reference(torch.arange(64))
    later_pg, later_vg = task.get_reference(torch.arange(64) + 1000)

    assert torch.equal(pg, later_pg)
    assert torch.count_nonzero(vg) == 0
    assert torch.count_nonzero(later_vg) == 0
    assert torch.all(torch.abs(task.initial_ball_position - pg) >= 0.1 - 1.0e-6)
    assert torch.equal(env.ball_position, task.initial_ball_position)


@pytest.mark.parametrize(
    ("trajectory_type", "quarter_period_value"),
    (("sine", 1.0), ("triangle", 1.0), ("trapezoid", 1.0)),
)
def test_dynamic_waveforms_and_finite_difference_velocity(
    trajectory_type: str,
    quarter_period_value: float,
):
    task = make_task(
        2,
        trajectory_types=(trajectory_type,),
        dynamic_center_range=(0.35, 0.35),
        amplitude_range=(0.1, 0.1),
        period_range=(4.0, 4.0),
        phase_range=(0.0, 0.0),
    )
    env = FakeEnv(2)
    task.sample_reset(env, torch.arange(2))

    pg_zero, _ = task.get_reference(torch.zeros(2))
    pg_quarter, vg_quarter = task.get_reference(torch.full((2,), 10.0))
    pg_next, _ = task.get_reference(torch.full((2,), 11.0))

    assert torch.allclose(pg_zero, torch.full((2,), 0.35), atol=1.0e-6)
    assert torch.allclose(
        pg_quarter,
        torch.full((2,), 0.35 + 0.1 * quarter_period_value),
        atol=1.0e-6,
    )
    assert torch.allclose(vg_quarter, (pg_next - pg_quarter) / 0.1, atol=1.0e-6)
    assert torch.allclose(env.ball_position, pg_zero, atol=1.0e-6)


def test_reference_preview_shape_offsets_and_values():
    task = make_task(
        3,
        trajectory_types=("sine",),
        dynamic_center_range=(0.35, 0.35),
        amplitude_range=(0.1, 0.1),
        period_range=(4.0, 4.0),
        phase_range=(0.0, 0.0),
    )
    task.sample_reset(FakeEnv(3), torch.arange(3))
    steps = torch.tensor([0, 1, 2])

    pg_preview, vg_preview = task.get_reference_preview(steps, future_steps=5)

    assert pg_preview.shape == (3, 6)
    assert vg_preview.shape == (3, 6)
    for offset in range(6):
        pg, vg = task.get_reference(steps + offset)
        assert torch.allclose(pg_preview[:, offset], pg)
        assert torch.allclose(vg_preview[:, offset], vg)
    with pytest.raises(ValueError, match="non-negative"):
        task.get_reference_preview(steps, future_steps=-1)


def test_partial_reset_preserves_other_environment_parameters():
    torch.manual_seed(7)
    task = make_task(6)
    env = FakeEnv(6)
    task.sample_reset(env, torch.arange(6))
    untouched = torch.tensor([0, 2, 4, 5])
    reset_ids = torch.tensor([1, 3])
    snapshots = {
        name: getattr(task, name).clone()
        for name in (
            "trajectory_type_id",
            "center",
            "amplitude",
            "period",
            "phase",
            "initial_ball_position",
            "previous_abs_error",
        )
    }

    task.sample_reset(env, reset_ids)

    for name, snapshot in snapshots.items():
        assert torch.equal(getattr(task, name)[untouched], snapshot[untouched])


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        ({"trajectory_types": ()}, "at least one"),
        ({"trajectory_types": ("sawtooth",)}, "Unsupported"),
        ({"trajectory_types": ("sine", "sine")}, "duplicates"),
        ({"trajectory_types": ("sine",), "trajectory_type_weights": (1.0, 1.0)}, "match"),
        ({"trajectory_types": ("sine",), "trajectory_type_weights": (-1.0,)}, "non-negative"),
        ({"trajectory_types": ("sine",), "trajectory_type_weights": (0.0,)}, "positive sum"),
        ({"period_range": (0.0, 1.0)}, "strictly positive"),
        ({"amplitude_range": (-0.1, 0.1)}, "non-negative"),
        ({"dynamic_center_range": (0.1, 0.6), "amplitude_range": (0.2, 0.2)}, "dynamic reference"),
        ({"constant_initialization_mode": "invalid"}, "initialization_mode"),
        (
            {
                "trajectory_types": ("constant",),
                "constant_goal_range": (0.35, 0.35),
                "initial_ball_position_range": (0.3, 0.4),
                "min_initial_reference_distance": 0.06,
            },
            "infeasible",
        ),
    ),
)
def test_invalid_configuration_fails_fast(overrides: dict[str, object], message: str):
    with pytest.raises(ValueError, match=message):
        make_task(**overrides)


def test_common_reward_uses_reference_velocity_progress_and_final_termination():
    task = make_task(
        2,
        trajectory_types=("constant",),
        constant_initialization_mode="fixed",
        position_weight=1.0,
        velocity_weight=2.0,
        command_weight=3.0,
        action_weight=4.0,
        progress_weight=5.0,
        failure_penalty=10.0,
        max_error_for_failure=0.5,
    )
    task.previous_abs_error[:] = torch.tensor([0.4, 0.3])
    state = {
        "pb": torch.tensor([0.2, 0.6]),
        "pg": torch.zeros(2),
        "vb": torch.tensor([0.3, 0.4]),
        "vg": torch.tensor([0.1, 0.1]),
        "theta": torch.zeros(2),
        "theta_limit": torch.full((2,), math.pi),
    }
    command = {"command_z": torch.tensor([0.1, 0.2])}
    action = torch.tensor([[0.2], [0.1]])
    terminated = torch.tensor([False, True])

    reward = task.compute_reward(state, command, action, terminated=terminated)
    error = state["pb"] - state["pg"]
    velocity_error = state["vb"] - state["vg"]
    expected = -error.square() - 2.0 * velocity_error.square()
    expected -= 3.0 * command["command_z"].square() + 4.0 * action.squeeze(-1).square()
    expected += 5.0 * (torch.tensor([0.4, 0.3]) - error.abs())
    expected -= 10.0 * terminated.float()

    assert torch.allclose(reward, expected)
    assert torch.equal(task.previous_abs_error, error.abs())
    assert torch.equal(task.compute_task_dones(state), torch.tensor([False, True]))

