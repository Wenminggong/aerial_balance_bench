from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from environments.tasks.random_reference_trajectories import (
    TRAJECTORY_TYPE_TO_ID,
    RandomReferenceTrajectories,
)


def make_cfg(**overrides) -> SimpleNamespace:
    values = {
        "random_b_spline_degree": 3,
        "random_b_spline_num_control_points": 6,
        "random_b_spline_duration_s": 20.0,
        "random_b_spline_position_range": (0.10, 0.60),
        "random_b_spline_start_position": 0.35,
        "random_b_spline_end_position": 0.35,
        "random_ramp_dwell_num_segments": 5,
        "random_ramp_dwell_duration_s": 20.0,
        "random_ramp_dwell_start_position": 0.35,
        "random_ramp_dwell_target_ranges": ((0.10, 0.30), (0.40, 0.60)),
        "random_ramp_duration_range": (1.0, 3.0),
        "random_dwell_duration_range": (0.0, 4.0),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def make_generator(num_envs: int = 2, **overrides) -> RandomReferenceTrajectories:
    return RandomReferenceTrajectories(make_cfg(**overrides), num_envs, "cpu", 0.0, 0.7)


def test_trajectory_type_ids_preserve_legacy_contract():
    assert TRAJECTORY_TYPE_TO_ID == {
        "sine": 0,
        "triangle": 1,
        "trapezoid": 2,
        "constant": 3,
        "random_b_spline": 4,
        "random_ramp_dwell": 5,
    }


def test_random_b_spline_has_fixed_endpoints_convex_hull_and_smooth_internal_knots():
    torch.manual_seed(4)
    generator = make_generator(2)
    env_ids = torch.arange(2)
    type_ids = torch.full((2,), TRAJECTORY_TYPE_TO_ID["random_b_spline"])
    generator.sample_reset(env_ids, type_ids)

    sample_times = torch.linspace(0.0, 20.0, 401).repeat(2, 1)
    positions = generator.get_position(sample_times, type_ids)
    control_min = generator.b_spline_control_positions.min(dim=-1).values
    control_max = generator.b_spline_control_positions.max(dim=-1).values

    assert torch.allclose(positions[:, 0], torch.full((2,), 0.35), atol=1.0e-6)
    assert torch.allclose(positions[:, -1], torch.full((2,), 0.35), atol=1.0e-6)
    assert torch.all(positions >= control_min[:, None] - 1.0e-6)
    assert torch.all(positions <= control_max[:, None] + 1.0e-6)

    knot = generator._b_spline_knots[generator.b_spline_degree + 1].item()
    h = 1.0e-3
    local_times = torch.tensor([[knot - h, knot, knot + h]]).repeat(2, 1)
    local_positions = generator.get_position(local_times, type_ids)
    left_velocity = (local_positions[:, 1] - local_positions[:, 0]) / h
    right_velocity = (local_positions[:, 2] - local_positions[:, 1]) / h
    assert torch.allclose(left_velocity, right_velocity, atol=2.0e-3, rtol=2.0e-2)


def test_random_b_spline_degree_is_configurable():
    generator = make_generator(
        1,
        random_b_spline_degree=2,
        random_b_spline_num_control_points=5,
    )
    type_ids = torch.tensor([TRAJECTORY_TYPE_TO_ID["random_b_spline"]])
    generator.sample_reset(torch.tensor([0]), type_ids)

    positions = generator.get_position(torch.linspace(0.0, 20.0, 101).unsqueeze(0), type_ids)

    assert positions.shape == (1, 101)
    assert torch.isfinite(positions).all()
    assert positions[0, 0].item() == pytest.approx(0.35)
    assert positions[0, -1].item() == pytest.approx(0.35)


def test_ramp_dwell_exact_ramp_dwell_and_final_hold():
    generator = make_generator(
        1,
        random_ramp_dwell_num_segments=1,
        random_ramp_dwell_duration_s=4.0,
        random_ramp_dwell_start_position=0.30,
        random_ramp_dwell_target_ranges=((0.50, 0.50),),
        random_ramp_duration_range=(2.0, 2.0),
        random_dwell_duration_range=(1.0, 1.0),
    )
    type_ids = torch.tensor([TRAJECTORY_TYPE_TO_ID["random_ramp_dwell"]])
    generator.sample_reset(torch.tensor([0]), type_ids)

    times = torch.tensor([[0.0, 1.0, 2.0, 2.5, 3.0, 4.0, 10.0]])
    positions = generator.get_position(times, type_ids)

    assert torch.allclose(
        positions,
        torch.tensor([[0.30, 0.40, 0.50, 0.50, 0.50, 0.50, 0.50]]),
        atol=1.0e-6,
    )


def test_ramp_dwell_clips_final_ramp_to_trajectory_duration():
    generator = make_generator(
        1,
        random_ramp_dwell_num_segments=1,
        random_ramp_dwell_duration_s=1.0,
        random_ramp_dwell_start_position=0.30,
        random_ramp_dwell_target_ranges=((0.50, 0.50),),
        random_ramp_duration_range=(2.0, 2.0),
        random_dwell_duration_range=(1.0, 1.0),
    )
    type_ids = torch.tensor([TRAJECTORY_TYPE_TO_ID["random_ramp_dwell"]])
    generator.sample_reset(torch.tensor([0]), type_ids)

    positions = generator.get_position(torch.tensor([[0.0, 0.5, 1.0, 2.0]]), type_ids)

    assert torch.allclose(positions, torch.tensor([[0.30, 0.35, 0.40, 0.40]]), atol=1.0e-6)


def test_sampling_is_seed_reproducible_and_partial_reset_preserves_other_envs():
    type_ids = torch.tensor(
        [
            TRAJECTORY_TYPE_TO_ID["random_b_spline"],
            TRAJECTORY_TYPE_TO_ID["random_ramp_dwell"],
            TRAJECTORY_TYPE_TO_ID["random_b_spline"],
            TRAJECTORY_TYPE_TO_ID["random_ramp_dwell"],
        ]
    )
    torch.manual_seed(17)
    first = make_generator(4)
    first.sample_reset(torch.arange(4), type_ids)
    torch.manual_seed(17)
    second = make_generator(4)
    second.sample_reset(torch.arange(4), type_ids)

    buffer_names = (
        "b_spline_control_positions",
        "ramp_start_positions",
        "ramp_end_positions",
        "ramp_start_times",
        "ramp_end_times",
        "dwell_end_times",
        "ramp_dwell_final_position",
    )
    for name in buffer_names:
        assert torch.equal(getattr(first, name), getattr(second, name))
    assert not torch.equal(
        first.b_spline_control_positions[0],
        first.b_spline_control_positions[2],
    )

    snapshots = {name: getattr(first, name).clone() for name in buffer_names}
    first.sample_reset(torch.tensor([1, 2]), type_ids)
    untouched = torch.tensor([0, 3])
    for name, snapshot in snapshots.items():
        assert torch.equal(getattr(first, name)[untouched], snapshot[untouched])


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        ({"random_b_spline_degree": 0}, "positive integer"),
        (
            {"random_b_spline_degree": 4, "random_b_spline_num_control_points": 4},
            "smaller",
        ),
        ({"random_b_spline_duration_s": 0.0}, "finite and positive"),
        ({"random_b_spline_position_range": (-0.1, 0.6)}, "beam position"),
        ({"random_ramp_dwell_num_segments": 0}, "positive integer"),
        ({"random_ramp_dwell_target_ranges": ()}, "must not be empty"),
        ({"random_ramp_duration_range": (0.0, 1.0)}, "strictly positive"),
        ({"random_dwell_duration_range": (-1.0, 1.0)}, "non-negative"),
    ),
)
def test_invalid_random_reference_configuration_fails_fast(
    overrides: dict[str, object],
    message: str,
):
    with pytest.raises(ValueError, match=message):
        make_generator(**overrides)
