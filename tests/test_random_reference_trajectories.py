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
        "random_b_spline_sampling_mode": "paired_alternating_extrema",
        "random_b_spline_degree": 3,
        "random_b_spline_num_control_points": 12,
        "random_b_spline_duration_s": 20.0,
        "random_b_spline_position_range": (0.10, 0.60),
        "random_b_spline_extrema_ranges": ((0.10, 0.30), (0.40, 0.60)),
        "random_b_spline_start_position": 0.35,
        "random_b_spline_end_position": 0.35,
        "random_ramp_dwell_sampling_mode": "half_cycle_mixture",
        "random_ramp_dwell_num_segments": 6,
        "random_ramp_dwell_duration_s": 20.0,
        "random_ramp_dwell_start_position": 0.35,
        "random_ramp_dwell_target_ranges": ((0.10, 0.30), (0.40, 0.60)),
        "random_ramp_duration_range": (1.0, 3.0),
        "random_dwell_duration_range": (0.0, 4.0),
        "random_ramp_dwell_half_cycle_duration_range": (4.0, 5.0),
        "random_ramp_dwell_continuous_ramp_probability": 0.5,
        "random_ramp_dwell_ramp_fraction_range": (0.45, 0.55),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def make_generator(
    num_envs: int = 2,
    *,
    reference_horizon_s: float | None = None,
    **overrides,
) -> RandomReferenceTrajectories:
    return RandomReferenceTrajectories(
        make_cfg(**overrides),
        num_envs,
        "cpu",
        0.0,
        0.7,
        reference_horizon_s=reference_horizon_s,
    )


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


def test_random_b_spline_uses_paired_alternating_extrema():
    torch.manual_seed(9)
    generator = make_generator(32)
    type_ids = torch.full((32,), TRAJECTORY_TYPE_TO_ID["random_b_spline"])
    generator.sample_reset(torch.arange(32), type_ids)

    extrema_blocks = generator.b_spline_control_positions[:, 1:-1].reshape(32, -1, 2)
    block_positions = extrema_blocks[..., 0]
    low_block = (block_positions >= 0.10) & (block_positions <= 0.30)
    high_block = (block_positions >= 0.40) & (block_positions <= 0.60)

    assert torch.equal(extrema_blocks[..., 0], extrema_blocks[..., 1])
    assert torch.all(low_block | high_block)
    assert torch.all(low_block[:, 1:] == high_block[:, :-1])
    assert torch.all(high_block[:, 1:] == low_block[:, :-1])

    config_info = generator.get_config_info()["random_b_spline"]
    assert config_info["sampling_mode"] == "paired_alternating_extrema"
    assert config_info["extrema_ranges"] == [[0.10, 0.30], [0.40, 0.60]]


def test_random_b_spline_frequency_matched_distribution_regression():
    torch.manual_seed(41)
    num_envs = 512
    generator = make_generator(num_envs, reference_horizon_s=20.5)
    type_ids = torch.full((num_envs,), TRAJECTORY_TYPE_TO_ID["random_b_spline"])
    generator.sample_reset(torch.arange(num_envs), type_ids)

    step_dt = 1.0 / 60.0
    sample_times = torch.arange(0.0, 20.0 + step_dt / 2.0, step_dt).repeat(num_envs, 1)
    positions = generator.get_position(sample_times, type_ids)
    velocities = torch.diff(positions, dim=-1) / step_dt
    median_span = torch.median(positions.max(dim=-1).values - positions.min(dim=-1).values)
    median_max_speed = torch.median(velocities.abs().max(dim=-1).values)
    median_rms_speed = torch.median(torch.sqrt(velocities.square().mean(dim=-1)))

    assert 0.28 <= median_span.item() <= 0.42
    assert 0.14 <= median_max_speed.item() <= 0.24
    assert 0.055 <= median_rms_speed.item() <= 0.105


def test_random_b_spline_degree_is_configurable():
    generator = make_generator(
        1,
        random_b_spline_sampling_mode="uniform",
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


def test_random_b_spline_extends_to_required_reference_horizon():
    generator = make_generator(
        1,
        reference_horizon_s=20.5,
        random_b_spline_sampling_mode="uniform",
        random_b_spline_position_range=(0.10, 0.10),
    )
    type_ids = torch.tensor([TRAJECTORY_TYPE_TO_ID["random_b_spline"]])
    generator.sample_reset(torch.tensor([0]), type_ids)

    positions = generator.get_position(
        torch.tensor([[20.0, 20.1, 20.4, 20.5, 20.6]]),
        type_ids,
    )

    assert generator.b_spline_duration_s == pytest.approx(20.5)
    assert generator.get_config_info()["random_b_spline"]["duration_s"] == pytest.approx(20.0)
    assert not torch.allclose(positions[:, 0], positions[:, 1])
    assert torch.all((positions >= 0.10) & (positions <= 0.35))
    assert positions[0, 3].item() == pytest.approx(0.35)
    assert positions[0, 4].item() == pytest.approx(0.35)

    h = 1.0e-3
    boundary_positions = generator.get_position(
        torch.tensor([[20.0 - h, 20.0, 20.0 + h]]),
        type_ids,
    )
    left_velocity = (boundary_positions[:, 1] - boundary_positions[:, 0]) / h
    right_velocity = (boundary_positions[:, 2] - boundary_positions[:, 1]) / h
    assert torch.allclose(left_velocity, right_velocity, atol=2.0e-3, rtol=2.0e-2)


def test_ramp_dwell_exact_ramp_dwell_and_defensive_final_hold():
    generator = make_generator(
        1,
        random_ramp_dwell_sampling_mode="independent",
        random_ramp_dwell_num_segments=1,
        random_ramp_dwell_duration_s=4.0,
        random_ramp_dwell_start_position=0.30,
        random_ramp_dwell_target_ranges=((0.50, 0.50),),
        random_ramp_duration_range=(2.0, 2.0),
        random_dwell_duration_range=(2.0, 2.0),
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


def test_half_cycle_continuous_profile_has_alternating_targets_and_no_dwells():
    torch.manual_seed(15)
    generator = make_generator(
        16,
        random_ramp_dwell_continuous_ramp_probability=1.0,
        random_ramp_dwell_half_cycle_duration_range=(4.0, 4.0),
    )
    type_ids = torch.full((16,), TRAJECTORY_TYPE_TO_ID["random_ramp_dwell"])
    generator.sample_reset(torch.arange(16), type_ids)

    ramp_durations = generator.ramp_end_times - generator.ramp_start_times
    dwell_durations = generator.dwell_end_times - generator.ramp_end_times
    full_targets = generator.ramp_end_positions[:, :5]
    low_target = (full_targets >= 0.10) & (full_targets <= 0.30)
    high_target = (full_targets >= 0.40) & (full_targets <= 0.60)

    assert torch.all(generator.ramp_dwell_continuous_profile)
    assert torch.allclose(generator.ramp_half_cycle_durations, torch.full((16, 6), 4.0))
    assert torch.allclose(ramp_durations[:, 0], torch.full((16,), 2.0))
    assert torch.allclose(ramp_durations[:, 1:5], torch.full((16, 4), 4.0))
    assert torch.count_nonzero(dwell_durations) == 0
    assert torch.all(low_target | high_target)
    assert torch.all(low_target[:, 1:] == high_target[:, :-1])
    assert torch.all(high_target[:, 1:] == low_target[:, :-1])


def test_half_cycle_ramp_dwell_profile_has_short_first_ramp_and_positive_dwells():
    torch.manual_seed(16)
    generator = make_generator(
        8,
        random_ramp_dwell_continuous_ramp_probability=0.0,
        random_ramp_dwell_half_cycle_duration_range=(4.0, 4.0),
        random_ramp_dwell_ramp_fraction_range=(0.5, 0.5),
    )
    type_ids = torch.full((8,), TRAJECTORY_TYPE_TO_ID["random_ramp_dwell"])
    generator.sample_reset(torch.arange(8), type_ids)

    ramp_durations = generator.ramp_end_times - generator.ramp_start_times
    dwell_durations = generator.dwell_end_times - generator.ramp_end_times

    assert torch.count_nonzero(generator.ramp_dwell_continuous_profile) == 0
    assert torch.allclose(ramp_durations[:, 0], torch.full((8,), 1.0))
    assert torch.allclose(ramp_durations[:, 1:5], torch.full((8, 4), 2.0))
    assert torch.allclose(dwell_durations[:, :5], torch.full((8, 5), 2.0))
    assert torch.allclose(generator.dwell_end_times[:, 4], torch.full((8,), 19.0))


def test_half_cycle_duration_and_slope_support_cover_eight_to_ten_second_targets():
    generator = make_generator(1)
    low_range, high_range = generator.ramp_dwell_target_ranges
    half_cycle_low, half_cycle_high = generator.half_cycle_duration_range
    fraction_low, fraction_high = generator.ramp_fraction_range
    minimum_transition = high_range[0] - low_range[1]
    maximum_transition = high_range[1] - low_range[0]

    continuous_slope_support = (
        minimum_transition / half_cycle_high,
        maximum_transition / half_cycle_low,
    )
    dwell_slope_support = (
        minimum_transition / (fraction_high * half_cycle_high),
        maximum_transition / (fraction_low * half_cycle_low),
    )
    triangle_slope_support = (4.0 * 0.10 / 10.0, 4.0 * 0.25 / 8.0)
    trapezoid_slope_support = (8.0 * 0.10 / 10.0, 8.0 * 0.25 / 8.0)

    assert generator.half_cycle_duration_range == pytest.approx((4.0, 5.0))
    assert continuous_slope_support[0] <= triangle_slope_support[0]
    assert continuous_slope_support[1] >= triangle_slope_support[1]
    assert dwell_slope_support[0] <= trapezoid_slope_support[0]
    assert dwell_slope_support[1] >= trapezoid_slope_support[1]


def test_ramp_dwell_extends_past_configured_duration_to_reference_horizon():
    generator = make_generator(
        1,
        reference_horizon_s=5.0,
        random_ramp_dwell_sampling_mode="independent",
        random_ramp_dwell_num_segments=1,
        random_ramp_dwell_duration_s=4.0,
        random_ramp_dwell_start_position=0.30,
        random_ramp_dwell_target_ranges=((0.50, 0.50),),
        random_ramp_duration_range=(2.0, 5.0),
        random_dwell_duration_range=(0.0, 0.0),
    )
    type_ids = torch.tensor([TRAJECTORY_TYPE_TO_ID["random_ramp_dwell"]])
    generator.sample_reset(torch.tensor([0]), type_ids)

    positions = generator.get_position(
        torch.tensor([[0.0, 4.0, 4.5, 5.0, 6.0]]),
        type_ids,
    )

    assert generator.ramp_dwell_duration_s == pytest.approx(5.0)
    assert generator.get_config_info()["random_ramp_dwell"]["duration_s"] == pytest.approx(4.0)
    assert torch.allclose(
        positions,
        torch.tensor([[0.30, 0.46, 0.48, 0.50, 0.50]]),
        atol=1.0e-6,
    )
    assert generator.dwell_end_times[0, -1].item() == pytest.approx(5.0)


def test_ramp_dwell_clips_final_ramp_to_trajectory_duration():
    generator = make_generator(
        1,
        random_ramp_dwell_sampling_mode="independent",
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


def test_ramp_dwell_rejects_ranges_that_cannot_cover_reference_horizon():
    generator = make_generator(
        1,
        reference_horizon_s=5.0,
        random_ramp_dwell_sampling_mode="independent",
        random_ramp_dwell_num_segments=1,
        random_ramp_dwell_duration_s=4.0,
        random_ramp_duration_range=(1.0, 1.0),
        random_dwell_duration_range=(1.0, 1.0),
    )
    type_ids = torch.tensor([TRAJECTORY_TYPE_TO_ID["random_ramp_dwell"]])

    with pytest.raises(ValueError, match="can cover at most"):
        generator.sample_reset(torch.tensor([0]), type_ids)


def test_half_cycle_mode_rejects_configuration_that_cannot_cover_horizon():
    with pytest.raises(ValueError, match="cannot cover"):
        make_generator(
            1,
            reference_horizon_s=28.0,
            random_ramp_dwell_continuous_ramp_probability=1.0,
        )


def test_half_cycle_mode_extends_durations_within_range_to_cover_horizon():
    torch.manual_seed(27)
    generator = make_generator(
        4,
        reference_horizon_s=20.5,
        random_ramp_dwell_num_segments=5,
        random_ramp_dwell_continuous_ramp_probability=1.0,
    )
    type_ids = torch.full((4,), TRAJECTORY_TYPE_TO_ID["random_ramp_dwell"])
    generator.sample_reset(torch.arange(4), type_ids)

    half_cycles = generator.ramp_half_cycle_durations
    scheduled_duration = 0.5 * half_cycles[:, 0] + half_cycles[:, 1:].sum(dim=-1)

    assert torch.all(half_cycles >= 4.0)
    assert torch.all(half_cycles <= 5.0)
    assert torch.all(scheduled_duration >= 20.5 - 1.0e-5)
    assert torch.allclose(generator.dwell_end_times[:, -1], torch.full((4,), 20.5))


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
        "ramp_half_cycle_durations",
        "ramp_fractions",
        "ramp_dwell_continuous_profile",
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
        ({"random_b_spline_sampling_mode": "other"}, "must be one of"),
        ({"random_b_spline_num_control_points": 11}, "even"),
        ({"random_b_spline_extrema_ranges": ((0.1, 0.3),)}, "two strictly ordered"),
        (
            {"random_b_spline_extrema_ranges": ((0.1, 0.3), (0.6, 0.7))},
            "within random_b_spline_position_range",
        ),
        ({"random_ramp_dwell_num_segments": 0}, "positive integer"),
        ({"random_ramp_dwell_target_ranges": ()}, "must not be empty"),
        ({"random_ramp_dwell_sampling_mode": "other"}, "must be one of"),
        ({"random_ramp_dwell_target_ranges": ((0.1, 0.3),)}, "two strictly ordered"),
        ({"random_ramp_duration_range": (0.0, 1.0)}, "strictly positive"),
        ({"random_dwell_duration_range": (-1.0, 1.0)}, "non-negative"),
        ({"random_ramp_dwell_half_cycle_duration_range": (0.0, 5.0)}, "strictly positive"),
        ({"random_ramp_dwell_continuous_ramp_probability": 1.1}, "lie in"),
        ({"random_ramp_dwell_ramp_fraction_range": (0.0, 0.5)}, "strictly between"),
    ),
)
def test_invalid_random_reference_configuration_fails_fast(
    overrides: dict[str, object],
    message: str,
):
    with pytest.raises(ValueError, match=message):
        make_generator(**overrides)


def test_invalid_reference_horizon_fails_fast():
    with pytest.raises(ValueError, match="reference_horizon_s"):
        make_generator(reference_horizon_s=0.0)
