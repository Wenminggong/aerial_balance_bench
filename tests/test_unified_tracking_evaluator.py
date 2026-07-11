from __future__ import annotations

import math

import pytest
import torch

from environments.evaluation.unified_tracking_evaluator import (
    UnifiedTrackingEvaluator,
    UnifiedTrackingEvaluatorCfg,
)
from environments.tasks.unified_tracking_task import TRAJECTORY_TYPE_TO_ID


def make_evaluator(num_envs: int = 2) -> UnifiedTrackingEvaluator:
    cfg = UnifiedTrackingEvaluatorCfg(
        episode_length_s=1.0,
        num_eval_episodes=2,
        target_zone=0.01,
        steady_state_window_s=0.2,
    )
    return UnifiedTrackingEvaluator(cfg, num_envs, "cpu", step_dt=0.1)


def test_metric_schema_is_stable_and_empty_categories_are_nan():
    evaluator = make_evaluator()
    metrics = evaluator.get_metrics()

    expected_keys = {
        "completed_episodes",
        "evaluation_complete",
        "mean_absolute_error",
        "root_mean_square_error",
        "maximum_absolute_error",
        "constant_success_rate",
        "constant_steady_state_error",
        "constant_steady_state_error_std",
        "constant_convergence_time",
        "constant_convergence_time_std",
        "constant_climbing_time",
        "constant_climbing_time_std",
    }
    for trajectory_name in TRAJECTORY_TYPE_TO_ID:
        expected_keys.update(
            {
                f"{trajectory_name}_completed_episodes",
                f"{trajectory_name}_mean_absolute_error",
                f"{trajectory_name}_root_mean_square_error",
                f"{trajectory_name}_maximum_absolute_error",
            }
        )

    assert set(metrics) == expected_keys
    assert metrics["completed_episodes"].item() == 0.0
    assert not metrics["evaluation_complete"].item()
    assert math.isnan(metrics["triangle_mean_absolute_error"].item())
    assert math.isnan(metrics["constant_success_rate"].item())


def test_global_per_type_and_constant_metrics():
    evaluator = make_evaluator()
    type_ids = torch.tensor(
        [TRAJECTORY_TYPE_TO_ID["constant"], TRAJECTORY_TYPE_TO_ID["sine"]],
        dtype=torch.long,
    )

    evaluator.update(
        {"pb": torch.tensor([0.0, 0.2]), "pg": torch.zeros(2)},
        terminated=torch.zeros(2, dtype=torch.bool),
        time_outs=torch.zeros(2, dtype=torch.bool),
        episode_length_buf=torch.ones(2, dtype=torch.long),
        trajectory_type_id=type_ids,
    )
    metrics = evaluator.update(
        {"pb": torch.tensor([0.0, 0.4]), "pg": torch.zeros(2)},
        terminated=torch.zeros(2, dtype=torch.bool),
        time_outs=torch.ones(2, dtype=torch.bool),
        episode_length_buf=torch.full((2,), 2, dtype=torch.long),
        trajectory_type_id=type_ids,
    )

    sine_rmse = math.sqrt((0.2**2 + 0.4**2) / 2.0)
    assert metrics["completed_episodes"].item() == 2.0
    assert metrics["evaluation_complete"].item()
    assert metrics["constant_completed_episodes"].item() == 1.0
    assert metrics["sine_completed_episodes"].item() == 1.0
    assert metrics["triangle_completed_episodes"].item() == 0.0
    assert metrics["mean_absolute_error"].item() == pytest.approx(0.15)
    assert metrics["root_mean_square_error"].item() == pytest.approx(sine_rmse / 2.0)
    assert metrics["maximum_absolute_error"].item() == pytest.approx(0.2)
    assert metrics["constant_mean_absolute_error"].item() == 0.0
    assert metrics["sine_mean_absolute_error"].item() == pytest.approx(0.3)
    assert metrics["sine_root_mean_square_error"].item() == pytest.approx(sine_rmse)
    assert metrics["sine_maximum_absolute_error"].item() == pytest.approx(0.4)
    assert math.isnan(metrics["trapezoid_mean_absolute_error"].item())
    assert metrics["constant_success_rate"].item() == 1.0
    assert metrics["constant_steady_state_error"].item() == 0.0
    assert metrics["constant_convergence_time"].item() == pytest.approx(0.1)
    assert metrics["constant_climbing_time"].item() == pytest.approx(0.1)
    assert metrics["constant_convergence_time_std"].item() == 0.0


def test_episode_reset_only_clears_selected_environments():
    evaluator = make_evaluator(3)
    evaluator.episode_abs_error_sum[:] = torch.tensor([1.0, 2.0, 3.0])
    evaluator.episode_samples[:] = 4.0
    evaluator.climbing_time[:] = torch.tensor([0.1, 0.2, 0.3])

    evaluator.reset_episode(torch.tensor([1]))

    assert torch.equal(evaluator.episode_abs_error_sum, torch.tensor([1.0, 0.0, 3.0]))
    assert torch.equal(evaluator.episode_samples, torch.tensor([4.0, 0.0, 4.0]))
    assert evaluator.climbing_time[0].item() == pytest.approx(0.1)
    assert evaluator.climbing_time[1].item() == pytest.approx(evaluator.timeout_value)
    assert evaluator.climbing_time[2].item() == pytest.approx(0.3)


def test_update_rejects_unknown_type_ids_and_wrong_shapes():
    evaluator = make_evaluator()
    state = {"pb": torch.zeros(2), "pg": torch.zeros(2)}
    zeros = torch.zeros(2, dtype=torch.bool)
    lengths = torch.zeros(2, dtype=torch.long)

    with pytest.raises(ValueError, match="unsupported"):
        evaluator.update(state, zeros, zeros, lengths, torch.tensor([0, 99]))
    with pytest.raises(ValueError, match="shape"):
        evaluator.update(state, zeros, zeros, lengths, torch.tensor([0]))

