from __future__ import annotations

import csv
import importlib.util
import math
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "scripts" / "summarize_rl_eval_summary.py"
SPEC = importlib.util.spec_from_file_location("summarize_rl_eval_summary", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
summary_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(summary_module)


def _write_rows(path: Path, rows: list[dict[str, object]]):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_summarize_supports_tracking_metrics(tmp_path: Path):
    source = tmp_path / "summary.csv"
    output = tmp_path / "aggregate.csv"
    _write_rows(
        source,
        [
            {
                "benchmark_completed_episodes": 10,
                "benchmark_mean_absolute_error": 0.1,
                "benchmark_root_mean_square_error": 0.2,
                "benchmark_maximum_absolute_error": 0.3,
                "benchmark_sine_completed_episodes": 4,
                "benchmark_sine_mean_absolute_error": 0.05,
                "policy_compute_time_mean": 0.001,
            },
            {
                "benchmark_completed_episodes": 30,
                "benchmark_mean_absolute_error": 0.3,
                "benchmark_root_mean_square_error": 0.4,
                "benchmark_maximum_absolute_error": 0.5,
                "benchmark_sine_completed_episodes": 6,
                "benchmark_sine_mean_absolute_error": 0.15,
                "policy_compute_time_mean": 0.003,
            },
        ],
    )

    summary = summary_module.summarize(source, output)

    assert math.isclose(summary["benchmark_mean_absolute_error_mean"], 0.25)
    assert math.isclose(summary["benchmark_sine_mean_absolute_error_mean"], 0.11)
    assert summary["benchmark_sine_mean_absolute_error_weight_column"] == (
        "benchmark_sine_completed_episodes"
    )
    assert math.isclose(summary["policy_compute_time_mean_mean"], 0.002)
    assert output.exists()


def test_summarize_keeps_target_position_pooled_metrics(tmp_path: Path):
    source = tmp_path / "summary.csv"
    output = tmp_path / "aggregate.csv"
    _write_rows(
        source,
        [
            {
                "benchmark_completed_episodes": 5,
                "benchmark_success_rate": 0.8,
                "benchmark_steady_state_error": 0.02,
                "benchmark_steady_state_error_std": 0.01,
                "benchmark_convergence_time": 1.0,
                "benchmark_convergence_time_std": 0.2,
                "benchmark_climbing_time": 0.5,
                "benchmark_climbing_time_std": 0.1,
                "policy_compute_time_mean": 0.001,
            }
        ],
    )

    summary = summary_module.summarize(source, output)

    assert math.isclose(summary["benchmark_success_rate_mean"], 0.8)
    assert math.isclose(summary["benchmark_steady_state_error_mean"], 0.02)
    assert math.isclose(summary["benchmark_steady_state_error_std"], 0.01)
