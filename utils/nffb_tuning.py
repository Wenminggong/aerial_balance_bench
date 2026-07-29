"""Rollout analysis helpers for NFFB reference-tracking tuning."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import yaml


SATURATION_FIELDS = (
    "policy_reference_acceleration_saturated",
    "policy_ball_acceleration_saturated",
    "policy_theta_command_saturated",
    "policy_omega_command_saturated",
    "policy_velocity_saturated",
    "policy_acceleration_saturated",
)

DEFAULT_TRAJECTORY_TYPE_TO_ID = {
    "sine": 0,
    "triangle": 1,
    "trapezoid": 2,
    "constant": 3,
    "random_b_spline": 4,
    "random_ramp_dwell": 5,
}


def _finite_mean(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(np.mean(array)) if array.size else float("nan")


def _finite_percentile(values: Iterable[float], percentile: float) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(np.percentile(array, percentile)) if array.size else float("nan")


def _safe_rate(values: np.ndarray) -> float:
    return float(np.mean(values.astype(np.float64))) if values.size else 0.0


def _wrap_phase(phase: float) -> float:
    return float((phase + math.pi) % (2.0 * math.pi) - math.pi)


def _fit_sine(
    signal: np.ndarray,
    time_s: np.ndarray,
    angular_frequency: float,
) -> tuple[float, float, float]:
    design = np.column_stack(
        (
            np.ones_like(time_s),
            np.sin(angular_frequency * time_s),
            np.cos(angular_frequency * time_s),
        )
    )
    coefficients = np.linalg.lstsq(design, signal, rcond=None)[0]
    amplitude = float(np.hypot(coefficients[1], coefficients[2]))
    phase = float(np.arctan2(coefficients[2], coefficients[1]))
    return float(coefficients[0]), amplitude, phase


def _frequency_response(
    pb: np.ndarray,
    pg: np.ndarray,
    period_s: float,
    step_dt: float,
) -> dict[str, float]:
    if (
        not math.isfinite(period_s)
        or period_s <= 0.0
        or not math.isfinite(step_dt)
        or step_dt <= 0.0
    ):
        return {
            "steady_nrmse": float("nan"),
            "amplitude_gain": float("nan"),
            "phase_error_deg": float("nan"),
            "phase_lag_s": float("nan"),
        }

    steady_start = int(math.ceil(period_s / step_dt))
    if pb.size - steady_start < max(12, int(math.ceil(0.75 * period_s / step_dt))):
        return {
            "steady_nrmse": float("nan"),
            "amplitude_gain": float("nan"),
            "phase_error_deg": float("nan"),
            "phase_lag_s": float("nan"),
        }

    pb_steady = pb[steady_start:]
    pg_steady = pg[steady_start:]
    time_s = np.arange(pb_steady.size, dtype=np.float64) * step_dt
    angular_frequency = 2.0 * math.pi / period_s
    _, pb_amplitude, pb_phase = _fit_sine(pb_steady, time_s, angular_frequency)
    _, pg_amplitude, pg_phase = _fit_sine(pg_steady, time_s, angular_frequency)
    phase_error = _wrap_phase(pb_phase - pg_phase)
    steady_rmse = float(np.sqrt(np.mean(np.square(pb_steady - pg_steady))))
    return {
        "steady_nrmse": (
            steady_rmse / pg_amplitude if pg_amplitude > 1.0e-9 else float("nan")
        ),
        "amplitude_gain": (
            pb_amplitude / pg_amplitude if pg_amplitude > 1.0e-9 else float("nan")
        ),
        "phase_error_deg": math.degrees(phase_error),
        "phase_lag_s": -phase_error / angular_frequency,
    }


def _load_metadata(run_dir: Path) -> dict[str, Any]:
    metadata_path = run_dir / "resolved_run.yaml"
    if not metadata_path.exists():
        return {}
    with open(metadata_path, encoding="utf-8") as stream:
        return yaml.safe_load(stream) or {}


def _episode_ranges(done: np.ndarray) -> list[tuple[int, int, slice]]:
    ranges: list[tuple[int, int, slice]] = []
    _, num_envs = done.shape
    for env_id in range(num_envs):
        start = 0
        episode_index = 0
        for end in np.flatnonzero(done[:, env_id]):
            ranges.append((env_id, episode_index, slice(start, int(end) + 1)))
            start = int(end) + 1
            episode_index += 1
    return ranges


def _parameter_value(
    rollout: Mapping[str, np.ndarray],
    key: str,
    episode_slice: slice,
    env_id: int,
    default: float,
) -> float:
    values = rollout.get(key)
    if values is None:
        return default
    episode_values = np.asarray(values[episode_slice, env_id], dtype=np.float64)
    episode_values = episode_values[np.isfinite(episode_values)]
    return float(np.median(episode_values)) if episode_values.size else default


def _episode_signal(
    rollout: Mapping[str, np.ndarray],
    key: str,
    episode_slice: slice,
    env_id: int,
) -> np.ndarray | None:
    values = rollout.get(key)
    if values is None:
        return None
    return np.asarray(values[episode_slice, env_id], dtype=np.float64)


def analyze_rollout(
    run_dir: str | Path,
    *,
    safe_position_min: float = 0.02,
    safe_position_max: float = 0.68,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Compute per-episode diagnostics and legacy sine aggregates."""
    run_dir = Path(run_dir).expanduser().resolve()
    rollout_path = run_dir / "rollout.npz"
    if not rollout_path.exists():
        raise FileNotFoundError(f"Missing rollout: {rollout_path}")

    metadata = _load_metadata(run_dir)
    step_dt = float(metadata.get("step_dt", 1.0 / 60.0))
    omega_limit = float(metadata.get("constraints", {}).get("omega_max", 0.5))
    configured_type_mapping = metadata.get(
        "trajectory_type_to_id",
        DEFAULT_TRAJECTORY_TYPE_TO_ID,
    )
    type_id_to_name = {
        int(type_id): str(name)
        for name, type_id in dict(configured_type_mapping).items()
    }
    with np.load(rollout_path, allow_pickle=False) as loaded:
        rollout = {key: loaded[key] for key in loaded.files}

    observations = np.asarray(rollout["observations"], dtype=np.float64)
    actions = np.asarray(rollout["actions"], dtype=np.float64)[..., 0]
    terminated = np.asarray(rollout["terminated"], dtype=bool)
    truncated = np.asarray(rollout["truncated"], dtype=bool)
    done = terminated | truncated
    if observations.ndim != 3 or done.shape != observations.shape[:2]:
        raise ValueError("Rollout observations and done arrays have incompatible shapes.")

    fields = tuple(str(name) for name in rollout.get("observation_fields", ()))
    field_indices = {name: index for index, name in enumerate(fields)}
    pb_index = field_indices.get("pb", 0)
    pg_index = field_indices.get("pg", 9)
    theta_index = field_indices.get("theta", 3)

    episode_rows: list[dict[str, Any]] = []
    for env_id, episode_index, episode_slice in _episode_ranges(done):
        pb = observations[episode_slice, env_id, pb_index]
        pg = observations[episode_slice, env_id, pg_index]
        theta = observations[episode_slice, env_id, theta_index]
        action = actions[episode_slice, env_id]
        error = pb - pg
        amplitude = _parameter_value(
            rollout,
            "step_trajectory_amplitude",
            episode_slice,
            env_id,
            float("nan"),
        )
        period_s = _parameter_value(
            rollout,
            "step_trajectory_period",
            episode_slice,
            env_id,
            float("nan"),
        )
        center = _parameter_value(
            rollout,
            "step_trajectory_center",
            episode_slice,
            env_id,
            float("nan"),
        )
        phase = _parameter_value(
            rollout,
            "step_trajectory_phase",
            episode_slice,
            env_id,
            float("nan"),
        )
        trajectory_type_value = _parameter_value(
            rollout,
            "step_trajectory_type_id",
            episode_slice,
            env_id,
            float("nan"),
        )
        trajectory_type_id = (
            int(round(trajectory_type_value))
            if math.isfinite(trajectory_type_value)
            else -1
        )
        trajectory_type = type_id_to_name.get(
            trajectory_type_id,
            "unknown" if trajectory_type_id < 0 else f"type_{trajectory_type_id}",
        )
        primary_signals = (pb, pg, theta, action)
        nonfinite = any(not np.isfinite(signal).all() for signal in primary_signals)
        mae = float(np.mean(np.abs(error))) if not nonfinite else float("inf")
        rmse = float(np.sqrt(np.mean(np.square(error)))) if not nonfinite else float("inf")
        maxe = float(np.max(np.abs(error))) if not nonfinite else float("inf")
        nrmse = (
            rmse / amplitude
            if math.isfinite(amplitude) and amplitude > 1.0e-9
            else float("nan")
        )
        frequency = (
            _frequency_response(pb, pg, period_s, step_dt)
            if not nonfinite
            else {
                "steady_nrmse": float("nan"),
                "amplitude_gain": float("nan"),
                "phase_error_deg": float("nan"),
                "phase_lag_s": float("nan"),
            }
        )
        steady_start = min(
            error.size,
            max(0, int(math.ceil(period_s / step_dt)))
            if math.isfinite(period_s) and period_s > 0.0
            else 0,
        )

        row: dict[str, Any] = {
            "env_id": env_id,
            "episode_index": episode_index,
            "trajectory_type_id": trajectory_type_id,
            "trajectory_type": trajectory_type,
            "samples": error.size,
            "duration_s": error.size * step_dt,
            "center": center,
            "amplitude": amplitude,
            "period_s": period_s,
            "phase": phase,
            "mae": mae,
            "rmse": rmse,
            "maxe": maxe,
            "nrmse": nrmse,
            "signed_error": float(np.mean(error)) if not nonfinite else float("inf"),
            "min_pb": float(np.min(pb)) if not nonfinite else float("nan"),
            "max_pb": float(np.max(pb)) if not nonfinite else float("nan"),
            "boundary_violation": bool(
                nonfinite
                or np.any(pb < safe_position_min)
                or np.any(pb > safe_position_max)
            ),
            "terminated": bool(terminated[episode_slice, env_id][-1]),
            "truncated": bool(truncated[episode_slice, env_id][-1]),
            "nonfinite": bool(nonfinite),
            "action_rms": (
                float(np.sqrt(np.mean(np.square(action)))) if not nonfinite else float("inf")
            ),
            **frequency,
        }

        theta_d = _episode_signal(rollout, "policy_theta_d", episode_slice, env_id)
        theta_star = _episode_signal(rollout, "policy_theta_star", episode_slice, env_id)
        velocity_command = _episode_signal(
            rollout,
            "policy_velocity_command",
            episode_slice,
            env_id,
        )
        vrz_index = field_indices.get("vrz")
        row["theta_tracking_rms"] = (
            float(np.sqrt(np.mean(np.square(theta - theta_d))))
            if theta_d is not None
            else float("nan")
        )
        row["theta_filter_rms"] = (
            float(np.sqrt(np.mean(np.square(theta_d - theta_star))))
            if theta_d is not None and theta_star is not None
            else float("nan")
        )
        row["velocity_tracking_rms"] = (
            float(
                np.sqrt(
                    np.mean(
                        np.square(
                            observations[episode_slice, env_id, vrz_index]
                            - velocity_command
                        )
                    )
                )
            )
            if velocity_command is not None and vrz_index is not None
            else float("nan")
        )

        for field in SATURATION_FIELDS:
            signal = _episode_signal(rollout, field, episode_slice, env_id)
            if signal is None and field == "policy_omega_command_saturated":
                omega_d = _episode_signal(rollout, "policy_omega_d", episode_slice, env_id)
                signal = (
                    np.abs(omega_d) >= max(omega_limit - 1.0e-6, 0.0)
                    if omega_d is not None
                    else np.zeros(error.size, dtype=bool)
                )
            if signal is None:
                signal = np.zeros(error.size, dtype=bool)
            signal = np.asarray(signal, dtype=bool)
            row[f"{field}_rate"] = _safe_rate(signal)
            row[f"steady_{field}_rate"] = _safe_rate(signal[steady_start:])
            tail_start = min(signal.size, int(math.floor(0.75 * signal.size)))
            row[f"tail_{field}_rate"] = _safe_rate(signal[tail_start:])

        episode_rows.append(row)

    aggregate = aggregate_episode_metrics(episode_rows)
    aggregate.update(
        {
            "run_dir": str(run_dir),
            "step_dt": step_dt,
            "safe_position_min": safe_position_min,
            "safe_position_max": safe_position_max,
        }
    )
    return episode_rows, aggregate


def aggregate_mixed_episode_metrics(
    episode_rows: list[Mapping[str, Any]],
    *,
    required_types: Iterable[str],
    min_episodes_per_type: int,
    full_saturation_rate: float = 0.05,
    tail_saturation_rate: float = 0.01,
) -> dict[str, Any]:
    """Aggregate mixed-reference metrics with per-family coverage gates."""
    required_types = tuple(str(name) for name in required_types)
    aggregate = aggregate_episode_metrics(episode_rows)
    aggregate["required_trajectory_types"] = list(required_types)
    aggregate["min_episodes_per_type"] = int(min_episodes_per_type)

    type_metrics: dict[str, dict[str, Any]] = {}
    coverage_shortfall = 0
    worst_type_p95_rmse = float("-inf")
    for trajectory_type in required_types:
        rows = [
            row
            for row in episode_rows
            if str(row.get("trajectory_type", "unknown")) == trajectory_type
        ]
        count = len(rows)
        coverage_shortfall += max(int(min_episodes_per_type) - count, 0)
        metrics = {
            "episode_count": count,
            "termination_count": sum(bool(row["terminated"]) for row in rows),
            "boundary_violation_count": sum(
                bool(row["boundary_violation"]) for row in rows
            ),
            "nonfinite_count": sum(bool(row["nonfinite"]) for row in rows),
            "mean_mae": _finite_mean(float(row["mae"]) for row in rows),
            "mean_rmse": _finite_mean(float(row["rmse"]) for row in rows),
            "p95_rmse": _finite_percentile(
                (float(row["rmse"]) for row in rows),
                95.0,
            ),
            "mean_maxe": _finite_mean(float(row["maxe"]) for row in rows),
            "p95_maxe": _finite_percentile(
                (float(row["maxe"]) for row in rows),
                95.0,
            ),
            "mean_abs_signed_error": _finite_mean(
                abs(float(row["signed_error"])) for row in rows
            ),
        }
        for field in SATURATION_FIELDS:
            metrics[f"mean_{field}_rate"] = _finite_mean(
                float(row[f"{field}_rate"]) for row in rows
            )
            metrics[f"mean_tail_{field}_rate"] = _finite_mean(
                float(
                    row.get(
                        f"tail_{field}_rate",
                        row.get(f"steady_{field}_rate", 0.0),
                    )
                )
                for row in rows
            )
        metrics["max_full_saturation_rate"] = max(
            (
                float(metrics[f"mean_{field}_rate"])
                for field in SATURATION_FIELDS
                if math.isfinite(float(metrics[f"mean_{field}_rate"]))
            ),
            default=float("nan"),
        )
        metrics["max_tail_saturation_rate"] = max(
            (
                float(metrics[f"mean_tail_{field}_rate"])
                for field in SATURATION_FIELDS
                if math.isfinite(float(metrics[f"mean_tail_{field}_rate"]))
            ),
            default=float("nan"),
        )
        type_metrics[trajectory_type] = metrics
        if math.isfinite(float(metrics["p95_rmse"])):
            worst_type_p95_rmse = max(
                worst_type_p95_rmse,
                float(metrics["p95_rmse"]),
            )

        prefix = trajectory_type
        for name, value in metrics.items():
            aggregate[f"{prefix}_{name}"] = value

    aggregate["type_metrics"] = type_metrics
    aggregate["coverage_shortfall"] = coverage_shortfall
    aggregate["worst_type_p95_rmse"] = (
        worst_type_p95_rmse
        if math.isfinite(worst_type_p95_rmse)
        else float("nan")
    )
    aggregate["max_tail_saturation_rate"] = max(
        (
            float(metrics["max_tail_saturation_rate"])
            for metrics in type_metrics.values()
            if math.isfinite(float(metrics["max_tail_saturation_rate"]))
        ),
        default=float("inf"),
    )
    aggregate["coverage_failure"] = coverage_shortfall > 0
    aggregate["hard_failure"] = bool(
        aggregate["coverage_failure"]
        or int(aggregate.get("termination_count", 0)) > 0
        or int(aggregate.get("boundary_violation_count", 0)) > 0
        or int(aggregate.get("nonfinite_count", 0)) > 0
        or float(aggregate.get("max_full_saturation_rate", float("inf")))
        > float(full_saturation_rate)
        or float(aggregate["max_tail_saturation_rate"])
        > float(tail_saturation_rate)
    )
    return aggregate


def mixed_tuning_rank_key(metrics: Mapping[str, Any]) -> tuple[float, ...]:
    """Rank mixed-reference candidates by feasibility and absolute errors."""

    def finite_or_infinity(name: str) -> float:
        value = float(metrics.get(name, float("nan")))
        return value if math.isfinite(value) else float("inf")

    failure_count = (
        max(int(metrics.get("termination_count", 0)), 0)
        + max(int(metrics.get("boundary_violation_count", 0)), 0)
        + max(int(metrics.get("nonfinite_count", 0)), 0)
    )
    full_saturation_excess = max(
        finite_or_infinity("max_full_saturation_rate") - 0.05,
        0.0,
    )
    tail_saturation_excess = max(
        finite_or_infinity("max_tail_saturation_rate") - 0.01,
        0.0,
    )

    return (
        float(bool(metrics.get("hard_failure", True))),
        float(max(int(metrics.get("coverage_shortfall", 0)), 0)),
        float(failure_count),
        full_saturation_excess,
        tail_saturation_excess,
        finite_or_infinity("worst_type_p95_rmse"),
        finite_or_infinity("mean_rmse"),
        finite_or_infinity("p95_maxe"),
        finite_or_infinity("max_full_saturation_rate"),
        finite_or_infinity("max_tail_saturation_rate"),
        finite_or_infinity("mean_action_rms"),
        finite_or_infinity("bandwidth_cost"),
    )


def aggregate_episode_metrics(episode_rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate episode-level metrics into leaderboard fields."""
    if not episode_rows:
        return {
            "episode_count": 0,
            "mean_mae": float("nan"),
            "mean_rmse": float("nan"),
            "p95_rmse": float("nan"),
            "p95_nrmse": float("nan"),
            "p95_maxe": float("nan"),
            "hard_failure": True,
        }

    aggregate: dict[str, Any] = {
        "episode_count": len(episode_rows),
        "termination_count": sum(bool(row["terminated"]) for row in episode_rows),
        "boundary_violation_count": sum(
            bool(row["boundary_violation"]) for row in episode_rows
        ),
        "nonfinite_count": sum(bool(row["nonfinite"]) for row in episode_rows),
        "mean_mae": _finite_mean(float(row["mae"]) for row in episode_rows),
        "mean_rmse": _finite_mean(float(row["rmse"]) for row in episode_rows),
        "p95_rmse": _finite_percentile(
            (float(row["rmse"]) for row in episode_rows),
            95.0,
        ),
        "mean_maxe": _finite_mean(float(row["maxe"]) for row in episode_rows),
        "p95_maxe": _finite_percentile(
            (float(row["maxe"]) for row in episode_rows),
            95.0,
        ),
        "mean_nrmse": _finite_mean(float(row["nrmse"]) for row in episode_rows),
        "p95_nrmse": _finite_percentile(
            (float(row["nrmse"]) for row in episode_rows),
            95.0,
        ),
        "mean_abs_signed_error": _finite_mean(
            abs(float(row["signed_error"])) for row in episode_rows
        ),
        "bias_over_1cm_rate": _finite_mean(
            float(abs(float(row["signed_error"])) > 0.01) for row in episode_rows
        ),
        "mean_action_rms": _finite_mean(float(row["action_rms"]) for row in episode_rows),
        "mean_theta_filter_rms": _finite_mean(
            float(row["theta_filter_rms"]) for row in episode_rows
        ),
        "mean_theta_tracking_rms": _finite_mean(
            float(row["theta_tracking_rms"]) for row in episode_rows
        ),
        "mean_velocity_tracking_rms": _finite_mean(
            float(row["velocity_tracking_rms"]) for row in episode_rows
        ),
        "mean_steady_nrmse": _finite_mean(
            float(row["steady_nrmse"]) for row in episode_rows
        ),
        "p95_steady_nrmse": _finite_percentile(
            (float(row["steady_nrmse"]) for row in episode_rows),
            95.0,
        ),
        "mean_abs_phase_error_deg": _finite_mean(
            abs(float(row["phase_error_deg"])) for row in episode_rows
        ),
        "p95_abs_phase_error_deg": _finite_percentile(
            (abs(float(row["phase_error_deg"])) for row in episode_rows),
            95.0,
        ),
        "mean_abs_gain_error": _finite_mean(
            abs(float(row["amplitude_gain"]) - 1.0) for row in episode_rows
        ),
        "p95_abs_gain_error": _finite_percentile(
            (abs(float(row["amplitude_gain"]) - 1.0) for row in episode_rows),
            95.0,
        ),
        "min_amplitude_gain": min(
            (
                float(row["amplitude_gain"])
                for row in episode_rows
                if math.isfinite(float(row["amplitude_gain"]))
            ),
            default=float("nan"),
        ),
        "max_amplitude_gain": max(
            (
                float(row["amplitude_gain"])
                for row in episode_rows
                if math.isfinite(float(row["amplitude_gain"]))
            ),
            default=float("nan"),
        ),
    }
    for field in SATURATION_FIELDS:
        aggregate[f"mean_{field}_rate"] = _finite_mean(
            float(row[f"{field}_rate"]) for row in episode_rows
        )
        aggregate[f"mean_steady_{field}_rate"] = _finite_mean(
            float(row[f"steady_{field}_rate"]) for row in episode_rows
        )

    full_saturation_rates = [
        float(aggregate[f"mean_{field}_rate"]) for field in SATURATION_FIELDS
    ]
    steady_saturation_rates = [
        float(aggregate[f"mean_steady_{field}_rate"]) for field in SATURATION_FIELDS
    ]
    aggregate["max_full_saturation_rate"] = max(full_saturation_rates, default=0.0)
    aggregate["max_steady_saturation_rate"] = max(
        steady_saturation_rates,
        default=0.0,
    )
    aggregate["hard_failure"] = bool(
        aggregate["termination_count"] > 0
        or aggregate["boundary_violation_count"] > 0
        or aggregate["nonfinite_count"] > 0
        or aggregate["max_full_saturation_rate"] > 0.05
        or aggregate["max_steady_saturation_rate"] > 0.01
    )
    return aggregate


def tuning_rank_key(metrics: Mapping[str, Any]) -> tuple[float, ...]:
    """Return the lexicographic candidate ranking defined by the tuning plan."""
    def finite_or_infinity(name: str) -> float:
        value = float(metrics.get(name, float("nan")))
        return value if math.isfinite(value) else float("inf")

    return (
        float(bool(metrics.get("hard_failure", True))),
        finite_or_infinity("p95_nrmse"),
        finite_or_infinity("mean_rmse"),
        finite_or_infinity("p95_abs_phase_error_deg"),
        finite_or_infinity("mean_abs_gain_error"),
        finite_or_infinity("mean_action_rms"),
        finite_or_infinity("bandwidth_cost"),
    )


def evaluate_acceptance(
    selected: Mapping[str, Any],
    baseline: Mapping[str, Any],
    stress: Mapping[str, Any],
    thresholds: Mapping[str, float],
    *,
    feedforward_rmse_improvement: float,
    feedforward_phase_improvement_deg: float,
) -> dict[str, Any]:
    """Evaluate final and stress metrics against explicit acceptance thresholds."""
    baseline_rmse = float(baseline.get("mean_rmse", float("nan")))
    selected_rmse = float(selected.get("mean_rmse", float("nan")))
    rmse_improvement = (
        1.0 - selected_rmse / baseline_rmse
        if math.isfinite(baseline_rmse) and baseline_rmse > 0.0
        else float("nan")
    )
    checks = {
        "episode_count": int(selected.get("episode_count", 0))
        >= int(thresholds["minimum_episodes"]),
        "no_nonfinite": int(selected.get("nonfinite_count", 0)) == 0,
        "no_termination": int(selected.get("termination_count", 0)) == 0,
        "no_boundary_violation": int(selected.get("boundary_violation_count", 0)) == 0,
        "mean_mae": float(selected.get("mean_mae", float("inf")))
        <= thresholds["mean_mae"],
        "mean_rmse": selected_rmse <= thresholds["mean_rmse"],
        "p95_rmse": float(selected.get("p95_rmse", float("inf")))
        <= thresholds["p95_rmse"],
        "p95_nrmse": float(selected.get("p95_nrmse", float("inf")))
        <= thresholds["p95_nrmse"],
        "p95_maxe": float(selected.get("p95_maxe", float("inf")))
        <= thresholds["p95_maxe"],
        "rmse_improvement": rmse_improvement >= thresholds["rmse_improvement"],
        "full_saturation": float(selected.get("max_full_saturation_rate", float("inf")))
        <= thresholds["full_saturation_rate"],
        "steady_saturation": float(
            selected.get("max_steady_saturation_rate", float("inf"))
        )
        <= thresholds["steady_saturation_rate"],
        "stress_gain_min": float(stress.get("min_amplitude_gain", float("-inf")))
        >= thresholds["stress_gain_min"],
        "stress_gain_max": float(stress.get("max_amplitude_gain", float("inf")))
        <= thresholds["stress_gain_max"],
        "stress_phase": float(stress.get("p95_abs_phase_error_deg", float("inf")))
        <= thresholds["stress_phase_error_deg"],
        "stress_steady_nrmse": float(stress.get("p95_steady_nrmse", float("inf")))
        <= thresholds["stress_steady_nrmse"],
        "feedforward_ablation": (
            feedforward_rmse_improvement
            >= thresholds["feedforward_rmse_improvement"]
            or feedforward_phase_improvement_deg
            >= thresholds["feedforward_phase_improvement_deg"]
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "rmse_improvement": rmse_improvement,
        "feedforward_rmse_improvement": feedforward_rmse_improvement,
        "feedforward_phase_improvement_deg": feedforward_phase_improvement_deg,
    }
