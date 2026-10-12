#!/usr/bin/env python3
"""Fit per-trajectory pure delays from a simulated vertical-velocity rollout."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import numpy as np
import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# Reuse the trajectory/model representations, fractional-delay interpolation,
# prediction metrics, and CSV writer used for real-world response identification.
from fit_real_velocity_response import (
    ModelFit,
    Trajectory,
    delayed_command,
    evaluate,
    write_csv,
)

import matplotlib.pyplot as plt


DEFAULT_SAMPLE_RATE_HZ = 60.0
COMMAND_FIELD = "step_command_z"
RESPONSE_FIELD = "observations[..., vrz]"


@dataclass
class SimulationTrajectory:
    """One episode segment extracted from a parallel simulation rollout."""

    env_id: int
    episode_index: int
    start_step: int
    end_step: int
    data: Trajectory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit an independent pure delay y(t)=u(t-delay) to every parallel "
            "vertical-velocity trajectory in a simulation rollout."
        )
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--run-dir",
        type=Path,
        help="Run directory containing rollout.npz and, normally, resolved_run.yaml.",
    )
    input_group.add_argument("--rollout", type=Path, help="Path to rollout.npz.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: <run-dir>/sim_velocity_delay_fit).",
    )
    parser.add_argument(
        "--step-dt",
        type=float,
        default=None,
        help=(
            "Control-step duration in seconds. By default it is read from "
            "resolved_run.yaml, with a 1/60 s fallback."
        ),
    )
    parser.add_argument(
        "--max-delay",
        type=float,
        default=0.8,
        help="Upper bound of the fitted pure delay in seconds.",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=30,
        help="Minimum aligned samples required for an episode segment.",
    )
    parser.add_argument(
        "--min-command-std",
        type=float,
        default=1.0e-6,
        help="Skip episode segments whose command standard deviation is below this value.",
    )
    parser.add_argument("--bins", type=int, default=30, help="Histogram bin count.")
    parser.add_argument("--dpi", type=int, default=180, help="Distribution-plot DPI.")
    return parser.parse_args()


def resolve_rollout_path(run_dir: Path | None, rollout: Path | None) -> Path:
    """Resolve and validate the input rollout path."""
    path = (
        rollout.expanduser().resolve()
        if rollout is not None
        else run_dir.expanduser().resolve() / "rollout.npz"
    )
    if not path.is_file():
        raise FileNotFoundError(f"Rollout file not found: {path}")
    return path


def resolve_step_dt(rollout_path: Path, override: float | None) -> tuple[float, str]:
    """Resolve the control-step duration from CLI, metadata, or the 60 Hz default."""
    if override is not None:
        if not math.isfinite(override) or override <= 0.0:
            raise ValueError("--step-dt must be finite and positive.")
        return float(override), "command_line"

    resolved_path = rollout_path.parent / "resolved_run.yaml"
    if resolved_path.is_file():
        with resolved_path.open("r", encoding="utf-8") as stream:
            metadata = yaml.safe_load(stream) or {}
        value = metadata.get("control_step_dt")
        if value is not None:
            step_dt = float(value)
            if not math.isfinite(step_dt) or step_dt <= 0.0:
                raise ValueError(
                    f"control_step_dt in {resolved_path} must be finite and positive."
                )
            return step_dt, str(resolved_path)

    return 1.0 / DEFAULT_SAMPLE_RATE_HZ, "default_60_hz"


def _validate_rollout_array(
    name: str,
    value: np.ndarray,
    expected_shape: tuple[int, ...],
) -> None:
    if value.shape != expected_shape:
        raise ValueError(f"{name} must have shape {expected_shape}, got {value.shape}.")


def _append_segment(
    trajectories: list[SimulationTrajectory],
    *,
    rollout_path: Path,
    env_id: int,
    episode_index: int,
    start_step: int,
    stop_step: int,
    command: np.ndarray,
    response: np.ndarray,
    min_samples: int,
    min_command_std: float,
) -> bool:
    """Append one half-open command segment, returning whether it was usable."""
    command_segment = np.asarray(
        command[start_step:stop_step, env_id], dtype=np.float64
    )
    response_segment = np.asarray(
        response[start_step:stop_step, env_id], dtype=np.float64
    )
    if command_segment.size < min_samples:
        return False
    if not np.all(np.isfinite(command_segment)) or not np.all(np.isfinite(response_segment)):
        return False
    if float(np.std(command_segment)) < min_command_std:
        return False

    trajectories.append(
        SimulationTrajectory(
            env_id=env_id,
            episode_index=episode_index,
            start_step=start_step,
            end_step=stop_step - 1,
            data=Trajectory(
                goal=f"env_{env_id}",
                file_path=rollout_path,
                file_name=rollout_path.name,
                episode_index=episode_index,
                batch="simulation",
                command=command_segment,
                response=response_segment,
            ),
        )
    )
    return True


def load_simulation_trajectories(
    rollout_path: Path,
    *,
    min_samples: int,
    min_command_std: float,
) -> tuple[list[SimulationTrajectory], dict[str, int]]:
    """Load, align, and split all parallel command/velocity trajectories."""
    if min_samples < 3:
        raise ValueError("--min-samples must be at least 3.")
    if not math.isfinite(min_command_std) or min_command_std < 0.0:
        raise ValueError("--min-command-std must be finite and non-negative.")

    with np.load(rollout_path, allow_pickle=False) as rollout:
        required = {"observations", "observation_fields", COMMAND_FIELD}
        missing = sorted(required.difference(rollout.files))
        if missing:
            raise KeyError(f"Missing required rollout fields: {', '.join(missing)}")

        observations = np.asarray(rollout["observations"])
        command = np.asarray(rollout[COMMAND_FIELD])
        observation_fields = [str(value) for value in rollout["observation_fields"]]
        if observations.ndim != 3:
            raise ValueError(
                "observations must have shape (steps, environments, fields), "
                f"got {observations.shape}."
            )
        if "vrz" not in observation_fields:
            raise KeyError("Observation field 'vrz' is missing from rollout.npz.")

        num_steps, num_envs, num_fields = observations.shape
        if len(observation_fields) != num_fields:
            raise ValueError(
                "observation_fields length does not match the last observations dimension."
            )
        _validate_rollout_array(COMMAND_FIELD, command, (num_steps, num_envs))
        if num_steps < 2:
            raise ValueError("At least two rollout steps are required for response alignment.")

        response = np.asarray(
            observations[1:, :, observation_fields.index("vrz")], dtype=np.float64
        )
        aligned_command = np.asarray(command[:-1], dtype=np.float64)

        done = np.zeros((num_steps, num_envs), dtype=bool)
        for name in ("terminated", "truncated"):
            if name in rollout.files:
                values = np.asarray(rollout[name], dtype=bool)
                _validate_rollout_array(name, values, (num_steps, num_envs))
                done |= values

    trajectories: list[SimulationTrajectory] = []
    candidate_count = 0
    skipped_count = 0
    # A terminal transition has no matching post-step observation in this rollout:
    # observations[t + 1] is either absent or belongs to the reset episode. Exclude
    # it, then resume at command step t + 1.
    for env_id in range(num_envs):
        episode_index = 0
        start_step = 0
        for terminal_step in np.flatnonzero(done[:, env_id]):
            stop_step = min(int(terminal_step), num_steps - 1)
            candidate_count += 1
            if not _append_segment(
                trajectories,
                rollout_path=rollout_path,
                env_id=env_id,
                episode_index=episode_index,
                start_step=start_step,
                stop_step=stop_step,
                command=aligned_command,
                response=response,
                min_samples=min_samples,
                min_command_std=min_command_std,
            ):
                skipped_count += 1
            episode_index += 1
            start_step = int(terminal_step) + 1

        if start_step < num_steps - 1:
            candidate_count += 1
            if not _append_segment(
                trajectories,
                rollout_path=rollout_path,
                env_id=env_id,
                episode_index=episode_index,
                start_step=start_step,
                stop_step=num_steps - 1,
                command=aligned_command,
                response=response,
                min_samples=min_samples,
                min_command_std=min_command_std,
            ):
                skipped_count += 1

    return trajectories, {
        "rollout_steps": num_steps,
        "num_envs": num_envs,
        "candidate_trajectory_count": candidate_count,
        "skipped_trajectory_count": skipped_count,
    }


def fit_trajectory_delays(
    trajectories: list[SimulationTrajectory],
    *,
    step_dt: float,
    max_delay_s: float,
) -> list[dict[str, Any]]:
    """Fit one pure-delay model to each independent trajectory."""
    if not math.isfinite(max_delay_s) or max_delay_s <= 0.0:
        raise ValueError("--max-delay must be finite and positive.")

    rows: list[dict[str, Any]] = []
    boundary_tolerance_s = max(1.0e-6, 0.01 * step_dt)
    for item in trajectories:
        fit = fit_pure_delay(
            item.data,
            step_dt=step_dt,
            max_delay_s=max_delay_s,
        )
        rows.append(
            {
                "env_id": item.env_id,
                "episode_index": item.episode_index,
                "start_step": item.start_step,
                "end_step": item.end_step,
                "sample_count": item.data.response.size,
                "delay_s": fit.delay_s,
                "delay_ms": 1000.0 * fit.delay_s,
                "fractional_delay_samples": fit.delay_s / step_dt,
                "rmse_mps": fit.rmse_mps,
                "mae_mps": fit.mae_mps,
                "r2": fit.r2,
                "at_lower_bound": fit.delay_s <= boundary_tolerance_s,
                "at_upper_bound": fit.delay_s >= max_delay_s - boundary_tolerance_s,
            }
        )
    return rows


def fit_pure_delay(
    trajectory: Trajectory,
    *,
    step_dt: float,
    max_delay_s: float,
) -> ModelFit:
    """Return the globally optimal fractionally interpolated pure delay.

    Within each one-sample delay interval, ``delayed_command`` is affine in the
    fractional delay. The squared prediction error is therefore a convex
    quadratic whose constrained minimizer can be computed analytically. Scanning
    all intervals avoids the local-minimum ambiguity of a single bounded scalar
    search while retaining the existing real-data script's delay semantics.
    """
    if not math.isfinite(step_dt) or step_dt <= 0.0:
        raise ValueError("step_dt must be finite and positive.")
    if not math.isfinite(max_delay_s) or max_delay_s <= 0.0:
        raise ValueError("max_delay_s must be finite and positive.")

    command = trajectory.command
    response = trajectory.response
    interval_count = max(1, int(math.ceil(max_delay_s / step_dt)))
    best_delay_s = 0.0
    best_mse = float("inf")

    for interval_index in range(interval_count):
        lower_s = interval_index * step_dt
        upper_s = min((interval_index + 1) * step_dt, max_delay_s)
        lower_prediction = delayed_command(command, lower_s, step_dt)
        upper_prediction = delayed_command(command, lower_s + step_dt, step_dt)
        direction = upper_prediction - lower_prediction
        denominator = float(np.dot(direction, direction))
        max_fraction = (upper_s - lower_s) / step_dt
        if denominator > 0.0:
            fraction = float(
                np.clip(
                    np.dot(direction, response - lower_prediction) / denominator,
                    0.0,
                    max_fraction,
                )
            )
        else:
            fraction = 0.0
        prediction = lower_prediction + fraction * direction
        mse = float(np.mean(np.square(response - prediction)))
        delay_s = lower_s + fraction * step_dt
        if mse < best_mse:
            best_mse = mse
            best_delay_s = delay_s

    provisional = ModelFit(
        model="unit_delay",
        delay_s=best_delay_s,
        tau_s=None,
        gain=1.0,
        bias_mps=0.0,
        rmse_mps=0.0,
        mae_mps=0.0,
        r2=0.0,
    )
    return ModelFit(
        model="unit_delay",
        delay_s=best_delay_s,
        tau_s=None,
        gain=1.0,
        bias_mps=0.0,
        **evaluate([trajectory], provisional, step_dt),
    )


def summarize_delay_fits(
    rows: list[dict[str, Any]],
    *,
    rollout_path: Path,
    step_dt: float,
    step_dt_source: str,
    max_delay_s: float,
    load_info: dict[str, int],
) -> dict[str, Any]:
    """Build serializable distribution and fit-quality statistics."""
    if not rows:
        raise RuntimeError("No usable simulation trajectories were found.")

    delays_s = np.asarray([row["delay_s"] for row in rows], dtype=np.float64)
    rmse = np.asarray([row["rmse_mps"] for row in rows], dtype=np.float64)
    r2 = np.asarray([row["r2"] for row in rows], dtype=np.float64)
    finite_r2 = r2[np.isfinite(r2)]
    delay_stats_s = {
        "mean": float(np.mean(delays_s)),
        "std": float(np.std(delays_s)),
        "lower": float(np.min(delays_s)),
        "upper": float(np.max(delays_s)),
        "median": float(np.median(delays_s)),
        "q25": float(np.quantile(delays_s, 0.25)),
        "q75": float(np.quantile(delays_s, 0.75)),
    }
    return {
        "rollout_path": str(rollout_path),
        "model": "pure_delay",
        "model_equation": "v_hat[k] = u(t_k - delay_s)",
        "command_field": COMMAND_FIELD,
        "response_field": RESPONSE_FIELD,
        "alignment": (
            "command[t] is paired with observations[t + 1, vrz]; terminal "
            "transitions are excluded"
        ),
        "step_dt_s": step_dt,
        "sample_rate_hz": 1.0 / step_dt,
        "step_dt_source": step_dt_source,
        "optimization_delay_bounds_s": [0.0, max_delay_s],
        **load_info,
        "fitted_trajectory_count": len(rows),
        "delay_s": delay_stats_s,
        "delay_ms": {key: 1000.0 * value for key, value in delay_stats_s.items()},
        "fit_quality": {
            "mean_rmse_mps": float(np.mean(rmse)),
            "std_rmse_mps": float(np.std(rmse)),
            "mean_r2": float(np.mean(finite_r2)) if finite_r2.size else None,
            "lower_bound_fit_count": sum(bool(row["at_lower_bound"]) for row in rows),
            "upper_bound_fit_count": sum(bool(row["at_upper_bound"]) for row in rows),
        },
    }


def plot_delay_distribution(
    rows: list[dict[str, Any]],
    output_path: Path,
    *,
    bins: int,
    dpi: int,
) -> None:
    """Plot a histogram and empirical CDF of fitted trajectory delays."""
    if bins <= 0:
        raise ValueError("--bins must be positive.")
    if dpi <= 0:
        raise ValueError("--dpi must be positive.")

    delays_ms = 1000.0 * np.asarray([row["delay_s"] for row in rows], dtype=np.float64)
    mean_ms = float(np.mean(delays_ms))
    std_ms = float(np.std(delays_ms))
    lower_ms = float(np.min(delays_ms))
    upper_ms = float(np.max(delays_ms))

    figure, axes = plt.subplots(1, 2, figsize=(12.0, 4.6), constrained_layout=True)
    histogram_axis, cdf_axis = axes
    histogram_axis.hist(
        delays_ms,
        bins=min(bins, max(1, delays_ms.size)),
        color="#4c78a8",
        edgecolor="white",
        alpha=0.9,
    )
    histogram_axis.axvline(mean_ms, color="#d62728", linewidth=1.8, label=f"mean = {mean_ms:.2f} ms")
    histogram_axis.axvline(
        mean_ms - std_ms,
        color="#ff7f0e",
        linewidth=1.2,
        linestyle="--",
        label=f"mean +/- std = {std_ms:.2f} ms",
    )
    histogram_axis.axvline(
        mean_ms + std_ms,
        color="#ff7f0e",
        linewidth=1.2,
        linestyle="--",
    )
    histogram_axis.set_xlabel("Fitted delay (ms)")
    histogram_axis.set_ylabel("Trajectory count")
    histogram_axis.set_title("Histogram")
    histogram_axis.grid(axis="y", alpha=0.25)
    histogram_axis.legend(fontsize=9)

    sorted_delays = np.sort(delays_ms)
    cumulative = np.arange(1, sorted_delays.size + 1, dtype=np.float64) / sorted_delays.size
    cdf_axis.step(sorted_delays, cumulative, where="post", color="#59a14f", linewidth=1.8)
    cdf_axis.axvline(lower_ms, color="0.4", linestyle=":", label=f"lower = {lower_ms:.2f} ms")
    cdf_axis.axvline(upper_ms, color="black", linestyle=":", label=f"upper = {upper_ms:.2f} ms")
    cdf_axis.set_xlabel("Fitted delay (ms)")
    cdf_axis.set_ylabel("Empirical cumulative probability")
    cdf_axis.set_ylim(0.0, 1.02)
    cdf_axis.set_title("Empirical CDF")
    cdf_axis.grid(alpha=0.25)
    cdf_axis.legend(fontsize=9)

    figure.suptitle(
        f"Simulation vertical-velocity pure-delay fits (n={delays_ms.size})",
        fontsize=14,
    )
    figure.savefig(output_path, dpi=dpi)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    rollout_path = resolve_rollout_path(args.run_dir, args.rollout)
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else rollout_path.parent / "sim_velocity_delay_fit"
    )
    step_dt, step_dt_source = resolve_step_dt(rollout_path, args.step_dt)
    trajectories, load_info = load_simulation_trajectories(
        rollout_path,
        min_samples=args.min_samples,
        min_command_std=args.min_command_std,
    )
    if not trajectories:
        raise RuntimeError(
            "No usable trajectories remained after episode alignment and filtering."
        )

    rows = fit_trajectory_delays(
        trajectories,
        step_dt=step_dt,
        max_delay_s=args.max_delay,
    )
    summary = summarize_delay_fits(
        rows,
        rollout_path=rollout_path,
        step_dt=step_dt,
        step_dt_source=step_dt_source,
        max_delay_s=args.max_delay,
        load_info=load_info,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "trajectory_delay_fits.csv", rows)
    (output_dir / "delay_fit_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    plot_delay_distribution(
        rows,
        output_dir / "delay_distribution.png",
        bins=args.bins,
        dpi=args.dpi,
    )

    stats_ms = summary["delay_ms"]
    print(
        "[INFO] Fitted pure delays for "
        f"{summary['fitted_trajectory_count']} trajectories: "
        f"mean={stats_ms['mean']:.3f} ms, std={stats_ms['std']:.3f} ms, "
        f"lower={stats_ms['lower']:.3f} ms, upper={stats_ms['upper']:.3f} ms."
    )
    print(
        f"[INFO] Mean RMSE={summary['fit_quality']['mean_rmse_mps']:.6f} m/s; "
        f"mean R2={summary['fit_quality']['mean_r2']}."
    )
    print(f"[INFO] Results written to {output_dir}")


if __name__ == "__main__":
    main()
