"""Plot representative NFFB or CPID unified-tracking episodes."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import warnings

import numpy as np
import yaml

MPLCONFIGDIR = Path(os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib"))
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
warnings.filterwarnings("ignore", message="Unable to import Axes3D.*", category=UserWarning)
import matplotlib.pyplot as plt


DEFAULT_RUN_DIR = Path("logs/nffb/unified_tracking/nffb_unified_sine_test")
DEFAULT_STEP_DT = 1.0 / 60.0


@dataclass(frozen=True)
class ControllerSignals:
    """Controller-specific desired-signal fields."""

    controller_name: str
    theta_desired_key: str
    vertical_velocity_desired_key: str


@dataclass(frozen=True)
class VisualizationResult:
    """Representative-episode selection and generated output."""

    controller_name: str
    theta_desired_key: str
    vertical_velocity_desired_key: str
    best_id: int
    best_rmse: float
    worst_id: int
    worst_rmse: float
    output_path: Path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Visualize the lowest- and highest-RMSE episodes in an NFFB or CPID rollout."
    )
    parser.add_argument(
        "run_dir",
        nargs="?",
        type=Path,
        default=DEFAULT_RUN_DIR,
        help="Run directory containing rollout.npz (default: %(default)s).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output image path (default: RUN_DIR/CONTROLLER_REFERENCE_best_worst_episodes.png).",
    )
    parser.add_argument("--dpi", type=int, default=300, help="Output image resolution.")
    return parser.parse_args()


def _observation_index(rollout: np.lib.npyio.NpzFile, field: str) -> int:
    fields = [str(value) for value in rollout["observation_fields"]]
    try:
        return fields.index(field)
    except ValueError as exc:
        raise KeyError(f"Observation field {field!r} is missing from rollout.npz.") from exc


def _load_resolved_run(run_dir: Path) -> dict:
    resolved_path = run_dir / "resolved_run.yaml"
    if not resolved_path.exists():
        return {}
    with resolved_path.open(encoding="utf-8") as stream:
        return yaml.safe_load(stream) or {}


def _step_dt(run_dir: Path, resolved_run: dict) -> float:
    for key in ("step_dt", "policy_step_dt"):
        value = resolved_run.get(key)
        if value is not None:
            return float(value)

    predictor_cfg = resolved_run.get("resolved_policy_config", {}).get("state_predictor", {})
    predictor_step_dt = predictor_cfg.get("step_dt")
    if predictor_step_dt is not None and predictor_step_dt != "auto":
        return float(predictor_step_dt)

    env_config_path = run_dir / "input_env_config.yaml"
    if env_config_path.exists():
        with env_config_path.open(encoding="utf-8") as stream:
            env_config = yaml.safe_load(stream) or {}
        sim_config = env_config.get("sim", {})
        sim_dt = float(sim_config.get("dt", 1.0 / 180.0))
        decimation = int(sim_config.get("decimation", env_config.get("decimation", 3)))
        return sim_dt * decimation

    return DEFAULT_STEP_DT


def _controller_signals(rollout: np.lib.npyio.NpzFile) -> ControllerSignals:
    keys = set(rollout.files)
    if "policy_theta_d" in keys:
        if "policy_velocity_desired" not in keys:
            raise KeyError(
                "Detected an NFFB rollout from 'policy_theta_d', but "
                "'policy_velocity_desired' is missing."
            )
        return ControllerSignals(
            controller_name="NFFB",
            theta_desired_key="policy_theta_d",
            vertical_velocity_desired_key="policy_velocity_desired",
        )

    if "policy_theta_ref" in keys:
        velocity_candidates = (
            "policy_predictor_command_z",
            "step_vrz_cmd",
            "step_command_z",
        )
        velocity_key = next((key for key in velocity_candidates if key in keys), None)
        if velocity_key is None:
            raise KeyError(
                "Detected a CPID rollout from 'policy_theta_ref', but no absolute "
                f"vertical-velocity command was found. Tried: {velocity_candidates}."
            )
        return ControllerSignals(
            controller_name="CPID",
            theta_desired_key="policy_theta_ref",
            vertical_velocity_desired_key=velocity_key,
        )

    raise ValueError(
        "Unsupported rollout controller. Expected NFFB field 'policy_theta_d' "
        "or CPID field 'policy_theta_ref'."
    )


def _reference_label(resolved_run: dict) -> str:
    trajectory_types = resolved_run.get("trajectory_types", [])
    if isinstance(trajectory_types, str):
        trajectory_types = [trajectory_types]
    if len(trajectory_types) == 1:
        return str(trajectory_types[0]).replace("_", " ").title()
    if len(trajectory_types) > 1:
        return "Mixed Reference"
    return "Unified Reference"


def _default_output_path(
    run_dir: Path,
    controller_name: str,
    reference_label: str,
) -> Path:
    controller_slug = controller_name.lower().replace(" ", "_")
    reference_slug = reference_label.lower().replace(" ", "_")
    return run_dir / f"{controller_slug}_{reference_slug}_best_worst_episodes.png"


def _validate_signal(rollout: np.lib.npyio.NpzFile, key: str, expected_shape: tuple[int, int]) -> np.ndarray:
    if key not in rollout.files:
        raise KeyError(f"Required rollout signal {key!r} is missing.")
    values = np.asarray(rollout[key])
    if values.shape != expected_shape:
        raise ValueError(f"Signal {key!r} must have shape {expected_shape}, got {values.shape}.")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"Signal {key!r} contains non-finite values.")
    return values


def _plot_pair(
    ax: plt.Axes,
    time_s: np.ndarray,
    desired: np.ndarray,
    actual: np.ndarray,
    episode_ids: tuple[int, int],
    colors: tuple[str, str],
    quantity: str,
) -> None:
    ranks = ("Best", "Worst")
    for rank, episode_id, color in zip(ranks, episode_ids, colors):
        ax.plot(
            time_s,
            desired[:, episode_id],
            color=color,
            linestyle="--",
            linewidth=2.0,
            label=f"{rank} (episode ID {episode_id}) — {quantity}",
        )
        ax.plot(
            time_s,
            actual[:, episode_id],
            color=color,
            linestyle="-",
            linewidth=1.8,
            label=f"{rank} (episode ID {episode_id}) — actual",
        )
    ax.grid(True, which="major", linewidth=0.7, alpha=0.35)
    ax.legend(loc="best", ncol=2, fontsize=9, framealpha=0.95)


def visualize(
    run_dir: Path,
    output_path: Path | None,
    dpi: int,
) -> VisualizationResult:
    """Select representative episodes and create the requested plot."""
    rollout_path = run_dir / "rollout.npz"
    if not rollout_path.exists():
        raise FileNotFoundError(f"Rollout file does not exist: {rollout_path}")

    resolved_run = _load_resolved_run(run_dir)
    reference_label = _reference_label(resolved_run)
    with np.load(rollout_path, allow_pickle=False) as rollout:
        observations = np.asarray(rollout["observations"])
        if observations.ndim != 3:
            raise ValueError(
                f"'observations' must have shape (steps, episodes, fields), got {observations.shape}."
            )
        num_steps, num_episodes, _ = observations.shape
        signal_shape = (num_steps, num_episodes)
        controller_signals = _controller_signals(rollout)

        pb = observations[..., _observation_index(rollout, "pb")]
        pg = observations[..., _observation_index(rollout, "pg")]
        theta = observations[..., _observation_index(rollout, "theta")]
        vertical_velocity = observations[..., _observation_index(rollout, "vrz")]
        theta_desired = _validate_signal(
            rollout,
            controller_signals.theta_desired_key,
            signal_shape,
        )
        vertical_velocity_desired = _validate_signal(
            rollout,
            controller_signals.vertical_velocity_desired_key,
            signal_shape,
        )

        # Match UnifiedTrackingEvaluator: compute each episode's RMSE from the
        # post-step ball and reference positions accumulated by the evaluator.
        error_pb = _validate_signal(rollout, "step_pb", signal_shape)
        error_pg = _validate_signal(rollout, "step_pg", signal_shape)
        episode_rmse = np.sqrt(
            np.mean(
                np.square(error_pb.astype(np.float64) - error_pg.astype(np.float64)),
                axis=0,
            )
        )
        best_id = int(np.argmin(episode_rmse))
        worst_id = int(np.argmax(episode_rmse))
        best_rmse = float(episode_rmse[best_id])
        worst_rmse = float(episode_rmse[worst_id])

        time_s = np.arange(num_steps, dtype=np.float64) * _step_dt(run_dir, resolved_run)
        episode_ids = (best_id, worst_id)
        colors = ("tab:blue", "tab:orange")

        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
        _plot_pair(
            axes[0],
            time_s,
            pg,
            pb,
            episode_ids,
            colors,
            "reference",
        )
        axes[0].set_title("(a) Ball Position Tracking")
        axes[0].set_ylabel("Ball position (m)")

        _plot_pair(
            axes[1],
            time_s,
            theta_desired,
            theta,
            episode_ids,
            colors,
            r"desired $\theta$",
        )
        axes[1].set_title(r"(b) Beam Angle Tracking")
        axes[1].set_ylabel(r"Beam angle $\theta$ (rad)")

        _plot_pair(
            axes[2],
            time_s,
            vertical_velocity_desired,
            vertical_velocity,
            episode_ids,
            colors,
            r"desired $v_z$",
        )
        axes[2].set_title("(c) Drone Vertical Velocity Tracking")
        axes[2].set_ylabel(r"Vertical velocity $v_z$ (m/s)")
        axes[2].set_xlabel("Time (s)")
        axes[2].set_xlim(time_s[0], time_s[-1])

        fig.suptitle(
            f"{controller_signals.controller_name} {reference_label} Tracking: "
            "Best and Worst Episodes\n"
            f"Best ID {best_id}: RMSE = {best_rmse:.5f} m    |    "
            f"Worst ID {worst_id}: RMSE = {worst_rmse:.5f} m",
            fontsize=15,
        )
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.945))
        if output_path is None:
            output_path = _default_output_path(
                run_dir,
                controller_signals.controller_name,
                reference_label,
            )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    return VisualizationResult(
        controller_name=controller_signals.controller_name,
        theta_desired_key=controller_signals.theta_desired_key,
        vertical_velocity_desired_key=controller_signals.vertical_velocity_desired_key,
        best_id=best_id,
        best_rmse=best_rmse,
        worst_id=worst_id,
        worst_rmse=worst_rmse,
        output_path=output_path,
    )


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    output_path = (
        args.output.expanduser().resolve()
        if args.output is not None
        else None
    )
    result = visualize(run_dir, output_path, args.dpi)
    print(f"Detected controller: {result.controller_name}")
    print(f"Desired theta signal: {result.theta_desired_key}")
    print(f"Desired vertical-velocity signal: {result.vertical_velocity_desired_key}")
    print(f"Best episode ID: {result.best_id} (RMSE={result.best_rmse:.8f} m)")
    print(f"Worst episode ID: {result.worst_id} (RMSE={result.worst_rmse:.8f} m)")
    print(f"Saved visualization to: {result.output_path}")


if __name__ == "__main__":
    main()
