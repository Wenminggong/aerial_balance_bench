"""Plot the best and worst NFFB sine-tracking episodes from a saved rollout."""

from __future__ import annotations

import argparse
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
DEFAULT_OUTPUT_NAME = "nffb_sine_best_worst_episodes.png"
DEFAULT_STEP_DT = 1.0 / 60.0


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Visualize the lowest- and highest-RMSE episodes in an NFFB rollout."
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
        help=f"Output image path (default: RUN_DIR/{DEFAULT_OUTPUT_NAME}).",
    )
    parser.add_argument("--dpi", type=int, default=300, help="Output image resolution.")
    return parser.parse_args()


def _observation_index(rollout: np.lib.npyio.NpzFile, field: str) -> int:
    fields = [str(value) for value in rollout["observation_fields"]]
    try:
        return fields.index(field)
    except ValueError as exc:
        raise KeyError(f"Observation field {field!r} is missing from rollout.npz.") from exc


def _step_dt(run_dir: Path) -> float:
    resolved_path = run_dir / "resolved_run.yaml"
    if not resolved_path.exists():
        return DEFAULT_STEP_DT
    with resolved_path.open(encoding="utf-8") as stream:
        resolved = yaml.safe_load(stream) or {}
    return float(resolved.get("step_dt", resolved.get("policy_step_dt", DEFAULT_STEP_DT)))


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


def visualize(run_dir: Path, output_path: Path, dpi: int) -> tuple[int, float, int, float]:
    """Select representative episodes, create the requested plot, and return their RMSEs."""
    rollout_path = run_dir / "rollout.npz"
    if not rollout_path.exists():
        raise FileNotFoundError(f"Rollout file does not exist: {rollout_path}")

    with np.load(rollout_path, allow_pickle=False) as rollout:
        observations = np.asarray(rollout["observations"])
        if observations.ndim != 3:
            raise ValueError(
                f"'observations' must have shape (steps, episodes, fields), got {observations.shape}."
            )
        num_steps, num_episodes, _ = observations.shape
        signal_shape = (num_steps, num_episodes)

        pb = observations[..., _observation_index(rollout, "pb")]
        pg = observations[..., _observation_index(rollout, "pg")]
        theta = observations[..., _observation_index(rollout, "theta")]
        vertical_velocity = observations[..., _observation_index(rollout, "vrz")]
        theta_desired = _validate_signal(rollout, "policy_theta_d", signal_shape)
        vertical_velocity_desired = _validate_signal(
            rollout,
            "policy_velocity_desired",
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

        time_s = np.arange(num_steps, dtype=np.float64) * _step_dt(run_dir)
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
            "NFFB Sine Tracking: Best and Worst Episodes\n"
            f"Best ID {best_id}: RMSE = {best_rmse:.5f} m    |    "
            f"Worst ID {worst_id}: RMSE = {worst_rmse:.5f} m",
            fontsize=15,
        )
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.945))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    return best_id, best_rmse, worst_id, worst_rmse


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    output_path = (
        args.output.expanduser().resolve()
        if args.output is not None
        else run_dir / DEFAULT_OUTPUT_NAME
    )
    best_id, best_rmse, worst_id, worst_rmse = visualize(run_dir, output_path, args.dpi)
    print(f"Best episode ID: {best_id} (RMSE={best_rmse:.8f} m)")
    print(f"Worst episode ID: {worst_id} (RMSE={worst_rmse:.8f} m)")
    print(f"Saved visualization to: {output_path}")


if __name__ == "__main__":
    main()
