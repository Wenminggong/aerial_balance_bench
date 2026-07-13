#!/usr/bin/env python3
"""Plot selected real-world acceleration-control trials from CSV logs."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trial_dirs", nargs="+", type=Path, help="Trial directories containing samples.csv.")
    parser.add_argument("--output", type=Path, required=True, help="Output image path.")
    parser.add_argument("--dpi", type=int, default=300, help="Output image resolution.")
    parser.add_argument(
        "--plot-human-vertical-velocity",
        action="store_true",
        help="Add a fourth subplot showing the measured human vertical velocity.",
    )
    return parser.parse_args()


def load_trial(trial_dir: Path, *, include_human_vertical_velocity: bool = False) -> dict[str, np.ndarray]:
    csv_path = trial_dir / "samples.csv"
    with csv_path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"No samples found in {csv_path}")

    keys = ["t_trial", "obs_pb", "obs_theta", "actual_arz_up", "goal_pos"]
    if include_human_vertical_velocity:
        keys.append("human_vel_z")
    missing = [key for key in keys if key not in rows[0]]
    if missing:
        raise ValueError(f"Missing columns in {csv_path}: {', '.join(missing)}")
    return {key: np.asarray([float(row[key]) for row in rows], dtype=np.float64) for key in keys}


def main() -> None:
    args = parse_args()
    trials = [
        load_trial(path, include_human_vertical_velocity=args.plot_human_vertical_velocity)
        for path in args.trial_dirs
    ]
    goals = np.concatenate([trial["goal_pos"] for trial in trials])
    finite_goals = goals[np.isfinite(goals)]
    if finite_goals.size == 0:
        raise ValueError("No finite goal position found in the selected trials.")
    goal = float(np.median(finite_goals))

    colors = plt.get_cmap("tab10").colors
    series = [
        ("obs_pb", "Ball Position (m)"),
        ("obs_theta", r"Beam Angle $\theta$ (rad)"),
        ("actual_arz_up", r"Actual Vertical Acceleration (m/s$^2$)"),
    ]
    if args.plot_human_vertical_velocity:
        series.append(("human_vel_z", "Human Vertical Velocity (m/s)"))

    fig_height = 3.5 * len(series)
    fig, axes = plt.subplots(len(series), 1, figsize=(10, fig_height), constrained_layout=True)

    for index, trial in enumerate(trials):
        time = trial["t_trial"] - trial["t_trial"][0]
        for axis, (key, _) in zip(axes, series):
            axis.plot(time, trial[key], color=colors[index % len(colors)], linewidth=2.0, label=f"Trial {index + 1}")

    axes[0].axhline(goal, color="black", linestyle="--", linewidth=1.8, label=f"Target ({goal:.2f} m)")
    for axis, (_, ylabel) in zip(axes, series):
        axis.set_xlabel("Time (s)")
        axis.set_ylabel(ylabel)
        axis.grid(True, which="major", linestyle="--", alpha=0.35)
        axis.tick_params(axis="both", which="both", direction="out")
        axis.legend(loc="best", frameon=True, ncol=2)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
