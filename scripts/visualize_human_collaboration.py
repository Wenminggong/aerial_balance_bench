#!/usr/bin/env python3
"""Plot real-world trials by human motion mode from samples.npz logs.

Run from the repository root with no arguments for structured-residual NMPC.
Use --data-dir to plot another controller's human collaboration results.
Human velocity falls back to a local linear position fit when not recorded;
the plot and JSON manifest explicitly identify this offline estimate.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / (
    "logs/real_world/acados_nmpc_cpp/human_collaboration/structured_nmpc"
)


def estimate_velocity(time: np.ndarray, position: np.ndarray, window_s: float) -> np.ndarray:
    """Fit position against actual timestamps in a centered time window.

    Windows become one-sided at the endpoints. Missing samples are ignored;
    fewer than two finite positions produce NaN, never a fabricated zero.
    """
    velocity = np.full_like(position, np.nan)
    for i, center in enumerate(time):
        left, right = np.searchsorted(time, [center - window_s / 2, center + window_s / 2])
        valid = np.isfinite(position[left:right])
        if np.count_nonzero(valid) < 2:
            continue
        t = time[left:right][valid]
        p = position[left:right][valid]
        t = t - t.mean()
        velocity[i] = np.dot(t, p - p.mean()) / np.dot(t, t)
    return velocity


def load_trial(path: Path, window_s: float) -> tuple[dict[str, np.ndarray], dict]:
    """Read observation coordinates (vertical up-positive, angle in radians)."""
    with np.load(path, allow_pickle=False) as archive:
        keys = ("t_trial", "obs_pb", "obs_theta", "obs_vrz", "obs_pg")
        data = {key: np.asarray(archive[key], dtype=float) for key in keys}
        time = data["t_trial"]
        if time.ndim != 1 or time.size < 2 or not np.isfinite(time).all() or np.any(np.diff(time) <= 0):
            raise ValueError(f"Expected finite, strictly increasing trial timestamps: {path}")
        for key in keys[1:]:
            if data[key].shape != time.shape or not np.isfinite(data[key]).all():
                raise ValueError(f"Invalid or misaligned required field {key}: {path}")
        info = {"file": str(path), "samples": len(time), "time_start_s": float(time[0]),
                "time_end_s": float(time[-1]), "human_velocity_source": "unavailable"}
        for key in ("human_pos_z", "human_vel_z"):
            if key in archive:
                values = np.asarray(archive[key], dtype=float)
                if values.shape != time.shape:
                    raise ValueError(f"Misaligned field {key}: {path}")
                if np.isfinite(values).any():
                    data[key] = values
        if "human_vel_z" in data:
            if "human_estimate_valid" in archive:
                valid = np.asarray(archive["human_estimate_valid"], dtype=float)
                if valid.shape != time.shape:
                    raise ValueError(f"Misaligned human_estimate_valid: {path}")
                invalid = ~np.isfinite(valid) | (valid <= 0)
                data["human_vel_z"][invalid] = np.nan
                info["human_velocity_invalid_samples"] = int(invalid.sum())
            info["human_velocity_source"] = "recorded human_vel_z"
        elif "human_pos_z" in data:
            data["human_vel_z"] = estimate_velocity(time, data["human_pos_z"], window_s)
            info["human_velocity_source"] = "offline local linear fit of human_pos_z against t_trial"
            info["human_velocity_window_s"] = window_s
    return data, info


def plot_mode(mode: Path, output: Path, window_s: float) -> dict:
    """Overlay every trial in one motion folder, retaining its full duration."""
    paths = sorted(mode.glob("*/samples.npz"))
    trials = [load_trial(path, window_s) for path in paths]
    panels = [("obs_pb", "Ball position\n(m)"), ("obs_theta", "Beam angle\n(rad)"),
              ("obs_vrz", "Drone vertical velocity\n(m/s)")]
    if any("human_pos_z" in data or "human_vel_z" in data for data, _ in trials):
        panels += [("human_pos_z", "Human vertical position\n(m)"),
                   ("human_vel_z", "Human vertical velocity\n(m/s)")]
    fig, axes = plt.subplots(len(panels), 1, figsize=(12, 2.05 * len(panels) + 0.8),
                             sharex=True, layout="constrained")
    colors = plt.get_cmap("tab10").colors
    for i, (data, info) in enumerate(trials):
        label = f"Trial {i + 1}"
        info["label"] = label
        for ax, (key, ylabel) in zip(axes, panels):
            if key in data:
                ax.plot(data["t_trial"], data[key], color=colors[i % len(colors)],
                        linewidth=1.35, label=label)
            ax.set_ylabel(ylabel)
        after_5s = data["t_trial"] > 5.0
        if np.any(after_5s):
            lower = float(data["obs_pb"][after_5s].min())
            upper = float(data["obs_pb"][after_5s].max())
            axes[0].fill_between(
                [5.0, data["t_trial"][-1]], lower, upper,
                color=colors[i % len(colors)], alpha=0.18, linewidth=0, zorder=1,
                label=f"{label} range after 5 s",
            )
            info["ball_position_range_after_5s_m"] = {"min": lower, "max": upper}
    # A single dashed line suffices when every trial has the same constant goal.
    goal = trials[0][0]["obs_pg"][0]
    constant_goal = all(np.allclose(data["obs_pg"], goal, rtol=0, atol=1e-10) for data, _ in trials)
    end = max(data["t_trial"][-1] for data, _ in trials)
    if constant_goal:
        axes[0].axhline(goal, color="#343434", linestyle="--", linewidth=1.6,
                        label=f"Desired position ({goal:g} m)")
    else:
        for i, (data, _) in enumerate(trials):
            axes[0].plot(data["t_trial"], data["obs_pg"], linestyle="--",
                         color=colors[i % len(colors)], label=f"Trial {i + 1} desired")
    for ax in axes:
        ax.grid(True, color="#dce1e7", linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlim(0, end)
        ax.margins(y=0.16)
        if ax.lines:
            ax.legend(loc="upper right", ncol=3, fontsize=9, framealpha=0.9)
        else:
            ax.text(0.5, 0.5, "Not recorded", ha="center", transform=ax.transAxes)
    axes[0].set_ylim(0.0, 0.7)
    estimated = any(info["human_velocity_source"].startswith("offline") for _, info in trials)
    if estimated:
        axes[-1].set_title(f"Human velocity estimated from position: {window_s:g} s local linear fit",
                           loc="left", fontsize=10, color="#555555")
    axes[-1].set_xlabel("Time since trial start (s)")
    controller_title = "Structured-residual NMPC"
    if mode.parent.name.endswith("_with_human_motion"):
        controller_title += " with human motion"
    fig.suptitle(f"{controller_title} | {mode.name.replace('_', ' ').capitalize()}",
                 fontsize=15, fontweight="bold")
    files = []
    for suffix in ("png", "pdf"):
        path = output / f"{mode.name}.{suffix}"
        fig.savefig(path, dpi=220, facecolor="white")
        files.append(str(path))
    plt.close(fig)
    print(f"{mode.name}: {len(trials)} trials -> {files[0]}")
    return {"mode": mode.name, "figures": files, "trials": [info for _, info in trials]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, help="Defaults to <data-dir>/visualizations.")
    parser.add_argument("--human-velocity-window-s", type=float, default=0.2)
    args = parser.parse_args()
    if not np.isfinite(args.human_velocity_window_s) or args.human_velocity_window_s <= 0:
        parser.error("--human-velocity-window-s must be finite and positive")
    data_dir = args.data_dir.expanduser().resolve()
    modes = sorted(path for path in data_dir.iterdir() if path.is_dir() and any(path.glob("*/samples.npz")))
    if not modes:
        parser.error(f"No <mode>/<trial>/samples.npz files under {data_dir}")
    output = args.output_dir.expanduser().resolve() if args.output_dir else data_dir / "visualizations"
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 11, "axes.labelsize": 11, "pdf.fonttype": 42})
    results = [plot_mode(mode, output, args.human_velocity_window_s) for mode in modes]
    manifest = {"time_source": "t_trial (seconds since trial start)",
                "fields": {"ball_position_m": "obs_pb", "desired_position_m": "obs_pg",
                           "beam_angle_rad": "obs_theta", "drone_velocity_up_mps": "obs_vrz",
                           "human_position_up_m": "human_pos_z"},
                "modes": results}
    (output / "plot_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
