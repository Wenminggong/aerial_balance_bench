#!/usr/bin/env python3
"""Save 3:4 PNG plots of ball and human vertical positions by motion mode.

Run without arguments for structured_nmpc. Shading preserves each trial's
minimum-to-maximum ball position after 5 s within the displayed 30 s window.
Outputs are separate from the original five-panel visualizations.
"""

from __future__ import annotations

import argparse
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


def load_positions(path: Path) -> dict[str, np.ndarray]:
    """Load measured positions and the reference against trial timestamps."""
    with np.load(path, allow_pickle=False) as archive:
        data = {key: np.asarray(archive[key], dtype=float)
                for key in ("t_trial", "obs_pb", "obs_pg", "human_pos_z")}
    time = data["t_trial"]
    if time.ndim != 1 or time.size < 2 or np.any(np.diff(time) <= 0):
        raise ValueError(f"Expected strictly increasing trial timestamps: {path}")
    for key, values in data.items():
        if values.shape != time.shape or not np.isfinite(values).all():
            raise ValueError(f"Invalid or misaligned {key}: {path}")
    return data


def plot_mode(mode: Path, output_dir: Path) -> Path:
    """Overlay trials with fixed ball and time limits on a 6-by-8-inch canvas."""
    trials = [load_positions(path) for path in sorted(mode.glob("*/samples.npz"))]
    fig, axes = plt.subplots(2, 1, figsize=(6, 8), sharex=True, layout="constrained")
    colors = plt.get_cmap("tab10").colors
    for i, data in enumerate(trials):
        time = data["t_trial"]
        color = colors[i % len(colors)]
        label = f"Trial {i + 1}"
        for ax, key in zip(axes, ("obs_pb", "human_pos_z")):
            ax.plot(time, data[key], color=color, linewidth=1.4, label=label)
        selected = (time > 5.0) & (time <= 30.0)
        if np.any(selected):
            lower, upper = data["obs_pb"][selected].min(), data["obs_pb"][selected].max()
            axes[0].fill_between(
                [5.0, time[selected][-1]], lower, upper, color=color,
                alpha=0.18, linewidth=0, zorder=1, label=f"{label} range after 5 s",
            )
    goal = trials[0]["obs_pg"][0]
    if all(np.allclose(data["obs_pg"], goal, rtol=0, atol=1e-10) for data in trials):
        axes[0].axhline(goal, color="black", linestyle="--", linewidth=1.5,
                        label=f"Target ({goal:g} m)")
    else:
        for i, data in enumerate(trials):
            axes[0].plot(data["t_trial"], data["obs_pg"], color="black",
                         linestyle="--", linewidth=1.5, label=f"Trial {i + 1} target")
    axes[0].set_ylabel("Ball position (m)")
    axes[0].set_ylim(0, 0.7)
    axes[0].set_yticks(np.arange(0, 0.71, 0.1))
    axes[1].set_ylabel("Human vertical position (m)")
    axes[1].margins(y=0.12)
    axes[1].set_xlabel("Time (s)")
    for ax in axes:
        ax.set_xlim(0, 30)
        ax.set_xticks(np.arange(0, 31, 5))
        ax.grid(True, color="#dce1e7", linewidth=0.6)
        ax.spines[["top", "right", "bottom", "left"]].set_visible(True)
        ax.legend(loc="upper right", ncol=2, fontsize=8,
                  frameon=True, framealpha=0.9)
        ax.tick_params(axis="x", labelbottom=True)
    mode_title = mode.name.replace('_', ' ').capitalize()
    if mode.parent.name.endswith("_with_human_motion"):
        mode_title = f"With human motion | {mode_title}"
    fig.suptitle(f"Structured-residual NMPC\n{mode_title}",
                 fontsize=13, fontweight="bold")
    output = output_dir / f"{mode.name}.png"
    # Keep the full canvas: tight cropping would change the requested 3:4 ratio.
    fig.savefig(output, dpi=240, facecolor="white", bbox_inches=None)
    plt.close(fig)
    print(f"{mode.name}: {len(trials)} trials -> {output}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path,
                        help="Defaults to <data-dir>/visualizations_positions.")
    args = parser.parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    if not data_dir.is_dir():
        parser.error(f"Data directory does not exist: {data_dir}")
    modes = sorted(path for path in data_dir.iterdir()
                   if path.is_dir() and any(path.glob("*/samples.npz")))
    if not modes:
        parser.error(f"No <mode>/<trial>/samples.npz files under {data_dir}")
    output = args.output_dir.expanduser().resolve() if args.output_dir else data_dir / "visualizations_positions"
    output.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"font.size": 10, "savefig.bbox": None}):
        for mode in modes:
            plot_mode(mode, output)


if __name__ == "__main__":
    main()
