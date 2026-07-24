"""Visualize random references and the final in-episode preview window."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

MPLCONFIGDIR = Path(os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib"))
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
STEP_DT = 1.0 / 60.0
EPISODE_LENGTHS_S = (20.0, 10.0)
PREVIEW_FUTURE_STEPS = 30


def _load_random_reference_module():
    module_path = REPO_ROOT / "environments" / "tasks" / "random_reference_trajectories.py"
    spec = importlib.util.spec_from_file_location("random_reference_trajectories", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load random-reference module from {module_path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_config(config_path: Path) -> tuple[SimpleNamespace, int]:
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    return SimpleNamespace(**config["unified_tracking_task"]), int(config["env"]["seed"])


def _sample_reference(
    trajectory_type: str,
    config_path: Path,
    episode_length_s: float,
) -> dict[str, object]:
    module = _load_random_reference_module()
    cfg, seed = _load_config(config_path)
    episode_steps = math.ceil(episode_length_s / STEP_DT)
    reference_horizon_s = (
        episode_steps + PREVIEW_FUTURE_STEPS + 1
    ) * STEP_DT

    torch.manual_seed(seed)
    generator = module.RandomReferenceTrajectories(
        cfg,
        1,
        "cpu",
        0.0,
        0.70,
        reference_horizon_s=reference_horizon_s,
    )
    type_ids = torch.tensor([module.TRAJECTORY_TYPE_TO_ID[trajectory_type]])
    generator.sample_reset(torch.tensor([0]), type_ids)

    if trajectory_type == "random_b_spline":
        generated_horizon_s = generator.b_spline_duration_s
        sampled_parameters = {
            "control_positions": generator.b_spline_control_positions[0].tolist(),
            "knots": generator._b_spline_knots.tolist(),
        }
    else:
        generated_horizon_s = generator.ramp_dwell_duration_s
        sampled_parameters = {
            "ramp_start_times": generator.ramp_start_times[0].tolist(),
            "ramp_end_times": generator.ramp_end_times[0].tolist(),
            "dwell_end_times": generator.dwell_end_times[0].tolist(),
            "ramp_start_positions": generator.ramp_start_positions[0].tolist(),
            "ramp_end_positions": generator.ramp_end_positions[0].tolist(),
        }

    dense_samples = max(1201, int(math.ceil(generated_horizon_s / STEP_DT)) * 4 + 1)
    dense_times = torch.linspace(0.0, generated_horizon_s, dense_samples)
    dense_positions = generator.get_position(dense_times.unsqueeze(0), type_ids)[0]

    last_episode_step = episode_steps - 1
    preview_steps = last_episode_step + torch.arange(PREVIEW_FUTURE_STEPS + 1)
    preview_times = preview_steps.to(dtype=torch.float32) * STEP_DT
    preview_positions = generator.get_position(preview_times.unsqueeze(0), type_ids)[0]
    velocity_lookahead_time = (last_episode_step + PREVIEW_FUTURE_STEPS + 1) * STEP_DT
    velocity_lookahead_position = generator.get_position(
        torch.tensor([[velocity_lookahead_time]], dtype=torch.float32),
        type_ids,
    )[0, 0]

    return {
        "episode_length_s": episode_length_s,
        "episode_steps": episode_steps,
        "required_horizon_s": reference_horizon_s,
        "generated_horizon_s": generated_horizon_s,
        "last_episode_time_s": last_episode_step * STEP_DT,
        "dense_times": dense_times,
        "dense_positions": dense_positions,
        "preview_times": preview_times,
        "preview_positions": preview_positions,
        "velocity_lookahead_time": velocity_lookahead_time,
        "velocity_lookahead_position": velocity_lookahead_position,
        "sampled_parameters": sampled_parameters,
    }


def _plot_family(
    trajectory_type: str,
    config_path: Path,
    output_path: Path,
) -> list[dict[str, object]]:
    samples = [
        _sample_reference(trajectory_type, config_path, episode_length_s)
        for episode_length_s in EPISODE_LENGTHS_S
    ]
    maximum_horizon = max(float(sample["generated_horizon_s"]) for sample in samples)

    figure, axes = plt.subplots(
        2,
        1,
        figsize=(12.0, 8.5),
        sharex=True,
        sharey=True,
    )
    figure.subplots_adjust(left=0.08, right=0.985, bottom=0.08, top=0.79, hspace=0.26)
    display_name = {
        "random_b_spline": "Random B-spline",
        "random_ramp_dwell": "Random ramp-dwell",
    }[trajectory_type]

    for axis, sample in zip(axes, samples, strict=True):
        episode_length_s = float(sample["episode_length_s"])
        generated_horizon_s = float(sample["generated_horizon_s"])
        dense_times = sample["dense_times"]
        dense_positions = sample["dense_positions"]
        preview_times = sample["preview_times"]
        preview_positions = sample["preview_positions"]

        axis.plot(
            dense_times,
            dense_positions,
            color="#94a3b8",
            linewidth=1.8,
            label="Full generated reference",
            zorder=1,
        )
        episode_mask = dense_times <= episode_length_s
        axis.plot(
            dense_times[episode_mask],
            dense_positions[episode_mask],
            color="#2563eb",
            linewidth=2.4,
            label="Reference during episode",
            zorder=2,
        )
        axis.axvspan(
            float(preview_times[0]),
            float(preview_times[-1]),
            color="#f59e0b",
            alpha=0.13,
            label="Final-step preview interval",
            zorder=0,
        )
        axis.plot(
            preview_times,
            preview_positions,
            color="#ea580c",
            linewidth=2.2,
            marker="o",
            markersize=3.2,
            markevery=2,
            label="Preview samples (offsets 0...30)",
            zorder=4,
        )
        axis.scatter(
            [sample["velocity_lookahead_time"]],
            [sample["velocity_lookahead_position"]],
            color="#7c3aed",
            marker="D",
            s=34,
            label=r"Extra $p_g(t+dt)$ for final $v_g$",
            zorder=5,
        )
        axis.axvline(
            episode_length_s,
            color="#dc2626",
            linestyle="--",
            linewidth=1.7,
            label="Episode end",
            zorder=3,
        )
        axis.axvline(
            generated_horizon_s,
            color="#16a34a",
            linestyle=":",
            linewidth=1.7,
            label="Generated-reference horizon",
            zorder=3,
        )
        axis.set_title(
            f"Episode length = {episode_length_s:.0f} s  |  "
            f"generated horizon = {generated_horizon_s:.3f} s"
        )
        axis.set_ylim(0.0, 0.70)
        axis.set_ylabel(r"Reference position $p_g$ (m)")
        axis.grid(True, color="#cbd5e1", alpha=0.65, linewidth=0.7)
        axis.text(
            0.012,
            0.965,
            f"last episode observation: {sample['last_episode_time_s']:.3f} s\n"
            f"preview: {float(preview_times[0]):.3f}–{float(preview_times[-1]):.3f} s\n"
            f"preview change: {float(preview_positions[-1] - preview_positions[0]):+.4f} m",
            transform=axis.transAxes,
            va="top",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.88, "edgecolor": "#cbd5e1"},
        )

    axes[-1].set_xlim(0.0, maximum_horizon + 0.10)
    axes[-1].set_xlabel("Time (s)")
    handles, labels = axes[0].get_legend_handles_labels()
    unique_entries = dict(zip(labels, handles, strict=True))
    figure.legend(
        unique_entries.values(),
        unique_entries.keys(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.895),
        ncol=4,
        frameon=False,
        fontsize=9,
    )
    figure.suptitle(
        f"{display_name}: episode reference and final 30-step (0.5 s) preview\n"
        "seed = 666, control period = 1/60 s",
        y=0.985,
        fontsize=14,
        fontweight="bold",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return samples


def _json_summary(trajectory_type: str, samples: list[dict[str, object]]) -> dict[str, object]:
    return {
        "trajectory_type": trajectory_type,
        "seed": 666,
        "step_dt": STEP_DT,
        "preview_future_steps": PREVIEW_FUTURE_STEPS,
        "configurations": [
            {
                key: value
                for key, value in sample.items()
                if key
                in {
                    "episode_length_s",
                    "episode_steps",
                    "required_horizon_s",
                    "generated_horizon_s",
                    "last_episode_time_s",
                    "velocity_lookahead_time",
                    "sampled_parameters",
                }
            }
            for sample in samples
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "docs" / "figures",
    )
    args = parser.parse_args()

    jobs = (
        (
            "random_b_spline",
            REPO_ROOT / "environments" / "configs" / "unified_tracking_random_b_spline.yaml",
            args.output_dir / "random_b_spline_preview_comparison.png",
        ),
        (
            "random_ramp_dwell",
            REPO_ROOT / "environments" / "configs" / "unified_tracking_random_ramp_dwell.yaml",
            args.output_dir / "random_ramp_dwell_preview_comparison.png",
        ),
    )
    summaries = []
    for trajectory_type, config_path, output_path in jobs:
        samples = _plot_family(trajectory_type, config_path, output_path)
        summaries.append(_json_summary(trajectory_type, samples))
        print(f"Saved {output_path}")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
