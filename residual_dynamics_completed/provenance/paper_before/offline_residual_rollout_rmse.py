#!/usr/bin/env python3
"""Recompute and plot the paper's held-out causal rollout comparison.

The script performs inference only.  It compares the augmented nominal dynamics
with the frozen profile-plus-friction residual on the same held-out windows used
by the residual-dynamics evaluation.  Each window is rolled out once to 60
control steps, and metrics are extracted at ten measured endpoints.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch


FIGURE_DIR = Path(__file__).resolve().parent
PAPER_DIR = FIGURE_DIR.parent
WORKSPACE_DIR = PAPER_DIR.parents[1]
AERIAL_REPO = WORKSPACE_DIR / "aerial_balance_bench"
if not AERIAL_REPO.is_dir():
    raise RuntimeError(f"Aerial-Balance-Bench repository not found: {AERIAL_REPO}")
sys.path.insert(0, str(AERIAL_REPO))

from residual_dynamics.dynamics import rollout_causal_dynamics  # noqa: E402
from residual_dynamics_v2.config import load_config  # noqa: E402
from residual_dynamics_v2.data import load_manifest, load_trials  # noqa: E402
from residual_dynamics_v2.evaluation import _refs_for_evaluation  # noqa: E402
from residual_dynamics_v2.profile_friction_probe import (  # noqa: E402
    DEFAULT_CONFIG,
    RESULT_PATH,
    ProbeSettings,
    ProfileFrictionResidual,
)
from residual_dynamics_v2.training import WindowStore  # noqa: E402


HORIZONS = (6, 12, 18, 24, 30, 36, 42, 48, 54, 60)
BATCH_SIZE = 128
EXPECTED_WINDOW_COUNT = 2028
CHECKPOINT_PATH = RESULT_PATH.parent / "checkpoints" / "profile_plus_friction.pt"
DETAIL_PATH = RESULT_PATH.parent / "detailed_rollout_metrics.json"
CSV_PATH = FIGURE_DIR / "offline_residual_rollout_rmse.csv"
PDF_PATH = FIGURE_DIR / "offline_residual_rollout_rmse.pdf"
PNG_PATH = FIGURE_DIR / "offline_residual_rollout_rmse.png"


def _empty_totals() -> dict[str, dict[int, dict[str, float]]]:
    return {
        model: {
            horizon: {"pb_sq": 0.0, "vb_sq": 0.0, "count": 0}
            for horizon in HORIZONS
        }
        for model in ("nominal", "structured")
    }


def recompute_metrics() -> tuple[list[dict[str, float]], dict[str, object]]:
    config = load_config(DEFAULT_CONFIG)
    manifest = load_manifest(config)
    summary = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    if summary["dataset_hash"] != manifest["dataset_hash"]:
        raise RuntimeError("Frozen model and prepared dataset hashes differ.")

    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    if checkpoint["dataset_hash"] != manifest["dataset_hash"]:
        raise RuntimeError("Structured checkpoint and prepared dataset hashes differ.")
    structured = ProfileFrictionResidual(
        profile_enabled=checkpoint["profile_enabled"],
        friction_enabled=checkpoint["friction_enabled"],
        acceleration_gain=checkpoint["acceleration_gain"],
        gravity_mps2=checkpoint["gravity_mps2"],
        settings=ProbeSettings(**checkpoint["settings"]),
    )
    structured.load_state_dict(checkpoint["state_dict"], strict=True)
    structured.eval()
    models = {"nominal": None, "structured": structured}
    trials = load_trials(config, "test")
    store = WindowStore(trials, max(HORIZONS), config, manifest)
    references = _refs_for_evaluation(store)
    if len(references) != EXPECTED_WINDOW_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_WINDOW_COUNT} held-out windows, got {len(references)}."
        )

    totals = _empty_totals()
    for offset in range(0, len(references), BATCH_SIZE):
        part = references[offset : offset + BATCH_SIZE]
        batch = store.make_batch(part, torch.device("cpu"))
        with torch.no_grad():
            rollouts = {
                name: rollout_causal_dynamics(
                    model,
                    batch.causal_state[:, 0],
                    batch.commands,
                    batch.dt,
                    batch.constants,
                    batch.geometry,
                    batch.response,
                )
                for name, model in models.items()
            }
        for name, rollout in rollouts.items():
            for horizon in HORIZONS:
                pb_error = (
                    rollout.states[:, horizon, 0]
                    - batch.causal_state[:, horizon, 0]
                )
                vb_error = (
                    rollout.states[:, horizon, 1]
                    - batch.causal_state[:, horizon, 1]
                )
                bucket = totals[name][horizon]
                bucket["pb_sq"] += float(pb_error.square().sum())
                bucket["vb_sq"] += float(vb_error.square().sum())
                bucket["count"] += len(part)

    rows: list[dict[str, float]] = []
    for horizon in HORIZONS:
        values: dict[str, float] = {}
        for name in models:
            bucket = totals[name][horizon]
            count = int(bucket["count"])
            values[f"{name}_pb_rmse_m"] = math.sqrt(bucket["pb_sq"] / count)
            values[f"{name}_vb_rmse_mps"] = math.sqrt(bucket["vb_sq"] / count)
        rows.append(
            {
                "horizon_steps": horizon,
                "horizon_seconds": horizon / 60.0,
                **values,
                "pb_rmse_reduction_pct": 100.0
                * (
                    1.0
                    - values["structured_pb_rmse_m"]
                    / values["nominal_pb_rmse_m"]
                ),
                "vb_rmse_reduction_pct": 100.0
                * (
                    1.0
                    - values["structured_vb_rmse_mps"]
                    / values["nominal_vb_rmse_mps"]
                ),
            }
        )
    return rows, summary


def verify_existing_endpoints(rows: list[dict[str, float]]) -> None:
    detailed = json.loads(DETAIL_PATH.read_text(encoding="utf-8"))
    existing = detailed["splits"]["test"]["overall"]["all"]
    by_horizon = {int(row["horizon_steps"]): row for row in rows}
    maximum_error = 0.0
    for horizon in (6, 12, 30, 60):
        for old_name, new_name in (
            ("nominal", "nominal"),
            ("profile_plus_friction", "structured"),
        ):
            prior = existing[f"h{horizon}"][old_name]
            current = by_horizon[horizon]
            maximum_error = max(
                maximum_error,
                abs(prior["pb_rmse_m"] - current[f"{new_name}_pb_rmse_m"]),
                abs(prior["vb_rmse_mps"] - current[f"{new_name}_vb_rmse_mps"]),
            )
    if maximum_error > 1.0e-12:
        raise RuntimeError(
            "Recomputed established horizons do not match the frozen evaluation: "
            f"maximum absolute difference={maximum_error:.3e}."
        )
    print(f"Endpoint parity check passed (maximum difference {maximum_error:.3e}).")


def save_csv(rows: list[dict[str, float]]) -> None:
    fieldnames = list(rows[0].keys())
    with CSV_PATH.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot(rows: list[dict[str, float]]) -> None:
    horizons = [int(row["horizon_steps"]) for row in rows]
    nominal_pb = [100.0 * row["nominal_pb_rmse_m"] for row in rows]
    structured_pb = [100.0 * row["structured_pb_rmse_m"] for row in rows]
    nominal_vb = [row["nominal_vb_rmse_mps"] for row in rows]
    structured_vb = [row["structured_vb_rmse_mps"] for row in rows]

    nominal_color = "#24557A"
    nominal_fill = "#8FB7DE"
    structured_color = "#934F0B"
    structured_fill = "#F3B46B"
    style = {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "hatch.linewidth": 1.2,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
    with plt.rc_context(style):
        fig, axes = plt.subplots(
            1,
            2,
            figsize=(7.2, 3.4),
            constrained_layout=True,
        )
        for axis, nominal, structured, ylabel in (
            (axes[0], nominal_pb, structured_pb, "Position RMSE (cm)"),
            (axes[1], nominal_vb, structured_vb, "Velocity RMSE (m/s)"),
        ):
            axis.plot(
                horizons,
                nominal,
                color=nominal_color,
                linewidth=2.2,
                linestyle="--",
                marker="o",
                markersize=7.0,
                markerfacecolor=nominal_fill,
                markeredgecolor=nominal_color,
                markeredgewidth=1.2,
                label="Nominal",
            )
            axis.plot(
                horizons,
                structured,
                color=structured_color,
                linewidth=2.2,
                linestyle="-",
                marker="s",
                markersize=6.8,
                markerfacecolor=structured_fill,
                markeredgecolor=structured_color,
                markeredgewidth=1.2,
                label="Structured residual",
            )
            axis.set_ylabel(ylabel, fontsize=16, fontweight="bold")
            axis.set_xticks((6, 18, 30, 42, 54))
            axis.set_xlim(4, 62)
            axis.set_ylim(bottom=0.0)
            axis.tick_params(axis="both", labelsize=14, width=1.2, length=4)
            for label in axis.get_xticklabels() + axis.get_yticklabels():
                label.set_fontweight("semibold")
            for spine in axis.spines.values():
                spine.set_linewidth(1.2)
            axis.grid(
                axis="y",
                color="#B8B8B8",
                linewidth=0.9,
                alpha=0.45,
            )
        fig.supxlabel(
            "Rollout horizon (60-Hz steps)",
            fontsize=16,
            fontweight="bold",
        )
        legend = axes[0].legend(
            fontsize=12,
            frameon=True,
            facecolor="white",
            framealpha=0.95,
            edgecolor="#777777",
        )
        legend.get_frame().set_linewidth(1.0)
        fig.savefig(PDF_PATH, bbox_inches="tight")
        fig.savefig(PNG_PATH, dpi=400, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    rows, summary = recompute_metrics()
    verify_existing_endpoints(rows)
    save_csv(rows)
    plot(rows)

    test_models = summary["splits"]["test"]["models"]
    point_nominal = test_models["nominal"]["point"]
    point_structured = test_models["profile_plus_friction"]["point"]
    print(
        "Point acceleration RMSE: "
        f"nominal={point_nominal['rmse']:.10f}, "
        f"structured={point_structured['rmse']:.10f}, "
        f"count={point_nominal['count']}"
    )
    print(f"Saved {CSV_PATH}")
    print(f"Saved {PDF_PATH}")
    print(f"Saved {PNG_PATH}")


if __name__ == "__main__":
    main()
