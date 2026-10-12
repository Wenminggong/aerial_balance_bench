"""Detailed, inference-only analysis for the profile/friction structure probe.

The probe summary stores the fitted one-dimensional functions on a dense grid and
the complete two-dimensional black-box coefficients. This module reconstructs the
frozen candidates from those values, verifies parity with the original aggregate
RMSE, and adds MAE, stratified rollout metrics, and trajectory visualizations. It
does not create an optimizer or perform any training.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import OrderedDict
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from residual_dynamics.dynamics import rollout_causal_dynamics
from residual_dynamics.model import load_portable_weights

from .config import Config, load_config
from .data import load_manifest, load_trials
from .evaluation import _refs_for_evaluation
from .model import load_checkpoint
from .profile_friction_probe import (
    DEFAULT_CONFIG,
    RESULT_PATH,
    ProbeSettings,
    ProfileFrictionResidual,
)
from .training import WindowReference, WindowStore, resolve_run_dir


OUTPUT_DIR = RESULT_PATH.parent
DETAIL_PATH = OUTPUT_DIR / "detailed_rollout_metrics.json"
FIGURE_DIR = OUTPUT_DIR / "figures"
HORIZONS = (6, 12, 30, 33, 60)

MODEL_LABELS = OrderedDict(
    (
        ("nominal", "Nominal"),
        ("rbf_pv", "RBF-PV"),
        ("profile_only", "Profile only"),
        ("friction_only", "Friction only"),
        ("profile_plus_friction", "Profile + friction"),
        ("profile_plus_blackbox", "Profile + black-box"),
        ("profile_plus_friction_plus_kappa", "Profile + friction + kappa"),
        ("cap0p25_mlp", "CAP0P25 MLP"),
    )
)

MODEL_COLORS = {
    "nominal": "#111111",
    "rbf_pv": "#7b61a8",
    "profile_only": "#2b6cb0",
    "friction_only": "#7a7a7a",
    "profile_plus_friction": "#2f855a",
    "profile_plus_blackbox": "#dd6b20",
    "profile_plus_friction_plus_kappa": "#00838f",
    "cap0p25_mlp": "#c53030",
}


def _load_summary(path: Path) -> dict[str, Any]:
    result = json.loads(path.read_text(encoding="utf-8"))
    required = {"profile_plus_blackbox", "profile_plus_friction_plus_kappa"}
    missing = required - set(result.get("parameters", {}))
    if missing:
        raise RuntimeError(f"The probe summary is missing candidates: {sorted(missing)}")
    return result


def _recover_coefficients(
    model: ProfileFrictionResidual,
    position_m: Sequence[float],
    sampled_values: Sequence[float],
) -> torch.Tensor:
    position = torch.as_tensor(position_m, dtype=torch.float64)
    basis = model.basis(position).detach().numpy()
    values = np.asarray(sampled_values, dtype=np.float64)
    coefficients, _, _, _ = np.linalg.lstsq(basis, values, rcond=None)
    reconstructed = basis @ coefficients
    if not np.allclose(reconstructed, values, atol=2.0e-10, rtol=0.0):
        raise RuntimeError(
            "Stored probe curve cannot be reconstructed from the configured basis: "
            f"max error={np.max(np.abs(reconstructed - values)):.3e}."
        )
    return torch.from_numpy(coefficients).to(torch.float64)


def reconstruct_probe_candidates(
    config: Config, summary: Mapping[str, Any]
) -> dict[str, ProfileFrictionResidual]:
    """Recover the frozen probe models from summary data without fitting."""

    settings = ProbeSettings(**summary["settings"])
    constants = load_trials(config, "train")[0].constants
    definitions = (
        ("profile_only", True, False, False, False),
        ("friction_only", False, True, False, False),
        ("profile_plus_friction", True, True, False, False),
        ("profile_plus_blackbox", True, False, True, False),
        ("profile_plus_friction_plus_kappa", True, True, False, True),
    )
    result: dict[str, ProfileFrictionResidual] = {}
    for name, profile, friction, blackbox, kappa in definitions:
        model = ProfileFrictionResidual(
            profile_enabled=profile,
            friction_enabled=friction,
            blackbox_enabled=blackbox,
            kappa_enabled=kappa,
            acceleration_gain=constants.acceleration_gain,
            gravity_mps2=constants.gravity_mps2,
            settings=settings,
        )
        stored = summary["parameters"][name]
        position = stored["position_m"]
        with torch.no_grad():
            model.delta_coefficients.copy_(
                _recover_coefficients(
                    model,
                    position,
                    np.deg2rad(stored["delta_theta_deg"]),
                )
            )
            model.coulomb_coefficients.copy_(
                _recover_coefficients(model, position, stored["coulomb_mps2"])
            )
            model.viscous_coefficients.copy_(
                _recover_coefficients(model, position, stored["viscous_sinv"])
            )
            model.blackbox_coefficients.copy_(
                torch.as_tensor(
                    stored["blackbox_coefficients_mps2"], dtype=torch.float64
                )
            )
            model.kappa_coefficients.copy_(
                torch.as_tensor(
                    stored.get("kappa_coefficients", [0.0] * 81),
                    dtype=torch.float64,
                )
            )
        result[name] = model.eval()
    return result


def load_frozen_models(
    config: Config, summary: Mapping[str, Any]
) -> OrderedDict[str, torch.nn.Module | None]:
    rbf, _ = load_checkpoint(resolve_run_dir(config) / "best_checkpoint.pt")
    mlp, _ = load_portable_weights(config.paths.baseline_weights)
    candidates = reconstruct_probe_candidates(config, summary)
    return OrderedDict(
        (
            ("nominal", None),
            ("rbf_pv", rbf.eval()),
            ("profile_only", candidates["profile_only"]),
            ("friction_only", candidates["friction_only"]),
            ("profile_plus_friction", candidates["profile_plus_friction"]),
            ("profile_plus_blackbox", candidates["profile_plus_blackbox"]),
            (
                "profile_plus_friction_plus_kappa",
                candidates["profile_plus_friction_plus_kappa"],
            ),
            ("cap0p25_mlp", mlp.eval()),
        )
    )


def _new_total() -> list[float]:
    # pb squared, vb squared, pb absolute, vb absolute, sample count
    return [0.0, 0.0, 0.0, 0.0, 0]


def _add_error(total: list[float], pb_error: float, vb_error: float) -> None:
    total[0] += pb_error * pb_error
    total[1] += vb_error * vb_error
    total[2] += abs(pb_error)
    total[3] += abs(vb_error)
    total[4] += 1


def _finalize_total(total: Sequence[float]) -> dict[str, float | int]:
    count = int(total[4])
    if count <= 0:
        raise ValueError("Cannot finalize an empty rollout metric bucket.")
    return {
        "count": count,
        "pb_rmse_m": math.sqrt(total[0] / count),
        "vb_rmse_mps": math.sqrt(total[1] / count),
        "pb_mae_m": total[2] / count,
        "vb_mae_mps": total[3] / count,
    }


def _position_labels(config: Config) -> list[str]:
    edges = config.legacy.data.position_bin_edges_m
    return [
        f"[{low:.1f},{high:.1f}{']' if index == len(edges) - 2 else ')'}"
        for index, (low, high) in enumerate(zip(edges[:-1], edges[1:]))
    ]


def _reference_type(group: str) -> str:
    return "constant" if group.startswith("constant") else group.split("_", 1)[0]


def _accumulate_split(
    models: Mapping[str, torch.nn.Module | None],
    store: WindowStore,
    refs: Sequence[WindowReference],
    config: Config,
) -> dict[str, Any]:
    categories: dict[str, dict[str, dict[str, dict[int, list[float]]]]] = {
        "overall": {},
        "by_position": {},
        "by_group": {},
        "by_reference_type": {},
        "by_session": {},
    }

    def ensure(category: str, label: str) -> None:
        categories[category].setdefault(
            label,
            {
                name: {horizon: _new_total() for horizon in HORIZONS}
                for name in models
            },
        )

    ensure("overall", "all")
    position_labels = _position_labels(config)
    for offset in range(0, len(refs), 128):
        part = refs[offset : offset + 128]
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
        for local, ref in enumerate(part):
            trial = store.trials[ref.trial_index]
            labels = {
                "overall": "all",
                "by_position": position_labels[ref.position_bin],
                "by_group": trial.group,
                "by_reference_type": _reference_type(trial.group),
                "by_session": trial.session,
            }
            for category, label in labels.items():
                ensure(category, label)
            for name, rollout in rollouts.items():
                for horizon in HORIZONS:
                    pb_error = float(
                        rollout.states[local, horizon, 0]
                        - batch.causal_state[local, horizon, 0]
                    )
                    vb_error = float(
                        rollout.states[local, horizon, 1]
                        - batch.causal_state[local, horizon, 1]
                    )
                    for category, label in labels.items():
                        _add_error(
                            categories[category][label][name][horizon],
                            pb_error,
                            vb_error,
                        )

    finalized: dict[str, Any] = {}
    for category, labels in categories.items():
        finalized[category] = {}
        for label, by_model in labels.items():
            finalized[category][label] = {}
            for horizon in HORIZONS:
                key = f"h{horizon}"
                values = {
                    name: _finalize_total(by_horizon[horizon])
                    for name, by_horizon in by_model.items()
                }
                nominal = values["nominal"]
                for value in values.values():
                    value["pb_mse_improvement_pct"] = 100.0 * (
                        1.0
                        - (value["pb_rmse_m"] / max(nominal["pb_rmse_m"], 1e-15))
                        ** 2
                    )
                    value["vb_mse_improvement_pct"] = 100.0 * (
                        1.0
                        - (
                            value["vb_rmse_mps"]
                            / max(nominal["vb_rmse_mps"], 1e-15)
                        )
                        ** 2
                    )
                    value["pb_mae_improvement_pct"] = 100.0 * (
                        1.0
                        - value["pb_mae_m"] / max(nominal["pb_mae_m"], 1e-15)
                    )
                    value["vb_mae_improvement_pct"] = 100.0 * (
                        1.0
                        - value["vb_mae_mps"]
                        / max(nominal["vb_mae_mps"], 1e-15)
                    )
                finalized[category][label][key] = values
    return finalized


def _verify_original_rmse(
    detailed: Mapping[str, Any], original: Mapping[str, Any]
) -> dict[str, float | bool]:
    maximum = 0.0
    for split in ("validation", "test", "stress"):
        for horizon in HORIZONS:
            current = detailed["splits"][split]["overall"]["all"][f"h{horizon}"]
            for name in MODEL_LABELS:
                expected = original["splits"][split]["models"][name]["rollout"][
                    f"h{horizon}"
                ]
                maximum = max(
                    maximum,
                    abs(current[name]["pb_rmse_m"] - expected["pb_rmse_m"]),
                    abs(current[name]["vb_rmse_mps"] - expected["vb_rmse_mps"]),
                )
    return {"max_absolute_error": maximum, "passed": maximum <= 2.0e-9}


def _plot_metric_curves(
    split: str, values: Mapping[str, Any], metric: str, output: Path
) -> None:
    suffix = "RMSE" if metric == "rmse" else "MAE"
    fields = (
        (f"pb_{metric}_m", f"Ball position {suffix} (m)"),
        (f"vb_{metric}_mps", f"Ball velocity {suffix} (m/s)"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.8), constrained_layout=True)
    for axis, (field, ylabel) in zip(axes, fields):
        for name, label in MODEL_LABELS.items():
            y = [values[f"h{h}"][name][field] for h in HORIZONS]
            axis.plot(
                HORIZONS,
                y,
                marker="o",
                linewidth=2.0 if name != "nominal" else 2.4,
                linestyle="--" if name == "nominal" else "-",
                color=MODEL_COLORS[name],
                label=label,
            )
        axis.set_xlabel("Causal rollout horizon (steps at 60 Hz)")
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.25)
        axis.set_xticks(HORIZONS)
    axes[0].legend(fontsize=8, ncol=2)
    fig.suptitle(f"{split.capitalize()} causal rollout {suffix}")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _plot_improvement_heatmap(
    split: str,
    category: str,
    labels: Sequence[str],
    values: Mapping[str, Any],
    output: Path,
) -> None:
    models = [name for name in MODEL_LABELS if name != "nominal"]
    matrices = []
    for field in ("pb_mse_improvement_pct", "vb_mse_improvement_pct"):
        matrices.append(
            np.asarray(
                [
                    [values[label]["h33"][name][field] for label in labels]
                    for name in models
                ],
                dtype=np.float64,
            )
        )
    limit = max(5.0, float(max(np.max(np.abs(matrix)) for matrix in matrices)))
    fig, axes = plt.subplots(1, 2, figsize=(max(11.0, 1.4 * len(labels) + 7), 5.5), constrained_layout=True)
    image = None
    for axis, matrix, state in zip(axes, matrices, ("position", "velocity")):
        image = axis.imshow(matrix, cmap="RdYlGn", vmin=-limit, vmax=limit, aspect="auto")
        axis.set_xticks(np.arange(len(labels)), labels=labels, rotation=35, ha="right")
        axis.set_yticks(
            np.arange(len(models)), labels=[MODEL_LABELS[name] for name in models]
        )
        axis.set_title(f"H33 {state} MSE improvement vs nominal (%)")
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                axis.text(
                    column,
                    row,
                    f"{matrix[row, column]:.1f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="black",
                )
    assert image is not None
    fig.colorbar(image, ax=axes, shrink=0.80, label="Improvement (%)")
    fig.suptitle(f"{split.capitalize()} H33 improvement by {category}")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _model_rollouts(
    models: Mapping[str, torch.nn.Module | None], batch: Any
) -> dict[str, Any]:
    with torch.no_grad():
        return {
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


def _percentiles(values: np.ndarray) -> dict[str, float]:
    return {
        "min": float(np.min(values)),
        "p01": float(np.percentile(values, 1)),
        "p50": float(np.percentile(values, 50)),
        "p99": float(np.percentile(values, 99)),
        "max": float(np.max(values)),
    }


def _kappa_diagnostics(
    model: ProfileFrictionResidual,
    config: Config,
    figure_dir: Path,
) -> dict[str, Any]:
    pb = torch.linspace(0.0, 0.7, 71, dtype=torch.float64)
    vb = torch.linspace(-0.8, 0.8, 81, dtype=torch.float64)
    grid_pb, grid_vb = torch.meshgrid(pb, vb, indexing="ij")
    grid_features = torch.zeros((grid_pb.numel(), 5), dtype=torch.float64)
    grid_features[:, 0] = grid_pb.reshape(-1)
    grid_features[:, 1] = grid_vb.reshape(-1)
    with torch.no_grad():
        grid_kappa = model.components(grid_features)["kappa"].numpy()
    coefficients = model.kappa_coefficients.detach().numpy()
    result: dict[str, Any] = {
        "configured_abs_limit": model.settings.kappa_limit,
        "coefficients": {
            **_percentiles(coefficients),
            "abs_limit_fraction": float(
                np.mean(np.abs(coefficients) >= model.settings.kappa_limit - 1.0e-12)
            ),
        },
        "outer_grid": {
            **_percentiles(grid_kappa),
            "abs_ge_0p095_fraction": float(np.mean(np.abs(grid_kappa) >= 0.095)),
        },
        "by_split": {},
    }
    for split in ("validation", "test", "stress"):
        ratios: list[np.ndarray] = []
        velocity_gradients: list[np.ndarray] = []
        direct_gain_changes: list[np.ndarray] = []
        observed_kappa: list[np.ndarray] = []
        for trial in load_trials(config, split):
            mask = trial.arrays["rbf_valid"].astype(bool)
            pb_values = torch.as_tensor(
                trial.arrays["pb"][mask], dtype=torch.float64
            )
            vb_values = torch.as_tensor(
                trial.arrays["vb"][mask], dtype=torch.float64
            ).requires_grad_(True)
            theta = torch.as_tensor(
                trial.arrays["theta"][mask], dtype=torch.float64
            ).requires_grad_(True)
            omega = torch.as_tensor(
                trial.arrays["omega"][mask], dtype=torch.float64
            )
            constants = trial.constants
            a_nom = constants.acceleration_gain * (
                (
                    pb_values
                    + constants.ball_position_offset_m
                    - constants.plank_length_m
                )
                * omega.square()
                - constants.gravity_mps2 * torch.sin(theta)
            )
            features = torch.stack((pb_values, vb_values, theta, omega, a_nom), -1)
            residual = model(features)
            total = a_nom + residual
            nominal_gradient = torch.autograd.grad(
                a_nom.sum(), theta, retain_graph=True
            )[0]
            total_gradient = torch.autograd.grad(
                total.sum(), theta, retain_graph=True
            )[0]
            velocity_gradient = torch.autograd.grad(residual.sum(), vb_values)[0]
            ratios.append((total_gradient / nominal_gradient).detach().numpy())
            velocity_gradients.append(velocity_gradient.detach().numpy())
            with torch.no_grad():
                observed_kappa.append(
                    model.components(features.detach())["kappa"].numpy()
                )
            independent = features.detach().requires_grad_(True)
            independent_residual = model(independent)
            direct = torch.autograd.grad(
                independent_residual.sum(), independent
            )[0][:, 4]
            direct_gain_changes.append(direct.detach().numpy())
        ratio = np.concatenate(ratios)
        rv = np.concatenate(velocity_gradients)
        direct = np.concatenate(direct_gain_changes)
        kappa = np.concatenate(observed_kappa)
        result["by_split"][split] = {
            "sample_count": int(kappa.size),
            "observed_kappa": {
                **_percentiles(kappa),
                "near_negative_limit_fraction": float(np.mean(kappa <= -0.095)),
                "near_abs_limit_fraction": float(np.mean(np.abs(kappa) >= 0.095)),
                "mean": float(np.mean(kappa)),
            },
            "direct_residual_gain_wrt_a_nom": _percentiles(direct),
            "total_theta_gain_ratio": {
                **_percentiles(ratio),
                "outside_0p9_1p1_fraction": float(
                    np.mean((ratio < 0.9) | (ratio > 1.1))
                ),
            },
            "residual_velocity_gradient_sinv": {
                **_percentiles(rv),
                "above_0p1_fraction": float(np.mean(rv > 0.1)),
            },
        }

    surface = grid_kappa.reshape(grid_pb.shape)
    fig, axis = plt.subplots(figsize=(8.0, 5.2), constrained_layout=True)
    image = axis.pcolormesh(
        pb.numpy(),
        vb.numpy(),
        surface.T,
        shading="auto",
        cmap="coolwarm",
        vmin=-model.settings.kappa_limit,
        vmax=model.settings.kappa_limit,
    )
    axis.set_xlabel("Ball position $p_b$ (m)")
    axis.set_ylabel("Ball velocity $v_b$ (m/s)")
    axis.set_title(r"Learned bounded gain $\kappa(p_b,v_b)$")
    fig.colorbar(image, ax=axis, label=r"$\kappa$")
    path = figure_dir / "profile_friction_kappa_surface.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    result["figure"] = path.name
    return result


def _representative_trial_plots(
    models: Mapping[str, torch.nn.Module | None],
    config: Config,
    manifest: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    trials = load_trials(config, "test")
    trial_index = next(
        index for index, trial in enumerate(trials) if trial.reference_type == "sine"
    )
    trial = trials[trial_index]
    store = WindowStore(trials, 60, config, manifest)
    trial_refs = [ref for ref in store.references if ref.trial_index == trial_index]
    if not trial_refs:
        raise RuntimeError(f"No H60 windows exist for representative trial {trial.trial_uid}.")

    rolling_refs = trial_refs[::5]
    rolling_time: list[float] = []
    rolling_actual_pb: list[float] = []
    rolling_actual_vb: list[float] = []
    rolling_predictions = {
        name: {"pb": [], "vb": []} for name in models
    }
    for offset in range(0, len(rolling_refs), 128):
        part = rolling_refs[offset : offset + 128]
        batch = store.make_batch(part, torch.device("cpu"))
        rollouts = _model_rollouts(models, batch)
        for local, ref in enumerate(part):
            endpoint = ref.start + 60
            rolling_time.append(float(trial.arrays["time_s"][endpoint]))
            rolling_actual_pb.append(float(batch.causal_state[local, 60, 0]))
            rolling_actual_vb.append(float(batch.causal_state[local, 60, 1]))
            for name, rollout in rollouts.items():
                rolling_predictions[name]["pb"].append(
                    float(rollout.states[local, 60, 0])
                )
                rolling_predictions[name]["vb"].append(
                    float(rollout.states[local, 60, 1])
                )

    time = np.asarray(rolling_time)
    time -= time[0]
    fig, axes = plt.subplots(2, 1, figsize=(13.0, 7.2), sharex=True, constrained_layout=True)
    axes[0].plot(time, rolling_actual_pb, color="black", linewidth=2.6, label="Observed")
    axes[1].plot(time, rolling_actual_vb, color="black", linewidth=2.6, label="Observed")
    for name, label in MODEL_LABELS.items():
        axes[0].plot(time, rolling_predictions[name]["pb"], color=MODEL_COLORS[name], linewidth=1.25, alpha=0.9, label=label)
        axes[1].plot(time, rolling_predictions[name]["vb"], color=MODEL_COLORS[name], linewidth=1.25, alpha=0.9, label=label)
    axes[0].set_ylabel("H60 predicted $p_b$ (m)")
    axes[1].set_ylabel("H60 predicted $v_b$ (m/s)")
    axes[1].set_xlabel("Endpoint time within plotted trial (s)")
    for axis in axes:
        axis.grid(True, alpha=0.25)
    axes[0].legend(fontsize=7, ncol=4)
    fig.suptitle(f"Rolling H60 prediction — {trial.trial_uid}")
    rolling_path = output_dir / "representative_trial_h60_rolling.png"
    fig.savefig(rolling_path, dpi=180)
    plt.close(fig)

    chosen_ref = trial_refs[len(trial_refs) // 2]
    batch = store.make_batch([chosen_ref], torch.device("cpu"))
    rollouts = _model_rollouts(models, batch)
    relative_time = np.concatenate(
        ([0.0], np.cumsum(batch.dt[0].detach().numpy()))
    )
    fig, axes = plt.subplots(2, 1, figsize=(10.0, 7.0), sharex=True, constrained_layout=True)
    axes[0].plot(relative_time, batch.causal_state[0, :, 0], color="black", linewidth=2.8, label="Observed")
    axes[1].plot(relative_time, batch.causal_state[0, :, 1], color="black", linewidth=2.8, label="Observed")
    for name, label in MODEL_LABELS.items():
        states = rollouts[name].states[0]
        axes[0].plot(relative_time, states[:, 0], color=MODEL_COLORS[name], linewidth=1.5, label=label)
        axes[1].plot(relative_time, states[:, 1], color=MODEL_COLORS[name], linewidth=1.5, label=label)
    axes[0].set_ylabel("$p_b$ (m)")
    axes[1].set_ylabel("$v_b$ (m/s)")
    axes[1].set_xlabel("Time from rollout initialization (s)")
    for axis in axes:
        axis.grid(True, alpha=0.25)
    axes[0].legend(fontsize=7, ncol=4)
    fig.suptitle(
        f"Median valid H60 window — {trial.trial_uid}\n"
        f"start={float(trial.arrays['time_s'][chosen_ref.start]):.3f} s"
    )
    window_path = output_dir / "representative_trial_window_h60.png"
    fig.savefig(window_path, dpi=180)
    plt.close(fig)
    return {
        "split": "test",
        "trial_uid": trial.trial_uid,
        "rolling_horizon_steps": 60,
        "rolling_stride_steps": 5,
        "rolling_window_count": len(rolling_refs),
        "single_window_start_index": chosen_ref.start,
        "single_window_start_time_s": float(trial.arrays["time_s"][chosen_ref.start]),
        "selection_rule": "first sine test trial; median valid H60 window",
        "figures": [rolling_path.name, window_path.name],
    }


def run(
    config: Config,
    summary_path: Path = RESULT_PATH,
    output_dir: Path = OUTPUT_DIR,
) -> dict[str, Any]:
    summary = _load_summary(summary_path)
    manifest = load_manifest(config)
    if summary["dataset_hash"] != manifest["dataset_hash"]:
        raise RuntimeError("Probe summary and prepared dataset hashes differ.")
    models = load_frozen_models(config, summary)
    result: dict[str, Any] = {
        "schema_version": 1,
        "purpose": "inference-only detailed comparison for profile/friction probe",
        "training_performed": False,
        "source_summary": str(summary_path),
        "dataset_hash": manifest["dataset_hash"],
        "config_hash": config.config_hash,
        "horizons_steps": list(HORIZONS),
        "model_labels": dict(MODEL_LABELS),
        "splits": {},
    }
    for split in ("validation", "test", "stress"):
        trials = load_trials(config, split)
        store = WindowStore(trials, max(HORIZONS), config, manifest)
        refs = _refs_for_evaluation(store)
        result["splits"][split] = {
            "evaluation_window_count": len(refs),
            **_accumulate_split(models, store, refs, config),
        }
    result["parity_with_original_summary"] = _verify_original_rmse(result, summary)
    if not result["parity_with_original_summary"]["passed"]:
        raise RuntimeError(
            "Frozen-model reconstruction did not reproduce the original summary: "
            f"{result['parity_with_original_summary']}"
        )

    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    result["model_diagnostics"] = {
        "profile_plus_friction_plus_kappa": _kappa_diagnostics(
            models["profile_plus_friction_plus_kappa"], config, figure_dir
        )
    }
    for split, split_result in result["splits"].items():
        overall = split_result["overall"]["all"]
        _plot_metric_curves(
            split, overall, "rmse", figure_dir / f"rollout_rmse_vs_horizon_{split}.png"
        )
        _plot_metric_curves(
            split, overall, "mae", figure_dir / f"rollout_mae_vs_horizon_{split}.png"
        )
        for category, title in (
            ("by_position", "position interval"),
            ("by_reference_type", "trajectory type"),
        ):
            values = split_result[category]
            if category == "by_position":
                labels = [label for label in _position_labels(config) if label in values]
            else:
                preferred = ("constant", "sine", "triangle", "trapezoid")
                labels = [label for label in preferred if label in values]
                labels.extend(sorted(set(values) - set(labels)))
            _plot_improvement_heatmap(
                split,
                title,
                labels,
                values,
                figure_dir / f"h33_mse_improvement_{category}_{split}.png",
            )
    result["representative_trial"] = _representative_trial_plots(
        models, config, manifest, figure_dir
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    detail_path = output_dir / DETAIL_PATH.name
    detail_path.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--summary", type=Path, default=RESULT_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    result = run(load_config(args.config), args.summary, args.output_dir)
    concise = {
        split: {
            name: values
            for name, values in split_result["overall"]["all"]["h33"].items()
        }
        for split, split_result in result["splits"].items()
    }
    print(json.dumps(concise, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
