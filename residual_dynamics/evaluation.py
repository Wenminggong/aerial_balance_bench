"""Held-out validation of nominal and learned residual ball dynamics."""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

os.environ.setdefault("MPLCONFIGDIR", "/tmp/aerial_balance_residual_mpl")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import AppConfig
from .data import (
    TrialData,
    load_manifest,
    load_processed_trials,
    nominal_acceleration,
    ood_distances,
    position_bin_indices,
    position_bin_labels,
)
from .dynamics import (
    causal_features_torch,
    causal_residual_torch,
    rollout_ball_dynamics,
    rollout_causal_dynamics,
)
from .model import ResidualMLP, load_checkpoint
from .training import WindowReference, WindowStore, resolve_run_dir


def evaluate_model(
    config: AppConfig,
    run_dir: str | Path | None = None,
    *,
    include_holdout: bool = False,
) -> dict[str, Any]:
    """Evaluate validation, and optionally the one-shot test/stress splits."""

    resolved_run_dir = resolve_run_dir(config, run_dir)
    checkpoint_path = resolved_run_dir / "best_checkpoint.pt"
    model, checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    model.eval()
    manifest = load_manifest(config)
    if checkpoint["dataset_hash"] != manifest["dataset_hash"]:
        raise RuntimeError("Checkpoint and processed dataset hashes do not match.")
    if checkpoint["config_hash"] != config.config_hash:
        raise RuntimeError(
            "Checkpoint and evaluation config hashes do not match; pass the same "
            "seed/Jacobian overrides used for training."
        )
    trials = load_processed_trials(config, split="validation")
    validation, point_arrays, example = _evaluate_split(
        model, trials, manifest, config, split="validation"
    )
    point = validation["point"]
    rollout = validation["rollout"]
    sensitivity = validation["alignment_sensitivity"]
    diagnostics = validation["diagnostics"]
    acceptance = validation["acceptance"]

    training_summary = _read_json(resolved_run_dir / "training_summary.json")
    summary: dict[str, Any] = {
        "schema_version": 3 if config.data.split_assignments else (
            2 if config.causal_rollout.enabled else 1
        ),
        "training_variant": (
            "causal_rollout" if config.causal_rollout.enabled else "teacher_forced"
        ),
        "run_name": training_summary["run_name"],
        "run_dir": training_summary["run_dir"],
        "results_dir": config.paths.results_dir.name,
        "config_hash": config.config_hash,
        "dataset_hash": manifest["dataset_hash"],
        "seed": config.training.seed,
        "split_counts": manifest["split_counts"],
        "validation_is_final_evaluation": not bool(config.data.split_assignments),
        "independent_test_set": bool(config.data.split_assignments),
        "training": training_summary,
        "point": point,
        "rollout": rollout,
        "alignment_sensitivity": sensitivity,
        "diagnostics": diagnostics,
        "acceptance": acceptance,
        "splits": {"validation": validation},
    }
    if include_holdout and config.data.split_assignments:
        for split in ("test", "stress"):
            holdout_trials = load_processed_trials(config, split=split)
            holdout, _, _ = _evaluate_split(
                model, holdout_trials, manifest, config, split=split
            )
            summary["splits"][split] = holdout
        summary["export_eligible"] = bool(
            all(
                summary["splits"][split]["acceptance"]["passed"]
                for split in ("validation", "test", "stress")
            )
        )
    else:
        summary["export_eligible"] = False
    config.paths.results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = config.paths.results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    _write_json(config.paths.results_dir / "summary.json", summary)
    _write_metrics_csv(config.paths.results_dir / "metrics.csv", summary)
    np.savez_compressed(
        resolved_run_dir / "validation_predictions.npz",
        **point_arrays,
        **example,
    )
    _plot_training_history(resolved_run_dir / "history.csv", figures_dir)
    _plot_point_residual(point_arrays, figures_dir)
    _plot_rollout_metrics(rollout, figures_dir)
    _plot_example_rollout(example, figures_dir)
    _write_json(
        resolved_run_dir / "evaluation_summary.json",
        summary,
    )
    return summary


def _evaluate_split(
    model: ResidualMLP,
    trials: Sequence[TrialData],
    manifest: dict[str, Any],
    config: AppConfig,
    *,
    split: str,
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, np.ndarray]]:
    point, point_arrays = _point_metrics(model, trials, manifest, config)
    rollout, example = _rollout_metrics(model, trials, manifest, config)
    diagnostics = (
        _causal_diagnostics(model, trials, manifest, config)
        if config.causal_rollout.enabled
        else _diagnostics(model, trials, manifest, config)
    )
    control_sensitivity = (
        _control_sensitivity(model, trials, manifest, config)
        if config.control_channel.enabled
        else {}
    )
    diagnostics["control_sensitivity"] = control_sensitivity
    acceptance = _acceptance(point, rollout, diagnostics, config, split=split)
    result = {
        "split": split,
        "trial_count": len(trials),
        "point": point,
        "rollout": rollout,
        "alignment_sensitivity": _alignment_sensitivity(model, trials),
        "diagnostics": diagnostics,
        "acceptance": acceptance,
    }
    return result, point_arrays, example


def _point_metrics(
    model: ResidualMLP,
    trials: Sequence[TrialData],
    manifest: dict[str, Any],
    config: AppConfig,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    all_features = np.concatenate([trial.arrays["features"] for trial in trials], axis=0)
    all_targets = np.concatenate([trial.arrays["residual_target"] for trial in trials])
    all_groups = np.concatenate(
        [np.full(trial.size, trial.group, dtype="U32") for trial in trials]
    )
    all_sessions = np.concatenate(
        [np.full(trial.size, trial.session, dtype="U32") for trial in trials]
    )
    all_failure = np.concatenate(
        [np.full(trial.size, trial.is_failure_prefix) for trial in trials]
    ).astype(bool)
    all_pb = np.concatenate([trial.arrays["pb"] for trial in trials])
    all_vb = np.concatenate([trial.arrays["vb"] for trial in trials])
    all_a_nom = np.concatenate([trial.arrays["a_nom"] for trial in trials])
    with torch.no_grad():
        prediction = (
            model(torch.from_numpy(all_features).to(dtype=torch.float64)).cpu().numpy()
        )
    result = {
        "overall": {
            "nominal": _regression_metrics(np.zeros_like(all_targets), all_targets),
            "mlp": _regression_metrics(prediction, all_targets),
        },
        "by_group": {},
        "by_session": {},
        "by_position": {},
        "by_speed_regime": {},
        "by_failure_prefix": {},
    }
    for group in sorted(set(all_groups.tolist())):
        mask = all_groups == group
        result["by_group"][group] = {
            "count": int(mask.sum()),
            "nominal": _regression_metrics(np.zeros(mask.sum()), all_targets[mask]),
            "mlp": _regression_metrics(prediction[mask], all_targets[mask]),
        }
    for session in sorted(set(all_sessions.tolist())):
        mask = all_sessions == session
        result["by_session"][session] = {
            "count": int(mask.sum()),
            "nominal": _regression_metrics(np.zeros(mask.sum()), all_targets[mask]),
            "mlp": _regression_metrics(prediction[mask], all_targets[mask]),
        }
    position_edges = np.asarray(config.data.position_bin_edges_m, dtype=np.float64)
    for index in range(position_edges.size - 1):
        lower, upper = position_edges[index], position_edges[index + 1]
        mask = (all_pb >= lower) & (
            all_pb <= upper if index == 4 else all_pb < upper
        )
        label = f"[{lower:.3f},{upper:.3f}{']' if index == 4 else ')'}"
        if mask.any():
            result["by_position"][label] = {
                "count": int(mask.sum()),
                "nominal": _regression_metrics(np.zeros(mask.sum()), all_targets[mask]),
                "mlp": _regression_metrics(prediction[mask], all_targets[mask]),
            }
    threshold = config.data.speed_threshold_mps
    speed_masks = {
        "negative": all_vb < -threshold,
        "low_speed": np.abs(all_vb) <= threshold,
        "positive": all_vb > threshold,
    }
    for label, mask in speed_masks.items():
        result["by_speed_regime"][label] = {
            "count": int(mask.sum()),
            "nominal": _regression_metrics(np.zeros(mask.sum()), all_targets[mask]),
            "mlp": _regression_metrics(prediction[mask], all_targets[mask]),
        }
    for label, mask in {
        "complete_trial": ~all_failure,
        "failure_prefix": all_failure,
    }.items():
        if mask.any():
            result["by_failure_prefix"][label] = {
                "count": int(mask.sum()),
                "nominal": _regression_metrics(np.zeros(mask.sum()), all_targets[mask]),
                "mlp": _regression_metrics(prediction[mask], all_targets[mask]),
            }
    arrays = {
        "point_pb": all_pb,
        "point_vb": all_vb,
        "point_a_nom": all_a_nom,
        "point_residual_target": all_targets,
        "point_residual_prediction": prediction,
        "point_group": all_groups,
    }
    return result, arrays


def _rollout_metrics(
    model: ResidualMLP,
    trials: Sequence[TrialData],
    manifest: dict[str, Any],
    config: AppConfig,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    if config.causal_rollout.enabled:
        return _causal_rollout_metrics(model, trials, manifest, config)
    max_horizon = max(config.data.evaluation_steps)
    store = WindowStore(trials, max_horizon)
    refs = store.references[:: config.data.validation_stride]
    by_group_refs: dict[str, list[WindowReference]] = {}
    for ref in refs:
        by_group_refs.setdefault(trials[ref.trial_index].group, []).append(ref)
    result = {"overall": {}, "by_group": {}}
    overall_accumulator = _empty_rollout_accumulator(config.data.evaluation_steps)
    example: dict[str, np.ndarray] | None = None
    for group, group_refs in sorted(by_group_refs.items()):
        accumulator = _empty_rollout_accumulator(config.data.evaluation_steps)
        for offset in range(0, len(group_refs), config.training.batch_size):
            batch_refs = group_refs[offset : offset + config.training.batch_size]
            batch = store.make_batch(batch_refs, torch.device("cpu"))
            with torch.no_grad():
                mlp_pb, mlp_vb = rollout_ball_dynamics(
                    model,
                    batch.pb[:, 0],
                    batch.vb[:, 0],
                    batch.theta,
                    batch.omega,
                    batch.dt,
                    batch.constants,
                )
                nominal_pb, nominal_vb = rollout_ball_dynamics(
                    None,
                    batch.pb[:, 0],
                    batch.vb[:, 0],
                    batch.theta,
                    batch.omega,
                    batch.dt,
                    batch.constants,
                )
            _update_rollout_accumulator(
                accumulator, batch, mlp_pb, mlp_vb, nominal_pb, nominal_vb, config
            )
            _update_rollout_accumulator(
                overall_accumulator,
                batch,
                mlp_pb,
                mlp_vb,
                nominal_pb,
                nominal_vb,
                config,
            )
            if example is None:
                horizon_time = np.concatenate(
                    ([0.0], np.cumsum(batch.dt[0].cpu().numpy()))
                )
                example = {
                    "example_time_s": horizon_time,
                    "example_true_pb": batch.pb[0].cpu().numpy(),
                    "example_true_vb": batch.vb[0].cpu().numpy(),
                    "example_mlp_pb": mlp_pb[0].cpu().numpy(),
                    "example_mlp_vb": mlp_vb[0].cpu().numpy(),
                    "example_nominal_pb": nominal_pb[0].cpu().numpy(),
                    "example_nominal_vb": nominal_vb[0].cpu().numpy(),
                    "example_group": np.asarray(group),
                }
        result["by_group"][group] = _finalize_rollout_accumulator(accumulator, config)
    result["overall"] = _finalize_rollout_accumulator(overall_accumulator, config)
    if example is None:
        raise RuntimeError("No rollout example was generated.")
    return result, example


def _causal_rollout_metrics(
    model: ResidualMLP,
    trials: Sequence[TrialData],
    manifest: dict[str, Any],
    config: AppConfig,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    max_horizon = max(config.data.evaluation_steps)
    store = WindowStore(trials, max_horizon, config, manifest["normalization"])
    refs = store.references[:: config.data.validation_stride]
    result: dict[str, Any] = {
        "overall": {},
        "by_group": {},
        "by_session": {},
        "by_position": {},
        "by_speed_regime": {},
        "by_failure_prefix": {},
    }
    overall, example = _evaluate_causal_reference_set(model, store, refs, config)
    result["overall"] = overall

    labels = position_bin_labels(config.data.position_bin_edges_m)
    scope_functions = {
        "by_group": lambda ref: trials[ref.trial_index].group,
        "by_session": lambda ref: trials[ref.trial_index].session,
        "by_position": lambda ref: (
            labels[ref.position_bin] if ref.position_bin >= 0 else "out_of_range"
        ),
        "by_speed_regime": lambda ref: {
            -1: "negative",
            0: "low_speed",
            1: "positive",
        }[ref.regime],
        "by_failure_prefix": lambda ref: (
            "failure_prefix"
            if trials[ref.trial_index].is_failure_prefix
            else "complete_trial"
        ),
    }
    for scope, key_function in scope_functions.items():
        grouped: dict[str, list[WindowReference]] = {}
        for ref in refs:
            grouped.setdefault(str(key_function(ref)), []).append(ref)
        for key, selected in sorted(grouped.items()):
            metrics, _ = _evaluate_causal_reference_set(
                model, store, selected, config, include_example=False
            )
            result[scope][key] = metrics
    return result, example


def _evaluate_causal_reference_set(
    model: ResidualMLP,
    store: WindowStore,
    references: Sequence[WindowReference],
    config: AppConfig,
    *,
    include_example: bool = True,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    accumulator = _empty_causal_rollout_accumulator(config.data.evaluation_steps)
    example: dict[str, np.ndarray] | None = None
    for offset in range(0, len(references), config.training.batch_size):
        batch_refs = references[offset : offset + config.training.batch_size]
        batch = store.make_batch(batch_refs, torch.device("cpu"))
        assert batch.causal_state is not None and batch.commands is not None
        assert batch.geometry is not None and batch.response is not None
        assert batch.smooth_pb is not None and batch.smooth_vb is not None
        with torch.no_grad():
            mlp = rollout_causal_dynamics(
                model,
                batch.causal_state[:, 0],
                batch.commands,
                batch.dt,
                batch.constants,
                batch.geometry,
                batch.response,
            )
            nominal = rollout_causal_dynamics(
                None,
                batch.causal_state[:, 0],
                batch.commands,
                batch.dt,
                batch.constants,
                batch.geometry,
                batch.response,
            )
        _update_causal_rollout_accumulator(
            accumulator, batch, mlp.states, nominal.states, config
        )
        if include_example and example is None:
            trial = store.trials[batch_refs[0].trial_index]
            horizon_time = np.concatenate(([0.0], np.cumsum(batch.dt[0].numpy())))
            example = {
                "example_time_s": horizon_time,
                "example_true_pb": batch.causal_state[0, :, 0].numpy(),
                "example_true_vb": batch.causal_state[0, :, 1].numpy(),
                "example_smooth_pb": batch.smooth_pb[0].numpy(),
                "example_smooth_vb": batch.smooth_vb[0].numpy(),
                "example_true_theta": batch.causal_state[0, :, 2].numpy(),
                "example_true_vrz": batch.causal_state[0, :, 3].numpy(),
                "example_mlp_pb": mlp.states[0, :, 0].numpy(),
                "example_mlp_vb": mlp.states[0, :, 1].numpy(),
                "example_mlp_theta": mlp.states[0, :, 2].numpy(),
                "example_mlp_vrz": mlp.states[0, :, 3].numpy(),
                "example_nominal_pb": nominal.states[0, :, 0].numpy(),
                "example_nominal_vb": nominal.states[0, :, 1].numpy(),
                "example_group": np.asarray(trial.group),
                "example_session": np.asarray(trial.session),
            }
    if not references:
        raise RuntimeError("No causal rollout reference was available.")
    return (
        _finalize_causal_rollout_accumulator(accumulator, config),
        {} if example is None else example,
    )


def _empty_causal_rollout_accumulator(
    steps: Sequence[int],
) -> dict[int, dict[str, float]]:
    names = (
        "mlp_causal_pb_sse",
        "mlp_causal_vb_sse",
        "nominal_causal_pb_sse",
        "nominal_causal_vb_sse",
        "mlp_smooth_pb_sse",
        "mlp_smooth_vb_sse",
        "nominal_smooth_pb_sse",
        "nominal_smooth_vb_sse",
        "theta_sse",
        "theta_error_sum",
        "vrz_sse",
        "vrz_error_sum",
    )
    return {
        step: {"count": 0.0, "dt_sum": 0.0, **{name: 0.0 for name in names}}
        for step in steps
    }


def _update_causal_rollout_accumulator(
    accumulator: dict[int, dict[str, float]],
    batch: Any,
    mlp_states: torch.Tensor,
    nominal_states: torch.Tensor,
    config: AppConfig,
) -> None:
    assert batch.causal_state is not None
    assert batch.smooth_pb is not None and batch.smooth_vb is not None
    for step in config.data.evaluation_steps:
        values = accumulator[step]
        causal_pb = batch.causal_state[:, step, 0]
        causal_vb = batch.causal_state[:, step, 1]
        smooth_pb = batch.smooth_pb[:, step]
        smooth_vb = batch.smooth_vb[:, step]
        count = float(causal_pb.numel())
        values["count"] += count
        values["dt_sum"] += float(batch.dt[:, :step].sum())
        for prefix, states in (("mlp", mlp_states), ("nominal", nominal_states)):
            values[f"{prefix}_causal_pb_sse"] += float(
                (states[:, step, 0] - causal_pb).square().sum()
            )
            values[f"{prefix}_causal_vb_sse"] += float(
                (states[:, step, 1] - causal_vb).square().sum()
            )
            values[f"{prefix}_smooth_pb_sse"] += float(
                (states[:, step, 0] - smooth_pb).square().sum()
            )
            values[f"{prefix}_smooth_vb_sse"] += float(
                (states[:, step, 1] - smooth_vb).square().sum()
            )
        theta_error = mlp_states[:, step, 2] - batch.causal_state[:, step, 2]
        vrz_error = mlp_states[:, step, 3] - batch.causal_state[:, step, 3]
        values["theta_sse"] += float(theta_error.square().sum())
        values["theta_error_sum"] += float(theta_error.sum())
        values["vrz_sse"] += float(vrz_error.square().sum())
        values["vrz_error_sum"] += float(vrz_error.sum())


def _finalize_causal_rollout_accumulator(
    accumulator: dict[int, dict[str, float]],
    config: AppConfig,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for step in config.data.evaluation_steps:
        values = accumulator[step]
        count = max(values["count"], 1.0)

        def target_metrics(target: str) -> dict[str, Any]:
            mlp_pb = values[f"mlp_{target}_pb_sse"] / count
            mlp_vb = values[f"mlp_{target}_vb_sse"] / count
            nominal_pb = values[f"nominal_{target}_pb_sse"] / count
            nominal_vb = values[f"nominal_{target}_vb_sse"] / count
            return {
                "nominal": {
                    "pb_rmse_m": math.sqrt(nominal_pb),
                    "vb_rmse_mps": math.sqrt(nominal_vb),
                },
                "mlp": {
                    "pb_rmse_m": math.sqrt(mlp_pb),
                    "vb_rmse_mps": math.sqrt(mlp_vb),
                },
                "improvement": {
                    "pb_mse_fraction": 1.0 - mlp_pb / max(nominal_pb, 1.0e-15),
                    "vb_mse_fraction": 1.0 - mlp_vb / max(nominal_vb, 1.0e-15),
                },
            }

        causal = target_metrics("causal")
        smooth = target_metrics("smooth")
        result[f"h{step}"] = {
            "count": int(values["count"]),
            "horizon_s": values["dt_sum"] / count,
            **causal,
            "causal": causal,
            "smooth": smooth,
            "response_chain": {
                "theta_rmse_rad": math.sqrt(values["theta_sse"] / count),
                "theta_bias_rad": values["theta_error_sum"] / count,
                "vrz_rmse_mps": math.sqrt(values["vrz_sse"] / count),
                "vrz_bias_mps": values["vrz_error_sum"] / count,
            },
        }
    return result


def _empty_rollout_accumulator(steps: Sequence[int]) -> dict[int, dict[str, float]]:
    return {
        step: {
            "count": 0.0,
            "mlp_pb_sse": 0.0,
            "mlp_vb_sse": 0.0,
            "nominal_pb_sse": 0.0,
            "nominal_vb_sse": 0.0,
            "dt_sum": 0.0,
        }
        for step in steps
    }


def _update_rollout_accumulator(
    accumulator: dict[int, dict[str, float]],
    batch: Any,
    mlp_pb: torch.Tensor,
    mlp_vb: torch.Tensor,
    nominal_pb: torch.Tensor,
    nominal_vb: torch.Tensor,
    config: AppConfig,
) -> None:
    for step in config.data.evaluation_steps:
        target_pb, target_vb = batch.pb[:, step], batch.vb[:, step]
        values = accumulator[step]
        values["count"] += float(target_pb.numel())
        values["mlp_pb_sse"] += float((mlp_pb[:, step] - target_pb).square().sum())
        values["mlp_vb_sse"] += float((mlp_vb[:, step] - target_vb).square().sum())
        values["nominal_pb_sse"] += float(
            (nominal_pb[:, step] - target_pb).square().sum()
        )
        values["nominal_vb_sse"] += float(
            (nominal_vb[:, step] - target_vb).square().sum()
        )
        values["dt_sum"] += float(batch.dt[:, :step].sum())


def _finalize_rollout_accumulator(
    accumulator: dict[int, dict[str, float]],
    config: AppConfig,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for step in config.data.evaluation_steps:
        values = accumulator[step]
        count = max(values["count"], 1.0)
        horizon_s = values["dt_sum"] / count
        mlp_pb_mse = values["mlp_pb_sse"] / count
        mlp_vb_mse = values["mlp_vb_sse"] / count
        nominal_pb_mse = values["nominal_pb_sse"] / count
        nominal_vb_mse = values["nominal_vb_sse"] / count
        result[f"h{step}"] = {
            "count": int(values["count"]),
            "horizon_s": horizon_s,
            "nominal": {
                "pb_rmse_m": math.sqrt(nominal_pb_mse),
                "vb_rmse_mps": math.sqrt(nominal_vb_mse),
            },
            "mlp": {
                "pb_rmse_m": math.sqrt(mlp_pb_mse),
                "vb_rmse_mps": math.sqrt(mlp_vb_mse),
            },
            "improvement": {
                "pb_mse_fraction": 1.0 - mlp_pb_mse / max(nominal_pb_mse, 1e-15),
                "vb_mse_fraction": 1.0 - mlp_vb_mse / max(nominal_vb_mse, 1e-15),
            },
        }
    return result


def _alignment_sensitivity(
    model: ResidualMLP,
    trials: Sequence[TrialData],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for shift in (-1, 0, 1):
        targets: list[np.ndarray] = []
        predictions: list[np.ndarray] = []
        for trial in trials:
            arrays = trial.arrays
            if shift < 0:
                base, shifted = slice(-shift, None), slice(None, shift)
            elif shift > 0:
                base, shifted = slice(None, -shift), slice(shift, None)
            else:
                base, shifted = slice(None), slice(None)
            pb = arrays["pb"][base]
            vb = arrays["vb"][base]
            ab = arrays["ab"][base]
            theta = arrays["theta"][shifted]
            omega = arrays["omega"][shifted]
            a_nom = nominal_acceleration(pb, theta, omega, trial.constants)
            features = np.column_stack((pb, vb, theta, omega, a_nom))
            with torch.no_grad():
                pred = model(torch.from_numpy(features).to(torch.float64)).cpu().numpy()
            targets.append(ab - a_nom)
            predictions.append(pred)
        target = np.concatenate(targets)
        prediction = np.concatenate(predictions)
        result[f"shift_{shift:+d}"] = _regression_metrics(prediction, target)
    result["formal_shift"] = 0
    return result


def _diagnostics(
    model: ResidualMLP,
    trials: Sequence[TrialData],
    manifest: dict[str, Any],
    config: AppConfig,
) -> dict[str, Any]:
    features = np.concatenate([trial.arrays["features"] for trial in trials], axis=0)
    a_nom = np.concatenate([trial.arrays["a_nom"] for trial in trials])
    vb = np.concatenate([trial.arrays["vb"] for trial in trials])
    is_constant = np.concatenate(
        [np.full(trial.size, trial.reference_type == "constant") for trial in trials]
    ).astype(bool)
    tensor = torch.from_numpy(features).to(torch.float64)
    with torch.no_grad():
        residual = model(tensor).cpu().numpy()
    normalized = ((tensor - model.input_mean) / model.input_std).detach().requires_grad_(True)
    normalized_output = model.normalized_residual(normalized)
    gradient = torch.autograd.grad(normalized_output.sum(), normalized)[0]
    gradient_norm = torch.linalg.vector_norm(gradient, dim=-1).detach().cpu().numpy()
    train_min = np.asarray(manifest["normalization"]["min"])
    train_max = np.asarray(manifest["normalization"]["max"])
    ood = np.any((features < train_min) | (features > train_max), axis=1)
    static_mask = is_constant & (np.abs(vb) <= config.data.speed_threshold_mps)
    static_total = np.abs(a_nom[static_mask] + residual[static_mask])
    return {
        "sample_count": int(features.shape[0]),
        "residual_saturation_rate": float(
            np.mean(np.abs(residual) >= 0.95 * model.output_limit_mps2)
        ),
        "residual_min_mps2": float(residual.min()),
        "residual_max_mps2": float(residual.max()),
        "gradient_norm_p95": float(np.percentile(gradient_norm, 95.0)),
        "gradient_norm_p99": float(np.percentile(gradient_norm, 99.0)),
        "gradient_norm_max": float(gradient_norm.max()),
        "gradients_finite": bool(np.isfinite(gradient_norm).all()),
        "outputs_finite": bool(np.isfinite(residual).all()),
        "ood_fraction": float(ood.mean()),
        "static_sample_count": int(static_mask.sum()),
        "static_total_acc_median_mps2": (
            float(np.median(static_total)) if static_total.size else None
        ),
        "static_total_acc_p95_mps2": (
            float(np.percentile(static_total, 95.0)) if static_total.size else None
        ),
    }


def _causal_diagnostics(
    model: ResidualMLP,
    trials: Sequence[TrialData],
    manifest: dict[str, Any],
    config: AppConfig,
) -> dict[str, Any]:
    features = np.concatenate([trial.arrays["features"] for trial in trials], axis=0)
    a_nom = np.concatenate([trial.arrays["a_nom"] for trial in trials])
    vb = np.concatenate([trial.arrays["vb"] for trial in trials])
    is_constant = np.concatenate(
        [np.full(trial.size, trial.reference_type == "constant") for trial in trials]
    ).astype(bool)
    feature_tensor = torch.from_numpy(features).to(torch.float64)
    with torch.no_grad():
        point_residual = model(feature_tensor).numpy()
    point_ood_distances = ood_distances(features, manifest["normalization"])
    point_ood = point_ood_distances["combined"] > 0.0

    max_horizon = max(config.data.evaluation_steps)
    store = WindowStore(trials, max_horizon, config, manifest["normalization"])
    refs = store.references[:: config.data.validation_stride]
    jacobian_norms: list[np.ndarray] = []
    residual_changes: list[np.ndarray] = []
    theta_ratios: list[np.ndarray] = []
    anti_damping_values: list[np.ndarray] = []
    rollout_ood_count = 0
    rollout_feature_count = 0
    for offset in range(0, len(refs), config.training.batch_size):
        batch = store.make_batch(
            refs[offset : offset + config.training.batch_size], torch.device("cpu")
        )
        assert batch.causal_state is not None and batch.commands is not None
        assert batch.geometry is not None and batch.response is not None
        assert batch.state_scales is not None
        with torch.no_grad():
            rollout = rollout_causal_dynamics(
                model,
                batch.causal_state[:, 0],
                batch.commands,
                batch.dt,
                batch.constants,
                batch.geometry,
                batch.response,
            )
        all_states = torch.cat(
            (
                rollout.states,
                rollout.stage_states.reshape(
                    rollout.stage_states.shape[0], -1, 4
                ),
            ),
            dim=1,
        )
        states_per_window = all_states.shape[1]
        flat_states = all_states.reshape(-1, 4).detach().requires_grad_(True)
        constants = batch.constants.repeat_interleave(states_per_window)
        geometry = batch.geometry.repeat_interleave(states_per_window)
        state_features, _, _ = causal_features_torch(flat_states, constants, geometry)
        total, nominal, _, _ = causal_residual_torch(
            model, flat_states, constants, geometry
        )
        total_jacobian = torch.autograd.grad(
            total.sum(), flat_states, retain_graph=True
        )[0]
        nominal_jacobian = torch.autograd.grad(nominal.sum(), flat_states)[0]
        jacobian = total_jacobian - nominal_jacobian
        norms = torch.linalg.vector_norm(
            jacobian * batch.state_scales, dim=-1
        ).detach().numpy()
        jacobian_norms.append(norms)
        theta_mask = (
            nominal_jacobian[:, 2].abs()
            >= config.control_channel.nominal_theta_gain_floor
        )
        if bool(theta_mask.any()):
            theta_ratios.append(
                (
                    total_jacobian[theta_mask, 2]
                    / nominal_jacobian[theta_mask, 2]
                )
                .detach()
                .numpy()
            )
        anti_damping_values.append(total_jacobian[:, 1].detach().numpy())
        state_features_np = state_features.detach().numpy()
        rollout_ood_count += int(
            np.sum(
                ood_distances(
                    state_features_np, manifest["normalization"]
                )["combined"]
                > 0.0
            )
        )
        rollout_feature_count += int(state_features_np.shape[0])

        endpoint_count = rollout.states.shape[1]
        endpoint_constants = batch.constants.repeat_interleave(endpoint_count)
        endpoint_geometry = batch.geometry.repeat_interleave(endpoint_count)
        endpoint_features, _, _ = causal_features_torch(
            rollout.states.reshape(-1, 4), endpoint_constants, endpoint_geometry
        )
        endpoint_residual = model(endpoint_features).reshape(rollout.states.shape[:2])
        residual_changes.append(
            torch.abs(torch.diff(endpoint_residual, dim=1)).detach().numpy().reshape(-1)
        )

    jacobian_values = np.concatenate(jacobian_norms)
    theta_ratio_values = (
        np.concatenate(theta_ratios) if theta_ratios else np.ones(1, dtype=np.float64)
    )
    anti_damping = np.concatenate(anti_damping_values)
    residual_change_values = np.concatenate(residual_changes)
    queue_errors = np.concatenate(
        [
            trial.arrays["response_command"] - trial.arrays["pending_commands"][:, 0]
            for trial in trials
        ]
    )
    static_mask = is_constant & (np.abs(vb) <= config.data.speed_threshold_mps)
    static_total = np.abs(a_nom[static_mask] + point_residual[static_mask])
    limit = config.causal_rollout.jacobian_norm_limit_mps2
    channel = {
        "theta_ratio_count": int(theta_ratio_values.size),
        "theta_ratio_sign_flip_rate": float(np.mean(theta_ratio_values <= 0.0)),
        "theta_ratio_min": float(theta_ratio_values.min()),
        "theta_ratio_p01": float(np.percentile(theta_ratio_values, 1.0)),
        "theta_ratio_p99": float(np.percentile(theta_ratio_values, 99.0)),
        "theta_ratio_max": float(theta_ratio_values.max()),
        "anti_damping_p99_sinv": float(np.percentile(anti_damping, 99.0)),
        "anti_damping_max_sinv": float(anti_damping.max()),
    }
    if config.control_channel.enabled:
        channel["outer_safety_set"] = _outer_safety_diagnostics(
            model, trials, manifest, config
        )
    return {
        "sample_count": int(features.shape[0]),
        "residual_saturation_rate": float(
            np.mean(np.abs(point_residual) >= 0.95 * model.output_limit_mps2)
        ),
        "residual_min_mps2": float(point_residual.min()),
        "residual_max_mps2": float(point_residual.max()),
        "jacobian_state_scale_names": manifest["normalization"]["state_scale_names"],
        "jacobian_state_scales": manifest["normalization"]["state_scales"],
        "jacobian_norm_limit_mps2": limit,
        "jacobian_norm_p50_mps2": float(np.percentile(jacobian_values, 50.0)),
        "jacobian_norm_p95_mps2": float(np.percentile(jacobian_values, 95.0)),
        "jacobian_norm_p99_mps2": float(np.percentile(jacobian_values, 99.0)),
        "jacobian_norm_max_mps2": float(jacobian_values.max()),
        # Backwards-compatible names used by the report renderer.
        "gradient_norm_p95": float(np.percentile(jacobian_values, 95.0)),
        "gradient_norm_p99": float(np.percentile(jacobian_values, 99.0)),
        "gradient_norm_max": float(jacobian_values.max()),
        "jacobian_p99_within_limit": bool(
            np.percentile(jacobian_values, 99.0)
            <= config.causal_rollout.jacobian_p99_factor * limit
        ),
        "jacobian_max_within_limit": bool(
            jacobian_values.max() <= config.causal_rollout.jacobian_max_factor * limit
        ),
        "adjacent_residual_change_p50_mps2": float(
            np.percentile(residual_change_values, 50.0)
        ),
        "adjacent_residual_change_p95_mps2": float(
            np.percentile(residual_change_values, 95.0)
        ),
        "adjacent_residual_change_p99_mps2": float(
            np.percentile(residual_change_values, 99.0)
        ),
        "adjacent_residual_change_max_mps2": float(residual_change_values.max()),
        "command_queue_alignment": {
            "sample_count": int(queue_errors.size),
            "rmse_mps": float(np.sqrt(np.mean(queue_errors**2))),
            "max_abs_error_mps": float(np.max(np.abs(queue_errors))),
            "passed": bool(np.max(np.abs(queue_errors)) <= 1.0e-9),
        },
        "ood_fraction": float(point_ood.mean()),
        "ood_marginal_distance_p99": float(
            np.percentile(point_ood_distances["marginal"], 99.0)
        ),
        "ood_mahalanobis_distance_p99": float(
            np.percentile(point_ood_distances["mahalanobis"], 99.0)
        ),
        "ood_joint_excess_p99": float(
            np.percentile(point_ood_distances["joint_excess"], 99.0)
        ),
        "rollout_state_ood_fraction": float(
            rollout_ood_count / max(rollout_feature_count, 1)
        ),
        "static_sample_count": int(static_mask.sum()),
        "static_total_acc_median_mps2": (
            float(np.median(static_total)) if static_total.size else None
        ),
        "static_total_acc_p95_mps2": (
            float(np.percentile(static_total, 95.0)) if static_total.size else None
        ),
        "gradients_finite": bool(np.isfinite(jacobian_values).all()),
        "outputs_finite": bool(np.isfinite(point_residual).all()),
        "channel": channel,
    }


class _AppliedResidual(torch.nn.Module):
    def __init__(
        self,
        model: ResidualMLP,
        normalization: dict[str, Any],
        *,
        scale: float = 1.0,
        width: float = 1.0,
        min_weight: float = 0.0,
    ) -> None:
        super().__init__()
        self.model = model
        self.normalization = normalization
        self.scale = float(scale)
        self.width = float(width)
        self.min_weight = float(min_weight)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return (
            self.scale
            * _ood_weight_torch(
                features,
                self.normalization,
                width=self.width,
                min_weight=self.min_weight,
            )
            * self.model(features)
        )


def _ood_weight_torch(
    features: torch.Tensor,
    normalization: dict[str, Any],
    *,
    width: float,
    min_weight: float,
) -> torch.Tensor:
    std = torch.as_tensor(
        normalization["std"], device=features.device, dtype=features.dtype
    )
    ood = normalization.get("ood")
    if ood:
        lower = torch.as_tensor(ood["lower"], device=features.device, dtype=features.dtype)
        upper = torch.as_tensor(ood["upper"], device=features.device, dtype=features.dtype)
    else:
        lower = torch.as_tensor(normalization["min"], device=features.device, dtype=features.dtype)
        upper = torch.as_tensor(normalization["max"], device=features.device, dtype=features.dtype)
    violation = torch.maximum(
        torch.zeros_like(features),
        torch.maximum((lower - features) / std, (features - upper) / std),
    )
    marginal_squared = violation.square().sum(dim=-1)
    joint_squared = torch.zeros_like(marginal_squared)
    if ood:
        mean = torch.as_tensor(
            normalization["mean"], device=features.device, dtype=features.dtype
        )
        inverse = torch.as_tensor(
            ood["inverse_correlation"], device=features.device, dtype=features.dtype
        )
        z = (features - mean) / std
        mahalanobis = torch.sqrt(
            torch.clamp(torch.einsum("...i,ij,...j->...", z, inverse, z), min=0.0)
        )
        joint = torch.relu(mahalanobis - float(ood["mahalanobis_radius"]))
        joint_squared = joint.square()
    distance_squared = marginal_squared + joint_squared
    return min_weight + (1.0 - min_weight) * torch.exp(
        -distance_squared / (2.0 * width**2)
    )


def _outer_safety_diagnostics(
    model: ResidualMLP,
    trials: Sequence[TrialData],
    manifest: dict[str, Any],
    config: AppConfig,
) -> dict[str, Any]:
    count = config.control_channel.outer_safety_points
    base = (torch.arange(count, dtype=torch.float64) + 0.5) / count

    def dimension(low: float, high: float, shift: int) -> torch.Tensor:
        return low + (high - low) * torch.roll(base, shifts=shift)

    channel = config.control_channel
    states = torch.stack(
        (
            dimension(config.data.ball_position_min_m, config.data.ball_position_max_m, 0),
            dimension(channel.collocation_vb_min_mps, channel.collocation_vb_max_mps, 977),
            dimension(channel.outer_theta_min_rad, channel.outer_theta_max_rad, 1951),
            dimension(channel.collocation_vrz_min_mps, channel.collocation_vrz_max_mps, 3011),
        ),
        dim=-1,
    ).requires_grad_(True)
    indices = np.arange(count) % len(trials)
    constants = [trials[int(index)].constants for index in indices]
    from .dynamics import TorchGeometryConstants, TorchPhysicalConstants

    physical = TorchPhysicalConstants(
        plank_length_m=torch.tensor(
            [item.plank_length_m for item in constants], dtype=torch.float64
        ),
        ball_position_offset_m=torch.tensor(
            [item.ball_position_offset_m for item in constants], dtype=torch.float64
        ),
        gravity_mps2=torch.tensor(
            [item.gravity_mps2 for item in constants], dtype=torch.float64
        ),
        acceleration_gain=torch.tensor(
            [item.acceleration_gain for item in constants], dtype=torch.float64
        ),
    )
    geometry = TorchGeometryConstants(
        plank_length_m=torch.tensor(
            [item.plank_length_m for item in constants], dtype=torch.float64
        ),
        rope_length_m=torch.tensor(
            [item.rope_length_m for item in constants], dtype=torch.float64
        ),
    )
    features, nominal, _ = causal_features_torch(states, physical, geometry)
    applied = _AppliedResidual(model, manifest["normalization"])(features)
    total = nominal + applied
    total_jacobian = torch.autograd.grad(total.sum(), states, retain_graph=True)[0]
    nominal_jacobian = torch.autograd.grad(nominal.sum(), states)[0]
    mask = nominal_jacobian[:, 2].abs() >= channel.nominal_theta_gain_floor
    ratio = (total_jacobian[mask, 2] / nominal_jacobian[mask, 2]).detach().numpy()
    anti = total_jacobian[:, 1].detach().numpy()
    weights = _ood_weight_torch(
        features.detach(), manifest["normalization"], width=1.0, min_weight=0.0
    ).numpy()
    return {
        "sample_count": count,
        "all_finite": bool(
            np.isfinite(ratio).all()
            and np.isfinite(anti).all()
            and np.isfinite(weights).all()
        ),
        "theta_ratio_sign_flip_rate": float(np.mean(ratio <= 0.0)),
        "theta_ratio_min": float(ratio.min()),
        "theta_ratio_p01": float(np.percentile(ratio, 1.0)),
        "theta_ratio_p99": float(np.percentile(ratio, 99.0)),
        "theta_ratio_max": float(ratio.max()),
        "anti_damping_p99_sinv": float(np.percentile(anti, 99.0)),
        "anti_damping_max_sinv": float(anti.max()),
        "ood_weight_min": float(weights.min()),
        "ood_weight_p01": float(np.percentile(weights, 1.0)),
    }


def _control_sensitivity(
    model: ResidualMLP,
    trials: Sequence[TrialData],
    manifest: dict[str, Any],
    config: AppConfig,
) -> dict[str, Any]:
    horizon = max(config.control_channel.sensitivity_horizons)
    store = WindowStore(trials, horizon, config, manifest["normalization"])
    references = _stratified_sensitivity_references(
        store.references, trials, config.control_channel.sensitivity_max_windows
    )
    applied_model = _AppliedResidual(model, manifest["normalization"])
    result: dict[str, Any] = {}
    for selected_horizon in config.control_channel.sensitivity_horizons:
        cosine_values: list[np.ndarray] = []
        ratio_values: list[np.ndarray] = []
        residual_norms: list[np.ndarray] = []
        nominal_norms: list[np.ndarray] = []
        for offset in range(0, len(references), min(config.training.batch_size, 32)):
            refs = references[offset : offset + min(config.training.batch_size, 32)]
            batch = store.make_batch(refs, torch.device("cpu"))
            assert batch.causal_state is not None and batch.commands is not None
            assert batch.geometry is not None and batch.response is not None
            commands = batch.commands[:, :selected_horizon].detach().requires_grad_(True)
            dt = batch.dt[:, :selected_horizon]
            residual_rollout = rollout_causal_dynamics(
                applied_model,
                batch.causal_state[:, 0],
                commands,
                dt,
                batch.constants,
                batch.geometry,
                batch.response,
            )
            residual_pb = torch.autograd.grad(
                residual_rollout.states[:, -1, 0].sum(),
                commands,
                retain_graph=True,
            )[0]
            residual_vb = torch.autograd.grad(
                residual_rollout.states[:, -1, 1].sum(), commands
            )[0]
            residual_sensitivity = torch.cat((residual_pb, residual_vb), dim=-1)

            nominal_commands = commands.detach().requires_grad_(True)
            nominal_rollout = rollout_causal_dynamics(
                None,
                batch.causal_state[:, 0],
                nominal_commands,
                dt,
                batch.constants,
                batch.geometry,
                batch.response,
            )
            nominal_pb = torch.autograd.grad(
                nominal_rollout.states[:, -1, 0].sum(),
                nominal_commands,
                retain_graph=True,
            )[0]
            nominal_vb = torch.autograd.grad(
                nominal_rollout.states[:, -1, 1].sum(), nominal_commands
            )[0]
            nominal_sensitivity = torch.cat((nominal_pb, nominal_vb), dim=-1)
            residual_norm = torch.linalg.vector_norm(residual_sensitivity, dim=-1)
            nominal_norm = torch.linalg.vector_norm(nominal_sensitivity, dim=-1)
            cosine = torch.nn.functional.cosine_similarity(
                residual_sensitivity, nominal_sensitivity, dim=-1, eps=1.0e-15
            )
            cosine_values.append(cosine.detach().numpy())
            ratio_values.append(
                (residual_norm / nominal_norm.clamp_min(1.0e-15)).detach().numpy()
            )
            residual_norms.append(residual_norm.detach().numpy())
            nominal_norms.append(nominal_norm.detach().numpy())
        cosine = np.concatenate(cosine_values)
        ratio = np.concatenate(ratio_values)
        residual_norm = np.concatenate(residual_norms)
        nominal_norm = np.concatenate(nominal_norms)
        result[f"h{selected_horizon}"] = _sensitivity_statistics(
            cosine, ratio, residual_norm, nominal_norm
        )
    return result


def _sensitivity_statistics(
    cosine: np.ndarray,
    ratio: np.ndarray,
    residual_norm: np.ndarray,
    nominal_norm: np.ndarray,
) -> dict[str, Any]:
    return {
        "window_count": int(cosine.size),
        "cosine_p01": float(np.percentile(cosine, 1.0)),
        "cosine_median": float(np.median(cosine)),
        "norm_ratio_p01": float(np.percentile(ratio, 1.0)),
        "norm_ratio_p99": float(np.percentile(ratio, 99.0)),
        "near_zero_fraction": float(
            np.mean(residual_norm <= 0.1 * np.maximum(nominal_norm, 1.0e-15))
        ),
        "sign_reversal_fraction": float(np.mean(cosine <= 0.0)),
        "all_finite": bool(np.isfinite(cosine).all() and np.isfinite(ratio).all()),
    }


def _stratified_sensitivity_references(
    references: Sequence[WindowReference],
    trials: Sequence[TrialData],
    maximum: int,
) -> list[WindowReference]:
    buckets: dict[tuple[str, int], list[WindowReference]] = {}
    for ref in references:
        key = (trials[ref.trial_index].session, ref.position_bin)
        buckets.setdefault(key, []).append(ref)
    selected: list[WindowReference] = []
    depth = 0
    keys = sorted(buckets)
    while len(selected) < maximum:
        added = False
        for key in keys:
            bucket = buckets[key]
            if depth < len(bucket):
                selected.append(bucket[depth])
                added = True
                if len(selected) == maximum:
                    break
        if not added:
            break
        depth += 1
    return selected


def _acceptance(
    point: dict[str, Any],
    rollout: dict[str, Any],
    diagnostics: dict[str, Any],
    config: AppConfig,
    *,
    split: str = "validation",
) -> dict[str, Any]:
    residual_ok = (
        point["overall"]["mlp"]["rmse"]
        <= config.acceptance.residual_rmse_max_mps2
    )
    primary_horizon = "h33" if config.causal_rollout.enabled else "h30"
    primary = rollout["overall"][primary_horizon]
    position_improvement_ok = (
        primary["improvement"]["pb_mse_fraction"]
        >= config.acceptance.rollout_improvement_min
    )
    velocity_improvement_ok = (
        primary["improvement"]["vb_mse_fraction"]
        >= config.acceptance.rollout_improvement_min
    )
    dynamic_checks: dict[str, bool] = {}
    dynamic_horizon = "h33" if config.causal_rollout.enabled else "h30"
    for group, group_metrics in rollout["by_group"].items():
        if group.startswith("constant"):
            continue
        metric = group_metrics[dynamic_horizon]
        pb_ratio = metric["mlp"]["pb_rmse_m"] / max(
            metric["nominal"]["pb_rmse_m"], 1e-15
        )
        vb_ratio = metric["mlp"]["vb_rmse_mps"] / max(
            metric["nominal"]["vb_rmse_mps"], 1e-15
        )
        dynamic_checks[group] = max(pb_ratio, vb_ratio) <= (
            1.0 + config.acceptance.per_dynamic_degradation_max
        )
    static_value = diagnostics["static_total_acc_median_mps2"]
    static_ok = static_value is not None and (
        static_value <= config.acceptance.static_total_acc_median_max_mps2
    )
    saturation_ok = (
        diagnostics["residual_saturation_rate"]
        <= config.acceptance.saturation_rate_max
    )
    finite_ok = diagnostics["gradients_finite"] and diagnostics["outputs_finite"]
    checks = {
        "residual_rmse": bool(residual_ok),
        f"{primary_horizon}_position_improvement": bool(position_improvement_ok),
        f"{primary_horizon}_velocity_improvement": bool(velocity_improvement_ok),
        "all_dynamic_groups_no_large_degradation": bool(all(dynamic_checks.values())),
        "static_total_acceleration": bool(static_ok),
        "residual_saturation": bool(saturation_ok),
        "finite_outputs_and_gradients": bool(finite_ok),
    }
    if config.causal_rollout.enabled:
        response_horizon = "h60" if "h60" in rollout["overall"] else dynamic_horizon
        response = rollout["overall"][response_horizon]["response_chain"]
        # Bias gates are intentionally in physical units: residual acceleration cannot
        # affect theta/vrz and must not be allowed to hide response-model mismatch.
        response_bias_ok = (
            abs(response["theta_bias_rad"]) <= 0.005
            and abs(response["vrz_bias_mps"]) <= 0.02
        )
        diagnostics["response_chain_gate"] = {
            "horizon": response_horizon,
            **response,
            "theta_bias_limit_rad": 0.005,
            "vrz_bias_limit_mps": 0.02,
            "systematic_bias_detected": not response_bias_ok,
        }
        checks.update(
            {
                "jacobian_p99_within_limit": bool(
                    diagnostics["jacobian_p99_within_limit"]
                ),
                "jacobian_max_within_limit": bool(
                    diagnostics["jacobian_max_within_limit"]
                ),
                "command_queue_alignment": bool(
                    diagnostics["command_queue_alignment"]["passed"]
                ),
                "response_chain_no_systematic_bias": bool(response_bias_ok),
            }
        )
    if config.control_channel.enabled:
        checks.update(_channel_acceptance_checks(diagnostics, config, split=split))
    if split == "stress":
        metric = rollout["overall"][primary_horizon]
        pb_ratio = metric["mlp"]["pb_rmse_m"] / max(
            metric["nominal"]["pb_rmse_m"], 1.0e-15
        )
        vb_ratio = metric["mlp"]["vb_rmse_mps"] / max(
            metric["nominal"]["vb_rmse_mps"], 1.0e-15
        )
        channel_checks = (
            _channel_acceptance_checks(diagnostics, config, split=split)
            if config.control_channel.enabled
            else {}
        )
        checks = {
            "finite_outputs_gradients_and_rollout": bool(finite_ok),
            "residual_saturation": bool(saturation_ok),
            f"{primary_horizon}_pb_degradation_within_10pct": bool(pb_ratio <= 1.1),
            f"{primary_horizon}_vb_degradation_within_10pct": bool(vb_ratio <= 1.1),
            "command_queue_alignment": bool(
                diagnostics["command_queue_alignment"]["passed"]
            ),
            **channel_checks,
        }
    return {
        "passed": bool(all(checks.values())),
        "eligible_for_real_world_test": bool(all(checks.values())),
        "checks": checks,
        "dynamic_group_checks": dynamic_checks,
    }


def _channel_acceptance_checks(
    diagnostics: dict[str, Any],
    config: AppConfig,
    *,
    split: str,
) -> dict[str, bool]:
    channel = config.control_channel
    local = diagnostics["channel"]
    outer = local["outer_safety_set"]

    def theta_ok(values: dict[str, Any]) -> bool:
        return bool(
            values["theta_ratio_sign_flip_rate"] == 0.0
            and values["theta_ratio_p01"] >= channel.theta_ratio_soft_min
            and values["theta_ratio_p99"] <= channel.theta_ratio_soft_max
            and values["theta_ratio_min"] >= channel.theta_ratio_hard_min
            and values["theta_ratio_max"] <= channel.theta_ratio_hard_max
        )

    def damping_ok(values: dict[str, Any]) -> bool:
        return bool(
            values["anti_damping_p99_sinv"]
            <= channel.anti_damping_hard_p99_sinv
            and values["anti_damping_max_sinv"]
            <= channel.anti_damping_hard_max_sinv
        )

    sensitivity = diagnostics["control_sensitivity"]
    if split == "stress":
        sensitivity_ok = all(
            values["all_finite"]
            and values["near_zero_fraction"] == 0.0
            and values["sign_reversal_fraction"] == 0.0
            for values in sensitivity.values()
        )
    else:
        sensitivity_ok = all(
            values["all_finite"]
            and values["cosine_p01"] >= channel.sensitivity_cosine_p01_min
            and values["norm_ratio_p01"] >= channel.sensitivity_ratio_p01_min
            and values["norm_ratio_p99"] <= channel.sensitivity_ratio_p99_max
            and values["near_zero_fraction"] == 0.0
            and values["sign_reversal_fraction"] == 0.0
            for values in sensitivity.values()
        )
    return {
        "theta_sector": theta_ok(local),
        "anti_damping": damping_ok(local),
        "outer_safety_set_finite": bool(outer["all_finite"]),
        "outer_theta_sector": theta_ok(outer),
        "outer_anti_damping": damping_ok(outer),
        "control_sensitivity": bool(sensitivity_ok),
    }


def _regression_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    if target.size == 0:
        return {"count": 0, "rmse": None, "mae": None, "bias": None, "r2": None}
    error = prediction - target
    denominator = float(np.square(target - target.mean()).sum())
    return {
        "count": int(target.size),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "mae": float(np.mean(np.abs(error))),
        "bias": float(np.mean(error)),
        "r2": float(1.0 - np.square(error).sum() / max(denominator, 1e-15)),
    }


def _write_metrics_csv(path: Path, summary: dict[str, Any]) -> None:
    rows: list[dict[str, Any]] = []
    point = summary["point"]
    for scope, values in [("overall", point["overall"]), *point["by_group"].items()]:
        for model_name in ("nominal", "mlp"):
            rows.append(
                {
                    "category": "point",
                    "scope": scope,
                    "horizon_steps": "",
                    "model": model_name,
                    **values[model_name],
                }
            )
    for scope, horizons in [
        ("overall", summary["rollout"]["overall"]),
        *summary["rollout"]["by_group"].items(),
    ]:
        for horizon, values in horizons.items():
            for model_name in ("nominal", "mlp"):
                rows.append(
                    {
                        "category": "rollout",
                        "scope": scope,
                        "horizon_steps": int(horizon[1:]),
                        "model": model_name,
                        "count": values["count"],
                        "horizon_s": values["horizon_s"],
                        **values[model_name],
                        **(
                            {
                                "pb_mse_improvement": values["improvement"][
                                    "pb_mse_fraction"
                                ],
                                "vb_mse_improvement": values["improvement"][
                                    "vb_mse_fraction"
                                ],
                            }
                            if model_name == "mlp"
                            else {}
                        ),
                    }
                )
    keys = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _plot_training_history(history_path: Path, figures_dir: Path) -> None:
    if not history_path.is_file():
        return
    data = np.genfromtxt(history_path, delimiter=",", names=True)
    if data.size == 0:
        return
    data = np.atleast_1d(data)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.semilogy(data["epoch"], data["train_total_loss"], label="Train")
    ax.semilogy(data["epoch"], data["validation_total_loss"], label="Validation")
    ax.set(xlabel="Epoch", ylabel="Composite loss", title="Residual MLP training")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "training_history.png", dpi=160)
    plt.close(fig)


def _plot_point_residual(arrays: dict[str, np.ndarray], figures_dir: Path) -> None:
    target = arrays["point_residual_target"]
    prediction = arrays["point_residual_prediction"]
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2))
    axes[0].hexbin(target, prediction, gridsize=45, mincnt=1, cmap="viridis")
    lower = min(float(target.min()), float(prediction.min()))
    upper = max(float(target.max()), float(prediction.max()))
    axes[0].plot([lower, upper], [lower, upper], "k--", linewidth=1)
    axes[0].set(xlabel="Target residual (m/s²)", ylabel="MLP residual (m/s²)")
    axes[1].hexbin(
        arrays["point_pb"], target - prediction, gridsize=45, mincnt=1, cmap="coolwarm"
    )
    axes[1].axhline(0.0, color="k", linestyle="--", linewidth=1)
    axes[1].set(xlabel="Ball position (m)", ylabel="Residual error (m/s²)")
    fig.tight_layout()
    fig.savefig(figures_dir / "point_residual_validation.png", dpi=160)
    plt.close(fig)


def _plot_rollout_metrics(rollout: dict[str, Any], figures_dir: Path) -> None:
    keys = sorted(rollout["overall"], key=lambda item: int(item[1:]))
    horizons = [rollout["overall"][key]["horizon_s"] for key in keys]
    width = 0.34
    x = np.arange(len(keys))
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2))
    for axis, state, ylabel in (
        (axes[0], "pb_rmse_m", "Position RMSE (m)"),
        (axes[1], "vb_rmse_mps", "Velocity RMSE (m/s)"),
    ):
        nominal = [rollout["overall"][key]["nominal"][state] for key in keys]
        mlp = [rollout["overall"][key]["mlp"][state] for key in keys]
        axis.bar(x - width / 2, nominal, width, label="Nominal")
        axis.bar(x + width / 2, mlp, width, label="Residual MLP")
        axis.set_xticks(x, [f"{value:.2f}s" for value in horizons])
        axis.set(xlabel="Rollout horizon", ylabel=ylabel)
        axis.grid(axis="y", alpha=0.25)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "rollout_rmse_validation.png", dpi=160)
    plt.close(fig)


def _plot_example_rollout(example: dict[str, np.ndarray], figures_dir: Path) -> None:
    time_s = example["example_time_s"]
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.0), sharex=True)
    for axis, suffix, ylabel in (
        (axes[0], "pb", "Ball position (m)"),
        (axes[1], "vb", "Ball velocity (m/s)"),
    ):
        axis.plot(time_s, example[f"example_true_{suffix}"], label="Measured")
        axis.plot(time_s, example[f"example_nominal_{suffix}"], label="Nominal")
        axis.plot(time_s, example[f"example_mlp_{suffix}"], label="Residual MLP")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
    axes[0].legend()
    axes[1].set_xlabel("Horizon time (s)")
    fig.tight_layout()
    fig.savefig(figures_dir / "example_rollout_validation.png", dpi=160)
    plt.close(fig)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
