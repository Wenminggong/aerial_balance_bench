"""Held-out evaluation and hard gates for the two-input RBF residual."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from residual_dynamics.model import load_portable_weights as load_mlp_weights
from residual_dynamics.dynamics import rollout_causal_dynamics

from .config import Config
from .data import load_manifest, load_trials, ood_numpy, position_bins, velocity_bins
from .model import load_checkpoint, save_portable_json
from .training import WindowReference, WindowStore, integral_errors, resolve_run_dir


def regression(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    error = prediction - target
    centered = target - target.mean()
    return {
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "bias": float(np.mean(error)),
        "r2": float(1.0 - np.sum(error**2) / max(np.sum(centered**2), 1.0e-15)),
    }


def shadow_full_state_ood(
    features: np.ndarray, normalization: Mapping[str, Any]
) -> dict[str, Any]:
    """Evaluate the legacy five-state OOD descriptor without applying it."""

    values = np.asarray(features, dtype=np.float64)
    mean = np.asarray(normalization["mean"], dtype=np.float64)
    std = np.maximum(np.asarray(normalization["std"], dtype=np.float64), 1.0e-12)
    ood = normalization["ood"]
    lower = np.asarray(ood["lower"], dtype=np.float64)
    upper = np.asarray(ood["upper"], dtype=np.float64)
    marginal_components = np.maximum(
        0.0, np.maximum((lower - values) / std, (values - upper) / std)
    )
    marginal = np.linalg.norm(marginal_components, axis=-1)
    z = (values - mean) / std
    inverse = np.asarray(ood["inverse_correlation"], dtype=np.float64)
    mahalanobis = np.sqrt(
        np.maximum(0.0, np.einsum("...i,ij,...j->...", z, inverse, z))
    )
    joint_excess = np.maximum(0.0, mahalanobis - float(ood["mahalanobis_radius"]))
    distance = np.sqrt(marginal**2 + joint_excess**2)
    return {
        "diagnostic_only": True,
        "sample_count": int(values.shape[0]),
        "fraction_outside": float(np.mean(distance > 0.0)),
        "distance_p50": float(np.percentile(distance, 50.0)),
        "distance_p99": float(np.percentile(distance, 99.0)),
        "distance_max": float(np.max(distance)),
        "marginal_p99": float(np.percentile(marginal, 99.0)),
        "mahalanobis_p99": float(np.percentile(mahalanobis, 99.0)),
        "joint_excess_p99": float(np.percentile(joint_excess, 99.0)),
        "all_finite": bool(
            np.isfinite(distance).all() and np.isfinite(mahalanobis).all()
        ),
    }


def command_queue_alignment(trials: Sequence[Any]) -> dict[str, Any]:
    """Verify that the response input is the head of the logged pending queue."""

    errors = np.concatenate(
        [
            trial.arrays["response_command"]
            - trial.arrays["pending_commands"][:, 0]
            for trial in trials
        ]
    )
    finite = np.isfinite(errors)
    max_error = float(np.max(np.abs(errors[finite]))) if finite.any() else math.inf
    return {
        "sample_count": int(errors.size),
        "finite_count": int(finite.sum()),
        "rmse_mps": float(np.sqrt(np.mean(errors[finite] ** 2)))
        if finite.any()
        else math.inf,
        "max_abs_error_mps": max_error,
        "passed": bool(finite.all() and max_error <= 1.0e-9),
    }


def _point_metrics(model: Any, baseline: Any | None, trials: Sequence[Any], manifest: Mapping[str, Any], config: Config) -> dict[str, Any]:
    features = np.concatenate([trial.arrays["features"] for trial in trials])
    rbf_features = features[:, :2]
    targets = np.concatenate([trial.arrays["residual_target"] for trial in trials])
    valid = np.concatenate([trial.arrays["rbf_valid"] for trial in trials]).astype(bool)
    pb, vb, anom = features[:, 0], features[:, 1], features[:, 4]
    groups = np.concatenate([np.full(trial.size, trial.group, dtype="U32") for trial in trials])
    sessions = np.concatenate([np.full(trial.size, trial.session, dtype="U32") for trial in trials])
    with torch.no_grad():
        rbf_prediction = model(torch.from_numpy(features).to(torch.float64)).numpy()
        baseline_prediction = (
            None
            if baseline is None
            else baseline(torch.from_numpy(features).to(torch.float64)).numpy()
        )
    result: dict[str, Any] = {
        "overall": {
            "count": int(valid.sum()),
            "nominal": regression(np.zeros(valid.sum()), targets[valid]),
            "rbf": regression(rbf_prediction[valid], targets[valid]),
            "baseline_mlp": None
            if baseline_prediction is None
            else regression(baseline_prediction[valid], targets[valid]),
        },
        "excluded_audit": {
            "count": int((~valid).sum()),
            "outputs_finite": bool(np.isfinite(rbf_prediction[~valid]).all()),
            "ood_finite": bool(
                np.isfinite(ood_numpy(rbf_features[~valid], manifest["normalization"])["distance"]).all()
            ),
        },
        "by_group": {},
        "by_session": {},
        "by_position": {},
        "by_velocity": {},
        "by_position_velocity": {},
        "by_actuation": {},
    }

    def add_category(target_dict: dict[str, Any], label: str, mask: np.ndarray) -> None:
        use = mask & valid
        if not use.any():
            return
        target_dict[label] = {
            "count": int(use.sum()),
            "nominal": regression(np.zeros(use.sum()), targets[use]),
            "rbf": regression(rbf_prediction[use], targets[use]),
        }

    for value in sorted(set(groups.tolist())):
        add_category(result["by_group"], value, groups == value)
    for value in sorted(set(sessions.tolist())):
        add_category(result["by_session"], value, sessions == value)
    pedges = np.asarray(config.legacy.data.position_bin_edges_m)
    for index, (low, high) in enumerate(zip(pedges[:-1], pedges[1:])):
        mask = (pb >= low) & (pb <= high if index == len(pedges) - 2 else pb < high)
        add_category(result["by_position"], f"[{low:.2f},{high:.2f}{']' if index == 6 else ')'}", mask)
    vedges = np.asarray(config.sampling.velocity_bin_edges_mps)
    for index, (low, high) in enumerate(zip(vedges[:-1], vedges[1:])):
        mask = (vb >= low) & (vb < high)
        if np.isclose(low, -0.02):
            mask = (vb >= low) & (vb <= high)
        add_category(result["by_velocity"], f"vbin_{index}", mask)
    pbins = position_bins(pb, config.legacy.data.position_bin_edges_m)
    vbins = velocity_bins(vb, config.sampling.velocity_bin_edges_mps)
    for pbin, vbin in sorted(set(zip(pbins.tolist(), vbins.tolist()))):
        if pbin >= 0:
            add_category(
                result["by_position_velocity"],
                f"p{pbin}_v{vbin}",
                (pbins == pbin) & (vbins == vbin),
            )
    threshold = config.sampling.actuation_threshold_mps2
    for label, mask in {
        "negative": anom < -threshold,
        "near_zero": np.abs(anom) <= threshold,
        "positive": anom > threshold,
    }.items():
        add_category(result["by_actuation"], label, mask)
    ood = ood_numpy(rbf_features, manifest["normalization"])
    static = valid & np.char.startswith(groups, "constant") & (np.abs(vb) <= 0.02)
    result["diagnostics"] = {
        "output_limit_mps2": float(model.output_limit_mps2),
        "max_abs_output_mps2": float(np.max(np.abs(rbf_prediction))),
        "saturation_rate": float(np.mean(np.abs(rbf_prediction[valid]) >= 0.95 * model.output_limit_mps2)),
        "ood_fraction": float(np.mean(ood["distance"][valid] > 0.0)),
        "ood_weight_mean": float(np.mean(ood["weight"][valid])),
        "ood_weight_min": float(np.min(ood["weight"])),
        "static_total_acc_median_mps2": float(
            np.median(np.abs(features[static, 4] + rbf_prediction[static]))
        ),
        "outputs_finite": bool(np.isfinite(rbf_prediction).all()),
        "shadow_full_state_ood": shadow_full_state_ood(
            features, manifest["shadow_full_state_normalization"]
        ),
    }
    return result


def _refs_for_evaluation(store: WindowStore, maximum: int = 2048) -> list[WindowReference]:
    refs = store.references[::5]
    if len(refs) > maximum:
        indices = np.linspace(0, len(refs) - 1, maximum, dtype=int)
        refs = [refs[index] for index in indices]
    return list(refs)


def _integral_metrics(model: Any, store: WindowStore, refs: Sequence[WindowReference], config: Config) -> dict[str, Any]:
    squared = {h: [0.0, 0.0, 0] for h in config.identification_loss.horizons}
    for offset in range(0, len(refs), 256):
        part = refs[offset : offset + 256]
        batch = store.make_batch(part, torch.device("cpu"))
        with torch.no_grad():
            values = integral_errors(model, batch, config)
        for horizon, (rbf_error, nominal_error) in values.items():
            squared[horizon][0] += float(rbf_error.square().sum())
            squared[horizon][1] += float(nominal_error.square().sum())
            squared[horizon][2] += len(part)
    result = {}
    for horizon, (rbf_sum, nominal_sum, count) in squared.items():
        rbf_rmse = math.sqrt(rbf_sum / count)
        nominal_rmse = math.sqrt(nominal_sum / count)
        result[f"h{horizon}"] = {
            "horizon_s": horizon / 60.0,
            "rbf_rmse_mps2": rbf_rmse,
            "nominal_rmse_mps2": nominal_rmse,
            "improvement_fraction": 1.0 - (rbf_rmse / max(nominal_rmse, 1.0e-12)) ** 2,
        }
    return {"overall": result}


def _rollout_metrics(model: Any, baseline: Any | None, store: WindowStore, refs: Sequence[WindowReference], config: Config) -> dict[str, Any]:
    horizons = tuple(config.legacy.data.evaluation_steps)
    totals: dict[str, dict[int, list[float]]] = {
        "rbf": {h: [0.0, 0.0, 0.0, 0.0, 0] for h in horizons},
        "nominal": {h: [0.0, 0.0, 0.0, 0.0, 0] for h in horizons},
    }
    if baseline is not None:
        totals["baseline_mlp"] = {h: [0.0, 0.0, 0.0, 0.0, 0] for h in horizons}
    category_totals: dict[
        str, dict[str, dict[str, dict[int, list[float]]]]
    ] = {
        "by_group": {},
        "by_session": {},
        "by_position_velocity": {},
        "by_actuation": {},
    }
    models = {"rbf": model, "nominal": None}
    if baseline is not None:
        models["baseline_mlp"] = baseline
    for offset in range(0, len(refs), 128):
        part = refs[offset : offset + 128]
        batch = store.make_batch(part, torch.device("cpu"))
        with torch.no_grad():
            rollouts = {
                name: rollout_causal_dynamics(
                    candidate,
                    batch.causal_state[:, 0],
                    batch.commands,
                    batch.dt,
                    batch.constants,
                    batch.geometry,
                    batch.response,
                )
                for name, candidate in models.items()
            }
        for local, ref in enumerate(part):
            trial = store.trials[ref.trial_index]
            labels = {
                "by_group": trial.group,
                "by_session": trial.session,
                "by_position_velocity": f"p{ref.position_bin}_v{ref.velocity_bin}",
                "by_actuation": { -1: "negative", 0: "near_zero", 1: "positive" }[
                    ref.actuation_regime
                ],
            }
            for category, label in labels.items():
                category_totals[category].setdefault(
                    label,
                    {name: {h: [0.0, 0.0, 0] for h in horizons} for name in models},
                )
            for name, rollout in rollouts.items():
                for horizon in horizons:
                    p_error = float(rollout.states[local, horizon, 0] - batch.causal_state[local, horizon, 0])
                    v_error = float(rollout.states[local, horizon, 1] - batch.causal_state[local, horizon, 1])
                    theta_error = float(rollout.states[local, horizon, 2] - batch.causal_state[local, horizon, 2])
                    vrz_error = float(rollout.states[local, horizon, 3] - batch.causal_state[local, horizon, 3])
                    value = totals[name][horizon]
                    value[0] += p_error**2
                    value[1] += v_error**2
                    value[2] += theta_error
                    value[3] += vrz_error
                    value[4] += 1
                    for category, label in labels.items():
                        category_value = category_totals[category][label][name][horizon]
                        category_value[0] += p_error**2
                        category_value[1] += v_error**2
                        category_value[2] += 1

    def finalize(values: dict[str, dict[int, list[float]]]) -> dict[str, Any]:
        result = {}
        for name, by_horizon in values.items():
            result[name] = {}
            for horizon, value in by_horizon.items():
                count = value[-1]
                result[name][f"h{horizon}"] = {
                    "pb_rmse_m": math.sqrt(value[0] / count),
                    "vb_rmse_mps": math.sqrt(value[1] / count),
                    "theta_bias_rad": value[2] / count if len(value) == 5 else None,
                    "vrz_bias_mps": value[3] / count if len(value) == 5 else None,
                    "count": count,
                }
        return result

    overall_by_model = finalize(totals)
    overall: dict[str, Any] = {}
    for horizon in horizons:
        key = f"h{horizon}"
        overall[key] = {
            "horizon_s": horizon / 60.0,
            **{name: values[key] for name, values in overall_by_model.items()},
        }
        nominal = overall[key]["nominal"]
        rbf = overall[key]["rbf"]
        overall[key]["improvement"] = {
            "pb_mse_fraction": 1.0 - (rbf["pb_rmse_m"] / max(nominal["pb_rmse_m"], 1e-12)) ** 2,
            "vb_mse_fraction": 1.0 - (rbf["vb_rmse_mps"] / max(nominal["vb_rmse_mps"], 1e-12)) ** 2,
        }
    finalized_categories: dict[str, Any] = {}
    for category, labels in category_totals.items():
        finalized_categories[category] = {}
        for label, values in labels.items():
            finalized = finalize(values)
            finalized_categories[category][label] = {}
            for horizon in horizons:
                key = f"h{horizon}"
                finalized_categories[category][label][key] = {
                    name: item[key] for name, item in finalized.items()
                }
    return {"overall": overall, **finalized_categories}


def _derivative_metrics(model: Any) -> dict[str, Any]:
    pb = torch.linspace(0.0, 0.7, 71, dtype=torch.float64)
    vb = torch.linspace(-0.8, 0.8, 81, dtype=torch.float64)
    gp, gv = torch.meshgrid(pb, vb, indexing="ij")
    points = torch.stack((gp.reshape(-1), gv.reshape(-1)), -1).requires_grad_(True)
    values = model(points)
    gradient = torch.autograd.grad(values.sum(), points)[0].detach().numpy()
    rp, rv = gradient[:, 0], gradient[:, 1]
    # Full feature derivative verifies that ignored exogenous inputs are exact zeros.
    full = torch.zeros((512, 5), dtype=torch.float64, requires_grad=True)
    full_values = model(full)
    full_gradient = torch.autograd.grad(full_values.sum(), full)[0].detach().numpy()
    return {
        "rp_min_sinv2": float(rp.min()),
        "rp_p01_sinv2": float(np.percentile(rp, 1)),
        "rp_p99_sinv2": float(np.percentile(rp, 99)),
        "rp_max_sinv2": float(rp.max()),
        "rp_abs_p99_sinv2": float(np.percentile(np.abs(rp), 99)),
        "rp_abs_max_sinv2": float(np.max(np.abs(rp))),
        "rv_min_sinv": float(rv.min()),
        "rv_p01_sinv": float(np.percentile(rv, 1)),
        "rv_p50_sinv": float(np.percentile(rv, 50)),
        "rv_p99_sinv": float(np.percentile(rv, 99)),
        "rv_max_sinv": float(rv.max()),
        "direct_theta_residual_gradient_max": float(np.max(np.abs(full_gradient[:, 2]))),
        "direct_omega_residual_gradient_max": float(np.max(np.abs(full_gradient[:, 3]))),
        "direct_anom_residual_gradient_max": float(np.max(np.abs(full_gradient[:, 4]))),
        "all_finite": bool(np.isfinite(gradient).all()),
    }


def _sensitivity_metrics(model: Any, store: WindowStore, refs: Sequence[WindowReference], config: Config) -> dict[str, Any]:
    selected = list(refs)
    if len(selected) > 64:
        indices = np.linspace(0, len(selected) - 1, 64, dtype=int)
        selected = [selected[index] for index in indices]
    result = {}
    for horizon in (6, 12, 33):
        cosines, ratios = [], []
        for offset in range(0, len(selected), 16):
            part = selected[offset : offset + 16]
            batch = store.make_batch(part, torch.device("cpu"))
            commands = batch.commands[:, :horizon].detach().clone().requires_grad_(True)
            rbf = rollout_causal_dynamics(
                model, batch.causal_state[:, 0], commands, batch.dt[:, :horizon],
                batch.constants, batch.geometry, batch.response
            )
            rbf_p = torch.autograd.grad(rbf.states[:, -1, 0].sum(), commands, retain_graph=True)[0]
            rbf_v = torch.autograd.grad(rbf.states[:, -1, 1].sum(), commands)[0]
            rbf_s = torch.cat((rbf_p, rbf_v), dim=-1)
            nominal_commands = batch.commands[:, :horizon].detach().clone().requires_grad_(True)
            nominal = rollout_causal_dynamics(
                None, batch.causal_state[:, 0], nominal_commands, batch.dt[:, :horizon],
                batch.constants, batch.geometry, batch.response
            )
            nom_p = torch.autograd.grad(nominal.states[:, -1, 0].sum(), nominal_commands, retain_graph=True)[0]
            nom_v = torch.autograd.grad(nominal.states[:, -1, 1].sum(), nominal_commands)[0]
            nom_s = torch.cat((nom_p, nom_v), dim=-1)
            rnorm = torch.linalg.vector_norm(rbf_s, dim=-1)
            nnorm = torch.linalg.vector_norm(nom_s, dim=-1).clamp_min(1.0e-15)
            cosine = torch.nn.functional.cosine_similarity(rbf_s, nom_s, dim=-1, eps=1.0e-15)
            cosines.extend(cosine.detach().numpy().tolist())
            ratios.extend((rnorm / nnorm).detach().numpy().tolist())
        result[f"h{horizon}"] = {
            "cosine_p01": float(np.percentile(cosines, 1)),
            "cosine_min": float(np.min(cosines)),
            "norm_ratio_p01": float(np.percentile(ratios, 1)),
            "norm_ratio_p99": float(np.percentile(ratios, 99)),
            "near_zero_rate": float(np.mean(np.asarray(ratios) <= 0.1)),
            "sign_flip_rate": float(np.mean(np.asarray(cosines) <= 0.0)),
            "count": len(ratios),
        }
    return result


def _acceptance(split: str, point: Mapping[str, Any], integral: Mapping[str, Any], rollout: Mapping[str, Any], derivatives: Mapping[str, Any], sensitivity: Mapping[str, Any], config: Config) -> dict[str, Any]:
    accept = config.acceptance
    gates: dict[str, bool] = {
        "finite_outputs_gradients_rollout": bool(
            point["diagnostics"]["outputs_finite"] and derivatives["all_finite"]
        ),
        "output_cap": point["diagnostics"]["max_abs_output_mps2"]
        <= point["diagnostics"]["output_limit_mps2"] + 1.0e-12,
        "direct_control_channel": max(
            derivatives["direct_theta_residual_gradient_max"],
            derivatives["direct_omega_residual_gradient_max"],
            derivatives["direct_anom_residual_gradient_max"],
        )
        <= accept.direct_channel_tolerance,
    }
    if split != "stress":
        gates.update(
            {
                "point_not_worse_than_nominal": point["overall"]["rbf"]["rmse"]
                <= point["overall"]["nominal"]["rmse"],
                "integral_improvement": all(
                    item["improvement_fraction"] >= accept.integral_improvement_min
                    for item in integral["overall"].values()
                ),
                "h12_h33_not_worse": all(
                    rollout["overall"][f"h{h}"]["improvement"][state] >= 0.0
                    for h in (12, 33)
                    for state in ("pb_mse_fraction", "vb_mse_fraction")
                ),
                "dynamic_groups_no_large_degradation": all(
                    (
                        values["h33"]["rbf"][state]
                        / max(values["h33"]["nominal"][state], 1.0e-12)
                    )
                    ** 2
                    <= 1.0 + accept.per_dynamic_degradation_max
                    for group, values in rollout["by_group"].items()
                    if not group.startswith("constant")
                    for state in ("pb_rmse_m", "vb_rmse_mps")
                ),
                "saturation_rate": point["diagnostics"]["saturation_rate"]
                <= accept.saturation_rate_max,
                "static_total_acceleration": point["diagnostics"]["static_total_acc_median_mps2"]
                <= accept.static_total_acc_max_mps2,
                "rv_percentiles": derivatives["rv_p01_sinv"] >= accept.rv_p01_min_sinv
                and derivatives["rv_p99_sinv"] <= accept.rv_p99_max_sinv,
                "rv_extrema": derivatives["rv_min_sinv"] >= accept.rv_min_sinv
                and derivatives["rv_max_sinv"] <= accept.rv_max_sinv,
                "rp_limits": derivatives["rp_abs_p99_sinv2"] <= accept.rp_p99_abs_max_sinv2
                and derivatives["rp_abs_max_sinv2"] <= accept.rp_abs_max_sinv2,
            }
        )
    else:
        gates["stress_h33_no_large_degradation"] = all(
            rollout["overall"]["h33"]["rbf"][state]
            / max(rollout["overall"]["h33"]["nominal"][state], 1.0e-12)
            <= 1.0 + accept.stress_rollout_degradation_max
            for state in ("pb_rmse_m", "vb_rmse_mps")
        )
    gates["control_sensitivity"] = all(
        item["cosine_p01"] >= accept.sensitivity_cosine_p01_min
        and item["norm_ratio_p01"] >= accept.sensitivity_ratio_p01_min
        and item["norm_ratio_p99"] <= accept.sensitivity_ratio_p99_max
        and item["sign_flip_rate"] == 0.0
        for item in sensitivity.values()
    )
    return {"passed": bool(all(gates.values())), "gates": gates}


def evaluate_model(config: Config, run_dir: str | Path | None = None, *, include_holdout: bool = True) -> dict[str, Any]:
    resolved = resolve_run_dir(config, run_dir)
    model, checkpoint = load_checkpoint(resolved / "best_checkpoint.pt")
    model.eval()
    manifest = load_manifest(config)
    if checkpoint["dataset_hash"] != manifest["dataset_hash"]:
        raise RuntimeError("Checkpoint and RBF dataset hashes differ.")
    baseline = None
    if config.paths.baseline_weights and config.paths.baseline_weights.is_file():
        baseline, _ = load_mlp_weights(config.paths.baseline_weights)
        baseline.eval()
    splits = ["validation"] + (["test", "stress"] if include_holdout else [])
    summary: dict[str, Any] = {
        "schema_version": 1,
        "model_id": "acceleration_residual_rbf2_pv_v1",
        "run_dir": str(resolved),
        "config_hash": config.config_hash,
        "dataset_hash": manifest["dataset_hash"],
        "selected_stage": checkpoint["stage"],
        "selected_epoch": checkpoint["epoch"],
        "effective_cap_mps2": model.output_limit_mps2,
        "split_counts": manifest["split_counts"],
        "splits": {},
    }
    derivatives = _derivative_metrics(model)
    for split in splits:
        trials = load_trials(config, split)
        store = WindowStore(trials, max(config.legacy.data.evaluation_steps), config, manifest)
        refs = _refs_for_evaluation(store)
        point = _point_metrics(model, baseline, trials, manifest, config)
        integral = _integral_metrics(model, store, refs, config)
        rollout = _rollout_metrics(model, baseline, store, refs, config)
        sensitivity = _sensitivity_metrics(model, store, refs, config)
        acceptance = _acceptance(
            split, point, integral, rollout, derivatives, sensitivity, config
        )
        response = rollout["overall"]["h60"]["rbf"]
        response_gate = {
            "theta_bias_limit_rad": 0.005,
            "vrz_bias_limit_mps": 0.01,
            "theta_bias_rad": response["theta_bias_rad"],
            "vrz_bias_mps": response["vrz_bias_mps"],
            "passed": abs(response["theta_bias_rad"]) <= 0.005
            and abs(response["vrz_bias_mps"]) <= 0.01,
        }
        queue_alignment = command_queue_alignment(trials)
        summary["splits"][split] = {
            "trial_count": len(trials),
            "window_count": len(refs),
            "point": point,
            "integral": integral,
            "rollout": rollout,
            "derivatives": derivatives,
            "sensitivity": sensitivity,
            "response_chain_gate": response_gate,
            "command_queue_alignment": queue_alignment,
            "acceptance": acceptance,
        }
    required = [name for name in ("validation", "test", "stress") if name in summary["splits"]]
    integration_eligible = len(required) == 3 and all(
        summary["splits"][name]["acceptance"]["passed"] for name in required
    )
    response_ok = all(
        summary["splits"][name]["response_chain_gate"]["passed"]
        for name in required
        if name != "stress"
    )
    queue_ok = all(
        summary["splits"][name]["command_queue_alignment"]["passed"]
        for name in required
    )
    summary["eligible_for_residual_integration"] = integration_eligible
    # This workflow deliberately does not implement C++/CasADi parity.
    summary["eligible_for_real_world_deployment"] = False
    summary["deployment_blockers"] = [
        *([] if integration_eligible else ["offline residual hard gates failed"]),
        *([] if response_ok else ["response-chain gate failed"]),
        *([] if queue_ok else ["command queue alignment failed"]),
        "C++/CasADi deployment parity not implemented in this offline scope",
    ]
    config.paths.results_dir.mkdir(parents=True, exist_ok=True)
    _write_json(config.paths.results_dir / "summary.json", summary)
    _write_metrics_csv(config.paths.results_dir / "metrics.csv", summary)
    _write_json(resolved / "evaluation_summary.json", summary)
    portable_path = resolved / "model.json"
    with portable_path.open("r", encoding="utf-8") as stream:
        portable = json.load(stream)
    portable["metadata"].update(
        {
            "eligible_for_residual_integration": integration_eligible,
            "eligible_for_real_world_deployment": False,
            "eligibility_pending_evaluation": False,
            "acceptance_summary": {
                name: summary["splits"][name]["acceptance"] for name in required
            },
            "response_chain_summary": {
                name: summary["splits"][name]["response_chain_gate"]
                for name in required
            },
            "command_queue_alignment": {
                name: summary["splits"][name]["command_queue_alignment"]
                for name in required
            },
            "deployment_blockers": summary["deployment_blockers"],
        }
    )
    save_portable_json(portable_path, portable)
    summary["portable_model_sha256"] = _sha256(portable_path)
    _write_json(config.paths.results_dir / "summary.json", summary)
    _write_json(resolved / "evaluation_summary.json", summary)
    _plot_results(summary, config.paths.results_dir / "figures")
    return summary


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _write_metrics_csv(path: Path, summary: Mapping[str, Any]) -> None:
    rows = []
    for split, values in summary["splits"].items():
        rows.extend(
            [
                {"split": split, "category": "point", "metric": "nominal_rmse", "value": values["point"]["overall"]["nominal"]["rmse"]},
                {"split": split, "category": "point", "metric": "rbf_rmse", "value": values["point"]["overall"]["rbf"]["rmse"]},
                {"split": split, "category": "rollout", "metric": "h33_pb_rmse", "value": values["rollout"]["overall"]["h33"]["rbf"]["pb_rmse_m"]},
                {"split": split, "category": "rollout", "metric": "h33_vb_rmse", "value": values["rollout"]["overall"]["h33"]["rbf"]["vb_rmse_mps"]},
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("split", "category", "metric", "value"))
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _plot_results(summary: Mapping[str, Any], figures_dir: Path) -> None:
    import os

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/aerial_balance_rbf_mpl")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures_dir.mkdir(parents=True, exist_ok=True)
    validation = summary["splits"]["validation"]
    history_path = Path(summary["run_dir"]) / "history.csv"
    if history_path.is_file():
        with history_path.open("r", encoding="utf-8") as stream:
            history = list(csv.DictReader(stream))
        fig, axis = plt.subplots(figsize=(7, 3.5))
        for stage in ("stage1", "stage2"):
            rows = [row for row in history if row["stage"] == stage]
            if rows:
                axis.plot(
                    [int(row["epoch"]) for row in rows],
                    [float(row["validation_objective"]) for row in rows],
                    label=stage,
                )
        axis.set(xlabel="epoch", ylabel="validation objective")
        axis.grid(alpha=0.25)
        axis.legend()
        fig.tight_layout()
        fig.savefig(figures_dir / "training_history.png", dpi=160)
        plt.close(fig)

    point = validation["point"]["overall"]
    point_names = ["nominal", "RBF"]
    point_values = [point["nominal"]["rmse"], point["rbf"]["rmse"]]
    if point.get("baseline_mlp") is not None:
        point_names.append("CAP0P25 MLP")
        point_values.append(point["baseline_mlp"]["rmse"])
    fig, axis = plt.subplots(figsize=(6, 3.5))
    axis.bar(point_names, point_values)
    axis.set(ylabel="point residual RMSE (m/s²)")
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures_dir / "point_validation.png", dpi=160)
    plt.close(fig)

    integral = validation["integral"]["overall"]
    integral_horizons = [3, 6, 12]
    x = np.arange(len(integral_horizons))
    fig, axis = plt.subplots(figsize=(6, 3.5))
    axis.bar(x - 0.18, [integral[f"h{h}"]["nominal_rmse_mps2"] for h in integral_horizons], 0.36, label="nominal")
    axis.bar(x + 0.18, [integral[f"h{h}"]["rbf_rmse_mps2"] for h in integral_horizons], 0.36, label="RBF")
    axis.set(xticks=x, xticklabels=[f"h{h}" for h in integral_horizons], ylabel="equivalent acceleration RMSE (m/s²)")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "integral_validation.png", dpi=160)
    plt.close(fig)
    horizons = [6, 12, 30, 33, 60]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    for axis, state, label in zip(axes, ("pb_rmse_m", "vb_rmse_mps"), ("pb RMSE (m)", "vb RMSE (m/s)")):
        axis.plot(horizons, [validation["rollout"]["overall"][f"h{h}"]["nominal"][state] for h in horizons], marker="o", label="nominal")
        axis.plot(horizons, [validation["rollout"]["overall"][f"h{h}"]["rbf"][state] for h in horizons], marker="o", label="RBF")
        axis.set(xlabel="horizon steps", ylabel=label)
        axis.grid(alpha=0.25)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "rollout_validation.png", dpi=160)
    plt.close(fig)

    derivative = validation["derivatives"]
    fig, axis = plt.subplots(figsize=(6, 3.5))
    labels = ["|rp| p99", "|rp| max", "rv p01", "rv p99"]
    values = [derivative["rp_abs_p99_sinv2"], derivative["rp_abs_max_sinv2"], derivative["rv_p01_sinv"], derivative["rv_p99_sinv"]]
    axis.bar(labels, values)
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.tick_params(axis="x", rotation=20)
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures_dir / "derivative_summary.png", dpi=160)
    plt.close(fig)

    sensitivity = validation["sensitivity"]
    keys = ["h6", "h12", "h33"]
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
    axes[0].bar(keys, [sensitivity[key]["cosine_p01"] for key in keys])
    axes[0].set(ylabel="cosine P1", ylim=(0.95, 1.00001))
    axes[1].bar(keys, [sensitivity[key]["norm_ratio_p01"] for key in keys], label="P1")
    axes[1].scatter(keys, [sensitivity[key]["norm_ratio_p99"] for key in keys], color="tab:red", label="P99")
    axes[1].set(ylabel="sensitivity norm ratio")
    axes[1].legend()
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures_dir / "control_sensitivity.png", dpi=160)
    plt.close(fig)
