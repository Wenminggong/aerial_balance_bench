"""Controlled Jacobian-limit screening for the causal rollout training variant."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import AppConfig, with_training_overrides
from .evaluation import evaluate_model
from .data import load_manifest
from .training import train_model


SCREENING_CANDIDATES = (
    ("jacobian_off", 0.0, 0.02),
    ("lmax_0p01", 0.1, 0.01),
    ("lmax_0p02", 0.1, 0.02),
    ("lmax_0p04", 0.1, 0.04),
)
CHANNEL_CANDIDATES = (
    ("channel_off", 0.0),
    ("channel_1", 1.0),
    ("channel_10", 10.0),
)
REPEAT_SEEDS = (7, 17, 27)
SCREEN_STEPS_PER_EPOCH = 16
SCREEN_MAX_EPOCHS = 80
SCREEN_PATIENCE = 20
SCREEN_VALIDATION_WINDOWS = 1024


def run_screening(config: AppConfig) -> dict[str, Any]:
    if not config.causal_rollout.enabled:
        raise ValueError("Jacobian screening requires causal_rollout.enabled=true.")
    if config.control_channel.enabled:
        return _run_channel_screening(config)
    policy = _selection_policy(config)
    initial: list[dict[str, Any]] = []
    for name, weight, limit in SCREENING_CANDIDATES:
        variant = _small_screen_config(with_training_overrides(
            config,
            seed=7,
            jacobian_weight=weight,
            jacobian_norm_limit_mps2=limit,
        ))
        run_dir, summary = _train_or_reuse(variant)
        initial.append(
            _candidate_record(name, weight, limit, 7, run_dir, summary, variant)
        )

    eligible = [record for record in initial if record["hard_filters_passed"]]
    selection_pool = eligible
    if not selection_pool and policy["allow_ineligible_fallback"]:
        selection_pool = initial
    selected = (
        min(selection_pool, key=lambda item: item["h33_causal_score"])
        if selection_pool
        else None
    )
    repetitions: list[dict[str, Any]] = []
    if selected is not None:
        for seed in policy["repeat_seeds"]:
            variant = with_training_overrides(
                config,
                seed=seed,
                jacobian_weight=selected["jacobian_weight"],
                jacobian_norm_limit_mps2=selected["jacobian_norm_limit_mps2"],
            )
            run_dir, summary = _train_or_reuse(variant)
            repetitions.append(
                _candidate_record(
                    selected["candidate"],
                    selected["jacobian_weight"],
                    selected["jacobian_norm_limit_mps2"],
                    seed,
                    run_dir,
                    summary,
                    variant,
                )
            )

    stability_passed = bool(
        repetitions
        and {record["seed"] for record in repetitions}
        == set(policy["repeat_seeds"])
        and all(record["hard_filters_passed"] for record in repetitions)
    )
    selected_run = None
    if repetitions and (stability_passed or policy["allow_ineligible_fallback"]):
        selected_run = min(repetitions, key=lambda item: item["h33_causal_score"])
    final_summary: dict[str, Any] | None = None
    if selected_run is not None and policy["evaluate_holdout_for_selected"]:
        final_config = with_training_overrides(
            config,
            seed=selected_run["seed"],
            jacobian_weight=selected_run["jacobian_weight"],
            jacobian_norm_limit_mps2=selected_run["jacobian_norm_limit_mps2"],
        )
        final_summary = evaluate_model(
            final_config,
            selected_run["run_dir"],
            include_holdout=True,
        )
    export_eligible = bool(
        final_summary is not None and final_summary.get("export_eligible", False)
    )
    forced_selection = bool(
        selected_run is not None
        and (not selected["hard_filters_passed"] or not stability_passed)
    )
    result = {
        "schema_version": 1,
        "selection_rule": (
            "Reject Jacobian/residual/dynamic/response failures, then minimize the "
            "normalized h33 causal pb/vb rollout score. If explicitly enabled and no "
            "candidate passes, select the lowest-score candidate as a diagnostic model."
        ),
        "selection_policy": policy,
        "initial_seed": 7,
        "screening_budget": {
            "steps_per_epoch": SCREEN_STEPS_PER_EPOCH,
            "max_epochs": SCREEN_MAX_EPOCHS,
            "early_stopping_patience": SCREEN_PATIENCE,
            "validation_max_windows": SCREEN_VALIDATION_WINDOWS,
        },
        "v1_baseline": _load_v1_baseline(config),
        "initial_candidates": initial,
        "screening_selected_candidate": (
            None if selected is None else selected["candidate"]
        ),
        "selected_candidate": None if selected_run is None else selected_run["candidate"],
        "selected_screening_run_dir": None if selected is None else selected["run_dir"],
        "selected_run_dir": None if selected_run is None else selected_run["run_dir"],
        "selected_seed": None if selected_run is None else selected_run["seed"],
        "seed_repetitions": repetitions,
        "seed_stability_passed": stability_passed,
        "forced_selection": forced_selection,
        "final_evaluation": final_summary,
        "test_and_stress_passed": export_eligible,
        "export_blocked": not export_eligible,
    }
    output = config.paths.results_dir / "screening_summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return result


def _selection_policy(config: AppConfig) -> dict[str, Any]:
    """Parse optional screening overrides while preserving the legacy protocol."""

    raw = config.raw.get("screening", {})
    if not isinstance(raw, dict):
        raise ValueError("screening must be a mapping when provided.")
    repeat_seeds_raw = raw.get("repeat_seeds", REPEAT_SEEDS)
    if not isinstance(repeat_seeds_raw, (list, tuple)) or not repeat_seeds_raw:
        raise ValueError("screening.repeat_seeds must be a non-empty sequence.")
    repeat_seeds = tuple(int(seed) for seed in repeat_seeds_raw)
    if len(repeat_seeds) != len(set(repeat_seeds)):
        raise ValueError("screening.repeat_seeds must contain unique seeds.")
    return {
        "allow_ineligible_fallback": bool(
            raw.get("allow_ineligible_fallback", False)
        ),
        "repeat_seeds": list(repeat_seeds),
        "evaluate_holdout_for_selected": bool(
            raw.get("evaluate_holdout_for_selected", False)
        ),
    }


def _run_channel_screening(config: AppConfig) -> dict[str, Any]:
    initial: list[dict[str, Any]] = []
    for name, theta_weight in CHANNEL_CANDIDATES:
        variant = _small_screen_config(
            with_training_overrides(
                config,
                seed=7,
                theta_channel_weight=theta_weight,
            )
        )
        run_dir, summary = _train_or_reuse(variant)
        initial.append(
            _channel_candidate_record(
                name, theta_weight, 7, run_dir, summary, screening=True
            )
        )

    constrained = [record for record in initial if record["theta_channel_weight"] > 0.0]
    if constrained and not any(
        record["hard_filters"].get("theta_sector", False)
        and record["hard_filters"].get("outer_theta_sector", False)
        for record in constrained
    ):
        variant = _small_screen_config(
            with_training_overrides(
                config, seed=7, theta_channel_weight=30.0
            )
        )
        run_dir, summary = _train_or_reuse(variant)
        initial.append(
            _channel_candidate_record(
                "channel_30", 30.0, 7, run_dir, summary, screening=True
            )
        )

    eligible = [
        record
        for record in initial
        if record["theta_channel_weight"] > 0.0 and record["hard_filters_passed"]
    ]
    screened = min(eligible, key=lambda item: item["h33_causal_score"]) if eligible else None
    repetitions: list[dict[str, Any]] = []
    if screened is not None:
        for seed in REPEAT_SEEDS:
            variant = with_training_overrides(
                config,
                seed=seed,
                theta_channel_weight=screened["theta_channel_weight"],
            )
            run_dir, summary = _train_or_reuse(variant)
            repetitions.append(
                _channel_candidate_record(
                    screened["candidate"],
                    screened["theta_channel_weight"],
                    seed,
                    run_dir,
                    summary,
                    screening=False,
                )
            )
    stability_passed = bool(
        repetitions
        and {record["seed"] for record in repetitions} == set(REPEAT_SEEDS)
        and all(record["hard_filters_passed"] for record in repetitions)
    )
    selected_seed = (
        min(repetitions, key=lambda item: item["h33_causal_score"])
        if stability_passed
        else None
    )
    final_summary: dict[str, Any] | None = None
    if selected_seed is not None:
        final_config = with_training_overrides(
            config,
            seed=selected_seed["seed"],
            theta_channel_weight=selected_seed["theta_channel_weight"],
        )
        final_summary = evaluate_model(
            final_config,
            selected_seed["run_dir"],
            include_holdout=True,
        )
    export_eligible = bool(final_summary and final_summary.get("export_eligible"))
    result = {
        "schema_version": 2,
        "selection_rule": (
            "Apply validation safety gates, exclude channel_off, minimize normalized "
            "validation h33 causal pb/vb score, require all three full-budget seeds, "
            "then run the one-shot test and stress gates."
        ),
        "initial_seed": 7,
        "screening_budget": {
            "steps_per_epoch": SCREEN_STEPS_PER_EPOCH,
            "max_epochs": SCREEN_MAX_EPOCHS,
            "early_stopping_patience": SCREEN_PATIENCE,
            "validation_max_windows": SCREEN_VALIDATION_WINDOWS,
        },
        "initial_candidates": initial,
        "screening_selected_candidate": (
            None if screened is None else screened["candidate"]
        ),
        "selected_candidate": (
            selected_seed["candidate"] if export_eligible else None
        ),
        "selected_screening_run_dir": None if screened is None else screened["run_dir"],
        "selected_run_dir": None if not export_eligible else selected_seed["run_dir"],
        "selected_seed": None if selected_seed is None else selected_seed["seed"],
        "seed_repetitions": repetitions,
        "seed_stability_passed": stability_passed,
        "test_and_stress_passed": export_eligible,
        "final_evaluation": final_summary,
        "export_blocked": not export_eligible,
    }
    output = config.paths.results_dir / "screening_summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return result


def _small_screen_config(config: AppConfig) -> AppConfig:
    return with_training_overrides(
        config,
        steps_per_epoch=SCREEN_STEPS_PER_EPOCH,
        max_epochs=SCREEN_MAX_EPOCHS,
        early_stopping_patience=SCREEN_PATIENCE,
        validation_max_windows=SCREEN_VALIDATION_WINDOWS,
    )


def _train_or_reuse(config: AppConfig) -> tuple[Path, dict[str, Any]]:
    """Reuse only an exact config/dataset evaluation; otherwise train afresh."""

    dataset_hash = load_manifest(config)["dataset_hash"]
    if config.paths.training_runs_dir.is_dir():
        for summary_path in sorted(
            config.paths.training_runs_dir.glob("*/training_summary.json"), reverse=True
        ):
            try:
                training = json.loads(summary_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            run_dir = summary_path.parent
            evaluation_path = run_dir / "evaluation_summary.json"
            if (
                training.get("config_hash") == config.config_hash
                and training.get("dataset_hash") == dataset_hash
                and evaluation_path.is_file()
            ):
                evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
                if evaluation.get("config_hash") == config.config_hash:
                    return run_dir, evaluation
    run_dir = train_model(config)
    return run_dir, evaluate_model(config, run_dir)


def _load_v1_baseline(config: AppConfig) -> dict[str, Any] | None:
    summary_path = config.config_path.parents[1] / "results" / "summary.json"
    if not summary_path.is_file():
        return None
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    point = summary["point"]["overall"]["mlp"]
    h30 = summary["rollout"]["overall"]["h30"]
    return {
        "run_name": summary["run_name"],
        "seed": summary["seed"],
        "residual_rmse_mps2": point["rmse"],
        "h30_pb_rmse_m": h30["mlp"]["pb_rmse_m"],
        "h30_vb_rmse_mps": h30["mlp"]["vb_rmse_mps"],
        "note": "Teacher-forced v1.0 baseline; target semantics differ from causal rollout.",
    }


def _candidate_record(
    name: str,
    weight: float,
    limit: float,
    seed: int,
    run_dir: Path,
    summary: dict[str, Any],
    config: AppConfig,
) -> dict[str, Any]:
    diagnostics = summary["diagnostics"]
    point_rmse = float(summary["point"]["overall"]["mlp"]["rmse"])
    h33 = summary["rollout"]["overall"]["h33"]["causal"]["mlp"]
    dynamic_ok = all(summary["acceptance"]["dynamic_group_checks"].values())
    response_ok = summary["acceptance"]["checks"][
        "response_chain_no_systematic_bias"
    ]
    filters = {
        "jacobian_p99": bool(diagnostics["jacobian_p99_within_limit"]),
        "jacobian_max": bool(diagnostics["jacobian_max_within_limit"]),
        "residual_rmse": point_rmse
        <= config.acceptance.residual_rmse_max_mps2,
        "dynamic_rollout_degradation": bool(dynamic_ok),
        "response_chain": bool(response_ok),
        "command_queue_alignment": bool(
            diagnostics["command_queue_alignment"]["passed"]
        ),
    }
    score = (
        (float(h33["pb_rmse_m"]) / 0.01) ** 2
        + (float(h33["vb_rmse_mps"]) / 0.05) ** 2
    )
    return {
        "candidate": name,
        "seed": seed,
        "jacobian_weight": weight,
        "jacobian_norm_limit_mps2": limit,
        "run_dir": str(run_dir),
        "run_name": summary["run_name"],
        "hard_filters": filters,
        "hard_filters_passed": bool(all(filters.values())),
        "h33_causal_score": score,
        "h33_causal_pb_rmse_m": h33["pb_rmse_m"],
        "h33_causal_vb_rmse_mps": h33["vb_rmse_mps"],
        "residual_rmse_mps2": point_rmse,
        "jacobian_p99_mps2": diagnostics["jacobian_norm_p99_mps2"],
        "jacobian_max_mps2": diagnostics["jacobian_norm_max_mps2"],
    }


def _channel_candidate_record(
    name: str,
    theta_weight: float,
    seed: int,
    run_dir: Path,
    summary: dict[str, Any],
    *,
    screening: bool,
) -> dict[str, Any]:
    diagnostics = summary["diagnostics"]
    point_rmse = float(summary["point"]["overall"]["mlp"]["rmse"])
    h33 = summary["rollout"]["overall"]["h33"]["causal"]["mlp"]
    checks = dict(summary["acceptance"]["checks"])
    safety_names = (
        "residual_rmse",
        "all_dynamic_groups_no_large_degradation",
        "residual_saturation",
        "finite_outputs_and_gradients",
        "jacobian_p99_within_limit",
        "jacobian_max_within_limit",
        "command_queue_alignment",
        "response_chain_no_systematic_bias",
        "theta_sector",
        "anti_damping",
        "outer_safety_set_finite",
        "outer_theta_sector",
        "outer_anti_damping",
        "control_sensitivity",
    )
    filters = {name: bool(checks.get(name, False)) for name in safety_names}
    # The small budget is still required to improve h33 and the static regime; these
    # are model-quality gates, not test-set feedback.
    filters["h33_position_improvement"] = bool(
        checks.get("h33_position_improvement", False)
    )
    filters["h33_velocity_improvement"] = bool(
        checks.get("h33_velocity_improvement", False)
    )
    filters["static_total_acceleration"] = bool(
        checks.get("static_total_acceleration", False)
    )
    score = (
        (float(h33["pb_rmse_m"]) / 0.01) ** 2
        + (float(h33["vb_rmse_mps"]) / 0.05) ** 2
    )
    channel = diagnostics["channel"]
    return {
        "candidate": name,
        "seed": seed,
        "screening_budget": screening,
        "theta_channel_weight": theta_weight,
        "run_dir": str(run_dir),
        "run_name": summary["run_name"],
        "hard_filters": filters,
        "hard_filters_passed": bool(all(filters.values())),
        "h33_causal_score": score,
        "h33_causal_pb_rmse_m": h33["pb_rmse_m"],
        "h33_causal_vb_rmse_mps": h33["vb_rmse_mps"],
        "residual_rmse_mps2": point_rmse,
        "theta_ratio_p01": channel["theta_ratio_p01"],
        "theta_ratio_p99": channel["theta_ratio_p99"],
        "anti_damping_p99_sinv": channel["anti_damping_p99_sinv"],
        "anti_damping_max_sinv": channel["anti_damping_max_sinv"],
    }
