#!/usr/bin/env python3
"""Run one recoverable acados NMPC objective-weight tuning trial per invocation."""

from __future__ import annotations

import argparse
import csv
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Iterable, Mapping

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_PARENT = PROJECT_ROOT.parent
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from aerial_balance_bench.utils.acados_nmpc_tuning import (  # noqa: E402
    ObjectiveCandidate,
    TUNING_TRAJECTORY_TYPES,
    analyze_acados_rollout,
    compare_screen_candidate,
    evaluate_validation,
    leaderboard_key,
    screen_rejection_reasons,
    sha256_file,
)


DEFAULT_CONFIG = (
    PROJECT_ROOT / "baselines" / "configs" / "acados_nmpc_objective_tuning.yaml"
)
EVAL_SCRIPT = PROJECT_ROOT / "scripts" / "acados_nmpc_unified_eval.py"
STATE_VERSION = 1
PROTOCOL_LENGTH = 11
FIXED_ENV_CONFIG = PROJECT_ROOT / "environments" / "configs" / "unified_tracking_mixed.yaml"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--stage", choices=("next", "validate"), default="next")
    parser.add_argument(
        "--candidate",
        type=Path,
        help="Full stage_weights/terminal_weights YAML replacing the automatic recommendation.",
    )
    parser.add_argument("--analyze-run", type=Path, help="Analyze one existing run and exit.")
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse an already completed deterministic run directory.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    return parser.parse_args()


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with open(Path(path).expanduser().resolve(), encoding="utf-8") as stream:
        return yaml.safe_load(stream) or {}


def _save_yaml(data: Mapping[str, Any], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as stream:
        yaml.safe_dump(dict(data), stream, sort_keys=False)
    os.replace(temporary, destination)


def _write_csv(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    materialized = [dict(row) for row in rows]
    if not materialized:
        return
    fieldnames: list[str] = []
    for row in materialized:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(materialized)
    os.replace(temporary, destination)


def _resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _strict_keys(values: Mapping[str, Any], allowed: set[str], section: str) -> None:
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"Unknown field(s) in {section}: {', '.join(unknown)}")


def _validate_tuning_config(config: Mapping[str, Any]) -> None:
    _strict_keys(
        config,
        {
            "name",
            "output_root",
            "env_config",
            "policy_config",
            "fixed",
            "screen",
            "validation",
            "timing",
            "trajectory_types",
            "baseline",
            "screen_gates",
            "validation_gates",
        },
        "tuning config",
    )
    _strict_keys(
        config.get("fixed", {}),
        {
            "n_horizon",
            "rollout_steps",
            "batch_threads",
            "safe_position_min",
            "safe_position_max",
            "weight_min",
            "weight_max",
            "max_screen_trials",
        },
        "fixed",
    )
    _strict_keys(
        config.get("screen", {}),
        {"seed", "num_envs", "target_episodes", "minimum_per_type"},
        "screen",
    )
    _strict_keys(
        config.get("validation", {}),
        {"seed", "num_envs", "target_episodes", "minimum_per_type"},
        "validation",
    )
    _strict_keys(
        config.get("timing", {}),
        {"seed", "num_envs", "max_steps", "warmup_steps", "deadline_s"},
        "timing",
    )
    _strict_keys(
        config.get("screen_gates", {}),
        {
            "solver_success_min",
            "fallback_max",
            "action_saturation_max",
            "tail_action_saturation_max",
            "objective_improvement_min",
            "per_type_p90_regression_max",
        },
        "screen_gates",
    )
    _strict_keys(
        config.get("validation_gates", {}),
        {
            "solver_success_min",
            "fallback_max",
            "action_saturation_max",
            "tail_action_saturation_max",
            "slack_active_max",
            "max_slack",
            "balanced_rmse_improvement_min",
            "worst_p90_improvement_min",
            "constant_steady_mae_max",
            "controller_time_p99_max_s",
        },
        "validation_gates",
    )
    fixed = config.get("fixed", {})
    screen = config.get("screen", {})
    validation = config.get("validation", {})
    if int(fixed.get("n_horizon", -1)) != 30:
        raise ValueError("Objective tuning requires fixed.n_horizon=30.")
    if int(fixed.get("rollout_steps", -1)) != 600:
        raise ValueError("Objective tuning requires fixed.rollout_steps=600.")
    expected_values = {
        "fixed.batch_threads": (fixed.get("batch_threads"), 8),
        "fixed.max_screen_trials": (fixed.get("max_screen_trials"), 14),
        "fixed.weight_min": (float(fixed.get("weight_min", float("nan"))), 1.0e-3),
        "fixed.weight_max": (float(fixed.get("weight_max", float("nan"))), 1.0e3),
        "screen.seed": (screen.get("seed"), 666),
        "screen.num_envs": (screen.get("num_envs"), 48),
        "screen.target_episodes": (screen.get("target_episodes"), 48),
        "screen.minimum_per_type": (screen.get("minimum_per_type"), 8),
        "validation.seed": (validation.get("seed"), 667),
        "validation.num_envs": (validation.get("num_envs"), 96),
        "validation.target_episodes": (validation.get("target_episodes"), 96),
        "validation.minimum_per_type": (validation.get("minimum_per_type"), 20),
    }
    changed = [name for name, (value, expected) in expected_values.items() if value != expected]
    fixed_gate_values = {
        "screen_gates.solver_success_min": 0.99,
        "screen_gates.fallback_max": 0.01,
        "screen_gates.action_saturation_max": 0.20,
        "screen_gates.tail_action_saturation_max": 0.05,
        "screen_gates.objective_improvement_min": 0.03,
        "screen_gates.per_type_p90_regression_max": 0.05,
        "validation_gates.solver_success_min": 0.999,
        "validation_gates.fallback_max": 0.001,
        "validation_gates.action_saturation_max": 0.10,
        "validation_gates.tail_action_saturation_max": 0.02,
        "validation_gates.slack_active_max": 0.001,
        "validation_gates.max_slack": 1.0e-3,
        "validation_gates.balanced_rmse_improvement_min": 0.30,
        "validation_gates.worst_p90_improvement_min": 0.20,
        "validation_gates.constant_steady_mae_max": 0.05,
        "validation_gates.controller_time_p99_max_s": 1.0 / 60.0,
    }
    for dotted_name, expected in fixed_gate_values.items():
        section, name = dotted_name.split(".", 1)
        value = float(config.get(section, {}).get(name, float("nan")))
        if not math.isclose(value, expected, rel_tol=0.0, abs_tol=1.0e-12):
            changed.append(dotted_name)
    if changed:
        raise ValueError(
            "The objective_v1 protocol fixes these fields: " + ", ".join(changed)
        )
    if tuple(config.get("trajectory_types", ())) != TUNING_TRAJECTORY_TYPES:
        raise ValueError(
            "trajectory_types must be exactly constant, random_b_spline, random_ramp_dwell."
        )


def _policy_section(policy: Mapping[str, Any]) -> Mapping[str, Any]:
    return policy.get("acados_nmpc_policy", policy.get("policy", policy))


def _validate_fixed_policy(policy: Mapping[str, Any], expected_horizon: int) -> None:
    section = _policy_section(policy)
    horizon = int(section.get("solver", {}).get("n_horizon", -1))
    if horizon != expected_horizon or horizon != 30:
        raise ValueError(f"Policy solver.n_horizon must remain 30; got {horizon}.")
    ObjectiveCandidate.from_policy(policy)


def _fixed_policy_fingerprint(policy: Mapping[str, Any]) -> str:
    """Hash all policy fields except the two searchable objective arrays."""
    fixed = deepcopy(dict(_policy_section(policy)))
    fixed.pop("objective", None)
    payload = json.dumps(fixed, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _state_paths(root: Path) -> dict[str, Path]:
    return {
        "state": root / "session_state.yaml",
        "history": root / "history.csv",
        "leaderboard_yaml": root / "leaderboard.yaml",
        "leaderboard_csv": root / "leaderboard.csv",
        "recommendation": root / "next_recommendation.yaml",
    }


def _initialize_or_load_session(
    config: Mapping[str, Any],
    config_path: Path,
    env_path: Path,
    policy_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    paths = _state_paths(output_root)
    env_hash = sha256_file(env_path)
    fixed_policy_hash = _fixed_policy_fingerprint(_load_yaml(policy_path))
    if paths["state"].exists():
        state = _load_yaml(paths["state"])
        if int(state.get("version", -1)) != STATE_VERSION:
            raise ValueError("Unsupported objective-tuning session state version.")
        if Path(state["env_config_path"]).resolve() != env_path:
            raise RuntimeError("The session is bound to a different environment config path.")
        if state.get("env_config_sha256") != env_hash:
            raise RuntimeError(
                "unified_tracking_mixed.yaml changed after this tuning session started; "
                "restore it or start a new output_root."
            )
        if int(state.get("n_horizon", -1)) != 30:
            raise RuntimeError("The saved session does not use n_horizon=30.")
        if "fixed_policy_sha256" not in state and not state.get("screen_trials"):
            state["fixed_policy_sha256"] = fixed_policy_hash
            _save_yaml(state, paths["state"])
        if state.get("fixed_policy_sha256") != fixed_policy_hash:
            raise RuntimeError(
                "A non-objective field in the acados policy changed after this tuning "
                "session started; restore it or start a new output_root."
            )
        return state

    configured_baseline = ObjectiveCandidate.from_mapping(config.get("baseline", {}))
    policy_baseline = ObjectiveCandidate.from_policy(_load_yaml(policy_path))
    if configured_baseline != policy_baseline:
        raise ValueError(
            "The configured baseline objective does not match the current policy YAML. "
            "Use a new tuning config/session when changing the formal baseline."
        )
    output_root.mkdir(parents=True, exist_ok=True)
    for directory in ("runs", "generated_configs", "stdout"):
        (output_root / directory).mkdir(parents=True, exist_ok=True)
    state = {
        "version": STATE_VERSION,
        "name": config.get("name", "acados_nmpc_objective_v1"),
        "created_at": _utc_now(),
        "updated_at": _utc_now(),
        "tuning_config_path": str(config_path),
        "env_config_path": str(env_path),
        "env_config_sha256": env_hash,
        "policy_config_path": str(policy_path),
        "fixed_policy_sha256": fixed_policy_hash,
        "n_horizon": 30,
        "protocol_cursor": 0,
        "baseline_candidate": configured_baseline.as_dict(),
        "baseline_candidate_id": configured_baseline.identifier,
        "incumbent_candidate": configured_baseline.as_dict(),
        "incumbent_candidate_id": None,
        "screen_trials": [],
        "validation": {},
        "status": "screening",
        "promoted": False,
    }
    _save_yaml(config, output_root / "tuning_config_snapshot.yaml")
    _save_yaml(state, paths["state"])
    return state


def _record_by_label(state: Mapping[str, Any], label: str) -> Mapping[str, Any] | None:
    for record in reversed(state.get("screen_trials", [])):
        if record.get("label") == label:
            return record
    return None


def _propose_next(state: dict[str, Any]) -> dict[str, Any] | None:
    """Advance skipped diagnostics and return exactly one deterministic proposal."""
    incumbent = ObjectiveCandidate.from_mapping(state["incumbent_candidate"])
    metrics = {}
    if state.get("incumbent_candidate_id"):
        incumbent_record = next(
            record
            for record in state["screen_trials"]
            if record["candidate_id"] == state["incumbent_candidate_id"]
        )
        metrics = incumbent_record["metrics"]

    while int(state.get("protocol_cursor", 0)) < PROTOCOL_LENGTH:
        step = int(state["protocol_cursor"])
        state["protocol_cursor"] = step + 1
        if step == 0:
            return _proposal(incumbent, "velocity", 8.0, "velocity_x8", step)
        if step == 1:
            previous = _record_by_label(state, "velocity_x8")
            factor = 2.0 if previous and previous.get("accepted") else 3.0
            label = "velocity_x16" if factor == 2.0 else "velocity_x3"
            return _proposal(incumbent, "velocity", factor, label, step)
        if step == 2:
            return _proposal(incumbent, "command", 4.0, "command_x4", step)
        if step == 3:
            previous = _record_by_label(state, "command_x4")
            if previous and previous.get("accepted"):
                continue
            return _proposal(incumbent, "command", 2.0, "command_x2_fallback", step)
        if step == 4:
            return _proposal(incumbent, "vrz", 4.0, "vrz_x4", step)
        if step == 5:
            previous = _record_by_label(state, "vrz_x4")
            if previous and previous.get("accepted"):
                continue
            return _proposal(incumbent, "vrz", 2.0, "vrz_x2_fallback", step)
        if step == 6:
            overshoot = float(metrics.get("constant_mean_overshoot", float("nan")))
            saturation = float(metrics.get("action_saturation_rate", float("inf")))
            variation = float(metrics.get("action_variation", float("inf")))
            if math.isfinite(overshoot) and overshoot > 0.02 and saturation <= 0.10:
                return _proposal(incumbent, "action", 0.25, "action_x0p25", step)
            if saturation > 0.10 or variation > 0.01:
                return _proposal(incumbent, "action", 2.0, "action_x2", step)
            continue
        if step == 7:
            return _proposal(incumbent, "omega", 4.0, "omega_x4", step)
        if step == 8:
            crossings = float(metrics.get("constant_mean_zero_crossings", 0.0))
            theta_rms = float(metrics.get("theta_rms", 0.0))
            tail = float(metrics.get("balanced_tail_rmse", 0.0))
            rmse = float(metrics.get("balanced_mean_rmse", float("inf")))
            if crossings > 2.0 or theta_rms > 0.15 or tail >= rmse:
                return _proposal(incumbent, "theta", 2.0, "theta_x2", step)
            continue
        if step == 9:
            stable = (
                not screen_rejection_reasons(metrics)
                and int(metrics.get("termination_count", 1)) == 0
                and int(metrics.get("boundary_violation_count", 1)) == 0
            )
            if not stable:
                continue
            overshoot = float(metrics.get("constant_mean_overshoot", float("nan")))
            steady = float(metrics.get("constant_steady_mae", float("inf")))
            rmse = float(metrics.get("balanced_mean_rmse", float("inf")))
            saturation = float(metrics.get("action_saturation_rate", 1.0))
            if math.isfinite(overshoot) and overshoot > 0.02:
                return _proposal(incumbent, "position", 0.5, "position_x0p5", step)
            if (steady > 0.05 or rmse > 0.08) and saturation < 0.10:
                return _proposal(incumbent, "position", 2.0, "position_x2", step)
            continue
        if step == 10:
            stable = (
                int(metrics.get("termination_count", 1)) == 0
                and int(metrics.get("boundary_violation_count", 1)) == 0
            )
            dynamic_tail = [
                float(metrics.get("per_type", {}).get(name, {}).get("mean_tail_rmse", 0.0))
                for name in TUNING_TRAJECTORY_TYPES[1:]
            ]
            if stable and max(dynamic_tail, default=0.0) > 0.05:
                return _proposal(incumbent, "terminal_all", 2.0, "terminal_all_x2", step)
            continue
    return None


def _proposal(
    base: ObjectiveCandidate,
    group: str,
    factor: float,
    label: str,
    protocol_step: int,
) -> dict[str, Any]:
    candidate = base.scale_group(group, factor)
    return {
        "label": label,
        "group": group,
        "factor": factor,
        "protocol_step": protocol_step,
        "base_candidate_id": base.identifier,
        "candidate_id": candidate.identifier,
        "candidate": candidate.as_dict(),
        "reason": _proposal_reason(group, factor),
    }


def _proposal_reason(group: str, factor: float) -> str:
    reasons = {
        "velocity": "increase ball velocity-error damping",
        "command": "penalize accumulated velocity command",
        "vrz": "penalize modeled response velocity",
        "action": "adapt increment authority from overshoot/saturation diagnostics",
        "omega": "increase beam-rate damping",
        "theta": "increase beam-angle damping after residual oscillation",
        "position": "adjust position aggressiveness after closed-loop stabilization",
        "terminal_all": "test stronger terminal shaping after dynamic-tail stabilization",
    }
    return f"{reasons[group]} (x{factor:g})"


def _flatten_record(record: Mapping[str, Any]) -> dict[str, Any]:
    metrics = record.get("metrics", {})
    candidate = ObjectiveCandidate.from_mapping(record["candidate"])
    row = {
        "trial_index": record.get("trial_index"),
        "trial_id": record.get("trial_id"),
        "label": record.get("label"),
        "candidate_id": record.get("candidate_id"),
        "accepted": record.get("accepted"),
        "decision": record.get("decision", {}).get("reason"),
        "run_dir": record.get("run_dir"),
        "stage_weights": candidate.stage_weights,
        "terminal_weights": candidate.terminal_weights,
    }
    for key, value in metrics.items():
        if not isinstance(value, (dict, list, tuple)):
            row[key] = value
    for trajectory_type, values in metrics.get("per_type", {}).items():
        for key, value in values.items():
            row[f"{trajectory_type}_{key}"] = value
    return row


def _write_session_artifacts(state: dict[str, Any], output_root: Path) -> None:
    state["updated_at"] = _utc_now()
    paths = _state_paths(output_root)
    _save_yaml(state, paths["state"])
    history = [_flatten_record(record) for record in state.get("screen_trials", [])]
    _write_csv(paths["history"], history)
    leaderboard = sorted(state.get("screen_trials", []), key=leaderboard_key)
    compact = [_flatten_record(record) for record in leaderboard]
    _save_yaml({"candidates": compact}, paths["leaderboard_yaml"])
    _write_csv(paths["leaderboard_csv"], compact)
    if state.get("status") == "screening":
        preview_state = deepcopy(state)
        recommendation = _propose_next(preview_state)
        if recommendation is None:
            _save_yaml(
                {"status": "screen_complete", "incumbent": state["incumbent_candidate"]},
                paths["recommendation"],
            )
        else:
            _save_yaml(recommendation, paths["recommendation"])


def _relative_to_project(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _build_run_config(
    *,
    env_path: Path,
    policy_path: Path,
    candidate: ObjectiveCandidate,
    output_root: Path,
    run_name: str,
    seed: int,
    target_episodes: int,
    max_steps: int,
    batch_threads: int,
) -> dict[str, Any]:
    return {
        "env_config": _relative_to_project(env_path),
        "policy_config": _relative_to_project(policy_path),
        "policy_overrides": {
            **candidate.policy_overrides(),
            "solver": {"num_threads_in_batch_solve": int(batch_threads)},
        },
        "seed": int(seed),
        "runner": {
            "target_episodes": int(target_episodes),
            "max_steps": int(max_steps),
            "stop_on_target_episodes": False,
            "render": False,
            "save_rollout": True,
        },
        "logging": {
            "root_dir": _relative_to_project(output_root / "runs"),
            "run_name": run_name,
        },
    }


def _launch_or_reuse(
    *,
    run_config: Mapping[str, Any],
    generated_config_path: Path,
    run_dir: Path,
    stdout_path: Path,
    python: str,
    num_envs: int,
    resume: bool,
    dry_run: bool,
) -> bool:
    """Launch no more than one rollout and return whether results are available."""
    _save_yaml(run_config, generated_config_path)
    if resume and (run_dir / "rollout.npz").exists():
        print(f"[INFO] Reusing completed rollout: {run_dir}")
        return True
    command = [
        python,
        str(EVAL_SCRIPT),
        "--config",
        str(generated_config_path),
        "--num_envs",
        str(num_envs),
        "--headless",
    ]
    print("[INFO] Launching one rollout:")
    print(" ".join(command))
    if dry_run:
        return False
    child_env = os.environ.copy()
    acados_source = Path(
        child_env.get("ACADOS_SOURCE_DIR", PROJECT_ROOT.parent / "acados")
    ).expanduser()
    acados_lib = acados_source / "lib"
    if acados_lib.is_dir():
        existing_library_path = child_env.get("LD_LIBRARY_PATH", "")
        library_entries = [entry for entry in existing_library_path.split(":") if entry]
        if str(acados_lib) not in library_entries:
            child_env["LD_LIBRARY_PATH"] = ":".join(
                (str(acados_lib), *library_entries)
            )
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stdout_path, "w", encoding="utf-8") as stream:
        try:
            subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=True,
                env=child_env,
            )
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"Evaluation process failed; inspect {stdout_path}."
            ) from exc
    if not (run_dir / "rollout.npz").exists():
        raise RuntimeError(
            f"Evaluation finished without rollout.npz; inspect {stdout_path}."
        )
    return True


def _trial_name(index: int, label: str, candidate: ObjectiveCandidate) -> str:
    safe_label = re.sub(r"[^a-zA-Z0-9_.-]+", "_", label)
    return f"screen_{index:02d}_{safe_label}_{candidate.identifier[-8:]}"


def _analyze_and_write(
    run_dir: Path,
    *,
    minimum_per_type: int,
    fixed: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows, metrics = analyze_acados_rollout(
        run_dir,
        minimum_per_type=minimum_per_type,
        safe_position_min=float(fixed.get("safe_position_min", 0.02)),
        safe_position_max=float(fixed.get("safe_position_max", 0.68)),
    )
    _write_csv(run_dir / "first_episode_metrics.csv", rows)
    _save_yaml(metrics, run_dir / "tuning_metrics.yaml")
    return rows, metrics


def _terminal_tail_improved(
    candidate: Mapping[str, Any], incumbent: Mapping[str, Any]
) -> bool:
    def dynamic_tail(metrics: Mapping[str, Any]) -> float:
        return float(
            np.mean(
                [
                    metrics["per_type"][name]["mean_tail_rmse"]
                    for name in TUNING_TRAJECTORY_TYPES[1:]
                ]
            )
        )

    return dynamic_tail(candidate) < dynamic_tail(incumbent)


def _record_screen_trial(
    state: dict[str, Any],
    proposal: Mapping[str, Any],
    candidate: ObjectiveCandidate,
    run_dir: Path,
    run_config_path: Path,
    metrics: Mapping[str, Any],
) -> dict[str, Any]:
    baseline_trial = not state["screen_trials"]
    if baseline_trial:
        decision = {
            "accepted": True,
            "reason": "formal screen baseline",
            "hard_rejection": bool(screen_rejection_reasons(metrics)),
        }
    else:
        incumbent_record = next(
            record
            for record in state["screen_trials"]
            if record["candidate_id"] == state["incumbent_candidate_id"]
        )
        decision = compare_screen_candidate(metrics, incumbent_record["metrics"])
        if (
            proposal.get("group") == "terminal_all"
            and decision["accepted"]
            and not _terminal_tail_improved(metrics, incumbent_record["metrics"])
        ):
            decision = {
                "accepted": False,
                "reason": "terminal-all trial did not improve dynamic tail error",
                "hard_rejection": False,
            }
    accepted = bool(decision["accepted"])
    record = {
        "trial_index": len(state["screen_trials"]),
        "trial_id": run_dir.name,
        "label": proposal.get("label", "baseline"),
        "group": proposal.get("group", "baseline"),
        "factor": proposal.get("factor", 1.0),
        "protocol_step": proposal.get("protocol_step"),
        "candidate_id": candidate.identifier,
        "candidate": candidate.as_dict(),
        "run_dir": str(run_dir),
        "run_config_path": str(run_config_path),
        "completed_at": _utc_now(),
        "accepted": accepted,
        "decision": decision,
        "metrics": dict(metrics),
    }
    state["screen_trials"].append(record)
    if accepted:
        state["incumbent_candidate_id"] = candidate.identifier
        state["incumbent_candidate"] = candidate.as_dict()
    return record


def _run_next(
    args: argparse.Namespace,
    config: Mapping[str, Any],
    state: dict[str, Any],
    output_root: Path,
    env_path: Path,
    policy_path: Path,
) -> None:
    fixed = config["fixed"]
    max_trials = int(fixed.get("max_screen_trials", 14))
    if len(state["screen_trials"]) >= max_trials:
        state["status"] = "screen_complete"
        _write_session_artifacts(state, output_root)
        print(f"[INFO] Screen budget ({max_trials}) is exhausted.")
        return

    if not state["screen_trials"]:
        candidate = ObjectiveCandidate.from_mapping(state["baseline_candidate"])
        proposal = {
            "label": "baseline",
            "group": "baseline",
            "factor": 1.0,
            "protocol_step": None,
            "candidate_id": candidate.identifier,
            "candidate": candidate.as_dict(),
            "reason": "establish a formal 10 s screen baseline",
        }
    else:
        proposal = _propose_next(state)
        if proposal is None:
            state["status"] = "screen_complete"
            _write_session_artifacts(state, output_root)
            print("[INFO] Sequential screen protocol is complete; run --stage validate.")
            return
        candidate = ObjectiveCandidate.from_mapping(proposal["candidate"])

    if args.candidate is not None:
        if not state["screen_trials"]:
            raise ValueError("Run the formal baseline before using --candidate.")
        candidate = ObjectiveCandidate.from_mapping(_load_yaml(args.candidate))
        proposal = {
            **proposal,
            "label": f"manual_{candidate.identifier[-8:]}",
            "candidate_id": candidate.identifier,
            "candidate": candidate.as_dict(),
            "reason": f"manual override of protocol step {proposal.get('protocol_step')}",
        }

    completed_same_candidate = next(
        (
            record
            for record in state["screen_trials"]
            if record.get("candidate_id") == candidate.identifier
        ),
        None,
    )
    if args.resume and completed_same_candidate is not None:
        _write_session_artifacts(state, output_root)
        print(
            f"[INFO] Candidate {candidate.identifier} was already completed as "
            f"{completed_same_candidate['trial_id']}; no rollout was launched."
        )
        return

    trial_index = len(state["screen_trials"])
    run_name = _trial_name(trial_index, str(proposal["label"]), candidate)
    run_dir = output_root / "runs" / run_name
    generated = output_root / "generated_configs" / f"{run_name}.yaml"
    stdout = output_root / "stdout" / f"{run_name}.log"
    screen = config["screen"]
    run_config = _build_run_config(
        env_path=env_path,
        policy_path=policy_path,
        candidate=candidate,
        output_root=output_root,
        run_name=run_name,
        seed=int(screen["seed"]),
        target_episodes=int(screen["target_episodes"]),
        max_steps=int(fixed["rollout_steps"]),
        batch_threads=int(fixed["batch_threads"]),
    )
    available = _launch_or_reuse(
        run_config=run_config,
        generated_config_path=generated,
        run_dir=run_dir,
        stdout_path=stdout,
        python=args.python,
        num_envs=int(screen["num_envs"]),
        resume=args.resume,
        dry_run=args.dry_run,
    )
    if not available:
        print(f"[DRY RUN] Candidate: {candidate.as_dict()}")
        return
    _, metrics = _analyze_and_write(
        run_dir,
        minimum_per_type=int(screen["minimum_per_type"]),
        fixed=fixed,
    )
    record = _record_screen_trial(state, proposal, candidate, run_dir, generated, metrics)
    if int(proposal.get("protocol_step") or -1) >= 0:
        state["protocol_cursor"] = max(
            int(state["protocol_cursor"]), int(proposal["protocol_step"]) + 1
        )
    _save_yaml(record, run_dir / "tuning_trial.yaml")
    _write_session_artifacts(state, output_root)
    print(
        f"[RESULT] {run_name}: accepted={record['accepted']}, "
        f"J={metrics['objective_score']:.6g}, reason={record['decision']['reason']}"
    )
    print(f"[INFO] Next recommendation: {output_root / 'next_recommendation.yaml'}")


def _validation_run_spec(
    config: Mapping[str, Any],
    state: Mapping[str, Any],
    output_root: Path,
    env_path: Path,
    policy_path: Path,
) -> tuple[str, ObjectiveCandidate, dict[str, Any], Path, Path, Path, int]:
    candidate = ObjectiveCandidate.from_mapping(state["incumbent_candidate"])
    validation = config["validation"]
    run_name = f"validation_{candidate.identifier[-8:]}_seed{validation['seed']}"
    run_dir = output_root / "runs" / run_name
    generated = output_root / "generated_configs" / f"{run_name}.yaml"
    stdout = output_root / "stdout" / f"{run_name}.log"
    run_config = _build_run_config(
        env_path=env_path,
        policy_path=policy_path,
        candidate=candidate,
        output_root=output_root,
        run_name=run_name,
        seed=int(validation["seed"]),
        target_episodes=int(validation["target_episodes"]),
        max_steps=int(config["fixed"]["rollout_steps"]),
        batch_threads=int(config["fixed"]["batch_threads"]),
    )
    return run_name, candidate, run_config, run_dir, generated, stdout, int(validation["num_envs"])


def _timing_run_spec(
    config: Mapping[str, Any],
    state: Mapping[str, Any],
    output_root: Path,
    env_path: Path,
    policy_path: Path,
) -> tuple[str, ObjectiveCandidate, dict[str, Any], Path, Path, Path, int]:
    candidate = ObjectiveCandidate.from_mapping(state["incumbent_candidate"])
    timing = config["timing"]
    run_name = f"timing_single_{candidate.identifier[-8:]}_seed{timing['seed']}"
    run_dir = output_root / "runs" / run_name
    generated = output_root / "generated_configs" / f"{run_name}.yaml"
    stdout = output_root / "stdout" / f"{run_name}.log"
    run_config = _build_run_config(
        env_path=env_path,
        policy_path=policy_path,
        candidate=candidate,
        output_root=output_root,
        run_name=run_name,
        seed=int(timing["seed"]),
        target_episodes=1,
        max_steps=int(timing["max_steps"]),
        batch_threads=1,
    )
    return run_name, candidate, run_config, run_dir, generated, stdout, 1


def _timing_metrics(run_dir: Path, warmup_steps: int) -> dict[str, Any]:
    with np.load(run_dir / "rollout.npz", allow_pickle=False) as rollout:
        values = np.asarray(rollout["policy_act_wall_time"], dtype=np.float64)
    finite = values[warmup_steps:]
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        raise ValueError("No finite post-warmup policy timing samples were recorded.")
    metrics = {
        "warmup_steps": int(warmup_steps),
        "sample_count": int(finite.size),
        "mean_s": float(np.mean(finite)),
        "p50_s": float(np.percentile(finite, 50)),
        "p95_s": float(np.percentile(finite, 95)),
        "p99_s": float(np.percentile(finite, 99)),
        "max_s": float(np.max(finite)),
        "deadline_miss_rate": float(np.mean(finite > 1.0 / 60.0)),
    }
    _save_yaml(metrics, run_dir / "single_env_timing.yaml")
    return metrics


def _promote_policy(policy_path: Path, candidate: ObjectiveCandidate) -> None:
    """Replace only two YAML lines, preserving all non-objective configuration text."""
    source = policy_path.read_text(encoding="utf-8")
    replacements = {
        "stage_weights": candidate.stage_weights,
        "terminal_weights": candidate.terminal_weights,
    }
    updated = source
    for field, values in replacements.items():
        pattern = re.compile(rf"^(\s*{field}:\s*)\[[^\n]*\]\s*$", re.MULTILINE)
        formatted = "[" + ", ".join(f"{value:g}" for value in values) + "]"
        updated, count = pattern.subn(lambda match: match.group(1) + formatted, updated)
        if count != 1:
            raise RuntimeError(f"Expected exactly one {field} line in {policy_path}; found {count}.")
    temporary = policy_path.with_suffix(policy_path.suffix + ".tmp")
    temporary.write_text(updated, encoding="utf-8")
    os.replace(temporary, policy_path)


def _write_final_report(
    config: Mapping[str, Any],
    state: dict[str, Any],
    output_root: Path,
    policy_path: Path,
) -> None:
    validation = state["validation"]
    screen_baseline = state["screen_trials"][0]["metrics"]
    timing = validation["timing_metrics"]
    decision = evaluate_validation(
        validation["metrics"], screen_baseline, timing_p99_s=float(timing["p99_s"])
    )
    candidate = ObjectiveCandidate.from_mapping(state["incumbent_candidate"])
    validation["decision"] = decision
    validation["completed_at"] = _utc_now()
    state["status"] = "validation_passed" if decision["passed"] else "validation_failed"
    if decision["passed"] and not state.get("promoted", False):
        _promote_policy(policy_path, candidate)
        state["promoted"] = True
        state["promoted_at"] = _utc_now()

    full_policy = deepcopy(_load_yaml(policy_path))
    policy_section = full_policy.get("acados_nmpc_policy", full_policy.get("policy", full_policy))
    policy_section["objective"] = candidate.as_dict()
    _save_yaml(full_policy, output_root / "recommended_policy.yaml")
    selection = {
        "passed": decision["passed"],
        "promoted": state["promoted"],
        "candidate_id": candidate.identifier,
        "objective": candidate.as_dict(),
        "policy_config_path": str(policy_path),
        "environment_sha256": state["env_config_sha256"],
        "screen_baseline": screen_baseline,
        "screen_best": next(
            record["metrics"]
            for record in state["screen_trials"]
            if record["candidate_id"] == state["incumbent_candidate_id"]
        ),
        "validation": validation["metrics"],
        "single_env_timing": timing,
        "decision": decision,
    }
    _save_yaml(selection, output_root / "final_selection.yaml")
    failure_lines = [name for name, passed in decision["checks"].items() if not passed]
    report = [
        "# acados NMPC objective tuning report",
        "",
        f"- Result: {'PASS' if decision['passed'] else 'FAIL (provisional best only)'}",
        f"- Candidate: `{candidate.identifier}`",
        f"- Promoted to default policy: `{state['promoted']}`",
        f"- Validation balanced RMSE: `{validation['metrics']['balanced_mean_rmse']:.6g}`",
        f"- Validation worst-type p90 RMSE: `{validation['metrics']['worst_type_p90_rmse']:.6g}`",
        f"- Single-environment act() p99: `{timing['p99_s'] * 1e3:.3f} ms`",
        "",
        "## Objective weights",
        "",
        f"- Stage: `{list(candidate.stage_weights)}`",
        f"- Terminal: `{list(candidate.terminal_weights)}`",
    ]
    if failure_lines:
        report.extend(("", "## Failed promotion gates", ""))
        report.extend(f"- `{name}`" for name in failure_lines)
    (output_root / "final_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def _run_validate(
    args: argparse.Namespace,
    config: Mapping[str, Any],
    state: dict[str, Any],
    output_root: Path,
    env_path: Path,
    policy_path: Path,
) -> None:
    if not state["screen_trials"]:
        raise RuntimeError("Run the screen baseline before validation.")
    if args.candidate is not None:
        raise ValueError("--candidate is supported only with --stage next.")
    validation_state = state.setdefault("validation", {})
    candidate_id = ObjectiveCandidate.from_mapping(state["incumbent_candidate"]).identifier
    if validation_state.get("candidate_id") not in (None, candidate_id):
        raise RuntimeError(
            "The incumbent changed after validation began; remove the stale validation state "
            "or validate the recorded candidate."
        )
    validation_state["candidate_id"] = candidate_id

    if "metrics" not in validation_state:
        spec = _validation_run_spec(config, state, output_root, env_path, policy_path)
        run_name, _, run_config, run_dir, generated, stdout, num_envs = spec
        available = _launch_or_reuse(
            run_config=run_config,
            generated_config_path=generated,
            run_dir=run_dir,
            stdout_path=stdout,
            python=args.python,
            num_envs=num_envs,
            resume=args.resume,
            dry_run=args.dry_run,
        )
        if not available:
            return
        _, metrics = _analyze_and_write(
            run_dir,
            minimum_per_type=int(config["validation"]["minimum_per_type"]),
            fixed=config["fixed"],
        )
        validation_state.update(
            {"run_id": run_name, "run_dir": str(run_dir), "metrics": metrics}
        )
        _write_session_artifacts(state, output_root)
        print(
            "[INFO] Validation statistics recorded. Run --stage validate once more for "
            "the independent single-environment timing rollout."
        )
        return

    if "timing_metrics" not in validation_state:
        spec = _timing_run_spec(config, state, output_root, env_path, policy_path)
        run_name, _, run_config, run_dir, generated, stdout, num_envs = spec
        available = _launch_or_reuse(
            run_config=run_config,
            generated_config_path=generated,
            run_dir=run_dir,
            stdout_path=stdout,
            python=args.python,
            num_envs=num_envs,
            resume=args.resume,
            dry_run=args.dry_run,
        )
        if not available:
            return
        timing = _timing_metrics(run_dir, int(config["timing"]["warmup_steps"]))
        validation_state.update(
            {"timing_run_id": run_name, "timing_run_dir": str(run_dir), "timing_metrics": timing}
        )
        _write_final_report(config, state, output_root, policy_path)
        _write_session_artifacts(state, output_root)
        print(
            f"[RESULT] Validation passed={validation_state['decision']['passed']}; "
            f"promoted={state['promoted']}. See {output_root / 'final_report.md'}"
        )
        return

    if "decision" not in validation_state:
        _write_final_report(config, state, output_root, policy_path)
        _write_session_artifacts(state, output_root)
    print(f"[INFO] Validation is already complete: {output_root / 'final_report.md'}")


def _analyze_existing(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    run_dir = args.analyze_run.expanduser().resolve()
    fixed = config["fixed"]
    rows, metrics = _analyze_and_write(
        run_dir,
        minimum_per_type=0,
        fixed=fixed,
    )
    print(f"[INFO] Analyzed {len(rows)} first episodes from {run_dir}")
    print(yaml.safe_dump(metrics, sort_keys=False))


def main() -> None:
    args = _parse_args()
    config_path = args.config.expanduser().resolve()
    config = _load_yaml(config_path)
    _validate_tuning_config(config)
    if args.analyze_run is not None:
        _analyze_existing(args, config)
        return

    env_path = _resolve_path(config["env_config"])
    policy_path = _resolve_path(config["policy_config"])
    output_root = _resolve_path(config["output_root"])
    if env_path != FIXED_ENV_CONFIG.resolve():
        raise ValueError(
            "objective_v1 is fixed to environments/configs/unified_tracking_mixed.yaml; "
            f"got {env_path}."
        )
    _validate_fixed_policy(_load_yaml(policy_path), int(config["fixed"]["n_horizon"]))
    state = _initialize_or_load_session(
        config, config_path, env_path, policy_path, output_root
    )
    if args.stage == "next":
        _run_next(args, config, state, output_root, env_path, policy_path)
    else:
        _run_validate(args, config, state, output_root, env_path, policy_path)


if __name__ == "__main__":
    main()
