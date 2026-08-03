"""Analysis and deterministic search helpers for acados NMPC objective tuning."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import yaml


TRAJECTORY_TYPE_TO_ID = {
    "sine": 0,
    "triangle": 1,
    "trapezoid": 2,
    "constant": 3,
    "random_b_spline": 4,
    "random_ramp_dwell": 5,
}
TUNING_TRAJECTORY_TYPES = (
    "constant",
    "random_b_spline",
    "random_ramp_dwell",
)
STAGE_WEIGHT_NAMES = (
    "position_error",
    "velocity_error",
    "beam_angle",
    "beam_rate",
    "response_velocity",
    "command",
    "action_increment",
)
TERMINAL_WEIGHT_NAMES = STAGE_WEIGHT_NAMES[:-1]
WEIGHT_MIN = 1.0e-3
WEIGHT_MAX = 1.0e3


def _finite_mean(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(np.mean(array)) if array.size else float("nan")


def _finite_percentile(values: Iterable[float], percentile: float) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(np.percentile(array, percentile)) if array.size else float("nan")


def _rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values)))) if values.size else float("nan")


def _tail(values: np.ndarray, samples: int) -> np.ndarray:
    return values[-min(max(samples, 1), values.size) :] if values.size else values


def _as_signal(
    rollout: Mapping[str, np.ndarray],
    key: str,
    episode_slice: slice,
    env_id: int,
) -> np.ndarray | None:
    if key not in rollout:
        return None
    values = np.asarray(rollout[key][episode_slice, env_id], dtype=np.float64)
    return values.reshape(values.shape[0], -1)[:, 0] if values.ndim > 1 else values


def _median_signal(
    rollout: Mapping[str, np.ndarray],
    key: str,
    episode_slice: slice,
    env_id: int,
    default: float = float("nan"),
) -> float:
    signal = _as_signal(rollout, key, episode_slice, env_id)
    if signal is None:
        return default
    finite = signal[np.isfinite(signal)]
    return float(np.median(finite)) if finite.size else default


def _zero_crossings(error: np.ndarray, epsilon: float = 1.0e-3) -> int:
    sign = np.sign(error)
    sign[np.abs(error) <= epsilon] = 0.0
    nonzero = sign[sign != 0.0]
    return int(np.count_nonzero(nonzero[1:] != nonzero[:-1])) if nonzero.size > 1 else 0


def _overshoot(pb: np.ndarray, pg: np.ndarray) -> float:
    if not pb.size:
        return float("nan")
    goal = float(np.median(_tail(pg, max(1, pg.size // 10))))
    direction = math.copysign(1.0, goal - float(pb[0])) if goal != float(pb[0]) else 1.0
    return max(float(np.max(direction * (pb - goal))), 0.0)


def sha256_file(path: str | Path) -> str:
    """Return a streaming SHA256 digest used to freeze the environment config."""
    digest = hashlib.sha256()
    with open(Path(path).expanduser().resolve(), "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class ObjectiveCandidate:
    """The only policy fields a tuning candidate is allowed to change."""

    stage_weights: tuple[float, ...]
    terminal_weights: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.stage_weights) != len(STAGE_WEIGHT_NAMES):
            raise ValueError(f"stage_weights must contain {len(STAGE_WEIGHT_NAMES)} values.")
        if len(self.terminal_weights) != len(TERMINAL_WEIGHT_NAMES):
            raise ValueError(
                f"terminal_weights must contain {len(TERMINAL_WEIGHT_NAMES)} values."
            )
        values = (*self.stage_weights, *self.terminal_weights)
        if any(not math.isfinite(value) for value in values):
            raise ValueError("All objective weights must be finite.")
        if any(value < WEIGHT_MIN or value > WEIGHT_MAX for value in values):
            raise ValueError(
                f"All objective weights must be in [{WEIGHT_MIN:g}, {WEIGHT_MAX:g}]."
            )

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "ObjectiveCandidate":
        allowed = {"stage_weights", "terminal_weights", "objective"}
        unknown = sorted(set(values) - allowed)
        if unknown:
            raise ValueError(
                "Candidate YAML may contain only objective weights; unknown fields: "
                + ", ".join(unknown)
            )
        if "objective" in values and set(values) != {"objective"}:
            raise ValueError(
                "Use either an objective mapping or top-level stage/terminal weights, not both."
            )
        objective = values.get("objective", values)
        if not isinstance(objective, Mapping):
            raise TypeError("candidate.objective must be a mapping.")
        unknown_objective = sorted(
            set(objective) - {"stage_weights", "terminal_weights"}
        )
        if unknown_objective:
            raise ValueError(
                "Candidate objective contains unknown fields: "
                + ", ".join(unknown_objective)
            )
        missing = [
            key for key in ("stage_weights", "terminal_weights") if key not in objective
        ]
        if missing:
            raise ValueError("Candidate must specify full " + " and ".join(missing) + ".")
        return cls(
            tuple(float(value) for value in objective["stage_weights"]),
            tuple(float(value) for value in objective["terminal_weights"]),
        )

    @classmethod
    def from_policy(cls, policy: Mapping[str, Any]) -> "ObjectiveCandidate":
        section = policy.get("acados_nmpc_policy", policy.get("policy", policy))
        return cls.from_mapping({"objective": section.get("objective", {})})

    @property
    def identifier(self) -> str:
        payload = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"))
        return "objective_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]

    def as_dict(self) -> dict[str, list[float]]:
        return {
            "stage_weights": list(self.stage_weights),
            "terminal_weights": list(self.terminal_weights),
        }

    def policy_overrides(self) -> dict[str, Any]:
        return {"objective": self.as_dict()}

    def scale_group(self, group: str, factor: float) -> "ObjectiveCandidate":
        """Scale one stage/terminal pair while preserving their current ratio."""
        if not math.isfinite(factor) or factor <= 0.0:
            raise ValueError("Weight scale factor must be finite and positive.")
        stage = list(self.stage_weights)
        terminal = list(self.terminal_weights)
        if group == "terminal_all":
            terminal = [value * factor for value in terminal]
        else:
            aliases = {
                "position": 0,
                "velocity": 1,
                "theta": 2,
                "omega": 3,
                "vrz": 4,
                "command": 5,
                "action": 6,
            }
            if group not in aliases:
                raise ValueError(f"Unknown objective weight group: {group}")
            index = aliases[group]
            stage[index] *= factor
            if index < len(terminal):
                terminal[index] *= factor
        return ObjectiveCandidate(tuple(stage), tuple(terminal))


def first_episode_ranges(done: np.ndarray) -> list[tuple[int, slice]]:
    """Return at most one complete episode per environment."""
    done = np.asarray(done, dtype=bool)
    if done.ndim != 2:
        raise ValueError("done must have shape (steps, num_envs).")
    ranges: list[tuple[int, slice]] = []
    for env_id in range(done.shape[1]):
        ends = np.flatnonzero(done[:, env_id])
        if ends.size:
            ranges.append((env_id, slice(0, int(ends[0]) + 1)))
    return ranges


def _resolved_value(metadata: Mapping[str, Any], section: str, name: str, default: float) -> float:
    policy = metadata.get("resolved_policy_config", {})
    return float(policy.get(section, {}).get(name, default))


def analyze_acados_rollout(
    run_dir: str | Path,
    *,
    required_trajectory_types: Sequence[str] = TUNING_TRAJECTORY_TYPES,
    minimum_per_type: int = 0,
    safe_position_min: float = 0.02,
    safe_position_max: float = 0.68,
    slack_active_tolerance: float = 1.0e-8,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Analyze only the first completed episode from every vectorized environment."""
    run_dir = Path(run_dir).expanduser().resolve()
    rollout_path = run_dir / "rollout.npz"
    if not rollout_path.exists():
        raise FileNotFoundError(f"Missing rollout: {rollout_path}")
    metadata_path = run_dir / "resolved_run.yaml"
    metadata = {}
    if metadata_path.exists():
        with open(metadata_path, encoding="utf-8") as stream:
            metadata = yaml.safe_load(stream) or {}
    with np.load(rollout_path, allow_pickle=False) as loaded:
        rollout = {key: loaded[key] for key in loaded.files}

    required_rollout_fields = {
        "observations",
        "observation_fields",
        "actions",
        "terminated",
        "truncated",
        "step_trajectory_type_id",
        "step_command_z",
        "acados_nmpc_solver_success",
        "acados_nmpc_fallback",
        "acados_nmpc_max_slack",
    }
    missing = sorted(required_rollout_fields - set(rollout))
    if missing:
        raise ValueError("acados tuning rollout is missing fields: " + ", ".join(missing))

    observations = np.asarray(rollout["observations"], dtype=np.float64)
    actions = np.asarray(rollout["actions"], dtype=np.float64)
    if actions.ndim == 3:
        actions = actions[..., 0]
    terminated = np.asarray(rollout["terminated"], dtype=bool)
    truncated = np.asarray(rollout["truncated"], dtype=bool)
    done = terminated | truncated
    if observations.ndim != 3 or done.shape != observations.shape[:2]:
        raise ValueError("Rollout observations and done arrays have incompatible shapes.")
    if actions.shape != done.shape:
        raise ValueError("Rollout actions and done arrays have incompatible shapes.")

    fields = tuple(str(value) for value in rollout.get("observation_fields", ()))
    indices = {name: index for index, name in enumerate(fields)}
    required_fields = {"pb": 0, "vb": 1, "theta": 3, "vrz": 7, "pg": 9}
    signal_indices = {name: indices.get(name, fallback) for name, fallback in required_fields.items()}
    step_dt = float(metadata.get("control_step_dt", metadata.get("step_dt", 1.0 / 60.0)))
    max_acc = _resolved_value(metadata, "constraints", "max_acc", 5.0)
    action_limit = abs(max_acc * step_dt)
    steady_samples = max(1, int(round(1.0 / step_dt)))
    type_id_to_name = {value: key for key, value in TRAJECTORY_TYPE_TO_ID.items()}

    rows: list[dict[str, Any]] = []
    for env_id, episode_slice in first_episode_ranges(done):
        def observation_signal(name: str) -> np.ndarray:
            return observations[episode_slice, env_id, signal_indices[name]]

        pb = _as_signal(rollout, "step_pb", episode_slice, env_id)
        pg = _as_signal(rollout, "step_pg", episode_slice, env_id)
        command = _as_signal(rollout, "step_command_z", episode_slice, env_id)
        pb = observation_signal("pb") if pb is None else pb
        pg = observation_signal("pg") if pg is None else pg
        command = np.zeros_like(pb) if command is None else command
        vb = observation_signal("vb")
        theta = observation_signal("theta")
        vrz = observation_signal("vrz")
        action = actions[episode_slice, env_id]
        error = pb - pg
        tail_error = _tail(error, steady_samples)
        tail_action = _tail(action, steady_samples)
        solver_success = _as_signal(
            rollout, "acados_nmpc_solver_success", episode_slice, env_id
        )
        fallback = _as_signal(rollout, "acados_nmpc_fallback", episode_slice, env_id)
        slack = _as_signal(rollout, "acados_nmpc_max_slack", episode_slice, env_id)
        solver_success = np.ones_like(error) if solver_success is None else solver_success
        fallback = np.zeros_like(error) if fallback is None else fallback
        slack = np.zeros_like(error) if slack is None else slack
        type_id_value = _median_signal(
            rollout, "step_trajectory_type_id", episode_slice, env_id
        )
        type_id = int(round(type_id_value)) if math.isfinite(type_id_value) else -1
        trajectory_type = type_id_to_name.get(type_id, f"type_{type_id}")
        primary = (pb, pg, vb, theta, vrz, command, action, solver_success, fallback, slack)
        nonfinite = any(not np.isfinite(signal).all() for signal in primary)
        finite_error = np.isfinite(error)
        safe_error = error[finite_error]
        abs_error = np.abs(safe_error)
        saturation = np.abs(action) >= max(action_limit - 1.0e-7, 0.0)
        row = {
            "env_id": env_id,
            "episode_index": 0,
            "trajectory_type": trajectory_type,
            "trajectory_type_id": type_id,
            "samples": int(error.size),
            "duration_s": float(error.size * step_dt),
            "nonfinite": nonfinite,
            "terminated": bool(terminated[episode_slice.stop - 1, env_id]),
            "truncated": bool(truncated[episode_slice.stop - 1, env_id]),
            "boundary_violation": bool(
                np.any(pb < safe_position_min) or np.any(pb > safe_position_max)
            ),
            "mae": float(np.mean(abs_error)) if abs_error.size and not nonfinite else float("inf"),
            "rmse": _rms(safe_error) if safe_error.size and not nonfinite else float("inf"),
            "p90_abs_error": (
                float(np.percentile(abs_error, 90))
                if abs_error.size and not nonfinite
                else float("inf")
            ),
            "tail_rmse": _rms(tail_error) if not nonfinite else float("inf"),
            "max_abs_error": float(np.max(abs_error)) if abs_error.size else float("inf"),
            "signed_bias": float(np.mean(safe_error)) if safe_error.size else float("inf"),
            "constant_steady_mae": (
                float(np.mean(np.abs(tail_error)))
                if trajectory_type == "constant" and not nonfinite
                else float("nan")
            ),
            "constant_converged": (
                bool(np.mean(np.abs(tail_error)) <= 0.01)
                if trajectory_type == "constant" and not nonfinite
                else False
            ),
            "constant_overshoot": (
                _overshoot(pb, pg) if trajectory_type == "constant" and not nonfinite else float("nan")
            ),
            "constant_zero_crossings": (
                _zero_crossings(error)
                if trajectory_type == "constant" and not nonfinite
                else 0
            ),
            "action_rms": _rms(action),
            "action_variation": (
                float(np.mean(np.abs(np.diff(action)))) if action.size > 1 else 0.0
            ),
            "action_change_rate": (
                float(np.mean(np.abs(np.diff(action))) / step_dt)
                if action.size > 1
                else 0.0
            ),
            "action_saturation_rate": float(np.mean(saturation)),
            "tail_action_saturation_rate": float(
                np.mean(np.abs(tail_action) >= max(action_limit - 1.0e-7, 0.0))
            ),
            "command_rms": _rms(command),
            "command_peak": float(np.max(np.abs(command))),
            "vb_rms": _rms(vb),
            "vb_peak": float(np.max(np.abs(vb))),
            "theta_rms": _rms(theta),
            "theta_peak": float(np.max(np.abs(theta))),
            "vrz_rms": _rms(vrz),
            "vrz_peak": float(np.max(np.abs(vrz))),
            "solver_success_rate": float(np.mean(solver_success.astype(bool))),
            "fallback_rate": float(np.mean(fallback.astype(bool))),
            "slack_active_rate": float(np.mean(slack > slack_active_tolerance)),
            "max_slack": float(np.max(slack)),
        }
        rows.append(row)

    per_type: dict[str, dict[str, Any]] = {}
    for name in required_trajectory_types:
        selected = [row for row in rows if row["trajectory_type"] == name]
        per_type[name] = {
            "episode_count": len(selected),
            "mean_rmse": _finite_mean(row["rmse"] for row in selected),
            "p90_rmse": _finite_percentile((row["rmse"] for row in selected), 90),
            "mean_tail_rmse": _finite_mean(row["tail_rmse"] for row in selected),
            "mean_mae": _finite_mean(row["mae"] for row in selected),
            "mean_max_abs_error": _finite_mean(row["max_abs_error"] for row in selected),
            "mean_signed_bias": _finite_mean(row["signed_bias"] for row in selected),
        }

    complete_types = [per_type[name] for name in required_trajectory_types]
    balanced_rmse = _finite_mean(values["mean_rmse"] for values in complete_types)
    worst_p90 = max(
        (float(values["p90_rmse"]) for values in complete_types),
        default=float("nan"),
    )
    balanced_tail = _finite_mean(values["mean_tail_rmse"] for values in complete_types)
    constant_rows = [row for row in rows if row["trajectory_type"] == "constant"]
    constant_steady = _finite_mean(row["constant_steady_mae"] for row in constant_rows)
    objective_score = (
        0.45 * balanced_rmse
        + 0.30 * worst_p90
        + 0.15 * balanced_tail
        + 0.10 * constant_steady
    )
    sample_count = sum(int(row["samples"]) for row in rows)

    def weighted_rate(key: str) -> float:
        if not sample_count:
            return float("nan")
        return float(
            sum(float(row[key]) * int(row["samples"]) for row in rows) / sample_count
        )

    coverage = {name: int(per_type[name]["episode_count"]) for name in required_trajectory_types}
    coverage_ok = all(value >= minimum_per_type for value in coverage.values())
    metrics = {
        "first_episode_only": True,
        "episode_count": len(rows),
        "num_envs_in_rollout": int(done.shape[1]),
        "incomplete_environment_count": int(done.shape[1] - len(rows)),
        "trajectory_coverage": coverage,
        "minimum_per_type": int(minimum_per_type),
        "coverage_ok": coverage_ok,
        "nonfinite_count": sum(bool(row["nonfinite"]) for row in rows),
        "termination_count": sum(bool(row["terminated"]) for row in rows),
        "boundary_violation_count": sum(bool(row["boundary_violation"]) for row in rows),
        "balanced_mean_rmse": balanced_rmse,
        "worst_type_p90_rmse": worst_p90,
        "balanced_tail_rmse": balanced_tail,
        "constant_steady_mae": constant_steady,
        "constant_convergence_rate": _finite_mean(
            float(row["constant_converged"]) for row in constant_rows
        ),
        "constant_mean_overshoot": _finite_mean(
            row["constant_overshoot"] for row in constant_rows
        ),
        "constant_mean_zero_crossings": _finite_mean(
            row["constant_zero_crossings"] for row in constant_rows
        ),
        "objective_score": objective_score,
        "action_rms": _finite_mean(row["action_rms"] for row in rows),
        "action_variation": _finite_mean(row["action_variation"] for row in rows),
        "action_change_rate": _finite_mean(row["action_change_rate"] for row in rows),
        "action_saturation_rate": weighted_rate("action_saturation_rate"),
        "tail_action_saturation_rate": _finite_mean(
            row["tail_action_saturation_rate"] for row in rows
        ),
        "command_rms": _finite_mean(row["command_rms"] for row in rows),
        "command_peak": max((float(row["command_peak"]) for row in rows), default=float("nan")),
        "vb_rms": _finite_mean(row["vb_rms"] for row in rows),
        "vb_peak": max((float(row["vb_peak"]) for row in rows), default=float("nan")),
        "theta_rms": _finite_mean(row["theta_rms"] for row in rows),
        "theta_peak": max((float(row["theta_peak"]) for row in rows), default=float("nan")),
        "vrz_rms": _finite_mean(row["vrz_rms"] for row in rows),
        "vrz_peak": max((float(row["vrz_peak"]) for row in rows), default=float("nan")),
        "solver_success_rate": weighted_rate("solver_success_rate"),
        "fallback_rate": weighted_rate("fallback_rate"),
        "slack_active_rate": weighted_rate("slack_active_rate"),
        "max_slack": max((float(row["max_slack"]) for row in rows), default=float("nan")),
        "per_type": per_type,
    }
    metrics["screen_hard_rejection"] = screen_rejection_reasons(metrics)
    return rows, metrics


def screen_rejection_reasons(metrics: Mapping[str, Any]) -> list[str]:
    """Apply the protocol's non-negotiable screen gates."""
    checks = (
        (int(metrics.get("nonfinite_count", 0)) > 0, "nonfinite values detected"),
        (not bool(metrics.get("coverage_ok", False)), "trajectory coverage is insufficient"),
        (float(metrics.get("solver_success_rate", 0.0)) < 0.99, "solver success < 99%"),
        (float(metrics.get("fallback_rate", 1.0)) > 0.01, "fallback > 1%"),
        (float(metrics.get("action_saturation_rate", 1.0)) > 0.20, "action saturation > 20%"),
        (
            float(metrics.get("tail_action_saturation_rate", 1.0)) > 0.05,
            "tail action saturation > 5%",
        ),
    )
    return [reason for failed, reason in checks if failed]


def compare_screen_candidate(
    candidate: Mapping[str, Any], incumbent: Mapping[str, Any]
) -> dict[str, Any]:
    """Return a deterministic accept/reject decision for one screen trial."""
    hard_reasons = screen_rejection_reasons(candidate)
    if hard_reasons:
        return {"accepted": False, "reason": "; ".join(hard_reasons), "hard_rejection": True}
    candidate_terminations = int(candidate.get("termination_count", 0))
    incumbent_terminations = int(incumbent.get("termination_count", 0))
    candidate_boundaries = int(candidate.get("boundary_violation_count", 0))
    incumbent_boundaries = int(incumbent.get("boundary_violation_count", 0))
    safety_not_worse = (
        candidate_terminations <= incumbent_terminations
        and candidate_boundaries <= incumbent_boundaries
    )
    safety_better = safety_not_worse and (
        candidate_terminations < incumbent_terminations
        or candidate_boundaries < incumbent_boundaries
    )
    if (incumbent_terminations or incumbent_boundaries) and safety_better:
        return {
            "accepted": True,
            "reason": "termination/boundary count decreased without safety regression",
            "hard_rejection": False,
        }
    if not safety_not_worse:
        return {
            "accepted": False,
            "reason": "termination or boundary count regressed",
            "hard_rejection": False,
        }
    candidate_score = float(candidate.get("objective_score", float("inf")))
    incumbent_score = float(incumbent.get("objective_score", float("inf")))
    relative_improvement = (
        (incumbent_score - candidate_score) / incumbent_score
        if math.isfinite(incumbent_score) and incumbent_score > 0.0
        else float("-inf")
    )
    p90_ok = True
    for trajectory_type in TUNING_TRAJECTORY_TYPES:
        candidate_p90 = float(candidate.get("per_type", {}).get(trajectory_type, {}).get("p90_rmse", float("inf")))
        incumbent_p90 = float(incumbent.get("per_type", {}).get(trajectory_type, {}).get("p90_rmse", float("inf")))
        p90_ok &= candidate_p90 <= 1.05 * incumbent_p90
    if relative_improvement >= 0.03 and p90_ok:
        return {
            "accepted": True,
            "reason": f"balanced objective improved by {100.0 * relative_improvement:.2f}%",
            "hard_rejection": False,
            "relative_improvement": relative_improvement,
        }
    lower_action = float(candidate.get("action_rms", float("inf"))) < float(
        incumbent.get("action_rms", float("inf"))
    )
    if relative_improvement > -0.03 and p90_ok and lower_action:
        return {
            "accepted": True,
            "reason": "objective change was below 3%; retained lower-action-RMS candidate",
            "hard_rejection": False,
            "relative_improvement": relative_improvement,
        }
    return {
        "accepted": False,
        "reason": "no significant balanced improvement or a type p90 regressed by >5%",
        "hard_rejection": False,
        "relative_improvement": relative_improvement,
    }


def evaluate_validation(
    validation: Mapping[str, Any],
    screen_baseline: Mapping[str, Any],
    *,
    timing_p99_s: float | None = None,
) -> dict[str, Any]:
    """Evaluate all promotion gates, including the separate one-env timing run."""
    baseline_rmse = float(screen_baseline.get("balanced_mean_rmse", float("nan")))
    baseline_worst = float(screen_baseline.get("worst_type_p90_rmse", float("nan")))
    rmse_improvement = (
        1.0 - float(validation.get("balanced_mean_rmse", float("inf"))) / baseline_rmse
        if baseline_rmse > 0.0
        else float("-inf")
    )
    worst_improvement = (
        1.0 - float(validation.get("worst_type_p90_rmse", float("inf"))) / baseline_worst
        if baseline_worst > 0.0
        else float("-inf")
    )
    coverage = validation.get("trajectory_coverage", {})
    checks = {
        "episode_count_96": int(validation.get("episode_count", 0)) >= 96,
        "minimum_20_per_type": all(int(coverage.get(name, 0)) >= 20 for name in TUNING_TRAJECTORY_TYPES),
        "no_nonfinite": int(validation.get("nonfinite_count", 0)) == 0,
        "no_termination": int(validation.get("termination_count", 0)) == 0,
        "no_boundary_event": int(validation.get("boundary_violation_count", 0)) == 0,
        "solver_success": float(validation.get("solver_success_rate", 0.0)) >= 0.999,
        "fallback": float(validation.get("fallback_rate", 1.0)) <= 0.001,
        "action_saturation": float(validation.get("action_saturation_rate", 1.0)) <= 0.10,
        "tail_action_saturation": float(validation.get("tail_action_saturation_rate", 1.0)) <= 0.02,
        "slack_active": float(validation.get("slack_active_rate", 1.0)) <= 0.001,
        "max_slack": float(validation.get("max_slack", float("inf"))) <= 1.0e-3,
        "balanced_rmse_improvement": rmse_improvement >= 0.30,
        "worst_p90_improvement": worst_improvement >= 0.20,
        "constant_steady_mae": float(validation.get("constant_steady_mae", float("inf"))) <= 0.05,
        "single_env_timing_p99": timing_p99_s is not None and timing_p99_s <= 1.0 / 60.0,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "balanced_rmse_improvement": rmse_improvement,
        "worst_p90_improvement": worst_improvement,
        "timing_p99_s": timing_p99_s,
    }


def leaderboard_key(record: Mapping[str, Any]) -> tuple[Any, ...]:
    """Sort feasible, safe, low-objective candidates first."""
    metrics = record.get("metrics", {})
    return (
        bool(screen_rejection_reasons(metrics)),
        int(metrics.get("termination_count", 10**9)) + int(metrics.get("boundary_violation_count", 10**9)),
        float(metrics.get("objective_score", float("inf"))),
        float(metrics.get("action_rms", float("inf"))),
    )
