#!/usr/bin/env python3
"""Run coarse or local-grid NFFB tuning on configured mixed references."""

from __future__ import annotations

import argparse
import csv
import math
import subprocess
import sys
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from itertools import product
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_PARENT = PROJECT_ROOT.parent
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from aerial_balance_bench.utils.nffb_tuning import (  # noqa: E402
    aggregate_mixed_episode_metrics,
    analyze_rollout,
    mixed_tuning_rank_key,
)


DEFAULT_CONFIG = PROJECT_ROOT / "baselines" / "configs" / "nffb_mixed_tuning.yaml"
EVAL_SCRIPT = PROJECT_ROOT / "scripts" / "nffb_policy_eval.py"
STAGE_ORDER = ("baseline", "screen", "validate")
GRID_STRATEGY = "grid"
COARSE_STRATEGY = "coarse"


@dataclass(frozen=True)
class Candidate:
    """Searchable NFFB gains used by the mixed-reference workflow."""

    outer_omega: float
    outer_zeta: float
    integral_pole: float
    filter_omega: float
    filter_zeta: float
    k_theta: float
    omega_max: float
    feedforward: bool = True

    @property
    def bandwidth_cost(self) -> float:
        return self.outer_omega + self.filter_omega + self.k_theta

    @property
    def identifier(self) -> str:
        values = (
            f"wo{self.outer_omega:g}",
            f"zo{self.outer_zeta:g}",
            f"a{self.integral_pole:g}",
            f"wf{self.filter_omega:g}",
            f"zf{self.filter_zeta:g}",
            f"kt{self.k_theta:g}",
            f"om{self.omega_max:g}",
            "ff1" if self.feedforward else "ff0",
        )
        return "_".join(value.replace(".", "p") for value in values)

    def overrides(self) -> dict[str, Any]:
        return {
            "outer_loop": {
                "natural_frequency": self.outer_omega,
                "damping_ratio": self.outer_zeta,
                "integral_pole": self.integral_pole,
                "acceleration_feedforward_enabled": self.feedforward,
            },
            "command_filter": {
                "natural_frequency": self.filter_omega,
                "damping_ratio": self.filter_zeta,
            },
            "inner_loop": {"k_theta": self.k_theta},
            "constraints": {"omega_max": self.omega_max},
        }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--stage",
        choices=(*STAGE_ORDER, "all"),
        default="all",
        help="Run through this stage, including its prerequisites.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used for nffb_policy_eval.py.",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse completed rollout directories.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate configs and print commands without launching Isaac Sim.",
    )
    parser.add_argument(
        "--analyze-run",
        type=Path,
        help="Analyze one existing mixed-reference run directory and exit.",
    )
    return parser.parse_args()


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as stream:
        return yaml.safe_load(stream) or {}


def _save_yaml(data: Mapping[str, Any], path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as stream:
        yaml.safe_dump(dict(data), stream, sort_keys=False)


def _write_csv(path: str | Path, rows: Iterable[Mapping[str, Any]]):
    rows = [dict(row) for row in rows]
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _deep_update(target: dict[str, Any], values: Mapping[str, Any]):
    for key, value in values.items():
        if isinstance(value, Mapping) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = deepcopy(value)


def _resolve_path(path: str | Path) -> Path:
    path = Path(path).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _policy_section(policy_config: Mapping[str, Any]) -> Mapping[str, Any]:
    return policy_config.get("nffb_policy", policy_config.get("policy", policy_config))


def _candidate_from_policy(policy_config: Mapping[str, Any]) -> Candidate:
    policy = _policy_section(policy_config)
    outer = policy.get("outer_loop", {})
    command_filter = policy.get("command_filter", {})
    inner = policy.get("inner_loop", {})
    constraints = policy.get("constraints", {})
    return Candidate(
        outer_omega=float(outer.get("natural_frequency", 1.0)),
        outer_zeta=float(outer.get("damping_ratio", 1.0)),
        integral_pole=float(outer.get("integral_pole", 0.0)),
        filter_omega=float(command_filter.get("natural_frequency", 2.5)),
        filter_zeta=float(command_filter.get("damping_ratio", 1.0)),
        k_theta=float(inner.get("k_theta", 4.0)),
        omega_max=float(constraints.get("omega_max", 0.5)),
        feedforward=bool(outer.get("acceleration_feedforward_enabled", True)),
    )


def _candidate_fields(candidate: Candidate) -> dict[str, Any]:
    fields = asdict(candidate)
    fields["candidate_id"] = candidate.identifier
    fields["bandwidth_cost"] = candidate.bandwidth_cost
    return fields


def _candidate_from_mapping(values: Mapping[str, Any]) -> Candidate:
    """Build a fully specified candidate from a tuning config mapping."""
    required = tuple(Candidate.__dataclass_fields__)
    missing = [name for name in required if name not in values]
    unknown = sorted(set(values) - set(required))
    if missing or unknown:
        raise ValueError(
            "initial_candidate must specify exactly the Candidate fields; "
            f"missing={missing}, unknown={unknown}."
        )
    return Candidate(
        outer_omega=float(values["outer_omega"]),
        outer_zeta=float(values["outer_zeta"]),
        integral_pole=float(values["integral_pole"]),
        filter_omega=float(values["filter_omega"]),
        filter_zeta=float(values["filter_zeta"]),
        k_theta=float(values["k_theta"]),
        omega_max=float(values["omega_max"]),
        feedforward=bool(values["feedforward"]),
    )


def _grid_candidates(config: Mapping[str, Any]) -> tuple[Candidate, list[Candidate]]:
    """Expand and validate the bounded Cartesian grid in a tuning config."""
    search = config.get("search", {})
    if str(search.get("strategy", COARSE_STRATEGY)) != GRID_STRATEGY:
        raise ValueError("Grid candidates require search.strategy='grid'.")

    center = _candidate_from_mapping(config.get("initial_candidate", {}))
    parameter_values = search.get("parameters", {})
    candidate_fields = set(Candidate.__dataclass_fields__)
    unknown = sorted(set(parameter_values) - candidate_fields)
    if unknown:
        raise ValueError(f"Unknown Candidate grid fields: {unknown}.")
    if not parameter_values:
        raise ValueError("Grid search.parameters must contain at least one field.")

    names = list(parameter_values)
    values_by_name: list[list[Any]] = []
    for name in names:
        values = list(parameter_values[name])
        if not values:
            raise ValueError(f"Grid field {name!r} has no values.")
        normalized = [
            bool(value) if name == "feedforward" else float(value)
            for value in values
        ]
        if len(set(normalized)) != len(normalized):
            raise ValueError(f"Grid field {name!r} contains duplicate values.")
        values_by_name.append(normalized)

    raw_count = math.prod(len(values) for values in values_by_name)
    maximum = int(search.get("max_combinations", 30))
    if maximum <= 0 or raw_count > maximum:
        raise ValueError(
            f"Grid expands to {raw_count} combinations; maximum is {maximum}."
        )

    candidates = [
        replace(center, **dict(zip(names, combination)))
        for combination in product(*values_by_name)
    ]
    candidates = _deduplicate(candidates)
    if len(candidates) != raw_count:
        raise ValueError("Grid expansion produced duplicate candidates.")
    if center not in candidates:
        raise ValueError("Grid must include initial_candidate as one combination.")
    return center, candidates


def _validate_grid_policy_contract(config: Mapping[str, Any]):
    """Require the no-predictor/no-response-inverse contract for grid tuning."""
    defaults = config.get("policy_defaults", {})
    predictor = defaults.get("state_predictor", {})
    compensation = defaults.get("velocity_response_compensation", {})
    if bool(predictor.get("enabled", False)) or int(
        predictor.get("delay_step", 0)
    ) != 0:
        raise ValueError("Grid tuning requires state_predictor disabled with delay_step=0.")
    if bool(compensation.get("enabled", False)):
        raise ValueError("Grid tuning requires velocity_response_compensation disabled.")


def _deduplicate(candidates: Iterable[Candidate]) -> list[Candidate]:
    result: list[Candidate] = []
    seen: set[Candidate] = set()
    for candidate in candidates:
        if candidate not in seen:
            seen.add(candidate)
            result.append(candidate)
    return result


def _stage_enabled(target_stage: str, stage: str) -> bool:
    if target_stage == "all":
        return True
    return STAGE_ORDER.index(stage) <= STAGE_ORDER.index(target_stage)


def _best(
    candidate_results: Iterable[tuple[Candidate, Mapping[str, Any]]],
    count: int = 1,
) -> list[tuple[Candidate, Mapping[str, Any]]]:
    return sorted(
        candidate_results,
        key=lambda item: mixed_tuning_rank_key(item[1]),
    )[:count]


class TuningSession:
    """Run resumable mixed-reference NFFB trials through the existing evaluator."""

    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        python: str,
        resume: bool,
        dry_run: bool,
    ):
        self.config = dict(config)
        self.python = python
        self.resume = resume
        self.dry_run = dry_run
        self.base_run_config_path = _resolve_path(config["base_run_config"])
        self.base_env_config_path = _resolve_path(config["base_env_config"])
        self.base_policy_config_path = _resolve_path(config["base_policy_config"])
        self.sine_policy_config_path = _resolve_path(config["sine_policy_config"])
        self.base_run_config = _load_yaml(self.base_run_config_path)
        self.base_env_config = _load_yaml(self.base_env_config_path)
        self.base_policy_config = _load_yaml(self.base_policy_config_path)
        self.sine_policy_config = _load_yaml(self.sine_policy_config_path)
        self.required_types = tuple(str(name) for name in config["required_types"])
        self.policy_defaults = deepcopy(dict(config["policy_defaults"]))
        self._validate_environment_contract()

        output_root = _resolve_path(config["output_root"])
        self.session_root = output_root / str(config.get("session_name", "coarse_v1"))
        self.generated_dir = self.session_root / "_generated"
        self.runs_root = self.session_root / "runs"
        self.stdout_root = self.session_root / "stdout"
        for directory in (self.generated_dir, self.runs_root, self.stdout_root):
            directory.mkdir(parents=True, exist_ok=True)

        self.results: list[dict[str, Any]] = []
        self.episode_rows: dict[str, list[dict[str, Any]]] = {}
        self._trial_cache: dict[tuple[str, int, int, int], dict[str, Any]] = {}

    def _validate_environment_contract(self):
        task_types = tuple(
            str(name)
            for name in self.base_env_config.get("unified_tracking_task", {}).get(
                "trajectory_types",
                (),
            )
        )
        if task_types != self.required_types:
            raise ValueError(
                "Mixed tuning required_types must exactly match the base environment "
                f"trajectory_types; got {self.required_types} versus {task_types}."
            )
        if self.base_env_config.get("task_name") != "unified_tracking":
            raise ValueError("Mixed NFFB tuning requires task_name='unified_tracking'.")
        if self.base_env_config.get("interface_name") != "velocity":
            raise ValueError("Mixed NFFB tuning requires interface_name='velocity'.")
        robustness = self.base_env_config.get("robustness", {})
        if not (
            bool(robustness.get("enabled", False))
            and bool(robustness.get("velocity_response_enabled", False))
        ):
            raise ValueError(
                "Mixed NFFB tuning requires the environment velocity response to be enabled."
            )

    def _run_config(
        self,
        candidate: Candidate,
        *,
        trial_id: str,
        episodes: int,
    ) -> Path:
        run_config = deepcopy(self.base_run_config)
        run_config["env_config"] = str(self.base_env_config_path)
        run_config["policy_config"] = str(self.base_policy_config_path)
        policy_overrides = deepcopy(run_config.get("policy_overrides", {}))
        _deep_update(policy_overrides, self.policy_defaults)
        _deep_update(policy_overrides, candidate.overrides())
        run_config["policy_overrides"] = policy_overrides
        run_config["runner"] = {
            **run_config.get("runner", {}),
            "target_episodes": episodes,
            "max_steps": None,
            "render": False,
            "save_rollout": True,
        }
        run_config["logging"] = {
            "root_dir": str(self.runs_root),
            "run_name": trial_id,
        }
        path = self.generated_dir / "run" / f"{trial_id}.yaml"
        _save_yaml(run_config, path)
        return path

    def run_trial(
        self,
        candidate: Candidate,
        *,
        stage: str,
        seed: int,
        num_envs: int,
        episodes: int,
        min_episodes_per_type: int,
    ) -> dict[str, Any]:
        cache_key = (candidate.identifier, seed, num_envs, episodes)
        cached = self._trial_cache.get(cache_key)
        if cached is not None:
            return cached

        trial_id = (
            f"{stage}__{candidate.identifier}__seed{seed}"
            f"__n{num_envs}__ep{episodes}"
        )
        run_dir = self.runs_root / trial_id
        run_config_path = self._run_config(
            candidate,
            trial_id=trial_id,
            episodes=episodes,
        )
        command = [
            self.python,
            str(EVAL_SCRIPT),
            "--config",
            str(run_config_path),
            "--episodes",
            str(episodes),
            "--num_envs",
            str(num_envs),
            "--seed",
            str(seed),
            "--run_name",
            trial_id,
            "--headless",
        ]
        if not (self.resume and (run_dir / "rollout.npz").exists()):
            if run_dir.exists() and not self.resume:
                raise FileExistsError(
                    f"Trial directory exists; use --resume or a new session: {run_dir}"
                )
            print("[RUN]", " ".join(command), flush=True)
            if self.dry_run:
                result = {
                    **_candidate_fields(candidate),
                    "stage": stage,
                    "seed": seed,
                    "num_envs": num_envs,
                    "episodes": episodes,
                    "dry_run": True,
                }
                self._trial_cache[cache_key] = result
                return result
            stdout_path = self.stdout_root / f"{trial_id}.log"
            with open(stdout_path, "w", encoding="utf-8") as stream:
                completed = subprocess.run(
                    command,
                    cwd=PROJECT_ROOT,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            if completed.returncode != 0 or not (run_dir / "rollout.npz").exists():
                raise RuntimeError(
                    "NFFB trial failed or did not produce rollout.npz "
                    f"(status={completed.returncode}); see {stdout_path}"
                )
        else:
            print(f"[RESUME] {trial_id}", flush=True)

        episode_rows, _ = analyze_rollout(run_dir)
        metrics = aggregate_mixed_episode_metrics(
            episode_rows,
            required_types=self.required_types,
            min_episodes_per_type=min_episodes_per_type,
            full_saturation_rate=float(self.config["acceptance"]["full_saturation_rate"]),
            tail_saturation_rate=float(self.config["acceptance"]["tail_saturation_rate"]),
        )
        result = {
            **_candidate_fields(candidate),
            "stage": stage,
            "seed": seed,
            "num_envs": num_envs,
            "episodes": episodes,
            "run_dir": str(run_dir),
            **metrics,
        }
        self.results.append(result)
        self.episode_rows[trial_id] = episode_rows
        self._trial_cache[cache_key] = result
        _write_csv(run_dir / "episode_metrics.csv", episode_rows)
        _save_yaml(metrics.get("type_metrics", {}), run_dir / "type_metrics.yaml")
        _save_yaml(result, run_dir / "tuning_metrics.yaml")
        _save_yaml(
            {
                "candidate": _candidate_fields(candidate),
                "stage": stage,
                "seed": seed,
                "episodes": episodes,
                "num_envs": num_envs,
                "min_episodes_per_type": min_episodes_per_type,
                "command": command,
            },
            run_dir / "tuning_trial.yaml",
        )
        self.write_leaderboard()
        return result

    def write_leaderboard(self):
        if not self.results:
            return
        sorted_results = sorted(self.results, key=mixed_tuning_rank_key)
        _write_csv(self.session_root / "leaderboard.csv", sorted_results)
        _save_yaml({"trials": sorted_results}, self.session_root / "leaderboard.yaml")

    def effective_policy_config(self, candidate: Candidate) -> dict[str, Any]:
        effective = deepcopy(self.base_policy_config)
        policy = effective.get("nffb_policy")
        if not isinstance(policy, dict):
            policy = effective
        _deep_update(policy, candidate.overrides())
        _deep_update(policy, self.policy_defaults)
        output_policy_name = str(
            self.config.get("output_policy_name", "nffb_unified_mixed_coarse")
        )
        if "policy_name" in policy:
            policy["policy_name"] = output_policy_name
        else:
            policy["name"] = output_policy_name
        return effective


def _screen_settings(config: Mapping[str, Any]) -> tuple[int, int, int, int]:
    screen = config["screen"]
    return (
        int(screen["seed"]),
        int(screen["num_envs"]),
        int(screen["episodes"]),
        int(screen["min_episodes_per_type"]),
    )


def _run_screen_candidates(
    session: TuningSession,
    candidates: Iterable[Candidate],
    *,
    stage: str,
) -> list[tuple[Candidate, dict[str, Any]]]:
    seed, num_envs, episodes, minimum = _screen_settings(session.config)
    return [
        (
            candidate,
            session.run_trial(
                candidate,
                stage=stage,
                seed=seed,
                num_envs=num_envs,
                episodes=episodes,
                min_episodes_per_type=minimum,
            ),
        )
        for candidate in _deduplicate(candidates)
    ]


def _dry_run_candidates(session: TuningSession):
    generic = _candidate_from_policy(session.base_policy_config)
    sine = _candidate_from_policy(session.sine_policy_config)
    search = session.config["search"]
    candidates = [generic, sine]
    candidates.extend(
        replace(
            generic,
            filter_omega=float(pair[0]),
            k_theta=float(pair[1]),
            integral_pole=0.0,
            filter_zeta=1.0,
            omega_max=0.5,
            feedforward=True,
        )
        for pair in search["filter_inner_pairs"]
    )
    template = candidates[2]
    candidates.extend(
        replace(
            template,
            outer_omega=float(pair[0]),
            outer_zeta=float(pair[1]),
        )
        for pair in search["outer_pairs"]
    )
    _run_screen_candidates(session, candidates, stage="dry_run")


def _comparison(selected: Mapping[str, Any], baseline: Mapping[str, Any]) -> dict[str, Any]:
    def improvement(name: str) -> float:
        selected_value = float(selected.get(name, float("nan")))
        baseline_value = float(baseline.get(name, float("nan")))
        if not (
            math.isfinite(selected_value)
            and math.isfinite(baseline_value)
            and baseline_value > 0.0
        ):
            return float("nan")
        return 1.0 - selected_value / baseline_value

    result = {
        "mean_rmse_improvement": improvement("mean_rmse"),
        "worst_type_p95_rmse_improvement": improvement(
            "worst_type_p95_rmse"
        ),
        "p95_maxe_improvement": improvement("p95_maxe"),
    }
    selected_types = selected.get("type_metrics", {})
    baseline_types = baseline.get("type_metrics", {})
    result["per_type"] = {
        trajectory_type: {
            "mean_rmse_improvement": (
                1.0
                - float(selected_types[trajectory_type]["mean_rmse"])
                / float(baseline_types[trajectory_type]["mean_rmse"])
                if float(baseline_types[trajectory_type]["mean_rmse"]) > 0.0
                else float("nan")
            ),
            "p95_rmse_improvement": (
                1.0
                - float(selected_types[trajectory_type]["p95_rmse"])
                / float(baseline_types[trajectory_type]["p95_rmse"])
                if float(baseline_types[trajectory_type]["p95_rmse"]) > 0.0
                else float("nan")
            ),
        }
        for trajectory_type in selected_types
        if trajectory_type in baseline_types
    }
    return result


def _acceptance(config: Mapping[str, Any], metrics: Mapping[str, Any]) -> dict[str, Any]:
    checks = {
        "coverage": not bool(metrics.get("coverage_failure", True)),
        "no_nonfinite": int(metrics.get("nonfinite_count", 0)) == 0,
        "no_termination": int(metrics.get("termination_count", 0)) == 0,
        "no_boundary_violation": int(metrics.get("boundary_violation_count", 0))
        == 0,
        "full_saturation": float(
            metrics.get("max_full_saturation_rate", float("inf"))
        )
        <= float(config["acceptance"]["full_saturation_rate"]),
        "tail_saturation": float(
            metrics.get("max_tail_saturation_rate", float("inf"))
        )
        <= float(config["acceptance"]["tail_saturation_rate"]),
    }
    return {"passed": all(checks.values()), "checks": checks}


def _run_grid_search(session: TuningSession, target_stage: str):
    """Evaluate every configured grid point and always report the observed best."""
    if target_stage != "all":
        raise ValueError("Grid strategy supports only --stage all.")

    config = session.config
    _validate_grid_policy_contract(config)
    center, candidates = _grid_candidates(config)
    evaluation = config.get("evaluation", {})
    seed = int(evaluation["seed"])
    num_envs = int(evaluation["num_envs"])
    episodes = int(evaluation["episodes"])
    minimum = int(evaluation["min_episodes_per_type"])
    if seed < 0 or num_envs <= 0 or episodes <= 0 or minimum <= 0:
        raise ValueError(
            "Grid evaluation values must use a fixed seed and positive budgets."
        )

    manifest = {
        "strategy": GRID_STRATEGY,
        "candidate_count": len(candidates),
        "initial_candidate": _candidate_fields(center),
        "evaluation": {
            "seed": seed,
            "num_envs": num_envs,
            "episodes": episodes,
            "min_episodes_per_type": minimum,
        },
        "candidates": [_candidate_fields(candidate) for candidate in candidates],
    }
    _save_yaml(manifest, session.session_root / "grid_manifest.yaml")

    grid_results = [
        (
            candidate,
            session.run_trial(
                candidate,
                stage="grid",
                seed=seed,
                num_envs=num_envs,
                episodes=episodes,
                min_episodes_per_type=minimum,
            ),
        )
        for candidate in candidates
    ]
    if session.dry_run:
        return

    selected, selected_metrics = _best(grid_results)[0]
    metrics_by_candidate = {
        candidate.identifier: metrics for candidate, metrics in grid_results
    }
    center_metrics = metrics_by_candidate[center.identifier]
    effective_policy = session.effective_policy_config(selected)
    selection = {
        "candidate": _candidate_fields(selected),
        "selected_evaluation": selected_metrics,
        "initial_candidate": _candidate_fields(center),
        "initial_evaluation": center_metrics,
        "comparison_to_initial": _comparison(selected_metrics, center_metrics),
        "acceptance": _acceptance(config, selected_metrics),
        "policy_config": effective_policy,
    }
    _save_yaml(selection, session.session_root / "best_selection.yaml")
    _save_yaml(effective_policy, session.session_root / "best_observed_policy.yaml")
    _write_csv(
        session.session_root / "grid_summary.csv",
        [metrics for _, metrics in _best(grid_results, count=len(grid_results))],
    )
    print(
        f"[RESULT] grid acceptance="
        f"{'PASS' if selection['acceptance']['passed'] else 'FAIL'} "
        f"candidate={selected.identifier} "
        f"policy={session.session_root / 'best_observed_policy.yaml'}",
        flush=True,
    )


def run_search(session: TuningSession, target_stage: str):
    strategy = str(session.config.get("search", {}).get("strategy", COARSE_STRATEGY))
    if strategy == GRID_STRATEGY:
        _run_grid_search(session, target_stage)
        return
    if strategy != COARSE_STRATEGY:
        raise ValueError(f"Unsupported mixed tuning search.strategy={strategy!r}.")

    if session.dry_run:
        _dry_run_candidates(session)
        return

    config = session.config
    search = config["search"]
    generic = _candidate_from_policy(session.base_policy_config)
    sine = _candidate_from_policy(session.sine_policy_config)
    baseline_results = _run_screen_candidates(
        session,
        (generic, sine),
        stage="baseline",
    )
    if target_stage == "baseline":
        return

    best_baseline = _best(baseline_results)[0][0]
    structure_candidates = [
        replace(
            best_baseline,
            integral_pole=0.0,
            filter_omega=float(pair[0]),
            filter_zeta=1.0,
            k_theta=float(pair[1]),
            omega_max=0.5,
            feedforward=True,
        )
        for pair in search["filter_inner_pairs"]
    ]
    structure_results = _run_screen_candidates(
        session,
        structure_candidates,
        stage="structure",
    )
    best_structure = _best(structure_results)[0][0]
    outer_candidates = [
        replace(
            best_structure,
            outer_omega=float(pair[0]),
            outer_zeta=float(pair[1]),
        )
        for pair in search["outer_pairs"]
    ]
    outer_results = _run_screen_candidates(
        session,
        outer_candidates,
        stage="outer",
    )

    screen_pool = [*baseline_results, *structure_results, *outer_results]
    current_best, current_metrics = _best(screen_pool)[0]
    integral_results: list[tuple[Candidate, dict[str, Any]]] = []
    constant_bias = float(
        current_metrics.get("constant_mean_abs_signed_error", float("nan"))
    )
    if math.isfinite(constant_bias) and constant_bias > float(
        search["integral_bias_threshold"]
    ):
        integral_candidates = [
            replace(current_best, integral_pole=float(value))
            for value in search["integral_poles"]
        ]
        integral_results = _run_screen_candidates(
            session,
            integral_candidates,
            stage="integral",
        )
        screen_pool.extend(integral_results)

    new_results = [*structure_results, *outer_results, *integral_results]
    best_before_recovery = _best(screen_pool)[0]
    baseline_ids = {generic.identifier, sine.identifier}
    recovery_needed = (
        not any(not bool(metrics.get("hard_failure", True)) for _, metrics in new_results)
        or best_before_recovery[0].identifier in baseline_ids
    )
    recovery_results: list[tuple[Candidate, dict[str, Any]]] = []
    if recovery_needed:
        recovery_base = best_before_recovery[0]
        recovery_candidates = [
            replace(
                recovery_base,
                **{
                    str(spec["field"]): getattr(recovery_base, str(spec["field"]))
                    * float(spec["scale"])
                },
            )
            for spec in search["recovery_perturbations"]
        ]
        recovery_results = _run_screen_candidates(
            session,
            recovery_candidates,
            stage="recovery",
        )
        screen_pool.extend(recovery_results)

    screen_selection = {
        "top_candidates": [
            {"candidate": _candidate_fields(candidate), "metrics": metrics}
            for candidate, metrics in _best(screen_pool, count=3)
        ],
        "baseline_candidate_ids": sorted(baseline_ids),
        "recovery_triggered": recovery_needed,
    }
    _save_yaml(screen_selection, session.session_root / "screen_selection.yaml")
    if target_stage == "screen":
        return

    validation = config["validation"]
    validation_candidates = _deduplicate(
        [candidate for candidate, _ in _best(screen_pool, count=3)]
        + [generic, sine]
    )
    validation_results = [
        (
            candidate,
            session.run_trial(
                candidate,
                stage="validate",
                seed=int(validation["seed"]),
                num_envs=int(validation["num_envs"]),
                episodes=int(validation["episodes"]),
                min_episodes_per_type=int(validation["min_episodes_per_type"]),
            ),
        )
        for candidate in validation_candidates
    ]
    selected, selected_metrics = _best(validation_results)[0]
    validation_by_id = {
        candidate.identifier: metrics for candidate, metrics in validation_results
    }
    generic_metrics = validation_by_id[generic.identifier]
    sine_metrics = validation_by_id[sine.identifier]
    acceptance = _acceptance(config, selected_metrics)
    effective_policy = session.effective_policy_config(selected)
    selection = {
        "candidate": _candidate_fields(selected),
        "selected_validation": selected_metrics,
        "generic_validation": generic_metrics,
        "sine_validation": sine_metrics,
        "comparison_to_generic": _comparison(selected_metrics, generic_metrics),
        "comparison_to_sine": _comparison(selected_metrics, sine_metrics),
        "acceptance": acceptance,
        "policy_config": effective_policy,
    }
    _save_yaml(selection, session.session_root / "final_selection.yaml")
    _write_csv(
        session.session_root / "validation_summary.csv",
        [metrics for _, metrics in _best(validation_results, count=len(validation_results))],
    )
    output_policy_path = _resolve_path(config["output_policy_config"])
    if acceptance["passed"]:
        _save_yaml(effective_policy, output_policy_path)
    print(
        f"[RESULT] acceptance={'PASS' if acceptance['passed'] else 'FAIL'} "
        f"candidate={selected.identifier} "
        + (
            f"policy={output_policy_path}"
            if acceptance["passed"]
            else "policy=not_written"
        ),
        flush=True,
    )


def main():
    args = _parse_args()
    config_path = args.config.expanduser().resolve()
    config = _load_yaml(config_path)
    if args.analyze_run is not None:
        rows, _ = analyze_rollout(args.analyze_run)
        metrics = aggregate_mixed_episode_metrics(
            rows,
            required_types=config["required_types"],
            min_episodes_per_type=0,
            full_saturation_rate=float(config["acceptance"]["full_saturation_rate"]),
            tail_saturation_rate=float(config["acceptance"]["tail_saturation_rate"]),
        )
        _write_csv(args.analyze_run / "episode_metrics.csv", rows)
        _save_yaml(metrics, args.analyze_run / "tuning_metrics.yaml")
        print(yaml.safe_dump(metrics, sort_keys=False))
        return

    session = TuningSession(
        config,
        python=args.python,
        resume=args.resume,
        dry_run=args.dry_run,
    )
    _save_yaml(config, session.session_root / "input_tuning_config.yaml")
    run_search(session, args.stage)


if __name__ == "__main__":
    main()
