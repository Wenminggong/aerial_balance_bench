#!/usr/bin/env python3
"""Run the staged NFFB sine-reference parameter search."""

from __future__ import annotations

import argparse
import csv
import math
import subprocess
import sys
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_PARENT = PROJECT_ROOT.parent
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from aerial_balance_bench.utils.nffb_tuning import (  # noqa: E402
    aggregate_episode_metrics,
    analyze_rollout,
    evaluate_acceptance,
    tuning_rank_key,
)


DEFAULT_CONFIG = PROJECT_ROOT / "baselines" / "configs" / "nffb_sine_tuning.yaml"
EVAL_SCRIPT = PROJECT_ROOT / "scripts" / "nffb_policy_eval.py"
STAGE_ORDER = ("baseline", "filter", "outer", "local", "stress", "validate")


@dataclass(frozen=True)
class Candidate:
    """Searchable NFFB parameters."""

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
        help="Run through this stage, including its dependencies.",
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
        help="Analyze one existing run directory and exit.",
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


def _default_candidate(policy_config: Mapping[str, Any]) -> Candidate:
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


def _deduplicate(candidates: Iterable[Candidate]) -> list[Candidate]:
    result: list[Candidate] = []
    seen: set[Candidate] = set()
    for candidate in candidates:
        if candidate not in seen:
            seen.add(candidate)
            result.append(candidate)
    return result


def _stress_gate(metrics: Mapping[str, Any], acceptance: Mapping[str, float]) -> bool:
    return bool(
        int(metrics.get("termination_count", 0)) == 0
        and int(metrics.get("boundary_violation_count", 0)) == 0
        and int(metrics.get("nonfinite_count", 0)) == 0
        and float(metrics.get("min_amplitude_gain", float("-inf")))
        >= float(acceptance["stress_gain_min"])
        and float(metrics.get("max_amplitude_gain", float("inf")))
        <= float(acceptance["stress_gain_max"])
        and float(metrics.get("p95_abs_phase_error_deg", float("inf")))
        <= float(acceptance["stress_phase_error_deg"])
        and float(metrics.get("p95_steady_nrmse", float("inf")))
        <= float(acceptance["stress_steady_nrmse"])
    )


def _best_stress(
    candidate_results: Iterable[tuple[Candidate, Mapping[str, Any]]],
    acceptance: Mapping[str, float],
    *,
    count: int = 1,
) -> list[tuple[Candidate, Mapping[str, Any]]]:
    candidate_results = list(candidate_results)
    passing = [
        result
        for result in candidate_results
        if _stress_gate(result[1], acceptance)
    ]
    return _best(passing or candidate_results, count=count)


class TuningSession:
    """Orchestrate resumable NFFB evaluation subprocesses."""

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
        self.base_run_config = _load_yaml(self.base_run_config_path)
        self.base_env_config = _load_yaml(self.base_env_config_path)
        self.base_policy_config = _load_yaml(self.base_policy_config_path)

        output_root = _resolve_path(config["output_root"])
        self.session_root = output_root / str(config.get("session_name", "phase0_acc5"))
        self.generated_dir = self.session_root / "_generated"
        self.runs_root = self.session_root / "runs"
        self.stdout_root = self.session_root / "stdout"
        for directory in (
            self.generated_dir,
            self.runs_root,
            self.stdout_root,
        ):
            directory.mkdir(parents=True, exist_ok=True)

        self.results: list[dict[str, Any]] = []
        self.episode_rows: dict[str, list[dict[str, Any]]] = {}

    def _environment_config(
        self,
        *,
        seed: int,
        num_envs: int,
        profile: Mapping[str, float] | None,
    ) -> Path:
        env_config = deepcopy(self.base_env_config)
        env_config.setdefault("env", {})
        env_config["env"]["seed"] = seed
        env_config["env"]["num_envs"] = num_envs
        env_config["env"]["episode_length_s"] = float(
            self.config["experiment"]["episode_length_s"]
        )
        task = env_config.setdefault("unified_tracking_task", {})
        task["trajectory_types"] = ["sine"]
        task["trajectory_type_weights"] = [1.0]
        task["phase_range"] = [0.0, 0.0]
        if profile is not None:
            task["dynamic_center_range"] = [profile["center"], profile["center"]]
            task["amplitude_range"] = [profile["amplitude"], profile["amplitude"]]
            task["period_range"] = [profile["period"], profile["period"]]
        velocity = env_config.setdefault("velocity_interface", {})
        velocity["max_acc"] = 5.0
        velocity["max_velocity"] = 0.0
        robustness = env_config.setdefault("robustness", {})
        robustness["enabled"] = False
        robustness["action_delay_enabled"] = False
        robustness["delay_step"] = 0

        profile_name = "random" if profile is None else str(profile["name"])
        path = self.generated_dir / "env" / f"{profile_name}_seed{seed}_n{num_envs}.yaml"
        _save_yaml(env_config, path)
        return path

    def _run_config(
        self,
        candidate: Candidate,
        *,
        trial_id: str,
        env_config_path: Path,
        episodes: int,
    ) -> Path:
        run_config = deepcopy(self.base_run_config)
        run_config["env_config"] = str(env_config_path)
        run_config["policy_config"] = str(self.base_policy_config_path)
        run_config["policy_overrides"] = candidate.overrides()
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
        profile: Mapping[str, float] | None = None,
    ) -> dict[str, Any]:
        profile_name = "random" if profile is None else str(profile["name"])
        trial_id = f"{stage}__{candidate.identifier}__{profile_name}__seed{seed}"
        run_dir = self.runs_root / trial_id
        env_config_path = self._environment_config(
            seed=seed,
            num_envs=num_envs,
            profile=profile,
        )
        run_config_path = self._run_config(
            candidate,
            trial_id=trial_id,
            env_config_path=env_config_path,
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
                    f"Trial directory exists; use --resume or choose another session: {run_dir}"
                )
            print("[RUN]", " ".join(command), flush=True)
            if self.dry_run:
                return {
                    **_candidate_fields(candidate),
                    "stage": stage,
                    "profile": profile_name,
                    "seed": seed,
                    "dry_run": True,
                }
            stdout_path = self.stdout_root / f"{trial_id}.log"
            with open(stdout_path, "w", encoding="utf-8") as stream:
                result = subprocess.run(
                    command,
                    cwd=PROJECT_ROOT,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            rollout_path = run_dir / "rollout.npz"
            if result.returncode != 0 or not rollout_path.exists():
                raise RuntimeError(
                    "NFFB trial failed or did not produce rollout.npz "
                    f"(status={result.returncode}); see {stdout_path}"
                )
        else:
            print(f"[RESUME] {trial_id}", flush=True)

        episode_rows, metrics = analyze_rollout(run_dir)
        result_row = {
            **_candidate_fields(candidate),
            "stage": stage,
            "profile": profile_name,
            "seed": seed,
            **metrics,
        }
        self.results.append(result_row)
        self.episode_rows[trial_id] = episode_rows
        _write_csv(run_dir / "episode_metrics.csv", episode_rows)
        _save_yaml(result_row, run_dir / "tuning_metrics.yaml")
        _save_yaml(
            {
                "candidate": _candidate_fields(candidate),
                "stage": stage,
                "profile": profile_name,
                "seed": seed,
                "episodes": episodes,
                "num_envs": num_envs,
                "command": command,
            },
            run_dir / "tuning_trial.yaml",
        )
        self.write_leaderboard()
        return result_row

    def write_leaderboard(self):
        if not self.results:
            return
        sorted_results = sorted(self.results, key=tuning_rank_key)
        _write_csv(self.session_root / "leaderboard.csv", sorted_results)
        _save_yaml(
            {"trials": sorted_results},
            self.session_root / "leaderboard.yaml",
        )

    def combine_trials(
        self,
        candidate: Candidate,
        trial_results: Iterable[Mapping[str, Any]],
        *,
        stage: str,
    ) -> dict[str, Any]:
        rows: list[dict[str, Any]] = []
        for trial_result in trial_results:
            run_name = Path(str(trial_result["run_dir"])).name
            rows.extend(self.episode_rows[run_name])
        aggregate = {
            **_candidate_fields(candidate),
            "stage": stage,
            **aggregate_episode_metrics(rows),
        }
        return aggregate


def _screen_settings(config: Mapping[str, Any]) -> tuple[int, int, int]:
    screen = config["screen"]
    return int(screen["seed"]), int(screen["num_envs"]), int(screen["episodes"])


def _run_random_candidates(
    session: TuningSession,
    candidates: Iterable[Candidate],
    *,
    stage: str,
) -> list[tuple[Candidate, dict[str, Any]]]:
    seed, num_envs, episodes = _screen_settings(session.config)
    results = []
    for candidate in _deduplicate(candidates):
        existing = next(
            (
                result
                for result in session.results
                if result.get("candidate_id") == candidate.identifier
                and result.get("profile") == "random"
                and int(result.get("seed", -1)) == seed
                and int(result.get("episode_count", 0)) >= episodes
            ),
            None,
        )
        if existing is not None:
            print(
                f"[REUSE-CANDIDATE] {candidate.identifier} from {existing['stage']}",
                flush=True,
            )
            results.append((candidate, existing))
            continue
        result = session.run_trial(
            candidate,
            stage=stage,
            seed=seed,
            num_envs=num_envs,
            episodes=episodes,
        )
        results.append((candidate, result))
    return results


def _best(
    candidate_results: Iterable[tuple[Candidate, Mapping[str, Any]]],
    count: int = 1,
) -> list[tuple[Candidate, Mapping[str, Any]]]:
    return sorted(candidate_results, key=lambda item: tuning_rank_key(item[1]))[:count]


def _run_stress(
    session: TuningSession,
    candidates: Iterable[Candidate],
    *,
    stage: str,
    profile_names: Iterable[str] | None = None,
    stop_after_hard_failure: bool = False,
) -> list[tuple[Candidate, dict[str, Any]]]:
    results = []
    seed = int(session.config["screen"]["seed"])
    profiles = session.config["stress_profiles"]
    if profile_names is not None:
        requested_names = set(profile_names)
        profiles = [
            profile for profile in profiles if str(profile["name"]) in requested_names
        ]
        missing_names = requested_names - {
            str(profile["name"]) for profile in profiles
        }
        if missing_names:
            raise ValueError(
                "Unknown stress profile name(s): "
                + ", ".join(sorted(missing_names))
            )
    for candidate in _deduplicate(candidates):
        trials = []
        for profile in profiles:
            profile_name = str(profile["name"])
            existing = next(
                (
                    result
                    for result in session.results
                    if result.get("candidate_id") == candidate.identifier
                    and result.get("profile") == profile_name
                    and int(result.get("seed", -1)) == seed
                    and int(result.get("episode_count", 0)) >= 1
                ),
                None,
            )
            if existing is not None:
                print(
                    f"[REUSE-PROFILE] {candidate.identifier} "
                    f"profile={profile_name} from {existing['stage']}",
                    flush=True,
                )
                trial = existing
            else:
                trial = session.run_trial(
                    candidate,
                    stage=stage,
                    seed=seed,
                    num_envs=1,
                    episodes=1,
                    profile=profile,
                )
            trials.append(trial)
            if stop_after_hard_failure and bool(trial.get("hard_failure", True)):
                break
        aggregate = session.combine_trials(candidate, trials, stage=stage)
        results.append((candidate, aggregate))
    _write_csv(
        session.session_root / f"{stage}_summary.csv",
        [metrics for _, metrics in sorted(results, key=lambda item: tuning_rank_key(item[1]))],
    )
    return results


def _effective_policy_config(
    base_policy_config: Mapping[str, Any],
    candidate: Candidate,
) -> dict[str, Any]:
    effective = deepcopy(dict(base_policy_config))
    policy = effective.get("nffb_policy")
    if not isinstance(policy, dict):
        policy = effective
    _deep_update(policy, candidate.overrides())
    if "policy_name" in policy:
        policy["policy_name"] = "nffb_sine_phase0_acc5"
    else:
        policy["name"] = "nffb_sine_phase0_acc5"
    return effective


def _stage_enabled(target_stage: str, stage: str) -> bool:
    if target_stage == "all":
        return True
    return STAGE_ORDER.index(stage) <= STAGE_ORDER.index(target_stage)


def run_search(session: TuningSession, target_stage: str):
    config = session.config
    search = config["search"]
    acceptance_cfg = config["acceptance"]
    default = _default_candidate(session.base_policy_config)
    recommended_cfg = search["recommended"]
    recommended = Candidate(
        outer_omega=float(recommended_cfg["outer_omega"]),
        outer_zeta=float(recommended_cfg["outer_zeta"]),
        integral_pole=0.0,
        filter_omega=float(recommended_cfg["filter_omega"]),
        filter_zeta=float(recommended_cfg["filter_zeta"]),
        k_theta=float(recommended_cfg["k_theta"]),
        omega_max=float(recommended_cfg["omega_max"]),
    )

    baseline_results = _run_random_candidates(
        session,
        (default, recommended),
        stage="baseline",
    )
    if target_stage == "baseline" or session.dry_run:
        return

    filter_candidates = [
        replace(
            default,
            filter_omega=float(pair[0]),
            k_theta=float(pair[1]),
            integral_pole=0.0,
            filter_zeta=1.0,
            omega_max=0.5,
            feedforward=True,
        )
        for pair in search["filter_inner_pairs"]
    ]
    filter_results = _run_random_candidates(
        session,
        filter_candidates,
        stage="filter",
    )
    top_filters = _best(filter_results, count=2)
    if target_stage == "filter":
        return

    best_filter = top_filters[0][0]
    second_filter = top_filters[1][0]
    primary_outer_results: list[tuple[Candidate, dict[str, Any]]] = []
    damping_ratios = sorted(float(value) for value in search["outer_damping_ratios"])
    outer_frequencies = sorted(float(value) for value in search["outer_frequencies"])
    for frequency_index, outer_omega in enumerate(outer_frequencies):
        lowest_damping_candidate = replace(
            best_filter,
            outer_omega=outer_omega,
            outer_zeta=damping_ratios[0],
        )
        lowest_damping_result = _run_random_candidates(
            session,
            (lowest_damping_candidate,),
            stage="outer",
        )
        primary_outer_results.extend(lowest_damping_result)
        if (
            frequency_index > 0
            and bool(lowest_damping_result[0][1].get("hard_failure", True))
        ):
            print(
                "[PRUNE] Lowest outer damping is infeasible at "
                f"omega_o={outer_omega:g}; skipping larger damping/frequency values.",
                flush=True,
            )
            break
        remaining_candidates = [
            replace(
                best_filter,
                outer_omega=outer_omega,
                outer_zeta=outer_zeta,
            )
            for outer_zeta in damping_ratios[1:]
        ]
        primary_outer_results.extend(
            _run_random_candidates(
                session,
                remaining_candidates,
                stage="outer",
            )
        )

    outer_seed_results = [top_filters[0], *primary_outer_results]
    top_outer = _best(outer_seed_results, count=3)
    cross_candidates = [
        replace(
            second_filter,
            outer_omega=candidate.outer_omega,
            outer_zeta=candidate.outer_zeta,
        )
        for candidate, _ in top_outer
    ]
    cross_results: list[tuple[Candidate, dict[str, Any]]] = []
    for cross_candidate in _deduplicate(cross_candidates):
        if cross_candidate == second_filter:
            cross_results.append(top_filters[1])
        else:
            cross_results.extend(
                _run_random_candidates(
                    session,
                    (cross_candidate,),
                    stage="outer_cross",
                )
            )
    outer_results = [*top_filters, *primary_outer_results, *cross_results]
    if target_stage == "outer":
        return

    best_outer = _best(outer_results, count=1)[0][0]
    delta = float(search["local_fraction"])
    local_candidates = [
        best_outer,
        replace(best_outer, filter_omega=best_outer.filter_omega * (1.0 - delta)),
        replace(best_outer, filter_omega=best_outer.filter_omega * (1.0 + delta)),
        replace(best_outer, k_theta=best_outer.k_theta * (1.0 - delta)),
        replace(best_outer, k_theta=best_outer.k_theta * (1.0 + delta)),
        *[
            replace(best_outer, filter_zeta=float(value))
            for value in search["filter_damping_ratios"]
        ],
        *[
            replace(best_outer, outer_zeta=float(value))
            for value in search.get("local_outer_damping_ratios", ())
        ],
    ]
    local_results = _run_random_candidates(
        session,
        local_candidates,
        stage="local",
    )
    best_local, best_local_metrics = _best(
        [*outer_results, *local_results, *baseline_results],
        count=1,
    )[0]
    omega_saturation = float(
        best_local_metrics.get(
            "mean_steady_policy_omega_command_saturated_rate",
            0.0,
        )
    )
    if omega_saturation > 0.01 and math.isclose(best_local.omega_max, 0.5):
        omega_result = _run_random_candidates(
            session,
            (replace(best_local, omega_max=0.6),),
            stage="local_omega",
        )
        local_results.extend(omega_result)

    screen_pool = [
        *filter_results,
        *outer_results,
        *local_results,
        *baseline_results,
    ]
    top_screen = [candidate for candidate, _ in _best(screen_pool, count=3)]
    if target_stage == "local":
        return

    stress_results = _run_stress(session, top_screen, stage="stress")
    selected, selected_stress = _best_stress(
        stress_results,
        acceptance_cfg,
    )[0]

    if bool(selected_stress.get("hard_failure", True)):
        safe_fallbacks: list[Candidate] = []
        used_candidates = set(top_screen)
        for filter_omega in search.get(
            "safe_fallback_filter_frequencies",
            (10.0, 8.0, 6.0),
        ):
            matching = [
                result
                for result in screen_pool
                if result[0] not in used_candidates
                and math.isclose(
                    result[0].filter_omega,
                    float(filter_omega),
                    rel_tol=0.0,
                    abs_tol=1.0e-9,
                )
                and not bool(result[1].get("hard_failure", True))
            ]
            if not matching:
                continue
            nominal_outer = [
                result
                for result in matching
                if math.isclose(result[0].outer_omega, 1.0)
                and math.isclose(result[0].outer_zeta, 1.0)
            ]
            if nominal_outer:
                matching = nominal_outer
            fallback = _best(matching, count=1)[0][0]
            safe_fallbacks.append(fallback)
            used_candidates.add(fallback)
        if safe_fallbacks:
            safe_stress = _run_stress(
                session,
                safe_fallbacks,
                stage="safe_stress",
                stop_after_hard_failure=True,
            )
            stress_results.extend(safe_stress)
            selected, selected_stress = _best_stress(
                stress_results,
                acceptance_cfg,
            )[0]

    initial_high_bandwidth_failed = any(
        bool(metrics.get("hard_failure", True))
        for _, metrics in stress_results
        if metrics.get("stage") == "stress"
    )
    if (
        not _stress_gate(selected_stress, acceptance_cfg)
        and bool(search.get("extend_bandwidth_on_failure", True))
        and not initial_high_bandwidth_failed
        and float(
            selected_stress.get(
                "mean_policy_acceleration_saturated_rate",
                0.0,
            )
        )
        == 0.0
    ):
        extended_candidates = [
            replace(
                selected,
                filter_omega=float(pair[0]),
                k_theta=float(pair[1]),
            )
            for pair in search["extended_filter_inner_pairs"]
        ]
        extended_screen = _run_random_candidates(
            session,
            extended_candidates,
            stage="extended",
        )
        extended_top = [candidate for candidate, _ in _best(extended_screen, count=2)]
        extended_stress = _run_stress(
            session,
            extended_top,
            stage="extended_stress",
        )
        stress_results.extend(extended_stress)
        selected, selected_stress = _best_stress(
            stress_results,
            acceptance_cfg,
        )[0]

    if not _stress_gate(selected_stress, acceptance_cfg):
        recovery_candidates = [
            replace(
                default,
                outer_omega=1.0,
                outer_zeta=1.0,
                integral_pole=0.0,
                filter_omega=float(pair[0]),
                filter_zeta=1.0,
                k_theta=float(pair[1]),
                omega_max=0.5,
                feedforward=True,
            )
            for pair in search.get("recovery_filter_inner_pairs", ())
        ]
        recovery_candidates.extend(
            replace(
                default,
                outer_omega=float(values[0]),
                outer_zeta=1.0,
                integral_pole=0.0,
                filter_omega=float(values[1]),
                filter_zeta=1.0,
                k_theta=float(values[2]),
                omega_max=0.5,
                feedforward=True,
            )
            for values in search.get(
                "recovery_outer_filter_inner_candidates",
                (),
            )
        )
        recovery_critical = _run_stress(
            session,
            recovery_candidates,
            stage="recovery_critical",
            profile_names=("large_fast",),
        )
        viable_recovery = [
            result
            for result in recovery_critical
            if not bool(result[1].get("hard_failure", True))
        ]
        if viable_recovery:
            passing_critical = [
                result
                for result in viable_recovery
                if _stress_gate(result[1], acceptance_cfg)
            ]
            recovery_top = [
                candidate
                for candidate, _ in _best(
                    passing_critical or viable_recovery,
                    count=3,
                )
            ]
            recovery_stress = _run_stress(
                session,
                recovery_top,
                stage="recovery_stress",
            )
            stress_results.extend(recovery_stress)
            selected, selected_stress = _best_stress(
                stress_results,
                acceptance_cfg,
            )[0]

    if (
        _stress_gate(selected_stress, acceptance_cfg)
        and float(selected_stress.get("bias_over_1cm_rate", 0.0)) > 0.20
    ):
        integral_candidates = [
            replace(selected, integral_pole=float(alpha))
            for alpha in search["integral_poles"]
        ]
        integral_screen = _run_random_candidates(
            session,
            integral_candidates,
            stage="integral",
        )
        integral_best = _best(
            [(selected, best_local_metrics), *integral_screen],
            count=1,
        )[0][0]
        if integral_best != selected:
            integral_stress = _run_stress(
                session,
                (integral_best,),
                stage="integral_stress",
            )
            stress_results.extend(integral_stress)
            selected, selected_stress = _best_stress(
                stress_results,
                acceptance_cfg,
            )[0]

    ablation_candidate = replace(selected, feedforward=False)
    ablation_result = _run_random_candidates(
        session,
        (ablation_candidate,),
        stage="ablation",
    )[0][1]
    selected_screen_result = _run_random_candidates(
        session,
        (selected,),
        stage="selected_screen",
    )[0][1]
    feedforward_rmse_improvement = (
        1.0
        - float(selected_screen_result["mean_rmse"])
        / float(ablation_result["mean_rmse"])
        if float(ablation_result["mean_rmse"]) > 0.0
        else float("nan")
    )
    feedforward_phase_improvement = (
        float(ablation_result["p95_abs_phase_error_deg"])
        - float(selected_screen_result["p95_abs_phase_error_deg"])
    )
    if target_stage == "stress":
        _save_yaml(
            {
                "candidate": _candidate_fields(selected),
                "stress_metrics": selected_stress,
                "feedforward_rmse_improvement": feedforward_rmse_improvement,
                "feedforward_phase_improvement_deg": feedforward_phase_improvement,
            },
            session.session_root / "provisional_selection.yaml",
        )
        return

    validation = config["validation"]
    selected_trials = []
    baseline_trials = []
    for seed in validation["seeds"]:
        selected_trials.append(
            session.run_trial(
                selected,
                stage="validate_selected",
                seed=int(seed),
                num_envs=int(validation["num_envs"]),
                episodes=int(validation["episodes_per_seed"]),
            )
        )
        baseline_trials.append(
            session.run_trial(
                default,
                stage="validate_baseline",
                seed=int(seed),
                num_envs=int(validation["num_envs"]),
                episodes=int(validation["episodes_per_seed"]),
            )
        )
    selected_validation = session.combine_trials(
        selected,
        selected_trials,
        stage="validate_selected",
    )
    baseline_validation = session.combine_trials(
        default,
        baseline_trials,
        stage="validate_baseline",
    )
    acceptance = evaluate_acceptance(
        selected_validation,
        baseline_validation,
        selected_stress,
        acceptance_cfg,
        feedforward_rmse_improvement=feedforward_rmse_improvement,
        feedforward_phase_improvement_deg=feedforward_phase_improvement,
    )

    effective_policy = _effective_policy_config(session.base_policy_config, selected)
    selection = {
        "candidate": _candidate_fields(selected),
        "selected_validation": selected_validation,
        "baseline_validation": baseline_validation,
        "stress_metrics": selected_stress,
        "acceptance": acceptance,
        "policy_config": effective_policy,
    }
    _save_yaml(selection, session.session_root / "final_selection.yaml")
    _save_yaml(
        effective_policy,
        session.session_root / "nffb_sine_phase0_acc5.yaml",
    )
    _write_csv(
        session.session_root / "validation_summary.csv",
        [selected_validation, baseline_validation],
    )
    print(
        f"[RESULT] acceptance={'PASS' if acceptance['passed'] else 'FAIL'} "
        f"candidate={selected.identifier}",
        flush=True,
    )


def main():
    args = _parse_args()
    if args.analyze_run is not None:
        rows, metrics = analyze_rollout(args.analyze_run)
        _write_csv(args.analyze_run / "episode_metrics.csv", rows)
        _save_yaml(metrics, args.analyze_run / "tuning_metrics.yaml")
        print(yaml.safe_dump(metrics, sort_keys=False))
        return

    config_path = args.config.expanduser().resolve()
    config = _load_yaml(config_path)
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
