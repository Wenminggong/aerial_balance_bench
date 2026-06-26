#!/usr/bin/env python3
"""Run velocity-interface CPID tuning candidates for max_acc=5.0."""

from __future__ import annotations

import argparse
import csv
import math
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_ROOT = PROJECT_ROOT / "logs" / "cpid_velocity_tuning"
CONFIG_ROOT = LOG_ROOT / "configs"


@dataclass(frozen=True)
class Candidate:
    name: str
    angle_kp: float
    angle_ti: float
    angle_td: float
    max_theta_change: float
    velocity_kp: float
    velocity_ti: float
    velocity_td: float


BASELINE = Candidate(
    name="baseline_old_gains",
    angle_kp=0.5,
    angle_ti=1.5,
    angle_td=0.6,
    max_theta_change=0.001745329252,
    velocity_kp=10.0,
    velocity_ti=1000.0,
    velocity_td=0.0,
)


STAGES: dict[str, list[Candidate]] = {
    "baseline": [BASELINE],
    "coarse": [
        Candidate("c001_low_inner", 0.5, 1.5, 0.6, 0.001745329252, 1.0, 1000.0, 0.0),
        Candidate("c002_mid_inner", 0.5, 1.5, 0.6, 0.001745329252, 2.0, 1000.0, 0.0),
        Candidate("c003_inner_damped", 0.5, 1.5, 0.6, 0.001745329252, 3.0, 1000.0, 0.2),
        Candidate("c004_tighter_theta", 0.5, 1.5, 0.6, 0.0007, 5.0, 1000.0, 0.2),
        Candidate("c005_fast_outer_tight", 0.7, 2.0, 0.5, 0.0007, 4.0, 1000.0, 0.2),
        Candidate("c006_slow_outer_tight", 0.35, 2.0, 0.7, 0.0007, 4.0, 1000.0, 0.2),
        Candidate("c007_tight_no_damp", 0.5, 2.0, 0.6, 0.0007, 4.0, 1000.0, 0.0),
        Candidate("c008_medium_theta", 0.5, 2.0, 0.6, 0.0010, 4.0, 1000.0, 0.2),
        Candidate("c009_small_theta", 0.5, 2.0, 0.6, 0.0005, 4.0, 1000.0, 0.2),
        Candidate("c010_inner_i", 0.5, 1.0, 0.6, 0.0007, 4.0, 4.0, 0.2),
        Candidate("c011_more_damp", 0.5, 2.0, 0.8, 0.0007, 4.0, 1000.0, 0.4),
        Candidate("c012_aggressive_damped", 0.7, 2.0, 0.8, 0.0010, 5.0, 1000.0, 0.4),
    ],
    "refine": [
        Candidate("r001_theta0006_v3", 0.45, 2.0, 0.7, 0.0006, 3.0, 1000.0, 0.2),
        Candidate("r002_theta0007_v3", 0.45, 2.0, 0.7, 0.0007, 3.0, 1000.0, 0.2),
        Candidate("r003_theta0008_v3", 0.45, 2.0, 0.7, 0.0008, 3.0, 1000.0, 0.2),
        Candidate("r004_theta0007_v2", 0.45, 2.0, 0.7, 0.0007, 2.0, 1000.0, 0.2),
        Candidate("r005_theta0007_v4", 0.45, 2.0, 0.7, 0.0007, 4.0, 1000.0, 0.2),
        Candidate("r006_kp055_td07", 0.55, 2.0, 0.7, 0.0007, 3.0, 1000.0, 0.2),
        Candidate("r007_kp055_td06", 0.55, 2.0, 0.6, 0.0007, 3.0, 1000.0, 0.2),
        Candidate("r008_kp045_td09", 0.45, 2.0, 0.9, 0.0007, 3.0, 1000.0, 0.3),
        Candidate("r009_ti15", 0.45, 1.5, 0.7, 0.0007, 3.0, 1000.0, 0.2),
        Candidate("r010_inner_ti20", 0.45, 2.0, 0.7, 0.0007, 3.0, 20.0, 0.2),
    ],
    "refine2": [
        Candidate("q001_theta0008_v2", 0.45, 2.0, 0.7, 0.0008, 2.0, 1000.0, 0.2),
        Candidate("q002_theta0008_v25", 0.45, 2.0, 0.7, 0.0008, 2.5, 1000.0, 0.2),
        Candidate("q003_theta0008_v35", 0.45, 2.0, 0.7, 0.0008, 3.5, 1000.0, 0.2),
        Candidate("q004_theta0009_v25", 0.45, 2.0, 0.7, 0.0009, 2.5, 1000.0, 0.2),
        Candidate("q005_theta00075_v25", 0.45, 2.0, 0.7, 0.00075, 2.5, 1000.0, 0.2),
        Candidate("q006_kp040_v25", 0.40, 2.0, 0.7, 0.0008, 2.5, 1000.0, 0.2),
        Candidate("q007_kp050_v25", 0.50, 2.0, 0.7, 0.0008, 2.5, 1000.0, 0.2),
        Candidate("q008_td08_v25", 0.45, 2.0, 0.8, 0.0008, 2.5, 1000.0, 0.25),
    ],
    "high_refine": [
        Candidate("h001_v8", 0.5, 1.5, 0.6, 0.001745329252, 8.0, 1000.0, 0.0),
        Candidate("h002_v12", 0.5, 1.5, 0.6, 0.001745329252, 12.0, 1000.0, 0.0),
        Candidate("h003_v10_d01", 0.5, 1.5, 0.6, 0.001745329252, 10.0, 1000.0, 0.1),
        Candidate("h004_v8_d01", 0.5, 1.5, 0.6, 0.001745329252, 8.0, 1000.0, 0.1),
        Candidate("h005_outer_td08_v10", 0.5, 1.5, 0.8, 0.001745329252, 10.0, 1000.0, 0.0),
        Candidate("h006_outer_ti2_td08_v10_d01", 0.5, 2.0, 0.8, 0.001745329252, 10.0, 1000.0, 0.1),
        Candidate("h007_theta0012_v8_d02", 0.5, 2.0, 0.8, 0.0012, 8.0, 1000.0, 0.2),
        Candidate("h008_theta0010_v6_d03", 0.5, 2.0, 0.8, 0.0010, 6.0, 1000.0, 0.3),
        Candidate("h009_c011_theta0008", 0.5, 2.0, 0.8, 0.0008, 4.0, 1000.0, 0.4),
        Candidate("h010_c011_v5", 0.5, 2.0, 0.8, 0.0007, 5.0, 1000.0, 0.4),
        Candidate("h011_c011_v6", 0.5, 2.0, 0.8, 0.0007, 6.0, 1000.0, 0.4),
        Candidate("h012_c011_d05", 0.5, 2.0, 0.8, 0.0007, 4.0, 1000.0, 0.5),
        Candidate("h013_c011_atd09", 0.5, 2.0, 0.9, 0.0007, 4.0, 1000.0, 0.4),
        Candidate("h014_c011_ti18", 0.5, 1.8, 0.8, 0.0007, 4.0, 1000.0, 0.4),
    ],
    "final": [],
}


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as stream:
        return yaml.safe_load(stream) or {}


def _save_yaml(data: dict[str, Any], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as stream:
        yaml.safe_dump(data, stream, sort_keys=False)


def _candidate_from_args(args: argparse.Namespace) -> Candidate:
    missing = [
        name
        for name in (
            "angle_kp",
            "angle_ti",
            "angle_td",
            "max_theta_change",
            "velocity_kp",
            "velocity_ti",
            "velocity_td",
        )
        if getattr(args, name) is None
    ]
    if missing:
        raise ValueError(f"--stage custom requires: {', '.join('--' + name.replace('_', '-') for name in missing)}")
    return Candidate(
        name=args.name or "custom",
        angle_kp=float(args.angle_kp),
        angle_ti=float(args.angle_ti),
        angle_td=float(args.angle_td),
        max_theta_change=float(args.max_theta_change),
        velocity_kp=float(args.velocity_kp),
        velocity_ti=float(args.velocity_ti),
        velocity_td=float(args.velocity_td),
    )


def _policy_config(candidate: Candidate, max_acc: float) -> dict[str, Any]:
    return {
        "policy_name": "cpid",
        "angle_pid": {
            "kp": candidate.angle_kp,
            "ti": candidate.angle_ti,
            "td": candidate.angle_td,
            "max_theta_change": candidate.max_theta_change,
        },
        "velocity_pid": {
            "kp": candidate.velocity_kp,
            "ti": candidate.velocity_ti,
            "td": candidate.velocity_td,
            "max_acc": max_acc,
        },
        "state_predictor": {
            "enabled": False,
            "delay_step": 15,
            "solver": "rk4",
            "step_dt": "auto",
            "max_acc": "auto",
            "max_velocity": "auto",
            "plank_length": "auto",
            "rope_length": "auto",
            "gravity": "auto",
            "ball_mass": 0.0005,
            "ball_radius": "auto",
            "ball_inertia_ratio": 0.4,
            "epsilon": 1.0e-9,
        },
    }


def _env_config(max_acc: float, num_envs: int, seed: int) -> dict[str, Any]:
    config = _load_yaml(PROJECT_ROOT / "environments" / "configs" / "template_eval_delay_free.yaml")
    config.setdefault("env", {})
    config["env"]["seed"] = seed
    config["env"]["num_envs"] = num_envs
    config.setdefault("velocity_interface", {})
    config["velocity_interface"]["max_acc"] = max_acc
    config["velocity_interface"]["max_velocity"] = 0.0
    config.setdefault("runner", {})
    config["runner"]["render"] = False
    config["runner"]["save_rollout"] = False
    config.setdefault("logging", {})
    config["logging"]["root_dir"] = "logs/cpid_velocity_tuning"
    return config


def _run_config(stage: str, run_name: str, episodes: int) -> dict[str, Any]:
    return {
        "env_config": f"logs/cpid_velocity_tuning/configs/env_{stage}_{run_name}.yaml",
        "policy_config": f"logs/cpid_velocity_tuning/configs/policy_{stage}_{run_name}.yaml",
        "runner": {
            "target_episodes": episodes,
            "max_steps": None,
            "render": False,
            "save_rollout": False,
        },
        "logging": {
            "root_dir": "logs/cpid_velocity_tuning",
            "run_name": run_name,
        },
    }


def _read_last_summary(run_dir: Path) -> dict[str, str]:
    summary_path = run_dir / "summary.csv"
    with open(summary_path, encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise RuntimeError(f"No summary rows found in {summary_path}")
    return rows[-1]


def _score(row: dict[str, str]) -> float:
    success = float(row.get("benchmark_success_rate", "nan"))
    error = float(row.get("benchmark_steady_state_error", "nan"))
    convergence = float(row.get("benchmark_convergence_time", "nan"))
    reward = float(row.get("mean_reward", "nan"))
    if math.isnan(success) or math.isnan(error) or math.isnan(convergence):
        return float("-inf")
    return 1000.0 * success - 100.0 * error - convergence + 0.01 * reward


def _append_tuning_row(path: Path, row: dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with open(path, "a", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _run_candidate(args: argparse.Namespace, stage: str, index: int, candidate: Candidate) -> dict[str, Any]:
    run_name = f"{stage}_{index:03d}_{candidate.name}"
    run_dir = LOG_ROOT / run_name
    if args.skip_existing and (run_dir / "summary.csv").exists():
        summary = _read_last_summary(run_dir)
        status = "skipped_existing"
        elapsed = 0.0
    else:
        policy_path = CONFIG_ROOT / f"policy_{stage}_{run_name}.yaml"
        env_path = CONFIG_ROOT / f"env_{stage}_{run_name}.yaml"
        run_path = CONFIG_ROOT / f"run_{stage}_{run_name}.yaml"
        _save_yaml(_policy_config(candidate, args.max_acc), policy_path)
        _save_yaml(_env_config(args.max_acc, args.num_envs, args.seed), env_path)
        _save_yaml(_run_config(stage, run_name, args.episodes), run_path)

        command = [
            args.python_bin,
            str(PROJECT_ROOT / "scripts" / "cpid_policy_eval.py"),
            "--config",
            str(run_path),
            "--episodes",
            str(args.episodes),
            "--num_envs",
            str(args.num_envs),
            "--headless",
        ]
        log_path = LOG_ROOT / "driver_logs" / f"{run_name}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        start = time.perf_counter()
        with open(log_path, "w", encoding="utf-8") as log_stream:
            result = subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                stdout=log_stream,
                stderr=subprocess.STDOUT,
                check=False,
            )
        elapsed = time.perf_counter() - start
        if result.returncode != 0:
            raise RuntimeError(f"{run_name} failed with exit code {result.returncode}; see {log_path}")
        if not (run_dir / "summary.csv").exists():
            raise RuntimeError(f"{run_name} did not produce summary.csv; see {log_path}")
        summary = _read_last_summary(run_dir)
        status = "ok"

    tuning_row: dict[str, Any] = {
        "stage": stage,
        "candidate": candidate.name,
        "run_name": run_name,
        "status": status,
        "elapsed_s": f"{elapsed:.3f}",
        "max_acc": args.max_acc,
        "angle_kp": candidate.angle_kp,
        "angle_ti": candidate.angle_ti,
        "angle_td": candidate.angle_td,
        "max_theta_change": candidate.max_theta_change,
        "velocity_kp": candidate.velocity_kp,
        "velocity_ti": candidate.velocity_ti,
        "velocity_td": candidate.velocity_td,
        "score": _score(summary),
    }
    for key in (
        "target_episodes",
        "completed_episodes",
        "rollout_steps",
        "mean_reward",
        "benchmark_success_rate",
        "benchmark_steady_state_error",
        "benchmark_steady_state_error_std",
        "benchmark_convergence_time",
        "benchmark_convergence_time_std",
        "benchmark_climbing_time",
        "benchmark_climbing_time_std",
    ):
        tuning_row[key] = summary.get(key, "")
    _append_tuning_row(LOG_ROOT / "tuning_results.csv", tuning_row)
    return tuning_row


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=tuple(STAGES.keys()) + ("custom",), default="baseline")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--num-envs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument("--max-acc", type=float, default=5.0)
    parser.add_argument("--python-bin", type=str, default=sys.executable)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--angle-kp", type=float, default=None)
    parser.add_argument("--angle-ti", type=float, default=None)
    parser.add_argument("--angle-td", type=float, default=None)
    parser.add_argument("--max-theta-change", type=float, default=None)
    parser.add_argument("--velocity-kp", type=float, default=None)
    parser.add_argument("--velocity-ti", type=float, default=None)
    parser.add_argument("--velocity-td", type=float, default=None)
    return parser.parse_args()


def main():
    args = _parse_args()
    if args.stage == "custom":
        candidates = [_candidate_from_args(args)]
    else:
        candidates = STAGES[args.stage]
    if not candidates:
        raise ValueError(f"Stage {args.stage!r} has no built-in candidates; use --stage custom.")

    for index, candidate in enumerate(candidates, start=1):
        print(f"[INFO] Running {args.stage} candidate {index}/{len(candidates)}: {candidate.name}", flush=True)
        row = _run_candidate(args, args.stage, index, candidate)
        print(
            "[INFO] "
            f"{row['run_name']} success={row['benchmark_success_rate']} "
            f"error={row['benchmark_steady_state_error']} "
            f"conv={row['benchmark_convergence_time']} score={row['score']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
