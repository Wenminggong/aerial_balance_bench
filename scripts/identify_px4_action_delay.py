#!/usr/bin/env python3
"""Offline PX4 vertical action-delay identification.

The script reads data collected by ``px4_action_delay_test.py`` and estimates
one effective command delay for each trial.  It then groups the estimates by
control mode and saves per-file results, mode-level statistics, and a comparison
plot.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable

import numpy as np


STRING_COLUMNS = {"phase", "control_mode", "excitation_type", "safety_state"}
CONTROL_MODES = ("position", "velocity", "acceleration", "thrust")
MODE_LABELS = {
    "position": "Position",
    "velocity": "Velocity",
    "acceleration": "Acceleration",
    "thrust": "Thrust",
}
MODE_DEFAULT_AR_ORDER = {
    "position": 2,
    "velocity": 1,
    "acceleration": 1,
    "thrust": 1,
}
MODE_DEFAULT_MAX_DELAY_S = {
    "position": 1.0,
    "velocity": 0.8,
    "acceleration": 0.6,
    "thrust": 0.6,
}


@dataclass(frozen=True)
class SignalSet:
    mode: str
    time_s: np.ndarray
    command: np.ndarray
    response: np.ndarray
    input_signal: str
    response_signal: str
    excitation_type: str
    dt_s: float


@dataclass(frozen=True)
class DelayEstimate:
    data_file: str
    trial_name: str
    control_mode: str
    excitation_type: str
    delay_s: float
    delay_ms: float
    delay_samples: float
    dt_s: float
    sample_count: int
    duration_s: float
    input_signal: str
    response_signal: str
    ar_order: int
    max_delay_s: float
    mse: float
    r2: float
    ar_only_r2: float
    input_mse_improvement: float
    curve_sharpness: float
    boundary_solution: bool
    quality: str
    method: str
    correlation_peak: float


@dataclass(frozen=True)
class ModeSummary:
    control_mode: str
    count: int
    mean_delay_s: float
    std_delay_s: float
    mean_delay_ms: float
    std_delay_ms: float
    mean_delay_samples: float
    std_delay_samples: float
    median_delay_ms: float
    min_delay_ms: float
    max_delay_ms: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Identify PX4 vertical action delay from offline action-delay test data."
        )
    )
    parser.add_argument(
        "data_path",
        type=Path,
        help="A samples file or a directory containing action-delay test data.",
    )
    parser.add_argument(
        "--data-source",
        choices=("real", "simulation"),
        default="real",
        help=(
            "Input data format. 'real' reads PX4 action-delay samples files; "
            "'simulation' reads benchmark rollout.npz files."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory for CSV/JSON/figure outputs. Defaults to "
            "<data_path>/delay_identification for directories, or the file parent."
        ),
    )
    parser.add_argument(
        "--max-delay-s",
        type=float,
        default=None,
        help="Override the mode-specific maximum delay scanned by the estimator.",
    )
    parser.add_argument(
        "--method",
        choices=("arx", "xcorr"),
        default="arx",
        help=(
            "Delay estimator to use. 'arx' estimates a pure input delay with "
            "the existing ARX residual scan; 'xcorr' estimates a total apparent "
            "delay from normalized cross-correlation."
        ),
    )
    parser.add_argument(
        "--lag-step-fraction",
        type=float,
        default=0.25,
        help="Delay-grid step as a fraction of the measured sample period.",
    )
    parser.add_argument(
        "--trim-start-s",
        type=float,
        default=0.0,
        help="Discard this much time from the start of each trial before fitting.",
    )
    parser.add_argument(
        "--trim-end-s",
        type=float,
        default=0.0,
        help="Discard this much time from the end of each trial before fitting.",
    )
    parser.add_argument(
        "--smooth-response-window-s",
        type=float,
        default=0.0,
        help=(
            "Optional centered moving-average response smoothing window. The "
            "default is 0 to avoid phase bias."
        ),
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=80,
        help="Minimum usable samples required after filtering and trimming.",
    )
    parser.add_argument(
        "--include-unsafe",
        action="store_true",
        help="Keep samples whose safety_state is not ok.",
    )
    parser.add_argument(
        "--response-target",
        choices=("auto", "drone", "beam"),
        default="auto",
        help=(
            "Response used for delay fitting. 'auto' uses beam_theta_rel when "
            "beam-system data is detected, otherwise it uses drone state."
        ),
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip writing the summary comparison plot.",
    )
    return parser.parse_args()


def discover_data_files(path: Path) -> list[Path]:
    if path.is_file():
        if path.suffix.lower() not in {".npz", ".csv"}:
            raise ValueError(f"Unsupported data file type: {path}")
        return [path]

    if not path.exists():
        raise FileNotFoundError(path)

    npz_files = [p for p in sorted(path.rglob("*.npz")) if is_candidate_data_file(p)]
    csv_files = [p for p in sorted(path.rglob("*.csv")) if is_candidate_data_file(p)]
    sample_npz_dirs = {p.parent for p in npz_files if p.name == "samples.npz"}

    files: list[Path] = []
    files.extend(npz_files)
    for csv_path in csv_files:
        if csv_path.name == "samples.csv" and csv_path.parent in sample_npz_dirs:
            continue
        files.append(csv_path)
    return sorted(files)


def discover_simulation_data_files(path: Path) -> list[Path]:
    if path.is_file():
        if path.name != "rollout.npz":
            raise ValueError(f"Simulation mode expects rollout.npz, got: {path}")
        return [path]

    if not path.exists():
        raise FileNotFoundError(path)

    return sorted(path.rglob("rollout.npz"))


def is_candidate_data_file(path: Path) -> bool:
    if path.parent.name == "delay_identification":
        return False
    if path.name.startswith("action_delay_"):
        return False
    return True


def load_data_file(path: Path) -> dict[str, np.ndarray]:
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=True) as data:
            return {key: np.asarray(data[key]) for key in data.files}
    if path.suffix.lower() == ".csv":
        return load_csv_file(path)
    raise ValueError(f"Unsupported data file type: {path}")


def load_csv_file(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise ValueError("empty CSV file")

    columns: dict[str, list[object]] = {name: [] for name in reader.fieldnames or []}
    for row in rows:
        for name, raw_value in row.items():
            if name in STRING_COLUMNS:
                columns[name].append("" if raw_value is None else str(raw_value))
                continue
            columns[name].append(parse_float(raw_value))

    arrays: dict[str, np.ndarray] = {}
    for name, values in columns.items():
        if name in STRING_COLUMNS:
            arrays[name] = np.asarray(values, dtype=str)
        else:
            arrays[name] = np.asarray(values, dtype=np.float64)
    return arrays


def parse_float(raw_value: object) -> float:
    if raw_value is None:
        return math.nan
    text = str(raw_value).strip()
    if not text:
        return math.nan
    try:
        return float(text)
    except ValueError:
        return math.nan


def load_metadata(path: Path) -> dict[str, object]:
    metadata_path = path.parent / "metadata.json"
    if not metadata_path.exists():
        return {}
    try:
        with metadata_path.open() as f:
            loaded = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}
    if isinstance(loaded, dict):
        return loaded
    return {}


def infer_control_mode(data: dict[str, np.ndarray], path: Path) -> str:
    if "control_mode" in data and len(data["control_mode"]) > 0:
        mode = normalise_control_mode(str(data["control_mode"][0]))
        if mode:
            return mode

    metadata = load_metadata(path)
    parameters = metadata.get("parameters", {})
    if isinstance(parameters, dict):
        mode = normalise_control_mode(str(parameters.get("control_mode", "")))
        if mode:
            return mode

    mode = normalise_control_mode(path.as_posix())
    if mode:
        return mode
    raise ValueError("cannot infer control mode")


def normalise_control_mode(text: str) -> str:
    lowered = text.lower()
    if re.search(r"\bposition\b|_position_|position_", lowered):
        return "position"
    if re.search(r"\bvelocity\b|_velocity_|velocity_", lowered):
        return "velocity"
    if re.search(r"\bacceleration\b|_acceleration_|acceleration_|\baccel\b", lowered):
        return "acceleration"
    if re.search(r"\bthrust\b|_thrust_|thrust_", lowered):
        return "thrust"
    return ""


def infer_simulation_control_mode(path: Path) -> str:
    text = path.parent.name.lower()
    if re.search(r"(^|_)velocity(_|$)|velocity_delay", text):
        return "velocity"
    if re.search(r"(^|_)acceleration(_|$)|acceleration_delay|(^|_)accel(_|$)", text):
        return "acceleration"
    if re.search(r"(^|_)thrust(_|$)|thrust_delay", text):
        return "thrust"
    if re.search(r"(^|_)position_delay|(^|_)position_control|control_position", text):
        return "position"
    raise ValueError(f"cannot infer simulation control mode from directory name: {path.parent.name}")


def first_string(data: dict[str, np.ndarray], key: str, default: str = "") -> str:
    if key not in data or len(data[key]) == 0:
        return default
    return str(data[key][0])


def as_float_array(data: dict[str, np.ndarray], key: str) -> np.ndarray | None:
    if key not in data:
        return None
    try:
        return np.asarray(data[key], dtype=np.float64)
    except (TypeError, ValueError):
        return None


def load_simulation_step_dt(path: Path) -> float:
    resolved_run_path = path.parent / "resolved_run.yaml"
    if not resolved_run_path.exists():
        return 1.0 / 60.0

    try:
        text = resolved_run_path.read_text()
    except OSError:
        return 1.0 / 60.0

    match = re.search(r"(?m)^\s*policy_step_dt:\s*([0-9.eE+-]+)\s*$", text)
    if match is None:
        return 1.0 / 60.0

    try:
        dt_s = float(match.group(1))
    except ValueError:
        return 1.0 / 60.0
    if not math.isfinite(dt_s) or dt_s <= 0.0:
        return 1.0 / 60.0
    return dt_s


def decode_text(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode()
    return str(value)


def simulation_field_map(data: dict[str, np.ndarray]) -> dict[str, int]:
    fields = data.get("observation_fields")
    if fields is None:
        return {}
    return {decode_text(field): index for index, field in enumerate(np.asarray(fields).tolist())}


def simulation_observation(data: dict[str, np.ndarray], field: str) -> np.ndarray | None:
    observations = as_float_array(data, "observations")
    if observations is None or observations.ndim != 3:
        return None
    fields = simulation_field_map(data)
    if field not in fields:
        return None
    index = fields[field]
    if index < 0 or index >= observations.shape[2]:
        return None
    return observations[:, :, index]


def simulation_matrix(data: dict[str, np.ndarray], key: str) -> np.ndarray | None:
    values = as_float_array(data, key)
    if values is None or values.ndim != 2:
        return None
    return values


def simulation_command_candidates(data: dict[str, np.ndarray], mode: str) -> tuple[tuple[str, np.ndarray | None], ...]:
    if mode == "position":
        return (
            ("step_drz_cmd", simulation_matrix(data, "step_drz_cmd")),
            ("step_command_z", simulation_matrix(data, "step_command_z")),
        )
    if mode == "velocity":
        return (
            ("step_vrz_cmd", simulation_matrix(data, "step_vrz_cmd")),
            ("step_command_z", simulation_matrix(data, "step_command_z")),
        )
    if mode == "acceleration":
        return (
            ("step_target_height_acc", simulation_matrix(data, "step_target_height_acc")),
            ("step_command_z", simulation_matrix(data, "step_command_z")),
        )
    if mode == "thrust":
        frz_cmd = simulation_matrix(data, "step_frz_cmd")
        hover_force = simulation_matrix(data, "step_hover_force")
        force_offset = None
        if frz_cmd is not None and hover_force is not None and frz_cmd.shape == hover_force.shape:
            force_offset = frz_cmd - hover_force
        return (
            ("step_delta_frz_cmd", simulation_matrix(data, "step_delta_frz_cmd")),
            ("step_frz_cmd_minus_step_hover_force", force_offset),
            ("step_command_z", simulation_matrix(data, "step_command_z")),
        )
    raise ValueError(f"unsupported simulation control mode: {mode}")


def simulation_response_candidates(
    data: dict[str, np.ndarray],
    mode: str,
    response_target: str,
) -> tuple[tuple[str, np.ndarray | None], ...]:
    if response_target in {"auto", "beam"}:
        return (("sim_theta", simulation_observation(data, "theta")),)
    if mode == "position":
        return (("sim_drz", simulation_observation(data, "drz")),)
    if mode == "velocity":
        return (("sim_vrz", simulation_observation(data, "vrz")),)
    if mode in {"acceleration", "thrust"}:
        return (("sim_arz", simulation_observation(data, "arz")),)
    raise ValueError(f"unsupported simulation control mode: {mode}")


def is_beam_system_data(data: dict[str, np.ndarray], path: Path) -> bool:
    if "beam_theta_rel" in data or "beam_theta" in data:
        return True

    metadata = load_metadata(path)
    system_type = str(metadata.get("system_type", "")).lower()
    parameters = metadata.get("parameters", {})
    if isinstance(parameters, dict):
        system_type = f"{system_type} {parameters.get('system_type', '')}".lower()
    return "beam" in system_type


def select_signal(
    candidates: Iterable[tuple[str, np.ndarray | None]],
    min_samples: int,
) -> tuple[str, np.ndarray]:
    for name, values in candidates:
        if values is None:
            continue
        finite = np.isfinite(values)
        if int(np.count_nonzero(finite)) < min_samples:
            continue
        if float(np.nanstd(values[finite])) <= 1e-12:
            continue
        return name, values
    candidate_names = ", ".join(name for name, _ in candidates)
    raise ValueError(f"no usable signal among candidates: {candidate_names}")


def build_signal_set(
    data: dict[str, np.ndarray],
    path: Path,
    *,
    trim_start_s: float,
    trim_end_s: float,
    smooth_response_window_s: float,
    min_samples: int,
    include_unsafe: bool,
    response_target: str,
) -> SignalSet:
    mode = infer_control_mode(data, path)
    time = as_float_array(data, "t_trial")
    if time is None:
        t_ros = as_float_array(data, "t_ros")
        if t_ros is None:
            raise ValueError("missing t_trial/t_ros")
        time = t_ros - float(t_ros[0])

    cmd_up = as_float_array(data, "cmd_up")
    pos_sp_z = as_float_array(data, "pos_sp_z")
    vel_sp_z = as_float_array(data, "vel_sp_z")
    acc_sp_z = as_float_array(data, "acc_sp_z")
    thrust_norm = as_float_array(data, "thrust_norm")
    z = as_float_array(data, "z")
    vz = as_float_array(data, "vz")
    az = as_float_array(data, "az")
    beam_theta_rel = as_float_array(data, "beam_theta_rel")
    beam_theta = as_float_array(data, "beam_theta")

    if mode == "position":
        input_signal, command = select_signal(
            (
                ("cmd_up", cmd_up),
                ("pos_sp_up_abs_from_neg_pos_sp_z", negate(pos_sp_z)),
            ),
            min_samples,
        )
        drone_response_candidates = (("position_up_from_neg_z", negate(z)),)
    elif mode == "velocity":
        input_signal, command = select_signal(
            (
                ("cmd_up", cmd_up),
                ("vel_sp_up_from_neg_vel_sp_z", negate(vel_sp_z)),
            ),
            min_samples,
        )
        drone_response_candidates = (("velocity_up_from_neg_vz", negate(vz)),)
    elif mode == "acceleration":
        input_signal, command = select_signal(
            (
                ("cmd_up", cmd_up),
                ("acc_sp_up_from_neg_acc_sp_z", negate(acc_sp_z)),
            ),
            min_samples,
        )
        drone_response_candidates = (("acceleration_up_from_neg_az", negate(az)),)
    elif mode == "thrust":
        input_signal, command = select_signal(
            (
                ("thrust_norm", thrust_norm),
                ("cmd_up", cmd_up),
            ),
            min_samples,
        )
        drone_response_candidates = (("acceleration_up_from_neg_az", negate(az)),)
    else:
        raise ValueError(f"unsupported control mode: {mode}")

    use_beam_response = response_target == "beam" or (
        response_target == "auto" and is_beam_system_data(data, path)
    )
    if use_beam_response:
        response_signal, response = select_signal(
            (
                ("beam_theta_rel", beam_theta_rel),
                ("beam_theta", beam_theta),
            ),
            min_samples,
        )
    else:
        response_signal, response = select_signal(drone_response_candidates, min_samples)

    mask = np.isfinite(time) & np.isfinite(command) & np.isfinite(response)
    if "phase" in data:
        phase = np.asarray(data["phase"], dtype=str)
        if len(phase) == len(mask):
            mask &= phase == "excitation"
    if not include_unsafe and "safety_state" in data:
        safety_state = np.asarray(data["safety_state"], dtype=str)
        if len(safety_state) == len(mask):
            mask &= safety_state == "ok"

    time, command, response = time[mask], command[mask], response[mask]
    if len(time) < min_samples:
        raise ValueError(f"only {len(time)} usable samples")

    time, command, response = sort_and_deduplicate(time, command, response)
    time, command, response, dt_s = resample_uniform(time, command, response)
    time, command, response = trim_trial(time, command, response, trim_start_s, trim_end_s)

    if smooth_response_window_s > 0.0:
        response = centered_moving_average(response, max(1, int(round(smooth_response_window_s / dt_s))))

    if len(time) < min_samples:
        raise ValueError(f"only {len(time)} samples after resampling/trimming")

    excitation_type = first_string(data, "excitation_type", default="unknown")
    return SignalSet(
        mode=mode,
        time_s=time,
        command=command,
        response=response,
        input_signal=input_signal,
        response_signal=response_signal,
        excitation_type=excitation_type,
        dt_s=dt_s,
    )


def build_array_signal_set(
    *,
    mode: str,
    time: np.ndarray,
    command: np.ndarray,
    response: np.ndarray,
    input_signal: str,
    response_signal: str,
    excitation_type: str,
    trim_start_s: float,
    trim_end_s: float,
    smooth_response_window_s: float,
    min_samples: int,
) -> SignalSet:
    mask = np.isfinite(time) & np.isfinite(command) & np.isfinite(response)
    time, command, response = time[mask], command[mask], response[mask]
    if len(time) < min_samples:
        raise ValueError(f"only {len(time)} usable samples")

    time, command, response = sort_and_deduplicate(time, command, response)
    time, command, response, dt_s = resample_uniform(time, command, response)
    time, command, response = trim_trial(time, command, response, trim_start_s, trim_end_s)

    if smooth_response_window_s > 0.0:
        response = centered_moving_average(response, max(1, int(round(smooth_response_window_s / dt_s))))

    if len(time) < min_samples:
        raise ValueError(f"only {len(time)} samples after resampling/trimming")

    return SignalSet(
        mode=mode,
        time_s=time,
        command=command,
        response=response,
        input_signal=input_signal,
        response_signal=response_signal,
        excitation_type=excitation_type,
        dt_s=dt_s,
    )


def negate(values: np.ndarray | None) -> np.ndarray | None:
    if values is None:
        return None
    return -values


def sort_and_deduplicate(
    time: np.ndarray,
    command: np.ndarray,
    response: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(time)
    time, command, response = time[order], command[order], response[order]
    unique_time, unique_indices = np.unique(time, return_index=True)
    return unique_time, command[unique_indices], response[unique_indices]


def resample_uniform(
    time: np.ndarray,
    command: np.ndarray,
    response: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    diffs = np.diff(time)
    positive_diffs = diffs[diffs > 0.0]
    if len(positive_diffs) == 0:
        raise ValueError("time stamps are not increasing")
    dt_s = float(np.median(positive_diffs))
    if not math.isfinite(dt_s) or dt_s <= 0.0:
        raise ValueError(f"invalid sample period: {dt_s}")

    uniform_time = np.arange(float(time[0]), float(time[-1]) + 0.5 * dt_s, dt_s)
    uniform_command = np.interp(uniform_time, time, command)
    uniform_response = np.interp(uniform_time, time, response)
    uniform_time = uniform_time - float(uniform_time[0])
    return uniform_time, uniform_command, uniform_response, dt_s


def trim_trial(
    time: np.ndarray,
    command: np.ndarray,
    response: np.ndarray,
    trim_start_s: float,
    trim_end_s: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    start_t = float(time[0]) + max(trim_start_s, 0.0)
    end_t = float(time[-1]) - max(trim_end_s, 0.0)
    keep = (time >= start_t) & (time <= end_t)
    if not np.any(keep):
        raise ValueError("trimming removed all samples")
    trimmed_time = time[keep]
    trimmed_time = trimmed_time - float(trimmed_time[0])
    return trimmed_time, command[keep], response[keep]


def centered_moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values.copy()
    before = window // 2
    after = window - 1 - before
    padded = np.pad(values, (before, after), mode="edge")
    kernel = np.ones(window, dtype=np.float64) / float(window)
    return np.convolve(padded, kernel, mode="valid")


def estimate_delay(
    signals: SignalSet,
    *,
    method: str,
    max_delay_s: float,
    lag_step_fraction: float,
    min_samples: int,
) -> tuple[float, dict[str, float | bool | str]]:
    if method == "arx":
        return estimate_delay_arx(
            signals,
            max_delay_s=max_delay_s,
            lag_step_fraction=lag_step_fraction,
            min_samples=min_samples,
        )
    if method == "xcorr":
        return estimate_delay_xcorr(
            signals,
            max_delay_s=max_delay_s,
            lag_step_fraction=lag_step_fraction,
            min_samples=min_samples,
        )
    raise ValueError(f"unsupported delay estimation method: {method}")


def estimate_delay_arx(
    signals: SignalSet,
    *,
    max_delay_s: float,
    lag_step_fraction: float,
    min_samples: int,
) -> tuple[float, dict[str, float | bool | str]]:
    ar_order = MODE_DEFAULT_AR_ORDER[signals.mode]
    dt_s = signals.dt_s
    lag_step_s = max(dt_s * lag_step_fraction, dt_s / 20.0)
    max_delay_s = min(max_delay_s, max(0.0, float(signals.time_s[-1]) - dt_s * (ar_order + min_samples)))
    if max_delay_s <= 0.0:
        raise ValueError("trial is too short for the requested delay search")

    candidate_delays = np.arange(0.0, max_delay_s + 0.5 * lag_step_s, lag_step_s)
    mse_values = np.full_like(candidate_delays, fill_value=np.nan, dtype=np.float64)
    r2_values = np.full_like(candidate_delays, fill_value=np.nan, dtype=np.float64)

    for index, delay_s in enumerate(candidate_delays):
        fit = fit_arx_at_delay(
            signals.time_s,
            signals.command,
            signals.response,
            delay_s,
            ar_order,
            min_samples,
        )
        if fit is None:
            continue
        mse_values[index] = fit["mse"]
        r2_values[index] = fit["r2"]

    if not np.any(np.isfinite(mse_values)):
        raise ValueError("no valid ARX fit in delay search range")

    best_index = int(np.nanargmin(mse_values))
    delay_s = refine_delay_quadratic(candidate_delays, mse_values, best_index)
    final_fit = fit_arx_at_delay(
        signals.time_s,
        signals.command,
        signals.response,
        delay_s,
        ar_order,
        min_samples,
    )
    if final_fit is None:
        delay_s = float(candidate_delays[best_index])
        final_fit = fit_arx_at_delay(
            signals.time_s,
            signals.command,
            signals.response,
            delay_s,
            ar_order,
            min_samples,
        )
    if final_fit is None:
        raise ValueError("best delay produced an invalid ARX fit")

    ar_only_fit = fit_ar_only(signals.response, ar_order, min_samples)
    ar_only_mse = float(ar_only_fit["mse"])
    input_mse_improvement = 0.0
    if math.isfinite(ar_only_mse) and ar_only_mse > 0.0:
        input_mse_improvement = max(0.0, (ar_only_mse - float(final_fit["mse"])) / ar_only_mse)

    finite_mse = mse_values[np.isfinite(mse_values)]
    curve_sharpness = 0.0
    if len(finite_mse) > 0 and float(np.nanmedian(finite_mse)) > 0.0:
        curve_sharpness = max(
            0.0,
            (float(np.nanmedian(finite_mse)) - float(final_fit["mse"]))
            / float(np.nanmedian(finite_mse)),
        )

    boundary_solution = best_index == 0 or best_index == len(candidate_delays) - 1
    metrics: dict[str, float | bool | str] = {
        "method": "arx",
        "mse": float(final_fit["mse"]),
        "r2": float(final_fit["r2"]),
        "ar_only_r2": float(ar_only_fit["r2"]),
        "input_mse_improvement": float(input_mse_improvement),
        "curve_sharpness": float(curve_sharpness),
        "correlation_peak": math.nan,
        "boundary_solution": bool(boundary_solution),
        "ar_order": float(ar_order),
        "max_delay_s": float(max_delay_s),
    }
    return delay_s, metrics


def estimate_delay_xcorr(
    signals: SignalSet,
    *,
    max_delay_s: float,
    lag_step_fraction: float,
    min_samples: int,
) -> tuple[float, dict[str, float | bool | str]]:
    dt_s = signals.dt_s
    lag_step_s = max(dt_s * lag_step_fraction, dt_s / 20.0)
    max_delay_s = min(max_delay_s, max(0.0, float(signals.time_s[-1]) - dt_s * min_samples))
    if max_delay_s <= 0.0:
        raise ValueError("trial is too short for the requested delay search")

    candidate_delays = np.arange(0.0, max_delay_s + 0.5 * lag_step_s, lag_step_s)
    correlations = np.full_like(candidate_delays, fill_value=np.nan, dtype=np.float64)

    for index, delay_s in enumerate(candidate_delays):
        correlations[index] = correlation_at_delay(
            signals.time_s,
            signals.command,
            signals.response,
            delay_s,
            min_samples,
        )

    objectives = np.abs(correlations)
    if not np.any(np.isfinite(objectives)):
        raise ValueError("no valid cross-correlation in delay search range")

    best_index = int(np.nanargmax(objectives))
    delay_s = refine_delay_quadratic(candidate_delays, -objectives, best_index)
    correlation_peak = correlation_at_delay(
        signals.time_s,
        signals.command,
        signals.response,
        delay_s,
        min_samples,
    )
    if not math.isfinite(float(correlation_peak)):
        delay_s = float(candidate_delays[best_index])
        correlation_peak = float(correlations[best_index])

    finite_objectives = objectives[np.isfinite(objectives)]
    curve_sharpness = 0.0
    peak_abs = abs(float(correlation_peak))
    if len(finite_objectives) > 0 and peak_abs > 0.0:
        curve_sharpness = max(
            0.0,
            (peak_abs - float(np.nanmedian(finite_objectives))) / peak_abs,
        )

    boundary_solution = best_index == 0 or best_index == len(candidate_delays) - 1
    metrics: dict[str, float | bool | str] = {
        "method": "xcorr",
        "mse": math.nan,
        "r2": math.nan,
        "ar_only_r2": math.nan,
        "input_mse_improvement": math.nan,
        "curve_sharpness": float(curve_sharpness),
        "correlation_peak": float(correlation_peak),
        "boundary_solution": bool(boundary_solution),
        "ar_order": 0.0,
        "max_delay_s": float(max_delay_s),
    }
    return delay_s, metrics


def fit_arx_at_delay(
    time_s: np.ndarray,
    command: np.ndarray,
    response: np.ndarray,
    delay_s: float,
    ar_order: int,
    min_samples: int,
) -> dict[str, float] | None:
    delayed_command = np.interp(
        time_s - delay_s,
        time_s,
        command,
        left=np.nan,
        right=np.nan,
    )
    matrix, target = build_arx_matrix(response, delayed_command, ar_order, include_command=True)
    if len(target) < min_samples:
        return None
    return fit_least_squares(matrix, target)


def correlation_at_delay(
    time_s: np.ndarray,
    command: np.ndarray,
    response: np.ndarray,
    delay_s: float,
    min_samples: int,
) -> float:
    delayed_command = np.interp(
        time_s - delay_s,
        time_s,
        command,
        left=np.nan,
        right=np.nan,
    )
    mask = np.isfinite(delayed_command) & np.isfinite(response)
    if int(np.count_nonzero(mask)) < min_samples:
        return math.nan

    command_window = delayed_command[mask]
    response_window = response[mask]
    command_window = command_window - float(np.mean(command_window))
    response_window = response_window - float(np.mean(response_window))

    command_norm = float(np.sqrt(np.sum(command_window * command_window)))
    response_norm = float(np.sqrt(np.sum(response_window * response_window)))
    if command_norm <= 0.0 or response_norm <= 0.0:
        return math.nan

    return float(np.sum(command_window * response_window) / (command_norm * response_norm))


def fit_ar_only(response: np.ndarray, ar_order: int, min_samples: int) -> dict[str, float]:
    matrix, target = build_arx_matrix(response, None, ar_order, include_command=False)
    if len(target) < min_samples:
        return {"mse": math.nan, "r2": math.nan}
    return fit_least_squares(matrix, target)


def build_arx_matrix(
    response: np.ndarray,
    delayed_command: np.ndarray | None,
    ar_order: int,
    *,
    include_command: bool,
) -> tuple[np.ndarray, np.ndarray]:
    rows: list[list[float]] = []
    targets: list[float] = []
    for index in range(ar_order, len(response)):
        row = [1.0]
        row.extend(float(response[index - lag]) for lag in range(1, ar_order + 1))
        if include_command:
            if delayed_command is None or not math.isfinite(float(delayed_command[index])):
                continue
            row.append(float(delayed_command[index]))
        if not all(math.isfinite(value) for value in row):
            continue
        target = float(response[index])
        if not math.isfinite(target):
            continue
        rows.append(row)
        targets.append(target)
    return np.asarray(rows, dtype=np.float64), np.asarray(targets, dtype=np.float64)


def fit_least_squares(matrix: np.ndarray, target: np.ndarray) -> dict[str, float]:
    coefficients, *_ = np.linalg.lstsq(matrix, target, rcond=None)
    prediction = matrix @ coefficients
    residual = target - prediction
    mse = float(np.mean(residual * residual))
    baseline_mse = float(np.mean((target - float(np.mean(target))) ** 2))
    r2 = math.nan
    if baseline_mse > 0.0:
        r2 = 1.0 - mse / baseline_mse
    return {"mse": mse, "r2": float(r2)}


def refine_delay_quadratic(
    candidate_delays: np.ndarray,
    mse_values: np.ndarray,
    best_index: int,
) -> float:
    if best_index <= 0 or best_index >= len(candidate_delays) - 1:
        return float(candidate_delays[best_index])

    left = max(0, best_index - 2)
    right = min(len(candidate_delays), best_index + 3)
    x = candidate_delays[left:right]
    y = mse_values[left:right]
    valid = np.isfinite(y)
    if int(np.count_nonzero(valid)) < 3:
        return float(candidate_delays[best_index])

    coefficients = np.polyfit(x[valid], y[valid], 2)
    a, b, _ = [float(value) for value in coefficients]
    if a <= 0.0:
        return float(candidate_delays[best_index])

    vertex = -b / (2.0 * a)
    lower = float(candidate_delays[max(0, best_index - 1)])
    upper = float(candidate_delays[min(len(candidate_delays) - 1, best_index + 1)])
    if lower <= vertex <= upper:
        return float(vertex)
    return float(candidate_delays[best_index])


def classify_quality(
    *,
    r2: float,
    input_mse_improvement: float,
    curve_sharpness: float,
    boundary_solution: bool,
) -> str:
    flags: list[str] = []
    if boundary_solution:
        flags.append("boundary")
    if not math.isfinite(r2) or r2 < 0.5:
        flags.append("low_r2")
    if input_mse_improvement < 0.05:
        flags.append("weak_input_effect")
    if curve_sharpness < 0.02:
        flags.append("flat_objective")
    return "ok" if not flags else ";".join(flags)


def classify_xcorr_quality(
    *,
    correlation_peak: float,
    curve_sharpness: float,
    boundary_solution: bool,
) -> str:
    flags: list[str] = []
    if boundary_solution:
        flags.append("boundary")
    if not math.isfinite(correlation_peak) or abs(correlation_peak) < 0.15:
        flags.append("weak_correlation")
    if curve_sharpness < 0.02:
        flags.append("flat_objective")
    return "ok" if not flags else ";".join(flags)


def estimate_signal_set(
    *,
    data_file: str,
    trial_name: str,
    signals: SignalSet,
    args: argparse.Namespace,
) -> DelayEstimate:
    max_delay_s = (
        float(args.max_delay_s)
        if args.max_delay_s is not None
        else MODE_DEFAULT_MAX_DELAY_S[signals.mode]
    )
    delay_s, metrics = estimate_delay(
        signals,
        method=str(args.method),
        max_delay_s=max_delay_s,
        lag_step_fraction=float(args.lag_step_fraction),
        min_samples=int(args.min_samples),
    )
    if str(metrics["method"]) == "xcorr":
        quality = classify_xcorr_quality(
            correlation_peak=float(metrics["correlation_peak"]),
            curve_sharpness=float(metrics["curve_sharpness"]),
            boundary_solution=bool(metrics["boundary_solution"]),
        )
    else:
        quality = classify_quality(
            r2=float(metrics["r2"]),
            input_mse_improvement=float(metrics["input_mse_improvement"]),
            curve_sharpness=float(metrics["curve_sharpness"]),
            boundary_solution=bool(metrics["boundary_solution"]),
        )

    return DelayEstimate(
        data_file=data_file,
        trial_name=trial_name,
        control_mode=signals.mode,
        excitation_type=signals.excitation_type,
        delay_s=float(delay_s),
        delay_ms=float(delay_s * 1000.0),
        delay_samples=float(delay_s / signals.dt_s),
        dt_s=float(signals.dt_s),
        sample_count=int(len(signals.time_s)),
        duration_s=float(signals.time_s[-1] - signals.time_s[0]),
        input_signal=signals.input_signal,
        response_signal=signals.response_signal,
        ar_order=int(metrics["ar_order"]),
        max_delay_s=float(metrics["max_delay_s"]),
        mse=float(metrics["mse"]),
        r2=float(metrics["r2"]),
        ar_only_r2=float(metrics["ar_only_r2"]),
        input_mse_improvement=float(metrics["input_mse_improvement"]),
        curve_sharpness=float(metrics["curve_sharpness"]),
        boundary_solution=bool(metrics["boundary_solution"]),
        quality=quality,
        method=str(metrics["method"]),
        correlation_peak=float(metrics["correlation_peak"]),
    )


def estimate_file(
    path: Path,
    args: argparse.Namespace,
) -> DelayEstimate:
    data = load_data_file(path)
    signals = build_signal_set(
        data,
        path,
        trim_start_s=args.trim_start_s,
        trim_end_s=args.trim_end_s,
        smooth_response_window_s=args.smooth_response_window_s,
        min_samples=args.min_samples,
        include_unsafe=args.include_unsafe,
        response_target=args.response_target,
    )
    return estimate_signal_set(
        data_file=str(path),
        trial_name=path.parent.name if path.name.startswith("samples.") else path.stem,
        signals=signals,
        args=args,
    )


def simulation_done_matrix(data: dict[str, np.ndarray], steps: int, num_envs: int) -> np.ndarray | None:
    done = None
    for key in ("terminated", "truncated"):
        values = data.get(key)
        if values is None:
            continue
        matrix = np.asarray(values, dtype=bool)
        if matrix.shape != (steps, num_envs):
            continue
        done = matrix if done is None else (done | matrix)
    return done


def slice_simulation_candidate(
    candidate: tuple[str, np.ndarray | None],
    *,
    env_index: int,
    end_index: int,
) -> tuple[str, np.ndarray | None]:
    name, values = candidate
    if values is None or values.ndim != 2 or env_index >= values.shape[1]:
        return name, None
    return name, values[: min(end_index, values.shape[0]), env_index]


def estimate_simulation_file(
    path: Path,
    args: argparse.Namespace,
) -> tuple[list[DelayEstimate], list[dict[str, str]]]:
    data = load_data_file(path)
    observations = as_float_array(data, "observations")
    if observations is None or observations.ndim != 3:
        raise ValueError("simulation rollout is missing 3-D observations")

    steps, num_envs, _ = observations.shape
    mode = infer_simulation_control_mode(path)
    dt_s = load_simulation_step_dt(path)
    command_candidates = simulation_command_candidates(data, mode)
    response_candidates = simulation_response_candidates(data, mode, str(args.response_target))
    done = simulation_done_matrix(data, steps, num_envs)

    estimates: list[DelayEstimate] = []
    skipped: list[dict[str, str]] = []
    for env_index in range(num_envs):
        try:
            end_index = steps
            if done is not None:
                done_indices = np.nonzero(done[:, env_index])[0]
                if len(done_indices) > 0:
                    end_index = int(done_indices[0]) + 1
            time = np.arange(end_index, dtype=np.float64) * dt_s

            env_command_candidates = tuple(
                slice_simulation_candidate(candidate, env_index=env_index, end_index=end_index)
                for candidate in command_candidates
            )
            env_response_candidates = tuple(
                slice_simulation_candidate(candidate, env_index=env_index, end_index=end_index)
                for candidate in response_candidates
            )
            input_signal, command = select_signal(env_command_candidates, int(args.min_samples))
            response_signal, response = select_signal(env_response_candidates, int(args.min_samples))
            usable_length = min(len(time), len(command), len(response))

            signals = build_array_signal_set(
                mode=mode,
                time=time[:usable_length],
                command=command[:usable_length],
                response=response[:usable_length],
                input_signal=input_signal,
                response_signal=response_signal,
                excitation_type="simulation",
                trim_start_s=float(args.trim_start_s),
                trim_end_s=float(args.trim_end_s),
                smooth_response_window_s=float(args.smooth_response_window_s),
                min_samples=int(args.min_samples),
            )
            estimates.append(
                estimate_signal_set(
                    data_file=str(path),
                    trial_name=f"{path.parent.name}_env{env_index:04d}",
                    signals=signals,
                    args=args,
                )
            )
        except Exception as exc:  # noqa: BLE001 - keep other parallel envs usable.
            skipped.append({"file": f"{path}#env{env_index:04d}", "reason": str(exc)})
    return estimates, skipped


def summarise(estimates: list[DelayEstimate]) -> list[ModeSummary]:
    summaries: list[ModeSummary] = []
    for mode in CONTROL_MODES:
        mode_estimates = [estimate for estimate in estimates if estimate.control_mode == mode]
        if not mode_estimates:
            continue
        delays_s = np.asarray([estimate.delay_s for estimate in mode_estimates], dtype=np.float64)
        delays_samples = np.asarray([estimate.delay_samples for estimate in mode_estimates], dtype=np.float64)
        std_s = float(np.std(delays_s, ddof=1)) if len(delays_s) > 1 else 0.0
        std_samples = float(np.std(delays_samples, ddof=1)) if len(delays_samples) > 1 else 0.0
        summaries.append(
            ModeSummary(
                control_mode=mode,
                count=int(len(delays_s)),
                mean_delay_s=float(np.mean(delays_s)),
                std_delay_s=std_s,
                mean_delay_ms=float(np.mean(delays_s) * 1000.0),
                std_delay_ms=float(std_s * 1000.0),
                mean_delay_samples=float(np.mean(delays_samples)),
                std_delay_samples=std_samples,
                median_delay_ms=float(np.median(delays_s) * 1000.0),
                min_delay_ms=float(np.min(delays_s) * 1000.0),
                max_delay_ms=float(np.max(delays_s) * 1000.0),
            )
        )
    return summaries


def write_csv(path: Path, rows: list[object]) -> None:
    if not rows:
        return
    dictionaries = [asdict(row) for row in rows]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(dictionaries[0].keys()))
        writer.writeheader()
        writer.writerows(dictionaries)


def write_json(
    path: Path,
    *,
    args: argparse.Namespace,
    estimates: list[DelayEstimate],
    summaries: list[ModeSummary],
    skipped: list[dict[str, str]],
) -> None:
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "data_path": str(args.data_path),
        "settings": {
            "data_source": args.data_source,
            "max_delay_s": args.max_delay_s,
            "method": args.method,
            "lag_step_fraction": args.lag_step_fraction,
            "trim_start_s": args.trim_start_s,
            "trim_end_s": args.trim_end_s,
            "smooth_response_window_s": args.smooth_response_window_s,
            "min_samples": args.min_samples,
            "include_unsafe": args.include_unsafe,
            "response_target": args.response_target,
            "method_description": (
                "fractional-delay ARX residual scan with quadratic refinement"
                if args.method == "arx"
                else "normalized cross-correlation scan with quadratic peak refinement"
            ),
        },
        "summaries": [asdict(summary) for summary in summaries],
        "file_estimates": [asdict(estimate) for estimate in estimates],
        "skipped_files": skipped,
    }
    with path.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def write_summary_plot(path: Path, summaries: list[ModeSummary]) -> None:
    cache_dir = path.parent / ".matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Unable to import Axes3D.*")
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

    modes = [summary.control_mode for summary in summaries]
    labels = [MODE_LABELS[mode] for mode in modes]
    means = [summary.mean_delay_ms for summary in summaries]
    stds = [summary.std_delay_ms for summary in summaries]

    fig, ax = plt.subplots(figsize=(8.0, 4.8), dpi=160)
    colors = ["#4477aa", "#66c2a5", "#e6ab02", "#cc6677"]
    bar_colors = [colors[CONTROL_MODES.index(mode)] for mode in modes]
    ax.bar(labels, means, yerr=stds, capsize=6, color=bar_colors, edgecolor="#222222", linewidth=0.8)
    ax.set_ylabel("Estimated delay (ms)")
    ax.set_xlabel("Control mode")
    ax.set_title("PX4 vertical action-delay identification")
    ax.grid(axis="y", alpha=0.25, linewidth=0.8)
    ax.set_axisbelow(True)

    upper = max((mean + std for mean, std in zip(means, stds)), default=0.0)
    ax.set_ylim(0.0, max(upper * 1.25, 10.0))
    for index, summary in enumerate(summaries):
        ax.text(
            index,
            summary.mean_delay_ms + summary.std_delay_ms + max(upper, 1.0) * 0.04,
            f"{summary.mean_delay_ms:.1f} +/- {summary.std_delay_ms:.1f} ms\nn={summary.count}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def default_output_dir(data_path: Path) -> Path:
    if data_path.is_dir():
        return data_path / "delay_identification"
    return data_path.parent / "delay_identification"


def print_summary(estimates: list[DelayEstimate], summaries: list[ModeSummary], skipped: list[dict[str, str]]) -> None:
    print(f"Estimated {len(estimates)} file(s); skipped {len(skipped)} file(s).")
    for summary in summaries:
        print(
            f"{summary.control_mode:>12s}: "
            f"{summary.mean_delay_ms:7.2f} +/- {summary.std_delay_ms:6.2f} ms "
            f"(n={summary.count})"
        )
    if skipped:
        print("Skipped files:")
        for item in skipped:
            print(f"  {item['file']}: {item['reason']}")


def main() -> None:
    args = parse_args()
    data_path = args.data_path.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else default_output_dir(data_path).resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    estimates: list[DelayEstimate] = []
    skipped: list[dict[str, str]] = []
    data_files = (
        discover_simulation_data_files(data_path)
        if args.data_source == "simulation"
        else discover_data_files(data_path)
    )
    for data_file in data_files:
        try:
            if args.data_source == "simulation":
                file_estimates, file_skipped = estimate_simulation_file(data_file, args)
                estimates.extend(file_estimates)
                skipped.extend(file_skipped)
            else:
                estimates.append(estimate_file(data_file, args))
        except Exception as exc:  # noqa: BLE001 - report bad data files and continue.
            skipped.append({"file": str(data_file), "reason": str(exc)})

    if not estimates:
        raise SystemExit("No valid action-delay data files were identified.")

    summaries = summarise(estimates)
    write_csv(output_dir / "action_delay_file_estimates.csv", estimates)
    write_csv(output_dir / "action_delay_summary.csv", summaries)
    write_json(
        output_dir / "action_delay_summary.json",
        args=args,
        estimates=estimates,
        summaries=summaries,
        skipped=skipped,
    )

    if not args.no_plot:
        try:
            write_summary_plot(output_dir / "action_delay_summary.png", summaries)
        except Exception as exc:  # noqa: BLE001 - CSV/JSON outputs are still useful.
            skipped.append({"file": str(output_dir / "action_delay_summary.png"), "reason": str(exc)})
            write_json(
                output_dir / "action_delay_summary.json",
                args=args,
                estimates=estimates,
                summaries=summaries,
                skipped=skipped,
            )

    print_summary(estimates, summaries, skipped)
    print(f"Saved outputs to: {output_dir}")


if __name__ == "__main__":
    main()
