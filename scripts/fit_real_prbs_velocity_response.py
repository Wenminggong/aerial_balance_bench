#!/usr/bin/env python3
"""Identify vertical-velocity response models from real-world excitation trials.

Trial directories may occur at any depth below ``--data-dir``.  Each trial
directory must contain ``samples.npz`` (preferred) or ``samples.csv`` and may
contain ``metadata.json``.  By default, ``vel_sp_z`` is treated as the input
and ``vz`` as the measured response; both use PX4's NED sign convention.  All
models use the causal alignment ``u[k] -> y[k+1]``.  Use
``--excitation-type all`` to pool PRBS, sine, and any other excitation types.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.optimize import differential_evolution, minimize_scalar
from scipy.signal import lfilter

MPLCONFIGDIR = Path(os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib"))
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
warnings.filterwarnings("ignore", message="Unable to import Axes3D.*", category=UserWarning)
import matplotlib.pyplot as plt


MODEL_NAMES = ("unit_delay", "gain_bias_delay", "first_order", "fopdt")
MODEL_LABELS = {
    "unit_delay": "UGD",
    "gain_bias_delay": "GBD",
    "first_order": "FO",
    "fopdt": "FOPDT",
}


@dataclass
class Trial:
    """One uniformly sampled command/response trial."""

    trial_id: str
    trial_index: int
    directory: Path
    source_file: Path
    time_s: np.ndarray
    command_mps: np.ndarray
    response_mps: np.ndarray
    dt_s: float
    phase: str
    control_mode: str
    excitation_type: str
    metadata: dict[str, Any]


@dataclass
class ModelFit:
    """Parameters and simulation-error metrics for one candidate model."""

    model: str
    delay_s: float
    tau_s: float | None
    gain: float
    bias_mps: float
    rmse_mps: float
    mae_mps: float
    r2: float
    fit_percent: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit four velocity-response models to real excitation trials and perform "
            "leave-one-trial-out validation."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("logs/velocity_response_identification"),
        help=(
            "Directory recursively searched for trial directories containing "
            "samples.npz or samples.csv."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output directory (default: <data-dir>/prbs_velocity_response_fit for "
            "PRBS-only fits, otherwise <data-dir>/velocity_response_fit)."
        ),
    )
    parser.add_argument("--time-field", default="t_trial", help="Sample-time field.")
    parser.add_argument(
        "--input-field", default="vel_sp_z", help="Command/input field in samples files."
    )
    parser.add_argument(
        "--response-field", default="vz", help="Measured-response field in samples files."
    )
    parser.add_argument(
        "--phase", default="excitation", help="Only use rows with this phase; empty uses all rows."
    )
    parser.add_argument(
        "--control-mode",
        default="velocity",
        help="Only use rows with this control_mode; empty uses all rows.",
    )
    parser.add_argument(
        "--excitation-type",
        default="prbs",
        help=(
            "Excitation type(s) to use. Accepts one value, comma-separated values, "
            "or 'all'/'*'/empty to pool every excitation type."
        ),
    )
    parser.add_argument(
        "--max-delay",
        type=float,
        default=0.5,
        help="Maximum additional pure delay after one-sample alignment, in seconds.",
    )
    parser.add_argument(
        "--max-tau", type=float, default=1.5, help="Maximum fitted time constant in seconds."
    )
    parser.add_argument(
        "--min-samples", type=int, default=30, help="Minimum usable samples required per trial."
    )
    parser.add_argument("--seed", type=int, default=7, help="Differential-evolution seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_delay <= 0.0 or args.max_tau <= 0.0:
        raise ValueError("--max-delay and --max-tau must be positive.")
    if args.min_samples < 3:
        raise ValueError("--min-samples must be at least 3.")

    data_dir = args.data_dir.expanduser().resolve()
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")
    default_output_name = (
        "prbs_velocity_response_fit"
        if parse_filter_values(args.excitation_type) == ("prbs",)
        else "velocity_response_fit"
    )
    output_dir = (
        args.output_dir or data_dir / default_output_name
    ).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    trials = load_trials(
        data_dir,
        time_field=args.time_field,
        input_field=args.input_field,
        response_field=args.response_field,
        phase=args.phase,
        control_mode=args.control_mode,
        excitation_type=args.excitation_type,
        min_samples=args.min_samples,
    )
    if len(trials) < 2:
        raise RuntimeError(
            f"Leave-one-trial-out validation requires at least two usable trials; "
            f"found {len(trials)} below {data_dir}."
        )

    fit_kwargs = {
        "max_delay_s": args.max_delay,
        "max_tau_s": args.max_tau,
        "seed": args.seed,
    }

    individual_fits: dict[tuple[str, str], ModelFit] = {}
    individual_rows: list[dict[str, Any]] = []
    for trial in trials:
        for model in MODEL_NAMES:
            fit = fit_model([trial], model=model, **fit_kwargs)
            individual_fits[(trial.trial_id, model)] = fit
            individual_rows.append(
                {
                    **trial_columns(trial),
                    **fit_columns(fit),
                }
            )

    overall_fits: dict[str, ModelFit] = {}
    overall_rows: list[dict[str, Any]] = []
    for model in MODEL_NAMES:
        fit = fit_model(trials, model=model, **fit_kwargs)
        overall_fits[model] = fit
        overall_rows.append(
            {
                "trial_count": len(trials),
                "sample_count": sum(trial.response_mps.size for trial in trials),
                "scored_sample_count": sum(
                    trial.response_mps.size - 1 for trial in trials
                ),
                **fit_columns(fit),
            }
        )

    loo_fits, loo_rows = leave_one_trial_out(trials, **fit_kwargs)
    comparison_rows = summarize_comparison(individual_rows, loo_rows)

    write_csv(output_dir / "individual_trial_fits.csv", individual_rows)
    write_csv(output_dir / "overall_fits.csv", overall_rows)
    write_csv(output_dir / "leave_one_trial_out.csv", loo_rows)
    write_csv(output_dir / "model_comparison_summary.csv", comparison_rows)
    write_csv(
        output_dir / "fopdt_trial_parameters.csv",
        [row for row in individual_rows if row["model"] == "fopdt"],
    )
    save_predictions(
        output_dir / "predictions.npz",
        trials,
        individual_fits,
        overall_fits,
        loo_fits,
    )
    plot_model_comparison(loo_rows, output_dir / "model_fit_and_leave_one_out.png")
    plot_fopdt_parameters(
        trials,
        individual_fits,
        output_dir / "fopdt_parameters_across_trials.png",
    )

    summary = build_summary(
        data_dir=data_dir,
        args=args,
        trials=trials,
        overall_fits=overall_fits,
        comparison_rows=comparison_rows,
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    excitation_counts = count_trial_values(trials, "excitation_type")
    excitation_description = ", ".join(
        f"{name or 'unknown'}={count}" for name, count in excitation_counts.items()
    )
    print(
        f"[INFO] Loaded {len(trials)} velocity-response trials "
        f"({sum(t.response_mps.size for t in trials)} samples; {excitation_description})."
    )
    for model in MODEL_NAMES:
        fit = overall_fits[model]
        print(
            f"[INFO] {model:>15}: delay={1.0e3 * fit.delay_s:7.2f} ms, "
            f"tau={format_optional_ms(fit.tau_s):>9}, gain={fit.gain:7.4f}, "
            f"bias={fit.bias_mps:+.5f} m/s, RMSE={fit.rmse_mps:.5f} m/s, "
            f"R2={fit.r2:.4f}"
        )
    print(f"[INFO] Results written to {output_dir}")


def load_trials(
    data_dir: Path,
    *,
    time_field: str,
    input_field: str,
    response_field: str,
    phase: str,
    control_mode: str,
    excitation_type: str,
    min_samples: int,
) -> list[Trial]:
    """Recursively load, filter, and validate all trial directories."""
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    trials: list[Trial] = []
    candidates = discover_trial_directories(data_dir)
    for fallback_index, directory in enumerate(candidates, start=1):
        relative_directory = directory.relative_to(data_dir)
        trial_id = (
            directory.name
            if relative_directory == Path(".")
            else relative_directory.as_posix()
        )
        npz_path = directory / "samples.npz"
        csv_path = directory / "samples.csv"
        if npz_path.is_file():
            sample_data = load_npz(npz_path)
            source_file = npz_path
        elif csv_path.is_file():
            sample_data = load_csv(csv_path)
            source_file = csv_path
        else:
            continue

        metadata_path = directory / "metadata.json"
        metadata = (
            json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata_path.is_file()
            else {}
        )
        required = (time_field, input_field, response_field)
        missing = [field for field in required if field not in sample_data]
        if missing:
            warnings.warn(f"Skipping {trial_id}: missing fields {missing}.")
            continue

        time_s = np.asarray(sample_data[time_field], dtype=np.float64).reshape(-1)
        command = np.asarray(sample_data[input_field], dtype=np.float64).reshape(-1)
        response = np.asarray(sample_data[response_field], dtype=np.float64).reshape(-1)
        if not (time_s.size == command.size == response.size):
            warnings.warn(f"Skipping {trial_id}: required fields have different lengths.")
            continue

        mask = np.isfinite(time_s) & np.isfinite(command) & np.isfinite(response)
        mask &= match_optional_field(sample_data, "phase", phase, time_s.size)
        mask &= match_optional_field(sample_data, "control_mode", control_mode, time_s.size)
        mask &= match_optional_field(
            sample_data, "excitation_type", excitation_type, time_s.size
        )
        if not np.any(mask):
            continue

        selected_phase = summarize_trial_field(sample_data, "phase", mask, metadata)
        selected_control_mode = summarize_trial_field(
            sample_data, "control_mode", mask, metadata
        )
        selected_excitation_type = summarize_trial_field(
            sample_data, "excitation_type", mask, metadata
        )
        time_s, command, response = time_s[mask], command[mask], response[mask]
        if time_s.size < min_samples:
            warnings.warn(
                f"Skipping {trial_id}: only {time_s.size} usable samples "
                f"(minimum {min_samples})."
            )
            continue

        order = np.argsort(time_s, kind="stable")
        time_s, command, response = time_s[order], command[order], response[order]
        unique = np.concatenate(([True], np.diff(time_s) > 0.0))
        time_s, command, response = time_s[unique], command[unique], response[unique]
        if time_s.size < min_samples:
            warnings.warn(f"Skipping {trial_id}: too few unique sample times.")
            continue
        time_s = time_s - time_s[0]
        sample_intervals = np.diff(time_s)
        dt_s = float(np.median(sample_intervals))
        if not np.isfinite(dt_s) or dt_s <= 0.0:
            warnings.warn(f"Skipping {trial_id}: invalid sample interval.")
            continue
        relative_jitter = float(np.max(np.abs(sample_intervals - dt_s)) / dt_s)
        if relative_jitter > 0.05:
            warnings.warn(
                f"{trial_id} has {100.0 * relative_jitter:.1f}% time-grid jitter; "
                "the first-order simulation uses the median sample interval."
            )

        raw_trial_index = metadata.get("trial_index", fallback_index)
        try:
            trial_index = int(raw_trial_index)
        except (TypeError, ValueError):
            trial_index = fallback_index
        trials.append(
            Trial(
                trial_id=trial_id,
                trial_index=trial_index,
                directory=directory.resolve(),
                source_file=source_file.resolve(),
                time_s=time_s,
                command_mps=command,
                response_mps=response,
                dt_s=dt_s,
                phase=selected_phase,
                control_mode=selected_control_mode,
                excitation_type=selected_excitation_type,
                metadata=metadata,
            )
        )

    trials.sort(
        key=lambda trial: (
            Path(trial.trial_id).parent.as_posix(),
            trial.trial_index,
            trial.trial_id,
        )
    )
    return trials


def discover_trial_directories(data_dir: Path) -> list[Path]:
    """Find trial directories at any depth, preferring NPZ when both formats exist."""
    candidates: set[Path] = set()
    for filename in ("samples.npz", "samples.csv"):
        candidates.update(
            sample_file.parent
            for sample_file in data_dir.rglob(filename)
            if sample_file.is_file()
        )
    return sorted(candidates, key=lambda path: path.relative_to(data_dir).as_posix())


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def load_csv(path: Path) -> dict[str, np.ndarray]:
    table = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    if table.dtype.names is None:
        raise ValueError(f"CSV file does not have a named header: {path}")
    table = np.atleast_1d(table)
    return {name: np.asarray(table[name]) for name in table.dtype.names}


def match_optional_field(
    sample_data: dict[str, np.ndarray], field: str, expected: str, size: int
) -> np.ndarray:
    if field not in sample_data:
        return np.ones(size, dtype=bool)
    values = np.asarray(sample_data[field]).astype(str).reshape(-1)
    if values.size != size:
        raise ValueError(f"Optional field {field!r} has {values.size} rows, expected {size}.")
    expected_values = parse_filter_values(expected)
    if not expected_values:
        return np.ones(size, dtype=bool)
    normalized_values = np.char.lower(np.char.strip(values))
    return np.isin(normalized_values, expected_values)


def parse_filter_values(value: str) -> tuple[str, ...]:
    """Parse a scalar/comma-separated filter; an all-token disables filtering."""
    tokens = tuple(part.strip().lower() for part in value.split(",") if part.strip())
    if not tokens or any(token in {"all", "*", "any"} for token in tokens):
        return ()
    return tuple(dict.fromkeys(tokens))


def summarize_trial_field(
    sample_data: dict[str, np.ndarray],
    field: str,
    mask: np.ndarray,
    metadata: dict[str, Any],
) -> str:
    """Return the selected field values, with metadata as a legacy fallback."""
    if field in sample_data:
        values = np.asarray(sample_data[field]).astype(str).reshape(-1)
        if values.size != mask.size:
            raise ValueError(
                f"Optional field {field!r} has {values.size} rows, expected {mask.size}."
            )
        unique_values = sorted(
            {value.strip() for value in values[mask] if value.strip()}
        )
        if unique_values:
            return ",".join(unique_values)

    parameters = metadata.get("parameters", {})
    fallback = metadata.get(
        field, parameters.get(field) if isinstance(parameters, dict) else None
    )
    return "" if fallback is None else str(fallback)


def response_aligned_command(trial: Trial, delay_s: float) -> np.ndarray:
    """Align ``u[k]`` with ``y[k+1]`` and apply an additional fractional delay."""
    command_times = np.empty_like(trial.time_s)
    command_times[0] = trial.time_s[0] - delay_s
    command_times[1:] = trial.time_s[:-1] - delay_s
    return np.interp(
        command_times,
        trial.time_s,
        trial.command_mps,
        left=float(trial.command_mps[0]),
        right=float(trial.command_mps[-1]),
    )


def first_order_design(
    trial: Trial, delay_s: float, tau_s: float
) -> tuple[np.ndarray, np.ndarray]:
    """Build the exact linear-in-gain-and-bias simulation-error design."""
    alpha = float(np.exp(-trial.dt_s / tau_s))
    delayed = response_aligned_command(trial, delay_s)
    forcing = delayed.copy()
    forcing[0] = 0.0
    command_basis = lfilter([1.0 - alpha], [1.0, -alpha], forcing)
    sample_index = np.arange(trial.response_mps.size, dtype=np.float64)
    initial_basis = np.power(alpha, sample_index)
    bias_basis = 1.0 - initial_basis
    target = trial.response_mps - trial.response_mps[0] * initial_basis
    return np.column_stack((command_basis, bias_basis)), target


def fit_model(
    trials: list[Trial],
    *,
    model: str,
    max_delay_s: float,
    max_tau_s: float,
    seed: int,
) -> ModelFit:
    """Minimize pooled simulation error for one candidate model."""
    if model not in MODEL_NAMES:
        raise ValueError(f"Unknown model {model!r}; expected one of {MODEL_NAMES}.")
    if not trials:
        raise ValueError("At least one trial is required for fitting.")

    min_dt_s = min(trial.dt_s for trial in trials)
    min_tau_s = max(1.0e-4, 0.2 * min_dt_s)
    if max_tau_s <= min_tau_s:
        raise ValueError(
            f"max_tau_s={max_tau_s} must exceed the minimum {min_tau_s:.6g}."
        )

    def solve_static(delay_s: float) -> tuple[float, float, float]:
        delayed = np.concatenate(
            [response_aligned_command(trial, delay_s)[1:] for trial in trials]
        )
        target = np.concatenate([trial.response_mps[1:] for trial in trials])
        matrix = np.column_stack((delayed, np.ones(delayed.size)))
        gain, bias = np.linalg.lstsq(matrix, target, rcond=None)[0]
        residual = target - matrix @ np.array([gain, bias])
        return float(np.mean(np.square(residual))), float(gain), float(bias)

    def solve_first_order(delay_s: float, tau_s: float) -> tuple[float, float, float]:
        designs = [first_order_design(trial, delay_s, tau_s) for trial in trials]
        matrix = np.vstack([design[0] for design in designs])
        target = np.concatenate([design[1] for design in designs])
        gain, bias = np.linalg.lstsq(matrix, target, rcond=None)[0]
        residual = target - matrix @ np.array([gain, bias])
        return float(np.mean(np.square(residual))), float(gain), float(bias)

    if model == "unit_delay":
        objective = lambda delay: prediction_mse(
            trials,
            ModelFit(
                model=model,
                delay_s=float(delay),
                tau_s=None,
                gain=1.0,
                bias_mps=0.0,
                rmse_mps=0.0,
                mae_mps=0.0,
                r2=0.0,
                fit_percent=0.0,
            ),
        )
        delay_s = minimize_delay(objective, max_delay_s)
        tau_s, gain, bias = None, 1.0, 0.0
    elif model == "gain_bias_delay":
        delay_s = minimize_delay(
            lambda delay: solve_static(float(delay))[0], max_delay_s
        )
        tau_s = None
        _, gain, bias = solve_static(delay_s)
    elif model == "first_order":
        result = minimize_scalar(
            lambda tau: solve_first_order(0.0, float(tau))[0],
            bounds=(min_tau_s, max_tau_s),
            method="bounded",
            options={"xatol": 1.0e-7},
        )
        delay_s, tau_s = 0.0, float(result.x)
        _, gain, bias = solve_first_order(delay_s, tau_s)
    else:
        result = differential_evolution(
            lambda values: solve_first_order(float(values[0]), float(values[1]))[0],
            bounds=((0.0, max_delay_s), (min_tau_s, max_tau_s)),
            seed=seed,
            popsize=8,
            tol=1.0e-7,
            polish=True,
            workers=1,
            updating="immediate",
        )
        delay_s, tau_s = float(result.x[0]), float(result.x[1])
        _, gain, bias = solve_first_order(delay_s, tau_s)

    provisional = ModelFit(
        model=model,
        delay_s=delay_s,
        tau_s=tau_s,
        gain=gain,
        bias_mps=bias,
        rmse_mps=0.0,
        mae_mps=0.0,
        r2=0.0,
        fit_percent=0.0,
    )
    fitted_values = asdict(provisional)
    fitted_values.update(evaluate(trials, provisional))
    return ModelFit(**fitted_values)


def minimize_delay(objective: Any, max_delay_s: float) -> float:
    """Globally search a possibly multimodal delay objective in one dimension."""
    grid = np.linspace(0.0, max_delay_s, 201)
    values = np.asarray([float(objective(delay_s)) for delay_s in grid])
    candidate_indices = [
        index
        for index in range(1, grid.size - 1)
        if values[index] <= values[index - 1] and values[index] <= values[index + 1]
    ]
    best_delay = float(grid[int(np.argmin(values))])
    best_value = float(np.min(values))
    for index in candidate_indices:
        result = minimize_scalar(
            objective,
            bounds=(float(grid[index - 1]), float(grid[index + 1])),
            method="bounded",
            options={"xatol": 1.0e-7},
        )
        if float(result.fun) < best_value:
            best_delay, best_value = float(result.x), float(result.fun)
    return best_delay


def predict(trial: Trial, fit: ModelFit) -> np.ndarray:
    delayed = response_aligned_command(trial, fit.delay_s)
    if fit.tau_s is None:
        prediction = fit.gain * delayed + fit.bias_mps
        prediction[0] = trial.response_mps[0]
        return prediction

    alpha = float(np.exp(-trial.dt_s / fit.tau_s))
    prediction = np.empty_like(trial.response_mps)
    prediction[0] = trial.response_mps[0]
    for index in range(1, prediction.size):
        target = fit.gain * delayed[index] + fit.bias_mps
        prediction[index] = alpha * prediction[index - 1] + (1.0 - alpha) * target
    return prediction


def prediction_mse(trials: Iterable[Trial], fit: ModelFit) -> float:
    residuals = [trial.response_mps[1:] - predict(trial, fit)[1:] for trial in trials]
    return float(np.mean(np.square(np.concatenate(residuals))))


def evaluate(trials: Iterable[Trial], fit: ModelFit) -> dict[str, float]:
    actual_parts: list[np.ndarray] = []
    predicted_parts: list[np.ndarray] = []
    for trial in trials:
        actual_parts.append(trial.response_mps[1:])
        predicted_parts.append(predict(trial, fit)[1:])
    actual = np.concatenate(actual_parts)
    predicted = np.concatenate(predicted_parts)
    residual = actual - predicted
    residual_norm = float(np.linalg.norm(residual))
    centered_norm = float(np.linalg.norm(actual - np.mean(actual)))
    r2 = (
        float(1.0 - residual_norm**2 / centered_norm**2)
        if centered_norm > 0.0
        else 0.0
    )
    fit_percent = 100.0 * (1.0 - residual_norm / centered_norm) if centered_norm > 0.0 else 0.0
    return {
        "rmse_mps": float(np.sqrt(np.mean(np.square(residual)))),
        "mae_mps": float(np.mean(np.abs(residual))),
        "r2": r2,
        "fit_percent": float(fit_percent),
    }


def leave_one_trial_out(
    trials: list[Trial],
    *,
    max_delay_s: float,
    max_tau_s: float,
    seed: int,
) -> tuple[dict[tuple[str, str], ModelFit], list[dict[str, Any]]]:
    fits: dict[tuple[str, str], ModelFit] = {}
    rows: list[dict[str, Any]] = []
    for fold_index, held_out in enumerate(trials):
        training = [trial for trial in trials if trial.trial_id != held_out.trial_id]
        for model in MODEL_NAMES:
            fit = fit_model(
                training,
                model=model,
                max_delay_s=max_delay_s,
                max_tau_s=max_tau_s,
                seed=seed + fold_index,
            )
            fits[(held_out.trial_id, model)] = fit
            held_out_metrics = evaluate([held_out], fit)
            rows.append(
                {
                    "held_out_trial_id": held_out.trial_id,
                    "held_out_trial_index": held_out.trial_index,
                    "held_out_phase": held_out.phase,
                    "held_out_control_mode": held_out.control_mode,
                    "held_out_excitation_type": held_out.excitation_type,
                    "held_out_sample_count": held_out.response_mps.size,
                    "held_out_scored_sample_count": held_out.response_mps.size - 1,
                    "train_trial_count": len(training),
                    "train_sample_count": sum(t.response_mps.size for t in training),
                    "train_scored_sample_count": sum(
                        t.response_mps.size - 1 for t in training
                    ),
                    "model": model,
                    "train_delay_s": fit.delay_s,
                    "train_tau_s": fit.tau_s,
                    "train_gain": fit.gain,
                    "train_bias_mps": fit.bias_mps,
                    "train_rmse_mps": fit.rmse_mps,
                    "train_mae_mps": fit.mae_mps,
                    "train_r2": fit.r2,
                    "train_fit_percent": fit.fit_percent,
                    "held_out_rmse_mps": held_out_metrics["rmse_mps"],
                    "held_out_mae_mps": held_out_metrics["mae_mps"],
                    "held_out_r2": held_out_metrics["r2"],
                    "held_out_fit_percent": held_out_metrics["fit_percent"],
                }
            )
    return fits, rows


def summarize_comparison(
    individual_rows: list[dict[str, Any]], loo_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in MODEL_NAMES:
        fitted = [row for row in individual_rows if row["model"] == model]
        held_out = [row for row in loo_rows if row["model"] == model]
        row: dict[str, Any] = {"model": model, "trial_count": len(fitted)}
        for prefix, selected, field_prefix in (
            ("individual_fit", fitted, ""),
            ("leave_one_out_train", held_out, "train_"),
            ("leave_one_out", held_out, "held_out_"),
        ):
            for metric in ("rmse_mps", "mae_mps", "r2", "fit_percent"):
                values = np.asarray(
                    [float(item[f"{field_prefix}{metric}"]) for item in selected],
                    dtype=np.float64,
                )
                row[f"{prefix}_{metric}_mean"] = float(np.mean(values))
                row[f"{prefix}_{metric}_std"] = sample_std(values)
                row[f"{prefix}_{metric}_median"] = float(np.median(values))
        rows.append(row)
    return rows


def trial_columns(trial: Trial) -> dict[str, Any]:
    return {
        "trial_id": trial.trial_id,
        "trial_index": trial.trial_index,
        "phase": trial.phase,
        "control_mode": trial.control_mode,
        "excitation_type": trial.excitation_type,
        "source_file": str(trial.source_file),
        "sample_count": trial.response_mps.size,
        "scored_sample_count": trial.response_mps.size - 1,
        "duration_s": float(trial.time_s[-1] - trial.time_s[0]),
        "sample_interval_s": trial.dt_s,
        "sample_rate_hz": 1.0 / trial.dt_s,
    }


def fit_columns(fit: ModelFit) -> dict[str, Any]:
    return asdict(fit)


def sample_std(values: np.ndarray) -> float:
    return float(np.std(values, ddof=1)) if values.size > 1 else 0.0


def serialized_filter(value: str) -> str | list[str] | None:
    """Represent a CLI filter unambiguously in the JSON summary."""
    values = parse_filter_values(value)
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    return list(values)


def count_trial_values(trials: Iterable[Trial], field: str) -> dict[str, int]:
    """Count trial-level categorical values in stable alphabetical order."""
    counts: dict[str, int] = {}
    for trial in trials:
        value = str(getattr(trial, field))
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def trial_plot_label(trial: Trial) -> str:
    """Use a compact group-aware label for recursively discovered trials."""
    parent = Path(trial.trial_id).parent.as_posix()
    if parent == ".":
        return f"Trial {trial.trial_index}"
    return f"{parent}/T{trial.trial_index}"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_predictions(
    path: Path,
    trials: list[Trial],
    individual_fits: dict[tuple[str, str], ModelFit],
    overall_fits: dict[str, ModelFit],
    loo_fits: dict[tuple[str, str], ModelFit],
) -> None:
    arrays: dict[str, np.ndarray] = {}
    used_keys: set[str] = set()
    for trial in trials:
        base_key = re.sub(r"[^A-Za-z0-9_]+", "_", trial.trial_id).strip("_") or "trial"
        key = base_key
        suffix = 2
        while key in used_keys:
            key = f"{base_key}_{suffix}"
            suffix += 1
        used_keys.add(key)
        arrays[f"{key}__time_s"] = trial.time_s
        arrays[f"{key}__command_mps"] = trial.command_mps
        arrays[f"{key}__response_aligned_command_mps"] = response_aligned_command(
            trial, 0.0
        )
        arrays[f"{key}__response_mps"] = trial.response_mps
        for model in MODEL_NAMES:
            arrays[f"{key}__{model}__individual"] = predict(
                trial, individual_fits[(trial.trial_id, model)]
            )
            arrays[f"{key}__{model}__overall"] = predict(trial, overall_fits[model])
            arrays[f"{key}__{model}__leave_one_out"] = predict(
                trial, loo_fits[(trial.trial_id, model)]
            )
    np.savez_compressed(path, **arrays)


def plot_model_comparison(loo_rows: list[dict[str, Any]], output_path: Path) -> None:
    with plt.rc_context(
        {"hatch.linewidth": 1.2, "pdf.fonttype": 42, "ps.fonttype": 42}
    ):
        figure, axes = plt.subplots(
            1, 2, figsize=(12.4, 4.5), constrained_layout=True
        )
        panels = (
            (axes[0], "rmse_mps", "RMSE (m/s)"),
            (axes[1], "r2", "R²"),
        )
        positions = np.arange(len(MODEL_NAMES), dtype=np.float64)
        width = 0.34
        styles = (
            {
                "facecolor": "#8FB7DE",
                "edgecolor": "#24557A",
                "hatch": "///",
                "label": "Training fit",
            },
            {
                "facecolor": "#F3B46B",
                "edgecolor": "#934F0B",
                "hatch": "\\\\\\",
                "label": "Held-out trial",
            },
        )
        for axis, metric, ylabel in panels:
            train_values = [
                np.asarray(
                    [
                        float(row[f"train_{metric}"])
                        for row in loo_rows
                        if row["model"] == model
                    ]
                )
                for model in MODEL_NAMES
            ]
            held_out_values = [
                np.asarray(
                    [
                        float(row[f"held_out_{metric}"])
                        for row in loo_rows
                        if row["model"] == model
                    ]
                )
                for model in MODEL_NAMES
            ]
            train_means = np.asarray([np.mean(values) for values in train_values])
            train_stds = np.asarray([sample_std(values) for values in train_values])
            held_out_means = np.asarray([np.mean(values) for values in held_out_values])
            held_out_stds = np.asarray([sample_std(values) for values in held_out_values])
            axis.bar(
                positions - width / 2.0,
                train_means,
                width,
                yerr=train_stds,
                linewidth=1.4,
                capsize=5,
                error_kw={"elinewidth": 1.6, "capthick": 1.6},
                **styles[0],
            )
            axis.bar(
                positions + width / 2.0,
                held_out_means,
                width,
                yerr=held_out_stds,
                linewidth=1.4,
                capsize=5,
                error_kw={"elinewidth": 1.6, "capthick": 1.6},
                **styles[1],
            )
            axis.set_xticks(positions, [MODEL_LABELS[model] for model in MODEL_NAMES])
            axis.set_ylabel(ylabel, fontsize=16, fontweight="bold")
            axis.tick_params(axis="both", labelsize=14, width=1.3, length=5)
            for tick_label in (*axis.get_xticklabels(), *axis.get_yticklabels()):
                tick_label.set_fontweight("semibold")
            for spine in axis.spines.values():
                spine.set_linewidth(1.2)
            axis.grid(axis="y", color="#B8B8B8", linewidth=0.9, alpha=0.45)
            axis.set_axisbelow(True)
            axis.set_ylim(bottom=0.0)
        axes[0].legend(
            fontsize=13,
            frameon=True,
            framealpha=0.95,
            edgecolor="#777777",
        )
        figure.savefig(output_path, dpi=400, bbox_inches="tight", facecolor="white")
        figure.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
        plt.close(figure)


def plot_fopdt_parameters(
    trials: list[Trial],
    individual_fits: dict[tuple[str, str], ModelFit],
    output_path: Path,
) -> None:
    labels = [trial_plot_label(trial) for trial in trials]
    fits = [individual_fits[(trial.trial_id, "fopdt")] for trial in trials]
    values = np.vstack(
        (
            np.asarray([fit.gain for fit in fits]),
            1.0e3 * np.asarray([fit.bias_mps for fit in fits]),
            1.0e3 * np.asarray([float(fit.tau_s) for fit in fits]),
            1.0e3 * np.asarray([fit.delay_s for fit in fits]),
        )
    )
    means = np.mean(values, axis=1)
    standard_deviations = np.asarray([sample_std(row) for row in values])
    denominators = np.maximum(np.abs(means), np.finfo(np.float64).eps)
    relative_deviations = 100.0 * (values - means[:, None]) / denominators[:, None]
    color_limit = max(10.0, 10.0 * np.ceil(np.max(np.abs(relative_deviations)) / 10.0))

    row_names = (
        "Gain K",
        "Bias b (mm/s)",
        "Time constant tau (ms)",
        "Additional pure delay L (ms)",
    )
    row_labels = [
        f"{name}\nmean +/- SD: {mean:.3g} +/- {std:.2g}"
        for name, mean, std in zip(row_names, means, standard_deviations)
    ]
    figure_width = min(19.0, max(10.5, 4.0 + 0.45 * len(trials)))
    figure, axis = plt.subplots(figsize=(figure_width, 4.8), constrained_layout=True)
    image = axis.imshow(
        relative_deviations,
        cmap="RdBu_r",
        vmin=-color_limit,
        vmax=color_limit,
        aspect="auto",
    )
    formats = (".3f", ".3f", ".1f", ".1f")
    for row_index in range(values.shape[0]):
        for column_index in range(values.shape[1]):
            deviation = relative_deviations[row_index, column_index]
            text_color = "white" if abs(deviation) > 0.58 * color_limit else "black"
            axis.text(
                column_index,
                row_index,
                format(values[row_index, column_index], formats[row_index]),
                ha="center",
                va="center",
                color=text_color,
                fontsize=10,
                fontweight="semibold",
            )
    axis.set_xticks(np.arange(len(trials)), labels, rotation=55, ha="right")
    axis.set_yticks(np.arange(len(row_labels)), row_labels)
    axis.set_xlabel("Individually fitted trial")
    axis.tick_params(axis="both", length=0)
    colorbar = figure.colorbar(image, ax=axis, pad=0.025, shrink=0.9)
    colorbar.set_label("Deviation from parameter mean (%)")
    figure.suptitle("FOPDT parameters across excitation trials", fontsize=15)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def build_summary(
    *,
    data_dir: Path,
    args: argparse.Namespace,
    trials: list[Trial],
    overall_fits: dict[str, ModelFit],
    comparison_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "data_dir": str(data_dir),
        "fields": {
            "time": args.time_field,
            "input": args.input_field,
            "response": args.response_field,
            "phase_filter": serialized_filter(args.phase),
            "control_mode_filter": serialized_filter(args.control_mode),
            "excitation_type_filter": serialized_filter(args.excitation_type),
        },
        "bounds": {"max_delay_s": args.max_delay, "max_tau_s": args.max_tau},
        "trial_count": len(trials),
        "excitation_type_trial_counts": count_trial_values(trials, "excitation_type"),
        "sample_count": sum(trial.response_mps.size for trial in trials),
        "scored_sample_count": sum(trial.response_mps.size - 1 for trial in trials),
        "trials": [
            {
                **trial_columns(trial),
                "source_file": str(trial.source_file.relative_to(data_dir)),
            }
            for trial in trials
        ],
        "model_equations": {
            "unit_delay": "y_hat[k+1] = u(t_k - L)",
            "gain_bias_delay": "y_hat[k+1] = K*u(t_k - L) + b",
            "first_order": (
                "y_hat[k+1] = alpha*y_hat[k] + (1-alpha)*(K*u(t_k) + b)"
            ),
            "fopdt": (
                "y_hat[k+1] = alpha*y_hat[k] + "
                "(1-alpha)*(K*u(t_k - L) + b)"
            ),
            "discrete_first_order": (
                "alpha = exp(-dt/tau); L is additional to the fixed one-sample "
                "causal alignment"
            ),
        },
        "causal_alignment": (
            "Command u[k] predicts measured response y[k+1]. The first response "
            "sample has no preceding in-trial command, is used only as the dynamic "
            "initial condition, and is excluded from every fit/error metric."
        ),
        "delay_interpolation": (
            "For response index k>=1, interpolate the command at t[k-1]-L and hold "
            "the first command before trial start. Reported L is the additional pure "
            "delay beyond the fixed one-sample alignment."
        ),
        "dynamic_initial_condition": (
            "Each first-order simulation is initialized from that trial's first measured response."
        ),
        "overall_fits": {model: asdict(fit) for model, fit in overall_fits.items()},
        "comparison_across_trials": comparison_rows,
        "standard_deviation": "Sample standard deviation (ddof=1).",
    }


def format_optional_ms(value_s: float | None) -> str:
    return "--" if value_s is None else f"{1.0e3 * value_s:.2f} ms"


if __name__ == "__main__":
    main()
