#!/usr/bin/env python3
"""Identify real-world vertical-velocity response models from CPID logs."""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.optimize import brentq, differential_evolution, minimize_scalar
from scipy.signal import lfilter

MPLCONFIGDIR = Path(os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib"))
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
warnings.filterwarnings("ignore", message="Unable to import Axes3D.*", category=UserWarning)
import matplotlib.pyplot as plt


MODEL_NAMES = (
    "unit_delay",
    "gain_delay",
    "first_order",
    "fopdt",
    "sopdt",
    "fopdt_omega_du",
    "fopdt_omega_theta_du",
)
AUGMENTED_MODEL_NAMES = ("fopdt_omega_du", "fopdt_omega_theta_du")


@dataclass
class Trajectory:
    """One aligned command/response trajectory."""

    goal: str
    file_path: Path
    file_name: str
    episode_index: int
    batch: str
    command: np.ndarray
    response: np.ndarray
    theta: np.ndarray | None = None
    omega: np.ndarray | None = None


@dataclass
class ModelFit:
    """Identified model parameters and fit metrics."""

    model: str
    delay_s: float
    tau_s: float | None
    gain: float
    bias_mps: float
    rmse_mps: float
    mae_mps: float
    r2: float
    tau2_s: float | None = None
    omega_gain_m_per_rad: float | None = None
    theta_gain_mps_per_rad: float | None = None
    command_rate_gain_s: float | None = None
    low_frequency_delay_s: float | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit delay, first-order, cascaded second-order, and causal state-augmented "
            "velocity-response models to real-world CPID logs."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("logs/real_world/cpid"),
        help="Root containing goal_*/*.pkl files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: <data-dir>/velocity_response_fit).",
    )
    parser.add_argument("--sample-rate", type=float, default=60.0, help="Log sample rate in Hz.")
    parser.add_argument("--max-delay", type=float, default=0.8, help="Maximum fitted delay in seconds.")
    parser.add_argument("--max-tau", type=float, default=1.5, help="Maximum fitted time constant in seconds.")
    parser.add_argument("--plot-duration", type=float, default=12.0, help="Seconds shown for example trajectories.")
    parser.add_argument("--seed", type=int, default=7, help="Optimizer seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.sample_rate <= 0.0:
        raise ValueError("--sample-rate must be positive.")
    if args.max_delay <= 0.0 or args.max_tau <= 0.0:
        raise ValueError("--max-delay and --max-tau must be positive.")

    data_dir = args.data_dir.expanduser().resolve()
    output_dir = (args.output_dir or data_dir / "velocity_response_fit").expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    trajectories = load_trajectories(data_dir)
    if not trajectories:
        raise RuntimeError(f"No usable target_vel_z/drone_vel_z trajectories found below {data_dir}.")

    dt_s = 1.0 / args.sample_rate
    fit_kwargs = {
        "dt_s": dt_s,
        "max_delay_s": args.max_delay,
        "max_tau_s": args.max_tau,
        "seed": args.seed,
    }

    scopes = {
        "overall": trajectories,
        "jan_feb": [item for item in trajectories if item.batch == "jan_feb"],
        "may": [item for item in trajectories if item.batch == "may"],
    }
    scope_fits: dict[str, dict[str, ModelFit]] = {}
    comparison_rows: list[dict[str, Any]] = []
    for scope, records in scopes.items():
        if not records:
            continue
        scope_fits[scope] = {}
        for model in MODEL_NAMES:
            fit = fit_model(records, model=model, **fit_kwargs)
            scope_fits[scope][model] = fit
            comparison_rows.append(
                {
                    "scope": scope,
                    "trajectory_count": len(records),
                    "sample_count": sum(item.response.size for item in records),
                    **asdict(fit),
                    "step_63_time_s": (
                        None
                        if fit.model in AUGMENTED_MODEL_NAMES
                        else step_response_time_s(fit, fraction=1.0 - np.exp(-1.0))
                    ),
                }
            )

    episode_rows: list[dict[str, Any]] = []
    for item in trajectories:
        fit = fit_model([item], model="fopdt", **fit_kwargs)
        episode_rows.append(
            {
                "goal": item.goal,
                "file_name": item.file_name,
                "file_path": str(item.file_path.relative_to(data_dir)),
                "episode_index": item.episode_index,
                "batch": item.batch,
                "sample_count": item.response.size,
                **asdict(fit),
                "step_63_time_s": fit.delay_s + float(fit.tau_s),
            }
        )

    cv_rows = leave_one_file_out(trajectories, **fit_kwargs)
    cv_summary = summarize_cv(cv_rows)
    episode_summary = summarize_episode_fits(episode_rows)
    residual_rows = residual_diagnostics(trajectories, scope_fits["overall"], dt_s)

    write_csv(output_dir / "model_comparison.csv", comparison_rows)
    write_csv(output_dir / "episode_fopdt_parameters.csv", episode_rows)
    write_csv(output_dir / "leave_one_file_out_metrics.csv", cv_rows)
    write_csv(output_dir / "leave_one_file_out_summary.csv", cv_summary)
    write_csv(output_dir / "residual_diagnostics.csv", residual_rows)

    overall_fopdt = scope_fits["overall"]["fopdt"]
    overall_sopdt = scope_fits["overall"]["sopdt"]
    overall_omega_du = scope_fits["overall"]["fopdt_omega_du"]
    overall_omega_theta_du = scope_fits["overall"]["fopdt_omega_theta_du"]
    discrete = discrete_fopdt_coefficients(overall_fopdt, dt_s)
    discrete_sopdt = discrete_sopdt_coefficients(overall_sopdt, dt_s)
    discrete_omega_du = discrete_augmented_coefficients(overall_omega_du, dt_s)
    discrete_omega_theta_du = discrete_augmented_coefficients(
        overall_omega_theta_du, dt_s
    )
    summary = {
        "data_dir": str(data_dir),
        "sample_rate_hz": args.sample_rate,
        "dt_s": dt_s,
        "trajectory_count": len(trajectories),
        "file_count": len({item.file_path for item in trajectories}),
        "sample_count": sum(item.response.size for item in trajectories),
        "model_equation": "v_hat[k] = a*v_hat[k-1] + (1-a)*(K*u_delayed[k] + bias)",
        "delay_interpolation": "linear interpolation at t_k - delay_s, held at the initial command before t=0",
        "continuous_time_equation": (
            "tau_s*d(v_hat(t))/dt + v_hat(t) = K*u(t - delay_s) + bias"
        ),
        "sopdt_continuous_time_equations": [
            "tau_fast_s*d(x(t))/dt + x(t) = u(t - delay_s)",
            "tau_slow_s*d(v_hat(t))/dt + v_hat(t) = K*x(t) + bias",
        ],
        "recommended_overall_fopdt": asdict(overall_fopdt),
        "recommended_discrete_60hz": discrete,
        "overall_sopdt": asdict(overall_sopdt),
        "sopdt_discrete_60hz": discrete_sopdt,
        "overall_fopdt_omega_du": asdict(overall_omega_du),
        "fopdt_omega_du_discrete_60hz": discrete_omega_du,
        "overall_fopdt_omega_theta_du": asdict(overall_omega_theta_du),
        "fopdt_omega_theta_du_discrete_60hz": discrete_omega_theta_du,
        "augmented_model_equation": (
            "v_hat[k] = a*v_hat[k-1] + (1-a)*(K*u_delayed[k] + "
            "c_omega*omega[k-1] + c_theta*theta[k-1] + "
            "c_du*(u_delayed[k]-u_delayed[k-1])/dt_s + bias)"
        ),
        "step_response": {
            "fopdt_time_to_63_percent_s": step_response_time_s(
                overall_fopdt, fraction=1.0 - np.exp(-1.0)
            ),
            "fopdt_time_to_95_percent_s": step_response_time_s(
                overall_fopdt, fraction=0.95
            ),
            "sopdt_time_to_63_percent_s": step_response_time_s(
                overall_sopdt, fraction=1.0 - np.exp(-1.0)
            ),
            "sopdt_time_to_95_percent_s": step_response_time_s(
                overall_sopdt, fraction=0.95
            ),
        },
        "scope_fits": {
            scope: {model: asdict(fit) for model, fit in fits.items()}
            for scope, fits in scope_fits.items()
        },
        "episode_fopdt_distribution": episode_summary,
        "leave_one_file_out_summary": cv_summary,
        "residual_diagnostics": residual_rows,
        "augmented_normalized_design_condition_number": {
            model: normalized_augmented_design_condition_number(
                trajectories, scope_fits["overall"][model], dt_s
            )
            for model in AUGMENTED_MODEL_NAMES
        },
        "notes": [
            "The logs are closed-loop CPID experiments rather than dedicated persistently exciting identification runs.",
            "Parameters therefore describe the observed command-to-estimated-velocity response in this dataset, including communication, estimator, and low-level-control effects.",
            "Use the leave-one-file-out metrics and episode spread when judging generalization; a single deterministic model does not explain all run-to-run variation.",
            "Augmented models compute command rate from the fractionally delayed target_vel_z; the logged action field is not used as a proxy for command increment.",
            "Theta and omega are lagged by one sample in augmented predictions so that step k never uses a state sampled simultaneously with the response at step k.",
            "The command-rate term and explicit delay are phase-confounded on smooth inputs; low_frequency_delay_s = delay_s - command_rate_gain_s / gain is also reported.",
        ],
    }
    (output_dir / "velocity_response_fit_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    plot_results(
        trajectories,
        scope_fits,
        cv_summary,
        output_dir / "velocity_response_model_fit.png",
        dt_s=dt_s,
        duration_s=args.plot_duration,
    )

    print(f"[INFO] Loaded {len(trajectories)} trajectories from {summary['file_count']} files.")
    print(
        "[INFO] Overall FOPDT: "
        f"delay={1000.0 * overall_fopdt.delay_s:.1f} ms, "
        f"tau={1000.0 * float(overall_fopdt.tau_s):.1f} ms, "
        f"gain={overall_fopdt.gain:.4f}, bias={overall_fopdt.bias_mps:.5f} m/s, "
        f"RMSE={overall_fopdt.rmse_mps:.5f} m/s, R2={overall_fopdt.r2:.4f}."
    )
    print(
        "[INFO] Overall SOPDT: "
        f"delay={1000.0 * overall_sopdt.delay_s:.1f} ms, "
        f"tau_fast={1000.0 * float(overall_sopdt.tau_s):.1f} ms, "
        f"tau_slow={1000.0 * float(overall_sopdt.tau2_s):.1f} ms, "
        f"gain={overall_sopdt.gain:.4f}, bias={overall_sopdt.bias_mps:.5f} m/s, "
        f"RMSE={overall_sopdt.rmse_mps:.5f} m/s, R2={overall_sopdt.r2:.4f}."
    )
    for fit in (overall_omega_du, overall_omega_theta_du):
        print(
            f"[INFO] Overall {fit.model}: "
            f"delay={1000.0 * fit.delay_s:.1f} ms, "
            f"tau={1000.0 * float(fit.tau_s):.1f} ms, "
            f"gain={fit.gain:.4f}, omega_gain={float(fit.omega_gain_m_per_rad):.4f}, "
            f"theta_gain={fit.theta_gain_mps_per_rad}, "
            f"command_rate_gain={float(fit.command_rate_gain_s):.5f} s, "
            f"low_frequency_delay={1000.0 * float(fit.low_frequency_delay_s):.1f} ms, "
            f"RMSE={fit.rmse_mps:.5f} m/s, R2={fit.r2:.4f}."
        )
    print(f"[INFO] Results written to {output_dir}")


def load_trajectories(data_dir: Path) -> list[Trajectory]:
    """Load and align all command/response pairs."""
    trajectories: list[Trajectory] = []
    for path in sorted(data_dir.glob("goal_*/*.pkl")):
        with path.open("rb") as stream:
            data = pickle.load(stream)
        if not isinstance(data, dict) or "target_vel_z" not in data or "drone_vel_z" not in data:
            continue
        commands = as_trajectory_list(data["target_vel_z"])
        responses = as_trajectory_list(data["drone_vel_z"])
        thetas = as_trajectory_list(data["theta"]) if "theta" in data else [None] * len(commands)
        omegas = as_trajectory_list(data["omega"]) if "omega" in data else [None] * len(commands)
        for episode_index, (command_raw, response_raw, theta_raw, omega_raw) in enumerate(
            zip(commands, responses, thetas, omegas)
        ):
            command = np.asarray(command_raw, dtype=np.float64).reshape(-1)
            response = np.asarray(response_raw, dtype=np.float64).reshape(-1)
            theta = (
                None
                if theta_raw is None
                else np.asarray(theta_raw, dtype=np.float64).reshape(-1)
            )
            omega = (
                None
                if omega_raw is None
                else np.asarray(omega_raw, dtype=np.float64).reshape(-1)
            )
            sizes = [command.size, response.size]
            if theta is not None:
                sizes.append(theta.size)
            if omega is not None:
                sizes.append(omega.size)
            count = min(sizes)
            command = command[:count]
            response = response[:count]
            valid = np.isfinite(command) & np.isfinite(response)
            if theta is not None:
                theta = theta[:count]
                valid &= np.isfinite(theta)
            if omega is not None:
                omega = omega[:count]
                valid &= np.isfinite(omega)
            if count < 3 or not np.all(valid):
                continue
            trajectories.append(
                Trajectory(
                    goal=path.parent.name,
                    file_path=path.resolve(),
                    file_name=path.name,
                    episode_index=episode_index,
                    batch=infer_batch(path.name),
                    command=command,
                    response=response,
                    theta=theta,
                    omega=omega,
                )
            )
    return trajectories


def as_trajectory_list(value: Any) -> list[Any]:
    if isinstance(value, np.ndarray) and value.ndim == 1:
        return [value]
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def infer_batch(file_name: str) -> str:
    if "_202601" in file_name or "_202602" in file_name:
        return "jan_feb"
    if "_202605" in file_name:
        return "may"
    return "other"


def delayed_command(command: np.ndarray, delay_s: float, dt_s: float) -> np.ndarray:
    sample_positions = np.arange(command.size, dtype=np.float64) - delay_s / dt_s
    return np.interp(
        sample_positions,
        np.arange(command.size, dtype=np.float64),
        command,
        left=float(command[0]),
        right=float(command[-1]),
    )


def first_order_design(
    item: Trajectory, delay_s: float, tau_s: float, dt_s: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return design columns and target for a simulation-error FOPDT fit."""
    alpha = float(np.exp(-dt_s / tau_s))
    delayed = delayed_command(item.command, delay_s, dt_s)
    forcing = delayed.copy()
    forcing[0] = 0.0
    command_basis = lfilter([1.0 - alpha], [1.0, -alpha], forcing)
    sample_index = np.arange(item.response.size, dtype=np.float64)
    initial_response = item.response[0] * np.power(alpha, sample_index)
    bias_basis = 1.0 - np.power(alpha, sample_index)
    return np.column_stack((command_basis, bias_basis)), item.response - initial_response


def previous_sample(values: np.ndarray) -> np.ndarray:
    """Return a causal one-sample-lagged copy, holding the first sample at the boundary."""
    return np.concatenate((values[:1], values[:-1]))


def augmented_first_order_design(
    item: Trajectory,
    delay_s: float,
    tau_s: float,
    dt_s: float,
    include_theta: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the exact linear design for a causal state-augmented FOPDT rollout."""
    if item.omega is None or (include_theta and item.theta is None):
        required = "theta and omega" if include_theta else "omega"
        raise ValueError(f"The augmented FOPDT model requires {required} in every trajectory.")

    alpha = float(np.exp(-dt_s / tau_s))
    delayed = delayed_command(item.command, delay_s, dt_s)
    command_rate = np.diff(delayed, prepend=delayed[0]) / dt_s
    signals = [delayed, previous_sample(item.omega)]
    if include_theta:
        assert item.theta is not None
        signals.append(previous_sample(item.theta))
    signals.append(command_rate)

    columns: list[np.ndarray] = []
    for signal in signals:
        forcing = signal.copy()
        forcing[0] = 0.0
        columns.append(lfilter([1.0 - alpha], [1.0, -alpha], forcing))

    sample_index = np.arange(item.response.size, dtype=np.float64)
    initial_response = item.response[0] * np.power(alpha, sample_index)
    columns.append(1.0 - np.power(alpha, sample_index))
    return np.column_stack(columns), item.response - initial_response


def second_order_design(
    item: Trajectory,
    delay_s: float,
    tau_fast_s: float,
    tau_slow_s: float,
    dt_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return design columns and target for a cascaded simulation-error SOPDT fit."""
    alpha_fast = float(np.exp(-dt_s / tau_fast_s))
    alpha_slow = float(np.exp(-dt_s / tau_slow_s))
    delayed = delayed_command(item.command, delay_s, dt_s)
    inner_state, _ = lfilter(
        [1.0 - alpha_fast],
        [1.0, -alpha_fast],
        delayed,
        zi=[alpha_fast * delayed[0]],
    )
    forcing = inner_state.copy()
    forcing[0] = 0.0
    command_basis = lfilter([1.0 - alpha_slow], [1.0, -alpha_slow], forcing)
    sample_index = np.arange(item.response.size, dtype=np.float64)
    initial_response = item.response[0] * np.power(alpha_slow, sample_index)
    bias_basis = 1.0 - np.power(alpha_slow, sample_index)
    return np.column_stack((command_basis, bias_basis)), item.response - initial_response


def fit_model(
    records: list[Trajectory],
    model: str,
    dt_s: float,
    max_delay_s: float,
    max_tau_s: float,
    seed: int,
) -> ModelFit:
    if model not in MODEL_NAMES:
        raise ValueError(f"Unknown model {model!r}; expected one of {MODEL_NAMES}.")

    def solve_linear(
        delay_s: float, tau_s: float | None, tau2_s: float | None = None
    ) -> tuple[float, float, float]:
        matrices: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        for item in records:
            if tau_s is None:
                delayed = delayed_command(item.command, delay_s, dt_s)
                matrices.append(np.column_stack((delayed, np.ones(delayed.size))))
                targets.append(item.response)
            elif tau2_s is None:
                matrix, target = first_order_design(item, delay_s, tau_s, dt_s)
                matrices.append(matrix)
                targets.append(target)
            else:
                matrix, target = second_order_design(item, delay_s, tau_s, tau2_s, dt_s)
                matrices.append(matrix)
                targets.append(target)
        matrix = np.vstack(matrices)
        target = np.concatenate(targets)
        gain, bias = np.linalg.lstsq(matrix, target, rcond=None)[0]
        mse = float(np.mean(np.square(target - matrix @ np.array([gain, bias]))))
        return mse, float(gain), float(bias)

    def solve_augmented_linear(
        delay_s: float, tau_s: float, include_theta: bool
    ) -> tuple[float, np.ndarray]:
        designs = [
            augmented_first_order_design(item, delay_s, tau_s, dt_s, include_theta)
            for item in records
        ]
        matrix = np.vstack([design[0] for design in designs])
        target = np.concatenate([design[1] for design in designs])
        coefficients = np.linalg.lstsq(matrix, target, rcond=None)[0]
        mse = float(np.mean(np.square(target - matrix @ coefficients)))
        return mse, coefficients

    omega_gain = None
    theta_gain = None
    command_rate_gain = None

    if model == "unit_delay":
        objective = lambda delay: prediction_mse(
            records,
            ModelFit(model, float(delay), None, 1.0, 0.0, 0.0, 0.0, 0.0),
            dt_s,
        )
        result = minimize_scalar(objective, bounds=(0.0, max_delay_s), method="bounded")
        delay_s, tau_s, tau2_s, gain, bias = float(result.x), None, None, 1.0, 0.0
    elif model == "gain_delay":
        result = minimize_scalar(
            lambda delay: solve_linear(float(delay), None)[0],
            bounds=(0.0, max_delay_s),
            method="bounded",
            options={"xatol": 1.0e-7},
        )
        delay_s, tau_s, tau2_s = float(result.x), None, None
        _, gain, bias = solve_linear(delay_s, tau_s)
    elif model == "first_order":
        result = minimize_scalar(
            lambda tau: solve_linear(0.0, float(tau))[0],
            bounds=(max(1.0e-3, 0.2 * dt_s), max_tau_s),
            method="bounded",
            options={"xatol": 1.0e-7},
        )
        delay_s, tau_s, tau2_s = 0.0, float(result.x), None
        _, gain, bias = solve_linear(delay_s, tau_s)
    elif model == "fopdt":
        result = differential_evolution(
            lambda values: solve_linear(float(values[0]), float(values[1]))[0],
            bounds=((0.0, max_delay_s), (max(1.0e-3, 0.2 * dt_s), max_tau_s)),
            seed=seed,
            popsize=8,
            tol=1.0e-6,
            polish=True,
            workers=1,
            updating="immediate",
        )
        delay_s, tau_s, tau2_s = float(result.x[0]), float(result.x[1]), None
        _, gain, bias = solve_linear(delay_s, tau_s)
    elif model == "sopdt":
        min_tau_s = max(1.0e-3, 0.2 * dt_s)

        def sopdt_objective(values: np.ndarray) -> float:
            tau_fast_s, tau_slow_s = sorted((float(values[1]), float(values[2])))
            return solve_linear(float(values[0]), tau_fast_s, tau_slow_s)[0]

        result = differential_evolution(
            sopdt_objective,
            bounds=((0.0, max_delay_s), (min_tau_s, max_tau_s), (min_tau_s, max_tau_s)),
            seed=seed,
            popsize=8,
            tol=1.0e-6,
            polish=True,
            workers=1,
            updating="immediate",
        )
        delay_s = float(result.x[0])
        tau_s, tau2_s = sorted((float(result.x[1]), float(result.x[2])))
        _, gain, bias = solve_linear(delay_s, tau_s, tau2_s)

    else:
        include_theta = model == "fopdt_omega_theta_du"
        result = differential_evolution(
            lambda values: solve_augmented_linear(
                float(values[0]), float(values[1]), include_theta
            )[0],
            bounds=(
                (0.0, max_delay_s),
                (max(1.0e-3, 0.2 * dt_s), max_tau_s),
            ),
            seed=seed,
            popsize=8,
            tol=1.0e-6,
            polish=True,
            workers=1,
            updating="immediate",
        )
        delay_s, tau_s, tau2_s = float(result.x[0]), float(result.x[1]), None
        _, coefficients = solve_augmented_linear(delay_s, tau_s, include_theta)
        if include_theta:
            gain, omega_gain, theta_gain, command_rate_gain, bias = map(
                float, coefficients
            )
        else:
            gain, omega_gain, command_rate_gain, bias = map(float, coefficients)

    provisional = ModelFit(
        model=model,
        delay_s=delay_s,
        tau_s=tau_s,
        gain=gain,
        bias_mps=bias,
        rmse_mps=0.0,
        mae_mps=0.0,
        r2=0.0,
        tau2_s=tau2_s,
        omega_gain_m_per_rad=omega_gain,
        theta_gain_mps_per_rad=theta_gain,
        command_rate_gain_s=command_rate_gain,
        low_frequency_delay_s=(
            None
            if command_rate_gain is None or abs(gain) < 1.0e-12
            else delay_s - command_rate_gain / gain
        ),
    )
    metrics = evaluate(records, provisional, dt_s)
    return ModelFit(
        model=model,
        delay_s=delay_s,
        tau_s=tau_s,
        gain=gain,
        bias_mps=bias,
        **metrics,
        tau2_s=tau2_s,
        omega_gain_m_per_rad=omega_gain,
        theta_gain_mps_per_rad=theta_gain,
        command_rate_gain_s=command_rate_gain,
        low_frequency_delay_s=(
            None
            if command_rate_gain is None or abs(gain) < 1.0e-12
            else delay_s - command_rate_gain / gain
        ),
    )


def predict(item: Trajectory, fit: ModelFit, dt_s: float) -> np.ndarray:
    delayed = delayed_command(item.command, fit.delay_s, dt_s)
    if fit.tau_s is None:
        return fit.gain * delayed + fit.bias_mps
    if fit.model in AUGMENTED_MODEL_NAMES:
        if item.omega is None or fit.omega_gain_m_per_rad is None:
            raise ValueError("The augmented FOPDT prediction requires omega data and its gain.")
        command_rate = np.diff(delayed, prepend=delayed[0]) / dt_s
        omega = previous_sample(item.omega)
        theta = None if item.theta is None else previous_sample(item.theta)
        alpha = float(np.exp(-dt_s / fit.tau_s))
        prediction = np.empty_like(item.response)
        prediction[0] = item.response[0]
        for index in range(1, prediction.size):
            target = fit.gain * delayed[index]
            target += fit.omega_gain_m_per_rad * omega[index]
            target += float(fit.command_rate_gain_s) * command_rate[index]
            if fit.theta_gain_mps_per_rad is not None:
                if theta is None:
                    raise ValueError("The theta-augmented FOPDT prediction requires theta data.")
                target += fit.theta_gain_mps_per_rad * theta[index]
            target += fit.bias_mps
            prediction[index] = alpha * prediction[index - 1] + (1.0 - alpha) * target
        return prediction
    if fit.tau2_s is not None:
        alpha_fast = float(np.exp(-dt_s / fit.tau_s))
        alpha_slow = float(np.exp(-dt_s / fit.tau2_s))
        inner_state = np.empty_like(item.response)
        prediction = np.empty_like(item.response)
        inner_state[0] = delayed[0]
        prediction[0] = item.response[0]
        for index in range(1, prediction.size):
            inner_state[index] = (
                alpha_fast * inner_state[index - 1] + (1.0 - alpha_fast) * delayed[index]
            )
            target = fit.gain * inner_state[index] + fit.bias_mps
            prediction[index] = alpha_slow * prediction[index - 1] + (1.0 - alpha_slow) * target
        return prediction
    alpha = float(np.exp(-dt_s / fit.tau_s))
    prediction = np.empty_like(item.response)
    prediction[0] = item.response[0]
    for index in range(1, prediction.size):
        target = fit.gain * delayed[index] + fit.bias_mps
        prediction[index] = alpha * prediction[index - 1] + (1.0 - alpha) * target
    return prediction


def prediction_mse(records: Iterable[Trajectory], fit: ModelFit, dt_s: float) -> float:
    errors = [item.response - predict(item, fit, dt_s) for item in records]
    return float(np.mean(np.square(np.concatenate(errors))))


def evaluate(records: Iterable[Trajectory], fit: ModelFit, dt_s: float) -> dict[str, float]:
    actual_parts: list[np.ndarray] = []
    predicted_parts: list[np.ndarray] = []
    for item in records:
        actual_parts.append(item.response)
        predicted_parts.append(predict(item, fit, dt_s))
    actual = np.concatenate(actual_parts)
    predicted = np.concatenate(predicted_parts)
    residual = actual - predicted
    denominator = float(np.sum(np.square(actual - np.mean(actual))))
    return {
        "rmse_mps": float(np.sqrt(np.mean(np.square(residual)))),
        "mae_mps": float(np.mean(np.abs(residual))),
        "r2": float(1.0 - np.sum(np.square(residual)) / denominator) if denominator > 0.0 else float("nan"),
    }


def leave_one_file_out(
    trajectories: list[Trajectory],
    dt_s: float,
    max_delay_s: float,
    max_tau_s: float,
    seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    files = sorted({item.file_path for item in trajectories})
    for fold_index, held_out_file in enumerate(files):
        train = [item for item in trajectories if item.file_path != held_out_file]
        test = [item for item in trajectories if item.file_path == held_out_file]
        for model in MODEL_NAMES:
            fit = fit_model(
                train,
                model=model,
                dt_s=dt_s,
                max_delay_s=max_delay_s,
                max_tau_s=max_tau_s,
                seed=seed + fold_index,
            )
            metrics = evaluate(test, fit, dt_s)
            rows.append(
                {
                    "held_out_goal": test[0].goal,
                    "held_out_file": held_out_file.name,
                    "held_out_batch": test[0].batch,
                    "test_trajectory_count": len(test),
                    "test_sample_count": sum(item.response.size for item in test),
                    "model": model,
                    "train_delay_s": fit.delay_s,
                    "train_tau_s": fit.tau_s,
                    "train_tau2_s": fit.tau2_s,
                    "train_gain": fit.gain,
                    "train_bias_mps": fit.bias_mps,
                    "train_omega_gain_m_per_rad": fit.omega_gain_m_per_rad,
                    "train_theta_gain_mps_per_rad": fit.theta_gain_mps_per_rad,
                    "train_command_rate_gain_s": fit.command_rate_gain_s,
                    "train_low_frequency_delay_s": fit.low_frequency_delay_s,
                    **metrics,
                }
            )
    return rows


def summarize_cv(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for model in MODEL_NAMES:
        selected = [row for row in rows if row["model"] == model]
        sample_count = sum(int(row["test_sample_count"]) for row in selected)
        weighted_mse = sum(
            float(row["rmse_mps"]) ** 2 * int(row["test_sample_count"]) for row in selected
        ) / sample_count
        weighted_mae = sum(
            float(row["mae_mps"]) * int(row["test_sample_count"]) for row in selected
        ) / sample_count
        summary.append(
            {
                "model": model,
                "fold_count": len(selected),
                "sample_count": sample_count,
                "weighted_rmse_mps": float(np.sqrt(weighted_mse)),
                "weighted_mae_mps": weighted_mae,
                "mean_fold_r2": float(np.mean([float(row["r2"]) for row in selected])),
                "median_fold_r2": float(np.median([float(row["r2"]) for row in selected])),
            }
        )
    return summary


def summarize_episode_fits(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    fields = ("delay_s", "tau_s", "gain", "bias_mps", "step_63_time_s", "rmse_mps", "r2")
    summary: dict[str, dict[str, float]] = {}
    for field in fields:
        values = np.asarray([float(row[field]) for row in rows], dtype=np.float64)
        summary[field] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "median": float(np.median(values)),
            "q25": float(np.quantile(values, 0.25)),
            "q75": float(np.quantile(values, 0.75)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }
    return summary


def residual_diagnostics(
    records: list[Trajectory], fits: dict[str, ModelFit], dt_s: float
) -> list[dict[str, Any]]:
    """Summarize within-trajectory residual autocorrelation without crossing resets."""
    lags = (1, 8, 16, 30, 60)
    rows: list[dict[str, Any]] = []
    for model in MODEL_NAMES:
        fit = fits[model]
        residuals = [item.response - predict(item, fit, dt_s) for item in records]
        combined = np.concatenate(residuals)
        row: dict[str, Any] = {
            "model": model,
            "residual_mean_mps": float(np.mean(combined)),
            "residual_std_mps": float(np.std(combined)),
        }
        for lag in lags:
            previous = np.concatenate(
                [residual[:-lag] for residual in residuals if residual.size > lag]
            )
            current = np.concatenate(
                [residual[lag:] for residual in residuals if residual.size > lag]
            )
            row[f"residual_acf_lag_{lag}"] = float(np.corrcoef(previous, current)[0, 1])
        rows.append(row)
    return rows


def normalized_augmented_design_condition_number(
    records: list[Trajectory], fit: ModelFit, dt_s: float
) -> float:
    """Return the condition number after scaling each augmented design column to unit norm."""
    if fit.model not in AUGMENTED_MODEL_NAMES or fit.tau_s is None:
        raise ValueError(f"Expected an augmented first-order fit, got {fit.model!r}.")
    include_theta = fit.model == "fopdt_omega_theta_du"
    matrix = np.vstack(
        [
            augmented_first_order_design(
                item, fit.delay_s, fit.tau_s, dt_s, include_theta
            )[0]
            for item in records
        ]
    )
    norms = np.linalg.norm(matrix, axis=0)
    if np.any(norms <= 0.0):
        return float("inf")
    return float(np.linalg.cond(matrix / norms))


def discrete_fopdt_coefficients(fit: ModelFit, dt_s: float) -> dict[str, float]:
    if fit.tau_s is None:
        raise ValueError("Discrete FOPDT coefficients require a time constant.")
    alpha = float(np.exp(-dt_s / fit.tau_s))
    return {
        "alpha": alpha,
        "command_coefficient": (1.0 - alpha) * fit.gain,
        "bias_coefficient_mps": (1.0 - alpha) * fit.bias_mps,
        "fractional_delay_samples": fit.delay_s / dt_s,
    }


def discrete_sopdt_coefficients(fit: ModelFit, dt_s: float) -> dict[str, float]:
    if fit.tau_s is None or fit.tau2_s is None:
        raise ValueError("Discrete SOPDT coefficients require two time constants.")
    return {
        "alpha_fast": float(np.exp(-dt_s / fit.tau_s)),
        "alpha_slow": float(np.exp(-dt_s / fit.tau2_s)),
        "gain": fit.gain,
        "bias_mps": fit.bias_mps,
        "fractional_delay_samples": fit.delay_s / dt_s,
    }


def discrete_augmented_coefficients(fit: ModelFit, dt_s: float) -> dict[str, float | None]:
    if fit.tau_s is None or fit.omega_gain_m_per_rad is None:
        raise ValueError("Discrete augmented coefficients require tau and omega gain.")
    if fit.command_rate_gain_s is None:
        raise ValueError("Discrete augmented coefficients require a command-rate gain.")
    alpha = float(np.exp(-dt_s / fit.tau_s))
    update_weight = 1.0 - alpha
    return {
        "alpha": alpha,
        "fractional_delay_samples": fit.delay_s / dt_s,
        "low_frequency_delay_s": fit.low_frequency_delay_s,
        "state_lag_samples": 1.0,
        "command_coefficient": update_weight * fit.gain,
        "omega_coefficient_m_per_rad": update_weight * fit.omega_gain_m_per_rad,
        "theta_coefficient_mps_per_rad": (
            None
            if fit.theta_gain_mps_per_rad is None
            else update_weight * fit.theta_gain_mps_per_rad
        ),
        "command_rate_coefficient_s": update_weight * fit.command_rate_gain_s,
        "command_increment_coefficient": (
            update_weight * fit.command_rate_gain_s / dt_s
        ),
        "bias_coefficient_mps": update_weight * fit.bias_mps,
    }


def step_response_time_s(fit: ModelFit, fraction: float) -> float:
    """Return the continuous-time delay-plus-rise time to a normalized step fraction."""
    if not 0.0 < fraction < 1.0:
        raise ValueError(f"Step fraction must be in (0, 1), got {fraction}.")
    if fit.tau_s is None:
        return fit.delay_s
    if fit.tau2_s is None:
        return fit.delay_s - fit.tau_s * np.log(1.0 - fraction)

    tau_fast_s = fit.tau_s
    tau_slow_s = fit.tau2_s

    def response(time_s: float) -> float:
        if np.isclose(tau_fast_s, tau_slow_s, rtol=1.0e-7, atol=1.0e-10):
            ratio = time_s / tau_fast_s
            return 1.0 - (1.0 + ratio) * np.exp(-ratio)
        numerator = tau_slow_s * np.exp(-time_s / tau_slow_s)
        numerator -= tau_fast_s * np.exp(-time_s / tau_fast_s)
        return 1.0 - numerator / (tau_slow_s - tau_fast_s)

    upper_s = 50.0 * max(tau_fast_s, tau_slow_s)
    rise_time_s = brentq(lambda time_s: response(time_s) - fraction, 0.0, upper_s)
    return fit.delay_s + rise_time_s


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_results(
    trajectories: list[Trajectory],
    scope_fits: dict[str, dict[str, ModelFit]],
    cv_summary: list[dict[str, Any]],
    output_path: Path,
    dt_s: float,
    duration_s: float,
) -> None:
    examples: list[Trajectory] = []
    for goal in sorted({item.goal for item in trajectories}):
        candidates = [item for item in trajectories if item.goal == goal]
        examples.append(max(candidates, key=lambda item: float(np.std(item.command))))

    figure, axes = plt.subplots(2, 2, figsize=(15.0, 9.0), constrained_layout=True)
    trajectory_axes = [axes[0, 0], axes[0, 1], axes[1, 0]]
    overall_fopdt = scope_fits["overall"]["fopdt"]
    overall_sopdt = scope_fits["overall"]["sopdt"]
    overall_omega_du = scope_fits["overall"]["fopdt_omega_du"]
    overall_omega_theta_du = scope_fits["overall"]["fopdt_omega_theta_du"]
    overall_delay = scope_fits["overall"]["gain_delay"]
    for axis, item in zip(trajectory_axes, examples):
        count = min(item.response.size, max(2, int(duration_s / dt_s)))
        time_s = np.arange(count) * dt_s
        axis.plot(time_s, item.command[:count], color="0.6", linewidth=1.1, label="command")
        axis.plot(time_s, item.response[:count], color="black", linewidth=1.3, label="measured")
        axis.plot(
            time_s,
            predict(item, overall_delay, dt_s)[:count],
            color="#d95f02",
            linewidth=1.2,
            label="gain + delay",
        )
        axis.plot(
            time_s,
            predict(item, overall_fopdt, dt_s)[:count],
            color="#1b9e77",
            linewidth=1.5,
            label="FOPDT",
        )
        axis.plot(
            time_s,
            predict(item, overall_sopdt, dt_s)[:count],
            color="#377eb8",
            linewidth=1.3,
            linestyle="--",
            label="SOPDT",
        )
        axis.plot(
            time_s,
            predict(item, overall_omega_du, dt_s)[:count],
            color="#984ea3",
            linewidth=1.2,
            label="FOPDT + omega + du",
        )
        axis.plot(
            time_s,
            predict(item, overall_omega_theta_du, dt_s)[:count],
            color="#e7298a",
            linewidth=1.2,
            linestyle=":",
            label="FOPDT + omega + theta + du",
        )
        axis.set_title(f"{item.goal}, {item.batch}, episode {item.episode_index}")
        axis.set_xlabel("Time (s)")
        axis.set_ylabel("Vertical velocity (m/s)")
        axis.grid(alpha=0.25)
    trajectory_axes[0].legend(ncol=2, fontsize=8)

    metric_axis = axes[1, 1]
    in_sample = [scope_fits["overall"][model].r2 for model in MODEL_NAMES]
    cv_r2 = [
        next(float(row["mean_fold_r2"]) for row in cv_summary if row["model"] == model)
        for model in MODEL_NAMES
    ]
    positions = np.arange(len(MODEL_NAMES))
    width = 0.36
    metric_axis.bar(positions - width / 2, in_sample, width, label="in-sample R2", color="#7570b3")
    metric_axis.bar(positions + width / 2, cv_r2, width, label="LOFO mean R2", color="#66a61e")
    metric_axis.set_xticks(positions, [name.replace("_", "\n") for name in MODEL_NAMES])
    metric_axis.set_ylabel("R2")
    metric_axis.set_title("Model comparison")
    metric_axis.grid(axis="y", alpha=0.25)
    metric_axis.legend(fontsize=9)

    figure.suptitle("Real-world vertical-velocity response identification", fontsize=15)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    main()
