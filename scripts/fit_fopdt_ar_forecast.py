#!/usr/bin/env python3
"""Fit an AR residual model on top of FOPDT and evaluate causal online forecasts."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from fit_real_velocity_response import (
    ModelFit,
    Trajectory,
    fit_model,
    load_trajectories,
    predict,
)

import matplotlib.pyplot as plt


@dataclass
class ARFit:
    """Conditional least-squares AR residual model."""

    order: int
    intercept_mps: float
    coefficients: list[float]
    sample_count: int
    innovation_rmse_mps: float
    spectral_radius: float
    normalized_design_condition_number: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit FOPDT + AR(p) and evaluate causal multi-step online forecasts."
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
        help="Output directory (default: <data-dir>/velocity_response_fit/fopdt_ar<order>).",
    )
    parser.add_argument("--sample-rate", type=float, default=60.0)
    parser.add_argument("--ar-order", type=int, default=8)
    parser.add_argument("--forecast-horizon", type=int, default=8)
    parser.add_argument("--max-delay", type=float, default=0.8)
    parser.add_argument("--max-tau", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.sample_rate <= 0.0:
        raise ValueError("--sample-rate must be positive.")
    if args.ar_order <= 0 or args.forecast_horizon <= 0:
        raise ValueError("--ar-order and --forecast-horizon must be positive integers.")

    data_dir = args.data_dir.expanduser().resolve()
    output_dir = (
        args.output_dir
        or data_dir / "velocity_response_fit" / f"fopdt_ar{args.ar_order}"
    ).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_trajectories(data_dir)
    if not records:
        raise RuntimeError(f"No usable trajectories found below {data_dir}.")

    dt_s = 1.0 / args.sample_rate
    fit_kwargs = {
        "dt_s": dt_s,
        "max_delay_s": args.max_delay,
        "max_tau_s": args.max_tau,
    }
    overall_fopdt = fit_model(records, model="fopdt", seed=args.seed, **fit_kwargs)
    overall_ar = fit_residual_ar(records, overall_fopdt, dt_s, args.ar_order)
    in_sample_arrays = online_forecast_arrays(
        records, overall_fopdt, overall_ar, dt_s, args.forecast_horizon
    )
    in_sample_rows = summarize_forecasts(
        in_sample_arrays, "in_sample", dt_s, args.forecast_horizon, args.ar_order
    )

    fold_rows: list[dict[str, Any]] = []
    lofo_arrays = empty_forecast_arrays(args.forecast_horizon)
    files = sorted({record.file_path for record in records})
    for fold_index, held_out_file in enumerate(files):
        train = [record for record in records if record.file_path != held_out_file]
        test = [record for record in records if record.file_path == held_out_file]
        fopdt = fit_model(
            train,
            model="fopdt",
            seed=args.seed + fold_index,
            **fit_kwargs,
        )
        ar_fit = fit_residual_ar(train, fopdt, dt_s, args.ar_order)
        arrays = online_forecast_arrays(test, fopdt, ar_fit, dt_s, args.forecast_horizon)
        append_forecast_arrays(lofo_arrays, arrays)
        fold_metrics = summarize_forecasts(
            arrays,
            "leave_one_file_out_fold",
            dt_s,
            args.forecast_horizon,
            args.ar_order,
        )
        ar_coefficients = {
            f"train_ar_phi_{index + 1}": value
            for index, value in enumerate(ar_fit.coefficients)
        }
        for metrics in fold_metrics:
            fold_rows.append(
                {
                    "held_out_goal": test[0].goal,
                    "held_out_file": held_out_file.name,
                    "held_out_batch": test[0].batch,
                    "test_trajectory_count": len(test),
                    "train_fopdt_delay_s": fopdt.delay_s,
                    "train_fopdt_tau_s": fopdt.tau_s,
                    "train_fopdt_gain": fopdt.gain,
                    "train_fopdt_bias_mps": fopdt.bias_mps,
                    "train_ar_intercept_mps": ar_fit.intercept_mps,
                    "train_ar_spectral_radius": ar_fit.spectral_radius,
                    **ar_coefficients,
                    **metrics,
                }
            )

    lofo_rows = summarize_forecasts(
        lofo_arrays,
        "leave_one_file_out",
        dt_s,
        args.forecast_horizon,
        args.ar_order,
    )
    comparison_rows = in_sample_rows + lofo_rows
    write_csv(output_dir / "forecast_metrics_by_horizon.csv", comparison_rows)
    write_csv(output_dir / "leave_one_file_out_forecast_metrics.csv", fold_rows)

    horizon = args.forecast_horizon
    in_sample_horizon = rows_at_horizon(in_sample_rows, horizon)
    lofo_horizon = rows_at_horizon(lofo_rows, horizon)
    summary = {
        "data_dir": str(data_dir),
        "sample_rate_hz": args.sample_rate,
        "dt_s": dt_s,
        "trajectory_count": len(records),
        "file_count": len(files),
        "sample_count": sum(record.response.size for record in records),
        "ar_order": args.ar_order,
        "forecast_horizon_steps": horizon,
        "forecast_horizon_s": horizon * dt_s,
        "future_command_assumption": (
            "At each forecast origin, unknown commands after the origin are held at the "
            "latest observed command. No logged future command is read."
        ),
        "online_fopdt_definition": (
            "At every forecast origin, the FOPDT state is initialized from the latest "
            "observed velocity and then propagated without reading future responses."
        ),
        "online_information_set": (
            "FOPDT+AR uses the free-running FOPDT nominal state and actual-minus-nominal "
            "residuals through the forecast origin; residuals inside the forecast window "
            "are recursively predicted."
        ),
        "overall_fopdt": asdict(overall_fopdt),
        "overall_ar": asdict(overall_ar),
        "in_sample_horizon_comparison": in_sample_horizon,
        "leave_one_file_out_horizon_comparison": lofo_horizon,
        "forecast_metrics_by_horizon": comparison_rows,
        "notes": [
            (
                "AR coefficients are fitted by conditional least squares on free-running "
                "FOPDT residuals."
            ),
            (
                "The AR model is a forecast correction that requires observed residual "
                "history; it is not an autonomous command-to-velocity simulation model."
            ),
            (
                "Leave-one-file-out folds refit both FOPDT and AR parameters using only "
                "the other files."
            ),
            (
                "Forecasts from adjacent origins overlap, so forecast_count is not a "
                "count of statistically independent trials."
            ),
        ],
    }
    (output_dir / "fopdt_ar_forecast_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    plot_forecast_comparison(
        comparison_rows,
        output_dir / "forecast_comparison.png",
        args.forecast_horizon,
        args.ar_order,
    )

    print(
        f"[INFO] Overall FOPDT delay: {overall_fopdt.delay_s / dt_s:.3f} steps "
        f"({1000.0 * overall_fopdt.delay_s:.1f} ms)."
    )
    print(
        f"[INFO] AR({overall_ar.order}) spectral radius: {overall_ar.spectral_radius:.6f}; "
        f"innovation RMSE: {overall_ar.innovation_rmse_mps:.6f} m/s."
    )
    for scope, selected in (
        ("in-sample", in_sample_horizon),
        ("leave-one-file-out", lofo_horizon),
    ):
        baseline = selected["fopdt_online"]
        augmented = selected[f"fopdt_ar{args.ar_order}"]
        print(
            f"[INFO] {scope} horizon {horizon}: online FOPDT "
            f"RMSE={baseline['rmse_mps']:.6f}, R2={baseline['r2']:.6f}; "
            f"FOPDT+AR({args.ar_order}) "
            f"RMSE={augmented['rmse_mps']:.6f}, R2={augmented['r2']:.6f}; "
            f"RMSE reduction="
            f"{augmented['rmse_reduction_vs_online_fopdt_percent']:.2f}%."
        )
    print(f"[INFO] Results written to {output_dir}")


def fit_residual_ar(
    records: list[Trajectory], fopdt: ModelFit, dt_s: float, order: int
) -> ARFit:
    """Fit AR coefficients to free-running FOPDT residuals by conditional least squares."""
    matrices: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for record in records:
        residual = record.response - predict(record, fopdt, dt_s)
        if residual.size <= order:
            continue
        lag_matrix = np.column_stack(
            [residual[order - lag : residual.size - lag] for lag in range(1, order + 1)]
        )
        matrices.append(np.column_stack((np.ones(lag_matrix.shape[0]), lag_matrix)))
        targets.append(residual[order:])
    if not matrices:
        raise ValueError(f"No trajectory is long enough to fit AR({order}).")

    matrix = np.vstack(matrices)
    target = np.concatenate(targets)
    coefficients = np.linalg.lstsq(matrix, target, rcond=None)[0]
    innovation = target - matrix @ coefficients
    phi = coefficients[1:]
    companion = np.zeros((order, order), dtype=np.float64)
    companion[0] = phi
    if order > 1:
        companion[1:, :-1] = np.eye(order - 1)
    norms = np.linalg.norm(matrix, axis=0)
    condition_number = (
        float("inf")
        if np.any(norms <= 0.0)
        else float(np.linalg.cond(matrix / norms))
    )
    return ARFit(
        order=order,
        intercept_mps=float(coefficients[0]),
        coefficients=[float(value) for value in phi],
        sample_count=target.size,
        innovation_rmse_mps=float(np.sqrt(np.mean(np.square(innovation)))),
        spectral_radius=float(np.max(np.abs(np.linalg.eigvals(companion)))),
        normalized_design_condition_number=condition_number,
    )


def empty_forecast_arrays(horizon: int) -> dict[str, list[list[np.ndarray]]]:
    return {
        "actual": [[] for _ in range(horizon)],
        "fopdt_free_run": [[] for _ in range(horizon)],
        "fopdt_online": [[] for _ in range(horizon)],
        "fopdt_ar": [[] for _ in range(horizon)],
    }


def append_forecast_arrays(
    target: dict[str, list[list[np.ndarray]]],
    source: dict[str, list[list[np.ndarray]]],
) -> None:
    for key in target:
        for horizon_index in range(len(target[key])):
            target[key][horizon_index].extend(source[key][horizon_index])


def online_forecast_arrays(
    records: list[Trajectory],
    fopdt: ModelFit,
    ar_fit: ARFit,
    dt_s: float,
    horizon: int,
) -> dict[str, list[list[np.ndarray]]]:
    """Generate causal rolling forecasts without reading future responses or commands."""
    if fopdt.tau_s is None:
        raise ValueError("Online FOPDT forecasts require a time constant.")
    arrays = empty_forecast_arrays(horizon)
    alpha = float(np.exp(-dt_s / fopdt.tau_s))
    delay_samples = fopdt.delay_s / dt_s
    phi = np.asarray(ar_fit.coefficients, dtype=np.float64)

    for record in records:
        nominal = predict(record, fopdt, dt_s)
        residual = record.response - nominal
        actual_parts = [[] for _ in range(horizon)]
        free_run_parts = [[] for _ in range(horizon)]
        online_parts = [[] for _ in range(horizon)]
        ar_parts = [[] for _ in range(horizon)]
        first_origin = ar_fit.order - 1
        last_origin = record.response.size - horizon - 1
        for origin in range(first_origin, last_origin + 1):
            nominal_state = float(nominal[origin])
            online_state = float(record.response[origin])
            residual_history = [
                float(residual[origin - lag]) for lag in range(ar_fit.order)
            ]
            for step in range(1, horizon + 1):
                sample_index = origin + step
                delayed_position = sample_index - delay_samples
                if delayed_position <= origin:
                    delayed_input = float(
                        np.interp(
                            delayed_position,
                            np.arange(origin + 1, dtype=np.float64),
                            record.command[: origin + 1],
                            left=float(record.command[0]),
                            right=float(record.command[origin]),
                        )
                    )
                else:
                    delayed_input = float(record.command[origin])
                target = fopdt.gain * delayed_input + fopdt.bias_mps
                nominal_state = alpha * nominal_state + (1.0 - alpha) * target
                online_state = alpha * online_state + (1.0 - alpha) * target
                residual_forecast = ar_fit.intercept_mps + float(phi @ residual_history)
                residual_history = [residual_forecast, *residual_history[:-1]]
                actual_parts[step - 1].append(float(record.response[sample_index]))
                free_run_parts[step - 1].append(nominal_state)
                online_parts[step - 1].append(online_state)
                ar_parts[step - 1].append(nominal_state + residual_forecast)

        for step in range(horizon):
            arrays["actual"][step].append(np.asarray(actual_parts[step], dtype=np.float64))
            arrays["fopdt_free_run"][step].append(
                np.asarray(free_run_parts[step], dtype=np.float64)
            )
            arrays["fopdt_online"][step].append(
                np.asarray(online_parts[step], dtype=np.float64)
            )
            arrays["fopdt_ar"][step].append(np.asarray(ar_parts[step], dtype=np.float64))
    return arrays


def summarize_forecasts(
    arrays: dict[str, list[list[np.ndarray]]],
    scope: str,
    dt_s: float,
    horizon: int,
    ar_order: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for step in range(1, horizon + 1):
        actual = np.concatenate(arrays["actual"][step - 1])
        predictions = {
            "fopdt_free_run": np.concatenate(arrays["fopdt_free_run"][step - 1]),
            "fopdt_online": np.concatenate(arrays["fopdt_online"][step - 1]),
            f"fopdt_ar{ar_order}": np.concatenate(arrays["fopdt_ar"][step - 1]),
        }
        denominator = float(np.sum(np.square(actual - np.mean(actual))))
        baseline_rmse = float(
            np.sqrt(np.mean(np.square(actual - predictions["fopdt_online"])))
        )
        for model, prediction_values in predictions.items():
            residual = actual - prediction_values
            rmse = float(np.sqrt(np.mean(np.square(residual))))
            rows.append(
                {
                    "scope": scope,
                    "model": model,
                    "horizon_steps": step,
                    "horizon_s": step * dt_s,
                    "forecast_count": actual.size,
                    "rmse_mps": rmse,
                    "mae_mps": float(np.mean(np.abs(residual))),
                    "r2": float(1.0 - np.sum(np.square(residual)) / denominator),
                    "rmse_reduction_vs_online_fopdt_percent": (
                        100.0 * (baseline_rmse - rmse) / baseline_rmse
                    ),
                }
            )
    return rows


def rows_at_horizon(
    rows: list[dict[str, Any]], horizon: int
) -> dict[str, dict[str, Any]]:
    selected = [row for row in rows if int(row["horizon_steps"]) == horizon]
    return {str(row["model"]): row for row in selected}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_forecast_comparison(
    rows: list[dict[str, Any]], output_path: Path, horizon: int, ar_order: int
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), constrained_layout=True)
    styles = {
        ("in_sample", "fopdt_free_run"): (
            "#7570b3",
            ":",
            "FOPDT free-run, in-sample",
        ),
        ("in_sample", "fopdt_online"): (
            "#1b9e77",
            "-",
            "FOPDT online, in-sample",
        ),
        ("in_sample", f"fopdt_ar{ar_order}"): (
            "#d95f02",
            "-",
            f"FOPDT + AR({ar_order}), in-sample",
        ),
        ("leave_one_file_out", "fopdt_free_run"): (
            "#7570b3",
            ":",
            "FOPDT free-run, LOFO",
        ),
        ("leave_one_file_out", "fopdt_online"): (
            "#1b9e77",
            "--",
            "FOPDT online, LOFO",
        ),
        ("leave_one_file_out", f"fopdt_ar{ar_order}"): (
            "#d95f02",
            "--",
            f"FOPDT + AR({ar_order}), LOFO",
        ),
    }
    for key, (color, linestyle, label) in styles.items():
        scope, model = key
        selected = sorted(
            [row for row in rows if row["scope"] == scope and row["model"] == model],
            key=lambda row: int(row["horizon_steps"]),
        )
        x = [int(row["horizon_steps"]) for row in selected]
        axes[0].plot(
            x,
            [float(row["rmse_mps"]) for row in selected],
            color=color,
            linestyle=linestyle,
            marker="o",
            label=label,
        )
        axes[1].plot(
            x,
            [float(row["r2"]) for row in selected],
            color=color,
            linestyle=linestyle,
            marker="o",
            label=label,
        )
    axes[0].set_ylabel("RMSE (m/s)")
    axes[1].set_ylabel("R2")
    for axis in axes:
        axis.set_xlabel("Forecast horizon (steps)")
        axis.set_xticks(range(1, horizon + 1))
        axis.grid(alpha=0.25)
    axes[0].set_title("Causal online forecast RMSE")
    axes[1].set_title("Causal online forecast R2")
    axes[0].legend(fontsize=8)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    main()
