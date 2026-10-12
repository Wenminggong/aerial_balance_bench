#!/usr/bin/env python3
"""Reproducible task-dependent adequacy analysis for predictor-plus-NMPC.

The script deliberately separates three levels of evidence:

1. exact FOPDT identities and frequency-domain envelopes;
2. a local, unconstrained finite-horizon KKT model of the implemented NMPC,
   composed with the forward delay predictor and the delayed nonlinear plant;
3. externally supplied Isaac/real experiment summaries (never fabricated here).

The reduced closed loop uses the same five-state dynamics, objective weights,
increment bound, sampling period, delay queue, and predictor timing as the
acados controller.  Its finite-horizon quadratic solution is the local
fixed-active-set approximation of the nonlinear NMPC, not a global guarantee.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from functools import lru_cache
import itertools
import json
import math
from pathlib import Path
import sys
from typing import Iterable, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baselines.acados_nmpc_core import (  # noqa: E402
    AcadosNMPCControllerCfg,
    AcadosNMPCModelCfg,
    AcadosNMPCObjectiveCfg,
    AcadosNMPCSolverCfg,
    discrete_dynamics_numpy,
)


PARAMETER_NAMES = ("delay_s", "tau_s", "gain", "bias_mps")
PERIODS_S = (2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0)
ALPHA_GRID = tuple(float(value) for value in np.arange(0.0, 2.0001, 0.25))
ABSOLUTE_ADEQUACY_M = 0.010
RELATIVE_ADEQUACY = 0.10
RELATIVE_DENOMINATOR_FLOOR_M = 0.001


@dataclass(frozen=True)
class ResponseParameters:
    """Up-positive FOPDT parameters with total, rather than additional, delay."""

    delay_s: float
    tau_s: float
    gain: float
    bias_mps: float

    def vector(self) -> np.ndarray:
        return np.asarray(
            [self.delay_s, self.tau_s, self.gain, self.bias_mps],
            dtype=np.float64,
        )

    @classmethod
    def from_vector(cls, values: Sequence[float]) -> "ResponseParameters":
        return cls(*(float(value) for value in values))


@dataclass(frozen=True)
class ParameterStatistics:
    count: int
    mean: ResponseParameters
    std: ResponseParameters
    minimum: ResponseParameters
    p05: ResponseParameters
    median: ResponseParameters
    p95: ResponseParameters
    maximum: ResponseParameters
    pooled_fit: ResponseParameters | None
    pooled_rmse_mps: float | None
    pooled_r2: float | None


@dataclass(frozen=True)
class SimulationMetrics:
    tracking_rmse_m: float
    tracking_mae_m: float
    mismatch_rmse_m: float
    relative_degradation: float
    amplitude_ratio: float
    phase_lag_deg: float
    saturation_rate: float


def _quantile(values: np.ndarray, probability: float) -> np.ndarray:
    try:
        return np.quantile(values, probability, axis=0, method="nearest")
    except TypeError:  # NumPy < 1.22.
        return np.quantile(values, probability, axis=0, interpolation="nearest")


def load_identification_statistics(
    trial_csv: Path,
    *,
    sample_period_s: float,
    overall_csv: Path | None = None,
) -> tuple[ParameterStatistics, np.ndarray]:
    """Load PRBS fits and convert extra delay/NED bias to paper conventions."""

    rows = list(csv.DictReader(trial_csv.open(encoding="utf-8")))
    if not rows:
        raise ValueError(f"No identification rows found in {trial_csv}.")
    values = np.asarray(
        [
            [
                float(row["delay_s"]) + sample_period_s,
                float(row["tau_s"]),
                float(row["gain"]),
                -float(row["bias_mps"]),
            ]
            for row in rows
        ],
        dtype=np.float64,
    )
    if np.any(~np.isfinite(values)) or np.any(values[:, :3] <= 0.0):
        raise ValueError("Identification parameters must be finite, with L/tau/K positive.")

    pooled_fit = None
    pooled_rmse = None
    pooled_r2 = None
    if overall_csv is not None and overall_csv.exists():
        overall_rows = list(csv.DictReader(overall_csv.open(encoding="utf-8")))
        fopdt_rows = [row for row in overall_rows if row.get("model") == "fopdt"]
        if len(fopdt_rows) != 1:
            raise ValueError(f"Expected one pooled FOPDT row in {overall_csv}.")
        row = fopdt_rows[0]
        pooled_fit = ResponseParameters(
            delay_s=float(row["delay_s"]) + sample_period_s,
            tau_s=float(row["tau_s"]),
            gain=float(row["gain"]),
            bias_mps=-float(row["bias_mps"]),
        )
        pooled_rmse = float(row["rmse_mps"])
        pooled_r2 = float(row["r2"])

    statistics = ParameterStatistics(
        count=values.shape[0],
        mean=ResponseParameters.from_vector(np.mean(values, axis=0)),
        std=ResponseParameters.from_vector(np.std(values, axis=0, ddof=1)),
        minimum=ResponseParameters.from_vector(np.min(values, axis=0)),
        p05=ResponseParameters.from_vector(_quantile(values, 0.05)),
        median=ResponseParameters.from_vector(np.median(values, axis=0)),
        p95=ResponseParameters.from_vector(_quantile(values, 0.95)),
        maximum=ResponseParameters.from_vector(np.max(values, axis=0)),
        pooled_fit=pooled_fit,
        pooled_rmse_mps=pooled_rmse,
        pooled_r2=pooled_r2,
    )
    return statistics, values


def scaled_bounds(statistics: ParameterStatistics, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    """Return the asymmetric empirical box scaled about the trial-wise mean."""

    if not math.isfinite(alpha) or alpha < 0.0:
        raise ValueError("alpha must be finite and non-negative.")
    center = statistics.mean.vector()
    lower = center - alpha * (center - statistics.minimum.vector())
    upper = center + alpha * (statistics.maximum.vector() - center)
    if np.any(lower[:3] <= 0.0):
        raise ValueError(f"Scaled L/tau/K bounds became non-positive at alpha={alpha:g}.")
    return lower, upper


def uncertainty_vertices(statistics: ParameterStatistics, alpha: float) -> np.ndarray:
    lower, upper = scaled_bounds(statistics, alpha)
    if alpha == 0.0:
        return lower.reshape(1, -1)
    return np.asarray(list(itertools.product(*zip(lower, upper))), dtype=np.float64)


def fopdt_transfer(parameters: ResponseParameters | np.ndarray, omega_rad_s: float) -> complex:
    values = parameters.vector() if isinstance(parameters, ResponseParameters) else np.asarray(parameters)
    delay_s, tau_s, gain = (float(value) for value in values[:3])
    return gain * np.exp(-1j * omega_rad_s * delay_s) / (1.0 + 1j * omega_rad_s * tau_s)


def fopdt_log_sensitivity(parameters: ResponseParameters, omega_rad_s: float) -> np.ndarray:
    """Return d log(G) / d[L, tau, K] for the non-DC channel."""

    omega = float(omega_rad_s)
    return np.asarray(
        [
            -1j * omega,
            -1j * omega / (1.0 + 1j * omega * parameters.tau_s),
            1.0 / parameters.gain,
        ],
        dtype=np.complex128,
    )


def exact_response_error_step(
    true_velocity: float,
    nominal_velocity: float,
    true_command: float,
    nominal_command: float,
    true_parameters: ResponseParameters,
    nominal_parameters: ResponseParameters,
    sample_period_s: float,
) -> tuple[float, float, float]:
    """Advance both responses and return the exact recurrence error."""

    a = math.exp(-sample_period_s / true_parameters.tau_s)
    a_hat = math.exp(-sample_period_s / nominal_parameters.tau_s)
    target = true_parameters.gain * true_command + true_parameters.bias_mps
    target_hat = nominal_parameters.gain * nominal_command + nominal_parameters.bias_mps
    next_true = a * true_velocity + (1.0 - a) * target
    next_nominal = a_hat * nominal_velocity + (1.0 - a_hat) * target_hat
    recurrence = (
        a * (true_velocity - nominal_velocity)
        + (a - a_hat) * nominal_velocity
        + (1.0 - a) * target
        - (1.0 - a_hat) * target_hat
    )
    return next_true, next_nominal, recurrence


def central_difference_jacobian(
    function,
    point: np.ndarray,
    steps: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Return a central-difference Jacobian and a step-halving discrepancy."""

    point = np.asarray(point, dtype=np.float64)
    steps = np.asarray(steps, dtype=np.float64)
    if point.ndim != 1 or steps.shape != point.shape or np.any(steps <= 0.0):
        raise ValueError("point and positive steps must be equal-length one-dimensional arrays.")

    def evaluate(scale: float) -> np.ndarray:
        columns = []
        for index, step in enumerate(steps):
            perturbation = np.zeros_like(point)
            perturbation[index] = scale * step
            plus = np.atleast_1d(np.asarray(function(point + perturbation), dtype=np.float64))
            minus = np.atleast_1d(np.asarray(function(point - perturbation), dtype=np.float64))
            columns.append((plus - minus) / (2.0 * scale * step))
        return np.column_stack(columns)

    coarse = evaluate(1.0)
    fine = evaluate(0.5)
    denominator = max(float(np.linalg.norm(fine)), 1.0e-12)
    discrepancy = float(np.linalg.norm(fine - coarse) / denominator)
    return fine, discrepancy


def periodic_sensitivity_solution(
    state_jacobians: Sequence[np.ndarray],
    parameter_jacobians: Sequence[np.ndarray],
) -> tuple[np.ndarray, float]:
    """Solve the steady periodic sensitivity S[k+1]=A[k]S[k]+E[k]."""

    if not state_jacobians or len(state_jacobians) != len(parameter_jacobians):
        raise ValueError("A non-empty, equal-length periodic Jacobian sequence is required.")
    matrices_a = [np.asarray(value, dtype=np.float64) for value in state_jacobians]
    matrices_e = [np.asarray(value, dtype=np.float64) for value in parameter_jacobians]
    state_dim = matrices_a[0].shape[0]
    parameter_dim = matrices_e[0].shape[1]
    if any(value.shape != (state_dim, state_dim) for value in matrices_a):
        raise ValueError("All state Jacobians must be square and share a shape.")
    if any(value.shape != (state_dim, parameter_dim) for value in matrices_e):
        raise ValueError("All parameter Jacobians must share shape (nx, ntheta).")

    monodromy = np.eye(state_dim)
    accumulated = np.zeros((state_dim, parameter_dim), dtype=np.float64)
    for matrix_a, matrix_e in zip(matrices_a, matrices_e):
        accumulated = matrix_a @ accumulated + matrix_e
        monodromy = matrix_a @ monodromy
    spectral_radius = float(np.max(np.abs(np.linalg.eigvals(monodromy))))
    if spectral_radius >= 1.0:
        raise ValueError(
            f"Periodic local sensitivity requires spectral radius < 1; got {spectral_radius:.6g}."
        )
    initial = np.linalg.solve(np.eye(state_dim) - monodromy, accumulated)
    sensitivities = [initial]
    current = initial
    for matrix_a, matrix_e in zip(matrices_a, matrices_e):
        current = matrix_a @ current + matrix_e
        sensitivities.append(current)
    return np.stack(sensitivities[:-1]), spectral_radius


def worst_case_periodic_rms(
    output_sensitivities: np.ndarray,
    lower_parameter_error: np.ndarray,
    upper_parameter_error: np.ndarray,
) -> float:
    """Evaluate the exact vertex maximum of a scalar-output linear RMS map."""

    sensitivities = np.asarray(output_sensitivities, dtype=np.float64)
    lower = np.asarray(lower_parameter_error, dtype=np.float64)
    upper = np.asarray(upper_parameter_error, dtype=np.float64)
    if sensitivities.ndim != 2 or lower.shape != upper.shape or sensitivities.shape[1] != lower.size:
        raise ValueError("Expected sensitivities (time, ntheta) and matching parameter bounds.")
    vertices = np.asarray(list(itertools.product(*zip(lower, upper))), dtype=np.float64)
    errors = sensitivities @ vertices.T
    return float(np.max(np.sqrt(np.mean(np.square(errors), axis=0))))


def reference_energy_bandwidth(
    reference: np.ndarray,
    sample_period_s: float,
    *,
    energy_fraction: float = 0.95,
) -> float:
    """Return the angular frequency containing a fraction of zero-mean energy."""

    signal = np.asarray(reference, dtype=np.float64).reshape(-1)
    if signal.size < 2 or sample_period_s <= 0.0 or not 0.0 < energy_fraction <= 1.0:
        raise ValueError("A multi-sample signal, positive dt, and fraction in (0,1] are required.")
    spectrum = np.fft.rfft(signal - np.mean(signal))
    energy = np.square(np.abs(spectrum))
    if energy.size:
        energy[0] = 0.0
    total = float(np.sum(energy))
    if total <= 1.0e-24:
        return 0.0
    index = int(np.searchsorted(np.cumsum(energy), energy_fraction * total, side="left"))
    frequencies_hz = np.fft.rfftfreq(signal.size, d=sample_period_s)
    return float(2.0 * math.pi * frequencies_hz[min(index, frequencies_hz.size - 1)])


def frequency_envelope(
    statistics: ParameterStatistics,
    periods_s: Sequence[float],
    alpha_grid: Sequence[float],
    *,
    sample_period_s: float,
) -> list[dict[str, float]]:
    """Compute exact structure and parameter mismatch envelopes."""

    nominal = statistics.mean
    quantized_delay = round(nominal.delay_s / sample_period_s) * sample_period_s
    delay_only = ResponseParameters(quantized_delay, 1.0, 1.0, 0.0)
    rows: list[dict[str, float]] = []
    for period_s in periods_s:
        omega = 2.0 * math.pi / float(period_s)
        nominal_transfer = fopdt_transfer(nominal, omega)
        delay_only_transfer = np.exp(-1j * omega * delay_only.delay_s)
        for alpha in alpha_grid:
            vertices = uncertainty_vertices(statistics, float(alpha))
            transfers = np.asarray([fopdt_transfer(vertex, omega) for vertex in vertices])
            parameter_errors = np.abs(transfers - nominal_transfer)
            structure_errors = np.abs(transfers - delay_only_transfer)
            relative_errors = np.abs(transfers / nominal_transfer - 1.0)
            quantization_phase = omega * abs(nominal.delay_s - quantized_delay)
            rows.append(
                {
                    "period_s": float(period_s),
                    "omega_rad_s": omega,
                    "alpha": float(alpha),
                    "parameter_mismatch_abs_max": float(np.max(parameter_errors)),
                    "parameter_mismatch_relative_max": float(np.max(relative_errors)),
                    "structure_mismatch_abs_max": float(np.max(structure_errors)),
                    "structure_to_parameter_ratio": (
                        float(np.max(structure_errors) / np.max(parameter_errors))
                        if np.max(parameter_errors) > 1.0e-12
                        else 0.0
                    ),
                    "structure_ratio_defined": float(np.max(parameter_errors) > 1.0e-12),
                    "delay_quantization_phase_deg": math.degrees(quantization_phase),
                }
            )
    return rows


def joint_trial_frequency_envelope(
    statistics: ParameterStatistics,
    trial_parameters: np.ndarray,
    periods_s: Sequence[float],
) -> list[dict[str, float]]:
    """Compare the alpha=1 rectangle with the 26 observed joint points."""

    nominal = statistics.mean
    rows: list[dict[str, float]] = []
    box_vertices = uncertainty_vertices(statistics, 1.0)
    for period_s in periods_s:
        omega = 2.0 * math.pi / float(period_s)
        nominal_transfer = fopdt_transfer(nominal, omega)
        box_max = max(
            abs(fopdt_transfer(vertex, omega) - nominal_transfer)
            for vertex in box_vertices
        )
        joint_max = max(
            abs(fopdt_transfer(parameters, omega) - nominal_transfer)
            for parameters in np.asarray(trial_parameters, dtype=np.float64)
        )
        rows.append(
            {
                "period_s": float(period_s),
                "omega_rad_s": omega,
                "box_parameter_mismatch_abs_max": float(box_max),
                "joint_trial_mismatch_abs_max": float(joint_max),
                "box_to_joint_ratio": float(box_max / max(joint_max, 1.0e-12)),
            }
        )
    return rows


def dc_offset_envelope(
    statistics: ParameterStatistics,
    alpha_grid: Sequence[float],
) -> list[dict[str, float]]:
    """Evaluate the constant-command velocity offset induced by K/b mismatch."""

    nominal = statistics.mean
    nominal_equilibrium_command = -nominal.bias_mps / nominal.gain
    rows: list[dict[str, float]] = []
    for alpha in alpha_grid:
        vertices = uncertainty_vertices(statistics, float(alpha))
        offsets = vertices[:, 2] * nominal_equilibrium_command + vertices[:, 3]
        rows.append(
            {
                "alpha": float(alpha),
                "nominal_equilibrium_command_mps": nominal_equilibrium_command,
                "velocity_offset_min_mps": float(np.min(offsets)),
                "velocity_offset_max_mps": float(np.max(offsets)),
                "velocity_offset_abs_max_mps": float(np.max(np.abs(offsets))),
            }
        )
    return rows


def predictor_boundary_envelope(
    statistics: ParameterStatistics,
    periods_s: Sequence[float],
    alpha_grid: Sequence[float],
    *,
    sample_period_s: float,
    command_amplitude_mps: float = 0.30,
    phase_count: int = 48,
) -> list[dict[str, float]]:
    """Worst D-step predictor boundary error for a sinusoidal command history."""

    if command_amplitude_mps <= 0.0 or phase_count < 4:
        raise ValueError("A positive command amplitude and at least four phases are required.")
    nominal = statistics.mean
    nominal_delay_steps = max(0, int(round(nominal.delay_s / sample_period_s)))
    implemented_nominal = ResponseParameters(
        delay_s=nominal_delay_steps * sample_period_s,
        tau_s=nominal.tau_s,
        gain=nominal.gain,
        bias_mps=nominal.bias_mps,
    )
    cfg = _resolved_controller_cfg(sample_period_s)
    phases = np.linspace(0.0, 2.0 * math.pi, phase_count, endpoint=False)
    rows: list[dict[str, float]] = []
    for period_s in periods_s:
        omega = 2.0 * math.pi / float(period_s)
        for alpha in alpha_grid:
            worst = np.zeros(4, dtype=np.float64)
            for vertex in uncertainty_vertices(statistics, float(alpha)):
                true_parameters = ResponseParameters.from_vector(vertex)
                true_transfer = fopdt_transfer(true_parameters, omega)
                for phase in phases:
                    initial_velocity = (
                        command_amplitude_mps
                        * abs(true_transfer)
                        * math.sin(phase + float(np.angle(true_transfer)))
                        + true_parameters.bias_mps
                    )
                    true_state = np.asarray([0.35, 0.0, 0.0, initial_velocity])
                    nominal_state = true_state.copy()
                    for step_id in range(nominal_delay_steps):
                        time_s = step_id * sample_period_s
                        true_command = command_amplitude_mps * math.sin(
                            omega * (time_s - true_parameters.delay_s) + phase
                        )
                        nominal_command = command_amplitude_mps * math.sin(
                            omega * (time_s - implemented_nominal.delay_s) + phase
                        )
                        true_state = nonlinear_step(
                            true_state,
                            true_command,
                            true_parameters,
                            cfg,
                        )
                        nominal_state = nonlinear_step(
                            nominal_state,
                            nominal_command,
                            implemented_nominal,
                            cfg,
                        )
                    worst = np.maximum(worst, np.abs(true_state - nominal_state))
            rows.append(
                {
                    "period_s": float(period_s),
                    "omega_rad_s": omega,
                    "alpha": float(alpha),
                    "delay_steps": float(nominal_delay_steps),
                    "command_amplitude_mps": command_amplitude_mps,
                    "pb_error_max_m": float(worst[0]),
                    "vb_error_max_mps": float(worst[1]),
                    "theta_error_max_rad": float(worst[2]),
                    "vrz_error_max_mps": float(worst[3]),
                }
            )
    return rows


@lru_cache(maxsize=4)
def _resolved_controller_cfg(sample_period_s: float) -> AcadosNMPCControllerCfg:
    return AcadosNMPCControllerCfg(
        model=AcadosNMPCModelCfg(
            plank_length=1.06,
            rope_length=0.9,
            ball_position_offset=0.33,
            gravity=9.81,
            ball_mass=0.0005,
            ball_radius=0.023,
            ball_inertia_ratio=0.4,
            epsilon=1.0e-9,
        ),
        objective=AcadosNMPCObjectiveCfg(
            stage_weights=[20.0, 4.0, 0.1, 0.2, 0.2, 0.05, 0.25],
            terminal_weights=[100.0, 20.0, 0.5, 1.0, 1.0, 0.05],
        ),
        solver=AcadosNMPCSolverCfg(n_horizon=30, step_dt=sample_period_s),
    )


def nonlinear_step(
    state: np.ndarray,
    held_command: float,
    parameters: ResponseParameters,
    cfg: AcadosNMPCControllerCfg,
) -> np.ndarray:
    """Advance [pb,vb,theta,vrz] with the paper's exact-response RK4 map."""

    pb, vb, theta, vrz = (float(value) for value in state)
    dt = float(cfg.solver.step_dt)
    target = parameters.gain * float(held_command) + parameters.bias_mps
    if parameters.tau_s <= float(cfg.model.epsilon):
        vrz_start = vrz_half = vrz_end = target
    else:
        vrz_start = vrz
        vrz_half = target + (vrz - target) * math.exp(-0.5 * dt / parameters.tau_s)
        vrz_end = target + (vrz - target) * math.exp(-dt / parameters.tau_s)

    plank_length = float(cfg.model.plank_length)
    rope_length = float(cfg.model.rope_length)
    offset = float(cfg.model.ball_position_offset)
    gravity = abs(float(cfg.model.gravity))
    gamma = 1.0 / (1.0 + float(cfg.model.ball_inertia_ratio))
    epsilon = float(cfg.model.epsilon)

    def derivatives(values: np.ndarray, response_velocity: float) -> np.ndarray:
        position, velocity, angle = (float(value) for value in values)
        sin_beta = (plank_length / rope_length) * (1.0 - math.cos(angle))
        sin_beta = min(max(sin_beta, -1.0 + epsilon), 1.0 - epsilon)
        beta = math.asin(sin_beta)
        denominator = plank_length * math.cos(beta - angle)
        if abs(denominator) < epsilon:
            denominator = math.copysign(epsilon, denominator if denominator != 0.0 else 1.0)
        omega = -response_velocity * math.cos(beta) / denominator
        acceleration = gamma * (
            (position + offset - plank_length) * omega * omega
            - gravity * math.sin(angle)
        )
        return np.asarray([velocity, acceleration, omega], dtype=np.float64)

    physical = np.asarray([pb, vb, theta], dtype=np.float64)
    k1 = derivatives(physical, vrz_start)
    k2 = derivatives(physical + 0.5 * dt * k1, vrz_half)
    k3 = derivatives(physical + 0.5 * dt * k2, vrz_half)
    k4 = derivatives(physical + dt * k3, vrz_end)
    physical_next = physical + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return np.asarray([*physical_next, vrz_end], dtype=np.float64)


def linearize_internal_model(
    parameters: ResponseParameters,
    cfg: AcadosNMPCControllerCfg,
    *,
    center_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Central-difference the five-state delay-free NMPC model at equilibrium."""

    command_equilibrium = -parameters.bias_mps / parameters.gain
    equilibrium = np.asarray([center_m, 0.0, 0.0, 0.0, command_equilibrium])
    stage_parameters = np.asarray(
        [[center_m, 0.0, parameters.tau_s, parameters.gain, parameters.bias_mps]],
        dtype=np.float64,
    )

    def step(state: np.ndarray, action: float) -> np.ndarray:
        return discrete_dynamics_numpy(
            state.reshape(1, -1),
            np.asarray([[action]], dtype=np.float64),
            stage_parameters,
            cfg,
        )[0]

    state_steps = np.asarray([1.0e-5, 1.0e-5, 1.0e-6, 1.0e-5, 1.0e-5])
    matrix_a = np.empty((5, 5), dtype=np.float64)
    for index, epsilon in enumerate(state_steps):
        perturbation = np.zeros(5)
        perturbation[index] = epsilon
        matrix_a[:, index] = (
            step(equilibrium + perturbation, 0.0)
            - step(equilibrium - perturbation, 0.0)
        ) / (2.0 * epsilon)
    epsilon_u = 1.0e-5
    matrix_b = (
        step(equilibrium, epsilon_u) - step(equilibrium, -epsilon_u)
    ).reshape(-1, 1) / (2.0 * epsilon_u)
    return matrix_a, matrix_b, equilibrium


class LocalFiniteHorizonNMPC:
    """Fixed-active-set quadratic approximation of the nonlinear NMPC KKT map."""

    def __init__(
        self,
        parameters: ResponseParameters,
        cfg: AcadosNMPCControllerCfg,
        *,
        center_m: float,
    ):
        self.parameters = parameters
        self.cfg = cfg
        self.center_m = float(center_m)
        self.matrix_a, self.matrix_b, self.equilibrium = linearize_internal_model(
            parameters,
            cfg,
            center_m=self.center_m,
        )
        self.n_horizon = int(cfg.solver.n_horizon)
        self.action_limit = 5.0 * float(cfg.solver.step_dt)
        self._gain_state, self._gain_reference = self._condense_kkt_map()

    def _stage_output_matrices(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        plank_length = float(self.cfg.model.plank_length)
        matrix_c = np.zeros((7, 5), dtype=np.float64)
        matrix_d = np.zeros((7, 1), dtype=np.float64)
        reference_map = np.zeros((7, 2), dtype=np.float64)
        matrix_c[0, 0] = 1.0
        matrix_c[1, 1] = 1.0
        matrix_c[2, 2] = 1.0
        matrix_c[3, 3] = -1.0 / plank_length
        matrix_c[4, 3] = 1.0
        matrix_c[5, 4] = 1.0
        matrix_d[5, 0] = 1.0
        matrix_d[6, 0] = 1.0
        reference_map[0, 0] = 1.0
        reference_map[1, 1] = 1.0
        return matrix_c, matrix_d, reference_map

    def _terminal_output_matrices(self) -> tuple[np.ndarray, np.ndarray]:
        plank_length = float(self.cfg.model.plank_length)
        matrix_c = np.zeros((6, 5), dtype=np.float64)
        reference_map = np.zeros((6, 2), dtype=np.float64)
        matrix_c[0, 0] = 1.0
        matrix_c[1, 1] = 1.0
        matrix_c[2, 2] = 1.0
        matrix_c[3, 3] = -1.0 / plank_length
        matrix_c[4, 3] = 1.0
        matrix_c[5, 4] = 1.0
        reference_map[0, 0] = 1.0
        reference_map[1, 1] = 1.0
        return matrix_c, reference_map

    def _condense_kkt_map(self) -> tuple[np.ndarray, np.ndarray]:
        n = self.n_horizon
        nx = self.matrix_a.shape[0]
        prediction_x = [np.linalg.matrix_power(self.matrix_a, stage) for stage in range(n + 1)]
        prediction_u: list[np.ndarray] = []
        for stage in range(n + 1):
            matrix = np.zeros((nx, n), dtype=np.float64)
            for action_id in range(stage):
                matrix[:, action_id : action_id + 1] = (
                    np.linalg.matrix_power(self.matrix_a, stage - 1 - action_id)
                    @ self.matrix_b
                )
            prediction_u.append(matrix)

        stage_c, stage_d, stage_r = self._stage_output_matrices()
        terminal_c, terminal_r = self._terminal_output_matrices()
        output_rows = 7 * n + 6
        output_x = np.zeros((output_rows, nx), dtype=np.float64)
        output_u = np.zeros((output_rows, n), dtype=np.float64)
        output_r = np.zeros((output_rows, 2 * (n + 1)), dtype=np.float64)
        weights = np.zeros(output_rows, dtype=np.float64)
        stage_weights = np.asarray(self.cfg.objective.stage_weights, dtype=np.float64)
        terminal_weights = np.asarray(self.cfg.objective.terminal_weights, dtype=np.float64)

        cursor = 0
        for stage in range(n):
            rows = slice(cursor, cursor + 7)
            output_x[rows] = stage_c @ prediction_x[stage]
            output_u[rows] = stage_c @ prediction_u[stage]
            output_u[rows, stage : stage + 1] += stage_d
            output_r[rows, 2 * stage : 2 * stage + 2] = stage_r
            weights[rows] = stage_weights
            cursor += 7
        rows = slice(cursor, cursor + 6)
        output_x[rows] = terminal_c @ prediction_x[n]
        output_u[rows] = terminal_c @ prediction_u[n]
        output_r[rows, 2 * n : 2 * n + 2] = terminal_r
        weights[rows] = terminal_weights

        weighted_output_u = weights[:, None] * output_u
        hessian = output_u.T @ weighted_output_u
        hessian += 1.0e-9 * np.eye(n)
        solve_x = np.linalg.solve(hessian, output_u.T @ (weights[:, None] * output_x))
        solve_r = np.linalg.solve(hessian, output_u.T @ (weights[:, None] * output_r))
        return -solve_x[:1], solve_r[:1]

    def action(self, predicted_state: np.ndarray, references: np.ndarray) -> tuple[float, bool]:
        if references.shape != (self.n_horizon + 1, 2):
            raise ValueError(
                f"Expected references shape {(self.n_horizon + 1, 2)}, got {references.shape}."
            )
        state_deviation = np.asarray(predicted_state, dtype=np.float64) - self.equilibrium
        reference_deviation = references.copy()
        reference_deviation[:, 0] -= self.center_m
        unconstrained = float((
            self._gain_state @ state_deviation
            + self._gain_reference @ reference_deviation.reshape(-1)
        ).item())
        action = float(np.clip(unconstrained, -self.action_limit, self.action_limit))
        return action, not math.isclose(action, unconstrained, rel_tol=0.0, abs_tol=1.0e-12)


class CompositeReducedController:
    """Forward delay predictor followed by the local finite-horizon NMPC map."""

    def __init__(
        self,
        parameters: ResponseParameters,
        cfg: AcadosNMPCControllerCfg,
        *,
        center_m: float,
        response_aware: bool,
    ):
        self.parameters = parameters if response_aware else ResponseParameters(
            delay_s=parameters.delay_s,
            tau_s=1.0e-9,
            gain=1.0,
            bias_mps=0.0,
        )
        self.delay_steps = max(0, int(round(parameters.delay_s / float(cfg.solver.step_dt))))
        self.cfg = cfg
        self.mpc = LocalFiniteHorizonNMPC(self.parameters, cfg, center_m=center_m)

    def predict(self, measured_state: np.ndarray, pending_commands: Sequence[float]) -> np.ndarray:
        predicted = np.asarray(measured_state, dtype=np.float64).copy()
        for command in pending_commands:
            predicted = nonlinear_step(predicted, float(command), self.parameters, self.cfg)
        return predicted

    def action(
        self,
        measured_state: np.ndarray,
        current_command: float,
        pending_commands: Sequence[float],
        references: np.ndarray,
    ) -> tuple[float, bool]:
        predicted = self.predict(measured_state, pending_commands)
        model_state = np.asarray([*predicted, current_command], dtype=np.float64)
        return self.mpc.action(model_state, references)


_COMPOSITE_CONTROLLER_CACHE: dict[tuple, CompositeReducedController] = {}


def cached_composite_controller(
    parameters: ResponseParameters,
    *,
    sample_period_s: float,
    center_m: float,
    response_aware: bool,
) -> CompositeReducedController:
    key = (
        *parameters.vector().tolist(),
        float(sample_period_s),
        float(center_m),
        bool(response_aware),
    )
    controller = _COMPOSITE_CONTROLLER_CACHE.get(key)
    if controller is None:
        controller = CompositeReducedController(
            parameters,
            _resolved_controller_cfg(sample_period_s),
            center_m=center_m,
            response_aware=response_aware,
        )
        _COMPOSITE_CONTROLLER_CACHE[key] = controller
    return controller


def sine_reference(
    indices: np.ndarray,
    *,
    sample_period_s: float,
    period_s: float,
    center_m: float,
    amplitude_m: float,
) -> np.ndarray:
    time_s = np.asarray(indices, dtype=np.float64) * sample_period_s
    omega = 2.0 * math.pi / period_s
    return np.column_stack(
        (
            center_m + amplitude_m * np.sin(omega * time_s),
            amplitude_m * omega * np.cos(omega * time_s),
        )
    )


def _queue_step(queue: list[float], current_command: float) -> float:
    if not queue:
        return current_command
    executed = float(queue.pop(0))
    queue.append(float(current_command))
    return executed


def simulate_reduced_closed_loop(
    true_parameters: ResponseParameters,
    controller_parameters: ResponseParameters,
    *,
    response_aware: bool,
    period_s: float,
    sample_period_s: float,
    center_m: float,
    amplitude_m: float,
    score_periods: int = 2,
    burn_periods: int = 3,
) -> dict[str, np.ndarray | float]:
    cfg = _resolved_controller_cfg(sample_period_s)
    controller = cached_composite_controller(
        controller_parameters,
        sample_period_s=sample_period_s,
        center_m=center_m,
        response_aware=response_aware,
    )
    actual_delay = max(0, int(round(true_parameters.delay_s / sample_period_s)))
    actual_queue = [0.0] * actual_delay
    nominal_queue = [0.0] * controller.delay_steps
    steps_per_period = int(round(period_s / sample_period_s))
    total_steps = (burn_periods + score_periods) * steps_per_period
    state = np.asarray([center_m, 0.0, 0.0, 0.0], dtype=np.float64)
    current_command = 0.0
    ball_positions = np.empty(total_steps, dtype=np.float64)
    references = sine_reference(
        np.arange(total_steps + controller.delay_steps + controller.mpc.n_horizon + 1),
        sample_period_s=sample_period_s,
        period_s=period_s,
        center_m=center_m,
        amplitude_m=amplitude_m,
    )
    saturation_count = 0
    actions = np.empty(total_steps, dtype=np.float64)

    for step in range(total_steps):
        preview_start = step + controller.delay_steps
        preview = references[
            preview_start : preview_start + controller.mpc.n_horizon + 1
        ]
        action, saturated = controller.action(
            state,
            current_command,
            nominal_queue,
            preview,
        )
        current_command += action
        executed_command = _queue_step(actual_queue, current_command)
        _queue_step(nominal_queue, current_command)
        state = nonlinear_step(state, executed_command, true_parameters, cfg)
        ball_positions[step] = state[0]
        actions[step] = action
        saturation_count += int(saturated)

    score_start = burn_periods * steps_per_period
    return {
        "ball_position": ball_positions[score_start:],
        "reference": references[1 : total_steps + 1, 0][score_start:],
        "action": actions[score_start:],
        "saturation_rate": saturation_count / total_steps,
        "steps_per_period": steps_per_period,
    }


def augmented_closed_loop_step(
    augmented_state: np.ndarray,
    true_parameters: ResponseParameters,
    controller_parameters: ResponseParameters,
    references: np.ndarray,
    *,
    sample_period_s: float,
    center_m: float,
) -> tuple[np.ndarray, bool]:
    """Advance the physical state, incremental command, and known delay queue.

    This map is the object linearized in the periodic analysis.  The integer
    queue length is held fixed locally; the residual/fractional delay is
    deliberately handled by the exact frequency and predictor-boundary
    analyses rather than assigned a spurious derivative through ``round``.
    """

    controller = cached_composite_controller(
        controller_parameters,
        sample_period_s=sample_period_s,
        center_m=center_m,
        response_aware=True,
    )
    delay_steps = controller.delay_steps
    state = np.asarray(augmented_state[:4], dtype=np.float64)
    current_command = float(augmented_state[4])
    queue = np.asarray(augmented_state[5:], dtype=np.float64)
    if queue.shape != (delay_steps,):
        raise ValueError(f"Expected an augmented queue of length {delay_steps}, got {queue.shape}.")

    action, saturated = controller.action(state, current_command, queue, references)
    command_next = current_command + action
    if delay_steps:
        executed_command = float(queue[0])
        queue_next = np.concatenate((queue[1:], np.asarray([command_next])))
    else:
        executed_command = command_next
        queue_next = queue
    state_next = nonlinear_step(
        state,
        executed_command,
        true_parameters,
        _resolved_controller_cfg(sample_period_s),
    )
    return np.concatenate((state_next, np.asarray([command_next]), queue_next)), saturated


def periodic_composite_sensitivity_rows(
    statistics: ParameterStatistics,
    periods_s: Sequence[float],
    alpha_grid: Sequence[float],
    *,
    sample_period_s: float,
    center_m: float,
    amplitude_m: float,
) -> list[dict[str, float | str | None]]:
    """Linearize the complete predictor--controller--queue map on sine orbits.

    Two parameter Jacobians are formed: the mean-controller branch perturbs
    only the plant, whereas the oracle branch perturbs both plant and
    predictor/NMPC.  Their periodic sensitivities are subtracted to obtain the
    local paired model-mismatch map.  The delay column is zero within a fixed
    integer queue; its non-smooth queue changes and fractional residual are
    reported separately by :func:`frequency_envelope` and
    :func:`predictor_boundary_envelope`.
    """

    nominal = statistics.mean
    delay_steps = max(0, int(round(nominal.delay_s / sample_period_s)))
    state_steps = np.asarray(
        [1.0e-5, 1.0e-5, 1.0e-6, 1.0e-5, 1.0e-5, *([1.0e-5] * delay_steps)],
        dtype=np.float64,
    )
    parameter_steps = np.asarray([1.0e-4, 1.0e-4, 1.0e-4, 1.0e-5], dtype=np.float64)
    rows: list[dict[str, float | str | None]] = []

    for period_s in periods_s:
        steps_per_period = int(round(float(period_s) / sample_period_s))
        burn_periods = 5
        horizon = _resolved_controller_cfg(sample_period_s).solver.n_horizon
        total_steps = (burn_periods + 1) * steps_per_period
        references_all = sine_reference(
            np.arange(total_steps + delay_steps + horizon + 1),
            sample_period_s=sample_period_s,
            period_s=float(period_s),
            center_m=center_m,
            amplitude_m=amplitude_m,
        )
        augmented = np.concatenate(
            (np.asarray([center_m, 0.0, 0.0, 0.0, 0.0]), np.zeros(delay_steps))
        )
        orbit: list[np.ndarray] = []
        previews: list[np.ndarray] = []
        nominal_saturation_count = 0
        for step in range(total_steps):
            preview_start = step + delay_steps
            preview = references_all[preview_start : preview_start + horizon + 1]
            if step >= burn_periods * steps_per_period:
                orbit.append(augmented.copy())
                previews.append(preview.copy())
            augmented, saturated = augmented_closed_loop_step(
                augmented,
                nominal,
                nominal,
                preview,
                sample_period_s=sample_period_s,
                center_m=center_m,
            )
            if step >= burn_periods * steps_per_period:
                nominal_saturation_count += int(saturated)

        matrices_a: list[np.ndarray] = []
        matrices_mean: list[np.ndarray] = []
        matrices_oracle: list[np.ndarray] = []
        max_fd_discrepancy = 0.0
        for state, preview in zip(orbit, previews):
            def state_map(value: np.ndarray) -> np.ndarray:
                return augmented_closed_loop_step(
                    value,
                    nominal,
                    nominal,
                    preview,
                    sample_period_s=sample_period_s,
                    center_m=center_m,
                )[0]

            matrix_a, discrepancy_a = central_difference_jacobian(
                state_map,
                state,
                state_steps,
            )

            def mean_parameter_map(value: np.ndarray) -> np.ndarray:
                return augmented_closed_loop_step(
                    state,
                    ResponseParameters.from_vector(value),
                    nominal,
                    preview,
                    sample_period_s=sample_period_s,
                    center_m=center_m,
                )[0]

            def oracle_parameter_map(value: np.ndarray) -> np.ndarray:
                parameters = ResponseParameters.from_vector(value)
                return augmented_closed_loop_step(
                    state,
                    parameters,
                    parameters,
                    preview,
                    sample_period_s=sample_period_s,
                    center_m=center_m,
                )[0]

            matrix_mean, discrepancy_mean = central_difference_jacobian(
                mean_parameter_map,
                nominal.vector(),
                parameter_steps,
            )
            matrix_oracle, discrepancy_oracle = central_difference_jacobian(
                oracle_parameter_map,
                nominal.vector(),
                parameter_steps,
            )
            matrices_a.append(matrix_a)
            matrices_mean.append(matrix_mean)
            matrices_oracle.append(matrix_oracle)
            max_fd_discrepancy = max(
                max_fd_discrepancy,
                discrepancy_a,
                discrepancy_mean,
                discrepancy_oracle,
            )

        spectral_radius: float | None = None
        paired_sensitivities: np.ndarray | None = None
        try:
            mean_sensitivities, spectral_radius = periodic_sensitivity_solution(
                matrices_a,
                matrices_mean,
            )
            oracle_sensitivities, _ = periodic_sensitivity_solution(
                matrices_a,
                matrices_oracle,
            )
            paired_sensitivities = mean_sensitivities - oracle_sensitivities
        except (ValueError, np.linalg.LinAlgError):
            monodromy = np.eye(matrices_a[0].shape[0])
            for matrix_a in matrices_a:
                monodromy = matrix_a @ monodromy
            spectral_radius = float(np.max(np.abs(np.linalg.eigvals(monodromy))))

        saturation_rate = nominal_saturation_count / steps_per_period
        locally_valid = bool(
            paired_sensitivities is not None
            and spectral_radius is not None
            and spectral_radius < 1.0
            and max_fd_discrepancy <= 0.05
            and saturation_rate <= 0.05
        )
        for alpha in alpha_grid:
            linear_rms: float | None = None
            if paired_sensitivities is not None:
                lower, upper = scaled_bounds(statistics, float(alpha))
                linear_rms = worst_case_periodic_rms(
                    paired_sensitivities[:, 0, :],
                    lower - nominal.vector(),
                    upper - nominal.vector(),
                )
            rows.append(
                {
                    "period_s": float(period_s),
                    "omega_rad_s": 2.0 * math.pi / float(period_s),
                    "alpha": float(alpha),
                    "linear_paired_rms_m": linear_rms,
                    "monodromy_spectral_radius": spectral_radius,
                    "fd_step_halving_relative_change": max_fd_discrepancy,
                    "nominal_saturation_rate": saturation_rate,
                    "local_ltv_valid": float(locally_valid),
                    "delay_sensitivity_mode": "fixed_integer_queue; residual handled separately",
                }
            )
    return rows


def _sinusoid_fit(signal: np.ndarray, period_samples: int) -> tuple[float, float]:
    indices = np.arange(signal.size, dtype=np.float64)
    angle = 2.0 * math.pi * indices / period_samples
    design = np.column_stack((np.sin(angle), np.cos(angle), np.ones_like(angle)))
    coefficients = np.linalg.lstsq(design, signal, rcond=None)[0]
    amplitude = float(np.hypot(coefficients[0], coefficients[1]))
    phase = float(math.atan2(coefficients[1], coefficients[0]))
    return amplitude, phase


def paired_metrics(
    rollout: dict[str, np.ndarray | float],
    oracle: dict[str, np.ndarray | float],
    *,
    amplitude_m: float,
) -> SimulationMetrics:
    ball = np.asarray(rollout["ball_position"])
    ball_oracle = np.asarray(oracle["ball_position"])
    reference = np.asarray(rollout["reference"])
    oracle_reference = np.asarray(oracle["reference"])
    if not np.allclose(reference, oracle_reference, rtol=0.0, atol=1.0e-12):
        raise ValueError("Paired rollouts must use identical references.")
    tracking_error = ball - reference
    oracle_tracking_error = ball_oracle - reference
    tracking_rmse = float(np.sqrt(np.mean(np.square(tracking_error))))
    oracle_rmse = float(np.sqrt(np.mean(np.square(oracle_tracking_error))))
    mismatch_rmse = float(np.sqrt(np.mean(np.square(ball - ball_oracle))))
    relative = max(0.0, tracking_rmse - oracle_rmse) / max(
        oracle_rmse,
        RELATIVE_DENOMINATOR_FLOOR_M,
    )
    period_samples = int(rollout["steps_per_period"])
    fitted_amplitude, fitted_phase = _sinusoid_fit(ball, period_samples)
    _, reference_phase = _sinusoid_fit(reference, period_samples)
    phase_lag = math.degrees(math.atan2(
        math.sin(fitted_phase - reference_phase),
        math.cos(fitted_phase - reference_phase),
    ))
    return SimulationMetrics(
        tracking_rmse_m=tracking_rmse,
        tracking_mae_m=float(np.mean(np.abs(tracking_error))),
        mismatch_rmse_m=mismatch_rmse,
        relative_degradation=float(relative),
        amplitude_ratio=fitted_amplitude / max(amplitude_m, 1.0e-12),
        phase_lag_deg=phase_lag,
        saturation_rate=float(rollout["saturation_rate"]),
    )


def adequacy_class(delay_metrics: SimulationMetrics, mean_metrics: SimulationMetrics, oracle_rmse: float) -> str:
    def adequate(metrics: SimulationMetrics) -> bool:
        return (
            metrics.mismatch_rmse_m <= ABSOLUTE_ADEQUACY_M
            and metrics.relative_degradation <= RELATIVE_ADEQUACY
        )

    if adequate(delay_metrics):
        return "model_insensitive"
    if adequate(mean_metrics):
        return "structure_needed"
    if oracle_rmse <= ABSOLUTE_ADEQUACY_M:
        return "parameter_accuracy_needed"
    return "control_limited"


def reduced_closed_loop_grid(
    statistics: ParameterStatistics,
    periods_s: Sequence[float],
    alpha_grid: Sequence[float],
    *,
    sample_period_s: float,
    center_m: float,
    amplitude_m: float,
    max_vertices: int | None = None,
) -> list[dict[str, float | str]]:
    """Evaluate worst-case paired metrics over every empirical-box vertex."""

    nominal = statistics.mean
    rows: list[dict[str, float | str]] = []
    for period_s in periods_s:
        for alpha in alpha_grid:
            vertices = uncertainty_vertices(statistics, float(alpha))
            if max_vertices is not None and vertices.shape[0] > max_vertices:
                selected = np.linspace(0, vertices.shape[0] - 1, max_vertices, dtype=int)
                vertices = vertices[selected]
            delay_results: list[SimulationMetrics] = []
            mean_results: list[SimulationMetrics] = []
            oracle_rmses: list[float] = []
            for vertex in vertices:
                true_parameters = ResponseParameters.from_vector(vertex)
                oracle = simulate_reduced_closed_loop(
                    true_parameters,
                    true_parameters,
                    response_aware=True,
                    period_s=float(period_s),
                    sample_period_s=sample_period_s,
                    center_m=center_m,
                    amplitude_m=amplitude_m,
                )
                mean_rollout = simulate_reduced_closed_loop(
                    true_parameters,
                    nominal,
                    response_aware=True,
                    period_s=float(period_s),
                    sample_period_s=sample_period_s,
                    center_m=center_m,
                    amplitude_m=amplitude_m,
                )
                delay_rollout = simulate_reduced_closed_loop(
                    true_parameters,
                    nominal,
                    response_aware=False,
                    period_s=float(period_s),
                    sample_period_s=sample_period_s,
                    center_m=center_m,
                    amplitude_m=amplitude_m,
                )
                oracle_metrics = paired_metrics(
                    oracle,
                    oracle,
                    amplitude_m=amplitude_m,
                )
                mean_metrics = paired_metrics(
                    mean_rollout,
                    oracle,
                    amplitude_m=amplitude_m,
                )
                delay_metrics = paired_metrics(
                    delay_rollout,
                    oracle,
                    amplitude_m=amplitude_m,
                )
                oracle_rmses.append(oracle_metrics.tracking_rmse_m)
                mean_results.append(mean_metrics)
                delay_results.append(delay_metrics)

            def envelope(results: Sequence[SimulationMetrics]) -> SimulationMetrics:
                return SimulationMetrics(
                    tracking_rmse_m=max(result.tracking_rmse_m for result in results),
                    tracking_mae_m=max(result.tracking_mae_m for result in results),
                    mismatch_rmse_m=max(result.mismatch_rmse_m for result in results),
                    relative_degradation=max(result.relative_degradation for result in results),
                    amplitude_ratio=max(result.amplitude_ratio for result in results),
                    phase_lag_deg=max(abs(result.phase_lag_deg) for result in results),
                    saturation_rate=max(result.saturation_rate for result in results),
                )

            # Each reported column is a worst-case envelope.  The maximizing
            # vertex can differ by metric, which is intentional for a robust
            # adequacy statement and avoids optimistic cross-metric pairing.
            worst_mean = envelope(mean_results)
            worst_delay = envelope(delay_results)
            worst_oracle = max(oracle_rmses)
            rows.append(
                {
                    "period_s": float(period_s),
                    "omega_rad_s": 2.0 * math.pi / float(period_s),
                    "alpha": float(alpha),
                    "vertex_count": float(vertices.shape[0]),
                    "delay_tracking_rmse_m": worst_delay.tracking_rmse_m,
                    "delay_mismatch_rmse_m": worst_delay.mismatch_rmse_m,
                    "delay_relative_degradation": worst_delay.relative_degradation,
                    "delay_saturation_rate": worst_delay.saturation_rate,
                    "mean_tracking_rmse_m": worst_mean.tracking_rmse_m,
                    "mean_mismatch_rmse_m": worst_mean.mismatch_rmse_m,
                    "mean_relative_degradation": worst_mean.relative_degradation,
                    "mean_saturation_rate": worst_mean.saturation_rate,
                    "oracle_tracking_rmse_m": worst_oracle,
                    "local_mean_valid": float(worst_mean.saturation_rate <= 0.05),
                    "local_delay_valid": float(worst_delay.saturation_rate <= 0.05),
                    "adequacy_class": adequacy_class(worst_delay, worst_mean, worst_oracle),
                }
            )
    return rows


def critical_alpha_rows(grid_rows: Sequence[dict[str, float | str]]) -> list[dict[str, float]]:
    """Interpolate the two adequacy crossings on the sampled nested boxes."""

    output: list[dict[str, float]] = []
    periods = sorted({float(row["period_s"]) for row in grid_rows})
    for period in periods:
        rows = sorted(
            (row for row in grid_rows if float(row["period_s"]) == period),
            key=lambda row: float(row["alpha"]),
        )

        def crossing(field: str, threshold: float) -> tuple[float, float, float, float]:
            previous_alpha = float(rows[0]["alpha"])
            previous_value = float(rows[0][field])
            if previous_value > threshold:
                return 0.0, 1.0, 0.0, 0.0
            for row in rows[1:]:
                alpha = float(row["alpha"])
                value = float(row[field])
                if value > threshold:
                    if value == previous_value:
                        estimate = previous_alpha
                    else:
                        fraction = (threshold - previous_value) / (value - previous_value)
                        estimate = previous_alpha + fraction * (alpha - previous_alpha)
                    return estimate, 1.0, previous_alpha, alpha
                previous_alpha = alpha
                previous_value = value
            final_alpha = float(rows[-1]["alpha"])
            return final_alpha, 0.0, final_alpha, final_alpha

        absolute = crossing("mean_mismatch_rmse_m", ABSOLUTE_ADEQUACY_M)
        relative = crossing("mean_relative_degradation", RELATIVE_ADEQUACY)

        output.append(
            {
                "period_s": period,
                "omega_rad_s": 2.0 * math.pi / period,
                "alpha_crit_10mm": absolute[0],
                "alpha_crit_10mm_crossed": absolute[1],
                "alpha_crit_10mm_bracket_low": absolute[2],
                "alpha_crit_10mm_bracket_high": absolute[3],
                "alpha_crit_10percent": relative[0],
                "alpha_crit_10percent_crossed": relative[1],
                "alpha_crit_10percent_bracket_low": relative[2],
                "alpha_crit_10percent_bracket_high": relative[3],
            }
        )
    return output


def linearity_check_rows(
    periodic_rows: Sequence[dict[str, float | str | None]],
    grid_rows: Sequence[dict[str, float | str]],
) -> list[dict[str, float]]:
    """Pair local first-order RMS with the nonlinear reduced-model envelope."""

    grid_by_key = {
        (float(row["period_s"]), float(row["alpha"])): row for row in grid_rows
    }
    output: list[dict[str, float]] = []
    for row in periodic_rows:
        key = (float(row["period_s"]), float(row["alpha"]))
        grid = grid_by_key.get(key)
        linear_value = row["linear_paired_rms_m"]
        if grid is None or linear_value is None:
            continue
        nonlinear_value = float(grid["mean_mismatch_rmse_m"])
        absolute_difference = abs(float(linear_value) - nonlinear_value)
        if nonlinear_value <= 1.0e-12 and float(linear_value) <= 1.0e-12:
            relative_difference = 0.0
        else:
            relative_difference = absolute_difference / max(nonlinear_value, 1.0e-12)
        accepted = bool(
            float(row["local_ltv_valid"]) > 0.5
            and float(grid["local_mean_valid"]) > 0.5
            and relative_difference <= 0.15
        )
        output.append(
            {
                "period_s": key[0],
                "alpha": key[1],
                "linear_paired_rms_m": float(linear_value),
                "nonlinear_paired_rms_m": nonlinear_value,
                "relative_difference": relative_difference,
                "local_linearity_accepted": float(accepted),
            }
        )
    return output


def _mean_oracle_envelope_at_alpha(
    statistics: ParameterStatistics,
    period_s: float,
    alpha: float,
    *,
    sample_period_s: float,
    center_m: float,
    amplitude_m: float,
) -> dict[str, float]:
    """Evaluate the two paired adequacy metrics without the delay-only branch."""

    mean_metrics: list[SimulationMetrics] = []
    oracle_metrics: list[SimulationMetrics] = []
    for vertex in uncertainty_vertices(statistics, alpha):
        true_parameters = ResponseParameters.from_vector(vertex)
        oracle = simulate_reduced_closed_loop(
            true_parameters,
            true_parameters,
            response_aware=True,
            period_s=period_s,
            sample_period_s=sample_period_s,
            center_m=center_m,
            amplitude_m=amplitude_m,
        )
        mean_rollout = simulate_reduced_closed_loop(
            true_parameters,
            statistics.mean,
            response_aware=True,
            period_s=period_s,
            sample_period_s=sample_period_s,
            center_m=center_m,
            amplitude_m=amplitude_m,
        )
        oracle_metrics.append(paired_metrics(oracle, oracle, amplitude_m=amplitude_m))
        mean_metrics.append(paired_metrics(mean_rollout, oracle, amplitude_m=amplitude_m))
    return {
        "mean_mismatch_rmse_m": max(value.mismatch_rmse_m for value in mean_metrics),
        "mean_relative_degradation": max(value.relative_degradation for value in mean_metrics),
        "mean_saturation_rate": max(value.saturation_rate for value in mean_metrics),
        "oracle_tracking_rmse_m": max(value.tracking_rmse_m for value in oracle_metrics),
        "oracle_saturation_rate": max(value.saturation_rate for value in oracle_metrics),
    }


def joint_trial_closed_loop_rows(
    statistics: ParameterStatistics,
    trial_parameters: np.ndarray,
    periods_s: Sequence[float],
    box_grid_rows: Sequence[dict[str, float | str]],
    *,
    sample_period_s: float,
    center_m: float,
    amplitude_m: float,
) -> list[dict[str, float]]:
    """Check rectangular-box conservatism against the 26 observed joint fits."""

    alpha_one_box = {
        float(row["period_s"]): row
        for row in box_grid_rows
        if math.isclose(float(row["alpha"]), 1.0)
    }
    rows: list[dict[str, float]] = []
    for period_s in periods_s:
        mean_metrics: list[SimulationMetrics] = []
        oracle_metrics: list[SimulationMetrics] = []
        for values in np.asarray(trial_parameters, dtype=np.float64):
            true_parameters = ResponseParameters.from_vector(values)
            oracle = simulate_reduced_closed_loop(
                true_parameters,
                true_parameters,
                response_aware=True,
                period_s=float(period_s),
                sample_period_s=sample_period_s,
                center_m=center_m,
                amplitude_m=amplitude_m,
            )
            mean_rollout = simulate_reduced_closed_loop(
                true_parameters,
                statistics.mean,
                response_aware=True,
                period_s=float(period_s),
                sample_period_s=sample_period_s,
                center_m=center_m,
                amplitude_m=amplitude_m,
            )
            oracle_metrics.append(paired_metrics(oracle, oracle, amplitude_m=amplitude_m))
            mean_metrics.append(paired_metrics(mean_rollout, oracle, amplitude_m=amplitude_m))
        joint_mismatch = max(value.mismatch_rmse_m for value in mean_metrics)
        box_row = alpha_one_box[float(period_s)]
        box_mismatch = float(box_row["mean_mismatch_rmse_m"])
        rows.append(
            {
                "period_s": float(period_s),
                "joint_trial_count": float(len(mean_metrics)),
                "joint_mismatch_rmse_m": joint_mismatch,
                "joint_relative_degradation": max(
                    value.relative_degradation for value in mean_metrics
                ),
                "joint_mean_saturation_rate": max(
                    value.saturation_rate for value in mean_metrics
                ),
                "joint_oracle_tracking_rmse_m": max(
                    value.tracking_rmse_m for value in oracle_metrics
                ),
                "box_mismatch_rmse_m": box_mismatch,
                "box_to_joint_mismatch_ratio": box_mismatch / max(joint_mismatch, 1.0e-12),
            }
        )
    return rows


def locality_check_rows(
    statistics: ParameterStatistics,
    *,
    sample_period_s: float,
) -> list[dict[str, float | str]]:
    """Check amplitude and center dependence at representative periods/boxes."""

    cases = (
        ("baseline", 0.35, 0.15),
        ("amplitude_0p10", 0.35, 0.10),
        ("amplitude_0p20", 0.35, 0.20),
        ("center_0p20", 0.20, 0.15),
        ("center_0p50", 0.50, 0.15),
    )
    rows: list[dict[str, float | str]] = []
    for label, center_m, amplitude_m in cases:
        for period_s in (4.0, 8.0):
            for alpha in (0.5, 1.0):
                metrics = _mean_oracle_envelope_at_alpha(
                    statistics,
                    period_s,
                    alpha,
                    sample_period_s=sample_period_s,
                    center_m=center_m,
                    amplitude_m=amplitude_m,
                )
                rows.append(
                    {
                        "case": label,
                        "center_m": center_m,
                        "amplitude_m": amplitude_m,
                        "period_s": period_s,
                        "alpha": alpha,
                        **metrics,
                        "local_mean_valid": float(metrics["mean_saturation_rate"] <= 0.05),
                    }
                )
    return rows


def refine_critical_alpha_rows(
    statistics: ParameterStatistics,
    coarse_rows: Sequence[dict[str, float]],
    *,
    sample_period_s: float,
    center_m: float,
    amplitude_m: float,
    tolerance: float = 1.0e-3,
) -> list[dict[str, float]]:
    """Bisect nonlinear adequacy crossings inside coarse-grid brackets."""

    if tolerance <= 0.0:
        raise ValueError("The critical-alpha bisection tolerance must be positive.")
    cache: dict[tuple[float, float], dict[str, float]] = {}

    def evaluate(period_s: float, alpha: float) -> dict[str, float]:
        key = (period_s, alpha)
        if key not in cache:
            cache[key] = _mean_oracle_envelope_at_alpha(
                statistics,
                period_s,
                alpha,
                sample_period_s=sample_period_s,
                center_m=center_m,
                amplitude_m=amplitude_m,
            )
        return cache[key]

    output: list[dict[str, float]] = []
    for coarse in coarse_rows:
        period_s = float(coarse["period_s"])
        result: dict[str, float] = {
            "period_s": period_s,
            "omega_rad_s": 2.0 * math.pi / period_s,
        }
        for suffix, field, threshold in (
            ("10mm", "mean_mismatch_rmse_m", ABSOLUTE_ADEQUACY_M),
            ("10percent", "mean_relative_degradation", RELATIVE_ADEQUACY),
        ):
            crossed = bool(coarse[f"alpha_crit_{suffix}_crossed"])
            if not crossed:
                alpha = float(coarse[f"alpha_crit_{suffix}"])
                metrics = evaluate(period_s, alpha)
                result[f"alpha_crit_{suffix}"] = alpha
                result[f"alpha_crit_{suffix}_crossed"] = 0.0
                result[f"alpha_crit_{suffix}_bracket_low"] = alpha
                result[f"alpha_crit_{suffix}_bracket_high"] = alpha
                result[f"alpha_crit_{suffix}_local_valid"] = float(
                    metrics["mean_saturation_rate"] <= 0.05
                    and metrics["oracle_saturation_rate"] <= 0.05
                )
                continue

            low = float(coarse[f"alpha_crit_{suffix}_bracket_low"])
            high = float(coarse[f"alpha_crit_{suffix}_bracket_high"])
            low_metrics = evaluate(period_s, low)
            high_metrics = evaluate(period_s, high)
            if low_metrics[field] > threshold or high_metrics[field] <= threshold:
                raise ValueError(
                    f"Invalid {suffix} bracket [{low}, {high}] at T={period_s}: "
                    f"values {low_metrics[field]} and {high_metrics[field]}."
                )
            while high - low > tolerance:
                midpoint = 0.5 * (low + high)
                midpoint_metrics = evaluate(period_s, midpoint)
                if midpoint_metrics[field] <= threshold:
                    low = midpoint
                    low_metrics = midpoint_metrics
                else:
                    high = midpoint
                    high_metrics = midpoint_metrics
            result[f"alpha_crit_{suffix}"] = 0.5 * (low + high)
            result[f"alpha_crit_{suffix}_crossed"] = 1.0
            result[f"alpha_crit_{suffix}_bracket_low"] = low
            result[f"alpha_crit_{suffix}_bracket_high"] = high
            result[f"alpha_crit_{suffix}_local_valid"] = float(
                max(
                    low_metrics["mean_saturation_rate"],
                    high_metrics["mean_saturation_rate"],
                    low_metrics["oracle_saturation_rate"],
                    high_metrics["oracle_saturation_rate"],
                )
                <= 0.05
            )
        output.append(result)
    return output


def _write_csv(path: Path, rows: Sequence[dict]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV {path}.")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _json_ready(value):
    if isinstance(value, ResponseParameters):
        return asdict(value)
    if isinstance(value, ParameterStatistics):
        return {key: _json_ready(item) for key, item in asdict(value).items()}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _read_single_summary(path: Path) -> dict[str, str]:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    if len(rows) != 1:
        raise ValueError(f"Expected exactly one result row in {path}, got {len(rows)}.")
    return rows[0]


def existing_isaac_crosscheck_rows(project_root: Path) -> list[dict[str, float | str]]:
    """Load pre-existing, independently generated Isaac cross-checks.

    These legacy runs use their archived input configurations and are not used
    to fit the new PRBS-envelope boundary.  Keeping them in a separate table
    prevents accidental presentation as paired samples from the reduced model.
    """

    root = project_root / "logs" / "acados_nmpc_unified_tracking"
    specifications = (
        (
            "structure_ablation",
            "response-aware NMPC",
            "standard response; delay-free",
            "mixed constant/B-spline/ramp-dwell",
            "acados_nmpc_unified_mixed_standard_vel_response_gauss_delay_free",
        ),
        (
            "structure_ablation",
            "response-unaware NMPC",
            "standard response; delay-free",
            "mixed constant/B-spline/ramp-dwell",
            "acados_nmpc_wo_response_unified_mixed_standard_vel_response_gauss_delay_free",
        ),
        (
            "bandwidth_stress",
            "standard uncertainty",
            "tau=[0.15,0.16], K=[0.84,0.88], b=[-0.004,-0.003]",
            "sine/triangle/trapezoid, T=4--6 s",
            "acados_nmpc_unified_stt_4-6_standard_vel_response_gauss_delay",
        ),
        (
            "bandwidth_stress",
            "extended uncertainty",
            "tau=[0.10,0.21], K=[0.72,1.00], b=[-0.010,0.003]",
            "sine/triangle/trapezoid, T=4--6 s",
            "acados_nmpc_unified_stt_4-6_extended_vel_response_gauss_delay",
        ),
        (
            "bandwidth_stress",
            "standard uncertainty",
            "tau=[0.15,0.16], K=[0.84,0.88], b=[-0.004,-0.003]",
            "sine/triangle/trapezoid, T=8--10 s",
            "acados_nmpc_unified_stt_8-10_standard_vel_response_gauss_delay",
        ),
        (
            "bandwidth_stress",
            "extended uncertainty",
            "tau=[0.10,0.21], K=[0.72,1.00], b=[-0.010,0.003]",
            "sine/triangle/trapezoid, T=8--10 s",
            "acados_nmpc_unified_stt_8-10_extended_vel_response_gauss_delay",
        ),
    )
    rows: list[dict[str, float | str]] = []
    for study, condition, domain, reference, run_name in specifications:
        summary_path = root / run_name / "summary.csv"
        if not summary_path.exists():
            continue
        summary = _read_single_summary(summary_path)
        rows.append(
            {
                "study": study,
                "condition": condition,
                "parameter_domain": domain,
                "reference_group": reference,
                "run_name": run_name,
                "episodes": float(summary["benchmark_completed_episodes"]),
                "mae_m": float(summary["benchmark_mean_absolute_error"]),
                "rmse_m": float(summary["benchmark_root_mean_square_error"]),
                "solver_success_rate": float(summary["solver_success_rate"]),
                "fallback_rate": float(summary["fallback_rate"]),
            }
        )
    return rows


def _format_boundary(row: dict[str, float], field: str) -> str:
    if not bool(row[f"{field}_crossed"]):
        return f">{row[field]:.2f}"
    low = row[f"{field}_bracket_low"]
    high = row[f"{field}_bracket_high"]
    if high > low and math.isclose(row[field], low, abs_tol=1.0e-9):
        return f"[{low:.2f},{high:.2f})"
    return f"{row[field]:.3f}"


def write_latex_table_data(
    path: Path,
    statistics: ParameterStatistics,
    dc_rows: Sequence[dict[str, float]],
    frequency_rows: Sequence[dict[str, float]],
    predictor_rows: Sequence[dict[str, float]],
    grid_rows: Sequence[dict[str, float | str]],
    critical_rows: Sequence[dict[str, float]],
    periodic_rows: Sequence[dict[str, float | str | None]],
    linearity_rows: Sequence[dict[str, float]],
    isaac_rows: Sequence[dict[str, float | str]],
) -> None:
    """Emit publication-ready row macros without making the report depend on them."""

    lines = [
        "% Generated by scripts/analyze_task_dependent_model_adequacy.py.",
        "% The standalone technical appendix embeds the same values directly.",
        rf"\providecommand{{\AdequacyTrialCount}}{{{statistics.count}}}",
        rf"\providecommand{{\AdequacyMeanDelay}}{{{statistics.mean.delay_s:.7f}}}",
        rf"\providecommand{{\AdequacyMeanTau}}{{{statistics.mean.tau_s:.7f}}}",
        rf"\providecommand{{\AdequacyMeanGain}}{{{statistics.mean.gain:.7f}}}",
        rf"\providecommand{{\AdequacyMeanBias}}{{{statistics.mean.bias_mps:.7f}}}",
    ]
    alpha_one_dc = next(row for row in dc_rows if math.isclose(row["alpha"], 1.0))
    lines.append(
        rf"\providecommand{{\AdequacyDCOffsetMaxMmps}}"
        rf"{{{1000.0 * alpha_one_dc['velocity_offset_abs_max_mps']:.3f}}}"
    )

    alpha_one_frequency = sorted(
        (row for row in frequency_rows if math.isclose(row["alpha"], 1.0)),
        key=lambda row: row["period_s"],
    )
    lines.append(r"\providecommand{\AdequacyFrequencyRows}{%")
    for row in alpha_one_frequency:
        lines.append(
            f"{row['period_s']:.0f} & {row['omega_rad_s']:.3f} & "
            f"{100.0 * row['parameter_mismatch_relative_max']:.2f} & "
            f"{row['structure_to_parameter_ratio']:.2f} & "
            f"{row['delay_quantization_phase_deg']:.3f} \\\\"
        )
    lines.append("}")

    alpha_one_predictor = sorted(
        (row for row in predictor_rows if math.isclose(row["alpha"], 1.0)),
        key=lambda row: row["period_s"],
    )
    lines.append(r"\providecommand{\AdequacyPredictorRows}{%")
    for row in alpha_one_predictor:
        lines.append(
            f"{row['period_s']:.0f} & {1000.0 * row['pb_error_max_m']:.4f} & "
            f"{row['vrz_error_max_mps']:.5f} & "
            f"{math.degrees(row['theta_error_max_rad']):.4f} \\\\"
        )
    lines.append("}")

    if grid_rows:
        alpha_one_grid = sorted(
            (row for row in grid_rows if math.isclose(float(row["alpha"]), 1.0)),
            key=lambda row: float(row["period_s"]),
        )
        lines.append(r"\providecommand{\AdequacyReducedRows}{%")
        for row in alpha_one_grid:
            if float(row["local_mean_valid"]) < 0.5:
                lines.append(
                    f"{float(row['period_s']):.0f} & -- & -- & "
                    f"{1000.0 * float(row['oracle_tracking_rmse_m']):.3f} & invalid \\\\"
                )
                continue
            else:
                status = "valid"
            lines.append(
                f"{float(row['period_s']):.0f} & "
                f"{1000.0 * float(row['mean_mismatch_rmse_m']):.3f} & "
                f"{100.0 * float(row['mean_relative_degradation']):.1f} & "
                f"{1000.0 * float(row['oracle_tracking_rmse_m']):.3f} & {status} \\\\"
            )
        lines.append("}")

    if critical_rows:
        lines.append(r"\providecommand{\AdequacyBoundaryRows}{%")
        for row in sorted(critical_rows, key=lambda item: item["period_s"]):
            lines.append(
                f"{row['period_s']:.0f} & "
                f"{_format_boundary(row, 'alpha_crit_10mm')} & "
                f"{_format_boundary(row, 'alpha_crit_10percent')} \\\\"
            )
        lines.append("}")

    accepted_linearity = [
        row for row in linearity_rows
        if math.isclose(float(row["alpha"]), 0.5)
        and float(row["local_linearity_accepted"]) > 0.5
    ]
    if accepted_linearity:
        lines.append(r"\providecommand{\AdequacyPeriodicRows}{%")
        periodic_by_period = {
            float(row["period_s"]): row
            for row in periodic_rows
            if math.isclose(float(row["alpha"]), 0.5)
        }
        for row in sorted(accepted_linearity, key=lambda item: float(item["period_s"])):
            periodic = periodic_by_period[float(row["period_s"])]
            lines.append(
                f"{float(row['period_s']):.0f} & "
                f"{1000.0 * float(row['linear_paired_rms_m']):.3f} & "
                f"{1000.0 * float(row['nonlinear_paired_rms_m']):.3f} & "
                f"{100.0 * float(row['relative_difference']):.2f} & "
                f"{float(periodic['monodromy_spectral_radius']):.4f} \\\\"
            )
        lines.append("}")

    if isaac_rows:
        lines.append(r"\providecommand{\AdequacyIsaacRows}{%")
        for row in isaac_rows:
            label = str(row["condition"]).replace("_", r"\_")
            lines.append(
                f"{label} & {1000.0 * float(row['mae_m']):.2f} & "
                f"{1000.0 * float(row['rmse_m']):.2f} & "
                f"{100.0 * float(row['solver_success_rate']):.3f} \\\\"
            )
        lines.append("}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot_results(
    output_dir: Path,
    frequency_rows: Sequence[dict[str, float]],
    grid_rows: Sequence[dict[str, float | str]],
    critical_rows: Sequence[dict[str, float]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    alpha_one = [row for row in frequency_rows if math.isclose(row["alpha"], 1.0)]
    alpha_one.sort(key=lambda row: row["omega_rad_s"])
    figure, axis = plt.subplots(figsize=(5.2, 3.2))
    axis.plot(
        [row["omega_rad_s"] for row in alpha_one],
        [row["parameter_mismatch_relative_max"] * 100.0 for row in alpha_one],
        "o-",
        label="Mean FOPDT: parameter mismatch",
    )
    axis.plot(
        [row["omega_rad_s"] for row in alpha_one],
        [row["structure_mismatch_abs_max"] * 100.0 for row in alpha_one],
        "s-",
        label="Delay-only: structure mismatch",
    )
    axis.set_xlabel(r"Reference frequency $\omega$ (rad/s)")
    axis.set_ylabel("Worst-case response mismatch (%)")
    axis.grid(True, alpha=0.3)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output_dir / "response_mismatch_vs_frequency.pdf")
    plt.close(figure)

    periods = sorted({float(row["period_s"]) for row in grid_rows})
    alphas = sorted({float(row["alpha"]) for row in grid_rows})
    matrix = np.full((len(alphas), len(periods)), np.nan)
    for row in grid_rows:
        matrix[alphas.index(float(row["alpha"])), periods.index(float(row["period_s"]))] = (
            1000.0 * float(row["mean_mismatch_rmse_m"])
        )
    figure, axis = plt.subplots(figsize=(5.4, 3.4))
    image = axis.imshow(matrix, origin="lower", aspect="auto", cmap="viridis")
    axis.set_xticks(range(len(periods)), [f"{value:g}" for value in periods])
    axis.set_yticks(range(len(alphas)), [f"{value:g}" for value in alphas])
    axis.set_xlabel("Reference period T (s)")
    axis.set_ylabel(r"Uncertainty scale $\alpha$")
    colorbar = figure.colorbar(image, ax=axis)
    colorbar.set_label("Worst-case Mean-FOPDT mismatch RMS (mm)")
    figure.tight_layout()
    figure.savefig(output_dir / "bandwidth_uncertainty_heatmap.pdf")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(5.2, 3.2))
    ordered = sorted(critical_rows, key=lambda row: row["omega_rad_s"])
    axis.plot(
        [row["omega_rad_s"] for row in ordered],
        [row["alpha_crit_10mm"] for row in ordered],
        "o-",
        label="10 mm paired mismatch",
    )
    axis.plot(
        [row["omega_rad_s"] for row in ordered],
        [row["alpha_crit_10percent"] for row in ordered],
        "s-",
        label="10% tracking degradation",
    )
    axis.set_xlabel(r"Reference frequency $\omega$ (rad/s)")
    axis.set_ylabel(r"Critical uncertainty scale $\alpha_{\rm crit}$")
    axis.set_ylim(bottom=0.0)
    axis.grid(True, alpha=0.3)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output_dir / "adequacy_boundary.pdf")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--identification-dir",
        type=Path,
        default=Path("logs/velocity_response_identification/prbs_velocity_response_fit"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/tmp/task_dependent_model_adequacy"),
    )
    parser.add_argument("--sample-rate-hz", type=float, default=60.0)
    parser.add_argument("--center-m", type=float, default=0.35)
    parser.add_argument("--amplitude-m", type=float, default=0.15)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use benchmark periods, alpha={0,1,2}, and four box vertices for a smoke run.",
    )
    parser.add_argument(
        "--skip-reduced-grid",
        action="store_true",
        help="Only compute identification and exact frequency-domain results.",
    )
    parser.add_argument(
        "--reuse-reduced-grid",
        type=Path,
        default=None,
        help="Reuse reduced_closed_loop_grid.csv from a prior identical run.",
    )
    parser.add_argument(
        "--skip-periodic-ltv",
        action="store_true",
        help="Skip finite-difference monodromy and periodic sensitivity calculations.",
    )
    parser.add_argument(
        "--refine-critical-boundaries",
        action="store_true",
        help="Bisect nonlinear critical-alpha crossings to the requested tolerance.",
    )
    parser.add_argument("--critical-alpha-tolerance", type=float, default=1.0e-3)
    parser.add_argument(
        "--evaluate-joint-trials",
        action="store_true",
        help="Run the reduced closed loop at all 26 observed joint parameter points.",
    )
    parser.add_argument(
        "--evaluate-locality-checks",
        action="store_true",
        help="Evaluate the planned 0.10/0.20 m amplitude and 0.20/0.50 m center checks.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.sample_rate_hz <= 0.0 or args.amplitude_m <= 0.0:
        raise ValueError("sample rate and amplitude must be positive.")
    if args.skip_reduced_grid and args.reuse_reduced_grid is not None:
        raise ValueError("--skip-reduced-grid and --reuse-reduced-grid are mutually exclusive.")
    sample_period_s = 1.0 / args.sample_rate_hz
    trial_csv = args.identification_dir / "fopdt_trial_parameters.csv"
    overall_csv = args.identification_dir / "overall_fits.csv"
    statistics, trial_parameters = load_identification_statistics(
        trial_csv,
        sample_period_s=sample_period_s,
        overall_csv=overall_csv,
    )
    periods = (4.0, 6.0, 8.0, 10.0) if args.quick else PERIODS_S
    alpha_grid = (0.0, 1.0, 2.0) if args.quick else ALPHA_GRID
    frequency_rows = frequency_envelope(
        statistics,
        periods,
        alpha_grid,
        sample_period_s=sample_period_s,
    )
    joint_frequency_rows = joint_trial_frequency_envelope(
        statistics,
        trial_parameters,
        periods,
    )
    dc_rows = dc_offset_envelope(statistics, alpha_grid)
    predictor_rows = predictor_boundary_envelope(
        statistics,
        periods,
        alpha_grid,
        sample_period_s=sample_period_s,
    )
    periodic_rows: list[dict[str, float | str | None]] = []
    if not args.skip_periodic_ltv:
        periodic_rows = periodic_composite_sensitivity_rows(
            statistics,
            periods,
            alpha_grid,
            sample_period_s=sample_period_s,
            center_m=args.center_m,
            amplitude_m=args.amplitude_m,
        )
    grid_rows: list[dict[str, float | str]] = []
    critical_rows: list[dict[str, float]] = []
    refined_critical_rows: list[dict[str, float]] = []
    linearity_rows: list[dict[str, float]] = []
    joint_closed_loop_rows: list[dict[str, float]] = []
    locality_rows: list[dict[str, float | str]] = []
    if args.reuse_reduced_grid is not None:
        reused_path = args.reuse_reduced_grid
        if reused_path.is_dir():
            reused_path = reused_path / "reduced_closed_loop_grid.csv"
        grid_rows = list(csv.DictReader(reused_path.open(encoding="utf-8")))
        if not grid_rows:
            raise ValueError(f"No reduced-grid rows found in {reused_path}.")
        expected_keys = {(period, alpha) for period in periods for alpha in alpha_grid}
        actual_keys = {
            (float(row["period_s"]), float(row["alpha"])) for row in grid_rows
        }
        if actual_keys != expected_keys:
            raise ValueError(
                "Reused grid does not match the requested period/alpha grid: "
                f"expected {len(expected_keys)} keys, got {len(actual_keys)}."
            )
        critical_rows = critical_alpha_rows(grid_rows)
        linearity_rows = linearity_check_rows(periodic_rows, grid_rows)
    elif not args.skip_reduced_grid:
        grid_rows = reduced_closed_loop_grid(
            statistics,
            periods,
            alpha_grid,
            sample_period_s=sample_period_s,
            center_m=args.center_m,
            amplitude_m=args.amplitude_m,
            max_vertices=4 if args.quick else None,
        )
        critical_rows = critical_alpha_rows(grid_rows)
        linearity_rows = linearity_check_rows(periodic_rows, grid_rows)
    if args.refine_critical_boundaries:
        if not critical_rows:
            raise ValueError("Critical-boundary refinement requires a reduced grid.")
        refined_critical_rows = refine_critical_alpha_rows(
            statistics,
            critical_rows,
            sample_period_s=sample_period_s,
            center_m=args.center_m,
            amplitude_m=args.amplitude_m,
            tolerance=args.critical_alpha_tolerance,
        )
    if args.evaluate_joint_trials:
        if not grid_rows:
            raise ValueError("Joint-trial reduced evaluation requires a reduced grid.")
        joint_closed_loop_rows = joint_trial_closed_loop_rows(
            statistics,
            trial_parameters,
            periods,
            grid_rows,
            sample_period_s=sample_period_s,
            center_m=args.center_m,
            amplitude_m=args.amplitude_m,
        )
    if args.evaluate_locality_checks:
        locality_rows = locality_check_rows(
            statistics,
            sample_period_s=sample_period_s,
        )
    isaac_rows = existing_isaac_crosscheck_rows(PROJECT_ROOT)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "frequency_envelope.csv", frequency_rows)
    _write_csv(args.output_dir / "joint_trial_frequency_envelope.csv", joint_frequency_rows)
    _write_csv(args.output_dir / "dc_offset_envelope.csv", dc_rows)
    _write_csv(args.output_dir / "predictor_boundary_envelope.csv", predictor_rows)
    if periodic_rows:
        _write_csv(args.output_dir / "periodic_ltv_sensitivity.csv", periodic_rows)
    if grid_rows:
        _write_csv(args.output_dir / "reduced_closed_loop_grid.csv", grid_rows)
        _write_csv(args.output_dir / "critical_alpha.csv", critical_rows)
        if refined_critical_rows:
            _write_csv(args.output_dir / "critical_alpha_refined.csv", refined_critical_rows)
        if linearity_rows:
            _write_csv(args.output_dir / "linearity_check.csv", linearity_rows)
        if joint_closed_loop_rows:
            _write_csv(args.output_dir / "joint_trial_closed_loop.csv", joint_closed_loop_rows)
        if locality_rows:
            _write_csv(args.output_dir / "locality_checks.csv", locality_rows)
        _plot_results(args.output_dir, frequency_rows, grid_rows, critical_rows)
    if isaac_rows:
        _write_csv(args.output_dir / "existing_isaac_crosschecks.csv", isaac_rows)
    write_latex_table_data(
        args.output_dir / "latex_table_data.tex",
        statistics,
        dc_rows,
        frequency_rows,
        predictor_rows,
        grid_rows,
        refined_critical_rows or critical_rows,
        periodic_rows,
        linearity_rows,
        isaac_rows,
    )
    payload = {
        "analysis_scope": {
            "exact": "FOPDT identities and frequency envelopes",
            "local": "fixed-active-set finite-horizon KKT controller plus forward predictor",
            "not_claimed": (
                "global stability, nonlinear NMPC guarantee, or an Isaac-validated "
                "boundary for the new PRBS parameter box"
            ),
        },
        "sample_period_s": sample_period_s,
        "statistics": statistics,
        "trial_parameters_up_positive": trial_parameters,
        "periods_s": periods,
        "alpha_grid": alpha_grid,
        "absolute_adequacy_m": ABSOLUTE_ADEQUACY_M,
        "relative_adequacy": RELATIVE_ADEQUACY,
        "frequency_rows": frequency_rows,
        "joint_trial_frequency_rows": joint_frequency_rows,
        "dc_offset_rows": dc_rows,
        "predictor_boundary_rows": predictor_rows,
        "periodic_ltv_rows": periodic_rows,
        "reduced_grid_rows": grid_rows,
        "critical_alpha_rows": critical_rows,
        "refined_critical_alpha_rows": refined_critical_rows,
        "linearity_check_rows": linearity_rows,
        "joint_trial_closed_loop_rows": joint_closed_loop_rows,
        "locality_check_rows": locality_rows,
        "existing_isaac_crosschecks": isaac_rows,
    }
    (args.output_dir / "analysis_results.json").write_text(
        json.dumps(_json_ready(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"Wrote task-dependent adequacy results to {args.output_dir}")


if __name__ == "__main__":
    main()
