"""acados OCP core for unified nonlinear reference tracking.

The module deliberately keeps NumPy model helpers importable without CasADi
or acados.  Optional solver dependencies are checked only when a controller is
constructed, so installing this benchmark does not make the legacy baselines
depend on acados.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import ctypes
from dataclasses import asdict, dataclass, field
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time
from typing import Any

import numpy as np


ACADOS_SUPPORTED_VERSION = "0.5.4"
ACADOS_INSTALL_HINT = (
    "AcadosNMPCPolicy requires acados v0.5.4 and its Python template interface. "
    "Build acados with HPIPM, install <ACADOS_SOURCE_DIR>/interfaces/acados_template, "
    "then set ACADOS_SOURCE_DIR and add <ACADOS_SOURCE_DIR>/lib to LD_LIBRARY_PATH. "
    "See docs/acados_unified_nmpc.md for the complete installation procedure."
)

try:  # Optional dependency: keep importing ``aerial_balance_bench.baselines`` safe.
    import casadi as ca
    from acados_template import (
        AcadosModel,
        AcadosOcp,
        AcadosOcpBatchSolver,
        AcadosOcpSolver,
    )
except Exception as exc:  # pragma: no cover - depends on an external installation.
    ca = None
    AcadosModel = None
    AcadosOcp = None
    AcadosOcpBatchSolver = None
    AcadosOcpSolver = None
    _ACADOS_IMPORT_ERROR: Exception | None = exc
else:  # pragma: no cover - exercised by the optional acados test job.
    _ACADOS_IMPORT_ERROR = None

_ACADOS_PREFLIGHT_LIB: Any | None = None


NX = 5
NU = 1
NP = 5
NY = 7
NY_E = 6
NH = 3

STATE_FIELDS = ("pb", "vb", "theta", "vrz", "command_z")
INPUT_FIELDS = ("delta_command_z",)
PARAMETER_FIELDS = ("pg", "vg", "tau_s", "gain", "bias")
STAGE_RESIDUAL_FIELDS = (
    "pb_minus_pg",
    "vb_minus_vg",
    "theta",
    "omega",
    "vrz",
    "command_z_next",
    "delta_command_z",
)
TERMINAL_RESIDUAL_FIELDS = STAGE_RESIDUAL_FIELDS[:-1]
SOFT_CONSTRAINT_FIELDS = ("pb", "theta", "pb_minus_pg")


def acados_available() -> bool:
    """Return whether both CasADi and ``acados_template`` imported successfully."""
    return _ACADOS_IMPORT_ERROR is None


def require_acados() -> None:
    """Raise the policy-facing optional-dependency error."""
    if _ACADOS_IMPORT_ERROR is not None:
        raise ImportError(ACADOS_INSTALL_HINT) from _ACADOS_IMPORT_ERROR

    source_dir = acados_source_directory()
    template_version = acados_template_distribution_version()
    if source_dir is None or not source_dir.is_dir():
        raise RuntimeError(
            "Cannot verify the acados source release. Set ACADOS_SOURCE_DIR to the "
            f"acados v{ACADOS_SUPPORTED_VERSION} Git checkout. The acados_template "
            f"distribution reports {template_version}, but that metadata is not a "
            "reliable acados release identifier."
        )

    module_path = acados_template_module_path()
    expected_template_dir = (source_dir / "interfaces" / "acados_template").resolve()
    if module_path is not None and not _path_is_relative_to(
        module_path, expected_template_dir
    ):
        raise RuntimeError(
            "acados source/Python interface mismatch: ACADOS_SOURCE_DIR points to "
            f"{source_dir}, but acados_template was imported from {module_path}. "
            f"Reinstall it with `python -m pip install -e {expected_template_dir}` "
            "using the runner's Python environment."
        )

    version = acados_source_version()
    match = re.search(r"\d+\.\d+\.\d+", version)
    if match is None:
        raise RuntimeError(
            f"Cannot determine the acados release at {source_dir}. Check out the exact "
            f"v{ACADOS_SUPPORTED_VERSION} tag in a Git clone and keep ACADOS_SOURCE_DIR "
            "pointing to that clone."
        )
    if match.group(0) != ACADOS_SUPPORTED_VERSION:
        raise RuntimeError(
            f"AcadosNMPCPolicy supports acados v{ACADOS_SUPPORTED_VERSION}; detected source "
            f"release {version} at {source_dir}. "
            "Install the supported release to keep generated-code and batch APIs reproducible."
        )
    _validate_acados_installation(source_dir)


def acados_template_distribution_version() -> str:
    """Return Python distribution metadata for diagnostics, never release gating."""
    try:
        return importlib.metadata.version("acados_template")
    except importlib.metadata.PackageNotFoundError:
        pass
    if acados_available():
        module = __import__("acados_template")
        version = getattr(module, "__version__", None)
        if version:
            return str(version)
    return "unavailable"


def acados_template_module_path() -> Path | None:
    """Return the imported template package path when it can be resolved."""
    if not acados_available():
        return None
    module = __import__("acados_template")
    module_file = getattr(module, "__file__", None)
    return Path(module_file).resolve() if module_file else None


def acados_source_directory() -> Path | None:
    """Resolve the acados checkout from the environment or an editable install."""
    configured = os.environ.get("ACADOS_SOURCE_DIR", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()

    module_path = acados_template_module_path()
    if module_path is None:
        return None
    for candidate in module_path.parents:
        template_dir = candidate / "interfaces" / "acados_template"
        if (candidate / ".git").exists() and template_dir.is_dir():
            return candidate.resolve()
    return None


def _git_acados_output(*arguments: str) -> str | None:
    source_dir = acados_source_directory()
    if source_dir is None or not source_dir.is_dir():
        return None
    try:
        result = subprocess.run(
            ["git", "-C", str(source_dir), *arguments],
            check=False,
            capture_output=True,
            text=True,
            timeout=5.0,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    output = result.stdout.strip()
    return output if result.returncode == 0 and output else None


def acados_source_version() -> str:
    """Return the exact Git release tag of the configured acados checkout."""
    tag = _git_acados_output("describe", "--tags", "--exact-match", "HEAD")
    if tag is None:
        return "unknown"
    match = re.fullmatch(r"v?(\d+\.\d+\.\d+)", tag)
    return match.group(1) if match is not None else tag


def acados_source_revision() -> str:
    """Return the source commit used to invalidate generated-code caches."""
    return _git_acados_output("rev-parse", "HEAD") or "unknown"


def _path_is_relative_to(path: Path, directory: Path) -> bool:
    try:
        path.relative_to(directory)
    except ValueError:
        return False
    return True


def _validate_acados_installation(source_dir: Path) -> None:
    """Fail before code generation when C headers/libraries were only built, not installed."""
    header = source_dir / "include" / "acados_c" / "ocp_nlp_interface.h"
    library_candidates = (
        source_dir / "lib" / "libacados.so",
        source_dir / "lib" / "libacados.dylib",
        source_dir / "bin" / "acados.dll",
        source_dir / "bin" / "libacados.dll",
    )
    library = next((path for path in library_candidates if path.is_file()), None)
    if not header.is_file() or library is None:
        raise RuntimeError(
            "The acados C core was configured/built but not installed into "
            f"ACADOS_SOURCE_DIR={source_dir}. Missing "
            f"{'header ' + str(header) if not header.is_file() else 'libacados shared library'}. "
            f"Run `cmake --build {source_dir / 'build'} --target install --parallel`, "
            "then restart the runner."
        )

    global _ACADOS_PREFLIGHT_LIB
    if _ACADOS_PREFLIGHT_LIB is None:
        try:
            _ACADOS_PREFLIGHT_LIB = ctypes.CDLL(str(library))
        except OSError as exc:
            lib_dir = source_dir / "lib"
            raise RuntimeError(
                f"Found {library}, but its dependent libraries could not be loaded: {exc}. "
                f"Export `LD_LIBRARY_PATH={lib_dir}:$LD_LIBRARY_PATH` before starting "
                "Python/Isaac Sim, then rerun the command."
            ) from exc


def installed_acados_version() -> str:
    """Return the acados source release, falling back to diagnostic metadata."""
    if not acados_available():
        return "unavailable"
    source_version = acados_source_version()
    if source_version != "unknown":
        return source_version
    return acados_template_distribution_version()


@dataclass
class AcadosNMPCModelCfg:
    """Physical constants in the paper model."""

    plank_length: float | str = "auto"
    rope_length: float | str = "auto"
    ball_position_offset: float | str = "auto"
    gravity: float | str = "auto"
    ball_mass: float | str = 0.0005
    ball_radius: float | str = "auto"
    ball_inertia_ratio: float = 0.4
    epsilon: float = 1.0e-6


@dataclass
class AcadosNMPCObjectiveCfg:
    """Diagonal nonlinear least-squares weights in residual order."""

    stage_weights: list[float] = field(
        default_factory=lambda: [5.0, 0.5, 0.1, 0.05, 0.05, 0.5, 1.0]
    )
    terminal_weights: list[float] = field(
        default_factory=lambda: [25.0, 2.5, 0.5, 0.25, 0.25, 0.5]
    )


@dataclass
class AcadosNMPCConstraintsCfg:
    """Hard actuator bounds and softened state/tracking bounds."""

    ball_position_min: float = 0.0
    ball_position_max: float = 0.70
    theta_min: float = -math.radians(40.0)
    theta_max: float = math.radians(40.0)
    tracking_error_min: float = -0.5
    tracking_error_max: float = 0.5
    slack_l1: float = 1.0e4
    slack_l2: float = 1.0e4
    max_acc: float | str = "auto"
    max_velocity: float | str = "auto"


@dataclass
class AcadosNMPCSolverCfg:
    """Explicit acados options; no solver-version defaults are relied upon."""

    n_horizon: int = 30
    step_dt: float | str = "auto"
    nlp_solver_type: str = "SQP_RTI"
    qp_solver: str = "PARTIAL_CONDENSING_HPIPM"
    hessian_approx: str = "GAUSS_NEWTON"
    integrator_type: str = "DISCRETE"
    qp_solver_cond_N: int = 10
    qp_solver_warm_start: int = 1
    levenberg_marquardt: float = 0.0
    print_level: int = 0
    num_threads_in_batch_solve: int = 1


@dataclass
class AcadosNMPCControllerCfg:
    """Fully resolved configuration consumed by the OCP controller."""

    model: AcadosNMPCModelCfg = field(default_factory=AcadosNMPCModelCfg)
    objective: AcadosNMPCObjectiveCfg = field(default_factory=AcadosNMPCObjectiveCfg)
    constraints: AcadosNMPCConstraintsCfg = field(default_factory=AcadosNMPCConstraintsCfg)
    solver: AcadosNMPCSolverCfg = field(default_factory=AcadosNMPCSolverCfg)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "AcadosNMPCControllerCfg":
        cfg = cls()
        if not data:
            return cfg
        _strict_update(cfg, data, "acados_nmpc_controller")
        return cfg


@dataclass
class AcadosNMPCStepResult:
    """One synchronized single/batch solve result."""

    action: np.ndarray
    status: np.ndarray
    solver_time: np.ndarray
    qp_time: np.ndarray
    sqp_iterations: np.ndarray
    qp_iterations: np.ndarray
    cost: np.ndarray
    max_slack: np.ndarray
    fallback: np.ndarray
    wall_time: float


def _strict_update(target: Any, values: Mapping[str, Any], section: str) -> None:
    unknown = sorted(key for key in values if not hasattr(target, key))
    if unknown:
        raise ValueError(f"Unknown field(s) in {section}: {', '.join(unknown)}")
    for key, value in values.items():
        current = getattr(target, key)
        if hasattr(current, "__dataclass_fields__"):
            if not isinstance(value, Mapping):
                raise TypeError(f"{section}.{key} must be a mapping.")
            _strict_update(current, value, f"{section}.{key}")
        else:
            setattr(target, key, value)


def validate_controller_cfg(cfg: AcadosNMPCControllerCfg) -> None:
    """Validate a resolved controller config before generating solver code."""
    model = cfg.model
    positive = {
        "model.plank_length": model.plank_length,
        "model.rope_length": model.rope_length,
        "model.gravity": model.gravity,
        "model.ball_mass": model.ball_mass,
        "model.ball_radius": model.ball_radius,
        "model.epsilon": model.epsilon,
        "constraints.max_acc": cfg.constraints.max_acc,
        "solver.step_dt": cfg.solver.step_dt,
    }
    for name, value in positive.items():
        if isinstance(value, str) or not math.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError(f"{name} must be resolved to a finite positive value, got {value!r}.")
    if not math.isfinite(float(model.ball_position_offset)):
        raise ValueError("model.ball_position_offset must be finite.")
    if not math.isfinite(float(model.ball_inertia_ratio)) or float(model.ball_inertia_ratio) < 0.0:
        raise ValueError("model.ball_inertia_ratio must be finite and non-negative.")
    if int(cfg.solver.n_horizon) <= 0:
        raise ValueError("solver.n_horizon must be positive.")
    if not 1 <= int(cfg.solver.qp_solver_cond_N) <= int(cfg.solver.n_horizon):
        raise ValueError("solver.qp_solver_cond_N must lie in [1, n_horizon].")
    if len(cfg.objective.stage_weights) != NY:
        raise ValueError(f"objective.stage_weights must contain {NY} values in residual order.")
    if len(cfg.objective.terminal_weights) != NY_E:
        raise ValueError(f"objective.terminal_weights must contain {NY_E} values in residual order.")
    weights = [*cfg.objective.stage_weights, *cfg.objective.terminal_weights]
    if any(not math.isfinite(float(value)) or float(value) < 0.0 for value in weights):
        raise ValueError("All objective weights must be finite and non-negative.")
    bounds = cfg.constraints
    for lower_name, upper_name in (
        ("ball_position_min", "ball_position_max"),
        ("theta_min", "theta_max"),
        ("tracking_error_min", "tracking_error_max"),
    ):
        lower = float(getattr(bounds, lower_name))
        upper = float(getattr(bounds, upper_name))
        if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
            raise ValueError(f"constraints.{lower_name} must be smaller than {upper_name}.")
    max_velocity = float(bounds.max_velocity)
    if not math.isfinite(max_velocity) or max_velocity < 0.0:
        raise ValueError("constraints.max_velocity must be finite and non-negative.")
    for name in ("slack_l1", "slack_l2"):
        value = float(getattr(bounds, name))
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"constraints.{name} must be finite and non-negative.")


def _safe_denominator_numpy(value: np.ndarray, epsilon: float) -> np.ndarray:
    sign = np.where(value >= 0.0, 1.0, -1.0)
    return np.where(np.abs(value) < epsilon, sign * epsilon, value)


def geometry_numpy(
    theta: np.ndarray,
    vrz: np.ndarray,
    cfg: AcadosNMPCModelCfg,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(beta, omega)`` using the benchmark's protected geometry."""
    theta = np.asarray(theta, dtype=np.float64)
    vrz = np.asarray(vrz, dtype=np.float64)
    epsilon = float(cfg.epsilon)
    sin_beta = (float(cfg.plank_length) / float(cfg.rope_length)) * (1.0 - np.cos(theta))
    beta = np.arcsin(np.clip(sin_beta, -1.0 + epsilon, 1.0 - epsilon))
    denominator = float(cfg.plank_length) * _safe_denominator_numpy(
        np.cos(beta - theta), epsilon
    )
    omega = -vrz * np.cos(beta) / denominator
    return beta, omega


def response_profile_numpy(
    vrz: np.ndarray,
    command_next: np.ndarray,
    tau_s: np.ndarray,
    gain: np.ndarray,
    bias: np.ndarray,
    elapsed_s: float,
    epsilon: float,
) -> np.ndarray:
    """Exact held-command first-order response at a time within one stage."""
    vrz, command_next, tau_s, gain, bias = np.broadcast_arrays(
        *[np.asarray(value, dtype=np.float64) for value in (vrz, command_next, tau_s, gain, bias)]
    )
    target = gain * command_next + bias
    static = tau_s <= epsilon
    tau_safe = np.maximum(tau_s, epsilon)
    dynamic = target + (vrz - target) * np.exp(-float(elapsed_s) / tau_safe)
    return np.where(static, target, dynamic)


def _derivatives_numpy(
    state: np.ndarray,
    vrz: np.ndarray,
    cfg: AcadosNMPCModelCfg,
) -> np.ndarray:
    pb, vb, theta = np.moveaxis(np.asarray(state, dtype=np.float64), -1, 0)
    _, omega = geometry_numpy(theta, vrz, cfg)
    inertia = float(cfg.ball_inertia_ratio) * float(cfg.ball_mass) * float(cfg.ball_radius) ** 2
    effective_mass = inertia / float(cfg.ball_radius) ** 2 + float(cfg.ball_mass)
    gamma = float(cfg.ball_mass) / effective_mass
    ab = gamma * (
        (pb + float(cfg.ball_position_offset) - float(cfg.plank_length)) * omega**2
        - abs(float(cfg.gravity)) * np.sin(theta)
    )
    return np.stack((vb, ab, omega), axis=-1)


def discrete_dynamics_numpy(
    states: np.ndarray,
    actions: np.ndarray,
    stage_parameters: np.ndarray,
    cfg: AcadosNMPCControllerCfg,
) -> np.ndarray:
    """Vectorized exact-response/RK4 discrete dynamics used by tests and tools.

    ``states[..., :]`` follows :data:`STATE_FIELDS`, ``actions[..., :]`` follows
    :data:`INPUT_FIELDS`, and ``stage_parameters[..., :]`` follows
    :data:`PARAMETER_FIELDS`.
    """
    states = np.asarray(states, dtype=np.float64)
    actions = np.asarray(actions, dtype=np.float64)
    parameters = np.asarray(stage_parameters, dtype=np.float64)
    if states.shape[-1] != NX or actions.shape[-1] != NU or parameters.shape[-1] != NP:
        raise ValueError(
            f"Expected final dimensions x/u/p={NX}/{NU}/{NP}, got "
            f"{states.shape}/{actions.shape}/{parameters.shape}."
        )
    lead_shape = np.broadcast_shapes(states.shape[:-1], actions.shape[:-1], parameters.shape[:-1])
    x = np.broadcast_to(states, (*lead_shape, NX))
    u = np.broadcast_to(actions, (*lead_shape, NU))
    p = np.broadcast_to(parameters, (*lead_shape, NP))
    pb_vb_theta = x[..., :3]
    vrz = x[..., 3]
    command_next = x[..., 4] + u[..., 0]
    tau_s, gain, bias = p[..., 2], p[..., 3], p[..., 4]
    dt = float(cfg.solver.step_dt)
    epsilon = float(cfg.model.epsilon)

    vrz_0 = response_profile_numpy(vrz, command_next, tau_s, gain, bias, 0.0, epsilon)
    vrz_half = response_profile_numpy(vrz, command_next, tau_s, gain, bias, 0.5 * dt, epsilon)
    vrz_end = response_profile_numpy(vrz, command_next, tau_s, gain, bias, dt, epsilon)
    k1 = _derivatives_numpy(pb_vb_theta, vrz_0, cfg.model)
    k2 = _derivatives_numpy(pb_vb_theta + 0.5 * dt * k1, vrz_half, cfg.model)
    k3 = _derivatives_numpy(pb_vb_theta + 0.5 * dt * k2, vrz_half, cfg.model)
    k4 = _derivatives_numpy(pb_vb_theta + dt * k3, vrz_end, cfg.model)
    physical_next = pb_vb_theta + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return np.concatenate(
        (physical_next, vrz_end[..., None], command_next[..., None]), axis=-1
    )


def controller_fingerprint(cfg: AcadosNMPCControllerCfg) -> str:
    """Return the deterministic generated-code cache key."""
    payload = {
        "schema": 2,
        "acados_supported_version": ACADOS_SUPPORTED_VERSION,
        "acados_source_version": installed_acados_version(),
        "acados_source_revision": acados_source_revision(),
        "dimensions": {"nx": NX, "nu": NU, "np": NP, "ny": NY, "ny_e": NY_E, "nh": NH},
        "controller": asdict(cfg),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _safe_denominator_casadi(value: Any, epsilon: float) -> Any:
    sign = ca.if_else(value >= 0.0, 1.0, -1.0)
    return ca.if_else(ca.fabs(value) < epsilon, sign * epsilon, value)


def _geometry_casadi(theta: Any, vrz: Any, cfg: AcadosNMPCModelCfg) -> tuple[Any, Any]:
    epsilon = float(cfg.epsilon)
    sin_beta = (float(cfg.plank_length) / float(cfg.rope_length)) * (1.0 - ca.cos(theta))
    sin_beta = ca.fmin(ca.fmax(sin_beta, -1.0 + epsilon), 1.0 - epsilon)
    beta = ca.asin(sin_beta)
    denominator = float(cfg.plank_length) * _safe_denominator_casadi(
        ca.cos(beta - theta), epsilon
    )
    return beta, -vrz * ca.cos(beta) / denominator


def _response_casadi(vrz: Any, command_next: Any, tau_s: Any, gain: Any, bias: Any, elapsed: float, epsilon: float) -> Any:
    target = gain * command_next + bias
    tau_safe = ca.fmax(tau_s, epsilon)
    dynamic = target + (vrz - target) * ca.exp(-elapsed / tau_safe)
    return ca.if_else(tau_s <= epsilon, target, dynamic)


def _derivatives_casadi(state: Any, vrz: Any, cfg: AcadosNMPCModelCfg) -> Any:
    pb, vb, theta = state[0], state[1], state[2]
    _, omega = _geometry_casadi(theta, vrz, cfg)
    inertia = float(cfg.ball_inertia_ratio) * float(cfg.ball_mass) * float(cfg.ball_radius) ** 2
    effective_mass = inertia / float(cfg.ball_radius) ** 2 + float(cfg.ball_mass)
    gamma = float(cfg.ball_mass) / effective_mass
    ab = gamma * (
        (pb + float(cfg.ball_position_offset) - float(cfg.plank_length)) * omega**2
        - abs(float(cfg.gravity)) * ca.sin(theta)
    )
    return ca.vertcat(vb, ab, omega)


def build_acados_ocp(cfg: AcadosNMPCControllerCfg, model_name: str) -> Any:
    """Map the documented discrete NLP to an :class:`AcadosOcp`."""
    require_acados()
    validate_controller_cfg(cfg)
    x = ca.SX.sym("x", NX)
    u = ca.SX.sym("u", NU)
    p = ca.SX.sym("p", NP)
    pg, vg, tau_s, gain, bias = (p[index] for index in range(NP))
    command_next = x[4] + u[0]
    dt = float(cfg.solver.step_dt)
    epsilon = float(cfg.model.epsilon)

    vrz_0 = _response_casadi(x[3], command_next, tau_s, gain, bias, 0.0, epsilon)
    vrz_half = _response_casadi(x[3], command_next, tau_s, gain, bias, 0.5 * dt, epsilon)
    vrz_end = _response_casadi(x[3], command_next, tau_s, gain, bias, dt, epsilon)
    physical = x[:3]
    k1 = _derivatives_casadi(physical, vrz_0, cfg.model)
    k2 = _derivatives_casadi(physical + 0.5 * dt * k1, vrz_half, cfg.model)
    k3 = _derivatives_casadi(physical + 0.5 * dt * k2, vrz_half, cfg.model)
    k4 = _derivatives_casadi(physical + dt * k3, vrz_end, cfg.model)
    physical_next = physical + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    x_next = ca.vertcat(physical_next, vrz_end, command_next)

    _, omega = _geometry_casadi(x[2], x[3], cfg.model)
    stage_y = ca.vertcat(x[0] - pg, x[1] - vg, x[2], omega, x[3], command_next, u[0])
    terminal_y = ca.vertcat(x[0] - pg, x[1] - vg, x[2], omega, x[3], x[4])
    soft_h = ca.vertcat(x[0], x[2], x[0] - pg)

    model = AcadosModel()
    model.name = model_name
    model.x = x
    model.u = u
    model.p = p
    model.disc_dyn_expr = x_next
    model.cost_y_expr = stage_y
    model.cost_y_expr_0 = stage_y
    model.cost_y_expr_e = terminal_y
    model.con_h_expr = soft_h
    model.con_h_expr_0 = soft_h
    model.con_h_expr_e = soft_h

    ocp = AcadosOcp()
    ocp.model = model
    n_horizon = int(cfg.solver.n_horizon)
    ocp.solver_options.N_horizon = n_horizon
    ocp.solver_options.tf = n_horizon * dt
    ocp.solver_options.integrator_type = str(cfg.solver.integrator_type)
    ocp.solver_options.nlp_solver_type = str(cfg.solver.nlp_solver_type)
    ocp.solver_options.qp_solver = str(cfg.solver.qp_solver)
    ocp.solver_options.hessian_approx = str(cfg.solver.hessian_approx)
    ocp.solver_options.qp_solver_cond_N = int(cfg.solver.qp_solver_cond_N)
    ocp.solver_options.qp_solver_warm_start = int(cfg.solver.qp_solver_warm_start)
    ocp.solver_options.levenberg_marquardt = float(cfg.solver.levenberg_marquardt)
    ocp.solver_options.print_level = int(cfg.solver.print_level)
    # The paper objective is a discrete sum with the configured matrices used
    # literally, not acados' default time-step-scaled running cost.
    ocp.solver_options.cost_scaling = np.ones(n_horizon + 1)

    ocp.cost.cost_type = "NONLINEAR_LS"
    ocp.cost.cost_type_0 = "NONLINEAR_LS"
    ocp.cost.cost_type_e = "NONLINEAR_LS"
    ocp.cost.W = np.diag(np.asarray(cfg.objective.stage_weights, dtype=np.float64))
    ocp.cost.W_0 = ocp.cost.W.copy()
    ocp.cost.W_e = np.diag(np.asarray(cfg.objective.terminal_weights, dtype=np.float64))
    ocp.cost.yref = np.zeros(NY)
    ocp.cost.yref_0 = np.zeros(NY)
    ocp.cost.yref_e = np.zeros(NY_E)

    constraints = cfg.constraints
    delta_limit = float(constraints.max_acc) * dt
    ocp.constraints.idxbu = np.array([0], dtype=np.int64)
    ocp.constraints.lbu = np.array([-delta_limit])
    ocp.constraints.ubu = np.array([delta_limit])
    ocp.constraints.x0 = np.zeros(NX)
    ocp.constraints.lh = np.array(
        [constraints.ball_position_min, constraints.theta_min, constraints.tracking_error_min]
    )
    ocp.constraints.uh = np.array(
        [constraints.ball_position_max, constraints.theta_max, constraints.tracking_error_max]
    )
    ocp.constraints.lh_e = ocp.constraints.lh.copy()
    ocp.constraints.uh_e = ocp.constraints.uh.copy()
    ocp.constraints.lh_0 = ocp.constraints.lh.copy()
    ocp.constraints.uh_0 = ocp.constraints.uh.copy()
    soft_ids = np.arange(NH, dtype=np.int64)
    ocp.constraints.idxsh = soft_ids
    ocp.constraints.idxsh_0 = soft_ids
    ocp.constraints.idxsh_e = soft_ids
    ocp.cost.zl = np.full(NH, float(constraints.slack_l1))
    ocp.cost.zu = np.full(NH, float(constraints.slack_l1))
    ocp.cost.Zl = np.full(NH, float(constraints.slack_l2))
    ocp.cost.Zu = np.full(NH, float(constraints.slack_l2))
    ocp.cost.zl_e = ocp.cost.zl.copy()
    ocp.cost.zu_e = ocp.cost.zu.copy()
    ocp.cost.Zl_e = ocp.cost.Zl.copy()
    ocp.cost.Zu_e = ocp.cost.Zu.copy()
    ocp.cost.zl_0 = ocp.cost.zl.copy()
    ocp.cost.zu_0 = ocp.cost.zu.copy()
    ocp.cost.Zl_0 = ocp.cost.Zl.copy()
    ocp.cost.Zu_0 = ocp.cost.Zu.copy()

    max_velocity = float(constraints.max_velocity)
    if max_velocity > 0.0:
        velocity_ids = np.array([3, 4], dtype=np.int64)
        velocity_min = np.full(2, -max_velocity)
        velocity_max = np.full(2, max_velocity)
        ocp.constraints.idxbx = velocity_ids
        ocp.constraints.lbx = velocity_min
        ocp.constraints.ubx = velocity_max
        ocp.constraints.idxbx_e = velocity_ids
        ocp.constraints.lbx_e = velocity_min
        ocp.constraints.ubx_e = velocity_max

    ocp.parameter_values = np.array([0.35, 0.0, 0.155, 0.86, -0.0035])
    return ocp


class AcadosNMPCController:
    """Own one acados solver or one native acados batch solver."""

    def __init__(
        self,
        cfg: AcadosNMPCControllerCfg,
        num_envs: int,
        *,
        build_root: str | Path,
    ):
        require_acados()
        validate_controller_cfg(cfg)
        self.cfg = cfg
        self.num_envs = int(num_envs)
        if self.num_envs <= 0:
            raise ValueError("AcadosNMPCController requires num_envs > 0.")
        self.n_horizon = int(cfg.solver.n_horizon)
        self.fingerprint = controller_fingerprint(cfg)
        self.build_dir = Path(build_root).resolve() / self.fingerprint
        self.build_dir.mkdir(parents=True, exist_ok=True)
        model_name = f"acados_unified_nmpc_{self.fingerprint}"
        self.ocp = build_acados_ocp(cfg, model_name)
        self.ocp.solver_options.with_batch_functionality = self.num_envs > 1
        code_export_directory = self.build_dir / "c_generated_code"
        self.ocp.code_gen_opts.code_export_directory = str(code_export_directory)
        json_file = str(self.build_dir / f"{model_name}.json")
        solver_library_names = (
            f"libacados_ocp_solver_{model_name}.so",
            f"libacados_ocp_solver_{model_name}.dylib",
            f"acados_ocp_solver_{model_name}.dll",
            f"libacados_ocp_solver_{model_name}.dll",
        )
        reuse_cache = Path(json_file).is_file() and any(
            (code_export_directory / name).is_file() for name in solver_library_names
        )
        if self.num_envs == 1:
            self._batch_solver = None
            solver = AcadosOcpSolver(
                self.ocp,
                json_file=json_file,
                generate=not reuse_cache,
                build=not reuse_cache,
                check_reuse_possible=True,
            )
            self._solvers = [solver]
        else:
            try:
                self._batch_solver = AcadosOcpBatchSolver(
                    self.ocp,
                    N_batch_init=self.num_envs,
                    num_threads_in_batch_solve=int(cfg.solver.num_threads_in_batch_solve),
                    json_file=json_file,
                    generate=not reuse_cache,
                    build=not reuse_cache,
                    check_code_reuse_possible=True,
                )
            except TypeError:  # v0.5.4 accepts a positional batch size.
                self._batch_solver = AcadosOcpBatchSolver(
                    self.ocp,
                    self.num_envs,
                    num_threads_in_batch_solve=int(cfg.solver.num_threads_in_batch_solve),
                    json_file=json_file,
                    generate=not reuse_cache,
                    build=not reuse_cache,
                    check_code_reuse_possible=True,
                )
            self._solvers = list(self._batch_solver.ocp_solvers)
        self.reset()

    @property
    def is_batch(self) -> bool:
        return self._batch_solver is not None

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Reset only the selected solver iterates."""
        ids = range(self.num_envs) if env_ids is None else (int(value) for value in env_ids)
        for env_id in ids:
            if env_id < 0 or env_id >= self.num_envs:
                raise IndexError(f"Solver env id {env_id} is outside [0, {self.num_envs}).")
            solver = self._solvers[env_id]
            try:
                solver.reset(reset_qp_solver_mem=1)
            except TypeError:
                solver.reset()
            zero_x = np.zeros(NX)
            zero_u = np.zeros(NU)
            for stage in range(self.n_horizon):
                solver.set(stage, "x", zero_x)
                solver.set(stage, "u", zero_u)
            solver.set(self.n_horizon, "x", zero_x)

    def solve(
        self,
        initial_states: np.ndarray,
        stage_parameters: np.ndarray,
    ) -> AcadosNMPCStepResult:
        """Set all environments, perform one single/batch solve, and validate output."""
        initial_states = np.asarray(initial_states, dtype=np.float64)
        parameters = np.asarray(stage_parameters, dtype=np.float64)
        expected_p_shape = (self.num_envs, self.n_horizon + 1, NP)
        if initial_states.shape != (self.num_envs, NX):
            raise ValueError(
                f"initial_states must have shape ({self.num_envs}, {NX}), got {initial_states.shape}."
            )
        if parameters.shape != expected_p_shape:
            raise ValueError(f"stage_parameters must have shape {expected_p_shape}, got {parameters.shape}.")
        if not np.all(np.isfinite(initial_states)) or not np.all(np.isfinite(parameters)):
            raise ValueError("Initial states and stage parameters must be finite.")

        start = time.perf_counter()
        if self._batch_solver is None:
            solver = self._solvers[0]
            solver.constraints_set(0, "lbx", initial_states[0])
            solver.constraints_set(0, "ubx", initial_states[0])
            for stage in range(self.n_horizon + 1):
                solver.set(stage, "p", parameters[0, stage])
        else:
            self._batch_solver.constraints_set(0, "lbx", initial_states)
            self._batch_solver.constraints_set(0, "ubx", initial_states)
            for stage in range(self.n_horizon + 1):
                self._batch_solver.set(stage, "p", parameters[:, stage, :])

        call_exception: Exception | None = None
        try:
            if self._batch_solver is None:
                raw_statuses = np.array([self._solvers[0].solve()], dtype=np.int64)
            else:
                batch_status = self._batch_solver.solve()
                raw_statuses = np.asarray(
                    getattr(self._batch_solver, "status", batch_status), dtype=np.int64
                ).reshape(-1)
                if raw_statuses.size == 1:
                    raw_statuses = np.asarray(
                        [self._status_from_solver(solver, int(raw_statuses[0])) for solver in self._solvers]
                    )
        except Exception as exc:  # Do not pass a partially updated solution to the environment.
            call_exception = exc
            raw_statuses = np.full(self.num_envs, -1, dtype=np.int64)
        wall_time = time.perf_counter() - start

        action = np.zeros((self.num_envs, NU), dtype=np.float64)
        fallback = np.ones(self.num_envs, dtype=bool)
        solver_time = np.full(self.num_envs, np.nan)
        qp_time = np.full(self.num_envs, np.nan)
        sqp_iterations = np.full(self.num_envs, np.nan)
        qp_iterations = np.full(self.num_envs, np.nan)
        cost = np.full(self.num_envs, np.nan)
        max_slack = np.full(self.num_envs, np.nan)
        failed_ids: list[int] = []
        delta_limit = float(self.cfg.constraints.max_acc) * float(self.cfg.solver.step_dt)

        for env_id, solver in enumerate(self._solvers):
            status = int(raw_statuses[env_id]) if env_id < raw_statuses.size else -1
            solver_time[env_id] = self._stat(solver, "time_tot")
            qp_time[env_id] = self._stat(solver, "time_qp")
            sqp_iterations[env_id] = self._stat(solver, "sqp_iter")
            qp_iterations[env_id] = self._stat(solver, "qp_iter")
            try:
                cost[env_id] = float(solver.get_cost())
            except Exception:
                pass
            max_slack[env_id] = self._max_slack(solver)
            if call_exception is None and status == 0:
                try:
                    candidate = np.asarray(solver.get(0, "u"), dtype=np.float64).reshape(NU)
                except Exception:
                    candidate = np.full(NU, np.nan)
                if np.all(np.isfinite(candidate)) and np.all(np.abs(candidate) <= delta_limit + 1.0e-8):
                    action[env_id] = np.clip(candidate, -delta_limit, delta_limit)
                    fallback[env_id] = False
                    self._shift_warm_start(solver)
            if fallback[env_id]:
                failed_ids.append(env_id)
        if failed_ids:
            self.reset(failed_ids)
        return AcadosNMPCStepResult(
            action=action,
            status=raw_statuses,
            solver_time=solver_time,
            qp_time=qp_time,
            sqp_iterations=sqp_iterations,
            qp_iterations=qp_iterations,
            cost=cost,
            max_slack=max_slack,
            fallback=fallback,
            wall_time=wall_time,
        )

    @staticmethod
    def _status_from_solver(solver: Any, fallback: int) -> int:
        for name in ("status",):
            value = getattr(solver, name, None)
            if value is not None:
                return int(value)
        try:
            return int(solver.get_status())
        except Exception:
            return fallback

    @staticmethod
    def _stat(solver: Any, name: str) -> float:
        try:
            value = np.asarray(solver.get_stats(name), dtype=np.float64)
            return float(np.max(value)) if value.size else float("nan")
        except Exception:
            return float("nan")

    def _max_slack(self, solver: Any) -> float:
        maximum = 0.0
        found = False
        for stage in range(self.n_horizon + 1):
            for field_name in ("sl", "su"):
                try:
                    value = np.asarray(solver.get(stage, field_name), dtype=np.float64)
                except Exception:
                    continue
                if value.size:
                    maximum = max(maximum, float(np.max(np.abs(value))))
                    found = True
        return maximum if found else float("nan")

    def _shift_warm_start(self, solver: Any) -> None:
        try:
            states = [np.asarray(solver.get(stage, "x")) for stage in range(self.n_horizon + 1)]
            inputs = [np.asarray(solver.get(stage, "u")) for stage in range(self.n_horizon)]
            for stage in range(self.n_horizon):
                solver.set(stage, "x", states[min(stage + 1, self.n_horizon)])
                solver.set(stage, "u", inputs[min(stage + 1, self.n_horizon - 1)])
            solver.set(self.n_horizon, "x", states[-1])
        except Exception:
            # A valid first action remains safe; failed warm-start shifting merely
            # causes acados to retain its previous iterate for the next call.
            return

    def close(self) -> None:
        """Release Python references; generated artifacts remain in the build cache."""
        self._solvers.clear()
        self._batch_solver = None
