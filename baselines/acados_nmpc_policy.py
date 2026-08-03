"""Unified reference-preview NMPC policy backed by acados."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
import math
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

from .acados_nmpc_core import (
    ACADOS_SUPPORTED_VERSION,
    NP,
    AcadosNMPCConstraintsCfg,
    AcadosNMPCController,
    AcadosNMPCControllerCfg,
    AcadosNMPCModelCfg,
    AcadosNMPCObjectiveCfg,
    AcadosNMPCSolverCfg,
    acados_source_revision,
    acados_template_distribution_version,
    controller_fingerprint,
    installed_acados_version,
    validate_controller_cfg,
)
from .base_policy import BasePolicy, BasePolicyCfg, ObservationIndex
from .model_state_predictor import VelocityModelStatePredictor, VelocityModelStatePredictorCfg


LEGACY_OBSERVATION_FIELDS = (
    "pb",
    "vb",
    "ab",
    "theta",
    "omega",
    "alpha",
    "drz",
    "vrz",
    "arz",
    "pg",
    "a_prev",
)


@dataclass
class AcadosNMPCResponseCfg:
    """Nominal low-level first-order velocity response in the OCP."""

    enabled: bool = True
    tau_s: float | str = 0.155
    gain: float | str = 0.86
    bias: float | str = -0.0035


@dataclass
class AcadosNMPCPolicyCfg(BasePolicyCfg):
    """Configuration of the independent acados unified-NMPC policy."""

    name: str = "acados_unified_nmpc"
    build_root: str = "build/acados_unified_nmpc"
    model: AcadosNMPCModelCfg = field(default_factory=AcadosNMPCModelCfg)
    objective: AcadosNMPCObjectiveCfg = field(default_factory=AcadosNMPCObjectiveCfg)
    constraints: AcadosNMPCConstraintsCfg = field(default_factory=AcadosNMPCConstraintsCfg)
    solver: AcadosNMPCSolverCfg = field(default_factory=AcadosNMPCSolverCfg)
    response: AcadosNMPCResponseCfg = field(default_factory=AcadosNMPCResponseCfg)
    state_predictor: VelocityModelStatePredictorCfg = field(
        default_factory=VelocityModelStatePredictorCfg
    )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "AcadosNMPCPolicyCfg":
        """Build a strict config from either a policy section or a policy file."""
        cfg = cls()
        if not data:
            return cfg
        policy_data = data.get("acados_nmpc_policy", data.get("policy", data))
        if not isinstance(policy_data, Mapping):
            raise TypeError("acados_nmpc_policy must be a mapping.")
        _strict_update(cfg, policy_data, "acados_nmpc_policy")
        return cfg


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


class AcadosNMPCPolicy(BasePolicy):
    """Real-time unified tracking NMPC with an external delay predictor."""

    cfg: AcadosNMPCPolicyCfg

    def __init__(
        self,
        cfg: AcadosNMPCPolicyCfg,
        num_envs: int,
        device: str | torch.device,
        step_dt: float,
        *,
        raw_observation_fields: Sequence[str],
    ):
        super().__init__(cfg, num_envs, device)
        self.step_dt = float(step_dt)
        if not math.isfinite(self.step_dt) or self.step_dt <= 0.0:
            raise ValueError("AcadosNMPCPolicy requires a finite step_dt > 0.")
        self.raw_observation_fields = tuple(str(name) for name in raw_observation_fields)
        if not self.raw_observation_fields:
            raise ValueError("AcadosNMPCPolicy requires raw_observation_fields.")
        if len(set(self.raw_observation_fields)) != len(self.raw_observation_fields):
            raise ValueError("raw_observation_fields must be unique.")

        self._resolve_cfg_defaults()
        self.controller_cfg = AcadosNMPCControllerCfg(
            model=cfg.model,
            objective=cfg.objective,
            constraints=cfg.constraints,
            solver=cfg.solver,
        )
        validate_controller_cfg(self.controller_cfg)
        self.n_horizon = int(cfg.solver.n_horizon)
        self.state_predictor = VelocityModelStatePredictor(
            cfg.state_predictor, num_envs, self.device
        )
        self.reference_base_offset = (
            self.state_predictor.delay_step if self.state_predictor.active else 0
        )
        self._validate_predictor_response_match()
        self._validate_predictor_controller_match()
        self._field_indices = self._validate_and_index_fields()

        build_root = Path(cfg.build_root).expanduser()
        if not build_root.is_absolute():
            build_root = Path(__file__).resolve().parents[1] / build_root
        self.controller = AcadosNMPCController(
            self.controller_cfg,
            num_envs,
            build_root=build_root,
        )

        self.model_state = torch.zeros((self.num_envs, 5), device=self.device)
        self.action = torch.zeros((self.num_envs, 1), device=self.device)
        self.reference_pg = torch.zeros(
            (self.num_envs, self.n_horizon + 1), device=self.device
        )
        self.reference_vg = torch.zeros_like(self.reference_pg)
        self.solver_status = torch.full(
            (self.num_envs,), -1.0, dtype=torch.float32, device=self.device
        )
        self.solver_success = torch.zeros((self.num_envs,), device=self.device)
        self.solver_time = torch.full((self.num_envs,), float("nan"), device=self.device)
        self.qp_time = torch.full_like(self.solver_time, float("nan"))
        self.sqp_iterations = torch.full_like(self.solver_time, float("nan"))
        self.qp_iterations = torch.full_like(self.solver_time, float("nan"))
        self.cost = torch.full_like(self.solver_time, float("nan"))
        self.max_slack = torch.full_like(self.solver_time, float("nan"))
        self.fallback = torch.ones((self.num_envs,), device=self.device)
        self.controller_wall_time = torch.zeros((self.num_envs,), device=self.device)
        self.policy_wall_time = torch.zeros((self.num_envs,), device=self.device)

    @property
    def consumed_observation_fields(self) -> tuple[str, ...]:
        """Named fields actually read by this policy."""
        ordered = [*LEGACY_OBSERVATION_FIELDS]
        for offset in range(self.reference_base_offset, self.reference_base_offset + self.n_horizon + 1):
            pg_name = self._pg_field_name(offset)
            vg_name = f"vg_{offset}"
            for name in (pg_name, vg_name):
                if name not in ordered:
                    ordered.append(name)
        if self.state_predictor.active:
            for offset in range(self.state_predictor.delay_step + 1):
                name = self._pg_field_name(offset)
                if name not in ordered:
                    ordered.append(name)
        return tuple(ordered)

    def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None):
        """Reset predictor and acados iterate only for selected environments."""
        ids = self._env_ids_tensor(env_ids)
        if ids.numel() == 0:
            return
        id_list = [int(value) for value in ids.detach().cpu().tolist()]
        self.controller.reset(id_list)
        self.state_predictor.reset(ids)
        self.model_state[ids] = 0.0
        self.action[ids] = 0.0
        self.reference_pg[ids] = 0.0
        self.reference_vg[ids] = 0.0
        self.solver_status[ids] = -1.0
        self.solver_success[ids] = 0.0
        for buffer in (
            self.solver_time,
            self.qp_time,
            self.sqp_iterations,
            self.qp_iterations,
            self.cost,
            self.max_slack,
        ):
            buffer[ids] = float("nan")
        self.fallback[ids] = 1.0
        self.controller_wall_time[ids] = 0.0
        self.policy_wall_time[ids] = 0.0

    def act(
        self,
        observations: dict[str, torch.Tensor] | torch.Tensor,
        extras: dict | None = None,
    ) -> torch.Tensor:
        """Compute one bounded velocity-command increment per environment."""
        del extras
        policy_start = time.perf_counter()
        raw = self._extract_policy_observation(observations)
        expected_shape = (self.num_envs, len(self.raw_observation_fields))
        if raw.shape != expected_shape:
            raise ValueError(
                f"AcadosNMPCPolicy expects raw observation shape {expected_shape}, "
                f"got {tuple(raw.shape)}."
            )
        if not torch.all(torch.isfinite(raw)):
            raise ValueError("AcadosNMPCPolicy observation must be finite.")

        legacy = raw[:, self._field_indices["legacy"]]
        if self.state_predictor.active:
            delay_refs = raw[:, self._field_indices["predictor_pg"]]
            model_observation = self.state_predictor.predict(
                legacy, reference_positions=delay_refs
            )
        else:
            model_observation = self.state_predictor.predict(legacy)

        state = torch.stack(
            (
                model_observation[:, ObservationIndex.PB],
                model_observation[:, ObservationIndex.VB],
                model_observation[:, ObservationIndex.THETA],
                model_observation[:, ObservationIndex.VRZ],
                self.state_predictor.command_z[:, 0],
            ),
            dim=-1,
        )
        pg = raw[:, self._field_indices["ocp_pg"]]
        vg = raw[:, self._field_indices["ocp_vg"]]
        parameters = torch.empty(
            (self.num_envs, self.n_horizon + 1, NP),
            device=self.device,
            dtype=torch.float32,
        )
        parameters[..., 0] = pg
        parameters[..., 1] = vg
        parameters[..., 2] = float(self.cfg.response.tau_s)
        parameters[..., 3] = float(self.cfg.response.gain)
        parameters[..., 4] = float(self.cfg.response.bias)

        try:
            result = self.controller.solve(
                state.detach().cpu().numpy(), parameters.detach().cpu().numpy()
            )
            action = torch.as_tensor(
                result.action, dtype=torch.float32, device=self.device
            )
            status = torch.as_tensor(
                result.status, dtype=torch.float32, device=self.device
            )
            fallback = torch.as_tensor(
                result.fallback, dtype=torch.bool, device=self.device
            )
            invalid = (
                action.shape != (self.num_envs, 1)
                or status.numel() != self.num_envs
                or fallback.numel() != self.num_envs
            )
            if invalid:
                raise RuntimeError("acados controller returned inconsistent batch dimensions.")
            delta_limit = float(self.cfg.constraints.max_acc) * self.step_dt
            invalid_env = (
                ~torch.isfinite(action[:, 0])
                | (torch.abs(action[:, 0]) > delta_limit + 1.0e-6)
                | (status != 0.0)
                | fallback
            )
            action[invalid_env] = 0.0
            fallback = invalid_env
            self._copy_result_diagnostics(result, status, fallback)
        except Exception:
            # Runtime solver failures are contained at the policy boundary. The
            # command increment remains zero and the predictor queue is advanced
            # consistently with the command that the environment receives.
            try:
                self.controller.reset()
            except Exception:
                pass
            action = torch.zeros((self.num_envs, 1), device=self.device)
            self.solver_status.fill_(-1.0)
            self.solver_success.zero_()
            self.solver_time.fill_(float("nan"))
            self.qp_time.fill_(float("nan"))
            self.sqp_iterations.fill_(float("nan"))
            self.qp_iterations.fill_(float("nan"))
            self.cost.fill_(float("nan"))
            self.max_slack.fill_(float("nan"))
            self.fallback.fill_(1.0)
            self.controller_wall_time.fill_(time.perf_counter() - policy_start)

        self.model_state.copy_(state)
        self.action.copy_(action)
        self.reference_pg.copy_(pg)
        self.reference_vg.copy_(vg)
        self.state_predictor.update_after_action(action)
        self.policy_wall_time.fill_(time.perf_counter() - policy_start)
        return action.clone()

    def _copy_result_diagnostics(self, result: Any, status: torch.Tensor, fallback: torch.Tensor) -> None:
        self.solver_status.copy_(status.reshape(self.num_envs))
        self.fallback.copy_(fallback.to(dtype=torch.float32).reshape(self.num_envs))
        self.solver_success.copy_(((status == 0.0) & ~fallback).to(dtype=torch.float32))
        for name, source in (
            ("solver_time", result.solver_time),
            ("qp_time", result.qp_time),
            ("sqp_iterations", result.sqp_iterations),
            ("qp_iterations", result.qp_iterations),
            ("cost", result.cost),
            ("max_slack", result.max_slack),
        ):
            target = getattr(self, name)
            target.copy_(torch.as_tensor(source, dtype=torch.float32, device=self.device))
        self.controller_wall_time.fill_(float(result.wall_time))

    def get_state(self) -> dict[str, torch.Tensor]:
        """Return per-environment solver, preview, and predictor diagnostics."""
        return {
            "acados_nmpc_state_pb": self.model_state[:, 0],
            "acados_nmpc_state_vb": self.model_state[:, 1],
            "acados_nmpc_state_theta": self.model_state[:, 2],
            "acados_nmpc_state_vrz": self.model_state[:, 3],
            "acados_nmpc_state_command_z": self.model_state[:, 4],
            "acados_nmpc_reference_pg_0": self.reference_pg[:, 0],
            "acados_nmpc_reference_vg_0": self.reference_vg[:, 0],
            "acados_nmpc_reference_base_offset": torch.full(
                (self.num_envs,), float(self.reference_base_offset), device=self.device
            ),
            "acados_nmpc_action": self.action[:, 0],
            "acados_nmpc_solver_status": self.solver_status,
            "acados_nmpc_solver_success": self.solver_success,
            "acados_nmpc_solver_time": self.solver_time,
            "acados_nmpc_qp_time": self.qp_time,
            "acados_nmpc_sqp_iterations": self.sqp_iterations,
            "acados_nmpc_qp_iterations": self.qp_iterations,
            "acados_nmpc_cost": self.cost,
            "acados_nmpc_max_slack": self.max_slack,
            "acados_nmpc_fallback": self.fallback,
            "acados_nmpc_controller_wall_time": self.controller_wall_time,
            "acados_nmpc_policy_wall_time": self.policy_wall_time,
            **self.state_predictor.get_state(),
        }

    def get_resolved_metadata(self) -> dict[str, Any]:
        """Return formulation/build metadata for ``resolved_run.yaml``."""
        return {
            "acados_supported_version": ACADOS_SUPPORTED_VERSION,
            "acados_installed_version": installed_acados_version(),
            "acados_source_revision": acados_source_revision(),
            "acados_template_distribution_version": acados_template_distribution_version(),
            "acados_build_fingerprint": controller_fingerprint(self.controller_cfg),
            "acados_build_directory": str(self.controller.build_dir),
            "effective_horizon": self.n_horizon,
            "reference_base_offset": self.reference_base_offset,
            "required_raw_preview_horizon": self.reference_base_offset + self.n_horizon,
            "consumed_observation_fields": list(self.consumed_observation_fields),
            "resolved_policy_config": asdict(self.cfg),
        }

    def close(self):
        """Release solver objects."""
        self.controller.close()

    def to(self, device: str | torch.device):
        device = torch.device(device)
        for name, value in vars(self).items():
            if isinstance(value, torch.Tensor):
                setattr(self, name, value.to(device=device))
        self.state_predictor.to(device)
        self.device = device
        return self

    def _resolve_cfg_defaults(self) -> None:
        cfg = self.cfg
        defaults = {
            "plank_length": 1.06,
            "rope_length": 0.9,
            "ball_position_offset": 0.33,
            "gravity": 9.81,
            "ball_mass": 0.0005,
            "ball_radius": 0.023,
        }
        for name, default in defaults.items():
            if _is_auto(getattr(cfg.model, name)):
                setattr(cfg.model, name, default)
        if _is_auto(cfg.solver.step_dt):
            cfg.solver.step_dt = self.step_dt
        if not math.isclose(float(cfg.solver.step_dt), self.step_dt, rel_tol=0.0, abs_tol=1.0e-9):
            raise ValueError(
                "acados_nmpc_policy.solver.step_dt must match the environment control step: "
                f"{cfg.solver.step_dt} != {self.step_dt}."
            )
        if _is_auto(cfg.constraints.max_acc):
            cfg.constraints.max_acc = 0.5
        if _is_auto(cfg.constraints.max_velocity):
            cfg.constraints.max_velocity = 0.0
        if not cfg.response.enabled:
            cfg.response.tau_s = 0.0
            cfg.response.gain = 1.0
            cfg.response.bias = 0.0
        else:
            response_defaults = {"tau_s": 0.155, "gain": 0.86, "bias": -0.0035}
            for name, default in response_defaults.items():
                if _is_auto(getattr(cfg.response, name)):
                    setattr(cfg.response, name, default)
        tau = float(cfg.response.tau_s)
        gain = float(cfg.response.gain)
        bias = float(cfg.response.bias)
        if not math.isfinite(tau) or tau < 0.0:
            raise ValueError("response.tau_s must be finite and non-negative.")
        if not math.isfinite(gain) or gain <= 0.0:
            raise ValueError("response.gain must be finite and positive.")
        if not math.isfinite(bias):
            raise ValueError("response.bias must be finite.")

        predictor = cfg.state_predictor
        if _is_auto(predictor.delay_step):
            predictor.delay_step = 0
        if _is_auto(predictor.step_dt) or float(predictor.step_dt) <= 0.0:
            predictor.step_dt = self.step_dt
        if _is_auto(predictor.max_acc):
            predictor.max_acc = float(cfg.constraints.max_acc)
        if _is_auto(predictor.max_velocity):
            predictor.max_velocity = float(cfg.constraints.max_velocity)
        for name in (
            "plank_length",
            "rope_length",
            "ball_position_offset",
            "gravity",
            "ball_mass",
            "ball_radius",
            "ball_inertia_ratio",
            "epsilon",
        ):
            if _is_auto(getattr(predictor, name)):
                setattr(predictor, name, getattr(cfg.model, name))
        if _is_auto(predictor.velocity_response_tau_s):
            predictor.velocity_response_tau_s = tau
        if _is_auto(predictor.velocity_response_gain):
            predictor.velocity_response_gain = gain
        if _is_auto(predictor.velocity_response_bias):
            predictor.velocity_response_bias = bias
        if _is_auto(predictor.velocity_response_max_abs_velocity):
            predictor.velocity_response_max_abs_velocity = 0.0

    def _validate_predictor_response_match(self) -> None:
        if not self.state_predictor.active:
            return
        if not self.state_predictor.velocity_response_enabled:
            raise ValueError(
                "An active AcadosNMPCPolicy state predictor must enable velocity_response_enabled "
                "because the OCP models that response."
            )
        pairs = {
            "tau_s": (float(self.cfg.response.tau_s), self.state_predictor.velocity_response_tau_s),
            "gain": (float(self.cfg.response.gain), self.state_predictor.velocity_response_gain),
            "bias": (float(self.cfg.response.bias), self.state_predictor.velocity_response_bias),
        }
        mismatched = [
            name
            for name, (ocp_value, predictor_value) in pairs.items()
            if not math.isclose(ocp_value, float(predictor_value), rel_tol=0.0, abs_tol=1.0e-9)
        ]
        if mismatched:
            raise ValueError(
                "Active predictor response parameters must equal the Acados NMPC response "
                f"parameters; mismatched field(s): {', '.join(mismatched)}."
            )
        if float(self.state_predictor.velocity_response_max_abs_velocity) != 0.0:
            raise ValueError(
                "AcadosNMPCPolicy requires state_predictor.velocity_response_max_abs_velocity=0; "
                "use constraints.max_velocity for hard command/response bounds."
            )

    def _validate_predictor_controller_match(self) -> None:
        pairs = {
            "step_dt": (self.step_dt, self.state_predictor.step_dt),
            "max_acc": (float(self.cfg.constraints.max_acc), self.state_predictor.max_acc),
            "max_velocity": (
                float(self.cfg.constraints.max_velocity),
                self.state_predictor.max_velocity,
            ),
        }
        mismatched = [
            name
            for name, (controller_value, predictor_value) in pairs.items()
            if not math.isclose(
                float(controller_value), float(predictor_value), rel_tol=0.0, abs_tol=1.0e-9
            )
        ]
        if mismatched:
            raise ValueError(
                "The state predictor command accumulator must match the Acados NMPC/environment "
                f"settings; mismatched field(s): {', '.join(mismatched)}."
            )

    def _validate_and_index_fields(self) -> dict[str, list[int]]:
        fields = self.raw_observation_fields
        missing_legacy = [name for name in LEGACY_OBSERVATION_FIELDS if name not in fields]
        if missing_legacy:
            raise ValueError(
                "AcadosNMPCPolicy raw observation is missing legacy field(s): "
                + ", ".join(missing_legacy)
            )
        required_horizon = self.reference_base_offset + self.n_horizon
        required_pg = [self._pg_field_name(offset) for offset in range(required_horizon + 1)]
        required_vg = [f"vg_{offset}" for offset in range(required_horizon + 1)]
        missing = [name for name in (*required_pg, *required_vg) if name not in fields]
        if missing:
            raise ValueError(
                "AcadosNMPCPolicy reference preview is too short or malformed. Missing field(s): "
                f"{', '.join(missing)}. Configure reference_preview.future_steps >= "
                f"{required_horizon}."
            )
        base = self.reference_base_offset
        return {
            "legacy": [fields.index(name) for name in LEGACY_OBSERVATION_FIELDS],
            "predictor_pg": [
                fields.index(self._pg_field_name(offset))
                for offset in range(self.state_predictor.delay_step + 1)
            ],
            "ocp_pg": [
                fields.index(self._pg_field_name(offset))
                for offset in range(base, base + self.n_horizon + 1)
            ],
            "ocp_vg": [
                fields.index(f"vg_{offset}")
                for offset in range(base, base + self.n_horizon + 1)
            ],
        }

    def _pg_field_name(self, offset: int) -> str:
        if offset == 0:
            if "pg" in self.raw_observation_fields:
                return "pg"
            return "pg_0"
        return f"pg_{offset}"


def _is_auto(value: object) -> bool:
    return isinstance(value, str) and value.lower() == "auto"
