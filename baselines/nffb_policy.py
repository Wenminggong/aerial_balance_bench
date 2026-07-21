"""Nonlinear feedforward--feedback baseline for unified reference tracking."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field

import torch

from .base_policy import BasePolicy, BasePolicyCfg, ObservationIndex
from .model_state_predictor import (
    VelocityModelStatePredictor,
    VelocityModelStatePredictorCfg,
    extract_reference_positions,
)
from .velocity_interface_model import VelocityInterfaceModel, VelocityInterfaceModelCfg
from .velocity_response_compensator import (
    FirstOrderVelocityResponseCompensator,
    NFFBVelocityResponseCompensationCfg,
)


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


def validate_nffb_environment_contract(
    *,
    task_name: str,
    interface_name: str,
    reference_preview_enabled: bool,
    reference_preview_future_steps: int,
    robustness_enabled: bool,
    action_delay_enabled: bool,
    delay_step: int,
    state_predictor_enabled: bool = False,
    state_predictor_delay_step: int = 0,
):
    """Validate the environment and predictor features supported by NFFB."""
    if task_name != "unified_tracking":
        raise ValueError("NFFB supports only task_name='unified_tracking'.")
    if interface_name != "velocity":
        raise ValueError("NFFB supports only interface_name='velocity'.")
    delay_active = robustness_enabled and action_delay_enabled and int(delay_step) > 0
    predictor_active = state_predictor_enabled and int(state_predictor_delay_step) > 0
    if delay_active and not predictor_active:
        raise ValueError(
            "NFFB action delay requires an active state predictor with a matching delay_step."
        )
    if predictor_active and not delay_active:
        raise ValueError(
            "NFFB state predictor must not be active when the environment action delay is inactive."
        )
    if delay_active and int(state_predictor_delay_step) != int(delay_step):
        raise ValueError(
            "NFFB state predictor delay_step must match robustness.delay_step; "
            f"got {state_predictor_delay_step} != {delay_step}."
        )

    required_future_steps = int(delay_step) + 1 if delay_active else 1
    if (
        not reference_preview_enabled
        or int(reference_preview_future_steps) < required_future_steps
    ):
        raise ValueError(
            "NFFB requires reference_preview.enabled=true and "
            f"future_steps >= {required_future_steps}."
        )


@dataclass
class NFFBModelCfg:
    """Model parameters, with ``auto`` values resolved by the runner."""

    plank_length: float | str = "auto"
    rope_length: float | str = "auto"
    ball_position_offset: float | str = "auto"
    gravity: float | str = "auto"
    ball_mass: float | str = "auto"
    ball_radius: float | str = "auto"
    ball_inertia_ratio: float = 0.4
    epsilon: float = 1.0e-6


@dataclass
class NFFBOuterLoopCfg:
    """Bandwidth-form outer ball-tracking controller settings."""

    natural_frequency: float = 1.0
    damping_ratio: float = 1.0
    integral_pole: float = 0.0
    integral_limit: float = 0.25
    acceleration_feedforward_enabled: bool = True


@dataclass
class NFFBCommandFilterCfg:
    """Second-order beam-angle command filter settings."""

    natural_frequency: float = 2.5
    damping_ratio: float = 1.0


@dataclass
class NFFBInnerLoopCfg:
    """Beam-angle feedback settings."""

    k_theta: float = 4.0


@dataclass
class NFFBConstraintsCfg:
    """Reference, beam, and drone command constraints."""

    theta_max: float = 40.0 * math.pi / 180.0
    omega_max: float = 0.5
    max_acc: float | str = "auto"
    max_velocity: float | str = "auto"
    max_reference_acceleration: float | str = "auto"


@dataclass
class NFFBPolicyCfg(BasePolicyCfg):
    """Complete configuration for the NFFB velocity-interface baseline."""

    name: str = "nffb"
    model: NFFBModelCfg = field(default_factory=NFFBModelCfg)
    outer_loop: NFFBOuterLoopCfg = field(default_factory=NFFBOuterLoopCfg)
    command_filter: NFFBCommandFilterCfg = field(default_factory=NFFBCommandFilterCfg)
    inner_loop: NFFBInnerLoopCfg = field(default_factory=NFFBInnerLoopCfg)
    constraints: NFFBConstraintsCfg = field(default_factory=NFFBConstraintsCfg)
    state_predictor: VelocityModelStatePredictorCfg = field(
        default_factory=VelocityModelStatePredictorCfg
    )
    velocity_response_compensation: NFFBVelocityResponseCompensationCfg = field(
        default_factory=NFFBVelocityResponseCompensationCfg
    )

    @classmethod
    def from_dict(cls, data: Mapping | None) -> "NFFBPolicyCfg":
        """Build an NFFB configuration from a YAML dictionary."""
        cfg = cls()
        if not data:
            return cfg
        policy_data = data.get("nffb_policy", data.get("policy", data))
        if "name" in policy_data:
            cfg.name = str(policy_data["name"])
        if "policy_name" in policy_data:
            cfg.name = str(policy_data["policy_name"])
        _update_dataclass(cfg.model, policy_data.get("model"))
        _update_dataclass(cfg.outer_loop, policy_data.get("outer_loop"))
        _update_dataclass(cfg.command_filter, policy_data.get("command_filter"))
        _update_dataclass(cfg.inner_loop, policy_data.get("inner_loop"))
        _update_dataclass(cfg.constraints, policy_data.get("constraints"))
        cfg.state_predictor = VelocityModelStatePredictorCfg.from_dict(
            policy_data.get("state_predictor")
        )
        cfg.velocity_response_compensation = (
            NFFBVelocityResponseCompensationCfg.from_dict(
                policy_data.get("velocity_response_compensation")
            )
        )
        return cfg


class NFFBPolicy(BasePolicy):
    """Vectorized nonlinear feedforward--feedback tracking policy."""

    cfg: NFFBPolicyCfg

    def __init__(
        self,
        cfg: NFFBPolicyCfg,
        num_envs: int,
        device: str | torch.device,
        step_dt: float,
        raw_observation_dim: int,
        raw_observation_fields: Sequence[str],
    ):
        super().__init__(cfg, num_envs, device)
        self.step_dt = float(step_dt)
        if not math.isfinite(self.step_dt) or self.step_dt <= 0.0:
            raise ValueError("NFFBPolicy requires a finite positive step_dt.")

        self.raw_observation_dim = int(raw_observation_dim)
        self.raw_observation_fields = tuple(str(name) for name in raw_observation_fields)
        self._resolve_auto_defaults()
        self._validate_config()

        model_cfg = VelocityInterfaceModelCfg(
            plank_length=float(cfg.model.plank_length),
            rope_length=float(cfg.model.rope_length),
            ball_position_offset=float(cfg.model.ball_position_offset),
            gravity=float(cfg.model.gravity),
            ball_mass=float(cfg.model.ball_mass),
            ball_radius=float(cfg.model.ball_radius),
            ball_inertia_ratio=float(cfg.model.ball_inertia_ratio),
            epsilon=float(cfg.model.epsilon),
        )
        self.model = VelocityInterfaceModel(model_cfg)

        outer = cfg.outer_loop
        omega_o = float(outer.natural_frequency)
        zeta_o = float(outer.damping_ratio)
        alpha = float(outer.integral_pole)
        self.kp = omega_o**2 + 2.0 * alpha * zeta_o * omega_o
        self.kv = 2.0 * zeta_o * omega_o + alpha
        self.ki = alpha * omega_o**2

        constraints = cfg.constraints
        self.theta_max = float(constraints.theta_max)
        self.omega_max = float(constraints.omega_max)
        self.max_acc = float(constraints.max_acc)
        self.max_velocity = float(constraints.max_velocity)
        if _is_auto(constraints.max_reference_acceleration):
            constraints.max_reference_acceleration = (
                self.model.gamma * self.model.gravity * math.sin(self.theta_max)
            )
        self.max_reference_acceleration = float(constraints.max_reference_acceleration)
        self.max_velocity_change = self.max_acc * self.step_dt
        self.saturation_tolerance = max(float(cfg.model.epsilon), 1.0e-7)

        self._resolve_predictor_defaults()
        self._resolve_velocity_response_compensation_defaults()
        self.state_predictor = VelocityModelStatePredictor(
            cfg.state_predictor,
            num_envs,
            self.device,
        )
        self.velocity_response_compensator = (
            FirstOrderVelocityResponseCompensator(
                cfg.velocity_response_compensation,
                num_envs,
                self.device,
                self.step_dt,
                epsilon=float(cfg.model.epsilon),
            )
        )
        self.reference_offset = (
            self.state_predictor.delay_step if self.state_predictor.active else 0
        )
        self._validate_predictor_alignment()
        self._validate_velocity_response_compensation_alignment()
        self._field_indices = self._validate_observation_spec()
        self._allocate_buffers()

    def _allocate_buffers(self):
        shape = (self.num_envs, 1)
        float_buffers = (
            "p_ref",
            "v_ref",
            "a_ref_raw",
            "a_ref_limited",
            "a_ref_used",
            "position_error",
            "velocity_error",
            "integral_error",
            "ball_acceleration_raw",
            "ball_acceleration_min",
            "ball_acceleration_max",
            "ball_acceleration_command",
            "theta_star",
            "theta_d",
            "omega_d",
            "filter_alpha",
            "theta_error",
            "omega_command",
            "beta",
            "velocity_raw",
            "velocity_desired",
            "velocity_command",
            "predictor_command_sync_error",
            "last_action",
        )
        for name in float_buffers:
            setattr(self, name, torch.zeros(shape, dtype=torch.float32, device=self.device))

        bool_buffers = (
            "reference_acceleration_saturated",
            "ball_acceleration_saturated",
            "theta_command_saturated",
            "omega_command_saturated",
            "velocity_saturated",
            "acceleration_saturated",
            "anti_windup_frozen",
        )
        for name in bool_buffers:
            setattr(self, name, torch.zeros(shape, dtype=torch.bool, device=self.device))
        self.filter_needs_init = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

    def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None):
        """Reset all or selected controller states."""
        env_ids = self._env_ids_tensor(env_ids)
        if env_ids.numel() == 0:
            return
        for value in vars(self).values():
            if isinstance(value, torch.Tensor) and value.shape[:1] == (self.num_envs,):
                if value.dtype == torch.bool:
                    value[env_ids] = False
                else:
                    value[env_ids] = 0.0
        self.filter_needs_init[env_ids] = True
        self.state_predictor.reset(env_ids)
        self.velocity_response_compensator.reset(env_ids)

    def act(self, observations: dict[str, torch.Tensor] | torch.Tensor, extras: dict | None = None) -> torch.Tensor:
        """Return a physical vertical-velocity increment ``delta_vrz``."""
        del extras
        observation = self._extract_policy_observation(observations)
        expected_shape = (self.num_envs, self.raw_observation_dim)
        if observation.shape != expected_shape:
            raise ValueError(f"NFFBPolicy expects observation shape {expected_shape}, got {tuple(observation.shape)}.")
        if not bool(torch.isfinite(observation).all()):
            raise ValueError("NFFBPolicy received a non-finite observation.")

        reference_positions = None
        if self.state_predictor.active:
            reference_positions = extract_reference_positions(
                observation,
                self.raw_observation_fields,
                self.state_predictor.delay_step,
            )
        model_observation = self.state_predictor.predict(
            observation[:, :11],
            reference_positions=reference_positions,
        )

        pb = model_observation[:, ObservationIndex.PB : ObservationIndex.PB + 1]
        vb = model_observation[:, ObservationIndex.VB : ObservationIndex.VB + 1]
        theta = model_observation[:, ObservationIndex.THETA : ObservationIndex.THETA + 1]
        omega = model_observation[:, ObservationIndex.OMEGA : ObservationIndex.OMEGA + 1]

        self.p_ref.copy_(
            model_observation[:, ObservationIndex.PG : ObservationIndex.PG + 1]
        )
        self.v_ref.copy_(self._field(observation, f"vg_{self.reference_offset}"))
        vg_next = self._field(observation, f"vg_{self.reference_offset + 1}")
        self.a_ref_raw.copy_((vg_next - self.v_ref) / self.step_dt)
        self.a_ref_limited.copy_(
            torch.clamp(
                self.a_ref_raw,
                min=-self.max_reference_acceleration,
                max=self.max_reference_acceleration,
            )
        )
        self.reference_acceleration_saturated.copy_(
            torch.abs(self.a_ref_raw - self.a_ref_limited) > self.saturation_tolerance
        )
        if self.cfg.outer_loop.acceleration_feedforward_enabled:
            self.a_ref_used.copy_(self.a_ref_limited)
        else:
            self.a_ref_used.zero_()

        self.position_error.copy_(pb - self.p_ref)
        self.velocity_error.copy_(vb - self.v_ref)
        self.ball_acceleration_raw.copy_(
            self.a_ref_used
            - self.kp * self.position_error
            - self.kv * self.velocity_error
            - self.ki * self.integral_error
        )

        acceleration_min, acceleration_max = self.model.feasible_ball_acceleration(
            pb,
            omega,
            self.theta_max,
        )
        self.ball_acceleration_min.copy_(acceleration_min)
        self.ball_acceleration_max.copy_(acceleration_max)
        self.ball_acceleration_command.copy_(
            torch.maximum(
                torch.minimum(self.ball_acceleration_raw, self.ball_acceleration_max),
                self.ball_acceleration_min,
            )
        )
        self.ball_acceleration_saturated.copy_(
            torch.abs(self.ball_acceleration_raw - self.ball_acceleration_command)
            > self.saturation_tolerance
        )
        self.theta_star.copy_(
            self.model.desired_beam_angle(
                pb,
                omega,
                self.ball_acceleration_command,
                self.theta_max,
            )
        )

        self._update_command_filter(theta, omega)
        self.theta_error.copy_(theta - self.theta_d)
        self.omega_command.copy_(
            self.omega_d - float(self.cfg.inner_loop.k_theta) * self.theta_error
        )
        self.beta.copy_(self.model.rope_angle(theta))
        self.velocity_raw.copy_(self.model.vertical_velocity(theta, self.omega_command))

        pending_commands = None
        if self.state_predictor.active:
            pending_commands = self.state_predictor.get_pending_commands()

        if self.velocity_response_compensator.enabled:
            compensated_command = self.velocity_response_compensator.compensate(
                self.velocity_raw,
                self.velocity_command,
                pending_commands=pending_commands,
                max_input_change=self.max_velocity_change,
                max_abs_input=self.max_velocity,
            )
            self.velocity_desired.copy_(
                self.velocity_response_compensator.limited_output_z
            )
            self.velocity_saturated.copy_(
                self.velocity_response_compensator.output_saturated
                | self.velocity_response_compensator.predicted_output_clipped
                | self.velocity_response_compensator.input_velocity_saturated
            )
            requested_action = compensated_command - self.velocity_command
            self.last_action.copy_(
                torch.clamp(
                    requested_action,
                    min=-self.max_velocity_change,
                    max=self.max_velocity_change,
                )
            )
            self.acceleration_saturated.copy_(
                self.velocity_response_compensator.input_acceleration_saturated
                | (
                    torch.abs(requested_action - self.last_action)
                    > self.saturation_tolerance
                )
            )
            self.velocity_command.add_(self.last_action)
            if self.max_velocity > 0.0:
                self.velocity_command.clamp_(
                    min=-self.max_velocity,
                    max=self.max_velocity,
                )

            if pending_commands is not None and pending_commands.shape[0] > 0:
                executed_input = pending_commands[0]
            else:
                executed_input = self.velocity_command
            self.velocity_response_compensator.advance(executed_input)
        else:
            if self.max_velocity > 0.0:
                self.velocity_desired.copy_(
                    torch.clamp(
                        self.velocity_raw,
                        min=-self.max_velocity,
                        max=self.max_velocity,
                    )
                )
            else:
                self.velocity_desired.copy_(self.velocity_raw)
            self.velocity_saturated.copy_(
                torch.abs(self.velocity_raw - self.velocity_desired)
                > self.saturation_tolerance
            )

            requested_action = self.velocity_desired - self.velocity_command
            self.last_action.copy_(
                torch.clamp(
                    requested_action,
                    min=-self.max_velocity_change,
                    max=self.max_velocity_change,
                )
            )
            self.acceleration_saturated.copy_(
                torch.abs(requested_action - self.last_action)
                > self.saturation_tolerance
            )
            self.velocity_command.add_(self.last_action)
            if self.max_velocity > 0.0:
                self.velocity_command.clamp_(
                    min=-self.max_velocity,
                    max=self.max_velocity,
                )

        self._update_integral()
        if not bool(torch.isfinite(self.last_action).all()):
            raise FloatingPointError("NFFBPolicy produced a non-finite action.")
        self.state_predictor.update_after_action(self.last_action)
        self.predictor_command_sync_error.copy_(
            self.velocity_command - self.state_predictor.command_z
        )
        if self.state_predictor.active and bool(
            torch.any(
                torch.abs(self.predictor_command_sync_error)
                > self.saturation_tolerance
            )
        ):
            raise RuntimeError(
                "NFFB velocity_command diverged from the state predictor command_z."
            )
        return self.last_action.clone()

    def _update_command_filter(self, theta: torch.Tensor, omega: torch.Tensor):
        init_mask = self.filter_needs_init.clone()
        active_mask = ~init_mask
        if torch.any(active_mask):
            omega_f = float(self.cfg.command_filter.natural_frequency)
            zeta_f = float(self.cfg.command_filter.damping_ratio)
            self.filter_alpha.copy_(
                omega_f**2 * (self.theta_star - self.theta_d)
                - 2.0 * zeta_f * omega_f * self.omega_d
            )
            omega_unclipped = self.omega_d + self.step_dt * self.filter_alpha
            omega_candidate = torch.clamp(
                omega_unclipped,
                min=-self.omega_max,
                max=self.omega_max,
            )
            theta_unclipped = self.theta_d + self.step_dt * omega_candidate
            theta_candidate = torch.clamp(
                theta_unclipped,
                min=-self.theta_max,
                max=self.theta_max,
            )
            hit_upper = (theta_candidate >= self.theta_max) & (omega_candidate > 0.0)
            hit_lower = (theta_candidate <= -self.theta_max) & (omega_candidate < 0.0)
            outward = hit_upper | hit_lower
            omega_candidate = torch.where(outward, torch.zeros_like(omega_candidate), omega_candidate)
            self.theta_d[active_mask] = theta_candidate[active_mask]
            self.omega_d[active_mask] = omega_candidate[active_mask]
            self.theta_command_saturated[active_mask] = (
                (torch.abs(theta_unclipped - theta_candidate) > self.saturation_tolerance)
                | outward
            )[active_mask]
            self.omega_command_saturated[active_mask] = (
                torch.abs(omega_unclipped - omega_candidate) > self.saturation_tolerance
            )[active_mask]

        if torch.any(init_mask):
            self.theta_d[init_mask] = torch.clamp(
                theta[init_mask],
                min=-self.theta_max,
                max=self.theta_max,
            )
            self.omega_d[init_mask] = torch.clamp(
                omega[init_mask],
                min=-self.omega_max,
                max=self.omega_max,
            )
            self.filter_alpha[init_mask] = 0.0
            self.theta_command_saturated[init_mask] = False
            self.omega_command_saturated[init_mask] = False
            self.filter_needs_init[init_mask] = False

    def _update_integral(self):
        if self.ki <= 0.0:
            self.integral_error.zero_()
            self.anti_windup_frozen.zero_()
            return

        upper_saturated = (
            self.ball_acceleration_raw > self.ball_acceleration_max + self.saturation_tolerance
        )
        lower_saturated = (
            self.ball_acceleration_raw < self.ball_acceleration_min - self.saturation_tolerance
        )
        pushes_outer_further = (upper_saturated & (self.position_error < 0.0)) | (
            lower_saturated & (self.position_error > 0.0)
        )
        freeze = (
            pushes_outer_further
            | self.theta_command_saturated
            | self.omega_command_saturated
            | self.velocity_saturated
            | self.acceleration_saturated
        )
        self.anti_windup_frozen.copy_(freeze)
        integral_delta = torch.where(freeze, torch.zeros_like(self.position_error), self.position_error)
        self.integral_error.add_(self.step_dt * integral_delta)
        integral_limit = float(self.cfg.outer_loop.integral_limit)
        if integral_limit > 0.0:
            self.integral_error.clamp_(min=-integral_limit, max=integral_limit)

    def get_state(self) -> dict[str, torch.Tensor]:
        """Return controller states and saturation diagnostics."""
        state_names = (
            "p_ref",
            "v_ref",
            "a_ref_raw",
            "a_ref_limited",
            "a_ref_used",
            "position_error",
            "velocity_error",
            "integral_error",
            "ball_acceleration_raw",
            "ball_acceleration_min",
            "ball_acceleration_max",
            "ball_acceleration_command",
            "theta_star",
            "theta_d",
            "omega_d",
            "filter_alpha",
            "theta_error",
            "omega_command",
            "beta",
            "velocity_raw",
            "velocity_desired",
            "velocity_command",
            "predictor_command_sync_error",
            "last_action",
            "reference_acceleration_saturated",
            "ball_acceleration_saturated",
            "theta_command_saturated",
            "omega_command_saturated",
            "velocity_saturated",
            "acceleration_saturated",
            "anti_windup_frozen",
        )
        state = {f"policy_{name}": getattr(self, name)[:, 0] for name in state_names}
        state["policy_reference_offset"] = torch.full(
            (self.num_envs,),
            float(self.reference_offset),
            device=self.device,
        )
        state.update(self.state_predictor.get_state())
        state.update(self.velocity_response_compensator.get_state())
        return state

    def get_config_info(self) -> dict[str, object]:
        """Return resolved scalar controller settings for run metadata."""
        return {
            "policy_name": self.cfg.name,
            "outer_kp": self.kp,
            "outer_kv": self.kv,
            "outer_ki": self.ki,
            "step_dt": self.step_dt,
            "max_velocity_change": self.max_velocity_change,
            "model_gamma": self.model.gamma,
            "model_effective_mass": self.model.effective_mass,
            "model": asdict(self.model.cfg),
            "outer_loop": asdict(self.cfg.outer_loop),
            "command_filter": asdict(self.cfg.command_filter),
            "inner_loop": asdict(self.cfg.inner_loop),
            "constraints": asdict(self.cfg.constraints),
            "state_predictor": asdict(self.cfg.state_predictor),
            "state_predictor_active": self.state_predictor.active,
            "state_predictor_reference_offset": self.reference_offset,
            "state_predictor_reference_fields": self._reference_field_names(),
            "velocity_response_compensation": asdict(
                self.cfg.velocity_response_compensation
            ),
            "velocity_response_compensation_active": (
                self.velocity_response_compensator.enabled
            ),
            "velocity_response_compensation_parameter_source": (
                self.velocity_response_compensator.parameter_source
            ),
        }

    def to(self, device: str | torch.device):
        """Move policy buffers to a device and return self."""
        device = torch.device(device)
        for name, value in vars(self).items():
            if isinstance(value, torch.Tensor):
                setattr(self, name, value.to(device=device))
        self.state_predictor.to(device)
        self.velocity_response_compensator.to(device)
        self.device = device
        return self

    def _field(self, observation: torch.Tensor, name: str) -> torch.Tensor:
        index = self._field_indices[name]
        return observation[:, index : index + 1]

    def _validate_observation_spec(self) -> dict[str, int]:
        if self.raw_observation_dim != len(self.raw_observation_fields):
            raise ValueError(
                "raw_observation_dim must match raw_observation_fields length: "
                f"{self.raw_observation_dim} != {len(self.raw_observation_fields)}."
            )
        if self.raw_observation_fields[:11] != LEGACY_OBSERVATION_FIELDS:
            raise ValueError("NFFBPolicy requires the unchanged 11-D legacy observation prefix.")
        if len(set(self.raw_observation_fields)) != len(self.raw_observation_fields):
            raise ValueError("raw_observation_fields must not contain duplicate names.")
        indices = {name: index for index, name in enumerate(self.raw_observation_fields)}
        required_fields = [
            *(f"pg_{offset}" for offset in range(1, max(self.reference_offset, 1) + 1)),
            f"vg_{self.reference_offset}",
            f"vg_{self.reference_offset + 1}",
        ]
        missing = [name for name in required_fields if name not in indices]
        if missing:
            raise ValueError(
                "NFFBPolicy requires reference preview horizon with future_steps >= "
                f"{self.reference_offset + 1}; missing {missing}."
            )
        return indices

    def _reference_field_names(self) -> list[str]:
        fields = ["pg"]
        if self.state_predictor.active:
            fields.extend(
                f"pg_{offset}"
                for offset in range(1, self.state_predictor.delay_step + 1)
            )
        fields.extend(
            (
                f"vg_{self.reference_offset}",
                f"vg_{self.reference_offset + 1}",
            )
        )
        return fields

    def _resolve_predictor_defaults(self):
        predictor = self.cfg.state_predictor
        if _is_auto(predictor.delay_step):
            predictor.delay_step = 0
        if _is_auto(predictor.step_dt) or float(predictor.step_dt) <= 0.0:
            predictor.step_dt = self.step_dt
        if _is_auto(predictor.max_acc) or float(predictor.max_acc) <= 0.0:
            predictor.max_acc = self.max_acc
        if _is_auto(predictor.max_velocity):
            predictor.max_velocity = self.max_velocity

        model_values = {
            "plank_length": self.cfg.model.plank_length,
            "rope_length": self.cfg.model.rope_length,
            "ball_position_offset": self.cfg.model.ball_position_offset,
            "gravity": self.cfg.model.gravity,
            "ball_mass": self.cfg.model.ball_mass,
            "ball_radius": self.cfg.model.ball_radius,
            "ball_inertia_ratio": self.cfg.model.ball_inertia_ratio,
            "epsilon": self.cfg.model.epsilon,
        }
        for name, value in model_values.items():
            if _is_auto(getattr(predictor, name)):
                setattr(predictor, name, value)

    def _resolve_velocity_response_compensation_defaults(self):
        compensation = self.cfg.velocity_response_compensation
        source = compensation.normalized_parameter_source
        compensation.parameter_source = source
        if source != "state_predictor":
            return

        predictor = self.cfg.state_predictor
        response_values = {
            "tau_s": predictor.velocity_response_tau_s,
            "gain": predictor.velocity_response_gain,
            "bias": predictor.velocity_response_bias,
            "max_abs_velocity": predictor.velocity_response_max_abs_velocity,
        }
        for name, value in response_values.items():
            setattr(compensation, name, value)

    def _validate_predictor_alignment(self):
        if not self.state_predictor.active:
            return

        expected_values = {
            "step_dt": self.step_dt,
            "max_acc": self.max_acc,
            "max_velocity": self.max_velocity,
            "plank_length": self.model.plank_length,
            "rope_length": self.model.rope_length,
            "ball_position_offset": self.model.ball_position_offset,
            "gravity": self.model.gravity,
            "ball_mass": self.model.ball_mass,
            "ball_radius": self.model.ball_radius,
            "ball_inertia_ratio": self.model.ball_inertia_ratio,
            "epsilon": self.model.epsilon,
        }
        for name, expected in expected_values.items():
            actual = float(getattr(self.state_predictor, name))
            if not math.isclose(
                actual,
                float(expected),
                rel_tol=1.0e-7,
                abs_tol=1.0e-9,
            ):
                raise ValueError(
                    f"NFFB state_predictor.{name} must match the resolved NFFB "
                    f"value; got {actual} != {float(expected)}."
                )

    def _validate_velocity_response_compensation_alignment(self):
        compensation = self.velocity_response_compensator
        predictor = self.state_predictor
        if (
            not compensation.enabled
            or not predictor.active
            or not predictor.velocity_response_enabled
        ):
            return

        expected_values = {
            "tau_s": predictor.velocity_response_tau_s,
            "gain": predictor.velocity_response_gain,
            "bias": predictor.velocity_response_bias,
            "max_abs_velocity": predictor.velocity_response_max_abs_velocity,
        }
        for name, expected in expected_values.items():
            actual = float(getattr(compensation, name))
            if not math.isclose(
                actual,
                float(expected),
                rel_tol=1.0e-7,
                abs_tol=1.0e-9,
            ):
                raise ValueError(
                    "NFFB velocity_response_compensation."
                    f"{name} must match the active state predictor response model; "
                    f"got {actual} != {float(expected)}."
                )

    def _resolve_auto_defaults(self):
        model_defaults = {
            "plank_length": 1.06,
            "rope_length": 0.9,
            "ball_position_offset": 0.33,
            "gravity": 9.81,
            "ball_mass": 0.0005,
            "ball_radius": 0.023,
        }
        for name, default in model_defaults.items():
            if _is_auto(getattr(self.cfg.model, name)):
                setattr(self.cfg.model, name, default)
        if _is_auto(self.cfg.constraints.max_acc):
            self.cfg.constraints.max_acc = 0.5
        if _is_auto(self.cfg.constraints.max_velocity):
            self.cfg.constraints.max_velocity = 0.0

    def _validate_config(self):
        outer = self.cfg.outer_loop
        command_filter = self.cfg.command_filter
        inner = self.cfg.inner_loop
        constraints = self.cfg.constraints
        positive_values = {
            "outer_loop.natural_frequency": outer.natural_frequency,
            "outer_loop.damping_ratio": outer.damping_ratio,
            "command_filter.natural_frequency": command_filter.natural_frequency,
            "command_filter.damping_ratio": command_filter.damping_ratio,
            "inner_loop.k_theta": inner.k_theta,
            "constraints.theta_max": constraints.theta_max,
            "constraints.omega_max": constraints.omega_max,
            "constraints.max_acc": constraints.max_acc,
        }
        for name, value in positive_values.items():
            value = float(value)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")
        if float(constraints.theta_max) >= 0.5 * math.pi:
            raise ValueError("constraints.theta_max must be smaller than pi/2.")
        non_negative_values = {
            "outer_loop.integral_pole": outer.integral_pole,
            "outer_loop.integral_limit": outer.integral_limit,
            "constraints.max_velocity": constraints.max_velocity,
        }
        for name, value in non_negative_values.items():
            value = float(value)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative.")
        if not _is_auto(constraints.max_reference_acceleration):
            value = float(constraints.max_reference_acceleration)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError("constraints.max_reference_acceleration must be positive or 'auto'.")


def _update_dataclass(target, values: Mapping | None):
    if not values:
        return
    unknown = sorted(key for key in values if not hasattr(target, key))
    if unknown:
        raise ValueError(
            f"Unknown NFFB configuration field(s) for {type(target).__name__}: {', '.join(unknown)}"
        )
    for key, value in values.items():
        setattr(target, key, value)


def _is_auto(value) -> bool:
    return isinstance(value, str) and value.lower() == "auto"
