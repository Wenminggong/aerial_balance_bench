"""Nonlinear ball--beam model helpers for the velocity command interface."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass
class VelocityInterfaceModelCfg:
    """Physical parameters used by the velocity-interface model."""

    plank_length: float = 1.06
    rope_length: float = 0.9
    ball_position_offset: float = 0.33
    gravity: float = 9.81
    ball_mass: float = 0.0005
    ball_radius: float = 0.023
    ball_inertia_ratio: float = 0.4
    epsilon: float = 1.0e-6


class VelocityInterfaceModel:
    """Vectorized nonlinear model shared by model-based velocity policies.

    The benchmark observation ``pb`` omits the slide/block geometry ahead of
    the active beam interval.  The paper model uses the physical coordinate
    measured along the complete beam, so all dynamics use
    ``pb_model = pb + ball_position_offset``.
    """

    def __init__(self, cfg: VelocityInterfaceModelCfg):
        self.cfg = cfg
        self.plank_length = float(cfg.plank_length)
        self.rope_length = float(cfg.rope_length)
        self.ball_position_offset = float(cfg.ball_position_offset)
        self.gravity = abs(float(cfg.gravity))
        self.ball_mass = float(cfg.ball_mass)
        self.ball_radius = float(cfg.ball_radius)
        self.ball_inertia_ratio = float(cfg.ball_inertia_ratio)
        self.epsilon = float(cfg.epsilon)
        self._validate()

    @property
    def effective_mass(self) -> float:
        """Return ``J_b / r_b^2 + m_b``."""
        ball_inertia = self.ball_inertia_ratio * self.ball_mass * self.ball_radius**2
        return ball_inertia / self.ball_radius**2 + self.ball_mass

    @property
    def gamma(self) -> float:
        """Return the rolling-mass ratio ``m_b / M_b``."""
        return self.ball_mass / self.effective_mass

    def model_ball_position(self, benchmark_pb: torch.Tensor) -> torch.Tensor:
        """Convert the benchmark beam coordinate to the paper-model coordinate."""
        return benchmark_pb + self.ball_position_offset

    def rope_angle(self, theta: torch.Tensor) -> torch.Tensor:
        """Return the rope angle ``beta(theta)`` with a safe arcsine domain."""
        sin_beta = (self.plank_length / self.rope_length) * (1.0 - torch.cos(theta))
        sin_beta = torch.clamp(sin_beta, min=-1.0 + self.epsilon, max=1.0 - self.epsilon)
        return torch.asin(sin_beta)

    def beam_angular_velocity(self, theta: torch.Tensor, vertical_velocity: torch.Tensor) -> torch.Tensor:
        """Map drone vertical velocity to beam angular velocity."""
        beta = self.rope_angle(theta)
        denominator = self.plank_length * self._safe_denominator(torch.cos(beta - theta))
        return -vertical_velocity * torch.cos(beta) / denominator

    def vertical_velocity(self, theta: torch.Tensor, angular_velocity: torch.Tensor) -> torch.Tensor:
        """Map a desired beam angular velocity to drone vertical velocity."""
        beta = self.rope_angle(theta)
        cos_beta = self._safe_denominator(torch.cos(beta))
        return -self.plank_length * angular_velocity * torch.cos(beta - theta) / cos_beta

    def ball_acceleration(
        self,
        benchmark_pb: torch.Tensor,
        theta: torch.Tensor,
        omega: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate the complete nonlinear rolling-ball acceleration."""
        pb_model = self.model_ball_position(benchmark_pb)
        centripetal = (pb_model - self.plank_length) * omega.square()
        return self.gamma * (centripetal - self.gravity * torch.sin(theta))

    def feasible_ball_acceleration(
        self,
        benchmark_pb: torch.Tensor,
        omega: torch.Tensor,
        theta_max: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the acceleration interval induced by ``|theta| <= theta_max``."""
        theta_max = float(theta_max)
        if not math.isfinite(theta_max) or theta_max <= 0.0 or theta_max >= 0.5 * math.pi:
            raise ValueError("theta_max must be finite and lie in (0, pi/2).")
        pb_model = self.model_ball_position(benchmark_pb)
        centripetal = (pb_model - self.plank_length) * omega.square()
        gravity_component = self.gravity * math.sin(theta_max)
        acceleration_min = self.gamma * (centripetal - gravity_component)
        acceleration_max = self.gamma * (centripetal + gravity_component)
        return acceleration_min, acceleration_max

    def desired_beam_angle(
        self,
        benchmark_pb: torch.Tensor,
        omega: torch.Tensor,
        desired_ball_acceleration: torch.Tensor,
        theta_max: float,
    ) -> torch.Tensor:
        """Invert the ball dynamics for a feasible desired beam angle."""
        theta_max = float(theta_max)
        if not math.isfinite(theta_max) or theta_max <= 0.0 or theta_max >= 0.5 * math.pi:
            raise ValueError("theta_max must be finite and lie in (0, pi/2).")
        pb_model = self.model_ball_position(benchmark_pb)
        q = (
            (pb_model - self.plank_length) * omega.square()
            - desired_ball_acceleration / self.gamma
        ) / self.gravity
        sin_limit = math.sin(theta_max)
        return torch.asin(torch.clamp(q, min=-sin_limit, max=sin_limit))

    def _safe_denominator(self, value: torch.Tensor) -> torch.Tensor:
        sign = torch.where(value >= 0.0, torch.ones_like(value), -torch.ones_like(value))
        return torch.where(torch.abs(value) < self.epsilon, sign * self.epsilon, value)

    def _validate(self):
        positive_fields = {
            "plank_length": self.plank_length,
            "rope_length": self.rope_length,
            "gravity": self.gravity,
            "ball_mass": self.ball_mass,
            "ball_radius": self.ball_radius,
            "epsilon": self.epsilon,
        }
        for name, value in positive_fields.items():
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")
        if not math.isfinite(self.ball_position_offset):
            raise ValueError("ball_position_offset must be finite.")
        if not math.isfinite(self.ball_inertia_ratio) or self.ball_inertia_ratio < 0.0:
            raise ValueError("ball_inertia_ratio must be finite and non-negative.")
        if self.epsilon >= 1.0:
            raise ValueError("epsilon must be smaller than 1.")
