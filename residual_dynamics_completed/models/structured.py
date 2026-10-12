"""Physical residual, migrated with checkpoint-compatible parameter names."""
from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import nn

@dataclass(frozen=True)
class ProbeSettings:
    seed: int = 7
    center_count: int = 13
    width_m: float = 0.065
    velocity_transition_mps: float = 0.025
    delta_limit_rad: float = 0.035
    coulomb_limit_mps2: float = 0.15
    viscous_limit_sinv: float = 0.50
    blackbox_limit_mps2: float = 0.25
    kappa_limit: float = 0.10
    output_limit_mps2: float = 0.25
    stage1_epochs: int = 80
    stage1_steps: int = 16
    stage2_epochs: int = 50
    stage2_steps: int = 8
    batch_size: int = 256
    validation_windows: int = 512


class ProfileFrictionResidual(nn.Module):
    """Bounded local-profile term plus optional structured or black-box residual."""

    def __init__(
        self,
        *,
        profile_enabled: bool,
        friction_enabled: bool,
        blackbox_enabled: bool = False,
        kappa_enabled: bool = False,
        acceleration_gain: float,
        gravity_mps2: float,
        settings: ProbeSettings,
    ) -> None:
        super().__init__()
        self.profile_enabled = bool(profile_enabled)
        self.friction_enabled = bool(friction_enabled)
        self.blackbox_enabled = bool(blackbox_enabled)
        self.kappa_enabled = bool(kappa_enabled)
        self.acceleration_gain = float(acceleration_gain)
        self.gravity_mps2 = float(gravity_mps2)
        self.settings = settings
        self.output_limit_mps2 = settings.output_limit_mps2
        self.register_buffer(
            "centers",
            torch.linspace(0.0, 0.7, settings.center_count, dtype=torch.float64),
        )
        blackbox_pb = torch.linspace(0.0, 0.7, 9, dtype=torch.float64)
        blackbox_vb = torch.tensor(
            [-0.50, -0.30, -0.15, -0.06, 0.0, 0.06, 0.15, 0.30, 0.50],
            dtype=torch.float64,
        )
        grid_pb, grid_vb = torch.meshgrid(blackbox_pb, blackbox_vb, indexing="ij")
        self.register_buffer(
            "blackbox_centers",
            torch.stack((grid_pb.reshape(-1), grid_vb.reshape(-1)), dim=-1),
        )
        self.delta_coefficients = nn.Parameter(
            torch.zeros(settings.center_count, dtype=torch.float64),
            requires_grad=self.profile_enabled,
        )
        self.coulomb_coefficients = nn.Parameter(
            torch.zeros(settings.center_count, dtype=torch.float64),
            requires_grad=self.friction_enabled,
        )
        self.viscous_coefficients = nn.Parameter(
            torch.zeros(settings.center_count, dtype=torch.float64),
            requires_grad=self.friction_enabled,
        )
        self.blackbox_coefficients = nn.Parameter(
            torch.zeros(81, dtype=torch.float64),
            requires_grad=self.blackbox_enabled,
        )
        self.kappa_coefficients = nn.Parameter(
            torch.zeros(81, dtype=torch.float64),
            requires_grad=self.kappa_enabled,
        )

    def basis(self, position: torch.Tensor) -> torch.Tensor:
        phi = torch.exp(
            -0.5
            * ((position[..., None] - self.centers) / self.settings.width_m).square()
        )
        return phi / (1.0e-12 + phi.sum(dim=-1, keepdim=True))

    def blackbox_basis(
        self, position: torch.Tensor, velocity: torch.Tensor
    ) -> torch.Tensor:
        delta_p = (position[..., None] - self.blackbox_centers[:, 0]) / 0.10
        delta_v = (velocity[..., None] - self.blackbox_centers[:, 1]) / 0.10
        phi = torch.exp(-0.5 * (delta_p.square() + delta_v.square()))
        return phi / (1.0e-12 + phi.sum(dim=-1, keepdim=True))

    def components(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        position, velocity, theta, a_nom = (
            features[..., 0],
            features[..., 1],
            features[..., 2],
            features[..., 4],
        )
        basis = self.basis(position)
        delta = basis @ self.delta_coefficients
        profile = -self.acceleration_gain * self.gravity_mps2 * (
            torch.sin(theta + delta) - torch.sin(theta)
        )
        coulomb = basis @ self.coulomb_coefficients
        viscous = basis @ self.viscous_coefficients
        friction = -coulomb * torch.tanh(
            velocity / self.settings.velocity_transition_mps
        ) - viscous * velocity
        if self.blackbox_enabled or self.kappa_enabled:
            blackbox_basis = self.blackbox_basis(position, velocity)
            blackbox = blackbox_basis @ self.blackbox_coefficients
            kappa = blackbox_basis @ self.kappa_coefficients
        else:
            # Keep dormant state-dict entries for exact legacy checkpoint loading.
            blackbox = torch.zeros_like(position)
            kappa = torch.zeros_like(position)
        gain_residual = kappa * a_nom
        if not self.profile_enabled:
            profile = torch.zeros_like(profile)
            delta = torch.zeros_like(delta)
        if not self.friction_enabled:
            friction = torch.zeros_like(friction)
            coulomb = torch.zeros_like(coulomb)
            viscous = torch.zeros_like(viscous)
        if not self.blackbox_enabled:
            blackbox = torch.zeros_like(blackbox)
        if not self.kappa_enabled:
            kappa = torch.zeros_like(kappa)
            gain_residual = torch.zeros_like(gain_residual)
        raw = profile + friction + blackbox + gain_residual
        applied = self.output_limit_mps2 * torch.tanh(raw / self.output_limit_mps2)
        return {
            "applied": applied,
            "raw": raw,
            "profile": profile,
            "friction": friction,
            "blackbox": blackbox,
            "kappa": kappa,
            "gain_residual": gain_residual,
            "delta": delta,
            "coulomb": coulomb,
            "viscous": viscous,
        }

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.components(features)["applied"]

    @torch.no_grad()
    def project(self) -> None:
        self.delta_coefficients.clamp_(
            -self.settings.delta_limit_rad, self.settings.delta_limit_rad
        )
        self.coulomb_coefficients.clamp_(0.0, self.settings.coulomb_limit_mps2)
        self.viscous_coefficients.clamp_(0.0, self.settings.viscous_limit_sinv)
        self.blackbox_coefficients.clamp_(
            -self.settings.blackbox_limit_mps2,
            self.settings.blackbox_limit_mps2,
        )
        self.kappa_coefficients.clamp_(
            -self.settings.kappa_limit,
            self.settings.kappa_limit,
        )

    def regularization(self) -> torch.Tensor:
        terms = []
        for values, scale in (
            (self.delta_coefficients, self.settings.delta_limit_rad),
            (self.coulomb_coefficients, self.settings.coulomb_limit_mps2),
            (self.viscous_coefficients, self.settings.viscous_limit_sinv),
        ):
            if values.requires_grad:
                normalized = values / scale
                first = normalized[1:] - normalized[:-1]
                second = normalized[2:] - 2.0 * normalized[1:-1] + normalized[:-2]
                terms.append(0.002 * first.square().mean())
                terms.append(0.01 * second.square().mean())
        if self.blackbox_coefficients.requires_grad:
            grid = self.blackbox_coefficients.reshape(9, 9) / self.settings.blackbox_limit_mps2
            first_p = grid[1:, :] - grid[:-1, :]
            first_v = grid[:, 1:] - grid[:, :-1]
            second_p = grid[2:, :] - 2.0 * grid[1:-1, :] + grid[:-2, :]
            second_v = grid[:, 2:] - 2.0 * grid[:, 1:-1] + grid[:, :-2]
            terms.extend(
                (
                    0.002 * first_p.square().mean(),
                    0.002 * first_v.square().mean(),
                    0.01 * second_p.square().mean(),
                    0.01 * second_v.square().mean(),
                )
            )
        if self.kappa_coefficients.requires_grad:
            grid = self.kappa_coefficients.reshape(9, 9) / self.settings.kappa_limit
            first_p = grid[1:, :] - grid[:-1, :]
            first_v = grid[:, 1:] - grid[:, :-1]
            second_p = grid[2:, :] - 2.0 * grid[1:-1, :] + grid[:-2, :]
            second_v = grid[:, 2:] - 2.0 * grid[:, 1:-1] + grid[:, :-2]
            terms.extend(
                (
                    0.002 * first_p.square().mean(),
                    0.002 * first_v.square().mean(),
                    0.01 * second_p.square().mean(),
                    0.01 * second_v.square().mean(),
                )
            )
        if not terms:
            return torch.zeros((), dtype=torch.float64)
        return torch.stack(terms).sum()
