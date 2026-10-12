"""Uniform five-feature to scalar-acceleration interface for six models."""
from __future__ import annotations
import torch
from torch import nn
from .structured import ProfileFrictionResidual, ProbeSettings
from .mlp import ResidualMLP


class Nominal(nn.Module):
    def forward(self, features):
        return torch.zeros_like(features[..., 0])


class RBFPV(nn.Module):
    def __init__(self):
        super().__init__()
        p = torch.linspace(0., .7, 9, dtype=torch.float64)
        v = torch.tensor([-.5, -.3, -.15, -.06, 0., .06, .15, .3, .5], dtype=torch.float64)
        pp, vv = torch.meshgrid(p, v, indexing="ij")
        self.register_buffer("centers", torch.stack((pp.flatten(), vv.flatten()), -1))
        self.coefficients = nn.Parameter(torch.zeros(81, dtype=torch.float64))

    def raw(self, features):
        delta = (features[..., None, :2] - self.centers) / .1
        phi = torch.exp(-.5 * delta.square().sum(-1))
        phi = phi / (1e-12 + phi.sum(-1, keepdim=True))
        return phi @ self.coefficients

    def forward(self, features):
        return .25 * torch.tanh(self.raw(features) / .25)

    def project(self):
        pass  # Deliberately no OOD gate and no hard coefficient clipping.

    def regularization(self):
        grid = self.coefficients.reshape(9, 9) / .25
        return (.002 * ((grid[1:] - grid[:-1]).square().mean()
                        + (grid[:, 1:] - grid[:, :-1]).square().mean())
                + .01 * ((grid[2:] - 2*grid[1:-1] + grid[:-2]).square().mean()
                         + (grid[:, 2:] - 2*grid[:, 1:-1] + grid[:, :-2]).square().mean()))


def create_model(name, manifest):
    if name == "nominal":
        return Nominal()
    if name == "rbf_pv":
        return RBFPV()
    if name == "mlp":
        norm = manifest["shadow_full_state_normalization"]
        return ResidualMLP(norm["mean"], norm["std"])
    if name not in ("structured", "profile_only", "friction_only"):
        raise ValueError(name)
    constants = next(e["constants"] for e in manifest["entries"] if e.get("included"))
    return ProfileFrictionResidual(
        profile_enabled=name != "friction_only", friction_enabled=name != "profile_only",
        acceleration_gain=constants["acceleration_gain"], gravity_mps2=constants["gravity_mps2"],
        settings=ProbeSettings())
