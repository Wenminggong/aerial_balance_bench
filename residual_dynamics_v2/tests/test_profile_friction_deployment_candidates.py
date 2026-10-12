from __future__ import annotations

from pathlib import Path

import pytest
import torch

from residual_dynamics_v2.profile_friction_deployment_candidates import (
    CANDIDATES,
    FORMULA_VERSION,
)
from residual_dynamics_v2.profile_friction_probe import (
    ProbeSettings,
    ProfileFrictionResidual,
)


RESULTS = Path(__file__).resolve().parents[1] / "results_profile_friction_probe"


def test_deployment_entrypoint_selects_only_the_two_approved_candidates() -> None:
    assert CANDIDATES == {
        "profile_only": {"profile_enabled": True, "friction_enabled": False},
        "profile_plus_friction": {
            "profile_enabled": True,
            "friction_enabled": True,
        },
    }


@pytest.mark.parametrize(
    ("candidate", "friction_enabled"),
    [("profile_only", False), ("profile_plus_friction", True)],
)
def test_frozen_checkpoint_reloads_and_is_finite(
    candidate: str, friction_enabled: bool
) -> None:
    checkpoint = torch.load(
        RESULTS / "checkpoints" / f"{candidate}.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert checkpoint["candidate"] == candidate
    assert checkpoint["formula_version"] == FORMULA_VERSION
    assert checkpoint["profile_enabled"] is True
    assert checkpoint["friction_enabled"] is friction_enabled
    assert len(checkpoint["state_dict"]["delta_coefficients"]) == 13
    model = ProfileFrictionResidual(
        profile_enabled=True,
        friction_enabled=friction_enabled,
        blackbox_enabled=False,
        kappa_enabled=False,
        acceleration_gain=checkpoint["acceleration_gain"],
        gravity_mps2=checkpoint["gravity_mps2"],
        settings=ProbeSettings(**checkpoint["settings"]),
    )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    output = model(
        torch.tensor([[0.15, 0.1, -0.04, 0.15, 0.3]], dtype=torch.float64)
    )
    assert output.shape == (1,)
    assert torch.isfinite(output).all()
    if not friction_enabled:
        components = model.components(
            torch.tensor([[0.15, 0.1, -0.04, 0.15, 0.3]], dtype=torch.float64)
        )
        assert torch.equal(components["friction"], torch.zeros_like(output))
