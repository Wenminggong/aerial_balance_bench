from __future__ import annotations

import numpy as np
import torch

from residual_dynamics_v2.profile_friction_detailed_analysis import (
    _add_error,
    _finalize_total,
    _new_total,
    _recover_coefficients,
)
from residual_dynamics_v2.profile_friction_probe import (
    ProbeSettings,
    ProfileFrictionResidual,
)


def _profile_model() -> ProfileFrictionResidual:
    return ProfileFrictionResidual(
        profile_enabled=True,
        friction_enabled=False,
        acceleration_gain=5.0 / 7.0,
        gravity_mps2=9.81,
        settings=ProbeSettings(),
    )


def test_dense_curve_recovers_frozen_probe_coefficients() -> None:
    model = _profile_model()
    expected = torch.linspace(-0.02, 0.025, 13, dtype=torch.float64)
    position = torch.linspace(0.0, 0.7, 141, dtype=torch.float64)
    sampled = model.basis(position) @ expected
    recovered = _recover_coefficients(model, position.tolist(), sampled.tolist())
    np.testing.assert_allclose(recovered.numpy(), expected.numpy(), atol=1.0e-11)


def test_rollout_total_reports_rmse_and_mae() -> None:
    total = _new_total()
    _add_error(total, -1.0, 2.0)
    _add_error(total, 3.0, -4.0)
    result = _finalize_total(total)
    assert result["count"] == 2
    assert np.isclose(result["pb_rmse_m"], np.sqrt(5.0))
    assert np.isclose(result["vb_rmse_mps"], np.sqrt(10.0))
    assert result["pb_mae_m"] == 2.0
    assert result["vb_mae_mps"] == 3.0
