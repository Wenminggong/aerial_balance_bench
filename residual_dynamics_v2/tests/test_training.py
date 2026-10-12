from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from residual_dynamics_v2.config import load_config
from residual_dynamics_v2.model import create_model
from residual_dynamics_v2.training import (
    WindowReference,
    WindowStore,
    integral_errors,
    should_select_stage2,
    trapezoidal_average,
)


CONFIG = Path(__file__).resolve().parents[1] / "configs" / "rbf_2input_causal_success.yaml"


def fake_manifest() -> dict:
    return {
        "dataset_hash": "test",
        "normalization": {
            "mean": [0.35, 0.0],
            "std": [0.15, 0.15],
            "lower_quantile": [0.0, -0.5],
            "upper_quantile": [0.7, 0.5],
            "inverse_correlation": [[1.0, 0.0], [0.0, 1.0]],
            "mahalanobis_threshold": 4.0,
            "decay_width_std": 1.0,
            "min_weight": 0.0,
            "effective_cap_mps2": 0.25,
            "state_scale_names": ["pb", "vb", "theta", "vrz"],
            "state_scales": [0.01, 0.01, 0.01, 0.01],
        },
    }


def test_trapezoidal_average_constant() -> None:
    values = torch.full((2, 13), 0.4, dtype=torch.float64)
    dt = torch.full((2, 12), 1.0 / 60.0, dtype=torch.float64)
    torch.testing.assert_close(trapezoidal_average(values, dt, 12), torch.full((2,), 0.4, dtype=torch.float64))


def test_zero_residual_integral_target_is_exact() -> None:
    config = load_config(CONFIG)
    model = create_model(config, fake_manifest())
    batch_size, length = 4, 34
    dt = torch.full((batch_size, length - 1), 1.0 / 60.0, dtype=torch.float64)
    a_nom = 0.2
    smooth_vb = torch.arange(length, dtype=torch.float64)[None, :].repeat(batch_size, 1) * a_nom / 60.0
    features = torch.zeros((batch_size, length, 5), dtype=torch.float64)
    features[..., 0] = 0.35
    features[..., 4] = a_nom
    batch = SimpleNamespace(features=features, dt=dt, smooth_vb=smooth_vb)
    errors = integral_errors(model, batch, config)
    for model_error, nominal_error in errors.values():
        torch.testing.assert_close(model_error, torch.zeros_like(model_error), atol=1e-14, rtol=0)
        torch.testing.assert_close(nominal_error, torch.zeros_like(nominal_error), atol=1e-14, rtol=0)


def test_three_stream_sampler_is_reproducible() -> None:
    config = load_config(CONFIG)
    store = WindowStore.__new__(WindowStore)
    store.config = config
    store.references = [WindowReference(0, index, 0, 0, 0) for index in range(10)]
    trial = SimpleNamespace(session="s", group="g")
    store.trials = [trial]
    hierarchy = {"s": {"g": store.references}}
    store.by_joint = {(0, 0): hierarchy}
    store.by_actuation = {(0, 0): {0: hierarchy}}
    first = store.sample(np.random.default_rng(7), 20)
    second = store.sample(np.random.default_rng(7), 20)
    assert first == second
    assert len(first) == 20


def test_stage2_selection_falls_back_on_unsafe_checkpoint() -> None:
    config = load_config(CONFIG)
    safe = {
        "selection_score": 0.9,
        "residual_saturation_rate": 0.0,
        "rp_min": -0.2,
        "rp_max": 0.2,
        "rv_min": -0.1,
        "rv_max": 0.05,
    }
    assert should_select_stage2(1.0, safe, config)
    unsafe = dict(safe, rv_max=0.3)
    assert not should_select_stage2(1.0, unsafe, config)
    assert not should_select_stage2(0.8, safe, config)
