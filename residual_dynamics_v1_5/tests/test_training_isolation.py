from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch

from residual_dynamics_v1_5.config import load_config
from residual_dynamics_v1_5.data import PhysicalConstants, TrialData
from residual_dynamics_v1_5.model import create_model
from residual_dynamics_v1_5.training import WindowStore, compute_deadzone_loss


CONFIG = Path("residual_dynamics_v1_5/configs/mlp_deadzone.yaml")


def _trial() -> TrialData:
    size = 80
    features = np.zeros((size, 5), dtype=np.float64)
    features[:, 0] = 0.35
    features[:40, 1] = 0.05
    features[40:, 1] = 0.0
    features[40:, 4] = 0.08
    dynamic = np.zeros(size, dtype=bool); dynamic[:40] = True
    static = np.zeros(size, dtype=bool); static[40:] = True
    valid = np.ones(size, dtype=bool)
    arrays = {
        "time_s": np.arange(size, dtype=np.float64) / 60.0,
        "features": features, "residual_target": np.zeros(size),
        "pb": features[:, 0], "vb": features[:, 1], "ab": np.zeros(size),
        "theta": np.zeros(size), "omega": np.zeros(size),
        "dt_s": np.full(size, 1.0 / 60.0), "dynamic_mask": dynamic,
        "static_mask": static, "transition_mask": np.zeros(size, dtype=bool),
        "valid_mask": valid, "segment_id": np.zeros(size, dtype=np.int32),
    }
    return TrialData("trial", "synthetic", "fit", "train", "constant", "none", arrays,
                     PhysicalConstants(1.06, 0.9, 0.33, 9.81, 1.0 / 1.4))


def test_stage_a_window_store_excludes_every_static_sample() -> None:
    store = WindowStore([_trial()], horizon=10, dynamic_only=True)
    assert store.references
    for ref in store.references:
        masks = store.trials[ref.trial_index].arrays
        assert masks["dynamic_mask"][ref.start:ref.start + 11].all()
        assert not masks["static_mask"][ref.start:ref.start + 11].any()


def test_stage_b_loss_changes_only_deadzone_when_mlp_is_frozen() -> None:
    config = load_config(CONFIG)
    model = create_model(config.model, {"mean": np.zeros(5), "std": np.ones(5)})
    model.set_mlp_trainable(False)
    store = WindowStore([_trial()], horizon=30, dynamic_only=False)
    static_refs = [ref for ref in store.references if ref.start >= 40][:4]
    batch = store.make_batch(static_refs, torch.device("cpu"))
    before = _hash(model.mlp)
    optimizer = torch.optim.Adam([model.raw_deadzone_threshold], lr=0.01)
    initial_q = model.raw_deadzone_threshold.detach().clone()
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        loss, _ = compute_deadzone_loss(model, batch, config)
        loss.backward(); optimizer.step()
    assert _hash(model.mlp) == before
    assert not torch.equal(model.raw_deadzone_threshold.detach(), initial_q)
    assert all(parameter.grad is None for parameter in model.mlp.parameters())


def _hash(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode()); digest.update(value.detach().numpy().tobytes())
    return digest.hexdigest()
