"""Optional migration audit: explicitly compares to read-only predecessor code."""
from dataclasses import asdict
import json
import numpy as np
import torch
from residual_dynamics_v2.config import load_config as old_config
from residual_dynamics_v2.data import load_manifest as old_manifest, load_trials as old_trials
from residual_dynamics_v2.training import WindowStore as OldStore
from residual_dynamics_v2.profile_friction_probe import ProfileFrictionResidual, ProbeSettings, _loss as old_loss
from residual_dynamics_completed.config import ROOT, load_config
from residual_dynamics_completed.data import make_store, load_manifest
from residual_dynamics_completed.models import create_model
from residual_dynamics_completed.losses import _loss

def main():
    torch.set_num_threads(1)
    c = old_config(ROOT.parent / "residual_dynamics_v2/configs/rbf_2input_causal_success.yaml")
    old = OldStore(old_trials(c, "train"), 33, c, old_manifest(c))
    new = make_store("train", 33)
    assert [asdict(r) for r in old.references] == [asdict(r) for r in new.references]
    a = old.sample(np.random.default_rng(7), 256)
    b = new.sample(np.random.default_rng(7), 256)
    assert [asdict(r) for r in a] == [asdict(r) for r in b]
    ba, bb = old.make_batch(a, torch.device("cpu")), new.make_batch(b, torch.device("cpu"))
    for key, value in vars(ba).items():
        if isinstance(value, torch.Tensor):
            assert torch.equal(value, getattr(bb, key)), key
    checkpoint = torch.load(ROOT / "provenance/old_structured/checkpoints/profile_plus_friction.pt", weights_only=False)
    m = create_model("structured", load_manifest())
    m.load_state_dict(checkpoint["state_dict"])
    legacy = ProfileFrictionResidual(profile_enabled=True, friction_enabled=True,
        acceleration_gain=m.acceleration_gain, gravity_mps2=m.gravity_mps2, settings=ProbeSettings())
    legacy.load_state_dict(checkpoint["state_dict"])
    assert torch.equal(m(bb.features), legacy(ba.features))
    results = {}
    for weight in (0., .5):
        x, y = old_loss(legacy, ba, c, rollout_weight=weight), _loss(m, bb, load_config(), rollout_weight=weight)
        assert torch.equal(x, y), (x.item(), y.item())
        x.backward(); y.backward()
        for p, q in zip(legacy.parameters(), m.parameters()):
            if p.grad is not None:
                assert torch.equal(p.grad, q.grad)
        legacy.zero_grad(); m.zero_grad()
        results[str(weight)] = x.item()
    result = {"forward_bitwise_equal": True, "sampling_bitwise_equal": True,
              "batch_bitwise_equal": True, "loss_and_gradients_bitwise_equal": True,
              "losses": results, "train_windows": len(new.references),
              "old_best_validation_score": checkpoint["training"]["best_validation_score"]}
    (ROOT / "provenance/parity.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result), flush=True)

if __name__ == "__main__":
    main()
