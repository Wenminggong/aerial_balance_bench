"""Offline acceptance contracts; requires no predecessor package or raw logs."""
import ast
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from residual_dynamics_completed.config import ROOT, MODEL_NAMES
from residual_dynamics_completed.data import load_manifest, load_trials, make_store, evaluation_references, sha256_file
from residual_dynamics_completed.models import create_model
from residual_dynamics_completed.dynamics import rollout_causal_dynamics


class Contracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.manifest = load_manifest()

    def test_copies_and_splits(self):
        record = json.loads((ROOT / "provenance/migration.json").read_text())
        for item in record["files"]:
            path = ROOT / "data/processed" / item["file"]
            self.assertFalse(path.is_symlink())
            self.assertEqual(sha256_file(path), item["sha256"])
        splits = [load_trials(s) for s in ("train", "validation", "test", "stress")]
        self.assertEqual([len(s) for s in splits], [79, 6, 6, 5])
        keys = [t.trial_uid for split in splits for t in split]
        self.assertEqual(len(keys), len(set(keys)))
        for split in splits[:3]:
            self.assertEqual({t.reference_type for t in split}, {"constant", "sine", "triangle", "trapezoid"})

    def test_train_only_normalization(self):
        values = np.concatenate([t.arrays["features"] for t in load_trials("train")])
        norm = self.manifest["shadow_full_state_normalization"]
        np.testing.assert_array_equal(values.mean(0), norm["mean"])
        np.testing.assert_array_equal(values.std(0), norm["std"])
        self.assertEqual(norm["sample_count"], len(values))

    def test_runtime_import_independence(self):
        for path in list(ROOT.glob("*.py")) + list((ROOT/"models").glob("*.py")):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    self.assertNotIn((node.module or "").split(".")[0], ("residual_dynamics", "residual_dynamics_v2"))
                if isinstance(node, ast.Import):
                    for name in node.names:
                        self.assertNotIn(name.name.split(".")[0], ("residual_dynamics", "residual_dynamics_v2"))

    def test_bounded_smooth_and_checkpoint(self):
        torch.manual_seed(71)
        for name in MODEL_NAMES:
            model = create_model(name, self.manifest)
            with torch.no_grad():
                for p in model.parameters():
                    if p.requires_grad:
                        p.uniform_(-.2, .2)
            if name != "nominal":
                model.project()
            x = torch.randn(512, 5, dtype=torch.float64, requires_grad=True)
            y = model(x)
            self.assertTrue(torch.isfinite(y).all())
            self.assertTrue((y.abs() <= .25).all())
            if name != "nominal":
                grad = torch.autograd.grad(y.sum(), x)[0]
                self.assertTrue(torch.isfinite(grad).all())
            if name in ("structured", "profile_only", "friction_only"):
                expected = .25*torch.tanh(model.components(x)["raw"]/.25)
                torch.testing.assert_close(y, expected, rtol=0, atol=0)
            elif name == "mlp":
                z = model.layers((x-model.input_mean)/model.input_std).squeeze(-1)
                torch.testing.assert_close(y, .25*torch.tanh(z), rtol=0, atol=0)
            elif name == "rbf_pv":
                torch.testing.assert_close(y, .25*torch.tanh(model.raw(x)/.25), rtol=0, atol=0)
            if name == "friction_only":
                self.assertTrue((y*x[:, 1] <= 1e-15).all())
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory)/"weights.pt"
                torch.save(model.state_dict(), path)
                clone = create_model(name, self.manifest)
                clone.load_state_dict(torch.load(path, weights_only=True))
                torch.testing.assert_close(model(x), clone(x), rtol=0, atol=0)

    def test_windows_and_zero_rollout(self):
        store = make_store("test", 60)
        refs = evaluation_references(store)
        self.assertEqual(len(refs), 2028)
        self.assertEqual(len(evaluation_references(make_store("validation", 33), 512)), 512)
        batch = store.make_batch(refs[:8], torch.device("cpu"))
        args = (batch.causal_state[:, 0], batch.commands, batch.dt, batch.constants, batch.geometry, batch.response)
        expected = rollout_causal_dynamics(None, *args).states
        for name in MODEL_NAMES:
            model = create_model(name, self.manifest)
            actual = rollout_causal_dynamics(model, *args).states
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
