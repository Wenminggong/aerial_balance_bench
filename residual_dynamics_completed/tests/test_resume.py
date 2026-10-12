"""Fault-injection test of actual training/resume entry point (tiny test budget)."""
import contextlib
import copy
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
from residual_dynamics_completed import train as trainer
from residual_dynamics_completed.config import ROOT, configuration
from residual_dynamics_completed.data import make_store, evaluation_references


class Resume(unittest.TestCase):
    def test_epoch_and_stage_boundary_resume(self):
        cfg = copy.deepcopy(configuration())
        cfg["batch_size"] = 8
        for stage in cfg["stages"]:
            stage.update(epochs=2, steps=1)
        real_save = trainer.save_checkpoint
        store = make_store("validation", 33)
        refs = evaluation_references(store, 16)

        def run(directory, stop_after=None):
            root = Path(directory)
            (root/"data/processed").mkdir(parents=True)
            shutil.copy2(ROOT/"data/processed/manifest.json", root/"data/processed/manifest.json")
            calls = 0

            def save(path, payload):
                nonlocal calls
                real_save(path, payload)
                calls += 1
                if calls == stop_after:
                    raise InterruptedError("Injected failure after durable epoch checkpoint")

            with contextlib.ExitStack() as stack:
                stack.enter_context(patch.object(trainer, "ROOT", root))
                stack.enter_context(patch.object(trainer, "configuration", return_value=cfg))
                stack.enter_context(patch.object(trainer, "runtime_sources", return_value={"test": "fixed"}))
                stack.enter_context(patch.object(trainer, "make_store", return_value=store))
                stack.enter_context(patch.object(trainer, "evaluation_references", return_value=refs))
                stack.enter_context(patch.object(trainer.subprocess, "check_output", return_value="test\n"))
                stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                if stop_after is not None:
                    with patch.object(trainer, "save_checkpoint", side_effect=save):
                        with self.assertRaises(InterruptedError):
                            trainer.train("mlp")
                    trainer.train("mlp", resume=True)
                else:
                    trainer.train("mlp")
            summary = json.loads((root/"runs/mlp_completed.json").read_text())
            path = root/summary["run"]
            return torch.load(path/"best.pt", weights_only=False), json.loads((path/"history.json").read_text())

        with tempfile.TemporaryDirectory() as directory:
            full, history = run(Path(directory)/"full")
            for boundary in (1, 2):
                resumed, resumed_history = run(Path(directory)/f"resume_{boundary}", boundary)
                self.assertEqual(history, resumed_history)
                self.assertEqual(full["best_score"], resumed["best_score"])
                for key,value in full["state_dict"].items():
                    torch.testing.assert_close(value, resumed["state_dict"][key], rtol=0, atol=0)

if __name__ == "__main__":
    unittest.main()
