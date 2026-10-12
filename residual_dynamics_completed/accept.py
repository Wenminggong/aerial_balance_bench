"""Recompute exported errors/aggregates and enforce phase-one acceptance."""
from __future__ import annotations
import csv
import json
import subprocess
import sys
import numpy as np
import torch
from .config import ROOT, MODEL_NAMES
from .data import sha256_file
from .evaluate import load_model, aggregate
from .train import seed_all


def accept():
    seed_all()
    out = ROOT / "results/test"
    metadata = json.loads((out / "metadata.json").read_text())
    for path, digest in metadata["artifact_hashes"].items():
        assert sha256_file(out / path) == digest, path
    with (out / "window_errors.csv").open() as stream:
        errors = list(csv.DictReader(stream))
    with (out / "metrics_long.csv").open() as stream:
        saved = list(csv.DictReader(stream))
    recomputed = aggregate(errors)
    for actual, expected in zip(recomputed, saved, strict=True):
        for key, value in actual.items():
            if isinstance(value, (int, float)):
                assert np.isfinite(float(expected[key]))
                np.testing.assert_allclose(value, float(expected[key]), rtol=1e-13, atol=1e-15)
            else:
                assert value == expected[key]
    target = np.load(out / "common_inputs.npz")["measured_states"]
    for name in MODEL_NAMES:
        predictions = np.load(out / f"{name}_predictions.npz")["states"]
        np.testing.assert_array_equal(predictions[:, 0], target[:, 0])
        assert predictions.shape == (2028, 61, 4)
        rows = [r for r in errors if r["model"] == name]
        assert len(rows) == 2028*10
        assert len({(r["window_id"],r["horizon"]) for r in rows}) == 2028*10
        for row in rows:
            i, h = int(row["window_id"]), int(row["horizon"])
            assert float(row["position_error_m"]) == predictions[i,h,0]-target[i,h,0]
            assert float(row["velocity_error_mps"]) == predictions[i,h,1]-target[i,h,1]
        assert np.isfinite(predictions).all()
        assert metadata["diagnostics"][name]["nonfinite_stage_residuals"] == 0
    summaries = [metadata["models"][n] for n in MODEL_NAMES if n != "nominal"]
    for key in ("train_window_hash", "validation_window_hash", "manifest_sha256", "config_hash", "sources"):
        assert all(s["metadata"][key] == summaries[0]["metadata"][key] for s in summaries), key
    model, new = load_model("structured")
    old_path = ROOT / "provenance/old_structured/checkpoints/profile_plus_friction.pt"
    old = torch.load(old_path, weights_only=False)
    differences = {key: float(torch.max(torch.abs(value-old["state_dict"][key])))
                   for key,value in model.state_dict().items()}
    exact = all(v == 0 for v in differences.values())
    assert new["best_validation_score"] == old["training"]["best_validation_score"]
    history = json.loads((ROOT / new["run"] / "history.json").read_text())
    history_equal = history == old["training"]["history"]
    history_max_difference = 0.
    assert len(history) == len(old["training"]["history"])
    for new_row, old_row in zip(history, old["training"]["history"]):
        assert new_row.keys() == old_row.keys()
        for key,value in new_row.items():
            if isinstance(value, (int,float)):
                history_max_difference = max(history_max_difference, abs(value-old_row[key]))
            else:
                assert value == old_row[key]
    # Reproduction is numerical, not a claim about byte-identical serialization.
    assert max(differences.values()) < 1e-12 and history_max_difference < 1e-12
    tests = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", str(ROOT/"tests"), "-v"],
                           text=True, capture_output=True, cwd=ROOT.parent)
    (ROOT / "results/contract_tests.txt").write_text(tests.stdout+tests.stderr)
    assert tests.returncode == 0, tests.stderr
    parity = json.loads((ROOT / "provenance/parity.json").read_text())
    assert all(parity[k] for k in ("forward_bitwise_equal", "sampling_bitwise_equal", "batch_bitwise_equal", "loss_and_gradients_bitwise_equal"))
    result = {"passed": True, "models": list(MODEL_NAMES), "test_windows_per_model": 2028,
        "window_error_rows": len(errors), "aggregates_recomputed": len(recomputed),
        "common_training_and_validation_windows": True, "stored_predictions_match_errors": True,
        "nonfinite_predictions": 0, "contract_tests_passed": True,
        "structured_state_dict_bitwise_equal": exact, "structured_history_bitwise_equal": history_equal,
        "structured_numerically_reproduced": True, "reproduction_absolute_tolerance": 1e-12,
        "structured_history_max_absolute_difference": history_max_difference,
        "structured_parameter_max_differences": differences, "old_structured_checkpoint_sha256": sha256_file(old_path),
        "new_structured_checkpoint_sha256": new["checkpoint_sha256"],
        "checkpoint_container_note": "Different checkpoint containers; tensors and history agree within floating-point precision, not bitwise.",
        "migration_parity": parity, "test_metadata_sha256": sha256_file(out/"metadata.json")}
    (ROOT / "results/acceptance.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    accept()
