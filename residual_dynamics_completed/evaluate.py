"""One causal 60-step rollout per shared window, with no state resets."""
from __future__ import annotations
import argparse
import csv
import json
from dataclasses import fields
import numpy as np
import torch
from .config import ROOT, MODEL_NAMES, configuration, config_hash
from .data import make_store, load_manifest, evaluation_references, window_hash, window_records, sha256_file
from .dynamics import rollout_causal_dynamics, causal_features_torch
from .models import create_model
from .train import seed_all


def load_model(name):
    model = create_model(name, load_manifest())
    if name == "nominal":
        return model.eval(), {"model": name, "trained": False}
    summary = json.loads((ROOT / "runs" / f"{name}_completed.json").read_text())
    path = ROOT / summary["run"] / "best.pt"
    if sha256_file(path) != summary["checkpoint_sha256"]:
        raise ValueError(f"Checkpoint hash mismatch: {path}")
    checkpoint = torch.load(path, weights_only=False)
    if checkpoint["metadata"]["config_hash"] != config_hash():
        raise ValueError("Configuration mismatch")
    model.load_state_dict(checkpoint["state_dict"])
    return model.eval(), summary


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows):
    grouped = {}
    for row in rows:
        for scope, key in (("all", "all"), ("trial", row["trial_uid"]), ("reference_type", row["reference_type"])):
            grouped.setdefault((row["model"], int(row["horizon"]), scope, key), []).append(row)
    result = []
    for (model, horizon, scope, key), group in sorted(grouped.items()):
        p = np.asarray([float(r["position_error_m"]) for r in group])
        v = np.asarray([float(r["velocity_error_mps"]) for r in group])
        # Do not mask nonfinite values: aggregate NaN/Inf remains visible and fails acceptance.
        result.append({"model": model, "horizon": horizon, "scope": scope, "group": key,
                       "windows": len(group), "position_rmse_m": float(np.sqrt(np.mean(p*p))),
                       "velocity_rmse_mps": float(np.sqrt(np.mean(v*v))),
                       "position_mae_m": float(np.mean(np.abs(p))), "velocity_mae_mps": float(np.mean(np.abs(v))),
                       "nonfinite_windows": int(sum(not np.isfinite(float(r["position_error_m"])) or
                                                    not np.isfinite(float(r["velocity_error_mps"])) for r in group))})
    return result


def evaluate(split="test"):
    seed_all()
    store = make_store(split, 60)
    refs = evaluation_references(store, 2048)
    if split == "test" and len(refs) != 2028:
        raise ValueError(f"Unexpected test window count: {len(refs)}")
    out = ROOT / "results" / split
    out.mkdir(parents=True, exist_ok=True)
    if (out / "metadata.json").exists():
        raise FileExistsError(f"Completed evaluation exists: {out}; preserve it before rerunning")
    windows = window_records(store, refs)
    (out / "windows.json").write_text(json.dumps(windows, indent=2)+"\n")
    all_rows, diagnostics, model_metadata = [], {}, {}
    targets, commands, timesteps = [], [], []
    for offset in range(0, len(refs), 256):
        batch = store.make_batch(refs[offset:offset+256], torch.device("cpu"))
        targets.append(batch.causal_state.numpy())
        commands.append(batch.commands.numpy())
        timesteps.append(batch.dt.numpy())
    np.savez_compressed(out / "common_inputs.npz", measured_states=np.concatenate(targets),
                        commands=np.concatenate(commands), dt=np.concatenate(timesteps))
    for name in MODEL_NAMES:
        model, model_metadata[name] = load_model(name)
        predictions, saturated, total_stage_points, nonfinite_stage_points = [], 0, 0, 0
        with torch.no_grad():
            for offset in range(0, len(refs), 256):
                chunk = refs[offset:offset+256]
                batch = store.make_batch(chunk, torch.device("cpu"))
                rollout = rollout_causal_dynamics(model, batch.causal_state[:, 0], batch.commands,
                    batch.dt, batch.constants, batch.geometry, batch.response)
                predictions.append(rollout.states.numpy())
                expand = lambda obj: type(obj)(**{f.name: getattr(obj, f.name)[:, None, None] for f in fields(obj)})
                features, _, _ = causal_features_torch(rollout.stage_states, expand(batch.constants), expand(batch.geometry))
                residual = model(features)
                saturated += int((residual.abs() >= .99*.25).sum())
                total_stage_points += residual.numel()
                nonfinite_stage_points += int((~torch.isfinite(residual)).sum())
                for local, ref in enumerate(chunk):
                    trial = store.trials[ref.trial_index]
                    for horizon in configuration()["test"]["horizons"]:
                        errors = rollout.states[local, horizon] - batch.causal_state[local, horizon]
                        all_rows.append({"model": name, "window_id": offset+local, "trial_uid": trial.trial_uid,
                            "reference_type": trial.reference_type, "start": ref.start, "horizon": horizon,
                            "elapsed_s": float(batch.dt[local, :horizon].sum()),
                            "position_error_m": float(errors[0]), "velocity_error_mps": float(errors[1])})
            point_errors = []
            point_saturated, point_count = 0, 0
            for trial in store.trials:
                mask = trial.arrays["rbf_valid"].astype(bool)
                features = torch.as_tensor(trial.arrays["features"][mask], dtype=torch.float64)
                residual = model(features).numpy()
                point_errors.append(residual - trial.arrays["residual_target"][mask])
                point_saturated += int(np.sum(np.abs(residual) >= .99*.25))
                point_count += residual.size
        predicted = np.concatenate(predictions)
        np.savez_compressed(out / f"{name}_predictions.npz", states=predicted)
        point_errors = np.concatenate(point_errors)
        diagnostics[name] = {"point_acceleration_rmse_mps2": float(np.sqrt(np.mean(point_errors**2))),
            "point_acceleration_mae_mps2": float(np.mean(np.abs(point_errors))), "point_count": point_count,
            "point_saturation_fraction": point_saturated/point_count,
            "rollout_stage_saturation_fraction": saturated/total_stage_points,
            "nonfinite_stage_residuals": nonfinite_stage_points,
            "nonfinite_prediction_values": int(np.sum(~np.isfinite(predicted))),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)}
        print(f"Evaluated {name}: {len(refs)} shared windows", flush=True)
    write_csv(out / "window_errors.csv", all_rows)
    write_csv(out / "metrics_long.csv", aggregate(all_rows))
    metadata = {"split": split, "windows": len(refs), "window_hash": window_hash(store, refs),
        "dataset_hash": load_manifest()["dataset_hash"], "config_hash": config_hash(),
        "protocol": configuration(), "models": model_metadata, "diagnostics": diagnostics,
        "evaluation_sources": {p: sha256_file(ROOT / p) for p in ("evaluate.py", "dynamics.py", "data.py")},
        "artifact_hashes": {p.name: sha256_file(p) for p in out.iterdir() if p.is_file()}}
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2)+"\n")
    if any(d["nonfinite_prediction_values"] or d["nonfinite_stage_residuals"] for d in diagnostics.values()):
        raise FloatingPointError("Nonfinite outputs recorded: experiment not accepted")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--split", choices=("test", "validation", "stress"), default="test")
    evaluate(parser.parse_args().split)
