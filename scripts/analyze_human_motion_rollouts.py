#!/usr/bin/env python3
"""Offline human-velocity hold and delay-prediction diagnostics from acados rollouts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml


def uninterrupted_pairs(done: np.ndarray, horizon: int) -> np.ndarray:
    """Mask k -> k+h pairs with no terminal transition in [k, k+h)."""
    if horizon < 0 or horizon >= len(done):
        raise ValueError("horizon must be non-negative and shorter than the rollout.")
    prefix = np.concatenate((np.zeros((1, done.shape[1]), dtype=int), np.cumsum(done, axis=0)))
    length = len(done) - horizon
    return (prefix[horizon:horizon + length] - prefix[:length]) == 0


def error_metrics(error: np.ndarray, valid: np.ndarray) -> dict[str, float | int | None]:
    values = np.asarray(error)[valid & np.isfinite(error)]
    return {
        "samples": int(values.size),
        "mae": float(np.mean(np.abs(values))) if values.size else None,
        "rmse": float(np.sqrt(np.mean(values ** 2))) if values.size else None,
    }


def analyze_run(directory: Path) -> dict:
    metadata = yaml.safe_load((directory / "resolved_run.yaml").read_text())
    summary = yaml.safe_load((directory / "timing_summary.yaml").read_text())
    summary = {
        key: None if isinstance(value, float) and not np.isfinite(value) else value
        for key, value in summary.items()
    }
    with np.load(directory / "rollout.npz", allow_pickle=False) as data:
        done = data["terminated"] | data["truncated"]
        observation = data["observations"]
        names = data["observation_fields"].tolist()
        delay = int(metadata["reference_base_offset"])
        result = {"directory": str(directory.resolve()), "summary": summary, "delay_steps": delay}
        if delay < len(done):
            valid = uninterrupted_pairs(done, delay)
            length = len(valid)
            result["predictor_error"] = {
                field: error_metrics(
                    data[f"policy_predicted_{field}"][:length]
                    - observation[delay:, :, names.index(field)],
                    valid,
                )
                for field in ("pb", "vb", "theta", "omega", "vrz")
            }
        # Reconstruct actual pre-action velocity for either controller, excluding
        # initial/reset observations whose measurements are not in prior step logs.
        post = data["step_human_vel_z"]
        pre = np.full_like(post, np.nan)
        pre[1:] = np.where(done[:-1], np.nan, post[:-1])
        result["constant_velocity_hold_error"] = {}
        for h in sorted(set([delay, delay + int(metadata["effective_horizon"])])):
            if h >= len(done):
                continue
            valid = uninterrupted_pairs(done, h)
            length = len(valid)
            result["constant_velocity_hold_error"][str(h)] = error_metrics(pre[:length] - pre[h:], valid)
        used = data["acados_nmpc_human_velocity_z_used"]
        enabled = bool(metadata["human_motion_enabled"])
        if enabled:
            result["measurement_alignment"] = error_metrics(used - pre, np.ones_like(done))
        else:
            result["measurement_alignment"] = {"disabled_used_max_abs": float(np.max(np.abs(used)))}
        result["max_abs_action"] = float(np.max(np.abs(data["actions"])))
        result["terminal_transitions"] = int(done.sum())
        result["human_velocity_rms"] = float(np.sqrt(np.mean(post ** 2)))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, help="Optional JSON output; otherwise print.")
    args = parser.parse_args()
    report = {path.name: analyze_run(path) for path in args.run_dirs}
    encoded = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    else:
        print(encoded, end="")


if __name__ == "__main__":
    main()
