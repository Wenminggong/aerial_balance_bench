#!/usr/bin/env python3
"""Summarize mean absolute ball-position errors over 5 < t_trial <= 30 s.

Run without arguments to process both structured NMPC collaboration datasets.
Each trial has equal weight in its motion-type mean. Trials that terminated
early are ignored entirely; only complete, valid windows are reported.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1] / "logs/real_world/acados_nmpc_cpp/human_collaboration"
DEFAULT_DIRS = [ROOT / name for name in ("structured_nmpc", "structured_nmpc_with_human_motion")]


def summarize_trial(path: Path, root: Path) -> dict | None:
    """Validate samples and distinguish full windows from partial recordings."""
    metadata = json.loads(path.with_name("metadata.json").read_text(encoding="utf-8"))
    if metadata.get("end_reason") != "tracking_duration":
        return None
    with np.load(path, allow_pickle=False) as data:
        time, position, goal = (np.asarray(data[key], dtype=float)
                                for key in ("t_trial", "obs_pb", "obs_pg"))
    if time.ndim != 1 or len(time) < 2 or not np.isfinite(time).all() or np.any(np.diff(time) <= 0):
        raise ValueError(f"Invalid trial timestamps: {path}")
    if position.shape != time.shape or goal.shape != time.shape:
        raise ValueError(f"Misaligned observations: {path}")
    selected = (time > 5.0) & (time <= 30.0)
    window_time = time[selected]
    error = np.abs(position[selected] - goal[selected])
    valid = np.isfinite(error)
    # Allow sampling jitter and the last sample falling just before 30 s.
    tolerance = 2 * float(np.median(np.diff(time)))
    complete = bool(
        window_time.size >= 2 and time[0] <= 5.0
        and window_time[0] <= 5.0 + tolerance
        and window_time[-1] >= 30.0 - tolerance
        and np.max(np.diff(window_time)) <= tolerance and valid.all()
    )
    available_mean = float(error[valid].mean()) if valid.any() else None
    status = "included" if complete else (
        "no_samples_after_5s" if not window_time.size else "incomplete_or_invalid_window"
    )
    return {
        "motion_type": path.parent.parent.name, "trial": path.parent.name,
        "source_file": str(path.relative_to(root)), "end_reason": metadata.get("end_reason", "unknown"),
        "trial_end_s": float(time[-1]), "window_samples": int(window_time.size),
        "invalid_window_samples": int((~valid).sum()),
        "window_first_sample_s": float(window_time[0]) if window_time.size else None,
        "window_last_sample_s": float(window_time[-1]) if window_time.size else None,
        "coverage_tolerance_s": tolerance, "status": status,
        "steady_state_error_m": available_mean if complete else None,
        "steady_state_error_cm": available_mean * 100 if complete else None,
        "available_window_mae_m": available_mean,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write full-precision values; unavailable statistics are blank."""
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize_directory(root: Path) -> None:
    paths = sorted(root.glob("*/*/samples.npz"))
    if not paths:
        raise ValueError(f"No <motion>/<trial>/samples.npz files under {root}")
    trials = [row for path in paths if (row := summarize_trial(path, root)) is not None
              and row["status"] == "included"]
    if not trials:
        raise ValueError(f"No completed trials with valid 5-30 s windows under {root}")
    summaries = []
    for mode in sorted({row["motion_type"] for row in trials}):
        records = [row for row in trials if row["motion_type"] == mode]
        errors = [row["steady_state_error_m"] for row in records if row["status"] == "included"]
        mean = float(np.mean(errors)) if errors else None
        std = float(np.std(errors, ddof=0)) if errors else None
        summaries.append({
            "motion_type": mode, "total_trials": len(records), "included_trials": len(errors),
            "excluded_trials": len(records) - len(errors), "window_start_s": 5.0, "window_end_s": 30.0,
            "mean_steady_state_error_m": mean, "mean_steady_state_error_cm": mean * 100 if errors else None,
            "std_steady_state_error_m": std, "std_steady_state_error_cm": std * 100 if errors else None,
        })
        value = f"{mean:.9f} m ({mean * 100:.6f} cm)" if errors else "N/A"
        print(f"{root.name}/{mode}: {value}; included {len(errors)}/{len(records)} trials")
    write_csv(root / "steady_state_error_5_30s_summary.csv", summaries)
    write_csv(root / "steady_state_error_5_30s_trials.csv", trials)
    report = {
        "definition": "Per-trial sample mean of abs(obs_pb - obs_pg) for 5 < t_trial <= 30 seconds.",
        "aggregation": "Arithmetic mean across included trials, with equal weight per trial.",
        "std_definition": "Population standard deviation (ddof=0) across included trial means.",
        "coverage_policy": "Only trials with end_reason=tracking_duration and a complete valid window are reported. Requires coverage of the entire window, allowing two median sample periods at boundaries and between samples, and finite errors. Early-terminated trials are ignored entirely.",
        "missing_values": "null in JSON and blank in CSV mean unavailable.",
        "summary": summaries, "trials": trials,
    }
    (root / "steady_state_error_5_30s_report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dirs", type=Path, nargs="+", default=DEFAULT_DIRS)
    args = parser.parse_args()
    for directory in args.data_dirs:
        summarize_directory(directory.expanduser().resolve())


if __name__ == "__main__":
    main()
