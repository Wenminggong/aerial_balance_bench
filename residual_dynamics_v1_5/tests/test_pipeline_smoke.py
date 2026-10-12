from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from residual_dynamics_v1_5.config import load_config
from residual_dynamics_v1_5.data import prepare_datasets
from residual_dynamics_v1_5.evaluation import evaluate_model, test_model as run_test_evaluation
from residual_dynamics_v1_5.reporting import END_MARKER, START_MARKER, update_report
from residual_dynamics_v1_5.training import train_model


STANDARD = (
    "constant_015", "constant_035", "constant_055", "sine_period_6",
    "sine_period_8", "triangle_period_6", "triangle_period_8",
    "trapezoid_period_6", "trapezoid_period_8",
)


def test_prepare_train_evaluate_test_report_smoke(tmp_path: Path) -> None:
    fit_source, test_source = tmp_path / "fit", tmp_path / "test"
    _write_fit(fit_source); _write_test(test_source)
    report_path = tmp_path / "REPORT.md"
    report_path.write_text(f"# Smoke\n\n{START_MARKER}\nold\n{END_MARKER}\n")
    base = load_config("residual_dynamics_v1_5/configs/mlp_deadzone.yaml")
    paths = replace(
        base.paths, fit_source_dir=fit_source, test_source_dir=test_source,
        processed_dir=tmp_path / "processed",
        training_runs_dir=tmp_path / "runs" / "training",
        wandb_dir=tmp_path / "runs" / "wandb", results_dir=tmp_path / "results",
        report_path=report_path,
    )
    training = replace(
        base.training, batch_size=4, steps_per_epoch=1,
        dynamic_max_epochs=1, dynamic_early_stopping_patience=1,
        deadzone_max_epochs=1, deadzone_early_stopping_patience=1,
        validation_max_windows=8,
    )
    runtime = replace(base.runtime, device="cpu", benchmark_warmup_steps=1, benchmark_steps=1)
    wandb = replace(base.wandb, enabled=False, mode="disabled")
    config = replace(base, paths=paths, training=training, runtime=runtime, wandb=wandb)

    manifests = prepare_datasets(config)
    assert manifests["fit"]["split_counts"] == {"train": 20, "validation": 9, "excluded": 1}
    assert manifests["test"]["split_counts"] == {"test": 14, "stress": 5}
    run_dir = train_model(config)
    validation = evaluate_model(config, run_dir)
    final = run_test_evaluation(config, run_dir)
    update_report(config)

    assert validation["trial_count"] == 9
    assert final["test"]["trial_count"] == 14
    assert final["stress"]["trial_count"] == 5
    assert final["training"]["mlp_frozen_unchanged"]
    assert (run_dir / "best_dynamic_checkpoint.pt").is_file()
    assert (run_dir / "best_composite_checkpoint.pt").is_file()
    assert (run_dir / "weights_v1_5.npz").is_file()
    assert "自动生成的实验结果" in report_path.read_text()


def _write_fit(root: Path) -> None:
    for group_index, group in enumerate(STANDARD):
        for trial_index in (1, 2, 3):
            _write_trial(root, group, trial_index, "tracking_duration", group_index)
    _write_trial(root, "constant_035_2", 1, "tracking_duration", 9)
    _write_trial(root, "constant_035_2", 2, "tracking_duration", 9)
    _write_trial(root, "constant_035_2", 3, "safety_stale_beam", 9)


def _write_test(root: Path) -> None:
    for index in range(14):
        _write_trial(root, f"test_group_{index:02d}", 1, "tracking_duration", index)
    reasons = ("solver_fallback", "solver_fallback", "solver_fallback",
               "safety_beam_angle_violation", "manual_hover")
    for index, reason in enumerate(reasons):
        _write_trial(root, f"stress_group_{index:02d}", 1, reason, index + 14)


def _write_trial(root: Path, group: str, trial_index: int, end_reason: str,
                 phase_offset: int) -> None:
    trial_dir = root / group / f"synthetic_trial_{trial_index:03d}_{end_reason}"
    trial_dir.mkdir(parents=True)
    metadata = {
        "end_reason": end_reason,
        "parameters": {
            "controller.model.ball_mass": 0.0005,
            "controller.model.ball_radius": 0.023,
            "controller.model.ball_inertia_ratio": 0.4,
            "controller.model.plank_length": 1.06,
            "controller.model.rope_length": 0.9,
            "controller.model.ball_position_offset": 0.33,
            "controller.model.gravity": 9.81,
        },
    }
    (trial_dir / "metadata.json").write_text(json.dumps(metadata))
    with (trial_dir / "samples.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=(
            "t_trial", "phase", "obs_pb", "obs_theta", "obs_vrz",
        ))
        writer.writeheader()
        for sample in range(240):
            time_s = sample / 60.0
            if sample < 60:
                pb = 0.35 + 0.0001 * np.sin(0.1 * phase_offset)
            else:
                phase = np.pi * (time_s - 1.0) + 0.01 * phase_offset
                pb = 0.35 + 0.06 * np.sin(phase)
            writer.writerow({
                "t_trial": time_s, "phase": "tracking", "obs_pb": pb,
                "obs_theta": 0.005 * np.sin(0.4 * time_s + 0.01 * phase_offset),
                "obs_vrz": 0.02 * np.cos(0.4 * time_s + 0.01 * phase_offset),
            })
