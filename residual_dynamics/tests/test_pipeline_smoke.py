from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from residual_dynamics.config import load_config
from residual_dynamics.data import load_processed_trials, prepare_dataset
from residual_dynamics.evaluation import evaluate_model
from residual_dynamics.reporting import BEGIN_MARKER, END_MARKER, generate_report
from residual_dynamics.training import train_model


GROUPS = (
    "constant_015",
    "constant_035",
    "constant_055",
    "sine_period_6",
    "sine_period_8",
    "triangle_period_6",
    "triangle_period_8",
    "trapezoid_period_6",
    "trapezoid_period_8",
    "constant_035_2",
)


def test_end_to_end_smoke(tmp_path: Path) -> None:
    source = tmp_path / "raw"
    _write_synthetic_trials(source)
    report_path = tmp_path / "REPORT.md"
    report_path.write_text(f"# Report\n\n{BEGIN_MARKER}\nold\n{END_MARKER}\n")
    base = load_config("residual_dynamics/configs/mlp_5input.yaml")
    paths = replace(
        base.paths,
        source_dir=source,
        processed_dir=tmp_path / "processed",
        training_runs_dir=tmp_path / "runs" / "training",
        wandb_dir=tmp_path / "runs" / "wandb",
        results_dir=tmp_path / "results",
        report_path=report_path,
    )
    training = replace(
        base.training,
        batch_size=4,
        steps_per_epoch=1,
        max_epochs=1,
        early_stopping_patience=1,
        validation_max_windows=16,
    )
    runtime = replace(base.runtime, device="cpu", benchmark_warmup_steps=1, benchmark_steps=1)
    wandb = replace(base.wandb, enabled=False, mode="disabled")
    config = replace(base, paths=paths, training=training, runtime=runtime, wandb=wandb)

    manifest = prepare_dataset(config)
    assert manifest["split_counts"] == {"train": 20, "validation": 9, "excluded": 1}
    assert len(load_processed_trials(config, "train")) == 20
    assert len(load_processed_trials(config, "validation")) == 9
    run_dir = train_model(config)
    summary = evaluate_model(config, run_dir)
    assert summary["split_counts"]["train"] == 20
    assert (run_dir / "best_checkpoint.pt").is_file()
    assert (run_dir / "weights.npz").is_file()
    generate_report(config)
    assert "实际训练与验证结果" in report_path.read_text()


def test_causal_end_to_end_smoke(tmp_path: Path) -> None:
    source = tmp_path / "raw_causal"
    _write_synthetic_trials(source)
    report_path = tmp_path / "REPORT_CAUSAL.md"
    report_path.write_text(f"# Causal report\n\n{BEGIN_MARKER}\nold\n{END_MARKER}\n")
    base = load_config("residual_dynamics/configs/mlp_5input_causal_rollout.yaml")
    config = replace(
        base,
        paths=replace(
            base.paths,
            source_dir=source,
            processed_dir=tmp_path / "processed_causal",
            training_runs_dir=tmp_path / "runs_causal" / "training",
            wandb_dir=tmp_path / "runs_causal" / "wandb",
            results_dir=tmp_path / "results_causal",
            report_path=report_path,
        ),
        training=replace(
            base.training,
            batch_size=4,
            steps_per_epoch=1,
            max_epochs=1,
            early_stopping_patience=1,
            validation_max_windows=16,
        ),
        runtime=replace(
            base.runtime, device="cpu", benchmark_warmup_steps=1, benchmark_steps=1
        ),
        wandb=replace(base.wandb, enabled=False, mode="disabled"),
    )
    manifest = prepare_dataset(config)
    assert manifest["manifest_version"] == 2
    assert manifest["normalization"]["state_scale_names"] == [
        "pb",
        "vb",
        "theta",
        "vrz",
    ]
    run_dir = train_model(config)
    summary = evaluate_model(config, run_dir)
    assert summary["training_variant"] == "causal_rollout"
    assert "response_chain" in summary["rollout"]["overall"]["h60"]
    assert "jacobian_norm_p99_mps2" in summary["diagnostics"]
    assert summary["diagnostics"]["command_queue_alignment"]["passed"]
    generate_report(config)
    assert "SG 辅助目标与响应链诊断" in report_path.read_text()


def _write_synthetic_trials(source: Path) -> None:
    for group_index, group in enumerate(GROUPS):
        for trial_index in (1, 2, 3):
            end_reason = (
                "safety_stale_beam"
                if group == "constant_035_2" and trial_index == 3
                else "tracking_duration"
            )
            trial_dir = source / group / f"synthetic_trial_{trial_index:03d}_{end_reason}"
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
                    "controller.response.enabled": True,
                    "controller.response.tau_s": 0.187,
                    "controller.response.gain": 1.0,
                    "controller.response.bias": 0.0,
                    "controller.constraints.max_velocity": 3.0,
                    "predictor.delay_steps": 3,
                },
            }
            (trial_dir / "metadata.json").write_text(json.dumps(metadata))
            with (trial_dir / "samples.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=[
                        "t_trial",
                        "phase",
                        "obs_pb",
                        "obs_vb",
                        "obs_theta",
                        "obs_vrz",
                        "nmpc_command_z",
                        "pending_command_0",
                        "pending_command_1",
                        "pending_command_2",
                    ],
                )
                writer.writeheader()
                commands = [
                    0.05 * np.cos(0.6 * (sample / 60.0) + 0.1 * group_index)
                    for sample in range(110)
                ]
                for sample in range(110):
                    time_s = sample / 60.0
                    phase = 0.6 * time_s + 0.1 * group_index + 0.03 * trial_index
                    writer.writerow(
                        {
                            "t_trial": time_s,
                            "phase": "tracking",
                            "obs_pb": 0.35 + 0.03 * np.sin(phase),
                            "obs_vb": 0.018 * np.cos(phase),
                            "obs_theta": 0.01 * np.sin(phase + 0.2),
                            "obs_vrz": 0.05 * np.cos(phase + 0.2),
                            "nmpc_command_z": commands[sample],
                            "pending_command_0": (
                                "" if sample < 2 else commands[sample - 2]
                            ),
                            "pending_command_1": (
                                "" if sample < 2 else commands[sample - 1]
                            ),
                            "pending_command_2": (
                                "" if sample < 2 else commands[sample]
                            ),
                        }
                    )
