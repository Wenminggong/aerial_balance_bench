from __future__ import annotations

from dataclasses import replace
import csv
from pathlib import Path

import numpy as np
import pytest
import torch

from residual_dynamics.config import load_config
from residual_dynamics.data import (
    PhysicalConstants,
    TrialData,
    _assign_split,
    _ood_statistics,
    _process_causal_trial,
    ood_distances,
    position_bin_indices,
)
from residual_dynamics.dynamics import (
    TorchGeometryConstants,
    TorchPhysicalConstants,
    TorchResponseConstants,
    causal_residual_torch,
)
from residual_dynamics.evaluation import _acceptance, _sensitivity_statistics
from residual_dynamics.model import ResidualMLP
from residual_dynamics.training import (
    WindowBatch,
    WindowStore,
    _deterministic_collocation_batch,
    compute_batch_loss,
    control_channel_losses,
)
import residual_dynamics.screening as screening


CONFIG = Path("residual_dynamics/configs/mlp_5input_causal_v2.yaml")
SUCCESS_CONFIG = Path("residual_dynamics/configs/mlp_5input_causal_v2_success.yaml")
CAP0P5_CONFIG = Path(
    "residual_dynamics/configs/mlp_5input_causal_success_cap0p5.yaml"
)
CAP0P25_ABLATION_CONFIG = Path(
    "residual_dynamics/configs/mlp_5input_causal_success_cap0p25_ablation.yaml"
)
REBALANCED_CONFIG = Path(
    "residual_dynamics/configs/mlp_5input_causal_success_rebalanced.yaml"
)


def _trial(position: float, index: int, *, failure: bool = False) -> TrialData:
    size = 20
    vb = np.full(size, (-1 + index % 3) * 0.05, dtype=np.float64)
    pb = np.full(size, position, dtype=np.float64)
    zeros = np.zeros(size, dtype=np.float64)
    arrays = {
        "features": np.column_stack((pb, vb, zeros, zeros, zeros)),
        "residual_target": zeros.copy(),
        "pb": pb,
        "vb": vb,
        "theta": zeros.copy(),
        "omega": zeros.copy(),
        "dt_s": np.full(size, 1.0 / 60.0),
        "speed_regime": np.full(size, np.sign(vb[0]), dtype=np.int8),
        "segment_id": np.zeros(size, dtype=np.int32),
        "time_s": np.arange(size, dtype=np.float64) / 60.0,
    }
    return TrialData(
        trial_id=f"trial_{index:03d}",
        group=f"group_{index % 2}",
        session=f"session_{index % 3}",
        split="train",
        reference_type="constant",
        source_file="synthetic.csv",
        arrays=arrays,
        constants=PhysicalConstants(1.06, 0.9, 0.33, 9.81, 1.0 / 1.4),
        is_failure_prefix=failure,
    )


def test_position_bin_edges_include_both_rare_extremes() -> None:
    config = load_config(CONFIG)
    values = np.asarray([0.0, 0.099999, 0.1, 0.6, 0.699999, 0.7, -1e-9, 0.700001])
    bins = position_bin_indices(values, config.data.position_bin_edges_m)
    assert bins.tolist() == [0, 0, 1, 6, 6, 6, -1, -1]


def test_success_only_config_excludes_training_failures_but_keeps_stress() -> None:
    config = load_config(SUCCESS_CONFIG)
    failed = {"end_reason": "solver_fallback"}
    train_split, train_reason = _assign_split(
        "20260818",
        "trapezoid_period_8",
        "20260818214118_trial_001_solver_fallback",
        failed,
        config,
    )
    stress_split, stress_reason = _assign_split(
        "20260818",
        "sine_period_8",
        "20260818212000_trial_001_solver_fallback",
        failed,
        config,
    )
    assert train_split is None
    assert "disabled for supervised training" in train_reason
    assert stress_split == "stress"
    assert stress_reason == ""


def test_cap0p5_screening_policy_forces_one_seed_without_control_channel() -> None:
    config = load_config(CAP0P5_CONFIG)
    policy = screening._selection_policy(config)
    assert config.model.output_limit_mps2 == pytest.approx(0.5)
    assert config.acceptance.residual_rmse_max_mps2 == pytest.approx(0.15)
    assert not config.control_channel.enabled
    assert policy == {
        "allow_ineligible_fallback": True,
        "repeat_seeds": [7],
        "evaluate_holdout_for_selected": True,
    }


def test_cap0p25_ablation_changes_only_cap_from_selected_cap0p5_training() -> None:
    cap0p5 = load_config(CAP0P5_CONFIG)
    cap0p25 = load_config(CAP0P25_ABLATION_CONFIG)
    assert cap0p25.data == cap0p5.data
    assert cap0p25.loss == cap0p5.loss
    assert cap0p25.training == cap0p5.training
    assert cap0p25.runtime == cap0p5.runtime
    assert cap0p25.acceptance == cap0p5.acceptance
    assert cap0p25.ood == cap0p5.ood
    assert cap0p25.control_channel == cap0p5.control_channel
    assert cap0p25.causal_rollout == replace(
        cap0p5.causal_rollout, jacobian_weight=0.0
    )
    assert cap0p25.model.input_names == cap0p5.model.input_names
    assert cap0p25.model.hidden_dims == cap0p5.model.hidden_dims
    assert cap0p25.model.dtype == cap0p5.model.dtype
    assert cap0p25.model.output_limit_mps2 == pytest.approx(0.25)
    assert cap0p5.model.output_limit_mps2 == pytest.approx(0.5)


def test_rebalanced_config_fixes_single_seed_loss_and_jacobian_contract() -> None:
    base = load_config(CAP0P25_ABLATION_CONFIG)
    config = load_config(REBALANCED_CONFIG)
    assert config.data == base.data
    assert config.training == base.training
    assert config.training.seed == 7
    assert config.model == base.model
    assert config.model.output_limit_mps2 == pytest.approx(0.25)
    assert config.loss.horizon_weights == pytest.approx((0.4, 0.3, 0.2, 0.1))
    assert config.loss.point_weight == pytest.approx(5.0)
    assert config.loss.jacobian_weight == pytest.approx(0.0)
    assert config.causal_rollout == replace(
        base.causal_rollout,
        jacobian_weight=0.1,
        jacobian_norm_limit_mps2=0.04,
    )
    assert not config.control_channel.enabled


def test_rebalanced_causal_loss_reports_exact_weighted_budget() -> None:
    config = load_config(REBALANCED_CONFIG)
    batch_size = 2
    horizon = max(config.data.rollout_steps)
    constants = TorchPhysicalConstants(
        plank_length_m=torch.ones(batch_size, dtype=torch.float64),
        ball_position_offset_m=torch.zeros(batch_size, dtype=torch.float64),
        gravity_mps2=torch.zeros(batch_size, dtype=torch.float64),
        acceleration_gain=torch.ones(batch_size, dtype=torch.float64),
    )
    geometry = TorchGeometryConstants(
        plank_length_m=torch.ones(batch_size, dtype=torch.float64),
        rope_length_m=torch.ones(batch_size, dtype=torch.float64),
    )
    response = TorchResponseConstants(
        enabled=torch.zeros(batch_size, dtype=torch.bool),
        tau_s=torch.full((batch_size,), 0.2, dtype=torch.float64),
        gain=torch.ones(batch_size, dtype=torch.float64),
        bias=torch.zeros(batch_size, dtype=torch.float64),
        max_velocity_mps=torch.full((batch_size,), 3.0, dtype=torch.float64),
    )
    causal_state = torch.zeros((batch_size, horizon + 1, 4), dtype=torch.float64)
    causal_state[:, :, 0] = torch.linspace(
        0.0, 0.02, horizon + 1, dtype=torch.float64
    )
    batch = WindowBatch(
        features=torch.zeros((batch_size, horizon + 1, 5), dtype=torch.float64),
        residual_target=torch.full(
            (batch_size, horizon + 1), 0.1, dtype=torch.float64
        ),
        pb=causal_state[:, :, 0],
        vb=causal_state[:, :, 1],
        theta=causal_state[:, :, 2],
        omega=torch.zeros((batch_size, horizon + 1), dtype=torch.float64),
        dt=torch.full((batch_size, horizon), 1.0 / 60.0, dtype=torch.float64),
        constants=constants,
        causal_state=causal_state,
        smooth_pb=causal_state[:, :, 0],
        smooth_vb=causal_state[:, :, 1],
        commands=torch.zeros((batch_size, horizon), dtype=torch.float64),
        geometry=geometry,
        response=response,
        state_scales=torch.ones(4, dtype=torch.float64),
    )
    model = ResidualMLP(np.zeros(5), np.ones(5), output_limit_mps2=0.25)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.layers[0].weight[0, 0] = 1.0
        model.layers[2].weight[0, 0] = 1.0
        model.layers[4].weight[0, 0] = 1.0

    loss, details = compute_batch_loss(
        model, batch, config, include_jacobian=True
    )
    component_sum = (
        details["rollout_loss"]
        + details["weighted_point_residual_loss"]
        + details["weighted_jacobian_loss"]
        + details["weighted_theta_channel_loss"]
        + details["weighted_anti_damping_loss"]
    )
    horizon_sum = sum(
        details[f"weighted_rollout_loss_h{step}"]
        for step in config.data.rollout_steps
    )
    assert torch.allclose(loss, component_sum, rtol=0.0, atol=1.0e-12)
    assert torch.allclose(details["rollout_loss"], horizon_sum, rtol=0.0, atol=1.0e-12)
    assert details["jacobian_norm_max"].item() > 0.04
    assert details["jacobian_loss"].item() > 0.0
    fractions = (
        details["rollout_loss_fraction"]
        + details["point_loss_fraction"]
        + details["jacobian_loss_fraction"]
        + details["control_channel_loss_fraction"]
    )
    assert fractions.item() == pytest.approx(1.0)
    loss.backward()
    gradients = [value.grad for value in model.parameters() if value.grad is not None]
    assert gradients
    assert all(torch.isfinite(value).all() for value in gradients)


def test_forced_jacobian_screening_keeps_best_failed_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_config(CAP0P5_CONFIG)
    config = replace(
        config,
        paths=replace(config.paths, results_dir=tmp_path / "results"),
    )
    trained: list[tuple[float, float, int]] = []
    holdout_calls: list[bool] = []

    def fake_train(candidate_config):
        trained.append(
            (
                candidate_config.causal_rollout.jacobian_weight,
                candidate_config.causal_rollout.jacobian_norm_limit_mps2,
                candidate_config.training.max_epochs,
            )
        )
        return tmp_path / f"run_{len(trained)}", {"placeholder": True}

    def fake_record(name, weight, limit, seed, run_dir, summary, candidate_config):
        score = {"jacobian_off": 4.0, "lmax_0p01": 3.0,
                 "lmax_0p02": 1.0, "lmax_0p04": 2.0}[name]
        return {
            "candidate": name,
            "seed": seed,
            "jacobian_weight": weight,
            "jacobian_norm_limit_mps2": limit,
            "run_dir": str(run_dir),
            "hard_filters_passed": False,
            "h33_causal_score": score,
        }

    def fake_final(candidate_config, run_dir, *, include_holdout=False):
        holdout_calls.append(include_holdout)
        return {"export_eligible": False}

    monkeypatch.setattr(screening, "_train_or_reuse", fake_train)
    monkeypatch.setattr(screening, "_candidate_record", fake_record)
    monkeypatch.setattr(screening, "evaluate_model", fake_final)
    result = screening.run_screening(config)
    assert result["selected_candidate"] == "lmax_0p02"
    assert result["selected_seed"] == 7
    assert result["forced_selection"]
    assert result["export_blocked"]
    assert holdout_calls == [True]
    assert len(trained) == 5
    assert trained[-1] == pytest.approx((0.1, 0.02, 500))


def test_stress_acceptance_does_not_require_v2_channel_diagnostics() -> None:
    config = load_config(CAP0P5_CONFIG)
    point = {"overall": {"mlp": {"rmse": 0.1}}}
    rollout = {
        "overall": {
            "h33": {
                "nominal": {"pb_rmse_m": 1.0, "vb_rmse_mps": 1.0},
                "mlp": {"pb_rmse_m": 0.9, "vb_rmse_mps": 0.9},
                "improvement": {"pb_mse_fraction": 0.19, "vb_mse_fraction": 0.19},
            },
            "h60": {
                "response_chain": {
                    "theta_bias_rad": 0.0,
                    "vrz_bias_mps": 0.0,
                }
            },
        },
        "by_group": {},
    }
    diagnostics = {
        "static_total_acc_median_mps2": 0.0,
        "residual_saturation_rate": 0.0,
        "gradients_finite": True,
        "outputs_finite": True,
        "jacobian_p99_within_limit": True,
        "jacobian_max_within_limit": True,
        "command_queue_alignment": {"passed": True},
    }
    result = _acceptance(point, rollout, diagnostics, config, split="stress")
    assert result["passed"]
    assert "theta_sector" not in result["checks"]


def test_failure_terminal_row_is_excluded_before_filtering(tmp_path: Path) -> None:
    config = load_config(CONFIG)
    trial_dir = tmp_path / "20260820120000_trial_001_solver_fallback"
    trial_dir.mkdir()
    samples = trial_dir / "samples.csv"
    fieldnames = [
        "t_trial", "phase", "obs_pb", "obs_vb", "obs_theta", "obs_vrz",
        "nmpc_command_z", "pending_command_0", "pending_command_1",
        "pending_command_2", "safety_state", "end_reason",
    ]
    with samples.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for index in range(150):
            terminal = index == 130
            writer.writerow(
                {
                    "t_trial": index / 60.0,
                    "phase": "tracking",
                    "obs_pb": 1.0e6 if terminal else 0.35 + 0.01 * np.sin(index / 30.0),
                    "obs_vb": 1.0e6 if terminal else 0.0,
                    "obs_theta": 1.0e6 if terminal else 0.01,
                    "obs_vrz": 1.0e6 if terminal else 0.0,
                    "nmpc_command_z": 0.0,
                    "pending_command_0": 0.0,
                    "pending_command_1": 0.0,
                    "pending_command_2": 0.0,
                    "safety_state": "solver_fallback" if terminal else "ok",
                    "end_reason": "solver_fallback" if terminal else "running",
                }
            )
    metadata = {
        "end_reason": "solver_fallback",
        "parameters": {
            "controller.response.enabled": True,
            "controller.response.tau_s": 0.187,
            "controller.response.gain": 1.0,
            "controller.response.bias": 0.0,
            "controller.constraints.max_velocity": 3.0,
            "predictor.delay_steps": 3,
            "controller.model.ball_mass": 0.0005,
            "controller.model.ball_radius": 0.023,
            "controller.model.ball_inertia_ratio": 0.4,
            "controller.model.plank_length": 1.06,
            "controller.model.rope_length": 0.9,
            "controller.model.ball_position_offset": 0.33,
            "controller.model.gravity": -9.81,
        },
    }
    trial = _process_causal_trial(
        samples, metadata, "constant_035", "train", config, session="20260820"
    )
    assert trial.is_failure_prefix
    assert trial.terminal_time_s == pytest.approx(130 / 60.0)
    assert trial.truncated_sample_count == 130
    assert np.max(trial.arrays["pb"]) < 0.7
    assert np.isfinite(trial.arrays["residual_target"]).all()


def test_mixed_sampler_is_reproducible_and_covers_every_position_bin() -> None:
    config = load_config(CONFIG)
    config = replace(
        config,
        causal_rollout=replace(config.causal_rollout, enabled=False),
    )
    trials = [
        _trial(0.05 + 0.1 * index, index, failure=index == 6)
        for index in range(7)
    ]
    store = WindowStore(trials, horizon=3, config=config)
    first = store.balanced_references(np.random.default_rng(123), 256)
    second = store.balanced_references(np.random.default_rng(123), 256)
    assert first == second
    assert {reference.position_bin for reference in first} == set(range(7))


def test_joint_ood_detects_correlation_violation_inside_marginal_box() -> None:
    config = load_config(CONFIG)
    rng = np.random.default_rng(4)
    latent = rng.normal(size=5000)
    features = np.column_stack(
        [latent + 0.02 * rng.normal(size=latent.size) for _ in range(5)]
    )
    normalization = {
        "mean": features.mean(axis=0).tolist(),
        "std": features.std(axis=0).tolist(),
        "min": features.min(axis=0).tolist(),
        "max": features.max(axis=0).tolist(),
        "ood": _ood_statistics(features, config),
    }
    query = np.asarray([[1.0, -1.0, 0.5, -0.5, 0.0]])
    distances = ood_distances(query, normalization)
    assert distances["marginal"][0] == pytest.approx(0.0)
    assert distances["joint_excess"][0] > 0.0
    assert distances["combined"][0] > 0.0


def test_theta_sector_and_anti_damping_losses_have_expected_gates() -> None:
    config = load_config(CONFIG)
    nominal = torch.zeros((4, 4), dtype=torch.float64)
    nominal[:, 2] = -7.0
    theta_loss, damping_loss, ratio, _ = control_channel_losses(
        nominal.clone(), nominal, config
    )
    assert theta_loss.item() == pytest.approx(0.0)
    assert damping_loss.item() == pytest.approx(0.0)
    assert torch.allclose(ratio, torch.ones_like(ratio))

    cancelled = nominal.clone()
    cancelled[:, 2] = 0.0
    cancelled[:, 1] = 0.35
    theta_loss, damping_loss, ratio, anti = control_channel_losses(
        cancelled, nominal, config
    )
    assert theta_loss.item() > 0.0
    assert damping_loss.item() > 0.0
    assert torch.all(ratio == 0.0)
    assert torch.all(anti > config.control_channel.anti_damping_soft_max_sinv)


def test_collocation_supports_finite_second_order_backpropagation() -> None:
    config = load_config(CONFIG)
    batch_size = 8
    constants = TorchPhysicalConstants(
        plank_length_m=torch.full((batch_size,), 1.06, dtype=torch.float64),
        ball_position_offset_m=torch.full((batch_size,), 0.33, dtype=torch.float64),
        gravity_mps2=torch.full((batch_size,), 9.81, dtype=torch.float64),
        acceleration_gain=torch.full((batch_size,), 1.0 / 1.4, dtype=torch.float64),
    )
    geometry = TorchGeometryConstants(
        plank_length_m=torch.full((batch_size,), 1.06, dtype=torch.float64),
        rope_length_m=torch.full((batch_size,), 0.9, dtype=torch.float64),
    )
    batch = WindowBatch(
        features=torch.zeros((batch_size, 2, 5), dtype=torch.float64),
        residual_target=torch.zeros((batch_size, 2), dtype=torch.float64),
        pb=torch.zeros((batch_size, 2), dtype=torch.float64),
        vb=torch.zeros((batch_size, 2), dtype=torch.float64),
        theta=torch.zeros((batch_size, 2), dtype=torch.float64),
        omega=torch.zeros((batch_size, 2), dtype=torch.float64),
        dt=torch.full((batch_size, 1), 1.0 / 60.0, dtype=torch.float64),
        constants=constants,
        causal_state=torch.zeros((batch_size, 2, 4), dtype=torch.float64),
        geometry=geometry,
    )
    states, selected_constants, selected_geometry = _deterministic_collocation_batch(
        batch, config
    )
    states = states.requires_grad_(True)
    model = ResidualMLP(np.zeros(5), np.ones(5))
    torch.manual_seed(8)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(std=0.04)
    total, nominal_acc, _, _ = causal_residual_torch(
        model, states, selected_constants, selected_geometry
    )
    total_jacobian = torch.autograd.grad(
        total.sum(), states, create_graph=True, retain_graph=True
    )[0]
    nominal_jacobian = torch.autograd.grad(
        nominal_acc.sum(), states, create_graph=True
    )[0]
    theta_loss, damping_loss, _, _ = control_channel_losses(
        total_jacobian, nominal_jacobian, config
    )
    (theta_loss + damping_loss).backward()
    gradients = [value.grad for value in model.parameters() if value.grad is not None]
    assert gradients
    assert all(torch.isfinite(value).all() for value in gradients)


def test_nominal_control_sensitivity_statistics_are_identity() -> None:
    values = np.linspace(0.1, 1.0, 32)
    metrics = _sensitivity_statistics(
        np.ones_like(values), np.ones_like(values), values, values
    )
    assert metrics["cosine_p01"] == pytest.approx(1.0)
    assert metrics["norm_ratio_p01"] == pytest.approx(1.0)
    assert metrics["norm_ratio_p99"] == pytest.approx(1.0)
    assert metrics["near_zero_fraction"] == 0.0
    assert metrics["sign_reversal_fraction"] == 0.0


def test_test_and_stress_are_only_requested_after_validation_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_config(CONFIG)
    config = replace(
        config,
        paths=replace(config.paths, results_dir=tmp_path / "results"),
    )
    trained: list[tuple[int, float, bool]] = []
    final_calls: list[bool] = []

    def fake_train(candidate_config):
        trained.append(
            (
                candidate_config.training.seed,
                candidate_config.control_channel.theta_weight,
                candidate_config.training.max_epochs == 80,
            )
        )
        return tmp_path / f"run_{len(trained)}", {"placeholder": True}

    def fake_record(name, theta_weight, seed, run_dir, summary, *, screening):
        return {
            "candidate": name,
            "theta_channel_weight": theta_weight,
            "seed": seed,
            "run_dir": str(run_dir),
            "hard_filters": {"theta_sector": True, "outer_theta_sector": True},
            "hard_filters_passed": theta_weight > 0.0,
            "h33_causal_score": theta_weight + seed * 1.0e-3,
        }

    def fake_final(candidate_config, run_dir, *, include_holdout=False):
        final_calls.append(include_holdout)
        return {"export_eligible": True}

    monkeypatch.setattr(screening, "_train_or_reuse", fake_train)
    monkeypatch.setattr(screening, "_channel_candidate_record", fake_record)
    monkeypatch.setattr(screening, "evaluate_model", fake_final)
    result = screening._run_channel_screening(config)
    assert result["selected_candidate"] == "channel_1"
    assert final_calls == [True]
    assert len(trained) == 3 + len(screening.REPEAT_SEEDS)
