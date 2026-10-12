"""Three-stream sampling and two-stage training for the RBF residual."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import shutil
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from residual_dynamics.dynamics import rollout_causal_dynamics
from residual_dynamics.training import WindowStore as LegacyWindowStore

from .config import Config, StageConfig
from .data import (
    actuation_regimes,
    load_manifest,
    load_trials,
    position_bins,
    velocity_bins,
)
from .model import (
    ResidualRBF2D,
    checkpoint_payload,
    create_model,
    load_checkpoint,
    portable_payload,
    save_portable_json,
)


@dataclass(frozen=True)
class WindowReference:
    trial_index: int
    start: int
    position_bin: int
    velocity_bin: int
    actuation_regime: int


class WindowStore:
    """Filter causal windows and provide the approved three-stream sampler."""

    def __init__(self, trials: Sequence[Any], horizon: int, config: Config, manifest: Mapping[str, Any]):
        self.trials = list(trials)
        self.horizon = int(horizon)
        self.config = config
        self.legacy = LegacyWindowStore(
            self.trials, self.horizon, config.legacy, dict(manifest["normalization"])
        )
        self.references: list[WindowReference] = []
        for trial_index, trial in enumerate(self.trials):
            valid = trial.arrays["rbf_valid"].astype(bool)
            segments = trial.arrays["segment_id"]
            for start in range(0, trial.size - self.horizon):
                endpoint = start + self.horizon
                if segments[start] != segments[endpoint]:
                    continue
                if not bool(valid[start : endpoint + 1].all()):
                    continue
                middle = start + self.horizon // 2
                pbin = int(
                    position_bins(
                        np.asarray([trial.arrays["pb"][middle]]),
                        config.legacy.data.position_bin_edges_m,
                    )[0]
                )
                if pbin < 0:
                    continue
                vbin = int(
                    velocity_bins(
                        np.asarray([trial.arrays["vb"][middle]]),
                        config.sampling.velocity_bin_edges_mps,
                    )[0]
                )
                act = int(
                    actuation_regimes(
                        np.asarray([trial.arrays["a_nom"][middle]]),
                        config.sampling.actuation_threshold_mps2,
                    )[0]
                )
                self.references.append(WindowReference(trial_index, start, pbin, vbin, act))
        if not self.references:
            raise RuntimeError("No valid RBF windows were found.")
        self.by_joint: dict[tuple[int, int], dict[str, dict[str, list[WindowReference]]]] = {}
        self.by_actuation: dict[
            tuple[int, int], dict[int, dict[str, dict[str, list[WindowReference]]]]
        ] = {}
        for ref in self.references:
            trial = self.trials[ref.trial_index]
            key = (ref.position_bin, ref.velocity_bin)
            self.by_joint.setdefault(key, {}).setdefault(trial.session, {}).setdefault(
                trial.group, []
            ).append(ref)
            self.by_actuation.setdefault(key, {}).setdefault(
                ref.actuation_regime, {}
            ).setdefault(trial.session, {}).setdefault(trial.group, []).append(ref)

    def make_batch(self, references: Sequence[WindowReference], device: torch.device):
        return self.legacy.make_batch(references, device)

    @staticmethod
    def _hierarchical_pick(
        rng: np.random.Generator,
        value: dict[str, dict[str, list[WindowReference]]],
    ) -> WindowReference:
        sessions = sorted(value)
        session = sessions[int(rng.integers(len(sessions)))]
        groups = sorted(value[session])
        group = groups[int(rng.integers(len(groups)))]
        candidates = value[session][group]
        return candidates[int(rng.integers(len(candidates)))]

    def sample(self, rng: np.random.Generator, batch_size: int) -> list[WindowReference]:
        empirical = int(round(batch_size * self.config.sampling.empirical_fraction))
        joint = int(round(batch_size * self.config.sampling.joint_fraction))
        actuation = batch_size - empirical - joint
        result = [
            self.references[int(rng.integers(len(self.references)))] for _ in range(empirical)
        ]
        joint_keys = sorted(self.by_joint)
        for _ in range(joint):
            key = joint_keys[int(rng.integers(len(joint_keys)))]
            result.append(self._hierarchical_pick(rng, self.by_joint[key]))
        act_keys = sorted(self.by_actuation)
        for _ in range(actuation):
            key = act_keys[int(rng.integers(len(act_keys)))]
            regimes = sorted(self.by_actuation[key])
            regime = regimes[int(rng.integers(len(regimes)))]
            result.append(self._hierarchical_pick(rng, self.by_actuation[key][regime]))
        rng.shuffle(result)
        return result


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def trapezoidal_average(values: torch.Tensor, dt: torch.Tensor, horizon: int) -> torch.Tensor:
    duration = dt[:, :horizon].sum(dim=1).clamp_min(1.0e-12)
    integral = (
        0.5
        * (values[:, :horizon] + values[:, 1 : horizon + 1])
        * dt[:, :horizon]
    ).sum(dim=1)
    return integral / duration


def integral_errors(
    model: ResidualRBF2D, batch: Any, config: Config
) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    prediction = model(batch.features)
    result: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    for horizon in config.identification_loss.horizons:
        duration = batch.dt[:, :horizon].sum(dim=1).clamp_min(1.0e-12)
        nominal_average = trapezoidal_average(batch.features[..., 4], batch.dt, horizon)
        residual_average = trapezoidal_average(prediction, batch.dt, horizon)
        target = (
            (batch.smooth_vb[:, horizon] - batch.smooth_vb[:, 0]) / duration
            - nominal_average
        )
        result[horizon] = (residual_average - target, -target)
    return result


def curvature_loss(model: ResidualRBF2D) -> torch.Tensor:
    grid = model.coefficient_grid() / model.output_limit_mps2
    first_p = grid[1:, :] - grid[:-1, :]
    first_v = grid[:, 1:] - grid[:, :-1]
    second_p = grid[2:, :] - 2.0 * grid[1:-1, :] + grid[:-2, :]
    second_v = grid[:, 2:] - 2.0 * grid[:, 1:-1] + grid[:, :-2]
    return (
        first_p.square().mean()
        + first_v.square().mean()
        + second_p.square().mean()
        + second_v.square().mean()
    )


def deterministic_collocation(batch_size: int, config: Config, device: torch.device) -> torch.Tensor:
    count = max(1, int(round(batch_size * config.identification_loss.collocation_fraction)))
    base = (torch.arange(count, device=device, dtype=torch.float64) + 0.5) / count
    pb = 0.7 * base
    vb = -0.8 + 1.6 * torch.roll(base, shifts=max(1, count // 3))
    return torch.stack((pb, vb), dim=-1)


def derivative_terms(
    model: ResidualRBF2D, batch: Any, config: Config, *, create_graph: bool
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    midpoint = batch.features.shape[1] // 2
    real = batch.features[:, midpoint, :2].detach()
    collocation = deterministic_collocation(real.shape[0], config, real.device)
    points = torch.cat((real, collocation), dim=0).requires_grad_(True)
    values = model(points)
    gradient = torch.autograd.grad(values.sum(), points, create_graph=create_graph)[0]
    rp, rv = gradient[:, 0], gradient[:, 1]
    ident = config.identification_loss
    velocity = (
        torch.relu((ident.velocity_derivative_soft_min_sinv - rv) / 0.1).square()
        + torch.relu((rv - ident.velocity_derivative_soft_max_sinv) / 0.1).square()
    ).mean()
    position = torch.relu(
        (rp.abs() - ident.position_derivative_soft_abs_max_sinv2) / 0.5
    ).square().mean()
    return velocity, position, gradient


def identification_loss(
    model: ResidualRBF2D,
    batch: Any,
    config: Config,
    *,
    create_graph: bool,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    ident = config.identification_loss
    errors = integral_errors(model, batch, config)
    integral = torch.zeros((), dtype=torch.float64, device=batch.features.device)
    details: dict[str, torch.Tensor] = {}
    for horizon, weight in zip(ident.horizons, ident.horizon_weights):
        model_error, nominal_error = errors[horizon]
        loss = F.smooth_l1_loss(
            model_error, torch.zeros_like(model_error), beta=ident.huber_beta_mps2
        )
        integral = integral + weight * loss
        details[f"integral_model_mse_h{horizon}"] = model_error.square().mean()
        details[f"integral_nominal_mse_h{horizon}"] = nominal_error.square().mean()
    point_prediction = model(batch.features)
    point_error = point_prediction - batch.residual_target
    point = F.smooth_l1_loss(
        point_prediction, batch.residual_target, beta=ident.huber_beta_mps2
    )
    curve = curvature_loss(model)
    velocity_derivative, position_derivative, gradient = derivative_terms(
        model, batch, config, create_graph=create_graph
    )
    total = (
        ident.integral_weight * integral
        + ident.point_weight * point
        + ident.curvature_weight * curve
        + ident.velocity_derivative_weight * velocity_derivative
        + ident.position_derivative_weight * position_derivative
    )
    details.update(
        {
            "identification_loss": total,
            "integral_loss": integral,
            "point_loss": point,
            "curvature_loss": curve,
            "velocity_derivative_loss": velocity_derivative,
            "position_derivative_loss": position_derivative,
            "point_model_mse": point_error.square().mean(),
            "point_nominal_mse": batch.residual_target.square().mean(),
            "residual_saturation_rate": (
                point_prediction.abs() >= 0.95 * model.output_limit_mps2
            ).to(torch.float64).mean(),
            "rp_min": gradient[:, 0].min(),
            "rp_max": gradient[:, 0].max(),
            "rv_min": gradient[:, 1].min(),
            "rv_max": gradient[:, 1].max(),
        }
    )
    return total, details


def causal_rollout_loss(
    model: ResidualRBF2D, batch: Any, config: Config
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    rollout = rollout_causal_dynamics(
        model,
        batch.causal_state[:, 0],
        batch.commands,
        batch.dt,
        batch.constants,
        batch.geometry,
        batch.response,
    )
    nominal = rollout_causal_dynamics(
        None,
        batch.causal_state[:, 0],
        batch.commands,
        batch.dt,
        batch.constants,
        batch.geometry,
        batch.response,
    )
    cfg = config.rollout_loss
    total = torch.zeros((), dtype=torch.float64, device=batch.features.device)
    details: dict[str, torch.Tensor] = {}
    for horizon, weight in zip(cfg.horizons, cfg.horizon_weights):
        smooth_p = (rollout.states[:, horizon, 0] - batch.smooth_pb[:, horizon]) / cfg.position_scale_m
        smooth_v = (rollout.states[:, horizon, 1] - batch.smooth_vb[:, horizon]) / cfg.velocity_scale_mps
        causal_p = (rollout.states[:, horizon, 0] - batch.causal_state[:, horizon, 0]) / cfg.position_scale_m
        causal_v = (rollout.states[:, horizon, 1] - batch.causal_state[:, horizon, 1]) / cfg.velocity_scale_mps
        horizon_loss = (
            F.smooth_l1_loss(smooth_p, torch.zeros_like(smooth_p), beta=cfg.huber_beta_normalized)
            + F.smooth_l1_loss(smooth_v, torch.zeros_like(smooth_v), beta=cfg.huber_beta_normalized)
            + cfg.causal_auxiliary_weight
            * (
                F.smooth_l1_loss(causal_p, torch.zeros_like(causal_p), beta=cfg.huber_beta_normalized)
                + F.smooth_l1_loss(causal_v, torch.zeros_like(causal_v), beta=cfg.huber_beta_normalized)
            )
        )
        total = total + weight * horizon_loss
        details[f"rbf_pb_mse_h{horizon}"] = (
            rollout.states[:, horizon, 0] - batch.causal_state[:, horizon, 0]
        ).square().mean()
        details[f"rbf_vb_mse_h{horizon}"] = (
            rollout.states[:, horizon, 1] - batch.causal_state[:, horizon, 1]
        ).square().mean()
        details[f"nominal_pb_mse_h{horizon}"] = (
            nominal.states[:, horizon, 0] - batch.causal_state[:, horizon, 0]
        ).square().mean()
        details[f"nominal_vb_mse_h{horizon}"] = (
            nominal.states[:, horizon, 1] - batch.causal_state[:, horizon, 1]
        ).square().mean()
    details["causal_rollout_loss"] = total
    return total, details


def anchor_grid(device: torch.device) -> torch.Tensor:
    pb = torch.linspace(0.0, 0.7, 41, device=device, dtype=torch.float64)
    vb = torch.linspace(-0.8, 0.8, 41, device=device, dtype=torch.float64)
    grid_p, grid_v = torch.meshgrid(pb, vb, indexing="ij")
    return torch.stack((grid_p.reshape(-1), grid_v.reshape(-1)), dim=-1)


def stage2_loss(
    model: ResidualRBF2D,
    batch: Any,
    config: Config,
    anchors: torch.Tensor,
    reference_outputs: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    ident, details = identification_loss(model, batch, config, create_graph=True)
    rollout, rollout_details = causal_rollout_loss(model, batch, config)
    difference = (model(anchors) - reference_outputs).abs()
    trust = torch.relu(
        difference / config.rollout_loss.trust_output_threshold_mps2 - 1.0
    ).square().mean()
    total = (
        ident
        + config.rollout_loss.rollout_weight * rollout
        + config.rollout_loss.trust_weight * trust
    )
    details.update(rollout_details)
    details.update({"trust_loss": trust, "total_loss": total})
    return total, details


def _mean_details(accumulator: dict[str, float], details: Mapping[str, torch.Tensor], count: int) -> None:
    for key, value in details.items():
        if value.numel() == 1:
            accumulator[key] = accumulator.get(key, 0.0) + float(value.detach().cpu()) * count


def _validation_refs(store: WindowStore, config: Config) -> list[WindowReference]:
    refs = list(store.references[:: config.training.validation_stride])
    if len(refs) > config.training.validation_max_windows:
        indices = np.linspace(0, len(refs) - 1, config.training.validation_max_windows, dtype=int)
        refs = [refs[index] for index in indices]
    return refs


def validate(
    model: ResidualRBF2D,
    store: WindowStore,
    references: Sequence[WindowReference],
    config: Config,
    *,
    include_rollout: bool,
    batch_size: int,
) -> dict[str, float]:
    model.eval()
    accumulator: dict[str, float] = {}
    count = 0
    with torch.enable_grad():
        for offset in range(0, len(references), batch_size):
            refs = references[offset : offset + batch_size]
            batch = store.make_batch(refs, torch.device("cpu"))
            if include_rollout:
                _, details = stage2_loss(
                    model,
                    batch,
                    config,
                    anchor_grid(torch.device("cpu")),
                    model(anchor_grid(torch.device("cpu"))).detach(),
                )
            else:
                total, details = identification_loss(
                    model, batch, config, create_graph=False
                )
                details["total_loss"] = total
            _mean_details(accumulator, details, len(refs))
            count += len(refs)
    result = {key: value / count for key, value in accumulator.items()}
    if include_rollout:
        result["selection_score"] = selection_score(result)
    return result


def selection_score(metrics: Mapping[str, float]) -> float:
    integral_ratios = []
    for horizon in (3, 6, 12):
        integral_ratios.append(
            math.sqrt(metrics[f"integral_model_mse_h{horizon}"])
            / max(math.sqrt(metrics[f"integral_nominal_mse_h{horizon}"]), 1.0e-12)
        )
    integral = float(np.average(integral_ratios, weights=(0.5, 0.3, 0.2)))
    point = math.sqrt(metrics["point_model_mse"]) / max(
        math.sqrt(metrics["point_nominal_mse"]), 1.0e-12
    )
    pb = math.sqrt(metrics["rbf_pb_mse_h12"]) / max(
        math.sqrt(metrics["nominal_pb_mse_h12"]), 1.0e-12
    )
    vb = math.sqrt(metrics["rbf_vb_mse_h12"]) / max(
        math.sqrt(metrics["nominal_vb_mse_h12"]), 1.0e-12
    )
    return 0.50 * integral + 0.20 * point + 0.15 * pb + 0.15 * vb


def _epoch_safe(metrics: Mapping[str, float], config: Config) -> bool:
    accept = config.acceptance
    return bool(
        metrics.get("residual_saturation_rate", math.inf) <= accept.saturation_rate_max
        and metrics.get("rv_min", -math.inf) >= accept.rv_min_sinv
        and metrics.get("rv_max", math.inf) <= accept.rv_max_sinv
        and max(abs(metrics.get("rp_min", math.inf)), abs(metrics.get("rp_max", math.inf)))
        <= accept.rp_abs_max_sinv2
    )


def should_select_stage2(
    stage1_score: float, stage2_metrics: Mapping[str, float], config: Config
) -> bool:
    """Keep stage 1 unless stage 2 is safe and strictly improves validation."""

    return bool(
        _epoch_safe(stage2_metrics, config)
        and float(stage2_metrics["selection_score"]) < float(stage1_score)
    )


def _write_history(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _save_checkpoint(
    path: Path,
    model: ResidualRBF2D,
    config: Config,
    manifest: Mapping[str, Any],
    stage: str,
    epoch: int,
    validation: Mapping[str, Any],
) -> None:
    torch.save(
        checkpoint_payload(
            model, config, manifest, stage=stage, epoch=epoch, validation=validation
        ),
        path,
    )


def _train_stage(
    model: ResidualRBF2D,
    config: Config,
    manifest: Mapping[str, Any],
    train_store: WindowStore,
    validation_store: WindowStore,
    validation_refs: Sequence[WindowReference],
    stage_name: str,
    stage_config: StageConfig,
    checkpoint_path: Path,
    history: list[dict[str, Any]],
    *,
    reference_coefficients: torch.Tensor | None = None,
) -> dict[str, Any]:
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=stage_config.learning_rate, weight_decay=0.0
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=config.training.lr_factor, patience=config.training.lr_patience
    )
    rng = np.random.default_rng(config.training.seed + (0 if stage_name == "stage1" else 1000))
    anchors = anchor_grid(torch.device("cpu"))
    reference_outputs = None
    if reference_coefficients is not None:
        with torch.no_grad():
            reference_outputs = model(anchors).detach().clone()
    best_value = math.inf
    best_epoch = 0
    stale = 0
    stop_reason = "max_epochs"
    started = time.perf_counter()
    for epoch in range(1, stage_config.max_epochs + 1):
        model.train()
        train_sum = 0.0
        epoch_started = time.perf_counter()
        for _ in range(stage_config.steps_per_epoch):
            refs = train_store.sample(rng, stage_config.batch_size)
            batch = train_store.make_batch(refs, torch.device("cpu"))
            optimizer.zero_grad(set_to_none=True)
            if stage_name == "stage1":
                loss, _ = identification_loss(model, batch, config, create_graph=True)
            else:
                assert reference_outputs is not None
                loss, _ = stage2_loss(model, batch, config, anchors, reference_outputs)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite {stage_name} loss.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.gradient_clip_norm)
            optimizer.step()
            model.project_coefficients(
                reference_coefficients,
                config.rollout_loss.coefficient_radius_mps2
                if reference_coefficients is not None
                else None,
            )
            train_sum += float(loss.detach())
        validation = validate(
            model,
            validation_store,
            validation_refs,
            config,
            include_rollout=stage_name == "stage2",
            batch_size=stage_config.batch_size,
        )
        objective = (
            validation["selection_score"]
            if stage_name == "stage2"
            else validation["total_loss"]
        )
        safe = _epoch_safe(validation, config)
        scheduler.step(objective)
        improved = safe and objective < best_value - 1.0e-12
        if improved:
            best_value = objective
            best_epoch = epoch
            stale = 0
            _save_checkpoint(
                checkpoint_path, model, config, manifest, stage_name, epoch, validation
            )
        else:
            stale += 1
        history.append(
            {
                "stage": stage_name,
                "epoch": epoch,
                "train_total_loss": train_sum / stage_config.steps_per_epoch,
                "validation_objective": objective,
                "validation_safe": int(safe),
                "best_epoch": best_epoch,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "epoch_time_s": time.perf_counter() - epoch_started,
                **{f"validation_{key}": value for key, value in validation.items()},
            }
        )
        if stale >= stage_config.early_stopping_patience:
            stop_reason = "early_stopping"
            break
    if not checkpoint_path.is_file():
        # Preserve a diagnostic checkpoint even when no epoch passes the online gates.
        validation = validate(
            model,
            validation_store,
            validation_refs,
            config,
            include_rollout=stage_name == "stage2",
            batch_size=stage_config.batch_size,
        )
        _save_checkpoint(
            checkpoint_path, model, config, manifest, stage_name, stage_config.max_epochs, validation
        )
        best_value = validation.get("selection_score", validation["total_loss"])
        best_epoch = stage_config.max_epochs
    return {
        "stage": stage_name,
        "best_epoch": best_epoch,
        "best_objective": best_value,
        "epochs_completed": epoch,
        "stop_reason": stop_reason,
        "training_time_s": time.perf_counter() - started,
    }


def train_model(config: Config) -> Path:
    manifest = load_manifest(config)
    train_trials = load_trials(config, "train")
    validation_trials = load_trials(config, "validation")
    max_horizon = max(config.rollout_loss.horizons)
    train_store = WindowStore(train_trials, max_horizon, config, manifest)
    validation_store = WindowStore(validation_trials, max_horizon, config, manifest)
    validation_refs = _validation_refs(validation_store, config)
    set_seed(config.training.seed)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"rbf2_seed{config.training.seed}_{config.config_hash[:8]}_{timestamp}"
    run_dir = config.paths.training_runs_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    history: list[dict[str, Any]] = []
    total_started = time.perf_counter()

    model = create_model(config, manifest)
    stage1_path = run_dir / "stage1_best_checkpoint.pt"
    stage1 = _train_stage(
        model,
        config,
        manifest,
        train_store,
        validation_store,
        validation_refs,
        "stage1",
        config.training.stage1,
        stage1_path,
        history,
    )
    stage1_model, stage1_checkpoint = load_checkpoint(stage1_path)
    stage1_full = validate(
        stage1_model,
        validation_store,
        validation_refs,
        config,
        include_rollout=True,
        batch_size=config.training.stage2.batch_size,
    )
    stage1_score = stage1_full["selection_score"]

    model.load_state_dict(stage1_model.state_dict())
    reference_coefficients = stage1_model.coefficients.detach().clone()
    stage2_path = run_dir / "stage2_best_checkpoint.pt"
    stage2 = _train_stage(
        model,
        config,
        manifest,
        train_store,
        validation_store,
        validation_refs,
        "stage2",
        config.training.stage2,
        stage2_path,
        history,
        reference_coefficients=reference_coefficients,
    )
    stage2_model, stage2_checkpoint = load_checkpoint(stage2_path)
    stage2_metrics = dict(stage2_checkpoint["validation"])
    stage2_score = float(stage2_metrics["selection_score"])
    stage2_selected = should_select_stage2(stage1_score, stage2_metrics, config)
    final_source = stage2_path if stage2_selected else stage1_path
    final_path = run_dir / "best_checkpoint.pt"
    shutil.copy2(final_source, final_path)
    final_model, final_checkpoint = load_checkpoint(final_path)
    _write_history(run_dir / "history.csv", history)
    summary = {
        "run_name": run_name,
        "run_dir": str(run_dir),
        "seed": config.training.seed,
        "config_hash": config.config_hash,
        "dataset_hash": manifest["dataset_hash"],
        "effective_cap_mps2": final_model.output_limit_mps2,
        "stage1": stage1 | {"selection_score": stage1_score},
        "stage2": stage2 | {"selection_score": stage2_score},
        "stage2_selected": stage2_selected,
        "selected_stage": "stage2" if stage2_selected else "stage1",
        "selected_epoch": int(final_checkpoint["epoch"]),
        "total_training_time_s": time.perf_counter() - total_started,
        "checkpoint_sha256": _sha256(final_path),
        "training_device": "cpu",
        "validation_reference_count": len(validation_refs),
    }
    _write_json(run_dir / "training_summary.json", summary)
    _write_json(config.paths.training_runs_dir / "latest.json", {"run_dir": str(run_dir)})
    # Eligibility is finalized by evaluate; this diagnostic export is intentionally false.
    golden = np.asarray(
        [[0.0, 0.0], [0.1, -0.2], [0.15, 0.0], [0.35, -0.1], [0.35, 0.0],
         [0.35, 0.1], [0.55, -0.2], [0.55, 0.0], [0.7, 0.2], [0.8, 0.8]],
        dtype=np.float64,
    )
    payload = portable_payload(
        final_model,
        {
            **summary,
            "split_counts": manifest["split_counts"],
            "source_dirs": manifest["source_dirs"],
            "eligible_for_residual_integration": False,
            "eligible_for_real_world_deployment": False,
            "eligibility_pending_evaluation": True,
        },
        golden,
    )
    save_portable_json(run_dir / "model.json", payload)
    return run_dir


def resolve_run_dir(config: Config, value: str | Path | None = None) -> Path:
    if value is not None:
        path = Path(value).expanduser().resolve()
    else:
        latest = config.paths.training_runs_dir / "latest.json"
        if not latest.is_file():
            raise FileNotFoundError("No latest RBF training run exists.")
        with latest.open("r", encoding="utf-8") as stream:
            path = Path(json.load(stream)["run_dir"])
    if not (path / "best_checkpoint.pt").is_file():
        raise FileNotFoundError(f"Missing best checkpoint in {path}")
    return path
