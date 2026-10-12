"""Coarse train-only ablation for local beam profile and residual structure.

This is a diagnostic experiment, not a deployable model exporter. It compares
bounded local-profile, structured-friction, and two-dimensional black-box terms on
the immutable validation/test/stress splits used by the RBF-PV workflow.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from residual_dynamics.dynamics import rollout_causal_dynamics
from residual_dynamics.model import load_portable_weights

from .config import Config, load_config
from .data import load_manifest, load_trials
from .evaluation import _integral_metrics, _refs_for_evaluation, _rollout_metrics, regression
from .model import load_checkpoint
from .training import WindowReference, WindowStore, integral_errors, resolve_run_dir


DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "rbf_2input_causal_success.yaml"
RESULT_PATH = Path(__file__).resolve().parent / "results_profile_friction_probe" / "summary.json"


@dataclass(frozen=True)
class ProbeSettings:
    seed: int = 7
    center_count: int = 13
    width_m: float = 0.065
    velocity_transition_mps: float = 0.025
    delta_limit_rad: float = 0.035
    coulomb_limit_mps2: float = 0.15
    viscous_limit_sinv: float = 0.50
    blackbox_limit_mps2: float = 0.25
    kappa_limit: float = 0.10
    output_limit_mps2: float = 0.25
    stage1_epochs: int = 80
    stage1_steps: int = 16
    stage2_epochs: int = 50
    stage2_steps: int = 8
    batch_size: int = 256
    validation_windows: int = 512


class ProfileFrictionResidual(nn.Module):
    """Bounded local-profile term plus optional structured or black-box residual."""

    def __init__(
        self,
        *,
        profile_enabled: bool,
        friction_enabled: bool,
        blackbox_enabled: bool = False,
        kappa_enabled: bool = False,
        acceleration_gain: float,
        gravity_mps2: float,
        settings: ProbeSettings,
    ) -> None:
        super().__init__()
        self.profile_enabled = bool(profile_enabled)
        self.friction_enabled = bool(friction_enabled)
        self.blackbox_enabled = bool(blackbox_enabled)
        self.kappa_enabled = bool(kappa_enabled)
        self.acceleration_gain = float(acceleration_gain)
        self.gravity_mps2 = float(gravity_mps2)
        self.settings = settings
        self.output_limit_mps2 = settings.output_limit_mps2
        self.register_buffer(
            "centers",
            torch.linspace(0.0, 0.7, settings.center_count, dtype=torch.float64),
        )
        blackbox_pb = torch.linspace(0.0, 0.7, 9, dtype=torch.float64)
        blackbox_vb = torch.tensor(
            [-0.50, -0.30, -0.15, -0.06, 0.0, 0.06, 0.15, 0.30, 0.50],
            dtype=torch.float64,
        )
        grid_pb, grid_vb = torch.meshgrid(blackbox_pb, blackbox_vb, indexing="ij")
        self.register_buffer(
            "blackbox_centers",
            torch.stack((grid_pb.reshape(-1), grid_vb.reshape(-1)), dim=-1),
        )
        self.delta_coefficients = nn.Parameter(
            torch.zeros(settings.center_count, dtype=torch.float64),
            requires_grad=self.profile_enabled,
        )
        self.coulomb_coefficients = nn.Parameter(
            torch.zeros(settings.center_count, dtype=torch.float64),
            requires_grad=self.friction_enabled,
        )
        self.viscous_coefficients = nn.Parameter(
            torch.zeros(settings.center_count, dtype=torch.float64),
            requires_grad=self.friction_enabled,
        )
        self.blackbox_coefficients = nn.Parameter(
            torch.zeros(81, dtype=torch.float64),
            requires_grad=self.blackbox_enabled,
        )
        self.kappa_coefficients = nn.Parameter(
            torch.zeros(81, dtype=torch.float64),
            requires_grad=self.kappa_enabled,
        )

    def basis(self, position: torch.Tensor) -> torch.Tensor:
        phi = torch.exp(
            -0.5
            * ((position[..., None] - self.centers) / self.settings.width_m).square()
        )
        return phi / (1.0e-12 + phi.sum(dim=-1, keepdim=True))

    def blackbox_basis(
        self, position: torch.Tensor, velocity: torch.Tensor
    ) -> torch.Tensor:
        delta_p = (position[..., None] - self.blackbox_centers[:, 0]) / 0.10
        delta_v = (velocity[..., None] - self.blackbox_centers[:, 1]) / 0.10
        phi = torch.exp(-0.5 * (delta_p.square() + delta_v.square()))
        return phi / (1.0e-12 + phi.sum(dim=-1, keepdim=True))

    def components(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        position, velocity, theta, a_nom = (
            features[..., 0],
            features[..., 1],
            features[..., 2],
            features[..., 4],
        )
        basis = self.basis(position)
        delta = basis @ self.delta_coefficients
        profile = -self.acceleration_gain * self.gravity_mps2 * (
            torch.sin(theta + delta) - torch.sin(theta)
        )
        coulomb = basis @ self.coulomb_coefficients
        viscous = basis @ self.viscous_coefficients
        friction = -coulomb * torch.tanh(
            velocity / self.settings.velocity_transition_mps
        ) - viscous * velocity
        blackbox_basis = self.blackbox_basis(position, velocity)
        blackbox = blackbox_basis @ self.blackbox_coefficients
        kappa = blackbox_basis @ self.kappa_coefficients
        gain_residual = kappa * a_nom
        if not self.profile_enabled:
            profile = torch.zeros_like(profile)
            delta = torch.zeros_like(delta)
        if not self.friction_enabled:
            friction = torch.zeros_like(friction)
            coulomb = torch.zeros_like(coulomb)
            viscous = torch.zeros_like(viscous)
        if not self.blackbox_enabled:
            blackbox = torch.zeros_like(blackbox)
        if not self.kappa_enabled:
            kappa = torch.zeros_like(kappa)
            gain_residual = torch.zeros_like(gain_residual)
        raw = profile + friction + blackbox + gain_residual
        applied = self.output_limit_mps2 * torch.tanh(raw / self.output_limit_mps2)
        return {
            "applied": applied,
            "raw": raw,
            "profile": profile,
            "friction": friction,
            "blackbox": blackbox,
            "kappa": kappa,
            "gain_residual": gain_residual,
            "delta": delta,
            "coulomb": coulomb,
            "viscous": viscous,
        }

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.components(features)["applied"]

    @torch.no_grad()
    def project(self) -> None:
        self.delta_coefficients.clamp_(
            -self.settings.delta_limit_rad, self.settings.delta_limit_rad
        )
        self.coulomb_coefficients.clamp_(0.0, self.settings.coulomb_limit_mps2)
        self.viscous_coefficients.clamp_(0.0, self.settings.viscous_limit_sinv)
        self.blackbox_coefficients.clamp_(
            -self.settings.blackbox_limit_mps2,
            self.settings.blackbox_limit_mps2,
        )
        self.kappa_coefficients.clamp_(
            -self.settings.kappa_limit,
            self.settings.kappa_limit,
        )

    def regularization(self) -> torch.Tensor:
        terms = []
        for values, scale in (
            (self.delta_coefficients, self.settings.delta_limit_rad),
            (self.coulomb_coefficients, self.settings.coulomb_limit_mps2),
            (self.viscous_coefficients, self.settings.viscous_limit_sinv),
        ):
            if values.requires_grad:
                normalized = values / scale
                first = normalized[1:] - normalized[:-1]
                second = normalized[2:] - 2.0 * normalized[1:-1] + normalized[:-2]
                terms.append(0.002 * first.square().mean())
                terms.append(0.01 * second.square().mean())
        if self.blackbox_coefficients.requires_grad:
            grid = self.blackbox_coefficients.reshape(9, 9) / self.settings.blackbox_limit_mps2
            first_p = grid[1:, :] - grid[:-1, :]
            first_v = grid[:, 1:] - grid[:, :-1]
            second_p = grid[2:, :] - 2.0 * grid[1:-1, :] + grid[:-2, :]
            second_v = grid[:, 2:] - 2.0 * grid[:, 1:-1] + grid[:, :-2]
            terms.extend(
                (
                    0.002 * first_p.square().mean(),
                    0.002 * first_v.square().mean(),
                    0.01 * second_p.square().mean(),
                    0.01 * second_v.square().mean(),
                )
            )
        if self.kappa_coefficients.requires_grad:
            grid = self.kappa_coefficients.reshape(9, 9) / self.settings.kappa_limit
            first_p = grid[1:, :] - grid[:-1, :]
            first_v = grid[:, 1:] - grid[:, :-1]
            second_p = grid[2:, :] - 2.0 * grid[1:-1, :] + grid[:-2, :]
            second_v = grid[:, 2:] - 2.0 * grid[:, 1:-1] + grid[:, :-2]
            terms.extend(
                (
                    0.002 * first_p.square().mean(),
                    0.002 * first_v.square().mean(),
                    0.01 * second_p.square().mean(),
                    0.01 * second_v.square().mean(),
                )
            )
        if not terms:
            return torch.zeros((), dtype=torch.float64)
        return torch.stack(terms).sum()


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def _integral_loss(
    model: ProfileFrictionResidual, batch: Any, config: Config
) -> torch.Tensor:
    total = torch.zeros((), dtype=torch.float64)
    values = integral_errors(model, batch, config)
    for horizon, weight in zip((3, 6, 12), (0.50, 0.30, 0.20)):
        error = values[horizon][0]
        total = total + weight * F.smooth_l1_loss(
            error, torch.zeros_like(error), beta=0.02
        )
    return total


def _rollout_loss(
    model: ProfileFrictionResidual, batch: Any
) -> torch.Tensor:
    rollout = rollout_causal_dynamics(
        model,
        batch.causal_state[:, 0],
        batch.commands,
        batch.dt,
        batch.constants,
        batch.geometry,
        batch.response,
    )
    total = torch.zeros((), dtype=torch.float64)
    for horizon, weight in zip((6, 12, 30, 33), (0.35, 0.30, 0.20, 0.15)):
        smooth_p = (rollout.states[:, horizon, 0] - batch.smooth_pb[:, horizon]) / 0.01
        smooth_v = (rollout.states[:, horizon, 1] - batch.smooth_vb[:, horizon]) / 0.05
        causal_p = (rollout.states[:, horizon, 0] - batch.causal_state[:, horizon, 0]) / 0.01
        causal_v = (rollout.states[:, horizon, 1] - batch.causal_state[:, horizon, 1]) / 0.05
        loss = F.smooth_l1_loss(smooth_p, torch.zeros_like(smooth_p), beta=1.0)
        loss = loss + F.smooth_l1_loss(smooth_v, torch.zeros_like(smooth_v), beta=1.0)
        loss = loss + 0.25 * (
            F.smooth_l1_loss(causal_p, torch.zeros_like(causal_p), beta=1.0)
            + F.smooth_l1_loss(causal_v, torch.zeros_like(causal_v), beta=1.0)
        )
        total = total + weight * loss
    return total


def _loss(
    model: ProfileFrictionResidual,
    batch: Any,
    config: Config,
    *,
    rollout_weight: float,
) -> torch.Tensor:
    prediction = model(batch.features)
    point = F.smooth_l1_loss(prediction, batch.residual_target, beta=0.02)
    integral = _integral_loss(model, batch, config)
    rollout = (
        _rollout_loss(model, batch)
        if rollout_weight > 0.0
        else torch.zeros((), dtype=torch.float64)
    )
    return integral + 0.25 * point + rollout_weight * rollout + model.regularization()


def _selection_metrics(
    model: ProfileFrictionResidual,
    store: WindowStore,
    references: Sequence[WindowReference],
    config: Config,
) -> dict[str, float]:
    sums = {
        "point_model": 0.0,
        "point_nominal": 0.0,
        "integral_model": 0.0,
        "integral_nominal": 0.0,
        "h33_pb_model": 0.0,
        "h33_vb_model": 0.0,
        "h33_pb_nominal": 0.0,
        "h33_vb_nominal": 0.0,
        "count": 0.0,
        "point_count": 0.0,
    }
    for offset in range(0, len(references), 256):
        refs = references[offset : offset + 256]
        batch = store.make_batch(refs, torch.device("cpu"))
        with torch.no_grad():
            prediction = model(batch.features)
            point_error = prediction - batch.residual_target
            sums["point_model"] += float(point_error.square().sum())
            sums["point_nominal"] += float(batch.residual_target.square().sum())
            sums["point_count"] += float(point_error.numel())
            errors = integral_errors(model, batch, config)
            for horizon, weight in zip((3, 6, 12), (0.50, 0.30, 0.20)):
                model_error, nominal_error = errors[horizon]
                sums["integral_model"] += weight * float(model_error.square().sum())
                sums["integral_nominal"] += weight * float(nominal_error.square().sum())
            residual = rollout_causal_dynamics(
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
            for name, rollout in (("model", residual), ("nominal", nominal)):
                sums[f"h33_pb_{name}"] += float(
                    (rollout.states[:, 33, 0] - batch.causal_state[:, 33, 0]).square().sum()
                )
                sums[f"h33_vb_{name}"] += float(
                    (rollout.states[:, 33, 1] - batch.causal_state[:, 33, 1]).square().sum()
                )
            sums["count"] += len(refs)
    point_ratio = math.sqrt(sums["point_model"] / sums["point_count"]) / max(
        math.sqrt(sums["point_nominal"] / sums["point_count"]), 1.0e-12
    )
    integral_ratio = math.sqrt(sums["integral_model"] / sums["count"]) / max(
        math.sqrt(sums["integral_nominal"] / sums["count"]), 1.0e-12
    )
    pb_ratio = math.sqrt(sums["h33_pb_model"] / sums["count"]) / max(
        math.sqrt(sums["h33_pb_nominal"] / sums["count"]), 1.0e-12
    )
    vb_ratio = math.sqrt(sums["h33_vb_model"] / sums["count"]) / max(
        math.sqrt(sums["h33_vb_nominal"] / sums["count"]), 1.0e-12
    )
    return {
        "point_ratio": point_ratio,
        "integral_ratio": integral_ratio,
        "h33_pb_ratio": pb_ratio,
        "h33_vb_ratio": vb_ratio,
        "score": 0.15 * point_ratio
        + 0.20 * integral_ratio
        + 0.325 * pb_ratio
        + 0.325 * vb_ratio,
    }


def _fit(
    name: str,
    model: ProfileFrictionResidual,
    train_store: WindowStore,
    validation_store: WindowStore,
    validation_refs: Sequence[WindowReference],
    config: Config,
    settings: ProbeSettings,
) -> tuple[ProfileFrictionResidual, dict[str, Any]]:
    _seed(settings.seed)
    rng = np.random.default_rng(settings.seed)
    history = []
    best_state = copy.deepcopy(model.state_dict())
    best_score = math.inf
    for stage, epochs, steps, learning_rate, rollout_weight in (
        ("stage1", settings.stage1_epochs, settings.stage1_steps, 0.003, 0.0),
        ("stage2", settings.stage2_epochs, settings.stage2_steps, 0.0003, 0.50),
    ):
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        stale = 0
        patience = 20
        for epoch in range(1, epochs + 1):
            model.train()
            accumulated = 0.0
            for _ in range(steps):
                refs = train_store.sample(rng, settings.batch_size)
                batch = train_store.make_batch(refs, torch.device("cpu"))
                optimizer.zero_grad(set_to_none=True)
                loss = _loss(
                    model, batch, config, rollout_weight=rollout_weight
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                model.project()
                accumulated += float(loss.detach())
            metrics = _selection_metrics(
                model, validation_store, validation_refs, config
            )
            improved = metrics["score"] < best_score - 1.0e-6
            if improved:
                best_score = metrics["score"]
                best_state = copy.deepcopy(model.state_dict())
                stale = 0
            else:
                stale += 1
            history.append(
                {
                    "stage": stage,
                    "epoch": epoch,
                    "train_loss": accumulated / steps,
                    **metrics,
                    "best_score": best_score,
                }
            )
            if stale >= patience:
                break
        model.load_state_dict(best_state)
    return model, {
        "name": name,
        "best_validation_score": best_score,
        "epochs": {
            stage: sum(row["stage"] == stage for row in history)
            for stage in ("stage1", "stage2")
        },
        "history": history,
    }


def _point(model: nn.Module | None, trials: Sequence[Any]) -> dict[str, Any]:
    features = np.concatenate(
        [trial.arrays["features"][trial.arrays["rbf_valid"].astype(bool)] for trial in trials]
    )
    targets = np.concatenate(
        [trial.arrays["residual_target"][trial.arrays["rbf_valid"].astype(bool)] for trial in trials]
    )
    if model is None:
        prediction = np.zeros_like(targets)
    else:
        with torch.no_grad():
            prediction = model(torch.from_numpy(features).to(torch.float64)).numpy()
    return {
        **regression(prediction, targets),
        "count": int(targets.size),
        "prediction_std": float(prediction.std()),
        "prediction_abs_p95": float(np.percentile(np.abs(prediction), 95.0)),
        "prediction_abs_max": float(np.max(np.abs(prediction))),
    }


def _parameter_diagnostics(model: ProfileFrictionResidual) -> dict[str, Any]:
    grid = torch.linspace(0.0, 0.7, 141, dtype=torch.float64)
    features = torch.zeros((grid.numel(), 5), dtype=torch.float64)
    features[:, 0] = grid
    features[:, 1] = 0.2
    values = model.components(features)
    return {
        "position_m": grid.tolist(),
        "delta_theta_deg": torch.rad2deg(values["delta"]).detach().tolist(),
        "coulomb_mps2": values["coulomb"].detach().tolist(),
        "viscous_sinv": values["viscous"].detach().tolist(),
        "blackbox_at_v0p2_mps2": values["blackbox"].detach().tolist(),
        "blackbox_coefficients_mps2": model.blackbox_coefficients.detach().tolist(),
        "kappa_at_v0p2": values["kappa"].detach().tolist(),
        "kappa_coefficients": model.kappa_coefficients.detach().tolist(),
        "coefficient_bounds": {
            "delta_abs_max_deg": float(
                torch.rad2deg(model.delta_coefficients.abs().max()).detach()
            ),
            "coulomb_max_mps2": float(model.coulomb_coefficients.max().detach()),
            "viscous_max_sinv": float(model.viscous_coefficients.max().detach()),
            "blackbox_abs_max_mps2": float(
                model.blackbox_coefficients.abs().max().detach()
            ),
            "kappa_abs_max": float(model.kappa_coefficients.abs().max().detach()),
        },
    }


def run(config: Config, result_path: Path = RESULT_PATH) -> dict[str, Any]:
    settings = ProbeSettings()
    manifest = load_manifest(config)
    train_trials = load_trials(config, "train")
    validation_trials = load_trials(config, "validation")
    constants = train_trials[0].constants
    for trial in train_trials + validation_trials:
        if (
            trial.constants.acceleration_gain != constants.acceleration_gain
            or trial.constants.gravity_mps2 != constants.gravity_mps2
        ):
            raise RuntimeError("The coarse probe assumes shared gamma and gravity.")
    train_store = WindowStore(train_trials, 33, config, manifest)
    validation_store = WindowStore(validation_trials, 33, config, manifest)
    validation_refs = list(validation_store.references[::5])
    if len(validation_refs) > settings.validation_windows:
        indices = np.linspace(
            0, len(validation_refs) - 1, settings.validation_windows, dtype=int
        )
        validation_refs = [validation_refs[index] for index in indices]
    candidates = {}
    training = {}
    for name, profile, friction, blackbox, kappa in (
        ("profile_only", True, False, False, False),
        ("friction_only", False, True, False, False),
        ("profile_plus_friction", True, True, False, False),
        ("profile_plus_blackbox", True, False, True, False),
    ):
        model = ProfileFrictionResidual(
            profile_enabled=profile,
            friction_enabled=friction,
            blackbox_enabled=blackbox,
            kappa_enabled=kappa,
            acceleration_gain=constants.acceleration_gain,
            gravity_mps2=constants.gravity_mps2,
            settings=settings,
        )
        model, training[name] = _fit(
            name,
            model,
            train_store,
            validation_store,
            validation_refs,
            config,
            settings,
        )
        candidates[name] = model.eval()
    rbf, _ = load_checkpoint(resolve_run_dir(config) / "best_checkpoint.pt")
    mlp, _ = load_portable_weights(config.paths.baseline_weights)
    models: dict[str, nn.Module | None] = {
        "nominal": None,
        "rbf_pv": rbf.eval(),
        "cap0p25_mlp": mlp.eval(),
        **candidates,
    }
    result: dict[str, Any] = {
        "schema_version": 1,
        "purpose": "coarse profile/friction/black-box structure probe; not deployable",
        "dataset_hash": manifest["dataset_hash"],
        "config_hash": config.config_hash,
        "settings": settings.__dict__,
        "training": training,
        "parameters": {
            name: _parameter_diagnostics(model) for name, model in candidates.items()
        },
        "splits": {},
    }
    for split in ("validation", "test", "stress"):
        trials = load_trials(config, split)
        store = WindowStore(trials, 60, config, manifest)
        refs = _refs_for_evaluation(store)
        split_result: dict[str, Any] = {"models": {}}
        for name, model in models.items():
            point = _point(model, trials)
            if model is None:
                # Any residual model returns the same nominal denominator metrics.
                reference_model = candidates["profile_only"]
                integral = _integral_metrics(
                    reference_model, store, refs, config
                )["overall"]
                integral = {
                    key: {
                        "rmse_mps2": value["nominal_rmse_mps2"],
                    }
                    for key, value in integral.items()
                }
                rollout_raw = _rollout_metrics(
                    reference_model, None, store, refs, config
                )["overall"]
                rollout = {
                    key: value["nominal"] for key, value in rollout_raw.items()
                }
            else:
                integral_raw = _integral_metrics(model, store, refs, config)["overall"]
                integral = {
                    key: {"rmse_mps2": value["rbf_rmse_mps2"]}
                    for key, value in integral_raw.items()
                }
                rollout_raw = _rollout_metrics(model, None, store, refs, config)["overall"]
                rollout = {key: value["rbf"] for key, value in rollout_raw.items()}
            split_result["models"][name] = {
                "point": point,
                "integral": integral,
                "rollout": rollout,
            }
        result["splits"][split] = split_result
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--result", type=Path, default=RESULT_PATH)
    args = parser.parse_args()
    result = run(load_config(args.config), args.result)
    concise = {}
    for split, values in result["splits"].items():
        concise[split] = {
            name: {
                "point_rmse": model["point"]["rmse"],
                "h12_integral_rmse": model["integral"]["h12"]["rmse_mps2"],
                "h33_pb_rmse": model["rollout"]["h33"]["pb_rmse_m"],
                "h33_vb_rmse": model["rollout"]["h33"]["vb_rmse_mps"],
            }
            for name, model in values["models"].items()
        }
    print(json.dumps(concise, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
