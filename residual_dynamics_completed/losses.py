"""Migrated Structured data losses and validation-only selection score."""
from __future__ import annotations
import math
from typing import Any, Sequence
import torch
import torch.nn.functional as F
from .dynamics import rollout_causal_dynamics

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
