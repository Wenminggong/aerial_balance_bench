"""Configuration for the isolated two-input RBF residual workflow."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from residual_dynamics.config import AppConfig as LegacyConfig
from residual_dynamics.config import REPO_ROOT, load_config as load_legacy_config


@dataclass(frozen=True)
class RbfConfig:
    pb_centers_m: tuple[float, ...]
    vb_centers_mps: tuple[float, ...]
    pb_width_m: float
    vb_width_mps: float
    denominator_epsilon: float
    cap_quantile: float
    cap_step_mps2: float
    cap_min_mps2: float
    cap_max_mps2: float
    ood_decay_width_std: float
    ood_min_weight: float


@dataclass(frozen=True)
class SamplingConfig:
    empirical_fraction: float
    joint_fraction: float
    actuation_fraction: float
    velocity_bin_edges_mps: tuple[float, ...]
    actuation_threshold_mps2: float
    max_abs_velocity_mps: float
    min_windows_per_position_bin: int
    min_windows_per_speed_macro: int


@dataclass(frozen=True)
class IdentificationLossConfig:
    horizons: tuple[int, ...]
    horizon_weights: tuple[float, ...]
    integral_weight: float
    point_weight: float
    curvature_weight: float
    velocity_derivative_weight: float
    position_derivative_weight: float
    huber_beta_mps2: float
    velocity_derivative_soft_min_sinv: float
    velocity_derivative_soft_max_sinv: float
    position_derivative_soft_abs_max_sinv2: float
    collocation_fraction: float


@dataclass(frozen=True)
class RolloutLossConfig:
    horizons: tuple[int, ...]
    horizon_weights: tuple[float, ...]
    rollout_weight: float
    causal_auxiliary_weight: float
    position_scale_m: float
    velocity_scale_mps: float
    huber_beta_normalized: float
    trust_weight: float
    trust_output_threshold_mps2: float
    coefficient_radius_mps2: float


@dataclass(frozen=True)
class StageConfig:
    batch_size: int
    steps_per_epoch: int
    max_epochs: int
    learning_rate: float
    early_stopping_patience: int


@dataclass(frozen=True)
class TrainingConfig:
    seed: int
    gradient_clip_norm: float
    lr_patience: int
    lr_factor: float
    validation_stride: int
    validation_max_windows: int
    stage1: StageConfig
    stage2: StageConfig


@dataclass(frozen=True)
class AcceptanceConfig:
    integral_improvement_min: float
    per_dynamic_degradation_max: float
    saturation_rate_max: float
    static_total_acc_max_mps2: float
    rv_p01_min_sinv: float
    rv_p99_max_sinv: float
    rv_min_sinv: float
    rv_max_sinv: float
    rp_p99_abs_max_sinv2: float
    rp_abs_max_sinv2: float
    direct_channel_tolerance: float
    sensitivity_cosine_p01_min: float
    sensitivity_ratio_p01_min: float
    sensitivity_ratio_p99_max: float
    stress_rollout_degradation_max: float


@dataclass(frozen=True)
class PathsConfig:
    processed_dir: Path
    training_runs_dir: Path
    results_dir: Path
    report_path: Path
    baseline_weights: Path | None


@dataclass(frozen=True)
class Config:
    path: Path
    raw: dict[str, Any]
    config_hash: str
    legacy: LegacyConfig
    paths: PathsConfig
    rbf: RbfConfig
    sampling: SamplingConfig
    identification_loss: IdentificationLossConfig
    rollout_loss: RolloutLossConfig
    training: TrainingConfig
    acceptance: AcceptanceConfig


def _path(value: Any) -> Path:
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def _stage(value: dict[str, Any]) -> StageConfig:
    return StageConfig(
        batch_size=int(value["batch_size"]),
        steps_per_epoch=int(value["steps_per_epoch"]),
        max_epochs=int(value["max_epochs"]),
        learning_rate=float(value["learning_rate"]),
        early_stopping_patience=int(value["early_stopping_patience"]),
    )


def load_config(path: str | Path) -> Config:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    if not isinstance(raw, dict):
        raise ValueError("Configuration root must be a mapping.")
    digest = hashlib.sha256(
        json.dumps(raw, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    legacy = load_legacy_config(config_path)
    paths_raw = raw["paths"]
    baseline = raw.get("baseline", {}).get("weights")
    paths = PathsConfig(
        processed_dir=_path(paths_raw["processed_dir"]),
        training_runs_dir=_path(paths_raw["training_runs_dir"]),
        results_dir=_path(paths_raw["results_dir"]),
        report_path=_path(paths_raw["report_path"]),
        baseline_weights=None if not baseline else _path(baseline),
    )
    rbf_raw = raw["rbf"]
    rbf = RbfConfig(
        pb_centers_m=tuple(float(x) for x in rbf_raw["pb_centers_m"]),
        vb_centers_mps=tuple(float(x) for x in rbf_raw["vb_centers_mps"]),
        pb_width_m=float(rbf_raw["pb_width_m"]),
        vb_width_mps=float(rbf_raw["vb_width_mps"]),
        denominator_epsilon=float(rbf_raw["denominator_epsilon"]),
        cap_quantile=float(rbf_raw["cap_quantile"]),
        cap_step_mps2=float(rbf_raw["cap_step_mps2"]),
        cap_min_mps2=float(rbf_raw["cap_min_mps2"]),
        cap_max_mps2=float(rbf_raw["cap_max_mps2"]),
        ood_decay_width_std=float(rbf_raw["ood_decay_width_std"]),
        ood_min_weight=float(rbf_raw["ood_min_weight"]),
    )
    sampling_raw = raw["sampling"]
    sampling = SamplingConfig(
        empirical_fraction=float(sampling_raw["empirical_fraction"]),
        joint_fraction=float(sampling_raw["joint_fraction"]),
        actuation_fraction=float(sampling_raw["actuation_fraction"]),
        velocity_bin_edges_mps=tuple(
            float(x) for x in sampling_raw["velocity_bin_edges_mps"]
        ),
        actuation_threshold_mps2=float(sampling_raw["actuation_threshold_mps2"]),
        max_abs_velocity_mps=float(sampling_raw["max_abs_velocity_mps"]),
        min_windows_per_position_bin=int(
            sampling_raw["min_windows_per_position_bin"]
        ),
        min_windows_per_speed_macro=int(
            sampling_raw["min_windows_per_speed_macro"]
        ),
    )
    ident_raw = raw["identification_loss"]
    identification = IdentificationLossConfig(
        horizons=tuple(int(x) for x in ident_raw["horizons"]),
        horizon_weights=tuple(float(x) for x in ident_raw["horizon_weights"]),
        integral_weight=float(ident_raw["integral_weight"]),
        point_weight=float(ident_raw["point_weight"]),
        curvature_weight=float(ident_raw["curvature_weight"]),
        velocity_derivative_weight=float(ident_raw["velocity_derivative_weight"]),
        position_derivative_weight=float(ident_raw["position_derivative_weight"]),
        huber_beta_mps2=float(ident_raw["huber_beta_mps2"]),
        velocity_derivative_soft_min_sinv=float(
            ident_raw["velocity_derivative_soft_min_sinv"]
        ),
        velocity_derivative_soft_max_sinv=float(
            ident_raw["velocity_derivative_soft_max_sinv"]
        ),
        position_derivative_soft_abs_max_sinv2=float(
            ident_raw["position_derivative_soft_abs_max_sinv2"]
        ),
        collocation_fraction=float(ident_raw["collocation_fraction"]),
    )
    rollout_raw = raw["rollout_loss"]
    rollout = RolloutLossConfig(
        horizons=tuple(int(x) for x in rollout_raw["horizons"]),
        horizon_weights=tuple(float(x) for x in rollout_raw["horizon_weights"]),
        rollout_weight=float(rollout_raw["rollout_weight"]),
        causal_auxiliary_weight=float(rollout_raw["causal_auxiliary_weight"]),
        position_scale_m=float(rollout_raw["position_scale_m"]),
        velocity_scale_mps=float(rollout_raw["velocity_scale_mps"]),
        huber_beta_normalized=float(rollout_raw["huber_beta_normalized"]),
        trust_weight=float(rollout_raw["trust_weight"]),
        trust_output_threshold_mps2=float(
            rollout_raw["trust_output_threshold_mps2"]
        ),
        coefficient_radius_mps2=float(rollout_raw["coefficient_radius_mps2"]),
    )
    training_raw = raw["training_v2"]
    training = TrainingConfig(
        seed=int(training_raw["seed"]),
        gradient_clip_norm=float(training_raw["gradient_clip_norm"]),
        lr_patience=int(training_raw["lr_patience"]),
        lr_factor=float(training_raw["lr_factor"]),
        validation_stride=int(training_raw["validation_stride"]),
        validation_max_windows=int(training_raw["validation_max_windows"]),
        stage1=_stage(training_raw["stage1"]),
        stage2=_stage(training_raw["stage2"]),
    )
    accept_raw = raw["acceptance_v2"]
    acceptance = AcceptanceConfig(**{key: float(value) for key, value in accept_raw.items()})
    config = Config(
        path=config_path,
        raw=raw,
        config_hash=digest,
        legacy=legacy,
        paths=paths,
        rbf=rbf,
        sampling=sampling,
        identification_loss=identification,
        rollout_loss=rollout,
        training=training,
        acceptance=acceptance,
    )
    _validate(config)
    return config


def _validate(config: Config) -> None:
    if len(config.rbf.pb_centers_m) != 9 or len(config.rbf.vb_centers_mps) != 9:
        raise ValueError("The approved RBF design requires a 9x9 center grid.")
    fractions = (
        config.sampling.empirical_fraction
        + config.sampling.joint_fraction
        + config.sampling.actuation_fraction
    )
    if abs(fractions - 1.0) > 1.0e-12:
        raise ValueError("Sampling fractions must sum to one.")
    if len(config.identification_loss.horizons) != len(
        config.identification_loss.horizon_weights
    ):
        raise ValueError("Identification horizons and weights disagree.")
    if len(config.rollout_loss.horizons) != len(config.rollout_loss.horizon_weights):
        raise ValueError("Rollout horizons and weights disagree.")
    if config.rbf.cap_min_mps2 > config.rbf.cap_max_mps2:
        raise ValueError("Invalid RBF cap interval.")
