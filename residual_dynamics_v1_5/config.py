"""Configuration schema for the independent v1.5 residual workflow."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
INPUT_NAMES = ("pb", "vb", "theta", "omega", "a_nom")


def _path(value: Any) -> Path:
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def _tuple_int(value: Any, name: str) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{name} must be a non-empty sequence")
    return tuple(int(item) for item in value)


def _tuple_float(value: Any, name: str) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{name} must be a non-empty sequence")
    return tuple(float(item) for item in value)


def _tuple_str(value: Any, name: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be a sequence")
    return tuple(str(item) for item in value)


@dataclass(frozen=True)
class PathsConfig:
    fit_source_dir: Path
    test_source_dir: Path
    processed_dir: Path
    training_runs_dir: Path
    wandb_dir: Path
    results_dir: Path
    report_path: Path
    v1_weights_path: Path


@dataclass(frozen=True)
class DataConfig:
    standard_groups: tuple[str, ...]
    extra_training_groups: tuple[str, ...]
    time_field: str
    phase_field: str
    phase_value: str
    ball_field: str
    theta_field: str
    drone_velocity_field: str
    gap_factor: float
    ball_window: int
    ball_polyorder: int
    theta_window: int
    theta_polyorder: int
    drone_velocity_window: int
    drone_velocity_polyorder: int
    edge_trim: int
    min_segment_samples: int
    jump_threshold_m: float
    jump_dilation_samples: int
    direction_radius_samples: int
    static_persistence_samples: int
    dynamic_speed_mps: float
    static_speed_mps: float
    static_acceleration_mps2: float
    max_abs_velocity_mps: float
    max_abs_acceleration_mps2: float
    max_abs_residual_mps2: float
    rollout_steps: tuple[int, ...]
    evaluation_steps: tuple[int, ...]
    validation_stride: int


@dataclass(frozen=True)
class ModelConfig:
    input_names: tuple[str, ...]
    hidden_dims: tuple[int, int]
    output_limit_mps2: float
    dtype: str
    deadzone_min_mps2: float
    deadzone_max_mps2: float
    deadzone_initial_mps2: float
    deadzone_smoothing_mps2: float
    velocity_switch_mps: float


@dataclass(frozen=True)
class LossConfig:
    position_scale_m: float
    velocity_scale_mps: float
    horizon_weights: tuple[float, ...]
    point_weight: float
    huber_beta_mps2: float
    jacobian_weight: float
    zero_velocity_anchor_weight: float
    static_weight: float
    stick_bound_weight: float


@dataclass(frozen=True)
class TrainingConfig:
    seed: int
    batch_size: int
    steps_per_epoch: int
    dynamic_max_epochs: int
    dynamic_learning_rate: float
    dynamic_weight_decay: float
    dynamic_lr_patience: int
    dynamic_lr_factor: float
    dynamic_early_stopping_patience: int
    deadzone_max_epochs: int
    deadzone_learning_rate: float
    deadzone_early_stopping_patience: int
    gradient_clip_norm: float
    validation_max_windows: int


@dataclass(frozen=True)
class RuntimeConfig:
    device: str
    gpu_min_speedup: float
    benchmark_warmup_steps: int
    benchmark_steps: int


@dataclass(frozen=True)
class WandbConfig:
    enabled: bool
    mode: str
    project: str
    entity: str | None
    group: str
    tags: tuple[str, ...]
    log_batch_interval: int
    log_histogram_interval_epochs: int
    init_timeout_s: int


@dataclass(frozen=True)
class AcceptanceConfig:
    dynamic_v1_degradation_max: float
    rollout_improvement_min: float
    per_dynamic_degradation_max: float
    static_median_max_mps2: float
    static_p95_max_mps2: float
    static_false_motion_rate_max: float
    saturation_rate_max: float
    deadzone_boundary_margin_fraction: float


@dataclass(frozen=True)
class AppConfig:
    paths: PathsConfig
    data: DataConfig
    model: ModelConfig
    loss: LossConfig
    training: TrainingConfig
    runtime: RuntimeConfig
    wandb: WandbConfig
    acceptance: AcceptanceConfig
    raw: Mapping[str, Any]
    config_path: Path
    config_hash: str


def load_config(path: str | Path) -> AppConfig:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    if not isinstance(raw, dict):
        raise ValueError("Configuration root must be a mapping")
    digest_payload = json.dumps(raw, sort_keys=True, separators=(",", ":"), default=str)
    config_hash = hashlib.sha256(digest_payload.encode()).hexdigest()

    p = raw["paths"]
    paths = PathsConfig(**{name: _path(p[name]) for name in PathsConfig.__dataclass_fields__})
    d = raw["data"]
    data = DataConfig(
        standard_groups=_tuple_str(d["standard_groups"], "data.standard_groups"),
        extra_training_groups=_tuple_str(d["extra_training_groups"], "data.extra_training_groups"),
        time_field=str(d["time_field"]), phase_field=str(d["phase_field"]),
        phase_value=str(d["phase_value"]), ball_field=str(d["ball_field"]),
        theta_field=str(d["theta_field"]), drone_velocity_field=str(d["drone_velocity_field"]),
        gap_factor=float(d["gap_factor"]), ball_window=int(d["ball_window"]),
        ball_polyorder=int(d["ball_polyorder"]), theta_window=int(d["theta_window"]),
        theta_polyorder=int(d["theta_polyorder"]),
        drone_velocity_window=int(d["drone_velocity_window"]),
        drone_velocity_polyorder=int(d["drone_velocity_polyorder"]),
        edge_trim=int(d["edge_trim"]), min_segment_samples=int(d["min_segment_samples"]),
        jump_threshold_m=float(d["jump_threshold_m"]),
        jump_dilation_samples=int(d["jump_dilation_samples"]),
        direction_radius_samples=int(d["direction_radius_samples"]),
        static_persistence_samples=int(d["static_persistence_samples"]),
        dynamic_speed_mps=float(d["dynamic_speed_mps"]),
        static_speed_mps=float(d["static_speed_mps"]),
        static_acceleration_mps2=float(d["static_acceleration_mps2"]),
        max_abs_velocity_mps=float(d["max_abs_velocity_mps"]),
        max_abs_acceleration_mps2=float(d["max_abs_acceleration_mps2"]),
        max_abs_residual_mps2=float(d["max_abs_residual_mps2"]),
        rollout_steps=_tuple_int(d["rollout_steps"], "data.rollout_steps"),
        evaluation_steps=_tuple_int(d["evaluation_steps"], "data.evaluation_steps"),
        validation_stride=int(d["validation_stride"]),
    )
    m = raw["model"]
    model = ModelConfig(
        input_names=_tuple_str(m["input_names"], "model.input_names"),
        hidden_dims=tuple(_tuple_int(m["hidden_dims"], "model.hidden_dims")),
        output_limit_mps2=float(m["output_limit_mps2"]), dtype=str(m["dtype"]),
        deadzone_min_mps2=float(m["deadzone_min_mps2"]),
        deadzone_max_mps2=float(m["deadzone_max_mps2"]),
        deadzone_initial_mps2=float(m["deadzone_initial_mps2"]),
        deadzone_smoothing_mps2=float(m["deadzone_smoothing_mps2"]),
        velocity_switch_mps=float(m["velocity_switch_mps"]),
    )
    l = raw["loss"]
    loss = LossConfig(
        position_scale_m=float(l["position_scale_m"]),
        velocity_scale_mps=float(l["velocity_scale_mps"]),
        horizon_weights=_tuple_float(l["horizon_weights"], "loss.horizon_weights"),
        point_weight=float(l["point_weight"]), huber_beta_mps2=float(l["huber_beta_mps2"]),
        jacobian_weight=float(l["jacobian_weight"]),
        zero_velocity_anchor_weight=float(l["zero_velocity_anchor_weight"]),
        static_weight=float(l["static_weight"]), stick_bound_weight=float(l["stick_bound_weight"]),
    )
    t = raw["training"]
    training = TrainingConfig(**{name: (int(t[name]) if name not in {
        "dynamic_learning_rate", "dynamic_weight_decay", "dynamic_lr_factor",
        "deadzone_learning_rate", "gradient_clip_norm"} else float(t[name]))
        for name in TrainingConfig.__dataclass_fields__})
    r = raw["runtime"]
    runtime = RuntimeConfig(device=str(r["device"]), gpu_min_speedup=float(r["gpu_min_speedup"]),
                            benchmark_warmup_steps=int(r["benchmark_warmup_steps"]),
                            benchmark_steps=int(r["benchmark_steps"]))
    w = raw["wandb"]
    wandb = WandbConfig(
        enabled=bool(w["enabled"]), mode=str(w["mode"]), project=str(w["project"]),
        entity=None if w.get("entity") in (None, "") else str(w["entity"]), group=str(w["group"]),
        tags=_tuple_str(w["tags"], "wandb.tags"), log_batch_interval=int(w["log_batch_interval"]),
        log_histogram_interval_epochs=int(w["log_histogram_interval_epochs"]),
        init_timeout_s=int(w["init_timeout_s"]),
    )
    a = raw["acceptance"]
    acceptance = AcceptanceConfig(**{name: float(a[name]) for name in AcceptanceConfig.__dataclass_fields__})
    _validate(data, model, loss, training, runtime, wandb)
    return AppConfig(paths, data, model, loss, training, runtime, wandb, acceptance,
                     raw, config_path, config_hash)


def _validate(data: DataConfig, model: ModelConfig, loss: LossConfig,
              training: TrainingConfig, runtime: RuntimeConfig, wandb: WandbConfig) -> None:
    for name, window, order in (("ball", data.ball_window, data.ball_polyorder),
                                ("theta", data.theta_window, data.theta_polyorder),
                                ("drone", data.drone_velocity_window, data.drone_velocity_polyorder)):
        if window < 3 or window % 2 == 0 or order >= window:
            raise ValueError(f"Invalid {name} Savitzky-Golay settings")
    if data.edge_trim < data.ball_window // 2 or data.gap_factor <= 1:
        raise ValueError("Invalid filtering boundary configuration")
    if model.input_names != INPUT_NAMES or len(model.hidden_dims) != 2:
        raise ValueError(f"v1.5 requires input order {INPUT_NAMES} and two hidden layers")
    if model.dtype != "float64" or not (model.deadzone_min_mps2 < model.deadzone_initial_mps2 < model.deadzone_max_mps2):
        raise ValueError("Invalid model dtype or deadzone bounds")
    if abs(sum(loss.horizon_weights) - 1.0) > 1e-9 or len(loss.horizon_weights) != len(data.rollout_steps):
        raise ValueError("Horizon weights must match rollout steps and sum to one")
    if training.seed != 7 or training.batch_size <= 0:
        raise ValueError("v1.5 fixes seed=7 and requires a positive batch size")
    if runtime.device not in {"auto", "cpu", "cuda"} or runtime.gpu_min_speedup < 1:
        raise ValueError("Invalid runtime configuration")
    if wandb.mode not in {"auto", "online", "offline", "disabled"}:
        raise ValueError("Invalid W&B mode")
