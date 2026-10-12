"""Configuration loading and validation for residual-dynamics identification."""

from __future__ import annotations

import hashlib
import json
import copy
from dataclasses import dataclass
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
INPUT_NAMES = ("pb", "vb", "theta", "omega", "a_nom")


def _tuple_int(value: Any, name: str) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{name} must be a non-empty sequence.")
    return tuple(int(item) for item in value)


def _tuple_str(value: Any, name: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be a sequence.")
    return tuple(str(item) for item in value)


def _path(value: Any) -> Path:
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


@dataclass(frozen=True)
class PathsConfig:
    source_dir: Path | None
    source_dirs: tuple[Path, ...]
    processed_dir: Path
    training_runs_dir: Path
    wandb_dir: Path
    results_dir: Path
    report_path: Path

    @property
    def all_source_dirs(self) -> tuple[Path, ...]:
        if self.source_dirs:
            return self.source_dirs
        if self.source_dir is not None:
            return (self.source_dir,)
        return ()


@dataclass(frozen=True)
class SplitAssignment:
    session: str
    group: str
    trial_index: int
    split: str


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
    rollout_steps: tuple[int, ...]
    evaluation_steps: tuple[int, ...]
    validation_stride: int
    speed_threshold_mps: float
    split_assignments: tuple[SplitAssignment, ...]
    default_split: str
    include_failure_prefixes: bool
    include_failure_prefixes_in_train: bool
    failure_sampling_weight: float
    position_bin_edges_m: tuple[float, ...]
    empirical_sampling_fraction: float
    min_windows_per_position_bin: int
    ball_position_min_m: float
    ball_position_max_m: float


@dataclass(frozen=True)
class OodConfig:
    enabled: bool
    lower_quantile: float
    upper_quantile: float
    shrinkage: float
    mahalanobis_quantile: float


@dataclass(frozen=True)
class ControlChannelConfig:
    enabled: bool
    theta_weight: float
    anti_damping_weight: float
    theta_ratio_soft_min: float
    theta_ratio_soft_max: float
    theta_ratio_hard_min: float
    theta_ratio_hard_max: float
    nominal_theta_gain_floor: float
    anti_damping_soft_max_sinv: float
    anti_damping_hard_p99_sinv: float
    anti_damping_hard_max_sinv: float
    collocation_fraction: float
    collocation_vb_min_mps: float
    collocation_vb_max_mps: float
    collocation_theta_min_rad: float
    collocation_theta_max_rad: float
    collocation_vrz_min_mps: float
    collocation_vrz_max_mps: float
    outer_safety_points: int
    outer_theta_min_rad: float
    outer_theta_max_rad: float
    sensitivity_horizons: tuple[int, ...]
    sensitivity_max_windows: int
    sensitivity_cosine_p01_min: float
    sensitivity_ratio_p01_min: float
    sensitivity_ratio_p99_max: float


@dataclass(frozen=True)
class CausalRolloutConfig:
    enabled: bool
    ball_velocity_field: str
    command_field: str
    pending_command_fields: tuple[str, ...]
    delay_steps_parameter: str
    response_enabled_parameter: str
    response_tau_parameter: str
    response_gain_parameter: str
    response_bias_parameter: str
    max_velocity_parameter: str
    smooth_rollout_weight: float
    jacobian_weight: float
    jacobian_norm_limit_mps2: float
    jacobian_p99_factor: float
    jacobian_max_factor: float
    state_scale_floor: float


@dataclass(frozen=True)
class ModelConfig:
    input_names: tuple[str, ...]
    hidden_dims: tuple[int, ...]
    output_limit_mps2: float
    dtype: str


@dataclass(frozen=True)
class LossConfig:
    position_scale_m: float
    velocity_scale_mps: float
    horizon_weights: tuple[float, ...]
    point_weight: float
    huber_beta_mps2: float
    jacobian_weight: float


@dataclass(frozen=True)
class TrainingConfig:
    seed: int
    batch_size: int
    steps_per_epoch: int
    max_epochs: int
    learning_rate: float
    weight_decay: float
    lr_patience: int
    lr_factor: float
    early_stopping_patience: int
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
    residual_rmse_max_mps2: float
    rollout_improvement_min: float
    per_dynamic_degradation_max: float
    static_total_acc_median_max_mps2: float
    saturation_rate_max: float


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
    causal_rollout: CausalRolloutConfig
    ood: OodConfig
    control_channel: ControlChannelConfig
    raw: Mapping[str, Any]
    config_path: Path
    config_hash: str


def load_config(path: str | Path) -> AppConfig:
    """Load a YAML configuration and resolve all paths against the repository root."""

    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    if not isinstance(raw, dict):
        raise ValueError("Configuration root must be a mapping.")

    digest_payload = json.dumps(raw, sort_keys=True, separators=(",", ":"), default=str)
    config_hash = hashlib.sha256(digest_payload.encode("utf-8")).hexdigest()

    paths_raw = raw["paths"]
    has_source_dir = "source_dir" in paths_raw
    has_source_dirs = "source_dirs" in paths_raw
    if has_source_dir == has_source_dirs:
        raise ValueError("paths must define exactly one of source_dir or source_dirs.")
    paths = PathsConfig(
        source_dir=_path(paths_raw["source_dir"]) if has_source_dir else None,
        source_dirs=(
            tuple(_path(item) for item in paths_raw["source_dirs"])
            if has_source_dirs
            else ()
        ),
        processed_dir=_path(paths_raw["processed_dir"]),
        training_runs_dir=_path(paths_raw["training_runs_dir"]),
        wandb_dir=_path(paths_raw["wandb_dir"]),
        results_dir=_path(paths_raw["results_dir"]),
        report_path=_path(paths_raw["report_path"]),
    )

    data_raw = raw["data"]
    split_assignments: list[SplitAssignment] = []
    for split, selectors in data_raw.get("split_assignments", {}).items():
        for selector in selectors:
            split_assignments.append(
                SplitAssignment(
                    session=str(selector["session"]),
                    group=str(selector["group"]),
                    trial_index=int(selector["trial_index"]),
                    split=str(split),
                )
            )
    data = DataConfig(
        standard_groups=_tuple_str(data_raw["standard_groups"], "data.standard_groups"),
        extra_training_groups=_tuple_str(
            data_raw["extra_training_groups"], "data.extra_training_groups"
        ),
        time_field=str(data_raw["time_field"]),
        phase_field=str(data_raw["phase_field"]),
        phase_value=str(data_raw["phase_value"]),
        ball_field=str(data_raw["ball_field"]),
        theta_field=str(data_raw["theta_field"]),
        drone_velocity_field=str(data_raw["drone_velocity_field"]),
        gap_factor=float(data_raw["gap_factor"]),
        ball_window=int(data_raw["ball_window"]),
        ball_polyorder=int(data_raw["ball_polyorder"]),
        theta_window=int(data_raw["theta_window"]),
        theta_polyorder=int(data_raw["theta_polyorder"]),
        drone_velocity_window=int(data_raw["drone_velocity_window"]),
        drone_velocity_polyorder=int(data_raw["drone_velocity_polyorder"]),
        edge_trim=int(data_raw["edge_trim"]),
        min_segment_samples=int(data_raw["min_segment_samples"]),
        rollout_steps=_tuple_int(data_raw["rollout_steps"], "data.rollout_steps"),
        evaluation_steps=_tuple_int(
            data_raw["evaluation_steps"], "data.evaluation_steps"
        ),
        validation_stride=int(data_raw["validation_stride"]),
        speed_threshold_mps=float(data_raw["speed_threshold_mps"]),
        split_assignments=tuple(split_assignments),
        default_split=str(data_raw.get("default_split", "train")),
        include_failure_prefixes=bool(data_raw.get("include_failure_prefixes", False)),
        include_failure_prefixes_in_train=bool(
            data_raw.get(
                "include_failure_prefixes_in_train",
                data_raw.get("include_failure_prefixes", False),
            )
        ),
        failure_sampling_weight=float(data_raw.get("failure_sampling_weight", 1.0)),
        position_bin_edges_m=tuple(
            float(item)
            for item in data_raw.get(
                "position_bin_edges_m", (0.0, 0.2, 0.3, 0.4, 0.5, 0.7)
            )
        ),
        empirical_sampling_fraction=float(
            data_raw.get("empirical_sampling_fraction", 1.0)
        ),
        min_windows_per_position_bin=int(
            data_raw.get("min_windows_per_position_bin", 0)
        ),
        ball_position_min_m=float(data_raw.get("ball_position_min_m", 0.0)),
        ball_position_max_m=float(data_raw.get("ball_position_max_m", 0.7)),
    )

    ood_raw = raw.get("ood", {})
    ood = OodConfig(
        enabled=bool(ood_raw.get("enabled", False)),
        lower_quantile=float(ood_raw.get("lower_quantile", 0.005)),
        upper_quantile=float(ood_raw.get("upper_quantile", 0.995)),
        shrinkage=float(ood_raw.get("shrinkage", 0.05)),
        mahalanobis_quantile=float(ood_raw.get("mahalanobis_quantile", 0.995)),
    )

    channel_raw = raw.get("control_channel", {})
    control_channel = ControlChannelConfig(
        enabled=bool(channel_raw.get("enabled", False)),
        theta_weight=float(channel_raw.get("theta_weight", 0.0)),
        anti_damping_weight=float(channel_raw.get("anti_damping_weight", 0.0)),
        theta_ratio_soft_min=float(channel_raw.get("theta_ratio_soft_min", 0.8)),
        theta_ratio_soft_max=float(channel_raw.get("theta_ratio_soft_max", 1.2)),
        theta_ratio_hard_min=float(channel_raw.get("theta_ratio_hard_min", 0.7)),
        theta_ratio_hard_max=float(channel_raw.get("theta_ratio_hard_max", 1.3)),
        nominal_theta_gain_floor=float(channel_raw.get("nominal_theta_gain_floor", 1.0)),
        anti_damping_soft_max_sinv=float(
            channel_raw.get("anti_damping_soft_max_sinv", 0.2)
        ),
        anti_damping_hard_p99_sinv=float(
            channel_raw.get("anti_damping_hard_p99_sinv", 0.2)
        ),
        anti_damping_hard_max_sinv=float(
            channel_raw.get("anti_damping_hard_max_sinv", 0.3)
        ),
        collocation_fraction=float(channel_raw.get("collocation_fraction", 0.25)),
        collocation_vb_min_mps=float(channel_raw.get("collocation_vb_min_mps", -0.5)),
        collocation_vb_max_mps=float(channel_raw.get("collocation_vb_max_mps", 0.5)),
        collocation_theta_min_rad=float(
            channel_raw.get("collocation_theta_min_rad", -0.25)
        ),
        collocation_theta_max_rad=float(
            channel_raw.get("collocation_theta_max_rad", 0.25)
        ),
        collocation_vrz_min_mps=float(
            channel_raw.get("collocation_vrz_min_mps", -0.8)
        ),
        collocation_vrz_max_mps=float(
            channel_raw.get("collocation_vrz_max_mps", 0.8)
        ),
        outer_safety_points=int(channel_raw.get("outer_safety_points", 4096)),
        outer_theta_min_rad=float(
            channel_raw.get("outer_theta_min_rad", -0.6981317)
        ),
        outer_theta_max_rad=float(
            channel_raw.get("outer_theta_max_rad", 0.6981317)
        ),
        sensitivity_horizons=_tuple_int(
            channel_raw.get("sensitivity_horizons", (6, 12, 33)),
            "control_channel.sensitivity_horizons",
        ),
        sensitivity_max_windows=int(
            channel_raw.get("sensitivity_max_windows", 256)
        ),
        sensitivity_cosine_p01_min=float(
            channel_raw.get("sensitivity_cosine_p01_min", 0.95)
        ),
        sensitivity_ratio_p01_min=float(
            channel_raw.get("sensitivity_ratio_p01_min", 0.7)
        ),
        sensitivity_ratio_p99_max=float(
            channel_raw.get("sensitivity_ratio_p99_max", 1.3)
        ),
    )

    causal_raw = raw.get("causal_rollout", {})
    causal_rollout = CausalRolloutConfig(
        enabled=bool(causal_raw.get("enabled", False)),
        ball_velocity_field=str(causal_raw.get("ball_velocity_field", "obs_vb")),
        command_field=str(causal_raw.get("command_field", "nmpc_command_z")),
        pending_command_fields=_tuple_str(
            causal_raw.get(
                "pending_command_fields",
                ("pending_command_0", "pending_command_1", "pending_command_2"),
            ),
            "causal_rollout.pending_command_fields",
        ),
        delay_steps_parameter=str(
            causal_raw.get("delay_steps_parameter", "predictor.delay_steps")
        ),
        response_enabled_parameter=str(
            causal_raw.get(
                "response_enabled_parameter", "controller.response.enabled"
            )
        ),
        response_tau_parameter=str(
            causal_raw.get("response_tau_parameter", "controller.response.tau_s")
        ),
        response_gain_parameter=str(
            causal_raw.get("response_gain_parameter", "controller.response.gain")
        ),
        response_bias_parameter=str(
            causal_raw.get("response_bias_parameter", "controller.response.bias")
        ),
        max_velocity_parameter=str(
            causal_raw.get(
                "max_velocity_parameter", "controller.constraints.max_velocity"
            )
        ),
        smooth_rollout_weight=float(causal_raw.get("smooth_rollout_weight", 0.25)),
        jacobian_weight=float(causal_raw.get("jacobian_weight", 0.1)),
        jacobian_norm_limit_mps2=float(
            causal_raw.get("jacobian_norm_limit_mps2", 0.02)
        ),
        jacobian_p99_factor=float(causal_raw.get("jacobian_p99_factor", 1.25)),
        jacobian_max_factor=float(causal_raw.get("jacobian_max_factor", 2.0)),
        state_scale_floor=float(causal_raw.get("state_scale_floor", 1.0e-6)),
    )

    model_raw = raw["model"]
    model = ModelConfig(
        input_names=_tuple_str(model_raw["input_names"], "model.input_names"),
        hidden_dims=_tuple_int(model_raw["hidden_dims"], "model.hidden_dims"),
        output_limit_mps2=float(model_raw["output_limit_mps2"]),
        dtype=str(model_raw["dtype"]),
    )

    loss_raw = raw["loss"]
    loss = LossConfig(
        position_scale_m=float(loss_raw["position_scale_m"]),
        velocity_scale_mps=float(loss_raw["velocity_scale_mps"]),
        horizon_weights=tuple(float(item) for item in loss_raw["horizon_weights"]),
        point_weight=float(loss_raw["point_weight"]),
        huber_beta_mps2=float(loss_raw["huber_beta_mps2"]),
        jacobian_weight=float(loss_raw["jacobian_weight"]),
    )

    training_raw = raw["training"]
    training = TrainingConfig(
        seed=int(training_raw["seed"]),
        batch_size=int(training_raw["batch_size"]),
        steps_per_epoch=int(training_raw["steps_per_epoch"]),
        max_epochs=int(training_raw["max_epochs"]),
        learning_rate=float(training_raw["learning_rate"]),
        weight_decay=float(training_raw["weight_decay"]),
        lr_patience=int(training_raw["lr_patience"]),
        lr_factor=float(training_raw["lr_factor"]),
        early_stopping_patience=int(training_raw["early_stopping_patience"]),
        gradient_clip_norm=float(training_raw["gradient_clip_norm"]),
        validation_max_windows=int(training_raw["validation_max_windows"]),
    )

    runtime_raw = raw["runtime"]
    runtime = RuntimeConfig(
        device=str(runtime_raw["device"]),
        gpu_min_speedup=float(runtime_raw["gpu_min_speedup"]),
        benchmark_warmup_steps=int(runtime_raw["benchmark_warmup_steps"]),
        benchmark_steps=int(runtime_raw["benchmark_steps"]),
    )

    wandb_raw = raw["wandb"]
    wandb = WandbConfig(
        enabled=bool(wandb_raw["enabled"]),
        mode=str(wandb_raw["mode"]),
        project=str(wandb_raw["project"]),
        entity=None if wandb_raw.get("entity") in (None, "") else str(wandb_raw["entity"]),
        group=str(wandb_raw["group"]),
        tags=_tuple_str(wandb_raw["tags"], "wandb.tags"),
        log_batch_interval=int(wandb_raw["log_batch_interval"]),
        log_histogram_interval_epochs=int(wandb_raw["log_histogram_interval_epochs"]),
        init_timeout_s=int(wandb_raw["init_timeout_s"]),
    )

    acceptance_raw = raw["acceptance"]
    acceptance = AcceptanceConfig(
        residual_rmse_max_mps2=float(acceptance_raw["residual_rmse_max_mps2"]),
        rollout_improvement_min=float(acceptance_raw["rollout_improvement_min"]),
        per_dynamic_degradation_max=float(
            acceptance_raw["per_dynamic_degradation_max"]
        ),
        static_total_acc_median_max_mps2=float(
            acceptance_raw["static_total_acc_median_max_mps2"]
        ),
        saturation_rate_max=float(acceptance_raw["saturation_rate_max"]),
    )

    _validate(
        paths, data, model, loss, training, runtime, wandb, causal_rollout,
        ood, control_channel,
    )
    return AppConfig(
        paths=paths,
        data=data,
        model=model,
        loss=loss,
        training=training,
        runtime=runtime,
        wandb=wandb,
        acceptance=acceptance,
        causal_rollout=causal_rollout,
        ood=ood,
        control_channel=control_channel,
        raw=raw,
        config_path=config_path,
        config_hash=config_hash,
    )


def with_training_overrides(
    config: AppConfig,
    *,
    seed: int | None = None,
    jacobian_weight: float | None = None,
    jacobian_norm_limit_mps2: float | None = None,
    steps_per_epoch: int | None = None,
    max_epochs: int | None = None,
    early_stopping_patience: int | None = None,
    validation_max_windows: int | None = None,
    theta_channel_weight: float | None = None,
) -> AppConfig:
    """Return a reproducibly hashed training variant without changing data paths."""

    raw = copy.deepcopy(dict(config.raw))
    if seed is not None:
        raw["training"]["seed"] = int(seed)
    for name, value in (
        ("steps_per_epoch", steps_per_epoch),
        ("max_epochs", max_epochs),
        ("early_stopping_patience", early_stopping_patience),
        ("validation_max_windows", validation_max_windows),
    ):
        if value is not None:
            raw["training"][name] = int(value)
    if jacobian_weight is not None:
        raw.setdefault("causal_rollout", {})["jacobian_weight"] = float(
            jacobian_weight
        )
    if jacobian_norm_limit_mps2 is not None:
        raw.setdefault("causal_rollout", {})["jacobian_norm_limit_mps2"] = float(
            jacobian_norm_limit_mps2
        )
    if theta_channel_weight is not None:
        raw.setdefault("control_channel", {})["theta_weight"] = float(
            theta_channel_weight
        )
    digest_payload = json.dumps(raw, sort_keys=True, separators=(",", ":"), default=str)
    config_hash = hashlib.sha256(digest_payload.encode("utf-8")).hexdigest()
    result = replace(
        config,
        training=replace(
            config.training,
            seed=config.training.seed if seed is None else int(seed),
            steps_per_epoch=(
                config.training.steps_per_epoch
                if steps_per_epoch is None
                else int(steps_per_epoch)
            ),
            max_epochs=(
                config.training.max_epochs if max_epochs is None else int(max_epochs)
            ),
            early_stopping_patience=(
                config.training.early_stopping_patience
                if early_stopping_patience is None
                else int(early_stopping_patience)
            ),
            validation_max_windows=(
                config.training.validation_max_windows
                if validation_max_windows is None
                else int(validation_max_windows)
            ),
        ),
        causal_rollout=replace(
            config.causal_rollout,
            jacobian_weight=(
                config.causal_rollout.jacobian_weight
                if jacobian_weight is None
                else float(jacobian_weight)
            ),
            jacobian_norm_limit_mps2=(
                config.causal_rollout.jacobian_norm_limit_mps2
                if jacobian_norm_limit_mps2 is None
                else float(jacobian_norm_limit_mps2)
            ),
        ),
        control_channel=replace(
            config.control_channel,
            theta_weight=(
                config.control_channel.theta_weight
                if theta_channel_weight is None
                else float(theta_channel_weight)
            ),
        ),
        raw=raw,
        config_hash=config_hash,
    )
    _validate(
        result.paths,
        result.data,
        result.model,
        result.loss,
        result.training,
        result.runtime,
        result.wandb,
        result.causal_rollout,
        result.ood,
        result.control_channel,
    )
    return result


def _validate(
    paths: PathsConfig,
    data: DataConfig,
    model: ModelConfig,
    loss: LossConfig,
    training: TrainingConfig,
    runtime: RuntimeConfig,
    wandb: WandbConfig,
    causal_rollout: CausalRolloutConfig,
    ood: OodConfig,
    control_channel: ControlChannelConfig,
) -> None:
    if not paths.all_source_dirs or len(set(paths.all_source_dirs)) != len(paths.all_source_dirs):
        raise ValueError("At least one unique raw source directory is required.")
    for name, window, order in (
        ("ball", data.ball_window, data.ball_polyorder),
        ("theta", data.theta_window, data.theta_polyorder),
        ("drone_velocity", data.drone_velocity_window, data.drone_velocity_polyorder),
    ):
        if window < 3 or window % 2 == 0 or order >= window:
            raise ValueError(f"Invalid {name} Savitzky-Golay window/order.")
    if data.gap_factor <= 1.0 or data.edge_trim < data.ball_window // 2:
        raise ValueError("gap_factor and edge_trim are inconsistent with filtering.")
    if tuple(model.input_names) != INPUT_NAMES:
        raise ValueError(f"model.input_names must be exactly {INPUT_NAMES}.")
    if len(model.hidden_dims) != 2 or any(dim <= 0 for dim in model.hidden_dims):
        raise ValueError("The v1 MLP must contain exactly two positive hidden dimensions.")
    if model.dtype != "float64" or model.output_limit_mps2 <= 0.0:
        raise ValueError("The v1 model requires float64 and a positive output limit.")
    if len(loss.horizon_weights) != len(data.rollout_steps):
        raise ValueError("One horizon weight is required for every rollout step.")
    if abs(sum(loss.horizon_weights) - 1.0) > 1e-9:
        raise ValueError("loss.horizon_weights must sum to one.")
    if max(data.rollout_steps) > max(data.evaluation_steps):
        raise ValueError("evaluation_steps must cover all training rollout steps.")
    if data.default_split not in {"train", "validation", "test", "stress"}:
        raise ValueError("data.default_split is invalid.")
    if any(item.split not in {"train", "validation", "test", "stress"} for item in data.split_assignments):
        raise ValueError("data.split_assignments contains an invalid split.")
    selector_keys = [(item.session, item.group, item.trial_index) for item in data.split_assignments]
    if len(selector_keys) != len(set(selector_keys)):
        raise ValueError("data.split_assignments contains duplicate trial selectors.")
    edges = data.position_bin_edges_m
    if len(edges) < 2 or any(right <= left for left, right in zip(edges, edges[1:])):
        raise ValueError("data.position_bin_edges_m must be strictly increasing.")
    if edges[0] != data.ball_position_min_m or edges[-1] != data.ball_position_max_m:
        raise ValueError("Position-bin edges must span the configured ball-position range.")
    if not 0.0 <= data.empirical_sampling_fraction <= 1.0:
        raise ValueError("data.empirical_sampling_fraction must lie in [0, 1].")
    if not 0.0 < data.failure_sampling_weight <= 1.0:
        raise ValueError("data.failure_sampling_weight must lie in (0, 1].")
    if not 0.0 < ood.lower_quantile < ood.upper_quantile < 1.0:
        raise ValueError("OOD quantiles are invalid.")
    if not 0.0 <= ood.shrinkage <= 1.0 or not 0.0 < ood.mahalanobis_quantile < 1.0:
        raise ValueError("OOD shrinkage or Mahalanobis quantile is invalid.")
    if training.seed < 0 or training.batch_size <= 0 or training.steps_per_epoch <= 0:
        raise ValueError("Training requires a non-negative seed and positive batch settings.")
    if runtime.device not in {"auto", "cpu", "cuda"} or runtime.gpu_min_speedup < 1.0:
        raise ValueError("Invalid runtime device selection configuration.")
    if wandb.mode not in {"auto", "online", "offline", "disabled"}:
        raise ValueError("wandb.mode must be auto, online, offline, or disabled.")
    if causal_rollout.enabled:
        if len(causal_rollout.pending_command_fields) < 1:
            raise ValueError("causal_rollout.pending_command_fields must not be empty.")
        if not 0.0 <= causal_rollout.smooth_rollout_weight:
            raise ValueError("causal_rollout.smooth_rollout_weight must be non-negative.")
        if causal_rollout.jacobian_weight < 0.0:
            raise ValueError("causal_rollout.jacobian_weight must be non-negative.")
        if causal_rollout.jacobian_norm_limit_mps2 <= 0.0:
            raise ValueError("causal_rollout.jacobian_norm_limit_mps2 must be positive.")
        if causal_rollout.jacobian_p99_factor < 1.0 or causal_rollout.jacobian_max_factor < 1.0:
            raise ValueError("Causal Jacobian acceptance factors must be at least one.")
        if causal_rollout.state_scale_floor <= 0.0:
            raise ValueError("causal_rollout.state_scale_floor must be positive.")
    if control_channel.enabled:
        if not (
            control_channel.theta_ratio_hard_min
            <= control_channel.theta_ratio_soft_min
            < control_channel.theta_ratio_soft_max
            <= control_channel.theta_ratio_hard_max
        ):
            raise ValueError("Control-channel soft/hard theta sectors are inconsistent.")
        if control_channel.theta_weight < 0.0 or control_channel.anti_damping_weight < 0.0:
            raise ValueError("Control-channel loss weights must be non-negative.")
        if not 0.0 <= control_channel.collocation_fraction <= 1.0:
            raise ValueError("control_channel.collocation_fraction must lie in [0, 1].")
        if not (
            control_channel.collocation_vb_min_mps
            < control_channel.collocation_vb_max_mps
            and control_channel.collocation_theta_min_rad
            < control_channel.collocation_theta_max_rad
            and control_channel.collocation_vrz_min_mps
            < control_channel.collocation_vrz_max_mps
            and control_channel.outer_theta_min_rad
            < control_channel.outer_theta_max_rad
        ):
            raise ValueError("Control-channel collocation/outer ranges are invalid.")
        if control_channel.outer_safety_points <= 0 or control_channel.sensitivity_max_windows <= 0:
            raise ValueError("Control-channel evaluation sizes must be positive.")
