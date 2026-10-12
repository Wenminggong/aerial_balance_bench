"""Real-world log preprocessing and processed-dataset loading."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.signal import savgol_filter

from .config import AppConfig, REPO_ROOT


MANIFEST_VERSION = 1
CAUSAL_MANIFEST_VERSION = 2
TRIAL_RE = re.compile(r"_trial_(\d{3})_")


@dataclass(frozen=True)
class PhysicalConstants:
    plank_length_m: float
    rope_length_m: float
    ball_position_offset_m: float
    gravity_mps2: float
    acceleration_gain: float

    def as_dict(self) -> dict[str, float]:
        return {
            "plank_length_m": self.plank_length_m,
            "rope_length_m": self.rope_length_m,
            "ball_position_offset_m": self.ball_position_offset_m,
            "gravity_mps2": self.gravity_mps2,
            "acceleration_gain": self.acceleration_gain,
        }


@dataclass(frozen=True)
class ResponseConstants:
    enabled: bool
    tau_s: float
    gain: float
    bias: float
    max_velocity_mps: float
    delay_steps: int

    def as_dict(self) -> dict[str, float | int | bool]:
        return {
            "enabled": self.enabled,
            "tau_s": self.tau_s,
            "gain": self.gain,
            "bias": self.bias,
            "max_velocity_mps": self.max_velocity_mps,
            "delay_steps": self.delay_steps,
        }


@dataclass
class TrialData:
    trial_id: str
    group: str
    session: str
    split: str
    reference_type: str
    source_file: str
    arrays: dict[str, np.ndarray]
    constants: PhysicalConstants
    response: ResponseConstants | None = None
    end_reason: str = "tracking_duration"
    is_failure_prefix: bool = False
    terminal_time_s: float | None = None
    raw_sample_count: int = 0
    truncated_sample_count: int = 0
    policy_model_id: str = ""
    policy_scale: float = 0.0
    source_weights_sha256: str = ""

    @property
    def size(self) -> int:
        return int(self.arrays["time_s"].size)

    @property
    def trial_uid(self) -> str:
        return f"{self.session}/{self.group}/{self.trial_id}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative_repo_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.name


def calculate_geometry(
    theta: np.ndarray,
    vertical_velocity: np.ndarray,
    constants: PhysicalConstants,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate beta and omega with the same geometry used by the controller."""

    sin_beta = (constants.plank_length_m / constants.rope_length_m) * (
        1.0 - np.cos(theta)
    )
    sin_beta = np.clip(sin_beta, -1.0 + 1e-9, 1.0 - 1e-9)
    beta = np.arcsin(sin_beta)
    denominator = constants.plank_length_m * np.cos(beta - theta)
    denominator = np.where(
        np.abs(denominator) < 1e-9,
        np.copysign(1e-9, denominator + 1e-15),
        denominator,
    )
    omega = -vertical_velocity * np.cos(beta) / denominator
    return beta, omega


def nominal_acceleration(
    pb: np.ndarray,
    theta: np.ndarray,
    omega: np.ndarray,
    constants: PhysicalConstants,
) -> np.ndarray:
    return constants.acceleration_gain * (
        (pb + constants.ball_position_offset_m - constants.plank_length_m) * omega**2
        - constants.gravity_mps2 * np.sin(theta)
    )


def prepare_dataset(config: AppConfig) -> dict[str, Any]:
    """Process every eligible trial and write a self-contained training dataset."""

    source_dirs = config.paths.all_source_dirs
    processed_dir = config.paths.processed_dir
    trials_dir = processed_dir / "trials"
    missing_sources = [path for path in source_dirs if not path.is_dir()]
    if missing_sources:
        raise FileNotFoundError(f"Raw log directory does not exist: {missing_sources[0]}")
    trials_dir.mkdir(parents=True, exist_ok=True)

    entries: list[dict[str, Any]] = []
    processed_trials: list[TrialData] = []
    for source_dir in source_dirs:
        session = _session_id(source_dir)
        for metadata_path in sorted(source_dir.rglob("metadata.json")):
            trial_dir = metadata_path.parent
            samples_path = trial_dir / "samples.csv"
            group = trial_dir.parent.name
            metadata = _read_json(metadata_path)
            trial_index = _trial_index(trial_dir.name)
            entry: dict[str, Any] = {
                "trial_id": trial_dir.name,
                "trial_uid": f"{session}/{group}/{trial_dir.name}",
                "trial_index": trial_index,
                "group": group,
                "session": session,
                "metadata_path": relative_repo_path(metadata_path),
                "samples_path": relative_repo_path(samples_path),
                "metadata_sha256": sha256_file(metadata_path),
                "samples_sha256": (
                    sha256_file(samples_path) if samples_path.is_file() else None
                ),
                "end_reason": metadata.get("end_reason"),
                "included": False,
                "split": "excluded",
                "reason": None,
            }
            split, split_reason = _assign_split(
                session, group, trial_dir.name, metadata, config
            )
            if split is None:
                entry["reason"] = split_reason
                entries.append(entry)
                continue
            if not samples_path.is_file():
                entry["reason"] = "missing samples.csv"
                entries.append(entry)
                continue
            try:
                trial = _process_trial(
                    samples_path, metadata, group, split, config, session=session
                )
            except (ValueError, KeyError, OSError) as exc:
                entry["reason"] = f"processing failed: {exc}"
                entries.append(entry)
                continue

            output_name = f"{session}__{group}__{trial_dir.name}.npz"
            output_path = trials_dir / output_name
            _write_trial_npz(output_path, trial)
            entry.update(
                {
                    "included": True,
                    "split": split,
                    "reason": None,
                    "processed_path": output_path.relative_to(processed_dir).as_posix(),
                    "processed_sha256": sha256_file(output_path),
                    "sample_count": trial.size,
                    "raw_sample_count": trial.raw_sample_count,
                    "truncated_sample_count": trial.truncated_sample_count,
                    "terminal_time_s": trial.terminal_time_s,
                    "is_failure_prefix": trial.is_failure_prefix,
                    "policy_model_id": trial.policy_model_id,
                    "policy_scale": trial.policy_scale,
                    "source_weights_sha256": trial.source_weights_sha256,
                    "segment_count": int(np.unique(trial.arrays["segment_id"]).size),
                    "dt_median_s": float(np.median(trial.arrays["dt_s"])),
                    "constants": trial.constants.as_dict(),
                    "response": (
                        None if trial.response is None else trial.response.as_dict()
                    ),
                }
            )
            entries.append(entry)
            processed_trials.append(trial)

    included = [entry for entry in entries if entry["included"]]
    train_entries = [entry for entry in included if entry["split"] == "train"]
    validation_entries = [entry for entry in included if entry["split"] == "validation"]
    legacy_split = not config.data.split_assignments
    if legacy_split and (len(train_entries) != 20 or len(validation_entries) != 9):
        raise RuntimeError(
            "Expected the approved 20/9 train/validation split, found "
            f"{len(train_entries)}/{len(validation_entries)}."
        )
    if not legacy_split:
        for session in map(_session_id, source_dirs):
            for split in ("validation", "test"):
                count = sum(
                    entry["included"]
                    and entry["session"] == session
                    and entry["split"] == split
                    for entry in entries
                )
                if count != 1:
                    raise RuntimeError(
                        f"Expected exactly one included {split} trial for {session}, "
                        f"found {count}."
                    )

    train_features = np.concatenate(
        [trial.arrays["features"] for trial in processed_trials if trial.split == "train"],
        axis=0,
    )
    feature_mean = train_features.mean(axis=0)
    feature_std = train_features.std(axis=0)
    feature_std = np.maximum(feature_std, 1e-8)
    normalization = {
        "input_names": list(config.model.input_names),
        "mean": feature_mean.tolist(),
        "std": feature_std.tolist(),
        "min": train_features.min(axis=0).tolist(),
        "max": train_features.max(axis=0).tolist(),
        "sample_count": int(train_features.shape[0]),
        "source_split": "train",
    }
    if config.causal_rollout.enabled:
        state_differences = []
        for trial in processed_trials:
            if trial.split != "train":
                continue
            state = trial.arrays["causal_state"]
            same_segment = np.diff(trial.arrays["segment_id"]) == 0
            if np.any(same_segment):
                state_differences.append(np.abs(np.diff(state, axis=0)[same_segment]))
        if not state_differences:
            raise RuntimeError("No causal state differences were available for Jacobian scaling.")
        differences = np.concatenate(state_differences, axis=0)
        state_scales = np.maximum(
            np.percentile(differences, 95.0, axis=0),
            config.causal_rollout.state_scale_floor,
        )
        normalization["state_scale_names"] = ["pb", "vb", "theta", "vrz"]
        normalization["state_scales"] = state_scales.tolist()
    if config.ood.enabled:
        normalization["ood"] = _ood_statistics(train_features, config)
    _write_json(processed_dir / "normalization.json", normalization)

    coverage = _coverage_summary(processed_trials, config)
    if config.data.min_windows_per_position_bin > 0:
        counts = coverage["train"]["position_window_counts"]
        deficient = {
            label: count
            for label, count in counts.items()
            if count < config.data.min_windows_per_position_bin
        }
        if deficient:
            raise RuntimeError(
                "Insufficient train rollout coverage by position bin: "
                + ", ".join(f"{name}={count}" for name, count in deficient.items())
            )

    dataset_identity = {
        "manifest_version": 3 if config.data.split_assignments else (
            CAUSAL_MANIFEST_VERSION if config.causal_rollout.enabled else MANIFEST_VERSION
        ),
        "processed_array_names": sorted(processed_trials[0].arrays),
        "config_hash": config.config_hash,
        "included_sources": [
            {
                "samples_sha256": entry["samples_sha256"],
                "metadata_sha256": entry["metadata_sha256"],
                "split": entry["split"],
            }
            for entry in included
        ],
        "normalization": normalization,
    }
    dataset_hash = hashlib.sha256(
        json.dumps(dataset_identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    manifest = {
        "manifest_version": 3 if config.data.split_assignments else (
            CAUSAL_MANIFEST_VERSION if config.causal_rollout.enabled else MANIFEST_VERSION
        ),
        "dataset_hash": dataset_hash,
        "config_hash": config.config_hash,
        "source_dirs": [relative_repo_path(path) for path in source_dirs],
        "processed_dir": relative_repo_path(processed_dir),
        "effective_logging_rate_note": (
            "The nominal 240 Hz laser is represented by approximately 60 Hz control snapshots."
        ),
        "split_counts": (
            {
                split: sum(
                    entry.get("included") and entry.get("split") == split
                    for entry in entries
                )
                for split in ("train", "validation", "test", "stress")
            }
            if config.data.split_assignments
            else {
                split: sum(
                    entry.get("included") and entry.get("split") == split
                    for entry in entries
                )
                for split in ("train", "validation")
            }
        )
        | {"excluded": sum(not entry.get("included") for entry in entries)},
        "normalization": normalization,
        "coverage": coverage,
        "preprocessing": {
            "mode": "causal_rollout" if config.causal_rollout.enabled else "teacher_forced",
            "gap_factor": config.data.gap_factor,
            "ball_savgol": [config.data.ball_window, config.data.ball_polyorder],
            "theta_savgol": [config.data.theta_window, config.data.theta_polyorder],
            "drone_velocity_savgol": [
                config.data.drone_velocity_window,
                config.data.drone_velocity_polyorder,
            ],
            "edge_trim": config.data.edge_trim,
        },
        "entries": entries,
    }
    _write_json(processed_dir / "manifest.json", manifest)
    return manifest


def load_manifest(config: AppConfig) -> dict[str, Any]:
    manifest_path = config.paths.processed_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Processed manifest not found: {manifest_path}. Run the prepare command first."
        )
    manifest = _read_json(manifest_path)
    expected_version = 3 if config.data.split_assignments else (
        CAUSAL_MANIFEST_VERSION if config.causal_rollout.enabled else MANIFEST_VERSION
    )
    if manifest.get("manifest_version") != expected_version:
        raise ValueError("Unsupported processed dataset manifest version.")
    return manifest


def load_processed_trials(config: AppConfig, split: str | None = None) -> list[TrialData]:
    """Load processed trial arrays without consulting the raw log directory."""

    manifest = load_manifest(config)
    trials: list[TrialData] = []
    for entry in manifest["entries"]:
        if not entry.get("included") or (split is not None and entry["split"] != split):
            continue
        path = config.paths.processed_dir / entry["processed_path"]
        if sha256_file(path) != entry["processed_sha256"]:
            raise RuntimeError(f"Processed trial hash mismatch: {path}")
        with np.load(path, allow_pickle=False) as archive:
            arrays = {name: archive[name].copy() for name in archive.files}
        constants_dict = entry["constants"]
        constants = PhysicalConstants(**constants_dict)
        response_dict = entry.get("response")
        response = None if response_dict is None else ResponseConstants(**response_dict)
        trials.append(
            TrialData(
                trial_id=entry["trial_id"],
                group=entry["group"],
                session=entry["session"],
                split=entry["split"],
                reference_type=_reference_type(entry["group"]),
                source_file=entry["samples_path"],
                arrays=arrays,
                constants=constants,
                response=response,
                end_reason=str(entry.get("end_reason", "tracking_duration")),
                is_failure_prefix=bool(entry.get("is_failure_prefix", False)),
                terminal_time_s=entry.get("terminal_time_s"),
                raw_sample_count=int(entry.get("raw_sample_count", arrays["time_s"].size)),
                truncated_sample_count=int(
                    entry.get("truncated_sample_count", arrays["time_s"].size)
                ),
                policy_model_id=str(entry.get("policy_model_id", "")),
                policy_scale=float(entry.get("policy_scale", 0.0)),
                source_weights_sha256=str(entry.get("source_weights_sha256", "")),
            )
        )
    if not trials:
        raise RuntimeError(f"No processed trials found for split={split!r}.")
    return trials


def _assign_split(
    session: str,
    group: str,
    trial_id: str,
    metadata: dict[str, Any],
    config: AppConfig,
) -> tuple[str | None, str]:
    match = TRIAL_RE.search(trial_id)
    if match is None:
        return None, "trial index not found"
    trial_index = int(match.group(1))
    if config.data.split_assignments:
        selected = [
            item
            for item in config.data.split_assignments
            if (item.session, item.group, item.trial_index)
            == (session, group, trial_index)
        ]
        split = selected[0].split if selected else config.data.default_split
        completed = metadata.get("end_reason") == "tracking_duration"
        if split in {"validation", "test"} and not completed:
            return None, f"{split} selector requires end_reason=tracking_duration"
        if split == "stress" and completed:
            return None, "stress selector requires a non-completed trial"
        if not completed:
            if not config.data.include_failure_prefixes:
                return None, f"end_reason={metadata.get('end_reason')}"
            if split == "train" and not config.data.include_failure_prefixes_in_train:
                return None, (
                    f"end_reason={metadata.get('end_reason')}; "
                    "failure prefixes disabled for supervised training"
                )
        return split, ""
    if metadata.get("end_reason") != "tracking_duration":
        return None, f"end_reason={metadata.get('end_reason')}"
    if group in config.data.standard_groups:
        if trial_index in (1, 2):
            return "train", ""
        if trial_index == 3:
            return "validation", ""
        return None, f"unsupported standard trial index {trial_index}"
    if group in config.data.extra_training_groups:
        return "train", ""
    return None, "group is outside the approved dataset"


def _process_trial(
    samples_path: Path,
    metadata: dict[str, Any],
    group: str,
    split: str,
    config: AppConfig,
    session: str | None = None,
) -> TrialData:
    if config.causal_rollout.enabled:
        return _process_causal_trial(
            samples_path, metadata, group, split, config, session=session
        )
    fields = (
        config.data.time_field,
        config.data.phase_field,
        config.data.ball_field,
        config.data.theta_field,
        config.data.drone_velocity_field,
    )
    columns: dict[str, list[Any]] = {field: [] for field in fields}
    with samples_path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = [field for field in fields if field not in (reader.fieldnames or [])]
        if missing:
            raise KeyError(f"missing required CSV fields: {missing}")
        for row in reader:
            if row[config.data.phase_field] != config.data.phase_value:
                continue
            try:
                values = {
                    config.data.time_field: float(row[config.data.time_field]),
                    config.data.ball_field: float(row[config.data.ball_field]),
                    config.data.theta_field: float(row[config.data.theta_field]),
                    config.data.drone_velocity_field: float(
                        row[config.data.drone_velocity_field]
                    ),
                }
            except (TypeError, ValueError):
                continue
            if not all(math.isfinite(value) for value in values.values()):
                continue
            for name, value in values.items():
                columns[name].append(value)

    time_s = np.asarray(columns[config.data.time_field], dtype=np.float64)
    raw_pb = np.asarray(columns[config.data.ball_field], dtype=np.float64)
    raw_theta = np.asarray(columns[config.data.theta_field], dtype=np.float64)
    raw_vrz = np.asarray(columns[config.data.drone_velocity_field], dtype=np.float64)
    if time_s.size < config.data.min_segment_samples:
        raise ValueError("too few finite tracking samples")
    diffs = np.diff(time_s)
    if np.any(diffs <= 0.0):
        raise ValueError("tracking timestamps are not strictly increasing")

    constants = _physical_constants(metadata)
    typical_dt = float(np.median(diffs))
    boundaries = np.flatnonzero(diffs > config.data.gap_factor * typical_dt) + 1
    index_segments = np.split(np.arange(time_s.size), boundaries)
    pieces: list[dict[str, np.ndarray]] = []
    for segment_id, indices in enumerate(index_segments):
        if indices.size < config.data.min_segment_samples:
            continue
        piece = _process_segment(
            time_s[indices],
            raw_pb[indices],
            raw_theta[indices],
            raw_vrz[indices],
            segment_id,
            constants,
            config,
        )
        if piece is not None:
            pieces.append(piece)
    if not pieces:
        raise ValueError("no continuous segment survived preprocessing")
    arrays = {
        key: np.concatenate([piece[key] for piece in pieces], axis=0)
        for key in pieces[0]
    }
    if arrays["time_s"].size <= max(config.data.evaluation_steps):
        raise ValueError("processed trial is too short for evaluation")
    return TrialData(
        trial_id=samples_path.parent.name,
        group=group,
        session=session or group,
        split=split,
        reference_type=_reference_type(group),
        source_file=relative_repo_path(samples_path),
        arrays=arrays,
        constants=constants,
        end_reason=str(metadata.get("end_reason", "tracking_duration")),
        is_failure_prefix=metadata.get("end_reason") != "tracking_duration",
        raw_sample_count=int(time_s.size),
        truncated_sample_count=int(time_s.size),
        **_policy_provenance(metadata),
    )


def _process_causal_trial(
    samples_path: Path,
    metadata: dict[str, Any],
    group: str,
    split: str,
    config: AppConfig,
    session: str | None = None,
) -> TrialData:
    causal = config.causal_rollout
    numeric_fields = (
        config.data.time_field,
        config.data.ball_field,
        causal.ball_velocity_field,
        config.data.theta_field,
        config.data.drone_velocity_field,
        causal.command_field,
    )
    columns: dict[str, list[float]] = {name: [] for name in numeric_fields}
    pending_values: list[list[float]] = []
    raw_sample_count = 0
    terminal_time_s: float | None = None
    is_failure = metadata.get("end_reason") != "tracking_duration"
    with samples_path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        required = (*numeric_fields, config.data.phase_field, *causal.pending_command_fields)
        missing = [name for name in required if name not in (reader.fieldnames or [])]
        if missing:
            raise KeyError(f"missing required causal CSV fields: {missing}")
        for row in reader:
            if row[config.data.phase_field] != config.data.phase_value:
                continue
            raw_sample_count += 1
            if is_failure and _terminal_row(row):
                try:
                    terminal_time_s = float(row[config.data.time_field])
                except (TypeError, ValueError):
                    terminal_time_s = None
                break
            try:
                values = {name: float(row[name]) for name in numeric_fields}
            except (TypeError, ValueError):
                if is_failure:
                    break
                continue
            if not all(math.isfinite(value) for value in values.values()):
                if is_failure:
                    break
                continue
            if not (
                config.data.ball_position_min_m
                <= values[config.data.ball_field]
                <= config.data.ball_position_max_m
            ):
                continue
            for name, value in values.items():
                columns[name].append(value)
            pending_row = []
            for name in causal.pending_command_fields:
                try:
                    pending_row.append(float(row[name]))
                except (TypeError, ValueError):
                    pending_row.append(math.nan)
            pending_values.append(pending_row)

    time_s = np.asarray(columns[config.data.time_field], dtype=np.float64)
    if time_s.size < config.data.min_segment_samples:
        raise ValueError("too few finite causal tracking samples")
    diffs = np.diff(time_s)
    if np.any(diffs <= 0.0):
        raise ValueError("causal tracking timestamps are not strictly increasing")
    response = _response_constants(metadata, config)
    if response.delay_steps != len(causal.pending_command_fields):
        raise ValueError(
            "metadata delay_steps does not match configured pending-command field count"
        )

    values = [np.asarray(columns[name], dtype=np.float64) for name in numeric_fields[1:]]
    pending = np.asarray(pending_values, dtype=np.float64)
    constants = _physical_constants(metadata)
    typical_dt = float(np.median(diffs))
    gap_boundaries = np.flatnonzero(
        diffs > min(config.data.gap_factor, 1.5) * typical_dt
    ) + 1
    reset_rows = _pending_queue_mismatch_indices(
        values[-1], pending, response.delay_steps
    )
    if is_failure and reset_rows.size:
        first_reset = int(reset_rows[0])
        terminal_time_s = float(time_s[first_reset])
        time_s = time_s[:first_reset]
        values = [value[:first_reset] for value in values]
        pending = pending[:first_reset]
        if time_s.size < config.data.min_segment_samples:
            raise ValueError("failure prefix before pending-queue reset is too short")
        diffs = np.diff(time_s)
        typical_dt = float(np.median(diffs))
        gap_boundaries = np.flatnonzero(
            diffs > min(config.data.gap_factor, 1.5) * typical_dt
        ) + 1
        reset_rows = np.empty(0, dtype=np.int64)
    # Isolate each inconsistent diagnostic row so no training window crosses a
    # queue reset. Short isolated pieces are rejected below.
    reset_boundaries = np.concatenate((reset_rows, reset_rows + 1))
    boundaries = np.unique(np.concatenate((gap_boundaries, reset_boundaries)))
    boundaries = boundaries[(boundaries > 0) & (boundaries < time_s.size)]
    pieces: list[dict[str, np.ndarray]] = []
    for segment_id, indices in enumerate(np.split(np.arange(time_s.size), boundaries)):
        if indices.size < config.data.min_segment_samples:
            continue
        piece = _process_causal_segment(
            time_s[indices],
            *(value[indices] for value in values),
            pending[indices],
            segment_id,
            constants,
            response,
            config,
        )
        if piece is not None:
            pieces.append(piece)
    if not pieces:
        raise ValueError("no causal segment survived preprocessing")
    arrays = {key: np.concatenate([piece[key] for piece in pieces]) for key in pieces[0]}
    if arrays["time_s"].size <= max(config.data.evaluation_steps):
        raise ValueError("processed causal trial is too short for evaluation")
    return TrialData(
        trial_id=samples_path.parent.name,
        group=group,
        session=session or group,
        split=split,
        reference_type=_reference_type(group),
        source_file=relative_repo_path(samples_path),
        arrays=arrays,
        constants=constants,
        response=response,
        end_reason=str(metadata.get("end_reason", "tracking_duration")),
        is_failure_prefix=is_failure,
        terminal_time_s=terminal_time_s,
        raw_sample_count=raw_sample_count,
        truncated_sample_count=int(time_s.size),
        **_policy_provenance(metadata),
    )


def _process_causal_segment(
    time_s: np.ndarray,
    causal_pb: np.ndarray,
    causal_vb: np.ndarray,
    causal_theta: np.ndarray,
    causal_vrz: np.ndarray,
    command_z: np.ndarray,
    pending: np.ndarray,
    segment_id: int,
    constants: PhysicalConstants,
    response: ResponseConstants,
    config: AppConfig,
) -> dict[str, np.ndarray] | None:
    _validate_pending_queue(command_z, pending, response.delay_steps)
    dt = float(np.median(np.diff(time_s)))
    grid = time_s[0] + np.arange(int(np.floor((time_s[-1] - time_s[0]) / dt)) + 1) * dt
    if grid.size < config.data.min_segment_samples:
        return None
    pb = np.interp(grid, time_s, causal_pb)
    vb = np.interp(grid, time_s, causal_vb)
    theta = np.interp(grid, time_s, causal_theta)
    vrz = np.interp(grid, time_s, causal_vrz)
    command_indices, command = _zero_order_hold(time_s, command_z, grid)
    # Pending diagnostics are intentionally sparse in the log. Reconstruct every
    # queue from committed commands, after asserting the non-empty logged rows.
    pending_grid = np.full((grid.size, response.delay_steps), np.nan, dtype=np.float64)
    queue_lag = response.delay_steps - 1
    for index in range(queue_lag, grid.size):
        pending_grid[index] = command[index - queue_lag : index + 1]

    smooth_pb = savgol_filter(
        pb, config.data.ball_window, config.data.ball_polyorder, deriv=0, delta=dt,
        mode="interp",
    )
    smooth_vb = savgol_filter(
        pb, config.data.ball_window, config.data.ball_polyorder, deriv=1, delta=dt,
        mode="interp",
    )
    smooth_ab = savgol_filter(
        pb, config.data.ball_window, config.data.ball_polyorder, deriv=2, delta=dt,
        mode="interp",
    )
    _, omega = calculate_geometry(theta, vrz, constants)
    a_nom = nominal_acceleration(pb, theta, omega, constants)
    residual = smooth_ab - a_nom
    # The logger records the queue after the current command is committed.
    # Its front is therefore the command consumed by the next predictor interval.
    lag = max(0, queue_lag)
    response_command = pending_grid[:, 0].copy()

    trim = max(config.data.edge_trim, lag)
    selection = slice(trim, -config.data.edge_trim)
    grid, pb, vb, theta, vrz, command, response_command, pending_grid = (
        value[selection]
        for value in (
            grid,
            pb,
            vb,
            theta,
            vrz,
            command,
            response_command,
            pending_grid,
        )
    )
    smooth_pb, smooth_vb, smooth_ab, omega, a_nom, residual = (
        value[selection]
        for value in (smooth_pb, smooth_vb, smooth_ab, omega, a_nom, residual)
    )
    if grid.size <= max(config.data.evaluation_steps) or not np.isfinite(response_command).all():
        return None
    features = np.column_stack((pb, vb, theta, omega, a_nom)).astype(np.float64)
    causal_state = np.column_stack((pb, vb, theta, vrz)).astype(np.float64)
    speed_regime = np.where(
        vb < -config.data.speed_threshold_mps,
        -1,
        np.where(vb > config.data.speed_threshold_mps, 1, 0),
    ).astype(np.int8)
    size = grid.size
    return {
        "time_s": grid.astype(np.float64),
        "dt_s": np.full(size, dt, dtype=np.float64),
        "pb": pb.astype(np.float64),
        "vb": vb.astype(np.float64),
        "theta": theta.astype(np.float64),
        "vrz": vrz.astype(np.float64),
        "causal_state": causal_state,
        "smooth_pb": smooth_pb.astype(np.float64),
        "smooth_vb": smooth_vb.astype(np.float64),
        "smooth_ab": smooth_ab.astype(np.float64),
        "ab": smooth_ab.astype(np.float64),
        "omega": omega.astype(np.float64),
        "a_nom": a_nom.astype(np.float64),
        "residual_target": residual.astype(np.float64),
        "features": features,
        "command_z": command.astype(np.float64),
        "pending_commands": pending_grid.astype(np.float64),
        "response_command": response_command.astype(np.float64),
        "speed_regime": speed_regime,
        "segment_id": np.full(size, segment_id, dtype=np.int32),
        "valid": np.ones(size, dtype=np.bool_),
    }


def _validate_pending_queue(command: np.ndarray, pending: np.ndarray, delay_steps: int) -> None:
    if delay_steps <= 0:
        return
    if pending.shape != (command.size, delay_steps):
        raise ValueError("pending-command array shape does not match delay_steps")
    errors = []
    lag = delay_steps - 1
    for index in range(lag, command.size):
        if not np.isfinite(pending[index]).all():
            continue
        expected = command[index - lag : index + 1]
        errors.append(np.max(np.abs(pending[index] - expected)))
    if not errors:
        raise ValueError("no finite pending-command diagnostics were available")
    if float(np.max(errors)) > 1.0e-9:
        raise ValueError(f"pending-command reconstruction mismatch: max error={max(errors):.3e}")


def _pending_queue_mismatch_indices(
    command: np.ndarray,
    pending: np.ndarray,
    delay_steps: int,
) -> np.ndarray:
    if delay_steps <= 0:
        return np.empty(0, dtype=np.int64)
    if pending.shape != (command.size, delay_steps):
        raise ValueError("pending-command array shape does not match delay_steps")
    lag = delay_steps - 1
    mismatches = []
    for index in range(lag, command.size):
        if not np.isfinite(pending[index]).all():
            continue
        expected = command[index - lag : index + 1]
        if float(np.max(np.abs(pending[index] - expected))) > 1.0e-9:
            mismatches.append(index)
    return np.asarray(mismatches, dtype=np.int64)


def _zero_order_hold(
    source_time: np.ndarray,
    values: np.ndarray,
    query_time: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample the most recently committed value without future interpolation."""

    indices = np.clip(
        np.searchsorted(source_time, query_time, side="right") - 1,
        0,
        source_time.size - 1,
    )
    return indices, values[indices]


def _process_segment(
    time_s: np.ndarray,
    raw_pb: np.ndarray,
    raw_theta: np.ndarray,
    raw_vrz: np.ndarray,
    segment_id: int,
    constants: PhysicalConstants,
    config: AppConfig,
) -> dict[str, np.ndarray] | None:
    dt = float(np.median(np.diff(time_s)))
    grid = time_s[0] + np.arange(int(np.floor((time_s[-1] - time_s[0]) / dt)) + 1) * dt
    if grid.size < config.data.min_segment_samples:
        return None
    pb_interp = np.interp(grid, time_s, raw_pb)
    theta_interp = np.interp(grid, time_s, raw_theta)
    vrz_interp = np.interp(grid, time_s, raw_vrz)

    pb = savgol_filter(
        pb_interp,
        config.data.ball_window,
        config.data.ball_polyorder,
        deriv=0,
        delta=dt,
        mode="interp",
    )
    vb = savgol_filter(
        pb_interp,
        config.data.ball_window,
        config.data.ball_polyorder,
        deriv=1,
        delta=dt,
        mode="interp",
    )
    ab = savgol_filter(
        pb_interp,
        config.data.ball_window,
        config.data.ball_polyorder,
        deriv=2,
        delta=dt,
        mode="interp",
    )
    theta = savgol_filter(
        theta_interp,
        config.data.theta_window,
        config.data.theta_polyorder,
        mode="interp",
    )
    vrz = savgol_filter(
        vrz_interp,
        config.data.drone_velocity_window,
        config.data.drone_velocity_polyorder,
        mode="interp",
    )
    trim = config.data.edge_trim
    selection = slice(trim, -trim)
    grid = grid[selection]
    pb_interp = pb_interp[selection]
    theta_interp = theta_interp[selection]
    vrz_interp = vrz_interp[selection]
    pb = pb[selection]
    vb = vb[selection]
    ab = ab[selection]
    theta = theta[selection]
    vrz = vrz[selection]
    _, omega = calculate_geometry(theta, vrz, constants)
    a_nom = nominal_acceleration(pb, theta, omega, constants)
    residual = ab - a_nom
    features = np.column_stack((pb, vb, theta, omega, a_nom)).astype(np.float64)
    speed_regime = np.where(
        vb < -config.data.speed_threshold_mps,
        -1,
        np.where(vb > config.data.speed_threshold_mps, 1, 0),
    ).astype(np.int8)
    size = grid.size
    return {
        "time_s": grid.astype(np.float64),
        "dt_s": np.full(size, dt, dtype=np.float64),
        "raw_pb": pb_interp.astype(np.float64),
        "raw_theta": theta_interp.astype(np.float64),
        "raw_vrz": vrz_interp.astype(np.float64),
        "pb": pb.astype(np.float64),
        "vb": vb.astype(np.float64),
        "ab": ab.astype(np.float64),
        "theta": theta.astype(np.float64),
        "vrz": vrz.astype(np.float64),
        "omega": omega.astype(np.float64),
        "a_nom": a_nom.astype(np.float64),
        "residual_target": residual.astype(np.float64),
        "features": features,
        "speed_regime": speed_regime,
        "segment_id": np.full(size, segment_id, dtype=np.int32),
        "valid": np.ones(size, dtype=np.bool_),
    }


def _physical_constants(metadata: dict[str, Any]) -> PhysicalConstants:
    parameters = metadata["parameters"]
    mass = float(parameters["controller.model.ball_mass"])
    radius = float(parameters["controller.model.ball_radius"])
    inertia_ratio = float(parameters["controller.model.ball_inertia_ratio"])
    inertia = inertia_ratio * mass * radius**2
    effective_mass = inertia / radius**2 + mass
    return PhysicalConstants(
        plank_length_m=float(parameters["controller.model.plank_length"]),
        rope_length_m=float(parameters["controller.model.rope_length"]),
        ball_position_offset_m=float(parameters["controller.model.ball_position_offset"]),
        gravity_mps2=abs(float(parameters["controller.model.gravity"])),
        acceleration_gain=mass / effective_mass,
    )


def _response_constants(metadata: dict[str, Any], config: AppConfig) -> ResponseConstants:
    parameters = metadata["parameters"]
    causal = config.causal_rollout
    enabled = bool(parameters[causal.response_enabled_parameter])
    tau_s = float(parameters[causal.response_tau_parameter]) if enabled else 0.0
    gain = float(parameters[causal.response_gain_parameter]) if enabled else 1.0
    bias = float(parameters[causal.response_bias_parameter]) if enabled else 0.0
    max_velocity = float(parameters[causal.max_velocity_parameter])
    delay_steps = int(parameters[causal.delay_steps_parameter])
    if (
        tau_s < 0.0
        or gain <= 0.0
        or max_velocity <= 0.0
        or delay_steps < 0
        or not all(math.isfinite(value) for value in (tau_s, gain, bias, max_velocity))
    ):
        raise ValueError("invalid deployment response parameters")
    return ResponseConstants(enabled, tau_s, gain, bias, max_velocity, delay_steps)


def _write_trial_npz(path: Path, trial: TrialData) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **trial.arrays)


def _reference_type(group: str) -> str:
    if group.startswith("constant"):
        return "constant"
    return group.split("_", maxsplit=1)[0]


def _session_id(source_dir: Path) -> str:
    prefix = "real_world_acados_unified_"
    return source_dir.name[len(prefix) :] if source_dir.name.startswith(prefix) else source_dir.name


def _trial_index(trial_id: str) -> int | None:
    match = TRIAL_RE.search(trial_id)
    return None if match is None else int(match.group(1))


def _terminal_row(row: dict[str, str]) -> bool:
    safety = str(row.get("safety_state", "ok")).strip().lower()
    end_reason = str(row.get("end_reason", "running")).strip().lower()
    return safety not in {"", "ok"} or end_reason not in {"", "running"}


def _policy_provenance(metadata: dict[str, Any]) -> dict[str, Any]:
    policy = metadata.get("policy", {})
    residual = policy.get("residual_dynamics", {}) if isinstance(policy, dict) else {}
    parameters = metadata.get("parameters", {})
    return {
        "policy_model_id": str(residual.get("model_id", "")),
        "policy_scale": float(
            residual.get("scale", parameters.get("residual_dynamics.scale", 0.0)) or 0.0
        ),
        "source_weights_sha256": str(residual.get("source_weights_sha256", "")),
    }


def position_bin_indices(values: np.ndarray, edges: Iterable[float]) -> np.ndarray:
    edge_array = np.asarray(tuple(edges), dtype=np.float64)
    result = np.searchsorted(edge_array, values, side="right") - 1
    result[values == edge_array[-1]] = edge_array.size - 2
    outside = (values < edge_array[0]) | (values > edge_array[-1])
    result[outside] = -1
    return result.astype(np.int16)


def position_bin_labels(edges: Iterable[float]) -> list[str]:
    values = tuple(float(item) for item in edges)
    return [
        f"[{left:.1f},{right:.1f}{']' if index == len(values) - 2 else ')'}"
        for index, (left, right) in enumerate(zip(values, values[1:]))
    ]


def _ood_statistics(features: np.ndarray, config: AppConfig) -> dict[str, Any]:
    normalized = (features - features.mean(axis=0)) / np.maximum(features.std(axis=0), 1e-8)
    correlation = np.corrcoef(normalized, rowvar=False)
    shrinkage = config.ood.shrinkage
    shrunk = (1.0 - shrinkage) * correlation + shrinkage * np.eye(features.shape[1])
    inverse = np.linalg.inv(shrunk)
    mahalanobis_squared = np.einsum("ni,ij,nj->n", normalized, inverse, normalized)
    mahalanobis = np.sqrt(np.maximum(mahalanobis_squared, 0.0))
    return {
        "method": "quantile_mahalanobis_v1",
        "lower_quantile_probability": config.ood.lower_quantile,
        "upper_quantile_probability": config.ood.upper_quantile,
        "lower": np.quantile(features, config.ood.lower_quantile, axis=0).tolist(),
        "upper": np.quantile(features, config.ood.upper_quantile, axis=0).tolist(),
        "shrinkage": shrinkage,
        "inverse_correlation": inverse.tolist(),
        "mahalanobis_quantile_probability": config.ood.mahalanobis_quantile,
        "mahalanobis_radius": float(
            np.quantile(mahalanobis, config.ood.mahalanobis_quantile)
        ),
    }


def ood_distances(features: np.ndarray, normalization: dict[str, Any]) -> dict[str, np.ndarray]:
    values = np.asarray(features, dtype=np.float64)
    std = np.asarray(normalization["std"], dtype=np.float64)
    ood = normalization.get("ood")
    if not ood:
        lower = np.asarray(normalization["min"], dtype=np.float64)
        upper = np.asarray(normalization["max"], dtype=np.float64)
        violation = np.maximum(0.0, np.maximum((lower - values) / std, (values - upper) / std))
        marginal = np.linalg.norm(violation, axis=-1)
        zeros = np.zeros_like(marginal)
        return {"combined": marginal, "marginal": marginal, "mahalanobis": zeros, "joint_excess": zeros}
    lower = np.asarray(ood["lower"], dtype=np.float64)
    upper = np.asarray(ood["upper"], dtype=np.float64)
    violation = np.maximum(0.0, np.maximum((lower - values) / std, (values - upper) / std))
    marginal = np.linalg.norm(violation, axis=-1)
    mean = np.asarray(normalization["mean"], dtype=np.float64)
    z = (values - mean) / std
    inverse = np.asarray(ood["inverse_correlation"], dtype=np.float64)
    mahalanobis = np.sqrt(np.maximum(np.einsum("...i,ij,...j->...", z, inverse, z), 0.0))
    joint = np.maximum(0.0, mahalanobis - float(ood["mahalanobis_radius"]))
    return {
        "combined": np.sqrt(marginal**2 + joint**2),
        "marginal": marginal,
        "mahalanobis": mahalanobis,
        "joint_excess": joint,
    }


def _coverage_summary(trials: list[TrialData], config: AppConfig) -> dict[str, Any]:
    labels = position_bin_labels(config.data.position_bin_edges_m)
    result: dict[str, Any] = {}
    horizon = max(config.data.rollout_steps)
    for split in ("train", "validation", "test", "stress"):
        selected = [trial for trial in trials if trial.split == split]
        position_samples = np.zeros(len(labels), dtype=np.int64)
        position_windows = np.zeros(len(labels), dtype=np.int64)
        speed_windows = {-1: 0, 0: 0, 1: 0}
        sessions: dict[str, int] = {}
        groups: dict[str, int] = {}
        end_reasons: dict[str, int] = {}
        strata: dict[tuple[str, str, str, str, str], list[int]] = {}
        for trial in selected:
            sessions[trial.session] = sessions.get(trial.session, 0) + 1
            groups[trial.group] = groups.get(trial.group, 0) + 1
            end_reasons[trial.end_reason] = end_reasons.get(trial.end_reason, 0) + 1
            sample_bins = position_bin_indices(
                trial.arrays["pb"], config.data.position_bin_edges_m
            )
            for index in range(len(labels)):
                count = int(np.sum(sample_bins == index))
                position_samples[index] += count
                for regime, regime_name in ((-1, "negative"), (0, "low_speed"), (1, "positive")):
                    sample_count = int(
                        np.sum(
                            (sample_bins == index)
                            & (trial.arrays["speed_regime"] == regime)
                        )
                    )
                    key = (
                        trial.session,
                        trial.group,
                        trial.end_reason,
                        labels[index],
                        regime_name,
                    )
                    strata.setdefault(key, [0, 0])[0] += sample_count
            segment_ids = trial.arrays["segment_id"]
            for segment_id in np.unique(segment_ids):
                indices = np.flatnonzero(segment_ids == segment_id)
                if indices.size <= horizon:
                    continue
                for start in range(int(indices[0]), int(indices[-1] - horizon) + 1):
                    if segment_ids[start + horizon] != segment_id:
                        continue
                    midpoint = start + horizon // 2
                    position_bin = int(sample_bins[midpoint])
                    if position_bin >= 0:
                        position_windows[position_bin] += 1
                    speed = float(trial.arrays["vb"][midpoint])
                    regime = -1 if speed < -config.data.speed_threshold_mps else (
                        1 if speed > config.data.speed_threshold_mps else 0
                    )
                    speed_windows[regime] += 1
                    regime_name = {-1: "negative", 0: "low_speed", 1: "positive"}[regime]
                    key = (
                        trial.session,
                        trial.group,
                        trial.end_reason,
                        labels[position_bin],
                        regime_name,
                    )
                    strata.setdefault(key, [0, 0])[1] += 1
        result[split] = {
            "trial_count": len(selected),
            "sessions": sessions,
            "groups": groups,
            "end_reasons": end_reasons,
            "position_sample_counts": dict(zip(labels, position_samples.tolist())),
            "position_window_counts": dict(zip(labels, position_windows.tolist())),
            "speed_window_counts": {
                "negative": speed_windows[-1],
                "low_speed": speed_windows[0],
                "positive": speed_windows[1],
            },
            "strata": [
                {
                    "session": key[0],
                    "group": key[1],
                    "end_reason": key[2],
                    "position_bin": key[3],
                    "speed_regime": key[4],
                    "sample_count": counts[0],
                    "window_count": counts[1],
                }
                for key, counts in sorted(strata.items())
                if counts[0] or counts[1]
            ],
        }
    return result


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
