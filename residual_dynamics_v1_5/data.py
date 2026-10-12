"""Versioned preprocessing and fit/test manifests for residual dynamics v1.5."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy.signal import savgol_filter

from .config import AppConfig, REPO_ROOT


MANIFEST_VERSION = 1
TRIAL_RE = re.compile(r"_trial_(\d{3})_")


@dataclass(frozen=True)
class PhysicalConstants:
    plank_length_m: float
    rope_length_m: float
    ball_position_offset_m: float
    gravity_mps2: float
    acceleration_gain: float

    def as_dict(self) -> dict[str, float]:
        return dict(self.__dict__)


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

    @property
    def size(self) -> int:
        return int(self.arrays["time_s"].size)


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


def calculate_geometry(theta: np.ndarray, vertical_velocity: np.ndarray,
                       constants: PhysicalConstants) -> tuple[np.ndarray, np.ndarray]:
    sin_beta = (constants.plank_length_m / constants.rope_length_m) * (1.0 - np.cos(theta))
    sin_beta = np.clip(sin_beta, -1.0 + 1e-9, 1.0 - 1e-9)
    beta = np.arcsin(sin_beta)
    denominator = constants.plank_length_m * np.cos(beta - theta)
    denominator = np.where(np.abs(denominator) < 1e-9,
                           np.copysign(1e-9, denominator + 1e-15), denominator)
    omega = -vertical_velocity * np.cos(beta) / denominator
    return beta, omega


def nominal_acceleration(pb: np.ndarray, theta: np.ndarray, omega: np.ndarray,
                         constants: PhysicalConstants) -> np.ndarray:
    return constants.acceleration_gain * (
        (pb + constants.ball_position_offset_m - constants.plank_length_m) * omega**2
        - constants.gravity_mps2 * np.sin(theta)
    )


def prepare_datasets(config: AppConfig) -> dict[str, Any]:
    processed = config.paths.processed_dir
    processed.mkdir(parents=True, exist_ok=True)
    fit_manifest, fit_trials = _prepare_source(
        config.paths.fit_source_dir, "fit", processed / "fit_trials", config
    )
    counts = fit_manifest["split_counts"]
    if counts != {"train": 20, "validation": 9, "excluded": 1}:
        raise RuntimeError(f"Expected fit split 20/9/1, found {counts}")

    train_dynamic = np.concatenate([
        trial.arrays["features"][trial.arrays["dynamic_mask"]]
        for trial in fit_trials if trial.split == "train"
    ])
    if train_dynamic.size == 0:
        raise RuntimeError("No dynamic training samples survived preprocessing")
    normalization = {
        "input_names": list(config.model.input_names),
        "mean": train_dynamic.mean(axis=0).tolist(),
        "std": np.maximum(train_dynamic.std(axis=0), 1e-8).tolist(),
        "min": train_dynamic.min(axis=0).tolist(),
        "max": train_dynamic.max(axis=0).tolist(),
        "sample_count": int(train_dynamic.shape[0]),
        "source_split": "train_dynamic_only",
    }
    fit_manifest["normalization"] = normalization
    fit_manifest["dataset_hash"] = _manifest_hash(fit_manifest, normalization)
    _write_json(processed / "normalization.json", normalization)
    _write_json(processed / "fit_manifest.json", fit_manifest)

    test_manifest, _ = _prepare_source(
        config.paths.test_source_dir, "test", processed / "test_trials", config
    )
    test_counts = test_manifest["split_counts"]
    if test_counts != {"test": 14, "stress": 5}:
        raise RuntimeError(f"Expected test split 14/5, found {test_counts}")
    test_manifest["fit_dataset_hash"] = fit_manifest["dataset_hash"]
    test_manifest["dataset_hash"] = _manifest_hash(test_manifest, None)
    _write_json(processed / "test_manifest.json", test_manifest)
    return {"fit": fit_manifest, "test": test_manifest}


def _prepare_source(source_dir: Path, kind: str, output_dir: Path,
                    config: AppConfig) -> tuple[dict[str, Any], list[TrialData]]:
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Source directory does not exist: {source_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []
    trials: list[TrialData] = []
    for metadata_path in sorted(source_dir.rglob("metadata.json")):
        trial_dir = metadata_path.parent
        group = trial_dir.parent.name
        samples_path = trial_dir / "samples.csv"
        metadata = _read_json(metadata_path)
        split, reason = _assign_split(kind, group, trial_dir.name, metadata, config)
        entry: dict[str, Any] = {
            "trial_id": trial_dir.name, "group": group, "session": kind,
            "end_reason": metadata.get("end_reason"), "split": split or "excluded",
            "included": False, "reason": reason,
            "samples_path": relative_repo_path(samples_path),
            "metadata_path": relative_repo_path(metadata_path),
            "samples_sha256": sha256_file(samples_path) if samples_path.is_file() else None,
            "metadata_sha256": sha256_file(metadata_path),
        }
        if split is None or not samples_path.is_file():
            if not samples_path.is_file():
                entry["reason"] = "missing samples.csv"
            entries.append(entry)
            continue
        try:
            trial = _process_trial(samples_path, metadata, group, kind, split, config)
        except (KeyError, OSError, ValueError) as exc:
            entry["reason"] = f"processing failed: {exc}"
            entries.append(entry)
            continue
        output_path = output_dir / f"{group}__{trial_dir.name}.npz"
        np.savez_compressed(output_path, **trial.arrays)
        entry.update({
            "included": True, "reason": None,
            "processed_path": output_path.relative_to(config.paths.processed_dir).as_posix(),
            "processed_sha256": sha256_file(output_path), "sample_count": trial.size,
            "valid_count": int(trial.arrays["valid_mask"].sum()),
            "dynamic_count": int(trial.arrays["dynamic_mask"].sum()),
            "static_count": int(trial.arrays["static_mask"].sum()),
            "transition_count": int(trial.arrays["transition_mask"].sum()),
            "segment_count": int(np.unique(trial.arrays["segment_id"]).size),
            "constants": trial.constants.as_dict(),
        })
        trials.append(trial)
        entries.append(entry)
    split_counts: dict[str, int] = {}
    for entry in entries:
        split_counts[entry["split"]] = split_counts.get(entry["split"], 0) + 1
    return ({
        "manifest_version": MANIFEST_VERSION, "kind": kind,
        "source_dir": relative_repo_path(source_dir),
        "processed_dir": relative_repo_path(config.paths.processed_dir),
        "config_hash": config.config_hash, "split_counts": split_counts,
        "preprocessing": {
            "gap_factor": config.data.gap_factor,
            "ball_savgol": [config.data.ball_window, config.data.ball_polyorder],
            "theta_savgol": [config.data.theta_window, config.data.theta_polyorder],
            "drone_velocity_savgol": [config.data.drone_velocity_window, config.data.drone_velocity_polyorder],
            "edge_trim": config.data.edge_trim,
            "jump_threshold_m": config.data.jump_threshold_m,
            "jump_dilation_samples": config.data.jump_dilation_samples,
            "dynamic_speed_mps": config.data.dynamic_speed_mps,
            "static_speed_mps": config.data.static_speed_mps,
            "static_acceleration_mps2": config.data.static_acceleration_mps2,
        },
        "entries": entries,
    }, trials)


def _assign_split(kind: str, group: str, trial_id: str, metadata: dict[str, Any],
                  config: AppConfig) -> tuple[str | None, str | None]:
    end_reason = metadata.get("end_reason")
    if kind == "test":
        return ("test", None) if end_reason == "tracking_duration" else ("stress", None)
    if end_reason != "tracking_duration":
        return None, f"end_reason={end_reason}"
    match = TRIAL_RE.search(trial_id)
    if match is None:
        return None, "trial index not found"
    index = int(match.group(1))
    if group in config.data.standard_groups:
        if index in (1, 2):
            return "train", None
        if index == 3:
            return "validation", None
        return None, f"unsupported trial index {index}"
    if group in config.data.extra_training_groups:
        return "train", None
    return None, "group outside approved fit dataset"


def _process_trial(samples_path: Path, metadata: dict[str, Any], group: str,
                   session: str, split: str, config: AppConfig) -> TrialData:
    names = (config.data.time_field, config.data.ball_field,
             config.data.theta_field, config.data.drone_velocity_field)
    columns: dict[str, list[float]] = {name: [] for name in names}
    with samples_path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        required = (*names, config.data.phase_field)
        missing = [name for name in required if name not in (reader.fieldnames or [])]
        if missing:
            raise KeyError(f"missing fields: {missing}")
        for row in reader:
            if row[config.data.phase_field] != config.data.phase_value:
                continue
            try:
                values = {name: float(row[name]) for name in names}
            except (TypeError, ValueError):
                continue
            if all(math.isfinite(value) for value in values.values()):
                for name, value in values.items():
                    columns[name].append(value)
    time_s = np.asarray(columns[config.data.time_field], dtype=np.float64)
    if time_s.size < config.data.min_segment_samples:
        raise ValueError("too few finite tracking samples")
    diffs = np.diff(time_s)
    if np.any(diffs <= 0):
        raise ValueError("timestamps are not strictly increasing")
    arrays = [np.asarray(columns[name], dtype=np.float64) for name in names[1:]]
    constants = _physical_constants(metadata)
    typical_dt = float(np.median(diffs))
    boundaries = np.flatnonzero(diffs > config.data.gap_factor * typical_dt) + 1
    index_segments = np.split(np.arange(time_s.size), boundaries)
    pieces: list[dict[str, np.ndarray]] = []
    for segment_id, indices in enumerate(index_segments):
        if indices.size < config.data.min_segment_samples:
            continue
        piece = _process_segment(time_s[indices], *(value[indices] for value in arrays),
                                 segment_id, constants, split == "stress", config)
        if piece is not None:
            pieces.append(piece)
    if not pieces:
        raise ValueError("no segment survived preprocessing")
    combined = {key: np.concatenate([piece[key] for piece in pieces]) for key in pieces[0]}
    if combined["time_s"].size <= max(config.data.evaluation_steps):
        raise ValueError("processed trial too short")
    return TrialData(samples_path.parent.name, group, session, split,
                     _reference_type(group), relative_repo_path(samples_path), combined, constants)


def _process_segment(time_s: np.ndarray, raw_pb: np.ndarray, raw_theta: np.ndarray,
                     raw_vrz: np.ndarray, segment_id: int, constants: PhysicalConstants,
                     stress: bool, config: AppConfig) -> dict[str, np.ndarray] | None:
    dt = float(np.median(np.diff(time_s)))
    grid = time_s[0] + np.arange(int(np.floor((time_s[-1] - time_s[0]) / dt)) + 1) * dt
    if grid.size < config.data.min_segment_samples:
        return None
    pb_i = np.interp(grid, time_s, raw_pb)
    theta_i = np.interp(grid, time_s, raw_theta)
    vrz_i = np.interp(grid, time_s, raw_vrz)
    pb = savgol_filter(pb_i, config.data.ball_window, config.data.ball_polyorder,
                       deriv=0, delta=dt, mode="interp")
    vb = savgol_filter(pb_i, config.data.ball_window, config.data.ball_polyorder,
                       deriv=1, delta=dt, mode="interp")
    ab = savgol_filter(pb_i, config.data.ball_window, config.data.ball_polyorder,
                       deriv=2, delta=dt, mode="interp")
    theta = savgol_filter(theta_i, config.data.theta_window, config.data.theta_polyorder,
                          mode="interp")
    vrz = savgol_filter(vrz_i, config.data.drone_velocity_window,
                        config.data.drone_velocity_polyorder, mode="interp")
    trim = config.data.edge_trim
    sl = slice(trim, -trim)
    grid, pb_i, theta_i, vrz_i, pb, vb, ab, theta, vrz = (
        value[sl] for value in (grid, pb_i, theta_i, vrz_i, pb, vb, ab, theta, vrz)
    )
    _, omega = calculate_geometry(theta, vrz, constants)
    a_nom = nominal_acceleration(pb, theta, omega, constants)
    residual = ab - a_nom
    size = len(grid)
    jump = np.zeros(size, dtype=bool)
    jump_indices = np.flatnonzero(np.abs(np.diff(pb_i, prepend=pb_i[0])) > config.data.jump_threshold_m)
    for index in jump_indices:
        jump[max(0, index - config.data.jump_dilation_samples):
             min(size, index + config.data.jump_dilation_samples + 1)] = True
    if stress and jump_indices.size:
        jump[max(0, int(jump_indices[0]) - config.data.jump_dilation_samples):] = True
    finite = np.isfinite(np.column_stack((pb, vb, ab, theta, omega, a_nom, residual))).all(axis=1)
    valid = (finite & ~jump & (np.abs(vb) <= config.data.max_abs_velocity_mps)
             & (np.abs(ab) <= config.data.max_abs_acceleration_mps2)
             & (np.abs(residual) <= config.data.max_abs_residual_mps2))
    radius = config.data.direction_radius_samples
    span = 2 * radius + 1
    positive = np.convolve((valid & (vb >= config.data.dynamic_speed_mps)).astype(int),
                           np.ones(span, dtype=int), mode="same") == span
    negative = np.convolve((valid & (vb <= -config.data.dynamic_speed_mps)).astype(int),
                           np.ones(span, dtype=int), mode="same") == span
    dynamic = positive | negative
    static_candidate = valid & (np.abs(vb) <= config.data.static_speed_mps) & (
        np.abs(ab) <= config.data.static_acceleration_mps2)
    persistence = config.data.static_persistence_samples
    static = np.convolve(static_candidate.astype(int), np.ones(persistence, dtype=int),
                         mode="same") == persistence
    transition = valid & ~dynamic & ~static
    features = np.column_stack((pb, vb, theta, omega, a_nom)).astype(np.float64)
    return {
        "time_s": grid.astype(np.float64), "dt_s": np.full(size, dt),
        "raw_pb": pb_i.astype(np.float64), "raw_theta": theta_i.astype(np.float64),
        "raw_vrz": vrz_i.astype(np.float64), "pb": pb.astype(np.float64),
        "vb": vb.astype(np.float64), "ab": ab.astype(np.float64),
        "theta": theta.astype(np.float64), "vrz": vrz.astype(np.float64),
        "omega": omega.astype(np.float64), "a_nom": a_nom.astype(np.float64),
        "residual_target": residual.astype(np.float64), "features": features,
        "valid_mask": valid, "dynamic_mask": dynamic, "static_mask": static,
        "transition_mask": transition, "invalid_mask": ~valid, "jump_mask": jump,
        "segment_id": np.full(size, segment_id, dtype=np.int32),
    }


def load_manifest(config: AppConfig, kind: str) -> dict[str, Any]:
    path = config.paths.processed_dir / f"{kind}_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}; run prepare first")
    manifest = _read_json(path)
    if manifest.get("manifest_version") != MANIFEST_VERSION:
        raise ValueError("Unsupported manifest version")
    return manifest


def load_trials(config: AppConfig, splits: str | Sequence[str]) -> list[TrialData]:
    requested = {splits} if isinstance(splits, str) else set(splits)
    kind = "fit" if requested <= {"train", "validation"} else "test"
    manifest = load_manifest(config, kind)
    trials: list[TrialData] = []
    for entry in manifest["entries"]:
        if not entry.get("included") or entry["split"] not in requested:
            continue
        path = config.paths.processed_dir / entry["processed_path"]
        if sha256_file(path) != entry["processed_sha256"]:
            raise RuntimeError(f"Processed hash mismatch: {path}")
        with np.load(path, allow_pickle=False) as archive:
            arrays = {name: archive[name].copy() for name in archive.files}
        trials.append(TrialData(
            entry["trial_id"], entry["group"], entry["session"], entry["split"],
            _reference_type(entry["group"]), entry["samples_path"], arrays,
            PhysicalConstants(**entry["constants"]),
        ))
    if not trials:
        raise RuntimeError(f"No trials found for splits={sorted(requested)}")
    return trials


def _physical_constants(metadata: dict[str, Any]) -> PhysicalConstants:
    parameters = metadata["parameters"]
    mass = float(parameters["controller.model.ball_mass"])
    radius = float(parameters["controller.model.ball_radius"])
    inertia_ratio = float(parameters["controller.model.ball_inertia_ratio"])
    inertia = inertia_ratio * mass * radius**2
    effective_mass = inertia / radius**2 + mass
    return PhysicalConstants(
        float(parameters["controller.model.plank_length"]),
        float(parameters["controller.model.rope_length"]),
        float(parameters["controller.model.ball_position_offset"]),
        abs(float(parameters["controller.model.gravity"])), mass / effective_mass,
    )


def _reference_type(group: str) -> str:
    return "constant" if group.startswith("constant") else group.split("_", 1)[0]


def _manifest_hash(manifest: dict[str, Any], normalization: dict[str, Any] | None) -> str:
    identity = {
        "kind": manifest["kind"], "config_hash": manifest["config_hash"],
        "normalization": normalization,
        "sources": [{"samples": item["samples_sha256"], "metadata": item["metadata_sha256"],
                     "split": item["split"]} for item in manifest["entries"] if item["included"]],
    }
    return hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
