"""Prepare the RBF dataset while preserving the validated causal log contract."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from residual_dynamics.data import (
    TrialData,
    load_processed_trials as load_legacy_trials,
    prepare_dataset as prepare_legacy_dataset,
    sha256_file,
)

from .config import Config


RBF_MANIFEST_VERSION = 1


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _window_starts(
    arrays: dict[str, np.ndarray], horizon: int, *, stride: int = 1
) -> np.ndarray:
    valid = arrays["rbf_valid"].astype(bool)
    segments = arrays["segment_id"]
    starts: list[int] = []
    for segment in np.unique(segments):
        indices = np.flatnonzero((segments == segment) & valid)
        if indices.size <= horizon:
            continue
        first, last = int(indices[0]), int(indices[-1])
        for start in range(first, last - horizon + 1, stride):
            endpoint = start + horizon
            if segments[endpoint] != segment:
                continue
            if bool(valid[start : endpoint + 1].all()):
                starts.append(start)
    return np.asarray(starts, dtype=np.int64)


def position_bins(values: np.ndarray, edges: Iterable[float]) -> np.ndarray:
    edge_array = np.asarray(tuple(edges), dtype=np.float64)
    result = np.searchsorted(edge_array, values, side="right") - 1
    result[values == edge_array[-1]] = edge_array.size - 2
    outside = (values < edge_array[0]) | (values > edge_array[-1])
    result[outside] = -1
    return result.astype(np.int64)


def velocity_bins(values: np.ndarray, edges: Iterable[float]) -> np.ndarray:
    edge_array = np.asarray(tuple(edges), dtype=np.float64)
    result = np.searchsorted(edge_array, values, side="right") - 1
    # The central bin is closed on both sides by contract.
    central_lower = int(np.flatnonzero(np.isclose(edge_array, -0.02))[0])
    result[np.isclose(values, -0.02)] = central_lower
    result[np.isclose(values, 0.02)] = central_lower
    return np.clip(result, 0, edge_array.size - 2).astype(np.int64)


def actuation_regimes(values: np.ndarray, threshold: float) -> np.ndarray:
    return np.where(values < -threshold, -1, np.where(values > threshold, 1, 0)).astype(
        np.int8
    )


def _ood_statistics(features: np.ndarray, config: Config) -> dict[str, Any]:
    mean = features.mean(axis=0)
    std = np.maximum(features.std(axis=0), 1.0e-8)
    z = (features - mean) / std
    correlation = np.corrcoef(z, rowvar=False)
    shrinkage = float(config.legacy.ood.shrinkage)
    shrunk = (1.0 - shrinkage) * correlation + shrinkage * np.eye(2)
    inverse = np.linalg.inv(shrunk)
    distance = np.sqrt(np.einsum("ni,ij,nj->n", z, inverse, z))
    return {
        "input_names": ["pb", "vb"],
        "mean": mean.tolist(),
        "std": std.tolist(),
        "min": features.min(axis=0).tolist(),
        "max": features.max(axis=0).tolist(),
        "lower_quantile": np.quantile(
            features, config.legacy.ood.lower_quantile, axis=0
        ).tolist(),
        "upper_quantile": np.quantile(
            features, config.legacy.ood.upper_quantile, axis=0
        ).tolist(),
        "shrinkage": shrinkage,
        "correlation": correlation.tolist(),
        "inverse_correlation": inverse.tolist(),
        "mahalanobis_quantile": float(config.legacy.ood.mahalanobis_quantile),
        "mahalanobis_threshold": float(
            np.quantile(distance, config.legacy.ood.mahalanobis_quantile)
        ),
        "decay_width_std": config.rbf.ood_decay_width_std,
        "min_weight": config.rbf.ood_min_weight,
        "sample_count": int(features.shape[0]),
        "source_split": "train",
    }


def ood_numpy(features: np.ndarray, stats: dict[str, Any]) -> dict[str, np.ndarray]:
    values = np.asarray(features, dtype=np.float64)
    mean = np.asarray(stats["mean"])
    std = np.asarray(stats["std"])
    lower = np.asarray(stats["lower_quantile"])
    upper = np.asarray(stats["upper_quantile"])
    below = (lower - values) / std
    above = (values - upper) / std
    marginal_components = np.maximum(0.0, np.maximum(below, above))
    marginal = np.linalg.norm(marginal_components, axis=-1)
    z = (values - mean) / std
    inverse = np.asarray(stats["inverse_correlation"])
    mahalanobis = np.sqrt(np.maximum(0.0, np.einsum("...i,ij,...j->...", z, inverse, z)))
    joint_excess = np.maximum(0.0, mahalanobis - stats["mahalanobis_threshold"])
    distance = np.sqrt(marginal**2 + joint_excess**2)
    width = float(stats["decay_width_std"])
    minimum = float(stats["min_weight"])
    weight = minimum + (1.0 - minimum) * np.exp(-0.5 * (distance / width) ** 2)
    return {
        "marginal": marginal,
        "mahalanobis": mahalanobis,
        "joint_excess": joint_excess,
        "distance": distance,
        "weight": weight,
    }


def prepare_dataset(config: Config) -> dict[str, Any]:
    """Rebuild raw logs into the isolated cache and add RBF-specific contracts."""

    manifest = prepare_legacy_dataset(config.legacy)
    legacy_normalization = dict(manifest["normalization"])
    processed_dir = config.paths.processed_dir
    train_features: list[np.ndarray] = []
    train_targets: list[np.ndarray] = []
    coverage: dict[str, dict[str, Any]] = {}

    for entry in manifest["entries"]:
        if not entry.get("included"):
            continue
        path = processed_dir / entry["processed_path"]
        with np.load(path, allow_pickle=False) as archive:
            arrays = {name: archive[name].copy() for name in archive.files}
        required = (
            "time_s",
            "pb",
            "vb",
            "theta",
            "omega",
            "a_nom",
            "residual_target",
            "command_z",
            "pending_commands",
        )
        valid = np.ones(arrays["time_s"].shape, dtype=bool)
        for name in required:
            value = arrays[name]
            if value.ndim == 1:
                valid &= np.isfinite(value)
            else:
                valid &= np.isfinite(value).all(axis=tuple(range(1, value.ndim)))
        valid &= (arrays["pb"] >= 0.0) & (arrays["pb"] <= 0.7)
        valid &= np.abs(arrays["vb"]) <= config.sampling.max_abs_velocity_mps
        arrays["rbf_valid"] = valid
        arrays["rbf_features"] = np.column_stack((arrays["pb"], arrays["vb"])).astype(
            np.float64
        )
        np.savez_compressed(path, **arrays)
        entry["processed_sha256"] = sha256_file(path)
        entry["valid_point_count"] = int(valid.sum())
        entry["excluded_nonfinite_or_range_count"] = int((~valid).sum())
        entry["valid_integral_window_count"] = int(_window_starts(arrays, 12).size)
        entry["valid_causal_window_count"] = int(_window_starts(arrays, 33).size)
        if entry["split"] == "train":
            train_features.append(arrays["rbf_features"][valid])
            train_targets.append(arrays["residual_target"][valid])

    if not train_features:
        raise RuntimeError("The RBF train split is empty after validity filtering.")
    features = np.concatenate(train_features)
    targets = np.concatenate(train_targets)
    quantile = float(np.quantile(np.abs(targets), config.rbf.cap_quantile))
    data_cap = config.rbf.cap_step_mps2 * math.ceil(
        quantile / config.rbf.cap_step_mps2
    )
    effective_cap = float(
        np.clip(data_cap, config.rbf.cap_min_mps2, config.rbf.cap_max_mps2)
    )
    rbf_normalization = _ood_statistics(features, config)
    rbf_normalization.update(
        {
            "effective_cap_mps2": effective_cap,
            "cap_source_quantile": config.rbf.cap_quantile,
            "cap_source_value_mps2": quantile,
            "cap_step_mps2": config.rbf.cap_step_mps2,
        }
    )
    # Preserve state scales used by rollout diagnostics.
    rbf_normalization["state_scale_names"] = legacy_normalization["state_scale_names"]
    rbf_normalization["state_scales"] = legacy_normalization["state_scales"]

    # The read-only legacy loader verifies per-trial hashes. Persist the augmented
    # hashes before using it to reconstruct TrialData objects for coverage reporting.
    _write_json(processed_dir / "manifest.json", manifest)
    trials = load_legacy_trials(config.legacy)
    for split in ("train", "validation", "test", "stress"):
        selected = [trial for trial in trials if trial.split == split]
        if not selected:
            continue
        position_counts = np.zeros(7, dtype=np.int64)
        velocity_counts = np.zeros(9, dtype=np.int64)
        joint_counts = np.zeros((7, 9), dtype=np.int64)
        actuation_counts = {-1: 0, 0: 0, 1: 0}
        speed_macro = {-1: 0, 0: 0, 1: 0}
        strata: dict[tuple[str, str, int, int, int], int] = {}
        window_count = 0
        for trial in selected:
            starts = _window_starts(trial.arrays, 33)
            if starts.size == 0:
                continue
            mids = starts + 16
            pbins = position_bins(
                trial.arrays["pb"][mids], config.legacy.data.position_bin_edges_m
            )
            vbins = velocity_bins(
                trial.arrays["vb"][mids], config.sampling.velocity_bin_edges_mps
            )
            acts = actuation_regimes(
                trial.arrays["a_nom"][mids], config.sampling.actuation_threshold_mps2
            )
            macros = np.where(
                trial.arrays["vb"][mids] < -0.02,
                -1,
                np.where(trial.arrays["vb"][mids] > 0.02, 1, 0),
            )
            for pbin, vbin, act, macro in zip(pbins, vbins, acts, macros):
                if pbin < 0:
                    continue
                position_counts[pbin] += 1
                velocity_counts[vbin] += 1
                joint_counts[pbin, vbin] += 1
                actuation_counts[int(act)] += 1
                speed_macro[int(macro)] += 1
                key = (trial.session, trial.group, int(pbin), int(vbin), int(act))
                strata[key] = strata.get(key, 0) + 1
                window_count += 1
        coverage[split] = {
            "trial_count": len(selected),
            "valid_causal_window_count": window_count,
            "position_window_counts": position_counts.tolist(),
            "velocity_window_counts": velocity_counts.tolist(),
            "joint_window_counts": joint_counts.tolist(),
            "actuation_window_counts": {str(k): v for k, v in actuation_counts.items()},
            "speed_macro_window_counts": {str(k): v for k, v in speed_macro.items()},
            "strata": [
                {
                    "session": key[0],
                    "group": key[1],
                    "position_bin": key[2],
                    "velocity_bin": key[3],
                    "actuation_regime": key[4],
                    "window_count": count,
                }
                for key, count in sorted(strata.items())
            ],
        }

    deficient_position = [
        (index, count)
        for index, count in enumerate(coverage["train"]["position_window_counts"])
        if count < config.sampling.min_windows_per_position_bin
    ]
    deficient_speed = [
        (key, count)
        for key, count in coverage["train"]["speed_macro_window_counts"].items()
        if count < config.sampling.min_windows_per_speed_macro
    ]
    if deficient_position or deficient_speed:
        raise RuntimeError(
            f"Insufficient RBF coverage: position={deficient_position}, speed={deficient_speed}"
        )

    manifest["rbf_manifest_version"] = RBF_MANIFEST_VERSION
    manifest["config_hash"] = config.config_hash
    manifest["normalization"] = rbf_normalization
    manifest["shadow_full_state_normalization"] = legacy_normalization
    manifest["coverage"] = coverage
    manifest["rbf"] = {
        "input_names": ["pb", "vb"],
        "basis_count": len(config.rbf.pb_centers_m) * len(config.rbf.vb_centers_mps),
        "pb_centers_m": list(config.rbf.pb_centers_m),
        "vb_centers_mps": list(config.rbf.vb_centers_mps),
        "pb_width_m": config.rbf.pb_width_m,
        "vb_width_mps": config.rbf.vb_width_mps,
        "effective_cap_mps2": effective_cap,
    }
    identity = {
        "rbf_manifest_version": RBF_MANIFEST_VERSION,
        "config_hash": config.config_hash,
        "processed_sha256": [
            entry.get("processed_sha256")
            for entry in manifest["entries"]
            if entry.get("included")
        ],
        "normalization": rbf_normalization,
        "rbf": manifest["rbf"],
    }
    manifest["dataset_hash"] = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    _write_json(processed_dir / "normalization.json", rbf_normalization)
    _write_json(processed_dir / "manifest.json", manifest)
    return manifest


def load_manifest(config: Config) -> dict[str, Any]:
    path = config.paths.processed_dir / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing processed dataset: {path}. Run prepare first.")
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if value.get("rbf_manifest_version") != RBF_MANIFEST_VERSION:
        raise ValueError("Unsupported RBF manifest version.")
    if value.get("config_hash") != config.config_hash:
        raise RuntimeError("Processed dataset and configuration hashes differ.")
    return value


def load_trials(config: Config, split: str) -> list[TrialData]:
    load_manifest(config)
    return load_legacy_trials(config.legacy, split=split)
