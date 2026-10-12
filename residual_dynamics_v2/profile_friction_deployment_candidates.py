"""Retrain and freeze the deployable profile residual candidates.

This entry point deliberately trains only the two selected structured models.  It
keeps the exploratory comparison summary immutable and writes self-contained
checkpoints plus a compact, hash-pinned deployment training summary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .config import Config, load_config
from .data import load_manifest, load_trials
from .profile_friction_kappa_probe import _evaluate_candidate
from .profile_friction_probe import (
    DEFAULT_CONFIG,
    RESULT_PATH,
    ProbeSettings,
    ProfileFrictionResidual,
    _fit,
)
from .training import WindowStore


CHECKPOINT_DIR = RESULT_PATH.parent / "checkpoints"
DEPLOYMENT_SUMMARY_PATH = RESULT_PATH.parent / "deployment_candidates.json"
FORMULA_VERSION = "rbf_profile_friction_v1"
CANDIDATES = {
    "profile_only": {"profile_enabled": True, "friction_enabled": False},
    "profile_plus_friction": {"profile_enabled": True, "friction_enabled": True},
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _compact_metrics(splits: dict[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for split, values in splits.items():
        compact[split] = {
            "point": values["point"],
            "integral": values["integral"],
            "rollout": {
                horizon: values["rollout"][horizon]
                for horizon in ("h12", "h33", "h60")
                if horizon in values["rollout"]
            },
        }
    return compact


def _validation_references(
    validation_store: WindowStore, settings: ProbeSettings
) -> list[Any]:
    references = list(validation_store.references[::5])
    if len(references) > settings.validation_windows:
        indices = np.linspace(
            0, len(references) - 1, settings.validation_windows, dtype=int
        )
        references = [references[index] for index in indices]
    return references


def run(
    config: Config,
    *,
    source_summary_path: Path = RESULT_PATH,
    checkpoint_dir: Path = CHECKPOINT_DIR,
    deployment_summary_path: Path = DEPLOYMENT_SUMMARY_PATH,
    replace: bool = False,
) -> dict[str, Any]:
    source_summary = json.loads(source_summary_path.read_text(encoding="utf-8"))
    manifest = load_manifest(config)
    if source_summary.get("dataset_hash") != manifest.get("dataset_hash"):
        raise RuntimeError("Profile probe summary and prepared dataset hashes differ.")
    if source_summary.get("config_hash") != config.config_hash:
        raise RuntimeError("Profile probe summary and active config hashes differ.")

    settings = ProbeSettings(**source_summary["settings"])
    train_trials = load_trials(config, "train")
    validation_trials = load_trials(config, "validation")
    constants = train_trials[0].constants
    for trial in (*train_trials, *validation_trials):
        if (
            trial.constants.acceleration_gain != constants.acceleration_gain
            or trial.constants.gravity_mps2 != constants.gravity_mps2
        ):
            raise RuntimeError("Deployment candidates require shared gamma and gravity.")

    train_store = WindowStore(train_trials, 33, config, manifest)
    validation_store = WindowStore(validation_trials, 33, config, manifest)
    validation_refs = _validation_references(validation_store, settings)
    results: dict[str, Any] = {}

    for name, switches in CANDIDATES.items():
        checkpoint_path = checkpoint_dir / f"{name}.pt"
        if checkpoint_path.exists() and not replace:
            raise FileExistsError(
                f"Checkpoint already exists: {checkpoint_path}; pass --replace to overwrite."
            )
        model = ProfileFrictionResidual(
            profile_enabled=switches["profile_enabled"],
            friction_enabled=switches["friction_enabled"],
            blackbox_enabled=False,
            kappa_enabled=False,
            acceleration_gain=constants.acceleration_gain,
            gravity_mps2=constants.gravity_mps2,
            settings=settings,
        )
        model, training = _fit(
            name,
            model,
            train_store,
            validation_store,
            validation_refs,
            config,
            settings,
        )
        model.eval()
        splits = _evaluate_candidate(model, config, manifest)
        payload = {
            "schema_version": 1,
            "candidate": name,
            "formula_version": FORMULA_VERSION,
            "input_names": ["pb", "vb", "theta", "omega", "a_nom"],
            "profile_enabled": switches["profile_enabled"],
            "friction_enabled": switches["friction_enabled"],
            "acceleration_gain": constants.acceleration_gain,
            "gravity_mps2": constants.gravity_mps2,
            "settings": settings.__dict__,
            "dataset_hash": manifest["dataset_hash"],
            "config_hash": config.config_hash,
            "state_dict": model.state_dict(),
            "training": training,
            "metrics": _compact_metrics(splits),
            "runtime": {
                "torch_version": torch.__version__,
                "numpy_version": np.__version__,
            },
        }
        _atomic_torch_save(payload, checkpoint_path)
        results[name] = {
            "checkpoint": str(checkpoint_path.resolve()),
            "checkpoint_sha256": _sha256(checkpoint_path),
            "profile_enabled": switches["profile_enabled"],
            "friction_enabled": switches["friction_enabled"],
            "best_validation_score": training["best_validation_score"],
            "epochs": training["epochs"],
            "metrics": payload["metrics"],
        }

    summary = {
        "schema_version": 1,
        "formula_version": FORMULA_VERSION,
        "training_performed": list(CANDIDATES),
        "other_candidates_retrained": False,
        "source_probe_summary": str(source_summary_path.resolve()),
        "source_probe_summary_sha256": _sha256(source_summary_path),
        "dataset_hash": manifest["dataset_hash"],
        "config_hash": config.config_hash,
        "settings": settings.__dict__,
        "runtime": {
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
        },
        "candidates": results,
    }
    deployment_summary_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = deployment_summary_path.with_suffix(
        deployment_summary_path.suffix + f".{os.getpid()}.tmp"
    )
    temporary.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, deployment_summary_path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--source-summary", type=Path, default=RESULT_PATH)
    parser.add_argument("--checkpoint-dir", type=Path, default=CHECKPOINT_DIR)
    parser.add_argument("--deployment-summary", type=Path, default=DEPLOYMENT_SUMMARY_PATH)
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()
    summary = run(
        load_config(args.config),
        source_summary_path=args.source_summary,
        checkpoint_dir=args.checkpoint_dir,
        deployment_summary_path=args.deployment_summary,
        replace=args.replace,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
