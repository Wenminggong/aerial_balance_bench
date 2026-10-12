"""Train and append only the bounded profile + friction + gain candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from .config import Config, load_config
from .data import load_manifest, load_trials
from .evaluation import _integral_metrics, _refs_for_evaluation, _rollout_metrics
from .profile_friction_probe import (
    DEFAULT_CONFIG,
    RESULT_PATH,
    ProbeSettings,
    ProfileFrictionResidual,
    _fit,
    _parameter_diagnostics,
    _point,
)
from .training import WindowStore


CANDIDATE_NAME = "profile_plus_friction_plus_kappa"
CANDIDATE_PATH = RESULT_PATH.parent / "profile_friction_kappa_candidate.json"
CHECKPOINT_PATH = RESULT_PATH.parent / "profile_friction_kappa_candidate.pt"


def _stable_hash(value: Any) -> str:
    serialized = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _evaluate_candidate(
    model: ProfileFrictionResidual,
    config: Config,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for split in ("validation", "test", "stress"):
        trials = load_trials(config, split)
        store = WindowStore(trials, 60, config, manifest)
        refs = _refs_for_evaluation(store)
        integral_raw = _integral_metrics(model, store, refs, config)["overall"]
        rollout_raw = _rollout_metrics(model, None, store, refs, config)["overall"]
        result[split] = {
            "point": _point(model, trials),
            "integral": {
                key: {"rmse_mps2": value["rbf_rmse_mps2"]}
                for key, value in integral_raw.items()
            },
            "rollout": {key: value["rbf"] for key, value in rollout_raw.items()},
        }
    return result


def run(
    config: Config,
    summary_path: Path = RESULT_PATH,
    candidate_path: Path = CANDIDATE_PATH,
    checkpoint_path: Path = CHECKPOINT_PATH,
    *,
    replace: bool = False,
) -> dict[str, Any]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if CANDIDATE_NAME in summary.get("training", {}) and not replace:
        raise RuntimeError(
            f"{CANDIDATE_NAME} already exists; pass --replace to train it again."
        )
    manifest = load_manifest(config)
    if summary["dataset_hash"] != manifest["dataset_hash"]:
        raise RuntimeError("Probe summary and prepared dataset hashes differ.")

    untouched_names = sorted(
        set(summary["training"]) | set(summary["parameters"])
        | {
            name
            for split in summary["splits"].values()
            for name in split["models"]
        }
    )
    untouched_before = {
        "training": {
            name: summary["training"][name]
            for name in untouched_names
            if name in summary["training"] and name != CANDIDATE_NAME
        },
        "parameters": {
            name: summary["parameters"][name]
            for name in untouched_names
            if name in summary["parameters"] and name != CANDIDATE_NAME
        },
        "splits": {
            split: {
                name: values["models"][name]
                for name in untouched_names
                if name in values["models"] and name != CANDIDATE_NAME
            }
            for split, values in summary["splits"].items()
        },
    }
    untouched_hash = _stable_hash(untouched_before)

    settings_dict = dict(summary["settings"])
    settings_dict.setdefault("kappa_limit", 0.10)
    settings = ProbeSettings(**settings_dict)
    if not np.isclose(settings.kappa_limit, 0.10):
        raise RuntimeError("This comparison is fixed to |kappa| <= 0.10.")
    train_trials = load_trials(config, "train")
    validation_trials = load_trials(config, "validation")
    constants = train_trials[0].constants
    train_store = WindowStore(train_trials, 33, config, manifest)
    validation_store = WindowStore(validation_trials, 33, config, manifest)
    validation_refs = list(validation_store.references[::5])
    if len(validation_refs) > settings.validation_windows:
        indices = np.linspace(
            0, len(validation_refs) - 1, settings.validation_windows, dtype=int
        )
        validation_refs = [validation_refs[index] for index in indices]

    model = ProfileFrictionResidual(
        profile_enabled=True,
        friction_enabled=True,
        blackbox_enabled=False,
        kappa_enabled=True,
        acceleration_gain=constants.acceleration_gain,
        gravity_mps2=constants.gravity_mps2,
        settings=settings,
    )
    model, training = _fit(
        CANDIDATE_NAME,
        model,
        train_store,
        validation_store,
        validation_refs,
        config,
        settings,
    )
    model.eval()
    parameters = _parameter_diagnostics(model)
    splits = _evaluate_candidate(model, config, manifest)
    candidate = {
        "schema_version": 1,
        "candidate": CANDIDATE_NAME,
        "formula": "r_profile + r_friction + kappa(pb,vb) * a_nom",
        "kappa_limit": settings.kappa_limit,
        "training_performed": [CANDIDATE_NAME],
        "other_candidates_retrained": False,
        "dataset_hash": manifest["dataset_hash"],
        "config_hash": config.config_hash,
        "untouched_candidates_sha256": untouched_hash,
        "training": training,
        "parameters": parameters,
        "splits": splits,
    }

    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    candidate_path.write_text(
        json.dumps(candidate, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    torch.save(
        {
            "candidate": CANDIDATE_NAME,
            "state_dict": model.state_dict(),
            "settings": settings.__dict__,
            "dataset_hash": manifest["dataset_hash"],
            "config_hash": config.config_hash,
            "training": training,
        },
        checkpoint_path,
    )

    summary["settings"] = settings.__dict__
    summary["purpose"] = (
        "coarse profile/friction/black-box/bounded-gain structure probe; not deployable"
    )
    summary["training"][CANDIDATE_NAME] = training
    summary["parameters"][CANDIDATE_NAME] = parameters
    for split, values in splits.items():
        summary["splits"][split]["models"][CANDIDATE_NAME] = values
    untouched_after = {
        "training": {
            name: summary["training"][name]
            for name in untouched_before["training"]
        },
        "parameters": {
            name: summary["parameters"][name]
            for name in untouched_before["parameters"]
        },
        "splits": {
            split: {
                name: summary["splits"][split]["models"][name]
                for name in values
            }
            for split, values in untouched_before["splits"].items()
        },
    }
    if _stable_hash(untouched_after) != untouched_hash:
        raise RuntimeError("An existing candidate changed while appending kappa results.")
    summary["incremental_kappa_comparison"] = {
        "candidate_result": str(candidate_path),
        "checkpoint": str(checkpoint_path),
        "other_candidates_retrained": False,
        "untouched_candidates_sha256": untouched_hash,
    }
    temporary = summary_path.with_suffix(summary_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(summary_path)
    return candidate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--summary", type=Path, default=RESULT_PATH)
    parser.add_argument("--candidate-result", type=Path, default=CANDIDATE_PATH)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_PATH)
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()
    result = run(
        load_config(args.config),
        args.summary,
        args.candidate_result,
        args.checkpoint,
        replace=args.replace,
    )
    concise = {
        "candidate": CANDIDATE_NAME,
        "other_candidates_retrained": False,
        "best_validation_score": result["training"]["best_validation_score"],
        "kappa_abs_max": result["parameters"]["coefficient_bounds"][
            "kappa_abs_max"
        ],
        "splits": {
            split: {
                "point_rmse": values["point"]["rmse"],
                "h12_integral_rmse": values["integral"]["h12"]["rmse_mps2"],
                "h33_pb_rmse": values["rollout"]["h33"]["pb_rmse_m"],
                "h33_vb_rmse": values["rollout"]["h33"]["vb_rmse_mps"],
            }
            for split, values in result["splits"].items()
        },
    }
    print(json.dumps(concise, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
