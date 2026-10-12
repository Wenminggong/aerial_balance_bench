"""Command-line entry point for the residual-dynamics workflow."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from .config import AppConfig, load_config, with_training_overrides
from .data import prepare_dataset
from .evaluation import evaluate_model
from .reporting import generate_report
from .screening import run_screening
from .training import resolve_run_dir, train_model


DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "mlp_5input.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit real-world ball acceleration residuals.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "train", "evaluate", "report", "all", "screen"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
        subparser.add_argument(
            "--wandb-mode",
            choices=("auto", "online", "offline", "disabled"),
            default=None,
        )
        if command in {"evaluate", "report"}:
            subparser.add_argument("--run-dir", type=Path, default=None)
        if command in {"train", "evaluate", "all"}:
            subparser.add_argument("--seed", type=int, default=None)
            subparser.add_argument("--jacobian-weight", type=float, default=None)
            subparser.add_argument("--jacobian-limit", type=float, default=None)
            subparser.add_argument("--theta-channel-weight", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.wandb_mode is not None:
        config = replace(config, wandb=replace(config.wandb, mode=args.wandb_mode))
    if args.command in {"train", "evaluate", "all"}:
        config = with_training_overrides(
            config,
            seed=args.seed,
            jacobian_weight=args.jacobian_weight,
            jacobian_norm_limit_mps2=args.jacobian_limit,
            theta_channel_weight=args.theta_channel_weight,
        )

    if args.command == "prepare":
        manifest = prepare_dataset(config)
        _print({"dataset_hash": manifest["dataset_hash"], **manifest["split_counts"]})
        return
    if args.command == "train":
        run_dir = train_model(config)
        _print({"run_dir": str(run_dir)})
        return
    if args.command == "evaluate":
        summary = evaluate_model(config, args.run_dir, include_holdout=True)
        _print({"run_name": summary["run_name"], "acceptance": summary["acceptance"]})
        return
    if args.command == "report":
        if args.run_dir is not None:
            resolve_run_dir(config, args.run_dir)
        report = generate_report(config)
        _print({"report": str(report)})
        return
    if args.command == "screen":
        result = run_screening(config)
        _print(
            {
                "selected_candidate": result["selected_candidate"],
                "selected_run_dir": result["selected_run_dir"],
                "seed_stability_passed": result["seed_stability_passed"],
            }
        )
        return

    manifest = prepare_dataset(config)
    run_dir = train_model(config)
    summary = evaluate_model(config, run_dir, include_holdout=True)
    report = generate_report(config)
    _print(
        {
            "dataset_hash": manifest["dataset_hash"],
            "run_dir": str(run_dir),
            "acceptance": summary["acceptance"],
            "report": str(report),
        }
    )


def _print(value: object) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
