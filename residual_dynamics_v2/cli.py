"""CLI for preparing, fitting, evaluating, and documenting RBF-PV residuals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config
from .data import prepare_dataset
from .evaluation import evaluate_model
from .reporting import generate_report
from .training import train_model


DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "rbf_2input_causal_success.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit the two-input RBF acceleration residual.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "train", "evaluate", "report", "all"):
        child = subparsers.add_parser(command)
        child.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
        if command in {"train", "all"}:
            child.add_argument(
                "--wandb-mode", choices=("disabled",), default="disabled"
            )
        if command in {"evaluate", "report"}:
            child.add_argument("--run-dir", type=Path, default=None)
        if command == "evaluate":
            child.add_argument("--include-holdout", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.command == "prepare":
        manifest = prepare_dataset(config)
        _print({"dataset_hash": manifest["dataset_hash"], "split_counts": manifest["split_counts"]})
    elif args.command == "train":
        _print({"run_dir": str(train_model(config))})
    elif args.command == "evaluate":
        summary = evaluate_model(config, args.run_dir, include_holdout=args.include_holdout)
        _print(
            {
                "eligible_for_residual_integration": summary["eligible_for_residual_integration"],
                "eligible_for_real_world_deployment": summary["eligible_for_real_world_deployment"],
            }
        )
    elif args.command == "report":
        _print({"report": str(generate_report(config, args.run_dir))})
    else:
        manifest = prepare_dataset(config)
        run_dir = train_model(config)
        summary = evaluate_model(config, run_dir, include_holdout=True)
        report = generate_report(config, run_dir)
        _print(
            {
                "dataset_hash": manifest["dataset_hash"],
                "run_dir": str(run_dir),
                "eligible_for_residual_integration": summary["eligible_for_residual_integration"],
                "report": str(report),
            }
        )


def _print(value: object) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
