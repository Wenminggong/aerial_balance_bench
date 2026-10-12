"""Command-line entry point for the independent residual dynamics v1.5 workflow."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from .config import AppConfig, load_config
from .data import prepare_datasets
from .evaluation import evaluate_model, test_model
from .reporting import update_report
from .training import train_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Residual dynamics v1.5 offline pipeline")
    parser.add_argument("command", choices=("prepare", "train", "evaluate", "test", "report", "all"))
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run-dir", type=Path, default=None,
                        help="Training run directory; defaults to runs/training/latest.json")
    parser.add_argument("--wandb-mode", choices=("auto", "online", "offline", "disabled"),
                        default=None, help="Override W&B mode without changing the config hash")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    if args.wandb_mode is not None:
        config = replace(config, wandb=replace(
            config.wandb, mode=args.wandb_mode,
            enabled=args.wandb_mode != "disabled",
        ))
    result = _run(args.command, config, args.run_dir)
    print(json.dumps(_json_safe(result), indent=2, sort_keys=True))
    return 0


def _run(command: str, config: AppConfig, run_dir: Path | None):
    if command == "prepare":
        return prepare_datasets(config)
    if command == "train":
        return {"run_dir": str(train_model(config))}
    if command == "evaluate":
        return evaluate_model(config, run_dir)
    if command == "test":
        return test_model(config, run_dir)
    if command == "report":
        return {"report_path": str(update_report(config))}
    prepare_datasets(config)
    trained = train_model(config)
    validation = evaluate_model(config, trained)
    final = test_model(config, trained)
    report = update_report(config)
    return {
        "run_dir": str(trained), "validation_scope": validation["scope"],
        "acceptance": final["acceptance"], "report_path": str(report),
    }


def _json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    return value


if __name__ == "__main__":
    raise SystemExit(main())
