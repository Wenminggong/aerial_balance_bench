"""Robustness hooks for Aerial-Balance-Bench."""

__all__ = [
    "CommandDelayQueue",
    "PerEnvCommandDelayQueue",
    "RobustnessManager",
    "RobustnessManagerCfg",
    "VelocityResponseModel",
    "HumanBehaviorModel",
    "HUMAN_STEP_EXTRA_FIELDS",
]


def __getattr__(name: str):
    if name in {"HumanBehaviorModel", "HUMAN_STEP_EXTRA_FIELDS"}:
        from .human_behavior import HUMAN_STEP_EXTRA_FIELDS, HumanBehaviorModel

        return {"HumanBehaviorModel": HumanBehaviorModel, "HUMAN_STEP_EXTRA_FIELDS": HUMAN_STEP_EXTRA_FIELDS}[name]
    if name == "VelocityResponseModel":
        from .velocity_response import VelocityResponseModel

        return VelocityResponseModel
    if name in {
        "CommandDelayQueue",
        "PerEnvCommandDelayQueue",
        "RobustnessManager",
        "RobustnessManagerCfg",
    }:
        from .robustness_manager import (
            CommandDelayQueue,
            PerEnvCommandDelayQueue,
            RobustnessManager,
            RobustnessManagerCfg,
        )

        return {
            "CommandDelayQueue": CommandDelayQueue,
            "PerEnvCommandDelayQueue": PerEnvCommandDelayQueue,
            "RobustnessManager": RobustnessManager,
            "RobustnessManagerCfg": RobustnessManagerCfg,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
