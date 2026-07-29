"""Robustness hooks for Aerial-Balance-Bench."""

__all__ = [
    "CommandDelayQueue",
    "PerEnvCommandDelayQueue",
    "RobustnessManager",
    "RobustnessManagerCfg",
    "VelocityResponseModel",
]


def __getattr__(name: str):
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
