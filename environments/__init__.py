"""Simulation environments for Aerial-Balance-Bench."""

__all__ = ["AerialBalanceEnv", "AerialBalanceEnvCfg", "ReferencePreviewCfg", "build_observation_fields"]


def __getattr__(name: str):
    if name in {"ReferencePreviewCfg", "build_observation_fields"}:
        from .observation_schema import ReferencePreviewCfg, build_observation_fields

        return {
            "ReferencePreviewCfg": ReferencePreviewCfg,
            "build_observation_fields": build_observation_fields,
        }[name]
    if name in {"AerialBalanceEnv", "AerialBalanceEnvCfg"}:
        from .aerial_balance_env import AerialBalanceEnv, AerialBalanceEnvCfg

        return {
            "AerialBalanceEnv": AerialBalanceEnv,
            "AerialBalanceEnvCfg": AerialBalanceEnvCfg,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
