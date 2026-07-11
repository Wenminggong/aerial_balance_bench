"""Raw policy-observation layout helpers."""

from __future__ import annotations

from omni.isaac.lab.utils import configclass


LEGACY_OBSERVATION_FIELDS = (
    "pb",
    "vb",
    "ab",
    "theta",
    "omega",
    "alpha",
    "drz",
    "vrz",
    "arz",
    "pg",
    "a_prev",
)


@configclass
class ReferencePreviewCfg:
    """Configuration for reference position/velocity preview observations."""

    enabled: bool = False
    future_steps: int = 0


def build_observation_fields(preview_cfg: ReferencePreviewCfg) -> tuple[str, ...]:
    """Return the raw policy-observation field names for a preview configuration."""
    fields = list(LEGACY_OBSERVATION_FIELDS)
    if preview_cfg.future_steps < 0:
        raise ValueError("reference_preview.future_steps must be non-negative.")
    if not preview_cfg.enabled:
        return tuple(fields)
    fields.append("vg_0")
    for offset in range(1, preview_cfg.future_steps + 1):
        fields.extend((f"pg_{offset}", f"vg_{offset}"))
    return tuple(fields)
