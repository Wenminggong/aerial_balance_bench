from __future__ import annotations

import pytest

from environments.observation_schema import ReferencePreviewCfg, build_observation_fields


def test_disabled_preview_preserves_legacy_11_fields():
    fields = build_observation_fields(ReferencePreviewCfg(enabled=False, future_steps=0))

    assert fields == (
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


def test_preview_fields_keep_legacy_prefix_and_append_pairs():
    fields = build_observation_fields(ReferencePreviewCfg(enabled=True, future_steps=2))

    assert fields[:11] == build_observation_fields(ReferencePreviewCfg())
    assert fields[11:] == ("vg_0", "pg_1", "vg_1", "pg_2", "vg_2")
    assert len(fields) == 16


def test_negative_preview_horizon_is_invalid_even_when_disabled():
    with pytest.raises(ValueError, match="non-negative"):
        build_observation_fields(ReferencePreviewCfg(enabled=False, future_steps=-1))
