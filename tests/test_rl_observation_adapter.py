"""Unit tests for RL raw-observation adaptation."""

from __future__ import annotations

import pytest
import torch
from gymnasium import Env, spaces

from baselines.model_state_predictor import VelocityModelStatePredictorCfg
from baselines.rl_env_wrapper import NormalizedRLTrainingWrapper
from baselines.rl_observation_adapter import RLObservationAdapter, RLObservationAdapterCfg
from baselines.rl_policy import RLPolicy, RLPolicyCfg


def _adapter(
    mode: str,
    *,
    num_envs: int = 2,
    raw_dim: int | None = None,
    raw_fields: tuple[str, ...] | None = None,
) -> RLObservationAdapter:
    return RLObservationAdapter(
        RLObservationAdapterCfg(observation_mode=mode),
        num_envs=num_envs,
        device="cpu",
        raw_observation_dim=raw_dim,
        raw_observation_fields=raw_fields,
    )


def test_default_spec_preserves_legacy_full11_contract():
    adapter = _adapter("full11")
    observation = torch.arange(22, dtype=torch.float32).reshape(2, 11)

    policy_input = adapter.transform(observation)

    assert adapter.raw_observation_dim == 11
    assert adapter.input_dim == 11
    assert adapter.field_names == adapter.FULL11_FIELDS
    torch.testing.assert_close(policy_input, observation)


@pytest.mark.parametrize("mode", ["legacy8", "full11"])
def test_legacy_modes_ignore_appended_preview_fields(mode: str):
    legacy_observation = torch.tensor(
        [
            [0.2, 0.1, 0.0, 0.3, 0.4, 0.0, 0.5, 0.6, 0.0, 0.15, -0.02],
            [0.4, 0.2, 0.1, 0.5, 0.6, 0.1, 0.7, 0.8, 0.1, 0.30, 0.03],
        ],
        dtype=torch.float32,
    )
    extended_observation = torch.cat(
        (legacy_observation, torch.full((2, 11), 123.0)),
        dim=-1,
    )
    legacy_adapter = _adapter(mode)
    extended_adapter = _adapter(mode, raw_dim=22)

    expected = legacy_adapter.transform(legacy_observation)
    actual = extended_adapter.transform(extended_observation)

    torch.testing.assert_close(actual, expected)
    assert actual.shape[-1] == (8 if mode == "legacy8" else 11)


def test_reference_preview_horizon_zero_reorders_current_reference():
    adapter = _adapter("reference_preview", raw_dim=12)
    raw_observation = torch.arange(24, dtype=torch.float32).reshape(2, 12)

    policy_input = adapter.transform(raw_observation)
    expected = torch.cat(
        (
            raw_observation[:, :9],
            raw_observation[:, 10:11],
            raw_observation[:, 9:10],
            raw_observation[:, 11:12],
        ),
        dim=-1,
    )

    torch.testing.assert_close(policy_input, expected)
    assert adapter.input_dim == 12
    assert adapter.reference_preview_future_steps == 0
    assert adapter.preview_offsets == (0,)
    assert adapter.field_names[-3:] == ("a_prev", "pg_0", "vg_0")


def test_reference_preview_multiple_steps_preserves_interleaved_pairs():
    raw_fields = (
        *RLObservationAdapter.FULL11_FIELDS,
        "vg_0",
        "pg_1",
        "vg_1",
        "pg_2",
        "vg_2",
    )
    adapter = _adapter("reference_preview", raw_fields=raw_fields)
    raw_observation = torch.arange(32, dtype=torch.float32).reshape(2, 16)

    policy_input = adapter.transform(raw_observation)

    torch.testing.assert_close(policy_input[:, :9], raw_observation[:, :9])
    torch.testing.assert_close(policy_input[:, 9:10], raw_observation[:, 10:11])
    torch.testing.assert_close(policy_input[:, 10:11], raw_observation[:, 9:10])
    torch.testing.assert_close(policy_input[:, 11:], raw_observation[:, 11:])
    assert adapter.raw_observation_dim == 16
    assert adapter.reference_preview_future_steps == 2
    assert adapter.preview_offsets == (0, 1, 2)
    assert adapter.field_names == (
        "pb",
        "vb",
        "ab",
        "theta",
        "omega",
        "alpha",
        "drz",
        "vrz",
        "arz",
        "a_prev",
        "pg_0",
        "vg_0",
        "pg_1",
        "vg_1",
        "pg_2",
        "vg_2",
    )


def test_reference_preview_accepts_explicit_pg_zero_field_alias():
    raw_fields = list(RLObservationAdapter.FULL11_FIELDS)
    raw_fields[9] = "pg_0"
    raw_fields.append("vg_0")

    adapter = _adapter("reference_preview", raw_fields=tuple(raw_fields))

    assert adapter.raw_observation_dim == 12


@pytest.mark.parametrize("raw_dim", [11, 13, 15])
def test_reference_preview_rejects_dimensions_outside_formula(raw_dim: int):
    with pytest.raises(ValueError, match=r"12 \+ 2 \* future_steps"):
        _adapter("reference_preview", raw_dim=raw_dim)


def test_reference_preview_rejects_incorrect_field_order():
    raw_fields = (*RLObservationAdapter.FULL11_FIELDS, "pg_1")

    with pytest.raises(ValueError, match="raw_observation_fields must follow"):
        _adapter("reference_preview", raw_fields=raw_fields)


def test_reference_preview_rejects_runtime_horizon_mismatch():
    adapter = _adapter("reference_preview", raw_dim=12)

    with pytest.raises(ValueError, match="does not match its raw observation spec"):
        adapter.transform(torch.zeros((2, 14)))


def test_legacy_mode_runtime_shape_requires_only_legacy_prefix():
    adapter = _adapter("legacy8")

    output = adapter.transform(torch.zeros((2, 22)))

    assert output.shape == (2, 8)


class _RawPreviewEnv(Env):
    num_envs = 2
    device = "cpu"
    max_episode_length = 100
    observation_fields = (
        *RLObservationAdapter.FULL11_FIELDS,
        "vg_0",
        "pg_1",
        "vg_1",
    )
    raw_observation_dim = len(observation_fields)
    observation_space = spaces.Dict(
        {
            "policy": spaces.Box(
                low=-float("inf"),
                high=float("inf"),
                shape=(raw_observation_dim,),
            )
        }
    )
    action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,))


def test_training_wrapper_infers_raw_spec_from_environment():
    wrapper = NormalizedRLTrainingWrapper(
        _RawPreviewEnv(),
        RLObservationAdapterCfg(observation_mode="reference_preview"),
        physical_action_limit=0.5,
    )

    assert wrapper.raw_observation_dim == 14
    assert wrapper.raw_observation_fields == _RawPreviewEnv.observation_fields
    assert wrapper.policy_observation_dim == 14
    assert wrapper.adapter.reference_preview_future_steps == 1
    assert wrapper.single_observation_space["policy"].shape == (14,)


def test_training_wrapper_forwards_scalar_unified_benchmark_metrics():
    wrapper = NormalizedRLTrainingWrapper(
        _RawPreviewEnv(),
        RLObservationAdapterCfg(observation_mode="reference_preview"),
        physical_action_limit=0.5,
    )
    infos = {
        "benchmark": {
            "mean_absolute_error": torch.tensor(0.1),
            "constant_success_rate": torch.tensor(0.75),
            "per_environment_debug": torch.tensor([1.0, 2.0]),
        }
    }

    wrapper._copy_benchmark_metrics_to_episode_info(infos)

    assert infos["episode"]["mean_absolute_error"].item() == pytest.approx(0.1)
    assert infos["episode"]["constant_success_rate"].item() == pytest.approx(0.75)
    assert "per_environment_debug" not in infos["episode"]


def test_policy_rejects_reference_preview_with_active_predictor_before_model_build():
    cfg = RLPolicyCfg(
        observation_mode="reference_preview",
        load_checkpoint=False,
        state_predictor=VelocityModelStatePredictorCfg(enabled=True, delay_step=1),
    )

    with pytest.raises(ValueError, match="cannot be combined with an active state predictor"):
        RLPolicy(
            cfg,
            num_envs=2,
            device="cpu",
            step_dt=1.0 / 60.0,
            physical_action_limit=0.01,
            raw_observation_dim=12,
        )
