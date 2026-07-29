"""Unit tests for RL raw-observation adaptation."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml
from gymnasium import Env, spaces

from baselines.base_policy import ObservationIndex
from baselines.model_state_predictor import VelocityModelStatePredictorCfg
from baselines.rl_env_wrapper import NormalizedRLTrainingWrapper
from baselines.rl_observation_adapter import RLObservationAdapter, RLObservationAdapterCfg
from baselines.rl_policy import (
    RLPolicy,
    RLPolicyCfg,
    validate_rl_predictor_environment_contract,
)


def _adapter(
    mode: str,
    *,
    num_envs: int = 2,
    raw_dim: int | None = None,
    raw_fields: tuple[str, ...] | None = None,
    samples: int | None = None,
    base_offset: int = 0,
) -> RLObservationAdapter:
    return RLObservationAdapter(
        RLObservationAdapterCfg(
            observation_mode=mode,
            reference_preview_samples=samples,
        ),
        num_envs=num_envs,
        device="cpu",
        raw_observation_dim=raw_dim,
        raw_observation_fields=raw_fields,
        reference_preview_base_offset=base_offset,
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


def test_relative_reference_preview_samples_uniform_offsets_and_builds_expected_input():
    adapter = _adapter(
        "relative_reference_preview",
        num_envs=1,
        raw_dim=52,
        samples=10,
    )
    raw_observation = torch.zeros((1, 52), dtype=torch.float32)
    raw_observation[0, ObservationIndex.PB] = 0.60
    raw_observation[0, ObservationIndex.VB] = 0.40
    raw_observation[0, ObservationIndex.AB] = 0.30
    raw_observation[0, ObservationIndex.THETA] = 0.20
    raw_observation[0, ObservationIndex.OMEGA] = 0.10
    raw_observation[0, ObservationIndex.ALPHA] = -0.10
    raw_observation[0, ObservationIndex.DRZ] = 123.0
    raw_observation[0, ObservationIndex.VRZ] = -0.20
    raw_observation[0, ObservationIndex.ARZ] = -0.30
    raw_observation[0, ObservationIndex.PG] = 0.25
    raw_observation[0, ObservationIndex.A_PREV] = -0.05
    raw_observation[0, adapter.raw_observation_fields.index("vg_0")] = 0.10
    for offset in range(1, 21):
        pg_index = adapter.raw_observation_fields.index(f"pg_{offset}")
        vg_index = adapter.raw_observation_fields.index(f"vg_{offset}")
        raw_observation[0, pg_index] = 0.25 + 0.01 * offset
        raw_observation[0, vg_index] = 0.10 + 0.02 * offset

    policy_input = adapter.transform(raw_observation)
    sampled_offsets = tuple(range(2, 21, 2))
    expected = torch.tensor(
        [
            0.35,
            0.30,
            0.30,
            0.20,
            0.10,
            -0.10,
            -0.20,
            -0.30,
            -0.05,
            *(0.01 * offset for offset in sampled_offsets),
            *(0.02 * offset for offset in sampled_offsets),
        ],
        dtype=torch.float32,
    ).unsqueeze(0)

    torch.testing.assert_close(policy_input, expected)
    assert adapter.reference_preview_future_steps == 20
    assert adapter.effective_reference_preview_future_steps == 20
    assert adapter.reference_preview_base_offset == 0
    assert adapter.reference_preview_samples == 10
    assert adapter.preview_offsets == sampled_offsets
    assert adapter.preview_source_offsets == sampled_offsets
    assert adapter.sampled_reference_preview_offsets[-1] == 20
    assert adapter.input_dim == 29
    assert adapter.field_names == (
        *adapter.RELATIVE_REFERENCE_PREVIEW_BASE_FIELDS,
        *(f"delta_pg_{offset}" for offset in sampled_offsets),
        *(f"delta_vg_{offset}" for offset in sampled_offsets),
    )


def test_relative_reference_preview_samples_every_future_point_when_h_equals_k():
    adapter = _adapter(
        "relative_reference_preview",
        num_envs=1,
        raw_dim=18,
        samples=3,
    )

    assert adapter.reference_preview_future_steps == 3
    assert adapter.preview_offsets == (1, 2, 3)
    assert adapter.input_dim == 15


def test_relative_reference_preview_rebases_delayed_prediction_and_samples_raw_sources():
    training_adapter = _adapter(
        "relative_reference_preview",
        num_envs=1,
        raw_dim=72,
        samples=10,
    )
    adapter = _adapter(
        "relative_reference_preview",
        num_envs=1,
        raw_dim=88,
        samples=10,
        base_offset=8,
    )
    raw_observation = torch.zeros((1, 88), dtype=torch.float32)
    raw_observation[0, ObservationIndex.PB] = 2.0
    raw_observation[0, ObservationIndex.VB] = 3.0
    raw_observation[0, ObservationIndex.AB] = 0.3
    raw_observation[0, ObservationIndex.THETA] = 0.2
    raw_observation[0, ObservationIndex.OMEGA] = 0.1
    raw_observation[0, ObservationIndex.ALPHA] = -0.1
    raw_observation[0, ObservationIndex.VRZ] = -0.2
    raw_observation[0, ObservationIndex.ARZ] = -0.3
    # RLPolicy replaces legacy PG with the predictor's pg_D before adaptation.
    raw_observation[0, ObservationIndex.PG] = 1.8
    raw_observation[0, ObservationIndex.A_PREV] = -0.05
    raw_observation[0, adapter.raw_observation_fields.index("vg_0")] = 2.0
    for offset in range(1, 39):
        raw_observation[0, adapter.raw_observation_fields.index(f"pg_{offset}")] = (
            1.0 + 0.1 * offset
        )
        raw_observation[0, adapter.raw_observation_fields.index(f"vg_{offset}")] = (
            2.0 + 0.01 * offset
        )

    policy_input = adapter.transform(raw_observation)
    relative_offsets = tuple(range(3, 31, 3))
    source_offsets = tuple(range(11, 39, 3))
    expected = torch.tensor(
        [
            0.2,
            0.92,
            0.3,
            0.2,
            0.1,
            -0.1,
            -0.2,
            -0.3,
            -0.05,
            *(0.1 * offset for offset in relative_offsets),
            *(0.01 * offset for offset in relative_offsets),
        ],
        dtype=torch.float32,
    ).unsqueeze(0)

    torch.testing.assert_close(policy_input, expected)
    assert adapter.reference_preview_future_steps == 38
    assert adapter.reference_preview_base_offset == 8
    assert adapter.effective_reference_preview_future_steps == 30
    assert adapter.preview_offsets == relative_offsets
    assert adapter.preview_source_offsets == source_offsets
    assert adapter.preview_source_offsets[-1] == 38
    assert adapter.input_dim == 29
    assert training_adapter.reference_preview_future_steps == 30
    assert training_adapter.preview_offsets == relative_offsets
    assert training_adapter.preview_source_offsets == relative_offsets
    assert training_adapter.input_dim == adapter.input_dim
    assert training_adapter.field_names == adapter.field_names
    assert adapter.field_names == (
        *adapter.RELATIVE_REFERENCE_PREVIEW_BASE_FIELDS,
        *(f"delta_pg_{offset}" for offset in relative_offsets),
        *(f"delta_vg_{offset}" for offset in relative_offsets),
    )


@pytest.mark.parametrize(
    ("raw_dim", "samples", "exception", "match"),
    [
        (52, None, ValueError, "reference_preview_samples=K"),
        (52, 0, ValueError, "K=0"),
        (52, -1, ValueError, "K=-1"),
        (52, True, TypeError, "not bool"),
        (52, 2.5, TypeError, "positive integer"),
        (16, 3, ValueError, "H_effective=2, K=3"),
        (22, 2, ValueError, "H_effective=5, K=2"),
    ],
)
def test_relative_reference_preview_rejects_invalid_sampling_contract(
    raw_dim: int,
    samples,
    exception: type[Exception],
    match: str,
):
    with pytest.raises(exception, match=match):
        _adapter(
            "relative_reference_preview",
            raw_dim=raw_dim,
            samples=samples,
        )


@pytest.mark.parametrize(
    ("base_offset", "exception", "match"),
    [
        (True, TypeError, "not bool"),
        (-1, ValueError, "non-negative"),
        (21, ValueError, "D=21, H_raw=20"),
    ],
)
def test_relative_reference_preview_rejects_invalid_base_offset(
    base_offset,
    exception: type[Exception],
    match: str,
):
    with pytest.raises(exception, match=match):
        _adapter(
            "relative_reference_preview",
            raw_dim=52,
            samples=10,
            base_offset=base_offset,
        )


@pytest.mark.parametrize(
    ("raw_dim", "samples", "base_offset", "match"),
    [
        (32, 3, 8, "H_effective=2, K=3"),
        (88, 8, 8, "H_effective=30, K=8"),
    ],
)
def test_relative_reference_preview_rejects_invalid_effective_horizon(
    raw_dim: int,
    samples: int,
    base_offset: int,
    match: str,
):
    with pytest.raises(ValueError, match=match):
        _adapter(
            "relative_reference_preview",
            raw_dim=raw_dim,
            samples=samples,
            base_offset=base_offset,
        )


def test_non_relative_mode_rejects_reference_preview_base_offset():
    with pytest.raises(ValueError, match="only supported"):
        _adapter("reference_preview", raw_dim=12, base_offset=1)


def test_relative_reference_preview_rejects_missing_delayed_source_field():
    raw_fields = list(RLObservationAdapter._default_raw_observation_fields(88))
    raw_fields[raw_fields.index("pg_38")] = "missing_pg_38"

    with pytest.raises(ValueError, match="raw_observation_fields must follow"):
        _adapter(
            "relative_reference_preview",
            raw_fields=tuple(raw_fields),
            samples=10,
            base_offset=8,
        )


def test_relative_reference_preview_rejects_runtime_horizon_mismatch():
    adapter = _adapter(
        "relative_reference_preview",
        raw_dim=16,
        samples=2,
    )

    with pytest.raises(ValueError, match="does not match its raw observation spec"):
        adapter.transform(torch.zeros((2, 18)))


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


def test_training_wrapper_exposes_relative_reference_preview_space():
    wrapper = NormalizedRLTrainingWrapper(
        _RawPreviewEnv(),
        RLObservationAdapterCfg(
            observation_mode="relative_reference_preview",
            reference_preview_samples=1,
        ),
        physical_action_limit=0.5,
    )

    assert wrapper.policy_observation_dim == 11
    assert wrapper.adapter.preview_offsets == (1,)
    assert wrapper.single_observation_space["policy"].shape == (11,)


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


def test_policy_rejects_full_reference_preview_with_active_predictor_before_model_build():
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
            raw_observation_dim=14,
        )


@pytest.mark.parametrize("velocity_response_enabled", [False, True])
def test_policy_aligns_relative_preview_with_active_predictor(
    velocity_response_enabled: bool,
    monkeypatch: pytest.MonkeyPatch,
):
    raw_fields = RLObservationAdapter._default_raw_observation_fields(20)
    cfg = RLPolicyCfg(
        observation_mode="relative_reference_preview",
        reference_preview_samples=2,
        load_checkpoint=False,
        state_predictor=VelocityModelStatePredictorCfg(
            enabled=True,
            delay_step=2,
            solver="euler",
            step_dt=0.1,
            max_acc=1.0,
            velocity_response_enabled=velocity_response_enabled,
            velocity_response_tau_s=0.2,
        ),
    )
    policy = RLPolicy(
        cfg,
        num_envs=1,
        device="cpu",
        step_dt=0.1,
        physical_action_limit=0.1,
        raw_observation_fields=raw_fields,
    )
    captured: dict[str, torch.Tensor] = {}

    def _act(states: torch.Tensor, timestep: int, timesteps: int):
        del timestep, timesteps
        captured["states"] = states.clone()
        return torch.zeros((1, 1), device=states.device)

    monkeypatch.setattr(policy.agent, "act", _act)
    observation = torch.zeros((1, len(raw_fields)))
    observation[0, ObservationIndex.PB] = 0.35
    observation[0, ObservationIndex.PG] = 0.20
    observation[0, raw_fields.index("vg_0")] = 0.0
    positions = {1: 0.30, 2: 0.40, 3: 0.50, 4: 0.80}
    velocities = {1: 0.10, 2: 0.20, 3: 0.30, 4: 0.50}
    for offset in range(1, 5):
        observation[0, raw_fields.index(f"pg_{offset}")] = positions[offset]
        observation[0, raw_fields.index(f"vg_{offset}")] = velocities[offset]

    with torch.no_grad():
        policy.act({"policy": observation})

    assert policy.observation_adapter.reference_preview_future_steps == 4
    assert policy.observation_adapter.reference_preview_base_offset == 2
    assert policy.observation_adapter.effective_reference_preview_future_steps == 2
    assert policy.observation_adapter.preview_offsets == (1, 2)
    assert policy.observation_adapter.preview_source_offsets == (3, 4)
    torch.testing.assert_close(captured["states"][:, 9:11], torch.tensor([[0.10, 0.40]]))
    torch.testing.assert_close(captured["states"][:, 11:13], torch.tensor([[0.10, 0.30]]))
    assert torch.isfinite(captured["states"]).all()


def test_relative_reference_preview_configuration_parsing():
    data = {
        "observation_mode": "relative_reference_preview",
        "reference_preview_samples": 10,
    }

    adapter_cfg = RLObservationAdapterCfg.from_dict(data)
    policy_cfg = RLPolicyCfg.from_dict(data)

    assert adapter_cfg.observation_mode == "relative_reference_preview"
    assert adapter_cfg.reference_preview_samples == 10
    assert policy_cfg.observation_mode == "relative_reference_preview"
    assert policy_cfg.reference_preview_samples == 10


def test_predictor_environment_contract_accepts_aligned_relative_preview():
    validate_rl_predictor_environment_contract(
        observation_mode="relative_reference_preview",
        reference_preview_samples=10,
        predictor_enabled=True,
        predictor_delay_step=8,
        preview_enabled=True,
        preview_future_steps=38,
        robustness_enabled=True,
        action_delay_enabled=True,
        environment_delay_step=8,
    )

    # Fixed-target legacy observations need delay alignment but no preview.
    validate_rl_predictor_environment_contract(
        observation_mode="legacy8",
        reference_preview_samples=None,
        predictor_enabled=True,
        predictor_delay_step=8,
        preview_enabled=False,
        preview_future_steps=0,
        robustness_enabled=True,
        action_delay_enabled=True,
        environment_delay_step=8,
        moving_reference=False,
        context="RL target_position",
    )

    # A fixed nominal predictor horizon is allowed for per-environment random delay.
    validate_rl_predictor_environment_contract(
        observation_mode="relative_reference_preview",
        reference_preview_samples=10,
        predictor_enabled=True,
        predictor_delay_step=6,
        preview_enabled=True,
        preview_future_steps=36,
        robustness_enabled=True,
        action_delay_enabled=True,
        environment_delay_step=99,
        environment_delay_step_choices=(0, 4, 8),
    )

    # A delayed environment without compensation remains a supported baseline.
    validate_rl_predictor_environment_contract(
        observation_mode="relative_reference_preview",
        reference_preview_samples=10,
        predictor_enabled=False,
        predictor_delay_step=0,
        preview_enabled=True,
        preview_future_steps=30,
        robustness_enabled=True,
        action_delay_enabled=True,
        environment_delay_step=8,
    )


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"action_delay_enabled": False}, "active environment action delay"),
        ({"environment_delay_step": 7}, "must match robustness.delay_step"),
        ({"observation_mode": "reference_preview"}, "cannot be combined"),
        ({"preview_future_steps": 17}, "H_effective=9, K=10"),
        ({"preview_future_steps": 37}, "integer multiple of K"),
    ],
)
def test_predictor_environment_contract_rejects_misalignment(overrides: dict, match: str):
    values = {
        "observation_mode": "relative_reference_preview",
        "reference_preview_samples": 10,
        "predictor_enabled": True,
        "predictor_delay_step": 8,
        "preview_enabled": True,
        "preview_future_steps": 38,
        "robustness_enabled": True,
        "action_delay_enabled": True,
        "environment_delay_step": 8,
    }
    values.update(overrides)

    with pytest.raises(ValueError, match=match):
        validate_rl_predictor_environment_contract(**values)


def test_unified_rl_predictor_eval_configs_preserve_checkpoint_contract():
    project_root = Path(__file__).resolve().parents[1]
    config_root = project_root / "environments" / "configs"
    policy_path = (
        project_root
        / "baselines"
        / "configs"
        / "rl_rpo_relative_reference_preview_predictor_eval.yaml"
    )
    run_path = (
        project_root
        / "baselines"
        / "configs"
        / "rl_unified_tracking_rpo_predictor_eval.yaml"
    )
    policy_config = yaml.safe_load(policy_path.read_text(encoding="utf-8"))
    run_config = yaml.safe_load(run_path.read_text(encoding="utf-8"))

    assert policy_config["observation_mode"] == "relative_reference_preview"
    assert policy_config["reference_preview_samples"] == 10
    assert policy_config["state_predictor"]["enabled"] is True
    assert policy_config["state_predictor"]["delay_step"] == "auto"
    assert policy_config["state_predictor"]["velocity_response_enabled"] is True
    assert policy_config["state_predictor"]["velocity_response_tau_s"] == pytest.approx(0.139)
    assert policy_config["state_predictor"]["velocity_response_gain"] == pytest.approx(1.0)
    assert policy_config["state_predictor"]["velocity_response_bias"] == pytest.approx(-0.00055)
    assert run_config["policy_config"].endswith(
        "rl_rpo_relative_reference_preview_predictor_eval.yaml"
    )

    families = (
        "constant",
        "sine",
        "triangle",
        "trapezoid",
        "random_b_spline",
        "random_ramp_dwell",
    )
    for family in families:
        nominal_config = yaml.safe_load(
            (config_root / f"unified_tracking_{family}.yaml").read_text(encoding="utf-8")
        )
        delayed_config = yaml.safe_load(
            (config_root / f"unified_tracking_{family}_delay_d8_h38.yaml").read_text(
                encoding="utf-8"
            )
        )
        robustness = delayed_config["robustness"]

        assert nominal_config["reference_preview"]["future_steps"] == 30
        assert delayed_config["unified_tracking_task"]["trajectory_types"] == [family]
        assert delayed_config["reference_preview"] == {"enabled": True, "future_steps": 38}
        assert robustness["enabled"] is True
        assert robustness["action_delay_enabled"] is True
        assert robustness["delay_step"] == 8
        assert robustness["velocity_response_enabled"] is False


@pytest.mark.parametrize("mode", ["legacy8", "full11"])
def test_policy_legacy_modes_pass_future_reference_to_active_predictor(
    mode: str,
    monkeypatch: pytest.MonkeyPatch,
):
    raw_fields = (
        *RLObservationAdapter.FULL11_FIELDS,
        "vg_0",
        "pg_1",
        "vg_1",
        "pg_2",
        "vg_2",
    )
    cfg = RLPolicyCfg(
        observation_mode=mode,
        load_checkpoint=False,
        state_predictor=VelocityModelStatePredictorCfg(
            enabled=True,
            delay_step=2,
            solver="euler",
            step_dt=0.1,
            max_acc=1.0,
        ),
    )
    policy = RLPolicy(
        cfg,
        num_envs=2,
        device="cpu",
        step_dt=0.1,
        physical_action_limit=0.1,
        raw_observation_fields=raw_fields,
    )
    monkeypatch.setattr(
        policy.agent,
        "act",
        lambda states, timestep, timesteps: torch.zeros((2, 1), device=states.device),
    )
    observation = torch.zeros((2, len(raw_fields)))
    observation[:, ObservationIndex.PB] = 0.35
    observation[:, ObservationIndex.PG] = 0.35
    observation[:, raw_fields.index("pg_1")] = torch.tensor([0.40, 0.30])
    observation[:, raw_fields.index("pg_2")] = torch.tensor([0.50, 0.20])

    with torch.no_grad():
        policy.act({"policy": observation})
    state = policy.get_state()

    torch.testing.assert_close(state["policy_predicted_pg"], torch.tensor([0.50, 0.20]))
    torch.testing.assert_close(
        state["policy_predictor_reference_preview_used"],
        torch.ones(2),
    )


def test_policy_full11_active_predictor_holds_goal_without_preview(
    monkeypatch: pytest.MonkeyPatch,
):
    cfg = RLPolicyCfg(
        observation_mode="full11",
        load_checkpoint=False,
        state_predictor=VelocityModelStatePredictorCfg(
            enabled=True,
            delay_step=1,
            solver="euler",
            step_dt=0.1,
            max_acc=1.0,
        ),
    )
    policy = RLPolicy(
        cfg,
        num_envs=2,
        device="cpu",
        step_dt=0.1,
        physical_action_limit=0.1,
    )
    monkeypatch.setattr(
        policy.agent,
        "act",
        lambda states, timestep, timesteps: torch.zeros((2, 1), device=states.device),
    )
    observation = torch.zeros((2, 11))
    observation[:, ObservationIndex.PG] = torch.tensor([0.2, 0.4])

    with torch.no_grad():
        policy.act({"policy": observation})
    state = policy.get_state()

    torch.testing.assert_close(state["policy_predicted_pg"], torch.tensor([0.2, 0.4]))
    torch.testing.assert_close(
        state["policy_predictor_reference_preview_used"],
        torch.zeros(2),
    )
