from __future__ import annotations

import torch

from baselines.base_policy import ObservationIndex
from baselines.cpid_policy import CPIDPolicy, CPIDPolicyCfg
from baselines.model_state_predictor import VelocityModelStatePredictorCfg


PREVIEW_FIELDS = (
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
    "vg_0",
    "pg_1",
    "vg_1",
    "pg_2",
    "vg_2",
)


def _make_observation(pg: float = 0.3) -> torch.Tensor:
    observation = torch.zeros((2, 11), dtype=torch.float32)
    observation[:, ObservationIndex.PB] = torch.tensor([0.4, 0.2])
    observation[:, ObservationIndex.THETA] = torch.tensor([0.01, -0.02])
    observation[:, ObservationIndex.PG] = pg
    return observation


def test_cpid_ignores_unified_reference_preview_extension():
    policy = CPIDPolicy(CPIDPolicyCfg(), num_envs=2, device="cpu", step_dt=1.0 / 60.0)
    legacy_observation = _make_observation()
    preview_extension = torch.randn((2, 11), dtype=torch.float32)
    unified_observation = torch.cat((legacy_observation, preview_extension), dim=-1)

    with torch.no_grad():
        legacy_action = policy.act({"policy": legacy_observation})
    legacy_state = {name: value.clone() for name, value in policy.get_state().items()}

    policy.reset()
    with torch.no_grad():
        unified_action = policy.act({"policy": unified_observation})
    unified_state = policy.get_state()

    assert torch.equal(unified_action, legacy_action)
    for name, expected in legacy_state.items():
        assert torch.equal(unified_state[name], expected), name


def test_cpid_uses_current_reference_position_from_legacy_prefix():
    policy = CPIDPolicy(CPIDPolicyCfg(), num_envs=2, device="cpu", step_dt=1.0 / 60.0)

    with torch.no_grad():
        lower_goal_action = policy.act({"policy": _make_observation(pg=0.3)})
    lower_goal_error = policy.get_state()["policy_raw_error"].clone()

    policy.reset()
    with torch.no_grad():
        upper_goal_action = policy.act({"policy": _make_observation(pg=0.5)})
    upper_goal_error = policy.get_state()["policy_raw_error"].clone()

    assert torch.allclose(lower_goal_error, torch.tensor([0.1, -0.1]))
    assert torch.allclose(upper_goal_error, torch.tensor([-0.1, -0.3]))
    assert not torch.equal(lower_goal_action, upper_goal_action)


def test_cpid_active_predictor_uses_unified_future_reference():
    cfg = CPIDPolicyCfg(
        state_predictor=VelocityModelStatePredictorCfg(
            enabled=True,
            delay_step=2,
            solver="euler",
            step_dt=0.1,
            max_acc=10.0,
        )
    )
    policy = CPIDPolicy(
        cfg,
        num_envs=2,
        device="cpu",
        step_dt=0.1,
        raw_observation_fields=PREVIEW_FIELDS,
    )
    legacy_observation = _make_observation(pg=0.3)
    extension = torch.tensor(
        [
            [0.0, 0.4, 0.0, 0.5, 0.0],
            [0.0, 0.2, 0.0, 0.1, 0.0],
        ]
    )

    with torch.no_grad():
        policy.act({"policy": torch.cat((legacy_observation, extension), dim=-1)})
    state = policy.get_state()

    torch.testing.assert_close(state["policy_predicted_pg"], torch.tensor([0.5, 0.1]))
    torch.testing.assert_close(
        state["policy_predictor_reference_preview_used"],
        torch.ones(2),
    )


def test_cpid_active_predictor_without_named_preview_holds_current_reference():
    cfg = CPIDPolicyCfg(
        state_predictor=VelocityModelStatePredictorCfg(
            enabled=True,
            delay_step=2,
            solver="euler",
            step_dt=0.1,
            max_acc=10.0,
        )
    )
    policy = CPIDPolicy(cfg, num_envs=2, device="cpu", step_dt=0.1)

    with torch.no_grad():
        policy.act({"policy": _make_observation(pg=0.3)})
    state = policy.get_state()

    torch.testing.assert_close(state["policy_predicted_pg"], torch.full((2,), 0.3))
    torch.testing.assert_close(
        state["policy_predictor_reference_preview_used"],
        torch.zeros(2),
    )
