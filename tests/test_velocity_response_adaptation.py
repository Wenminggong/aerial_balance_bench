"""Tests for asymmetric RL velocity-response adaptation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from gymnasium import Env, spaces

from baselines.rl_env_wrapper import NormalizedRLTrainingWrapper
from baselines.rl_observation_adapter import RLObservationAdapter, RLObservationAdapterCfg
from baselines.rl_policy import RLPolicyCfg
from baselines.velocity_response_adaptation import (
    ADAPTIVE_CHECKPOINT_FORMAT_VERSION,
    PRIVILEGED_RESPONSE_FIELDS,
    VelocityResponseAdaptationCfg,
    VelocityResponseEncoder,
    VelocityResponseEncoderCfg,
    VelocityResponseHistoryBuffer,
    adaptive_checkpoint_metadata,
    load_velocity_response_encoder,
    validate_velocity_response_adaptation_environment,
)


BASE_DIM = 5


def _adaptation_cfg(history_length: int = 3, latent_dim: int = 2):
    return VelocityResponseAdaptationCfg(
        enabled=True,
        history_length=history_length,
        latent_dim=latent_dim,
        encoder=VelocityResponseEncoderCfg(hidden_dims=(4,), activation="relu"),
    )


def _transport_dim(cfg: VelocityResponseAdaptationCfg) -> int:
    return BASE_DIM + cfg.encoder_input_dim + len(PRIVILEGED_RESPONSE_FIELDS)


def _transport_space(cfg: VelocityResponseAdaptationCfg):
    return spaces.Box(low=-np.inf, high=np.inf, shape=(_transport_dim(cfg),), dtype=np.float32)


def _metadata(cfg: VelocityResponseAdaptationCfg, algorithm: str = "rpo"):
    return adaptive_checkpoint_metadata(
        algorithm=algorithm,
        observation_mode="full11",
        base_input_dim=BASE_DIM,
        total_input_dim=_transport_dim(cfg),
        input_fields=(
            *(f"field_{index}" for index in range(BASE_DIM)),
            *cfg.history_field_names,
            *PRIVILEGED_RESPONSE_FIELDS,
        ),
        adaptation_cfg=cfg,
    )


def test_history_buffer_keeps_two_oldest_to_newest_channels_and_partial_reset():
    history = VelocityResponseHistoryBuffer(2, 3, "cpu")

    history.update(torch.tensor([1.0, 10.0]), torch.tensor([0.1, 1.0]))
    history.update(torch.tensor([2.0]), torch.tensor([0.2]), env_ids=torch.tensor([0]))
    history.update(torch.tensor([3.0]), torch.tensor([0.3]), env_ids=torch.tensor([0]))

    torch.testing.assert_close(
        history.actual_velocity_history,
        torch.tensor([[1.0, 2.0, 3.0], [0.0, 0.0, 10.0]]),
    )
    torch.testing.assert_close(
        history.commanded_velocity_history,
        torch.tensor([[0.1, 0.2, 0.3], [0.0, 0.0, 1.0]]),
    )
    torch.testing.assert_close(
        history.encoder_input[0],
        torch.tensor([1.0, 2.0, 3.0, 0.1, 0.2, 0.3]),
    )
    torch.testing.assert_close(
        history.error_history[0],
        torch.tensor([0.9, 1.8, 2.7]),
    )

    history.reset(torch.tensor([0]))
    torch.testing.assert_close(history.encoder_input[0], torch.zeros(6))
    assert history.needs_transition.tolist() == [True, False]


def test_encoder_config_parsing_and_two_channel_output_shape():
    cfg = VelocityResponseAdaptationCfg.from_dict(
        {
            "enabled": True,
            "history_length": 5,
            "latent_dim": 3,
            "encoder": {"hidden_dims": [7, 6], "activation": "silu"},
        }
    )
    encoder = VelocityResponseEncoder(cfg.history_length, cfg.latent_dim, cfg.encoder)

    assert encoder(torch.zeros((4, 10))).shape == (4, 3)
    assert cfg.encoder_input_dim == 10
    assert cfg.history_field_names[:5] == cfg.actual_history_field_names
    assert cfg.history_field_names[5:] == cfg.command_history_field_names


def test_rl_policy_config_parses_adaptation_and_component_paths():
    cfg = RLPolicyCfg.from_dict(
        {
            "actor_critic_checkpoint_path": "heads.pt",
            "encoder_checkpoint_path": "encoder.pt",
            "velocity_response_adaptation": {
                "enabled": True,
                "history_length": 5,
                "latent_dim": 3,
                "encoder": {"hidden_dims": [7, 6], "activation": "elu"},
            },
        }
    )

    assert cfg.actor_critic_checkpoint_path == "heads.pt"
    assert cfg.encoder_checkpoint_path == "encoder.pt"
    assert cfg.velocity_response_adaptation.encoder_input_dim == 10
    assert cfg.velocity_response_adaptation.latent_dim == 3


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"interface_name": "position"}, "interface_name='velocity'"),
        (
            {
                "robustness_enabled": True,
                "action_delay_enabled": True,
                "delay_step": 2,
            },
            "effective action delay D=0",
        ),
        ({"predictor_active": True}, "active state predictor"),
    ],
)
def test_adaptation_environment_contract_rejects_unsupported_combinations(kwargs, message):
    values = {
        "interface_name": "velocity",
        "robustness_enabled": False,
        "action_delay_enabled": False,
        "delay_step": 0,
        "predictor_active": False,
    }
    values.update(kwargs)

    with pytest.raises(ValueError, match=message):
        validate_velocity_response_adaptation_environment(_adaptation_cfg(), **values)


class _FakeRobustness:
    def __init__(self):
        self.velocity_response_tau_s = torch.tensor([0.11, 0.22])
        self.velocity_response_gain = torch.tensor([0.91, 0.82])
        self.velocity_response_bias = torch.tensor([-0.01, -0.02])


class _AdaptiveWrapperEnv(Env):
    num_envs = 2
    device = "cpu"
    max_episode_length = 10
    observation_fields = RLObservationAdapter.FULL11_FIELDS
    raw_observation_dim = 11
    observation_space = spaces.Dict(
        {"policy": spaces.Box(low=-np.inf, high=np.inf, shape=(11,), dtype=np.float32)}
    )
    action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

    def __init__(self):
        super().__init__()
        self.robustness = _FakeRobustness()

    def reset(self, *, seed=None, options=None):
        del seed, options
        return {"policy": torch.zeros((2, 11))}, {"step": {"command_z": torch.zeros(2)}}

    def step(self, action):
        del action
        observation = torch.zeros((2, 11))
        observation[:, 7] = torch.tensor([0.4, 9.0])
        infos = {
            "step": {
                "command_z": torch.tensor([0.1, 3.0]),
                # Terminal extras intentionally contain the old parameters.
                "velocity_response_tau_s": torch.tensor([0.11, 0.22]),
            }
        }
        # Simulate autoreset resampling for environment 1 after extras are built.
        self.robustness.velocity_response_tau_s[1] = 0.55
        self.robustness.velocity_response_gain[1] = 0.75
        self.robustness.velocity_response_bias[1] = 0.03
        return (
            {"policy": observation},
            torch.zeros(2),
            torch.tensor([False, True]),
            torch.tensor([False, False]),
            infos,
        )


def test_training_wrapper_builds_transport_state_and_uses_post_autoreset_parameters():
    cfg = _adaptation_cfg()
    wrapper = NormalizedRLTrainingWrapper(
        _AdaptiveWrapperEnv(),
        RLObservationAdapterCfg(observation_mode="full11"),
        physical_action_limit=0.5,
        adaptation_cfg=cfg,
    )

    reset_observation, _ = wrapper.reset()
    next_observation, *_ = wrapper.step(torch.zeros((2, 1)))

    assert reset_observation["policy"].shape == (2, 20)
    torch.testing.assert_close(
        next_observation["policy"][0, 11:17],
        torch.tensor([0.0, 0.0, 0.4, 0.0, 0.0, 0.1]),
    )
    torch.testing.assert_close(next_observation["policy"][1, 11:17], torch.zeros(6))
    torch.testing.assert_close(
        next_observation["policy"][1, -3:],
        torch.tensor([0.55, 0.75, 0.03]),
    )


def test_asymmetric_models_have_disjoint_parameters_and_actor_only_encoder_gradient():
    pytest.importorskip("skrl")
    from baselines.rl_models import MLPNetworkCfg, make_models

    cfg = _adaptation_cfg()
    models = make_models(
        "rpo",
        MLPNetworkCfg(layer_num=2, hidden_dim=8),
        _transport_space(cfg),
        spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
        "cpu",
        adaptation_cfg=cfg,
        base_observation_dim=BASE_DIM,
    )
    actor = models["policy"]
    critic = models["value"]
    states = torch.randn((6, _transport_dim(cfg)))

    assert actor is not critic
    assert {id(parameter) for parameter in actor.parameters()}.isdisjoint(
        {id(parameter) for parameter in critic.parameters()}
    )
    actor.compute({"states": states, "alpha": 0.0}, "policy")[0].sum().backward()
    assert any(
        parameter.grad is not None and torch.any(parameter.grad != 0)
        for parameter in actor.response_encoder.parameters()
    )

    actor.zero_grad(set_to_none=True)
    critic.compute({"states": states}, "value")[0].sum().backward()
    assert all(parameter.grad is None for parameter in actor.response_encoder.parameters())


def test_actor_ignores_privileged_parameters_and_critic_ignores_histories():
    pytest.importorskip("skrl")
    from baselines.rl_models import MLPNetworkCfg, make_models

    cfg = _adaptation_cfg()
    models = make_models(
        "rpo",
        MLPNetworkCfg(layer_num=2, hidden_dim=8),
        _transport_space(cfg),
        spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
        "cpu",
        adaptation_cfg=cfg,
        base_observation_dim=BASE_DIM,
    )
    actor = models["policy"]
    critic = models["value"]
    with torch.no_grad():
        for model in models.values():
            for module in model.modules():
                if isinstance(module, torch.nn.Linear):
                    module.weight.fill_(0.1)
                    module.bias.zero_()

    states = torch.zeros((2, _transport_dim(cfg)))
    states[1, -3:] = torch.tensor([0.2, 0.9, -0.01])
    actor_by_params = actor.compute({"states": states, "alpha": 0.0}, "policy")
    critic_by_params = critic.compute({"states": states}, "value")[0]
    torch.testing.assert_close(actor_by_params[0][0], actor_by_params[0][1])
    torch.testing.assert_close(
        actor_by_params[2]["velocity_response_latent"][0],
        actor_by_params[2]["velocity_response_latent"][1],
    )
    assert not torch.allclose(critic_by_params[0], critic_by_params[1])

    states = torch.zeros((2, _transport_dim(cfg)))
    states[1, BASE_DIM : BASE_DIM + cfg.encoder_input_dim] = 1.0
    actor_by_history = actor.compute({"states": states, "alpha": 0.0}, "policy")
    critic_by_history = critic.compute({"states": states}, "value")[0]
    assert not torch.allclose(actor_by_history[0][0], actor_by_history[0][1])
    assert not torch.allclose(
        actor_by_history[2]["velocity_response_latent"][0],
        actor_by_history[2]["velocity_response_latent"][1],
    )
    torch.testing.assert_close(critic_by_history[0], critic_by_history[1])


def test_deployment_policy_uses_two_histories_and_zero_privileged_placeholders(monkeypatch):
    pytest.importorskip("skrl")
    from baselines.rl_policy import RLPolicy

    cfg = RLPolicyCfg(
        observation_mode="full11",
        load_checkpoint=False,
        velocity_response_adaptation=_adaptation_cfg(),
    )
    policy = RLPolicy(
        cfg,
        num_envs=2,
        device="cpu",
        step_dt=0.1,
        physical_action_limit=0.1,
    )
    actor = policy.models["policy"]

    def _act(states, timestep, timesteps):
        del timestep, timesteps
        return actor.act({"states": states, "alpha": 0.0}, role="policy")

    monkeypatch.setattr(policy.agent, "act", _act)
    observation = torch.zeros((2, 11))

    policy.act({"policy": observation}, {"step": {"command_z": torch.tensor([9.0, 9.0])}})
    torch.testing.assert_close(policy.response_history.encoder_input, torch.zeros((2, 6)))

    observation[:, 7] = torch.tensor([0.4, 0.6])
    policy.act({"policy": observation}, {"step": {"command_z": torch.tensor([0.1, 0.2])}})
    torch.testing.assert_close(
        policy.response_history.encoder_input,
        torch.tensor(
            [
                [0.0, 0.0, 0.4, 0.0, 0.0, 0.1],
                [0.0, 0.0, 0.6, 0.0, 0.0, 0.2],
            ]
        ),
    )
    torch.testing.assert_close(policy.policy_input[:, -3:], torch.zeros((2, 3)))

    policy.reset(torch.tensor([0]))
    observation[:, 7] = torch.tensor([0.9, 0.8])
    policy.act({"policy": observation}, {"step": {"command_z": torch.tensor([7.0, 0.3])}})
    torch.testing.assert_close(policy.response_history.encoder_input[0], torch.zeros(6))
    torch.testing.assert_close(
        policy.response_history.encoder_input[1],
        torch.tensor([0.0, 0.6, 0.8, 0.0, 0.2, 0.3]),
    )


def test_agent_optimizer_contains_each_asymmetric_parameter_once():
    pytest.importorskip("skrl")
    from baselines.rl_models import MLPNetworkCfg, make_agent_class_and_cfg, make_models

    cfg = _adaptation_cfg()
    observation_space = _transport_space(cfg)
    action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
    models = make_models(
        "ppo",
        MLPNetworkCfg(layer_num=2, hidden_dim=8),
        observation_space,
        action_space,
        "cpu",
        adaptation_cfg=cfg,
        base_observation_dim=BASE_DIM,
    )
    agent_cls, agent_cfg = make_agent_class_and_cfg(
        "ppo",
        {"state_preprocessor": None, "value_preprocessor": None},
        observation_space,
        "cpu",
        adaptive_checkpointing=True,
    )
    agent = agent_cls(
        models=models,
        memory=None,
        cfg=agent_cfg,
        observation_space=observation_space,
        action_space=action_space,
        device="cpu",
    )
    optimizer_parameters = agent.optimizer.param_groups[0]["params"]
    model_parameters = [*models["policy"].parameters(), *models["value"].parameters()]

    assert len(optimizer_parameters) == len({id(parameter) for parameter in optimizer_parameters})
    assert {id(parameter) for parameter in optimizer_parameters} == {
        id(parameter) for parameter in model_parameters
    }


@pytest.mark.parametrize("algorithm", ["ppo", "rpo"])
def test_encoder_updates_through_complete_agent_rollout(algorithm):
    pytest.importorskip("skrl")
    from skrl.memories.torch import RandomMemory

    from baselines.rl_models import MLPNetworkCfg, make_agent_class_and_cfg, make_models

    cfg = _adaptation_cfg()
    observation_space = _transport_space(cfg)
    action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
    models = make_models(
        algorithm,
        MLPNetworkCfg(layer_num=2, hidden_dim=8),
        observation_space,
        action_space,
        "cpu",
        adaptation_cfg=cfg,
        base_observation_dim=BASE_DIM,
    )
    memory = RandomMemory(memory_size=4, num_envs=2, device="cpu")
    overrides = {
        "rollouts": 4,
        "learning_epochs": 1,
        "mini_batches": 1,
        "learning_starts": 0,
        "experiment": {"write_interval": 0, "checkpoint_interval": 0, "wandb": False},
    }
    if algorithm == "rpo":
        overrides["alpha"] = 0.2
    agent_cls, agent_cfg = make_agent_class_and_cfg(
        algorithm,
        overrides,
        observation_space,
        "cpu",
        adaptive_checkpointing=True,
    )
    agent = agent_cls(
        models=models,
        memory=memory,
        cfg=agent_cfg,
        observation_space=observation_space,
        action_space=action_space,
        device="cpu",
    )
    agent.init()
    encoder_before = {
        key: value.detach().clone()
        for key, value in models["policy"].response_encoder.state_dict().items()
    }

    states = torch.randn((2, _transport_dim(cfg)))
    for timestep in range(4):
        actions = agent.act(states, timestep=timestep, timesteps=4)[0]
        next_states = torch.randn((2, _transport_dim(cfg)))
        agent.record_transition(
            states,
            actions,
            torch.randn((2, 1)),
            next_states,
            torch.zeros((2, 1), dtype=torch.bool),
            torch.zeros((2, 1), dtype=torch.bool),
            {},
            timestep=timestep,
            timesteps=4,
        )
        agent.post_interaction(timestep=timestep, timesteps=4)
        states = next_states

    assert any(
        not torch.equal(value, encoder_before[key])
        for key, value in models["policy"].response_encoder.state_dict().items()
    )
    assert "Learning / Velocity response encoder gradient norm" in agent.tracking_data
    assert "Critic / velocity_response_tau_s mean" in agent.tracking_data


def test_rpo_perturbation_changes_only_actor_action():
    pytest.importorskip("skrl")
    from baselines.rl_models import MLPNetworkCfg, make_models

    cfg = _adaptation_cfg()
    models = make_models(
        "rpo",
        MLPNetworkCfg(layer_num=2, hidden_dim=8),
        _transport_space(cfg),
        spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
        "cpu",
        adaptation_cfg=cfg,
        base_observation_dim=BASE_DIM,
    )
    actor = models["policy"]
    critic = models["value"]
    states = torch.randn((16, _transport_dim(cfg)))

    unperturbed = actor.compute({"states": states, "alpha": 0.0}, "policy")
    critic_before = critic.compute({"states": states}, "value")[0]
    perturbed = actor.compute({"states": states, "alpha": 0.5}, "policy")
    critic_after = critic.compute({"states": states}, "value")[0]

    torch.testing.assert_close(unperturbed[0], unperturbed[2]["no_pert_mean_actions"])
    torch.testing.assert_close(
        unperturbed[2]["velocity_response_latent"],
        perturbed[2]["velocity_response_latent"],
    )
    torch.testing.assert_close(critic_before, critic_after)
    assert not torch.allclose(perturbed[0], perturbed[2]["no_pert_mean_actions"])


def _make_adaptive_agent(algorithm: str, cfg: VelocityResponseAdaptationCfg, metadata):
    from baselines.rl_models import (
        MLPNetworkCfg,
        configure_adaptive_agent_checkpointing,
        make_agent_class_and_cfg,
        make_models,
    )

    observation_space = _transport_space(cfg)
    action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
    models = make_models(
        algorithm,
        MLPNetworkCfg(layer_num=2, hidden_dim=8),
        observation_space,
        action_space,
        "cpu",
        adaptation_cfg=cfg,
        base_observation_dim=BASE_DIM,
    )
    agent_cls, agent_cfg = make_agent_class_and_cfg(
        algorithm,
        {},
        observation_space,
        "cpu",
        adaptive_checkpointing=True,
    )
    agent = agent_cls(
        models=models,
        memory=None,
        cfg=agent_cfg,
        observation_space=observation_space,
        action_space=action_space,
        device="cpu",
    )
    configure_adaptive_agent_checkpointing(agent, metadata)
    return agent


def test_adaptive_component_checkpoint_round_trip_and_standalone_encoder(tmp_path: Path):
    pytest.importorskip("skrl")
    from baselines.rl_models import export_adaptive_agent_components, load_adaptive_agent_components

    cfg = _adaptation_cfg()
    metadata = _metadata(cfg)
    source_agent = _make_adaptive_agent("rpo", cfg, metadata)
    source_agent._state_preprocessor(
        torch.randn((32, _transport_dim(cfg))),
        train=True,
    )
    actor_path, encoder_path = export_adaptive_agent_components(source_agent, tmp_path, "test")
    restored_agent = _make_adaptive_agent("rpo", cfg, metadata)
    load_adaptive_agent_components(restored_agent, actor_path, encoder_path, metadata)

    for role in ("policy", "value"):
        for key, source_value in source_agent.models[role].state_dict().items():
            torch.testing.assert_close(
                source_value,
                restored_agent.models[role].state_dict()[key],
            )

    raw_history = torch.randn((4, cfg.encoder_input_dim))
    standalone_encoder = load_velocity_response_encoder(
        encoder_path,
        expected_metadata=metadata,
    )
    scaler_state = source_agent._state_preprocessor.state_dict()
    history_slice = slice(BASE_DIM, BASE_DIM + cfg.encoder_input_dim)
    normalized_history = (
        raw_history - scaler_state["running_mean"][history_slice].float()
    ) / (
        torch.sqrt(scaler_state["running_variance"][history_slice].float())
        + source_agent._state_preprocessor.epsilon
    )
    normalized_history = normalized_history.clamp(
        -source_agent._state_preprocessor.clip_threshold,
        source_agent._state_preprocessor.clip_threshold,
    )
    expected_latent = source_agent.models["policy"].response_encoder(normalized_history)
    torch.testing.assert_close(standalone_encoder(raw_history), expected_latent)

    actor_payload = torch.load(actor_path, map_location="cpu", weights_only=False)
    encoder_payload = torch.load(encoder_path, map_location="cpu", weights_only=False)
    assert actor_payload["format_version"] == ADAPTIVE_CHECKPOINT_FORMAT_VERSION == 2
    assert actor_payload["pair_id"] == encoder_payload["pair_id"]
    assert "critic_state_dict" in actor_payload


def test_complete_checkpoint_round_trip_uses_distinct_policy_and_value(tmp_path: Path):
    pytest.importorskip("skrl")
    from baselines.rl_models import validate_adaptive_agent_checkpoint

    cfg = _adaptation_cfg()
    metadata = _metadata(cfg, algorithm="ppo")
    source_agent = _make_adaptive_agent("ppo", cfg, metadata)
    with torch.no_grad():
        for model in source_agent.models.values():
            for parameter in model.parameters():
                parameter.uniform_(-0.2, 0.2)
    checkpoint_path = tmp_path / "agent.pt"
    source_agent.save(str(checkpoint_path))

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert {"policy", "value", "optimizer", "state_preprocessor", "adaptive_metadata"} <= set(
        payload
    )
    assert "shared_actor_critic" not in payload
    validate_adaptive_agent_checkpoint(checkpoint_path, metadata, "cpu")

    restored_agent = _make_adaptive_agent("ppo", cfg, metadata)
    restored_agent.load(str(checkpoint_path))
    for role in ("policy", "value"):
        for key, source_value in source_agent.models[role].state_dict().items():
            torch.testing.assert_close(
                source_value,
                restored_agent.models[role].state_dict()[key],
            )


def test_v1_complete_checkpoint_is_explicitly_rejected(tmp_path: Path):
    pytest.importorskip("skrl")
    from baselines.rl_models import validate_adaptive_agent_checkpoint

    path = tmp_path / "v1.pt"
    torch.save({"shared_actor_critic": {}}, path)
    with pytest.raises(ValueError, match="format v1"):
        validate_adaptive_agent_checkpoint(path, _metadata(_adaptation_cfg()), "cpu")


def test_disabled_adaptation_keeps_separate_legacy_models():
    pytest.importorskip("skrl")
    from baselines.rl_models import MLPActor, MLPCritic, MLPNetworkCfg, make_models

    observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(11,), dtype=np.float32)
    action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
    models = make_models(
        "ppo",
        MLPNetworkCfg(layer_num=2, hidden_dim=8),
        observation_space,
        action_space,
        "cpu",
    )

    assert isinstance(models["policy"], MLPActor)
    assert isinstance(models["value"], MLPCritic)
    assert models["policy"] is not models["value"]
    assert models["policy"].num_observations == 11
