"""Human dynamics tests independent of Isaac and of the historical project."""

from __future__ import annotations

import ast
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from environments.robustness.human_behavior import HUMAN_STEP_EXTRA_FIELDS, HumanBehaviorModel


ROOT = Path(__file__).resolve().parents[1]


def make_cfg(**overrides):
    # Read the actual flat config defaults without importing the Isaac decorator.
    tree = ast.parse((ROOT / "environments/robustness/robustness_manager.py").read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "RobustnessManagerCfg")
    defaults = {node.target.id: ast.literal_eval(node.value) for node in cls.body if isinstance(node, ast.AnnAssign)}
    defaults.update(enabled=True, external_disturbance_enabled=True)
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def make_model(mode="cooperative", n=1, **overrides):
    model = HumanBehaviorModel(make_cfg(human_behavior_type=mode, **overrides), n, "cpu")
    model.reset(torch.arange(n), torch.zeros(n))
    model.initialize_history(torch.arange(n), torch.zeros(n))
    return model


@pytest.mark.parametrize("mode", ["cooperative", "random-walk", "disturbed"])
def test_matches_independent_scalar_legacy_recurrence(mode):
    """Compare all stages against scalar equations, with a fixed noise sequence."""
    model = make_model(mode, human_pos_feedback_sample=False)
    dt = 1 / 60
    e1 = e2 = t1 = t2 = target_angle = pid_vel = ou = correction = command = 0.0
    scalar = lambda name: getattr(model, name).item()
    clamp = lambda x, lo, hi: min(max(x, lo), hi)
    integral = lambda kp, ti, e: kp * dt / ti * e if ti != 0 else 0.0
    cap = scalar("vel_max_acc") * dt
    for e, theta, pos, noise in [(0.08, 0.0, 0.0, 0.7), (0.082, 0.001, 0.07, -0.3),
                                  (-0.03, -0.001, -0.09, 0.4), (0.0, 0.0, 0.01, 0.0)]:
        da = scalar("angle_kp") * (e - e1) + integral(scalar("angle_kp"), scalar("angle_ti"), e)
        da += scalar("angle_kp") * scalar("angle_td") / dt * (e - 2 * e1 + e2)
        target_angle += clamp(da, -scalar("angle_max_theta_change"), scalar("angle_max_theta_change"))
        te = target_angle - theta
        dv = scalar("vel_kp") * (te - t1) + integral(scalar("vel_kp"), scalar("vel_ti"), te)
        dv += scalar("vel_kp") * scalar("vel_td") / dt * (te - 2 * t1 + t2)
        dv = clamp(dv, -cap, cap)
        pid_vel += dv
        gain = math.exp(-(e / scalar("behavior_sigma_d")) ** 2)
        coop = scalar("behavior_wc") * pid_vel
        disrupt = scalar("behavior_wd") * gain * math.tanh(e / scalar("behavior_e_scale"))
        effective = math.copysign(max(abs(pos) - scalar("pos_deadzone"), 0), pos)
        raw = model.cfg.human_pos_corr_max * math.tanh(scalar("pos_k_base") * effective / model.cfg.human_pos_corr_max)
        correction += dt / (scalar("pos_corr_tau") + dt) * (raw - correction)
        ou += scalar("ou_theta") * (model.cfg.external_disturbance_ou_mu - ou) * dt + scalar("ou_sigma") * math.sqrt(dt) * noise
        ou = clamp(ou, -model.cfg.external_disturbance_ou_clip, model.cfg.external_disturbance_ou_clip)
        desired = coop + disrupt - correction + ou
        increment = clamp(desired - command, -cap, cap)
        old_command = command
        command += increment
        model.prepare_step(torch.tensor([e]), torch.tensor([theta]), torch.tensor([pos]), dt, 3,
                           noise=torch.tensor([noise]))
        for name, expected in (("theta_target", target_angle), ("cpid_vel", pid_vel),
                               ("cpid_action", -dv / cap), ("cooperative_term", coop),
                               ("disturbed_term", disrupt), ("pos_feedback", correction),
                               ("ou_state", ou), ("vel_des", desired), ("acc_cmd", increment / dt)):
            assert scalar(name) == pytest.approx(expected, abs=2e-6), name
        for substep in range(1, 4):
            assert model.advance_substep().item() == pytest.approx(old_command + increment * substep / 3, abs=2e-7)
        assert model.advance_substep().item() == pytest.approx(command, abs=2e-7)
        e2, e1, t2, t1 = e1, e, t1, te


def test_reset_history_has_no_false_derivative_pulse_and_zero_ti_is_safe():
    model = make_model(human_angle_ti_min=0, human_angle_ti_max=0, human_vel_ti=0, human_ou_enabled=False)
    e = torch.tensor([0.1])
    model.initialize_history([0], e)
    model.prepare_step(e, torch.zeros(1), torch.zeros(1), 0.1, 2)
    assert model.theta_target.item() == 0
    assert model.vel_des.item() == 0


def test_disturbance_is_odd_and_decays_away_from_goal():
    model = make_model("disturbed", n=4, human_disturbed_wd_min=-0.15, human_disturbed_wd_max=-0.15,
                       human_disturbed_sigma_d_min=0.1, human_disturbed_sigma_d_max=0.1,
                       human_disturbed_e_scale_min=0.05, human_disturbed_e_scale_max=0.05,
                       human_ou_enabled=False)
    model.prepare_step(torch.tensor([0.04, -0.04, 0.0, 0.6]), torch.zeros(4), torch.zeros(4), 0.02, 2)
    term = model.disturbed_term[:, 0]
    assert term[0] < 0
    assert term[1] == -term[0]
    assert term[2] == 0
    assert abs(term[3]) < 1e-12
    assert torch.count_nonzero(model.cooperative_term) == 0


@pytest.mark.parametrize("tau", [0.0, 0.1])
def test_random_mode_recentring_deadzone_filter_and_disabled_noise(tau):
    model = make_model("random", n=3, human_pos_feedback_sample=False, human_pos_corr_tau=tau,
                       human_ou_enabled=False)
    model.ou_state.fill_(0.5)
    model.prepare_step(torch.ones(3), torch.zeros(3), torch.tensor([0.02, 0.15, -0.15]), 0.1, 2)
    expected = math.tanh(3 * 0.1) * 0.1 / (tau + 0.1)
    torch.testing.assert_close(model.vel_des[:, 0], torch.tensor([0, -expected, expected]))
    assert torch.count_nonzero(model.behavior_term) == 0
    assert torch.count_nonzero(model.ou_state) == 0


@pytest.mark.parametrize("mode", ["cooperative", "random-walk", "disturbed"])
def test_ou_update_is_control_rate_and_clipped(mode):
    model = make_model(mode, external_disturbance_ou_theta_range=(0.2, 0.2),
                       external_disturbance_ou_sigma_range=(0.1, 0.1), external_disturbance_ou_mu=0.01,
                       human_ou_theta=9.0, human_ou_sigma=8.0)
    model.prepare_step(torch.zeros(1), torch.zeros(1), torch.zeros(1), 0.1, 5, noise=torch.ones(1))
    expected = 0.2 * 0.01 * 0.1 + 0.1 * math.sqrt(0.1)
    assert model.ou_state.item() == pytest.approx(expected)
    for _ in range(5):
        model.advance_substep()
    assert model.ou_state.item() == pytest.approx(expected)
    model.prepare_step(torch.zeros(1), torch.zeros(1), torch.zeros(1), 0.1, 5, noise=torch.full((1,), 100.0))
    assert model.ou_state.item() == pytest.approx(0.05)


@pytest.mark.parametrize("mode", ["cooperative", "random-walk", "disturbed"])
def test_shared_ou_ranges_are_sampled_and_held_until_reset(mode):
    model = make_model(mode, n=32, external_disturbance_ou_theta_range=(0.7, 0.9),
                       external_disturbance_ou_sigma_range=(0.009, 0.011))
    for name, low, high in (("ou_theta", 0.7, 0.9), ("ou_sigma", 0.009, 0.011)):
        values = getattr(model, name)
        assert torch.all((values >= low) & (values <= high))
        assert torch.unique(values).numel() > 1
    before = (model.ou_theta.clone(), model.ou_sigma.clone())
    for _ in range(3):
        model.prepare_step(torch.zeros(32), torch.zeros(32), torch.zeros(32), 0.02, 1)
        model.advance_substep()
    assert torch.equal(model.ou_theta, before[0])
    assert torch.equal(model.ou_sigma, before[1])
    model.reset([1], torch.zeros(32))
    keep = torch.arange(32) != 1
    assert torch.equal(model.ou_theta[keep], before[0][keep])
    assert torch.equal(model.ou_sigma[keep], before[1][keep])
    assert 0.7 <= model.ou_theta[1].item() <= 0.9
    assert 0.009 <= model.ou_sigma[1].item() <= 0.011


@pytest.mark.parametrize("decimation", [1, 3, 10])
def test_ramp_substep_acceleration_limit(decimation):
    model = make_model("random-walk", human_ou_enabled=False, human_pos_feedback_sample=False)
    for position in (10.0, -10.0):
        model.prepare_step(torch.zeros(1), torch.zeros(1), torch.tensor([position]), 0.06, decimation)
        for _ in range(decimation):
            previous = model.vel_cmd.clone()
            current = model.advance_substep()
            assert (current - previous).abs().item() <= 0.5 * 0.06 / decimation + 1e-7
        torch.testing.assert_close(model.vel_cmd, model.vel_target)


def test_zero_acceleration_freezes_command_without_nan():
    model = make_model(human_vel_max_acc=0)
    model.prepare_step(torch.ones(1), torch.ones(1), torch.ones(1), 0.1, 3)
    assert model.advance_substep().item() == 0
    assert model.cpid_action.item() == 0
    for value in model.get_state().values():
        assert torch.isfinite(value).all()


@pytest.mark.parametrize("mode", ["cooperative", "random-walk", "disturbed"])
def test_partial_reset_preserves_other_rows_and_clears_all_history(mode):
    model = make_model(mode, n=3)
    model.prepare_step(torch.ones(3), torch.ones(3), torch.ones(3), 0.1, 3)
    model.advance_substep()
    before = {name: getattr(model, name).clone() for name in model._buffer_names}
    counter = model.ramp_counter.clone()
    model.reset([1], torch.tensor([0.0, 0.2, 0.0]))
    model.initialize_history([1], torch.tensor([0.0, 0.3, 0.0]))
    for name, previous in before.items():
        torch.testing.assert_close(getattr(model, name)[[0, 2]], previous[[0, 2]])
    assert torch.equal(model.ramp_counter[[0, 2]], counter[[0, 2]])
    assert model.error_prev1[1].item() == pytest.approx(0.3)
    assert model.error_prev2[1].item() == pytest.approx(0.3)
    for name in ("theta_target", "theta_error_prev1", "theta_error_prev2", "ou_state", "vel_cmd", "acc_mean"):
        assert getattr(model, name)[1].item() == 0
    model.reset([], torch.zeros(3))


def test_disabled_model_does_not_consume_rng_and_records_actual_velocity():
    before = torch.random.get_rng_state().clone()
    model = make_model(enabled=False)
    model.prepare_step(torch.zeros(1), torch.zeros(1), torch.zeros(1), 0.1, 3)
    model.advance_substep()
    assert torch.equal(torch.random.get_rng_state(), before)
    model.record_measurement(torch.tensor([0.2]), torch.tensor([0.3]), torch.tensor([0.4]))
    model.record_measurement(torch.tensor([0.25]), torch.tensor([0.35]), torch.tensor([0.6]))
    state = model.get_state()
    assert set(state) == set(HUMAN_STEP_EXTRA_FIELDS)
    assert state["human_vel_z"].item() == pytest.approx(0.35)
    assert state["human_acc_z"].item() == pytest.approx(0.5)
    assert state["human_target_vel_z"].item() == 0
    assert state["human_enabled"].item() == 0
    assert all(value.shape == (1,) for value in state.values())


@pytest.mark.parametrize("overrides", [
    {"human_behavior_type": "invalid"}, {"human_vel_max_acc": -1}, {"human_pos_corr_tau": float("nan")},
    {"human_behavior_e_scale": 0}, {"human_disturbed_sigma_d_min": -0.1},
    {"human_angle_kp_min": 9}, {"external_disturbance_ou_theta_range": (0.3, 0.1)},
    {"external_disturbance_ou_sigma_range": (0.1, float("inf"))},
    {"external_disturbance_ou_clip": -0.1},
])
def test_invalid_configuration_fails_early(overrides):
    with pytest.raises(ValueError):
        HumanBehaviorModel(make_cfg(**overrides), 1, "cpu")


@pytest.mark.parametrize("dt,decimation", [(0, 1), (float("nan"), 3), (0.1, 0), (0.1, 1.5), (0.1, True)])
def test_invalid_timing_fails_early(dt, decimation):
    with pytest.raises(ValueError):
        make_model().prepare_step(torch.zeros(1), torch.zeros(1), torch.zeros(1), dt, decimation)


def test_seeded_sampling_reproduces_parameters_and_trajectory():
    def run():
        torch.manual_seed(42)
        model = make_model(n=4)
        for _ in range(5):
            model.prepare_step(torch.ones(4) * 0.02, torch.zeros(4), torch.zeros(4), 0.1, 2)
            model.advance_substep()
            model.advance_substep()
        return {key: value.clone() for key, value in model.get_state().items()}
    first, second = run(), run()
    for key in first:
        assert torch.equal(first[key], second[key]), key


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_vectorized_model():
    model = HumanBehaviorModel(make_cfg(), 4, "cuda")
    values = torch.zeros(4, device="cuda")
    model.reset(torch.arange(4, device="cuda"), values)
    model.prepare_step(values, values, values, 1 / 60, 3)
    assert model.advance_substep().device.type == "cuda"
