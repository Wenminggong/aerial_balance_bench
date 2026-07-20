# Nonlinear Feedforward--Feedback Controller

This document describes the delay-free nonlinear feedforward--feedback
controller (NFFB) for `task_name: unified_tracking` and
`interface_name: velocity`. The implementation is vectorized over Isaac Lab
environments and returns the physical incremental velocity action expected by
`VelocityInterface`.

NFFB is the first stage of the planned delay-compensated NFFB controller. It
does not predict future plant state, command delay, or the optional FOPDT
velocity response.

## Supported contract

NFFB requires:

- `task_name: unified_tracking`;
- `interface_name: velocity`;
- `reference_preview.enabled: true`;
- `reference_preview.future_steps >= 1`;
- no active command-level action delay.

The controller leaves the legacy observation prefix unchanged:

```text
[pb, vb, ab, theta, omega, alpha, drz, vrz, arz, pg, a_prev]
```

It additionally consumes `vg_0` and `vg_1` from the existing reference
preview. No task, reward, evaluator, or reference waveform is modified.

## Model and coordinate convention

The paper dynamics use the ball coordinate measured along the complete beam.
The benchmark observation `pb` describes the active balancing interval and
removes the slide and beam-block geometry. Therefore NFFB first computes

```text
pb_model = pb + ball_position_offset
ball_position_offset = plank_slide_length + beam_block_offset
```

The default environment resolves this offset to

```text
0.03 + 0.30 = 0.33 m.
```

Using the observed `pb` directly in `(pb - L)` would produce a systematic
model error. The offset is automatically resolved by the evaluation runner
and stored in `resolved_run.yaml`.

With

```text
M_b = J_b / r_b^2 + m_b
gamma = m_b / M_b,
```

the rolling-ball model is

```text
ab = gamma * ((pb_model - L) * omega^2 - g * sin(theta)).
```

The rope angle and the velocity/beam-rate mappings are

```text
beta = asin((L / l) * (1 - cos(theta)))
omega = -vrz * cos(beta) / (L * cos(beta - theta))
vrz = -L * omega * cos(beta - theta) / cos(beta).
```

`baselines/velocity_interface_model.py` implements these equations. Arcsine
inputs are clipped and small geometry denominators are protected by
`model.epsilon`.

## Reference derivatives

The environment already defines reference velocity by a one-step forward
difference. NFFB preserves that discrete convention:

```text
p_ref = pg_0
v_ref = vg_0
a_ref_raw = (vg_1 - vg_0) / Ts.
```

This gives zero derivatives for constant references and remains consistent
with the reference velocity used by the unified reward and evaluator.
Triangle and trapezoid waveforms have discrete acceleration spikes at their
corners. NFFB does not change those benchmark references; it clips the
acceleration feedforward to `constraints.max_reference_acceleration`.

When the limit is `auto`, the resolved value is

```text
gamma * g * sin(theta_max).
```

Set `outer_loop.acceleration_feedforward_enabled: false` for a feedback-only
ablation while retaining the same reference and diagnostics.

## Outer loop and nonlinear inversion

The ball tracking errors are

```text
ep = pb - p_ref
ev = vb - v_ref.
```

The desired acceleration is

```text
a_ball_raw = a_ref - kp * ep - kv * ev - ki * integral_ep.
```

The default config expresses the gains through a desired outer bandwidth:

```text
kp = omega_o^2 + 2 * alpha * zeta_o * omega_o
kv = 2 * zeta_o * omega_o + alpha
ki = alpha * omega_o^2.
```

`alpha = 0` disables integral control. The supplied defaults resolve to
`kp = 1`, `kv = 2`, and `ki = 0`.

For the current `pb_model` and `omega`, the controller computes the ball
acceleration range achievable under `|theta| <= theta_max`:

```text
a_min = gamma * ((pb_model - L) * omega^2 - g * sin(theta_max))
a_max = gamma * ((pb_model - L) * omega^2 + g * sin(theta_max)).
```

After clipping `a_ball_raw` to this interval, nonlinear inversion gives

```text
theta_star = asin(
    ((pb_model - L) * omega^2 - a_ball / gamma) / g
).
```

The centripetal term is retained throughout.

## Command filter and inner loop

NFFB filters the raw beam-angle command using

```text
theta_d_dot = omega_d
omega_d_dot = omega_f^2 * (theta_star - theta_d)
              - 2 * zeta_f * omega_f * omega_d.
```

The implementation uses a semi-implicit Euler update: it updates and clips
`omega_d` first, then integrates and clips `theta_d`. An outward angular
velocity is set to zero when the angle reaches its bound. After a full or
partial reset, the first observation initializes `theta_d` and `omega_d` from
the measured beam state to avoid a command discontinuity.

The inner beam-angle loop is

```text
omega_command = omega_d - k_theta * (theta - theta_d).
```

The exact current-configuration geometry then maps `omega_command` to the raw
drone vertical-velocity command.

## Velocity action and saturation

The policy maintains its own absolute `velocity_command`, initialized to zero
at environment reset. In a delay-free run it mirrors
`VelocityInterface.command_velocity[:, 2]`.

Each step applies the following saturation order:

1. clip the reference acceleration feedforward;
2. clip desired ball acceleration to the feasible beam-angle interval;
3. limit filtered beam angle and angular rate;
4. optionally clip absolute vertical velocity;
5. clip the velocity increment to `max_acc * Ts`.

The returned action is

```text
delta_vrz = clip(
    velocity_desired - velocity_command,
    -max_acc * Ts,
    +max_acc * Ts
).
```

`constraints.max_acc` and `constraints.max_velocity` are resolved from the
environment. The benchmark convention `max_velocity: 0` disables the absolute
velocity limit. The runner rejects a policy `max_acc` that differs from the
environment value, preventing silent double-clipping with inconsistent
bounds.

## Integral anti-windup

Integral control is disabled by default. When `outer_loop.integral_pole > 0`,
conditional integration freezes the integral when:

- the desired ball acceleration is saturated and the position error would
  push it farther into saturation;
- the command-filter angle or angular rate is saturated;
- the absolute velocity is saturated; or
- the per-step acceleration limit is saturated.

Otherwise,

```text
integral_ep <- clip(
    integral_ep + Ts * ep,
    -integral_limit,
    +integral_limit
).
```

## Configuration and initial tuning

The supplied `baselines/configs/nffb.yaml` starts with:

| Parameter | Default |
| --- | ---: |
| Outer natural frequency | `1.0 rad/s` |
| Outer damping ratio | `1.0` |
| Integral pole | `0.0` |
| Command-filter natural frequency | `2.5 rad/s` |
| Command-filter damping ratio | `1.0` |
| Inner angle gain | `4.0 s^-1` |
| Controller beam-angle limit | `40 deg` |
| Commanded beam-rate limit | `0.5 rad/s` |
| Drone maximum acceleration | `auto` (`0.5 m/s^2` in supplied env configs) |
| Drone maximum velocity | `auto` (`0`, disabled, in supplied env configs) |

Tune in this order:

1. use constant references with `integral_pole: 0`;
2. adjust the outer bandwidth while keeping
   `omega_o < omega_f < k_theta`;
3. validate sine tracking and compare feedforward on/off;
4. validate corner saturation on triangle and trapezoid references;
5. enable a slow integral pole only if constant tracking has a repeatable
   steady-state bias.

Persistent angle, velocity, or increment saturation indicates that the
requested bandwidth is not feasible.

## Sine parameter search

The generic defaults above are intentionally conservative and are not the
selected high-performance parameters for every reference distribution. The
reproducible sine-tuning workflow is configured by
`baselines/configs/nffb_sine_tuning.yaml` and fixes:

```text
phase = 0
amplitude in [0.05, 0.20] m
period in [4, 10] s
max_acc = 5.0 m/s^2
max_velocity = 0
```

The resulting sine-specific policy is stored in
`baselines/configs/nffb_sine_phase0_acc5.yaml` and uses:

```text
omega_o = 1.35 rad/s, zeta_o = 1.0, alpha = 0
omega_f = 16.0 rad/s, zeta_f = 1.0
k_theta = 6.0 1/s, omega_max = 0.5 rad/s
```

The relatively low inner-loop gain is deliberate. Deterministic
`amplitude=0.20 m, period=4 s` tests showed that combining `omega_f >= 10`
with `k_theta >= 16` can track the random screening distribution while
driving the acceleration-limited velocity interface into persistent
saturation. The selected pair preserves the faster feedforward path without
crossing that actuator-limited stability boundary.

### Phase-zero sine result

The final paired validation uses `seed=666`, 500 parallel environments, and
one 20 s episode per environment. The selected policy and generic defaults
see the same sampled trajectories and initial states.

| Metric | Selected sine policy | Generic default |
| --- | ---: | ---: |
| Completed episodes | 500 | 500 |
| Terminations | 0 | 0 |
| Beam-edge margin violations | 0 | 124 |
| Mean MAE | 0.00602 m | 0.05259 m |
| Mean RMSE | 0.00998 m | 0.06236 m |
| P95 RMSE | 0.02031 m | 0.13678 m |
| P95 NRMSE | 0.1333 | 1.3550 |
| P95 MAXE | 0.08329 m | 0.28794 m |
| Mean absolute steady phase error | 1.59 deg | 8.43 deg |
| Maximum fitted amplitude gain | 1.0957 | 3.8620 |

Mean RMSE improves by 84.0%. All six deterministic stress profiles complete
without termination or entry into the `[0.02, 0.68] m` edge margin; their
fitted gains span `[0.9991, 1.0975]`, P95 absolute phase error is `2.99 deg`,
and P95 steady NRMSE is `0.0798`.

The strict aggregate acceptance report remains `FAIL` only because whole-
episode P95 MAXE is `0.08329 m`, above the requested `0.060 m`. The other
fifteen acceptance checks pass. This peak-error metric includes the
phase-zero reset transient, where the sine reference starts with nonzero
velocity while the ball starts nearly stationary. See
`logs/nffb/sine_tuning/phase0_acc5/final_selection.yaml` for the complete
gate-by-gate report.

Run through one stage, including its prerequisites:

```bash
conda run -n isaac-sim python scripts/tune_nffb_sine.py --stage filter
```

Run the complete resumable search and single-seed, 500-environment,
500-episode validation:

```bash
conda run -n isaac-sim python scripts/tune_nffb_sine.py --stage all
```

Completed trial directories are reused by default. Pass `--no-resume` only
with a new `session_name`, because existing trial directories are never
silently overwritten. The session writes:

- `leaderboard.csv` with lexicographic feasibility/performance ranking;
- per-run `episode_metrics.csv` and `tuning_metrics.yaml`;
- deterministic stress-profile summaries;
- `final_selection.yaml` with every acceptance gate;
- a standalone `nffb_sine_phase0_acc5.yaml` for the selected candidate.

The analyzer includes the initial phase-zero velocity mismatch in whole
episode MAE/RMSE/MAXE. Gain, phase, and steady NRMSE discard the first complete
reference period and fit the ball and goal signals at the configured sine
frequency. It also treats non-finite signals, early termination, entry into
the `[0.02, 0.68] m` beam-edge margin, and excessive saturation as hard
failures.

Analyze an existing NFFB rollout without starting Isaac Sim:

```bash
python scripts/tune_nffb_sine.py \
  --analyze-run logs/nffb/unified_tracking/<run_name>
```

## Evaluation

Run a short sine smoke test:

```bash
python3 scripts/nffb_policy_eval.py \
  --config baselines/configs/nffb_unified_tracking_eval.yaml \
  --env_config environments/configs/unified_tracking_sine.yaml \
  --policy_config baselines/configs/nffb_sine_phase0_acc5.yaml \
  --episodes 4 \
  --num_envs 4 \
  --run_name nffb_unified_sine_smoke \
  --headless
```

Run all singleton families:

```bash
TARGET_EPISODES=1000 NUM_ENVS=10 \
  scripts/run_nffb_policy_eval_configs.sh --headless
```

Each run saves the standard unified metrics, rollout observations/actions,
environment command states, and all `policy_*` controller diagnostics.
`scripts/visualize_rollout.py --groups policy` automatically plots the saved
NFFB states.

## Diagnostics

The rollout includes:

- reference position, velocity, raw/limited/used acceleration;
- position and velocity errors and integral state;
- raw/feasible/commanded ball acceleration;
- `theta_star`, filtered `theta_d/omega_d`, and filter acceleration;
- beam-angle error, angular-rate command, rope angle;
- raw/limited/accumulated velocity command and returned action;
- reference, ball-acceleration, filter-angle, filter-rate, velocity,
  acceleration, and anti-windup saturation flags.

## Current limitations and DC-NFFB extension

NFFB v1 deliberately fails fast when command delay is active. It also does not
predict the optional FOPDT velocity-response model. The future DC-NFFB stage
will:

1. predict plant state through the pending command-delay queue;
2. select `pg_d`, `vg_d`, and `ag_d` at the command execution step;
3. feed the aligned future state/reference into the same outer inversion,
   filter, inner loop, and constraint pipeline.
