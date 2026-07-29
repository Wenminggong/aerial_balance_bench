# Nonlinear Feedforward--Feedback Controller

This document describes the nonlinear feedforward--feedback controller (NFFB)
for `task_name: unified_tracking` and
`interface_name: velocity`. The implementation is vectorized over Isaac Lab
environments and returns the physical incremental velocity action expected by
`VelocityInterface`.

NFFB keeps its original delay-free path and optionally uses the shared
velocity-model state predictor for command-delay compensation. The predictor
can include the deterministic nominal first-order velocity response. NFFB also
provides a separately switchable exact inverse of that response between its
desired output velocity and the command sent to `VelocityInterface`.

## Supported contract

NFFB requires:

- `task_name: unified_tracking`;
- `interface_name: velocity`;
- `reference_preview.enabled: true`;
- in delay-free mode, `reference_preview.future_steps >= 1`;
- with a fixed action delay of `D > 0`, an active predictor with the same
  `delay_step` and `reference_preview.future_steps >= D + 1`;
- with random `delay_step_choices`, an active predictor with an explicitly
  configured fixed nominal `D` and `reference_preview.future_steps >= D + 1`.

The controller leaves the legacy observation prefix unchanged:

```text
[pb, vb, ab, theta, omega, alpha, drz, vrz, arz, pg, a_prev]
```

The delay-free path additionally consumes `vg_0` and `vg_1`. The delayed path
uses `pg_0 ... pg_D` for plant prediction, then consumes `vg_D` and
`vg_{D+1}` for NFFB. No task, reward, evaluator, or reference waveform is
modified.

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

With an active `D`-step state predictor, the same update is shifted to the
command execution time:

```text
p_ref = pg_D
v_ref = vg_D
a_ref_raw = (vg_{D+1} - vg_D) / Ts.
```

The plant state is first predicted through the pending delayed-command queue.
The NFFB integral, command filter, and anti-windup state then execute one normal
update using that predicted state; they are not replayed `D` times.

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

Without response compensation, the returned action remains:

```text
delta_vrz = clip(
    velocity_desired - velocity_command,
    -max_acc * Ts,
    +max_acc * Ts
).
```

This is the original NFFB path and remains the default.

### Optional first-order response inverse

Set `velocity_response_compensation.enabled: true` to interpret the nonlinear
inversion result as a desired response output rather than a direct interface
input. For the deterministic nominal response

```text
r_k = a * r_{k-1} + (1 - a) * (K * u_k + b)
a = exp(-Ts / tau),
```

NFFB calculates the unconstrained input

```text
u_raw = (((v_desired - a * r_pre) / (1 - a)) - b) / K.
```

For `tau = 0`, the static inverse is
`u_raw = (v_desired - b) / K`. Near-zero one-step response fractions are
rejected because their inverse is numerically ill-conditioned.

The compensator owns a nominal response state separate from the plant state
predictor. At `D=0`, `r_pre` is the current compensator state. With an active
`D`-step predictor, the compensator first advances that state through the
ordered pending-command queue, producing the response state immediately before
the new command executes. After generating the action, its persistent state is
advanced by only the command that executes in the current environment step.
Thus predictor lookahead does not replay the persistent response state `D`
times.

`parameter_source: state_predictor` copies
`velocity_response_tau_s/gain/bias/max_abs_velocity` from the resolved
predictor configuration. This works even when delay prediction is inactive.
`parameter_source: explicit` instead uses `tau_s/gain/bias/max_abs_velocity`
from the compensation section. Explicit `auto` values resolve only from fixed
robustness ranges; randomized ranges require a nominal value. If an active
response-aware predictor and an explicit inverse describe the same response,
their parameters must agree.

Each step applies the following saturation order:

1. clip the reference acceleration feedforward;
2. clip desired ball acceleration to the feasible beam-angle interval;
3. limit filtered beam angle and angular rate;
4. optionally clip the desired response output with
   `velocity_response_compensation.max_abs_velocity`;
5. apply the exact response inverse;
6. project its input onto the absolute command bound and the one-step reachable
   interval defined by `max_acc * Ts`.

`constraints.max_acc` and `constraints.max_velocity` are resolved from the
environment and constrain the input command, not the desired response output.
The benchmark convention `max_velocity: 0` disables the absolute input limit.
The runner rejects a policy `max_acc` that differs from the environment value,
preventing silent double-clipping with inconsistent bounds.

## Integral anti-windup

Integral control is disabled by default. When `outer_loop.integral_pole > 0`,
conditional integration freezes the integral when:

- the desired ball acceleration is saturated and the position error would
  push it farther into saturation;
- the command-filter angle or angular rate is saturated;
- the response output or inverse input velocity is saturated; or
- the inverse input reaches the per-step acceleration limit.

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
| Velocity-response inverse | disabled |
| Response parameter source | `state_predictor` |

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

## Mixed-reference coarse search

`scripts/tune_nffb_mixed.py` reuses the existing NFFB evaluator and rollout
analysis for the trajectory families selected by
`environments/configs/unified_tracking_mixed.yaml`. The tuning script does not
generate a replacement environment config: it only overrides the evaluator
seed, parallel environment count, episode target, and candidate policy gains.
Consequently the task distribution, episode duration, preview horizon,
velocity-interface limits, and robustness settings always come from the
checked-in environment YAML.

The supplied coarse-search config selects the current mixed families:

```text
constant, random_b_spline, random_ramp_dwell
```

The environment samples its final response from non-degenerate `tau/gain/bias`
ranges. Predictor `auto` values therefore cannot be resolved safely. The
current workflow keeps delay prediction and the NFFB response inverse disabled.
It leaves the response model enabled and records the range midpoints explicitly
for diagnostics and future predictor use:

```text
tau = 0.155 s, gain = 0.86, bias = -0.0035
```

Screening uses `seed=666` with 48 parallel one-episode environments. It starts
from the generic and sine-tuned policies, performs a small staged
filter/inner-loop and outer-loop search, conditionally checks slow integral
poles for constant-reference bias, and permits at most four local recovery
perturbations. The top three candidates and both baselines are evaluated with
the single independent validation seed `667` using 192 parallel episodes.

Run or resume the complete search:

```bash
conda run -n isaac-sim python scripts/tune_nffb_mixed.py --stage all
```

Generate and inspect every fixed trial command without launching Isaac Sim:

```bash
python3 scripts/tune_nffb_mixed.py --stage all --dry-run
```

The mixed analyzer records trajectory IDs/names per episode and reports global
and per-family absolute MAE/RMSE/MAXE, signed bias, boundary/termination state,
and saturation rates. A candidate must cover every configured family, avoid
non-finite/terminated/boundary episodes, keep full-run saturation at or below
5%, and keep final-quarter saturation at or below 1%. Feasible candidates are
ranked by worst-family P95 RMSE, then global RMSE and peak error. This is a
coarse feasibility search, not an optimal or near-optimal parameter claim.

The session writes `leaderboard.csv`, per-run `episode_metrics.csv` and
`type_metrics.yaml`, `screen_selection.yaml`, and `final_selection.yaml` under
`logs/nffb/mixed_tuning/coarse_v1_no_comp/`. A passing validation also writes
the standalone policy
`baselines/configs/nffb_unified_mixed_coarse_no_comp.yaml`, with
velocity-response compensation disabled.

### No-compensation coarse-v1 result

The no-compensation search completed all 20 screening trials and five
independent validation trials. The validation used the current 10-second mixed
environment, `seed=667`, and 192 parallel episodes. No candidate passed every
safety gate, so the workflow deliberately did not create
`baselines/configs/nffb_unified_mixed_coarse_no_comp.yaml`.

The safety-ranked provisional candidate used:

```text
omega_o = 0.68 rad/s, zeta_o = 1.0, alpha = 0
omega_f = 14.0 rad/s, zeta_f = 1.0
k_theta = 6.0 1/s, omega_max = 0.5 rad/s
```

Its response inverse was disabled. The validation result was:

| Metric | Provisional | Generic | Sine-tuned |
| --- | ---: | ---: | ---: |
| Completed episodes | 192 | 192 | 192 |
| Terminations | 0 | 0 | 1 |
| Safe-margin violations | 0 | 16 | 2 |
| Mean RMSE | 0.06814 m | 0.07417 m | 0.03926 m |
| Worst-family P95 RMSE | 0.18332 m | 0.17981 m | 0.13775 m |
| Full-run maximum saturation | 0.78% | 0.17% | 2.16% |
| Final-quarter maximum saturation | 1.61% | 0.44% | 3.77% |

The provisional candidate covered 62 constant, 59 random-B-spline, and 71
random-ramp-dwell episodes. Relative to the generic policy, it improved global
mean RMSE by about 8.1%, random-B-spline mean RMSE by 9.3%, and random-ramp-
dwell mean RMSE by 19.6%; constant mean RMSE was about 3.0% worse. It passed
coverage, non-finite, termination, boundary, and full-run saturation checks,
but failed the 1% final-quarter saturation gate. Treat
`logs/nffb/mixed_tuning/coarse_v1_no_comp/final_selection.yaml` as a negative
coarse-search report, not as a deployable selected policy.

### No-compensation grid-v1 result

The local grid used the coarse result as its center and evaluated the Cartesian
product below while holding `omega_f=14`, both damping ratios at 1.0,
`omega_max=0.5`, and acceleration feedforward enabled:

```text
omega_o = [0.55, 0.68, 0.80] rad/s
k_theta = [4.5, 5.25, 6.0] 1/s
alpha = [0.00, 0.05, 0.10]
```

Run or resume the 27-point search with:

```bash
conda run -n isaac-sim python scripts/tune_nffb_mixed.py \
  --config baselines/configs/nffb_mixed_grid_tuning.yaml \
  --stage all
```

All candidates used the same independent `seed=668`, 500 parallel environments,
and 500 episodes, for a total of 13,500 evaluated episodes. Each candidate
covered 181 constant, 164 random-B-spline, and 155 random-ramp-dwell episodes.
All 27 candidates had zero termination, boundary, and non-finite events, met
the per-family coverage requirement, and stayed below the 5% full-run
saturation threshold. All 27 failed only the 1% final-quarter saturation gate;
observed tail saturation ranged from 1.338% to 1.578%.

The lexicographic safety winner, the initial center, and the lowest-error grid
point were:

| Metric | Safety-ranked best | Initial center | Lowest-error point |
| --- | ---: | ---: | ---: |
| `omega_o / k_theta / alpha` | 0.55 / 4.5 / 0.10 | 0.68 / 6.0 / 0.00 | 0.80 / 6.0 / 0.10 |
| Mean RMSE | 0.07191 m | 0.06677 m | 0.05305 m |
| Worst-family P95 RMSE | 0.16433 m | 0.16558 m | 0.14169 m |
| Full-run maximum saturation | 0.565% | 0.644% | 0.826% |
| Final-quarter maximum saturation | 1.338% | 1.475% | 1.578% |
| Terminations / safe-margin violations | 0 / 0 | 0 / 0 | 0 / 0 |
| Strict acceptance | FAIL | FAIL | FAIL |

Relative to the initial center, the safety winner reduced final-quarter
saturation by about 9.3% and improved worst-family P95 RMSE by about 0.8%, but
its global mean RMSE was 7.7% worse. Its constant mean RMSE improved by 0.6%,
while random-B-spline and random-ramp-dwell mean RMSE worsened by 9.5% and
20.9%. The lowest-error point improved global mean RMSE by about 20.6%, but had
the largest tail saturation in the grid.

Every trial config, stdout log, rollout, per-episode/per-family metrics, and
leaderboard is stored under
`logs/nffb/mixed_tuning/grid_v1_no_comp_seed668/`. `best_selection.yaml` records
the failed acceptance checks and comparison with the center;
`best_observed_policy.yaml` is always written for reproducibility. Because this
is a single-seed local search and no point passed all gates, that policy is not
promoted to `baselines/configs/` and should not be described as a validated
unified controller.

### Earlier compensation-enabled coarse-v1 result

The completed single-seed validation used the current 10-second mixed
environment, `seed=667`, and 192 parallel episodes. No candidate passed every
safety gate, so the workflow deliberately did not create
`baselines/configs/nffb_unified_mixed_coarse.yaml`.

The safety-ranked provisional candidate retained the generic filter/inner
gains and used:

```text
omega_o = 0.85 rad/s, zeta_o = 1.0, alpha = 0
omega_f = 2.5 rad/s, zeta_f = 1.0
k_theta = 4.0 1/s, omega_max = 0.5 rad/s
```

Its response inverse remained enabled with the nominal midpoint
`0.155 / 0.86 / -0.0035`. The validation result was:

| Metric | Provisional | Generic | Sine-tuned |
| --- | ---: | ---: | ---: |
| Completed episodes | 192 | 192 | 192 |
| Terminations | 0 | 0 | 192 |
| Safe-margin violations | 2 | 5 | 121 |
| Mean RMSE | 0.06358 m | 0.06106 m | 0.14793 m |
| Worst-family P95 RMSE | 0.17546 m | 0.16736 m | 0.37462 m |
| Full-run maximum saturation | 2.78% | 2.72% | 97.61% |
| Final-quarter maximum saturation | 6.02% | 5.99% | 99.74% |

The provisional candidate covered 62 constant, 59 random-B-spline, and 71
random-ramp-dwell episodes. It is substantially safer than the sine-specific
policy and reduces its mean RMSE by about 57%, but it does not improve generic
tracking error and fails both the zero-safe-margin-violation and 1% tail-
saturation requirements. Treat `final_selection.yaml` as a negative coarse-
search report, not as a deployable selected policy. This result motivated the
independent no-compensation rerun above.

## Evaluation

Run a short delay-free smoke test:

```bash
python3 scripts/nffb_policy_eval.py \
  --config baselines/configs/nffb_unified_tracking_eval.yaml \
  --env_config environments/configs/unified_tracking_mixed.yaml \
  --policy_config baselines/configs/nffb_sine_phase0_acc5.yaml \
  --episodes 4 \
  --num_envs 4 \
  --run_name nffb_unified_sine_smoke \
  --headless
```

Run the predictor-aware `D=8` example:

```bash
python3 scripts/nffb_policy_eval.py \
  --config baselines/configs/nffb_unified_tracking_predictor_eval.yaml \
  --env_config environments/configs/unified_tracking_sine.yaml \
  --episodes 2 \
  --num_envs 2 \
  --headless
```

Run deterministic response-compensation checks without and with delay:

```bash
python3 scripts/nffb_policy_eval.py \
  --config baselines/configs/nffb_unified_tracking_response_compensation_eval.yaml \
  --episodes 2 --num_envs 2 --headless

python3 scripts/nffb_policy_eval.py \
  --config baselines/configs/nffb_unified_tracking_predictor_response_compensation_eval.yaml \
  --episodes 2 --num_envs 2 --headless
```

Run the current delayed deterministic singleton families:

```bash
RUN_CONFIG=baselines/configs/nffb_unified_tracking_predictor_eval.yaml \
REFERENCE_TYPES="constant sine triangle trapezoid" \
TARGET_EPISODES=1000 NUM_ENVS=10 \
  scripts/run_nffb_policy_eval_configs.sh --headless
```

The predictor-disabled run config must be paired with delay-free environment
configs.

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
- predicted 11-D plant state, predictor reference horizon, and command-sync
  error;
- compensator nominal/execution states, desired response output, raw/limited
  inverse input, reconstructed output, tracking error, parameter source, and
  saturation flags;
- reference, ball-acceleration, filter-angle, filter-rate, velocity,
  acceleration, and anti-windup saturation flags.

## Predictor limitations

The delayed prediction and inverse compensate the deterministic nominal model
only. They do not replay or invert Gaussian/OU response noise, mirror
reset-time per-environment randomized response parameters, model the low-level
SE(3) velocity loop, or advance the NFFB internal filter `D` times. Random
`tau/gain/bias` ranges therefore require explicit nominal predictor and
compensation values. Existing delay-free tuning results describe the original
delay-free plant and must not be interpreted as performance validation for the
delayed, response-aware configuration.
