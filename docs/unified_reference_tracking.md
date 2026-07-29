# Unified Reference Tracking

This document specifies the `unified_tracking` task, its reference generator, reset distributions, observation preview, reward, evaluator, configuration files, and compatibility guarantees. The implementation treats target-position balancing as constant-reference tracking and exposes two additional random families for held-out generalization evaluation.

## 1. Task model

For environment `i`, the desired ball position is

```text
pg_i(t) = c_i + A_i f_type_i(2 pi t / T_i + phi_i)
```

where `c` is the center, `A` is the amplitude, `T` is the period, and `phi` is the initial phase. The supported reference types and stable IDs are:

| Type | ID | Reference form | Notes |
| --- | ---: | --- | --- |
| `sine` | `0` | `sin(x)` | Smooth periodic reference. |
| `triangle` | `1` | Piecewise-linear triangle wave in `[-1, 1]` | Continuous position with slope changes at extrema. |
| `trapezoid` | `2` | `clip(2 triangle(x), -1, 1)` | Alternates between constant plateaus and linear ramps. |
| `constant` | `3` | `0` | `A` is forced to zero, so `pg(t) = c`. |
| `random_b_spline` | `4` | Clamped B-spline control polygon | Smooth, non-periodic reference with frequency-matched alternating extrema. |
| `random_ramp_dwell` | `5` | Random linear ramps and constant dwells | Non-periodic mixture of continuous-ramp and ramp-dwell profiles. |

The analytic expression above applies to the periodic and constant families.
The two random families use per-environment buffers sampled at reset. The type
ID mapping is part of the rollout and metric contract; IDs `0` through `3`
remain unchanged.

The triangle wave exactly follows the legacy trajectory generator. With `q = remainder(x + pi/2, 2 pi)`:

```text
triangle(x) = 2 q / pi - 1,  q < pi
              3 - 2 q / pi, q >= pi
```

The configured phase is added to `2 pi t / T` before evaluating any dynamic wave.

Reference velocity is the one-control-step forward finite difference used by the existing trajectory task:

```text
vg(t_k) = (pg(t_k + dt) - pg(t_k)) / dt
```

`dt` is the environment control period (`decimation * sim.dt`), not the raw physics step. Consequently, constant references always have `vg = 0`; triangle, trapezoid, and ramp-dwell velocities reflect their discrete slope and corner behavior. Random references are generated through the full episode, configured preview horizon, and the additional forward-difference sample needed by the final preview velocity. They therefore continue naturally past the configured base duration instead of holding its terminal value inside the observable horizon.

## 2. Sampling and reset behavior

`trajectory_types` selects the eligible families. A list with one entry creates a deterministic task family for evaluation; multiple entries enable per-environment mixed sampling. `trajectory_type_weights` has the same order as `trajectory_types`. An empty list means uniform probabilities; a non-empty list is normalized internally before categorical sampling.

Every reset samples and stores one type and its parameters per selected environment. Partial reset only changes the buffers belonging to `env_ids`; the type, parameters, initial position, and episode clock of every other environment remain unchanged.

### Reference parameter fields

| Field | Supplied default | Meaning |
| --- | --- | --- |
| `constant_goal_range` | `[0.10, 0.60]` | Uniform range for the center/goal of a constant reference. |
| `dynamic_center_range` | `[0.25, 0.45]` | Uniform center range for sine, triangle, and trapezoid references. |
| `amplitude_range` | `[0.05, 0.20]` | Uniform amplitude range for dynamic references. Constant references force amplitude to zero. |
| `period_range` | `[4.0, 10.0]` | Uniform period range in seconds for dynamic references. |
| `phase_range` | `[0, 2 pi]` | Uniform initial phase range in radians for dynamic references. |

For a sampled constant reference, the logged center is the sampled goal and the implementation stores the canonical placeholders `amplitude = 0`, `period = 1`, and `phase = 0`.

### Random B-spline fields

| Field | Default | Meaning |
| --- | --- | --- |
| `random_b_spline_sampling_mode` | `paired_alternating_extrema` | Frequency-matched paired extrema; use `uniform` for the legacy independent-control sampler. |
| `random_b_spline_degree` | `3` | B-spline degree; must be at least one and smaller than the control-point count. |
| `random_b_spline_num_control_points` | `12` | Fixed control-point count; the default leaves ten internal controls for five extrema pairs. |
| `random_b_spline_duration_s` | `20.0` | Configured base duration; the knot-vector duration is extended when the required reference horizon is longer. |
| `random_b_spline_position_range` | `[0.10, 0.60]` | Full control-point envelope and the legacy `uniform` sampling range. |
| `random_b_spline_extrema_ranges` | `[[0.10, 0.30], [0.40, 0.60]]` | Ordered low/high ranges used by paired alternating extrema. |
| `random_b_spline_start_position` | `0.35` | Fixed first control point and exact initial reference. |
| `random_b_spline_end_position` | `0.35` | Fixed final control point at the end of the generated reference horizon. |

In the default mode, each sampled internal extremum is copied into two adjacent
control points. The first pair randomly uses the low or high range and later
pairs alternate ranges, while their magnitudes remain independent. With the
20-second base duration, the resulting main reversal scale is approximately
4--5 seconds, matching half of the 8--10 second target period without directly
sampling a sine wave. The generator still evaluates the open-uniform clamped
B-spline directly in PyTorch, so its convex-hull bound and smoothness are
preserved. `uniform` retains the previous independently sampled controls.

### Random ramp-dwell fields

| Field | Default | Meaning |
| --- | --- | --- |
| `random_ramp_dwell_sampling_mode` | `half_cycle_mixture` | Frequency-matched episode profiles; use `independent` for the legacy duration sampler. |
| `random_ramp_dwell_num_segments` | `6` | Fixed number of ramp/dwell pairs. |
| `random_ramp_dwell_duration_s` | `20.0` | Configured base duration; generation is extended when the required reference horizon is longer. |
| `random_ramp_dwell_start_position` | `0.35` | Initial reference position. |
| `random_ramp_dwell_target_ranges` | `[[0.10, 0.30], [0.40, 0.60]]` | Ordered low/high target bands; the default mode alternates between them. |
| `random_ramp_dwell_half_cycle_duration_range` | `[4.0, 5.0]` | Independently sampled interval between successive extrema. |
| `random_ramp_dwell_continuous_ramp_probability` | `0.5` | Probability that an episode uses only continuous ramps and zero dwell. |
| `random_ramp_dwell_ramp_fraction_range` | `[0.45, 0.55]` | Fraction of each half-cycle occupied by the ramp in a ramp-dwell episode. |
| `random_ramp_duration_range` | `[1.0, 3.0]` | Legacy `independent` mode ramp-duration range. |
| `random_dwell_duration_range` | `[0.0, 4.0]` | Legacy `independent` mode dwell-duration range. |

The default mode samples one profile for the whole episode. Continuous-ramp
episodes use `ramp = half_cycle` and `dwell = 0`; ramp-dwell episodes split each
half-cycle using the sampled ramp fraction. Because the reference begins at the
center rather than an extremum, the first ramp is half of its normal duration.
Targets alternate low/high, but their positions and half-cycle durations are
sampled independently, so neither profile is periodic. Half-cycle durations are
extended only within their configured range when required to cover the complete
reference horizon; an impossible horizon fails fast. Six default segments cover
at least 22 seconds. `independent` retains the previous independently sampled
ramp/dwell durations and target bands.

Setting both ends of any range to the same value makes that parameter fixed. For example, a fixed sine experiment can use:

```yaml
unified_tracking_task:
  trajectory_types: [sine]
  trajectory_type_weights: [1.0]
  dynamic_center_range: [0.35, 0.35]
  amplitude_range: [0.10, 0.10]
  period_range: [6.0, 6.0]
  phase_range: [0.0, 0.0]
```

The configuration is rejected when it contains an unknown/duplicate type, invalid or negative weights, reversed ranges, a non-positive period, an infeasible initial-distance request, or a dynamic `center +/- amplitude` envelope outside the configured beam range.

### Initial-state modes

Constant and dynamic references have independent mode selectors, `constant_initialization_mode` and `dynamic_initialization_mode`:

| Mode | Initial ball position |
| --- | --- |
| `fixed` | `fixed_initial_ball_position` |
| `on_reference` | The sampled `pg(0)` for that environment |
| `independent_uniform` | A sample from `initial_ball_position_range` |

For `independent_uniform`, `min_initial_reference_distance` defines the minimum allowed `|pb(0) - pg(0)|`. The supplied configs use `fixed_initial_ball_position: 0.35`, `initial_ball_position_range: [0.10, 0.60]`, and minimum distance `0.05`. The mixed-training config uses `independent_uniform` for constant references and `on_reference` for dynamic references. This reproduces the important distinction between regulation from a nonzero initial error and motion tracking without coupling the reference generator to the physical reset implementation.

`on_reference` aligns position only. The physical reset still initializes ball velocity to zero, so a dynamic reference with nonzero `vg(0)` begins with a velocity-tracking error even though `pb(0) = pg(0)`.

Environment seeding controls task sampling. Reproducibility therefore requires the same seed, number of parallel environments, and reset sequence.

## 3. Reward and termination

All six reference types use the same reward:

```text
ep_k = pb_k - pg_k
ev_k = vb_k - vg_k

r_k = -position_weight * ep_k^2
      -velocity_weight * ev_k^2
      -command_weight * command_z_k^2
      -action_weight * action_k^2
      +progress_weight * (|ep_{k-1}| - |ep_k|)
      -failure_penalty * I[terminated_k]
```

`previous_abs_error` is initialized from the newly sampled physical/reference state and maintained per environment. The final failure term uses the environment's complete termination mask, so task error, beam-angle violation, and the ball leaving the beam receive the same penalty. A timeout is not a failure unless another termination condition is also true.

Task-specific termination occurs when

```text
|pb - pg| > max_error_for_failure
```

The environment still applies its existing beam-angle, beam-boundary, and episode-time-limit conditions. There is no constant-only goal bonus: a constant target is trained with exactly the same tracking objective as a moving reference. The two legacy tasks keep their original rewards and termination semantics.

## 4. Reference preview observation

Reference preview is configured independently of the task:

```yaml
reference_preview:
  enabled: true
  future_steps: 30
```

With preview disabled, the raw observation is the unchanged 11-D benchmark layout:

```text
[pb, vb, ab, theta, omega, alpha, drz, vrz, arz, pg, a_prev]
```

With preview enabled for horizon `H`, the legacy prefix remains value-for-value and index-for-index compatible, and the following fields are appended:

```text
[pb, vb, ab, theta, omega, alpha, drz, vrz, arz, pg_0, a_prev,
 vg_0, pg_1, vg_1, ..., pg_H, vg_H]
```

The raw dimension is

```text
D_raw = 12 + 2 H
```

Thus `H = 0` adds only current reference velocity and produces 12 dimensions; the standard unified `H = 30` configs produce 72 dimensions. Offsets are consecutive control steps `0..H`. Preview continues beyond the episode horizon according to the sampled reference, so the last in-episode observation has a complete, fixed-size preview. Random-reference generation also includes one extra control step for the forward finite difference defining `vg_H`.

The environment exposes the resolved `observation_fields`, `raw_observation_dim`, and `reference_preview_offsets`. Evaluation rollout archives store `observation_fields`, and visualization discovers `pg_*`/`vg_*` series from that metadata rather than assuming 11 dimensions. `resolved_run.yaml` records `raw_observation_dim`, `observation_fields`, `policy_input_dim`/`policy_input_fields` for RL, preview enabled/horizon/offsets, `trajectory_type_to_id`, configured `trajectory_types`, and `normalized_trajectory_type_weights`.

### RL adapter

The full-preview policy mode is:

```yaml
observation_mode: reference_preview
```

It produces this policy input:

```text
[pb, vb, ab, theta, omega, alpha, drz, vrz, arz, a_prev,
 pg_0, vg_0, pg_1, vg_1, ..., pg_H, vg_H]
```

The adapter obtains the raw dimension and field names from the environment; the policy YAML does not duplicate `H`. Existing `legacy8` and `full11` modes accept an extended raw observation but consume only the original 11-D prefix, preserving their network dimensions and old checkpoint behavior.

The relative, uniformly sampled policy mode is configured with:

```yaml
observation_mode: relative_reference_preview
reference_preview_samples: 10
```

For `K = reference_preview_samples`, it produces:

```text
[pb - pg_0, vb - vg_0, ab, theta, omega, alpha, vrz, arz, a_prev,
 pg_s - pg_0, pg_2s - pg_0, ..., pg_H - pg_0,
 vg_s - vg_0, vg_2s - vg_0, ..., vg_H - vg_0]
```

where `s = H / K` in the delay-free case. `K` must be a positive integer,
`H >= K`, and `H` must be divisible by `K`; invalid combinations fail during
adapter construction. The standard training contract `D = 0`, `H = 30`,
`K = 10` samples relative offsets `(3, 6, ..., 30)` and produces
`9 + 2K = 29` inputs. Position deltas form one contiguous block followed by
the velocity-delta block.

During predictor-enabled evaluation, `H_raw` denotes the environment preview
horizon and `D` is the active predictor/environment delay. The adapter uses

```text
H_effective = H_raw - D
s = H_effective / K
relative offsets = (s, 2s, ..., H_effective)
raw source offsets = (D+s, D+2s, ..., H_raw)
```

The predictor advances the plant state and legacy `PG` field to `k + D` using
`pg_0 ... pg_D`; it does not roll or rewrite the appended preview. The adapter
then uses predicted `pg_D` and raw `vg_D` as its bases and reads each sampled
`pg`/`vg` pair from its raw source offset. For `D=8`, `H_raw=38`, and `K=10`,
the relative field names remain `delta_pg_3 ... delta_pg_30` and
`delta_vg_3 ... delta_vg_30`, while the raw sources are offsets
`11, 14, ..., 38`. This is exactly the same 29-D network input contract as
delay-free `D=0`, `H=30`, `K=10` training.

The checkpoint contract therefore consists of the observation mode, effective
horizon, `K`, relative offsets, input dimension, and field order. Raw horizon
may increase from `H_effective` to `D + H_effective` during compensated
evaluation. A mismatched raw spec/dimension or a non-divisible effective
horizon fails before checkpoint use. Resolved RL metadata records raw horizon,
base offset, effective horizon, relative policy offsets, raw source offsets,
and predictor-consumed `pg_0 ... pg_D` fields.

Only `relative_reference_preview` is supported with an active predictor, and
only in RL evaluation/deployment. The environment action delay must actually
be enabled. Fixed-delay evaluation requires the same `delay_step` as the
predictor; random-delay evaluation uses an explicitly configured fixed nominal
predictor horizon. Full
`reference_preview` plus predictor remains unsupported, and `rl_train.py`
continues to reject every active predictor.

### Autoreset timing

At a terminal step, evaluator aggregation and `extras["step"]` describe the episode that just ended. Isaac Lab then resets completed environments and returns an observation for each new episode. Consumers must not combine a terminal step's task parameters from `extras` with the autoreset observation as though they belonged to one episode.

## 5. Evaluation contract

The unified evaluator always exposes a stable metric schema:

| Group | Exact keys |
| --- | --- |
| Global | `completed_episodes`, `evaluation_complete`, `mean_absolute_error`, `root_mean_square_error`, `maximum_absolute_error` |
| Per type | `<type>_completed_episodes`, `<type>_mean_absolute_error`, `<type>_root_mean_square_error`, `<type>_maximum_absolute_error` for each of the six type names |
| Constant regulation | `constant_success_rate`, `constant_steady_state_error`, `constant_steady_state_error_std`, `constant_convergence_time`, `constant_convergence_time_std`, `constant_climbing_time`, `constant_climbing_time_std` |

Per-type position-error metrics are episode averages and are then averaged across completed episodes of that type. If a type has no completed episodes, its count is zero and its numeric metrics are `NaN`, not zero. A constant metric standard deviation is zero with one completed constant episode and `NaN` with none. This makes missing coverage visible in mixed runs.

Evaluator configuration uses `num_eval_episodes` for `evaluation_complete`, `target_zone` for constant-reference entry/success tests, and `steady_state_window_s` for the final rolling absolute-error window. `constant_climbing_time` is the first target-zone entry. `constant_convergence_time` is the start of the final uninterrupted in-zone interval; an episode is successful when such an interval exists before the time limit.

The mixed evaluator is useful for sampling/coverage checks and a training-distribution aggregate. Policy comparisons should use singleton configs so every reported run has an unambiguous reference family and controlled episode count. The supplied mixed config deliberately remains a four-family distribution; the two random singleton configs are held out for generalization tests.

Each step exposes these sampled task fields for logging and audit:

```text
trajectory_type_id
trajectory_center
trajectory_amplitude
trajectory_period
trajectory_phase
initial_ball_position
```

## 6. Supplied configurations

| File | Purpose |
| --- | --- |
| `environments/configs/unified_tracking_mixed.yaml` | Equal-probability mixed training over all four types. |
| `environments/configs/unified_tracking_constant.yaml` | Constant-reference evaluation. |
| `environments/configs/unified_tracking_sine.yaml` | Sine-reference evaluation. |
| `environments/configs/unified_tracking_triangle.yaml` | Triangle-reference evaluation. |
| `environments/configs/unified_tracking_trapezoid.yaml` | Trapezoid-reference evaluation. |
| `environments/configs/unified_tracking_random_b_spline.yaml` | Frequency-matched, non-periodic smooth B-spline reference. |
| `environments/configs/unified_tracking_random_ramp_dwell.yaml` | Frequency-matched, non-periodic continuous-ramp/ramp-dwell mixture. |
| `environments/configs/unified_tracking_<family>_delay_d8_h38.yaml` | Six delay-only singleton evaluations with `D=8`, raw `H=38`, response disabled. |
| `baselines/configs/rl_rpo_reference_preview_train.yaml` | RPO training policy using preview observations. |
| `baselines/configs/rl_rpo_reference_preview_eval.yaml` | Deterministic preview-policy evaluation template. |
| `baselines/configs/rl_rpo_relative_reference_preview_train.yaml` | RPO training policy using the 29-D relative sampled preview. |
| `baselines/configs/rl_rpo_relative_reference_preview_eval.yaml` | Deterministic relative-preview evaluation template. |
| `baselines/configs/rl_rpo_relative_reference_preview_predictor_eval.yaml` | Deterministic relative-preview evaluation with the delay predictor active. |
| `baselines/configs/rl_unified_tracking_rpo_train.yaml` | Complete mixed RPO training run. |
| `baselines/configs/rl_unified_tracking_rpo_eval.yaml` | Complete RL evaluation run; switch singleton env with `--env_config`. |
| `baselines/configs/rl_unified_tracking_rpo_predictor_eval.yaml` | Complete compensated RL evaluation run for `D=8`, raw `H=38`. |
| `baselines/configs/cpid_unified_tracking_eval.yaml` | Unchanged CPID evaluation on the mixed task; switch singleton env with `--env_config`. |
| `baselines/configs/cpid_unified_tracking_predictor_eval.yaml` | CPID with an eight-step response-aware predictor and aligned reference preview. |
| `baselines/configs/nffb_unified_tracking_eval.yaml` | Delay-free NFFB evaluation run; switch singleton env with `--env_config`. |
| `baselines/configs/nffb_unified_tracking_predictor_eval.yaml` | NFFB with an eight-step response-aware predictor and `D+1` velocity preview. |
| `baselines/configs/nffb_sine_tuning.yaml` | Resumable phase-zero sine bandwidth search, stress tests, ablation, and validation. |

The mixed config and all six standard singleton configs use `H=30`, matching
the supplied relative-preview RL templates with `K=10`. The deterministic
constant/sine/triangle/trapezoid configs retain their `D=8` action-delay and
velocity-response robustness settings. Dedicated response/predictor configs
keep their own horizons, including the CPID triangle example with `H=D=8`.

### Plant-matched velocity response

The enabled velocity-response path is a feed-forward lead-lag stage rather
than an additional first-order lag in series with the existing controller.
The sampled range fields describe the requested final model

```text
G_real(s) = K_r / (tau_r s + 1), with output bias b_r,
```

while the fixed simulator fields describe the existing low-level response to
be inverted:

```yaml
velocity_response_sim_tau_s: 0.139
velocity_response_sim_gain: 1.0
velocity_response_sim_bias: -0.00055
velocity_response_sim_max_abs_velocity: 0.0
```

For control period `dt`, define `a_s = exp(-dt/tau_s)` and
`a_r = exp(-dt/tau_r)`. The target is advanced first, target-output noise and
the existing final-output limit are applied, and the stage sends

```text
u_ll[k] = (v_des[k+1] - a_s v_des[k]) / ((1-a_s) K_s) - b_s/K_s
```

to the velocity interface. Thus `velocity_response_executed_z` logs the
desired final output and `velocity_response_compensated_command_z` logs the
actual lead-lag command. Reset initializes the response output from the
measured `env.vrz`; partial reset touches only selected environments. Disabled
response remains a passthrough.

This inverse does not cancel the low-level controller's pure delay, and the
compensated command currently has no absolute-value or slew-rate limiter.
`velocity_response_sim_max_abs_velocity` must therefore remain `0.0`.
Controller-gain or mass randomization changes the physical plant, so one fixed
inverse can only be approximate in those cases. The previous semantics of
adding a standalone first-order response in series are no longer provided.

The CPID unified runner preserves the legacy 11-D controller input. When its
state predictor is inactive, CPID reacts to the current `pg` exactly as before.
When the predictor is active, the runner additionally supplies the named
position sequence `pg_0 ... pg_D` to the predictor. After predicting plant state
step \(j\), the tracking error is evaluated against `pg_j`, and the final
predicted observation contains `pg_D`. The reference velocities `vg_j` are not
used by the current model.

An active predictor on a moving-reference task requires
`reference_preview.enabled: true` and `reference_preview.future_steps >=
delay_step`. A shorter preview is rejected at startup; it is never extrapolated
or padded. Standard unified RL configs use `H=30`; the dedicated
`environments/configs/unified_tracking_triangle_predictor.yaml` example uses
`H=D=8`.

RL evaluation uses the same future-position path in `legacy8` and `full11`
observation modes. It also supports
`observation_mode=relative_reference_preview` by using predictor delay `D` as
the adapter's runtime-only base offset. The predictor consumes `pg_0 ... pg_D`
and advances the plant; the adapter independently rebases `pg`/`vg` at `D` and
samples the remaining horizon. `observation_mode=reference_preview` remains
unsupported with an active predictor.

The state predictor defaults to the nominal simulator response
`0.139 / 1.0 / -0.00055`, including for the six delay-only environment configs.
When the environment lead-lag stage is enabled, predictor parameters represent
the combined final deterministic response. Fixed, noise-free target ranges can
be resolved with `auto`, or supplied explicitly:

```yaml
policy_overrides:
  state_predictor:
    velocity_response_enabled: true
    velocity_response_tau_s: 0.155
    velocity_response_gain: 0.86
    velocity_response_bias: -0.0035
    velocity_response_max_abs_velocity: 0.0
```

Do not use `auto` for `tau/gain/bias` when enabled target ranges are randomized:
a single deterministic predictor cannot represent per-environment samples, and
startup validation rejects that collapse. If the environment response stage is
disabled, `auto` instead resolves the fixed `velocity_response_sim_*` scalars.

The predictor API is task-independent, so the environment preview generated
through either `TrajectoryTrackingTask.get_reference()` or
`UnifiedTrackingTask.get_reference_preview()` has the same `pg_0 ... pg_D`
contract. This change does not add legacy trajectory-tracking runners or NMPC
moving-task integration.

NFFB adds one extra reference requirement because its feedforward acceleration
uses a forward difference. With a delay of `D` steps it consumes the predicted
11-D plant state and `pg_D`, `vg_D`, `vg_{D+1}`, so the preview must satisfy
`H >= D + 1`. The delay-free path remains `D=0` and uses the original
`pg_0`, `vg_0`, and `vg_1` values. The state predictor, NFFB command filter,
integrator, and anti-windup state are not repeatedly advanced through the
delay horizon: the predictor alone rolls the plant forward, while the
controller states update once per actual policy cycle.

The NFFB runner resolves predictor geometry, mass, gravity, step time, command
limits, and delay from the environment and checks them against the resolved
NFFB model. Random velocity-response ranges cannot be collapsed silently:
the predictor-aware example supplies explicit nominal `tau/gain/bias` values.
Gaussian/OU noise and reset-time per-environment parameter samples remain
outside the predictor.

Mixed runs provide aggregate transfer and type-coverage evidence. Controller
comparisons should use the constant and dynamic singleton configs so that every
summary has a controlled reference family and episode count.

The NFFB tuning config deliberately specializes the sine singleton to
`phase=0`, `max_acc=5.0 m/s^2`, and `max_velocity=0`. It does not change the
reference generator or evaluator; generated fixed-profile configs exist only
under the ignored tuning log directory.

### Zero-action smoke tests

Run a small mixed smoke test:

```bash
python3 scripts/zero_action_policy_eval.py \
  --config environments/configs/unified_tracking_mixed.yaml \
  --episodes 4 \
  --num_envs 4 \
  --headless
```

Run all singleton smoke tests:

```bash
for type in constant sine triangle trapezoid random_b_spline random_ramp_dwell; do
  python3 scripts/zero_action_policy_eval.py \
    --config "environments/configs/unified_tracking_${type}.yaml" \
    --episodes 4 \
    --num_envs 4 \
    --run_name "zero_action_unified_${type}_smoke" \
    --headless
done
```

### Mixed RL training

```bash
python3 scripts/rl_train.py \
  --config baselines/configs/rl_unified_tracking_rpo_train.yaml \
  --num_envs 1024 \
  --max_iterations 300 \
  --headless
```

For a short construction/shape check, override `--num_envs 4 --max_iterations 1`.
Training uses `D=0`, `H=30`, and `K=10`; an active state predictor is rejected
by the training entry point.

### Per-type RL evaluation

Use the same preview checkpoint for every singleton config:

```bash
CHECKPOINT=/path/to/best_agent.pt
for type in constant sine triangle trapezoid; do
  python3 scripts/rl_policy_eval.py \
    --config baselines/configs/rl_unified_tracking_rpo_eval.yaml \
    --env_config "environments/configs/unified_tracking_${type}.yaml" \
    --checkpoint "$CHECKPOINT" \
    --episodes 1000 \
    --num_envs 10 \
    --run_name "rpo_unified_${type}_eval" \
    --headless
done
```

For eight-step delay compensation, switch both the run template and singleton
family configs while reusing the same 29-D checkpoint:

```bash
CHECKPOINT=/path/to/best_agent.pt
for type in constant sine triangle trapezoid random_b_spline random_ramp_dwell; do
  python3 scripts/rl_policy_eval.py \
    --config baselines/configs/rl_unified_tracking_rpo_predictor_eval.yaml \
    --env_config "environments/configs/unified_tracking_${type}_delay_d8_h38.yaml" \
    --checkpoint "$CHECKPOINT" \
    --episodes 1000 \
    --num_envs 10 \
    --run_name "rpo_unified_${type}_predictor_d8_eval" \
    --headless
done
```

Startup requires `robustness.enabled=true` and
`robustness.action_delay_enabled=true`. Fixed-delay evaluation requires matching
positive predictor and environment `delay_step` values. With non-empty random
`delay_step_choices`, configure a fixed nominal predictor `delay_step`; runtime
per-environment predictor horizons are intentionally unsupported. A delayed
environment with the predictor disabled is still accepted as the uncompensated
comparison baseline.

Use a new logging root/run name when changing metric schemas or experimental distributions. The CSV writer preserves an existing file's header, so reusing an unrelated legacy summary file could omit newly introduced columns.

`scripts/summarize_rl_eval_summary.py` discovers the metric groups that are present. Unified per-type aggregates are weighted by their corresponding `<type>_completed_episodes`; target-position pooled metrics remain supported.

## 7. Compatibility and limitations

| Component | Legacy tasks | `unified_tracking` | Reference preview |
| --- | --- | --- | --- |
| Environment | Unchanged | Supported | Optional; disabled by default |
| Zero-action runner | Supported | Supported | Supported |
| RL train/eval | Supported as before | Supported with velocity interface | `relative_reference_preview`, `reference_preview`, `legacy8`, and `full11` adapters supported |
| Existing RL checkpoints | Unchanged with their original mode/config | Not automatically transferable | Full preview requires matching raw `H`; relative preview requires matching effective `H`, `K`, offsets, and fields |
| CPID runner/policy | Supported for its existing task paths | Supported by the dedicated unified runner | Future positions consumed only by an active predictor |
| NMPC runner/policy | Supported for its existing task paths | Not supported | Not modified |
| NFFB runner/policy | Not exposed on legacy tasks | Supported with velocity interface | Delay-free requires `H >= 1`; delayed prediction requires `H >= D + 1` |
| Velocity state predictor | Existing combinations unchanged; task-independent future-reference API available | `pg_0 ... pg_D` supported in CPID, NFFB, RL legacy modes, and RL relative-preview evaluation | NFFB additionally consumes `vg_D` and `vg_{D+1}`; RL relative preview rebases at `D`; full RL preview remains unsupported |

The standard unified mixed/singleton YAML files use `H=30`; the six RL delay-only configs use `D=8`, raw `H=38`, and the same effective 30-step policy horizon. Specialized response/predictor configs retain their existing horizons. `target_position`, `trajectory_tracking`, the 11-D default observation, legacy evaluator keys, existing adapter modes, and legacy checkpoint input dimensions remain intact.
