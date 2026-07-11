# Unified Reference Tracking

This document specifies the `unified_tracking` task, its reference generator, reset distributions, observation preview, reward, evaluator, configuration files, and compatibility guarantees. The implementation treats target-position balancing as constant-reference tracking while keeping the original `target_position` and `trajectory_tracking` tasks unchanged.

## 1. Task model

For environment `i`, the desired ball position is

```text
pg_i(t) = c_i + A_i f_type_i(2 pi t / T_i + phi_i)
```

where `c` is the center, `A` is the amplitude, `T` is the period, and `phi` is the initial phase. The supported reference types and stable IDs are:

| Type | ID | Normalized wave `f(x)` | Notes |
| --- | ---: | --- | --- |
| `sine` | `0` | `sin(x)` | Smooth periodic reference. |
| `triangle` | `1` | Piecewise-linear triangle wave in `[-1, 1]` | Continuous position with slope changes at extrema. |
| `trapezoid` | `2` | `clip(2 triangle(x), -1, 1)` | Alternates between constant plateaus and linear ramps. |
| `constant` | `3` | `0` | `A` is forced to zero, so `pg(t) = c`. |

The type ID mapping is part of the rollout and metric contract. Do not renumber existing IDs when adding another reference family.

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

`dt` is the environment control period (`decimation * sim.dt`), not the raw physics step. Consequently, constant references always have `vg = 0`; triangle and trapezoid velocities reflect their discrete slope and corner behavior.

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

All four reference types use the same reward:

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
  future_steps: 5
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

Thus `H = 0` adds only current reference velocity and produces 12 dimensions; the supplied `H = 5` configs produce 22 dimensions. Offsets are consecutive control steps `0..H`. Preview continues beyond the episode horizon according to the sampled analytic reference, so the last in-episode observation has a complete, fixed-size preview.

The environment exposes the resolved `observation_fields`, `raw_observation_dim`, and `reference_preview_offsets`. Evaluation rollout archives store `observation_fields`, and visualization discovers `pg_*`/`vg_*` series from that metadata rather than assuming 11 dimensions. `resolved_run.yaml` records `raw_observation_dim`, `observation_fields`, `policy_input_dim`/`policy_input_fields` for RL, preview enabled/horizon/offsets, `trajectory_type_to_id`, configured `trajectory_types`, and `normalized_trajectory_type_weights`.

### RL adapter

The new policy mode is:

```yaml
observation_mode: reference_preview
```

It produces this policy input:

```text
[pb, vb, ab, theta, omega, alpha, drz, vrz, arz, a_prev,
 pg_0, vg_0, pg_1, vg_1, ..., pg_H, vg_H]
```

The adapter obtains the raw dimension and field names from the environment; the policy YAML does not duplicate `H`. Existing `legacy8` and `full11` modes accept an extended raw observation but consume only the original 11-D prefix, preserving their network dimensions and old checkpoint behavior.

Preview horizon is part of the checkpoint input contract. Training and evaluation must use the same `H` and field order. A mismatched raw spec/dimension produces an explicit error instead of silently loading an incompatible network. All supplied mixed and singleton configs use `H = 5`.

`reference_preview` policy mode cannot be combined with an active velocity-model state predictor (`enabled: true` with `delay_step > 0`). The current predictor assumes the legacy state layout and does not propagate a moving reference over its delay horizon; this unsupported combination fails fast. Predictor-disabled preview is the supported configuration in this release.

### Autoreset timing

At a terminal step, evaluator aggregation and `extras["step"]` describe the episode that just ended. Isaac Lab then resets completed environments and returns an observation for each new episode. Consumers must not combine a terminal step's task parameters from `extras` with the autoreset observation as though they belonged to one episode.

## 5. Evaluation contract

The unified evaluator always exposes a stable metric schema:

| Group | Exact keys |
| --- | --- |
| Global | `completed_episodes`, `evaluation_complete`, `mean_absolute_error`, `root_mean_square_error`, `maximum_absolute_error` |
| Per type | `<type>_completed_episodes`, `<type>_mean_absolute_error`, `<type>_root_mean_square_error`, `<type>_maximum_absolute_error` for each of the four type names |
| Constant regulation | `constant_success_rate`, `constant_steady_state_error`, `constant_steady_state_error_std`, `constant_convergence_time`, `constant_convergence_time_std`, `constant_climbing_time`, `constant_climbing_time_std` |

Per-type position-error metrics are episode averages and are then averaged across completed episodes of that type. If a type has no completed episodes, its count is zero and its numeric metrics are `NaN`, not zero. A constant metric standard deviation is zero with one completed constant episode and `NaN` with none. This makes missing coverage visible in mixed runs.

Evaluator configuration uses `num_eval_episodes` for `evaluation_complete`, `target_zone` for constant-reference entry/success tests, and `steady_state_window_s` for the final rolling absolute-error window. `constant_climbing_time` is the first target-zone entry. `constant_convergence_time` is the start of the final uninterrupted in-zone interval; an episode is successful when such an interval exists before the time limit.

The mixed evaluator is useful for sampling/coverage checks and a training-distribution aggregate. Policy comparisons should use the four singleton configs so every reported run has an unambiguous reference family and controlled episode count.

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
| `baselines/configs/rl_rpo_reference_preview_train.yaml` | RPO training policy using preview observations. |
| `baselines/configs/rl_rpo_reference_preview_eval.yaml` | Deterministic preview-policy evaluation template. |
| `baselines/configs/rl_unified_tracking_rpo_train.yaml` | Complete mixed RPO training run. |
| `baselines/configs/rl_unified_tracking_rpo_eval.yaml` | Complete RL evaluation run; switch singleton env with `--env_config`. |

The mixed and singleton configs share the same dynamic-parameter ranges and `H = 5` layout. They differ only in eligible reference types, evaluation scale, and log names.

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
for type in constant sine triangle trapezoid; do
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

Use a new logging root/run name when changing metric schemas or experimental distributions. The CSV writer preserves an existing file's header, so reusing an unrelated legacy summary file could omit newly introduced columns.

`scripts/summarize_rl_eval_summary.py` discovers the metric groups that are present. Unified per-type aggregates are weighted by their corresponding `<type>_completed_episodes`; target-position pooled metrics remain supported.

## 7. Compatibility and limitations

| Component | Legacy tasks | `unified_tracking` | Reference preview |
| --- | --- | --- | --- |
| Environment | Unchanged | Supported | Optional; disabled by default |
| Zero-action runner | Supported | Supported | Supported |
| RL train/eval | Supported as before | Supported with velocity interface | `reference_preview`, `legacy8`, and `full11` adapters supported |
| Existing RL checkpoints | Unchanged with their original mode/config | Not automatically transferable | Preview checkpoint requires matching `H` |
| CPID runner/policy | Supported for its existing task paths | Not supported | Not modified |
| NMPC runner/policy | Supported for its existing task paths | Not supported | Not modified |
| Velocity state predictor | Existing legacy combinations unchanged | Legacy-layout behavior only | Not supported with `reference_preview` mode |

No existing environment YAML is migrated. `target_position`, `trajectory_tracking`, the 11-D default observation, legacy evaluator keys, and legacy checkpoint input dimensions remain intact.
