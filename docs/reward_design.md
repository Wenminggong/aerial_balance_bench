# Reward Design

This document lists the reward parameters used by Aerial-Balance-Bench and maps the notation in the paper to the implementation fields in the code. The task-specific weights are defined in `TargetPositionTaskCfg`, `TrajectoryTrackingTaskCfg`, and `UnifiedTrackingTaskCfg`; the shared beam-angle limit `theta_max` is defined by `AerialBalanceEnvCfg.max_theta`.

## Target-Position Balancing

The target-position reward uses the ball-position error

```text
e_k = p_b(t_k) - p_g(t_k)
```

and is implemented as:

```text
r = r_object + r_control + r_failure + r_goal

r_object = -k_1 e_k^2 - k_2 v_b^2
r_control = -k_3 u_k^2 - k_4 a_k^2
r_failure = -k_5, if |theta| > theta_max or |e_k| > e_max; otherwise 0
r_goal = (c - k_7 |e_k|) exp(-k_6 |v_b|), if |e_k| < e_goal; otherwise 0
```

| Paper parameter | Code field | Default value | Source |
| --- | --- | ---: | --- |
| `k_1` | `position_weight` | `5.0` | `TargetPositionTaskCfg` |
| `k_2` | `ball_velocity_weight` | `0.5` | `TargetPositionTaskCfg` |
| `k_3` | `command_weight` | `0.5` | `TargetPositionTaskCfg` |
| `k_4` | `action_weight` | `1.0` | `TargetPositionTaskCfg` |
| `k_5` | `failure_penalty` | `500.0` | `TargetPositionTaskCfg` |
| `k_6` | `goal_velocity_decay` | `1.0` | `TargetPositionTaskCfg` |
| `k_7` | `goal_position_decay` | `60.0` | `TargetPositionTaskCfg` |
| `theta_max` | `max_theta` | `50 deg` (`0.8727 rad`) | `AerialBalanceEnvCfg` |
| `e_max` | `max_error_for_failure` | `0.7` | `TargetPositionTaskCfg` |
| `e_goal` | `goal_radius` | `0.05` | `TargetPositionTaskCfg` |
| `c` | `goal_bonus` | `5.0` | `TargetPositionTaskCfg` |

In code, `u_k` is the interface-specific `command_z`, and `a_k` is the high-level action increment. For the velocity interface, these correspond to the commanded vertical velocity and its increment.

## Trajectory Tracking

The trajectory-tracking reward uses the ball-position error and reference velocity:

```text
e_k = p_b(t_k) - p_g(t_k)
v_g,k = reference velocity at t_k
```

and is implemented as:

```text
r = r_object + r_control + r_failure + r_progress

r_object = -hat{k}_1 e_k^2 - hat{k}_2 (v_b - v_g,k)^2
r_control = -hat{k}_3 u_k^2 - hat{k}_4 a_k^2
r_failure = -hat{k}_5, if |theta| > theta_max or |e_k| > e_max; otherwise 0
r_progress = hat{k}_6 (|e_{k-1}| - |e_k|)
```

| Paper parameter | Code field | Default value | Source |
| --- | --- | ---: | --- |
| `hat{k}_1` | `position_weight` | `5.0` | `TrajectoryTrackingTaskCfg` |
| `hat{k}_2` | `velocity_weight` | `0.5` | `TrajectoryTrackingTaskCfg` |
| `hat{k}_3` | `command_weight` | `0.5` | `TrajectoryTrackingTaskCfg` |
| `hat{k}_4` | `action_weight` | `1.0` | `TrajectoryTrackingTaskCfg` |
| `hat{k}_5` | `failure_penalty` | `500.0` | `TrajectoryTrackingTaskCfg` |
| `hat{k}_6` | `progress_weight` | `1.0` | `TrajectoryTrackingTaskCfg` |
| `theta_max` | `max_theta` | `50 deg` (`0.8727 rad`) | `AerialBalanceEnvCfg` |
| `e_max` | `max_error_for_failure` | `0.5` | `TrajectoryTrackingTaskCfg` |

## Unified Reference Tracking

The `unified_tracking` task represents constant goals, sine, triangle, trapezoid, random B-spline, and random ramp-dwell trajectories with one reward. It uses both position and reference-velocity error:

```text
e_p,k = p_b(t_k) - p_g(t_k)
e_v,k = v_b(t_k) - v_g(t_k)
```

and is implemented as:

```text
r = r_object + r_control + r_progress + r_failure

r_object = -k_u1 e_p,k^2 - k_u2 e_v,k^2
r_control = -k_u3 u_k^2 - k_u4 a_k^2
r_progress = k_u6 (|e_p,k-1| - |e_p,k|)
r_failure = -k_u5, if the environment reports termination; otherwise 0
```

The environment termination mask includes task error beyond `e_max`, beam angle beyond `theta_max`, and the ball leaving the feasible beam interval. A time-limit truncation alone does not receive the failure penalty.

| Parameter | Code field | Default value | Source |
| --- | --- | ---: | --- |
| `k_u1` | `position_weight` | `5.0` | `UnifiedTrackingTaskCfg` |
| `k_u2` | `velocity_weight` | `0.5` | `UnifiedTrackingTaskCfg` |
| `k_u3` | `command_weight` | `0.5` | `UnifiedTrackingTaskCfg` |
| `k_u4` | `action_weight` | `1.0` | `UnifiedTrackingTaskCfg` |
| `k_u5` | `failure_penalty` | `500.0` | `UnifiedTrackingTaskCfg` |
| `k_u6` | `progress_weight` | `1.0` | `UnifiedTrackingTaskCfg` |
| `theta_max` | `max_theta` | `50 deg` (`0.8727 rad`) | `AerialBalanceEnvCfg` |
| `e_max` | `max_error_for_failure` | `0.5` | `UnifiedTrackingTaskCfg` |

For a constant reference, `v_g = 0`, so the velocity term naturally becomes a regulation penalty on `v_b`. There is no constant-only near-goal bonus; all sampled reference types receive the same objective. `previous_abs_error` is reset independently for each environment, including partial autoresets.

This new reward does not replace or modify either legacy reward. Existing `target_position` configs retain their goal bonus, and existing `trajectory_tracking` configs retain their original failure-mask behavior.


## Custom Configuration

All reward weights can be customized through the corresponding task section in the configuration file. See [Unified Reference Tracking](unified_reference_tracking.md) for the complete mixed-task configuration and observation-preview contract.
