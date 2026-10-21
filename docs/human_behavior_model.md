# Human endpoint behavior model

The beam endpoint previously driven by a pure OU disturbance is now driven by
an independent, vectorized human motion model. The existing vertical
`slider_holder_joint` represents the hand. No human body asset is required.
The model is implemented entirely in this repository and does not import the
historical Human_Drone_Balance project or any drone baseline controller.

## Protocol change and compatibility

`robustness.enabled && robustness.external_disturbance_enabled` enables human
motion. Both switches retain their existing meaning as the gate for endpoint
motion. `human_behavior_type` defaults to `random-walk`; `random` is an alias.
The other modes are `cooperative` and `disturbed`. Mode names are case-insensitive.
One mode is selected for the run, with independent parameters per environment,
resampled only when that environment resets. Runtime mode switching is not
supported.

**This replaces the historical pure OU protocol.** Old YAML files remain
loadable, but active disturbances now include position recentering, acceleration
limits, and control-rate OU updates instead of physics-rate OU updates. Old and
new disturbance results must not be treated as the same experimental condition.
Keep the source revision, input YAML, seed, and rollout diagnostics with results.

**2026-10-16: shared OU configuration for all human modes.** `disturbed` now
samples theta and sigma from the same `external_disturbance_ou_*_range` fields
as `cooperative` and `random-walk`. Its former fixed scalar inputs
`human_ou_theta` and `human_ou_sigma` remain accepted for old YAML loading but
are ignored. Disturbed trajectories therefore change with existing YAML files.
To retain the former theta/sigma values, explicitly set the shared ranges to
`[0.15, 0.15]` and `[0.01, 0.01]`; this retains parameter values, not identical
seeded trajectories, because reset now also draws OU parameters. Historical logs
and validation results are not rewritten.

Drone actions, interfaces, action delay, response randomization, physical
constants, tasks, rewards, terminations, and evaluators are unchanged. The policy
observation retains its legacy 11-D prefix and any configured reference preview.
Human measurements are diagnostics, not additional policy observation channels.
Controller/checkpoint compatibility does not guarantee the original performance
under human motion, especially under `disturbed` behavior.

## Equations and signs

At control step k, use the current task reference and physical state:

```text
e[k]       = pb[k] - pg[k]
delta_e    = e[k] - e[k-1]
delta2_e   = e[k] - 2 e[k-1] + e[k-2]
```

The two differences are **not divided by dt**. They reproduce the historical
controller inputs called `error_dot` and `error_ddot`; substituting `vb` or `ab`
would change the controller. Moving references are handled by differencing the
full error, so no task API for reference acceleration is needed.

The outer and inner incremental PID loops are:

```text
delta_theta = clip(Kp_a * delta_e + Kp_a * dt / Ti_a * e
                   + Kp_a * Td_a / dt * delta2_e, +/- max_theta_change)
theta_target += delta_theta
q = theta_target - theta
delta_v = clip(Kp_v * (q - q_prev) + Kp_v * dt / Ti_v * q
               + Kp_v * Td_v / dt * (q - 2*q_prev + q_prev2), +/- max_acc * dt)
human_cpid_action = -delta_v / (max_acc * dt)
v_cpid += delta_v
```

An integral contribution is zero wherever `Ti=0`. With zero maximum acceleration,
`delta_v` and the normalized action are zero without division by zero.
The human endpoint uses `-human_cpid_action * max_acc * dt`, hence the **positive**
`delta_v` accumulation above. The measured beam angle uses the environment's
existing sign convention; no extra angle or endpoint sign inversion is applied.

All modes use:

```text
gain        = exp(-(e / sigma_d)**2)
cooperative = wc * v_cpid
disturbed   = wd * gain * tanh(e / e_scale)
v_desired   = cooperative + disturbed - position_correction + ou_state
```

| Mode | wc | wd | Interpretation |
| --- | --- | --- | --- |
| cooperative | 1 | 0 | CPID helps move the ball toward the current goal |
| random-walk | 0 | 0 | Random drift with recentering |
| disturbed | 0 | sampled in [-0.2, -0.1] m/s | Error-dependent disruptive velocity |

The disruptive term is zero at exact zero error, has opposite signs on the two
sides of the goal, and decays far from the goal. It is a specific behavioral
model, not a guarantee of worst-case adversarial behavior.

Position correction uses the joint's default position `z_ref`, not a world
coordinate or the drone's height:

```text
p = joint_pos - z_ref
p_effective = sign(p) * max(abs(p) - deadzone, 0)
raw = corr_max * tanh(k_base * p_effective / corr_max)
correction += dt / (corr_tau + dt) * (raw - correction)
```

`corr_max=0` disables correction, and `corr_tau=0` gives the unfiltered value.
OU noise evolves once per control step, with independent standard-normal samples
per environment:

```text
ou_state += ou_theta * (mu - ou_state) * dt + ou_sigma * sqrt(dt) * normal_sample
ou_state  = clip(ou_state, +/- ou_clip)  # only if ou_clip > 0
```

`human_ou_enabled=false` clears the OU contribution. Final execution is:

```text
delta = clip(v_desired - v_command, +/- max_acc * dt)
v_target = v_command + delta
v_increment = delta / decimation
```

Each physics substep advances by `v_increment`; the last substep lands exactly
on `v_target`. Thus the command changes by at most `max_acc * physics_dt` per
substep. There is no additional total velocity cap or hard hand-position clamp.
The acceleration bound applies to the command, not to measured physical motion.

## Configuration

All fields below live directly under `robustness`; there is no nested human
configuration object. Values are finite; ranges must be ordered. Error scales
must be positive. Gains, times, OU ranges and limits must be nonnegative.
Zero integral times, zero filter time, and equal range endpoints are valid.

| Fields | Defaults | Use |
| --- | --- | --- |
| `human_behavior_type` | `random-walk` | Run-wide mode |
| `human_angle_kp_min/max` | 0.2 / 0.5 | Cooperative outer Kp |
| `human_angle_ti_min/max` | 1 / 3 | Cooperative outer Ti, s |
| `human_angle_td_min/max` | 0.6 / 0.9 | Cooperative outer Td, s |
| `human_angle_max_theta_change_deg` | 0.1 | Maximum angle increment per control step, degrees |
| `human_vel_kp_min/max` | 2 / 5 | Cooperative inner Kp |
| `human_vel_ti`, `human_vel_td` | 1000, 0 | Inner PID times, s |
| `human_vel_max_acc` | 0.5 | PID and final command acceleration limit, m/s² |
| `human_pos_feedback_sample` | true | Sample recentering parameters on reset |
| `human_pos_k_base_min/max` | 2 / 4 | Recentring gain, 1/s |
| `human_pos_deadzone_min/max` | 0.03 / 0.08 | Deadzone, m |
| `human_pos_corr_tau_min/max` | 0 / 0.01 | Filter time, s |
| `human_pos_k_base`, `human_pos_deadzone`, `human_pos_corr_tau` | 3, 0.05, 0.01 | Used when recentering sampling is disabled |
| `human_pos_corr_max` | 1 | Recentring velocity saturation, m/s |
| `human_ou_enabled` | true | Enable OU contribution |
| `external_disturbance_ou_mu` | 0 | Shared OU mean, m/s |
| `external_disturbance_ou_clip` | 0.05 | Shared OU saturation, m/s; zero disables clipping |
| `external_disturbance_ou_theta_range` | [0.1, 0.3] | All modes: OU rate sampled on reset, 1/s |
| `external_disturbance_ou_sigma_range` | [0.01, 0.05] | All modes: OU diffusion sampled on reset, (m/s)/sqrt(s) |
| `human_behavior_cooperative_wc/wd` | 1 / 0 | Cooperative weights |
| `human_behavior_random_walk_wc/wd` | 0 / 0 | Random weights |
| `human_behavior_disturbed_wc` | 0 | Disturbed cooperative weight |
| `human_disturbed_wd_min/max` | -0.2 / -0.1 | Disturbed velocity weight, m/s |
| `human_disturbed_sigma_d_min/max` | 0.1 / 0.2 | Disturbed decay scale, m |
| `human_disturbed_e_scale_min/max` | 0.05 / 0.1 | Disturbed sign smoothing scale, m |
| `human_behavior_sigma_d`, `human_behavior_e_scale` | 0.1, 0.05 | Scales for other modes if their disruption weight is enabled |

`human_behavior_disturbed_wd=-0.1` is retained from the historical configuration,
but disturbed reset sampling overrides it. For a fixed weight, set both
`human_disturbed_wd_min/max` to the desired value. Non-cooperative PID parameters
are outer `(0.5, 1.5, 0.6)` and inner Kp `10`; its contribution is zero at the
default `wc=0`. The historical `human_random_walk_ou_*_min/max` fields are represented
by the existing `external_disturbance_ou_*_range` fields in this repository.

Each environment keeps its sampled OU parameters until it resets; partial reset
does not resample other environments. Equal bounds specify fixed parameters, e.g.:

```yaml
robustness:
  enabled: true
  external_disturbance_enabled: true
  human_behavior_type: disturbed
  human_ou_enabled: true
  external_disturbance_ou_mu: 0.0
  external_disturbance_ou_theta_range: [0.8, 0.8]
  external_disturbance_ou_sigma_range: [0.01, 0.01]
  external_disturbance_ou_clip: 0.03
```

The same fields apply when changing the mode to `cooperative` or `random-walk`.
Shared ranges specify the same distribution, not necessarily the same sampled
values across modes with the same seed: other behavior parameters also consume
random draws. Diagnostic fields `human_ou_theta` and `human_ou_sigma` still report
the actual sampled values; these log names are independent of the deprecated
configuration fields with the same names.

## Lifecycle and measurements

The manager calls `prepare_step` once before the physics loop, then submits
`advance_substep()` before each `scene.write_data_to_sim()`. After each scene
update it reads actual joint position, velocity and acceleration. At reset it
clears selected rows, resamples their parameters, and zeroes their joint command.
After the task and physical state are refreshed, both error histories are set
to the reset error, preventing an artificial first-step derivative pulse.
Other environments' histories and ramps are untouched. Disabled human dynamics
consume no random samples and hold a zero target, but measurements remain active.

`extras["step"]` contains one scalar per environment for each diagnostic. Existing
environment snapshot/cloning occurs before autoreset, preserving terminal-step
measurements, parameters and commands. There is no evaluator metric change.

| Field(s) | Meaning |
| --- | --- |
| `human_enabled`, `human_behavior_type_id` | Active flag; 0 cooperative, 1 random-walk, 2 disturbed (configured mode even when disabled) |
| `human_pos_z`, `human_pos_ref_z` | Joint displacement and default displacement, m |
| `human_vel_z` | Actual joint velocity at the end of the control step, m/s |
| `human_acc_z` | Mean actual joint acceleration over that step's physics samples, m/s² |
| `human_target_vel_z`, `human_target_acc_z` | Last submitted velocity and control-step commanded acceleration |
| `human_desired_vel_z` | Desired velocity before final acceleration limiting |
| `human_cooperative_term_z`, `human_disturbed_term_z`, `human_behavior_target_vel_z` | Behavior contributions and their sum, m/s |
| `human_pos_feedback_raw_vel_z`, `human_pos_feedback_vel_z` | Raw and filtered recentering contributions, m/s |
| `human_ou_drift_vel_z` | OU contribution, m/s |
| `human_cpid_action_z`, `human_pid_target_vel_z`, `human_cpid_sign_z` | Normalized PID action, accumulated PID velocity, smoothed error sign |
| `human_behavior_gain`, `human_behavior_wc/wd`, `human_behavior_sigma_d/e_scale` | Actual behavior parameters and Gaussian gate |
| `human_pos_k_base/deadzone/corr_tau`, `human_ou_theta/sigma` | Actual per-environment sampled parameters |
| `human_angle_kp/ti/td`, `human_vel_kp/ti/td/max_acc`, `human_angle_max_theta_change` | Actual PID parameters; logged angle increment is in radians |

Slash-separated table entries denote separate fields. The complete field list
is `HUMAN_STEP_EXTRA_FIELDS` in the pure Torch model module. The old
`external_disturbance_vel_z` remains a **command**, now an alias for
`human_target_vel_z`; it must not be interpreted as measured hand velocity.
The old OU parameter fields report the parameters actually used in the mode.

Seven evaluation runners (zero-action, CPID, NMPC, RL, CPID unified, acados NMPC,
NFFB) save these fields when `runner.save_rollout=true`. NPZ keys have a `step_`
prefix and shape `(steps, num_envs)`. Existing summary metrics and training
logging are unchanged. For example:

```python
import numpy as np

data = np.load("/path/to/rollout.npz")
measured = data["step_human_vel_z"]
commanded = data["step_human_target_vel_z"]
mode = data["step_human_behavior_type_id"]
```

## Short evaluation examples

From the repository root, in the Isaac Sim / Isaac Lab environment:

```bash
python scripts/zero_action_policy_eval.py --config environments/configs/human_cooperative_eval.yaml --episodes 2 --num_envs 2 --headless
python scripts/zero_action_policy_eval.py --config environments/configs/human_random_walk_eval.yaml --episodes 2 --num_envs 2 --headless
python scripts/zero_action_policy_eval.py --config environments/configs/human_disturbed_eval.yaml --episodes 2 --num_envs 2 --headless
python scripts/cpid_policy_eval.py --config baselines/configs/cpid_target_position_eval_delay_free.yaml --env_config environments/configs/human_cooperative_eval.yaml --episodes 2 --num_envs 2 --headless
```

The example environment horizon is 2 s. Zero-action runs verify the plant and
logging rather than successful balancing. For deterministic direction checks,
disable OU, disable recentering parameter sampling, set PID/behavior sampling
ranges to equal endpoints, and fix the seed. The normal task still owns initial
ball position sampling. Switching drone interface or task uses the existing
configuration fields; human behavior is independent of that choice.

## Validation

The tests compare all three modes against independent scalar recurrences using
fixed inputs and noise, and cover discrete differences, recentering, OU timing,
acceleration bounds, partial reset, no-RNG disabled operation, actual/commanded
velocity separation, terminal snapshots, and runner log selection.

```bash
python -m pytest -q tests/test_human_behavior.py tests/test_human_behavior_integration.py tests/test_robustness_velocity_response.py
```

Isaac smoke acceptance additionally requires finite physical states, correct
endpoint direction, compatible drone interfaces/tasks/robustness settings, and
complete NPZ fields through termination and reset. Pure Torch tests alone do not
establish those physical behavior checks or closed-loop balancing performance.

### Implementation verification (2026-10-15)

The selected model, integration, delay/response, observation, and reference/task
regression suite passed 155 tests. One CUDA-only unit test was skipped inside the
sandbox; the following actual GPU Isaac Sim runs succeeded outside the sandbox.
Each run used two environments and two completed episodes:

| Runner / mode | Interface / task | Horizon | Additional checks |
| --- | --- | --- | --- |
| Zero-action / cooperative | velocity / target_position | 2 s | Complete motion/parameter logs |
| Zero-action / random-walk | thrust / target_position | 2 s | Interface compatibility |
| Zero-action / disturbed | position / target_position | 2 s | Interface compatibility |
| Zero-action / cooperative | velocity / trajectory_tracking | 2 s | Moving reference |
| Zero-action / random-walk | velocity / unified_tracking | 2 s | Delay=2, velocity response and ball-mass variation together |
| Zero-action / cooperative and disturbed, separate runs | velocity / target_position | 0.5 s each | Fixed parameters, no OU; opposite hand/beam directions and ball motion toward/away from goal |
| CPID / cooperative | velocity / target_position | 2 s | Existing controller and NPZ logging together |

The fixed-direction runs started the ball at 0.40 m with goal 0.35 m. Cooperative
motion reduced ball position by approximately 0.00038 m over 0.5 s; disturbed
motion increased it by approximately 0.00503 m. These are direction checks, not
tracking-performance benchmarks. Rollout arrays were checked for complete human
fields, finite values, command acceleration bounds, and preserved terminal-step
velocities. Smoke artifacts were written under `/tmp/aerial_balance_human_smoke`,
not into existing experiment logs. Runtime NMPC/RL evaluation and long-horizon
performance tuning were not performed; their logging schemas were tested.

## Using actual motion in predictive control

The acados policy can use the current measured `human_vel_z` with
`acados_nmpc_policy.human_motion.enabled: true`. It holds this measurement across
the delay predictor and OCP horizon, refreshing it each control cycle. This does
not change the behavior generator or expose future human actions. The runner
reads current manager state to remain aligned with observations after autoreset;
it does not feed the terminal `step` snapshot back into the controller. See
[the model, timing, input and logging contract](acados_unified_nmpc.md#15-实测人速度的常值预测控制).
