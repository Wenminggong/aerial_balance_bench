# Residual Dynamics Identification

This directory contains the standalone, offline identification workflow for a
bounded five-input ball-acceleration residual model:

$$
r_a=f_\phi(p_b,v_b,\theta,\omega,a_{\mathrm{nom}}).
$$

It intentionally does not modify or import the real-world C++ controller, the
state predictor, or acados. The saved model is an identification artifact for a
later integration phase.

## Data contract

The raw source is
`logs/real_world/acados_nmpc_cpp/real_world_acados_unified_20260815`.
Only `prepare` reads that directory. Training and evaluation read compressed
per-trial files below `residual_dynamics/data/processed`.

The fixed split contains 20 training trials and 9 validation trials. Standard
`trial_001` and `trial_002` recordings are used for training, standard
`trial_003` recordings are used for validation, and both completed
`constant_035_2` recordings are added to training. There is no independent
test set; validation is also the final offline evaluation set.

## Reproduction

Run commands from the repository root:

```bash
conda run -n isaac-sim python -m residual_dynamics.cli prepare \
  --config residual_dynamics/configs/mlp_5input.yaml

conda run -n isaac-sim python -m residual_dynamics.cli train \
  --config residual_dynamics/configs/mlp_5input.yaml

conda run -n isaac-sim python -m residual_dynamics.cli evaluate \
  --config residual_dynamics/configs/mlp_5input.yaml

conda run -n isaac-sim python -m residual_dynamics.cli report \
  --config residual_dynamics/configs/mlp_5input.yaml
```

The whole workflow is also available as one command:

```bash
conda run -n isaac-sim python -m residual_dynamics.cli all \
  --config residual_dynamics/configs/mlp_5input.yaml
```

W&B is online-first and falls back to a local offline run if login or network
initialization fails. To force a mode for a particular run, pass one of:

```text
--wandb-mode online|offline|disabled
```

The local W&B directory, processed data, checkpoints, and complete predictions
are ignored by Git. Compact metrics, reviewed figures, and `REPORT.md` are
tracked.

## Tests

```bash
conda run -n isaac-sim python -m pytest residual_dynamics/tests -q
```

See [REPORT.md](REPORT.md) for the full preprocessing, model, training,
validation, and result specification.

## Deployment-isomorphic causal rollout variant

`configs/mlp_5input_causal_rollout.yaml` keeps the v1.0 `5-16-16-1` MLP,
`tanh` activations, `0.25 m/s²` output bound, and portable weight format. Its
training state is instead `[pb, vb, theta, vrz]`: every window consumes the
logged three-command pending queue first, then future committed commands. The
metadata velocity-response model and full RK4 dynamics recursively generate
all later network inputs. Causal `obs_pb/obs_vb` are the primary rollout
targets; SG-smoothed `pb/vb` are auxiliary targets with weight `0.25`.

The causal processed schema is version 2 and is not compatible with the v1.0
cache. Run `prepare` explicitly before training; the independent paths do not
overwrite v1.0 data, reports, or weights:

```bash
conda run -n isaac-sim python -m residual_dynamics.cli prepare \
  --config residual_dynamics/configs/mlp_5input_causal_rollout.yaml

conda run -n isaac-sim python -m residual_dynamics.cli train \
  --config residual_dynamics/configs/mlp_5input_causal_rollout.yaml

conda run -n isaac-sim python -m residual_dynamics.cli evaluate \
  --config residual_dynamics/configs/mlp_5input_causal_rollout.yaml

conda run -n isaac-sim python -m residual_dynamics.cli report \
  --config residual_dynamics/configs/mlp_5input_causal_rollout.yaml
```

The predefined Jacobian screen uses a fixed short budget (16 steps/epoch, at
most 80 epochs) at seed 7 with Jacobian disabled and with
`Lmax={0.01,0.02,0.04}`. It applies the hard validation filters, selects by the
normalized 33-step causal `pb/vb` score, and then retrains the selected setting
with the full configured budget at seeds 7, 17, and 27:

```bash
conda run -n isaac-sim python -m residual_dynamics.cli screen \
  --config residual_dynamics/configs/mlp_5input_causal_rollout.yaml \
  --wandb-mode disabled
```

Individual controlled runs can use `--seed`, `--jacobian-weight`, and
`--jacobian-limit`. The causal report is written to
[`REPORT_CAUSAL.md`](REPORT_CAUSAL.md), while screening writes
`results_causal/screening_summary.json`.

## Multi-session causal v2 with control-channel constraints

`configs/mlp_5input_causal_v2.yaml` is the independent version-3 workflow. It
uses the six 20260815--19 sessions, train/validation/test/stress selectors,
safe prefixes from failed trials, seven fixed 0.1 m position bins, a 50/50
empirical/coverage sampler, and train-only quantile plus shrinkage-Mahalanobis
OOD statistics.

The rollout loss still uses the same causal queue and response chain. In
addition, it differentiates total acceleration through
`theta -> geometry/omega -> a_nom -> MLP`, constrains the nominal theta-channel
gain ratio and anti-damping, and adds deterministic collocation and outer
safety points. The screen evaluates `lambda_theta={0,1,10}` and adds `30` only
when required. Test and stress are evaluated only after the setting and seed
have been selected on validation:

```bash
conda run -n isaac-sim python -m residual_dynamics.cli prepare \
  --config residual_dynamics/configs/mlp_5input_causal_v2.yaml

conda run -n isaac-sim python -m residual_dynamics.cli screen \
  --config residual_dynamics/configs/mlp_5input_causal_v2.yaml
```

An export is blocked unless validation, test, and stress all pass. A passing
model uses deployment version `v1_causal_v2`; until then no v2 model JSON or
generated acados solver is created. See
[`REPORT_CAUSAL_V2.md`](REPORT_CAUSAL_V2.md).
