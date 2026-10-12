# Two-input RBF residual dynamics

This package identifies the bounded acceleration residual

\[
r_a=b_{\mathrm{RBF}}(p_b,v_b)
\]

from six real-world sessions. The RBF is evaluated at every causal RK4 stage, but it
has no direct dependency on beam angle, beam rate, nominal acceleration, or the drone
velocity command channel. The existing `residual_dynamics` package is left unchanged;
its validated raw-log parser and causal physical model are imported read-only while all
new caches, training artifacts, results, exports, tests, and documentation live here.

Run from the repository root:

```bash
conda run -n isaac-sim python -m residual_dynamics_v2.cli prepare
conda run -n isaac-sim python -m residual_dynamics_v2.cli train --wandb-mode disabled
conda run -n isaac-sim python -m residual_dynamics_v2.cli evaluate --include-holdout
conda run -n isaac-sim python -m residual_dynamics_v2.cli report
```

Tests:

```bash
conda run -n isaac-sim pytest residual_dynamics_v2/tests -q
```

See [`REPORT_RBF_PV.md`](REPORT_RBF_PV.md) for the full data, model, loss,
selection, and measured-result contract.

The follow-up structural ablation comparing local beam profile, structured friction,
and an added two-dimensional black-box residual is documented in
[`PROFILE_FRICTION_PROBE.md`](PROFILE_FRICTION_PROBE.md).

The bounded `profile + friction + kappa(pb,vb) * a_nom` candidate is appended without
retraining earlier candidates by:

```bash
conda run -n isaac-sim \
  python -m residual_dynamics_v2.profile_friction_kappa_probe
```

The command refuses to overwrite an existing κ result unless `--replace` is supplied.

Regenerate its inference-only detailed metrics and figures without training:

```bash
MPLCONFIGDIR=/tmp/aerial_balance_mpl conda run -n isaac-sim \
  python -m residual_dynamics_v2.profile_friction_detailed_analysis
```
