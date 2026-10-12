# Residual dynamics comparison — final experiment report

## Protocol and provenance

Six models; 2,028 identical test windows; seed 7 only. Each window is integrated once to 60 steps without future-state resets. No test data were used for fitting or checkpoint selection.

Dataset hash: `03305dc728375d38fb373a713e669c1f71bd5ea084da9f5f21f9760eb0574498`.
Window hash: `839102797ff97b2a33890294120456b5332290b59737f16b8a8cad4eda9b1d1e`.

Full model/training configuration, source hashes, software versions and per-model checkpoint identities are in `results/test/metadata.json` and `runs/*/config.json`. Trial-level train/validation/test split is 79/6/6; five stress trials remain separate. Training data loss and schedule are common; regularization is representation-specific (see protocol.md).

## Training and model selection

| Model | Active parameters | Stage 1 / 2 epochs | Best epoch | Validation score | Time (s) |
|---|---:|---:|---|---:|---:|
| Nominal | 0 | — | — | 1.000000 | — |
| BPR | 13 | 52 / 50 | stage2 / 37 | 0.928864914 | 42.3 |
| FDR | 26 | 41 / 50 | stage2 / 50 | 0.975404624 | 41.6 |
| RBF-PV | 81 | 21 / 50 | stage2 / 50 | 0.965099944 | 85.6 |
| MLP | 385 | 46 / 50 | stage2 / 50 | 0.868704772 | 38.8 |
| Structured | 39 | 52 / 50 | stage2 / 37 | 0.923679027 | 51.3 |

## Held-out pooled RMSE

Pooled over windows, not averaged per-trial RMSE. Positive reduction means improvement over nominal.

| Model | H | Position (cm) | Velocity (cm/s) | Position reduction (%) | Velocity reduction (%) |
|---|---:|---:|---:|---:|---:|
| Nominal | 30 | 2.36402 | 7.17806 | 0.000 | 0.000 |
| BPR | 30 | 2.19919 | 6.49996 | 6.972 | 9.447 |
| FDR | 30 | 2.26892 | 6.84573 | 4.023 | 4.630 |
| RBF-PV | 30 | 2.18494 | 6.40424 | 7.575 | 10.780 |
| MLP | 30 | 1.89877 | 5.17842 | 19.680 | 27.858 |
| Structured | 30 | 2.14325 | 6.31813 | 9.339 | 11.980 |
| Nominal | 60 | 6.19106 | 11.41288 | 0.000 | 0.000 |
| BPR | 60 | 5.25073 | 9.56739 | 15.188 | 16.170 |
| FDR | 60 | 5.68224 | 10.36475 | 8.219 | 9.184 |
| RBF-PV | 60 | 5.23252 | 9.56051 | 15.483 | 16.231 |
| MLP | 60 | 3.94632 | 6.63433 | 36.258 | 41.870 |
| Structured | 60 | 4.97059 | 9.04156 | 19.713 | 20.778 |

## Horizon-dependent ranking

- H=30, position_rmse_m: MLP < Structured < RBF-PV < BPR < FDR < Nominal.
- H=30, velocity_rmse_mps: MLP < Structured < RBF-PV < BPR < FDR < Nominal.
- H=60, position_rmse_m: MLP < Structured < RBF-PV < BPR < FDR < Nominal.
- H=60, velocity_rmse_mps: MLP < Structured < RBF-PV < BPR < FDR < Nominal.

## Per-reference-type RMSE

| Reference | H | Model | Position (cm) | Velocity (cm/s) |
|---|---:|---|---:|---:|
| constant | 30 | FDR | 1.37356 | 4.95801 |
| constant | 30 | MLP | 0.83826 | 2.66940 |
| constant | 30 | Nominal | 1.51124 | 5.56840 |
| constant | 30 | BPR | 1.24096 | 4.38659 |
| constant | 30 | RBF-PV | 1.12858 | 3.93670 |
| constant | 30 | Structured | 1.13885 | 3.98623 |
| constant | 60 | FDR | 4.54578 | 9.08191 |
| constant | 60 | MLP | 1.98876 | 3.95909 |
| constant | 60 | Nominal | 5.28423 | 10.78152 |
| constant | 60 | BPR | 3.93041 | 7.99679 |
| constant | 60 | RBF-PV | 3.49998 | 7.40117 |
| constant | 60 | Structured | 3.46241 | 7.06225 |
| sine | 30 | FDR | 3.03229 | 8.36857 |
| sine | 30 | MLP | 2.72138 | 7.32898 |
| sine | 30 | Nominal | 3.16610 | 8.68558 |
| sine | 30 | BPR | 2.95623 | 8.00793 |
| sine | 30 | RBF-PV | 3.05592 | 8.42400 |
| sine | 30 | Structured | 2.90255 | 7.97641 |
| sine | 60 | FDR | 6.55462 | 10.60398 |
| sine | 60 | MLP | 5.46592 | 8.79612 |
| sine | 60 | Nominal | 7.16525 | 11.41379 |
| sine | 60 | BPR | 6.20068 | 10.43815 |
| sine | 60 | RBF-PV | 6.76356 | 11.42115 |
| sine | 60 | Structured | 6.01710 | 10.32024 |
| trapezoid | 30 | FDR | 3.94651 | 11.27070 |
| trapezoid | 30 | MLP | 3.52544 | 9.32478 |
| trapezoid | 30 | Nominal | 4.02750 | 11.40935 |
| trapezoid | 30 | BPR | 3.92884 | 11.13762 |
| trapezoid | 30 | RBF-PV | 3.91482 | 11.04997 |
| trapezoid | 30 | Structured | 3.88008 | 11.02209 |
| trapezoid | 60 | FDR | 8.92409 | 15.09964 |
| trapezoid | 60 | MLP | 7.22505 | 11.52778 |
| trapezoid | 60 | Nominal | 9.19510 | 15.44793 |
| trapezoid | 60 | BPR | 8.58527 | 14.08263 |
| trapezoid | 60 | RBF-PV | 8.75250 | 14.50841 |
| trapezoid | 60 | Structured | 8.39384 | 13.78570 |
| triangle | 30 | FDR | 1.09078 | 4.00035 |
| triangle | 30 | MLP | 0.55546 | 1.75326 |
| triangle | 30 | Nominal | 1.08682 | 4.00410 |
| triangle | 30 | BPR | 0.98362 | 3.65199 |
| triangle | 30 | RBF-PV | 0.94801 | 3.49536 |
| triangle | 30 | Structured | 0.96145 | 3.52949 |
| triangle | 60 | FDR | 3.52916 | 8.07111 |
| triangle | 60 | MLP | 1.43799 | 3.61082 |
| triangle | 60 | Nominal | 3.68224 | 8.45530 |
| triangle | 60 | BPR | 3.18732 | 7.58828 |
| triangle | 60 | RBF-PV | 2.93744 | 7.17693 |
| triangle | 60 | Structured | 2.97057 | 7.20552 |

## Interpretation for the paper

Structured improves over nominal at all ten evaluated horizons, and improves over both ablations and RBF-PV in pooled H=30 and H=60 position/velocity RMSE. MLP nevertheless has the lowest pooled offline errors, reducing H=60 position/velocity RMSE by 36.3%/41.9% versus Structured's 19.7%/20.8%. This does not support claiming that physical structure maximizes prediction accuracy. Structured uses 39 active coefficients versus MLP's 385 parameters, and its incorporation in predictive control has separate existing real-world evidence; there is no MLP closed-loop comparison in this experiment.

There are task-dependent exceptions: at H=30, RBF-PV is slightly better than Structured on constant-reference windows (both position and velocity), while the order reverses by H=60. FDR slightly worsens triangular-reference H=30 position RMSE relative to nominal; RBF-PV slightly worsens sinusoidal-reference H=60 velocity RMSE. Pooled ranking must not be generalized to every task.

## Auxiliary diagnostics

Saturation means absolute applied residual >= 99% of 0.25 m/s²; it is not hard clipping.

| Model | Point acceleration RMSE (m/s²) | RK4-stage saturation (%) | Nonfinite predictions |
|---|---:|---:|---:|
| Nominal | 0.213534 | 0.0000 | 0 |
| BPR | 0.203176 | 0.0000 | 0 |
| FDR | 0.215671 | 0.0000 | 0 |
| RBF-PV | 0.204304 | 0.0000 | 0 |
| MLP | 0.197000 | 0.1200 | 0 |
| Structured | 0.205187 | 0.0000 | 0 |

## Acceptance and reproduction

See `provenance/parity.json` for the bitwise forward, sampling, batch, loss and gradient comparison against the predecessor. `results/acceptance.json` records independent aggregate recomputation, checkpoint/Structured-retraining reproduction and contract tests. Paper updates must wait for that acceptance file to report passed.

## Limitations and claim boundaries

- Single training seed: no seed-variance bars or statistical superiority claims.
- Overlapping windows are correlated; 2,028 windows are not 2,028 independent experiments.
- Six held-out trials cover four reference types but provide limited per-type replication.
- The common data loss/selection budget does not make representation-specific regularizers identical; this comparison does not establish optimally tuned MLP/RBF performance.
- Five-input normalization is exactly the predecessor's training-only statistic, including its pre-mask training features; no validation or test samples enter it.
- Point acceleration targets depend on offline smoothing/differentiation; causal rollouts and measured endpoint states provide the primary control-relevant evidence.
- These are offline results; baseline ranking does not establish closed-loop superiority. Existing physical experiments and deployed weights are unchanged.

## Recorded reproduction outcome

Acceptance passed: True. Maximum Structured parameter difference: 1.388e-17; maximum training-history difference: 2.220e-16. Both stages used the same number of epochs and selected stage2/epoch37 with exactly the same validation score. This is floating-point-precision reproduction, not bitwise identity. The old runtime record does not include CPU thread count, so the original cause of the round-off-scale training-history differences cannot be uniquely assigned. The explicit migration comparison using the same saved parameters is bitwise identical for forward, sampler, loss and gradients. Old physical weights remain untouched; the new checkpoint container records the offline retraining provenance.
