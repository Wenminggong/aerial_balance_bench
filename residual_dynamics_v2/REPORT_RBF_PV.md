# 二维 RBF 球加速度残差模型

## 1. 设计动机

原五输入 MLP 将 `[pb,vb,theta,omega,a_nom]` 同时送入残差网络。由于标签本身为
`smooth_ab-a_nom`，网络可以通过学习负的 `a_nom` 通道降低离线 rollout 误差，同时
削弱 NMPC 的梁角控制增益。新模型只使用 `[pb,vb]`，但仍在每个未来 RK4 stage
重新计算，因此不是每次求解冻结的常数补偿。

## 2. 数据和划分

数据来自 20260815、17、18、19_01、19_02、19_03 六批日志。79 条成功 trial
用于训练，validation/test 各含来自六个 session 的一条成功 trial，五条失败安全前缀
只用于 stress。归一化、cap、RBF OOD 和 checkpoint 选择只使用 train。

训练窗口采用 40% empirical、40% `pb-vb` joint coverage 和20%条件于 `a_nom`
符号/幅值的 actuation-diversity coverage。`a_nom` 只参与采样和评价，不是模型输入。

## 3. RBF 和 OOD

模型使用固定的 9×9 Gaussian RBF，位置中心均匀覆盖 `[0,0.7]`，速度中心为
`[-0.5,-0.3,-0.15,-0.06,0,0.06,0.15,0.3,0.5]`，两维宽度均为0.1物理单位。
基函数通过 `1e-12+sum(phi)` 归一化，81个系数被投影到数据和安全共同确定的 cap。

cap 为 train `|residual label|` P95 向上取整到 `0.025 m/s²`，并限制在
`[0.15,0.25] m/s²`。当前数据预期得到 `0.25 m/s²`。

OOD 使用 train-only 的二维 marginal quantile、shrinkage correlation 和 Mahalanobis
阈值。符号动力学中的 OOD 只依赖 `pb,vb`，因此不会重新引入 `theta/omega/a_nom`
控制通道导数。

## 4. 两阶段损失

第一阶段使用 H=3/6/12 的短区间速度积分一致性，权重为0.50/0.30/0.20：

\[
L_1=L_{int}+0.5L_{point}+0.01L_{curv}+10L_v+5L_p.
\]

`L_int` 将 SG 速度增量扣除实测 nominal 加速度积分，与沿因果 `obs_pb/obs_vb`
计算的 RBF 积分比较。`L_point` 是 `beta=0.02` 的逐点 Huber 辅助损失；导数 hinge
将 `r_v` 的软区间限制在 `[-0.2,0.1] s^-1`，并限制 `|r_p|<=0.5 s^-2`。

第二阶段从第一阶段最优模型初始化：

\[
L_2=L_1+0.25L_{causal rollout}+10L_{trust}.
\]

rollout H=6/12/30/33 权重为0.50/0.30/0.15/0.05，使用归一化 Smooth-L1；SG
状态为主目标，因果观测为0.25权重辅助目标。系数被限制在第一阶段的 `±0.03 m/s²`
信赖域内，anchor-grid 输出超过 `0.02 m/s²` 的变化才受额外 hinge 惩罚。

最终 checkpoint 只由 validation 的 integral、point 和 h12 指标选择；若 stage2 未同时
通过在线导数门槛并改善选择分数，自动回退 stage1。test/stress 不参与选择。

## 5. 验收语义

`eligible_for_residual_integration` 由 validation/test/stress 的拟合、rollout、输出、
二维导数和控制灵敏度 gates 决定。`eligible_for_real_world_deployment` 还要求响应链、
queue 及尚未实施的 C++/CasADi parity，因此本离线工作流不会单独宣告可直接部署。

<!-- BEGIN GENERATED RESULTS -->
## 实际训练与评估结果

- Run：`rbf2_seed7_38053cc7_20260821_122825`
- seed：`7`；训练设备：`cpu`
- 有效 cap：`±0.250 m/s²`
- Stage 1：epoch 1，耗时 71.9 s，selection score 0.986841
- Stage 2：epoch 68，耗时 679.0 s，selection score 0.985935
- 最终选择：**stage2**
- Residual integration eligibility：**FAIL**
- Real-world deployment eligibility：**FAIL**

### 各 split 总体指标

| Split | Point nominal/RBF RMSE | Integral h12 nominal/RBF RMSE | h33 pb nominal/RBF | h33 vb nominal/RBF | Acceptance |
| --- | ---: | ---: | ---: | ---: | --- |
| `stress` | 0.425269 / 0.420198 | 0.425377 / 0.422283 | 0.053597 / 0.053300 | 0.152481 / 0.150277 | PASS |
| `test` | 0.213534 / 0.205562 | 0.212896 / 0.204556 | 0.026808 / 0.025431 | 0.078845 / 0.072863 | FAIL |
| `validation` | 0.135121 / 0.130663 | 0.173389 / 0.169794 | 0.025524 / 0.025156 | 0.070874 / 0.069085 | FAIL |

### Validation 逐点与旧模型对照

| 模型 | RMSE | MAE | Bias | R² |
| --- | ---: | ---: | ---: | ---: |
| Nominal | 0.135121 | 0.079713 | 0.015966 | -0.0142 |
| RBF-PV | 0.130663 | 0.075651 | 0.011666 | 0.0517 |
| CAP0P25 MLP | 0.128538 | 0.072483 | 0.010007 | 0.0822 |

### Validation 导数、OOD 与静态诊断

- `rp` abs P99/max：0.416142 / 0.433653 s⁻²
- `rv` min/P1/P50/P99/max：-0.231567 / -0.202069 / 0.000014 / 0.102471 / 0.110061 s⁻¹
- exogenous residual gradient max：theta=0.000e+00，omega=0.000e+00，a_nom=0.000e+00
- residual saturation rate：0.000%
- OOD fraction/mean weight/min weight：0.98% / 0.998103 / 0.000001
- shadow full-state OOD fraction/P99 distance：2.09% / 0.410693（仅诊断，不进入动力学）
- command queue alignment RMSE/max：0.000e+00 / 0.000e+00 m/s；PASS
- response-chain theta/vrz bias：0.005099 rad / -0.002803 m/s；FAIL
- static total acceleration median：0.031322 m/s²

### 结果解释

- 二维结构达成了首要安全目标：残差对 theta/omega/a_nom 的直接导数严格为零，控制通道不再依赖 soft loss 维持。
- Validation h33 的位置/速度 MSE 改善为 2.86% / 4.98%，属于保守但有限的预测收益。
- Validation h3/h6/h12 积分改善分别为 2.19% / 2.57% / 4.10%；短 horizon 未达到5% gate。
- 当前未通过不代表重新出现旧模型式控制通道破坏；主要剩余问题是残差可辨识收益偏小、静态偏差和 outer-grid 速度梯度轻微越界。

### Validation hard gates

| Gate | Result |
| --- | --- |
| `control_sensitivity` | PASS |
| `direct_control_channel` | PASS |
| `dynamic_groups_no_large_degradation` | PASS |
| `finite_outputs_gradients_rollout` | PASS |
| `h12_h33_not_worse` | PASS |
| `integral_improvement` | FAIL |
| `output_cap` | PASS |
| `point_not_worse_than_nominal` | PASS |
| `rp_limits` | PASS |
| `rv_extrema` | PASS |
| `rv_percentiles` | FAIL |
| `saturation_rate` | PASS |
| `static_total_acceleration` | FAIL |

### 部署阻塞项

- offline residual hard gates failed
- response-chain gate failed
- C++/CasADi deployment parity not implemented in this offline scope

### 图表

![Training history](results_rbf_pv/figures/training_history.png)

![Point validation](results_rbf_pv/figures/point_validation.png)

![Integral validation](results_rbf_pv/figures/integral_validation.png)

![Validation rollout](results_rbf_pv/figures/rollout_validation.png)

![Derivative summary](results_rbf_pv/figures/derivative_summary.png)

![Control sensitivity](results_rbf_pv/figures/control_sensitivity.png)

完整的 session、group、position-velocity cell 和 a_nom regime 分层指标保存在 `results_rbf_pv/summary.json`。

### 完整性

- dataset hash：`03305dc728375d38fb373a713e669c1f71bd5ea084da9f5f21f9760eb0574498`
- config hash：`38053cc7a282ebc75647030e0a5f19ad270e82554807d287735bdde45c463700`
- checkpoint hash：`4ce5ddc8f83b5d71e4609f3b63d76188c8d119caf3c8a9816cfcd51a10faa2ff`
- portable model hash：`20b04835e806748e37240edbf25a34fd76a00cde37e17a23a707898433b309f3`
<!-- END GENERATED RESULTS -->
