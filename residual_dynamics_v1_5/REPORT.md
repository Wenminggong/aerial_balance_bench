# 加速度残差动力学 v1.5：运动状态 MLP 与平滑静止死区

## 1. 问题定义与范围

v1.5 将球的加速度分成自由运动驱动力和低速静止约束。目标是避免 v1.0 用同一个连续 MLP 同时解释运动动力学和 stick/stiction 样本所产生的低速虚假加速度，同时保留 MLP 对局部坡度、梁弯曲及其他平滑动态误差的表示能力。本阶段只做离线拟合与验证，不修改状态预测器、NMPC 或 acados。

## 2. 数据与划分

拟合数据来自 `real_world_acados_unified_20260815`。九组标准配置的 trial 001/002 以及 `constant_035_2` 的两条完成实验进入训练，标准配置 trial 003 进入验证；stale-beam trial 排除，形成固定的 20/9 划分。验证集参与 early stopping，因此不是未知数据上的无偏测试。

外部评价来自 `real_world_acados_unified_20260817`：14 条 `tracking_duration` trial 冻结测试；3 条 solver fallback、1 条 beam safety 与1条 manual hover 作为 stress 数据。08-17 已用于此前的问题分析，因此这里称为回顾性外部测试，而不是论文级 prospective test。

## 3. 预处理和模式判定

每个连续段按实际中位周期重采样。球位置用31点三阶 Savitzky–Golay 同时获得位置、速度和加速度；梁角用11点三阶；无人机竖直速度用7点二阶。每段丢弃首尾15点。相邻原始球位置跳变超过0.025 m时膨胀屏蔽前后18点；stress trial 在第一处跳变前截断有效区域。

- `dynamic_mask`：`|vb|≥0.03 m/s`，且中心前后3点方向一致。
- `static_mask`：`|vb|≤0.01 m/s`、`|ab|≤0.03 m/s²`，连续至少7点。
- `transition_mask`：其余有效低速、换向或 breakaway 样本。
- `invalid_mask`：Laser跳变、非有限值、时间缺口或明显非物理导数。

输入归一化只使用20条训练 trial 中的动态有效样本。处理对象是约60 Hz控制快照，不被解释为完整240 Hz Laser原始观测。

## 4. 名义动力学与标签

名义几何和球加速度与当前控制器公式一致：

\[
\omega=-\frac{v_{rz}\cos\beta}{L\cos(\beta-\theta)},\qquad
a_{\mathrm{nom}}=\gamma[(p_b+p_0-L)\omega^2-g\sin\theta].
\]

阶段 A 的动态标签为 `ab_hat-a_nom`。不使用日志中的 `obs_ab`，也不直接对原始球位置做二次差分。

## 5. 组合模型

MLP 输入为 `[pb,vb,theta,omega,a_nom]`，网络结构为 `5→16→16→1`，隐藏激活为 `tanh`，输出限制为 `0.20*tanh(y) m/s²`。令动态残差为 \(r_d\)，则：

\[
d=a_{\mathrm{nom}}+r_d,
\]

\[
D(d;a_s,\tau_s)=\tau_s\left[\operatorname{softplus}\frac{d-a_s}{\tau_s}-
\operatorname{softplus}\frac{-d-a_s}{\tau_s}\right],
\]

\[
w_v=1-\exp[-(v_b/v_{\mathrm{sw}})^2],\qquad
a_b=(1-w_v)D+w_vd.
\]

固定 \(\tau_s=0.005\,m/s^2\)、\(v_{\mathrm{sw}}=0.03\,m/s\)。唯一的死区参数为
\(a_s=0.005+0.145\sigma(q_s)\,m/s^2\)，初始值0.05 m/s²。

## 6. 两阶段训练

阶段 A 仅使用完全处于 `dynamic_mask` 且不跨越换向或无效点的30步窗口。损失包含6/12/30步 RK4 位置与速度 rollout、权重0.25的逐点 Huber、`1e-4` Jacobian 正则，以及权重0.1的零速度偶对称锚定。RK4 每个 stage 都重新计算名义加速度、MLP与组合加速度。

阶段 B 冻结 MLP，只优化标量 \(q_s\)。它使用全部有效窗口，并加入静止总加速度和 stick-bound 损失。代码中的 stick-bound 项为

\[
\mathcal L_{\mathrm{stick}}=
\mathbb E\left[\left(\frac{\tau_s}{0.02}
\operatorname{softplus}\frac{|d|-a_s}{\tau_s}\right)^2\right].
\]

其中乘以 \(\tau_s\) 将无量纲 softplus margin 还原为加速度，再以验收尺度0.02 m/s²归一化；它与计划中的 softplus stick-bound 具有相同的零点和单调性，但明确了损失的量纲尺度。阶段结束时严格校验 MLP 权重哈希不变，不执行联合解冻微调。

训练固定 seed 7、float64、AdamW、batch 128；设备由完整30步 RK4、损失、反向传播和优化步的 CPU/GPU 基准决定。W&B 用于监控，所有 checkpoint 与逐样本数据只保存在本地。

## 7. 评价协议

统一比较 nominal、固定 v1.0、v1.5 的死区前 dynamic MLP 和完整 v1.5。报告动态逐点残差、全样本和 transition 加速度、0.1/0.2/0.5/1.0 s rollout、静止虚假运动、死区抑制、输出饱和、OOD 与输入梯度。08-17 测试期间不得改变权重、归一化、阈值或筛选规则。

## 8. 复现命令

完整运行命令见 [README.md](README.md)。正式验证命令为：

```bash
conda run -n isaac-sim python -m pytest \
  residual_dynamics/tests residual_dynamics_v1_5/tests -q

conda run -n isaac-sim python -m residual_dynamics_v1_5.cli all \
  --config residual_dynamics_v1_5/configs/mlp_deadzone.yaml
```

<!-- AUTO_RESULTS_START -->

## 自动生成的实验结果

> 本区块由 `python -m residual_dynamics_v1_5.cli report` 生成。

- Run：`v1_5_seed7_85aa5353_20260818_171603`
- 配置哈希：`85aa535394cdb3fbf64fb3134a012fa1b1363c8ea03a5b9307e1f21458970ccd`
- fit dataset 哈希：`06274425e956182d1b2a8affdd0e920c9a1ce54db56b3b3e980a2cdcbe81dce1`
- retrospective test dataset 哈希：`299f9cda7e20faeb2730a9fad2cdb7daca648cb703abdb4ced45440fdb575a2e`
- seed：`7`；设备：`cpu`
- 设备选择原因：`CUDA did not meet speedup threshold`
- 动态 MLP 最佳 epoch：`208`；死区最佳 epoch：`5`
- 拟合静止阈值：`0.071782 m/s²`
- 总训练时间：`458.1 s`
- W&B：`online`，run ID：`t7vgmg4a`
- W&B 初始化说明：`none`
- v1.5 checkpoint SHA-256：`aeed5b8d47168069f4b97d50115427b3d2283f644a03eacdb2dca3fe399a0ebc`
- 固定 v1.0 权重 SHA-256：`be25c1652ce47feeffe55aecf39ccc337dd33cc765f22832ae6c9d99265e9e0e`
- 总体验收：`PASS`

### 动态样本逐点残差

| 数据 | 模型 | RMSE (m/s²) | MAE | Bias | R² |
|---|---|---:|---:|---:|---:|
| validation | nominal | 0.078020 | 0.063960 | 0.026719 | -0.132861 |
| validation | v1 | 0.046245 | 0.036014 | 0.001730 | 0.601994 |
| validation | dynamic_mlp | 0.045256 | 0.035518 | 0.003124 | 0.618827 |
| validation | v1_5 | 0.045104 | 0.035364 | 0.003073 | 0.621379 |
| test | nominal | 0.133199 | 0.092052 | 0.035880 | -0.078240 |
| test | v1 | 0.102102 | 0.062377 | -0.012806 | 0.366448 |
| test | dynamic_mlp | 0.105845 | 0.063841 | -0.010530 | 0.319149 |
| test | v1_5 | 0.105780 | 0.063715 | -0.010419 | 0.319988 |
| stress | nominal | 0.182624 | 0.132270 | 0.038993 | -0.047766 |
| stress | v1 | 0.143448 | 0.097344 | -0.005015 | 0.353543 |
| stress | dynamic_mlp | 0.151339 | 0.100782 | -0.007698 | 0.280468 |
| stress | v1_5 | 0.151247 | 0.100537 | -0.007950 | 0.281335 |

### 多步 rollout

| 数据 | Horizon | 模型 | 位置 RMSE (m) | 速度 RMSE (m/s) | 位置 MSE 改善 | 速度 MSE 改善 |
|---|---:|---|---:|---:|---:|---:|
| validation | 0.100s | nominal | 0.000521 | 0.010246 | N/A | N/A |
| validation | 0.100s | v1 | 0.000376 | 0.007314 | 47.7% | 49.1% |
| validation | 0.100s | dynamic_mlp | 0.000382 | 0.007444 | 46.1% | 47.2% |
| validation | 0.100s | v1_5 | 0.000381 | 0.007404 | 46.4% | 47.8% |
| validation | 0.200s | nominal | 0.001853 | 0.017910 | N/A | N/A |
| validation | 0.200s | v1 | 0.001178 | 0.010799 | 59.5% | 63.6% |
| validation | 0.200s | dynamic_mlp | 0.001210 | 0.011168 | 57.4% | 61.1% |
| validation | 0.200s | v1_5 | 0.001203 | 0.011123 | 57.8% | 61.4% |
| validation | 0.500s | nominal | 0.009493 | 0.036221 | N/A | N/A |
| validation | 0.500s | v1 | 0.003952 | 0.013263 | 82.7% | 86.6% |
| validation | 0.500s | dynamic_mlp | 0.004258 | 0.014564 | 79.9% | 83.8% |
| validation | 0.500s | v1_5 | 0.004267 | 0.014758 | 79.8% | 83.4% |
| validation | 1.000s | nominal | 0.033826 | 0.064680 | N/A | N/A |
| validation | 1.000s | v1 | 0.010569 | 0.017535 | 90.2% | 92.7% |
| validation | 1.000s | dynamic_mlp | 0.011552 | 0.019341 | 88.3% | 91.1% |
| validation | 1.000s | v1_5 | 0.011447 | 0.019060 | 88.5% | 91.3% |
| test | 0.100s | nominal | 0.000734 | 0.014150 | N/A | N/A |
| test | 0.100s | v1 | 0.000616 | 0.011654 | 29.5% | 32.2% |
| test | 0.100s | dynamic_mlp | 0.000613 | 0.011608 | 30.3% | 32.7% |
| test | 0.100s | v1_5 | 0.000605 | 0.011447 | 31.9% | 34.6% |
| test | 0.200s | nominal | 0.002530 | 0.024391 | N/A | N/A |
| test | 0.200s | v1 | 0.001964 | 0.018372 | 39.7% | 43.3% |
| test | 0.200s | dynamic_mlp | 0.001946 | 0.018212 | 40.8% | 44.2% |
| test | 0.200s | v1_5 | 0.001910 | 0.017859 | 43.0% | 46.4% |
| test | 0.500s | nominal | 0.012865 | 0.048410 | N/A | N/A |
| test | 0.500s | v1 | 0.008318 | 0.028887 | 58.2% | 64.4% |
| test | 0.500s | dynamic_mlp | 0.008198 | 0.028680 | 59.4% | 64.9% |
| test | 0.500s | v1_5 | 0.007866 | 0.027277 | 62.6% | 68.3% |
| test | 1.000s | nominal | 0.044735 | 0.084411 | N/A | N/A |
| test | 1.000s | v1 | 0.023915 | 0.040417 | 71.4% | 77.1% |
| test | 1.000s | dynamic_mlp | 0.023505 | 0.040607 | 72.4% | 76.9% |
| test | 1.000s | v1_5 | 0.021385 | 0.036623 | 77.1% | 81.2% |
| stress | 0.100s | nominal | 0.001228 | 0.022836 | N/A | N/A |
| stress | 0.100s | v1 | 0.001104 | 0.020072 | 19.1% | 22.7% |
| stress | 0.100s | dynamic_mlp | 0.001125 | 0.020508 | 16.1% | 19.3% |
| stress | 0.100s | v1_5 | 0.001125 | 0.020542 | 16.0% | 19.1% |
| stress | 0.200s | nominal | 0.003864 | 0.036690 | N/A | N/A |
| stress | 0.200s | v1 | 0.003223 | 0.029808 | 30.4% | 34.0% |
| stress | 0.200s | dynamic_mlp | 0.003321 | 0.030868 | 26.1% | 29.2% |
| stress | 0.200s | v1_5 | 0.003323 | 0.030888 | 26.0% | 29.1% |
| stress | 0.500s | nominal | 0.017619 | 0.064890 | N/A | N/A |
| stress | 0.500s | v1 | 0.012317 | 0.042509 | 51.1% | 57.1% |
| stress | 0.500s | dynamic_mlp | 0.013198 | 0.046306 | 43.9% | 49.1% |
| stress | 0.500s | v1_5 | 0.013211 | 0.046743 | 43.8% | 48.1% |
| stress | 1.000s | nominal | 0.054801 | 0.101673 | N/A | N/A |
| stress | 1.000s | v1 | 0.029844 | 0.049840 | 70.3% | 76.0% |
| stress | 1.000s | dynamic_mlp | 0.033361 | 0.055321 | 62.9% | 70.4% |
| stress | 1.000s | v1_5 | 0.032896 | 0.054178 | 64.0% | 71.6% |

### 静止诊断

| 数据 | 模型 | Median | P95 | 虚假运动率 |
|---|---|---:|---:|---:|
| validation | nominal | 0.074289 | 0.121255 | 96.8% |
| validation | v1 | 0.016875 | 0.069231 | 41.3% |
| validation | dynamic_mlp | 0.013274 | 0.042976 | 33.6% |
| validation | v1_5 | 0.000028 | 0.000744 | 0.0% |
| test | nominal | 0.051281 | 0.157835 | 80.1% |
| test | v1 | 0.026806 | 0.111278 | 61.9% |
| test | dynamic_mlp | 0.025099 | 0.089311 | 58.4% |
| test | v1_5 | 0.000099 | 0.018086 | 4.9% |
| stress | nominal | 0.065520 | 0.100832 | 100.0% |
| stress | v1 | 0.020045 | 0.227155 | 50.0% |
| stress | dynamic_mlp | 0.063540 | 0.288291 | 100.0% |
| stress | v1_5 | 0.002462 | 0.218252 | 19.6% |

### 验收条件

- `PASS` — `deadzone_not_at_boundary`
- `PASS` — `test_20260817220124_trial_001_tracking_duration_no_degradation`
- `PASS` — `test_20260817220343_trial_003_tracking_duration_no_degradation`
- `PASS` — `test_20260817221148_trial_001_tracking_duration_no_degradation`
- `PASS` — `test_20260817221232_trial_002_tracking_duration_no_degradation`
- `PASS` — `test_20260817221315_trial_003_tracking_duration_no_degradation`
- `PASS` — `test_20260817222433_trial_001_tracking_duration_no_degradation`
- `PASS` — `test_finite`
- `PASS` — `test_h30_position`
- `PASS` — `test_h30_velocity`
- `PASS` — `test_saturation`
- `PASS` — `test_static_false_motion`
- `PASS` — `test_static_median`
- `PASS` — `test_static_p95`
- `PASS` — `validation_20260815200306_trial_003_tracking_duration_no_degradation`
- `PASS` — `validation_20260815201206_trial_003_tracking_duration_no_degradation`
- `PASS` — `validation_20260815202009_trial_003_tracking_duration_no_degradation`
- `PASS` — `validation_20260815203032_trial_003_tracking_duration_no_degradation`
- `PASS` — `validation_20260815204723_trial_003_tracking_duration_no_degradation`
- `PASS` — `validation_20260815205548_trial_003_tracking_duration_no_degradation`
- `PASS` — `validation_dynamic_vs_v1`
- `PASS` — `validation_finite`
- `PASS` — `validation_h30_position`
- `PASS` — `validation_h30_velocity`
- `PASS` — `validation_saturation`
- `PASS` — `validation_static_false_motion`
- `PASS` — `validation_static_median`
- `PASS` — `validation_static_p95`

### 诊断边界

| 数据 | OOD比例 | MLP饱和率 | 死区激活比例 | 梯度范数P99 |
|---|---:|---:|---:|---:|
| validation | 0.0% | 0.1% | 55.2% | 1.452098 |
| test | 2.4% | 1.0% | 69.9% | 1.280473 |
| stress | 11.4% | 3.8% | 35.7% | 1.594547 |

08-17 数据是回顾性外部测试：它没有参与权重、归一化或死区参数拟合，但此前已参与方案分析，因此不能视作完全未见的 prospective test。

<!-- AUTO_RESULTS_END -->

## 9. 局限与部署前工作

当前模型使用单 seed、固定 held-out validation 和回顾性外部测试；闭环数据存在状态—控制相关性，训练 rollout 又把实测 \(\theta,\omega\) 当作外生序列。平滑死区是静摩擦的可微近似，不能显式描述 breakaway、滞回、碰撞和历史依赖。模型部署前仍需完成独立 prospective 数据验证、闭环稳定性/实时性分析、残差缩放保护、OOD 策略，以及 predictor 与 NMPC 一致的离散实现。
