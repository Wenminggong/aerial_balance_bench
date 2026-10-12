# Residual Dynamics v1.5

该目录是独立于 `residual_dynamics/` v1.0 的离线训练与评价包。它训练一个只面向运动状态的五输入 MLP，并在 MLP 给出的自由驱动力外增加一个单参数平滑静止死区。包内不导入 v1.0 的 Python 代码；仅在离线比较时读取固定的 v1.0 便携权重。

## 模型

五个输入按固定顺序排列：

```text
[pb, vb, theta, omega, a_nom]
```

动态 MLP 为 `5→16→16→1`，隐藏层使用 `tanh`，输出为 `0.20*tanh(y)`。令其输出为 `r_d`，自由驱动力、平滑死区、速度融合和最终加速度分别为：

```text
d = a_nom + r_d
D(d) = tau * [softplus((d-a_s)/tau) - softplus((-d-a_s)/tau)]
w_v = 1 - exp[-(vb/v_sw)^2]
a_b = (1-w_v) D(d) + w_v d
```

其中 `tau=0.005 m/s²`、`v_sw=0.03 m/s` 固定，只学习 `a_s∈[0.005,0.15] m/s²`。阶段 A 只用稳定运动样本训练 MLP；阶段 B 冻结 MLP，只拟合 `a_s`。

## 数据协议

- 2026-08-15：20 条训练、9 条验证、1 条 stale-beam 排除。
- 2026-08-17：14 条完成 trial 作为回顾性外部测试，5 条未完成 trial 作为 stress 分析。
- 训练归一化只由 20 条训练 trial 中的 `dynamic_mask` 样本计算。
- 保存的是约 60 Hz 控制快照，并非完整 240 Hz Laser 原始流。
- 处理数据、checkpoint、完整预测和 W&B 缓存保留在本目录，但由 `.gitignore` 排除。

## 运行

从仓库根目录执行：

```bash
conda run -n isaac-sim python -m residual_dynamics_v1_5.cli prepare \
  --config residual_dynamics_v1_5/configs/mlp_deadzone.yaml

conda run -n isaac-sim python -m residual_dynamics_v1_5.cli train \
  --config residual_dynamics_v1_5/configs/mlp_deadzone.yaml

conda run -n isaac-sim python -m residual_dynamics_v1_5.cli evaluate \
  --config residual_dynamics_v1_5/configs/mlp_deadzone.yaml

conda run -n isaac-sim python -m residual_dynamics_v1_5.cli test \
  --config residual_dynamics_v1_5/configs/mlp_deadzone.yaml

conda run -n isaac-sim python -m residual_dynamics_v1_5.cli report \
  --config residual_dynamics_v1_5/configs/mlp_deadzone.yaml
```

完整流程可用：

```bash
conda run -n isaac-sim python -m residual_dynamics_v1_5.cli all \
  --config residual_dynamics_v1_5/configs/mlp_deadzone.yaml
```

W&B 默认在线优先并在初始化失败时回落到离线模式。可用 `--wandb-mode disabled` 做无网络测试。W&B 只记录配置、哈希和聚合指标，不上传原始 CSV、NPZ、checkpoint 或逐样本预测。

## 测试

```bash
conda run -n isaac-sim python -m pytest \
  residual_dynamics/tests residual_dynamics_v1_5/tests -q
```

`results/summary.json` 是机器可读的最终汇总，`REPORT.md` 给出完整方法和自动生成结果。该模块尚未接入 C++、状态预测器或 acados，离线改善不构成闭环稳定性保证。
