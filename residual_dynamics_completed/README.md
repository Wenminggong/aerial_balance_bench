# Residual Dynamics Completed

独立的六模型离线残差动力学实验：Nominal、Profile-only、Friction-only、RBF-PV、MLP、Structured。
所有模型使用复制后的同一份 79/6/6 数据划分，以及一致的 tanh 输出、数据损失、验证选择和因果 rollout 协议。

论文及图例将 Profile-only 命名为 Beam-Profile Residual（BPR），将 Friction-only 命名为
Friction-Inspired Dissipative Residual（FDR）。RBF-PV 全名为 Position–Velocity Radial Basis Function Model，
MLP 为 Multilayer Perceptron。上述是显示名称更新；CSV、checkpoint 和 CLI 中的
`profile_only`、`friction_only` 等内部 ID 保持不变，训练与评估结果未改变。

## 交付入口

- [最终实验报告](docs/experiment_report.md)：全部模型配置、训练结果、H=30/60 和逐轨迹类型对比、局限。
- [设计与固定协议](docs/protocol.md)：模型公式、损失、窗口和命令队列定义。
- [验收记录](results/acceptance.json)：重算、数值检查和 Structured 重训复现。
- [数据迁移校验](provenance/migration.json) 与 [迁移一致性](provenance/parity.json)。
- [六模型图](results/test/six_model_rollout_comparison.pdf) 与 [长表 CSV](results/test/metrics_long.csv)。
- `results/test/window_errors.csv`：逐窗口×时域×模型误差；`*_predictions.npz` 保存完整 60 步预测。
- `runs/*/best.pt` / `last.pt` / `history.json` / `config.json`：最优权重、恢复点、历史、实际配置和 hashes。

## 运行环境

从 aerial_balance_bench 根目录执行。已验收环境：Python 3.10、PyTorch 2.8.0+cu129、NumPy 1.26.4；推理和训练均为 CPU 单线程 float64。
Matplotlib 用于绘图；不需要 Isaac Sim 仿真、GPU、W&B、旧 residual 包或原始日志。

```bash
# 基础验收（无需旧目录）
conda run --no-capture-output -n isaac-sim python -m unittest discover -s residual_dynamics_completed/tests -v

# 五模型从头训练，Structured 优先；已有相同配置完成结果会跳过
conda run --no-capture-output -n isaac-sim python -m residual_dynamics_completed.train --model all

# 单模型及同配置 epoch 边界恢复
conda run --no-capture-output -n isaac-sim python -m residual_dynamics_completed.train --model mlp --resume

# 统一测试（拒绝覆盖已存在的完整测试结果）
conda run --no-capture-output -n isaac-sim python -m residual_dynamics_completed.evaluate

# 重算验收和报告/绘图
conda run --no-capture-output -n isaac-sim python -m residual_dynamics_completed.accept
conda run --no-capture-output -n isaac-sim python -m residual_dynamics_completed.compare
```

`--model` 支持 nominal、profile_only、friction_only、rbf_pv、mlp、structured、all；nominal 不训练。
评估可显式使用 `--split validation` 或 `--split stress`，与 test 分目录保存；默认只运行正文 test。
`compare --split ...` 对应生成图件和该 split 报告。不要以 stress/test 调参。
如需重新执行已完成的评估，应先将 `results/test` 移到明确命名的归档目录，再运行；不要删除原始实验。
固定 `configs/comparison.json` 记录批准协议；不是任意配置覆盖接口。训练源代码/配置改变会创建不同 hash 的 run，绝不覆盖旧 run。

## 仅供迁移时的命令

以下两条是唯一需要原 residual_dynamics / residual_dynamics_v2 的命令；日常训练、评估、验收无需执行。

```bash
python3 residual_dynamics_completed/provenance/migrate.py
conda run --no-capture-output -n isaac-sim python -m residual_dynamics_completed.provenance.check_migration
```

数据是实际副本而非符号链接；原 manifest 的旧路径只作为溯源，不被运行时解引用。
旧 Structured checkpoint 副本仅用于复现核对，不作为任一模型的训练初始化。
该目录不修改实物部署权重；新离线结果不替换既有实物实验数据。

## 结果解读

本实验是单种子比较，并未预设 Structured 在所有模型和指标上最优。
MLP 离线误差最低；Structured 的价值应结合物理结构、紧凑表示及已有闭环验证来表述，不能由此声称其闭环优于 MLP。
Structured 的重训在浮点精度范围复现旧权重；准确差异见验收 JSON，勿称 checkpoint 文件逐字节相同。
