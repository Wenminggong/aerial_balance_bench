# 实测人速度 NMPC 扩展：验证记录

2026-10-15，在 Isaac Sim 4.2 / Isaac Lab 1.4、acados 0.5.4 环境完成。
实现和配置见 [acados 模型文档第 15 节](acados_unified_nmpc.md#15-实测人速度的常值预测控制)。

## 范围和配对设置

新增开关同时影响延迟 predictor、OCP 动力学和角速度代价，因此以下比较衡量的是
这一组合扩展的效果，不单独归因于 predictor 或某个代价项。

先执行 8 组短测：`random-walk/cooperative × enabled=false/true × D=0/4`，
每组 2 个环境、每个环境 2 个 1 秒 episode。随后执行 4 组配对比较：
两种人行为各自比较反应式与实测人速度建模，每组 16 个环境、20 秒、seed=666，
统一固定目标任务，D=4、N=30、h=1/60，原始观测维数 80（11 维前缀和预览）。
四组均完成 16 个完整 episode，均在第 1200 步 timeout，无提前失败或中途 reset。

基础配置取自当时的 `unified_tracking_constant.yaml` 和 `acados_nmpc_unified.yaml`。
没有修改当前控制权重或人参数；实验副本只调整环境数量、时长、人行为模式和
短测的延迟设置。两组 policy 的唯一区别为统一开关。

- 人 OU：mu=0，theta∈[0.5,1.0]，sigma∈[0.01,0.05]，clip=0.03 m/s。
- 人最大加速度 0.5 m/s²，位置回正参数采样保持开启。
- NMPC 标称响应 tau=0.187 s、gain=1、bias=-0.00485 m/s。
- 环境保留响应参数随机化和高斯响应噪声；并非无噪声理想模型实验。
- 阶段权重 `[20,4,0.1,0.2,0.2,0.05,0.25]`，终端权重 `[100,20,0.5,1,1,0.05]`。

逐项检查确认：每个配对的环境配置、初始完整观测、人参数相同，整个 rollout
的 OU 状态逐元素相同。random-walk 的实际人速度最大差异约 6.8e-7 m/s；
cooperative 的最大差异约 0.138 m/s，这是人 CPID 根据不同闭环下的小球状态反馈的结果，
不是随机驱动或初值不同。完整检查见 `pair_checks.json`。

## 原 evaluator 结果

单位均为 m；末秒误差沿用 evaluator 的最后 1 秒平均绝对误差定义。
MAE/RMSE 沿用原 evaluator 的 episode 聚合，未改变指标或成功判据。

| 人模式 | 控制模型 | MAE | RMSE | 末秒误差 | 成功率 |
|---|---|---:|---:|---:|---:|
| random-walk | 反应式 | 0.026203 | 0.064190 | 0.005623 | 81.25% |
| random-walk | 实测人速度 | 0.023376 | 0.063664 | 0.001803 | 100.00% |
| cooperative | 反应式 | 0.024300 | 0.065731 | 0.002926 | 100.00% |
| cooperative | 实测人速度 | 0.023136 | 0.063585 | 0.001852 | 100.00% |

random-walk 的 MAE 降低 10.8%，末秒误差降低 67.9%。
cooperative 的 MAE 降低 4.8%，末秒误差降低 36.7%。
这是单个 seed、每组 16 个 episode 的工程验证结果，不能据此宣称对其他人参数、
任务或种子普遍改善。cooperative 两组成功率均已达到 100%，此指标没有区分度。

## 延迟跨度的状态预测误差

将动作前记录的 `policy_predicted_*[k]` 对齐 `observations[k+4]`，排除跨 reset
样本，每组每个状态共有 19136 个环境时刻样本。以下为全样本 RMSE，不是 evaluator 指标。

| 人模式 | 控制模型 | pb (m) | vb (m/s) | theta (rad) | omega (rad/s) | vrz (m/s) |
|---|---|---:|---:|---:|---:|---:|
| random-walk | 反应式 | 0.0001201 | 0.0023277 | 0.0012334 | 0.0223055 | 0.0185463 |
| random-walk | 实测人速度 | 0.0001165 | 0.0022523 | 0.0008777 | 0.0186488 | 0.0185703 |
| cooperative | 反应式 | 0.0000847 | 0.0016622 | 0.0015236 | 0.0260695 | 0.0184132 |
| cooperative | 实测人速度 | 0.0000880 | 0.0015960 | 0.0009195 | 0.0198693 | 0.0184325 |

两种人行为下梁角与角速度预测误差均下降；cooperative 的球位置预测 RMSE 略有
增加，所以不是所有状态分量都改善。无人机速度误差基本不变，与本次未修改
无人机响应模型的范围一致。两组闭环访问的状态不同，这不是同一状态样本上的
离线模型精度消融实验。

## 常值速度假设的误差

从实际 `step_human_vel_z[k-1]` 重建动作前人速度，对比未来实际动作前速度，
只做离线诊断。排除初始／reset 后无法重建的测量及跨 reset 样本。
这里报告两个区间末端的误差，不是整个 horizon 内各阶段误差的平均值。

| 人模式 | 控制模型 | D=4 (0.067 s) 速度 RMSE (m/s) | D+N=34 (0.567 s) 速度 RMSE (m/s) |
|---|---|---:|---:|
| random-walk | 反应式 | 0.007554 | 0.018244 |
| random-walk | 实测人速度 | 0.007554 | 0.018244 |
| cooperative | 反应式 | 0.009260 | 0.022679 |
| cooperative | 实测人速度 | 0.009772 | 0.037143 |

每组短／长区间分别有 19120／18640 个有效环境时刻样本。cooperative 在建模后
长区间外推误差明显增大：人会根据新的球运动改变动作，约半秒的常值假设不一定
可靠。当前闭环指标改善，不能被解释为未来人的完整轨迹预测更准确。控制器
每个周期重测速度，后续研究仍需要区分当前相对运动建模的收益与长期外推误差。

## 求解、时序和性能

| 人模式 | 控制模型 | 求解成功率 | fallback | 整批 act 平均 (ms) | p95 (ms) | deadline miss |
|---|---|---:|---:|---:|---:|---:|
| random-walk | 反应式 | 100.0% | 0.0% | 22.800 | 24.826 | 100.0% |
| random-walk | 实测人速度 | 100.0% | 0.0% | 22.524 | 23.690 | 100.0% |
| cooperative | 反应式 | 100.0% | 0.0% | 23.032 | 24.768 | 100.0% |
| cooperative | 实测人速度 | 100.0% | 0.0% | 22.084 | 23.391 | 100.0% |

时限是 16.667 ms。这里统计的是整批 16 个环境的 `policy.act`，包括输入处理、
Torch predictor、CPU/GPU 数据传输和 batch solve，包含第一步；不包含环境物理 step。
这台机器上的整批控制未达到 60 Hz，不能用单个 acados 内部 solve 时间替代端到端
时限验证，也不能直接推断单环境性能。两模式间小于 1 ms 的差异不作性能优劣结论。

8 组短测全部 solver success=100%、fallback=0%。12 组运行均检查了有限观测、
动作上限、日志形状。开启模式下，所有非 reset 步 `used[k+1]` 和
`step_human_vel_z[k]` 完全相等；短测 reset 后首个动作使用新 episode 的零初始
人速度，上一 episode 的非零终止速度仍保留在 step 日志。D=4 时两个 used 字段
相同，D=0 时 predictor used 为零。关闭模式全部 used 为零，实际人速度仍正常记录。

## 自动化验证与既有失败

新增 16 项参数化测试全部通过。最终全仓 pytest 结果：**390 passed、1 skipped、4 failed**。
跳过项是沙箱内 CUDA 不可用的人模型设备测试；上述 12 组 Isaac 验证实际使用 GPU。
本次修改相关的模型、acados、predictor、参考预览、动作延迟、人行为和日志测试均通过。
改动 Python 文件编译、YAML 严格加载和 `git diff --check` 通过。

4 项失败在独立临时副本中替换为修改前的 acados/predictor 源码后同样复现，
未为本次扩展修改这些既有行为或断言：

- `test_nffb_policy.py::test_compensation_evaluation_configs_resolve_and_construct` 的 D=8 配置：期望 tau=0.155，当前配置解析为 0.187。
- `test_rl_observation_adapter.py::test_unified_rl_predictor_eval_configs_preserve_checkpoint_contract`：期望 delay_step=`auto`，当前配置为 8。
- `test_velocity_response_compensator.py::test_exact_inverse_matches_simulator_nominal_step`。
- `test_velocity_response_compensator.py::test_pending_commands_are_applied_before_inverse`。

后两项是原速度响应补偿测试的数值断言失败。完整输出保存在 `tests_final.log`
和 `baseline_tests.log`，未把全仓测试表述为全部通过。

## 产物与复算

产物独立归档到
[`logs/human_drone_collaboration/human_motion_validation_20261015`](../logs/human_drone_collaboration/human_motion_validation_20261015)，
未覆盖历史实验。每个 `results/<run_name>` 内有输入环境、policy、run YAML、
resolved metadata、summary 和 rollout NPZ；`analysis.json` 包含完整离线指标，
`pair_checks.json` 包含配对一致性检查。输入中的 `/tmp` 路径保留原始运行来源。

```bash
python scripts/analyze_human_motion_rollouts.py \
  logs/human_drone_collaboration/human_motion_validation_20261015/results/paired_* \
  --output /tmp/human_motion_recomputed.json
```

需要精确复跑时，通过 `--env_config` 和 `--policy_config` 指向相应目录中保存的
`input_env_config.yaml` 与 `input_policy_config.yaml`，并使用新的 `--run_name`。
基础配置后续可以继续调参，归档输入记录的是本次实验实际使用的值。
