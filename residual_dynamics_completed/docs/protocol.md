# 固定实验协议

## 范围与隔离

此目录是独立实验包。运行时仅依赖 Python、NumPy、PyTorch 和 Matplotlib，不需要 Isaac Sim、旧 residual 包或原始日志。
`provenance/migrate.py` 和 `provenance/check_migration.py` 是仅在迁移时访问旧目录的显式审计工具，不被训练或评估导入。
原始数据、旧实现、旧权重以及论文实物实验没有被修改。

## 数据与采样

完整复制 processed_rbf_pv，逐文件 SHA-256 校验；manifest 不改写。划分为 79 train、6 validation、6 test、5 stress。
每个训练/验证/测试划分覆盖 constant、sine、triangle、trapezoid；不混入 stress。
数据加载顺序保持 manifest 顺序。窗口保持同一 segment，并要求包含端点在内的全部 rbf_valid 为真。
窗口中点位置落在 [0,0.7] m 内；位置、速度、actuation 分箱及 0.4 empirical / 0.4 joint / 0.2 actuation 三流分层采样忠实迁移 v2。
训练窗口长 33，136,964 个候选；所有模型使用相同采样规则、seed 和 batch size。各模型可能因独立早停在不同阶段边界进入第二阶段，因此不宣称全过程抽样序列逐 batch 相同。
验证候选每 5 个取一个，再以整数 linspace 均匀下采样至最多 512；测试窗口长 60，stride 5，上限 2048，实际为 2028。
MLP 使用 manifest.shadow_full_state_normalization 中的训练集五维 mean/std，逐位复核其为 79 个训练 trial 的原始 processed features 统计量（不是重新按 rbf_valid 过滤计算），不含验证/测试数据。

## 模型

统一接口为五维 features = [p_b, v_b, theta, omega, a_nom] 到标量加速度残差；每个模型只消费其所需输入。

- Nominal: r_a = 0，无可训练参数。
- Profile-only: 13 个位置归一化高斯基，中心均匀覆盖 [0,0.7] m、宽度 0.065 m；delta 的系数约束为 ±0.035 rad。raw = -gamma*g*[sin(theta+delta(p))-sin(theta)]。13 个可训练参数。
- Friction-only: 同样 13 个位置基，raw = -c(p)*tanh(v/0.025)-d(p)*v；Coulomb-like 系数在 [0,0.15] m/s²、viscous 系数在 [0,0.50] s⁻¹ 内。26 个可训练参数，输出始终与速度反向或为零。
- Structured: 两项相加后统一输出变换，39 个可训练参数。为直接加载旧 checkpoint，保留不参与运算/优化的 dormant blackbox/kappa state-dict entries；不属于本次模型结构。
- RBF-PV: 9×9 归一化二维高斯基；p 中心均匀覆盖 [0,0.7] m，v 中心为 [-0.50,-0.30,-0.15,-0.06,0,0.06,0.15,0.30,0.50] m/s，宽度分别 0.10 m、0.10 m/s。raw 为 81 个自由系数的基展开，不使用 OOD gate 或系数硬限幅。
- MLP: 标准化后的五输入 5–16–16–1 网络，隐藏层 tanh；385 个可训练参数。最后一层零初始化，其他层沿用 PyTorch 初始化。保留 v1 输出 0.25*tanh(z)，物理 raw 定义为 0.25*z。

所有模型只应用一次 r_a = 0.25*tanh(raw/0.25) m/s²；在训练、验证、RK4 各 stage 和评估中一致，不做事后额外限幅。
物理参数投影不等于输出裁剪。输出饱和诊断阈值为 |r_a| >= 0.99*0.25。

## 损失与模型选择

固定 seed=7、CPU 单线程、float64、确定性 PyTorch、Adam、batch=256、梯度范数裁剪 1.0。
第一阶段最多 80 epochs×16 steps、lr=0.003；第二阶段最多 50×8、lr=0.0003。
各阶段 patience=20，改善阈值 1e-6。第二阶段从全局最优状态开始、重新建立 Adam，随机生成器延续第一阶段状态。

共同数据损失 L = L_integral + 0.25 L_point + w_roll L_roll，另加表示相关正则。

- 点加速度 Huber beta=0.02 m/s²。
- 积分一致性将 residual 在记录特征上作梯形积分，除以时间长度，比较 smoothed velocity 的平均变化率减 nominal 平均加速度。H={3,6,12}，权重 {0.50,0.30,0.20}，Huber beta=0.02。
- 第二阶段 w_roll=0.5（第一阶段 0）。H={6,12,30,33}，权重 {0.35,0.30,0.20,0.15}。位置、速度尺度 0.01 m、0.05 m/s，归一化 Huber beta=1。平滑位置/速度误差加 0.25 倍 measured causal position/velocity 辅助误差。
- Profile/friction: 各激活系数按物理上限归一化，一阶差分平方均值×0.002，加二阶差分平方均值×0.01。
- RBF-PV: 网格系数除以 0.25 后，沿两个轴的一阶差分平方均值各乘 0.002，二阶各乘 0.01。
- MLP: 所有参数平方和×1e-5；Adam 无额外 weight decay。

验证分数 S=0.15 R_pt + 0.20 R_int + 0.325 R_p,33 + 0.325 R_v,33。
每项为相对 nominal 的 RMSE 比值；积分 RMSE 使用三个时域的加权平方误差。模型选择不使用测试集。
配置文件是固定实验的记录，不是通用超参数搜索 API；改变结构/损失常数时需同步代码、配置、测试和报告。

## 因果评估与度量

初态 [pb,vb,theta,vrz]、pending 队列、记录的后续 committed command、dt、geometry、FOPDT 全部共用。
先消耗初始 pending 队列，再使用 committed[1:]；重建结果逐 batch 与日志 response_command 对齐校验。
一阶速度响应解析求解 RK4 的起点、中点、终点；物理状态用 RK4 推进。
每个 RK4 stage 由预测状态重算 omega 和 a_nom，以及各模型所需特征。不用未来测量状态重置任何一步。
一次连续推进 60 步，在 6,12,...,60 保存 endpoint 误差；真实 dt 保留，60 Hz 为近似说明。
主结果为 measured causal position/velocity RMSE；同时保存 MAE、点加速度误差、饱和比例。
报告 pooled window RMSE，以及逐 trial、逐轨迹类型分组指标。非有限状态/残差保留且使验收失败。
窗口重叠，不作为独立重复；单种子不绘制 seed 标准差带，不进行显著性优越声明。

## 产物与验收

每模型保存 config/源代码 hash、数据/window hash、软件版本、训练历史、best.pt、last.pt、best epoch、score、耗时。
last.pt 保存 optimizer、随机状态、阶段/epoch 和早停状态，支持同配置的 epoch 边界精确恢复。
不同配置/训练源版本使用不同 hash 运行目录；不覆盖已完成的运行。results/test 不允许无提示覆盖。
accept.py 从逐窗口 CSV 重新计算所有汇总，并从保存的状态预测再次核对每个误差。
Structured 重训和旧 checkpoint 的 state_dict 与完整训练历史须明确核对，原实物权重不被替换。
