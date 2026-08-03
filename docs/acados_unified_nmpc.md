# acados Unified-NMPC：模型、优化问题与实现

本文档给出 'AcadosNMPCPolicy' 的完整数学 formulation 以及论文符号到实际实现的逐项映射。该控制器是独立的新 baseline；原有基于 do-mpc 的 'NMPCPolicy'、runner 和 YAML 均未被替换或修改。

## 1. 问题定义和实际向量

控制周期记为 $h>0$，预测步数为 $N$，当前时刻为 $k$。对任意预测 stage $i=0,\ldots,N$，定义

$$
\mathbf{x}_{i|k}
=
\begin{bmatrix}
p_{b,i|k} & v_{b,i|k} & \theta_{i|k} & v_{rz,i|k} & c_{z,i|k}
\end{bmatrix}^{\!\top}
\in\mathbb{R}^{5},
$$

其中 $p_b$ 是 benchmark 球位置，$v_b$ 是球速度，$\theta$ 是梁角，$v_{rz}$ 是无人机实际竖直响应速度，$c_z$ 是 velocity interface 的累积速度命令。控制输入为

$$
\mathbf{u}_{i|k}=u_{i|k}=\Delta c_{z,i|k}\in\mathbb{R},
\qquad c^+_{z,i|k}=c_{z,i|k}+u_{i|k}.
$$

每个 stage 的时变参数为

$$
\mathbf{p}_{i|k}
=
\begin{bmatrix}
p_{g,i|k} & v_{g,i|k} & \tau & K & b
\end{bmatrix}^{\!\top}
\in\mathbb{R}^{5},
$$

其中 $p_g,v_g$ 是位置和速度参考，$\tau,K,b$ 分别是一阶速度响应的时间常数、静态增益和偏置。第一版在一次 rollout 内使用配置给定的标称 $(\tau,K,b)$，不读取逐环境 oracle 参数，也不做在线辨识。

实际向量顺序固定如下，CasADi、acados、日志和测试代码不得重新排列：

| 向量 | 维数 | 实际分量顺序 |
|---|---:|---|
| $\mathbf{x}$ | 5 | '[pb, vb, theta, vrz, command_z]' |
| $\mathbf{u}$ | 1 | '[delta_command_z]' |
| $\mathbf{p}$ | 5 | '[pg, vg, tau_s, gain, bias]' |

## 2. 连续几何和球动力学

令 $L$ 为梁长，$l_r$ 为绳长，$d_b$ 为 benchmark 球坐标到论文物理坐标的偏移。实现中

$$
d_b=\texttt{plank\_slide\_length}+\texttt{beam\_block\_offset}=0.33\ {\rm m}.
$$

绳角和梁角速度为

$$
s_\beta(\theta)=\frac{L}{l_r}(1-\cos\theta),\qquad
\beta(\theta)=\arcsin\!\left(\operatorname{clip}(s_\beta,-1+\epsilon,1-\epsilon)\right),
$$

$$
\omega(\theta,v_{rz})
=-
\frac{v_{rz}\cos\beta(\theta)}
{L\,\operatorname{safe}_{\epsilon}(\cos(\beta(\theta)-\theta))},
$$

其中

$$
\operatorname{safe}_{\epsilon}(q)=
\begin{cases}
q,&|q|\geq\epsilon,\\
\epsilon,&0\leq q<\epsilon,\\
-\epsilon,&-\epsilon<q<0.
\end{cases}
$$

球的等效质量、滚动系数与加速度为

$$
J_b=\rho_m m_b r_b^2,\qquad
M_b=m_b+\frac{J_b}{r_b^2},\qquad
\gamma=\frac{m_b}{M_b},
$$

$$
a_b(p_b,\theta,v_{rz})
=\gamma\left[(p_b+d_b-L)\omega(\theta,v_{rz})^2-g\sin\theta\right].
$$

给定瞬时 $v_{rz}$，物理状态 $\mathbf{z}=[p_b,v_b,\theta]^\top$ 的导数为

$$
\dot{\mathbf{z}}
=\mathbf{f}(\mathbf{z},v_{rz})
=
\begin{bmatrix}
v_b\\ a_b(p_b,\theta,v_{rz})\\ \omega(\theta,v_{rz})
\end{bmatrix}.
$$

这些定义与 'VelocityInterfaceModel' 使用相同的坐标偏移、arcsine 截断和分母保护。

## 3. 一阶速度响应和精确离散

在一个 stage 内，命令 $c_z^+=c_z+u$ 保持不变，其静态响应目标为

$$
\bar v_{rz}=Kc_z^+ + b.
$$

当 $\tau>0$ 时，stage 内任意 $s\in[0,h]$ 的精确响应为

$$
v_{rz}(s)=\bar v_{rz}+\left(v_{rz}(0)-\bar v_{rz}\right)e^{-s/\tau}.
$$

当 $\tau=0$ 时模型明确定义为静态增益/偏置响应：

$$
v_{rz}(s)=\bar v_{rz},\qquad s\in[0,h].
$$

在 RK4 的三个时间节点使用精确响应。记

$$
v_0=v_{rz}(0),\qquad v_{1/2}=v_{rz}(h/2),\qquad v_1=v_{rz}(h),
$$

则

$$
\begin{aligned}
\mathbf{k}_1 &= \mathbf{f}(\mathbf{z}_{i|k},v_0),\\
\mathbf{k}_2 &= \mathbf{f}(\mathbf{z}_{i|k}+\tfrac{h}{2}\mathbf{k}_1,v_{1/2}),\\
\mathbf{k}_3 &= \mathbf{f}(\mathbf{z}_{i|k}+\tfrac{h}{2}\mathbf{k}_2,v_{1/2}),\\
\mathbf{k}_4 &= \mathbf{f}(\mathbf{z}_{i|k}+h\mathbf{k}_3,v_1),\\
\mathbf{z}_{i+1|k} &= \mathbf{z}_{i|k}
+\frac{h}{6}(\mathbf{k}_1+2\mathbf{k}_2+2\mathbf{k}_3+\mathbf{k}_4),\\
v_{rz,i+1|k} &= v_1,\\
c_{z,i+1|k} &= c^+_{z,i|k}.
\end{aligned}
$$

由此得到完整离散映射

$$
\mathbf{x}_{i+1|k}
=\mathbf{F}_h(\mathbf{x}_{i|k},\mathbf{u}_{i|k},\mathbf{p}_{i|k}).
$$

'baselines/acados_nmpc_core.py' 同时提供纯 NumPy 的 'discrete_dynamics_numpy()' 和等价的 CasADi SX 表达式。acados 使用 'integrator_type=DISCRETE'，不会对已经离散化的模型再次积分。

## 4. 优化目标

stage nonlinear least-squares residual 为

$$
\mathbf{y}_{i|k}=
\begin{bmatrix}
p_{b,i|k}-p_{g,i|k}\\
v_{b,i|k}-v_{g,i|k}\\
\theta_{i|k}\\
\omega(\theta_{i|k},v_{rz,i|k})\\
v_{rz,i|k}\\
c_{z,i|k}+u_{i|k}\\
u_{i|k}
\end{bmatrix}
\in\mathbb{R}^{7}.
$$

terminal residual 为

$$
\mathbf{y}_{N|k}^{e}=
\begin{bmatrix}
p_{b,N|k}-p_{g,N|k}\\
v_{b,N|k}-v_{g,N|k}\\
\theta_{N|k}\\
\omega(\theta_{N|k},v_{rz,N|k})\\
v_{rz,N|k}\\
c_{z,N|k}
\end{bmatrix}
\in\mathbb{R}^{6}.
$$

默认对角权重为

$$
\mathbf{W}=\operatorname{diag}(5.0,0.5,0.1,0.05,0.05,0.5,1.0),
$$

$$
\mathbf{W}_e=\operatorname{diag}(25.0,2.5,0.5,0.25,0.25,0.5).
$$

对软约束定义 $\mathbf{s}_{i}^{l},\mathbf{s}_{i}^{u}\in\mathbb{R}_{\geq0}^{3}$。完整目标为

$$
\begin{aligned}
J_k={}&\sum_{i=0}^{N-1}
\left[
\frac12\mathbf{y}_{i|k}^{\top}\mathbf{W}\mathbf{y}_{i|k}
+\mathbf{z}_{l}^{\top}\mathbf{s}_{i}^{l}
+\mathbf{z}_{u}^{\top}\mathbf{s}_{i}^{u}
+\frac12(\mathbf{s}_{i}^{l})^{\top}\mathbf{Z}_{l}\mathbf{s}_{i}^{l}
+\frac12(\mathbf{s}_{i}^{u})^{\top}\mathbf{Z}_{u}\mathbf{s}_{i}^{u}
\right]\\
&+\frac12(\mathbf{y}_{N|k}^{e})^{\top}\mathbf{W}_e\mathbf{y}_{N|k}^{e}
+\mathbf{z}_{l}^{\top}\mathbf{s}_{N}^{l}
+\mathbf{z}_{u}^{\top}\mathbf{s}_{N}^{u}
+\frac12(\mathbf{s}_{N}^{l})^{\top}\mathbf{Z}_{l}\mathbf{s}_{N}^{l}
+\frac12(\mathbf{s}_{N}^{u})^{\top}\mathbf{Z}_{u}\mathbf{s}_{N}^{u},
\end{aligned}
$$

其中默认

$$
\mathbf{z}_{l}=\mathbf{z}_{u}=10^4\mathbf{1}_3,\qquad
\mathbf{Z}_{l}=\mathbf{Z}_{u}=10^4\mathbf{I}_3.
$$

输入 $u=\Delta c_z$ 已直接惩罚命令增量并提供平滑作用；第一版没有跨 stage 的 action-jerk 项 $(u_i-u_{i-1})^2$。

## 5. 约束条件

初值和动力学等式为

$$
\mathbf{x}_{0|k}=\hat{\mathbf{x}}_k,\qquad
\mathbf{x}_{i+1|k}=\mathbf{F}_h(\mathbf{x}_{i|k},\mathbf{u}_{i|k},\mathbf{p}_{i|k}).
$$

velocity interface 的动作是速度命令增量，所以加速度上限映射为硬约束

$$
-a_{\max}h\leq u_{i|k}\leq a_{\max}h,\qquad i=0,\ldots,N-1.
$$

当配置 'max_velocity > 0' 时，在预测状态 (i=1,ldots,N) 加入硬约束

$$
-v_{\max}\leq v_{rz,i|k}\leq v_{\max},\qquad
-v_{\max}\leq c_{z,i|k}\leq v_{\max}.
$$

当 'max_velocity == 0' 时，OCP 完全不创建这两组界，而不是创建数值很大的伪界。

软非线性约束向量定义为

$$
\mathbf{h}(\mathbf{x}_{i|k},\mathbf{p}_{i|k})=
\begin{bmatrix}
p_{b,i|k}\\ \theta_{i|k}\\ p_{b,i|k}-p_{g,i|k}
\end{bmatrix}.
$$

默认边界为

$$
\underline{\mathbf{h}}=
\begin{bmatrix}0&-40^\circ&-0.5\end{bmatrix}^{\top},\qquad
\overline{\mathbf{h}}=
\begin{bmatrix}0.70&40^\circ&0.5\end{bmatrix}^{\top}.
$$

在所有 $i=0,\ldots,N$ 上施加

$$
\underline{\mathbf{h}}-\mathbf{s}_{i}^{l}
\leq \mathbf{h}(\mathbf{x}_{i|k},\mathbf{p}_{i|k})
\leq \overline{\mathbf{h}}+\mathbf{s}_{i}^{u},\qquad
\mathbf{s}_{i}^{l},\mathbf{s}_{i}^{u}\geq0.
$$

球位置表达物理梁区间，跟踪误差表达相对当前参考的允许偏差，因此二者同时保留。

## 6. 最终 OCP-NLP

堆叠决策变量为

$$
\boldsymbol{\chi}_k=
\left\{
\mathbf{x}_{0|k},\ldots,\mathbf{x}_{N|k},
\mathbf{u}_{0|k},\ldots,\mathbf{u}_{N-1|k},
\mathbf{s}_{0:N}^{l},\mathbf{s}_{0:N}^{u}
\right\}.
$$

给定参数序列 $\mathbf{P}_k=\{\mathbf{p}_{0|k},\ldots,\mathbf{p}_{N|k}\}$，控制器求解

$$
\begin{aligned}
\boldsymbol{\chi}_k^*=\arg\min_{\boldsymbol{\chi}_k}\quad &J_k\\
\text{s.t.}\quad
&\mathbf{x}_{0|k}=\hat{\mathbf{x}}_k,\\
&\mathbf{x}_{i+1|k}=\mathbf{F}_h(\mathbf{x}_{i|k},\mathbf{u}_{i|k},\mathbf{p}_{i|k}),&&i=0{:}N-1,\\
&-a_{\max}h\leq u_{i|k}\leq a_{\max}h,&&i=0{:}N-1,\\
&-v_{\max}\leq[v_{rz,i|k},c_{z,i|k}]^\top\leq v_{\max},&&i=1{:}N,\quad v_{\max}>0,\\
&\underline{\mathbf{h}}-\mathbf{s}_{i}^{l}
\leq\mathbf{h}(\mathbf{x}_{i|k},\mathbf{p}_{i|k})
\leq\overline{\mathbf{h}}+\mathbf{s}_{i}^{u},&&i=0{:}N,\\
&\mathbf{s}_{i}^{l},\mathbf{s}_{i}^{u}\geq0,&&i=0{:}N.
\end{aligned}
$$

采用 receding horizon，只向环境发送

$$
a_k=u_{0|k}^{*}=\Delta c_{z,k}.
$$

solver 状态非零、NaN/Inf、输出越界或调用异常时发送 $a_k=0$，保持当前命令不变并记录 fallback。

## 7. CasADi/acados 映射

| 论文对象 | acados 字段 | 实际顺序/设置 |
|---|---|---|
| $\mathbf{x}$ | 'model.x' | '[pb, vb, theta, vrz, command_z]' |
| $\mathbf{u}$ | 'model.u' | '[delta_command_z]' |
| $\mathbf{p}$ | 'model.p' | '[pg, vg, tau_s, gain, bias]' |
| $\mathbf{F}_h$ | 'model.disc_dyn_expr' | exact response + RK4，'DISCRETE' |
| $\mathbf{y}$ | 'model.cost_y_expr' | '[pb-pg, vb-vg, theta, omega, vrz, command_z+u, u]' |
| $\mathbf{y}^e$ | 'model.cost_y_expr_e' | '[pb-pg, vb-vg, theta, omega, vrz, command_z]' |
| 参考 | 'cost.yref', 'cost.yref_e' | 全零；'pg,vg' 已进入 residual |
| $\mathbf{h}$ | 'model.con_h_expr(_e)' | '[pb, theta, pb-pg]' |
| slack 索引 | 'constraints.idxsh(_e)' | '[0, 1, 2]' |
| slack 代价 | 'cost.zl/zu/Zl/Zu(_e)' | 每项默认 '1e4' |
| 输入界 | 'constraints.idxbu/lbu/ubu' | 'idxbu=[0]'，'±max_acc*step_dt' |
| 可选速度界 | 'constraints.idxbx(_e)' | '[3,4]'；仅 'max_velocity>0' 创建 |

solver 选项显式固定为：

    nlp_solver_type: SQP_RTI
    qp_solver: PARTIAL_CONDENSING_HPIPM
    hessian_approx: GAUSS_NEWTON
    integrator_type: DISCRETE
    n_horizon: 30
    qp_solver_cond_N: 10
    qp_solver_warm_start: 1
    levenberg_marquardt: 0.0
    print_level: 0

此外 'cost_scaling=ones(N+1)'，因此上述目标是离散求和，YAML 中的权重不会再被 (h) 隐式缩放。

单环境使用 'AcadosOcpSolver'。多环境使用原生 'AcadosOcpBatchSolver'：初值界和每一 stage 参数以 batch 数组设置，然后只调用一次 batch 'solve()'，没有 per-environment 进程或 pipe。

## 8. Predictor 与 reference preview

新 NMPC 核心本身不包含 delay queue。动作延迟补偿完全位于已有的 'VelocityModelStatePredictor' 中，禁止在两层重复推进同一延时。

令固定动作延迟为 $D$：

- predictor 未激活时，$\hat{\mathbf{x}}_k$ 使用当前观测的 '[pb, vb, theta, vrz]'，'command_z' 使用 predictor 已有命令 buffer；OCP 消费 raw preview 的 '0...N'。
- predictor 激活时，先把 'pg_0...pg_D' 传给 predictor，得到执行 pending command 后的等效无延时状态；OCP 消费原始 preview 的 'D...D+N'。
- 必须满足 raw preview $H\geq D+N$。默认无延时为 $D=0,H=N=30$；D=8 配置为 $D=8,H=38,N=30$。
- policy 依据 'raw_observation_fields' 查找 'pg_i/vg_i'，不依赖扩展 observation 的硬编码列号。原有 11-D prefix 不变。
- predictor 激活时，其 $(\tau,K,b)$、'step_dt'、'max_acc'、'max_velocity' 必须与 NMPC/环境一致。'velocity_response_max_abs_velocity' 必须为 0，速度限制统一由 OCP 硬约束表达。

每次动作（包括 fallback 零动作）之后都调用 'state_predictor.update_after_action()'，使 policy 命令 buffer、predictor queue 和环境收到的增量一致。partial reset 同时清空对应环境的命令、queue 和 solver iterate，不影响其他环境。

## 9. 配置到论文符号

| YAML/dataclass 字段 | 论文符号或功能 |
|---|---|
| 'model.plank_length' | $L$ |
| 'model.rope_length' | $l_r$ |
| 'model.ball_position_offset' | $d_b$ |
| 'model.gravity' | $g$ |
| 'model.ball_mass', 'ball_radius', 'ball_inertia_ratio' | $m_b,r_b,\rho_m$ |
| 'model.epsilon' | 几何保护 $\epsilon$ |
| 'response.tau_s', 'gain', 'bias' | $\tau,K,b$ |
| 'objective.stage_weights' | $\operatorname{diag}(\mathbf W)$ |
| 'objective.terminal_weights' | $\operatorname{diag}(\mathbf W_e)$ |
| 'constraints.*_min/max' | $\underline{\mathbf h},\overline{\mathbf h}$ |
| 'constraints.slack_l1/l2' | $\mathbf z,\mathbf Z$ 对角值 |
| 'constraints.max_acc' | $a_{\max}$ |
| 'constraints.max_velocity' | $v_{\max}$；0 表示不创建界 |
| 'solver.n_horizon', 'step_dt' | $N,h$ |
| 'state_predictor.delay_step' | 外部补偿延迟 $D$ |

'auto' 解析规则是确定性的：物理量、步长和 actuator limit 从环境解析；固定 response range 可解析为唯一值；随机 response range 必须在 policy YAML 给出 nominal 值。随项目提供的配置明确使用 $\tau=0.155$, $K=0.86$, $b=-0.0035$。

## 10. Solver 生命周期、缓存与日志

成功求解后，状态和输入轨迹向前移一格作为下一周期 warm start。失败 solver 的 iterate 被单独重置。非零 status、NaN/Inf、超过输入界、返回维数异常或调用异常均触发零增量 fallback。

生成代码位于：

    build/acados_unified_nmpc/<configuration-fingerprint>/

fingerprint 覆盖完整 model/objective/constraints/solver 配置、OCP 维数以及 acados 源码 release tag 和 commit；已有 JSON/共享库由 v0.5.4 的 code-reuse 检查复用。release 兼容性以源码仓库的 exact Git tag 为准，'acados_template' metadata 只写入 resolved metadata 用于诊断，不参与兼容性判断或 build cache key。'build/' 已加入 '.gitignore'，不得提交生成的 C、JSON 或共享库。

'get_state()' 提供逐环境 status、solver time、QP time、SQP/QP iteration、cost、最大 slack、fallback、controller wall time、完整 policy wall time和 predictor 状态。runner 的 'timing_summary.yaml/summary.csv' 还包含 controller time 的 mean、p50、p95、p99、max、60 Hz deadline-miss rate、solver-success rate 和 fallback rate。多环境 timing 是 batch wall time，不作为单环境 60 Hz 验收。

## 11. 安装、运行和验证

项目固定支持 [acados v0.5.4](https://github.com/acados/acados/releases/tag/v0.5.4)，不 vendoring acados。'conda_env.yml' 中原有 CasADi/do-mpc 依赖保留；acados 单独安装：

    git clone --branch v0.5.4 --recursive https://github.com/acados/acados.git /path/to/acados
    cmake -S /path/to/acados -B /path/to/acados/build \
      -DACADOS_WITH_OPENMP=ON -DACADOS_NUM_THREADS=1 -DCMAKE_BUILD_TYPE=Release
    cmake --build /path/to/acados/build --target install --parallel
    pip install -e /path/to/acados/interfaces/acados_template

    export ACADOS_SOURCE_DIR=/path/to/acados
    export LD_LIBRARY_PATH="$ACADOS_SOURCE_DIR/lib:$LD_LIBRARY_PATH"

其中 '--target install' 不可省略：普通 build 只会把 'libacados.so' 留在 'build/acados/'，而 generated solver 需要 '$ACADOS_SOURCE_DIR/include/acados_c/' 和 '$ACADOS_SOURCE_DIR/lib/' 中的已安装 headers/libraries。可在启动前检查：

    test -f "$ACADOS_SOURCE_DIR/include/acados_c/ocp_nlp_interface.h"
    test -f "$ACADOS_SOURCE_DIR/lib/libacados.so"

如果 import 失败，旧 baseline 和 'aerial_balance_bench.baselines' 仍可导入；只有实例化 'AcadosNMPCPolicy' 才给出安装提示。

acados v0.5.4 源码中的 'acados_template' distribution metadata 在部分安装方式下仍可能显示 '0.5.1'，因此不能用该值判定 acados release。policy 会执行：

    git -C "$ACADOS_SOURCE_DIR" describe --tags --exact-match HEAD

并要求结果为 'v0.5.4'，同时验证实际导入的 'acados_template.__file__' 位于 '$ACADOS_SOURCE_DIR/interfaces/acados_template'。resolved config 分别记录真实源码版本/commit 和仅供诊断的 template metadata。若路径不一致，应在运行 runner 的同一 Python 环境重新执行：

    python -m pip install -e "$ACADOS_SOURCE_DIR/interfaces/acados_template"

无延时 mixed：

    python3 scripts/acados_nmpc_unified_eval.py \
      --config baselines/configs/acados_nmpc_unified_tracking_eval.yaml \
      --num_envs 1 --episodes 10 --headless

D=8 predictor：

    python3 scripts/acados_nmpc_unified_eval.py \
      --config baselines/configs/acados_nmpc_unified_tracking_predictor_d8_eval.yaml \
      --num_envs 1 --episodes 10 --headless

更换 '--env_config environments/configs/unified_tracking_<family>.yaml' 可测试 constant、sine、triangle、trapezoid、random B-spline 和 random ramp-dwell；D=8 必须使用对应的 '_delay_d8_h38.yaml' 环境。

验证：

    python3 -m py_compile \
      baselines/acados_nmpc_core.py \
      baselines/acados_nmpc_policy.py \
      scripts/acados_nmpc_unified_eval.py

    pytest -q tests/test_acados_nmpc_core.py tests/test_acados_nmpc_policy.py
    pytest -q

无 acados 环境会运行 NumPy/model/policy contract 测试并跳过真正 OCP 生成测试；完整验收必须在 v0.5.4 环境执行。

## 12. 性能验收和已知限制

目标 CPU、单环境、warm solver、$N=30$ 的门槛：

- solver success rate $\geq99.9\%$；
- fallback rate $\leq0.1\%$；
- 完整 'act()' p99 $\leq16.67\,{\rm ms}$；
- deadline miss 必须独立报告，不能用均值替代。

仓库不提交机器相关的伪造 timing 结果。有效报告至少包含 resolved config、CPU、acados/build fingerprint、warm-up 次数、样本数、上述分位数和 success/fallback。'timing_summary.yaml' 结构为：

    controller_time_mean: <measured seconds>
    controller_time_p50: <measured seconds>
    controller_time_p95: <measured seconds>
    controller_time_p99: <measured seconds>
    controller_time_max: <measured seconds>
    deadline_miss_rate: <measured ratio>
    solver_success_rate: <measured ratio>
    fallback_rate: <measured ratio>

当前限制和 future work：

- NMPC 核心不建模 delay queue；仅支持外部 predictor 给出的等效无延时状态；
- predictor 保持既有固定/统一 delay 限制，不支持逐环境随机延时；
- 不支持在线 RLS 或 oracle response 参数；
- horizon 为均匀步长；
- 未实现 action-jerk penalty；
- 未实现独立硬实时 C 进程部署；Python runner 的 wall time 包含 tensor/NumPy setter 和 policy 数据流开销。

## 13. Objective 权重的顺序自适应调优

`scripts/tune_acados_nmpc_objective.py` 提供可恢复、每次最多启动一个 rollout 的调优流程。正式 session 固定使用当前 `environments/configs/unified_tracking_mixed.yaml`、均匀 $N=30$、600 control steps，并将环境文件的 SHA256 写入 `session_state.yaml`。session 建立后若环境文件发生变化，脚本会拒绝继续，从而避免把不同实验条件下的 trial 混入同一 leaderboard。候选 YAML 只能包含完整的 `stage_weights` 和 `terminal_weights`；model、response、constraints、horizon 和其他 solver 字段都不能由候选覆盖。

screen 使用 `seed=666`、48 个环境和每类至少 8 个首 episode；validation 使用独立的 `seed=667`、96 个环境和每类至少 20 个首 episode。调优 run 显式设置 `runner.stop_on_target_episodes=false`，所以即使部分环境提前终止/autoreset，也始终运行 600 步。分析器只截取每个环境第一次 `terminated|truncated` 之前的数据，后续 autoreset episode 不参与评分。普通 evaluator 配置的该选项默认为 `true`，原有提前停止行为不变。

类型均衡 screen 目标为

$$
J=0.45\overline e_{\rm RMSE}
 +0.30e_{\rm worst,p90}
 +0.15\overline e_{\rm tail}
 +0.10e_{\rm constant,steady}.
$$

分析产物还包含逐类型 RMSE/MAE/tail/max/bias、constant 最后 1 s 指标、overshoot/过零次数、action RMS/变化率/饱和率、command 与 $v_b,\theta,v_{rz}$ 的 RMS/峰值，以及 solver/fallback/slack/termination/boundary 诊断。状态机先建立 10 s 正式 baseline，首个候选固定把 $v_b-v_g$ 的 stage/terminal 权重从 $(0.5,2.5)$ 增大到 $(4,20)$，此后根据上一轮诊断一次只推荐一个权重组。完整配置和门槛见 `baselines/configs/acados_nmpc_objective_tuning.yaml`。

运行第一轮及后续单轮：

    conda run -n isaac-sim python scripts/tune_acados_nmpc_objective.py \
      --stage next

每次检查 `logs/acados_nmpc_unified_tracking/tuning/objective_v1/next_recommendation.yaml`、`history.csv` 和对应 run 的 `tuning_metrics.yaml`，然后再次执行同一命令。可用完整权重文件替换当前自动推荐：

    conda run -n isaac-sim python scripts/tune_acados_nmpc_objective.py \
      --stage next --candidate /path/to/objective_weights.yaml

已有 run 可只分析不启动 Isaac：

    python3 scripts/tune_acados_nmpc_objective.py \
      --analyze-run logs/acados_nmpc_unified_tracking/<run-directory>

验证分两次调用，以满足“单次最多一个 rollout”：第一次运行 96 环境统计验证，第二次运行一个环境的独立 warm timing（丢弃前 30 步）并完成 promotion 判定。

    conda run -n isaac-sim python scripts/tune_acados_nmpc_objective.py --stage validate
    conda run -n isaac-sim python scripts/tune_acados_nmpc_objective.py --stage validate

只有全部 validation gates 通过时，脚本才原位替换 `baselines/configs/acados_nmpc_unified.yaml` 中两行默认 objective weights；其余字段保持原文本不变。失败时只写 `final_selection.yaml`、`recommended_policy.yaml` 和 `final_report.md`，默认 policy 不变。`--resume` 默认复用已完成的确定性 run，`--no-resume` 显式重跑，`--dry-run` 只生成并打印下一次 run 配置。

## 14. 严格跟踪性能的二次权重调优

在固定 mixed 环境、$N=30$、模型、response、约束和求解器选项的条件下，`performance_v2` 只继续调整 NONLINEAR_LS 权重。最终采用的 stage 和 terminal 权重分别为

$$
W=\operatorname{diag}(20,4,0.1,0.2,0.2,0.05,0.25),
\qquad
W_N=\operatorname{diag}(100,20,0.5,1,1,0.05).
$$

其分量顺序仍分别为 $[p_b-p_g,v_b-v_g,\theta,\omega,v_{rz},c_z+u,u]$ 和 $[p_b-p_g,v_b-v_g,\theta,\omega,v_{rz},c_z]$。与第一轮最优权重相比，本轮将位置权重增大四倍、command 权重降至原来的 $1/20$、action-increment 权重降至原来的 $1/4$；其余权重不变。这样提高了位置闭环带宽，同时保留速度和梁运动阻尼。

独立 seed 669、96 个首 episode 的验证覆盖 34 个 constant、22 个 random B-spline 和 40 个 random ramp-dwell episode。结果为：constant 成功率 $100\%$、最后 1 s MAE $0.065\,\mathrm{mm}$，random B-spline 平均 MAE $9.42\,\mathrm{mm}$，random ramp-dwell 平均 MAE $6.35\,\mathrm{mm}$。该验证中无 termination、boundary violation、fallback 或 active slack，solver success rate 为 $100\%$。单环境独立计时丢弃前 30 步后使用 270 个样本，完整 `act()` p99 为 $3.66\,\mathrm{ms}$，最大值 $4.23\,\mathrm{ms}$，60 Hz deadline-miss rate 为 0。

完整的逐轮候选、独立验证和计时结果保存在 `logs/acados_nmpc_unified_tracking/tuning/performance_v2/`。这些数值是当前 mixed 配置和 nominal response 参数下的经验结果，不应解释为对未见轨迹、模型失配或真实系统的无条件性能保证。
