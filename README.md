# ME5418 个人最终版：机械臂扰动下的四足机器人强化学习行走控制

姓名：Yan Jianan  
项目：Go2-X5 arm-disturbance-aware locomotion  
算法：PPO Actor-Critic  
仿真平台：NVIDIA Isaac Gym Preview 4  

---

## 1. 项目目的

本项目研究的是一个带机械臂的四足机器人在机械臂运动扰动下的稳定行走控制问题。机器人由 Unitree Go2 四足机器人和 6 自由度 X5 机械臂组成。机械臂在运动过程中会改变机器人整体质心、惯性和接触受力状态，从而影响机器狗的行走稳定性。

本项目的目标是：

> 训练一个腿部控制策略，使机器狗在机械臂运动产生扰动的情况下，仍然能够稳定地跟踪底盘速度命令并保持机身平衡。

需要强调的是：

- 强化学习策略控制的是 **Go2 的 12 个腿部关节**；
- 机械臂不是由强化学习策略控制的；
- 机械臂运动被视为外部扰动；
- policy 需要根据机器人自身状态、目标速度命令和机械臂状态，输出腿部动作来补偿机械臂扰动。

---

## 2. 项目结构

```text
个人最终版/
├── stage1_gym/              # Isaac Gym 环境、任务逻辑、物理仿真封装
│   ├── config.py            # 任务参数、控制参数、command/arm motion 配置
│   ├── task.py              # observation、reward、机械臂轨迹、步态逻辑
│   ├── env.py               # 单机器人 Gym 环境
│   ├── physics.py           # 单机器人 Isaac Gym 物理封装
│   ├── vec_env.py           # 多环境并行训练环境
│   └── vec_physics.py       # 多机器人 Isaac Gym 并行物理封装
├── stage1_rl/               # PPO Actor-Critic 训练代码
│   ├── actor_critic.py      # Actor/Critic 网络
│   ├── ppo.py               # PPO 损失函数与更新逻辑
│   ├── rollout_storage.py   # rollout buffer 与 GAE
│   ├── runner.py            # 单环境训练 runner
│   └── evaluation.py        # policy 评估指标
├── train_vec.py             # 8192 并行环境训练入口
├── record_video.py          # 离屏录制 policy rollout 视频
├── evaluate.py              # policy 评估
├── results/                 # 本人训练结果、曲线、视频和报告
│   ├── policy.pt            # arm-aware 个人训练模型
│   ├── training.csv         # arm-aware 训练日志
│   ├── no_arm_baseline/     # 无机械臂扰动训练 baseline
│   ├── plots/               # 奖励曲线、损失曲线、跟踪误差图
│   ├── videos/              # 对比视频
│   └── reports/             # 中英文 PDF 报告
└── tests/                   # 单元测试
```

---

## 3. 仿真环境

本项目使用 NVIDIA Isaac Gym Preview 4 进行物理仿真。Isaac Gym 支持 GPU 并行仿真，适合强化学习中大规模采样。

主要仿真参数：

```text
physics_dt = 0.005 s
decimation = 4
policy/control dt = 0.02 s
control frequency = 50 Hz
physics frequency = 200 Hz
```

训练使用的硬件和环境：

```text
GPU: NVIDIA A100-SXM4-80GB
Python: 3.8.20
PyTorch: 2.3.x CUDA build
Isaac Gym: Preview 4
Gym: 0.26.2
NumPy: 1.23.5
```

---

## 4. 输入 Observation

策略输入是一个 81 维 observation，包含机器人状态、目标命令、机械臂状态和误差信息。

| 状态项 | 维度 | 说明 |
|---|---:|---|
| projected gravity | 3 | 身体坐标系下的重力方向，用于感知机身倾斜 |
| leg joint position | 12 | 12 个腿部关节位置 |
| leg joint velocity | 12 | 12 个腿部关节速度 |
| previous action | 12 | 上一时刻腿部动作 |
| velocity command | 3 | 目标底盘速度 `[vx, vy, yaw_rate]` |
| gait clock | 4 | 四条腿的步态相位 |
| base angular velocity | 3 | 机身角速度 |
| base linear velocity | 3 | 机身线速度 |
| height / pitch / roll | 3 | 当前机身高度、俯仰角、横滚角 |
| pose error | 3 | 目标姿态与当前姿态误差 |
| velocity error | 3 | 目标速度与实际速度误差 |
| arm position | 6 | 机械臂 6 个关节位置 |
| arm velocity | 6 | 机械臂 6 个关节速度 |
| arm command velocity | 6 | 机械臂目标运动速度 |
| posture command | 2 | 目标 pitch / roll |

总维度：

```text
3 + 12 + 12 + 12 + 3 + 4 + 3 + 3 + 3 + 3 + 3 + 6 + 6 + 6 + 2 = 81
```

其中机械臂相关输入：

```text
arm position
arm velocity
arm command velocity
```

是本项目的关键，因为它们使 policy 能够观察机械臂运动，从而学习如何补偿机械臂扰动。

---

## 5. 输出 Action

策略输出是 12 维连续动作：

```text
action ∈ R^12
```

对应 Go2 四条腿的 12 个关节。动作不是直接作为力矩输入，而是作为腿部关节目标位置偏移：

```text
leg_target = LEG_HOME + scaled_action
```

随后底层 PD 控制器根据目标位置产生关节力矩，并输入 Isaac Gym 物理仿真。

机械臂动作不是 RL policy 的输出。机械臂由外部轨迹生成器控制，用来产生扰动。

---

## 6. 目标速度命令与机械臂运动

### 6.1 目标速度命令

个人最终版参考小组最终代码，将目标速度设置为以前向行走为主：

```text
vx target:       0.12 ~ 0.35 m/s
vy target:      -0.10 ~ 0.10 m/s
yaw_rate target:-0.55 ~ 0.55 rad/s
```

目标速度不会突然跳变，而是在 5 到 9 秒内平滑过渡到新的目标命令。

这样设计的原因是：项目目标不是让狗随机向任意方向乱走，而是让它在主要前向行走过程中抵抗机械臂扰动。

### 6.2 机械臂运动

机械臂运动采用平滑随机 reaching 轨迹：

1. 随机采样机械臂目标位置；
2. 从当前机械臂位置平滑移动到目标位置；
3. 输出机械臂目标位置和目标速度；
4. 机械臂运动改变机器人质心和惯性，从而扰动底盘。

训练时，arm-aware policy 在机械臂运动扰动下学习。baseline policy 则在关闭机械臂扰动时训练。

---

## 7. 使用的算法

本项目使用 PPO（Proximal Policy Optimization）算法。

模型为 Actor-Critic 结构：

```text
Actor:  81 -> 512 -> 256 -> 128 -> 12
Critic: 81 -> 512 -> 256 -> 128 -> 1
Activation: ELU
```

Actor 输出 12 维腿部动作分布的均值；训练时从高斯分布中采样动作，推理时使用均值动作。Critic 输出状态价值 `V(s)`，用于计算 advantage。

PPO 损失函数：

```text
r_t(θ) = π_θ(a_t|s_t) / π_old(a_t|s_t)

L_policy = - E[ min(r_t A_t, clip(r_t, 1-ε, 1+ε) A_t) ]

L_value = E[ max((V_θ(s_t)-R_t)^2, (V_clip(s_t)-R_t)^2) ]

L_total = L_policy + c1 L_value - c2 E[H(π_θ)]
```

其中：

- `policy loss` 更新 Actor；
- `value loss` 更新 Critic；
- `entropy` 保持探索性；
- 最终通过 `total loss.backward()` 统一反向传播。

---

## 8. 奖励函数构建

奖励函数主要包括以下几类：

### 8.1 速度跟踪奖励

```text
tracking_lin_vel
tracking_ang_vel
```

鼓励实际底盘速度接近目标速度命令。

### 8.2 姿态与高度稳定

```text
orientation
base_height
lin_vel_z
ang_vel_xy
```

鼓励机器人保持身体水平、高度稳定，并减少上下跳动和姿态晃动。

### 8.3 步态与接触约束

```text
feet_slip
feet_clearance
tracking_contacts_force
tracking_contacts_velocity
raibert_heuristic
```

用于减少脚滑、鼓励合理摆腿高度和合理落脚点。

### 8.4 动作平滑与能耗

```text
action_rate
action_smoothness_1
action_smoothness_2
torques
loco_energy
dof_vel
dof_acc
```

用于避免高频抖动、大力矩和高能耗。

### 8.5 安全项

```text
collision
dof_pos_limits
```

用于避免碰撞和关节超限。

总体目标是让机器人：

```text
在机械臂运动扰动下，
稳定、平滑、低能耗地跟踪目标速度命令。
```

---

## 9. 训练设置

本人在 A100 GPU 上进行了并行训练，训练参数如下：

```text
num_envs = 8192
rollout_steps = 24
iterations = 100
PPO epochs = 5
minibatches = 8
learning_rate = 5e-4 with KL adaptation
gamma = 0.99
GAE lambda = 0.95
clip parameter = 0.2
entropy coefficient = 0.01
```

每个 iteration 收集：

```text
8192 × 24 = 196,608 transitions
```

100 个 iterations 共收集：

```text
19,660,800 environment steps
```

本人训练了两个 policy：

1. `arm-aware policy`
   - 机械臂运动；
   - policy 可观察机械臂状态；
   - 用于学习补偿机械臂扰动。

2. `no-arm baseline policy`
   - 训练时关闭机械臂扰动；
   - 用于对比普通行走策略遇到机械臂扰动时的表现。

---

## 10. 训练结果

### 10.1 曲线图

训练曲线位于：

```text
results/plots/reward_curve.png
results/plots/loss_curve.png
results/plots/tracking_error_comparison.png
```

包括：

- 奖励曲线；
- PPO 损失曲线；
- 有无机械臂扰动、有无针对性训练的跟踪误差对比。

### 10.2 定量结果

固定测试条件：

```text
command = [0.25, 0, 0]
seed = 9
steps = 500
```

结果如下：

| Case | reward_mean | vx_mae | vy_mae | yaw_mae | height_rmse | roll_rms | pitch_rms | terminations |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| no arm + trained | 0.01906 | 0.0364 | 0.0218 | 0.1022 | 0.0036 | 0.0196 | 0.0632 | 0 |
| arm + zero/no policy | 0.00302 | 0.2569 | 0.0455 | 0.0974 | 0.0242 | 0.1071 | 0.0472 | 0 |
| arm + no-arm baseline | 0.00830 | 0.2689 | 0.1309 | 0.2167 | 0.0550 | 0.5375 | 0.0523 | 3 |
| arm + arm-aware trained | 0.01822 | 0.0271 | 0.0172 | 0.0758 | 0.0063 | 0.0152 | 0.0333 | 0 |

结果说明：

- zero/no policy 无法主动跟踪目标前进速度；
- no-arm baseline 在机械臂运动扰动下表现明显下降；
- arm-aware policy 在机械臂运动下仍能保持较低的速度误差和姿态误差；
- arm-aware policy 没有发生 termination。

---

## 11. 对比视频

视频位于：

```text
results/videos/
```

包括：

```text
no_arm_trained_forward.mp4
arm_zero_forward.mp4
arm_locked_baseline_forward.mp4
arm_trained_forward.mp4
```

含义：

| 视频 | 机械臂状态 | 狗腿控制策略 |
|---|---|---|
| no_arm_trained_forward | 机械臂不动 | trained policy |
| arm_zero_forward | 机械臂运动 | zero action |
| arm_locked_baseline_forward | 机械臂运动 | no-arm baseline |
| arm_trained_forward | 机械臂运动 | arm-aware policy |

---

## 12. 最终提交说明

最终小组提交包采用的是 Yang Guangyu 训练和筛选后的 selected checkpoints：

```text
models/arm_aware_400.pt
models/locked_100.pt
```

本人个人训练主要用于：

1. 验证代码可以在 A100 环境中运行；
2. 复现并行 PPO 训练流程；
3. 生成训练曲线和对比图；
4. 分析 arm-aware policy 对机械臂扰动补偿的作用。

最终提交的正式模型和定量结果以 Yang Guangyu 提供的版本为准。

---

## 13. AI 使用声明

本人个人代码修改、训练脚本调整、实验结果分析等部分使用了 GPT-5.6-sol 辅助完成。报告文字组织、表达润色、中英文版本生成使用了 GPT-5.5 辅助完成。本人对最终提交内容、实验结果解释和报告质量负责。
