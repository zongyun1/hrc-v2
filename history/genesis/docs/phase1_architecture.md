# HRC-V2 第一阶段基础架构

## 目标

第一阶段只实现一个可重复运行的最小场景：

- 一个家庭环境（优先厨房或客厅的一小块区域）
- 一个 Franka 机械臂
- 一个最简单的 human avatar
- 一个可观察、可重置、可扩展的仿真环境接口

第一阶段不实现复杂人机接触、移动底盘、全身人体动力学或大规模 RL。代码结构需要为这些能力预留插槽，但实现保持简单。

## 后端

第一阶段使用 Isaac Lab/Isaac Sim 作为新项目仿真后端。旧的 Genesis 项目独立保留，不作为新项目的运行时依赖。

```text
Isaac Lab environment
├── household scene
├── Franka articulation
├── kinematic human avatar
├── cameras and basic sensors
└── task reset / observation / step
```

## 目录结构

```text
hrc-v2/
├── assets/                 # USD 和少量测试资源
├── configs/                # 场景、机器人和运行配置
├── docs/
├── hrc_v2/
│   ├── envs/               # Isaac Lab 环境
│   ├── robots/             # Franka 配置和控制
│   ├── human/              # avatar 和 HumanMotionController
│   ├── scenes/             # 家庭场景构建
│   ├── tasks/              # 第一阶段任务定义
│   └── common/             # 状态、坐标和日志 schema
├── scripts/
│   ├── play_phase1.py      # 单环境可视化运行
│   └── smoke_phase1.py     # 无 GUI 基础检查
└── README.md
```

## 核心接口

环境只暴露四个基本操作，后续 RL、遥操作和评测都建立在这层之上：

```python
class HouseholdEnv:
    def reset(self, seed: int | None = None) -> dict: ...
    def step(self, action) -> tuple[dict, float, bool, dict]: ...
    def observe(self) -> dict: ...
    def close(self) -> None: ...
```

第一阶段 action 先使用 Franka 关节位置目标和 gripper：

```text
arm_joint_target: 7 values
gripper_target:   1 value
```

未来增加 mobile manipulator 时扩展为：

```text
base_velocity:    [vx, vy, wz]
arm_joint_target: 7 values
gripper_target:   1 value
```

## Human avatar

第一阶段 avatar 采用 kinematic、非动力学人体，只提供可见 mesh 和简单碰撞 capsule。它通过统一的 `HumanMotionController` 更新：

```python
class HumanMotionController:
    def reset(self, seed=None): ...
    def set_goal(self, goal): ...
    def step(self, dt, *, robot_state=None, scene_state=None): ...
    def get_state(self): ...
```

最初只实现：

```text
idle → walk_to → idle
```

avatar 状态至少包含 root pose、future trajectory、collision capsules 和 action phase。后续可以替换为动作库、mocap、MotionGPT 或扩散模型，而不改环境接口。

## Household scene

第一阶段场景只保留必要元素：

- 地面和墙面
- 一张桌面或厨房台面
- 一个 Franka 工作区
- 2–3 个可交互物体（例如杯子、盒子、瓶子）
- 一个 avatar 初始位置和一条简单行走区域

所有物体优先使用 USD；原始 OBJ/GLB 只作为输入资产，不直接进入运行时。场景构建应集中在 `hrc_v2/scenes/`，任务代码不直接创建底层 prim。

## 第一阶段任务

先实现一个最小任务验证完整数据流：

```text
Franka reaches a marked object and returns to home pose.
```

avatar 同时执行 `walk_to`，但不参与接触和任务成功条件。验证内容包括：

1. 场景加载
2. Franka reset 和关节控制
3. avatar reset 和逐步更新
4. 相机观测
5. action / observation shape
6. episode termination 和日志

## 设计约束

- 不在任务中写死 USD prim path；通过 config 或 scene handle 获取。
- 不让 human controller 直接调用机器人控制器。
- 不让机器人 task 依赖 avatar 的具体模型，只依赖 `HumanMotionState`。
- 所有随机化都由 episode seed 控制。
- 先支持单环境，再考虑 Isaac Lab vectorized environments。
- 先使用 position/PD 控制，暂不引入复杂 whole-body controller。

## 实施顺序

1. 建立 Isaac Lab 项目启动脚本和最小环境。
2. 加载地面、台面、Franka 和相机。
3. 完成 Franka reset、home pose 和关节目标控制。
4. 加入最简单的 kinematic avatar 和 `idle/walk_to`。
5. 统一 observation、action、seed 和日志格式。
6. 完成一个无 GUI smoke test 和一个 GUI 播放脚本。
7. 再加入物体交互和 household task script。

## 完成标准

运行一条命令后，环境能够稳定完成至少 100 个 reset/step episode，并满足：

- Franka 不出现 NaN 或明显关节越界
- avatar 能从起点走到目标点
- 相机持续输出有效图像
- 同一个 seed 得到可复现的初始状态
- 日志记录 robot state、human state、seed 和 episode result

