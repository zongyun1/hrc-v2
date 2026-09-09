# Human-Present Mobile Manipulation in Genesis — 项目计划

> 状态：v0 草案 · 待 Phase 0 结果修订

---

## 0. 定位

**一句话**：首个将主流操作仿真中的物理接触抓取、与真实全身人体运动作为动态 agent 统一起来的 benchmark，在 avoid / place-relative-to-human / follow 三类任务上评测 VLA 的人机安全性。

**明确不做的事**：
- 不做 human motion retarget 到机器人（那是 humanoid imitation，另一个问题）
- 不主张"首个研究 human-present mobile manipulation"（Habitat 3.0 / PARTNR 占据了这个宽泛口号）
- 不在 v0 追求人对铰接物体的细粒度接触操作（生成模型做不到可用质量）

**四路交集**（novelty 的精确表述）：
真实全身人体运动（AMASS/SMPL-X 级别） × 物理接触抓取（非 magic grasp） × 移动操作 × VLA 人机安全评测

---

## 1. 技术路线总览

四层结构，**层与层之间只通过两个数据接口耦合**：

```
场景层   RoboCasa365 MJCF ──┬──> Genesis scene (物理/渲染/任务)
                            └──> occupancy voxel grid ──┐
                                                        │  接口 ①
人体层   TRUMANS/LINGO diffusion <──────────────────────┘
              + OmniControl 式 guidance
              + 任务脚本层（目标序列 → waypoint）
              └──> SMPL-X 序列 ─────────────┐
                                            │  接口 ②
机器人层  IK / motion planner ──> action ───┤
                                            │
耦合层   共仿真主循环（receding horizon）<──┘
```

**接口 ①** `occupancy grid`：静态部分预烘焙，动态部分（机器人预测 swept volume、开合的门、移动物体）以 overlay 形式逐帧叠加。这是人体模型与场景之间**唯一**的通道。

**接口 ②** `SMPL-X 序列`：`{transl, global_orient, body_pose, hand_pose}` + 时间戳 + 元数据。纯 npy/npz，不含任何仿真器概念。

**核心解耦原则**：人体轨迹生成器（`hpmm_motion`）**永远不 import genesis**。这保证它可以独立开发、独立验证、独立出图，即使仿真器那边卡住。

---

## 2. 代码库结构

```
hpmm/
├── hpmm_motion/              # A层：simulator-agnostic 人体轨迹生成
│   ├── backends/
│   │   ├── base.py           # 统一接口：sample(occ, waypoints, labels) -> SMPLX
│   │   ├── trumans.py        # TRUMANS wrapper
│   │   ├── lingo.py          # 语义更宽的备选
│   │   └── omnicontrol.py    # 空间约束 backend
│   ├── occupancy/
│   │   ├── voxelize.py       # mesh/MJCF -> 静态 occupancy（预烘焙）
│   │   ├── overlay.py        # 动态障碍叠加（机器人 swept volume 等）
│   │   └── crop.py           # pelvis-local 裁剪，batched
│   ├── guidance/
│   │   ├── spatial.py        # 关节位置约束（OmniControl 式）
│   │   └── clearance.py      # 到机器人表面的 SDF 惩罚梯度
│   ├── script/
│   │   ├── goals.py          # 采样人的目标序列（冰箱→台面→水槽→坐下）
│   │   └── planner.py        # 目标 -> 粗 waypoint（A*，含机器人占据）
│   └── io/
│       └── schema.py         # ★ 冻结的 SMPL-X 序列格式定义
│
├── hpmm_sim/                 # B层：Genesis 环境
│   ├── assets/
│   │   ├── export_robocasa.py    # robosuite runtime -> 扁平 MJCF + mesh 包
│   │   └── convert.py            # 材质/碰撞组/凸分解修补
│   ├── shim/
│   │   ├── site.py           # robosuite site 查询 -> Genesis entity accessor
│   │   ├── predicate.py      # success predicate 重定向
│   │   └── sampler.py        # placement sampler 重定向
│   ├── human/
│   │   ├── skeleton.py       # SMPL-X -> 刚体骨架 URDF（24 link）
│   │   └── driver.py         # 逐帧 set_qpos，kinematic 单向碰撞
│   ├── robot/
│   │   ├── ik.py             # IK 轨迹跟踪
│   │   └── gen.py            # object-centric 子轨迹变换（MimicGen 式）
│   └── envs/
│       └── cosim.py          # ★ 共仿真主循环
│
├── hpmm_data/
│   ├── generate.py           # 批量 episode 生成（n_envs 并行）
│   ├── filter.py             # 拒绝采样：穿透/失败/不可达
│   └── format.py             # 输出格式（robot obs/action + 人体 + 安全标注）
│
├── hpmm_eval/
│   ├── metrics.py            # 最小距离、接触事件、让行成功率、任务成功率
│   └── baselines/            # VLA 评测入口
│
├── tools/                    # 一次性脚本：预烘焙、可视化、sim2sim 对比
├── configs/
└── tests/
```

### 从第一天就要守住的三条不变量

1. **Batch-first**：所有张量按 `[n_envs, ...]` 组织，包括 occupancy 裁剪和更新。事后加 batch 维的代价极高。
2. **`hpmm_motion` 不依赖 `hpmm_sim`**：在 `tests/` 里加一条 import 检查强制执行。
3. **`io/schema.py` 尽早冻结**：两条腿并行开发时，这是唯一的契约。

---

## 3. 阶段划分

### Phase 0 — 可行性体检（~2 周）

**这是唯一一个结果会推翻后续计划的阶段，不要跳过。**

| # | 任务 | Go/No-Go 判据 |
|---|---|---|
| 0.1 | 检查 RoboVerse / MetaSim 是否已有 RoboCasa365 pack（genesis backend） | 有 → 省两周，直接用 |
| 0.2 | 导出 1 个 layout 的 MJCF，导入 Genesis，目视检查 | 铰接方向、初始位姿正确 |
| 0.3 | 挑 3–5 条 MuJoCo demo 在 Genesis 里 replay | **不是为了拿数据**，是为了验证导入正确性；能大致走完即可 |
| 0.4 | 场景体素化，TRUMANS 采样，人在厨房走 30 秒 | 不穿墙、不飘、不卡在 counter 下方 |
| 0.5 | 确认 TRUMANS `sample_hsi.py` 中 occupancy 是预烘焙还是逐帧查询 | 决定动态 overlay 的改造成本 |
| 0.6 | 确认 Genesis 中被外部驱动的 body 与 rigid solver 的碰撞方向性 | 需要"人挡得住机器人但不被推走" |

**交付物**：一份体检报告 + 两段视频（Genesis 里的厨房、厨房里走路的人）。

**风险出口**：若 0.4 失败（TRUMANS 在厨房几何上 zero-shot 不成立），改用 LINGO 或退到 waypoint-following 的纯 locomotion 模型；若 0.2 严重失败，考虑只取 RoboCasa 的资产库、自己搭场景。

---

### Phase 1 — 两条腿独立跑通（~4 周）

并行推进，互不阻塞。

**A 线（人体）**
- 冻结 `io/schema.py`
- occupancy 静态预烘焙 + 动态 overlay + batched local crop
- 任务脚本层：目标序列采样 → A* waypoint
- 接上 guidance：先只做静态障碍的 clearance 惩罚
- 产出：给定任意 RoboCasa layout，批量生成 N 条不碰撞的人体轨迹（离线，无机器人）

**B 线（仿真）**
- MJCF 导出/转换固化成脚本
- `shim/` 层：site、predicate、sampler 重定向 —— **这是本阶段工作量最大的一块，是关键路径**
- SMPL-X → 刚体骨架，逐帧驱动，能在 Genesis 里播放 A 线的输出
- 机器人 IK 轨迹在 Genesis 里原生生成，跑通 3–5 个 atomic task

**交付物**：一个 Genesis 场景里，机器人在做 pick-place，同时有个人在走动（两者互不感知，会穿模——这是预期的）。

---

### Phase 2 — 闭环共仿真（~4 周）

把两条腿接起来。

- 共仿真主循环：receding horizon，1–2 秒 chunk，异步生成下一段
- 机器人**未来 N 帧预测 swept volume** 写入 occupancy（不是当前体积）
- clearance guidance 接入机器人 SDF
- 三层避让叠加：waypoint 硬约束 + 采样时 guidance + 拒绝采样兜底
- **反应型 / 分神型开关**：occupancy 里是否注入机器人，做成配置项

**必做 ablation**：当前体积 vs. 预测 swept volume，测人的轨迹是否出现全局振荡。

**交付物**：人会给机器人让路的 episode 视频 + 让行成功率数字。

---

### Phase 3 — 数据集生成（~6 周）

- 三类任务定型：
  - **avoid**：人穿过机器人工作区，机器人需要让行/暂停
  - **place-relative-to-human**：放置位置取决于人的位置（递、避开、放在人够得到的地方）
  - **follow**：机器人跟随人移动
- 大规模并行生成（`n_envs` 拉满），三层过滤
- 数据格式定稿：robot obs/action + 人体 SMPL-X + 逐帧最小距离 + 安全违规标注
- 场景/layout/人体轨迹的 split 设计（seen / unseen）

**注意**：人是运动学驱动的，机器人撞上去会产生穿透或巨大接触冲量。**不要修**——直接把穿透深度/接触力作为安全违规信号，触发即终止 episode 并记录。这正是需要的 metric。

---

### Phase 4 — Benchmark + Baselines（~6 周）

- 安全 metric 定义：最小距离分布、接触事件率、让行成功率、任务成功率、完成时间惩罚
- VLA baseline 评测（π₀.₅ / GR00T / 自训 BC）
- 核心实验：**在"分神型人"数据上，现有 policy 的安全性有多差**
- 消融：反应型 vs 分神型人、有人 vs 无人训练

---

### Phase 5 — 论文（~4 周）

---

## 4. 关键设计决定

| 项目 | 决定 | 状态 |
|---|---|---|
| 仿真器 | Genesis | 已定 |
| 场景来源 | RoboCasa365（MJCF 导入） | 已定 |
| 机器人轨迹 | Genesis 里 IK 重新生成，不 replay MuJoCo demo | 已定 |
| 人体运动 | 在 Genesis 环境里闭环生成，不离线导入 | 已定 |
| 人体表示 | 运动学驱动刚体骨架（非物理跟踪） | 已定，v0 |
| 人-机器人约束 | 生成时禁止接触 | 已定，**后续会放松** |
| 人体动作源 | TRUMANS（备选 LINGO） | Phase 0 后确认 |
| 语义表达方式 | 空间约束 + 停留时间，**不用文本** | 已定，v0 |
| 铰接物体交互 | 脚本 + IK 硬摆，不用生成模型 | 已定 |

---

## 5. 风险登记

| 风险 | 影响 | 缓解 | 何时能知道 |
|---|---|---|---|
| TRUMANS 在厨房几何 zero-shot 失败 | 高 | 换 LINGO / 退到纯 locomotion | Phase 0.4 |
| shim 层工作量爆炸 | 高 | 先只支持 5 个 atomic task | Phase 1 |
| 动态 occupancy 导致轨迹振荡 | 中 | 预测 swept volume；提高 chunk 长度 | Phase 2 |
| TRUMANS 动作词表（10 类）撑不起厨房语义 | 中 | 空间约束表达语义；acknowledge 为 limitation | 已知 |
| 被抢发 | 中 | 每月扫 arXiv；保持 positioning 精确 | 持续 |
| Genesis kinematic body 碰撞方向性不符预期 | 中 | 极大质量 + 每帧强制覆写 state | Phase 0.6 |

---

## 6. 相关工作监控清单

**必须持续盯**（最接近的）：
- Habitat 3.0 / PARTNR — 占据宽泛口号，但用程序化 avatar + 抽象抓取
- HRIBench (arXiv 2607.13056) — 桌面级 Franka，近距离协作，非移动操作
- Virtual Community (arXiv 2508.14893) — **Genesis 上的人-机器人共处开放世界，avatar 实现可复用**

**次级**：
NavIsaacLab、HumanTHOR、HandoverSim、MobileH2R、M3Bench、Mimicking-Bench

**方法侧**：
OmniControl、GMD、TeSMo、SceneAdapt、AffordMotion、MaskedMimic、PACER/TRACE、UniHSI、DIMOS

**特别关注方向**：scene-aware motion generation × dynamic agent。这是目前的空白，也正因为是空白，很可能已有人在填 —— 每月扫一次。

---

## 7. License 备忘

- AMASS 及全部衍生（HumanML3D / BABEL / Motion-X++）：**严格非商用学术**
- CMU raw mocap、100STYLE (CC BY 4.0)：少数可商用替代
- Mixamo：允许商用但**禁止 ML 训练**
- 数据集发布前需明确 TRUMANS 的授权条款
