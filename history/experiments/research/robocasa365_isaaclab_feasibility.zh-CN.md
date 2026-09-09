# RoboCasa365 → Isaac Lab 迁移可行性调研

后续实现与 GPU 实测：[三任务迁移原型](../robocasa_migration/README.md)。下文保留调研阶段结论，实际进展以实现文档为准。

调研日期：2026-09-08。范围：官方文档、公开源码和本仓库实现审阅；尚未下载 RoboCasa 资产、执行转换或运行迁移任务。下文的可行性是工程判断，不能当作实测覆盖率。

**结论：场景和物体值得迁移，部分任务可以先做；完整 365 任务需要建立场景语义与任务适配层。最短路径是先验证 Isaac Lab Arena 已有的 Lightwheel RoboCasa 厨房，再对照原版迁移一个任务。** 本次未找到官方提供的 RoboCasa365 全套 Isaac Lab 兼容实现。

## 1. 已确认的现成能力

RoboCasa365 官方说明包含 365 个任务、2500+ 厨房场景、3200+ 物体，环境后端依赖 robosuite。原版已有 `demo_kitchen_scenes` 和 `demo_tasks`，可用于建立场景与任务的参考视频。见 [RoboCasa 官方仓库](https://github.com/robocasa/robocasa)。

Isaac Lab Arena 官方 [厨房任务目录](https://isaac-sim.github.io/IsaacLab-Arena/main/pages/example_workflows/kitchen_bench_catalog.html) 列出 31 个 DROID 环境 YAML。按页面逐行计数：17 个 `lightwheel`，14 个 `replicator`。Lightwheel 示例覆盖抓放、开柜门、开冰箱/冷冻室、开微波炉/烤箱、按烤面包机按钮和旋转温度旋钮。页面也提供 Pi 策略执行展示。**这些是环境配置数量，不是 31 个原版 RoboCasa365 任务已迁移的证据。**

[Arena 厨房构建教程](https://isaac-sim.github.io/IsaacLab-Arena/main/pages/example_workflows/agentic_env_gen/kitchen_pick_and_place/step_2_edit_environment_graph_spec.html) 明确称背景为 Lightwheel RoboCasa kitchen，示例注册名为 `lightwheel_kitchen_one_wall_coastal`。场景内台面通过 USD prim 路径引用，机器人位置和物体摆放由关系配置描述。这证明该类厨房已有可复用入口，但不能据此推断原版 2500+ 场景全部存在对应 USD。

资产入口：[Arena 资产文档](https://isaac-sim.github.io/IsaacLab-Arena/main/pages/advanced/assets_management.html) 说明默认资产公开可用，来源包括 S3 和 LightWheel SDK registry。本轮未验证具体厨房的下载、依赖解析或访问情况。教程和目录存在 YAML 文件命名差异，实际启动前应以固定版本 checkout 中存在的文件为准。

## 2. 各部分能迁移多少

| 部分 | 可迁移程度判断 | 需要处理的内容 |
|---|---|---|
| 厨房视觉外观、静态家具 | 高，适合第一阶段 | 优先复用已发布 USD；原版 MJCF 需转换并检查尺寸、坐标、纹理和实例层级 |
| 杯碗、食品、工具等刚体 | 高，但逐类验证物理 | 质量、惯量、摩擦与碰撞形状；碗、篮子、水槽要保留容纳空间，不能用封住开口的单一凸包 |
| 门、抽屉、旋钮、按钮 | 中高 | 关节轴、限位、驱动、阻尼、碰撞过滤和状态阈值 |
| 布局与风格随机化 | 配置可复用，生成器要适配 | 固定采样实例容易；运行时任意布局重建与 GPU 批量 reset 是另一层工作 |
| 任务说明、物体类别、目标关系 | 高 | 保留原始 task ID、物体/fixture 身份、区域定义和 split 信息 |
| reset、success、接触判定 | 概念可复用，实现需移植 | 从 MuJoCo 数据访问改成 Isaac Lab 状态与传感器；保留原判定语义 |
| PandaOmron 与控制器 | 部分可复用 | Franka 手臂已有基础；移动底盘、动作空间、控制频率、坐标和夹爪控制须匹配 |
| demonstration / 已训练策略 | 可用于学习参考，不能假定直接回放 | 状态数组不通用；动作要对齐；接触动力学与图像域变化需要闭环验证 |
| 365 个任务端到端 | 尚无法给出数量或百分比 | 必须逐任务盘点依赖并跑测试；本轮没有形成 365 项映射表 |

上述判断的源码依据：基类 [kitchen.py](https://raw.githubusercontent.com/robocasa/robocasa/main/robocasa/environments/kitchen/kitchen.py) 继承 robosuite 的 `ManipulationEnv`，并使用其 MJCF、传感器、机器人和场景组件。因此仅转换模型文件不会带走完整任务环境。

## 3. 任务迁移真正需要保留什么

原版 [kitchen_pick_place.py](https://github.com/robocasa/robocasa/blob/main/robocasa/environments/kitchen/atomic/kitchen_pick_place.py) 中，`PickPlaceCounterToCabinet` 判定物体在柜内且夹爪已远离；`PickPlaceCounterToSink` 使用水槽内部区域的部分检查并检查夹爪距离。不能简单替换成“物体靠近目标点”。

[object_utils.py](https://raw.githubusercontent.com/robocasa/robocasa/main/robocasa/utils/object_utils.py) 的 `obj_inside_of` 读取 fixture 的内部区域，以及 MuJoCo 中的物体位置和姿态，按物体边界点或中心判断。因此转换输出还应包含：fixture 类型、内部/摆放区域、对象边界、原名称到 USD prim 的映射、关节语义、相机和初始状态。建议独立保存为 `scene_manifest.json`，不要依赖转换器自动保留所有 site 和语义。

厨房任务也不必一律上真实流体。例如原版 [sink.py](https://raw.githubusercontent.com/robocasa/robocasa/main/robocasa/models/fixtures/sink.py) 根据把手关节判定水流状态，调整 water site 的可见性，并用位置检查物体是否在水下。迁移时先复现这种抽象；不能从“洗东西”这个任务名称就推断需要流体求解器。其他电器和复合任务仍需逐个查源码。

## 4. 两条路线

### A. 先在 Isaac Lab 里看场景、操作厨房

优先从 Arena 厨房目录选一个抓放和一个开门示例，复用它的场景、交互资产与任务结构。这样最快回答“场景是否合适、能否互动”。之后再接本仓库 Franka 和 human avatar。换机器人后需重新验证可达性、动作接口和策略表现。

限制：Arena 的同类任务不能直接称为原版 RoboCasa365 task；需要比对初始分布、对象资产、目标判定和机器人。

### B. 从原版 RoboCasa365 迁移精确场景与任务

建议流程：

1. 固定 RoboCasa/robosuite 版本、task、layout/style、对象实例和 seed，在原后端 reset 后导出实际组装的 MJCF、资产依赖及语义 manifest。
2. 用 [Isaac Lab 转换器](https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.sim.converters.html) 的 MJCF 路线转换；也可评估 [Lightwheel MJCF2USD](https://github.com/LightwheelAI/mjcf2usd) 作为备选。两者均需在实际容器中验证，不能假设复杂厨房无损转换。
3. 分离静态房间、可交互 fixture、动态物体和机器人，修复碰撞与 articulation，保存命名映射。
4. 根据 manifest 实现 reset 和对象/区域/关节查询，移植 success 与电器状态更新，再配置动作与相机。
5. 先通过单环境物理和成功/失败案例，再增加布局和对象随机化，最后测试向量化吞吐。

转换器解决资产格式，任务适配层解决环境行为。MuJoCo 到当前项目 PhysX 路径的摩擦、驱动和接触参数不能视为数值等价。

## 5. 与本仓库的衔接

当前 [Isaac 厨房说明](../isaac_kitchen/README.md) 记录使用 Isaac Lab `3.0.0-beta2-post1` 容器，已有 Franka 接触抓取、视频输出及 human avatar 接口；这些可以作为迁移验证基础。

但是 [run_kitchen.py](../isaac_kitchen/run_kitchen.py) 的厨房加载逻辑展开实例、移除物理 API 并停用关节，当前 Lightwheel Kitchen 主要承担视觉背景。要用原厨房柜门或道具完成 task，应新增保留交互物理的加载路径，并分类配置静态碰撞体、刚体和 articulation。

本仓库使用的 `Lightwheel_Kitchen/KitchenRoom.usd` 与 Arena 的 Lightwheel RoboCasa 场景注册项不能视为同一个资产包。

[Arena 当前安装文档](https://isaac-sim.github.io/IsaacLab-Arena/main/pages/quickstart/installation.html) 指向 Isaac Sim `6.0.1` / Isaac Lab `3.0.0`。它与现有 beta 容器有版本差异。应固定验证版本，先检查可用 API 和依赖；本轮没有验证 Arena 能直接装入现有容器，也没有修改远端环境。

## 6. 建议的第一轮验证与交付物

| 阶段 | 范围 | 验收和交付物 |
|---|---|---|
| 场景查看 | Arena 选 3 个不同厨房布局；原版选 1 个精确采样实例 | 每场景多视角图、环绕视频、资产清单；检查纹理、尺度、台面高度、关节数量和碰撞 |
| 首个原版 task | `PickPlaceCounterToSink`，固定布局和对象 | 真实抓取、松手、物体稳定留在水槽；原版区域/夹爪条件有对应实现；输出成功与故意失败案例 |
| 柜体交互 | 选一个开门任务，再接柜内放置 | 门由接触打开，关节轴与限位正确，reset 可重复；不能通过直接写门位置冒充策略操作 |
| 小规模扩展 | 3–5 个 task、多个 seed | 分别报告资产加载率、有效 reset 率、语义判定一致性、rollout 成功率和资源消耗 |

先产出“看得见、可交互、一个任务闭环”的证据，再决定是否批量转换。GPU 向量化另测 1/8/32 环境的显存和步速，完整厨房与相机的开销不能仅凭 Isaac Lab 支持并行就估算。

迁移后的结果在验证协议一致前应标为 RoboCasa365 的 Isaac Lab 移植版。官方 [leaderboard](https://github.com/robocasa-benchmark/leaderboard) 当前评测 50 个 target tasks，并记录 RoboCasa 版本；365 是框架任务总量，不能混作排行榜评测数量。

**推荐决策：先走 A 获得场景与交互预览，同时用 B 做一个 `PickPlaceCounterToSink` 对照样例。完成后才有依据估计批量迁移比例和投入。**
