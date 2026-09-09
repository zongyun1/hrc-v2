# RoboCasa365 三任务迁移实验

新增：[ServeTea 原版场景与任务判定迁移](SERVE_TEA.md)，保留 PandaOmron、微波炉取杯场景及餐桌杯碟接触条件。

原版对照入口：[RoboCasa / MuJoCo 环境与运行方式](NATIVE.md) · [原版与 Isaac Lab 并排视频、数值对照](../outputs/robocasa_migration/comparisons/native_vs_isaac/index.html) · [导航差异定位与已确认的问题](NAVIGATION_DIAGNOSIS.md)。

新增：[原版 PandaOmron 移动导航迁移](MOBILE.md)，使用官方 `NavigateKitchen` 任务。固定参考根修复后，完整导航与末段保持验证通过，最终距目标 7.82 mm；[观看成功录像](../outputs/robocasa_migration/probes/722228/NavigateKitchen/preview.mp4)。

原始源码位于本地与 AICR 项目的 `external/robocasa`、`external/robosuite`；固定版本见 [sources.json](sources.json)。第三方源码与下载资产不纳入主仓库 Git。

首批任务：`PickPlaceCounterToSink`、`OpenCabinet`、`OpenMicrowave`。实验采用原版任务生成的固定 layout/style/seed，不替换成其他厨房场景。

**当前定位：可运行的独立 Isaac Lab 迁移原型。场景、物理交互和任务判定已经实测；不是完整的 RoboCasa365 Gym/训练环境，也不等于三个机器人任务都成功。**

**最终结果：三个任务的原版场景均已转换并实际运行；固定 Franka 接触操作中，OpenMicrowave 通过，OpenCabinet 与 PickPlaceCounterToSink 尚未通过。** 微波炉最终打开约 90°，并通过额外两秒保持验证。[成功视频](../outputs/robocasa_migration/probes/721878/OpenMicrowave/preview.mp4) · [最终数值](../outputs/robocasa_migration/probes/721878/summary.json)。

## 运行

在 AICR 项目目录 `/scratch/jiabenchen_umass/yz/hrc-v2`：

```bash
sbatch robocasa_migration/prepare.sbatch
# 确认 outputs/robocasa_migration/source/export_results.json 中三个任务均 exported 后：
sbatch robocasa_migration/run_convert.sbatch
# 确认三个 usd/<task>/conversion.json 均 converted 后：
sbatch robocasa_migration/run_probe.sbatch
# 用机器人接触操作场景并录制视频：
sbatch --export=ALL,ROBOCASA_PROBE_MODE=robot robocasa_migration/run_probe.sbatch
# 复现微波炉成功试验：2400 步轨迹 + 240 步保持，物理步长 1/120 秒：
sbatch --export=ALL,ROBOCASA_PROBE_MODE=robot,ROBOCASA_ROBOT_STEPS=2400,ROBOCASA_HOLD_STEPS=240 \
  robocasa_migration/run_probe.sbatch OpenMicrowave
```

准备脚本创建独立 `.venv-robocasa`，安装场景导出的运行依赖；没有安装策略训练使用的全部 RoboCasa optional dependencies。转换和仿真使用项目现有 Isaac Lab beta 容器。资产下载根据上游 box_links JSON，记录 SHA-256 和解压目录，重跑复用已下载文件。

## 输出与验证层级

- `source/<task>/original.xml`：原版采样环境，包含机器人。
- `source/<task>/scene.xml`：用于转换的厨房与物体 MJCF，物体初始位姿已写入，机器人与原控制器单独适配。
- `source/<task>/manifest.json`：任务、对象、fixture 内部区域、关节、初始状态及语言信息。
- `source/<task>/predicate_checks.json`：正反例与原版 `_check_success()` 对照。这是状态判定测试，不是机器人演示。
- `usd/<task>/conversion.json`：实际转换状态和 USD 物理结构清单。
- `probes/<job>/<task>/`：Isaac Lab 仿真视频、图片、关节轨迹和结果。

`converted` 只表示 USD 转换完成；`simulated` 表示场景仿真探针跑完。开门探针直接对门关节施加力矩，验证关节和判定迁移，不计为机器人开门成功。`robot_task_success` 单独记录，不从转换或关节探针结果推断。

## 已完成的实测（2026-09-08）

- 原版导出作业 `721351`：三个任务全部成功，固定 layout=1、style=1、seed=0。
- 原版判定对照：三个任务各一个成功、一个失败构造状态，共六个状态的原版与移植判定一致。
- 转换作业 `721367`：三个完整厨房均成功转为 USD。场景分别包含 854 / 852 / 858 个 MuJoCo geom；运行时全部匹配到对应 USD prim。
- 物理探针 `721555`：梨在台面上稳定；柜门的直接关节驱动超过开门阈值。此类驱动不计为机器人完成任务。
- 接触式机器人作业 `721673`、`721763`：三个任务均完成仿真与视频记录，无直接写物体/门位姿来冒充动作。机器人使用固定安装的 Franka，初始 reset 后只通过机器人关节控制与场景接触操作。

| 任务 | 第一轮 `721673` | 修正朝向/贴图后 `721763` | 原版成功条件 |
|---|---|---|---|
| PickPlaceCounterToSink | 梨未抓起，最大抬升约 2.8 mm | 未形成有效抓取，最大抬升约 2.5 mm | 物体中心进入水槽内部区域，夹爪距物体 >25 cm |
| OpenCabinet | 实际拉开约 49.0°，未通过 | 抓取未保持，最终约 0.45°，未通过 | 指定柜门开度达到关节范围的 90%，此例约 80.96° |
| OpenMicrowave | 实际拉开约 47.9°，未通过 | 实际拉开约 79.2°，未通过 | 开度 ≥81° |

抓放针对性复测 `721786` 使用原版手臂安装位置和低位搬运路径，最大抬升约 11.6 mm，仍未完成搬运。

微波炉针对性复测 `721846` 达到约 89.5°，满足原版判定，但达标后的记录仅约 0.8 秒，未通过额外的末段 15 个采样点保持检查。最终作业 `721878` 增加两秒保持，总共 2640 个物理步，最终约 90.0°；`source_predicate_at_end=true`、`robot_task_success=true`。这是固定实例的一次通过，不能视为泛化成功率。

可查看：[修复贴图后的厨房图](../outputs/robocasa_migration/probes/721763/PickPlaceCounterToSink/start.png)、[抓放复测视频](../outputs/robocasa_migration/probes/721786/PickPlaceCounterToSink/preview.mp4)、[柜门首轮视频](../outputs/robocasa_migration/probes/721673/OpenCabinet/preview.mp4)、[微波炉第二轮视频](../outputs/robocasa_migration/probes/721763/OpenMicrowave/preview.mp4)。柜门首轮视频中的其他电器贴图尚未修复，不应用它判断最终材质质量。

详细数值：[首轮汇总](../outputs/robocasa_migration/probes/721673/summary.json)、[第二轮汇总](../outputs/robocasa_migration/probes/721763/summary.json)。这些是固定实例上的脚本控制调试结果，不是随机种子评测成功率。

完整 USD 和下载资产保存在 AICR 的 `/scratch/jiabenchen_umass/yz/hrc-v2/outputs/robocasa_migration/usd/` 与 `external/robocasa/robocasa/models/assets/`。本地已保留源码、导出元数据、截图、视频和结果 JSON。复制 USD 时应保留每个 `scene/` 下的 payload、Textures 和 repaired_textures 子目录。

## 实测发现并修复的迁移问题

1. **语义盒被当成碰撞体/可见几何。** 转换器没有完整保留 `contype=0, conaffinity=0` 和透明区域的用途。`geometry_rules.py` 按原 MJCF 恢复碰撞与可见性，三个场景匹配率均为 100%；保留实体碰撞体，仅关闭原本不应碰撞的几何。
2. **不同资产的同名贴图覆盖。** 多个电器都使用 `T_BC001.png` 等文件名，转换后曾出现木纹水槽、木纹洗碗机。现在按源路径生成独立贴图文件，并恢复 106 / 106 / 105 个材质绑定；已查看修复后的图像。
3. **无质量铰链辅助 body 与控制标记。** 场景加载时停用原版 EEF 控制标记，对无有效质量的 hinge helper 设置 1 g / 1e-6 kg·m²，以避免 PhysX 的自动回退。其他个别辅助节点仍有质量警告，动力学尚未做到逐项等价。
4. **机器人坐标与控制接口。** 原版 PandaOmron 手臂朝 +Y；移植版需相应旋转世界 Jacobian 到机器人根坐标系。开门和抓取还涉及接触保持、局部 IK 可达性与柜体避障，不能直接复用原版 action 数组。

现有三个 USD 已通过修复作业 `721894` 固化几何与贴图修复，贴图使用相对路径；可以通过 `repair.sbatch` 重做。后续新转换已在 `convert_scene.py` 中加入同样规则。相机为了查看内部场景隐藏前墙的显示，前墙物理碰撞仍保留。

## 验证边界与下一步

上述三项操作实验尚未接入 PandaOmron 移动底盘；新增的原版移动机器人导航试验单独记录在 [MOBILE.md](MOBILE.md)。原版 OSC 控制器、完整观测/action 接口、数据集回放、向量化 reset 或任务注册仍未迁移。当前保存了原版机器人状态用于后续适配，Franka 开门实验的安装高度有所调整。成功判定沿用原版定义，额外要求末段持续满足，未放宽阈值。

用于转换的 `scene.xml` 移除了原 actuator、sensor、tendon、equality、contact 和 keyframe 部分；相关原始信息保留在 `original.xml`。当前适配覆盖这三个任务所需的刚体、关节和判定，尚未证明所有接触过滤、耦合约束、摩擦和惯量在两后端等价。

下一步优先修复稳定抓取/把手保持与接触后的 IK 轨迹，再测试多个 seed。当前证据支持“场景、交互资产和判定可以迁移”，尚不支持“原版 policy 或全部 365 tasks 已兼容”。

本地检查：`python3 robocasa_migration/test_task_semantics.py`，覆盖左右方向相反的铰链、旋转的容纳区域、夹爪距离和语义几何碰撞标记；另外有 Python 编译及 shell 语法检查。
