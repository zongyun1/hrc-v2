# RoboCasa 厨房 + PandaOmron + HARP 人体动作

统一入口：`python -m isaac_human.run_kitchen_cosim`。
最新：[真实 TRUMANS 动作驱动 Adrian Keller 蒙皮](TRUMANS.md)。
它对应 coworker `6c7c467` 的 `tools/phase1_cosim_demo.py`：
人在厨房走动，移动底座停驻，Panda 机械臂做周期性摆动。没有人机避让闭环。

## 组成与迁移范围

| 组成 | Isaac Lab 实现 |
|---|---|
| 厨房 | 复用 `robocasa_migration` 已转换的 NavigateKitchen layout1/style1 场景 |
| 机器人 | 场景内原版 PandaOmron，保留平面移动关节、升降柱、Panda 和夹爪 |
| 物理修复 | 复用 `geometry_rules.py`、`mobile_physics.py` 的碰撞、贴图、惯性和世界固定参考根修复 |
| 人体 | `motion.py` 读取 coworker schema，`adapter.py` 驱动运动学胶囊和可选 SMPL-X 蒙皮 |
| 控制 | 关节范围中点为手臂基准，沿用 coworker 三个关节的正弦幅度和相位；控制增益为 Isaac 实现 |
| 时钟 | 120 Hz 物理，按秒采样人体与机械臂；15 fps 录像 |
| 接触 | 每个人体胶囊只过滤到 PandaOmron 刚体，不把整套厨房地面接触算成人机接触 |

默认底盘 XY 为 `(2, -1.15)`、yaw 为 π/2，通过平面关节在 reset 时放置。
运行中用位置目标停驻，不逐帧改写机器人 root pose。底盘具备移动关节，但本入口
复现停驻共仿真；实际导航仍使用 [现有导航入口](../robocasa_migration/MOBILE.md)。
厨房来自本项目同布局的任务导出，不是 coworker 缺失的原始 MJCF 文件，不能声称逐像素或动力学等价。

## 真实动作运行

### Adrian Keller 带贴图行走视频

也支持旧 Genesis 项目的 `custom_Adrian_Keller.glb`：复用 `isaac_kitchen/avatar.py`
的原始蒙皮、贴图和 `walking.py` 的交替落脚 IK。此模式不读取 SMPL-X 或 TRUMANS 动作。

```bash
sbatch isaac_human/run_kitchen_remote.sbatch \
  --legacy_avatar_glb /scratch/jiabenchen_umass/yz/assets/avatars/custom_Adrian_Keller.glb
```

默认 10 秒，从 `(4.1,-2.1,0)` 走到 `(0.9,-2.1,0)`；可用 `--walk_start X Y Z`、
`--walk_end X Y Z`、`--walk_duration SECONDS` 调整。禁止与 SMPL 动作参数混用。
额外检查终点误差 <1 cm 和脚部 IK 误差 <2.5 cm。碰撞沿用旧版的躯干和手掌代理，
没有全身碰撞胶囊，也没有人机接触力传感；相关力和距离字段为 `null`，不是零接触结论。

**已验证（2026-09-09）**：AICR 作业 `743842` 成功完成 10 秒录像，六项检查全部通过，
已检查贴图人物在厨房中的渲染画面。
[带蒙皮行走视频](../outputs/isaac_human/kitchen_743842/preview.mp4) ·
[结果](../outputs/isaac_human/kitchen_743842/result.json)。
该作业旧字段 `max_capsule_tracking_error_m` 在此模式实际表示躯干/手掌代理误差；
后续输出已改名为 `max_proxy_tracking_error_m`。

### Coworker SMPL-X 动作

在 AICR 项目根目录，先确保现有 `outputs/robocasa_migration/source/NavigateKitchen`
和 `usd/NavigateKitchen` 导出存在。准备步骤见上面的导航文档。

```bash
sbatch isaac_human/run_kitchen_remote.sbatch \
  --motion /path/to/motion_kitchen_walk.npz \
  --skeleton /path/to/smplx_skeleton_male.json \
  --skin /path/to/smplx_skin_male.npz
```

只有骨架时必须明确使用 `--capsules`。默认人体世界变换为恒等，不把源动作自动
搬到机器人旁边。需要调整时使用 `--human_yaw RADIANS --human_translation X Y Z`，
变换和文件 SHA-256 会写进 `result.json`。其他参数：`--robot_xy X Y`、
`--robot_yaw RADIANS`、`--camera_eye X Y Z`、`--camera_target X Y Z`。

`--check_assets` 可用 NumPy 环境做路径、动作、模板预检，不启动 Isaac。
USD 文件本身及依赖资产的可加载性仍需要仿真验证。

## GPU 集成诊断

真实 motion / skeleton / skin 导出尚未取得。以下显式合成胶囊人体沿通道平移，
没有真实步态、蒙皮或 TRUMANS 生成过程，仅用于检查整场集成：

```bash
python -m isaac_human.diagnostic_fixture outputs/isaac_human/diagnostic_inputs
sbatch isaac_human/run_kitchen_remote.sbatch \
  --motion outputs/isaac_human/diagnostic_inputs/motion.npz \
  --skeleton outputs/isaac_human/diagnostic_inputs/skeleton.json \
  --capsules --allow_diagnostic \
  --human_yaw 1.5707963267948966 --human_translation 2.5 -2.1 0
```

输出目录 `outputs/isaac_human/kitchen_JOB_ID/` 包含视频、首尾帧、轨迹和结果。
集成检查：完整动作播放、胶囊跟踪误差 <5 mm、底盘漂移 <5 cm、手臂实际运动幅度 >0.2 rad，
以及非有限机器人状态和空图像检查。合成输入通过时标记 `diagnostic_passed`。
接触只记录，不作为此无避让共仿真的失败条件；末端到人体胶囊的距离也不代表全机器人最小距离。
这些检查不等于安全基准、导航成功、抓取成功或人体接触力验证。

## 验证结果（2026-09-09）

AICR 作业 `743779` 完成了 8 秒合成动作，四项集成检查全部通过：

| 指标 | 实测 |
|---|---|
| 底盘最大位置偏移 | 3.27 cm（含摆臂期间瞬态；不是刚性锁定） |
| 手臂最大关节运动幅度 | 1.186 rad |
| 人体胶囊最大跟踪误差 | 0.000261 mm |
| 报告的人机接触力峰值 | 0 N；未做正接触灵敏度验证 |

[视频](../outputs/isaac_human/kitchen_743779/preview.mp4) ·
[画面](../outputs/isaac_human/kitchen_743779/preview.png) ·
[完整结果](../outputs/isaac_human/kitchen_743779/result.json)。首尾画面已检查。
14 项 CPU 回归测试通过，Python 编译和启动脚本语法检查通过：

```bash
python -m unittest isaac_human.test_motion isaac_human.test_kitchen -v
```

实际停驻控制使用位置 PD，没有施加导航控制器的显式摩擦力。
后来已从官方包部署 TRUMANS 并重新生成动作和人体资产，见上方 TRUMANS 说明。
