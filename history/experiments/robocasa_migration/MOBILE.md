# 原版 PandaOmron 移动导航迁移

后续诊断：[导航差异定位](NAVIGATION_DIAGNOSIS.md)。移根时遗漏的自碰撞设置已补齐，完整回归 `722412` 通过、最终距离 7.81 mm；[最新录像](../outputs/robocasa_migration/probes/722412/NavigateKitchen/preview.mp4)。默认显式摩擦仍有逐步交替抖动，不能以导航成功证明动力学等价。

任务为官方 `NavigateKitchen`，layout=1、style=1、seed=0；原版语言指令为 “Navigate to the microwave.”。从水槽下方柜体前移动到微波炉前，起终点约相距 1.97 m。

此任务测试携带机械臂的移动机器人导航，不包含抓取或开门。机器人来自原版场景的 PandaOmron（Omron 底盘、升降柱、Panda 手臂和夹爪），没有替换成固定 Franka。

**最新实测：修复固定参考根后，原版机器人导航通过。** 无相机作业 `722165` 和 RTX 录像作业 `722228` 均完成 5040 个物理步、退出码 `0:0`，最终距目标 7.82 mm，朝向误差余弦 0.99999993，末段 15 个采样点全部满足原版判定。两轮 630 个采样点的机器人状态一致。[成功视频](../outputs/robocasa_migration/probes/722228/NavigateKitchen/preview.mp4) · [录像试验结果](../outputs/robocasa_migration/probes/722228/summary.json) · [无相机结果](../outputs/robocasa_migration/probes/722165/summary.json)。这是 layout=1、style=1、seed=0 的固定实例验证。

## 本轮定位与修复（2026-09-08）

主要缺口在固定参考根的 articulation 定义。源 MJCF 的 `robot0_base` 是固定在世界 `(10, 10, 0)` 的参考节点，真正的底盘通过下方三个平面关节运动。导入 USD 把根 API 放在 `robot0_base` 刚体上，固定关节的 `body0` 又指向场景容器 `/base`。`mobile_physics.py` 现将该关节显式连接世界、根据组合后的世界变换重建关节帧，并把根 API 移到固定关节上。该定义符合 [OpenUSD 固定 articulation 的根节点约定](https://openusd.org/release/api/class_usd_physics_articulation_root_a_p_i.html)。

Lab 3 容器还识别导入层的 `NewtonArticulationRootAPI`；只删除标准 API 会残留第二个根。作业 `722158` 实际复现了两个根导致初始化失败，修复同时移除旧 Newton 根标记。场景中的机器人根识别改为使用适配器返回的固定关节路径，控制器 reset 时强制检查 `is_fixed_base`，避免错误结构继续跑出误导性的导航结果。底盘三个关节仍由速度目标驱动。

本轮保持原有路线、控制增益、摩擦近似、惯性恢复及默认求解设置。`722165` 中 XY 关节采样速度峰值分别为 0.138 / 0.141 m/s，yaw 为 0.058 rad/s；630 个采样点中参考根位置无变化。首次达标在第 2552 步（约 21.27 s），最终距离由 1.965 m 收敛到 0.00782 m。旧作业 `722057` 的几十 m/s 异常关节速度及错误漂移没有在这轮重现。

本地新增四项 USD 回归测试，覆盖带场景平移/旋转的世界锚点、重复加载、保留三个底盘自由度以及缺失/重复锚点报错。运行：在含 `usd-core` 的 Python 环境中执行 `python robocasa_migration/test_mobile_physics.py`。这些结构测试不替代 PhysX 动力学实测。

## 结构与控制

原版底盘通过两个平移关节和一个旋转关节模拟运动，没有显式轮胎驱动。直接转换会把三个关节合并为 PhysX D6。`mobile_structure.py` 将它们展开为三个串联关节，保留原始名称、轴向和旋转中心；新增三个 1 g 虚拟连杆。四组平移/转向状态下，对照 MuJoCo 的原机器人连杆位置和旋转矩阵，最大分量误差约 5.6e-17。此检查验证运动学，不证明动力学完全相同。

`mobile_navigation.py` 使用实际底盘位置和朝向做反馈，通过关节速度目标驱动底盘，手臂、升降柱和夹爪维持原版初始关节位置。只在 reset 写入关节状态，运行过程没有写 root pose 或逐帧传送机器人。控制器和增益是新实现，没有复用原版 OSC 或训练策略。

固定场景路线为：退到柜体前的通道、横移到目标附近、靠近微波炉。它不是通用导航规划器，没有测试其他布局或随机种子。

成功条件保持原版定义：底盘 XY 距目标 ≤0.20 m，朝向误差的余弦 ≥0.98。结果另要求最后 15 个采样点持续满足；采样间隔为 8 个物理步、物理步长为 1/120 秒。

## 复现

在 AICR 项目 `/scratch/jiabenchen_umass/yz/hrc-v2` 的 CPU 作业中执行：

```bash
.venv-robocasa/bin/python robocasa_migration/export_tasks.py --tasks NavigateKitchen
MUJOCO_GL=disable .venv-robocasa/bin/python robocasa_migration/mobile_structure.py \
  outputs/robocasa_migration/source/NavigateKitchen
```

确认源判定和运动学检查通过后，转换并运行（仿真必须等转换完成）：

```bash
sbatch robocasa_migration/run_convert.sbatch NavigateKitchen
sbatch --export=ALL,ROBOCASA_PROBE_MODE=robot,ROBOCASA_ROBOT_STEPS=4800,ROBOCASA_HOLD_STEPS=240 \
  robocasa_migration/run_probe.sbatch NavigateKitchen
python3 robocasa_migration/summarize.py JOB_ID --tasks NavigateKitchen
```

仅验证物理轨迹时，在 `--export` 中加 `ROBOCASA_NO_CAMERA=1`；作业 `722165` 使用该模式，汇总中的 `video` 为 `null`。录像沿用脚本默认的 RTX 分区。辅助录像作业 `722201` 在 B200 的渲染初始化阶段停滞后取消，没有有效录像或导航轨迹。

原始完整场景保存在 `original.xml`，去除控制定义但尚未展开底盘的场景保存在 `scene_unsplit.xml`，最终转换输入为 `scene.xml`。运动学对照保存在 `mobile_structure_checks.json`。

## 已发现的问题

- 默认零关节位置违反 Panda 第四关节的合法范围；改为原版 reset 状态初始化。
- 底盘多关节被合并为 D6；通过串联结构保持独立控制接口。
- 转换器把原版底盘 `frictionloss=250`、升降柱 `frictionloss=1000` 写入 `physxJoint:jointFriction`；力/力矩大小与摩擦系数的语义不一致。首轮完整仿真 `722019` 虽然收到速度指令，但 XY 位置基本不变，导航失败。后续加载时移除这四个旧摩擦系数，控制器按原数值施加 `-frictionloss*tanh(qvel/0.01)` 广义阻力；这是平滑动摩擦近似，未精确复现静摩擦。
- 部分底盘连杆质量依靠 MuJoCo 从几何推算，USD 未显式保存。`mobile_structure.py` 导出源模型计算的质量、质心、主惯量和主轴，`mobile_physics.py` 在加载时恢复。无质量辅助连杆仍使用 1 g，不能声称完全动力学等价。
- 1046 个源 geom 中匹配到 1045 个；缺失的是搅拌机透明、无碰撞的椭球语义区域 `stand_mixer_main_group_reg_bowl`，不用于本导航任务。其他场景的碰撞/贴图修复规则继续应用。
- 厨房水龙头仍有多关节合并警告，部分无质量辅助节点仍需额外适配；本试验不操作这些关节。

## 实测结果

| 作业 | 实际结果 |
|---|---|
| `721980` | 原版导出完成，原版与移植判定的正反例一致 |
| `721994` | 首次仿真初始化失败：Panda 第四关节默认零位置越界 |
| `722010`、`722012` | 串联底盘运动学检查通过，USD 中保留两个平移和一个旋转关节 |
| `722019` | 完整仿真完成，但底盘 XY 基本不动，未通过 |
| `722057` | 校正旧摩擦系数和源惯性后，完整运行 5040 步；底盘漂移约 0.274 m，距目标从 1.965 m 增至 2.198 m，未通过 |
| `722153` | GPU 节点 `a0016` 出现不可纠正 ECC 错误，场景加载前失败，无物理验证结果 |
| `722158` | 首次固定根适配暴露残留 Newton 根标记，因两个 articulation roots 初始化失败 |
| `722165` | 清理旧根标记、显式固定参考根后，无相机运行 5040 步，通过原版判定与末段保持检查 |
| `722228` | RTX 完整录像复测通过；采样机器人状态与 `722165` 一致，最终距离 7.82 mm |

`722057` 的关节速度出现几十 m/s 的瞬时值，而底盘本体仅缓慢漂移。这是固定根修复前的失败结果。由于 `722019` → `722057` 同时校正了摩擦和惯性，不能把那两轮表现差异单独归因于某一个改动。

高精度求解额外诊断 `722081` 停留在渲染初始化阶段后取消，重试 `722102` 未获 GPU、在排队阶段取消；没有产生有效对照数据。本轮仍使用默认求解设置。当前通过不代表与 MuJoCo 完全动力学等价；平滑摩擦、辅助连杆质量、其他布局/seed 与原版控制策略兼容性仍需分别验证。

本地四项判定测试、Python 编译和 Slurm shell 语法检查通过。完整 USD 与原资产留在 AICR，本地保留源码、任务元数据、转换结构报告、录像和结果。加载 USD 时需运行本目录的 `simulate_scene.py`，以应用 `mobile_physics.py` 中的惯性与摩擦适配。

[修正前汇总](../outputs/robocasa_migration/probes/722019/summary.json) · [修正后汇总](../outputs/robocasa_migration/probes/722057/summary.json) · [完整机器人视频](../outputs/robocasa_migration/probes/722057/NavigateKitchen/preview.mp4)
