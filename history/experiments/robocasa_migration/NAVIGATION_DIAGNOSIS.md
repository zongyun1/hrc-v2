# 导航差异定位（2026-09-08）

**结论：路线基本相同，但当前两端不是等价控制。此前对照还混入了两个实现错误：原版运行器的 OSC 目标初始化，以及 Isaac 移根时遗漏的自碰撞设置。这两项已修正；显式摩擦引起的高频抖动仍存在，不能因导航成功就视为动力学迁移完成。**

[诊断图](../outputs/robocasa_migration/comparisons/navigation_diagnosis/navigation_diagnosis.png) · [可导出 PDF](../outputs/robocasa_migration/comparisons/navigation_diagnosis/navigation_diagnosis.pdf) · [详细数值](../outputs/robocasa_migration/comparisons/navigation_diagnosis/diagnosis.json)

## 1. 原版对照脚本的 OSC 初始化错误：已修正

旧 `run_native.py` 在恢复 MuJoCo 关节状态后调用 `composite.reset()`。固定版本 `OSC.reset_goal()` 把世界坐标 `ref_pos/ref_ori_mat` 存为目标，而紧接着 `base_mode=1` 将控制器切到 `desired` 更新模式；配置中的 `input_ref_frame=base` 使该世界位姿被误当作底盘坐标目标。实际初始末端目标误差 **1.282 m**。

因此旧作业 `722362` 中，即使手臂 action 为零，最大单关节偏移仍达 **166.53°**，升降柱也受到异常手臂运动影响。它只能证明原环境运行过，不能作为正常原版机械臂姿态的基线；之前将其归为两种控制器自然差异的解释不充分。

修复仅在对照运行器内：恢复源关节后更新 nullspace 初始关节，先经原版 `achieved` 路径发送零增量，建立正确的底盘坐标目标，再进入导航模式。未修改第三方 robosuite/OSC 实现。增加目标位置和旋转矩阵检查，位置误差降至 `3.85e-16 m`。新作业 `722392` 的最大手臂关节偏移为 **0.65°**，导航完成，最终距离 **3.81 cm**，首次达标约 **8.65 s**。

## 2. 速度差主要被参数与驱动增益混淆

| 设置 | 修正后的原版 | 原版匹配导航参数 | Isaac 当前导航 |
|---|---:|---:|---:|
| 作业 | `722392` | `722393` | `722412` |
| XY 速度指令上限 | 0.6 m/s | 0.3 m/s | 0.3 m/s |
| 路点切换容差 | 0.12 m | 0.08 m | 0.08 m |
| XY 速度驱动增益 | 1000 | 1000 | 1500 |
| 关节摩擦参数 | 原版 frictionloss=250 | 同左 | `250*tanh(qvel/0.01)` 显式近似 |
| 控制 / 物理频率 | 20 / 500 Hz | 20 / 500 Hz | 120 / 120 Hz |
| 中段横移采样速度 | 约 0.347 m/s | 约 0.049 m/s | 约 0.14 m/s（含交替抖动） |
| 42 秒内导航成功 | 是 | 否，距目标约 0.595 m | 是 |

原版的 `0.6` 上限不是官方导航任务或官方策略参数，而是本工程对照运行器的手写导航参数。原版任务只提供环境与成功定义。

源 `original.xml` 的平移 actuator 为 `gainprm=1000`、速度反馈 `biasprm=0 0 -1000`，限力 ±600；Isaac 对所有三个底盘关节用了 damping=1500。忽略耦合、其他负载及饱和，滑动时可用以下近似解释主要速度差：

```text
v ≈ 指令速度 − frictionloss / 速度驱动增益
原版 0.6：0.6 − 250/1000 ≈ 0.35 m/s
原版 0.3：0.3 − 250/1000 ≈ 0.05 m/s
Isaac 0.3：0.3 − 250/1500 ≈ 0.133 m/s
```

这是稳态近似推断，与采样速度相符，不是完整动力学方程。仅匹配速度上限和路点容差仍未匹配底层驱动、控制频率或机械臂控制，不能称为同 action 的物理引擎比较。

## 3. Isaac 升降柱无指令上升约 8.2 cm：已修正

转换后的根原本带有 `newton:selfCollisionEnabled=false`。此前将 articulation root 移到世界固定关节时，只迁移了根 API，没有把这个属性带过去，导致新根回落到自碰撞默认值。PhysX schema 的自碰撞默认值为 true，Newton 属性与 PhysX 对应属性存在直接映射。[PhysX schema](https://docs.omniverse.nvidia.com/kit/docs/omni_usd_schema_physics/latest/physxschema/class_physx_schema_physx_articulation_a_p_i.html) · [Newton 映射](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/dev_guide/schemas/newtonSchema.html)

单项对照，其他控制和摩擦参数保持不变：

- `722394`：升降柱目标接近零，实际稳定在 **0.081994 m**。
- `722403`：把原先关闭自碰撞的属性保留到新根，实际为 **−4.07e-8 m**，异常上升消失。

完整回归 `722412` 运行 5040 步、退出码 `0:0`，导航与末段保持通过，最终距离 7.81 mm。其原版对照 `722392` 和四任务运行回归 `722420` 也均正常退出。

`mobile_physics.py` 现在转移源根明确设置的自碰撞值，既保留 false，也保留 true；这不是对所有机器人统一禁用自碰撞。新增测试覆盖两种值和重复加载。

## 4. 仍存在显式摩擦引起的 60 Hz 抖动

原版 MuJoCo 的 frictionloss 是求解器处理的干摩擦约束，允许接近静止时的约束力；目前 Isaac 用上一物理步速度计算显式 `-250*tanh(v/0.01)`，近零行为不等价。[MuJoCo 摩擦约束说明](https://mujoco.readthedocs.io/en/latest/computation/)

旧日志每 8 个物理步采样一次（15 Hz），恰好总取到交替振荡的同一相位。新增 `--sample_every 1 --no_camera` 后，在**相同的 0.25–0.5 秒零指令窗口**做对照：

| 指标 | 自碰撞已修正，显式摩擦保留 `722403` | 仅关闭显式摩擦 `722404` |
|---|---:|---:|
| yaw 速度 RMS | 0.04722 rad/s | 7.95e-6 rad/s |
| 30 个采样点中的速度符号切换次数 | 29 | 0 |

保留摩擦时每个物理步都切换方向，形成约 60 Hz 振荡；yaw 角峰峰值约 **0.000744 rad（0.043°）**，所以普通录像不容易看出。其幅度虽小，仍会污染速度、接触与控制响应对照。关闭摩擦的试验仅用于定位问题，**没有作为默认修复**，因为删除摩擦不等于忠实迁移。

两端末段停靠残差也不能用“Isaac 更准确”解释：驱动增益、原版近静止摩擦与显式平滑近似、采样率均不同。下一步应先匹配各轴原始增益，再实现稳定且保留静摩擦效应的适配，并用统一时间分辨率复测。

## 复现诊断

```bash
# 修正后的原版，匹配 Isaac 的导航上层参数
sbatch --export=ALL,ROBOCASA_NATIVE_SPEED_CAP=.3,ROBOCASA_NATIVE_WAYPOINT_TOLERANCE=.08 \
  robocasa_migration/run_native.sbatch

# 三秒 Isaac 高频诊断（保持默认显式摩擦）
sbatch --export=ALL,ROBOCASA_PROBE_MODE=robot,ROBOCASA_ROBOT_STEPS=360,ROBOCASA_NO_CAMERA=1,ROBOCASA_SAMPLE_EVERY=1 \
  robocasa_migration/run_probe.sbatch NavigateKitchen

# 加 ROBOCASA_MOBILE_FRICTION_SCALE=0 可做关闭摩擦的定位对照
# 该设置不代表忠实迁移；报告保存实际 scale。

# 从保存的单项试验重建图和 JSON，需要 numpy/matplotlib
python robocasa_migration/analyze_navigation.py --isaac-job ISAAC_FULL_JOB
```

高频诊断仅改采样，不改物理步长。成功保持要求按采样间隔折算约一秒，避免密集采样后只检查 15 个点导致保持时长缩短。历史试验 `722228`、`722362` 保留为问题证据，最新并排对照使用修正后的试验。
