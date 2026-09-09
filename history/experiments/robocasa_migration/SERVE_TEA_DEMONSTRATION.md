# ServeTea 成功示范驱动迁移（2026-09-08）

> 修正：下文 `724321`、`724337` 只通过原接触判定，用户指出杯子倾倒。姿态复测 `724938` 确认最终倾斜 **85.06°**，原版为 **3.22°**，因此不能作为直立放置成功。原判定不检查姿态，现增加独立的 `upright_task_success`（末段一秒每个采样倾斜 ≤10° 且原判定通过）。历史原始结果保留，对照页已更正。

## 倾倒问题跟进

最新作业 **725092 未通过**：夹爪材质收窄后，取杯约 19.36°（接近原版 20.22°），26 秒约 7.81°，碟子初始稳定；30.33 秒杯子仍为 8.9°且已接触碟子，但继续按示范下降，到 30.67 秒倾斜 107.8°，最终约 90°。此次 `robot_task_success=false`、`upright_task_success=false`。[录像与指标](../outputs/robocasa_migration/comparisons/serve_tea_upright_725092/index.html)。

本地已在 `serve_tea_scene.py` 缓存实时接触观测，在 `demonstration_policy.py` 增加满足直立和杯碟接触后的松爪/上提阶段：保持末端、0.4 秒张爪、等待到 0.8 秒，再用 1.5 秒上提 0.32 m。只通过关节控制执行。Python 编译通过，**尚未仿真验证**。自动审批拒绝了这两个新文件上传，指出用户此前明确批准的文件是另外两个，需要补充授权；远端仍是 725092 对应版本。

杯子在 8–10 秒取出时就开始翻转，到搬运阶段约 76°，并非仅在松爪时倒下。两个指垫仍接触杯子，说明双指接触不足以证明夹持姿态稳定。

- `724966`：单独加入完整杯姿态反馈仍失败，转腕过程中发生滑移/脱落，不能只依赖末段姿态补偿。
- USD 材质仅显式保留 `dynamicFriction`，`staticFriction` 无显式值（本环境 USD 默认 0）。源指垫滑动摩擦为 2，源 MuJoCo 同优先级接触使用摩擦系数的最大值；PhysX 默认组合规则不同。见 [MuJoCo 接触参数规则](https://mujoco.readthedocs.io/en/stable/modeling.html?highlight=urdf) 和 [USD 材质定义](https://openusd.org/dev/api/class_usd_physics_material_a_p_i.html)。这属于需要验证的接触模型映射，不是通过提高任意抓取力来固定杯子。
- `724983`：补静摩擦与 max 组合后，同一搬运阶段杯子倾斜约 7.37°；但对全部材质应用该映射扰动了碟子，完整任务仍失败。
- 本地已将摩擦修正收窄到通过 physics material binding 找到的夹爪指垫材质，新增测试确保桌面材质不被修改。用户明确批准提交后，两个文件已上传，移动物理测试 7 项通过（包含其他材质保持不变的断言），完整验证作业为 `725092`。此前自动审批拒绝期间未上传该修正。

已通过的检查：姿态数学与直立判定测试 3 项；原任务判定测试 7 项；收窄后移动物理测试 7 项（包括“其他材质保持不变”断言）。本地 Python 编译检查通过。带视频的 `724947` 卡在仿真应用初始化，已取消；`725062` 为收窄前配置，不能作为最终修复验证。

官方 episode 1 在原版执行真实动作成功；迁移相同场景和初态后，Isaac Lab 经网格、惯性和放置反馈修正，也完成取杯、搬运、放碟、松爪和撤离。成功作业为原版 `724088`、Isaac `724321`。这是一个场景的成功验证，不代表跨场景成功率或两个物理引擎等价。

独立复跑 `724337` 同样成功，完整采样轨迹与 `724321` 一致；[复跑结果](../outputs/robocasa_migration/probes/724337/ServeTea/result.json) 使用明确的 `verified_native_demonstration_with_placement_feedback` 模式标签。首次成功结果仍保留旧模式标签，其 `demonstration_feedback=true` 已记录启用反馈。

[并排录像与指标](../outputs/robocasa_migration/comparisons/serve_tea_demonstration/index.html) · [原版录像](../outputs/robocasa_migration/demonstrations/native/episode_000001/preview.mp4) · [Isaac 录像](../outputs/robocasa_migration/probes/724321/ServeTea/preview.mp4) · [Isaac 原始结果](../outputs/robocasa_migration/probes/724321/ServeTea/result.json)

## 为什么之前原版也失败

此前 layout 2 的实验让原版 OSC 跟踪一条失败的 Isaac 末端轨迹，并非执行已成功的官方示范；两边都抓空不能证明迁移正确。本轮改用官方人类数据集，先通过 `env.step(action)` 验证动作确实成功，再以实际成功执行得到的机器人轨迹作为迁移参考。运行中不逐帧恢复数据集状态。

数据来自官方注册表 `v1.0/pretrain/composite/ServeTea/20250805`：[官方归档](https://utexas.box.com/shared/static/2t87dmt65mb4glf4xp6ko1chyd01jr9b.tar)，大小 279029760 字节，SHA-1 `a5649e1f10c6e437d86545b5aed4b8710891bcf0`。选择 episode 1（layout 30 / style 24），671 个动作、33.55 秒。episode 0 动作执行未通过；episode 1、2 均通过，选择 1 并重复录制成功。不是所有标为示范的数据在当前运行环境里都能精确动作复现。

数据记录 RoboCasa 0.5.1；验证环境为 RoboCasa 1.0.1（`4f8a2980def75a55dff96b990745b83540425f09`）、robosuite 1.5.2（`5ce6643f3092639d08f7b0f90ed1c6a84f50552c`）、MuJoCo 3.3.1。Isaac 使用项目现有 3.0.0-beta2-post1 容器。版本差异和控制器内部状态限制了逐帧一致性，本报告依据实际完成任务而非假定原始状态重现。

## 定位到的问题与逐步验证

| 修正阶段 | 作业 | 实际结果 |
|---|---|---|
| 原版成功动作执行及录像 | 724088 | 原版成功判定通过，最后 17 个控制步为真 |
| 原版关节轨迹直接驱动 Isaac | 724110 | 合爪受阻，未完成抓取 |
| 单独修正夹爪摩擦映射 | 724126 | 仍失败，摩擦不是唯一原因 |
| 修复杯网格坐标 | 724213 / 724219 | 抓取、搬运通过；完整任务失败，碟子在开场被弹开 |
| 再恢复杯碟质量和惯量 | 724284 / 724288 | 开场碟子稳定；完整任务在下降放杯时推偏碟子 |
| 再加入放杯反馈 | 724321 | 完整任务通过，末段一秒所有采样均成功 |

1. **杯网格绕刚体 Z 轴错转 180°。** 源环境到导出 MJCF 的 21 个相关网格顶点误差为零，USD 中杯子的 17 个网格片段方向错误；修复后最大顶点误差约 8.92 nm。手指网格原本正确，实例化后的子节点名称可能误导匹配，所以审计使用所属刚体及祖先名称。修复保留原网格拓扑、材质与碰撞。刚体位姿或 FK 对齐不能代替网格验证。
2. **杯碟没有显式 USD 质量与惯性。** 转换结果仅有 RigidBodyAPI，缺失 MassAPI，PhysX 会自行推算。按源 MuJoCo 恢复质量、质心、主惯量和主轴：杯约 0.224142 kg，碟约 0.059860 kg。修复前碟子开场移动约 39 cm，修复后短测 4 秒仅约 4.9 mm。
3. **夹持位置存在跨引擎偏差。** 修复几何和惯性后仍不能只跟随原关节轨迹：实际杯子相对末端的位置不同，下降时会先碰碟并推偏。放杯反馈根据实时杯碟位置修正原末端参考，用 IK 生成关节目标；松爪开始后冻结修正量并跟随原撤离动作。此次反馈在 22.95 秒启动，冻结偏移约 `[3.3, 68.1, 39.3]` mm。

另有转换兼容性问题：个别材质负 shininess 导致 roughness 超界；仅在转换输入副本中裁剪材质参数。夹爪源 `frictionloss=1 N` 不能直接当成 PhysX 无量纲关节摩擦系数，示范模式清除该系数并施加平滑库仑摩擦力。源模型与原始 XML 均保留。

网格和动力学审计位于 `outputs/robocasa_migration/demonstrations/episode_000001/`：`native_mesh_export_check.json`、`mesh_repair.json`、`mesh_after_repair.json`、`saucer_mesh_check.json`、`object_dynamics_repair.json`（修复前）及 `object_dynamics_audit.json`（修复后）。

## 成功证据与范围

原版首次成功约 32.75 秒；Isaac 首次采样成功约 32.81 秒，运行到 37 秒。Isaac 末帧杯碟接触点 10 个、碟桌接触点 27 个，双指及侧面均脱离杯子，满足原判定的距离和接触要求，最后一秒全部采样通过。两端最终杯中心相差约 10.8 mm。

执行期间物体位姿写入计数为零；没有附着杯子、禁用碰撞或放宽成功判定。初始状态恢复发生在执行前。Isaac 跟随原版实际关节运动及夹爪 actuator 目标，并在放杯阶段使用仿真真值位置反馈；它不是原版 OSC 动作接口的直接移植，也不是基于视觉的策略。20 Hz 原版控制与 120 Hz Isaac 物理步长、接触求解和控制器不同；后续应在更多已验证成功示范上评估泛化。

## 在现有 AICR 环境复现

工作目录 `/scratch/jiabenchen_umass/yz/hrc-v2`。原版环境额外使用 `pyarrow==25.0.1` 读取动作、`usd-core==26.8` 审计 USD；官方资产包含 `generative_textures`。数据解压到 `outputs/robocasa_migration/demonstrations/ServeTea/lerobot`。

```bash
# 先在原版真实执行成功，再导出该 episode 初态。等待该作业成功结束。
sbatch --export=ALL,ROBOCASA_DEMO_EPISODE=1,ROBOCASA_DEMO_VIDEO=1 \
  robocasa_migration/run_demonstration.sbatch

# 再转换，并执行网格审计/修复及恢复源杯碟惯性。等待转换成功结束。
sbatch --export=ALL,ROBOCASA_MIGRATION_ROOT=outputs/robocasa_migration/demonstrations/episode_000001,ROBOCASA_REPAIR_MANIPULATION_MESHES=1 \
  robocasa_migration/run_convert.sbatch ServeTea

# 完整的真实接触执行；结果保存在 probes/<jobid>/ServeTea。
sbatch --export=ALL,ROBOCASA_MIGRATION_ROOT=outputs/robocasa_migration/demonstrations/episode_000001,ROBOCASA_DEMONSTRATION=outputs/robocasa_migration/demonstrations/native/episode_000001/result.json,ROBOCASA_DEMONSTRATION_FEEDBACK=1,ROBOCASA_PROBE_MODE=robot,ROBOCASA_ROBOT_STEPS=4200,ROBOCASA_HOLD_STEPS=240 \
  robocasa_migration/run_probe.sbatch ServeTea
```

上述网格修复仅对可验证的 180° Z 轴错误生效，不是通用任意网格修复。转换自动流程整合了本轮分别运行验证的修复工具。源场景和 USD 独立保存在 `demonstrations/episode_000001/`，旧 layout 2 实验结果仍保留。

代码检查：Python 编译与 sbatch shell 语法检查通过；AICR 作业 `724292` 中移动物理测试 6 项、任务判定测试 7 项通过。反馈控制由完整仿真实验验证。
