# ServeTea：原版 RoboCasa → Isaac Lab

新增：[抓取失败根因对照](SERVE_TEA_DIAGNOSIS.md) · [原版与 Isaac 并排录像](../outputs/robocasa_migration/comparisons/serve_tea/index.html)。两端均只形成左指侧面接触、未抓起杯子；补充指侧接触观测后已复现确认。

使用固定源码中的 [ServeTea](../external/robocasa/robocasa/environments/kitchen/composite/making_tea/serve_tea.py)，保留原厨房、微波炉、茶杯、缩放后的碟子、餐桌和 PandaOmron。实例为 **layout=2 / style=1 / seed=0**；layout=1 在原任务的排除列表中。

当前提供场景转换、实时对象/接触观测和成功判定，尚无完整取杯—导航—放杯策略，也未注册为 Gym 训练环境。`--robot_attempt` 和 `--joint_probe` 对此任务显式报错，避免错误调用已有的固定 Franka 操作脚本。

## 成功条件

严格沿用原版 `ServeTea._check_success()` 和 `object_utils`：

- 茶杯与碟子实际接触。
- 杯碟中心 XY 距离严格小于 `0.7 * saucer.horizontal_radius`，使用采样资产的实际半径。
- 原版右夹爪 EEF site 与杯中心距离严格大于 0.25 m。
- 碟子与指定餐桌实际接触。

不额外要求杯子直立、关微波炉门或执行倒茶动作。原版液体只是视觉标记，没有液体动力学。

`serve_tea_scene.py` 用两个过滤后的 Isaac Lab ContactSensor 读取 PhysX 接触对数量；不把距离或总接触力当作杯碟/桌面接触。餐桌碰撞几何从原始 fixture 的 `contact_geoms` 解析到 USD rigid body。夹爪位置来自当前机器人 body pose 和导出的 site 局部坐标。缺少接触或夹爪数据时直接报错。

## 成功示范迁移

官方 episode 1 已在原版及 Isaac Lab 完整执行成功，包含杯网格、惯性修复和放杯反馈。见 [成功示范迁移报告与复现步骤](SERVE_TEA_DEMONSTRATION.md) 和 [并排录像](../outputs/robocasa_migration/comparisons/serve_tea_demonstration/index.html)。下文为之前 layout 2 的场景与判定探针记录。

## 运行

在已安装原版运行依赖、下载完整资产的 AICR 项目 `/scratch/jiabenchen_umass/yz/hrc-v2` 中执行：

```bash
.venv-robocasa/bin/python robocasa_migration/export_tasks.py --tasks ServeTea --layout 2 --style 1 --seed 0
sbatch robocasa_migration/run_convert.sbatch ServeTea
# 转换成功后执行：
sbatch robocasa_migration/run_probe.sbatch ServeTea
# 单独的构造状态接触测试（直接将杯置于碟上方后自由落下，不计机器人成功）：
sbatch --export=ALL,ROBOCASA_SERVE_TEA_CONTACT_PROBE=1 robocasa_migration/run_probe.sbatch ServeTea
# 原版逐步运行及判定对照：
sbatch --export=ALL,ROBOCASA_NATIVE_STEPS=40,ROBOCASA_NATIVE_POLICY=zero robocasa_migration/run_native.sbatch ServeTea
```

导出自动执行移动底盘串联关节适配并生成 `mobile_dynamics.json`。场景探针保留机器人并用关节位置控制保持初态，模拟真实刚体/接触；不直接写杯碟位姿来完成任务。该探针的 `robot_task_success` 始终为 false。

## 验证记录

- 原版导出 `722660`：256 个 body、81 个关节、1500 个 geom；底盘四组运动学对照最大误差均为零。
- 原版真实接触正反例：初始状态、杯在碟上、杯悬空、碟离桌四例全部与移植判定一致。正例由 MuJoCo 实际碰撞几何构造，是判定测试，不是机器人完成任务。
- USD 转换 `722669`：350 个 mesh、1435 个 collision prim、258 个 rigid body。
- 原版运行 `722670`：40 个控制步的原版/移植判定一致。
- Isaac Lab 最终场景探针 `722832`：360 步 / 3 秒，1500 个 geom 全部匹配，153 个材质修复；末段一秒杯/碟位移范围分别小于 0.13 / 0.31 mm，碟桌接触持续成立。分组保持机器人关节后，夹爪从首帧到末帧偏移约 1.97 cm；控制器仍不是原版 OSC。
- Isaac Lab 接触正例 `722809`：第 120 步在碟子上方释放杯子，第 136 步开始满足成功条件，并保持至结束；末帧杯碟/碟桌接触点计数为 11 / 16。`mode=constructed_contact_predicate_probe`、`robot_task_success=false`，不计机器人搬运成功。
- 本地 `python3 robocasa_migration/test_task_semantics.py`：7 项测试通过，包含接触缺失、半径边界、夹爪距离边界与非有限值。

原 MJCF 的 actuator、sensor、tendon、equality、contact 和 keyframe 保留在 `original.xml`，场景转换文件中移除，约束与控制并非逐项物理等价。底盘适配添加虚拟连杆惯量；转换器将水槽的两轴把手合并为 D6 joint。当前证据不支持原版策略直接复用或动力学完全等价。

源状态、判定检查与运动学检查在 `outputs/robocasa_migration/source/ServeTea/`；完整 USD 与纹理已下载到本地，并保存在 AICR 的 `outputs/robocasa_migration/usd/ServeTea/`。探针保存 `start.png`、`preview.png`、`preview.mp4` 和含接触数量/杯碟位置的 `result.json`。

查看：[最终场景录像](../outputs/robocasa_migration/probes/722832/ServeTea/preview.mp4) · [最终场景截图](../outputs/robocasa_migration/probes/722832/ServeTea/preview.png) · [场景验证数据](../outputs/robocasa_migration/probes/722832/ServeTea/result.json) · [构造状态接触测试录像（非机器人演示）](../outputs/robocasa_migration/probes/722809/ServeTea/preview.mp4) · [原版判定对照](../outputs/robocasa_migration/source/ServeTea/predicate_checks.json) · [USD 入口](../outputs/robocasa_migration/usd/ServeTea/scene/scene.usda)。

USD 中的载荷与纹理使用相对路径，复制时保留整个 `scene/` 目录。初始关节恢复、移动底盘参考根与惯性修复由 `simulate_scene.py` 加载时执行；单独打开 USD 查看不等于运行了任务适配器。
