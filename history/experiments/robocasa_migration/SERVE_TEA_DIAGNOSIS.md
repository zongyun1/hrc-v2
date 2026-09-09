# ServeTea 抓取失败对照（2026-09-08）

> 后续进展：已用官方成功示范在原版和 Isaac 完成任务，见 [成功示范迁移报告](SERVE_TEA_DEMONSTRATION.md)。下文保留此前 layout 2 的失败诊断，其 FK 检查不能排除网格转换问题。

**当前直接失败原因是抓取没有形成双指夹持：两端都只出现左手指侧面碰杯，随后抓空。** 原版模型的运动学与迁移版高度一致；当前证据不支持把这次失败归结为 Isaac 独有的问题。任务仍未完成，尚未验证后续搬运和放杯。

[并排视频与数值](../outputs/robocasa_migration/comparisons/serve_tea/index.html) · [汇总 JSON](../outputs/robocasa_migration/comparisons/serve_tea/comparison.json)

## 实验与结果

固定导出实例 layout=2 / style=1 / seed=0，保留 PandaOmron、微波炉与杯子。没有运行期间写物体位姿、关闭碰撞、附着杯子或修改成功判定。

| 实验 | 作业 | 结果 |
|---|---|---|
| 当前 Isaac 策略重新运行，2100 步/17.5 秒，有录像 | 723571 | 第 1500 步报告 `Cup did not lift with the fingers`，最大杯中心升高 1.88 mm |
| 原版 RoboCasa / MuJoCo，OSC 跟踪 Isaac 已实现末端轨迹，350 控制步/17.5 秒，有录像 | 723597 | 最大杯中心升高 3.08 mm，未完成任务；51 个接触点记录全部为 `finger1_collision / teacup_g17`，无右指或指垫接触 |
| 原版模型离线 FK 与接触几何检查 | 723603 | 全轨迹最大末端位置误差 4.29 μm，姿态误差 2.19e-6 rad；杯固定在原初态时也只检出左指相关接触 |
| Isaac 补充手指侧面传感器，1600 步，无录像 | 723630 | 34 个采样帧有 `finger1_side_cup` 接触；右指侧面、两个指垫均为零；再次抓空 |

原版重放输入为之前的 `723477` 轨迹。本次 Isaac `723571` 与其前 262 个共同采样帧逐项一致；新复现额外保存的末帧不纳入这个前缀检查。原版使用官方 OSC 的 absolute/world 输入模式、原 actuator 与原接触几何，通过 `env.step()` 运行；夹爪通过官方增量接口跟踪开口，底盘保持零速度。

这不是官方专家策略的成功演示，也不是相同 action 或相同控制器的物理引擎等价测试。原版 OSC 接近杯子的跟踪误差最大约 25.5 mm、合爪阶段平均约 10.0 mm，所以不能据此断言两端动力学相同。两端独立出现相同的单侧接触，加上微米级 FK 对照，支持优先修正抓取轨迹与闭环条件。

## 代码中的具体问题

1. `serve_tea_policy.py` 用固定偏移 `[0, -0.057, 0.038]`（经杯的初始 yaw 旋转）指定杯把抓取点，使用 `cup_start` 而非当前杯姿态。本次进入 close 时杯子已经偏移约 9.54 mm，末端离固定目标仍约 8.83 mm。
2. reach/close 的阶段切换只要求时长和位置误差 `<0.018 m`，没有姿态误差或双指接触条件。这个容差对杯把抓取过粗：原指垫盒半尺寸仅 8×4×8 mm。满足 IK 阈值不表示两指夹住了杯把。
3. 合爪固定时长结束就进入 lift，直到杯子未升高 35 mm 才发现失败。没有在 close 阶段确认抓取形成，也没有检测侧面推杯后重新定位。
4. 旧接触日志里的 `finger1_cup` / `finger2_cup` 仅观察指垫所在的 tip rigid body，漏掉独立父 body 的手指侧面。现在新增 `finger1_side_cup` / `finger2_side_cup`，实际运行确认左指侧面接触。旧字段保留兼容，不能把它们的零值解读为整根手指没碰杯。

本次补充观测和对照工具，没有把未验证的抓取修改写入默认策略。下一步应先在原版完成可动态保持的双指杯把抓取，再迁移同一组明确的末端/开口目标；用实时杯姿态重新定位，单独收紧抓取位置和姿态阈值，并以双侧接触持续成立作为进入抬升的必要条件。静态接触候选不能代替真实闭合与抬升验证。

## 可复现入口

在 AICR 项目根目录运行（仿真命令应放在相应 CPU/GPU Slurm 作业中）：

```bash
# 原版真实动力学跟踪，不是 qpos 回放
.venv-robocasa/bin/python robocasa_migration/run_native.py \
  --task ServeTea --steps 350 --video \
  --replay outputs/robocasa_migration/probes/723477/ServeTea/result.json \
  --output outputs/robocasa_migration/native/tea_compare_723477

# 离线 FK，明确不计为 rollout
.venv-robocasa/bin/python robocasa_migration/diagnose_serve_tea_geometry.py \
  --isaac outputs/robocasa_migration/probes/723477/ServeTea/result.json \
  --output outputs/robocasa_migration/comparisons/serve_tea/geometry.json

# 原策略 + 指侧/指垫接触观测
sbatch --export=ALL,ROBOCASA_PROBE_MODE=robot,ROBOCASA_ROBOT_STEPS=1600,ROBOCASA_HOLD_STEPS=0,ROBOCASA_NO_CAMERA=1 \
  robocasa_migration/run_probe.sbatch ServeTea

# 本地或远程生成视频对照页
python3 robocasa_migration/compare_serve_tea.py \
  --native outputs/robocasa_migration/native/tea_compare_723477/ServeTea/result.json \
  --isaac outputs/robocasa_migration/probes/723571/ServeTea/result.json \
  --geometry outputs/robocasa_migration/comparisons/serve_tea/geometry.json \
  --output outputs/robocasa_migration/comparisons/serve_tea
```

四个实际作业均完成。新增 Python 文件通过编译检查，任务语义测试 7 项通过。本地移动物理测试因没有 `pxr` 无法运行；本次未修改移动物理实现。

原始证据：[原版运行](../outputs/robocasa_migration/native/tea_compare_723477/ServeTea/result.json)、[Isaac 新复现](../outputs/robocasa_migration/probes/723571/ServeTea/result.json)、[补充接触观测](../outputs/robocasa_migration/probes/723630/ServeTea/result.json)、[FK 检查](../outputs/robocasa_migration/comparisons/serve_tea/geometry.json)。
