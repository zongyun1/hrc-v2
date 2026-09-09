# Avatar C1 首轮实验（2026-09-08）

## 选择与范围

优先试 C1 的事件驱动交接。项目现有 Isaac 场景、人物资产和机器人抓取已经可用，首先验证机器人迟到、暂停时的人物反馈，工程依赖较少，能直接服务 robot benchmark。

参考 [Human-Robot Gym 的 robot→human 实现](https://github.com/TUMcps/human-robot-gym/blob/mirror/human_robot_gym/environments/manipulation/robot_human_handover_cartesian_env.py)：其动画在接物前循环，接物后继续撤回。此次是参考行为思路后的独立实现，不是官方 MuJoCo 环境复现，没有移植其动作资产、动画循环或 weld。

现有 demo 已有接收与释放的阶段门控。新增 `HandoverReceiver` 将人物决策独立为 reach → wait → receive → carry → complete；输入仅为当前手掌是否到位、代理是否捕获物体、夹爪开度、机器人与物体的距离、人物动作是否完成。它不接收机器人内部阶段或未来计划。机器人退开超过 15 cm 且两指均打开超过 3.5 cm，持续 12 个物理步（0.2 秒）后才允许人物搬走物体；原有机器人退开阶段结束条件也保留。

身体仍用已有 IK，手部仍用 `proximity_gated_6dof_spring_grasp`。捕获标志并非真实手指接触测量。本实验只验证行为层，不证明动作自然度、多样性或五指物理抓握有所提升，也不是完整 B1 动作库升级。

## 已执行的本地检查

```bash
python3 isaac_kitchen/check_interaction.py
python3 isaac_kitchen/replay_interaction.py outputs/isaac_kitchen_handover/713442 \
  --output outputs/avatar_c1_local/replay.json
```

事件门控检查通过：未捕获物体时等待 600 步仍不搬走；短暂满足距离与开度不触发；中途夹爪闭合会重置连续确认计数；连续 12 步满足条件才允许 carry；新建控制器恢复初始状态。

另对已有 AICR 作业 713442 的记录做离线观测回放。原记录 15 Hz，以零阶保持供给 60 Hz 控制器，因此事件时刻有采样误差。这不是新一轮物理仿真，也没有生成新动作视频。

| 新控制器事件 | 回放物理步 | 时间 |
| --- | ---: | ---: |
| 到位并等待 | 120 | 2.00 s |
| 接收代理已捕获物体 | 984 | 16.40 s |
| 连续确认机器人退开，允许搬走 | 1147 | 19.12 s |
| 人物搬运结束 | 1420 | 23.67 s |

原记录实际捕获步为 982，人物开始搬运为 1236。回放说明现有轨迹能满足新门控，不能推导加入门控后的闭环成功率。机器可读结果：`outputs/avatar_c1_local/replay.json`。

## 待运行的闭环对照

| 条件 | 人物控制 | 机器人变化 |
| --- | --- | --- |
| legacy_normal | 原有控制 | 正常 |
| c1_normal | 独立事件控制器 | 正常 |
| c1_delay | 同上 | 起步延迟 180 步 / 3 秒 |
| c1_pause | 同上 | offer 中段暂停 180 步 / 3 秒 |

额外等待步只扩展总时长，不改变 nominal steps、动作持续时间、IK、机器人抓持和人物持物参数。暂停冻结机器人参考轨迹的时钟，物理仿真继续，机器人仍闭环保持当前位置目标。每组输出视频、实际状态轨迹、接收/释放/搬运事件及任务成功判据；失败也会进入汇总。

四组各一次仅是功能 smoke test，不能作为统计成功率或相对旧版性能提升证据。若要证明延迟鲁棒性相对旧版改善，还需给旧版加同样延迟/暂停对照。

运行入口：`isaac_kitchen/run_interaction.sbatch`；可用 `KITCHEN_PROJECT_DIR` 指向包含 `isaac_kitchen/` 的独立实验目录。该脚本串行运行四组，申请一张 GPU，最长 30 分钟。`summarize_interaction.py` 汇总所有条件。

拟上传目录：`aicr:/scratch/jiabenchen_umass/yz/hrc-v2/experiments/avatar_c1_20260908/isaac_kitchen/`，不覆盖既有 demo。用户明确授权后已上传并提交 GPU 作业 **721134**。结果待运行验证。日志：`/scratch/jiabenchen_umass/yz/job_logs/avatar_c1_721134.log`；结果目录：`/scratch/jiabenchen_umass/yz/hrc-v2/experiments/avatar_c1_20260908/outputs/avatar_c1/721134/`。

首次作业 721134 在 a0016 的 Isaac 启动阶段报 `cudaErrorECCUncorrectable`，尚未进入实验；已取消并排除该节点，重提作业 **721136**。新日志和结果目录将上述路径中的 721134 替换为 721136。该基础设施失败不计入方法效果。
