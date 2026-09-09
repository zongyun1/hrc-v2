# Isaac Lab 家居人机交互进度

更新时间：2026-09-07。

厨房版本已完成最小人机交接：**人伸出手掌 → Franka 抓起木块 → 递到掌心 → 人接住 → 机器人松爪撤回 → 人带着木块收手**。代码已部署到 AICR，完整流程通过远程 GPU 仿真验证。

## 最新更新：人物先走到指定位置

仓储任务现在先让人物行走 1.2 m 到蓝色交接标记，站稳半秒再伸手；机器人等待掌心就位后递出包裹。包含交替落脚、腿部 IK 和顺序验收。作业 `713727` 通过完整 GPU 验证，人物最终稳定持有 8.13 s。

- [新版演示视频](../outputs/isaac_warehouse/713727/preview.mp4)
- [行走与交接验收](WAREHOUSE.md#最新先走到位再伸手)

## 最新：仓储移动操作机器人

已新增 **Clearpath Ridgeback + Franka** 仓储工作站：移动到取件台、抓起包裹、携物横移 1.5 m、递给工作人员、松爪撤回、人携物收手。AICR 作业 `713651` 完整通过，退出码 `0:0`；人收手后稳定持有 8.13 s，接取与携物最大跟随误差 1.31 mm。

- [45 秒演示视频](../outputs/isaac_warehouse/713651/preview.mp4)
- [实现、模型边界与完整指标](WAREHOUSE.md)
- 运行：`sbatch isaac_kitchen/run_warehouse.sbatch`

官方底盘使用平面 XY/yaw 关节速度控制，未模拟轮胎驱动动力学；人物沿用简化弹簧抓持。厨房入口保留。以下为厨房版本的已完成工作和验证记录。

## 已完成

| 内容 | 实现与验证 |
| --- | --- |
| 厨房场景 | Lightwheel Kitchen USD，加入桌面、Franka Panda 和 5.5 cm、100 g 木块。 |
| 机器人抓取 | 接触与摩擦抓起木块，抬升约 18 cm。机器人单独抓取任务 `713231` 通过。 |
| Human avatar | 复用旧 Genesis 项目的 GLB 人物资产，在 Isaac Lab 中实现骨骼蒙皮、双臂 IK、手掌目标控制和碰撞代理，无需 Genesis 运行时。 |
| 基础人物控制 | 支持伸手、复位、移动人物根节点及文件命令控制；自动互动任务 `713268`、手动控制任务 `713272` 通过。手动测试中的伸手被复位中断；到达精度由独立双手 IK 检查覆盖，最大误差约 0.92 mm。 |
| 木块交给人 | 掌心朝上、手指弯曲、接取距离判定、简化人手抓持、机器人松爪撤回、人手携物收回；任务 `713442` 通过。 |
| 输出与验收 | 自动保存视频、末帧、运动轨迹及抓取/交接指标；检查不通过会将任务标记为失败。 |

## 最新交接验证

AICR 任务 `713442`：`COMPLETED`，退出码 `0:0`，运行耗时 3 分 49 秒。共 1500 个物理步，模拟 25 秒；物理与控制 60 Hz，视频 15 FPS、960×720。

| 指标 | 实测结果 |
| --- | --- |
| 交接前稳定抓取的最小抬升 | 17.2 cm |
| 人手携物收回距离 | 21.1 cm |
| 收回后的稳定持有时间 | 1.33 s |
| 接取后首个物理步的木块位移 | 0.058 mm |
| 接取与携物全过程最大跟随误差 | 1.88 mm |
| 最终持有阶段机器人抓取中心与木块最小距离 | 39.0 cm |
| 最终机器人两指总开度 | 8.0 cm |

- [演示视频](../outputs/isaac_kitchen_handover/713442/preview.mp4)
- [交接指标](../outputs/isaac_kitchen_handover/713442/handover_result.json)
- [完整轨迹与验收结果](../outputs/isaac_kitchen_handover/713442/result.json)
- 远程结果目录：`/scratch/jiabenchen_umass/yz/hrc-v2/outputs/isaac_kitchen/713442/`

## 实现入口与复现

- [run_kitchen.py](run_kitchen.py)：场景构建、机器人控制、交接状态机、渲染和验收。
- [avatar.py](avatar.py)：人物骨骼、蒙皮、IK、手掌及基础控制。
- [handover.py](handover.py)：接取距离判定与六自由度弹簧抓持力。
- [avatar_command.py](avatar_command.py)：远程无界面人物控制命令。
- [HANDOVER.md](HANDOVER.md)：交接模型、控制流程和验收条件。
- [AVATAR.md](AVATAR.md)：人物控制接口与测试记录。

当前远程项目位于 `/scratch/jiabenchen_umass/yz/hrc-v2`。运行默认交接：

```bash
ssh aicr
cd /scratch/jiabenchen_umass/yz/hrc-v2
sbatch isaac_kitchen/run_remote.sbatch
```

结果写入 `outputs/isaac_kitchen/<SLURM_JOB_ID>/`，日志位于 `/scratch/jiabenchen_umass/yz/job_logs/franka_kitchen_<SLURM_JOB_ID>.log`。运行环境为已有的 Isaac Lab 3 beta Apptainer 容器；具体路径见 [run_remote.sbatch](run_remote.sbatch)。

## 当前边界与后续工作

目前是固定场景和目标的脚本化最小实现，尚未作为完整 BEHAVIOR-1K benchmark 接入，也未训练策略。厨房家具目前作为视觉场景，桌面、木块、机器人和人物碰撞代理参与相关物理交互。

机器人通过手指接触抓取；人是运动学 avatar，人手用简化弹簧力/力矩保持木块，保留木块重力与刚体碰撞，没有在交接时瞬移木块。尚未实现逐根人手手指的真实接触抓持。

已解决机器人撤回时的 IK 发散：保存递出前的实际关节姿态，并沿关节轨迹返回。当前验收覆盖一次完整成功运行，尚未验证随机位置、不同物体或大量重复运行的成功率。

后续可按需求推进（尚未实现）：

1. 增加不同木块位置、人物位置和重复运行测试，统计交接成功率。
2. 将手动人物控制接入交接状态机，支持等待、取消和重新接取。
3. 增加人手接触细节、家具物理交互与更多家居任务。
