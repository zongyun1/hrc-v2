# Ridgeback + Franka 仓储人机交接

场景切换为程序化仓储工作站，含货架、纸箱、取件台和交接通道；家具参与碰撞。
机器人采用 NVIDIA 官方 Clearpath Ridgeback + Franka 资产，7 轴手臂和双指夹爪安装在移动底盘上。

人物先从交接点右侧 1.2 m 处走到蓝色标记，站稳 0.5 s 后才伸手；机器人到达人前后必须等掌心就位才能递出。

机器人流程：底盘从取件点后方 1.4 m 处出发 → 停靠取件台 → 接触抓起 5.5 cm、100 g 方块（小件包裹代理）→ 抬升并稳定持有 → 横向移动 1.5 m 到工作人员处 → 等待人手就位 → 递到掌心 → 松爪撤回 → 人携物收手。

```bash
rsync -az isaac_kitchen/ aicr:/scratch/jiabenchen_umass/yz/hrc-v2/isaac_kitchen/
ssh aicr 'sbatch /scratch/jiabenchen_umass/yz/hrc-v2/isaac_kitchen/run_warehouse.sbatch'
```

运行 2700 物理步（45 秒），视频 15 FPS。输出位于远程
`/scratch/jiabenchen_umass/yz/hrc-v2/outputs/isaac_warehouse/<jobid>/`：
`preview.mp4`、`preview.png`、`result.json`、`handover_result.json`、`avatar_result.json`。
原厨房入口 `run_remote.sbatch` 保持可用。

验收覆盖稳定抓起、运输时包裹与夹爪的距离、底盘最终停靠误差、人手接取连续性、机器人松爪撤回和人手持有。
移动时保持手臂关节目标，底盘采用限速的闭环速度控制；接近目标并减速后才进入下一个阶段。

模型边界：官方资产使用 XY 平移与 yaw 转动关节表达底盘运动，未模拟轮胎驱动力、侧滑或里程计。本例是已知空闲路线的脚本控制，不含地图导航或动态避障；人手沿用接近判定加弹簧抓持模型。机器人抓取和运输中的包裹仍为动态刚体。

资产与配置来源：
- [NVIDIA 机器人资产目录](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/assets/usd_assets_robots.html)
- [Isaac Lab Ridgeback 配置](https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_assets/isaaclab_assets/robots/ridgeback_franka.py)

## 最新：先走到位再伸手

AICR 作业 `713727` 完整通过，退出码 `0:0`，耗时 4 分 38 秒。
人物行走 1.2 m，第 360 步到达，第 390 步开始伸手，第 511 步掌心就位；第 1748 步接取，最终稳定持有 8.13 s。物理步长 1/60 秒。验收额外确认所有行走帧均未启动接取伸手，所有伸手帧人物均已在目标位置。

- [新版 45 秒视频](../outputs/isaac_warehouse/713727/preview.mp4)
- [行走画面](../outputs/isaac_warehouse/713727/walking.png)
- [到位后伸手画面](../outputs/isaac_warehouse/713727/reaching.png)
- [完整验收报告](../outputs/isaac_warehouse/713727/result.json)
- [人物行走与动作时序](../outputs/isaac_warehouse/713727/avatar_result.json)

## 已验证运行

AICR 作业 `713651` 于 2026-09-07 完成，退出码 `0:0`，耗时 4 分 34 秒。完整 15 个控制阶段全部执行，`result.json` 状态为 `passed`。录像 45 秒、675 帧、960×720、15 FPS；已核对交接帧与最终持有帧。

- [完整演示视频](../outputs/isaac_warehouse/713651/preview.mp4)
- [交接画面](../outputs/isaac_warehouse/713651/handover.png)
- [最终持有画面](../outputs/isaac_warehouse/713651/preview.png)
- [完整验收与轨迹](../outputs/isaac_warehouse/713651/result.json)
- [交接指标](../outputs/isaac_warehouse/713651/handover_result.json)

| 指标 | 实测 |
| --- | --- |
| 抓起后最小抬升 | 18.1 cm |
| 携物横移 | 1.5 m |
| 运输阶段最大包裹—夹爪中心距离 | 3.72 mm |
| 最终底盘停靠误差（模拟关节坐标） | 0.0124 mm |
| 接取瞬间包裹位移 | 0.0194 mm |
| 接取与携物收手全过程最大跟随误差 | 1.31 mm |
| 人手携物移动距离 | 18.95 cm |
| 人收手后稳定持有 | 8.13 s |
| 最终包裹与机器人夹爪中心最小距离 | 25.87 cm |

这是固定初始条件的一次完整成功验证，未统计随机场景成功率。

## 实现细节与检查

5.1 资产中的虚拟 `world` 链接在组合场景中显式固定，并把 articulation root 放到该固定连接上；真实底盘仍通过 XY/yaw 关节移动。控制器按固定或浮动 articulation 的实际形状选择雅可比的刚体行和关节列；仓储任务使用世界坐标求解末端目标。

导航使用抬高夹爪的姿态，避开桌面。为适应移动底盘上的安装高度，仓储任务的接取掌心高度为 0.97 m，收手后为 1.15 m；人物站位与厨房不同。

独立人物轨迹检查（不启动 Isaac Sim）：

```bash
python isaac_kitchen/check_mobile_reach.py /path/to/custom_Adrian_Keller.glb
```

需要 NumPy 和 SciPy。检查 31 个接取至收手的轨迹点，验证掌心误差小于 5 mm、掌心朝上和骨长保持；该检查不替代 GPU 物理交接验收。

## 人物先走到位再伸手

人物初始位置为 `(2.10, 1.65, 0)`，交接站位为 `(0.90, 1.65, 0)`（相对 `--robot_xy` 平移）。6 秒内走完 1.2 m，双脚交替迈步、落脚，采用腿部两段 IK 和轻微屈膝保持落脚目标。到达误差低于 5 mm，站稳半秒后启动两秒的伸手动作。

验收报告新增 `avatar.walking`，记录到达时间、开始伸手时间、行走距离及双脚跟踪误差。验收要求到达 → 站稳 → 伸手 → 掌心就位的顺序成立，再检查完整物理交接。`avatar_result.json` 的轨迹同时记录根位置、双脚位置和行走状态。

这是运动学骨骼行走动画，尚未模拟人体平衡及脚部接触动力学。独立检查：

```bash
python isaac_kitchen/check_walking.py /path/to/custom_Adrian_Keller.glb
```
