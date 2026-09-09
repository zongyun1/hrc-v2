# Lightwheel NavigateKitchen + TRUMANS 避人导航

**最新白色机械臂视频（作业 744897）**：
[完整视频](../outputs/lw_benchhub_lab3/results/human_744897/preview.mp4) ·
[最终画面](../outputs/lw_benchhub_lab3/results/human_744897/preview.png) ·
[结果](../outputs/lw_benchhub_lab3/results/human_744897/result.json)。
2026-09-09 重新完成相同任务，Slurm `COMPLETED / 0:0`，8 项验收全部通过。
恢复 35 处白色材质绑定，已检查停让与最终画面；视频 960×720、213 帧、21.3 秒，
完整解码检查通过。最终位置误差 17.37 cm，报告人机接触力峰值 0 N。

入口为 `run_human_navigation.py`，使用已迁移的 LW-BenchHub 与现有 Isaac Lab 3
容器。原生任务为 `NavigateKitchen / PandaOmron-Rel / robocasakitchen-4-2 / seed 0`。
机器人沿厨房通道驶向任务采样的目标；人体接近时停让，空间清空后恢复行驶。

## 实现

- 机械臂恢复为资产自带的 `PlasticWhite` 白色材质。`robot_materials.py` 在场景初始化时
  将 35 处 `OmniSurface` 透光材质绑定替换为各关节已有的白色材质；材质修改只作用于
  当前组合场景。原资产的透射权重为 0.85，是旧视频机械臂透明的原因。

- 读取此前作业 `744000` 生成的真实 TRUMANS 动作、匹配 SMPL-X 骨架及蒙皮。
  强制检查 `backend=trumans`，记录输入 SHA-256，不接受合成替代。
- 使用 SMPL-X 蒙皮显示、逐骨骼运动学碰撞胶囊。机器人与人体保持物理碰撞；
  每个胶囊的接触传感器只过滤到机器人刚体。
- 人体经过固定 yaw/translation 放到当前厨房通道，延迟 4 秒开始，按源时序播放
  9.867 秒，结束后保持末帧。人体通道中心向后侧柜台外移 60 cm；
  对全部 297 帧检查胶囊包围盒位于已审查的通道矩形内，避免岛台与冰箱干涉。该片段原本由另一厨房布局条件生成，当前是重放实验，
  没有为此 Lightwheel 场景重新运行 TRUMANS，也没有进行在线人体重规划。
- 使用当前机器人位置和人体胶囊几何的 privileged 控制。机器人用半径 0.65 m 的
  XY 圆盘近似，人体所有胶囊投影到 XY，查询当前与未来 0.8 秒的间距。
  间距 <0.30 m 停让；>0.45 m 持续 0.4 秒后恢复。
- 路线是该固定厨房的人工通道航点；进开阔处后再转向，沿柜台平行接近后停靠。
  控制停靠点距原生目标向通道侧偏移 16 cm，属于原生 20 cm 成功范围。不支持任意厨房自动规划。
  如果起终点偏离已检查的位置，入口拒绝执行，要求重新审查通道。
- 执行原生 action manager 的相对关节目标、scene 写入、physics step 和 scene update。
  人体按每个物理步更新，运行中没有机器人/目标状态注入。
  显式 stepping 保留终点状态用于持续成功检查，不触发 RL 环境自动 reset。此演示运行 21.22 秒，未执行原生 8 秒 timeout，
  因此不作为原生 benchmark 时限下的成功率结果；正式接口测试使用 `hrc_bench`。
- `migrate.py` 修复 NavigateKitchen 成功判定中遗留的 WXYZ 朝向读取，改为 Lab 3 XYZW。
  成功距离和朝向阈值保持原生值：20 cm 与 `cos(yaw_error) >= 0.98`。

## 运行

把本目录新入口、`navigation.py` 和 `robot_materials.py` 同步到现有独立目录，并确保相邻
`../hrc-v2/isaac_human/` 模块及 `trumans_744000` 资产存在。

```bash
cd /scratch/jiabenchen_umass/yz/lw-benchhub-lab3
python3 migrate.py source
sbatch --exclude=a0016,a0018 run_human.sbatch
```

输出 `results/human_JOB_ID/`：结果 JSON、逐控制步轨迹、视频、首尾帧、停让/恢复截图。
启动脚本显式检查结果状态，仿真 shutdown 即使吞掉 Python 异常也不会报成功。

```bash
python -m unittest lw_benchhub_lab3.test_navigation lw_benchhub_lab3.test_migrate isaac_human.test_motion
```

## 验收

要求原生成功持续 1 秒、实际触发停让、恢复执行、完整动作播放、停让期间位移 <5 cm、
保守圆盘与胶囊间距为正、报告人机接触力峰值 <0.1 N、胶囊位置跟踪误差 <5 mm。
力阈值用于原型诊断，尚未验证正接触灵敏度或人体冲击力精度；圆盘是避让代理，
并非精确全机器人网格距离或通用安全证明。

## 实测结果：2026-09-09

最终作业 **744828** 在 a0019 的 RTX PRO 6000 Blackwell 上通过，
Slurm `COMPLETED / 0:0`，8 项验收全部通过。已检查停让、恢复、终点画面。
修正后的画面中，人体沿通道走过，冰箱保持关闭。

| 项目 | 实测 |
|---|---|
| 仿真总时长 | 21.22 秒 |
| 停让 / 恢复 | 0.82 秒 / 6.36 秒 |
| 停让期间最大底盘位移 | 4.02 mm |
| 最小圆盘—人体胶囊投影间距 | 29.04 cm（避让代理距离） |
| 报告人机接触力峰值 | 0 N |
| 胶囊中心最大跟踪误差 | 0.000249 mm |
| 最终距原生目标位置 | 17.37 cm（原生阈值 20 cm） |
| 最终朝向误差 | 0.000206 rad |
| 原生成功持续检查 | 连续 50 个控制步，控制周期 0.02 秒 |

[完整视频](../outputs/lw_benchhub_lab3/results/human_744828/preview.mp4) ·
[停让画面](../outputs/lw_benchhub_lab3/results/human_744828/yield.png) ·
[恢复画面](../outputs/lw_benchhub_lab3/results/human_744828/resume.png) ·
[最终画面](../outputs/lw_benchhub_lab3/results/human_744828/preview.png) ·
[结果](../outputs/lw_benchhub_lab3/results/human_744828/result.json) ·
[轨迹](../outputs/lw_benchhub_lab3/results/human_744828/trace.json) ·
[日志](../outputs/lw_benchhub_lab3/gpu_744828.log)。

20 项 CPU 测试通过，覆盖真实动作刚体变换、297 帧通道包围盒、朝向、底盘命令、
停让预测/恢复以及已有 motion 和迁移回归。648 个迁移文件语法与重复执行一致性通过，
迁移补丁的反向应用检查通过。

调试中 `744761` 曾通过人机/导航数值检查，但人体碰到冰箱，未作为最终演示；
`744770` 调整人体位置后在目标外约 24 cm 受柜体限制，已终止并修正停靠航点。
当前结果只证明该固定场景的一次完整导航与停让演示，没有验证任意场景泛化。
