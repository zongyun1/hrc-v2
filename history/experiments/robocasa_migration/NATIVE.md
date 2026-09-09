# 原版 RoboCasa 与 Isaac Lab 对照环境

已生成：[并排视频与数值对照](../outputs/robocasa_migration/comparisons/native_vs_isaac/index.html) · [修正后的原版导航录像](../outputs/robocasa_migration/native/722392/NavigateKitchen/preview.mp4) · [数值 JSON](../outputs/robocasa_migration/comparisons/native_vs_isaac/comparison.json) · [导航差异定位](NAVIGATION_DIAGNOSIS.md)。

## 实测（2026-09-08）

- 作业 `722392`：修正对照运行器的 OSC 坐标初始化后，原版导航完成 840 个控制步（42 秒），退出码 `0:0`，首次达标约 8.65 秒，最终距离 3.81 cm，末段 20 个采样点全部通过；原版判定与迁移判定在所有 840 个状态上相同。
- 作业 `722420`：OSC 修正后，NavigateKitchen、PickPlaceCounterToSink、OpenCabinet、OpenMicrowave 各运行 40 个零动作控制步，全部成功加载/运行并通过逐步判定一致性检查，退出码 `0:0`。此检查不计为任务完成。
- 初始 qpos 恢复误差为零；四任务全部刚体的最大初始位置误差约 3.77–10.29 μm，低于 0.1 mm 检查容差。
- 对照使用 Isaac 作业 `722412`：保留移根时的自碰撞设置后，导航与末段保持通过，最终距离 7.81 mm。原版和迁移版的控制参数不同，不将到达时间和最终距离作为引擎性能排名。

两端录像各 42 秒，原版为 840 帧 / 20 fps，Isaac 为 630 帧 / 15 fps。已检查首末帧与媒体文件。**旧作业 `722362` 虽满足底盘导航判定，但含运行器的 OSC 初始化错误，不能用于正常原版机械臂姿态对照。** 新版使用原版 `achieved` 目标更新接口初始化正确坐标系，再切入底盘导航模式；未修改 robosuite 源码。墙地面材质差异仍存在，详细归因见诊断文档。

## 环境位置

AICR 已安装原版 RoboCasa，可通过原版 `robosuite.make()`、`env.reset()`、`env.step()` 运行，使用 MuJoCo 和官方 PandaOmron 控制器。此前的 `export_tasks.py` 只执行导出和判定探针；现在 `run_native.py` 增加完整运行、录像及动作/状态保存。

| 项目 | 路径或版本 |
|---|---|
| AICR 项目 | `/scratch/jiabenchen_umass/yz/hrc-v2` |
| Python | `.venv-robocasa/bin/python`，Python 3.11 |
| RoboCasa | `1.0.1`，`external/robocasa` |
| robosuite | `1.5.2`，`external/robosuite` |
| MuJoCo | `3.3.1` |
| NumPy | `2.2.5` |
| 固定源码提交 | [sources.json](sources.json) |

本地 Mac 保留源码和结果；完整资产与上述虚拟环境位于 AICR。该环境与 Isaac Lab 容器分开，当前安装范围覆盖 PandaOmron 任务仿真、导出与录像，没有为训练、GR1 全身 IK 等额外用途安装所有可选包。启动时有关 `mimicgen`、`mink`、`robosuite_models` 的提示不影响这里验证的 PandaOmron 任务。

如果在新 AICR 工作副本中缺少原版环境，已有 `sbatch robocasa_migration/prepare.sbatch` 会安装固定版本的仿真依赖、下载资产并导出三个操作任务；导航再执行 `.venv-robocasa/bin/python robocasa_migration/export_tasks.py --tasks NavigateKitchen`。已有导出实例用于对照时，无需重新导出或覆盖它们。官方通用安装和交互演示见 [安装说明](https://robocasa.ai/docs/build/html/introduction/installation.html) 与 [基本使用](https://robocasa.ai/docs/build/html/introduction/basic_usage.html)。

## 运行原版

在 AICR 项目根目录执行：

```bash
# CPU 原版导航，无录像；默认 840 个控制步 = 42 秒
sbatch robocasa_migration/run_native.sbatch NavigateKitchen

# 原版导航 + MuJoCo EGL 录像
sbatch --partition=rtx-devel,rtx-batch --gres=gpu:1 \
  --export=ALL,ROBOCASA_NATIVE_VIDEO=1 \
  robocasa_migration/run_native.sbatch NavigateKitchen

# 四个环境各运行 40 个零动作控制步，检查 reset/step 和判定
sbatch --export=ALL,ROBOCASA_NATIVE_STEPS=40,ROBOCASA_NATIVE_POLICY=zero \
  robocasa_migration/run_native.sbatch \
  NavigateKitchen PickPlaceCounterToSink OpenCabinet OpenMicrowave
```

不经过 Slurm 时，在有相应运行资源的机器上使用：

```bash
.venv-robocasa/bin/python robocasa_migration/run_native.py \
  --task NavigateKitchen --steps 840 --video \
  --output outputs/robocasa_migration/native/manual
```

输出为 `outputs/robocasa_migration/native/<JOB_ID>/<TASK>/`：

- `result.json`：版本、源码提交、源 XML SHA-256、任务实例、初始误差、控制频率、轨迹与原版成功判定。
- `rollout.npz`：每步提交到 `env.step()` 前的动作、完整 qpos/qvel、仿真时间、关节名称及 qpos/qvel 地址。
- `preview.mp4`、`start.png`、`preview.png`：开启录像时保存。

底盘控制通过原版 `HYBRID_MOBILE_BASE` 的 action 接口执行，机械臂使用原版 OSC。仅初始化时恢复状态；运行期间不直接写机器人位姿，也不替换原 MJCF 的关节、惯性、actuator 或摩擦定义。

## 对齐范围

原版运行器读取迁移时保存的 `original.xml`、`manifest.json` 和 `mujoco_reset.npz`，保留原版任务类和判定。它检查采样到的 fixture/导航目标，清空 MuJoCo 残留求解状态后恢复 qpos/qvel，并核对全部刚体初始位置。源 XML 导出的浮点精度会带来约十微米的位置差，因此使用 0.1 mm 初始检查容差，同时保存实际误差。

每个控制步都核对原版 `_check_success()` 与迁移判定函数；任何不一致均报错。导航另外检查末段 20 个控制采样点持续成功。其他三个任务的零动作检查只验证环境能运行，不表示机器人已完成任务。

原版导航采用同一通道路线，但为了适配原 actuator 和静摩擦，使用 0.6 m/s 指令上限、0.12 m 路点切换容差；Isaac 运行器为 0.3 m/s、0.08 m。原版是 20 Hz 控制、MuJoCo 原始物理步长；Isaac 控制为 120 Hz。**这是相同任务实例的运行对照，不是相同动作、相同控制器下的物理引擎等价性测试。** 下一步分析动力学差异时，应先统一驱动输入、步长、采样和初态，再比较关节响应与接触。

## 生成对照页

```bash
python3 robocasa_migration/compare_backends.py \
  --native outputs/robocasa_migration/native/NATIVE_JOB/NavigateKitchen/result.json \
  --isaac outputs/robocasa_migration/probes/ISAAC_JOB/NavigateKitchen/result.json \
  --output outputs/robocasa_migration/comparisons/native_vs_isaac
```

打开输出目录的 `index.html`，可同时播放、暂停、按共同时间轴拖动两个视频，并查看成功状态、距离、朝向和时间指标。`comparison.json` 保留数值与限制说明。将两端录像与结果一同下载后可离线使用；旧版 Isaac 结果未保存源 XML 哈希，因此报告不声称能独立证明历史结果的源快照完全一致。
