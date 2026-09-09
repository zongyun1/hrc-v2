# HRC Bench — Lightwheel + TRUMANS

项目默认 benchmark 已切换为 **LW-BenchHub / Isaac Lab 3**。统一入口为
`python -m hrc_bench`，配置为 [`configs/lightwheel.json`](configs/lightwheel.json)。
完整模块职责见 [当前代码架构](docs/architecture.md)。
可通过同一接口开启或关闭基于 TRUMANS 的人体运动。

目前范围：单环境、单人体、预生成真实 TRUMANS 动作重放；默认任务为
`NavigateKitchen / PandaOmron-Rel / robocasakitchen-4-2`。默认 runner 使用零动作验证
环境与人体生命周期，`status=passed` 表示接口验证通过，任务成功与超时单独记录。
这不是已训练的机器人 baseline，也没有把原 Genesis 任务逐一迁移到 Lightwheel。

## 环境准备

使用已验证的 Isaac Lab 3 容器与固定 LW-BenchHub 提交：
`b2bcb2d00edef691f9fcc49039cbf0bcc7464605`。

准备包含 LFS 资产和 Arena 子模块的上游源码后，通过
[`lw_benchhub_lab3/setup.sh`](lw_benchhub_lab3/setup.sh) 建立隔离迁移目录；
`LW_SOURCE_BASE` 指向准备好的源码，`LW_RUNTIME` 指向 Isaac Lab 容器。
安装细节见 [Lightwheel 迁移说明](lw_benchhub_lab3/README.md)。运行时设置：

```bash
export LIGHTWHEEL_ROOT=/path/to/lw-benchhub-lab3/source
```

人体资产不包含在 Git 中。把已有的 `motion_kitchen_walk.npz`、
`smplx_skeleton_male.json`、`smplx_skin_male.npz` 放到配置指定的位置，或修改配置路径。
配置里的路径相对该 JSON 文件解析；运动元数据必须声明 `backend=trumans`。
外观蒙皮可省略，此时显示碰撞胶囊。缺少真实动作会明确报错，不自动生成替代动作。

```bash
# NumPy 环境可运行，不需要启动 Isaac。
python -m hrc_bench --check-assets

# 无人模式不要求人体资产。
python -m hrc_bench --no-human --check-assets

# 在已经配置好的 Isaac Lab runtime 内运行。
python -m hrc_bench --headless --enable_cameras --episodes 2
python -m hrc_bench --headless --enable_cameras --no-human --episodes 2

# 本项目现有 AICR 环境。
sbatch scripts/run_benchmark.sbatch --episodes 2
```

`--episode-seconds` 覆盖时限；默认 12 秒是明确声明的项目扩展协议。
Lightwheel 原任务默认 8 秒，比较不同策略/有人无人条件时必须统一时限。
输出到 `outputs/benchmark/`，包括 `result.json`，开启相机时另有视频和预览图。

## 最小人体 API

```python
from hrc_bench import HumanMotionConfig
from hrc_bench.human import TrumanHumanMotion

human = TrumanHumanMotion(HumanMotionConfig(
    enabled=True,
    motion='/path/motion_kitchen_walk.npz',
    skeleton='/path/smplx_skeleton_male.json',
    skin='/path/smplx_skin_male.npz',  # 可省略
))
human.reset()
human.advance(0.02)
state = human.observe()  # 当前仿真时间、源时间、骨架位置、播放完成状态
```

`start_delay_s`、`yaw`、`translation` 控制固定时序和世界变换。动作保留源速度，
播完保持末帧。`reset()` 复位时钟与接触统计，`last_episode` 保存上一段统计。
NumPy-only 模式不提供物理接触测量，接触字段为 `None`。

在仿真中，`hrc_bench.lightwheel.make_env()` 通过 prestartup 创建人体，
零维 action term 按 physics substep 更新人体，非终止的统计 term 采集末子步结果，
原生 action reset 同步复位人体。机器人动作维度不增加；标准 `env.step` 的
计数、观测、termination、timeout、自动 reset 和 recorder 流程保留。

```python
# Isaac AppLauncher 启动前 configure_source；启动后创建环境。
from hrc_bench import BenchmarkConfig
from hrc_bench.lightwheel import configure_source, make_env

cfg = BenchmarkConfig.load('configs/lightwheel.json')
configure_source(cfg.lightwheel_root)
# app = AppLauncher(...).app
# env = make_env(cfg, device='cuda:0', cameras=True)
# obs, info = env.reset()
# obs, reward, terminated, truncated, info = env.step(robot_action)
# terminal_human = env.hrc_step_snapshot  # 自动 reset 前的该控制步状态
# current_human = env.human_motion.observe()
```

`observe()` 只返回当前人体状态；机器人默认 policy observation 不自动加入人体真值。
模型是否能访问人体骨架，由之后的 baseline 协议单独定义。
配置内的人体放置仅针对当前布局校验；换布局必须重新验证人体与家具碰撞。

## 验证与参考

```bash
python -m unittest hrc_bench.test_human -v
```

真实资产不存在时，对应 CPU 测试显示 skip，不能替代真实动作验收。

- [人体接口 GPU 验证记录](docs/lightwheel_interface_validation.md)
- [已有脚本避人导航与白色机械臂视频](lw_benchhub_lab3/HUMAN_NAVIGATION.md)
- [历史代码归档](history/README.md)

旧模拟器的代码、配置、依赖和脚本均归档到 `history/genesis/`，阶段实验归档到
`history/experiments/`；默认运行和测试仅使用当前 Lightwheel 模块。人体读取与蒙皮复用模块的
来源见 [PROVENANCE](isaac_human/vendor/PROVENANCE.md)。
