# LW-BenchHub — Isaac Lab 3 + TRUMANS

本仓库以 **Lightwheel 原生源码布局** 为开发基础。`lw_benchhub/`、
`lw_benchhub_tasks/`、`lw_benchhub_rl/`、`policy/` 和原生启动脚本均位于根目录；
已验证的 Isaac Lab 3 适配直接合入这些源码。默认执行当前仓库代码。

```text
hrc-v2/
├── lw_benchhub/           Lightwheel 核心：场景、机器人、动作、环境及工具
├── lw_benchhub_tasks/     原生任务与评测定义
├── lw_benchhub_rl/        原生强化学习配置与入口
├── policy/               原生策略接口
├── configs/              原生配置，以及 lightwheel.json 人体实验配置
├── third_party/IsaacLab-Arena/  固定版本 Arena，已合入 Lab 3 适配
├── train*.sh / eval*.sh / teleop.sh / env_server.sh
├── pyproject.toml / setup.py / install.sh
├── hrc_bench/             TRUMANS 扩展入口及生命周期集成
├── isaac_human/           人体运动、蒙皮、胶囊与接触组件
├── tools/isaaclab3/       安装、资产校验、迁移记录及回归测试
├── docs/                 开发说明与验证记录
├── history/              旧架构和阶段实验归档
├── external/             本地运行依赖、模型与缓存（不提交）
└── outputs/              动作数据与运行产物（不提交）
```

## 开发与运行

直接在 `lw_benchhub/` 修改环境与机器人，在 `lw_benchhub_tasks/` 添加任务，
在 `policy/` 或 `lw_benchhub_rl/` 开发策略。原生入口参数参见
[Lightwheel 上游说明](docs/lightwheel_upstream.md)。原生任务全集和训练路径尚未逐一验证；
已验证范围是当前固定厨房的任务接口、TRUMANS 重放和示例导航。

本项目使用已有 Isaac Lab 3 容器，不安装上游旧 Isaac Sim/Lab 运行时：

```bash
# LW_RUNTIME 可覆盖现有 Isaac Lab 3 容器路径。
bash install.sh

# 在已配置的 Isaac Lab runtime 内，从根目录运行。
python -m hrc_bench --headless --enable_cameras --episodes 2
# AICR 默认项目路径；可用 HRC_PROJECT_ROOT 指定其他 checkout。
sbatch scripts/run_benchmark.sbatch --episodes 2
```

`hrc_bench` 默认读取 `configs/lightwheel.json`，源码路径就是本仓库根目录。
默认 runner 使用零动作验证接口，不代表机器人已完成任务或训练。
人体接口、资产路径与 12 秒扩展评测时限见 [TRUMANS 使用说明](docs/hrc_usage.md)。

## 外部资产与来源

源码固定版本和 42 个上游 LFS 资产的路径、大小及 SHA-256 记录在 `upstream.json`。
大型 USD/策略资产不重复上传到本仓库。准备与清单版本一致、已下载 LFS 的上游
checkout 后，将资产复制到原生代码预期的位置：

```bash
python -m tools.isaaclab3.assets --source /path/to/downloaded/LW-BenchHub
python -m tools.isaaclab3.assets  # 检查当前根目录的资产
```

Arena 源码以 vendored 方式保存，保留其许可证；不安装 Arena 子模块中旧 Isaac Lab。
原始来源和本项目改动范围见 [源码来源](docs/upstream.md)。
TRUMANS 模型及动作文件仍独立管理，见 `isaac_human/vendor/PROVENANCE.md`。

```bash
python -m unittest hrc_bench.test_human isaac_human.test_motion tools.isaaclab3.test_migrate tools.isaaclab3.test_navigation
python -m hrc_bench --no-human --check-assets
```

[架构说明](docs/architecture.md) · [验证记录](docs/lightwheel_interface_validation.md) ·
[历史代码](history/README.md)
