# LW-BenchHub 源码运行验证

验证日期：2026-09-08。目标是在不使用此前迁移实验结论的情况下，下载上游源码，实际启动一个原生厨房任务。

## 固定版本

- LW-BenchHub：`b2bcb2d00edef691f9fcc49039cbf0bcc7464605`
- Isaac Lab-Arena：`c7b70779f103e10d690d1a13863e8d77da7fc782`
- Isaac Lab：`6acdd82a1633732d32bb575e3d792e34fdeb437e`，VERSION 为 `2.3.0`
- Isaac Sim：`5.0.0`（上游安装脚本指定）
- Lightwheel SDK：`1.0.3`

本地源码在 `external/LW-BenchHub/`。本机未安装 git-lfs，本地主要用于源码阅读；远程 `aicr:/scratch/jiabenchen_umass/yz/lw-benchhub-check/source/` 已下载 28 个 LFS 文件和锁定的子模块，源码目录约 1.9 GB。

## 验证方式

`smoke.py` 使用上游 `parse_env_cfg` 构建 `OpenDrawer`、`PandaOmron-Rel`、`robocasakitchen-4-2`，尝试 reset、120 个零动作控制步、RGB 渲染和第二次 reset。检查机器人与关节资产状态是否为有限数。结果写入 `result.json`，有渲染帧时写出预览图片和视频。

该检查不加载任务策略，不把零动作运行视为机器人成功打开抽屉。也不验证全部任务、并行训练或 avatar 集成。

## 已复现的准备问题

1. Arena 的嵌套子模块使用 GitHub SSH 地址。远程没有相应 GitHub SSH 认证；用单次 `git -c url.https://github.com/.insteadOf=git@github.com:` 参数即可下载公开仓库，无需修改账号配置。
2. 宿主系统 glibc 为 2.34，Isaac Sim 5.0 的 wheel 为 manylinux_2_35。使用已有 Ubuntu 容器提供 glibc 2.39，容器里的仿真软件另外安装到隔离 Python 3.11 环境；不使用容器自带的 Python 3.12 / Isaac Lab 3.0 作为本次仿真运行时。
3. 上游 `lw_benchhub/core/tasks/base.py:25` 使用 `from lightwheel_sdk.loader import ENDPOINT`，SDK 1.0.3 不再从该位置导出。独立 Python 导入已复现 `ImportError`。`sdk-1.0.3.patch` 把导入改到 SDK 实际定义它的 `lightwheel_sdk.client`。
4. 源码安装沿用上游 `install.sh` 中的 flatdict 4.0.1 → 4.0.0 临时安装处理，安装后恢复该依赖仓库源码。

## 资产请求

使用官方 SDK 请求 `robocasa / robocasakitchen / layout 4 / style 2` 已成功，下载压缩包约 133.19 MB。没有遇到登录或授权拦截。

场景版本 ID：`3139818f-3716-4a9c-bb91-a1dfde94de66`。

## 2026-09-09 GPU 验证

作业 `744254` 在 RTX PRO 6000 Blackwell 上通过 **无相机** 原生任务检查：

- `OpenDrawer` / `PandaOmron-Rel` / `robocasakitchen-4-2`。
- 动作空间 `(1, 11)`，完成 120 个零动作控制步；关节位置、速度、根状态和刚体状态均为有限数。
- 第二次 reset 成功，初始和 reset 后任务成功标志均为 false。
- 通过 fixture 接口将抽屉置为全开，原生成功判定为 true；置为全关则为 false。
- 耗时约 50 秒，没有渲染帧，也没有运行完成任务的机器人策略。

[完整结果](../outputs/lw_benchhub_check/results/744254/result.json) · [运行日志](../outputs/lw_benchhub_check/gpu_744254.log)。

复现时使用当前 `run.sbatch`，设置 `LW_ENABLE_CAMERAS=0`，传入 `--asset-timeout 60`。
本次独立 runtime 的 `warp-lang` 从 1.17.0 调整为 1.7.1；使用 `--cleanenv` 并显式传入运行环境。
这些调整没有解决相机启动崩溃，不能据此断言 Warp 是崩溃原因。

已修复验证脚本自己的问题：原先只在 step 循环使用 `inference_mode`，
造成第二次 reset 在该模式外修改 inference tensor 时失败；改为 `no_grad` 后通过。
检查现在也会拒绝空画面、常量画面和非有限数画面，并记录抽屉判定探针。

带相机启动在 `744163`、`744191`、`744209`、`744253`、`744258` 中发生原生崩溃，
尚未进入任务验证。清理环境变量、调整 Warp、换节点、使用 Isaac Sim 自带 experience、
关闭 Fabric scene delegate 的对照均未通过；移除 cmeel 动态库搜索路径的 `744319` 也崩溃。
日志伴随 Warp CUDA UUID 错误，
但尚不能把这个错误认定为崩溃根因。

资产 API 在旧作业 `729197`、`729205` 中分别出现 10 秒和 60 秒超时；
本次无相机运行下载/采样成功，说明这不是已确认的授权拦截。

## Human 接入状态

`isaac_human.adapter.IsaacHuman` 增加 `quaternion_order='wxyz'`，用于此处 Lab 2.x 的
初始姿态与 `write_root_pose_to_sim` 接口；默认仍为现有 Lab 3 的 XYZW 接口。
12 项 motion 单元测试通过，但尚未完成 LW-BenchHub 中的人体共仿真，
也未实现或验证“人开抽屉、机器人放物体”的完整协作任务。
