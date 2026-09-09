# LW-BenchHub → Isaac Lab 3 相机适配

新增：[NavigateKitchen + 真实 TRUMANS 人体避让任务](HUMAN_NAVIGATION.md)，
包含通道导航、停让/恢复控制、原生成功判定与接触记录。下文为此前 OpenDrawer smoke 的结果。

目标运行时为现有 `isaac-lab_3.0.0-beta2-post1_amd64.sif`，使用容器的
Python 3.12、Torch 2.10.0+cu128 和 Warp 1.13.0。容器中的 `isaaclab` Python
包元数据为 `6.1.16`；这不等同于容器产品版本。不会加载上游嵌套的旧 Isaac Lab。

当前适配验证对象：`OpenDrawer` / `PandaOmron-Rel` / `robocasakitchen-4-2`。
**2026-09-09 已通过带相机的完整 smoke 验证。** 最终作业 `744660` 在 a0017
的 RTX PRO 6000 Blackwell 上完成，Slurm 状态 `COMPLETED / 0:0`，约 79.4 秒。

- 960×720 RGB，30 帧视频；路径追踪每像素 64 samples，避免 NGX 不可用时的实时渲染噪点。
- 动作空间 `(1, 11)`，完成 120 个零动作控制步，关节位置/速度和根状态均为有限数。
- 第二次 reset 成功，初始与重置后的任务成功标志均为 false。
- 原生 fixture 全开时成功判定 true，全关时 false；保存对应截图。
- 前一作业 `744644` 在 a0019 同样通过，使用实时渲染，画面噪点较多。

[预览图片](../outputs/lw_benchhub_lab3/results/744660/preview.png) ·
[视频](../outputs/lw_benchhub_lab3/results/744660/preview.mp4) ·
[抽屉打开](../outputs/lw_benchhub_lab3/results/744660/probe_open.png) ·
[完整结果](../outputs/lw_benchhub_lab3/results/744660/result.json) ·
[日志](../outputs/lw_benchhub_lab3/gpu_744660.log)

验证范围是上述单个原生任务的初始化、相机和交互接口。未验证机器人策略完成任务、
全部任务/机器人、训练、遥操作、录像数据集导出或 Human 共仿真。

## 源码与迁移

- LW-BenchHub: `b2bcb2d00edef691f9fcc49039cbf0bcc7464605`
- IsaacLab-Arena: `c7b70779f103e10d690d1a13863e8d77da7fc782`
- 远程独立目录：`aicr:/scratch/jiabenchen_umass/yz/lw-benchhub-lab3`
- `source/` 从已下载 LFS 资产的 `lw-benchhub-check/source/` 独立复制。
- 对独立源码执行 `python3 migrate.py source`。迁移脚本可重复执行。
- `lw-benchhub.patch` 和 `arena.patch` 为各仓库对应 HEAD 的迁移差异，供审阅；
  正常安装运行迁移脚本即可，无需重复应用 patch。
- 使用 `requirements-extra.txt` 向 `deps/` 安装附加依赖（`--no-deps --target`），
  再以同样参数 editable 安装 `source/` 与 `source/third_party/IsaacLab-Arena/`。
  核心仿真软件由容器提供。
- `setup.sh` 汇总独立复制、版本检查、迁移与依赖安装步骤；`LW_SOURCE_BASE` 可指定
  已下载 LFS 的相同提交源码，`LW_RUNTIME` 可指定该容器的位置。

迁移包括显式导入新版导出位置、PhysX 后端配置、ProxyArray 的 Torch 访问、
配置四元数 WXYZ → XYZW，以及保留 Arena Pose 的 WXYZ 序列化约定并在边界转换。
停用复制旧 Isaac Lab 私有实现的 stepping/reset/manager monkey patches。
fixture 改为新版关节写入接口与 int32 索引；底盘放置仅使用总接触力，因此移除
不再符合新版 PhysX 匹配约束的地板过滤器，保留底盘接触传感器。

## 验证

```bash
python3 -m unittest tools.isaaclab3.test_migrate
# aicr 上，在上述独立目录：
sbatch --exclude=a0016,a0018 run.sbatch
```

节点排除是本次集群排障设置：a0016 出现 CUDA 不可纠正 ECC 错误。
脚本检查初始化、120 个零动作控制步、有限状态、第二次 reset、抽屉开关成功判定，
并通过显式 Camera sensor 保存 RGB 图片和视频。
抽屉判定使用 fixture 状态注入，不表示机器人策略完成任务。
每次作业输出 `results/<jobid>/result.json` 和 `gpu_<jobid>.log`。

本地已检查全部 648 个 Python 文件迁移后的语法，以及重复执行不改变结果。
4 个迁移测试通过；从原始源码迁移后的文件能够反向应用两个实际 GPU 源码补丁，
确认复现脚本与已执行版本一致。该检查也覆盖四元数转换包装移除的幂等性。

该 beta 仍有部分旧 API 的弃用警告，不能据此宣称兼容未来所有 Isaac Lab 3.x。
SDK 场景与物体资产仍由上游服务提供；依赖版本固定不代表远端资产采样永久固定。
