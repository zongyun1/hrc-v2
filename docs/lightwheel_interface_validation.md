# Lightwheel 默认入口与最小 TRUMANS 接口验证

本次新增 `hrc_bench` 作为默认入口，使用标准 `ManagerBasedRLEnv.step/reset`。
机器人动作为零，测试环境和人体接口；不以零动作运行宣称任务策略成功。

CPU：5 项接口测试（配置/禁用、真实动作时钟/延迟/末帧/重置、错误来源拒绝、
逐子步接触采集与复位）通过。连同 motion 和迁移回归，共 21 项通过；另有 4 项导航测试通过。
真实资产缺失时相关测试跳过；Git 中不包含人体模型资产或动作文件。

首次 GPU 双 episode 检查 `745028` 通过：每个 episode 为 1 秒，使用原生 timeout
和自动 reset。机器人动作空间为 11 维，人体时钟与原生 episode 同步复位。

## 最终 GPU 验证：2026-09-09

| 模式 | Slurm 作业 | episode | 控制步 | 结果 |
|---|---|---|---|---|
| 真实 TRUMANS 人体 | 745061 | 2 × 12 秒 | 600 + 600 | passed |
| 无人体 | 745062 | 2 × 1 秒 | 50 + 50 | passed |

两组均使用标准 `env.step`，动作空间为 `[1, 11]`，控制周期 0.02 秒。
每轮由原生 timeout 结束，`task_success=false`，自动 reset 后时钟均为 0。
有人模式每轮完整播放 9.8667 秒源动作并保持末帧；接触传感器报告机器人接触峰值
0 N，最大胶囊跟踪误差约 2.49e-7 m。无人模式不创建 avatar，接触测量为 null。
这组零动作测试验证接口生命周期，不验证机器人避人策略效果。

有人视频为 960×720、240 帧、24 秒；完整解码通过，预览确认白色机械臂和人体显示。
本地产物（不提交 Git）：
- `outputs/benchmark/745061/{result.json,preview.mp4,preview.png}`
- `outputs/benchmark/745062/{result.json,preview.mp4,preview.png}`

另从隔离提交索引导出完整 Git 树执行 21 项测试，20 项通过、1 项真实资产测试
因未包含资产而跳过；无人资产预检无需 NumPy/Isaac 可运行。带真实资产的工作区
已完成上述 21 项 CPU 测试及 GPU 验证。

## 原生源码提升到仓库根目录：2026-09-09

作业 `745720` 使用新根目录运行两个 1 秒 episode，通过标准 timeout / reset 检查，
共 100 控制步，动作空间 `[1, 11]`，人体复位时钟 `[0, 0, 0]`。
`source_paths` 确认实际加载：

- `/scratch/jiabenchen_umass/yz/hrc-v2/lw_benchhub/__init__.py`
- `/scratch/jiabenchen_umass/yz/hrc-v2/third_party/IsaacLab-Arena/isaaclab_arena/__init__.py`

25 项本地 CPU 回归通过；从提交树导出的无资产 checkout 中，23 项通过、2 项跳过。
648 个迁移源码文件通过语法与幂等检查；42 个上游 LFS 资产在 GPU 端全部通过 SHA-256
校验。源码完整性检查确认原生跟踪文件（除显式外置的大资产、Git/编辑器配置及未使用
子模块外）全部纳入本仓库。此运行仍为零动作接口测试，不代表策略成功率。

本地结果：`outputs/benchmark/745720/`。
