# 原生 Lightwheel 根目录架构

本项目本身就是可编辑的 Lightwheel checkout。Isaac Lab 3 适配已经合入根目录
`lw_benchhub/`、`lw_benchhub_tasks/`、`lw_benchhub_rl/` 及
`third_party/IsaacLab-Arena/`；不再通过 `external/LW-BenchHub` 或独立 `source/` 运行。

- 原生开发层：`lw_benchhub` 核心、`lw_benchhub_tasks` 任务、`lw_benchhub_rl` 强化学习、`policy` 策略。
- 扩展层：`hrc_bench` 组合原生环境与人体生命周期，`isaac_human` 提供运动学碰撞及视觉实现。
- 工具层：`tools/isaaclab3` 安装当前 checkout、核验资产、保存迁移与回归工具。
- 历史层：`history` 不参与当前运行和默认测试；当前代码不得导入历史模块。

```text
原生 train / eval / teleop        python -m hrc_bench
              ↓                         ↓
        lw_benchhub ← configs / hrc_bench.lightwheel
              ↓                         ↓
       Isaac Lab managers       TrumanHumanMotion → isaac_human
              ↓
  third_party/IsaacLab-Arena
```

`configs/lightwheel.json` 中 `lightwheel_root: ".."` 相对配置文件指向项目根目录。
运行结果记录 Lightwheel 和 Arena 实际模块路径，便于确认运行的是当前修改。
`external/isaaclab3` 只存附加依赖与缓存。安装使用 editable 当前项目与 Arena；
上游旧 Isaac Lab / GR00T 子模块不包含在本项目运行链路中。

`pytest.ini` 默认只收集当前已维护的 HRC、motion 和适配回归，避免自动执行需要
专用 GPU/资产的上游测试与历史实验。根目录源码仍保留上游测试供按需显式运行。

已验证的扩展支持一个环境、一个人体、预生成 TRUMANS 重放。默认零动作 runner
验证标准 step/reset 生命周期；避人导航示例不是通用已训练策略。
