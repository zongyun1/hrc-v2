# 当前代码架构：Lightwheel + TRUMANS

唯一 benchmark 后端为 LW-BenchHub，运行在已验证的 Isaac Lab 3 runtime 上。
任务场景、机器人、观测、动作、奖励和结束判定均由 Lightwheel 原生配置构建。
项目在其生命周期中挂接人体运动，不再从旧 Genesis `envs/` 或 `config/` 建立任务。

```text
configs/lightwheel.json
        ↓
hrc_bench/__main__.py           默认命令行和验证结果输出
        ↓
hrc_bench/lightwheel.py         原生环境构建、人体时钟及终止前状态采集
        ├── LW-BenchHub / Isaac Lab managers（external runtime）
        ├── lw_benchhub_lab3/   Lab 3 API 适配、白色材质、导航示例
        └── hrc_bench/human.py  最小 TRUMANS 生命周期接口
                    └── isaac_human/  motion、FK、胶囊、蒙皮、接触采集
```

| 目录 | 当前职责 |
|---|---|
| `hrc_bench/` | benchmark 配置、入口、原生环境组合、人体 API |
| `configs/` | 当前 benchmark 配置；默认 Lightwheel NavigateKitchen |
| `lw_benchhub_lab3/` | 固定版本迁移、依赖设置、材质修复及导航示例 |
| `isaac_human/` | 当前人体运行组件与来源记录 |
| `scripts/` | 当前 benchmark 集群启动脚本 |
| `docs/` | 当前架构与验证结果 |
| `history/` | 归档的 Genesis 架构和阶段实验，不参与默认运行或测试 |
| `external/` | 本机外部源码及模型依赖，不提交 |
| `outputs/` | 动作资产、视频和运行结果，不提交 |

保留 `isaac_human` 和 `lw_benchhub_lab3` 模块名以兼容已验证的远程安装与导入路径；
二者都是当前 Lightwheel 运行链路的组件。旧版独立架构只存在于 `history/`。

## 开发边界

新增任务、机器人与策略通过 Lightwheel 配置和原生 `env.step(action)` 扩展。
人体仅通过 `TrumanHumanMotion` 同步时钟；策略是否观察人体真值应在评测协议中明确。
禁止当前代码导入历史代码，或重新引用旧 Genesis requirements。
根 `requirements.txt` 只提供 CPU 基础依赖；仿真使用 Isaac Lab runtime 与
`lw_benchhub_lab3/requirements-extra.txt`。

默认测试集合由 `pytest.ini` 限定在三个当前模块内。无需 pytest 的验证命令：

```bash
python -m unittest hrc_bench.test_human isaac_human.test_motion lw_benchhub_lab3.test_migrate lw_benchhub_lab3.test_navigation
python -m hrc_bench --no-human --check-assets
```

当前默认 runner 是零动作接口验证，支持单环境人体动作重放；尚未接入训练好的策略。
