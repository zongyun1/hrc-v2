# Lightwheel 人体运行组件

此模块是当前 `hrc_bench` 的人体后端，不依赖 Genesis。

- `motion.py`：TRUMANS 运动读取、SLERP、骨架 FK、坐标变换。
- `adapter.py`：Isaac Lab 运动学胶囊、蒙皮与机器人接触采集。
- `vendor/`：复用的动作 schema、骨架和蒙皮计算，来源见 [PROVENANCE](vendor/PROVENANCE.md)。
- `run_demo.py` / `diagnostic_fixture.py`：适配器诊断及测试辅助，不作为 benchmark 入口。
- `test_motion.py`：运动与适配器的 CPU 回归。

统一使用 [hrc_bench](../hrc_bench/human.py) 的 `TrumanHumanMotion` 接口接入任务。
当前只支持单环境、单人体、预生成动作重放。
旧厨房共仿真、retarget 与离线动作生成实验已移到
[history/experiments/isaac_human](../history/experiments/isaac_human/)。
