# Household Human Motion 扩展方案

## 目标

将 avatar 从少量固定动作片段扩展为可组合、可随机化的家庭活动角色。重点是任务覆盖和动作多样性；人机接触不是第一阶段目标。

## 总体架构

```text
Household task script
        ↓
Human intent / target locations
        ↓
Motion primitive or motion-library retrieval
        ↓
SMPL/Mixamo retargeting
        ↓
Root and joint trajectory refinement
        ↓
Genesis HumanMotionController
```

任务逻辑只描述“人要做什么”，motion backend 决定“具体怎么做”。统一接口见 `envs/avatar/human_motion.py`。

## 动作数据和模型

- [AMASS](https://amass.is.tue.mpg.de/) 提供统一 SMPL 参数，超过 40 小时、300 个 subject、11,000 多段动作，适合作为 locomotion、坐下、弯腰、伸手等基础动作先验。
- [EPIC-KITCHENS-100](https://epic-kitchens.github.io/2024) 有约 100 小时家庭厨房视频、90K action segments、97 个 verb 和 300 个 noun，适合扩展 household 行为词汇和任务脚本。它主要是第一视角视频，不能直接作为 Genesis 骨骼轨迹。
- [MotionGPT](https://github.com/OpenMotionLab/MotionGPT) 支持文本、初始姿态、末姿态和关键姿态条件，可作为后续生成式 backend。生成结果仍需 retarget、脚底接触和地面穿透修正。

## Primitive 设计

第一批建议覆盖：

```text
walk_to, turn, stop, sidestep
stand, sit, crouch, bend, kneel
reach, point, inspect, wipe, stir
pick, place, carry, open, close, pour, push
idle, walk_to_reach, reach_to_idle, sit_to_stand
```

每个 primitive 应支持位置、朝向、左右手、速度、停顿、随机风格和 seed，并输出 root trajectory、joint trajectory、动作阶段和置信度。

## 任务脚本

```yaml
name: prepare_tea
human:
  - walk_to: kettle
  - reach: kettle_handle
  - pick: kettle
  - walk_to: cup
  - pour: cup
  - place: kettle
  - idle: 2.0
```

同一任务通过改变路线、速度、手、停顿、目标位置和动作顺序生成多个 episode。任务应保持语义不变，动作实现可以变化。

## Backend 路线

1. `ReplayMotionController`：包装当前 authored/replay clip，保证回归兼容。
2. `ScriptedMotionController`：按 primitive 执行确定性 household 脚本。
3. `MotionLibraryController`：从 AMASS、Mixamo 和自采动作检索、拼接和随机化。
4. `GeneratedMotionController`：接入 MotionGPT 或扩散模型，生成新动作后再做 retarget 和物理修正。

## 分阶段实施

1. 将现有 replay motion 接入统一 controller。
2. 实现 `walk_to`、`reach`、`pick`、`place`、`sit`、`idle`。
3. 建立 household primitive registry 和 YAML task script。
4. 加入路径、速度、左右手、停顿和 seed 随机化。
5. 接入 AMASS/Mixamo library，并加入动作拼接和 foot-contact 修正。
6. 最后接生成模型，用生成结果扩充动作库，而不是直接替换任务系统。

## 验收指标

- 同一脚本可生成多个明显不同但语义一致的动作。
- root 不穿墙、不漂移，脚底不持续滑动。
- 关键动作阶段和目标物体时间对齐。
- 旧 authored clip 的结果保持可复现。
- 每个 episode 可记录 seed、primitive 序列和 motion backend，便于回放。

