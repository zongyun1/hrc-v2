# TRUMANS 动作 → Adrian Keller 蒙皮 → Isaac Lab

这条入口保留 Adrian Keller 的 GLB 外观，动作来自实际运行的 TRUMANS 扩散模型。
不调用 `WalkMotion` 或腿部程序化 IK。默认保留源时间、根节点 XY 和全身关节旋转，
以每个关节的绑定姿态校正不同骨架；根高度按两个人体的髋部离地高度比缩放。
手指保持原 GLB 绑定姿态。体型差异仍可能造成脚滑或轻微穿地。

## 已生成的动作

- 官方代码：<https://github.com/jnnan/trumans_utils>，commit `d74e48d9db346189031ddfe3ef1f69fe3beb2bbc`。
- 官方模型包：README 链接的 `trumans_demo.zip`，下载到 AICR `external/trumans-runtime/`。
- 复用 coworker `6c7c467` 的 SMPL-X 提取和动作导出器，没有改动模型或采样算法。
- `trumans_occupancy.py` 从现有 RoboCasa 导出的 MuJoCo 碰撞几何世界 AABB 构建 2 cm 网格。
  排除机器人，保留厨房；这是静态场景条件，没有机器人轨迹条件或闭环避让。
- 作业 `744000` 生成成功：前四次质量筛选拒绝，seed 4 被接受，297 帧，
  播放时长 9.867 秒，沿厨房通道从右向左。沿用 coworker 的 30 fps 假设，
  TRUMANS 本身没有在导出里声明帧率。
- 导出器判定：悬空比例 0%，中位人体高度 1.712 m，整体地面对齐约 +3.6 cm。
  该质量门槛不等于完整脚接触/碰撞验证。

资产目录：`outputs/isaac_human/trumans_744000/`，含动作、匹配骨架、源蒙皮和变换。
源蒙皮在 Adrian 重定向模式中仅用于源人体几何/脚底高度校准，不作为显示网格。

## 运行

```bash
sbatch isaac_human/run_kitchen_remote.sbatch \
  --legacy_avatar_glb /scratch/jiabenchen_umass/yz/assets/avatars/custom_Adrian_Keller.glb \
  --retarget_trumans \
  --motion outputs/isaac_human/trumans_744000/motion_kitchen_walk.npz \
  --skeleton outputs/isaac_human/trumans_744000/smplx_skeleton_male.json \
  --skin outputs/isaac_human/trumans_744000/smplx_skin_male.npz
```

`--retarget_trumans` 要求输入元数据 `backend=trumans`，拒绝合成诊断输入。
输入 SHA-256、源采样元数据、身高缩放和旋转映射误差均写入结果。
预检还会把 FK 与源 SMPL-X 缓存关节位置比较，超过 1 cm 则拒绝运行。

重新生成：`sbatch isaac_human/generate_trumans.sbatch`，结果写入新作业目录。
要求官方源代码和模型包已按 README 解压到 `external/trumans_utils`；启动脚本只在
独立 `external/trumans-runtime/deps` 安装补充依赖，不改动 Isaac 容器。
脚本修改该独立运行目录的采样配置，因此同一目录不能并行运行多个生成作业。

## 验证

15 项 CPU 测试及实际 GLB 的 `check_retarget.py` 检查通过：静止蒙皮、绑定帧校正、
水平位移保持。对实际动作抽查 20 帧，蒙皮最低点范围约 -1.74 cm 至 +1.37 cm，
髋部高度缩放为 0.94539。尚未证明脚步完全无滑动。
机器人保持原 Phase 1 摆臂；人体只有旧版躯干和手掌碰撞代理，不报告全身安全距离。

GPU 录像作业 `744065` 完成，退出码 `0:0`，五项集成检查全部通过。
源缓存关节与 FK 最大位置误差约 `2.84e-7 m`，重定向旋转矩阵最大分量误差约 `5.55e-16`。
已检查中段与结尾渲染画面。
[真实 TRUMANS 带蒙皮视频](../outputs/isaac_human/kitchen_744065/preview.mp4) ·
[结果](../outputs/isaac_human/kitchen_744065/result.json)。
