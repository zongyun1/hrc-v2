# Phase 0 可行性体检报告（进行中）

> 状态：0.1 / 0.5 / 0.6 已完成（2026-09-07）；0.2 / 0.3 / 0.4 待做

---

## 0.6 — Genesis 外部驱动 body 的碰撞方向性 ✅ PASS

**问题**：能否做到"人挡得住机器人，但不被机器人推走"（运动学驱动的人体 vs rigid solver 机器人）。

**方法**：`tools/test_kinematic_collision.py`。倾斜重力（-x）把一个动态"机器人"box 持续压向"人"box，headless 跑 400 步，三个变体对比。

| 变体 | 人 max 漂移 | 机器人被挡 | 接触力 | 结论 |
|---|---|---|---|---|
| `fixed`（人=固定基座） | 0.000 mm | ✅ 停在 0.33（边界 0.35） | 4292 N | PASS（基线） |
| `override`（自由体 + 1e6 kg + 每帧 `set_pos(zero_velocity=True)`） | **0.582 mm** | ✅ 停在 0.33 | 4217 N | **PASS** |
| `heavy_nooverride`（仅大质量、不覆写） | 24286 mm（飞走） | ❌ | 5203 N | FAIL |

**结论**：
- 计划里的缓解方案（"极大质量 + 每帧强制覆写 state"）**成立**，且关键在"每帧覆写"——大质量本身对持续外力（如重力）无效（重力加速度与质量无关），必须每帧 `set_pos(..., zero_velocity=True)` 把人拉回目标位并清零速度。
- 覆写方案下人几乎不动（0.58mm，来自一步内的积分残差），同时 rigid solver 仍生成接触、正常挡住机器人。方向性符合预期。

**给后续阶段的提醒**：
- Genesis 默认软接触，观测到约 **20mm 穿透深度**（4200N 挤压下）。这直接关系到 Phase 3 "用穿透深度/接触力作安全违规信号"——阈值要按这个软度标定；若要更硬的接触需调 solver 参数。
- 人体骨架驱动应统一走 `set_qpos(..., zero_velocity=True)` 每帧覆写（对应 `hpmm_sim/human/driver.py`）。

---

## 0.1 — RoboVerse/MetaSim 是否已有 RoboCasa365 的 Genesis pack ❌ No-Go（省不掉）

**结论**：**没有现成的，省不下这两周，必须自己写 RoboCasa(robosuite MJCF) → Genesis 导出/转换管线。**

证据要点：
1. MetaSim 有 Genesis backend（`packages/metasim/.../sim/genesis/genesis.py`，672 行，`pip install -e ".[genesis]"`），成熟度为"能跑基础刚体/机械臂任务，但不完整"——源码里有多处 `NotImplementedError`、非均匀缩放静默降级（line 168-169）、state/observation 的 `TODO`。
2. RoboVerse/MetaSim **完全没有** RoboCasa/RoboCasa365 资产（两仓库 grep `robocasa` = 0 命中；论文 16 个 source benchmark 无 RoboCasa；列表里的 "RoboSuite & MimicGen" 只是 robosuite 少量基础任务，≠ RoboCasa 的 365 厨房场景）。
3. 全网无任何 RoboCasa→Genesis 公开移植（Genesis 官方 examples、RoboCasa 官方、RoboVerse 均无）。旁证：ManiSkill(SAPIEN) 侧有人尝试导入 RoboCasa（issue #1075，仍有失败），可作参考但不能直接用。

**建议路线**：
- 走 `gs.morphs.MJCF(...)` 直接喂 MJCF，不走 URDF 中转；先让 robosuite 把 1 个 layout dump 成完整 MJCF。
- 重点排查转换损耗（优先级）：articulation 关节 axis/range/父子链 → 碰撞凸分解 → 非均匀 scale（需 bake 进 mesh）→ 材质/纹理。
- 可读 MetaSim 的 `genesis.py` 借鉴其 object loading 逻辑。
- 范围控制：先跑通 1 layout + 1 task 端到端，再批量。

---

## 0.5 — TRUMANS occupancy 是预烘焙还是逐帧 ✅（混合，改造难度中低）

**结论**：三层结构的"**预烘焙全局栅格 + 分段局部裁剪**"，不是纯静态也不是逐去噪步动态：
1. **全局层（预烘焙、全程复用）**：场景离线体素化成 `(300,100,400)` bool 栅格 `.npy`，启动一次性加载，运行时只做坐标→索引查表，不重算。范围硬编码 x∈[-3,3]/y∈[0,2]/z∈[-4,4]。
2. **局部层（每个运动分段 crop 一次）**：网络实际输入是以当前 pelvis 为中心裁的 `32³` patch（bbox `[-0.6,0.6]×[0,1.2]×[-0.6,0.6]`）。
3. **去噪循环内（常量）**：单次 `p_sample_loop`（100 步 DDPM）里 occ 只算一次、当常量透传，逐去噪步不更新。

即"动态性"粒度 = **每个 autoregressive 分段（~32 帧）重算一次局部 patch**。

仓库：`github.com/jnnan/trumans_utils`；关键文件 `sample_hsi.py`、`datasets/trumans.py`（`get_occ_for_points` / `add_object_points`）、`models/synhsi.py`（`p_sample_loop`）。

**动态 overlay（机器人 swept volume）改造点评估——中低难度**：
- 已有理想挂钩 `add_object_points`：把机器人 swept volume 采样成世界坐标点云，像 `object_occ` 一样叠加进 `occ`。**关键坑**：`self.scene_occ` 是持久张量，叠加前必须 `occ.clone()`，否则污染全局栅格。
- 主要工作量：现 occ 是"每分段一个 patch"、分段内 32 帧共享同一 occ。要做到"逐帧动态障碍"需把 `p_sample_loop` 的 occ 从单 patch 扩成逐帧 patch `[B, seq_len, 32³]`，并按帧叠加当时的机器人体积——可能要动 `Unet` 的 occ 输入维度（最重一项，视模型而定）。
- 对齐注意：y-up 坐标系；swept volume 必须落在硬编码范围内（越界=视为已占据）；`add_object_points` 只做 xz 平移无旋转，机器人有朝向需先做完整 SE(3) 变换。

---

## 0.2 — 导出 1 个 layout MJCF → 导入 Genesis 目视检查 ✅ PASS

**环境**：robosuite 1.5.2 + robocasa（源码 editable，独立 `third_party/robocasa-venv` Python 3.11，资产 23G 已下），与 Genesis 环境解耦。

**导出**（`tools/export_robocasa_mjcf.py`，robocasa venv）：`Kitchen` env, layout=1, style=1, robot=PandaOmron，`env.model.get_xml()` → `exports/robocasa_layout1_style1.xml`（354KB）。
- 173 meshes / 69 textures，**所有 file 引用是绝对路径**（241 处）→ Genesis 可从任意位置解析，无需重写路径。
- 59 joints = **18 hinge（柜门/家电门）+ 21 slide（抽屉）**，154 bodies → articulation 完整保留。

**导入**（`tools/import_robocasa_to_genesis.py` / `render_robocasa.py`，Genesis venv）：`gs.morphs.MJCF(file=...)` 加载成功。
- Genesis 侧：**links=171, geoms=519, n_dofs=59**（dof 数与 XML joint 数一致），build + step 正常。
- 目视（`exports/robocasa_{iso,overview,top}.png`）：galley 厨房，深色上下柜（门/抽屉**闭合且齐平**）、浅色台面、水槽、洗碗机、灶台+烤箱、壁挂微波炉、抽油烟机、砖墙 backsplash。**初始位姿正确、材质/纹理正常、几何完整**。

**Go/No-Go 判据（铰接方向、初始位姿正确）满足。**

**观测到的坑 / 提醒**（写进转换脚本时要处理）：
1. **凸分解耗时**：519 geoms 的 coacd 凸分解首次 build 约 2.5–4 分钟（Genesis 有缓存，二次更快）。批量生成前需评估。
2. **柜门质量可疑**：多个 `*_door_main` link 质量 0.03kg vs 几何估计 4.6kg（RoboCasa 原作者设的超轻质量）→ Genesis 警告。会影响门的动力学，Phase 1 驱动家电时要修正质量。
3. **PandaOmron neutral qpos 超关节限位** + neutral 自碰撞（已自动过滤 9 对）。机器人在 Genesis 里重新生成轨迹时要设合理初始位姿。
4. **约束 solver timeconst 被从 0.001 改到 0.02**（<2*substep_dt）。软接触，与 0.6 观测的 ~20mm 穿透一致。
5. **场景不在原点**：厨房中心约 (2.67, -0.38, 1.09)，galley 贴 y≈0 后墙，房间向 -y 展开到约 y=-3.2；**机器人默认停在 (10,10)**；另有一批 2cm 的 marker geom 在 z≈10.46（RoboCasa placement-sample 站点）——相机取景/体素化时要用鲁棒边界排除这些离群点。
6. **房间是封闭墙、无天花板**：demo 靠 `EnclosingWallRenderWrapper` 把墙设半透明才能从外看进去；离线渲染需把相机放房间内部或从顶部俯视。

**尚未验证**：铰接**运动方向**（开门是否朝正确方向摆）静态渲染看不出，建议 0.3 里设一个门/抽屉的 qpos 再渲染确认。

---

## 0.4 — TRUMANS 采样，人在场景里走 · Milestone A ✅ PASS（demo 场景）

**环境**：独立 `third_party/trumans-venv`（Python 3.10）。代码 `jnnan/trumans_utils`，权重/数据来自用户提供的 `trumans_demo.zip`（含 `SMPLX_MALE.npz`，无 license 阻塞）。
- requirements 钉的 `torch==1.11+cu113` **在 4070(sm_89) 上跑不了**，改用 `torch 2.14+cu130`（CUDA 正常）。
- 唯一版本坑：`vit-pytorch` 必须 **1.4.4**（新版本 cls_token/pos_embedding 少一维，checkpoint 加载 size mismatch）。

**跑法**：`run_trumans_walk.py` 脚本化一条直线轨迹（沿 z −3→3，x=0，y=0），调 `sample_wrapper(trajectory, obj_locs={})`，`action_type='none'`（纯 locomotion），demo 场景 `background`。

**结果**（465 帧 SMPL-X，`walk_vertices.npy`；图 `exports/trumans_walk_{side,top}.png`）：
- **脚不飘**：每帧最低点 y ∈ [−0.045, −0.009]（均值 −0.024）→ 脚贴地。
- **身高正常**：头顶 y ~1.70–1.74m，身高 1.74m。
- **跟随轨迹**：质心 z −2.94→2.99（要求 −3→3），x 稳定 ~0，水平路径 6.59m。
- 侧视图 12 帧叠加：清晰的人形从左走到右，脚落在地面线上，姿态自然。

**结论**：TRUMANS pipeline 在本机跑通、权重加载正确、生成可用质量的行走。**模型本身不是风险。**

## 0.4 · Milestone B — 厨房几何 zero-shot ✅ PASS（0.4 核心风险解除）

**体素化**（`tools/voxelize_kitchen_for_trumans.py`）：把 RoboCasa 厨房 geom 的世界 AABB 填进 TRUMANS `(300,100,400)` 栅格。坐标重映射（右手、循环、无镜像）：`Tx=RCy-ycen, Ty=RCz(高度), Tz=RCx-xcen`（xcen=2.75, ycen=-1.5，存于 `exports/kitchen_transform.json`）。楼层层按约定置满。occupied frac **0.133**（与 demo 0.13 一致）。占据切片图 `exports/kitchen_occ_slices.png` 确认：矩形房间墙 + galley counter 实心条 + 前方开放可走地面 + sink 缺口。

**Zero-shot 采样**（`run_trumans_kitchen.py`，scene_name=kitchen，沿开放地面走 Tx=-0.2 / Tz -2.3→2.3）：381 帧全 SMPL-X。
- 跟随轨迹（质心 Tz -2.32→2.36），脚贴地（不飘）。
- **穿透检测**（人体顶点落入非楼层占据格的比例）：均值 **0.018%**，单帧最大 0.80%，**>1% 的帧 = 0/381**。→ 人在厨房里走**不穿墙、不穿 counter、不卡台面下**。

**结论：TRUMANS 在 RoboCasa 厨房几何上 zero-shot 成立。风险登记里"TRUMANS 厨房 zero-shot 失败（高风险）"—— 解除。**

**渲染视频**（`tools/export_kitchen_mesh.py` + `render_pyrender_video.py`）：
- Genesis camera 不渲染 debug mesh（仅 viewer 可见）→ 改用 pyrender EGL 离线渲染：Genesis 导出的厨房合并网格（按 fixture 名上色）+ 逐帧人体网格（T→RC 逆变换）。
- 产物：**`exports/kitchen_human_walk.mp4`**（381 帧，人在厨房 galley 里走过 counter 前）。样帧 `exports/kw_final_30.png` 等。

---

## Phase 0 结论

| # | 任务 | 状态 |
|---|---|---|
| 0.1 | RoboCasa365 现成 Genesis pack | ✅ No-Go：必须自写 exporter |
| 0.2 | 导出 MJCF → 导入 Genesis | ✅ PASS |
| 0.3 | MuJoCo demo replay | ⏭️ 跳过（计划已定不 replay，价值低） |
| 0.4-A | TRUMANS 生成行走 | ✅ PASS |
| 0.4-B | 厨房 zero-shot + 视频 | ✅ PASS |
| 0.5 | TRUMANS occupancy 机制 | ✅ 已厘清 |
| 0.6 | Genesis kinematic 碰撞方向性 | ✅ PASS |

**Phase 0 可行性体检通过。** 两大高风险项（厨房导入、TRUMANS 厨房 zero-shot）均验证成立，人-场景解耦管线（RoboCasa MJCF → Genesis / TRUMANS 体素 → 人体运动 → 合成渲染）端到端跑通。可进入 Phase 1。

---

> **关于本报告引用的脚本**：Phase 0 的一次性验证脚本（`test_viewer.py`、
> `test_kinematic_collision.py`、`test_robot_sim.py`、`diagnose_bounds.py`、
> `import_robocasa_to_genesis.py`、`render_robocasa.py`、`export_kitchen_mesh.py`、
> `render_pyrender_video.py`、`render_kitchen_human_video.py`）在被取代后已删除。
> 它们各自的结论与数字都保留在本报告与 `progress.md` 里。取代关系：
> 运动学碰撞由真实的 22 段人体骨架桥接取代（`tools/verify_human_bridge.py`）；
> 机器人控制由真实任务复现取代（`tools/replay_demo_ik.py`）；
> pyrender 渲染路径由 Genesis 相机取代（人体现在是场景里的真实实体）；
> 场景边界诊断由 `hpmm/hpmm_sim/vis/camera.py:room_bounds()` 取代。
