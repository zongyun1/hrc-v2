# HARPv2 进度

> 活文档，持续更新。环境、资产与流水线的使用方法见 [../README.md](../README.md)；
> 详细可行性数据见 [phase0-feasibility-report.md](phase0-feasibility-report.md)。
> 最近更新：2026-09-08

---

## 当前状态：Phase 1 进行中 —— 人已经真正进入 Genesis

Phase 0 全部通过后，Phase 1 的第一刀切在**最大的未验证风险**：把 TRUMANS 生成的
SMPL-X 运动变成 Genesis 里一个会碰撞的物理实体（此前「人」只存在于 pyrender 视频里）。
这一环已打通并量化验证。

```
RoboCasa MJCF ──→ Genesis（场景 + 机器人物理/IK）
      └──→ 体素化 occupancy ──→ TRUMANS ──→ MotionSequence(schema v1.0)
                                                  └──→ SMPL-X 刚体骨架 ──→ Genesis 实体
```

---

## 基础设施

| 项 | 状态 |
|---|---|
| Git | 已初始化（1 commit：初始提交）|
| Genesis | submodule `./genesis`（官方 main）|
| Genesis env | uv + `genesis/.venv` (Py3.12) + torch 2.14 cu126；GPU(4070) 可用，WSLg viewer 可开 |
| robosuite/RoboCasa | `third_party/robocasa-venv` (Py3.11)，robosuite 1.5.2 + robocasa 源码，23G 资产已下 |
| TRUMANS | `third_party/trumans-venv` (Py3.10) + `trumans_utils` + 用户提供 `trumans_demo.zip`（权重+SMPL-X）|
| 代码骨架 | `hpmm/`（`hpmm_motion` / `hpmm_sim` / `hpmm_data` / `hpmm_eval`）+ `hpmm/tests` |

`third_party/` 全部 gitignore。跑测试：`cd genesis && PYTHONPATH=../hpmm uv run python -m pytest ../hpmm/tests -q`。

---

## Phase 0 可行性体检 — 全部通过

| # | 任务 | 结论 |
|---|---|---|
| 0.1 | 现成 RoboCasa365 Genesis pack | ❌ No-Go：必须自写 exporter |
| 0.2 | MJCF 导出 → Genesis 导入 | ✅ layout1：171 links / 519 geoms / 59 dof，铰接+位姿+材质正确 |
| 0.4-A | TRUMANS 生成行走 | ✅ 465 帧，脚贴地、身高对、跟随轨迹 |
| 0.4-B | 厨房 zero-shot + 视频 | ✅ 穿透均值 0.018%、0 帧 >1%；有渲染视频 |
| 0.5 | TRUMANS occupancy 机制 | ✅ 预烘焙全局栅格 + 分段裁 32³ 局部 patch |
| 0.6 | Genesis kinematic 碰撞方向性 | ✅ 每帧覆写「挡机器人不被推走」（软接触 ~20mm 穿透）|
| 0.3 | MuJoCo demo replay | ⏭️ 跳过（计划已定不 replay）|

---

## Phase 1 已完成部分

### ★ `io/schema.py` 冻结（v1.0）

两条腿之间唯一的契约（计划的接口②）。约定：

- **batch-first**：所有序列量 `[B, T, ...]`，单条序列也是 `B=1`。
- **世界系即场景系**：Z-up、米、RoboCasa/Genesis 世界坐标。生成器自己的坐标系
  （TRUMANS 是 Y-up 且以场景为中心）在 backend 内部转换完，下游永远看不到。
- **旋转一律 axis-angle**，SMPL-X 顺序；`transl` 是 SMPL-X 自己那个 `transl`。
- **只依赖 numpy**：三个 venv 里都能 import。
- 存 `.npz`，`meta` 以 JSON 串内嵌，文件自描述。

不变量 #2（`hpmm_motion` 不许 import genesis/hpmm_sim）用 AST 静态扫描强制，
不依赖运行时 import，因此在没装 Genesis 的 TRUMANS venv 里也成立。

### ★ 人 → Genesis 桥接（本阶段最大风险，已解除）

`hpmm_sim/human/skeleton.py` + `driver.py`：

- 用 **MJCF ball joint** 而非 N 个自由胶囊。Genesis 的 MJCF parser 支持
  `mjJNT_BALL → JOINT_TYPE.SPHERICAL`，所以 SMPL-X 的关节旋转能以**四元数直接灌进
  `set_qpos`** —— 无需 axis-angle→euler 分解、无万向锁、肢体也不会在求解器里散架。
- 结构：pelvis 带 `freejoint`（7 qpos）+ 其余 21 个关节各一个 ball joint（4 qpos）
  = **91 qpos / 69 dof / 22 link / 26 胶囊**。
- 骨架模板由 `tools/extract_smplx_skeleton.py` 从 SMPL-X 模型蒸馏成一个可提交的 JSON，
  **Genesis 端因此不需要 smplx 包，也不需要那 100MB 模型文件**。
  肢体粗细不是手调的：按 LBS 权重把顶点归到主导关节，取到骨轴垂距的 75 分位数
  （投影落在骨段外的点直接丢弃，否则手掌会把小臂撑到 0.14 m）。
- 自碰撞用 `contype=1 / conaffinity=2` 过滤（Genesis 的 MJCF 碰撞过滤是 entity 内部
  局部的，所以不影响人-厨房碰撞）；直接写掩码而不用 `<contact><exclude>`，
  可避开 Genesis 的掩码重解算。

**验证结果**（`tools/verify_human_bridge.py`）：

| 指标 | 结果 |
|---|---|
| 姿态传递误差（Genesis link 原点 vs SMPL-X 关节） | 均值 **0.000 mm** / 最大 **0.001 mm** |
| 每帧覆写下的累积漂移（50 个覆写周期） | **0.000 mm** |
| 单步内位移（覆写之间的瞬时偏离，不累积） | 3.2 mm |
| 不覆写、自由下落 50 步 | 1089 mm ← 说明覆写是必需的，不是优化 |
| 总质量 | **70.0 kg** |

质量要专门解一下密度：胶囊按解剖学重叠（骨盆处三根共用），体积重复计数约
**5.2 倍**（0.368 m³ vs 真人 ~0.07 m³），所以密度由目标体重反解得 190 kg/m³。
人是运动学驱动的，自身惯量不影响运动，但接触力是 Phase 3 的安全信号，体重得是真的。

### ★ 真人网格：胶囊只留碰撞，表面用 SMPL-X 蒙皮

胶囊是**物理引擎看到的东西**，但在视频里读起来是一堆球。现在表面换成真正的 SMPL-X 网格
（10475 顶点 / 20908 面），胶囊改为 `visualization=False` —— 它们继续干碰撞的活，只是不再被画出来。

网格由**骨架自身的 link 位姿做线性混合蒙皮**驱动，逐帧用 `set_vverts` 覆写世界系顶点
（morph 需 `enable_custom_vverts=True`）：

    v_world = Σ_j w_ij · [ R_j (v_rest − rest_joint_j) + p_j ]

这正是 SMPL 的 LBS，利用了「静止姿态下每个 link 坐标系都是到其关节的纯平移」这一事实。

这样绕开了两条更贵的路：**在 Genesis 环境里引入 smplx**（外加 100 MB 受许可限制的模型文件），
或**存顶点轨迹**（10475×3×T，每段几十 MB）。省略的只有 SMPL-X 的姿态修正 blendshape，
肘膝处褶皱略少，视频尺度看不出来。

蒙皮资产 `hpmm/assets/smplx_skin_male.npz` 仅 **380 KB**（静止顶点 + 面 + 每顶点 top-4 权重，
手指/面部关节按 `fold` 归到最近的身体祖先），已在 `.gitignore` 加例外让它进版本库。

**验证**：换皮前后物理数值**逐项一致** —— 人机最近 0.462 m、1270/1270 步地面接触、自碰撞 0，
说明只换了外观没动仿真。另有单测钉住「静止 link 位姿必须还原出原网格」（误差 < 1e-6 m），
这条如果绑定坐标系搞错，误差会是米级而不是微米级。

### ★ TRUMANS 采样质量：约 40% 抽样是废的

`tools/probe_trumans_variance.py`，同一条厨房轨迹抽 5 次：

| seed | float% | 中位身高 | 判定 |
|---|---|---|---|
| 0 | 0.0 | 1.753 | ✅ |
| 1 | 27.3 | 1.351 | ❌ 蹲伏 + 悬空 |
| 2 | 0.0 | 1.749 | ✅ |
| 3 | 0.0 | 1.711 | ✅ |
| 4 | 17.1 | 1.633 | ❌ |

采样器无 seed 且随机，失败模式是**身体蹲伏前倾并整体悬空半米**。已排除
「action_id 泄漏」（`act_type='none'` 确实把 action_label 清零）和「占据栅格把人抬高」
（行走线上只有 1 层地板体素）—— 就是扩散采样方差。
→ 导出器内置**拒绝采样门控**（`float_frac ≤ 5%` 且 `中位身高 ≥ 1.65 m`）+ 可复现 seed。
这是计划里三层避让中「拒绝采样兜底」的第一个离线实例。

另外发现一个**系统性的 −0.088 m 垂直偏置**（TRUMANS 把脚底放在自己地板面以下），
导出时按中位脚底高度整体平移到 z=0，偏移量记录在 `meta`。

### ★ 运动时间轴必须与求解器 dt 解耦（踩到的坑）

第一版共仿真循环每次 `scene.step()`（dt=0.01）推进 `stride` 个运动帧，结果人比物理
**快 6.7 倍**。视频上完全看不出问题（只是走得快），但接触力、接近速度这类指标全废，
Phase 2 的 receding-horizon 共仿真会直接崩。

→ `HumanDriver.set_time(seq, t)`：按**仿真时间**取帧（`int(t * seq.fps)`，两端 clamp），
运动 fps 与 dt 从此无关。已加回归测试（100 Hz 步进遍历 30 fps 序列必须按序、不重不漏）。

顺带一个 Genesis 录像的坑：`start_recording(fps=30)` 会把渲染帧**按仿真时钟抽稀**，
100 Hz 步进下请求 30 fps 会三帧丢二（实际输出 100/3 fps）。录像 fps 应设成 `1/dt`。

### ★ 可操作物体：用**任务 env** 导出，而不是基础 Kitchen env

Phase 0 导出的是基础 `Kitchen` env —— 那是个**空厨房外壳**，0 个自由刚体。
robosuite 的**任务 env** 才会把物体 model 并进 arena。实测 `PickPlaceCounterToCabinet`
（layout1/style1/seed0）：

- `env.model.get_xml()` → 379 KB / 216 meshes / **3 个自由刚体**
  （`obj_main` + `distr_counter_main` + `distr_cab_main`），自由关节在 qpos[59:80]。
- 语言指令：`"Pick the sugar cube from the counter and place it in the cabinet."`
- 注册表里有 **396 个 env**（RoboCasa365 的任务全在）。

**但物体位置不在 XML 里**：RoboCasa 在 `env.reset()` 时用 placement sampler 采样，
写进 `sim.data.qpos`；XML 只有 body 的作者默认位姿。所以复现一个 episode 需要两个产物：

1. MJCF（场景 + 机器人 + 物体 model）
2. reset 后的状态 JSON（**按关节名索引**，不按下标 —— Genesis 自己排 dof 顺序，
   按位置对齐会看似能跑但把场景悄悄搞乱）

→ `tools/export_robocasa_task.py`（robocasa venv）+ `hpmm_sim/shim/sampler.py`
（Genesis 侧 `TaskInstance.apply()`）。这就是计划里 `shim/sampler.py`「placement sampler
重定向」——不是重写它们的采样器，是把它的**输出**重定向到 Genesis。

### ★ Genesis MJCF `align` 默认值会静默错位自由物体（真 bug，已修）

`gs.morphs.MJCF` 的 `align` 文档写 "Default to False"，但字段实际默认是 `None`，
而 `None` 的行为等同 `True` —— 浮动基座 link 会被**重定基到质心**。这改变了自由关节
`qpos` 位置的**语义**：Genesis 把**质心**放在 MuJoCo 放**body 原点**的地方，
于是几何体整体偏移一个 `body_ipos`。

最小复现（单 body + freejoint + 几何偏 0.1 m，同一个 `qpos=(1,2,3)`）：

| | 盒子中心 | 对 MuJoCo 误差 |
|---|---|---|
| MuJoCo | (1.1, 2, 3) | — |
| `align=None`（默认）| (1.0, 2, 3) | **100 mm** |
| `align=True` | (1.0, 2, 3) | 100 mm |
| `align=False` | (1.1, 2, 3) | **0 mm** |

坑在于 `link.get_pos()` **仍然**返回 body 原点，两者差一个 `ipos`，很容易误判成
"getter 约定不同"。不是 —— 几何体真的移位了。

影响范围：**只影响单体自由物体，不影响多体铰接实体**。人体 pelvis 的 `ipos` 有 20.4 mm，
但 `align` 取 None 还是 False，姿态误差都是 0.0003 mm —— 人体从来没受影响。
RoboCasa 物体受影响：落点偏 0.01 / 1.39 / 12.45 mm，**每个恰好等于自己的 `|body_ipos|`**
（正是这个吻合定位了根因）。这个量级小到像数值噪声，大到足以让 IK 抓取瞄错地方。

→ 统一收口到 `hpmm_sim/assets/load.py:mjcf()`，强制 `align=False`；
**不要再直接调 `gs.morphs.MJCF`**。已加回归测试。

修复前后（`PickPlaceCounterToCabinet` L1S1 seed0，62 个关节全匹配 0 缺失）：

| 物体 | 修复前误差 | 修复后 | 重力 100 步后 |
|---|---|---|---|
| `obj`（sugar cube）| 0.01 mm | **0.00 mm** | 稳定（0.05 mm）|
| `distr_counter` | 12.45 mm | **0.00 mm** | 稳定（0.29 mm，**修复前滑走 40 mm**）|
| `distr_cab` | 1.39 mm | **0.00 mm** | 稳定（1.69 mm）|

注意最后一列：修复坐标系顺带修了动力学 —— 那个干扰物之前滑走，是因为它本来就摆错了位置。

（排查中先后证伪了两个假设：`env.reset()` 后 `body_xpos` 陈旧 —— 实测与 `qpos` 差 0.000 mm；
以及 body 的 `pos` 属性非零 —— 三个物体都是 `pos="0 0 0"`。）

### ★ 渲染：封闭房间里相机会拍到墙外（三个视频曾全废）

**症状**：三个"交付物"视频全是一片灰墙。而且**没有任何报错** —— 编码正常、Genesis 打印
`Video saved to ...`、所有数值都对（指标来自状态查询，与相机无关）。
教训：**报视频前必须抽帧看图**，不能只看日志和数字。

**病因**：RoboCasa 房间是不透明封闭盒子且**无天花板**。相机若按"从主体偏移一段"来放，
经常落到房间外面，拍到砖墙外立面。

**房间轮廓怎么算，试错了三次，每次算出来的数字都看着合理但都是错的**：

1. 对**所有** link 位置取百分位 —— robosuite 未摆放的机器人停在 `(10,10)`，
   它那 ~20 个 link 占 layout1 的 12%，百分位挡不住。房间中心算到 (5, 4.3)，在建筑外面。
2. 对**非机器人** link 取百分位 —— 得到的是**家具**轮廓。家具全贴后墙，开阔地板消失了
   （layout1 算成 y∈[-1.5, 0]，真值是 [-3.14, 0.14]），相机被放进橱柜堆里。
3. 用 geom 包围**球** —— `geom_rbound` 对一块平薄墙板极大，轮廓被推到建筑外几米。

**正解**：RoboCasa **给外壳命过名** —— `wall_left_room_main` / `wall_front_room_main` /
`floor_room_main` 等，直接读。墙体以自身高度为中心，所以墙顶 = 2 × 中心 z。

**站位规则**：галley 厨房里"背离主体后退"没有好方向（柜台贴一面墙、主体本就在中间，
任何固定规则最终都会退进柜子）。改成 `open_floor_position()`：在轮廓内搜索**离家具最远**
的点（即开阔地板），并限制在距目标的合理站位区间内。

**更好的一条路**：`mount_camera()` 直接复现 RoboCasa 自己的 `cam_configs`
（已由 `import_robocasa_demo.py` 存进 demo JSON）。相机挂在机器人底盘上，永远对着工作区；
`robot0_agentview_center` 正是他们数据集视频用的视角，**所以我们渲的画面可与人类示教直接对比**。
MuJoCo 相机看向局部 **−Z**、**+Y** 朝上，解析成世界系 pos+lookat+up，不把 raw transform 交给 Genesis。

**顺带两条**：Genesis 默认 `ambient_light=(0.1,0.1,0.1)` 在封闭房间里太暗，
`room_vis_options()` 提到 0.35 并补一盏反向光。
另外**不能同时跑两个 Genesis 进程** —— 会触发 `Software rendering context detected`
且其中一个静默死掉；而 shell 管道以 `| tail` 结尾时退出码来自 `tail`，失败看起来像成功。

### 其它

- `hpmm_sim/assets/convert.py`：robosuite 把机器人停在 `(10,10)`，而 `robot0_base`
  在 Genesis 里是固定体，只能靠移动底盘的滑动关节挪 —— 那会把导航要用的 dof 全烧掉。
  所以**在 XML 层改基座位姿**，移动关节留给真正的导航。
- `hpmm_sim/robot/panda_omron.py`：按关节名前缀切出 arm(7)/base(4)/gripper(2)/场景铰接
  的 dof 分组，并修掉「neutral qpos 超关节限位」（落到限位中点）。
  底盘 4 dof = 前进滑动 / 侧移滑动 / 偏航铰链 / 升降柱(0–0.34 m)。

---

## ★ RoboCasa 示教数据集：已下载并打通导入

决策已定：**A 作种子喂给 B/C**。下了 `PickPlaceCounterToCabinet` 的 human/pretrain，
**只有 99 MB**（脚本预告的 "several Gb" 是针对全量的）。

格式是 **LeRobot v2.1**：108 条 episode / 24225 帧 / **20 Hz**，`robot_type=PandaOmron`。
但真正有用的不是 parquet，而是每条 episode 的 `extras/` 目录：

| 文件 | 内容 |
|---|---|
| `model.xml.gz` | 该 episode 的**精确 MuJoCo 模型**（638 KB 解压后）|
| `states.npz` | 完整状态轨迹，`[time] + qpos + qvel` |
| `ep_meta.json` | layout/style、语言指令、物体配置、机器人初始基座位姿 |

`modality.json` 还说明 `observation.state`(16) 里 **EEF 位姿是直接给的**
（`end_effector_position_relative` [7:10] + `rotation` [10:14]）—— 但那是**相对移动底盘**的，
一个需要先被移动的底盘反旋转才有意义的任务空间参考，是个容易被误用的参考。
所以改用**对录制 qpos 跑 MuJoCo FK**，直接拿 grip site 的**世界系**位姿。

三个必须处理的坑：

1. **资产路径指向录制机器**（`/root/robocasa/...`、`/opt/conda/envs/robocasa/...`）。
   按 `robocasa`/`robosuite` 的 assets 标记切分后重挂到本仓库，这样不依赖对方的目录布局。
2. **`states` 是 robosuite 的扁平 `MjSimState`**，必须用编译后模型自己的 `nq`/`nv` 切分，
   不能猜。实测 episode 0：nq=96 / nv=93，正好 `1+96+93=190` 对上。
3. **材质属性越界会让 Genesis 算出复数**。RoboCasa layout 34 的凳子材质写了
   `shininess="-1"`，另有 22 个材质 `specular` 是 2.0/3.0（MuJoCo 合法范围都是 [0,1]，
   但 MuJoCo 容忍）。Genesis 算 `glossiness = shininess * 128 = -128`，再算
   `roughness = (2/(glossiness+2))**0.25 = (-0.015873)**0.25` —— **Python 里负数开四次方返回复数**，
   于是构建挂在 `Invalid attribute 'color[0]' ... Got (0.2509+0.2509j)`，
   而这条报错**不指向任何材质，也完全指不到病因**。
   （`0.015873**0.25 × cos(π/4) = 0.25098`，与报错值逐位吻合 —— 这才是定位它的依据。）
   → `assets/convert.py:sanitize_materials()` 钳位到合法区间；本次修了 6 个 `shininess` +
   16 个 `specular`。测试把这段算术本身也钉住了，免得日后重构掉了理由。

→ `tools/import_robocasa_demo.py`，输出与 `shim/sampler.TaskInstance` 同构的 JSON
（所以 Genesis 侧复用同一条加载路径）+ 轨迹 npz。
实测 episode 0：layout **34** / style **35**（demo 跨多个 layout，不都是 1/1），
指令 `"Pick the cereal from the counter and place it in the cabinet."`，
232 帧 / 11.6 s，**EEF 走了 2.068 m**，z 从 1.04 升到 1.58（够进柜内）。

→ `tools/replay_demo_ik.py`：**手臂用 IK 解**（限定 7 个 arm dof，否则求解器会去"解"柜门），
**底盘/升降柱按录制值回放**（那是机器人在房间里的站位而非操作选择），
**夹爪按录制指位**（携带开合时序）。

**结果（episode 0，layout 34/style 35，"把麦片从台面拿到柜子里"）**：

| 指标 | 第一版 | 修复后 |
|---|---|---|
| EEF 跟踪误差 mean / median | 14.1 / 12.4 mm | **7.8 / 6.8 mm** |
| 20 mm 内的帧数 | 185/232 | **232/232** |
| 物体抬升 | 15 mm（**没抓起来**）| **554.7 mm（抓起并放入柜中）** |
| 干扰物扰动 | — | 0.1 / 0.2 mm |

两个失败原因都在机制层面，不是调参：

1. **夹爪抓不住**：记录的"闭合"指位是 38 mm —— 那正是**麦片盒的宽度**，是手指被物体挡住的位置。
   把手指位置控制到这个值，控制器一到位误差归零，**夹持力为零**。
   改成闭合时**压过物体**（命令到 0），由接触把手指停在物体处，力由 force range 限住。
   夹持力设 **20 N**（robosuite 自己 finger actuator 的 `forcerange` 就是 ±20 N）——
   麦片盒只有 65 g，摩擦 0.8 下只需 0.32 N 切向力，20 N 绰绰有余。
   （先用 100 N 验证机制成立，再降到 20 N 物理值复验：**依然抓起**，抬升 575.6 mm、
   跟踪误差完全一致 —— 所以复现成立于机器人自身的夹爪能力范围内，
   不是靠一个它并不具备的超能力夹爪。）
2. **14 mm 跟踪误差**：与"参考每帧移动 9 mm"同量级，说明控制器滞后约一帧。
   增益 kp 800→3000 后降到 7.8 mm。

归因是干净的：手臂关节滞后 mean **0.0182 rad**，Panda 连杆约 0.4 m，
`0.0182 × 0.4 ≈ 7.3 mm` ≈ 观测到的 7.8 mm —— **残差几乎全是 PD 滞后，IK 本身基本精确**。
要再压就得提高子步频率或做前馈，不是 IK 的问题。

关于"不 replay MuJoCo demo"：那条反对的是 replay **action** —— action 序列跨引擎无意义，
它要由特定控制器、特定增益、特定接触模型来解释。**任务空间**的 EEF 路径是另一回事，
它是任务的属性而非控制器的属性，用 Genesis 自己的 IK 跟踪是真正的重生成，
也正是"同一 episode 换个物体位置、或者旁边走过一个人"还能重跑的前提。

---

## 已决策：IK 的 reference trajectory 从哪来

IK 只是求解器（给定 EEF 目标位姿求关节角），**不产生目标**。要在 Genesis 里复现一个
RoboCasa 任务，reference 必须另有来源。三条路：

| | 来源 | 需要下载数据集 | 抓取位姿 | 可规模化 |
|---|---|---|---|---|
| A | RoboCasa 人类/MimicGen 示教的 **EEF 位姿轨迹** | 是（`scripts/download_datasets.py`）| 白拿 | 否（固定几十条）|
| B | 由**物体位姿 + fixture site** 合成关键帧（`robot/gen.py`）| 否 | **需自己合成** | 是 |
| C | Genesis 原生规划器（`plan_path()` + `RRT`/`RRTConnect`）在关键帧间做无碰撞规划 | 否 | 同 B | 是 |

两点澄清：

- **「不 replay MuJoCo demo」反对的是 replay *action*，不是反对拿示教轨迹当参考。**
  跨引擎 replay action 一定崩（controller / gain / 接触模型都不同），但 replay
  **EEF 路径**再用 IK 跟踪是稳的 —— 那是任务空间的量，与控制器无关。这个区分建议写回计划。
- **纯 B 绕不开抓取位姿从哪来**。RoboCasa 物体不带抓取标注。而 MimicGen 的真正设计
  恰恰是「拿一条人类示教的 object-centric 子轨迹，变换到新物体位姿上」—— 抓取姿态
  是从示教里白拿的。

**已定：A 作为种子喂给 B/C**，而非二选一。每任务下几十条示教做 seed，抓取姿态与运动
风格从示教来，物体位姿变了做 object-centric 变换，再用 IK/RRT 在 Genesis 落地。
纯 B 只在愿意自己做 antipodal 抓取合成时才成立。

Genesis 侧能力已确认齐备：`RigidEntity.inverse_kinematics` / `inverse_kinematics_multilink`
/ `plan_path`，`genesis/utils/path_planning.py` 提供 `RRT` 与 `RRTConnect`。

---

## 追加验证（Phase 0 之后）

- **TRUMANS 动作生成逻辑**：走路/其他动作共用同一条件扩散模型，靠 `action_label` one-hot 区分；scene occupancy 是网络输入（真被环境 condition，但**软引导**；goal waypoint 才是硬约束）。→ 印证计划 Phase 2 三层避让必要性。**暂定只用纯走路。**
- **TRUMANS 返回的 vertices 比返回的参数晚一个 Adam step**（`optimize_smpl` 在
  `optimizer.step()` 之前存 `vertices_output`），差异均值 6.6 mm / 最大 166 mm。
  所以顶点一律**由参数重算**，不用它自带的。
- **机器人 simulation**：PandaOmron 在 Genesis 可 PD 控制（EEF 移 0.34m、抽屉能开、场景 0mm 漂移）；robosuite 任务逻辑层不迁移，需 shim 层重写。
- **场景泛化**：layout 1/2/4/6 均能导出；导入路径通用（layout1 已完整验证）。
- **铰接件质量**（用 MuJoCo 编译后读 `body_mass` 实测，非估算）：RoboCasa 把橱柜门碰撞盒
  写成 `density="10"`（MuJoCo 默认 1000，木板约 600–750）。20 扇橱柜门 min/中位/max =
  **0.031 / 0.076 / 17.99 kg**；家电门（冰箱/烤箱/洗碗机）10.5–28.3 kg 属正常；抽屉 12.1 kg。
  固定台面/柜体 0.14–1.0 kg 但无关节，无所谓。这更像 RoboCasa 有意的仿真调参
  （轻门好开、冲量小）而非坏资产 —— **但 31 克的柜门撞到人等于没撞**，
  在把接触力当 Phase 3 安全信号之前要专门决定怎么处理。

---

## 关键决定 / 约定

- **暂用纯走路**作为人体运动方法（手部交互动作需补 hand sampler，延后）。
- **不需要批量场景同时运行** → 凸分解 build 时间不是阻塞项。
- 机器人轨迹在 Genesis 里用 IK 重生成，不 replay robosuite（沿用计划既定）。
- 人体表示 = **ball-joint 铰接刚体骨架**（非 N 个自由体，非物理跟踪）。
- schema 保持 batch-first；`n_envs=0` 的非批处理场景在 **driver 边界**处压掉 batch 维，
  不往上游传染。

---

## 待解工程点（按当前优先级）

1. robosuite 任务逻辑层在 Genesis 重写（shim 层：site / predicate / sampler）—— **关键路径**。
2. 机器人 IK：`hpmm_sim/robot/ik.py`，跑通 3–5 个 atomic task。
3. A 线人体：occupancy 动态 overlay + batched local crop + A* waypoint + clearance guidance。
4. 手部交互动作补 hand sampler（延后）。

---

## 产出物

- **代码** `hpmm/`：`hpmm_motion/io/schema.py`（冻结契约）、`hpmm_sim/human/{skeleton,driver}.py`、
  `hpmm_sim/assets/{convert,load}.py`、`hpmm_sim/robot/panda_omron.py`、
  `hpmm_sim/shim/sampler.py`、`hpmm_sim/human/skin.py`、`hpmm_sim/vis/camera.py`、
  `hpmm/tests/`（25 项通过）。
- **资产** `hpmm/assets/`：`smplx_skeleton_male.json`（骨架模板，22 关节 / 26 骨 / 身高 1.794 m）、
  `smplx_skin_male.npz`（蒙皮，10475 顶点 / 20908 面 / top-4 权重，380 KB）。
- **工具** `tools/`（12 个，均为仍在用的流水线环节；Phase 0 的一次性验证脚本已随其被取代而删除）：
  `export_robocasa_{mjcf,task}.py`、`voxelize_kitchen_for_trumans.py`、`extract_smplx_skeleton.py`、
  `export_trumans_motion.py`、`probe_trumans_variance.py`、`import_robocasa_demo.py`、
  `replay_demo_ik.py`、`phase1_cosim_demo.py`、`probe_camera.py`、
  `verify_human_bridge.py`、`verify_task_instance.py`。
  命名约定：`verify_*` 是需要 GPU 和已构建场景的验证程序，`hpmm/tests/` 才是 pytest 单测。
- **视频** `exports/`：`demo_ik.mp4`（RoboCasa sample 的 IK 复现，用 RoboCasa 自己的
  `robot0_agentview_center` 机载相机）、`phase1_cosim.mp4`（人 + 机器人同场景，Phase 1 交付物）。
  （`human_in_kitchen.mp4` 已删：内容被 `phase1_cosim.mp4` 完全覆盖；
  `tools/verify_human_bridge.py --video` 随时可再出。）
- **流水线输入** `exports/`（脚本依赖，勿删）：`kitchen_transform.json`（RC↔TRUMANS 重映射）、
  `motion_kitchen_walk.npz`（通过门控的人体运动）、`robocasa_layout1_style1.xml`（基础厨房）、
  `task_PickPlaceCounterToCabinet_L1S1_seed0.{xml,json}`（任务实例）、
  `demo_PickPlaceCounterToCabinet_ep000.{xml,json}` + `_traj.npz`（示教场景 + 参考轨迹）。
  已清掉 Phase 0 的过时产物（pyrender 视频、场景静图、layout2/4/6 导出、以及每次运行都会
  重新生成的 `human_smplx.xml` / `*_placed.xml`）——2.7 MB。
- **文档** `docs/`：本文件 + `phase0-feasibility-report.md`。

---

## 变更日志

- **2026-09-08**：Phase 1 启动。冻结 `io/schema.py`；建 `hpmm/` 骨架 + 解耦不变量测试；
  打通 SMPL-X → Genesis 刚体骨架（姿态误差 0.000 mm、累积漂移 0 mm、70 kg）；
  量化 TRUMANS 采样方差（~40% 废）并加拒绝采样门控 + 地面对齐；
  机器人摆进厨房；修掉运动时间轴与 dt 耦合的 bug；产出人 + 机器人同场景的 Phase 1 交付物视频
  （12.7 s / 1270 帧，人机最近 0.462 m，人体自碰撞 0）；打通 RoboCasa 任务实例导出
  （含物体 + placement）→ Genesis 复现；定位并修复 Genesis MJCF `align` 默认值导致的
  自由物体静默错位；实测更正柜门质量数据。
- **2026-09-07**：完成环境搭建 + Genesis submodule；跑完 Phase 0 全部子项；追加验证 TRUMANS 动作逻辑、机器人可控性、场景泛化；建立本进度文档。
