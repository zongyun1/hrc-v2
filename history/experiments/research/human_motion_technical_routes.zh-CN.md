# Human motion：基于公开代码的实验路线

核查日期：2026-09-07。配套 [robot benchmark 轻量升级方案](../research/human_motion_benchmark_plan.zh-CN.md)。本次只核查公开文档和源码，未安装、下载完整数据、训练或运行仿真。

2026-09-09 补充：加入 TRUMANS motion 来源（D1），已核查项目页和官方 README；尚未下载数据或接入仿真。

目标仍是：human 作为 robot 的交互变量，动作逼真、多样、实现成本可控。本页只选可不训练 human 的公开模块，主实验不要求动作神经网络。C1–C5 是具体代码复用路线，与主文 B1–B3 的动作生成分类对应；它们可组合，但实验时应先分开验证。

## 路线总览

| ID | 基于的公开工作/代码 | 实际提供什么 | 本项目最值得测什么 | 原生平台 / Lab 适配判断 |
| --- | --- | --- | --- | --- |
| C1 | [Human-Robot Gym](https://github.com/TUMcps/human-robot-gym) | 人体动画、交互状态机、双向交接、等待与随机化 | robot 迟到、停顿、交接失败时 human 的反馈 | MuJoCo/robosuite；移植行为层预计中等工作量 |
| C2 | [Habitat 3.0 / habitat-lab](https://github.com/facebookresearch/habitat-lab) | mocap 行走、转身、停止及目标驱动伸手 | 交接点变化时全身动作的连续性和可达性 | Habitat；移植姿态生成层预计中等工作量 |
| C3 | [HandoverSim](https://github.com/NVlabs/handover-sim) + [GenH2R 环境](https://github.com/chenjy2003/genh2r) | 真实手物轨迹、释放协议、接近时暂停与接取评测 | 不同握法、速度、遮挡下 robot 能否独立接住 | PyBullet / EasySim 等原运行栈；Lab 适配预计中等 |
| C4 | [普通 Motion Matching](https://github.com/orangeduck/Motion-Matching) | 动作检索、平滑过渡、脚部 IK | 走近、停止、转身、撤回的自然度 | C++/raylib demo；提取算法预计中等 |
| C5 | [Environment-aware Motion Matching](https://github.com/UPC-ViRVIG/Environment-aware-Motion-Matching) | 根据障碍物与附近角色选择姿态及路径 | human 绕过 robot、侧身让路、改变通过姿态 | Unity/C#；保留在线反馈的 Lab 移植预计中高 |
| D1 | [TRUMANS](https://jnnan.github.io/trumans/) / [官方代码与数据入口](https://github.com/jnnan/trumans_utils) | 室内人—场景交互动作数据及动作生成方法 | 为 B1 提供全身、手部与物体同步参考，扩充室内交互动作 | SMPL-X / Blender；数据回放需骨架、坐标与物体适配 |

这里的成本是工程判断，尚未测量。没有核实到以上项目提供官方 Isaac Lab 即插即用接口。“原生 demo 可运行”和“接入本项目后可靠”是两项工作。C4/C5 本身不提供完整手物交互，必须接 B1 交互片段和接触执行层。

## C1：Human-Robot Gym 的动作库与事件状态机

**定位：优先借鉴的交互行为框架。** 原项目用于 robot 学习和评测；README 明确训练脚本为可选，因此可以只运行其人类动画、任务逻辑及手工控制器。

可读的核心入口：

- [human → robot](https://github.com/TUMcps/human-robot-gym/blob/mirror/human_robot_gym/environments/manipulation/human_robot_handover_cartesian_env.py)。
- [robot → human](https://github.com/TUMcps/human-robot-gym/blob/mirror/human_robot_gym/environments/manipulation/robot_human_handover_cartesian_env.py)。
- [公开动作资产目录](https://github.com/TUMcps/human-robot-gym/tree/mirror/human_robot_gym/models/assets/human/animations)。当前默认分支为 `mirror`。

复用其呈递、等待、撤退等阶段、动画时间调制、位置朝向随机化和状态保存。映射为本项目的 `set_goal / step / status` 接口；再接更丰富的人—物片段。只换状态机不会自动改善外观或动作自然度，视觉提升仍依赖动作数据与重定向。

**首次实验：** 一个递物任务，固定物体和机器人，将 robot 到达时机设置为正常、延迟和中途暂停，比较人是否合理等待与继续。先限制变量，确认反馈流程，再换动作与交接点。

**接触边界：** 原交接实现用人手侧 mocap body 与物体之间的 weld 管理持物；它是明确的抓持简化，不是五指摩擦抓握。要满足局部物理手主线，只复用动作和事件层，另接实际手指接触。原框架的安全控制器若会修改 robot 指令，移植评测时需统一配置并记录，不能使不同 robot 方法接受不同干预。

**Lab 新增工作：** 动作解析与骨架映射、人体显示、阶段输入输出、接触读取及 reset；不必搬入整套 robot 训练和安全控制依赖。完整物理手属于额外的共用工作，未包含在行为层迁移成本内。

## C2：Habitat 3.0 的行走与伸手控制器

**定位：无需学习的全身运动学控制底座。** [Habitat 3.0](https://aihabitat.org/habitat3/) 研究人机协作，但可单独复用其 humanoid controller，无须训练论文中的 robot policy。

主要代码是 [HumanoidRearrangeController](https://github.com/facebookresearch/habitat-lab/blob/main/habitat-lab/habitat/articulated_agent_controllers/humanoid_rearrange_controller.py)。行走从 mocap 推进帧与朝向，伸手使用预采样姿态网格的插值。`calculate_walk_pose_directional()`、`calculate_reach_pose()` 与 `_trilinear_interpolate_pose()` 是核心入口；运行时姿态查询不需要动作网络。使用现成 motion/pose 文件与重新制作这些资产是不同工作，不能据此认定资产生成流程没有额外依赖。

通过 [humanoid_actions.py](https://github.com/facebookresearch/habitat-lab/blob/main/habitat-lab/habitat/tasks/rearrange/actions/humanoid_actions.py) 借鉴关节命令和抓取阶段，把目标位置映射为行走、停下、伸手等动作。

**首次实验：** 同一人物走向多个交接点并伸手，测脚滑、身体/手腕连续性、可达范围，以及超出范围时的行为。动作数据覆盖不足的目标应拒绝或重新站位，不无限外插。

**接触边界：** [KinematicHumanoid](https://github.com/facebookresearch/habitat-lab/blob/main/habitat-lab/habitat/articulated_agents/humanoids/kinematic_humanoid.py) 明确采用运动学人体；[grasp manager](https://github.com/facebookresearch/habitat-lab/blob/main/habitat-lab/habitat/tasks/rearrange/rearrange_grasp_manager.py) 存在 snap、固定约束或持物状态更新。不能把原版 pick 当成真实手指接触；本项目需要替换这部分执行方式。

**Lab 新增工作：** 保留姿态计算，替换 Habitat/Magnum 状态和动作接口、导航输入、骨架与资产路径；先复用公开示例对应的动作资产，验证自有角色的重定向后再扩大人物范围。

## C3：HandoverSim 的真实手物轨迹 + GenH2R 的反馈

**定位：最贴近 robot 接取评测的专项路线。** [HandoverSim 论文](https://handover-sim.github.io/assets/chao_icra2022.pdf) 使用 DexYCB 手物动作与 MANO 手，提供接触释放和 benchmark；它主要是手部场景，需要额外添加全身动作。

复用 [handover_env.py](https://github.com/NVlabs/handover-sim/blob/master/handover/handover_env.py) 的接触阶段、[ycb.py](https://github.com/NVlabs/handover-sim/blob/master/handover/ycb.py) 的驱动/释放管理，以及 [benchmark_wrapper.py](https://github.com/NVlabs/handover-sim/blob/master/handover/benchmark_wrapper.py) 的释放后成功判据。

原实现让手与物体同步跟踪记录，物体有六自由度辅助驱动，并关闭 human-hand/object 碰撞；释放时撤销物体驱动。这能作为公开协议对照，但不能声称复现了真实人手摩擦。若接入局部物理手，保留手物参考和评测协议，替换持物驱动，并重新验证成功判据。

[GenH2R，CVPR 2024](https://arxiv.org/html/2401.00929v2) 增加合成轨迹与交互反馈。公开的 [env/handover_env.py](https://github.com/chenjy2003/genh2r/blob/master/env/handover_env.py) 包含 `stop_moving_dist`，robot 靠近时暂停推进手物轨迹。这部分不要求运行或训练它的 robot 网络。

**公开度限制：** GenH2R README 仍说明场景构造、演示生成和训练相关代码尚待整理，不能承诺直接调用完整的大规模轨迹生成器。可以在现有轨迹上自行实现小幅 Bézier 路径、速度和方向变化，但这是本项目新增的程序化扩增；还需约束人体可达和动作自然度。

**首次实验：** 相同握法与物体，比较轨迹持续播放和 robot 接近时暂停；再逐一改变速度、方向和遮挡。全身动画是否加入作为单独变量。MANO 与动作数据需按官方流程另行获取，不等同于仓库内已经包含全部资产。

**Lab 新增工作：** 手和物体资产、碰撞分组、驱动或局部物理手、接触区域判定、轨迹读取及释放协议。由于简化持物与旧版存在机制重叠，此路线优先贡献真实手部参考和公开评测协议，不作为主要接触创新。

## C4：普通 Motion Matching + 人—物交互片段

**定位：B2 的较轻代码实现候选。** [orangeduck/Motion-Matching](https://github.com/orangeduck/Motion-Matching) 同时有传统与 learned 实现。仅选择传统的特征距离检索、原始动作帧读取、平滑过渡和脚部 IK。

已核实 [controller.cpp](https://github.com/orangeduck/Motion-Matching/blob/main/controller.cpp) 默认 `lmm_enabled=false`，传统路径使用 [database.h](https://github.com/orangeduck/Motion-Matching/blob/main/database.h) 中的 `database_search`。**原 demo 启动仍无条件加载网络文件**；若要彻底无权重依赖，提取算法时需移除这些加载及 learned 分支，不能只关闭 UI 开关。

**首次实验：** 先做走近、转身、停止和撤回；接触任务使用独立片段。与 B1 固定片段切换相比，测转换突变、脚滑和覆盖率。持物阶段的手物特征与阶段过滤是新增设计，原代码只提供脚接触修正。

**Lab 新增工作：** 提取 C++ 检索/过渡，或重写为 Python/NumPy；接入动作数据库与骨架，替换 raylib 渲染。首轮保留简单全库搜索，数据量成为瓶颈后再优化。

代码为 MIT，附带动作来自 [LAFAN1](https://github.com/ubisoft/ubisoft-laforge-animation-dataset)，数据许可与代码不同；可替换为项目可使用的动作集。算法无须神经网络训练，但数据库预处理仍需时间。

## C5：Environment-aware Motion Matching + 空间交互

**定位：环境变化与机器人附近行为的升级路线。** [SIGGRAPH Asia 2025 官方代码](https://github.com/UPC-ViRVIG/Environment-aware-Motion-Matching) 在动作检索中加入身体占据空间与障碍代价，使姿态和路径配合周围环境与角色。

原生 Unity 项目提供 `EMM_NoDependencies` 场景，运行所需动作数据已包含；无需额外购买该论文展示用的美术包。底层依赖 [JLPM22/MotionMatching](https://github.com/JLPM22/MotionMatching)，采用预处理特征和在线搜索，不要求训练动作模型。

**首次实验：** human 穿过 robot 附近通道，在统一避让规则下改变 robot 位置与占据空间，测是否选择侧身、转向或等待，以及人机最小距离和动作自然度。robot 链节需形成几何代理；human 的反应只基于当前可用状态与固定延迟，不读取 robot 的未来动作。

**接触边界：** 论文中的持箱与姿态变化是动画/几何能力，不等同于实际手指抓持。正式手物任务仍需 B1 交互片段与同一接触层。

**Lab 新增工作：** 移植 Unity/C# 的环境特征、搜索、姿态与轨迹适配；工程量高于 C4。可先在原生场景看效果，但如果只离线导出动画，就失去了对当前 robot 的在线避让，不能把两种实验混为同一能力。

## D1：TRUMANS 作为 motion 来源

**定位：加入 B1 真实交互动作库，也可为 B2 提供候选片段。** [TRUMANS 项目页](https://jnnan.github.io/trumans/) 对应 CVPR 2024 的 *Scaling Up Dynamic Human-Scene Interaction Modeling*，提供室内全身人—场景交互记录，并提出场景与动作条件下的自回归扩散生成方法。

两种用法分别安排：

- **当前主线：数据回放与片段检索。** 使用记录的身体、手部和物体运动，不需要运行生成网络。官方 [数据说明](https://github.com/jnnan/trumans_utils#trumans-dataset) 列出 SMPL-X、动作标注、场景及物体资产；保留片段和帧索引，排除标记的错误帧。
- **可选扩展：预训练动作生成。** 官方 [demo](https://github.com/jnnan/trumans_utils#human-motion-synthesis-in-editable-indoor-scenes) 支持编辑室内布局和绘制轨迹，需另取 checkpoint、数据和 SMPL-X 模型。它依赖动作网络，单独评估，不归入 C1–C5 的非学习主实验。

**建议接入步骤（本项目待实现）：** 先选一个有完整人体与物体参考的短片段，在作者工具中预览；转换至目标仿真的坐标系和人体骨架，核对帧率、尺度、手指姿态与物体对齐，再导出项目动作片段。官方场景/物体数据使用 y-up，需与目标坐标约定核对。接触阶段与等待/释放事件由本项目校验和补齐。

**首次实验：** 先验证单片段全身与物体同步回放，再小幅调整站位、方向和播放速度，检查脚滑、穿插与手物偏移。需要机器人交互时，组合 C1 状态机；TRUMANS 片段是否覆盖目标交接动作要逐条检查。

**接触边界：** motion 参考和视觉接触不保证物理抓持成功。物体轨迹仅作参考，正式评测仍通过共用接触层执行，并检查释放后 robot 独立持有。尚未验证 Isaac Lab/RoboCasa 接入、数据下载可用性或运行效果。

## 可选手部增强：接触区域重定向

[Kinematic Motion Retargeting for Contact-Rich Anthropomorphic Manipulations，TOG 2025](https://www.andrew.cmu.edu/user/aslakshm/pdfs/TOG2025KinematicMotionRetargeting.pdf) 有 [公开 C++ 代码](https://github.com/lakshmipathyarjun6/kinematic-motion-retargeting)。它通过手表面对应关系、接触区域与 IK，把人手—物体运动重定向到不同手型，不需要训练动作网络。

可用于 B1/C1/C3 的离线手姿修正，尤其是直接复制关节角导致接触区域错位的情况。它输出运动学轨迹，不保证动力学抓握；原代码为 Maya 插件，需要 Maya、C++ 构建及目标手表面准备。由于工具链较重，只有手型差异确实成为瓶颈时再选，不作为当前必装依赖。

## 代码、数据与实验安排

[GRAB](https://github.com/otaheri/GRAB) 可提供数据读取、人体/手指/物体表示与接触信息；[HUMOTO](https://github.com/adobe-research/humoto) 有 Mixamo 兼容的人—物资产及公开子集。后者仓库主要提供数据预览与资源，不能当作已实现的完整控制器；也不能把完整数据集当作全部已开放下载。它们用于增强上述路线的动作来源，不另算独立控制路线。

建议先分别检查 **C1 的交互阶段**与 **C4 的身体动作**，再与 B1 的真实交互片段组合；D1/TRUMANS 纳入动作数据准备，先验证一个片段再扩充。以 robot 接取为核心时，C3 是直接的协议与数据对照。需要现成目标驱动伸手时试 C2；需要在线空间避让时再试 C5。这是基于代码结构与目标的选型判断，尚不是性能排序。

每条路线先测作者原生设置，再测 Lab 适配。动作、事件和接触分别记录：原版绑定/辅助驱动只作明确标注的对照；局部物理手作为共同的接触升级，不能把某个项目已有的 snap/weld 包装成新抓握能力。所有 robot 方法使用同一套 human 行为配置和释放后独立持有判据。
