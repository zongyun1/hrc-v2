# Basic human avatar control

The kitchen demo now includes the old benchmark's `custom_Adrian_Keller.glb`.
The model is copied from the existing remote Genesis asset directory into
`/scratch/jiabenchen_umass/yz/assets/avatars/`; no Genesis runtime is imported.
The old benchmark source used as reference is
`/Users/rielyyunzong/Desktop/genesis-hr-bench/envs/avatar/`.

The `--task grasp` demonstration is **human reaches → robot grasps/lifts → human
retracts**. The new default `--task handover` adds object transfer; see
[the handover implementation and checks](HANDOVER.md). The robot waits for the actual human palm to arrive within 2.5 cm
of the request position. The existing physical Franka grasp checks still apply.
The avatar report additionally verifies hand movement and return to the idle
pose. The grasp-only mode is a basic gesture-triggered interaction.

## Run on AICR

```bash
sbatch /scratch/jiabenchen_umass/yz/hrc-v2/isaac_kitchen/run_remote.sbatch
```

The handover default is 1500 steps / 25 seconds; `--task grasp --steps 960`
selects the prior gesture demo. `--task grasp --no_avatar --steps 720`
selects the previous robot-only demonstration. `--avatar_glb PATH` overrides
the model; the loader currently supports the legacy Z-up, meter-scale,
uncompressed skinned GLBs with the same named arm joints, not arbitrary assets.
The appearance and source model rights remain those of the original asset.

## Python control API

Create `AvatarController` before `sim.reset()` and call `avatar.step(dt)` before
every physics step. Call `avatar.update(dt)` after the physics step to refresh
and verify collision proxy poses. Position arguments use **world XYZ, meters**; yaw is radians
about world Z. These interfaces intentionally resemble the old controller's
`reset`, `step`, `spare` and hand-control methods.

```python
from avatar import AvatarController

avatar = AvatarController(sim.stage, glb_path, output / "avatar_textures",
                          position=(1.25, -0.95, 0), yaw=-1.570796,
                          device=sim.device)
sim.reset()
avatar.reach_hand((0.83, -0.90, 1.10), hand="right", duration=2.0)
while not avatar.spare():
    avatar.step(sim.get_physics_dt())
    sim.step()
    avatar.update(sim.get_physics_dt())

palm_xyz = avatar.get_hand_pos("right")
avatar.set_base_pose((1.30, -0.95, 0), yaw=-1.570796)
avatar.reset()
```

`spare()` reports that the timed motion ended, not that an unreachable target
was reached. Compare `get_hand_pos()` to the target when gating robot actions,
as the demo does. Two-bone IK clamps out-of-reach targets. Base pose control is
kinematic repositioning, without a walking animation or navigation planner.

## Live control on a headless server

Launch manual mode with a command file in the shared project directory:

```bash
cd /scratch/jiabenchen_umass/yz/hrc-v2
sbatch isaac_kitchen/run_remote.sbatch --task grasp --avatar_mode manual \
  --avatar_command_file /scratch/jiabenchen_umass/yz/hrc-v2/outputs/avatar_command.json
```

While it runs, use a second SSH terminal. This helper only needs standard Python
and writes commands atomically. Commands are consumed once per file revision:

```bash
cd /scratch/jiabenchen_umass/yz/hrc-v2
python3 isaac_kitchen/avatar_command.py --file outputs/avatar_command.json \
  reach --hand right --target 0.83 -0.90 1.10 --duration 2
python3 isaac_kitchen/avatar_command.py --file outputs/avatar_command.json \
  reach --hand left --target 0.83 -1.15 1.10 --duration 2
python3 isaac_kitchen/avatar_command.py --file outputs/avatar_command.json \
  base --position 1.30 -0.95 0 --yaw -1.570796
python3 isaac_kitchen/avatar_command.py --file outputs/avatar_command.json reset
```

Manual mode leaves the robot grasp running and disables the scripted human
gesture/return success checks. A previously written command is applied at
startup; use a new command path for a fresh session. Each new command replaces
the current motion for that hand; base changes cancel both hand motions.
Shared filesystem caching can delay commands sent from the login node. Wait
for `AVATAR_COMMAND` in the job log before sending the next command, otherwise
the newer file can replace a command before the simulator reads it. For prompt
local delivery, run the helper on the allocated node:

```bash
srun --jobid YOUR_JOB_ID --overlap -n 1 -c 1 python3 \
  /scratch/jiabenchen_umass/yz/hrc-v2/isaac_kitchen/avatar_command.py \
  --file /scratch/jiabenchen_umass/yz/hrc-v2/outputs/avatar_command.json \
  reach --hand right --target 0.83 -0.90 1.10 --duration 2
```

The demo records a finite rollout and exits, so issue commands while the job
is running, or increase `--steps` and the Slurm time limit for a longer session.

## Representation and limits

The controller reads the original skeleton, inverse bind matrices, vertex
weights and base-color textures, and updates the skinned USD meshes. It follows
the [glTF 2.0 skinning convention](https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#skins).
Face blend shapes, authored locomotion clips and detailed hand articulation are
not part of this first implementation.

The torso capsule and two palm proxies are PhysX kinematic collision bodies.
In handover mode, the receiving right palm uses a flat box and compliant spring grasp forces.
The render mesh, arms and legs do not have detailed collision geometry. The
avatar is position-controlled and does not react dynamically to robot forces;
this is not a full human dynamics or collision-avoidance system. The grasp-only
mode does not attach the cube. Handover mode uses the documented proximity-gated
spring grip to hold the cube; neither mode teleports it.

`check_avatar_rig.py MODEL.glb` checks the actual asset's bind-pose consistency,
finite skinning and six left/right palm targets, including preservation of arm
bone lengths. Run it with the container Python (NumPy/SciPy required).
`avatar_result.json` records the hand trajectories and interaction event steps;
`result.json` includes both the robot checks and the avatar summary.

## Verified automatic interaction

AICR job `713268` passed on 2026-09-07. The right palm reached the request
position at step 120 (2 seconds), triggering Franka. The avatar retracted at
step 670; final return error was 2.78 mm. Its right hand swept 42.31 cm. The
robot lifted the block 17.79 cm and held it for 6.33 seconds. The full human
model, reaching pose and final holding frame were visually inspected.

Video and reports are in `outputs/isaac_kitchen_avatar/713268/` locally and
`/scratch/jiabenchen_umass/yz/hrc-v2/outputs/isaac_kitchen/713268/` remotely.
The CPU rig check passed all six hand targets with maximum error 0.92 mm and
maximum bind-mesh deviation 0.00047 mm.

Manual control job `713272` also completed successfully in 2 minutes 53 seconds
on four CPU cores and one RTX GPU. The command log and sampled states confirm
left-hand motion (26.89 cm displacement), a 5 cm base translation, and reset to
the starting pose. Reset interrupted the left-hand reach before arrival, so this
run verifies command application/cancellation; completed two-hand target
accuracy is covered by the six-target CPU check above. Maximum physics proxy
tracking error was below 0.0001 mm. The robot grasp continued to pass.
Outputs: `outputs/isaac_kitchen_avatar/713272/` locally.

Copied GLB SHA-256:
`938973382976411449e5979c548926b96c997ef90117ff701868465009f41d16`.

## 走到指定位置

`avatar.walk_to([x, y, z], duration=6.0)` 启动水平地面的运动学行走；`avatar.is_walking()` 表示是否仍在行走，`spare()` 同时考虑行走与手部动作。行走不旋转人物朝向，当前仓储路线沿人物朝向布置。

`walking.py` 实现双脚交替落脚、腿部两段 IK 和足部朝向保持；不模拟人体平衡或脚部接触力。仓储任务在走到位后等 30 个物理步才调用 `reach_hand`，并记录及验证该顺序。

真实 GLB 的独立行走检查：`python isaac_kitchen/check_walking.py /path/to/avatar.glb`。361 帧检查覆盖腿长保持、落脚误差、足部连续性和最终位置；完整仿真验证见 [仓储文档](WAREHOUSE.md)。
