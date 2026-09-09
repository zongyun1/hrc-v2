> 新增场景：[Ridgeback + Franka 仓储移动交接](WAREHOUSE.md)，使用 `run_warehouse.sbatch` 启动。人物先走到蓝色标记再伸手；AICR 作业 `713727` 已通过完整验证，[观看新版视频](../outputs/isaac_warehouse/713727/preview.mp4)。

# Franka + human avatar kitchen on AICR

Current implementation and remote validation: [项目进度（2026-09-07）](PROGRESS.md).

This standalone Isaac Lab demo references the Lightwheel Kitchen USD and adds a
fixed-base Franka Panda, a work table, a 5.5 cm, 100 g wooden-colored block,
and the previous Genesis benchmark's skinned human avatar. The default task
now transfers the block from Franka to the human's upward-facing palm.
See [handover model, sequence and checks](HANDOVER.md).
See [avatar controls and implementation](AVATAR.md) for the Python API and
headless live commands.
A differential IK controller approaches from above, descends with open fingers,
closes the parallel gripper, lifts by 18 cm and holds. Approach and descent
advance only after the hand reaches the waypoint within 8 mm and its
orientation is aligned; the fingers stay open during this alignment.
Franka grasps through simulated contact and friction. The human receives the
block through proximity-gated spring grasp forces; the block is never teleported.
This is a scripted interaction, not a trained policy.

The default run has 1500 physics steps (25 seconds), saving a 960×720 image,
a 15 FPS video and a JSON report with cube, grasp-center and finger positions.

Robot-to-human handover passed on AICR in job `713442` on 2026-09-07:
the human carried the block 21.1 cm after release and held it for 1.33 seconds,
with at most 1.88 mm tracking error throughout reception and carry.
See [verified video and metrics](HANDOVER.md#verified-run).

Previous robot-only physical grasp verified on AICR on 2026-09-07: job `713231`, completed with
exit code 0 in 2 minutes 3 seconds. The block rose from z=0.827500 m to
z=1.005444 m (17.79 cm). During 4.33 seconds of holding, minimum lift was
17.22 cm and maximum distance to the grasp center was 1.75 mm. Contact and
holding video frames were visually checked. Outputs:

- Remote: `/scratch/jiabenchen_umass/yz/hrc-v2/outputs/isaac_kitchen/713231/`
- Local: `outputs/isaac_kitchen_grasp/713231/`

The controller uses the installed Isaac Lab 3 XYZW quaternion convention.
This verifies one fixed-pose scripted grasp; randomized grasping has not been tested.

Kitchen fixtures are visual scenery in this version: instances are expanded,
physics APIs are removed, and joints/physics scenes are deactivated in the
composed stage. The source USD is unchanged.
The robot, work table and test cube have physics. Opening kitchen doors or
grasping the original kitchen props requires enabling and configuring them.

The remote project is `/scratch/jiabenchen_umass/yz/hrc-v2`.
Do not use the old `yz/code` symlink for this demo; it points to `genesis-hrc`.
The existing Genesis project and its dependencies are independent of this demo.

## Run

From this local repository, copy only the demo directory:

```bash
rsync -az isaac_kitchen/ aicr:/scratch/jiabenchen_umass/yz/hrc-v2/isaac_kitchen/
ssh aicr 'bash /scratch/jiabenchen_umass/yz/hrc-v2/isaac_kitchen/setup_remote.sh'
ssh aicr 'sbatch /scratch/jiabenchen_umass/yz/hrc-v2/isaac_kitchen/run_remote.sbatch'
```

The Slurm script requests one RTX GPU and four CPU cores for ten minutes. Logs are in
`/scratch/jiabenchen_umass/yz/job_logs/franka_kitchen_<jobid>.log`.
Results are in `hrc-v2/outputs/isaac_kitchen/<jobid>/` (one directory per run):

- `preview.png`: final holding frame.
- `preview.mp4`: 15 FPS overview video.
- `result.json`: grasp/interaction success, kitchen bounds and sampled states.
- `avatar_result.json`: human hand trajectories and robot-trigger events.
- `handover_result.json`: capture, release, withdrawal and human holding checks.

Success requires at least one second of holding, with the block at least 12 cm
above its initial resting height throughout the final hold, within 4.5 cm of the finger grasp center,
and with the combined finger opening above 2.5 cm. The script also rejects
non-finite robot states and empty camera images. A failed grasp saves diagnostic
video and JSON before raising an error. The Slurm wrapper checks the JSON status
because the simulator shutdown can otherwise mask Python exceptions.

`--steps` must be at least 1200 for handover, 720 for the grasp gesture, or 600
for manual / robot-only grasp modes; phase durations scale with the requested count.
For example, append `--steps 1800` to the `sbatch` command for a slower run.

Recording uses path tracing with 64 samples per pixel, because the container's
NGX-based real-time denoising is unavailable. A short settling period
precedes recording. Kit logs and shader cache persist under `yz/.cache/kitchen-kit-*`.

## Runtime

The launcher uses the existing, read-only official container:

```
/scratch/jiabenchen_umass/moment-contact-memory/external/isaaclab300b2p1_ngc_container_v1/isaac-lab_3.0.0-beta2-post1_amd64.sif
```

It uses Apptainer `--nv --fakeroot --writable-tmpfs`, with a separate home/cache
under `yz/.cache/kitchen-home`. `--fakeroot` maps the current user inside the
container; it is needed to access the image's root-owned `/isaac-sim` directory.
No host administrator privileges or driver changes are required.

Host pip installation of Isaac Sim 5.1 was unsuitable because the host glibc is
2.34. The older 5.1 container also crashed in RTX initialization on driver
595.71.05, matching [this upstream report](https://github.com/isaac-sim/IsaacSim/issues/687).

The beta container's default Franka URL returns 404. The demo pins the official
5.1 `panda_instanceable.usd` instead; `--franka_usd` can override it. This changes
the robot asset version, not the simulator runtime. First use downloads the
robot's referenced meshes and materials from NVIDIA's asset server.

## Assets

[Lightwheel Kitchen](https://github.com/LightwheelAI/Lightwheel_Kitchen),
licensed CC BY-NC 4.0. Asset directory:

```
/scratch/jiabenchen_umass/yz/assets/lightwheel-kitchen/Collected_KitchenRoom/KitchenRoom.usd
```

The setup script uses the GitHub LFS archive because the README's Google Storage
download returned 404. It verifies the archive SHA-256 before first extraction.
Keep the extracted folder hierarchy intact for textures and appliance references.

`inspect_asset.py` is an optional CPU-only USD bounds inspector (requires
`usd-core`); run it on a compute node. The login node did not support the USD
wheel's CPU instructions. It is not required by the simulation.
