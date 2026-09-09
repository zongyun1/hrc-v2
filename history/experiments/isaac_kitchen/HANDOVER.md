# Robot-to-human block handover

The default kitchen task now runs the complete sequence: the human offers an
upward-facing palm, Franka grasps and lifts the block, raises it clear of the
palm before translating horizontally above it,
lowers the block, waits for the human grasp, opens both robot fingers, retreats,
and lets the human carry the block toward their body.

```bash
sbatch /scratch/jiabenchen_umass/yz/hrc-v2/isaac_kitchen/run_remote.sbatch
```

The default is 1500 steps / 25 seconds of simulation. Handover requires
`--avatar_mode demo` and at least 1200 steps. To use the previous modes:

```bash
sbatch isaac_kitchen/run_remote.sbatch --task grasp --steps 960
sbatch isaac_kitchen/run_remote.sbatch --task grasp --no_avatar --steps 720
sbatch isaac_kitchen/run_remote.sbatch --task grasp --avatar_mode manual \
  --avatar_command_file /scratch/jiabenchen_umass/yz/hrc-v2/outputs/avatar_command.json
```

## Physical model

Franka picks up the dynamic block with finger contact and friction. The human
remains a kinematic avatar. Its receiving palm has a small, flat collision box;
the skin turns palm-up and curls the fingers during reception.

`HumanGrasp` activates only after the block is within 1.5 cm horizontally and
2.7–5.5 cm vertically above the palm center. It records the existing relative
position and orientation, then applies a damped six-degree-of-freedom spring
force/torque to the dynamic block through Isaac Lab's wrench composer. No block
pose writes, dynamic attachment joints or gravity disabling are used.

The spring uses 80 N/m position stiffness, 5.6 N·s/m damping, palm-velocity
feedforward and an upward holding force balancing the 100 g block's weight.
Rotational gains are 0.006 N·m/rad and 0.0011 N·m·s/rad. Total force and torque
are limited to 5 N and 0.015 N·m. These gains target this block at 60 Hz.

This is a **simplified compliant human grasp**, not a simulation of individual
human finger contact forces. The block retains gravity, contact and rigid-body
dynamics. The robot releases only after this grip is active, and withdraws only
after both robot fingers have opened beyond 3.5 cm. The human then moves its hand
while the block follows through the applied grip forces.

Withdrawal interpolates back to the measured arm joint configuration saved
before offering the block. This avoids Cartesian IK divergence near the
extended-arm transfer pose. Physics and control run at 60 Hz; skin mesh updates
and recorded rendering run at 15 Hz.

## Success checks

The original physical robot grasp checks apply before the transfer. During
the final human hold, the checks additionally require:

- At least one second of holding after the hand carry.
- Less than 1 cm block movement on the first physics step after capture, and
  less than 1.5 cm palm-relative tracking error throughout reception and carry.
- Block position within 1.5 cm of its captured palm-relative position.
- Robot grasp center at least 15 cm from the block, with total finger opening
  greater than 7 cm.
- Block height above 1 m and human hand carry distance greater than 15 cm.
- Kinematic collision proxies tracking the skin within 5 mm.

`handover_result.json` records capture/release/carry steps and these metrics.
`result.json` contains the complete phase and object trajectory. Video and
reports are saved even if a final success check fails.

## Verified run

AICR job `713442` passed on 2026-09-07 with 1500 physics steps. Franka lifted
the block at least 17.2 cm before transfer, then opened its fingers to 8.0 cm
and withdrew. The human carried the block 21.1 cm and held it for 1.33 seconds;
the robot grasp center remained at least 39.0 cm away during the final hold.
Capture moved the block only 0.058 mm on the first physics step. Maximum
palm-relative tracking error throughout reception and carry was 1.88 mm.

- Local video: [preview.mp4](../outputs/isaac_kitchen_handover/713442/preview.mp4)
- Local metrics: [handover_result.json](../outputs/isaac_kitchen_handover/713442/handover_result.json)
- Remote output: `/scratch/jiabenchen_umass/yz/hrc-v2/outputs/isaac_kitchen/713442/`
