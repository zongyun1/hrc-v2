**HARPv2 human motion → Isaac Lab 3: yield-then-reach demo**

Full kitchen integration: [RoboCasa + PandaOmron + human co-simulation](KITCHEN.md),
matching the coworker's Phase 1 scene structure with a parked mobile robot.
Latest: [generated TRUMANS motion retargeted to the textured Adrian Keller avatar](TRUMANS.md),
validated in GPU render job `744065`.

The coworker's NumPy motion schema, skeleton template and LBS skinning are reused
with provenance in [vendor/PROVENANCE.md](vendor/PROVENANCE.md). Genesis is not a
runtime dependency. [motion.py](motion.py) interpolates source rotations with
SLERP, performs FK, and preserves the SMPL-X rest-pelvis convention. It supports
batched motion/skin calculations; [adapter.py](adapter.py) deliberately accepts
only one human for this first simulation scene.

The Isaac adapter authors one kinematic collision capsule per source bone and
an optional skinned surface. Self-collision pairs are filtered; robot/world
collisions remain enabled. Every capsule has a contact sensor filtered to the
Franka, so floor contact is not counted as robot contact. This is a prescribed
animated obstacle, not a dynamically balanced human or a physical finger grasp.
Pose writing follows the existing Isaac avatar approach. Reported forces are
diagnostics, not validated biomechanical impact measurements.

The first task is **yield, then reach**. A walking clip passes beside a table.
Franka holds its configuration while any human capsule enters the goal's 45 cm
protected region, waits for 55 cm clearance sustained for 0.3 seconds, then
reaches a green target. The controller uses privileged capsule geometry and
differential IK. It is a task demonstration, not general collision avoidance,
RoboCasa manipulation, or a learned policy. The motion is relocated by one
recorded world yaw/translation; its time scale and pose sequence are preserved.

Required acceptance checks: human entered the region, robot resumed, hand held
within 2.5 cm of the goal for at least one second, no reported robot contact above
0.1 N, and capsule tracking error below 5 mm. These are prototype diagnostic
thresholds. The test scene intentionally starts robot execution only after an
observed human intrusion and clearance, rather than counting an empty scene as
successful avoidance.

**Inputs**

Obtain these files from the coworker's existing exports; they are not in Git:

- `motion_kitchen_walk.npz`: `MotionSequence` schema v1.0, one sequence (`B=1`).
- `smplx_skeleton_male.json`: the matching body template.
- `smplx_skin_male.npz`: matching surface/weights; optional with explicit `--capsules`.

The template assumes the coworker's `betas=0` body. Nonzero shape coefficients,
gender mismatches, invalid/padded frames, mismatched skin rest joints and invalid
weights are rejected. Finger poses are stored by the source format but this
22-joint adapter does not articulate fingers. A walking clip must have at least
0.7 m net root displacement for this task. Complex paths may not cross/clear the
chosen goal region and should fail the task rather than be silently warped.

Preflight from the repository root (NumPy only):

```bash
python -m isaac_human.run_demo \
  --motion /path/to/motion_kitchen_walk.npz \
  --skeleton /path/to/smplx_skeleton_male.json \
  --skin /path/to/smplx_skin_male.npz --check_assets
```

Run in the project's existing Isaac Lab 3 container, from the project root:

```bash
sbatch isaac_human/run_remote.sbatch \
  --motion /path/to/motion_kitchen_walk.npz \
  --skeleton /path/to/smplx_skeleton_male.json \
  --skin /path/to/smplx_skin_male.npz
```

The launcher targets `/scratch/jiabenchen_umass/yz/hrc-v2`, requests one RTX GPU,
48 GB RAM and ten minutes, and uses the existing read-only beta container. It
checks JSON status because application shutdown may otherwise mask errors.
Output: `outputs/isaac_human/JOB_ID/{result.json,trace.json,preview.mp4,start.png,yield.png,preview.png}`.
Inputs are hashed and their metadata and scene transform are saved in the result.

**Validation status, 2026-09-09**

Eleven CPU regression tests pass: time interpolation/clamping, FK and pelvis offset,
distinct batched skinning, world transforms, invalid inputs, quaternion signs,
capsule orientation/distance, source joint-cache consistency, yielding
hysteresis/reset and no Genesis imports.
Python compilation and shell syntax pass. The actual body/motion inputs were
not found locally or in the checked AICR project tree.

GPU integration validation passed under AICR job `743626` using the synthetic
fixture in the project's Isaac Lab 3 beta container. All five checks passed:
yield at 3.017 s, resume at 5.533 s, final goal hold 7.467 s, peak reported
robot contact 0 N, and maximum capsule-center tracking error below 0.0001 mm.
Rendered frames were inspected. [Video](../outputs/isaac_human/743626/preview.mp4)
and [result](../outputs/isaac_human/743626/result.json) are saved locally and on
AICR. These validate the adapter/control integration, not real human motion,
skin rendering, or sensor sensitivity to positive contacts. The source-motion
run still needs the real export files.

```bash
python -m unittest isaac_human.test_motion -v
```

For API integration testing only, a clearly labelled synthetic capsule fixture
can be generated. It is not TRUMANS or SMPL-X motion and has no body-model skin:

```bash
python -m isaac_human.diagnostic_fixture outputs/isaac_human/diagnostic_inputs
sbatch isaac_human/run_remote.sbatch \
  --motion outputs/isaac_human/diagnostic_inputs/motion.npz \
  --skeleton outputs/isaac_human/diagnostic_inputs/skeleton.json \
  --capsules --allow_diagnostic
```

A passing fixture run produces `diagnostic_passed`, never the real-motion demo's
`passed` label. No synthetic fixture is silently substituted for missing assets.
