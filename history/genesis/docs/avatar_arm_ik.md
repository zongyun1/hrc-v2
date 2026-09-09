# Avatar right-arm IK — current state and open wrist-orientation TODO

## Public action

Tasks can drive the avatar's palm through a pick → place trajectory with a
single call on `AvatarController`:

```python
self.avatar.pick_and_place(
    pick_pos=world_xyz_1,
    place_pos=world_xyz_2,
    attach_obj=actor,      # optional; attached at pick, detached at place
    hand_id=1,             # 0=left, 1=right
    # optional pacing / arc knobs:
    approach_frames=80, transport_frames=120, retract_frames=100,
    approach_arc=0.15, transport_arc=0.20, retract_arc=0.15,
)
while not self.avatar.spare():
    self.step_sim()
```

The motion plans all frames up-front in `envs/avatar/motions/pick_place_motion.py`
and replays them via the standard `AvatarController.step()` integration —
same shape as `play_animation`.

## Implementation notes (for extending / debugging the motion)

`envs/tasks/debug_avatar_pickplace.py` is the integration test and usage
example.  Validated end-to-end: sub-mm palm accuracy, smooth motion, no
wrist wobble.

- **Seed**: `Mixamo_global_processing(pose[3:].reshape(-1,4), global_mat,
  global_mat_inv, skin_base_rot)` → `skin.calculate_real_pos(mixamo_pose,
  base_trans)` with NO `root_id`.  This populates `skin.transforms` in the
  world-axis-swapped frame that fabrik expects to read from.
- **Solve**: `skin.ik_solve("RightHand", "RightShoulder", palm_target_world
  - wrist_to_palm_offset)` returns a new `node_trans`.  Assign it to
  `avatar.robot.node_trans` and call `avatar.robot.update()` to render.
- **Do NOT overwrite `skin.transforms`** with the solve output.  fabrik
  expects to read the original seeded chain on every call — mixing the
  output frame (Hips-relative via `forward_kinematics(root_id=skin_joints[0])`)
  with the world-axis-swapped seed frame sends fabrik into bogus territory.
- **Palm rather than wrist**: the solver targets the wrist joint.  To aim
  the palm at a world point, subtract the idle wrist→palm offset
  (`avatar.robot.get_palm_center(1) − avatar.get_hand_pos(1)` at rest)
  from the palm target to get the wrist target, then refine 1–2 passes
  against `avatar.robot.get_palm_center(1)` for sub-mm convergence.
- **Cache + replay** inside the motion: all phase frames are pre-computed,
  then replayed one-per-sim-step, so there are no discontinuities at phase
  boundaries.

The canonical upstream pattern we modelled on is
`mawm/datagen/Virtual-Community/modules/avatar/reach_obj_motion.py`.

## Natural stance: joint (hand, shift) search — no cross-body carry

``AvatarController._natural_pick_place_pose`` (the ``natural=True`` path
used by assist tasks) no longer stands the avatar at the pick/place
midpoint with the hand chosen from the pick side — that design put pick
and place on opposite sides of the body by construction, so wide spans
always produced a cross-body carry with the forearm sweeping through the
torso.  It now searches both hands × lateral stance shifts (±0.35 m,
clipped by ``body_x/y_bounds``) and scores cross-body penalty (both
endpoints ≥5 cm onto the chosen hand's side), shoulder-reach feasibility
(soft 0.60 m / hard 0.68 m, live-measured shoulder offsets), and a small
|shift| preference.  Forced ``hand_id`` is honored (stance-only
adaptation); an infeasible search falls back to the exact legacy stance;
residual cross-body spans are logged
(``[avatar] natural pose: residual cross-body ...``).
``probe_avatar_palm_down.py`` reports ``min_arm_torso_clearance`` to
check penetration numerically (<0.05 m = inside the torso; the idle
stance itself measures ~0.10–0.12 m from the body axis).

## Attach offset (palm-down carry)

With the palm facing down the attached object must hang BELOW the palm.
``attach_palm_offset`` (default 0.05 m, `pick_and_place` kwarg) does this
in two coupled halves — changing only one of them is wrong:

- the PALM waypoints are raised by +offset·ẑ in ``start()`` (so the
  palm approaches 5 cm above the task's pick/place points), and
- the object snaps to ``palm − offset·ẑ`` at the attach keyframe.

Net: no teleport at attach (the object is already ~where the snap puts
it), the object rides 5 cm under the palm through the carry, and at
detach it releases exactly at the task's original ``place_pos`` — task
success semantics unchanged.  Offsetting only the object would sink it
into the table at attach (task pick lifts are ~1 cm); raising only the
palm would leave the object embedded in the hand.  The offset is gated
on ``palm_down and attach_obj is not None`` so gesture reaches and the
knife handover keep their exact old trajectories.

## Effective reach (don't under-estimate it)

The 2-bone skeleton is upper=0.271m + forearm=0.255m = **0.526m shoulder→wrist**.
The *naive* reachable zone on a z=0.815m table around the shoulder
`(-0.161, 0.717, 1.206)` is a disc of radius `sqrt(0.526² − 0.391²) ≈ 0.352m`.

But in practice the palm reaches ~**0.10m further** than that for two reasons:

1. The **palm is offset ~9cm from the wrist** (wrist→palm offset measured at
   idle, stored in `PickPlaceMotion._wrist_to_palm`).  The effective palm
   reach is roughly `0.526 + 0.09 ≈ 0.62m` from shoulder.
2. The refinement loop inside `_solve_palm_ik` pushes the **wrist target**
   further forward whenever the palm falls short (`wrist_target −= palm_err`).
   This keeps converging as long as the wrist target is reachable — so the
   usable palm zone is a little fatter than the wrist zone.

Empirically, palm targets up to ~**0.65m from the shoulder** (e.g.
`(-0.32, 0.22, 0.815)` — across the table, 20cm beyond the naive limit)
converge to the target within ~1.5cm and ship `SUCCESS` (seen in job
55793219, `data/avatar_pickplace_debug_55793219.log`).

If a target IS truly out of reach, fabrik just clamps the palm to the sphere
surface — there's no blow-up, the motion just lands short.  Easiest sanity
check before running: verify `np.linalg.norm(target − live_shoulder) < 0.65`.

## SOLVED (2026-07-05) — palm-down via post-IK hand-subtree rotation

Palm-down is now the DEFAULT for `pick_and_place`.  The fix is not in the
IK at all: after each frame's position solve,
`envs/avatar/motions/palm_orient.py::rotate_hand_subtree_palm_down`
rigidly rotates the hand subtree (hand + fingers) about the PALM CENTRE
in global-transform coordinates, so the reached palm position stays fixed
while the palm faces down.  The weight ramps 0→1 over the first half of
the approach, holds 1.0 from attach through detach, and fades out over
the retract.

Default mode ``"reference"`` retargets a NATURAL palm-down wrist posture
harvested from the authored clips instead of computing a minimal
correction (which pinned the normal but inherited FABRIK's arbitrary
wrist twist — visibly weird wrist/finger tilt).
`scripts/probe_palm_ref_scan.py` scans every clip loaded by the
controller for reach-like frames whose palm already faces down;
`scripts/probe_palm_ref_harvest.py` re-extracts the chosen frames with
full context into `envs/avatar/motions/palm_ref_pose.json` — currently
right ← ``Inspect[22]`` (down 0.9996, wrist bend 10°) and left ←
``take_from_human_intention[202]`` (down 0.9996, bend 20°).

At runtime (``palm_orient.apply_reference_arm_pose``, v6) FABRIK
contributes **joint positions only**; the rotation frame of EVERY arm
bone — Arm, ForeArm, ForeArm1, ForeArm2, Hand — is rebuilt as
``swing(ref bone dir → IK bone dir) @ Rz(yaw) @ R_ref_bone``: all twist
along the chain comes from the authored clip, the swing is the minimal
rotation onto the IK geometry.  Why full-arm: FABRIK replaces every
chain bone's rotation with a shortest-arc aim (zero twist control), so
the forearm and upper arm come out arbitrarily rolled — correcting only
the hand (v1–v5) can never fix that.  The hand subtree rotates rigidly
about the WRIST so it stays glued to the forearm tip (the earlier
palm-centre pivot silently detached the wrist joint from the forearm),
and the resulting palm-position shift is re-converged inside
``_solve_palm_ik`` — the orientation is applied before each refinement
error measurement, so the ORIENTED palm lands on target.  Palm-down is
restored by distributed pronation (FA1 ⅓, FA2 ⅔ rotation-only, hand
full) plus a 15°-capped residual bend; palm-downness measures ≥ 0.975
rather than exactly 1.0 on steep reaches — deliberate, trading a few
degrees of "down" for an untwisted arm.  Interior bones get
rotation-only edits (their joint positions are the IK's); this is valid
because the row transforms carry position and rotation independently.

**Shoulder-twist clamp (v8).**  Swinging the reference onto steep IK
reach directions can compose upper-arm roll outside anything the
authored clips contain.  ``scripts/probe_shoulder_twist_scan.py``
measures the clips' natural envelope — upper-arm twist defined as the
swing-twist residual about the bone axis versus the idle rest frame
swung onto the current bone direction — over every loaded clip frame
(~9.3k): left [−97.6°, +36.6°], right [−35.5°, +91.0°] (2.5–97.5 pct),
stored in ``palm_ref_pose.json``.  At runtime the retargeted Arm frame's
twist is measured by the same metric and clamped into the envelope
(rotation-only edit on the Arm bone; forearm/hand frames and all joint
positions untouched).  Re-run the scan if the motion library changes.
Falls back to ``"hybrid"`` (forearm pronation + minimal residual bend)
if the JSON is missing or the forearm is vertical; ``"forearm_roll"``
and ``"align"`` remain as diagnostic modes.

Two implementation traps fixed in the v4 iteration (2026-07-07) — read
before touching the subtree-edit code:

- **Row-vector transform convention.**  Node/global transforms act as
  ``p' = p @ M`` (rotation block stores the column matrix TRANSPOSED,
  position in row ``[3, :3]``).  Composing a world rotation ``D`` onto a
  bone's global transform is therefore a RIGHT-multiply of the rotation
  block by ``D.T``.  A left-multiply produces correct bone POSITIONS
  (so any position-derived metric — e.g. the palm normal from knuckle
  positions — still reads perfect) while feeding garbage rotation frames
  to the skinning: the mesh renders twisted at every hand joint.  Always
  review the RENDERED SKIN, not just position metrics.
- **Twist-bone layout.**  ``ForeArm1``/``ForeArm2`` are sibling twist
  HELPERS next to the hand (children of ``ForeArm``), not chain ancestors
  of it, in this benchmark's Mixamo rigs.  Rotating "their subtree" does
  not move the hand.  The correct pronation distribution is: hand subtree
  receives the full correction; the helpers receive 1/3 and 2/3 of the
  forearm-axis twist component so the forearm skin blends
  elbow(0) → FA1(⅓) → FA2(⅔) → hand(1), matching how the authored clips
  animate them.  ``rotate_hand_subtree_palm_down`` auto-detects the
  chain-vs-sibling layout at runtime.

Two API gotchas:

- the `e1×e2` hand-frame normal MIRRORS between hands — palm-down is
  e3 = −Z for the right hand but e3 = **+Z** for the left;
- `take_from_human_safety.py`'s knife handover passes
  ``palm_down=False`` deliberately (it presents a knife handle with its
  own pose interpolation).

Verified with `scripts/probe_avatar_palm_down.py` on
`blocks_ranking_size_assist`: before e3_z ≈ +0.9 (palm up), after exactly
±1.0 at attach/mid-transport/detach for both hands.  ``ease_in_out=True``
(also default) additionally smoothsteps each phase's pacing so the arm
accelerates/decelerates instead of moving at constant speed.

The rest of this section is kept as history of the earlier failed
attempts (composing deltas into ``hand_node.rotation``):

The palm previously ended up in whatever orientation fabrik computes from
the 2-bone chain's position solve.  For reach targets on a table, the
palm does NOT naturally face down — it comes out tilted or even facing
up, depending on the starting chain positions and the target geometry.

Attempts so far (all unsuccessful):
1. **Compose a delta quaternion onto `hand_node.rotation`** — read current
   palm normal from `robot._get_hand_frame(1)`, compute axis/angle that
   would align it with world −z, apply to the `RightHand` node's local
   quaternion, re-run `forward_kinematics`.
2. Tried both **left-multiply and right-multiply** of the delta
   (`quat_mul(delta, cur)` vs `quat_mul(cur, delta)`).  Right-multiply
   gets a partial correction; neither fully aligns.
3. Tried **per-iteration application** (inside the refinement loop) and
   **once after position convergence** — both produce wrist wobble or
   mis-aligned palm (jobs 55791899/55791950/55792100 in the iteration
   log).

Suspected root causes:
- The skin-frame ↔ world-frame mapping is translation-only for positions
  (`skin = world + [0, 0, 0.959]`) but rotation semantics between
  `hand_node.rotation` (local) and the world-frame palm normal aren't
  straightforwardly a pure rotation — they're conjugated by all ancestor
  bone rotations (RightShoulder, RightArm, RightForeArm, RightForeArm1/2
  twist bones).  A correct delta needs to be conjugated by the parent's
  world rotation: `delta_local = R_parent_world.T @ delta_world @
  R_parent_world`.
- The RightHand node has twist sub-bones (`RightForeArm1`, `RightForeArm2`,
  nodes 33/34) between it and the fabrik-modified `RightForeArm`.  Pose
  twist is propagated through those sub-bones; fabrik doesn't touch
  them.  The right place to inject palm-down might be on those twist
  bones, not on `RightHand`.
- FABRIK is a position-only solver — it does not conserve any
  orientation objective.  A proper fix likely needs a **6-DoF IK** that
  takes both `palm_target_pos` and `palm_target_normal` as objectives,
  or a post-IK analytic wrist-roll that rotates around the forearm axis
  to pin palm-normal without moving the palm position.

Recommended next step for whoever picks this up:
1. Read `envs/avatar/robot.py::_get_hand_frame` to understand the
   `(e1=thumb_axis, e2=finger_axis, e3=palm_normal)` frame.
2. Compute parent-world rotation by extracting `node_trans[ForeArm2][:3, :3].T`
   (remember the transpose convention from `parse_matrix`) and conjugate
   the desired world-frame delta into the hand's local frame before
   composing into `hand_node.rotation`.
3. If that still doesn't converge cleanly, rotate one of the forearm
   twist nodes (33 or 34) around the forearm axis instead.  Angle =
   signed angle between current palm_normal and desired palm_normal
   measured in the plane perpendicular to the forearm.
4. Test against `envs/tasks/debug_avatar_pickplace.py`; the diagnostic
   log line `palm normal (world) = ...` is the single-sentence success
   metric (want `[*, *, -1]` within ~0.05).

Validation logs for failed attempts live in
`data/avatar_pickplace_debug_557919{99,50}.log` and
`data/avatar_pickplace_debug_55792{100,354}.log`.
