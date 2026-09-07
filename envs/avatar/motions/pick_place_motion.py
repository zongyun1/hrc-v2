"""Pick-and-place motion: avatar reaches with its arm to a pick position,
attaches an object, transports it to a place position, releases.

General-purpose: callable from any task via AvatarController.pick_and_place(
    pick_pos=world_xyz1, place_pos=world_xyz2, attach_obj=actor, hand_id=1).

Uses Genesis's built-in FABRIK IK (`skin.ik_solve`) along the shoulder →
hand chain to bend only the arm, with the body held in the rest pose.
The target is the PALM (not the wrist): we subtract the idle wrist→palm
offset from the palm target, solve wrist IK, then iteratively refine the
wrist target against the actual palm position for sub-mm convergence.

Wrist orientation: fabrik is position-only, so by default the palm ends
up tilted or facing up on table reaches.  ``palm_down=True`` (the default)
post-rotates the hand subtree about the palm centre each frame so the palm
faces world -Z — ramped in over the approach, held through transport, and
ramped out during retract (see ``palm_orient.py``).
"""
import numpy as np
import genesis as gs
import genesis.utils.geom as geom_utils
from scipy.spatial.transform import Rotation as R

from .base_motion_module import BaseMotionModule
from .palm_orient import (
    apply_reference_arm_pose,
    capture_arm_pose,
    rotate_hand_subtree_palm_down,
)
from ..utils import AvatarState, ActionStatus, MIXAMO_JOINT_NUM, MIXAMO_SKIN_JOINT_NUM


def _smoothstep(x):
    x = float(np.clip(x, 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)


class PickPlaceMotion(BaseMotionModule):
    def __init__(self, motion_name, robot):
        super().__init__(motion_name, robot)
        self.at_frame = 0
        self.frames = []       # list[np.ndarray] — node_trans per frame
        self.attach_frame = 0
        self.detach_frame = 0
        self._attach_obj = None
        self._hand_id = 1
        self._ease = False
        # Per-frame palm-down state, set by start(); zero for callers that
        # use _solve_palm_ik directly (e.g. StyledPickPlaceMotion).
        self._orient_weight = 0.0
        self._orient_mode = "reference"
        self._orient_enabled = False
        self._rest_arm_pose = None
        self._attach_palm_offset = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self, pick_pos, place_pos, attach_obj=None, hand_id=1,
              approach_frames=80, transport_frames=120, retract_frames=100,
              approach_arc=0.15, transport_arc=0.20, retract_arc=0.15,
              refine_iters=2, palm_down=True, palm_down_mode="reference",
              palm_down_weight=1.0, palm_down_ramp_frac=0.5,
              ease_in_out=True, attach_palm_offset=0.05):
        """Plan the full trajectory up-front and hand it to step() for replay.

        pick_pos / place_pos: world xyz for the PALM centre.
        attach_obj: optional Actor / entity to attach on reaching pick_pos.
        hand_id: 0 = left, 1 = right.
        approach/transport/retract frames + arcs control motion pacing.
        palm_down: post-rotate the hand subtree so the palm faces world -Z
          (full weight from attach through detach; blended in over the first
          ``palm_down_ramp_frac`` of the approach and back out over the
          retract).  ``palm_down_mode``: "hybrid" (forearm pronation + minimal
          wrist bend, exact), "forearm_roll" (pronation only, approximate) or
          "align" (single shortest-arc rotation).
        ease_in_out: smoothstep time-warp on each phase so the arm
          accelerates/decelerates instead of moving at constant speed.
        attach_palm_offset: with a palm-down carry the object should hang
          BELOW the palm, not sit at the palm centre (the palm-up-era
          attach put the object centre at the knuckle plane — half inside
          the hand).  When palm_down and attach_obj are set, the PALM
          trajectory targets pick/place + offset·ẑ while the object
          attaches offset below the palm — so the object never teleports
          at attach and still lands exactly at place_pos at detach.
        """
        if self.robot.action_state != AvatarState.NO_ACTION:
            gs.logger.warning(
                f"Cannot start {self.motion_name}: AvatarState is "
                f"{self.robot.action_state}."
            )
            return
        if self.robot.base_state != AvatarState.STANDING:
            gs.logger.warning(
                f"Cannot start {self.motion_name}: BaseState is "
                f"{self.robot.base_state}."
            )
            return

        self._hand_id = hand_id
        self._attach_obj = attach_obj

        # Live palm position BEFORE we re-seed the skin chain — this is where
        # the hand visibly is right now, and it's what the approach should
        # start from so back-to-back picks flow without a teleport.
        start_palm = np.asarray(
            self.robot.get_palm_center(hand_id), dtype=np.float64
        ).ravel()[:3]

        # Live arm bone frames at the same moment: the orientation blend's
        # rest endpoint (approach ramps from it, retract returns to it).
        self._rest_arm_pose = capture_arm_pose(self.robot, hand_id)
        self._orient_enabled = bool(palm_down) and float(palm_down_weight) > 0.0

        # Seed skin.transforms once so fabrik reads a consistent rest chain.
        self._seed_skin_transforms()

        # Measure the REST palm and wrist→palm offset for target conversion.
        rest_palm = np.asarray(
            self.robot.get_palm_center(hand_id), dtype=np.float64
        ).ravel()[:3]
        rest_wrist, _ = self.robot._get_hand_ref(hand_id)
        rest_wrist = np.asarray(rest_wrist, dtype=np.float64).ravel()[:3]
        self._idle_palm = rest_palm
        self._wrist_to_palm = rest_palm - rest_wrist

        pick = np.asarray(pick_pos, dtype=np.float64).ravel()[:3]
        place = np.asarray(place_pos, dtype=np.float64).ravel()[:3]
        hand_side = "Right" if hand_id == 1 else "Left"

        # Palm-down carry: raise the PALM waypoints so the object (which
        # attaches attach_palm_offset below the palm) sits at the task's
        # requested pick/place heights.
        self._attach_palm_offset = (
            float(attach_palm_offset)
            if (palm_down and attach_obj is not None) else 0.0
        )
        offset_vec = np.array([0.0, 0.0, self._attach_palm_offset])
        pick = pick + offset_vec
        place = place + offset_vec

        # Plan three phases; each element is one node_trans per sim frame.
        # Approach begins from the LIVE palm — keeps motion continuous when
        # chained from a prior pick_and_place that ended at place_pos.
        # The palm-down orientation is applied INSIDE _solve_palm_ik (per
        # frame, weight-ramped) so the position refinement compensates the
        # palm shift the orientation edit introduces — the wrist pivots in
        # place and the palm still lands on target.
        self._ease = bool(ease_in_out)
        self._orient_mode = str(palm_down_mode)
        n_a, n_t = int(approach_frames), int(transport_frames)
        n_r = max(0, int(retract_frames))
        if palm_down and float(palm_down_weight) > 0.0:
            weights = self._palm_down_weights(
                n_a, n_t, n_r,
                float(palm_down_weight), float(palm_down_ramp_frac),
            )
        else:
            weights = [0.0] * (n_a + n_t + n_r)

        segments = [(start_palm, pick, n_a, approach_arc),
                    (pick, place, n_t, transport_arc)]
        if n_r > 0:
            segments.append((place, rest_palm, n_r, retract_arc))
        self.frames = []
        k = 0
        n_approach = 0
        n_transport = 0
        for seg_i, (a, b, n, arc) in enumerate(segments):
            for p in self._arc_interp(a, b, n, arc):
                self._orient_weight = weights[k] if k < len(weights) else 0.0
                self.frames.append(
                    self._solve_palm_ik(hand_side, p, refine_iters)
                )
                k += 1
            if seg_i == 0:
                n_approach = len(self.frames)
            elif seg_i == 1:
                n_transport = len(self.frames) - n_approach
        self._orient_weight = 0.0
        phase_approach = self.frames[:n_approach]
        phase_transport = self.frames[n_approach:n_approach + n_transport]
        # Attach at the last frame of the approach (hand at pick).
        self.attach_frame = max(0, len(phase_approach) - 1)
        # Detach at the last frame of transport (hand at place).
        self.detach_frame = len(phase_approach) + len(phase_transport) - 1
        self.at_frame = 0

        self.robot.action_state = self.motion_name
        self.robot.action_status = ActionStatus.ONGOING

    def step(self, skip_avatar_animation=False):
        """Advance one frame.  Called by AvatarController.step()."""
        if not self.frames:
            self.robot.action_state = AvatarState.NO_ACTION
            self.robot.action_status = ActionStatus.SUCCEED
            return

        if skip_avatar_animation:
            self.at_frame = len(self.frames) - 1

        if self.at_frame >= len(self.frames):
            self.robot.action_state = AvatarState.NO_ACTION
            self.robot.action_status = ActionStatus.SUCCEED
            return

        # Apply this frame's node_trans.
        self.robot.node_trans = self.frames[self.at_frame]

        # Attach at pick keyframe.  Palm-up era snapped the object TO the
        # palm centre; with a palm-down carry the object hangs
        # _attach_palm_offset below the palm instead (the palm waypoints
        # were raised by the same amount in start(), so the object does
        # not teleport here and lands at the task's place_pos at detach).
        if (self._attach_obj is not None
                and self.at_frame >= self.attach_frame):
            entity = getattr(self._attach_obj, "entity", self._attach_obj)
            palm_now = np.asarray(
                self.robot.get_palm_center(self._hand_id), dtype=np.float64
            ).ravel()[:3]
            snap = palm_now - np.array(
                [0.0, 0.0, float(getattr(self, "_attach_palm_offset", 0.0))]
            )
            try:
                entity.set_pos(snap.astype(float))
            except Exception:
                pass
            self.robot.attach_object_to_hand(self._hand_id, entity)
            self._attach_obj = None

        # Detach at place keyframe.
        if self.detach_frame is not None and self.at_frame >= self.detach_frame:
            self.robot.detach_object(self._hand_id)
            self.detach_frame = None

        self.at_frame += 1

        # Complete: mark spare.
        if self.at_frame >= len(self.frames):
            self.robot.action_state = AvatarState.NO_ACTION
            self.robot.action_status = ActionStatus.SUCCEED

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _arc_interp(self, start, end, n_steps, arc_height):
        start = np.asarray(start, dtype=np.float64)
        end = np.asarray(end, dtype=np.float64)
        for k in range(1, int(n_steps) + 1):
            t = k / float(n_steps)
            if getattr(self, "_ease", False):
                t = _smoothstep(t)
            pos = start * (1.0 - t) + end * t
            if arc_height > 0:
                pos = pos.copy()
                pos[2] += arc_height * 4.0 * t * (1.0 - t)
            yield pos

    @staticmethod
    def _palm_down_weights(n_approach, n_transport, n_retract,
                           peak_weight, ramp_frac):
        """Per-frame palm-down blend: 0 → peak over the first ``ramp_frac``
        of the approach, held at peak through transport (so attach + detach
        keyframes see the full correction), fading back to 0 over the
        retract (the hand returns to its natural rest orientation)."""
        ramp_frac = min(max(ramp_frac, 1e-3), 1.0)
        weights = []
        for k in range(n_approach):
            t = (k + 1) / float(max(1, n_approach))
            weights.append(peak_weight * _smoothstep(t / ramp_frac))
        weights.extend([peak_weight] * n_transport)
        for k in range(n_retract):
            t = (k + 1) / float(max(1, n_retract))
            weights.append(peak_weight * (1.0 - _smoothstep(t)))
        return weights

    def _seed_skin_transforms(self):
        pose = self.robot.pose
        motion_trans = pose[:3]
        joint_quats = pose[3:].reshape(-1, 4)

        skin_base_rot = geom_utils.R_to_quat(
            geom_utils.euler_to_R(
                R.from_matrix(self.robot.global_rot)
                .as_euler("xyz", degrees=True)[[0, 2, 1]]
            ) @ geom_utils.quat_to_R(pose[3:7])
        )

        mixamo_pose = np.array([[0.0, 0.0, 0.0, 1.0]] * MIXAMO_JOINT_NUM)
        global_quat = np.concatenate(
            [skin_base_rot.reshape(1, -1),
             joint_quats[1:MIXAMO_SKIN_JOINT_NUM]]
        )
        result = np.matmul(
            np.matmul(self.robot.global_mat_inv,
                      geom_utils.quat_to_R(global_quat)),
            self.robot.global_mat,
        )
        mixamo_pose[:MIXAMO_SKIN_JOINT_NUM] = geom_utils.R_to_quat(result)[:, [1, 2, 3, 0]]

        total_trans = (self.robot.global_rot @ self.robot.base_rot @ motion_trans
                       + self.robot.global_trans)
        base_trans = total_trans[[1, 2, 0]]

        self.robot.skin.calculate_real_pos(mixamo_pose, base_trans)

    def _solve_palm_ik(self, hand_side, palm_target, refine_iters):
        """Solve IK so the palm ends at palm_target; return a fresh
        node_trans matrix list suitable for assignment to robot.node_trans.

        When self._orient_weight > 0 the palm-down orientation is applied
        after each IK pass, BEFORE measuring the palm error — so the
        refinement drives the ORIENTED palm onto the target."""
        wrist_target = palm_target - self._wrist_to_palm
        self._ik_step(hand_side, wrist_target)
        self._apply_orientation()
        for _ in range(refine_iters):
            actual = np.asarray(
                self.robot.get_palm_center(self._hand_id), dtype=np.float64
            ).ravel()[:3]
            err = actual - palm_target
            if np.linalg.norm(err) < 1e-3:
                break
            wrist_target = wrist_target - err
            self._ik_step(hand_side, wrist_target)
            self._apply_orientation()
        return self.robot.node_trans.copy()

    def _apply_orientation(self):
        """Apply the palm-down retarget for the current frame weight.

        In "reference" mode the retarget itself is always full-strength;
        the ramp weight w becomes the BLEND between the live rest posture
        captured at start() (w=0) and the palm-down reference (w=1).  The
        orientation therefore always interpolates between two natural
        postures — never toward raw FABRIK frames, whose arbitrary twist
        made the palm flip outward/inward as the hand returned to the
        body during retract."""
        if not getattr(self, "_orient_enabled", False):
            return
        w = float(getattr(self, "_orient_weight", 0.0))
        nt = None
        if self._orient_mode == "reference":
            nt = apply_reference_arm_pose(
                self.robot, self.robot.node_trans, self._hand_id,
                weight=1.0, rest_pose=self._rest_arm_pose,
                rest_blend=1.0 - w,
            )
        if nt is None:
            # Fallback (no full-arm reference data): the old delta-based
            # ramp toward FABRIK.  Skip near-zero weights.
            if w <= 1e-3:
                return
            fallback = ("hybrid" if self._orient_mode == "reference"
                        else self._orient_mode)
            nt = rotate_hand_subtree_palm_down(
                self.robot, self.robot.node_trans, self._hand_id,
                weight=w, mode=fallback,
            )
        self.robot.node_trans = nt
        self.robot.update()

    def _ik_step(self, hand_side, wrist_target_world):
        skin = self.robot.skin
        new_nt = skin.ik_solve(
            hand_id=f"{hand_side}Hand",
            root_id=f"{hand_side}Shoulder",
            target_pos=np.asarray(wrist_target_world, dtype=np.float64).copy(),
            max_iterations=50,
            tolerance=1e-4,
        )
        self.robot.node_trans = new_nt
        # Refresh skin so get_palm_center reads the new pose.
        self.robot.update()
