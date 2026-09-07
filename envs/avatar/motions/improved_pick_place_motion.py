"""Improved styled pick-place motion (side-by-side, not wired into tasks).

Targets two visual issues found in the production
``StyledPickPlaceMotion`` path (Inspect2 prior, baseline measured in
``data/current_pickplace_baseline/v0``):

1. **Right-hand keyframe snap** — ``keyframe_extra_correction=0.45`` lets the
   palm leap to the exact pick/place at attach/detach frames while neighbour
   frames are clipped to ``max_correction``, producing 20-32 cm single-frame
   teleports.  ``frame_repeat`` dwells on the snapped frame, making the
   pop-in/out very visible.  Fix: smooth-ramp the IK target around attach and
   detach (cosine schedule) so the palm slides into the target over a
   user-tunable window rather than snapping.

2. **Left-hand wander** — the authored clip animates both arms; the IK only
   corrects the active arm so the idle arm replays whatever the clip does
   (~58 cm AABB span in the baseline).  Fix: after IK on each frame, restore
   the inactive arm subtree's ``node_trans`` from a spawn snapshot so the
   idle hand stays put.

NOT registered anywhere.  Consumed only by the debug smoke task at
``envs/tasks/debug_improved_pickplace.py``.  Shared task and controller code
remain untouched.
"""
import numpy as np

from .styled_pick_place_motion import StyledPickPlaceMotion
from .pick_place_motion import PickPlaceMotion
from ..utils import (
    AvatarState,
    ActionStatus,
    Mirrored_Mixamo_data,
    Mixamo_data_to_controller_pose,
    Mixamo_node_processing,
)


def _smoothstep(x):
    x = float(np.clip(x, 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)


class ImprovedPickPlaceMotion(StyledPickPlaceMotion):
    """Drop-in-shaped variant of ``StyledPickPlaceMotion`` with smoother
    keyframe blending, an idle-arm freezer, and a left-hand-active variant
    that runs IK on the *mirrored* clip rather than driving the right-arm
    clip across the body."""

    def __init__(self, motion_name, motion_data, robot, name=None):
        super().__init__(motion_name, motion_data, robot, name=name)
        # Stash the originals so we can swap in mirrored frames for hand_id=0
        # without losing the right-hand path.
        self._original_data = list(self.data)
        self._original_node_data = list(self.node_data)
        self._mirrored_data = None
        self._mirrored_node_data = None
        try:
            self._build_mirrored_frames(motion_data)
        except Exception as exc:  # pragma: no cover - logged + skipped
            print(
                f"[improved_pick_place] mirrored frames unavailable: {exc!r}",
                flush=True,
            )

    def _build_mirrored_frames(self, motion_data):
        """Precompute pose vectors + node_trans for the X-mirrored clip.

        ``Mirrored_Mixamo_data`` swaps the left/right arm/finger chains and
        flips the X axis (root trans + joint quats around X).  After this
        flip, what used to be the right-arm picking motion is now performed
        by the LEFT arm in world space, with the right arm in the original
        clip's idle pose.  We use this when ``hand_id=0`` so the IK target
        and the authored animation agree on which arm is active.
        """
        mir = Mirrored_Mixamo_data(motion_data)
        vgeom = self.robot.skin.links[0]._vgeoms[0]
        self._mirrored_data = []
        self._mirrored_node_data = []
        for i in range(mir["trans"].shape[0]):
            pose = Mixamo_data_to_controller_pose(
                mir["trans"][i],
                mir["rot"][i],
                mir["joint"][i],
            )
            self._mirrored_data.append(pose)
            self._mirrored_node_data.append(
                Mixamo_node_processing(
                    vgeom, pose, self.global_mat, self.global_mat_inv
                )
            )

    def start(
        self,
        pick_pos,
        place_pos,
        attach_obj=None,
        hand_id=1,
        attach_frame=None,
        detach_frame=None,
        max_correction=0.30,
        # No special boost at the keyframe — the smooth ramp handles it.
        keyframe_extra_correction=0.0,
        transport_arc=0.10,
        refine_iters=2,
        style_residual_weight=0.55,
        palm_down=False,
        palm_down_weight=1.0,
        palm_down_mode="forearm_roll",
        palm_down_contact_only=False,
        palm_down_contact_window=18,
        frame_repeat=1,
        # Match production defaults — Inspect2's authored body lean is what
        # gives the avatar effective reach at real-task body distances (~1 m
        # forward of pick).  Locking the root kills the lean.
        lock_root_motion=False,
        lock_lower_body=False,
        # Number of frames over which the pick/place offset ramps in/out
        # around the attach and detach keyframes.  Larger => smoother but
        # longer settle time before contact.
        keyframe_blend_window=18,
        # Snapshot of inactive-arm node_trans at avatar's spawn pose is used
        # to suppress idle-arm wander when this is True.
        freeze_idle_arm=True,
        # When ``hand_id == 0`` (left-hand active), swap the clip data to the
        # X-mirrored Inspect2 frames so the LEFT arm performs the authored
        # picking arc and the RIGHT arm settles into the clip's original
        # idle pose.  This avoids the cross-body IK regression we saw with
        # the right-arm clip plus a left-palm target.
        mirror_for_left=True,
    ):
        self._keyframe_blend_window = int(max(1, keyframe_blend_window))
        self._freeze_idle_arm = bool(freeze_idle_arm)
        # Stash the spawn node_trans BEFORE super().start() mutates the
        # avatar.  We restore inactive-arm rows from this each frame.
        self._spawn_node = np.asarray(
            self.robot.node_trans, dtype=np.float64
        ).copy()
        # Cache subtree indices for the inactive arm.
        self._idle_arm_indices = self._inactive_arm_indices(int(hand_id))
        # If the user asked for a left-hand pick AND mirrored frames are
        # available, point self.data / self.node_data at the mirrored
        # arrays for the duration of super().start().  The parent reads
        # both of these inside _measure_style_palms + _apply_style_frame +
        # the IK refinement loop, then captures the final corrected frames
        # into self.frames / self.pose_frames — so restoring afterwards is
        # safe for replay.
        self._used_mirrored = (
            bool(mirror_for_left)
            and int(hand_id) == 0
            and self._mirrored_node_data is not None
            and self._mirrored_data is not None
        )
        if self._used_mirrored:
            self.data = list(self._mirrored_data)
            self.node_data = list(self._mirrored_node_data)
        else:
            self.data = list(self._original_data)
            self.node_data = list(self._original_node_data)
        try:
            super().start(
                pick_pos=pick_pos,
                place_pos=place_pos,
                attach_obj=attach_obj,
                hand_id=hand_id,
                attach_frame=attach_frame,
                detach_frame=detach_frame,
                max_correction=max_correction,
                keyframe_extra_correction=keyframe_extra_correction,
                transport_arc=transport_arc,
                refine_iters=refine_iters,
                style_residual_weight=style_residual_weight,
                palm_down=palm_down,
                palm_down_weight=palm_down_weight,
                palm_down_mode=palm_down_mode,
                palm_down_contact_only=palm_down_contact_only,
                palm_down_contact_window=palm_down_contact_window,
                frame_repeat=frame_repeat,
                lock_root_motion=lock_root_motion,
                lock_lower_body=lock_lower_body,
            )
        finally:
            # Always restore the originals so the next start() doesn't see
            # double-mirrored data.  self.frames / self.pose_frames already
            # captured the (possibly mirrored) frames during super().start().
            self.data = list(self._original_data)
            self.node_data = list(self._original_node_data)
        # Apply the idle-arm freeze to every produced frame.
        if self._freeze_idle_arm and self._idle_arm_indices:
            spawn = self._spawn_node
            for k, node in enumerate(self.frames):
                if node is None:
                    continue
                if node.shape != spawn.shape:
                    continue
                for idx in self._idle_arm_indices:
                    if idx < node.shape[0] and idx < spawn.shape[0]:
                        node[idx] = spawn[idx]

    def _inactive_arm_indices(self, active_hand_id):
        # hand_id 0 = left active => freeze RIGHT arm subtree, and vice versa.
        side = "Right" if int(active_hand_id) == 0 else "Left"
        try:
            shoulder_idx = int(self.robot.skin.node_findup[f"{side}Shoulder"])
        except Exception:
            return []
        return [int(i) for i in self._descendant_indices(shoulder_idx)]

    def _desired_palm_for_frame(
        self,
        i,
        pick,
        place,
        style_palms,
        style_pick,
        style_place,
        transport_arc,
        style_residual_weight,
    ):
        """Smooth-ramped pick/place offset around attach + detach keyframes.

        Replaces the parent's piecewise definition (which applies the FULL
        pick - style_pick offset everywhere up to the attach frame and the
        FULL place - style_place offset everywhere after detach).  Both
        endpoints jam the palm to the exact pick/place at the keyframe
        boundaries — a 20-32 cm teleport when combined with a clipped
        neighbour frame.

        Here we feather the offset with a cosine schedule centred on the
        attach and detach keyframes.  The IK still drives the palm to the
        exact pick/place at the keyframes, but the previous/next frames see
        a partial offset, so the IK targets stay close together and the
        clipping band stops generating discontinuities.
        """
        w = int(getattr(self, "_keyframe_blend_window", 18))
        a = int(self._attach_frame)
        d = int(self._detach_frame)

        # 1.0 at the attach keyframe, fading to 0 over `w` frames before it.
        def ramp_in(i, kf):
            if i >= kf:
                return 1.0
            return _smoothstep((i - (kf - w)) / float(w))

        # 1.0 at the detach keyframe, fading to 0 over `w` frames after it.
        def ramp_out(i, kf):
            if i <= kf:
                return 1.0
            return _smoothstep(((kf + w) - i) / float(w))

        if i <= a:
            alpha = ramp_in(i, a)
            return style_palms[i] + alpha * (pick - style_pick)

        if i <= d:
            denom = max(1, d - a)
            t = (i - a) / float(denom)
            source_line = style_pick * (1.0 - t) + style_place * t
            target_line = pick * (1.0 - t) + place * t
            residual = style_palms[i] - source_line
            desired = target_line + float(style_residual_weight) * residual
            if transport_arc > 0:
                desired = desired.copy()
                desired[2] += transport_arc * 4.0 * t * (1.0 - t)
            return desired

        alpha = ramp_out(i, d)
        return style_palms[i] + alpha * (place - style_place)

    def step(self, skip_avatar_animation=False):
        """Same as parent step, but enforces the idle-arm freeze AFTER
        ``self.robot.update()`` so the bone-update doesn't undo the fix."""
        if not self.frames:
            self.robot.action_state = AvatarState.NO_ACTION
            self.robot.action_status = ActionStatus.SUCCEED
            return
        super().step(skip_avatar_animation=skip_avatar_animation)
        # super().step() may have just transitioned to NO_ACTION.
        if not self._freeze_idle_arm or not self._idle_arm_indices:
            return
        nt = np.asarray(self.robot.node_trans, dtype=np.float64)
        spawn = self._spawn_node
        if nt.shape != spawn.shape:
            return
        nt = nt.copy()
        for idx in self._idle_arm_indices:
            if idx < nt.shape[0] and idx < spawn.shape[0]:
                nt[idx] = spawn[idx]
        self.robot.node_trans = nt
        self.robot.update()
