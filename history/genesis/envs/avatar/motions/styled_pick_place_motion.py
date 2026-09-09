"""Style-guided tabletop pick/place motion for the avatar.

This module keeps the existing authored motion as the arm-style prior and
uses FABRIK only to correct the palm to requested tabletop pick/place points.
It is intentionally separate from ``PickPlaceMotion`` so tasks can keep using
the proven functional path until this one is visually accepted.
"""
import numpy as np
import genesis as gs
from scipy.spatial.transform import Rotation as R

from .play_animation_motion import PlayAnimationMotion
from .pick_place_motion import PickPlaceMotion
from .palm_orient import (
    descendant_indices,
    rotate_hand_subtree_palm_down,
    rotation_between,
    roll_about_axis_toward,
)
from ..utils import AvatarState, ActionStatus


class StyledPickPlaceMotion(PlayAnimationMotion):
    """Replay an authored clip with bounded palm-target IK corrections."""

    def __init__(self, motion_name, motion_data, robot, name=None):
        super().__init__(motion_name, motion_data, robot, name=name)
        self._attach_obj = None
        self._hand_id = 1
        self._attach_frame = 0
        self._detach_frame = None
        self._corrected_node_data = []
        self.frames = []
        self.pose_frames = []
        self.attach_frame = 0
        self.detach_frame = 0
        self._debug = {}

    def start(
        self,
        pick_pos,
        place_pos,
        attach_obj=None,
        hand_id=1,
        attach_frame=None,
        detach_frame=None,
        max_correction=0.22,
        keyframe_extra_correction=0.30,
        transport_arc=0.12,
        refine_iters=2,
        style_residual_weight=0.65,
        palm_down=False,
        palm_down_weight=1.0,
        palm_down_mode="forearm_roll",
        palm_down_contact_only=False,
        palm_down_contact_window=18,
        frame_repeat=1,
        lock_root_motion=False,
        lock_lower_body=False,
    ):
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

        n = len(self.node_data)
        if n <= 1:
            gs.logger.warning(f"Cannot start {self.motion_name}: empty style clip.")
            return

        self._hand_id = int(hand_id)
        self._attach_obj = attach_obj
        self._attach_frame = (
            int(attach_frame) if attach_frame is not None else max(0, n // 4)
        )
        self._detach_frame = (
            int(detach_frame) if detach_frame is not None else min(n - 2, 3 * n // 4)
        )
        self._attach_frame = int(np.clip(self._attach_frame, 0, n - 1))
        self._detach_frame = int(np.clip(self._detach_frame, self._attach_frame, n - 1))
        self.attach_frame = self._attach_frame
        self.detach_frame = self._detach_frame

        pick = np.asarray(pick_pos, dtype=np.float64).ravel()[:3]
        place = np.asarray(place_pos, dtype=np.float64).ravel()[:3]

        saved = self._snapshot_robot_pose()
        style_palms = self._measure_style_palms(
            saved,
            root_pose=saved["pose"] if lock_root_motion else None,
        )
        style_pick = style_palms[self._attach_frame].copy()
        style_place = style_palms[self._detach_frame].copy()

        self._corrected_node_data = []
        max_err = 0.0
        clipped = 0

        hand_side = "Right" if self._hand_id == 1 else "Left"
        tmp_solver = PickPlaceMotion("_styled_pick_place_solver", self.robot)

        for i in range(n):
            self._apply_style_frame(
                i,
                root_pose=saved["pose"] if lock_root_motion else None,
            )
            tmp_solver._seed_skin_transforms()
            self.robot.update()

            desired = self._desired_palm_for_frame(
                i=i,
                pick=pick,
                place=place,
                style_palms=style_palms,
                style_pick=style_pick,
                style_place=style_place,
                transport_arc=float(transport_arc),
                style_residual_weight=float(style_residual_weight),
            )

            current = np.asarray(
                self.robot.get_palm_center(self._hand_id), dtype=np.float64
            ).ravel()[:3]
            delta = desired - current
            dist = float(np.linalg.norm(delta))
            limit = float(max_correction)
            if i in (self._attach_frame, self._detach_frame):
                limit += float(keyframe_extra_correction)
            if dist > limit > 0:
                desired = current + delta / (dist + 1e-9) * limit
                clipped += 1

            tmp_solver._hand_id = self._hand_id
            tmp_solver._idle_palm = np.asarray(
                self.robot.get_palm_center(self._hand_id), dtype=np.float64
            ).ravel()[:3]
            wrist, _ = self.robot._get_hand_ref(self._hand_id)
            tmp_solver._wrist_to_palm = tmp_solver._idle_palm - np.asarray(
                wrist, dtype=np.float64
            ).ravel()[:3]
            node = tmp_solver._solve_palm_ik(hand_side, desired, int(refine_iters))
            if lock_lower_body:
                self.robot.node_trans = node
                self._copy_lower_body_nodes(saved["node_trans"])
                self.robot.update()
                node = self.robot.node_trans.copy()
            if palm_down:
                frame_weight = float(palm_down_weight)
                if palm_down_contact_only:
                    frame_weight *= self._contact_weight(
                        i,
                        window=max(1, int(palm_down_contact_window)),
                    )
                node = self._rotate_hand_subtree_palm_down(
                    node,
                    hand_id=self._hand_id,
                    weight=frame_weight,
                    mode=str(palm_down_mode),
                )
                self.robot.node_trans = node
                self.robot.update()
            self._corrected_node_data.append(node.copy())
            self.pose_frames.append(self.robot.pose.copy())

            actual = np.asarray(
                self.robot.get_palm_center(self._hand_id), dtype=np.float64
            ).ravel()[:3]
            max_err = max(max_err, float(np.linalg.norm(actual - desired)))

        self._restore_robot_pose(saved)
        repeat = max(1, int(frame_repeat))
        if repeat > 1:
            repeated_frames = []
            repeated_poses = []
            for pose, frame in zip(self.pose_frames, self._corrected_node_data):
                repeated_poses.extend([pose] * repeat)
                repeated_frames.extend([frame] * repeat)
            self.pose_frames = repeated_poses
            self.frames = repeated_frames
            self.attach_frame = self._attach_frame * repeat
            self.detach_frame = self._detach_frame * repeat
        else:
            self.frames = self._corrected_node_data
            self.attach_frame = self._attach_frame
            self.detach_frame = self._detach_frame
        self._debug = {
            "style_pick": style_pick,
            "style_place": style_place,
            "target_pick": pick,
            "target_place": place,
            "max_palm_error": max_err,
            "clipped_frames": clipped,
            "n_frames": n,
            "palm_down": bool(palm_down),
            "palm_down_mode": str(palm_down_mode),
            "palm_down_contact_only": bool(palm_down_contact_only),
            "frame_repeat": repeat,
            "lock_root_motion": bool(lock_root_motion),
            "lock_lower_body": bool(lock_lower_body),
        }
        if clipped:
            print(
                f"[styled_pick_place] clipped {clipped}/{n} frames "
                f"(max_correction={float(max_correction):.3f}m)",
                flush=True,
            )

        self.robot.action_state = self.motion_name
        self.robot.action_status = ActionStatus.ONGOING
        self.at_frame = 0

    def step(self, skip_avatar_animation=False):
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

        self.robot.pose = self._pose_for_replay_frame(self.at_frame)
        self.robot.node_trans = self.frames[self.at_frame]
        self.robot.global_mat = self.global_mat
        self.robot.global_mat_inv = self.global_mat_inv
        # Refresh skin immediately. Attachment uses palm geometry, and waiting
        # for AvatarController.step() to call robot.update() would read the
        # previous frame's hand pose.
        self.robot.update()

        if (self._attach_obj is not None
                and self.at_frame >= self.attach_frame):
            entity = getattr(self._attach_obj, "entity", self._attach_obj)
            palm_now = np.asarray(
                self.robot.get_palm_center(self._hand_id), dtype=np.float64
            ).ravel()[:3]
            try:
                entity.set_pos(palm_now.astype(float))
            except Exception:
                pass
            self.robot.attach_object_to_hand(self._hand_id, entity)
            self._attach_obj = None

        if self.detach_frame is not None and self.at_frame >= self.detach_frame:
            self.robot.detach_object(self._hand_id)
            self.detach_frame = None

        self.at_frame += 1
        if self.at_frame >= len(self.frames):
            self.robot.action_state = AvatarState.NO_ACTION
            self.robot.action_status = ActionStatus.SUCCEED

    def _pose_for_replay_frame(self, frame_idx):
        if not self.pose_frames:
            return self.robot.pose
        n_pose = len(self.pose_frames)
        n_frame = len(self.frames)
        if n_frame == n_pose:
            return self.pose_frames[min(int(frame_idx), n_pose - 1)]

        # Existing tasks may prepend hold frames and shift attach/detach.
        delay = int(self.attach_frame) - int(self._attach_frame)
        if delay > 0 and n_frame == n_pose + delay:
            idx = max(0, int(frame_idx) - delay)
            return self.pose_frames[min(idx, n_pose - 1)]

        # Some tasks slow the avatar by repeating every frame.
        if n_pose > 0 and n_frame % n_pose == 0:
            repeat = max(1, n_frame // n_pose)
            idx = int(frame_idx) // repeat
            return self.pose_frames[min(idx, n_pose - 1)]

        # Fallback for arbitrary external frame surgery: preserve progression.
        t = int(frame_idx) / max(1, n_frame - 1)
        idx = int(round(t * (n_pose - 1)))
        return self.pose_frames[min(max(idx, 0), n_pose - 1)]

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
        if i <= self._attach_frame:
            return style_palms[i] + (pick - style_pick)

        if i <= self._detach_frame:
            denom = max(1, self._detach_frame - self._attach_frame)
            t = (i - self._attach_frame) / float(denom)
            source_line = style_pick * (1.0 - t) + style_place * t
            target_line = pick * (1.0 - t) + place * t
            residual = style_palms[i] - source_line
            desired = target_line + float(style_residual_weight) * residual
            if transport_arc > 0:
                desired = desired.copy()
                desired[2] += transport_arc * 4.0 * t * (1.0 - t)
            return desired

        return style_palms[i] + (place - style_place)

    def _contact_weight(self, i, window):
        da = abs(int(i) - int(self._attach_frame))
        dd = abs(int(i) - int(self._detach_frame))
        d = min(da, dd)
        if d >= window:
            return 0.0
        # Smoothstep: 1 at contact, 0 at window boundary.
        x = 1.0 - d / float(window)
        return x * x * (3.0 - 2.0 * x)

    def _snapshot_robot_pose(self):
        return {
            "pose": self.robot.pose.copy(),
            "node_trans": self.robot.node_trans.copy(),
            "global_mat": self.robot.global_mat.copy(),
            "global_mat_inv": self.robot.global_mat_inv.copy(),
        }

    def _restore_robot_pose(self, saved):
        self.robot.pose = saved["pose"].copy()
        self.robot.node_trans = saved["node_trans"].copy()
        self.robot.global_mat = saved["global_mat"].copy()
        self.robot.global_mat_inv = saved["global_mat_inv"].copy()
        self.robot.update()

    def _apply_style_frame(self, i, root_pose=None):
        pose = self.data[i].copy()
        if root_pose is not None:
            # Keep the authored joint rotations, but do not replay the clip's
            # root/body translation or facing.  This is useful for tabletop
            # retargeting where the avatar is already placed at a safe stance.
            pose[:7] = np.asarray(root_pose, dtype=np.float64).ravel()[:7]
        self.robot.pose = pose
        self.robot.node_trans = self.node_data[i].copy()
        self.robot.global_mat = self.global_mat
        self.robot.global_mat_inv = self.global_mat_inv

    def _copy_lower_body_nodes(self, source_node_trans):
        src = np.asarray(source_node_trans, dtype=np.float64)
        try:
            hips_idx = int(self.robot.skin.node_findup["Hips"])
            if hips_idx < self.robot.node_trans.shape[0] and hips_idx < src.shape[0]:
                self.robot.node_trans[hips_idx] = src[hips_idx]
        except Exception:
            pass
        for root_name in ("LeftUpLeg", "RightUpLeg"):
            try:
                root_idx = int(self.robot.skin.node_findup[root_name])
            except Exception:
                continue
            for idx in self._descendant_indices(root_idx):
                if idx < self.robot.node_trans.shape[0] and idx < src.shape[0]:
                    self.robot.node_trans[idx] = src[idx]

    def _measure_style_palms(self, saved, root_pose=None):
        palms = []
        for i in range(len(self.node_data)):
            self._apply_style_frame(i, root_pose=root_pose)
            self.robot.update()
            palms.append(
                np.asarray(
                    self.robot.get_palm_center(self._hand_id), dtype=np.float64
                ).ravel()[:3].copy()
            )
        self._restore_robot_pose(saved)
        return np.asarray(palms, dtype=np.float64)

    def _rotate_hand_subtree_palm_down(
        self, node_trans, hand_id, weight=1.0, mode="forearm_roll"
    ):
        """Rotate hand/finger nodes so the palm normal points toward world -Z.

        Delegates to the shared implementation in ``palm_orient`` (also used
        by the legacy ``PickPlaceMotion`` path).
        """
        return rotate_hand_subtree_palm_down(
            self.robot, node_trans, hand_id, weight=weight, mode=mode
        )

    def _descendant_indices(self, root_idx):
        return descendant_indices(self.robot.skin, root_idx)

    def _rotation_between(self, src, dst):
        return rotation_between(src, dst)

    def _roll_about_axis_toward(self, src, dst, axis):
        return roll_about_axis_toward(src, dst, axis)
