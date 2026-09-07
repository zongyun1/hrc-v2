"""Play animation by name from motion_data."""
import numpy as np
from .replay_motion_module import ReplayMotionModule
from ..utils import AvatarState, ActionStatus, Mirrored_Mixamo_data, Mixamo_data_to_controller_pose
import genesis as gs


class PlayAnimationMotion(ReplayMotionModule):
    def __init__(self, motion_name, motion_data, robot, name=None):
        super().__init__(motion_name, motion_data, robot, name)
        self._attach_obj = None
        self._attach_hand_id = 1
        self._attach_mode = "hand"
        self._attach_frame = len(self.data) - 1
        self._detach_frame = None
        self._two_hand_offset = np.zeros(3, dtype=np.float64)
        self._two_hand_snap = False
        self.mirrored_data = []
        mirrored_motion_data = Mirrored_Mixamo_data(motion_data)
        for i in range(mirrored_motion_data["trans"].shape[0]):
            self.mirrored_data.append(
                Mixamo_data_to_controller_pose(
                    mirrored_motion_data["trans"][i],
                    mirrored_motion_data["rot"][i],
                    mirrored_motion_data["joint"][i],
                )
            )

    def start(self, attach_obj=None, hand_id=1, attach_frames_early=0,
              attach_frame=None, detach_frame=None, attach_mode="hand",
              two_hand_offset=None, two_hand_snap=False):
        """Start motion. If attach_obj is set, attach it to hand at a chosen frame.

        attach_frames_early: attach this many frames before the last frame (default 0 = last frame).
        attach_frame: absolute attach frame index; overrides attach_frames_early when set.
        detach_frame: absolute detach frame index; if set, detaches at this frame.
        """
        if self.robot.action_state != AvatarState.NO_ACTION:
            gs.logger.warning(f"Cannot start motion {self.motion_name}: AvatarState is {self.robot.action_state}.")
            return
        if self.robot.base_state != AvatarState.STANDING:
            gs.logger.warning(f"Cannot start motion {self.motion_name}: BaseState is {self.robot.base_state}")
            return
        self.robot.action_state = self.motion_name
        self.robot.action_status = ActionStatus.ONGOING
        self.at_stage = 0
        self.at_frame = 0
        self._attach_obj = attach_obj
        self._attach_hand_id = hand_id
        self._attach_mode = attach_mode
        self._two_hand_offset = (
            np.zeros(3, dtype=np.float64)
            if two_hand_offset is None
            else np.asarray(two_hand_offset, dtype=np.float64)
        )
        self._two_hand_snap = bool(two_hand_snap)
        if attach_frame is not None:
            self._attach_frame = max(0, int(attach_frame))
        else:
            self._attach_frame = max(0, len(self.data) - 1 - attach_frames_early)
        self._detach_frame = int(detach_frame) if detach_frame is not None else None

    def step(self, skip_avatar_animation=False):
        self.at_frame += 1
        if skip_avatar_animation:
            self.at_frame = len(self.data) - 1
        self.at_frame = min(self.at_frame, len(self.data) - 1)
        # Attach object at the designated frame
        attach_frame = getattr(self, "_attach_frame", len(self.data) - 1)
        if self.at_frame >= attach_frame and getattr(self, "_attach_obj", None) is not None:
            entity = getattr(self._attach_obj, "actor", self._attach_obj)
            if self._attach_mode in ("two_hand", "both_hands"):
                self.robot.attach_object_to_two_hands(
                    entity, self._two_hand_offset, snap_to_frame=self._two_hand_snap
                )
            else:
                self.robot.attach_object_to_hand(self._attach_hand_id, entity)
            self._attach_obj = None
        # Detach object at the designated frame
        if getattr(self, "_detach_frame", None) is not None and self.at_frame >= self._detach_frame:
            if self._attach_mode in ("two_hand", "both_hands"):
                self.robot.detach_object_from_two_hands()
            else:
                self.robot.detach_object(self._attach_hand_id)
            self._detach_frame = None
        self.robot.pose = self.data[self.at_frame]
        self.robot.node_trans = self.node_data[self.at_frame]
        self.robot.global_mat = self.global_mat
        self.robot.global_mat_inv = self.global_mat_inv
        if self.at_frame == len(self.data) - 1:
            self.robot.action_state = AvatarState.NO_ACTION
            self.robot.action_status = ActionStatus.SUCCEED
