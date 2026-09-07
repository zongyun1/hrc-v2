"""Avatar drawer-pull motion.

Specialized variant of PickPlaceMotion for opening small cabinet drawers.
It drives the palm from the live pose to the closed handle, then to the
open handle, then retracts.  To keep CPU smoke runs practical, IK is solved
only at key poses and the solved skin transforms are blended for playback.
"""

import numpy as np

from .pick_place_motion import PickPlaceMotion
from ..utils import AvatarState, ActionStatus


class DrawerPullMotion(PickPlaceMotion):
    def __init__(self, motion_name, robot):
        super().__init__(motion_name, robot)
        self.pull_start_frame = 0
        self.pull_end_frame = 0

    def start(self, handle_start, handle_end, hand_id=0,
              approach_frames=30, pull_frames=50, retract_frames=30,
              retract_offset=(0.12, 0.02, 0.0), refine_iters=1):
        if self.robot.action_state != AvatarState.NO_ACTION:
            return
        if self.robot.base_state != AvatarState.STANDING:
            return

        self._hand_id = int(hand_id)
        self._attach_obj = None

        start_palm = np.asarray(
            self.robot.get_palm_center(self._hand_id), dtype=np.float64,
        ).ravel()[:3]
        handle_start = np.asarray(handle_start, dtype=np.float64).ravel()[:3]
        handle_end = np.asarray(handle_end, dtype=np.float64).ravel()[:3]
        retract = handle_end + np.asarray(retract_offset, dtype=np.float64).ravel()[:3]

        self._seed_skin_transforms()
        rest_palm = np.asarray(
            self.robot.get_palm_center(self._hand_id), dtype=np.float64,
        ).ravel()[:3]
        rest_wrist, _ = self.robot._get_hand_ref(self._hand_id)
        rest_wrist = np.asarray(rest_wrist, dtype=np.float64).ravel()[:3]
        self._idle_palm = rest_palm
        self._wrist_to_palm = rest_palm - rest_wrist

        side = "Left" if self._hand_id == 0 else "Right"
        key_start = self._solve_palm_ik(side, start_palm, refine_iters)
        key_handle = self._solve_palm_ik(side, handle_start, refine_iters)
        key_open = self._solve_palm_ik(side, handle_end, refine_iters)
        key_retract = self._solve_palm_ik(side, retract, refine_iters)

        frames = []
        frames.extend(self._blend_nodes(key_start, key_handle, approach_frames))
        self.pull_start_frame = len(frames)
        frames.extend(self._blend_nodes(key_handle, key_open, pull_frames))
        self.pull_end_frame = max(self.pull_start_frame, len(frames) - 1)
        frames.extend(self._blend_nodes(key_open, key_retract, retract_frames))

        self.frames = frames
        self.attach_frame = self.pull_start_frame
        self.detach_frame = self.pull_end_frame
        self.at_frame = 0
        self.robot.action_state = self.motion_name
        self.robot.action_status = ActionStatus.ONGOING

    def pull_progress(self) -> float:
        denom = max(1, int(self.pull_end_frame) - int(self.pull_start_frame))
        return float(np.clip((int(self.at_frame) - int(self.pull_start_frame)) / denom, 0.0, 1.0))

    def _blend_nodes(self, a, b, n_steps):
        n_steps = max(1, int(n_steps))
        out = []
        for i in range(1, n_steps + 1):
            t = i / float(n_steps)
            s = t * t * (3.0 - 2.0 * t)
            out.append((1.0 - s) * np.asarray(a) + s * np.asarray(b))
        return out
