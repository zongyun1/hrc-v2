"""Smooth pose-to-pose transition (e.g. motion end → idle, or idle → motion start).

Bridges between two avatar states by interpolating `robot.pose` (root trans,
root quat, per-joint quats) with SLERP for all rotations, plus a linear blend of
`global_mat` / `global_mat_inv`.  `node_trans` is rebuilt each frame from the
interpolated pose so the skin mesh stays consistent with the blended quats.
"""
import numpy as np

from .base_motion_module import BaseMotionModule
from ..utils import AvatarState, ActionStatus, Mixamo_node_processing


def _slerp(q0, q1, t):
    """Spherical linear interpolation of unit quaternions (w, x, y, z)."""
    q0 = np.asarray(q0, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        out = q0 + t * (q1 - q0)
        n = np.linalg.norm(out)
        return out / n if n > 1e-12 else q0
    theta_0 = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_theta_0 = np.sin(theta_0)
    theta = theta_0 * t
    s0 = np.cos(theta) - dot * np.sin(theta) / sin_theta_0
    s1 = np.sin(theta) / sin_theta_0
    return s0 * q0 + s1 * q1


class TransitionMotion(BaseMotionModule):
    """Blend the avatar's current pose to a target pose over `frames` steps."""

    def __init__(self, robot, target_pose, target_node, target_mat, target_mat_inv, frames):
        super().__init__("transition", robot)
        self.target_pose = np.asarray(target_pose, dtype=np.float64).copy()
        self.target_node = np.asarray(target_node).copy() if len(target_node) else target_node
        self.target_mat = np.asarray(target_mat).copy()
        self.target_mat_inv = np.asarray(target_mat_inv).copy()
        self.n_frames = int(max(1, frames))
        self.at_frame = 0
        self.start_pose = None
        self.start_mat = None
        self.start_mat_inv = None

    def start(self):
        if self.robot.action_state != AvatarState.NO_ACTION:
            import genesis as gs
            gs.logger.warning(
                f"Cannot start transition: AvatarState is {self.robot.action_state}."
            )
            return
        self.start_pose = np.asarray(self.robot.pose, dtype=np.float64).copy()
        self.start_mat = np.asarray(self.robot.global_mat).copy()
        self.start_mat_inv = np.asarray(self.robot.global_mat_inv).copy()
        self.at_frame = 0
        self.robot.action_state = self.motion_name
        self.robot.action_status = ActionStatus.ONGOING

    def step(self, skip_avatar_animation=False):
        self.at_frame += 1
        if skip_avatar_animation:
            self.at_frame = self.n_frames

        if self.at_frame >= self.n_frames:
            self.robot.pose = self.target_pose
            self.robot.node_trans = self.target_node
            self.robot.global_mat = self.target_mat
            self.robot.global_mat_inv = self.target_mat_inv
            self.robot.action_state = AvatarState.NO_ACTION
            self.robot.action_status = ActionStatus.SUCCEED
            return

        t = self.at_frame / self.n_frames

        interp = np.empty_like(self.target_pose)
        interp[:3] = (1 - t) * self.start_pose[:3] + t * self.target_pose[:3]
        interp[3:7] = _slerp(self.start_pose[3:7], self.target_pose[3:7], t)
        n_joint_quats = (len(self.target_pose) - 7) // 4
        for j in range(n_joint_quats):
            a = 7 + 4 * j
            interp[a:a + 4] = _slerp(self.start_pose[a:a + 4], self.target_pose[a:a + 4], t)
        self.robot.pose = interp

        mat_blend = (1 - t) * self.start_mat + t * self.target_mat
        mat_inv_blend = (1 - t) * self.start_mat_inv + t * self.target_mat_inv
        self.robot.global_mat = mat_blend
        self.robot.global_mat_inv = mat_inv_blend

        vgeom = self.robot.skin.links[0]._vgeoms[0]
        self.robot.node_trans = Mixamo_node_processing(
            vgeom, interp, mat_blend, mat_inv_blend
        )
