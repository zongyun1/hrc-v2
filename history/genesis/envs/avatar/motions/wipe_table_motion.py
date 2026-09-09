"""Wipe-table avatar motion.

Replays a body-posture motion (default: ``WipeTable`` from motion.pkl, sliced
from mouse.blend frames 225-310) while overriding the right-arm IK chain
per-frame so the right palm tracks a user-supplied trajectory.

The body keeps the leaning-over-the-counter posture from the source motion
(legs / torso / left arm / head all unchanged), only the right arm bends to
reach the wiping target. This is the avatar-side primitive for the
``counter_wiping_assistance`` task.

The IK helpers (`_seed_skin_transforms`, `_solve_palm_ik`, `_ik_step`) are
duplicated from `pick_place_motion.py` rather than imported / shared, so
this module does not modify any cross-task code path.
"""
from __future__ import annotations

import numpy as np
import genesis as gs
import genesis.utils.geom as geom_utils
from scipy.spatial.transform import Rotation as R

from .base_motion_module import BaseMotionModule
from ..utils import (
    AvatarState,
    ActionStatus,
    MIXAMO_JOINT_NUM,
    MIXAMO_SKIN_JOINT_NUM,
    Mixamo_data_to_controller_pose,
    Mixamo_node_processing,
)


class WipeTableMotion(BaseMotionModule):
    """Replays a body-posture motion + IK overlay on the right arm.

    Lifecycle:
        m = WipeTableMotion("wipe_table", avatar.robot)
        m.start(body_motion_data, palm_targets, attach_obj=sponge_entity)
        avatar.motion_modules["wipe_table"] = m
        avatar.robot.action_state = "wipe_table"
        # ... task loop calls step_sim() which dispatches to m.step()
    """

    def __init__(self, motion_name: str, robot):
        super().__init__(motion_name, robot)
        self.body_poses: list[np.ndarray] = []
        self.frames: list[np.ndarray] = []   # node_trans per frame, post-IK
        self.global_mat: np.ndarray | None = None
        self.global_mat_inv: np.ndarray | None = None
        self.at_frame: int = 0

        self._attach_obj = None
        self._attach_hand_id: int = 1
        self._wrist_to_palm: np.ndarray = np.zeros(3)

        # XY-only sponge follow: per-step the wipe motion pins the
        # sponge xy to (palm.xy + offset), z to a fixed counter height,
        # and orientation to a fixed "facing-down" quaternion. Avoids
        # the kinematic attach_object_to_hand, which made the sponge
        # rotate with the palm and flip its wiping face up. None ⇒
        # disabled (caller didn't pass a sponge).
        self._sponge_entity = None
        self._sponge_xy_offset: np.ndarray = np.zeros(2)
        self._sponge_z: float | None = None
        self._sponge_quat: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(
        self,
        body_motion_data: dict,
        palm_targets: np.ndarray,
        attach_obj=None,
        hand_id: int = 1,
        refine_iters: int = 2,
        step_repeat: int = 1,
        sponge_entity=None,
        sponge_xy_offset=(0.0, 0.0),
        sponge_z: float | None = None,
        sponge_quat: np.ndarray | None = None,
    ):
        """Plan all frames up-front, then replay them via ``step()``.

        Args:
            body_motion_data: dict with keys ``trans`` ``rot`` ``joint``
                ``mat`` (the format used inside ``motion.pkl``).
                The body trajectory (head, torso, legs, left arm) is taken
                from this. The motion is resampled to len(palm_targets) via
                nearest-frame interpolation.
            palm_targets: array of shape ``(T, 3)`` — world xyz targets for
                the right palm at each replay frame.
            attach_obj: optional Genesis entity to kinematic-attach to the
                right hand at frame 0 (e.g. a sponge).
            hand_id: 0 = left, 1 = right. Default right (the avatar's
                wiping arm).
            refine_iters: refinement passes inside the IK solver per frame.
            step_repeat: hold each motion frame for this many sim ticks
                (= video frames). 1 = full speed (~0.17 sec sim per
                86-frame cycle), 8 = ~1.4 sec sim per cycle (visible
                back-and-forth in the rasterizer video).
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

        palm_targets = np.asarray(palm_targets, dtype=np.float64).reshape(-1, 3)
        T = palm_targets.shape[0]
        if T == 0:
            gs.logger.warning(f"{self.motion_name}: palm_targets is empty")
            return

        body_T = int(body_motion_data["trans"].shape[0])
        # Ping-pong loop the body cycle: 0,1,…,body_T-1, body_T-2,…,1,0,1,…
        # A plain `% body_T` jumps from frame body_T-1 back to 0 each cycle,
        # which is visible as a "snap" in the body posture when frame 0 and
        # frame body_T-1 don't match (mouse.blend WipeTable doesn't loop
        # closure-perfectly). Ping-pong avoids that by reversing direction
        # at the endpoints.
        if body_T > 1:
            period = 2 * (body_T - 1)
            in_period = np.arange(T, dtype=np.int64) % period
            idx = np.where(
                in_period < body_T,
                in_period,
                period - in_period,
            ).astype(int)
        else:
            idx = np.zeros(T, dtype=int)

        body_trans = body_motion_data["trans"][idx]
        body_rot = body_motion_data["rot"][idx]
        body_joint = body_motion_data["joint"][idx]
        body_mat = body_motion_data["mat"][idx]

        self.global_mat = body_mat[0]
        self.global_mat_inv = np.array([np.linalg.inv(m) for m in self.global_mat])

        # `_seed_skin_transforms` and `_solve_palm_ik` read the per-bone
        # coord matrices off `self.robot`. Push the WipeTable mats now so
        # the precompute loop below sees the right basis from the first
        # iteration.
        self.robot.global_mat = self.global_mat
        self.robot.global_mat_inv = self.global_mat_inv

        vgeom = self.robot.skin.links[0]._vgeoms[0]

        # Pre-build body-only controller poses + node_trans (these are
        # what the IK overlay starts from each frame).
        self.body_poses = []
        body_node_trans: list[np.ndarray] = []
        for i in range(T):
            pose_i = Mixamo_data_to_controller_pose(
                body_trans[i], body_rot[i], body_joint[i],
            )
            nt_i = Mixamo_node_processing(
                vgeom, pose_i, self.global_mat, self.global_mat_inv,
            )
            self.body_poses.append(pose_i)
            body_node_trans.append(nt_i)

        # Calibrate wrist→palm offset from the first body frame's rest pose.
        hand_side = "Right" if hand_id == 1 else "Left"
        self.robot.pose = self.body_poses[0]
        self.robot.node_trans = body_node_trans[0]
        self._seed_skin_transforms()
        rest_palm = np.asarray(
            self.robot.get_palm_center(hand_id), dtype=np.float64
        ).ravel()[:3]
        rest_wrist, _ = self.robot._get_hand_ref(hand_id)
        rest_wrist = np.asarray(rest_wrist, dtype=np.float64).ravel()[:3]
        self._wrist_to_palm = rest_palm - rest_wrist

        # Pre-compute IK overlay for every frame.
        self.frames = []
        for i in range(T):
            self.robot.pose = self.body_poses[i]
            self.robot.node_trans = body_node_trans[i]
            self._seed_skin_transforms()
            new_nt = self._solve_palm_ik(
                hand_side, palm_targets[i], hand_id, refine_iters,
            )
            self.frames.append(new_nt.copy())

        # Hold each motion frame for step_repeat sim ticks so the wiping
        # back-and-forth is visible at video rate (sim runs at 500 Hz so
        # an 86-frame cycle at step_repeat=1 lasts only 0.17 s sim).
        if step_repeat > 1:
            sr = int(step_repeat)
            self.frames = [f for f in self.frames for _ in range(sr)]
            self.body_poses = [b for b in self.body_poses for _ in range(sr)]

        self.at_frame = 0
        self._attach_obj = attach_obj
        self._attach_hand_id = hand_id

        # XY-only sponge follow: caller must pass an existing entity (the
        # sponge is NOT auto-loaded). z and quat are world-frame; xy is
        # offset from the per-step palm xy.
        self._sponge_entity = sponge_entity
        self._sponge_xy_offset = np.asarray(sponge_xy_offset, dtype=np.float64).reshape(2)
        self._sponge_z = (None if sponge_z is None else float(sponge_z))
        self._sponge_quat = (
            None if sponge_quat is None
            else np.asarray(sponge_quat, dtype=np.float64).reshape(4)
        )

        self.robot.action_state = self.motion_name
        self.robot.action_status = ActionStatus.ONGOING

    def step(self, skip_avatar_animation: bool = False):
        """One sim-step advance. Called by ``AvatarController.step()``."""
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

        self.robot.pose = self.body_poses[self.at_frame]
        self.robot.node_trans = self.frames[self.at_frame]
        self.robot.global_mat = self.global_mat
        self.robot.global_mat_inv = self.global_mat_inv

        # Attach at frame 0 (sponge stays in palm for the whole motion).
        if self._attach_obj is not None and self.at_frame == 0:
            entity = getattr(self._attach_obj, "entity", self._attach_obj)
            self.robot.attach_object_to_hand(self._attach_hand_id, entity)
            self._attach_obj = None

        # XY-only sponge follow. Pin z + orientation to the values caller
        # supplied; only let xy track the palm. This keeps the wiping
        # face pointing down and at a constant counter-skim height even
        # when the body / palm rotate during the wipe.
        if self._sponge_entity is not None:
            palm = np.asarray(
                self.robot.get_palm_center(self._attach_hand_id),
                dtype=np.float64,
            ).ravel()[:3]
            sx = float(palm[0] + self._sponge_xy_offset[0])
            sy = float(palm[1] + self._sponge_xy_offset[1])
            sz = (float(self._sponge_z) if self._sponge_z is not None
                  else float(palm[2]))
            try:
                self._sponge_entity.set_pos(np.array([sx, sy, sz], dtype=np.float64))
                if self._sponge_quat is not None:
                    self._sponge_entity.set_quat(self._sponge_quat)
            except Exception:
                pass

        self.at_frame += 1

        if self.at_frame >= len(self.frames):
            self.robot.action_state = AvatarState.NO_ACTION
            self.robot.action_status = ActionStatus.SUCCEED

    # ------------------------------------------------------------------
    # IK helpers — duplicated from pick_place_motion.py so this module
    # does not introduce a cross-task dependency.
    # ------------------------------------------------------------------

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
        mixamo_pose[:MIXAMO_SKIN_JOINT_NUM] = (
            geom_utils.R_to_quat(result)[:, [1, 2, 3, 0]]
        )

        total_trans = (self.robot.global_rot @ self.robot.base_rot @ motion_trans
                       + self.robot.global_trans)
        base_trans = total_trans[[1, 2, 0]]
        self.robot.skin.calculate_real_pos(mixamo_pose, base_trans)

    def _solve_palm_ik(
        self, hand_side: str, palm_target: np.ndarray, hand_id: int,
        refine_iters: int,
    ) -> np.ndarray:
        wrist_target = palm_target - self._wrist_to_palm
        self._ik_step(hand_side, wrist_target)
        for _ in range(refine_iters):
            actual = np.asarray(
                self.robot.get_palm_center(hand_id), dtype=np.float64
            ).ravel()[:3]
            err = actual - palm_target
            if float(np.linalg.norm(err)) < 1e-3:
                break
            wrist_target = wrist_target - err
            self._ik_step(hand_side, wrist_target)
        return self.robot.node_trans.copy()

    def _ik_step(self, hand_side: str, wrist_target_world: np.ndarray):
        skin = self.robot.skin
        new_nt = skin.ik_solve(
            hand_id=f"{hand_side}Hand",
            root_id=f"{hand_side}Shoulder",
            target_pos=np.asarray(wrist_target_world, dtype=np.float64).copy(),
            max_iterations=50,
            tolerance=1e-4,
        )
        self.robot.node_trans = new_nt
        self.robot.update()
