"""Stamp documents: the avatar stamps two documents while the arm feeds them.

Robot: physics-based push only (no grasp/attach).  Flow:
  avatar stamps doc 1 -> arm clears it -> arm feeds doc 2 -> avatar stamps doc 2.

Avatar: plays `stamp_crop` once per doc with the seal attached to the
right palm.  Ink pad sits at the dip XY measured by a dry-run calibration
in `reset()`.
"""

import os
import time
from pathlib import Path
import numpy as np
import transforms3d as t3d

from ..avatar.utils import ActionStatus, AvatarState
from ..base_task import BaseTask, TABLE_HEIGHT
from ..grasp import tcp_to_link_pose
from ..robot.base import _get_dof_idx
from ..utils import Pose, create_primitive, load_mesh, load_object, to_numpy


_TOP_DOWN_R = np.array([
    [1.0,  0.0,  0.0],
    [0.0, -1.0,  0.0],
    [0.0,  0.0, -1.0],
])
_TOP_DOWN_Q = t3d.quaternions.mat2quat(_TOP_DOWN_R)
_TOP_DOWN_R_Y_PUSH = t3d.euler.euler2mat(0.0, 0.0, np.pi / 2.0, axes="sxyz") @ _TOP_DOWN_R
_TOP_DOWN_Q_Y_PUSH = t3d.quaternions.mat2quat(_TOP_DOWN_R_Y_PUSH)
_IDENTITY_Q = np.array([1.0, 0.0, 0.0, 0.0])

# Doc shape (10×10×2.5 cm) and feeder slots, all proven reachable by the
# right arm's screw planner.
_DOC_HALF = (0.05, 0.05, 0.0125)
_DOC_FEEDER_XY = [(0.12, -0.15), (0.22, -0.15), (0.32, -0.15)]

# Per-seed jitter limits.  X column shift is shared across docs (preserves
# push geometry); Y is per-doc.
# Doc size ±12% keeps half_z above stamp-mark thickness and half_x/y above
# stamp radius (0.0266 m).  Avatar pos+yaw are small enough that the
# calibrated stamp_xy stays in right-arm reach.
_FEEDER_X_JITTER = 0.02
_FEEDER_Y_JITTER = 0.015
_DOC_COUNT = 2
_DOC_SIZE_SCALE_RANGE = (0.88, 1.12)
_AVATAR_POS_JITTER_XY = 0.03
_AVATAR_YAW_JITTER_DEG = 5.0

_DOC_COLOR_POOL = [
    (0.95, 0.92, 0.85),   # cream
    (0.96, 0.96, 0.92),   # off-white
    (0.88, 0.86, 0.80),   # warm beige
    (0.92, 0.90, 0.88),   # linen
    (0.94, 0.88, 0.78),   # manila
]

_STAMP_MOTION = "stamp_crop"     # one dip-stamp cycle, replayed per doc
_STAMP_HAND_ID = 1
_AVATAR_FRAME_RATIO = 12.5       # ~5× slower than default so dips read clearly

# 100_seal mesh: body length 9.6 cm along local +Y, origin at the −Y end.
# Setting pos to palm − body_length·ẑ puts the +Y handle at the palm.
_STAMP_BODY_LEN = 0.0964

# Stamp base disc: x/z half-extent 0.5313 · scale(0.05) = 0.0266 m.
_STAMP_MARK_RADIUS = 0.0266
_STAMP_MARK_HALF_H = 0.00025
_PUSH_GRIPPER_VAL = 0.35
_FIRST_DOC_PUSH_EDGE_FRACTION = 0.8

_INKPAD_OBJ = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "output", "inkpad", "inkpad.obj",
)


class StampDocuments(BaseTask):
    """Arm pushes docs into a stamp zone; avatar dips & stamps each in turn."""

    INSTRUCTION = "push each document into the stamp zone so the human can stamp it"
    use_avatar = True
    # Avatar shifted +0.30 X / +0.11 Z so the motion's stamp/dip positions
    # (avatar-local x ≈ −0.17 for right hand) map into right-arm reach.
    avatar_init_pos = np.array([0.30, 0.89, -0.18 + 0.11])
    avatar_init_rot = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=np.float64)

    # Capture 1-in-6 keeps the 30 fps video at sim dt=0.002 (12 ms/frame).
    VIDEO_STRIDE = 6
    recording_camera_pos = [1.1, 0.0, 2.05]
    recording_camera_lookat = [0.0, 0.3, 0.85]

    # Wider table for the Franka side while preserving the avatar-side
    # workspace used by the stamp and ink-pad calibration.
    TABLE_HALF_SIZE = (0.60, 0.65)
    TABLE_CENTER_XY = (0.0, -0.25)
    TABLE_THICKNESS_VALUE = 0.05
    TABLE_LEG_RADIUS = 0.025

    # Avatar arm capsule pushed to the planner so screw plans route around it.
    _ARM_CAPSULE_RADIUS = 0.06
    _ARM_CAPSULE_N_ALONG = 14
    _ARM_CAPSULE_N_AROUND = 6

    # Filled by `_calibrate_motion_key_frames` and `load_actors`.
    _stamp_xy = np.array([0.10, 0.10])
    _ink_xy = np.array([-0.10, 0.30])
    _doc_half = _DOC_HALF

    # Avatar control flags.
    _stamp_ready = False     # set True after `_attach_stamp_to_palm`
    _freeze_avatar = False   # if True, avatar.update() instead of avatar.step()

    def __init__(self, config: dict = None):
        cfg = dict(config or {})
        cfg.setdefault("robot_type", "franka")
        cfg.setdefault("robot_single_arm", True)
        super().__init__(cfg)

    def _create_table(self, table_height=TABLE_HEIGHT):
        # Y span [-0.90, +0.40]: enough support for the default Franka base
        # at y=-0.65 without shortening the avatar-side stamping area.
        self.TABLE_HEIGHT = table_height
        self.TABLE_THICKNESS = self.TABLE_THICKNESS_VALUE
        self.TABLE_TOP_Z = table_height + self.TABLE_THICKNESS / 2
        self.table = create_primitive(
            self.scene,
            "box",
            Pose(p=[self.TABLE_CENTER_XY[0], self.TABLE_CENTER_XY[1], table_height]),
            size={"half_size": (
                self.TABLE_HALF_SIZE[0],
                self.TABLE_HALF_SIZE[1],
                self.TABLE_THICKNESS / 2,
            )},
            color=(0.80, 0.75, 0.65),
            is_static=True,
        )
        self._remember_table_bounds(
            half_size=self.TABLE_HALF_SIZE,
            center_xy=self.TABLE_CENTER_XY,
        )
        leg_h = table_height - self.TABLE_THICKNESS / 2
        leg_x = self.TABLE_HALF_SIZE[0] - 0.05
        leg_y = self.TABLE_HALF_SIZE[1] - 0.05
        cx, cy = self.TABLE_CENTER_XY
        for x, y in (
            (cx - leg_x, cy - leg_y),
            (cx + leg_x, cy - leg_y),
            (cx - leg_x, cy + leg_y),
            (cx + leg_x, cy + leg_y),
        ):
            create_primitive(
                self.scene,
                "cylinder",
                Pose(p=[x, y, leg_h / 2]),
                size={"radius": self.TABLE_LEG_RADIUS, "half_length": leg_h / 2},
                color=(0.5, 0.5, 0.5),
                is_static=True,
            )

    # ------------------------------------------------------------------
    # Sim step
    # ------------------------------------------------------------------
    def step_sim(self):
        self.scene.step()
        if self.robot is not None:
            self.robot.on_post_step(self.scene)
        self._sync_gripper_attached()
        if self.avatar is not None:
            if self._freeze_avatar:
                self.avatar.robot.update()
            else:
                self.avatar.step()
        self._rigid_vertical_stamp()
        self._update_stamp_marks()
        if self.avatar_collider is not None:
            self.avatar_collider.update()
        if self.avatar_collision_checker is not None or self.avatar_safety_checker is not None:
            self._avatar_collision_tick += 1
            if self._avatar_collision_tick >= self._avatar_collision_stride:
                self._avatar_collision_tick = 0
                if self.avatar_collision_checker is not None:
                    result = self.avatar_collision_checker.check()
                    result["frame"] = self.FRAME_IDX
                    self.avatar_collision_log.append(result)
                    if result["collided"] and not self.avatar_collided:
                        self.avatar_collided = True
                if self.avatar_safety_checker is not None:
                    result = self.avatar_safety_checker.check()
                    result["frame"] = self.FRAME_IDX
                    self.avatar_safety_log.append(result)
        if self.vla_recorder is not None:
            self.vla_recorder.tick(self)
        self.capture_frame()
        if self._record_stride is not None:
            if self._record_tick % self._record_stride == 0:
                self.record_frame()
            self._record_tick += 1

    def _update_stamp_marks(self):
        """Sync each attached red ink-mark pose to its host doc."""
        if not getattr(self, "stamp_marks", None):
            return
        for i, mark in enumerate(self.stamp_marks):
            doc = self._stamp_mark_attached[i]
            offset = self._stamp_mark_local_offset[i]
            if doc is None or offset is None:
                continue
            doc_pos = to_numpy(doc.get_pos()).ravel()
            doc_quat = to_numpy(doc.get_quat()).ravel()
            R = t3d.quaternions.quat2mat(doc_quat)
            mark.set_pos((doc_pos + R @ offset).astype(np.float64))
            mark.set_quat(doc_quat.astype(np.float64))

    def _attach_stamp_mark(self, doc_idx: int, doc, mark_xy,
                           doc_pos_override=None, doc_quat_override=None):
        """Snap mark `doc_idx` at the stamp's world XY on the doc's top face,
        store the local offset for tracking, and record whether the stamp
        landed inside the doc footprint (consumed by `check_success`).

        `doc_pos_override` / `doc_quat_override`: the doc's pose snapshotted at
        the contact instant.  The rigid seal drags the doc as the palm retracts,
        so a live `get_pos()` read here sees the doc already pulled away from
        where the stamp actually pressed; the offset must be computed against
        the doc pose at the same instant the seal XY was sampled.
        """
        if doc_pos_override is not None:
            doc_pos = np.asarray(doc_pos_override, dtype=np.float64).ravel()[:3]
        else:
            doc_pos = to_numpy(doc.get_pos()).ravel()
        if doc_quat_override is not None:
            doc_quat = np.asarray(doc_quat_override, dtype=np.float64).ravel()[:4]
        else:
            doc_quat = to_numpy(doc.get_quat()).ravel()
        top_z = doc_pos[2] + self._doc_half[2] + _STAMP_MARK_HALF_H + 0.0005
        world_pos = np.array([float(mark_xy[0]), float(mark_xy[1]), top_z])
        R = t3d.quaternions.quat2mat(doc_quat)
        local_offset = R.T @ (world_pos - doc_pos)
        self._stamp_mark_attached[doc_idx] = doc
        self._stamp_mark_local_offset[doc_idx] = local_offset
        self.stamp_marks[doc_idx].set_pos(world_pos.astype(np.float64))
        self.stamp_marks[doc_idx].set_quat(doc_quat.astype(np.float64))
        # On-doc: count a mark when the stamp disc visibly overlaps the
        # document.  The avatar stamps near the paper edge, so requiring the
        # disc centre to be strictly inside rejects valid-looking stamps.
        margin = 2.0 * _STAMP_MARK_RADIUS
        on_doc = bool(
            abs(local_offset[0]) <= self._doc_half[0] + margin
            and abs(local_offset[1]) <= self._doc_half[1] + margin
        )
        self._stamp_within_doc[doc_idx] = on_doc

    def _rigid_vertical_stamp(self):
        """Hold the stamp base pointing straight down at palm_z − body_len.

        The Mixamo motion's hand pose isn't perfectly palm-down at the
        stamp frame, so a rigid attach would leave the stamp tilted.  We
        override pos+quat every sim step instead.
        """
        if not self._stamp_ready or self.avatar is None:
            return
        palm = np.asarray(
            self.avatar.robot.get_palm_center(_STAMP_HAND_ID),
            dtype=np.float64,
        ).ravel()
        origin_pos = palm - np.array([0.0, 0.0, _STAMP_BODY_LEN], dtype=np.float64)
        # Clamp the stamp face so it hovers just above the document top instead
        # of penetrating it.  The seal is a kinematic prop driven by set_pos,
        # so when its face dips below the doc top the contact solver shoves the
        # (lighter, movable) document — under Genesis 1.0.0 the descending seal
        # bulldozed the doc ~0.23 m forward before it could stamp, so the stamp
        # never landed on the paper.  A small hover removes the contact while
        # still reading as a stamp; the red mark is placed in code, not by
        # physical contact.
        min_face_z = self.TABLE_TOP_Z + float(self._doc_half[2]) + 0.002
        if origin_pos[2] < min_face_z:
            origin_pos[2] = min_face_z
        # Stamp local +Y → world +Z (body points up from base to handle).
        R_stamp = np.array([
            [1.0, 0.0,  0.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0,  0.0],
        ])
        q = t3d.quaternions.mat2quat(R_stamp)
        self.stamp_actor.entity.set_pos(origin_pos)
        self.stamp_actor.entity.set_quat(np.asarray(q, dtype=np.float64))

    # ------------------------------------------------------------------
    # Per-seed scene
    # ------------------------------------------------------------------

    def reset(self, seed: int = 0):
        obs = super().reset(seed=seed)
        self._prepare_episode_start()
        return obs

    def _prepare_episode_start(self):
        recorder = self.vla_recorder
        record_stride = self._record_stride
        try:
            self.vla_recorder = None
            self._record_stride = None
            self._calibrate_motion_key_frames()
            self._place_documents_for_stamp_sequence()
            for _ in range(20):
                self.step_sim()
        finally:
            self.vla_recorder = recorder
            self._record_stride = record_stride
            self._clear_episode_buffers()

    def eval_pre_policy_warmup(self):
        """Run the first visible avatar stamp before policy control starts."""
        if self.avatar is None or not self.document_actors:
            return None
        if getattr(self, "_eval_avatar_warmup_done", False):
            return None
        self._eval_avatar_warmup_done = True
        self._attach_stamp_to_palm()
        self._freeze_avatar = True
        self._play_one_stamp_cycle(doc_idx=0, doc=self.document_actors[0])
        self._freeze_avatar = True
        return self.get_obs()

    def _place_documents_for_stamp_sequence(self):
        if not self.document_actors:
            return

        half_z = self._doc_half[2]
        stamp_doc_pos = np.array(
            [float(self._stamp_xy[0]), float(self._stamp_xy[1]),
             self.TABLE_TOP_Z + half_z + 0.002],
            dtype=np.float64,
        )
        self.document_actors[0].set_pos(stamp_doc_pos)

        doc1_xy = None
        if len(self.document_actors) > 1:
            xmin = self.TABLE_CENTER_XY[0] - self.TABLE_HALF_SIZE[0] + 0.12
            xmax = self.TABLE_CENTER_XY[0] + self.TABLE_HALF_SIZE[0] - 0.12
            ymin = self.TABLE_CENTER_XY[1] - self.TABLE_HALF_SIZE[1] + 0.12
            ymax = self.TABLE_CENTER_XY[1] + self.TABLE_HALF_SIZE[1] - 0.12
            doc1_xy = np.array([
                np.clip(float(self._stamp_xy[0]) + 0.24, xmin, xmax),
                np.clip(float(self._stamp_xy[1]) - 0.24, ymin, ymax),
            ], dtype=np.float64)
            doc1_pos = np.array(
                [doc1_xy[0], doc1_xy[1], self.TABLE_TOP_Z + half_z + 0.002],
                dtype=np.float64,
            )
            self.document_actors[1].set_pos(doc1_pos)

        if hasattr(self, "_feeder_xy") and self._feeder_xy:
            self._feeder_xy[0] = (float(self._stamp_xy[0]),
                                  float(self._stamp_xy[1]))
            if len(self._feeder_xy) > 1 and doc1_xy is not None:
                self._feeder_xy[1] = (float(doc1_xy[0]), float(doc1_xy[1]))

    def _clear_episode_buffers(self):
        if self.vla_recorder is not None:
            self.vla_recorder._captures.clear()
            self.vla_recorder._tick = 0
            self.vla_recorder._t0 = time.time()
        self.traj_data = []
        self.FRAME_IDX = 0
        self.avatar_collided = False
        self.avatar_collision_log = []
        self.avatar_safety_log = []
        self.avatar_safety_intervention_log = []
        self._avatar_collision_tick = 0

    def load_actors(self):
        """Spawn docs (count, size, colour, feeder XY all per-seed), the
        seal, the ink pad, and one red mark cylinder per doc.

        Avatar pose (XY + yaw) is jittered before `_init_avatar` runs;
        the ink pad position then follows automatically because
        `_calibrate_motion_key_frames` derives it from the world palm
        trajectory, which moves with the avatar.
        """
        table_top = self.TABLE_TOP_Z

        # Avatar pose jitter — write instance attrs so reset() picks them up.
        base_pos = np.asarray(StampDocuments.avatar_init_pos, dtype=np.float64)
        base_rot = np.asarray(StampDocuments.avatar_init_rot, dtype=np.float64)
        avatar_dx = float(np.random.uniform(-_AVATAR_POS_JITTER_XY, _AVATAR_POS_JITTER_XY))
        avatar_dy = float(np.random.uniform(-_AVATAR_POS_JITTER_XY, _AVATAR_POS_JITTER_XY))
        avatar_dyaw = float(np.deg2rad(np.random.uniform(
            -_AVATAR_YAW_JITTER_DEG, _AVATAR_YAW_JITTER_DEG)))
        c, s = np.cos(avatar_dyaw), np.sin(avatar_dyaw)
        Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        self.avatar_init_pos = base_pos + np.array([avatar_dx, avatar_dy, 0.0])
        self.avatar_init_rot = Rz @ base_rot

        # Doc count + size + feeder positions.
        n_docs = _DOC_COUNT
        size_scale = float(np.random.uniform(*_DOC_SIZE_SCALE_RANGE))
        self._doc_half = tuple(float(h) * size_scale for h in _DOC_HALF)
        col_dx = float(np.random.uniform(-_FEEDER_X_JITTER, _FEEDER_X_JITTER))
        self._feeder_xy = []
        for (x, y) in _DOC_FEEDER_XY[:n_docs]:
            dy = float(np.random.uniform(-_FEEDER_Y_JITTER, _FEEDER_Y_JITTER))
            self._feeder_xy.append((float(x) + col_dx, float(y) + dy))

        # Documents.
        self.document_actors = []
        z = table_top + self._doc_half[2] + 0.002
        for (x, y) in self._feeder_xy:
            color = _DOC_COLOR_POOL[int(np.random.randint(len(_DOC_COLOR_POOL)))]
            self.document_actors.append(create_primitive(
                self.scene, "box",
                Pose([x, y, z]),
                size={"half_size": self._doc_half},
                color=color,
                is_static=False,
            ))

        # Stamp (seated in palm during reset()) and ink pad (moved in
        # reset() once calibration locates the dip XY).
        self.stamp_actor = load_object(
            self.scene,
            Pose([0.30, 0.60, table_top + 0.30], _IDENTITY_Q),
            "100_seal", model_id=0, convex=True, is_static=False,
        )
        if Path(_INKPAD_OBJ).is_file():
            try:
                self.ink_entity = load_mesh(
                    self.scene, _INKPAD_OBJ,
                    Pose([self._ink_xy[0], self._ink_xy[1], table_top], _IDENTITY_Q),
                    scale=1.0, is_static=True,
                )
            except FileNotFoundError:
                self.ink_entity = None
        else:
            self.ink_entity = None
        if self.ink_entity is None:
            self.ink_entity = create_primitive(
                self.scene, "box",
                Pose([self._ink_xy[0], self._ink_xy[1], table_top + 0.005], _IDENTITY_Q),
                size={"half_size": (0.045, 0.030, 0.005)},
                color=(0.05, 0.04, 0.04),
                is_static=True,
            )

        # Red mark cylinders, one per doc.  Spawned under the table with
        # collision off; snapped to the doc's top face by `_attach_stamp_mark`
        # the moment that doc's stamp action fires.
        self.stamp_marks = []
        for i in range(len(self._feeder_xy)):
            self.stamp_marks.append(create_primitive(
                self.scene, "cylinder",
                Pose([0.0, 0.0, -0.5 - 0.01 * i]),
                size={"radius": _STAMP_MARK_RADIUS,
                      "half_length": _STAMP_MARK_HALF_H},
                color=(0.85, 0.08, 0.08),
                is_static=True, collision=False,
            ))
        n = len(self._feeder_xy)
        self._stamp_mark_attached: list = [None] * n
        self._stamp_mark_local_offset: list = [None] * n
        self._stamp_within_doc: list = [False] * n

    # ------------------------------------------------------------------
    # Calibration: dry-run the stamp motion to find dip / stamp positions
    # ------------------------------------------------------------------
    def _calibrate_motion_key_frames(self):
        """Replay `stamp_crop` once on the avatar without rendering, record
        the right-hand trajectory, and pick the dip and stamp frames.

        Dip = local-min hand z while hand y is large (closer to body).
        Stamp = local-min hand z while hand y is small (extended forward).
        """
        if self.avatar is None:
            return

        self.avatar.frame_ratio = _AVATAR_FRAME_RATIO
        self.avatar.play_animation(_STAMP_MOTION, hand_id=_STAMP_HAND_ID)
        trajectory = []
        self._freeze_avatar = False
        while not self.avatar.spare():
            self.step_sim()
            palm = to_numpy(
                self.avatar.robot.get_palm_center(_STAMP_HAND_ID)
            ).ravel()
            trajectory.append(palm.copy())
        trajectory = np.asarray(trajectory)

        y_med = np.median(trajectory[:, 1])
        fwd_mask = trajectory[:, 1] < y_med
        back_mask = ~fwd_mask
        stamp_idx = np.argmin(np.where(fwd_mask, trajectory[:, 2], np.inf))
        dip_idx = np.argmin(np.where(back_mask, trajectory[:, 2], np.inf))

        self._stamp_xy = trajectory[stamp_idx, :2].copy()
        self._ink_xy = trajectory[dip_idx, :2].copy()
        # trajectory[0] is sampled after PlayAnimationMotion advances from
        # frame 0 to frame 1, so convert the sample index back to at_frame.
        self._stamp_motion_frame = int(stamp_idx) + 1

        ink_new = np.array([self._ink_xy[0], self._ink_xy[1], self.TABLE_TOP_Z])
        self.ink_entity.set_pos(ink_new.astype(np.float64))

        # Reset to idle so calibration's intra-motion translation undoes,
        # then seed avatar to frame 0 so the play_once() transition reads
        # as a continuation rather than a jump from idle.
        self.avatar.reset(
            np.asarray(self.avatar_init_pos, dtype=np.float64).copy(),
            np.asarray(self.avatar_init_rot, dtype=np.float64).copy(),
        )
        for _ in range(20):
            self.step_sim()
        self._seed_avatar_to_stamp_frame0()

    def _seed_avatar_to_stamp_frame0(self):
        """Pose avatar at frame 0 of `_STAMP_MOTION` without advancing it."""
        if self.avatar is None:
            return
        self.avatar.frame_ratio = _AVATAR_FRAME_RATIO
        self.avatar.play_animation(_STAMP_MOTION, hand_id=_STAMP_HAND_ID)
        mod = self.avatar.motion_modules[_STAMP_MOTION]
        self.avatar.robot.pose = mod.data[0]
        self.avatar.robot.node_trans = mod.node_data[0]
        self.avatar.robot.global_mat = mod.global_mat
        self.avatar.robot.global_mat_inv = mod.global_mat_inv
        self.avatar.robot.action_state = AvatarState.NO_ACTION
        self.avatar.robot.action_status = ActionStatus.SUCCEED
        self.avatar.motion_modules.pop(_STAMP_MOTION, None)
        self.avatar.robot.update()
        self._update_arm_obstacles("right")

    # ------------------------------------------------------------------
    # Avatar arm capsule for the planner
    # ------------------------------------------------------------------
    def _update_arm_obstacles(self, arm_tag: str = "right"):
        """Push a point-cloud capsule covering the avatar's right
        elbow→pinky into the mplib planner so screw/RRT plans route around
        the avatar arm instead of through it."""
        if self.avatar is None:
            return
        try:
            skin = self.avatar.robot.skin
            elbow = to_numpy(skin.get_global_translation("RightForeArm")[0]).ravel()[:3]
            pinky = to_numpy(skin.get_global_translation("RightHandPinky1")[0]).ravel()[:3]
        except Exception:
            return

        axis = pinky - elbow
        length = float(np.linalg.norm(axis))
        if length < 1e-4:
            return
        axis = axis / length
        ref = np.array([0.0, 0.0, 1.0]) if abs(axis[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        u = np.cross(axis, ref)
        u = u / (np.linalg.norm(u) + 1e-8)
        v = np.cross(axis, u)

        angles = np.linspace(0.0, 2.0 * np.pi, self._ARM_CAPSULE_N_AROUND, endpoint=False)
        pts = []
        for t in np.linspace(0.0, 1.0, self._ARM_CAPSULE_N_ALONG):
            center = elbow + t * (pinky - elbow)
            for a in angles:
                pts.append(center + self._ARM_CAPSULE_RADIUS * (np.cos(a) * u + np.sin(a) * v))
        pts = np.asarray(pts, dtype=np.float64)

        try:
            arm = self.robot.get_arm(arm_tag)
            arm.planner.update_obstacles(pts, resolution=0.02)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Avatar play
    # ------------------------------------------------------------------
    def _attach_stamp_to_palm(self):
        """Mark the stamp 'ready' so `_rigid_vertical_stamp` overrides its
        pose every sim step.  We don't use avatar.attach_object_to_hand:
        the Mixamo hand pose isn't perfectly palm-down at the stamp frame,
        so a rigid attach would leave the stamp tilted."""
        if self.avatar is None:
            return
        self.step_sim()
        self._stamp_ready = True
        self._rigid_vertical_stamp()

    def _play_one_stamp_cycle(self, doc_idx: int | None = None, doc=None):
        """Play one `stamp_crop` cycle (dip → stamp).  When doc/doc_idx
        are given, the red mark snaps onto the doc at the local-min palm-z
        within the stamp half of the trajectory."""
        if self.avatar is None:
            return
        self._freeze_avatar = False
        self.avatar.frame_ratio = _AVATAR_FRAME_RATIO
        self.avatar.play_animation(_STAMP_MOTION, hand_id=_STAMP_HAND_ID)
        motion = self.avatar.motion_modules[_STAMP_MOTION]

        stamp_half_y_cutoff = 0.5 * (float(self._stamp_xy[1]) + float(self._ink_xy[1]))
        stamp_is_fwd = float(self._stamp_xy[1]) < float(self._ink_xy[1])

        # Pin the document for the duration of the stamp.  The avatar's
        # descending hand/seal slides the (light, freshly-settled) doc ~0.23 m
        # forward under Genesis 1.0.0 — it flees the stamp zone before the seal
        # lands, so the imprint never falls on the paper.  A real document being
        # stamped does NOT slide, so we hold it at the pose the robot left it at.
        # (The robot's deliberate push happens earlier with the avatar frozen;
        # this only freezes the doc while the human stamps it.)
        pin_pos = None
        pin_quat = None
        if doc is not None:
            cur = to_numpy(doc.get_pos()).ravel().copy()
            # Pin the XY at the calibrated stamp position (where the seal lands)
            # rather than the possibly-already-drifted current XY, keeping the
            # current resting Z.  Both documents are stamped at stamp_xy (doc 0
            # is placed there; doc 1 is pushed there by the robot first).
            pin_pos = np.array(
                [float(self._stamp_xy[0]), float(self._stamp_xy[1]), float(cur[2])],
                dtype=np.float64,
            )
            pin_quat = to_numpy(doc.get_quat()).ravel().copy()

        # Track the deepest seal press while the hand is over the stamp zone.
        # The earlier "first local min that rises again" heuristic missed
        # (motion ended before the rise, or `near_doc` failed) and fell back to
        # the stale calibrated stamp_xy. Snapshot the seal XY AND the doc pose
        # together at the global minimum press instead; with the doc pinned the
        # seal is provably over the paper there.
        min_base_z = np.inf
        best_palm_xy = None
        best_doc_pos = None
        best_doc_quat = None
        stamped = False

        while not self.avatar.spare():
            self.step_sim()
            if doc is not None and pin_pos is not None:
                doc.set_pos(pin_pos.astype(np.float64))
                doc.set_quat(pin_quat.astype(np.float64))
            if doc is None or doc_idx is None:
                continue
            palm = to_numpy(
                self.avatar.robot.get_palm_center(_STAMP_HAND_ID)
            ).ravel()
            base_z = float(palm[2]) - _STAMP_BODY_LEN
            in_stamp_half = (palm[1] < stamp_half_y_cutoff) if stamp_is_fwd \
                else (palm[1] > stamp_half_y_cutoff)
            if in_stamp_half and base_z < min_base_z:
                min_base_z = base_z
                best_palm_xy = palm[:2].copy()
                best_doc_pos = to_numpy(doc.get_pos()).ravel().copy()
                best_doc_quat = to_numpy(doc.get_quat()).ravel().copy()
            if (not stamped and in_stamp_half
                    and motion.at_frame >= self._stamp_motion_frame):
                self._attach_stamp_mark(
                    doc_idx, doc, mark_xy=palm[:2],
                    doc_pos_override=to_numpy(doc.get_pos()).ravel().copy(),
                    doc_quat_override=to_numpy(doc.get_quat()).ravel().copy())
                stamped = True

        if doc is not None and doc_idx is not None and not stamped:
            if best_palm_xy is not None:
                self._attach_stamp_mark(
                    doc_idx, doc, mark_xy=best_palm_xy,
                    doc_pos_override=best_doc_pos,
                    doc_quat_override=best_doc_quat)
            else:
                # Hand never entered the stamp half (shouldn't happen) — fall
                # back to the calibrated stamp XY against the live doc pose.
                self._attach_stamp_mark(doc_idx, doc, mark_xy=self._stamp_xy)

        self._freeze_avatar = True
        self._update_arm_obstacles("right")

    # ------------------------------------------------------------------
    # Arm push (screw IK with Cartesian fallback — no RRT elbow flips)
    # ------------------------------------------------------------------
    def _tcp(self, xyz, push_axis: int | None = None) -> Pose:
        q = _TOP_DOWN_Q_Y_PUSH if push_axis == 1 else _TOP_DOWN_Q
        return Pose(np.array(xyz, dtype=np.float64), q)

    def _solve_ik_top_down(self, xyz, arm_tag, push_axis: int | None = None):
        arm = self.robot.get_arm(arm_tag)
        link = tcp_to_link_pose(self._tcp(xyz, push_axis=push_axis), arm.tcp_offset)
        pose7 = link.to_pose7()
        ik_fn = getattr(arm.entity, "inverse_kinematics", None)
        if ik_fn is None:
            return None
        try:
            cur = to_numpy(arm.entity.get_qpos()).ravel()
            sol = ik_fn(
                link=arm.ee_link,
                pos=np.array(pose7[:3]),
                quat=np.array(pose7[3:7]),
                init_qpos=cur,
            )
        except Exception:
            return None
        if sol is None:
            return None
        return to_numpy(sol).ravel()[:arm.n_arm]

    def _move_cartesian(self, start_xyz, end_xyz, arm_tag,
                        push_axis: int | None = None,
                        n_steps=30, sim_per_step=15):
        """Linear Cartesian interpolation, IK per waypoint, PD-controlled."""
        arm = self.robot.get_arm(arm_tag)
        base_target = getattr(arm, "_cached_target", None)
        if base_target is None:
            base_target = to_numpy(arm.entity.get_qpos())
        base_target = np.array(base_target, dtype=float).copy()
        start = np.asarray(start_xyz, dtype=float)
        end = np.asarray(end_xyz, dtype=float)
        qpos_full = base_target.copy()
        for i in range(1, n_steps + 1):
            t = i / n_steps
            arm_qpos = self._solve_ik_top_down(
                start * (1 - t) + end * t, arm_tag, push_axis=push_axis)
            if arm_qpos is None:
                continue
            qpos_full = base_target.copy()
            for j_idx, j in enumerate(arm.arm_joints):
                if j is None:
                    continue
                dof_idx = _get_dof_idx(j)
                if dof_idx is not None:
                    qpos_full[dof_idx] = float(arm_qpos[j_idx])
            arm.entity.control_dofs_position(qpos_full)
            for _ in range(sim_per_step):
                self.step_sim()
        arm._cached_target = qpos_full.copy()
        return True

    def _move_screw(self, target_xyz, arm_tag, push_axis: int | None = None):
        """Straight-line MPLib screw plan; falls back to manual Cartesian
        interp on screw failure."""
        arm = self.robot.get_arm(arm_tag)
        link = tcp_to_link_pose(self._tcp(target_xyz, push_axis=push_axis), arm.tcp_offset)
        try:
            result = arm.planner.plan_screw_path(arm.get_arm_qpos(), link.to_pose7())
            if getattr(result, "success", False):
                self.execute_plan(result, arm_tag)
                return True
        except Exception:
            pass
        cur_ee = np.array(arm.get_ee_pose()[:3], dtype=float)
        self._move_cartesian(
            cur_ee, np.asarray(target_xyz, dtype=float), arm_tag, push_axis=push_axis)
        return True

    def _hold_arm(self, target_xyz, arm_tag, n_steps: int = 60,
                  push_axis: int | None = None):
        """Hold the arm with PD target at IK(target_xyz).  Holding the
        commanded TCP (not measured qpos) lets PD pull back if the screw
        trajectory overshot the stop."""
        arm = self.robot.get_arm(arm_tag)
        arm_qpos = self._solve_ik_top_down(
            np.asarray(target_xyz, dtype=float), arm_tag, push_axis=push_axis)
        if arm_qpos is None:
            arm_qpos = to_numpy(arm.get_arm_qpos()).ravel()
        hold_gripper = arm.gripper_val
        for _ in range(n_steps):
            self.robot.set_arm_joints(arm_qpos, arm_tag)
            self.robot.set_gripper(hold_gripper, arm_tag)
            self.step_sim()

    def _push_doc(self, doc, target_xy, arm_tag,
                  y_only: bool = False, precise: bool = True) -> bool:
        """Two centerline axis pushes: one X push, then one Y push."""
        if not precise:
            tgt = np.asarray(target_xy, dtype=np.float64)
            src = to_numpy(doc.get_pos()).ravel()[:2]
            if abs(tgt[0] - src[0]) > 0.01:
                if not self._push_axis_midline(doc, axis=0,
                                               target_val=float(tgt[0]),
                                               arm_tag=arm_tag):
                    return False
            if y_only:
                return True
            src = to_numpy(doc.get_pos()).ravel()[:2]
            if abs(tgt[1] - src[1]) > 0.01:
                if not self._push_axis_midline(doc, axis=1,
                                               target_val=float(tgt[1]),
                                               arm_tag=arm_tag):
                    return False
            return True

        max_segment = 0.10 if precise else 1.0
        max_segments = 4 if precise else 1
        tgt = np.asarray(target_xy, dtype=np.float64)
        src = to_numpy(doc.get_pos()).ravel()[:2]
        if abs(tgt[0] - src[0]) > 0.01:
            if not self._push_axis(doc, axis=0, target_val=float(tgt[0]),
                                   arm_tag=arm_tag, max_segment=max_segment,
                                   max_segments=max_segments):
                return False
        if y_only:
            return True
        src = to_numpy(doc.get_pos()).ravel()[:2]
        if abs(tgt[1] - src[1]) > 0.01:
            if not self._push_axis(doc, axis=1, target_val=float(tgt[1]),
                                   arm_tag=arm_tag, max_segment=max_segment,
                                   max_segments=max_segments):
                return False
        return True

    def _push_axis_midline(self, doc, axis: int, target_val: float,
                           arm_tag: str,
                           line_offset: float | None = None) -> bool:
        """Single Piper-style push along one axis.

        The commanded TCP is the fingertip center, not the true side-contact
        face.  For thin documents the stable behavior is to use a partially
        closed gripper and stop the TCP behind the requested document center
        so the paper, not the TCP, lands at the target.  `line_offset` shifts
        the contact line along the perpendicular axis for obstacle clearance.
        """
        p = to_numpy(doc.get_pos()).ravel()
        src_xy = p[:2].copy()
        remaining = target_val - src_xy[axis]
        if abs(remaining) < 0.01:
            return True

        perp_axis = 1 - int(axis)
        line_xy = src_xy.copy()
        if line_offset is not None:
            line_xy[perp_axis] = src_xy[perp_axis] + float(line_offset)

        direction = 1.0 if remaining > 0 else -1.0
        push_dir = np.zeros(2, dtype=np.float64)
        push_dir[axis] = direction

        doc_half_along = float(self._doc_half[axis])
        behind_gap = 0.085
        precontact_gap = 0.018
        contact_offset = doc_half_along + 0.010
        push_z = self.TABLE_TOP_Z + float(self._doc_half[2]) + 0.010
        safe_z = self.TABLE_TOP_Z + 0.18

        behind_xy = line_xy - push_dir * (doc_half_along + behind_gap)
        precontact_xy = line_xy - push_dir * (doc_half_along + precontact_gap)
        stop_xy = line_xy.copy()
        stop_xy[axis] = float(target_val) - direction * contact_offset
        retract_xy = stop_xy - push_dir * 0.070

        for xyz in (
            (behind_xy[0], behind_xy[1], safe_z),
            (behind_xy[0], behind_xy[1], push_z),
            (precontact_xy[0], precontact_xy[1], push_z),
            (stop_xy[0], stop_xy[1], push_z),
        ):
            if not self._move_screw(list(xyz), arm_tag, push_axis=axis):
                return False
        self._hold_arm(
            [stop_xy[0], stop_xy[1], push_z], arm_tag, n_steps=45, push_axis=axis)
        if not self._move_screw(
                [retract_xy[0], retract_xy[1], safe_z], arm_tag, push_axis=axis):
            return False
        for _ in range(30):
            self.step_sim()
        return True

    def _clear_first_doc_push(self, doc, target_x: float, arm_tag: str) -> bool:
        """Clear the stamped document with one x-axis push.

        The avatar hand is intentionally left where the stamp motion ends.
        To avoid clipping that hand, contact the document on the y edge
        farther from the current palm instead of pushing through its center.
        """
        doc_xy = to_numpy(doc.get_pos()).ravel()[:2]
        edge_offset = _FIRST_DOC_PUSH_EDGE_FRACTION * float(self._doc_half[1])
        line_offset = -edge_offset
        if self.avatar is not None:
            try:
                palm_xy = np.asarray(
                    self.avatar.robot.get_palm_center(_STAMP_HAND_ID),
                    dtype=np.float64,
                ).ravel()[:2]
                sign = -1.0 if palm_xy[1] >= doc_xy[1] else 1.0
                line_offset = sign * edge_offset
            except Exception:
                pass
        return self._push_axis_midline(
            doc, axis=0, target_val=target_x, arm_tag=arm_tag,
            line_offset=line_offset,
        )

    def _push_axis(self, doc, axis: int, target_val: float, arm_tag,
                   max_segment: float = 0.10,
                   max_segments: int = 4,
                   min_progress: float = 0.01) -> bool:
        """Axis-aligned push, segmented with perp-axis re-alignment.

        Long pushes split into ≤max_segment chunks; between segments we
        re-fetch the doc pose and re-align the TCP perp-axis with the doc's
        current center, so transverse drift can't accumulate.

        Contact: TCP is at the gripper's pushing face (no stamp_radius
        offset), so it stops at `seg_end − direction · doc_half_along` and
        the doc's edge lands at seg_end.
        """
        doc_half_along = self._doc_half[axis]
        push_z = self.TABLE_TOP_Z + 0.002
        safe_z = self.TABLE_TOP_Z + 0.18

        prev_src_axis = None
        for _ in range(max_segments):
            p = to_numpy(doc.get_pos()).ravel()
            src_xy = p[:2].copy()
            remaining = target_val - src_xy[axis]
            if abs(remaining) < 0.005:
                return True
            # Bail if the previous segment didn't move the doc — would
            # otherwise loop forever under repeated screw failures.
            if prev_src_axis is not None:
                if abs(src_xy[axis] - prev_src_axis) < min_progress:
                    return True
            prev_src_axis = src_xy[axis]

            direction = 1.0 if remaining > 0 else -1.0
            push_dir = np.zeros(2)
            push_dir[axis] = direction
            seg_len = min(abs(remaining), max_segment)
            seg_end = src_xy[axis] + direction * seg_len
            behind_xy = src_xy - push_dir * (doc_half_along + 0.06)
            stop_xy = src_xy.copy()
            stop_xy[axis] = seg_end - direction * doc_half_along
            retract_xy = stop_xy - push_dir * 0.05

            for xyz in (
                (behind_xy[0], behind_xy[1], safe_z),
                (behind_xy[0], behind_xy[1], push_z),
                (stop_xy[0],   stop_xy[1],   push_z),
            ):
                if not self._move_screw(list(xyz), arm_tag):
                    return False
            self._hold_arm([stop_xy[0], stop_xy[1], push_z], arm_tag, n_steps=60)
            if not self._move_screw([retract_xy[0], retract_xy[1], safe_z], arm_tag):
                return False
        return True

    # ------------------------------------------------------------------
    def play_once(self) -> bool:
        self._attach_stamp_to_palm()
        arm_tag = "right"
        self.set_gripper(_PUSH_GRIPPER_VAL, arm_tag)

        # Standby: behind-doc in −Y, up high — clear of the avatar's stamp
        # arc so the avatar has an unobstructed approach to the doc.
        standby_xyz = [self._stamp_xy[0],
                       float(self._stamp_xy[1]) - 0.20,
                       self.TABLE_TOP_Z + 0.22]

        # Avatar frozen by default; `_play_one_stamp_cycle` un/refreezes.
        self._freeze_avatar = True

        if len(self.document_actors) != _DOC_COUNT:
            return False

        first_doc, second_doc = self.document_actors

        # Doc 0 was placed at stamp_xy during reset(); start clear of the
        # avatar and let the avatar stamp it before the arm moves it away.
        self._move_screw(standby_xyz, arm_tag)
        self._play_one_stamp_cycle(doc_idx=0, doc=first_doc)

        src = to_numpy(first_doc.get_pos()).ravel()[:2]
        clear_x = max(
            self.TABLE_CENTER_XY[0] - self.TABLE_HALF_SIZE[0] + 0.12,
            float(src[0]) - 0.18,
        )
        if not self._clear_first_doc_push(first_doc, clear_x, arm_tag):
            return False

        if not self._push_doc(second_doc, self._stamp_xy, arm_tag, precise=False):
            return False
        self._move_screw(standby_xyz, arm_tag)
        self._play_one_stamp_cycle(doc_idx=1, doc=second_doc)

        self._freeze_avatar = False
        return True

    def check_success(self) -> bool:
        if not self.plan_success:
            return False
        # Success = every doc has its stamp mark inside the doc edge.  We no
        # longer require docs to have moved from the feeder XY: the last
        # stamped doc intentionally stays at the stamp pose (the rollout
        # ends as soon as it gets stamped).
        within = getattr(self, "_stamp_within_doc", None)
        if len(within or []) != _DOC_COUNT or not all(within):
            return False
        return True
