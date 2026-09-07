"""Place burger and fries (assist): same scene as ``place_burger_fries``
but with a human avatar that tosses an extra hamburger onto the tray
using the ``Trash_clip`` motion.  The robot keeps placing its 1-2
burgers + 1-2 fries; the avatar adds one more burger.

Geometry pipeline (mirrors ``dump_bin_assist``):

  1. Pre-bake the avatar's left-palm trajectory by running the motion
     once at the default pose (``palm_at_attach`` at clip frame 25,
     ``palm_at_detach`` at clip frame 57).
  2. Choose ``drop_target`` inside the tray xy region — offset toward
     the avatar's side so the dropped burger doesn't collide with the
     robot's drop slots in the centre.
  3. ``shift_xy = drop_target - palm_at_detach.xy``.
  4. ``T_new = avatar_init + shift``; ``pickup_world = palm_at_attach +
     shift`` (with the same lift/forward fixups dump_bin_assist uses).
  5. Reset avatar at ``T_new``, replay ``Trash_clip`` with attach at
     frame 25 and a hold-and-zero release at frame 57.

Avatar runs FIRST so the tray is empty when its burger lands, then the
robot picks its red hamburgers + fries from the table to fill the
remaining tray slots.

Eval: every robot-placed item is on the tray (parent's check) AND the
avatar's hamburger landed on the tray.
"""

from __future__ import annotations

import numpy as np

from ..avatar.eval_mode_mixin import EvalModeAvatarMixin
from ..utils import Pose, load_object, to_numpy
from ..task_bases.place_burger_fries import (
    PlaceBurgerFries,
    _HAMBURG_ID,
)


_GENERATED_MOTIONS_PKL = "avatars/motions/generated_motions.pkl"


class PlaceBurgerFriesAssist(EvalModeAvatarMixin, PlaceBurgerFries):
    INSTRUCTION = "place the hamburger and the french fries on the tray while the human adds one food item"

    use_avatar = True

    AVATAR_BASE_ROT = np.array(
        [[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=np.float64
    )
    avatar_init_pos = np.array([-0.20, 0.55, -0.18])
    avatar_init_rot = AVATAR_BASE_ROT
    AVATAR_YAW_DEG = 0.0
    AVATAR_LIFT_Z = 0.05
    AVATAR_BACK_OFFSET = 0.05
    AVATAR_PALM_FORWARD_OFFSET = 0.04

    ASSIST_MOTION_NAME = "Trash_clip"
    ATTACH_FRAME = 25
    DETACH_FRAME = 57
    POST_FALL_FRAMES = 500
    IDLE_RETURN_FRAMES = 120
    AVATAR_HAND_ID = 0

    AVATAR_TRAY_OFFSET = (-0.10, +0.06)
    AVATAR_PICKUP_XY = (0.24, 0.24)
    AVATAR_PICK_LIFT = 0.020
    AVATAR_PLACE_LIFT = 0.090
    AVATAR_APPROACH_FRAMES = 260
    AVATAR_TRANSPORT_FRAMES = 360
    AVATAR_RETRACT_FRAMES = 180
    AVATAR_NATURAL_BODY_MARGIN = 0.34
    AVATAR_NATURAL_Y_BOUNDS = (0.28, 0.62)
    AVATAR_NATURAL_YAW_LIMIT_DEG = 30.0
    AVATAR_NATURAL_SETTLE_STEPS = 20
    AVATAR_PRE_MOTION_SETTLE_STEPS = 80
    AVATAR_HOLD_ZERO_VEL_STEPS = 10
    AVATAR_SPAWN_OFFSCREEN_POS = (8.0, 8.0, 0.05)
    AVATAR_PICKUP_Z_CLEARANCE = 0.050
    EVAL_TRIGGER_STEP_MIN = 20
    EVAL_TRIGGER_STEP_MAX = 60

    def __init__(self, config: dict = None):
        cfg = dict(config or {})
        avatar_cfg = dict(cfg.get("avatar", {}))
        avatar_cfg.setdefault("generated_motion_data", _GENERATED_MOTIONS_PKL)
        cfg["avatar"] = avatar_cfg
        super().__init__(cfg)
        self.avatar_burger = None
        self._avatar_burger_drop_xy = None

    def load_actors(self):
        super().load_actors()
        self.avatar_burger = load_object(
            self.scene,
            Pose(self.AVATAR_SPAWN_OFFSCREEN_POS, self._Q_BURGER),
            _HAMBURG_ID, model_id=0,
            convex=True, is_static=False, friction=4.0,
        )

    def _avatar_burger_entity(self):
        return getattr(self.avatar_burger, "entity", self.avatar_burger)

    def _set_avatar_burger_pose(self, pos, quat):
        ent = self._avatar_burger_entity()
        ent.set_pos(np.asarray(pos, dtype=float))
        ent.set_quat(np.asarray(quat, dtype=float))

    def _zero_avatar_burger_velocity(self, vel):
        self._avatar_burger_entity().set_dofs_velocity(vel)

    def _get_avatar_burger_pos(self):
        return to_numpy(self._avatar_burger_entity().get_pos()).ravel()[:3]

    def _avatar_init_pos_lifted(self) -> np.ndarray:
        p = np.array(self.avatar_init_pos, dtype=float).copy()
        p[2] += self.AVATAR_LIFT_Z
        return p

    def _avatar_rot_with_yaw(self) -> np.ndarray:
        yaw = np.deg2rad(self.AVATAR_YAW_DEG)
        cz, sz = np.cos(yaw), np.sin(yaw)
        Rz = np.array(
            [[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float64
        )
        return Rz @ np.array(self.AVATAR_BASE_ROT, dtype=np.float64)

    def _avatar_palm_center(self, hand_id: int) -> np.ndarray:
        side = "Left" if hand_id == 0 else "Right"
        pts = []
        for finger in ("Index", "Middle", "Ring", "Pinky"):
            bone = f"{side}Hand{finger}1"
            pos = np.asarray(
                self.avatar.robot.skin.get_global_translation(bone)[0]
            ).ravel()[:3]
            pts.append(pos)
        return np.mean(pts, axis=0)

    def _avatar_wrist(self, hand_id: int) -> np.ndarray:
        side = "Left" if hand_id == 0 else "Right"
        return np.asarray(
            self.avatar.robot.skin.get_global_translation(f"{side}Hand")[0]
        ).ravel()[:3]

    def _avatar_dry_run(self):
        init_pos = self._avatar_init_pos_lifted()
        init_rot = self._avatar_rot_with_yaw()
        self.avatar.reset(global_trans=init_pos, global_rot=init_rot)
        self.avatar.play_animation(self.ASSIST_MOTION_NAME)

        palm_at_attach = palm_at_detach = wrist_at_attach = None
        step = 0
        while not self.avatar.spare() and step < 1000:
            self.scene.step()
            self.avatar.step()
            if step == self.ATTACH_FRAME:
                palm_at_attach = self._avatar_palm_center(self.AVATAR_HAND_ID)
                wrist_at_attach = self._avatar_wrist(self.AVATAR_HAND_ID)
            if step == self.DETACH_FRAME:
                palm_at_detach = self._avatar_palm_center(self.AVATAR_HAND_ID)
            step += 1
        if palm_at_detach is None:
            palm_at_detach = self._avatar_palm_center(self.AVATAR_HAND_ID)
        if palm_at_attach is None:
            raise RuntimeError(
                f"dry-run never reached attach frame {self.ATTACH_FRAME} "
                f"(loop exited at step {step})"
            )
        return palm_at_attach, palm_at_detach, wrist_at_attach

    def _start_avatar_burger_assist(self) -> tuple[np.ndarray, np.ndarray]:
        tray_xy = np.array(self._tray_xy, dtype=float)
        drop_target_xy = tray_xy + np.array(self.AVATAR_TRAY_OFFSET, dtype=float)
        self._avatar_burger_drop_xy = tuple(drop_target_xy.tolist())

        pickup_world = np.array([
            self.AVATAR_PICKUP_XY[0],
            self.AVATAR_PICKUP_XY[1],
            self.TABLE_TOP_Z + self.AVATAR_PICKUP_Z_CLEARANCE,
        ], dtype=float)
        pick_pos = pickup_world.copy()
        pick_pos[2] += self.AVATAR_PICK_LIFT
        place_pos = np.array([
            drop_target_xy[0],
            drop_target_xy[1],
            self.TABLE_TOP_Z + self.AVATAR_PLACE_LIFT,
        ], dtype=float)
        self._set_avatar_burger_pose(pickup_world, self._Q_BURGER)
        self.avatar.reset(
            global_trans=self._avatar_init_pos_lifted(),
            global_rot=self.AVATAR_BASE_ROT,
        )
        self.avatar.pick_and_place(
            pick_pos=pick_pos,
            place_pos=place_pos,
            attach_obj=self.avatar_burger,
            hand_id=None,
            approach_frames=self.AVATAR_APPROACH_FRAMES,
            transport_frames=self.AVATAR_TRANSPORT_FRAMES,
            retract_frames=self.AVATAR_RETRACT_FRAMES,
            natural=True,
            body_margin=self.AVATAR_NATURAL_BODY_MARGIN,
            body_y_bounds=self.AVATAR_NATURAL_Y_BOUNDS,
            yaw_limit_deg=self.AVATAR_NATURAL_YAW_LIMIT_DEG,
            settle_steps=self.AVATAR_NATURAL_SETTLE_STEPS,
        )
        self.avatar_collided = False
        self.avatar_collision_log = []
        return pick_pos, place_pos

    def play_once(self) -> bool:
        arm_tag = "right"
        self._boost_finger_pd(arm_tag)

        if self.avatar is None:
            return super().play_once()

        pick_pos, place_pos = self._avatar_assist_geometry_without_start()
        self._set_avatar_burger_pose(
            np.array([
                self.AVATAR_PICKUP_XY[0],
                self.AVATAR_PICKUP_XY[1],
                self.TABLE_TOP_Z + self.AVATAR_PICKUP_Z_CLEARANCE,
            ], dtype=float),
            self._Q_BURGER,
        )
        self.avatar.reset(
            global_trans=self._avatar_init_pos_lifted(),
            global_rot=self.AVATAR_BASE_ROT,
        )

        for _ in range(self.AVATAR_PRE_MOTION_SETTLE_STEPS):
            self.step_sim()

        self._avatar_real_run(pick_pos, place_pos)

        for _ in range(self.POST_FALL_FRAMES):
            self.step_sim()

        return super().play_once()

    def _avatar_assist_geometry_without_start(self) -> tuple[np.ndarray, np.ndarray]:
        tray_xy = np.array(self._tray_xy, dtype=float)
        drop_target_xy = tray_xy + np.array(self.AVATAR_TRAY_OFFSET, dtype=float)
        self._avatar_burger_drop_xy = tuple(drop_target_xy.tolist())
        pickup_world = np.array([
            self.AVATAR_PICKUP_XY[0],
            self.AVATAR_PICKUP_XY[1],
            self.TABLE_TOP_Z + self.AVATAR_PICKUP_Z_CLEARANCE,
        ], dtype=float)
        pick_pos = pickup_world.copy()
        pick_pos[2] += self.AVATAR_PICK_LIFT
        place_pos = np.array([
            drop_target_xy[0],
            drop_target_xy[1],
            self.TABLE_TOP_Z + self.AVATAR_PLACE_LIFT,
        ], dtype=float)
        return pick_pos, place_pos

    def _avatar_real_run(self, pick_pos: np.ndarray, place_pos: np.ndarray):
        self.avatar.pick_and_place(
            pick_pos=pick_pos,
            place_pos=place_pos,
            attach_obj=self.avatar_burger,
            hand_id=None,
            approach_frames=self.AVATAR_APPROACH_FRAMES,
            transport_frames=self.AVATAR_TRANSPORT_FRAMES,
            retract_frames=self.AVATAR_RETRACT_FRAMES,
            natural=True,
            body_margin=self.AVATAR_NATURAL_BODY_MARGIN,
            body_y_bounds=self.AVATAR_NATURAL_Y_BOUNDS,
            yaw_limit_deg=self.AVATAR_NATURAL_YAW_LIMIT_DEG,
            settle_steps=self.AVATAR_NATURAL_SETTLE_STEPS,
        )

        zero6 = np.zeros(6, dtype=np.float64)
        while not self.avatar.spare():
            self.step_sim()
        for _ in range(self.AVATAR_HOLD_ZERO_VEL_STEPS):
            try:
                self._zero_avatar_burger_velocity(zero6)
            except Exception:
                break
            self.step_sim()

    # ------------------------------------------------------------------
    # Eval-mode avatar assist
    # ------------------------------------------------------------------

    def _eval_at_reset(self) -> None:
        self._eval_assist_trigger_step = 0
        self._eval_assist_fired = True
        self._eval_assist_failed = False
        try:
            self._start_avatar_burger_assist()
        except Exception as exc:
            self._eval_assist_failed = True
            if self.config.get("debug_eval_avatar", False):
                print(f"[burger_fries_assist/eval] avatar assist failed: {exc}", flush=True)

    def _eval_at_step(self, step_idx: int) -> None:
        if (getattr(self, "_eval_assist_fired", False)
                or getattr(self, "_eval_assist_failed", False)):
            return
        if step_idx < int(getattr(self, "_eval_assist_trigger_step", self.EVAL_TRIGGER_STEP_MIN)):
            return
        self._eval_assist_fired = True
        try:
            self._start_avatar_burger_assist()
        except Exception as exc:
            self._eval_assist_failed = True
            if self.config.get("debug_eval_avatar", False):
                print(f"[burger_fries_assist/eval] avatar assist failed: {exc}", flush=True)

    def evaluate(self) -> dict:
        metrics = super().evaluate()
        metrics.update({
            "eval_mode": bool(self.config.get("eval_mode", False)),
            "eval_assist_trigger_step": int(getattr(
                self, "_eval_assist_trigger_step", -1,
            )),
            "eval_assist_fired": bool(getattr(self, "_eval_assist_fired", False)),
            "eval_assist_failed": bool(getattr(self, "_eval_assist_failed", False)),
            "eval_policy_step_count": int(getattr(
                self, "_eval_policy_step_count", 0,
            )),
        })
        return metrics

    def check_success(self) -> bool:
        parent_pass = super().check_success()
        if not hasattr(self, "avatar_burger") or self.avatar_burger is None:
            return parent_pass

        tray_x, tray_y = self._live_tray_center_xy()
        p = self._get_avatar_burger_pos()
        dx = abs(float(p[0]) - tray_x)
        dy = abs(float(p[1]) - tray_y)
        on_tray_xy = dx <= self.TRAY_HALF_X and dy <= self.TRAY_HALF_Y
        on_table_z = float(p[2]) >= self.TABLE_TOP_Z - self.SUCCESS_TABLE_Z_MARGIN
        avatar_ok = on_tray_xy and on_table_z
        return parent_pass and avatar_ok
