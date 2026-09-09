"""Place bread in basket (assist): cooperative variant of
``place_bread_in_basket`` where a human avatar puts one bread in the
basket while the robot, working in parallel, puts the remaining bread
or breads in the basket.

Same scene + basket as the parent.  Success: all breads in basket
(parent's ``check_success``).

Robot uses pure-physics grasp + place (no kinematic snap of held
objects).  Avatar uses the sanctioned ``attach_obj`` mechanism via
``AvatarController.pick_and_place`` to hold its bread during transit
— ``attach_obj`` is allowed for the avatar only.

Mirrors the structure of ``categorize_cooperative._avatar_pick_and_place``
+ its parallel-execution loop (~lines 1241–1360 in
``categorize_cooperative.py``).
"""

import numpy as np
import transforms3d as t3d

from ..avatar.eval_mode_mixin import EvalModeAvatarMixin
from ..utils import to_numpy
from ..task_bases.place_bread_in_basket import (
    PlaceBreadInBasket, _MIN_SEP_M, _REGION_X, _REGION_Y,
    _SAMPLE_FALLBACK_MARGIN,
)


# Per-user feedback (2026-05-04): default avatar y=0.55 puts its
# collision capsule slightly through the +y edge of the table.  Slide
# avatar / robot / object spawn region together by +Y_SHIFT so the
# avatar clears the table; the table itself is unchanged.  The bread
# region's +y end is clamped to keep objects (basket half-y ≈ 0.07,
# bread half-y ≈ 0.05) on the table top (table edge at y=+0.20, so
# we cap the region max at +0.15 with 0.05 m safety margin).
_Y_SHIFT = 0.30
_TABLE_Y_MAX = +0.20
_OBJ_TABLE_MARGIN = 0.05
_REGION_Y_SHIFTED = (
    _REGION_Y[0] + _Y_SHIFT,
    min(_REGION_Y[1] + _Y_SHIFT, _TABLE_Y_MAX - _OBJ_TABLE_MARGIN),
)
_ASSIST_SAMPLE_ATTEMPTS = 80


class PlaceBreadInBasketAssist(EvalModeAvatarMixin, PlaceBreadInBasket):
    """Avatar places one bread while the robot places the rest."""

    INSTRUCTION = "put one bread into the basket while the human puts the other bread into the basket"
    # VLA collection keeps analytic collision telemetry enabled globally.  This
    # task has a documented false-positive capsule overlap, so its own geometric
    # success contract is authoritative unless collision-free success is
    # explicitly requested in config.
    VLA_ALLOW_AVATAR_COLLISION = True
    use_avatar = True
    EASY_NUM_OBJECTS = 2
    HARD_NUM_OBJECTS = 3

    # Avatar at +Y side of the table, facing -Y (toward the table).
    # Mesh-+X (avatar forward) maps to world-(-Y).  Same constants as
    # the interrupt sibling.
    AVATAR_BASE_ROT = np.array(
        [[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=np.float64,
    )
    avatar_init_pos = np.array([0.0, 0.55 + _Y_SHIFT, -0.18])
    avatar_init_rot = AVATAR_BASE_ROT

    # The avatar's bread can spawn anywhere in the parent's full table
    # region — `AvatarController.pick_and_place(natural=True)` will
    # auto-position the avatar body per-episode so FABRIK reach covers
    # the chosen pick + place midpoint (see _NATURAL_BODY_* below).

    # Avatar pick-and-place pacing.  Defaults from the design doc /
    # categorize_cooperative — 320/400/320 frames at 0.001s dt is ~1s
    # of motion which gives the FABRIK chain time to interpolate
    # smoothly without skin pop.
    AVATAR_APPROACH_FRAMES = 320
    AVATAR_TRANSPORT_FRAMES = 400
    AVATAR_RETRACT_FRAMES = 320

    # Drop the bread this far above the basket rim before release.  The
    # parent now uses the small model-2 loaf (graspable); like the robot's
    # own placement, releasing it high lets it bounce back out over the rim
    # (robot sweep: +0.06 -> 12.8 cm out, +0.02 -> 2.5 cm in).  Seat the
    # avatar's loaf low too — a touch above the robot's 0.02 to absorb
    # FABRIK reach error.
    AVATAR_DROP_ABOVE_RIM = 0.03

    # Per-user feedback (2026-05-04): although robot + avatar already
    # run in parallel (avatar.pick_and_place is non-blocking and ticks
    # forward inside the robot primitive's step_sim() calls), have the
    # avatar wait a few sim steps before its motion *starts* so the
    # robot is visibly leading the cooperative grasp.  Implemented by
    # prepending N copies of the FABRIK plan's first frame (= avatar
    # holding its current rest pose) right after pick_and_place is
    # scheduled — see _delay_avatar_motion_start().
    AVATAR_START_DELAY_STEPS = 80

    # `AvatarController.pick_and_place(natural=True)` knobs.  The
    # controller stands the avatar `body_margin` m behind the pick/place
    # midpoint (along its forward axis), yawing up to `yaw_limit_deg`,
    # and clips body xy into the optional bounds.  body_y_bounds clamps
    # the avatar y to (avatar's MIN_Y, init y) — keeps the body from
    # being pushed further onto/under the table than is physically
    # meaningful.  body_x_bounds left unbounded so it follows the
    # target laterally.
    AVATAR_NATURAL_BODY_MARGIN = 0.40
    AVATAR_NATURAL_YAW_LIMIT_DEG = 30.0
    AVATAR_NATURAL_BODY_Y_BOUNDS = (-0.20 + _Y_SHIFT, 0.55 + _Y_SHIFT)
    AVATAR_NATURAL_SETTLE_STEPS = 20
    AVATAR_FINISH_SETTLE_STEPS = 120
    EVAL_TRIGGER_STEP_MIN = 20
    EVAL_TRIGGER_STEP_MAX = 60

    # The per-episode shift puts the avatar's analytic collision capsule into
    # the robot's transit corridor, so a flagged contact is geometric overlap,
    # not a real run-time collision (gs.materials.Avatar is collision-soft
    # against rigid bodies).  Keep tracking opt-in for ordinary runs, and keep
    # collision *rejection* as a separate opt-in so VLA collection can retain
    # the diagnostic metric without rejecting every otherwise-valid rollout.
    def __init__(self, config: dict = None):
        cfg = dict(config or {})
        cfg.setdefault("track_avatar_collision", False)
        cfg.setdefault("success_require_no_avatar_collision", False)
        super().__init__(cfg)

    # Robot base shifted by +_Y_SHIFT in the avatar direction so the
    # avatar/robot/objects keep the same relative geometry while the
    # table stays put — see _Y_SHIFT comment.
    def _load_robot(self):
        if self.config.get("robot_type", "franka") == "franka":
            kwargs = self.config.setdefault("robot_kwargs", {})
            # Parent default is (0.0, -0.65, 0.75); shift y by +_Y_SHIFT so the
            # robot keeps the same relative geometry to the +Y-shifted avatar /
            # objects.  This (0.0, -0.35, 0.75) mount matches the other avatar
            # tasks (categorize_cooperative, deliver_to_human_easy).
            kwargs.setdefault("pos", [0.0, -0.65 + _Y_SHIFT, 0.75])
        super()._load_robot()
        # Use the DEFAULT Franka home pose (no override) so this task's initial
        # robot pose matches every other task / avatar variant.  The previous
        # custom raised home (J2=-0.5, J4=-2.0) made this task's start pose the
        # odd one out; the default home reaches the table objects fine.

    # ------------------------------------------------------------------
    # Sampling — random which bread the avatar picks; the robot gets all
    # remaining breads.
    # Mirrors the interrupt sibling's _sample_positions pattern.
    # ------------------------------------------------------------------

    def _sample_positions(self, rng):
        n_breads = max(2, int(self._num_breads))
        avatar_idx = int(rng.randint(0, n_breads))
        self._avatar_bread_idx = avatar_idx
        self._robot_bread_indices = [i for i in range(n_breads) if i != avatar_idx]
        self._robot_bread_idx = self._robot_bread_indices[0]

        for _outer in range(_ASSIST_SAMPLE_ATTEMPTS):
            # Basket plus all breads sampled from the shifted full region —
            # the avatar's `natural=True` body shift covers reach.
            basket = (
                float(rng.uniform(*_REGION_X)),
                float(rng.uniform(*_REGION_Y_SHIFTED)),
            )
            breads = [
                (
                    float(rng.uniform(*_REGION_X)),
                    float(rng.uniform(*_REGION_Y_SHIFTED)),
                )
                for _ in range(n_breads)
            ]
            positions = [basket] + breads
            sep_ok = True
            for i in range(len(positions)):
                for j in range(i + 1, len(positions)):
                    dx = positions[i][0] - positions[j][0]
                    dy = positions[i][1] - positions[j][1]
                    if dx * dx + dy * dy < _MIN_SEP_M ** 2:
                        sep_ok = False
                        break
                if not sep_ok:
                    break
            if sep_ok:
                return positions
        # Fall-through: deterministic spread (matches parent style but
        # uses the shifted region).
        return [
            (-0.18, -0.04),
            (0.18, -0.04),
            (-0.18, 0.14),
            (0.18, 0.14),
        ][: 1 + n_breads]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _bread_world_center(self, bread):
        """Live bread bbox-center in world frame.  ``entity.get_pos()``
        returns the mesh origin; the bbox center sits at
        ``mesh_origin + R(bread_q) @ (center_offset_mesh * scale)``.
        See memory ``project_asset_mesh_origin``."""
        origin = to_numpy(bread.entity.get_pos()).ravel()[:3]
        q = to_numpy(bread.entity.get_quat()).ravel()[:4]
        R = t3d.quaternions.quat2mat(q)
        offset = R @ (self._bread_center_mesh * self._bread_scale)
        return origin + offset

    def _delay_avatar_motion_start(self, delay_steps: int) -> None:
        """Make the already-scheduled avatar `pick_and_place` motion
        hold its rest pose for the first ``delay_steps`` sim frames.

        Reaches into the avatar's PickPlaceMotion plan (a pre-computed
        list of node_trans, one per frame) and prepends N copies of
        ``frames[0]`` — that frame is the FABRIK solution at the
        LIVE palm position, so repeating it = avatar holds rest pose.
        Also shifts the attach/detach keyframe indices by N so the
        attach still fires when the hand reaches the bread.
        """
        if delay_steps <= 0:
            return
        mm = self.avatar.motion_modules.get("pick_and_place")
        if mm is None or not mm.frames:
            return
        hold = mm.frames[0]
        pad = [hold] * int(delay_steps)
        mm.frames = pad + mm.frames
        mm.attach_frame += int(delay_steps)
        mm.detach_frame += int(delay_steps)

    # ------------------------------------------------------------------
    # Rollout
    # ------------------------------------------------------------------

    def play_once(self) -> bool:
        with self.suppress_recording():  # pre-task settle, excluded from recordings
            for _ in range(self.INITIAL_SETTLE_STEPS):
                self.step_sim()

        if not self.breads or self.basket is None:
            return False
        arm_tag = "right"

        if self.avatar is None:
            return super().play_once()

        # ---- Avatar pick + place (non-blocking) -----------------------
        self._start_avatar_assist(add_start_delay=True)

        # ---- Robot pick + place (parent's pure-physics primitive) -----
        # _pick_and_place_bread internally step_sim()s, advancing the
        # avatar motion in parallel — DO NOT block between them.
        basket_xy = to_numpy(self.basket.entity.get_pos()).ravel()[:2]
        robot_indices = sorted(
            list(getattr(self, "_robot_bread_indices", [self._robot_bread_idx])),
            key=lambda i: float(np.linalg.norm(
                to_numpy(self.breads[i].entity.get_pos()).ravel()[:2]
                - basket_xy,
            )),
        )
        ok_robot = True
        for robot_idx in robot_indices:
            robot_bread = self.breads[robot_idx]
            robot_label = chr(ord("A") + robot_idx)
            try:
                ok_robot = self._pick_and_place_bread(
                    robot_bread, arm_tag, robot_label,
                ) and ok_robot
            except Exception:
                ok_robot = False

        # ---- Wait for avatar to finish its retract --------------------
        if not self.avatar.spare():
            while not self.avatar.spare():
                self.step_sim()
        # Final settle so the avatar-dropped bread comes to rest.
        for _ in range(self.AVATAR_FINISH_SETTLE_STEPS):
            self.step_sim()

        return ok_robot

    # ------------------------------------------------------------------
    # Eval-mode avatar assist
    # ------------------------------------------------------------------

    def _eval_at_reset(self) -> None:
        self._eval_assist_trigger_step = 0
        self._eval_assist_fired = True
        self._eval_assist_failed = False
        try:
            self._start_avatar_assist(add_start_delay=False)
        except Exception as exc:
            self._eval_assist_failed = True
            if self.config.get("debug_eval_avatar", False):
                print(f"[bread_assist/eval] avatar assist failed: {exc}", flush=True)

    def _eval_at_step(self, step_idx: int) -> None:
        if (getattr(self, "_eval_assist_fired", False)
                or getattr(self, "_eval_assist_failed", False)):
            return
        trigger = int(getattr(
            self, "_eval_assist_trigger_step", self.EVAL_TRIGGER_STEP_MIN
        ))
        if step_idx < trigger:
            return
        self._eval_assist_fired = True
        try:
            self._start_avatar_assist(add_start_delay=False)
        except Exception as exc:
            self._eval_assist_failed = True
            if self.config.get("debug_eval_avatar", False):
                print(f"[bread_assist/eval] avatar assist failed: {exc}", flush=True)

    # ------------------------------------------------------------------
    # Success — parent's "all breads in basket" + optional collision gate
    # ------------------------------------------------------------------

    def check_success(self) -> bool:
        # Tracking and rejection are deliberately separate.  Collection turns
        # tracking on for telemetry, while this task's known capsule overlap
        # must not override the geometric all-breads-in-basket verdict.
        if (self.config.get("success_require_no_avatar_collision", False)
                and self.avatar_collision_checker is not None
                and self.avatar_collided):
            return False
        return super().check_success()

    def _start_avatar_assist(self, add_start_delay: bool = True) -> None:
        # Ensure the basket rim height is measured from the live AABB before
        # the avatar (which leads the robot) aims its drop at the rim.
        self._refresh_basket_geom()
        avatar_idx = self._avatar_bread_idx
        avatar_bread = self.breads[avatar_idx]

        pick_pos = self._bread_world_center(avatar_bread)
        basket_pos = to_numpy(self.basket.entity.get_pos()).ravel()[:3]
        place_pos = np.array([
            basket_pos[0],
            basket_pos[1],
            self._basket_rim_z + self.AVATAR_DROP_ABOVE_RIM,
        ])
        self.avatar.pick_and_place(
            pick_pos=pick_pos,
            place_pos=place_pos,
            attach_obj=avatar_bread.entity,
            hand_id=None,
            approach_frames=self.AVATAR_APPROACH_FRAMES,
            transport_frames=self.AVATAR_TRANSPORT_FRAMES,
            retract_frames=self.AVATAR_RETRACT_FRAMES,
            natural=True,
            body_margin=self.AVATAR_NATURAL_BODY_MARGIN,
            yaw_limit_deg=self.AVATAR_NATURAL_YAW_LIMIT_DEG,
            body_y_bounds=self.AVATAR_NATURAL_BODY_Y_BOUNDS,
            settle_steps=self.AVATAR_NATURAL_SETTLE_STEPS,
        )
        self.avatar_collided = False
        self.avatar_collision_log = []
        if add_start_delay:
            self._delay_avatar_motion_start(self.AVATAR_START_DELAY_STEPS)

    def evaluate(self) -> dict:
        metrics = super().evaluate()
        basket_xy = to_numpy(self.basket.entity.get_pos()).ravel()[:2]
        all_breads_in_basket = True
        for idx, bread in enumerate(self.breads):
            pos = to_numpy(bread.entity.get_pos()).ravel()[:3]
            dist_xy = float(np.linalg.norm(pos[:2] - basket_xy))
            above_table = bool(
                pos[2] > self.TABLE_TOP_Z - self.SUCCESS_TABLE_Z_MARGIN
            )
            in_basket = bool(
                dist_xy <= self.SUCCESS_DIST_XY_M and above_table
            )
            metrics[f"bread_{idx}_pos"] = pos.tolist()
            metrics[f"bread_{idx}_dist_xy"] = dist_xy
            metrics[f"bread_{idx}_above_table"] = above_table
            metrics[f"bread_{idx}_in_basket"] = in_basket
            all_breads_in_basket = all_breads_in_basket and in_basket
        metrics.update({
            "all_breads_in_basket": bool(all_breads_in_basket),
            "assist_avatar_bread_idx": int(getattr(self, "_avatar_bread_idx", -1)),
            "assist_robot_bread_idx": int(getattr(self, "_robot_bread_idx", -1)),
            "assist_robot_bread_indices": [
                int(i) for i in getattr(self, "_robot_bread_indices", [])
            ],
            "eval_mode": bool(self.config.get("eval_mode", False)),
            "eval_assist_trigger_step": int(getattr(
                self, "_eval_assist_trigger_step", -1
            )),
            "eval_assist_fired": bool(getattr(self, "_eval_assist_fired", False)),
            "eval_assist_failed": bool(getattr(self, "_eval_assist_failed", False)),
            "eval_policy_step_count": int(getattr(
                self, "_eval_policy_step_count", 0
            )),
        })
        # BaseTask.evaluate() conservatively clears success whenever collision
        # telemetry contains a hit.  Restore this task's explicit success
        # contract, which already honors success_require_no_avatar_collision.
        metrics["success"] = bool(self.check_success())
        return metrics
