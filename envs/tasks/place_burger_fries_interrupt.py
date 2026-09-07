"""Place burger and fries (interrupt): same scene as
``place_burger_fries`` but with a human avatar that picks up one of the
food items mid-task.

Flow per episode:
  1. Robot approaches the interrupted item (above-pose, no commit).
  2. Avatar plays the ``Inspect`` motion with ``attach_obj`` = the
     interrupted item.  Avatar lifts, holds, and sets it back on the table.
  3. Robot retreats so it doesn't collide with the avatar.
  4. While the avatar holds the interrupted item, the robot picks the
     OTHER food items (in food_actors order, skipping the interrupted
     index) and places them on the tray.
  5. Robot waits for ``avatar.spare()``.
  6. Robot reads the interrupted item's LATEST pose (after the avatar
     puts it back down) and picks it.

All food items spawn in the same avatar-reachable band so placement does not
reveal which one the human will inspect.
"""

from __future__ import annotations

import numpy as np

from ..avatar.inspect_motion_mixin import InspectMotionMixin, STANDARD_INSPECT_POOL
from ..utils import to_numpy
from ..task_bases.place_burger_fries import PlaceBurgerFries


class PlaceBurgerFriesInterrupt(InspectMotionMixin, PlaceBurgerFries):
    INSTRUCTION = "place the hamburger and the french fries on the tray; if the human inspects one item midway through, wait and then continue"

    use_avatar = True

    AVATAR_BASE_ROT = np.array(
        [[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=np.float64
    )
    avatar_init_pos = np.array([0.05, 0.55, -0.18])
    avatar_init_rot = AVATAR_BASE_ROT

    INSPECT_MOTION_NAME = "Inspect1"
    INSPECT_ATTACH_FRAME = 34
    INSPECT_DETACH_FRAME = 187
    INSPECT_POOL = STANDARD_INSPECT_POOL
    AVATAR_MOTION_SLOW = 10.0
    AVATAR_HAND_ID = 1
    AVATAR_EXTRA_Z = 0.04

    AVATAR_CLEARANCE = 0.20
    AVATAR_MIN_Y = 0.10
    CALIBRATION_RESET_SETTLE_STEPS = 20
    CALIBRATION_SETTLE_STEPS = 30
    RETREAT_Y_FROM_BASE = 0.25
    RETREAT_Z_ABOVE_TABLE = 0.30
    PRE_INTERRUPT_SETTLE_STEPS = 80
    # None ⇒ avatar freezes on the inspect motion's last frame (pose + position).
    INSPECT_RETURN_TO_IDLE_FRAMES = None
    PRE_ATTACH_WAIT_MARGIN_STEPS = 100
    POST_FAILED_PICK_SETTLE_STEPS = 40
    POST_RELEASE_SETTLE_STEPS = 120
    EVAL_TRIGGER_STEP_MIN = 20
    EVAL_TRIGGER_STEP_MAX = 100
    EVAL_INTERRUPT_DISTANCE_M = 0.18
    DRYRUN_RESET_SETTLE_STEPS = 20

    # Lift higher than parent (0.22) so the long horizontal transit (which
    # passes near previously-placed items on the tray when the avatar is in
    # scene and routes the arm through atypical configs) doesn't brush them.
    TRANSPORT_Z_ABOVE_TABLE = 0.32

    # Shared food spawn band near the +Y side of the table.  Every food item
    # uses this band so the robot cannot infer the interrupt target from
    # which item is uniquely close enough for the avatar to inspect.
    FOOD_REACH_X_RANGE = (-0.05, 0.20)
    FOOD_REACH_Y_RANGE = (-0.10, 0.06)
    INTERRUPTED_REACHABLE_Y = FOOD_REACH_Y_RANGE[0]

    def __init__(self, config: dict = None):
        super().__init__(config)
        # Index into self.food_actors of the item the avatar picks up.
        self._interrupted_food_idx: int = 0

    def reset(self, seed: int = 0):
        obs = super().reset(seed=seed)
        if self.avatar is not None:
            self.avatar.frame_ratio = self.AVATAR_MOTION_SLOW
            self._calibrate_avatar_for_inspect()
        self._eval_policy_step_count = 0
        self._eval_interrupt_fired = False
        self._eval_interrupt_failed = False
        self._eval_trigger_step = None
        if self.avatar is not None and bool(self.config.get("eval_mode", False)):
            lo = int(self.config.get("eval_trigger_step_min", self.EVAL_TRIGGER_STEP_MIN))
            hi = int(self.config.get("eval_trigger_step_max", self.EVAL_TRIGGER_STEP_MAX))
            if hi < lo:
                hi = lo
            self._eval_trigger_step = int(np.random.randint(lo, hi + 1))
        return obs

    def _sample_spawn_xy(self, xy_range, occupied, max_tries: int = None):
        return super()._sample_spawn_xy(
            (self.FOOD_REACH_X_RANGE, self.FOOD_REACH_Y_RANGE),
            occupied,
            max_tries=max_tries,
        )

    def load_actors(self):
        super().load_actors()

        # All spawned foods are in the shared avatar-reachable band; choose
        # uniformly so placement does not leak which item will be inspected.
        self._interrupted_food_idx = int(np.random.randint(len(self.food_spawn_xys)))

    @property
    def _attach_frame_scaled(self):
        return int(round(self.INSPECT_ATTACH_FRAME * self.AVATAR_MOTION_SLOW))

    @property
    def _detach_frame_scaled(self):
        return int(round(self.INSPECT_DETACH_FRAME * self.AVATAR_MOTION_SLOW))

    def _interrupted_pair(self):
        return self.food_actors[self._interrupted_food_idx]

    def take_action(self, action, action_type: str = "qpos"):
        if self.avatar is not None and bool(self.config.get("eval_mode", False)):
            self._eval_step_avatar()
        return super().take_action(action, action_type=action_type)

    def _closest_food_to_ee(self):
        if not getattr(self, "food_actors", None):
            return None, float("inf")
        ee = np.asarray(self.robot.get_arm("right").get_ee_pose()[:3], dtype=float)
        best_i = None
        best_d = float("inf")
        for i, (actor, _spec) in enumerate(self.food_actors):
            p = to_numpy(actor.entity.get_pos()).ravel()[:3].astype(float)
            if p[2] < self.TABLE_TOP_Z - 0.02:
                continue
            d = float(np.linalg.norm(ee[:2] - p[:2]))
            if d < best_d:
                best_i = i
                best_d = d
        return best_i, best_d

    def _eval_step_avatar(self):
        self._eval_policy_step_count = int(getattr(
            self, "_eval_policy_step_count", 0,
        )) + 1
        if getattr(self, "_eval_interrupt_fired", False):
            return
        trigger = getattr(self, "_eval_trigger_step", None)
        if trigger is None or self._eval_policy_step_count < int(trigger):
            return
        idx, dist = self._closest_food_to_ee()
        threshold = float(self.config.get(
            "eval_interrupt_distance_m", self.EVAL_INTERRUPT_DISTANCE_M,
        ))
        # Prefer proximity-triggered interrupts; force a fallback at the
        # sampled trigger so random-policy videos still exercise the avatar.
        if idx is None:
            self._eval_interrupt_fired = True
            self._eval_interrupt_failed = True
            return
        if dist > threshold and not bool(self.config.get("eval_interrupt_force_at_trigger", True)):
            return
        self._eval_interrupt_fired = True
        self._interrupted_food_idx = int(idx)
        try:
            self._calibrate_avatar_for_inspect()
            int_actor, _int_spec = self._interrupted_pair()
            self._play_inspect_motion(
                attach_obj=int_actor.entity,
                hand_id=self.AVATAR_HAND_ID,
                attach_frame=self._attach_frame_scaled,
                detach_frame=self._detach_frame_scaled,
                return_to_idle_after=self.INSPECT_RETURN_TO_IDLE_FRAMES,
            )
        except Exception as exc:
            self._eval_interrupt_failed = True
            if self.config.get("debug_eval_avatar", False):
                print(f"[burger_fries_interrupt/eval] interrupt failed: {exc}", flush=True)

    def _calibrate_avatar_for_inspect(self):
        actor, _ = self._interrupted_pair()
        target_palm = to_numpy(actor.entity.get_pos()).ravel()[:3].astype(float)

        palm_default = self._dryrun_capture_palm()
        if palm_default is None:
            self.avatar.reset(
                np.asarray(self.avatar_init_pos, dtype=np.float64).copy(),
                np.asarray(self.avatar_init_rot, dtype=np.float64).copy(),
            )
            for _ in range(self.CALIBRATION_RESET_SETTLE_STEPS):
                self.step_sim()
            return

        shift = target_palm - palm_default
        shift[2] += self.AVATAR_EXTRA_Z
        shifted_pos = np.asarray(self.avatar_init_pos, dtype=np.float64) + shift

        self.avatar.reset(
            shifted_pos.copy(),
            np.asarray(self.avatar_init_rot, dtype=np.float64).copy(),
        )
        for _ in range(self.CALIBRATION_SETTLE_STEPS):
            self.step_sim()

        self.avatar_collided = False
        self.avatar_collision_log = []

    def _dryrun_capture_palm(self):
        cache_key = (
            self.INSPECT_MOTION_NAME,
            float(self.AVATAR_MOTION_SLOW),
            self.AVATAR_HAND_ID,
            tuple(np.asarray(self.avatar_init_pos, dtype=np.float64).ravel()),
        )
        cached = getattr(type(self), "_palm_default_cache", {}).get(cache_key)
        if cached is not None:
            return cached.copy()

        self.avatar.reset(
            np.asarray(self.avatar_init_pos, dtype=np.float64).copy(),
            np.asarray(self.avatar_init_rot, dtype=np.float64).copy(),
        )
        self.avatar.play_animation(self.INSPECT_MOTION_NAME)
        palm_default = None
        attach_step = self._attach_frame_scaled
        step = 0
        while step <= attach_step and not self.avatar.spare():
            self.step_sim()
            if step == attach_step:
                palm_default = np.asarray(
                    self.avatar.robot.get_palm_center(self.AVATAR_HAND_ID),
                    dtype=np.float64,
                ).copy()
                break
            step += 1
        self.avatar.reset(
            np.asarray(self.avatar_init_pos, dtype=np.float64).copy(),
            np.asarray(self.avatar_init_rot, dtype=np.float64).copy(),
        )
        for _ in range(self.DRYRUN_RESET_SETTLE_STEPS):
            self.step_sim()
        self.avatar_collided = False
        self.avatar_collision_log = []
        if palm_default is not None:
            cache = getattr(type(self), "_palm_default_cache", None)
            if cache is None:
                cache = {}
                type(self)._palm_default_cache = cache
            cache[cache_key] = palm_default.copy()
        return palm_default

    def _inspect_palm_at_frame(self, frame_idx: int):
        """Measure inspect-pose palm position without playing a visible motion."""
        if self.avatar is None:
            return None
        if self.INSPECT_MOTION_NAME not in self.avatar.motion_modules:
            self.avatar.play_animation(self.INSPECT_MOTION_NAME)
        motion = self.avatar.motion_modules.get(self.INSPECT_MOTION_NAME)
        if motion is None or not getattr(motion, "data", None):
            return None
        idx = int(np.clip(int(frame_idx), 0, len(motion.data) - 1))
        self.avatar.robot.pose = motion.data[idx]
        self.avatar.robot.node_trans = motion.node_data[idx]
        self.avatar.robot.global_mat = motion.global_mat
        self.avatar.robot.global_mat_inv = motion.global_mat_inv
        self.avatar.robot.update()
        palm = np.asarray(
            self.avatar.robot.get_palm_center(self.AVATAR_HAND_ID),
            dtype=np.float64,
        ).copy()
        self.avatar.reset(
            np.asarray(self.avatar_init_pos, dtype=np.float64).copy(),
            np.asarray(self.avatar_init_rot, dtype=np.float64).copy(),
        )
        return palm

    def _retreat_safe(self, arm_tag: str):
        arm = self.robot.get_arm(arm_tag)
        arm_base = np.array(arm.origin_pose.p, dtype=float)
        retreat_z = self.TABLE_TOP_Z + self.RETREAT_Z_ABOVE_TABLE
        retreat_pos = np.array([
            arm_base[0],
            arm_base[1] + self.RETREAT_Y_FROM_BASE,
            retreat_z,
        ])
        self._move_screw(retreat_pos, arm_tag)

    def _wait_until_avatar_attach(self):
        motion = self.avatar.motion_modules.get(self.INSPECT_MOTION_NAME)
        if motion is None:
            return
        while (
            not self.avatar.spare()
            and int(getattr(motion, "at_frame", 0)) < self._attach_frame_scaled
        ):
            self.step_sim()

    def play_once(self) -> bool:
        from ..grasp import tcp_to_link_pose

        arm_tag = "right"
        self._boost_finger_pd(arm_tag)
        arm = self.robot.get_arm(arm_tag)
        tcp_offset = arm.tcp_offset

        for _ in range(self.PRE_INTERRUPT_SETTLE_STEPS):
            self.step_sim()

        if self.avatar is None:
            return super().play_once()

        int_actor, int_spec = self._interrupted_pair()
        int_pos = to_numpy(int_actor.entity.get_pos()).ravel()[:3].astype(float)

        n_total = len(self.food_actors)
        slots = self._drop_slots(n_total)
        tray_x, tray_y = self._tray_xy
        transport_z = self.TABLE_TOP_Z + self.TRANSPORT_Z_ABOVE_TABLE

        # ---- Phase 1: robot shows intent on the interrupted item ----
        self.open_gripper(arm_tag)
        above_pos = np.array([int_pos[0], int_pos[1], transport_z])
        above_link = tcp_to_link_pose(self._top_down_tcp(above_pos), tcp_offset)
        self._move_seeded(above_link.to_pose7(), arm_tag)

        # ---- Phase 2: avatar plays Inspect with attach ----
        self._play_inspect_motion(
            attach_obj=int_actor.entity,
            hand_id=self.AVATAR_HAND_ID,
            attach_frame=self._attach_frame_scaled,
            detach_frame=self._detach_frame_scaled,
            return_to_idle_after=self.INSPECT_RETURN_TO_IDLE_FRAMES,
        )

        # ---- Phase 3: robot retreats so the avatar can grab ----
        self._retreat_safe(arm_tag)
        self._wait_until_avatar_attach()
        for _ in range(self.PRE_ATTACH_WAIT_MARGIN_STEPS):
            self.step_sim()

        # ---- Phase 4: pick the OTHER items (pure physics) ----
        self.plan_success = True

        for i, (actor, spec) in enumerate(self.food_actors):
            if i == self._interrupted_food_idx:
                continue
            dx, dy = slots[i]
            try:
                ok = self._pick_and_drop(
                    actor, spec.asset_id, arm_tag,
                    close_value=spec.close_value,
                    below_center=spec.below_center,
                    drop_xy=(tray_x + dx, tray_y + dy),
                    tcp_yaw_deg=spec.tcp_yaw_deg,
                    transit_steps=spec.transit_steps,
                    transit_sim=spec.transit_sim,
                    align_to_object_yaw=spec.align_to_object_yaw,
                )
            except Exception:
                ok = False
            if not ok:
                self.open_gripper(arm_tag)
                for _ in range(self.POST_FAILED_PICK_SETTLE_STEPS):
                    self.step_sim()

        # ---- Phase 5: wait for avatar to release ----
        if not self.avatar.spare():
            while not self.avatar.spare():
                self.step_sim()
        for _ in range(self.POST_RELEASE_SETTLE_STEPS):
            self.step_sim()

        # ---- Phase 6: pick the released item using its LATEST pose ----
        dx, dy = slots[self._interrupted_food_idx]
        try:
            ok = self._pick_and_drop(
                int_actor, int_spec.asset_id, arm_tag,
                close_value=int_spec.close_value,
                below_center=int_spec.below_center,
                drop_xy=(tray_x + dx, tray_y + dy),
                tcp_yaw_deg=int_spec.tcp_yaw_deg,
                transit_steps=int_spec.transit_steps,
                transit_sim=int_spec.transit_sim,
                align_to_object_yaw=int_spec.align_to_object_yaw,
            )
        except Exception:
            ok = False
        if not ok:
            self.open_gripper(arm_tag)
            for _ in range(self.POST_FAILED_PICK_SETTLE_STEPS):
                self.step_sim()

        for _ in range(self.FINAL_SETTLE_STEPS):
            self.step_sim()
        return True

    def evaluate(self) -> dict:
        metrics = super().evaluate()
        trigger_step = getattr(self, "_eval_trigger_step", -1)
        if trigger_step is None:
            trigger_step = -1
        metrics.update({
            "eval_mode": bool(self.config.get("eval_mode", False)),
            "eval_trigger_step": int(trigger_step),
            "eval_policy_step_count": int(getattr(
                self, "_eval_policy_step_count", 0,
            )),
            "eval_interrupt_fired": bool(getattr(
                self, "_eval_interrupt_fired", False,
            )),
            "eval_interrupt_failed": bool(getattr(
                self, "_eval_interrupt_failed", False,
            )),
            "interrupted_food_idx": int(getattr(
                self, "_interrupted_food_idx", -1,
            )),
        })
        return metrics
