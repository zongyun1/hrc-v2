"""Interrupt variant for placing foods in a skillet.

The robot first places the skillet on the stove.  When it starts reaching
for one food item, a human avatar inspects that same item, sets it back on
the table, and the robot resumes from the item's latest physics pose.
"""

from __future__ import annotations

import numpy as np
import transforms3d as t3d

from ..avatar.inspect_motion_mixin import InspectMotionMixin, STANDARD_INSPECT_POOL
from ..grasp import tcp_to_link_pose
from ..utils import Pose
from ..task_bases.place_food_in_skillet import FoodSpec, PlaceFoodInSkillet


class PlaceFoodInSkilletInterrupt(InspectMotionMixin, PlaceFoodInSkillet):
    """Robot places food in a skillet while a human briefly inspects one item."""

    INSTRUCTION = (
        "place the skillet on the stove and put the foods in it; the human "
        "may inspect one food item while you are reaching for it"
    )
    use_avatar = True
    _force_dynamic_pan = True

    AVATAR_BASE_ROT = np.array(
        [[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=np.float64,
    )
    avatar_init_pos = np.array([0.0, 1.05, -0.34])
    avatar_init_rot = AVATAR_BASE_ROT

    INSPECT_MOTION_NAME = "Inspect1"
    INSPECT_ATTACH_FRAME = 34
    INSPECT_DETACH_FRAME = 187
    INSPECT_POOL = STANDARD_INSPECT_POOL
    AVATAR_MOTION_SLOW = 10.0
    AVATAR_HAND_ID = 1
    AVATAR_EXTRA_Z = 0.04

    EVAL_TRIGGER_STEP_MIN = 100
    EVAL_TRIGGER_STEP_MAX = 200
    EVAL_GLIDE_POLICY_STEPS = 20
    EVAL_FORCE_TRIGGER_DIST_XY = 0.11
    PACE_TRIGGER_PROGRESS = 0.82
    PACE_TRIGGER_DIST_XY = 0.14
    PACE_PROGRESS_START_DIST_XY = 0.42
    PACE_MIN_POLICY_STEPS = 20
    PACE_REQUIRE_PAN_READY = True
    PACE_PAN_READY_XY = 0.12
    EVAL_ROBOT_RETREAT_POLICY_STEPS = 35
    EVAL_ROBOT_RETREAT_BLEND = 0.50
    PACE_FALLBACK_TRIGGER_STEP = 240

    def __init__(self, config: dict = None):
        cfg = dict(config or {})
        cfg.setdefault("track_avatar_collision", True)
        super().__init__(cfg)
        if "force_dynamic_pan" in cfg:
            self._force_dynamic_pan = bool(cfg["force_dynamic_pan"])
        self._interrupted_idx = 0

    def reset(self, seed: int = 0):
        obs = super().reset(seed=seed)
        if self.avatar is None:
            return obs
        self.avatar.frame_ratio = self.AVATAR_MOTION_SLOW
        self._select_interrupted_food()
        if bool(self.config.get("eval_mode", False)):
            self._init_eval_interrupt_state()
        else:
            self._calibrate_avatar_for_inspect()
        return obs

    @property
    def _attach_frame_scaled(self) -> int:
        return int(round(self.INSPECT_ATTACH_FRAME * self.AVATAR_MOTION_SLOW))

    @property
    def _detach_frame_scaled(self) -> int:
        return int(round(self.INSPECT_DETACH_FRAME * self.AVATAR_MOTION_SLOW))

    def _select_interrupted_food(self) -> None:
        if not self.food_actors:
            self._interrupted_idx = 0
            return

        requested = self.config.get("interrupt_food", "burger")
        requested = None if requested is None else str(requested).lower()
        if requested == "random" or self.config.get("randomize_interrupt_food", False):
            self._interrupted_idx = int(np.random.randint(len(self.food_actors)))
        else:
            self._interrupted_idx = 0
            for idx, (_actor, spec) in enumerate(self.food_actors):
                if spec.name == requested:
                    self._interrupted_idx = idx
                    break

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

        self.avatar.play_animation(self.INSPECT_MOTION_NAME)
        palm_default = None
        step = 0
        attach_step = self._attach_frame_scaled
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
        for _ in range(20):
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

    def _calibrate_avatar_for_inspect(self):
        actor, spec = self.food_actors[self._interrupted_idx]
        target_palm = self._actor_center(actor, spec.asset_id, spec.model_id)

        palm_default = self._dryrun_capture_palm()
        if palm_default is None:
            return

        shift = target_palm - palm_default
        shift[2] += self.AVATAR_EXTRA_Z
        shifted_pos = np.asarray(self.avatar_init_pos, dtype=np.float64) + shift
        self.avatar.reset(
            shifted_pos.copy(),
            np.asarray(self.avatar_init_rot, dtype=np.float64).copy(),
        )
        for _ in range(30):
            self.step_sim()

    def _retreat_safe(self, arm_tag: str) -> None:
        arm = self.robot.get_arm(arm_tag)
        arm_base = np.array(arm.origin_pose.p, dtype=float)
        retreat_p = np.array([
            arm_base[0],
            arm_base[1] + 0.25,
            self.TABLE_TOP_Z + 0.30,
        ])
        r_tcp_down = np.array(
            [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]],
            dtype=np.float64,
        )
        retreat_link = tcp_to_link_pose(
            Pose(retreat_p, t3d.quaternions.mat2quat(r_tcp_down)),
            arm.tcp_offset,
        )
        self.move_and_execute(retreat_link.to_pose7(), arm_tag)

    def _wait_until_avatar_attach(self):
        motion = self.avatar.motion_modules.get(self.INSPECT_MOTION_NAME)
        if motion is None:
            return
        while (
            not self.avatar.spare()
            and int(getattr(motion, "at_frame", 0)) < self._attach_frame_scaled
        ):
            self.step_sim()

    def _move_above_food_intent(self, actor, spec: FoodSpec, arm_tag: str) -> None:
        arm = self.robot.get_arm(arm_tag)
        center = self._actor_center(actor, spec.asset_id, spec.model_id)
        above = np.array([
            center[0],
            center[1],
            self.TABLE_TOP_Z + self.FOOD_TRANSPORT_ABOVE_TABLE,
        ])
        above_link = tcp_to_link_pose(
            self._rotated_top_down_tcp(above, spec.tcp_yaw_deg),
            arm.tcp_offset,
        )
        self.open_gripper(arm_tag, num_steps=80)
        self._move_seeded(above_link.to_pose7(), arm_tag)

    def play_once(self) -> bool:
        arm_tag = "right"
        self._boost_finger_pd(arm_tag)
        with self.suppress_recording():  # pre-task settle, excluded from recordings
            for _ in range(60):
                self.step_sim()

        if self.avatar is None:
            return super().play_once()
        if not self._place_pan_on_cooktop(arm_tag):
            return False

        interrupted_actor, interrupted_spec = self.food_actors[self._interrupted_idx]
        self._move_above_food_intent(interrupted_actor, interrupted_spec, arm_tag)

        self._play_inspect_motion(
            attach_obj=interrupted_actor.entity,
            hand_id=self.AVATAR_HAND_ID,
            attach_frame=self._attach_frame_scaled,
            detach_frame=self._detach_frame_scaled,
            return_to_idle_after=None,  # freeze on last inspect frame (pose + position)
        )
        self._retreat_safe(arm_tag)
        self._wait_until_avatar_attach()
        for _ in range(100):
            self.step_sim()

        ok_any = False
        for idx, (actor, spec) in enumerate(self.food_actors):
            if idx == self._interrupted_idx:
                continue
            ok = self._pick_and_drop_food(actor, spec, arm_tag)
            ok_any = ok_any or ok

        if not self.avatar.spare():
            while not self.avatar.spare():
                self.step_sim()
        for _ in range(120):
            self.step_sim()

        ok = self._pick_and_drop_food(interrupted_actor, interrupted_spec, arm_tag)
        ok_any = ok_any or ok

        for _ in range(180):
            self.step_sim()
        return ok_any

    def _init_eval_interrupt_state(self) -> None:
        self._eval_palm_default = self._dryrun_capture_palm()
        if self._eval_palm_default is None:
            self._eval_interrupt_fired = True
        else:
            trigger_min = int(self.config.get(
                "eval_trigger_step_min",
                self.EVAL_TRIGGER_STEP_MIN,
            ))
            trigger_max = int(self.config.get(
                "eval_trigger_step_max",
                self.EVAL_TRIGGER_STEP_MAX,
            ))
            trigger_max = max(trigger_min, trigger_max)
            self._eval_trigger_step = int(np.random.randint(trigger_min, trigger_max + 1))
            self._eval_interrupt_fired = False
        self._eval_interrupt_trigger_mode = str(
            self.config.get("eval_interrupt_trigger_mode", "random")
        ).lower()
        self._eval_glide_active = False
        self._eval_glide_remaining = 0
        self._eval_glide_start_pos = None
        self._eval_glide_target_pos = None
        self._eval_target_food = None
        self._eval_retreat_active = False
        self._eval_retreat_remaining = 0
        self._eval_retreat_target = None
        self._eval_retreat_last_cmd = None
        self._eval_pending_avatar_start = None
        self._policy_step_count = 0
        self._pace_target_idx = None
        self._pace_initial_dist_xy = None
        self._pace_best_progress = 0.0
        self._pace_trigger_reason = None
        self._pace_trigger_step = None
        self._pace_trigger_progress = None
        self._pace_trigger_dist_xy = None
        self._eval_glide_policy_steps = int(self.config.get(
            "eval_glide_policy_steps",
            self.EVAL_GLIDE_POLICY_STEPS,
        ))
        self._eval_glide_policy_steps = max(1, self._eval_glide_policy_steps)

    def _eval_step_avatar(self) -> None:
        self._policy_step_count += 1

        if self._eval_retreat_active:
            self._eval_retreat_remaining -= 1
            if self._eval_retreat_remaining <= 0:
                self._eval_retreat_active = False
                self._start_eval_avatar_glide()
            return

        if self._eval_glide_active:
            self._eval_glide_remaining -= 1
            done = self._eval_glide_policy_steps - self._eval_glide_remaining
            progress = min(1.0, max(0.0, done / float(self._eval_glide_policy_steps)))
            cur = (
                (1.0 - progress) * self._eval_glide_start_pos
                + progress * self._eval_glide_target_pos
            )
            self.avatar.reset(cur.copy(), np.asarray(self.avatar_init_rot, dtype=np.float64).copy())
            if self._eval_glide_remaining <= 0:
                self._eval_glide_active = False
                actor, spec = self._eval_target_food
                food_pos = self._actor_center(actor, spec.asset_id, spec.model_id)
                if food_pos[2] < self.TABLE_TOP_Z - 0.05:
                    return
                self._play_inspect_motion(
                    attach_obj=actor.entity,
                    hand_id=self.AVATAR_HAND_ID,
                    attach_frame=self._attach_frame_scaled,
                    detach_frame=self._detach_frame_scaled,
                    return_to_idle_after=None,  # freeze on last inspect frame (pose + position)
                )
            return

        if self._eval_interrupt_fired:
            return
        target_idx, target_dist = self._closest_table_food_candidate()
        should_trigger = False
        trigger_reason = None
        mode = getattr(self, "_eval_interrupt_trigger_mode", "random")
        if mode == "pace":
            should_trigger, trigger_reason = self._pace_should_trigger(
                target_idx, target_dist,
            )
        else:
            force_trigger = (
                target_idx is not None
                and target_dist <= self.EVAL_FORCE_TRIGGER_DIST_XY
            )
            should_trigger = (
                self._policy_step_count >= self._eval_trigger_step
                or force_trigger
            )
            trigger_reason = "force_dist" if force_trigger else "random_step"

        if not should_trigger:
            return
        self._eval_interrupt_fired = True
        self._pace_trigger_reason = trigger_reason
        self._pace_trigger_step = int(self._policy_step_count)
        self._pace_trigger_progress = float(getattr(
            self, "_pace_best_progress", 0.0,
        ))
        self._pace_trigger_dist_xy = (
            None if not np.isfinite(target_dist) else float(target_dist)
        )

        if target_idx is None:
            return

        actor, spec = self.food_actors[target_idx]
        target_pos = self._actor_center(actor, spec.asset_id, spec.model_id)
        shift = target_pos - self._eval_palm_default
        shift[2] += self.AVATAR_EXTRA_Z
        target_avatar_pos = np.asarray(self.avatar_init_pos, dtype=np.float64) + shift

        self._eval_pending_avatar_start = target_avatar_pos.copy()
        self._eval_target_food = (actor, spec)
        self._start_eval_robot_retreat()

    def _start_eval_robot_retreat(self) -> None:
        steps = int(self.config.get(
            "eval_robot_retreat_policy_steps",
            self.EVAL_ROBOT_RETREAT_POLICY_STEPS,
        ))
        if steps <= 0:
            self._start_eval_avatar_glide()
            return
        arm = self.robot.get_arm("right")
        self._eval_retreat_target = np.asarray(arm.homestate, dtype=np.float64)
        self._eval_retreat_remaining = steps
        self._eval_retreat_active = True

    def _apply_eval_robot_retreat(self) -> bool:
        if not self._eval_retreat_active or self._eval_retreat_target is None:
            return False
        arm = self.robot.get_arm("right")
        current = np.asarray(arm.get_arm_qpos(), dtype=np.float64)
        target = np.asarray(self._eval_retreat_target, dtype=np.float64)
        alpha = float(self.config.get(
            "eval_robot_retreat_blend",
            self.EVAL_ROBOT_RETREAT_BLEND,
        ))
        alpha = float(np.clip(alpha, 0.0, 1.0))
        cmd = current * (1.0 - alpha) + target * alpha
        self._eval_retreat_last_cmd = cmd.copy()
        self.robot.set_arm_joints(cmd, "right")
        self.robot.set_gripper(1.0, "right")
        return True

    def _start_eval_avatar_glide(self) -> None:
        if self._eval_pending_avatar_start is None or self._eval_target_food is None:
            return
        self._eval_glide_start_pos = np.asarray(self.avatar_init_pos, dtype=np.float64).copy()
        self._eval_glide_target_pos = np.asarray(
            self._eval_pending_avatar_start, dtype=np.float64,
        ).copy()
        self._eval_glide_remaining = self._eval_glide_policy_steps
        self._eval_glide_active = True
        self._eval_pending_avatar_start = None

    def _pace_should_trigger(self, target_idx, target_dist):
        """PACE-style action-completion trigger for ACT eval.

        The scheduler waits until the pan is near the cooktop, then estimates
        completion of the robot's current approach-to-food action from
        EE-to-food xy distance. ACT itself remains unchanged.
        """
        if target_idx is None or not np.isfinite(target_dist):
            return False, None
        if bool(self.config.get(
            "pace_require_pan_ready", self.PACE_REQUIRE_PAN_READY,
        )) and not self._pace_pan_ready():
            fallback_step = int(self.config.get(
                "pace_fallback_trigger_step", self.PACE_FALLBACK_TRIGGER_STEP,
            ))
            if (
                fallback_step > 0
                and self._policy_step_count >= fallback_step
                and target_idx is not None
            ):
                return True, "pace_fallback_step"
            return False, None
        min_steps = int(self.config.get(
            "pace_min_policy_steps", self.PACE_MIN_POLICY_STEPS,
        ))
        if self._policy_step_count < min_steps:
            return False, None

        start_dist = float(self.config.get(
            "pace_progress_start_dist_xy", self.PACE_PROGRESS_START_DIST_XY,
        ))
        if self._pace_target_idx != target_idx:
            self._pace_target_idx = int(target_idx)
            self._pace_initial_dist_xy = max(float(target_dist), start_dist)
            self._pace_best_progress = 0.0

        denom = max(1e-6, float(self._pace_initial_dist_xy))
        progress = 1.0 - float(target_dist) / denom
        progress = float(np.clip(progress, 0.0, 1.0))
        self._pace_best_progress = max(float(self._pace_best_progress), progress)

        trigger_progress = float(self.config.get(
            "pace_trigger_progress", self.PACE_TRIGGER_PROGRESS,
        ))
        trigger_dist = float(self.config.get(
            "pace_trigger_dist_xy", self.PACE_TRIGGER_DIST_XY,
        ))
        if self._pace_best_progress >= trigger_progress:
            return True, "pace_progress"
        if target_dist <= trigger_dist:
            return True, "pace_dist"
        return False, None

    def _pace_pan_ready(self) -> bool:
        bowl = self._pan_bowl_center()
        dxy = float(np.linalg.norm(bowl[:2] - self._cooktop_xy))
        threshold = float(self.config.get(
            "pace_pan_ready_xy", self.PACE_PAN_READY_XY,
        ))
        return bool(dxy <= threshold and bowl[2] > self.TABLE_TOP_Z)

    def _find_closest_table_food(self):
        best_idx, _ = self._closest_table_food_candidate()
        return best_idx

    def _closest_table_food_candidate(self):
        arm = self.robot.get_arm("right")
        ee_pos = np.array(arm.get_ee_pose()[:3], dtype=float)
        bowl_xy = self._pan_bowl_center()[:2]

        best_idx = None
        best_d = float("inf")
        for idx, (actor, spec) in enumerate(self.food_actors):
            p = self._actor_center(actor, spec.asset_id, spec.model_id)
            in_skillet = (
                np.linalg.norm(p[:2] - bowl_xy) < self.FOOD_SUCCESS_XY
                and p[2] > self.TABLE_TOP_Z + 0.02
            )
            if in_skillet:
                continue
            if p[2] < self.TABLE_TOP_Z - 0.05 or p[2] > self.TABLE_TOP_Z + 0.30:
                continue
            d = float(np.linalg.norm(p[:2] - ee_pos[:2]))
            if d < best_d:
                best_idx = idx
                best_d = d
        return best_idx, best_d

    def check_success(self) -> bool:
        if self.avatar_collided:
            return False
        return super().check_success()

    def evaluate(self) -> dict:
        out = super().evaluate()
        if bool(self.config.get("eval_mode", False)):
            out["pace_interrupt"] = {
                "enabled": getattr(
                    self, "_eval_interrupt_trigger_mode", "random",
                ) == "pace",
                "trigger_mode": getattr(
                    self, "_eval_interrupt_trigger_mode", "random",
                ),
                "fired": bool(getattr(self, "_eval_interrupt_fired", False)),
                "trigger_step": getattr(self, "_pace_trigger_step", None),
                "trigger_reason": getattr(self, "_pace_trigger_reason", None),
                "trigger_progress": getattr(
                    self, "_pace_trigger_progress", None,
                ),
                "trigger_dist_xy": getattr(
                    self, "_pace_trigger_dist_xy", None,
                ),
                "best_progress": float(getattr(
                    self, "_pace_best_progress", 0.0,
                )),
                "target_idx": getattr(self, "_pace_target_idx", None),
                "policy_steps": int(getattr(self, "_policy_step_count", 0)),
                "pan_ready": self._pace_pan_ready(),
                "retreat_active": bool(getattr(
                    self, "_eval_retreat_active", False,
                )),
            }
        return out

    def take_action(self, action, action_type: str = "qpos"):
        if self.avatar is not None and bool(self.config.get("eval_mode", False)):
            self._eval_step_avatar()
            if self._apply_eval_robot_retreat():
                retreat_cmd = getattr(self, "_eval_retreat_last_cmd", None)
                if retreat_cmd is None:
                    retreat_cmd = self.robot.get_arm("right").get_arm_qpos()
                action = np.concatenate([
                    np.asarray(retreat_cmd, dtype=np.float64),
                    [1.0],
                ])
                action_type = "qpos_abs"
                return PlaceFoodInSkillet.take_action(self, action, action_type=action_type)
        return super().take_action(action, action_type=action_type)
