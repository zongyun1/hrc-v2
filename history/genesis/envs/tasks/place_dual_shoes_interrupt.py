"""Place dual shoes (interrupt).

Same receiver and pure-physics shoe primitive as ``place_dual_shoes``, but a
human avatar inspects one shoe midway through.  The robot approaches the
interrupted shoe, yields while the avatar picks it up and puts it back, places
the other shoe, then resumes from the released shoe's live pose.
"""

import numpy as np
import transforms3d as t3d

from ..avatar.inspect_motion_mixin import InspectMotionMixin, STANDARD_INSPECT_POOL
from ..grasp import tcp_to_link_pose
from ..utils import Pose, to_numpy
from ..task_bases.place_dual_shoes import (
    PlaceDualShoes, _REGION_X,
)


class PlaceDualShoesInterrupt(InspectMotionMixin, PlaceDualShoes):
    """Robot puts two shoes in a shoebox while the human inspects one shoe."""

    INSTRUCTION = (
        "put the shoes in the shoebox; the human will inspect one shoe "
        "midway through - wait, then continue"
    )
    use_avatar = True

    # Avatar at +Y side of the table, facing -Y toward the table.
    AVATAR_BASE_ROT = np.array(
        [[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=np.float64,
    )
    # Inspect's calibrated ready stance for shoes is behind and lower than
    # the generic bread/dump interrupt home pose.  Starting closer to that
    # stance avoids a large visible glide before the avatar begins motion.
    avatar_init_pos = np.array([0.0, 0.82, -0.34])
    avatar_init_rot = AVATAR_BASE_ROT

    INSPECT_MOTION_NAME = "Inspect1"
    INSPECT_ATTACH_FRAME = 34
    INSPECT_DETACH_FRAME = 187
    INSPECT_POOL = STANDARD_INSPECT_POOL
    AVATAR_MOTION_SLOW = 10.0
    AVATAR_HAND_ID = 1
    AVATAR_EXTRA_Z = 0.04

    # Shared shoe spawn region near the +Y table edge.  Both shoes are
    # sampled here so the robot cannot infer the interrupt target from
    # which shoe is uniquely close enough for the avatar to inspect.
    BOX_X_LO, BOX_X_HI = -0.18, 0.10
    BOX_Y_LO, BOX_Y_HI = -0.34, -0.12
    # Shoe spawn band must be reachable by BOTH the avatar's Inspect clip
    # (shoulder ~y=+0.82) AND the Franka top-down pick (base y=-0.65, reach
    # ~0.62 m).  The old band (-0.08,+0.06) sat at 0.57-0.71 m from the robot
    # base, so most shoes were beyond the top-down reach and every final
    # robot pick pre-RRT-failed (diag sweep 61293938: interrupt s2/s3/s4 all
    # reach fails -> 0/5).  Pull the band -Y into robot reach; the Inspect
    # clip leans far enough to still reach it (validate on re-sweep).  Narrow
    # x so the far corners stay inside the ~0.62 m reach circle.
    SHOE_REACH_X_LO, SHOE_REACH_X_HI = -0.08, 0.08
    SHOE_REACH_Y_LO, SHOE_REACH_Y_HI = -0.18, -0.06
    INT_X_LO, INT_X_HI = SHOE_REACH_X_LO, SHOE_REACH_X_HI
    INT_Y_LO, INT_Y_HI = SHOE_REACH_Y_LO, SHOE_REACH_Y_HI
    OTHER_Y_LO, OTHER_Y_HI = SHOE_REACH_Y_LO, SHOE_REACH_Y_HI
    SHOE_MIN_SEP_M = 0.18
    BOX_SHOE_MIN_SEP_M = 0.32

    # Eval/testbed mode: policy calls take_action(action) -> obs while the
    # avatar fires the Inspect interrupt once at a random policy step.
    EVAL_TRIGGER_STEP_MIN = 100
    EVAL_TRIGGER_STEP_MAX = 200
    EVAL_FORCE_TRIGGER_DIST_XY = 0.12
    EVAL_GLIDE_POLICY_STEPS = 20
    EVAL_ROBOT_RETREAT_POLICY_STEPS = 35
    EVAL_ROBOT_RETREAT_BLEND = 0.50
    EVAL_ROBOT_RETURN_POLICY_STEPS = 35
    EVAL_ROBOT_RETURN_BLEND = 0.50
    PACE_TRIGGER_PROGRESS = 0.82
    PACE_TRIGGER_DIST_XY = 0.14
    PACE_PROGRESS_START_DIST_XY = 0.42
    PACE_MIN_POLICY_STEPS = 20
    LAYOUT_SAMPLE_ATTEMPTS = 80
    DRYRUN_RESET_SETTLE_STEPS = 20
    CALIBRATION_SETTLE_STEPS = 30
    RETREAT_Y_FROM_BASE = 0.25
    RETREAT_Z_ABOVE_TABLE = 0.30
    INTENT_Z_ABOVE_TABLE = 0.20
    INITIAL_SETTLE_STEPS = 60
    # None ⇒ avatar freezes on the inspect motion's last frame (pose + position).
    INSPECT_RETURN_TO_IDLE_FRAMES = None
    PRE_ATTACH_WAIT_MARGIN_STEPS = 100
    POST_AVATAR_RELEASE_SETTLE_STEPS = 150
    EVAL_TABLE_Z_LOWER_MARGIN = 0.05
    EVAL_TABLE_Z_UPPER_MARGIN = 0.35

    def __init__(self, config: dict = None):
        cfg = dict(config or {})
        cfg.setdefault("track_avatar_collision", True)
        super().__init__(cfg)

    # ------------------------------------------------------------------
    # Sampling: both shoes use the same avatar-reachable region; the box keeps
    # its separate robot-side region.
    # ------------------------------------------------------------------

    def _sample_layout_positions(self, rng):
        n_shoes = max(1, int(getattr(self, "_num_shoes", 1)))
        self._interrupted_idx = int(rng.randint(0, n_shoes))
        for _ in range(self.LAYOUT_SAMPLE_ATTEMPTS):
            box = (
                float(rng.uniform(self.BOX_X_LO, self.BOX_X_HI)),
                float(rng.uniform(self.BOX_Y_LO, self.BOX_Y_HI)),
            )
            shoes = [
                (
                    float(rng.uniform(self.SHOE_REACH_X_LO, self.SHOE_REACH_X_HI)),
                    float(rng.uniform(self.SHOE_REACH_Y_LO, self.SHOE_REACH_Y_HI)),
                )
                for _ in range(n_shoes)
            ]
            positions = [box] + shoes
            if self._layout_clear(positions):
                return positions

        fallback = [(-0.18, -0.34), (0.16, 0.02), (-0.04, 0.04)]
        return fallback[: 1 + n_shoes]

    def _layout_clear(self, positions) -> bool:
        box = positions[0]
        shoes = positions[1:]
        for shoe in shoes:
            dx = box[0] - shoe[0]
            dy = box[1] - shoe[1]
            if dx * dx + dy * dy < self.BOX_SHOE_MIN_SEP_M ** 2:
                return False
        for i in range(len(shoes)):
            for j in range(i + 1, len(shoes)):
                dx = shoes[i][0] - shoes[j][0]
                dy = shoes[i][1] - shoes[j][1]
                if dx * dx + dy * dy < self.SHOE_MIN_SEP_M ** 2:
                    return False
        return True

    # ------------------------------------------------------------------
    # Reset-time avatar calibration.
    # ------------------------------------------------------------------

    def reset(self, seed: int = 0):
        obs = super().reset(seed=seed)
        if self.avatar is None:
            if (
                bool(self.config.get("eval_mode", False))
                and bool(self.config.get("pace_no_human_ablation", False))
            ):
                self._init_eval_interrupt_state()
            return obs
        self.avatar.frame_ratio = self.AVATAR_MOTION_SLOW
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
        # Calibration only needs the palm pose at attach. Stop the dry-run
        # as soon as that frame is reached so reset does not play a full
        # visible inspect pass before the real interruption.
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

    def _calibrate_avatar_for_inspect(self):
        shoe = self.shoes[self._interrupted_idx]
        target_palm = np.asarray(self._shoe_pos_quat(shoe)[0], dtype=float)
        palm_default = self._dryrun_capture_palm()
        if palm_default is None:
            return

        shift = target_palm - palm_default
        shift[2] += self.AVATAR_EXTRA_Z
        shifted_pos = (
            np.asarray(self.avatar_init_pos, dtype=np.float64) + shift
        )
        self.avatar.reset(
            shifted_pos.copy(),
            np.asarray(self.avatar_init_rot, dtype=np.float64).copy(),
        )
        for _ in range(self.CALIBRATION_SETTLE_STEPS):
            self.step_sim()

    # ------------------------------------------------------------------
    # Robot choreography.
    # ------------------------------------------------------------------

    def _retreat_safe(self, arm_tag: str):
        arm = self.robot.get_arm(arm_tag)
        tcp_offset = arm.tcp_offset
        arm_base = np.asarray(arm.origin_pose.p, dtype=float)
        retreat_p = np.array([
            arm_base[0],
            arm_base[1] + self.RETREAT_Y_FROM_BASE,
            self.TABLE_TOP_Z + self.RETREAT_Z_ABOVE_TABLE,
        ])
        q_tcp_down = t3d.quaternions.mat2quat(np.array(
            [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]],
            dtype=np.float64,
        ))
        retreat_link = tcp_to_link_pose(
            Pose(retreat_p, q_tcp_down), tcp_offset,
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

    def _approach_interrupted_shoe(self, shoe, arm_tag: str) -> bool:
        arm = self.robot.get_arm(arm_tag)
        tcp_offset = arm.tcp_offset
        q_tcp_down = t3d.quaternions.mat2quat(np.array(
            [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]],
            dtype=np.float64,
        ))
        shoe_pos = np.asarray(self._shoe_pos_quat(shoe)[0])
        above_p = np.array([
            shoe_pos[0],
            shoe_pos[1],
            self.TABLE_TOP_Z + self.INTENT_Z_ABOVE_TABLE,
        ])
        above_link = tcp_to_link_pose(Pose(above_p, q_tcp_down), tcp_offset)
        return self.move_and_execute(above_link.to_pose7(), arm_tag) is not None

    def take_action(self, action, action_type: str = "qpos"):
        if bool(self.config.get("eval_mode", False)) and (
            self.avatar is not None
            or bool(self.config.get("pace_no_human_ablation", False))
        ):
            self._eval_step_avatar()
            if self._apply_eval_pace_override():
                override_cmd = getattr(self, "_eval_override_last_cmd", None)
                if override_cmd is None:
                    override_cmd = getattr(self, "_eval_return_last_cmd", None)
                if override_cmd is None:
                    override_cmd = getattr(self, "_eval_retreat_last_cmd", None)
                if override_cmd is None:
                    override_cmd = self.robot.get_arm("right").get_arm_qpos()
                action = np.concatenate([
                    np.asarray(override_cmd, dtype=np.float64),
                    [float(getattr(self, "_eval_saved_gripper", 1.0))],
                ])
                action_type = "qpos_abs"
                return PlaceDualShoes.take_action(
                    self, action, action_type=action_type,
                )
        return super().take_action(action, action_type=action_type)

    def play_once(self) -> bool:
        with self.suppress_recording():  # pre-task settle, excluded from recordings
            for _ in range(self.INITIAL_SETTLE_STEPS):
                self.step_sim()

        if not self.shoes or self.shoebox is None:
            return False
        if self.avatar is None:
            return super().play_once()

        arm_tag = "right"
        interrupted_idx = int(self._interrupted_idx)
        int_shoe = self.shoes[interrupted_idx]
        int_label = chr(ord("A") + interrupted_idx)

        self.open_gripper(arm_tag)
        self._approach_interrupted_shoe(int_shoe, arm_tag)

        self._play_inspect_motion(
            attach_obj=int_shoe.entity,
            hand_id=self.AVATAR_HAND_ID,
            attach_frame=self._attach_frame_scaled,
            detach_frame=self._detach_frame_scaled,
            return_to_idle_after=self.INSPECT_RETURN_TO_IDLE_FRAMES,
        )

        self._retreat_safe(arm_tag)
        self._wait_until_avatar_attach()
        for _ in range(self.PRE_ATTACH_WAIT_MARGIN_STEPS):
            self.step_sim()

        any_success = False
        for other_idx, other_shoe in enumerate(self.shoes):
            if other_idx == interrupted_idx:
                continue
            other_label = chr(ord("A") + other_idx)
            try:
                ok = self._pick_and_place_shoe(
                    other_shoe, arm_tag, other_label, slot_idx=0,
                )
            except Exception:
                ok = False
            any_success = any_success or ok

        if not self.avatar.spare():
            while not self.avatar.spare():
                self.step_sim()
        for _ in range(self.POST_AVATAR_RELEASE_SETTLE_STEPS):
            self.step_sim()

        try:
            ok = self._pick_and_place_shoe(
                int_shoe, arm_tag, int_label,
                slot_idx=1 if len(self.shoes) > 1 else 0,
            )
        except Exception:
            ok = False
        any_success = any_success or ok
        return any_success

    # ------------------------------------------------------------------
    # Eval-mode interrupt state machine.
    # ------------------------------------------------------------------

    def _init_eval_interrupt_state(self):
        no_human_pace = bool(self.config.get("pace_no_human_ablation", False))
        self._eval_palm_default = (
            None if no_human_pace else self._dryrun_capture_palm()
        )
        if self._eval_palm_default is None:
            self._eval_interrupt_fired = not no_human_pace
            trigger_min = int(self.config.get(
                "eval_trigger_step_min", self.EVAL_TRIGGER_STEP_MIN,
            ))
            trigger_max = int(self.config.get(
                "eval_trigger_step_max", self.EVAL_TRIGGER_STEP_MAX,
            ))
            if trigger_max < trigger_min:
                trigger_max = trigger_min
            self._eval_trigger_step = int(np.random.randint(
                trigger_min, trigger_max + 1,
            ))
        else:
            trigger_min = int(self.config.get(
                "eval_trigger_step_min", self.EVAL_TRIGGER_STEP_MIN,
            ))
            trigger_max = int(self.config.get(
                "eval_trigger_step_max", self.EVAL_TRIGGER_STEP_MAX,
            ))
            if trigger_max < trigger_min:
                trigger_max = trigger_min
            self._eval_trigger_step = int(np.random.randint(
                trigger_min, trigger_max + 1,
            ))
            self._eval_interrupt_fired = False
        self._eval_interrupt_trigger_mode = str(
            self.config.get("eval_interrupt_trigger_mode", "random")
        ).lower()
        self._eval_glide_active = False
        self._eval_glide_remaining = 0
        self._eval_glide_start_pos = None
        self._eval_glide_target_pos = None
        self._eval_target_shoe = None
        self._eval_retreat_active = False
        self._eval_retreat_remaining = 0
        self._eval_retreat_target = None
        self._eval_retreat_last_cmd = None
        self._eval_override_last_cmd = None
        self._eval_return_active = False
        self._eval_return_remaining = 0
        self._eval_return_target = None
        self._eval_return_last_cmd = None
        self._eval_reset_policy_after_pace = False
        self._eval_return_finish_after_apply = False
        self._eval_waiting_avatar_done = False
        self._eval_saved_qpos = None
        self._eval_saved_gripper = 1.0
        self._eval_pending_interrupt_idx = None
        self._pace_target_idx = None
        self._pace_initial_dist_xy = None
        self._pace_best_progress = 0.0
        self._pace_trigger_reason = None
        self._pace_trigger_step = None
        self._pace_trigger_progress = None
        self._pace_trigger_dist_xy = None
        self._policy_step_count = 0

    def _eval_step_avatar(self):
        self._policy_step_count += 1

        if self._eval_return_active:
            self._eval_return_remaining -= 1
            if self._eval_return_remaining <= 0:
                self._eval_return_finish_after_apply = True
            return

        if self._eval_waiting_avatar_done:
            if self.avatar is not None and self.avatar.spare():
                self._eval_waiting_avatar_done = False
                self._start_eval_robot_return()
            return

        if self._eval_retreat_active:
            self._eval_retreat_remaining -= 1
            if self._eval_retreat_remaining <= 0:
                self._eval_retreat_active = False
                if bool(self.config.get("pace_no_human_ablation", False)):
                    self._start_eval_robot_return()
                else:
                    self._start_eval_avatar_glide()
            return

        if self._eval_glide_active:
            self._eval_glide_remaining -= 1
            done = self.EVAL_GLIDE_POLICY_STEPS - self._eval_glide_remaining
            progress = min(1.0, max(
                0.0, done / float(self.EVAL_GLIDE_POLICY_STEPS),
            ))
            cur = (
                (1.0 - progress) * self._eval_glide_start_pos
                + progress * self._eval_glide_target_pos
            )
            rot = np.asarray(self.avatar_init_rot, dtype=np.float64).copy()
            self.avatar.reset(cur.copy(), rot)
            if self._eval_glide_remaining <= 0:
                self._eval_glide_active = False
                target_shoe = self._eval_target_shoe
                shoe_pos = np.asarray(self._shoe_pos_quat(target_shoe)[0])
                if shoe_pos[2] < self.TABLE_TOP_Z - self.EVAL_TABLE_Z_LOWER_MARGIN:
                    return
                self.avatar.play_animation(
                    self.INSPECT_MOTION_NAME,
                    attach_obj=target_shoe.entity,
                    hand_id=self.AVATAR_HAND_ID,
                    attach_frame=self._attach_frame_scaled,
                    detach_frame=self._detach_frame_scaled,
                    return_to_idle_after=self.INSPECT_RETURN_TO_IDLE_FRAMES,
                )
                self._eval_waiting_avatar_done = True
            return

        if self._eval_interrupt_fired:
            return

        target_idx, dist = self._closest_table_shoe_to_ee()
        should_trigger = False
        trigger_reason = None
        mode = getattr(self, "_eval_interrupt_trigger_mode", "random")
        if mode == "pace":
            should_trigger, trigger_reason = self._pace_should_trigger(
                target_idx, dist,
            )
        else:
            force_trigger = (
                target_idx is not None
                and dist <= float(self.config.get(
                    "eval_force_trigger_dist_xy",
                    self.EVAL_FORCE_TRIGGER_DIST_XY,
                ))
            )
            should_trigger = (
                self._policy_step_count >= self._eval_trigger_step
            ) or force_trigger
            trigger_reason = "force_dist" if force_trigger else "random_step"

        if not should_trigger:
            return
        if target_idx is None:
            self._eval_interrupt_fired = True
            return

        self._eval_interrupt_fired = True
        self._interrupted_idx = int(target_idx)
        self._eval_pending_interrupt_idx = int(target_idx)
        self._eval_target_shoe = self.shoes[target_idx]
        self._pace_trigger_reason = trigger_reason
        self._pace_trigger_step = int(self._policy_step_count)
        self._pace_trigger_progress = float(self._pace_best_progress)
        self._pace_trigger_dist_xy = (
            None if not np.isfinite(dist) else float(dist)
        )
        arm = self.robot.get_arm("right")
        self._eval_saved_qpos = np.asarray(
            arm.get_arm_qpos(), dtype=np.float64,
        ).copy()
        self._eval_saved_gripper = float(getattr(arm, "gripper_val", 1.0))
        self._start_eval_robot_retreat()

    def _start_eval_avatar_glide(self):
        if self.avatar is None:
            self._start_eval_robot_return()
            return
        target_idx = getattr(self, "_eval_pending_interrupt_idx", None)
        if target_idx is None:
            return
        target_shoe = self.shoes[int(target_idx)]
        target_pos = np.asarray(self._shoe_pos_quat(target_shoe)[0], dtype=float)
        if target_pos[2] < self.TABLE_TOP_Z - self.EVAL_TABLE_Z_LOWER_MARGIN:
            return
        shift = target_pos - self._eval_palm_default
        shift[2] += self.AVATAR_EXTRA_Z
        target_avatar_pos = (
            np.asarray(self.avatar_init_pos, dtype=np.float64) + shift
        )

        self._eval_glide_start_pos = (
            np.asarray(self.avatar_init_pos, dtype=np.float64).copy()
        )
        self._eval_glide_target_pos = target_avatar_pos.copy()
        self._eval_glide_remaining = self.EVAL_GLIDE_POLICY_STEPS
        self._eval_glide_active = True
        self._eval_target_shoe = target_shoe

    def _pace_should_trigger(self, idx, dist):
        if idx is None or not np.isfinite(dist):
            return False, None
        min_steps = int(self.config.get(
            "pace_min_policy_steps", self.PACE_MIN_POLICY_STEPS,
        ))
        if self._policy_step_count < min_steps:
            return False, None
        start_dist = float(self.config.get(
            "pace_progress_start_dist_xy", self.PACE_PROGRESS_START_DIST_XY,
        ))
        if self._pace_target_idx != idx:
            self._pace_target_idx = int(idx)
            self._pace_initial_dist_xy = max(float(dist), start_dist)
            self._pace_best_progress = 0.0
        denom = max(1e-6, float(self._pace_initial_dist_xy))
        progress = 1.0 - float(dist) / denom
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
        if dist <= trigger_dist:
            return True, "pace_dist"
        return False, None

    def _start_eval_robot_retreat(self):
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

    def _apply_eval_pace_override(self):
        if getattr(self, "_eval_return_active", False):
            return self._apply_eval_robot_return()
        if (
            not getattr(self, "_eval_retreat_active", False)
            and not getattr(self, "_eval_waiting_avatar_done", False)
        ):
            return False
        return self._apply_eval_robot_retreat()

    def _apply_eval_robot_retreat(self):
        arm = self.robot.get_arm("right")
        if self._eval_retreat_target is None:
            self._eval_retreat_target = np.asarray(arm.homestate, dtype=np.float64)
        current = np.asarray(arm.get_arm_qpos(), dtype=np.float64)
        target = np.asarray(self._eval_retreat_target, dtype=np.float64)
        alpha = float(self.config.get(
            "eval_robot_retreat_blend",
            self.EVAL_ROBOT_RETREAT_BLEND,
        ))
        alpha = float(np.clip(alpha, 0.0, 1.0))
        cmd = current * (1.0 - alpha) + target * alpha
        self._eval_retreat_last_cmd = cmd.copy()
        self._eval_override_last_cmd = cmd.copy()
        self.robot.set_arm_joints(cmd, "right")
        self.robot.set_gripper(1.0, "right")
        return True

    def _start_eval_robot_return(self):
        target = getattr(self, "_eval_saved_qpos", None)
        if target is None:
            return
        steps = int(self.config.get(
            "eval_robot_return_policy_steps",
            self.EVAL_ROBOT_RETURN_POLICY_STEPS,
        ))
        if steps <= 0:
            self._eval_pending_interrupt_idx = None
            self._eval_return_active = False
            return
        self._eval_return_target = np.asarray(target, dtype=np.float64).copy()
        self._eval_return_remaining = steps
        self._eval_return_active = True

    def _apply_eval_robot_return(self):
        if not getattr(self, "_eval_return_active", False):
            return False
        if self._eval_return_target is None:
            return False
        arm = self.robot.get_arm("right")
        current = np.asarray(arm.get_arm_qpos(), dtype=np.float64)
        target = np.asarray(self._eval_return_target, dtype=np.float64)
        alpha = float(self.config.get(
            "eval_robot_return_blend",
            self.EVAL_ROBOT_RETURN_BLEND,
        ))
        alpha = float(np.clip(alpha, 0.0, 1.0))
        cmd = current * (1.0 - alpha) + target * alpha
        self._eval_return_last_cmd = cmd.copy()
        self._eval_override_last_cmd = cmd.copy()
        self.robot.set_arm_joints(cmd, "right")
        self.robot.set_gripper(
            float(getattr(self, "_eval_saved_gripper", 1.0)), "right",
        )
        if getattr(self, "_eval_return_finish_after_apply", False):
            self._eval_return_finish_after_apply = False
            self._eval_return_active = False
            self._eval_return_target = None
            self._eval_pending_interrupt_idx = None
            self._eval_reset_policy_after_pace = True
        return True

    def eval_skip_policy_action(self) -> bool:
        return bool(self.config.get("eval_mode", False)) and (
            getattr(self, "_eval_retreat_active", False)
            or getattr(self, "_eval_waiting_avatar_done", False)
            or getattr(self, "_eval_return_active", False)
        )

    def consume_eval_policy_reset_requested(self) -> bool:
        flag = bool(getattr(self, "_eval_reset_policy_after_pace", False))
        self._eval_reset_policy_after_pace = False
        return flag

    def _find_closest_table_shoe(self):
        idx, _ = self._closest_table_shoe_to_ee()
        return idx

    def _closest_table_shoe_to_ee(self):
        arm = self.robot.get_arm("right")
        ee_pos = np.array(arm.get_ee_pose()[:3], dtype=float)
        box_xy = to_numpy(self.shoebox.entity.get_pos()).ravel()[:2]

        best_idx = None
        best_d = float("inf")
        for i, shoe in enumerate(self.shoes):
            p = np.asarray(self._shoe_pos_quat(shoe)[0])
            in_box = (
                np.linalg.norm(p[:2] - box_xy) < self.SUCCESS_DIST_XY_M
                and p[2] > self.TABLE_TOP_Z + self.SUCCESS_Z_FLOOR_MARGIN_M
            )
            if in_box:
                continue
            if (
                p[2] < self.TABLE_TOP_Z - self.EVAL_TABLE_Z_LOWER_MARGIN
                or p[2] > self.TABLE_TOP_Z + self.EVAL_TABLE_Z_UPPER_MARGIN
            ):
                continue
            d = float(np.linalg.norm(p[:2] - ee_pos[:2]))
            if d < best_d:
                best_d = d
                best_idx = i
        return best_idx, best_d

    # ------------------------------------------------------------------
    # Evaluation metrics.
    # ------------------------------------------------------------------

    def _shoe_in_box_metrics(self) -> dict:
        metrics = {}
        if not self.shoes or self.shoebox is None:
            return metrics

        box_pos = to_numpy(self.shoebox.entity.get_pos()).ravel()[:3]
        box_xy = box_pos[:2]
        box_floor_z = self.TABLE_TOP_Z + self.SUCCESS_Z_FLOOR_MARGIN_M
        metrics["shoebox_pos"] = box_pos.tolist()
        metrics["shoebox_floor_z_threshold"] = float(box_floor_z)
        metrics["success_dist_xy_threshold"] = float(self.SUCCESS_DIST_XY_M)
        all_in = True
        for i, shoe in enumerate(self.shoes):
            label = chr(ord("A") + i)
            p = np.asarray(self._shoe_pos_quat(shoe)[0])
            dxy = float(np.linalg.norm(p[:2] - box_xy))
            above_floor = bool(p[2] > box_floor_z)
            in_box = bool(dxy <= self.SUCCESS_DIST_XY_M and above_floor)
            all_in = all_in and in_box
            prefix = f"shoe_{label.lower()}"
            metrics[f"{prefix}_pos"] = p.tolist()
            metrics[f"{prefix}_dist_xy"] = dxy
            metrics[f"{prefix}_above_floor"] = above_floor
            metrics[f"{prefix}_in_box"] = in_box
        metrics["all_shoes_in_box"] = bool(all_in)
        return metrics

    def check_success(self) -> bool:
        if self.avatar_collided:
            return False
        if bool(self.config.get("eval_mode", False)) and (
            getattr(self, "_eval_retreat_active", False)
            or getattr(self, "_eval_glide_active", False)
            or getattr(self, "_eval_waiting_avatar_done", False)
            or getattr(self, "_eval_return_active", False)
            or getattr(self, "_eval_pending_interrupt_idx", None) is not None
        ):
            return False
        return super().check_success()

    def evaluate(self) -> dict:
        metrics = super().evaluate()
        metrics.update(self._shoe_in_box_metrics())
        if bool(self.config.get("eval_mode", False)):
            metrics.update({
                "eval_mode": True,
                "eval_trigger_step": int(getattr(
                    self, "_eval_trigger_step", -1,
                )),
                "eval_policy_step_count": int(getattr(
                    self, "_policy_step_count", 0,
                )),
                "eval_interrupt_fired": bool(getattr(
                    self, "_eval_interrupt_fired", False,
                )),
                "eval_glide_active": bool(getattr(
                    self, "_eval_glide_active", False,
                )),
                "pace_interrupt": {
                    "enabled": getattr(
                        self, "_eval_interrupt_trigger_mode", "random",
                    ) == "pace",
                    "trigger_mode": getattr(
                        self, "_eval_interrupt_trigger_mode", "random",
                    ),
                    "fired": bool(getattr(
                        self, "_eval_interrupt_fired", False,
                    )),
                    "trigger_step": getattr(self, "_pace_trigger_step", None),
                    "trigger_reason": getattr(
                        self, "_pace_trigger_reason", None,
                    ),
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
                    "retreat_active": bool(getattr(
                        self, "_eval_retreat_active", False,
                    )),
                    "waiting_avatar_done": bool(getattr(
                        self, "_eval_waiting_avatar_done", False,
                    )),
                    "return_active": bool(getattr(
                        self, "_eval_return_active", False,
                    )),
                },
            })
        return metrics
