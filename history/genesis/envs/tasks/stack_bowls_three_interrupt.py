"""Stack three bowls (interrupt): same scene as ``stack_bowls_three``
but a human avatar picks up + inspects one of the two stacked bowls
mid-task; the robot must wait, then resume on the bowl's settled-after-
detach pose.

Flow per episode:
  1. The task samples whether the avatar interrupts the 2nd or 3rd
     stack layer.  For layer 2, the interruption happens before any
     non-base bowl is stacked; for layer 3, the robot first stacks the
     other non-base bowl, then the interruption happens.
  2. Robot shows intent above the interrupted bowl (top-down EE pose,
     no commit) so the video reads as "robot was reaching for it".
  3. Avatar starts the ``Bowl_clip`` motion with ``attach_obj=`` the
     interrupted bowl — bowl attaches to a two-hand palm frame at frame
     ATTACH and detaches at DETACH.  Avatar lifts, holds, sets back.
  4. Robot retreats toward the robot base (out of the avatar's lean
     cone) and waits.
  5. ``avatar.spare()`` → bowl is back on the table at a slightly
     different pose; settle ~120 steps.
  6. Robot reads the bowl's NEW pose and runs the same
     ``_pick_and_stack_one`` primitive on it.

Geometry calibration (same recipe as ``dump_bin_interrupt`` /
``place_bread_in_basket_interrupt``):
  * Pre-bake: dry-run the Bowl_clip motion at ``avatar_init_pos`` and
    measure the avatar's two-hand bowl frame at the (scaled) attach
    frame. Cached on the class — depends only on motion + frame_ratio
    + home pose.
  * Constraint solve: avatar shift = bowl_pos − bowl_frame_default.
    After shifting, the two-hand frame meets the interrupted bowl at
    attach.
  * Both non-base bowls spawn in the same avatar-reachable subregion so
    placement does not reveal which bowl the human will inspect.

Eval inherits the parent's loose "stacked bowl center inside bowl 0's
horizontal footprint and higher z" gate, plus a hard fail on
``self.avatar_collided`` (avatar collision tracking is forced on in __init__).

Robot motion is pure physics (PD only — no kinematic teleport on the
arm or the held bowl).  The avatar attaches the bowl via
``play_animation(attach_obj=...)``, which is the sanctioned avatar
mechanism.
"""

import contextlib

import numpy as np
import transforms3d as t3d

from .. import base_task as _base_task
from ..grasp import GraspPose, tcp_to_link_pose
from ..utils import Pose, load_object, to_numpy
from ..task_bases.stack_bowls_three import (
    StackBowlsThree,
    _BOWL_ID,
    _BOWL_MODEL_ID,
    _NUM_BOWLS,
    _MIN_SEP_M,
    _quat_mul,
    _read_extents,
    _read_scale,
)


# Shared avatar-reachable subregion for non-base bowls.  Strict subset of
# the parent's full bowl-spawn region chosen so the Bowl_clip shift keeps the
# avatar beside the table while not revealing which non-base bowl is targeted.
_REACH_X_LO, _REACH_X_HI = -0.12, 0.18
_REACH_Y_LO, _REACH_Y_HI = -0.16, 0.04
_INT_X_LO, _INT_X_HI = _REACH_X_LO, _REACH_X_HI
_INT_Y_LO, _INT_Y_HI = _REACH_Y_LO, _REACH_Y_HI
_BASE_X_LO, _BASE_X_HI = -0.08, 0.10
_BASE_Y_LO, _BASE_Y_HI = -0.10, 0.02
_LIFT_X_LO, _LIFT_X_HI = _REACH_X_LO, _REACH_X_HI
_LIFT_Y_LO, _LIFT_Y_HI = _REACH_Y_LO, _REACH_Y_HI
_INTERRUPT_SAMPLE_ATTEMPTS = 256
_INTERRUPT_FALLBACK_XY = [(-0.12, 0.08), (0.14, 0.06), (0.02, -0.12)]


def _sample_bowl_xy_with_interrupt(rng, n_bowls):
    """Same sampling pattern as the parent's `_sample_three_xy`, but
    with both non-base bowls constrained to the same avatar-reach subregion.
    Min-spacing rejection sampling; falls back to a deterministic spread on
    repeated rejection.
    """
    for _ in range(_INTERRUPT_SAMPLE_ATTEMPTS):
        pts = []
        ok = True
        for slot in range(n_bowls):
            if slot == 0:
                x = float(rng.uniform(_BASE_X_LO, _BASE_X_HI))
                y = float(rng.uniform(_BASE_Y_LO, _BASE_Y_HI))
            else:
                x = float(rng.uniform(_LIFT_X_LO, _LIFT_X_HI))
                y = float(rng.uniform(_LIFT_Y_LO, _LIFT_Y_HI))
            for px, py in pts:
                if (x - px) ** 2 + (y - py) ** 2 < _MIN_SEP_M ** 2:
                    ok = False
                    break
            if not ok:
                break
            pts.append((x, y))
        if ok and len(pts) == n_bowls:
            return pts
    fallback = list(_INTERRUPT_FALLBACK_XY)
    fallback[1] = (_REACH_X_LO + 0.05, _REACH_Y_HI - 0.03)
    fallback[2] = (_REACH_X_HI - 0.05, _REACH_Y_LO + 0.03)
    return fallback[:n_bowls]


class StackBowlsThreeInterrupt(StackBowlsThree):
    """Robot stacks three bowls; a human avatar picks up + inspects
    one of the two stacked bowls (bowl 1 or 2) mid-task — robot must
    wait, then resume on the bowl's settled-after-detach pose.

    The interruption layer is randomized between the 2nd and 3rd stack
    layer.  Bowl 0 stays the base; the avatar can handle either bowl 1
    or bowl 2 before the robot stacks it.
    """

    INSTRUCTION = "stack the three bowls on top of each other; if the human inspects one bowl midway through, wait and then continue"
    use_avatar = True

    # Avatar at +Y side of the table, facing -Y (toward the table).
    # Maps mesh-+X (avatar forward) to world-(-Y).  Same convention as
    # dump_bin_interrupt / place_bread_in_basket_interrupt.
    AVATAR_BASE_ROT = np.array(
        [[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=np.float64,
    )
    avatar_init_pos = np.array([0.0, 0.55, -0.18])
    avatar_init_rot = AVATAR_BASE_ROT

    BOWL_MOTION_NAME = "Bowl_clip"
    BOWL_ATTACH_FRAME = 45
    BOWL_DETACH_FRAME = 168
    AVATAR_MOTION_SLOW = 5.0
    BOWL_TWO_HAND_OFFSET = np.array([0.0, 0.0, -0.015], dtype=np.float64)

    # After the authored bowl frame was restored, object-local +y is
    # world-up.  A small lift keeps the fingers clear of the lower wall
    # while retaining substantially more rim engagement than the former
    # 15 mm correction.
    GRASP_Z_LIFT = 0.005
    INTERRUPT_SEED_OFFSET = 4104
    DRYRUN_RESET_SETTLE_STEPS = 20
    CALIBRATION_SETTLE_STEPS = 30
    RETREAT_Y_FROM_BASE = 0.25
    RETREAT_Z_ABOVE_TABLE = 0.30
    INTENT_Z_ABOVE_TABLE = 0.20
    # None ⇒ avatar freezes on the bowl motion's last frame (pose + position).
    BOWL_RETURN_TO_IDLE_FRAMES = None
    PRE_ATTACH_WAIT_MARGIN_STEPS = 100
    POST_RELEASE_SETTLE_STEPS = 120
    EVAL_TRIGGER_STEP_MIN = 20
    EVAL_TRIGGER_STEP_MAX = 100
    EVAL_INTERRUPT_DISTANCE_M = 0.18
    EVAL_FORCE_TRIGGER_DIST_XY = 0.12
    PACE_TRIGGER_PROGRESS = 0.82
    PACE_TRIGGER_DIST_XY = 0.14
    PACE_PROGRESS_START_DIST_XY = 0.42
    PACE_MIN_POLICY_STEPS = 20
    EVAL_ROBOT_RETREAT_POLICY_STEPS = 35
    EVAL_ROBOT_RETREAT_BLEND = 0.50
    EVAL_ROBOT_RETURN_POLICY_STEPS = 35
    EVAL_ROBOT_RETURN_BLEND = 0.50

    def __init__(self, config: dict = None):
        # Force avatar-collision tracking on so check_success can hard-
        # fail any episode where the robot bumps the avatar.  Caller may
        # still pass `track_avatar_collision` explicitly to override.
        cfg = dict(config or {})
        cfg.setdefault("track_avatar_collision", True)
        avatar_cfg = dict(cfg.get("avatar", {}))
        avatar_cfg.setdefault(
            "generated_motion_data",
            "avatars/motions/generated_motions.pkl",
        )
        cfg["avatar"] = avatar_cfg
        super().__init__(cfg)

    # ------------------------------------------------------------------
    # Scene setup — re-implements parent's `load_actors` so the non-base
    # bowls' xy positions are constrained to the shared avatar-reach subregion.
    # ------------------------------------------------------------------

    def reset(self, seed: int = 0):
        self._episode_seed = int(seed)
        obs = super().reset(seed=seed)
        no_human_pace = bool(self.config.get("pace_no_human_ablation", False))
        if self.avatar is None and not no_human_pace:
            return obs
        if self.avatar is not None:
            self.avatar.frame_ratio = self.AVATAR_MOTION_SLOW
            self._calibrate_avatar_for_bowl_motion()
        self._eval_policy_step_count = 0
        self._eval_interrupt_fired = False
        self._eval_interrupt_failed = False
        self._eval_trigger_step = None
        if bool(self.config.get("eval_mode", False)):
            lo = int(self.config.get("eval_trigger_step_min", self.EVAL_TRIGGER_STEP_MIN))
            hi = int(self.config.get("eval_trigger_step_max", self.EVAL_TRIGGER_STEP_MAX))
            if hi < lo:
                hi = lo
            self._eval_trigger_step = int(np.random.randint(lo, hi + 1))
        self._eval_interrupt_trigger_mode = str(
            self.config.get("eval_interrupt_trigger_mode", "random")
        ).lower()
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
        self._eval_visual_human_started = False
        self._eval_visual_pace_state = None
        self._eval_visual_pace_detector = None
        self._pace_target_idx = None
        self._pace_initial_dist_xy = None
        self._pace_best_progress = 0.0
        self._pace_trigger_reason = None
        self._pace_trigger_step = None
        self._pace_trigger_progress = None
        self._pace_trigger_dist_xy = None
        return obs

    def load_actors(self):
        # Pick the interrupted bowl ∈ {1, 2}.  Bowl 0 is the stack base —
        # moving the base ruins the stack target (per brief).  Separately
        # pick whether the interruption occurs on the 2nd or 3rd stack layer.
        # Tie this high-level task choice directly to the episode seed,
        # instead of the global RNG state after robot/camera setup, so seed
        # sweeps are reproducible and small sweeps cover both interruption
        # timings.
        seed = int(getattr(self, "_episode_seed", 0))
        rng_top = np.random.RandomState(seed + self.INTERRUPT_SEED_OFFSET)
        if self.simplified_mode_enabled():
            self._interrupted_idx = 1
            self._interrupt_stack_layer = 2
        else:
            self._interrupted_idx = int(rng_top.choice([1, 2]))
            self._interrupt_stack_layer = int(rng_top.choice([2, 3]))

        upright_q = self.BOWL_UPRIGHT_QUAT
        table_top = self.TABLE_TOP_Z

        rng = np.random.RandomState(np.random.randint(0, 2 ** 31 - 1))
        positions = _sample_bowl_xy_with_interrupt(rng, self._num_bowls)

        self._bowl_scale = _read_scale(_BOWL_ID, _BOWL_MODEL_ID)
        bowl_extents = _read_extents(_BOWL_ID, _BOWL_MODEL_ID)
        self._bowl_extents = bowl_extents
        self._bowl_height = float(bowl_extents[1]) * self._bowl_scale

        yaws = [
            float(rng.uniform(-self.BOWL_YAW_LIMIT, self.BOWL_YAW_LIMIT))
            for _ in range(self._num_bowls)
        ]

        spawn_z = table_top + self.BOWL_SPAWN_Z_CLEARANCE
        self.bowls = []
        self._spawn_xy = []
        for i, ((cx, cy), yaw) in enumerate(zip(positions, yaws)):
            yaw_q = np.asarray(
                t3d.quaternions.mat2quat(
                    t3d.euler.euler2mat(0, 0, yaw, "sxyz")
                ),
                dtype=np.float64,
            )
            full_q = _quat_mul(yaw_q, upright_q)
            actor = load_object(
                self.scene,
                Pose([cx, cy, spawn_z], full_q),
                _BOWL_ID, model_id=_BOWL_MODEL_ID,
                convex=True, is_static=False,
                friction=self.BOWL_FRICTION,
                density=self.BOWL_DENSITY,
            )
            self.bowls.append(actor)
            self._spawn_xy.append((cx, cy))

        self._stacked_indices = list(range(1, self._num_bowls))

    # ------------------------------------------------------------------
    # Reset + avatar calibration
    # ------------------------------------------------------------------

    @property
    def _attach_frame_scaled(self) -> int:
        return int(round(self.BOWL_ATTACH_FRAME * self.AVATAR_MOTION_SLOW))

    @property
    def _detach_frame_scaled(self) -> int:
        return int(round(self.BOWL_DETACH_FRAME * self.AVATAR_MOTION_SLOW))

    def _bowl_frame_center(self):
        center, _rot = self.avatar.robot._get_two_hand_frame(
            self.BOWL_TWO_HAND_OFFSET
        )
        return np.asarray(center, dtype=np.float64).copy()

    def _dryrun_capture_bowl_frame(self):
        """Play Bowl_clip at the avatar home pose, capture the two-hand
        bowl frame at the (scaled) attach frame, then restore home.

        Cached on the class — the two-hand frame trajectory is determined by
        motion + frame_ratio + avatar home pose, not bowl positions.
        After seed 0 every subsequent reset re-uses the cached value,
        saving the dry-run sim cost.
        """
        cache_key = (
            self.BOWL_MOTION_NAME,
            float(self.AVATAR_MOTION_SLOW),
            tuple(np.asarray(self.BOWL_TWO_HAND_OFFSET, dtype=np.float64).ravel()),
            tuple(np.asarray(self.avatar_init_pos, dtype=np.float64).ravel()),
        )
        cached = getattr(type(self), "_bowl_frame_default_cache", {}).get(cache_key)
        if cached is not None:
            return cached.copy()

        self.avatar.play_bowl_motion(snap_to_hands=False)
        bowl_frame_default = None
        step = 0
        attach_step = self._attach_frame_scaled
        while step <= attach_step and not self.avatar.spare():
            self.step_sim()
            if step == attach_step:
                bowl_frame_default = self._bowl_frame_center()
                break
            step += 1
        # Restore avatar to home (clears any residual pose from the
        # dry-run Bowl_clip).
        self.avatar.reset(
            np.asarray(self.avatar_init_pos, dtype=np.float64).copy(),
            np.asarray(self.avatar_init_rot, dtype=np.float64).copy(),
        )
        for _ in range(self.DRYRUN_RESET_SETTLE_STEPS):
            self.step_sim()
        # Clear avatar-collision flags accumulated during the dry run.
        self.avatar_collided = False
        self.avatar_collision_log = []
        if bowl_frame_default is not None:
            cache = getattr(type(self), "_bowl_frame_default_cache", None)
            if cache is None:
                cache = {}
                type(self)._bowl_frame_default_cache = cache
            cache[cache_key] = bowl_frame_default.copy()
        return bowl_frame_default

    def _calibrate_avatar_for_bowl_motion(self):
        """Shift avatar so the two-hand bowl frame meets the interrupted
        bowl at Bowl_clip's attach frame."""
        bowl = self.bowls[self._interrupted_idx]
        target_frame = to_numpy(bowl.entity.get_pos()).ravel()[:3].astype(float)

        bowl_frame_default = self._dryrun_capture_bowl_frame()
        if bowl_frame_default is None:
            return

        shift = target_frame - bowl_frame_default
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
    # Grasp up-lift override — wraps `_pick_and_stack_one` so the rim
    # grasp TCP sits `GRASP_Z_LIFT` higher than the YAML pose.
    # ------------------------------------------------------------------

    @contextlib.contextmanager
    def _with_grasp_z_lift(self):
        """Monkey-patch ``envs.base_task.load_grasp_poses`` to return
        copies whose up component is shifted by ``GRASP_Z_LIFT`` (in
        world metres).  The authored bowl mesh uses object-local +y as
        world-up.  ``_scaled_pose`` later multiplies by ``object_scale``
        for ``mesh_unit`` poses, so we pre-divide to land the world shift
        at exactly ``GRASP_Z_LIFT``.
        """
        delta = float(self.GRASP_Z_LIFT)
        if delta == 0.0:
            yield
            return

        original = _base_task.load_grasp_poses
        scale = float(self._bowl_scale)

        def _patched(*args, **kwargs):
            grasps = original(*args, **kwargs)
            lifted = []
            for g in grasps:
                local_up = (
                    delta / scale if g.scale_frame == "mesh_unit" else delta
                )
                new_pos = np.asarray(g.pose.p, dtype=np.float64).copy()
                new_pos[1] += local_up
                ng = GraspPose(
                    name=g.name,
                    position=new_pos,
                    quaternion=np.asarray(g.pose.q, dtype=np.float64).copy(),
                    pre_distance=g.pre_distance,
                    source=g.source,
                    scale_frame=g.scale_frame,
                    category=g.category,
                    kinematic_pin=g.kinematic_pin,
                )
                lifted.append(ng)
            return lifted

        _base_task.load_grasp_poses = _patched
        try:
            yield
        finally:
            _base_task.load_grasp_poses = original

    def _pick_and_stack_one(self, src_idx, stack_position):
        """Execute one physical grasp using the interrupt-safe lifted pose."""
        with self._with_grasp_z_lift():
            return super()._pick_and_stack_one(src_idx, stack_position)

    # ------------------------------------------------------------------
    # Robot retreat — pull the EE toward the robot base + lift, so the
    # gripper is well clear of the avatar's forward-lean cone before the
    # avatar's attach frame fires.
    # ------------------------------------------------------------------

    def _retreat_safe(self, arm_tag: str):
        arm = self.robot.get_arm(arm_tag)
        tcp_offset = arm.tcp_offset
        arm_base = np.array(arm.origin_pose.p, dtype=float)
        retreat_p = np.array([
            arm_base[0],
            arm_base[1] + self.RETREAT_Y_FROM_BASE,
            self.TABLE_TOP_Z + self.RETREAT_Z_ABOVE_TABLE,
        ])
        # Top-down TCP rotation (same convention as parent
        # _pick_and_stack_one's place pose).
        R_tcp_down = np.array(
            [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]],
            dtype=np.float64,
        )
        q_tcp_down = t3d.quaternions.mat2quat(R_tcp_down)
        retreat_link = tcp_to_link_pose(
            Pose(retreat_p, q_tcp_down), tcp_offset,
        )
        self.move_and_execute(retreat_link.to_pose7(), arm_tag)

    def _wait_until_avatar_attach(self):
        motion = self.avatar.motion_modules.get(self.BOWL_MOTION_NAME)
        if motion is None:
            return
        while (
            not self.avatar.spare()
            and int(getattr(motion, "at_frame", 0)) < self._attach_frame_scaled
        ):
            self.step_sim()

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
                return StackBowlsThree.take_action(self, action, action_type=action_type)
        return super().take_action(action, action_type=action_type)

    def _eval_robot_yield_enabled(self) -> bool:
        return bool(self.config.get("eval_robot_yield_enabled", True))

    def _closest_interrupt_bowl_to_ee(self):
        if not getattr(self, "bowls", None):
            return None, float("inf")
        ee = np.asarray(self.robot.get_arm("right").get_ee_pose()[:3], dtype=float)
        best_i = None
        best_d = float("inf")
        for i in (1, 2):
            if i >= len(self.bowls):
                continue
            p = self._bowl_xyz(i)
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
        mode = getattr(self, "_eval_interrupt_trigger_mode", "random")
        if mode == "visual_pace" and not getattr(self, "_eval_visual_human_started", False):
            start_step = int(self.config.get("pace_visual_human_start_step", 20))
            if self._eval_policy_step_count >= start_step:
                idx, _ = self._closest_interrupt_bowl_to_ee()
                if idx is None:
                    idx = int(getattr(self, "_interrupted_idx", 1))
                self._eval_pending_interrupt_idx = int(idx)
                self._start_eval_visual_human_motion()
        if getattr(self, "_eval_return_active", False):
            self._eval_return_remaining -= 1
            if self._eval_return_remaining <= 0:
                self._eval_return_finish_after_apply = True
            return
        if getattr(self, "_eval_waiting_avatar_done", False):
            if self.avatar is not None and self.avatar.spare():
                self._eval_waiting_avatar_done = False
                self._start_eval_robot_return()
            return
        if getattr(self, "_eval_retreat_active", False):
            self._eval_retreat_remaining -= 1
            if self._eval_retreat_remaining <= 0:
                self._eval_retreat_active = False
                if bool(self.config.get("pace_no_human_ablation", False)):
                    self._start_eval_robot_return()
                elif mode == "visual_pace":
                    if self.avatar is not None and self.avatar.spare():
                        self._start_eval_robot_return()
                    else:
                        self._eval_waiting_avatar_done = True
                else:
                    self._start_eval_bowl_interrupt()
            return
        if getattr(self, "_eval_interrupt_fired", False):
            return
        idx, dist = self._closest_interrupt_bowl_to_ee()
        should_trigger = False
        trigger_reason = None
        if mode == "pace":
            should_trigger, trigger_reason = self._pace_should_trigger(idx, dist)
        elif mode == "visual_pace":
            should_trigger, trigger_reason = self._visual_pace_should_trigger()
        else:
            trigger = getattr(self, "_eval_trigger_step", None)
            force_trigger = (
                idx is not None
                and dist <= float(self.config.get(
                    "eval_force_trigger_dist_xy",
                    self.EVAL_FORCE_TRIGGER_DIST_XY,
                ))
            )
            should_trigger = (
                trigger is not None
                and self._eval_policy_step_count >= int(trigger)
            ) or force_trigger
            trigger_reason = "force_dist" if force_trigger else "random_step"

        if not should_trigger:
            return
        threshold = float(self.config.get(
            "eval_interrupt_distance_m", self.EVAL_INTERRUPT_DISTANCE_M,
        ))
        if mode == "visual_pace" and getattr(self, "_eval_pending_interrupt_idx", None) is not None:
            idx = int(self._eval_pending_interrupt_idx)
            dist = float("nan")
        if idx is None:
            self._eval_interrupt_fired = True
            self._eval_interrupt_failed = True
            return
        if dist > threshold and not bool(self.config.get("eval_interrupt_force_at_trigger", True)):
            return
        self._eval_interrupt_fired = True
        self._pace_trigger_reason = trigger_reason
        self._pace_trigger_step = int(self._eval_policy_step_count)
        self._pace_trigger_progress = float(getattr(
            self, "_pace_best_progress", 0.0,
        ))
        self._pace_trigger_dist_xy = (
            None if not np.isfinite(dist) else float(dist)
        )
        self._interrupted_idx = int(idx)
        self._eval_pending_interrupt_idx = int(idx)
        arm = self.robot.get_arm("right")
        self._eval_saved_qpos = np.asarray(arm.get_arm_qpos(), dtype=np.float64).copy()
        self._eval_saved_gripper = float(getattr(arm, "gripper_val", 1.0))
        if self._eval_robot_yield_enabled():
            self._start_eval_robot_retreat()

    def set_eval_visual_pace_detector(self, detector):
        self._eval_visual_pace_detector = detector

    def set_eval_visual_pace_state(self, state):
        self._eval_visual_pace_state = state
        if state is not None:
            self._pace_best_progress = max(
                float(getattr(self, "_pace_best_progress", 0.0)),
                float(getattr(state, "best_progress", 0.0)),
            )

    def _visual_pace_should_trigger(self):
        min_steps = int(self.config.get(
            "pace_min_policy_steps", self.PACE_MIN_POLICY_STEPS,
        ))
        if self._eval_policy_step_count < min_steps:
            return False, None
        detector = getattr(self, "_eval_visual_pace_detector", None)
        if detector is None:
            return False, None
        should_trigger, reason = detector.should_trigger()
        return bool(should_trigger), reason

    def notify_eval_visual_pace_trigger(self, *, step: int, reason: str | None, state=None):
        """Record a generic visual-yield trigger in task-level PACE metrics.

        The generic wrapper owns the robot retreat/return.  Marking the task
        interrupt as fired prevents the task-local yield controller from
        consuming the same detector event on the following take_action call,
        while preserving the avatar motion and metrics.
        """
        if str(getattr(self, "_eval_interrupt_trigger_mode", "")).lower() != "visual_pace":
            return
        self._eval_interrupt_fired = True
        self._pace_trigger_reason = reason
        self._pace_trigger_step = int(step) + 1
        best_progress = getattr(state, "best_progress", None)
        if best_progress is None:
            best_progress = getattr(self, "_pace_best_progress", 0.0)
        self._pace_trigger_progress = float(best_progress)
        self._pace_trigger_dist_xy = None
        if getattr(self, "_eval_pending_interrupt_idx", None) is None:
            idx, _ = self._closest_interrupt_bowl_to_ee()
            if idx is not None:
                self._eval_pending_interrupt_idx = int(idx)
                self._interrupted_idx = int(idx)

    def _pace_should_trigger(self, idx, dist):
        if idx is None or not np.isfinite(dist):
            return False, None
        min_steps = int(self.config.get(
            "pace_min_policy_steps", self.PACE_MIN_POLICY_STEPS,
        ))
        if self._eval_policy_step_count < min_steps:
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
            self._start_eval_bowl_interrupt()
            return
        arm = self.robot.get_arm("right")
        self._eval_retreat_target = np.asarray(arm.homestate, dtype=np.float64)
        self._eval_retreat_remaining = steps
        self._eval_retreat_active = True

    def _apply_eval_pace_override(self):
        if not self._eval_robot_yield_enabled():
            return False
        if getattr(self, "_eval_return_active", False):
            return self._apply_eval_robot_return()
        if (
            not getattr(self, "_eval_retreat_active", False)
            and not getattr(self, "_eval_waiting_avatar_done", False)
        ):
            return False
        return self._apply_eval_robot_retreat()

    def _apply_eval_robot_retreat(self):
        if self._eval_retreat_target is None:
            arm = self.robot.get_arm("right")
            self._eval_retreat_target = np.asarray(arm.homestate, dtype=np.float64)
        if self._eval_retreat_target is None:
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
        self.robot.set_gripper(float(getattr(self, "_eval_saved_gripper", 1.0)), "right")
        if getattr(self, "_eval_return_finish_after_apply", False):
            self._eval_return_finish_after_apply = False
            self._eval_return_active = False
            self._eval_return_target = None
            self._eval_pending_interrupt_idx = None
            self._eval_reset_policy_after_pace = True
        return True

    def eval_skip_policy_action(self) -> bool:
        return (
            bool(self.config.get("eval_mode", False))
            and self._eval_robot_yield_enabled()
        ) and (
            getattr(self, "_eval_retreat_active", False)
            or getattr(self, "_eval_waiting_avatar_done", False)
            or getattr(self, "_eval_return_active", False)
        )

    def consume_eval_policy_reset_requested(self) -> bool:
        flag = bool(getattr(self, "_eval_reset_policy_after_pace", False))
        self._eval_reset_policy_after_pace = False
        return flag

    def _start_eval_bowl_interrupt(self):
        if self.avatar is None:
            self._start_eval_robot_return()
            return
        idx = getattr(self, "_eval_pending_interrupt_idx", None)
        if idx is None:
            return
        try:
            self._interrupted_idx = int(idx)
            self._calibrate_avatar_for_bowl_motion()
            self.avatar.play_bowl_motion(
                attach_obj=self.bowls[self._interrupted_idx].entity,
                return_to_idle_after=self.BOWL_RETURN_TO_IDLE_FRAMES,
                snap_to_hands=False,
            )
            self._eval_waiting_avatar_done = True
        except Exception as exc:
            self._eval_interrupt_failed = True
            if self.config.get("debug_eval_avatar", False):
                print(f"[stack_bowls_interrupt/eval] interrupt failed: {exc}", flush=True)
        finally:
            self._eval_pending_interrupt_idx = None

    def _start_eval_visual_human_motion(self):
        self._eval_visual_human_started = True
        if self.avatar is None:
            return
        idx = getattr(self, "_eval_pending_interrupt_idx", None)
        if idx is None:
            idx = int(getattr(self, "_interrupted_idx", 1))
            self._eval_pending_interrupt_idx = idx
        try:
            self._interrupted_idx = int(idx)
            self._calibrate_avatar_for_bowl_motion()
            self.avatar.play_bowl_motion(
                attach_obj=self.bowls[self._interrupted_idx].entity,
                return_to_idle_after=self.BOWL_RETURN_TO_IDLE_FRAMES,
                snap_to_hands=False,
            )
        except Exception as exc:
            self._eval_interrupt_failed = True
            if self.config.get("debug_eval_avatar", False):
                print(f"[stack_bowls_interrupt/eval] visual motion failed: {exc}", flush=True)

    def _run_avatar_interrupt(self, interrupted_idx: int, arm_tag: str) -> bool:
        """Show robot intent, let the avatar operate the bowl, then wait
        until the bowl has been returned to the table."""
        arm = self.robot.get_arm(arm_tag)
        tcp_offset = arm.tcp_offset

        # ---- Robot shows intent above the interrupted bowl -----------
        self.open_gripper(arm_tag)
        int_pos = self._bowl_xyz(interrupted_idx)
        intent_z = self.TABLE_TOP_Z + self.INTENT_Z_ABOVE_TABLE
        above_p = np.array([int_pos[0], int_pos[1], intent_z])
        R_tcp_down = np.array(
            [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]],
            dtype=np.float64,
        )
        q_tcp_down = t3d.quaternions.mat2quat(R_tcp_down)
        above_link = tcp_to_link_pose(
            Pose(above_p, q_tcp_down), tcp_offset,
        )
        self.move_and_execute(above_link.to_pose7(), arm_tag)

        # ---- Avatar plays Bowl_clip with two-hand attach -------------
        int_bowl = self.bowls[interrupted_idx]
        self.avatar.play_bowl_motion(
            attach_obj=int_bowl.entity,
            return_to_idle_after=self.BOWL_RETURN_TO_IDLE_FRAMES,
            snap_to_hands=False,
        )

        # ---- Robot retreats so the avatar can grab the bowl ----------
        self._retreat_safe(arm_tag)
        # Step until the avatar reaches the attach frame so the bowl is
        # visibly held before the robot moves on.
        self._wait_until_avatar_attach()
        for _ in range(self.PRE_ATTACH_WAIT_MARGIN_STEPS):
            self.step_sim()

        # ---- Wait for avatar to release the bowl ---------------------
        if not self.avatar.spare():
            while not self.avatar.spare():
                self.step_sim()
        # Settle the released bowl on the table.
        for _ in range(self.POST_RELEASE_SETTLE_STEPS):
            self.step_sim()
        return True

    def _stack_target_for_placed(self, placed_indices):
        base_xyz = self._bowl_xyz(0)
        top_z = max(self._bowl_xyz(i)[2] for i in placed_indices)
        return (base_xyz[0], base_xyz[1], top_z + self._bowl_height)

    # ------------------------------------------------------------------
    # Rollout
    # ------------------------------------------------------------------

    def play_once(self) -> bool:
        # Settle the bowls before any robot motion.
        with self.suppress_recording():  # pre-task settle, excluded from recordings
            for _ in range(self.INITIAL_SETTLE_STEPS):
                self.step_sim()
        if any(b is None for b in self.bowls):
            return False

        if self.avatar is None:
            return super().play_once()

        arm_tag = "right"

        interrupted_idx = self._interrupted_idx
        # _stacked_indices = [1, 2]; the OTHER stacked bowl is whichever
        # of {1, 2} isn't the interrupted one.
        other_idx = 3 - interrupted_idx

        if self._interrupt_stack_layer == 2:
            # Avatar interrupts the first bowl the robot would stack.
            if not self._run_avatar_interrupt(interrupted_idx, arm_tag):
                return False

            first_target = self._stack_target_for_placed([0])
            if not self._pick_and_stack_one(interrupted_idx, first_target):
                return False

            if self.simplified_mode_enabled():
                for _ in range(self.FINAL_SETTLE_STEPS):
                    self.step_sim()
                return True

            second_target = self._stack_target_for_placed([0, interrupted_idx])
            if not self._pick_and_stack_one(other_idx, second_target):
                return False

        else:
            # Avatar interrupts the final bowl; this is the original flow.
            first_target = self._stack_target_for_placed([0])
            if not self._pick_and_stack_one(other_idx, first_target):
                return False

            if not self._run_avatar_interrupt(interrupted_idx, arm_tag):
                return False

            second_target = self._stack_target_for_placed([0, other_idx])
            if not self._pick_and_stack_one(interrupted_idx, second_target):
                return False

        for _ in range(self.FINAL_SETTLE_STEPS):
            self.step_sim()
        return True

    # ------------------------------------------------------------------
    # Success — inherit parent's loose footprint + higher-z gate
    # plus a hard-fail on avatar collision.
    # ------------------------------------------------------------------

    def check_success(self) -> bool:
        if self.avatar_collided:
            return False
        if bool(self.config.get("eval_mode", False)) and (
            getattr(self, "_eval_retreat_active", False)
            or getattr(self, "_eval_waiting_avatar_done", False)
            or getattr(self, "_eval_return_active", False)
            or getattr(self, "_eval_pending_interrupt_idx", None) is not None
        ):
            return False
        return super().check_success()

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
            "interrupted_bowl_idx": int(getattr(self, "_interrupted_idx", -1)),
            "pace_interrupt": {
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
                "trigger_dist_xy": getattr(self, "_pace_trigger_dist_xy", None),
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
