"""Dump-bin (assist): same scene as ``dump_bin`` but with a
human avatar that picks up an extra coloured cube from the table and tosses
it into the trash can using the ``Trash_clip`` generated motion.

Geometry pipeline (fulfilling user's "bin region → drop pose → avatar
pose → object pose" constraint chain):

  1. Pre-bake the avatar's left-palm trajectory by running the motion
     once at the default ``avatar_init_pos``/``avatar_init_rot`` and
     reading ``palm_at_attach`` (clip frame 25 ≈ orig 60) and
     ``palm_at_detach`` (clip frame 57 ≈ orig 92) in world coords.
  2. Choose a drop target inside the bin's xy region (default = bin
     centre).  Because rotation is fixed, the avatar must shift by
     ``shift_xy = drop_target_xy − palm_at_detach.xy`` to land its
     drop palm above the bin.
  3. Apply the same shift to ``palm_at_attach`` to obtain
     ``pickup_world`` — that becomes the avatar's blue cube spawn pose.
  4. Reset the avatar at the shifted ``T_new`` and replay the motion,
     attaching the cube at clip frame 25 and detaching at clip frame 57.
     Sim ``POST_FALL_FRAMES`` more steps so the cube falls into the bin.

The robot keeps doing its existing serial pick-and-drop on
``self.objects`` — it is not aware of the avatar's cube.

Evaluation + scene + bin geometry inherited from ``DumpBin``.
``check_success`` adds the avatar cube to the in-bin count.
"""

import numpy as np

from ..avatar.eval_mode_mixin import EvalModeAvatarMixin
from ..utils import Pose, create_primitive, to_numpy
from ..task_bases.dump_bin import DumpBin


_GENERATED_MOTIONS_PKL = "avatars/motions/generated_motions.pkl"


class DumpBinAssist(EvalModeAvatarMixin, DumpBin):
    INSTRUCTION = "empty the table cubes into the trash bin while the human tosses another cube into the bin"

    use_avatar = True
    EASY_NUM_OBJECTS = 1
    HARD_NUM_OBJECTS = 2

    AVATAR_BASE_ROT = np.array(
        [[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=np.float64
    )
    # Initial avatar pose (re-shifted at runtime so the drop lands above
    # the bin).  This pose only needs to be a clean standing pose for
    # the dry-run; the final avatar position is computed from the
    # measured palm trajectory.
    avatar_init_pos = np.array([0.65, 0.45, -0.18])
    avatar_init_rot = AVATAR_BASE_ROT
    # Extra yaw composed onto AVATAR_BASE_ROT.  Negative = clockwise
    # viewed from +z (top-down).  Purpose: rotate the body away from
    # the table at the throw pose so the avatar's torso doesn't clip
    # through the table edge.
    AVATAR_YAW_DEG = -45.0
    # Cube rests on the table at pickup_world.xy and is snapped to the palm
    # only at the attach instant.
    AVATAR_LIFT_Z = 0.0
    # Shift drop_target this many metres from bin center *toward* the
    # avatar's initial position.  Net effect: the avatar's body
    # backs up by the same amount (because the constraint chain holds
    # palm@detach == drop_target), giving more clearance from the
    # table edge at the throw pose.  Cube still lands inside the bin
    # so long as the offset stays under min(x_half, y_half).
    AVATAR_BACK_OFFSET = 0.05
    AVATAR_PALM_FORWARD_OFFSET = 0.0
    # Post-solve avatar shift. Cube stays at pickup_world; the captured
    # hand-frame offset absorbs the avatar body shift.
    AVATAR_UP_SHIFT = 0.0
    AVATAR_TOWARD_ARM_SHIFT = 0.0
    # Extra body shift away from the bin increases clearance between the
    # avatar idle arm and the robot's bin-side drop/retreat airspace.
    AVATAR_AWAY_FROM_BIN_SHIFT = 0.0
    # Post-solve body shift in the avatar-frame left direction. Cube is
    # unchanged and the captured hand-frame offset absorbs the shift.
    AVATAR_LEFT_SHIFT = 0.0
    # Franka base xy (cf. envs.robot.franka_robot.FrankaRobot default
    # `pos=[0.0, -0.65, 0.75]`).  Used to define "toward arm".
    _FRANKA_BASE_XY = (0.0, -0.65)

    ASSIST_MOTION_NAME = "Trash_clip"
    ATTACH_FRAME = 25       # clip-frame (= orig 60)
    DETACH_FRAME = 57       # clip-frame (= orig 92)
    # Robot and avatar run in parallel. The frame ratio stretches the throw
    # clip, then AvatarController eases back to idle after detach.
    AVATAR_FRAME_RATIO = 18
    AVATAR_DELAY_STEPS = 0
    ROBOT_START_AFTER_AVATAR = True
    ROBOT_START_WAIT_MAX_STEPS = 1400
    POST_DETACH_HOLD_FRAMES = 10  # zero cube velocity after auto-detach
    IDLE_RETURN_FRAMES = 120      # smooth ease back to idle pose post-motion (4x slower than v9)
    POST_FALL_FRAMES = 500  # sim steps after detach so the cube falls + settles
                            # (cube drop height ≈ palm@57.z − bin floor ≈ 0.5 m;
                            # at dt=0.002 s, 500 steps = 1 s of physics, plenty)
    AVATAR_HAND_ID = 0      # 0 = left, 1 = right
    AVATAR_CUBE_HALF = 0.020
    # Cubes are randomly coloured per episode from this palette.
    CUBE_PALETTE = (
        (0.20, 0.30, 0.85),  # blue
        (0.20, 0.75, 0.35),  # green
        (0.95, 0.85, 0.20),  # yellow
        (0.20, 0.80, 0.85),  # cyan
        (0.85, 0.30, 0.85),  # magenta
        (0.55, 0.30, 0.85),  # purple
        (0.95, 0.55, 0.75),  # pink
        (0.10, 0.65, 0.55),  # teal
    )
    AVATAR_CUBE_PALETTE = CUBE_PALETTE
    # High park used only while the avatar hand approaches the assist cube.
    # The robot starts in its normal home posture, then eases to this posture
    # shortly before the avatar attach frame so the retreat visually matches
    # the human approach instead of happening at motion start.
    ROBOT_PARK_QPOS = np.array(
        [0.0, -0.75, 0.0, -2.35, 0.0, 2.00, 0.78539816],
        dtype=np.float64,
    )
    ROBOT_RETREAT_STEPS = 180
    DROP_HOVER_OVER_RIM = 0.10
    AVATAR_DRY_RUN_MAX_STEPS = 1000
    AVATAR_CUBE_PARK_POS = (8.0, 8.0, 0.025)
    AVATAR_CUBE_TABLE_CLEARANCE = 0.001
    AVATAR_POST_DRY_RUN_SETTLE_STEPS = 80
    AVATAR_PARALLEL_TAIL_STEPS = 600
    AVATAR_POST_HOLD_TAIL_STEPS = 700
    EVAL_TRIGGER_STEP_MIN = 20
    EVAL_TRIGGER_STEP_MAX = 60
    COLOR_RESAMPLE_ATTEMPTS = 8
    IDENTITY_QUAT = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    def __init__(self, config: dict = None):
        cfg = dict(config or {})
        # Make sure the AvatarController loads our Trash_clip motion.
        avatar_cfg = dict(cfg.get("avatar", {}))
        avatar_cfg.setdefault("generated_motion_data", _GENERATED_MOTIONS_PKL)
        cfg["avatar"] = avatar_cfg
        super().__init__(cfg)

    # ------------------------------------------------------------------
    # Scene setup — extend parent with one extra cube for the avatar.
    # The cube starts at a far-away placeholder position; it gets
    # teleported to ``pickup_world`` once the dry run computes the
    # avatar geometry.
    # ------------------------------------------------------------------

    def load_actors(self):
        # Parent resolves the robot-side count from easy/hard mode; the assist
        # avatar always gets its own extra cube below.
        super().load_actors()
        # Pick a per-episode random colour from the curated palette.
        # np.random was seeded by BaseTask.reset(seed) before this call.
        idx = int(np.random.randint(len(self.AVATAR_CUBE_PALETTE)))
        cube_color = self.AVATAR_CUBE_PALETTE[idx]
        self.avatar_cube = create_primitive(
            self.scene,
            "box",
            Pose(self.AVATAR_CUBE_PARK_POS),
            size={"half_size": (self.AVATAR_CUBE_HALF,) * 3},
            color=cube_color,
        )

    def _random_cube_color(self, avoid_idx: int = None):
        idx = int(np.random.randint(len(self.CUBE_PALETTE)))
        if avoid_idx is not None and len(self.CUBE_PALETTE) > 1:
            for _ in range(self.COLOR_RESAMPLE_ATTEMPTS):
                if idx != avoid_idx:
                    break
                idx = int(np.random.randint(len(self.CUBE_PALETTE)))
        return idx, self.CUBE_PALETTE[idx]

    def _load_cubes(self):
        """Spawn robot-side dump objects with per-episode random colours."""
        spawn_xs, spawn_ys = self._sample_cube_xys()
        self.objects = []
        self._object_pick_radii = {}
        self._object_labels = {}
        self._object_model_ids = {}
        self._object_close_values = {}
        # Use the shared object catalog when an object is forced
        # (config object_name/dump_object_name) or random catalog objects are
        # requested; otherwise fall back to the legacy random primitive cube.
        # (`_forced_dump_object_id` was the pre-catalog forcing flag and no
        # longer exists after the object-catalog refactor.)
        forced = bool(
            self.config.get("object_name") or self.config.get("dump_object_name")
        )
        if forced or self._dump_random_objects:
            self._episode_dump_specs = [
                entry.as_spec()
                for entry in self.resolve_target_objects(self.num_objects, replace=False)
            ]
        used = []
        for i in range(self.num_objects):
            if forced or self._dump_random_objects:
                actor, radius, label = self._spawn_dump_object(
                    float(spawn_xs[i]), float(spawn_ys[i]), i,
                )
            else:
                idx, cube_color = self._random_cube_color(
                    avoid_idx=used[-1] if used else None
                )
                used.append(idx)
                actor = create_primitive(
                    self.scene,
                    "box",
                    Pose([
                        float(spawn_xs[i]),
                        float(spawn_ys[i]),
                        self.TABLE_TOP_Z + self.OBJECT_HALF + self.PRIMITIVE_OBJECT_Z_CLEARANCE,
                    ]),
                    size={"half_size": (self.OBJECT_HALF,) * 3},
                    color=cube_color,
                )
                radius, label = self.OBJECT_HALF, "cube"
            self.objects.append(actor)
            self._object_pick_radii[id(actor)] = float(radius)
            self._object_labels[id(actor)] = str(label)
            self._object_model_ids[id(actor)] = int(
                getattr(self, "_last_dump_object_spec", {}).get("model_id", 0)
            )
            if forced or self._dump_random_objects:
                spec = getattr(self, "_last_dump_object_spec", {})
                if "close_value" in spec:
                    self._object_close_values[id(actor)] = float(spec["close_value"])

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

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
        """Average of the 4 finger knuckles (Index1/Middle1/Ring1/Pinky1)
        — same recipe as ``envs.avatar.robot.get_palm_center``."""
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
        """World position of the wrist joint (`{Side}Hand` bone)."""
        side = "Left" if hand_id == 0 else "Right"
        return np.asarray(
            self.avatar.robot.skin.get_global_translation(f"{side}Hand")[0]
        ).ravel()[:3]

    def _avatar_dry_run(self):
        """Step the motion at the default pose, return
        ``(palm_at_attach_world, palm_at_detach_world, wrist_at_attach_world)``."""
        init_pos = self._avatar_init_pos_lifted()
        init_rot = self._avatar_rot_with_yaw()
        self.avatar.reset(global_trans=init_pos, global_rot=init_rot)
        self.avatar.play_animation(self.ASSIST_MOTION_NAME)

        palm_at_attach = palm_at_detach = wrist_at_attach = None
        step = 0
        while not self.avatar.spare() and step < self.AVATAR_DRY_RUN_MAX_STEPS:
            self.scene.step()           # raw step — skip step_sim() so we
            self.avatar.step()          # don't write to the video recorder
            if step == self.ATTACH_FRAME:
                palm_at_attach = self._avatar_palm_center(self.AVATAR_HAND_ID)
                wrist_at_attach = self._avatar_wrist(self.AVATAR_HAND_ID)
            if step == self.DETACH_FRAME:
                palm_at_detach = self._avatar_palm_center(self.AVATAR_HAND_ID)
            step += 1
        # If avatar.spare() goes True on the same step DETACH_FRAME would
        # have fired, the loop body never runs — sample once more.
        if palm_at_detach is None:
            palm_at_detach = self._avatar_palm_center(self.AVATAR_HAND_ID)
        if palm_at_attach is None:
            raise RuntimeError(
                f"dry-run never reached attach frame {self.ATTACH_FRAME} "
                f"(loop exited at step {step})"
            )
        return palm_at_attach, palm_at_detach, wrist_at_attach

    def _prepare_avatar_assist_geometry(self) -> np.ndarray:
        palm_at_attach, palm_at_detach, _ = self._avatar_dry_run()

        # Constraint solve: drop above bin xy → avatar shift → pickup.
        # Pick a drop target shifted from bin centre toward the avatar
        # (in xy).  Because shift_xy = drop_target − palm@detach, this
        # also shifts T_new toward the avatar's initial pose, i.e. it
        # backs the body away from the table edge by the same amount.
        bin_xy = np.array(self.big_bin_pose.p[:2], dtype=float)
        avatar_init_xy = np.array(self.avatar_init_pos[:2], dtype=float)
        toward_avatar = avatar_init_xy - bin_xy
        norm = np.linalg.norm(toward_avatar)
        if norm > 1e-9:
            toward_avatar = toward_avatar / norm
        drop_target_xy = bin_xy + toward_avatar * self.AVATAR_BACK_OFFSET
        shift_xy = drop_target_xy - palm_at_detach[:2]
        T_new = self._avatar_init_pos_lifted().copy()
        T_new[:2] += shift_xy
        pickup_world = palm_at_attach.copy()
        pickup_world[:2] += shift_xy
        # Force pickup z to table_top + cube_half so the cube can collide
        # stably instead of spawning inside the table.
        cube_rest_z = self.TABLE_TOP_Z + self.AVATAR_CUBE_HALF + self.AVATAR_CUBE_TABLE_CLEARANCE
        pickup_world[2] = cube_rest_z

        # Apply rigid-body shifts to T_new. Cube stays at pickup_world and
        # the captured hand-frame offset absorbs the avatar body shifts.
        # toward_arm = unit xy from current T_new toward Franka base.
        franka_xy = np.array(self._FRANKA_BASE_XY, dtype=float)
        toward_arm_xy = franka_xy - T_new[:2]
        ta_norm = np.linalg.norm(toward_arm_xy)
        if ta_norm > 1e-9:
            toward_arm_xy = toward_arm_xy / ta_norm
        T_new[:2] += toward_arm_xy * self.AVATAR_TOWARD_ARM_SHIFT
        T_new[2] += self.AVATAR_UP_SHIFT
        # Direction taken from avatar_init→bin so it stays stable across
        # drop_target changes.
        away_from_bin = avatar_init_xy - bin_xy
        afb_norm = np.linalg.norm(away_from_bin)
        if afb_norm > 1e-9:
            away_from_bin = away_from_bin / afb_norm
        T_new[:2] += away_from_bin * self.AVATAR_AWAY_FROM_BIN_SHIFT

        # forward_xy = toward_arm_xy (already a unit vector).  Left =
        # rotate forward by +90° about world +z = (-fy, fx).
        left_xy = np.array(
            [-toward_arm_xy[1], toward_arm_xy[0]], dtype=float
        )
        T_new[:2] += left_xy * self.AVATAR_LEFT_SHIFT

        # ---- Position avatar cube on the table at pickup.xy, reset avatar ----
        self.avatar_cube.set_pos(pickup_world.astype(float))
        self.avatar_cube.set_quat(self.IDENTITY_QUAT)
        # Zero any prior velocity so the cube settles cleanly under
        # gravity rather than carrying jitter from earlier sim steps.
        try:
            self.avatar_cube.set_dofs_velocity(np.zeros(6, dtype=np.float64))
        except Exception:
            pass
        self.avatar.reset(
            global_trans=T_new,
            global_rot=self._avatar_rot_with_yaw(),
        )
        # The dry-run / default-pose reset can populate collision logs
        # before the real episode geometry is installed.  Start grading
        # only after the avatar is shifted to its solved assist pose.
        self.avatar_collided = False
        self.avatar_collision_log = []
        self._avatar_collision_tick = 0

        # ---- 1) Settle table cubes (parent does this implicitly) ----
        for _ in range(self.AVATAR_POST_DRY_RUN_SETTLE_STEPS):
            self.step_sim()
        return pickup_world

    # ------------------------------------------------------------------
    # Rollout
    # ------------------------------------------------------------------

    def play_once(self) -> bool:
        arm_tag = "right"
        self._boost_finger_pd(arm_tag)

        # ---- 0) Dry-run avatar to discover palm trajectory ----
        if self.avatar is None:
            return super().play_once()

        pickup_world = self._prepare_avatar_assist_geometry()

        # ---- 2) Robot + avatar in parallel via patched step_sim ----
        # Robot starts pick at sim_step 0; avatar's play_animation
        # fires after AVATAR_DELAY_STEPS sim steps so the robot has
        # a head-start.  AVATAR_FRAME_RATIO stretches the motion so
        # the throw is ~3x slower than v7.  Auto-attach / auto-detach
        # come from PlayAnimationMotion.start(attach_obj=...,
        # attach_frame=..., detach_frame=...).  After auto-detach we
        # hold-and-zero the cube for POST_DETACH_HOLD_FRAMES sim
        # steps to kill the throw velocity.
        self._run_parallel(pickup_world, arm_tag)

        return True

    def play_blind_once(self) -> bool:
        arm_tag = "right"
        self._boost_finger_pd(arm_tag)

        for _ in range(self.TABLE_SETTLE_STEPS):
            self.step_sim()

        if self.avatar is not None and bool(self.config.get("eval_mode", False)):
            self._eval_at_step(int(getattr(
                self, "_eval_assist_trigger_step", self.EVAL_TRIGGER_STEP_MIN,
            )))

        if not self.objects:
            return False
        seed = int(self.config.get("eval_seed", 0))
        rng = np.random.RandomState(seed + 62017)
        idx = int(rng.randint(0, len(self.objects)))
        self._blind_robot_object_idx = idx
        try:
            ok = self._pick_and_drop_cube(self.objects[idx], arm_tag)
        except Exception:
            ok = False
        if not ok:
            self.open_gripper(arm_tag)
            for _ in range(self.FAILURE_RECOVERY_STEPS):
                self.step_sim()
        if self.avatar is not None:
            tail = 0
            while not self.avatar.spare() and tail < self.AVATAR_PARALLEL_TAIL_STEPS:
                self.step_sim()
                tail += 1
        return True

    # ------------------------------------------------------------------
    # Eval-mode avatar assist
    # ------------------------------------------------------------------

    def _eval_at_reset(self) -> None:
        self._eval_assist_trigger_step = 0
        self._eval_assist_fired = True
        self._eval_assist_failed = False
        try:
            self._start_eval_avatar_assist()
        except Exception as exc:
            self._eval_assist_failed = True
            if self.config.get("debug_eval_avatar", False):
                print(f"[dump_bin_assist/eval] avatar assist failed: {exc}", flush=True)

    def _eval_at_step(self, step_idx: int) -> None:
        if (getattr(self, "_eval_assist_fired", False)
                or getattr(self, "_eval_assist_failed", False)):
            return
        if step_idx < int(getattr(self, "_eval_assist_trigger_step", self.EVAL_TRIGGER_STEP_MIN)):
            return
        self._eval_assist_fired = True
        try:
            self._start_eval_avatar_assist()
        except Exception as exc:
            self._eval_assist_failed = True
            if self.config.get("debug_eval_avatar", False):
                print(f"[dump_bin_assist/eval] avatar assist failed: {exc}", flush=True)

    def _start_eval_avatar_assist(self) -> None:
        self._prepare_avatar_assist_geometry()
        self.avatar.frame_ratio = self.AVATAR_FRAME_RATIO
        self.avatar.play_animation(
            self.ASSIST_MOTION_NAME,
            attach_obj=self.avatar_cube,
            hand_id=self.AVATAR_HAND_ID,
            attach_frame=int(self.ATTACH_FRAME * self.AVATAR_FRAME_RATIO),
            detach_frame=int(self.DETACH_FRAME * self.AVATAR_FRAME_RATIO),
            return_to_idle_after=self.IDLE_RETURN_FRAMES,
        )

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
            "blind_robot_object_idx": int(getattr(
                self, "_blind_robot_object_idx", -1,
            )),
        })
        return metrics

    def _run_parallel(self, pickup_world: np.ndarray, arm_tag: str):
        """Run the robot's pick-and-drop sequence with the avatar
        animating in parallel.

        Approach: monkey-patch ``self.step_sim`` for the duration of
        the parallel phase.  Every wrapped sim step
          (a) increments a shared counter,
          (b) at counter == AVATAR_DELAY_STEPS kicks off
              ``avatar.play_animation`` with auto-attach +
              auto-detach at the resampled motion frames,
          (c) detects the auto-detach edge (attached_object goes
              None) and sets up POST_DETACH_HOLD_FRAMES of
              pin-position + zero-velocity so the cube doesn't
              inherit the throw velocity,
          (d) finally calls the original step_sim (which advances
              physics + ``avatar.step()`` + recorders).
        Then the robot's normal pick-and-drop loop runs underneath,
        and ``execute_plan`` ticking through ``self.step_sim`` makes
        the avatar advance in lockstep.
        """
        self.avatar.frame_ratio = self.AVATAR_FRAME_RATIO
        attach_resampled = int(self.ATTACH_FRAME * self.AVATAR_FRAME_RATIO)
        detach_resampled = int(self.DETACH_FRAME * self.AVATAR_FRAME_RATIO)
        zero6 = np.zeros(6, dtype=np.float64)
        identity_q = self.IDENTITY_QUAT
        HAND_ID = self.AVATAR_HAND_ID

        state = {
            "sim_step": 0,
            "avatar_started": False,
            "prev_attached": None,
            "post_hold": 0,
            "drop_pos": None,
            "attach_logged": False,
            "detach_logged": False,
        }

        original_step_sim = type(self).step_sim.__get__(self, type(self))

        def patched_step_sim():
            state["sim_step"] += 1
            # Kick off avatar after delay
            if (not state["avatar_started"]
                    and state["sim_step"] >= self.AVATAR_DELAY_STEPS):
                state["avatar_started"] = True
                self.avatar.play_animation(
                    self.ASSIST_MOTION_NAME,
                    attach_obj=self.avatar_cube,
                    hand_id=HAND_ID,
                    attach_frame=attach_resampled,
                    detach_frame=detach_resampled,
                    return_to_idle_after=self.IDLE_RETURN_FRAMES,
                )
            # Detect attach / detach edges
            if state["avatar_started"]:
                cur = self.avatar.robot.attached_object[HAND_ID]
                if (cur is not None and not state["attach_logged"]):
                    state["attach_logged"] = True
                if (state["prev_attached"] is not None
                        and cur is None
                        and not state["detach_logged"]):
                    state["drop_pos"] = self._avatar_palm_center(HAND_ID).astype(float)
                    state["post_hold"] = self.POST_DETACH_HOLD_FRAMES
                    state["detach_logged"] = True
                state["prev_attached"] = cur
            # Hold cube + zero velocity for K steps post-detach
            if state["post_hold"] > 0 and state["drop_pos"] is not None:
                self.avatar_cube.set_pos(state["drop_pos"])
                self.avatar_cube.set_quat(identity_q)
                try:
                    self.avatar_cube.set_dofs_velocity(zero6)
                except Exception:
                    pass
                state["post_hold"] -= 1
            return original_step_sim()

        # Install patched step_sim on self (instance attribute shadows class method).
        self.step_sim = patched_step_sim
        try:
            if self.ROBOT_START_AFTER_AVATAR:
                waited = 0
                park_started = False
                park_start_step = max(
                    self.AVATAR_DELAY_STEPS + 1,
                    attach_resampled - self.ROBOT_RETREAT_STEPS,
                )
                while waited < self.ROBOT_START_WAIT_MAX_STEPS:
                    if (not park_started) and state["sim_step"] >= park_start_step:
                        park_started = True
                        self._move_robot_to_park(arm_tag)
                        continue
                    self.step_sim()
                    waited += 1
                    if state["detach_logged"] and self.avatar.spare():
                        break
            for i, cube in enumerate(self.objects):
                try:
                    ok = self._pick_and_drop_cube(cube, arm_tag)
                except Exception:
                    ok = False
                if not ok:
                    self.open_gripper(arm_tag)
                    for _ in range(self.FAILURE_RECOVERY_STEPS):
                        self.step_sim()
            # If avatar hasn't finished by the time the robot is done,
            # let it tick to completion (with patched step_sim still in
            # place so any pending detach + post-hold fires).
            tail = 0
            while not self.avatar.spare() and tail < self.AVATAR_PARALLEL_TAIL_STEPS:
                self.step_sim()
                tail += 1
            # Drain any remaining post-detach hold frames after motion.
            while state["post_hold"] > 0 and tail < self.AVATAR_POST_HOLD_TAIL_STEPS:
                self.step_sim()
                tail += 1
        finally:
            # Restore class-level step_sim.  Setting to None lets the
            # descriptor lookup fall back to the class attribute.
            try:
                del self.step_sim
            except AttributeError:
                pass
            self.avatar.frame_ratio = 1.0

    def _move_robot_to_park(self, arm_tag: str):
        arm = self.robot.get_arm(arm_tag)
        start_q = np.array(arm.get_arm_qpos(), dtype=float)
        target_q = np.array(self.ROBOT_PARK_QPOS, dtype=float)
        for i in range(1, self.ROBOT_RETREAT_STEPS + 1):
            t = i / self.ROBOT_RETREAT_STEPS
            s = t * t * (3.0 - 2.0 * t)
            q = start_q * (1.0 - s) + target_q * s
            self.robot.set_arm_joints(q, arm_tag)
            self.step_sim()

    # ------------------------------------------------------------------
    # Success — add the avatar's cube to the in-bin count.
    # ------------------------------------------------------------------

    def check_success(self) -> bool:
        parent_pass = super().check_success()
        if not hasattr(self, "avatar_cube"):
            return parent_pass

        big_x, big_y = self.big_bin_pose.p[:2]
        rim_z = self._big_bin_rim_z
        x_half, y_half = self._big_bin_xy_half
        p = to_numpy(self.avatar_cube.get_pos()).ravel()[:3]
        in_xy = abs(p[0] - big_x) < x_half and abs(p[1] - big_y) < y_half
        below_r = p[2] < rim_z + self.SUCCESS_RIM_MARGIN
        above_f = p[2] > self.SUCCESS_FLOOR_Z_MIN
        avatar_inside = in_xy and below_r and above_f
        # Pass condition: parent passed AND avatar's cube is in the bin.
        return parent_pass and avatar_inside
