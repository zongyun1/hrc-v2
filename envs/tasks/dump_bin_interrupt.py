"""Dump-bin (interrupt): same scene as ``dump_bin`` but with a
human avatar that picks up one of the cubes mid-task.

Flow per episode:
  1. Robot picks zero or more non-interrupted cubes, depending on the
     randomly sampled interrupt pick ordinal.
  2. Robot approaches cube[INTERRUPTED_IDX] (above-pose, no commit) when
     that ordinal is reached.
  3. Avatar starts the ``Inspect`` motion (categorize_interrupt's
     verbatim animation) with ``attach_obj`` = the interrupted cube.
     The cube is attached to the avatar's hand at frame ATTACH and
     detached at DETACH — the avatar lifts it, holds it, and sets it
     back on the table.
  4. Robot retreats to a safe pose (toward the robot base, away from
     the avatar's lean cone).
  5. While the avatar holds the interrupted cube, the robot picks any
     remaining OTHER cubes and drops them in the bin — all pure physics.
  6. Robot waits for ``avatar.spare()``.
  7. Robot reads cube[INTERRUPTED_IDX]'s LATEST pose (after the avatar
     puts it back down) and picks it via the same pure-physics primitive.

Geometry pipeline (mirrors dump_bin_assist's "dry run → shift" idiom):
  * Pre-bake: dry-run the Inspect motion at ``avatar_init_pos`` and
    measure the avatar's right-palm position at ATTACH frame.
  * Constraint solve: avatar shift = cube[INTERRUPTED].pos -
    palm_at_attach.  After shifting, the avatar palm meets the cube
    exactly at the attach frame so the attach is geometrically correct.
  * Every cube starts inside the same avatar-reachable band near the
    +Y table edge, so the interrupted target is not revealed by placement.

Eval: success = ALL robot-side cubes inside the bin.

Robot motion is pure physics (no kinematic teleport/attach).  Avatar
attaches the cube to its hand via ``play_animation(attach_obj=...)``,
which is the sanctioned avatar mechanism.
"""

import numpy as np

from ..avatar.inspect_motion_mixin import InspectMotionMixin, STANDARD_INSPECT_POOL
from ..task_bases.dump_bin import DumpBin


class DumpBinInterrupt(InspectMotionMixin, DumpBin):
    INSTRUCTION = (
        "place each cube into the trash can; the human will inspect "
        "one cube midway through - wait, then continue."
    )

    use_avatar = True
    HARD_NUM_OBJECTS = 3

    # Avatar at +Y side of the table, facing -Y (toward the table).
    # AVATAR_BASE_ROT maps mesh-+X (avatar forward) to world-(-Y).
    AVATAR_BASE_ROT = np.array(
        [[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=np.float64
    )
    avatar_init_pos = np.array([0.0, 0.55, -0.18])
    avatar_init_rot = AVATAR_BASE_ROT

    # categorize_interrupt's Inspect motion + frame markers.  The same
    # 10× slowdown reads as a deliberate inspection in video.
    INSPECT_MOTION_NAME   = "Inspect1"
    INSPECT_ATTACH_FRAME  = 34       # original frame
    INSPECT_DETACH_FRAME  = 187      # original frame
    INSPECT_POOL = STANDARD_INSPECT_POOL
    AVATAR_MOTION_SLOW    = 10.0     # avatar.frame_ratio
    AVATAR_HAND_ID        = 1        # 1 = right hand
    AVATAR_EXTRA_Z        = 0.04     # palm-above-cube margin so attach
                                     # doesn't sink into the cube
    EVAL_HIDDEN_AVATAR_POS = np.array([0.0, 4.0, -0.18], dtype=np.float64)
    _AVATAR_OBSTACLE_RES = 0.04
    _AVATAR_INFLATE_FACTOR = 1.8

    # Shared avatar-reachable spawn band near the +Y table edge.  All cubes
    # are sampled here so the robot cannot infer the interrupt target from
    # which object is uniquely close enough for the avatar to inspect.
    AVATAR_REACH_SPAWN_X_RANGE = (-0.18, 0.18)
    AVATAR_REACH_SPAWN_Y_RANGE = (-0.12, 0.06)
    INTERRUPT_SPAWN_X_RANGE = AVATAR_REACH_SPAWN_X_RANGE
    INTERRUPT_SPAWN_Y_RANGE = AVATAR_REACH_SPAWN_Y_RANGE
    INTERRUPT_LAYOUT_SAMPLE_ATTEMPTS = 80

    # Threshold for success — override parent's ⌈N/2⌉ to require ALL
    # cubes in bin (avatar puts its cube back; robot must pick that too).
    SUCCESS_THRESHOLD = None         # None ⇒ num_objects (all in)

    # ------------------------------------------------------------------
    # Eval-mode interrupt schedule (config["eval_mode"]=True)
    # Scripted-flow (collect.py / play_once) ignores these.
    # ------------------------------------------------------------------
    # Range from which the trigger step is drawn each episode (in policy
    # steps — one per take_action call ≈ 50 ms / 20 Hz with default
    # action_substeps=25).  100-300 → 5-15s into the rollout, late enough
    # for the policy to commit to a cube and early enough to leave time
    # for it to recover.
    EVAL_TRIGGER_STEP_MIN = 100
    EVAL_TRIGGER_STEP_MAX = 300
    # Linear-interpolated avatar slide from idle to the grasp pose.
    EVAL_GLIDE_POLICY_STEPS = 20
    EVAL_TABLE_Z_LOWER_MARGIN = 0.05
    EVAL_TABLE_Z_UPPER_MARGIN = 0.30
    EVAL_AVATAR_BIN_CLEARANCE = 0.20
    EVAL_FORCE_TRIGGER_DIST_XY = 0.11
    EVAL_ROBOT_RETREAT_POLICY_STEPS = 35
    EVAL_ROBOT_RETREAT_BLEND = 0.50
    EVAL_ROBOT_RETURN_POLICY_STEPS = 35
    EVAL_ROBOT_RETURN_BLEND = 0.50
    PACE_TRIGGER_PROGRESS = 0.60
    PACE_TRIGGER_DIST_XY = 0.18
    PACE_PROGRESS_START_DIST_XY = 0.42
    PACE_MIN_POLICY_STEPS = 20
    CALIBRATION_SETTLE_STEPS = 20
    POST_CALIBRATION_SETTLE_STEPS = 30
    POST_ATTACH_WAIT_MARGIN_STEPS = 100
    POST_RELEASE_SETTLE_STEPS = 120
    POST_AVATAR_RETURN_SETTLE_STEPS = 30
    # None ⇒ the avatar freezes on the inspect motion's last frame (pose +
    # position) instead of easing back to the neutral idle pose.
    INSPECT_RETURN_TO_IDLE_FRAMES = None
    RETREAT_BACK_DISTANCE = 0.15
    RETREAT_UP_DISTANCE = 0.15

    # ------------------------------------------------------------------
    # Scene setup
    # ------------------------------------------------------------------

    def _sample_cube_xys(self):
        """Override parent: pick a random interrupted cube, then spawn every
        cube in the same avatar-reachable sub-region with min-spacing
        rejection.  ``self._interrupted_idx`` is selected uniformly per seed.
        """
        if bool(self.config.get("interrupt_use_base_cube_layout", False)):
            self._interrupted_idx = 0
            self._interrupt_pick_ordinal = 0
            return super()._sample_cube_xys()
        self._interrupted_idx = int(np.random.randint(0, self.num_objects))
        self._interrupt_pick_ordinal = self._sample_interrupt_pick_ordinal()
        SX_LO, SX_HI = self.AVATAR_REACH_SPAWN_X_RANGE
        SY_LO, SY_HI = self.AVATAR_REACH_SPAWN_Y_RANGE
        min_spacing_sq = self.MIN_OBJECT_SPACING ** 2
        for _ in range(self.INTERRUPT_LAYOUT_SAMPLE_ATTEMPTS):
            xs = np.empty(self.num_objects)
            ys = np.empty(self.num_objects)
            for i in range(self.num_objects):
                xs[i] = np.random.uniform(SX_LO, SX_HI)
                ys[i] = np.random.uniform(SY_LO, SY_HI)
            ok = True
            for i in range(self.num_objects):
                for j in range(i + 1, self.num_objects):
                    if (xs[i] - xs[j]) ** 2 + (ys[i] - ys[j]) ** 2 < min_spacing_sq:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                return xs, ys
        raise RuntimeError("[dump_bin_interrupt] could not sample non-overlapping layout")

    def _sample_interrupt_pick_ordinal(self) -> int:
        """Return the 0-based robot pick ordinal where the avatar interrupts.

        Config ``interrupt_pick_ordinal`` may force a 1-based ordinal for
        review videos (1, 2, or 3).  Without it, every episode samples
        uniformly across all robot pick ordinals.
        """
        forced = self.config.get("interrupt_pick_ordinal")
        if forced is not None:
            ordinal = int(forced) - 1
            if not 0 <= ordinal < self.num_objects:
                raise ValueError(
                    "interrupt_pick_ordinal must be in "
                    f"[1, {self.num_objects}], got {forced!r}"
                )
            return ordinal
        return int(np.random.randint(0, self.num_objects))

    # ------------------------------------------------------------------
    # Calibration in reset() — mirrors categorize_interrupt
    # ------------------------------------------------------------------

    def reset(self, seed: int = 0):
        """Two flows share this entry point:

        - **Scripted (collect.py / play_once)** — pre-shifts the avatar so
          its palm meets ``self.objects[_interrupted_idx]`` exactly at the
          (scaled) attach frame.  The whole rollout is choreographed in
          ``play_once``.
        - **Eval (config["eval_mode"]=True)** — captures ``palm_default``
          via dry-run but leaves the avatar at idle.  A state machine in
          ``take_action`` fires the interrupt at a random step T∈[100,300]
          on whichever cube is currently closest to the gripper (and still
          on the table).  See `_eval_step_avatar`.

        The dry-run runs BEFORE ``start_video`` (collect.py / eval.py both
        start video AFTER reset returns), so the calibration sweep is
        invisible in the recorded video.
        """
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
        if self._eval_hide_avatar_until_trigger_enabled():
            self._hide_eval_avatar_until_trigger()
            if not self.config.get("skip_reset_obs", False):
                obs = self.get_obs()
        return obs

    @property
    def _attach_frame_scaled(self):
        return int(round(self.INSPECT_ATTACH_FRAME * self.AVATAR_MOTION_SLOW))

    @property
    def _detach_frame_scaled(self):
        return int(round(self.INSPECT_DETACH_FRAME * self.AVATAR_MOTION_SLOW))

    def _dryrun_capture_palm(self):
        """Play the Inspect motion at the avatar's home pose, capture palm
        position at the (scaled) attach frame, then teleport the avatar back
        to its home pose.  Returns ``palm_default`` (np.float64[3]) or None
        if the attach frame was never reached.
        """
        # BaseTask.reset() creates the VLA recorder before this task runs
        # its calibration dry-run.  Normal MP4 recording starts after reset,
        # but VLA capture is ticked from step_sim immediately, so suppress it
        # here or the saved episode contains this hidden Inspect replay.
        saved_vla_recorder = self.vla_recorder
        saved_record_stride = self._record_stride
        self.vla_recorder = None
        self._record_stride = None
        try:
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
            # Restore avatar to home (clears any residual pose from Inspect).
            self.avatar.reset(
                np.asarray(self.avatar_init_pos, dtype=np.float64).copy(),
                np.asarray(self.avatar_init_rot, dtype=np.float64).copy(),
            )
            for _ in range(self.CALIBRATION_SETTLE_STEPS):
                self.step_sim()
        finally:
            self.vla_recorder = saved_vla_recorder
            self._record_stride = saved_record_stride
        # Calibration ran the Inspect motion in dry-run mode; clear any
        # avatar-collision flags accumulated during that sweep.
        self.avatar_collided = False
        self.avatar_collision_log = []
        return palm_default

    def _calibrate_avatar_for_inspect(self):
        """Scripted-flow calibration: dry-run + shift the avatar so its palm
        lands exactly on ``self.objects[self._interrupted_idx]`` at the attach
        frame.  Validates that the shifted position stays beside the table
        and not overlapping the bin.
        """
        cube = self.objects[self._interrupted_idx]
        target_palm = self._dump_object_center(cube).astype(float)

        palm_default = self._dryrun_capture_palm()
        if palm_default is None:
            return

        shift = target_palm - palm_default
        shift[2] += self.AVATAR_EXTRA_Z
        shifted_pos = np.asarray(self.avatar_init_pos, dtype=np.float64) + shift
        idle_pos = self._set_inspect_motion_root_pos(shifted_pos)

        self.avatar.reset(
            idle_pos.copy(),
            np.asarray(self.avatar_init_rot, dtype=np.float64).copy(),
        )
        for _ in range(self.POST_CALIBRATION_SETTLE_STEPS):
            self.step_sim()

    # ------------------------------------------------------------------
    # Eval-mode interrupt state machine (config["eval_mode"]=True)
    # ------------------------------------------------------------------

    def _init_eval_interrupt_state(self):
        """Capture ``palm_default`` once (dry-run) and arm a state machine
        that fires the interrupt at a random step T in
        ``[EVAL_TRIGGER_STEP_MIN, EVAL_TRIGGER_STEP_MAX]``.  Avatar stays
        at home until the trigger fires.
        """
        no_human_pace = bool(self.config.get("pace_no_human_ablation", False))
        self._eval_palm_default = None if no_human_pace else self._dryrun_capture_palm()
        trigger_min = int(self.config.get(
            "eval_trigger_step_min",
            self.EVAL_TRIGGER_STEP_MIN,
        ))
        trigger_max = int(self.config.get(
            "eval_trigger_step_max",
            self.EVAL_TRIGGER_STEP_MAX,
        ))
        trigger_max = max(trigger_min, trigger_max)
        if self._eval_palm_default is None:
            self._eval_interrupt_fired = not no_human_pace
            self._eval_trigger_step = int(np.random.randint(
                trigger_min, trigger_max + 1))
        else:
            self._eval_trigger_step = int(np.random.randint(
                trigger_min, trigger_max + 1))
            self._eval_interrupt_fired = False
        self._eval_interrupt_trigger_mode = str(
            self.config.get("eval_interrupt_trigger_mode", "random")
        ).lower()
        self._eval_glide_active = False
        self._eval_glide_remaining = 0
        self._eval_glide_start_pos = None
        self._eval_glide_target_pos = None
        self._eval_target_cube = None
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

    def _eval_hide_avatar_until_trigger_enabled(self) -> bool:
        return (
            bool(self.config.get("eval_mode", False))
            and bool(self.config.get("eval_hide_avatar_until_trigger", False))
            and self.avatar is not None
        )

    def _eval_hidden_avatar_pos(self) -> np.ndarray:
        pos = self.config.get("eval_hidden_avatar_pos")
        if pos is None:
            return self.EVAL_HIDDEN_AVATAR_POS.copy()
        arr = np.asarray(pos, dtype=np.float64).ravel()
        if arr.size != 3:
            raise ValueError("eval_hidden_avatar_pos must be a 3-vector")
        return arr.copy()

    def _hide_eval_avatar_until_trigger(self) -> None:
        self.avatar.reset(
            self._eval_hidden_avatar_pos(),
            np.asarray(self.avatar_init_rot, dtype=np.float64).copy(),
        )

    def _eval_robot_yield_enabled(self) -> bool:
        return bool(self.config.get("eval_robot_yield_enabled", True))

    def take_action(self, action, action_type: str = "qpos"):
        """Eval mode adds a per-step hook that drives the interrupt state
        machine before delegating to ``BaseTask.take_action`` for the
        physics + observation update.
        """
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
                return DumpBin.take_action(self, action, action_type=action_type)
        return super().take_action(action, action_type=action_type)

    def play_blind_once(self) -> bool:
        """Non-full-state scripted baseline.

        The robot follows the ordinary dump-bin script and does not know
        which cube or timestep the avatar will interrupt.  While the scripted
        robot motion advances via `step_sim`, this wrapper ticks the eval-mode
        avatar interrupt state machine at the configured policy rate.
        """
        if self.avatar is None or not bool(self.config.get("eval_mode", False)):
            return DumpBin.play_once(self)

        action_substeps = max(1, int(self.config.get("action_substeps", 25)))
        orig_step_sim = self.step_sim
        tick = {"n": 0}

        def step_sim_with_eval_avatar():
            tick["n"] += 1
            if tick["n"] >= action_substeps:
                tick["n"] = 0
                self._eval_step_avatar()
            orig_step_sim()

        self.step_sim = step_sim_with_eval_avatar
        try:
            return DumpBin.play_once(self)
        finally:
            self.step_sim = orig_step_sim

    def _eval_step_avatar(self):
        """Per-policy-step state machine:
          1. If a glide is active, advance one step (linear-interpolated
             avatar pose).  When glide finishes, fire the Inspect animation
             with the target cube attached.
          2. Else if the trigger step has been reached and the interrupt
             hasn't fired, find the closest on-table cube, validate that
             the required avatar shift stays beside the table / off the
             bin, and start the glide.  If no candidate qualifies, latch
             the fired flag so we don't re-try and stay idle.
        """
        self._policy_step_count += 1

        if self._eval_return_active:
            self._eval_return_remaining -= 1
            if self._eval_return_remaining <= 0:
                self._eval_return_finish_after_apply = True
            return

        if self._eval_waiting_avatar_done:
            if self.avatar is not None and self.avatar.spare():
                self._eval_waiting_avatar_done = False
                if self._eval_robot_yield_enabled():
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

        # ---- Glide in progress -----------------------------------------
        if self._eval_glide_active:
            self._eval_glide_remaining -= 1
            glide_steps = max(1, int(getattr(
                self, "_eval_glide_policy_steps",
                self.EVAL_GLIDE_POLICY_STEPS,
            )))
            done_steps = glide_steps - self._eval_glide_remaining
            progress = min(1.0, max(0.0, done_steps / float(glide_steps)))
            cur = ((1.0 - progress) * self._eval_glide_start_pos +
                   progress * self._eval_glide_target_pos)
            rot = np.asarray(self.avatar_init_rot, dtype=np.float64).copy()
            self.avatar.reset(cur.copy(), rot)
            if self._eval_glide_remaining <= 0:
                self._eval_glide_active = False
                target_cube = self._eval_target_cube
                cube_pos = self._dump_object_center(target_cube)
                # Re-check the cube hasn't fallen / been picked during the
                # glide.  If it's no longer on the table, abort (avatar is
                # already at the grasp pose; just leave it there in idle).
                if cube_pos[2] < self.TABLE_TOP_Z - self.EVAL_TABLE_Z_LOWER_MARGIN:
                    return
                self._play_inspect_motion(
                    attach_obj=getattr(target_cube, "entity", target_cube),
                    hand_id=self.AVATAR_HAND_ID,
                    attach_frame=self._attach_frame_scaled,
                    detach_frame=self._detach_frame_scaled,
                    return_to_idle_after=self.INSPECT_RETURN_TO_IDLE_FRAMES,
                )
                self._eval_waiting_avatar_done = True
            return

        # ---- Should we trigger? ----------------------------------------
        if self._eval_interrupt_fired:
            return
        target_idx, target_dist = self._closest_table_cube_candidate()
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
                and target_dist <= float(self.config.get(
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
        # Latch immediately so we only try once even if no candidate found.
        self._eval_interrupt_fired = True

        if target_idx is None:
            return

        self._interrupted_idx = int(target_idx)
        self._eval_pending_interrupt_idx = int(target_idx)
        self._eval_target_cube = self.objects[target_idx]
        self._pace_trigger_reason = trigger_reason
        self._pace_trigger_step = int(self._policy_step_count)
        self._pace_trigger_progress = float(self._pace_best_progress)
        self._pace_trigger_dist_xy = (
            None if not np.isfinite(target_dist) else float(target_dist)
        )
        arm = self.robot.get_arm("right")
        self._eval_saved_qpos = np.asarray(
            arm.get_arm_qpos(), dtype=np.float64,
        ).copy()
        self._eval_saved_gripper = float(getattr(arm, "gripper_val", 1.0))
        if self._eval_robot_yield_enabled():
            self._start_eval_robot_retreat()
        else:
            self._start_eval_avatar_glide()

    def _start_eval_avatar_glide(self):
        if self.avatar is None:
            self._start_eval_robot_return()
            return
        target_idx = getattr(self, "_eval_pending_interrupt_idx", None)
        if target_idx is None:
            return
        target_cube = self.objects[int(target_idx)]
        target_cube_pos = self._dump_object_center(target_cube).astype(float)
        if target_cube_pos[2] < self.TABLE_TOP_Z - self.EVAL_TABLE_Z_LOWER_MARGIN:
            self._start_eval_robot_return()
            return
        shift = target_cube_pos - self._eval_palm_default
        shift[2] += self.AVATAR_EXTRA_Z
        target_avatar_pos = (
            np.asarray(self.avatar_init_pos, dtype=np.float64) + shift
        )

        # Validate the shifted avatar isn't on top of the bin (would be
        # geometrically nonsensical).  If invalid, stay idle.
        bin_x, bin_y = self.big_bin_pose.p[:2]
        bin_xh, bin_yh = self._big_bin_xy_half
        bad_xy = (
            abs(target_avatar_pos[0] - bin_x) < bin_xh + self.EVAL_AVATAR_BIN_CLEARANCE and
            abs(target_avatar_pos[1] - bin_y) < bin_yh + self.EVAL_AVATAR_BIN_CLEARANCE
        )
        if bad_xy:
            self._start_eval_robot_return()
            return

        self._eval_glide_start_pos = np.asarray(self.avatar_init_pos, dtype=np.float64).copy()
        self._eval_glide_target_pos = target_avatar_pos.copy()
        self._eval_glide_policy_steps = int(self.config.get(
            "eval_glide_policy_steps",
            self.EVAL_GLIDE_POLICY_STEPS,
        ))
        if self._eval_glide_policy_steps <= 0:
            self.avatar.reset(target_avatar_pos.copy(), np.asarray(
                self.avatar_init_rot, dtype=np.float64,
            ).copy())
            self._eval_target_cube = target_cube
            self._play_inspect_motion(
                attach_obj=getattr(target_cube, "entity", target_cube),
                hand_id=self.AVATAR_HAND_ID,
                attach_frame=self._attach_frame_scaled,
                detach_frame=self._detach_frame_scaled,
                return_to_idle_after=self.INSPECT_RETURN_TO_IDLE_FRAMES,
            )
            self._eval_waiting_avatar_done = True
            return
        self._eval_glide_remaining = self._eval_glide_policy_steps
        self._eval_glide_active = True
        self._eval_target_cube = target_cube

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

    def _find_closest_table_cube(self):
        best_idx, _ = self._closest_table_cube_candidate()
        return best_idx

    def _closest_table_cube_candidate(self):
        """Return the index of the cube closest (in xy) to the right-arm EE
        that's still resting on the table and inside the avatar-reachable
        interrupt zone.  Restricting the eval target to this zone keeps the
        avatar on the +Y side instead of sliding into the table to reach a
        robot-side cube.  Returns (None, inf) if no cube qualifies.
        """
        arm = self.robot.get_arm("right")
        ee_pos = np.array(arm.get_ee_pose()[:3], dtype=float)
        big_x, big_y = self.big_bin_pose.p[:2]
        x_half, y_half = self._big_bin_xy_half
        rim_z = self._big_bin_rim_z

        best_idx = None
        best_d = float("inf")
        INT_X_LO, INT_X_HI = self.INTERRUPT_SPAWN_X_RANGE
        INT_Y_LO, INT_Y_HI = self.INTERRUPT_SPAWN_Y_RANGE
        for i, cube in enumerate(self.objects):
            p = self._dump_object_center(cube)
            in_bin = (abs(p[0] - big_x) < x_half and
                      abs(p[1] - big_y) < y_half and
                      p[2] < rim_z + self.SUCCESS_RIM_MARGIN)
            if in_bin:
                continue
            # On-table z window: cube resting on top of the table.
            if (
                p[2] < self.TABLE_TOP_Z - self.EVAL_TABLE_Z_LOWER_MARGIN or
                p[2] > self.TABLE_TOP_Z + self.EVAL_TABLE_Z_UPPER_MARGIN
            ):
                continue
            if (
                not bool(self.config.get("interrupt_use_base_cube_layout", False))
                and not (INT_X_LO <= p[0] <= INT_X_HI and
                         INT_Y_LO <= p[1] <= INT_Y_HI)
            ):
                continue
            d = float(np.linalg.norm(p[:2] - ee_pos[:2]))
            if d < best_d:
                best_d = d
                best_idx = i
        return best_idx, best_d

    # ------------------------------------------------------------------
    # Robot retreat
    # ------------------------------------------------------------------

    def _retreat_safe(self, arm_tag: str):
        """Pull the EE toward the robot base (−Y from cube zone) and lift
        — clears the avatar's forward-lean cone before the avatar's
        attach frame fires.
        """
        from ..grasp import tcp_to_link_pose

        arm = self.robot.get_arm(arm_tag)
        arm_base = np.array(arm.origin_pose.p, dtype=float)
        current_tcp = np.array(arm.get_ee_pose()[:3], dtype=float)
        to_base_xy = arm_base[:2] - current_tcp[:2]
        norm = float(np.linalg.norm(to_base_xy))
        if norm < 1e-6:
            to_base_xy = np.array([0.0, -1.0], dtype=float)
        else:
            to_base_xy = to_base_xy / norm
        retreat_pos = current_tcp.copy()
        retreat_pos[:2] += to_base_xy * self.RETREAT_BACK_DISTANCE
        retreat_pos[2] += self.RETREAT_UP_DISTANCE
        retreat_link = tcp_to_link_pose(self._top_down_tcp(retreat_pos), arm.tcp_offset)
        self._move_seeded(retreat_link.to_pose7(), arm_tag)

    def _avatar_obstacle_points(self) -> np.ndarray | None:
        if self.avatar_collider is None:
            return None
        pts = []
        for _name, pa, pb, r in self.avatar_collider.current_capsules():
            pa = np.asarray(pa, dtype=np.float64)
            pb = np.asarray(pb, dtype=np.float64)
            seg = pb - pa
            length = float(np.linalg.norm(seg))
            r_inflated = float(r) * self._AVATAR_INFLATE_FACTOR
            if length < 1e-6:
                pts.append(pa)
                continue
            axis = seg / length
            tmp = (
                np.array([0.0, 0.0, 1.0])
                if abs(axis[2]) < 0.9
                else np.array([1.0, 0.0, 0.0])
            )
            u = np.cross(axis, tmp)
            u = u / (np.linalg.norm(u) + 1e-12)
            v = np.cross(axis, u)
            v = v / (np.linalg.norm(v) + 1e-12)
            n_axis = max(2, int(length / 0.04) + 1)
            for t in np.linspace(0.0, 1.0, n_axis):
                center = pa + t * seg
                pts.append(center)
                for theta in np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False):
                    pts.append(center + r_inflated * (np.cos(theta) * u + np.sin(theta) * v))
        return np.asarray(pts, dtype=np.float64) if pts else None

    def _refresh_avatar_planner_obstacles(self, arm_tag: str) -> None:
        pts = self._avatar_obstacle_points()
        if pts is None or pts.size == 0:
            return
        arm = self.robot.get_arm(arm_tag)
        try:
            arm.planner.update_obstacles(pts, resolution=self._AVATAR_OBSTACLE_RES)
        except Exception as e:
            print(f"[dump_bin_interrupt] obstacle update failed: {e}")

    def _move_seeded(self, link_pose7, arm_tag):
        self._refresh_avatar_planner_obstacles(arm_tag)
        return super()._move_seeded(link_pose7, arm_tag)

    def _move_screw(self, target_pos, arm_tag):
        self._refresh_avatar_planner_obstacles(arm_tag)
        return super()._move_screw(target_pos, arm_tag)

    def _wait_until_avatar_attach(self):
        motion = self.avatar.motion_modules.get(self.INSPECT_MOTION_NAME)
        if motion is None:
            return
        while (
            not self.avatar.spare()
            and int(getattr(motion, "at_frame", 0)) < self._attach_frame_scaled
        ):
            self.step_sim()

    def _return_avatar_home_after_inspect(self):
        # The avatar now freezes on the inspect motion's last frame (pose +
        # position) instead of sliding back to avatar_init_pos.  Hold in place
        # for the same settle window so the released cube comes to rest while
        # the human stands where the motion ended.
        if self.avatar is None:
            return
        for _ in range(max(1, int(self.POST_AVATAR_RETURN_SETTLE_STEPS))):
            self.step_sim()

    # ------------------------------------------------------------------
    # Rollout
    # ------------------------------------------------------------------

    def play_once(self) -> bool:
        from ..grasp import tcp_to_link_pose

        arm_tag = "right"
        self._boost_finger_pd(arm_tag)
        arm = self.robot.get_arm(arm_tag)
        tcp_offset = arm.tcp_offset

        # Settle table objects.
        for _ in range(self.TABLE_SETTLE_STEPS):
            self.step_sim()

        if self.avatar is None:
            return super().play_once()

        interrupted_idx = self._interrupted_idx
        interrupt_ordinal = int(getattr(self, "_interrupt_pick_ordinal", 0))
        int_cube = self.objects[interrupted_idx]

        transport_z = self.TABLE_TOP_Z + self.TRANSPORT_Z_ABOVE_TABLE

        # Build a normal pick sequence with the interrupted cube placed at
        # the sampled ordinal.  This randomizes whether the human interrupts
        # the robot's 1st, 2nd, or 3rd attempted pick while preserving the
        # avatar-reachable interrupted cube identity chosen at reset.
        other_indices = [i for i in range(self.num_objects) if i != interrupted_idx]
        pick_order = (
            other_indices[:interrupt_ordinal] +
            [interrupted_idx] +
            other_indices[interrupt_ordinal:]
        )
        self._interrupt_pick_order = list(pick_order)

        # ---- Phase 1: pick cubes before the interrupt ordinal ----
        self.plan_success = True
        for idx in pick_order[:interrupt_ordinal]:
            try:
                ok = self._pick_and_drop_cube(self.objects[idx], arm_tag)
            except Exception:
                ok = False
            if not ok:
                self.open_gripper(arm_tag)
                for _ in range(self.FAILURE_RECOVERY_STEPS):
                    self.step_sim()

        # ---- Phase 2: robot shows intent on the interrupted cube ----
        int_pos = self._dump_object_center(int_cube).astype(float)
        self.open_gripper(arm_tag)
        above_pos = np.array([int_pos[0], int_pos[1], transport_z])
        above_link = tcp_to_link_pose(self._top_down_tcp(above_pos), tcp_offset)
        self._move_seeded(above_link.to_pose7(), arm_tag)

        # ---- Phase 3: avatar plays Inspect with attach ----
        # Non-blocking — frames will tick during subsequent step_sim()s.
        self._play_inspect_motion(
            attach_obj=getattr(int_cube, "entity", int_cube),
            hand_id=self.AVATAR_HAND_ID,
            attach_frame=self._attach_frame_scaled,
            detach_frame=self._detach_frame_scaled,
            return_to_idle_after=self.INSPECT_RETURN_TO_IDLE_FRAMES,
        )

        # ---- Phase 4: robot retreats so the avatar can grab the cube ----
        self._retreat_safe(arm_tag)
        # Keep stepping until the avatar reaches the attach frame so the
        # cube is visibly held before the robot moves on.
        self._wait_until_avatar_attach()
        for _ in range(self.POST_ATTACH_WAIT_MARGIN_STEPS):
            self.step_sim()

        # ---- Phase 5: wait for avatar to release the interrupted cube ----
        # Keep the robot retreated until the human has finished the inspection.
        # Earlier versions tried to pick remaining cubes while the avatar held
        # the interrupted one; that made the expert full-state path plan through
        # the avatar on seeds where the remaining cube was still near the human.
        if not self.avatar.spare():
            while not self.avatar.spare():
                self.step_sim()
        self._return_avatar_home_after_inspect()
        # Give the released cube a moment to settle on the table.
        for _ in range(self.POST_RELEASE_SETTLE_STEPS):
            self.step_sim()

        # ---- Phase 6: pick remaining non-interrupted cubes ----
        for idx in pick_order[interrupt_ordinal + 1:]:
            cube = self.objects[idx]
            try:
                ok = self._pick_and_drop_cube(cube, arm_tag)
            except Exception:
                ok = False
            if not ok:
                self.open_gripper(arm_tag)
                for _ in range(self.FAILURE_RECOVERY_STEPS):
                    self.step_sim()

        # ---- Phase 7: pick the released cube using its LATEST pose ----
        # _pick_and_drop_cube reads the live object position inside, so it always
        # uses the current world position (avatar may have moved it).
        try:
            ok = self._pick_and_drop_cube(int_cube, arm_tag)
        except Exception:
            ok = False
        if not ok:
            self.open_gripper(arm_tag)
            for _ in range(self.FAILURE_RECOVERY_STEPS):
                self.step_sim()

        self._ensure_final_gripper_z(arm_tag)
        return True

    def evaluate(self) -> dict:
        out = super().evaluate()
        if hasattr(self, "_interrupted_idx"):
            out["interrupted_object_index"] = int(self._interrupted_idx)
        if hasattr(self, "_interrupt_pick_ordinal"):
            out["interrupt_pick_ordinal"] = int(self._interrupt_pick_ordinal) + 1
        if hasattr(self, "_interrupt_pick_order"):
            out["robot_pick_order"] = [int(i) for i in self._interrupt_pick_order]
        if bool(self.config.get("eval_mode", False)):
            out["eval_mode"] = True
            out["eval_trigger_step"] = int(getattr(self, "_eval_trigger_step", -1))
            out["eval_policy_step_count"] = int(getattr(
                self, "_policy_step_count", 0,
            ))
            out["eval_interrupt_fired"] = bool(getattr(
                self, "_eval_interrupt_fired", False,
            ))
            out["eval_glide_active"] = bool(getattr(
                self, "_eval_glide_active", False,
            ))
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
            }
        return out

    # ------------------------------------------------------------------
    # Eval — require ALL cubes in the bin
    # ------------------------------------------------------------------

    def check_success(self) -> bool:
        if self.avatar_collision_summary().get("any_collision"):
            return False
        if bool(self.config.get("eval_mode", False)) and (
            getattr(self, "_eval_retreat_active", False)
            or getattr(self, "_eval_glide_active", False)
            or getattr(self, "_eval_waiting_avatar_done", False)
            or getattr(self, "_eval_return_active", False)
            or getattr(self, "_eval_pending_interrupt_idx", None) is not None
        ):
            return False

        big_x, big_y = self.big_bin_pose.p[:2]
        rim_z = self._big_bin_rim_z
        x_half, y_half = self._big_bin_xy_half

        threshold = self.SUCCESS_THRESHOLD or self.num_objects

        in_count = 0
        for i, cube in enumerate(self.objects):
            p = self._dump_object_center(cube)
            in_xy = abs(p[0] - big_x) < x_half and abs(p[1] - big_y) < y_half
            below_r = p[2] < rim_z + self.SUCCESS_RIM_MARGIN
            above_f = p[2] > self.SUCCESS_FLOOR_Z_MIN
            inside = in_xy and below_r and above_f
            if inside:
                in_count += 1

        return in_count >= threshold
