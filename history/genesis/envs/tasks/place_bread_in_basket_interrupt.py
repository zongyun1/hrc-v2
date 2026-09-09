"""Place bread in basket (interrupt): same scene as
``place_bread_in_basket`` but a human avatar inspects one of the two
breads mid-task.

Flow per episode:
  1. Randomly choose whether the interrupt happens on the robot's 1st
     or 2nd bread pick.
  2. Robot completes any earlier bread picks.
  3. Robot approaches above the *interrupted* bread (above-pose, no commit).
  4. Avatar starts the ``Inspect`` motion (categorize_interrupt's
     verbatim animation) with ``attach_obj`` = the interrupted bread.
     The bread is attached to the avatar's right hand at frame ATTACH
     and detached at DETACH — the avatar lifts it, holds it, and sets it
     back on the table.
  5. Robot retreats to a safe pose (toward the robot base, away from
     the avatar's lean cone).
  6. Robot handles any later non-interrupted bread picks when present.
  7. Robot waits for ``avatar.spare()``.
  8. Robot reads the released bread's LATEST pose (after the avatar
     puts it back) and places it via the same primitive.

Geometry pipeline (mirrors ``dump_bin_interrupt``):
  * Pre-bake: dry-run the Inspect motion at ``avatar_init_pos`` and
    measure the avatar's right-palm position at ATTACH frame.
  * Constraint solve: avatar shift = bread.pos − palm_at_attach.
    After shifting, the avatar palm meets the bread exactly at attach.
  * Both breads spawn in the same avatar-reachable subregion so
    placement does not reveal which one the human will inspect.

Eval: success = both breads in basket (parent's ``check_success``).

Robot motion is pure physics (no kinematic teleport/attach).  Avatar
attaches the bread to its hand via ``play_animation(attach_obj=...)``,
the sanctioned avatar mechanism.
"""

import numpy as np
import transforms3d as t3d

from ..avatar.inspect_motion_mixin import InspectMotionMixin, STANDARD_INSPECT_POOL
from ..grasp import tcp_to_link_pose
from ..utils import Pose, to_numpy
from ..task_bases.place_bread_in_basket import (
    PlaceBreadInBasket, _MIN_SEP_M, _REGION_X, _REGION_Y,
    _SAMPLE_FALLBACK_MARGIN,
)


_INTERRUPT_SAMPLE_ATTEMPTS = 80


class PlaceBreadInBasketInterrupt(InspectMotionMixin, PlaceBreadInBasket):
    """Robot picks two pieces of bread; a human avatar inspects one of
    them mid-task — robot must wait, then continue."""

    INSTRUCTION = "put both breads into the basket; if the human inspects one bread midway through, wait and then continue"
    use_avatar = True

    def __init__(self, config: dict = None):
        # Avatar collision tracking is opt-in per BaseTask; force it on
        # so `check_success` can fail any episode where the robot bumps
        # the avatar.  Caller can still pass `track_avatar_collision`
        # explicitly to override.
        cfg = dict(config or {})
        cfg.setdefault("track_avatar_collision", True)
        super().__init__(cfg)

    # ------------------------------------------------------------------
    # Avatar configuration (mirrors dump_bin_interrupt)
    # ------------------------------------------------------------------

    # Avatar at +Y side of the table, facing -Y (toward the table).
    # Maps mesh-+X (avatar forward) to world-(-Y).
    AVATAR_BASE_ROT = np.array(
        [[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=np.float64,
    )
    avatar_init_pos = np.array([0.0, 0.55, -0.18])
    avatar_init_rot = AVATAR_BASE_ROT

    # categorize_interrupt's Inspect motion + frame markers.  10× slow-
    # down reads as a deliberate inspection in video.
    INSPECT_MOTION_NAME   = "Inspect1"
    INSPECT_ATTACH_FRAME  = 34       # original frame index
    INSPECT_DETACH_FRAME  = 187      # original frame index
    INSPECT_POOL = STANDARD_INSPECT_POOL
    AVATAR_MOTION_SLOW    = 10.0     # avatar.frame_ratio
    AVATAR_HAND_ID        = 1        # 1 = right hand
    AVATAR_EXTRA_Z        = 0.04     # palm-above-bread margin so attach
                                     # doesn't sink into the bread
    DRYRUN_RESET_SETTLE_STEPS = 20
    CALIBRATION_SETTLE_STEPS = 30
    RETREAT_Y_FROM_BASE = 0.25
    RETREAT_Z_ABOVE_TABLE = 0.30
    INTENT_Z_ABOVE_TABLE = 0.18
    # None ⇒ avatar freezes on the inspect motion's last frame (pose + position).
    INSPECT_RETURN_TO_IDLE_FRAMES = None
    PRE_ATTACH_WAIT_MARGIN_STEPS = 100
    POST_RELEASE_SETTLE_STEPS = 120

    # ------------------------------------------------------------------
    # Shared bread spawn region near the +Y table edge.  Both breads are
    # sampled here so the robot cannot infer the interrupt target from
    # which bread is uniquely close enough for the avatar to inspect.
    # ------------------------------------------------------------------
    INT_X_LO, INT_X_HI = -0.10, 0.20
    INT_Y_LO, INT_Y_HI = -0.20, 0.00

    # ------------------------------------------------------------------
    # Eval-mode interrupt schedule (config["eval_mode"]=True).  Scripted
    # collect.py / play_once flow ignores these; the eval loop in
    # take_action triggers the interrupt at a random policy-step T in
    # [EVAL_TRIGGER_STEP_MIN, EVAL_TRIGGER_STEP_MAX] and grabs the bread
    # currently closest to the gripper (still on the table).
    # ------------------------------------------------------------------
    EVAL_TRIGGER_STEP_MIN = 100
    EVAL_TRIGGER_STEP_MAX = 200
    EVAL_GLIDE_POLICY_STEPS = 20
    EVAL_TABLE_Z_LOWER_MARGIN = 0.05
    EVAL_TABLE_Z_UPPER_MARGIN = 0.30
    EVAL_BASKET_Z_MARGIN = 0.02
    EVAL_FORCE_TRIGGER_DIST_XY = 0.11
    PACE_TRIGGER_PROGRESS = 0.82
    PACE_TRIGGER_DIST_XY = 0.14
    PACE_PROGRESS_START_DIST_XY = 0.42
    PACE_MIN_POLICY_STEPS = 20

    # ------------------------------------------------------------------
    # Sampling override — random interrupted pick ordinal (0 or 1), with
    # both breads sampled from the same avatar-reachable subregion.  Bread
    # ``self.breads[i]`` corresponds to ``positions[i + 1]`` (parent unpacks
    # as basket, bread_a, bread_b).
    # ------------------------------------------------------------------

    def _scripted_pick_order_from_positions(self, basket_xy, bread_xys):
        basket = np.asarray(basket_xy, dtype=np.float64)
        bread_xys = [np.asarray(xy, dtype=np.float64) for xy in bread_xys]
        return sorted(
            range(len(bread_xys)),
            key=lambda i: float(np.linalg.norm(bread_xys[i] - basket)),
        )

    def _choose_interrupt_pick_ordinal(self, rng) -> int:
        n_breads = max(1, int(getattr(self, "_num_breads", 1)))
        if "interrupt_pick_ordinal" in self.config:
            return int(self.config["interrupt_pick_ordinal"]) % n_breads
        if self.config.get("random_interrupt_target", True):
            return int(rng.randint(0, n_breads))
        return 0

    def _sample_positions(self, rng):
        # Randomly pick which robot pick ordinal will be interrupted.  Both
        # breads use the same reachable band; after sampling, map the ordinal
        # through the scripted pick order to choose the actual interrupted idx.
        interrupted_pick_ordinal = self._choose_interrupt_pick_ordinal(rng)
        self._interrupted_pick_ordinal = interrupted_pick_ordinal

        for _outer in range(_INTERRUPT_SAMPLE_ATTEMPTS):
            basket = (
                float(rng.uniform(*_REGION_X)),
                float(rng.uniform(*_REGION_Y)),
            )
            bread_xys = [
                (
                    float(rng.uniform(self.INT_X_LO, self.INT_X_HI)),
                    float(rng.uniform(self.INT_Y_LO, self.INT_Y_HI)),
                )
                for _ in range(max(1, int(getattr(self, "_num_breads", 1))))
            ]
            positions = [basket] + bread_xys
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
                pick_order = self._scripted_pick_order_from_positions(
                    basket, bread_xys,
                )
                self._interrupted_idx = int(pick_order[interrupted_pick_ordinal])
                return positions
        # Fall-through: deterministic spread.
        basket = (
            float(_REGION_X[0] + _SAMPLE_FALLBACK_MARGIN),
            float(_REGION_Y[0] + _SAMPLE_FALLBACK_MARGIN),
        )
        bread_a = (
            float(self.INT_X_LO + _SAMPLE_FALLBACK_MARGIN),
            float(self.INT_Y_LO + _SAMPLE_FALLBACK_MARGIN),
        )
        bread_xys = [bread_a]
        if max(1, int(getattr(self, "_num_breads", 1))) > 1:
            bread_xys.append((
                float(self.INT_X_HI - _SAMPLE_FALLBACK_MARGIN),
                float(self.INT_Y_HI - _SAMPLE_FALLBACK_MARGIN),
            ))
        positions = [basket] + bread_xys
        pick_order = self._scripted_pick_order_from_positions(
            positions[0], bread_xys,
        )
        self._interrupted_idx = int(pick_order[interrupted_pick_ordinal])
        return positions

    # ------------------------------------------------------------------
    # Reset + avatar calibration
    # ------------------------------------------------------------------

    def reset(self, seed: int = 0):
        obs = super().reset(seed=seed)
        if self.avatar is None:
            return obs
        self.avatar.frame_ratio = self.AVATAR_MOTION_SLOW
        if bool(self.config.get("eval_mode", False)):
            # Eval mode: leave the avatar idle; the take_action state
            # machine fires the interrupt at a random step on whichever
            # bread is closest to the gripper at trigger time.
            self._init_eval_interrupt_state()
        else:
            # Scripted mode: pre-shift the avatar so its palm meets the
            # interrupted bread exactly at the (scaled) attach frame.
            self._calibrate_avatar_for_inspect()
        return obs

    @property
    def _attach_frame_scaled(self) -> int:
        return int(round(self.INSPECT_ATTACH_FRAME * self.AVATAR_MOTION_SLOW))

    @property
    def _detach_frame_scaled(self) -> int:
        return int(round(self.INSPECT_DETACH_FRAME * self.AVATAR_MOTION_SLOW))

    def _dryrun_capture_palm(self):
        """Play the Inspect motion at the avatar's home pose, capture
        palm position at the (scaled) attach frame, then teleport
        avatar back to home.

        Cached on the class — the palm trajectory depends only on the
        motion name + frame_ratio + avatar init pose, not on bread/
        basket positions.  After seed 0 every subsequent reset re-uses
        the cached value, saving ~3 min/seed of dry-run sim."""
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
        # there so reset does not play a full visible inspect pass before
        # the actual interruption.
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
        # Clear avatar-collision flags accumulated during the dry run.
        self.avatar_collided = False
        self.avatar_collision_log = []
        # Cache for the rest of the process — same motion + frame_ratio
        # always lands the palm at the same world point given the same
        # avatar_init_pos.
        if palm_default is not None:
            cache = getattr(type(self), "_palm_default_cache", None)
            if cache is None:
                cache = {}
                type(self)._palm_default_cache = cache
            cache[cache_key] = palm_default.copy()
        return palm_default

    def _calibrate_avatar_for_inspect(self):
        """Dry-run + shift the avatar so its palm meets the interrupted
        bread exactly at the attach frame."""
        bread = self.breads[self._interrupted_idx]
        target_palm = to_numpy(bread.entity.get_pos()).ravel()[:3].astype(float)

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
    # Robot retreat (clear the avatar's lean cone so the avatar can
    # reach in).
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
        # Top-down TCP rotation, same convention as parent.
        R_tcp_down = np.array(
            [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]],
            dtype=np.float64,
        )
        q_tcp_down = t3d.quaternions.mat2quat(R_tcp_down)
        retreat_link = tcp_to_link_pose(
            Pose(retreat_p, q_tcp_down), tcp_offset,
        )
        self.move_and_execute(retreat_link.to_pose7(), arm_tag)

    def _scripted_pick_order(self):
        basket_xy = to_numpy(self.basket.entity.get_pos()).ravel()[:2]
        return sorted(
            range(len(self.breads)),
            key=lambda i: float(np.linalg.norm(
                to_numpy(self.breads[i].entity.get_pos()).ravel()[:2]
                - basket_xy,
            )),
        )

    def _wait_until_avatar_attach(self):
        motion = self.avatar.motion_modules.get(self.INSPECT_MOTION_NAME)
        if motion is None:
            return
        while (
            not self.avatar.spare()
            and int(getattr(motion, "at_frame", 0)) < self._attach_frame_scaled
        ):
            self.step_sim()

    # ------------------------------------------------------------------
    # Rollout
    # ------------------------------------------------------------------

    def take_action(self, action, action_type: str = "qpos"):
        """Eval mode adds a per-step hook that drives the interrupt
        state machine before delegating to the standard
        ``BaseTask.take_action`` for physics + observation update.
        Returns the observation as the baseline expects."""
        if (self.avatar is not None and
                bool(self.config.get("eval_mode", False))):
            self._eval_step_avatar()
        return super().take_action(action, action_type=action_type)

    def play_once(self) -> bool:
        with self.suppress_recording():  # pre-task settle, excluded from recordings
            for _ in range(self.INITIAL_SETTLE_STEPS):
                self.step_sim()

        if not self.breads or self.basket is None:
            return False
        self._refresh_basket_geom()
        arm_tag = "right"

        if self.avatar is None:
            return super().play_once()

        pick_order = self._scripted_pick_order()
        interrupted_pick_ordinal = int(getattr(
            self, "_interrupted_pick_ordinal", 0,
        )) % len(pick_order)
        interrupted_idx = int(pick_order[interrupted_pick_ordinal])
        self._interrupted_idx = interrupted_idx
        int_bread = self.breads[interrupted_idx]
        int_label = chr(ord("A") + interrupted_idx)

        arm = self.robot.get_arm(arm_tag)
        tcp_offset = arm.tcp_offset
        R_tcp_down = np.array(
            [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]],
            dtype=np.float64,
        )
        q_tcp_down = t3d.quaternions.mat2quat(R_tcp_down)
        intent_z = self.TABLE_TOP_Z + self.INTENT_Z_ABOVE_TABLE

        # ---- Phase 1: complete picks before the interrupted ordinal ---
        any_success = False
        for idx in pick_order[:interrupted_pick_ordinal]:
            label = chr(ord("A") + idx)
            try:
                ok = self._pick_and_place_bread(
                    self.breads[idx], arm_tag, label,
                )
            except Exception:
                ok = False
            if not ok:
                return False
            any_success = True

        # ---- Phase 2: robot shows intent on the interrupted bread ----
        self.open_gripper(arm_tag)
        int_pos = to_numpy(int_bread.entity.get_pos()).ravel()[:3]
        above_p = np.array([int_pos[0], int_pos[1], intent_z])
        above_link = tcp_to_link_pose(
            Pose(above_p, q_tcp_down), tcp_offset,
        )
        self.move_and_execute(above_link.to_pose7(), arm_tag)

        # ---- Phase 3: avatar plays Inspect with attach ----
        self._play_inspect_motion(
            attach_obj=int_bread.entity,
            hand_id=self.AVATAR_HAND_ID,
            attach_frame=self._attach_frame_scaled,
            detach_frame=self._detach_frame_scaled,
            return_to_idle_after=self.INSPECT_RETURN_TO_IDLE_FRAMES,
        )

        # ---- Phase 4: robot retreats so the avatar can grab the bread --
        self._retreat_safe(arm_tag)
        # Step until the avatar has reached the attach frame so the
        # bread is visibly held before any later robot pick starts.
        self._wait_until_avatar_attach()
        for _ in range(self.PRE_ATTACH_WAIT_MARGIN_STEPS):
            self.step_sim()

        # ---- Phase 5: complete later non-interrupted picks -----------
        for idx in pick_order[interrupted_pick_ordinal + 1:]:
            label = chr(ord("A") + idx)
            try:
                ok = self._pick_and_place_bread(
                    self.breads[idx], arm_tag, label,
                )
            except Exception:
                ok = False
            if not ok:
                return False
            any_success = True

        # ---- Phase 6: wait for avatar to release the interrupted bread --
        if not self.avatar.spare():
            while not self.avatar.spare():
                self.step_sim()
        # Give the released bread a moment to settle on the table.
        for _ in range(self.POST_RELEASE_SETTLE_STEPS):
            self.step_sim()

        # ---- Phase 7: pick + place the released bread (latest pose) ----
        try:
            ok = self._pick_and_place_bread(int_bread, arm_tag, int_label)
        except Exception:
            ok = False
        any_success = any_success or ok

        return any_success

    # ------------------------------------------------------------------
    # Eval-mode interrupt state machine (config["eval_mode"]=True).
    # Mirrors `dump_bin_interrupt._eval_step_avatar` — the policy
    # baseline calls `take_action(action) -> obs` in a loop, and this
    # hook fires the interrupt at a random step.  The robot's policy is
    # the BASELINE (not our scripted play_once); avatar logic runs
    # alongside the baseline's actions.
    # ------------------------------------------------------------------

    def _init_eval_interrupt_state(self):
        """Capture ``palm_default`` once (dry-run) and arm a state
        machine that fires the interrupt at a random policy step T in
        ``[EVAL_TRIGGER_STEP_MIN, EVAL_TRIGGER_STEP_MAX]``."""
        self._eval_palm_default = self._dryrun_capture_palm()
        if self._eval_palm_default is None:
            self._eval_interrupt_fired = True
        else:
            lo = int(self.config.get("eval_trigger_step_min", self.EVAL_TRIGGER_STEP_MIN))
            hi = int(self.config.get("eval_trigger_step_max", self.EVAL_TRIGGER_STEP_MAX))
            if hi < lo:
                hi = lo
            self._eval_trigger_step = int(np.random.randint(lo, hi + 1))
            self._eval_interrupt_fired = False
        self._eval_interrupt_trigger_mode = str(
            self.config.get("eval_interrupt_trigger_mode", "random")
        ).lower()
        self._eval_glide_active = False
        self._eval_glide_remaining = 0
        self._eval_glide_start_pos = None
        self._eval_glide_target_pos = None
        self._eval_target_bread = None
        self._policy_step_count = 0
        self._pace_target_idx = None
        self._pace_initial_dist_xy = None
        self._pace_best_progress = 0.0
        self._pace_trigger_reason = None
        self._pace_trigger_step = None
        self._pace_trigger_progress = None
        self._pace_trigger_dist_xy = None

    def _eval_step_avatar(self):
        """Per-policy-step state machine:
          1. If a glide is active, advance one step (linear-interpolated
             avatar pose).  When glide finishes, fire the Inspect
             animation with the target bread attached.
          2. Else if the trigger step has been reached and the interrupt
             hasn't fired, find the closest on-table bread, validate the
             required avatar shift, start the glide.  If no bread
             qualifies, latch the fired flag so we don't re-try.
        """
        self._policy_step_count += 1

        # ---- Glide in progress ----------------------------------------
        if self._eval_glide_active:
            self._eval_glide_remaining -= 1
            glide_steps = max(1, int(self.config.get(
                "eval_glide_policy_steps", self.EVAL_GLIDE_POLICY_STEPS,
            )))
            done = glide_steps - self._eval_glide_remaining
            progress = min(1.0, max(0.0, done / float(glide_steps)))
            cur = ((1.0 - progress) * self._eval_glide_start_pos +
                   progress * self._eval_glide_target_pos)
            rot = np.asarray(self.avatar_init_rot, dtype=np.float64).copy()
            self.avatar.reset(cur.copy(), rot)
            if self._eval_glide_remaining <= 0:
                self._eval_glide_active = False
                target_bread = self._eval_target_bread
                bread_pos = to_numpy(target_bread.entity.get_pos()).ravel()[:3]
                # Re-check the bread hasn't fallen / been picked during
                # the glide.  If it's no longer on the table, abort.
                if bread_pos[2] < self.TABLE_TOP_Z - self.EVAL_TABLE_Z_LOWER_MARGIN:
                    return
                self._play_inspect_motion(
                    attach_obj=target_bread.entity,
                    hand_id=self.AVATAR_HAND_ID,
                    attach_frame=self._attach_frame_scaled,
                    detach_frame=self._detach_frame_scaled,
                    return_to_idle_after=self.INSPECT_RETURN_TO_IDLE_FRAMES,
                )
            return

        # ---- Should we trigger? ---------------------------------------
        if self._eval_interrupt_fired:
            return
        target_idx, target_dist = self._closest_table_bread_candidate()
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
        # Latch immediately so we only try once even if no candidate found.
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

        target_bread = self.breads[target_idx]
        target_pos = to_numpy(target_bread.entity.get_pos()).ravel()[:3].astype(float)
        shift = target_pos - self._eval_palm_default
        shift[2] += self.AVATAR_EXTRA_Z
        target_avatar_pos = (
            np.asarray(self.avatar_init_pos, dtype=np.float64) + shift
        )

        self._eval_glide_start_pos = (
            np.asarray(self.avatar_init_pos, dtype=np.float64).copy()
        )
        self._eval_glide_target_pos = target_avatar_pos.copy()
        self._eval_glide_remaining = max(1, int(self.config.get(
            "eval_glide_policy_steps", self.EVAL_GLIDE_POLICY_STEPS,
        )))
        self._eval_glide_active = True
        self._eval_target_bread = target_bread

    def _pace_should_trigger(self, target_idx, target_dist):
        """Estimate completion of the robot's current approach-to-bread
        action from EE-to-bread xy distance and fire the human inspection
        when the action is nearly complete.

        This is a lightweight sim-side PACE analogue for policy eval: ACT
        remains unchanged, while the environment uses full-state progress
        to decide when the interrupt should happen.
        """
        if target_idx is None or not np.isfinite(target_dist):
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

    def _find_closest_table_bread(self):
        best_idx, _ = self._closest_table_bread_candidate()
        return best_idx

    def _closest_table_bread_candidate(self):
        """Return the index of the bread closest (in xy) to the right-
        arm EE that's still resting on the table (not in the basket,
        not on the floor).  Returns (None, inf) if none qualify."""
        arm = self.robot.get_arm("right")
        ee_pos = np.array(arm.get_ee_pose()[:3], dtype=float)
        basket_xy = to_numpy(self.basket.entity.get_pos()).ravel()[:2]

        best_idx = None
        best_d = float("inf")
        for i, bread in enumerate(self.breads):
            p = to_numpy(bread.entity.get_pos()).ravel()[:3]
            in_basket = (
                np.linalg.norm(p[:2] - basket_xy) < self.SUCCESS_DIST_XY_M
                and p[2] < self._basket_rim_z + self.EVAL_BASKET_Z_MARGIN
            )
            if in_basket:
                continue
            # On-table z window.
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
    # Success metric — both breads in basket AND no avatar collision
    # ------------------------------------------------------------------

    def check_success(self) -> bool:
        """Inherits parent's "both breads inside basket" + adds a hard
        fail on avatar collision.  Avatar collision tracking is forced
        on in __init__ so this metric is meaningful by default."""
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
            }
        return out
