"""Stack bowls (assist): cooperative variant of ``stack_bowls_three``.

Same scene + grasp pipeline as the parent ``StackBowlsThree``.  Bowl
0 stays put as the base; the robot picks bowl 1 and stacks it on bowl
0; the avatar picks bowl 2 and drops it on the stack.  In hard mode,
bowl 3 is an extra robot-side bowl, so the robot stacks two bowls while
the avatar stacks one.  Both sequences kick off concurrently — the
avatar's FABRIK pick-and-place motion runs as ``self.step_sim`` ticks
under the robot's planned trajectories.

The avatar uses the sanctioned ``attach_obj`` mechanism via
``AvatarController.pick_and_place`` (FABRIK shoulder→hand chain) —
``attach_obj`` is allowed for the avatar.  The robot
uses the parent's pure-physics rim grasp + lift + place primitive
(``_pick_and_stack_one``) — no kinematic snap of the held bowl.

Mirrors the structure of ``place_bread_in_basket_assist``: avatar
non-blocking ``pick_and_place`` issued first, robot serial pick under
the patched-``step_sim`` driving both forward.

Success: parent's loose footprint + higher-z check, plus a
no-robot-↔-avatar-collision gate.
"""

import numpy as np
import transforms3d as t3d

from ..avatar.eval_mode_mixin import EvalModeAvatarMixin
from ..utils import Pose, load_object, to_numpy
from ..task_bases.stack_bowls_three import (
    StackBowlsThree,
    _BOWL_ID, _BOWL_MODEL_ID, _NUM_BOWLS,
    _MIN_SEP_M, _REGION_X, _REGION_Y,
    _read_extents, _read_scale, _quat_mul,
)


class StackBowlsThreeAssist(EvalModeAvatarMixin, StackBowlsThree):
    """Avatar stacks one bowl while the robot stacks the rest."""

    INSTRUCTION = "stack one bowl on the base bowl while the human stacks another bowl on the base"
    use_avatar = True
    EASY_NUM_OBJECTS = _NUM_BOWLS
    HARD_NUM_OBJECTS = _NUM_BOWLS + 1

    # Avatar at +Y side of the table, facing -Y (toward the table).
    # Mesh-+X (avatar forward) maps to world-(-Y).  Same convention as
    # place_bread_in_basket_assist.
    AVATAR_BASE_ROT = np.array(
        [[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=np.float64,
    )
    avatar_init_pos = np.array([0.0, 0.55, -0.18])
    avatar_init_rot = AVATAR_BASE_ROT

    # Avatar-side bowl targets.  Avatar-only motion debugging tested
    # forward offsets [0, +0.05, +0.10, +0.15, +0.20] from the initial
    # seed-0 geometry; +0.10/+0.15 looked best, and +0.05..+0.20 were
    # usable.  Sample the avatar pick bowl and base/drop bowl in this
    # farther-forward band so the arm reaches naturally while the body
    # stays outside the table.
    AVATAR_X_LO, AVATAR_X_HI = -0.18, 0.18
    AVATAR_Y_LO, AVATAR_Y_HI = -0.10, 0.04
    ASSIST_SAMPLE_ATTEMPTS = 256

    # Avatar pick-and-place pacing.  Shorter than the original
    # bread-ass 320/400/320 recipe so the long FABRIK precompute does
    # not dominate each episode, but still smooth at one replay frame
    # per sim step.
    AVATAR_APPROACH_FRAMES = 80
    AVATAR_TRANSPORT_FRAMES = 120
    AVATAR_RETRACT_FRAMES = 80

    # How far above the bowl's mesh-origin the palm should aim at pick
    # time.  002_bowl mesh origin is at the bowl bottom; we want the
    # palm a bit above the rim so the FABRIK chain doesn't try to
    # plunge through the table.
    AVATAR_PICK_LIFT = 0.08

    # Drop bowl 2 above the (anticipated) stack top.  By the time the
    # avatar finishes its approach+transport, the robot should have
    # placed bowl 1 on bowl 0.  Conservative drop:
    # a couple bowl-heights above bowl 0's mesh origin (= bowl 0
    # bottom), with a 6-cm clearance on top so gravity does the
    # final settling.
    AVATAR_DROP_BOWLS_ABOVE = 2.0   # in units of bowl_height
    AVATAR_DROP_CLEARANCE_M = 0.06

    # The parent table is centered at y=-0.35 and spans up to y=+0.20.
    # Keep the avatar pelvis comfortably outside that edge; the old
    # bread-ass style shift could place the avatar at y≈0.10, visibly
    # inside the table.
    AVATAR_TABLE_Y_MAX = 0.20
    AVATAR_MIN_BODY_Y = AVATAR_TABLE_Y_MAX + 0.18
    AVATAR_MAX_BODY_Y = 0.62
    AVATAR_BODY_MARGIN = 0.32
    AVATAR_BODY_X_BOUNDS = (-0.22, 0.22)
    AVATAR_YAW_LIMIT_DEG = 30.0
    AVATAR_FINISH_WAIT_STEPS = 1500
    EVAL_TRIGGER_STEP_MIN = 20
    EVAL_TRIGGER_STEP_MAX = 60

    def __init__(self, config: dict = None):
        cfg = dict(config or {})
        # Avatar collision tracking is OFF by default — same finding as
        # bread-ass v3 (the avatar capsule overlaps the robot's transit
        # airspace).  Opt-in via ``cfg["track_avatar_collision"]=True``.
        cfg.setdefault("track_avatar_collision", False)
        super().__init__(cfg)

    # ------------------------------------------------------------------
    # Position sampling — bowl 0 (base/drop) and bowl 2 (avatar pick)
    # inside the avatar-natural subregion; all other non-base bowls are
    # robot picks sampled in the parent's region.
    # ------------------------------------------------------------------

    def _sample_assist_xy(self, rng):
        n_bowls = int(getattr(self, "_num_bowls", _NUM_BOWLS))
        if self.simplified_mode_enabled():
            return [(-0.08, -0.02), (0.14, -0.16), (0.10, -0.08)]
        for _ in range(self.ASSIST_SAMPLE_ATTEMPTS):
            pts = []
            ok = True
            for i in range(n_bowls):
                if i in (0, 2):
                    x = float(rng.uniform(self.AVATAR_X_LO, self.AVATAR_X_HI))
                    y = float(rng.uniform(self.AVATAR_Y_LO, self.AVATAR_Y_HI))
                else:
                    x = float(rng.uniform(*_REGION_X))
                    y = float(rng.uniform(*_REGION_Y))
                for px, py in pts:
                    if (x - px) ** 2 + (y - py) ** 2 < _MIN_SEP_M ** 2:
                        ok = False
                        break
                if not ok:
                    break
                pts.append((x, y))
            if ok and len(pts) == n_bowls:
                return pts
        # Fallback: deterministic spread inside the avatar subregion.
        fallback = [
            (self.AVATAR_X_LO + 0.04, self.AVATAR_Y_HI),
            (_REGION_X[1] - 0.05, _REGION_Y[0] + 0.05),
            (
                self.AVATAR_X_HI - 0.04,
                self.AVATAR_Y_LO,
            ),
            (_REGION_X[0] + 0.05, _REGION_Y[1] - 0.05),
        ]
        return fallback[:n_bowls]

    def load_actors(self):
        # Mirror the parent's load_actors but use the assist-aware
        # position sampler so bowl 2 lands inside the avatar's reach.
        upright_q = self.BOWL_UPRIGHT_QUAT
        table_top = self.TABLE_TOP_Z

        rng = np.random.RandomState(np.random.randint(0, 2 ** 31 - 1))
        positions = self._sample_assist_xy(rng)

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
                t3d.quaternions.mat2quat(t3d.euler.euler2mat(0, 0, yaw, "sxyz")),
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

        # Bowl 0 = base (stays put); bowl 2 = avatar pick; every other
        # non-base bowl is a robot pick.  In hard mode this gives the robot
        # two bowls while the avatar still handles one.
        self._stacked_indices = list(range(1, self._num_bowls))
        self._avatar_bowl_idx = 2
        self._robot_bowl_indices = [
            i for i in self._stacked_indices if i != self._avatar_bowl_idx
        ]
        self._robot_bowl_idx = self._robot_bowl_indices[0]

    def _start_avatar_assist(self) -> bool:
        if self.avatar is None or any(b is None for b in self.bowls):
            return False

        avatar_bowl = self.bowls[self._avatar_bowl_idx]

        # Pick: aim the palm just above the bowl rim.  Bowl mesh origin
        # = bowl bottom (after upright_q, mesh-y is world-z).
        bowl_origin = to_numpy(avatar_bowl.entity.get_pos()).ravel()[:3]
        pick_pos = bowl_origin + np.array([0.0, 0.0, self.AVATAR_PICK_LIFT])

        # Drop: above the base bowl xy at a height that clears the
        # robot's stack of bowl 1 on bowl 0.  Conservative — gravity
        # nests bowl 2 the rest of the way.
        base_xyz = self._bowl_xyz(0)
        drop_z = (
            self.TABLE_TOP_Z
            + self.AVATAR_DROP_BOWLS_ABOVE * self._bowl_height
            + self.AVATAR_DROP_CLEARANCE_M
        )
        place_pos = np.array([base_xyz[0], base_xyz[1], drop_z])

        self.avatar.pick_and_place(
            pick_pos=pick_pos,
            place_pos=place_pos,
            attach_obj=avatar_bowl.entity,
            hand_id=0,
            approach_frames=self.AVATAR_APPROACH_FRAMES,
            transport_frames=self.AVATAR_TRANSPORT_FRAMES,
            retract_frames=self.AVATAR_RETRACT_FRAMES,
            natural=True,
            body_margin=self.AVATAR_BODY_MARGIN,
            body_x_bounds=self.AVATAR_BODY_X_BOUNDS,
            body_y_bounds=(self.AVATAR_MIN_BODY_Y, self.AVATAR_MAX_BODY_Y),
            yaw_limit_deg=self.AVATAR_YAW_LIMIT_DEG,
        )
        self.avatar_collided = False
        self.avatar_collision_log = []
        return True

    # ------------------------------------------------------------------
    # Rollout — kick off avatar pick_and_place, then run robot serial
    # pick under the same step_sim ticks (which advance avatar in
    # lockstep).
    # ------------------------------------------------------------------

    def play_once(self) -> bool:
        with self.suppress_recording():  # pre-task settle, excluded from recordings
            for _ in range(self.INITIAL_SETTLE_STEPS):
                self.step_sim()
        if any(b is None for b in self.bowls):
            return False
        arm_tag = "right"

        if self.avatar is None:
            return super().play_once()

        self._start_avatar_assist()

        # ---- Robot stacks its assigned bowls (parent primitive) ----------
        # _pick_and_stack_one internally step_sim()s, advancing the
        # avatar motion in parallel — DO NOT block between them.
        base_xyz = self._bowl_xyz(0)
        # In the assist layout, the avatar is also dropping onto the
        # same base stack.  Aim the robot's release one bowl-height above
        # the parent first-bowl target so it does not place into the base
        # bowl cavity while the avatar stack motion is converging.
        ok_robot = True
        robot_indices = list(getattr(
            self, "_robot_bowl_indices", [self._robot_bowl_idx],
        ))
        for stack_i, robot_idx in enumerate(robot_indices):
            placed_so_far = [0] + robot_indices[:stack_i]
            top_z = max(self._bowl_xyz(i)[2] for i in placed_so_far)
            stack_top_z = top_z + 2.0 * self._bowl_height
            stack_position = (base_xyz[0], base_xyz[1], stack_top_z)
            try:
                ok_robot = self._pick_and_stack_one(robot_idx, stack_position) and ok_robot
            except Exception:
                ok_robot = False

        # ---- Wait for avatar to finish its retract -----------------------
        if not self.avatar.spare():
            tail = 0
            while not self.avatar.spare() and tail < self.AVATAR_FINISH_WAIT_STEPS:
                self.step_sim()
                tail += 1

        # Final settle so the avatar-dropped bowl comes to rest.
        for _ in range(self.FINAL_SETTLE_STEPS):
            self.step_sim()
        return ok_robot

    def _eval_at_reset(self) -> None:
        self._eval_assist_trigger_step = 0
        self._eval_assist_fired = True
        self._eval_assist_failed = False
        if not self._start_avatar_assist():
            self._eval_assist_failed = True

    def _eval_at_step(self, step_idx: int) -> None:
        if (getattr(self, "_eval_assist_fired", False)
                or getattr(self, "_eval_assist_failed", False)):
            return
        if step_idx < int(getattr(self, "_eval_assist_trigger_step", self.EVAL_TRIGGER_STEP_MIN)):
            return
        self._eval_assist_fired = True
        if not self._start_avatar_assist():
            self._eval_assist_failed = True

    # ------------------------------------------------------------------
    # Success — parent's geometric check.  Honour the standard
    # ``track_avatar_collision`` knob: when it is enabled (opt-in) any
    # robot↔avatar contact during the parallel motion fails the
    # episode regardless of the geometric verdict.
    # ------------------------------------------------------------------

    def check_success(self) -> bool:
        if (self.config.get("track_avatar_collision", False)
                and self.avatar_collided):
            return False
        return super().check_success()

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
            "assist_avatar_bowl_idx": int(getattr(self, "_avatar_bowl_idx", -1)),
            "assist_robot_bowl_idx": int(getattr(self, "_robot_bowl_idx", -1)),
            "assist_robot_bowl_indices": [
                int(i) for i in getattr(self, "_robot_bowl_indices", [])
            ],
        })
        return metrics
