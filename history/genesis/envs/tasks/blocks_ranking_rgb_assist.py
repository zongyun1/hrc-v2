"""Assist variant for ``blocks_ranking_rgb``.

The avatar places one randomly selected RGB cube into its target slot while
the robot places the remaining two cubes with the parent's pure-physics
top-down primitive.  The avatar uses the sanctioned hand ``attach_obj`` path
through ``AvatarController.pick_and_place``.
"""

import numpy as np

from ..avatar.eval_mode_mixin import EvalModeAvatarMixin
from ..utils import Pose, create_primitive, to_numpy
from ..task_bases.blocks_ranking_rgb import BlocksRankingRGB


class BlocksRankingRGBAssist(EvalModeAvatarMixin, BlocksRankingRGB):
    """Human places one block while the robot places the remaining blocks."""

    INSTRUCTION = (
        "place the red, green, and blue blocks in left-to-right order while "
        "the human helps place one block"
    )

    use_avatar = True

    # Avatar stands at the +Y side of the table and faces -Y.
    AVATAR_BASE_ROT = np.array(
        [[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=np.float64
    )
    avatar_init_pos = np.array([0.0, 0.55, -0.18])
    avatar_init_rot = AVATAR_BASE_ROT

    AVATAR_COLOR_CHOICES = ("red", "green", "blue")
    # Keep avatar visibly active while the robot performs its first pick.
    # The robot spends several hundred sim steps just reaching pre-grasp;
    # shorter avatar motions finish before robot-object contact and read
    # as sequential rather than cooperative.
    AVATAR_APPROACH_FRAMES = 260
    AVATAR_TRANSPORT_FRAMES = 360
    AVATAR_RETRACT_FRAMES = 180
    AVATAR_START_DELAY_STEPS = (100, 200)
    AVATAR_PICK_LIFT = 0.010
    AVATAR_PLACE_LIFT = 0.075
    AVATAR_NATURAL_BODY_MARGIN = 0.34
    AVATAR_NATURAL_YAW_LIMIT_DEG = 30.0
    AVATAR_NATURAL_BODY_Y_BOUNDS = (0.28, 0.62)
    AVATAR_NATURAL_SETTLE_STEPS = 20
    EVAL_TRIGGER_STEP_MIN = 100
    EVAL_TRIGGER_STEP_MAX = 200

    # Put the avatar-owned block on the near (+Y) side of the table so the
    # FABRIK reach stays plausible from the avatar's body pose.  Robot-owned
    # blocks keep the parent's spawn band.
    AVATAR_SPAWN_X = (-0.22, 0.22)
    AVATAR_SPAWN_Y = (-0.03, 0.16)
    ROBOT_SPAWN_X = (-0.24, 0.24)
    ROBOT_SPAWN_Y = (-0.38, -0.22)
    AVATAR_LAYOUT_SAMPLE_ATTEMPTS = 160

    def __init__(self, config: dict = None):
        cfg = dict(config or {})
        # Geometry overlap between the avatar capsule and the robot transit
        # corridor is possible for this tight tabletop scene; make analytic
        # collision tracking opt-in, as in other assist variants.
        cfg.setdefault("track_avatar_collision", False)
        super().__init__(cfg)

    def _sample_spawn_xys_for_avatar(self, avatar_color: str):
        colors = list(self.CUBE_COLORS.keys())
        xys = {}
        for _ in range(self.AVATAR_LAYOUT_SAMPLE_ATTEMPTS):
            candidate = {
                avatar_color: self._layout_xy([
                    np.random.uniform(*self.AVATAR_SPAWN_X),
                    np.random.uniform(*self.AVATAR_SPAWN_Y),
                ])
            }
            for color in colors:
                if color == avatar_color:
                    continue
                candidate[color] = self._layout_xy([
                    np.random.uniform(*self.ROBOT_SPAWN_X),
                    np.random.uniform(*self.ROBOT_SPAWN_Y),
                ])

            ok = True
            for i, c0 in enumerate(colors):
                for c1 in colors[i + 1:]:
                    if np.linalg.norm(candidate[c0] - candidate[c1]) < self.MIN_BLOCK_SPACING:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                return candidate
        raise RuntimeError("[blocks_rgb_assist] could not sample non-overlapping cubes")

    def load_actors(self):
        cube_z = self.TABLE_TOP_Z + self.CUBE_HALF + self.CUBE_Z_CLEARANCE
        self._avatar_color = str(np.random.choice(self.AVATAR_COLOR_CHOICES))
        self._robot_colors = tuple(
            c for c in ("red", "green", "blue") if c != self._avatar_color
        )
        self._avatar_start_delay_steps = int(
            np.random.randint(
                self.AVATAR_START_DELAY_STEPS[0],
                self.AVATAR_START_DELAY_STEPS[1] + 1,
            )
        )
        xys = self._sample_spawn_xys_for_avatar(self._avatar_color)

        self.blocks = {}
        for color_name in np.random.permutation(list(self.CUBE_COLORS.keys())):
            xy = xys[color_name]
            cube = create_primitive(
                self.scene,
                "box",
                Pose([float(xy[0]), float(xy[1]), cube_z]),
                size={"half_size": (self.CUBE_HALF,) * 3},
                color=self.CUBE_COLORS[color_name],
                is_static=False,
            )
            self.blocks[color_name] = cube

        self.target_slots = {
            color: self._target_slot_pos(
                color, self.TABLE_TOP_Z + self.CUBE_HALF
            )
            for color in self.CUBE_COLORS
        }

    def _block_center(self, color_name: str) -> np.ndarray:
        return to_numpy(self.blocks[color_name].get_pos()).ravel()[:3]

    def _start_avatar_pick_and_place(self, add_start_delay: bool = True) -> None:
        color_name = self._avatar_color
        cube = self.blocks[color_name]
        pick_pos = self._block_center(color_name).copy()
        pick_pos[2] += self.AVATAR_PICK_LIFT
        slot = self.target_slots[color_name]
        place_pos = np.array([
            slot[0],
            slot[1],
            self.TABLE_TOP_Z + self.CUBE_HALF + self.AVATAR_PLACE_LIFT,
        ], dtype=float)
        self.avatar.pick_and_place(
            pick_pos=pick_pos,
            place_pos=place_pos,
            attach_obj=cube,
            hand_id=None,
            approach_frames=self.AVATAR_APPROACH_FRAMES,
            transport_frames=self.AVATAR_TRANSPORT_FRAMES,
            retract_frames=self.AVATAR_RETRACT_FRAMES,
            natural=True,
            body_margin=self.AVATAR_NATURAL_BODY_MARGIN,
            body_y_bounds=self.AVATAR_NATURAL_BODY_Y_BOUNDS,
            yaw_limit_deg=self.AVATAR_NATURAL_YAW_LIMIT_DEG,
            settle_steps=self.AVATAR_NATURAL_SETTLE_STEPS,
        )
        self.avatar_collided = False
        self.avatar_collision_log = []
        if add_start_delay:
            self._delay_avatar_motion_start(self._avatar_start_delay_steps)

    def _delay_avatar_motion_start(self, delay_steps: int) -> None:
        if delay_steps <= 0:
            return
        mm = self.avatar.motion_modules.get("pick_and_place")
        if mm is None or not mm.frames:
            return
        hold = mm.frames[0]
        mm.frames = [hold] * int(delay_steps) + mm.frames
        mm.attach_frame += int(delay_steps)
        mm.detach_frame += int(delay_steps)

    def play_once(self) -> bool:
        arm_tag = "right"
        self._boost_finger_pd(arm_tag)

        with self.suppress_recording():  # pre-task settle, excluded from recordings
            for _ in range(self.TABLE_SETTLE_STEPS):
                self.step_sim()

        if self.avatar is None:
            return super().play_once()

        self._start_avatar_pick_and_place(add_start_delay=True)

        ok_all = True
        for color_name in self._robot_colors:
            try:
                ok = self._pick_and_place_block(color_name, arm_tag)
            except Exception:
                ok = False
            if not ok:
                ok_all = False
                self.open_gripper(arm_tag)
                for _ in range(self.FAILURE_RECOVERY_STEPS):
                    self.step_sim()

        if not self.avatar.spare():
            while not self.avatar.spare():
                self.step_sim()

        for _ in range(self.FINAL_SETTLE_STEPS):
            self.step_sim()

        return ok_all

    def _eval_at_reset(self) -> None:
        self._eval_assist_trigger_step = 0
        self._eval_assist_fired = True
        self._eval_assist_failed = False
        try:
            self._start_avatar_pick_and_place(add_start_delay=False)
        except Exception as exc:
            self._eval_assist_failed = True
            if self.config.get("debug_eval_avatar", False):
                print(f"[blocks_rgb_assist/eval] avatar assist failed: {exc}", flush=True)

    def _eval_at_step(self, step_idx: int) -> None:
        if getattr(self, "_eval_assist_fired", False):
            return
        trigger = int(getattr(self, "_eval_assist_trigger_step", self.EVAL_TRIGGER_STEP_MIN))
        if step_idx < trigger:
            return
        self._eval_assist_fired = True
        self._start_avatar_pick_and_place(add_start_delay=False)
