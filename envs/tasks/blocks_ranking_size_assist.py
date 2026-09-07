"""Assist variant for ``blocks_ranking_size``.

The avatar places one randomly selected size block into its ranked slot while
the robot places the remaining two blocks with the parent's pure-physics
top-down primitive.
"""

import numpy as np

from ..avatar.eval_mode_mixin import EvalModeAvatarMixin
from ..utils import Pose, create_primitive, to_numpy
from ..task_bases.blocks_ranking_size import BlocksRankingSize


class BlocksRankingSizeAssist(EvalModeAvatarMixin, BlocksRankingSize):
    """Human places one size block while the robot places the remaining blocks."""

    INSTRUCTION = (
        "place the RGB blocks in largest-to-smallest order while the human "
        "helps place one block"
    )

    use_avatar = True

    # Match blocks_ranking_rgb_assist: avatar stands at +Y and faces -Y.
    AVATAR_BASE_ROT = np.array(
        [[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=np.float64
    )
    avatar_init_pos = np.array([0.0, 0.55, -0.18])
    avatar_init_rot = AVATAR_BASE_ROT

    AVATAR_APPROACH_FRAMES = 260
    AVATAR_TRANSPORT_FRAMES = 360
    AVATAR_RETRACT_FRAMES = 180
    AVATAR_START_DELAY_STEPS = (100, 400)
    AVATAR_PICK_LIFT = 0.010
    AVATAR_PLACE_LIFT = 0.075
    AVATAR_NATURAL_BODY_MARGIN = 0.34
    AVATAR_NATURAL_YAW_LIMIT_DEG = 30.0
    AVATAR_NATURAL_BODY_Y_BOUNDS = (0.28, 0.62)
    AVATAR_NATURAL_SETTLE_STEPS = 20
    EVAL_TRIGGER_STEP_MIN = 100
    EVAL_TRIGGER_STEP_MAX = 200

    AVATAR_SPAWN_X = (-0.22, 0.22)
    AVATAR_SPAWN_Y = (-0.03, 0.16)
    ROBOT_SPAWN_X = (-0.24, 0.24)
    ROBOT_SPAWN_Y = (-0.38, -0.22)
    AVATAR_LAYOUT_SAMPLE_ATTEMPTS = 160

    def __init__(self, config: dict = None):
        cfg = dict(config or {})
        cfg.setdefault("track_avatar_collision", False)
        super().__init__(cfg)

    def _sample_spawn_xys_for_avatar(self, avatar_size: str):
        xys = {}
        for _ in range(self.AVATAR_LAYOUT_SAMPLE_ATTEMPTS):
            candidate = {
                avatar_size: self._layout_xy([
                    np.random.uniform(*self.AVATAR_SPAWN_X),
                    np.random.uniform(*self.AVATAR_SPAWN_Y),
                ])
            }
            for size_name in self.SIZE_ORDER:
                if size_name == avatar_size:
                    continue
                candidate[size_name] = self._layout_xy([
                    np.random.uniform(*self.ROBOT_SPAWN_X),
                    np.random.uniform(*self.ROBOT_SPAWN_Y),
                ])

            ok = True
            for i, s0 in enumerate(self.SIZE_ORDER):
                for s1 in self.SIZE_ORDER[i + 1:]:
                    if np.linalg.norm(candidate[s0] - candidate[s1]) < self.MIN_BLOCK_SPACING:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                return candidate
        raise RuntimeError("[blocks_size_assist] could not sample non-overlapping cubes")

    def load_actors(self):
        color_names = list(self.CUBE_COLORS.keys())
        color_order = np.random.permutation(color_names)
        size_order = np.random.permutation(list(self.SIZE_ORDER))

        self._avatar_size = str(np.random.choice(self.SIZE_ORDER))
        self._robot_sizes = tuple(
            s for s in self.SIZE_ORDER if s != self._avatar_size
        )
        self._avatar_start_delay_steps = int(
            np.random.randint(
                self.AVATAR_START_DELAY_STEPS[0],
                self.AVATAR_START_DELAY_STEPS[1] + 1,
            )
        )
        xys = self._sample_spawn_xys_for_avatar(self._avatar_size)

        self.blocks = {}
        self.block_colors = {}
        for i, size_name in enumerate(size_order):
            half = float(self.SIZE_HALVES[size_name])
            color_name = str(color_order[i])
            xy = xys[size_name]
            cube_z = self.TABLE_TOP_Z + half + self.CUBE_Z_CLEARANCE
            cube = create_primitive(
                self.scene,
                "box",
                Pose([float(xy[0]), float(xy[1]), cube_z]),
                size={"half_size": (half,) * 3},
                color=self.CUBE_COLORS[color_name],
                is_static=False,
            )
            self.blocks[size_name] = cube
            self.block_colors[size_name] = color_name

        self.target_slots = {
            size_name: self._target_slot_pos(
                size_name,
                self.TABLE_TOP_Z + float(self.SIZE_HALVES[size_name]),
            )
            for size_name in self.SIZE_ORDER
        }

    def _block_center(self, size_name: str) -> np.ndarray:
        return to_numpy(self.blocks[size_name].get_pos()).ravel()[:3]

    def _start_avatar_pick_and_place(self, add_start_delay: bool = True) -> None:
        size_name = self._avatar_size
        cube = self.blocks[size_name]
        half = float(self.SIZE_HALVES[size_name])
        pick_pos = self._block_center(size_name).copy()
        pick_pos[2] += self.AVATAR_PICK_LIFT
        slot = self.target_slots[size_name]
        place_pos = np.array([
            slot[0],
            slot[1],
            self.TABLE_TOP_Z + half + self.AVATAR_PLACE_LIFT,
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
        for size_name in self._robot_sizes:
            try:
                ok = self._pick_and_place_block(size_name, arm_tag)
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
                print(f"[blocks_size_assist/eval] avatar assist failed: {exc}", flush=True)

    def _eval_at_step(self, step_idx: int) -> None:
        if getattr(self, "_eval_assist_fired", False):
            return
        trigger = int(getattr(
            self, "_eval_assist_trigger_step", self.EVAL_TRIGGER_STEP_MIN
        ))
        if step_idx < trigger:
            return
        self._eval_assist_fired = True
        self._start_avatar_pick_and_place(add_start_delay=False)

    def evaluate(self) -> dict:
        metrics = super().evaluate()
        metrics.update({
            "assist_avatar_size": getattr(self, "_avatar_size", None),
            "assist_robot_sizes": list(getattr(self, "_robot_sizes", ())),
            "assist_avatar_start_delay_steps": int(getattr(
                self, "_avatar_start_delay_steps", 0
            )),
            "assist_avatar_spawn_y_band": list(self.AVATAR_SPAWN_Y),
            "assist_robot_spawn_y_band": list(self.ROBOT_SPAWN_Y),
            "eval_mode": bool(self.config.get("eval_mode", False)),
            "eval_assist_trigger_step": int(getattr(
                self, "_eval_assist_trigger_step", -1
            )),
            "eval_assist_fired": bool(getattr(self, "_eval_assist_fired", False)),
            "eval_policy_step_count": int(getattr(
                self, "_eval_policy_step_count", 0
            )),
        })
        return metrics
