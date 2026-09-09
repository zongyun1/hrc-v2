"""Blocks ranking RGB: arrange red, green, and blue cubes left-to-right.

Robot-only v1 of RoboTwin's ``blocks_ranking_rgb`` task.  A single Franka
sorts three primitive cubes into a fixed row ordered by color: red, green,
blue along increasing world X.
"""

import numpy as np

from ..base_task import BaseTask, TABLE_HEIGHT
from ..genesis_compat import update_dofs_force_range_compat, update_dofs_kp_kv_compat
from ..manipulation import PickSpec, PlaceSpec, TopDownPickPlaceMixin
from ..utils import Pose, create_primitive, to_numpy


class BlocksRankingRGB(TopDownPickPlaceMixin, BaseTask):
    """Robot places red, green, and blue blocks in left-to-right order."""

    INSTRUCTION = (
        "place the red block, green block, and blue block in a row from "
        "left to right"
    )
    use_avatar = False

    CUBE_HALF = 0.020
    CUBE_COLORS = {
        "red": (0.85, 0.08, 0.06),
        "green": (0.10, 0.60, 0.18),
        "blue": (0.08, 0.22, 0.85),
    }

    # World-X row: red, green, blue.  The row sits forward of the random
    # spawn zone so the task is visually legible and leaves clean approach
    # columns for top-down placement.
    SLOT_XS = {
        "red": -0.16,
        "green": 0.0,
        "blue": 0.16,
    }
    SLOT_Y = -0.08

    TRANSPORT_Z_ABOVE_TABLE = 0.20
    RELEASE_Z_ABOVE_TABLE = 0.080
    SUCCESS_DIST_XY = 0.055
    MIN_BLOCK_SPACING = 0.105
    ON_TABLE_HEIGHT_FRACTION = 0.35

    TABLE_THICKNESS_VALUE = 0.05
    TABLE_HALF_SIZE = (0.55, 0.50)
    TABLE_CENTER_XY = (0.0, -0.30)
    TABLE_LEG_RADIUS = 0.025
    TABLE_LEG_XY = (
        (-0.50, -0.75),
        (0.50, -0.75),
        (-0.50, 0.15),
        (0.50, 0.15),
    )

    SPAWN_X_RANGE = (-0.24, 0.24)
    SPAWN_Y_RANGE = (-0.38, -0.22)
    SPAWN_SAMPLE_ATTEMPTS = 80
    CUBE_Z_CLEARANCE = 0.002

    TABLE_SETTLE_STEPS = 80
    FAILURE_RECOVERY_STEPS = 60
    FINAL_SETTLE_STEPS = 200

    FINGER_KP = 9000.0
    FINGER_KV = 250.0
    FINGER_FORCE_LIMIT = 200.0

    recording_camera_pos = [0.85, -0.65, 1.55]
    recording_camera_lookat = [0.0, -0.22, 0.82]
    side_camera_pos = [0.0, -0.25, 2.45]
    side_camera_lookat = [0.0, -0.22, 0.78]

    # Real collection setting: rotate the authored sim layout by -90 deg
    # around world Z so the robot side moves from world y- to world x-.
    REAL_AXIS_ROT = np.array(
        [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )

    def __init__(self, config: dict = None):
        cfg = dict(config or {})
        cfg.setdefault("robot_type", "franka")
        cfg.setdefault("robot_single_arm", True)
        self._collection_setting = str(
            cfg.get("setting", cfg.get("collection_setting", "sim"))
        ).strip().lower()
        self._configure_collection_layout()
        super().__init__(cfg)

    def _real_setting_enabled(self) -> bool:
        return self._collection_setting == "real"

    def _layout_xy(self, xy) -> np.ndarray:
        arr = np.asarray(xy, dtype=np.float64).ravel()[:2]
        if not self._real_setting_enabled():
            return arr.copy()
        return np.array([arr[1], -arr[0]], dtype=np.float64)

    def _layout_xyz(self, xyz) -> np.ndarray:
        arr = np.asarray(xyz, dtype=np.float64).ravel()[:3]
        out = arr.copy()
        out[:2] = self._layout_xy(arr[:2])
        return out

    def _layout_xys(self, xys) -> np.ndarray:
        arr = np.asarray(xys, dtype=np.float64)
        if not self._real_setting_enabled():
            return arr.copy()
        out = arr.copy()
        out[:, 0] = arr[:, 1]
        out[:, 1] = -arr[:, 0]
        return out

    def _configure_collection_layout(self) -> None:
        if not self._real_setting_enabled():
            return
        cls = type(self)
        self.TABLE_CENTER_XY = tuple(self._layout_xy(cls.TABLE_CENTER_XY))
        self.TABLE_HALF_SIZE = (
            float(cls.TABLE_HALF_SIZE[1]),
            float(cls.TABLE_HALF_SIZE[0]),
        )
        self.TABLE_LEG_XY = tuple(
            tuple(self._layout_xy(xy)) for xy in cls.TABLE_LEG_XY
        )
        for attr in (
            "recording_camera_pos",
            "recording_camera_lookat",
            "side_camera_pos",
            "side_camera_lookat",
        ):
            value = getattr(cls, attr, None)
            if value is not None:
                setattr(self, attr, self._layout_xyz(value).tolist())
        if hasattr(cls, "avatar_init_pos"):
            self.avatar_init_pos = self._layout_xyz(cls.avatar_init_pos)
        if hasattr(cls, "avatar_init_rot"):
            self.avatar_init_rot = self.REAL_AXIS_ROT @ np.asarray(
                cls.avatar_init_rot, dtype=np.float64
            )
        self._R_DOWN = self.REAL_AXIS_ROT @ np.asarray(cls._R_DOWN, dtype=np.float64)

    def _create_table(self, table_height=TABLE_HEIGHT):
        self.TABLE_HEIGHT = table_height
        self.TABLE_THICKNESS = self.TABLE_THICKNESS_VALUE
        self.TABLE_TOP_Z = table_height + self.TABLE_THICKNESS / 2
        self._table_top_z_fixed = False
        if self._try_create_table_variant(
            table_height,
            half_size=self.TABLE_HALF_SIZE,
            center_xy=self.TABLE_CENTER_XY,
        ):
            return
        self.table = create_primitive(
            self.scene,
            "box",
            Pose(p=[*self.TABLE_CENTER_XY, table_height]),
            size={"half_size": (*self.TABLE_HALF_SIZE, self.TABLE_THICKNESS / 2)},
            color=(0.80, 0.75, 0.65),
            is_static=True,
        )
        leg_h = table_height - self.TABLE_THICKNESS / 2
        for x, y in self.TABLE_LEG_XY:
            create_primitive(
                self.scene,
                "cylinder",
                Pose(p=[x, y, leg_h / 2]),
                size={"radius": self.TABLE_LEG_RADIUS, "half_length": leg_h / 2},
                color=(0.50, 0.50, 0.50),
                is_static=True,
            )

    def _load_robot(self):
        self.config.setdefault("robot_type", "franka")
        if self.config["robot_type"] == "franka" and self._real_setting_enabled():
            kwargs = self.config.setdefault("robot_kwargs", {})
            kwargs.setdefault("pos", self._layout_xyz([0.0, -0.65, 0.75]))
            kwargs.setdefault("quat", [1.0, 0.0, 0.0, 0.0])
        super()._load_robot()

    def _sample_spawn_xys(self):
        """Sample a random non-overlapping layout away from the target row."""
        x_lo, x_hi = self.SPAWN_X_RANGE
        # Avoid the near-base band around y < -0.40: top-down Franka IK can
        # fail there for left-side cubes even though the position is close.
        y_lo, y_hi = self.SPAWN_Y_RANGE
        for _ in range(self.SPAWN_SAMPLE_ATTEMPTS):
            xys = np.column_stack([
                np.random.uniform(x_lo, x_hi, 3),
                np.random.uniform(y_lo, y_hi, 3),
            ])
            ok = True
            for i in range(3):
                for j in range(i + 1, 3):
                    if np.linalg.norm(xys[i] - xys[j]) < self.MIN_BLOCK_SPACING:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                return self._layout_xys(xys)
        raise RuntimeError("[blocks_rgb] could not sample non-overlapping cubes")

    def _target_slot_pos(self, key: str, z: float) -> np.ndarray:
        xy = self._layout_xy([self.SLOT_XS[key], self.SLOT_Y])
        return np.array([xy[0], xy[1], z], dtype=float)

    def _order_axis_value(self, pos) -> float:
        p = np.asarray(pos, dtype=np.float64).ravel()
        if self._real_setting_enabled():
            # Authored +X maps to world -Y under the real-axis rotation.
            return float(-p[1])
        return float(p[0])

    def load_actors(self):
        cube_z = self.TABLE_TOP_Z + self.CUBE_HALF + self.CUBE_Z_CLEARANCE
        colors = list(self.CUBE_COLORS.keys())
        order = np.random.permutation(colors)
        xys = self._sample_spawn_xys()

        self.blocks = {}
        for i, color_name in enumerate(order):
            cube = create_primitive(
                self.scene,
                "box",
                Pose([float(xys[i, 0]), float(xys[i, 1]), cube_z]),
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

    def _boost_finger_pd(self, arm_tag: str = "right"):
        arm = self.robot.get_arm(arm_tag)
        update_dofs_kp_kv_compat(
            arm.entity,
            arm._finger_dof_indices,
            kp_value=self.FINGER_KP,
            kv_value=self.FINGER_KV,
        )
        update_dofs_force_range_compat(
            arm.entity,
            arm._finger_dof_indices,
            lower_value=-self.FINGER_FORCE_LIMIT,
            upper_value=+self.FINGER_FORCE_LIMIT,
        )

    def _pick_and_place_block(self, color_name: str, arm_tag: str = "right") -> bool:
        cube = self.blocks[color_name]
        target = self.target_slots[color_name]
        transport_z = self.TABLE_TOP_Z + self.TRANSPORT_Z_ABOVE_TABLE
        release_z = self.TABLE_TOP_Z + self.RELEASE_Z_ABOVE_TABLE

        pick = PickSpec(
            get_center=lambda c=cube: to_numpy(c.get_pos()).ravel()[:3],
            radius=self.CUBE_HALF,
            label=f"{color_name} block",
        )
        place = PlaceSpec(
            pos=np.array([target[0], target[1], release_z], dtype=float),
            label=f"{color_name} slot",
            transport_z=transport_z,
            release=True,
        )
        return bool(self.pick_and_place(pick, place, arm_tag))

    def play_once(self) -> bool:
        arm_tag = "right"
        self._boost_finger_pd(arm_tag)

        # Pre-task object settle; excluded from recordings.
        with self.suppress_recording():
            for _ in range(self.TABLE_SETTLE_STEPS):
                self.step_sim()

        ok_all = True
        for color_name in ("red", "green", "blue"):
            try:
                ok = self._pick_and_place_block(color_name, arm_tag)
            except Exception:
                ok = False
            if not ok:
                ok_all = False
                self.open_gripper(arm_tag)
                for _ in range(self.FAILURE_RECOVERY_STEPS):
                    self.step_sim()

        for _ in range(self.FINAL_SETTLE_STEPS):
            self.step_sim()
        return ok_all

    def check_success(self) -> bool:
        positions = {}
        slot_ok = True
        on_table_ok = True

        for color_name, cube in self.blocks.items():
            p = to_numpy(cube.get_pos()).ravel()[:3]
            positions[color_name] = p
            target = self.target_slots[color_name]
            dist_xy = float(np.linalg.norm(p[:2] - target[:2]))
            in_slot = dist_xy <= self.SUCCESS_DIST_XY
            on_table = p[2] >= self.TABLE_TOP_Z + self.CUBE_HALF * self.ON_TABLE_HEIGHT_FRACTION
            slot_ok = slot_ok and in_slot
            on_table_ok = on_table_ok and on_table

        order_ok = (
            self._order_axis_value(positions["red"]) <
            self._order_axis_value(positions["green"]) <
            self._order_axis_value(positions["blue"])
        )
        success = bool(self.plan_success and slot_ok and on_table_ok and order_ok)
        return success

    def evaluate(self) -> dict:
        metrics = super().evaluate()
        positions = {}
        dist_xy = {}
        slot_ok = {}
        on_table_ok = {}

        for color_name, cube in self.blocks.items():
            p = to_numpy(cube.get_pos()).ravel()[:3]
            target = self.target_slots[color_name]
            dxy = float(np.linalg.norm(p[:2] - target[:2]))
            positions[color_name] = p.tolist()
            dist_xy[color_name] = dxy
            slot_ok[color_name] = bool(dxy <= self.SUCCESS_DIST_XY)
            on_table_ok[color_name] = bool(
                p[2] >= self.TABLE_TOP_Z + self.CUBE_HALF * self.ON_TABLE_HEIGHT_FRACTION
            )

        order_ok = bool(
            self._order_axis_value(positions["red"]) <
            self._order_axis_value(positions["green"]) <
            self._order_axis_value(positions["blue"])
        )
        metrics.update({
            "block_positions": positions,
            "block_dist_xy": dist_xy,
            "block_slot_ok": slot_ok,
            "block_on_table_ok": on_table_ok,
            "block_all_slots_ok": bool(all(slot_ok.values())),
            "block_all_on_table_ok": bool(all(on_table_ok.values())),
            "block_order_ok": order_ok,
            "block_success_dist_xy": self.SUCCESS_DIST_XY,
        })
        return metrics
