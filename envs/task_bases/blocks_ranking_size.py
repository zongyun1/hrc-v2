"""Blocks ranking size: arrange RGB cubes from largest to smallest.

Robot-only single-Franka version of RoboTwin's ``blocks_ranking_size`` task.
Three visible RGB primitive cubes have independently randomized sizes each
episode; the robot sorts them into a fixed row by size, not by color.
"""

import numpy as np

from ..base_task import BaseTask
from ..manipulation import PickSpec, PlaceSpec
from ..utils import Pose, create_primitive, to_numpy
from .blocks_ranking_rgb import BlocksRankingRGB


class BlocksRankingSize(BlocksRankingRGB):
    """Robot places three differently sized RGB blocks in largest-to-smallest order."""

    INSTRUCTION = (
        "place the largest, medium, and smallest blocks in a row from left "
        "to right"
    )

    # Full side lengths: 6.0 cm, 4.8 cm, 3.6 cm.  All are within the Franka
    # gripper's comfortable top-down grasp range while remaining visually
    # distinct in camera views.
    SIZE_HALVES = {
        "large": 0.030,
        "medium": 0.024,
        "small": 0.018,
    }
    SIZE_ORDER = ("large", "medium", "small")

    SLOT_XS = {
        "large": -0.16,
        "medium": 0.0,
        "small": 0.16,
    }

    SUCCESS_DIST_XY = 0.060

    def load_actors(self):
        size_names = list(self.SIZE_ORDER)
        color_names = list(self.CUBE_COLORS.keys())
        color_order = np.random.permutation(color_names)
        size_order = np.random.permutation(size_names)
        xys = self._sample_spawn_xys()

        self.blocks = {}
        self.block_colors = {}
        for i, size_name in enumerate(size_order):
            half = float(self.SIZE_HALVES[size_name])
            color_name = str(color_order[i])
            cube_z = self.TABLE_TOP_Z + half + self.CUBE_Z_CLEARANCE
            cube = create_primitive(
                self.scene,
                "box",
                Pose([float(xys[i, 0]), float(xys[i, 1]), cube_z]),
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

    def _pick_and_place_block(self, size_name: str, arm_tag: str = "right") -> bool:
        cube = self.blocks[size_name]
        target = self.target_slots[size_name]
        half = float(self.SIZE_HALVES[size_name])
        color_name = self.block_colors.get(size_name, "unknown")
        transport_z = self.TABLE_TOP_Z + self.TRANSPORT_Z_ABOVE_TABLE
        release_z = self.TABLE_TOP_Z + self.RELEASE_Z_ABOVE_TABLE

        pick = PickSpec(
            get_center=lambda c=cube: to_numpy(c.get_pos()).ravel()[:3],
            radius=half,
            label=f"{size_name} {color_name} block",
        )
        place = PlaceSpec(
            pos=np.array([target[0], target[1], release_z], dtype=float),
            label=f"{size_name} slot",
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
        for size_name in self.SIZE_ORDER:
            try:
                ok = self._pick_and_place_block(size_name, arm_tag)
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

        for size_name in self.SIZE_ORDER:
            cube = self.blocks[size_name]
            half = float(self.SIZE_HALVES[size_name])
            p = to_numpy(cube.get_pos()).ravel()[:3]
            positions[size_name] = p
            target = self.target_slots[size_name]
            dist_xy = float(np.linalg.norm(p[:2] - target[:2]))
            in_slot = dist_xy <= self.SUCCESS_DIST_XY
            on_table = p[2] >= self.TABLE_TOP_Z + half * self.ON_TABLE_HEIGHT_FRACTION
            slot_ok = slot_ok and in_slot
            on_table_ok = on_table_ok and on_table

        order_ok = (
            self._order_axis_value(positions["large"]) <
            self._order_axis_value(positions["medium"]) <
            self._order_axis_value(positions["small"])
        )
        success = bool(self.plan_success and slot_ok and on_table_ok and order_ok)
        return success

    def evaluate(self) -> dict:
        metrics = BaseTask.evaluate(self)
        positions = {}
        dist_xy = {}
        slot_ok = {}
        on_table_ok = {}

        for size_name in self.SIZE_ORDER:
            cube = self.blocks[size_name]
            half = float(self.SIZE_HALVES[size_name])
            p = to_numpy(cube.get_pos()).ravel()[:3]
            target = self.target_slots[size_name]
            dxy = float(np.linalg.norm(p[:2] - target[:2]))
            positions[size_name] = p.tolist()
            dist_xy[size_name] = dxy
            slot_ok[size_name] = bool(dxy <= self.SUCCESS_DIST_XY)
            on_table_ok[size_name] = bool(
                p[2] >= self.TABLE_TOP_Z + half * self.ON_TABLE_HEIGHT_FRACTION
            )

        order_ok = bool(
            self._order_axis_value(positions["large"]) <
            self._order_axis_value(positions["medium"]) <
            self._order_axis_value(positions["small"])
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
