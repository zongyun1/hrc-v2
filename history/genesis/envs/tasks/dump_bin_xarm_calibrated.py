"""xArm-only dump-bin task in the calibrated real-robot coordinate frame."""

from __future__ import annotations

import numpy as np

from ..base_task import TABLE_HEIGHT
from ..task_bases.dump_bin import DumpBin
from ..utils import Pose, create_primitive
from .dump_bin_assist import DumpBinAssist
from .dump_bin_interrupt import DumpBinInterrupt


_R_OLD_TO_XARM = np.array(
    [
        [0.0, 1.0],
        [-1.0, 0.0],
    ],
    dtype=np.float64,
)
_R3_OLD_TO_XARM = np.array(
    [
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)
_OLD_BASE_XY = np.array([0.0, -0.65], dtype=np.float64)
_OLD_BASE_Z = 0.66


def _xy(xy):
    xy = np.asarray(xy, dtype=np.float64)
    return tuple((_R_OLD_TO_XARM @ (xy - _OLD_BASE_XY)).tolist())


def _z(z):
    return float(z - _OLD_BASE_Z)


def _xyz(xyz):
    xyz = np.asarray(xyz, dtype=np.float64)
    return np.array([*_xy(xyz[:2]), _z(xyz[2])], dtype=np.float64)


def _rot(rot):
    return _R3_OLD_TO_XARM @ np.asarray(rot, dtype=np.float64)


def _range_xy(x_range, y_range):
    corners = [
        _xy((x, y))
        for x in x_range
        for y in y_range
    ]
    xs = [p[0] for p in corners]
    ys = [p[1] for p in corners]
    return tuple(sorted((min(xs), max(xs)))), tuple(sorted((min(ys), max(ys))))


class DumpBinXArmCalibratedMixin:
    """xArm identity-frame calibration shared by dump-bin variants."""

    # Old xArm task coordinates were collected with the arm base at
    # p=(0, -0.65, 0.66), q=+90deg about z.  Express those same physical
    # positions in the new xArm identity frame.
    _R_DOWN = np.array([
        [0.0, -1.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
    ], dtype=np.float64)

    TABLE_HEIGHT_XARM = _z(TABLE_HEIGHT)
    TABLE_CENTER_XY = _xy((0.0, -0.35))
    TABLE_LEG_XY = tuple(_xy(xy) for xy in DumpBin.TABLE_LEG_XY)
    BIG_BIN_XY = tuple(np.asarray(_xy(DumpBin.BIG_BIN_XY)) + np.array([-0.10, 0.10]))
    BIG_BIN_FLOOR_Z = _z(0.0)
    SPAWN_X_RANGE = tuple(sorted([y + 0.65 for y in DumpBin.SPAWN_Y_RANGE]))
    SPAWN_Y_RANGE = tuple(sorted([-x for x in DumpBin.SPAWN_X_RANGE]))
    SUCCESS_FLOOR_Z_MIN = _z(DumpBin.SUCCESS_FLOOR_Z_MIN)

    recording_camera_pos = [1.05, -0.55, 0.58]
    recording_camera_lookat = [0.32, -0.05, 0.08]
    side_camera_pos = [1.05, -0.55, 0.58]
    side_camera_lookat = [0.32, -0.05, 0.08]
    PICK_PLACE_SETTLE_ABOVE_PLACE_STEPS = 0
    PICK_PLACE_RELEASE_OPEN_STEPS = 10
    PICK_PLACE_RELEASE_SETTLE_STEPS = 120
    POST_DROP_SETTLE_STEPS = 120
    XARM_HOME_QPOS = [
        0.000000,
        -0.243858,
        -0.000029,
        0.905387,
        -0.000016,
        1.156799,
        0.000000,
    ]

    def __init__(self, config: dict = None):
        cfg = dict(config or {})
        cfg["robot_type"] = "xarm7"
        cfg.setdefault("robot_kwargs", {})
        cfg["robot_kwargs"].setdefault("pos", [0.0, 0.0, 0.0])
        cfg["robot_kwargs"].setdefault("quat", [1.0, 0.0, 0.0, 0.0])
        cfg.setdefault("planner_override", "mplib")
        cfg.setdefault("xarm_calibrated_head_camera", True)
        cfg.setdefault("vla_head_camera", True)
        cfg.setdefault("randomize_table", True)
        cfg.setdefault("debug_table_randomization", True)
        cfg.setdefault("table_variant_name", "sapien-table-22339")
        cfg.setdefault("floor_center", [0.0, 0.0, _z(0.0)])
        super().__init__(cfg)

    def _create_floor(self):
        self.config.setdefault("floor_center", [0.0, 0.0, _z(0.0)])
        super()._create_floor()

    def _create_table(self, table_height=TABLE_HEIGHT):
        super()._create_table(table_height=self.TABLE_HEIGHT_XARM)

    def _load_robot(self):
        super()._load_robot()
        if getattr(self.robot, "right_arm", None) is not None:
            self.robot.right_arm.homestate = list(self.XARM_HOME_QPOS)

    def _ensure_final_gripper_z(self, arm_tag: str = "right") -> None:
        return


class DumpBinXArmCalibrated(DumpBinXArmCalibratedMixin, DumpBin):
    """Dump-bin task with xArm at identity and scene objects remapped to it."""


class DumpBinXArmCalibratedAssist(DumpBinXArmCalibratedMixin, DumpBinAssist):
    """Assist dump-bin task in the calibrated xArm identity frame."""

    DUMP_CUBE_COLOR = (1.0, 0.0, 0.0)
    avatar_init_pos = _xyz(DumpBinAssist.avatar_init_pos)
    AVATAR_BASE_ROT = _rot(DumpBinAssist.AVATAR_BASE_ROT)
    avatar_init_rot = AVATAR_BASE_ROT
    _FRANKA_BASE_XY = (0.0, 0.0)
    ROBOT_PARK_QPOS = np.array(DumpBinXArmCalibratedMixin.XARM_HOME_QPOS, dtype=np.float64)

    def __init__(self, config: dict = None):
        cfg = dict(config or {})
        cfg.setdefault("dump_cube_color", self.DUMP_CUBE_COLOR)
        super().__init__(cfg)

    def _resolve_num_objects(self) -> int:
        return 1

    def load_actors(self):
        DumpBin.load_actors(self)
        self.avatar_cube = create_primitive(
            self.scene,
            "box",
            Pose(self.AVATAR_CUBE_PARK_POS),
            size={"half_size": (self.AVATAR_CUBE_HALF,) * 3},
            color=self.DUMP_CUBE_COLOR,
        )

    def _load_cubes(self):
        DumpBin._load_cubes(self)


class DumpBinXArmCalibratedInterrupt(DumpBinXArmCalibratedMixin, DumpBinInterrupt):
    """Interrupt dump-bin task in the calibrated xArm identity frame."""

    DUMP_CUBE_COLOR = (1.0, 0.0, 0.0)
    AVATAR_BASE_ROT = _rot(DumpBinInterrupt.AVATAR_BASE_ROT)
    avatar_init_pos = _xyz(DumpBinInterrupt.avatar_init_pos)
    avatar_init_rot = AVATAR_BASE_ROT
    AVATAR_REACH_SPAWN_X_RANGE, AVATAR_REACH_SPAWN_Y_RANGE = _range_xy(
        DumpBinInterrupt.AVATAR_REACH_SPAWN_X_RANGE,
        DumpBinInterrupt.AVATAR_REACH_SPAWN_Y_RANGE,
    )
    INTERRUPT_SPAWN_X_RANGE = AVATAR_REACH_SPAWN_X_RANGE
    INTERRUPT_SPAWN_Y_RANGE = AVATAR_REACH_SPAWN_Y_RANGE
    RETREAT_Y_FROM_BASE = 0.0

    def __init__(self, config: dict = None):
        cfg = dict(config or {})
        cfg.setdefault("dump_cube_color", self.DUMP_CUBE_COLOR)
        super().__init__(cfg)
