"""Factory packing-station scene builder.

A small, reusable "factory work cell" in the spirit of an Amazon
fulfillment-center packing station: an industrial worktable shared by a
human worker (avatar, left side) and a robot arm (mounted on the right
side of the bench), with a warehouse storage rack as the backdrop, yellow
safety-tape floor markings around the cell, and a loaded pallet off to
the side. Everything is static furniture built from primitives — the
scene introduces **no graspable objects**; tasks spawn their own task
objects on the bench from ``load_actors()``.

Usage (mixin form — preferred for new factory tasks):

    from envs.base_task import BaseTask
    from envs.scenes.factory import FactorySceneMixin

    class FactoryKitPacking(FactorySceneMixin, BaseTask):
        use_avatar = True

        def __init__(self, config=None):
            cfg = dict(config or {})
            cfg.setdefault("robot_type", "franka")
            cfg.setdefault("robot_single_arm", True)
            super().__init__(cfg)

        def load_actors(self):
            self.build_factory()
            # then spawn task objects on the bench here.
            # self.factory_rack / self.factory_pallet / self.factory_tape
            # hold the furniture entities if a task ever needs them.

Customizing (class attributes, mirror of KitchenSceneMixin):
    - ``FACTORY_RACK``: ``FactoryRack`` config or ``None`` to skip.
    - ``FACTORY_FLOOR_TAPE``: ``FactoryFloorTape`` config or ``None``.
    - ``FACTORY_PALLET``: ``FactoryPallet`` config or ``None``.
    - ``FACTORY_TABLE_COLOR`` / ``FACTORY_TABLE_HALF``: bench look/size.
    - ``FACTORY_ROBOT_KWARGS``: robot mount pose on the bench.

Frame convention (same as kitchen): bench centered at the origin,
robot mounted on the bench top at (+0.60, -0.30), avatar stands at
(-0.75, -0.90) facing the bench (+y). Reachable bench area roughly
x ∈ [-0.8, 0.8], y ∈ [-0.42, 0.42], bench top z = 0.765.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field

from ..utils import Pose, create_primitive


# ---------------------------------------------------------------------------
# Furniture specs
# ---------------------------------------------------------------------------


@dataclass
class FactoryRack:
    """Warehouse storage rack standing behind the bench (higher +Y).

    Built from primitives: four square uprights (safety orange, like
    industrial pallet racking) + flat shelves (gray). Optionally dressed
    with a few static cardboard "stock" boxes so the backdrop reads as a
    warehouse — they are ``is_static=True`` decoration, not task objects.
    """

    width: float = 2.00               # along world X
    depth: float = 0.45               # along world Y
    height: float = 1.85              # upright height from floor
    y_front: float = 0.85             # world Y of the rack's front face
    post_half: float = 0.025          # square upright half-width
    shelf_thickness: float = 0.04
    shelf_heights: tuple = (0.50, 0.95, 1.40)   # shelf TOP surfaces (z)
    post_color: tuple = (0.80, 0.33, 0.10)      # safety orange
    shelf_color: tuple = (0.52, 0.54, 0.56)     # galvanized steel

    # Static stock boxes per shelf: (shelf_index, x_center, width, depth, height)
    stock_boxes: tuple = (
        (0, -0.70, 0.42, 0.38, 0.30),
        (0, -0.05, 0.36, 0.36, 0.26),
        (0, +0.62, 0.48, 0.40, 0.32),
        (1, -0.45, 0.40, 0.36, 0.28),
        (1, +0.35, 0.34, 0.34, 0.24),
        (2, -0.15, 0.44, 0.38, 0.28),
        (2, +0.65, 0.30, 0.32, 0.22),
    )
    box_color: tuple = (0.72, 0.56, 0.38)       # kraft cardboard


@dataclass
class FactoryFloorTape:
    """Yellow safety-tape rectangle painted on the floor around the work
    cell. Render-only (``collision=False``) so physics is untouched."""

    half_x: float = 1.70
    half_y: float = 1.55
    center_y: float = -0.10           # cell extends toward the worker side
    tape_width: float = 0.06
    color: tuple = (0.88, 0.76, 0.10)  # safety yellow


@dataclass
class FactoryPallet:
    """Wooden pallet with a couple of stacked sealed boxes, parked at the
    side of the cell (outbound staging). Pure static decoration."""

    center_xy: tuple = (1.55, 0.45)
    width: float = 0.95               # along world X
    depth: float = 0.80               # along world Y
    deck_height: float = 0.13
    wood_color: tuple = (0.55, 0.42, 0.26)
    # Stacked boxes: (dx, dy, width, depth, height, z_bottom-on-stack-order)
    boxes: tuple = (
        (-0.08, 0.00, 0.55, 0.50, 0.34, 0),
        (+0.02, 0.05, 0.42, 0.40, 0.28, 1),
    )
    box_color: tuple = (0.70, 0.54, 0.36)


# ---------------------------------------------------------------------------
# Builders (functional form)
# ---------------------------------------------------------------------------


def build_factory_rack(task, cfg: FactoryRack | None = None, verbose: bool = True) -> dict:
    """Add the warehouse rack behind the bench. Returns dict with
    ``"posts"``, ``"shelves"``, ``"boxes"`` entity lists."""
    cfg = cfg or FactoryRack()
    out: dict = {"posts": [], "shelves": [], "boxes": []}

    w_half = cfg.width / 2.0
    y_center = cfg.y_front + cfg.depth / 2.0

    # Four corner uprights.
    for sx in (-1, +1):
        for sy in (-1, +1):
            px = sx * (w_half - cfg.post_half)
            py = y_center + sy * (cfg.depth / 2.0 - cfg.post_half)
            out["posts"].append(create_primitive(
                task.scene, "box",
                Pose(p=[px, py, cfg.height / 2.0]),
                size={"half_size": (cfg.post_half, cfg.post_half, cfg.height / 2.0)},
                color=cfg.post_color, is_static=True,
            ))

    # Shelves (top surface at each configured height).
    shelf_tops = []
    for top_z in cfg.shelf_heights:
        z_center = top_z - cfg.shelf_thickness / 2.0
        shelf_tops.append(top_z)
        out["shelves"].append(create_primitive(
            task.scene, "box",
            Pose(p=[0.0, y_center, z_center]),
            size={"half_size": (w_half, cfg.depth / 2.0, cfg.shelf_thickness / 2.0)},
            color=cfg.shelf_color, is_static=True,
        ))

    # Static stock boxes resting on the shelves.
    for shelf_idx, x, bw, bd, bh in cfg.stock_boxes:
        if shelf_idx >= len(shelf_tops):
            continue
        z = shelf_tops[shelf_idx] + bh / 2.0
        out["boxes"].append(create_primitive(
            task.scene, "box",
            Pose(p=[x, y_center, z]),
            size={"half_size": (bw / 2.0, bd / 2.0, bh / 2.0)},
            color=cfg.box_color, is_static=True,
        ))

    if verbose:
        print(f"[factory] + rack y_front={cfg.y_front:.2f} "
              f"size=({cfg.width},{cfg.depth},{cfg.height}) "
              f"shelves={len(out['shelves'])} stock_boxes={len(out['boxes'])}")
    return out


def build_factory_floor_tape(task, cfg: FactoryFloorTape | None = None,
                             verbose: bool = True) -> list:
    """Paint the yellow work-cell border on the floor (render-only)."""
    cfg = cfg or FactoryFloorTape()
    strips = []
    t_half = cfg.tape_width / 2.0
    z = 0.0015                        # just above the floor plane
    cy = cfg.center_y
    # Two strips along X (front/back), two along Y (left/right).
    specs = [
        ((0.0, cy - cfg.half_y), (cfg.half_x + t_half, t_half)),
        ((0.0, cy + cfg.half_y), (cfg.half_x + t_half, t_half)),
        ((-cfg.half_x, cy), (t_half, cfg.half_y + t_half)),
        ((+cfg.half_x, cy), (t_half, cfg.half_y + t_half)),
    ]
    for (px, py), (hx, hy) in specs:
        strips.append(create_primitive(
            task.scene, "box",
            Pose(p=[px, py, z]),
            size={"half_size": (hx, hy, 0.0012)},
            color=cfg.color, is_static=True, collision=False,
        ))
    if verbose:
        print(f"[factory] + floor tape cell=({2*cfg.half_x:.1f}x{2*cfg.half_y:.1f}) m")
    return strips


def build_factory_pallet(task, cfg: FactoryPallet | None = None,
                         verbose: bool = True) -> dict:
    """Add the side pallet + stacked sealed boxes. Returns dict with
    ``"deck"`` and ``"boxes"``."""
    cfg = cfg or FactoryPallet()
    out: dict = {"boxes": []}
    cx, cy = cfg.center_xy
    out["deck"] = create_primitive(
        task.scene, "box",
        Pose(p=[cx, cy, cfg.deck_height / 2.0]),
        size={"half_size": (cfg.width / 2.0, cfg.depth / 2.0, cfg.deck_height / 2.0)},
        color=cfg.wood_color, is_static=True,
    )
    # Stack boxes in declared order (z_order 0 sits on the deck).
    z_cursor = {0: cfg.deck_height}
    for dx, dy, bw, bd, bh, z_order in sorted(cfg.boxes, key=lambda b: b[5]):
        base = z_cursor.get(z_order, cfg.deck_height)
        out["boxes"].append(create_primitive(
            task.scene, "box",
            Pose(p=[cx + dx, cy + dy, base + bh / 2.0]),
            size={"half_size": (bw / 2.0, bd / 2.0, bh / 2.0)},
            color=cfg.box_color, is_static=True,
        ))
        z_cursor[z_order + 1] = base + bh
    if verbose:
        print(f"[factory] + pallet at ({cx:+.2f},{cy:+.2f}) boxes={len(out['boxes'])}")
    return out


# ---------------------------------------------------------------------------
# Mixin
# ---------------------------------------------------------------------------


class FactorySceneMixin:
    """Drop-in mixin for BaseTask subclasses set in the factory work cell.

    Tasks inherit from it (``class MyTask(FactorySceneMixin, BaseTask)``)
    and call ``self.build_factory()`` from ``load_actors()``, or rely on
    the default ``load_actors`` provided here. Mirrors KitchenSceneMixin.
    """

    # Camera overrides — 3/4 view framing both the worker (left) and the
    # arm (right) with the rack as backdrop; side camera from the robot's
    # side for the opposite angle.
    recording_camera_pos = [-1.70, -1.60, 1.70]
    recording_camera_lookat = [0.00, 0.15, 0.90]
    side_camera_pos = [+1.90, -0.65, 1.35]
    side_camera_lookat = [0.00, 0.05, 0.90]

    # Avatar defaults — the worker stands at the LEFT half of the bench,
    # facing it (+y). Rotation: R_z(-90°) so local +X → world +Y.
    avatar_init_pos = np.array([-0.75, -0.90, -0.18])
    avatar_init_rot = np.array(
        [[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64,
    )

    FACTORY_RACK: FactoryRack | None = FactoryRack()
    FACTORY_FLOOR_TAPE: FactoryFloorTape | None = FactoryFloorTape()
    FACTORY_PALLET: FactoryPallet | None = FactoryPallet()

    # Industrial packing bench: 1.70 × 0.95 m (same footprint as the
    # kitchen counter so the proven robot/avatar placement carries over),
    # steel-gray top on dark box legs.
    FACTORY_TABLE_HALF = (0.85, 0.475)
    FACTORY_TABLE_COLOR: tuple = (0.61, 0.63, 0.66)   # brushed steel
    FACTORY_TABLE_LEG_COLOR: tuple = (0.22, 0.23, 0.25)
    FACTORY_TABLE_LEG_XY = ((-0.78, -0.41), (0.78, -0.41),
                            (-0.78, 0.41), (0.78, 0.41))

    # Franka mounted on the bench top, right side (kitchen-proven pose).
    FACTORY_ROBOT_KWARGS: dict = {"pos": [0.60, -0.30, 0.765]}

    def _load_robot(self):
        """Mount the robot on the bench instead of BaseTask's floating
        default in front of the table edge."""
        if self.FACTORY_ROBOT_KWARGS:
            kwargs = dict(self.config.get("robot_kwargs", {}) or {})
            for k, v in self.FACTORY_ROBOT_KWARGS.items():
                kwargs.setdefault(k, v)
            self.config["robot_kwargs"] = kwargs
        super()._load_robot()

    def _create_table(self, table_height: float = 0.74):
        """Industrial packing bench replacing BaseTask's default table."""
        self.TABLE_HEIGHT = table_height
        self.TABLE_THICKNESS = 0.05
        self.TABLE_TOP_Z = table_height + self.TABLE_THICKNESS / 2
        dx, dy = np.asarray(self.table_offset, dtype=float)

        hx, hy = self.FACTORY_TABLE_HALF
        self._table_top_z_fixed = False
        if self._try_create_table_variant(table_height, half_size=(hx, hy), center_xy=(dx, dy)):
            return
        self.table = create_primitive(
            self.scene, "box",
            Pose(p=[dx, dy, table_height]),
            size={"half_size": (hx, hy, self.TABLE_THICKNESS / 2)},
            color=self.FACTORY_TABLE_COLOR,
            is_static=True,
        )
        leg_h = table_height - self.TABLE_THICKNESS / 2
        for x, y in self.FACTORY_TABLE_LEG_XY:
            create_primitive(
                self.scene, "box",
                Pose(p=[x + dx, y + dy, leg_h / 2]),
                size={"half_size": (0.035, 0.035, leg_h / 2)},
                color=self.FACTORY_TABLE_LEG_COLOR,
                is_static=True,
            )

    def build_factory(self) -> None:
        """Build rack + floor tape + pallet. Stores entities on
        ``self.factory_rack`` / ``self.factory_tape`` / ``self.factory_pallet``."""
        self.factory_rack = (
            build_factory_rack(self, self.FACTORY_RACK)
            if self.FACTORY_RACK is not None else {}
        )
        self.factory_tape = (
            build_factory_floor_tape(self, self.FACTORY_FLOOR_TAPE)
            if self.FACTORY_FLOOR_TAPE is not None else []
        )
        self.factory_pallet = (
            build_factory_pallet(self, self.FACTORY_PALLET)
            if self.FACTORY_PALLET is not None else {}
        )

    def load_actors(self):
        """Default: just build the factory cell. Subclasses override to
        add their task objects alongside."""
        self.build_factory()
