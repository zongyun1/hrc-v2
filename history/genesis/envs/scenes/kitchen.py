"""Kitchen-table scene builder.

Provides a reusable, seed-deterministic initializer that dresses the default
BaseTask tabletop with kitchen items (cutboard, knife, bowl, plate, mug,
bread, bottles, skillet, kettle, ...). Every item's scale and z-placement
is computed from the **actual mesh bounds** (trimesh), not from the
``model_data*.json`` ``scale`` / ``extents`` fields, which are stale for
several SAPIEN-derived assets in this repo.

Usage (mixin form — preferred for new tasks):

    from envs.base_task import BaseTask
    from envs.scenes.kitchen import KitchenSceneMixin

    class CutBread(KitchenSceneMixin, BaseTask):
        use_avatar = True

        def __init__(self, config=None):
            cfg = dict(config or {})
            cfg.setdefault("robot_type", "franka")
            cfg.setdefault("robot_single_arm", True)
            super().__init__(cfg)

        def load_actors(self):
            self.build_kitchen()
            # reach into self.kitchen_items["cutboard"], ["knife"], ...
            # self.kitchen_backdrop["wall"], ["shelf"] also available.
            # then spawn any task-specific extras here.

Customizing:
    - ``KITCHEN_LAYOUT``: override the class attribute with your own list
      of ``KitchenItem`` specs (see ``DEFAULT_KITCHEN_LAYOUT`` for a
      template).
    - ``KITCHEN_BACKDROP``: pass a ``KitchenBackdrop`` instance, or set to
      ``None`` to skip the wall / shelf.
    - ``KITCHEN_TABLE_COLOR``: recolours the table top (``None`` keeps
      BaseTask's default pale tan).

Usage (functional form — for one-off scripts or non-BaseTask scenes):

    from envs.scenes.kitchen import build_kitchen_backdrop, build_kitchen_table
    build_kitchen_backdrop(task=self)
    items = build_kitchen_table(task=self)

The returned dict is keyed by each spec's ``role`` so downstream code can
reliably fetch ``items["knife"]`` etc.
"""

from __future__ import annotations

import functools
import numpy as np
import transforms3d as t3d
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from ..utils import Pose, load_mesh, create_primitive, ASSETS_PATH, _find_mesh_file


# ---------------------------------------------------------------------------
# Item spec
# ---------------------------------------------------------------------------


@dataclass
class KitchenShelfItem:
    """Static decoration placed on the bottom shelf of the wall cabinet —
    purely visual (``is_static=True``, convex), so it can never obstruct
    the robot or settle dynamics."""

    role: str
    asset: str
    model_id: int = 0
    target_dim_m: float = 0.12
    euler_deg: tuple[float, float, float] = (90.0, 0.0, 0.0)
    x: float = 0.0              # x offset along the cabinet width
    y_offset: float = 0.00      # y offset within the cabinet depth
    description: str = ""


@dataclass
class KitchenItem:
    """One item to drop on the kitchen table.

    Attributes:
        role: short stable key (``"cutboard"``, ``"knife"``, ``"plate"``);
            the returned dict is keyed by this.
        asset: folder name under ``assets/objects/``.
        model_id: which mesh variant (``model_data{N}.json`` / ``base{N}.glb``).
        target_dim_m: desired longest-axis dimension of the rotated mesh,
            in meters. We pick an isotropic scale so the bounding-box's
            longest axis hits that value.
        euler_deg: (roll, pitch, yaw) degrees applied to the mesh before
            placement — enough to make the item sit upright / flat.
        xy: (x, y) center on the table top. The item's world-AABB is
            centered on this XY and its bottom sits on the table top.
        is_static: fix in place (True for cutboard / skillet).
        convex: convexify the collision shape.
        sink_z: extra z offset pushing the item DOWN (positive = sink into
            the table). Negative hovers above the surface.
        extra_z: additional rise above the table top — e.g. for stacking a
            bread on a plate (set = plate thickness).
        description: free-form note, only for logging.
    """

    role: str
    asset: str
    model_id: int = 0
    target_dim_m: float = 0.10
    euler_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)
    xy: tuple[float, float] = (0.0, 0.0)
    is_static: bool = False
    convex: bool = True
    sink_z: float = 0.0
    extra_z: float = 0.0
    description: str = ""


# ---------------------------------------------------------------------------
# Frame convention
# ---------------------------------------------------------------------------
#
# Default FrankaRobot base at (0, -0.65, 0.75) facing +Y. Table is centered
# at the origin, 1.2 × 0.7 × 0.05 m. Reachable area on the table roughly:
#   x ∈ [-0.45, 0.45]   y ∈ [-0.25, 0.28]
#
# Most of our kitchen assets use +Y as their upright axis. We therefore
# default every item to Euler (90°, 0, 0) which maps local +Y → world +Z.
# Items that need a different orientation (knives, forks) override.

_UP_FROM_LOCAL_Y = (90.0, 0.0, 0.0)   # default: local +Y is up
# 034_knife: lie flat with the blade face on the table, long axis along
# world -X. Mesh: local +X is thickness (0.07), +Y is handle→tip (1.46),
# +Z is blade height (0.36). Euler (0, -90, 90) maps local +X → world +Z
# (thin axis vertical) and local +Y → world -X (long axis horizontal).
_FLAT_KNIFE = (0.0, -90.0, 90.0)
_FLAT_FORK = (0.0, -90.0, 90.0)
_FREE = (0.0, 0.0, 0.0)


# Cooktop location — kept as a module constant so the skillet can be placed
# exactly on top of it. If you move one, move both. Now sits on the LEFT
# (avatar's cooking side) of the counter.
COOKTOP_XY: tuple[float, float] = (-0.55, 0.28)
COOKTOP_HEIGHT_M: float = 0.04

# The Franka is mounted at (+0.60, -0.30) on the counter; keep a ~0.30 m
# clear bubble around that point so the arm has room to swing. Items below
# are placed so none sits inside the region x > +0.25 AND y > -0.15.


DEFAULT_KITCHEN_LAYOUT: list[KitchenItem] = [
    # --- skillet on the cooktop (far back-left, avatar's side) ---
    KitchenItem(
        role="skillet",
        asset="106_skillet",
        model_id=0,
        target_dim_m=0.26,
        euler_deg=_UP_FROM_LOCAL_Y,
        xy=COOKTOP_XY,
        is_static=True,
        convex=True,
        extra_z=COOKTOP_HEIGHT_M,     # rest on cooktop surface, not table
        description="frying pan resting on the cooktop",
    ),
    # --- kettle: back-left, next to the cooktop (avatar's reach) ---
    KitchenItem(
        role="kettle",
        asset="091_kettle",
        model_id=0,
        target_dim_m=0.22,
        euler_deg=_UP_FROM_LOCAL_Y,
        xy=(-0.22, 0.30),
        is_static=False,
        convex=True,
        description="kettle",
    ),
    # --- workspace: cutting board + knife in front of the avatar ---
    KitchenItem(
        role="cutboard",
        asset="104_board",
        model_id=0,
        target_dim_m=0.30,
        euler_deg=_UP_FROM_LOCAL_Y,
        xy=(-0.58, -0.05),
        is_static=True,
        convex=True,
        description="wooden cutting board, lying flat",
    ),
    KitchenItem(
        role="knife",
        asset="034_knife",
        model_id=0,
        target_dim_m=0.22,
        euler_deg=_FLAT_KNIFE,
        xy=(-0.58, -0.12),    # sits on the cutboard, slight y offset
        is_static=False,
        convex=True,
        extra_z=0.025,   # rest on cutboard (board is ~0.02 m thick)
        description="kitchen knife on the cutboard",
    ),
    # --- eating setup: plate + bread mid-counter, between avatar and robot ---
    KitchenItem(
        role="plate",
        asset="003_plate",
        model_id=0,
        target_dim_m=0.22,
        euler_deg=_UP_FROM_LOCAL_Y,
        xy=(-0.20, -0.12),
        is_static=False,
        convex=True,
        description="dinner plate",
    ),
    KitchenItem(
        role="bread",
        asset="075_bread",
        model_id=0,
        target_dim_m=0.09,
        euler_deg=_UP_FROM_LOCAL_Y,
        xy=(-0.20, -0.12),
        is_static=False,
        convex=True,
        extra_z=0.010,   # sits in the plate's concavity (convex hull is ~flat)
        description="bread on plate",
    ),
    # --- bowl + apple: avatar's left ---
    KitchenItem(
        role="bowl",
        asset="002_bowl",
        model_id=1,       # 002_bowl has no model_data0
        target_dim_m=0.16,
        euler_deg=_UP_FROM_LOCAL_Y,
        xy=(-0.40, -0.35),
        is_static=False,
        convex=True,
        description="small bowl",
    ),
    KitchenItem(
        role="apple",
        asset="035_apple",
        model_id=0,
        target_dim_m=0.08,
        euler_deg=_FREE,
        xy=(-0.72, -0.35),
        is_static=False,
        convex=True,
        description="apple on the counter",
    ),
    # --- back row: bottles, stretched along the back edge ---
    KitchenItem(
        role="vinegar",
        asset="066_vinegar",
        model_id=0,
        target_dim_m=0.22,
        euler_deg=_UP_FROM_LOCAL_Y,
        xy=(0.08, 0.35),
        is_static=False,
        convex=True,
        description="vinegar bottle",
    ),
    KitchenItem(
        role="soy_sauce",
        asset="065_soy-sauce",
        model_id=0,
        target_dim_m=0.20,
        euler_deg=_UP_FROM_LOCAL_Y,
        xy=(0.28, 0.35),
        is_static=False,
        convex=True,
        description="soy sauce bottle",
    ),
    KitchenItem(
        role="oil",
        asset="029_olive-oil",
        model_id=0,
        target_dim_m=0.22,
        euler_deg=_UP_FROM_LOCAL_Y,
        xy=(0.48, 0.35),
        is_static=False,
        convex=True,
        description="olive oil bottle",
    ),
    # --- dressing items spread on avatar's side / front ---
    KitchenItem(
        role="baguette",
        asset="054_baguette",
        model_id=2,
        target_dim_m=0.26,
        euler_deg=(90.0, 0.0, 90.0),     # local Z (long) → world +X
        xy=(-0.05, -0.33),                # front-center, clear of robot base at y=-0.30
        is_static=False,
        convex=True,
        description="baguette on the counter",
    ),
    KitchenItem(
        role="mug",
        asset="039_mug",
        model_id=0,
        target_dim_m=0.10,
        euler_deg=_UP_FROM_LOCAL_Y,
        xy=(0.05, 0.05),                  # mid-counter, avatar can reach
        is_static=False,
        convex=True,
        description="ceramic mug",
    ),
    KitchenItem(
        role="wineglass",
        asset="088_wineglass",
        model_id=0,
        target_dim_m=0.16,
        euler_deg=_UP_FROM_LOCAL_Y,
        xy=(0.20, -0.05),                 # mid-counter, clear of robot + kettle
        is_static=False,
        convex=True,
        description="wine glass on the counter",
    ),
    KitchenItem(
        role="milk_box",
        asset="038_milk-box",
        model_id=0,
        target_dim_m=0.12,
        euler_deg=_UP_FROM_LOCAL_Y,
        xy=(-0.78, 0.10),                 # far-left back, next to avatar
        is_static=False,
        convex=True,
        description="milk carton",
    ),
]


# ---------------------------------------------------------------------------
# Backsplash wall + extras
# ---------------------------------------------------------------------------


@dataclass
class KitchenBackdrop:
    """Static backsplash wall + (optional) open-front wall cabinet.

    The cabinet is built from five boxes (back, bottom, top, left, right)
    so the front is open — items placed on the bottom shelf stay visible
    from the counter-side camera. We don't have a clean single-asset wall
    cupboard in ``assets/objects/``, so constructing it from primitives is
    the simplest way to honour the "upper layer should be a cabinet with
    items inside" ask without pulling in an articulated drawer unit.

    Colour tuples are ``gs.surfaces.Default(color=)`` values, RGB in [0, 1].
    """

    # Backsplash wall placed behind the table (higher +Y).
    wall_width: float = 2.50              # along world X (matches 1.6 m counter + margin)
    wall_height: float = 1.25             # along world Z, from floor
    wall_thickness: float = 0.03
    wall_y_offset: float = 0.02           # gap between table back edge and wall
    wall_color: tuple = (0.93, 0.91, 0.86)  # warm off-white tile

    # Open-front wall cabinet mounted above the counter (rendered iff True).
    cabinet: bool = True
    cabinet_width: float = 1.70           # along world X (= counter width)
    cabinet_depth: float = 0.30           # along world Y
    cabinet_height: float = 0.35          # along world Z (inner height between shelves)
    cabinet_panel_thickness: float = 0.02
    cabinet_clearance: float = 0.55       # gap between table top and cabinet bottom
    cabinet_color: tuple = (0.55, 0.38, 0.22)  # stained wood

    # A floor accent tile to make the room feel enclosed.
    floor_tile: bool = False
    floor_tile_half: tuple = (1.80, 1.50)
    floor_tile_color: tuple = (0.85, 0.84, 0.82)

    # Decorative items placed on the cabinet's bottom shelf. Tuple of
    # ``KitchenShelfItem``.
    cabinet_items: tuple = ()


def build_kitchen_backdrop(
    task,
    *,
    table_y_max: float | None = None,
    table_top_z: float | None = None,
    cfg: KitchenBackdrop | None = None,
    verbose: bool = True,
) -> dict:
    """Add the backsplash wall + optional open-front wall cabinet + floor tile.

    Called BEFORE ``scene.build()`` (i.e. from ``load_actors``) just like
    ``build_kitchen_table``. Returns a dict with keys:
        "wall"           — backsplash wall entity
        "cabinet"        — dict of the 5 cabinet panels (back/bottom/top/left/right)
        "cabinet_items"  — dict role→entity of items on the bottom shelf
        "floor"          — floor tile (only if enabled)
    """
    cfg = cfg or KitchenBackdrop()
    if table_top_z is None:
        table_top_z = float(getattr(task, "TABLE_TOP_Z", 0.765))
    if table_y_max is None:
        # Read the counter back edge from the mixin's half-size if present.
        half = getattr(task, "KITCHEN_TABLE_HALF", (0.60, 0.35))
        table_y_max = float(half[1])

    out: dict = {}
    # --- backsplash wall ---
    wall_y = table_y_max + cfg.wall_y_offset + cfg.wall_thickness / 2.0
    wall_z = cfg.wall_height / 2.0   # bottom on floor
    out["wall"] = create_primitive(
        task.scene, "box",
        Pose(p=[0.0, wall_y, wall_z]),
        size={"half_size": (cfg.wall_width / 2.0,
                            cfg.wall_thickness / 2.0,
                            cfg.wall_height / 2.0)},
        color=cfg.wall_color,
        is_static=True,
    )
    if verbose:
        print(f"[kitchen-bg] + wall  y={wall_y:.3f}  z={wall_z:.3f}  "
              f"size=({cfg.wall_width},{cfg.wall_thickness},{cfg.wall_height})")

    # --- open-front wall cabinet ---
    cabinet_bottom_top_z = None
    item_y_center = None
    if cfg.cabinet:
        t = cfg.cabinet_panel_thickness
        # Outer shell: back panel flush against the wall's front face, then
        # the cabinet extends forward by ``cabinet_depth``.
        wall_front_y = wall_y - cfg.wall_thickness / 2.0
        shell_y_center = wall_front_y - cfg.cabinet_depth / 2.0
        back_y_center = wall_front_y - t / 2.0

        # Vertical layout. Bottom panel sits ``cabinet_clearance`` above the
        # counter; the top panel is ``cabinet_height`` above the bottom's
        # top surface. Items rest on the bottom panel's top surface.
        bottom_z_center = table_top_z + cfg.cabinet_clearance + t / 2.0
        cabinet_bottom_top_z = bottom_z_center + t / 2.0
        top_z_center = cabinet_bottom_top_z + cfg.cabinet_height + t / 2.0
        mid_z = (bottom_z_center + top_z_center) / 2.0

        w_half = cfg.cabinet_width / 2.0
        d_half = cfg.cabinet_depth / 2.0

        # Five panels: back, bottom, top, left, right. Top/bottom/sides span
        # the full cabinet depth and butt against the back panel — no gaps.
        out.setdefault("cabinet", {})["back"] = create_primitive(
            task.scene, "box",
            Pose(p=[0.0, back_y_center, mid_z]),
            size={"half_size": (w_half, t / 2.0,
                                (cfg.cabinet_height + t) / 2.0)},
            color=cfg.cabinet_color, is_static=True,
        )
        out["cabinet"]["bottom"] = create_primitive(
            task.scene, "box",
            Pose(p=[0.0, shell_y_center, bottom_z_center]),
            size={"half_size": (w_half, d_half, t / 2.0)},
            color=cfg.cabinet_color, is_static=True,
        )
        out["cabinet"]["top"] = create_primitive(
            task.scene, "box",
            Pose(p=[0.0, shell_y_center, top_z_center]),
            size={"half_size": (w_half, d_half, t / 2.0)},
            color=cfg.cabinet_color, is_static=True,
        )
        for sign, name in ((-1, "left"), (+1, "right")):
            out["cabinet"][name] = create_primitive(
                task.scene, "box",
                Pose(p=[sign * (w_half - t / 2.0), shell_y_center, mid_z]),
                size={"half_size": (t / 2.0, d_half,
                                    (cfg.cabinet_height + t) / 2.0)},
                color=cfg.cabinet_color, is_static=True,
            )

        # Items sit on the bottom shelf, centered in the usable interior —
        # from (wall_front_y - t) (front face of back panel) forward by
        # (cabinet_depth - t).
        item_y_center = wall_front_y - (cfg.cabinet_depth + t) / 2.0

        if verbose:
            print(f"[kitchen-bg] + cabinet shell_y_center={shell_y_center:.3f}  "
                  f"bottom_top_z={cabinet_bottom_top_z:.3f}  "
                  f"top_bottom_z={top_z_center - t/2:.3f}  "
                  f"item_y_center={item_y_center:.3f}")

        # Items on the bottom shelf. Static + convex so no physics surprises.
        out["cabinet_items"] = {}
        for s in cfg.cabinet_items:
            try:
                bmin, bmax, mesh_path = _measure_mesh(s.asset, s.model_id)
            except FileNotFoundError as err:
                if verbose:
                    print(f"[kitchen-bg] skip cabinet/{s.role}: {err}")
                continue
            q = _euler_deg_to_quat_wxyz(s.euler_deg)
            wmin_u, wmax_u = _world_aabb(bmin, bmax, 1.0, q)
            max_axis = float((wmax_u - wmin_u).max())
            if max_axis <= 1e-6:
                continue
            scale = s.target_dim_m / max_axis
            wmin, wmax = _world_aabb(bmin, bmax, scale, q)
            wc = (wmin + wmax) / 2.0
            py = item_y_center + s.y_offset - wc[1]
            px = s.x - wc[0]
            pz = cabinet_bottom_top_z - wmin[2]   # sit on bottom shelf
            entity = load_mesh(
                task.scene, mesh_path, Pose([px, py, pz], q),
                scale=(scale, scale, scale), is_static=True, convex=True,
            )
            out["cabinet_items"][s.role] = entity
            if verbose:
                sz = wmax - wmin
                print(f"[kitchen-bg] + cabinet/{s.role:10s} {s.asset:14s} "
                      f"x={s.x:+.2f} size=({sz[0]:.2f},{sz[1]:.2f},{sz[2]:.2f})")

    # --- floor accent tile (under the counter) ---
    if cfg.floor_tile:
        hx, hy = cfg.floor_tile_half
        out["floor"] = create_primitive(
            task.scene, "box",
            Pose(p=[0.0, 0.0, 0.001]),   # just above z=0 so it paints over the plane
            size={"half_size": (hx, hy, 0.001)},
            color=cfg.floor_tile_color,
            is_static=True,
            collision=False,             # the world plane already provides collision
        )
        if verbose:
            print(f"[kitchen-bg] + floor half=({hx},{hy})")

    return out


def build_kitchen_cooktop(
    task,
    *,
    xy: tuple[float, float] = COOKTOP_XY,
    width: float = 0.42,
    depth: float = 0.42,
    height: float = COOKTOP_HEIGHT_M,
    burner_radius: float | None = None,
    burner_count: int = 2,
    table_top_z: float | None = None,
    body_color: tuple = (0.08, 0.08, 0.09),     # near-black induction glass
    burner_color: tuple = (0.18, 0.08, 0.05),   # dim-red ring (unlit ~ dark)
    verbose: bool = True,
) -> dict:
    """Build a stylized induction cooktop (flat dark box + faint burner rings)
    on the counter. No articulation — just a visual surface so future tasks
    can "cook" something by moving a pan onto it. Returns a dict with
    ``"body"`` and ``"burners"`` entries."""
    if table_top_z is None:
        table_top_z = float(getattr(task, "TABLE_TOP_Z", 0.765))

    out: dict = {}
    cx, cy = xy
    cz = table_top_z + height / 2.0
    out["body"] = create_primitive(
        task.scene, "box",
        Pose(p=[cx, cy, cz]),
        size={"half_size": (width / 2.0, depth / 2.0, height / 2.0)},
        color=body_color, is_static=True,
    )

    r = burner_radius if burner_radius is not None else min(width, depth) * 0.22
    ring_h = 0.003
    ring_z = table_top_z + height + ring_h / 2.0
    offsets = []
    if burner_count >= 1:
        offsets.append((-width * 0.22, +depth * 0.22))
    if burner_count >= 2:
        offsets.append((+width * 0.22, -depth * 0.22))
    if burner_count >= 3:
        offsets.append((+width * 0.22, +depth * 0.22))
    if burner_count >= 4:
        offsets.append((-width * 0.22, -depth * 0.22))

    burners: list = []
    for dx, dy in offsets:
        burners.append(create_primitive(
            task.scene, "cylinder",
            Pose(p=[cx + dx, cy + dy, ring_z]),
            size={"radius": r, "half_length": ring_h / 2.0},
            color=burner_color, is_static=True,
        ))
    out["burners"] = burners
    if verbose:
        print(f"[kitchen-cooktop] xy=({cx:+.2f},{cy:+.2f}) "
              f"size=({width:.2f},{depth:.2f},{height:.2f}) burners={len(burners)}")
    return out


# ---------------------------------------------------------------------------
# Mesh measurement (trimesh)
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=64)
def _measure_mesh(asset: str, model_id: int) -> tuple[np.ndarray, np.ndarray, Path]:
    """Return (bounds_min, bounds_max, mesh_path) of the mesh in its local
    frame. Cached because trimesh load is not free."""
    import trimesh  # local import so non-kitchen tasks don't pay
    base = ASSETS_PATH / "objects" / asset
    for sub in ("visual", "collision", "."):
        d = base if sub == "." else base / sub
        if not d.exists():
            continue
        m = _find_mesh_file(d, model_id)
        if m is not None:
            mesh = trimesh.load(str(m), force="mesh")
            b = np.asarray(mesh.bounds, dtype=np.float64)  # (2, 3)
            return b[0], b[1], m
    raise FileNotFoundError(f"No mesh for {asset} model_id={model_id}")


def _euler_deg_to_quat_wxyz(euler_deg: Sequence[float]) -> np.ndarray:
    rx, ry, rz = (float(a) * np.pi / 180.0 for a in euler_deg)
    R = t3d.euler.euler2mat(rx, ry, rz, axes="sxyz")
    return np.asarray(t3d.quaternions.mat2quat(R), dtype=np.float64)


def _world_aabb(
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
    scale: float,
    quat_wxyz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """World-frame AABB min/max for a mesh placed at world origin with the
    given rotation + isotropic scale (mesh local points first scaled, then
    rotated)."""
    R = t3d.quaternions.quat2mat(quat_wxyz)
    # 8 corners of the local AABB
    corners = np.array([
        [bounds_min[0], bounds_min[1], bounds_min[2]],
        [bounds_min[0], bounds_min[1], bounds_max[2]],
        [bounds_min[0], bounds_max[1], bounds_min[2]],
        [bounds_min[0], bounds_max[1], bounds_max[2]],
        [bounds_max[0], bounds_min[1], bounds_min[2]],
        [bounds_max[0], bounds_min[1], bounds_max[2]],
        [bounds_max[0], bounds_max[1], bounds_min[2]],
        [bounds_max[0], bounds_max[1], bounds_max[2]],
    ]) * scale
    world_corners = corners @ R.T  # (8,3) @ (3,3).T = (8,3)
    return world_corners.min(0), world_corners.max(0)


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def build_kitchen_table(
    task,
    *,
    layout: Sequence[KitchenItem] | None = None,
    table_top_z: float | None = None,
    verbose: bool = True,
) -> dict:
    """Spawn a kitchen-dressing of items on the task's existing table.

    Called from ``load_actors()``. The task must already have created its
    table (via ``BaseTask._create_table``) so that ``task.TABLE_TOP_Z`` is
    valid. Returns a dict mapping each item's ``role`` to the loaded
    Genesis entity.

    Z-placement: we measure each mesh's local AABB with trimesh, apply the
    requested rotation + scale, then choose ``pose.z`` so the resulting
    world AABB bottom sits exactly on ``table_top + extra_z - sink_z``.
    X/Y are chosen so the world AABB is centered on ``spec.xy`` (this
    means ``pose.p`` is offset from the spec's xy by the rotated mesh
    centroid offset).
    """
    if layout is None:
        layout = DEFAULT_KITCHEN_LAYOUT
    if table_top_z is None:
        table_top_z = float(getattr(task, "TABLE_TOP_Z", 0.765))

    entities: dict = {}
    for spec in layout:
        try:
            bmin, bmax, mesh_path = _measure_mesh(spec.asset, spec.model_id)
        except FileNotFoundError as err:
            if verbose:
                print(f"[kitchen] skip {spec.role}: {err}")
            continue
        local_ext = bmax - bmin

        quat = _euler_deg_to_quat_wxyz(spec.euler_deg)

        # Rotated extents (unit scale) tell us the longest world-axis; pick
        # scale to hit target_dim_m on that axis.
        world_min_unit, world_max_unit = _world_aabb(bmin, bmax, 1.0, quat)
        world_ext_unit = world_max_unit - world_min_unit
        max_axis = float(world_ext_unit.max())
        if max_axis <= 1e-6:
            if verbose:
                print(f"[kitchen] skip {spec.role}: degenerate bounds")
            continue
        scale = float(spec.target_dim_m) / max_axis

        # Recompute world AABB at the chosen scale.
        world_min, world_max = _world_aabb(bmin, bmax, scale, quat)
        world_center = (world_min + world_max) / 2.0
        # pose.p is the world position of the mesh LOCAL ORIGIN. The world
        # AABB min/max above were computed assuming pose=(0,0,0), so:
        #   mesh-origin + world_min = actual-world-AABB-min
        # We want actual-world-AABB-bottom-Z = table_top + extra_z - sink_z
        # and actual-world-AABB-center-XY = spec.xy
        target_bottom = table_top_z + spec.extra_z - spec.sink_z
        px = spec.xy[0] - world_center[0]
        py = spec.xy[1] - world_center[1]
        pz = target_bottom - world_min[2]

        pose = Pose([px, py, pz], quat)
        entity = load_mesh(
            task.scene, mesh_path, pose,
            scale=(scale, scale, scale),
            is_static=spec.is_static,
            convex=spec.convex,
        )
        entities[spec.role] = entity
        world_size = world_max - world_min
        if verbose:
            print(
                f"[kitchen] + {spec.role:10s} {spec.asset:16s} "
                f"scale={scale:.3f}  origin=({px:+.2f},{py:+.2f},{pz:.3f})  "
                f"size=({world_size[0]:.2f},{world_size[1]:.2f},{world_size[2]:.2f})"
            )

    if verbose:
        print(f"[kitchen] spawned {len(entities)}/{len(layout)} items on "
              f"table at z={table_top_z:.3f}")
    return entities


# ---------------------------------------------------------------------------
# Mixin
# ---------------------------------------------------------------------------


# Items placed on the bottom shelf of the wall cabinet. X positions span
# roughly the cabinet's interior width (= cabinet_width - 2·panel_thickness).
# Default cabinet is 1.60 m wide, so usable x is roughly ±0.76.
DEFAULT_CABINET_ITEMS: tuple = (
    KitchenShelfItem(role="tea_box",   asset="112_tea-box",    model_id=0, target_dim_m=0.14, x=-0.62),
    KitchenShelfItem(role="coffee",    asset="113_coffee-box", model_id=0, target_dim_m=0.14, x=-0.42),
    KitchenShelfItem(role="can_a",     asset="071_can",        model_id=0, target_dim_m=0.13, x=-0.22),
    KitchenShelfItem(role="can_b",     asset="071_can",        model_id=1, target_dim_m=0.13, x=-0.04),
    KitchenShelfItem(role="jam",       asset="031_jam-jar",    model_id=0, target_dim_m=0.14, x=+0.14),
    KitchenShelfItem(role="milk_tea",  asset="101_milk-tea",   model_id=0, target_dim_m=0.16, x=+0.34),
    KitchenShelfItem(role="soap",      asset="107_soap",       model_id=0, target_dim_m=0.10, x=+0.52),
    KitchenShelfItem(role="roll",      asset="028_roll-paper", model_id=0, target_dim_m=0.16, x=+0.68),
)
# Backwards-compat alias — some older code may still reference this name.
DEFAULT_SHELF_ITEMS = DEFAULT_CABINET_ITEMS


class KitchenSceneMixin:
    """Drop-in mixin for BaseTask subclasses that want a kitchen setting.

    Tasks inherit from it and either call ``self.build_kitchen()`` from
    their own ``load_actors()`` or rely on the default ``load_actors``
    provided here. The mixin also overrides camera defaults to a 3/4 view
    that frames the whole counter.

    Class attributes:
        KITCHEN_LAYOUT: custom list of ``KitchenItem`` specs (overrides
            ``DEFAULT_KITCHEN_LAYOUT``).
        KITCHEN_BACKDROP: ``KitchenBackdrop`` config, or ``None`` to skip
            the backsplash wall / shelf entirely.
        KITCHEN_TABLE_COLOR: override ``BaseTask._create_table``'s default
            pale-tan with a warmer wood tone. Set to ``None`` to use the
            BaseTask default.
    """

    # Camera overrides — 3/4 view from the front, angled down. Recording
    # camera sits on the avatar's (left) side so you can see both the
    # cook and the arm across the counter; side camera is flipped to the
    # robot's (right) side for the opposite angle.
    recording_camera_pos = [-1.20, -1.10, 1.45]
    recording_camera_lookat = [0.00, 0.00, 0.90]
    side_camera_pos = [+1.55, -0.15, 1.25]
    side_camera_lookat = [0.00, 0.00, 0.90]

    # Avatar defaults — stand the human on the LEFT (cooking) side of the
    # counter, right next to the cooktop, facing the table (+y). The robot
    # is on the opposite (right) side so the two share the counter without
    # overlapping. Rotation: R_z(-90°) so local +X → world +Y.
    avatar_init_pos = np.array([-0.75, -0.90, -0.18])
    avatar_init_rot = np.array(
        [[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64,
    )

    KITCHEN_LAYOUT: Sequence[KitchenItem] | None = None
    KITCHEN_BACKDROP: KitchenBackdrop | None = KitchenBackdrop(
        cabinet_items=DEFAULT_CABINET_ITEMS,
    )
    KITCHEN_TABLE_COLOR: tuple | None = (0.56, 0.38, 0.22)  # warm walnut

    # Cooktop config — pass ``None`` to skip. Keys are forwarded to
    # ``build_kitchen_cooktop``. The ``xy`` here must match the skillet's
    # ``xy`` in ``DEFAULT_KITCHEN_LAYOUT`` so the pan sits on the cooktop.
    KITCHEN_COOKTOP: dict | None = {"xy": COOKTOP_XY}

    # Enlarged kitchen counter: 1.70 × 0.95 m (vs. 1.20 × 0.70 default).
    # Extra depth lets the Franka mount on the counter without floating in
    # the air in front of the table edge.
    KITCHEN_TABLE_HALF = (0.85, 0.475)
    KITCHEN_TABLE_LEG_XY = ((-0.80, -0.43), (0.80, -0.43),
                             (-0.80, 0.43), (0.80, 0.43))

    # Franka base gets placed on the RIGHT side of the counter, mounted on
    # top of the counter surface (z = TABLE_TOP_Z) so it isn't floating.
    # ``quat`` is kept at the Franka default (90° Z rotation so the arm
    # extends in +y). Tasks that need a different pose can override.
    KITCHEN_ROBOT_KWARGS: dict = {"pos": [0.60, -0.30, 0.765]}

    def _load_robot(self):
        """Inject ``KITCHEN_ROBOT_KWARGS`` into the robot config so every
        kitchen task mounts its robot on the counter (instead of floating
        0.22 m in front of the table edge like BaseTask's default)."""
        if self.KITCHEN_ROBOT_KWARGS:
            kwargs = dict(self.config.get("robot_kwargs", {}) or {})
            for k, v in self.KITCHEN_ROBOT_KWARGS.items():
                kwargs.setdefault(k, v)
            self.config["robot_kwargs"] = kwargs
        super()._load_robot()

    def _create_table(self, table_height: float = 0.74):
        """Override BaseTask._create_table to build a larger counter with a
        warm walnut finish."""
        import numpy as _np
        from ..utils import create_primitive as _create_primitive, Pose as _Pose

        self.TABLE_HEIGHT = table_height
        self.TABLE_THICKNESS = 0.05
        self.TABLE_TOP_Z = table_height + self.TABLE_THICKNESS / 2
        dx, dy = _np.asarray(self.table_offset, dtype=float)

        top_color = self.KITCHEN_TABLE_COLOR or (0.8, 0.75, 0.65)
        hx, hy = self.KITCHEN_TABLE_HALF
        self._table_top_z_fixed = False
        if self._try_create_table_variant(table_height, half_size=(hx, hy), center_xy=(dx, dy)):
            return
        self.table = _create_primitive(
            self.scene, "box",
            _Pose(p=[dx, dy, table_height]),
            size={"half_size": (hx, hy, self.TABLE_THICKNESS / 2)},
            color=top_color,
            is_static=True,
        )
        leg_h = table_height - self.TABLE_THICKNESS / 2
        for x, y in self.KITCHEN_TABLE_LEG_XY:
            _create_primitive(
                self.scene, "cylinder",
                _Pose(p=[x + dx, y + dy, leg_h / 2]),
                size={"radius": 0.025, "half_length": leg_h / 2},
                color=(0.35, 0.25, 0.18),    # darker wood to match top
                is_static=True,
            )

    def build_kitchen(self) -> dict:
        """Build the backdrop (wall + cabinet) + cooktop + dress the table."""
        if self.KITCHEN_BACKDROP is not None:
            self.kitchen_backdrop = build_kitchen_backdrop(
                self, cfg=self.KITCHEN_BACKDROP,
            )
        else:
            self.kitchen_backdrop = {}

        # Cooktop is built BEFORE the items so the skillet can settle onto it.
        if self.KITCHEN_COOKTOP is not None:
            self.kitchen_cooktop = build_kitchen_cooktop(self, **self.KITCHEN_COOKTOP)
        else:
            self.kitchen_cooktop = {}

        layout = self.KITCHEN_LAYOUT if self.KITCHEN_LAYOUT is not None else DEFAULT_KITCHEN_LAYOUT
        self.kitchen_items = build_kitchen_table(self, layout=layout)
        return self.kitchen_items

    def load_actors(self):
        """Default: just build the kitchen. Subclasses should override to
        add task-specific objects alongside."""
        self.build_kitchen()
