"""Place dual shoes: Franka picks 2 shoes off the table and drops them
into a stationary open-top shoebox.  Single-arm, no avatar — RoboTwin's
``place_dual_shoes`` task scaled down to one Franka.

**Pure physics** — for both robot and held objects:
  - Robot arm moves only via ``move_and_execute`` → ``execute_plan`` (PD
    trajectory tracking).  No ``set_qpos`` / ``teleport_arm_joints`` /
    ``attach_to_gripper`` calls during the rollout.  The only mutations
    we apply to the arm are PD-gain bumps on the fingers
    (``set_dofs_kp/kv/force_range``) — control-gain changes, not state
    teleports.  This matches the recipe in ``place_burger_fries``.
  - Shoes are held entirely by friction × normal force × gripper PD.  No
    kinematic snap, no ``entity.set_pos``/``set_quat`` on shoes during
    play_once.

Skips RoboTwin's "tip pointing left" orientation gate — success here
just requires both shoes to land inside the shoebox footprint and
above its floor.

Two shoes + the shoebox xy are randomized inside a 0.40 × 0.40 m
region with a 0.22 m minimum pairwise separation (shoebox is ~26 cm
along its long axis; that sep keeps shoes from spawning under the
shoebox lip).
"""

import json
import os
from pathlib import Path

import genesis as gs
import numpy as np
import transforms3d as t3d

from ..base_task import BaseTask
from ..manipulation import TopDownPickPlaceMixin
from ..genesis_compat import update_dofs_force_range_compat, update_dofs_kp_kv_compat
from ..utils import (
    ArticulationActor, Pose, load_object, create_primitive, to_numpy,
    ASSETS_PATH, TABLE_HEIGHT,
)
from ..grasp import tcp_to_link_pose, load_grasp_poses
from ..object_catalog import get_entry


# Target object identity comes from the central catalog (single source);
# model_id=0 is the variant whose grasp_poses_franka.yml has 4 verified
# top-down grasps (center / heel × y000 / y090 axis).  The 2 toe variants
# in the YAML are blacklisted in data/grasp_helper_grid/041_shoe.json
# (rate=0/5 — toe is too thin for parallel-jaw friction).
_SHOE_ENTRY = get_entry("041_shoe")
_SHOE_ID = _SHOE_ENTRY.object_id
_SHOE_MODEL_ID = _SHOE_ENTRY.model_id
_BOX_ID = "007_shoe-box"   # receptacle (not a manipulated target)

# Randomization region — covers the Franka's reachable workspace with
# the base mounted at (0, -0.65, 0.75).  Slightly wider than
# place_bread_in_basket's 0.40 × 0.40 m to give the shoebox + 2 shoes
# more spread.
# Spawn region tightened to the Franka's *reachable annulus* (base at
# [0,-0.65]).  Under Genesis 1.0.0 the shoe's top-down pre-grasp IK
# succeeds out to ~0.52 m from base and fails (no mplib solution) at
# >=0.56 m; the old (-0.38,+0.06) y-band let shoes spawn up to 0.71 m
# away, so ~half of seeds were ungraspable.  The new band keeps every
# corner within reach:
#   nearest  (x=0,    y=-0.34) -> 0.31 m
#   farthest (x=0.20, y=-0.16) -> 0.53 m
# (Empirically the shoe's top-down pre-grasp IK starts failing at ~0.54 m.
# A still-tighter 0.36x0.15 band tested slightly worse — it forces the
# box+shoe layout sampler into its diagonal fallback too often — so this
# 0.40x0.18 band is kept.)  The shoebox (drop target, approached from
# above) is reachable across the whole band.
_REGION_X = (-0.20, +0.20)   # 0.40 m wide
_REGION_Y = (-0.34, -0.16)   # 0.18 m deep, within the reachable annulus
# Shoebox is 26 cm along its long axis; shoes are 22 cm long.  0.24 m
# min sep prevents the box's long edge (or a yawed corner) from
# clipping a shoe spawn — needs a bit more headroom than
# place_bread_in_basket's 0.18 m because of the larger box footprint.
_MIN_SEP_M = 0.24
_SAMPLE_BATCH_ATTEMPTS = 64
_SAMPLE_POINT_ATTEMPTS = 64
_SAMPLE_FALLBACK_MARGIN = 0.05
# Box yaw randomization range (about world-z, composed with upright_q).
# ±60° lets the box face roughly any direction while keeping its long
# axis broadly across the reach corridor.
_BOX_YAW_RANGE = (-np.pi / 3, +np.pi / 3)

# Shoe mesh contact is soft/thin enough that the generic Franka "strong grip"
# recipe visibly drives the fingers into the shoe.  Keep the gripper forceful
# enough to survive transit, but avoid saturating it as hard as box/cube tasks.
_SHOE_FINGER_KP = 8000.0
_SHOE_FINGER_KV = 250.0
# ADAPTIVE cage close.  set_gripper is NORMALIZED (1=open → 8 cm total gap
# on the Franka, 0=closed), so the original 0.04 commanded a ~3 mm gap on a
# ~7.3 cm-wide hollow shoe: at the old 55–80 N force cap the fingers could
# only satisfy it by crushing until they tunnelled through the thin CoACD
# side walls INTO the cavity — the shoe was caged from inside and a finger
# stayed wedged under the collar at release ("finger stuck inside the
# shoe").  Width-MATCHED gaps (0.65/0.60 sized from the visual mesh) then
# proved fragile the other way: the CoACD collision surface runs several
# mm NARROWER than the visual mesh and settled shoes sink/tilt 1–3 cm
# (debug 61724798: fingers reached a 49.5 mm gap through a visually
# 54–58 mm heel without contact), so fixed gaps kept missing.
# The robust recipe is a DEEP target with a MODERATE force cap: fingers
# advance until they stall on the real collision surface wherever it is
# (self-calibrating to CoACD shrink, sink, and tilt), and 35 N is well
# below the sustained-crush regime that tunnels through the walls while
# 35 N × friction 5 ≈ 175 N shear capacity carries the ~3 N shoe through
# transit accelerations (25 N held the lift but lost the shoe mid-carry
# on 2/8 seeds, smoke 61725335).
_SHOE_FINGER_FORCE_N = 35.0
_SHOE_CLOSE_TARGET = 0.30
_SHOE_RELEASE_FORCE_N = 100.0
# Grip-point offsets from the live-AABB centre along the shoe's length
# axis.  The ankle collar + opening span the MIDDLE-REAR of the shoe
# (model 0 top-rim z in [-0.091, +0.009] of a ±0.104 length, collar peak
# 9.1 cm): a centre descent puts the palm on the collar (palm face sits
# ~7 cm above the table at the near-sole grasp depth) and the fingers
# over the cavity.  Two grippable ends exist; the pick prefers the RAISED
# end of a pitched shoe, else the end FARTHER from the shoebox (a toe/heel
# resting against the box wall makes that end's descent land on the box
# rim — smoke 61721621 seeds 1/2):
#   - TOE/vamp  (+0.045): closed top 5.4–6.1 cm (palm clears at the
#     near-sole grasp depth), walls 66→54 mm across the pad band.
#   - HEEL      (-0.068): collar rim 7.3 cm above the sole, near-parallel
#     quarter-panel walls 56–58 mm.
# Both ends use the adaptive _SHOE_CLOSE_TARGET above and the top-anchored
# grasp height below.
_SHOE_GRIP_TOEWARD_M = 0.045
_SHOE_GRIP_HEELWARD_M = 0.068
_SHOE_MIN_LIFT_DZ = 0.05
# Fingertip height measured UP from the shoe's MESH-ORIGIN z (= the sole
# plane at the body centre; mesh y=0 is the sole).  Neither the table nor
# the AABB is a reliable anchor: a settled shoe's hull can sink 1.7-2.8 cm
# into the tabletop (load concentrates on a sharp CoACD sole piece, so a
# table-anchored fingertip lands 4-7 cm above the ACTUAL sole in the
# narrow collar zone → graze or collar-hook), and the AABB top/centre
# drift with tilt + yaw (AABB support inflation).  The origin z rides
# with the sink exactly.  Heel grasp sits 8 mm higher so the palm face
# (fingertip + ~5 cm) clears the 7.4 cm heel-collar rim.
_SHOE_GRASP_ABOVE_SOLE_TOE_M = 0.020
_SHOE_GRASP_ABOVE_SOLE_HEEL_M = 0.028
_SHOE_GRASP_TABLE_CLEAR_M = 0.015
_SHOE_TRANSPORT_Z_ABOVE_TABLE = 0.22


def _read_model_data(name: str, model_id: int = 0) -> dict:
    p = ASSETS_PATH / "objects" / name / f"model_data{model_id}.json"
    with open(p) as f:
        return json.load(f)


def _read_scale(name: str, model_id: int = 0) -> np.ndarray | float:
    """Return the asset scale.  Some assets (007_shoe-box) use per-axis
    scales — return a 3-vector in that case so the URDF helper can apply
    them per-axis to the visual mesh."""
    raw = _read_model_data(name, model_id).get("scale", 1.0)
    if isinstance(raw, (list, tuple)):
        if len(raw) == 1:
            return float(raw[0])
        return np.array(raw, dtype=np.float64)
    return float(raw)


def _read_bbox_center(name: str, model_id: int = 0) -> np.ndarray:
    raw = _read_model_data(name, model_id).get("center", [0.0, 0.0, 0.0])
    return np.asarray(raw, dtype=np.float64)


def _ensure_shoebox_urdf(scale: np.ndarray) -> tuple[Path, np.ndarray, np.ndarray]:
    """Generate a one-link URDF for 007_shoe-box with **primitive
    box-wall collision** (floor + 4 sides) and the GLB mesh as visual.

    Why not CoACD on the GLB: the shoebox wall is ~5 mm thick at scale
    [0.47, 0.43, 0.40]; CoACD would split it into thin convex slabs
    that shoes can wedge through during the drop.  Five thick boxes
    (1 cm thick each in world units) physically can't be tunneled.

    URDF dimensions are baked in METERS (visual mesh pre-scaled by
    `scale`, box sizes pre-multiplied by `scale`), so the URDF morph
    is loaded with scale=1.0 — no ambiguity about whether Genesis's
    URDF scale applies to box primitives.

    Returns ``(urdf_path, half_extents_world_meshframe,
    bbox_min_meshframe_meters)`` where:

      - ``half_extents_world_meshframe`` is the box's half-extents
        in (mesh-x, mesh-y, mesh-z) order, in meters.  After upright_q
        is applied to the URDF body these map to (world-x, world-z,
        world-y).
      - ``bbox_min_meshframe_meters`` is the bbox-min in mesh frame,
        meters; mesh-y component is the floor's local-y location
        (mesh y=0 → floor on the table after upright_q).

    Cached at /tmp on first call.
    """
    # v2: atomic OBJ/URDF writes (see below) — bumped from v1 so any
    # truncated cache left by a pre-fix run (the source of the intermittent
    # "At least one mesh must be present in file" load error) is regenerated.
    cache_tag = "shoebox_v2_box"
    cache_dir = Path("/tmp") / f"genesis_hr_{cache_tag}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    visual_obj = cache_dir / "shoebox_visual.obj"
    urdf_path = cache_dir / "shoebox.urdf"

    # Per-axis scale vector.
    if isinstance(scale, (int, float)):
        scale_vec = np.array([scale, scale, scale], dtype=np.float64)
    else:
        scale_vec = np.asarray(scale, dtype=np.float64).reshape(3)

    # Bbox computed via trimesh on the GLB (in mesh-unscaled units).
    # Numbers here came from a one-time trimesh.load + bounds query
    # (see initial implementation notes); cached as constants so the
    # task module doesn't import trimesh on every reset.
    bbox_min_mesh = np.array([-0.18601, 0.0, -0.31977], dtype=np.float64)
    bbox_max_mesh = np.array([+0.18601, 0.23583, +0.31977], dtype=np.float64)

    # Pre-scale visual mesh + bake URDF box dims in meters.
    bbox_min_m = bbox_min_mesh * scale_vec
    bbox_max_m = bbox_max_mesh * scale_vec
    ext_m = bbox_max_m - bbox_min_m
    cen_m = (bbox_min_m + bbox_max_m) / 2.0

    # Race-safe OBJ generation. Many task processes load the shoebox
    # concurrently (parallel collect/sweep). A bare ``if not exists: export``
    # lets process B read a half-written OBJ during process A's ``mesh.export``
    # -> trimesh "At least one mesh must be present in file". Mirror the
    # ensure_inertial_urdf recipe: reuse only a non-empty file, otherwise
    # export to a unique temp path and ``os.replace`` (atomic on POSIX, so a
    # reader sees either no file or a complete one — never a partial write).
    if not (visual_obj.exists() and visual_obj.stat().st_size > 0):
        import trimesh
        asset_dir = ASSETS_PATH / "objects" / _BOX_ID
        glb = asset_dir / "visual" / "base0.glb"
        mesh = trimesh.load(str(glb), force="mesh")
        # Per-axis pre-scale to meters so URDF morph scale=1.0.
        mesh.apply_scale(scale_vec)
        # Geometry-only OBJ: rebuild a bare Trimesh (drop UV/material) so
        # export writes a SINGLE .obj file with no sibling .mtl / texture
        # PNG. trimesh writes those siblings non-atomically, and a
        # concurrent reader loading the OBJ would read a half-written
        # texture -> "image file is truncated". The shoebox visual is
        # cosmetic (collision is the primitive box walls below), so a
        # plain untextured mesh is fine and races become impossible.
        geom = trimesh.Trimesh(
            vertices=np.asarray(mesh.vertices), faces=np.asarray(mesh.faces),
            process=False,
        )
        tmp_obj = visual_obj.with_name(f"{visual_obj.stem}.{os.getpid()}.tmp.obj")
        geom.export(str(tmp_obj))
        os.replace(tmp_obj, visual_obj)

    wall_t = 0.01    # 1 cm wall thickness in world units
    floor_t = 0.01

    # Floor: thin slab at bottom of bbox, full x/z extents.
    floor_size = (ext_m[0], floor_t, ext_m[2])
    floor_center = (cen_m[0], bbox_min_m[1] + floor_t / 2, cen_m[2])

    # Side walls span y from top-of-floor to top-of-bbox.
    wall_y_lo = bbox_min_m[1] + floor_t
    wall_y_hi = bbox_max_m[1]
    wall_y_size = wall_y_hi - wall_y_lo
    wall_y_center = (wall_y_lo + wall_y_hi) / 2

    # Left/right (mesh-x) walls: thin in x, full in z.
    side_size = (wall_t, wall_y_size, ext_m[2])
    left_center = (bbox_min_m[0] + wall_t / 2, wall_y_center, cen_m[2])
    right_center = (bbox_max_m[0] - wall_t / 2, wall_y_center, cen_m[2])

    # Front/back (mesh-z) walls: thin in z, full in x.
    end_size = (ext_m[0], wall_y_size, wall_t)
    front_center = (cen_m[0], wall_y_center, bbox_max_m[2] - wall_t / 2)
    back_center = (cen_m[0], wall_y_center, bbox_min_m[2] + wall_t / 2)

    def _box_block(name: str, size, center) -> str:
        return (
            f'    <collision name="{name}">\n'
            f'      <origin xyz="{center[0]} {center[1]} {center[2]}" rpy="0 0 0"/>\n'
            f'      <geometry>\n'
            f'        <box size="{size[0]} {size[1]} {size[2]}"/>\n'
            f'      </geometry>\n'
            f'    </collision>'
        )

    collisions = "\n".join([
        _box_block("floor", floor_size, floor_center),
        _box_block("left", side_size, left_center),
        _box_block("right", side_size, right_center),
        _box_block("front", end_size, front_center),
        _box_block("back", end_size, back_center),
    ])

    # Same race-safety for the URDF (atomic write). The URDF references the
    # OBJ by name; since the OBJ is written (atomically) before this, any
    # process that sees a complete URDF also sees a complete OBJ.
    if not (urdf_path.exists() and urdf_path.stat().st_size > 0):
        tmp_urdf = urdf_path.with_name(f"{urdf_path.stem}.{os.getpid()}.tmp.urdf")
        tmp_urdf.write_text(
            f"""<?xml version="1.0"?>
<robot name="shoebox">
  <link name="base">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="2.50"/>
      <inertia ixx="2e-2" ixy="0" ixz="0" iyy="2e-2" iyz="0" izz="2e-2"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <mesh filename="{visual_obj.name}"/>
      </geometry>
    </visual>
{collisions}
  </link>
</robot>
"""
        )
        os.replace(tmp_urdf, urdf_path)
    half_ext_m = ext_m / 2.0
    return urdf_path, half_ext_m, bbox_min_m


def _sample_n_xy(
    n: int,
    rng: np.random.RandomState,
    max_outer: int = _SAMPLE_BATCH_ATTEMPTS,
):
    """Rejection-sample n xy positions inside the region with pairwise
    minimum separation."""
    for _outer in range(max_outer):
        positions: list[tuple[float, float]] = []
        ok = True
        for _ in range(n):
            placed = False
            for _ in range(_SAMPLE_POINT_ATTEMPTS):
                x = float(rng.uniform(*_REGION_X))
                y = float(rng.uniform(*_REGION_Y))
                if all(
                    (x - px) ** 2 + (y - py) ** 2 >= _MIN_SEP_M ** 2
                    for px, py in positions
                ):
                    positions.append((x, y))
                    placed = True
                    break
            if not placed:
                ok = False
                break
        if ok:
            return positions
    # Fallback: spread points along the diagonal of the region.
    xs = np.linspace(
        _REGION_X[0] + _SAMPLE_FALLBACK_MARGIN,
        _REGION_X[1] - _SAMPLE_FALLBACK_MARGIN,
        n,
    )
    ys = np.linspace(
        _REGION_Y[0] + _SAMPLE_FALLBACK_MARGIN,
        _REGION_Y[1] - _SAMPLE_FALLBACK_MARGIN,
        n,
    )
    return [(float(xs[i]), float(ys[i])) for i in range(n)]


class PlaceDualShoes(TopDownPickPlaceMixin, BaseTask):
    """Robot picks two shoes from the table and drops both into a
    stationary open-top shoebox."""

    INSTRUCTION = "put the shoes in the shoebox"
    OBJECT_SET = ["041_shoe"]
    use_avatar = False

    TABLE_THICKNESS_VALUE = 0.05
    TABLE_HALF_SIZE = (0.6, 0.55)
    TABLE_CENTER_XY = (0.0, -0.35)
    TABLE_LEG_RADIUS = 0.025
    TABLE_LEG_XY = (
        (-0.55, -0.85),
        (0.55, -0.85),
        (-0.55, 0.15),
        (0.55, 0.15),
    )

    LIFT_HEIGHT = 0.20         # m above grasp before transit
    APPROACH_HEIGHT = 0.22     # m above box rim before lowering
    DROP_HEIGHT = 0.08         # m above box rim at release

    # Both shoes must end up inside the box footprint for success.
    # Shoebox interior is ~16 × 25 cm.  Half the diagonal ≈ 0.149 m;
    # 0.18 m gives some slack for shoes that drop near the rim.
    SUCCESS_DIST_XY_M = 0.18
    # Shoe must end up above the box floor (within 1 cm tolerance).
    # Shoes resting on the box floor sit ~5 cm above the floor (their
    # half-height); a stricter z-floor check would reject those.
    SUCCESS_Z_FLOOR_MARGIN_M = -0.01
    BOX_FLOOR_Z_CLEARANCE = 0.001
    SHOE_SPAWN_Z_CLEARANCE = 0.04   # sole ~1 cm above table (origin↔sole ≈0.028 m)
    SHOE_FRICTION = 5.0
    BOX_FRICTION = 2.0
    PRE_GRASP_SETTLE_STEPS = 30
    PRE_CLOSE_SETTLE_STEPS = 30
    CLOSE_SETTLE_STEPS = 80
    FAILED_GRASP_SETTLE_STEPS = 30
    RELEASE_GRIPPER_STEPS = 260
    RELEASE_SETTLE_STEPS = 240
    POST_PLACE_SETTLE_STEPS = 60
    INITIAL_SETTLE_STEPS = 60

    recording_camera_pos = [1.1, 0.0, 1.95]
    recording_camera_lookat = [0.0, 0.0, 0.85]

    def __init__(self, config: dict = None):
        cfg = dict(config or {})
        cfg.setdefault("robot_type", "franka")
        super().__init__(cfg)
        # Registry/override hook (validates config["object_name"]); the shoe is
        # a size-1 white-list so this always resolves to 041_shoe.
        self.resolve_target_object()
        self._num_shoes = self.difficulty_object_count()

    # ------------------------------------------------------------------
    # Scene setup
    # ------------------------------------------------------------------

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
            self.scene, "box", Pose(p=[*self.TABLE_CENTER_XY, table_height]),
            size={"half_size": (*self.TABLE_HALF_SIZE, self.TABLE_THICKNESS / 2)},
            color=(0.8, 0.75, 0.65), is_static=True,
        )
        leg_h = table_height - self.TABLE_THICKNESS / 2
        for x, y in self.TABLE_LEG_XY:
            create_primitive(
                self.scene, "cylinder",
                Pose(p=[x, y, leg_h / 2]),
                size={"radius": self.TABLE_LEG_RADIUS, "half_length": leg_h / 2},
                color=(0.5, 0.5, 0.5), is_static=True,
            )

    def _load_robot(self):
        super()._load_robot()
        # Raise the Franka home pose so the EE starts well above the table —
        # avoids the gripper colliding with shoes spawned upright (~21 cm
        # tall). Default home has j4=-150° / j6=+168°; this is the ready
        # pose: shoulder back, elbow up, wrist forward. Non-Franka
        # embodiments keep their own homestate.
        if self.config.get("robot_type") == "franka" and self.robot.right_arm is not None:
            self.robot.right_arm.homestate = [
                0.0, -np.pi / 4, 0.0, -3.0 * np.pi / 4,
                0.0, np.pi / 2, np.pi / 4,
            ]

    def _sample_layout_positions(self, rng):
        """Return ``[(box_xy), ...shoe_xy]`` for this mode.

        Interrupt/assist variants override this to constrain one actor
        while keeping the parent's shoebox + shoe loading logic.
        """
        return _sample_n_xy(1 + int(self._num_shoes), rng)

    def load_actors(self):
        # mesh-y maps to world-z under upright_q; the shoe and shoebox
        # GLBs are both y-up.
        upright_q = np.array([0.70710678, 0.70710678, 0.0, 0.0], dtype=np.float64)
        table_top = self.TABLE_TOP_Z

        rng = np.random.RandomState(np.random.randint(0, 2 ** 31 - 1))
        positions = self._sample_layout_positions(rng)
        (box_x, box_y) = positions[0]
        shoe_xys = positions[1:]

        # ---- Shoebox (static receiver) ----
        # 1-link URDF with primitive walls; visual is the original GLB.
        # fixed=True so the box doesn't slide when shoes drop in — for
        # this task the box is just a target volume, not something the
        # robot touches.
        self._box_scale = _read_scale(_BOX_ID, 0)
        urdf_path, half_ext_mesh_m, bbox_min_mesh_m = _ensure_shoebox_urdf(
            self._box_scale,
        )
        # In mesh-frame meters: half_ext (mesh-x, mesh-y, mesh-z),
        # bbox_min (mesh-x, mesh-y, mesh-z).  After upright_q applied
        # to the URDF body, mesh-y → world-z and mesh-z → world-y, so:
        #   world-x extent = mesh-x extent  → box_w
        #   world-y extent = mesh-z extent  → box_l (along world y)
        #   world-z extent = mesh-y extent  → box_h
        box_w = float(2 * half_ext_mesh_m[0])
        box_h = float(2 * half_ext_mesh_m[1])
        box_l = float(2 * half_ext_mesh_m[2])
        # Random shoebox yaw (z-rotation, composed onto upright_q so the
        # box stays floor-down but rotated about world-z).  Compose order:
        # final = yaw ∘ upright (apply upright first to bring mesh-y →
        # world-z, then yaw spins about world-z).
        box_yaw = float(rng.uniform(*_BOX_YAW_RANGE))
        box_yaw_q = t3d.quaternions.axangle2quat(
            [0.0, 0.0, 1.0], box_yaw, is_normalized=True,
        )
        box_q = t3d.quaternions.qmult(box_yaw_q, upright_q)
        box_q /= np.linalg.norm(box_q)
        # mesh y = 0 is the box floor.  After upright_q the box's mesh
        # origin (at floor level) sits at the URDF's world position; so
        # placing the URDF at world-z = table_top puts the floor on the
        # table.
        self.box_pose = Pose([box_x, box_y, table_top + self.BOX_FLOOR_Z_CLEARANCE], box_q)
        self._box_yaw = box_yaw
        box_urdf_kwargs = dict(
            file=str(urdf_path.absolute()),
            pos=tuple(self.box_pose.p),
            quat=tuple(self.box_pose.q),
            scale=1.0,
            fixed=True,
        )
        # NYX renders a URDF as a sub-scene that re-parses the original file and
        # maps each file link 1:1 onto a Genesis link; Genesis's default
        # merge_fixed_links=True collapses fixed links and breaks that mapping
        # (RuntimeError at scene.build). Keep the un-merged tree only under NYX.
        if str(self.config.get("renderer", "")) == "nyx":
            box_urdf_kwargs["merge_fixed_links"] = False
        box_entity = self.scene.add_entity(
            gs.morphs.URDF(**box_urdf_kwargs),
            material=gs.materials.Rigid(friction=self.BOX_FRICTION),
        )
        self.shoebox = ArticulationActor(box_entity, {}, _BOX_ID)
        # Box rim in world: floor at table_top + ε, walls of height box_h.
        self._box_rim_z = float(table_top + self.BOX_FLOOR_Z_CLEARANCE + box_h)
        self._box_xy_half = np.array([box_w / 2, box_l / 2])

        # ---- Shoes (graspable) ----
        self._shoe_scale = _read_scale(_SHOE_ID, _SHOE_MODEL_ID)
        shoe_extents = np.array(
            _read_model_data(_SHOE_ID, _SHOE_MODEL_ID)["extents"],
            dtype=np.float64,
        )
        # World height after upright_q (mesh-y → world-z).  Shoes spawn
        # in their natural sole-down orientation (mesh-y axis vertical).
        # NOTE: an earlier v7 attempt spawned them STANDING (length
        # axis vertical via identity quat) per user request, but every
        # grasp's pre-RRT failed because the existing mesh-frame grasps
        # produce a horizontal gripper approach with an inverted wrist
        # — outside the Franka's IK space at the available home pose.
        # The "easier to grasp" benefit of a taller shoe was offset by
        # the unreachable pre-grasp.  Sole-down + raised home pose (see
        # _load_robot) gives the gripper enough table clearance.
        if isinstance(self._shoe_scale, np.ndarray):
            shoe_h = float(shoe_extents[1] * self._shoe_scale[1])
        else:
            shoe_h = float(shoe_extents[1] * self._shoe_scale)
        self._shoe_h = shoe_h
        self._shoe_center_mesh = _read_bbox_center(_SHOE_ID, _SHOE_MODEL_ID)

        self.shoes = []
        self.shoe_poses = []
        # Random per-shoe yaw about world-z so the toe direction varies
        # seed-to-seed; composed onto upright_q (sole-down).
        for (sx, sy) in shoe_xys:
            yaw = float(rng.uniform(-np.pi, np.pi))
            yaw_q = t3d.quaternions.axangle2quat(
                [0.0, 0.0, 1.0], yaw, is_normalized=True,
            )
            pose_q = t3d.quaternions.qmult(yaw_q, upright_q)
            pose_q /= np.linalg.norm(pose_q)
            # Spawn the shoe's sole ~1 cm above the table so it drops cleanly
            # and rests on the surface.  The mesh origin sits ~0.028 m above
            # the sole, so the old origin-at-table_top+0.02 put the sole ~0.8 cm
            # INTO the table; that spawn-time overlap let the convex hull
            # penetrate ~7 cm and wedge (height probe 61312848: bottom z=0.697
            # vs table 0.765).  A clearance of 0.04 lifts the sole clear with a
            # ~1 cm settle drop (lifting by 0.5*shoe_h instead dropped it ~6 cm
            # and tipped it, regressing the grasp — run5).
            spawn_z = table_top + self.SHOE_SPAWN_Z_CLEARANCE
            pose = Pose([sx, sy, spawn_z], pose_q)
            actor = load_object(
                self.scene, pose, _SHOE_ID, model_id=_SHOE_MODEL_ID,
                convex=True, is_static=False,
                friction=self.SHOE_FRICTION,
            )
            self.shoe_poses.append(pose)
            self.shoes.append(actor)

        # Custom check_success — TargetSpec only handles a single
        # object.  Set a stub so BaseTask.evaluate() prints something
        # useful for the first shoe; check_success below ignores it.
        from ..base_task import TargetSpec
        self.target = TargetSpec(
            object=self.shoes[0].entity,
            position=lambda: to_numpy(self.shoebox.entity.get_pos()).ravel()[:3],
            label="shoebox",
        )

    # ------------------------------------------------------------------
    # Pick + place primitive
    # ------------------------------------------------------------------

    # In-box slot offset (along the box's mesh-x = SHORT axis, world-frame
    # rotated by box yaw).  ±2 cm keeps both drops well inside the rim;
    # the success metric only requires both shoes inside the box footprint.
    SLOT_OFFSET_M = 0.020

    # ------------------------------------------------------------------
    # Robust top-down pick (live-AABB grip point + object-yaw alignment)
    # ------------------------------------------------------------------

    def _top_down_tcp(self, pos, yaw: float = 0.0):
        """Override the mixin's fixed top-down TCP so every mixin motion
        (`_move_seeded` / `_move_screw`) descends with the jaws rotated to
        ``self._grip_yaw_deg`` about the approach axis.  Set the attribute
        before the pick, reset to 0 after.  yaw=0 → world-Y jaws.

        Accepts (and ignores) the mixin's own `yaw` kwarg — this override
        always uses `self._grip_yaw_deg` instead, but `_move_screw` /
        `_move_cartesian` unconditionally pass `yaw=tcp_yaw` (default
        0.0), so the signature must accept it or every screw/cartesian
        call raises TypeError (caught by scripts/collect.py's episode
        try/except, silently producing 0/N with no grasp-diagnostic
        prints at all — this task's smoke tests must show `[shoe_pick]`
        / `[shoe_dbg]` lines, not just bare FAILs)."""
        yaw = float(getattr(self, "_grip_yaw_deg", 0.0))
        if abs(yaw) < 1e-6:
            return super()._top_down_tcp(pos)
        yr = np.deg2rad(yaw)
        cz, sz = np.cos(yr), np.sin(yr)
        Rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
        R = self._R_DOWN @ Rz
        return Pose(np.asarray(pos, dtype=float), t3d.quaternions.mat2quat(R))

    @staticmethod
    def _shoe_world_aabb(shoe):
        aabb = to_numpy(shoe.entity.get_AABB()).reshape(-1, 3)
        lo, hi = aabb[0], aabb[1]
        return lo, hi, 0.5 * (lo + hi)

    @staticmethod
    def _shoe_pos_quat(shoe):
        """Live base pose, read from the base LINK rather than the entity.

        After the avatar's kinematic attach (which writes entity
        set_pos/set_quat every frame during the inspect) the entity-level
        pos/quat can stay FROZEN at the detach-time values while the body
        physically falls and settles — eval 61747470 measured the entity
        origin 20 cm from the collision AABB centre, impossible for this
        rigid body (its mesh origin is ≤4.5 cm from the AABB centre).
        The link state tracks the simulated body."""
        ent = shoe.entity
        links = getattr(ent, "links", None)
        if links:
            link = links[0]
            return (
                to_numpy(link.get_pos()).ravel()[:3],
                to_numpy(link.get_quat()).ravel()[:4],
            )
        return (
            to_numpy(ent.get_pos()).ravel()[:3],
            to_numpy(ent.get_quat()).ravel()[:4],
        )

    def _object_grip_yaw(self, shoe) -> float:
        """tcp_yaw (deg) that closes the top-down jaws across the shoe's
        measured SHORT horizontal (mesh-x = width) axis.

        BUG FIXED (user report: gripper was closing along the shoe's LONG
        axis instead of its thin axis): ``_top_down_tcp`` builds
        ``_R_DOWN @ Rz(tcp_yaw)``, and ``_R_DOWN`` is a REFLECTION
        (det = -1, needed to point the gripper straight down). Composing
        a rotation with a reflection INVERTS the effective in-plane
        rotation direction, so the naive "tcp_yaw = theta - 90" mapping
        (theta = world-yaw of mesh-x) only happens to align the jaws
        with the width axis at yaw multiples of 90° — at every other
        shoe orientation it drifts away from width and toward length,
        worst case (45°/135°) grabbing squarely along the LONG axis.
        Verified analytically and by an exhaustive 15°-step sweep over
        all orientations: "90 - theta" aligns the closing axis with
        mesh-x for every angle (dot product ±1.000 throughout, vs.
        "theta - 90" which was correct only at 0/90/180/270 and
        perfectly WRONG — aligned with the long axis — at 45/135/etc).

        Wrapped to [-90, 90) — the jaws are 180°-symmetric, and raw
        values like -210° made the seeded IK pick contorted wrist
        branches (debug 61724798)."""
        q = np.asarray(self._shoe_pos_quat(shoe)[1], dtype=float)
        R = t3d.quaternions.quat2mat(q)
        x_axis = R @ np.array([1.0, 0.0, 0.0])
        theta = float(np.degrees(np.arctan2(x_axis[1], x_axis[0])))
        return ((180.0 - theta) % 180.0) - 90.0

    def _shoe_toe_dir_xy(self, shoe) -> np.ndarray:
        """Unit world-xy direction from heel to toe (mesh +z; the asset's
        orientation_point marks the toe at mesh z=+0.98).  Zero vector if
        the shoe is standing on end (length axis vertical)."""
        q = np.asarray(self._shoe_pos_quat(shoe)[1], dtype=float)
        R = t3d.quaternions.quat2mat(q)
        toe = R @ np.array([0.0, 0.0, 1.0])
        n = float(np.linalg.norm(toe[:2]))
        if n < 1e-6:
            return np.zeros(2)
        return toe[:2] / n

    def _choose_grip_end(self, shoe) -> float:
        """+1.0 → grip the toe/vamp end, -1.0 → grip the heel end.

        A settled shoe is often PITCHED (one end dipped — load concentrates
        on a sharp CoACD sole piece and that end sinks into the table): the
        dipped end's grip band is partly below the tabletop, so prefer the
        RAISED end when the pitch is significant.  Otherwise pick the end
        whose grip point lies farther from the shoebox centre: the spawn
        separation (0.24 m centre-to-centre) still allows a shoe's toe or
        heel to rest against a yawed box corner, and a grip point on that
        end descends onto the box rim and closes on air."""
        q = np.asarray(self._shoe_pos_quat(shoe)[1], dtype=float)
        R = t3d.quaternions.quat2mat(q)
        toe_z = float(R[2, 2])   # world-z component of mesh +z (toe dir)
        if os.environ.get("SHOE_DEBUG"):
            print(f"[shoe_dbg] choose_end: toe_z={toe_z:+.3f} "
                  f"quat={q.round(3).tolist()}")
        if abs(toe_z) > 0.04:    # pitched ≥ ~2.3°: grip the raised end
            return 1.0 if toe_z > 0 else -1.0
        # Untilted: default to the HEEL.  The heel is 5.5-6.0 cm wide →
        # ≥1 cm/side descent clearance inside the 8 cm jaw span, whereas
        # the 7.3 cm vamp leaves only ~3.5 mm/side and the descending
        # fingers graze the walls, pitching + sinking the shoe before the
        # close (smoke 61729562: every vamp-end pick of an untilted shoe
        # failed with dz≈0 while every heel-end pick lifted).  Fall back
        # to the vamp only when the heel grip point sits near the shoebox
        # (its descent would land on the box rim).
        origin_xy = self._shoe_pos_quat(shoe)[0][:2]
        toe_dir = self._shoe_toe_dir_xy(shoe)
        box_xy = to_numpy(self.shoebox.entity.get_pos()).ravel()[:2]
        heel_pt = origin_xy - _SHOE_GRIP_HEELWARD_M * toe_dir
        toe_pt = origin_xy + _SHOE_GRIP_TOEWARD_M * toe_dir
        d_heel = float(np.linalg.norm(heel_pt - box_xy))
        d_toe = float(np.linalg.norm(toe_pt - box_xy))
        if d_heel >= 0.16 or d_heel >= d_toe:
            return -1.0
        return 1.0

    def _shoe_grip_xy(self, shoe, grip_end: float) -> np.ndarray:
        """Grip point for the chosen end: the shoe's MESH ORIGIN shifted
        along the length axis onto the vamp (toe end) or the heel counter
        — both closed-top regions clear of the ankle opening.

        The origin, not the live-AABB centre: the mesh origin sits exactly
        at the body centre at sole level (bounds are x/z-symmetric with
        y=0 at the sole), so origin + offset is EXACT in the shoe frame.
        The AABB centre drifts with yaw (a yawed shoe's AABB support
        inflates by up to W·sinθ·cosθ ≈ 3.6 cm) and with tilt, which put
        the "vamp" grip 7.2 cm from the toe tip — over the collar front —
        on yawed seeds (smoke 61725617 seed 1)."""
        origin_xy = self._shoe_pos_quat(shoe)[0][:2]
        off = _SHOE_GRIP_TOEWARD_M if grip_end > 0 else -_SHOE_GRIP_HEELWARD_M
        return origin_xy + off * self._shoe_toe_dir_xy(shoe)

    def _close_until_contact(
        self, arm, arm_tag: str, shoe, bite_m: float = 0.028,
    ) -> float:
        """Stage the gripper close and stop at FIRST CONTACT.

        Commanding the deep _SHOE_CLOSE_TARGET directly leaves a huge PD
        error after contact and the fingers creep ~15 mm/side through the
        thin CoACD walls (they visibly embed in the shoe — user review
        feedback; a finger-stall detector alone misses it because the
        walls never hard-stop the fingers at 35 N).  Contact signal:
        the SHOE MOVES at first real touch — watch its link pose during a
        fast staged close and freeze at the first displacement, then hold
        a fixed retention squeeze ``bite_m`` (total gap reduction across
        both fingers; default 28 mm = ~7 mm/side — the depth that held
        through the lift/carry without a retry in isolated testing).
        Finger-stall detection is kept as a backup (shoe braced against
        something).  A missed grasp reaches the deep floor target and the
        lift-dz guard catches it as before.

        ONE-SHOT: this data-collection pipeline must never retry a grasp
        (repeated attempts corrupt imitation-learning trajectories), so
        there is no escalating-bite mechanism here — a single fixed depth,
        chosen to favor retention over minimal visual penetration. If it
        fails to hold, the pick fails once and stays failed. Returns the
        held target.
        """
        def _gap() -> float:
            q = to_numpy(arm.entity.get_qpos()).ravel()
            return float(q[-2] + q[-1])

        p0 = np.asarray(self._shoe_pos_quat(shoe)[0], dtype=float)
        target = 1.0
        prev_gap = _gap()
        stall = 0
        stage = 0
        touched = False
        while target > _SHOE_CLOSE_TARGET:
            target = max(_SHOE_CLOSE_TARGET, target - 0.06)
            self.set_gripper(target, arm_tag)
            for _ in range(8):
                self.step_sim()
            p = np.asarray(self._shoe_pos_quat(shoe)[0], dtype=float)
            # Motion counts as CONTACT only once the jaws are physically
            # near the body (residual settle/graze motion at wide gaps
            # fired a false touch at 68 mm > the 55-58 mm heel, smoke
            # 61756115 → fingers held 5 mm short of the surface).
            if (
                _gap() < 0.066
                and float(np.linalg.norm(p - p0)) > 0.002
            ):
                touched = True
                break
            gap = _gap()
            stage += 1
            if stage > 2 and prev_gap - gap < 0.0015:
                stall += 1
                if stall >= 2:
                    touched = True
                    break
            else:
                stall = 0
            prev_gap = gap
        # Carry grip: a pure surface pinch cannot lift this shoe — the
        # heel is a horizontal wedge and the clamp squirts it out along
        # the taper (smoke 61756503: lift dz=0 at 5 mm bite/70 N).  A
        # bite keys the fingertips into the wall compliance (prevents the
        # squirt) while staying visually near-surface — the position
        # target still bounds penetration regardless of force, so the
        # fingers can never sink deeper (the old deep target embedded
        # them ~15 mm/side).  Stiff PD + raised cap give the bite real
        # retention.
        hold = max(_SHOE_CLOSE_TARGET, (_gap() - bite_m) / 0.08)
        update_dofs_kp_kv_compat(
            arm.entity, arm._finger_dof_indices,
            kp_value=20000.0, kv_value=400.0,
        )
        update_dofs_force_range_compat(
            arm.entity, arm._finger_dof_indices,
            lower_value=-70.0, upper_value=70.0,
        )
        self.set_gripper(hold, arm_tag)
        for _ in range(self.CLOSE_SETTLE_STEPS):
            self.step_sim()
        if os.environ.get("SHOE_DEBUG"):
            print(f"[shoe_dbg] close: touched={touched} hold={hold:.3f} "
                  f"gap={_gap():.4f}")
        return hold

    def _topdown_pick_shoe(self, shoe, arm_tag: str, label: str) -> bool:
        """Top-down pick at the vamp or heel (whichever end is farther
        from the box) with the jaws squared to the shoe's short axis.

        ONE-SHOT: a single descend → close → lift attempt, no retry of
        any kind. This is a data-collection pipeline — a policy trained
        on trajectories containing "try, fail, back off, try again" would
        learn that pattern, so a real failure here must be a real,
        single, final failure (gripper opened, return False), never
        something the task papers over with a second attempt.

        Leaves the arm lifted with the shoe held on success; on any
        failure the gripper is opened and this returns False."""
        arm = self.robot.get_arm(arm_tag)
        arm._cached_target = np.asarray(
            to_numpy(arm.entity.get_qpos()), dtype=float,
        ).ravel().copy()
        transport_z = self.TABLE_TOP_Z + _SHOE_TRANSPORT_Z_ABOVE_TABLE

        try:
            grip_end = self._choose_grip_end(shoe)
            grip_xy = self._shoe_grip_xy(shoe, grip_end)
            self._grip_yaw_deg = self._object_grip_yaw(shoe)
            origin_z = float(self._shoe_pos_quat(shoe)[0][2])
            above_sole = (
                _SHOE_GRASP_ABOVE_SOLE_TOE_M if grip_end > 0
                else _SHOE_GRASP_ABOVE_SOLE_HEEL_M
            )
            grasp_z = max(
                origin_z + above_sole,
                self.TABLE_TOP_Z + _SHOE_GRASP_TABLE_CLEAR_M,
            )
            z_before = float(self._shoe_pos_quat(shoe)[0][2])

            self.open_gripper(arm_tag)
            above = np.array([grip_xy[0], grip_xy[1], transport_z])
            above_link = tcp_to_link_pose(self._top_down_tcp(above), arm.tcp_offset)
            if self._move_seeded(above_link.to_pose7(), arm_tag) is None:
                print(f"[shoe_pick] {label}: move-above failed")
                return False

            # Descent legs fail FAST: a silently failed screw leaves the
            # TCP at hover height and the close then grips air (dz=0
            # lift-slip with no diagnostic).
            if not self._move_screw(
                np.array([grip_xy[0], grip_xy[1], grasp_z + 0.06]), arm_tag,
            ):
                print(f"[shoe_pick] {label}: hover descent failed")
                return False
            grip_xy = self._shoe_grip_xy(shoe, grip_end)
            if not self._move_screw(
                np.array([grip_xy[0], grip_xy[1], grasp_z]), arm_tag,
            ):
                print(f"[shoe_pick] {label}: grasp descent failed")
                return False
            for _ in range(self.PRE_CLOSE_SETTLE_STEPS):
                self.step_sim()

            self._close_until_contact(arm, arm_tag, shoe)

            if os.environ.get("SHOE_DEBUG"):
                tcp_now = np.asarray(arm.get_ee_pose(), dtype=float)
                q_now = to_numpy(arm.entity.get_qpos()).ravel()
                lo_d, hi_d, c_d = self._shoe_world_aabb(shoe)
                print(f"[shoe_dbg] {label}: end={grip_end:+.0f} "
                      f"grip_xy=({grip_xy[0]:+.3f},{grip_xy[1]:+.3f}) "
                      f"grasp_z={grasp_z:.3f} yaw={self._grip_yaw_deg:+.1f}")
                print(f"[shoe_dbg] {label}: tcp=({tcp_now[0]:+.3f},"
                      f"{tcp_now[1]:+.3f},{tcp_now[2]:.3f}) "
                      f"fingers={q_now[-2:].round(4).tolist()}")
                print(f"[shoe_dbg] {label}: shoe_aabb lo=({lo_d[0]:+.3f},"
                      f"{lo_d[1]:+.3f},{lo_d[2]:.3f}) hi=({hi_d[0]:+.3f},"
                      f"{hi_d[1]:+.3f},{hi_d[2]:.3f}) "
                      f"c=({c_d[0]:+.3f},{c_d[1]:+.3f})")
                sq = np.asarray(self._shoe_pos_quat(shoe)[1])
                print(f"[shoe_dbg] {label}: shoe_quat={sq.round(3).tolist()} "
                      f"toe_dir={self._shoe_toe_dir_xy(shoe).round(3).tolist()}")

            if not self._move_screw(
                np.array([grip_xy[0], grip_xy[1], transport_z]), arm_tag,
            ):
                print(f"[shoe_pick] {label}: lift failed")
                self.open_gripper(arm_tag)
                return False
            for _ in range(self.PRE_CLOSE_SETTLE_STEPS):
                self.step_sim()

            z_after = float(self._shoe_pos_quat(shoe)[0][2])
            if z_after - z_before < _SHOE_MIN_LIFT_DZ:
                print(f"[shoe_pick] {label}: lift slipped "
                      f"(dz={z_after - z_before:.3f} < {_SHOE_MIN_LIFT_DZ})")
                self.open_gripper(arm_tag)
                for _ in range(self.FAILED_GRASP_SETTLE_STEPS):
                    self.step_sim()
                return False
            # Keep this wrist yaw through the drop so the carry doesn't
            # twist the hanging shoe (the drop is yaw-agnostic anyway).
            self._carry_yaw_deg = self._grip_yaw_deg
            return True
        finally:
            self._grip_yaw_deg = 0.0

    def _pick_and_place_shoe(
        self, shoe, arm_tag: str, label: str, slot_idx: int,
    ) -> bool:
        """Single-shoe pick → lift → over shoebox → align with box long
        axis → drop in slot_idx side of the box → retreat.

        ``slot_idx`` (0 or 1) selects which side of the box short axis the
        shoe is dropped on, so two shoes coexist side-by-side in the box.
        Returns True iff every motion stage planned/executed."""
        arm = self.robot.get_arm(arm_tag)
        tcp_offset = arm.tcp_offset

        # Settle then read live shoe pose.
        for _ in range(self.PRE_GRASP_SETTLE_STEPS):
            self.step_sim()
        self.open_gripper(arm_tag)

        # Set shoe-specific finger PD + force_range BEFORE the close.
        update_dofs_kp_kv_compat(
            arm.entity,
            arm._finger_dof_indices,
            kp_value=_SHOE_FINGER_KP,
            kv_value=_SHOE_FINGER_KV,
        )
        update_dofs_force_range_compat(
            arm.entity,
            arm._finger_dof_indices,
            lower_value=-_SHOE_FINGER_FORCE_N,
            upper_value=+_SHOE_FINGER_FORCE_N,
        )

        # Robust top-down pick at the vamp or heel (the end farther from
        # the box; see _SHOE_GRIP_TOEWARD_M block), jaws squared to the
        # shoe's short (width) axis and closed to a width-matched gap.
        # The old YAML grasp poses targeted a TCP z ~16 cm below the shoe
        # top (z-up-frame bug, probe 61305929) so the jaws hit the table
        # and never purchased the shoe.
        if os.environ.get("SHOE_DEBUG"):
            print(f"[shoe_dbg] {label}: plan_success pre-pick: "
                  f"{self.plan_success}")
        if not self._topdown_pick_shoe(shoe, arm_tag, label):
            return False
        if os.environ.get("SHOE_DEBUG"):
            print(f"[shoe_dbg] {label}: plan_success post-pick: "
                  f"{self.plan_success}")

        # ---- Pendant drop.  The end grip makes the shoe hang pitched
        # (roughly vertical), so its horizontal cross-section (~7×9 cm)
        # fits the 16 cm box opening at ANY wrist yaw — no shoe-to-box
        # axis alignment is needed (the old aligned RRT drop also missed
        # the box by 0.20–0.27 m: move_and_execute doesn't FK-verify and
        # mplib silently relaxes the low drop pose, smoke 61721621).
        # Instead reuse the pick's FK-accurate primitives: seeded-IK move
        # above the slot, then a straight screw descent.
        box_pos = to_numpy(self.shoebox.entity.get_pos()).ravel()[:3]
        box_quat = to_numpy(self.shoebox.entity.get_quat()).ravel()[:4]
        R_box = t3d.quaternions.quat2mat(box_quat)
        # Box mesh-x (short axis) direction in world; the URDF was placed
        # with upright_q ∘ Rz(box_yaw) so it lies in the world xy plane.
        box_short_xy = R_box[:2, 0]
        box_short_xy /= max(np.linalg.norm(box_short_xy), 1e-9)

        # Slot offset along the box's SHORT axis so two shoes coexist in
        # the box (one on each half).  slot_idx 0 → -short, 1 → +short.
        offset_sign = -1.0 if int(slot_idx) == 0 else +1.0
        slot_xy = box_pos[:2] + offset_sign * self.SLOT_OFFSET_M * box_short_xy

        above_p = np.array([
            slot_xy[0], slot_xy[1],
            self._box_rim_z + self.APPROACH_HEIGHT,
        ])
        # Carry with the pick's wrist yaw (drop is yaw-agnostic; twisting
        # mid-carry only shakes the hanging shoe).
        self._grip_yaw_deg = float(getattr(self, "_carry_yaw_deg", 0.0))
        try:
            above_link = tcp_to_link_pose(
                self._top_down_tcp(above_p), tcp_offset,
            )
            if self._move_seeded(above_link.to_pose7(), arm_tag) is None:
                # Can't reach the box: release here (guarded failure on the
                # table) rather than return while holding — the next pick's
                # open_gripper would dump the shoe at a random spot.
                print(f"[shoe_pick] {label}: move above box failed — "
                      "releasing in place")
                self.open_gripper(arm_tag, num_steps=self.RELEASE_GRIPPER_STEPS)
                for _ in range(self.RELEASE_SETTLE_STEPS):
                    self.step_sim()
                return False

            # Let the pendant shoe's swing from the long lateral carry
            # (grip point to above the box, often ~0.3-0.4 m) damp out
            # BEFORE measuring its hang geometry / releasing.  Without
            # this the shoe is still swinging when the gripper opens: the
            # TCP tracks the drop target within mm (verified via SHOE_DEBUG)
            # but the shoe itself lands 0.2-0.28 m away — this was the
            # dominant failure mode on the harness (4-5 of 7 fails,
            # eval_final4 / smoke 61810212), NOT a grip-strength issue.
            for _ in range(self.PRE_CLOSE_SETTLE_STEPS * 3):
                self.step_sim()
            if os.environ.get("SHOE_DEBUG"):
                sp = np.asarray(self._shoe_pos_quat(shoe)[0])
                print(f"[shoe_dbg] {label}: shoe_xy after carry+settle="
                      f"({sp[0]:+.3f},{sp[1]:+.3f}) vs slot_xy="
                      f"({slot_xy[0]:+.3f},{slot_xy[1]:+.3f})")

            # NOTE: aiming the descent at the shoe's own live xy (instead
            # of slot_xy) was tried and reverted — the shoe's offset from
            # the TCP is not fixed enough to correct open-loop: a single
            # combined xy+z move barely dragged the shoe at all (smoke
            # 61810688, 3/8), and splitting it into a horizontal
            # correction leg + settle STILL regressed (smoke 61810877,
            # 3/8) because moving again changes the offset again.  Plain
            # slot_xy with the settle above (no correction) is the best
            # validated config (smoke 61810543: 6/8).

            # Screw straight down until the hanging shoe's lowest point is
            # just above the box floor (measured from the live AABB, so the
            # actual hang pitch/slip is accounted for), keeping the
            # fingertips above the rim so the gripper never enters the wall
            # zone.  A failed descent just releases from the hover — the
            # shoe drops in from ~7 cm above the rim.
            tcp_now = np.asarray(arm.get_ee_pose(), dtype=float)
            if os.environ.get("SHOE_DEBUG"):
                print(f"[shoe_dbg] {label}: drop above_p="
                      f"({above_p[0]:+.3f},{above_p[1]:+.3f},{above_p[2]:.3f})"
                      f" tcp_after_above=({tcp_now[0]:+.3f},{tcp_now[1]:+.3f},"
                      f"{tcp_now[2]:.3f})")
            tcp_z_now = float(tcp_now[2])
            shoe_lo_z = float(self._shoe_world_aabb(shoe)[0][2])
            floor_z = self.TABLE_TOP_Z + self.BOX_FLOOR_Z_CLEARANCE
            drop_z = tcp_z_now - (shoe_lo_z - (floor_z + 0.01))
            drop_z = max(drop_z, self._box_rim_z + 0.02)
            screw_ok = self._move_screw(
                np.array([slot_xy[0], slot_xy[1], drop_z]), arm_tag,
            )
            if os.environ.get("SHOE_DEBUG"):
                tcp_d = np.asarray(arm.get_ee_pose(), dtype=float)
                shoe_d = np.asarray(self._shoe_pos_quat(shoe)[0])
                print(f"[shoe_dbg] {label}: drop screw_ok={bool(screw_ok)} "
                      f"target=({slot_xy[0]:+.3f},{slot_xy[1]:+.3f},"
                      f"{drop_z:.3f}) tcp=({tcp_d[0]:+.3f},{tcp_d[1]:+.3f},"
                      f"{tcp_d[2]:.3f}) shoe_xy=({shoe_d[0]:+.3f},"
                      f"{shoe_d[1]:+.3f})")

            # Release.  The width-matched close keeps the fingers OUTSIDE
            # the shoe walls, so opening frees the shoe directly; the
            # raised release force cap is kept as headroom in case a wall
            # stayed compressed against a finger pad during transit.
            update_dofs_force_range_compat(
                arm.entity,
                arm._finger_dof_indices,
                lower_value=-_SHOE_RELEASE_FORCE_N,
                upper_value=+_SHOE_RELEASE_FORCE_N,
            )
            self.open_gripper(arm_tag, num_steps=self.RELEASE_GRIPPER_STEPS)
            # Hold at the drop pose so the shoe settles clear of the
            # fingers under gravity before the arm moves.  Pure physics —
            # no kinematic detach.
            for _ in range(self.RELEASE_SETTLE_STEPS):
                self.step_sim()

            # Retreat straight upward.
            self._move_screw(above_p, arm_tag)
        finally:
            self._grip_yaw_deg = 0.0

        if os.environ.get("SHOE_DEBUG"):
            print(f"[shoe_dbg] {label}: plan_success post-drop: "
                  f"{self.plan_success}")
        for _ in range(self.POST_PLACE_SETTLE_STEPS):
            self.step_sim()
        return True

    # ------------------------------------------------------------------
    # Main task sequence
    # ------------------------------------------------------------------

    def play_once(self) -> bool:
        # Pre-task object settle; excluded from recordings.  NOTE: a
        # settled shoe's hull may legitimately rest 1-2.5 cm below the
        # nominal tabletop (load concentrates on a sharp CoACD sole
        # piece); re-seating cannot fix it (the sink is the contact
        # equilibrium — re-drops deterministically reproduce it), so the
        # pick anchors its grasp height to the live AABB TOP instead.
        with self.suppress_recording():
            for _ in range(self.INITIAL_SETTLE_STEPS):
                self.step_sim()

        if not self.shoes or self.shoebox is None:
            return False

        arm_tag = "right"

        # Sort shoes so the closer-to-box one goes first; leaves more
        # room for the second drop and avoids stacking exactly.
        box_xy = to_numpy(self.shoebox.entity.get_pos()).ravel()[:2]
        order = sorted(
            range(len(self.shoes)),
            key=lambda i: float(np.linalg.norm(
                to_numpy(self.shoes[i].entity.get_pos()).ravel()[:2] - box_xy,
            )),
        )

        any_success = False
        for slot, idx in enumerate(order):
            # slot 0/1 → -/+ side of the box short axis.  First (closer)
            # shoe takes the "-short" slot; second takes the "+short"
            # slot — purely cosmetic, the success criterion is symmetric.
            label = chr(ord("A") + idx)
            ok = self._pick_and_place_shoe(
                self.shoes[idx], arm_tag, label, slot_idx=slot,
            )
            any_success = any_success or ok
        return any_success

    # ------------------------------------------------------------------
    # Success
    # ------------------------------------------------------------------

    def check_success(self) -> bool:
        """Both shoes must end up within `SUCCESS_DIST_XY_M` (xy) of
        the shoebox centre and at z above the box floor."""
        if not self.plan_success:
            return False
        box_pos = to_numpy(self.shoebox.entity.get_pos()).ravel()[:3]
        box_xy = box_pos[:2]
        # Floor of the box is at table_top (URDF placed at table_top + ε).
        box_floor_z = self.TABLE_TOP_Z + self.SUCCESS_Z_FLOOR_MARGIN_M
        all_in = True
        for i, sh in enumerate(self.shoes):
            p = np.asarray(self._shoe_pos_quat(sh)[0])
            dxy = float(np.linalg.norm(p[:2] - box_xy))
            above_floor = p[2] > box_floor_z
            ok = dxy <= self.SUCCESS_DIST_XY_M and above_floor
            if not ok:
                all_in = False
        return all_in
