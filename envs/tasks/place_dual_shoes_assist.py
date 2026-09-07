"""Place dual shoes (assist): cooperative variant of ``place_dual_shoes``.

The avatar places one shoe in the shoebox while the robot places the other
shoe in parallel.  Robot-side shoe handling is inherited from the parent
pure-physics recipe; the avatar uses the sanctioned ``attach_obj`` path via
``AvatarController.pick_and_place``.

Base task summary: Franka picks 2 shoes off the table and drops them into a
stationary open-top shoebox.  Single-arm, no avatar — RoboTwin's
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

Two shoes + the shoebox xy are randomized in an assist-specific strip
near the avatar side of the table.  Keeping the interaction closer to
the avatar edge avoids the avatar body crossing into the table during
natural pick/place motion.
"""

import json
import os
from pathlib import Path

import genesis as gs
import numpy as np
import transforms3d as t3d
import yaml

from ..base_task import BaseTask
from ..manipulation import TopDownPickPlaceMixin
from ..genesis_compat import update_dofs_force_range_compat, update_dofs_kp_kv_compat
from ..utils import (
    ArticulationActor, Pose, load_object, create_primitive, to_numpy,
    ASSETS_PATH, TABLE_HEIGHT,
)
from ..grasp import GraspPose, tcp_to_link_pose, load_grasp_poses
from ..object_catalog import get_entry


# Target object identity from the central catalog (single source).
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
_REGION_X = (-0.22, +0.22)   # 0.44 m wide
_REGION_Y = (-0.38, +0.06)   # 0.44 m wide, shifted -y to keep IK in reach
# Cooperative layout bands.  The two shoes must be reachable by BOTH the
# avatar (shoulder ~y=+0.55, FABRIK reach ~0.75 m) AND the Franka top-down
# pick (base y=-0.65, reach ~0.62 m).  The old single strip (-0.16,+0.06)
# pushed shoes out to ~0.71 m from the robot base, so the +Y half was
# unreachable for the robot and every robot-shoe grasp pre-RRT-failed
# (diag sweep 61293938: assist s2, all interrupt reach fails).  Restrict
# the SHOES to the overlap band (robot ≤0.61 m, avatar ≤0.71 m); the
# shoebox is only ever approached top-down by the robot, so give it its
# own more -Y band that stays solidly in robot reach and clears the shoes.
_ASSIST_SHOE_REGION_Y = (-0.16, -0.04)
# The box is placed INTO by BOTH the avatar and the robot, so it must stay
# in the avatar's reach too.  Diag sweep 61294730 (box at -0.34..-0.18) put
# it past the avatar's place reach -> avatar shoe missed (s0/s3).  Keep the
# box in the avatar-reachable band the baseline used, just trimmed -Y enough
# to stay solidly in robot reach.
_ASSIST_BOX_REGION_Y = (-0.18, -0.08)
# Back-compat alias (unused by the split sampler below; kept so any external
# reference still resolves).
_ASSIST_REGION_Y = _ASSIST_SHOE_REGION_Y
# Shoebox is 26 cm along its long axis; shoes are 22 cm long.  0.24 m
# min sep prevents the box's long edge (or a yawed corner) from
# clipping a shoe spawn — needs a bit more headroom than
# place_bread_in_basket's 0.18 m because of the larger box footprint.
_MIN_SEP_M = 0.24
_ASSIST_MIN_SEP_M = 0.20
# Box yaw randomization range (about world-z, composed with upright_q).
# ±60° lets the box face roughly any direction while keeping its long
# axis broadly across the reach corridor.
_BOX_YAW_RANGE = (-np.pi / 3, +np.pi / 3)

# Shoe mesh contact is soft/thin enough that the generic Franka "strong grip"
# recipe visibly drives the fingers into the shoe.  Keep the gripper forceful
# enough to survive transit, but avoid saturating it as hard as box/cube tasks.
_SHOE_FINGER_KP = 8000.0
_SHOE_FINGER_KV = 250.0
# ADAPTIVE cage close, mirroring the base task (see its comment): deep
# normalized target + moderate force cap so the fingers stall on the real
# CoACD collision surface (which runs several mm narrower than the visual
# mesh) instead of either missing it (fixed width-matched gaps) or
# crushing through the thin walls into the cavity (the old 0.04 target at
# 80 N — the "finger stuck inside the shoe" bug).
_SHOE_FINGER_FORCE_N = 35.0
_SHOE_CLOSE_TARGET = 0.30
_SHOE_MIN_LIFT_DZ = 0.05
_SHOE_TRANSPORT_Z_ABOVE_TABLE = 0.22
# Grip-point offsets along the shoe length axis (mirrors the base task;
# see its comment block): the ankle collar + opening span the middle-rear
# of the shoe, so the pick grips the vamp (toe end) or the heel counter —
# whichever end is FARTHER from the shoebox (a toe/heel resting against
# the box makes that end's descent land on the box rim).
_SHOE_GRIP_TOEWARD_M = 0.045
_SHOE_GRIP_HEELWARD_M = 0.068
# Fingertip height above the shoe's mesh-origin z (= sole plane at the
# body centre) + table clamp (mirrors the base task: settled shoes can
# sink 1.7-2.8 cm into the tabletop and their AABB drifts with tilt/yaw,
# so anchor the grasp to the origin, which rides with the shoe exactly).
_SHOE_GRASP_ABOVE_SOLE_TOE_M = 0.020
_SHOE_GRASP_ABOVE_SOLE_HEEL_M = 0.028
_SHOE_GRASP_TABLE_CLEAR_M = 0.015


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
    # v2: atomic, geometry-only OBJ/URDF writes (race-safe — see base
    # task place_dual_shoes._ensure_shoebox_urdf for the rationale).
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

    if not (visual_obj.exists() and visual_obj.stat().st_size > 0):
        import trimesh
        asset_dir = ASSETS_PATH / "objects" / _BOX_ID
        glb = asset_dir / "visual" / "base0.glb"
        mesh = trimesh.load(str(glb), force="mesh")
        # Per-axis pre-scale to meters so URDF morph scale=1.0.
        mesh.apply_scale(scale_vec)
        # Geometry-only OBJ (no sibling .mtl / texture PNG) + atomic
        # rename: a concurrent reader never sees a partial mesh or a
        # half-written texture. See base task for full rationale.
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
    max_outer: int = 64,
    region_x: tuple[float, float] = _REGION_X,
    region_y: tuple[float, float] = _REGION_Y,
    min_sep: float = _MIN_SEP_M,
):
    """Rejection-sample n xy positions inside the region with pairwise
    minimum separation."""
    for _outer in range(max_outer):
        positions: list[tuple[float, float]] = []
        ok = True
        for _ in range(n):
            placed = False
            for _ in range(64):
                x = float(rng.uniform(*region_x))
                y = float(rng.uniform(*region_y))
                if all(
                    (x - px) ** 2 + (y - py) ** 2 >= min_sep ** 2
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
    xs = np.linspace(region_x[0] + 0.05, region_x[1] - 0.05, n)
    ys = np.linspace(region_y[0] + 0.05, region_y[1] - 0.05, n)
    return [(float(xs[i]), float(ys[i])) for i in range(n)]


def _load_extra_manual_shoe_grasps() -> list[GraspPose]:
    """Load annotator grasps from the manual-grasps workspace.

    The task's normal loader reads ``assets/objects/041_shoe``.  This
    helper prepends the actively edited annotation file so smoke videos
    test the latest manual pose without changing global grasp loading.
    """
    path = ASSETS_PATH / "manual_grasps" / "grasp_poses_041_shoe_franka_manual.yml"
    if not path.exists():
        return []
    data = yaml.safe_load(open(path)) or {}
    grasps = []
    for item in data.get("grasps", []):
        grasps.append(GraspPose(
            name=item.get("name", "manual_extra"),
            position=item["position"],
            quaternion=item["quaternion"],
            pre_distance=item.get("pre_distance", 0.12),
            source=item.get("source", "manual"),
            scale_frame=item.get("scale_frame", "mesh_unit"),
            category=item.get("category"),
            kinematic_pin=item.get("kinematic_pin", False),
        ))
    return grasps


class PlaceDualShoesAssist(TopDownPickPlaceMixin, BaseTask):
    """Avatar and robot cooperatively place one shoe each in the shoebox."""

    INSTRUCTION = (
        "the human will put one shoe in the shoebox; the robot puts the "
        "other shoe in the shoebox — work in parallel"
    )
    OBJECT_SET = ["041_shoe"]
    use_avatar = True

    # Avatar at +Y side of the table, facing -Y toward the table.
    AVATAR_BASE_ROT = np.array(
        [[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=np.float64,
    )
    avatar_init_pos = np.array([0.0, 0.55, -0.18])
    avatar_init_rot = AVATAR_BASE_ROT

    LIFT_HEIGHT = 0.20         # m above grasp before transit
    APPROACH_HEIGHT = 0.22     # m above box rim before lowering
    DROP_HEIGHT = 0.08         # m above box rim at release
    # Box URDF is placed at table_top + this clearance (the fork hardcodes
    # the placement; the constant is needed by the pendant-drop depth calc).
    BOX_FLOOR_Z_CLEARANCE = 0.001

    AVATAR_APPROACH_FRAMES = 30
    AVATAR_TRANSPORT_FRAMES = 45
    AVATAR_RETRACT_FRAMES = 30
    AVATAR_FRAME_REPEAT = 6
    AVATAR_REFINE_ITERS = 1
    AVATAR_DROP_ABOVE_RIM = 0.10
    AVATAR_NATURAL_BODY_MARGIN = 0.40
    AVATAR_NATURAL_YAW_LIMIT_DEG = 30.0
    AVATAR_NATURAL_BODY_Y_BOUNDS = (0.22, 0.70)
    AVATAR_NATURAL_SETTLE_STEPS = 20

    # Eval/testbed mode: policy calls take_action(action)->obs while the
    # avatar starts its shoe assist once at a randomized policy step.
    EVAL_TRIGGER_STEP_MIN = 20
    EVAL_TRIGGER_STEP_MAX = 60

    # Both shoes must end up inside the box footprint for success.
    # Shoebox interior is ~16 × 25 cm.  Half the diagonal ≈ 0.149 m;
    # 0.18 m gives some slack for shoes that drop near the rim.
    SUCCESS_DIST_XY_M = 0.18
    # Shoe must end up above the box floor (within 1 cm tolerance).
    # Shoes resting on the box floor sit ~5 cm above the floor (their
    # half-height); a stricter z-floor check would reject those.
    SUCCESS_Z_FLOOR_MARGIN_M = -0.01

    recording_camera_pos = [1.1, 0.0, 1.95]
    recording_camera_lookat = [0.0, 0.0, 0.85]

    def __init__(self, config: dict = None):
        cfg = dict(config or {})
        cfg.setdefault("robot_type", "franka")
        cfg.setdefault("track_avatar_collision", False)
        super().__init__(cfg)
        self.resolve_target_object()   # registry/override hook (size-1 set)
        self._eval_trigger_step = None
        self._eval_avatar_started = False
        self._eval_avatar_failed = False
        self._policy_step_count = 0
        self._blind_robot_order = None
        self._blind_robot_first_idx = None

    def reset(self, seed: int = 0):
        obs = super().reset(seed=seed)
        self._policy_step_count = 0
        self._eval_avatar_started = False
        self._eval_avatar_failed = False
        self._blind_robot_order = self._sample_blind_robot_order()
        self._blind_robot_first_idx = (
            int(self._blind_robot_order[0]) if self._blind_robot_order else None
        )
        if self.avatar is not None and bool(self.config.get("eval_mode", False)):
            self._eval_trigger_step = 0
            self._eval_avatar_started = True
            if not self._start_avatar_assist():
                self._eval_avatar_failed = True
        else:
            self._eval_trigger_step = None
        return obs

    def take_action(self, action, action_type: str = "qpos"):
        if self.avatar is not None and bool(self.config.get("eval_mode", False)):
            self._eval_step_avatar()
        return super().take_action(action, action_type=action_type)

    def _eval_step_avatar(self) -> None:
        self._policy_step_count += 1
        if self._eval_avatar_started or self._eval_avatar_failed:
            return
        if self._eval_trigger_step is None:
            return
        if self._policy_step_count < self._eval_trigger_step:
            return
        self._eval_avatar_started = True
        if self._start_avatar_assist():
            pass
        else:
            self._eval_avatar_failed = True

    # ------------------------------------------------------------------
    # Scene setup
    # ------------------------------------------------------------------

    def _create_table(self, table_height=TABLE_HEIGHT):
        self.TABLE_HEIGHT = table_height
        self.TABLE_THICKNESS = 0.05
        self.TABLE_TOP_Z = table_height + self.TABLE_THICKNESS / 2
        self.table = create_primitive(
            self.scene, "box", Pose(p=[0.0, -0.35, table_height]),
            size={"half_size": (0.6, 0.55, self.TABLE_THICKNESS / 2)},
            color=(0.8, 0.75, 0.65), is_static=True,
        )
        self._remember_table_bounds(half_size=(0.6, 0.55), center_xy=(0.0, -0.35))
        leg_h = table_height - self.TABLE_THICKNESS / 2
        for x, y in [(-0.55, -0.85), (0.55, -0.85), (-0.55, 0.15), (0.55, 0.15)]:
            create_primitive(
                self.scene, "cylinder",
                Pose(p=[x, y, leg_h / 2]),
                size={"radius": 0.025, "half_length": leg_h / 2},
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

    def load_actors(self):
        # mesh-y maps to world-z under upright_q; the shoe and shoebox
        # GLBs are both y-up.
        upright_q = np.array([0.70710678, 0.70710678, 0.0, 0.0], dtype=np.float64)
        table_top = self.TABLE_TOP_Z

        rng = np.random.RandomState(np.random.randint(0, 2 ** 31 - 1))
        positions = self._sample_layout_positions(rng)
        (box_x, box_y), (shoe_a_xy), (shoe_b_xy) = (
            positions[0], positions[1], positions[2],
        )

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
        self.box_pose = Pose([box_x, box_y, table_top + 0.001], box_q)
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
            material=gs.materials.Rigid(friction=2.0),
        )
        self.shoebox = ArticulationActor(box_entity, {}, _BOX_ID)
        # Box rim in world: floor at table_top + ε, walls of height box_h.
        self._box_rim_z = float(table_top + 0.001 + box_h)
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
        for (sx, sy) in (shoe_a_xy, shoe_b_xy):
            yaw = float(rng.uniform(-np.pi, np.pi))
            yaw_q = t3d.quaternions.axangle2quat(
                [0.0, 0.0, 1.0], yaw, is_normalized=True,
            )
            pose_q = t3d.quaternions.qmult(yaw_q, upright_q)
            pose_q /= np.linalg.norm(pose_q)
            # Spawn the sole ~1 cm above the table (origin↔sole ≈0.028 m) so it
            # drops cleanly.  origin-at-table_top+0.02 sank the sole into the
            # table (height probe 61312848); lifting by 0.5*shoe_h instead
            # dropped it ~6 cm and tipped it, regressing the grasp (run5).
            spawn_z = table_top + 0.04
            pose = Pose([sx, sy, spawn_z], pose_q)
            actor = load_object(
                self.scene, pose, _SHOE_ID, model_id=_SHOE_MODEL_ID,
                convex=True, is_static=False,
                friction=5.0,
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

    def _sample_layout_positions(self, rng):
        # Box from the (avatar+robot)-reachable central band; shoes from the
        # overlap band.  Reject shoe samples within _ASSIST_MIN_SEP_M of the
        # box or each other so the shoebox lip never clips a shoe spawn.
        for _outer in range(64):
            box = (
                float(rng.uniform(*_REGION_X)),
                float(rng.uniform(*_ASSIST_BOX_REGION_Y)),
            )
            placed = [box]
            ok = True
            for _ in range(2):
                got = False
                for _ in range(64):
                    cand = (
                        float(rng.uniform(*_REGION_X)),
                        float(rng.uniform(*_ASSIST_SHOE_REGION_Y)),
                    )
                    if all(
                        (cand[0] - px) ** 2 + (cand[1] - py) ** 2
                        >= _ASSIST_MIN_SEP_M ** 2
                        for px, py in placed
                    ):
                        placed.append(cand)
                        got = True
                        break
                if not got:
                    ok = False
                    break
            if ok:
                self._assign_shoes_by_side(placed)
                return placed
        # Fallback: box centered, shoes spread to opposite x-sides so the two
        # agents work apart.
        placed = [
            (0.0, float(np.mean(_ASSIST_BOX_REGION_Y))),
            (_REGION_X[0] + 0.05, float(np.mean(_ASSIST_SHOE_REGION_Y))),
            (_REGION_X[1] - 0.05, float(np.mean(_ASSIST_SHOE_REGION_Y))),
        ]
        self._assign_shoes_by_side(placed)
        return placed

    def _assign_shoes_by_side(self, placed) -> None:
        """Give the avatar the +x shoe and the robot the -x shoe so the two
        agents reach into opposite halves of the table and their arms do not
        cross over the shared shoebox (diag sweep 61294730 s2: link6 hit the
        avatar torso when both worked the same column).  ``placed`` is
        ``[box_xy, shoe0_xy, shoe1_xy]``."""
        shoe0_x = placed[1][0]
        shoe1_x = placed[2][0]
        # idx of the +x (avatar-side) shoe; ties -> idx 0.
        avatar_idx = 0 if shoe0_x >= shoe1_x else 1
        self._avatar_shoe_idx = int(avatar_idx)
        self._robot_shoe_idx = 1 - int(avatar_idx)

    def _sample_blind_robot_order(self) -> list[int]:
        n = len(getattr(self, "shoes", []) or [])
        if n <= 0:
            return []
        # Blind expert must not derive its first shoe from eval_seed, since
        # the avatar target is also sampled from the seeded task layout.  Use
        # the task RNG stream after layout sampling instead: reproducible, but
        # not an explicit second seeded channel that can encode the avatar's
        # chosen shoe.
        order = np.random.permutation(n)
        return [int(i) for i in order]

    # ------------------------------------------------------------------
    # Pick + place primitive
    # ------------------------------------------------------------------

    # In-box slot offset (along the box's mesh-x = SHORT axis, world-frame
    # rotated by box yaw).  Keep both targets closer to center than the old
    # 4.5 cm slots; seed 1 showed edge drops can leave a shoe outside.
    SLOT_OFFSET_M = 0.030
    AVATAR_SLOT_OFFSET_M = 0.015

    def _box_slot_xy(
        self, slot_idx: int, slot_offset: float | None = None,
    ) -> np.ndarray:
        box_pos = to_numpy(self.shoebox.entity.get_pos()).ravel()[:3]
        box_quat = to_numpy(self.shoebox.entity.get_quat()).ravel()[:4]
        R_box = t3d.quaternions.quat2mat(box_quat)
        box_short_xy = R_box[:, 0][:2]
        box_short_xy /= max(np.linalg.norm(box_short_xy), 1e-9)
        offset_sign = -1.0 if int(slot_idx) == 0 else +1.0
        offset_m = self.SLOT_OFFSET_M if slot_offset is None else float(slot_offset)
        return (
            np.array([box_pos[0], box_pos[1]])
            + offset_sign * offset_m * box_short_xy
        )

    def _shoe_world_center(self, shoe) -> np.ndarray:
        origin, q = self._shoe_pos_quat(shoe)
        R = t3d.quaternions.quat2mat(q)
        scale = self._shoe_scale
        if isinstance(scale, np.ndarray):
            offset = self._shoe_center_mesh * scale
        else:
            offset = self._shoe_center_mesh * float(scale)
        return origin + R @ offset

    def _slow_avatar_pick_place(self, repeat: int) -> None:
        if repeat <= 1:
            return
        mm = self.avatar.motion_modules.get("pick_and_place")
        if mm is None or not mm.frames:
            return
        repeated = []
        for frame in mm.frames:
            repeated.extend([frame] * int(repeat))
        mm.frames = repeated
        mm.attach_frame = int(mm.attach_frame) * int(repeat)
        mm.detach_frame = int(mm.detach_frame) * int(repeat)

    # ------------------------------------------------------------------
    # Robust top-down pick (live-AABB grip point + object-yaw alignment).
    # Mirrors the base task place_dual_shoes; this is a standalone fork so
    # the methods are duplicated by hand.
    # ------------------------------------------------------------------

    def _top_down_tcp(self, pos, yaw: float = 0.0):
        # Accepts (and ignores) the mixin's own `yaw` kwarg — see the
        # base task's _top_down_tcp docstring; _move_screw/_move_cartesian
        # unconditionally pass yaw=tcp_yaw so the signature must accept it.
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
        """Live base pose from the base LINK (mirrors the base task: the
        entity-level pos/quat can stay frozen at the avatar's detach-time
        values while the body physically settles)."""
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
        # BUG FIXED (mirrors the base task — see its docstring): tcp_yaw
        # must be "90 - theta", not "theta - 90". _top_down_tcp composes
        # yaw with the REFLECTION _R_DOWN (det=-1), which inverts the
        # effective rotation direction, so "theta - 90" only aligned the
        # jaws with the shoe's width at yaw multiples of 90° and grabbed
        # squarely along the LONG axis at 45/135/etc (user report).
        # Wrapped to [-90, 90): jaws are 180°-symmetric and raw values
        # like -210° made the seeded IK pick contorted wrist branches.
        q = np.asarray(self._shoe_pos_quat(shoe)[1], dtype=float)
        R = t3d.quaternions.quat2mat(q)
        x_axis = R @ np.array([1.0, 0.0, 0.0])
        theta = float(np.degrees(np.arctan2(x_axis[1], x_axis[0])))
        return ((180.0 - theta) % 180.0) - 90.0

    def _shoe_toe_dir_xy(self, shoe) -> np.ndarray:
        """Unit world-xy heel→toe direction (mesh +z); zero if vertical."""
        q = np.asarray(self._shoe_pos_quat(shoe)[1], dtype=float)
        R = t3d.quaternions.quat2mat(q)
        toe = R @ np.array([0.0, 0.0, 1.0])
        n = float(np.linalg.norm(toe[:2]))
        if n < 1e-6:
            return np.zeros(2)
        return toe[:2] / n

    def _choose_grip_end(self, shoe) -> float:
        """+1.0 → grip the toe/vamp end, -1.0 → grip the heel end: the
        RAISED end of a pitched shoe, else default HEEL (wider descent
        clearance — see base task), vamp only when the heel grip point
        sits near the shoebox."""
        q = np.asarray(self._shoe_pos_quat(shoe)[1], dtype=float)
        R = t3d.quaternions.quat2mat(q)
        toe_z = float(R[2, 2])
        if abs(toe_z) > 0.04:
            return 1.0 if toe_z > 0 else -1.0
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
        """Grip point on the chosen closed-top end (vamp or heel),
        anchored to the shoe's MESH ORIGIN (= body centre at sole level;
        exact in the shoe frame — the AABB centre drifts with yaw/tilt,
        see base task)."""
        origin_xy = self._shoe_pos_quat(shoe)[0][:2]
        off = _SHOE_GRIP_TOEWARD_M if grip_end > 0 else -_SHOE_GRIP_HEELWARD_M
        return origin_xy + off * self._shoe_toe_dir_xy(shoe)

    def _close_until_contact(
        self, arm, arm_tag: str, shoe, bite_m: float = 0.028,
    ) -> float:
        """Stage the close and stop at first contact, detected by SHOE
        MOTION (mirrors the base task: thin CoACD walls never hard-stop
        the fingers, so finger-stall alone misses contact and the
        fingertips visibly embed).  Fixed ~7 mm/side retention, ONE-SHOT
        — no escalation on retry (this is a data-collection pipeline;
        retries are forbidden, see _topdown_pick_shoe)."""
        def _gap() -> float:
            q = to_numpy(arm.entity.get_qpos()).ravel()
            return float(q[-2] + q[-1])

        p0 = np.asarray(self._shoe_pos_quat(shoe)[0], dtype=float)
        target = 1.0
        prev_gap = _gap()
        stall = 0
        stage = 0
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
                break
            gap = _gap()
            stage += 1
            if stage > 2 and prev_gap - gap < 0.0015:
                stall += 1
                if stall >= 2:
                    break
            else:
                stall = 0
            prev_gap = gap
        # Carry grip: a pure surface pinch cannot lift this shoe — the
        # heel is a horizontal wedge and the clamp squirts it out along
        # the taper (smoke 61756503: lift dz=0 at 5 mm bite/70 N).  An
        # ~8 mm/side bite keys the fingertips into the wall compliance
        # (prevents the squirt) while staying visually near-surface —
        # the position target still bounds penetration regardless of
        # force, so the fingers can never sink deeper (the old deep
        # target embedded them ~15 mm/side).  Stiff PD + raised cap give
        # the bite real retention.
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
        for _ in range(80):
            self.step_sim()
        return hold

    def _topdown_pick_shoe(self, shoe, arm_tag: str, label: str) -> bool:
        """ONE-SHOT pick: single descend → close → lift, no retry of any
        kind (this is a data-collection pipeline; see the base task's
        docstring for why retries are forbidden here)."""
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
                print(f"[shoe_assist] {label}: move-above failed")
                return False

            # Descent legs fail FAST (a silently failed screw leaves the
            # TCP at hover and the close grips air — dz=0 with no log).
            if not self._move_screw(
                np.array([grip_xy[0], grip_xy[1], grasp_z + 0.06]), arm_tag,
            ):
                print(f"[shoe_assist] {label}: hover descent failed")
                return False
            grip_xy = self._shoe_grip_xy(shoe, grip_end)
            if not self._move_screw(
                np.array([grip_xy[0], grip_xy[1], grasp_z]), arm_tag,
            ):
                print(f"[shoe_assist] {label}: grasp descent failed")
                return False
            for _ in range(30):
                self.step_sim()

            self._close_until_contact(arm, arm_tag, shoe)

            if not self._move_screw(
                np.array([grip_xy[0], grip_xy[1], transport_z]), arm_tag,
            ):
                print(f"[shoe_assist] {label}: lift failed")
                self.open_gripper(arm_tag)
                return False
            for _ in range(30):
                self.step_sim()

            z_after = float(self._shoe_pos_quat(shoe)[0][2])
            if z_after - z_before < _SHOE_MIN_LIFT_DZ:
                print(f"[shoe_assist] {label}: lift slipped "
                      f"(dz={z_after - z_before:.3f} < {_SHOE_MIN_LIFT_DZ})")
                self.open_gripper(arm_tag)
                for _ in range(30):
                    self.step_sim()
                return False
            # Keep this wrist yaw through the drop (drop is yaw-agnostic;
            # twisting mid-carry only shakes the hanging shoe).
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

        # Settle before reading the live shoe geometry.
        for _ in range(30):
            self.step_sim()

        self.open_gripper(arm_tag)

        # Shoe-specific finger PD + force_range BEFORE the close.
        update_dofs_kp_kv_compat(
            arm.entity, arm._finger_dof_indices,
            kp_value=_SHOE_FINGER_KP, kv_value=_SHOE_FINGER_KV,
        )
        update_dofs_force_range_compat(
            arm.entity, arm._finger_dof_indices,
            lower_value=-_SHOE_FINGER_FORCE_N, upper_value=+_SHOE_FINGER_FORCE_N,
        )

        # Robust top-down pick at the vamp or heel (the end farther from
        # the box), jaws squared to the shoe's short axis and closed to a
        # width-matched gap.  Replaces the YAML grasp poses that targeted
        # a TCP z ~16 cm below the shoe top (z-up frame bug, probe
        # 61305929) so the jaws hit the table and never purchased.
        if os.environ.get("SHOE_DEBUG"):
            print(f"[shoe_dbg] {label}: plan_success pre-pick: "
                  f"{self.plan_success}")
        if not self._topdown_pick_shoe(shoe, arm_tag, label):
            return False
        if os.environ.get("SHOE_DEBUG"):
            print(f"[shoe_dbg] {label}: plan_success post-pick: "
                  f"{self.plan_success}")

        # ---- Pendant drop (mirrors the base task).  The end grip makes
        # the shoe hang roughly vertical, so its horizontal cross-section
        # fits the box opening at any wrist yaw — no axis alignment; and
        # the FK-accurate seeded-IK/screw primitives replace the raw RRT
        # drop whose endpoint mplib silently relaxed 0.20–0.27 m off the
        # box (smoke 61721621).
        slot_xy = self._box_slot_xy(slot_idx)

        above_p = np.array([
            slot_xy[0], slot_xy[1],
            self._box_rim_z + self.APPROACH_HEIGHT,
        ])
        self._grip_yaw_deg = float(getattr(self, "_carry_yaw_deg", 0.0))
        try:
            above_link = tcp_to_link_pose(
                self._top_down_tcp(above_p), tcp_offset,
            )
            if self._move_seeded(above_link.to_pose7(), arm_tag) is None:
                print(f"[shoe_assist] {label}: move above box failed — "
                      "releasing in place")
                self.open_gripper(arm_tag, num_steps=260)
                for _ in range(220):
                    self.step_sim()
                return False

            # Let the pendant shoe's swing from the lateral carry damp out
            # before releasing (mirrors the base task — the TCP tracks the
            # drop target within mm but an un-damped shoe lands 0.2-0.28m
            # away; this was the dominant eval failure mode).
            for _ in range(90):
                self.step_sim()

            # NOTE: aiming the descent at the shoe's own live xy (instead
            # of slot_xy) was tried and reverted (mirrors the base task —
            # see its comment): the shoe's offset from the TCP isn't
            # fixed enough to correct open-loop, and re-moving to correct
            # it just perturbs it again.  Plain slot_xy with the settle
            # above is the best validated config.

            # Screw straight down until the hanging shoe's lowest point is
            # just above the box floor (live AABB), fingertips kept above
            # the rim.  A failed descent releases from the hover instead.
            tcp_z_now = float(np.asarray(arm.get_ee_pose(), dtype=float)[2])
            shoe_lo_z = float(self._shoe_world_aabb(shoe)[0][2])
            floor_z = self.TABLE_TOP_Z + self.BOX_FLOOR_Z_CLEARANCE
            drop_z = tcp_z_now - (shoe_lo_z - (floor_z + 0.01))
            drop_z = max(drop_z, self._box_rim_z + 0.02)
            self._move_screw(
                np.array([slot_xy[0], slot_xy[1], drop_z]), arm_tag,
            )

            # Release.  The width-matched close keeps the fingers OUTSIDE
            # the shoe walls, so open directly.  Pure physics — no
            # kinematic detach.
            self.open_gripper(arm_tag, num_steps=260)
            for _ in range(220):
                self.step_sim()

            # Retreat straight upward.
            self._move_screw(above_p, arm_tag)
        finally:
            self._grip_yaw_deg = 0.0

        if os.environ.get("SHOE_DEBUG"):
            print(f"[shoe_dbg] {label}: plan_success post-drop: "
                  f"{self.plan_success}")
        for _ in range(60):
            self.step_sim()
        return True

    # ------------------------------------------------------------------
    # Main task sequence
    # ------------------------------------------------------------------

    def _start_avatar_assist(self) -> bool:
        if not self.shoes or self.shoebox is None or self.avatar is None:
            return False

        avatar_idx = int(getattr(self, "_avatar_shoe_idx", 0))
        avatar_shoe = self.shoes[avatar_idx]
        avatar_label = chr(ord("A") + avatar_idx)
        avatar_slot = 0


        pick_pos = self._shoe_world_center(avatar_shoe)
        slot_xy = self._box_slot_xy(avatar_slot, self.AVATAR_SLOT_OFFSET_M)
        place_pos = np.array([
            slot_xy[0],
            slot_xy[1],
            self._box_rim_z + self.AVATAR_DROP_ABOVE_RIM,
        ])
        self.avatar.pick_and_place(
            pick_pos=pick_pos,
            place_pos=place_pos,
            attach_obj=avatar_shoe.entity,
            hand_id=None,
            approach_frames=self.AVATAR_APPROACH_FRAMES,
            transport_frames=self.AVATAR_TRANSPORT_FRAMES,
            retract_frames=self.AVATAR_RETRACT_FRAMES,
            natural=True,
            body_margin=self.AVATAR_NATURAL_BODY_MARGIN,
            yaw_limit_deg=self.AVATAR_NATURAL_YAW_LIMIT_DEG,
            body_y_bounds=self.AVATAR_NATURAL_BODY_Y_BOUNDS,
            settle_steps=self.AVATAR_NATURAL_SETTLE_STEPS,
            refine_iters=self.AVATAR_REFINE_ITERS,
        )
        self._slow_avatar_pick_place(self.AVATAR_FRAME_REPEAT)
        self.avatar_collided = False
        self.avatar_collision_log = []
        return True

    def play_once(self) -> bool:
        with self.suppress_recording():  # pre-task settle, excluded from recordings
            for _ in range(60):
                self.step_sim()

        if not self.shoes or self.shoebox is None:
            return False

        arm_tag = "right"
        if self.avatar is None:
            return self._play_robot_only_fallback(arm_tag)

        avatar_idx = int(getattr(self, "_avatar_shoe_idx", 0))
        robot_idx = int(getattr(self, "_robot_shoe_idx", 1 - avatar_idx))
        robot_shoe = self.shoes[robot_idx]
        robot_label = chr(ord("A") + robot_idx)
        robot_slot = 1


        # ---- Avatar pick + place (non-blocking) -----------------------
        if not self._start_avatar_assist():
            return False

        # ---- Robot pick + place (pure-physics primitive) --------------
        try:
            ok_robot = self._pick_and_place_shoe(
                robot_shoe, arm_tag, robot_label, slot_idx=robot_slot,
            )
        except Exception as e:
            print(f"[shoe_assist] {robot_label}: pick_and_place raised: {e!r}")
            ok_robot = False

        if not self.avatar.spare():
            while not self.avatar.spare():
                self.step_sim()
        for _ in range(150):
            self.step_sim()
        return ok_robot

    def _shoe_in_box(self, shoe) -> bool:
        box_pos = to_numpy(self.shoebox.entity.get_pos()).ravel()[:3]
        p = np.asarray(self._shoe_pos_quat(shoe)[0])
        dxy = float(np.linalg.norm(p[:2] - box_pos[:2]))
        return bool(
            dxy <= self.SUCCESS_DIST_XY_M
            and p[2] > self.TABLE_TOP_Z + self.SUCCESS_Z_FLOOR_MARGIN_M
        )

    def play_blind_once(self) -> bool:
        for _ in range(60):
            self.step_sim()

        if not self.shoes or self.shoebox is None:
            return False

        arm_tag = "right"
        if self.avatar is None:
            return self._play_robot_only_fallback(arm_tag)

        ok_any = False
        order = list(self._blind_robot_order or self._sample_blind_robot_order())
        if not order:
            return False
        self._blind_robot_order = order
        self._blind_robot_first_idx = int(order[0])
        # Blind expert commits to one shoe.  It must not recover by watching
        # which shoe the avatar already placed and then choosing the other.
        idx = int(order[0])
        shoe = self.shoes[idx]
        label = chr(ord("A") + idx)
        try:
            ok_any = bool(self._pick_and_place_shoe(
                shoe, arm_tag, label, slot_idx=0,
            ))
        except Exception as e:
            print(f"[shoe_assist] {label}: blind pick_and_place raised: {e!r}")
            ok_any = False
        if self.avatar is not None:
            tail = 0
            while not self.avatar.spare() and tail < 1200:
                self.step_sim()
                tail += 1
        for _ in range(150):
            self.step_sim()
        if os.environ.get("SHOE_DEBUG"):
            print(f"[shoe_dbg] blind end: ok_any={ok_any} "
                  f"plan_success={self.plan_success}")
        return ok_any

    def _play_robot_only_fallback(self, arm_tag: str) -> bool:
        """Serial parent-style fallback for debug configs with no avatar."""
        box_xy = to_numpy(self.shoebox.entity.get_pos()).ravel()[:2]
        order = sorted(
            range(len(self.shoes)),
            key=lambda i: float(np.linalg.norm(
                to_numpy(self.shoes[i].entity.get_pos()).ravel()[:2] - box_xy,
            )),
        )
        any_success = False
        for slot, idx in enumerate(order):
            label = chr(ord("A") + idx)
            ok = self._pick_and_place_shoe(
                self.shoes[idx], arm_tag, label, slot_idx=slot,
            )
            any_success = any_success or ok
        return any_success

    # ------------------------------------------------------------------
    # Success
    # ------------------------------------------------------------------

    def _shoe_in_box_metrics(self) -> dict:
        metrics = {}
        if not self.shoes or self.shoebox is None:
            return metrics

        box_pos = to_numpy(self.shoebox.entity.get_pos()).ravel()[:3]
        box_xy = box_pos[:2]
        box_floor_z = self.TABLE_TOP_Z + self.SUCCESS_Z_FLOOR_MARGIN_M
        metrics["shoebox_pos"] = box_pos.tolist()
        metrics["shoebox_floor_z_threshold"] = float(box_floor_z)
        metrics["success_dist_xy_threshold"] = float(self.SUCCESS_DIST_XY_M)

        all_in = True
        for i, sh in enumerate(self.shoes):
            label = chr(ord("A") + i)
            p = np.asarray(self._shoe_pos_quat(sh)[0])
            dxy = float(np.linalg.norm(p[:2] - box_xy))
            above_floor = bool(p[2] > box_floor_z)
            in_box = bool(dxy <= self.SUCCESS_DIST_XY_M and above_floor)
            all_in = all_in and in_box
            prefix = f"shoe_{label.lower()}"
            metrics[f"{prefix}_pos"] = p.tolist()
            metrics[f"{prefix}_dist_xy"] = dxy
            metrics[f"{prefix}_above_floor"] = above_floor
            metrics[f"{prefix}_in_box"] = in_box
        metrics["all_shoes_in_box"] = bool(all_in)
        return metrics

    def check_success(self) -> bool:
        """Both shoes must end up within `SUCCESS_DIST_XY_M` (xy) of
        the shoebox centre and at z above the box floor."""
        if (self.avatar_collision_checker is not None
                and self.avatar_collided):
            return False
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

    def evaluate(self) -> dict:
        metrics = super().evaluate()
        metrics.update(self._shoe_in_box_metrics())
        avatar_idx = int(getattr(self, "_avatar_shoe_idx", -1))
        robot_idx = int(getattr(self, "_robot_shoe_idx", -1))
        blind_order = list(getattr(self, "_blind_robot_order", []) or [])
        blind_first = getattr(self, "_blind_robot_first_idx", None)
        metrics.update({
            "assist_avatar_shoe_idx": avatar_idx,
            "assist_robot_shoe_idx": robot_idx,
            "blind_robot_order": blind_order,
            "blind_robot_first_idx": (
                int(blind_first) if blind_first is not None else -1
            ),
            "blind_same_first_as_avatar": (
                bool(int(blind_first) == avatar_idx)
                if blind_first is not None and avatar_idx >= 0
                else False
            ),
        })
        if bool(self.config.get("eval_mode", False)):
            metrics.update({
                "eval_mode": True,
                "eval_trigger_step": self._eval_trigger_step,
                "eval_policy_step_count": int(self._policy_step_count),
                "eval_avatar_started": bool(self._eval_avatar_started),
                "eval_avatar_failed": bool(self._eval_avatar_failed),
            })
        return metrics
