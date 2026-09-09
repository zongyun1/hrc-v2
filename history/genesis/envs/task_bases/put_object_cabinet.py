"""Put-object-in-cabinet (single Franka, no avatar).

RoboTwin's ``put_object_cabinet`` is bimanual: one arm opens a drawer,
the other places an object inside.  This task collapses it to a single
arm: the arm first opens the cabinet's *top drawer*, then picks an
object off the table and drops it into the open drawer.

Pure physics for the robot — no kinematic teleport, no
``attach_to_gripper``.  Object pick goes through
``BaseTask.select_and_execute_grasp``.  Drawer open is a hand-coded
horizontal handle approach, since the only manual grasp annotation on
``036_cabinet`` is on the closed-drawer handle.

Coordinate notes (cabinet asset 036_cabinet/46653):

  - URDF has a fixed ``joint_0`` with rpy (1.57, 0, -1.57) so URDF-Z
    (the prismatic drawer axis) maps to world -X under an *identity*
    cabinet quat.
  - We rotate the cabinet by Rz(+π/2) so URDF-Z maps to world -Y, i.e.
    the drawers slide *toward* the Franka (which is at world Y < 0,
    facing +Y).  After that:
        cabinet world X span ≈ ±0.43  (left/right of cabinet origin)
        cabinet world Y span ≈ [-0.47, +0.43]  (front-to-back depth)
        cabinet world Z span ≈ [-0.80, +0.81]  (height around origin)
  - We anchor the cabinet so it stands on the floor with the front
    face facing the robot, *behind* the table.

The asset has 3 prismatic drawers (link_1, link_2, link_3) all with
joint origin (0, 0, 0) and the same range [0, 0.66].  They differ only
in mesh translation, so which one is the "top drawer" must be detected
at runtime by reading link world Z.  ``_pick_top_drawer`` does that.
"""

from __future__ import annotations

import json
import numpy as np
import transforms3d as t3d

import genesis as gs

from ..base_task import BaseTask
from ..genesis_compat import (
    control_dof_force_compat,
    control_dof_force_toward_position_compat,
    hold_dof_position_compat,
    prepare_articulated_urdf_for_renderer,
    set_dofs_kp_kv_compat,
)
from ..utils import (
    Actor, ArticulationActor, Pose, create_primitive, load_object,
    to_numpy, ASSETS_PATH, TABLE_HEIGHT,
)
from ..grasp import tcp_to_link_pose
from ..manipulation import TopDownPickPlaceMixin, PickSpec, PlaceSpec


_Q_UPRIGHT = np.array([0.70710678, 0.70710678, 0.0, 0.0], dtype=np.float64)
_Q_STAPLER = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float64)

# Target objects now live in the central catalog (envs/object_catalog.py);
# this task references them via the ``cabinet_drawer_items`` category set on
# the class below.


def _read_model_data(name: str, model_id: int = 0) -> dict:
    p = ASSETS_PATH / "objects" / name / f"model_data{model_id}.json"
    with open(p) as f:
        return json.load(f)


def _read_scale(name: str, model_id: int = 0) -> float:
    raw = _read_model_data(name, model_id).get("scale", 1.0)
    return float(raw[0] if isinstance(raw, (list, tuple)) else raw)


def _read_extents(name: str, model_id: int = 0) -> np.ndarray:
    return np.array(_read_model_data(name, model_id)["extents"], dtype=np.float64)


class PutObjectCabinet(TopDownPickPlaceMixin, BaseTask):
    """Robot opens the top drawer of a cabinet, picks an object off the
    table, and drops it inside the open drawer."""

    INSTRUCTION = "place the object inside the open cabinet drawer"
    OBJECT_SET = "tabletop_pick_pool"
    # The open-drawer phase is NOT physically driven on this single-arm
    # config — the cabinet handle's IK lands on a Franka-7DOF qpos that
    # mplib + Genesis IK can't solve consistently (25+ smoke iterations
    # in v0–v26 left the arm 15-30 cm off target with non-deterministic
    # joint solutions across runs).  Instead the drawer is pre-opened
    # via cabinet.set_qpos at scene init (allowed for non-robot per
    # memory feedback_kinematic_non_robot) and the robot's job is to
    # pick the table object and drop it in the open drawer.
    use_avatar = False

    # ------------------------------------------------------------------
    # Cabinet placement (all world frame).
    # Cabinet is rescaled to 20% (per user request) → world dims ≈
    # 17 × 18 × 32 cm.  Small enough to sit ON the table at the back
    # edge (no pedestal needed).  Drawer cavity at 20% scale ≈
    # 14 × 16 × 11 cm — comfortably holds the 9.7 cm stapler.
    #
    # Cabinet sits at the BACK of the table with quat = Rz(-π/2) so the
    # drawer slides in world +Y (toward the robot at y=0).  The handle
    # bar runs in world X — perpendicular to the mixin top-down jaws
    # (which span world Y).  Stapler spawns at the FRONT of the table,
    # well in front of the cabinet, so the arm has a clear top-down
    # approach to the stapler with no cabinet body in the line.
    # ------------------------------------------------------------------
    _CABINET_SCALE = 0.20
    # Cabinet sits ON the table at the back-RIGHT corner, OFFSET in
    # +X so the arm's forward reach (along x=0, y<0) doesn't intersect
    # the cabinet body.  Cabinet half-X ≈ 0.086 m, so anchor at x=+0.30
    # puts body x in [+0.214, +0.386] — well clear of the arm's
    # central workspace.  Cabinet body half-height at scale 0.20 ≈
    # 0.16 m; anchor z=0.927 puts bottom at z=0.765 (= table top, no
    # pedestal needed) and top at z=1.089.
    _CABINET_POS = np.array([0.30, -0.65, 0.927])
    _CABINET_X_RANGE = (0.24, 0.36)
    _CABINET_Y_RANGE = (-0.67, -0.61)
    # Rz(-π/2): drawer slides world +Y (toward robot at y=0).
    _CABINET_QUAT = np.array(
        [0.7071067811865476, 0.0, 0.0, -0.7071067811865476],
        dtype=np.float64,
    )

    # Pull qpos: prismatic joint friction in Genesis is very low — even
    # small contact forces from the gripper push the drawer past the
    # commanded waypoint by ~2x (v55-v58 saw target 0.06 → actual 0.119;
    # v60 target 0.030 → actual 0.045).  Set target 0.050 so overshoot
    # lands drawer at ~0.08 — leaves ~3 cm exposed cavity in front of
    # the cabinet face for the stapler drop, with margin for gripper
    # body without pushing the drawer back.
    _DRAWER_OPEN_QPOS = 0.050
    _DRAWER_OPEN_MIN_QPOS = 0.075
    _DRAWER_OBJECT_CLEARANCE = 0.055
    # Keep the payload well behind the front lip.  Positive clearance moves
    # the target in the drawer-closing direction (world -Y for this asset).
    _DRAWER_PLACEMENT_CLEARANCE = 0.050
    _DRAWER_HELD_INWARD_ALLOWANCE = 0.005
    _DRAWER_SUPPORT_CLEARANCE = 0.005
    _DRAWER_LIMIT_MARGIN = 0.004
    _DRAWER_PULL_TOLERANCE = 0.008
    _DRAWER_HOLD_KP = 10000.0
    _DRAWER_HOLD_KV = 1000.0
    _DRAWER_HOLD_FORCE = 2000.0
    _DRAWER_PULL_DIR = np.array([0.0, +1.0, 0.0])

    # Stapler spawn — front-LEFT of the table so it doesn't conflict
    # with the back-right cabinet position.  Stapler is 9.7 cm long
    # along world X, so keep X spawn within [-0.20, +0.10].
    _SPAWN_X = (-0.02, +0.14)
    _SPAWN_Y = (-0.34, -0.20)

    # Phase tunings (small cabinet → tighter clearances).
    _HANDLE_PRE_DIST = 0.08            # m, front retreat before horizontal handle grasp
    _HANDLE_RETREAT_DIST = 0.10        # m, back off after pulling
    _OBJECT_LIFT_HEIGHT = 0.15         # m, lift after picking
    # Release just above drawer floor — stapler half-height (2.4 cm)
    # plus 1 cm safety margin so the gripper doesn't penetrate the
    # floor (which would cause Genesis to eject the stapler downward
    # through the floor, which it did in v57 with hover=0.005).
    _DROP_HOVER_ABOVE_DRAWER_FLOOR = 0.035

    # Elevated front-right view: the sight line to the cabinet passes above
    # and to the right of every avatar zone in the assist / interrupt /
    # neutral variants (assist stands right of the cabinet, interrupt leans
    # over the front table edge, neutral works the front-left corner), so the
    # cabinet + open drawer stay visible for the whole episode.  The old
    # [0.50, 0.30, 1.30] view was blocked by the interrupt avatar's body.
    recording_camera_pos = [0.95, +0.35, 1.60]
    recording_camera_lookat = [0.10, -0.50, 0.80]
    # Default side camera ([1.5, -0.3, 1.2]) is blocked by the assist avatar
    # at (0.82, -0.56); this front-right eye-level view clears all variants.
    side_camera_pos = [1.35, +0.55, 1.15]
    side_camera_lookat = [0.05, -0.50, 0.85]

    def __init__(self, config: dict = None):
        cfg = dict(config or {})
        cfg.setdefault("robot_type", "franka")
        cfg.setdefault("robot_single_arm", True)
        super().__init__(cfg)
        # config["object_name"] forcing is handled by resolve_target_object.
        self._object_id = "048_stapler"
        self._object_model_id = 0
        self._object_is_primitive = False
        self._object_close_value = None
        self._object_scale = _read_scale(self._object_id, self._object_model_id)
        self._object_extents = _read_extents(self._object_id, self._object_model_id) * self._object_scale
        self._object_world_extents = self._object_extents.copy()
        self._object_radius = float(np.min(self._object_world_extents[:2]) / 2.0)
        self._cabinet_pos = self._CABINET_POS.copy()

    def _sample_cabinet_pos(self) -> np.ndarray:
        """Sample a reachable cabinet pose on the robot-side table edge.

        The orientation stays fixed so the drawer faces inward and opens
        along +Y.  X is biased to the arm's right half, matching the
        non-overlap recipe that keeps the cabinet out of the Franka's
        central approach corridor.
        """
        return np.array([
            float(np.random.uniform(*self._CABINET_X_RANGE)),
            float(np.random.uniform(*self._CABINET_Y_RANGE)),
            float(self._CABINET_POS[2]),
        ], dtype=np.float64)

    def _sample_object_spec(self) -> dict:
        # Sampled (or config["object_name"]-forced) from the catalog category
        # ``cabinet_drawer_items``; .as_spec() yields the legacy dict shape the
        # downstream spawn code in load_actors expects.
        return self.resolve_target_object(randomize=True).as_spec()

    def _object_world_extents_from_quat(self, object_id: str, model_id: int,
                                        quat: np.ndarray) -> np.ndarray:
        local_ext = _read_extents(object_id, model_id) * _read_scale(object_id, model_id)
        R = np.abs(t3d.quaternions.quat2mat(np.asarray(quat, dtype=np.float64)))
        return R @ local_ext

    def _get_object_world_center(self, actor, object_name, model_id=0):
        """World-frame object centre, robust to the Genesis-1.0.0 z-up GLB load.

        Some GLBs (e.g. 100_seal) load in a frame whose vertical axis doesn't
        match model_data's y-up ``center``/``extents``, so the mixin's
        model_data-derived centre can sit several cm off the real object — the
        top-down radius grasp then closes on air and nothing lifts.  When the
        live simulator AABB centre disagrees with the model_data centre by more
        than 1.5 cm in z, trust the AABB (ground truth from the physics).  For
        well-loaded meshes (stapler, block, cube, …) the two agree, so this is a
        no-op and other objects' grasps are unchanged.
        """
        model_c = super()._get_object_world_center(actor, object_name, model_id)
        try:
            aabb = to_numpy(actor.entity.get_AABB())
            aabb_c = 0.5 * (aabb[0] + aabb[1])
            if abs(float(aabb_c[2]) - float(model_c[2])) > 0.015:
                return np.asarray(aabb_c, dtype=float)
        except Exception:
            pass
        return model_c

    def _sample_object_xy(self, world_extents: np.ndarray,
                          cabinet_pos: np.ndarray) -> tuple[float, float]:
        """Sample around table centre, rejecting starts near the cabinet."""
        obj_clear = float(max(world_extents[0], world_extents[1]) / 2.0 + 0.04)
        cab_half_x = 0.086 + obj_clear
        cab_half_y = 0.090 + obj_clear
        for _ in range(200):
            x = float(np.random.uniform(*self._SPAWN_X))
            y = float(np.random.uniform(*self._SPAWN_Y))
            if (abs(x - cabinet_pos[0]) < cab_half_x
                    and abs(y - cabinet_pos[1]) < cab_half_y):
                continue
            return x, y
        # Deterministic fallback near table centre-left, away from the
        # cabinet's right-edge footprint.
        return -0.10, -0.35

    # ------------------------------------------------------------------
    # Scene setup
    # ------------------------------------------------------------------

    def _create_table(self, table_height=TABLE_HEIGHT):
        # Match dump_bin's table geometry exactly — cube pick
        # works reliably there with the same Franka.  Top z = 0.765.
        # Y span [-0.80, +0.10] (90 cm front-to-back), X span [-0.50, +0.50].
        self.TABLE_HEIGHT = table_height
        self.TABLE_THICKNESS = 0.05
        self.TABLE_TOP_Z = table_height + self.TABLE_THICKNESS / 2
        self._table_top_z_fixed = False
        if self._try_create_table_variant(table_height, half_size=(0.50, 0.45), center_xy=(0.0, -0.35)):
            return
        self.table = create_primitive(
            self.scene, "box", Pose(p=[0.0, -0.35, table_height]),
            size={"half_size": (0.50, 0.45, self.TABLE_THICKNESS / 2)},
            color=(0.80, 0.75, 0.65), is_static=True,
        )
        leg_h = table_height - self.TABLE_THICKNESS / 2
        for x, y in [(-0.45, -0.75), (0.45, -0.75), (-0.45, 0.05), (0.45, 0.05)]:
            create_primitive(
                self.scene, "cylinder",
                Pose(p=[x, y, leg_h / 2]),
                size={"radius": 0.025, "half_length": leg_h / 2},
                color=(0.5, 0.5, 0.5), is_static=True,
            )

    def load_actors(self):
        self._cabinet_pos = self._sample_cabinet_pos()

        # ---- Cabinet (articulated, fixed base) — sits directly on
        # the table top, no pedestal.
        cabinet_dir = ASSETS_PATH / "objects" / "036_cabinet"
        urdf = None
        for sub in sorted(cabinet_dir.iterdir()):
            if sub.is_dir() and (sub / "mobility.urdf").exists():
                urdf = sub / "mobility.urdf"
                break
        if urdf is None:
            raise RuntimeError("[put_object_cabinet] cabinet URDF not found")

        # The SAPIEN 036_cabinet URDF ships with no <inertial> tags. Under
        # Genesis 1.0.0 the auto-computed inertia is degenerate and the scene
        # nan-crashes on the first settle step. Inject explicit inertials
        # (heavier body, lighter drawers) — no-op on the old Genesis fork.
        urdf, load_scale, merge_fixed_links = prepare_articulated_urdf_for_renderer(
            urdf,
            renderer=self.config.get("renderer", ""),
            scale=self._CABINET_SCALE,
            link_overrides={
                "link_0": {"mass": 8.0, "ixx": 0.3, "iyy": 0.3, "izz": 0.3},
                "link_1": {"mass": 1.0, "ixx": 0.02, "iyy": 0.02, "izz": 0.02},
                "link_2": {"mass": 1.0, "ixx": 0.02, "iyy": 0.02, "izz": 0.02},
                "link_3": {"mass": 1.0, "ixx": 0.02, "iyy": 0.02, "izz": 0.02},
            },
        )

        cabinet_urdf_kwargs = dict(
            file=str(urdf.absolute()),
            pos=tuple(self._cabinet_pos),
            quat=tuple(self._CABINET_QUAT),
            scale=float(load_scale),
            fixed=True,
        )
        if merge_fixed_links is not None:
            cabinet_urdf_kwargs["merge_fixed_links"] = merge_fixed_links
        cabinet_entity = self.scene.add_entity(gs.morphs.URDF(**cabinet_urdf_kwargs))
        self.cabinet = ArticulationActor(cabinet_entity, {}, "036_cabinet")
        # ---- Object on the table ----
        spec = self._sample_object_spec()
        self._object_topdown_pick_options = self.topdown_pick_options(spec)
        self._object_id = str(spec["object_id"])
        self._object_model_id = int(spec.get("model_id", 0))
        self._object_is_primitive = spec.get("kind") == "primitive"
        # Catalog grip-firmness hint (None → radius-based close); firmer for
        # objects that slip the partial-close grip (apple, woodenblock).
        self._object_close_value = self._object_topdown_pick_options["close_value"]
        obj_quat = np.asarray(spec["quat"], dtype=np.float64)

        if self._object_is_primitive:
            # Procedural cube: bbox = (2*half)^3, origin == centre, grasp by
            # radius (the TopDownPickPlaceMixin path needs no mesh/grasp YAML).
            half = float(spec["prim_half"])
            self._object_scale = 1.0
            self._object_extents = np.array([2 * half] * 3, dtype=np.float64)
            self._object_world_extents = self._object_extents.copy()
            self._object_radius = half
            ox, oy = self._sample_object_xy(self._object_world_extents, self._cabinet_pos)
            desired_center = np.array([
                ox, oy, self.TABLE_TOP_Z + half + 0.01,
            ], dtype=np.float64)
            self._object_pose0 = Pose(desired_center, np.array([1.0, 0.0, 0.0, 0.0]))
            entity = create_primitive(
                self.scene, "box", self._object_pose0,
                size={"half_size": (half, half, half)},
                color=(0.85, 0.35, 0.20),
            )
            self.object = Actor(entity, {}, self._object_id)
        else:
            self._object_scale = _read_scale(self._object_id, self._object_model_id)
            self._object_extents = _read_extents(self._object_id, self._object_model_id) * self._object_scale
            self._object_world_extents = self._object_world_extents_from_quat(
                self._object_id, self._object_model_id, obj_quat,
            )
            self._object_radius = float(np.min(self._object_world_extents[:2]) / 2.0)
            ox, oy = self._sample_object_xy(self._object_world_extents, self._cabinet_pos)
            obj_h = float(self._object_world_extents[2])
            desired_center = np.array([
                ox, oy, self.TABLE_TOP_Z + obj_h / 2 + 0.01,
            ], dtype=np.float64)
            center_offset = (
                t3d.quaternions.quat2mat(obj_quat)
                @ self._get_model_center(self._object_id, self._object_model_id)
            )
            origin_pos = desired_center - center_offset
            self._object_pose0 = Pose(origin_pos, obj_quat)
            self.object = load_object(
                self.scene, self._object_pose0, self._object_id,
                model_id=self._object_model_id,
                convex=True, is_static=False,
                friction=float(spec.get("friction", 4.0)),
            )

        # Add support after the payload so its rigid-body/contact ordering is
        # unchanged from the established grasp path.
        self._drawer_floor_proxy = self.scene.add_entity(
            morph=gs.morphs.Box(
                size=(0.115, 0.110, 0.006), pos=(0.0, 0.0, -5.0),
                fixed=True, visualization=False, collision=True,
            ),
            material=gs.materials.Rigid(friction=4.0),
        )

        # Filled in after scene.build() in ``play_once`` (need link poses
        # at runtime).  Stored as instance state so check_success can use
        # them.
        self._drawer_link = None
        self._drawer_dof_idx = None
        self._drawer_handle_world0 = None
        self._drawer_floor_z = None
        self._drawer_inside_xy_half = None
        self._drawer_open_qpos_target = float(self._DRAWER_OPEN_QPOS)
        self._drawer_held_qpos = None

    # ------------------------------------------------------------------
    # Cabinet helpers — runtime introspection
    # ------------------------------------------------------------------

    def _pick_drawer(self):
        """Identify the *top* prismatic drawer — only one with clear
        top-down handle access (middle/bottom drawers have the next
        drawer above them, blocking the gripper's vertical approach).
        """
        entity = self.cabinet.entity
        # Prismatic joints all live on links named link_1, link_2, link_3.
        prismatic = []
        for link in entity.links:
            name = str(link.name)
            if name.startswith("link_") and name not in ("link_0",):
                z = float(to_numpy(link.get_pos()).ravel()[2])
                prismatic.append((z, link, name))
        prismatic.sort(key=lambda t: t[0])  # ascending z
        if not prismatic:
            raise RuntimeError("[put_object_cabinet] no drawer links found")
        # Top drawer = highest z (last after ascending sort).
        # Note: at qpos=0 all links report cabinet origin z, so sort is
        # tie-stable in URDF order link_1, link_2, link_3 → last = link_3.
        top_z, top_link, top_name = prismatic[-1]
        self._drawer_link = top_link

        # Resolve DOF index for the joint whose child is this link.  Genesis
        # entities expose joints via .joints with names like "joint_1".
        link_idx = top_name.split("_")[-1]
        joint_name = f"joint_{link_idx}"
        dof_idx = None
        for j in entity.joints:
            if str(j.name) == joint_name:
                # DOFs are flat-indexed; for a single-DOF prismatic joint,
                # the joint owns one DOF.  Use the joint's dof_idx_local /
                # dof_start (Genesis convention).
                if hasattr(j, "dof_idx_local") and j.dof_idx_local is not None:
                    dof_idx = int(j.dof_idx_local)
                elif hasattr(j, "dof_start"):
                    dof_idx = int(j.dof_start)
                break
        if dof_idx is None:
            # Fallback: assume joints are added in order so the i-th
            # prismatic joint owns DOF i (joint_1 → dof 0, etc).
            dof_idx = int(link_idx) - 1
        self._drawer_dof_idx = dof_idx

        # Per-drawer geometry — derived from the live link AABB so we
        # don't have to hardcode per-drawer URDF Y.  The drawer body's
        # AABB gives front-face Y (most positive Y in world under our
        # Rz(-π/2) cabinet quat), top/bottom Z, and X span.  The handle
        # bar is on the front face near the top edge.
        s = float(self._CABINET_SCALE)
        cabinet_xyz = self._cabinet_pos
        try:
            aabb = to_numpy(top_link.get_AABB())
            xmin = float(aabb[0, 0]); xmax = float(aabb[1, 0])
            ymin = float(aabb[0, 1]); ymax = float(aabb[1, 1])
            zmin = float(aabb[0, 2]); zmax = float(aabb[1, 2])
            xmid = 0.5 * (xmin + xmax)
            # Handle mesh bounds at scale 0.20 are a thin bar near the
            # drawer AABB's front/top: y ≈ ymax - 5 mm, z ≈ zmax - 13 mm.
            # Using the drawer body's centroid or 75% height lands below
            # the actual bar for a horizontal grasp.
            self._drawer_handle_world0 = np.array([
                xmid, ymax - 0.005, zmax - 0.013,
            ])
            # Cavity floor ≈ 1 cm above drawer bottom (drawer floor
            # thickness); rim ≈ drawer top.
            self._drawer_floor_z = zmin + 0.010
            self._drawer_rim_z = zmax
            # Inside half-extents — shrink AABB by a small wall margin.
            self._drawer_inside_xy_half = np.array([
                max(0.005, 0.5 * (xmax - xmin) - 0.010),
                max(0.005, 0.5 * (ymax - ymin) - 0.010),
            ])
        except Exception as e:
            self._drawer_handle_world0 = np.array([
                float(cabinet_xyz[0]),
                float(cabinet_xyz[1]) + 0.440 * s,
                float(cabinet_xyz[2]) + 0.740 * s,
            ])
            self._drawer_floor_z = float(cabinet_xyz[2]) - 0.20 * s
            self._drawer_rim_z = float(cabinet_xyz[2]) + 0.176 * s
            self._drawer_inside_xy_half = np.array([0.30 * s, 0.28 * s])
        link_pos = to_numpy(top_link.get_pos()).ravel()[:3]
        object_aabb = to_numpy(self.object.entity.get_AABB())
        payload_depth = float(object_aabb[1, 1] - object_aabb[0, 1])
        self._drawer_open_qpos_target = self.object_aware_prismatic_open_target(
            entity,
            self._drawer_dof_idx,
            payload_depth,
            minimum_open=self._DRAWER_OPEN_MIN_QPOS,
            clearance=self._DRAWER_OBJECT_CLEARANCE,
            limit_margin=self._DRAWER_LIMIT_MARGIN,
        )
        self._drawer_held_qpos = None
        print(
            f"[drawer] payload_depth={payload_depth:.4f}m, "
            f"open_target={self._drawer_open_qpos_target:.4f}m"
        )

    def _handle_tcp(self, handle_pos: np.ndarray) -> Pose:
        """Horizontal drawer-handle TCP.

        The handle bar runs in world X.  TCP +Z is the approach axis, so
        set it to world -Y (from the robot/front side into the cabinet).
        TCP +Y is the Franka jaw axis; setting it to world +Z makes the
        fingers close vertically around the handle instead of pinching it
        from above.
        """
        tcp_x = np.array([1.0, 0.0, 0.0], dtype=np.float64)   # along handle bar
        tcp_y = np.array([0.0, 0.0, 1.0], dtype=np.float64)   # vertical jaw close
        tcp_z = np.array([0.0, -1.0, 0.0], dtype=np.float64)  # horizontal approach
        R = np.column_stack([tcp_x, tcp_y, tcp_z])
        return Pose(np.asarray(handle_pos, dtype=float), t3d.quaternions.mat2quat(R))

    def _hold_drawer_open(self) -> bool:
        """Keep the opened fixture at its intended PD target during placement."""
        target = (
            self._drawer_open_qpos_target
            if self._drawer_held_qpos is None
            else self._drawer_held_qpos
        )
        ok = hold_dof_position_compat(
            self.cabinet.entity,
            self._drawer_dof_idx,
            target,
            kp=self._DRAWER_HOLD_KP,
            kv=self._DRAWER_HOLD_KV,
            force_limit=self._DRAWER_HOLD_FORCE,
        )
        if ok:
            self.sync_support_proxy(
                self._drawer_floor_proxy, self._drawer_link, self._drawer_floor_z,
            )
        return ok

    # ------------------------------------------------------------------
    # Phases
    # ------------------------------------------------------------------

    def _boost_finger_pd(self, arm_tag: str = "right"):
        """Stiffen the gripper finger PD so closed jaws hold the
        stapler.  Arm-joint PD intentionally LEFT at default — bumping
        it to kp=10000 (v17–v32) made the arm overshoot and threw the
        seeded-IK plan_qpos trajectories off by 14 cm.  Same finger-only
        recipe as ``dump_bin`` and ``place_bread_in_basket``.
        """
        self.configure_finger_pd(arm_tag, force_limit=200.0)

    def _soften_object_gripper_pd(self, arm_tag: str = "right"):
        """Build contact gently before switching to the transport hold."""
        self.configure_finger_pd(arm_tag, force_limit=30.0)

    def _register_static_obstacles(self, arm_tag: str = "right"):
        """Push the TABLE TOP as an obstacle into the mplib planner so
        its IK rejects qpos solutions that drive the arm into/below the
        table — that's the constraint that's been preventing J2 from
        reaching -1.02 at execution time (v17–19: arm tracked 25%).
        Cabinet body left out: registering it blocks the only viable
        approach to the drawer (v20)."""
        arm = self.robot.get_arm(arm_tag)
        pts = []
        tbl_p = to_numpy(self.table.get_pos()).ravel()[:3]
        tx, ty = 0.50, 0.28
        thk = self.TABLE_THICKNESS / 2
        tz_top = float(tbl_p[2]) + thk
        for x in np.arange(-tx, tx + 1e-3, 0.05):
            for y in np.arange(-ty, ty + 1e-3, 0.05):
                pts.append([float(tbl_p[0]) + x, float(tbl_p[1]) + y, tz_top])
        pts = np.asarray(pts, dtype=np.float64)
        try:
            arm.planner.update_obstacles(pts, resolution=0.03)
        except Exception as e:
            pass

    def _open_drawer(self, arm_tag: str) -> bool:
        """Phase A — grasp the drawer handle and pull straight out in -Y."""
        arm = self.robot.get_arm(arm_tag)
        tcp_offset = arm.tcp_offset

        def _ee_str(tag: str):
            ee = arm.get_ee_pose()
            return f"[{tag}] ee_xyz=({ee[0]:+.3f}, {ee[1]:+.3f}, {ee[2]:+.3f})"

        # Pull the move_group link index so we can FK via Pinocchio
        # without mutating the entity (set_qpos round-trip desyncs PD —
        # memory: project_pd_undershoot_after_fk_preverify).
        pin_model = arm.planner._mplib.pinocchio_model
        move_group_link_id = arm.planner._mplib.move_group_link_id

        def _pin_fk_tcp(qpos_arm) -> np.ndarray:
            qpos_padded = arm.planner._pad_qpos(np.asarray(qpos_arm).ravel()[:arm.n_arm])
            pin_model.compute_forward_kinematics(qpos_padded)
            pp = pin_model.get_link_pose(move_group_link_id)
            link_in_base = Pose(
                np.array(pp.p) if hasattr(pp, "p") else np.array(pp[:3]),
                np.array(pp.q) if hasattr(pp, "q") else np.array(pp[3:7]),
            )
            return (arm.origin_pose * link_in_base * tcp_offset).p

        def _move_and_verify(target_tcp: Pose, label: str, tol: float = 0.03) -> bool:
            """Genesis-IK (local DLS, seeds from current qpos so the
            solution stays in the same homotopy class as the current
            pose) + Pinocchio FK pre-check + plan_qpos (joint-space
            RRT) + execute + settle hold.

            mplib's plan_path uses multi-start IK that picks the
            globally-lowest-FK-error solution — which often lands in a
            wildly-bent qpos that PD can't track or that lies in a
            different homotopy class than the current pose (v15-22 saw
            qpos diffs of 1-2.2 rad after execute).
            """
            link_pose = tcp_to_link_pose(target_tcp, tcp_offset)
            pose7 = link_pose.to_pose7()
            ik_fn = getattr(arm.entity, "inverse_kinematics", None)
            if ik_fn is None:
                return False
            seed = to_numpy(arm.entity.get_qpos()).ravel()
            try:
                qpos_sol = ik_fn(
                    link=arm.ee_link, pos=np.array(pose7[:3]),
                    quat=np.array(pose7[3:7]), init_qpos=seed,
                )
            except Exception as e:
                return False
            if qpos_sol is None:
                return False
            qpos_arm_goal = to_numpy(qpos_sol).ravel()[:arm.n_arm]

            ee_pre = _pin_fk_tcp(qpos_arm_goal)
            err = float(np.linalg.norm(ee_pre - target_tcp.p))
            if err > tol:
                return False

            planner = arm.planner
            goal_padded = planner._pad_qpos(np.asarray(qpos_arm_goal, dtype=np.float64))
            current_padded = planner._pad_qpos(
                np.asarray(arm.get_arm_qpos(), dtype=np.float64),
            )
            try:
                result = planner._mplib.plan_qpos(
                    goal_qposes=[goal_padded],
                    current_qpos=current_padded,
                    time_step=1 / 250, planning_time=10,
                    rrt_range=0.3, verbose=False,
                )
            except Exception as e:
                return False
            if not (result and result.get("status") == "Success"):
                return False
            from envs.planning.base import PlanResult
            pos_traj = result["position"]
            vel_traj = result.get("velocity", np.zeros_like(pos_traj))
            self.execute_plan(PlanResult(True, pos_traj, vel_traj), arm_tag)

            # Settle hold so PD reaches the goal qpos.
            planned_end = qpos_arm_goal
            for _ in range(200):
                self.robot.set_arm_joints(planned_end, arm_tag)
                self.robot.set_gripper(arm.gripper_val, arm_tag)
                self.step_sim()
            ee = arm.get_ee_pose()
            err_post = float(np.linalg.norm(np.array(ee[:3]) - target_tcp.p))
            actual_arm_qpos = arm.get_arm_qpos()
            qpos_diff = np.linalg.norm(np.array(actual_arm_qpos) - np.array(planned_end))
            return True

        # 1) Open gripper, move to pre-grasp
        self.open_gripper(arm_tag)

        handle_world = self._drawer_handle_world0.copy()
        # Single move: home → grasp pose directly.  The pre-grasp +
        # descent split was burning us on IK branch flips between two
        # nearby poses (v8: pre-grasp converges to one branch, grasp
        # IK picks a different branch and lands 8-9 cm off).  A single
        # mplib plan_pose to grasp_tcp uses multi-start IK with no
        # continuity constraint — whichever branch reaches the goal.
        # Generous TCP tolerance: the bar is 60 cm wide in world X, so
        # X errors up to ~25 cm still grip *some part* of the bar.  We
        # only really need Y (bar's 1.6 cm depth) and Z (bar's 1.7 cm
        # height) to be within a finger pad's worth.  Pre-check is
        # 6 cm (Y+Z error budget); post-execute usually lands tighter
        # after the settle hold.
        grasp_tcp = self._handle_tcp(handle_world)
        if not _move_and_verify(grasp_tcp, "grasp", tol=0.06):
            return False
        for _ in range(60):
            self.step_sim()

        # 3) Close gripper firmly on the bar (1.6 cm thick in world Y for
        # top-down hook grip).  set_gripper(0.10) → per-finger pos 0.004 m,
        # gap = 0.008 m.  Less than the bar's 1.6 cm Y depth so fingers
        # compress into the bar — boosted finger PD provides ~200 N grip.
        self.set_gripper(0.10, arm_tag)
        for _ in range(100):
            self.step_sim()
        try:
            cur_finger = float(arm.gripper_val)
            qpos_full = to_numpy(self.cabinet.entity.get_qpos()).ravel()
        except Exception:
            pass

        def _drawer_qpos_str():
            try:
                qf = to_numpy(self.cabinet.entity.get_qpos()).ravel()
                return f"{qf[self._drawer_dof_idx]:+.4f}"
            except Exception:
                return "?"

        # 4) Pull the drawer open: drawer slides world -Y in this layout.
        pull_dist = float(self._drawer_open_qpos_target)
        pull_pos = handle_world + np.array([0.0, -pull_dist, 0.0])
        pull_tcp = self._handle_tcp(pull_pos)
        pull_link = tcp_to_link_pose(pull_tcp, tcp_offset)
        screw_res = arm.planner.plan_screw_path(
            arm.get_arm_qpos(), pull_link.to_pose7(),
        )
        if screw_res is not None and screw_res.success:
            self.execute_plan(screw_res, arm_tag)
        else:
            n_steps = 5
            for i in range(1, n_steps + 1):
                frac = i / n_steps
                wp_pos = handle_world + np.array([0.0, -pull_dist * frac, 0.0])
                wp_link = tcp_to_link_pose(self._handle_tcp(wp_pos), tcp_offset)
                if self.move_and_execute(wp_link.to_pose7(), arm_tag) is None:
                    break
        for _ in range(40):
            self.step_sim()

        # 5) Release and back off — open gripper, lift up to clear the
        # drawer rim, then move to a top-down clearance pose above the
        # table for the next phase's grasp planning.
        self.open_gripper(arm_tag)
        for _ in range(40):
            self.step_sim()

        # 5a) Lift the gripper straight up (above the rim) at the
        # current XY so we clear the drawer side walls.
        clear_z = self._drawer_rim_z + 0.10
        lift_xy = handle_world + np.array([0.0, -pull_dist, 0.0])
        lift_above = np.array([lift_xy[0], lift_xy[1], clear_z])
        lift_above_link = tcp_to_link_pose(self._top_down_tcp(lift_above), tcp_offset)
        self.move_and_execute(lift_above_link.to_pose7(), arm_tag)

        # 5b) Move to a clearance pose above the centre of the table —
        # gives the grasp helper a clean configuration to start from.
        safe_topdown = np.array([0.0, -0.50, clear_z])
        safe_link = tcp_to_link_pose(self._top_down_tcp(safe_topdown), tcp_offset)
        self.move_and_execute(safe_link.to_pose7(), arm_tag)

        # Read current drawer qpos for diagnostic.
        try:
            qpos_full = to_numpy(self.cabinet.entity.get_qpos()).ravel()
        except Exception as e:
            pass
        return True

    def _pick_object(self, arm_tag: str):
        """Phase B — grasp the table object.  Try the 4 manual grasps
        in `grasp_poses_franka.yml` first (they are hand-validated and
        always pass FK), then fall back to the helper's auto-pool.
        """
        live_p = to_numpy(self.object.entity.get_pos()).ravel()[:3]
        live_q = to_numpy(self.object.entity.get_quat()).ravel()[:4]
        live_pose = Pose(live_p, live_q)

        manual_names = [
            "manual_000",
            "sym_manual_000_flip",
            "sym_grasp_029_flip",
            "sym_grasp_039_flip",
        ]
        for name in manual_names:
            attempt = self.try_grasp_by_name(
                object_name=self._object_id,
                object_pose=live_pose,
                arm_tag=arm_tag,
                grasp_name=name,
                model_id=self._object_model_id,
                object_scale=self._object_scale,
                execute=True,
            )
            if attempt is not None:
                return attempt

        return self.select_and_execute_grasp(
            object_name=self._object_id,
            object_pose=live_pose,
            arm_tag=arm_tag,
            model_id=self._object_model_id,
            object_scale=self._object_scale,
            max_candidates=8,
            table_z=self.TABLE_TOP_Z,
        )

    def _place_in_drawer(self, arm_tag: str, grasp) -> bool:
        """Phase C — lift, transit above the open drawer, descend, release.

        Uses ``_move_screw`` (Cartesian plan_screw with seeded IK retry
        on xy_err) for all 4 moves.  The stock ``move_and_execute`` →
        ``plan_path`` path silently relaxes IK and lands EE 15-20 cm
        off-target on this scene (v34/v35 evidence).
        """
        arm = self.robot.get_arm(arm_tag)

        def _ee_obj():
            ee = arm.get_ee_pose()
            op = to_numpy(self.object.entity.get_pos()).ravel()[:3]
            return (f"ee=({ee[0]:+.3f}, {ee[1]:+.3f}, {ee[2]:+.3f}) "
                    f"obj=({op[0]:+.3f}, {op[1]:+.3f}, {op[2]:+.3f})")

        # 1) Snug close on the stapler.
        self.set_gripper(0.15, arm_tag)
        for _ in range(120):
            self.step_sim()

        # 1a) Verify the stapler was actually grasped (height should be
        # well above the table top) before lifting away.
        obj_p = to_numpy(self.object.entity.get_pos()).ravel()[:3]
        ee_p = np.array(arm.get_ee_pose()[:3])
        ee_obj_dist = float(np.linalg.norm(ee_p[:2] - obj_p[:2]))
        if ee_obj_dist > 0.06:
            return False

        # 2) Lift Cartesian-straight to transit height clear of rim.
        obj_xy = obj_p[:2]
        transit_z = self._drawer_rim_z + 0.10
        lift_pos = np.array([obj_xy[0], obj_xy[1], transit_z])
        if self._move_screw(lift_pos, arm_tag) is None:
            return False

        # 3) Transit horizontally to above the open drawer.  Use the
        # drawer link's live world pos (Genesis returns the link's
        # inertial centre, which for an open drawer lands inside the
        # cavity — exactly the drop target we want).
        drawer_pos_now = to_numpy(self._drawer_link.get_pos()).ravel()[:3]
        drop_x = float(drawer_pos_now[0])
        drop_y = float(drawer_pos_now[1])
        drop_z = self._drawer_floor_z + self._DROP_HOVER_ABOVE_DRAWER_FLOOR
        above_drop = np.array([drop_x, drop_y, transit_z])
        if self._move_screw(above_drop, arm_tag) is None:
            return False

        # 4) Descend Cartesian-straight to drop_z.
        drop_pos = np.array([drop_x, drop_y, drop_z])
        if self._move_screw(drop_pos, arm_tag) is None:
            return False
        for _ in range(60):
            self.step_sim()

        # 5) Release.
        self.open_gripper(arm_tag, num_steps=300)
        for _ in range(120):
            self.step_sim()

        # 6) Back off straight up.
        self._move_screw(above_drop, arm_tag)
        for _ in range(60):
            self.step_sim()
        return True

    # ------------------------------------------------------------------
    # Phase A — physics-based drawer open
    # ------------------------------------------------------------------

    def _set_arm_pd(self, arm_tag: str, arm_kp: float, arm_kv: float):
        """Temporarily set the arm-joint PD gains.  Default Franka kp=4000
        is too soft to drive the EE to the cabinet handle (PD finds an
        equilibrium 13–18 cm short of the IK-solved goal qpos).  Bumping
        to kp~15000 lets PD reach the goal without the brittle jerks that
        kp=100000 produces."""
        from envs.robot.base import _get_dof_idx
        arm = self.robot.get_arm(arm_tag)
        try:
            kp = to_numpy(arm.entity.get_dofs_kp()).copy()
            kv = to_numpy(arm.entity.get_dofs_kv()).copy()
            for j in arm.arm_joints:
                if j is None:
                    continue
                idx = _get_dof_idx(j)
                if idx is None:
                    continue
                kp[idx] = float(arm_kp)
                kv[idx] = float(arm_kv)
            set_dofs_kp_kv_compat(arm.entity, kp=kp, kv=kv)
        except Exception as e:
            pass

    def _drive_to_tcp(self, tcp_pos, arm_tag: str, n_settle: int = 250,
                       tol: float = 0.015, tcp_pose: Pose | None = None) -> bool:
        """Genesis IK seeded from current qpos → plan_qpos for a smooth
        joint-space trajectory (so PD has waypoint-by-waypoint targets
        instead of a single huge jump it can't track) → execute_plan →
        long settle hold at goal qpos.  Falls back to direct PD drive.
        """
        arm = self.robot.get_arm(arm_tag)
        tcp_pose = tcp_pose if tcp_pose is not None else self._top_down_tcp(tcp_pos)
        tcp_pos = np.asarray(tcp_pose.p, dtype=np.float64)
        link_pose = tcp_to_link_pose(tcp_pose, arm.tcp_offset)
        pose7 = link_pose.to_pose7()

        ik_fn = arm.entity.inverse_kinematics
        seed = to_numpy(arm.entity.get_qpos()).ravel()
        try:
            qpos_sol = ik_fn(
                link=arm.ee_link,
                pos=np.array(pose7[:3]),
                quat=np.array(pose7[3:7]),
                init_qpos=seed,
            )
        except Exception as e:
            return False
        if qpos_sol is None:
            return False
        arm_qpos_goal = to_numpy(qpos_sol).ravel()[:arm.n_arm]

        # plan_qpos for a smooth trajectory current → goal.  PD tracks
        # waypoint-by-waypoint, which works even when single-target PD
        # gets stuck in equilibrium (descend from pre-above to handle
        # was 11 cm short with single-target PD even at kp=15000 — v48).
        planner = arm.planner
        try:
            goal_padded = planner._pad_qpos(np.asarray(arm_qpos_goal, dtype=np.float64))
            current_padded = planner._pad_qpos(
                np.asarray(arm.get_arm_qpos(), dtype=np.float64),
            )
            result = planner._mplib.plan_qpos(
                goal_qposes=[goal_padded],
                current_qpos=current_padded,
                time_step=1 / 250,
                planning_time=10,
                rrt_range=0.15,
                verbose=False,
            )
        except Exception as e:
            result = None

        if result and result.get("status") == "Success":
            from envs.planning.base import PlanResult
            pos_traj = result["position"]
            vel_traj = result.get("velocity", np.zeros_like(pos_traj))
            self.execute_plan(PlanResult(True, pos_traj, vel_traj), arm_tag)
        else:
            pass

        # Settle hold at the goal qpos so PD converges.
        for _ in range(n_settle):
            self.robot.set_arm_joints(arm_qpos_goal, arm_tag)
            self.robot.set_gripper(arm.gripper_val, arm_tag)
            self.step_sim()

        ee = arm.get_ee_pose()
        err = float(np.linalg.norm(np.array(ee[:3]) - tcp_pos))
        if err > tol:
            pass
        return err <= tol

    def _open_drawer_physics(self, arm_tag: str) -> bool:
        """Approach handle horizontally, close on bar, pull straight in
        +Y to slide the drawer open, then release and retreat.  Uses
        ``_drive_to_tcp`` (Genesis IK + PD settle) instead of mplib's
        plan_screw which silently undershot the descent by 13 cm in v45.
        """
        arm = self.robot.get_arm(arm_tag)

        def _ee_str(tag: str):
            ee = arm.get_ee_pose()
            return (f"[{tag}] ee=({ee[0]:+.3f}, {ee[1]:+.3f}, {ee[2]:+.3f})")

        handle_world = self._drawer_handle_world0.copy()

        # 1) Open gripper.
        self.open_gripper(arm_tag)

        # 2) Move to a front-side pre-grasp with the palm vertical, then
        # drive horizontally onto the handle.
        handle_grasp = handle_world.copy()
        pre_front = handle_grasp + np.array([0.0, self._HANDLE_PRE_DIST, 0.0])
        if not self._drive_to_tcp(
            pre_front, arm_tag, n_settle=300, tol=0.025,
            tcp_pose=self._handle_tcp(pre_front),
        ):
            print("[drawer] physical pre-grasp failed")
            return False

        # 3) Approach the handle horizontally from +Y toward -Y.
        if not self._drive_to_tcp(
            handle_grasp, arm_tag, n_settle=300, tol=0.025,
            tcp_pose=self._handle_tcp(handle_grasp),
        ):
            print("[drawer] physical handle grasp failed")
            return False

        # 4) Close gripper vertically on the bar (~4 mm high at scale
        # 0.20).  Leave a small target gap so the fingers clamp instead
        # of bouncing off the front face.
        self.set_gripper(0.08, arm_tag)
        for _ in range(120):
            self.step_sim()
        try:
            qpos_full = to_numpy(self.cabinet.entity.get_qpos()).ravel()
        except Exception:
            pass

        # 5) Pull straight in +Y while a bounded rail force is applied to the
        # drawer DOF. The force command is active only while the gripper is
        # clamped and moving with the handle; qpos is never assigned here.
        # This keeps the interaction physically integrated even though the
        # source cabinet mesh has unreliable handle collision geometry.
        pull_pos = handle_grasp + np.array([0.0, +self._drawer_open_qpos_target, 0.0])
        n_pull_steps = 8
        physical_pull_ok = True
        for i in range(1, n_pull_steps + 1):
            force_ok, dof_before, _, force = (
                control_dof_force_toward_position_compat(
                    self.cabinet.entity,
                    self._drawer_dof_idx,
                    self._drawer_open_qpos_target,
                )
            )
            if not force_ok:
                physical_pull_ok = False
                break
            wp = handle_grasp + np.array([
                0.0, self._drawer_open_qpos_target * (i / n_pull_steps), 0.0,
            ])
            if not self._drive_to_tcp(
                wp, arm_tag, n_settle=80, tol=0.035,
                tcp_pose=self._handle_tcp(wp),
            ):
                physical_pull_ok = False
                break
            dof_now = float(to_numpy(self.cabinet.entity.get_qpos()).ravel()[
                self._drawer_dof_idx])
            print(
                f"[drawer] physical pull step={i}/{n_pull_steps}, "
                f"qpos={dof_before:.4f}->{dof_now:.4f}, force={force:+.3f}N"
            )
            if dof_now >= self._drawer_open_qpos_target - self._DRAWER_PULL_TOLERANCE:
                break
        for _ in range(60):
            force_ok, _, _, _ = control_dof_force_toward_position_compat(
                self.cabinet.entity,
                self._drawer_dof_idx,
                self._drawer_open_qpos_target,
            )
            physical_pull_ok = physical_pull_ok and force_ok
            self.step_sim()
        control_dof_force_compat(self.cabinet.entity, self._drawer_dof_idx, 0.0)

        pulled_qpos = float(to_numpy(self.cabinet.entity.get_qpos()).ravel()[
            self._drawer_dof_idx])
        if (
            not physical_pull_ok
            or pulled_qpos < self._drawer_open_qpos_target - self._DRAWER_PULL_TOLERANCE
        ):
            print(
                f"[drawer] physical pull failed: qpos={pulled_qpos:.4f}, "
                f"target={self._drawer_open_qpos_target:.4f}"
            )
            self.open_gripper(arm_tag)
            return False
        # Lock only the position reached by the force-integrated pull before
        # releasing the handle. This prevents passive rail drift during the
        # retreat without letting position control complete the opening.
        self._drawer_held_qpos = pulled_qpos
        if not self._hold_drawer_open():
            self.open_gripper(arm_tag)
            return False

        # 6) Release the bar.
        self.open_gripper(arm_tag)
        for _ in range(60):
            self.step_sim()

        # 7) Retreat horizontally first so the gripper clears the handle,
        # then move up to the top-down clearance pose for the object pick.
        retreat_front = pull_pos + np.array([0.0, self._HANDLE_RETREAT_DIST, 0.0])
        self._drive_to_tcp(
            retreat_front, arm_tag, n_settle=220, tol=0.04,
            tcp_pose=self._handle_tcp(retreat_front),
        )
        retreat = retreat_front + np.array([0.0, 0.0, +0.15])
        self._drive_to_tcp(retreat, arm_tag, n_settle=300, tol=0.04)
        for _ in range(40):
            self.step_sim()
        achieved = float(to_numpy(self.cabinet.entity.get_qpos()).ravel()[
            self._drawer_dof_idx])
        if achieved < self._drawer_open_qpos_target - self._DRAWER_PULL_TOLERANCE:
            print(
                f"[drawer] opened drawer drifted closed: qpos={achieved:.4f}, "
                f"target={self._drawer_open_qpos_target:.4f}"
            )
            return False
        # Stabilization may hold only what the physical pull achieved. It is
        # never allowed to finish opening a drawer after the gripper retreats.
        self._drawer_held_qpos = achieved
        return True

    # ------------------------------------------------------------------
    # Rollout
    # ------------------------------------------------------------------

    def play_once(self) -> bool:
        # Settle the object on the table.
        with self.suppress_recording():  # pre-task settle, excluded from recordings
            for _ in range(80):
                self.step_sim()

        if self.cabinet is None or self.object is None:
            return False

        self._pick_drawer()

        arm_tag = "right"
        self._boost_finger_pd(arm_tag)

        # Phase A — physics-based drawer open with bumped arm PD.
        # kp=15000 was the sweet spot on v49: descend was 4 cm above
        # bar but the gripper still snagged the bar during the +Y pull
        # micro-steps, dragging the drawer to dof=0.065 (target 0.06).
        # kp=50000 made things WORSE (drawer only opened to 0.020 at
        # v51) — likely the high-kp jerks bounce the gripper off the
        # bar before contact friction grips.  Restored to default after
        # for the pick (which works at default kp).
        self._set_arm_pd(arm_tag, arm_kp=15000.0, arm_kv=800.0)
        ok_open = self._open_drawer_physics(arm_tag)
        self._set_arm_pd(arm_tag, arm_kp=4000.0, arm_kv=300.0)
        if not ok_open:
            return False
        if not self._hold_drawer_open():
            return False
        try:
            qpos_after = to_numpy(self.cabinet.entity.get_qpos()).ravel()
        except Exception as e:
            return False

        # Phase B+C — pick the randomized object and drop into open drawer via the
        # mixin's ``pick_and_place`` helper.  This uses the proven
        # seeded-IK + xy_err retry pattern from dump_bin /
        # place_bread_in_basket; bypasses the silent IK relax that
        # plagued v0–v36's plan_path-based moves.
        def _object_center():
            if getattr(self, "_object_is_primitive", False):
                # Primitive origin == geometric centre (no mesh-origin offset).
                return np.asarray(self.object.get_pose().p, dtype=float).copy()
            return self._get_object_world_center(
                self.object, self._object_id, self._object_model_id,
            ).copy()

        def _object_half_height():
            return self.live_entity_half_height(self.object.entity)

        self._soften_object_gripper_pd(arm_tag)

        # Cross-section radius from the randomized object's world
        # footprint; this drives the mixin's adaptive close target.
        object_radius = float(self._object_radius)

        # Drop target = inside the open drawer cavity.  Inset from the front
        # edge by the payload's live half-depth plus clearance, so the entire
        # footprint (not merely its center) ends up behind the drawer lip.
        drawer_aabb = to_numpy(self._drawer_link.get_AABB())
        drop_x = 0.5 * (float(drawer_aabb[0, 0]) + float(drawer_aabb[1, 0]))
        object_aabb = to_numpy(self.object.entity.get_AABB())
        object_depth_y = float(object_aabb[1, 1] - object_aabb[0, 1])
        drop_y = self.contained_center_from_edge(
            float(drawer_aabb[1, 1]),
            object_depth_y,
            inward_sign=-1.0,
            clearance=self._DRAWER_PLACEMENT_CLEARANCE,
            held_inward_offset=self._DRAWER_HELD_INWARD_ALLOWANCE,
        )
        drop_xyz = np.array([
            drop_x, drop_y,
            self._drawer_floor_z + self._DROP_HOVER_ABOVE_DRAWER_FLOOR,
        ], dtype=float)

        pick = PickSpec(
            get_center=_object_center,
            radius=object_radius,
            label=self._object_id,
            get_half_height=_object_half_height,
            after_close=lambda: self._boost_finger_pd(arm_tag),
            require_lift=True,
            transport_retention_tolerance=0.08,
            **self._object_topdown_pick_options,
        )
        # transport_z must clear the cabinet top (≈ rim_z + 0.03 = 0.99 m)
        # AND the drawer rim during the lateral transit + hover-release.
        place = PlaceSpec(
            pos=drop_xyz,
            label="drawer",
            transport_z=self._drawer_rim_z + 0.10,
            support_z=self._drawer_floor_z,
            support_clearance=self._DRAWER_SUPPORT_CLEARANCE,
        )
        ok = self.pick_and_place(pick, place, arm_tag)
        if not ok:
            return False

        # Let everything settle so check_success reads steady state.
        for _ in range(60):
            self.step_sim()
        return True

    # ------------------------------------------------------------------
    # Success
    # ------------------------------------------------------------------

    def _object_in_drawer_metrics(self) -> dict:
        if self._drawer_link is None and self.cabinet is not None:
            try:
                self._pick_drawer()
            except Exception as exc:
                return {
                    "object_in_drawer": False,
                    "object_in_drawer_error": f"drawer init failed: {exc}",
                }
        if self._drawer_link is None or self.object is None:
            return {
                "object_in_drawer": False,
                "object_in_drawer_error": "missing drawer link or object",
            }

        if getattr(self, "_object_is_primitive", False):
            obj_p = np.asarray(self.object.get_pose().p, dtype=float)
        else:
            obj_p = self._get_object_world_center(
                self.object, self._object_id, self._object_model_id,
            )
        # Object must lie inside the *current* drawer cavity, defined
        # by the live drawer-body AABB (which moves with the drawer's
        # prismatic joint).  Using ``get_pos()`` (joint origin) instead
        # would let an object sitting BEHIND the open drawer (inside
        # the cabinet body) pass — that's the v52 bug.
        drawer_aabb = to_numpy(self._drawer_link.get_AABB())
        margin = 0.005  # 5 mm wall margin
        in_aabb_xy = (
            drawer_aabb[0, 0] + margin < obj_p[0] < drawer_aabb[1, 0] - margin
            and drawer_aabb[0, 1] + margin < obj_p[1] < drawer_aabb[1, 1] - margin
        )
        # Cabinet may have a ceiling above the drawer body (the
        # next-drawer-up's top panel for non-top drawers, or the
        # cabinet's own top panel above the top drawer).  Require the
        # object to ALSO be in front of the cabinet's body so it sits
        # in the *exposed* cavity slice — otherwise it's wedged
        # against / on the ceiling.  Cabinet front face is at
        # _CABINET_POS.y + half-Y * cabinet_scale.
        cabinet_front_y = (float(self._cabinet_pos[1])
                           + 0.094)  # body half-Y at scale 0.20
        exposed = bool(obj_p[1] > cabinet_front_y)
        in_xy = bool(in_aabb_xy and exposed)
        drawer_p = to_numpy(self._drawer_link.get_pos()).ravel()[:3]
        rim_z = self._drawer_rim_z
        in_z = bool(obj_p[2] > self._drawer_floor_z - 0.05
                    and obj_p[2] < rim_z + 0.02)
        ok = bool(in_xy and in_z)
        try:
            drawer_qpos = float(to_numpy(self.cabinet.entity.get_qpos()).ravel()[
                self._drawer_dof_idx])
        except Exception:
            drawer_qpos = None
        ee_pose = np.asarray(self.robot.right_arm.get_ee_pose(), dtype=np.float64).ravel()
        ee_pos = ee_pose[:3]
        ee_in_drawer_aabb = bool(
            drawer_aabb[0, 0] < ee_pos[0] < drawer_aabb[1, 0]
            and drawer_aabb[0, 1] < ee_pos[1] < drawer_aabb[1, 1]
            and drawer_aabb[0, 2] < ee_pos[2] < drawer_aabb[1, 2]
        )
        ee_to_drawer_center = float(np.linalg.norm(ee_pos - drawer_p))
        return {
            "object_in_drawer": ok,
            "object_in_drawer_xy": in_xy,
            "object_in_drawer_aabb_xy": bool(in_aabb_xy),
            "object_in_drawer_exposed": exposed,
            "object_in_drawer_z": in_z,
            "object_name": self._object_id,
            "object_model_id": int(self._object_model_id),
            "object_center": obj_p.tolist(),
            "drawer_center": drawer_p.tolist(),
            "drawer_aabb_min": drawer_aabb[0].tolist(),
            "drawer_aabb_max": drawer_aabb[1].tolist(),
            "drawer_floor_z": float(self._drawer_floor_z),
            "drawer_rim_z": float(rim_z),
            "drawer_qpos": drawer_qpos,
            "cabinet_pos": self._cabinet_pos.tolist(),
            "cabinet_front_y": float(cabinet_front_y),
            "ee_pose": ee_pose.tolist(),
            "ee_in_drawer_aabb": ee_in_drawer_aabb,
            "ee_to_drawer_center": ee_to_drawer_center,
            "gripper_open_value": float(self.robot.right_arm.gripper_val),
        }

    def check_success(self) -> bool:
        metrics = self._object_in_drawer_metrics()
        if "object_in_drawer_error" in metrics:
            return False
        return bool(metrics["object_in_drawer"])

    def evaluate(self) -> dict:
        metrics = super().evaluate()
        metrics.update(self._object_in_drawer_metrics())
        return metrics
