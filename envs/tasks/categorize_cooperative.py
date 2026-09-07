"""Categorize cooperative: human and robot sort objects into matching baskets."""

import os
import json
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import transforms3d as t3d
import genesis as gs

from ..avatar.eval_mode_mixin import EvalModeAvatarMixin
from ..base_task import BaseTask
from ..genesis_compat import set_dofs_kp_kv_compat
from ..utils import Actor, Pose, load_object, create_primitive, ASSETS_PATH, place_decorative_objects
from ..grasp import tcp_to_link_pose
from ..object_catalog import object_set_pairs, get_entry
from ..manipulation import (
    TopDownPickPlaceMixin,
    PickSpec as RadiusPickSpec,
    PlaceSpec as RadiusPlaceSpec,
)


@dataclass
class PickSpec:
    """Describes what to pick.

    ``get_center`` is called multiple times during the pick (initial plan,
    re-read after pre-grasp in case approach bumped the object, and for debug
    logging after grip/lift).  Keep it cheap.

    ``radius`` is the cross-section half-extent used to size the partial
    gripper close (``radius * 0.7``) and the below-equator grasp offset.

    ``close_target_m`` (optional) overrides the formula-derived gripper
    close target (in metres, finger half-gap).  Read from the object's
    `model_data{N}.json` `grasp_close_target_m` field.  When None, the
    fallback formula in `_grasp_object` runs.

    ``actor`` (optional) is the Genesis actor for the picked object —
    used by ``_update_planner_obstacles`` to skip its aabb during the
    pre-grasp / place transit so the planner doesn't reject the goal
    as in-collision with the held object.
    """

    get_center: Callable[[], np.ndarray]
    radius: float
    label: str = "object"
    close_target_m: Optional[float] = None
    actor: Optional[object] = None


@dataclass
class PlaceSpec:
    """Describes where to drop.

    ``pos`` is the world (x, y) of the drop target (z field is ignored when
    ``drop_from_z`` is set — which it always is for container-style drops).
    ``drop_from_z`` set = hover at that height and release (basket pattern).
    ``drop_from_z`` None = descend TCP to ``pos.z`` before release.

    ``basket_idx`` (optional) is the destination basket index — passed to
    ``_update_planner_obstacles`` as ``skip_basket_idx`` during the descent
    so the basket's own walls don't block the drop path.
    """

    pos: np.ndarray
    label: str = "place"
    drop_from_z: Optional[float] = None
    basket_idx: Optional[int] = None


class CategorizeCooperative(TopDownPickPlaceMixin, EvalModeAvatarMixin, BaseTask):
    """Robot categorizes loose objects into baskets that already contain a matching example.

    Three categories, each with a basket containing one static example object
    and one loose object on the table that the robot must sort into the correct
    basket.  The avatar cooperates by sorting some objects on its side
    (placeholder until motion is available).

    Scene layout (top view, +Y points from robot toward avatar):
      - Robot (Franka) at y=-0.5
      - Large table centered at y=0.1
      - Three loose objects near robot side (y=-0.05)
      - Three baskets further from robot (y=0.20) with example objects
      - Avatar at y=1.2
    """

    INSTRUCTION = "sort the objects into the matching baskets"

    use_avatar = True
    # Avatar stands ~0.22 m behind the table edge (y=0.45) so the body is not
    # clipping the table and the motion is visually comfortable.  With the
    # body at y=0.67 the right shoulder sits at world y≈0.69 and the left at
    # the mirror.  Each loose object on the avatar side (free XY in
    # x∈[-0.32,+0.32], y∈[+0.32,+0.42]) is reached by the closer hand
    # (see `_avatar_pick_and_place`): right hand for x≤0, left hand for x>0.
    # Worst-case shoulder→target ≈ 0.39 m at the far-Y corners — well inside
    # the ~0.65 m palm reach.
    avatar_init_pos = np.array([0.0, 0.67, -0.18])
    avatar_init_rot = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=np.float64)

    # Table cropped toward the robot side but with a small band behind the
    # baskets on the avatar side to hold the avatar-side loose objects
    # (half_size_y=0.45, dy=0.0 → y ∈ [-0.45, 0.45]).  Baskets at y=0.22,
    # avatar-side loose objects at y=0.40 (6 cm beyond basket rear edge).
    table_offset = np.array([0.0, 0.0])
    _TABLE_HALF_SIZE_Y = 0.45

    # VLA primary view: near top-down over the full 1.60 x 0.90 m table.
    static_camera_list = [{
        "name": "head_camera",
        "position": [0.0, 0.005, 2.20],
        "forward": [0.0, -0.005, -1.30],
    }]

    recording_camera_pos = [1.1, -0.3, 1.7]
    recording_camera_lookat = [0.0, 0.00, 0.75]
    # Top-down view of table — so we can see inside the baskets.
    side_camera_pos = [0.0, -0.05, 2.2]
    side_camera_lookat = [0.0, -0.05, 0.80]

    # Pure top-down gripper orientation — Franka 7-DOF has no singularity here.
    _R_DOWN = np.array([
        [1.0,  0.0,  0.0],
        [0.0, -1.0,  0.0],
        [0.0,  0.0, -1.0],
    ])

    MIN_TCP_CLEARANCE = 0.0

    # Push an inflated point-cloud of all baskets + sort_targets to mplib
    # before each transit / descent.  Required for cate-inte's column
    # transits (validated 10/10 PASS by row 2 cate-inte agent).  On
    # cate-coop the random-XY layout creates much longer diagonal
    # transits — empirically this caused `plan_screw` to fail on every
    # move and plan_pose RRT to route into bad IK branches (gripper
    # closes high, swings during descent, knocks objects).  Default OFF
    # on cate-coop while we work the residual control issues; cate-inte
    # overrides it back to True.  v8 2026-04-28 re-enabled — without it,
    # cross-table transits in seeds 0/2 sweep the arm wrist over the
    # remaining objects and knock the woodenblock off the table edge to
    # y=-0.85 (floor), making the final pick unrecoverable.  The chase
    # fix only catches small displacements, not floor-knock.  Inflation
    # forces plan_pose to detour around all spawned objects.  The branch-
    # flip joint-delta heuristic + chase fix should handle the IK badness
    # that previously motivated the False default.
    _USE_OBSTACLE_INFLATION = True

    # Override DEFAULT_AVATAR_POOL: celebrity_Jensen_Huang.glb crashes Genesis
    # mesh-loader during scene build (seed 0 cc_v3 2026-04-28: SIGABRT after
    # avatar load). Pool below is the remainder of DEFAULT_AVATAR_POOL.
    AVATAR_POOL = [
        "avatars/models/custom_Adrian_Keller.glb",
        "avatars/models/custom_Clara_Voss.glb",
        "avatars/models/custom_Malik_Okoro.glb",
    ]

    # Per-object Z offset (positive = grasp HIGHER than center).  Replaces the
    # default `obj_center[2] - radius * 0.3` formula for objects that need a
    # specific grasp height.  Seal is a tall slender cylinder.  User 2026-04-29
    # asked for +3 cm but +0.022 (+30 mm above original) led to transit slip
    # because the grip ended up at the seal's top and the heavier bottom
    # pivoted out under transit g-load.  Compromise: +0.015 (mid-upper body,
    # ~7 mm below top).
    OBJECT_GRASP_Z_OFFSET = {
        "100_seal": 0.015,
    }

    # Compensate for PD controller Z overshoot — command this far below the
    # true target so the actual TCP lands at the desired height.
    PD_Z_COMPENSATION = 0.03

    # v15h: route the robot pick-side approach through the shared
    # ``BaseTask.select_and_execute_grasp`` helper using the manually
    # annotated YAML grasps.  The annotations live in mesh-origin frame,
    # so the helper drops the TCP at the right grasping point even when
    # mesh origin differs from bbox centre.  v15g failed because it then
    # ran a "chase" that pulled the gripper toward bbox centre at the
    # same Z — plowing through the object.  Fix: trust the YAML and skip
    # the chase on this path.
    USE_YAML_GRASPS = True

    # All robot-side picks route through BaseTask.select_and_execute_grasp.
    # Older versions skipped the wooden block because only yaw45 manual
    # grasps existed; the asset now has center top-down Franka grasps, so
    # the legacy synthetic fallback is no longer needed.
    YAML_GRASP_SKIP = set()

    # Subclasses can prefer task-validated grasp names before falling back to
    # the generic ranker.  Cooperative keeps the generic standard-helper order:
    # the full preferred object set regressed cooperative rollouts, but the
    # wooden block should avoid body-center yaw0/yaw90 grasps that can carry
    # the block loosely and drop it during transport.
    PREFERRED_GRASP_NAMES = {
        "035_apple": (
            "manual_center_p0_yaw0",
            "manual_center_p0_yaw2",
        ),
        "086_woodenblock": (
            "manual_topdown_yaw45_center",
            "manual_topdown_yaw45_higher",
        ),
    }
    # Object-specific grasp allowlists.  These are exclusive: once an object
    # is listed here, the generic YAML ranker must not fall back to any other
    # annotation.
    EXCLUSIVE_GRASP_NAMES = {
        # The milk carton's folded top is slippery, so all top-down annotations
        # are banned.  These four candidates approach its upright body from the
        # side and were validated through the full categorize placement path.
        "038_milk-box": (
            "grasp_065",  # validated full categorize pick/place
            "grasp_058",  # validated full categorize pick/place
            "grasp_005",  # validated full categorize pick/place
            "grasp_078",  # validated full categorize pick/place
        ),
        # The radius path descends below the cube's equator.  At some table
        # locations its lateral pose correction contacts a lower corner and
        # chases the cube across the table.  Use the asset's face-centred,
        # top-down body grasps instead; try both closing axes and finger flips
        # before giving up.
        "073_rubikscube": (
            "manual_topdown_close_x",
            "manual_topdown_close_x_flipfinger",
            "manual_topdown_close_y",
            "manual_topdown_close_y_flipfinger",
        ),
    }
    YAML_ONLY_GRASP_LABELS = frozenset(EXCLUSIVE_GRASP_NAMES)

    # The stapler rests with its long dimension in the yaw-0 jaw direction.
    # Closing there pushes on the two ends and ejects it.  Rotate the top-down
    # parallel jaws by 90 degrees so they close across the short dimension;
    # yaw 0 is deliberately no longer a candidate for this object.
    RADIUS_GRASP_YAW = {
        "048_stapler": 0.5 * np.pi,
    }

    # Apple pose selection is workspace-sensitive.  The authored yaw-0 pose is
    # unreachable on the left side, where the radius pose is validated.  On
    # the right, the lower underhand grasp retains mid-workspace carries while
    # the center-height grasp is stable at the far edge.  These thresholds are
    # evaluated from the live object center, not the episode seed.
    APPLE_YAML_MIN_X = 0.0
    APPLE_CENTER_MIN_X = 0.20
    APPLE_CENTER_GRASP_NAMES = (
        "manual_center_p0_yaw0",
        "manual_underhand_p0_yaw0",
    )
    APPLE_UNDERHAND_GRASP_NAMES = (
        "manual_underhand_p0_yaw0",
        "manual_center_p0_yaw0",
    )

    # Disabled by default.  A column-aligned cooperative spawn was tested but
    # regressed the rollout, so keep the original free-XY layout here.
    ROBOT_MATCH_BASKET_COLUMNS = False
    ROBOT_COLUMN_X_JITTER = 0.015
    ROBOT_COLUMN_Y_RANGE = (-0.16, -0.08)

    # Eval/testbed mode: policies call take_action(action)->obs while the
    # avatar begins its cooperative sorting sequence at a seeded random policy
    # step.  This keeps baseline rollouts from depending on scripted play_once.
    EVAL_TRIGGER_STEP_MIN = 40
    EVAL_TRIGGER_STEP_MAX = 100

    # Objects are sampled per-episode from the shared catalog pool
    # ``tabletop_pick_pool`` (NUM_CATEGORIES distinct draws, one per basket).
    # CATEGORIES holds a default subset for any pre-load reference; load_actors
    # re-samples ``self._cats`` each episode.
    OBJECT_SET = "tabletop_pick_pool"
    NUM_CATEGORIES = 3

    # Mesh loading preserves the authored Y-up frame. Map local +Y to world +Z
    # so jars, blocks, and seals stand on their intended bases.
    SPAWN_UPRIGHT_QUAT = (0.70710678, 0.70710678, 0.0, 0.0)

    # Grasp strategy.  The per-object YAML grasp annotations (``_pick_and_place_yaml``)
    # are object-local and angled, so under Genesis 1.2.0 their approach sweeps
    # light objects out of the jaws (stapler/block knocked 8–45 cm) or closes on
    # air.  The orientation-agnostic top-down radius grasp from
    # ``TopDownPickPlaceMixin`` — the exact path that lifts the same
    # apple/woodenblock/seal/rubikscube 3/3 in ``put_object_cabinet`` — descends
    # straight down onto the object's live-AABB centre and is far more robust.
    USE_RADIUS_GRASP = True

    # Objects whose resting AABB is taller than this (m) are skipped by the
    # radius grasp. With the authored-frame upright spawn the objects stand on
    # their base, so a top-down grasp handles them up to ~10 cm tall (jar
    # 9.5cm, block 7cm,
    # seal/rubik/cube all pass + lift+place).  Only the 12cm milk-box carton is
    # too tall to purchase top-down, so it is left standing on the table.
    RADIUS_MAX_GRASP_HEIGHT = 0.10

    # Object-level emergency bans for the radius path.  Keep this empty: an
    # object sampled as a required task target must not be silently left on the
    # table.  Apple model 1 is handled by the task-local 0.58 close below; the
    # task-native grasp/lift/hold probe retained it with <0.5 mm drift.
    RADIUS_SKIP_LABELS = set()

    # Categorize-local spawn friction overrides (else catalog friction).  The
    # 086_woodenblock@1 catalog friction (5.0, tuned for put_object_cabinet's
    # radius grasp) makes the block's contact solver blow up when it is RELEASED
    # into the basket (z→-40 / NaN).  At 2.5 the release is stable and the block
    # still grips + lifts + places (2/3), so categorize uses the gentler value.
    SPAWN_FRICTION_OVERRIDE = {"086_woodenblock": 2.5}

    # Categorize-local gripper close fraction overrides for the radius grasp
    # (0=closed, 1=open-ish).  Tuned for this pool's models + the below-equator
    # scoop; takes priority over the shared catalog ``close_value`` so cabinet
    # is not regressed.  Round/domed objects need a LOOSE cage (the scoop +
    # palm holds them; a tight close ejects them).
    RADIUS_CLOSE_OVERRIDE = {
        # apple@1 ~5.2 cm round: 0.35 (catalog) crush-ejects, 0.70 is too loose
        # to contact (fingers stop at 0.028 > 0.026 radius).  ~0.58 lets the
        # jaws touch + lightly squeeze so the below-equator scoop holds it.
        "035_apple": 0.58,
    }
    CATEGORIES = object_set_pairs(
        ["035_apple@1", "086_woodenblock@1", "100_seal@2"]
    )

    def _setup_scene(self):
        """Use smaller dt for better collision response during grasping.
        Honours --config raytracer.yml by building the renderer from config;
        without this, gs.Scene defaulted to rasterizer and our LuisaRender
        config was silently ignored.
        """
        try:
            backend = gs.gpu if os.environ.get("GENESIS_BACKEND") == "gpu" else gs.cpu
            gs.init(backend=backend, logging_level="error")
        except Exception:
            pass
        renderer = self._build_renderer()
        scene_kwargs = dict(
            show_viewer=self.config.get("show_viewer", False),
            sim_options=gs.options.SimOptions(dt=0.001),
        )
        if renderer is not None:
            scene_kwargs["renderer"] = renderer
        self.scene = gs.Scene(**scene_kwargs)
        renderer_type = self.config.get("renderer", "rasterizer").lower()
        self.scene.add_entity(gs.morphs.Plane(visualization=not self._hide_ground_visuals()))
        self._create_floor()

    def _load_robot(self):
        self.config.setdefault("robot_type", "franka")
        if self.config["robot_type"] == "franka":
            kwargs = self.config.setdefault("robot_kwargs", {})
            kwargs.setdefault("pos", [0.0, -0.35, 0.75])
            kwargs.setdefault("quat", [0.707, 0.0, 0.0, 0.707])
        super()._load_robot()

    def _boost_arm_pd(self):
        """Increase arm joint PD gains for better positioning accuracy."""
        from ..robot.franka_robot import _get_dof_idx, to_numpy
        arm = self.robot.get_arm("right")
        try:
            kp = to_numpy(arm.entity.get_dofs_kp()).copy()
            kv = to_numpy(arm.entity.get_dofs_kv()).copy()
        except Exception:
            return
        for j in arm.arm_joints:
            if j is not None:
                idx = _get_dof_idx(j)
                if idx is not None:
                    kp[idx] = 20000.0
                    kv[idx] = 1000.0
        # Medium finger kp — kp=15000 penetrates meshes (~390N per finger),
        # kp=800 is too weak.  kp=2500 holds ~45N per finger at full
        # compression: firm enough for 50g objects without crushing.
        for j in arm.finger_joints:
            if j is not None:
                idx = _get_dof_idx(j)
                if idx is not None:
                    kp[idx] = 2500.0
                    kv[idx] = 250.0
        set_dofs_kp_kv_compat(arm.entity, kp=kp, kv=kv)

    def _create_table(self, table_height=0.74):
        """Larger table to accommodate both robot and avatar sides."""
        self.TABLE_HEIGHT = table_height
        self.TABLE_THICKNESS = 0.05
        self.TABLE_TOP_Z = table_height + self.TABLE_THICKNESS / 2
        self._table_top_z_fixed = False
        dx, dy = self.table_offset
        if self._try_create_table_variant(table_height, half_size=(0.80, self._TABLE_HALF_SIZE_Y), center_xy=(dx, dy)):
            return

        self.table = create_primitive(
            self.scene, "box",
            Pose(p=[dx, dy, table_height]),
            size={"half_size": (0.80, self._TABLE_HALF_SIZE_Y, self.TABLE_THICKNESS / 2)},
            color=(0.8, 0.75, 0.65),
            is_static=True,
        )
        leg_h = table_height - self.TABLE_THICKNESS / 2
        _leg_y = self._TABLE_HALF_SIZE_Y - 0.05
        for x, y in [(-0.75, -_leg_y), (0.75, -_leg_y), (-0.75, _leg_y), (0.75, _leg_y)]:
            create_primitive(
                self.scene, "cylinder",
                Pose(p=[x + dx, y + dy, leg_h / 2]),
                size={"radius": 0.025, "half_length": leg_h / 2},
                color=(0.5, 0.5, 0.5),
                is_static=True,
            )

    @staticmethod
    def _get_object_cross_section_radius(object_name: str, model_id: int = 0) -> float:
        if object_name == "cube":
            return float(get_entry("cube").prim_half)
        md_path = ASSETS_PATH / "objects" / object_name / f"model_data{model_id}.json"
        with open(md_path) as f:
            md = json.load(f)
        extents = np.array(md["extents"])
        scale = np.array(md.get("scale", [1, 1, 1]))
        return float(np.min(extents * scale) / 2)

    @staticmethod
    def _get_grasp_close_target(object_name: str, model_id: int = 0) -> Optional[float]:
        """Per-object gripper close target in metres (finger half-gap).
        Returns None if the object's model_data has no `grasp_close_target_m`
        field — caller falls back to the radius-based formula."""
        md_path = ASSETS_PATH / "objects" / object_name / f"model_data{model_id}.json"
        if not md_path.exists():
            return None
        with open(md_path) as f:
            md = json.load(f)
        v = md.get("grasp_close_target_m")
        return float(v) if v is not None else None

    @staticmethod
    def _get_object_scale(object_name: str, model_id: int = 0) -> float:
        """Runtime mesh scale applied by load_object for this model variant."""
        md_path = ASSETS_PATH / "objects" / object_name / f"model_data{model_id}.json"
        if not md_path.exists():
            return 1.0
        with open(md_path) as f:
            md = json.load(f)
        raw = md.get("scale", 1.0)
        if isinstance(raw, (list, tuple)):
            return float(raw[0])
        return float(raw)

    def _select_and_execute_task_grasp(
        self,
        object_name: str,
        object_pose: Pose,
        arm_tag: str,
        *,
        model_id: int,
        object_scale: float,
        max_candidates: int = 20,
        preferred_names=None,
    ):
        """Run the standard grasp helper with task-local candidate priority."""
        exclusive_names = self.EXCLUSIVE_GRASP_NAMES.get(object_name)
        if object_name == "035_apple":
            x = float(np.asarray(object_pose.p, dtype=float)[0])
            exclusive_names = (
                self.APPLE_CENTER_GRASP_NAMES
                if x >= self.APPLE_CENTER_MIN_X
                else self.APPLE_UNDERHAND_GRASP_NAMES
            )
        if exclusive_names is not None:
            preferred_names = exclusive_names
        elif preferred_names is None:
            active = getattr(self, "_active_preferred_grasp_names", None)
            if active is not None and object_name in active:
                preferred_names = active[object_name]
            else:
                preferred_names = self.PREFERRED_GRASP_NAMES.get(object_name, ())
        for grasp_name in preferred_names:
            result = self.try_grasp_by_name(
                object_name,
                object_pose,
                arm_tag,
                grasp_name,
                model_id=model_id,
                robot_type="franka",
                object_scale=object_scale,
                execute=True,
            )
            if result is not None:
                return result

        # An exclusive allowlist is also a ban list for everything omitted.
        # Do not let the generic ranker silently reintroduce a banned milk-box
        # top grasp or an unvalidated apple/Rubik pose after these candidates
        # fail planning.
        if exclusive_names is not None:
            return None

        return self.select_and_execute_grasp(
            object_name, object_pose, arm_tag,
            model_id=model_id, max_candidates=max_candidates,
            table_z=self.TABLE_TOP_Z, robot_type="franka",
            object_scale=object_scale,
        )

    @staticmethod
    def _get_model_center(object_name: str, model_id: int = 0) -> np.ndarray:
        md_path = ASSETS_PATH / "objects" / object_name / f"model_data{model_id}.json"
        with open(md_path) as f:
            md = json.load(f)
        center = np.array(md["center"])
        scale = np.array(md.get("scale", [1, 1, 1]))
        return center * scale

    def _get_object_world_center(self, actor, object_name: str, model_id: int = 0) -> np.ndarray:
        if object_name == "cube":
            # Primitive origin == geometric centre (no mesh-origin offset).
            return np.asarray(actor.get_pose().p, dtype=float)
        model_center = self._get_model_center(object_name, model_id)
        pose = actor.get_pose()
        R = t3d.quaternions.quat2mat(np.array(pose.q))
        model_c = np.array(pose.p) + R @ model_center
        # Live-AABB centre override (mirrors put_object_cabinet): some GLBs load
        # in a frame whose vertical axis doesn't match model_data's y-up
        # center/extents, so the model_data centre sits several cm off the real
        # object and the top-down radius grasp closes on air.  When the live
        # simulator AABB centre disagrees by >1.5 cm in z, trust the physics.
        try:
            from ..robot.franka_robot import to_numpy as _to_numpy
            aabb = np.asarray(_to_numpy(actor.entity.get_AABB()), dtype=float)
            aabb_c = 0.5 * (aabb[0] + aabb[1])
            if abs(float(aabb_c[2]) - float(model_c[2])) > 0.015:
                return aabb_c
        except Exception:
            pass
        return model_c

    def _spawn_categorize_object(self, obj_name, model_id, x, y, table_top, quat):
        """Spawn one loose sortable object — a procedural cube for the
        ``cube`` pool entry (origin == centre, sits on the table), else the
        mesh asset."""
        if obj_name == "cube":
            half = float(get_entry("cube").prim_half)
            entity = create_primitive(
                self.scene, "box",
                Pose([x, y, table_top + half], np.array([1.0, 0.0, 0.0, 0.0])),
                size={"half_size": (half, half, half)},
                color=(0.85, 0.35, 0.20),
            )
            return Actor(entity, {}, "cube")
        # Use the catalog's per-object validated friction (4.0–5.0) instead of
        # a flat 2.5 — the loose categorize objects were slipping out of the
        # parallel-jaw grip on the lift under Genesis 1.2.0 because this task
        # ignored the per-object friction the cabinet/dump grasp tunings rely
        # on.  Cap at the Genesis Rigid friction limit (5.0).
        if obj_name in self.SPAWN_FRICTION_OVERRIDE:
            obj_friction = float(self.SPAWN_FRICTION_OVERRIDE[obj_name])
        else:
            try:
                obj_friction = float(min(5.0, get_entry(obj_name).friction))
            except Exception:
                obj_friction = 2.5
        return load_object(
            self.scene, Pose([x, y, table_top], quat),
            obj_name, model_id=model_id, convex=True,
            is_static=False, friction=obj_friction,
        )

    def load_actors(self):
        """Cooperative layout (+Y = robot → human):

            robot (y=-0.35)
              │
              ▼ robot-side loose objects (random XY in y∈[-0.20,-0.05])
              │
              ▼ boxes (y=+0.15)
              │
              ▼ avatar-side loose objects (random XY in y∈[+0.32,+0.42])
              │
            human (y=+0.52)

        Per-seed randomization (via np.random.seed set in BaseTask.reset):
          - `basket_perm` permutes which CATEGORY sits in which basket
            column.
          - `robot_perm` and `avatar_perm` are INDEPENDENT permutations
            that pick which CATEGORY each robot-side / avatar-side loose
            object carries.  Each object's ``basket_idx`` is derived by
            category match (basket holding the matching example) — both
            agents must categorise, and the avatar-side and robot-side
            orderings are decoupled from each other.
          - Robot-side and avatar-side loose objects are placed at
            **random XY** within their respective Y strips (rejection-
            sampled with min-separation + basket-clear constraints) — no
            longer aligned to a fixed row of column X's.
          - Basket column X still gets ±2 cm jitter.
          - `place_decorative_objects` scatters 3–5 clutter objects on
            the outer table strips, avoiding the actual sampled object
            positions and the robot base footprint.

        Boxes are built from 5 primitives (floor + 4 walls) — same pattern
        as categorize_interrupt — because the articulated 100194 URDF has
        lid pieces that trap thrown objects on the rim.
        """
        table_top = self.TABLE_TOP_Z + 0.02
        upright_q = np.asarray(self.SPAWN_UPRIGHT_QUAT, dtype=float)

        # Per-episode: sample NUM_CATEGORIES distinct objects from the shared
        # catalog pool, one per basket (each a distinct category).  cube is a
        # primitive (handled via the radius-based pick path below).
        pool = object_set_pairs(self.OBJECT_SET)
        pick = np.random.choice(len(pool), self.NUM_CATEGORIES, replace=False)
        cats = [pool[int(i)] for i in pick]
        if self.simplified_mode_enabled():
            cats = [cats[int(np.random.randint(0, len(cats)))]]
        self._cats = cats
        # Per-seed basket category permutation.
        basket_perm = np.random.permutation(len(cats))
        # Robot-side and avatar-side each get their OWN independent
        # category-to-slot permutation; each object's ``basket_idx`` is
        # then derived by category match via ``basket_col_of_cat``.
        robot_perm = np.random.permutation(len(cats))
        avatar_perm = np.random.permutation(len(cats))
        basket_col_of_cat = np.argsort(basket_perm)

        # Per-seed X jitter on each basket column (±2 cm).  Loose
        # objects no longer share these column X's — they are sampled
        # freely within an XY region per side.
        base_col_xs = np.linspace(-0.25, 0.25, len(cats))
        col_xs = base_col_xs + np.random.uniform(-0.02, 0.02, size=len(cats))
        col_xs = [float(x) for x in col_xs]

        # --- Three open-top boxes (primitive floor + 4 walls) -----------
        box_y = 0.15
        box_half_x = 0.096
        box_half_y = 0.096
        box_half_z = 0.056
        wall_t = 0.005
        box_color = (0.7, 0.5, 0.3)
        self.basket_poses = []
        self.basket_actors = []
        # Baskets sit at the column X positions, in basket_perm order.
        # Each basket holds its category example (visible inside the box).
        # Anchor flush with TABLE_TOP_Z (not table_top = TABLE_TOP_Z + 0.02)
        # — the +0.02 is a spawn offset for loose objects to fall under
        # gravity; static baskets must sit on the table, not float above.
        for col_idx, bx in enumerate(col_xs):
            cz = self.TABLE_TOP_Z + box_half_z
            self.basket_poses.append(
                Pose([bx, box_y, cz], np.array([1.0, 0.0, 0.0, 0.0]))
            )
            # Floor.
            create_primitive(
                self.scene, "box",
                Pose(p=[bx, box_y, cz - box_half_z + wall_t]),
                size={"half_size": (box_half_x, box_half_y, wall_t)},
                color=box_color, is_static=True,
            )
            # Four walls.
            for wx, wy, hx, hy in [
                (-box_half_x + wall_t, 0.0, wall_t, box_half_y),
                (+box_half_x - wall_t, 0.0, wall_t, box_half_y),
                (0.0, -box_half_y + wall_t, box_half_x, wall_t),
                (0.0, +box_half_y - wall_t, box_half_x, wall_t),
            ]:
                create_primitive(
                    self.scene, "box",
                    Pose(p=[bx + wx, box_y + wy, cz]),
                    size={"half_size": (hx, hy, box_half_z)},
                    color=box_color, is_static=True,
                )
        self._example_actors = []  # no preloaded examples in this layout

        # --- Rejection-sample XY positions per side --------------------
        # Robot-side strip: y ∈ [-0.20, -0.05] (well clear of basket
        # back edge at y≈+0.054 and the robot base at y=-0.35).
        # Avatar-side strip: y ∈ [+0.32, +0.42] (well clear of basket
        # front edge at y≈+0.246 and avatar body at y≈+0.55).
        # X stays inside ±0.32 — wider than the previous ±0.25 columns
        # but still inside Franka's reliable top-down IK reach and the
        # avatar's nearest-hand reach (left hand serves x≤0, right
        # x>0; worst-case shoulder→hand ≈ 0.62 m, under the ~0.65 m
        # palm-reach budget).
        loose_min_sep = 0.10        # 10 cm pairwise separation
        basket_clear = 0.13         # keep clear of basket centres
        robot_region = (-0.32, 0.32, -0.20, -0.05)
        avatar_region = (-0.32, 0.32, 0.32, 0.42)

        def _sample_xy(region, n_existing_pts, *, avoid_robot_base=False,
                        max_tries=400):
            xmin, xmax, ymin, ymax = region
            for _try in range(max_tries):
                x = float(np.random.uniform(xmin, xmax))
                y = float(np.random.uniform(ymin, ymax))
                # Stay clear of basket centres.
                if any(np.hypot(x - bx, y - 0.15) < basket_clear for bx in col_xs):
                    continue
                # Pairwise separation against earlier loose-object picks.
                if any(np.hypot(x - px, y - py) < loose_min_sep
                       for px, py in n_existing_pts):
                    continue
                # Robot-base clearance (only matters for the robot strip).
                if avoid_robot_base and np.hypot(x, y - (-0.35)) < 0.22:
                    continue
                return x, y
            i = len(n_existing_pts)
            return col_xs[i % len(col_xs)], (ymin + ymax) / 2

        robot_pts = []
        avatar_pts = []
        for _ in range(len(cats)):
            robot_pts.append(_sample_xy(robot_region, robot_pts,
                                         avoid_robot_base=True))
        for _ in range(len(cats)):
            avatar_pts.append(_sample_xy(avatar_region, avatar_pts))

        # --- Robot-side loose objects (independent permutation) --------
        # basket_idx is computed by category match — each object goes to
        # whichever basket holds the matching example, regardless of
        # where it spawned.
        self._sort_targets = []  # (actor, name, model_id, basket_idx)
        for slot_i in range(len(cats)):
            cat_idx = int(robot_perm[slot_i])
            obj_name, model_id = cats[cat_idx]
            if self.ROBOT_MATCH_BASKET_COLUMNS:
                basket_col = int(basket_col_of_cat[cat_idx])
                ox = float(col_xs[basket_col] + np.random.uniform(
                    -self.ROBOT_COLUMN_X_JITTER,
                    self.ROBOT_COLUMN_X_JITTER,
                ))
                oy = float(np.random.uniform(*self.ROBOT_COLUMN_Y_RANGE))
            else:
                ox, oy = robot_pts[slot_i]
            if obj_name == "086_woodenblock" and not self.ROBOT_MATCH_BASKET_COLUMNS:
                # The block is held by a standard friction side-clamp.  Keep
                # it near its destination column so the robust grasp does not
                # need a long cross-table carry that can let it slip.
                basket_col = int(basket_col_of_cat[cat_idx])
                ox = float(col_xs[basket_col])
                if basket_col == 1:
                    ox -= 0.06
                elif basket_col == 2:
                    ox -= 0.045
                oy = -0.085
            if obj_name == "100_seal" and not self.ROBOT_MATCH_BASKET_COLUMNS:
                # Seal grasps are reach-sensitive in Franka layouts; reuse the
                # tight strip validated for the standard helper instead of the
                # wider free robot-side sampler.
                ox = float(np.random.uniform(0.00, 0.05))
                oy = float(np.random.uniform(-0.10, -0.07))
            actor = self._spawn_categorize_object(obj_name, model_id, ox, oy, table_top, upright_q)
            basket_idx = int(basket_col_of_cat[cat_idx])
            self._sort_targets.append((actor, obj_name, model_id, basket_idx))

        # --- Avatar-side loose objects (independent permutation, free XY)
        self._avatar_sort_targets = []
        if not self.no_human_enabled():
            for slot_i in range(len(cats)):
                cat_idx = int(avatar_perm[slot_i])
                obj_name, model_id = cats[cat_idx]
                ox, oy = avatar_pts[slot_i]
                actor = self._spawn_categorize_object(obj_name, model_id, ox, oy, table_top, upright_q)
                basket_idx = int(basket_col_of_cat[cat_idx])
                self._avatar_sort_targets.append((actor, obj_name, model_id, basket_idx))

        # Universal BaseTask clutter handles table-random objects.
        self._clutter_actors = []

    def _top_down_tcp(self, pos, yaw: float = 0.0):
        """Top-down TCP with optional world-Z yaw for anisotropic objects."""
        Rz = t3d.euler.euler2mat(0.0, 0.0, float(yaw), axes="sxyz")
        q = t3d.quaternions.mat2quat(Rz @ self._R_DOWN)
        return Pose(np.asarray(pos, dtype=float), q)

    def _move_to_safe(self, arm_tag):
        arm = self.robot.get_arm(arm_tag)
        tcp_offset = arm.tcp_offset
        arm_base = np.array(arm.origin_pose.p)
        # Must clear 28cm-tall basket rim (box top at TABLE_TOP_Z + 0.30).
        safe_pos = np.array([arm_base[0], arm_base[1] + 0.30, self.TABLE_TOP_Z + 0.45])
        safe_link = tcp_to_link_pose(self._top_down_tcp(safe_pos), tcp_offset)
        self.move_and_execute(safe_link.to_pose7(), arm_tag)

    def _solve_ik(self, tcp_pose, arm_tag):
        """Solve IK for a TCP pose, return arm joint positions or None.

        Seeds ``init_qpos`` from the PREVIOUSLY COMMANDED pose (``arm._cached_target``)
        when available, else from the current measured qpos.  Seeding from the
        measured qpos while the arm is mid-motion lets PD lag propagate into
        the IK search, which can drag the solver across a homotopy boundary
        and flip the wrist 360° — that's the "arm rotates and throws the
        object" failure mode.  ``_cached_target`` is the pose we asked the
        arm to hold LAST, and it's continuous with the pose we'll ask for
        NEXT, so IK stays in-branch.
        """
        arm = self.robot.get_arm(arm_tag)
        tcp_offset = arm.tcp_offset
        link_pose = tcp_to_link_pose(tcp_pose, tcp_offset)
        pose7 = link_pose.to_pose7()

        ik_fn = getattr(arm.entity, "inverse_kinematics", None)
        if ik_fn is None:
            return None
        try:
            from ..robot.franka_robot import to_numpy
            seed_qpos = getattr(arm, "_cached_target", None)
            if seed_qpos is None:
                seed_qpos = to_numpy(arm.entity.get_qpos())
            seed_qpos = np.asarray(seed_qpos, dtype=np.float64).ravel()
            qpos_sol = ik_fn(
                link=arm.ee_link,
                pos=np.array(pose7[:3]),
                quat=np.array(pose7[3:7]),
                init_qpos=seed_qpos,
            )
            if qpos_sol is None:
                return None
            return to_numpy(qpos_sol).ravel()[:arm.n_arm]
        except Exception as e:
            return None

    def _teleport_to_tcp(self, tcp_pose, arm_tag):
        """Cartesian PD motion to the target TCP — same pattern as
        ``categorize_interrupt`` (which ships 10/10 on the sweep).

        The legacy name is retained so the call site reads the same.  A
        joint-space ``set_arm_joints`` + settle was tried first and turned
        out stochastic (1–2/3 pass rate) because the joint-space
        interpolation can sweep the wrist laterally and knock the target
        object before the fingers arrive.  Stepping in Cartesian — 30
        straight-line waypoints from the arm's actual current EE to the
        target — keeps every intermediate pose on a near-vertical descent
        with IK locked to one branch, so nothing passes over the object
        horizontally.
        """
        arm = self.robot.get_arm(arm_tag)
        current_ee = np.array(arm.get_ee_pose()[:3], dtype=float)
        target_pos = np.array(tcp_pose.p, dtype=float)
        self._move_cartesian(
            current_ee, target_pos, arm_tag,
            n_steps=30, sim_per_step=15,
        )
        return True

    # Default slowdown factor for plan replay.  Each MPLib waypoint is
    # replayed across `_REPLAY_SUBSTEPS` sim ticks (linearly interpolated)
    # so peak joint velocity = qpos_step / (sim_dt * substeps).  At the
    # default qpos_step=0.1 with sim_dt=0.001, a single-step replay drives
    # joints at 100 rad/s — well past the PD tracking budget — and the
    # wrist visibly swings during the descent and lift.  4× substepping
    # caps peak velocity at 25 rad/s, which the boosted PD (kp=20000,
    # kv=2500) can track without lag.
    _REPLAY_SUBSTEPS = 4

    def _slow_replay(self, pos, vel, arm_tag):
        """Upsample a planned qpos trajectory `_REPLAY_SUBSTEPS`× via linear
        interpolation, then hand to ``execute_plan``.  Same semantics as
        calling ``execute_plan`` directly but with N× more sim ticks per
        original waypoint, so joint velocity is reduced N×.
        """
        from ..planning.base import PlanResult
        sub = int(self._REPLAY_SUBSTEPS)
        if pos is None or pos.shape[0] < 2 or sub <= 1:
            self.execute_plan(PlanResult(True, pos, vel), arm_tag)
            return
        n_orig = pos.shape[0]
        n_new = (n_orig - 1) * sub + 1
        t_orig = np.linspace(0.0, 1.0, n_orig)
        t_new = np.linspace(0.0, 1.0, n_new)
        pos_up = np.empty((n_new, pos.shape[1]), dtype=pos.dtype)
        for j in range(pos.shape[1]):
            pos_up[:, j] = np.interp(t_new, t_orig, pos[:, j])
        vel_up = np.zeros_like(pos_up)
        self.execute_plan(PlanResult(True, pos_up, vel_up), arm_tag)

    def _move_seeded(self, link_pose7, arm_tag):
        """RRT-plan with the goal qpos pre-computed by our seeded IK.

        MPLib's ``plan_pose`` (what ``move_and_execute`` uses) runs its own
        IK internally, ignoring our ``_cached_target`` seed.  That's how
        the pre-grasp step lands in a wrist-wrapped IK branch that later
        Cartesian steps then continue from — producing the ~180°+ wrist
        spins between the safe-pose and the grasp descent.  Solving IK
        ourselves with the seed and feeding the result into mplib's
        ``plan_qpos`` (joint-space RRT goal) makes the plan continuous
        with the prior motion's branch.
        """
        from ..robot.franka_robot import to_numpy as _to_numpy
        arm = self.robot.get_arm(arm_tag)
        ik_fn = getattr(arm.entity, "inverse_kinematics", None)
        if ik_fn is None:
            return self.move_and_execute(link_pose7, arm_tag)
        seed = getattr(arm, "_cached_target", None)
        if seed is None:
            seed = _to_numpy(arm.entity.get_qpos())
        seed = np.asarray(seed, dtype=np.float64).ravel()
        try:
            qpos_sol = ik_fn(
                link=arm.ee_link,
                pos=np.array(link_pose7[:3]),
                quat=np.array(link_pose7[3:7]),
                init_qpos=seed,
            )
        except Exception as e:
            return self.move_and_execute(link_pose7, arm_tag)
        if qpos_sol is None:
            return self.move_and_execute(link_pose7, arm_tag)
        arm_qpos_goal = _to_numpy(qpos_sol).ravel()[:arm.n_arm]

        # Joint-7 (wrist roll) un-wrap: a top-down EE has a redundant wrist DOF
        # that IK can return as `seed[6] ± 2π` — kinematically equivalent but
        # makes the arm spin a full revolution during execution.  If the IK
        # answer differs from the seed by ≈2π on j7, wrap it to the closer
        # branch.  User feedback 2026-04-29: "robot rotates widely" during
        # path execution — this is the cause.
        if len(arm_qpos_goal) >= 7 and len(seed) >= 7:
            j7_seed = float(seed[6])
            j7_sol = float(arm_qpos_goal[6])
            j7_diff = j7_sol - j7_seed
            if j7_diff > np.pi:
                arm_qpos_goal[6] = j7_sol - 2 * np.pi
            elif j7_diff < -np.pi:
                arm_qpos_goal[6] = j7_sol + 2 * np.pi

        # Verify IK convergence by joint-distance heuristic BEFORE planning.
        # Genesis DLS IK has no convergence gate — it can return a qpos far
        # from the seed (a different IK branch) when the target is unreachable
        # or ill-conditioned.  Such a goal makes plan_qpos chart a wild path
        # that ejects any held object mid-trajectory (cf. seed 2 woodenblock
        # 2026-04-28: post-exec xy_err=0.50m).  If the IK answer requires a
        # bigger joint swing than `IK_BRANCH_FLIP_THRESHOLD`, route directly
        # to plan_pose (MPLib's internal IK multi-starts) instead.
        seed_arm = np.asarray(seed, dtype=np.float64).ravel()[:arm.n_arm]
        joint_delta = float(np.linalg.norm(arm_qpos_goal - seed_arm))
        if joint_delta > 3.5:  # ~200 deg cumulative across 7 joints
            return self.move_and_execute(link_pose7, arm_tag)

        planner = arm.planner
        goal_padded = planner._pad_qpos(np.asarray(arm_qpos_goal, dtype=np.float64))
        current_padded = planner._pad_qpos(np.asarray(arm.get_arm_qpos(), dtype=np.float64))
        try:
            result = planner._mplib.plan_qpos(
                goal_qposes=[goal_padded],
                current_qpos=current_padded,
                time_step=1 / 250,
                planning_time=10,
                rrt_range=0.3,
                verbose=False,
            )
        except Exception as e:
            return self.move_and_execute(link_pose7, arm_tag)
        if result and result.get("status") == "Success":
            pos = result["position"]
            vel = result.get("velocity", np.zeros_like(pos))
            self._slow_replay(pos, vel, arm_tag)
            # Verify arm actually reached the target XY.  Genesis IK can
            # converge to a bad local minimum when the seed is far from
            # the target; plan_qpos executes to that bad goal and leaves
            # the EE 20–30 cm off.  Compare xy only (link/tcp z differ by
            # tcp_offset, but for top-down poses the xy is shared).  If
            # off, retry via plan_pose (MPLib's internal IK multi-starts).
            ee_xy = np.array(arm.get_ee_pose()[:2], dtype=float)
            tgt_xy = np.array(link_pose7[:2], dtype=float)
            xy_err = float(np.linalg.norm(ee_xy - tgt_xy))
            if xy_err > 0.05:
                return self.move_and_execute(link_pose7, arm_tag)
            return result
        return self.move_and_execute(link_pose7, arm_tag)

    def _move_screw(self, target_pos, arm_tag):
        """Straight-line Cartesian motion via MPLib plan_screw; holds gripper.

        Retries plan_screw with decreasing qpos_step (0.1, 0.05, 0.02) — finer
        resolution lets MPLib solve the IK branch continuity constraint on
        waypoints that failed at coarser step sizes (the default qpos_step=0.1
        can miss valid screw paths when the 7-DOF redundant joint needs to
        rotate by more than 0.1 rad between waypoints).  Only falls back to
        `plan_pose` if all three resolutions fail — the plan_pose RRT path
        has been observed to swing laterally by 2-16 cm for descents,
        knocking target objects off their spawn (measured on seeds 30, 60).
        """
        arm = self.robot.get_arm(arm_tag)
        tcp_offset = arm.tcp_offset
        tcp = self._top_down_tcp(target_pos)
        link_pose = tcp_to_link_pose(tcp, tcp_offset)
        pose7 = link_pose.to_pose7()
        planner = arm.planner

        # Non-mplib planners (GenesisIKPlanner): use the planner's own
        # Cartesian straight-line solver instead of the mplib screw backend.
        if not hasattr(planner, "_mplib"):
            screw_fn = getattr(planner, "plan_screw_path", None)
            res = None
            if screw_fn is not None:
                try:
                    res = screw_fn(arm.get_arm_qpos(), pose7, log=False)
                except Exception as e:
                    print(f"[cate_coop] plan_screw_path exc: {e}")
            if res is not None and res.success and res.position.size > 0:
                self._slow_replay(res.position, res.velocity, arm_tag)
                return True
            self._move_seeded(pose7, arm_tag)
            return True

        goal = planner._to_mplib_pose(pose7)
        for qpos_step in (0.1, 0.05, 0.02):
            qpos = planner._pad_qpos(arm.get_arm_qpos())
            try:
                result = planner._mplib.plan_screw(
                    goal_pose=goal,
                    current_qpos=qpos,
                    time_step=1 / 250,
                    qpos_step=qpos_step,
                    wrt_world=True,
                    verbose=False,
                )
            except Exception as e:
                continue
            if result and result.get("status") == "Success":
                pos = result["position"]
                vel = result.get("velocity", np.zeros_like(pos))
                self._slow_replay(pos, vel, arm_tag)
                return True
        # Route fallback through `_move_seeded` so the IK is seeded from the
        # previously commanded qpos.  Unseeded `plan_pose` flipped the wrist
        # 180° on cate-coop seed-8 seal descent (15.6 cm xy err).
        self._move_seeded(pose7, arm_tag)
        return True

    def _move_tcp_screw(self, target_tcp: Pose, arm_tag: str) -> bool:
        """Screw to a TCP pose while preserving the requested TCP orientation.

        Used after a manual/YAML grasp so the immediate lift does not coerce
        the wrist back to this task's synthetic top-down orientation.
        """
        arm = self.robot.get_arm(arm_tag)
        link_pose = tcp_to_link_pose(target_tcp, arm.tcp_offset)
        result = arm.planner.plan_screw_path(
            arm.get_arm_qpos(), link_pose.to_pose7(), log=False,
        )
        if not result.success:
            return self.move_and_execute(link_pose.to_pose7(), arm_tag) is not None
        # Manual-grasp lift/transport must use the same bounded replay speed as
        # the radius path.  Raw planner replay can retain boxes but sheds round
        # objects (notably the apple) during a long lateral carry even after a
        # clean lift.  Preserve the requested TCP orientation while slowing
        # each joint-space segment by `_REPLAY_SUBSTEPS`.
        self._slow_replay(result.position, result.velocity, arm_tag)
        return True

    def _joint_interp_to(self, pose7, arm_tag, n_steps=40, sim_per_step=12):
        """Manual joint-space linear interpolation from measured qpos to
        seeded-IK goal qpos.  Short (~30-40 steps), bounded lateral sweep.
        Falls back to plan_pose if IK fails or post-exec xy is off.
        """
        from ..robot.franka_robot import to_numpy, _get_dof_idx
        arm = self.robot.get_arm(arm_tag)
        ik_fn = getattr(arm.entity, "inverse_kinematics", None)
        if ik_fn is None:
            return self.move_and_execute(pose7, arm_tag)
        measured = np.asarray(to_numpy(arm.entity.get_qpos()), dtype=float).ravel()
        try:
            qpos_sol = ik_fn(
                link=arm.ee_link,
                pos=np.array(pose7[:3]),
                quat=np.array(pose7[3:7]),
                init_qpos=measured,
            )
        except Exception as e:
            return self.move_and_execute(pose7, arm_tag)
        if qpos_sol is None:
            return self.move_and_execute(pose7, arm_tag)
        goal_full = np.asarray(to_numpy(qpos_sol), dtype=float).ravel()
        base = getattr(arm, "_cached_target", None)
        if base is None:
            base = measured
        base = np.asarray(base, dtype=float).copy()
        # Build goal qpos: arm joints from IK, fingers from _cached_target.
        target_qpos = base.copy()
        for j_idx, j in enumerate(arm.arm_joints):
            if j is not None:
                dof_idx = _get_dof_idx(j)
                if dof_idx is not None:
                    target_qpos[dof_idx] = float(goal_full[dof_idx])
        # Linear interp in joint space from measured to target_qpos.
        start_qpos = base.copy()
        for j_idx, j in enumerate(arm.arm_joints):
            if j is not None:
                dof_idx = _get_dof_idx(j)
                if dof_idx is not None:
                    start_qpos[dof_idx] = float(measured[dof_idx])
        for step in range(1, n_steps + 1):
            t = step / n_steps
            interp = start_qpos * (1 - t) + target_qpos * t
            # Keep fingers from _cached_target (don't interp grip value).
            for j_idx, j in enumerate(arm.finger_joints):
                if j is not None:
                    dof_idx = _get_dof_idx(j)
                    if dof_idx is not None:
                        interp[dof_idx] = float(base[dof_idx])
            arm.entity.control_dofs_position(interp)
            for _ in range(sim_per_step):
                self.step_sim()
        arm._cached_target = interp.copy()
        # Post-exec validation — if Genesis IK landed at local minimum, retry.
        ee_xy = np.array(arm.get_ee_pose()[:2], dtype=float)
        tgt_xy = np.array(pose7[:2], dtype=float)
        xy_err = float(np.linalg.norm(ee_xy - tgt_xy))
        if xy_err > 0.05:
            self.move_and_execute(pose7, arm_tag)
        return True

    def _move_cartesian(self, start_pos, end_pos, arm_tag, n_steps=50, sim_per_step=20):
        """Slow Cartesian interpolation — IK at each waypoint, PD controlled.

        This is much smoother than the RRT motion planner, keeping grip
        forces stable during transport.
        """
        arm = self.robot.get_arm(arm_tag)
        from ..robot.franka_robot import to_numpy, _get_dof_idx
        # `base_target` preserves FINGER grip from the last commanded target —
        # reading measured finger qpos during a closed grip returns a
        # slightly compressed value, and commanding that back would decay
        # grip force to zero.  Arm joints, however, need to be reseeded from
        # the MEASURED qpos at entry so the first waypoint's IK uses the
        # branch the arm is PHYSICALLY in (not a stale commanded pose left
        # over from MPLib/pre-grasp, which due to PD tracking lag differs
        # from reality by 1–3 cm).  Without this refresh, waypoint 1's IK
        # picks a neighboring branch and the wrist flips 180°+ on the first
        # commanded step — that's the "arm rotates weirdly sometimes"
        # failure mode.
        base_target = getattr(arm, "_cached_target", None)
        if base_target is None:
            base_target = to_numpy(arm.entity.get_qpos())
        base_target = np.array(base_target, dtype=float).copy()
        measured_qpos = np.asarray(to_numpy(arm.entity.get_qpos()), dtype=float).ravel()
        seed_target = base_target.copy()
        for j in arm.arm_joints:
            if j is not None:
                dof_idx = _get_dof_idx(j)
                if dof_idx is not None:
                    seed_target[dof_idx] = float(measured_qpos[dof_idx])
        arm._cached_target = seed_target  # so `_solve_ik` at waypoint 1 seeds from here
        qpos_full = base_target.copy()
        for i in range(1, n_steps + 1):
            t = i / n_steps
            pos = start_pos * (1 - t) + end_pos * t
            tcp = self._top_down_tcp(pos)
            arm_qpos = self._solve_ik(tcp, arm_tag)
            if arm_qpos is None:
                continue
            qpos_full = base_target.copy()
            for j_idx, j in enumerate(arm.arm_joints):
                if j is not None:
                    dof_idx = _get_dof_idx(j)
                    if dof_idx is not None:
                        qpos_full[dof_idx] = float(arm_qpos[j_idx])
            arm.entity.control_dofs_position(qpos_full)
            # Refresh the IK seed to THIS waypoint's solution BEFORE the next
            # waypoint solves IK — keeps the next step's IK in the same
            # homotopy class as this step.  Without this, the next waypoint's
            # `_solve_ik` reads `_cached_target` from before the trajectory
            # started (or a stale cache) and can hop to a different branch.
            arm._cached_target = qpos_full.copy()
            for _ in range(sim_per_step):
                self.step_sim()

    # Basket geometry (built from primitives in load_actors) + obstacle
    # inflation margins fed to mplib so RRT routes around basket walls.
    _BASKET_HALF_X = 0.096
    _BASKET_HALF_Y = 0.096
    _BASKET_HALF_Z = 0.056
    _BASKET_INNER_HALF_XY = 0.086
    _BASKET_FOOTPRINT_MARGIN = 0.002
    _OBSTACLE_MARGIN_XY = 0.020
    _OBSTACLE_MARGIN_Z = 0.010
    _OBSTACLE_RESOLUTION = 0.015

    @staticmethod
    def _sample_aabb_points(center, half, step=0.025):
        xs = np.arange(center[0] - half[0], center[0] + half[0] + 1e-6, step)
        ys = np.arange(center[1] - half[1], center[1] + half[1] + 1e-6, step)
        zs = np.arange(center[2] - half[2], center[2] + half[2] + 1e-6, step)
        X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
        return np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)

    def _basket_aabb_points(self, basket_idx):
        bp = np.array(self.basket_poses[basket_idx].p, dtype=float)
        half = np.array([
            self._BASKET_HALF_X + self._OBSTACLE_MARGIN_XY,
            self._BASKET_HALF_Y + self._OBSTACLE_MARGIN_XY,
            self._BASKET_HALF_Z + self._OBSTACLE_MARGIN_Z,
        ])
        return self._sample_aabb_points(bp, half, step=0.025)

    def _actor_aabb_points(self, actor, name, model_id):
        r = float(self._get_object_cross_section_radius(name, model_id))
        center = np.asarray(self._get_object_world_center(actor, name, model_id), dtype=float)
        half = np.array([
            r + self._OBSTACLE_MARGIN_XY,
            r + self._OBSTACLE_MARGIN_XY,
            r + self._OBSTACLE_MARGIN_Z,
        ])
        return self._sample_aabb_points(center, half, step=0.02)

    @staticmethod
    def _to_numpy_array(value) -> np.ndarray:
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        return np.asarray(value, dtype=float)

    def _actor_footprint_aabb(self, actor, obj_name, model_id):
        entity = getattr(actor, "entity", None)
        if entity is not None and hasattr(entity, "get_AABB"):
            try:
                aabb = self._to_numpy_array(entity.get_AABB()).reshape(-1, 3)
                lo = np.min(aabb, axis=0)
                hi = np.max(aabb, axis=0)
                return lo[:2], hi[:2]
            except Exception:
                pass

        center = self._get_object_world_center(actor, obj_name, model_id)
        ext = np.asarray(self._get_model_extents(obj_name, model_id), dtype=float)
        half_xy = np.max(ext[:2]) / 2.0
        return center[:2] - half_xy, center[:2] + half_xy

    @staticmethod
    def _get_model_extents(object_name: str, model_id: int = 0) -> np.ndarray:
        md_path = ASSETS_PATH / "objects" / object_name / f"model_data{model_id}.json"
        with open(md_path) as f:
            md = json.load(f)
        extents = np.array(md["extents"], dtype=float)
        scale = np.array(md.get("scale", [1, 1, 1]), dtype=float)
        return extents * scale

    def _update_planner_obstacles(self, arm_tag, skip_actor=None, skip_basket_idx=None):
        """Refresh mplib's obstacle cloud with inflated baskets + on-table objects.

        ``skip_actor`` (held object) and ``skip_basket_idx`` (destination basket
        during descent) are excluded so the active place-target is not rejected
        as in-collision.
        """
        pts = []
        for i in range(len(self.basket_poses)):
            if skip_basket_idx is not None and i == skip_basket_idx:
                continue
            pts.append(self._basket_aabb_points(i))
        for actor, obj_name, model_id, _ in self._sort_targets:
            if skip_actor is not None and actor is skip_actor:
                continue
            center = np.asarray(self._get_object_world_center(actor, obj_name, model_id), dtype=float)
            if any(np.linalg.norm(center[:2] - np.asarray(bp.p, dtype=float)[:2]) <= 0.11
                   for bp in self.basket_poses):
                continue
            pts.append(self._actor_aabb_points(actor, obj_name, model_id))
        pc = np.concatenate(pts, axis=0) if pts else np.zeros((0, 3))
        self.robot.get_arm(arm_tag).planner.update_obstacles(pc, resolution=self._OBSTACLE_RESOLUTION)

    def pick_and_place(self, pick: PickSpec, place: PlaceSpec, arm_tag: str) -> bool:
        """Dispatch to radius grasp by default and explicit YAML exceptions."""
        if pick.label == "035_apple":
            center = np.asarray(pick.get_center(), dtype=float)
            if float(center[0]) >= self.APPLE_YAML_MIN_X:
                return self._pick_and_place_yaml(pick, place, arm_tag)
        if pick.label in self.YAML_ONLY_GRASP_LABELS:
            return self._pick_and_place_yaml(pick, place, arm_tag)
        if not self.USE_RADIUS_GRASP:
            return self._pick_and_place_yaml(pick, place, arm_tag)

        actor = pick.actor
        if actor is None:
            return False
        # Skip-list: round objects the parallel jaw can't hold (they eject).
        if pick.label in self.RADIUS_SKIP_LABELS:
            return False

        # --- Regression guards (so ungraspable objects fail GRACEFULLY) -------
        # (a) Height cap: objects taller than a stable top-down grip can hold
        #     (the 12 cm milk-box carton) are left standing on the table rather
        #     than grabbed near the top and tipped.
        from ..robot.franka_robot import to_numpy as _to_numpy
        try:
            aabb0 = np.asarray(_to_numpy(actor.entity.get_AABB()), dtype=float)
            obj_height = float(aabb0[1][2] - aabb0[0][2])
        except Exception:
            obj_height = 0.0
        if obj_height > self.RADIUS_MAX_GRASP_HEIGHT:
            return False
        # Live cross-section radius from the simulator AABB (robust to the
        # model_data extents being off under the z-up GLB load); fall back to
        # the model_data cross-section if the AABB read fails.
        obj_radius = float(pick.radius)
        try:
            aabb = np.asarray(_to_numpy(actor.entity.get_AABB()), dtype=float)
            ext_xy = (aabb[1] - aabb[0])[:2]
            obj_radius = float(np.min(ext_xy) / 2.0)
        except Exception:
            pass
        # Per-object close fraction.  Priority: categorize-local override (tuned
        # for THIS pool's models + the radius below-equator scoop) → catalog
        # ``close_value`` → mixin radius formula.  The local map exists because
        # the catalog values are shared with put_object_cabinet, where a tighter
        # apple close (0.35) holds; here the radius scoop on apple@1 ejects the
        # round fruit at 0.35, so it needs a looser cage.
        close_value = None
        if pick.label in self.RADIUS_CLOSE_OVERRIDE:
            close_value = self.RADIUS_CLOSE_OVERRIDE[pick.label]
        else:
            try:
                close_value = get_entry(pick.label).close_value
            except Exception:
                close_value = None
        radius_pick = RadiusPickSpec(
            get_center=pick.get_center,
            radius=obj_radius,
            label=pick.label,
            close_value=close_value,
            get_grasp_yaw=(
                (lambda yaw=float(self.RADIUS_GRASP_YAW[pick.label]): yaw)
                if pick.label in self.RADIUS_GRASP_YAW
                else None
            ),
        )
        radius_place = RadiusPlaceSpec(
            pos=np.asarray(place.pos, dtype=float),
            label=place.label,
            drop_from_z=place.drop_from_z,
            release=True,
        )
        return bool(
            TopDownPickPlaceMixin.pick_and_place(self, radius_pick, radius_place, arm_tag)
        )

    def _pick_and_place_yaml(self, pick: PickSpec, place: PlaceSpec, arm_tag: str) -> bool:
        """Pick a loose object (described by ``pick``) and drop it at ``place``.

        Semantics:
          - Top-down approach, below-equator grasp offset (``radius * 0.3``).
          - Two-phase transit: lateral at safe_z, vertical descent to pre-grasp.
          - Partial close sized to ``radius * 0.7`` (avoids mesh penetration).
          - Hover-and-release over ``place.pos`` if ``drop_from_z`` is set;
            otherwise descend to ``place.pos.z`` before opening the gripper.
        """
        arm = self.robot.get_arm(arm_tag)
        tcp_offset = arm.tcp_offset

        # Reset IK seed to MEASURED qpos at the start of every pick.  The
        # prior pick's final ``_cached_target`` is a qpos at a *different*
        # column's basket; feeding it to Genesis IK for the NEW pre-grasp
        # target can lock onto a bad local minimum where FK doesn't match
        # the target (seen on seed 20 D3 apple — 28 cm err, never touched
        # the object).  Measured qpos is always a valid starting config.
        from ..robot.franka_robot import to_numpy as _to_numpy
        arm._cached_target = np.asarray(
            _to_numpy(arm.entity.get_qpos()), dtype=float
        ).ravel().copy()

        obj_center = pick.get_center()
        obj_radius = pick.radius

        # Transport ABOVE the drop target.  Default safe_z clears the basket
        # rim (baskets are 28cm tall; top at TABLE_TOP_Z + 30cm).  Callers
        # drop into deeper containers by passing a higher ``drop_from_z``.
        safe_z = place.drop_from_z if place.drop_from_z is not None else self.TABLE_TOP_Z + 0.38

        # Open gripper
        self.open_gripper(arm_tag)

        if not self.USE_YAML_GRASPS:
            return False
        if pick.label in self.YAML_GRASP_SKIP:
            return False

        if self._USE_OBSTACLE_INFLATION:
            self._update_planner_obstacles(arm_tag, skip_actor=pick.actor)
        actor = pick.actor
        if actor is None:
            return False

        model_id = None
        for _a, _n, _m, _ in self._sort_targets:
            if _a is actor:
                model_id = _m
                break
        if model_id is None:
            return False

        obj_pose_for_grasp = actor.get_pose()
        object_scale = self._get_object_scale(pick.label, model_id)
        result = self._select_and_execute_task_grasp(
            pick.label, obj_pose_for_grasp, arm_tag,
            model_id=model_id, object_scale=object_scale,
            max_candidates=20,
        )
        if result is None:
            return False

        _grasp_link, _pre_link, grasp = result
        ee_pose = np.array(arm.get_ee_pose(), dtype=float)
        grasp_pos = ee_pose[:3].copy()
        grasp_quat = ee_pose[3:7].copy()
        obj_pre_close = pick.get_center()

        # Close target priority:
        #   1. Per-object metadata `grasp_close_target_m` (set on PickSpec from
        #      the object's `model_data{N}.json`).  Used to override the
        #      formula for objects that need a tighter clamp than the size
        #      heuristic gives (e.g. tall cylinders like 100_seal where
        #      friction along the axis must hold the object's weight).
        #   2. Adaptive formula.  Round/small meshes (r<3 cm, e.g. apple) need
        #      a shallow squeeze (`r-0.005`) — deeper commands let fingers
        #      overpower mesh contact before caging, shoving the object
        #      sideways.  Flat/large meshes (r>=3 cm, e.g. wooden block) need
        #      a deep squeeze (`r*0.7`) — fingers contact a flat face
        #      squarely, the block wins the PD push and stops fingers at the
        #      face with ~100 N firm grip.  Verified: deliver apple → 4/4
        #      with shallow; cate-coop block → 3/3 with deep.
        r = float(obj_radius)
        ct_meta = getattr(pick, "close_target_m", None)
        if ct_meta is not None:
            close_target = max(0.010, float(ct_meta))
        elif r < 0.030:
            close_target = max(0.010, r - 0.005)
        else:
            close_target = max(0.010, r * 0.7)
        self.set_gripper(close_target / 0.04, arm_tag)
        # Settle 300 ticks (was 200) so the gripper-object contact reaches
        # static equilibrium before the lift accelerates the object — too
        # short a settle leaves residual finger velocity that gets reversed
        # by the lift's upward command, breaking the friction grip.
        for _ in range(300):
            self.step_sim()
        # Use the FRESH object centre for the lift target — `obj_center` was
        # captured at top-of-pick and is stale if the pre-grasp transit
        # bumped the object.
        # Re-read object centre for the lift target — `obj_center_now` is
        # only set in the v14b synthetic branch; for the YAML path we need
        # a fresh read here regardless.
        lift_obj_center = pick.get_center()
        lift_mid = grasp_pos.copy()
        lift_mid[2] = lift_obj_center[2] + obj_radius  # lift to just above object top
        if not self._move_tcp_screw(Pose(lift_mid, grasp_quat), arm_tag):
            self.open_gripper(arm_tag)
            for _ in range(80):
                self.step_sim()
            return False

        # Debug: check grip
        from ..robot.franka_robot import _get_dof_idx, to_numpy
        qpos_after = to_numpy(arm.entity.get_qpos())
        finger_qpos = []
        for j in arm.finger_joints:
            if j is not None:
                idx = _get_dof_idx(j)
                if idx is not None:
                    finger_qpos.append(float(qpos_after[idx]))
        obj_pos_after = pick.get_center()

        # Lift to safe height via screw motion (straight-line, gripper held).
        # Keep the YAML grasp orientation through transport; switching back
        # to the synthetic top-down TCP can rotate round objects out of the
        # fingers after an otherwise good grasp.
        lift_pos = np.array([grasp_pos[0], grasp_pos[1], safe_z])
        if not self._move_tcp_screw(Pose(lift_pos, grasp_quat), arm_tag):
            self.open_gripper(arm_tag)
            for _ in range(80):
                self.step_sim()
            return False

        obj_pos_lifted = pick.get_center()
        lift_dz = float(obj_pos_lifted[2] - obj_pos_after[2])
        if pick.label != "100_seal" and lift_dz < 0.04:
            self.open_gripper(arm_tag)
            for _ in range(120):
                self.step_sim()
            return False

        # Transport to above drop target, then release. No multi-segment
        # drop — a second screw motion into the container occasionally
        # flips the wrist and hurls the object.  For hover-and-release
        # (basket pattern) the object falls cleanly from safe_z.
        # Keep the destination basket in the obstacle set — the path goes
        # ABOVE all baskets at safe_z, so a planner detour that dips toward
        # another basket gets repelled.
        if self._USE_OBSTACLE_INFLATION:
            self._update_planner_obstacles(arm_tag, skip_actor=pick.actor)
        place_xyz = np.array([place.pos[0], place.pos[1], safe_z], dtype=float)
        if not self._move_tcp_screw(Pose(place_xyz, grasp_quat), arm_tag):
            self.open_gripper(arm_tag)
            for _ in range(80):
                self.step_sim()
            return False

        # Settle so the object stops swinging in the gripper.
        for _ in range(100):
            self.step_sim()

        obj_pos_transport = pick.get_center()

        # If caller wants a descent-before-release (e.g. place on a plate),
        # move TCP down to place.pos.z first.
        if place.drop_from_z is None and len(place.pos) >= 3:
            # Drop the destination basket from obstacles so its own walls
            # don't block the descent.
            if self._USE_OBSTACLE_INFLATION:
                self._update_planner_obstacles(
                    arm_tag,
                    skip_actor=pick.actor,
                    skip_basket_idx=place.basket_idx,
                )
            descend_pos = np.array([place.pos[0], place.pos[1], float(place.pos[2])])
            if not self._move_tcp_screw(Pose(descend_pos, grasp_quat), arm_tag):
                self.open_gripper(arm_tag)
                for _ in range(80):
                    self.step_sim()
                return False
            for _ in range(50):
                self.step_sim()

        # Release.  Let the object drop under gravity.
        self.open_gripper(arm_tag)
        for _ in range(250):
            self.step_sim()

        obj_pos_release = pick.get_center()

        return True

    def _open_basket_lids(self):
        """Set basket lid joints to URDF-default (open) pose and hold with
        high PD gains so falling objects don't push the lids closed."""
        for actor in self.basket_actors:
            entity = getattr(actor, "entity", None)
            if entity is None:
                continue
            try:
                q = entity.get_qpos()
                import torch
                q_np = q.cpu().numpy() if hasattr(q, "cpu") else np.asarray(q)
                q_np = np.asarray(q_np).ravel().astype(float)
                q_np[:] = 0.0
                entity.set_qpos(q_np)
                # High kp on every lid DoF so they stay open
                try:
                    kp = entity.get_dofs_kp()
                    kv = entity.get_dofs_kv()
                    kp_np = kp.cpu().numpy() if hasattr(kp, "cpu") else np.asarray(kp)
                    kv_np = kv.cpu().numpy() if hasattr(kv, "cpu") else np.asarray(kv)
                    kp_np = np.asarray(kp_np).ravel().astype(float)
                    kv_np = np.asarray(kv_np).ravel().astype(float)
                    kp_np[:] = 500.0
                    kv_np[:] = 50.0
                    set_dofs_kp_kv_compat(entity, kp=kp_np, kv=kv_np)
                    entity.control_dofs_position(q_np)
                except Exception as e:
                    pass
            except Exception as e:
                pass

    def _avatar_pick_and_place(self, actor, obj_name, model_id, basket_idx,
                                wait=True,
                                approach_frames=320, transport_frames=400,
                                retract_frames=320):
        """Avatar picks its loose object via real shoulder→hand IK (fabrik)
        and drops it above the matching box.  Delegates to the general
        `AvatarController.pick_and_place` action — see
        `envs/avatar/motions/pick_place_motion.py` and `docs/avatar_arm_ik.md`.

        Hand selection: right hand for x ≤ 0, left hand for x > 0.  Since
        the left shoulder is a mirror of the right across x=0, using the
        closer hand keeps every target well inside the ~0.65 m palm reach.

        When `wait=False` the call kicks off the motion non-blocking and
        returns immediately — subsequent `step_sim()` calls (e.g. from the
        robot's concurrent pick-and-place) will advance the avatar motion
        in parallel.  Use `wait=True` if the caller is not already stepping
        the simulator for some other reason.
        """
        if self.avatar is None:
            return False
        pick_pos = self._get_object_world_center(actor, obj_name, model_id)
        basket_pos = np.array(self.basket_poses[basket_idx].p, dtype=float)
        # Drop slightly above the box rim so gravity seats the object inside.
        place_pos = basket_pos.copy()
        place_pos[2] += 0.10

        # Mirror arm choice: right hand (1) for left-side targets, left hand
        # (0) for right-side targets.  The midpoint (x=0) is assigned to the
        # right hand by convention.
        hand_id = 0 if float(pick_pos[0]) > 0.0 else 1

        self.avatar.pick_and_place(
            pick_pos=pick_pos,
            place_pos=place_pos,
            attach_obj=actor,
            hand_id=hand_id,
            approach_frames=approach_frames,
            transport_frames=transport_frames,
            retract_frames=retract_frames,
        )
        if wait:
            while not self.avatar.spare():
                self.step_sim()
        return True

    # ------------------------------------------------------------------
    # Eval-mode hooks (loop pattern): the avatar must keep firing
    # pick-and-place motions across the rollout, one per object on its
    # half of the table, while the policy independently sorts the
    # robot's objects.  The first motion starts at a seeded random policy
    # step; each later motion queues when the previous one finishes.
    # ------------------------------------------------------------------
    def _eval_at_reset(self):
        # Permute the avatar's objects so the order varies across seeds
        # — keeps the scenario stochastic without changing what it tests.
        self._eval_avatar_pick_order = [
            int(i) for i in np.random.permutation(len(self._avatar_sort_targets))
        ]
        self._eval_avatar_target_idx = 0
        self._eval_trigger_step = 0
        self._eval_avatar_started = True
        # Open the basket lids up front (same as play_once) so both the
        # avatar and the policy see consistent geometry.
        if hasattr(self, "_open_basket_lids"):
            self._open_basket_lids()
        self._eval_fire_next_avatar_motion()

    def _eval_at_step(self, step_idx: int):
        if (not self._eval_avatar_started
                and step_idx >= getattr(self, "_eval_trigger_step", 0)):
            self._eval_avatar_started = True
            self._eval_fire_next_avatar_motion()
            return
        if not self._eval_avatar_started:
            return

        # As soon as the avatar finishes its current motion, queue the next
        # one until the avatar-side objects are exhausted.
        if (self._eval_avatar_target_idx < len(self._eval_avatar_pick_order)
                and self.avatar.spare()):
            self._eval_fire_next_avatar_motion()

    def _eval_fire_next_avatar_motion(self):
        i = self._eval_avatar_target_idx
        idx = self._eval_avatar_pick_order[i]
        actor, obj_name, model_id, basket_idx = self._avatar_sort_targets[idx]
        self._avatar_pick_and_place(
            actor, obj_name, model_id, basket_idx, wait=False,
        )
        self._eval_avatar_target_idx = i + 1

    def play_blind_once(self) -> bool:
        arm_tag = "right"
        self._boost_arm_pd()
        self._open_basket_lids()
        self._move_to_safe(arm_tag)

        def _robot_sort_idx(idx):
            actor, obj_name, model_id, basket_idx = self._sort_targets[idx]
            pick = PickSpec(
                get_center=lambda a=actor, n=obj_name, m=model_id: self._get_object_world_center(a, n, m),
                radius=self._get_object_cross_section_radius(obj_name, model_id),
                label=obj_name,
                close_target_m=self._get_grasp_close_target(obj_name, model_id),
                actor=actor,
            )
            basket_pos = np.array(self.basket_poses[basket_idx].p, dtype=float)
            drop_from_z = basket_pos[2] + (0.10 if obj_name == "100_seal" else 0.18)
            place = PlaceSpec(
                pos=basket_pos,
                label=f"basket {basket_idx}",
                basket_idx=basket_idx,
                drop_from_z=float(drop_from_z),
            )
            ok = self.pick_and_place(pick, place, arm_tag)
            self._move_to_safe(arm_tag)
            obj_center = self._get_object_world_center(actor, obj_name, model_id)
            dist_xy = float(np.linalg.norm(obj_center[:2] - basket_pos[:2]))
            return bool(ok and dist_xy <= 0.10)

        any_success = False
        self._blind_robot_order = [
            int(i) for i in np.random.permutation(len(self._sort_targets))
        ]
        for idx in self._blind_robot_order:
            ok_robot = _robot_sort_idx(idx)
            any_success = any_success or ok_robot
        if self.avatar is not None:
            tail = 0
            while not self.avatar.spare() and tail < 2400:
                self.step_sim()
                tail += 1
        for _ in range(200):
            self.step_sim()
        return any_success

    def play_once(self) -> bool:
        arm_tag = "right"
        self._boost_arm_pd()
        self._open_basket_lids()
        self._move_to_safe(arm_tag)

        # Cooperative loop: the avatar chooses the category order, and the
        # robot follows that same category on its side.  The robot/avatar
        # target lists are independently permuted during layout, so matching
        # by list index is wrong; match by object/category name instead.
        avatar_order = [
            int(i) for i in np.random.permutation(len(self._avatar_sort_targets))
        ]
        self._full_state_avatar_order = list(avatar_order)
        self._full_state_robot_order = []
        robot_idx_by_name = {
            obj_name: int(i)
            for i, (_actor, obj_name, _model_id, _basket_idx)
            in enumerate(self._sort_targets)
        }
        seal_positions = [
            pos for pos, avatar_idx in enumerate(avatar_order)
            if self._avatar_sort_targets[avatar_idx][1] == "100_seal"
        ]
        if seal_positions and seal_positions[0] == len(avatar_order) - 1 and len(avatar_order) > 1:
            seal_avatar_idx = avatar_order[seal_positions[0]]
            robot_seal_idx = robot_idx_by_name.get("100_seal")
            if robot_seal_idx is not None and self._sort_targets[robot_seal_idx][3] == 0:
                avatar_order.pop(seal_positions[0])
                avatar_order.insert(1, seal_avatar_idx)
        any_success = False

        def _robot_sort_idx(idx, *, targets=None):
            targets = self._sort_targets if targets is None else targets
            actor, obj_name, model_id, basket_idx = targets[idx]
            pick = PickSpec(
                get_center=lambda a=actor, n=obj_name, m=model_id: self._get_object_world_center(a, n, m),
                radius=self._get_object_cross_section_radius(obj_name, model_id),
                label=obj_name,
                close_target_m=self._get_grasp_close_target(obj_name, model_id),
                actor=actor,
            )
            basket_pos = np.array(self.basket_poses[basket_idx].p, dtype=float)
            drop_from_z = basket_pos[2] + (0.10 if obj_name == "100_seal" else 0.18)
            place = PlaceSpec(
                pos=basket_pos,
                label=f"basket {basket_idx}",
                basket_idx=basket_idx,
                drop_from_z=float(drop_from_z),
            )
            ok = self.pick_and_place(pick, place, arm_tag)
            self._move_to_safe(arm_tag)
            obj_center = self._get_object_world_center(actor, obj_name, model_id)
            dist_xy = float(np.linalg.norm(obj_center[:2] - basket_pos[:2]))
            placed = bool(dist_xy <= 0.10)
            if not placed:
                pass
            return bool(ok and placed)

        if self.avatar is None:
            for idx in range(len(self._sort_targets)):
                ok_robot = _robot_sort_idx(idx)
                any_success = any_success or ok_robot
            if self.no_human_enabled():
                for _ in range(200):
                    self.step_sim()
                return any_success
            avatar_targets = getattr(self, "_avatar_sort_targets", [])
            for idx in range(len(avatar_targets)):
                ok_robot = _robot_sort_idx(idx, targets=avatar_targets)
                any_success = any_success or ok_robot
            for _ in range(200):
                self.step_sim()
            return any_success

        for avatar_idx in avatar_order:
            a_actor, a_name, a_model, a_basket = self._avatar_sort_targets[avatar_idx]
            robot_idx = robot_idx_by_name.get(a_name)
            if robot_idx is None:
                continue
            self._full_state_robot_order.append(int(robot_idx))
            ok_avatar = self._avatar_pick_and_place(
                a_actor, a_name, a_model, a_basket, wait=False
            )

            ok_robot = _robot_sort_idx(robot_idx)
            any_success = any_success or ok_avatar or ok_robot

            # Wait for avatar to finish its retract before next iteration
            # (skin chain re-seed).
            while not self.avatar.spare():
                self.step_sim()

        # Final settle so everything comes to rest for the check.
        for _ in range(200):
            self.step_sim()

        return any_success

    def evaluate(self) -> dict:
        metrics = super().evaluate()
        robot_order = list(getattr(self, "_blind_robot_order", []) or [])
        avatar_order = list(
            getattr(self, "_eval_avatar_pick_order", [])
            or getattr(self, "_full_state_avatar_order", [])
            or []
        )
        full_state_robot_order = list(getattr(self, "_full_state_robot_order", []) or [])
        metrics.update({
            "robot_target_names": [
                obj_name for _actor, obj_name, _model_id, _basket_idx in self._sort_targets
            ],
            "avatar_target_names": [
                obj_name for _actor, obj_name, _model_id, _basket_idx in self._avatar_sort_targets
            ],
            "n_robot_targets": int(len(self._sort_targets)),
            "n_avatar_targets": int(len(self._avatar_sort_targets)),
            "eval_avatar_order": [
                self._avatar_sort_targets[int(i)][1]
                for i in avatar_order
                if int(i) < len(self._avatar_sort_targets)
            ],
            "blind_robot_order": [
                self._sort_targets[int(i)][1]
                for i in robot_order
                if int(i) < len(self._sort_targets)
            ],
            "full_state_robot_order": [
                self._sort_targets[int(i)][1]
                for i in full_state_robot_order
                if int(i) < len(self._sort_targets)
            ],
        })
        return metrics

    def _success_target_groups(self):
        groups = [("robot", self._sort_targets)]
        if not self.no_human_enabled():
            groups.append(("avatar", self._avatar_sort_targets))
        return groups

    def check_success(self) -> bool:
        if not self.plan_success:
            return False

        # Success requires the object footprint, not just its center, to sit
        # inside the basket.  Center-only checks accepted long objects perched
        # on a rim (e.g. seal seed 2: center dist 7 cm, visible failure).
        center_threshold = 0.10
        footprint_half = self._BASKET_INNER_HALF_XY + self._BASKET_FOOTPRINT_MARGIN
        all_correct = True
        for side, targets in self._success_target_groups():
            for actor, obj_name, model_id, basket_idx in targets:
                obj_center = self._get_object_world_center(actor, obj_name, model_id)
                basket_pos = np.array(self.basket_poses[basket_idx].p)
                dist_xy = np.linalg.norm(obj_center[:2] - basket_pos[:2])
                lo_xy, hi_xy = self._actor_footprint_aabb(actor, obj_name, model_id)
                rel_lo = lo_xy - basket_pos[:2]
                rel_hi = hi_xy - basket_pos[:2]
                footprint_ok = bool(
                    np.all(rel_lo >= -footprint_half) and
                    np.all(rel_hi <= footprint_half)
                )
                center_ok = bool(dist_xy <= center_threshold)
                if not (center_ok and footprint_ok):
                    all_correct = False
                else:
                    pass

        # When avatar-collision tracking is enabled, any robot↔avatar contact
        # during the episode is a failure regardless of basket placement.
        if self.avatar_collision_checker is not None and self.avatar_collided:
            summary = self.avatar_collision_summary()
            all_correct = False

        return all_correct

    def _basket_metrics(self) -> dict:
        center_threshold = 0.10
        footprint_half = self._BASKET_INNER_HALF_XY + self._BASKET_FOOTPRINT_MARGIN
        per_object = []
        all_correct = True
        for side, targets in self._success_target_groups():
            for actor, obj_name, model_id, basket_idx in targets:
                obj_center = self._get_object_world_center(actor, obj_name, model_id)
                basket_pos = np.array(self.basket_poses[basket_idx].p, dtype=float)
                dist_xy = float(np.linalg.norm(obj_center[:2] - basket_pos[:2]))
                lo_xy, hi_xy = self._actor_footprint_aabb(actor, obj_name, model_id)
                rel_lo = lo_xy - basket_pos[:2]
                rel_hi = hi_xy - basket_pos[:2]
                center_ok = bool(dist_xy <= center_threshold)
                footprint_ok = bool(
                    np.all(rel_lo >= -footprint_half)
                    and np.all(rel_hi <= footprint_half)
                )
                ok = bool(center_ok and footprint_ok)
                all_correct = all_correct and ok
                per_object.append({
                    "side": side,
                    "object": obj_name,
                    "model_id": int(model_id),
                    "basket_idx": int(basket_idx),
                    "object_pos": obj_center.tolist(),
                    "basket_pos": basket_pos.tolist(),
                    "dist_xy": dist_xy,
                    "rel_aabb_lo_xy": rel_lo.tolist(),
                    "rel_aabb_hi_xy": rel_hi.tolist(),
                    "center_success": center_ok,
                    "footprint_success": footprint_ok,
                    "success": ok,
                })
        return {
            "basket_center_threshold": center_threshold,
            "basket_footprint_half_xy": float(footprint_half),
            "basket_all_correct": bool(all_correct),
            "basket_objects": per_object,
        }

    def evaluate(self) -> dict:
        metrics = super().evaluate()
        metrics.update(self._basket_metrics())
        if bool(self.config.get("eval_mode", False)):
            metrics.update({
                "eval_mode": True,
                "eval_trigger_step": getattr(self, "_eval_trigger_step", None),
                "eval_policy_step_count": int(getattr(
                    self, "_eval_policy_step_count", 0,
                )),
                "eval_avatar_started": bool(getattr(
                    self, "_eval_avatar_started", False,
                )),
                "eval_avatar_target_idx": int(getattr(
                    self, "_eval_avatar_target_idx", 0,
                )),
                "eval_avatar_order": [
                    self._avatar_sort_targets[i][1]
                    for i in getattr(self, "_eval_avatar_pick_order", [])
                ],
            })
        return metrics
