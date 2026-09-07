"""Place bread in basket: Franka picks 2 pieces of bread off the table
and drops them both into a stationary bread basket.  Single-arm, no
avatar.

Pure physics — no kinematic snap of the held bread to the gripper, no
``entity.set_pos`` on the bread once the rollout is running.  The bread
is held entirely by friction × normal force × gripper PD.

Bread A, bread B, and the basket xy are randomized inside a 0.40 ×
0.40 m region with a 0.18 m minimum pairwise separation.
"""

import json

import numpy as np
import transforms3d as t3d

from ..base_task import BaseTask
from ..genesis_compat import update_dofs_kp_kv_compat
from ..utils import (
    Actor, Pose, load_mesh, load_object, create_primitive, to_numpy,
    ASSETS_PATH, TABLE_HEIGHT,
)
from ..grasp import tcp_to_link_pose, load_grasp_poses
from ..object_catalog import get_entry


# Target object identity from the central catalog.
#
# NOTE (Genesis 1.0.0 graspability): the catalog default model_id=0 is a flat
# ~10 × 10 cm SQUARE loaf.  No parallel-jaw grip survives it — face-on needs
# 10 cm > the Franka's 8 cm jaw, and a diagonal grip only catches a corner
# wedge that ejects on lift (verified _bread_lift_exec_diag.py: bread rises
# 0.016 m while the EE rises 0.16 m -> SLIP, at every close value).  We pin
# this task to model_id=2 instead — a small ~5.8 × 3.7 × 5.1 cm roll whose
# shorter footprint axis (5.1 cm) fits the jaw, so a FACE-ON top-down grip
# gets full-width flat contact and holds through the lift+transit (verified
# _bread_m2_compress_sweep.py: gap <= 3.5 cm -> LIFTED 8/8).  The matching
# face-on grasps live in grasp_poses_franka.yml under model_ids:[2].  This is
# a task-local override (the shared catalog is untouched).
_BREAD_ENTRY = get_entry("075_bread")
_BREAD_ID = _BREAD_ENTRY.object_id
_BREAD_MODEL_ID = 2
_BASKET_ID = "076_breadbasket"   # receptacle (not a manipulated target)

# Randomization region inside Franka reach with the base mounted at
# (0, -0.65, 0.75).  Place pose puts the EE-link at z ≈ 1.11 m above
# ground, so the horizontal reach budget at place height is ~0.71 m
# from the base.  The corner (±0.20, +0.04) has horizontal distance
# sqrt(0.04 + 0.476) = 0.72 m — at the IK edge but still solvable.
#
# NOTE (Genesis 1.0.0): an earlier diagnosis blamed a near-base dead zone
# and shrank this region.  That was a red herring — the real bug was the
# grasp FRAME (see _pick_and_place_bread's logical-pose fix).  Once the
# grasp resolves top-down, scripts/_bread_topdown_band_diag.py shows the
# manual grasps are IK-reachable across this whole region (6/6 down to
# hd≈0.29 m), so the original region is kept as-is.
_REGION_X = (-0.20, +0.20)   # 0.40 m wide
_REGION_Y = (-0.36, +0.04)   # 0.40 m wide, shifted -y to keep IK in reach
# Min pairwise xy separation between the 3 placed objects (basket +
# 2 breads).  0.18 m gives bread/basket centres ~10 cm of free table
# between them — enough that the 10 cm-wide loaves don't overlap and
# enough that the gripper finger swing during the second pick doesn't
# bump into the basket or the first bread.
_MIN_SEP_M = 0.18
_SAMPLE_BATCH_ATTEMPTS = 64
_SAMPLE_POINT_ATTEMPTS = 64
_SAMPLE_FALLBACK_MARGIN = 0.05


def _read_model_data(name: str, model_id: int = 0) -> dict:
    p = ASSETS_PATH / "objects" / name / f"model_data{model_id}.json"
    with open(p) as f:
        return json.load(f)


def _read_scale(name: str, model_id: int = 0) -> float:
    raw = _read_model_data(name, model_id).get("scale", 1.0)
    return float(raw[0] if isinstance(raw, (list, tuple)) else raw)

def _read_bbox_center(name: str, model_id: int = 0) -> np.ndarray:
    """Bbox center in mesh-local frame (NOT scaled).  See memory
    `project_asset_mesh_origin` — `entity.get_pos()` returns the mesh
    origin, which can be far from the bbox center."""
    raw = _read_model_data(name, model_id).get("center", [0.0, 0.0, 0.0])
    return np.asarray(raw, dtype=np.float64)


def _sample_n_xy(
    n: int,
    rng: np.random.RandomState,
    max_outer: int = _SAMPLE_BATCH_ATTEMPTS,
):
    """Rejection-sample n xy positions inside the region with pairwise
    minimum separation.  Restarts the whole batch if a single position
    can't be placed within max_outer inner tries."""
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
    # Fallback: spread the points along the diagonal of the region.
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


class PlaceBreadInBasket(BaseTask):
    """Robot picks two pieces of bread from the table and places them
    both into a stationary bread basket."""

    INSTRUCTION = "put both pieces of bread into the basket"
    OBJECT_SET = ["075_bread"]
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

    # Authored-frame convention: mesh +Y maps to world +Z for both spawning
    # and grasp annotation composition.
    UPRIGHT_QUAT = np.array(
        [0.70710678, 0.70710678, 0.0, 0.0], dtype=np.float64,
    )
    # Mesh loading preserves the authored Y-up frame, so spawn with the same
    # explicit +Y-to-world-+Z rotation used by the grasp annotations.
    SPAWN_QUAT = UPRIGHT_QUAT.copy()
    BASKET_STATIC_Z_CLEARANCE = 0.001
    BASKET_FRICTION = 5.0
    BASKET_COACD_ERROR_THRESHOLD = 0.0
    BREAD_SPAWN_Z_CLEARANCE = 0.01
    # Spawn the (dynamic) bread this far above the table top and let it settle
    # flat — avoids depending on a particular mesh-axis being "up".
    BREAD_SPAWN_DROP = 0.04
    BREAD_FRICTION = 4.0
    # Drop the grasp TCP below the loaf's bbox centre so the fingers straddle
    # the body rather than grazing the top.  model_id=2 is ~3.7 cm thick, so a
    # ~1.1 cm drop puts the fingers around the lower third (verified
    # _bread_m2_compress_sweep.py: z_drop frac 0.3 and 0.5 both LIFTED).
    BREAD_GRASP_Z_DROP = 0.011
    # Face-on grip: close to a fixed ~2.8 cm finger gap.  The loaf's shorter
    # footprint axis is ~5.1 cm, so this presses ~2.3 cm into the body — firm
    # enough to hold through transit (a looser >3.5 cm gap slips; see sweep)
    # without the over-squeeze that ejects round food.
    BREAD_CLOSE_GAP_M = 0.028

    LIFT_HEIGHT = 0.18         # m above grasp before transit
    APPROACH_HEIGHT = 0.18     # m above basket rim before lowering
    # Release just above the rim, not 6 cm up: the model-2 loaf is light and a
    # high release lets it bounce back out over the rim (verified
    # _bread_place_sweep.py: drop_h +0.06 -> 12.8 cm out; +0.02 -> 2.5 cm in).
    # +0.02 still clears the first loaf when the second is stacked on top.
    DROP_HEIGHT = 0.02         # m above basket rim at release
    PRE_GRASP_SETTLE_STEPS = 30
    PRE_CLOSE_SETTLE_STEPS = 30
    CLOSE_SETTLE_STEPS = 60
    RELEASE_SETTLE_STEPS = 40
    POST_PLACE_SETTLE_STEPS = 60
    INITIAL_SETTLE_STEPS = 60
    BREAD_MIN_GRIPPER_GAP = 0.005
    FRANKA_GRIPPER_MAX_WIDTH = 0.08
    GRIPPER_VALUE_RANGE = (0.05, 0.95)
    FINGER_KP = 9000.0
    FINGER_KV = 250.0

    # Two breads, both must end up inside the basket footprint for
    # success.  0.10 m allows the second bread to land on top of the
    # first (basket interior is ~19 × 14 cm) without the test failing.
    SUCCESS_DIST_XY_M = 0.10
    SUCCESS_TABLE_Z_MARGIN = 0.02

    recording_camera_pos = [1.1, 0.0, 1.95]
    recording_camera_lookat = [0.0, 0.0, 0.85]

    def __init__(self, config: dict = None):
        cfg = dict(config or {})
        # Tell the grasp loader which robot YAML overlay to merge —
        # 075_bread keeps its manual top-down grasps in
        # `grasp_poses_franka.yml`, not the generic `grasp_poses.yml`.
        cfg.setdefault("robot_type", "franka")
        super().__init__(cfg)
        self.resolve_target_object()   # registry/override hook (size-1 set)
        self._num_breads = self.difficulty_object_count()

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

    def _sample_positions(self, rng):
        """Return basket plus the configured bread xy tuples —
        rejection-sampled within `_REGION_X` × `_REGION_Y` with min
        pairwise separation.  Subclasses (e.g. interrupt variant)
        override to constrain individual entries (e.g. force bread B
        into an avatar-reachable subregion)."""
        return _sample_n_xy(1 + int(self._num_breads), rng)

    def load_actors(self):
        # Spawn bread + basket with their authored local +Y mapped to world +Z.
        upright_q = self.SPAWN_QUAT
        table_top = self.TABLE_TOP_Z

        rng = np.random.RandomState(np.random.randint(0, 2 ** 31 - 1))
        positions = self._sample_positions(rng)
        (basket_x, basket_y) = positions[0]
        bread_xys = positions[1:]

        # ---- Basket (stationary receiver) ----
        # Split visual and collision so the original basket surface is
        # rendered while the invisible CoACD body keeps the interior open.
        self._basket_scale = _read_scale(_BASKET_ID, 0)
        basket_extents = np.array(
            _read_model_data(_BASKET_ID, 0)["extents"], dtype=np.float64,
        )
        # Rough pre-build estimates only (true rim height + interior are read
        # from the live world AABB post-settle via _refresh_basket_geom()).
        # Under the upright spawn the mesh-y axis is the shallow basket depth.
        basket_w = float(basket_extents[0]) * self._basket_scale
        basket_h = float(basket_extents[1]) * self._basket_scale
        basket_d = float(basket_extents[2]) * self._basket_scale
        self.basket_pose = Pose(
            [
                basket_x,
                basket_y,
                table_top + self.BASKET_STATIC_Z_CLEARANCE,
            ],
            upright_q,
        )
        basket_glb = ASSETS_PATH / "objects" / _BASKET_ID / "visual" / "base0.glb"
        self.basket_visual_entity = load_mesh(
            self.scene, basket_glb, self.basket_pose,
            scale=(self._basket_scale,) * 3,
            convex=False,
            is_static=True,
            collision=False,
            visual=True,
        )
        basket_entity = load_mesh(
            self.scene, basket_glb, self.basket_pose,
            scale=(self._basket_scale,) * 3,
            convex=True,
            is_static=True,
            visual=False,
            friction=self.BASKET_FRICTION,
            decompose_object_error_threshold=self.BASKET_COACD_ERROR_THRESHOLD,
        )
        self.basket = Actor(basket_entity, {}, _BASKET_ID)
        # Pre-build fallbacks; the true rim z + interior half-extent are read
        # from the live world AABB once the scene is built + settled
        # (_refresh_basket_geom, called at the start of play_once / avatar
        # assist).  All downstream uses query the basket's live position too,
        # so the xy stays current.
        self._basket_rim_z = table_top + basket_h
        self._basket_xy_half = np.array([basket_w / 2, basket_d / 2])
        self._basket_geom_measured = False

        # ---- Bread (graspable) — small ~5.8 × 5.1 × 3.75 cm roll (upright).
        self._bread_scale = _read_scale(_BREAD_ID, _BREAD_MODEL_ID)
        bread_extents = np.array(
            _read_model_data(_BREAD_ID, _BREAD_MODEL_ID)["extents"], dtype=np.float64,
        )
        # Pre-build grip-width estimate (the close is a fixed gap, so this is
        # only informational); the two smallest mesh axes are the horizontal
        # footprint when the roll lies flat.
        sorted_ext = np.sort(bread_extents)[:2] * self._bread_scale
        self._bread_grip_width_m = float(sorted_ext.min())
        self._bread_center_mesh = _read_bbox_center(_BREAD_ID, _BREAD_MODEL_ID)

        self.breads = []
        self.bread_poses = []
        for (bx, by) in bread_xys:
            pose = Pose(
                [
                    bx,
                    by,
                    # Spawn a few cm above the table and let it settle flat —
                    # orientation-agnostic (no reliance on a mesh-axis height).
                    table_top + self.BREAD_SPAWN_DROP,
                ],
                upright_q,
            )
            actor = load_object(
                self.scene, pose, _BREAD_ID, model_id=_BREAD_MODEL_ID,
                convex=True, is_static=False,
                friction=self.BREAD_FRICTION,
            )
            self.bread_poses.append(pose)
            self.breads.append(actor)

        # Custom check_success — TargetSpec only handles a single
        # object.  Set a stub so BaseTask's evaluate() prints something
        # useful for the first bread; check_success below ignores it.
        from ..base_task import TargetSpec
        self.target = TargetSpec(
            object=self.breads[0].entity,
            position=lambda: to_numpy(self.basket.entity.get_pos()).ravel()[:3],
            label="basket",
        )

    # ------------------------------------------------------------------
    # Live basket geometry
    # ------------------------------------------------------------------

    def _refresh_basket_geom(self):
        """Measure the static basket's true rim height + interior half-extent
        from its live world AABB.  Orientation-robust — replaces the old
        hardcoded mesh-extents axis that was only correct for the (buggy)
        tipped spawn.  Cheap + idempotent (basket is static)."""
        if getattr(self, "basket", None) is None:
            return
        try:
            aabb = to_numpy(self.basket.entity.get_AABB()).reshape(2, 3)
        except Exception:
            return
        self._basket_rim_z = float(aabb[1, 2])
        self._basket_xy_half = (aabb[1, :2] - aabb[0, :2]) / 2.0
        self._basket_geom_measured = True

    # ------------------------------------------------------------------
    # Pick + place primitive
    # ------------------------------------------------------------------

    def _pick_and_place_bread(self, bread, arm_tag: str, label: str) -> bool:
        """Single-bread pick → lift → over basket → drop → retreat.
        Returns True iff every motion stage planned/executed; failure
        of any stage aborts and returns False (the bread is left
        wherever it currently is)."""
        arm = self.robot.get_arm(arm_tag)
        tcp_offset = arm.tcp_offset
        if not getattr(self, "_basket_geom_measured", False):
            self._refresh_basket_geom()

        # Re-read live bread pose; settle a bit so the bread is at rest
        # before the grasp planner samples it.
        for _ in range(self.PRE_GRASP_SETTLE_STEPS):
            self.step_sim()
        bread_origin_p = to_numpy(bread.entity.get_pos()).ravel()[:3]
        bread_q = to_numpy(bread.entity.get_quat()).ravel()[:4]
        R_bread = t3d.quaternions.quat2mat(bread_q)
        center_offset_world = R_bread @ (
            self._bread_center_mesh * self._bread_scale
        )
        # A flat loaf can settle with an arbitrary 180-degree flip. Build a
        # flip-invariant grasp pose: keep the bread's footprint yaw about
        # world Z, then compose UPRIGHT_QUAT (mesh +Y -> world +Z). This
        # guarantees a proper top-down approach (approach_z=[0,0,-1]) regardless
        # of the settle flip.  All downstream grasp.to_world() math reuses this
        # logical pose so the lift/place targeting stays consistent.
        mesh_x_world = R_bread @ np.array([1.0, 0.0, 0.0])
        footprint_yaw = float(np.arctan2(mesh_x_world[1], mesh_x_world[0]))
        yaw_q = np.asarray(t3d.quaternions.mat2quat(
            t3d.euler.euler2mat(0, 0, footprint_yaw, "sxyz")), dtype=np.float64)
        logical_q = np.asarray(
            t3d.quaternions.qmult(yaw_q, self.UPRIGHT_QUAT), dtype=np.float64)
        grasp_center = bread_origin_p + center_offset_world
        grasp_center[2] -= self.BREAD_GRASP_Z_DROP  # straddle the loaf body
        live_bread = Pose(grasp_center, logical_q)

        self.open_gripper(arm_tag)

        manual_grasps = [
            g for g in load_grasp_poses(
                _BREAD_ID, model_id=_BREAD_MODEL_ID, robot_type="franka",
            )
            if g.source == "manual"
        ]
        if not manual_grasps:
            return False

        result = None
        grasp = None
        for g in manual_grasps:
            attempt = self.try_grasp_by_name(
                object_name=_BREAD_ID,
                object_pose=live_bread,
                arm_tag=arm_tag,
                grasp_name=g.name,
                model_id=_BREAD_MODEL_ID,
                object_scale=self._bread_scale,
                execute=True,
            )
            if attempt is not None:
                _, _, grasp = attempt
                result = attempt
                break
        if result is None:
            return False

        grasp_tcp = grasp.to_world(live_bread, self._bread_scale)

        # Firm face-on close: drive to a fixed finger gap that presses ~2 cm
        # into the loaf body.  A looser close (gap > 3.5 cm) lets the flat loaf
        # slip out on lift; this gap holds (see _bread_m2_compress_sweep.py).
        for _ in range(self.PRE_CLOSE_SETTLE_STEPS):
            self.step_sim()
        target_gap = max(self.BREAD_MIN_GRIPPER_GAP, self.BREAD_CLOSE_GAP_M)
        gripper_v = float(np.clip(
            target_gap / self.FRANKA_GRIPPER_MAX_WIDTH,
            *self.GRIPPER_VALUE_RANGE,
        ))
        self.set_gripper(gripper_v, arm_tag)
        for _ in range(self.CLOSE_SETTLE_STEPS):
            self.step_sim()

        # PD/force boost on the gripper fingers (per oil_bottle_recovery
        # pattern).  Already applied if a previous bread ran the same
        # boost — the per-DOF set is idempotent.
        update_dofs_kp_kv_compat(
            arm.entity,
            arm._finger_dof_indices,
            kp_value=self.FINGER_KP,
            kv_value=self.FINGER_KV,
        )

        # Lift along world +z, preserving TCP orientation.
        lift_tcp = Pose(
            grasp_tcp.p + np.array([0.0, 0.0, self.LIFT_HEIGHT]),
            grasp_tcp.q,
        )
        lift_link = tcp_to_link_pose(lift_tcp, tcp_offset)
        if self.move_and_execute(lift_link.to_pose7(), arm_tag) is None:
            return False

        # Move above the basket with a top-down TCP.
        basket_pos = to_numpy(self.basket.entity.get_pos()).ravel()[:3]
        R_tcp_down = np.array(
            [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]],
            dtype=np.float64,
        )
        q_tcp_down = t3d.quaternions.mat2quat(R_tcp_down)

        above_p = np.array([
            basket_pos[0], basket_pos[1],
            self._basket_rim_z + self.APPROACH_HEIGHT,
        ])
        above_link = tcp_to_link_pose(Pose(above_p, q_tcp_down), tcp_offset)
        if self.move_and_execute(above_link.to_pose7(), arm_tag) is None:
            return False

        # Lower to drop height (released just above the rim — see DROP_HEIGHT).
        drop_p = np.array([
            basket_pos[0], basket_pos[1],
            self._basket_rim_z + self.DROP_HEIGHT,
        ])
        drop_link = tcp_to_link_pose(Pose(drop_p, q_tcp_down), tcp_offset)
        if self.move_and_execute(drop_link.to_pose7(), arm_tag) is None:
            return False

        # Release.
        self.open_gripper(arm_tag)
        for _ in range(self.RELEASE_SETTLE_STEPS):
            self.step_sim()

        # Retreat upward so the gripper is clear of the basket — leave
        # the EE high so the next pick's curr→pre RRT starts from a
        # safe pose.
        retreat_link = tcp_to_link_pose(Pose(above_p, q_tcp_down), tcp_offset)
        self.move_and_execute(retreat_link.to_pose7(), arm_tag)

        for _ in range(self.POST_PLACE_SETTLE_STEPS):
            self.step_sim()
        return True

    # ------------------------------------------------------------------
    # Main task sequence
    # ------------------------------------------------------------------

    def play_once(self) -> bool:
        # Pre-task object settle; excluded from recordings.
        with self.suppress_recording():
            for _ in range(self.INITIAL_SETTLE_STEPS):
                self.step_sim()

        if not self.breads or self.basket is None:
            return False

        self._refresh_basket_geom()
        arm_tag = "right"

        # Sort breads so the closer-to-basket one goes first — that
        # leaves more room in the basket for the second one and avoids
        # the second drop landing exactly on top of the first.
        basket_xy = to_numpy(self.basket.entity.get_pos()).ravel()[:2]
        order = sorted(
            range(len(self.breads)),
            key=lambda i: float(np.linalg.norm(
                to_numpy(self.breads[i].entity.get_pos()).ravel()[:2]
                - basket_xy,
            )),
        )

        any_success = False
        for idx in order:
            label = chr(ord("A") + idx)
            ok = self._pick_and_place_bread(self.breads[idx], arm_tag, label)
            any_success = any_success or ok
            # Even if one bread failed, try the other — the second
            # might still place even if the first slipped during grasp.
        return any_success

    # ------------------------------------------------------------------
    # Success
    # ------------------------------------------------------------------

    def check_success(self) -> bool:
        """Both breads must end up within `SUCCESS_DIST_XY_M` (xy) of
        the basket centre and above the table top."""
        if not self.plan_success:
            return False
        basket_xy = to_numpy(self.basket.entity.get_pos()).ravel()[:2]
        all_in = True
        for i, b in enumerate(self.breads):
            p = to_numpy(b.entity.get_pos()).ravel()[:3]
            dxy = float(np.linalg.norm(p[:2] - basket_xy))
            above_table = p[2] > self.TABLE_TOP_Z - self.SUCCESS_TABLE_Z_MARGIN
            ok = dxy <= self.SUCCESS_DIST_XY_M and above_table
            if not ok:
                all_in = False
        return all_in
