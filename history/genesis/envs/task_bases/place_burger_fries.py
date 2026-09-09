"""Place burger and fries.

Single-arm Franka adaptation of RoboTwin's bimanual ``place_burger_fries``
(https://robotwin-platform.github.io/doc/tasks/place_burger_fries.html).
The robot picks a hamburger and a french-fries carton from the right side
of the table and drops them onto a tray on the left side, sequentially with
a single arm.  No avatar.

Randomization (per-reset, seeded by ``np.random.seed(seed)``):
- One burger + one fries carton (default), or two of each in hard mode.
- Object positions drawn from per-type spawn rectangles with rejection
  sampling so spawned items never start within 10 cm of each other.
- Tray xy drawn from a small range (~15 cm wide) on the left, kept inside
  the Franka's dexterous workspace.
- Tray colour drawn from a uniform RGB box.  Genesis's ``Default`` surface
  override is applied even though the GLB ships its own material — visual
  fidelity depends on whether the renderer respects the override (it does
  for the rasterizer; raytracer behaviour is untested).

Drop slots (positions on the tray) scale with item count: 1→centre,
2→left/right, 3→top-pair + bottom-centre, 4→2×2 grid.

Picking uses the lower-level helpers from ``TopDownPickPlaceMixin``
(``_top_down_tcp`` / ``_move_seeded`` / ``_move_screw`` / ``_move_cartesian``)
with a custom close + lift step so we can pin a per-object close target,
verify post-lift that the object actually came with the gripper, and use
slow Cartesian interpolation for the long horizontal transit (plan_screw
oscillates the gripper PD enough to slip the slim 3.3 cm fries box).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import transforms3d as t3d

from ..base_task import BaseTask
from ..genesis_compat import update_dofs_force_range_compat, update_dofs_kp_kv_compat
from ..manipulation import TopDownPickPlaceMixin
from ..utils import (
    Actor,
    Pose,
    load_object,
    to_numpy,
)
from ..grasp import tcp_to_link_pose


_HAMBURG_ID = "006_hamburg"
_FRIES_ID = "005_french-fries"
_TRAY_ID = "008_tray"


@dataclass
class FoodSpec:
    """Per-food parameters: asset id, model variant, spawn pose, grasp."""
    name: str
    asset_id: str
    model_id: int
    spawn_quat: np.ndarray
    spawn_z_offset: float       # z above TABLE_TOP_Z at spawn
    friction: float
    close_value: float          # set_gripper target (0=closed, 1=open)
    below_center: float         # grip pose z = obj_center.z - below_center
    tcp_yaw_deg: float          # rotate top-down gripper about approach axis
    spawn_xy_range: tuple       # ((xlo, xhi), (ylo, yhi))
    # Per-spec horizontal transit override (0 = use the class default).  A
    # big flat box needs many small Cartesian steps so it cannot pitch out
    # of the grip; a compact item is fine with the coarser class default and
    # creeps if held for the extra sim time, so the two are tuned separately.
    transit_steps: int = 0
    transit_sim: int = 0
    # When True, rotate the top-down grip so the jaws close across the
    # object's MEASURED short horizontal (mesh-X) axis at grasp time, instead
    # of a fixed world-aligned tcp_yaw_deg.  A long/flat object that settles a
    # few degrees off its spawn yaw is otherwise gripped on a corner and
    # dangles/pitches out; aligning the jaws to its real axis grips it square.
    align_to_object_yaw: bool = False


# upright = mesh-Y → world-Z (most assets are y-up)
_Q_UPRIGHT = np.array([0.70710678, 0.70710678, 0.0, 0.0], dtype=np.float64)
# legacy burger orientation = R_z(90°) * R_x(90°) — also mesh-Y → world-Z,
# differs only in xy yaw.  Keep for backwards compatibility w/ v17 sweeps.
_Q_BURGER = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float64)


# 006_hamburg model_0 is a wide flat 6.6 cm disc — a parallel jaw clamps its
# full diameter with near-zero compression and any wrench ejects it on lift
# (Genesis-1.0.0 non-caging-grasp wall; round-1 fix1/fix2/fix3 all hit this).
# model_5 is the ELONGATED patty variant (4.7 x 2.4 x 7.6 cm): under _Q_BURGER
# its 4.7 cm narrow axis lands on world-Y, exactly the default top-down finger
# closing axis, so the jaws clamp the narrow waist and the long axis gives a
# long contact line — the flat-object playbook from fix1's bread loaf.
# close_value 0.50 (NOT tighter): on this thin mesh a firm squeeze crush-EJECTS
# the patty before it lifts (sweep: 0.35/0.40 fail, 0.45/0.50 hold; 0.50 best
# lift_dz +0.048). 0.50 ≈ 4.0 cm finger gap = ~0.7 cm gentle compression on the
# 4.7 cm waist. friction 5.0 (capped) carries the vertical load through transit.
_BURGER_SPEC = FoodSpec(
    name="burger", asset_id=_HAMBURG_ID, model_id=5,
    spawn_quat=_Q_BURGER,
    spawn_z_offset=0.05, friction=5.0,
    close_value=0.50, below_center=0.005, tcp_yaw_deg=0.0,
    spawn_xy_range=((+0.05, +0.20), (-0.08, +0.05)),
)

# 005_french-fries carton.  Tuned via a grasp-parameter sweep; the surviving
# recipe and the dead ends behind each knob:
#   * model_0 (7.06 x 9.62 x 3.27 cm), NOT model_1: model_1 is a 5.6 cm-wide
#     HOLLOW thin-walled carton (CoACD -> 11 pieces) whose walls flex and
#     slip; model_0's wider, more solid body gives the jaws a form-closure
#     grip.  (model_3/2 are >8 cm wide -> the fingers graze them, dz=0.)
#   * Laid flat with mesh +Y mapped to world +Z, gripped top-down across the
#     7.06 cm short axis.
#     Standing the carton never lifts (the jaws slide off thin walls /
#     tapered top, dz=0).
#   * align_to_object_yaw=True is the single biggest win: a fixed
#     world-aligned grip catches the settled (slightly-yawed) long box on a
#     corner and it dangles out (0/N); squaring the jaws to the box's
#     measured short axis took it to ~2/3 (sweep4/5).
#   * spawn_z_offset 0.005 (a low drop, NOT 0.02): dropping the box 2 cm lets
#     it bounce to a random yaw/pitch; settling it flat from ~rest cut the
#     per-seed variance and lifted success 2/5 -> 3/5 (sweep6).
#   * close 0.25 + friction 5.0 (capped); DEFAULT transit (smoothing the
#     Cartesian transit to 80-160 steps consistently made it WORSE, 0/N).
# Net: a marginal-but-real ~3/5 expert grasp on a genuinely hard hollow
# carton (the burger beside it is ~3/3, so the task is ~60% end-to-end).
_FRIES_SPEC = FoodSpec(
    name="fries", asset_id=_FRIES_ID, model_id=0,
    spawn_quat=_Q_UPRIGHT,
    spawn_z_offset=0.005, friction=5.0,
    close_value=0.25, below_center=0.005, tcp_yaw_deg=90.0,
    spawn_xy_range=((-0.02, +0.15), (-0.22, -0.08)),
    align_to_object_yaw=True,
)


class PlaceBurgerFries(TopDownPickPlaceMixin, BaseTask):
    """Robot picks a hamburger and a french-fries carton (one of each in
    the default layout, two of each in hard mode) and places them all on a
    cafeteria tray.
    """

    INSTRUCTION = "place the hamburger and the french fries on the tray"

    use_avatar = False

    # Keep the original avatar-side edge at y=+0.35 and extend only toward the
    # Franka at y=-0.65. The robot-side edge at y=-0.80 places the whole base
    # 0.15 m inboard instead of leaving it suspended beyond the tabletop.
    TABLE_HALF_SIZE = (0.60, 0.575)
    TABLE_CENTER_XY = (0.0, -0.225)

    # Shared by neutral / assist / interrupt variants.
    # VLA primary view: near top-down over the full default 1.20 x 0.70 m table.
    static_camera_list = [{
        "name": "head_camera",
        "position": [0.0, 0.005, 1.90],
        "forward": [0.0, -0.005, -1.05],
    }]

    # The tray is authored Y-up; map its thin/local-Y axis onto world Z so the
    # 30.7 x 18.7 cm face lies flat on the table.
    _Q_TRAY = _Q_UPRIGHT
    # Kept as a class attribute for the assist subclass which needs to set
    # the avatar-tossed burger's quaternion.
    _Q_BURGER = _Q_BURGER

    # ---- Layout ----------------------------------------------------------
    # Tray on the left.  Each food has its own spawn xy range (set in its
    # FoodSpec at module scope).
    TRAY_X_RANGE = (-0.25, -0.10)
    TRAY_Y_RANGE = (+0.00, +0.10)

    # Min separation between any two spawned food items (world XY).
    SPAWN_MIN_SEP = 0.085
    # Small margin around the tray footprint when sampling initial food poses.
    # Without this, the right edge of a randomized tray can overlap the food
    # spawn rectangles and some items start inside the tray.
    SPAWN_TRAY_CLEARANCE = 0.005

    # Tray colour range (uniform RGB) — kept for log only; GLB material
    # overrides ``surface=Default(color=)`` so this is informational.
    TRAY_COLOR_LO = (0.20, 0.20, 0.20)
    TRAY_COLOR_HI = (0.95, 0.95, 0.95)

    # ---- Tray geometry (008_tray GLB) ------------------------------------
    # World AABB: 30.7 × 18.7 × 3.1 cm.  Success tolerance is half-extent
    # + 1 cm overhang in each axis (rectangular check).
    TRAY_RIM_Z = 0.015         # ≈ half tray thickness
    TRAY_HALF_X = 0.165
    TRAY_HALF_Y = 0.105

    # ---- Pick parameters --------------------------------------------------
    TRANSPORT_Z_ABOVE_TABLE = 0.22
    DROP_HOVER_Z_ABOVE_TRAY = 0.06
    LIFT_HEIGHT_ABOVE_GRASP = 0.18

    # Single-shot pick: ONE attempt per object, no re-pick on slip.  VLA data
    # collection wants clean 1-shot pick-and-place demonstrations — a re-pick
    # (the arm going back to grab a slipped object) pollutes the trajectory, so
    # if an object isn't placed on the first try the episode simply fails.
    SPAWN_SAMPLE_ATTEMPTS = 1000
    SPAWN_FALLBACK_GRID_COUNT = 31
    TRAY_FRICTION = 2.0
    FINGER_KP = 9000.0
    FINGER_KV = 250.0
    FINGER_FORCE_LIMIT = 80.0
    TABLE_FALL_Z_MARGIN = 0.05
    GRASP_TABLE_Z_MARGIN = 0.003
    PRE_GRASP_HEIGHT = 0.15
    GRASP_SETTLE_STEPS = 100
    GRASP_ALIGNMENT_XY_TOL = 0.015
    GRASP_ALIGNMENT_SETTLE_STEPS = 80
    CLOSE_SETTLE_STEPS = 200
    POST_LIFT_SETTLE_STEPS = 60
    LIFT_DZ_MIN = 0.05
    LIFT_FAIL_RELEASE_SETTLE_STEPS = 50
    # Default horizontal transit (compact items, e.g. the burger).  The big
    # flat fries carton overrides this per-spec with many small steps — see
    # FoodSpec.transit_steps / _FRIES_SPEC.
    TRANSIT_CARTESIAN_STEPS = 40
    TRANSIT_SIM_PER_STEP = 20
    POST_TRANSIT_SETTLE_STEPS = 50
    TRANSIT_SLIP_XY_TOL = 0.10
    DROP_SETTLE_STEPS = 40
    RELEASE_SETTLE_STEPS = 200
    FINAL_SETTLE_STEPS = 150
    SUCCESS_TABLE_Z_MARGIN = 0.02

    def __init__(self, config: dict = None):
        cfg = dict(config or {})
        cfg.setdefault("robot_type", "franka")
        cfg.setdefault("robot_single_arm", True)
        super().__init__(cfg)

        self.tray_actor = None
        # Unified list of (Actor, FoodSpec) for all spawned food items.
        # Burger is always index 0; optionals follow in declared order.
        self.food_actors: list[tuple[Actor, FoodSpec]] = []
        self.food_spawn_xys: list[tuple[float, float]] = []
        self._tray_xy: tuple[float, float] = (-0.20, +0.05)
        self._tray_color: tuple = (0.5, 0.5, 0.5, 1.0)

    def _create_table(self, table_height=0.74):
        """Build the extended footprint identically for every table variant."""
        return self._create_rectangular_table(
            table_height=table_height,
            half_size=self.TABLE_HALF_SIZE,
            center_xy=self.TABLE_CENTER_XY,
        )

    # ------------------------------------------------------------------
    # Drop slot layouts on the tray (offsets relative to tray centre xy).
    # ------------------------------------------------------------------
    @staticmethod
    def _drop_slots(n: int) -> list[tuple[float, float]]:
        if n == 1:
            return [(0.0, 0.0)]
        if n == 2:
            return [(-0.06, 0.0), (+0.06, 0.0)]
        if n == 3:
            return [(-0.07, +0.03), (+0.07, +0.03), (0.0, -0.04)]
        if n == 4:
            return [(-0.07, +0.03), (+0.07, +0.03),
                    (-0.07, -0.03), (+0.07, -0.03)]
        raise ValueError(f"unsupported drop slot count: {n}")

    # ------------------------------------------------------------------
    # Random non-overlapping spawn position.
    # ------------------------------------------------------------------
    def _xy_overlaps_tray(self, x: float, y: float) -> bool:
        tray_x, tray_y = self._tray_xy
        clearance = self.SPAWN_TRAY_CLEARANCE
        return (
            abs(float(x) - tray_x) <= self.TRAY_HALF_X + clearance
            and abs(float(y) - tray_y) <= self.TRAY_HALF_Y + clearance
        )

    def _sample_spawn_xy(self, xy_range, occupied, max_tries: int = None
                         ) -> tuple[float, float]:
        max_tries = self.SPAWN_SAMPLE_ATTEMPTS if max_tries is None else max_tries
        (xlo, xhi), (ylo, yhi) = xy_range
        for _ in range(max_tries):
            x = float(np.random.uniform(xlo, xhi))
            y = float(np.random.uniform(ylo, yhi))
            if self._xy_overlaps_tray(x, y):
                continue
            if all(float(np.hypot(x - ox, y - oy)) >= self.SPAWN_MIN_SEP
                   for ox, oy in occupied):
                return x, y
        best = None
        best_dist = -np.inf
        for x in np.linspace(xlo, xhi, self.SPAWN_FALLBACK_GRID_COUNT):
            for y in np.linspace(ylo, yhi, self.SPAWN_FALLBACK_GRID_COUNT):
                x = float(x)
                y = float(y)
                if self._xy_overlaps_tray(x, y):
                    continue
                min_dist = min(
                    (float(np.hypot(x - ox, y - oy)) for ox, oy in occupied),
                    default=np.inf,
                )
                if min_dist >= self.SPAWN_MIN_SEP and min_dist > best_dist:
                    best = (x, y)
                    best_dist = min_dist
        if best is not None:
            return best
        raise RuntimeError(
            f"could not find a non-overlapping spawn outside tray in {xy_range} "
            f"after {max_tries} tries (occupied={occupied}, tray_xy={self._tray_xy})"
        )

    # The plate is the ``008_tray`` GLB (rectangular cafeteria tray w/ rim,
    # ~30.7 × 18.7 × 3.1 cm).  Per-task colour is intentionally NOT applied
    # here: the GLB ships with baked PBR materials that the renderer
    # prefers over ``surface=Default(color=…)``, so ``self._tray_color`` is
    # kept around for the launcher log only.

    def _add_tray_with_color(self, pose: Pose, color):
        """Spawn the 008_tray GLB as a DYNAMIC, physics-simulated body.

        A static tray sat wherever it was placed — and because 008_tray's mesh
        origin is ~1.3 cm below its geometric centre, placing the origin at
        ``table_top + TRAY_RIM_Z`` left the tray hovering ~1.2 cm above the
        table.  As a dynamic body it just falls during the reset-settle and
        rests flat on the table under gravity (real contact), so it neither
        floats nor needs the origin offset hand-computed.  ``color`` is logged
        but unused (the GLB's baked material wins over a surface override)."""
        return load_object(
            self.scene, pose, _TRAY_ID,
            model_id=0, convex=True, is_static=False, friction=self.TRAY_FRICTION,
        )

    # ------------------------------------------------------------------
    # Scene
    # ------------------------------------------------------------------
    def _food_specs_for_episode(self) -> list[FoodSpec]:
        """One hamburger + one fries carton in the default (simplified)
        layout; a second burger + fries pair in hard mode."""
        if self.hard_mode_enabled():
            return [_BURGER_SPEC, _FRIES_SPEC, _BURGER_SPEC, _FRIES_SPEC]
        return [_BURGER_SPEC, _FRIES_SPEC]

    def load_actors(self):
        table_top = self.TABLE_TOP_Z

        # ---- Random tray position ----
        tray_x = float(np.random.uniform(*self.TRAY_X_RANGE))
        tray_y = float(np.random.uniform(*self.TRAY_Y_RANGE))
        self._tray_xy = (tray_x, tray_y)

        rgb = [float(np.random.uniform(lo, hi))
               for lo, hi in zip(self.TRAY_COLOR_LO, self.TRAY_COLOR_HI)]
        self._tray_color = tuple(rgb + [1.0])

        tray_pose = Pose(
            [tray_x, tray_y, table_top + self.TRAY_RIM_Z],
            self._Q_TRAY,
        )
        self.tray_actor = self._add_tray_with_color(tray_pose, self._tray_color)

        # ---- Pick which foods to spawn this episode ----
        specs: list[FoodSpec] = self._food_specs_for_episode()

        # ---- Spawn each food, rejection-sampling its xy against earlier ones ----
        self.food_actors = []
        self.food_spawn_xys = []
        occupied: list[tuple[float, float]] = []
        for spec in specs:
            x, y = self._sample_spawn_xy(spec.spawn_xy_range, occupied)
            actor = load_object(
                self.scene,
                Pose([x, y, table_top + spec.spawn_z_offset], spec.spawn_quat),
                spec.asset_id, model_id=spec.model_id,
                convex=True, is_static=False, friction=spec.friction,
            )
            self.food_actors.append((actor, spec))
            self.food_spawn_xys.append((x, y))
            occupied.append((x, y))

    # ------------------------------------------------------------------
    # Finger PD + force-range boost.
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # One-shot pick wrapper.
    # ------------------------------------------------------------------
    def _rotated_top_down_tcp(self, pos, yaw_deg: float) -> Pose:
        """Top-down gripper rotated about the approach (TCP-Z) axis by
        ``yaw_deg``.  Used to grip narrow boxy objects across their short
        dimension (yaw=90°): default top-down closes fingers along world Y;
        +90° yaw closes fingers along world X.
        """
        if abs(yaw_deg) < 1e-6:
            return self._top_down_tcp(pos)
        yaw = np.deg2rad(yaw_deg)
        cz, sz = np.cos(yaw), np.sin(yaw)
        Rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
        R = self._R_DOWN @ Rz
        q = t3d.quaternions.mat2quat(R)
        return Pose(np.asarray(pos, dtype=float), q)

    def _pick_and_drop(self, actor, asset_name: str, arm_tag: str,
                       close_value: float, below_center: float,
                       drop_xy: tuple[float, float],
                       tcp_yaw_deg: float = 0.0,
                       transit_steps: int = 0, transit_sim: int = 0,
                       align_to_object_yaw: bool = False) -> bool:
        return self._pick_once(
            actor, asset_name, arm_tag,
            close_value=close_value, below_center=below_center,
            drop_xy=drop_xy, tcp_yaw_deg=tcp_yaw_deg,
            transit_steps=transit_steps, transit_sim=transit_sim,
            align_to_object_yaw=align_to_object_yaw,
        )

    def _get_object_world_center(self, actor, object_name: str, model_id: int = 0):
        """For the fries carton, grasp at the LIVE world-AABB centre instead of
        the model_data centre. The simulator AABB centre is the authoritative
        geometric centre after settling, so the jaws close symmetrically even
        if this asset's metadata is stale. Scoped to the fries only; the
        burger's grasp tuning is unchanged.
        """
        if object_name == _FRIES_ID:
            try:
                aabb = to_numpy(actor.entity.get_AABB())
                return np.asarray(0.5 * (aabb[0] + aabb[1]), dtype=float)
            except Exception:
                pass
        return super()._get_object_world_center(actor, object_name, model_id)

    def _object_grip_yaw(self, actor) -> float:
        """tcp_yaw (deg) that closes the top-down jaws across the object's
        measured short horizontal (mesh-X) axis.  tcp_yaw 0 closes along
        world-Y, so the offset is (world-yaw of mesh-X) - 90."""
        q = np.asarray(to_numpy(actor.get_pose().q), dtype=float).ravel()
        R = t3d.quaternions.quat2mat(q)
        x_axis = R @ np.array([1.0, 0.0, 0.0])
        theta = float(np.degrees(np.arctan2(x_axis[1], x_axis[0])))
        return theta - 90.0

    def _pick_once(self, actor, asset_name: str, arm_tag: str,
                   close_value: float, below_center: float,
                   drop_xy: tuple[float, float],
                   tcp_yaw_deg: float = 0.0,
                   transit_steps: int = 0, transit_sim: int = 0,
                   align_to_object_yaw: bool = False) -> bool:
        from ..robot.franka_robot import to_numpy as _to_numpy
        arm = self.robot.get_arm(arm_tag)
        tcp_offset = arm.tcp_offset

        arm._cached_target = np.asarray(
            _to_numpy(arm.entity.get_qpos()), dtype=float,
        ).ravel().copy()

        transport_z = self.TABLE_TOP_Z + self.TRANSPORT_Z_ABOVE_TABLE
        drop_z = self.TABLE_TOP_Z + self.TRAY_RIM_Z + self.DROP_HOVER_Z_ABOVE_TRAY

        obj_center = self._get_object_world_center(actor, asset_name, 0)
        eff_yaw = self._object_grip_yaw(actor) if align_to_object_yaw else tcp_yaw_deg

        grasp_pos = obj_center.copy()
        grasp_pos[2] -= below_center
        grasp_pos[2] = max(
            grasp_pos[2], self.TABLE_TOP_Z + self.GRASP_TABLE_Z_MARGIN,
        )

        self.open_gripper(arm_tag)

        above_pos = np.array([obj_center[0], obj_center[1], transport_z])
        above_link = tcp_to_link_pose(self._rotated_top_down_tcp(above_pos, eff_yaw), tcp_offset)
        if self._move_seeded(above_link.to_pose7(), arm_tag) is None:
            return False

        pre_pos = obj_center.copy()
        pre_pos[2] += self.PRE_GRASP_HEIGHT
        self._move_screw(pre_pos, arm_tag)

        obj_now = self._get_object_world_center(actor, asset_name, 0)
        grasp_pos[0] = obj_now[0]
        grasp_pos[1] = obj_now[1]
        self._move_screw(grasp_pos, arm_tag)
        for _ in range(self.GRASP_SETTLE_STEPS):
            self.step_sim()

        ee_pos = np.array(arm.get_ee_pose()[:3], dtype=float)
        xy_err = float(np.linalg.norm(ee_pos[:2] - grasp_pos[:2]))
        if xy_err > self.GRASP_ALIGNMENT_XY_TOL:
            grasp_link = tcp_to_link_pose(self._rotated_top_down_tcp(grasp_pos, eff_yaw), tcp_offset)
            self._move_seeded(grasp_link.to_pose7(), arm_tag)
            for _ in range(self.GRASP_ALIGNMENT_SETTLE_STEPS):
                self.step_sim()

        z_before = float(self._get_object_world_center(actor, asset_name, 0)[2])
        self.set_gripper(close_value, arm_tag)
        for _ in range(self.CLOSE_SETTLE_STEPS):
            self.step_sim()

        lift_pos = np.array([grasp_pos[0], grasp_pos[1], transport_z])
        if not self._move_screw(lift_pos, arm_tag):
            return False
        for _ in range(self.POST_LIFT_SETTLE_STEPS):
            self.step_sim()
        last_pos = lift_pos

        z_after = float(self._get_object_world_center(actor, asset_name, 0)[2])
        dz = z_after - z_before
        if dz < self.LIFT_DZ_MIN:
            self.open_gripper(arm_tag)
            for _ in range(self.LIFT_FAIL_RELEASE_SETTLE_STEPS):
                self.step_sim()
            return False

        above_drop = np.array([drop_xy[0], drop_xy[1], transport_z])
        n_steps = transit_steps if transit_steps else self.TRANSIT_CARTESIAN_STEPS
        sim_per_step = transit_sim if transit_sim else self.TRANSIT_SIM_PER_STEP
        self._move_cartesian(
            np.asarray(last_pos, dtype=float),
            above_drop, arm_tag,
            n_steps=n_steps,
            sim_per_step=sim_per_step,
        )
        for _ in range(self.POST_TRANSIT_SETTLE_STEPS):
            self.step_sim()

        ee_xy = np.array(arm.get_ee_pose()[:2], dtype=float)
        obj_xy = self._get_object_world_center(actor, asset_name, 0)[:2]
        post_transit_xy = float(np.linalg.norm(ee_xy - obj_xy))
        if post_transit_xy > self.TRANSIT_SLIP_XY_TOL:
            return False

        drop_pos = np.array([drop_xy[0], drop_xy[1], drop_z])
        self._move_screw(drop_pos, arm_tag)
        for _ in range(self.DROP_SETTLE_STEPS):
            self.step_sim()

        self.open_gripper(arm_tag)
        for _ in range(self.RELEASE_SETTLE_STEPS):
            self.step_sim()
        return True

    # ------------------------------------------------------------------
    # Rollout
    # ------------------------------------------------------------------
    def play_once(self) -> bool:
        arm_tag = "right"
        self._boost_finger_pd(arm_tag)

        n_total = len(self.food_actors)
        slots = self._drop_slots(n_total)
        tray_x, tray_y = self._tray_xy

        for i, (actor, spec) in enumerate(self.food_actors):
            dx, dy = slots[i]
            if not self._pick_and_drop(
                actor, spec.asset_id, arm_tag,
                close_value=spec.close_value,
                below_center=spec.below_center,
                drop_xy=(tray_x + dx, tray_y + dy),
                tcp_yaw_deg=spec.tcp_yaw_deg,
                transit_steps=spec.transit_steps,
                transit_sim=spec.transit_sim,
                align_to_object_yaw=spec.align_to_object_yaw,
            ):
                return False

        for _ in range(self.FINAL_SETTLE_STEPS):
            self.step_sim()
        return True

    # ------------------------------------------------------------------
    # Success: every spawned food item lies within the tray footprint and
    # is at or above the table top (didn't fall on the floor).
    # ------------------------------------------------------------------
    def _live_tray_center_xy(self) -> tuple[float, float]:
        """Live world xy of the (now dynamic) tray's geometric centre.  Falls
        back to the spawn xy if the AABB read fails."""
        if self.tray_actor is not None:
            try:
                aabb = to_numpy(self.tray_actor.entity.get_AABB())
                c = 0.5 * (aabb[0] + aabb[1])
                return float(c[0]), float(c[1])
            except Exception:
                pass
        return self._tray_xy

    def check_success(self) -> bool:
        if not self.plan_success:
            return False

        tray_x, tray_y = self._live_tray_center_xy()
        ok = True
        for actor, _spec in self.food_actors:
            pos = to_numpy(actor.get_pose().p).ravel()[:3]
            dx = abs(float(pos[0]) - tray_x)
            dy = abs(float(pos[1]) - tray_y)
            on_tray_xy = dx <= self.TRAY_HALF_X and dy <= self.TRAY_HALF_Y
            on_table_z = float(pos[2]) >= self.TABLE_TOP_Z - self.SUCCESS_TABLE_Z_MARGIN
            this_ok = on_tray_xy and on_table_z
            ok = ok and this_ok
        return ok
