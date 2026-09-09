"""Place foods in a skillet on a stove.

Single-Franka, physics-only adaptation of RoboTwin's
``place_bread_skillet`` idea.  The robot first picks a dynamic
``106_skillet`` by its handle and places it on the kitchen cooktop, then
places bread, apple, and burger into the skillet.  The only fixed scene
objects are the floor, table/counter, cooktop, and robot base.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import transforms3d as t3d

from ..base_task import BaseTask, TargetSpec
from ..genesis_compat import update_dofs_force_range_compat, update_dofs_kp_kv_compat
from ..grasp import tcp_to_link_pose
from ..manipulation import TopDownPickPlaceMixin
from ..scenes.kitchen import KitchenSceneMixin
from ..utils import Actor, ASSETS_PATH, Pose, load_mesh, load_object, to_numpy


_PAN_ID = "106_skillet"
_BREAD_ID = "075_bread"
_APPLE_ID = "035_apple"
_BURGER_ID = "006_hamburg"

_Q_UPRIGHT = np.array([0.70710678, 0.70710678, 0.0, 0.0], dtype=np.float64)
_Q_BURGER = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float64)
_Q_BURGER_MANUAL = np.array(
    [-0.013478, 0.703591, -0.709927, -0.027948], dtype=np.float64,
)


@dataclass(frozen=True)
class FoodSpec:
    name: str
    asset_id: str
    model_id: int
    spawn_xy: tuple[float, float]
    spawn_quat: np.ndarray
    spawn_z_offset: float
    friction: float
    close_value: float
    below_center: float
    tcp_yaw_deg: float = 0.0
    use_grasp_helper: bool = False
    side_grasp: bool = False


def _read_model_data(name: str, model_id: int = 0) -> dict:
    with open(ASSETS_PATH / "objects" / name / f"model_data{model_id}.json") as f:
        return json.load(f)


def _read_scale(name: str, model_id: int = 0) -> float:
    raw = _read_model_data(name, model_id).get("scale", 1.0)
    return float(raw[0] if isinstance(raw, (list, tuple)) else raw)


def _scaled_center(name: str, model_id: int = 0) -> np.ndarray:
    md = _read_model_data(name, model_id)
    raw_scale = md.get("scale", [1.0, 1.0, 1.0])
    scale = np.asarray(raw_scale if isinstance(raw_scale, list) else [raw_scale] * 3)
    return np.asarray(md.get("center", [0.0, 0.0, 0.0]), dtype=float) * scale


def _scaled_extents(name: str, model_id: int = 0) -> np.ndarray:
    md = _read_model_data(name, model_id)
    ext = np.asarray(md.get("extents", [1.0, 1.0, 1.0]), dtype=float)
    raw_scale = md.get("scale", [1.0, 1.0, 1.0])
    scale = np.asarray(raw_scale if isinstance(raw_scale, list) else [raw_scale] * 3)
    return ext * scale


class PlaceFoodInSkillet(TopDownPickPlaceMixin, KitchenSceneMixin, BaseTask):
    """Robot places a skillet on the cooktop, then puts foods into it."""

    INSTRUCTION = "place the skillet on the stove and put the foods in it"
    use_avatar = False

    # Keep the first version focused on the task objects.
    FORWARD_Y_SHIFT = 0.20
    KITCHEN_LAYOUT = ()
    KITCHEN_BACKDROP = None
    KITCHEN_COOKTOP = {"xy": (-0.12, 0.08), "width": 0.42, "depth": 0.42}
    KITCHEN_ROBOT_KWARGS = {"pos": [0.0, -0.35, 0.765]}

    # VLA/table camera: near top-down, centered slightly toward the avatar side,
    # and high enough to include the full 1.70 x 0.95 m kitchen table.  With
    # fovy=60 and 640x480, z=2.35 gives about 2.44 m x 1.83 m coverage at
    # table height.
    static_camera_list = [{
        "name": "head_camera",
        "position": [0.0, 0.081, 2.35],
        "forward": [0.0, -0.001, -1.585],
    }]
    recording_camera_pos = [0.0, 0.081, 2.35]
    recording_camera_lookat = [0.0, 0.080, 0.765]
    side_camera_pos = [-1.35, -1.45, 1.85]
    side_camera_lookat = [-0.05, 0.02, 0.90]

    VIDEO_STRIDE = 6

    FINGER_KP = 9000.0
    FINGER_KV = 250.0
    FINGER_FORCE_N = 80.0

    PAN_LIFT_M = 0.18
    PAN_TRANSPORT_Z = 1.05
    PAN_RELEASE_ABOVE_COOKTOP = 0.035
    FOOD_TRANSPORT_ABOVE_TABLE = 0.28
    FOOD_DROP_ABOVE_RIM = 0.055

    # Skillet mesh facts: local +Y is rim-up under _Q_UPRIGHT; local +Z is
    # handle-to-pan-body direction.  The functional point in model_data0 is
    # near the pan bowl center, in unscaled mesh units.
    PAN_BOWL_LOCAL_MESH = np.array([0.012, 0.130, 0.509], dtype=float)
    PAN_SUCCESS_XY = 0.09
    FOOD_SUCCESS_XY = 0.115
    PAN_SPAWN_JITTER_XY = 0.015
    FOOD_SPAWN_JITTER_XY = 0.020
    RANDOM_FOOD_POOL = ("burger", "bread", "apple")

    def __init__(self, config: dict = None):
        cfg = dict(config or {})
        cfg.setdefault("robot_type", "franka")
        cfg.setdefault("robot_single_arm", True)
        super().__init__(cfg)
        self.pan_actor = None
        self.food_actors: list[tuple[Actor, FoodSpec]] = []
        self._cooktop_xy = np.array(self.KITCHEN_COOKTOP["xy"], dtype=float)
        self._pan_scale = _read_scale(_PAN_ID, 0)
        self._pan_bowl_local = self.PAN_BOWL_LOCAL_MESH * self._pan_scale
        self._pan_rim_above_origin = 0.272 * self._pan_scale

    # ------------------------------------------------------------------
    # Scene setup
    # ------------------------------------------------------------------

    def _enabled_food_specs(self) -> list[FoodSpec]:
        all_specs = {
            # Bread model 1 is the smaller bun-like variant used by
            # place_burger_fries; it is much easier to pick and fits the pan.
            "bread": FoodSpec(
                "bread", _BREAD_ID, 1, (-0.34, -0.10), _Q_UPRIGHT,
                0.055, 4.0, 0.30, 0.005, 0.0, False, False,
            ),
            "apple": FoodSpec(
                # Model 0 uses the task-specific one-shot side grasp.  The old
                # (0.14, 0.34) spawn was 0.70 m from the Franka base, so keep
                # it in the reachable, cooktop-clear easy workspace used by
                # the burger.  A failed grasp remains a failed episode.
                "apple", _APPLE_ID, 0,
                (0.20, 0.00) if self._pan_starts_on_cooktop() else (0.42, -0.20),
                _Q_UPRIGHT,
                0.045, 4.0, 0.30, 0.005, 0.0, False, True,
            ),
            "burger": FoodSpec(
                # Reachable-annulus spawn.  The old (0.14, 0.34) sat 0.70 m
                # from the Franka base at [0,-0.35] — beyond reach under
                # Genesis 1.0.0, so the top-down pick reached out, failed to
                # lift, and the (simplified-mode) base task always failed.
                # (0.20, 0.00) is 0.40 m from base and clear of the cooktop
                # footprint (x in [-0.33, 0.09]); bread (-0.34,-0.10) is
                # already in reach (0.42 m).
                # model_5 = boxy wrapped burger.  Use the exact known-good
                # place_burger_fries pairing: _Q_BURGER lays local +Y upward
                # while putting the 4.7 cm waist on the default jaw-closing
                # axis, and close 0.50 gives gentle compression.  Tighter
                # 0.35/0.40 closes eject or lose this thin mesh.  This remains
                # a single close/lift attempt; no regrasp is attempted.
                # (0.20, 0.00) is only valid in simplified mode (pan starts on
                # the cooktop).  In hard mode the pan spawns at (0.25, 0.03)
                # and its AABB covers x [0.17, 0.35] — a burger at (0.20, 0.00)
                # spawns INSIDE the pan footprint, leaning on the bowl rim
                # (measured: probe_food_pan.py).  Move it clear while staying
                # in the Franka reach annulus (dist 0.48 m from base [0,-0.35]).
                "burger", _BURGER_ID, 5,
                (0.20, 0.00) if self._pan_starts_on_cooktop() else (0.42, -0.12),
                _Q_BURGER,
                0.055, 5.0, 0.50, 0.005, 0.0, False, False,
            ),
        }

        requested = self.config.get("foods")
        if requested is None:
            if self.simplified_mode_enabled() and bool(
                self.config.get("random_object", False)
            ):
                pool = [
                    str(name).lower()
                    for name in self.config.get("food_pool", self.RANDOM_FOOD_POOL)
                    if str(name).lower() in all_specs
                ]
                if not pool:
                    raise ValueError("place_food_in_skillet food_pool is empty")
                requested = [str(np.random.choice(pool))]
            else:
                requested = (
                    ["burger"]
                    if self.simplified_mode_enabled()
                    else ["burger", "bread"]
                )
        requested = [str(x).lower() for x in requested]
        selected = [all_specs[name] for name in requested if name in all_specs]
        self.selected_food_names = tuple(spec.name for spec in selected)
        print(f"[skillet] selected_foods={list(self.selected_food_names)}")
        return selected

    def _jitter_xy(self, xy: tuple[float, float], amount: float) -> tuple[float, float]:
        if not self.config.get("randomize_initial_positions", True):
            return float(xy[0]), float(xy[1])
        dx, dy = np.random.uniform(-float(amount), float(amount), size=2)
        return float(xy[0] + dx), float(xy[1] + dy)

    def _pan_starts_on_cooktop(self) -> bool:
        """Simplified layouts pre-place the pan; hard/interrupt spawn it on
        the counter for the robot to pick.  Food spawns keyed off this too."""
        return (
            self.simplified_mode_enabled()
            and not bool(getattr(self, "_force_dynamic_pan", False))
        )

    def load_actors(self):
        self.build_kitchen()
        table_top = self.TABLE_TOP_Z

        # In simplified/base mode the pan is part of the workspace target, not a
        # second pick-place object: it starts already on the cooktop so the
        # robot only manipulates the food.  It is a real (dynamic) physics body
        # either way -- it settles flat on the cooktop under gravity during the
        # pre-task settle rather than being a frozen static fixture.
        simplified_pan = self._pan_starts_on_cooktop()
        if simplified_pan:
            # Spawn just above the cooktop centre so it settles flat onto it.
            origin = self._pan_origin_target_on_cooktop()
            pan_pose = Pose(origin + np.array([0.0, 0.0, 0.015]), _Q_UPRIGHT)
        else:
            # Dynamic skillet starts on the right-front counter; the robot picks
            # it up and places it on the cooktop.  CoACD is forced so the pan
            # interior is not treated as one filled convex hull.
            pan_x, pan_y = self._jitter_xy((0.25, 0.03), self.PAN_SPAWN_JITTER_XY)
            pan_pose = Pose([pan_x, pan_y, table_top + 0.050], _Q_UPRIGHT)
        # friction 1.5 / density 800: the handle grip is a geometric
        # width-matched pinch (not friction-driven), and the old 5.0/2500
        # values made the 0.76 kg pan tip instead of slide when centred on
        # the cooktop and droop ~47 deg in the pinch (measured, probe19).
        pan_entity = load_mesh(
            self.scene,
            ASSETS_PATH / "objects" / _PAN_ID / "visual" / "base0.glb",
            pan_pose,
            scale=(self._pan_scale,) * 3,
            convex=True,
            is_static=False,
            friction=1.5,
            density=800.0,
            decompose_object_error_threshold=0.0,
        )
        self.pan_actor = Actor(pan_entity, _read_model_data(_PAN_ID, 0), _PAN_ID)

        self.food_actors = []
        for spec in self._enabled_food_specs():
            x, y = self._jitter_xy(spec.spawn_xy, self.FOOD_SPAWN_JITTER_XY)
            actor = load_object(
                self.scene,
                Pose([x, y, table_top + spec.spawn_z_offset], spec.spawn_quat),
                spec.asset_id,
                model_id=spec.model_id,
                convex=True,
                is_static=False,
                friction=spec.friction,
            )
            self.food_actors.append((actor, spec))

        self.target = TargetSpec(
            object=self.pan_actor.entity,
            position=lambda: np.array([self._cooktop_xy[0], self._cooktop_xy[1], self.TABLE_TOP_Z]),
            label="cooktop",
        )

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    def _actor_pose(self, actor: Actor) -> Pose:
        return Pose(
            to_numpy(actor.entity.get_pos()).ravel()[:3],
            to_numpy(actor.entity.get_quat()).ravel()[:4],
        )

    def _actor_center(self, actor: Actor, asset_id: str, model_id: int = 0) -> np.ndarray:
        pose = actor.get_pose()
        R = t3d.quaternions.quat2mat(np.asarray(pose.q, dtype=float))
        return np.asarray(pose.p, dtype=float) + R @ _scaled_center(asset_id, model_id)

    def _pan_aabb(self):
        """World-frame (lo, hi) corners of the pan, from the live entity AABB.

        The skillet is spawned with authored local +Y mapped to world +Z. We
        read the bowl geometry from the live AABB so it remains robust to pose.
        """
        ab = np.asarray(
            to_numpy(self.pan_actor.entity.get_AABB()), dtype=float,
        ).reshape(-1, 3)
        return ab.min(0), ab.max(0)

    def _pan_bowl_center(self) -> np.ndarray:
        # The mesh origin sits BETWEEN bowl and handle: the true bowl centre is
        # 5.5 cm from the origin on the body side (mesh-measured).  Using the
        # origin xy here made foods drop onto the handle-side rim.
        pos = to_numpy(self.pan_actor.entity.get_pos()).ravel()[:3]
        lo, _hi = self._pan_aabb()
        body_dir = -self._pan_handle_dir_xy()
        return np.array([
            float(pos[0] + self.PAN_BOWL_FROM_ORIGIN * body_dir[0]),
            float(pos[1] + self.PAN_BOWL_FROM_ORIGIN * body_dir[1]),
            float(lo[2]) + 0.012,
        ])

    # Flat-pan mesh origin sits ~2.4mm above its own bottom face (measured);
    # add that so the bottom rests flush on the cooktop, no floating gap.
    PAN_ORIGIN_ABOVE_BOTTOM = 0.0024

    def _pan_origin_target_on_cooktop(self) -> np.ndarray:
        # Land the BOWL centre on the cooktop centre: the mesh origin is
        # 5.5 cm on the handle side of the bowl, so back the origin target off
        # along the live bowl->handle direction.  Rest the pan bottom flush on
        # the cooktop surface.
        cooktop_top = self.TABLE_TOP_Z + float(self.KITCHEN_COOKTOP.get("height", 0.04))
        h = self._pan_handle_dir_xy()
        return np.array([
            self._cooktop_xy[0] + self.PAN_BOWL_FROM_ORIGIN * h[0],
            self._cooktop_xy[1] + self.PAN_BOWL_FROM_ORIGIN * h[1],
            cooktop_top + self.PAN_ORIGIN_ABOVE_BOTTOM,
        ])

    def _pan_rim_z(self) -> float:
        _lo, hi = self._pan_aabb()
        return float(hi[2])

    def _rotated_top_down_tcp(self, pos, yaw_deg: float) -> Pose:
        if abs(float(yaw_deg)) < 1e-6:
            return self._top_down_tcp(pos)
        yaw = np.deg2rad(float(yaw_deg))
        Rz = np.array([
            [np.cos(yaw), -np.sin(yaw), 0.0],
            [np.sin(yaw), +np.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ])
        return Pose(
            np.asarray(pos, dtype=float),
            t3d.quaternions.mat2quat(self._R_DOWN @ Rz),
        )

    # ------------------------------------------------------------------
    # Control helpers
    # ------------------------------------------------------------------

    FINGER_FRICTION = 4.0

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
            lower_value=-self.FINGER_FORCE_N,
            upper_value=+self.FINGER_FORCE_N,
        )
        # Higher pad friction slows the drooping handle bar's creep through
        # the width-matched pinch (task-scoped: the scene rebuilds per reset).
        for link in getattr(arm.entity, "links", []):
            if "finger" not in str(getattr(link, "name", "")).lower():
                continue
            for g in getattr(link, "geoms", []):
                try:
                    g.set_friction(float(self.FINGER_FRICTION))
                except Exception:
                    pass

    def _move_tcp_cartesian(
        self,
        start_tcp: Pose,
        end_tcp: Pose,
        arm_tag: str,
        n_steps: int = 48,
        sim_per_step: int = 18,
    ) -> bool:
        """Smooth TCP interpolation with a fixed TCP orientation.

        Used for the skillet after grasping the handle.  Long RRT
        transits can wrap the Franka wrist and shake the pan loose; this
        keeps the handle pose continuous while PD-control still executes
        the actual arm motion.
        """
        arm = self.robot.get_arm(arm_tag)
        from ..robot.franka_robot import _get_dof_idx, to_numpy as _to_numpy

        base_target = getattr(arm, "_cached_target", None)
        if base_target is None:
            base_target = _to_numpy(arm.entity.get_qpos())
        base_target = np.asarray(base_target, dtype=float).ravel().copy()

        measured = np.asarray(_to_numpy(arm.entity.get_qpos()), dtype=float).ravel()
        seed_target = base_target.copy()
        for joint in arm.arm_joints:
            idx = _get_dof_idx(joint) if joint is not None else None
            if idx is not None:
                seed_target[idx] = float(measured[idx])
        arm._cached_target = seed_target

        qpos_full = base_target.copy()
        start_p = np.asarray(start_tcp.p, dtype=float)
        end_p = np.asarray(end_tcp.p, dtype=float)
        for i in range(1, int(n_steps) + 1):
            alpha = i / float(n_steps)
            pos = start_p * (1.0 - alpha) + end_p * alpha
            tcp = Pose(pos, end_tcp.q)
            arm_qpos = self._solve_ik(tcp, arm_tag)
            if arm_qpos is None:
                return False
            qpos_full = base_target.copy()
            for j_idx, joint in enumerate(arm.arm_joints):
                idx = _get_dof_idx(joint) if joint is not None else None
                if idx is not None:
                    qpos_full[idx] = float(arm_qpos[j_idx])
            arm.entity.control_dofs_position(qpos_full)
            arm._cached_target = qpos_full.copy()
            self.robot.set_gripper(arm.gripper_val, arm_tag)
            for _ in range(int(sim_per_step)):
                self.step_sim()
        return True

    # Flat pan: the handle extends from the bowl along the pan's local -Z
    # (world +Y at spawn).  Grasp it top-down with the fingers closing across
    # the 1.5 cm handle width.  Mesh-measured (base0.glb, scale 0.14): the
    # handle is a bar at RIM height, 2.5-3.3 cm ABOVE THE PAN BOTTOM — not a
    # low tab near the table — so the grasp z must come from the live pan
    # AABB bottom, not from TABLE_TOP_Z.
    PAN_HANDLE_GRASP_OFFSET = 0.075   # along bowl->handle dir from the origin
    PAN_HANDLE_Z_ABOVE_BOTTOM = 0.029  # centre of the measured handle band
    # Bowl centre sits 5.5 cm from the mesh origin on the body side (the
    # origin is between bowl and handle, NOT at the bowl centre).
    PAN_BOWL_FROM_ORIGIN = 0.055

    def _pan_handle_dir_xy(self) -> np.ndarray:
        """Live bowl->handle direction (local -Z), projected to world xy."""
        if self.pan_actor is None:
            # Called during load_actors before the pan exists (spawn-target
            # computation): the pan spawns at _Q_UPRIGHT, handle along +Y.
            return np.array([0.0, 1.0, 0.0])
        quat = to_numpy(self.pan_actor.entity.get_quat()).ravel()[:4]
        R = t3d.quaternions.quat2mat(np.asarray(quat, dtype=float))
        d = R @ np.array([0.0, 0.0, -1.0])
        d[2] = 0.0
        n = float(np.linalg.norm(d))
        return d / n if n > 1e-6 else np.array([0.0, 1.0, 0.0])

    def _pan_handle_grasp(self, offset: float) -> tuple[np.ndarray, float]:
        """Grasp point on the handle + finger yaw closing across its width."""
        pan_pos = to_numpy(self.pan_actor.entity.get_pos()).ravel()[:3]
        lo, _hi = self._pan_aabb()
        h = self._pan_handle_dir_xy()
        grasp_p = np.array([
            float(pan_pos[0] + offset * h[0]),
            float(pan_pos[1] + offset * h[1]),
            float(lo[2]) + self.PAN_HANDLE_Z_ABOVE_BOTTOM,
        ])
        # Closing axis at yaw t is (-sin t, -cos t); solve for it perpendicular
        # to the handle direction.  For the spawn pose (handle +Y) this is 90.
        yaw_deg = float(np.rad2deg(np.arctan2(-h[1], h[0])))
        return grasp_p, yaw_deg

    # Width-matched close for the 22 mm handle bar: full close (0.0) drives
    # 80 N straight through Genesis's compliant contacts (measured: fingers
    # pass 8-10 mm through the bar).  0.22 -> finger qpos ~0.009/side = a
    # geometric pinch that lifts without squeezing the bar out.
    PAN_HANDLE_CLOSE = 0.22
    # The drooping pan creeps out of the pinch in ~1200 sim steps from close
    # (measured): the lift+transit+descent below are paced to finish well
    # inside that budget, and the pan is supported again from touchdown on.
    PAN_DRAG_ITERS = 3
    # Pivot-flatten drifts the bowl ~0.12 m along the handle direction as the
    # drooped pan rotates flat about its resting edge (measured, 3 seeds):
    # aim upstream by this much so the pan lands centred directly and the
    # drag loop below is only a fallback.
    PAN_LANDING_DRIFT = 0.12

    def _pan_bar_z_band(self) -> tuple[float, float] | None:
        """z-range of the handle bar's COLLISION hulls (live), or None."""
        try:
            pan = self.pan_actor.entity
            pan_pos = to_numpy(pan.get_pos()).ravel()[:3]
            av = np.concatenate([
                np.asarray(to_numpy(g.get_verts()), dtype=float).reshape(-1, 3)
                for g in pan.geoms
            ], 0)
            h = self._pan_handle_dir_xy()
            rel = av - pan_pos
            along = rel[:, 0] * h[0] + rel[:, 1] * h[1]
            perp = np.abs(rel[:, 0] * (-h[1]) + rel[:, 1] * h[0])
            bar = av[(along > 0.03) & (perp < 0.02)]
            if len(bar) == 0:
                return None
            return float(bar[:, 2].min()), float(bar[:, 2].max())
        except Exception:
            return None

    def _place_pan_on_cooktop(self, arm_tag: str) -> bool:
        arm = self.robot.get_arm(arm_tag)
        tcp_offset = arm.tcp_offset
        cooktop_top = self.TABLE_TOP_Z + float(self.KITCHEN_COOKTOP.get("height", 0.04))

        for _ in range(40):
            self.step_sim()

        # ---- grasp the handle bar once ----
        z_table = float(to_numpy(self.pan_actor.entity.get_pos()).ravel()[2])
        self.open_gripper(arm_tag)
        grasp_p, yaw = self._pan_handle_grasp(self.PAN_HANDLE_GRASP_OFFSET)
        band = self._pan_bar_z_band()
        if band is not None:
            grasp_p[2] = (band[0] + band[1]) / 2
        above_p = np.array([grasp_p[0], grasp_p[1], self.TABLE_TOP_Z + 0.28])

        above_link = tcp_to_link_pose(
            self._rotated_top_down_tcp(above_p, yaw), tcp_offset,
        )
        if self._move_seeded(above_link.to_pose7(), arm_tag) is None:
            return False
        if not self._move_tcp_cartesian(
            self._rotated_top_down_tcp(above_p, yaw),
            self._rotated_top_down_tcp(grasp_p, yaw),
            arm_tag, n_steps=30, sim_per_step=18,
        ):
            return False
        for _ in range(200):
            self.step_sim()  # let PD converge before closing
        self.set_gripper(self.PAN_HANDLE_CLOSE, arm_tag, num_steps=100)
        for _ in range(120):
            self.step_sim()

        # Fast low lift: the drooping pan creeps out of the pinch on a
        # ~1200-step clock, so the airborne phase must be brisk.
        lift_p = grasp_p + np.array([0.0, 0.0, 0.12])
        if not self._move_tcp_cartesian(
            self._rotated_top_down_tcp(grasp_p, yaw),
            self._rotated_top_down_tcp(lift_p, yaw),
            arm_tag, n_steps=16, sim_per_step=12,
        ):
            return False
        z_after = float(to_numpy(self.pan_actor.entity.get_pos()).ravel()[2])
        if z_after - z_table < 0.04:
            print(f"[skillet] pan lift check failed "
                  f"(dz={z_after - z_table:.3f}m)")
            self.open_gripper(arm_tag, num_steps=80)
            for _ in range(120):
                self.step_sim()
            return False

        # ---- fast transit: aim the LIVE pan origin over the target,
        # upstream by the pivot-flatten landing drift ----
        target_origin = self._pan_origin_target_on_cooktop()
        tcp_now = np.asarray(arm.get_ee_pose()[:3], dtype=float)
        pan_now = to_numpy(self.pan_actor.entity.get_pos()).ravel()[:3]
        off_xy = pan_now[:2] - tcp_now[:2]
        hdir = self._pan_handle_dir_xy()
        above_t = np.array([
            float(target_origin[0] - off_xy[0] - self.PAN_LANDING_DRIFT * hdir[0]),
            float(target_origin[1] - off_xy[1] - self.PAN_LANDING_DRIFT * hdir[1]),
            self.PAN_TRANSPORT_Z,
        ])
        if not self._move_tcp_cartesian(
            self._rotated_top_down_tcp(tcp_now, yaw),
            self._rotated_top_down_tcp(above_t, yaw),
            arm_tag, n_steps=36, sim_per_step=10,
        ):
            return False
        if float(to_numpy(self.pan_actor.entity.get_pos()).ravel()[2]) <= z_table + 0.03:
            print("[skillet] pan slipped out during transit")
            return False

        # ---- pivot-flatten: descend PAST first touch; the drooped pan
        # rotates flat about its resting edge while still gripped ----
        tcp_end_z = cooktop_top + self.PAN_HANDLE_Z_ABOVE_BOTTOM + 0.004
        cur = above_t.copy()
        touched = False
        while cur[2] > tcp_end_z:
            step_dn = 0.030 if not touched else 0.015
            nxt = cur + np.array([0.0, 0.0, -step_dn])
            nxt[2] = max(nxt[2], tcp_end_z)
            if not self._move_tcp_cartesian(
                self._rotated_top_down_tcp(cur, yaw),
                self._rotated_top_down_tcp(nxt, yaw),
                arm_tag, n_steps=3, sim_per_step=10,
            ):
                return False
            cur = nxt
            lo_now, _hi = self._pan_aabb()
            if not touched and float(lo_now[2]) <= cooktop_top + 0.006:
                touched = True  # supported from here on: creep clock stopped

        # ---- centre the flat pan by dragging it while still gripped ----
        for _ in range(self.PAN_DRAG_ITERS):
            lo, hi = self._pan_aabb()
            if float(hi[2] - lo[2]) > 0.06:
                break  # tilted: stop nudging, release and let it settle
            bowl = self._pan_bowl_center()
            err = np.array(self._cooktop_xy, dtype=float) - bowl[:2]
            # exit at 0.05 so post-release settling stays inside the 0.09
            # success gate (measured: a 0.07 exit left the bowl at 0.0975)
            if float(np.linalg.norm(err)) <= 0.05:
                break
            step = np.clip(err, -0.08, 0.08)
            nxt = cur + np.array([step[0], step[1], 0.0])
            if not self._move_tcp_cartesian(
                self._rotated_top_down_tcp(cur, yaw),
                self._rotated_top_down_tcp(nxt, yaw),
                arm_tag, n_steps=14, sim_per_step=15,
            ):
                break
            cur = nxt
            for _ in range(40):
                self.step_sim()

        self.open_gripper(arm_tag, num_steps=80)
        for _ in range(200):
            self.step_sim()

        self._move_tcp_cartesian(
            self._rotated_top_down_tcp(cur, yaw),
            self._rotated_top_down_tcp(
                np.array([cur[0], cur[1], self.PAN_TRANSPORT_Z]), yaw,
            ),
            arm_tag, n_steps=16, sim_per_step=12,
        )
        for _ in range(120):
            self.step_sim()
        return True

    def _gentle_place_food(self, actor, spec, arm_tag, hover_tcp, make_tcp):
        """Lower the gripped food to just above the bowl floor, then release.

        Releasing from above the rim free-falls the food into the skillet,
        where elongated items (e.g. the patty) tip 90 deg onto an edge.
        Instead, measure the live food-center-to-TCP offset at the current
        hover pose and descend so the food center ends ~1.5 cm above the bowl
        interior, depositing it flat.  ``make_tcp(pos)`` builds the TCP pose at
        a world xyz with the path's grasp orientation.  Returns the release
        TCP pose, or ``None`` on IK failure.
        """
        bowl = self._pan_bowl_center()
        food_z = float(self._actor_center(actor, spec.asset_id, spec.model_id)[2])
        held_off = food_z - float(np.asarray(hover_tcp.p, dtype=float)[2])
        ext = _scaled_extents(spec.asset_id, spec.model_id)
        food_half = 0.5 * float(np.min(ext))
        target_food_z = bowl[2] + food_half + 0.015
        target_tcp_z = target_food_z - held_off
        # Never drive the fingers more than 1 cm below the rim.
        target_tcp_z = max(target_tcp_z, self._pan_rim_z() - 0.01)
        release_tcp = make_tcp(np.array([bowl[0], bowl[1], target_tcp_z]))
        if not self._move_tcp_cartesian(
            hover_tcp, release_tcp, arm_tag, n_steps=44, sim_per_step=22,
        ):
            return None
        # PD tracking lags the commanded pose by 1-2 cm; releasing before it
        # converges drops the food short (onto the near rim).  Hold the last
        # command until the TCP xy is actually over the bowl.
        target_xy = np.asarray(release_tcp.p, dtype=float)[:2]
        for _ in range(200):
            self.step_sim()
            tcp_xy = np.asarray(
                self.robot.get_arm(arm_tag).get_ee_pose()[:2], dtype=float,
            )
            if float(np.linalg.norm(tcp_xy - target_xy)) < 0.006:
                break
        self.open_gripper(arm_tag, num_steps=100)
        for _ in range(180):
            self.step_sim()
        return release_tcp

    def _pick_and_drop_food(self, actor: Actor, spec: FoodSpec, arm_tag: str) -> bool:
        if spec.side_grasp:
            return self._pick_and_drop_food_side_grasp(actor, spec, arm_tag)
        if spec.use_grasp_helper:
            return self._pick_and_drop_food_with_grasp_helper(actor, spec, arm_tag)

        arm = self.robot.get_arm(arm_tag)
        tcp_offset = arm.tcp_offset
        transport_z = self.TABLE_TOP_Z + self.FOOD_TRANSPORT_ABOVE_TABLE

        center = self._actor_center(actor, spec.asset_id, spec.model_id)
        grasp_pos = center.copy()
        grasp_pos[2] = max(self.TABLE_TOP_Z + 0.004, grasp_pos[2] - spec.below_center)

        self.open_gripper(arm_tag, num_steps=80)
        above = np.array([center[0], center[1], transport_z])
        above_link = tcp_to_link_pose(
            self._rotated_top_down_tcp(above, spec.tcp_yaw_deg), tcp_offset,
        )
        if self._move_seeded(above_link.to_pose7(), arm_tag) is None:
            return False

        self._move_screw(np.array([center[0], center[1], center[2] + 0.13]), arm_tag)
        center = self._actor_center(actor, spec.asset_id, spec.model_id)
        grasp_pos[:2] = center[:2]
        self._move_screw(grasp_pos, arm_tag)
        for _ in range(80):
            self.step_sim()

        z_before = float(self._actor_center(actor, spec.asset_id, spec.model_id)[2])
        self.set_gripper(spec.close_value, arm_tag, num_steps=120)
        for _ in range(150):
            self.step_sim()

        self._move_screw(np.array([grasp_pos[0], grasp_pos[1], transport_z]), arm_tag)
        for _ in range(60):
            self.step_sim()
        z_after = float(self._actor_center(actor, spec.asset_id, spec.model_id)[2])
        if z_after - z_before < 0.04:
            self.open_gripper(arm_tag, num_steps=80)
            return False
        if spec.name == "apple":
            # The round apple cages best with a shallow close on the table,
            # then needs a firmer hold once its weight is on the fingers.
            self.set_gripper(0.20, arm_tag, num_steps=100)
            for _ in range(160):
                self.step_sim()

        bowl = self._pan_bowl_center()
        transit_steps = 90 if spec.name == "apple" else 36
        transit_sim_steps = 30 if spec.name == "apple" else 18
        lift_tcp = self._rotated_top_down_tcp(
            np.array([grasp_pos[0], grasp_pos[1], transport_z]),
            spec.tcp_yaw_deg,
        )
        above_drop_tcp = self._rotated_top_down_tcp(
            np.array([bowl[0], bowl[1], transport_z]),
            spec.tcp_yaw_deg,
        )
        if not self._move_tcp_cartesian(
            lift_tcp,
            above_drop_tcp,
            arm_tag,
            n_steps=transit_steps,
            sim_per_step=transit_sim_steps,
        ):
            return False
        for _ in range(50):
            self.step_sim()

        make_tcp = lambda pos: self._rotated_top_down_tcp(pos, spec.tcp_yaw_deg)
        if self._gentle_place_food(actor, spec, arm_tag, above_drop_tcp, make_tcp) is None:
            return False

        retreat_tcp = self._rotated_top_down_tcp(
            np.array([bowl[0], bowl[1], transport_z]),
            spec.tcp_yaw_deg,
        )
        self.move_and_execute(tcp_to_link_pose(retreat_tcp, tcp_offset).to_pose7(), arm_tag)
        for _ in range(50):
            self.step_sim()
        return True

    def _side_grasp_tcp(self, center: np.ndarray) -> Pose:
        """Side grasp for round foods.

        TCP +Z points from robot side into the object, TCP +X is world-up,
        so the Franka fingers close horizontally around the equator rather
        than pressing down on the round mesh.
        """
        tcp_x = np.array([0.0, 0.0, 1.0])
        tcp_z = np.array([0.0, 1.0, 0.0])
        tcp_y = np.cross(tcp_z, tcp_x)
        R = np.column_stack([tcp_x, tcp_y, tcp_z])
        return Pose(np.asarray(center, dtype=float), t3d.quaternions.mat2quat(R))

    def _pick_and_drop_food_side_grasp(
        self,
        actor: Actor,
        spec: FoodSpec,
        arm_tag: str,
    ) -> bool:
        arm = self.robot.get_arm(arm_tag)
        tcp_offset = arm.tcp_offset
        transport_z = self.TABLE_TOP_Z + self.FOOD_TRANSPORT_ABOVE_TABLE

        center = self._actor_center(actor, spec.asset_id, spec.model_id)
        grasp_p = center.copy()
        grasp_p[2] = max(self.TABLE_TOP_Z + 0.025, grasp_p[2] + 0.002)
        grasp_tcp = self._side_grasp_tcp(grasp_p)
        R = t3d.quaternions.quat2mat(grasp_tcp.q)
        pre_tcp = Pose(grasp_tcp.p - 0.12 * R[:, 2], grasp_tcp.q)

        self.open_gripper(arm_tag, num_steps=80)
        if self.move_and_execute(tcp_to_link_pose(pre_tcp, tcp_offset).to_pose7(), arm_tag) is None:
            return False
        if not self._move_tcp_cartesian(pre_tcp, grasp_tcp, arm_tag, n_steps=28, sim_per_step=18):
            return False
        for _ in range(80):
            self.step_sim()

        z_before = float(self._actor_center(actor, spec.asset_id, spec.model_id)[2])
        self.set_gripper(spec.close_value, arm_tag, num_steps=140)
        for _ in range(180):
            self.step_sim()

        lift_tcp = Pose(
            grasp_tcp.p + np.array([0.0, 0.0, self.PAN_LIFT_M]),
            grasp_tcp.q,
        )
        if not self._move_tcp_cartesian(
            grasp_tcp, lift_tcp, arm_tag, n_steps=40, sim_per_step=18,
        ):
            return False
        for _ in range(60):
            self.step_sim()
        z_after = float(self._actor_center(actor, spec.asset_id, spec.model_id)[2])
        if z_after - z_before < 0.04:
            self.open_gripper(arm_tag, num_steps=80)
            return False

        bowl = self._pan_bowl_center()
        above_drop = Pose(np.array([bowl[0], bowl[1], transport_z]), grasp_tcp.q)
        if not self._move_tcp_cartesian(
            lift_tcp, above_drop, arm_tag, n_steps=44, sim_per_step=18,
        ):
            return False
        make_tcp = lambda pos: Pose(np.asarray(pos, dtype=float), grasp_tcp.q)
        if self._gentle_place_food(actor, spec, arm_tag, above_drop, make_tcp) is None:
            return False

        retreat_tcp = Pose(np.array([bowl[0], bowl[1], transport_z]), grasp_tcp.q)
        self.move_and_execute(tcp_to_link_pose(retreat_tcp, tcp_offset).to_pose7(), arm_tag)
        for _ in range(50):
            self.step_sim()
        return True

    def _pick_and_drop_food_with_grasp_helper(
        self,
        actor: Actor,
        spec: FoodSpec,
        arm_tag: str,
    ) -> bool:
        arm = self.robot.get_arm(arm_tag)
        tcp_offset = arm.tcp_offset
        transport_z = self.TABLE_TOP_Z + self.FOOD_TRANSPORT_ABOVE_TABLE

        for _ in range(40):
            self.step_sim()
        object_pose = self._actor_pose(actor)
        scale = _read_scale(spec.asset_id, spec.model_id)

        self.open_gripper(arm_tag, num_steps=80)
        result = self.select_and_execute_grasp(
            object_name=spec.asset_id,
            object_pose=object_pose,
            arm_tag=arm_tag,
            model_id=spec.model_id,
            robot_type="franka",
            max_candidates=8,
            object_scale=scale,
            table_z=self.TABLE_TOP_Z,
        )
        if result is None:
            return False
        _, _, grasp = result

        grasp_tcp = grasp.to_world(object_pose, scale)
        z_before = float(self._actor_center(actor, spec.asset_id, spec.model_id)[2])
        self.set_gripper(spec.close_value, arm_tag, num_steps=140)
        for _ in range(160):
            self.step_sim()

        lift_tcp = Pose(
            grasp_tcp.p + np.array([0.0, 0.0, self.PAN_LIFT_M]),
            grasp_tcp.q,
        )
        if not self._move_tcp_cartesian(
            grasp_tcp, lift_tcp, arm_tag, n_steps=40, sim_per_step=18,
        ):
            return False
        for _ in range(50):
            self.step_sim()
        z_after = float(self._actor_center(actor, spec.asset_id, spec.model_id)[2])
        if z_after - z_before < 0.04:
            self.open_gripper(arm_tag, num_steps=80)
            return False

        bowl = self._pan_bowl_center()
        above_drop = Pose(np.array([bowl[0], bowl[1], transport_z]), grasp_tcp.q)
        if not self._move_tcp_cartesian(
            lift_tcp, above_drop, arm_tag, n_steps=44, sim_per_step=18,
        ):
            return False
        for _ in range(50):
            self.step_sim()

        make_tcp = lambda pos: Pose(np.asarray(pos, dtype=float), grasp_tcp.q)
        if self._gentle_place_food(actor, spec, arm_tag, above_drop, make_tcp) is None:
            return False

        retreat_tcp = Pose(np.array([bowl[0], bowl[1], transport_z]), grasp_tcp.q)
        self.move_and_execute(tcp_to_link_pose(retreat_tcp, tcp_offset).to_pose7(), arm_tag)
        for _ in range(50):
            self.step_sim()
        return True

    def _food_in_skillet(self, actor: Actor, spec: FoodSpec) -> bool:
        bowl = self._pan_bowl_center()
        p = self._actor_center(actor, spec.asset_id, spec.model_id)
        dxy = float(np.linalg.norm(p[:2] - bowl[:2]))
        z_ok = p[2] > self.TABLE_TOP_Z + 0.02
        return bool(dxy <= self.FOOD_SUCCESS_XY and z_ok)

    # ------------------------------------------------------------------
    # Rollout and success
    # ------------------------------------------------------------------

    def play_once(self) -> bool:
        arm_tag = "right"
        self._boost_finger_pd(arm_tag)
        with self.suppress_recording():  # pre-task settle, excluded from recordings
            for _ in range(60):
                self.step_sim()

        # Key off the pan's actual spawn, not the difficulty flag: interrupt
        # forces a dynamic pan even in simplified mode, and its eval fallback
        # runs THIS play_once — gating on simplified_mode skipped the pan
        # phase there and left the pan on the counter forever.
        if not self._pan_starts_on_cooktop():
            if not self._place_pan_on_cooktop(arm_tag):
                return False

        ok_any = False
        for actor, spec in self.food_actors:
            if self._food_in_skillet(actor, spec):
                ok_any = True
                continue
            ok = self._pick_and_drop_food(actor, spec, arm_tag)
            ok_any = ok_any or ok
        for _ in range(180):
            self.step_sim()
        return ok_any

    def check_success(self) -> bool:
        if not self.plan_success:
            return False

        bowl = self._pan_bowl_center()
        pan_dxy = float(np.linalg.norm(bowl[:2] - self._cooktop_xy))
        pan_ok = pan_dxy <= self.PAN_SUCCESS_XY and bowl[2] > self.TABLE_TOP_Z

        all_food_ok = True
        for actor, spec in self.food_actors:
            ok = self._food_in_skillet(actor, spec)
            all_food_ok = all_food_ok and ok
        return bool(pan_ok and all_food_ok)
