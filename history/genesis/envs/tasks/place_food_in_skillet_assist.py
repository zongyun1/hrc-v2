"""Assist/cooperative variant scaffold for placing foods in a skillet.

Copied from ``place_food_in_skillet``.  Intended behavior:
the robot places the skillet on the stove, then robot and avatar each
place one food item into the skillet, working in parallel.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, replace

import numpy as np
import transforms3d as t3d

from ..avatar.eval_mode_mixin import EvalModeAvatarMixin
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


class PlaceFoodInSkilletAssist(EvalModeAvatarMixin, TopDownPickPlaceMixin, KitchenSceneMixin, BaseTask):
    """Robot and avatar cooperatively place foods into the skillet."""

    INSTRUCTION = (
        "place the skillet on the stove and put the foods in it while the "
        "human helps place one food"
    )
    use_avatar = True

    # Keep the first version focused on the task objects.
    FORWARD_Y_SHIFT = 0.20
    KITCHEN_LAYOUT = ()
    KITCHEN_BACKDROP = None
    KITCHEN_COOKTOP = {"xy": (-0.12, 0.08), "width": 0.42, "depth": 0.42}
    KITCHEN_ROBOT_KWARGS = {"pos": [0.0, -0.35, 0.765]}

    # Slightly closer camera than the kitchen default; frames pan + foods.
    recording_camera_pos = [0.85, -1.05, 1.55]
    recording_camera_lookat = [-0.10, 0.06, 0.88]
    side_camera_pos = [-1.10, -0.50, 1.35]
    side_camera_lookat = [-0.10, 0.06, 0.88]

    VIDEO_STRIDE = 6

    # Avatar stands on the +Y side of the table, facing the task region.
    AVATAR_BASE_ROT = np.array(
        [[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=np.float64,
    )
    avatar_init_pos = np.array([0.0, 0.72, -0.18], dtype=np.float64)
    avatar_init_rot = AVATAR_BASE_ROT

    AVATAR_APPROACH_FRAMES = 260
    AVATAR_TRANSPORT_FRAMES = 320
    AVATAR_RETRACT_FRAMES = 240
    # 0.12 let the avatar-dropped food free-fall ~15 cm and punch through the
    # compliant pan-bottom hull into the cooktop (measured: burger came to
    # rest at z 0.763, embedded under the pan).  Drop from just above the rim.
    AVATAR_DROP_ABOVE_RIM = 0.05
    AVATAR_START_DELAY_RANGE = (40, 180)
    AVATAR_NATURAL_BODY_MARGIN = 0.36
    AVATAR_NATURAL_BODY_Y_BOUNDS = (0.42, 0.82)
    AVATAR_NATURAL_YAW_LIMIT_DEG = 35.0
    AVATAR_NATURAL_SETTLE_STEPS = 20
    EVAL_TRIGGER_STEP_MIN = 20
    EVAL_TRIGGER_STEP_MAX = 60

    PAN_LIFT_M = 0.18
    PAN_TRANSPORT_Z = 1.05
    PAN_RELEASE_ABOVE_COOKTOP = 0.035
    FOOD_TRANSPORT_ABOVE_TABLE = 0.42
    FOOD_DROP_ABOVE_RIM = 0.12

    # Skillet mesh facts: local +Y is rim-up under _Q_UPRIGHT; local +Z is
    # handle-to-pan-body direction.  The functional point in model_data0 is
    # near the pan bowl center, in unscaled mesh units.
    PAN_BOWL_LOCAL_MESH = np.array([0.012, 0.130, 0.509], dtype=float)
    PAN_SUCCESS_XY = 0.09
    FOOD_SUCCESS_XY = 0.115
    PAN_SPAWN_JITTER_XY = 0.015
    FOOD_SPAWN_JITTER_XY = 0.030
    RANDOM_FOOD_POOL = ("burger", "bread", "apple")
    FOOD_RANDOM_REGIONS = (
        ((-0.44, -0.28), (-0.15, 0.16)),
        ((0.12, 0.28), (-0.12, 0.34)),
        ((-0.10, 0.10), (0.30, 0.38)),
    )
    APPLE_RANDOM_REGIONS = (
        ((0.12, 0.28), (0.18, 0.34)),
        ((-0.08, 0.08), (0.30, 0.38)),
    )
    BREAD_RANDOM_REGIONS = (
        ((0.12, 0.28), (0.18, 0.34)),
    )
    FOOD_MIN_SEP = 0.18
    AVATAR_RANDOM_FOOD_POOL = ("burger", "apple")

    def __init__(self, config: dict = None):
        cfg = dict(config or {})
        cfg.setdefault("robot_type", "franka")
        cfg.setdefault("robot_single_arm", True)
        cfg.setdefault("track_avatar_collision", False)
        super().__init__(cfg)
        self.pan_actor = None
        self.food_actors: list[tuple[Actor, FoodSpec]] = []
        self._avatar_food_idx = None
        self._robot_food_indices: list[int] = []
        self._cooktop_xy = np.array(self.KITCHEN_COOKTOP["xy"], dtype=float)
        self._pan_scale = _read_scale(_PAN_ID, 0)
        self._pan_bowl_local = self.PAN_BOWL_LOCAL_MESH * self._pan_scale
        self._pan_rim_above_origin = 0.272 * self._pan_scale

    # ------------------------------------------------------------------
    # Scene setup
    # ------------------------------------------------------------------

    def _enabled_food_specs(self) -> list[FoodSpec]:
        pool = [name for name in self.AVATAR_RANDOM_FOOD_POOL if name != "bread"]
        avatar_food = str(np.random.choice(pool))
        requested = list(np.random.permutation(["bread", avatar_food]))
        all_specs = {
            # Bread model 1 is the smaller bun-like variant used by
            # place_burger_fries; it is much easier to pick and fits the pan.
            "bread": FoodSpec(
                "bread", _BREAD_ID, 1, (-0.34, -0.10), _Q_UPRIGHT,
                0.055, 4.0, 0.30, 0.005, 0.0, False, False,
            ),
            "apple": FoodSpec(
                # Match the known-good place_burger_fries apple primitive:
                # model 0, top-down grasp, shallow close.  Put it just
                # behind/right of the cooktop, outside both the cooktop
                # footprint and the skillet handle's transfer path.
                "apple", _APPLE_ID, 0, (0.14, 0.34), _Q_UPRIGHT,
                0.045, 4.0, 0.30, 0.005, 0.0, False, True,
            ),
            "burger": FoodSpec(
                # model_5 = boxy wrapped burger. Map authored local +Y onto
                # world +Z so it lies flat in the pan. Box grips reliably via
                # the top-down primitive (use_grasp_helper=False); close 0.40
                # cages without crushing; friction 5.0.  Mirrors the base fix
                # (envs/task_bases/place_food_in_skillet.py).
                "burger", _BURGER_ID, 5, (0.14, 0.34), _Q_UPRIGHT,
                0.055, 5.0, 0.40, 0.005, 0.0, False, False,
            ),
        }
        return [all_specs[name] for name in requested]

    def _sample_food_spawn_xys(
        self,
        specs: list[FoodSpec],
    ) -> list[tuple[float, float]]:
        if not self.config.get("randomize_initial_positions", True):
            return []
        for _ in range(80):
            out = []
            for spec in specs:
                regions = (
                    self.APPLE_RANDOM_REGIONS
                    if spec.asset_id == _APPLE_ID
                    else self.BREAD_RANDOM_REGIONS
                    if spec.asset_id == _BREAD_ID
                    else self.FOOD_RANDOM_REGIONS
                )
                xr, yr = regions[int(np.random.randint(len(regions)))]
                out.append((
                    float(np.random.uniform(*xr)),
                    float(np.random.uniform(*yr)),
                ))
            ok = True
            for i in range(len(out)):
                for j in range(i + 1, len(out)):
                    if np.linalg.norm(np.asarray(out[i]) - np.asarray(out[j])) < self.FOOD_MIN_SEP:
                        ok = False
                        break
                if not ok:
                    break
            # The dynamic pan spawns at ~(0.25, 0.03) with an AABB footprint of
            # x +-0.08 / y +-0.135 around its origin: reject food samples that
            # would land inside it (they end up leaning on the bowl rim and
            # foul the handle grasp).
            if ok:
                pan_xy = np.asarray(
                    getattr(self, "_pan_spawn_xy", (0.25, 0.03)), dtype=float,
                )
                for x, y in out:
                    if abs(x - pan_xy[0]) < 0.14 and abs(y - pan_xy[1]) < 0.20:
                        ok = False
                        break
            if ok:
                return out
        return [
            self._jitter_xy(spec.spawn_xy, self.FOOD_SPAWN_JITTER_XY)
            for spec in specs
        ]

    def _jitter_xy(self, xy: tuple[float, float], amount: float) -> tuple[float, float]:
        if not self.config.get("randomize_initial_positions", True):
            return float(xy[0]), float(xy[1])
        dx, dy = np.random.uniform(-float(amount), float(amount), size=2)
        return float(xy[0] + dx), float(xy[1] + dy)

    def load_actors(self):
        self.build_kitchen()
        table_top = self.TABLE_TOP_Z

        # Dynamic skillet starts on the right-front counter with the same
        # upright orientation used by kitchen scenes.  CoACD is forced so
        # the pan interior is not treated as one filled convex hull.
        pan_x, pan_y = self._jitter_xy((0.25, 0.03), self.PAN_SPAWN_JITTER_XY)
        self._pan_spawn_xy = (pan_x, pan_y)
        pan_pose = Pose([pan_x, pan_y, table_top + 0.050], _Q_UPRIGHT)
        # friction 1.5 / density 800: the handle grip is a geometric pinch —
        # the old 5.0/2500 tipped the pan when centring it on the cooktop and
        # drooped it ~47 deg in the pinch (mirrors the base task; probe19).
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
        specs = self._enabled_food_specs()
        sampled_xys = self._sample_food_spawn_xys(specs)
        for i, spec in enumerate(specs):
            if sampled_xys:
                x, y = sampled_xys[i]
                spec = replace(spec, spawn_xy=(x, y))
            else:
                x, y = self._jitter_xy(spec.spawn_xy, self.FOOD_SPAWN_JITTER_XY)
                spec = replace(spec, spawn_xy=(x, y))
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

    def _food_center(self, actor: Actor, spec: FoodSpec) -> np.ndarray:
        return self._actor_center(actor, spec.asset_id, spec.model_id)

    def _pan_aabb(self):
        """World-frame (lo, hi) of the pan from its live AABB.

        Pan is spawned with authored local +Y mapped to world +Z. Bowl geometry
        is read from the live AABB so it stays correct regardless of pose.
        Mirrors envs/task_bases/place_food_in_skillet.py.
        """
        ab = np.asarray(
            to_numpy(self.pan_actor.entity.get_AABB()), dtype=float,
        ).reshape(-1, 3)
        return ab.min(0), ab.max(0)

    # Handle-grasp geometry — mirrors envs/task_bases/place_food_in_skillet.py
    # (this file is a FORK, not a subclass).  Mesh-measured facts: the handle
    # is a 1.5 cm-wide bar at RIM height (2.5-3.3 cm above the pan bottom),
    # and the mesh origin sits between bowl and handle (bowl centre 5.5 cm
    # away on the body side).
    PAN_HANDLE_GRASP_OFFSET = 0.075
    PAN_HANDLE_Z_ABOVE_BOTTOM = 0.029
    PAN_BOWL_FROM_ORIGIN = 0.055

    def _pan_handle_dir_xy(self) -> np.ndarray:
        """Live bowl->handle direction (local -Z), projected to world xy."""
        if self.pan_actor is None:
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
        yaw_deg = float(np.rad2deg(np.arctan2(-h[1], h[0])))
        return grasp_p, yaw_deg

    def _pan_bowl_center(self) -> np.ndarray:
        # True bowl centre is 5.5 cm from the mesh origin on the body side;
        # using the origin xy dropped foods onto the handle-side rim.
        pos = to_numpy(self.pan_actor.entity.get_pos()).ravel()[:3]
        lo, _hi = self._pan_aabb()
        body_dir = -self._pan_handle_dir_xy()
        return np.array([
            float(pos[0] + self.PAN_BOWL_FROM_ORIGIN * body_dir[0]),
            float(pos[1] + self.PAN_BOWL_FROM_ORIGIN * body_dir[1]),
            float(lo[2]) + 0.012,
        ])

    PAN_ORIGIN_ABOVE_BOTTOM = 0.0024

    def _pan_origin_target_on_cooktop(self) -> np.ndarray:
        # Land the BOWL centre on the cooktop centre (origin is 5.5 cm on the
        # handle side); rest the pan bottom flush on the cooktop.
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
            kp_value=9000.0,
            kv_value=250.0,
        )
        update_dofs_force_range_compat(
            arm.entity,
            arm._finger_dof_indices,
            lower_value=-80.0,
            upper_value=+80.0,
        )
        # Higher pad friction slows the drooping handle bar's creep through
        # the width-matched pinch (mirrors the base task).
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

    # Width-matched close for the 22 mm handle bar (mirrors the base task):
    # full close (0.0) drives 80 N straight through Genesis's compliant
    # contacts; 0.22 gives a geometric pinch that lifts without ejecting.
    PAN_HANDLE_CLOSE = 0.22
    PAN_DRAG_ITERS = 3
    # Pivot-flatten drifts the bowl ~0.12 m along the handle direction: aim
    # upstream so the pan lands centred directly (mirrors the base task).
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
        # Mirrors the base task's pick-carry-place: width-matched pinch on
        # the handle bar, fast carry inside the ~1200-step pinch-creep
        # budget, pivot-flatten landing, gripped drag to centre.
        arm = self.robot.get_arm(arm_tag)
        tcp_offset = arm.tcp_offset
        cooktop_top = self.TABLE_TOP_Z + float(self.KITCHEN_COOKTOP.get("height", 0.04))

        for _ in range(40):
            self.step_sim()

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

        lift_p = grasp_p + np.array([0.0, 0.0, 0.12])
        if not self._move_tcp_cartesian(
            self._rotated_top_down_tcp(grasp_p, yaw),
            self._rotated_top_down_tcp(lift_p, yaw),
            arm_tag, n_steps=16, sim_per_step=12,
        ):
            return False
        z_after = float(to_numpy(self.pan_actor.entity.get_pos()).ravel()[2])
        if z_after - z_table < 0.04:
            print(f"[skillet_assist] pan lift check failed "
                  f"(dz={z_after - z_table:.3f}m)")
            self.open_gripper(arm_tag, num_steps=80)
            for _ in range(120):
                self.step_sim()
            return False

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
            print("[skillet_assist] pan slipped out during transit")
            return False

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
                touched = True

        for _ in range(self.PAN_DRAG_ITERS):
            lo, hi = self._pan_aabb()
            if float(hi[2] - lo[2]) > 0.06:
                break
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
        release_tcp = self._gentle_place_food(actor, spec, arm_tag, above_drop_tcp, make_tcp)
        if release_tcp is None:
            return False

        retreat_tcp = self._rotated_top_down_tcp(
            np.array([bowl[0], bowl[1], transport_z]),
            spec.tcp_yaw_deg,
        )
        self._move_tcp_cartesian(
            release_tcp,
            retreat_tcp,
            arm_tag,
            n_steps=36,
            sim_per_step=18,
        )
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
        release_tcp = self._gentle_place_food(actor, spec, arm_tag, above_drop, make_tcp)
        if release_tcp is None:
            return False

        retreat_tcp = Pose(np.array([bowl[0], bowl[1], transport_z]), grasp_tcp.q)
        self._move_tcp_cartesian(
            release_tcp,
            retreat_tcp,
            arm_tag,
            n_steps=36,
            sim_per_step=18,
        )
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
        center = self._actor_center(actor, spec.asset_id, spec.model_id)
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
        release_tcp = self._gentle_place_food(actor, spec, arm_tag, above_drop, make_tcp)
        if release_tcp is None:
            return False

        retreat_tcp = Pose(np.array([bowl[0], bowl[1], transport_z]), grasp_tcp.q)
        self._move_tcp_cartesian(
            release_tcp,
            retreat_tcp,
            arm_tag,
            n_steps=36,
            sim_per_step=18,
        )
        for _ in range(50):
            self.step_sim()
        return True

    # ------------------------------------------------------------------
    # Rollout and success
    # ------------------------------------------------------------------

    def _food_in_skillet(self, actor: Actor, spec: FoodSpec) -> bool:
        bowl = self._pan_bowl_center()
        p = self._actor_center(actor, spec.asset_id, spec.model_id)
        dxy = float(np.linalg.norm(p[:2] - bowl[:2]))
        return bool(dxy <= self.FOOD_SUCCESS_XY and p[2] > self.TABLE_TOP_Z + 0.02)

    def _choose_avatar_food(self) -> tuple[int, list[int]]:
        candidates = [
            i for i, (_, spec) in enumerate(self.food_actors)
            if spec.asset_id != _PAN_ID
        ]
        if not candidates:
            return -1, []
        avatar_only_candidates = [
            i for i in candidates
            if self.food_actors[i][1].asset_id in (_BURGER_ID, _APPLE_ID)
        ]
        # Per-episode randomness is seeded by BaseTask.reset(seed).
        if avatar_only_candidates:
            # Current Franka grasps are reliable for bread here; burger
            # and apple remain randomized scene foods but are avatar-owned.
            avatar_idx = int(np.random.choice(avatar_only_candidates))
        else:
            avatar_idx = int(np.random.choice(candidates))
        robot_indices = [i for i in range(len(self.food_actors)) if i != avatar_idx]
        self._avatar_food_idx = avatar_idx
        self._robot_food_indices = robot_indices
        return avatar_idx, robot_indices

    def _delay_avatar_motion_start(self, delay_steps: int) -> None:
        if delay_steps <= 0 or self.avatar is None:
            return
        motion = self.avatar.motion_modules.get("pick_and_place")
        if motion is None or not getattr(motion, "frames", None):
            return
        hold = motion.frames[0]
        motion.frames = [hold] * int(delay_steps) + motion.frames
        motion.attach_frame += int(delay_steps)
        motion.detach_frame += int(delay_steps)

    def _start_avatar_food_place(self, actor: Actor, spec: FoodSpec) -> None:
        bowl = self._pan_bowl_center()
        pick_pos = self._food_center(actor, spec)
        place_pos = np.array([
            bowl[0],
            bowl[1],
            max(self._pan_rim_z() + self.AVATAR_DROP_ABOVE_RIM, self.TABLE_TOP_Z + 0.18),
        ])
        old_legacy = os.environ.get("AVATAR_PICK_PLACE_LEGACY")
        os.environ["AVATAR_PICK_PLACE_LEGACY"] = "1"
        try:
            self.avatar.pick_and_place(
                pick_pos=pick_pos,
                place_pos=place_pos,
                attach_obj=actor.entity,
                hand_id=None,
                approach_frames=self.AVATAR_APPROACH_FRAMES,
                transport_frames=self.AVATAR_TRANSPORT_FRAMES,
                retract_frames=self.AVATAR_RETRACT_FRAMES,
                natural=True,
                body_margin=self.AVATAR_NATURAL_BODY_MARGIN,
                body_y_bounds=self.AVATAR_NATURAL_BODY_Y_BOUNDS,
                yaw_limit_deg=self.AVATAR_NATURAL_YAW_LIMIT_DEG,
                settle_steps=self.AVATAR_NATURAL_SETTLE_STEPS,
            )
        finally:
            if old_legacy is None:
                os.environ.pop("AVATAR_PICK_PLACE_LEGACY", None)
            else:
                os.environ["AVATAR_PICK_PLACE_LEGACY"] = old_legacy
        self.avatar_collided = False
        self.avatar_collision_log = []
        delay = int(np.random.randint(
            int(self.AVATAR_START_DELAY_RANGE[0]),
            int(self.AVATAR_START_DELAY_RANGE[1]) + 1,
        ))
        self._delay_avatar_motion_start(delay)

    def _eval_at_reset(self) -> None:
        self._eval_assist_trigger_step = int(np.random.randint(
            self.EVAL_TRIGGER_STEP_MIN,
            self.EVAL_TRIGGER_STEP_MAX + 1,
        ))
        self._eval_assist_fired = False
        self._eval_assist_failed = False

    def _eval_at_step(self, step_idx: int) -> None:
        if (getattr(self, "_eval_assist_fired", False)
                or getattr(self, "_eval_assist_failed", False)):
            return
        if step_idx < int(getattr(self, "_eval_assist_trigger_step", self.EVAL_TRIGGER_STEP_MIN)):
            return
        self._eval_assist_fired = True
        try:
            avatar_idx, _robot_indices = self._choose_avatar_food()
            if avatar_idx < 0:
                self._eval_assist_failed = True
                return
            actor, spec = self.food_actors[avatar_idx]
            self._start_avatar_food_place(actor, spec)
        except Exception as exc:
            self._eval_assist_failed = True
            if self.config.get("debug_eval_avatar", False):
                print(f"[food_skillet_assist/eval] avatar assist failed: {exc}", flush=True)

    def play_once(self) -> bool:
        arm_tag = "right"
        self._boost_finger_pd(arm_tag)
        with self.suppress_recording():  # pre-task settle, excluded from recordings
            for _ in range(60):
                self.step_sim()

        if not self._place_pan_on_cooktop(arm_tag):
            return False

        if self.avatar is None or len(self.food_actors) < 2:
            ok_any = False
            for actor, spec in self.food_actors:
                ok = self._pick_and_drop_food(actor, spec, arm_tag)
                ok_any = ok_any or ok
                if not ok:
                    pass
            for _ in range(180):
                self.step_sim()
            return ok_any

        avatar_idx, robot_indices = self._choose_avatar_food()
        if avatar_idx < 0:
            return False
        avatar_actor, avatar_spec = self.food_actors[avatar_idx]

        self._start_avatar_food_place(avatar_actor, avatar_spec)

        ok_any = False
        for idx in robot_indices:
            actor, spec = self.food_actors[idx]
            ok = self._pick_and_drop_food(actor, spec, arm_tag)
            ok_any = ok_any or ok
            if not ok:
                pass
        if not self.avatar.spare():
            tail = 0
            while not self.avatar.spare() and tail < 1200:
                self.step_sim()
                tail += 1
        for _ in range(180):
            self.step_sim()
        return ok_any

    def play_blind_once(self) -> bool:
        arm_tag = "right"
        self._boost_finger_pd(arm_tag)
        seed = int(self.config.get("eval_seed", 0))
        rng = np.random.RandomState(seed + 50005)
        order = [int(i) for i in rng.permutation(len(self.food_actors))]
        self._blind_robot_food_order = order
        self._blind_robot_food_attempts = []
        for _ in range(60):
            self.step_sim()

        if not self._place_pan_on_cooktop(arm_tag):
            return False

        if (
            self.avatar is not None
            and bool(self.config.get("eval_mode", False))
            and not getattr(self, "_eval_assist_fired", False)
            and not getattr(self, "_eval_assist_failed", False)
        ):
            self._eval_at_step(int(getattr(
                self, "_eval_assist_trigger_step", self.EVAL_TRIGGER_STEP_MIN,
            )))

        ok_any = False
        for attempt_i, idx in enumerate(order):
            actor, spec = self.food_actors[idx]
            already_done = self._food_in_skillet(actor, spec)
            self._blind_robot_food_attempts.append({
                "idx": int(idx),
                "name": spec.name,
                "already_in_skillet": bool(already_done),
            })
            if already_done and attempt_i > 0:
                continue
            ok = self._pick_and_drop_food(actor, spec, arm_tag)
            ok_any = ok_any or ok
            if self.check_success():
                break
        if self.avatar is not None:
            tail = 0
            while not self.avatar.spare() and tail < 1200:
                self.step_sim()
                tail += 1
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

    def evaluate(self) -> dict:
        metrics = super().evaluate()
        metrics.update({
            "eval_mode": bool(self.config.get("eval_mode", False)),
            "eval_assist_trigger_step": int(getattr(
                self, "_eval_assist_trigger_step", -1,
            )),
            "eval_assist_fired": bool(getattr(self, "_eval_assist_fired", False)),
            "eval_assist_failed": bool(getattr(self, "_eval_assist_failed", False)),
            "eval_policy_step_count": int(getattr(
                self, "_eval_policy_step_count", 0,
            )),
            "assist_avatar_food_idx": int(
                -1 if getattr(self, "_avatar_food_idx", None) is None
                else getattr(self, "_avatar_food_idx")
            ),
            "blind_robot_food_order": [
                int(i) for i in getattr(self, "_blind_robot_food_order", [])
            ],
            "blind_robot_food_attempts": getattr(
                self, "_blind_robot_food_attempts", []
            ),
            "blind_robot_avatar_same_first": bool(
                getattr(self, "_blind_robot_food_order", [-999])[0]
                == int(
                    -1 if getattr(self, "_avatar_food_idx", None) is None
                    else getattr(self, "_avatar_food_idx")
                )
            ) if getattr(self, "_blind_robot_food_order", []) else False,
        })
        return metrics
