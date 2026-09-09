"""Oil Bottle Recovery: a knocked-over oil bottle next to a frying human;
the robot rights it and moves it to a safe zone away from the cooking area.

Scene
-----
- Kitchen counter (``KitchenSceneMixin``) with the same drops as
  ``frying_with_robot_pour`` (skillet/mug/wineglass/plate/bread cleared
  from the cooktop footprint), plus the default ``oil`` entry dropped.
  The recovered bottle is placed upright in a middle-counter safe zone,
  away from the back-row vinegar / soy-sauce bottles.
- Avatar holds pan + fork and runs the ``frying`` shake-loop, identical
  to ``frying_with_robot_pour``.
- A single ``029_olive-oil`` bottle is loaded dynamically near the
  avatar's right-hand swing arc (east of the cooktop). After the
  animation has been running long enough for the right hand to swing
  past, the bottle is "knocked over": gravity takes the rest.

Flow
----
1. Ease into frame 0 of the frying motion, snap pan + fork to hands,
   and start the long ``frying`` shake-loop.
2. Settle, then trigger knock-over (tilt + small lateral velocity in
   the swing direction).  Step physics until the bottle rests on its
   side.
3. Robot picks the fallen bottle (``select_and_execute_grasp`` on the
   live tipped pose), with avatar capsule pointcloud fed to the planner
   so the transit avoids the avatar's arm.
4. Lift, restore upright in mid-air over the safe zone.
5. Lateral transit + descend to the safe-zone xy.  Release.
6. Drain the rest of the frying loop.
"""

from __future__ import annotations

import json
import numpy as np
import transforms3d as t3d

from ..base_task import BaseTask
from ..genesis_compat import set_dofs_kp_kv_compat, update_dofs_kp_kv_compat
from ..utils import Pose, to_numpy
from ..scenes.kitchen import (
    KitchenSceneMixin,
    DEFAULT_KITCHEN_LAYOUT,
    KitchenItem,
)
from .frying_with_robot_pour import (
    _FRYING_MOTION_PKL,
    _PAN_ASSET,
    _FORK_ASSET,
    _OIL_TARGET_DIM_M,
    _PAN_TARGET_LEN_M,
    _FORK_TARGET_LEN_M,
    _AVATAR_FRAME_RATIO,
    _DROP_ROLES,
    _upright_q,
    _mesh_fit_scale,
    FryingWithRobotPour,
)

from ..object_catalog import get_entry

_RECOVERY_OIL_ENTRY = get_entry("029_olive-oil")
_RECOVERY_OIL_ID = _RECOVERY_OIL_ENTRY.object_id
_RECOVERY_OIL_MODEL_ID = _RECOVERY_OIL_ENTRY.model_id
_OIL_JITTER_XY = 0.015
_SAFE_TARGET_JITTER_XY = 0.02

# Spawn the bottle east of the cooktop (cooktop is at (0.0, -0.20)) inside
# the avatar's right-hand swing arc.  The right hand drives the fork
# above the pan during frying; this xy puts the bottle on the counter
# slightly east-and-south of the pan so the right hand passes overhead
# during shake cycles.
_BOTTLE_INIT_XY = (0.25, -0.35)

# Safe zone — middle counter, not the old back-row oil slot.  This keeps
# the carried bottle away from the vinegar / soy-sauce bottles at y=+0.35.
_SAFE_TARGET_XY = (0.36, 0.08)

# Tilt seed applied to the upright bottle once the avatar's right hand
# has swung close enough to "look like it bumped it".  At 50° the bottle
# is well past its angle of repose for a straight-walled cylinder and
# falls under gravity rather than re-righting itself.
_KNOCK_TILT_DEG = 50.0

# Small lateral push velocity applied at the same moment so the bottle
# slides a few centimetres in the swing direction.  Genesis exposes
# ``set_dofs_velocity`` on the entity's free root-joint dofs (3 trans +
# 3 rot); we wrap the call in try/except because the API is not
# guaranteed across Genesis versions.
_KNOCK_PUSH_VEL = 0.30  # m/s, +x (east — bottle falls into open counter, away from west cooktop / avatar)

# Hand-to-bottle xy distance at which we trigger the knock-over.  The
# avatar's right hand sweeps across an ~20 cm arc above the pan during
# each shake cycle; 0.25 m is the closest approach we can expect when
# the bottle is parked at the east edge of that arc.
_KNOCK_TRIGGER_DIST_M = 0.25

# Upper bound on number of physics steps we'll wait for the trigger
# before falling back to an unconditional knock-over.  The 2120-frame
# frying motion at frame_ratio=4 covers ~17 s = ~8500 physics steps, so
# 1500 is well within one shake cycle.
_KNOCK_TRIGGER_TIMEOUT_STEPS = 1500

# Settle window after the knock.  v14 showed the bottle still drifting
# 21 mm in x during the pre-grasp move — the gradual-tilt approach gives
# the bottle nontrivial momentum that takes longer to dissipate than 200
# steps.  Bumped to 600 to make the post-knock pose stable.
_KNOCK_SETTLE_STEPS = 1500  # was 600 — v38 showed bottle was at only 22° tilt (unstable, mid-fall) when sampling started; gripper contact during descent kept rolling it.  1500 steps (3 s real-time) lets it settle into a stable horizontal pose before sampling.

# Friction matches frying_with_robot_pour — verified to keep the bottle
# in the gripper through a 30 cm horizontal swing without slip.  v44m:
# back to v43's 2.0 (the upright attempt is dropped, so we don't need
# frying's 4.0; v44l with 4.0 + close=0.30 actually slipped immediately
# on the lateral place because close=0.30 leaves a 2.4 cm gap, less
# compression than v43's full-close + PD boost).
_OIL_FRICTION = 2.0

# Partial close avoids crushing the bottle fully shut in the gripper while
# still giving enough contact for the short carry to the middle counter.
_OIL_GRIP_CLOSE_TARGET = 0.04
_OIL_FINGER_KP = 9000.0
_OIL_FINGER_KV = 250.0
_PRE_UPRIGHT_LIFT_M = 0.12
_UPRIGHT_PLACE_HOVER_M = 0.13
_UPRIGHT_ENDPOINT_HOLD_STEPS = 360
_SUCCESS_DIST_XY_M = 0.15
_SUCCESS_TILT_DEG = 30.0

def _read_object_scale(asset: str, model_id: int = 0) -> float:
    from ..utils import ASSETS_PATH

    path = ASSETS_PATH / "objects" / asset / f"model_data{model_id}.json"
    if not path.exists():
        return 1.0
    raw = json.load(open(path)).get("scale", 1.0)
    return float(raw[0] if isinstance(raw, (list, tuple)) else raw)


def _read_object_height(
    asset: str,
    model_id: int = 0,
    fallback: float = _OIL_TARGET_DIM_M,
) -> float:
    from ..utils import ASSETS_PATH

    path = ASSETS_PATH / "objects" / asset / f"model_data{model_id}.json"
    if not path.exists():
        return fallback
    data = json.load(open(path))
    extents = data.get("extents")
    if not extents:
        return fallback
    scale = _read_object_scale(asset, model_id)
    return float(max(extents) * scale)


def _upright_origin_z_for_bottom(
    asset: str,
    model_id: int,
    bottom_z: float,
    fallback_height: float = _OIL_TARGET_DIM_M,
) -> float:
    from ..utils import ASSETS_PATH

    path = ASSETS_PATH / "objects" / asset / f"model_data{model_id}.json"
    if not path.exists():
        return float(bottom_z + fallback_height / 2)
    data = json.load(open(path))
    center = np.asarray(data.get("center", [0.0, 0.0, 0.0]), dtype=np.float64)
    extents = np.asarray(data.get("extents", [0.0, fallback_height, 0.0]), dtype=np.float64)
    scale = _read_object_scale(asset, model_id)
    bmin = (center - extents / 2.0) * scale
    bmax = (center + extents / 2.0) * scale
    corners = np.array([
        [x, y, z]
        for x in (bmin[0], bmax[0])
        for y in (bmin[1], bmax[1])
        for z in (bmin[2], bmax[2])
    ], dtype=np.float64)
    R = t3d.quaternions.quat2mat(_upright_q())
    world_min = (R @ corners.T).T.min(axis=0)
    return float(bottom_z - world_min[2])


class OilBottleRecovery(KitchenSceneMixin, BaseTask):
    """Robot recovers a knocked-over oil bottle from the cooking zone."""

    INSTRUCTION = "recover the knocked-over oil bottle from the cooking area and place it upright"
    OBJECT_SET = ["029_olive-oil"]
    use_avatar = True
    EVAL_TRIGGER_STEP_MIN = 40
    EVAL_TRIGGER_STEP_MAX = 100
    EVAL_KNOCK_TIMEOUT_STEPS = 80

    # Drop the same conflict items as frying, the default ``oil`` (we
    # spawn our own dynamic copy), and the ``baguette`` (default
    # position xy=(-0.05, -0.33) sits right next to the cooktop and gets
    # in the way of the gripper's approach to the fallen bottle).
    KITCHEN_LAYOUT = tuple(
        KitchenItem(
            role=it.role, asset=it.asset, model_id=it.model_id,
            target_dim_m=it.target_dim_m,
            euler_deg=it.euler_deg, xy=it.xy,
            is_static=it.is_static, convex=it.convex,
            sink_z=it.sink_z, extra_z=it.extra_z, description=it.description,
        )
        for it in DEFAULT_KITCHEN_LAYOUT
        if it.role not in (_DROP_ROLES | {"oil", "baguette"})
    )

    # Default kitchen mount on the south side of the cooktop.  Earlier
    # iterations placed the robot at (+0.55, +0.30) — north of the
    # cooktop — which put the fallen bottle (y≈-0.33) directly *behind*
    # the robot.  mplib's IK then had to flip joint1 by ~180°, producing
    # approximate solutions whose execution landed the gripper 0.9 m off
    # target.  Mounting at (0.60, -0.30) (same as frying_with_robot_pour
    # and the rest of the kitchen tasks) puts both the bottle and the
    # safe zone (0.48, 0.35) inside Franka's natural forward reach, with
    # joint1 swings under 100°.
    KITCHEN_ROBOT_KWARGS = {"pos": [0.60, -0.30, 0.765]}

    # Same cooktop position as frying so the held pan lands above it.
    KITCHEN_COOKTOP = {"xy": (0.0, -0.20)}

    # Avatar pose copied from frying.
    avatar_init_pos = np.array([+0.05, -1.05, -0.03])

    # Cameras roughly mirror frying but pulled back a bit so the bottle
    # at (0.25, -0.35), the avatar, and the safe zone at (0.48, 0.35)
    # are all in frame.
    recording_camera_pos = [-0.55, -0.55, 1.10]
    recording_camera_lookat = [+0.20, -0.05, 0.90]
    side_camera_pos = [+0.20, -0.10, 2.05]
    side_camera_lookat = [+0.20, -0.10, 0.85]

    def __init__(self, config=None):
        cfg = dict(config or {})
        cfg.setdefault("robot_type", "franka")
        cfg.setdefault("robot_single_arm", True)
        # Avatar capsules feed the planner as obstacle pointcloud each
        # plan, identical to frying_with_robot_pour.
        cfg.setdefault("use_avatar_collider", True)
        cfg.setdefault("randomize_avatar", True)
        avatar_cfg = dict(cfg.get("avatar") or {})
        avatar_cfg.setdefault("generated_motion_data", _FRYING_MOTION_PKL)
        cfg["avatar"] = avatar_cfg
        super().__init__(cfg)
        self.resolve_target_object()   # registry/override hook (size-1 set)

    # ------------------------------------------------------------------
    # Scene
    # ------------------------------------------------------------------

    def reset(self, seed: int = 0):
        obs = super().reset(seed=seed)
        if self.avatar is not None and bool(self.config.get("eval_mode", False)):
            self._init_eval_recovery_state()
            obs = self.get_obs()
        return obs

    def take_action(self, action, action_type: str = "qpos"):
        if self.avatar is not None and bool(self.config.get("eval_mode", False)):
            self._eval_step_avatar()
        obs = super().take_action(action, action_type=action_type)
        if bool(self.config.get("eval_mode", False)):
            m = self._recovery_metrics()
            self.eval_success = bool(
                self.plan_success
                and m["target_dist_xy"] < _SUCCESS_DIST_XY_M
                and m["target_tilt_deg"] < _SUCCESS_TILT_DEG
                and m["avatar_ok"]
            )
        return obs

    def load_actors(self):
        from ..utils import load_mesh, ASSETS_PATH

        self.build_kitchen()

        if bool(self.config.get("randomize_layout", False)):
            bottle_jitter = np.random.uniform(
                -_OIL_JITTER_XY, _OIL_JITTER_XY, size=2,
            )
            safe_jitter = np.random.uniform(
                -_SAFE_TARGET_JITTER_XY, _SAFE_TARGET_JITTER_XY, size=2,
            )
        else:
            bottle_jitter = np.zeros(2, dtype=np.float64)
            safe_jitter = np.zeros(2, dtype=np.float64)
        self._bottle_init_xy = (
            float(_BOTTLE_INIT_XY[0] + bottle_jitter[0]),
            float(_BOTTLE_INIT_XY[1] + bottle_jitter[1]),
        )
        self._safe_target_xy = (
            float(_SAFE_TARGET_XY[0] + safe_jitter[0]),
            float(_SAFE_TARGET_XY[1] + safe_jitter[1]),
        )

        # --- Oil bottle: dynamic, using the current recovery asset and
        #     runtime-scale body grasp annotations. ---
        oil_scale = _read_object_scale(_RECOVERY_OIL_ID, _RECOVERY_OIL_MODEL_ID)
        oil_height = _read_object_height(_RECOVERY_OIL_ID, _RECOVERY_OIL_MODEL_ID)
        self._oil_scale = float(oil_scale)
        self._oil_target_dim_m = oil_height
        self._oil_upright_origin_z_on_table = _upright_origin_z_for_bottom(
            _RECOVERY_OIL_ID,
            _RECOVERY_OIL_MODEL_ID,
            self.TABLE_TOP_Z + 0.005,
            oil_height,
        )
        oil_init_pose = Pose(
            np.array([
                self._bottle_init_xy[0],
                self._bottle_init_xy[1],
                self._oil_upright_origin_z_on_table,
            ]),
            _upright_q(),
        )
        self.oil_entity = load_mesh(
            self.scene,
            ASSETS_PATH / "objects" / _RECOVERY_OIL_ID / "visual" / f"base{_RECOVERY_OIL_MODEL_ID}.glb",
            oil_init_pose,
            scale=(oil_scale, oil_scale, oil_scale),
            convex=True,
            is_static=False,
        )
        try:
            self.oil_entity.set_friction(_OIL_FRICTION)
        except Exception:
            pass

        # --- Pan + fork: same kinematic-driven props as frying. ---
        self._pan_scale = _mesh_fit_scale(_PAN_ASSET, 0, _PAN_TARGET_LEN_M)
        self._fork_scale = _mesh_fit_scale(_FORK_ASSET, 0, _FORK_TARGET_LEN_M)
        hover_pan = Pose(
            [self.avatar_init_pos[0] - 0.30, self.avatar_init_pos[1] + 0.30,
             self.TABLE_TOP_Z + 0.30],
            _upright_q(),
        )
        hover_fork = Pose(
            [self.avatar_init_pos[0] + 0.30, self.avatar_init_pos[1] + 0.30,
             self.TABLE_TOP_Z + 0.30],
            _upright_q(),
        )
        self.pan_entity = load_mesh(
            self.scene,
            ASSETS_PATH / "objects" / _PAN_ASSET / "visual" / "base0.glb",
            hover_pan, scale=(self._pan_scale,) * 3, convex=True,
            is_static=True,
        )
        self.fork_entity = load_mesh(
            self.scene,
            ASSETS_PATH / "objects" / _FORK_ASSET / "visual" / "base0.glb",
            hover_fork, scale=(self._fork_scale,) * 3, convex=True,
            is_static=True,
        )

    # ------------------------------------------------------------------
    # Borrow the pan + fork drivers, hand-frame helpers, planner-obstacle
    # refresh, and bottle-to-link transform from FryingWithRobotPour.
    # ------------------------------------------------------------------

    _PAN_GRIP_LOCAL_Z = FryingWithRobotPour._PAN_GRIP_LOCAL_Z
    _FORK_GRIP_LOCAL_X = FryingWithRobotPour._FORK_GRIP_LOCAL_X
    _FORK_FORWARD_GRIP_SHIFT_M = FryingWithRobotPour._FORK_FORWARD_GRIP_SHIFT_M
    _FORK_RIGHT_GRIP_SHIFT_M = FryingWithRobotPour._FORK_RIGHT_GRIP_SHIFT_M
    _PAN_BODY_CENTER_LOCAL_Z = FryingWithRobotPour._PAN_BODY_CENTER_LOCAL_Z
    # v44d: spatula attach tweaks restored — the v44/v44b/v44c grasp
    # regression turned out to be the parent's recently bumped
    # `_AVATAR_INFLATE_FACTOR=1.6` (see override above), not the spatula
    # compute.  These nudges only affect the spatula's visible pose.
    _FORK_NE_SHIFT_M = 0.03
    _FORK_UP_SHIFT_M = 0.02
    _FORK_EXTRA_DOWNTILT_DEG = -5.0
    _AVATAR_OBSTACLE_RES = FryingWithRobotPour._AVATAR_OBSTACLE_RES
    # v44d: override the parent's inflate (recently bumped to 1.6 for the
    # frying-pour swing) back down to 1.0.  At 1.6 the avatar's right-arm
    # capsule near the cooktop fattens enough to push mplib's pre-grasp
    # IK into a wider RRT branch search; v44/b/c all landed grasp 36–40
    # cm off target (silent IK relax) where v43 with the slimmer obstacle
    # pcd snapped to the bottle on attempt 1.  OBR's grasp at (0.34,
    # −0.35, 0.79) is south of the cooktop and doesn't need the extra
    # padding the pour-and-tilt swing does.
    _AVATAR_INFLATE_FACTOR = 1.0
    # FryingWithRobotPour._avatar_obstacle_points reads this when sampling
    # arm/hand capsules. OilBottleRecovery reuses that method but keeps the
    # base avatar inflation slimmer, so use no extra task-local arm padding.
    _AVATAR_ACTIVE_ARM_EXTRA_INFLATE_M = 0.0
    # v8 transit produced a 24278-step plan_screw trajectory (toppra duration
    # 97 s).  Bumped to 32 000 so the long but legitimate cartesian transit
    # from grasp xy back to the safe-zone xy can complete.
    _MAX_TRAJECTORY_STEPS = 32000
    _CABINET_X_HALF = FryingWithRobotPour._CABINET_X_HALF
    _CABINET_Y_RANGE = FryingWithRobotPour._CABINET_Y_RANGE
    _CABINET_Z_RANGE = FryingWithRobotPour._CABINET_Z_RANGE

    _get_entity_pose = FryingWithRobotPour._get_entity_pose
    _attach_items_to_hands = FryingWithRobotPour._attach_items_to_hands
    _drive_pan_horizontal = FryingWithRobotPour._drive_pan_horizontal
    _drive_fork_at_pan = FryingWithRobotPour._drive_fork_at_pan
    _avatar_obstacle_points = FryingWithRobotPour._avatar_obstacle_points
    _cabinet_obstacle_points = FryingWithRobotPour._cabinet_obstacle_points
    _bottle_to_link = FryingWithRobotPour._bottle_to_link
    step_sim = FryingWithRobotPour.step_sim

    # Cooktop occupies x=(-0.76,-0.34), y=(0.07,0.49), z up to 0.78.
    # The parent's `_refresh_planner_obstacles` adds avatar + cabinet
    # but NOT the cooktop, so v33-v36 plan_path reported Success on
    # paths that swept the elbow through the cooktop region — Genesis
    # physics then blocked j1 at runtime.  This override appends a
    # cooktop pcd to the parent's pcd so plan_path actually avoids it.
    _COOKTOP_X = (-0.78, -0.32)  # 2-cm margin around (-0.76, -0.34)
    _COOKTOP_Y = (+0.05, +0.51)
    _COOKTOP_Z = (0.74, 0.80)

    def _cooktop_obstacle_points(self) -> np.ndarray:
        step = self._AVATAR_OBSTACLE_RES
        xs = np.arange(self._COOKTOP_X[0], self._COOKTOP_X[1] + step, step)
        ys = np.arange(self._COOKTOP_Y[0], self._COOKTOP_Y[1] + step, step)
        zs = np.arange(self._COOKTOP_Z[0], self._COOKTOP_Z[1] + step, step)
        XX, YY, ZZ = np.meshgrid(xs, ys, zs, indexing="ij")
        return np.stack([XX.ravel(), YY.ravel(), ZZ.ravel()], axis=-1).astype(np.float64)

    def _refresh_planner_obstacles(self, arm_tag: str):
        """Parent's avatar+cabinet pcd + cooktop pcd."""
        avatar_pts = self._avatar_obstacle_points()
        cabinet_pts = self._cabinet_obstacle_points()
        cooktop_pts = self._cooktop_obstacle_points()
        parts = [p for p in (avatar_pts, cabinet_pts, cooktop_pts) if p is not None]
        if not parts:
            return
        pts = np.concatenate(parts, axis=0)
        arm = self.robot.get_arm(arm_tag)
        try:
            arm.planner.update_obstacles(pts, resolution=self._AVATAR_OBSTACLE_RES)
        except Exception:
            pass

    def _plan_path_with_fk_check(
        self, link_target, arm_tag, arm,
        max_fk_retries=5, max_snap_retries=10, snap_steps=100,
        fk_tol=0.005, log=True,
    ):
        """``plan_path`` (mplib RRT + plan_pose IK) with **two layers**
        of retry:

        1. **Snapshot retry**: if plan_path returns "invalid start
           state" / "Cannot find valid solution" because the live
           avatar capsules block IK or the start qpos collides with
           the avatar that moved since sampling, step the sim
           ``snap_steps`` frames (avatar continues frying) and refresh
           the obstacle pcd, then retry.  Up to ``max_snap_retries``
           snapshots.

        2. **FK retry within a snapshot**: even when plan_path returns
           Success, mplib can silently relax IK — the trajectory's end
           qpos has FK 1-3 cm off the requested pose (project memory
           ``mplib_silent_ik_relax``).  At each snapshot we try up to
           ``max_fk_retries`` times and keep the trajectory whose end
           qpos has the smallest FK error.  Each plan_path call uses a
           fresh RRT seed, so different attempts often converge to
           different IK branches.

        No Genesis IK fallback (user constraint).  No set_qpos for the
        robot — the trajectory is played through the standard PD
        controller via ``execute_plan``.
        """
        import torch as _torch
        target_pos = np.asarray(link_target.p, dtype=np.float64)
        best_result = None
        best_err = float("inf")
        for snap_idx in range(max_snap_retries):
            self._refresh_planner_obstacles(arm_tag)
            for attempt in range(max_fk_retries):
                res = arm.planner.plan_path(
                    arm.get_arm_qpos(), link_target.to_pose7(), log=False,
                )
                if res is None or not res.success or res.position.size == 0:
                    continue
                end_q = res.position[-1]
                saved = arm.entity.get_qpos()
                full = arm._build_qpos_with_arm(end_q)
                arm.entity.set_qpos(_torch.tensor(full, dtype=_torch.float32)
                                    if not hasattr(full, 'cpu') else full)
                arm.entity.get_links_pos()
                ee = to_numpy(arm.ee_link.get_pos()).ravel()[:3]
                arm.entity.set_qpos(saved)
                arm.entity.get_links_pos()
                fk_err = float(np.linalg.norm(ee - target_pos))
                if fk_err < best_err:
                    best_err = fk_err
                    best_result = res
                if fk_err < fk_tol:
                    return res, fk_err
            # Avatar continues animating; step forward and try again so
            # the live frying motion has a chance to clear the IK region.
            for _ in range(snap_steps):
                self.step_sim()
        return best_result, best_err

    # ------------------------------------------------------------------
    # Knock-over
    # ------------------------------------------------------------------

    def _right_hand_xy(self) -> np.ndarray | None:
        """World-XY of the avatar's right-hand frame, or None if no avatar."""
        if self.avatar is None:
            return None
        try:
            hand_pos, _ = self.avatar.robot._get_hand_frame(1)
        except Exception:
            return None
        return np.asarray(hand_pos, dtype=np.float64)[:2]

    def _knock_over_bottle(self):
        """Apply a gradual angular velocity about world +y to topple the
        bottle smoothly eastward (long axis ends along world +x).  This
        avoids the previous +y-tilt + post-knock ``set_quat`` snap that
        looked discontinuous; the bottle now falls under physics from
        the moment of contact and settles naturally with long axis +x."""
        # Angular velocity = 50° / 0.4 s ≈ 2.18 rad/s about world +y; the
        # top of the upright bottle swings in +x direction and past the
        # angle-of-repose (~30°) gravity completes the topple.
        omega = float(np.deg2rad(_KNOCK_TILT_DEG)) / 0.4
        try:
            # 6 free-body dofs: [vx, vy, vz, wx, wy, wz].  +x push for
            # the slide eastward and +y angular for the tilt direction.
            vel = np.array(
                [_KNOCK_PUSH_VEL, 0.0, 0.0, 0.0, +omega, 0.0],
                dtype=np.float64,
            )
            self.oil_entity.set_dofs_velocity(vel)
        except Exception:
            pass

    def _wait_for_knock_trigger(self) -> bool:
        """Step the sim until the right-hand passes within
        ``_KNOCK_TRIGGER_DIST_M`` of the bottle xy, or the timeout
        expires.  Returns True on trigger, False on timeout."""
        for i in range(_KNOCK_TRIGGER_TIMEOUT_STEPS):
            self.step_sim()
            if i % 10 != 0:
                continue
            hand_xy = self._right_hand_xy()
            if hand_xy is None:
                continue
            bottle = self._get_entity_pose(
                self.oil_entity, Pose(np.zeros(3), _upright_q()),
            )
            d = float(np.linalg.norm(hand_xy - np.asarray(bottle.p[:2])))
            if d < _KNOCK_TRIGGER_DIST_M:
                return True
        return False

    # ------------------------------------------------------------------
    # Eval/testbed mode: policy eval runs a pre-policy warmup by default so
    # the bottle is already knocked over for the first policy observation.
    # If that hook is disabled/unavailable, take_action keeps the older
    # delayed randomized knock-over as a fallback.
    # ------------------------------------------------------------------

    def _init_eval_recovery_state(self):
        self._attach_items_to_hands()
        self.avatar.frame_ratio = _AVATAR_FRAME_RATIO
        lo = int(self.config.get("eval_trigger_step_min", self.EVAL_TRIGGER_STEP_MIN))
        hi = int(self.config.get("eval_trigger_step_max", self.EVAL_TRIGGER_STEP_MAX))
        if hi < lo:
            hi = lo
        self._eval_trigger_step = int(np.random.randint(lo, hi + 1))
        self._eval_frying_started = False
        self._eval_knock_done = False
        self._eval_knock_wait_steps = 0
        self._eval_policy_step_count = 0

    def eval_pre_policy_warmup(self):
        """Match the scripted data-collection start state for policy eval.

        ``play_once`` knocks the oil bottle over and lets it settle before the
        robot begins recovery.  Policy eval bypasses ``play_once``, so do the
        same physics setup here before the first policy observation.
        """
        if (
            self.avatar is None
            or not bool(self.config.get("eval_mode", False))
            or not bool(self.config.get("eval_avatar_pre_policy", True))
        ):
            return self.get_obs()

        if not getattr(self, "_eval_frying_started", False):
            self.avatar.play_animation("frying", return_to_idle_after=30)
            self._eval_frying_started = True

        if not getattr(self, "_eval_knock_done", False):
            self._wait_for_knock_trigger()
            self._knock_over_bottle()
            self._eval_knock_done = True
            self._eval_knock_wait_steps = 0

        for _ in range(int(self.config.get(
            "eval_knock_settle_steps", _KNOCK_SETTLE_STEPS,
        ))):
            self.step_sim()

        # Keep this counter policy-relative; warmup should not consume eval
        # policy steps or shift any downstream policy-step diagnostics.
        self._eval_policy_step_count = 0
        return self.get_obs()

    def _eval_step_avatar(self):
        self._eval_policy_step_count += 1
        if (not self._eval_frying_started and
                self._eval_policy_step_count >= self._eval_trigger_step):
            self.avatar.play_animation("frying", return_to_idle_after=30)
            self._eval_frying_started = True

        if not self._eval_frying_started or self._eval_knock_done:
            return

        self._eval_knock_wait_steps += 1
        hand_xy = self._right_hand_xy()
        bottle = self._get_entity_pose(
            self.oil_entity, Pose(np.zeros(3), _upright_q()),
        )
        d = float("inf") if hand_xy is None else float(
            np.linalg.norm(hand_xy - np.asarray(bottle.p[:2]))
        )
        timeout = self._eval_knock_wait_steps >= int(
            self.config.get("eval_knock_timeout_steps", self.EVAL_KNOCK_TIMEOUT_STEPS)
        )
        if d < _KNOCK_TRIGGER_DIST_M or timeout:
            self._knock_over_bottle()
            self._eval_knock_done = True

    # ------------------------------------------------------------------
    # Main rollout
    # ------------------------------------------------------------------

    def play_once(self) -> bool:
        if self.avatar is None:
            with self.suppress_recording():  # pre-task settle, excluded from recordings
                for _ in range(120):
                    self.step_sim()
            return False

        arm_tag = "right"
        arm = self.robot.get_arm(arm_tag)

        # --- Phase 1: ease into frying-start pose, snap pan/fork. ---
        self._attach_items_to_hands()

        # Brief settle so the bottle (loaded above the counter) lands.
        for _ in range(60):
            self.step_sim()

        # --- Phase 2: start the long shake-loop, wait for the trigger,
        # then knock the bottle over. ---
        self.avatar.frame_ratio = _AVATAR_FRAME_RATIO
        self.avatar.play_animation("frying", return_to_idle_after=30)

        self._wait_for_knock_trigger()
        self._knock_over_bottle()

        for _ in range(_KNOCK_SETTLE_STEPS):
            self.step_sim()

        # No post-knock set_quat / set_pos.  The eastward angular kick
        # (about world +y) carries the bottle past angle-of-repose; pure
        # physics then drops the long axis to world +x and the cylinder
        # rolls into a stable resting pose.  Earlier `set_quat` snaps
        # produced a visible discontinuity in the video and were only
        # needed because the prior knock direction (about -x, toward +y)
        # left the bottle in a metastable cap-prop equilibrium at 22°.

        fallen_pose = self._get_entity_pose(
            self.oil_entity, Pose(np.zeros(3), _upright_q()),
        )

        # --- Phase 3: pick the fallen bottle via the shared helper. ---
        # `BaseTask.select_and_execute_grasp` is the repo-standard robot
        # grasp path: curr→pre via RRT, pre→grasp via screw, FK-checked,
        # then executed with PD control. No robot-side attach/pin is used.
        self._refresh_planner_obstacles(arm_tag)
        self.open_gripper(arm_tag)

        # Boost arm-joint PD gains 4× before the pre-grasp move (kept
        # from prior iterations — fixes 15-18° steady-state error on
        # joints 1-3 under gravity load with bent-shoulder configs).
        try:
            saved_arm_kp = to_numpy(arm.entity.get_dofs_kp()).copy()
            saved_arm_kv = to_numpy(arm.entity.get_dofs_kv()).copy()
            new_kp = saved_arm_kp.copy()
            new_kv = saved_arm_kv.copy()
            from ..robot.franka_robot import _get_dof_idx as _dof_idx
            arm_idxs = []
            for j in arm.arm_joints:
                if j is None:
                    continue
                idx = _dof_idx(j)
                if idx is None or idx < 0 or idx >= len(new_kp):
                    continue
                arm_idxs.append(idx)
                new_kp[idx] = 16000.0
                new_kv[idx] = 800.0
            set_dofs_kp_kv_compat(arm.entity, kp=new_kp, kv=new_kv)
        except Exception:
            pass

        # Slow the avatar 4× during the grasp (still playing — user
        # directive — but moves less per sim step so its arm doesn't
        # sweep into the robot's planned path during execution).
        saved_frame_ratio = self.avatar.frame_ratio
        self.avatar.frame_ratio = saved_frame_ratio * 4

        grasp_choice = self.select_and_execute_grasp(
            object_name=_RECOVERY_OIL_ID,
            object_pose=fallen_pose,
            arm_tag=arm_tag,
            model_id=_RECOVERY_OIL_MODEL_ID,
            object_scale=self._oil_scale,
            categories=["pour_side_primary"],
            max_candidates=15,
            pre_dist=0.04,
        )
        if grasp_choice is None:
            return False
        grasp_link, _pre_link, _chosen_g = grasp_choice

        desired_tcp = grasp_link * arm.tcp_offset
        actual_tcp = Pose.from_pose7(arm.get_ee_pose())
        tcp_err = float(np.linalg.norm(actual_tcp.p - desired_tcp.p))
        if tcp_err > 0.015:
            self._refresh_planner_obstacles(arm_tag)
            refine = arm.planner.plan_screw_path(
                arm.get_arm_qpos(), grasp_link.to_pose7(),
            )
            if not refine.success:
                return False
            self.execute_plan(refine, arm_tag)
            for _ in range(30):
                self.step_sim()
            actual_tcp = Pose.from_pose7(arm.get_ee_pose())
            tcp_err = float(np.linalg.norm(actual_tcp.p - desired_tcp.p))

        # Soft, partial close.  Do this before the finger PD change so the
        # visible close is not a high-force full crush.
        self.set_gripper(_OIL_GRIP_CLOSE_TARGET, arm_tag, num_steps=120)
        for _ in range(30):
            self.step_sim()

        # Moderate per-DOF gripper PD: enough to hold the side grasp, but
        # far below the old 50k/1k full-crush setting.
        update_dofs_kp_kv_compat(
            arm.entity,
            arm._finger_dof_indices,
            kp_value=_OIL_FINGER_KP,
            kv_value=_OIL_FINGER_KV,
        )
        self.robot.set_gripper(_OIL_GRIP_CLOSE_TARGET, arm_tag)
        for _ in range(30):
            self.step_sim()

        # Clear the table before commanding the large rotation that rights the
        # bottle.  The prior single-shot reorientation could sweep the bottle
        # through the counter, so physics contact blocked the arm far from the
        # planned endpoint even though mplib found a valid robot-only path.
        lift_tcp0 = Pose.from_pose7(arm.get_ee_pose())
        lift_tcp_pose = Pose(
            lift_tcp0.p + np.array([0.0, 0.0, _PRE_UPRIGHT_LIFT_M]),
            lift_tcp0.q,
        )
        lift_link_pose = lift_tcp_pose * arm.tcp_offset.inv()
        self._refresh_planner_obstacles(arm_tag)
        lift_res = arm.planner.plan_screw_path(
            arm.get_arm_qpos(), lift_link_pose.to_pose7(),
        )
        if not lift_res.success:
            lift_res, _ = self._plan_path_with_fk_check(
                lift_link_pose, arm_tag, arm, fk_tol=0.008,
            )
        if lift_res is None or not lift_res.success:
            return False
        self.execute_plan(lift_res, arm_tag)
        if lift_res.position.size > 0:
            final_q = lift_res.position[-1]
            for _ in range(180):
                self.robot.set_arm_joints(final_q, arm_tag)
                self.robot.set_gripper(_OIL_GRIP_CLOSE_TARGET, arm_tag)
                self.step_sim()

        # Diagnostics: dump TCP pose + bottle pose at cache time so we can
        # debug bottle-frame target divergence (v0 produced wildly off
        # link targets — we don't yet trust _bottle_to_link in this task).
        ee_pose = Pose.from_pose7(arm.get_ee_pose())
        bottle_now = self._get_entity_pose(self.oil_entity, fallen_pose)
        self._T_ee_bottle = ee_pose.inv() * bottle_now

        # --- Phase 4: carry to the middle counter and place upright. ---

        def _link_pose_at_tcp(tcp_p, tcp_q) -> Pose:
            return (Pose(np.asarray(tcp_p, dtype=np.float64),
                         np.asarray(tcp_q, dtype=np.float64))
                    * arm.tcp_offset.inv())

        def _link_at_tcp(tcp_p, tcp_q):
            return _link_pose_at_tcp(tcp_p, tcp_q).to_pose7()

        # Compute the TCP pose that would put the bottle's own frame upright
        # at the middle-counter target.  This is pure motion planning; the
        # bottle remains held by contact, not attached to the gripper.
        upright_bottle_hover = Pose(
            np.array([
                self._safe_target_xy[0],
                self._safe_target_xy[1],
                self._oil_upright_origin_z_on_table + _UPRIGHT_PLACE_HOVER_M,
            ], dtype=np.float64),
            _upright_q(),
        )
        hover_tcp = upright_bottle_hover * self._T_ee_bottle.inv()
        hover_link_pose = hover_tcp * arm.tcp_offset.inv()
        self._refresh_planner_obstacles(arm_tag)
        hover_res, hover_err = self._plan_path_with_fk_check(
            hover_link_pose, arm_tag, arm, fk_tol=0.005,
        )
        if hover_res is None or hover_err > 0.05:
            return False
        self.execute_plan(hover_res, arm_tag)
        if hover_res.position.size > 0:
            final_q = hover_res.position[-1]
            for _ in range(_UPRIGHT_ENDPOINT_HOLD_STEPS):
                self.robot.set_arm_joints(final_q, arm_tag)
                self.robot.set_gripper(_OIL_GRIP_CLOSE_TARGET, arm_tag)
                self.step_sim()

        upright_bottle_place = Pose(
            np.array([
                self._safe_target_xy[0],
                self._safe_target_xy[1],
                self._oil_upright_origin_z_on_table,
            ], dtype=np.float64),
            _upright_q(),
        )
        place_tcp = upright_bottle_place * self._T_ee_bottle.inv()
        place_link_pose = place_tcp * arm.tcp_offset.inv()
        self._refresh_planner_obstacles(arm_tag)
        place_res = arm.planner.plan_screw_path(
            arm.get_arm_qpos(), place_link_pose.to_pose7(),
        )
        if not place_res.success:
            place_res, place_err = self._plan_path_with_fk_check(
                place_link_pose, arm_tag, arm, fk_tol=0.006,
            )
            if place_res is None or place_err > 0.05:
                return False
        self.execute_plan(place_res, arm_tag)
        if place_res.position.size > 0:
            final_q = place_res.position[-1]
            for _ in range(_UPRIGHT_ENDPOINT_HOLD_STEPS):
                self.robot.set_arm_joints(final_q, arm_tag)
                self.robot.set_gripper(_OIL_GRIP_CLOSE_TARGET, arm_tag)
                self.step_sim()

        self.open_gripper(arm_tag)
        for _ in range(30):
            self.step_sim()
        for _ in range(120):
            self.step_sim()

        # Retreat: small +z lift to clear the released bottle.
        ee_after = Pose.from_pose7(arm.get_ee_pose())
        retreat_tcp_p = ee_after.p + np.array([0.0, 0.0, 0.10])
        retreat_link = _link_at_tcp(retreat_tcp_p, ee_after.q)
        self._refresh_planner_obstacles(arm_tag)
        self.move_and_execute(retreat_link, arm_tag)

        # Restore avatar frame_ratio after the grasp phase.
        self.avatar.frame_ratio = saved_frame_ratio

        # --- Phase 6: short tail after recovery.  Do not drain the full
        # frying loop; the result is visible once the robot retreats.
        for _ in range(180):
            if self.avatar.spare():
                break
            self.step_sim()

        return True

    def check_success(self) -> bool:
        """Bottle ends upright inside the safe zone with no avatar collisions."""
        m = self._recovery_metrics()
        ok = bool(
            self.plan_success
            and m["target_dist_xy"] < _SUCCESS_DIST_XY_M
            and m["target_tilt_deg"] < _SUCCESS_TILT_DEG
            and m["avatar_ok"]
        )
        return ok

    def _recovery_metrics(self) -> dict:
        safe_xy = np.asarray(
            getattr(self, "_safe_target_xy", _SAFE_TARGET_XY),
            dtype=np.float64,
        )
        if self.oil_entity is None:
            pose = Pose(np.zeros(3), _upright_q())
        else:
            pose = self._get_entity_pose(
                self.oil_entity, Pose(np.zeros(3), _upright_q()),
            )
        R = t3d.quaternions.quat2mat(np.asarray(pose.q, dtype=np.float64))
        bottle_up_world = R[:, 1]
        cos_tilt = float(np.clip(bottle_up_world[2], -1.0, 1.0))
        tilt_deg = float(np.degrees(np.arccos(cos_tilt)))
        dxy = float(np.linalg.norm(np.asarray(pose.p[:2]) - safe_xy))
        dist_3d = float(np.linalg.norm(np.asarray(pose.p) - np.array([
            safe_xy[0],
            safe_xy[1],
            getattr(self, "_oil_upright_origin_z_on_table", self.TABLE_TOP_Z),
        ], dtype=np.float64)))
        dz = float(pose.p[2] - getattr(
            self, "_oil_upright_origin_z_on_table", self.TABLE_TOP_Z,
        ))
        avatar_ok = not getattr(self, "avatar_collided", False)
        return {
            "target_label": "recovered_oil_bottle",
            "target_pos": [
                float(safe_xy[0]),
                float(safe_xy[1]),
                float(getattr(self, "_oil_upright_origin_z_on_table", self.TABLE_TOP_Z)),
            ],
            "target_object_pos": np.asarray(pose.p, dtype=np.float64).tolist(),
            "target_dist_xy": dxy,
            "target_dist_3d": dist_3d,
            "target_dz": dz,
            "target_tilt_deg": tilt_deg,
            "target_dist_xy_threshold": _SUCCESS_DIST_XY_M,
            "target_tilt_deg_threshold": _SUCCESS_TILT_DEG,
            "avatar_ok": bool(avatar_ok),
            "eval_mode": bool(self.config.get("eval_mode", False)),
            "eval_trigger_step": getattr(self, "_eval_trigger_step", None),
            "eval_knock_done": bool(getattr(self, "_eval_knock_done", False)),
            "bottle_init_xy": list(getattr(self, "_bottle_init_xy", _BOTTLE_INIT_XY)),
            "safe_target_xy": safe_xy.tolist(),
        }

    def evaluate(self) -> dict:
        metrics = super().evaluate()
        metrics.update(self._recovery_metrics())
        return metrics


# ------------------------------------------------------------------
# Quaternion helpers (Pose convention: w, x, y, z)
# ------------------------------------------------------------------

def _quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product q1 ⊗ q2 in (w, x, y, z) convention."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], dtype=np.float64)
