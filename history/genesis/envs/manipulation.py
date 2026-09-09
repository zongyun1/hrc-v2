"""Top-down pick-and-place primitives shared across manipulation tasks.

Originally grown inside ``CategorizeCooperative`` over many iteration cycles
(fixes 1–13); extracted to a mixin so other tasks (``DeliverToHumanEasy``,
etc.) can share the same tuned grasp pipeline without forking.

Design:
  - ``TopDownPickPlaceMixin`` depends on methods that live on ``BaseTask``
    (``step_sim``, ``move_and_execute``, ``execute_plan``, ``open_gripper``,
    ``set_gripper``) and on the arm's ``planner`` (mplib wrapper).
  - ``pick_and_place`` takes ``PickSpec`` / ``PlaceSpec`` so task code stays
    declarative: "what to pick, where to drop".
"""

from dataclasses import dataclass
from typing import Callable, Optional
import json

import numpy as np
import transforms3d as t3d

from .utils import Pose, ASSETS_PATH, to_numpy
from .grasp import tcp_to_link_pose
from .genesis_compat import (
    update_dofs_force_range_compat,
    update_dofs_kp_kv_compat,
)


@dataclass
class PickSpec:
    """Describes what to pick.

    ``get_center`` is called multiple times during the pick (initial plan,
    re-read after pre-grasp in case approach bumped the object, and for debug
    logging after grip/lift).  Keep it cheap.

    ``radius`` is the cross-section half-extent used to size the partial
    gripper close (``radius * 0.7``) and the below-equator grasp offset.
    """

    get_center: Callable[[], np.ndarray]
    radius: float
    label: str = "object"
    close_value: Optional[float] = None
    # Optional live vertical half-extent.  Support-surface placement uses this
    # after transport, so rotated or irregular objects are released without
    # intersecting the receiving surface.
    get_half_height: Optional[Callable[[], float]] = None
    # Fine alignment for objects whose top-down jaw clearance is close to the
    # Franka's 8 cm maximum opening.  This offsets the commanded approach and
    # grasp XY while ``get_center`` continues to report true object position.
    approach_xy_offset: tuple[float, float] = (0.0, 0.0)
    # Optional live rotation of the top-down TCP about world Z.  This lets a
    # validated parallel-jaw grasp follow an object that a human rotated.
    get_grasp_yaw: Optional[Callable[[], float]] = None
    # Optional handover hook invoked after finger contact has settled and
    # immediately before lift (for example, detach a human kinematic hold).
    after_close: Optional[Callable[[], None]] = None
    # Positive values move the TCP grasp higher on tall objects so the finger
    # tips remain above the tabletop instead of pressing through it.
    grasp_z_offset: float = 0.0
    # Optional settle after commanding a fully open gripper, before approach.
    preopen_steps: int = 0
    # Strict expert-data gates.  They are opt-in for backward compatibility;
    # tasks that need a transported object should reject a planned motion when
    # the object never lifted or slipped away en route.
    require_lift: bool = False
    min_lift_delta: Optional[float] = None
    transport_retention_tolerance: Optional[float] = None
    # Reject a grasp when the object's center is implausibly far from the TCP
    # after the initial contact-building lift.  This catches handovers where
    # releasing a kinematic human constraint expels the object from between
    # the fingers even though the subsequent arm motion happens to raise it.
    max_grasp_offset: Optional[float] = None


@dataclass
class PlaceSpec:
    """Describes where to drop.

    ``pos`` is the world (x, y[, z]) of the drop target.

    ``transport_z`` is the height used for *lateral transit* (approach over
    the object, lift after grasp, transit to above the drop pose).  Default
    ``None`` → ``TABLE_TOP_Z + 0.38`` (clears basket rims for cate_coop).
    Tasks with no tall obstacles (e.g. deliver-to-hand) can lower this to
    reduce lift distance and stay within the robot's dexterous workspace.

    ``drop_from_z`` set  → hover at that height and release (basket pattern);
    no descent before opening the gripper.
    ``drop_from_z`` None → after transit at ``transport_z``, descend TCP to
    ``pos.z`` before release (plate / into-hand pattern).

    ``release`` False → do NOT open the gripper at the end; arm holds the
    object at ``pos`` (deliver-to-human pattern: arm extends, object stays
    in gripper near the human's palm).
    """

    pos: np.ndarray
    label: str = "place"
    transport_z: Optional[float] = None
    drop_from_z: Optional[float] = None
    # Object-aware alternative to ``drop_from_z``.  The helper measures the
    # held object-to-TCP offset after transport and chooses a TCP release height
    # that leaves the object's live lower bound above ``support_z`` by
    # ``support_clearance``.  Requires ``PickSpec.get_half_height``.
    support_z: Optional[float] = None
    support_clearance: float = 0.008
    release: bool = True
    # Grasp-only diagnostics can stop immediately after the vertical lift,
    # before any transport/placement motion. The caller may then hold and
    # record the lifted pose for an exact number of simulation steps.
    stop_after_lift: bool = False
    # When True, descend to the release height via the seeded IK + joint-space
    # RRT path (``_move_seeded``) instead of the straight Cartesian screw.
    # Needed for deep targets (e.g. dump's floor bin) where the vertical screw
    # crosses a wrist singularity and the fallback misplaces the object; the
    # seeded path preserves the branch from the above-target pose.  Default
    # False keeps the screw descent for table-height baskets (categorize).
    descent_seeded: bool = False


class TopDownPickPlaceMixin:
    """Top-down pick-and-place with seeded IK + plan_screw descent.

    Subclasses must also extend ``BaseTask``.  Expects the arm to carry a
    ``_cached_target`` attribute (populated lazily) and the planner to expose
    ``_mplib`` / ``_pad_qpos`` / ``_to_mplib_pose`` (the project's
    ``MPLibPlanner`` wrapper already does).
    """

    # Pure top-down gripper orientation — Franka 7-DOF has no singularity here.
    _R_DOWN = np.array([
        [1.0,  0.0,  0.0],
        [0.0, -1.0,  0.0],
        [0.0,  0.0, -1.0],
    ])

    @staticmethod
    def support_release_tcp_z(
        support_z: float,
        object_half_height: float,
        held_center_offset_z: float,
        clearance: float = 0.008,
    ) -> float:
        """TCP height that releases a held object just above a support plane."""
        return (
            float(support_z)
            + max(0.0, float(object_half_height))
            + max(0.0, float(clearance))
            - float(held_center_offset_z)
        )

    def configure_finger_pd(
        self,
        arm_tag: str,
        *,
        force_limit: float,
        kp: float = 9000.0,
        kv: float = 250.0,
    ) -> None:
        """Apply one symmetric contact/hold profile to parallel fingers."""
        arm = self.robot.get_arm(arm_tag)
        update_dofs_kp_kv_compat(
            arm.entity,
            arm._finger_dof_indices,
            kp_value=float(kp),
            kv_value=float(kv),
        )
        limit = abs(float(force_limit))
        update_dofs_force_range_compat(
            arm.entity,
            arm._finger_dof_indices,
            lower_value=-limit,
            upper_value=+limit,
        )

    @staticmethod
    def live_entity_half_height(entity) -> float:
        """Return the current world-Z half-extent from a Genesis entity AABB."""
        aabb = np.asarray(to_numpy(entity.get_AABB()), dtype=float)
        if aabb.shape != (2, 3):
            raise ValueError(f"expected entity AABB shape (2, 3), got {aabb.shape}")
        return max(0.0, 0.5 * float(aabb[1, 2] - aabb[0, 2]))

    @staticmethod
    def live_entity_aabb_center(entity) -> np.ndarray:
        """Return the live physical-bounds center used for contact targeting."""
        aabb = np.asarray(to_numpy(entity.get_AABB()), dtype=float)
        if aabb.shape != (2, 3):
            raise ValueError(f"expected entity AABB shape (2, 3), got {aabb.shape}")
        return 0.5 * (aabb[0] + aabb[1])

    @staticmethod
    def sync_support_proxy(proxy, support_link, support_z: float,
                           half_thickness: float = 0.003) -> None:
        """Align an invisible fixed support with a live articulated link."""
        if proxy is None or support_link is None:
            return
        aabb = np.asarray(to_numpy(support_link.get_AABB()), dtype=float)
        center = np.array([
            0.5 * (float(aabb[0, 0]) + float(aabb[1, 0])),
            0.5 * (float(aabb[0, 1]) + float(aabb[1, 1])),
            float(support_z) - max(0.0, float(half_thickness)),
        ])
        proxy.set_pos(center)

    @staticmethod
    def topdown_pick_options(object_spec: dict) -> dict:
        """Translate catalog grasp hints into reusable ``PickSpec`` options.

        The ``topdown_*`` keys are controller-specific overrides. Generic
        catalog hints remain the fallback, so top-down tasks share one set of
        precedence rules instead of reimplementing them per task.
        """
        spec = dict(object_spec or {})
        return {
            "close_value": spec.get(
                "topdown_close_value", spec.get("close_value"),
            ),
            "grasp_z_offset": float(spec.get(
                "topdown_grasp_z_offset", spec.get("grasp_z_offset", 0.0),
            )),
            "approach_xy_offset": tuple(spec.get(
                "topdown_approach_xy_offset", (0.0, 0.0),
            )),
            "preopen_steps": int(spec.get("topdown_preopen_steps", 0)),
        }

    @staticmethod
    def object_aware_prismatic_open_target(
        entity,
        dof_index: int,
        payload_extent: float,
        *,
        minimum_open: float = 0.075,
        clearance: float = 0.030,
        limit_margin: float = 0.004,
    ) -> float:
        """Choose a fixture opening that exposes the payload plus clearance.

        The desired opening is bounded by the articulation's live DOF limits,
        making the rule reusable for drawers and sliding bins of other scales.
        """
        desired = max(
            float(minimum_open),
            max(0.0, float(payload_extent)) + max(0.0, float(clearance)),
        )
        try:
            limits = entity.get_dofs_limit()
            if isinstance(limits, (tuple, list)) and len(limits) == 2:
                lower = np.asarray(to_numpy(limits[0]), dtype=float).reshape(-1)
                upper = np.asarray(to_numpy(limits[1]), dtype=float).reshape(-1)
            else:
                arr = np.asarray(to_numpy(limits), dtype=float)
                if arr.ndim != 2 or 2 not in arr.shape:
                    raise ValueError(f"unsupported DOF-limit shape {arr.shape}")
                if arr.shape[0] == 2:
                    lower, upper = arr[0].reshape(-1), arr[1].reshape(-1)
                else:
                    lower, upper = arr[:, 0].reshape(-1), arr[:, 1].reshape(-1)
            idx = int(dof_index)
            lo = float(lower[idx]) + max(0.0, float(limit_margin))
            hi = float(upper[idx]) - max(0.0, float(limit_margin))
            if hi >= lo:
                return float(np.clip(desired, lo, hi))
        except Exception:
            pass
        return desired

    @staticmethod
    def contained_center_from_edge(
        edge: float,
        object_extent: float,
        *,
        inward_sign: float = -1.0,
        clearance: float = 0.010,
        held_inward_offset: float = 0.0,
    ) -> float:
        """Place a held object's full extent inward of a container edge.

        ``held_inward_offset`` is the conservative portion of the measured or
        expected object-to-TCP offset that already points into the container.
        """
        sign = -1.0 if float(inward_sign) < 0.0 else 1.0
        inset = max(
            0.0,
            0.5 * max(0.0, float(object_extent))
            + max(0.0, float(clearance))
            - max(0.0, float(held_inward_offset)),
        )
        return float(edge) + sign * inset

    @staticmethod
    def relative_parallel_jaw_yaw(current_quat, reference_quat) -> float:
        """Shortest yaw that preserves a reference parallel-jaw alignment."""
        current_R = t3d.quaternions.quat2mat(
            np.asarray(current_quat, dtype=float),
        )
        reference_R = t3d.quaternions.quat2mat(
            np.asarray(reference_quat, dtype=float),
        )
        relative_R = current_R @ reference_R.T
        yaw = float(np.arctan2(relative_R[1, 0], relative_R[0, 0]))
        # Parallel jaws are equivalent after 180 degrees.  Keep the requested
        # wrist turn in the closest half-turn to reduce IK branch changes.
        return float((yaw + 0.5 * np.pi) % np.pi - 0.5 * np.pi)

    # ------------------------------------------------------------------
    # Object geometry helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _get_object_cross_section_radius(object_name: str, model_id: int = 0) -> float:
        """Cross-section half-extent (smallest horizontal dim) for top-down grasps."""
        md_path = ASSETS_PATH / "objects" / object_name / f"model_data{model_id}.json"
        with open(md_path) as f:
            md = json.load(f)
        extents = np.array(md["extents"])
        scale = np.array(md.get("scale", [1, 1, 1]))
        return float(np.min(extents * scale) / 2)

    @staticmethod
    def _get_model_center(object_name: str, model_id: int = 0) -> np.ndarray:
        """Model-space centre offset (scaled) — see ``project_asset_mesh_origin``."""
        md_path = ASSETS_PATH / "objects" / object_name / f"model_data{model_id}.json"
        with open(md_path) as f:
            md = json.load(f)
        center = np.array(md["center"])
        scale = np.array(md.get("scale", [1, 1, 1]))
        return center * scale

    def _get_object_world_center(self, actor, object_name: str, model_id: int = 0) -> np.ndarray:
        """World-frame geometric centre.  ``actor.get_pose().p`` returns the
        mesh origin which may be offset from the bbox centre — rotate the
        model-space centre into world frame and add.
        """
        model_center = self._get_model_center(object_name, model_id)
        pose = actor.get_pose()
        R = t3d.quaternions.quat2mat(np.array(pose.q))
        return np.array(pose.p) + R @ model_center

    # ------------------------------------------------------------------
    # TCP / IK helpers
    # ------------------------------------------------------------------
    def _top_down_tcp(self, pos, yaw: float = 0.0) -> Pose:
        Rz = t3d.euler.euler2mat(0.0, 0.0, float(yaw), axes="sxyz")
        q = t3d.quaternions.mat2quat(Rz @ self._R_DOWN)
        return Pose(np.asarray(pos, dtype=float), q)

    def _solve_ik(self, tcp_pose, arm_tag):
        """Solve IK for a TCP pose, return arm joint positions or None.

        Seeds ``init_qpos`` from the PREVIOUSLY COMMANDED pose (``arm._cached_target``)
        when available, else from the current measured qpos.  Seeding from the
        measured qpos while the arm is mid-motion lets PD lag propagate into
        the IK search, which can drag the solver across a homotopy boundary
        and flip the wrist 360° — that's the "arm rotates and throws the
        object" failure mode.
        """
        arm = self.robot.get_arm(arm_tag)
        tcp_offset = arm.tcp_offset
        link_pose = tcp_to_link_pose(tcp_pose, tcp_offset)
        pose7 = link_pose.to_pose7()

        ik_fn = getattr(arm.entity, "inverse_kinematics", None)
        if ik_fn is None:
            return None
        try:
            from .robot.franka_robot import to_numpy
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
            print(f"[pnp] IK exception: {e}")
            return None

    # ------------------------------------------------------------------
    # Motion primitives
    # ------------------------------------------------------------------
    def _move_cartesian(
        self, start_pos, end_pos, arm_tag, n_steps=50, sim_per_step=20,
        tcp_yaw: float = 0.0,
    ):
        """Slow Cartesian interpolation — IK at each waypoint, PD controlled.

        Much smoother than the RRT motion planner, keeping grip forces stable
        during transport.
        """
        arm = self.robot.get_arm(arm_tag)
        from .robot.franka_robot import to_numpy, _get_dof_idx
        # ``base_target`` preserves FINGER grip from the last commanded target —
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
            tcp = self._top_down_tcp(pos, yaw=tcp_yaw)
            arm_qpos = self._solve_ik(tcp, arm_tag)
            if arm_qpos is None:
                print(f"[pnp] Cartesian IK failed at step {i}/{n_steps}")
                continue
            qpos_full = base_target.copy()
            for j_idx, j in enumerate(arm.arm_joints):
                if j is not None:
                    dof_idx = _get_dof_idx(j)
                    if dof_idx is not None:
                        qpos_full[dof_idx] = float(arm_qpos[j_idx])
            arm.entity.control_dofs_position(qpos_full)
            # Refresh IK seed to THIS waypoint's solution BEFORE next waypoint
            # solves IK — keeps next step's IK in same homotopy class.
            arm._cached_target = qpos_full.copy()
            for _ in range(sim_per_step):
                self.step_sim()

    def _move_seeded(self, link_pose7, arm_tag):
        """RRT-plan with the goal qpos pre-computed by our seeded IK.

        MPLib's ``plan_pose`` (what ``move_and_execute`` uses) runs its own
        IK internally, ignoring our ``_cached_target`` seed.  That's how
        the pre-grasp step lands in a wrist-wrapped IK branch that later
        Cartesian steps then continue from.  Solving IK ourselves with the
        seed and feeding the result into ``plan_qpos`` (joint-space RRT goal)
        keeps the plan continuous with the prior motion's branch.
        """
        from .robot.franka_robot import to_numpy as _to_numpy
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
            print(f"[pnp] _move_seeded IK exc: {e}, fallback")
            return self.move_and_execute(link_pose7, arm_tag)
        if qpos_sol is None:
            return self.move_and_execute(link_pose7, arm_tag)
        arm_qpos_goal = _to_numpy(qpos_sol).ravel()[:arm.n_arm]

        planner = arm.planner
        goal_padded = planner._pad_qpos(np.asarray(arm_qpos_goal, dtype=np.float64))
        current_padded = planner._pad_qpos(np.asarray(arm.get_arm_qpos(), dtype=np.float64))
        try:
            result = planner._mplib.plan_qpos(
                goal_qposes=[goal_padded],
                current_qpos=current_padded,
                time_step=1 / 250,
                planning_time=getattr(planner, "_timeout", None) or 4,
                rrt_range=0.3,
                verbose=False,
            )
        except Exception as e:
            print(f"[pnp] plan_qpos exc: {e}, fallback to plan_pose")
            return self.move_and_execute(link_pose7, arm_tag)
        if result and result.get("status") == "Success":
            from .planning.base import PlanResult
            pos = result["position"]
            vel = result.get("velocity", np.zeros_like(pos))
            plan = PlanResult(True, pos, vel)
            self.execute_plan(plan, arm_tag)
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
                print(f"[pnp] _move_seeded post-exec xy_err={xy_err:.3f}m, retry plan_pose")
                return self.move_and_execute(link_pose7, arm_tag)
            return result
        print(f"[pnp] plan_qpos failed, fallback to plan_pose")
        return self.move_and_execute(link_pose7, arm_tag)

    def _move_screw(self, target_pos, arm_tag, tcp_yaw: float = 0.0):
        """Straight-line Cartesian motion via MPLib plan_screw; holds gripper.

        Retries plan_screw with decreasing qpos_step (0.1, 0.05, 0.02) — finer
        resolution lets MPLib solve the IK branch continuity constraint on
        waypoints that failed at coarser step sizes.  Only falls back to
        ``plan_pose`` if all three resolutions fail — plan_pose RRT has been
        observed to swing laterally by 2-16 cm on descents, knocking target
        objects (measured on cate_coop seeds 30, 60).
        """
        arm = self.robot.get_arm(arm_tag)
        tcp_offset = arm.tcp_offset
        tcp = self._top_down_tcp(target_pos, yaw=tcp_yaw)
        link_pose = tcp_to_link_pose(tcp, tcp_offset)
        pose7 = link_pose.to_pose7()
        planner = arm.planner

        # Non-mplib planners (e.g. GenesisIKPlanner) have no `_mplib` backend.
        # Use their own Cartesian straight-line solver — `plan_screw_path`
        # returns a PlanResult directly — and fall back to the seeded
        # joint-space plan if it can't hold the screw.
        if not hasattr(planner, "_mplib"):
            screw_fn = getattr(planner, "plan_screw_path", None)
            res = None
            if screw_fn is not None:
                try:
                    res = screw_fn(arm.get_arm_qpos(), pose7, log=False)
                except Exception as e:
                    print(f"[pnp] plan_screw_path exc: {e}")
            if res is not None and res.success and res.position.size > 0:
                self.execute_plan(res, arm_tag)
                return True
            print("[pnp] plan_screw_path unavailable/failed, using seeded fallback")
            self._move_seeded(pose7, arm_tag)
            return

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
                print(f"[pnp] plan_screw qpos_step={qpos_step} exc: {e}")
                continue
            if result and result.get("status") == "Success":
                from .planning.base import PlanResult
                pos = result["position"]
                vel = result.get("velocity", np.zeros_like(pos))
                if qpos_step != 0.1:
                    print(f"[pnp] plan_screw succeeded at qpos_step={qpos_step}")
                self.execute_plan(PlanResult(True, pos, vel), arm_tag)
                return True
        # Route fallback through `_move_seeded` so the IK is seeded from the
        # previously commanded qpos (`_cached_target`) instead of MPLib's
        # unseeded multi-start.  Unseeded `plan_pose` was observed to flip
        # the wrist 180° on cate-coop seed-8 seal descent (target xy
        # [0.009,−0.079] → actual [−0.022, 0.061], 15.6 cm off).
        # `_move_seeded` itself falls through to `move_and_execute` if
        # `plan_qpos` fails, so the bottom of the chain is unchanged.
        print(f"[pnp] plan_screw all qpos_steps failed, using seeded fallback")
        self._move_seeded(pose7, arm_tag)
        return True

    # ------------------------------------------------------------------
    # The big one
    # ------------------------------------------------------------------
    def pick_and_place(self, pick: PickSpec, place: PlaceSpec, arm_tag: str) -> bool:
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
        # location; feeding it to Genesis IK for the NEW pre-grasp target
        # can lock onto a bad local minimum where FK doesn't match the
        # target (seen on cate_coop seed 20 — 28 cm err, never touched the
        # object).  Measured qpos is always a valid starting config.
        from .robot.franka_robot import to_numpy as _to_numpy
        arm._cached_target = np.asarray(
            _to_numpy(arm.entity.get_qpos()), dtype=float
        ).ravel().copy()

        obj_center = pick.get_center()
        obj_radius = pick.radius
        approach_xy_offset = np.asarray(pick.approach_xy_offset, dtype=float)
        if approach_xy_offset.shape != (2,):
            raise ValueError("PickSpec.approach_xy_offset must contain exactly two values")
        grasp_yaw = (
            float(pick.get_grasp_yaw())
            if pick.get_grasp_yaw is not None
            else 0.0
        )
        oriented_grasp = abs(grasp_yaw) > 1e-8

        def _pick_tcp(pos):
            if not oriented_grasp:
                return self._top_down_tcp(pos)
            return TopDownPickPlaceMixin._top_down_tcp(
                self, pos, yaw=grasp_yaw,
            )

        def _pick_screw(pos):
            if not oriented_grasp:
                return self._move_screw(pos, arm_tag)
            return TopDownPickPlaceMixin._move_screw(
                self, pos, arm_tag, tcp_yaw=grasp_yaw,
            )

        print(f"[pnp] picking {pick.label} -> {place.label}, "
              f"center={obj_center}, radius={obj_radius:.4f}, "
              f"yaw={np.degrees(grasp_yaw):+.1f}deg")

        # Grasp below object center — below-equator technique pushes object
        # UP into gripper palm when fingers close on lower hemisphere.
        grasp_pos = obj_center.copy()
        grasp_pos[2] -= obj_radius * 0.3  # 30% of radius below center
        grasp_pos[2] += float(pick.grasp_z_offset)
        min_z = self.TABLE_TOP_Z + 0.002
        grasp_pos[2] = max(grasp_pos[2], min_z)

        # Transport height for all lateral moves.  Default clears basket rims
        # (baskets 28cm tall; top at TABLE_TOP_Z + 30cm).  Tasks without tall
        # obstacles can lower this via ``place.transport_z``.
        if place.transport_z is not None:
            safe_z = place.transport_z
        else:
            safe_z = self.TABLE_TOP_Z + 0.38

        # Open gripper
        self.open_gripper(arm_tag)
        for _ in range(max(0, int(pick.preopen_steps))):
            self.step_sim()

        # Two-phase approach to avoid sweeping other objects:
        #   Phase A: lateral transit at safe_z (well above any table object).
        #   Phase B: vertical descent from safe_z to pre-grasp (15 cm above obj).
        above_pos = np.array([obj_center[0], obj_center[1], safe_z])
        above_pos[:2] += approach_xy_offset
        above_link = tcp_to_link_pose(
            _pick_tcp(above_pos), tcp_offset,
        )
        if self._move_seeded(above_link.to_pose7(), arm_tag) is None:
            return False

        pre_pos = obj_center.copy()
        pre_pos[:2] += approach_xy_offset
        pre_pos[2] += 0.15
        # Vertical-only descent from safe_z to pre-grasp — plan_screw should
        # succeed trivially for a straight-down motion at fixed xy.
        _pick_screw(pre_pos)

        # Re-read object center after MPLib in case the pre-grasp motion
        # bumped it.
        obj_center_now = pick.get_center()
        grasp_pos[:2] = obj_center_now[:2] + approach_xy_offset

        # Descent from pre-grasp straight down to grasp pose.
        _pick_screw(grasp_pos)

        # Settle at grasp pose so arm reaches zero velocity before closing.
        for _ in range(100):
            self.step_sim()

        # Close the loop on the reached pose before the one and only finger
        # close. IK branches track the last Cartesian descent differently, so
        # the live error here is more reliable than a fixed catalog bias or a
        # pre-grasp estimate. This is a small pose refinement, not a re-grasp.
        ee_pos = np.asarray(arm.get_ee_pose()[:3], dtype=float)
        desired_grasp_xy = pick.get_center()[:2] + approach_xy_offset
        alignment_delta = desired_grasp_xy - ee_pos[:2]
        if float(np.linalg.norm(alignment_delta)) > 0.002:
            corrected_grasp_pos = grasp_pos.copy()
            corrected_grasp_pos[:2] += alignment_delta
            _pick_screw(corrected_grasp_pos)
            grasp_pos = corrected_grasp_pos
            for _ in range(100):
                self.step_sim()
            corrected_ee = np.asarray(arm.get_ee_pose()[:3], dtype=float)
            residual = desired_grasp_xy - corrected_ee[:2]
            print(
                f"[pnp]   pre-close xy correction={alignment_delta}, "
                f"residual={residual}"
            )

        # Verify position
        ee_pos = np.array(arm.get_ee_pose()[:3])
        print(f"[pnp]   grasp target={grasp_pos}, actual_ee={ee_pos}, "
              f"err={np.linalg.norm(ee_pos - grasp_pos):.4f}m")

        # If Cartesian descent landed the EE far from the grasp target in
        # XY, Genesis IK at each waypoint converged to a bad branch.  Refine
        # by snapping to the top-down link pose via MPLib plan_pose
        # (internal multi-start IK) and re-settling.
        xy_err = float(np.linalg.norm(ee_pos[:2] - grasp_pos[:2]))
        if xy_err > 0.03:
            # Route the descent refinement through `_move_seeded` for the same
            # IK-flip reason as the `_move_screw` fallback above.
            print(f"[pnp]   descent xy_err={xy_err:.3f}m > 3cm; refining via seeded plan")
            grasp_link = tcp_to_link_pose(
                _pick_tcp(grasp_pos), tcp_offset,
            )
            self._move_seeded(grasp_link.to_pose7(), arm_tag)
            for _ in range(100):
                self.step_sim()
            ee_pos = np.array(arm.get_ee_pose()[:3])
            print(f"[pnp]   refined ee={ee_pos}, err={np.linalg.norm(ee_pos - grasp_pos):.4f}m")

        # Adaptive close target.  Round/small meshes (r<3 cm, e.g. apple) need
        # a shallow squeeze (`r-0.005`) — deeper commands let fingers overpower
        # mesh contact before caging, shoving the object sideways.  Flat/large
        # meshes (r>=3 cm, e.g. wooden block) need a deep squeeze (`r*0.7`) —
        # fingers contact a flat face squarely, the block wins the PD push and
        # stops fingers at the face with ~100 N firm grip.  Verified: deliver
        # apple → 4/4 with shallow; cate-coop block → 3/3 with deep.
        if pick.close_value is not None:
            gripper_close = float(np.clip(pick.close_value, 0.0, 1.0))
        else:
            r = float(obj_radius)
            if r < 0.030:
                close_target = max(0.010, r - 0.005)
            else:
                close_target = max(0.010, r * 0.7)
            gripper_close = close_target / 0.04
        self.set_gripper(gripper_close, arm_tag)
        for _ in range(200):
            self.step_sim()
        if pick.after_close is not None:
            pick.after_close()
            for _ in range(40):
                self.step_sim()

        lift_start = grasp_pos.copy()
        lift_mid = grasp_pos.copy()
        lift_mid[2] = obj_center[2] + obj_radius  # lift to just above object top
        if oriented_grasp:
            TopDownPickPlaceMixin._move_cartesian(
                self, lift_start, lift_mid, arm_tag,
                n_steps=30, sim_per_step=30, tcp_yaw=grasp_yaw,
            )
        else:
            self._move_cartesian(
                lift_start, lift_mid, arm_tag,
                n_steps=30, sim_per_step=30,
            )

        # Debug: check grip
        from .robot.franka_robot import _get_dof_idx, to_numpy
        qpos_after = to_numpy(arm.entity.get_qpos())
        finger_qpos = []
        for j in arm.finger_joints:
            if j is not None:
                idx = _get_dof_idx(j)
                if idx is not None:
                    finger_qpos.append(float(qpos_after[idx]))
        obj_pos_after = pick.get_center()
        print(f"[pnp]   fingers={finger_qpos}, obj={obj_pos_after}")

        if pick.max_grasp_offset is not None:
            ee_pos_after = np.asarray(arm.get_ee_pose()[:3], dtype=float)
            grasp_offset = float(np.linalg.norm(
                np.asarray(obj_pos_after, dtype=float) - ee_pos_after,
            ))
            max_grasp_offset = float(pick.max_grasp_offset)
            if grasp_offset > max_grasp_offset:
                print(
                    f"[pnp]   grasp retention failed: object/TCP offset "
                    f"{grasp_offset:.4f}m > {max_grasp_offset:.4f}m"
                )
                return False

        # Lift to safe height via screw motion (straight-line, gripper held)
        lift_pos = np.array([grasp_pos[0], grasp_pos[1], safe_z])
        _pick_screw(lift_pos)

        obj_pos_lifted = pick.get_center()
        lift_delta = float(obj_pos_lifted[2] - obj_pos_after[2])
        print(f"[pnp]   after lift: {obj_pos_lifted} (dz={lift_delta:.4f})")
        if pick.require_lift:
            min_lift = (
                float(pick.min_lift_delta)
                if pick.min_lift_delta is not None
                else max(0.03, 0.5 * float(obj_radius))
            )
            if lift_delta < min_lift:
                print(
                    f"[pnp]   lift retention failed: dz={lift_delta:.4f}m "
                    f"< {min_lift:.4f}m"
                )
                return False

        ee_pos_lifted = np.asarray(arm.get_ee_pose()[:3], dtype=float)
        held_offset_lift = np.asarray(obj_pos_lifted, dtype=float) - ee_pos_lifted

        if place.stop_after_lift:
            return True

        # Transport to above drop target at safe_z.
        target_tcp_xy = np.asarray(place.pos[:2], dtype=float).copy()
        place_xyz = np.array([
            target_tcp_xy[0], target_tcp_xy[1], safe_z,
        ], dtype=float)
        _pick_screw(place_xyz)

        # Settle so the object stops swinging in the gripper.
        place_settle_steps = int(getattr(
            self, "PICK_PLACE_SETTLE_ABOVE_PLACE_STEPS",
            getattr(self, "config", {}).get("pick_place_settle_above_place_steps", 100),
        ))
        for _ in range(max(0, place_settle_steps)):
            self.step_sim()

        obj_pos_transport = pick.get_center()
        print(f"[pnp]   above place: tcp={place_xyz}, obj={obj_pos_transport}")
        ee_pos_transport = np.asarray(arm.get_ee_pose()[:3], dtype=float)
        held_offset_transport = (
            np.asarray(obj_pos_transport, dtype=float) - ee_pos_transport
        )
        if pick.transport_retention_tolerance is not None:
            retention_error = float(np.linalg.norm(
                held_offset_transport - held_offset_lift,
            ))
            tolerance = float(pick.transport_retention_tolerance)
            if retention_error > tolerance:
                print(
                    f"[pnp]   transport retention failed: offset change "
                    f"{retention_error:.4f}m > {tolerance:.4f}m"
                )
                return False
        # Descend to release height.  ``drop_from_z`` overrides place.pos[2]
        # (hover-release pattern); otherwise descend to place.pos[2].  A
        # single vertical screw — multi-segment descents have been observed
        # to flip the wrist on cate_coop and hurl the object.
        if place.support_z is not None:
            if pick.get_half_height is None:
                raise ValueError(
                    "PlaceSpec.support_z requires PickSpec.get_half_height"
                )
            object_half_height = float(pick.get_half_height())
            release_z = self.support_release_tcp_z(
                place.support_z,
                object_half_height,
                held_offset_transport[2],
                place.support_clearance,
            )
            release_z = min(float(safe_z), release_z)
            print(
                f"[pnp]   support release: support_z={place.support_z:.4f}, "
                f"half_h={object_half_height:.4f}, "
                f"held_dz={held_offset_transport[2]:+.4f}, "
                f"tcp_z={release_z:.4f}"
            )
        elif place.drop_from_z is not None:
            release_z = float(place.drop_from_z)
        elif len(place.pos) >= 3:
            release_z = float(place.pos[2])
        else:
            release_z = safe_z  # no explicit z → release at transit height
        if abs(release_z - safe_z) > 1e-4:
            descend_pos = np.array([
                target_tcp_xy[0], target_tcp_xy[1], release_z,
            ])
            if getattr(place, "descent_seeded", False):
                # Seeded IK + RRT descent (preserves the wrist branch from the
                # above-target pose) for deep targets where the screw fails.
                descend_link = tcp_to_link_pose(
                    _pick_tcp(descend_pos), tcp_offset,
                )
                self._move_seeded(descend_link.to_pose7(), arm_tag)
            else:
                _pick_screw(descend_pos)
            for _ in range(50):
                self.step_sim()

        if place.release:
            # Release.  Let the object drop under gravity.
            self.open_gripper(arm_tag)
            release_open_steps = int(getattr(
                self, "PICK_PLACE_RELEASE_OPEN_STEPS",
                getattr(self, "config", {}).get("pick_place_release_open_steps", 40),
            ))
            for _ in range(max(0, release_open_steps)):
                self.step_sim()
            after_release = getattr(self, "_after_pick_place_release", None)
            if after_release is not None:
                release_pos = np.array(
                    [place.pos[0], place.pos[1], release_z],
                    dtype=float,
                )
                after_release(pick, place, arm_tag, release_pos)
            release_settle_steps = int(getattr(
                self, "PICK_PLACE_RELEASE_SETTLE_STEPS",
                getattr(self, "config", {}).get("pick_place_release_settle_steps", 210),
            ))
            for _ in range(max(0, release_settle_steps)):
                self.step_sim()
            obj_pos_release = pick.get_center()
            print(f"[pnp]   after release: {obj_pos_release}")
        else:
            # Hold object in gripper at deliver pose (deliver-to-human).
            hold_steps = int(getattr(
                self, "PICK_PLACE_HOLD_STEPS",
                getattr(self, "config", {}).get("pick_place_hold_steps", 100),
            ))
            for _ in range(max(0, hold_steps)):
                self.step_sim()
            obj_pos_held = pick.get_center()
            print(f"[pnp]   holding (no release): {obj_pos_held}")

        return True
