"""Categorize interrupt: human inspects an object the robot is about to grasp."""

import numpy as np
import transforms3d as t3d

from ..avatar.inspect_motion_mixin import InspectMotionMixin, STANDARD_INSPECT_POOL
from ..genesis_compat import set_dofs_kp_kv_compat
from ..utils import Pose, load_object, create_primitive, to_numpy, place_decorative_objects
from ..grasp import tcp_to_link_pose
from .categorize_cooperative import CategorizeCooperative, PickSpec, PlaceSpec
from ..object_catalog import object_set_pairs


class CategorizeInterrupt(InspectMotionMixin, CategorizeCooperative):
    """Robot categorizes objects, but must yield when the human inspects one.

    Inherits cooperative's robust pick/place (Cartesian descent + screw motion,
    lid opening, below-equator grasp) and its box containers (100194 cardboard
    box).  Overrides layout, adds Inspect-motion interruption phase.

    Layout (+Y points from robot toward avatar):
      robot (y=-0.5) | boxes (y=-0.20) | objects (y=0.10) | avatar (y=1.2)
    The avatar leans over the middle object and inspects it while the robot is
    still approaching — the robot detects this, retreats, sorts the other two
    objects first, and comes back for the interrupted one after the avatar
    releases it.
    """

    INSTRUCTION = "sort the objects into the matching baskets"

    # Re-enable the mplib obstacle-inflation that the parent disables —
    # row 2 cate-inte agent validated this end-to-end (10/10 PASS) on
    # cate-inte's column-aligned transits.
    _USE_OBSTACLE_INFLATION = True

    # The interrupt layout is already column-aligned, and the preferred
    # standard-helper candidates validated 5/5 on seeds 1-5.
    PREFERRED_GRASP_NAMES = {
        "035_apple": (
            "manual_center_p0_yaw0",
            "manual_center_p0_yaw2",
            "manual_underhand_p0_yaw0",
            "manual_underhand_p0_yaw2",
        ),
        "086_woodenblock": (
            "manual_topdown_yaw45_center",
            "manual_topdown_yaw45_higher",
        ),
        "100_seal": (
            "manual_topdown_X_center_m2",
            "manual_topdown_Y_center_m2",
            "manual_grasp_015_yaw90_deeper_upper",
        ),
    }

    # Table extended in −Y so the robot base (at y=-0.5) sits on the tabletop
    # instead of floating behind it.  Front (+Y, avatar-side) edge unchanged at
    # y=+0.35 so the avatar still has room to lean in during Inspect.
    #   back edge = dy − half_y = -0.15 − 0.50 = -0.65  (robot at y=-0.5 sits 0.15 m inboard)
    #   front edge = dy + half_y = -0.15 + 0.50 = +0.35
    table_offset = np.array([0.0, -0.15])
    TABLE_HALF_SIZE_Y = 0.50
    TABLE_LEGS_HALF_Y = 0.45

    # Avatar root sits ~0.18m below the mesh feet in cooperative. The final
    # per-episode root pose is shifted by
    # `_calibrate_avatar_for_inspect` so the Inspect palm reaches the selected
    # object.
    avatar_init_pos = np.array([0.0, 0.70, -0.18])

    # Extra Z lift applied to the avatar on top of the palm-meets-object shift
    # computed by `_calibrate_avatar_for_inspect`.  Keeps palm just above the
    # resting object at attach instead of sunk into the table.
    AVATAR_EXTRA_Z = 0.06

    # Legacy fallback only.  By default the interrupt target is sampled from
    # the scripted robot pick order so the human interrupts whichever ordinal
    # pick the robot is about to reach for.
    INTERRUPTED_IDX = 1

    # Put robot-side loose objects closer to the avatar-side table edge than
    # the original y=0.10 layout.  Calibration shifts the avatar root so palm
    # meets this object at attach; moving the target band toward +Y keeps the
    # avatar body out of the tabletop while preserving robot reach.
    INTERRUPT_OBJECT_Y = 0.18

    # Motion slowdown factor: resample Mixamo frames to play 10x slower so
    # the Inspect motion reads as a deliberate pause in the final video.
    AVATAR_MOTION_SLOW = 10.0

    # Inspect motion timing (after clipping frames 13..225 of original).
    # Original F47 -> clipped index 34 (attach), F200 -> clipped 187 (detach).
    # These are in the ORIGINAL (frame_ratio=1.0) frame space; they get
    # scaled by AVATAR_MOTION_SLOW at runtime.
    INSPECT_MOTION_NAME = "Inspect1"
    INSPECT_ATTACH_FRAME = 34
    INSPECT_DETACH_FRAME = 187
    INSPECT_POOL = STANDARD_INSPECT_POOL

    # Retreat is empty-handed and only needs to clear the avatar's workspace,
    # so it can use a faster replay than object transport/grasp motions.
    RETREAT_REPLAY_SUBSTEPS = 1

    # VLA primary view: near top-down over the lifted 1.60 x 1.00 m table.
    static_camera_list = [{
        "name": "head_camera",
        "position": [0.0, -0.145, 2.85],
        "forward": [0.0, -0.005, -1.35],
    }]

    # Side view of table and avatar inspecting object.
    recording_camera_pos = [1.7, 0.05, 1.85]
    recording_camera_lookat = [0.0, 0.0, 1.55]
    # Top-down secondary, nudged toward the robot-arm side (-Y).
    side_camera_pos = [0.0, -0.25, 2.7]
    side_camera_lookat = [0.0, -0.25, 1.30]

    # Unified with the cooperative base: objects sampled per-episode from the
    # shared pool (NUM_CATEGORIES distinct draws).  Inherits OBJECT_SET,
    # NUM_CATEGORIES, and the default CATEGORIES from CategorizeCooperative.

    def _scripted_pick_order(self):
        """Object indices in the order the robot will attempt them."""
        if hasattr(self, "_sort_targets"):
            n_targets = len(self._sort_targets)
        else:
            n_targets = len(self.CATEGORIES)
        return sorted(range(n_targets), reverse=True)

    def _boost_arm_pd(self):
        """Same as parent, but stronger finger grip to hold the apple through
        the screw-motion transport.  kp=800 (parent) gave ~8N per finger —
        not enough for a round 5cm apple that has just fallen from the avatar's
        hand and may be on a slight tilt.  kp=2500 gives ~25N which holds
        during transport without the mesh-penetration seen at kp=15000.
        """
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
        for j in arm.finger_joints:
            if j is not None:
                idx = _get_dof_idx(j)
                if idx is not None:
                    kp[idx] = 2500.0
                    kv[idx] = 200.0
        set_dofs_kp_kv_compat(arm.entity, kp=kp, kv=kv)

    def _load_robot(self):
        """Pull Franka back so it doesn't block the avatar's +Y lean-in."""
        self.config.setdefault("robot_type", "franka")
        if self.config["robot_type"] == "franka":
            kwargs = self.config.setdefault("robot_kwargs", {})
            kwargs.setdefault("pos", [0.0, -0.5, 0.75])
            kwargs.setdefault("quat", [0.707, 0.0, 0.0, 0.707])
        super()._load_robot()

    def _create_table(self, table_height=0.74):
        """Smaller table (half_y=0.40) so the avatar's torso clears the edge
        during the Inspect motion."""
        dx, dy = self.table_offset
        return self._create_rectangular_table(
            table_height=table_height,
            half_size=(0.80, self.TABLE_HALF_SIZE_Y),
            center_xy=(dx, dy),
        )

    def load_actors(self):
        """Boxes on robot side (y=-0.20), loose objects on avatar side,
        same X column per category so each picked object moves along a roughly
        straight line back to its matching box.

        Per-seed randomization (seeded through BaseTask.reset via np.random.seed):
          - CATEGORIES are permuted
          - The interrupted robot target is sampled from all three objects
          - Column X gets ±2 cm jitter
          - Loose-object Y gets ±1.5 cm jitter
          - `place_decorative_objects` adds 3–4 random pool objects on the
            outer table strips, avoiding robot base and the sort corridors.
        """
        table_top = self.TABLE_TOP_Z + 0.02
        upright_q = np.asarray(self.SPAWN_UPRIGHT_QUAT, dtype=float)

        # Per-episode: sample NUM_CATEGORIES distinct objects from the shared
        # catalog pool (one per basket), then permute (and reduce for the
        # simplified single-object mode).
        pool = object_set_pairs(self.OBJECT_SET)
        sampled = np.random.choice(len(pool), self.NUM_CATEGORIES, replace=False)
        cats = [pool[int(i)] for i in sampled]
        perm = np.random.permutation(len(cats))
        if self.simplified_mode_enabled():
            perm = perm[:1]
        cats = [cats[i] for i in perm]
        self._cats = cats

        # Pick which robot-side object the avatar will inspect.  Default is a
        # random ordinal from the scripted robot pick order; review/debug runs
        # can force a pick ordinal, spawned index, or object name.
        forced_obj = (
            self.config.get("interrupt_target_object")
            or self.config.get("interrupt_object_name")
        )
        pick_order = sorted(range(len(cats)), reverse=True)
        if forced_obj:
            matches = [i for i, (obj_name, _) in enumerate(cats) if obj_name == forced_obj]
            if not matches:
                raise ValueError(
                    f"interrupt_target_object={forced_obj!r} not in spawned categories "
                    f"{[obj_name for obj_name, _ in cats]}"
                )
            self._interrupted_idx = int(matches[0])
            self._interrupted_pick_ordinal = int(pick_order.index(self._interrupted_idx))
        elif "interrupt_target_idx" in self.config:
            self._interrupted_idx = int(self.config["interrupt_target_idx"]) % len(cats)
            self._interrupted_pick_ordinal = int(pick_order.index(self._interrupted_idx))
        elif "interrupt_pick_ordinal" in self.config:
            self._interrupted_pick_ordinal = (
                int(self.config["interrupt_pick_ordinal"]) % len(pick_order)
            )
            self._interrupted_idx = int(pick_order[self._interrupted_pick_ordinal])
        elif self.config.get("random_interrupt_target", True):
            self._interrupted_pick_ordinal = int(np.random.randint(0, len(pick_order)))
            self._interrupted_idx = int(pick_order[self._interrupted_pick_ordinal])
        else:
            self._interrupted_idx = self.INTERRUPTED_IDX
            self._interrupted_pick_ordinal = int(pick_order.index(self._interrupted_idx))

        # Per-seed X jitter on each column (±2 cm) — boxes and loose objects
        # share the same jittered column X so the transport line is straight.
        col_xs = np.linspace(-0.25, 0.25, len(cats)) + np.random.uniform(
            -0.02, 0.02, size=len(cats)
        )
        print(f"[layout] cat_perm={list(perm)}, col_xs={col_xs}")

        box_y = -0.20
        # 0.8x of prior 24x24x14cm -> 19.2x19.2x11.2cm, so boxes look less
        # crowded on the table.
        box_half_x = 0.096
        box_half_y = 0.096
        box_half_z = 0.056
        wall_t = 0.005

        self.basket_poses = []
        self.basket_actors = []
        box_color = (0.7, 0.5, 0.3)
        for bx in col_xs:
            bx = float(bx)
            # Anchor basket bottom flush with TABLE_TOP_Z (not table_top, which
            # is TABLE_TOP_Z + 0.02 — that 2 cm offset is for spawning loose
            # objects so they fall under gravity; static baskets must NOT use
            # it or they appear to float above the table.)
            cz = self.TABLE_TOP_Z + box_half_z
            # basket_pos stored here is the outer box CENTER — the place/drop
            # logic inherited from CategorizeCooperative adds +0.20 to its z
            # to get a drop position above the rim, which is 7 cm above
            # center, so drop ends up 13 cm above the rim.
            self.basket_poses.append(Pose([bx, box_y, cz], np.array([1.0, 0.0, 0.0, 0.0])))

            # Floor (thin slab at the bottom).
            create_primitive(
                self.scene, "box",
                Pose(p=[bx, box_y, cz - box_half_z + wall_t]),
                size={"half_size": (box_half_x, box_half_y, wall_t)},
                color=box_color, is_static=True,
            )
            # Four walls — thin boxes standing around the floor, full height.
            for wx, wy, hx, hy in [
                (-box_half_x + wall_t, 0.0, wall_t, box_half_y),  # -X wall
                (+box_half_x - wall_t, 0.0, wall_t, box_half_y),  # +X wall
                (0.0, -box_half_y + wall_t, box_half_x, wall_t),  # -Y wall
                (0.0, +box_half_y - wall_t, box_half_x, wall_t),  # +Y wall
            ]:
                create_primitive(
                    self.scene, "box",
                    Pose(p=[bx + wx, box_y + wy, cz]),
                    size={"half_size": (hx, hy, box_half_z)},
                    color=box_color, is_static=True,
                )

        # Loose objects on the avatar side, same X columns as boxes (with
        # independent ±1.5 cm Y jitter per object).
        obj_y = self.INTERRUPT_OBJECT_Y
        self._sort_targets = []
        for i, (obj_name, model_id) in enumerate(cats):
            oy = obj_y + float(np.random.uniform(-0.015, 0.015))
            # _spawn_categorize_object (inherited) handles the cube primitive.
            actor = self._spawn_categorize_object(
                obj_name, model_id, float(col_xs[i]), oy, table_top, upright_q,
            )
            self._sort_targets.append((actor, obj_name, model_id, i))

        self._clutter_actors = []

        # Parent's check_success iterates these lists — keep them empty so
        # only the three sort_targets are evaluated.
        self._example_actors = []
        self._avatar_sort_targets = []

    def _open_basket_lids(self):
        return

    def reset(self, seed: int = 0):
        """Calibrate avatar before the episode is recorded.

        The calibration dry-runs the Inspect motion to measure where the
        avatar's palm lands at the attach frame, then shifts the avatar so
        palm meets the interrupted object on the table.  Running this here
        (before `start_video` is called) keeps the pre-motion out of the
        final video.
        """
        obs = super().reset(seed=seed)
        if self.avatar is not None:
            # Slow down avatar motion by resampling (more frames per step).
            self.avatar.frame_ratio = self.AVATAR_MOTION_SLOW
            self._calibrate_avatar_for_inspect(self._interrupted_idx)
        # Calibration ran the Inspect motion as a dry-run with the avatar at
        # its default (un-shifted) pose, which sweeps the right forearm
        # through the robot's home pose. Clear those setup-time collisions
        # so the success gate only counts contacts during play_once.
        self.avatar_collided = False
        self.avatar_collision_log = []
        return obs

    @property
    def _inspect_attach_frame_scaled(self):
        return int(round(self.INSPECT_ATTACH_FRAME * self.AVATAR_MOTION_SLOW))

    @property
    def _inspect_detach_frame_scaled(self):
        return int(round(self.INSPECT_DETACH_FRAME * self.AVATAR_MOTION_SLOW))

    def _calibrate_avatar_for_inspect(self, interrupted_idx):
        """Dry-run the Inspect motion with the default avatar pose, capture the
        palm position at the attach frame, then reset the avatar shifted so
        its palm lands exactly on the resting interrupted object."""
        int_actor = self._sort_targets[interrupted_idx][0]
        obj_pos = to_numpy(int_actor.entity.get_pos()).ravel()[:3]
        target_palm = np.array(obj_pos, dtype=np.float64)

        # This calibration happens inside reset().  Normal MP4 capture starts
        # after reset, but VLA capture is ticked directly from step_sim once
        # BaseTask.reset creates the recorder, so suppress recording here.
        saved_vla_recorder = self.vla_recorder
        saved_record_stride = self._record_stride
        self.vla_recorder = None
        self._record_stride = None
        try:
            self.avatar.play_animation(self.INSPECT_MOTION_NAME)
            palm_default = None
            step = 0
            attach_frame_scaled = self._inspect_attach_frame_scaled
            while not self.avatar.spare():
                self.step_sim()
                if step == attach_frame_scaled:
                    palm_default = np.asarray(
                        self.avatar.robot.get_palm_center(1), dtype=np.float64
                    ).copy()
                step += 1
        finally:
            self.vla_recorder = saved_vla_recorder
            self._record_stride = saved_record_stride

        if palm_default is None:
            print("[categorize_interrupt] WARNING: could not capture palm at attach frame")
            self.avatar.reset(
                np.asarray(self.avatar_init_pos, dtype=np.float64).copy(),
                np.asarray(self.avatar_init_rot, dtype=np.float64).copy(),
            )
            for _ in range(20):
                self.step_sim()
            return

        shift = target_palm - palm_default
        shift[2] += self.AVATAR_EXTRA_Z
        shifted_pos = np.asarray(self.avatar_init_pos, dtype=np.float64) + shift
        self.avatar.reset(
            shifted_pos.copy(),
            np.asarray(self.avatar_init_rot, dtype=np.float64).copy(),
        )
        for _ in range(30):
            self.step_sim()
        print(f"[categorize_interrupt] shifted avatar by {shift} -> new pos {shifted_pos} "
              f"(object rests at {obj_pos})")

    def _retreat_from_object(self, arm_tag):
        """Retreat away from the avatar first (−Y), then up — a straight
        lift overlapped with the avatar's forward-lean during Inspect.

        Uses a screw-motion from current EE position to a safe pose in front
        of the robot base so the arm pulls toward the robot (−Y) while lifting.
        """
        arm = self.robot.get_arm(arm_tag)
        safe_z = self.TABLE_TOP_Z + 0.45
        arm_base = np.array(arm.origin_pose.p)

        # Target: front-of-robot-base in (x=arm_base_x), y=arm_base_y+0.30
        # ≈ −0.20 world, z=safe_z.  The screw path gives a single diagonal
        # motion that clears the avatar's approach cone.
        retreat_pos = np.array([arm_base[0], arm_base[1] + 0.30, safe_z])
        old_replay_substeps = self._REPLAY_SUBSTEPS
        try:
            self._REPLAY_SUBSTEPS = self.RETREAT_REPLAY_SUBSTEPS
            self._move_screw(retreat_pos, arm_tag)
        finally:
            self._REPLAY_SUBSTEPS = old_replay_substeps
        self._move_to_safe(arm_tag)

    def _teleport_to_tcp(self, tcp_pose, arm_tag):
        """Override parent's set_qpos-based teleport with Cartesian PD motion
        so the robot's grasp approach is fully physics-simulated (no cheating).
        """
        arm = self.robot.get_arm(arm_tag)
        current_ee = np.array(arm.get_ee_pose()[:3], dtype=float)
        target_pos = np.array(tcp_pose.p, dtype=float)
        self._move_cartesian(
            current_ee, target_pos, arm_tag,
            n_steps=30, sim_per_step=15,
        )
        return True

    # Eval-mode interrupt state machine.  The cooperative parent queues
    # avatar-side sorting motions, but this task deliberately has no
    # `_avatar_sort_targets`; its avatar behavior is a timed Inspect
    # interruption of one robot target.
    EVAL_TRIGGER_STEP_MIN = 80
    EVAL_TRIGGER_STEP_MAX = 180

    def _eval_at_reset(self):
        self._eval_inspect_fired = False
        lo = int(self.config.get(
            "eval_trigger_step_min", self.EVAL_TRIGGER_STEP_MIN,
        ))
        hi = int(self.config.get(
            "eval_trigger_step_max", self.EVAL_TRIGGER_STEP_MAX,
        ))
        if hi < lo:
            hi = lo
        self._eval_trigger_step = int(np.random.randint(lo, hi + 1))
        print(f"[categorize_interrupt] eval mode: trigger step="
              f"{self._eval_trigger_step}")

    def _eval_at_step(self, step_idx: int):
        if self._eval_inspect_fired or step_idx < self._eval_trigger_step:
            return
        self._eval_inspect_fired = True
        interrupted_idx = self._interrupted_idx
        int_actor, int_name, _, _ = self._sort_targets[interrupted_idx]
        print(f"[categorize_interrupt] eval: triggering Inspect for "
              f"{int_name} at step {step_idx}")
        self._play_inspect_motion(
            attach_obj=int_actor.entity,
            hand_id=1,
            attach_frame=self._inspect_attach_frame_scaled,
            detach_frame=self._inspect_detach_frame_scaled,
            return_to_idle_after=None,  # freeze on last inspect frame (pose + position)
        )

    def _basket_metrics(self) -> dict:
        center_threshold = 0.10
        footprint_half = self._BASKET_INNER_HALF_XY + self._BASKET_FOOTPRINT_MARGIN
        per_object = []
        all_correct = True
        for actor, obj_name, model_id, basket_idx in self._sort_targets:
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
        metrics.update({
            "interrupted_idx": int(getattr(self, "_interrupted_idx", -1)),
            "interrupted_pick_ordinal": int(getattr(
                self, "_interrupted_pick_ordinal", -1,
            )),
            "interrupted_object": (
                self._sort_targets[self._interrupted_idx][1]
                if hasattr(self, "_sort_targets")
                and hasattr(self, "_interrupted_idx")
                and 0 <= self._interrupted_idx < len(self._sort_targets)
                else None
            ),
        })
        if bool(self.config.get("eval_mode", False)):
            metrics.update({
                "eval_mode": True,
                "eval_trigger_step": getattr(self, "_eval_trigger_step", None),
                "eval_policy_step_count": int(getattr(
                    self, "_eval_policy_step_count", 0,
                )),
                "eval_inspect_fired": bool(getattr(
                    self, "_eval_inspect_fired", False,
                )),
            })
        return metrics

    def play_once(self) -> bool:
        arm_tag = "right"
        self._boost_arm_pd()
        self._open_basket_lids()

        interrupted_idx = self._interrupted_idx
        int_actor, int_name, int_model_id, int_basket = self._sort_targets[interrupted_idx]
        interrupted_pick_ordinal = int(getattr(self, "_interrupted_pick_ordinal", -1))
        pick_order = self._scripted_pick_order()
        print(
            f"[categorize_interrupt] interrupted_pick_ordinal="
            f"{interrupted_pick_ordinal}, interrupted_idx={interrupted_idx} "
            f"({int_name}), pick_order={pick_order}"
        )

        arm = self.robot.get_arm(arm_tag)
        tcp_offset = arm.tcp_offset
        safe_z = self.TABLE_TOP_Z + 0.45

        def pick_place_target(target_idx: int) -> bool:
            actor, obj_name, model_id, basket_idx = self._sort_targets[target_idx]
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
            if not self.pick_and_place(pick, place, arm_tag):
                return False
            self._move_to_safe(arm_tag)
            return True

        # Phase 1: complete scripted picks before the interrupted ordinal. The
        # avatar should interrupt the object right before the robot reaches for
        # that pick, not always before the first pick.
        for ordinal, target_idx in enumerate(pick_order):
            if target_idx == interrupted_idx:
                break
            actor, obj_name, _, _ = self._sort_targets[target_idx]
            print(
                f"[categorize_interrupt] pre-interrupt pick ordinal={ordinal} "
                f"idx={target_idx} ({obj_name})"
            )
            if not pick_place_target(target_idx):
                return False

        # Phase 2: robot approaches the interrupted object (shows intent).
        self._move_to_safe(arm_tag)
        obj_center = self._get_object_world_center(int_actor, int_name, int_model_id)
        above_obj = np.array([obj_center[0], obj_center[1], safe_z])
        above_link = tcp_to_link_pose(self._top_down_tcp(above_obj), tcp_offset)
        self.open_gripper(arm_tag)
        self.move_and_execute(above_link.to_pose7(), arm_tag)

        # Descend partway — robot has "committed" to this object.
        pre_pos = obj_center.copy()
        pre_pos[2] += 0.12
        pre_link = tcp_to_link_pose(self._top_down_tcp(pre_pos), tcp_offset)
        self.move_and_execute(pre_link.to_pose7(), arm_tag)

        # Phase 3: avatar plays Inspect, attaching/detaching the interrupted object.
        # `return_to_idle_after=None` freezes the avatar on the Inspect motion's
        # last frame (pose + position) while the robot finishes sorting.
        if self.avatar is not None:
            self._play_inspect_motion(
                attach_obj=int_actor.entity,
                hand_id=1,
                attach_frame=self._inspect_attach_frame_scaled,
                detach_frame=self._inspect_detach_frame_scaled,
                return_to_idle_after=None,  # freeze on last inspect frame (pose + position)
            )

        # Phase 4: robot detects conflict and retreats to safe home.
        print(f"[categorize_interrupt] RETREATING — human is inspecting {int_name}")
        self._retreat_from_object(arm_tag)

        # Short wait while the human inspects — robot stays at safe.
        for _ in range(200):
            self.step_sim()

        # Phase 5: sort later non-interrupted objects first. Reset plan_success
        # since the approach/retreat may have triggered non-fatal IK failures.
        self.plan_success = True
        interrupted_order_pos = pick_order.index(interrupted_idx)
        for ordinal, target_idx in enumerate(pick_order[interrupted_order_pos + 1:],
                                             start=interrupted_order_pos + 1):
            actor, obj_name, _, _ = self._sort_targets[target_idx]
            print(
                f"[categorize_interrupt] post-interrupt pick ordinal={ordinal} "
                f"idx={target_idx} ({obj_name})"
            )
            if not pick_place_target(target_idx):
                return False

        # Phase 6: make sure the avatar has finished its Inspect + idle return
        # before the robot comes back for the interrupted object.  The idle
        # return is triggered automatically inside `play_animation`, so by the
        # time the robot is done sorting the non-interrupted objects it should
        # already be well underway.
        if self.avatar is not None:
            while not self.avatar.spare():
                self.step_sim()
        for _ in range(100):
            self.step_sim()

        # Phase 7: pick the interrupted object now that the human is done.
        if not pick_place_target(interrupted_idx):
            return False
        for _ in range(100):
            self.step_sim()

        return True

    def play_blind_once(self) -> bool:
        """Expert baseline: do the categorization task without privileged
        knowledge of the avatar's future interrupt.

        The eval-mode avatar hook is ticked from step_sim because scripted
        expert rollouts do not call take_action().
        """
        orig_step_sim = self.step_sim
        tick_interval = max(1, int(self.config.get(
            "eval_parent_rollout_tick_interval", 5,
        )))
        counters = {"sim": 0, "policy": 0}

        def driven_step_sim():
            orig_step_sim()
            counters["sim"] += 1
            if counters["sim"] % tick_interval != 0:
                return
            if self.avatar is None or not bool(self.config.get("eval_mode", False)):
                return
            counters["policy"] += 1
            self._eval_at_step(counters["policy"])

        def sort_target(target_idx: int) -> bool:
            actor, obj_name, model_id, basket_idx = self._sort_targets[target_idx]
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
            ok = self.pick_and_place(pick, place, "right")
            self._move_to_safe("right")
            obj_center = self._get_object_world_center(actor, obj_name, model_id)
            placed = bool(float(np.linalg.norm(obj_center[:2] - basket_pos[:2])) <= 0.10)
            return bool(ok and placed)

        self.step_sim = driven_step_sim
        try:
            self._boost_arm_pd()
            self._open_basket_lids()
            self._move_to_safe("right")
            any_success = False
            for target_idx in self._scripted_pick_order():
                ok = sort_target(target_idx)
                any_success = any_success or ok
            for _ in range(200):
                self.step_sim()
            return any_success
        finally:
            self.step_sim = orig_step_sim
