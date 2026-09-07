"""Interrupt variant for ``blocks_ranking_size``.

The robot sorts the RGB cubes by size, but a human avatar inspects one
randomly chosen size block mid-task.  The robot first approaches that block to
show intent, retreats while the avatar picks it up with the Inspect motion,
sorts the other two blocks, then re-reads the inspected block's released pose
and sorts it.
"""

import numpy as np

from ..avatar.inspect_motion_mixin import InspectMotionMixin, STANDARD_INSPECT_POOL
from ..grasp import tcp_to_link_pose
from ..manipulation import PickSpec, PlaceSpec
from ..utils import Pose, create_primitive, to_numpy
from ..task_bases.blocks_ranking_size import BlocksRankingSize


class BlocksRankingSizeInterrupt(InspectMotionMixin, BlocksRankingSize):
    """Human inspects one size block while the robot is ordering RGB blocks by size."""

    INSTRUCTION = (
        "place the RGB blocks in largest-to-smallest order; wait while the "
        "human inspects one block, then continue"
    )

    use_avatar = True

    AVATAR_BASE_ROT = np.array(
        [[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=np.float64
    )
    avatar_init_pos = np.array([0.0, 0.55, -0.38])
    avatar_init_rot = AVATAR_BASE_ROT

    INSPECT_MOTION_NAME = "Inspect1"
    INSPECT_ATTACH_FRAME = 34
    INSPECT_DETACH_FRAME = 187
    INSPECT_POOL = STANDARD_INSPECT_POOL
    AVATAR_MOTION_SLOW = 10.0
    AVATAR_HAND_ID = 1
    AVATAR_EXTRA_Z = 0.04

    # Shared avatar-reachable spawn band near the +Y table edge.  All blocks
    # are sampled here so the robot cannot infer the interrupt target from
    # which object is uniquely close enough for the avatar to inspect.
    AVATAR_REACH_SPAWN_X_RANGE = (-0.18, 0.18)
    AVATAR_REACH_SPAWN_Y_RANGE = (-0.02, 0.13)
    INTERRUPT_SPAWN_X_RANGE = AVATAR_REACH_SPAWN_X_RANGE
    INTERRUPT_SPAWN_Y_RANGE = AVATAR_REACH_SPAWN_Y_RANGE

    EVAL_TRIGGER_STEP_MIN = 100
    EVAL_TRIGGER_STEP_MAX = 200
    EVAL_GLIDE_POLICY_STEPS = 20
    SCRIPT_TARGET_SEED_OFFSET = 7919
    SCRIPT_TIME_SEED_OFFSET = 13
    CALIBRATION_SETTLE_STEPS = 20
    POST_CALIBRATION_SETTLE_STEPS = 30
    RETREAT_Y_FROM_BASE = 0.25
    RETREAT_Z_ABOVE_TABLE = 0.30
    # None ⇒ avatar freezes on the inspect motion's last frame (pose + position).
    INSPECT_RETURN_TO_IDLE_FRAMES = None
    POST_ATTACH_WAIT_MARGIN_STEPS = 100
    POST_RELEASE_SETTLE_STEPS = 120
    EVAL_TABLE_Z_LOWER_MARGIN = 0.05
    EVAL_TABLE_Z_UPPER_MARGIN = 0.30

    def reset(self, seed: int = 0):
        self._episode_seed = int(seed)
        self._released_interrupted_pos = None
        self._eval_target_size = None
        self._eval_interrupt_started = False
        self._script_target_rng = np.random.default_rng(
            self._episode_seed + self.SCRIPT_TARGET_SEED_OFFSET
        )
        time_rng = np.random.default_rng(
            self._episode_seed + self.SCRIPT_TIME_SEED_OFFSET
        )
        # Number of non-interrupted blocks to place before the avatar
        # interrupts.  0 = interrupt immediately; 2 = interrupt late.
        self._scripted_prefix_count = int(time_rng.integers(0, 3))
        obs = super().reset(seed=seed)
        if self.avatar is None:
            return obs
        self.avatar.frame_ratio = self.AVATAR_MOTION_SLOW
        if bool(self.config.get("eval_mode", False)):
            self._init_eval_interrupt_state()
        else:
            self._calibrate_avatar_for_inspect()
        return obs

    def take_action(self, action, action_type: str = "qpos"):
        if self.avatar is not None and bool(self.config.get("eval_mode", False)):
            self._eval_step_avatar()
        return super().take_action(action, action_type=action_type)

    def load_actors(self):
        size_names = list(self.SIZE_ORDER)
        color_names = list(self.CUBE_COLORS.keys())
        color_order = np.random.permutation(color_names)
        size_order = np.random.permutation(size_names)
        target_rng = getattr(self, "_script_target_rng", None)
        if target_rng is None:
            size_idx = int(np.random.randint(len(size_names)))
        else:
            size_idx = int(target_rng.integers(0, len(size_names)))
        self._interrupted_size = size_names[size_idx]
        xys = self._sample_spawn_xys_for_order(size_order)

        self.blocks = {}
        self.block_colors = {}
        for i, size_name in enumerate(size_order):
            half = float(self.SIZE_HALVES[size_name])
            color_name = str(color_order[i])
            cube_z = self.TABLE_TOP_Z + half + self.CUBE_Z_CLEARANCE
            cube = create_primitive(
                self.scene,
                "box",
                Pose([float(xys[i, 0]), float(xys[i, 1]), cube_z]),
                size={"half_size": (half,) * 3},
                color=self.CUBE_COLORS[color_name],
                is_static=False,
            )
            self.blocks[size_name] = cube
            self.block_colors[size_name] = color_name

        self.target_slots = {
            size_name: self._target_slot_pos(
                size_name,
                self.TABLE_TOP_Z + float(self.SIZE_HALVES[size_name]),
            )
            for size_name in self.SIZE_ORDER
        }

    def _sample_spawn_xys_for_order(self, order):
        x_lo, x_hi = self.AVATAR_REACH_SPAWN_X_RANGE
        y_lo, y_hi = self.AVATAR_REACH_SPAWN_Y_RANGE
        for _ in range(self.SPAWN_SAMPLE_ATTEMPTS):
            xys = np.zeros((3, 2), dtype=float)
            for i, _size_name in enumerate(order):
                xys[i, 0] = np.random.uniform(x_lo, x_hi)
                xys[i, 1] = np.random.uniform(y_lo, y_hi)
            ok = True
            for i in range(3):
                for j in range(i + 1, 3):
                    if np.linalg.norm(xys[i] - xys[j]) < self.MIN_BLOCK_SPACING:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                return self._layout_xys(xys)
        raise RuntimeError("[blocks_size_interrupt] could not sample non-overlapping cubes")

    @property
    def _attach_frame_scaled(self) -> int:
        return int(round(self.INSPECT_ATTACH_FRAME * self.AVATAR_MOTION_SLOW))

    @property
    def _detach_frame_scaled(self) -> int:
        return int(round(self.INSPECT_DETACH_FRAME * self.AVATAR_MOTION_SLOW))

    def _dryrun_capture_palm(self):
        cache_key = (
            self.INSPECT_MOTION_NAME,
            float(self.AVATAR_MOTION_SLOW),
            self.AVATAR_HAND_ID,
            tuple(np.asarray(self.avatar_init_pos, dtype=np.float64).ravel()),
        )
        cached = getattr(type(self), "_palm_default_cache", {}).get(cache_key)
        if cached is not None:
            return cached.copy()

        from ..avatar.utils import Mixamo_data_to_controller_pose, Mixamo_node_processing

        md = self.avatar.motion_data.get(self.INSPECT_MOTION_NAME)
        if md is None:
            return None
        src_len = int(md["trans"].shape[0])
        scaled_len = max(1, int(src_len * self.AVATAR_MOTION_SLOW))
        scaled_indices = np.round(np.linspace(0, src_len - 1, scaled_len)).astype(int)
        frame = min(self._attach_frame_scaled, scaled_len - 1)
        src_idx = int(scaled_indices[frame])

        pose = Mixamo_data_to_controller_pose(
            md["trans"][src_idx],
            md["rot"][src_idx],
            md["joint"][src_idx],
        )
        global_mat = md["mat"][0]
        global_mat_inv = np.array([np.linalg.inv(m) for m in global_mat])
        vgeom = self.avatar.robot.skin.links[0]._vgeoms[0]
        node_trans = Mixamo_node_processing(vgeom, pose, global_mat, global_mat_inv)

        self.avatar.robot.pose = pose
        self.avatar.robot.node_trans = node_trans
        self.avatar.robot.global_mat = global_mat
        self.avatar.robot.global_mat_inv = global_mat_inv
        self.avatar.robot.update()
        palm_default = np.asarray(
            self.avatar.robot.get_palm_center(self.AVATAR_HAND_ID),
            dtype=np.float64,
        ).copy()

        self.avatar.reset(
            np.asarray(self.avatar_init_pos, dtype=np.float64).copy(),
            np.asarray(self.avatar_init_rot, dtype=np.float64).copy(),
        )
        for _ in range(self.CALIBRATION_SETTLE_STEPS):
            self.step_sim()
        self.avatar_collided = False
        self.avatar_collision_log = []

        if palm_default is not None:
            cache = getattr(type(self), "_palm_default_cache", None)
            if cache is None:
                cache = {}
                type(self)._palm_default_cache = cache
            cache[cache_key] = palm_default.copy()
        return palm_default

    def _calibrate_avatar_for_inspect(self):
        cube = self.blocks[self._interrupted_size]
        target_palm = to_numpy(cube.get_pos()).ravel()[:3].astype(float)
        palm_default = self._dryrun_capture_palm()
        if palm_default is None:
            return

        shift = target_palm - palm_default
        shift[2] += self.AVATAR_EXTRA_Z
        shifted_pos = np.asarray(self.avatar_init_pos, dtype=np.float64) + shift
        self.avatar.reset(
            shifted_pos.copy(),
            np.asarray(self.avatar_init_rot, dtype=np.float64).copy(),
        )
        for _ in range(self.POST_CALIBRATION_SETTLE_STEPS):
            self.step_sim()

    def _retreat_safe(self, arm_tag: str):
        arm = self.robot.get_arm(arm_tag)
        arm_base = np.array(arm.origin_pose.p, dtype=float)
        retreat_xy = arm_base[:2] + self._layout_xy([0.0, self.RETREAT_Y_FROM_BASE])
        retreat_p = np.array([
            retreat_xy[0],
            retreat_xy[1],
            self.TABLE_TOP_Z + self.RETREAT_Z_ABOVE_TABLE,
        ])
        self._move_screw(retreat_p, arm_tag)

    def _wait_until_avatar_attach(self):
        motion = self.avatar.motion_modules.get(self.INSPECT_MOTION_NAME)
        if motion is None:
            return
        while (
            not self.avatar.spare()
            and int(getattr(motion, "at_frame", 0)) < self._attach_frame_scaled
        ):
            self.step_sim()

    def _pick_and_place_block(self, size_name: str, arm_tag: str = "right") -> bool:
        cube = self.blocks[size_name]
        target = self.target_slots[size_name]
        half = float(self.SIZE_HALVES[size_name])
        color_name = self.block_colors.get(size_name, "unknown")
        transport_z = self.TABLE_TOP_Z + self.TRANSPORT_Z_ABOVE_TABLE
        release_z = self.TABLE_TOP_Z + self.RELEASE_Z_ABOVE_TABLE

        pick = PickSpec(
            get_center=lambda c=cube: to_numpy(c.get_pos()).ravel()[:3],
            radius=half,
            label=f"{size_name} {color_name} block",
        )
        place = PlaceSpec(
            pos=np.array([target[0], target[1], release_z], dtype=float),
            label=f"{size_name} slot",
            transport_z=transport_z,
            release=True,
        )
        return bool(self.pick_and_place(pick, place, arm_tag))

    def play_once(self) -> bool:
        arm_tag = "right"
        self._boost_finger_pd(arm_tag)

        with self.suppress_recording():  # pre-task settle, excluded from recordings
            for _ in range(self.TABLE_SETTLE_STEPS):
                self.step_sim()

        if self.avatar is None:
            return super().play_once()

        int_size = self._interrupted_size
        int_cube = self.blocks[int_size]
        non_interrupted = [s for s in self.SIZE_ORDER if s != int_size]
        prefix_count = int(getattr(self, "_scripted_prefix_count", 0))
        prefix_count = min(max(prefix_count, 0), len(non_interrupted))
        before_interrupt = non_interrupted[:prefix_count]
        during_interrupt = non_interrupted[prefix_count:]
        ok_all = True

        def place_block(size_name):
            try:
                ok = self._pick_and_place_block(size_name, arm_tag)
            except Exception:
                ok = False
            if not ok:
                self.open_gripper(arm_tag)
                for _ in range(self.FAILURE_RECOVERY_STEPS):
                    self.step_sim()
            return bool(ok)

        self.plan_success = True
        for size_name in before_interrupt:
            ok_all = place_block(size_name) and ok_all

        # Re-read in case earlier robot motion jostled the interrupted block.
        int_pos = to_numpy(int_cube.get_pos()).ravel()[:3].astype(float)
        arm = self.robot.get_arm(arm_tag)
        self.open_gripper(arm_tag)
        intent_pos = np.array([
            int_pos[0],
            int_pos[1],
            self.TABLE_TOP_Z + self.TRANSPORT_Z_ABOVE_TABLE,
        ])
        above_link = tcp_to_link_pose(self._top_down_tcp(intent_pos), arm.tcp_offset)
        self._move_seeded(above_link.to_pose7(), arm_tag)

        self._play_inspect_motion(
            attach_obj=int_cube,
            hand_id=self.AVATAR_HAND_ID,
            attach_frame=self._attach_frame_scaled,
            detach_frame=self._detach_frame_scaled,
            return_to_idle_after=self.INSPECT_RETURN_TO_IDLE_FRAMES,
        )

        self._retreat_safe(arm_tag)
        self._wait_until_avatar_attach()
        for _ in range(self.POST_ATTACH_WAIT_MARGIN_STEPS):
            self.step_sim()

        for size_name in during_interrupt:
            ok_all = place_block(size_name) and ok_all

        if not self.avatar.spare():
            while not self.avatar.spare():
                self.step_sim()
        for _ in range(self.POST_RELEASE_SETTLE_STEPS):
            self.step_sim()

        post_release_pos = to_numpy(int_cube.get_pos()).ravel()[:3]
        self._released_interrupted_pos = post_release_pos.copy()
        ok = place_block(int_size)
        ok_all = ok and ok_all

        for _ in range(self.FINAL_SETTLE_STEPS):
            self.step_sim()
        return ok_all

    def _init_eval_interrupt_state(self):
        self._eval_palm_default = self._dryrun_capture_palm()
        if self._eval_palm_default is None:
            self._eval_interrupt_fired = True
        else:
            self._eval_trigger_step = int(np.random.randint(
                self.EVAL_TRIGGER_STEP_MIN,
                self.EVAL_TRIGGER_STEP_MAX + 1,
            ))
            self._eval_interrupt_fired = False
        self._eval_glide_active = False
        self._eval_glide_remaining = 0
        self._eval_glide_start_pos = None
        self._eval_glide_target_pos = None
        self._eval_target_block = None
        self._eval_target_size = None
        self._eval_interrupt_started = False
        self._policy_step_count = 0

    def _eval_step_avatar(self):
        self._policy_step_count += 1

        if self._eval_glide_active:
            self._eval_glide_remaining -= 1
            done = self.EVAL_GLIDE_POLICY_STEPS - self._eval_glide_remaining
            progress = min(1.0, max(0.0, done / float(self.EVAL_GLIDE_POLICY_STEPS)))
            cur = (
                (1.0 - progress) * self._eval_glide_start_pos +
                progress * self._eval_glide_target_pos
            )
            self.avatar.reset(
                cur.copy(),
                np.asarray(self.avatar_init_rot, dtype=np.float64).copy(),
            )
            if self._eval_glide_remaining <= 0:
                self._eval_glide_active = False
                cube = self._eval_target_block
                cube_pos = to_numpy(cube.get_pos()).ravel()[:3]
                if cube_pos[2] < self.TABLE_TOP_Z - self.EVAL_TABLE_Z_LOWER_MARGIN:
                    return
                self._play_inspect_motion(
                    attach_obj=cube,
                    hand_id=self.AVATAR_HAND_ID,
                    attach_frame=self._attach_frame_scaled,
                    detach_frame=self._detach_frame_scaled,
                    return_to_idle_after=self.INSPECT_RETURN_TO_IDLE_FRAMES,
                )
                self._eval_interrupt_started = True
            return

        if self._eval_interrupt_fired:
            return
        if self._policy_step_count < self._eval_trigger_step:
            return
        self._eval_interrupt_fired = True

        size_name = self._find_closest_table_block()
        if size_name is None:
            return
        cube = self.blocks[size_name]
        target_pos = to_numpy(cube.get_pos()).ravel()[:3].astype(float)
        shift = target_pos - self._eval_palm_default
        shift[2] += self.AVATAR_EXTRA_Z
        target_avatar_pos = np.asarray(self.avatar_init_pos, dtype=np.float64) + shift

        self._eval_glide_start_pos = np.asarray(self.avatar_init_pos, dtype=np.float64).copy()
        self._eval_glide_target_pos = target_avatar_pos.copy()
        self._eval_glide_remaining = self.EVAL_GLIDE_POLICY_STEPS
        self._eval_glide_active = True
        self._eval_target_block = cube
        self._eval_target_size = size_name

    def _find_closest_table_block(self):
        arm = self.robot.get_arm("right")
        ee_pos = np.array(arm.get_ee_pose()[:3], dtype=float)
        best_size = None
        best_d = float("inf")
        for size_name, cube in self.blocks.items():
            p = to_numpy(cube.get_pos()).ravel()[:3]
            target = self.target_slots[size_name]
            in_slot = np.linalg.norm(p[:2] - target[:2]) <= self.SUCCESS_DIST_XY
            if in_slot:
                continue
            if (
                p[2] < self.TABLE_TOP_Z - self.EVAL_TABLE_Z_LOWER_MARGIN or
                p[2] > self.TABLE_TOP_Z + self.EVAL_TABLE_Z_UPPER_MARGIN
            ):
                continue
            d = float(np.linalg.norm(p[:2] - ee_pos[:2]))
            if d < best_d:
                best_d = d
                best_size = size_name
        return best_size

    def evaluate(self) -> dict:
        metrics = super().evaluate()
        metrics.update({
            "interrupted_size": getattr(self, "_interrupted_size", None),
            "interrupted_color": self.block_colors.get(
                getattr(self, "_interrupted_size", None),
                None,
            ),
            "scripted_prefix_count": int(getattr(self, "_scripted_prefix_count", 0)),
            "interrupt_motion": getattr(self, "INSPECT_MOTION_NAME", None),
            "interrupt_attach_frame": int(getattr(self, "INSPECT_ATTACH_FRAME", -1)),
            "interrupt_detach_frame": int(getattr(self, "INSPECT_DETACH_FRAME", -1)),
            "eval_mode": bool(self.config.get("eval_mode", False)),
            "eval_trigger_step": getattr(self, "_eval_trigger_step", None),
            "eval_policy_step_count": int(getattr(self, "_policy_step_count", 0)),
            "eval_interrupt_fired": bool(getattr(self, "_eval_interrupt_fired", False)),
            "eval_interrupt_started": bool(getattr(self, "_eval_interrupt_started", False)),
            "eval_target_size": getattr(self, "_eval_target_size", None),
        })
        released = getattr(self, "_released_interrupted_pos", None)
        if released is not None:
            metrics["released_interrupted_pos"] = np.asarray(released).tolist()
        return metrics
