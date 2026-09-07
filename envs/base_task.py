"""Base task: slim foundation for all manipulation tasks."""

import contextlib
import os
import pickle
import random
import zlib
import numpy as np
import torch
import genesis as gs
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable, Literal, Optional, Union
from pathlib import Path

from .utils import Pose, create_primitive, place_decorative_objects, to_numpy, ASSETS_PATH
from .genesis_compat import update_dofs_force_range_compat, update_dofs_kp_kv_compat
from .table_assets import add_table_variant, table_randomization_enabled
from .scene_init import (
    apply_scene_init_post_settle,
    apply_scene_init_pre_avatar,
    prepare_config_for_scene_init,
)
from .grasp import load_grasp_poses, tcp_to_link_pose, GraspPose
from .robot import PiperRobot, FrankaRobot, StretchRobot, XArm7Robot, ARXX5Robot, UR5WSGRobot
from .camera import Camera, NyxCameraManager
from .exr_envmap import normalize_env_offset
from .avatar.model_list import DEFAULT_AVATAR_POOL

TABLE_HEIGHT = 0.74

# Default renderer when config["renderer"] is absent (e.g. a task constructed
# from a bare dict, bypassing config/default.yml). Matches config/default.yml so
# the two can't drift. nyx is a SMART default: _resolve_renderer() transparently
# falls back to the rasterizer when nyx can't run (CPU backend, no CUDA GPU, or
# an incompatible gs_nyx_plugin), so this never breaks CPU/debug runs.
DEFAULT_RENDERER = "nyx"


@dataclass
class TargetSpec:
    """Per-task evaluation target.

    Tasks opt in by setting ``self.target = TargetSpec(...)`` in ``load_actors``.
    ``BaseTask.evaluate()`` then reports position / orientation / collision
    metrics; the default ``check_success()`` passes when ``plan_success`` is
    True and every configured threshold holds, so simple tasks don't need
    their own success logic.

    Geometry hooks:
      object:          actor/entity whose final pose is measured (exposes
                       ``.entity.get_pos()``). None disables target metrics.
      position:        ``np.ndarray`` for a fixed target, or a zero-arg
                       callable returning one (use a callable when the
                       target moves — e.g. the avatar's palm).
      label:           human-readable tag used in console/pickle output
                       (e.g. "plate", "hand", "cup_rim").
      object_pos_fn:   optional zero-arg callable returning the object's
                       world position, used instead of ``entity.get_pos()``
                       (e.g. bbox centre instead of mesh origin).
      object_axis_fn:  optional zero-arg callable returning the object's
                       "up" direction in world frame (unit vector).  Enables
                       tilt metrics / thresholds.

    Success thresholds (all optional; set only what matters for the task):
      success_dist_xy:         pass if lateral distance <= this (m)
      success_dist_3d:         pass if 3-D distance <= this (m)
      success_dz_min:          pass if obj.z - tgt.z >= this (m)
      success_tilt_min_deg:    pass if angle(object_axis, world +Z) >= this
      success_tilt_max_deg:    pass if angle(object_axis, world +Z) <= this
      success_require_no_avatar_collision:
          pass only if the avatar-collision tracker was enabled and never
          triggered this episode.  No-op when tracking is off.
    """
    object: Any = None
    position: Optional[Union[np.ndarray, Callable[[], np.ndarray]]] = None
    label: str = "target"
    object_pos_fn: Optional[Callable[[], np.ndarray]] = None
    object_axis_fn: Optional[Callable[[], np.ndarray]] = None
    success_dist_xy: Optional[float] = None
    success_dist_3d: Optional[float] = None
    success_dz_min: Optional[float] = None
    success_tilt_min_deg: Optional[float] = None
    success_tilt_max_deg: Optional[float] = None
    success_require_no_avatar_collision: bool = False

class BaseTask:
    """Base class for all manipulation tasks.

    Lifecycle:
        1. __init__(config) — parse config
        2. reset(seed) — build scene, load actors, return initial obs
        3. play_once() — scripted rollout (data collection)
           OR take_action(action) loop — policy evaluation
        4. check_success() — evaluate task completion

    Subclasses must override:
        - load_actors() — add task-specific objects to scene
        - play_once() — scripted task execution
        - check_success() — success evaluation
    """

    use_avatar = False  # Override in subclass to enable avatar
    use_table = True    # Override to False for floor-only tasks (mobile pick, etc.)

    # Shared easy/hard object-count knobs for tasks whose difficulty is mainly
    # the number of task-relevant objects. Subclasses override these values.
    EASY_NUM_OBJECTS = 1
    HARD_NUM_OBJECTS = 2

    # Universal tabletop clutter defaults. These are real dynamic collision
    # bodies when table_random_objects is enabled.
    TABLE_RANDOM_OBJECT_COUNT = 2
    TABLE_RANDOM_OBJECT_TARGET_SIZE = 0.09
    TABLE_RANDOM_OBJECT_TABLE_MARGIN = 0.07
    TABLE_RANDOM_OBJECT_EXCLUSION_RADIUS = 0.20
    TABLE_RANDOM_OBJECT_ROBOT_BASE_RADIUS = 0.30
    TABLE_RANDOM_OBJECT_TYPE_ALIASES = {
        "block": ("086_woodenblock", "108_block"),
        "cube": ("086_woodenblock", "108_block"),
        "woodenblock": ("086_woodenblock",),
        "bowl": ("002_bowl",),
        "shoe": ("041_shoe",),
        "bread": ("075_bread",),
        "burger": ("006_hamburg",),
        "hamburg": ("006_hamburg",),
        "fries": ("005_french-fries",),
        "french-fries": ("005_french-fries",),
        "apple": ("035_apple",),
        "can": ("071_can", "105_sauce-can"),
        "cup": ("021_cup",),
        "mug": ("039_mug",),
        "phone": ("077_phone",),
        "stapler": ("048_stapler",),
        "tissue": ("023_tissue-box",),
        "milk": ("038_milk-box",),
        "jam": ("031_jam-jar",),
        "oil": ("029_olive-oil",),
        "seal": ("100_seal",),
    }

    # Short imperative natural-language instruction (Bridge V2 style).
    # Required for OpenVLA / Bridge-format data collection. Subclasses must
    # set this; collect.py writes it into each episode record.  May contain a
    # ``{object}`` slot, filled with the resolved target's display name by
    # ``resolve_target_object`` (see envs/object_catalog.py).
    INSTRUCTION = None

    # Target-object selection (see envs/object_catalog.py).  Either an inline
    # white-list of object keys (``["041_shoe"]``) or a category name
    # (``"cabinet_drawer_items"``).  Tasks that pick/place a catalog object set
    # this and call ``resolve_target_object()``; None means the task does not
    # use the catalog (e.g. primitive-only blocks tasks, multi-object trays).
    OBJECT_SET = None

    # Single global reset-settle budget: sim steps to run after scene build so
    # dynamic objects fall and settle before any task begins.  Keep this
    # uniform across tasks; override only via config["settle_steps"] for an
    # explicit experiment/debug run.
    SETTLE_STEPS = 400

    # Capture one video frame every VIDEO_STRIDE sim steps. 1 = every step.
    # Tasks with long rollouts should bump this to 6+ to cut render cost —
    # rendering is the dominant per-step wall-clock cost. Override via subclass
    # class-attr or via config["video_stride"].
    VIDEO_STRIDE = 1

    # Record one observation (image + proprio) every RECORD_STRIDE step_sim
    # calls into self.traj_data. None = recording disabled (default; preserves
    # legacy behavior where only tasks that explicitly call self.record_frame()
    # populate traj_data). Override via config["record_stride"]. Genesis default
    # dt=0.002s, so 50 ≈ 10 Hz, 100 ≈ 5 Hz (Bridge-V2-style cadence).
    RECORD_STRIDE = None

    # Pool of avatar glb paths (relative to ASSETS_PATH). Consulted when
    # config["randomize_avatar"] is True. Subclasses can override with a
    # task-specific subset; None falls back to DEFAULT_AVATAR_POOL below.
    AVATAR_POOL = None

    # Default skin settings used when config doesn't specify them.
    DEFAULT_AVATAR_SKIN = {
        "glb_path": "avatars/models/custom_Adrian_Keller.glb",
        "euler": (-90, 0, 90),
        "pos": (0.0, 0.0, -0.959008030),
    }

    FLOOR_HALF_SIZE = (2.2, 2.2)
    FLOOR_COLOR = (0.64, 0.65, 0.62)
    FLOOR_THICKNESS = 0.004

    # ------------------------------------------------------------------
    # Avatar-collision tracking (opt-in, all default off).  Turn on via
    # `config["track_avatar_collision"] = True` (or the matching
    # --track-avatar-collision CLI flag).  See
    # envs/avatar/collision_checker.py for the analytic method and
    # scripts/debug_avatar_collision.py for a standalone demo.
    DEFAULT_COLLISION_CHECK_STRIDE = 30   # ~60 ms real-time at 500 Hz sim
    DEFAULT_COLLISION_MARGIN = 0.0        # 0 = strict overlap; +ve = near-miss

    def __init__(self, config: dict = None):
        self.config = prepare_config_for_scene_init(config or {}, (config or {}).get("scene_init"))
        self.instruction = self.config.get("instruction") or type(self).INSTRUCTION
        # Base instruction string (may contain a ``{object}`` slot); kept so
        # resolve_target_object can re-template it per reset.
        self._instruction_template = self.instruction
        self._target_entry = None          # last-resolved catalog ObjectEntry
        self._episode_seed = 0
        self._episode_rngs = {}
        self.scene = None
        self.robot = None
        self.cameras = None
        self.avatar = None
        self.plan_success = True
        self.eval_success = False
        self.take_action_cnt = 0
        self.FRAME_IDX = 0

        # Avatar-collision tracking state.  `avatar_collider` is built
        # pre-build iff `track_avatar_collision` (or `use_avatar_collider`)
        # is on; `avatar_collision_checker` is built post-build.
        self.avatar_collider = None
        self.avatar_collision_checker = None
        self.avatar_safety_checker = None
        self.avatar_retreat_checker = None
        self.avatar_collided = False       # sticky — first hit sets True
        self.avatar_collision_log = []     # list of per-check dicts
        self.avatar_safety_log = []        # inflated-envelope near-miss checks
        self.avatar_safety_intervention_log = []
        self._avatar_retreat_hold_count = 0
        self._avatar_retreat_tick = 0
        self._avatar_retreat_active_cached = False
        self._avatar_retreat_release_state = {}
        self._avatar_collision_stride = int(self.config.get(
            "collision_check_stride", self.DEFAULT_COLLISION_CHECK_STRIDE))
        self._avatar_collision_stride = max(1, self._avatar_collision_stride)
        self._avatar_collision_tick = 0

        # Trajectory recording
        self.left_joint_path = []
        self.right_joint_path = []
        self.traj_data = []

        # Video recording
        self._video_frames = []
        self._video_path = None
        self._video_stride = int(self.config.get("video_stride", self.VIDEO_STRIDE))
        self._video_stride = max(1, self._video_stride)
        self._video_cap_tick = 0
        # When True, step_sim records nothing (video/VLA/trajectory).  Used to
        # exclude pre-task scene settling + avatar/scene setup from the
        # recordings — see BaseTask.suppress_recording().
        self._record_suppressed = False

        # Trajectory recording stride. Off by default; set via config to opt in.
        rs = self.config.get("record_stride", self.RECORD_STRIDE)
        self._record_stride = int(rs) if rs is not None else None
        self._record_tick = 0

        # Optional whitelist of camera names to keep in recorded obs.rgb.
        # None = keep all cameras (legacy behavior). Set to e.g. ["head_camera"]
        # to dump only the cameras you actually need — cuts pickle size by ~5x
        # on a 5-camera scene.
        rc = self.config.get("record_cameras")
        self._record_cameras = list(rc) if rc else None

        # Gripper attachment (for DEBUG_TELEPORT / kinematic grasp)
        # {arm_tag: (entity, relative_pose)} where relative_pose is Pose of object in EE frame
        self._gripper_attached = {}

        # Target-object evaluation spec (see `TargetSpec` docstring).
        # Tasks opt in by replacing this in `load_actors`; the default
        # spec has no object/position so target metrics are omitted.
        self.target = TargetSpec()
        self.table_variant_name = None
        self._table_top_z_fixed = False
        self._table_center_xy = np.asarray(getattr(self, "table_offset", (0.0, 0.0)), dtype=float)[:2]
        self._table_half_size_xy = np.asarray((0.6, 0.35), dtype=float)
        self._table_random_object_actors = []
        self._table_random_object_exclusions = []
        self.table_random_objects = bool(
            self.config.get(
                "table_random_objects",
                self.config.get(
                    "task_irrelevant_objects",
                    self.config.get("with_irrelevant_objects", False),
                ),
            )
        )
        # Backward-compatible attribute name used by older task code.
        self.task_irrelevant_objects = self.table_random_objects
        difficulty = str(self.config.get("difficulty", "")).strip().lower()
        self.hard = bool(
            self.config.get("hard", self.config.get("task_hard", False))
            or difficulty == "hard"
        )

        # Optional VLA training-data recorder.  Built in reset() iff
        # config["vla_recording"]["enabled"]; ticked from step_sim.
        self.vla_recorder = None
        self.scene_init_replay = {"enabled": False}

    # ------------------------------------------------------------------
    # Target-object selection (see envs/object_catalog.py)
    # ------------------------------------------------------------------

    def resolve_target_object(self, *, randomize: bool = True):
        """Resolve this task's target object from ``OBJECT_SET``.

        ``config["object_name"]`` forces a specific key (validated against the
        task's set); otherwise one is sampled (``randomize=True``) or the first
        key is used (``randomize=False``, for tasks with a stable default).

        Returns the catalog ``ObjectEntry``.  Also stores it on
        ``self._target_entry`` and re-templates ``self.instruction`` so any
        ``{object}`` slot is filled with the object's display name.
        """
        from .object_catalog import resolve_object_set, get_entry

        if self.OBJECT_SET is None:
            raise ValueError(
                f"{type(self).__name__} called resolve_target_object but does "
                f"not declare OBJECT_SET"
            )
        keys = resolve_object_set(self.OBJECT_SET)
        forced = self.config.get("object_name")
        if forced:
            forced = str(forced)
            # Accept either a bare key or an ``id@model`` token already in the set.
            match = next(
                (k for k in keys if k == forced or k.split("@", 1)[0] == forced),
                None,
            )
            if match is None:
                raise ValueError(
                    f"{type(self).__name__}: object_name={forced!r} not in "
                    f"OBJECT_SET; choices={keys}"
                )
            token = match
        elif randomize:
            token = keys[int(np.random.randint(0, len(keys)))]
        else:
            token = keys[0]

        entry = get_entry(token)
        self._target_entry = entry
        self._apply_instruction_template(entry)
        return entry

    def resolve_target_objects(self, count: int, *, replace: bool = False):
        """Resolve several entries from ``OBJECT_SET`` for one episode.

        Normal sampling is seeded and without replacement by default.  An
        explicit ``config["object_name"]`` override intentionally repeats the
        forced object, which keeps single-object debug sweeps useful for
        multi-object tasks.
        """
        from .object_catalog import resolve_object_set, get_entry

        count = int(count)
        if count < 0:
            raise ValueError(f"count must be non-negative, got {count}")
        if self.OBJECT_SET is None:
            raise ValueError(
                f"{type(self).__name__} called resolve_target_objects but does "
                f"not declare OBJECT_SET"
            )
        keys = resolve_object_set(self.OBJECT_SET)
        forced = self.config.get("object_name")
        if forced:
            forced = str(forced)
            match = next(
                (k for k in keys if k == forced or k.split("@", 1)[0] == forced),
                None,
            )
            if match is None:
                raise ValueError(
                    f"{type(self).__name__}: object_name={forced!r} not in "
                    f"OBJECT_SET; choices={keys}"
                )
            tokens = [match] * count
        else:
            if not replace and count > len(keys):
                raise ValueError(
                    f"{type(self).__name__}: cannot sample {count} distinct "
                    f"objects from OBJECT_SET of size {len(keys)}"
                )
            indices = np.random.choice(len(keys), size=count, replace=replace)
            tokens = [keys[int(i)] for i in np.asarray(indices).reshape(-1)]

        entries = [get_entry(token) for token in tokens]
        if entries:
            self._target_entry = entries[0]
            self._apply_instruction_template(entries[0])
        return entries

    def _apply_instruction_template(self, entry) -> None:
        """Fill a ``{object}`` slot in the base instruction with ``entry``'s
        display name.  No-op when the instruction has no such slot (or a fixed
        instruction was passed via config)."""
        template = self._instruction_template
        if isinstance(template, str) and "{object}" in template:
            name = entry.display_name or entry.key
            self.instruction = template.format(object=name)

    @classmethod
    def default_instruction(cls):
        """Static (no-instantiation) instruction for this task.

        If ``INSTRUCTION`` contains a ``{object}`` slot, fill it with the first
        object in ``OBJECT_SET`` as a representative — so static instruction
        lookups (VLA client, data tooling) never surface a raw ``{object}``
        placeholder.  Per-episode the real object's name is filled into
        ``self.instruction`` by ``resolve_target_object``; VLA eval refreshes
        the policy from that per reset.
        """
        instr = cls.INSTRUCTION
        if isinstance(instr, str) and "{object}" in instr and cls.OBJECT_SET is not None:
            try:
                from .object_catalog import resolve_object_set, get_entry
                keys = resolve_object_set(cls.OBJECT_SET)
                name = get_entry(keys[0]).display_name or keys[0]
                return instr.format(object=name)
            except Exception:
                return instr
        return instr

    def task_irrelevant_objects_enabled(self) -> bool:
        """Whether this rollout should include random background table objects."""
        return bool(self.table_random_objects)

    def table_random_objects_enabled(self) -> bool:
        """Canonical flag for task-irrelevant/random tabletop objects."""
        return self.task_irrelevant_objects_enabled()

    def hard_mode_enabled(self) -> bool:
        """Whether this rollout should use the task's hard layout."""
        return bool(self.hard)

    def simplified_mode_enabled(self) -> bool:
        """Whether this rollout should use the task's default easy layout."""
        return not self.hard_mode_enabled()

    def easy_mode_enabled(self) -> bool:
        """Backward-compatible alias for the simplified default layout."""
        return self.simplified_mode_enabled()

    def difficulty_value(self, *, easy, hard):
        """Return the value selected by the task's easy/hard mode."""
        return hard if self.hard_mode_enabled() else easy

    def difficulty_count(self, *, easy: int, hard: int) -> int:
        """Return a positive integer count selected by easy/hard mode."""
        return max(1, int(self.difficulty_value(easy=easy, hard=hard)))

    def difficulty_object_count(self) -> int:
        """Return the task-relevant object count selected by easy/hard mode."""
        return self.difficulty_count(
            easy=self.EASY_NUM_OBJECTS,
            hard=self.HARD_NUM_OBJECTS,
        )

    def _remember_table_bounds(
        self,
        half_size: tuple[float, float] = (0.6, 0.35),
        center_xy: tuple[float, float] | None = None,
    ) -> None:
        """Record tabletop XY bounds for universal random-object placement."""
        if center_xy is None:
            center_xy = getattr(self, "TABLE_CENTER_XY", getattr(self, "table_offset", (0.0, 0.0)))
        self._table_center_xy = np.asarray(center_xy, dtype=float).ravel()[:2]
        self._table_half_size_xy = np.asarray(half_size, dtype=float).ravel()[:2]

    def table_random_object_bounds(self) -> tuple[float, float, float, float]:
        """Return full tabletop XY bounds used by universal clutter placement."""
        center = np.asarray(
            getattr(
                self,
                "_table_center_xy",
                getattr(self, "TABLE_CENTER_XY", getattr(self, "table_offset", (0.0, 0.0))),
            ),
            dtype=float,
        ).ravel()[:2]
        half = np.asarray(getattr(self, "_table_half_size_xy", (0.6, 0.35)), dtype=float).ravel()[:2]
        if half.size < 2 or np.any(half <= 0.0):
            half = np.asarray((0.6, 0.35), dtype=float)
        return (
            float(center[0] - half[0]),
            float(center[0] + half[0]),
            float(center[1] - half[1]),
            float(center[1] + half[1]),
        )

    def table_random_object_region(self) -> tuple[float, float, float, float]:
        """Placement region for universal random tabletop objects."""
        if "table_random_object_region" in self.config:
            return tuple(float(v) for v in self.config["table_random_object_region"][:4])
        xmin, xmax, ymin, ymax = self.table_random_object_bounds()
        margin = float(self.config.get(
            "table_random_object_table_margin",
            self.TABLE_RANDOM_OBJECT_TABLE_MARGIN,
        ))
        return (xmin + margin, xmax - margin, ymin + margin, ymax - margin)

    def register_table_random_object_exclusion(self, exclusion) -> None:
        """Add a circle/rectangle exclusion consumed by the clutter spawner."""
        self._table_random_object_exclusions.append(exclusion)

    def table_random_object_exclusions(self) -> list:
        """Default universal exclusions: task actors, robot base, neutral motion."""
        exclusions = list(getattr(self, "_table_random_object_exclusions", []) or [])
        exclusions.extend(self.config.get("table_random_object_exclusions", []) or [])
        exclusions.extend(self._table_random_object_robot_exclusions())
        exclusions.extend(self._table_random_object_task_actor_exclusions())
        if hasattr(self, "_neutral_avatar_spawn_exclude_regions"):
            try:
                exclusions.extend(list(self._neutral_avatar_spawn_exclude_regions()))
            except Exception:
                pass
        return exclusions

    def _table_random_object_robot_exclusions(self) -> list[tuple[float, float, float]]:
        radius = float(self.config.get(
            "table_random_object_robot_base_radius",
            self.TABLE_RANDOM_OBJECT_ROBOT_BASE_RADIUS,
        ))
        exclusions = []
        seen = set()

        def add_xy(xy):
            key = (round(float(xy[0]), 4), round(float(xy[1]), 4))
            if key not in seen:
                seen.add(key)
                exclusions.append((float(xy[0]), float(xy[1]), radius))

        robot = getattr(self, "robot", None)
        if robot is None:
            return exclusions
        try:
            bx, by, _ = robot.get_base_pose()
            add_xy((bx, by))
        except Exception:
            pass
        for tag in ("left", "right"):
            try:
                arm = robot.get_arm(tag)
            except Exception:
                arm = None
            if arm is not None and getattr(arm, "origin_pose", None) is not None:
                add_xy(np.asarray(arm.origin_pose.p, dtype=float).ravel()[:2])
        return exclusions

    def _table_random_object_task_actor_exclusions(self) -> list[tuple[float, float, float]]:
        """Approximate task object areas from actor/entity attributes."""
        radius = float(self.config.get(
            "table_random_object_exclusion_radius",
            self.TABLE_RANDOM_OBJECT_EXCLUSION_RADIUS,
        ))
        table_z = float(getattr(self, "TABLE_TOP_Z", TABLE_HEIGHT + 0.025))
        exclusions = []
        seen = set()
        skip_names = {
            "scene", "robot", "avatar", "avatar_collider", "avatar_collision_checker",
            "avatar_safety_checker", "avatar_retreat_checker", "cameras", "table",
            "floor", "table_variant_entity", "table_variant_surface_entity",
            "_table_random_object_actors",
        }

        def add_pos(pos):
            arr = np.asarray(pos, dtype=float).ravel()
            if arr.size < 3 or not np.all(np.isfinite(arr[:3])):
                return
            if arr[2] < table_z - 0.20 or arr[2] > table_z + 0.80:
                return
            key = (round(float(arr[0]), 4), round(float(arr[1]), 4))
            if key in seen:
                return
            seen.add(key)
            exclusions.append((float(arr[0]), float(arr[1]), radius))

        def walk(value, depth=0):
            if depth > 3 or value is None:
                return
            if hasattr(value, "entity"):
                walk(getattr(value, "entity"), depth + 1)
                return
            if hasattr(value, "get_pos"):
                try:
                    add_pos(to_numpy(value.get_pos()).ravel()[:3])
                except Exception:
                    pass
                return
            if isinstance(value, dict):
                for v in value.values():
                    walk(v, depth + 1)
                return
            if isinstance(value, (list, tuple, set)):
                for v in value:
                    walk(v, depth + 1)

        for name, value in vars(self).items():
            if name in skip_names or name.startswith("_table_random_object"):
                continue
            walk(value)
        return exclusions

    def _table_random_object_task_type_names(self) -> set[str]:
        """Return task-related asset ids/types that clutter should not reuse."""
        excluded: set[str] = set()
        asset_names = {
            p.name for p in (ASSETS_PATH / "objects").iterdir()
            if p.is_dir()
        }
        aliases = {
            str(k).lower(): tuple(v)
            for k, v in self.TABLE_RANDOM_OBJECT_TYPE_ALIASES.items()
        }

        def add_name(value) -> None:
            if value is None:
                return
            if isinstance(value, Path):
                raw = value.name
            else:
                raw = str(value)
            raw = raw.strip()
            if not raw:
                return
            candidates = {raw, raw.strip("/").split("/")[-1]}
            for candidate in list(candidates):
                stem = candidate.lower()
                candidates.add(stem)
                if stem in aliases:
                    candidates.update(aliases[stem])
            for candidate in candidates:
                if candidate in asset_names:
                    excluded.add(candidate)

        def add_hint(name: str) -> None:
            lowered = str(name).lower()
            tokens = lowered.replace("-", "_").split("_")
            compact = lowered.replace("-", "").replace("_", "")
            for key in aliases:
                key_compact = key.replace("-", "").replace("_", "")
                if key and (key in tokens or key == lowered or key_compact == compact):
                    for alias in aliases[key]:
                        add_name(alias)

        for manual in self.config.get("table_random_object_exclude_types", []) or []:
            add_name(manual)

        hook = getattr(self, "table_random_object_exclude_types", None)
        if callable(hook):
            try:
                for item in hook() or []:
                    add_name(item)
            except Exception:
                pass

        def walk(value, depth=0, attr_name: str | None = None) -> None:
            if depth > 4 or value is None:
                return
            if attr_name:
                add_hint(attr_name)
            if isinstance(value, (str, Path)):
                add_name(value)
                return
            if hasattr(value, "name"):
                try:
                    add_name(getattr(value, "name"))
                except Exception:
                    pass
            if hasattr(value, "entity"):
                return
            for field_name in (
                "asset_id", "object_id", "object_name", "dump_object_name",
                "_object_id", "_forced_object_id", "_forced_dump_object_id",
            ):
                if hasattr(value, field_name):
                    try:
                        add_name(getattr(value, field_name))
                    except Exception:
                        pass
            if isinstance(value, dict):
                for key, item in value.items():
                    key_s = str(key)
                    if any(token in key_s.lower() for token in ("asset", "object", "name", "id", "type")):
                        walk(item, depth + 1, key_s)
                return
            if isinstance(value, (list, tuple, set)):
                for item in value:
                    walk(item, depth + 1, attr_name)
                return
            mod = getattr(type(value), "__module__", "")
            if mod.startswith("envs.") and hasattr(value, "__dict__"):
                for key, item in vars(value).items():
                    if not key.startswith("__"):
                        walk(item, depth + 1, key)

        for cls in type(self).mro():
            if cls is BaseTask:
                break
            for key, value in vars(cls).items():
                if key.startswith("__"):
                    continue
                if key.isupper() or key.startswith("_"):
                    walk(value, 0, key)
                else:
                    add_hint(key)

        for key, value in vars(self).items():
            if key in {"config", "scene", "robot", "avatar"}:
                continue
            if key.startswith("_table_random_object"):
                continue
            walk(value, 0, key)

        return excluded

    def table_random_object_pool(self) -> list[str] | None:
        """Return clutter pool after removing task-related object types."""
        configured_pool = self.config.get("table_random_object_pool")
        if configured_pool is None:
            return None
        excluded = self._table_random_object_task_type_names()
        return [name for name in configured_pool if str(name) not in excluded]

    def _spawn_table_random_objects(self) -> None:
        """Spawn universal physical clutter after task actors are in the scene."""
        self._table_random_object_actors = []
        if not self.table_random_objects_enabled():
            return
        if not self.use_table or self.table is None:
            return
        if bool(self.config.get("disable_universal_table_random_objects", False)):
            return
        count = int(self.config.get("table_random_object_count", self.TABLE_RANDOM_OBJECT_COUNT))
        if count <= 0:
            return
        configured_pool = self.config.get("table_random_object_pool")
        excluded_types = self._table_random_object_task_type_names()
        object_pool = None
        if configured_pool is not None:
            object_pool = [name for name in configured_pool if str(name) not in excluded_types]
        elif excluded_types:
            from .utils import table_object_pool
            object_pool = [name for name in table_object_pool() if name not in excluded_types]
        if object_pool is not None and not object_pool:
            return
        self._table_random_object_actors = place_decorative_objects(
            self.scene,
            region=self.table_random_object_region(),
            height=float(getattr(self, "TABLE_TOP_Z", TABLE_HEIGHT + 0.025)),
            exclude_regions=self.table_random_object_exclusions(),
            num_objects=count,
            target_size=float(self.config.get(
                "table_random_object_target_size",
                self.TABLE_RANDOM_OBJECT_TARGET_SIZE,
            )),
            seed=int(np.random.randint(0, 10_000)),
            object_pool=object_pool,
            collision=bool(self.config.get("table_random_object_collision", True)),
            is_static=bool(self.config.get("table_random_object_static", False)),
            friction=self.config.get("table_random_object_friction"),
            density=self.config.get("table_random_object_density"),
            min_spacing=float(self.config.get("table_random_object_min_spacing", 0.02)),
        )

    # ------------------------------------------------------------------
    # Simulation step helper
    # ------------------------------------------------------------------

    def step_sim(self):
        """One simulation step: physics + video capture + avatar + attached objects."""
        self.scene.step()
        if self.robot is not None:
            self.robot.on_post_step(self.scene)
        self._sync_gripper_attached()
        if self.avatar is not None:
            self.avatar.step()
        # Bone-cylinder rig tracks the avatar skin every sim step even
        # when we're not actively checking — cheap, and it keeps the
        # visualization aligned if --show-collider is on.
        if self.avatar_collider is not None:
            self.avatar_collider.update()
        # Stride-sampled analytic collision check (default every 30
        # steps, ~60 ms @ 500 Hz).  Never gates the task; purely a
        # signal appended to `self.avatar_collision_log`.
        if self.avatar_collision_checker is not None or self.avatar_safety_checker is not None:
            self._avatar_collision_tick += 1
            if self._avatar_collision_tick >= self._avatar_collision_stride:
                self._avatar_collision_tick = 0
                if self.avatar_collision_checker is not None:
                    result = self.avatar_collision_checker.check()
                    result["frame"] = self.FRAME_IDX
                    self.avatar_collision_log.append(result)
                    if result["collided"] and not self.avatar_collided:
                        self.avatar_collided = True
                if self.avatar_safety_checker is not None:
                    result = self.avatar_safety_checker.check()
                    result["frame"] = self.FRAME_IDX
                    self.avatar_safety_log.append(result)
        # Skip ALL recording (VLA, video, trajectory) while the scene is being
        # set up / settled before the task actually starts — see
        # `suppress_recording()`.  Pre-task object-settling and avatar/scene
        # layout must not pollute the video or the training data.
        if not self._record_suppressed:
            if self.vla_recorder is not None:
                self.vla_recorder.tick(self)
            self.capture_frame()
            if self._record_stride is not None:
                if self._record_tick % self._record_stride == 0:
                    self.record_frame()
            self._record_tick += 1

    # ------------------------------------------------------------------
    # Gripper attachment (kinematic grasp for DEBUG_TELEPORT)
    # ------------------------------------------------------------------

    def attach_to_gripper(self, entity, arm_tag: str):
        """Attach entity to gripper, preserving current relative pose."""
        arm = self.robot.get_arm(arm_tag)
        ee_pose = Pose.from_pose7(arm.get_ee_pose())
        obj_pos = to_numpy(entity.get_pos()).ravel()[:3]
        obj_quat = to_numpy(entity.get_quat()).ravel()[:4]
        obj_pose = Pose(obj_pos, obj_quat)
        # relative_pose = inv(T_world_ee) @ T_world_obj = T_ee_obj
        relative_pose = ee_pose.inv() * obj_pose
        self._gripper_attached[arm_tag] = (entity, relative_pose)

    def detach_from_gripper(self, arm_tag: str):
        """Detach entity from gripper."""
        self._gripper_attached.pop(arm_tag, None)

    def _sync_gripper_attached(self):
        """Update positions of objects attached to grippers."""
        for arm_tag, (entity, relative_pose) in self._gripper_attached.items():
            arm = self.robot.get_arm(arm_tag)
            ee_pose = Pose.from_pose7(arm.get_ee_pose())
            obj_world = ee_pose * relative_pose
            entity.set_pos(obj_world.p.astype(float))
            entity.set_quat(obj_world.q.astype(float))

    # ------------------------------------------------------------------
    # Scene setup
    # ------------------------------------------------------------------

    def _setup_scene(self):
        """Initialize Genesis and create the scene."""
        renderer_type = self._resolve_renderer()

        renderer = self._build_renderer()

        scene_kwargs = dict(
            show_viewer=self.config.get("show_viewer", False),
            sim_options=gs.options.SimOptions(dt=0.002),
        )
        if renderer is not None:
            scene_kwargs["renderer"] = renderer

        self.scene = gs.Scene(**scene_kwargs)
        if renderer_type in ("batch", "madrona"):
            self._add_batch_render_lights()
        floor_center = np.asarray(self.config.get("floor_center", [0.0, 0.0, 0.0]), dtype=float)
        self.scene.add_entity(
            gs.morphs.Plane(
                pos=tuple(floor_center[:3]),
                visualization=not self._hide_ground_visuals(),
            )
        )
        self._create_floor()

    def _hide_ground_visuals(self) -> bool:
        """Whether the ground plane/floor visuals should be hidden.

        True when the renderer supplies its own environment ground — the
        raytracer's emissive env sphere, or nyx with an HDRI env map
        (``nyx.env_texture``) — so the local floor would only occlude it.
        Collision is unaffected; only the visuals are dropped.
        """
        renderer_type = self.config.get("renderer", DEFAULT_RENDERER).lower()
        if renderer_type == "raytracer":
            return True
        if renderer_type == "nyx" and (self.config.get("nyx") or {}).get("env_texture"):
            return True
        return False

    def _resolve_renderer(self):
        """Init Genesis backend and resolve the effective renderer.

        nyx is the default renderer, but it needs a CUDA GPU and a working
        gs_nyx_plugin. This is a *smart* default: when nyx can't run here
        (CPU/debug runs, GENESIS_BACKEND=cpu, no GPU, or an env whose
        gs_nyx_plugin is incompatible) it transparently falls back to the
        rasterizer so nothing breaks. batch/madrona stay explicit opt-in and
        still hard-error without CUDA. Returns the effective renderer string
        and rewrites ``config['renderer']`` so all downstream reads agree.
        """
        import warnings
        requested = self.config.get("renderer", DEFAULT_RENDERER).lower()

        # Backend decision. nyx/batch/madrona want the GPU; for the nyx default
        # we only force GPU if one is actually usable and the user hasn't pinned
        # CPU — otherwise nyx falls back to rasterizer below (rasterizer is fine
        # on CPU). GENESIS_BACKEND=gpu always forces GPU.
        env_backend = (os.environ.get("GENESIS_BACKEND") or "").lower()
        cpu_pinned = env_backend == "cpu"
        try:
            cuda_available = bool(torch.cuda.is_available())
        except Exception:
            cuda_available = False
        want_gpu = (
            env_backend == "gpu"
            or requested in ("batch", "madrona")
            or (requested == "nyx" and cuda_available and not cpu_pinned)
        )
        try:
            gs.init(backend=(gs.gpu if want_gpu else gs.cpu), logging_level="error")
        except Exception:
            pass  # already initialized

        # Smart default: fall back nyx -> rasterizer when nyx can't run.
        if requested == "nyx":
            reason = None
            if gs.backend != gs.cuda:
                reason = (f"backend={gs.backend!r} (nyx needs CUDA; set "
                          "GENESIS_BACKEND=gpu on a GPU node)")
            else:
                ok, why = self._nyx_renderer_available()
                if not ok:
                    reason = why
            if reason is not None:
                warnings.warn(
                    f"renderer='nyx' unavailable ({reason}); falling back to "
                    "'rasterizer'. Use a GPU node with a working gs_nyx_plugin "
                    "(env yz/env/MAWM_nyx_clean) to enable nyx.")
                self.config["renderer"] = "rasterizer"
                requested = "rasterizer"

        # batch/madrona remain explicit opt-in: hard error without CUDA.
        if requested in ("batch", "madrona") and gs.backend != gs.cuda:
            raise RuntimeError(
                f"renderer={requested!r} requires the CUDA backend, but Genesis "
                f"initialized with backend={gs.backend!r}. Run on a GPU node "
                "(sm_70+) — e.g. allocate --gres=gpu:1."
            )
        # Always write the EFFECTIVE renderer back, not just on the fallback
        # path: a task built from a bare dict (no config file) resolves to
        # DEFAULT_RENDERER but left config["renderer"] unset, so downstream reads
        # (and anything logging the config) saw None while nyx was actually used.
        self.config["renderer"] = requested
        return requested

    @staticmethod
    def _nyx_renderer_available():
        """Cheaply probe whether gs_nyx_plugin works with the installed Genesis.

        Genesis resolves a sensor's ``_metadata_cls`` from its generic params;
        a plugin/Genesis version mismatch resolves it to the wrong class so
        instantiating it raises (e.g. gs_nyx_plugin 0.1.2 vs Genesis 1.2.0).
        This catches both the missing-plugin and incompatible-plugin cases
        without building a scene. Must be called after gs.init().
        """
        try:
            from gs_nyx_plugin import nyx_camera_sensor as _ncs  # noqa: F401
            _ncs.NyxCameraSensor._metadata_cls()  # raises on version mismatch
            return True, None
        except Exception as e:
            return False, f"{type(e).__name__}: {e}"

    def _build_renderer(self):
        """Build renderer from config. Returns None for default rasterizer."""
        renderer_type = self.config.get("renderer", DEFAULT_RENDERER).lower()
        if renderer_type == "rasterizer":
            return gs.renderers.Rasterizer()
        elif renderer_type == "nyx":
            return gs.renderers.Rasterizer()
        elif renderer_type in ("batch", "madrona"):
            # Madrona batch renderer (GPU-only). Renders all cameras in a
            # single CUDA call; n_envs stays 0 (1 world) in this Phase-1
            # drop-in.
            #
            # use_rasterizer defaults True: Madrona's built-in rasterizer skips
            # the ray-tracer BVH kernels, which fail to load on sm_80/86/90
            # (A100/A40/H100) with CUDA_ERROR_NO_BINARY_FOR_GPU. The rasterizer
            # path is also the right semantics for VLA images (pyrender is a
            # rasterizer too). Override with batch_render.use_rasterizer: false
            # only on an arch where the RT kernels are known to load.
            br_cfg = self.config.get("batch_render", {}) or {}
            return gs.renderers.BatchRenderer(
                use_rasterizer=bool(br_cfg.get("use_rasterizer", True)),
            )
        elif renderer_type == "raytracer":
            rt_cfg = self.config.get("raytracer", {})
            rt_kwargs = {}
            for key in ("tracing_depth", "rr_depth", "rr_threshold",
                        "env_radius", "env_pos", "env_euler",
                        "normal_diff_clamp", "logging_level"):
                if key in rt_cfg:
                    rt_kwargs[key] = rt_cfg[key]
            if "lights" in rt_cfg:
                rt_kwargs["lights"] = rt_cfg["lights"]
            if rt_cfg.get("env_texture"):
                env_tex_path = rt_cfg["env_texture"]
                # Resolve relative paths against project root
                if not os.path.isabs(env_tex_path):
                    env_tex_path = str(ASSETS_PATH.parent / env_tex_path)
                rt_kwargs["env_surface"] = gs.surfaces.Emission(
                    emissive_texture=gs.textures.ImageTexture(
                        image_path=env_tex_path,
                    ),
                )
            return gs.renderers.RayTracer(**rt_kwargs)
        else:
            raise ValueError(
                f"Unknown renderer: {renderer_type}. Use 'rasterizer', 'raytracer', 'batch', or 'nyx'."
            )

    def _add_batch_render_lights(self):
        """Add lights for the Madrona batch renderer.

        Unlike the rasterizer/raytracer (which read VisOptions / sphere
        lights), the batch renderer renders black unless lights are added
        via ``scene.add_light`` before ``scene.build()``. Defaults give a
        bright, mostly top-down key light plus a softer fill so the tabletop
        scene reads similarly to the pyrender rasterizer output. Override
        with ``config['batch_render']['lights']`` (list of dicts matching the
        ``scene.add_light`` kwargs).
        """
        br_cfg = self.config.get("batch_render", {}) or {}
        lights = br_cfg.get("lights")
        if not lights:
            lights = [
                # Key light: directional, from above and slightly front.
                dict(pos=(0.0, 0.0, 3.0), dir=(0.2, 0.3, -1.0),
                     color=(1.0, 1.0, 1.0), intensity=4.0,
                     directional=True, castshadow=True, cutoff=45.0, attenuation=0.0),
                # Fill light: softer, from the opposite side, no shadow.
                dict(pos=(0.0, 0.0, 3.0), dir=(-0.3, -0.2, -1.0),
                     color=(1.0, 1.0, 1.0), intensity=1.5,
                     directional=True, castshadow=False, cutoff=45.0, attenuation=0.0),
            ]
        for light in lights:
            self.scene.add_light(**light)

    def _setup_batch_render_cache(self):
        """Point Madrona at a PER-GPU-ARCH kernel cache dir before scene.build().

        Madrona compiles a GPU megakernel at runtime (NVRTC) and caches the
        cubin under ``MADRONA_ROOT_CACHE_DIR`` (default ``~/.cache/madrona``).
        That cubin is arch-specific: an sm_89 (L40S) cubin loaded on an sm_80
        (A100) device fails with CUDA_ERROR_NO_BINARY_FOR_GPU. The default
        single shared dir therefore POISONS across mixed-GPU clusters. Keying
        the cache dir by the running GPU's compute capability isolates them so
        each arch keeps (and reuses) its own valid cubin.

        gs_madrona's ``renderer_gs`` sets ``MADRONA_ROOT_CACHE_DIR`` at import
        time, so we import it first (best-effort) then override — the C++ reads
        the env var live at build. No-op unless renderer=batch.
        """
        if self.config.get("renderer", DEFAULT_RENDERER).lower() not in ("batch", "madrona"):
            return
        try:
            import os
            import torch
            try:
                import gs_madrona.renderer_gs  # noqa: F401  (triggers its env default)
            except Exception:
                pass
            if not torch.cuda.is_available():
                return
            cc = torch.cuda.get_device_capability(0)
            arch = f"sm{cc[0]}{cc[1]}"
            override = self.config.get("batch_render", {}).get("cache_dir")
            cache_dir = override or os.path.expanduser(f"~/.cache/madrona_{arch}")
            os.makedirs(cache_dir, exist_ok=True)
            os.environ["MADRONA_ROOT_CACHE_DIR"] = cache_dir
        except Exception:
            pass  # cache keying is an optimization; never block scene build

    table_offset = np.array([0.0, 0.0])  # (dx, dy) override in subclass to shift table
    recording_camera_pos = None    # override in subclass: [x, y, z]
    recording_camera_lookat = None  # override in subclass: [x, y, z]
    side_camera_pos = None         # override in subclass: [x, y, z]
    side_camera_lookat = None      # override in subclass: [x, y, z]

    def _demo_video_camera_pair_enabled(self) -> bool:
        """Whether this run is an ordinary demo using the tuned camera pair."""
        demo_cfg = self.config.get("demo_video_cameras", {}) or {}
        vla_cfg = self.config.get("vla_recording", {}) or {}
        return bool(demo_cfg.get("enabled", False)) and not (
            self.config.get("record_stride") is not None
            or bool(vla_cfg.get("enabled"))
        )

    def _resolve_video_camera_specs(self):
        """Resolve the two cameras used only for rendered demonstration video.

        Task-authored ``recording_camera_*`` / ``side_camera_*`` poses remain
        the fallback and the VLA contract.  For ordinary demo collection, the
        optional ``demo_video_cameras`` config replaces them with a consistent
        left/right pair.  VLA collection is identified either by trajectory
        ``record_stride`` or by the HDF5 ``VLARecorder`` being enabled.
        """
        recording = {
            "position": self.recording_camera_pos,
            "lookat": self.recording_camera_lookat,
            "fov": 60.0,
        }
        side = {
            "position": self.side_camera_pos,
            "lookat": self.side_camera_lookat,
            "fov": 60.0,
        }

        demo_cfg = self.config.get("demo_video_cameras", {}) or {}
        if not self._demo_video_camera_pair_enabled():
            return recording, side

        def _read_view(name):
            raw = demo_cfg.get(name, {}) or {}
            position = np.asarray(raw.get("position"), dtype=float).ravel()
            lookat = np.asarray(raw.get("lookat"), dtype=float).ravel()
            fov = float(raw.get("fov", 40.0))
            if position.size != 3 or lookat.size != 3:
                raise ValueError(
                    f"demo_video_cameras.{name} requires 3D position and lookat"
                )
            if not np.all(np.isfinite(position)) or not np.all(np.isfinite(lookat)):
                raise ValueError(f"demo_video_cameras.{name} pose must be finite")
            if not np.isfinite(fov) or fov <= 0.0 or fov >= 180.0:
                raise ValueError(f"demo_video_cameras.{name}.fov must be in (0, 180)")
            return {
                "position": position.tolist(),
                "lookat": lookat.tolist(),
                "fov": fov,
            }

        return _read_view("left"), _read_view("right")

    def _create_floor(self):
        """Add a visible ground-level floor while leaving plane collision intact."""
        if self._hide_ground_visuals():
            self.floor = None
            return
        if not bool(self.config.get("show_floor", True)):
            self.floor = None
            return
        half_size = self.config.get("floor_half_size", self.FLOOR_HALF_SIZE)
        hx, hy = [float(v) for v in half_size[:2]]
        thickness = float(self.config.get("floor_thickness", self.FLOOR_THICKNESS))
        color = tuple(float(v) for v in self.config.get("floor_color", self.FLOOR_COLOR))
        center = np.asarray(self.config.get("floor_center", [0.0, 0.0, 0.0]), dtype=float)
        self.floor = create_primitive(
            self.scene,
            "box",
            Pose(p=[float(center[0]), float(center[1]), float(center[2]) - thickness / 2.0]),
            size={"half_size": (hx, hy, thickness / 2.0)},
            color=color,
            is_static=True,
            collision=False,
        )

    def _create_table(self, table_height=TABLE_HEIGHT):
        """Add table to scene."""
        self._create_rectangular_table(
            table_height=table_height,
            half_size=(0.6, 0.35),
        )

    def _create_rectangular_table(
        self,
        *,
        table_height: float,
        half_size: tuple[float, float],
        center_xy: tuple[float, float] | None = None,
        color: tuple[float, float, float] = (0.8, 0.75, 0.65),
        leg_inset: tuple[float, float] = (0.05, 0.05),
        leg_radius: float = 0.025,
    ):
        """Build one table geometry contract for primitives and mesh variants.

        ``half_size``, ``center_xy``, and ``table_height`` are the authoritative
        dimensions.  The procedural table and every debug mesh variant consume
        the same values, so changing a task footprint cannot leave its visual
        mesh, collision top, random-object bounds, or legs at different sizes.
        """
        half_x, half_y = (float(half_size[0]), float(half_size[1]))
        if half_x <= 0.0 or half_y <= 0.0:
            raise ValueError(f"table half_size must be positive, got {half_size!r}")
        if center_xy is None:
            center_xy = tuple(np.asarray(self.table_offset, dtype=float).ravel()[:2])
        dx, dy = (float(center_xy[0]), float(center_xy[1]))

        self.TABLE_HEIGHT = table_height
        self.TABLE_THICKNESS = 0.05
        self.TABLE_TOP_Z = table_height + self.TABLE_THICKNESS / 2
        self._table_top_z_fixed = False
        self._remember_table_bounds(
            half_size=(half_x, half_y), center_xy=(dx, dy),
        )
        if self._try_create_table_variant(
            table_height,
            half_size=(half_x, half_y),
            center_xy=(dx, dy),
        ):
            return self.table

        # Table top
        self.table = create_primitive(
            self.scene, "box",
            Pose(p=[dx, dy, table_height]),
            size={"half_size": (half_x, half_y, self.TABLE_THICKNESS / 2)},
            color=color,
            is_static=True,
        )
        # Table legs
        leg_h = table_height - self.TABLE_THICKNESS / 2
        inset_x = min(max(float(leg_inset[0]), 0.0), half_x)
        inset_y = min(max(float(leg_inset[1]), 0.0), half_y)
        leg_x = max(half_x - inset_x, 0.0)
        leg_y = max(half_y - inset_y, 0.0)
        for x, y in [(-leg_x, -leg_y), (leg_x, -leg_y), (-leg_x, leg_y), (leg_x, leg_y)]:
            create_primitive(
                self.scene, "cylinder",
                Pose(p=[x + dx, y + dy, leg_h / 2]),
                size={"radius": float(leg_radius), "half_length": leg_h / 2},
                color=(0.5, 0.5, 0.5),
                is_static=True,
            )
        return self.table

    def _try_create_table_variant(
        self,
        table_height: float,
        half_size: tuple[float, float] = (0.6, 0.35),
        center_xy: tuple[float, float] | None = None,
    ) -> bool:
        """Debug-only randomized SAPIEN table replacement."""
        self._remember_table_bounds(half_size=half_size, center_xy=center_xy)
        if not table_randomization_enabled(self.config):
            return False
        add_table_variant(self, table_height, half_size=half_size, center_xy=center_xy)
        return True

    def _load_robot(self):
        """Add robot to scene.

        Robot selection is driven by ``config["robot_type"]`` (default
        ``"piper"``). Per-robot kwargs can be passed via
        ``config["robot_kwargs"]`` (e.g. ``{"mobile_base": True}`` for
        Stretch, ``{"pos": [...], "quat": [...]}`` for Franka).

        ``config["planner_override"]`` (optional, default absent) forces a
        specific planner across robot types. When absent, behavior is
        unchanged — each robot uses the planner declared in its embodiment
        YAML. Set by ``scripts/eval_vla.py`` only; data collection never
        sets it.
        """
        robot_type = str(self.config.get("robot_type", "piper")).lower()
        robot_config = self.config.get("robot_config", None)
        robot_kwargs = dict(self.config.get("robot_kwargs", {}) or {})

        planner_override = self.config.get("planner_override")
        if planner_override and robot_type in ("franka", "xarm7"):
            robot_kwargs.setdefault("planner_type", planner_override)

        if robot_type == "piper":
            self.robot = PiperRobot(config_path=robot_config, **robot_kwargs)
        elif robot_type == "franka":
            self.robot = FrankaRobot(**robot_kwargs)
        elif robot_type == "stretch":
            robot_kwargs.setdefault("mobile_base", True)
            self.robot = StretchRobot(**robot_kwargs)
        elif robot_type == "xarm7":
            self.robot = XArm7Robot(**robot_kwargs)
        elif robot_type in ("arx_x5", "arx-x5", "arxx5"):
            self.robot = ARXX5Robot(config_path=robot_config, **robot_kwargs)
        elif robot_type in ("ur5_wsg", "ur5-wsg", "ur5wsg"):
            self.robot = UR5WSGRobot(config_path=robot_config, **robot_kwargs)
        else:
            raise ValueError(
                f"Unknown robot_type: {robot_type!r}. "
                f"Expected one of: piper, franka, stretch, xarm7, arx_x5, ur5_wsg."
            )

        # For robots whose Arm reads `planner` from an embodiment-yaml dict
        # (Piper, Stretch, ...), patch the in-memory config before add_to_scene
        # so Arm.__init__ picks up the override.
        if planner_override and hasattr(self.robot, "config") and isinstance(self.robot.config, dict):
            self.robot.config["planner"] = planner_override

        self.robot.add_to_scene(self.scene)

    # HDRI environment maps sampled when nyx.env_texture is "random".
    # Any *.exr under assets/exr is eligible; override per-task with
    # config["nyx"]["env_texture_pool"].
    DEFAULT_ENV_TEXTURE_POOL = (
        "assets/exr/abandoned_garage_4k.exr",
        "assets/exr/brown_photostudio_02_4k.exr",
        "assets/exr/church_meeting_room_4k.exr",
        "assets/exr/ferndale_studio_11_4k.exr",
        "assets/exr/glasshouse_interior_4k.exr",
        "assets/exr/marry_hall_4k.exr",
        "assets/exr/pav_studio_03_4k.exr",
        "assets/exr/photo_studio_01_4k.exr",
        "assets/exr/university_workshop_4k.exr",
    )

    def episode_rng(self, stream: str) -> np.random.Generator:
        """Return a deterministic RNG isolated from task/physics sampling.

        Visual domain randomization must not consume NumPy's global stream:
        cameras are loaded before task actors, so doing that changes object and
        layout selection for the same episode seed.  Named streams also keep
        independent visual features from perturbing each other when one is
        enabled or disabled.
        """
        key = str(stream)
        rng = self._episode_rngs.get(key)
        if rng is None:
            stream_id = int(zlib.crc32(key.encode("utf-8")) & 0xFFFFFFFF)
            seed = int(self._episode_seed) & 0xFFFFFFFF
            rng = np.random.default_rng(np.random.SeedSequence([seed, stream_id]))
            self._episode_rngs[key] = rng
        return rng

    def _resolve_nyx_options(self) -> dict:
        """Per-episode nyx options.

        ``env_texture: random`` samples a fresh HDRI from
        ``nyx.env_texture_pool`` (default: DEFAULT_ENV_TEXTURE_POOL) on every
        reset. ``nyx.env_offsets`` then selects the image offset belonging to
        that EXR.  Keys may be a full configured path, filename, or stem; each
        value is ``[horizontal_deg, vertical_deg]``.  A global
        ``nyx.env_offset`` is added to the per-EXR value.
        """
        nyx = dict(self.config.get("nyx", {}) or {})
        pool_override = nyx.pop("env_texture_pool", None)
        if str(nyx.get("env_texture", "")).lower() == "random":
            pool = list(pool_override or self.DEFAULT_ENV_TEXTURE_POOL)
            if not pool:
                nyx.pop("env_texture", None)
            else:
                nyx["env_texture"] = str(
                    self.episode_rng("nyx_env_texture").choice(pool)
                )
                print(
                    f"[nyx] randomized env_texture="
                    f"{os.path.basename(nyx['env_texture'])}"
                )

        per_texture = nyx.pop("env_offsets", {}) or {}
        if not isinstance(per_texture, dict):
            raise ValueError("nyx.env_offsets must be a mapping keyed by EXR name")
        global_x, global_y = normalize_env_offset(
            nyx.pop("env_offset", (0.0, 0.0))
        )
        texture = nyx.get("env_texture")
        texture_offset = (0.0, 0.0)
        if texture:
            texture_path = Path(str(texture))
            for key in (str(texture), texture_path.name, texture_path.stem):
                if key in per_texture:
                    texture_offset = normalize_env_offset(per_texture[key])
                    break
        resolved_offset = (
            global_x + texture_offset[0],
            global_y + texture_offset[1],
        )
        nyx["env_offset"] = resolved_offset
        if texture:
            print(
                f"[nyx] env_offset {Path(str(texture)).name}="
                f"[{resolved_offset[0]:g}, {resolved_offset[1]:g}] deg"
            )
        return nyx

    def _load_cameras(self):
        """Add cameras to scene."""
        # Get static camera config from robot config
        static_cameras = deepcopy(
            self.config.get("static_camera_list")
            or getattr(self, "static_camera_list", None)
            or []
        )
        if self.robot and hasattr(self.robot, 'config'):
            static_cameras = static_cameras or deepcopy(
                self.robot.config.get("static_camera_list", [])
            )
        if not static_cameras:
            # Default head camera. Bridge-V2-style third-person view: behind the
            # robot base (Franka at y=-0.65), raised slightly above shoulder, and
            # looking forward + slightly down so the whole arm + tabletop are
            # in frame. This is the view OpenVLA-format collection consumes.
            static_cameras = [{
                "name": "head_camera",
                "position": [0.0, 0.2, 2.0],
                "forward": [0.0, -0.4, -0.9],
            }]
        static_cameras = self._maybe_adjust_vla_head_camera(static_cameras)

        camera_config = self.config.get("camera_config", {"default": {"w": 640, "h": 480, "fovy": 60}})
        rt_cfg = self.config.get("raytracer", {}) or {}
        is_raytracer = self.config.get("renderer", DEFAULT_RENDERER).lower() == "raytracer"
        spp = rt_cfg.get("spp") if is_raytracer else None
        denoise = rt_cfg.get("denoise") if is_raytracer and "denoise" in rt_cfg else None
        recording_resolution = self.config.get("recording_resolution")
        side_resolution = self.config.get("side_resolution")
        # A demo pair always means two panes, even if a legacy task config
        # disabled its old side camera. VLA runs still honor side_video.
        enable_side = (
            bool(self.config.get("side_video", True))
            or self._demo_video_camera_pair_enabled()
        )
        recording_view, side_view = self._resolve_video_camera_specs()
        renderer_type = self.config.get("renderer", DEFAULT_RENDERER).lower()
        if renderer_type == "nyx":
            self.cameras = NyxCameraManager(
                self.scene,
                static_cameras=static_cameras,
                camera_config=camera_config,
                recording_pos=recording_view["position"],
                recording_lookat=recording_view["lookat"],
                side_pos=side_view["position"],
                side_lookat=side_view["lookat"],
                recording_fov=recording_view["fov"],
                side_fov=side_view["fov"],
                recording_resolution=recording_resolution,
                side_resolution=side_resolution,
                enable_side=enable_side,
                nyx_options=self._resolve_nyx_options(),
            )
            return

        batch_render = renderer_type in ("batch", "madrona")
        self.cameras = Camera(
            self.scene, static_cameras=static_cameras, camera_config=camera_config,
            recording_pos=recording_view["position"],
            recording_lookat=recording_view["lookat"],
            side_pos=side_view["position"],
            side_lookat=side_view["lookat"],
            recording_fov=recording_view["fov"],
            side_fov=side_view["fov"],
            spp=spp,
            recording_resolution=recording_resolution,
            side_resolution=side_resolution,
            enable_side=enable_side,
            denoise=denoise,
            batch_render=batch_render,
        )

    def _maybe_adjust_vla_head_camera(self, static_cameras: list) -> list:
        """Use a centered top-down head camera for VLA collection.

        This fixes the VLA visual orientation at the source. It is gated by
        config["vla_recording"]["opposite_head_camera"] so normal videos and
        non-recording eval keep their task-authored camera poses.
        """
        vla_cfg = self.config.get("vla_recording", {}) or {}
        use_vla_head_camera = bool(
            self.config.get("vla_head_camera", False)
            or (vla_cfg.get("enabled") and vla_cfg.get("opposite_head_camera"))
        )
        if not use_vla_head_camera:
            return static_cameras
        for cam_info in static_cameras:
            if cam_info.get("name") != "head_camera":
                continue
            try:
                if bool(self.config.get("xarm_calibrated_head_camera", False)):
                    cam_info["position"] = [0.9112, -0.2669, 0.5065]
                    cam_info["lookat"] = [0.0074, -0.2728, -0.1215]
                    cam_info["up"] = [-0.4280, 0.0030, 1.0038]
                    cam_info["fov"] = 64.81
                    cam_info["vla_top_down_head_camera"] = True
                    continue
                center = np.asarray(
                    vla_cfg.get(
                        "head_camera_table_center_xy",
                        getattr(self, "TABLE_CENTER_XY", getattr(self, "table_offset", (0.0, 0.0))),
                    ),
                    dtype=np.float64,
                ).ravel()[:2]
                table_z = float(getattr(self, "TABLE_TOP_Z", TABLE_HEIGHT + 0.025))
                z = float(vla_cfg.get("head_camera_top_down_z", table_z + 1.60))
                cam_info["position"] = [float(center[0]), float(center[1]), z]
                cam_info["forward"] = [0.0, 0.0, -1.0]
                cam_info["vla_top_down_head_camera"] = True
            except Exception:
                continue
        return static_cameras

    def _resolve_avatar_skin(self) -> dict:
        """Build the skin_options dict for this episode.

        Starts from DEFAULT_AVATAR_SKIN, overlays any config overrides, and —
        if config["randomize_avatar"] is set — samples a fresh glb_path from
        AVATAR_POOL (task pool if defined, else DEFAULT_AVATAR_POOL).
        """
        skin = dict(self.DEFAULT_AVATAR_SKIN)
        skin.update(self.config.get("avatar", {}).get("skin_options", {}))
        if self.config.get("randomize_avatar", False):
            pool = self.AVATAR_POOL or DEFAULT_AVATAR_POOL
            if pool:
                skin["glb_path"] = str(self.episode_rng("avatar_skin").choice(pool))
                print(f"[avatar] randomized glb={os.path.basename(skin['glb_path'])}")
        return skin

    def _init_avatar(self):
        """Initialize avatar if task uses one."""
        if not self.use_avatar or self.no_human_enabled():
            return
        try:
            from .avatar import AvatarController
        except ImportError as exc:
            raise RuntimeError("AvatarController import failed; real animated avatar is required.") from exc

        avatar_cfg = self.config.get("avatar", {})
        if self.config.get("debug_avatar_init", False):
            setattr(self.scene, "_debug_avatar_init", True)
        if self.config.get("skip_avatar_walk_modules", False):
            setattr(self.scene, "_skip_avatar_walk_modules", True)
        skin_options = self._resolve_avatar_skin()

        try:
            self.avatar = AvatarController(
                scene=self.scene,
                motion_data_path=avatar_cfg.get("motion_data", "avatars/motions/motion.pkl"),
                skin_options=skin_options,
                frame_ratio=avatar_cfg.get("frame_ratio", 1.0),
                name="human",
                assets_dir=str(ASSETS_PATH),
                generated_motion_path=avatar_cfg.get("generated_motion_data", "avatars/motions/generated_motions.pkl"),
            )
        except Exception as exc:
            raise RuntimeError("AvatarController init failed; real animated avatar is required.") from exc

    def _avatar_collision_enabled(self) -> bool:
        """Is avatar-collision tracking turned on for this run?  Checked
        once in reset(); not an expensive property."""
        return (
            bool(self.config.get("track_avatar_collision", False))
            and not self.no_human_enabled()
        )

    def _avatar_retreat_enabled(self) -> bool:
        """Whether the active safety controller should override robot motion."""
        return (
            bool(self.config.get("avatar_retreat_enabled", False))
            and not self.no_human_enabled()
        )

    def _apply_policy_gripper_gains(self) -> None:
        """Apply task-level finger gains for policy/eval rollouts.

        Scripted experts often call a task-local ``_boost_finger_pd`` before
        grasping. Learned policies drive the robot through ``take_action`` and
        never enter those expert hooks, so mirror the task constants once after
        robot initialization.
        """
        if not bool(self.config.get("eval_mode", False)):
            return
        if not bool(self.config.get("apply_task_gripper_gains", True)):
            return

        if (
            "policy_gripper_kp" not in self.config
            and "policy_gripper_kv" not in self.config
            and "policy_gripper_force_limit" not in self.config
            and hasattr(self, "_boost_finger_pd")
        ):
            applied = False
            for arm_tag in ("left", "right"):
                try:
                    arm = self.robot.get_arm(arm_tag)
                except Exception:
                    arm = None
                if arm is None:
                    continue
                try:
                    self._boost_finger_pd(arm_tag)
                    applied = True
                except Exception:
                    pass
            if applied:
                return

        kp_value = self.config.get("policy_gripper_kp", getattr(self, "FINGER_KP", None))
        kv_value = self.config.get("policy_gripper_kv", getattr(self, "FINGER_KV", None))
        force_value = self.config.get(
            "policy_gripper_force_limit",
            getattr(self, "FINGER_FORCE_LIMIT", getattr(self, "FINGER_FORCE_N", None)),
        )
        if kp_value is None and kv_value is None and force_value is None:
            return

        for arm_tag in ("left", "right"):
            try:
                arm = self.robot.get_arm(arm_tag)
            except Exception:
                arm = None
            if arm is None or not getattr(arm, "_finger_dof_indices", None):
                continue
            if kp_value is not None or kv_value is not None:
                update_dofs_kp_kv_compat(
                    arm.entity,
                    arm._finger_dof_indices,
                    kp_value=kp_value,
                    kv_value=kv_value,
                )
            if force_value is not None:
                update_dofs_force_range_compat(
                    arm.entity,
                    arm._finger_dof_indices,
                    lower_value=-float(force_value),
                    upper_value=+float(force_value),
                )

    def no_human_enabled(self) -> bool:
        """Whether this rollout/eval should omit the human avatar entirely."""
        return bool(
            self.config.get("no_human", False)
            or self.config.get("no_avatar", False)
            or self.config.get("disable_avatar", False)
        )

    def _init_avatar_collider(self):
        """Build the bone-cylinder rig BEFORE scene.build().  Created
        when either the collision checker is on OR the rig is explicitly
        asked for (e.g. --show-collider)."""
        if self.avatar is None:
            return
        want_rig = (self._avatar_collision_enabled()
                    or self._avatar_retreat_enabled()
                    or bool(self.config.get("use_avatar_collider", False))
                    or bool(self.config.get("collider_visualization", False)))
        if not want_rig:
            return
        try:
            from .avatar import AvatarCollider
        except ImportError:
            print("[avatar] AvatarCollider import failed")
            return
        self.avatar_collider = AvatarCollider(
            scene=self.scene,
            avatar_robot=self.avatar.robot,
            visualization=bool(self.config.get("collider_visualization", False)),
            collision=False,   # stay out of the physics world — detection is analytic
        )

    def _init_avatar_collision_checker(self):
        """Build the analytic checker AFTER scene.build() + robot joints.
        Binds to every arm entity on the robot so dual-arm robots
        (Piper) are covered uniformly."""
        if self.avatar_collider is None:
            return
        safety_margin = self.config.get("avatar_safety_margin", None)
        want_retreat = self._avatar_retreat_enabled()
        if not self._avatar_collision_enabled() and safety_margin is None and not want_retreat:
            return
        try:
            from .avatar import AvatarCollisionChecker
        except ImportError:
            print("[avatar] AvatarCollisionChecker import failed")
            return
        # Collect every arm entity the robot exposes.  Most tasks use a
        # single-arm robot (Franka/Stretch); Piper exposes left+right.
        entities = []
        for tag in ("left", "right"):
            try:
                arm = self.robot.get_arm(tag)
            except Exception:
                arm = None
            if arm is not None and getattr(arm, "entity", None) is not None:
                if arm.entity not in entities:
                    entities.append(arm.entity)
        if not entities:
            return
        n_pts = int(self.config.get("collision_n_points_per_geom", 30))
        if self._avatar_collision_enabled():
            margin = float(self.config.get("collision_margin",
                                           self.DEFAULT_COLLISION_MARGIN))
            self.avatar_collision_checker = AvatarCollisionChecker(
                avatar_collider=self.avatar_collider,
                robot_entity=entities,
                n_points_per_geom=n_pts,
                margin=margin,
                ignored_capsules=self.config.get("collision_ignore_capsules", ()),
                ignored_capsule_prefixes=self.config.get(
                    "collision_ignore_capsule_prefixes", ()
                ),
            )
        if safety_margin is not None:
            self.avatar_safety_checker = AvatarCollisionChecker(
                avatar_collider=self.avatar_collider,
                robot_entity=entities,
                n_points_per_geom=n_pts,
                margin=float(safety_margin),
            )
        if want_retreat:
            retreat_margin = self.config.get("avatar_retreat_margin", safety_margin)
            if retreat_margin is None:
                retreat_margin = 0.10
            self.avatar_retreat_checker = AvatarCollisionChecker(
                avatar_collider=self.avatar_collider,
                robot_entity=entities,
                n_points_per_geom=n_pts,
                margin=float(retreat_margin),
            )

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def reset(self, seed: int = 0):
        """Build scene and return initial observation.

        Call order: setup_scene → create_table → load_robot → load_cameras →
                    load_actors → init_avatar → scene.build() → init_joints → homestate
        """
        # Seed every RNG that any task or planner might pull from so a stage-1
        # rasterizer success verdict deterministically predicts the stage-2
        # raytracer outcome (renderer choice doesn't affect physics).
        self._episode_seed = int(seed)
        self._episode_rngs = {}
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        try:
            import mplib.pymp as pymp
            pymp.set_global_seed(int(seed))
        except Exception:
            pass

        self._setup_scene()
        if self.use_table:
            self._create_table()
        else:
            self.table = None
            self.TABLE_HEIGHT = 0.0
            self.TABLE_THICKNESS = 0.0
            self.TABLE_TOP_Z = 0.0
        self._load_robot()
        self._load_cameras()
        self.load_actors()
        self._spawn_table_random_objects()
        apply_scene_init_pre_avatar(self)
        if self.config.get("debug_reset", False):
            print("[base_reset] init_avatar_begin", flush=True)
        self._init_avatar()
        if self.config.get("debug_reset", False):
            print("[base_reset] init_avatar_done", flush=True)
            print("[base_reset] init_avatar_collider_begin", flush=True)
        self._init_avatar_collider()
        if self.config.get("debug_reset", False):
            print("[base_reset] init_avatar_collider_done", flush=True)
            print("[base_reset] scene_build_begin", flush=True)

        self._setup_batch_render_cache()
        self.scene.build()
        if self.config.get("debug_reset", False):
            print("[base_reset] scene_build_done", flush=True)

        self.robot.init_joints(self.scene)
        if self.config.get("robot_type", "franka") == "franka" and self.cameras is not None:
            self.cameras.attach_franka_wrist_camera(self.robot)
        self.robot.move_to_homestate()
        self._init_avatar_collision_checker()
        if self.robot.left_arm is not None:
            self.robot.open_gripper("left")
        self.robot.open_gripper("right")
        self._apply_policy_gripper_gains()

        # Reset avatar position after scene build
        if self.avatar is not None:
            avatar_pos = np.asarray(
                getattr(self, "avatar_init_pos", [0.0, 0.0, 0.0]), dtype=np.float64
            )
            avatar_rot = np.asarray(
                getattr(self, "avatar_init_rot", np.eye(3)), dtype=np.float64
            )
            if self.config.get("debug_reset", False):
                print("[base_reset] avatar_reset_begin", flush=True)
            self.avatar.reset(
                avatar_pos,
                avatar_rot,
                update_mesh=not bool(self.config.get("skip_avatar_reset_update", False)),
            )
        if self.config.get("debug_reset", False):
            print("[base_reset] avatar_reset_done", flush=True)

        # Let dynamic objects fall and settle before the task starts.  This
        # intentionally uses BaseTask.SETTLE_STEPS, not subclass attributes, so
        # all tasks share one reset-settle budget.
        settle_steps = int(self.config.get("settle_steps", BaseTask.SETTLE_STEPS))
        if self.config.get("debug_reset", False):
            print(f"[base_reset] settle_begin steps={settle_steps}", flush=True)
        self._settle_scene(settle_steps)
        self.scene_init_replay = apply_scene_init_post_settle(self)
        if self.config.get("debug_reset", False):
            print("[base_reset] settle_done", flush=True)

        self.robot.set_origin_endpose()
        if self.config.get("debug_reset", False):
            print("[base_reset] origin_endpose_done", flush=True)

        # Reset state
        self.plan_success = True
        self.eval_success = False
        self.take_action_cnt = 0
        self.FRAME_IDX = 0
        self.left_joint_path = []
        self.right_joint_path = []
        self.traj_data = []

        # Avatar-collision episode state.  Collider + checker objects
        # themselves stay built; only the recorded results are cleared.
        self.avatar_collided = False
        self.avatar_collision_log = []
        self.avatar_safety_log = []
        self.avatar_safety_intervention_log = []
        self._avatar_retreat_hold_count = 0
        self._avatar_retreat_tick = 0
        self._avatar_retreat_active_cached = False
        self._avatar_retreat_release_state = {}
        self._avatar_collision_tick = 0

        # Build VLA recorder if requested.  Off by default so existing
        # collect.py / eval.py runs are zero-overhead.
        vla_cfg = self.config.get("vla_recording", {}) or {}
        if vla_cfg.get("enabled"):
            from .vla_recorder import VLARecorder
            self.vla_recorder = VLARecorder(
                out_dir=vla_cfg["out_dir"],
                cams=vla_cfg.get("cams", ("recording", "right_wrist")),
                jpeg_quality=int(vla_cfg.get("jpeg_quality", 92)),
                capture_stride=int(vla_cfg.get("capture_stride",
                                               self.config.get("action_substeps", 25))),
            )
        else:
            self.vla_recorder = None

        # Compute actual table top Z
        if self.table is not None and not getattr(self, "_table_top_z_fixed", False):
            pos = to_numpy(self.table.get_pos()).ravel()
            self.TABLE_TOP_Z = float(pos[2]) + self.TABLE_THICKNESS / 2

        if self.config.get("skip_reset_obs", False):
            if self.config.get("debug_reset", False):
                print("[base_reset] skip_reset_obs returning", flush=True)
            return None
        return self.get_obs()

    def _settle_scene(self, n_steps: int = None):
        """Run sim steps so gravity-loaded dynamic objects fall and come to
        rest before the task begins.  Uses the global BaseTask.SETTLE_STEPS by
        default; callers can pass n_steps explicitly for debug/special cases."""
        steps = BaseTask.SETTLE_STEPS if n_steps is None else n_steps
        for _ in range(int(steps)):
            self.scene.step()

    def _align_avatar_head_hand_to_xy(
        self,
        target_xy,
        hand_id: int = 1,
        motion_name: str | None = None,
        motion_frame: int = -1,
    ) -> float | None:
        """Yaw the avatar so the Head→Palm XY vector points at target_xy.

        Returns the applied yaw delta in degrees.  Call after `avatar.reset`
        has populated skin transforms and before scene settling.
        """
        if self.avatar is None or getattr(self.avatar.robot, "skin", None) is None:
            return None
        target_xy = np.asarray(target_xy, dtype=np.float64).ravel()[:2]
        motion = None
        if motion_name is not None:
            self.avatar.play_animation(str(motion_name))
            motion = self.avatar.motion_modules.get(str(motion_name))
            if motion is None or not getattr(motion, "data", None):
                return None
            frame = int(motion_frame)
            if frame < 0:
                frame = len(motion.data) + frame
            frame = int(np.clip(frame, 0, len(motion.data) - 1))
            def apply_motion_frame():
                robot = self.avatar.robot
                robot.pose = motion.data[frame]
                robot.node_trans = motion.node_data[frame]
                robot.global_mat = motion.global_mat
                robot.global_mat_inv = motion.global_mat_inv
                robot.update()
        else:
            def apply_motion_frame():
                return None

        skin = self.avatar.robot.skin
        side = "Left" if int(hand_id) == 0 else "Right"
        total_yaw = 0.0
        pos = np.asarray(self.avatar.robot.global_trans, dtype=np.float64).copy()
        rot = np.asarray(self.avatar.robot.global_rot, dtype=np.float64).copy()
        for _ in range(10):
            apply_motion_frame()
            try:
                neck = to_numpy(skin.get_global_translation("Neck")[0]).ravel()[:3]
                head_top = to_numpy(skin.get_global_translation("HeadTop_End")[0]).ravel()[:3]
                head = 0.5 * (neck + head_top)
                wrist = to_numpy(skin.get_global_translation(f"{side}Hand")[0]).ravel()[:3]
                mid1 = to_numpy(skin.get_global_translation(f"{side}HandMiddle1")[0]).ravel()[:3]
                hand = 0.5 * (wrist + mid1)
            except Exception:
                return None

            current = np.asarray(hand[:2] - head[:2], dtype=np.float64)
            desired = np.asarray(target_xy - head[:2], dtype=np.float64)
            cur_n = float(np.linalg.norm(current))
            dst_n = float(np.linalg.norm(desired))
            if cur_n < 1e-8 or dst_n < 1e-8:
                return None
            current /= cur_n
            desired /= dst_n
            cross = current[0] * desired[1] - current[1] * desired[0]
            dot = float(np.clip(np.dot(current, desired), -1.0, 1.0))
            yaw = float(np.arctan2(cross, dot))
            if abs(yaw) < np.deg2rad(0.05):
                break
            cz, sz = np.cos(yaw), np.sin(yaw)
            rot_z = np.array(
                [[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]],
                dtype=np.float64,
            )
            rot = rot_z @ rot
            total_yaw += yaw
            self.avatar.reset(pos, rot)

        self.avatar.reset(pos, rot)
        self.avatar_init_pos = pos
        self.avatar_init_rot = rot
        if self.avatar_collider is not None:
            self.avatar_collider.update()
        return float(np.rad2deg(total_yaw))

    def load_actors(self):
        """Override in subclass: add task-specific objects to scene."""
        pass

    def play_once(self) -> bool:
        """Override in subclass: scripted task rollout. Return True on success."""
        raise NotImplementedError

    def check_success(self) -> bool:
        """Default success check.

        Passes when ``plan_success`` is True **and** every threshold set on
        ``self.target`` holds (distance / tilt / collision).  Tasks with
        bespoke criteria (grip forces, joint angles, multi-object …) should
        override this.
        """
        if not self.plan_success:
            return False
        spec = self.target
        metrics = self._compute_target_metrics()

        if spec.success_dist_xy is not None:
            if metrics is None or metrics["dist_xy"] > spec.success_dist_xy:
                return False
        if spec.success_dist_3d is not None:
            if metrics is None or metrics["dist_3d"] > spec.success_dist_3d:
                return False
        if spec.success_dz_min is not None:
            if metrics is None or metrics["dz"] < spec.success_dz_min:
                return False
        if spec.success_tilt_min_deg is not None:
            if metrics is None or "tilt_deg" not in metrics \
                    or metrics["tilt_deg"] < spec.success_tilt_min_deg:
                return False
        if spec.success_tilt_max_deg is not None:
            if metrics is None or "tilt_deg" not in metrics \
                    or metrics["tilt_deg"] > spec.success_tilt_max_deg:
                return False
        if spec.success_require_no_avatar_collision:
            summary = self.avatar_collision_summary()
            if not summary.get("enabled") or summary.get("any_collision"):
                return False
        return True

    # ------------------------------------------------------------------
    # Avatar-collision query API
    # ------------------------------------------------------------------

    def avatar_collision_occurred(self) -> bool:
        """Did any robot↔avatar collision register this episode?"""
        return bool(self.avatar_collided)

    def latest_avatar_collision(self) -> dict | None:
        """Most recent checker result, or None if never checked."""
        return self.avatar_collision_log[-1] if self.avatar_collision_log else None

    def avatar_collision_summary(self) -> dict:
        """Episode-level roll-up of the per-step log.

        Returns keys:
          enabled, n_checks, n_collisions, any_collision,
          deepest_depth_m, deepest_pair, first_collision_frame,
          per_capsule_frames  (dict: capsule_name → n_frames)
        """
        if self.avatar_collision_checker is None:
            return {"enabled": False}
        log = self.avatar_collision_log
        n_checks = len(log)
        collided = [r for r in log if r.get("collided")]
        deepest = None
        deepest_pair = None
        if collided:
            worst = min(collided, key=lambda r: r["min_distance"])
            deepest = float(worst["min_distance"])
            deepest_pair = worst.get("closest_pair")
        first_frame = collided[0]["frame"] if collided else None
        per_capsule: dict = {}
        ignored_capsule: dict = {}
        for r in collided:
            # Every (link, capsule) pair in `pairs` contributes one
            # frame-hit to that capsule's count.
            for _link, cap, _d in (r.get("pairs") or []):
                per_capsule[cap] = per_capsule.get(cap, 0) + 1
        for r in log:
            for _link, cap, _d in (r.get("ignored_pairs") or []):
                ignored_capsule[cap] = ignored_capsule.get(cap, 0) + 1
        return {
            "enabled": True,
            "n_checks": n_checks,
            "n_collisions": len(collided),
            "n_ignored_collisions": sum(
                1 for r in log if (r.get("ignored_pairs") or [])
            ),
            "any_collision": bool(collided),
            "deepest_depth_m": deepest,
            "deepest_pair": deepest_pair,
            "first_collision_frame": first_frame,
            "per_capsule_frames": per_capsule,
            "ignored_capsule_frames": ignored_capsule,
        }

    def avatar_safety_summary(self) -> dict:
        """Episode-level summary for the inflated avatar safety envelope."""
        if self.avatar_safety_checker is None:
            return {"enabled": False}
        log = self.avatar_safety_log
        n_checks = len(log)
        violations = [r for r in log if r.get("collided")]
        closest = None
        closest_pair = None
        if log:
            best = min(log, key=lambda r: r["min_distance"])
            closest = float(best["min_distance"])
            closest_pair = best.get("closest_pair")
        first_frame = violations[0]["frame"] if violations else None
        return {
            "enabled": True,
            "margin_m": float(self.avatar_safety_checker.margin),
            "n_checks": n_checks,
            "n_violations": len(violations),
            "any_violation": bool(violations),
            "min_distance_m": closest,
            "closest_pair": closest_pair,
            "first_violation_frame": first_frame,
        }

    def avatar_safety_intervention_summary(self) -> dict:
        """Episode-level summary for active retreat overrides."""
        if self.avatar_retreat_checker is None:
            return {"enabled": False}
        log = self.avatar_safety_intervention_log
        closest = None
        if log:
            closest = float(min(r["min_distance"] for r in log))
        return {
            "enabled": True,
            "margin_m": float(self.avatar_retreat_checker.margin),
            "n_interventions": len(log),
            "any_intervention": bool(log),
            "min_distance_m": closest,
            "first_intervention_frame": log[0]["frame"] if log else None,
        }

    def safe_distance_summary(self) -> dict:
        """Episode-level closest robot-avatar signed distance.

        Positive values are clearance, zero is contact, and negative values
        are penetration depth.  Distances use the same sampled robot collision
        geometry and avatar capsule / palm-box model as avatar_collision.
        """
        if self.avatar_collision_checker is None:
            return {"enabled": False}

        def _distance_or_inf(record: dict) -> float:
            try:
                d = float(record.get("min_distance", float("inf")))
            except (TypeError, ValueError):
                return float("inf")
            return d if np.isfinite(d) else float("inf")

        n_checks = len(self.avatar_collision_log)
        log = [
            r for r in self.avatar_collision_log
            if np.isfinite(_distance_or_inf(r))
        ]
        if not log:
            return {
                "enabled": True,
                "n_checks": n_checks,
                "n_valid_distance_checks": 0,
                "safe_distance_m": None,
                "closest_pair": None,
                "closest_frame": None,
            }
        best = min(log, key=_distance_or_inf)
        return {
            "enabled": True,
            "n_checks": n_checks,
            "n_valid_distance_checks": len(log),
            "safe_distance_m": _distance_or_inf(best),
            "closest_pair": best.get("closest_pair"),
            "closest_frame": best.get("frame"),
        }

    # ------------------------------------------------------------------
    # Evaluation metrics
    # ------------------------------------------------------------------

    def _compute_target_metrics(self) -> Optional[dict]:
        """Resolve the spec into a raw-metrics dict, or None if unavailable.

        Keys: object_pos, target_pos, dist_xy, dist_3d, dz, and (if the spec
        supplies ``object_axis_fn``) tilt_deg = angle(axis, world +Z).
        """
        spec = self.target
        if spec.object is None or spec.position is None:
            return None

        if callable(spec.object_pos_fn):
            try:
                obj_pos = np.asarray(spec.object_pos_fn(),
                                     dtype=float).ravel()[:3]
            except Exception:
                return None
        else:
            entity = getattr(spec.object, "entity", spec.object)
            try:
                obj_pos = np.asarray(to_numpy(entity.get_pos()),
                                     dtype=float).ravel()[:3]
            except Exception:
                return None

        try:
            tgt_raw = spec.position() if callable(spec.position) else spec.position
            tgt_pos = np.asarray(tgt_raw, dtype=float).ravel()[:3]
        except Exception:
            return None

        metrics = {
            "object_pos": obj_pos,
            "target_pos": tgt_pos,
            "dist_xy": float(np.linalg.norm(obj_pos[:2] - tgt_pos[:2])),
            "dist_3d": float(np.linalg.norm(obj_pos - tgt_pos)),
            "dz": float(obj_pos[2] - tgt_pos[2]),
        }

        if callable(spec.object_axis_fn):
            try:
                axis = np.asarray(spec.object_axis_fn(),
                                  dtype=float).ravel()[:3]
                n = float(np.linalg.norm(axis))
                if n > 1e-8:
                    axis = axis / n
                    metrics["tilt_deg"] = float(np.degrees(
                        np.arccos(np.clip(axis[2], -1.0, 1.0))))
            except Exception:
                pass

        return metrics

    # Mapping from TargetSpec threshold attrs → output-dict keys.  Used by
    # evaluate() so thresholds that are set on the spec are echoed into the
    # pickled metrics dict for downstream analysis.
    _THRESHOLD_KEYS = (
        ("success_dist_xy",      "target_dist_xy_threshold"),
        ("success_dist_3d",      "target_dist_3d_threshold"),
        ("success_dz_min",       "target_dz_min_threshold"),
        ("success_tilt_min_deg", "target_tilt_min_deg_threshold"),
        ("success_tilt_max_deg", "target_tilt_max_deg_threshold"),
    )

    def evaluate(self) -> dict:
        """Standard per-episode metrics.

        Keys always present:
          success, plan_success, avatar_collision.
        Keys present iff ``self.target`` has an object + position:
          target_label, target_object_pos, target_pos,
          target_dist_xy, target_dist_3d, target_dz,
          target_tilt_deg (if object_axis_fn set),
          target_*_threshold (one per configured spec threshold).
        """
        out = {"plan_success": bool(self.plan_success), "success": False}
        try:
            out["success"] = bool(self.check_success())
        except NotImplementedError:
            out["check_success_error"] = "not implemented"
        except Exception as e:
            out["check_success_error"] = str(e)

        metrics = self._compute_target_metrics()
        if metrics is not None:
            spec = self.target
            out["target_label"] = spec.label
            out["target_object_pos"] = metrics["object_pos"].tolist()
            out["target_pos"] = metrics["target_pos"].tolist()
            out["target_dist_xy"] = metrics["dist_xy"]
            out["target_dist_3d"] = metrics["dist_3d"]
            out["target_dz"] = metrics["dz"]
            if "tilt_deg" in metrics:
                out["target_tilt_deg"] = metrics["tilt_deg"]
            for spec_key, out_key in self._THRESHOLD_KEYS:
                v = getattr(spec, spec_key)
                if v is not None:
                    out[out_key] = v
            if self.target.success_require_no_avatar_collision:
                out["target_require_no_avatar_collision"] = True

        avatar_collision = self.avatar_collision_summary()
        if avatar_collision.get("any_collision"):
            out["success"] = False
        out["avatar_collision"] = avatar_collision
        out["safe_distance"] = self.safe_distance_summary()
        out["avatar_safety"] = self.avatar_safety_summary()
        out["avatar_safety_intervention"] = self.avatar_safety_intervention_summary()
        return out

    # ------------------------------------------------------------------
    # Observation and action (testbed mode)
    # ------------------------------------------------------------------

    def get_obs(self) -> dict:
        """Get current observation."""
        right_ee = self.robot.get_ee_pose("right")
        left_ee = self.robot.get_ee_pose("left") if self.robot.left_arm is not None else None
        right_pose = Pose.from_pose7(right_ee) if right_ee is not None else None
        left_pose = Pose.from_pose7(left_ee) if left_ee is not None else None
        self.cameras.update_wrist_cameras(left_pose, right_pose)
        self.cameras.render_all()

        obs = {
            "rgb": self.cameras.get_all_rgb(),
            "depth": self.cameras.get_all_depth(),
            # Per-camera 3x3 intrinsic + 4x4 cam-to-world extrinsic matrices.
            # Wrist intrinsics are static; wrist extrinsics track the EE
            # (update_wrist_cameras above ran moments ago). Consumers that
            # don't need 3D info simply ignore these keys (DP3 lifts depth
            # to a point cloud here).
            "camera_intrinsics": {
                name: self.cameras.get_intrinsic(name)
                for name in self.cameras.get_camera_names()
            },
            "camera_extrinsics": {
                name: self.cameras.get_extrinsic(name)
                for name in self.cameras.get_camera_names()
            },
            "joint_state": {
                "right": self.robot.get_right_joint_state(),
            },
            "ee_pose": {
                "right": right_ee,
            },
            "gripper_val": {
                "right": self.robot.get_right_gripper_val(),
            },
        }
        # Include left arm data if available
        if self.robot.left_arm is not None:
            obs["joint_state"]["left"] = self.robot.get_left_joint_state()
            obs["ee_pose"]["left"] = left_ee
            obs["gripper_val"]["left"] = self.robot.get_left_gripper_val()
        return obs

    def _avatar_retreat_status(self) -> tuple[bool, dict | None]:
        """Return whether active safety should override the nominal command."""
        if self.avatar_retreat_checker is None:
            return False, None
        result = self.avatar_retreat_checker.check()
        min_distance = float(result.get("min_distance", float("inf")))
        margin = float(self.avatar_retreat_checker.margin)
        release_margin = float(self.config.get(
            "avatar_retreat_release_margin",
            margin + float(self.config.get("avatar_retreat_hysteresis", 0.05)),
        ))
        hold_steps = int(self.config.get("avatar_retreat_hold_steps", 8))

        if min_distance <= margin:
            self._avatar_retreat_hold_count = max(1, hold_steps)
            active = True
        elif self._avatar_retreat_hold_count > 0 and min_distance < release_margin:
            self._avatar_retreat_hold_count -= 1
            active = True
        else:
            self._avatar_retreat_hold_count = 0
            active = False

        if active:
            self.avatar_safety_intervention_log.append({
                "frame": int(self.FRAME_IDX),
                "min_distance": min_distance,
                "margin": margin,
                "release_margin": release_margin,
                "closest_pair": result.get("closest_pair"),
            })
        return active, result

    def _closest_avatar_point_to(self, point: np.ndarray) -> np.ndarray | None:
        """Closest point on the current avatar capsule set to a world point."""
        if self.avatar_collider is None:
            return None
        capsules = self.avatar_collider.current_capsules()
        if not capsules:
            return None
        point = np.asarray(point, dtype=np.float64).ravel()[:3]
        best_point = None
        best_dist = float("inf")
        for _name, A, B, radius in capsules:
            A = np.asarray(A, dtype=np.float64)
            B = np.asarray(B, dtype=np.float64)
            AB = B - A
            ab_sq = float(np.dot(AB, AB))
            if ab_sq < 1e-12:
                continue
            t = float(np.clip(np.dot(point - A, AB) / ab_sq, 0.0, 1.0))
            closest = A + t * AB
            signed = float(np.linalg.norm(point - closest) - float(radius))
            if signed < best_dist:
                best_dist = signed
                best_point = closest
        return best_point

    def _avatar_retreat_command_pose(self, arm_tag: str, pose: np.ndarray) -> bool:
        arm = self.robot.get_arm(arm_tag)
        result = arm.planner.solve_ik(arm.get_arm_qpos(), pose, num_waypoints=1, log=False)
        if result.success and result.position.size > 0:
            self.robot.set_arm_joints(result.position[-1], arm_tag)
            return True
        return False

    def _avatar_retreat_putdown(self, arm_tag: str, pose: np.ndarray) -> bool:
        if not bool(self.config.get("avatar_retreat_putdown_enabled", True)):
            return False
        arm = self.robot.get_arm(arm_tag)
        closed_threshold = float(self.config.get("avatar_retreat_holding_gripper_threshold", 0.45))
        if arm.gripper_val > closed_threshold and arm_tag not in self._avatar_retreat_release_state:
            return False

        state = self._avatar_retreat_release_state.setdefault(
            arm_tag, {"phase": "clear", "ticks": 0}
        )
        table_z = float(getattr(self, "TABLE_TOP_Z", 0.765))
        putdown_z = table_z + float(self.config.get("avatar_retreat_putdown_ee_z", 0.105))
        lower_step = float(self.config.get("avatar_retreat_putdown_step", 0.015))
        release_steps = int(self.config.get("avatar_retreat_release_steps", 80))
        release_margin = float(self.config.get(
            "avatar_retreat_release_margin",
            float(self.avatar_retreat_checker.margin)
            + float(self.config.get("avatar_retreat_hysteresis", 0.05)),
        ))

        if state["phase"] == "clear":
            active, result = self._avatar_retreat_status()
            min_distance = float((result or {}).get("min_distance", float("inf")))
            if min_distance >= release_margin:
                state["phase"] = "lower"
            else:
                return False

        if state["phase"] == "lower":
            target = pose.copy()
            target[2] = max(putdown_z, pose[2] - lower_step)
            ok = self._avatar_retreat_command_pose(arm_tag, target)
            if pose[2] <= putdown_z + 0.015:
                state["phase"] = "release"
                state["ticks"] = 0
            return ok

        if state["phase"] == "release":
            self.robot.set_gripper(1.0, arm_tag)
            if arm_tag in self._gripper_attached:
                self.detach_from_gripper(arm_tag)
            state["ticks"] += 1
            if state["ticks"] >= release_steps:
                state["phase"] = "done"
            return True

        return False

    def _apply_avatar_retreat(self, arm_tag: str) -> bool:
        """Command one arm up and away from the closest avatar capsule."""
        if self.avatar_retreat_checker is None:
            return False
        arm = self.robot.get_arm(arm_tag)
        pose = np.asarray(arm.get_ee_pose(), dtype=np.float64).ravel()[:7]
        if pose.size < 7:
            return False
        closest = self._closest_avatar_point_to(pose[:3])
        if closest is None:
            return False

        if self._avatar_retreat_putdown(arm_tag, pose):
            return True

        away_xy = pose[:2] - closest[:2]
        norm_xy = float(np.linalg.norm(away_xy))
        if norm_xy < 1e-5:
            # Default away from the nominal robot side of the table.
            away_xy = np.array([0.0, -1.0], dtype=np.float64)
            norm_xy = 1.0
        away_xy = away_xy / norm_xy
        z_weight = float(self.config.get("avatar_retreat_z_weight", 0.25))
        direction = np.array([away_xy[0], away_xy[1], max(0.0, z_weight)], dtype=np.float64)
        direction /= float(np.linalg.norm(direction) + 1e-12)

        step = float(self.config.get("avatar_retreat_step", 0.025))
        retreat_pose = pose.copy()
        retreat_pose[:3] = pose[:3] + step * direction
        min_z = float(self.config.get("avatar_retreat_min_z", getattr(self, "TABLE_TOP_Z", 0.0) + 0.03))
        max_z = float(self.config.get("avatar_retreat_max_z", getattr(self, "TABLE_TOP_Z", 0.0) + 0.24))
        retreat_pose[2] = float(np.clip(retreat_pose[2], min_z, max_z))

        return self._avatar_retreat_command_pose(arm_tag, retreat_pose)

    def _apply_avatar_retreat_if_needed(self, arm_tags: tuple[str, ...]) -> bool:
        if self.avatar_retreat_checker is None:
            return False
        stride = max(1, int(self.config.get("avatar_retreat_control_stride", 10)))
        self._avatar_retreat_tick += 1
        if self._avatar_retreat_tick < stride:
            if not self._avatar_retreat_active_cached:
                return False
            applied = False
            for arm_tag in arm_tags:
                try:
                    applied = self._apply_avatar_retreat(arm_tag) or applied
                except Exception:
                    continue
            return applied
        self._avatar_retreat_tick = 0
        active, _result = self._avatar_retreat_status()
        if not active:
            active = any(
                self._avatar_retreat_release_state.get(tag, {}).get("phase")
                in ("clear", "lower", "release")
                for tag in arm_tags
            )
        self._avatar_retreat_active_cached = bool(active)
        if not active:
            return False
        applied = False
        for arm_tag in arm_tags:
            try:
                applied = self._apply_avatar_retreat(arm_tag) or applied
            except Exception:
                continue
        return applied

    def take_action(self, action: np.ndarray, action_type: Literal["qpos", "qpos_abs", "ee"] = "qpos"):
        """Apply action and return new observation.

        Args:
            action: action vector. For qpos: [left_arm_delta(6), left_gripper(1), right_arm_delta(6), right_gripper(1)]
                    For qpos_abs: [left_arm_qpos(6), left_gripper(1), right_arm_qpos(6), right_gripper(1)]
                    For ee: [left_ee_pose(7), right_ee_pose(7)]
            action_type: "qpos" for joint position deltas, "qpos_abs" for absolute joint targets,
                         "ee" for end-effector poses
        """
        if self.eval_success:
            return self.get_obs()

        self.take_action_cnt += 1
        action = np.asarray(action, dtype=np.float64).ravel()

        interpolate_qpos = bool(self.config.get("interpolate_qpos_actions", False)) and action_type in {"qpos", "qpos_abs"}
        qpos_interp_targets = []
        gripper_close_hold_steps = 0

        def _filtered_gripper_target(current, target):
            nonlocal gripper_close_hold_steps
            if target is None:
                return None
            target = float(target)
            current = float(current)
            max_delta = self.config.get("qpos_gripper_max_delta_per_action")
            if max_delta is not None:
                max_delta = max(0.0, float(max_delta))
                target = current + float(np.clip(target - current, -max_delta, max_delta))
            if target < current:
                gripper_close_hold_steps = max(
                    gripper_close_hold_steps,
                    int(self.config.get("qpos_gripper_close_hold_steps", 0)),
                )
            return target

        if action_type in {"qpos", "qpos_abs"}:
            n_arm = self.robot.right_arm.n_arm
            if self.robot.left_arm is not None:
                left_qpos = self.robot.left_arm.get_arm_qpos()
                right_qpos = self.robot.right_arm.get_arm_qpos()
                if action_type == "qpos_abs":
                    left_target = action[:n_arm]
                    right_target = action[n_arm + 1: 2 * n_arm + 1]
                else:
                    left_target = left_qpos + action[:n_arm]
                    right_target = right_qpos + action[n_arm + 1: 2 * n_arm + 1]
                left_gripper = _filtered_gripper_target(
                    self.robot.get_arm("left").gripper_val,
                    float(action[n_arm]) if len(action) > n_arm else None,
                )
                right_gripper = _filtered_gripper_target(
                    self.robot.get_arm("right").gripper_val,
                    float(action[2 * n_arm + 1]) if len(action) > 2 * n_arm + 1 else None,
                )
                if interpolate_qpos:
                    qpos_interp_targets.extend([
                        ("left", left_qpos, left_target, self.robot.get_arm("left").gripper_val, left_gripper),
                        ("right", right_qpos, right_target, self.robot.get_arm("right").gripper_val, right_gripper),
                    ])
                else:
                    self.robot.set_arm_joints(left_target, "left")
                    self.robot.set_arm_joints(right_target, "right")
                    if left_gripper is not None:
                        self.robot.set_gripper(left_gripper, "left")
                    if right_gripper is not None:
                        self.robot.set_gripper(right_gripper, "right")
            else:
                # Single arm mode: action is [arm_delta_or_qpos(7), gripper(1)]
                right_qpos = self.robot.right_arm.get_arm_qpos()
                right_target = action[:n_arm] if action_type == "qpos_abs" else right_qpos + action[:n_arm]
                right_gripper = _filtered_gripper_target(
                    self.robot.right_arm.gripper_val,
                    float(action[n_arm]) if len(action) > n_arm else None,
                )
                if interpolate_qpos:
                    qpos_interp_targets.append(
                        ("right", right_qpos, right_target, self.robot.right_arm.gripper_val, right_gripper)
                    )
                else:
                    self.robot.set_arm_joints(right_target, "right")
                    if right_gripper is not None:
                        self.robot.set_gripper(right_gripper, "right")
        else:
            # EE pose mode: 8 per arm = [xyz, qw, qx, qy, qz, gripper].
            # Learned EE policies often emit near-zero deltas during the
            # recorded stationary prefix. Re-solving IK for a target already at
            # the current TCP can pick a slightly different null-space solution,
            # which creates drift even though the training label is no-op.
            def _move_if_not_current(pose, arm_tag: str):
                target = np.asarray(pose, dtype=np.float64).ravel()[:7]
                current = np.asarray(
                    self.robot.get_arm(arm_tag).get_ee_pose(), dtype=np.float64,
                ).ravel()[:7]
                pos_err = float(np.linalg.norm(target[:3] - current[:3]))
                q_target = target[3:7] / max(np.linalg.norm(target[3:7]), 1e-12)
                q_current = current[3:7] / max(np.linalg.norm(current[3:7]), 1e-12)
                quat_dot = abs(float(np.clip(np.dot(q_target, q_current), -1.0, 1.0)))
                ang_err = float(2.0 * np.arccos(quat_dot))
                if pos_err <= 1e-3 and ang_err <= 1e-3:
                    return
                self._move_to_pose(target, arm_tag)

            if self.robot.left_arm is not None:
                _move_if_not_current(action[:7], "left")
                _move_if_not_current(action[8:15], "right")
                if len(action) > 7:
                    self.robot.set_gripper(float(action[7]), "left")
                if len(action) > 15:
                    self.robot.set_gripper(float(action[15]), "right")
            else:
                _move_if_not_current(action[:7], "right")
                if len(action) > 7:
                    self.robot.set_gripper(float(action[7]), "right")

        # Run N physics substeps so the PD controller has time to converge to the
        # new target. Default 25 @ dt=0.002 → 50 ms per policy action (≈20 Hz ctrl).
        arm_tags = ("left", "right") if self.robot.left_arm is not None else ("right",)
        action_substeps = int(self.config.get("action_substeps", 25))
        immediate_gripper = bool(self.config.get("qpos_gripper_immediate", False))
        pause_arm_on_close = bool(self.config.get("qpos_gripper_pause_arm_on_close", False))
        for substep in range(action_substeps):
            if qpos_interp_targets:
                frac = (substep + 1) / max(1, action_substeps)
                for arm_tag, start_qpos, target_qpos, start_gripper, target_gripper in qpos_interp_targets:
                    gripper_closing = target_gripper is not None and float(target_gripper) < float(start_gripper)
                    if pause_arm_on_close and gripper_closing:
                        q = start_qpos
                    else:
                        q = start_qpos * (1.0 - frac) + target_qpos * frac
                    self.robot.set_arm_joints(q, arm_tag)
                    if target_gripper is not None:
                        g = (
                            target_gripper
                            if immediate_gripper
                            else start_gripper * (1.0 - frac) + target_gripper * frac
                        )
                        self.robot.set_gripper(float(g), arm_tag)
            self._apply_avatar_retreat_if_needed(arm_tags)
            self.step_sim()
        for _ in range(gripper_close_hold_steps):
            self._apply_avatar_retreat_if_needed(arm_tags)
            self.step_sim()

        return self.get_obs()

    # ------------------------------------------------------------------
    # Motion helpers (for scripted rollouts)
    # ------------------------------------------------------------------

    def move_to_pose(self, pose, arm_tag: str):
        """Plan and execute motion to target pose. Returns plan result."""
        if not self.plan_success:
            return None
        if pose is None:
            self.plan_success = False
            return None

        pose7 = np.asarray(pose, dtype=np.float64).ravel()[:7]
        arm = self.robot.get_arm(arm_tag)
        result = arm.planner.plan_path(arm.get_arm_qpos(), pose7)

        if arm_tag == "left":
            self.left_joint_path.append(deepcopy(result.as_dict()))
        else:
            self.right_joint_path.append(deepcopy(result.as_dict()))

        if not result.success:
            self.plan_success = False
            return None

        return result

    def execute_plan(self, result, arm_tag: str, gripper_result=None):
        """Execute a planned trajectory."""
        if result is None:
            return

        teleport = os.environ.get("DEBUG_TELEPORT", "").strip() == "1"
        use_setqpos = os.environ.get("USE_SETQPOS", "").strip() == "1"
        n_steps = result.position.shape[0] if result.position.size > 0 else 0
        print(f"[execute_plan] trajectory steps={n_steps}, teleport={teleport}, setqpos={use_setqpos}")
        gripper_steps = len(gripper_result) if gripper_result is not None else 0
        max_steps = max(n_steps, gripper_steps)

        # Hold current gripper value throughout motion unless gripper_result overrides
        arm = self.robot.get_arm(arm_tag)
        hold_gripper_val = arm.gripper_val

        i = 0
        extra_steps = 0
        max_extra_cfg = self.config.get("avatar_retreat_max_extra_steps", 800)
        max_extra_steps = None if max_extra_cfg is None else int(max_extra_cfg)
        while i < max_steps:
            retreating = self._apply_avatar_retreat_if_needed((arm_tag,))
            if i < n_steps and not retreating:
                if teleport or use_setqpos:
                    self.robot.teleport_arm_joints(result.position[i], arm_tag)
                else:
                    self.robot.set_arm_joints(result.position[i], arm_tag)
            if retreating:
                pass
            elif gripper_result is not None and i < gripper_steps:
                self.robot.set_gripper(float(gripper_result[i]), arm_tag)
            else:
                self.robot.set_gripper(hold_gripper_val, arm_tag)

            self.step_sim()
            if retreating:
                extra_steps += 1
                if max_extra_steps is not None and extra_steps > max_extra_steps:
                    print("[avatar-retreat] max extra retreat steps reached; resuming nominal plan")
                    self._avatar_retreat_hold_count = 0
                    self._avatar_retreat_active_cached = False
                    i += 1
            else:
                i += 1

    def move_and_execute(self, pose, arm_tag: str):
        """Plan + execute in one call."""
        result = self.move_to_pose(pose, arm_tag)
        if result is not None:
            self.execute_plan(result, arm_tag)
        return result

    # ------------------------------------------------------------------
    # Pre-grasp direction sampling
    # ------------------------------------------------------------------

    def find_best_pregrasp(
        self,
        grasp_tcp: "Pose",
        arm_tag: str,
        pre_dist: float = 0.12,
        constraints: list = None,
        n_az: int = 12,
        n_el: int = 8,
        bar_axis: np.ndarray = None,
    ) -> "Pose":
        """Sample pre-grasp retreat directions and return the link pose with least IK error.

        Args:
            grasp_tcp: TCP pose at the grasp point (world frame).
            arm_tag: which arm to use.
            pre_dist: how far to back off from grasp point.
            constraints: list of (normal_vec, min_dot) tuples. A candidate direction
                d is valid only if dot(d, normal) > min_dot for ALL constraints.
                Example: [([1,0,0], 0.1)] requires d to have positive X component.
            n_az: number of azimuth samples.
            n_el: number of elevation samples.
            bar_axis: optional axis for TCP +X (e.g. handle bar direction).
                If None, uses [0, 0, 1] as fallback.

        Returns:
            Link-frame Pose for the best pre-grasp (ready for move_and_execute).
        """
        from .grasp import tcp_to_link_pose as _tcp_to_link

        arm = self.robot.get_arm(arm_tag)
        tcp_offset = arm.tcp_offset
        current_qpos = arm.get_arm_qpos()
        grasp_pos = grasp_tcp.p

        # Default approach direction from grasp TCP +Z
        R_grasp = grasp_tcp.to_matrix()[:3, :3]
        default_approach = R_grasp[:, 2]

        constraints = constraints or []
        best_err = float("inf")
        best_link = None
        best_dir = None

        for az_deg in np.linspace(-80, 80, n_az):
            for el_deg in np.linspace(-60, 30, n_el):
                az = np.deg2rad(az_deg)
                el = np.deg2rad(el_deg)

                # Direction in local frame where default approach = +X
                d_local = np.array([
                    np.cos(el) * np.cos(az),
                    np.cos(el) * np.sin(az),
                    np.sin(el),
                ])

                # Rotate so local +X aligns with default approach
                ax = np.array([1.0, 0.0, 0.0])
                v = np.cross(ax, default_approach)
                c = float(np.dot(ax, default_approach))
                if np.linalg.norm(v) < 1e-8:
                    R_rot = np.eye(3) if c > 0 else np.diag([-1.0, 1.0, -1.0])
                else:
                    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
                    R_rot = np.eye(3) + vx + vx @ vx / (1.0 + c)

                d_world = R_rot @ d_local
                d_world /= np.linalg.norm(d_world) + 1e-12

                # Check constraints
                valid = True
                for normal, min_dot in constraints:
                    if np.dot(d_world, np.asarray(normal, dtype=np.float64)) < min_dot:
                        valid = False
                        break
                if not valid:
                    continue

                # Build pre-grasp TCP pose
                pre_pos = grasp_pos - d_world * pre_dist

                # TCP frame: +Z = d_world (approach), +X = bar_axis projected
                tcp_z = d_world.copy()
                if bar_axis is not None:
                    tcp_x = bar_axis - np.dot(bar_axis, tcp_z) * tcp_z
                    n = np.linalg.norm(tcp_x)
                    tcp_x = tcp_x / n if n > 1e-6 else np.array([0.0, 0.0, 1.0])
                else:
                    # Pick axis least aligned with tcp_z
                    up = np.array([0.0, 0.0, 1.0])
                    tcp_x = up - np.dot(up, tcp_z) * tcp_z
                    n = np.linalg.norm(tcp_x)
                    tcp_x = tcp_x / n if n > 1e-6 else np.array([1.0, 0.0, 0.0])
                tcp_y = np.cross(tcp_z, tcp_x)
                tcp_y /= np.linalg.norm(tcp_y) + 1e-8
                R_tcp = np.column_stack([tcp_x, tcp_y, tcp_z])
                import transforms3d as t3d
                q = t3d.quaternions.mat2quat(R_tcp)

                pre_tcp = Pose(pre_pos, q)
                pre_link = _tcp_to_link(pre_tcp, tcp_offset)

                # Quick IK check
                result = arm.planner.solve_ik(
                    current_qpos, pre_link.to_pose7(), num_waypoints=1, log=False,
                )
                if not result.success:
                    continue

                # Measure actual EE error
                arm.teleport_arm_joints(result.position[-1])
                self.scene.step()
                ee_pos = np.array(arm.get_ee_pose()[:3])
                err = float(np.linalg.norm(ee_pos - pre_link.p))

                if err < best_err:
                    best_err = err
                    best_link = pre_link
                    best_dir = d_world

        # Restore arm
        arm.teleport_arm_joints(current_qpos)
        self.scene.step()

        if best_link is not None:
            print(f"[pregrasp] best dir={best_dir}, error={best_err:.4f}")
        else:
            print("[pregrasp] no valid direction found, using default approach")
            pre_pos = grasp_pos - default_approach * pre_dist
            pre_tcp = Pose(pre_pos, grasp_tcp.q)
            best_link = _tcp_to_link(pre_tcp, tcp_offset)

        return best_link

    # ------------------------------------------------------------------
    # General grasp selection
    # ------------------------------------------------------------------

    def select_grasp(
        self,
        object_name: str,
        object_pose: Pose,
        arm_tag: str,
        model_id: int = None,
        robot_type: str = None,
        max_candidates: int = 10,
        table_z: float = None,
        pre_dist: float = None,
        object_scale: float = 1.0,
    ):
        """Select the best feasible grasp from saved grasp poses.

        Loads grasps from YAML, ranks them by approach-direction heuristic,
        then tries mplib planning on top candidates until one succeeds.

        Args:
            object_name: object directory name (e.g. "001_bottle").
            object_pose: current object pose in world frame.
            arm_tag: which arm to use.
            model_id: optional model variant filter.
            robot_type: optional robot type for robot-specific grasps (e.g. "franka").
            max_candidates: max grasps to try planning for.
            table_z: table surface height; grasps below this are discarded.
                     Defaults to self.TABLE_TOP_Z if available.
            pre_dist: override pre-grasp distance. If None, uses each grasp's own value.
            object_scale: mesh scale used when loading the object in the task.
                Grasp poses are stored at scale=1; this
                scales the local positions to match the loaded mesh size.

        Returns:
            (grasp_link, pre_link, grasp_pose) on success — link poses for IK
                and the GraspPose object for reference.
            None if no feasible grasp found.
        """
        import transforms3d as t3d

        if table_z is None:
            table_z = getattr(self, "TABLE_TOP_Z", 0.74)
        if robot_type is None:
            robot_type = self.config.get("robot_type")

        arm = self.robot.get_arm(arm_tag)
        tcp_offset = arm.tcp_offset
        robot_base = arm.origin_pose.p

        grasps = load_grasp_poses(object_name, model_id=model_id, robot_type=robot_type)
        if not grasps:
            print(f"[select_grasp] No grasp poses found for {object_name}")
            return None

        # --- Score and rank ---
        scored = []
        print(f"[select_grasp] {len(grasps)} grasps loaded, table_z={table_z}, robot_base={robot_base}")
        for g in grasps:
            tcp_world = g.to_world(object_pose, object_scale)
            R_tcp = t3d.quaternions.quat2mat(tcp_world.q)
            tcp_z = R_tcp[:, 2]  # approach direction

            # 1. Pre-grasp → grasp path must not cross through the object.
            #    The pre-grasp must be farther from object center than the grasp,
            #    otherwise the gripper sweeps through the object body.
            d = pre_dist if pre_dist is not None else g.pre_distance
            pre_pos = tcp_world.p - d * tcp_z
            dist_grasp = float(np.linalg.norm(tcp_world.p - object_pose.p))
            dist_pre = float(np.linalg.norm(pre_pos - object_pose.p))
            if dist_pre < dist_grasp:
                print(f"[select_grasp]   {g.name}: FILTERED pre-grasp closer to object than grasp "
                      f"(pre={dist_pre:.3f} < grasp={dist_grasp:.3f}), path would cross object")
                continue

            # 2. Approach alignment: tcp_z should point from robot toward object
            #    so pre-grasp (= grasp - pre_dist * tcp_z) is on the robot's side
            to_obj = object_pose.p - robot_base
            to_obj[2] = 0.0  # horizontal only
            norm = np.linalg.norm(to_obj)
            if norm > 1e-6:
                to_obj /= norm
            approach_score = float(np.dot(tcp_z[:2], to_obj[:2]))

            # 3. Height check: TCP must be above table with margin for fingers
            finger_margin = 0.01
            if tcp_world.p[2] < table_z + finger_margin:
                print(f"[select_grasp]   {g.name}: FILTERED height {tcp_world.p[2]:.3f} < {table_z + finger_margin:.3f}")
                continue

            # 4. Pre-grasp must also be above table (pre_pos computed in filter 1)
            if pre_pos[2] < table_z + finger_margin:
                print(f"[select_grasp]   {g.name}: FILTERED pre-grasp height {pre_pos[2]:.3f} < {table_z + finger_margin:.3f}")
                continue

            scored.append((approach_score, g))

        print(f"[select_grasp] {len(scored)}/{len(grasps)} passed filters")
        if not scored:
            print(f"[select_grasp] All grasps filtered out for {object_name}")
            return None

        # Manual (hand-verified) grasps are preferred over auto grasps;
        # within each source group, rank by approach-alignment score.
        scored.sort(key=lambda x: (0 if x[1].source == "manual" else 1, -x[0]))

        # --- Try planning for top candidates ---
        n_try = min(max_candidates, len(scored))
        current_qpos = arm.get_arm_qpos()

        for i in range(n_try):
            score, g = scored[i]
            tcp_world = g.to_world(object_pose, object_scale)
            d = pre_dist if pre_dist is not None else g.pre_distance
            pre_tcp = g.pre_grasp_world(object_pose, object_scale) if pre_dist is None else \
                Pose(tcp_world.p - d * t3d.quaternions.quat2mat(tcp_world.q)[:, 2], tcp_world.q)

            grasp_link = tcp_to_link_pose(tcp_world, tcp_offset)
            pre_link = tcp_to_link_pose(pre_tcp, tcp_offset)

            # Try planning: current → pre-grasp → grasp
            result_pre = arm.planner.plan_path(current_qpos, pre_link.to_pose7())
            if not result_pre.success:
                print(f"[select_grasp] {g.name} (score={score:.2f}): pre-grasp plan failed")
                continue

            # Plan grasp from end of pre-grasp trajectory
            end_qpos = result_pre.position[-1] if result_pre.position.size > 0 else current_qpos
            result_grasp = arm.planner.plan_path(end_qpos, grasp_link.to_pose7())
            if not result_grasp.success:
                print(f"[select_grasp] {g.name} (score={score:.2f}): grasp plan failed")
                continue

            # FK quality check: set qpos and read link pos without stepping physics
            grasp_final_qpos = result_grasp.position[-1] if result_grasp.position.size > 0 else end_qpos
            saved_qpos_raw = arm.entity.get_qpos()
            full_qpos = arm._build_qpos_with_arm(grasp_final_qpos)
            arm.entity.set_qpos(torch.tensor(full_qpos, dtype=torch.float32) if not hasattr(full_qpos, 'cpu') else full_qpos)
            arm.entity.get_links_pos()  # trigger FK update without physics step
            ee_link_pos = to_numpy(arm.ee_link.get_pos()).ravel()[:3]
            fk_err = float(np.linalg.norm(ee_link_pos - grasp_link.p))
            arm.entity.set_qpos(saved_qpos_raw)  # restore without stepping
            arm.entity.get_links_pos()

            if fk_err > 0.03:
                print(f"[select_grasp] {g.name} (score={score:.2f}): FK error {fk_err:.4f}m > 0.03m, skipping")
                continue

            print(f"[select_grasp] {g.name} (score={score:.2f}): SUCCESS (FK err={fk_err:.4f}m, tried {i+1}/{n_try})")
            return grasp_link, pre_link, g

        print(f"[select_grasp] No feasible grasp found after {n_try} candidates")
        return None

    def _plan_one_grasp(
        self,
        grasp: GraspPose,
        object_pose: Pose,
        arm_tag: str,
        pre_dist: float = None,
        object_scale: float = 1.0,
        log_prefix: str = "",
    ):
        """Plan curr→pre via RRT and pre→grasp via plan_screw for ONE grasp.
        FK-check the planned grasp endpoint.  Returns
        ``(result_pre, result_grasp, grasp_link, pre_link)`` on success or
        ``None`` if any step fails.  Does NOT execute — caller drives that.
        """
        import transforms3d as t3d

        arm = self.robot.get_arm(arm_tag)
        tcp_offset = arm.tcp_offset

        tcp_world = grasp.to_world(object_pose, object_scale)
        d = pre_dist if pre_dist is not None else grasp.pre_distance
        if pre_dist is None:
            pre_tcp = grasp.pre_grasp_world(object_pose, object_scale)
        else:
            R = t3d.quaternions.quat2mat(tcp_world.q)
            pre_tcp = Pose(tcp_world.p - d * R[:, 2], tcp_world.q)
        grasp_link = tcp_to_link_pose(tcp_world, tcp_offset)
        pre_link = tcp_to_link_pose(pre_tcp, tcp_offset)

        current_qpos = arm.get_arm_qpos()

        result_pre = arm.planner.plan_path(
            current_qpos, pre_link.to_pose7(),
        )
        if not result_pre.success:
            print(f"[grasp_pick]   {log_prefix}{grasp.name}: pre RRT failed")
            return None
        end_pre_qpos = (
            result_pre.position[-1]
            if result_pre.position.size > 0 else current_qpos
        )

        plan_screw = getattr(arm.planner, "plan_screw_path", None)
        if plan_screw is None:
            print(f"[grasp_pick]   {log_prefix}{grasp.name}: planner has no plan_screw_path")
            return None
        result_grasp = plan_screw(end_pre_qpos, grasp_link.to_pose7())
        if not result_grasp.success:
            print(f"[grasp_pick]   {log_prefix}{grasp.name}: screw failed "
                  f"— skipping (no RRT fallback by design)")
            return None

        grasp_final_qpos = (
            result_grasp.position[-1]
            if result_grasp.position.size > 0 else end_pre_qpos
        )
        saved = arm.entity.get_qpos()
        full_qpos = arm._build_qpos_with_arm(grasp_final_qpos)
        arm.entity.set_qpos(
            torch.tensor(full_qpos, dtype=torch.float32)
            if not hasattr(full_qpos, "cpu") else full_qpos
        )
        arm.entity.get_links_pos()
        ee_link_pos = to_numpy(arm.ee_link.get_pos()).ravel()[:3]
        fk_err = float(np.linalg.norm(ee_link_pos - grasp_link.p))
        arm.entity.set_qpos(saved)
        arm.entity.get_links_pos()

        if fk_err > 0.03:
            print(f"[grasp_pick]   {log_prefix}{grasp.name}: "
                  f"FK err {fk_err:.4f}m > 3cm")
            return None

        print(f"[grasp_pick]   {log_prefix}{grasp.name}: planned OK "
              f"(FK err={fk_err:.4f}m)")
        return result_pre, result_grasp, grasp_link, pre_link

    def select_and_execute_grasp(
        self,
        object_name: str,
        object_pose: Pose,
        arm_tag: str,
        model_id: int = None,
        robot_type: str = None,
        max_candidates: int = 10,
        table_z: float = None,
        pre_dist: float = None,
        object_scale: float = 1.0,
        categories: list = None,
    ):
        """Pre-flight + execute pattern.  For each ranked grasp candidate:
        plan curr→pre via RRT, plan pre→grasp via plan_screw (straight line
        along the grasp's approach axis, NO RRT fallback).  If both legs
        plan and the FK check passes, execute the cached trajectories;
        otherwise advance to the next candidate.  This guarantees that the
        final approach segment moves exactly along the grasp z-axis (no
        wrist swing through the object) and never silently RRTs around.

        Returns ``(grasp_link, pre_link, grasp_pose)`` on success — same
        triple as :meth:`select_grasp` so callers can chain close + lift
        unchanged.  Returns ``None`` if every candidate is rejected; the
        caller should iterate to the next object or try a different grasp
        pool.
        """
        import transforms3d as t3d

        if table_z is None:
            table_z = getattr(self, "TABLE_TOP_Z", 0.74)
        if robot_type is None:
            robot_type = self.config.get("robot_type")

        arm = self.robot.get_arm(arm_tag)
        robot_base = arm.origin_pose.p

        grasps = load_grasp_poses(
            object_name, model_id=model_id, robot_type=robot_type,
        )
        if not grasps:
            print(f"[grasp_pick] {object_name}: no grasps in YAML")
            return None

        scored = []
        for g in grasps:
            if categories is not None and g.category not in categories:
                continue
            tcp_world = g.to_world(object_pose, object_scale)
            R_tcp = t3d.quaternions.quat2mat(tcp_world.q)
            tcp_z = R_tcp[:, 2]
            d = pre_dist if pre_dist is not None else g.pre_distance
            pre_pos = tcp_world.p - d * tcp_z
            dist_grasp = float(np.linalg.norm(tcp_world.p - object_pose.p))
            dist_pre = float(np.linalg.norm(pre_pos - object_pose.p))
            if dist_pre < dist_grasp:
                continue
            to_obj = object_pose.p - robot_base
            to_obj[2] = 0.0
            n = np.linalg.norm(to_obj)
            if n > 1e-6:
                to_obj /= n
            approach_score = float(np.dot(tcp_z[:2], to_obj[:2]))
            finger_margin = 0.01
            if tcp_world.p[2] < table_z + finger_margin:
                continue
            if pre_pos[2] < table_z + finger_margin:
                continue
            scored.append((approach_score, g))

        if not scored:
            print(f"[grasp_pick] {object_name}: all {len(grasps)} grasps filtered out")
            return None

        scored.sort(key=lambda x: (0 if x[1].source == "manual" else 1, -x[0]))
        n_try = min(max_candidates, len(scored))
        print(f"[grasp_pick] {object_name}: {len(scored)} feasible candidates, "
              f"trying top {n_try}")

        for i in range(n_try):
            score, g = scored[i]
            plans = self._plan_one_grasp(
                g, object_pose, arm_tag,
                pre_dist=pre_dist, object_scale=object_scale,
                log_prefix=f"(score={score:+.2f}) ",
            )
            if plans is None:
                continue
            result_pre, result_grasp, grasp_link, pre_link = plans
            print(f"[grasp_pick]   {g.name}: SUCCESS — executing "
                  f"(cand {i+1}/{n_try})")
            self.execute_plan(result_pre, arm_tag)
            self.execute_plan(result_grasp, arm_tag)
            return grasp_link, pre_link, g

        print(f"[grasp_pick] {object_name}: no candidate planned successfully "
              f"after {n_try} tries")
        return None

    def try_grasp_by_name(
        self,
        object_name: str,
        object_pose: Pose,
        arm_tag: str,
        grasp_name: str,
        model_id: int = None,
        robot_type: str = None,
        pre_dist: float = None,
        object_scale: float = 1.0,
        execute: bool = True,
    ):
        """Force a specific named grasp.  Skips the ranker; runs the same
        plan + FK check as :meth:`select_and_execute_grasp`.  When
        ``execute=True`` (default) runs ``execute_plan`` for both legs and
        returns ``(grasp_link, pre_link, grasp_pose)``.  When
        ``execute=False`` returns ``(result_pre, result_grasp, grasp_link,
        pre_link, grasp_pose)`` so the caller can interleave snapshots
        between the legs.  Returns ``None`` if the grasp isn't in the YAML
        or any planning step fails.
        """
        if robot_type is None:
            robot_type = self.config.get("robot_type")

        grasps = load_grasp_poses(
            object_name, model_id=model_id, robot_type=robot_type,
        )
        match = next((g for g in grasps if g.name == grasp_name), None)
        if match is None:
            print(f"[grasp_pick] {object_name}: grasp_name={grasp_name!r} not in YAML "
                  f"({len(grasps)} grasps available)")
            return None

        plans = self._plan_one_grasp(
            match, object_pose, arm_tag,
            pre_dist=pre_dist, object_scale=object_scale,
            log_prefix="[forced] ",
        )
        if plans is None:
            return None
        result_pre, result_grasp, grasp_link, pre_link = plans
        if execute:
            self.execute_plan(result_pre, arm_tag)
            self.execute_plan(result_grasp, arm_tag)
            return grasp_link, pre_link, match
        return result_pre, result_grasp, grasp_link, pre_link, match

    # ------------------------------------------------------------------
    # Gripper helpers
    # ------------------------------------------------------------------

    def set_gripper(self, val: float, arm_tag: str, num_steps: int = 200):
        """Gradually set gripper value."""
        arm = self.robot.get_arm(arm_tag)
        current = arm.gripper_val
        trajectory = np.linspace(current, val, num_steps)
        for v in trajectory:
            self.robot.set_gripper(float(v), arm_tag)
            self.step_sim()

    def open_gripper(self, arm_tag: str, num_steps: int = 200):
        self.set_gripper(1.0, arm_tag, num_steps)

    def close_gripper(self, arm_tag: str, num_steps: int = 200):
        self.set_gripper(0.0, arm_tag, num_steps)

    # ------------------------------------------------------------------
    # Video recording
    # ------------------------------------------------------------------

    def start_video(self, path: str):
        """Start recording video frames from the recording camera."""
        self._video_path = path
        self._video_frames = []

    @contextlib.contextmanager
    def suppress_recording(self):
        """Context manager: inside it, ``step_sim`` records nothing (video,
        VLA, trajectory).  Wrap pre-task scene settling / avatar+scene layout
        so those static "let everything fall to rest" steps never land in the
        video or the training data.  Nests safely (restores the prior state)."""
        prev = self._record_suppressed
        self._record_suppressed = True
        try:
            yield
        finally:
            self._record_suppressed = prev

    def capture_frame(self):
        """Capture one frame from the recording + side cameras for video."""
        if self._video_path is None:
            return
        self._video_cap_tick += 1
        if self._video_cap_tick % self._video_stride != 0:
            return
        from .camera import _finalize_rgb
        cam_entry = self.cameras._cameras.get("recording")
        if cam_entry is None:
            return
        cam = cam_entry[0]
        out = cam.render(rgb=True)
        rgb_raw = out[0] if isinstance(out, (list, tuple)) and len(out) > 0 else None
        rec_frame = _finalize_rgb(cam, rgb_raw)
        if rec_frame is None:
            return

        # Tile with side camera if available
        side_entry = self.cameras._cameras.get("side")
        if side_entry is not None:
            side_cam = side_entry[0]
            side_out = side_cam.render(rgb=True)
            side_raw = side_out[0] if isinstance(side_out, (list, tuple)) and len(side_out) > 0 else None
            side_frame = _finalize_rgb(side_cam, side_raw)
            if side_frame is not None:
                rec_frame = np.concatenate([rec_frame, side_frame], axis=1)

        self._video_frames.append(rec_frame)

    def save_video(self, fps: int = 30):
        """Save captured frames as MP4 video."""
        if not self._video_frames or self._video_path is None:
            return
        import imageio
        os.makedirs(os.path.dirname(self._video_path), exist_ok=True)
        writer_kwargs = {"fps": fps}
        quality = self.config.get("video_quality")
        if quality is not None:
            writer_kwargs["quality"] = int(quality)
        ffmpeg_params = self.config.get("video_ffmpeg_params")
        if ffmpeg_params:
            writer_kwargs["ffmpeg_params"] = [str(x) for x in ffmpeg_params]
        try:
            writer = imageio.get_writer(
                self._video_path,
                codec=self.config.get("video_codec", "libx264"),
                macro_block_size=int(self.config.get("video_macro_block_size", 1)),
                **writer_kwargs,
            )
        except TypeError:
            writer = imageio.get_writer(self._video_path, **writer_kwargs)
        for frame in self._video_frames:
            writer.append_data(frame)
        writer.close()
        print(f"  Video saved: {self._video_path} ({len(self._video_frames)} frames)")
        self._video_frames = []

    # ------------------------------------------------------------------
    # Data saving
    # ------------------------------------------------------------------

    def record_frame(self):
        """Record current state as a trajectory frame."""
        obs = self.get_obs()
        if self._record_cameras is not None:
            if "rgb" in obs:
                obs["rgb"] = {k: v for k, v in obs["rgb"].items() if k in self._record_cameras}
            if "depth" in obs:
                obs["depth"] = {k: v for k, v in obs["depth"].items() if k in self._record_cameras}
        self.traj_data.append(obs)
        self.FRAME_IDX += 1

    def save_trajectory(self, path: str):
        """Save recorded trajectory to pickle file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = {
            "observations": self.traj_data,
            "left_joint_path": self.left_joint_path,
            "right_joint_path": self.right_joint_path,
            "instruction": self.instruction,
            "evaluation": self.evaluate(),
        }
        # Only attach the per-step collision log when the checker was
        # active this episode.  (The summary is already in the
        # `evaluation` dict via `avatar_collision_summary()`.)
        if payload["evaluation"].get("avatar_collision", {}).get("enabled"):
            payload["avatar_collision_log"] = self.avatar_collision_log
        if payload["evaluation"].get("avatar_safety_intervention", {}).get("enabled"):
            payload["avatar_safety_intervention_log"] = self.avatar_safety_intervention_log
        with open(path, "wb") as f:
            pickle.dump(payload, f)

    def _move_to_pose(self, pose, arm_tag: str):
        """IK-based single-step move (for take_action ee mode)."""
        arm = self.robot.get_arm(arm_tag)
        result = arm.planner.solve_ik(arm.get_arm_qpos(), pose, num_waypoints=1)
        if result.success and result.position.size > 0:
            self.robot.set_arm_joints(result.position[-1], arm_tag)
