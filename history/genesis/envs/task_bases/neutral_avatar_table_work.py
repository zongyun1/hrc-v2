"""Shared neutral-avatar table-work behavior.

Neutral variants keep the robot's original task objects untouched by the
avatar.  The avatar performs a separate tabletop pick/place using props
created by this mixin, while robot planning can include the live avatar
capsule point cloud as an mplib obstacle.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import transforms3d as t3d

from ..avatar.utils import ActionStatus, AvatarState
from ..base_task import BaseTask
from ..genesis_compat import ensure_inertial_urdf, ensure_nyx_urdf
from ..utils import (
    ASSETS_PATH,
    Pose,
    create_primitive,
    load_mesh,
    load_object,
    load_urdf,
    to_numpy,
)


ROOT_PATH = Path(__file__).resolve().parents[2]
_NEUTRAL_MOTION_PKL = str(
    ROOT_PATH / "assets" / "neutral_motions" / "raw_blender_interaction_clips.pkl"
)
_NEUTRAL_MOTION_META = str(
    ROOT_PATH / "assets" / "neutral_motions" / "raw_blender_interaction_clips.json"
)


def _resolve_repo_path(p) -> str:
    """Resolve a config-provided path portably.

    Absolute paths pass through; relative paths resolve against the repo root
    (ROOT_PATH), so configs can use repo-relative paths like
    ``assets/neutral_motions/typing_clip.json`` and stay portable across
    machines regardless of the current working directory.
    """
    p = Path(str(p))
    return str(p if p.is_absolute() else ROOT_PATH / p)


@dataclass(frozen=True)
class NeutralTaskSpec:
    """Declarative neutral-avatar placement config for one task.

    The mixin still consumes the historical ``NEUTRAL_*`` class attributes.
    ``apply_to_class`` hydrates those attributes from one structured spec, so
    task files can stay declarative while existing rollout code remains stable.
    """

    work_xy: tuple[float, float] | None = None
    mouse_work_xy: tuple[float, float] | None = None
    sponge_work_xy: tuple[float, float] | None = None
    table_xy_bounds: tuple[float, float, float, float] | None = None
    max_table_edge_gap: float | None = None
    min_table_edge_gap: float | None = None
    avatar_as_planner_obstacle: bool | None = None
    use_region_placement: bool | None = None
    blocked_avatar_sides: tuple[str, ...] | None = None
    work_candidates: tuple[tuple[float, float], ...] | None = None
    mouse_work_candidates: tuple[tuple[float, float], ...] | None = None
    sponge_work_candidates: tuple[tuple[float, float], ...] | None = None
    work_region: tuple[float, float, float, float] | None = None
    mouse_work_region: tuple[float, float, float, float] | None = None
    sponge_work_region: tuple[float, float, float, float] | None = None
    sponge_side_samples: int | None = None

    @staticmethod
    def _xy(value: tuple[float, float]) -> np.ndarray:
        return np.asarray(value, dtype=np.float64)

    @staticmethod
    def _bounds(value: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
        return tuple(float(v) for v in value)

    @staticmethod
    def _candidates(value: tuple[tuple[float, float], ...]) -> list[np.ndarray]:
        return [np.asarray(xy, dtype=np.float64) for xy in value]

    def apply_to_class(self, cls: type) -> None:
        mapping = {
            "work_xy": ("NEUTRAL_WORK_XY", self._xy),
            "mouse_work_xy": ("NEUTRAL_MOUSE_WORK_XY", self._xy),
            "sponge_work_xy": ("NEUTRAL_SPONGE_WORK_XY", self._xy),
            "table_xy_bounds": ("NEUTRAL_TABLE_XY_BOUNDS", self._bounds),
            "max_table_edge_gap": ("NEUTRAL_AVATAR_MAX_TABLE_EDGE_GAP", float),
            "min_table_edge_gap": ("NEUTRAL_AVATAR_MIN_TABLE_EDGE_GAP", float),
            "avatar_as_planner_obstacle": ("NEUTRAL_AVATAR_AS_PLANNER_OBSTACLE", bool),
            "use_region_placement": ("NEUTRAL_USE_REGION_PLACEMENT", bool),
            "blocked_avatar_sides": (
                "NEUTRAL_AVATAR_BLOCKED_SIDES",
                lambda value: tuple(str(side) for side in value),
            ),
            "work_candidates": ("NEUTRAL_WORK_CANDIDATES", self._candidates),
            "mouse_work_candidates": ("NEUTRAL_MOUSE_WORK_CANDIDATES", self._candidates),
            "sponge_work_candidates": ("NEUTRAL_SPONGE_WORK_CANDIDATES", self._candidates),
            "work_region": ("NEUTRAL_WORK_REGION", self._bounds),
            "mouse_work_region": ("NEUTRAL_MOUSE_WORK_REGION", self._bounds),
            "sponge_work_region": ("NEUTRAL_SPONGE_WORK_REGION", self._bounds),
            "sponge_side_samples": ("NEUTRAL_SPONGE_SIDE_SAMPLES", int),
        }
        for field_name, (attr_name, converter) in mapping.items():
            value = getattr(self, field_name)
            if value is not None:
                setattr(cls, attr_name, converter(value))


class NeutralAvatarTableWorkMixin:
    """Add an unrelated avatar tabletop job to a robot task."""

    use_avatar = True
    NEUTRAL_TASK_SPEC: NeutralTaskSpec | None = None
    NEUTRAL_MOTION_FOOTPRINTS: dict[str, dict] = {}
    NEUTRAL_AVATAR_MOTION_CHOICES = (
        "wipe_table_2_reverse",
        "wipe_table_2",
        "put_objects_in_bowl",
        "take_objects_from_bowl",
        "use_mouse",
        "open_bottle",
        "writing",
        "using_a_fax",
        "Typing",
        "sitting_clap",
    )
    # Short handles AND the historical raw clip names (KIT/GRAB stage-ii) both
    # resolve to the current clean motion keys, so older configs / saved runs
    # keep working after the rename.
    NEUTRAL_AVATAR_MOTION_ALIASES = {
        # wipe_table_2_reverse replaces the retired wipe_table_1 clip.
        "wipe_table_1": "wipe_table_2_reverse",
        "wiping_table": "wipe_table_2_reverse",
        "wipe_table": "wipe_table_2_reverse",
        "sponge": "wipe_table_2_reverse",
        "01_KIT_wipe_table_wiping_the_table01_stageii": "wipe_table_2_reverse",
        "wipe_table_reverse": "wipe_table_2_reverse",
        "wipe_table_2_reverse": "wipe_table_2_reverse",
        # wipe_table_2
        "03_KIT_wipe_table_wiping_the_table05_stageii": "wipe_table_2",
        # put_objects_in_bowl
        "putting_cans_into_plate": "put_objects_in_bowl",
        "put_cans_into_plate": "put_objects_in_bowl",
        "cans_plate": "put_objects_in_bowl",
        "can_plate": "put_objects_in_bowl",
        "09_KIT_put_objects_in_bowl_put_objects_in_mixing_bowl_02_stageii": "put_objects_in_bowl",
        # take_objects_from_bowl (reverse of put_objects_in_bowl: lift cans out)
        "reverse_cans": "take_objects_from_bowl",
        "take_cans_from_plate": "take_objects_from_bowl",
        "unload_bowl": "take_objects_from_bowl",
        # use_mouse
        "mouse": "use_mouse",
        "using_mouse": "use_mouse",
        "14_GRAB_use_mouse_mouse_use_1_stageii": "use_mouse",
        # open_bottle
        "bottle": "open_bottle",
        "open_water_bottle": "open_bottle",
        "18_GRAB_open_water_bottle_waterbottle_open_1_stageii": "open_bottle",
        # writing
        "write": "writing",
        "writing_on_table": "writing",
        # using_a_fax
        "laptop": "using_a_fax",
        "fax": "using_a_fax",
        "using_laptop": "using_a_fax",
        # Typing (seated laptop+mouse)
        "typing": "Typing",
        "type": "Typing",
        "typing_on_laptop": "Typing",
        # sitting_clap (seated clapping, chair only)
        "clap": "sitting_clap",
        "clapping": "sitting_clap",
        "sitting_clapping": "sitting_clap",
    }

    # Motion-metadata object handles are intentionally presentation-oriented
    # (``bottle``, ``mouse``, ``object_1``), while robot targets use catalog
    # asset IDs (``001_bottle``, ``047_mouse``, ...).  Resolve the fixed props
    # here so the neutral-motion sampler can compare actual object identity.
    NEUTRAL_PROP_ASSET_IDS = {
        "bottle": "001_bottle",
        "mouse": "047_mouse",
        "phone": "077_phone",
        "pen": "058_markpen",
        "notebook": "092_notebook",
        "sponge": "cc0_sponge_3",
        "plate": "003_plate",
        "laptop": "015_laptop",
        "chair": "sapien-chair-179",
    }

    AVATAR_BASE_ROT = np.array(
        [[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=np.float64
    )
    avatar_init_pos = np.array([0.08, 0.58, -0.18], dtype=np.float64)
    avatar_init_rot = AVATAR_BASE_ROT

    NEUTRAL_PROP_HALF = 0.018
    NEUTRAL_SPONGE_HALF_THICKNESS = 0.015
    NEUTRAL_PROP_PICK_XY = np.array([0.34, 0.10], dtype=np.float64)
    NEUTRAL_PROP_PLACE_XY = np.array([0.46, -0.08], dtype=np.float64)
    NEUTRAL_AVATAR_NATURAL_BODY_MARGIN = 0.34
    NEUTRAL_AVATAR_NATURAL_BODY_Y_BOUNDS = (0.28, 0.66)
    NEUTRAL_AVATAR_MAX_TABLE_EDGE_GAP = 0.62
    NEUTRAL_AVATAR_TURN_RAD = 0.40
    NEUTRAL_AVATAR_TURN_STEP_RAD = 0.002
    NEUTRAL_AVATAR_MOTION_SLOW = 25.0
    NEUTRAL_EVAL_TRIGGER_STEP_MIN = 10
    NEUTRAL_EVAL_TRIGGER_STEP_MAX = 60
    NEUTRAL_AVATAR_HOLD_OFFSET = 0.05
    NEUTRAL_AVATAR_BODY_Z_OFFSET = 0.0
    NEUTRAL_PUT_OBJECTS_AVATAR_TABLE_Z_OFFSET = 0.0
    NEUTRAL_PUT_OBJECTS_BODY_EDGE_GAP = 0.48
    NEUTRAL_AVATAR_BOTTLE_HOLD_OFFSET = -0.04
    NEUTRAL_AVATAR_BOTTLE_BODY_Z_OFFSET = 0.0
    NEUTRAL_AVATAR_MOUSE_BODY_Z_OFFSET = 0.04
    NEUTRAL_MOUSE_CENTER_Z_OFFSET = 0.012
    # For the table-facing bottle clip, avatar-forward is world -Y and
    # avatar-right is world -X.
    NEUTRAL_AVATAR_BOTTLE_EXTRA_BODY_OFFSET = np.zeros(3, dtype=np.float64)
    NEUTRAL_AVATAR_BOTTLE_HAND_SWITCH_FRAMES = (44, 162)
    NEUTRAL_AVATAR_MOUSE_IDLE_Y_OFFSET = 0.0
    NEUTRAL_MOUSE_CLIP_TRIM = (8, 120)
    NEUTRAL_MOUSE_ATTACHED_FORWARD_LOOPS = 3
    NEUTRAL_PUT_OBJECTS_CAN_MODEL_IDS = (0, 1, 2, 3, 5, 6)
    NEUTRAL_PUT_OBJECTS_CAN_REST_Z_OFFSET = 0.002
    # The put-/take-objects motion's two handled props default to soda cans
    # (071_can, whose mesh origin is at the base, so resting z only needs a
    # small tabletop clearance).
    # Both the asset and the resting offset are config-overridable so a motion
    # can swap in e.g. apples (035_apple, mesh origin at the base → small offset
    # so the fruit rests just above the tabletop). Override via
    # `neutral_put_objects_object` / `neutral_put_objects_object_rest_z_offset`.
    NEUTRAL_PUT_OBJECTS_OBJECT = "071_can"
    NEUTRAL_PUT_OBJECTS_OBJECT_REST_Z_OFFSET = None  # None → can base clearance
    NEUTRAL_CAN_CLEARANCE_MARGIN = 0.015
    NEUTRAL_CAN_GRIPPER_CLEARANCE_RADIUS = 0.035
    NEUTRAL_CAN_UPRIGHT_DOT_MIN = 0.85
    NEUTRAL_AVATAR_TRANSITION_BASE_FRAMES = 40
    NEUTRAL_WORK_XY = np.array([0.34, 0.10], dtype=np.float64)
    NEUTRAL_MOUSE_WORK_XY = np.array([0.34, 0.22], dtype=np.float64)
    NEUTRAL_SPONGE_WORK_XY = None
    NEUTRAL_PUT_OBJECTS_IMAGE_SHIFT_XY = np.array([0.0, 0.20], dtype=np.float64)
    NEUTRAL_TABLE_XY_BOUNDS = (-0.50, 0.50, -0.80, 0.18)
    NEUTRAL_CLEARANCE_MARGIN = 0.015
    NEUTRAL_PROP_RADII = {
        "sponge": 0.070,
        "mouse": 0.050,
        "bottle": 0.040,
        "bottle_1": 0.055,
        "bottle_2": 0.055,
        "bottle_3": 0.055,
        "object_1": 0.040,
        "object_2": 0.040,
    }
    NEUTRAL_BOWL_HALF_XY = np.array([0.12, 0.08], dtype=np.float64)
    NEUTRAL_PUT_OBJECTS_PLATE_EDGE_MARGIN = 0.06
    NEUTRAL_PUT_OBJECTS_SWEEP_TABLE_MARGIN = 0.06
    NEUTRAL_PUT_OBJECTS_TABLE_MARGIN = 0.15
    NEUTRAL_SPONGE_TABLE_MARGIN = 0.040
    NEUTRAL_BOTTLE_TABLE_MARGIN = 0.100
    NEUTRAL_AVATAR_MIN_TABLE_EDGE_GAP = 0.160
    NEUTRAL_USE_REGION_PLACEMENT = True
    NEUTRAL_ARM_REGION_BONES = (
        ("LeftArm", 0.045),
        ("LeftForeArm", 0.040),
        ("LeftHand", 0.035),
        ("LeftHandMiddle1", 0.025),
        ("RightArm", 0.045),
        ("RightForeArm", 0.040),
        ("RightHand", 0.035),
        ("RightHandMiddle1", 0.025),
    )
    NEUTRAL_ROBOT_REGION_RADIUS = 0.12
    NEUTRAL_ROBOT_BASE_REGION_RADIUS = 0.20

    # Seated-writing neutral clip (`writing`): the avatar sits on a chair just
    # off the table edge and writes in a notebook with a pen in the right
    # hand.  The avatar–pen–book–chair relative poses and the avatar–table
    # edge gap are all fixed; only the anchor (book center) slides along the
    # chosen table edge.  Reference values come from
    # `scripts/render_writing.py --mode probe` with the avatar reset at the
    # origin facing +X (motion frame).
    NEUTRAL_WRITING_BOOK_LOCAL_XY = (0.5213, -0.0375)  # book center, avatar frame
    NEUTRAL_WRITING_BOOK_EDGE_INSET = 0.22       # book center -> table edge
    NEUTRAL_WRITING_TABLE_TOP_REF = 0.7413       # hand-fit tabletop at body z=0
    NEUTRAL_WRITING_SEAT_HEIGHT_REF = 0.46       # chair seat height at body z=0
    NEUTRAL_WRITING_BOOK_SCALE = 0.18            # 092_notebook -> ~0.34 x 0.22 m
    NEUTRAL_WRITING_BOOK_HALF_LONG = 0.171       # long edge, across the writer
    NEUTRAL_WRITING_BOOK_HALF_DEPTH = 0.113      # depth, along the facing dir
    NEUTRAL_WRITING_PEN_SCALE = 0.1              # 058_markpen -> 0.19 m long
    NEUTRAL_WRITING_PEN_FWD = 0.01               # pen tip from palm, finger dir
    NEUTRAL_WRITING_PEN_TILT_DEG = 18.0          # lean back toward the writer
    NEUTRAL_WRITING_CHAIR_RAW_BOTTOM_Z = 0.832048  # sapien-chair-179 raw geom
    NEUTRAL_WRITING_CHAIR_RAW_SEAT_H = 0.786
    NEUTRAL_WRITING_BOOK_CLEARANCE = 0.02        # extra ring around the book
    NEUTRAL_WRITING_TABLE_EDGE_MARGIN = 0.03     # book AABB -> lateral edges
    NEUTRAL_WRITING_CANDIDATE_STEP = 0.05        # lateral sweep step along edge
    # Over-table body envelope of the seated writer (avatar frame, facing +X):
    # both hands rest over the book and the head leans in past the table
    # edge.  Checked at selection time, and used to keep the anchor as far
    # from the robot's workspace as possible (the live capsule obstacles only
    # protect mplib plans, not screw/IK moves).
    NEUTRAL_WRITING_BODY_POINTS = (
        ("body:right_hand", (0.5213, -0.0375), 0.09),
        ("body:left_hand", (0.616, 0.108), 0.09),
        ("body:head", (0.32, 0.0), 0.14),
    )
    NEUTRAL_WRITING_ROBOT_REGION_PREFIXES = ("robot_base:", "gripper:")
    # Extra standoff between the body envelope and everything the gripper
    # visits (task objects, slots, targets): static disks don't capture the
    # arm's sweep, so contact-distance alone is not enough.
    NEUTRAL_WRITING_BODY_SWEEP_MARGIN = 0.08

    # Standing-laptop neutral clip (`using_a_fax`): the avatar stands just off
    # the table edge typing on an open SAPIEN laptop (015_laptop/9960).  The
    # avatar–laptop relative pose and the avatar–table-edge gap are fixed;
    # only the anchor (laptop origin) slides along the chosen table edge.
    # Reference values come from the v9 calibration of
    # `scripts/render_fax_laptop.py` (probe data/laptop_fax/probe.json with
    # --laptop-scale 0.28 --avatar-yaw-deg 30 --avatar-back-offset 0.10
    # --avatar-left-offset 0.20).
    NEUTRAL_LAPTOP_LOCAL_XY = (0.79575, -0.054349)  # laptop origin, avatar frame
    NEUTRAL_LAPTOP_YAW_LOCAL_DEG = -13.6678      # laptop yaw - avatar yaw
    NEUTRAL_LAPTOP_EDGE_INSET = 0.20964          # laptop origin -> table edge
    NEUTRAL_LAPTOP_TABLE_TOP_REF = 0.937215      # implied tabletop at body z=0
    NEUTRAL_LAPTOP_SCALE = 0.28                  # 015_laptop/9960 -> ~0.32 m wide
    NEUTRAL_LAPTOP_RAW_BOTTOM_Z = 0.223          # pre-scale shell bottom below origin
    NEUTRAL_LAPTOP_RAW_X_BOUNDS = (-0.713, 0.297)  # pre-scale: front edge .. lid lean
    NEUTRAL_LAPTOP_RAW_Y_BOUNDS = (-0.582, 0.571)
    NEUTRAL_LAPTOP_LID_QPOS = 0.0                # open lid; held against gravity
    NEUTRAL_LAPTOP_CLEARANCE = 0.02              # extra ring around the laptop
    NEUTRAL_LAPTOP_TABLE_EDGE_MARGIN = 0.03      # laptop AABB -> lateral edges
    NEUTRAL_LAPTOP_CANDIDATE_STEP = 0.05         # lateral sweep step along edge
    NEUTRAL_LAPTOP_CLIP_TRIM = (122, 414)        # stationary typing window
    NEUTRAL_LAPTOP_PINGPONG_LOOPS = 3
    # Minimum gap (m) the laptop footprint must keep from every task region
    # for a placement to be accepted (a non-overlap floor).  The selector
    # then picks the MAX-clearance placement among those that pass, so the
    # laptop ends up as far from the robot's working area as the table allows
    # — dump_bin yields ~9 cm, the blocks tasks ~4-5 cm.
    NEUTRAL_LAPTOP_TASK_CLEARANCE = 0.01
    # Sub-centimetre grazes (e.g. the forearm skimming the open screen tip)
    # are not a meaningful "touch"; require this much penetration (m) into a
    # neutral object before the collision checker flags it.
    NEUTRAL_OBJECT_COLLISION_TOLERANCE = 0.012

    # Seated-typing neutral clip (`Typing`): the avatar sits on a chair just
    # off the table edge typing on an open SAPIEN laptop (015_laptop/9960)
    # with a mouse (047_mouse) on its right.  Combines the writing chair logic
    # with the laptop-on-table logic.  The avatar-laptop-mouse relative poses
    # and the avatar-table-edge gap are fixed; only the anchor (laptop origin)
    # slides along the chosen table edge.  Reference values come from
    # `scripts/render_typing.py --mode probe` with the avatar reset at the
    # origin (identity rotation) — so the avatar-local frame is the world
    # frame and the probed world poses ARE the local-frame constants.
    NEUTRAL_TYPING_LAPTOP_LOCAL_XY = (0.476, 0.057)   # laptop origin, avatar frame
    NEUTRAL_TYPING_LAPTOP_YAW_LOCAL_DEG = 6.07        # laptop yaw - avatar yaw
    NEUTRAL_TYPING_MOUSE_LOCAL_XY = (0.546, -0.229)   # mouse center, avatar frame
    NEUTRAL_TYPING_MOUSE_YAW_LOCAL_DEG = -83.93       # mouse yaw - avatar yaw
    NEUTRAL_TYPING_MOUSE_Z_OFFSET = 0.012             # mouse origin above tabletop
    NEUTRAL_TYPING_EDGE_INSET = 0.21                  # laptop origin -> table edge
    NEUTRAL_TYPING_TABLE_TOP_REF = 0.6976             # implied tabletop at body z=0
    NEUTRAL_TYPING_SEAT_HEIGHT_REF = 0.46             # chair seat height at body z=0
    NEUTRAL_TYPING_SCALE = 0.28                       # 015_laptop/9960 -> ~0.32 m wide
    NEUTRAL_TYPING_MOUSE_SCALE_MODEL_ID = 0           # 047_mouse model_data0 (scale 0.5)
    NEUTRAL_TYPING_LID_QPOS = 0.0                     # open lid; held against gravity
    NEUTRAL_TYPING_CLEARANCE = 0.02                   # extra ring around the footprint
    NEUTRAL_TYPING_TABLE_EDGE_MARGIN = 0.03           # footprint AABB -> lateral edges
    NEUTRAL_TYPING_CANDIDATE_STEP = 0.05              # lateral sweep step along edge
    NEUTRAL_TYPING_CLIP_TRIM = (15, 480)              # drop T-pose blend-in/out frames
    NEUTRAL_TYPING_TASK_CLEARANCE = 0.01              # min footprint gap to task regions
    NEUTRAL_TYPING_MOUSE_HALF = 0.05                  # mouse footprint half-extent (m)

    # Seated-clapping neutral clip (`sitting_clap`): the avatar sits on the
    # same SAPIEN chair as the writing/typing clips, alongside the table with
    # no held/tabletop prop.  The edge anchor is the nearest table edge point
    # in the avatar frame; only the anchor slides along a clear table side.
    NEUTRAL_CLAP_TABLE_DISTANCE = 0.34
    NEUTRAL_CLAP_EDGE_ANCHOR_LOCAL_XY = (NEUTRAL_CLAP_TABLE_DISTANCE, 0.0)
    NEUTRAL_CLAP_TABLE_TOP_REF = 0.74
    NEUTRAL_CLAP_SEAT_HEIGHT_REF = 0.46
    NEUTRAL_CLAP_BODY_Z_OFFSET = -0.075
    NEUTRAL_CLAP_CHAIR_Z_OFFSET = -0.10
    NEUTRAL_CLAP_CLIP_TRIM = (14, 142)
    NEUTRAL_CLAP_EDGE_INSET = 0.02
    NEUTRAL_CLAP_TABLE_EDGE_MARGIN = 0.03
    NEUTRAL_CLAP_CANDIDATE_STEP = 0.05
    NEUTRAL_CLAP_TASK_CLEARANCE = 0.01
    NEUTRAL_CLAP_BODY_SWEEP_MARGIN = 0.08
    # Avatar-local points around the clapping hands/head, relative to the
    # table-edge anchor, used as the on-table clearance footprint.
    NEUTRAL_CLAP_LOCAL_REGION_POINTS = (
        ("body:hands", (0.22, 0.0), 0.16),
        ("body:left_hand", (0.22, 0.12), 0.10),
        ("body:right_hand", (0.22, -0.12), 0.10),
        ("body:head", (-0.04, 0.0), 0.14),
    )

    # Loop the neutral motion back-and-forth (forward -> reverse -> forward
    # ...) for the whole episode instead of playing once and freezing.  A
    # forward+reverse ping-pong clip is built at start time (seamless
    # turnarounds at both ends) and rewound on completion; events fire on
    # the forward half each pass.  Default on for every neutral motion/task.
    NEUTRAL_AVATAR_LOOP_MOTION = True

    _AVATAR_OBSTACLE_RES = 0.04
    _AVATAR_INFLATE_FACTOR = 1.6
    NEUTRAL_AVATAR_AS_PLANNER_OBSTACLE = True
    # The neutral objects (the human's laptop / book / chair / cans / etc.) are
    # also off-limits to the robot: their world AABBs are pushed into the
    # planner obstacle cloud so plans route around them, and into the analytic
    # collision checker so a touch trips the avatar-collision success gate.
    NEUTRAL_OBJECT_AS_OBSTACLE = True
    NEUTRAL_OBJECT_OBSTACLE_INFLATE = 0.01  # metres added around each object AABB

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        spec = cls.__dict__.get("NEUTRAL_TASK_SPEC")
        if spec is None:
            return
        if isinstance(spec, dict):
            spec = NeutralTaskSpec(**spec)
            cls.NEUTRAL_TASK_SPEC = spec
        if not isinstance(spec, NeutralTaskSpec):
            raise TypeError(
                f"{cls.__name__}.NEUTRAL_TASK_SPEC must be NeutralTaskSpec or dict"
            )
        spec.apply_to_class(cls)

    def __init__(self, config: dict = None):
        cfg = dict(config or {})
        cfg.setdefault("use_avatar_collider", True)
        cfg.setdefault("track_avatar_collision", True)
        avatar_cfg = dict(cfg.get("avatar") or {})
        motion_slow = float(cfg.get(
            "neutral_avatar_motion_slow",
            self.NEUTRAL_AVATAR_MOTION_SLOW,
        ))
        avatar_cfg.setdefault(
            "generated_motion_data",
            _resolve_repo_path(cfg.get("neutral_avatar_motion_pkl", _NEUTRAL_MOTION_PKL)),
        )
        avatar_cfg.setdefault("frame_ratio", motion_slow)
        cfg["avatar"] = avatar_cfg
        super().__init__(cfg)

    def _neutral_avatar_motion_slow(self) -> float:
        return max(
            1.0,
            float(self.config.get(
                "neutral_avatar_motion_slow",
                self.NEUTRAL_AVATAR_MOTION_SLOW,
            )),
        )

    def _neutral_avatar_forced_motion_requested(self) -> bool:
        return bool(str((self.config or {}).get("neutral_avatar_motion", "")).strip())

    def _apply_neutral_camera_overrides(self) -> None:
        """Let a neutral config reposition the recording/side review cameras.

        BaseTask builds the scene cameras from ``self.recording_camera_*`` /
        ``self.side_camera_*`` (class attrs, default None). A neutral motion
        whose avatar would block the default angle can set these via config —
        e.g. the seated take-objects clip. No-op unless the keys are present, so
        every other task keeps its default cameras.
        """
        for cfg_key, attr in (
            ("neutral_recording_camera_pos", "recording_camera_pos"),
            ("neutral_recording_camera_lookat", "recording_camera_lookat"),
            ("neutral_side_camera_pos", "side_camera_pos"),
            ("neutral_side_camera_lookat", "side_camera_lookat"),
        ):
            val = (self.config or {}).get(cfg_key)
            if val is not None:
                setattr(self, attr, list(val))

    def load_actors(self):
        # Keep the historical RNG order: sample the initial neutral job before
        # the parent task samples/spawns its robot targets.  Once those targets
        # are known, reject or replace a semantically duplicate neutral job.
        self._resolve_neutral_avatar_job()
        super().load_actors()
        self._ensure_neutral_avatar_job_is_target_distinct()
        self._load_neutral_avatar_props()
        self._load_neutral_avatar_region_debug_tiles()

    def reset(self, seed: int = 0):
        if hasattr(self, "_neutral_original_step_sim"):
            self.step_sim = self._neutral_original_step_sim
            del self._neutral_original_step_sim
        if hasattr(self, "_neutral_writing_orig_step_sim"):
            self.step_sim = self._neutral_writing_orig_step_sim
            del self._neutral_writing_orig_step_sim
        self._neutral_avatar_reset_seed = int(seed)
        self._neutral_avatar_started = False
        self._neutral_laptop_hold_state = None
        self._neutral_obb_cache = {}
        self._neutral_avatar_job = None
        self._neutral_avatar_step_state = None
        self._neutral_avatar_motion_footprint = None
        self._neutral_avatar_selected_work_xy = None
        self._neutral_avatar_region_debug_records = []
        self._neutral_avatar_region_debug_entities = []
        self._neutral_avatar_can_model_ids = {}
        self._neutral_avatar_layout_ready = False
        self._neutral_eval_policy_step_count = 0
        self._neutral_eval_trigger_step = None
        self._neutral_eval_avatar_failed = False
        self._apply_neutral_camera_overrides()
        obs = super().reset(seed=seed)
        if (
            self.avatar is not None
            and bool(self.config.get("eval_mode", False))
            and "neutral_avatar_strict_layout" not in self.config
        ):
            self.config["neutral_avatar_strict_layout"] = False
        if bool(self.config.get("neutral_avatar_skip_start_layout", False)):
            self._neutral_avatar_layout_ready = False
            return obs
        if (self.config or {}).get("scene_init"):
            self._neutral_avatar_layout_ready = False
        self._prepare_neutral_avatar_start_layout()
        self._install_neutral_object_collision_detection()
        if self.avatar is not None and bool(self.config.get("eval_mode", False)):
            lo = int(self.config.get(
                "neutral_eval_trigger_step_min",
                self.NEUTRAL_EVAL_TRIGGER_STEP_MIN,
            ))
            hi = int(self.config.get(
                "neutral_eval_trigger_step_max",
                self.NEUTRAL_EVAL_TRIGGER_STEP_MAX,
            ))
            if hi < lo:
                hi = lo
            self._neutral_eval_trigger_step = int(np.random.randint(lo, hi + 1))
        return obs

    def take_action(self, action, action_type: str = "qpos"):
        if self.avatar is not None and bool(self.config.get("eval_mode", False)):
            self._neutral_eval_policy_step_count = (
                int(getattr(self, "_neutral_eval_policy_step_count", 0)) + 1
            )
            trigger = getattr(self, "_neutral_eval_trigger_step", None)
            if (
                trigger is not None
                and not getattr(self, "_neutral_avatar_started", False)
                and not getattr(self, "_neutral_eval_avatar_failed", False)
                and self._neutral_eval_policy_step_count >= int(trigger)
            ):
                try:
                    self._start_neutral_avatar_table_work(add_start_delay=False)
                except Exception as exc:
                    self._neutral_eval_avatar_failed = True
                    if self.config.get("debug_eval_avatar", False):
                        print(
                            f"[neutral_avatar/eval] start failed: {exc}",
                            flush=True,
                        )
        return super().take_action(action, action_type=action_type)

    def _load_neutral_avatar_props(self):
        if self._neutral_avatar_job is None:
            self._resolve_neutral_avatar_job()
        job = self._neutral_avatar_job
        if job["kind"] == "writing":
            self._load_neutral_writing_props(job)
            return
        if job["kind"] == "laptop":
            self._load_neutral_laptop_props(job)
            return
        if job["kind"] == "typing":
            self._load_neutral_typing_props(job)
            return
        if job["kind"] == "sitting_clap":
            self._load_neutral_sitting_clap_props(job)
            return
        work_xy = self._neutral_avatar_default_work_xy(job)
        self.neutral_avatar_props = {}

        for obj_idx, obj_name in enumerate(job["objects"]):
            prop_xy = self._neutral_avatar_prebuild_prop_xy(job, work_xy, obj_name, obj_idx)
            pos = np.array(
                [prop_xy[0], prop_xy[1], self._neutral_avatar_prop_center_z(obj_name)],
                dtype=float,
            )
            self.neutral_avatar_props[obj_name] = self._create_neutral_avatar_prop(
                obj_name, pos
            )
        if job["kind"] == "put_objects_in_bowl" and bool(self.config.get(
            "neutral_put_objects_plate",
            self._neutral_put_objects_clip_defaults().get("use_plate", True),
        )):
            self.neutral_avatar_plate = load_mesh(
                self.scene,
                ASSETS_PATH / "objects" / "003_plate" / "visual" / "base0.glb",
                Pose(
                    [float(work_xy[0]), float(work_xy[1]), self.TABLE_TOP_Z + 0.020],
                    self._neutral_avatar_prop_quat("plate"),
                ),
                scale=0.025,
                convex=True,
                is_static=False,
                collision=True,
                friction=3.0,
                density=250.0,
            )
            self.neutral_avatar_bowl = self.neutral_avatar_plate

        first_obj = job["objects"][0]
        self.neutral_avatar_prop = self.neutral_avatar_props[first_obj]
        self.neutral_avatar_target = np.array(
            [
                self._neutral_avatar_default_work_xy(job)[0],
                self._neutral_avatar_default_work_xy(job)[1],
                self.TABLE_TOP_Z,
            ],
            dtype=float,
        )

    # ------------------------------------------------------------------
    # Seated-writing neutral clip
    # ------------------------------------------------------------------

    def _neutral_writing_dz(self) -> float:
        """Vertical shift from the probe reference tabletop to this task's."""
        table_top = float(getattr(self, "TABLE_TOP_Z", 0.765))
        return table_top - float(self.NEUTRAL_WRITING_TABLE_TOP_REF)

    def _neutral_writing_chair_scale(self) -> float:
        seat = float(self.NEUTRAL_WRITING_SEAT_HEIGHT_REF) + self._neutral_writing_dz()
        return seat / float(self.NEUTRAL_WRITING_CHAIR_RAW_SEAT_H)

    def _neutral_urdf_merge_fixed_links(self) -> bool | None:
        """NYX renders URDF entities as sub-scenes that re-parse the file and
        need the file's link tree to map 1:1 onto the simulated links, so the
        chair / laptop URDFs must keep their fixed links un-merged under NYX.
        None keeps Genesis's default (merged) for every other renderer."""
        return False if str(self.config.get("renderer", "")) == "nyx" else None

    def _neutral_chair_urdf(self):
        """sapien-chair-179 mobility.urdf ships no <inertial> tags; with the
        NYX un-merged link tree Genesis 1.x auto-inertia goes degenerate and
        the first settle step nan-crashes (same failure as 036_cabinet)."""
        return ensure_inertial_urdf(
            ASSETS_PATH / "objects" / "sapien-chair-179" / "mobility.urdf"
        )

    def _neutral_laptop_urdf(self):
        """015_laptop/9960 mobility.urdf also ships without <inertial> tags."""
        return ensure_inertial_urdf(
            ASSETS_PATH / "objects" / "015_laptop" / "9960" / "mobility.urdf"
        )

    def _neutral_load_prop_urdf(self, urdf_path, pose: Pose, scale: float):
        """Load a fixed chair/laptop prop URDF, NYX-safe.

        NYX's sub-scene re-parse drops fixed-joint child links (the chair
        body / laptop keyboard hang under a fixed root joint → they vanish,
        leaving floating wheels / a detached lid) and ignores Genesis's
        runtime ``scale``. Under NYX, load a sibling file with the fixed
        joints pre-merged and the scale baked in (see ensure_nyx_urdf).
        """
        merge = self._neutral_urdf_merge_fixed_links()
        if merge is False:
            urdf_path = ensure_nyx_urdf(urdf_path, scale)
            scale = 1.0
        return load_urdf(
            self.scene, urdf_path, pose,
            scale=float(scale), fix_root=True, merge_fixed_links=merge,
        )

    def _load_neutral_writing_props(self, job: dict) -> None:
        """Create the pen / notebook / chair, parked until layout selection.

        All three are repositioned by `_prepare_neutral_writing_layout`; the
        pen is fixed-base because the avatar hand attachment drives it purely
        kinematically (set_pos/set_quat each step).
        """
        chair_scale = self._neutral_writing_chair_scale()
        self.neutral_avatar_props = {
            "pen": load_mesh(
                self.scene,
                ASSETS_PATH / "objects" / "058_markpen" / "visual" / "base0.glb",
                Pose([6.0, 6.4, 1.0]),
                scale=float(self.NEUTRAL_WRITING_PEN_SCALE),
                is_static=True,
                collision=False,
            )
        }
        self.neutral_avatar_prop = self.neutral_avatar_props["pen"]
        self.neutral_avatar_book = load_mesh(
            self.scene,
            ASSETS_PATH / "objects" / "092_notebook" / "visual" / "base0.glb",
            Pose([6.0, 6.8, 1.0]),
            scale=float(self.NEUTRAL_WRITING_BOOK_SCALE),
            is_static=True,
            collision=False,
        )
        self.neutral_avatar_chair = self._neutral_load_prop_urdf(
            self._neutral_chair_urdf(),
            Pose([6.0, 6.0, float(self.NEUTRAL_WRITING_CHAIR_RAW_BOTTOM_Z) * chair_scale]),
            chair_scale,
        )
        self.neutral_avatar_target = np.array(
            [6.0, 6.8, float(getattr(self, "TABLE_TOP_Z", 0.765))], dtype=float
        )

    _NEUTRAL_WRITING_SIDE_INWARD = {
        "back": (0.0, -1.0),
        "left": (1.0, 0.0),
        "right": (-1.0, 0.0),
    }

    def _neutral_writing_side_candidates(self, job: dict) -> list[tuple[str, np.ndarray]]:
        """(side, book-center) candidates at the fixed edge inset, center-out.

        The book inset from the table edge is fixed, so the only placement
        freedom is the lateral slide along the edge.  Lateral spans come from
        the task's side work regions (which encode "not the robot side"),
        falling back to the generic side bands.
        """
        xmin, xmax, ymin, ymax = self._neutral_table_xy_bounds()
        inset = float(self.NEUTRAL_WRITING_BOOK_EDGE_INSET)
        half_long = float(self.NEUTRAL_WRITING_BOOK_HALF_LONG)
        edge_margin = float(self.NEUTRAL_WRITING_TABLE_EDGE_MARGIN)
        step = float(self.NEUTRAL_WRITING_CANDIDATE_STEP)
        side_regions = list(self._neutral_avatar_allowed_side_work_regions(job) or [])
        if not side_regions:
            side_regions = list(self._neutral_avatar_default_side_work_regions())
        elif self._neutral_avatar_forced_motion_requested():
            known = {
                (str(side), tuple(float(v) for v in region))
                for side, region in side_regions
            }
            for side, region in self._neutral_avatar_default_side_work_regions():
                key = (str(side), tuple(float(v) for v in region))
                if key not in known:
                    side_regions.append((side, region))
                    known.add(key)
        forced_side = str(self.config.get("neutral_avatar_force_side", "") or "")
        out: list[tuple[str, np.ndarray]] = []
        seen_sides = set()
        for side, region in side_regions:
            side = str(side)
            if side not in self._NEUTRAL_WRITING_SIDE_INWARD or side in seen_sides:
                continue
            if forced_side and side != forced_side:
                continue
            seen_sides.add(side)
            x0, x1, y0, y1 = [float(v) for v in region]
            if side == "back":
                lat_lo = max(x0, float(xmin) + half_long + edge_margin)
                lat_hi = min(x1, float(xmax) - half_long - edge_margin)
                fixed = float(ymax) - inset
                to_xy = lambda lat, fx=fixed: np.array([lat, fx], dtype=np.float64)
            else:
                lat_lo = max(y0, float(ymin) + half_long + edge_margin)
                lat_hi = min(y1, float(ymax) - half_long - edge_margin)
                fixed = (float(xmin) + inset) if side == "left" else (float(xmax) - inset)
                to_xy = lambda lat, fx=fixed: np.array([fx, lat], dtype=np.float64)
            if lat_hi < lat_lo:
                continue
            center = 0.5 * (lat_lo + lat_hi)
            lats = [center]
            k = 1
            while center + k * step <= lat_hi or center - k * step >= lat_lo:
                if center + k * step <= lat_hi:
                    lats.append(center + k * step)
                if center - k * step >= lat_lo:
                    lats.append(center - k * step)
                k += 1
            for lat in lats:
                out.append((side, to_xy(float(lat))))
        return out

    def _neutral_writing_footprint(self, side: str) -> dict:
        """Neutral region of the writing motion = the book's region."""
        clearance = float(self.NEUTRAL_WRITING_BOOK_CLEARANCE)
        half_long = float(self.NEUTRAL_WRITING_BOOK_HALF_LONG) + clearance
        half_depth = float(self.NEUTRAL_WRITING_BOOK_HALF_DEPTH) + clearance
        if side == "back":
            half = np.array([half_long, half_depth], dtype=np.float64)
        else:
            half = np.array([half_depth, half_long], dtype=np.float64)
        return {
            "motion": "writing",
            "kind": "writing",
            "min_offset": -half,
            "max_offset": half,
            "size": 2.0 * half,
            "region_points": [],
        }

    def _prepare_neutral_writing_layout(self, job: dict) -> bool:
        """Fixed-relative-pose layout for the seated writing clip.

        Pose chain (all fixed): book center = anchor on the table at the edge
        inset; avatar root = anchor - R_body @ BOOK_LOCAL_XY (which puts the
        body at a fixed gap outside the table edge); chair under the avatar;
        pen attached near-vertical at the page under the right hand.
        """
        self.avatar.frame_ratio = self._neutral_avatar_motion_slow()
        table_top = float(getattr(self, "TABLE_TOP_Z", 0.765))
        dz = self._neutral_writing_dz()
        task_regions = self.neutral_task_regions()
        candidates = self._neutral_writing_side_candidates(job)
        if not candidates:
            raise RuntimeError(
                "no writing-side candidates: table bounds/side regions leave "
                "no room for the notebook at the fixed edge inset"
            )

        margin = float(self.NEUTRAL_CLEARANCE_MARGIN)
        book_local = np.asarray(self.NEUTRAL_WRITING_BOOK_LOCAL_XY, dtype=np.float64)
        body_points = [
            (str(label), np.asarray(p, dtype=np.float64).ravel()[:2], float(r))
            for label, p, r in self.NEUTRAL_WRITING_BODY_POINTS
        ]
        robot_prefixes = tuple(self.NEUTRAL_WRITING_ROBOT_REGION_PREFIXES)
        rot2_by_side = {}
        # The robot's planned sweeps are not captured by the static task-object
        # disks, so among valid candidates pick the one farthest from the
        # robot regions (base + gripper/TCP) — the seated writer leans well
        # over the tabletop and cannot dodge the arm.
        selected = None        # best (book + body) clear candidate
        best_book_only = None  # (robot_dist, side, xy, footprint, record)
        best_any = None        # (book clearance, side, xy, footprint, record)
        debug_records = []
        for side, xy in candidates:
            footprint = self._neutral_writing_footprint(side)
            book_clear = self._neutral_footprint_is_clear(xy, footprint, task_regions)
            min_xy = xy + footprint["min_offset"]
            max_xy = xy + footprint["max_offset"]
            rot2 = rot2_by_side.get(side)
            if rot2 is None:
                rot2 = np.asarray(
                    self._neutral_avatar_body_rot_for_side(side, job),
                    dtype=np.float64,
                )[:2, :2]
                rot2_by_side[side] = rot2
            pts_world = [
                (label, xy + rot2 @ (p - book_local), r)
                for label, p, r in body_points
            ]
            body_clear = True
            book_worst = float("inf")
            robot_dist = float("inf")
            for center, radius, label in task_regions:
                center = np.asarray(center, dtype=np.float64).ravel()[:2]
                radius = float(radius)
                clamped = np.minimum(np.maximum(center, min_xy), max_xy)
                d_book = float(np.linalg.norm(center - clamped)) - radius - margin
                book_worst = min(book_worst, d_book)
                d_body = min(
                    float(np.linalg.norm(center - pt)) - radius - r_pt - margin
                    for _lbl, pt, r_pt in pts_world
                )
                if str(label).startswith(robot_prefixes):
                    robot_dist = min(robot_dist, d_book, d_body)
                    # The arm sweeps, so keep the seated body envelope a clear
                    # margin from the robot base/gripper regions.  Short
                    # tabletop objects are NOT gated here: the head/hands lean
                    # ABOVE them (verified 0 robot-avatar collisions), and the
                    # book footprint already prevents table-surface overlap.
                    if d_body < float(self.NEUTRAL_WRITING_BODY_SWEEP_MARGIN):
                        body_clear = False
            record = {
                "xy": np.asarray(xy, dtype=np.float64).copy(),
                "side": side,
                "footprint": footprint,
                "footprint_clear": bool(book_clear),
                "neutral_region_clear": bool(book_clear),
                "footprint_table_ok": True,
                "side_region_ok": True,
                "body_ok": bool(body_clear),
                "prop_clearance": robot_dist if np.isfinite(robot_dist) else 1.0,
                "prop_table_clearance": book_worst if np.isfinite(book_worst) else 1.0,
                "selected": False,
            }
            debug_records.append(record)
            if book_clear and body_clear:
                if selected is None or robot_dist > selected[0]:
                    selected = (robot_dist, side, xy, footprint, record)
            elif book_clear:
                if best_book_only is None or robot_dist > best_book_only[0]:
                    best_book_only = (robot_dist, side, xy, footprint, record)
            if best_any is None or book_worst > best_any[0]:
                best_any = (book_worst, side, xy, footprint, record)
        self._neutral_avatar_region_debug_records = debug_records
        if selected is None:
            # Body-clear is mandatory under strict layout: a grazing-margin
            # anchor reads as a robot-through-avatar collision later, and a
            # raise lets the collector retry with a new task-object seed.
            if bool(self.config.get("neutral_avatar_strict_layout", True)):
                raise RuntimeError(
                    "no writing layout candidate keeps the book region and the "
                    "seated body envelope clear of task objects; motion=writing"
                )
            if best_book_only is not None:
                selected = best_book_only
                self._neutral_debug(
                    "[neutral_avatar] writing: body envelope grazes a task "
                    "region on every clear candidate; using the "
                    "farthest-from-robot one"
                )
        if selected is None:
            _worst, side, xy, footprint, record = best_any
            record["selected"] = True
            selected = (None, side, xy, footprint, record)
            self._neutral_debug(
                f"[neutral_avatar] WARNING: no clear writing candidate; using "
                f"{np.asarray(xy).round(3).tolist()} (clearance={_worst:.3f})"
            )
        _robot_dist, side, anchor_xy, footprint, record = selected
        record["selected"] = True
        if bool(self.config.get("neutral_avatar_debug_layout", False)):
            print(f"[writing_layout] table_bounds={self._neutral_table_xy_bounds()}")
            for center, radius, label in task_regions:
                print(
                    f"[writing_layout] region {label}: "
                    f"center={np.round(np.asarray(center, dtype=float).ravel()[:2], 3).tolist()} "
                    f"r={float(radius):.3f}"
                )
            for rec in debug_records:
                print(
                    f"[writing_layout] cand side={rec['side']} "
                    f"xy={rec['xy'].round(3).tolist()} book={rec['footprint_clear']} "
                    f"body={rec['body_ok']} robot_dist={float(rec['prop_clearance']):.3f}"
                    + (" <- selected" if rec.get("selected") else "")
                )
        anchor_xy = np.asarray(anchor_xy, dtype=np.float64)

        body_rot = self._neutral_avatar_body_rot_for_side(side, job)
        rot2 = np.asarray(body_rot, dtype=np.float64)[:2, :2]
        book_local = np.asarray(self.NEUTRAL_WRITING_BOOK_LOCAL_XY, dtype=np.float64)
        body_xy = anchor_xy - rot2 @ book_local
        body_pos = np.array([body_xy[0], body_xy[1], dz], dtype=np.float64)
        forward = rot2 @ np.array([1.0, 0.0], dtype=np.float64)
        facing = float(np.arctan2(forward[1], forward[0]))

        # Chair under the avatar, wheels on the floor, backrest behind her.
        chair = self.neutral_avatar_chair
        chair_scale = self._neutral_writing_chair_scale()
        chair_yaw = facing + np.pi  # asset faces -X
        chair.set_pos(np.array([
            body_xy[0],
            body_xy[1],
            float(self.NEUTRAL_WRITING_CHAIR_RAW_BOTTOM_Z) * chair_scale,
        ], dtype=float))
        chair.set_quat(np.array(
            [np.cos(chair_yaw / 2.0), 0.0, 0.0, np.sin(chair_yaw / 2.0)],
            dtype=float,
        ))
        if getattr(chair, "n_qs", 0):
            chair_hold = np.zeros((int(chair.n_qs),), dtype=np.float64)
            chair.set_qpos(chair_hold)
            # Wheel / gas-lift joints sag over long rollouts; re-pin per step.
            if not hasattr(self, "_neutral_writing_orig_step_sim"):
                original_step_sim = self.step_sim
                self._neutral_writing_orig_step_sim = original_step_sim

                def chair_pinned_step_sim():
                    original_step_sim()
                    try:
                        chair.set_qpos(chair_hold)
                    except Exception:
                        pass

                self.step_sim = chair_pinned_step_sim

        # Notebook flat on the table, long edge across the writing direction.
        book_R = (
            t3d.euler.euler2mat(0.0, 0.0, facing + np.pi / 2.0, "sxyz")
            @ t3d.euler.euler2mat(np.pi / 2.0, 0.0, 0.0, "sxyz")
        )
        self.neutral_avatar_book.set_pos(np.array(
            [anchor_xy[0], anchor_xy[1], table_top + 0.001], dtype=float
        ))
        self.neutral_avatar_book.set_quat(
            t3d.quaternions.mat2quat(book_R).astype(float)
        )

        # Seat the avatar and hold the first writing frame.
        self.avatar.reset(body_pos, np.asarray(body_rot, dtype=np.float64))
        self._neutral_avatar_hold_motion_frame_idle(job["motion"], 0)

        # Pen near-vertical at the page: tip just above the tabletop, slightly
        # ahead of the palm along the finger direction, leaned back 18 deg.
        pen = self.neutral_avatar_props["pen"]
        hand_pos, hand_R = self.avatar.robot._get_hand_frame(1)
        finger = np.asarray(hand_R, dtype=np.float64)[:, 1].copy()
        finger[2] = 0.0
        norm = float(np.linalg.norm(finger))
        finger = finger / norm if norm > 1e-6 else np.array([forward[0], forward[1], 0.0])
        pen_fwd = float(self.NEUTRAL_WRITING_PEN_FWD)
        tip = np.array([
            float(hand_pos[0]) + finger[0] * pen_fwd,
            float(hand_pos[1]) + finger[1] * pen_fwd,
            table_top + 0.002,
        ], dtype=np.float64)
        tilt_axis = np.array([-finger[1], finger[0], 0.0], dtype=np.float64)
        pen_R = (
            t3d.axangles.axangle2mat(
                tilt_axis, np.deg2rad(float(self.NEUTRAL_WRITING_PEN_TILT_DEG))
            )
            @ t3d.euler.euler2mat(np.pi / 2.0, 0.0, 0.0, "sxyz")
        )
        pen.set_pos(tip.astype(float))
        pen.set_quat(t3d.quaternions.mat2quat(pen_R).astype(float))
        self.avatar.robot.attach_object_to_hand(1, pen)
        # The pen is held for the whole clip; nothing left for the event loop.
        job["events"] = []

        self._neutral_avatar_selected_work_xy = anchor_xy
        self._neutral_avatar_motion_footprint = footprint
        type(self).NEUTRAL_MOTION_FOOTPRINTS[str(job["base_motion"])] = {
            "kind": "writing",
            "min_offset": np.asarray(footprint["min_offset"], dtype=float).tolist(),
            "max_offset": np.asarray(footprint["max_offset"], dtype=float).tolist(),
            "size": np.asarray(footprint["size"], dtype=float).tolist(),
        }
        self.neutral_avatar_target = np.array(
            [anchor_xy[0], anchor_xy[1], table_top], dtype=np.float64
        )
        self._neutral_avatar_body_pos = body_pos
        self._neutral_avatar_body_rot = np.asarray(body_rot, dtype=np.float64)
        self._neutral_avatar_layout_ready = True
        self.avatar_collided = False
        self.avatar_collision_log = []
        self._avatar_collision_tick = 0
        self._neutral_debug(
            f"[neutral_avatar] writing layout side={side} "
            f"book={anchor_xy.round(3).tolist()} body={body_pos.round(3).tolist()}"
        )
        self._create_neutral_avatar_region_debug_overlays(job, task_regions)
        return True

    # ------------------------------------------------------------------
    # Standing-laptop neutral clip
    # ------------------------------------------------------------------

    def _neutral_laptop_dz(self) -> float:
        """Vertical shift from the calibration tabletop to this task's."""
        table_top = float(getattr(self, "TABLE_TOP_Z", 0.765))
        return table_top - float(self.NEUTRAL_LAPTOP_TABLE_TOP_REF)

    def _load_neutral_laptop_props(self, job: dict) -> None:
        """Create the laptop, parked until layout selection.

        The laptop is fixed-base (a static scene prop); its base pose is set
        by `_prepare_neutral_laptop_layout` and its lid joint is re-held at
        the open angle every sim step by the patched step loop.
        """
        scale = float(self.NEUTRAL_LAPTOP_SCALE)
        laptop = self._neutral_load_prop_urdf(
            self._neutral_laptop_urdf(),
            Pose([6.0, 6.0, float(self.NEUTRAL_LAPTOP_RAW_BOTTOM_Z) * scale + 1.0]),
            scale,
        )
        self.neutral_avatar_props = {"laptop": laptop}
        self.neutral_avatar_prop = laptop
        self._neutral_laptop_hold_state = None
        self.neutral_avatar_target = np.array(
            [6.0, 6.0, float(getattr(self, "TABLE_TOP_Z", 0.765))], dtype=float
        )

    def _install_neutral_laptop_pingpong_subclip(self, job: dict) -> None:
        """Trim to the stationary typing window and ping-pong loop it."""
        if job.get("laptop_pingpong") is not None:
            return
        src_name = str(job["base_motion"])
        md = self.avatar.motion_data[src_name]
        n = int(md["trans"].shape[0])
        start, end = self.NEUTRAL_LAPTOP_CLIP_TRIM
        start = int(np.clip(start, 0, n - 2))
        end = int(np.clip(end, start + 1, n - 1))
        loops = int(self.config.get(
            "neutral_avatar_laptop_pingpong_loops",
            self.NEUTRAL_LAPTOP_PINGPONG_LOOPS,
        ))
        loops = int(np.clip(loops, 1, 6))
        forward = list(range(start, end + 1))
        backward = list(range(end - 1, start, -1))
        indices = list(forward)
        for _ in range(loops - 1):
            indices.extend(backward)
            indices.extend(forward)
        indices = np.asarray(indices, dtype=int)
        dst_name = f"{src_name}_laptop_pingpong_{loops}"
        self.avatar.motion_data[dst_name] = {k: v[indices].copy() for k, v in md.items()}
        job["motion"] = dst_name
        job["laptop_pingpong"] = loops
        job["clip"] = dict(job["clip"])
        job["clip"]["num_frames"] = int(indices.shape[0])

    def _neutral_laptop_world_yaws(self, side: str) -> tuple[float, float]:
        """(laptop_yaw, avatar_yaw) in world for a table side.

        The laptop asset's display faces its local -X, so its +X axis points
        inward (away from the avatar standing outside the edge).
        """
        inward = self._NEUTRAL_WRITING_SIDE_INWARD[str(side)]
        laptop_yaw = float(np.arctan2(float(inward[1]), float(inward[0])))
        avatar_yaw = laptop_yaw - float(np.deg2rad(self.NEUTRAL_LAPTOP_YAW_LOCAL_DEG))
        return laptop_yaw, avatar_yaw

    def _neutral_laptop_footprint(self, side: str) -> dict:
        """Neutral region of the laptop motion = the laptop's region."""
        scale = float(self.NEUTRAL_LAPTOP_SCALE)
        clearance = float(self.NEUTRAL_LAPTOP_CLEARANCE)
        x0, x1 = (float(v) * scale for v in self.NEUTRAL_LAPTOP_RAW_X_BOUNDS)
        y0, y1 = (float(v) * scale for v in self.NEUTRAL_LAPTOP_RAW_Y_BOUNDS)
        laptop_yaw, _ = self._neutral_laptop_world_yaws(side)
        c, s = float(np.cos(laptop_yaw)), float(np.sin(laptop_yaw))
        rot = np.array([[c, -s], [s, c]], dtype=np.float64)
        corners = (rot @ np.array(
            [[x0, y0], [x0, y1], [x1, y0], [x1, y1]], dtype=np.float64
        ).T).T
        min_off = corners.min(axis=0) - clearance
        max_off = corners.max(axis=0) + clearance
        # The laptop's front edge overhangs the table edge by a few cm at the
        # calibrated inset (by design — see render_fax_laptop.py v9).  Clamp
        # the outward extent to the edge so the recorded neutral region is the
        # on-table part and the inside-table placement check can pass.
        inset = float(self.NEUTRAL_LAPTOP_EDGE_INSET) - 1e-3
        if side == "back":
            max_off[1] = min(float(max_off[1]), inset)
        elif side == "left":
            min_off[0] = max(float(min_off[0]), -inset)
        elif side == "right":
            max_off[0] = min(float(max_off[0]), inset)
        return {
            "motion": "using_a_fax",
            "kind": "laptop",
            "min_offset": min_off,
            "max_offset": max_off,
            "size": max_off - min_off,
            "region_points": [],
        }

    def _neutral_laptop_side_candidates(self, job: dict) -> list[tuple[str, np.ndarray]]:
        """(side, laptop-origin) candidates at the fixed edge inset, center-out."""
        xmin, xmax, ymin, ymax = self._neutral_table_xy_bounds()
        inset = float(self.NEUTRAL_LAPTOP_EDGE_INSET)
        edge_margin = float(self.NEUTRAL_LAPTOP_TABLE_EDGE_MARGIN)
        step = float(self.NEUTRAL_LAPTOP_CANDIDATE_STEP)
        side_regions = list(self._neutral_avatar_allowed_side_work_regions(job) or [])
        if not side_regions:
            side_regions = list(self._neutral_avatar_default_side_work_regions())
        elif self._neutral_avatar_forced_motion_requested():
            known = {
                (str(side), tuple(float(v) for v in region))
                for side, region in side_regions
            }
            for side, region in self._neutral_avatar_default_side_work_regions():
                key = (str(side), tuple(float(v) for v in region))
                if key not in known:
                    side_regions.append((side, region))
                    known.add(key)
        forced_side = str(self.config.get("neutral_avatar_force_side", "") or "")
        out: list[tuple[str, np.ndarray]] = []
        seen_sides = set()
        for side, region in side_regions:
            side = str(side)
            if side not in self._NEUTRAL_WRITING_SIDE_INWARD or side in seen_sides:
                continue
            if forced_side and side != forced_side:
                continue
            seen_sides.add(side)
            footprint = self._neutral_laptop_footprint(side)
            min_off = np.asarray(footprint["min_offset"], dtype=np.float64)
            max_off = np.asarray(footprint["max_offset"], dtype=np.float64)
            x0, x1, y0, y1 = [float(v) for v in region]
            if side == "back":
                lat_lo = max(x0, float(xmin) - float(min_off[0]) + edge_margin)
                lat_hi = min(x1, float(xmax) - float(max_off[0]) - edge_margin)
                fixed = float(ymax) - inset
                to_xy = lambda lat, fx=fixed: np.array([lat, fx], dtype=np.float64)
            else:
                lat_lo = max(y0, float(ymin) - float(min_off[1]) + edge_margin)
                lat_hi = min(y1, float(ymax) - float(max_off[1]) - edge_margin)
                fixed = (float(xmin) + inset) if side == "left" else (float(xmax) - inset)
                to_xy = lambda lat, fx=fixed: np.array([fx, lat], dtype=np.float64)
            if lat_hi < lat_lo:
                continue
            center = 0.5 * (lat_lo + lat_hi)
            lats = [center]
            k = 1
            while center + k * step <= lat_hi or center - k * step >= lat_lo:
                if center + k * step <= lat_hi:
                    lats.append(center + k * step)
                if center - k * step >= lat_lo:
                    lats.append(center - k * step)
                k += 1
            for lat in lats:
                out.append((side, to_xy(float(lat))))
        return out

    def _prepare_neutral_laptop_layout(self, job: dict) -> bool:
        """Fixed-relative-pose layout for the standing laptop clip.

        Pose chain (all fixed): laptop origin = anchor on the table at the
        edge inset, display facing outward; avatar root = anchor -
        R(avatar_yaw) @ LAPTOP_LOCAL_XY (a fixed gap outside the table edge);
        avatar root z keeps the typing hands hovering over the keyboard.
        """
        self._install_neutral_laptop_pingpong_subclip(job)
        self.avatar.frame_ratio = self._neutral_avatar_motion_slow()
        table_top = float(getattr(self, "TABLE_TOP_Z", 0.765))
        dz = self._neutral_laptop_dz()
        task_regions = self.neutral_task_regions()
        candidates = self._neutral_laptop_side_candidates(job)
        if not candidates:
            raise RuntimeError(
                "no laptop-side candidates: table bounds/side regions leave "
                "no room for the laptop at the fixed edge inset"
            )

        # Pick the candidate that puts the laptop FARTHEST from every task
        # object/corridor region (max clearance), not just the first that
        # doesn't overlap.  A few-mm gap still lets the gripper clip the
        # laptop while it grasps/carries an adjacent object, so we maximise
        # the gap subject to a small non-overlap floor.  Different tasks have
        # different headroom (dump_bin's dense corridors leave ~9 cm; the
        # blocks tasks only ~4-5 cm), so a fixed threshold would either
        # under-protect one or starve the other — max-clearance adapts.
        min_clearance = float(self.config.get(
            "neutral_laptop_min_clearance",
            self.NEUTRAL_LAPTOP_TASK_CLEARANCE,
        ))
        debug_records = []
        scored = []  # (clearance, side, xy, footprint, record)
        for side, xy in candidates:
            footprint = self._neutral_laptop_footprint(side)
            in_table = self._neutral_footprint_inside_table(xy, footprint)
            min_xy = xy + footprint["min_offset"]
            max_xy = xy + footprint["max_offset"]
            worst = float("inf")
            for center, radius, _label in task_regions:
                center = np.asarray(center, dtype=np.float64).ravel()[:2]
                clamped = np.minimum(np.maximum(center, min_xy), max_xy)
                worst = min(
                    worst,
                    float(np.linalg.norm(center - clamped)) - float(radius),
                )
            clearance = worst if np.isfinite(worst) else 1.0
            record = {
                "xy": np.asarray(xy, dtype=np.float64).copy(),
                "side": side,
                "footprint": footprint,
                "footprint_clear": bool(in_table and clearance > 0.0),
                "neutral_region_clear": bool(in_table and clearance > 0.0),
                "footprint_table_ok": bool(in_table),
                "side_region_ok": True,
                "body_ok": True,
                "prop_clearance": clearance,
                "prop_table_clearance": 1.0,
                "selected": False,
            }
            debug_records.append(record)
            if in_table:
                scored.append((clearance, side, xy, footprint, record))
        self._neutral_avatar_region_debug_records = debug_records
        if bool(self.config.get("neutral_avatar_debug_layout", False)):
            print(f"[laptop_layout] table_bounds={self._neutral_table_xy_bounds()}")
            for center, radius, label in task_regions:
                print(
                    f"[laptop_layout] region {label}: "
                    f"center={np.round(np.asarray(center, dtype=float).ravel()[:2], 3).tolist()} "
                    f"r={float(radius):.3f}"
                )
            for rec in debug_records:
                print(
                    f"[laptop_layout] cand side={rec['side']} "
                    f"xy={rec['xy'].round(3).tolist()} clear={rec['footprint_clear']} "
                    f"clearance={float(rec['prop_clearance']):.3f}"
                )
        feasible = [c for c in scored if c[0] >= min_clearance]
        pool = feasible or scored
        if not pool:
            raise RuntimeError(
                "no laptop layout candidate fits inside the table at the fixed "
                "edge inset; motion=using_a_fax"
            )
        best = max(pool, key=lambda c: c[0])
        if not feasible and bool(self.config.get("neutral_avatar_strict_layout", True)):
            raise RuntimeError(
                "no laptop layout candidate clears task objects by the minimum "
                f"clearance {min_clearance:.3f} m; best={best[0]:.3f} m; "
                "motion=using_a_fax"
            )
        _clear, side, anchor_xy0, footprint, record = best
        record["selected"] = True
        selected = (side, anchor_xy0, footprint)
        self._neutral_debug(
            f"[neutral_avatar] laptop max-clearance pick {np.asarray(anchor_xy0).round(3).tolist()} "
            f"side={side} clearance={_clear:.3f}"
        )
        side, anchor_xy, footprint = selected
        anchor_xy = np.asarray(anchor_xy, dtype=np.float64)

        laptop_yaw, avatar_yaw = self._neutral_laptop_world_yaws(side)
        c, s = float(np.cos(avatar_yaw)), float(np.sin(avatar_yaw))
        rot2 = np.array([[c, -s], [s, c]], dtype=np.float64)
        body_rot = np.eye(3, dtype=np.float64)
        body_rot[:2, :2] = rot2
        laptop_local = np.asarray(self.NEUTRAL_LAPTOP_LOCAL_XY, dtype=np.float64)
        body_xy = anchor_xy - rot2 @ laptop_local
        body_pos = np.array([body_xy[0], body_xy[1], dz], dtype=np.float64)

        # Laptop on the table at the anchor, display facing the avatar; the
        # lid hold below keeps the revolute lid at the open angle.
        scale = float(self.NEUTRAL_LAPTOP_SCALE)
        laptop = self.neutral_avatar_props["laptop"]
        laptop.set_pos(np.array([
            anchor_xy[0],
            anchor_xy[1],
            table_top + float(self.NEUTRAL_LAPTOP_RAW_BOTTOM_Z) * scale,
        ], dtype=float))
        laptop.set_quat(np.array(
            [np.cos(laptop_yaw / 2.0), 0.0, 0.0, np.sin(laptop_yaw / 2.0)],
            dtype=float,
        ))
        lid_qpos = None
        if getattr(laptop, "n_qs", 0):
            lid_qpos = np.full(
                (int(laptop.n_qs),),
                float(self.NEUTRAL_LAPTOP_LID_QPOS),
                dtype=np.float64,
            )
            laptop.set_qpos(lid_qpos)
            try:
                # PD-hold the lid so it stays open even before the patched
                # step loop starts re-applying the qpos every step.
                laptop.set_dofs_kp(np.full((int(laptop.n_qs),), 50.0))
                laptop.set_dofs_kv(np.full((int(laptop.n_qs),), 5.0))
                laptop.control_dofs_position(lid_qpos)
            except Exception:
                pass
        self._neutral_laptop_hold_state = (laptop, lid_qpos)

        # Stand the avatar at the fixed relative pose, holding frame 0.
        self.avatar.reset(body_pos, body_rot)
        self._neutral_avatar_hold_motion_frame_idle(job["motion"], 0)

        self._neutral_avatar_selected_work_xy = anchor_xy
        self._neutral_avatar_motion_footprint = footprint
        type(self).NEUTRAL_MOTION_FOOTPRINTS[str(job["base_motion"])] = {
            "kind": "laptop",
            "min_offset": np.asarray(footprint["min_offset"], dtype=float).tolist(),
            "max_offset": np.asarray(footprint["max_offset"], dtype=float).tolist(),
            "size": np.asarray(footprint["size"], dtype=float).tolist(),
        }
        self.neutral_avatar_target = np.array(
            [anchor_xy[0], anchor_xy[1], table_top], dtype=np.float64
        )
        self._neutral_avatar_body_pos = body_pos
        self._neutral_avatar_body_rot = body_rot
        self._neutral_avatar_layout_ready = True
        self.avatar_collided = False
        self.avatar_collision_log = []
        self._avatar_collision_tick = 0
        self._neutral_debug(
            f"[neutral_avatar] laptop layout side={side} "
            f"laptop={anchor_xy.round(3).tolist()} body={body_pos.round(3).tolist()}"
        )
        self._create_neutral_avatar_region_debug_overlays(job, task_regions)
        return True

    # ------------------------------------------------------------------
    # Seated-typing neutral clip (chair + laptop + mouse)
    # ------------------------------------------------------------------

    def _neutral_typing_dz(self) -> float:
        """Vertical shift from the calibration tabletop to this task's."""
        table_top = float(getattr(self, "TABLE_TOP_Z", 0.765))
        return table_top - float(self.NEUTRAL_TYPING_TABLE_TOP_REF)

    def _neutral_typing_chair_scale(self) -> float:
        seat = float(self.NEUTRAL_TYPING_SEAT_HEIGHT_REF) + self._neutral_typing_dz()
        return seat / float(self.NEUTRAL_WRITING_CHAIR_RAW_SEAT_H)

    def _load_neutral_typing_props(self, job: dict) -> None:
        """Create the chair / laptop / mouse, parked until layout selection.

        Chair + laptop are repositioned by `_prepare_neutral_typing_layout`.
        The laptop is fixed-base (static prop) with its lid held open; the
        mouse is a static prop on the table; the chair is pinned per step.
        Nothing is held in hand (the hands just type), so there are no events.
        """
        scale = float(self.NEUTRAL_TYPING_SCALE)
        chair_scale = self._neutral_typing_chair_scale()
        laptop = self._neutral_load_prop_urdf(
            self._neutral_laptop_urdf(),
            Pose([6.0, 6.0, float(self.NEUTRAL_LAPTOP_RAW_BOTTOM_Z) * scale + 1.0]),
            scale,
        )
        mouse = load_object(
            self.scene,
            Pose([6.0, 6.4, 1.0], self._neutral_typing_mouse_quat(0.0)),
            "047_mouse",
            model_id=int(self.NEUTRAL_TYPING_MOUSE_SCALE_MODEL_ID),
            convex=True,
            is_static=True,
        ).entity
        self.neutral_avatar_props = {"laptop": laptop, "mouse": mouse}
        self.neutral_avatar_prop = laptop
        self.neutral_avatar_mouse = mouse
        self.neutral_avatar_chair = self._neutral_load_prop_urdf(
            self._neutral_chair_urdf(),
            Pose([6.0, 6.0, float(self.NEUTRAL_WRITING_CHAIR_RAW_BOTTOM_Z) * chair_scale]),
            chair_scale,
        )
        self._neutral_laptop_hold_state = None
        self.neutral_avatar_target = np.array(
            [6.0, 6.0, float(getattr(self, "TABLE_TOP_Z", 0.765))], dtype=float
        )

    def _neutral_typing_mouse_quat(self, mouse_yaw: float) -> np.ndarray:
        """Map local +Y up to world +Z, then yaw about world +Z."""
        mouse_R = (
            t3d.euler.euler2mat(0.0, 0.0, float(mouse_yaw), "sxyz")
            @ t3d.euler.euler2mat(np.pi / 2.0, 0.0, 0.0, "sxyz")
        )
        return t3d.quaternions.mat2quat(mouse_R).astype(float)

    def _install_neutral_typing_subclip(self, job: dict) -> None:
        """Trim the T-pose blend-in/out frames before the central loop wrap."""
        if job.get("typing_trim"):
            return
        src = str(job["base_motion"])
        md = self.avatar.motion_data[src]
        n = int(md["trans"].shape[0])
        start, end = self.NEUTRAL_TYPING_CLIP_TRIM
        start = int(np.clip(start, 0, n - 2))
        end = int(np.clip(end, start + 1, n - 1))
        idx = np.arange(start, end + 1, dtype=int)
        dst = f"{src}_typing_trim"
        self.avatar.motion_data[dst] = {k: v[idx].copy() for k, v in md.items()}
        job["motion"] = dst
        job["typing_trim"] = True
        job["clip"] = dict(job["clip"])
        job["clip"]["num_frames"] = int(idx.shape[0])

    def _neutral_typing_world_yaws(self, side: str) -> tuple[float, float]:
        """(laptop_yaw, avatar_yaw) in world for a table side.

        Laptop display faces its local -X, so +X points inward (toward the
        table, away from the seated avatar outside the edge).
        """
        inward = self._NEUTRAL_WRITING_SIDE_INWARD[str(side)]
        laptop_yaw = float(np.arctan2(float(inward[1]), float(inward[0])))
        avatar_yaw = laptop_yaw - float(np.deg2rad(self.NEUTRAL_TYPING_LAPTOP_YAW_LOCAL_DEG))
        return laptop_yaw, avatar_yaw

    def _neutral_typing_footprint(self, side: str) -> dict:
        """On-table neutral region = laptop AABB unioned with the mouse AABB.

        Offsets are relative to the anchor (laptop origin world xy).
        """
        scale = float(self.NEUTRAL_TYPING_SCALE)
        clearance = float(self.NEUTRAL_TYPING_CLEARANCE)
        laptop_yaw, avatar_yaw = self._neutral_typing_world_yaws(side)
        x0, x1 = (float(v) * scale for v in self.NEUTRAL_LAPTOP_RAW_X_BOUNDS)
        y0, y1 = (float(v) * scale for v in self.NEUTRAL_LAPTOP_RAW_Y_BOUNDS)
        cl, sl = float(np.cos(laptop_yaw)), float(np.sin(laptop_yaw))
        rot_l = np.array([[cl, -sl], [sl, cl]], dtype=np.float64)
        lap_corners = (rot_l @ np.array(
            [[x0, y0], [x0, y1], [x1, y0], [x1, y1]], dtype=np.float64
        ).T).T
        ca, sa = float(np.cos(avatar_yaw)), float(np.sin(avatar_yaw))
        rot_a = np.array([[ca, -sa], [sa, ca]], dtype=np.float64)
        mouse_off = rot_a @ (
            np.asarray(self.NEUTRAL_TYPING_MOUSE_LOCAL_XY, dtype=np.float64)
            - np.asarray(self.NEUTRAL_TYPING_LAPTOP_LOCAL_XY, dtype=np.float64)
        )
        mh = float(self.NEUTRAL_TYPING_MOUSE_HALF)
        mouse_corners = mouse_off + np.array(
            [[-mh, -mh], [-mh, mh], [mh, -mh], [mh, mh]], dtype=np.float64
        )
        corners = np.vstack([lap_corners, mouse_corners])
        min_off = corners.min(axis=0) - clearance
        max_off = corners.max(axis=0) + clearance
        # The laptop's front edge sits ~at the table edge at the calibrated
        # inset (by design, like the standing laptop kind).  Clamp the OUTWARD
        # extent (toward the avatar/edge) to the inset so the recorded region
        # is the on-table part and the inside-table check can pass.
        inset = float(self.NEUTRAL_TYPING_EDGE_INSET) - 1e-3
        if side == "back":
            max_off[1] = min(float(max_off[1]), inset)
        elif side == "left":
            min_off[0] = max(float(min_off[0]), -inset)
        elif side == "right":
            max_off[0] = min(float(max_off[0]), inset)
        return {
            "motion": "Typing",
            "kind": "typing",
            "min_offset": min_off,
            "max_offset": max_off,
            "size": max_off - min_off,
            "region_points": [],
        }

    def _neutral_typing_side_candidates(self, job: dict) -> list[tuple[str, np.ndarray]]:
        """(side, laptop-origin) candidates at the fixed edge inset, center-out."""
        xmin, xmax, ymin, ymax = self._neutral_table_xy_bounds()
        inset = float(self.NEUTRAL_TYPING_EDGE_INSET)
        edge_margin = float(self.NEUTRAL_TYPING_TABLE_EDGE_MARGIN)
        step = float(self.NEUTRAL_TYPING_CANDIDATE_STEP)
        side_regions = list(self._neutral_avatar_allowed_side_work_regions(job) or [])
        if not side_regions:
            side_regions = list(self._neutral_avatar_default_side_work_regions())
        forced_side = str(self.config.get("neutral_avatar_force_side", "") or "")
        out: list[tuple[str, np.ndarray]] = []
        seen_sides = set()
        for side, region in side_regions:
            side = str(side)
            if side not in self._NEUTRAL_WRITING_SIDE_INWARD or side in seen_sides:
                continue
            if forced_side and side != forced_side:
                continue
            seen_sides.add(side)
            footprint = self._neutral_typing_footprint(side)
            min_off = np.asarray(footprint["min_offset"], dtype=np.float64)
            max_off = np.asarray(footprint["max_offset"], dtype=np.float64)
            x0, x1, y0, y1 = [float(v) for v in region]
            if side == "back":
                lat_lo = max(x0, float(xmin) - float(min_off[0]) + edge_margin)
                lat_hi = min(x1, float(xmax) - float(max_off[0]) - edge_margin)
                fixed = float(ymax) - inset
                to_xy = lambda lat, fx=fixed: np.array([lat, fx], dtype=np.float64)
            else:
                lat_lo = max(y0, float(ymin) - float(min_off[1]) + edge_margin)
                lat_hi = min(y1, float(ymax) - float(max_off[1]) - edge_margin)
                fixed = (float(xmin) + inset) if side == "left" else (float(xmax) - inset)
                to_xy = lambda lat, fx=fixed: np.array([fx, lat], dtype=np.float64)
            if lat_hi < lat_lo:
                continue
            center = 0.5 * (lat_lo + lat_hi)
            lats = [center]
            k = 1
            while center + k * step <= lat_hi or center - k * step >= lat_lo:
                if center + k * step <= lat_hi:
                    lats.append(center + k * step)
                if center - k * step >= lat_lo:
                    lats.append(center - k * step)
                k += 1
            for lat in lats:
                out.append((side, to_xy(float(lat))))
        return out

    def _prepare_neutral_typing_layout(self, job: dict) -> bool:
        """Fixed-relative-pose layout for the seated typing clip.

        Pose chain (all fixed): laptop origin = anchor on the table at the
        edge inset, display facing the avatar; avatar root = anchor -
        R(avatar_yaw) @ LAPTOP_LOCAL_XY (a fixed gap outside the table edge);
        chair under the avatar; mouse on the table to the avatar's right.
        """
        self._install_neutral_typing_subclip(job)
        self.avatar.frame_ratio = self._neutral_avatar_motion_slow()
        table_top = float(getattr(self, "TABLE_TOP_Z", 0.765))
        dz = self._neutral_typing_dz()
        task_regions = self.neutral_task_regions()
        candidates = self._neutral_typing_side_candidates(job)
        if not candidates:
            raise RuntimeError(
                "no typing-side candidates: table bounds/side regions leave "
                "no room for the laptop+mouse at the fixed edge inset"
            )

        # Max-clearance anchor: farthest from every task region, subject to a
        # small non-overlap floor (same policy as the laptop kind).
        min_clearance = float(self.config.get(
            "neutral_typing_min_clearance",
            self.NEUTRAL_TYPING_TASK_CLEARANCE,
        ))
        debug_records = []
        scored = []  # (clearance, side, xy, footprint, record)
        for side, xy in candidates:
            footprint = self._neutral_typing_footprint(side)
            in_table = self._neutral_footprint_inside_table(xy, footprint)
            min_xy = xy + footprint["min_offset"]
            max_xy = xy + footprint["max_offset"]
            worst = float("inf")
            for center, radius, _label in task_regions:
                center = np.asarray(center, dtype=np.float64).ravel()[:2]
                clamped = np.minimum(np.maximum(center, min_xy), max_xy)
                worst = min(
                    worst,
                    float(np.linalg.norm(center - clamped)) - float(radius),
                )
            clearance = worst if np.isfinite(worst) else 1.0
            record = {
                "xy": np.asarray(xy, dtype=np.float64).copy(),
                "side": side,
                "footprint": footprint,
                "footprint_clear": bool(in_table and clearance > 0.0),
                "neutral_region_clear": bool(in_table and clearance > 0.0),
                "footprint_table_ok": bool(in_table),
                "side_region_ok": True,
                "body_ok": True,
                "prop_clearance": clearance,
                "prop_table_clearance": 1.0,
                "selected": False,
            }
            debug_records.append(record)
            if in_table:
                scored.append((clearance, side, xy, footprint, record))
        self._neutral_avatar_region_debug_records = debug_records
        if bool(self.config.get("neutral_avatar_debug_layout", False)):
            print(f"[typing_layout] table_bounds={self._neutral_table_xy_bounds()}")
            for rec in debug_records:
                print(
                    f"[typing_layout] cand side={rec['side']} "
                    f"xy={rec['xy'].round(3).tolist()} clear={rec['footprint_clear']} "
                    f"clearance={float(rec['prop_clearance']):.3f}"
                )
        feasible = [c for c in scored if c[0] >= min_clearance]
        pool = feasible or scored
        if not pool:
            raise RuntimeError(
                "no typing layout candidate fits inside the table at the fixed "
                "edge inset; motion=Typing"
            )
        best = max(pool, key=lambda c: c[0])
        if not feasible and bool(self.config.get("neutral_avatar_strict_layout", True)):
            raise RuntimeError(
                "no typing layout candidate clears task objects by the minimum "
                f"clearance {min_clearance:.3f} m; best={best[0]:.3f} m; motion=Typing"
            )
        _clear, side, anchor_xy, footprint, record = best
        record["selected"] = True
        anchor_xy = np.asarray(anchor_xy, dtype=np.float64)
        self._neutral_debug(
            f"[neutral_avatar] typing max-clearance pick {anchor_xy.round(3).tolist()} "
            f"side={side} clearance={_clear:.3f}"
        )

        laptop_yaw, avatar_yaw = self._neutral_typing_world_yaws(side)
        c, s = float(np.cos(avatar_yaw)), float(np.sin(avatar_yaw))
        rot2 = np.array([[c, -s], [s, c]], dtype=np.float64)
        body_rot = np.eye(3, dtype=np.float64)
        body_rot[:2, :2] = rot2
        laptop_local = np.asarray(self.NEUTRAL_TYPING_LAPTOP_LOCAL_XY, dtype=np.float64)
        body_xy = anchor_xy - rot2 @ laptop_local
        body_pos = np.array([body_xy[0], body_xy[1], dz], dtype=np.float64)
        forward = rot2 @ np.array([1.0, 0.0], dtype=np.float64)
        facing = float(np.arctan2(forward[1], forward[0]))

        # Laptop on the table at the anchor, display facing the avatar.
        scale = float(self.NEUTRAL_TYPING_SCALE)
        laptop = self.neutral_avatar_props["laptop"]
        laptop.set_pos(np.array([
            anchor_xy[0],
            anchor_xy[1],
            table_top + float(self.NEUTRAL_LAPTOP_RAW_BOTTOM_Z) * scale,
        ], dtype=float))
        laptop.set_quat(np.array(
            [np.cos(laptop_yaw / 2.0), 0.0, 0.0, np.sin(laptop_yaw / 2.0)],
            dtype=float,
        ))
        lid_qpos = None
        if getattr(laptop, "n_qs", 0):
            lid_qpos = np.full(
                (int(laptop.n_qs),), float(self.NEUTRAL_TYPING_LID_QPOS), dtype=np.float64
            )
            laptop.set_qpos(lid_qpos)
            try:
                laptop.set_dofs_kp(np.full((int(laptop.n_qs),), 50.0))
                laptop.set_dofs_kv(np.full((int(laptop.n_qs),), 5.0))
                laptop.control_dofs_position(lid_qpos)
            except Exception:
                pass
        self._neutral_laptop_hold_state = (laptop, lid_qpos)

        # Mouse on the table to the avatar's right, flat (upright + yaw).
        mouse_world = body_xy + rot2 @ np.asarray(
            self.NEUTRAL_TYPING_MOUSE_LOCAL_XY, dtype=np.float64
        )
        mouse_yaw = avatar_yaw + float(np.deg2rad(self.NEUTRAL_TYPING_MOUSE_YAW_LOCAL_DEG))
        self.neutral_avatar_mouse.set_pos(np.array([
            mouse_world[0], mouse_world[1],
            table_top + float(self.NEUTRAL_TYPING_MOUSE_Z_OFFSET),
        ], dtype=float))
        self.neutral_avatar_mouse.set_quat(self._neutral_typing_mouse_quat(mouse_yaw))

        # Chair under the avatar, wheels on the floor, backrest behind her.
        chair = self.neutral_avatar_chair
        chair_scale = self._neutral_typing_chair_scale()
        chair_yaw = facing + np.pi  # asset faces -X
        chair.set_pos(np.array([
            body_xy[0], body_xy[1],
            float(self.NEUTRAL_WRITING_CHAIR_RAW_BOTTOM_Z) * chair_scale
            + float(self.NEUTRAL_CLAP_CHAIR_Z_OFFSET),
        ], dtype=float))
        chair.set_quat(np.array(
            [np.cos(chair_yaw / 2.0), 0.0, 0.0, np.sin(chair_yaw / 2.0)], dtype=float
        ))
        if getattr(chair, "n_qs", 0):
            chair_hold = np.zeros((int(chair.n_qs),), dtype=np.float64)
            chair.set_qpos(chair_hold)
            # Wheel / gas-lift joints sag over long rollouts; re-pin per step.
            # Reuse the writing chair-pin attribute so reset() restores it.
            if not hasattr(self, "_neutral_writing_orig_step_sim"):
                original_step_sim = self.step_sim
                self._neutral_writing_orig_step_sim = original_step_sim

                def chair_pinned_step_sim():
                    original_step_sim()
                    try:
                        chair.set_qpos(chair_hold)
                    except Exception:
                        pass

                self.step_sim = chair_pinned_step_sim

        # Seat the avatar and hold the first typing frame.
        self.avatar.reset(body_pos, body_rot)
        self._neutral_avatar_hold_motion_frame_idle(job["motion"], 0)
        # Nothing is held; the hands just type.
        job["events"] = []

        self._neutral_avatar_selected_work_xy = anchor_xy
        self._neutral_avatar_motion_footprint = footprint
        type(self).NEUTRAL_MOTION_FOOTPRINTS[str(job["base_motion"])] = {
            "kind": "typing",
            "min_offset": np.asarray(footprint["min_offset"], dtype=float).tolist(),
            "max_offset": np.asarray(footprint["max_offset"], dtype=float).tolist(),
            "size": np.asarray(footprint["size"], dtype=float).tolist(),
        }
        self.neutral_avatar_target = np.array(
            [anchor_xy[0], anchor_xy[1], table_top], dtype=np.float64
        )
        self._neutral_avatar_body_pos = body_pos
        self._neutral_avatar_body_rot = body_rot
        self._neutral_avatar_layout_ready = True
        self.avatar_collided = False
        self.avatar_collision_log = []
        self._avatar_collision_tick = 0
        self._neutral_debug(
            f"[neutral_avatar] typing layout side={side} "
            f"laptop={anchor_xy.round(3).tolist()} body={body_pos.round(3).tolist()}"
        )
        self._create_neutral_avatar_region_debug_overlays(job, task_regions)
        return True

    # ------------------------------------------------------------------
    # Seated-clapping neutral clip (chair only)
    # ------------------------------------------------------------------

    def _neutral_clap_dz(self) -> float:
        """Vertical shift from the clap review tabletop to this task's."""
        table_top = float(getattr(self, "TABLE_TOP_Z", 0.765))
        return table_top - float(self.NEUTRAL_CLAP_TABLE_TOP_REF)

    def _neutral_clap_chair_scale(self) -> float:
        seat = float(self.NEUTRAL_CLAP_SEAT_HEIGHT_REF) + self._neutral_clap_dz()
        return seat / float(self.NEUTRAL_WRITING_CHAIR_RAW_SEAT_H)

    def _neutral_clap_body_z_offset(self) -> float:
        return float(self.config.get(
            "neutral_clap_body_z_offset",
            self.NEUTRAL_CLAP_BODY_Z_OFFSET,
        ))

    def _neutral_clap_chair_z_offset(self) -> float:
        return float(self.config.get(
            "neutral_clap_chair_z_offset",
            self.NEUTRAL_CLAP_CHAIR_Z_OFFSET,
        ))

    def _neutral_clap_edge_anchor_local_xy(self) -> np.ndarray:
        if "neutral_clap_table_distance" in self.config:
            return np.array([
                float(self.config["neutral_clap_table_distance"]),
                0.0,
            ], dtype=np.float64)
        return np.asarray(self.NEUTRAL_CLAP_EDGE_ANCHOR_LOCAL_XY, dtype=np.float64)

    def _load_neutral_sitting_clap_props(self, job: dict) -> None:
        chair_scale = self._neutral_clap_chair_scale()
        self.neutral_avatar_props = {}
        self.neutral_avatar_chair = self._neutral_load_prop_urdf(
            self._neutral_chair_urdf(),
            Pose([6.0, 6.0, float(self.NEUTRAL_WRITING_CHAIR_RAW_BOTTOM_Z) * chair_scale]),
            chair_scale,
        )
        self.neutral_avatar_target = np.array(
            [6.0, 6.0, float(getattr(self, "TABLE_TOP_Z", 0.765))], dtype=float
        )

    def _neutral_clap_footprint(self, side: str) -> dict:
        body_rot = np.asarray(
            self._neutral_avatar_body_rot_for_side(str(side), {"kind": "sitting_clap"}),
            dtype=np.float64,
        )
        rot2 = body_rot[:2, :2]
        points = []
        mins = []
        maxs = []
        for label, local_xy, radius in self.NEUTRAL_CLAP_LOCAL_REGION_POINTS:
            offset = rot2 @ np.asarray(local_xy, dtype=np.float64).ravel()[:2]
            radius = float(radius)
            points.append({
                "offset": offset.tolist(),
                "radius": radius,
                "label": str(label),
            })
            mins.append(offset - radius)
            maxs.append(offset + radius)
        min_offset = np.min(np.asarray(mins), axis=0)
        max_offset = np.max(np.asarray(maxs), axis=0)
        return {
            "motion": "sitting_clap",
            "kind": "sitting_clap",
            "min_offset": min_offset,
            "max_offset": max_offset,
            "size": max_offset - min_offset,
            "region_points": points,
        }

    def _neutral_clap_side_candidates(self, job: dict) -> list[tuple[str, np.ndarray]]:
        xmin, xmax, ymin, ymax = self._neutral_table_xy_bounds()
        inset = float(self.NEUTRAL_CLAP_EDGE_INSET)
        edge_margin = float(self.NEUTRAL_CLAP_TABLE_EDGE_MARGIN)
        step = float(self.NEUTRAL_CLAP_CANDIDATE_STEP)
        side_regions = list(self._neutral_avatar_default_side_work_regions())
        forced_side = str(self.config.get("neutral_avatar_force_side", "") or "")
        out: list[tuple[str, np.ndarray]] = []
        for side, region in side_regions:
            side = str(side)
            if side not in self._NEUTRAL_WRITING_SIDE_INWARD:
                continue
            if forced_side and side != forced_side:
                continue
            footprint = self._neutral_clap_footprint(side)
            min_off = np.asarray(footprint["min_offset"], dtype=np.float64)
            max_off = np.asarray(footprint["max_offset"], dtype=np.float64)
            x0, x1, y0, y1 = [float(v) for v in region]
            if side == "back":
                lat_lo = max(x0, float(xmin) - float(min_off[0]) + edge_margin)
                lat_hi = min(x1, float(xmax) - float(max_off[0]) - edge_margin)
                fixed = float(ymax) - inset
                to_xy = lambda lat, fx=fixed: np.array([lat, fx], dtype=np.float64)
            else:
                lat_lo = max(y0, float(ymin) - float(min_off[1]) + edge_margin)
                lat_hi = min(y1, float(ymax) - float(max_off[1]) - edge_margin)
                fixed = (float(xmin) + inset) if side == "left" else (float(xmax) - inset)
                to_xy = lambda lat, fx=fixed: np.array([fx, lat], dtype=np.float64)
            if lat_hi < lat_lo:
                continue
            center = 0.5 * (lat_lo + lat_hi)
            lats = [center]
            k = 1
            while center + k * step <= lat_hi or center - k * step >= lat_lo:
                if center + k * step <= lat_hi:
                    lats.append(center + k * step)
                if center - k * step >= lat_lo:
                    lats.append(center - k * step)
                k += 1
            for lat in lats:
                out.append((side, to_xy(float(lat))))
        return out

    def _install_neutral_clap_subclip(self, job: dict) -> None:
        """Trim to the selected clapping period before ping-pong looping."""
        if job.get("clap_trim"):
            return
        src = str(job["base_motion"])
        md = self.avatar.motion_data[src]
        n = int(md["trans"].shape[0])
        start, end = self.NEUTRAL_CLAP_CLIP_TRIM
        start = int(np.clip(start, 0, n - 2))
        end = int(np.clip(end, start + 1, n - 1))
        idx = np.arange(start, end + 1, dtype=int)
        dst = f"{src}_clap_trim_{start}_{end}"
        self.avatar.motion_data[dst] = {k: v[idx].copy() for k, v in md.items()}
        job["motion"] = dst
        job["clap_trim"] = True
        job["clip"] = dict(job["clip"])
        job["clip"]["clip_start"] = start
        job["clip"]["clip_end"] = end
        job["clip"]["num_frames"] = int(idx.shape[0])

    def _prepare_neutral_sitting_clap_layout(self, job: dict) -> bool:
        """Fixed-relative-pose layout for the seated clapping clip."""
        self._install_neutral_clap_subclip(job)
        self.avatar.frame_ratio = self._neutral_avatar_motion_slow()
        table_top = float(getattr(self, "TABLE_TOP_Z", 0.765))
        dz = self._neutral_clap_dz()
        task_regions = self.neutral_task_regions()
        candidates = self._neutral_clap_side_candidates(job)
        if not candidates:
            raise RuntimeError(
                "no sitting-clap side candidates: table bounds/side regions "
                "leave no room for the seated clapping footprint"
            )

        min_clearance = float(self.config.get(
            "neutral_clap_min_clearance",
            self.NEUTRAL_CLAP_TASK_CLEARANCE,
        ))
        anchor_local = self._neutral_clap_edge_anchor_local_xy()
        robot_prefixes = tuple(self.NEUTRAL_WRITING_ROBOT_REGION_PREFIXES)
        debug_records = []
        scored = []
        scored_any = []
        for side, anchor_xy in candidates:
            anchor_xy = np.asarray(anchor_xy, dtype=np.float64)
            footprint = self._neutral_clap_footprint(side)
            # The clapping footprint describes the seated body/hands around
            # the table edge.  Parts of that envelope intentionally live
            # outside the tabletop, unlike laptop/book props, so do not gate
            # it through the generic on-table AABB check.
            in_table = True
            min_xy = anchor_xy + np.asarray(footprint["min_offset"], dtype=np.float64)
            max_xy = anchor_xy + np.asarray(footprint["max_offset"], dtype=np.float64)
            body_rot = self._neutral_avatar_body_rot_for_side(side, job)
            rot2 = np.asarray(body_rot, dtype=np.float64)[:2, :2]
            body_xy = anchor_xy - rot2 @ anchor_local
            body_clear = True
            worst = float("inf")
            robot_dist = float("inf")
            for center, radius, label in task_regions:
                center = np.asarray(center, dtype=np.float64).ravel()[:2]
                clamped = np.minimum(np.maximum(center, min_xy), max_xy)
                d = float(np.linalg.norm(center - clamped)) - float(radius)
                worst = min(worst, d)
                if str(label).startswith(robot_prefixes):
                    body_d = float(np.linalg.norm(center - body_xy)) - float(radius) - 0.18
                    robot_dist = min(robot_dist, d, body_d)
                    if body_d < float(self.NEUTRAL_CLAP_BODY_SWEEP_MARGIN):
                        body_clear = False
            clearance = worst if np.isfinite(worst) else 1.0
            if not np.isfinite(robot_dist):
                robot_dist = clearance
            record = {
                "xy": anchor_xy.copy(),
                "side": side,
                "footprint": footprint,
                "footprint_clear": bool(in_table and clearance > 0.0),
                "neutral_region_clear": bool(in_table and clearance > 0.0),
                "footprint_table_ok": True,
                "side_region_ok": True,
                "body_ok": bool(body_clear),
                "prop_clearance": clearance,
                "prop_table_clearance": 1.0,
                "selected": False,
            }
            debug_records.append(record)
            if in_table:
                scored_any.append((clearance, robot_dist, side, anchor_xy, footprint, record))
            if in_table and body_clear:
                scored.append((clearance, robot_dist, side, anchor_xy, footprint, record))
        self._neutral_avatar_region_debug_records = debug_records
        pool = [c for c in scored if c[0] >= min_clearance] or scored
        if not pool:
            if bool(self.config.get("neutral_avatar_strict_layout", True)) or not scored_any:
                raise RuntimeError(
                    "no sitting-clap layout candidate clears task objects and the "
                    f"seated body envelope; motion={job['base_motion']}"
                )
            pool = scored_any
            self._neutral_debug(
                "[neutral_avatar] sitting_clap: body envelope grazes every "
                "candidate; using the best in-table placement"
            )
        _clear, _robot_dist, side, anchor_xy, footprint, record = max(
            pool, key=lambda c: (c[0], c[1])
        )
        record["selected"] = True
        body_rot = self._neutral_avatar_body_rot_for_side(side, job)
        rot2 = np.asarray(body_rot, dtype=np.float64)[:2, :2]
        body_xy = anchor_xy - rot2 @ anchor_local
        body_pos = np.array([
            body_xy[0],
            body_xy[1],
            dz + self._neutral_clap_body_z_offset(),
        ], dtype=np.float64)
        forward = rot2 @ np.array([1.0, 0.0], dtype=np.float64)
        facing = float(np.arctan2(forward[1], forward[0]))

        chair = self.neutral_avatar_chair
        chair_scale = self._neutral_clap_chair_scale()
        chair_yaw = facing + np.pi
        chair.set_pos(np.array([
            body_xy[0], body_xy[1],
            float(self.NEUTRAL_WRITING_CHAIR_RAW_BOTTOM_Z) * chair_scale
            + self._neutral_clap_chair_z_offset(),
        ], dtype=float))
        chair.set_quat(np.array(
            [np.cos(chair_yaw / 2.0), 0.0, 0.0, np.sin(chair_yaw / 2.0)],
            dtype=float,
        ))
        if getattr(chair, "n_qs", 0):
            chair_hold = np.zeros((int(chair.n_qs),), dtype=np.float64)
            chair.set_qpos(chair_hold)
            if not hasattr(self, "_neutral_writing_orig_step_sim"):
                original_step_sim = self.step_sim
                self._neutral_writing_orig_step_sim = original_step_sim

                def chair_pinned_step_sim():
                    original_step_sim()
                    try:
                        chair.set_qpos(chair_hold)
                    except Exception:
                        pass

                self.step_sim = chair_pinned_step_sim

        self.avatar.reset(body_pos, np.asarray(body_rot, dtype=np.float64))
        self._neutral_avatar_hold_motion_frame_idle(job["motion"], 0)
        job["events"] = []

        self._neutral_avatar_selected_work_xy = anchor_xy
        self._neutral_avatar_motion_footprint = footprint
        type(self).NEUTRAL_MOTION_FOOTPRINTS[str(job["base_motion"])] = {
            "kind": "sitting_clap",
            "min_offset": np.asarray(footprint["min_offset"], dtype=float).tolist(),
            "max_offset": np.asarray(footprint["max_offset"], dtype=float).tolist(),
            "size": np.asarray(footprint["size"], dtype=float).tolist(),
        }
        self.neutral_avatar_target = np.array(
            [anchor_xy[0], anchor_xy[1], table_top], dtype=np.float64
        )
        self._neutral_avatar_body_pos = body_pos
        self._neutral_avatar_body_rot = np.asarray(body_rot, dtype=np.float64)
        self._neutral_avatar_layout_ready = True
        self.avatar_collided = False
        self.avatar_collision_log = []
        self._avatar_collision_tick = 0
        self._neutral_debug(
            f"[neutral_avatar] sitting_clap layout side={side} "
            f"anchor={anchor_xy.round(3).tolist()} body={body_pos.round(3).tolist()}"
        )
        self._create_neutral_avatar_region_debug_overlays(job, task_regions)
        return True

    # ------------------------------------------------------------------
    # Standing phone-talker neutral clip ("talking")
    # ------------------------------------------------------------------
    # Avatar root in the table-edge anchor's frame (avatar local +x = forward,
    # toward the table).  ~0.2 m closer to the table than the laptop typist
    # (0.79575) so the standing talker reads as standing AT the table.
    NEUTRAL_PHONE_LOCAL_XY = (0.396, -0.054)   # ~0.4 m closer than laptop typist
    NEUTRAL_PHONE_BODY_Z = 0.0          # avatar root z (feet on the floor)
    NEUTRAL_PHONE_DOWN = 0.04           # lower the phone in world -z (matches render_talking)

    def _prepare_neutral_phone_layout(self, job: dict) -> bool:
        """Fixed-pose layout for the standing 'talking on the phone' clip.

        The avatar stands beside the table (no on-table prop) holding the phone
        in the LEFT hand for the whole clip.  Reuses the laptop side/yaw
        selection to face the table, but stands ~0.2 m closer with nothing on
        the table; the phone is snapped into the palm by the event system.
        """
        self._install_neutral_loop_subclip(job)
        self.avatar.frame_ratio = self._neutral_avatar_motion_slow()
        table_top = float(getattr(self, "TABLE_TOP_Z", 0.765))

        # Pick the table-edge anchor whose on-table footprint is FARTHEST from
        # every task object/corridor (max clearance) so the standing avatar
        # ends up on the side away from the robot's work — same proxy the
        # laptop layout uses, which keeps the arm out of the avatar.
        candidates = self._neutral_laptop_side_candidates(job)
        task_regions = self.neutral_task_regions()
        best = None  # (clearance, side, xy)
        for side_i, xy_i in candidates:
            fp = self._neutral_laptop_footprint(side_i)
            min_xy = np.asarray(xy_i) + fp["min_offset"]
            max_xy = np.asarray(xy_i) + fp["max_offset"]
            worst = float("inf")
            for center, radius, _label in task_regions:
                center = np.asarray(center, dtype=np.float64).ravel()[:2]
                clamped = np.minimum(np.maximum(center, min_xy), max_xy)
                worst = min(worst, float(np.linalg.norm(center - clamped)) - float(radius))
            clearance = worst if np.isfinite(worst) else 1.0
            if best is None or clearance > best[0]:
                best = (clearance, str(side_i), np.asarray(xy_i, dtype=np.float64))
        if best is not None:
            _clear, side, anchor_xy = best
        else:
            xmin, xmax, ymin, ymax = self._neutral_table_xy_bounds()
            side = "back"
            anchor_xy = np.array(
                [0.5 * (xmin + xmax), float(ymax) - float(self.NEUTRAL_LAPTOP_EDGE_INSET)],
                dtype=np.float64,
            )
        anchor_xy = np.asarray(anchor_xy, dtype=np.float64)

        _laptop_yaw, avatar_yaw = self._neutral_laptop_world_yaws(side)
        c, s = float(np.cos(avatar_yaw)), float(np.sin(avatar_yaw))
        rot2 = np.array([[c, -s], [s, c]], dtype=np.float64)
        body_rot = np.eye(3, dtype=np.float64)
        body_rot[:2, :2] = rot2
        local = np.asarray(self.NEUTRAL_PHONE_LOCAL_XY, dtype=np.float64)
        body_xy = anchor_xy - rot2 @ local
        body_pos = np.array(
            [body_xy[0], body_xy[1], float(self.NEUTRAL_PHONE_BODY_Z)], dtype=np.float64
        )

        self.avatar.reset(body_pos, body_rot)
        self._neutral_avatar_hold_motion_frame_idle(job["motion"], 0)

        half = 0.22
        footprint = {
            "motion": str(job["base_motion"]),
            "kind": "phone",
            "min_offset": np.array([-half, -half], dtype=np.float64),
            "max_offset": np.array([half, half], dtype=np.float64),
            "size": np.array([2 * half, 2 * half], dtype=np.float64),
            "region_points": [],
        }
        self._neutral_avatar_selected_work_xy = anchor_xy
        self._neutral_avatar_motion_footprint = footprint
        type(self).NEUTRAL_MOTION_FOOTPRINTS[str(job["base_motion"])] = {
            "kind": "phone",
            "min_offset": footprint["min_offset"].tolist(),
            "max_offset": footprint["max_offset"].tolist(),
            "size": footprint["size"].tolist(),
        }
        self.neutral_avatar_target = np.array(
            [anchor_xy[0], anchor_xy[1], table_top], dtype=np.float64
        )
        self._neutral_avatar_body_pos = body_pos
        self._neutral_avatar_body_rot = body_rot
        self._neutral_avatar_layout_ready = True
        self.avatar_collided = False
        self.avatar_collision_log = []
        self._avatar_collision_tick = 0
        self._neutral_debug(
            f"[neutral_avatar] phone layout side={side} "
            f"anchor={anchor_xy.round(3).tolist()} body={body_pos.round(3).tolist()}"
        )
        self._create_neutral_avatar_region_debug_overlays(job, self.neutral_task_regions())
        return True

    def _neutral_laptop_hold_lid(self) -> None:
        state = getattr(self, "_neutral_laptop_hold_state", None)
        if not state:
            return
        laptop, lid_qpos = state
        if lid_qpos is None:
            return
        try:
            laptop.set_qpos(lid_qpos)
        except Exception:
            pass

    def _load_neutral_avatar_region_debug_tiles(self) -> None:
        if not bool(self.config.get("neutral_avatar_debug_regions")):
            self._neutral_avatar_region_debug_tiles = {}
            return
        park = np.array([8.0, 8.0, self.TABLE_TOP_Z + 0.20], dtype=float)
        half = float(self.config.get("neutral_avatar_debug_region_tile_half", 0.025))
        count = int(self.config.get("neutral_avatar_debug_region_tile_count", 260))
        count = max(1, count)
        colors = {
            "table": (0.10, 0.25, 0.95, 0.30),
            "forbidden": (0.90, 0.10, 0.10, 0.35),
            "blocked": (1.0, 0.45, 0.05, 0.35),
            "clear": (0.05, 0.75, 0.20, 0.30),
            # Selected neutral motion swept region.
            "selected": (1.0, 0.92, 0.05, 0.45),
            # Reset-time neutral object footprint; separate from motion sweep.
            "object": (0.05, 0.85, 1.0, 0.45),
        }
        pools = {}
        for name, color in colors.items():
            pools[name] = [
                create_primitive(
                    self.scene,
                    "box",
                    Pose(park + np.array([0.0, 0.0, 0.002 * i])),
                    size={"half_size": (half, half, 0.002)},
                    color=color,
                    is_static=True,
                    collision=False,
                )
                for i in range(count)
            ]
        self._neutral_avatar_region_debug_tiles = pools

    def _create_neutral_avatar_prop(self, obj_name: str, pos: np.ndarray):
        z = float(pos[2])
        if obj_name == "sponge":
            return load_mesh(
                self.scene,
                ASSETS_PATH / "objects" / "cc0_sponge_3" / "visual" / "base0.glb",
                Pose(
                    [float(pos[0]), float(pos[1]), z],
                    self._neutral_avatar_prop_quat(obj_name),
                ),
                scale=1.0,
                is_static=False,
                collision=True,
                friction=3.0,
                density=80.0,
            )
        if obj_name == "mouse":
            actor = load_object(
                self.scene,
                Pose(
                    [float(pos[0]), float(pos[1]), z],
                    self._neutral_avatar_prop_quat(obj_name),
                ),
                "047_mouse",
                model_id=0,
                convex=True,
                is_static=False,
                friction=3.0,
                density=250.0,
            )
            return actor.entity
        if obj_name == "bottle" or str(obj_name).startswith("bottle_"):
            actor = load_object(
                self.scene,
                Pose(
                    [float(pos[0]), float(pos[1]), z],
                    self._neutral_avatar_prop_quat(obj_name),
                ),
                "001_bottle",
                model_id=13,
                convex=True,
                is_static=False,
                friction=3.0,
                density=250.0,
            )
            return actor.entity
        if obj_name == "phone":
            # Held-in-hand phone for the standing "talking" neutral motion;
            # snapped into the palm by the event system, never on the table.
            actor = load_object(
                self.scene,
                Pose(
                    [float(pos[0]), float(pos[1]), z],
                    self._neutral_avatar_prop_quat(obj_name),
                ),
                "077_phone",
                model_id=0,
                convex=True,
                is_static=False,
                friction=3.0,
                density=250.0,
            )
            return actor.entity
        actor = load_object(
            self.scene,
            Pose(
                [float(pos[0]), float(pos[1]), self._neutral_avatar_prop_center_z(obj_name)],
                self._neutral_avatar_prop_quat(obj_name),
            ),
            self._neutral_put_objects_asset(),
            model_id=self._neutral_avatar_can_model_id(obj_name),
            convex=False,
            is_static=False,
            friction=4.0,
            density=250.0,
        )
        return actor.entity

    def _neutral_put_objects_clip_defaults(self) -> dict:
        """Per-clip prop overrides baked into the motion (e.g. apples for the
        take-objects clip). Resolution order is config > clip > class default."""
        job = self._neutral_avatar_job or {}
        return dict(((job.get("clip") or {}).get("prop_defaults") or {}))

    def _neutral_put_objects_asset(self) -> str:
        defaults = self._neutral_put_objects_clip_defaults()
        return str(self.config.get(
            "neutral_put_objects_object",
            defaults.get("object", self.NEUTRAL_PUT_OBJECTS_OBJECT),
        ))

    def _neutral_avatar_can_model_id(self, obj_name: str) -> int:
        defaults = self._neutral_put_objects_clip_defaults()
        ids = list(self.config.get(
            "neutral_avatar_can_model_ids",
            defaults.get("model_ids", self.NEUTRAL_PUT_OBJECTS_CAN_MODEL_IDS),
        ))
        ids = [int(v) for v in ids]
        if not ids:
            ids = list(self.NEUTRAL_PUT_OBJECTS_CAN_MODEL_IDS)
        cache = getattr(self, "_neutral_avatar_can_model_ids", None)
        if cache is None:
            cache = {}
            self._neutral_avatar_can_model_ids = cache
        key = str(obj_name)
        if key not in cache:
            seed = int(getattr(self, "_neutral_avatar_reset_seed", 0))
            obj_offset = 1 if key == "object_1" else 2
            rng = np.random.default_rng(seed + 80409 + obj_offset * 1009)
            cache[key] = ids[int(rng.integers(0, len(ids)))]
        return int(cache[key])

    def _neutral_avatar_can_body_radius(self, obj_name: str) -> float:
        model_id = self._neutral_avatar_can_model_id(obj_name)
        md_path = ASSETS_PATH / "objects" / self._neutral_put_objects_asset() / f"model_data{model_id}.json"
        try:
            with open(md_path) as f:
                md = json.load(f)
            extents = np.asarray(md.get("extents", [1.6, 2.0, 1.6]), dtype=np.float64)
            scale = np.asarray(md.get("scale", [0.05, 0.05, 0.05]), dtype=np.float64)
            if scale.size == 1:
                scale = np.repeat(float(scale[0]), 3)
            return float(max(extents[0] * scale[0], extents[2] * scale[2]) * 0.5)
        except Exception:
            return float(self._neutral_prop_radius(obj_name))

    def _neutral_avatar_prebuild_prop_xy(
        self, job: dict, work_xy: np.ndarray, obj_name: str, obj_idx: int
    ) -> np.ndarray:
        """Keep reset-time neutral props stable before layout selection.

        Neutral props are created in ``load_actors()``, before BaseTask's
        initial settle and before ``_prepare_neutral_avatar_start_layout()``
        selects the final work anchor.  Clamp these temporary positions to the
        physical table footprint so bottle/sponge clips cannot start with a
        prop hanging over an edge and falling before the neutral layout pass.
        """
        xy = np.asarray(work_xy, dtype=np.float64).ravel()[:2].copy()
        prop_margin = self._neutral_avatar_table_margin_for_prop(job, obj_name)
        if job.get("kind") != "put_objects_in_bowl":
            return self._neutral_avatar_clamp_table_xy(xy, margin=prop_margin)
        offsets = (
            np.array([-0.075, 0.000], dtype=np.float64),
            np.array([0.075, 0.000], dtype=np.float64),
            np.array([0.000, 0.095], dtype=np.float64),
            np.array([0.000, -0.095], dtype=np.float64),
        )
        xy = xy + offsets[int(obj_idx) % len(offsets)]
        return self._neutral_avatar_clamp_table_xy(
            xy,
            margin=max(float(self.NEUTRAL_PUT_OBJECTS_TABLE_MARGIN), prop_margin),
        )

    @staticmethod
    def _neutral_object_identity_tokens(value) -> set[str]:
        """Return comparable identity tokens for a target or neutral prop.

        Catalog entries may include a model suffix (``035_apple@1``), while
        motion metadata usually uses a short noun (``apple``).  Keeping both
        the full asset ID and its numeric-prefix-free noun makes those forms
        compare equal without conflating distinct objects such as an oil
        bottle and the generic water bottle.
        """
        if value is None:
            return set()
        raw = str(value).strip().lower().replace("-", "_").replace(" ", "_")
        raw = raw.split("@", 1)[0]
        if not raw:
            return set()
        tokens = {raw}
        prefix, sep, tail = raw.partition("_")
        if sep and prefix.isdigit() and tail:
            tokens.add(tail)
        return tokens

    def _neutral_clip_object_ids(self, clip: dict) -> set[str]:
        """Actual prop identities used by one neutral-motion clip."""
        handles = {
            str(event.get("object"))
            for event in (clip.get("events") or [])
            if event.get("object") is not None
        }
        handles.update(str(obj) for obj in (clip.get("objects") or []))

        kind = str(clip.get("kind") or "")
        if not kind and any(
            str(event.get("type")) == "pick"
            for event in (clip.get("events") or [])
        ):
            kind = "put_objects_in_bowl"
        if kind == "writing":
            handles.update(("pen", "notebook", "chair"))
        elif kind == "laptop":
            handles.add("laptop")
        elif kind == "typing":
            handles.update(("laptop", "mouse", "chair"))
        elif kind == "sitting_clap":
            handles.add("chair")

        prop_defaults = dict(clip.get("prop_defaults") or {})
        if kind == "put_objects_in_bowl" and bool(self.config.get(
            "neutral_put_objects_plate",
            prop_defaults.get("use_plate", True),
        )):
            handles.add("plate")
        put_objects_asset = str(self.config.get(
            "neutral_put_objects_object",
            prop_defaults.get("object", self.NEUTRAL_PUT_OBJECTS_OBJECT),
        ))
        identities: set[str] = set()
        for handle in handles:
            if handle.startswith("object_"):
                asset_id = put_objects_asset
            elif handle.startswith("bottle_"):
                asset_id = self.NEUTRAL_PROP_ASSET_IDS["bottle"]
            else:
                asset_id = self.NEUTRAL_PROP_ASSET_IDS.get(handle, handle)
            identities.update(self._neutral_object_identity_tokens(asset_id))
        return identities

    def _neutral_robot_target_object_ids(self) -> set[str]:
        """Collect identities of the robot targets spawned for this episode.

        Neutral task families expose targets in a few established shapes:
        catalog ``_target_entry`` for single-object tasks, sampled dump specs,
        categorize ``_sort_targets``, and food ``(actor, FoodSpec)`` pairs.
        This is intentionally episode-specific, so a pooled task only loses a
        motion when its sampled target actually conflicts.
        """
        identities: set[str] = set()

        def add(value) -> None:
            identities.update(self._neutral_object_identity_tokens(value))

        entry = getattr(self, "_target_entry", None)
        if entry is not None:
            add(getattr(entry, "key", None))
            add(getattr(entry, "object_id", None))

        for spec in (getattr(self, "_episode_dump_specs", None) or []):
            if isinstance(spec, dict):
                add(spec.get("object_id") or spec.get("key"))

        for target in (getattr(self, "_sort_targets", None) or []):
            if isinstance(target, (tuple, list)) and len(target) >= 2:
                add(target[1])

        for item in (getattr(self, "food_actors", None) or []):
            if not isinstance(item, (tuple, list)) or len(item) < 2:
                continue
            spec = item[1]
            add(getattr(spec, "asset_id", None))
            add(getattr(spec, "object_id", None))

        return identities

    def _ensure_neutral_avatar_job_is_target_distinct(self) -> None:
        """Enforce that the avatar's props differ from robot target objects."""
        job = self._neutral_avatar_job or {}
        targets = self._neutral_robot_target_object_ids()
        conflicts = targets & self._neutral_clip_object_ids(job.get("clip") or {})
        if not conflicts:
            return
        motion = str(job.get("base_motion") or job.get("motion") or "")
        if self._neutral_avatar_forced_motion_requested():
            raise ValueError(
                f"neutral_avatar_motion={motion!r} duplicates robot target "
                f"object(s) {sorted(conflicts)}; choose a distinct neutral motion"
            )
        self._resolve_neutral_avatar_job(excluded_target_ids=targets)

    def _resolve_neutral_avatar_job(self, excluded_target_ids: set[str] | None = None):
        meta_path = _resolve_repo_path(self.config.get("neutral_avatar_motion_meta", _NEUTRAL_MOTION_META))
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        clips = meta["clips"]
        forced = self.config.get("neutral_avatar_motion")
        keys = sorted(clips)
        if forced:
            forced = str(forced)
            forced = str(self.NEUTRAL_AVATAR_MOTION_ALIASES.get(forced, forced))
            matches = [k for k in keys if k == forced or k.startswith(f"{forced}_")]
            if not matches:
                raise ValueError(
                    "neutral_avatar_motion must be a clip name/prefix or one "
                    f"of aliases {sorted(self.NEUTRAL_AVATAR_MOTION_ALIASES)}; "
                    f"got {forced!r}. Available clips: {keys}"
                )
            motion_name = matches[0]
        else:
            choices = list(self.config.get(
                "neutral_avatar_motion_choices",
                self.NEUTRAL_AVATAR_MOTION_CHOICES,
            ))
            choices = [name for name in choices if name in clips]
            if not choices:
                choices = keys
            if excluded_target_ids:
                choices = [
                    name for name in choices
                    if not (
                        set(excluded_target_ids)
                        & self._neutral_clip_object_ids(clips[name])
                    )
                ]
                if not choices:
                    raise ValueError(
                        "no neutral avatar motion remains after excluding robot "
                        f"target object(s) {sorted(excluded_target_ids)}"
                    )
                # A private generator avoids perturbing scene/layout RNG after
                # the parent task has already spawned its episode targets.
                seed = int(getattr(self, "_neutral_avatar_reset_seed", 0)) + 904_219
                rng = np.random.default_rng(seed)
                motion_name = choices[int(rng.integers(len(choices)))]
            else:
                motion_name = choices[int(np.random.randint(len(choices)))]
        clip = clips[motion_name]
        placement = dict(clip.get("neutral_placement", {}) or {})
        events = [dict(e) for e in clip["events"]]
        objects = []
        for event in events:
            obj = str(event["object"])
            if obj not in objects:
                objects.append(obj)
        if not objects:
            objects = [str(o) for o in (clip.get("objects") or [])]
        if clip.get("kind") is not None:
            kind = str(clip["kind"])
        else:
            kind = "put_objects_in_bowl" if any(e["type"] == "pick" for e in events) else objects[0]
        self._neutral_avatar_job = {
            "motion": motion_name,
            "base_motion": motion_name,
            "clip": clip,
            "events": events,
            "objects": objects,
            "kind": kind,
            "placement": placement,
        }

    def _neutral_avatar_task_placement(self, job: dict | None = None) -> dict:
        job = self._neutral_avatar_job if job is None else job
        if not job:
            return {}
        return dict((job.get("placement") or {}).get("task", {}) or {})

    def _neutral_avatar_raw_preview_adjustment(self, job: dict | None = None) -> dict:
        job = self._neutral_avatar_job if job is None else job
        if not job:
            return {}
        return dict((job.get("placement") or {}).get("raw_preview_adjustment", {}) or {})

    def _install_neutral_avatar_subclip(self, job: dict) -> None:
        if job.get("trim_start") is not None:
            return
        start, end = self.NEUTRAL_MOUSE_CLIP_TRIM
        src_name = str(job["base_motion"])
        md = self.avatar.motion_data[src_name]
        n = int(md["trans"].shape[0])
        start = int(np.clip(start, 0, n - 1))
        end = int(np.clip(end, start + 1, n))
        attach_frame = 63
        detach_frame = None
        for event in job["events"]:
            if str(event.get("object")) == "mouse" and event.get("type") == "detach":
                detach_frame = int(event["clip_frame"]) - start
                break
        n_trim = end - start
        if detach_frame is None:
            detach_frame = n_trim - 1
        attach_frame = int(np.clip(attach_frame, 0, n_trim - 2))
        detach_frame = int(np.clip(detach_frame, attach_frame + 1, n_trim - 1))

        forward_loops = int(self.config.get(
            "neutral_avatar_mouse_attached_forward_loops",
            self.NEUTRAL_MOUSE_ATTACHED_FORWARD_LOOPS,
        ))
        forward_loops = int(np.clip(forward_loops, 1, 5))
        dst_name = (
            f"{src_name}_neutral_trim_{int(start)}_{int(end)}"
            f"_mouse_pingpong_{forward_loops}"
        )

        prefix = list(range(start, start + attach_frame + 1))
        forward = list(range(start + attach_frame + 1, start + detach_frame + 1))
        backward = list(range(start + detach_frame - 1, start + attach_frame - 1, -1))
        attached = []
        for loop_idx in range(forward_loops):
            if loop_idx > 0:
                attached.extend(backward)
            attached.extend(forward)
        suffix = list(range(start + detach_frame + 1, end))
        indices = np.asarray(prefix + attached + suffix, dtype=int)

        self.avatar.motion_data[dst_name] = {k: v[indices].copy() for k, v in md.items()}
        job["motion"] = dst_name
        job["trim_start"] = start
        job["mouse_attached_forward_loops"] = forward_loops
        attach_len = detach_frame - attach_frame
        expanded_detach_frame = attach_frame + attach_len * forward_loops + attach_len * (forward_loops - 1)
        for event in job["events"]:
            event["clip_frame"] = int(event["clip_frame"]) - start
            if str(event.get("object")) == "mouse" and event.get("type") == "attach":
                event["clip_frame"] = attach_frame
            elif str(event.get("object")) == "mouse" and event.get("type") == "detach":
                event["clip_frame"] = expanded_detach_frame
            elif int(event["clip_frame"]) > detach_frame:
                event["clip_frame"] = int(event["clip_frame"]) + (expanded_detach_frame - detach_frame)
        job["events"] = [
            e for e in job["events"]
            if 0 <= int(e["clip_frame"]) < int(indices.shape[0])
        ]
        job["clip"] = dict(job["clip"])
        job["clip"]["num_frames"] = int(indices.shape[0])

    def _install_neutral_avatar_sponge_pingpong_subclip(self, job: dict) -> None:
        if job.get("sponge_pingpong") is not None:
            return
        src_name = str(job["base_motion"])
        md = self.avatar.motion_data[src_name]
        n = int(md["trans"].shape[0])
        attach_frame = None
        detach_frame = None
        for event in job["events"]:
            if str(event.get("object")) != "sponge":
                continue
            if event.get("type") == "attach":
                attach_frame = int(event["clip_frame"])
            elif event.get("type") == "detach":
                detach_frame = int(event["clip_frame"])
                break
        if attach_frame is None or detach_frame is None:
            return
        attach_frame = int(np.clip(attach_frame, 0, n - 2))
        detach_frame = int(np.clip(detach_frame, attach_frame + 1, n - 1))
        task_placement = self._neutral_avatar_task_placement(job)
        forward_loops = int(self.config.get(
            "neutral_avatar_sponge_attached_forward_loops",
            task_placement.get("sponge_attached_forward_loops", 3),
        ))
        forward_loops = int(np.clip(forward_loops, 1, 5))
        dst_name = f"{src_name}_sponge_pingpong_{forward_loops}"

        prefix = list(range(0, attach_frame + 1))
        forward = list(range(attach_frame + 1, detach_frame + 1))
        backward = list(range(detach_frame - 1, attach_frame - 1, -1))
        attached = []
        for loop_idx in range(forward_loops):
            if loop_idx > 0:
                attached.extend(backward)
            attached.extend(forward)
        suffix = list(range(detach_frame + 1, n))
        indices = np.asarray(prefix + attached + suffix, dtype=int)

        self.avatar.motion_data[dst_name] = {k: v[indices].copy() for k, v in md.items()}
        job["motion"] = dst_name
        job["sponge_pingpong"] = forward_loops
        attach_len = detach_frame - attach_frame
        expanded_detach_frame = attach_frame + attach_len * forward_loops + attach_len * (forward_loops - 1)
        for event in job["events"]:
            if str(event.get("object")) == "sponge" and event.get("type") == "detach":
                event["clip_frame"] = expanded_detach_frame
            elif int(event["clip_frame"]) > detach_frame:
                event["clip_frame"] = int(event["clip_frame"]) + (expanded_detach_frame - detach_frame)
        job["clip"] = dict(job["clip"])
        job["clip"]["num_frames"] = int(indices.shape[0])

    def _install_neutral_reverse_subclip(self, job: dict) -> None:
        """Time-reverse the put-objects clip into a take-objects-out clip.

        The ``take_objects_from_bowl`` motion reuses the put-objects-in-bowl
        machinery but plays the source frames backward, so the avatar lifts
        the two cans out of the plate and sets them down on the table.  The
        clip's events are authored directly in reversed-frame indexing with
        swapped attach/detach (``pick`` at the plate, ``put`` on the table),
        so this method only builds the reversed ``motion_data`` entry and
        repoints the job at it.  No-op unless the resolved clip declares
        ``reverse: true``.
        """
        clip = job.get("clip", {}) or {}
        if not bool(clip.get("reverse")) or job.get("reversed_installed"):
            return
        src_key = str(clip.get("source_motion") or job["base_motion"])
        md = self.avatar.motion_data.get(src_key)
        if md is None:
            raise RuntimeError(
                f"reverse neutral motion: source clip {src_key!r} is missing "
                "from avatar.motion_data"
            )
        n = int(md["trans"].shape[0])
        idx = np.arange(n - 1, -1, -1, dtype=int)
        dst = f"{src_key}_neutral_reverse"
        self.avatar.motion_data[dst] = {k: v[idx].copy() for k, v in md.items()}
        job["motion"] = dst
        job["reversed_installed"] = True
        job["clip"] = dict(clip)
        job["clip"]["num_frames"] = int(n)

    def _neutral_bowl_event_type(self, job: dict) -> str:
        """Event type whose hand position anchors the plate/bowl.

        Forward put-objects releases the cans into the plate (``put``); the
        reversed take-objects clip lifts them out (``pick``).  The plate sits
        at that event's hold position in either direction.
        """
        clip = (job or {}).get("clip", {}) or {}
        return "pick" if bool(clip.get("reverse")) else "put"

    def _neutral_avatar_default_work_xy(self, job: dict) -> np.ndarray:
        if job["kind"] == "sponge" and self.NEUTRAL_SPONGE_WORK_XY is not None:
            return np.asarray(self.NEUTRAL_SPONGE_WORK_XY, dtype=np.float64)
        if job["kind"] == "mouse":
            return np.asarray(self.NEUTRAL_MOUSE_WORK_XY, dtype=np.float64)
        xy = np.asarray(self.NEUTRAL_WORK_XY, dtype=np.float64)
        if job["kind"] == "put_objects_in_bowl":
            xy = xy + np.asarray(self.NEUTRAL_PUT_OBJECTS_IMAGE_SHIFT_XY, dtype=np.float64)
        return xy

    def _neutral_avatar_work_candidate_shift_xy(self, job: dict) -> np.ndarray:
        if job["kind"] == "put_objects_in_bowl":
            return np.asarray(self.NEUTRAL_PUT_OBJECTS_IMAGE_SHIFT_XY, dtype=np.float64)
        return np.zeros(2, dtype=np.float64)

    def _neutral_avatar_work_xy(self, job: dict) -> np.ndarray:
        if self._neutral_avatar_selected_work_xy is not None:
            return np.asarray(self._neutral_avatar_selected_work_xy, dtype=np.float64)
        return self._neutral_avatar_default_work_xy(job)

    def _neutral_zero_entity_velocity(self, entity) -> None:
        try:
            entity.set_dofs_velocity(np.zeros(6, dtype=float))
        except Exception:
            pass

    def _neutral_debug(self, message: str) -> None:
        return None

    def _neutral_avatar_body_offset(self, job: dict) -> np.ndarray:
        offset = np.zeros(3, dtype=np.float64)
        offset[2] += float(self.NEUTRAL_AVATAR_BODY_Z_OFFSET)
        # Extra body raise: config > clip default > 0 (no effect elsewhere).
        offset[2] += float(self.config.get(
            "neutral_avatar_body_z_offset",
            self._neutral_put_objects_clip_defaults().get("avatar_body_z_offset", 0.0),
        ))
        task_placement = self._neutral_avatar_task_placement(job)
        offset[2] += float(task_placement.get("avatar_body_z_offset", 0.0))
        if job.get("kind") == "bottle":
            offset += np.asarray(
                task_placement.get(
                    "bottle_body_extra_offset",
                    self.NEUTRAL_AVATAR_BOTTLE_EXTRA_BODY_OFFSET,
                ),
                dtype=np.float64,
            )
            offset[2] += float(task_placement.get(
                "bottle_body_z_offset",
                self.NEUTRAL_AVATAR_BOTTLE_BODY_Z_OFFSET,
            ))
        if job.get("kind") == "mouse":
            offset[2] += float(task_placement.get(
                "mouse_body_z_offset",
                self.NEUTRAL_AVATAR_MOUSE_BODY_Z_OFFSET,
            ))
        if job["kind"] == "put_objects_in_bowl":
            offset[2] += float(self.NEUTRAL_PUT_OBJECTS_AVATAR_TABLE_Z_OFFSET)
        return offset

    def _prepare_neutral_avatar_start_layout(self) -> bool:
        if self.avatar is None or getattr(self, "_neutral_avatar_layout_ready", False):
            return False
        job = self._neutral_avatar_job
        if not job:
            return False

        if job["kind"] == "writing":
            return self._prepare_neutral_writing_layout(job)

        if job["kind"] == "laptop":
            return self._prepare_neutral_laptop_layout(job)

        if job["kind"] == "phone":
            return self._prepare_neutral_phone_layout(job)

        if job["kind"] == "typing":
            return self._prepare_neutral_typing_layout(job)

        if job["kind"] == "sitting_clap":
            return self._prepare_neutral_sitting_clap_layout(job)

        if job["kind"] == "mouse":
            self._install_neutral_avatar_subclip(job)
        if job["kind"] == "sponge":
            self._install_neutral_avatar_sponge_pingpong_subclip(job)
        if job["kind"] == "put_objects_in_bowl":
            self._install_neutral_reverse_subclip(job)

        self.avatar.frame_ratio = self._neutral_avatar_motion_slow()
        if (
            job["kind"] == "put_objects_in_bowl"
            and self._neutral_avatar_motion_footprint is None
        ):
            prelim_xy = self._neutral_avatar_default_work_xy(job)
            prelim_rot = self._neutral_avatar_body_rot_for_work_xy(job, prelim_xy)
            self._neutral_avatar_measure_body_rot = prelim_rot
            _prelim_hold = self._neutral_avatar_measure_hold_points(job)
            self._neutral_avatar_motion_footprint = self._neutral_avatar_compute_motion_footprint(job)
            self._neutral_avatar_measure_body_rot = None
        candidates = self._neutral_avatar_work_candidates(job)
        task_regions = self.neutral_task_regions()
        forbidden = task_regions
        prop_forbidden = self._neutral_avatar_prop_forbidden_regions(task_regions)
        selected = None
        debug_records = []
        candidate_states = []
        pose_eval_cache: dict[tuple[float, ...], tuple[dict[int, np.ndarray], dict]] = {}
        first_event = next(e for e in job["events"] if e["type"] in ("attach", "pick"))
        for xy in candidates:
            body_rot_i = self._neutral_avatar_body_rot_for_work_xy(job, xy)
            rot_key = tuple(np.round(np.asarray(body_rot_i, dtype=np.float64).ravel(), 6))
            cached_eval = pose_eval_cache.get(rot_key)
            if cached_eval is None:
                self._neutral_avatar_measure_body_rot = body_rot_i
                hold_i = self._neutral_avatar_measure_hold_points(job)
                footprint_i = self._neutral_avatar_compute_motion_footprint(job)
                pose_eval_cache[rot_key] = (hold_i, footprint_i)
            else:
                hold_i, footprint_i = cached_eval
            body_pos_i = self._neutral_avatar_body_pos_for_work_xy(
                job, xy, hold_i,
            )
            footprint_clear = self._neutral_footprint_is_clear(xy, footprint_i, forbidden)
            neutral_region_clear = self._neutral_motion_region_is_clear(xy, footprint_i, task_regions)
            body_ok = self._neutral_avatar_body_pose_is_valid(job, xy, body_pos_i)
            side_fn = getattr(self, "_neutral_sponge_side_for_xy", None)
            side = side_fn(xy) if side_fn is not None else self._neutral_avatar_side_for_work_xy(job, xy)
            side_region_ok = self._neutral_footprint_inside_side_region(xy, footprint_i, side)
            footprint_table_ok = self._neutral_footprint_inside_table(xy, footprint_i)
            prop_clearance = self._neutral_initial_props_clearance(
                job, xy, body_pos_i, hold_i, prop_forbidden
            )
            prop_table_clearance = self._neutral_initial_props_table_clearance(
                job, xy, body_pos_i, hold_i
            )
            candidate_state = {
                "xy": np.asarray(xy, dtype=np.float64).copy(),
                "side": str(side),
                "footprint": footprint_i,
                "footprint_clear": bool(footprint_clear),
                "neutral_region_clear": bool(neutral_region_clear),
                "footprint_table_ok": bool(footprint_table_ok),
                "side_region_ok": bool(side_region_ok),
                "body_ok": bool(body_ok),
                "prop_clearance": float(prop_clearance),
                "prop_table_clearance": float(prop_table_clearance),
                "selected": False,
                "selected_tuple": (xy, body_rot_i, hold_i, footprint_i, body_pos_i),
            }
            candidate_states.append(candidate_state)
            debug_records.append(candidate_state)
            use_region_placement = bool(self.config.get(
                "neutral_use_region_placement",
                self.NEUTRAL_USE_REGION_PLACEMENT,
            ))
            placement_clear = neutral_region_clear if use_region_placement else footprint_clear
            placement_side_ok = side_region_ok
            props_ok = prop_clearance >= 0.0 and prop_table_clearance >= 0.0
            if placement_clear and placement_side_ok and footprint_table_ok and body_ok and props_ok:
                debug_records[-1]["selected"] = True
                selected = (xy, body_rot_i, hold_i, footprint_i, body_pos_i)
                side_text = f" side={side}" if side else ""
                self._neutral_debug(
                    f"[neutral_avatar] selected work_xy={xy.round(3).tolist()} "
                    f"for {job['base_motion']}{side_text}"
                )
                break
        if selected is None:
            if not candidate_states and job.get("kind") in ("sponge", "mouse", "bottle"):
                for xy in self._neutral_avatar_work_candidates(job):
                    body_rot_i = self._neutral_avatar_body_rot_for_work_xy(job, xy)
                    self._neutral_avatar_measure_body_rot = body_rot_i
                    hold_i = self._neutral_avatar_measure_hold_points(job)
                    footprint_i = self._neutral_avatar_compute_motion_footprint(job)
                    body_pos_i = self._neutral_avatar_body_pos_for_work_xy(job, xy, hold_i)
                    side = self._neutral_avatar_side_for_work_xy(job, xy)
                    candidate_state = {
                        "xy": np.asarray(xy, dtype=np.float64).copy(),
                        "side": str(side),
                        "footprint": footprint_i,
                        "footprint_clear": bool(self._neutral_footprint_is_clear(xy, footprint_i, forbidden)),
                        "neutral_region_clear": bool(self._neutral_motion_region_is_clear(xy, footprint_i, task_regions)),
                        "footprint_table_ok": bool(self._neutral_footprint_inside_table(xy, footprint_i)),
                        "side_region_ok": bool(self._neutral_footprint_inside_side_region(xy, footprint_i, side)),
                        "body_ok": bool(self._neutral_avatar_body_pose_is_valid(job, xy, body_pos_i)),
                        "prop_clearance": float(self._neutral_initial_props_clearance(
                            job, xy, body_pos_i, hold_i, prop_forbidden
                        )),
                        "prop_table_clearance": float(self._neutral_initial_props_table_clearance(
                            job, xy, body_pos_i, hold_i
                        )),
                        "selected": False,
                        "selected_tuple": (xy, body_rot_i, hold_i, footprint_i, body_pos_i),
                    }
                    candidate_states.append(candidate_state)
                    debug_records.append(candidate_state)
                self._neutral_avatar_measure_body_rot = None
            self._neutral_avatar_region_debug_records = debug_records
            safe_prop = [
                rec for rec in candidate_states
                if (
                    float(rec.get("prop_clearance", -1e9)) >= 0.0
                    and float(rec.get("prop_table_clearance", -1e9)) >= 0.0
                    and bool(rec.get("footprint_table_ok"))
                    and bool(rec.get("side_region_ok"))
                    and bool(rec.get("body_ok"))
                    and (
                        job.get("kind") not in ("sponge", "mouse", "bottle")
                        or bool(rec.get("neutral_region_clear"))
                    )
                )
            ]
            strict_layout = bool(self.config.get("neutral_avatar_strict_layout", True))
            if (
                strict_layout
                and job.get("kind") == "put_objects_in_bowl"
                and not safe_prop
            ):
                raise RuntimeError(
                    "no neutral can layout candidate clears task objects, "
                    "table bounds, and the robot gripper region"
                )
            if strict_layout and job.get("kind") in ("sponge", "mouse", "bottle") and not safe_prop:
                raise RuntimeError(
                    "no neutral layout candidate clears task objects, table bounds, "
                    f"and avatar body edge gap; motion={job['base_motion']} "
                    f"kind={job['kind']}"
                )
            if safe_prop:
                by_side: dict[str, list[dict]] = {}
                for rec in safe_prop:
                    by_side.setdefault(str(rec.get("side", "")), []).append(rec)
                sides = sorted(by_side)
                seed = int(getattr(self, "_neutral_avatar_reset_seed", 0))
                side = sides[seed % len(sides)]
                group = by_side[side]
                best = group[(seed // max(1, len(sides))) % len(group)]
            else:
                best = max(
                    candidate_states,
                    key=lambda rec: (
                        float(rec.get("prop_clearance", -1e9)) >= 0.0,
                        bool(rec.get("side_region_ok")),
                        bool(rec.get("body_ok")),
                        bool(rec.get("footprint_table_ok")),
                        bool(rec.get("footprint_clear")),
                        float(rec.get("prop_clearance", -1e9)),
                        float(rec.get("prop_table_clearance", -1e9)),
                    ),
                )
            best["selected"] = True
            selected = best["selected_tuple"]
            xy = selected[0]
            self._neutral_debug(
                f"[neutral_avatar] WARNING: no clear neutral candidate for "
                f"{job['base_motion']}; using {xy.round(3).tolist()} "
                f"(prop_clearance={float(best.get('prop_clearance', 0.0)):.3f})"
            )
        self._neutral_avatar_measure_body_rot = None
        self._neutral_avatar_region_debug_records = debug_records
        if job.get("kind") == "put_objects_in_bowl":
            xy, body_rot, hold_by_frame, footprint, body_pos = selected
            if not self._neutral_put_objects_footprint_is_table_safe(xy, footprint):
                xmin, xmax, ymin, ymax = self._neutral_table_xy_bounds()
                margin = float(
                    getattr(
                        self,
                        "NEUTRAL_PUT_OBJECTS_SWEEP_TABLE_MARGIN",
                        self.NEUTRAL_PUT_OBJECTS_PLATE_EDGE_MARGIN,
                    )
                )
                min_offset = np.asarray(footprint["min_offset"], dtype=np.float64).ravel()[:2]
                max_offset = np.asarray(footprint["max_offset"], dtype=np.float64).ravel()[:2]
                lo = np.array([float(xmin) + margin, float(ymin) + margin], dtype=np.float64) - min_offset
                hi = np.array([float(xmax) - margin, float(ymax) - margin], dtype=np.float64) - max_offset
                if np.all(lo <= hi):
                    xy = np.clip(np.asarray(xy, dtype=np.float64).ravel()[:2], lo, hi)
                    body_pos = self._neutral_avatar_body_pos_for_work_xy(job, xy, hold_by_frame)
                    selected = (xy, body_rot, hold_by_frame, footprint, body_pos)
        (
            self._neutral_avatar_selected_work_xy,
            body_rot,
            hold_by_frame,
            footprint,
            body_pos,
        ) = selected
        self._neutral_avatar_motion_footprint = footprint
        type(self).NEUTRAL_MOTION_FOOTPRINTS[str(job["base_motion"])] = {
            "kind": str(footprint["kind"]),
            "min_offset": np.asarray(footprint["min_offset"], dtype=float).tolist(),
            "max_offset": np.asarray(footprint["max_offset"], dtype=float).tolist(),
            "size": np.asarray(footprint["size"], dtype=float).tolist(),
        }
        self.neutral_avatar_target = np.array(
            [
                float(self._neutral_avatar_selected_work_xy[0]),
                float(self._neutral_avatar_selected_work_xy[1]),
                float(self.TABLE_TOP_Z),
            ],
            dtype=np.float64,
        )

        first_event = next(e for e in job["events"] if e["type"] in ("attach", "pick"))
        first_hold = hold_by_frame[int(first_event["clip_frame"])]
        first_target = np.array(
            [
                float(self._neutral_avatar_work_xy(job)[0]),
                float(self._neutral_avatar_work_xy(job)[1]),
                float(self._neutral_avatar_prop_center_z(str(first_event["object"]))),
            ],
            dtype=np.float64,
        )
        shift = first_target - first_hold
        body_rot = np.asarray(body_rot, dtype=np.float64).copy()

        self._neutral_avatar_body_rot = body_rot
        self._place_neutral_avatar_props(job, hold_by_frame, shift, body_pos=body_pos)
        self.avatar.reset(body_pos, body_rot)
        settle_steps = int(self.config.get("settle_steps", BaseTask.SETTLE_STEPS))
        for _ in range(max(0, settle_steps)):
            self.step_sim()
        if not bool(self.config.get("neutral_avatar_skip_reset_props_upright", False)):
            self._neutral_reset_can_props_upright(job)
        self.avatar.reset(body_pos, body_rot)
        if not bool(self.config.get("neutral_avatar_skip_hold_motion_frame_idle", False)):
            self._neutral_avatar_hold_motion_frame_idle(job["motion"], 0)

        self._neutral_avatar_body_pos = body_pos
        self._neutral_avatar_body_rot = body_rot
        self._neutral_avatar_layout_ready = True
        self.avatar_collided = False
        self.avatar_collision_log = []
        self._avatar_collision_tick = 0
        self._create_neutral_avatar_region_debug_overlays(job, forbidden)
        return True

    def _neutral_avatar_body_rot_for_work_xy(self, job: dict, xy: np.ndarray) -> np.ndarray:
        task_placement = self._neutral_avatar_task_placement(job)
        raw_adjustment = self._neutral_avatar_raw_preview_adjustment(job)
        side = self._neutral_avatar_side_for_work_xy(job, xy)
        if side:
            return self._neutral_avatar_body_rot_for_side(side, job)
        if bool(task_placement.get("use_raw_preview_avatar_rotation", False)):
            if raw_adjustment.get("avatar_face") == "image_down":
                return np.asarray(self.avatar_init_rot, dtype=np.float64).copy()
            return np.eye(3, dtype=np.float64)
        return np.asarray(self.avatar_init_rot, dtype=np.float64).copy()

    def _neutral_avatar_motion_visual_yaw_offset(self, job: dict | None = None) -> float:
        if job and job.get("kind") == "sponge":
            return np.pi / 2.0
        if job and job.get("kind") == "put_objects_in_bowl":
            return np.pi / 2.0
        if job and job.get("kind") == "sitting_clap":
            if "neutral_clap_visual_yaw_offset_deg" in self.config:
                return float(np.deg2rad(self.config["neutral_clap_visual_yaw_offset_deg"]))
            return 0.0
        return 0.0

    def _neutral_avatar_body_rot_for_side(self, side: str, job: dict | None = None) -> np.ndarray:
        if side == "back":
            yaw = 0.0
        elif side == "left":
            yaw = np.pi / 2.0
        elif side == "right":
            yaw = -np.pi / 2.0
        else:
            return np.asarray(self.avatar_init_rot, dtype=np.float64).copy()
        yaw += self._neutral_avatar_motion_visual_yaw_offset(job)
        c, s = float(np.cos(yaw)), float(np.sin(yaw))
        rz = np.array(
            [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        return rz @ np.asarray(self.avatar_init_rot, dtype=np.float64)

    def _neutral_avatar_body_pos_for_work_xy(
        self, job: dict, xy: np.ndarray, hold_by_frame: dict[int, np.ndarray]
    ) -> np.ndarray:
        first_event = next(e for e in job["events"] if e["type"] in ("attach", "pick"))
        first_hold = hold_by_frame[int(first_event["clip_frame"])]
        first_target = np.array(
            [
                float(np.asarray(xy, dtype=np.float64).ravel()[0]),
                float(np.asarray(xy, dtype=np.float64).ravel()[1]),
                float(self._neutral_avatar_prop_center_z(str(first_event["object"]))),
            ],
            dtype=np.float64,
        )
        shift = first_target - first_hold
        body_pos = (
            np.asarray(self.avatar_init_pos, dtype=np.float64).copy()
            + shift
            + self._neutral_avatar_body_offset(job)
        )
        task_placement = self._neutral_avatar_task_placement(job)
        if "avatar_root_table_z_offset" in task_placement:
            body_pos[2] = (
                float(self.TABLE_TOP_Z)
                + float(task_placement["avatar_root_table_z_offset"])
            )
        return body_pos

    def _neutral_avatar_clamp_body_edge_gap(
        self, body_pos: np.ndarray, target_xy: np.ndarray, side: str | None = None
    ) -> np.ndarray:
        body_pos = np.asarray(body_pos, dtype=np.float64).copy()
        target = np.asarray(target_xy, dtype=np.float64).ravel()[:2]
        xmin, xmax, ymin, ymax = self._neutral_table_xy_bounds()
        max_gap = float(self.config.get(
            "neutral_avatar_max_table_edge_gap",
            self.NEUTRAL_AVATAR_MAX_TABLE_EDGE_GAP,
        ))
        min_gap = float(self.config.get(
            "neutral_avatar_min_table_edge_gap",
            self.NEUTRAL_AVATAR_MIN_TABLE_EDGE_GAP,
        ))
        if max_gap <= 0.0 and min_gap <= 0.0:
            return body_pos
        max_gap = max(float(max_gap), float(min_gap))
        side = str(side or "")
        if side == "left":
            edge = 0
        elif side == "right":
            edge = 1
        elif side == "back":
            edge = 3
        else:
            dists = np.array([
                target[0] - float(xmin),
                float(xmax) - target[0],
                target[1] - float(ymin),
                float(ymax) - target[1],
            ], dtype=np.float64)
            edge = int(np.argmin(dists))
        if edge == 0:
            body_pos[0] = float(np.clip(body_pos[0], float(xmin) - max_gap, float(xmin) - min_gap))
        elif edge == 1:
            body_pos[0] = float(np.clip(body_pos[0], float(xmax) + min_gap, float(xmax) + max_gap))
        elif edge == 2:
            body_pos[1] = float(np.clip(body_pos[1], float(ymin) - max_gap, float(ymin) - min_gap))
        elif edge == 3:
            body_pos[1] = float(np.clip(body_pos[1], float(ymax) + min_gap, float(ymax) + max_gap))
        return body_pos

    def _neutral_avatar_body_edge_gap_is_valid(
        self, side: str, target_xy: np.ndarray, body_xy: np.ndarray
    ) -> bool:
        target = np.asarray(target_xy, dtype=np.float64).ravel()[:2]
        body = np.asarray(body_xy, dtype=np.float64).ravel()[:2]
        xmin, xmax, ymin, ymax = self._neutral_table_xy_bounds()
        min_gap = float(self.config.get(
            "neutral_avatar_min_table_edge_gap",
            self.NEUTRAL_AVATAR_MIN_TABLE_EDGE_GAP,
        ))
        if side == "back":
            gap = body[1] - float(ymax)
            return bool(min_gap <= gap and target[1] - body[1] < -0.18)
        if side == "right":
            gap = body[0] - float(xmax)
            return bool(min_gap <= gap and target[0] - body[0] < -0.18)
        if side == "left":
            gap = float(xmin) - body[0]
            return bool(min_gap <= gap and target[0] - body[0] > 0.18)
        return bool(
            body[0] <= float(xmin) - min_gap
            or body[0] >= float(xmax) + min_gap
            or body[1] <= float(ymin) - min_gap
            or body[1] >= float(ymax) + min_gap
        )

    def _neutral_avatar_push_body_outside_table(
        self, body_pos: np.ndarray, target_xy: np.ndarray, edge_gap: float
    ) -> np.ndarray:
        body_pos = np.asarray(body_pos, dtype=np.float64).copy()
        target = np.asarray(target_xy, dtype=np.float64).ravel()[:2]
        xmin, xmax, ymin, ymax = self._neutral_table_xy_bounds()
        dists = np.array([
            target[0] - float(xmin),
            float(xmax) - target[0],
            target[1] - float(ymin),
            float(ymax) - target[1],
        ], dtype=np.float64)
        edge = int(np.argmin(dists))
        gap = float(edge_gap)
        if edge == 0:
            body_pos[0] = float(xmin) - gap
        elif edge == 1:
            body_pos[0] = float(xmax) + gap
        elif edge == 2:
            body_pos[1] = float(ymin) - gap
        else:
            body_pos[1] = float(ymax) + gap
        return body_pos

    def _neutral_avatar_body_pose_is_valid(
        self, job: dict, xy: np.ndarray, body_pos: np.ndarray
    ) -> bool:
        if job.get("kind") in ("mouse", "sponge") and not bool(self.config.get(
            "neutral_avatar_require_surface_body_edge_gap",
            False,
        )):
            return True
        if job.get("kind") == "put_objects_in_bowl":
            body = np.asarray(body_pos, dtype=np.float64).ravel()[:2]
            side = self._neutral_avatar_side_for_work_xy(job, xy)
            target = np.asarray(xy, dtype=np.float64).ravel()[:2]
            return self._neutral_avatar_body_edge_gap_is_valid(side, target, body)
        if job.get("kind") == "bottle":
            target = np.asarray(xy, dtype=np.float64).ravel()[:2]
            body = np.asarray(body_pos, dtype=np.float64).ravel()[:2]
            side = self._neutral_avatar_side_for_work_xy(job, xy)
            return bool(
                self._neutral_avatar_body_edge_gap_is_valid(side, target, body)
                and np.linalg.norm(target - body) > 0.18
            )
        side = self._neutral_avatar_side_for_work_xy(job, xy)
        if side:
            target = np.asarray(xy, dtype=np.float64).ravel()[:2]
            body = np.asarray(body_pos, dtype=np.float64).ravel()[:2]
            return self._neutral_avatar_body_edge_gap_is_valid(side, target, body)
        return True

    def _create_neutral_avatar_region_debug_overlays(
        self, job: dict, forbidden: list[tuple[np.ndarray, float, str]]
    ) -> None:
        if not bool(self.config.get("neutral_avatar_debug_regions")):
            return
        base_z = float(self.TABLE_TOP_Z + 0.020)
        # Top-down region priority is encoded by height so overlapping tiles
        # render deterministically. Requested order:
        # red forbidden > yellow selected > orange rejected > blue table.
        # Green clear-but-not-selected stays below orange and above blue.
        layer_z = {
            "table": base_z,
            "clear": base_z + 0.010,
            "blocked": base_z + 0.020,
            "selected": base_z + 0.030,
            "object": base_z + 0.035,
            "forbidden": base_z + 0.040,
        }
        pools = getattr(self, "_neutral_avatar_region_debug_tiles", {}) or {}
        used = {name: 0 for name in pools}
        tile_half = float(self.config.get("neutral_avatar_debug_region_tile_half", 0.025))
        step = 2.0 * tile_half
        park = np.array([8.0, 8.0, base_z], dtype=float)

        for pool in pools.values():
            for ent in pool:
                ent.set_pos(park)

        def add_region(min_xy, max_xy, pool_name: str):
            pool = pools.get(pool_name)
            if not pool:
                return
            min_xy = np.asarray(min_xy, dtype=np.float64)
            max_xy = np.asarray(max_xy, dtype=np.float64)
            xs = np.arange(float(min_xy[0]) + tile_half, float(max_xy[0]), step)
            ys = np.arange(float(min_xy[1]) + tile_half, float(max_xy[1]), step)
            if xs.size == 0:
                xs = np.array([0.5 * (float(min_xy[0]) + float(max_xy[0]))])
            if ys.size == 0:
                ys = np.array([0.5 * (float(min_xy[1]) + float(max_xy[1]))])
            for x in xs:
                for y in ys:
                    idx = used.get(pool_name, 0)
                    if idx >= len(pool):
                        return
                    z = float(layer_z.get(pool_name, base_z))
                    pool[idx].set_pos(np.array([float(x), float(y), z], dtype=float))
                    used[pool_name] = idx + 1

        # Lowest to highest visual priority:
        # blue table, green clear, orange blocked, yellow selected, cyan object,
        # red forbidden.
        xmin, xmax, ymin, ymax = self._neutral_table_xy_bounds()
        add_region((xmin, ymin), (xmax, ymax), "table")

        # Candidate swept AABBs: blocked orange, clear green, selected yellow.
        records_by_pool = {"clear": [], "blocked": [], "selected": []}
        for rec in getattr(self, "_neutral_avatar_region_debug_records", []) or []:
            if rec.get("selected"):
                pool_name = "selected"
            elif rec.get("footprint_clear") and rec.get("side_region_ok", True) and rec.get("body_ok"):
                pool_name = "clear"
            else:
                pool_name = "blocked"
            records_by_pool[pool_name].append(rec)
        for pool_name in ("clear", "blocked", "selected"):
            for rec in records_by_pool[pool_name]:
                xy = np.asarray(rec["xy"], dtype=np.float64).ravel()[:2]
                fp = rec["footprint"]
                min_xy = xy + np.asarray(fp["min_offset"], dtype=np.float64)
                max_xy = xy + np.asarray(fp["max_offset"], dtype=np.float64)
                add_region(min_xy, max_xy, pool_name)

        # Initial neutral prop/object regions used by reset-time layout
        # clearance. These are cyan so they can be inspected independently
        # from the larger swept motion footprint.
        for obj_name, obj in sorted(getattr(self, "neutral_avatar_props", {}).items()):
            try:
                center = to_numpy(obj.get_pos()).ravel()[:2].astype(np.float64)
            except Exception:
                continue
            radius = float(self._neutral_prop_radius(str(obj_name)))
            if (
                job.get("kind") == "put_objects_in_bowl"
                and str(obj_name).startswith("object_")
            ):
                radius = float(self._neutral_avatar_can_body_radius(str(obj_name)))
            add_region(center - radius, center + radius, "object")
        if hasattr(self, "neutral_avatar_bowl"):
            try:
                center = to_numpy(self.neutral_avatar_bowl.get_pos()).ravel()[:2].astype(np.float64)
                half = np.asarray(self.NEUTRAL_BOWL_HALF_XY, dtype=np.float64).ravel()[:2]
                add_region(center - half, center + half, "object")
            except Exception:
                pass

        # Forbidden object/corridor regions are red and intentionally highest.
        for center, radius, _label in forbidden:
            c = np.asarray(center, dtype=np.float64).ravel()[:2]
            r = float(radius)
            add_region(c - r, c + r, "forbidden")

    def _neutral_avatar_hold_motion_frame_idle(self, motion_name: str, frame_idx: int) -> None:
        """Pose the avatar on a motion frame without advancing the animation."""
        if self.avatar is None:
            return
        if motion_name not in self.avatar.motion_modules:
            self.avatar.play_animation(motion_name)
        self._neutral_avatar_apply_motion_frame(motion_name, frame_idx)
        self.avatar.robot.action_state = AvatarState.NO_ACTION
        self.avatar.robot.action_status = ActionStatus.INIT

    def _start_neutral_avatar_table_work(self, add_start_delay: bool = True) -> bool:
        if self.avatar is None or getattr(self, "_neutral_avatar_started", False):
            return False
        job = self._neutral_avatar_job
        if not job:
            return False

        # All of this is pre-task scene/avatar setup (incl. the layout settle
        # that lets props fall to rest); none of it should land in the video or
        # the training data.
        with self.suppress_recording():
            self._prepare_neutral_avatar_start_layout()
            self._install_neutral_loop_subclip(job)
            body_pos = np.asarray(
                getattr(self, "_neutral_avatar_body_pos", self.avatar_init_pos),
                dtype=np.float64,
            )
            footprint = self._neutral_avatar_motion_footprint
            self.avatar.play_animation(job["motion"])
            event_frame_offset = 0
            if add_start_delay:
                event_frame_offset = self._delay_neutral_avatar_motion_start(
                    job["motion"],
                    int(self.config.get("neutral_avatar_start_delay_steps", 0))
                )
            self._patch_neutral_avatar_step_sim(job, event_frame_offset=event_frame_offset)
            self._neutral_avatar_apply_motion_frame(job["motion"], 0)
        self._neutral_avatar_started = True
        self.avatar_collided = False
        self.avatar_collision_log = []
        self._neutral_debug(
            f"[neutral_avatar] started {job['motion']} at {self._neutral_avatar_motion_slow():g}x slow, "
            f"body={body_pos.round(3).tolist()}, target={self.neutral_avatar_target.round(3).tolist()}, "
            f"footprint={np.round(footprint['min_offset'], 3).tolist()}.."
            f"{np.round(footprint['max_offset'], 3).tolist()}"
        )
        return True

    def _neutral_avatar_apply_motion_frame(self, motion_name: str, frame_idx: int) -> None:
        """Force the avatar skin to an exact animation frame."""
        motion = self.avatar.motion_modules.get(motion_name) if self.avatar is not None else None
        if motion is None or not getattr(motion, "data", None):
            return
        idx_raw = int(frame_idx)
        if idx_raw < 0:
            idx_raw = len(motion.data) + idx_raw
        idx = int(np.clip(idx_raw, 0, len(motion.data) - 1))
        motion.at_frame = idx
        self.avatar.robot.pose = motion.data[idx]
        self.avatar.robot.node_trans = motion.node_data[idx]
        self.avatar.robot.global_mat = motion.global_mat
        self.avatar.robot.global_mat_inv = motion.global_mat_inv
        self.avatar.robot.update()

    def _neutral_avatar_event_hold_offset(self, event: dict | None, obj_name: str) -> float:
        if event is not None and "hold_offset" in event:
            return float(event["hold_offset"])
        return (
            self.NEUTRAL_AVATAR_BOTTLE_HOLD_OFFSET
            if str(obj_name) == "bottle"
            else self.NEUTRAL_AVATAR_HOLD_OFFSET
        )

    def _neutral_avatar_hold_pos(
        self, hand_id: int, obj_name: str = "", event: dict | None = None
    ) -> np.ndarray:
        palm = np.asarray(self.avatar.robot.get_palm_center(int(hand_id)), dtype=np.float64)
        _, hand_frame = self.avatar.robot._get_hand_frame(int(hand_id))
        palm_normal = np.asarray(hand_frame[:, 2], dtype=np.float64)
        pos = palm + self._neutral_avatar_event_hold_offset(event, obj_name) * palm_normal
        local = None if event is None else event.get("hold_offset_local")
        if local is not None:
            local_arr = np.asarray(local, dtype=np.float64).ravel()
            if local_arr.size != 3:
                raise ValueError(f"hold_offset_local must have 3 values, got {local!r}")
            pos = pos + hand_frame @ local_arr
        return pos

    def _neutral_avatar_palm_center_pos(
        self, hand_id: int, event: dict | None = None
    ) -> np.ndarray:
        pos = np.asarray(self.avatar.robot.get_palm_center(int(hand_id)), dtype=np.float64).copy()
        if event is not None:
            _, hand_frame = self.avatar.robot._get_hand_frame(int(hand_id))
            local = event.get("hold_offset_local")
            if local is not None:
                local_arr = np.asarray(local, dtype=np.float64).ravel()
                if local_arr.size != 3:
                    raise ValueError(f"hold_offset_local must have 3 values, got {local!r}")
                pos = pos + hand_frame @ local_arr
        return pos

    def _neutral_avatar_surface_prop_pos(
        self, job: dict, obj_name: str, hand_id: int, event: dict
    ) -> np.ndarray:
        # Match render_raw_blender_interactions.py's default
        # surface_attach_strategy="palm_center_table_xy".
        pos = self._neutral_avatar_palm_center_pos(int(hand_id), event)
        pos = pos + self._neutral_avatar_initial_object_offset(job, obj_name, None)
        pos[2] = self._neutral_avatar_prop_center_z(obj_name)
        return pos

    def _neutral_avatar_event_anchor_pos(
        self, job: dict, hand_id: int, obj_name: str, event: dict
    ) -> np.ndarray:
        if str(obj_name) in ("sponge", "mouse"):
            return self._neutral_avatar_palm_center_pos(int(hand_id), event)
        return self._neutral_avatar_hold_pos(int(hand_id), obj_name, event)

    def _neutral_avatar_snap_prop_to_event_pose(
        self, job: dict, obj_name: str, hand_id: int, event: dict
    ) -> None:
        obj = self.neutral_avatar_props[obj_name]
        if obj_name == "phone":
            # Seat the phone in the palm exactly like scripts/render_talking.py
            # (palm centre, lowered a touch; screen faces the head via prop_quat).
            palm = np.asarray(
                self.avatar.robot.get_palm_center(int(hand_id)), dtype=np.float64
            )
            pos = palm + np.array([0.0, 0.0, -self.NEUTRAL_PHONE_DOWN], dtype=np.float64)
            obj.set_pos(pos.astype(float))
            obj.set_quat(self._neutral_avatar_prop_quat("phone"))
            self._neutral_zero_entity_velocity(obj)
            return
        if obj_name in ("sponge", "mouse"):
            pos = self._neutral_avatar_surface_prop_pos(job, obj_name, hand_id, event)
        else:
            pos = self._neutral_avatar_event_anchor_pos(job, hand_id, obj_name, event)
            pos = pos + self._neutral_avatar_initial_object_offset(job, obj_name, None)
            if obj_name == "bottle" or str(obj_name).startswith("bottle_"):
                pos[:2] = self._neutral_avatar_clamp_table_xy(
                    pos[:2],
                    margin=self._neutral_avatar_table_margin_for_prop(job, obj_name),
                )
                pos[2] = self._neutral_avatar_prop_center_z(obj_name)
        obj.set_pos(pos.astype(float))
        obj.set_quat(self._neutral_avatar_prop_quat(obj_name))
        self._neutral_zero_entity_velocity(obj)

    def _neutral_avatar_transition_frames(self) -> int:
        return max(
            1,
            int(round(
                float(self.NEUTRAL_AVATAR_TRANSITION_BASE_FRAMES)
                * self._neutral_avatar_motion_slow()
            )),
        )

    def _neutral_avatar_measure_hold_points(self, job: dict) -> dict[int, np.ndarray]:
        base_pos = np.asarray(self.avatar_init_pos, dtype=np.float64).copy()
        base_rot = np.asarray(
            getattr(self, "_neutral_avatar_measure_body_rot", None)
            if getattr(self, "_neutral_avatar_measure_body_rot", None) is not None
            else self.avatar_init_rot,
            dtype=np.float64,
        ).copy()
        self.avatar.reset(base_pos, base_rot)
        old_ratio = float(self.avatar.frame_ratio)
        self.avatar.frame_ratio = 1.0
        self.avatar.play_animation(job["motion"])
        needed = {
            int(e["clip_frame"]): e
            for e in job["events"]
            if e["type"] in ("attach", "pick", "put")
        }
        hold_by_frame = {}
        step = 0
        max_step = max(needed) + 2 if needed else 0
        while not self.avatar.spare() and step <= max_step:
            self.scene.step()
            self.avatar.step()
            if step in needed:
                event = needed[step]
                hand_id = int(event["hand_id"])
                obj_name = str(event["object"])
                hold_by_frame[step] = self._neutral_avatar_event_anchor_pos(job, hand_id, obj_name, event)
            step += 1
        self.avatar.frame_ratio = old_ratio
        missing = sorted(set(needed) - set(hold_by_frame))
        if missing:
            raise RuntimeError(f"neutral avatar dry-run missed frames {missing}")
        return hold_by_frame

    def _neutral_prop_radius(self, obj_name: str) -> float:
        return float(self.NEUTRAL_PROP_RADII.get(str(obj_name), 0.055))

    def _neutral_motion_table_margin(self, footprint: dict | None = None) -> float:
        kind = str((footprint or {}).get("kind", ""))
        if kind == "sponge":
            return float(self.config.get(
                "neutral_sponge_table_margin",
                self.NEUTRAL_SPONGE_TABLE_MARGIN,
            ))
        return 0.0

    def _neutral_avatar_table_margin_for_prop(self, job: dict, obj_name: str) -> float:
        is_can = (
            job.get("kind") == "put_objects_in_bowl"
            and str(obj_name).startswith("object_")
        )
        radius = (
            self._neutral_avatar_can_body_radius(obj_name)
            if is_can
            else self._neutral_prop_radius(obj_name)
        )
        margin = float(radius) + float(self.config.get("neutral_prop_table_margin", 0.010))
        if str(obj_name) == "bottle" or str(obj_name).startswith("bottle_"):
            margin = max(
                margin,
                float(self.config.get(
                    "neutral_bottle_table_margin",
                    self.NEUTRAL_BOTTLE_TABLE_MARGIN,
                )),
            )
        return margin

    def _neutral_avatar_prop_center_z(self, obj_name: str) -> float:
        if str(obj_name) == "sponge":
            return float(self.TABLE_TOP_Z + self.NEUTRAL_SPONGE_HALF_THICKNESS)
        if str(obj_name).startswith("object_"):
            defaults = self._neutral_put_objects_clip_defaults()
            off = self.config.get(
                "neutral_put_objects_object_rest_z_offset",
                defaults.get("rest_z_offset", self.NEUTRAL_PUT_OBJECTS_OBJECT_REST_Z_OFFSET),
            )
            if off is None:
                off = self.NEUTRAL_PUT_OBJECTS_CAN_REST_Z_OFFSET
            return float(self.TABLE_TOP_Z + float(off))
        if str(obj_name) == "bottle" or str(obj_name).startswith("bottle_"):
            return float(self.TABLE_TOP_Z + 0.060)
        if str(obj_name) == "mouse":
            task_placement = self._neutral_avatar_task_placement()
            offsets = dict(task_placement.get("object_center_z_offsets", {}) or {})
            return float(self.TABLE_TOP_Z + float(offsets.get(
                "mouse",
                self.NEUTRAL_MOUSE_CENTER_Z_OFFSET,
            )))
        return float(self.TABLE_TOP_Z + self.NEUTRAL_PROP_HALF + 0.002)

    def _neutral_avatar_compute_motion_footprint(self, job: dict) -> dict:
        """Dry-run the neutral animation and return object-swept XY offsets.

        The returned AABB is relative to the first attached/picked object's
        target point, so it can be translated to candidate table positions.
        It includes held-object sweeps and static helper objects such as the
        bowl used by the put-objects-in-bowl motion.
        """
        base_pos = np.asarray(self.avatar_init_pos, dtype=np.float64).copy()
        base_rot = np.asarray(
            getattr(self, "_neutral_avatar_measure_body_rot", None)
            if getattr(self, "_neutral_avatar_measure_body_rot", None) is not None
            else self.avatar_init_rot,
            dtype=np.float64,
        ).copy()
        self.avatar.reset(base_pos, base_rot)
        old_ratio = float(self.avatar.frame_ratio)
        self.avatar.frame_ratio = 1.0
        self.avatar.play_animation(job["motion"])

        events_by_frame: dict[int, list[dict]] = {}
        for event in job["events"]:
            events_by_frame.setdefault(int(event["clip_frame"]), []).append(event)
        first_event = next(e for e in job["events"] if e["type"] in ("attach", "pick"))
        first_frame = int(first_event["clip_frame"])
        first_hand = int(first_event["hand_id"])
        first_obj = str(first_event["object"])

        active: dict[str, tuple[int, float, dict]] = {}
        points = []
        region_samples = []
        first_anchor = None
        first_attach_by_object: set[str] = set()
        max_frame = max(
            int(job["clip"].get("num_frames", 0)),
            max(events_by_frame) + 2 if events_by_frame else 0,
        )
        step = 0
        while not self.avatar.spare() and step <= max_frame + 2:
            self.scene.step()
            self.avatar.step()
            for event in events_by_frame.get(step, []):
                obj_name = str(event["object"])
                hand_id = int(event["hand_id"])
                radius = self._neutral_prop_radius(obj_name)
                if event["type"] in ("attach", "pick"):
                    active[obj_name] = (hand_id, radius, event)
                    p = self._neutral_avatar_event_anchor_pos(job, hand_id, obj_name, event)
                    points.append((p[:2], radius, obj_name))
                    region_samples.append((p[:2].copy(), radius, obj_name))
                    first_attach_by_object.add(obj_name)
                    if step == first_frame and hand_id == first_hand and obj_name == first_obj:
                        first_anchor = p[:2].copy()
                elif event["type"] in ("detach", "put"):
                    if job.get("kind") in ("sponge", "mouse"):
                        active_event = active.get(obj_name, (hand_id, radius, event))[2]
                        hold_event = (
                            event
                            if "hold_offset" in event or "hold_offset_local" in event
                            else active_event
                        )
                        p = self._neutral_avatar_event_anchor_pos(job, hand_id, obj_name, hold_event)
                        points.append((p[:2], radius, obj_name))
                        region_samples.append((p[:2].copy(), radius, obj_name))
                    elif (
                        job.get("kind") == "put_objects_in_bowl"
                        and bool((job.get("clip") or {}).get("reverse"))
                    ):
                        # Reverse take-objects clip: the prop is SET DOWN here,
                        # so its target landing spot on the table must clear the
                        # task regions — the mirror of the forward clip clearing
                        # the can INIT (pick) positions added above.
                        p = self._neutral_avatar_event_anchor_pos(job, hand_id, obj_name, event)
                        points.append((p[:2], radius, obj_name))
                        region_samples.append((p[:2].copy(), radius, obj_name))
                    active.pop(obj_name, None)
                elif event["type"] == "switch":
                    if obj_name in active:
                        active[obj_name] = (hand_id, radius, event)

            for obj_name, (hand_id, radius, event) in active.items():
                if job.get("kind") == "bottle":
                    continue
                if job.get("kind") == "put_objects_in_bowl" and obj_name in first_attach_by_object:
                    continue
                p = self._neutral_avatar_event_anchor_pos(job, hand_id, obj_name, event)
                points.append((p[:2], radius, obj_name))
                region_samples.append((p[:2].copy(), radius, obj_name))
            step += 1

        self.avatar.frame_ratio = old_ratio
        if first_anchor is None:
            first_anchor = np.asarray(points[0][0], dtype=np.float64) if points else np.zeros(2)

        mins = []
        maxs = []
        for xy, radius, _obj_name in points:
            offset = np.asarray(xy, dtype=np.float64) - first_anchor
            mins.append(offset - radius)
            maxs.append(offset + radius)

        bowl_center_offset = None
        if job["kind"] == "put_objects_in_bowl":
            bowl_event_type = self._neutral_bowl_event_type(job)
            put_offsets = []
            for event in job["events"]:
                if event["type"] != bowl_event_type:
                    continue
                # Use the event hold position as the center of the bowl target.
                frame = int(event["clip_frame"])
                hand_id = int(event["hand_id"])
                self.avatar.reset(base_pos, base_rot)
                self.avatar.frame_ratio = 1.0
                self.avatar.play_animation(job["motion"])
                for _ in range(frame + 1):
                    self.scene.step()
                    self.avatar.step()
                put_offsets.append(
                    self._neutral_avatar_event_anchor_pos(
                        job, hand_id, str(event["object"]), event
                    )[:2]
                    - first_anchor
                )
                self.avatar.frame_ratio = old_ratio
            if put_offsets:
                bowl_center = np.mean(np.asarray(put_offsets), axis=0)
                bowl_center_offset = np.asarray(bowl_center, dtype=np.float64).copy()
                bowl_abs = first_anchor + bowl_center
                mins.append(bowl_center - self.NEUTRAL_BOWL_HALF_XY)
                maxs.append(bowl_center + self.NEUTRAL_BOWL_HALF_XY)
                bowl_radius = float(np.linalg.norm(self.NEUTRAL_BOWL_HALF_XY))
                region_samples.append((bowl_abs, bowl_radius, "bowl"))

        self.avatar.frame_ratio = old_ratio

        if not mins:
            mins = [np.array([-0.08, -0.08], dtype=np.float64)]
            maxs = [np.array([0.08, 0.08], dtype=np.float64)]

        min_offset = np.min(np.asarray(mins), axis=0)
        max_offset = np.max(np.asarray(maxs), axis=0)
        footprint = {
            "motion": job["base_motion"],
            "kind": job["kind"],
            "min_offset": min_offset,
            "max_offset": max_offset,
            "size": max_offset - min_offset,
        }
        if bowl_center_offset is not None:
            footprint["bowl_center_offset"] = bowl_center_offset
        region_points = []
        for xy, radius, label in region_samples:
            offset = np.asarray(xy, dtype=np.float64).ravel()[:2] - first_anchor
            region_points.append({
                "offset": offset.tolist(),
                "radius": float(radius),
                "label": str(label),
            })
        footprint["region_points"] = region_points
        return footprint

    def _neutral_avatar_work_candidates(self, job: dict) -> list[np.ndarray]:
        footprint = getattr(self, "_neutral_avatar_motion_footprint", None)
        motion_candidates = self._neutral_avatar_motion_work_candidates(job, footprint)
        if motion_candidates:
            return [np.asarray(xy, dtype=np.float64).ravel()[:2] for xy in motion_candidates]

        if footprint is None and job.get("kind") in ("mouse", "sponge"):
            side_aware = self._neutral_avatar_side_aware_work_candidates(job)
            if side_aware:
                return [np.asarray(xy, dtype=np.float64).ravel()[:2] for xy in side_aware]

        if job["kind"] == "sponge":
            attr = "NEUTRAL_SPONGE_WORK_CANDIDATES"
        elif job["kind"] == "mouse":
            attr = "NEUTRAL_MOUSE_WORK_CANDIDATES"
        else:
            attr = "NEUTRAL_WORK_CANDIDATES"
        raw = getattr(self, attr, None)
        raw_is_absolute_override = raw is not None
        if raw is None:
            if job["kind"] == "sponge":
                region_attr = "NEUTRAL_SPONGE_WORK_REGION"
            elif job["kind"] == "mouse":
                region_attr = "NEUTRAL_MOUSE_WORK_REGION"
            else:
                region_attr = "NEUTRAL_WORK_REGION"
            region = getattr(self, region_attr, None)
            if region is not None:
                arr = np.asarray(region, dtype=np.float64).ravel()
                if arr.size != 4:
                    raise ValueError(
                        f"{region_attr} must be (x_min, x_max, y_min, y_max); got {region!r}"
                    )
                x0, x1, y0, y1 = [float(v) for v in arr]
                n = int(self.config.get("neutral_avatar_region_candidates", 12))
                n = max(1, n)
                raw = [
                    np.array([
                        np.random.uniform(x0, x1),
                        np.random.uniform(y0, y1),
                    ], dtype=np.float64)
                    for _ in range(n)
                ]
                # Deterministic fallbacks keep selection robust when the
                # sampled points all clip a task object or table boundary.
                raw.extend([
                    np.array([(x0 + x1) * 0.5, (y0 + y1) * 0.5], dtype=np.float64),
                    np.array([x0, (y0 + y1) * 0.5], dtype=np.float64),
                    np.array([x1, (y0 + y1) * 0.5], dtype=np.float64),
                ])
        if raw is None:
            base = self._neutral_avatar_default_work_xy(job)
            offsets = np.array([
                [0.00, 0.00],
                [0.00, 0.08],
                [0.00, -0.08],
                [-0.10, 0.00],
                [0.10, 0.00],
                [-0.10, 0.08],
                [0.10, 0.08],
            ], dtype=np.float64)
            raw = [base + off for off in offsets]
            raw_is_absolute_override = False
        shift = (
            self._neutral_avatar_work_candidate_shift_xy(job)
            if raw_is_absolute_override
            else np.zeros(2, dtype=np.float64)
        )
        return [np.asarray(xy, dtype=np.float64).ravel()[:2] + shift for xy in raw]

    def _neutral_avatar_allowed_work_regions(self, job: dict) -> list[tuple[float, float, float, float]]:
        """No task-specific side rectangles.

        Neutral side placement is shared across tasks: candidates come from
        the generic left/back/right table bands and are then filtered against
        task objects, robot regions, and the motion footprint.
        """
        return []

    def _neutral_avatar_allowed_side_work_regions(self, job: dict) -> list[tuple[str, tuple[float, float, float, float]]]:
        """Task-specific side rectangles are intentionally disabled."""
        return []

    def _neutral_avatar_blocked_sides(self, job: dict | None = None) -> set[str]:
        raw = self.config.get(
            "neutral_avatar_blocked_sides",
            getattr(self, "NEUTRAL_AVATAR_BLOCKED_SIDES", ()),
        )
        if isinstance(raw, str):
            raw = [raw]
        return {str(side) for side in (raw or [])}

    def _neutral_avatar_default_side_work_regions(
        self,
        job: dict | None = None,
    ) -> list[tuple[str, tuple[float, float, float, float]]]:
        """Generic left/back/right anchor bands for motions with no task override."""
        xmin, xmax, ymin, ymax = self._neutral_table_xy_bounds()
        xmin, xmax, ymin, ymax = [float(v) for v in (xmin, xmax, ymin, ymax)]
        x_span = max(0.01, xmax - xmin)
        y_span = max(0.01, ymax - ymin)
        side_w = min(0.25, 0.28 * x_span)
        back_h = min(0.22, 0.24 * y_span)
        y0 = ymin + 0.24 * y_span
        y1 = ymax - 0.12 * y_span
        x0 = xmin + 0.18 * x_span
        x1 = xmax - 0.18 * x_span
        if y0 >= y1:
            cy = 0.5 * (ymin + ymax)
            half = min(0.10, 0.18 * y_span)
            y0, y1 = cy - half, cy + half
        if x0 >= x1:
            cx = 0.5 * (xmin + xmax)
            half = min(0.12, 0.18 * x_span)
            x0, x1 = cx - half, cx + half
        edge_gap = 0.030
        side_regions = [
            ("right", (xmax - side_w, xmax - edge_gap, y0, y1)),
            ("back", (x0, x1, ymax - back_h, ymax - edge_gap)),
            ("left", (xmin + edge_gap, xmin + side_w, y0, y1)),
        ]
        blocked = self._neutral_avatar_blocked_sides(job)
        if blocked:
            side_regions = [
                (side, region)
                for side, region in side_regions
                if str(side) not in blocked
            ]
        return side_regions

    def _neutral_table_xy_bounds(self) -> tuple[float, float, float, float]:
        center = getattr(self, "TABLE_CENTER_XY", None)
        half = getattr(self, "TABLE_HALF_SIZE", None)
        if center is not None and half is not None:
            cx, cy = [float(v) for v in np.asarray(center, dtype=np.float64).ravel()[:2]]
            hx, hy = [float(v) for v in np.asarray(half, dtype=np.float64).ravel()[:2]]
            if hx > 0.0 and hy > 0.0:
                return (cx - hx, cx + hx, cy - hy, cy + hy)
        return tuple(float(v) for v in getattr(
            self, "NEUTRAL_TABLE_XY_BOUNDS", self.NEUTRAL_TABLE_XY_BOUNDS
        ))

    def _neutral_put_objects_plate_center_is_safe(
        self, xy: np.ndarray, footprint: dict | None
    ) -> bool:
        if footprint is None or "bowl_center_offset" not in footprint:
            return True
        center = (
            np.asarray(xy, dtype=np.float64).ravel()[:2]
            + np.asarray(footprint["bowl_center_offset"], dtype=np.float64).ravel()[:2]
        )
        xmin, xmax, ymin, ymax = self._neutral_table_xy_bounds()
        margin_xy = (
            np.asarray(self.NEUTRAL_BOWL_HALF_XY, dtype=np.float64).ravel()[:2]
            + float(self.NEUTRAL_PUT_OBJECTS_PLATE_EDGE_MARGIN)
        )
        return bool(
            float(xmin) + margin_xy[0] <= center[0] <= float(xmax) - margin_xy[0]
            and float(ymin) + margin_xy[1] <= center[1] <= float(ymax) - margin_xy[1]
        )

    def _neutral_put_objects_footprint_is_table_safe(
        self, xy: np.ndarray, footprint: dict | None
    ) -> bool:
        if footprint is None:
            return True
        min_offset = np.asarray(footprint["min_offset"], dtype=np.float64).ravel()[:2]
        max_offset = np.asarray(footprint["max_offset"], dtype=np.float64).ravel()[:2]
        min_xy = np.asarray(xy, dtype=np.float64).ravel()[:2] + min_offset
        max_xy = np.asarray(xy, dtype=np.float64).ravel()[:2] + max_offset
        xmin, xmax, ymin, ymax = self._neutral_table_xy_bounds()
        margin = float(
            getattr(
                self,
                "NEUTRAL_PUT_OBJECTS_SWEEP_TABLE_MARGIN",
                self.NEUTRAL_PUT_OBJECTS_PLATE_EDGE_MARGIN,
            )
        )
        return bool(
            float(xmin) + margin <= min_xy[0] <= max_xy[0] <= float(xmax) - margin
            and float(ymin) + margin <= min_xy[1] <= max_xy[1] <= float(ymax) - margin
        )

    def _neutral_filter_put_objects_candidates(
        self, candidates: list[np.ndarray], footprint: dict | None
    ) -> list[np.ndarray]:
        return [
            np.asarray(xy, dtype=np.float64).ravel()[:2]
            for xy in candidates
            if self._neutral_put_objects_footprint_is_table_safe(xy, footprint)
        ]

    def _neutral_avatar_table_side_for_xy(self, xy: np.ndarray) -> str:
        arr = np.asarray(xy, dtype=np.float64).ravel()[:2]
        return self._neutral_avatar_default_table_side_for_xy(arr)

    def _neutral_avatar_default_table_side_for_xy(self, xy: np.ndarray) -> str:
        arr = np.asarray(xy, dtype=np.float64).ravel()[:2]
        xmin, xmax, _ymin, ymax = self._neutral_table_xy_bounds()
        dists = {
            "left": abs(float(arr[0]) - float(xmin)),
            "right": abs(float(xmax) - float(arr[0])),
            "back": abs(float(ymax) - float(arr[1])),
        }
        return min(dists, key=dists.get)

    def _neutral_avatar_side_for_work_xy(self, job: dict, xy: np.ndarray) -> str:
        arr = np.asarray(xy, dtype=np.float64).ravel()[:2]
        key = (round(float(arr[0]), 6), round(float(arr[1]), 6))
        if key in getattr(self, "_neutral_avatar_default_side_candidates", set()):
            return self._neutral_avatar_default_table_side_for_xy(arr)
        side = getattr(self, "_neutral_avatar_candidate_sides", {}).get(
            key
        )
        if side:
            return str(side)
        side = self._neutral_avatar_table_side_for_xy(xy)
        if side:
            return side
        if job.get("kind") in ("put_objects_in_bowl", "mouse", "sponge"):
            return self._neutral_avatar_default_table_side_for_xy(xy)
        return ""

    def _neutral_avatar_motion_work_candidates(
        self,
        job: dict,
        footprint: dict | None,
    ) -> list[np.ndarray] | None:
        """Motion-level candidate generator.

        The default samples anchor points from task-provided allowed regions.
        Motion/task subclasses can override for clips whose body or prop sweep
        needs special treatment.
        """
        if job.get("kind") == "put_objects_in_bowl" and footprint is not None:
            candidate_sides: dict[tuple[float, float], str] = {}
            self._neutral_avatar_candidate_sides = candidate_sides
            self._neutral_avatar_default_side_candidates = set()

            def add_candidate(out_list: list[np.ndarray], side_name: str, xy: np.ndarray) -> None:
                arr = np.asarray(xy, dtype=np.float64).ravel()[:2]
                out_list.append(arr)
                candidate_sides[
                    (round(float(arr[0]), 6), round(float(arr[1]), 6))
                ] = str(side_name)

            explicit_side_regions = list(self._neutral_avatar_allowed_side_work_regions(job) or [])
            default_side_regions = list(self._neutral_avatar_default_side_work_regions())
            side_regions = list(explicit_side_regions)
            default_side_keys = set()
            if not side_regions:
                side_regions = list(default_side_regions)
                default_side_keys = {
                    (str(side), tuple(float(v) for v in region))
                    for side, region in default_side_regions
                }
            elif self._neutral_avatar_forced_motion_requested():
                known = {
                    (str(side), tuple(float(v) for v in region))
                    for side, region in side_regions
                }
                for side, region in default_side_regions:
                    key = (str(side), tuple(float(v) for v in region))
                    if key in known:
                        continue
                    side_regions.append((side, region))
                    default_side_keys.add(key)
            if side_regions:
                n = int(self.config.get("neutral_avatar_region_candidates", 16))
                n = max(1, n)
                out: list[np.ndarray] = []
                rects = []
                xmin, xmax, ymin, ymax = self._neutral_table_xy_bounds()
                margin = float(
                    getattr(
                        self,
                        "NEUTRAL_PUT_OBJECTS_SWEEP_TABLE_MARGIN",
                        self.NEUTRAL_PUT_OBJECTS_PLATE_EDGE_MARGIN,
                    )
                )
                old_measure_rot = getattr(self, "_neutral_avatar_measure_body_rot", None)
                try:
                    for side, region in side_regions:
                        region_key = (str(side), tuple(float(v) for v in region))
                        x0, x1, y0, y1 = self._neutral_put_objects_side_region_for_sampling(
                            side, region
                        )
                        self._neutral_avatar_measure_body_rot = self._neutral_avatar_body_rot_for_side(side, job)
                        _hold = self._neutral_avatar_measure_hold_points(job)
                        side_fp = self._neutral_avatar_compute_motion_footprint(job)
                        min_offset = np.asarray(side_fp["min_offset"], dtype=np.float64).ravel()[:2]
                        max_offset = np.asarray(side_fp["max_offset"], dtype=np.float64).ravel()[:2]
                        table_lo = (
                            np.array([float(xmin) + margin, float(ymin) + margin], dtype=np.float64)
                            - min_offset
                        )
                        table_hi = (
                            np.array([float(xmax) - margin, float(ymax) - margin], dtype=np.float64)
                            - max_offset
                        )
                        lo = np.maximum(np.array([x0, y0], dtype=np.float64), table_lo)
                        hi = np.minimum(np.array([x1, y1], dtype=np.float64), table_hi)
                        if np.all(lo <= hi):
                            rects.append((lo, hi, side_fp, side, region_key in default_side_keys))
                finally:
                    self._neutral_avatar_measure_body_rot = old_measure_rot
                if rects:
                    for lo, hi, _side_fp, side, is_default_region in rects:
                        center = np.array([
                            (float(lo[0]) + float(hi[0])) * 0.5,
                            (float(lo[1]) + float(hi[1])) * 0.5,
                        ], dtype=np.float64)
                        add_candidate(out, side, center)
                        if is_default_region:
                            self._neutral_avatar_default_side_candidates.add((
                                round(float(center[0]), 6),
                                round(float(center[1]), 6),
                            ))
                    for _ in range(n):
                        lo, hi, _side_fp, side, is_default_region = rects[int(np.random.randint(0, len(rects)))]
                        candidate = np.array([
                            np.random.uniform(float(lo[0]), float(hi[0])),
                            np.random.uniform(float(lo[1]), float(hi[1])),
                        ], dtype=np.float64)
                        add_candidate(out, side, candidate)
                        if is_default_region:
                            self._neutral_avatar_default_side_candidates.add((
                                round(float(candidate[0]), 6),
                                round(float(candidate[1]), 6),
                            ))
                    return out
                if self.config.get("neutral_avatar_force_side"):
                    centers = []
                    for _side, region in side_regions:
                        x0, x1, y0, y1 = self._neutral_put_objects_side_region_for_sampling(
                            _side, region
                        )
                        center = np.array([(x0 + x1) * 0.5, (y0 + y1) * 0.5], dtype=np.float64)
                        add_candidate(centers, _side, center)
                    return centers or None

            xmin, xmax, ymin, ymax = self._neutral_table_xy_bounds()
            margin = float(self.NEUTRAL_PUT_OBJECTS_TABLE_MARGIN)
            min_offset = np.asarray(footprint["min_offset"], dtype=np.float64).ravel()[:2]
            max_offset = np.asarray(footprint["max_offset"], dtype=np.float64).ravel()[:2]
            lo = np.array([float(xmin) + margin, float(ymin) + margin], dtype=np.float64) - min_offset
            hi = np.array([float(xmax) - margin, float(ymax) - margin], dtype=np.float64) - max_offset
            if np.all(lo <= hi):
                n = int(self.config.get("neutral_avatar_region_candidates", 16))
                n = max(1, n)
                out = [
                    np.array([
                        np.random.uniform(float(lo[0]), float(hi[0])),
                        np.random.uniform(float(lo[1]), float(hi[1])),
                    ], dtype=np.float64)
                    for _ in range(n)
                ]
                out.append((lo + hi) * 0.5)
                filtered = self._neutral_filter_put_objects_candidates(out, footprint)
                if filtered:
                    return filtered
                return out

        explicit_side_regions = self._neutral_avatar_allowed_side_work_regions(job)
        side_regions = list(explicit_side_regions)
        default_side_keys = set()
        if job.get("kind") in ("bottle", "mouse", "sponge"):
            for side, region in self._neutral_avatar_default_side_work_regions():
                key = (str(side), tuple(float(v) for v in region))
                if key in {(str(s), tuple(float(v) for v in r)) for s, r in side_regions}:
                    continue
                default_side_keys.add(key)
                side_regions.append((side, region))
        if not side_regions:
            return None
        candidate_sides: dict[tuple[float, float], str] = {}
        self._neutral_avatar_candidate_sides = candidate_sides
        self._neutral_avatar_default_side_candidates = set()

        def add_candidate(out_list: list[np.ndarray], side_name: str, xy: np.ndarray) -> None:
            arr = np.asarray(xy, dtype=np.float64).ravel()[:2]
            out_list.append(arr)
            candidate_sides[
                (round(float(arr[0]), 6), round(float(arr[1]), 6))
            ] = str(side_name)

        default_n = 48 if job.get("kind") in ("bottle", "mouse", "sponge") else 16
        n = int(self.config.get("neutral_avatar_region_candidates", default_n))
        n = max(1, n)
        out: list[np.ndarray] = []
        xmin, xmax, ymin, ymax = self._neutral_table_xy_bounds()
        old_measure_rot = getattr(self, "_neutral_avatar_measure_body_rot", None)
        task_regions_for_anchor = self.neutral_task_regions()
        for side, region in side_regions:
            key = (str(side), tuple(float(v) for v in region))
            is_default_region = key in default_side_keys
            side_footprint = footprint
            if side_footprint is None:
                try:
                    self._neutral_avatar_measure_body_rot = self._neutral_avatar_body_rot_for_side(str(side), job)
                    _hold = self._neutral_avatar_measure_hold_points(job)
                    side_footprint = self._neutral_avatar_compute_motion_footprint(job)
                finally:
                    self._neutral_avatar_measure_body_rot = old_measure_rot
            if side_footprint is None:
                continue
            table_margin = self._neutral_motion_table_margin(side_footprint)
            min_offset = np.asarray(side_footprint["min_offset"], dtype=np.float64).ravel()[:2]
            max_offset = np.asarray(side_footprint["max_offset"], dtype=np.float64).ravel()[:2]
            lo = (
                np.array([float(xmin) + table_margin, float(ymin) + table_margin], dtype=np.float64)
                - min_offset
            )
            hi = (
                np.array([float(xmax) - table_margin, float(ymax) - table_margin], dtype=np.float64)
                - max_offset
            )
            if not np.all(lo <= hi):
                continue
            forbidden_rects = self._neutral_anchor_forbidden_rects(
                side_footprint, task_regions_for_anchor,
            )
            candidates = self._neutral_empty_anchor_rect_candidates(
                lo, hi, forbidden_rects,
            )
            for candidate in candidates[: max(1, n)]:
                arr = np.asarray(candidate, dtype=np.float64).ravel()[:2]
                add_candidate(out, str(side), arr)
                if bool(is_default_region):
                    self._neutral_avatar_default_side_candidates.add((
                        round(float(arr[0]), 6),
                        round(float(arr[1]), 6),
                    ))
        return out

    def _neutral_anchor_forbidden_rects(
        self,
        footprint: dict,
        task_regions: list[tuple[np.ndarray, float, str]],
    ) -> list[tuple[float, float, float, float]]:
        """Convert object regions into forbidden anchor rectangles.

        For a fixed motion AABB, an obstacle disk forbids anchors whose
        translated AABB overlaps that disk. This is the usual configuration
        space obstacle expansion specialized to axis-aligned rectangles.
        """
        min_offset = np.asarray(footprint["min_offset"], dtype=np.float64).ravel()[:2]
        max_offset = np.asarray(footprint["max_offset"], dtype=np.float64).ravel()[:2]
        margin = float(self.NEUTRAL_CLEARANCE_MARGIN)
        rects: list[tuple[float, float, float, float]] = []
        for center, radius, label in task_regions:
            if not self._neutral_task_region_is_physical(label):
                continue
            c = np.asarray(center, dtype=np.float64).ravel()[:2]
            r = float(radius) + margin
            rects.append((
                float(c[0] - r - max_offset[0]),
                float(c[0] + r - min_offset[0]),
                float(c[1] - r - max_offset[1]),
                float(c[1] + r - min_offset[1]),
            ))
        return rects

    def _neutral_empty_anchor_rect_candidates(
        self,
        lo: np.ndarray,
        hi: np.ndarray,
        forbidden_rects: list[tuple[float, float, float, float]],
    ) -> list[np.ndarray]:
        """Return deterministic centers of free arrangement cells, largest first."""
        lo = np.asarray(lo, dtype=np.float64).ravel()[:2]
        hi = np.asarray(hi, dtype=np.float64).ravel()[:2]
        if not np.all(lo <= hi):
            return []
        xs = [float(lo[0]), float(hi[0])]
        ys = [float(lo[1]), float(hi[1])]
        clipped = []
        for x0, x1, y0, y1 in forbidden_rects:
            cx0, cx1 = max(float(x0), float(lo[0])), min(float(x1), float(hi[0]))
            cy0, cy1 = max(float(y0), float(lo[1])), min(float(y1), float(hi[1]))
            if cx0 >= cx1 or cy0 >= cy1:
                continue
            clipped.append((cx0, cx1, cy0, cy1))
            xs.extend([cx0, cx1])
            ys.extend([cy0, cy1])
        xs = sorted(set(round(float(x), 6) for x in xs))
        ys = sorted(set(round(float(y), 6) for y in ys))
        cells: list[tuple[float, list[np.ndarray]]] = []
        for x0, x1 in zip(xs[:-1], xs[1:]):
            if x1 <= x0:
                continue
            for y0, y1 in zip(ys[:-1], ys[1:]):
                if y1 <= y0:
                    continue
                center = np.array([(x0 + x1) * 0.5, (y0 + y1) * 0.5], dtype=np.float64)
                blocked = False
                for rx0, rx1, ry0, ry1 in clipped:
                    if rx0 <= center[0] <= rx1 and ry0 <= center[1] <= ry1:
                        blocked = True
                        break
                if blocked:
                    continue
                area = float((x1 - x0) * (y1 - y0))
                dx = float(x1 - x0)
                dy = float(y1 - y0)
                probes = [
                    center,
                    np.array([x0 + 0.20 * dx, y0 + 0.20 * dy], dtype=np.float64),
                    np.array([x0 + 0.80 * dx, y0 + 0.20 * dy], dtype=np.float64),
                    np.array([x0 + 0.20 * dx, y0 + 0.80 * dy], dtype=np.float64),
                    np.array([x0 + 0.80 * dx, y0 + 0.80 * dy], dtype=np.float64),
                ]
                cells.append((area, probes))
        if not cells:
            center = (lo + hi) * 0.5
            return [center]
        cells.sort(key=lambda item: item[0], reverse=True)
        out: list[np.ndarray] = []
        seen: set[tuple[float, float]] = set()
        for _area, probes in cells:
            for probe in probes:
                key = (round(float(probe[0]), 6), round(float(probe[1]), 6))
                if key in seen:
                    continue
                seen.add(key)
                out.append(probe)
        return out

    def _neutral_task_region_is_physical(self, label: object) -> bool:
        """Return whether a task region represents real occupied geometry."""
        text = str(label)
        if text.startswith("slot:") or text == "target":
            return False
        return True

    def _neutral_put_objects_side_region_for_sampling(
        self, side: str, region: tuple[float, float, float, float]
    ) -> tuple[float, float, float, float]:
        """Use a narrower anchor strip for the plate/bowl neutral clip.

        The put-objects motion has a wider object sweep than mouse/sponge, so
        sampling the full edge strip can put held objects visually too close to
        the table boundary even when the swept AABB technically fits.
        """
        x0, x1, y0, y1 = [float(v) for v in region]
        frac = float(self.config.get("neutral_put_objects_side_width_fraction", 0.5))
        frac = min(1.0, max(0.05, frac))
        if side == "left":
            x0 = x1 - (x1 - x0) * frac
        elif side == "right":
            x1 = x0 + (x1 - x0) * frac
        elif side == "back":
            y0 = y1 - (y1 - y0) * frac
        return (x0, x1, y0, y1)

    def _neutral_avatar_side_aware_work_candidates(self, job: dict) -> list[np.ndarray] | None:
        """Sample anchors so the side-specific swept footprint fits a side band."""
        explicit_side_regions = self._neutral_avatar_allowed_side_work_regions(job)
        side_regions = explicit_side_regions
        if not side_regions and job.get("kind") in ("mouse", "sponge"):
            side_regions = self._neutral_avatar_default_side_work_regions()
        if not side_regions:
            return None
        candidate_sides: dict[tuple[float, float], str] = {}
        self._neutral_avatar_candidate_sides = candidate_sides
        self._neutral_avatar_default_side_candidates = set()

        def add_candidate(out_list: list[np.ndarray], side_name: str, xy: np.ndarray) -> None:
            arr = np.asarray(xy, dtype=np.float64).ravel()[:2]
            out_list.append(arr)
            candidate_sides[
                (round(float(arr[0]), 6), round(float(arr[1]), 6))
            ] = str(side_name)

        n = int(self.config.get("neutral_avatar_region_candidates", 16))
        n = max(1, n)
        out: list[np.ndarray] = []
        old_measure_rot = getattr(self, "_neutral_avatar_measure_body_rot", None)
        try:
            for side, region in side_regions:
                if explicit_side_regions:
                    x0, x1, y0, y1 = [float(v) for v in region]
                    center = np.array([(x0 + x1) * 0.5, (y0 + y1) * 0.5], dtype=np.float64)
                else:
                    xmin, xmax, ymin, ymax = self._neutral_table_xy_bounds()
                    center = np.array([
                        0.5 * (float(xmin) + float(xmax)),
                        0.5 * (float(ymin) + float(ymax)),
                    ], dtype=np.float64)
                self._neutral_avatar_measure_body_rot = self._neutral_avatar_body_rot_for_work_xy(job, center)
                _hold = self._neutral_avatar_measure_hold_points(job)
                fp = self._neutral_avatar_compute_motion_footprint(job)
                min_offset = np.asarray(fp["min_offset"], dtype=np.float64).ravel()[:2]
                max_offset = np.asarray(fp["max_offset"], dtype=np.float64).ravel()[:2]
                xmin, xmax, ymin, ymax = self._neutral_table_xy_bounds()
                table_margin = self._neutral_motion_table_margin(fp)
                if explicit_side_regions:
                    lo = np.array([x0, y0], dtype=np.float64) - min_offset
                    hi = np.array([x1, y1], dtype=np.float64) - max_offset
                else:
                    lo = np.array([float(xmin) + table_margin, float(ymin) + table_margin], dtype=np.float64) - min_offset
                    hi = np.array([float(xmax) - table_margin, float(ymax) - table_margin], dtype=np.float64) - max_offset
                table_lo = (
                    np.array([float(xmin) + table_margin, float(ymin) + table_margin], dtype=np.float64)
                    - min_offset
                )
                table_hi = (
                    np.array([float(xmax) - table_margin, float(ymax) - table_margin], dtype=np.float64)
                    - max_offset
                )
                lo = np.maximum(lo, table_lo)
                hi = np.minimum(hi, table_hi)
                if not np.all(lo <= hi):
                    continue
                per_region = max(1, int(np.ceil(float(n) / float(len(side_regions)))))
                for _ in range(per_region):
                    add_candidate(out, side, np.array([
                        np.random.uniform(float(lo[0]), float(hi[0])),
                        np.random.uniform(float(lo[1]), float(hi[1])),
                    ], dtype=np.float64))
                add_candidate(out, side, (lo + hi) * 0.5)
        finally:
            self._neutral_avatar_measure_body_rot = old_measure_rot
        if explicit_side_regions and job.get("kind") in ("mouse", "sponge"):
            old_measure_rot = getattr(self, "_neutral_avatar_measure_body_rot", None)
            try:
                xmin, xmax, ymin, ymax = self._neutral_table_xy_bounds()
                table_center = np.array([
                    0.5 * (float(xmin) + float(xmax)),
                    0.5 * (float(ymin) + float(ymax)),
                ], dtype=np.float64)
                for side, _region in self._neutral_avatar_default_side_work_regions():
                    self._neutral_avatar_measure_body_rot = self._neutral_avatar_body_rot_for_side(side, job)
                    _hold = self._neutral_avatar_measure_hold_points(job)
                    fp = self._neutral_avatar_compute_motion_footprint(job)
                    min_offset = np.asarray(fp["min_offset"], dtype=np.float64).ravel()[:2]
                    max_offset = np.asarray(fp["max_offset"], dtype=np.float64).ravel()[:2]
                    table_margin = self._neutral_motion_table_margin(fp)
                    lo = (
                        np.array([float(xmin) + table_margin, float(ymin) + table_margin], dtype=np.float64)
                        - min_offset
                    )
                    hi = (
                        np.array([float(xmax) - table_margin, float(ymax) - table_margin], dtype=np.float64)
                        - max_offset
                    )
                    if not np.all(lo <= hi):
                        continue
                    per_region = max(1, int(np.ceil(float(n) / 3.0)))
                    center_candidate = np.clip(table_center, lo, hi)
                    add_candidate(out, side, center_candidate)
                    self._neutral_avatar_default_side_candidates.add((
                        round(float(center_candidate[0]), 6),
                        round(float(center_candidate[1]), 6),
                    ))
                    for _ in range(per_region):
                        candidate = np.array([
                            np.random.uniform(float(lo[0]), float(hi[0])),
                            np.random.uniform(float(lo[1]), float(hi[1])),
                        ], dtype=np.float64)
                        add_candidate(out, side, candidate)
                        self._neutral_avatar_default_side_candidates.add((
                            round(float(candidate[0]), 6),
                            round(float(candidate[1]), 6),
                        ))
            finally:
                self._neutral_avatar_measure_body_rot = old_measure_rot
        return out or None

    def _neutral_footprint_inside_side_region(self, xy: np.ndarray, footprint: dict, side: str) -> bool:
        if not side:
            return True
        arr_for_key = np.asarray(xy, dtype=np.float64).ravel()[:2]
        if (
            round(float(arr_for_key[0]), 6),
            round(float(arr_for_key[1]), 6),
        ) in getattr(self, "_neutral_avatar_default_side_candidates", set()):
            return True
        explicit_side_regions = self._neutral_avatar_allowed_side_work_regions(self._neutral_avatar_job or {})
        if not explicit_side_regions and (self._neutral_avatar_job or {}).get("kind") != "put_objects_in_bowl":
            return True
        if (self._neutral_avatar_job or {}).get("kind") == "put_objects_in_bowl":
            arr = np.asarray(xy, dtype=np.float64).ravel()[:2]
            side_regions = (
                explicit_side_regions
                or self._neutral_avatar_default_side_work_regions()
            )
            for reg_side, region in side_regions:
                if str(reg_side) != str(side):
                    continue
                x0, x1, y0, y1 = [float(v) for v in region]
                if (
                    x0 - 1e-6 <= arr[0] <= x1 + 1e-6
                    and y0 - 1e-6 <= arr[1] <= y1 + 1e-6
                ):
                    return self._neutral_put_objects_footprint_is_table_safe(arr, footprint)
            return False
        min_xy = np.asarray(xy, dtype=np.float64).ravel()[:2] + np.asarray(
            footprint["min_offset"], dtype=np.float64
        ).ravel()[:2]
        max_xy = np.asarray(xy, dtype=np.float64).ravel()[:2] + np.asarray(
            footprint["max_offset"], dtype=np.float64
        ).ravel()[:2]
        side_regions = (
            explicit_side_regions
            or self._neutral_avatar_default_side_work_regions()
        )
        for reg_side, region in side_regions:
            if str(reg_side) != str(side):
                continue
            x0, x1, y0, y1 = [float(v) for v in region]
            if (
                min_xy[0] >= x0 - 1e-6
                and max_xy[0] <= x1 + 1e-6
                and min_xy[1] >= y0 - 1e-6
                and max_xy[1] <= y1 + 1e-6
            ):
                return True
        return False

    def _neutral_avatar_spawn_exclude_regions(self) -> list[tuple[float, float, float]]:
        """Approximate neutral work coverage before parent tasks place clutter."""
        if self._neutral_avatar_job is None:
            try:
                self._resolve_neutral_avatar_job()
            except Exception:
                return []
        job = self._neutral_avatar_job
        if not job:
            return []

        regions: list[tuple[float, float, float]] = []
        if job["kind"] == "writing":
            seen = set()
            for side, xy in self._neutral_writing_side_candidates(job):
                if side in seen:
                    continue
                seen.add(side)
                regions.append((float(xy[0]), float(xy[1]), 0.28))
            return regions
        if job["kind"] == "sitting_clap":
            seen = set()
            for side, xy in self._neutral_clap_side_candidates(job):
                if side in seen:
                    continue
                seen.add(side)
                regions.append((float(xy[0]), float(xy[1]), 0.28))
            return regions
        if job["kind"] == "sponge":
            raw_region = getattr(self, "NEUTRAL_SPONGE_WORK_REGION", None)
            if raw_region is not None:
                x0, x1, y0, y1 = [float(v) for v in np.asarray(raw_region).ravel()[:4]]
                center = np.array([(x0 + x1) * 0.5, (y0 + y1) * 0.5], dtype=np.float64)
                radius = 0.5 * float(np.hypot(x1 - x0, y1 - y0)) + 0.28
                regions.append((float(center[0]), float(center[1]), radius))
                return regions
            candidates = self._neutral_avatar_work_candidates(job)
            for xy in candidates:
                regions.append((float(xy[0]), float(xy[1]), 0.28))
            return regions

        xy = self._neutral_avatar_default_work_xy(job)
        radius = 0.24 if job["kind"] == "put_objects_in_bowl" else 0.18
        regions.append((float(xy[0]), float(xy[1]), radius))
        return regions

    def _select_neutral_avatar_work_xy(self, job: dict, footprint: dict) -> np.ndarray:
        candidates = self._neutral_avatar_work_candidates(job)
        forbidden = self._neutral_avatar_forbidden_regions()
        for xy in candidates:
            if self._neutral_footprint_is_clear(xy, footprint, forbidden):
                self._neutral_debug(
                    f"[neutral_avatar] selected work_xy={xy.round(3).tolist()} "
                    f"for {job['base_motion']}"
                )
                return xy
        xy = candidates[0]
        self._neutral_debug(
            f"[neutral_avatar] WARNING: no clear neutral candidate for "
            f"{job['base_motion']}; using {xy.round(3).tolist()}"
        )
        return xy

    def _neutral_footprint_is_clear(
        self, xy: np.ndarray, footprint: dict, forbidden: list[tuple[np.ndarray, float, str]]
    ) -> bool:
        min_xy = xy + np.asarray(footprint["min_offset"], dtype=np.float64)
        max_xy = xy + np.asarray(footprint["max_offset"], dtype=np.float64)
        xmin, xmax, ymin, ymax = self._neutral_table_xy_bounds()
        table_margin = self._neutral_motion_table_margin(footprint)
        if (
            min_xy[0] < float(xmin) + table_margin
            or max_xy[0] > float(xmax) - table_margin
            or min_xy[1] < float(ymin) + table_margin
            or max_xy[1] > float(ymax) - table_margin
        ):
            return False
        margin = float(self.NEUTRAL_CLEARANCE_MARGIN)
        for center, radius, _label in forbidden:
            center = np.asarray(center, dtype=np.float64).ravel()[:2]
            clamped = np.minimum(np.maximum(center, min_xy), max_xy)
            if float(np.linalg.norm(center - clamped)) < float(radius) + margin:
                return False
        return True

    def _neutral_footprint_inside_table(self, xy: np.ndarray, footprint: dict) -> bool:
        min_xy = np.asarray(xy, dtype=np.float64).ravel()[:2] + np.asarray(
            footprint["min_offset"], dtype=np.float64
        ).ravel()[:2]
        max_xy = np.asarray(xy, dtype=np.float64).ravel()[:2] + np.asarray(
            footprint["max_offset"], dtype=np.float64
        ).ravel()[:2]
        xmin, xmax, ymin, ymax = self._neutral_table_xy_bounds()
        table_margin = self._neutral_motion_table_margin(footprint)
        return bool(
            min_xy[0] >= float(xmin) + table_margin
            and max_xy[0] <= float(xmax) - table_margin
            and min_xy[1] >= float(ymin) + table_margin
            and max_xy[1] <= float(ymax) - table_margin
        )

    def _neutral_motion_region_is_clear(
        self,
        xy: np.ndarray,
        footprint: dict,
        task_regions: list[tuple[np.ndarray, float, str]],
    ) -> bool:
        """Check the swept neutral region against table bounds and task regions.

        ``region_points`` are stored as offsets from the neutral anchor.  Each
        point has a radius, which lets us represent arm keypoints and neutral
        objects without collapsing everything into one oversized AABB.
        """
        anchor = np.asarray(xy, dtype=np.float64).ravel()[:2]
        xmin, xmax, ymin, ymax = self._neutral_table_xy_bounds()
        points = footprint.get("region_points") or []
        if not points:
            return self._neutral_footprint_is_clear(anchor, footprint, task_regions)
        for item in points:
            if isinstance(item, dict):
                offset = np.asarray(item.get("offset", [0.0, 0.0]), dtype=np.float64).ravel()[:2]
                radius = float(item.get("radius", 0.05))
            else:
                offset = np.asarray(item[:2], dtype=np.float64).ravel()[:2]
                radius = float(item[2])
                label = str(item[3]) if len(item) > 3 else ""
            if isinstance(item, dict):
                label = str(item.get("label", ""))
            center = anchor + offset
            table_margin = self._neutral_motion_table_margin(footprint)
            if not label.startswith("arm:") and (
                center[0] - radius < float(xmin) + table_margin
                or center[0] + radius > float(xmax) - table_margin
                or center[1] - radius < float(ymin) + table_margin
                or center[1] + radius > float(ymax) - table_margin
            ):
                return False
            if label.startswith("arm:") and str(footprint.get("kind", "")) in ("sponge", "mouse", "bottle"):
                continue
            for task_center, task_radius, task_label in task_regions:
                if not self._neutral_task_region_is_physical(task_label):
                    continue
                task_center = np.asarray(task_center, dtype=np.float64).ravel()[:2]
                if float(np.linalg.norm(center - task_center)) < radius + float(task_radius):
                    return False
        return True

    def _neutral_initial_prop_clearance(
        self,
        job: dict,
        xy: np.ndarray,
        body_pos: np.ndarray,
        obj_name: str,
        forbidden: list[tuple[np.ndarray, float, str]],
    ) -> float:
        """Signed XY clearance for the prop at its reset pose.

        Positive means the reset prop center is outside all forbidden task
        objects by at least the configured margin; negative means overlap.
        """
        prop_xy = (
            np.asarray(xy, dtype=np.float64).ravel()[:2]
            + self._neutral_avatar_initial_object_offset(job, obj_name, body_pos)[:2]
        )
        prop_radius = self._neutral_prop_radius(obj_name)
        margin = float(self.NEUTRAL_CLEARANCE_MARGIN)
        best = float("inf")
        for center, radius, _label in forbidden:
            center = np.asarray(center, dtype=np.float64).ravel()[:2]
            clearance = float(np.linalg.norm(prop_xy - center)) - float(radius) - prop_radius - margin
            best = min(best, clearance)
        return best if np.isfinite(best) else 1.0

    def _neutral_initial_prop_layout(
        self,
        job: dict,
        xy: np.ndarray,
        body_pos: np.ndarray,
        hold_by_frame: dict[int, np.ndarray],
    ) -> list[tuple[str, np.ndarray, float, bool]]:
        first_event = next(e for e in job["events"] if e["type"] in ("attach", "pick"))
        first_hold = hold_by_frame[int(first_event["clip_frame"])]
        first_target = np.array(
            [
                float(np.asarray(xy, dtype=np.float64).ravel()[0]),
                float(np.asarray(xy, dtype=np.float64).ravel()[1]),
                float(self._neutral_avatar_prop_center_z(str(first_event["object"]))),
            ],
            dtype=np.float64,
        )
        shift = first_target - first_hold
        seen = set()
        out: list[tuple[str, np.ndarray, float, bool]] = []
        for event in job["events"]:
            if event["type"] not in ("attach", "pick"):
                continue
            obj_name = str(event["object"])
            if obj_name in seen:
                continue
            seen.add(obj_name)
            pos = hold_by_frame[int(event["clip_frame"])] + shift
            pos = pos + self._neutral_avatar_initial_object_offset(job, obj_name, body_pos)
            if job["kind"] == "put_objects_in_bowl":
                pos[2] = self._neutral_avatar_prop_center_z(obj_name)
            elif obj_name == "sponge":
                pos[2] = self._neutral_avatar_prop_center_z("sponge")
            is_can = (
                job.get("kind") == "put_objects_in_bowl"
                and str(obj_name).startswith("object_")
            )
            radius = (
                self._neutral_avatar_can_body_radius(obj_name)
                if is_can
                else self._neutral_prop_radius(obj_name)
            )
            out.append((obj_name, np.asarray(pos, dtype=np.float64), float(radius), bool(is_can)))
        return out

    def _neutral_initial_props_clearance(
        self,
        job: dict,
        xy: np.ndarray,
        body_pos: np.ndarray,
        hold_by_frame: dict[int, np.ndarray],
        forbidden: list[tuple[np.ndarray, float, str]],
    ) -> float:
        """Signed initial-layout clearance across all neutral props.

        For neutral can props, require a 5 cm buffer outside the can body
        radius. This prevents the can from starting in contact with task
        objects or the other neutral prop, which otherwise makes the can
        topple during the short neutral-layout settle.
        """
        layout = self._neutral_initial_prop_layout(job, xy, body_pos, hold_by_frame)
        if not layout:
            first_event = next(e for e in job["events"] if e["type"] in ("attach", "pick"))
            return self._neutral_initial_prop_clearance(
                job, xy, body_pos, str(first_event["object"]), forbidden
            )
        best = float("inf")
        default_margin = float(self.NEUTRAL_CLEARANCE_MARGIN)
        can_margin = float(self.config.get(
            "neutral_can_clearance_margin",
            self.NEUTRAL_CAN_CLEARANCE_MARGIN,
        ))
        for obj_name, pos, radius, is_can in layout:
            margin = can_margin if is_can else default_margin
            for center, task_radius, _label in forbidden:
                center = np.asarray(center, dtype=np.float64).ravel()[:2]
                clearance = (
                    float(np.linalg.norm(pos[:2] - center))
                    - float(task_radius)
                    - float(radius)
                    - margin
                )
                best = min(best, clearance)
        for i, (_name_a, pos_a, radius_a, is_can_a) in enumerate(layout):
            for _name_b, pos_b, radius_b, is_can_b in layout[i + 1:]:
                margin = can_margin if (is_can_a or is_can_b) else default_margin
                clearance = (
                    float(np.linalg.norm(pos_a[:2] - pos_b[:2]))
                    - float(radius_a)
                    - float(radius_b)
                    - margin
                )
                best = min(best, clearance)
        return best if np.isfinite(best) else 1.0

    def _neutral_initial_props_table_clearance(
        self,
        job: dict,
        xy: np.ndarray,
        body_pos: np.ndarray,
        hold_by_frame: dict[int, np.ndarray],
    ) -> float:
        """Signed clearance from reset-layout props to the real table edge."""
        layout = self._neutral_initial_prop_layout(job, xy, body_pos, hold_by_frame)
        if not layout:
            return 1.0
        xmin, xmax, ymin, ymax = self._neutral_table_xy_bounds()
        best = float("inf")
        for obj_name, pos, radius, _is_can in layout:
            p = np.asarray(pos, dtype=np.float64).ravel()[:2]
            r = max(
                float(radius) + self._neutral_motion_table_margin(
                    getattr(self, "_neutral_avatar_motion_footprint", None)
                ),
                self._neutral_avatar_table_margin_for_prop(job, str(obj_name)),
            )
            best = min(
                best,
                p[0] - (float(xmin) + r),
                (float(xmax) - r) - p[0],
                p[1] - (float(ymin) + r),
                (float(ymax) - r) - p[1],
            )
        return best if np.isfinite(best) else 1.0

    def neutral_task_regions(self) -> list[tuple[np.ndarray, float, str]]:
        """Task occupied/safety regions used by neutral placement.

        Task subclasses can override this API with more precise geometry. The
        default starts from existing task-object regions and adds one coarse
        base disk for each robot arm entity.
        """
        regions = list(self._neutral_avatar_forbidden_regions())
        robot = getattr(self, "robot", None)
        seen = set()
        for arm_name in ("left_arm", "right_arm"):
            arm = getattr(robot, arm_name, None)
            ent = getattr(arm, "entity", None)
            if ent is None or id(ent) in seen:
                continue
            seen.add(id(ent))
            pos = None
            for getter in (
                lambda: to_numpy(ent.get_pos()).ravel()[:2],
                lambda: np.asarray(getattr(arm, "base_pos"), dtype=np.float64).ravel()[:2],
            ):
                try:
                    candidate = np.asarray(getter(), dtype=np.float64).ravel()[:2]
                except Exception:
                    continue
                if np.all(np.isfinite(candidate)):
                    pos = candidate
                    break
            if pos is None:
                links = getattr(ent, "links", []) or []
                if links:
                    try:
                        candidate = to_numpy(links[0].get_pos()).ravel()[:2]
                        if np.all(np.isfinite(candidate)):
                            pos = np.asarray(candidate, dtype=np.float64)
                    except Exception:
                        pass
            if pos is not None:
                regions.append((
                    np.asarray(pos, dtype=np.float64),
                    float(self.config.get(
                        "neutral_robot_base_region_radius",
                        self.NEUTRAL_ROBOT_BASE_REGION_RADIUS,
                    )),
                    f"robot_base:{arm_name}",
                ))
        return regions

    def _neutral_avatar_prop_forbidden_regions(
        self, regions: list[tuple[np.ndarray, float, str]]
    ) -> list[tuple[np.ndarray, float, str]]:
        """Physical object regions used by reset-time prop overlap checks."""
        out: list[tuple[np.ndarray, float, str]] = []
        for center, radius, label in regions:
            label = str(label)
            if label.startswith("robot:"):
                continue
            if label.startswith("slot:") or label == "target":
                continue
            out.append((center, radius, label))
        out.extend(self._neutral_robot_gripper_forbidden_regions())
        return out

    def _neutral_robot_gripper_forbidden_regions(self) -> list[tuple[np.ndarray, float, str]]:
        """Hand/finger/TCP regions to keep neutral cans out of at reset."""
        robot = getattr(self, "robot", None)
        out: list[tuple[np.ndarray, float, str]] = []
        radius = float(self.config.get(
            "neutral_can_gripper_clearance_radius",
            self.NEUTRAL_CAN_GRIPPER_CLEARANCE_RADIUS,
        ))
        seen = set()
        for arm_name in ("left_arm", "right_arm"):
            arm = getattr(robot, arm_name, None)
            ent = getattr(arm, "entity", None)
            if ent is None or id(ent) in seen:
                continue
            seen.add(id(ent))
            for i, link in enumerate(getattr(ent, "links", []) or []):
                name = str(getattr(link, "name", "")).lower()
                if not any(token in name for token in ("hand", "finger", "gripper")):
                    continue
                try:
                    pos = to_numpy(link.get_pos()).ravel()[:2]
                except Exception:
                    continue
                if np.all(np.isfinite(pos)):
                    out.append((
                        np.asarray(pos, dtype=np.float64),
                        radius,
                        f"gripper:{arm_name}:{i}:{name}",
                    ))
            try:
                tcp = np.asarray(arm.get_ee_pose(), dtype=np.float64).ravel()[:3]
                if np.all(np.isfinite(tcp[:2])):
                    out.append((
                        tcp[:2].astype(np.float64),
                        radius,
                        f"gripper:{arm_name}:tcp",
                    ))
            except Exception:
                pass
        return out

    def _neutral_avatar_forbidden_regions(self) -> list[tuple[np.ndarray, float, str]]:
        regions: list[tuple[np.ndarray, float, str]] = []

        def add_xy(xy, radius: float, label: str):
            try:
                arr = np.asarray(xy, dtype=np.float64).ravel()[:2]
            except Exception:
                return
            if arr.shape[0] == 2 and np.all(np.isfinite(arr)):
                regions.append((arr, float(radius), label))

        def add_entity(obj, radius: float, label: str):
            ent = getattr(obj, "entity", obj)
            try:
                add_xy(to_numpy(ent.get_pos()).ravel()[:2], radius, label)
            except Exception:
                try:
                    add_xy(np.asarray(obj.get_pose().p, dtype=np.float64)[:2], radius, label)
                except Exception:
                    pass

        def entity_xy_radius(obj, fallback: float) -> tuple[np.ndarray, float] | None:
            ent = getattr(obj, "entity", obj)
            try:
                aabb = to_numpy(ent.get_AABB()).reshape(-1, 3)
                lo = np.min(aabb, axis=0)
                hi = np.max(aabb, axis=0)
                center = 0.5 * (lo[:2] + hi[:2])
                half = 0.5 * (hi[:2] - lo[:2])
                radius = float(max(half[0], half[1]))
                if np.all(np.isfinite(center)) and np.isfinite(radius) and radius > 1e-5:
                    return center.astype(np.float64), radius
            except Exception:
                pass
            try:
                pos = to_numpy(ent.get_pos()).ravel()[:2]
                if np.all(np.isfinite(pos)):
                    return pos.astype(np.float64), float(fallback)
            except Exception:
                pass
            return None

        if hasattr(self, "blocks"):
            for name, ent in getattr(self, "blocks", {}).items():
                add_entity(ent, 0.055, f"block:{name}")
        if hasattr(self, "target_slots"):
            for name, pos in getattr(self, "target_slots", {}).items():
                add_xy(np.asarray(pos)[:2], 0.060, f"slot:{name}")
        for attr, radius in (
            ("objects", 0.060),
            ("breads", 0.070),
            ("shoes", 0.090),
            ("bowls", 0.100),
        ):
            for i, obj in enumerate(getattr(self, attr, []) or []):
                add_entity(obj, radius, f"{attr}:{i}")
        if hasattr(self, "basket"):
            add_entity(self.basket, 0.130, "basket")
        if hasattr(self, "shoebox"):
            add_entity(self.shoebox, 0.150, "shoebox")
        if hasattr(self, "food_actors"):
            for actor, spec in getattr(self, "food_actors", []) or []:
                add_entity(actor, 0.070, f"food:{getattr(spec, 'name', '?')}")
        if hasattr(self, "tray_actor"):
            add_entity(self.tray_actor, 0.140, "tray")
        if hasattr(self, "pan_actor"):
            add_entity(self.pan_actor, 0.130, "pan")
        if hasattr(self, "_cooktop_xy"):
            add_xy(self._cooktop_xy, 0.180, "cooktop")
        if hasattr(self, "_cabinet_pos"):
            add_xy(np.asarray(self._cabinet_pos)[:2], 0.140, "cabinet")
        if hasattr(self, "object"):
            add_entity(self.object, 0.070, "cabinet_object")
        if hasattr(self, "basket_poses"):
            for i, pose in enumerate(getattr(self, "basket_poses", []) or []):
                half_x = float(getattr(self, "_BASKET_HALF_X", 0.096))
                half_y = float(getattr(self, "_BASKET_HALF_Y", 0.096))
                add_xy(np.asarray(pose.p)[:2], max(half_x, half_y), f"categorize_basket:{i}")
        if hasattr(self, "_sort_targets"):
            for i, item in enumerate(getattr(self, "_sort_targets", []) or []):
                actor, obj_name, model_id, _basket_idx = item
                radius = None
                get_radius = getattr(self, "_get_object_cross_section_radius", None)
                if get_radius is not None:
                    try:
                        radius = float(get_radius(str(obj_name), int(model_id)))
                    except Exception:
                        radius = None
                region = entity_xy_radius(actor, radius if radius is not None else 0.040)
                if region is not None:
                    center, actual_radius = region
                    add_xy(center, min(actual_radius, radius) if radius is not None else actual_radius, f"categorize_obj:{i}")
                else:
                    add_entity(actor, radius if radius is not None else 0.040, f"categorize_obj:{i}")

        target = getattr(self, "target", None)
        pos = getattr(target, "position", None)
        if pos is not None:
            try:
                add_xy(pos() if callable(pos) else pos, 0.100, "target")
            except Exception:
                pass
        return regions

    def _neutral_avatar_forward_left_xy(self, body_pos: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
        rot = getattr(self, "_neutral_avatar_measure_body_rot", None)
        if rot is None:
            rot = getattr(self, "_neutral_avatar_body_rot", None)
        if rot is not None:
            forward = self._neutral_avatar_semantic_forward_xy(
                getattr(self, "_neutral_avatar_job", None),
                np.asarray(rot, dtype=np.float64),
            )
            norm = float(np.linalg.norm(forward))
            forward = forward / norm if norm > 1e-6 else np.array([0.0, -1.0], dtype=np.float64)
        elif body_pos is None:
            forward = np.array([0.0, -1.0], dtype=np.float64)
        else:
            forward = np.asarray(self.neutral_avatar_target[:2], dtype=np.float64) - np.asarray(
                body_pos[:2], dtype=np.float64
            )
            norm = float(np.linalg.norm(forward))
            if norm < 1e-6:
                forward = np.array([0.0, -1.0], dtype=np.float64)
            else:
                forward = forward / norm
        left = np.array([-forward[1], forward[0]], dtype=np.float64)
        return forward, left

    def _neutral_avatar_semantic_forward_xy(
        self, job: dict | None, rot: np.ndarray
    ) -> np.ndarray:
        forward = np.asarray(rot, dtype=np.float64)[:2, 0].copy()
        offset = self._neutral_avatar_motion_visual_yaw_offset(job)
        if abs(float(offset)) > 1e-9:
            c, s = float(np.cos(-offset)), float(np.sin(-offset))
            forward = np.array(
                [c * forward[0] - s * forward[1], s * forward[0] + c * forward[1]],
                dtype=np.float64,
            )
        return forward

    def _neutral_avatar_initial_object_offset(
        self, job: dict, obj_name: str, body_pos: np.ndarray | None
    ) -> np.ndarray:
        task_placement = self._neutral_avatar_task_placement(job)
        offset = np.zeros(3, dtype=np.float64)
        right_by_name = dict(task_placement.get("object_image_right_by_name", {}) or {})
        up_by_name = dict(task_placement.get("object_image_up_by_name", {}) or {})
        image_right = float(right_by_name.get(str(obj_name), 0.0))
        image_up = float(up_by_name.get(str(obj_name), 0.0))
        if image_right or image_up:
            forward, left = self._neutral_avatar_forward_left_xy(body_pos)
            offset[:2] += image_right * left - image_up * forward
        xyz_by_name = dict(task_placement.get("object_xyz_offset_by_name", {}) or {})
        if str(obj_name) in xyz_by_name:
            arr = np.asarray(xyz_by_name[str(obj_name)], dtype=np.float64).ravel()
            if arr.size != 3:
                raise ValueError(
                    f"object_xyz_offset_by_name[{obj_name!r}] must have 3 values"
                )
            offset += arr
        return offset

    def _neutral_avatar_clamp_table_xy(self, xy: np.ndarray, margin: float | None = None) -> np.ndarray:
        margin = self.NEUTRAL_PUT_OBJECTS_TABLE_MARGIN if margin is None else float(margin)
        x0, x1, y0, y1 = [float(v) for v in self._neutral_table_xy_bounds()]
        arr = np.asarray(xy, dtype=np.float64).ravel()[:2].copy()
        arr[0] = float(np.clip(arr[0], x0 + margin, x1 - margin))
        arr[1] = float(np.clip(arr[1], y0 + margin, y1 - margin))
        return arr

    def _place_neutral_avatar_props(
        self,
        job: dict,
        hold_by_frame: dict[int, np.ndarray],
        shift: np.ndarray,
        body_pos: np.ndarray | None = None,
    ) -> None:
        for event in job["events"]:
            if event["type"] not in ("attach", "pick"):
                continue
            obj_name = str(event["object"])
            obj = self.neutral_avatar_props[obj_name]
            pos = hold_by_frame[int(event["clip_frame"])] + shift
            pos = pos + self._neutral_avatar_initial_object_offset(job, obj_name, body_pos)
            if job["kind"] == "put_objects_in_bowl":
                pos[2] = self._neutral_avatar_prop_center_z(obj_name)
            elif obj_name == "sponge":
                pos[2] = self._neutral_avatar_prop_center_z("sponge")
            elif obj_name == "bottle" or str(obj_name).startswith("bottle_"):
                pos[:2] = self._neutral_avatar_clamp_table_xy(
                    pos[:2],
                    margin=self._neutral_avatar_table_margin_for_prop(job, obj_name),
                )
                pos[2] = self._neutral_avatar_prop_center_z(obj_name)
            obj.set_pos(pos.astype(float))
            obj.set_quat(self._neutral_avatar_prop_quat(obj_name))
            self._neutral_zero_entity_velocity(obj)
            if job["kind"] == "put_objects_in_bowl":
                # Record the resting spawn pose so the patched step-sim can pin
                # a not-yet-picked prop in place. Near-spherical props (apples)
                # otherwise roll off the grab spot during the long approach.
                cache = getattr(self, "_neutral_put_objects_spawn_pose", None)
                if cache is None:
                    cache = {}
                    self._neutral_put_objects_spawn_pose = cache
                cache[obj_name] = (
                    pos.astype(float).copy(),
                    self._neutral_avatar_prop_quat(obj_name).astype(float).copy(),
                )

        if job["kind"] == "put_objects_in_bowl" and hasattr(self, "neutral_avatar_bowl"):
            bowl_event_type = self._neutral_bowl_event_type(job)
            put_points = [
                hold_by_frame[int(e["clip_frame"])] + shift
                for e in job["events"]
                if e["type"] == bowl_event_type
            ]
            if put_points:
                xy = np.mean(np.asarray(put_points)[:, :2], axis=0)
                z = float(self.TABLE_TOP_Z + 0.020)
                self.neutral_avatar_bowl.set_pos(np.array([xy[0], xy[1], z], dtype=float))
                self.neutral_avatar_bowl.set_quat(self._neutral_avatar_prop_quat("plate"))
                self._neutral_zero_entity_velocity(self.neutral_avatar_bowl)

    def _neutral_reset_can_props_upright(self, job: dict) -> None:
        if job.get("kind") != "put_objects_in_bowl":
            return
        for obj_name, obj in getattr(self, "neutral_avatar_props", {}).items():
            if not str(obj_name).startswith("object_"):
                continue
            try:
                pos = to_numpy(obj.get_pos()).ravel()[:3].astype(np.float64)
            except Exception:
                continue
            pos[2] = self._neutral_avatar_prop_center_z(str(obj_name))
            obj.set_pos(pos.astype(float))
            obj.set_quat(self._neutral_avatar_prop_quat(str(obj_name)))
            self._neutral_zero_entity_velocity(obj)

    def _neutral_can_prop_is_upright(self, obj_name: str, obj) -> bool:
        try:
            quat = to_numpy(obj.get_quat()).ravel()[:4]
            pos = to_numpy(obj.get_pos()).ravel()[:3]
        except Exception:
            return False
        axis = t3d.quaternions.quat2mat(quat) @ np.array([0.0, 1.0, 0.0], dtype=np.float64)
        dot = float(axis[2])
        z_min = float(self.TABLE_TOP_Z - 0.050)
        return bool(dot >= float(self.NEUTRAL_CAN_UPRIGHT_DOT_MIN) and float(pos[2]) >= z_min)

    def _neutral_assert_can_upright_before_pick(self, obj_name: str, obj) -> None:
        if not (
            str(obj_name).startswith("object_")
            and (self._neutral_avatar_job or {}).get("kind") == "put_objects_in_bowl"
        ):
            return
        # The upright gate is can-specific (a can must stand on its base). Props
        # without a canonical upright axis (e.g. apples) are exempt.
        if self._neutral_put_objects_asset() != "071_can":
            return
        if self._neutral_can_prop_is_upright(str(obj_name), obj):
            return
        self.plan_success = False
        try:
            pos = to_numpy(obj.get_pos()).ravel()[:3].astype(float)
            quat = to_numpy(obj.get_quat()).ravel()[:4].astype(float)
        except Exception:
            pos = np.full(3, np.nan)
            quat = np.full(4, np.nan)
        msg = (
            f"neutral can {obj_name} is not upright before avatar pick; "
            f"pos={np.round(pos, 4).tolist()} quat={np.round(quat, 4).tolist()}"
        )
        self._neutral_debug(f"[neutral_avatar] REJECT: {msg}")
        raise RuntimeError(msg)

    def _neutral_avatar_prop_quat(self, obj_name: str) -> np.ndarray:
        upright = np.array([0.7071067811865476, 0.7071067811865475, 0.0, 0.0])
        if obj_name == "mouse":
            task_placement = self._neutral_avatar_task_placement()
            yaw_deg = float(
                dict(task_placement.get("object_yaw_deg_by_name", {}) or {}).get("mouse", -90.0)
            )
            if bool(task_placement.get("object_yaw_relative_to_avatar", False)):
                rot = np.asarray(
                    getattr(self, "_neutral_avatar_body_rot", self.avatar_init_rot),
                    dtype=np.float64,
                )
                forward = rot[:, 0].copy()
                yaw_deg += float(np.rad2deg(np.arctan2(forward[1], forward[0])))
            yaw = np.deg2rad(yaw_deg)
            yaw_quat = np.array([np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)])
            return t3d.quaternions.qmult(yaw_quat, upright)
        if str(obj_name).startswith("object_"):
            return upright
        if obj_name in ("bottle", "sponge", "apple", "plate") or str(obj_name).startswith("bottle_"):
            return upright
        if obj_name == "phone":
            # Orient the phone in the LEFT palm with the screen toward the head
            # (back facing out) — built from the live hand frame, so it looks
            # like a natural phone-call hold once attached.  Same math as
            # scripts/render_talking.py.
            import genesis.utils.geom as geom_utils
            try:
                robot = self.avatar.robot
                _, R_hand = robot._get_hand_frame(0)
                palm = np.asarray(robot.get_palm_center(0), dtype=np.float64)
                head = np.asarray(
                    robot.skin.get_global_translation("Head")[0], dtype=np.float64
                ).ravel()[:3]
            except Exception:
                return np.array([1.0, 0.0, 0.0, 0.0])
            e2 = np.asarray(R_hand[:, 1], dtype=np.float64)
            y = head - palm
            y = y / (np.linalg.norm(y) + 1e-8)
            z = e2 - np.dot(e2, y) * y
            if np.linalg.norm(z) < 1e-6:
                z = np.array([0.0, 0.0, 1.0]) - y[2] * y
            z = z / (np.linalg.norm(z) + 1e-8)
            x = np.cross(y, z); x /= np.linalg.norm(x) + 1e-8
            z = np.cross(x, y)
            R = np.column_stack([x, y, z])
            return np.asarray(geom_utils.R_to_quat(R)).astype(np.float64)
        return np.array([1.0, 0.0, 0.0, 0.0])

    def _patch_neutral_avatar_step_sim(self, job: dict, event_frame_offset: int = 0) -> None:
        slow = self._neutral_avatar_motion_slow()
        num_frames = max(1, int(job.get("clip", {}).get("num_frames", 1)))
        scaled_len = max(1, int(num_frames * slow))
        scaled_to_clip = np.round(
            np.linspace(0, num_frames - 1, scaled_len)
        ).astype(int)
        scaled_events = []
        for event in job["events"]:
            e = dict(event)
            clip_frame = int(np.clip(int(e["clip_frame"]), 0, num_frames - 1))
            matches = np.flatnonzero(scaled_to_clip >= clip_frame)
            e["scaled_frame"] = (
                int(matches[0]) if matches.size else scaled_len - 1
            ) + int(event_frame_offset)
            scaled_events.append(e)

        state = {
            "handled": set(),
            "pending_animation_start": False,
            "surface_props": {},
            "resting_surface_props": {},
            "attached_objects": set(),
            "final_pose_restored": False,
            "final_reset_done": False,
        }
        original_step_sim = self.step_sim
        self._neutral_original_step_sim = original_step_sim
        self._neutral_avatar_step_state = state

        def patched_step_sim():
            original_step_sim()
            self._neutral_laptop_hold_lid()
            if "idle_body_pos" in state:
                idle_body = np.asarray(state["idle_body_pos"], dtype=np.float64)
                motion_body = np.asarray(state["motion_body_pos"], dtype=np.float64)
                if (
                    not state.get("pending_animation_start")
                    and not state.get("return_started")
                    and state.get("return_step") is None
                    and self.avatar.robot.action_state == "transition"
                    and len(state["handled"]) == len(scaled_events)
                ):
                    state["return_started"] = True
                    state["return_step"] = 0
                if state.get("pending_animation_start"):
                    total = max(1, int(state.get("ease_total", 1)))
                    step = min(total, int(state.get("ease_step", 0)) + 1)
                    state["ease_step"] = step
                    t = float(step) / float(total)
                    self.avatar.robot.global_trans = (1.0 - t) * idle_body + t * motion_body
                    self.avatar.robot.update()
                elif state.get("return_started"):
                    total = max(1, int(state.get("return_total", 1)))
                    step = min(total, int(state.get("return_step", 0)) + 1)
                    state["return_step"] = step
                    t = float(step) / float(total)
                    self.avatar.robot.global_trans = (1.0 - t) * motion_body + t * idle_body
                    self.avatar.robot.update()
                    if step >= total:
                        state["return_started"] = False
            if state.get("pending_animation_start") and self.avatar.spare():
                state["pending_animation_start"] = False
                self.avatar.robot.global_trans = np.asarray(
                    state.get("motion_body_pos", self.avatar.robot.global_trans),
                    dtype=np.float64,
                )
                self.avatar.robot.update()
                self.avatar.play_animation(job["motion"])
                self._neutral_avatar_apply_motion_frame(job["motion"], 0)
            motion = self.avatar.motion_modules.get(job["motion"])
            at_frame = int(getattr(motion, "at_frame", -1)) if motion is not None else -1
            for idx, event in enumerate(scaled_events):
                if idx in state["handled"] or at_frame < int(event["scaled_frame"]):
                    continue
                hand_id = int(event["hand_id"])
                obj_name = str(event["object"])
                obj = self.neutral_avatar_props[obj_name]
                if event["type"] in ("attach", "pick"):
                    self._neutral_assert_can_upright_before_pick(obj_name, obj)
                    if obj_name in ("sponge", "mouse"):
                        state["resting_surface_props"].pop(obj_name, None)
                        state["surface_props"][obj_name] = (hand_id, event)
                        self._neutral_avatar_snap_prop_to_event_pose(
                            job, obj_name, hand_id, event
                        )
                    else:
                        # Objects that start in the hand (e.g. the phone, init=
                        # "hand") must be seated into the palm BEFORE attaching,
                        # otherwise attach_object_to_hand captures the prop's
                        # table load-pose and it floats away from the hand.
                        if obj_name == "phone" or str(event.get("init")) == "hand":
                            self._neutral_avatar_snap_prop_to_event_pose(
                                job, obj_name, hand_id, event
                            )
                        self.avatar.robot.attach_object_to_hand(hand_id, obj)
                        self.avatar.robot.update()
                    state["attached_objects"].add(obj_name)
                    self._neutral_debug(
                        f"[neutral_avatar] {event['type']} {event['object']} "
                        f"at clip={event['clip_frame']} scaled={event['scaled_frame']}"
                    )
                elif event["type"] in ("detach", "put"):
                    if obj_name in state["surface_props"]:
                        surface_hand_id, surface_event = state["surface_props"][obj_name]
                        hold_event = (
                            event
                            if "hold_offset" in event or "hold_offset_local" in event
                            else surface_event
                        )
                        pos = self._neutral_avatar_surface_prop_pos(
                            job, obj_name, int(surface_hand_id), hold_event
                        )
                        self._neutral_avatar_snap_prop_to_event_pose(
                            job, obj_name, int(surface_hand_id), hold_event
                        )
                        state["resting_surface_props"][obj_name] = pos[:2].copy()
                        state["surface_props"].pop(obj_name, None)
                    else:
                        self.avatar.robot.detach_object(hand_id)
                        # Rest the released prop just above the tabletop at its
                        # XY: bottles always; put-/take-objects props only in the
                        # reversed (take-out) clip, where the prop is set down on
                        # the bare table (forward put drops into the plate and
                        # must NOT be snapped to table height).
                        rest_on_table = (
                            obj_name == "bottle"
                            or str(obj_name).startswith("bottle_")
                            or (
                                job.get("kind") == "put_objects_in_bowl"
                                and bool((job.get("clip") or {}).get("reverse"))
                                and str(obj_name).startswith("object_")
                            )
                        )
                        if rest_on_table:
                            pos = to_numpy(obj.get_pos()).ravel()[:3].astype(np.float64)
                            pos[2] = self._neutral_avatar_prop_center_z(obj_name)
                            obj.set_pos(pos.astype(float))
                            obj.set_quat(self._neutral_avatar_prop_quat(obj_name))
                    self._neutral_zero_entity_velocity(obj)
                    self._neutral_debug(
                        f"[neutral_avatar] {event['type']} {event['object']} "
                        f"at clip={event['clip_frame']} scaled={event['scaled_frame']}"
                    )
                elif event["type"] == "switch":
                    from_hand_id = int(event.get("from_hand_id", 1 - hand_id))
                    self.avatar.robot.detach_object(from_hand_id)
                    self.avatar.robot.detach_object(hand_id)
                    self._neutral_zero_entity_velocity(obj)
                    self.avatar.robot.attach_object_to_hand(hand_id, obj)
                    self.avatar.robot.update()
                    self._neutral_debug(
                        f"[neutral_avatar] switch {event['object']} "
                        f"hand {from_hand_id}->{hand_id} "
                        f"at clip={event['clip_frame']} scaled={event['scaled_frame']}"
                    )
                state["handled"].add(idx)
            # Pin put-/take-objects props at their resting spawn pose until the
            # avatar picks them up. Near-spherical props (apples) would roll off
            # the grab spot during the long approach otherwise. Gated to non-can
            # assets so the stable-can forward/reverse motions are untouched.
            if (
                job.get("kind") == "put_objects_in_bowl"
                and self._neutral_put_objects_asset() != "071_can"
            ):
                spawn = getattr(self, "_neutral_put_objects_spawn_pose", None) or {}
                for obj_name, (pin_pos, pin_quat) in spawn.items():
                    if obj_name in state["attached_objects"]:
                        continue
                    obj = self.neutral_avatar_props.get(obj_name)
                    if obj is None:
                        continue
                    obj.set_pos(np.asarray(pin_pos, dtype=float))
                    obj.set_quat(np.asarray(pin_quat, dtype=float))
                    self._neutral_zero_entity_velocity(obj)
            for obj_name, (hand_id, event) in list(state["surface_props"].items()):
                obj = self.neutral_avatar_props[obj_name]
                pos = self._neutral_avatar_surface_prop_pos(
                    job, obj_name, int(hand_id), event
                )
                obj.set_pos(pos.astype(float))
                obj.set_quat(self._neutral_avatar_prop_quat(obj_name))
                self._neutral_zero_entity_velocity(obj)
            for obj_name, xy in list(state["resting_surface_props"].items()):
                obj = self.neutral_avatar_props[obj_name]
                obj.set_pos(np.array(
                    [
                        float(xy[0]),
                        float(xy[1]),
                        self._neutral_avatar_prop_center_z(obj_name),
                    ],
                    dtype=float,
                ))
                obj.set_quat(self._neutral_avatar_prop_quat(obj_name))
                self._neutral_zero_entity_velocity(obj)
            if (
                getattr(self, "_neutral_avatar_started", False)
                and not state["surface_props"]
                and self.avatar.spare()
            ):
                if (
                    bool(job.get("loop"))
                    and not state.get("return_started")
                    and not state.get("pending_animation_start")
                ):
                    # Rewind for another back-and-forth pass; re-arm events
                    # (the manually-attached writing pen stays attached).
                    loop_motion = self.avatar.motion_modules.get(job["motion"])
                    if loop_motion is not None and getattr(loop_motion, "data", None):
                        loop_motion.at_frame = 0
                        self.avatar.robot.action_state = job["motion"]
                        self.avatar.robot.action_status = ActionStatus.ONGOING
                        state["handled"] = set()
                        state["resting_surface_props"] = {}
                        self._neutral_avatar_apply_motion_frame(job["motion"], 0)
                elif not state.get("final_pose_restored", False):
                    self._neutral_avatar_apply_motion_frame(job["motion"], 0)
                    state["final_pose_restored"] = True

        self.step_sim = patched_step_sim

    def _install_neutral_loop_subclip(self, job: dict) -> None:
        """Wrap ``job["motion"]`` into a forward+reverse ping-pong clip.

        The avatar plays the clip forward then backward, and the patched
        step-sim rewinds it on completion, so the neutral motion runs
        back-and-forth for the whole episode instead of freezing after one
        pass.  Events keep their indices (all in the forward half) and
        re-fire each pass when the rewind clears the handled set.  Applied
        centrally after kind-specific subclip installation, so it composes
        with the mouse/sponge/laptop subclips and the writing/cans clips.
        """
        if job.get("loop"):
            return
        if (
            job.get("kind") == "put_objects_in_bowl"
            and not bool((job.get("clip") or {}).get("reverse"))
        ):
            return
        if not bool(self.config.get(
            "neutral_avatar_loop_motion", self.NEUTRAL_AVATAR_LOOP_MOTION
        )):
            return
        name = str(job["motion"])
        md = self.avatar.motion_data.get(name)
        if md is None:
            return
        n = int(md["trans"].shape[0])
        if n < 3:
            return
        # Forward 0..n-1 then reverse n-2..1: each fold frame (n-1 and 0)
        # appears once, so rewinding from the last element (frame 1) back to
        # frame 0 is a clean turnaround with no duplicated/popped frame.
        seq = np.asarray(
            list(range(n)) + list(range(n - 2, 0, -1)), dtype=int
        )
        dst = f"{name}_neutral_loop"
        self.avatar.motion_data[dst] = {k: v[seq].copy() for k, v in md.items()}
        job["motion"] = dst
        job["clip"] = dict(job.get("clip", {}))
        job["clip"]["num_frames"] = int(seq.shape[0])
        job["loop"] = True

    def _delay_neutral_avatar_motion_start(self, motion_name: str, delay_steps: int) -> int:
        if delay_steps <= 0 or self.avatar is None:
            return 0
        motion = self.avatar.motion_modules.get(motion_name)
        data = getattr(motion, "data", None) if motion is not None else None
        node_data = getattr(motion, "node_data", None) if motion is not None else None
        if motion is None or data is None or len(data) == 0:
            return 0
        delay_steps = int(delay_steps)
        motion.data = [data[0]] * delay_steps + list(data)
        if node_data is not None and len(node_data) > 0:
            motion.node_data = [node_data[0]] * delay_steps + list(node_data)
        return delay_steps

    def _avatar_obstacle_points(self) -> np.ndarray | None:
        if self.avatar_collider is None:
            return None
        pts = []
        for _name, pa, pb, r in self.avatar_collider.current_capsules():
            pa = np.asarray(pa, dtype=np.float64)
            pb = np.asarray(pb, dtype=np.float64)
            seg = pb - pa
            length = float(np.linalg.norm(seg))
            r_inflated = float(r) * self._AVATAR_INFLATE_FACTOR
            if length < 1e-6:
                pts.append(pa)
                continue
            axis = seg / length
            tmp = np.array([0.0, 0.0, 1.0]) if abs(axis[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
            u = np.cross(axis, tmp)
            u = u / (np.linalg.norm(u) + 1e-12)
            v = np.cross(axis, u)
            v = v / (np.linalg.norm(v) + 1e-12)
            n_axis = max(2, int(length / 0.04) + 1)
            for t in np.linspace(0.0, 1.0, n_axis):
                center = pa + t * seg
                pts.append(center)
                for theta in np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False):
                    pts.append(center + r_inflated * (np.cos(theta) * u + np.sin(theta) * v))
        return np.asarray(pts, dtype=np.float64) if pts else None

    def _neutral_object_entities(self) -> list[tuple[str, object]]:
        """All neutral-object scene entities (laptop / book / chair / cans …).

        These belong to the human, never to the robot task, so the robot must
        avoid them.  Deduped by entity identity; ``Actor`` wrappers are
        unwrapped to their ``entity``.
        """
        out: list[tuple[str, object]] = []
        seen: set[int] = set()

        def add(name: str, obj) -> None:
            if obj is None:
                return
            ent = getattr(obj, "entity", obj)
            if ent is None or id(ent) in seen:
                return
            seen.add(id(ent))
            out.append((str(name), ent))

        for name, obj in (getattr(self, "neutral_avatar_props", {}) or {}).items():
            add(name, obj)
        add("book", getattr(self, "neutral_avatar_book", None))
        add("chair", getattr(self, "neutral_avatar_chair", None))
        add("plate", getattr(self, "neutral_avatar_plate", None))
        return out

    def _neutral_object_obb(self, ent) -> tuple | None:
        """Tight oriented bounding box of an entity in its current pose.

        Returns ``(center_world, axes, half_extents)`` where ``axes`` columns
        are the object's local frame in world coords.  An OBB (not an AABB)
        matters here because the neutral objects are yawed on the table — the
        laptop sits at ~16–90° — so an axis-aligned box of a rotated laptop
        over-covers a large empty wedge and the robot forearm clips that empty
        corner (false contacts of ~mm depth).  Cached per entity per reset
        (the laptop/book/chair are static and the laptop carries ~55 collision
        meshes, so re-collecting verts every check would be wasteful).
        """
        cache = getattr(self, "_neutral_obb_cache", None)
        if cache is None:
            cache = {}
            self._neutral_obb_cache = cache
        key = id(ent)
        if key in cache:
            return cache[key]
        verts = []
        for link in (getattr(ent, "links", None) or []):
            for geom in (getattr(link, "_geoms", None) or []):
                try:
                    v = to_numpy(geom.get_verts()).reshape(-1, 3)
                except Exception:
                    continue
                if v.size:
                    verts.append(v)
        if not verts:
            cache[key] = None
            return None
        V = np.concatenate(verts, axis=0)
        try:
            R = t3d.quaternions.quat2mat(to_numpy(ent.get_quat()).ravel()[:4])
        except Exception:
            R = np.eye(3, dtype=np.float64)
        local = V @ R  # world -> local coords (R columns are the local axes)
        lo = local.min(axis=0)
        hi = local.max(axis=0)
        half = 0.5 * (hi - lo)
        if not np.all(np.isfinite(half)) or np.any(half <= 1e-5):
            cache[key] = None
            return None
        center_world = R @ (0.5 * (lo + hi))
        obb = (
            center_world.astype(np.float64),
            R.astype(np.float64),
            half.astype(np.float64),
        )
        cache[key] = obb
        return obb

    def _neutral_object_aabb_boxes(self, shrink: float = 0.0) -> list[tuple]:
        """Neutral objects as tight oriented analytic-checker boxes.

        Returns ``(name, center, axes, half_extents)``.  Names are prefixed
        ``neutral_obj:`` so the collision log distinguishes object hits from
        avatar-body hits.  ``shrink`` (m) is subtracted from each half-extent
        — used by the collision detector as a grazing tolerance so a sub-cm
        brush of the thin screen tip is not flagged as a touch.
        """
        boxes = []
        for name, ent in self._neutral_object_entities():
            obb = self._neutral_object_obb(ent)
            if obb is None:
                continue
            center, axes, half = obb
            if shrink:
                half = np.maximum(np.asarray(half, dtype=np.float64) - float(shrink), 1e-3)
            boxes.append((f"neutral_obj:{name}", center, axes, half))
        return boxes

    def _neutral_object_obstacle_points(self) -> np.ndarray | None:
        """Surface-shell point samples on the inflated neutral-object OBBs."""
        res = float(self._AVATAR_OBSTACLE_RES)
        inflate = float(getattr(self, "NEUTRAL_OBJECT_OBSTACLE_INFLATE", 0.01))
        parts = []
        for _name, center, axes, half in self._neutral_object_aabb_boxes():
            h = np.asarray(half, dtype=np.float64) + inflate
            ns = [max(2, int(2.0 * h[i] / res) + 1) for i in range(3)]
            gx, gy, gz = np.meshgrid(
                np.linspace(-h[0], h[0], ns[0]),
                np.linspace(-h[1], h[1], ns[1]),
                np.linspace(-h[2], h[2], ns[2]),
                indexing="ij",
            )
            local = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)
            # Keep the surface shell only (≥1 coord within res/2 of the face).
            on_shell = (np.abs(np.abs(local) - h).min(axis=1) < res * 0.5)
            if on_shell.any():
                local = local[on_shell]
            world = np.asarray(center, dtype=np.float64) + local @ np.asarray(
                axes, dtype=np.float64
            ).T
            parts.append(world)
        if not parts:
            return None
        return np.concatenate(parts, axis=0)

    def _neutral_object_obstacles_enabled(self) -> bool:
        return bool(self.config.get(
            "neutral_object_as_obstacle",
            getattr(self, "NEUTRAL_OBJECT_AS_OBSTACLE", True),
        ))

    def _install_neutral_object_collision_detection(self) -> None:
        """Feed neutral-object AABBs into the analytic collision checker.

        The checker already tests robot collision-geom sample points against
        ``avatar_collider.current_boxes()``; we wrap that method so the
        neutral objects' world AABBs are appended.  A robot↔object overlap
        then trips the same ``avatar_collided`` flag / collision log that
        gates neutral-task success.
        """
        if not self._neutral_object_obstacles_enabled():
            return
        collider = getattr(self, "avatar_collider", None)
        if collider is None or getattr(collider, "_neutral_obj_boxes_wrapped", False):
            return
        orig_current_boxes = collider.current_boxes
        task = self
        tol = float(self.config.get(
            "neutral_object_collision_tolerance",
            getattr(self, "NEUTRAL_OBJECT_COLLISION_TOLERANCE", 0.0),
        ))

        def current_boxes_with_objects():
            boxes = list(orig_current_boxes())
            try:
                boxes.extend(task._neutral_object_aabb_boxes(shrink=tol))
            except Exception:
                pass
            return boxes

        collider.current_boxes = current_boxes_with_objects
        collider._neutral_obj_boxes_wrapped = True

    def _refresh_avatar_planner_obstacles(self, arm_tag: str) -> None:
        parts = []
        if bool(getattr(self, "NEUTRAL_AVATAR_AS_PLANNER_OBSTACLE", True)):
            avatar_pts = self._avatar_obstacle_points()
            if avatar_pts is not None and avatar_pts.size:
                parts.append(avatar_pts)
        if self._neutral_object_obstacles_enabled():
            obj_pts = self._neutral_object_obstacle_points()
            if obj_pts is not None and obj_pts.size:
                parts.append(obj_pts)
        if not parts:
            return
        pts = np.concatenate(parts, axis=0)
        arm = self.robot.get_arm(arm_tag)
        try:
            arm.planner.update_obstacles(pts, resolution=self._AVATAR_OBSTACLE_RES)
        except Exception as e:
            self._neutral_debug(f"[neutral_avatar] obstacle update failed: {e}")

    def move_to_pose(self, pose, arm_tag: str):
        self._refresh_avatar_planner_obstacles(arm_tag)
        return super().move_to_pose(pose, arm_tag)

    def _move_seeded(self, link_pose7, arm_tag):
        self._refresh_avatar_planner_obstacles(arm_tag)
        return super()._move_seeded(link_pose7, arm_tag)

    def _move_screw(self, target_pos, arm_tag):
        self._refresh_avatar_planner_obstacles(arm_tag)
        return super()._move_screw(target_pos, arm_tag)

    def neutral_before_robot_rollout(self) -> None:
        """Hook for neutral tasks that need setup after avatar start."""
        return None

    def neutral_after_robot_rollout(self, rollout_ok: bool) -> None:
        """Hook for neutral tasks that need cleanup after the parent rollout."""
        return None

    def neutral_parent_success(self) -> bool:
        """Parent task success gate used by the default neutral success check."""
        return bool(super().check_success())

    def _neutral_object_collided(self) -> bool:
        """True if any logged robot collision was with a neutral object."""
        for record in getattr(self, "avatar_collision_log", []) or []:
            for pair in record.get("pairs") or []:
                if len(pair) >= 2 and str(pair[1]).startswith("neutral_obj:"):
                    return True
        return False

    def neutral_avatar_success_gate(self) -> bool:
        """Shared avatar-side success gate.

        ``avatar_collided`` is the sticky OR of robot↔avatar-body and
        robot↔neutral-object overlaps (the neutral objects are merged into the
        analytic checker), so a touch on either side fails the episode.
        """
        return not bool(getattr(self, "avatar_collided", False))

    def play_once(self) -> bool:
        self._start_neutral_avatar_table_work(add_start_delay=True)
        self.neutral_before_robot_rollout()
        ok = bool(super().play_once())
        self.neutral_after_robot_rollout(ok)
        return ok

    def check_success(self) -> bool:
        parent_ok = self.neutral_parent_success()
        no_avatar_collision = self.neutral_avatar_success_gate()
        self._neutral_debug(
            f"[neutral_avatar] parent_ok={parent_ok}, "
            f"no_avatar_collision={no_avatar_collision}"
        )
        return bool(parent_ok and no_avatar_collision)

    def evaluate(self) -> dict:
        metrics = super().evaluate()
        metrics["avatar_collided"] = bool(getattr(self, "avatar_collided", False))
        metrics["neutral_object_collided"] = bool(self._neutral_object_collided())
        metrics["neutral_eval_trigger_step"] = getattr(
            self, "_neutral_eval_trigger_step", None,
        )
        metrics["neutral_eval_policy_step_count"] = int(getattr(
            self, "_neutral_eval_policy_step_count", 0,
        ))
        metrics["neutral_avatar_started"] = bool(getattr(
            self, "_neutral_avatar_started", False,
        ))
        metrics["neutral_eval_avatar_failed"] = bool(getattr(
            self, "_neutral_eval_avatar_failed", False,
        ))
        if hasattr(self, "neutral_avatar_prop"):
            prop_pos = to_numpy(self.neutral_avatar_prop.get_pos()).ravel()[:3]
            metrics["neutral_avatar_prop_pos"] = prop_pos.tolist()
            metrics["neutral_avatar_prop_dist_xy"] = float(
                np.linalg.norm(prop_pos[:2] - self.neutral_avatar_target[:2])
            )
        return metrics
