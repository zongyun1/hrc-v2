"""Factory line feeding (handover) — Task 2 in ``docs/factory_tasks.md``.

Robot feeds parts to a worker busy at their station, at the factory work
cell (``FactorySceneMixin`` bench, top z = 0.765).

Scene: the worker KNEELS on the floor beside the bench, inspecting a tall
sealed package (the work prop).  One tool (a screwdriver, laid flat)
spawns on the robot's side of the bench; a shallow open tray ("box")
sits at the bench's front edge, inside both the robot's and the kneeling
worker's reach envelopes — the robot delivers the tool into it.  The
round is a single handover.  Success requires the tool's geometric
CENTRE inside the tray interior (a long bar may overhang the rim; its
centre may not).

Interaction (per part, serial):
  1. The worker reaches its near-table hand OUT to the table-edge region
     and holds it there — a "give it here" gesture.  The reach is a
     FABRIK shoulder->hand solve seeded from the clip's last kneeling
     frame (body stays kneeling; only the arm moves).  The hand target is
     at the very front edge so the palm rests at the lip without
     penetrating the table top.
  2. The robot picks the part and places it into the region beside the
     reached hand, using the proven ``PlaceBurgerFries``-lineage pick
     pipeline (ready-pose branch reset, seeded mplib-IK guarded Cartesian
     legs, topple fail-fast).  The part stays in the region.
  3. The robot parks clear and the worker pulls its hand back, then
     resumes the kneeling work clip.

Two work-clip variants (``WORK_POOL``; pick via ``config["work_motion"]``
or ``randomize_work``): ``Kneeling_inspecting`` (default, kneels parallel
to the bench inspecting the package side) and ``Working_on_device``.

The worker's kneeling stance is calibrated per episode: an offscreen
dry-run of the clip measures the working palm anchor relative to the
avatar root (cached at class level per clip), so the clip's hands work at
the package.

Success (geometric): every part delivered into the table-edge region, and
no robot-avatar collision.

The avatar capsule pcd (including the reached arm) is pushed into the
mplib planner obstacles before every robot plan.
"""

from __future__ import annotations

import json

import numpy as np

from ..base_task import BaseTask
from ..grasp import tcp_to_link_pose
from ..manipulation import TopDownPickPlaceMixin
from ..scenes.factory import FactorySceneMixin
from ..task_bases.place_burger_fries import PlaceBurgerFries
from ..utils import ASSETS_PATH, Pose, create_primitive, load_object

# upright = mesh-Y -> world-Z (these assets are y-up)
_Q_UPRIGHT = np.array([0.70710678, 0.70710678, 0.0, 0.0], dtype=np.float64)

# Avatar root rotations (world-frame columns of the local basis).
_ROT_FACE_POS_Y = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64)
_ROT_FACE_NEG_X = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=np.float64)


def _read_model_data(name: str, mid: int) -> tuple[float, np.ndarray]:
    """(scale, extents) from model_data{mid}.json."""
    p = ASSETS_PATH / "objects" / name / f"model_data{mid}.json"
    with open(p) as f:
        d = json.load(f)
    raw = d.get("scale", 1.0)
    scale = float(raw[0] if isinstance(raw, (list, tuple)) else raw)
    return scale, np.asarray(d["extents"], dtype=np.float64)


class FactoryLineFeeding(FactorySceneMixin, TopDownPickPlaceMixin, BaseTask):
    """Robot delivers parts to a handover pad; kneeling worker takes each."""

    INSTRUCTION = (
        "pick up each part and place it on the handover pad "
        "for the worker"
    )

    use_avatar = True

    # ---- Work-loop variants -------------------------------------------------
    # Each entry fully describes one kneeling work loop:
    #   clip          — motion key in generated_motions.pkl
    #   rot           — avatar root rotation (world)
    #   hand_id       — FABRIK take hand (near-table hand)
    #   anchor_frame  — clip source frame where the working palm is read
    #   station       — container on the floor:
    #                   kind "open"  → open-top box (parts dropped inside)
    #                   kind "solid" → sealed package (parts placed on top)
    #                   (center_xy, inner_half, wall_h, wall_t, floor_t)
    #   anchor_dxy    — where the work anchor should land, relative to the
    #                   station center (xy) — e.g. the near wall face of
    #                   the package for the side-inspect clip
    #   drop_dxy      — take detach point, relative to the station center
    #   body_x_bounds / body_y_max — stance clamps (keep the kneeling body
    #                   clear of the bench front edge at y = -0.475)
    WORK_POOL = (
        {
            "clip": "Kneeling_inspecting",
            "rot": _ROT_FACE_NEG_X,
            "hand_id": 1,
            "anchor_frame": 75,
            # Tall sealed package the worker inspects (the work prop).  The
            # clip's working palm measures at z ~0.78 world, so an 0.80 m
            # package puts the inspecting hands on its side wall near the
            # top.  Parts the robot hands over end up in the table-edge
            # REGION (class-level), not on/around the package.
            "station": {
                "kind": "solid",
                "center_xy": (-0.80, -0.70),
                "inner_half": (0.23, 0.21),
                "wall_h": 0.80,
                "wall_t": 0.018,
                "floor_t": 0.012,
            },
            # Hands work at the package's near (+x) wall face.
            "anchor_dxy": (0.25, 0.0),
            "body_x_bounds": (-0.50, 0.25),
            "body_y_max": -0.60,
        },
        {
            "clip": "Working_on_device",
            "rot": _ROT_FACE_POS_Y,
            "hand_id": 1,
            "anchor_frame": 97,
            "station": {
                "center_xy": (-0.38, -0.64),
                "inner_half": (0.19, 0.16),
                "wall_h": 0.16,
                "wall_t": 0.015,
                "floor_t": 0.012,
            },
            "anchor_dxy": (0.0, 0.0),
            "body_x_bounds": (-0.75, 0.10),
            "body_y_max": -0.72,
        },
    )
    STATION_COLOR = (0.72, 0.56, 0.38)

    # ---- Handover region (table front edge) ------------------------------
    # The worker reaches a hand out to HAND_REACH; the robot places each
    # part into REGION_XY on the bench just inside that hand.  REGION_XY
    # sits in the robot's 0.40-0.60 m annulus from its base at (0.60,-0.30)
    # AND within the kneeling worker's ~0.6 m reach.  HAND_REACH is at the
    # very front edge (y just past -0.475) so the palm rests at the lip
    # without penetrating the table top (a target further inside the table
    # makes the hand clip through the surface).
    # Delivery target = a flat green region on the bench (a marked zone,
    # not a box).  Kept ~20 cm in x from the worker's presented hand so the
    # gripper never nears the fingers (redesign v2: a 0.1 cm graze failed a
    # run), and in the robot's 0.40-0.55 m annulus from its base at
    # (0.60,-0.30).
    #
    # A long tool, carried hanging, lays down on release and its body
    # settles offset from the held handle in a repeatable "lean" direction.
    # ``DROP_COMP`` shifts the robot's drop point so the tool's settled
    # CENTRE — not the held handle — lands at the region centre (measured
    # from a smoke and applied; the handle is dropped up-lean of centre).
    # Region centre.  x is the default (overridden per-episode along the
    # edge); y is pinned at runtime so the region's front edge overlaps the
    # table edge (see _region_edge_y).
    HANDOVER_XY = (0.15, -0.355)
    # Region sized to contain a long tool's lay-down + cylinder roll (the
    # screwdriver rolls up to ~10 cm perpendicular to its lying axis, in a
    # direction that varies with the spawn yaw).  half_y sets how far the
    # region reaches back from the table edge.
    REGION_PAD_HALF = (0.13, 0.12)            # success footprint (the pad)
    REGION_PAD_COLOR = (0.20, 0.62, 0.30)     # safety green
    # Drop at the region centre, NO compensation: the cylindrical tool
    # rolls/topples chaotically on landing, so trying to compensate the
    # ~4 cm lean shifted it 9 cm the other way (a 4 cm y-comp jumped x by
    # 0.09).  Uncompensated + LOW release lands the centre ~4 cm off the
    # aim (seed 0: (0.15,-0.341), d_xy (0.0,0.039)) — solidly inside the
    # generous region; that beats any aim correction.
    DROP_COMP = (0.0, 0.0)
    REGION_SLOT_DY = 0.0                       # single round: no slotting
    HAND_REACH_XY = (-0.05, -0.52)            # avatar palm target (at the edge)
    HAND_REACH_DZ = 0.07                      # palm hover above the table top

    # ---- Per-episode randomization (seeded by reset; on by default) -------
    RANDOMIZE = True
    # The green region sits at the bench FRONT EDGE — its front (−y) edge
    # overlaps the table edge (y = −FACTORY_TABLE_HALF[1]) — and only its
    # position ALONG the edge (x) is randomized.  The region centre y is
    # derived as table_edge + REGION_PAD_HALF[y] so the front edges coincide.
    # x range stays in the robot's 0.40-0.55 m drop annulus.  The worker's
    # presented hand tracks HAND_REGION_DX to the avatar side (−x) so the
    # ~20 cm gripper/finger separation holds at every region.
    REGION_X_RANGE = (0.06, 0.19)
    HAND_REGION_DX = 0.20
    HAND_REACH_Y = -0.52
    # The worker reaches for help at a random MID-LATE frame of the work
    # clip (it works for a randomized stretch, then asks).
    WORK_REACH_FRAC = (0.5, 0.9)

    # ---- Part (one tool per round) ----------------------------------------
    # v1 smoke + the kit/qc factory smokes showed top-down picks fail when
    # the target is < ~0.35 m horizontal from the bench-mounted base at
    # (+0.60, -0.30) (seeded IK lands 0.14-0.48 m off); spawn in the proven
    # 0.40-0.55 m annulus, staying clear of the <0.35 m near-base dead zone.
    #
    # One TOOL per round (the worker is being fed a part).  The screwdriver
    # is laid FLAT (long mesh-y axis -> world x via spawn_quat Rz(-90)) so
    # its 3 cm cylindrical HANDLE can be grasped top-down like a horizontal
    # can.  The handle sits at mesh -0.065 m (one end); ``grasp_off`` (mesh
    # frame, rotated into world per the live pose) targets it instead of
    # the geometric centre, which lands on the thin metal shaft.  Default
    # top-down close grips across the handle's ~3 cm diameter (no yaw).
    #
    # Per-part dict fields: asset, mid, x/y spawn ranges, yaw_range (deg,
    # random spin about world-z), friction, close, spawn_quat (lay-flat),
    # rest_half_h, grasp_off (mesh-frame offset centre->handle), grasp_axis
    # (mesh-frame finger-close direction across the handle).  The pick reads
    # the tool's live pose, so it grasps the handle and aligns the gripper
    # yaw to whatever position/orientation the tool settled in.  A small
    # drop gap keeps the settle near the laid-flat pose (clean first grasp).
    _Q_FLAT_X = (0.70710678, 0.0, 0.0, -0.70710678)   # mesh +y -> world +x
    # TOOL_POOL: one tool is chosen per episode (random when randomizing,
    # or forced via config["tool"]=<asset>).  Each is laid flat and grasped
    # at its handle (grasp_off) across its narrow axis (grasp_axis); the
    # pick reads the live pose so position+yaw are handled.
    TOOL_POOL = (
        {
            "asset": "032_screwdriver", "mid": 0,
            # Moderate spawn band + ±12° yaw.  x capped at 0.21 because the
            # cylinder rolls +x on settle (mixed ep6: spawned ≤0.22 rolled
            # to a 0.24 handle at the far-reach edge → descent IK flip).
            "x": (0.17, 0.21), "y": (-0.03, 0.03), "yaw_range": (-12.0, 12.0),
            "friction": 4.0, "close": 0.22,
            "spawn_quat": _Q_FLAT_X, "rest_half_h": 0.016, "spawn_gap": 0.002,
            "grasp_off": (0.0, -0.065, 0.0), "grasp_axis": (1.0, 0.0, 0.0),
        },
        {
            # Hammer laid flat (long mesh-y -> world x).  Handle toward -y;
            # grasp it across its narrow (mesh-x) face.  Flat head means it
            # does NOT roll on release (tighter placement than the rod).
            "asset": "020_hammer", "mid": 0,
            "x": (0.17, 0.22), "y": (-0.03, 0.03), "yaw_range": (-12.0, 12.0),
            "friction": 4.0, "close": 0.18,
            # rest_half_h MUST match the measured flat half-height (center
            # settled at z≈0.799 → 0.034 above the 0.765 top).  An earlier
            # 0.030 spawned the hammer 4 mm INTO the table, and the contact
            # kick spun the heavy offset head to wild yaws (mixed2 ep3/4:
            # grasp_yaw −88°/42°, grip slipped).  Spawn AT rest, tiny gap.
            "spawn_quat": _Q_FLAT_X, "rest_half_h": 0.034, "spawn_gap": 0.003,
            "grasp_off": (0.0, -0.045, 0.0), "grasp_axis": (1.0, 0.0, 0.0),
        },
    )
    PARTS = (TOOL_POOL[0],)       # default (overridden per-episode by select)
    ROBOT_BELOW_CENTER = 0.0

    # ---- Worker pacing -----------------------------------------------------
    AVATAR_MOTION_SLOW = 6.0
    AVATAR_MEASURE_POS = np.array([6.0, 6.0, -0.18])
    AVATAR_FALLBACK_POS = np.array([-0.38, -0.92, -0.18])
    FABRIK_REACH_WARN_M = 0.66

    # Reach-out gesture pacing (FABRIK shoulder->hand IK).  Slow + a gentle
    # arc reads as a deliberate "present a hand at the table edge" motion;
    # the hand holds there while the robot delivers, then retracts.
    REACH_FRAMES = 150
    REACH_ARC = 0.10
    RETRACT_FRAMES = 130
    WORK_REP_BETWEEN = 1          # work-clip reps played between handovers

    # ---- Robot pacing -------------------------------------------------------
    INITIAL_SETTLE_STEPS = 80
    DELIVER_SETTLE_STEPS = 120        # let the released part settle in the region
    FINAL_SETTLE_STEPS = 150
    # Canonical ready pose (Franka "ready": q1=0 faces the yawed-forward
    # bench, q7=0.785 mid-range).  Every pick starts here via a pure
    # joint-space move: the IK branch the approach picks is inherited by
    # the whole delivery, and v21 vs v22 showed it's a physics-jitter
    # coin flip otherwise (v22 entered the carry with q7 at its 2.8973
    # limit and the continuous carry solution ceased to exist).
    READY_QPOS = (0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785)
    WRIST_RECENTER_ABOVE = 2.0        # |q7| beyond this after lift → recenter

    # Pick-pipeline knobs (PlaceBurgerFries lineage; structure follows
    # factory_kit_packing's shared-bench fixes: mplib-checked approach with
    # avatar-snapshot retries, Cartesian verticals, xy-error abort gate).
    FINGER_KP = 9000.0
    FINGER_KV = 250.0
    FINGER_FORCE_LIMIT = 80.0
    TRANSPORT_Z_ABOVE_TABLE = 0.30
    DROP_Z_ABOVE_TABLE = 0.04        # release LOW so the bar lays down
    #                                  rather than toppling from height
    TABLE_FALL_Z_MARGIN = 0.05
    GRASP_TABLE_Z_MARGIN = 0.003
    PRE_GRASP_HEIGHT = 0.15
    GRASP_SETTLE_STEPS = 100
    GRASP_ALIGNMENT_XY_TOL = 0.015
    # v2 smoke: closing 3.5 cm off-center toppled the 7.1 cm can, and a
    # lying can slips out of the top-down close.  Never close unless the
    # EE is properly centered; contact-free alignment refinements generally
    # land at 1-2 mm.
    GRASP_ABORT_XY = 0.018
    GRASP_ALIGNMENT_SETTLE_STEPS = 80
    CLOSE_SETTLE_STEPS = 200
    POST_LIFT_SETTLE_STEPS = 60
    LIFT_DZ_MIN = 0.05
    LIFT_FAIL_RELEASE_SETTLE_STEPS = 50
    APPROACH_PLAN_RETRIES = 4
    APPROACH_PLAN_WAIT_STEPS = 150
    DESCENT_CARTESIAN_STEPS = 30
    DESCENT_SIM_PER_STEP = 20
    LIFT_CARTESIAN_STEPS = 30
    LIFT_SIM_PER_STEP = 20
    # 60 waypoints x 30 sim steps: the v3/v10 smokes slipped the can
    # during the lateral transit at 40x20 / 60x20; slower transport keeps
    # grip forces stable.
    TRANSIT_CARTESIAN_STEPS = 60
    TRANSIT_SIM_PER_STEP = 30
    POST_TRANSIT_SETTLE_STEPS = 50
    TRANSIT_SLIP_XY_TOL = 0.10
    DROP_SETTLE_STEPS = 40
    RELEASE_SETTLE_STEPS = 200

    _AVATAR_OBSTACLE_RES = 0.04
    _AVATAR_INFLATE_FACTOR = 1.5

    # Mount the Franka yawed +150° (vs the mixin's identity quat, which
    # faces it at world +x — the bench's right EDGE, 25 cm away).  The
    # work area spans azimuth 120° (parts) to 180° (handover pad) from
    # the base; 180° is outside the Franka joint-1 limit (±166°), which
    # forced contorted near-limit poses — the root cause of the lateral
    # IK drift, branch flips and plan_screw failures seen in smokes
    # v1-v8.  Yawed +150°, the same targets sit at -30°..+30°.
    FACTORY_ROBOT_KWARGS = {
        "pos": [0.60, -0.30, 0.765],
        "quat": [0.2588190, 0.0, 0.0, 0.9659258],
    }

    # Camera: pull back / down a touch from the mixin default so the
    # kneeling worker + ground container stay in frame next to the bench.
    recording_camera_pos = [-1.75, -2.05, 1.60]
    recording_camera_lookat = [0.05, -0.15, 0.70]

    # Reuse the burger-fries finger PD boost (plain unbound-method reuse —
    # no shared-code change).  The avatar obstacle pcd is inlined below:
    # borrowing it from FactoryInspectPack broke when that task's owner
    # iterated its signature mid-flight (v7 smoke, BOX_XY AttributeError).
    _boost_finger_pd = PlaceBurgerFries._boost_finger_pd

    _work_anchor_cache: dict = {}

    def __init__(self, config: dict = None):
        cfg = dict(config or {})
        cfg.setdefault("robot_type", "franka")
        cfg.setdefault("robot_single_arm", True)
        cfg.setdefault("use_avatar_collider", True)
        cfg.setdefault("track_avatar_collision", True)
        super().__init__(cfg)
        self.part_actors = []   # [(actor, asset, mid, close, grasp_off, half_h)]
        self._reached = []      # per-part: worker reached a hand out for it
        self._delivered = []    # per-part: robot placed it in the region

    # ------------------------------------------------------------------
    # Work-motion selection (before scene build — the station container
    # geometry depends on it)
    # ------------------------------------------------------------------
    def _select_work_entry(self, seed: int) -> None:
        forced = self.config.get("work_motion")
        if forced:
            matches = [e for e in self.WORK_POOL if e["clip"] == forced]
            if not matches:
                raise ValueError(
                    f"config['work_motion']={forced!r} not in pool: "
                    f"{[e['clip'] for e in self.WORK_POOL]}"
                )
            self._work = matches[0]
        elif self.config.get("randomize_work", False) and len(self.WORK_POOL) > 1:
            # Deterministic per seed: reset() reseeds with the same value.
            rng = np.random.RandomState(seed)
            self._work = self.WORK_POOL[int(rng.randint(len(self.WORK_POOL)))]
        else:
            self._work = self.WORK_POOL[0]
        print(f"[feed] work motion: {self._work['clip']}", flush=True)

    # ------------------------------------------------------------------
    # Scene
    # ------------------------------------------------------------------
    def reset(self, seed: int = 0):
        self._restore_step_sim()
        self._select_work_entry(seed)
        obs = super().reset(seed=seed)
        if self.avatar is not None:
            self.avatar.frame_ratio = float(self.AVATAR_MOTION_SLOW)
            self._calibrate_avatar_stance()
        return obs

    def _build_station(self):
        st = self._work["station"]
        bx, by = st["center_xy"]
        ihx, ihy = st["inner_half"]
        t, wh, ft = st["wall_t"], st["wall_h"], st["floor_t"]
        if st.get("kind", "open") == "solid":
            # Sealed package: the prop the worker inspects.  One static box.
            h = ft + wh
            self.station_parts = [create_primitive(
                self.scene, "box", Pose(p=[bx, by, h / 2]),
                size={"half_size": (ihx + t, ihy + t, h / 2)},
                color=self.STATION_COLOR, is_static=True,
            )]
            return
        ohx, ohy = ihx + t, ihy + t
        self.station_parts = [create_primitive(
            self.scene, "box", Pose(p=[bx, by, ft / 2]),
            size={"half_size": (ohx, ohy, ft / 2)},
            color=self.STATION_COLOR, is_static=True,
        )]
        wz = ft + wh / 2
        for sx in (-1, +1):
            self.station_parts.append(create_primitive(
                self.scene, "box", Pose(p=[bx + sx * (ihx + t / 2), by, wz]),
                size={"half_size": (t / 2, ohy, wh / 2)},
                color=self.STATION_COLOR, is_static=True,
            ))
        for sy in (-1, +1):
            self.station_parts.append(create_primitive(
                self.scene, "box", Pose(p=[bx, by + sy * (ihy + t / 2), wz]),
                size={"half_size": (ohx, t / 2, wh / 2)},
                color=self.STATION_COLOR, is_static=True,
            ))

    def _build_handover_pad(self):
        """Flat green region marker on the bench (render-only)."""
        hx, hy = self.REGION_PAD_HALF
        px, py = self.HANDOVER_XY
        self.handover_pad = create_primitive(
            self.scene, "box",
            Pose(p=[px, py, self.TABLE_TOP_Z + 0.0015]),
            size={"half_size": (hx, hy, 0.0012)},
            color=self.REGION_PAD_COLOR, is_static=True, collision=False,
        )

    def _hand_reach_target(self) -> np.ndarray:
        """World palm target for the reach-out gesture: at the table front
        edge, hovering just above the lip so the hand never penetrates the
        table top."""
        return np.array(
            [self.HAND_REACH_XY[0], self.HAND_REACH_XY[1],
             self.TABLE_TOP_Z + self.HAND_REACH_DZ],
            dtype=np.float64,
        )

    def _region_edge_y(self) -> float:
        """Region centre y so its front (−y) edge sits on the table edge."""
        table_edge_y = -float(self.FACTORY_TABLE_HALF[1]) + float(self.table_offset[1])
        return table_edge_y + float(self.REGION_PAD_HALF[1])

    def _randomize_region(self) -> None:
        """Per-episode: slide the edge region's position ALONG the table
        edge (x only; y pinned so its front edge overlaps the table edge),
        and put the worker's presented hand a fixed offset to the avatar
        side of it.  np.random is already seeded by reset()."""
        ry = self._region_edge_y()
        if not (self.RANDOMIZE and self.config.get("randomize", True)):
            self.HANDOVER_XY = (self.HANDOVER_XY[0], ry)
            self.HAND_REACH_XY = (self.HANDOVER_XY[0] - self.HAND_REGION_DX,
                                  self.HAND_REACH_Y)
            return
        rx = float(np.random.uniform(*self.REGION_X_RANGE))
        self.HANDOVER_XY = (rx, ry)
        self.HAND_REACH_XY = (rx - self.HAND_REGION_DX, self.HAND_REACH_Y)
        print(f"[feed] randomized region={np.round(self.HANDOVER_XY,3).tolist()} "
              f"hand={np.round(self.HAND_REACH_XY,3).tolist()}", flush=True)

    def _select_tool(self, randomize: bool) -> dict:
        """Pick one tool from TOOL_POOL: forced via config["tool"]=<asset>,
        else random when randomizing, else the first."""
        forced = self.config.get("tool")
        if forced:
            for s in self.TOOL_POOL:
                if s["asset"] == forced:
                    print(f"[feed] tool forced: {forced}", flush=True)
                    return s
            raise ValueError(f"config['tool']={forced!r} not in TOOL_POOL "
                             f"{[s['asset'] for s in self.TOOL_POOL]}")
        if randomize and len(self.TOOL_POOL) > 1:
            s = self.TOOL_POOL[int(np.random.randint(len(self.TOOL_POOL)))]
            print(f"[feed] tool randomized: {s['asset']}", flush=True)
            return s
        return self.TOOL_POOL[0]

    def load_actors(self):
        self.build_factory()
        self._randomize_region()
        self._build_station()
        self._build_handover_pad()
        self.part_actors = []
        randomize = self.RANDOMIZE and self.config.get("randomize", True)
        for spec in (self._select_tool(randomize),):
            asset, mid = spec["asset"], int(spec["mid"])
            half_h = float(spec["rest_half_h"])
            q = np.asarray(spec.get("spawn_quat", _Q_UPRIGHT), dtype=np.float64)
            grasp_off = np.asarray(spec.get("grasp_off", (0.0, 0.0, 0.0)),
                                   dtype=np.float64)
            grasp_axis = spec.get("grasp_axis")
            grasp_axis = (np.asarray(grasp_axis, dtype=np.float64)
                          if grasp_axis is not None else None)
            (xlo, xhi), (ylo, yhi) = spec["x"], spec["y"]
            gap = float(spec.get("spawn_gap", 0.01))
            x = float(np.random.uniform(xlo, xhi))
            y = float(np.random.uniform(ylo, yhi))
            # Random spin about world-z (composed onto the lay-flat quat).
            ylo_d, yhi_d = spec.get("yaw_range", (0.0, 0.0))
            if randomize and (yhi_d - ylo_d) > 1e-6:
                import transforms3d as _t3d
                psi = np.deg2rad(float(np.random.uniform(ylo_d, yhi_d)))
                q_yaw = np.array([np.cos(psi / 2), 0.0, 0.0, np.sin(psi / 2)])
                q = _t3d.quaternions.qmult(q_yaw, q)
            actor = load_object(
                self.scene,
                Pose([x, y, self.TABLE_TOP_Z + half_h + gap], q),
                asset, model_id=mid,
                convex=True, is_static=False, friction=float(spec["friction"]),
            )
            self.part_actors.append(
                (actor, asset, mid, float(spec["close"]), grasp_off,
                 grasp_axis, half_h))
        self._reached = [False] * len(self.part_actors)
        self._delivered = [False] * len(self.part_actors)

    @staticmethod
    def _entity(actor):
        return getattr(actor, "entity", actor)

    # ------------------------------------------------------------------
    # Avatar: kneeling stance calibrated so the clip works at the station
    # ------------------------------------------------------------------
    def _measure_work_anchor(self) -> np.ndarray:
        """Working palm anchor (palm center relative to avatar root) at the
        clip's mid-work frame, measured by an offscreen dry-run at
        frame_ratio=1.  Cached at class level per (clip, hand)."""
        clip = self._work["clip"]
        hand_id = int(self._work["hand_id"])
        key = (clip, hand_id)
        cached = FactoryLineFeeding._work_anchor_cache.get(key)
        if cached is not None:
            rel, lo, hi = cached
            self._hand_sweep_rel = (lo.copy(), hi.copy())
            return rel.copy()

        base_pos = np.asarray(self.AVATAR_MEASURE_POS, dtype=np.float64)
        base_rot = np.asarray(self._work["rot"], dtype=np.float64)
        self.avatar.reset(base_pos.copy(), base_rot.copy())
        old_ratio = float(self.avatar.frame_ratio)
        self.avatar.frame_ratio = 1.0
        self.avatar.play_animation(clip)
        rel = None
        step = 0
        target = int(self._work["anchor_frame"])
        # Track BOTH palms across the whole clip — the sweep envelope
        # tells us where parts must NOT rest (kinematic hands tunnel
        # resting parts; v12 smoke).
        sweep_lo = np.full(3, np.inf)
        sweep_hi = np.full(3, -np.inf)
        while not self.avatar.spare():
            self.scene.step()
            self.avatar.step()
            for h in (0, 1):
                p = np.asarray(
                    self.avatar.robot.get_palm_center(h), dtype=np.float64,
                ) - base_pos
                sweep_lo = np.minimum(sweep_lo, p)
                sweep_hi = np.maximum(sweep_hi, p)
            if step == target:
                palm = np.asarray(
                    self.avatar.robot.get_palm_center(hand_id),
                    dtype=np.float64,
                )
                rel = palm - base_pos
            step += 1
        self._hand_sweep_rel = (sweep_lo.copy(), sweep_hi.copy())
        self.avatar.frame_ratio = old_ratio
        if rel is None:
            raise RuntimeError(
                f"[feed] work-anchor dry-run missed frame {target} "
                f"(stopped at step {step})"
            )
        FactoryLineFeeding._work_anchor_cache[key] = (
            rel.copy(), sweep_lo.copy(), sweep_hi.copy(),
        )
        print(f"[feed] work palm anchor {clip} (rel to root): "
              f"{rel.round(3).tolist()}", flush=True)
        print(f"[feed] hand sweep envelope (rel): "
              f"lo={sweep_lo.round(3).tolist()} hi={sweep_hi.round(3).tolist()}",
              flush=True)
        return rel

    def _calibrate_avatar_stance(self) -> None:
        """Root the kneeling worker so the clip's hands work at the package,
        then sanity-log the FABRIK reach to the table-edge hand target."""
        try:
            rel = self._measure_work_anchor()
        except Exception as exc:
            print(f"[feed] WARNING work-anchor calibration failed ({exc}); "
                  f"using fallback stance", flush=True)
            rel = None

        body = np.asarray(self.AVATAR_FALLBACK_POS, dtype=np.float64).copy()
        if rel is not None:
            bx, by = self._work["station"]["center_xy"]
            adx, ady = self._work["anchor_dxy"]
            body[0] = (bx + adx) - rel[0]
            body[1] = (by + ady) - rel[1]
            raw = body.copy()
            body[0] = float(np.clip(body[0], *self._work["body_x_bounds"]))
            body[1] = min(float(body[1]), float(self._work["body_y_max"]))
            if np.linalg.norm(body[:2] - raw[:2]) > 1e-6:
                print(f"[feed] WARNING stance clamped "
                      f"{raw[:2].round(3).tolist()} -> "
                      f"{body[:2].round(3).tolist()}", flush=True)

        rot = np.asarray(self._work["rot"], dtype=np.float64)
        self.avatar.reset(body.copy(), rot.copy())
        for _ in range(30):
            self.step_sim()
        self.avatar_collided = False
        self.avatar_collision_log = []
        self._avatar_body = body

        hand = self._hand_reach_target()
        # Shoulder is ~1.2 m above the kneeling root; estimate reach so a
        # stance that puts the hand target out of range warns up front.
        reach = float(np.linalg.norm(hand - (body + np.array([0.0, 0.0, 1.15]))))
        print(f"[feed] stance body={body.round(3).tolist()} "
              f"hand_reach={reach:.2f} m -> target {hand.round(3).tolist()}",
              flush=True)
        if reach > float(self.FABRIK_REACH_WARN_M):
            print(f"[feed] WARNING reach target may exceed kneeling FABRIK "
                  f"reach ({reach:.2f} m)", flush=True)

    # ------------------------------------------------------------------
    # Worker: reach a hand out to the table-edge region, hold, retract.
    # The FABRIK reach seeds from the clip's last (kneeling) frame, so the
    # body stays kneeling and only the shoulder->hand chain moves.
    # ------------------------------------------------------------------
    def _play_work_reps(self, reps: int = 1) -> None:
        """Play the kneeling work clip `reps` times (restarting the cached
        module to avoid the per-call rebuild that OOM'd earlier loops)."""
        for _ in range(max(0, int(reps))):
            mod = self.avatar.motion_modules.get(self._work["clip"])
            if mod is not None and getattr(mod, "data", None):
                mod.start()
            else:
                self.avatar.play_animation(self._work["clip"])
            while not self.avatar.spare():
                self.step_sim()

    def _play_work_until_reach(self) -> None:
        """Work for a RANDOM mid-late stretch of the clip, then stop — the
        worker reaches for help at a randomized point of its task rather
        than always after a full rep.  Implemented as a [0, N] subclip whose
        end N is drawn from the mid-late band, so the clip finishes (spare)
        at that frame and the reach starts from that working pose."""
        clip = self._work["clip"]
        md = self.avatar.motion_data.get(clip)
        if md is None or not (self.RANDOMIZE and self.config.get("randomize", True)):
            self._play_work_reps(1)
            return
        n = int(md["trans"].shape[0])
        lo = int(self.WORK_REACH_FRAC[0] * n)
        hi = int(self.WORK_REACH_FRAC[1] * n)
        end = int(np.random.randint(lo, max(lo + 1, hi + 1)))
        dst = f"{clip}_work_0_{end}"
        if dst not in self.avatar.motion_data:
            sl = slice(0, end + 1)
            self.avatar.motion_data[dst] = {k: v[sl].copy() for k, v in md.items()}
        self.avatar.play_animation(dst)
        while not self.avatar.spare():
            self.step_sim()
        print(f"[feed] worker worked to frame {end}/{n} "
              f"({end / n:.0%}), now reaching for help", flush=True)

    def _reach_out(self, hand_id: int) -> None:
        """Reach the near-table hand to the edge target and hold it there."""
        self._rest_palm = np.asarray(
            self.avatar.robot.get_palm_center(hand_id), dtype=np.float64,
        ).ravel()[:3].copy()
        target = self._hand_reach_target()
        self.avatar.pick_and_place(
            pick_pos=target, place_pos=target,
            hand_id=hand_id,
            approach_frames=self.REACH_FRAMES,
            transport_frames=1, retract_frames=0,
            approach_arc=self.REACH_ARC,
        )
        while not self.avatar.spare():
            self.step_sim()
        palm = np.asarray(
            self.avatar.robot.get_palm_center(hand_id), dtype=np.float64,
        ).ravel()[:3]
        print(f"[feed] reach-out hand at {palm.round(3).tolist()} "
              f"(target {target.round(3).tolist()})", flush=True)

    def _retract_hand(self, hand_id: int) -> None:
        """Return the hand from the edge to its kneeling rest position."""
        cur = np.asarray(
            self.avatar.robot.get_palm_center(hand_id), dtype=np.float64,
        ).ravel()[:3]
        rest = getattr(self, "_rest_palm", cur)
        self.avatar.pick_and_place(
            pick_pos=cur, place_pos=rest,
            hand_id=hand_id,
            approach_frames=1, transport_frames=self.RETRACT_FRAMES,
            retract_frames=0, transport_arc=self.REACH_ARC,
        )
        while not self.avatar.spare():
            self.step_sim()

    def _restore_step_sim(self) -> None:
        # Back-compat no-op: the serial flow no longer patches step_sim.
        if hasattr(self, "_feed_original_step_sim"):
            self.step_sim = self._feed_original_step_sim
            del self._feed_original_step_sim

    # ------------------------------------------------------------------
    # Robot: avatar capsules as planner obstacles
    # ------------------------------------------------------------------
    def _avatar_obstacle_points(self):
        """Sampled point cloud over the avatar's current collision capsules
        (inflated), for the mplib planner obstacle set."""
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
            tmp = (
                np.array([0.0, 0.0, 1.0])
                if abs(axis[2]) < 0.9
                else np.array([1.0, 0.0, 0.0])
            )
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

    def _refresh_planner_obstacles(self, arm_tag: str) -> None:
        pts = self._avatar_obstacle_points()
        if pts is None:
            return
        arm = self.robot.get_arm(arm_tag)
        try:
            arm.planner.update_obstacles(pts, resolution=self._AVATAR_OBSTACLE_RES)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Robot: deliver one part to the pad, then park clear of the corridor
    # ------------------------------------------------------------------
    def _feed_move_planned(self, link_pose7, arm_tag: str):
        """Like the mixin's ``_move_seeded`` but the goal qpos comes from
        seeded mplib IK (return_closest + shortest-path wrap) instead of
        Genesis IK — which lands in twisted branches (q1 83° for a -29°
        target, v23 smoke) even when seeded from the canonical ready
        pose, dooming the subsequent carry.  Returns the plan result or
        None."""
        from ..robot.franka_robot import to_numpy as _to_numpy
        from ..planning.base import PlanResult
        arm = self.robot.get_arm(arm_tag)
        current = np.asarray(arm.get_arm_qpos(), dtype=np.float64).ravel()
        try:
            res = arm.planner.solve_ik(
                current.copy(), np.asarray(link_pose7).ravel()[:7],
                num_waypoints=2, log=False,
            )
        except Exception as e:
            print(f"[feed] approach IK exc: {e}", flush=True)
            return None
        if not getattr(res, "success", False) or res.position is None:
            return None
        goal = np.asarray(res.position, dtype=np.float64)[-1].ravel()[:arm.n_arm]
        planner = arm.planner
        try:
            result = planner._mplib.plan_qpos(
                goal_qposes=[planner._pad_qpos(goal)],
                current_qpos=planner._pad_qpos(current),
                time_step=1 / 250,
                planning_time=getattr(planner, "_timeout", None) or 4,
                rrt_range=0.3,
                verbose=False,
            )
        except Exception as e:
            print(f"[feed] approach plan_qpos exc: {e}", flush=True)
            return None
        if not result or result.get("status") != "Success":
            return None
        pos = result["position"]
        vel = result.get("velocity", np.zeros_like(pos))
        self.execute_plan(PlanResult(True, pos, vel), arm_tag)
        arm._cached_target = np.asarray(
            _to_numpy(arm.entity.get_qpos()), dtype=float).ravel().copy()
        ee_xy = np.array(arm.get_ee_pose()[:2], dtype=float)
        xy_err = float(np.linalg.norm(
            ee_xy - np.asarray(link_pose7, dtype=float).ravel()[:2]))
        if xy_err > 0.05:
            print(f"[feed] approach post-exec xy_err={xy_err:.3f}", flush=True)
            return None
        return result

    def _move_seeded_retry(self, link_pose7, arm_tag: str):
        """Planner-checked move with avatar-snapshot retries: if every
        mplib route is blocked by the worker's current pose, let the sim
        (and the work loop) advance, refresh the pcd, and try again."""
        for attempt in range(self.APPROACH_PLAN_RETRIES):
            self._refresh_planner_obstacles(arm_tag)
            res = self._feed_move_planned(link_pose7, arm_tag)
            if res is not None:
                return res
            print(f"[feed] approach plan blocked "
                  f"(attempt {attempt + 1}/{self.APPROACH_PLAN_RETRIES}); "
                  f"waiting {self.APPROACH_PLAN_WAIT_STEPS} steps", flush=True)
            for _ in range(self.APPROACH_PLAN_WAIT_STEPS):
                self.step_sim()
        return None

    def _fk_link_xy(self, arm, arm_qpos):
        """World xy of the planner move-group link at ``arm_qpos`` via
        mplib's pinocchio FK — stateless (no Genesis state touched, no PD
        desync).  None = FK unavailable.  The pinocchio model and base
        rotation are cached: this runs per-waypoint (hundreds of calls
        per leg) and re-fetching the model each call leaked the v17
        smoke to OOM."""
        try:
            cache = getattr(self, "_fk_cache", None)
            if cache is None or cache[0] is not arm.planner:
                import transforms3d as t3d
                planner = arm.planner
                pm = planner._mplib.robot.get_pinocchio_model()
                base = np.asarray(
                    planner._base_pose, dtype=np.float64).ravel()
                base_R = t3d.quaternions.quat2mat(base[3:7])
                cache = (planner, pm,
                         int(planner._mplib.move_group_link_id),
                         base[:3].copy(), base_R)
                self._fk_cache = cache
            planner, pm, link_id, base_p, base_R = cache
            padded = planner._pad_qpos(
                np.asarray(arm_qpos, dtype=np.float64).ravel())
            pm.compute_forward_kinematics(padded)
            raw = pm.get_link_pose(link_id)
            p = np.asarray(getattr(raw, "p", raw), dtype=np.float64).ravel()[:3]
            world_p = base_p + base_R @ p
            return world_p[:2]
        except Exception:
            return None

    def _guarded_vertical(self, start_pos, end_pos, arm_tag,
                          n_steps, sim_per_step,
                          xy_tol=0.02, max_consecutive_bad=3,
                          q_step_limit=0.30,
                          yaw_follow_base=False) -> bool:
        """Straight Cartesian move that FK-verifies every waypoint's IK
        solution BEFORE commanding it.  Genesis IK has no convergence
        gate and intermittently returns solutions several cm off the
        line (v5/v13 smokes: 6.7 / 7.9 cm — enough to rake the part with
        the open fingers).  Additionally gates the JOINT step between
        consecutive commanded waypoints: top-down poses are wrist-
        symmetric, so an FK-clean solution can sit on a flipped branch
        whose joint path whips the arm (v16 smoke flung the held can
        1.5 m mid-transit).  Bad solutions are skipped (seed stays at the
        last good waypoint); more than ``max_consecutive_bad`` in a row
        aborts (returns False) so the caller can retreat untouched.

        ``yaw_follow_base``: rotate the (top-down) TCP yaw with the
        base-to-EE azimuth along the path.  A fixed world yaw across a
        wide lateral sweep forces the wrist to counter-rotate into its
        q7 limit, after which EVERY IK solve sits on the flipped branch
        (v17/18 smokes: 28-59 consecutive flipped waypoints).  Use for
        lateral transits; carried-part yaw doesn't matter there.
        """
        from ..robot.franka_robot import to_numpy as _to_numpy, _get_dof_idx
        arm = self.robot.get_arm(arm_tag)
        base_target = getattr(arm, "_cached_target", None)
        if base_target is None:
            base_target = _to_numpy(arm.entity.get_qpos())
        base_target = np.array(base_target, dtype=float).copy()
        measured = np.asarray(
            _to_numpy(arm.entity.get_qpos()), dtype=float).ravel()
        seed = base_target.copy()
        for j in arm.arm_joints:
            if j is not None:
                di = _get_dof_idx(j)
                if di is not None:
                    seed[di] = float(measured[di])
        arm._cached_target = seed
        start = np.asarray(start_pos, dtype=float)
        end = np.asarray(end_pos, dtype=float)
        # Joint-step gate baseline: the measured arm joints at entry.
        prev_q = np.array([
            float(measured[_get_dof_idx(j)])
            for j in arm.arm_joints if j is not None
        ], dtype=float)
        import transforms3d as _t3d
        # Preserve the arm's CURRENT TCP yaw: every leg starts from the
        # yaw the previous motion left (forcing world-yaw-0 made the
        # first waypoint demand a 1.4+ rad wrist rotation after any
        # approach/recenter — v23/24 smokes — and the leg died fighting
        # it).  Carried-part yaw is irrelevant; descents just keep yaw.
        ee_now = np.asarray(arm.get_ee_pose(), dtype=float).ravel()
        R_now = _t3d.quaternions.quat2mat(ee_now[3:7])
        Rz_now = self._R_DOWN.T @ R_now
        psi0 = float(np.arctan2(Rz_now[1, 0], Rz_now[0, 0]))
        base_xy = None
        yaw0 = 0.0
        if yaw_follow_base:
            base_xy = np.asarray(arm.origin_pose.p, dtype=float)[:2]
            v0 = start[:2] - base_xy
            yaw0 = float(np.arctan2(v0[1], v0[0]))
        tcp_offset = arm.tcp_offset
        bad = 0
        skipped = 0
        flip_diagnosed = False
        for i in range(1, int(n_steps) + 1):
            t = i / float(n_steps)
            pos = start * (1.0 - t) + end * t
            dyaw = psi0
            if yaw_follow_base:
                v = pos[:2] - base_xy
                d = float(np.arctan2(v[1], v[0])) - yaw0
                dyaw = psi0 + float(np.arctan2(np.sin(d), np.cos(d)))
            c, s = np.cos(dyaw), np.sin(dyaw)
            Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
            tcp = Pose(np.asarray(pos, dtype=float),
                       _t3d.quaternions.mat2quat(self._R_DOWN @ Rz))
            # mplib (pinocchio) IK seeded from the previously COMMANDED
            # waypoint, with return_closest + shortest-path wrapping —
            # Genesis IK kept returning ±2pi-class branch jumps mid-leg
            # (v16-20 smokes) which this solver structurally avoids.
            arm_qpos = None
            try:
                link_pose7 = tcp_to_link_pose(tcp, tcp_offset).to_pose7()
                res = arm.planner.solve_ik(
                    prev_q.copy(), np.asarray(link_pose7).ravel()[:7],
                    num_waypoints=2, log=False,
                )
                if getattr(res, "success", False) and res.position is not None:
                    arm_qpos = np.asarray(
                        res.position, dtype=float)[-1].ravel()[:len(prev_q)]
            except Exception:
                arm_qpos = None
            if arm_qpos is None:
                # Fall back to the seeded Genesis IK for this waypoint.
                arm_qpos = self._solve_ik(tcp, arm_tag)
            ok = arm_qpos is not None
            if ok:
                fk_xy = self._fk_link_xy(arm, arm_qpos)
                if fk_xy is not None and float(
                        np.linalg.norm(fk_xy - pos[:2])) > xy_tol:
                    ok = False
            if ok:
                dq_vec = (np.asarray(arm_qpos, dtype=float).ravel()[:len(prev_q)]
                          - prev_q)
                dq = float(np.max(np.abs(dq_vec)))
                if dq > float(q_step_limit):
                    print(f"[feed] guarded move: branch flip at waypoint "
                          f"{i}/{n_steps} (max dq={dq:.2f} rad) — skipped",
                          flush=True)
                    if not flip_diagnosed:
                        flip_diagnosed = True
                        print(f"[feed]   flip dq per joint: "
                              f"{np.round(dq_vec, 2).tolist()}", flush=True)
                        print(f"[feed]   seed q: "
                              f"{np.round(prev_q, 2).tolist()}", flush=True)
                    ok = False
            if not ok:
                bad += 1
                skipped += 1
                if bad > max_consecutive_bad:
                    print(f"[feed] guarded vertical aborted at waypoint "
                          f"{i}/{n_steps} (consecutive bad IK)", flush=True)
                    return False
                continue
            bad = 0
            prev_q = np.asarray(arm_qpos, dtype=float).ravel()[:len(prev_q)].copy()
            qpos_full = base_target.copy()
            for j_idx, j in enumerate(arm.arm_joints):
                if j is not None:
                    di = _get_dof_idx(j)
                    if di is not None:
                        qpos_full[di] = float(arm_qpos[j_idx])
            arm.entity.control_dofs_position(qpos_full)
            arm._cached_target = qpos_full.copy()
            for _ in range(int(sim_per_step)):
                self.step_sim()
        if skipped:
            print(f"[feed] guarded vertical skipped {skipped}/{n_steps} "
                  f"bad waypoints", flush=True)
        return True

    def _arm_dof_indices(self, arm):
        from ..robot.franka_robot import _get_dof_idx
        return [
            _get_dof_idx(j) for j in arm.arm_joints
            if j is not None and _get_dof_idx(j) is not None
        ]

    def _goto_ready(self, arm_tag: str, n_steps: int = 100,
                    sim_per_step: int = 5) -> None:
        """Joint-space move to the canonical ready pose (no IK — resets
        the arm's branch deterministically).  Gripper targets preserved.
        Only call with an EMPTY gripper above the bench."""
        from ..robot.franka_robot import to_numpy as _to_numpy
        arm = self.robot.get_arm(arm_tag)
        base_target = getattr(arm, "_cached_target", None)
        if base_target is None:
            base_target = _to_numpy(arm.entity.get_qpos())
        base_target = np.array(base_target, dtype=float).copy()
        measured = np.asarray(
            _to_numpy(arm.entity.get_qpos()), dtype=float).ravel()
        dofs = self._arm_dof_indices(arm)
        q0 = np.array([measured[d] for d in dofs], dtype=float)
        q1 = np.asarray(self.READY_QPOS, dtype=float)[:len(dofs)]
        for k in range(1, int(n_steps) + 1):
            q = q0 + (q1 - q0) * (k / float(n_steps))
            full = base_target.copy()
            for d, v in zip(dofs, q):
                full[d] = float(v)
            arm.entity.control_dofs_position(full)
            arm._cached_target = full.copy()
            for _ in range(int(sim_per_step)):
                self.step_sim()

    def _recenter_wrist(self, arm_tag: str) -> None:
        """If q7 sits near its ±2.8973 limit after the lift, rotate it
        toward mid-range.  Pure q7 motion is a TCP null rotation — the
        held part spins in place at transport height, nothing else
        moves — and it restores the headroom the carry needs."""
        from ..robot.franka_robot import to_numpy as _to_numpy
        arm = self.robot.get_arm(arm_tag)
        measured = np.asarray(
            _to_numpy(arm.entity.get_qpos()), dtype=float).ravel()
        dofs = self._arm_dof_indices(arm)
        q7_dof = dofs[-1]
        cur = float(measured[q7_dof])
        if abs(cur) <= float(self.WRIST_RECENTER_ABOVE):
            return
        target = 0.785 if cur > 0 else -0.785
        print(f"[feed] wrist near limit (q7={cur:.2f}); recentering to "
              f"{target:.2f}", flush=True)
        base_target = getattr(arm, "_cached_target", None)
        if base_target is None:
            base_target = _to_numpy(arm.entity.get_qpos())
        base_target = np.array(base_target, dtype=float).copy()
        n_steps = 80
        for k in range(1, n_steps + 1):
            full = base_target.copy()
            full[q7_dof] = cur + (target - cur) * (k / float(n_steps))
            arm.entity.control_dofs_position(full)
            arm._cached_target = full.copy()
            for _ in range(5):
                self.step_sim()

    def _flip_wrist_for_grasp(self, arm_tag: str, thresh: float = 2.2) -> None:
        """Before the grasp descent: if the approach left q7 near its
        ±2.8973 limit, rotate it by π toward mid-range.  A parallel gripper
        closes the SAME line at q7 and q7±π, so this preserves the grasp
        orientation exactly while moving the wrist off the limit — which is
        what makes the descent IK flip and abort (edge-region seed-4
        hammer: q7=2.65 → descent aborted).  Done with the gripper open and
        nothing held, 15 cm above the tool, so it's a harmless spin."""
        from ..robot.franka_robot import to_numpy as _to_numpy
        arm = self.robot.get_arm(arm_tag)
        measured = np.asarray(
            _to_numpy(arm.entity.get_qpos()), dtype=float).ravel()
        dofs = self._arm_dof_indices(arm)
        q7_dof = dofs[-1]
        cur = float(measured[q7_dof])
        if abs(cur) <= float(thresh):
            return
        target = float(np.clip(cur - np.sign(cur) * np.pi, -2.85, 2.85))
        print(f"[feed] wrist near limit at grasp (q7={cur:.2f}); flipping "
              f"180° to {target:.2f} (same grasp)", flush=True)
        base_target = getattr(arm, "_cached_target", None)
        if base_target is None:
            base_target = _to_numpy(arm.entity.get_qpos())
        base_target = np.array(base_target, dtype=float).copy()
        n_steps = 70
        for k in range(1, n_steps + 1):
            full = base_target.copy()
            full[q7_dof] = cur + (target - cur) * (k / float(n_steps))
            arm.entity.control_dofs_position(full)
            arm._cached_target = full.copy()
            for _ in range(5):
                self.step_sim()

    def _region_drop_xy(self, i: int) -> tuple[float, float]:
        """Robot drop point for part i = region centre + DROP_COMP (so the
        tool's settled CENTRE lands at the region centre despite the
        handle-held bar laying down off-centre) + per-part Y slot."""
        rx, ry = self.HANDOVER_XY
        cx, cy = self.DROP_COMP
        n = max(1, len(self.part_actors))
        slot = (float(i) - (n - 1) / 2.0) * self.REGION_SLOT_DY
        return (rx + cx, ry + cy + slot)

    def _pick_point(self, actor, asset: str, mid: int, grasp_off) -> np.ndarray:
        """World grasp target: the object's geometric centre offset by
        ``grasp_off`` (mesh frame) rotated into world by the live pose —
        e.g. the screwdriver's handle instead of its thin-shaft centre."""
        center = np.asarray(
            self._get_object_world_center(actor, asset, mid), dtype=np.float64)
        off = np.asarray(grasp_off, dtype=np.float64) if grasp_off is not None \
            else np.zeros(3)
        if not np.any(off):
            return center
        import transforms3d as _t3d
        q = np.asarray(actor.get_pose().q, dtype=np.float64)
        return center + _t3d.quaternions.quat2mat(q) @ off

    def _rotated_top_down_tcp(self, pos, yaw_deg: float) -> Pose:
        """Top-down gripper rotated about the approach (TCP-z) axis by
        ``yaw_deg`` — rotates the finger close direction to match a tool's
        in-plane orientation."""
        if abs(float(yaw_deg)) < 1e-6:
            return self._top_down_tcp(pos)
        import transforms3d as _t3d
        yaw = np.deg2rad(float(yaw_deg))
        cz, sz = np.cos(yaw), np.sin(yaw)
        Rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
        q = _t3d.quaternions.mat2quat(self._R_DOWN @ Rz)
        return Pose(np.asarray(pos, dtype=float), q)

    def _grasp_yaw_deg(self, actor, grasp_axis) -> float:
        """Top-down gripper yaw so the fingers close along the tool's narrow
        grasp axis (``grasp_axis`` is a mesh-frame unit vector; default
        mesh-x = across the handle).  Lets the pick handle a tool spawned at
        any yaw.  Wrapped to [-90,90] (close is 180-symmetric) to keep the
        wrist away from its limit."""
        if grasp_axis is None:
            return 0.0
        import transforms3d as _t3d
        q = np.asarray(actor.get_pose().q, dtype=np.float64)
        ca = _t3d.quaternions.quat2mat(q) @ np.asarray(grasp_axis, dtype=float)
        yaw = np.degrees(np.arctan2(ca[1], ca[0])) - 90.0
        while yaw > 90.0:
            yaw -= 180.0
        while yaw < -90.0:
            yaw += 180.0
        return float(yaw)

    def _feed_pick_once(self, actor, asset: str, arm_tag: str,
                        close_value: float, drop_xy=None,
                        grasp_off=None, grasp_axis=None) -> bool:
        """One delivery attempt — PlaceBurgerFries flow with the
        factory_kit_packing shared-bench fixes: the big approach goes
        through mplib with avatar-snapshot retries; the short verticals
        (descend / lift / transit / drop) use ``_move_cartesian`` so a
        momentarily-blocking avatar pcd can't divert them into wild
        ``plan_pose`` fallback swings.  An xy-error gate aborts before
        closing on air (which only shoves the part around the bench)."""
        from ..robot.franka_robot import to_numpy as _to_numpy
        arm = self.robot.get_arm(arm_tag)
        tcp_offset = arm.tcp_offset
        # Reset the IK branch deterministically before every attempt.
        self._goto_ready(arm_tag)
        arm._cached_target = np.asarray(
            _to_numpy(arm.entity.get_qpos()), dtype=float,
        ).ravel().copy()

        transport_z = self.TABLE_TOP_Z + self.TRANSPORT_Z_ABOVE_TABLE
        drop_z = self.TABLE_TOP_Z + self.DROP_Z_ABOVE_TABLE
        drop_xy = tuple(self.HANDOVER_XY) if drop_xy is None else tuple(drop_xy)
        # Gripper yaw so the fingers close across the tool's handle whatever
        # orientation it spawned in.  The above-approach establishes it; the
        # guarded descents preserve the current ee yaw, so it carries down.
        gyaw = self._grasp_yaw_deg(actor, grasp_axis)
        gtcp = lambda p: self._rotated_top_down_tcp(p, gyaw)
        print(f"[feed] {asset} drop aimed at {np.round(drop_xy,3).tolist()} "
              f"z={drop_z:.3f} grasp_yaw={gyaw:.0f} (region {self.HANDOVER_XY})",
              flush=True)

        obj_center = self._pick_point(actor, asset, 0, grasp_off)
        grasp_pos = obj_center.copy()
        grasp_pos[2] -= self.ROBOT_BELOW_CENTER
        grasp_pos[2] = max(
            grasp_pos[2], self.TABLE_TOP_Z + self.GRASP_TABLE_Z_MARGIN,
        )

        self.open_gripper(arm_tag)

        above_pos = np.array([obj_center[0], obj_center[1], transport_z])
        above_link = tcp_to_link_pose(gtcp(above_pos), tcp_offset)
        if self._move_seeded_retry(above_link.to_pose7(), arm_tag) is None:
            return False
        # NOTE: a pre-descent 180° wrist flip was tried to clear the rare
        # grasp-descent IK abort, but it put the arm on a branch that
        # flung the tool during the CARRY (5/10 vs 9/10) — reverted.

        pre_pos = obj_center.copy()
        pre_pos[2] += self.PRE_GRASP_HEIGHT
        if not self._guarded_vertical(
                above_pos, pre_pos, arm_tag,
                n_steps=self.DESCENT_CARTESIAN_STEPS,
                sim_per_step=self.DESCENT_SIM_PER_STEP):
            return False

        # Center the EE over the part AT PRE-GRASP HEIGHT.  Lateral
        # corrections at grasp height (v4 smoke) shove the part with the
        # open fingers and topple it; up here the fingers are 15 cm clear.
        obj_now = self._pick_point(actor, asset, 0, grasp_off)
        pre_pos[0], pre_pos[1] = obj_now[0], obj_now[1]
        for _ in range(2):
            ee_pos = np.array(arm.get_ee_pose()[:3], dtype=float)
            pre_err = float(np.linalg.norm(ee_pos[:2] - pre_pos[:2]))
            if pre_err <= self.GRASP_ALIGNMENT_XY_TOL:
                break
            print(f"[feed] {asset} pre-grasp off by {pre_err:.3f}; "
                  f"re-centering at pre height", flush=True)
            pre_link = tcp_to_link_pose(gtcp(pre_pos), tcp_offset)
            self._move_seeded_retry(pre_link.to_pose7(), arm_tag)
            for _ in range(self.GRASP_ALIGNMENT_SETTLE_STEPS):
                self.step_sim()
        ee_pos = np.array(arm.get_ee_pose()[:3], dtype=float)
        pre_err = float(np.linalg.norm(ee_pos[:2] - pre_pos[:2]))
        if pre_err > self.GRASP_ABORT_XY:
            print(f"[feed] {asset} cannot center at pre height "
                  f"(err={pre_err:.3f}); aborting attempt untouched", flush=True)
            self._move_cartesian(
                ee_pos, np.array([ee_pos[0], ee_pos[1], transport_z]), arm_tag,
                n_steps=self.DESCENT_CARTESIAN_STEPS,
                sim_per_step=self.DESCENT_SIM_PER_STEP,
            )
            return False

        # Staged vertical descent.  A single pre->grasp Cartesian descent
        # can drift laterally on the way down (Genesis IK has no
        # convergence gate; v5 smoke drifted 6.7 cm and raked the can),
        # so descend in two hops with a drift check at mid height —
        # fingertips still clear of the part there, so a re-center or
        # abort stays contact-free.
        obj_now = self._pick_point(actor, asset, 0, grasp_off)
        grasp_pos[0] = obj_now[0]
        grasp_pos[1] = obj_now[1]
        mid_pos = grasp_pos.copy()
        mid_pos[2] = grasp_pos[2] + 0.09

        if not self._guarded_vertical(
                np.array([ee_pos[0], ee_pos[1], ee_pos[2]]), mid_pos, arm_tag,
                n_steps=self.DESCENT_CARTESIAN_STEPS,
                sim_per_step=self.DESCENT_SIM_PER_STEP):
            ee_pos = np.array(arm.get_ee_pose()[:3], dtype=float)
            self._move_cartesian(
                ee_pos, np.array([ee_pos[0], ee_pos[1], transport_z]),
                arm_tag,
                n_steps=self.DESCENT_CARTESIAN_STEPS,
                sim_per_step=self.DESCENT_SIM_PER_STEP,
            )
            return False
        for _ in range(40):
            self.step_sim()
        # Mid gate is the last contact-free checkpoint.  Open fingers
        # (8 cm gap) clear the 5.08 cm can by only 1.46 cm per side, so
        # require <= 8 mm centering before the final hop (v14 grazed and
        # toppled the can descending at the old 1.5 cm tolerance).
        mid_err = None
        for _try in range(3):
            ee_pos = np.array(arm.get_ee_pose()[:3], dtype=float)
            mid_err = float(np.linalg.norm(ee_pos[:2] - mid_pos[:2]))
            if mid_err <= 0.008:
                break
            print(f"[feed] {asset} off {mid_err:.3f} at mid descent; "
                  f"re-centering", flush=True)
            mid_link = tcp_to_link_pose(gtcp(mid_pos), tcp_offset)
            self._move_seeded_retry(mid_link.to_pose7(), arm_tag)
            for _ in range(self.GRASP_ALIGNMENT_SETTLE_STEPS):
                self.step_sim()
        ee_pos = np.array(arm.get_ee_pose()[:3], dtype=float)
        mid_err = float(np.linalg.norm(ee_pos[:2] - mid_pos[:2]))
        if mid_err > 0.012:
            print(f"[feed] {asset} cannot center at mid descent "
                  f"(err={mid_err:.3f}); aborting attempt untouched",
                  flush=True)
            self._move_cartesian(
                ee_pos, np.array([ee_pos[0], ee_pos[1], transport_z]),
                arm_tag,
                n_steps=self.DESCENT_CARTESIAN_STEPS,
                sim_per_step=self.DESCENT_SIM_PER_STEP,
            )
            return False

        # Final short hop (9 cm) from the verified-centered mid pose —
        # tightest gate: this is where a bad waypoint rakes the part.
        if not self._guarded_vertical(
                np.array([ee_pos[0], ee_pos[1], ee_pos[2]]), grasp_pos,
                arm_tag,
                n_steps=self.DESCENT_CARTESIAN_STEPS,
                sim_per_step=self.DESCENT_SIM_PER_STEP,
                xy_tol=0.015, max_consecutive_bad=2):
            ee_pos = np.array(arm.get_ee_pose()[:3], dtype=float)
            self._move_cartesian(
                ee_pos, np.array([ee_pos[0], ee_pos[1], transport_z]),
                arm_tag,
                n_steps=self.DESCENT_CARTESIAN_STEPS,
                sim_per_step=self.DESCENT_SIM_PER_STEP,
            )
            return False
        for _ in range(self.GRASP_SETTLE_STEPS):
            self.step_sim()

        ee_pos = np.array(arm.get_ee_pose()[:3], dtype=float)
        xy_err = float(np.linalg.norm(ee_pos[:2] - grasp_pos[:2]))
        print(f"[feed] {asset} grasp target={grasp_pos.round(3).tolist()} "
              f"ee={ee_pos.round(3).tolist()} xy_err={xy_err:.3f}", flush=True)
        if xy_err > self.GRASP_ABORT_XY:
            # NO lateral correction down here: retreat vertically and fail
            # before the one-shot gripper close.
            print(f"[feed] {asset} grasp off-center after descent; "
                  f"aborting attempt untouched", flush=True)
            self._move_cartesian(
                ee_pos, np.array([ee_pos[0], ee_pos[1], transport_z]), arm_tag,
                n_steps=self.DESCENT_CARTESIAN_STEPS,
                sim_per_step=self.DESCENT_SIM_PER_STEP,
            )
            return False

        z_before = float(self._get_object_world_center(actor, asset, 0)[2])
        self.set_gripper(close_value, arm_tag)
        for _ in range(self.CLOSE_SETTLE_STEPS):
            self.step_sim()

        # Lift / transit / drop run skip-only guarded (never command an
        # off-line waypoint — that's what drags the gripped part — but
        # don't abort mid-hold).
        lift_pos = np.array([grasp_pos[0], grasp_pos[1], transport_z])
        self._guarded_vertical(
            grasp_pos, lift_pos, arm_tag,
            n_steps=self.LIFT_CARTESIAN_STEPS,
            sim_per_step=self.LIFT_SIM_PER_STEP,
            max_consecutive_bad=10**6,
        )
        for _ in range(self.POST_LIFT_SETTLE_STEPS):
            self.step_sim()

        z_after = float(self._get_object_world_center(actor, asset, 0)[2])
        dz = z_after - z_before
        print(f"[feed] {asset} lift dz={dz:.3f}", flush=True)
        if dz < self.LIFT_DZ_MIN:
            self.open_gripper(arm_tag)
            for _ in range(self.LIFT_FAIL_RELEASE_SETTLE_STEPS):
                self.step_sim()
            return False

        # Restore wrist headroom before the carry if the grasp wound q7
        # near its limit (v22: carry branch died at q7=2.89).
        self._recenter_wrist(arm_tag)

        # Transit via a radially-bowed midpoint: a straight carry through
        # r~0.5 m of the base locks the seeded IK onto a flipped branch
        # from a consistent spot (v16-19; same finding as inspect_pack's
        # "carries too" memory addendum — bow the path out to r>=0.55).
        above_drop = np.array([drop_xy[0], drop_xy[1], transport_z])
        base_xy = np.asarray(arm.origin_pose.p, dtype=float)[:2]
        mid_xy = 0.5 * (lift_pos[:2] + above_drop[:2])
        v = mid_xy - base_xy
        r = float(np.linalg.norm(v))
        if r < 0.58:
            mid_xy = base_xy + v / max(r, 1e-6) * 0.58
        via = np.array([mid_xy[0], mid_xy[1], transport_z])
        for leg_start, leg_end in ((lift_pos, via), (via, above_drop)):
            self._guarded_vertical(
                leg_start, leg_end, arm_tag,
                n_steps=self.TRANSIT_CARTESIAN_STEPS // 2 + 5,
                sim_per_step=self.TRANSIT_SIM_PER_STEP,
                max_consecutive_bad=10**6,
                yaw_follow_base=True,
            )
        for _ in range(self.POST_TRANSIT_SETTLE_STEPS):
            self.step_sim()

        # Verify arrival — a branch-flip-skipped leg leaves the arm short
        # of the pad.  Retry the REMAINING transit with the guarded mover
        # (fresh seed often heals the branch); never joint-RRT while
        # holding — that's what whipped the can off in the v17 smoke.
        for _retry in range(2):
            ee_now = np.array(arm.get_ee_pose()[:3], dtype=float)
            if float(np.linalg.norm(ee_now[:2] - above_drop[:2])) <= 0.05:
                break
            print(f"[feed] {asset} transit fell short "
                  f"(ee={ee_now.round(3).tolist()}); retrying remaining leg",
                  flush=True)
            self._guarded_vertical(
                ee_now, above_drop, arm_tag,
                n_steps=self.TRANSIT_CARTESIAN_STEPS,
                sim_per_step=self.TRANSIT_SIM_PER_STEP,
                max_consecutive_bad=10**6,
                yaw_follow_base=True,
            )
            for _ in range(self.POST_TRANSIT_SETTLE_STEPS):
                self.step_sim()
        ee_now = np.array(arm.get_ee_pose()[:3], dtype=float)
        if float(np.linalg.norm(ee_now[:2] - above_drop[:2])) > 0.05:
            # Can't reach the pad with the part in hand — set it down
            # gently where we are and fail this one-shot delivery.
            print(f"[feed] {asset} cannot reach the pad; setting part "
                  f"down at {ee_now[:2].round(3).tolist()}", flush=True)
            for _try in range(2):
                ee_now = np.array(arm.get_ee_pose()[:3], dtype=float)
                if ee_now[2] <= drop_z + 0.05:
                    break
                self._guarded_vertical(
                    ee_now, np.array([ee_now[0], ee_now[1], drop_z]), arm_tag,
                    n_steps=self.DESCENT_CARTESIAN_STEPS,
                    sim_per_step=self.DESCENT_SIM_PER_STEP,
                    max_consecutive_bad=10**6,
                )
            ee_now = np.array(arm.get_ee_pose()[:3], dtype=float)
            if ee_now[2] > drop_z + 0.05:
                print(f"[feed] WARNING set-down descent stuck at "
                      f"z={ee_now[2]:.2f}; releasing high", flush=True)
            self.open_gripper(arm_tag)
            for _ in range(self.RELEASE_SETTLE_STEPS):
                self.step_sim()
            return False

        ee_xy = np.array(arm.get_ee_pose()[:2], dtype=float)
        obj_xy = self._get_object_world_center(actor, asset, 0)[:2]
        slip = float(np.linalg.norm(ee_xy - obj_xy))
        if slip > self.TRANSIT_SLIP_XY_TOL:
            print(f"[feed] {asset} slipped in transit ({slip:.3f}m)", flush=True)
            return False

        drop_pos = np.array([drop_xy[0], drop_xy[1], drop_z])
        ee_now = np.array(arm.get_ee_pose()[:3], dtype=float)
        self._guarded_vertical(
            ee_now, drop_pos, arm_tag,
            n_steps=self.DESCENT_CARTESIAN_STEPS,
            sim_per_step=self.DESCENT_SIM_PER_STEP,
            max_consecutive_bad=10**6,
        )
        for _ in range(self.DROP_SETTLE_STEPS):
            self.step_sim()

        self.open_gripper(arm_tag)
        for _ in range(self.RELEASE_SETTLE_STEPS):
            self.step_sim()
        return True

    def _deliver_part(self, i: int, arm_tag: str) -> bool:
        actor, asset, _mid, close, grasp_off, grasp_axis, half_h = \
            self.part_actors[i]
        # Fail-fast if the tool was knocked far off its resting height.
        fall_z = self.TABLE_TOP_Z + half_h - 0.04
        try:
            obj_z = float(self._get_object_world_center(actor, asset, 0)[2])
        except Exception:
            obj_z = float("-inf")
        if obj_z < fall_z:
            print(f"[feed] {asset} displaced low "
                  f"(z={obj_z:.3f} < {fall_z:.3f}); abandoning part",
                  flush=True)
            return False
        try:
            if self._feed_pick_once(
                actor, asset, arm_tag, close,
                drop_xy=self._region_drop_xy(i), grasp_off=grasp_off,
                grasp_axis=grasp_axis,
            ):
                return True
        except Exception as exc:
            print(f"[feed] robot delivery of {asset} raised: {exc}",
                  flush=True)
        print(f"[feed] robot delivery of {asset} failed", flush=True)
        return False

    def _park_arm(self, arm_tag: str) -> None:
        arm = self.robot.get_arm(arm_tag)
        # Retract straight UP first — a joint-space move from the low drop
        # pose can sweep across the pad and bat the just-released part
        # (v3 smoke: can displaced 0.17 m between release and take).
        ee = np.array(arm.get_ee_pose()[:3], dtype=float)
        up = np.array([ee[0], ee[1], self.TABLE_TOP_Z + self.TRANSPORT_Z_ABOVE_TABLE])
        self._guarded_vertical(
            ee, up, arm_tag,
            n_steps=self.DESCENT_CARTESIAN_STEPS,
            sim_per_step=self.DESCENT_SIM_PER_STEP,
            max_consecutive_bad=10**6,
        )
        # Park = the canonical ready pose (joint-space, branch-clean, and
        # its EE sits high over the bench center clear of the worker's
        # reach corridor to the pad).
        self._goto_ready(arm_tag)

    def _part_in_region(self, i: int, verbose: bool = True) -> bool:
        """Success gate: the tool's geometric CENTRE must lie inside the
        green region footprint (the bar may overhang; its centre may not)."""
        actor, asset, mid, *_ = self.part_actors[i]
        c = np.asarray(
            self._get_object_world_center(actor, asset, mid), dtype=float)
        rx, ry = self.HANDOVER_XY
        rhx, rhy = self.REGION_PAD_HALF
        z_lo, z_hi = self.TABLE_TOP_Z - 0.02, self.TABLE_TOP_Z + 0.10
        dx, dy = abs(float(c[0]) - rx), abs(float(c[1]) - ry)
        inside = (dx <= rhx and dy <= rhy and z_lo <= float(c[2]) <= z_hi)
        if verbose:
            print(f"[feed] region check part {i} ({asset}): "
                  f"center={c.round(3).tolist()} "
                  f"d_xy=({dx:.3f},{dy:.3f}) vs ({rhx},{rhy}) "
                  f"in_region={inside}", flush=True)
        return inside

    # ------------------------------------------------------------------
    # Rollout: per part — worker reaches out, robot delivers to the
    # table-edge region, worker retracts; the part stays in the region.
    # ------------------------------------------------------------------
    def play_once(self) -> bool:
        arm_tag = "right"
        self._boost_finger_pd(arm_tag)
        with self.suppress_recording():  # pre-task settle, excluded from recordings
            for _ in range(self.INITIAL_SETTLE_STEPS):
                self.step_sim()
        for actor, asset, mid, _c, goff, _ga, _h in self.part_actors:
            p = actor.get_pose()
            c = self._pick_point(actor, asset, mid, goff)
            print(f"[feed] {asset} settled pos={np.round(p.p,3).tolist()} "
                  f"quat={np.round(p.q,3).tolist()} handle={np.round(c,3).tolist()}",
                  flush=True)

        if self.avatar is None:
            # No-human fallback: just deliver every part to the region.
            ok = all(self._deliver_part(i, arm_tag)
                     for i in range(len(self.part_actors)))
            self._delivered = [ok] * len(self.part_actors)
            return ok

        hand_id = int(self._work["hand_id"])

        ok = True
        for i in range(len(self.part_actors)):
            # Worker does its task for a randomized stretch, then reaches
            # out for help at that point.
            self._play_work_until_reach()
            # 1) Worker presents a hand at the table edge and holds it.
            self._reach_out(hand_id)
            self._reached[i] = True
            # 2) Robot picks the part and places it in the region.
            if not self._deliver_part(i, arm_tag):
                self._retract_hand(hand_id)
                ok = False
                break
            for _ in range(self.DELIVER_SETTLE_STEPS):
                self.step_sim()
            if not self._part_in_region(i):
                print(f"[feed] part {i} not resting in the region after "
                      f"delivery; failing", flush=True)
                self._retract_hand(hand_id)
                ok = False
                break
            self._delivered[i] = True
            # 3) Robot parks clear, worker pulls the hand back to its work.
            self._park_arm(arm_tag)
            self._retract_hand(hand_id)

        for _ in range(self.FINAL_SETTLE_STEPS):
            self.step_sim()
        for j in range(len(self.part_actors)):
            self._part_in_region(j)
        return ok

    # ------------------------------------------------------------------
    # Success: every part delivered into the table-edge region, no
    # robot-avatar collision.
    # ------------------------------------------------------------------
    def check_success(self) -> bool:
        if not self.plan_success:
            return False
        if (self.config.get("track_avatar_collision", True)
                and getattr(self, "avatar_collided", False)):
            return False
        if not (all(self._reached) and all(self._delivered)):
            return False
        for i in range(len(self.part_actors)):
            if not self._part_in_region(i, verbose=False):
                return False
        return True

    def evaluate(self) -> dict:
        metrics = super().evaluate()
        per_part = {}
        for i, (actor, asset, mid, *_rest) in enumerate(self.part_actors):
            c = self._get_object_world_center(actor, asset, mid)
            per_part[asset] = {
                "reached": bool(self._reached[i]),
                "delivered": bool(self._delivered[i]),
                "in_region": bool(self._part_in_region(i, verbose=False)),
                "final_pos": [round(float(v), 3) for v in c],
            }
        metrics["parts"] = per_part
        metrics["work_motion"] = self._work["clip"]
        metrics["avatar_collided"] = bool(getattr(self, "avatar_collided", False))
        return metrics
