"""Factory inspect & pack (sequential cooperative QC) — Task 3 in
``docs/factory_tasks.md``.

Eval category: **intent** (a.k.a. cueing) — the human worker's inspect
gesture cues which product is ready, and the robot acts on that intent.
Registered in the eval-set ``intent`` group (``scripts/generate_eval_set.py``).

Quality-control pipeline at the factory work cell (``FactorySceneMixin``
bench, top z = 0.765): the human worker inspects products upstream, the
robot packs downstream — the robot may only pack an item after the human
has inspected it.

Scene: two distinct products sampled from the shared factory product pool
spawn on the worker's front band of the bench; an open-top shipping box
(five static primitives) sits on the robot's side.

Human (continuous, never waits for the robot): per item, the avatar is
placed so the item sits at the calibrated palm point of that item's
inspect motion, then plays the motion with a kinematic attach/detach
(``Inspect`` family — pick up, examine, set back down).  Between items the
avatar root *glides* (small interpolated steps, no teleport) from the
first inspect stance to the second — the "walk to the next product" beat.
Each item uses a different motion from the pool ("mixed" inspect styles).
The per-motion palm anchor is measured once with an offscreen dry-run at
``frame_ratio=1`` (cached at class level across episodes).

Robot (gated): waits until item i's inspect motion has finished (object
set down, hand withdrawn), then runs the proven ``PlaceBurgerFries``
pick pipeline (seeded RRT to above-object, screw descent, pinned close,
lift verification, slow Cartesian transit with slip detector, hover
release) to pack it into the box — while the human is already gliding to /
inspecting the next item.  The avatar capsule pcd is pushed into the mplib
planner obstacles before every pick.

The avatar pipeline (inspect -> glide -> inspect) advances inside a
patched ``step_sim`` so it runs concurrently with the robot's planned
trajectories — same pattern as ``factory_kit_packing``.

Success (geometric + gating): every item inside the box interior AABB,
every item's inspect motion completed (per-item state machine), and no
robot-avatar collision.
"""

from __future__ import annotations

import json

import numpy as np
import transforms3d as t3d

from ..avatar.inspect_motion_mixin import INSPECT_MOTION_SPECS
from ..base_task import BaseTask
from ..grasp import tcp_to_link_pose
from ..manipulation import TopDownPickPlaceMixin
from ..object_catalog import get_entry, resolve_object_set
from ..scenes.factory import FactorySceneMixin
from ..task_bases.place_burger_fries import PlaceBurgerFries
from ..utils import ASSETS_PATH, Pose, create_primitive, load_mesh, load_object

# upright = mesh-Y -> world-Z (these assets are y-up)
_Q_UPRIGHT = np.array([0.70710678, 0.70710678, 0.0, 0.0], dtype=np.float64)


def _read_model_data(name: str, mid: int) -> tuple[float, np.ndarray]:
    """(scale, extents) from model_data{mid}.json."""
    p = ASSETS_PATH / "objects" / name / f"model_data{mid}.json"
    with open(p) as f:
        d = json.load(f)
    raw = d.get("scale", 1.0)
    scale = float(raw[0] if isinstance(raw, (list, tuple)) else raw)
    return scale, np.asarray(d["extents"], dtype=np.float64)


def _factory_inspect_item_pool():
    close_values = {
        "113_coffee-box": 0.42,
        "038_milk-box": 0.45,
        "112_tea-box": 0.45,
        "023_tissue-box": 0.46,
        "073_rubikscube": 0.43,
        "086_woodenblock": 0.44,
    }
    return tuple(
        (entry.object_id, entry.model_id, 4.0, close_values[entry.key])
        for entry in (
            get_entry(token) for token in resolve_object_set("factory_product_pool")
        )
    )


class FactoryInspectPack(FactorySceneMixin, TopDownPickPlaceMixin, BaseTask):
    """Human QC-inspects each product; robot packs only inspected products."""

    INSTRUCTION = (
        "pack each product into the shipping box "
        "after the human has inspected it"
    )

    use_avatar = True
    OBJECT_SET = "factory_product_pool"

    # Mount the Franka yawed +150° so the worker-side bench band faces the
    # arm.  With the scene's identity-quat mount the milk-box spawn sits at
    # azimuth ~183° from the base — beyond the q1 limit (+-166°) — so the
    # vertical descent can never be screw-planned (v19: "no descent plan"
    # forever).  Same fix factory_kit_packing / factory_line_feeding
    # converged on (see memory factory-robot-mount-yaw).
    FACTORY_ROBOT_KWARGS = {
        "pos": [0.60, -0.30, 0.765],
        "quat": [0.2588190, 0.0, 0.0, 0.9659258],   # yaw +150 deg (w,x,y,z)
    }

    # ---- Outbound package: procedurally generated open cardboard carton
    # (kraft-noise texture, 4 splayed lid flaps — the opened-Amazon-box
    # look the user asked for; no library asset matches: 062_plasticbox
    # comes only in white/red/blue/green, the shoe-box is orange, and the
    # SAPIEN cartons are flat gray with trap-prone articulated lids).
    # Visual = generated GLB render-only; collision = 5 invisible
    # primitives (floor + walls; flaps are out of the gripper's path and
    # carry no collision).  Sized like the earlier 1.6x tote, which had
    # comfortable drop margins.
    BOX_XY = (0.42, 0.12)              # default; randomized per episode
    BOX_INNER_HALF = (0.15, 0.11)
    BOX_WALL_H = 0.12
    # Per-episode carton position (robot side).  Bounds keep both drop
    # slots in the robot's 0.42-0.60 m reach annulus, the carton on the
    # bench, and well clear of the items + worker (on the opposite -y band).
    CARTON_XY_RANGE = ((0.34, 0.44), (0.10, 0.20))
    CARTON_WALL_T = 0.008
    CARTON_FLOOR_T = 0.010
    # Flaps laid back wide-open (62 deg from vertical): tips reach only
    # z0 + 0.12 + 0.10*cos(62) ~= z0 + 0.167, so a reliable z0+0.30 carry
    # (object bottom ~z0+0.24) clears them by ~7 cm.  The earlier near-
    # vertical 35-deg flaps reached z0+0.21 and forced a z0+0.40 carry
    # whose per-waypoint IK broke near the reach limit.  Flatter flaps
    # also read more like an opened Amazon box.
    CARTON_FLAP_LEN = 0.10
    CARTON_FLAP_SPLAY_DEG = 62.0    # outward lean from vertical
    TOTE_FRICTION = 5.0

    # ---- Products ----------------------------------------------------------
    # (asset, model_id, x_range, y_range, friction, close_value, slot dx/dy,
    #  tcp_yaw_deg).  Spawn band = worker's front strip of the bench, inside
    # the robot's comfortable top-down reach from its base at (+0.60, -0.30).
    # Inspect order = list order: the near-robot item first, so the human
    # glides AWAY from the robot's first pick while it runs.
    # Both items are boxy on purpose: the v1 smoke used 071_can and the
    # inspect set-down tipped it onto its side, where the top-down pick
    # cannot lift it.  The set-down point is displaced from the pickup by
    # the motion's detach-attach palm offset (logged per episode as
    # "landing"), so spawn bands must keep the predicted landings inside
    # the robot's usable annulus — measured here AND by factory_kit_packing:
    # picks closer than ~0.35 m to the Franka base hit an IK dead zone
    # (v4: seeded move ended 0.54 m off at a 0.29 m-reach landing), and
    # beyond ~0.62 m they run out of reach.  Current landings: item 0
    # ~0.40 m, item 1 ~0.59 m.
    # v5 lessons: item 0's landing drifted to 9 cm from the bench front edge
    # (knocked off during the pick) and item 1's landing reached 0.59 m
    # (wrong-branch sprawl).  Bands below keep landings >=10 cm from the
    # edge and reaches at ~0.42 / ~0.54.
    # Pool of boxy, gripper-compatible products (min horizontal extent
    # < 7.5 cm so the top-down jaw closes on the narrowest face, and tall
    # enough not to lie flat on the inspect set-down).  Two DISTINCT items
    # are drawn per episode.  (asset, model_id, friction, close_value).
    ITEM_POOL = _factory_inspect_item_pool()
    # Two spawn bands on the worker's front strip + the carton drop slot
    # for each.  Band 0 is inspected first (clip forward) and packed first;
    # band 1 is inspected via the reversed clip.  Bands keep predicted
    # landings in the robot's reach annulus and >=10 cm off the bench edge.
    ITEM_BANDS = (
        ((0.16, 0.20), (-0.32, -0.29), (-0.06, 0.0)),
        ((0.04, 0.08), (-0.38, -0.35), (+0.06, 0.0)),
    )
    ROBOT_BELOW_CENTER = 0.005

    # ---- Inspect motion (one clip, played forward then in REVERSE) --------
    # Item 0 is inspected by the clip played forward (walk in, pick up,
    # examine, set down); item 1 by the SAME clip played backward from the
    # set-down frame — the reversed put-down is a pick-up, so it reads as
    # back-and-forth inspecting, and the avatar's pose is continuous
    # through the inter-item glide (user feedback on v16: gliding in the
    # neutral pose snapped against the clip's end frame).
    # Default motion chosen for palm height: the avatar root is
    # z-calibrated (stance z = item z - palm-anchor z), so motions authored
    # for a taller/lower surface sink/float the avatar.  Measured root-z
    # error vs the -0.18 baseline at this bench: Inspect3 +0.03, Inspect4
    # +0.02 (grounded); Inspect1 -0.12, Inspect2 +0.13 (visible).
    INSPECT_MOTION_POOL = ("Inspect1", "Inspect2", "Inspect3", "Inspect4")
    INSPECT_MOTION_DEFAULT = "Inspect3"
    AVATAR_MOTION_SLOW = 10.0
    AVATAR_HAND_ID = 1
    AVATAR_EXTRA_Z = 0.04
    AVATAR_MEASURE_POS = np.array([6.0, 6.0, -0.18])
    # Keep the pelvis clear of the bench front edge (y = -0.475) and off the
    # robot mount at x = +0.60.
    AVATAR_BODY_Y_MAX = -0.575
    AVATAR_BODY_X_BOUNDS = (-0.78, 0.42)

    # Avatar root glide between the two inspect stances: total sim steps and
    # how often the root is re-set (every step would rebuild the skin mesh
    # ~600x; every N steps moves ~4 mm per update — visually continuous).
    # During the glide the avatar holds the clip's end frame (the reversed
    # second clip starts from that same frame, so the whole transition is
    # pose-continuous).  After the reversed clip she simply stays where it
    # ends — no park walk (no realistic motion exists for one).
    GLIDE_STEPS = 600
    GLIDE_UPDATE_EVERY = 10

    # ---- Robot pacing --------------------------------------------------------
    INITIAL_SETTLE_STEPS = 80
    INSPECTED_SETTLE_STEPS = 120     # let the set-down item settle before pick
    WAIT_INSPECT_CAP_STEPS = 40000
    AVATAR_FINISH_WAIT_STEPS = 40000
    FINAL_SETTLE_STEPS = 150
    SUCCESS_Z_MAX_ABOVE_TABLE = 0.30

    # Knobs consumed by the reused PlaceBurgerFries pick pipeline (values
    # proven on this bench by factory_kit_packing).  TRAY_RIM_Z doubles as
    # the box wall height: drop z = TABLE_TOP_Z + TRAY_RIM_Z + hover.
    FINGER_KP = 9000.0
    FINGER_KV = 250.0
    FINGER_FORCE_LIMIT = 80.0
    # Carry height kept at the IK-reliable z0+0.30 (z0+0.40 broke the
    # per-waypoint carry IK near the reach limit — branch guard rejected
    # every waypoint).  The lid is flattened instead (flap tips ~z0+0.167)
    # so the carried item clears it by ~7 cm at this height.
    TRANSPORT_Z_ABOVE_TABLE = 0.30
    TRAY_RIM_Z = 0.12
    DROP_HOVER_Z_ABOVE_TRAY = 0.07
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
    TRANSIT_CARTESIAN_STEPS = 40
    TRANSIT_SIM_PER_STEP = 26   # slower carry — less inertial load on the grip
    POST_TRANSIT_SETTLE_STEPS = 50
    TRANSIT_SLIP_XY_TOL = 0.10
    DROP_SETTLE_STEPS = 40
    RELEASE_SETTLE_STEPS = 200

    _AVATAR_OBSTACLE_RES = 0.04
    _AVATAR_INFLATE_FACTOR = 1.8

    # Before lowering onto an item, wait until the worker's nearest capsule
    # is at least this far (xy) from the pick column, up to a step cap.
    DESCENT_CLEAR_XY = 0.16
    DESCENT_CLEAR_CAP_STEPS = 1500
    # Don't START a pick while the worker's nearest capsule is within this
    # xy of the item — avoids planning a pick into her reach (collision /
    # cramped mis-carry).  She retracts after inspecting the adjacent item.
    PICK_START_CLEAR_XY = 0.20
    # A palm at y >= this is reaching onto the bench; below it she's
    # retracted.  Picks wait until both palms are below (off the worktop).
    BENCH_CLEAR_Y = -0.55
    # Max planning/clearance stalls before abandoning a pick.  Sized to
    # outlast the worker's adjacent-item inspection (+park) so a concurrent
    # pick that she's blocking waits for her to clear rather than failing.
    MAX_PICK_STALLS = 28

    # Reuse pieces of the proven burger-fries pick pipeline (plain
    # unbound-method reuse — no shared-code change).  Its pick wrapper and
    # `_pick_once` are not reused: `_pack_item` waits for a clear corridor,
    # then makes one pick, and `_pick_once` below plans the whole approach
    # before moving (the original staged
    # hover-above -> pre -> grasp sequence read as "way above the object,
    # back and forth" when a mid-sequence leg failed — user feedback).
    _boost_finger_pd = PlaceBurgerFries._boost_finger_pd
    _rotated_top_down_tcp = PlaceBurgerFries._rotated_top_down_tcp

    _palm_anchor_cache: dict = {}

    def __init__(self, config: dict = None):
        cfg = dict(config or {})
        cfg.setdefault("robot_type", "franka")
        cfg.setdefault("robot_single_arm", True)
        cfg.setdefault("use_avatar_collider", True)
        cfg.setdefault("track_avatar_collision", True)
        super().__init__(cfg)
        self.item_actors = []     # [(actor, asset, mid, half_h, close, slot, yaw)]
        self._inspected = []      # per-item: detach fired (object set down)
        self._inspect_done = []   # per-item: motion finished (hand withdrawn)

    # ------------------------------------------------------------------
    # Scene
    # ------------------------------------------------------------------
    def reset(self, seed: int = 0):
        self._restore_step_sim()
        obs = super().reset(seed=seed)
        if self.avatar is not None:
            self.avatar.frame_ratio = float(self.AVATAR_MOTION_SLOW)
            self._select_inspect_motions()
            self._calibrate_avatar_stances()
            self._prepare_item_clips()
        return obs

    _CARTON_GLB = ASSETS_PATH / "objects" / "generated_qc_carton" / "carton.glb"

    def _ensure_carton_glb(self):
        """Generate the open-carton mesh once (seeded — deterministic):
        floor + 4 walls + 4 outward-splayed lid flaps, one kraft-paper
        noise texture mapped via a seam-tolerant planar projection."""
        if self._CARTON_GLB.exists():
            return
        import trimesh
        from PIL import Image
        rng = np.random.default_rng(7)

        ihx, ihy = self.BOX_INNER_HALF
        t, ft = self.CARTON_WALL_T, self.CARTON_FLOOR_T
        wh = self.BOX_WALL_H
        fl = self.CARTON_FLAP_LEN
        splay = np.deg2rad(self.CARTON_FLAP_SPLAY_DEG)
        ohx, ohy = ihx + t, ihy + t

        panels = []

        def _box(extents, transform):
            b = trimesh.creation.box(extents=extents)
            b.apply_transform(transform)
            panels.append(b)

        def _T(dx=0.0, dy=0.0, dz=0.0, rot=None):
            m = np.eye(4)
            if rot is not None:
                m[:3, :3] = rot
            m[:3, 3] = [dx, dy, dz]
            return m

        _box([2 * ohx, 2 * ohy, ft], _T(dz=ft / 2))
        for sx in (-1, +1):
            _box([t, 2 * ohy, wh], _T(dx=sx * (ihx + t / 2), dz=wh / 2))
        for sy in (-1, +1):
            _box([2 * ohx, t, wh], _T(dy=sy * (ihy + t / 2), dz=wh / 2))

        # Lid flaps: hinged at the wall tops, leaning outward from vertical.
        for sx in (-1, +1):
            ang = sx * splay
            rot = t3d.axangles.axangle2mat([0, 1, 0], ang)
            off = rot @ np.array([0.0, 0.0, fl / 2])
            _box([t * 0.75, 2 * ohy, fl],
                 _T(dx=sx * ohx + off[0], dy=0, dz=wh + off[2], rot=rot))
        for sy in (-1, +1):
            ang = -sy * splay
            rot = t3d.axangles.axangle2mat([1, 0, 0], ang)
            off = rot @ np.array([0.0, 0.0, fl / 2])
            _box([2 * ohx, t * 0.75, fl],
                 _T(dx=0, dy=sy * ohy + off[1], dz=wh + off[2], rot=rot))

        mesh = trimesh.util.concatenate(panels)

        # Kraft-paper texture: tan base + noise + faint corrugation streaks.
        size = 512
        base = np.array([181, 140, 92], dtype=np.float64)
        img = base[None, None, :] + rng.normal(0.0, 7.0, (size, size, 1))
        for _ in range(40):
            y = int(rng.integers(0, size))
            img[y:y + 1, :, :] -= rng.uniform(6.0, 16.0)
        for _ in range(12):
            x = int(rng.integers(0, size))
            w = int(rng.integers(2, 6))
            img[:, x:x + w, :] -= rng.uniform(4.0, 10.0)
        tex = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8), "RGB")

        v = mesh.vertices
        uv = np.stack([(v[:, 0] + v[:, 1]) * 1.6, (v[:, 2] + v[:, 1]) * 1.6], axis=1)
        mesh.visual = trimesh.visual.TextureVisuals(
            uv=uv, material=trimesh.visual.material.SimpleMaterial(image=tex),
        )
        self._CARTON_GLB.parent.mkdir(parents=True, exist_ok=True)
        mesh.export(self._CARTON_GLB)
        print(f"[qc] generated carton GLB at {self._CARTON_GLB}", flush=True)

    def _load_carton(self):
        bx, by = self.BOX_XY
        self._ensure_carton_glb()
        ihx, ihy = self.BOX_INNER_HALF
        t, wh = self.CARTON_WALL_T, self.BOX_WALL_H
        z0 = self.TABLE_TOP_Z
        self.carton_visual = load_mesh(
            self.scene, self._CARTON_GLB,
            Pose([bx, by, z0 + 0.001], np.array([1.0, 0.0, 0.0, 0.0])),
            convex=False, is_static=True, collision=False, visual=True,
        )
        # Invisible collision: floor + 4 walls (flaps stay render-only).
        self.carton_colliders = [create_primitive(
            self.scene, "box",
            Pose(p=[bx, by, z0 + self.CARTON_FLOOR_T / 2]),
            size={"half_size": (ihx + t, ihy + t, self.CARTON_FLOOR_T / 2)},
            is_static=True, visual=False,
        )]
        wz = z0 + wh / 2
        for sx in (-1, +1):
            self.carton_colliders.append(create_primitive(
                self.scene, "box",
                Pose(p=[bx + sx * (ihx + t / 2), by, wz]),
                size={"half_size": (t / 2, ihy + t, wh / 2)},
                is_static=True, visual=False,
            ))
        for sy in (-1, +1):
            self.carton_colliders.append(create_primitive(
                self.scene, "box",
                Pose(p=[bx, by + sy * (ihy + t / 2), wz]),
                size={"half_size": (ihx + t, t / 2, wh / 2)},
                is_static=True, visual=False,
            ))

    def load_actors(self):
        # Per-episode randomization (np.random already seeded by reset):
        #   - carton position within CARTON_XY_RANGE,
        #   - 2 distinct item TYPES from ITEM_POOL assigned to the 2 bands,
        #   - item POSITIONS within each band.
        (cxlo, cxhi), (cylo, cyhi) = self.CARTON_XY_RANGE
        self.BOX_XY = (float(np.random.uniform(cxlo, cxhi)),
                       float(np.random.uniform(cylo, cyhi)))
        self.build_factory()
        self._load_carton()

        pool = {(item[0], int(item[1])): item for item in self.ITEM_POOL}
        entries = self.resolve_target_objects(len(self.ITEM_BANDS), replace=False)
        chosen = [pool[(entry.object_id, int(entry.model_id))] for entry in entries]
        self.item_actors = []
        for (asset, mid, fric, close), ((xlo, xhi), (ylo, yhi), slot) in zip(
            chosen, self.ITEM_BANDS,
        ):
            scale, extents = _read_model_data(asset, mid)
            half_h = float(extents[1]) * scale / 2.0
            x = float(np.random.uniform(xlo, xhi))
            y = float(np.random.uniform(ylo, yhi))
            actor = load_object(
                self.scene,
                Pose([x, y, self.TABLE_TOP_Z + half_h + 0.01], _Q_UPRIGHT),
                asset, model_id=mid,
                convex=True, is_static=False, friction=fric,
            )
            self.item_actors.append((actor, asset, mid, half_h, close, slot, 0.0))
        self._inspected = [False] * len(self.item_actors)
        self._inspect_done = [False] * len(self.item_actors)
        print(f"[qc] randomized: carton={tuple(round(v, 3) for v in self.BOX_XY)} "
              f"items={[c[1] for c in self.item_actors]}", flush=True)

    @staticmethod
    def _entity(actor):
        return getattr(actor, "entity", actor)

    # ------------------------------------------------------------------
    # Avatar: motion selection + palm calibration + stances
    # ------------------------------------------------------------------
    def _resolve_spec(self, name: str):
        """Resolve a pool entry to a runtime InspectMotionSpec.  Motions with
        a ``clip`` get their subclip installed into the avatar's motion data
        (idempotent) and their frames converted to clip-local — same scheme
        as ``InspectMotionMixin._prepare_inspect_motion_entry``."""
        from dataclasses import replace
        spec = INSPECT_MOTION_SPECS[name]
        if spec.clip is None:
            return spec
        lo, hi = (int(spec.clip[0]), int(spec.clip[1]))
        dst = f"{spec.name}_clip_{lo}_{hi}"
        if dst not in self.avatar.motion_data:
            md = self.avatar.motion_data[spec.name]
            self.avatar.motion_data[dst] = {k: v[lo:hi + 1].copy() for k, v in md.items()}
        return replace(spec, name=dst, clip=None,
                       attach=spec.attach - lo, detach=spec.detach - lo)

    def _select_inspect_motions(self) -> None:
        """One source motion per episode; item 0 plays it forward, item 1
        plays it in reverse (back-and-forth inspecting).
        --randomize-inspect samples the source from the pool per seed."""
        if self.config.get("randomize_inspect", False):
            name = self.INSPECT_MOTION_POOL[
                int(np.random.randint(len(self.INSPECT_MOTION_POOL)))
            ]
        else:
            name = str(self.config.get(
                "inspect_motion_name", self.INSPECT_MOTION_DEFAULT,
            ))
        self._src_spec = self._resolve_spec(name)
        self._item_dirs = ["fwd" if i % 2 == 0 else "rev"
                           for i in range(len(self.ITEM_BANDS))]
        print(f"[qc] inspect motion: {name} "
              f"(dirs {self._item_dirs})", flush=True)

    def _scaled(self, frame: int) -> int:
        return int(round(frame * float(self.AVATAR_MOTION_SLOW)))

    def _measure_palm_anchors(self, motion_name: str, attach_frame: int,
                              detach_frame: int, hand_id: int):
        """(attach, detach) palm-center offsets (relative to avatar root),
        measured by an offscreen dry-run at frame_ratio=1.  The detach
        anchor predicts where the inspected item is set back down.  Cached
        at class level — rotation and root are fixed per task."""
        key = (motion_name, int(hand_id))
        cached = FactoryInspectPack._palm_anchor_cache.get(key)
        if cached is not None:
            return cached[0].copy(), cached[1].copy()

        base_pos = np.asarray(self.AVATAR_MEASURE_POS, dtype=np.float64)
        base_rot = np.asarray(self.avatar_init_rot, dtype=np.float64)
        self.avatar.reset(base_pos.copy(), base_rot.copy())
        old_ratio = float(self.avatar.frame_ratio)
        self.avatar.frame_ratio = 1.0
        self.avatar.play_animation(motion_name)
        rel_a = rel_d = None
        step = 0
        while not self.avatar.spare() and step <= detach_frame + 2:
            self.scene.step()
            self.avatar.step()
            if step in (attach_frame, detach_frame):
                palm = np.asarray(
                    self.avatar.robot.get_palm_center(int(hand_id)),
                    dtype=np.float64,
                )
                if step == attach_frame:
                    rel_a = palm - base_pos
                if step == detach_frame:
                    rel_d = palm - base_pos
                    break
            step += 1
        self.avatar.frame_ratio = old_ratio
        if rel_a is None or rel_d is None:
            raise RuntimeError(
                f"[qc] palm dry-run for {motion_name} missed "
                f"attach/detach frames {attach_frame}/{detach_frame} "
                f"(stopped at step {step})"
            )
        FactoryInspectPack._palm_anchor_cache[key] = (rel_a.copy(), rel_d.copy())
        print(f"[qc] palm anchors {motion_name}: attach={rel_a.round(3).tolist()} "
              f"detach={rel_d.round(3).tolist()} "
              f"setdown_offset={(rel_d - rel_a).round(3).tolist()}", flush=True)
        return rel_a, rel_d

    def _stance_for_item(self, i: int) -> np.ndarray:
        """Avatar root that puts item i at the calibrated palm point of its
        playback direction: forward picks at the attach anchor and sets
        down at the detach anchor; reversed playback swaps the two."""
        spec = self._src_spec
        rel_a, rel_d = self._measure_palm_anchors(
            spec.name, spec.attach, spec.detach, spec.hand_id,
        )
        pick_rel, place_rel = (
            (rel_a, rel_d) if self._item_dirs[i] == "fwd" else (rel_d, rel_a)
        )
        actor, asset, mid, *_ = self.item_actors[i]
        target = np.asarray(
            self._get_object_world_center(actor, asset, mid), dtype=np.float64,
        )
        body = target - pick_rel
        body[2] += float(self.AVATAR_EXTRA_Z)
        raw = body.copy()
        body[0] = float(np.clip(body[0], *self.AVATAR_BODY_X_BOUNDS))
        body[1] = min(float(body[1]), float(self.AVATAR_BODY_Y_MAX))
        if np.linalg.norm(body[:2] - raw[:2]) > 1e-6:
            print(f"[qc] WARNING stance {i} clamped "
                  f"{raw[:2].round(3).tolist()} -> {body[:2].round(3).tolist()}; "
                  f"the palm will miss the item by the same amount", flush=True)
        landing = body + place_rel
        base_xy = np.asarray(self.FACTORY_ROBOT_KWARGS["pos"][:2], dtype=np.float64)
        reach = float(np.linalg.norm(landing[:2] - base_xy))
        print(f"[qc] stance {i} ({spec.name} {self._item_dirs[i]}): "
              f"body={body.round(3).tolist()} item={target.round(3).tolist()} "
              f"landing~{landing[:2].round(3).tolist()} reach={reach:.2f}", flush=True)
        if reach > 0.65:
            print(f"[qc] WARNING landing {i} predicted outside robot reach "
                  f"({reach:.2f} m > 0.65 m)", flush=True)
        return body

    def _calibrate_avatar_stances(self) -> None:
        """Measure both stances, then place the avatar at stance 0."""
        self._stances = [self._stance_for_item(i)
                         for i in range(len(self.item_actors))]
        rot = np.asarray(self.avatar_init_rot, dtype=np.float64)
        self.avatar.reset(self._stances[0].copy(), rot.copy())
        for _ in range(30):
            self.step_sim()
        self.avatar_collided = False
        self.avatar_collision_log = []

    # ------------------------------------------------------------------
    # Motion boundary clipping.  Each Inspect clip covers "idle -> walk
    # near -> inspect -> walk back -> idle"; playing it whole makes the
    # avatar retreat ~1 m between the two items and walk all the way back
    # in (user feedback on v11).  Per item we slice the clip at frames
    # where the root has arrived near / not yet left the bench stance, so
    # between items she only takes the short glide.  The first item keeps
    # its walk-in (scene entry) and the last keeps its walk-back (exit
    # toward the park glide).
    # ------------------------------------------------------------------
    _CLIP_ARRIVE_RADIUS = 0.10

    def _root_offsets(self, motion_name: str) -> np.ndarray:
        """Rendered-root offset (world frame, relative to the avatar base)
        per clip frame: global_rot @ base_rot @ trans[f] — the transform
        AvatarRobot.get_global_xy applies to the playing motion's root."""
        md = self.avatar.motion_data[motion_name]
        trans = np.asarray(md["trans"], dtype=np.float64)
        R = (np.asarray(self.avatar_init_rot, dtype=np.float64)
             @ np.asarray(self.avatar.robot.base_rot, dtype=np.float64))
        return trans @ R.T

    def _prepare_item_clips(self) -> None:
        """Build the per-item runtime clips from the source motion: the
        forward clip is sliced at the *leave* frame (root still near the
        bench after the set-down withdrawal), and the reversed clip is
        that same slice played backward — it starts from the exact frame
        the forward clip froze on, so the inter-item glide (which holds
        that frame) is pose-continuous end to end.  In the reversed clip
        the original detach (put-down) becomes the pick-up and vice
        versa."""
        from dataclasses import replace
        spec = self._src_spec
        o = self._root_offsets(spec.name)
        n = len(o)
        e = n - 1
        for f in range(n - 1, spec.detach - 1, -1):
            if np.linalg.norm(o[f][:2] - o[spec.detach][:2]) < self._CLIP_ARRIVE_RADIUS:
                e = f
                break
        print(f"[qc] clip ({spec.name}): n={n} leave={e}, "
              f"root@0={o[0].round(2).tolist()} root@e={o[e].round(2).tolist()}",
              flush=True)
        md = self.avatar.motion_data[spec.name]
        fwd_name = f"{spec.name}_q0_{e}"
        if fwd_name not in self.avatar.motion_data:
            self.avatar.motion_data[fwd_name] = {
                k: v[0:e + 1].copy() for k, v in md.items()
            }
        rev_name = f"{spec.name}_rev_{e}"
        if rev_name not in self.avatar.motion_data:
            self.avatar.motion_data[rev_name] = {
                k: v[0:e + 1][::-1].copy() for k, v in md.items()
            }
        fwd_spec = replace(spec, name=fwd_name)
        rev_spec = replace(
            spec, name=rev_name,
            attach=e - spec.detach, detach=e - spec.attach,
        )
        self._item_motions = [
            fwd_spec if d == "fwd" else rev_spec for d in self._item_dirs
        ]
        self._frozen_clip_name = fwd_name

    # ------------------------------------------------------------------
    # Avatar pipeline: inspect -> glide -> inspect, inside patched step_sim
    # ------------------------------------------------------------------
    def _start_inspect(self, i: int) -> None:
        spec = self._item_motions[i]
        actor = self.item_actors[i][0]
        # The base must be at the item's stance when the clip starts (the
        # glide leaves it at stance + clip-start root offset).  No mesh
        # update — the next animation frame renders her at the glide-end
        # position, so no visible jump.
        self.avatar.reset(
            self._stances[i].copy(),
            np.asarray(self.avatar_init_rot, dtype=np.float64).copy(),
            update_mesh=False,
        )
        self.avatar.play_animation(
            spec.name,
            attach_obj=self._entity(actor),
            hand_id=spec.hand_id,
            attach_frame=self._scaled(spec.attach),
            detach_frame=self._scaled(spec.detach),
        )
        self._av_state = {"phase": "inspect", "idx": i, "glide_t": 0}
        print(f"[qc] avatar inspect {i} ({spec.name}) started", flush=True)

    def _apply_frozen_clip_end(self) -> None:
        """Re-render the forward clip's LAST frame at the avatar's current
        base — used during the glide so she holds the clip-end pose while
        sliding (a plain reset would snap her to the neutral pose; user
        feedback on v16).  Same mechanism as factory_kit_packing's
        ``_apply_clip_frame``."""
        motion = self.avatar.motion_modules.get(self._frozen_clip_name)
        if motion is None or not getattr(motion, "data", None):
            return
        self.avatar.robot.pose = motion.data[-1]
        self.avatar.robot.node_trans = motion.node_data[-1]
        self.avatar.robot.global_mat = motion.global_mat
        self.avatar.robot.global_mat_inv = motion.global_mat_inv
        self.avatar.robot.update()

    def _start_avatar_pipeline(self) -> None:
        self._start_inspect(0)
        original_step_sim = self.step_sim
        self._qc_original_step_sim = original_step_sim
        rot = np.asarray(self.avatar_init_rot, dtype=np.float64)

        def patched_step_sim():
            original_step_sim()
            st = self._av_state
            if st["phase"] == "inspect":
                i = st["idx"]
                spec = self._item_motions[i]
                motion = self.avatar.motion_modules.get(spec.name)
                at = int(getattr(motion, "at_frame", -1)) if motion is not None else -1
                if not self._inspected[i] and at >= self._scaled(spec.detach):
                    self._inspected[i] = True
                    print(f"[qc] item {i} set down (inspected)", flush=True)
                if self._inspected[i] and self.avatar.spare():
                    self._inspect_done[i] = True
                    print(f"[qc] avatar inspect {i} finished", flush=True)
                    if i + 1 < len(self.item_actors):
                        st["phase"] = "glide"
                        st["glide_t"] = 0
                        # Glide the BASE between the raw stances.  The
                        # frozen-frame render applies the clip's own root
                        # offset on top of the base, so the rendered root
                        # moves (stance0 + o[e]) -> (stance1 + o[e])
                        # continuously.  v17 added o[e] to these endpoints
                        # as well — double-counting it — which made her
                        # jump forward by ~0.3 m at glide start and back
                        # at glide end (user-reported sudden changes).
                        st["glide_from"] = self._stances[i].copy()
                        st["glide_to"] = self._stances[i + 1].copy()
                        st["glide_next"] = i + 1
                    else:
                        # Stay where the reversed clip ends — no park walk.
                        st["phase"] = "done"
                        print("[qc] avatar done (stays in place)", flush=True)
            elif st["phase"] == "glide":
                st["glide_t"] += 1
                t = st["glide_t"]
                n = int(self.GLIDE_STEPS)
                if t % self.GLIDE_UPDATE_EVERY == 0 or t >= n:
                    a = min(1.0, t / float(n))
                    pos = (1.0 - a) * st["glide_from"] + a * st["glide_to"]
                    self.avatar.reset(pos.copy(), rot.copy(), update_mesh=False)
                    self._apply_frozen_clip_end()
                if t >= n:
                    self._start_inspect(st["glide_next"])

        self.step_sim = patched_step_sim

    def _restore_step_sim(self) -> None:
        if hasattr(self, "_qc_original_step_sim"):
            self.step_sim = self._qc_original_step_sim
            del self._qc_original_step_sim

    # ------------------------------------------------------------------
    # Robot: avatar capsules as planner obstacles
    # ------------------------------------------------------------------
    def _avatar_obstacle_points(self):
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

    def _bench_sheet_points(self, hole_xys, hole_r: float = 0.16):
        """Two point layers just above the bench top with clearance holes
        at the given spots — (x, y) entries use ``hole_r``, (x, y, r)
        entries carry their own radius (the tote needs a wider one).

        Why: the planner pcd previously held ONLY avatar capsules, so RRT
        was free to route the arm low across the bench — physics then
        deflected it into a plow that bulldozed task items off the bench
        (smokes v8/v9).  The sheet forces every transit above the bench
        plane; the holes still admit the vertical pick/drop descents."""
        xs = np.arange(-0.85, 0.85 + 1e-9, 0.05)
        ys = np.arange(-0.475, 0.475 + 1e-9, 0.05)
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        xy = np.stack([X.ravel(), Y.ravel()], axis=1)
        keep = np.ones(len(xy), dtype=bool)
        # Always clear the robot's own mount: its base column intersects the
        # sheet, and a sheet point inside the base makes the START state
        # in-collision — mplib then rejects every plan (smoke v10).
        rx, ry = self.FACTORY_ROBOT_KWARGS["pos"][:2]
        keep &= np.hypot(xy[:, 0] - rx, xy[:, 1] - ry) > 0.28
        for hole in hole_xys:
            r = hole[2] if len(hole) > 2 else hole_r
            keep &= np.hypot(xy[:, 0] - hole[0], xy[:, 1] - hole[1]) > r
        xy = xy[keep]
        layers = [np.concatenate([xy, np.full((len(xy), 1), z)], axis=1)
                  for z in (0.78, 0.82)]
        return np.concatenate(layers, axis=0)

    def _other_item_points(self, skip_i: int):
        """Inflated AABB pcds of the not-yet-packed items other than the
        current target (their tops poke above the bench sheet)."""
        pts = []
        bx, by = self.BOX_XY
        for j, (actor, asset, mid, half_h, *_rest) in enumerate(self.item_actors):
            if j == skip_i:
                continue
            c = np.asarray(
                self._get_object_world_center(actor, asset, mid), dtype=np.float64,
            )
            if np.hypot(c[0] - bx, c[1] - by) < 0.20:
                continue  # already in/at the box — covered by the drop hole edge
            r = 0.07
            xs = np.arange(c[0] - r, c[0] + r + 1e-9, 0.04)
            ys = np.arange(c[1] - r, c[1] + r + 1e-9, 0.04)
            zs = np.arange(self.TABLE_TOP_Z, c[2] + half_h + 0.03, 0.04)
            X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
            pts.append(np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1))
        return pts

    def _human_zone_points(self, i_next: int):
        """Pcd over the avatar's KNOWN walk-in/stance region for item
        ``i_next`` — the off-bench strip she occupies while approaching and
        inspecting (y <= -0.44 keeps it clear of on-bench pick targets;
        over-bench protection comes from the bench sheet + live capsules).

        Why: obstacles are snapshotted when a pick starts, but the pick
        executes over ~1000+ sim steps during which the avatar walks in
        for the next inspection.  Her scripted trajectory is known, so
        block it ahead of time instead of reacting."""
        body = self._stances[i_next]
        actor, asset, mid, *_ = self.item_actors[i_next]
        item = np.asarray(
            self._get_object_world_center(actor, asset, mid), dtype=np.float64,
        )
        x_lo, x_hi = item[0] - 0.15, item[0] + 0.15
        y_lo = float(min(body[1], -0.85))
        y_hi = -0.44
        xs = np.arange(x_lo, x_hi + 1e-9, 0.06)
        ys = np.arange(y_lo, y_hi + 1e-9, 0.07)
        zs = np.arange(0.40, 1.45, 0.08)
        X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
        return np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)

    def _avatar_hands_off_bench(self) -> bool:
        """True when both of the worker's palms are retracted behind the
        bench-clear line (i.e. not reaching onto the worktop).  The two
        spawn bands are only ~0.12 m apart — too close to separate within
        the robot's reach — so a pick of one item while she reaches for the
        adjacent one grazes her hand even when she's clear of the target
        item itself (val seeds 2/3).  Gating picks on her hands being off
        the bench serializes only against her reach; she still moves
        continuously and never waits for the robot."""
        if getattr(self, "avatar", None) is None:
            return True
        try:
            for h in (0, 1):
                p = np.asarray(self.avatar.robot.get_palm_center(h), dtype=np.float64)
                if float(p[1]) >= self.BENCH_CLEAR_Y:
                    return False
        except Exception:
            return True
        return True

    def _avatar_min_xy_dist(self, xy) -> float:
        """Smallest xy distance from ``xy`` to any avatar collider capsule
        endpoint (covers hands AND forearms).  inf if no collider."""
        if self.avatar_collider is None:
            return float("inf")
        xy = np.asarray(xy, dtype=np.float64)[:2]
        best = float("inf")
        for _name, pa, pb, _r in self.avatar_collider.current_capsules():
            for p in (pa, pb):
                d = float(np.linalg.norm(np.asarray(p, dtype=np.float64)[:2] - xy))
                if d < best:
                    best = d
        return best

    def _refresh_planner_obstacles(self, arm_tag: str, block_next: int = None,
                                   pick_i: int = None) -> None:
        parts = []
        pts = self._avatar_obstacle_points()
        if pts is not None:
            parts.append(pts)
        holes = [tuple(self.BOX_XY) + (0.24,)]
        if pick_i is not None:
            actor, asset, mid, *_ = self.item_actors[pick_i]
            c = self._get_object_world_center(actor, asset, mid)
            holes.append((float(c[0]), float(c[1])))
            parts.extend(self._other_item_points(pick_i))
        parts.append(self._bench_sheet_points(holes))
        if block_next is not None and not self._inspect_done[block_next]:
            parts.append(self._human_zone_points(block_next))
        arm = self.robot.get_arm(arm_tag)
        try:
            arm.planner.update_obstacles(
                np.concatenate(parts, axis=0),
                resolution=self._AVATAR_OBSTACLE_RES,
            )
        except Exception:
            pass

    # ------------------------------------------------------------------
    # FK-gated moves.  Smoke v4/v5 showed the mixin's `_move_seeded` can
    # execute a wrong-IK-branch trajectory (post-exec xy error 0.5 m; the
    # arm visibly sprawls across the bench toward the parked worker)
    # because branch quality is only checked AFTER execution.  These
    # overrides verify the goal / plan-endpoint qpos with mplib's
    # pinocchio FK (no Genesis state is touched — avoids the known
    # set_qpos/PD desync) BEFORE anything is executed, and abort the move
    # instead of executing an unverified plan.
    # ------------------------------------------------------------------
    _FK_GATE_TOL = 0.06

    def _fk_gate_err(self, arm, arm_qpos, link_pose7):
        """World-frame position error of the planner move-group link FK at
        ``arm_qpos`` vs the link target.  None = FK unavailable."""
        try:
            planner = arm.planner
            pm = planner._mplib.robot.get_pinocchio_model()
            padded = planner._pad_qpos(np.asarray(arm_qpos, dtype=np.float64).ravel())
            pm.compute_forward_kinematics(padded)
            raw = pm.get_link_pose(int(planner._mplib.move_group_link_id))
            p = np.asarray(getattr(raw, "p", raw), dtype=np.float64).ravel()[:3]
            base = np.asarray(planner._base_pose, dtype=np.float64).ravel()
            world_p = base[:3] + t3d.quaternions.quat2mat(base[3:7]) @ p
            return float(np.linalg.norm(
                world_p - np.asarray(link_pose7[:3], dtype=np.float64)
            ))
        except Exception:
            return None

    def _move_pose_checked(self, link_pose7, arm_tag, attempts: int = 4):
        """``plan_path`` (mplib plan_pose) with the plan endpoint FK-verified
        BEFORE execution.  Returns None without moving when no verified plan
        is found, so the one-shot pick fails before the gripper closes."""
        arm = self.robot.get_arm(arm_tag)
        target = np.asarray(link_pose7, dtype=np.float64)
        best = None
        best_err = float("inf")
        for _ in range(attempts):
            res = arm.planner.plan_path(arm.get_arm_qpos(), target, log=False)
            if res is None or not res.success or res.position.size == 0:
                continue
            err = self._fk_gate_err(arm, res.position[-1][:arm.n_arm], target)
            if err is None:
                # FK unavailable — keep legacy behavior for this plan.
                best, best_err = res, -1.0
                break
            if err < best_err:
                best, best_err = res, err
            if err < 0.02:
                break
        if best is None:
            print("[qc] checked plan_pose: no plan found", flush=True)
            return None
        if best_err > self._FK_GATE_TOL:
            print(f"[qc] checked plan_pose: best endpoint FK err "
                  f"{best_err:.3f} m > {self._FK_GATE_TOL} — not executing", flush=True)
            return None
        self.execute_plan(best, arm_tag)
        return best

    def _move_seeded(self, link_pose7, arm_tag):
        """Mixin `_move_seeded` with two FK gates: the seeded-IK goal is
        verified before `plan_qpos`, and every plan_pose fallback goes
        through `_move_pose_checked` instead of `move_and_execute`."""
        from ..robot.franka_robot import to_numpy as _to_numpy
        arm = self.robot.get_arm(arm_tag)
        ik_fn = getattr(arm.entity, "inverse_kinematics", None)
        if ik_fn is None:
            return self._move_pose_checked(link_pose7, arm_tag)
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
        except Exception:
            return self._move_pose_checked(link_pose7, arm_tag)
        if qpos_sol is None:
            return self._move_pose_checked(link_pose7, arm_tag)
        arm_qpos_goal = _to_numpy(qpos_sol).ravel()[:arm.n_arm]

        # Gate 1: Genesis DLS IK has no convergence check — reject goals
        # whose FK is off-target instead of planning toward them.
        goal_err = self._fk_gate_err(arm, arm_qpos_goal, link_pose7)
        if goal_err is not None and goal_err > self._FK_GATE_TOL:
            print(f"[qc] seeded IK goal FK err {goal_err:.3f} m — "
                  f"using checked plan_pose", flush=True)
            return self._move_pose_checked(link_pose7, arm_tag)

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
        except Exception:
            return self._move_pose_checked(link_pose7, arm_tag)
        if result and result.get("status") == "Success":
            from ..planning.base import PlanResult
            pos = result["position"]
            vel = result.get("velocity", np.zeros_like(pos))
            self.execute_plan(PlanResult(True, pos, vel), arm_tag)
            ee_xy = np.array(arm.get_ee_pose()[:2], dtype=float)
            tgt_xy = np.array(link_pose7[:2], dtype=float)
            xy_err = float(np.linalg.norm(ee_xy - tgt_xy))
            if xy_err > 0.05:
                print(f"[qc] _move_seeded post-exec xy_err={xy_err:.3f} m, "
                      f"retry via checked plan_pose", flush=True)
                return self._move_pose_checked(link_pose7, arm_tag)
            return result
        return self._move_pose_checked(link_pose7, arm_tag)

    # ------------------------------------------------------------------
    # Plan-only variants (no execution) — the direct pick plans its whole
    # approach with these and only starts moving when every leg exists.
    # ------------------------------------------------------------------
    def _plan_pose_checked(self, link_pose7, arm_tag, attempts: int = 4,
                           use_seeded: bool = True):
        """Like `_move_seeded` + `_move_pose_checked` but WITHOUT executing:
        returns an FK-verified PlanResult to the pose, or None.
        ``use_seeded=False`` skips the deterministic seeded-IK goal and goes
        straight to plan_pose's random multistarts — used to get a
        DIFFERENT endpoint branch when the previous one was un-screwable."""
        from ..planning.base import PlanResult
        from ..robot.franka_robot import to_numpy as _to_numpy
        arm = self.robot.get_arm(arm_tag)
        target = np.asarray(link_pose7, dtype=np.float64)
        planner = arm.planner

        # Preferred: seeded IK goal (branch-continuous) + plan_qpos.
        ik_fn = getattr(arm.entity, "inverse_kinematics", None)
        if not use_seeded:
            ik_fn = None
        if ik_fn is not None:
            seed = getattr(arm, "_cached_target", None)
            if seed is None:
                seed = _to_numpy(arm.entity.get_qpos())
            seed = np.asarray(seed, dtype=np.float64).ravel()
            try:
                qpos_sol = ik_fn(
                    link=arm.ee_link,
                    pos=np.array(target[:3]),
                    quat=np.array(target[3:7]),
                    init_qpos=seed,
                )
            except Exception:
                qpos_sol = None
            if qpos_sol is not None:
                goal = _to_numpy(qpos_sol).ravel()[:arm.n_arm]
                err = self._fk_gate_err(arm, goal, target)
                if err is None or err <= self._FK_GATE_TOL:
                    try:
                        result = planner._mplib.plan_qpos(
                            goal_qposes=[planner._pad_qpos(goal)],
                            current_qpos=planner._pad_qpos(
                                np.asarray(arm.get_arm_qpos(), dtype=np.float64)
                            ),
                            time_step=1 / 250,
                            planning_time=getattr(planner, "_timeout", None) or 4,
                            rrt_range=0.3,
                            verbose=False,
                        )
                    except Exception:
                        result = None
                    if result and result.get("status") == "Success":
                        pos = result["position"]
                        vel = result.get("velocity", np.zeros_like(pos))
                        return PlanResult(True, pos, vel)

        # Fallback: plan_pose attempts, endpoint FK-verified before use.
        best, best_err = None, float("inf")
        for _ in range(attempts):
            res = planner.plan_path(arm.get_arm_qpos(), target, log=False)
            if res is None or not res.success or res.position.size == 0:
                continue
            err = self._fk_gate_err(arm, res.position[-1][:arm.n_arm], target)
            if err is None:
                return res
            if err < best_err:
                best, best_err = res, err
            if err < 0.02:
                break
        if best is not None and best_err <= self._FK_GATE_TOL:
            return best
        return None

    def _plan_screw_from(self, start_qpos, link_pose7, arm_tag):
        """plan_screw with an explicit start qpos (the previous leg's
        endpoint), so the vertical descent can be planned before the
        approach has been executed.  Returns PlanResult or None."""
        from ..planning.base import PlanResult
        arm = self.robot.get_arm(arm_tag)
        planner = arm.planner
        goal = planner._to_mplib_pose(np.asarray(link_pose7, dtype=np.float64))
        qpos = planner._pad_qpos(np.asarray(start_qpos, dtype=np.float64).ravel())
        for qpos_step in (0.1, 0.05, 0.02):
            try:
                result = planner._mplib.plan_screw(
                    goal_pose=goal,
                    current_qpos=qpos,
                    time_step=1 / 250,
                    qpos_step=qpos_step,
                    wrt_world=True,
                    verbose=False,
                )
            except Exception:
                continue
            if result and result.get("status") == "Success":
                pos = result["position"]
                vel = result.get("velocity", np.zeros_like(pos))
                return PlanResult(True, pos, vel)
        return None

    def _solve_ik(self, tcp_pose, arm_tag):
        """Carry-waypoint IK via seeded mplib (pinocchio CLIK,
        ``return_closest``) instead of the mixin's Genesis DLS.

        The recurring mid-carry slips (v24/v25 milk-box, at two different
        grip squeezes) were branch flips inside `_move_cartesian`'s
        waypoint loop — factory_line_feeding isolated the same failure as
        Genesis-IK-specific and immune to path shaping.  mplib's
        return_closest keeps each waypoint on the seed's branch; solutions
        that still jump >1 rad on any joint are rejected (waypoint skipped)
        rather than executed."""
        arm = self.robot.get_arm(arm_tag)
        planner = arm.planner
        if not hasattr(planner, "_mplib"):
            return super()._solve_ik(tcp_pose, arm_tag)
        from ..robot.franka_robot import _get_dof_idx
        from ..robot.franka_robot import to_numpy as _to_numpy
        seed_full = getattr(arm, "_cached_target", None)
        if seed_full is None:
            seed_full = _to_numpy(arm.entity.get_qpos())
        seed_full = np.asarray(seed_full, dtype=np.float64).ravel()
        seed_arm = []
        for j in arm.arm_joints:
            idx = _get_dof_idx(j) if j is not None else None
            seed_arm.append(float(seed_full[idx]) if idx is not None else 0.0)
        seed_arm = np.asarray(seed_arm, dtype=np.float64)

        pose7 = tcp_to_link_pose(tcp_pose, arm.tcp_offset).to_pose7()
        try:
            status, q = planner._mplib.IK(
                goal_pose=planner._to_mplib_pose(np.asarray(pose7)),
                start_qpos=planner._pad_qpos(seed_arm),
                return_closest=True,
                verbose=False,
            )
        except Exception:
            return super()._solve_ik(tcp_pose, arm_tag)
        if status != "Success" or q is None:
            return None
        q = np.asarray(q, dtype=np.float64).ravel()[:arm.n_arm]
        if float(np.max(np.abs(q - seed_arm[:arm.n_arm]))) > 1.0:
            return None   # branch flip — skip this waypoint, don't execute it
        return q

    # Keep the carry segment at least this far from the arm's base column:
    # a straight Cartesian carry that cuts closer forces the per-waypoint
    # IK to fold the arm against itself and corkscrew the wrist (the
    # "moves randomly for a while" on v20's first place — its straight
    # path passed 0.33 m from the base).
    TRANSIT_BASE_CLEARANCE = 0.45

    def _transit_segments(self, a: np.ndarray, b: np.ndarray):
        """Split the carry a->b at a bowed waypoint when the straight
        segment passes inside TRANSIT_BASE_CLEARANCE of the robot base:
        the segment's closest point to the base is pushed radially out to
        the clearance radius (same height)."""
        base = np.asarray(self.FACTORY_ROBOT_KWARGS["pos"][:2], dtype=np.float64)
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        d = b[:2] - a[:2]
        denom = float(d @ d)
        t = 0.5 if denom < 1e-9 else float(np.clip((base - a[:2]) @ d / denom, 0.0, 1.0))
        closest = a[:2] + t * d
        v = closest - base
        dist = float(np.linalg.norm(v))
        if dist >= self.TRANSIT_BASE_CLEARANCE or t in (0.0, 1.0):
            return [(a, b)]
        out_dir = v / dist if dist > 1e-9 else d[::-1] * np.array([-1.0, 1.0])
        w_xy = base + out_dir * self.TRANSIT_BASE_CLEARANCE
        w = np.array([w_xy[0], w_xy[1], a[2]], dtype=np.float64)
        print(f"[qc] carry bowed around base: closest {dist:.2f} m -> "
              f"via {w[:2].round(2).tolist()}", flush=True)
        return [(a, w), (w, b)]

    # ------------------------------------------------------------------
    # Direct pick: plan the COMPLETE approach (one move to 15 cm above the
    # item + the vertical screw to the grasp) before any motion.  If a leg
    # cannot be planned (e.g. the worker is still beside the item), the
    # robot stays put and `_pack_item`'s stall loop waits — no more
    # half-descended retreats or staged hovering at transport height
    # (user feedback on v18).
    # ------------------------------------------------------------------
    def _pick_once(self, actor, asset_name: str, arm_tag: str,
                   close_value: float, below_center: float,
                   drop_xy, tcp_yaw_deg: float = 0.0) -> bool:
        from ..robot.franka_robot import to_numpy as _to_numpy
        arm = self.robot.get_arm(arm_tag)
        tcp_offset = arm.tcp_offset

        arm._cached_target = np.asarray(
            _to_numpy(arm.entity.get_qpos()), dtype=float,
        ).ravel().copy()

        transport_z = self.TABLE_TOP_Z + self.TRANSPORT_Z_ABOVE_TABLE
        drop_z = self.TABLE_TOP_Z + self.TRAY_RIM_Z + self.DROP_HOVER_Z_ABOVE_TRAY

        obj_center = self._get_object_world_center(actor, asset_name, 0)
        grasp_pos = obj_center.copy()
        grasp_pos[2] -= below_center
        grasp_pos[2] = max(
            grasp_pos[2], self.TABLE_TOP_Z + self.GRASP_TABLE_Z_MARGIN,
        )
        # Plan approach + descent as a pair.  The descent (plan_screw) is
        # branch-sensitive: from some approach endpoints it cannot hold the
        # straight line, and the seeded endpoint is deterministic (v22:
        # 8 identical "no descent plan" stalls).  v23 tried unseeded
        # plan_pose branches for variety — the wrist-wrapped configuration
        # it picked then flung the box mid-carry.  Instead, vary the
        # TASK-SPACE approach in controlled ways, all seeded: different
        # pre-grasp heights and the 180°-flipped wrist (same closing axis,
        # different j7) — each gives a nearby, well-behaved IK solution
        # with a different descent feasibility.
        variants = [
            (self.PRE_GRASP_HEIGHT, tcp_yaw_deg),
            (self.PRE_GRASP_HEIGHT + 0.04, tcp_yaw_deg),
            (self.PRE_GRASP_HEIGHT - 0.04, tcp_yaw_deg),
            (self.PRE_GRASP_HEIGHT, tcp_yaw_deg + 180.0),
            (self.PRE_GRASP_HEIGHT + 0.04, tcp_yaw_deg + 180.0),
        ]
        plan_a = plan_b = grasp_link = None
        for k, (pre_h, yaw_k) in enumerate(variants):
            pre_pos = obj_center.copy()
            pre_pos[2] += pre_h
            pre_link_k = tcp_to_link_pose(
                self._rotated_top_down_tcp(pre_pos, yaw_k), tcp_offset,
            )
            grasp_link_k = tcp_to_link_pose(
                self._rotated_top_down_tcp(grasp_pos, yaw_k), tcp_offset,
            )
            cand_a = self._plan_pose_checked(
                pre_link_k.to_pose7(), arm_tag, attempts=2,
            )
            if cand_a is None:
                continue
            cand_b = self._plan_screw_from(
                cand_a.position[-1][:arm.n_arm], grasp_link_k.to_pose7(), arm_tag,
            )
            if cand_b is not None:
                plan_a, plan_b, grasp_link = cand_a, cand_b, grasp_link_k
                if k > 0:
                    print(f"[qc] pick: planned via variant {k} "
                          f"(pre_h={pre_h:.2f}, yaw={yaw_k:.0f})", flush=True)
                break
        if plan_a is None or plan_b is None:
            print("[qc] pick: no approach+descent plan pair — holding position",
                  flush=True)
            return False

        self.open_gripper(arm_tag)
        self.execute_plan(plan_a, arm_tag)
        # The descent is a vertical screw that ignores the avatar pcd, so if
        # the worker's arm is reaching across this pick column (she's
        # fetching the NEXT item nearby) the wrist can graze her forearm
        # (v29: a 0.0009 m link7-vs-r_forearm touch).  Hold above the item
        # until her capsules clear the column, then lower.  She moves
        # continuously, so this just waits out the crossing.
        cleared = 0
        while cleared < self.DESCENT_CLEAR_CAP_STEPS:
            if self._avatar_min_xy_dist(grasp_pos[:2]) >= self.DESCENT_CLEAR_XY:
                break
            self.step_sim()
            cleared += 1
        self.execute_plan(plan_b, arm_tag)
        for _ in range(self.GRASP_SETTLE_STEPS):
            self.step_sim()

        # Small seeded correction if PD drift left the EE off the grasp xy.
        ee_pos = np.array(arm.get_ee_pose()[:3], dtype=float)
        xy_err = float(np.linalg.norm(ee_pos[:2] - grasp_pos[:2]))
        if xy_err > self.GRASP_ALIGNMENT_XY_TOL:
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

        z_after = float(self._get_object_world_center(actor, asset_name, 0)[2])
        if z_after - z_before < self.LIFT_DZ_MIN:
            self.open_gripper(arm_tag)
            for _ in range(self.LIFT_FAIL_RELEASE_SETTLE_STEPS):
                self.step_sim()
            return False

        # Re-orient the wrist to the canonical yaw-0 top-down config while
        # the item is lifted clear.  The carry (`_move_cartesian`) commands
        # yaw-0 top-down waypoints; from a wrist-flipped grasp (the +180
        # descent variant) those poses are a 180 deg wrist change that is
        # UNREACHABLE at the raised carry height — every carry waypoint
        # IK-failed and dragged the box off (v27 milk-box).  A single
        # in-place re-orient up high (the box just spins 180 deg in hand,
        # harmless for a symmetric package) unifies the wrist so the carry
        # starts from the same config that already works for normal grasps.
        ee_q = np.array(arm.get_ee_pose()[3:7], dtype=float)
        canon_q = t3d.quaternions.mat2quat(self._R_DOWN)
        if abs(float(np.dot(ee_q, canon_q))) < 0.985:   # >~10 deg off canonical
            canon_link = tcp_to_link_pose(self._top_down_tcp(lift_pos), tcp_offset)
            self._move_seeded(canon_link.to_pose7(), arm_tag)
            for _ in range(self.POST_LIFT_SETTLE_STEPS):
                self.step_sim()

        above_drop = np.array([drop_xy[0], drop_xy[1], transport_z])
        for seg_a, seg_b in self._transit_segments(lift_pos, above_drop):
            self._move_cartesian(
                seg_a, seg_b, arm_tag,
                n_steps=self.TRANSIT_CARTESIAN_STEPS,
                sim_per_step=self.TRANSIT_SIM_PER_STEP,
            )
        for _ in range(self.POST_TRANSIT_SETTLE_STEPS):
            self.step_sim()

        ee_xy = np.array(arm.get_ee_pose()[:2], dtype=float)
        obj_xy = self._get_object_world_center(actor, asset_name, 0)[:2]
        if float(np.linalg.norm(ee_xy - obj_xy)) > self.TRANSIT_SLIP_XY_TOL:
            return False

        drop_pos = np.array([drop_xy[0], drop_xy[1], drop_z])
        self._move_screw(drop_pos, arm_tag)
        for _ in range(self.DROP_SETTLE_STEPS):
            self.step_sim()

        self.open_gripper(arm_tag)
        for _ in range(self.RELEASE_SETTLE_STEPS):
            self.step_sim()
        # Verify the item actually landed in the carton.  An undetected
        # mid-carry slip can release it short of the box while every other
        # gate passed (val seed 2); report failure without re-grasping.
        if not self._item_in_box(self._get_object_world_center(actor, asset_name, 0)):
            print(f"[qc] pick: {asset_name} not in box after release",
                  flush=True)
            return False
        return True

    # ------------------------------------------------------------------
    # Rollout
    # ------------------------------------------------------------------
    def _wait_inspect_done(self, i: int) -> bool:
        steps = 0
        while not self._inspect_done[i] and steps < self.WAIT_INSPECT_CAP_STEPS:
            self.step_sim()
            steps += 1
        if not self._inspect_done[i]:
            print(f"[qc] timed out waiting for inspect {i}", flush=True)
            return False
        for _ in range(self.INSPECTED_SETTLE_STEPS):
            self.step_sim()
        return True

    def _settled_grasp_yaw(self, actor, asset: str, mid: int):
        """TCP yaw closing across the item's narrowest horizontal face,
        computed from its settled orientation.

        The avatar's set-down releases the item with the palm's yaw, so it
        rests rotated relative to spawn.  A fixed yaw-0 grasp then closes
        across the rotated box's corners and shoots it away (v6: the
        coffee-box was fired off the bench).  Returns (yaw_deg, width_m).
        """
        from ..utils import to_numpy as _to_np
        ent = self._entity(actor)
        q = _to_np(ent.get_quat()).ravel()[:4]
        R = t3d.quaternions.quat2mat(q)
        scale, extents = _read_model_data(asset, mid)
        dims = extents * scale
        best = None
        for axis_i in range(3):
            d = R[:, axis_i]
            if abs(float(d[2])) > 0.5:
                continue  # mostly vertical — not a closing direction
            width = float(dims[axis_i])
            phi = float(np.degrees(np.arctan2(d[1], d[0])))
            if best is None or width < best[0]:
                best = (width, phi)
        if best is None:
            return 0.0, None
        width, phi = best
        # `_rotated_top_down_tcp`: yaw 0 closes along world Y, +90 along
        # world X (TCP z points down, so world yaw runs opposite).
        yaw_deg = 90.0 - phi
        while yaw_deg > 90.0:
            yaw_deg -= 180.0
        while yaw_deg < -90.0:
            yaw_deg += 180.0
        return float(yaw_deg), width

    def _pack_item(self, i: int, arm_tag: str) -> bool:
        """Wait for a clear corridor, then pick item i into its box slot once."""
        actor, asset, mid, _half_h, close, (dx, dy), _yaw = self.item_actors[i]
        bx, by = self.BOX_XY
        block_next = i + 1 if i + 1 < len(self.item_actors) else None
        stalls = 0
        # Generous stall budget: when the worker is still inspecting the
        # ADJACENT next item, her arm can block this pick's plan (or be in
        # collision range).  Rather than give up, wait — she moves
        # continuously and parks clear of the bench after her inspection
        # (~9 stalls), at which point the pick plans cleanly and safely.
        # This keeps concurrency when the geometry allows and only serializes
        # when she is actually in the way (validation seeds 0/2/3).
        while stalls < self.MAX_PICK_STALLS:
            # Pick-start clearance gate: never begin a pick while the worker's
            # arm is near this item.  A pick that DOES plan while she is close
            # still risks a hand/forearm collision (seed 2) or a cramped grasp
            # that mis-carries (seed 3); the no-plan case (seed 0) is also
            # covered.  Wait it out — she retracts/parks after inspecting the
            # adjacent item, then this clears.
            item_xy = self._get_object_world_center(actor, asset, mid)[:2]
            if (not self._avatar_hands_off_bench()
                    or self._avatar_min_xy_dist(item_xy) < self.PICK_START_CLEAR_XY):
                stalls += 1
                for _ in range(300):
                    self.step_sim()
                continue
            break
        else:
            print(f"[qc] robot pack of {asset} timed out waiting for clearance",
                  flush=True)
            return False

        self._refresh_planner_obstacles(arm_tag, block_next=block_next, pick_i=i)
        yaw, width = self._settled_grasp_yaw(actor, asset, mid)
        if width is not None and width > 0.075:
            print(f"[qc] WARNING {asset} narrowest horizontal face "
                  f"{width:.3f} m exceeds gripper opening", flush=True)
        print(f"[qc] pack {asset} one-shot (stalls {stalls}): "
              f"yaw={yaw:.0f} deg "
              f"width={width if width is None else round(width, 3)} "
              f"close={close:.2f}", flush=True)
        try:
            if self._pick_once(
                actor, asset, arm_tag,
                close_value=close,
                below_center=self.ROBOT_BELOW_CENTER,
                drop_xy=(bx + dx, by + dy),
                tcp_yaw_deg=yaw,
            ):
                return True
        except Exception as exc:
            print(f"[qc] robot pack of {asset} raised: {exc}", flush=True)
        print(f"[qc] robot pack of {asset} failed", flush=True)
        return False

    def play_once(self) -> bool:
        arm_tag = "right"
        self._boost_finger_pd(arm_tag)
        with self.suppress_recording():  # pre-task settle, excluded from recordings
            for _ in range(self.INITIAL_SETTLE_STEPS):
                self.step_sim()

        if self.avatar is None:
            # No-human fallback: skip the gate, just pack.
            self._inspected = [True] * len(self.item_actors)
            self._inspect_done = [True] * len(self.item_actors)
            return all(self._pack_item(i, arm_tag)
                       for i in range(len(self.item_actors)))

        self._start_avatar_pipeline()
        try:
            ok = True
            for i in range(len(self.item_actors)):
                if not self._wait_inspect_done(i):
                    ok = False
                    break
                ok = self._pack_item(i, arm_tag) and ok

            tail = 0
            while (self._av_state.get("phase") != "done"
                   and tail < self.AVATAR_FINISH_WAIT_STEPS):
                self.step_sim()
                tail += 1
        finally:
            self._restore_step_sim()
        for _ in range(self.FINAL_SETTLE_STEPS):
            self.step_sim()
        for i, (actor, asset, mid, *_rest) in enumerate(self.item_actors):
            c = self._get_object_world_center(actor, asset, mid)
            print(f"[qc] final {asset}: center={np.round(c, 3).tolist()} "
                  f"in_box={self._item_in_box(c)} "
                  f"rel_box={np.round(c[:2] - np.asarray(self.BOX_XY), 3).tolist()}",
                  flush=True)
        return ok

    # ------------------------------------------------------------------
    # Success: gated (all inspected) + geometric (all in box) + safe.
    # ------------------------------------------------------------------
    def _item_in_box(self, pos: np.ndarray) -> bool:
        bx, by = self.BOX_XY
        ihx, ihy = self.BOX_INNER_HALF
        z0 = self.TABLE_TOP_Z
        return (
            abs(float(pos[0]) - bx) <= ihx
            and abs(float(pos[1]) - by) <= ihy
            and z0 - 0.01 <= float(pos[2]) <= z0 + self.SUCCESS_Z_MAX_ABOVE_TABLE
        )

    def check_success(self) -> bool:
        if not self.plan_success:
            return False
        if (self.config.get("track_avatar_collision", True)
                and getattr(self, "avatar_collided", False)):
            return False
        if not all(self._inspected) or not all(self._inspect_done):
            return False
        for actor, asset, mid, *_rest in self.item_actors:
            if not self._item_in_box(self._get_object_world_center(actor, asset, mid)):
                return False
        return True

    def evaluate(self) -> dict:
        metrics = super().evaluate()
        per_item = {}
        for i, (actor, asset, mid, *_rest) in enumerate(self.item_actors):
            c = self._get_object_world_center(actor, asset, mid)
            per_item[asset] = {
                "inspected": bool(self._inspected[i]),
                "inspect_done": bool(self._inspect_done[i]),
                "in_box": bool(self._item_in_box(c)),
            }
        metrics["items"] = per_item
        metrics["avatar_collided"] = bool(getattr(self, "avatar_collided", False))
        return metrics
