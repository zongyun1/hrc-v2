"""Pool-driven inspect-motion mixin for interrupt tasks.

Interrupt tasks (`dump_bin_interrupt`, `categorize_interrupt`,
`place_burger_fries_interrupt`, `place_bread_in_basket_interrupt`,
`cont_bowl_cutlery_fetch`) used to hardcode three class attrs:

    INSPECT_MOTION_NAME   = "Inspect1"
    INSPECT_ATTACH_FRAME  = 34
    INSPECT_DETACH_FRAME  = 187

This mixin replaces that with an ``INSPECT_POOL`` of
``{"name", "attach", "detach", optional "slow"}`` entries.  The resolver
runs once per ``reset()`` (after ``BaseTask.reset`` seeds ``np.random``)
and writes those same three attribute names — plus optionally
``AVATAR_MOTION_SLOW`` — onto the instance, shadowing the class-level
defaults.  Existing read sites (`self.INSPECT_MOTION_NAME` etc.) keep
working unchanged.

When ``config["randomize_inspect"]`` is True and the pool has more than
one entry, the resolver samples uniformly via ``np.random.randint`` —
deterministic per ``reset(seed=…)``.  Otherwise the first entry wins, so
existing tasks that don't override ``INSPECT_POOL`` keep playing
``"Inspect1"`` exactly as before.  The old ``"Inspect"`` name remains a
compatibility alias for existing configs.

MRO note: place ``InspectMotionMixin`` before the task / other mixins so
``super().reset`` chains correctly:

    class DumpBinInterrupt(InspectMotionMixin, DumpBinBigBin):
        INSPECT_POOL = None  # → INSPECT_POOL_DEFAULT
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class InspectMotionSpec:
    """Motion-local metadata for one inspect animation.

    ``attach`` / ``detach`` are source-frame indices in the original motion.
    If ``clip`` is set, the mixin installs a runtime subclip and converts
    attach/detach to clip-local frame indices before task code sees them.
    """

    name: str
    attach: int
    detach: int
    hand_id: int = 1
    clip: tuple[int, int] | None = None
    slow: float | None = None
    attach_object: bool = True

    def as_pool_entry(self) -> dict:
        entry = {
            "name": self.name,
            "attach": int(self.attach),
            "detach": int(self.detach),
            "hand_id": int(self.hand_id),
            "attach_object": bool(self.attach_object),
        }
        if self.clip is not None:
            entry["clip"] = (int(self.clip[0]), int(self.clip[1]))
        if self.slow is not None:
            entry["slow"] = float(self.slow)
        return entry


INSPECT_MOTION_SPECS = {
    "Inspect1": InspectMotionSpec("Inspect1", attach=34, detach=187),
    "Inspect2": InspectMotionSpec("Inspect2", attach=31, detach=84),
    "Inspect3": InspectMotionSpec("Inspect3", attach=33, detach=164),
    # Source clip from retarget/raw_blender/inspect.blend.  Runtime tasks play
    # the clipped submotion but metadata stays in original source-frame units.
    "Inspect4": InspectMotionSpec("Inspect4", attach=61, detach=173, clip=(25, 196)),
    # Source clip from retarget/raw_blender/inspect5.blend.
    "Inspect5": InspectMotionSpec("Inspect5", attach=55, detach=133, clip=(15, 170)),
    # Source clip from retarget/raw_blender/inspect6.blend.
    "Inspect6": InspectMotionSpec("Inspect6", attach=94, detach=184, clip=(43, 212)),
    # Source clip from retarget/raw_blender/inspect7.blend.
    "Inspect7": InspectMotionSpec("Inspect7", attach=52, detach=196, clip=(29, 229)),
    # Source clip from retarget/raw_blender/inspect10.blend.
    "Inspect8": InspectMotionSpec(
        "Inspect8", attach=74, detach=185, clip=(19, 231),
    ),
    # Source clip from retarget/raw_blender/inspect9.blend.
    "Inspect9": InspectMotionSpec(
        "Inspect9", attach=69, detach=196, clip=(21, 247),
    ),
    # Source clip from retarget/raw_blender/inspect10.blend.
    "Inspect10": InspectMotionSpec(
        "Inspect10", attach=36, detach=184, clip=(17, 225),
    ),
}

INSPECT_MOTION_ALIASES = {
    "Inspect": "Inspect1",
}

STANDARD_INSPECT_POOL = tuple(
    INSPECT_MOTION_SPECS[name].as_pool_entry()
    for name in (
        "Inspect1", "Inspect2", "Inspect3", "Inspect4", "Inspect5",
        "Inspect6", "Inspect7", "Inspect8", "Inspect9", "Inspect10",
    )
)


# First entry is the legacy default — picked when randomization is off so
# existing scripted-flow tasks behave unchanged.  Add new motions to the
# end so randomized runs see them; keep frame indices in *original* (pre-
# slow) units — the read sites multiply by ``self.AVATAR_MOTION_SLOW``.
# Optional ``slow`` per-entry overrides the task's class-level
# ``AVATAR_MOTION_SLOW`` for that motion only.  ``attach=False`` means the
# avatar reaches the calibrated contact frame but does not kinematically
# attach the object; this is used by touch/withdraw interrupt variants.
INSPECT_POOL_DEFAULT = [
    *STANDARD_INSPECT_POOL,
    {"name": "Inspect1_touch_hold_reverse", "attach": 34, "detach": 34, "attach_object": False},
    {"name": "Inspect2_touch_hold_reverse", "attach": 31, "detach": 31, "attach_object": False},
]


class InspectMotionMixin:
    """See module docstring."""

    INSPECT_POOL = None  # Override per-task; None → INSPECT_POOL_DEFAULT.

    def reset(self, seed: int = 0):
        obs = super().reset(seed=seed)
        self._resolve_inspect_motion()
        return obs

    def _resolve_inspect_motion(self) -> None:
        pool = self.INSPECT_POOL or INSPECT_POOL_DEFAULT
        if not pool:
            return
        forced_name = self.config.get("inspect_motion_name")
        if forced_name:
            forced_name = INSPECT_MOTION_ALIASES.get(forced_name, forced_name)
            matches = [e for e in pool if e["name"] == forced_name]
            if not matches:
                raise ValueError(
                    f"config['inspect_motion_name']={forced_name!r} "
                    f"not in pool: {[e['name'] for e in pool]}"
                )
            entry = matches[0]
            print(f"[inspect] forced motion={entry['name']} (config override)")
        elif self.config.get("randomize_inspect", False) and len(pool) > 1:
            entry = pool[int(np.random.randint(len(pool)))]
            print(f"[inspect] randomized motion={entry['name']}")
        else:
            entry = pool[0]
        entry = self._prepare_inspect_motion_entry(dict(entry))
        self.INSPECT_MOTION_NAME  = entry["name"]
        self.INSPECT_ATTACH_FRAME = int(entry["attach"])
        self.INSPECT_DETACH_FRAME = int(entry["detach"])
        self.AVATAR_HAND_ID = int(entry.get("hand_id", getattr(self, "AVATAR_HAND_ID", 1)))
        self.INSPECT_ATTACH_OBJECT = bool(entry.get("attach_object", True))
        if "slow" in entry:
            self.AVATAR_MOTION_SLOW = float(entry["slow"])

    def _prepare_inspect_motion_entry(self, entry: dict) -> dict:
        """Install a configured source-frame subclip and return runtime frames."""
        clip = entry.get("clip")
        if clip is None:
            return entry
        if getattr(self, "avatar", None) is None:
            return entry
        src_name = str(entry["name"])
        start, end = (int(clip[0]), int(clip[1]))
        if src_name not in self.avatar.motion_data:
            return entry
        md = self.avatar.motion_data[src_name]
        n = int(md["trans"].shape[0])
        if start < 0 or end < start or end >= n:
            raise ValueError(
                f"Invalid inspect clip {start}..{end} for {src_name} length {n}"
            )
        dst_name = f"{src_name}_clip_{start}_{end}"
        if dst_name not in self.avatar.motion_data:
            sl = slice(start, end + 1)
            clipped = {k: v[sl].copy() for k, v in md.items()}
            self.avatar.motion_data[dst_name] = clipped
        entry["name"] = dst_name
        entry["source_name"] = src_name
        entry["source_attach"] = int(entry["attach"])
        entry["source_detach"] = int(entry["detach"])
        entry["attach"] = int(entry["attach"]) - start
        entry["detach"] = int(entry["detach"]) - start
        if entry["attach"] < 0 or entry["detach"] < entry["attach"]:
            raise ValueError(
                f"Inspect attach/detach must fall inside clip for {src_name}: {entry}"
            )
        return entry

    @property
    def _inspect_attach_frame_scaled(self) -> int:
        return int(round(self.INSPECT_ATTACH_FRAME * self.AVATAR_MOTION_SLOW))

    @property
    def _inspect_detach_frame_scaled(self) -> int:
        return int(round(self.INSPECT_DETACH_FRAME * self.AVATAR_MOTION_SLOW))

    @property
    def _attach_frame_scaled(self) -> int:
        return self._inspect_attach_frame_scaled

    @property
    def _detach_frame_scaled(self) -> int:
        return self._inspect_detach_frame_scaled

    def inspect_avatar_pos_for_object(self, object_pos, extra_z: float | None = None):
        """Return avatar root position that gives this motion a fixed object pose.

        This preserves the motion-specific avatar-object relative placement:
        the object can be placed first or the avatar can be placed first, but
        one side should be derived from this fixed relation.
        """
        palm_default = self._dryrun_capture_palm()
        if palm_default is None:
            return None
        target = np.asarray(object_pos, dtype=np.float64).ravel()[:3].copy()
        shift = target - np.asarray(palm_default, dtype=np.float64).ravel()[:3]
        shift[2] += float(
            getattr(self, "AVATAR_EXTRA_Z", 0.0)
            if extra_z is None
            else extra_z
        )
        return np.asarray(self.avatar_init_pos, dtype=np.float64).copy() + shift

    def reset_avatar_for_inspect_object(
        self,
        object_pos,
        extra_z: float | None = None,
        settle_steps: int | None = None,
    ) -> bool:
        """Place the avatar using the fixed relation for the selected motion."""
        if getattr(self, "avatar", None) is None:
            return False
        avatar_pos = self.inspect_avatar_pos_for_object(object_pos, extra_z=extra_z)
        if avatar_pos is None:
            return False
        idle_pos = self._set_inspect_motion_root_pos(avatar_pos)
        self.avatar.reset(
            idle_pos.copy(),
            np.asarray(self.avatar_init_rot, dtype=np.float64).copy(),
        )
        n = int(
            getattr(self, "CALIBRATION_SETTLE_STEPS", 0)
            if settle_steps is None
            else settle_steps
        )
        for _ in range(max(0, n)):
            self.step_sim()
        self.avatar_collided = False
        self.avatar_collision_log = []
        return True

    def _inspect_idle_position_for_motion_root(
        self,
        motion_root_pos,
        motion_frame: int = 0,
    ) -> np.ndarray:
        """Return a global idle root whose mesh matches a motion frame.

        This adjusts only the avatar's global idle position.  It never edits
        the authored motion arrays.
        """
        if getattr(self, "avatar", None) is None:
            return np.asarray(motion_root_pos, dtype=np.float64).copy()
        md = self.avatar.motion_data.get(self.INSPECT_MOTION_NAME)
        idle = self.avatar.motion_data.get("idle")
        if md is None or idle is None:
            return np.asarray(motion_root_pos, dtype=np.float64).copy()
        n = int(md["trans"].shape[0])
        frame = int(np.clip(int(motion_frame), 0, n - 1))
        motion_trans = np.asarray(md["trans"][frame], dtype=np.float64)
        idle_trans = np.asarray(idle["trans"][0], dtype=np.float64)
        rot = np.asarray(self.avatar_init_rot, dtype=np.float64)
        base_rot = np.asarray(self.avatar.robot.base_rot, dtype=np.float64)
        return (
            np.asarray(motion_root_pos, dtype=np.float64).copy()
            + rot @ base_rot @ (motion_trans - idle_trans)
        )

    def _set_inspect_motion_root_pos(self, motion_root_pos) -> np.ndarray:
        """Store the root used for playback and return the matching idle root."""
        root = np.asarray(motion_root_pos, dtype=np.float64).copy()
        self._inspect_motion_root_pos = root
        idle_start = self._inspect_idle_position_for_motion_root(root, motion_frame=0)
        self._inspect_idle_start_pos = idle_start
        return idle_start

    def _play_inspect_motion(self, attach_obj=None, hand_id=1,
                             attach_frame=None, detach_frame=None, **kwargs):
        """Play the selected inspect motion, honoring touch-only entries."""
        if not getattr(self, "INSPECT_ATTACH_OBJECT", True):
            attach_obj = None
            attach_frame = None
            detach_frame = None
        motion_root_pos = getattr(self, "_inspect_motion_root_pos", None)
        if motion_root_pos is not None:
            self.avatar.reset(
                np.asarray(motion_root_pos, dtype=np.float64).copy(),
                np.asarray(self.avatar_init_rot, dtype=np.float64).copy(),
            )
        return self.avatar.play_animation(
            self.INSPECT_MOTION_NAME,
            attach_obj=attach_obj,
            hand_id=hand_id,
            attach_frame=attach_frame,
            detach_frame=detach_frame,
            **kwargs,
        )
