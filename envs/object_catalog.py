"""Central target-object catalog and category registry.

This module is the **single source of truth** for the manipulable *target*
objects used by benchmark tasks (the object the robot picks/places — not the
random tabletop clutter, which lives in ``utils.table_object_pool``).

Two concepts, one mechanism:

- **white-list** — a task declares an explicit list of object keys it accepts
  (often size 1, e.g. the shoe task only accepts ``041_shoe``).
- **category** — a *named, shared* list of object keys that several tasks can
  reference (e.g. ``"cabinet_drawer_items"``).

Both are just lists of object keys; a white-list is an inline list and a
category is a named list resolved from ``CATEGORIES``.  A task sets the
``OBJECT_SET`` class attribute to either form and calls
``BaseTask.resolve_target_object()`` to pick one (forced via
``config["object_name"]`` or sampled).

Per-object metadata (spawn orientation, friction, gripper close value, model
variant, display name) lives here keyed by object key, so the same object
behaves consistently across tasks.  Tasks needing a genuinely task-local
tweak override at the call site; they do not redefine the object.

Model variants: a few assets are used at different ``model_id`` in different
tasks (e.g. ``035_apple`` at 0 vs 1).  An object-set entry may pin a variant
with ``"035_apple@1"``; otherwise the catalog's default ``model_id`` applies.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional


# Standard "y-up GLB → world-z-up" upright quaternion (w, x, y, z).
_Q_UPRIGHT = (0.70710678, 0.70710678, 0.0, 0.0)
# Stapler rests on its side; this quat stands it in a graspable pose.
_Q_STAPLER = (0.5, 0.5, 0.5, 0.5)


@dataclass(frozen=True)
class ObjectEntry:
    """Canonical metadata for one target object.

    ``kind == "mesh"``  → a GLB/URDF asset under ``assets/objects/<object_id>``.
    ``kind == "primitive"`` → a procedural primitive (e.g. dump_bin's cube);
    the owning task supplies the size/color, the catalog only records identity
    + physics defaults.
    """

    key: str                                   # registry key + cfg object_name handle
    kind: str = "mesh"                          # "mesh" | "primitive"
    object_id: str = ""                         # asset dir; "" for primitives
    model_id: int = 0
    display_name: str = ""                      # human/instruction name
    spawn_quat: tuple = _Q_UPRIGHT
    friction: float = 4.0
    scale: Optional[float] = None               # None → read model_data{N}.json
    grasp_robot_types: tuple = ("franka",)

    # --- manipulation hints (canonical per-object tweaks; None → task default) ---
    close_value: Optional[float] = None         # gripper close target
    grasp_z_offset: Optional[float] = None      # +z world bias applied to the grasp
    topdown_grasp_z_offset: Optional[float] = None  # +z bias for shared Cartesian picker
    topdown_approach_xy_offset: Optional[tuple] = None  # controller tracking compensation
    topdown_preopen_steps: Optional[int] = None  # wait for full jaw opening before approach
    topdown_close_value: Optional[float] = None  # task-independent Cartesian grip force
    pre_dist: Optional[float] = None            # pre-grasp standoff distance override (m)
    spawn_z_clearance: Optional[float] = None   # table-clearance gap at spawn (m)
    grasp_categories: Optional[tuple] = None    # restrict grasp annotations (e.g. ("handle",))
    finger_pd: Optional[dict] = None            # {"kp":.., "kv":.., "force":..} gripper PD profile
    prim_half: Optional[float] = None           # primitive box half-extent (m), kind=="primitive"

    def as_spec(self) -> dict:
        """Return the legacy per-task ``spec`` dict shape so existing
        downstream spawn code (which reads ``spec["object_id"]`` etc.) works
        unchanged after migration.  Optional manipulation hints are included
        only when set, under the same keys tasks already use."""
        spec = {
            "object_id": self.object_id if self.kind == "mesh" else self.key,
            "model_id": self.model_id,
            "quat": self.spawn_quat,
            "friction": self.friction,
            "kind": self.kind,
        }
        for field in (
            "close_value", "grasp_z_offset", "topdown_grasp_z_offset",
            "topdown_approach_xy_offset", "topdown_preopen_steps",
            "topdown_close_value", "pre_dist",
            "spawn_z_clearance", "grasp_categories", "finger_pd", "prim_half",
        ):
            val = getattr(self, field)
            if val is not None:
                spec[field] = val
        return spec


# ---------------------------------------------------------------------------
# Catalog — keyed by object key (== object_id for meshes, a slug for primitives)
# ---------------------------------------------------------------------------

_ENTRIES = (
    # --- cabinet drawer items (put_object_cabinet {,_assist,_interrupt,_neutral}) ---
    # grasp_z_offset is read by the _assist variant; harmless elsewhere.
    ObjectEntry("048_stapler", object_id="048_stapler", model_id=0,
                display_name="stapler", spawn_quat=_Q_STAPLER, friction=4.0,
                grasp_z_offset=0.02),
    ObjectEntry("023_tissue-box", object_id="023_tissue-box", model_id=0,
                display_name="tissue box", spawn_quat=_Q_UPRIGHT, friction=4.0,
                grasp_z_offset=-0.01,
                topdown_approach_xy_offset=(0.015, 0.010),
                topdown_preopen_steps=80),
    ObjectEntry("031_jam-jar", object_id="031_jam-jar", model_id=0,
                display_name="jam jar", spawn_quat=_Q_UPRIGHT, friction=4.0,
                grasp_z_offset=0.02, topdown_grasp_z_offset=0.0,
                # Runtime pre-grasp tracking compensation centers the wide
                # body across IK branches, so no fixed task-specific XY bias.
                topdown_approach_xy_offset=(0.0, 0.0),
                topdown_preopen_steps=80, topdown_close_value=0.55),
    ObjectEntry("071_can", object_id="071_can", model_id=0,
                display_name="can", spawn_quat=_Q_UPRIGHT, friction=5.0,
                close_value=0.40),

    # --- dump-bin items (dump_bin) ---
    ObjectEntry("cube", kind="primitive", display_name="block", friction=4.0,
                prim_half=0.020),
    ObjectEntry("038_milk-box", object_id="038_milk-box", model_id=0,
                display_name="milk box", spawn_quat=_Q_UPRIGHT, friction=4.0,
                close_value=0.40),

    # --- single-object task targets (white-lists of size 1) ---
    # friction/grasp tuning for these lives in the task class (e.g.
    # SHOE_FRICTION); the catalog is the identity + model-variant registry.
    ObjectEntry("041_shoe", object_id="041_shoe", model_id=0,
                display_name="shoe", spawn_quat=_Q_UPRIGHT, friction=5.0),
    ObjectEntry("002_bowl", object_id="002_bowl", model_id=1,
                display_name="bowl", spawn_quat=_Q_UPRIGHT, friction=5.0),
    ObjectEntry("075_bread", object_id="075_bread", model_id=0,
                display_name="bread", spawn_quat=_Q_UPRIGHT, friction=4.0),

    # --- categorize / multi-object task targets ---
    # Used at different model variants per task (e.g. apple @0 in
    # burger/skillet/take-from-human, @1 in categorize); object-set tokens pin
    # the variant with "035_apple@1".  Spawn quat / friction for these tasks
    # live in the task class, so the catalog records identity + default model.
    # close_value is read by cabinet; dump-bin applies task-local overrides
    # where its longer transport needs a different grip.
    ObjectEntry("035_apple", object_id="035_apple", model_id=0,
                display_name="apple", spawn_quat=_Q_UPRIGHT, friction=4.0,
                close_value=0.35),
    # friction 5.0 (was 4.0): the pool pins model_1 (token "086_woodenblock@1") —
    # a 5x5x7.13 cm block that spawns STANDING.  At friction 4.0 under Genesis
    # 1.0.0 the top-down cage was marginal (lifted only intermittently, ejected
    # on perturbation).  At the friction cap the grasp lifts reliably (3/3) and
    # holds through transit.  close_value 0.25 is the sweep optimum: 0.15
    # over-squeezes and ejects, 0.50 is too loose to lift.  Pick path is the
    # TopDownPickPlaceMixin radius grasp (NOT the grasp YAML), shared by
    # put_object_cabinet {,_assist,_interrupt,_neutral}.
    ObjectEntry("086_woodenblock", object_id="086_woodenblock", model_id=0,
                display_name="wooden block", spawn_quat=_Q_UPRIGHT, friction=5.0,
                close_value=0.25, topdown_grasp_z_offset=0.030),
    # 100_seal: the pool pins model_1 (catalog token "100_seal@1"), a compact
    # aspect-1.4 block-like figurine (~6.5 x 5.5 x 7.7 cm) — the cleanest variant
    # for a top-down jaw (model_2 was a thin elongated body that gripped the
    # tapered nose/tail and never lifted).  close_value 0.60 cages the ~5.5 cm
    # body.  GRASP + LIFT are solved (0% -> 100% lift) via the live-AABB
    # grasp-centre override in the task: model_data's y-up center+quat math sits
    # ~4 cm off under the Genesis-1.0.0 z-up GLB load, so the radius grasp used to
    # close on air.  Drawer PLACEMENT remains the shared drawer-floor tunneling
    # limit (the over-opened drawer's thin floor lets objects fall through; hits
    # rubikscube/woodenblock too) — see docs/0623/fix_seal.txt.
    ObjectEntry("100_seal", object_id="100_seal", model_id=0,
                display_name="seal", spawn_quat=_Q_UPRIGHT, friction=4.0,
                close_value=0.60,
                topdown_approach_xy_offset=(0.0, 0.010),
                topdown_preopen_steps=80),
    ObjectEntry("073_rubikscube", object_id="073_rubikscube", model_id=0,
                display_name="rubik's cube", spawn_quat=_Q_UPRIGHT, friction=4.0),

    # --- factory products ---
    # These box-shaped products share one pool across inspect+pack and kit
    # packing.  Task-local grasp/placement tuning still lives with the task.
    ObjectEntry("113_coffee-box", object_id="113_coffee-box", model_id=0,
                display_name="coffee box", spawn_quat=_Q_UPRIGHT, friction=4.0,
                topdown_grasp_z_offset=0.030),
    ObjectEntry("112_tea-box", object_id="112_tea-box", model_id=0,
                display_name="tea box", spawn_quat=_Q_UPRIGHT, friction=4.0,
                topdown_approach_xy_offset=(0.0, 0.010),
                topdown_preopen_steps=80),

    # --- other single-target tasks ---
    ObjectEntry("039_mug", object_id="039_mug", model_id=0,
                display_name="mug", spawn_quat=_Q_UPRIGHT, friction=4.0),
    ObjectEntry("029_olive-oil", object_id="029_olive-oil", model_id=0,
                display_name="oil bottle", spawn_quat=_Q_UPRIGHT, friction=4.0),
)

CATALOG: dict[str, ObjectEntry] = {e.key: e for e in _ENTRIES}


# ---------------------------------------------------------------------------
# Categories — named shared object-key lists
# ---------------------------------------------------------------------------

CATEGORIES: dict[str, list[str]] = {
    # Single shared pool for the tabletop pick-and-place families
    # (put_object_cabinet, dump_bin, categorize) and ALL their
    # base/assist/interrupt/neutral variants — variants within a family are
    # identical.  Model variants are pinned to the ones with validated Franka
    # grasps (apple/woodenblock/seal/rubikscube use the categorize-tuned
    # variants).  ``cube`` is a primitive (handled via radius-based grasp).
    "tabletop_pick_pool": [
        "048_stapler", "031_jam-jar", "cube", "038_milk-box",
        "035_apple@1", "086_woodenblock@1", "100_seal@1", "073_rubikscube@1",
    ],
    # Small objects that can be offered by a human or delivered to one using
    # the same top-down-pick affordance.  Keep this deliberately conservative;
    # additions must work in both transfer directions.
    "human_transfer_pool": [
        "035_apple@1", "073_rubikscube@1",
    ],
    # Shared by factory_inspect_pack and factory_kit_packing.  The pool is the
    # union of their previously separate validated product lists.
    "factory_product_pool": [
        "113_coffee-box", "038_milk-box", "112_tea-box",
        "023_tissue-box", "073_rubikscube", "086_woodenblock",
    ],
}


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------

def _parse_key(token: str) -> tuple[str, Optional[int]]:
    """Split an object-set token into ``(key, model_id_override)``.

    ``"035_apple@1"`` → ``("035_apple", 1)``; ``"041_shoe"`` → ``("041_shoe", None)``.
    """
    token = str(token)
    if "@" in token:
        key, _, mid = token.partition("@")
        return key, int(mid)
    return token, None


def resolve_object_set(object_set) -> list[str]:
    """Resolve a task ``OBJECT_SET`` to a flat list of object-set tokens.

    ``object_set`` is either a category name (str present in ``CATEGORIES``),
    a single object key (str), or an explicit list of keys/tokens.
    """
    if object_set is None:
        raise ValueError("object_set is None — task must declare OBJECT_SET")
    if isinstance(object_set, str):
        if object_set in CATEGORIES:
            return list(CATEGORIES[object_set])
        return [object_set]
    tokens: list[str] = []
    for item in object_set:
        item = str(item)
        if item in CATEGORIES:
            tokens.extend(CATEGORIES[item])
        else:
            tokens.append(item)
    if not tokens:
        raise ValueError(f"object_set {object_set!r} resolved to an empty list")
    return tokens


def object_set_pairs(object_set) -> list:
    """Resolve an object set to a list of ``(object_id, model_id)`` tuples.

    Convenience for multi-object tasks (e.g. categorize) whose spawn list is a
    sequence of ``(object_id, model_id)`` pairs.
    """
    pairs = []
    for token in resolve_object_set(object_set):
        e = get_entry(token)
        pairs.append((e.object_id or e.key, e.model_id))
    return pairs


def get_entry(token: str) -> ObjectEntry:
    """Return the catalog ``ObjectEntry`` for a token, applying any
    ``@model_id`` variant override."""
    key, model_override = _parse_key(token)
    if key not in CATALOG:
        raise KeyError(
            f"object key {key!r} not in catalog; known keys: {sorted(CATALOG)}"
        )
    entry = CATALOG[key]
    if model_override is not None and model_override != entry.model_id:
        entry = replace(entry, model_id=model_override)
    return entry
