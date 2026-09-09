"""Avatar bone-cylinder collider rig.

Builds a set of cylinder primitives that hug the avatar's skeleton
(one cylinder per major body segment).  Each step, world positions of
two skeleton bones (bone_a → bone_b) are read from the skin; the
cylinder is translated to the midpoint and rotated to the bone axis.

Intended uses:
  * visualization — see where the avatar "is" for debug video
  * collision detection against robot links (downstream;
    the segments expose start/end/radius for analytic checks)

Cylinder height is fixed at construction (Genesis primitives can't
resize), so the nominal length per segment is baked in.  Bones don't
stretch during animation, so this matches actual skin extents.

Typical usage (inside a task's scene setup, BEFORE scene.build):

    from envs.avatar.collider import AvatarCollider
    self.avatar_collider = AvatarCollider(self.scene, self.avatar.robot,
                                          visualization=True)
    # ... scene.build(), avatar.reset() ...
    # then each sim step:
    self.avatar_collider.update()
"""
import numpy as np
import genesis as gs
import transforms3d as t3d

from ..utils import to_numpy


def _avatar_material():
    avatar_cls = getattr(gs.materials, "Avatar", None)
    if avatar_cls is not None:
        return avatar_cls()
    return gs.materials.Rigid()


# (segment_name, bone_a, bone_b, nominal_length_m, radius_m)
#
# Lengths measured on the default Adrian_Keller.glb avatar in the idle
# T-pose (scripts/measure_avatar_bones.py).  All current avatars share
# the Mixamo adult proportions so these hold across the pool; override
# via the `segments` kwarg if a non-adult avatar is ever introduced.
#
# Radii picked by body part so the cylinder hugs the skin without
# looking comically thick — thicker around torso/thighs, thin on arms
# and hands.
def _hand_segments(side: str):
    """Palm + 15 phalanx cylinders for one hand.

    Palm is approximated by a single cylinder along wrist → middle-knuckle
    (captures the palm's main axis; radius is tuned to palm thickness).
    Each of the 5 fingers is split into 3 phalanx cylinders so curled
    fingers (fist) stay covered — a straight base-to-tip capsule would
    shortcut through the palm when the fingertips bend inward.

    Lengths are measured from the Adrian_Keller T-pose; radii shrink from
    the proximal phalanx (base-of-finger) out to the distal phalanx
    (fingertip) to hug each segment's real thickness.
    """
    p = "l" if side == "Left" else "r"
    out = []
    # Per-finger lengths (measured).  Thumb is indexed like the others
    # (Thumb1→Thumb2→Thumb3→Thumb4) so the generic segment loop works.
    finger_lens = {
        "Thumb":  [0.0333, 0.0285, 0.0372],
        "Index":  [0.0403, 0.0234, 0.0219],
        "Middle": [0.0422, 0.0302, 0.0189],
        "Ring":   [0.0387, 0.0264, 0.0196],
        "Pinky":  [0.0233, 0.0186, 0.0192],
    }
    # Phalanx radii (prox → mid → dist).  Thumb is slightly chubbier.
    finger_radii = {
        "Thumb":  [0.013, 0.012, 0.010],
        "Index":  [0.011, 0.010, 0.008],
        "Middle": [0.011, 0.010, 0.008],
        "Ring":   [0.010, 0.009, 0.008],
        "Pinky":  [0.009, 0.008, 0.007],
    }
    seg_tags = ("prox", "mid", "dist")
    for finger in ("Thumb", "Index", "Middle", "Ring", "Pinky"):
        for i, tag in enumerate(seg_tags, start=1):
            out.append((
                f"{p}_{finger.lower()}_{tag}",
                f"{side}Hand{finger}{i}",
                f"{side}Hand{finger}{i + 1}",
                finger_lens[finger][i - 1],
                finger_radii[finger][i - 1],
            ))
    return out


DEFAULT_SEGMENTS = [
    # name          bone_a            bone_b                 length   radius
    ("torso_lower", "Hips",           "Spine1",              0.277,   0.14),
    ("torso_upper", "Spine1",         "Neck",                0.267,   0.14),
    ("head",        "Neck",           "HeadTop_End",         0.266,   0.10),
    ("l_upper_arm", "LeftArm",        "LeftForeArm",         0.271,   0.055),
    ("l_forearm",   "LeftForeArm",    "LeftHand",            0.255,   0.050),
    ("r_upper_arm", "RightArm",       "RightForeArm",        0.271,   0.055),
    ("r_forearm",   "RightForeArm",   "RightHand",           0.255,   0.050),
    ("l_thigh",     "LeftUpLeg",      "LeftLeg",             0.427,   0.090),
    ("l_shin",      "LeftLeg",        "LeftFoot",            0.432,   0.065),
    ("r_thigh",     "RightUpLeg",     "RightLeg",            0.427,   0.090),
    ("r_shin",      "RightLeg",       "RightFoot",           0.432,   0.065),
    *_hand_segments("Left"),
    *_hand_segments("Right"),
]


# Analytic palm shape: an oriented box in the hand frame.  The old palm
# capsule used radius 4.2 cm around wrist -> middle-knuckle, which made the
# palm as thick sideways and normal-to-palm as it was long.  For robot/avatar
# contact gating, the palm is better modeled as broad and thin.
PALM_BOX_HALF_EXTENTS = np.array([0.045, 0.049, 0.018], dtype=np.float64)


def _axis_align_quat_wxyz(axis: np.ndarray) -> np.ndarray:
    """Quat (w,x,y,z) that rotates cylinder's local +Z onto `axis` (unit)."""
    z = np.array([0.0, 0.0, 1.0])
    d = float(np.dot(z, axis))
    if d > 0.999999:
        return np.array([1.0, 0.0, 0.0, 0.0])
    if d < -0.999999:
        # 180° flip around any axis ⟂ Z; pick X
        return np.array([0.0, 1.0, 0.0, 0.0])
    perp = np.cross(z, axis)
    perp /= np.linalg.norm(perp) + 1e-12
    angle = np.arccos(np.clip(d, -1.0, 1.0))
    s = np.sin(angle * 0.5)
    return np.array([np.cos(angle * 0.5), perp[0] * s, perp[1] * s, perp[2] * s])


class AvatarCollider:
    """Bone-cylinder collider rig for an AvatarController.robot skin.

    Parameters
    ----------
    scene : gs.Scene
        Scene under construction.  Must still be pre-`build()` so cylinder
        entities can be added.
    avatar_robot : AvatarRobot
        Typically `AvatarController.robot`.  Its `skin` is queried for bone
        world positions each frame.
    segments : list of tuple, optional
        Override `DEFAULT_SEGMENTS`.  Same schema:
        `(name, bone_a, bone_b, length, radius)`.
    color : tuple
        RGB(A) fill color for the cylinder.  Default semi-transparent red.
    visualization : bool
        Whether cylinders are rendered.  False makes them collision-only
        (future use).  Currently only True is exercised.
    collision : bool
        Whether cylinders participate in physics collision.  Defaults
        False — this rig is visual / analytic only; using the avatar
        material keeps the entity kinematic-but-non-colliding regardless.
    """

    def __init__(
        self,
        scene,
        avatar_robot,
        segments=None,
        color=(1.0, 0.25, 0.25, 0.55),
        visualization=True,
        collision=False,
    ):
        self.scene = scene
        self.robot = avatar_robot
        self.segments_def = list(segments if segments is not None else DEFAULT_SEGMENTS)

        # "Virtual rig" mode: no visualization and no collision means we
        # only need the segment *definitions* (bone pairs + radii) for
        # analytic collision queries via `current_capsules()`.  We skip
        # creating Genesis entities entirely — which also avoids the
        # ~187 ms/step cost of set_pos/set_quat on 43 cylinders.  This
        # is the default path when the collision checker is on without
        # --show-collider.
        self._virtual = (not visualization) and (not collision)

        self.entities = []  # [(name, bone_a, bone_b, radius, length, entity_or_None)]
        self.palm_boxes = [
            ("l_palm", 0, PALM_BOX_HALF_EXTENTS.copy()),
            ("r_palm", 1, PALM_BOX_HALF_EXTENTS.copy()),
        ]
        self.palm_box_entities = []  # [(name, hand_id, half_extents, entity_or_None)]

        if self._virtual:
            for name, bone_a, bone_b, length, radius in self.segments_def:
                self.entities.append(
                    (name, bone_a, bone_b, float(radius), float(length), None)
                )
            for name, hand_id, half_extents in self.palm_boxes:
                self.palm_box_entities.append((name, hand_id, half_extents, None))
            return

        mat_avatar = _avatar_material()
        surface = gs.surfaces.Default(color=color)
        for name, bone_a, bone_b, length, radius in self.segments_def:
            entity = scene.add_entity(
                material=mat_avatar,
                morph=gs.morphs.Cylinder(
                    radius=float(radius),
                    height=float(length),
                    pos=(0.0, 0.0, 5.0),   # offstage until first update()
                    visualization=visualization,
                    collision=collision,
                    fixed=False,
                ),
                surface=surface,
            )
            self.entities.append((name, bone_a, bone_b, float(radius),
                                  float(length), entity))
        for name, hand_id, half_extents in self.palm_boxes:
            entity = scene.add_entity(
                material=mat_avatar,
                morph=gs.morphs.Box(
                    size=tuple(float(2.0 * x) for x in half_extents),
                    pos=(0.0, 0.0, 5.0),
                    visualization=visualization,
                    collision=collision,
                    fixed=False,
                ),
                surface=surface,
            )
            self.palm_box_entities.append((name, hand_id, half_extents, entity))

    # ------------------------------------------------------------------
    # Per-step update
    # ------------------------------------------------------------------

    def _bone_pos(self, name: str):
        """World position of bone `name`, or None if lookup fails."""
        try:
            raw = self.robot.skin.get_global_translation(name)[0]
        except Exception:
            return None
        p = to_numpy(raw).ravel()[:3]
        return p.astype(np.float64)

    def update(self):
        """Align every cylinder to its current bone pair.  Call each sim step.

        No-op if the avatar has no skin attached or if the rig is in
        "virtual" mode (no Genesis entities to move).
        """
        if self.robot.skin is None or self._virtual:
            return
        for name, bone_a, bone_b, _r, _L, entity in self.entities:
            pa = self._bone_pos(bone_a)
            pb = self._bone_pos(bone_b)
            if pa is None or pb is None:
                continue
            axis = pb - pa
            norm = float(np.linalg.norm(axis))
            if norm < 1e-6:
                continue
            axis /= norm
            center = 0.5 * (pa + pb)
            quat = _axis_align_quat_wxyz(axis)
            entity.set_pos(center.astype(float))
            entity.set_quat(quat.astype(float))
        for _name, hand_id, half_extents, entity in self.palm_box_entities:
            if entity is None:
                continue
            try:
                hand_pos, axes = self.robot._get_hand_frame(hand_id)
            except Exception:
                continue
            center = np.asarray(hand_pos, dtype=np.float64) + axes[:, 1] * float(half_extents[1])
            quat = t3d.quaternions.mat2quat(np.asarray(axes, dtype=np.float64))
            entity.set_pos(center.astype(float))
            entity.set_quat(quat.astype(float))

    # ------------------------------------------------------------------
    # Queries (for downstream collision detection)
    # ------------------------------------------------------------------

    def current_capsules(self):
        """Return a list of `(name, start, end, radius)` for the current
        pose — useful for analytic point-vs-capsule distance checks
        against robot links without touching Genesis contact data."""
        out = []
        for name, bone_a, bone_b, r, _L, _e in self.entities:
            pa = self._bone_pos(bone_a)
            pb = self._bone_pos(bone_b)
            if pa is None or pb is None:
                continue
            out.append((name, pa, pb, r))
        return out

    def current_boxes(self):
        """Return oriented palm boxes as `(name, center, axes, half_extents)`.

        `axes` is a 3x3 matrix whose columns are the local +X/+Y/+Z axes:
        across-palm, wrist-to-fingers, and palm-normal respectively.
        """
        if self.robot.skin is None:
            return []
        out = []
        for name, hand_id, half_extents in self.palm_boxes:
            try:
                hand_pos, axes = self.robot._get_hand_frame(hand_id)
            except Exception:
                continue
            center = np.asarray(hand_pos, dtype=np.float64) + axes[:, 1] * float(half_extents[1])
            out.append((
                name,
                center.astype(np.float64),
                np.asarray(axes, dtype=np.float64),
                np.asarray(half_extents, dtype=np.float64),
            ))
        return out
