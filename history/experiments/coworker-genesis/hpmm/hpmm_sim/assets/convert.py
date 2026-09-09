"""Post-processing of the exported RoboCasa MJCF, before Genesis ever sees it.

The MJCF that ``robosuite`` hands us is geometrically faithful but carries a few defaults
that belong to *its* task harness rather than to ours. Fixing them in the XML — rather
than after import — keeps every downstream consumer (Genesis, the voxelizer, the video
tools) looking at the same scene.

Currently the one fix that matters: robosuite parks the robot at ``(10, 10)``, far outside
the kitchen, because its own placement sampler would move it later. Genesis imports the
whole file as a single entity whose ``robot0_base`` is a *fixed* body, so the robot can
only be relocated through the mobile-base slide joints. Driving those by twelve metres
would burn the very dofs navigation needs and put the base joints nowhere near their
neutral range, so the base pose is rewritten in the XML instead and the mobile joints are
left free to do their actual job.
"""

from __future__ import annotations

import pathlib
import xml.etree.ElementTree as ET

import numpy as np

ROBOT_BASE_BODY = "robot0_base"

#: MuJoCo's legal range for these material attributes. RoboCasa ships values outside it —
#: layout 34 has stool materials with ``shininess="-1"``, and 22 materials with
#: ``specular`` at 2.0 or 3.0. MuJoCo itself tolerates them; Genesis does not, because it
#: converts glossiness to roughness as ``(2 / (glossiness + 2)) ** 0.25`` with
#: ``glossiness = shininess * 128``. A negative shininess makes that a fourth root of a
#: negative number, which in Python is a **complex** number, and the scene then fails to
#: build on a texture validation error whose message ("Invalid attribute 'color[0]' ...
#: Got (0.2509+0.2509j)") points nowhere near the offending material.
MATERIAL_RANGES = {
    "shininess": (0.0, 1.0),
    "specular": (0.0, 1.0),
    "reflectance": (0.0, 1.0),
    "emission": (0.0, 1.0),
}


def find_body(root: ET.Element, name: str) -> ET.Element:
    for body in root.iter("body"):
        if body.get("name") == name:
            return body
    raise KeyError(f"body {name!r} not found in the MJCF")


def yaw_to_quat(yaw: float) -> tuple[float, float, float, float]:
    """Rotation about +z as a MuJoCo ``w x y z`` quaternion."""
    return (float(np.cos(yaw / 2)), 0.0, 0.0, float(np.sin(yaw / 2)))


def place_robot(mjcf_path, out_path, pos, yaw: float | None = None) -> pathlib.Path:
    """Rewrite the robot base pose in an exported RoboCasa MJCF.

    Parameters
    ----------
    pos : (3,)
        World position for ``robot0_base``. z should stay 0 — the wheeled base sits on
        the floor and the torso column carries the height.
    yaw : float, optional
        Heading in radians about +z. The export's own default is +pi/2, which points the
        base's local ``+x`` (the direction ``joint_mobile_forward`` slides along) at world
        ``+y``, i.e. into the counter run of the galley. Left unchanged when omitted.

    Returns
    -------
    The written path. Mesh and texture references in the export are absolute, so the copy
    resolves its assets from anywhere.
    """
    tree = ET.parse(str(mjcf_path))
    base = find_body(tree.getroot(), ROBOT_BASE_BODY)
    base.set("pos", " ".join(f"{float(v):.6g}" for v in pos))
    if yaw is not None:
        base.set("quat", " ".join(f"{v:.9g}" for v in yaw_to_quat(yaw)))
    out_path = pathlib.Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(str(out_path))
    return out_path


def sanitize_materials(root: ET.Element) -> dict[str, int]:
    """Clamp out-of-range material attributes in place. Returns a per-attribute count.

    Only values outside :data:`MATERIAL_RANGES` are touched, so a well-formed scene passes
    through byte-identical.
    """
    fixed: dict[str, int] = {}
    for material in root.iter("material"):
        for attr, (lo, hi) in MATERIAL_RANGES.items():
            raw = material.get(attr)
            if raw is None:
                continue
            values = [float(v) for v in raw.split()]
            clamped = [min(max(v, lo), hi) for v in values]
            if clamped != values:
                material.set(attr, " ".join(f"{v:.6g}" for v in clamped))
                fixed[attr] = fixed.get(attr, 0) + 1
    return fixed


def sanitize_materials_text(xml: str) -> tuple[str, dict[str, int]]:
    """:func:`sanitize_materials` for an in-memory MJCF string."""
    root = ET.fromstring(xml)
    fixed = sanitize_materials(root)
    return ET.tostring(root, encoding="unicode"), fixed


def robot_base_pose(mjcf_path) -> tuple[np.ndarray, np.ndarray]:
    """Read back ``(pos, quat)`` of the robot base, for checking a conversion."""
    base = find_body(ET.parse(str(mjcf_path)).getroot(), ROBOT_BASE_BODY)
    pos = np.array([float(v) for v in base.get("pos", "0 0 0").split()])
    quat = np.array([float(v) for v in base.get("quat", "1 0 0 0").split()])
    return pos, quat


__all__ = ["place_robot", "robot_base_pose", "find_body", "yaw_to_quat", "ROBOT_BASE_BODY",
           "sanitize_materials", "sanitize_materials_text", "MATERIAL_RANGES"]
