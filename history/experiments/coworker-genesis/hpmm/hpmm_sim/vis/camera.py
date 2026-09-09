"""Camera placement that actually sees inside a RoboCasa room.

RoboCasa kitchens are sealed boxes: opaque walls on every side, and **no ceiling**. A
camera positioned by offsetting from whatever is interesting — the end-effector, the mean
of a walk — lands outside the room as often as not, and renders the *exterior* of a brick
wall. That failure is silent: the video encodes fine, the numbers are unaffected, and the
result is a flat grey rectangle.

Two placements work, and this module provides both:

* :func:`look_into_room` — above the wall tops, horizontally inside the room's own
  footprint, angled down. Derived from the scene's geometry rather than hand-tuned per
  layout, because a magic number that happens to suit layout 1 is a bug waiting for
  layout 34.
* :func:`mount_camera` — reproduce one of RoboCasa's own ``cam_configs``, which are mounted
  on the robot and therefore always framed on the workspace. Preferred whenever the scene
  came with them.

The footprint is taken from the *room's own* links, with two exclusions that both come
from real failures:

* **The robot is not part of the room.** robosuite parks an unplaced robot at ``(10, 10)``,
  far outside the kitchen. Its ~20 links are 12% of layout 1's total, enough to survive any
  reasonable percentile, and they drag the computed room centre out through the back wall.
* **Stray markers.** RoboCasa leaves 2 cm placement-sample geoms at z ~ 10, hence
  percentiles rather than a raw min/max even after the robot is dropped.
"""

from __future__ import annotations

import numpy as np


def room_vis_options(ambient: float = 0.35):
    """Visual options that make a sealed room legible.

    Genesis defaults to ``ambient_light=(0.1, 0.1, 0.1)`` and a single directional light.
    That is fine in the open, but a RoboCasa kitchen is a box whose walls shadow most of
    the floor, and an overview shot comes out too dark to read. Lifting the ambient term
    and adding a second light from the opposite side fills those shadows without washing
    the scene out.
    """
    import genesis as gs

    return gs.options.VisOptions(
        ambient_light=(ambient, ambient, ambient),
        lights=[
            {"type": "directional", "dir": (-1, -1, -1), "color": (1.0, 1.0, 1.0), "intensity": 5.0},
            {"type": "directional", "dir": (1, 0.5, -1), "color": (1.0, 1.0, 1.0), "intensity": 3.0},
        ],
    )


def quat_to_R(quat) -> np.ndarray:
    """w-x-y-z quaternion -> 3x3 rotation matrix."""
    w, x, y, z = np.asarray(quat, dtype=np.float64)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


#: Link-name fragments that mark something as robot rather than room.
ROBOT_LINK_MARKERS = ("robot0", "mobilebase", "gripper", "manipulator", "eef_target")


def _link_positions(entity) -> tuple[list[str], np.ndarray]:
    names = [link.name for link in entity.links]
    return names, np.asarray(entity.get_links_pos().cpu()).reshape(-1, 3)


def room_bounds(entity, percentile: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    """Interior ``(lo, hi)`` of the room, in world coordinates.

    RoboCasa names its shell explicitly — ``wall_left_room_main``, ``wall_front_room_main``,
    ``floor_room_main`` and friends — so the walls are read directly rather than inferred.
    Inference gets this wrong in two different ways: link *origins* only cover the
    furniture (all of it against the back wall, so the open floor vanishes and the camera
    is placed among the counters), while geom bounding *spheres* are enormous for a flat
    wall plate and push the extent metres outside the building.

    Falls back to a percentile over non-robot links if a scene has no such shell.
    """
    names, positions = _link_positions(entity)
    shell = [i for i, n in enumerate(names)
             if (n.startswith("wall_") or n.startswith("floor_")) and "room" in n]
    if shell:
        p = positions[shell]
        lo, hi = p.min(axis=0), p.max(axis=0)
        # Walls are centred on their own height, so the top is twice the centre; the floor
        # link sits at the floor. That gives the vertical span a camera has to clear.
        wall_z = max((positions[i][2] for i in shell if names[i].startswith("wall_")), default=hi[2])
        return np.array([lo[0], lo[1], min(lo[2], 0.0)]), np.array([hi[0], hi[1], 2.0 * wall_z])

    keep = [i for i, n in enumerate(names)
            if not any(marker in n for marker in ROBOT_LINK_MARKERS)]
    p = positions[keep] if keep else positions
    return np.percentile(p, percentile, axis=0), np.percentile(p, 100.0 - percentile, axis=0)


def open_floor_position(entity, lookat, lo, hi, margin: float = 0.5,
                        near: float = 1.6, far: float = 4.5, grid: int = 41) -> np.ndarray:
    """The spot inside the room with the most clearance from furniture.

    "Back away from the subject" has no good direction in a galley kitchen: the counters
    line one wall and the subject is usually already near the middle, so any fixed rule
    eventually reverses into a cabinet. Instead, search the footprint for the point that is
    furthest from any fixture — that is the open floor, which is exactly where a person
    would stand to watch — subject to a sensible standoff from what we want to look at.
    """
    names, positions = _link_positions(entity)
    shell_or_robot = tuple(ROBOT_LINK_MARKERS) + ("wall_", "floor_")
    fixtures = np.array([p[:2] for n, p in zip(names, positions)
                         if not any(m in n for m in shell_or_robot)])

    xs = np.linspace(lo[0] + margin, hi[0] - margin, grid)
    ys = np.linspace(lo[1] + margin, hi[1] - margin, grid)
    candidates = np.stack(np.meshgrid(xs, ys, indexing="ij"), axis=-1).reshape(-1, 2)

    to_target = np.linalg.norm(candidates - np.asarray(lookat)[:2], axis=1)
    usable = (to_target >= near) & (to_target <= far)
    if not usable.any():                      # a room too small for the standoff band
        usable = to_target >= min(near, to_target.max())
    candidates, to_target = candidates[usable], to_target[usable]

    if len(fixtures):
        clearance = np.linalg.norm(candidates[:, None, :] - fixtures[None, :, :], axis=-1).min(axis=1)
    else:
        clearance = np.zeros(len(candidates))
    return candidates[int(np.argmax(clearance))]


def look_into_room(cam, entity, lookat, height: float = 2.1, margin: float = 0.5,
                   back: float = 2.2, azimuth: float | None = None):
    """Aim ``cam`` at ``lookat`` from a spot that is inside the room and above the walls.

    Parameters
    ----------
    height : float
        How high above the floor to sit. Kept below the wall tops so the view stays inside
        the room rather than looking down over it.
    margin : float
        How far inside the footprint to stay, so a wall never ends up between the camera
        and the subject.
    back : float, azimuth : float, optional
        Override the automatic placement with an explicit standoff and heading, for a shot
        that needs framing by hand.

    Returns
    -------
    The world position the camera was placed at.
    """
    lookat = np.asarray(lookat, dtype=np.float64)
    lo, hi = room_bounds(entity)

    if azimuth is None:
        pos_xy = open_floor_position(entity, lookat, lo, hi, margin=margin)
    else:
        direction = np.array([np.cos(azimuth), np.sin(azimuth)])
        pos_xy = np.clip(lookat[:2] + direction * back, lo[:2] + margin, hi[:2] - margin)

    pos = np.array([pos_xy[0], pos_xy[1], min(lo[2] + height, hi[2] - 0.2)])
    # Pin world up explicitly; leaving it to the default rolls the image when the camera
    # ends up close to level with its target.
    cam.set_pose(pos=tuple(pos), lookat=tuple(lookat), up=(0.0, 0.0, 1.0))
    return pos


def mount_camera(cam, link, cfg: dict, distance: float = 2.0):
    """Place ``cam`` at a RoboCasa ``cam_configs`` entry mounted on ``link``.

    RoboCasa stores each camera as a pose in its parent body's frame. MuJoCo cameras look
    down their own **-Z** with **+Y** up, so the local pose is resolved to a world
    position and forward/up direction rather than handed to Genesis as a raw transform —
    that keeps the convention explicit instead of relying on the two renderers agreeing.

    Call it again after stepping to follow a link that moves.
    """
    link_pos = np.asarray(link.get_pos().cpu()).reshape(3)
    link_quat = np.asarray(link.get_quat().cpu()).reshape(4)
    R_link = quat_to_R(link_quat)
    R_cam = R_link @ quat_to_R(cfg["quat"])
    pos = link_pos + R_link @ np.asarray(cfg["pos"], dtype=np.float64)
    forward = R_cam @ np.array([0.0, 0.0, -1.0])
    up = R_cam @ np.array([0.0, 1.0, 0.0])
    cam.set_pose(pos=tuple(pos), lookat=tuple(pos + forward * distance), up=tuple(up))
    return pos


__all__ = ["room_bounds", "look_into_room", "open_floor_position", "mount_camera",
           "room_vis_options", "quat_to_R"]
