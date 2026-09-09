"""Geometry and feedback for a single native NavigateKitchen crossing episode."""
import numpy as np
from isaac_human.motion import point_clearance


def yaw_xyzw(q):
    x, y, z, w = q
    return float(np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z)))


def wrap(angle):
    return (angle + np.pi) % (2*np.pi) - np.pi


def aisle_scenario(player, start, goal):
    """Fixed layout 4-2, seed 0: left aisle then rear aisle around the island."""
    start, goal = np.asarray(start, dtype=float), np.asarray(goal, dtype=float)
    if np.linalg.norm(start[:2]-[1.05, -2.975]) > .1 or np.linalg.norm(goal[:2]-[3.2, -.85]) > .1:
        raise ValueError('Scene placement changed; the calibrated aisle route must be reviewed')
    # Approach parallel to the counter before docking within the native 20 cm tolerance.
    waypoints = [np.array([1.15, -1.65, 0.]),
                 goal + np.array([0., -.45, 0.]), goal + np.array([0., -.16, 0.])]
    first = player.pose(0.)[0][0, 0]
    last = player.pose(player.end_time)[0][0, 0]
    travel = last[:2]-first[:2]
    if not 2.5 < np.linalg.norm(travel) < 3.5:
        raise ValueError('Motion travel does not match the calibrated rear aisle')
    yaw = -float(np.arctan2(travel[1], travel[0]))
    c, s = np.cos(yaw), np.sin(yaw)
    rotation = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
    translation = np.array([1.15, goal[1]-.60, 0.])-rotation @ np.r_[first[:2], 0.]
    return waypoints, yaw, translation


def clearance(player, time, base_xy, radius=.65, horizon=.8):
    """Conservative robot disk against projected full-body capsules, with lookahead."""
    values = []
    for t in np.linspace(time, time+horizon, 5):
        a, b, r = player.capsules(t)
        values.append(point_clearance(np.asarray(base_xy), a[0, :, :2], b[0, :, :2], r).min()-radius)
    return float(min(values))


def base_command(position, yaw, goal, goal_yaw, anchor_yaw, dt):
    delta = np.asarray(goal)[:2] - np.asarray(position)[:2]
    speed = delta * .8
    norm = np.linalg.norm(speed)
    if norm > .22:
        speed *= .22/norm
    c, s = np.cos(anchor_yaw), np.sin(anchor_yaw)
    local = np.array([[c, s], [-s, c]]) @ speed
    angular = np.clip(wrap(goal_yaw-yaw)*.8, -.35, .35)
    return {'mobilebase_forward': local[0]*dt/.01,
            'mobilebase_side': local[1]*dt/.01,
            'mobilebase_yaw': angular*dt/.02,
            'mobilebase_torso_height': 0.}
