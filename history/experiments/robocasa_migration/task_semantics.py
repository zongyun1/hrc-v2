"""Pure task predicates matching pinned RoboCasa task definitions."""
from math import acos, cos, degrees, dist, isfinite, sqrt


def upright_tilt_degrees(quat_xyzw):
    """Angle between the cup's local +Z axis and world +Z; invalid poses fail."""
    if len(quat_xyzw) != 4 or not all(map(isfinite, quat_xyzw)):
        return float('inf')
    norm = sqrt(sum(v*v for v in quat_xyzw))
    if norm < 1e-12:
        return float('inf')
    x, y, z, w = (v/norm for v in quat_xyzw)
    return degrees(acos(max(-1., min(1., 1-2*(x*x+y*y)))))


def navigation_success(base_position, base_yaw, target_position, target_yaw):
    return (all(map(isfinite, [*base_position[:2], base_yaw]))
            and dist(base_position[:2], target_position[:2]) <= 0.20
            and cos(target_yaw - base_yaw) >= 0.98)


def all_doors_open(positions, limits, threshold=0.90):
    if not positions or set(positions) != set(limits):
        return False
    for name, q in positions.items():
        lo, hi = limits[name]
        if hi <= lo or not isfinite(q):
            return False
        fraction = (q - lo) / (hi - lo)
        if lo < 0:
            fraction = 1 - fraction
        if fraction < threshold:
            return False
    return True


def point_inside_regions(point, regions):
    """Same center-only parallelepiped checks as OU.obj_inside_of(partial_check=True)."""
    if not all(map(isfinite, point)):
        return False
    def dot(a, b):
        return sum(x * y for x, y in zip(a, b))
    for p0, px, py, pz in regions.values():
        axes = [[b - a for a, b in zip(p0, p)] for p in (px, py, pz)]
        if all(dot(axis, axis) > 0 and dot(axis, p0) <= dot(axis, point) <= dot(axis, p)
               for axis, p in zip(axes, (px, py, pz))):
            return True
    return False


def counter_to_sink_success(object_position, gripper_position, regions):
    return point_inside_regions(object_position, regions) and dist(object_position, gripper_position) > 0.25


def serve_tea_success(teacup_position, saucer_position, gripper_position,
                      saucer_radius, teacup_saucer_contact, saucer_table_contact):
    """Original ServeTea: contact + strict XY radius + release + table contact."""
    return (all(map(isfinite, [*teacup_position, *saucer_position, *gripper_position, saucer_radius]))
            and saucer_radius > 0 and bool(teacup_saucer_contact) and bool(saucer_table_contact)
            and dist(teacup_position[:2], saucer_position[:2]) < 0.7 * saucer_radius
            and dist(teacup_position, gripper_position) > 0.25)


def native_serve_tea_contacts(env):
    return {"teacup_saucer": bool(env.check_contact(env.objects["teacup"], env.objects["saucer"])),
            "saucer_table": bool(env.check_contact(env.objects["saucer"], env.dining_table))}


def evaluate(manifest, body_positions, joint_positions, gripper_position=None, contacts=None):
    if manifest["success_spec"]["kind"] == "serve_tea":
        if gripper_position is None or contacts is None:
            raise ValueError("ServeTea requires actual gripper position and both contact observations")
        return serve_tea_success(
            body_positions[manifest["objects"]["teacup"]["body"]],
            body_positions[manifest["objects"]["saucer"]["body"]], gripper_position,
            manifest["objects"]["saucer"]["horizontal_radius"],
            contacts["teacup_saucer"], contacts["saucer_table"])

    target = manifest["fixtures"][manifest["target_fixture"]]
    if manifest["success_spec"]["kind"] == "all_doors_open":
        names = target["door_joints"]
        return all_doors_open({name: joint_positions[name] for name in names},
                              {name: manifest["joints"][name]["range"] for name in names})
    if gripper_position is None:
        raise ValueError("An actual gripper position is required for the pick-and-place task")
    return counter_to_sink_success(body_positions[manifest["objects"]["obj"]["body"]],
                                   gripper_position, target["internal_regions"])
