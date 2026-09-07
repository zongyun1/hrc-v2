"""Robot-only latest-Genesis smoke probe.

This intentionally avoids avatar and NYX. It verifies that core robot scene
construction, joint/link resolution, home-state qpos writes, PD control, and
optional rasterizer camera rendering still work under the active Genesis
installation.
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _preconfigure_render_platform() -> None:
    """Set OpenGL platform early enough for Genesis import when rendering."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--opengl-platform", choices=("egl", "osmesa"))
    parser.add_argument("--software-render", action="store_true")
    args, _ = parser.parse_known_args()

    if args.opengl_platform:
        os.environ.setdefault("GENESIS_OPENGL_PLATFORM", args.opengl_platform)
    elif args.software_render:
        os.environ.setdefault("GENESIS_SOFTWARE_RENDER", "1")
    elif args.render:
        os.environ.setdefault("GENESIS_OPENGL_PLATFORM", "egl")


_preconfigure_render_platform()

from envs.genesis_compat import configure_genesis_runtime, genesis_backend_from_env

configure_genesis_runtime()

import genesis as gs

from envs.robot import ARXX5Robot, FrankaRobot, PiperRobot, UR5WSGRobot, XArm7Robot


ROBOTS = {
    "piper": PiperRobot,
    "franka": FrankaRobot,
    "xarm7": XArm7Robot,
    "arx_x5": ARXX5Robot,
    "ur5_wsg": UR5WSGRobot,
}


def _init_genesis():
    try:
        gs.init(backend=genesis_backend_from_env(gs), logging_level="error")
    except Exception:
        pass


def probe_robot(name: str, render: bool = False) -> bool:
    cls = ROBOTS[name]
    print(f"{name}: create scene", flush=True)
    scene = gs.Scene(
        show_viewer=False,
        sim_options=gs.options.SimOptions(dt=0.002),
        renderer=gs.renderers.Rasterizer(),
    )
    scene.add_entity(gs.morphs.Plane())
    robot = cls()
    robot.add_to_scene(scene)
    cam = None
    if render:
        print(f"{name}: add camera", flush=True)
        cam = scene.add_camera(
            res=(160, 120),
            pos=(1.2, -1.2, 1.1),
            lookat=(0.0, -0.25, 0.8),
            fov=55,
        )
    print(f"{name}: build scene", flush=True)
    scene.build()
    print(f"{name}: init joints", flush=True)
    robot.init_joints(scene)
    robot.move_to_homestate()
    robot.open_gripper("right")
    for _ in range(5):
        scene.step()

    arm = robot.get_arm("right")
    ee = arm.get_ee_pose()
    qpos = arm.get_arm_qpos()
    msg = f"{name}: OK joints={len(arm.entity.joints)} arm_qpos_shape={tuple(qpos.shape)} ee_xyz={ee[:3]}"
    if render:
        print(f"{name}: render camera", flush=True)
        out = cam.render(rgb=True, depth=False)
        rgb = out[0] if isinstance(out, (list, tuple)) else out
        if hasattr(rgb, "cpu"):
            rgb = rgb.cpu().numpy()
        msg += f" render_shape={getattr(rgb, 'shape', None)}"
    print(msg, flush=True)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robots", nargs="+", default=list(ROBOTS), choices=sorted(ROBOTS))
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--opengl-platform", choices=("egl", "osmesa"))
    parser.add_argument("--software-render", action="store_true")
    args = parser.parse_args()

    _init_genesis()
    failures = []
    for name in args.robots:
        print(f"=== {name} ===", flush=True)
        try:
            probe_robot(name, render=args.render)
        except Exception as exc:
            failures.append(name)
            print(f"{name}: FAIL {type(exc).__name__}: {exc}", flush=True)
            traceback.print_exc()
    if failures:
        raise SystemExit(f"failed robots: {', '.join(failures)}")


if __name__ == "__main__":
    main()
