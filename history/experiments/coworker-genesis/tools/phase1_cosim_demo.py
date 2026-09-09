"""Phase 1 deliverable: robot and human in one Genesis scene, neither aware of the other.

The plan's milestone for the end of Phase 1 is deliberately modest — a kitchen in which
the arm works while a person walks through, with *no* coupling between them. They will
interpenetrate, and that is the expected outcome: closing that loop is Phase 2's job.
What this script establishes is that both legs now run against the same physics scene and
the same clock, which is the thing that was not true before.

It also computes the first version of the safety signal Phase 4 will report: the minimum
distance between the human's body and the robot over the episode, and how often the two
actually touch.

Usage (Genesis venv):
  cd genesis && uv run python ../tools/phase1_cosim_demo.py --video ../exports/phase1_cosim.mp4
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "hpmm"))

import genesis as gs  # noqa: E402

from hpmm_sim.assets import load as assets  # noqa: E402
from hpmm_sim.vis.camera import look_into_room, room_vis_options  # noqa: E402

from hpmm_motion.io.schema import MotionSequence  # noqa: E402
from hpmm_sim.assets.convert import place_robot  # noqa: E402
from hpmm_sim.human.driver import HumanDriver  # noqa: E402
from hpmm_sim.robot.panda_omron import PandaOmron  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--motion", default="../exports/motion_kitchen_walk.npz")
    ap.add_argument("--mjcf", default="../exports/robocasa_layout1_style1.xml")
    ap.add_argument("--robot_pos", type=float, nargs=3, default=(2.0, -1.15, 0.0))
    ap.add_argument("--video", default="../exports/phase1_cosim.mp4")
    ap.add_argument("--dt", type=float, default=0.01, help="solver timestep")
    args = ap.parse_args()

    seq = MotionSequence.load(args.motion)
    placed = place_robot(args.mjcf, "../exports/robocasa_layout1_style1_placed.xml",
                         args.robot_pos, yaw=np.pi / 2)
    print(f"motion: {seq}\nscene:  {placed}  robot base at {tuple(args.robot_pos)}")

    gs.init(backend=gs.gpu)
    scene = gs.Scene(show_viewer=False, sim_options=gs.options.SimOptions(dt=args.dt),
                     vis_options=room_vis_options())
    kitchen = scene.add_entity(assets.mjcf(placed))
    human = HumanDriver(scene, mjcf_path="../exports/human_smplx.xml")
    cam = scene.add_camera(res=(1280, 720), fov=55, GUI=False)
    scene.build()

    human.bind()
    robot = PandaOmron(kitchen)
    print(f"robot: {robot}")
    q_target = robot.reset_to_safe_pose()
    robot.set_gains()
    print(f"robot base at {np.round(robot.base_pos(), 2)}, eef at {np.round(robot.eef_pos(), 2)}")

    # Frame both: look down the galley from the open side of the room.
    walk = seq.joints[0, :, 0]
    centre = 0.5 * (walk.mean(axis=0) + robot.base_pos())
    # Inside the room, above the wall tops. These kitchens are sealed boxes with no
    # ceiling, so a camera merely offset from the subject renders a brick wall exterior.
    cam_pos = look_into_room(cam, kitchen, (centre[0], centre[1], 1.0))
    print(f"camera at {np.round(cam_pos, 2)} looking at {np.round(centre[:2], 2)}")
    # Record at the solver rate: Genesis decimates renders to hit the requested fps, so
    # asking for 30 while stepping at 100 Hz would silently drop two of every three.
    cam.start_recording(save_to_filename=args.video, fps=round(1.0 / args.dt))

    # The human plays at its own fps; the robot sweeps once over the same wall clock.
    n_steps = int(seq.duration / args.dt)
    min_dists, n_contact, n_self = [], 0, 0
    for step in range(n_steps):
        t = step * args.dt
        # Robot: a slow reach sweep, unaware of anyone walking past.
        phase = t / seq.duration
        q = q_target.copy()
        q[robot.arm[1]] = q_target[robot.arm[1]] + 0.55 * np.sin(2 * np.pi * phase)
        q[robot.arm[3]] = q_target[robot.arm[3]] + 0.45 * np.sin(2 * np.pi * phase + 1.0)
        q[robot.arm[5]] = q_target[robot.arm[5]] + 0.60 * np.sin(2 * np.pi * phase + 2.0)
        robot.hold(q)

        human.set_time(seq, t)
        scene.step()
        cam.render()

        # Safety signal, first cut: closest approach of any human joint to the arm's
        # end-effector and to the base column.
        hp = human.joint_positions()[0]
        d = min(np.linalg.norm(hp - robot.eef_pos(), axis=-1).min(),
                np.linalg.norm(hp - robot.base_pos(), axis=-1).min())
        min_dists.append(d)
        contacts = human.entity.get_contacts(with_entity=kitchen)
        if len(np.asarray(contacts["force_a"].cpu()).reshape(-1, 3)):
            n_contact += 1
        self_contacts = human.entity.get_contacts(with_entity=human.entity)
        n_self += len(np.asarray(self_contacts["force_a"].cpu()).reshape(-1, 3))
    cam.stop_recording()

    min_dists = np.array(min_dists)
    print(f"\nepisode: {seq.duration:.1f} s, {n_steps} steps at dt={args.dt}")
    print(f"closest human-robot approach: {min_dists.min():.3f} m "
          f"(median over episode {np.median(min_dists):.3f} m)")
    print(f"steps with any human-scene contact: {n_contact}/{n_steps} "
          f"(the feet on the kitchen floor; the human is driven into it, not resting on it)")
    print(f"human self-contacts over the episode: {n_self} "
          f"(must be 0 — the capsules overlap by construction and are mask-filtered)")
    print(f"saved {os.path.abspath(args.video)}")


if __name__ == "__main__":
    main()
