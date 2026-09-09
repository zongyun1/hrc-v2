"""Can Genesis reproduce one specific RoboCasa episode, objects and all?

Loading the task MJCF alone is not enough: RoboCasa samples object placements at
``env.reset()`` and stores them in ``qpos``, while the XML keeps only each body's authored
default. So this checks the full round trip —

    robosuite task env  --(export_robocasa_task.py)-->  MJCF + state JSON
                        --(shim/sampler.TaskInstance)-->  the same scene in Genesis

— by comparing where each object *actually ends up* in Genesis against where robosuite put
it. That equality is the precondition for generating a robot trajectory against this
episode: an IK plan aimed at a sugar cube that is not where it was sampled is a plan for a
different episode.

It also settles the second half of the question: whether the objects, once placed, are
dynamically simulated (they should fall/settle under gravity, not hover).

Usage (Genesis venv):
  cd genesis && uv run python ../tools/verify_task_instance.py \
      --state ../exports/task_PickPlaceCounterToCabinet_L1S1_seed0.json
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "hpmm"))

import genesis as gs  # noqa: E402

from hpmm_sim.assets import load as assets  # noqa: E402

from hpmm_sim.robot.panda_omron import PandaOmron  # noqa: E402
from hpmm_sim.shim.sampler import TaskInstance  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default="../exports/task_PickPlaceCounterToCabinet_L1S1_seed0.json")
    ap.add_argument("--settle", type=int, default=100, help="steps to let objects settle")
    ap.add_argument("--video", default=None)
    args = ap.parse_args()

    instance = TaskInstance.load(args.state)
    print(f"{instance}\nlanguage: {instance.language!r}\nmjcf: {instance.mjcf_path}")

    gs.init(backend=gs.gpu)
    scene = gs.Scene(show_viewer=False, sim_options=gs.options.SimOptions(dt=0.01))
    entity = scene.add_entity(assets.mjcf(instance.mjcf_path))
    cam = scene.add_camera(res=(1280, 720), fov=45, GUI=False) if args.video else None
    scene.build()

    print(f"entity: {len(entity.links)} links, {entity.n_qs} qpos, {entity.n_dofs} dofs")

    # Before applying the sampled state, objects sit at their authored XML defaults.
    def object_positions():
        return {k: np.asarray(entity.get_link(o["root_body"]).get_pos().cpu()).reshape(3)
                for k, o in instance.objects.items()}

    before = object_positions()
    report = instance.apply(entity, strict=False)
    print(f"applied placement: {report}")
    after = object_positions()

    print("\nobject placement fidelity (Genesis vs robosuite's sampled pose):")
    worst = 0.0
    for key in instance.objects:
        want, _ = instance.object_pose(key)
        moved = np.linalg.norm(after[key] - before[key])
        err = np.linalg.norm(after[key] - want)
        worst = max(worst, err)
        print(f"  {key:16s} want={np.round(want, 3)} got={np.round(after[key], 3)}  "
              f"err={err*1000:7.2f} mm   (moved {moved*1000:.1f} mm from the XML default)")
    print(f"worst placement error: {worst*1000:.2f} mm")

    # Are the objects actually dynamic? Settle under gravity and see if they stay put.
    robot = PandaOmron(entity)
    robot.set_gains()
    q_hold = np.asarray(entity.get_dofs_position().cpu()).copy()
    if cam is not None:
        centre = np.mean([after[k] for k in instance.objects], axis=0)
        cam.set_pose(pos=(centre[0] - 1.2, centre[1] - 1.6, centre[2] + 0.9),
                     lookat=(centre[0], centre[1], centre[2]))
        cam.start_recording(save_to_filename=args.video, fps=100)
    for _ in range(args.settle):
        robot.hold(q_hold)
        scene.step()
        if cam is not None:
            cam.render()
    if cam is not None:
        cam.stop_recording()
        print(f"saved {os.path.abspath(args.video)}")

    settled = object_positions()
    print(f"\nafter {args.settle} steps of gravity:")
    for key in instance.objects:
        drop = after[key] - settled[key]
        print(f"  {key:16s} moved {np.linalg.norm(drop)*1000:7.2f} mm "
              f"(dz={-drop[2]*1000:+.2f} mm) -> {'settled' if np.linalg.norm(drop) < 0.02 else 'FELL/SLID'}")


if __name__ == "__main__":
    main()
