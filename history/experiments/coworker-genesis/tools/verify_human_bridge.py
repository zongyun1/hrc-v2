"""Phase 1 bridge check: does a TRUMANS motion transfer intact onto a Genesis body?

This is the step that turns "human" from a rendered mesh into an actual physical entity.
It builds the rigid SMPL-X skeleton in a Genesis scene (optionally inside the RoboCasa
kitchen), drives it frame by frame from a MotionSequence, and measures three things:

1. **Pose fidelity** — each link origin is an SMPL-X body joint, so the world positions
   Genesis reports must equal the ``joints`` cache in the sequence. Any mismatch means
   the qpos layout, the ball-joint convention, or the rest offsets are wrong.
2. **Kinematic stability** — the per-frame override must hold the body exactly where it
   was put, with no drift between overrides (the Phase 0.6 property, now on 22 bodies).
3. **Mass** — a sanity check that the capsule geometry gives a human-sized body, since
   contact force is a Phase 3 safety signal.

Usage (Genesis venv):
  cd genesis && uv run python ../tools/verify_human_bridge.py [--kitchen] [--video]
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
from hpmm_sim.human.driver import HumanDriver  # noqa: E402
from hpmm_sim.human.skeleton import HumanSkeleton  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--motion", default="../exports/motion_kitchen_walk.npz")
    ap.add_argument("--kitchen", action="store_true", help="also load the RoboCasa scene")
    ap.add_argument("--mjcf", default="../exports/robocasa_layout1_style1.xml")
    ap.add_argument("--video", default=None, help="path to write an mp4 of the playback")
    ap.add_argument("--dt", type=float, default=0.01, help="solver timestep")
    args = ap.parse_args()

    seq = MotionSequence.load(args.motion)
    print(f"motion: {seq}  meta.backend={seq.meta.get('backend')} frame={seq.meta.get('frame')}")

    gs.init(backend=gs.gpu)
    scene = gs.Scene(show_viewer=False, sim_options=gs.options.SimOptions(dt=args.dt),
                     vis_options=room_vis_options())
    scene.add_entity(gs.morphs.Plane())
    kitchen = scene.add_entity(assets.mjcf(args.mjcf)) if args.kitchen else None

    skeleton = HumanSkeleton.from_template()
    human = HumanDriver(scene, skeleton, mjcf_path="../exports/human_smplx.xml")

    cam = None
    if args.video:
        cam = scene.add_camera(res=(1280, 720), fov=50, GUI=False)
    scene.build()
    human.bind()

    print(f"human entity: {len(human.link_names)} links, n_qs={human.entity.n_qs}, "
          f"n_dofs={human.entity.n_dofs}, mass={human.total_mass():.1f} kg")

    # Link order follows the MJCF's depth-first body order; map it back to SMPL-X indices.
    dfs = skeleton.dfs_order()
    link_to_joint = {skeleton.body_name(j): j for j in range(skeleton.n_joints)}
    order = np.array([link_to_joint[n] for n in human.link_names])
    assert list(order) == dfs, f"link order {list(order)} != dfs {dfs}"

    # 1. Pose fidelity, sampled across the sequence.
    errs = []
    for f in range(0, seq.n_frames, max(1, seq.n_frames // 20)):
        human.set_frame(seq, f)
        # Read before stepping: set_qpos already resolves forward kinematics, while a
        # step would integrate gravity and whatever contact the pose happens to be in.
        got = human.joint_positions()[0]                 # (22, 3) in link order
        want = seq.joints[0, f, order]                   # SMPL-X joints, same order
        errs.append(np.linalg.norm(got - want, axis=-1))
    errs = np.array(errs)
    print(f"pose transfer error: mean {errs.mean()*1000:.3f} mm  max {errs.max()*1000:.3f} mm")

    # 2. Kinematic stability. Two different quantities, easily confused:
    #    - excursion: how far the body moves *within* one step, before the next override
    #      pulls it back. Bounded, does not accumulate.
    #    - accumulation: whether repeated override cycles leave a residue. Must be zero,
    #      otherwise the human would slowly walk away from its own motion.
    human.set_frame(seq, 0)
    held = human.joint_positions()[0].copy()
    for _ in range(50):
        human.set_frame(seq, 0)
        scene.step()
    excursion = np.linalg.norm(human.joint_positions()[0] - held, axis=-1).max()
    human.set_frame(seq, 0)
    accumulated = np.linalg.norm(human.joint_positions()[0] - held, axis=-1).max()
    for _ in range(50):
        scene.step()                                      # no override on purpose
    unheld = np.linalg.norm(human.joint_positions()[0] - held, axis=-1).max()
    print(f"within-step excursion:  {excursion*1000:.3f} mm (bounded, reset every step)")
    print(f"accumulated drift:      {accumulated*1000:.3f} mm over 50 override cycles")
    print(f"without any override:   {unheld*1000:.1f} mm over 50 steps "
          f"(expected large — this is exactly what the override suppresses)")

    # 3. Playback (+ video).
    n_steps = int(seq.duration / args.dt)
    if cam is not None:
        pelvis = seq.joints[0, :, 0]
        centre = pelvis.mean(axis=0)
        # See the note in hpmm_sim/vis/camera: these rooms are sealed and ceiling-less, so
        # the camera has to sit inside the footprint and above the walls.
        target = kitchen if kitchen is not None else human.entity
        print(f"camera at {np.round(look_into_room(cam, target, (centre[0], centre[1], 1.0)), 2)}")
        # Record at the solver rate: Genesis decimates renders to hit the requested fps,
        # so asking for 30 while stepping at 100 Hz would silently drop two of every three.
        cam.start_recording(save_to_filename=args.video, fps=round(1.0 / args.dt))
    for step in range(n_steps):
        human.set_time(seq, step * args.dt)
        scene.step()
        if cam is not None:
            cam.render()
    if cam is not None:
        cam.stop_recording()
        print(f"saved {os.path.abspath(args.video)} ({n_steps} frames, {seq.duration:.1f} s)")

    if kitchen is not None:
        # Does the physical human actually touch the physical kitchen? A walk down open
        # floor should mostly not, so this is a low number, not zero, and mainly the feet.
        n_contact_steps, forces = 0, []
        for step in range(n_steps):
            human.set_time(seq, step * args.dt)
            scene.step()
            contacts = human.entity.get_contacts(with_entity=kitchen)
            force = np.asarray(contacts["force_a"].cpu()).reshape(-1, 3)
            if len(force):
                n_contact_steps += 1
                forces.append(np.linalg.norm(force, axis=-1).max())
        print(f"human-kitchen contact: {n_contact_steps}/{n_steps} steps"
              + (f", peak force {max(forces):.1f} N" if forces else ""))


if __name__ == "__main__":
    main()
