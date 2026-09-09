"""Reproduce a recorded RoboCasa demonstration in Genesis, through IK.

The plan's standing decision is not to replay MuJoCo demonstrations. That is about
replaying *actions*: an action sequence is meaningless across engines, because it is
interpreted by a specific controller with specific gains against a specific contact model.
A **task-space** reference is a different object entirely — the end-effector path is a
property of the task, not of the controller — so tracking it with Genesis's own IK is a
genuine re-generation, and it is what lets the same episode be re-run later with objects
moved or a person walking through it.

What is replayed directly, and why:

* **arm** — IK against the recorded grip-site pose, restricted to the 7 arm dofs so the
  solver cannot "solve" the problem by swinging a cabinet door.
* **mobile base and torso column** — driven from the recorded joint values. These are the
  robot's placement in the room rather than a manipulation choice, and re-deriving them
  would change what the arm's reference even means.
* **gripper** — the recorded open/close *timing*, but not the recorded finger positions.
  Those positions are where the fingers came to rest **against the object**: the demo's
  "closed" separation is 38 mm because that is how wide the cereal box is. Commanding that
  exact separation makes a position controller reach it with zero error and therefore
  squeeze with zero force, and the object is never actually held. So a closed frame
  commands the fingers shut past the object, under a force limit, and lets contact stop
  them where the object is.

Usage (Genesis venv):
  cd genesis && uv run python ../tools/replay_demo_ik.py \
      --state ../exports/demo_PickPlaceCounterToCabinet_ep000.json --video ../exports/demo_ik.mp4
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
from hpmm_sim.vis.camera import look_into_room, mount_camera  # noqa: E402


def mat_to_quat(mat: np.ndarray) -> np.ndarray:
    """Row-major 3x3 rotation matrices ``(N, 9)`` -> w-x-y-z quaternions ``(N, 4)``."""
    m = mat.reshape(-1, 3, 3)
    trace = m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2]
    q = np.empty((len(m), 4))
    # Branch on the largest diagonal term so the divisor is never near zero.
    big = trace > 0
    s = np.sqrt(np.maximum(trace[big] + 1.0, 1e-12)) * 2
    q[big, 0] = 0.25 * s
    q[big, 1] = (m[big, 2, 1] - m[big, 1, 2]) / s
    q[big, 2] = (m[big, 0, 2] - m[big, 2, 0]) / s
    q[big, 3] = (m[big, 1, 0] - m[big, 0, 1]) / s
    for i in np.flatnonzero(~big):
        d = np.array([m[i, 0, 0], m[i, 1, 1], m[i, 2, 2]])
        k = int(np.argmax(d))
        a, b = (k + 1) % 3, (k + 2) % 3
        s = np.sqrt(max(m[i, k, k] - m[i, a, a] - m[i, b, b] + 1.0, 1e-12)) * 2
        q[i, 0] = (m[i, b, a] - m[i, a, b]) / s
        q[i, 1 + k] = 0.25 * s
        q[i, 1 + a] = (m[i, a, k] + m[i, k, a]) / s
        q[i, 1 + b] = (m[i, b, k] + m[i, k, b]) / s
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default="../exports/demo_PickPlaceCounterToCabinet_ep000.json")
    ap.add_argument("--traj", default=None, help="defaults to <state stem>_traj.npz")
    ap.add_argument("--dt", type=float, default=0.01)
    ap.add_argument("--video", default=None)
    ap.add_argument("--camera", default="robot0_agentview_center",
                    help="which of RoboCasa's own cam_configs to render from")
    ap.add_argument("--kp", type=float, default=3000.0)
    ap.add_argument("--kv", type=float, default=200.0)
    # robosuite's own finger actuators are limited to +-20 N; anything more would be a
    # gripper this robot does not have, and a reproduction it could not actually perform.
    ap.add_argument("--grip_force", type=float, default=20.0, help="N per finger")
    args = ap.parse_args()

    instance = TaskInstance.load(args.state)
    traj_path = args.traj or str(instance.mjcf_path).replace(".xml", "_traj.npz")
    traj = np.load(traj_path)
    eef_pos, eef_quat = traj["eef_pos"], mat_to_quat(traj["eef_mat"])
    qpos_ref, gripper_ref = traj["qpos"], traj["gripper"]
    # The demo's finger separation bottoms out at the object's width, not at zero, so a
    # midpoint threshold recovers the open/close *intent* rather than the contact geometry.
    separation = gripper_ref[:, 0] - gripper_ref[:, 1]
    grip_closed = separation < 0.5 * (separation.max() + separation.min())
    demo_fps = float(traj["fps"])
    n_demo = len(eef_pos)
    print(f"{instance}\nlanguage: {instance.language!r}")
    print(f"reference: {n_demo} frames @ {demo_fps:g} Hz, EEF travels "
          f"{np.linalg.norm(np.diff(eef_pos, axis=0), axis=1).sum():.3f} m")
    flips = np.flatnonzero(np.diff(grip_closed.astype(int)) != 0) + 1
    print(f"gripper: separation {separation.max()*1000:.1f} mm open -> "
          f"{separation.min()*1000:.1f} mm closed (the object's width); "
          f"closes/opens at frames {flips.tolist()}")

    gs.init(backend=gs.gpu)
    scene = gs.Scene(show_viewer=False, sim_options=gs.options.SimOptions(dt=args.dt))
    entity = scene.add_entity(assets.mjcf(instance.mjcf_path))
    cam = scene.add_camera(res=(1280, 720), fov=45, GUI=False) if args.video else None
    scene.build()

    instance.apply(entity, strict=False)
    robot = PandaOmron(entity)
    robot.set_gains(kp=args.kp, kv=args.kv)
    if len(robot.gripper):
        entity.set_dofs_force_range(np.full(len(robot.gripper), -args.grip_force),
                                    np.full(len(robot.gripper), args.grip_force), robot.gripper)
    # Frame 0's recorded finger positions are the open pose for this gripper.
    grip_open = gripper_ref[0] if gripper_ref.shape[1] == len(robot.gripper) else None
    eef_link = entity.get_link(instance.__dict__.get("eef", {}).get("body", "gripper0_right_eef")
                               if isinstance(getattr(instance, "eef", None), dict) else "gripper0_right_eef")
    print(f"{robot}; tracking link {eef_link.name}")

    # Which recorded dofs feed which channel. Base and torso are replayed, the arm is solved.
    name_to_qadr = {}
    offset = 0
    for name, value in instance.joints.items():
        name_to_qadr[name] = (offset, len(value))
        offset += len(value)
    replay_dofs, replay_qcols = [], []
    for joint in entity.joints:
        if joint.name.startswith("mobilebase0_joint") and joint.name in name_to_qadr:
            start, width = name_to_qadr[joint.name]
            replay_dofs.extend(joint.dofs_idx_local)
            replay_qcols.extend(range(start, start + width))
    replay_dofs = np.asarray(replay_dofs, dtype=int)
    replay_qcols = np.asarray(replay_qcols, dtype=int)
    base_motion = np.abs(qpos_ref[:, replay_qcols] - qpos_ref[0, replay_qcols]).max()
    print(f"base/torso dofs replayed: {len(replay_dofs)}; they move {base_motion*1000:.1f} mm/rad over the episode")

    steps_per_frame = max(1, int(round(1.0 / (demo_fps * args.dt))))
    # Camera: prefer RoboCasa's own mount. It rides on the robot, so it is always framed
    # on the workspace — unlike an offset from the subject, which in these sealed,
    # ceiling-less rooms lands outside and renders the far side of a brick wall.
    cam_cfg = (instance.cam_configs or {}).get(args.camera) if hasattr(instance, "cam_configs") else None
    cam_link = entity.get_link(cam_cfg["parent_body"]) if cam_cfg else None
    if cam is not None:
        if cam_cfg is not None:
            mount_camera(cam, cam_link, cam_cfg)
            print(f"camera: RoboCasa mount {args.camera!r} on {cam_cfg['parent_body']}")
        else:
            look_into_room(cam, entity, eef_pos.mean(axis=0))
            print(f"camera: no cam_configs for {args.camera!r}; framed from inside the room")
        cam.start_recording(save_to_filename=args.video, fps=round(1.0 / args.dt))

    q_cmd = np.asarray(entity.get_dofs_position().cpu()).copy()
    errors, joint_lags, obj_z = [], [], []
    obj_body = instance.objects["obj"]["root_body"] if "obj" in instance.objects else None
    for i in range(n_demo):
        q_ik = entity.inverse_kinematics(
            link=eef_link, pos=eef_pos[i], quat=eef_quat[i],
            dofs_idx_local=robot.arm, init_qpos=None,
        )
        q_ik = np.asarray(q_ik.cpu()).reshape(-1)
        q_cmd[robot.arm] = q_ik[robot.arm]
        q_cmd[replay_dofs] = qpos_ref[i, replay_qcols]
        if len(robot.gripper):
            # Closed means "squeeze past the object", not "return to the recorded width" —
            # the latter is where contact stopped the fingers, and commanding it produces
            # no grip force at all. The force range set above bounds the squeeze.
            q_cmd[robot.gripper] = 0.0 if grip_closed[i] else grip_open
        robot.hold(q_cmd)
        for _ in range(steps_per_frame):
            scene.step()
            if cam is not None:
                if cam_cfg is not None:
                    mount_camera(cam, cam_link, cam_cfg)   # the mount rides the base
                cam.render()
        got = np.asarray(eef_link.get_pos().cpu()).reshape(3)
        errors.append(np.linalg.norm(got - eef_pos[i]))
        # Split the blame: how far the IK solution itself lands from the target, versus
        # how far the arm lags the joint angles it was told to reach.
        q_now = np.asarray(entity.get_dofs_position().cpu()).reshape(-1)
        joint_lags.append(np.abs(q_now[robot.arm] - q_cmd[robot.arm]).max())
        if obj_body:
            obj_z.append(float(np.asarray(entity.get_link(obj_body).get_pos().cpu()).reshape(3)[2]))
    if cam is not None:
        cam.stop_recording()
        print(f"saved {os.path.abspath(args.video)}")

    errors, joint_lags = np.array(errors), np.array(joint_lags)
    print("\nEEF tracking error vs the recorded reference:")
    print(f"  mean {errors.mean()*1000:7.1f} mm   median {np.median(errors)*1000:7.1f} mm   "
          f"p95 {np.percentile(errors,95)*1000:7.1f} mm   max {errors.max()*1000:7.1f} mm")
    print(f"  within 20 mm on {(errors < 0.02).sum()}/{n_demo} frames, "
          f"within 5 mm on {(errors < 0.005).sum()}/{n_demo}")
    print(f"  arm joint lag |achieved - commanded|: mean {joint_lags.mean():.4f} rad  "
          f"max {joint_lags.max():.4f} rad")
    step = np.linalg.norm(np.diff(eef_pos, axis=0), axis=1)
    print(f"  (for scale, the reference moves {step.mean()*1000:.1f} mm per frame)")
    if obj_z:
        obj_z = np.array(obj_z)
        lift = obj_z.max() - obj_z[0]
        print(f"\ngrasp: object z {obj_z[0]:.3f} -> peak {obj_z.max():.3f} -> end {obj_z[-1]:.3f} "
              f"(lifted {lift*1000:.1f} mm) -> {'PICKED UP' if lift > 0.03 else 'NOT PICKED UP'}")

    for key, obj in instance.objects.items():
        pos = np.asarray(entity.get_link(obj["root_body"]).get_pos().cpu()).reshape(3)
        start = np.asarray(obj["pos"])
        print(f"  object {key:14s} start z={start[2]:.3f} -> end z={pos[2]:.3f} "
              f"(moved {np.linalg.norm(pos-start)*1000:6.1f} mm)")


if __name__ == "__main__":
    main()
