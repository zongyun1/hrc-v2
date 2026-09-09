"""Export one RoboCasa *task instance* — scene, objects and their sampled poses.

The Phase 0 exporter built the base ``Kitchen`` env, which is an empty kitchen shell: no
manipulable objects at all. A **task** env is what carries them — robosuite merges each
object model into the arena, so ``env.model.get_xml()`` then contains free-floating bodies
alongside the fixtures.

Where the objects *are*, though, is not in the XML. RoboCasa samples placements at
``env.reset()`` and writes them into ``sim.data.qpos``; the XML only holds each body's
authored default. So reproducing a specific task instance takes two artefacts:

* the MJCF (scene + robot + object models), and
* a JSON of the post-reset state, keyed by joint name so the Genesis side can apply it
  without depending on either engine's dof ordering.

Together they pin down one episode's starting condition exactly, which is what an IK
trajectory in Genesis has to be generated against.

Usage (robocasa venv, from repo root):
  third_party/robocasa-venv/bin/python tools/export_robocasa_task.py \
      --task PickPlaceCounterToCabinet --layout 1 --style 1 --seed 0
"""

import argparse
import json
import os

import numpy as np
import robocasa  # noqa: F401  registers the RoboCasa environments into robosuite
import robosuite
from robosuite.controllers import load_composite_controller_config

FREE_JOINT = 0  # mujoco.mjtJoint.mjJNT_FREE
JOINT_NQ = {0: 7, 1: 4, 2: 1, 3: 1}  # free, ball, slide, hinge


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="PickPlaceCounterToCabinet")
    ap.add_argument("--robot", default="PandaOmron")
    ap.add_argument("--layout", type=int, default=1)
    ap.add_argument("--style", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", default="exports")
    args = ap.parse_args()

    env = robosuite.make(
        env_name=args.task,
        robots=args.robot,
        controller_configs=load_composite_controller_config(robot=args.robot),
        layout_and_style_ids=[[args.layout, args.style]],
        translucent_robot=False,
        has_renderer=False,
        has_offscreen_renderer=False,
        render_camera=None,
        ignore_done=True,
        use_camera_obs=False,
        control_freq=20,
        seed=args.seed,
    )
    env.reset()
    model, data = env.sim.model, env.sim.data

    stem = f"task_{args.task}_L{args.layout}S{args.style}_seed{args.seed}"
    os.makedirs(args.out_dir, exist_ok=True)

    xml = env.model.get_xml()
    xml_path = os.path.join(args.out_dir, stem + ".xml")
    with open(xml_path, "w") as f:
        f.write(xml)

    # Every joint's post-reset configuration, by name. Free joints come out as
    # [x y z qw qx qy qz], matching both MuJoCo's and Genesis's layout.
    joints = {}
    free_bodies = []
    for i in range(model.njnt):
        name = model.joint_id2name(i)
        adr = model.jnt_qposadr[i]
        n = JOINT_NQ[int(model.jnt_type[i])]
        joints[name] = [float(v) for v in data.qpos[adr:adr + n]]
        if int(model.jnt_type[i]) == FREE_JOINT:
            free_bodies.append(model.body_id2name(model.jnt_bodyid[i]))

    objects = {}
    for key, obj in getattr(env, "objects", {}).items():
        body = obj.root_body
        bid = model.body_name2id(body)
        objects[key] = {
            "root_body": body,
            "joints": [model.joint_id2name(j) for j in range(model.njnt)
                       if model.jnt_bodyid[j] == bid],
            "pos": [float(v) for v in data.body_xpos[bid]],
            "quat": [float(v) for v in data.body_xquat[bid]],
        }

    eef_site = "gripper0_right_grip_site"
    state = {
        "task": args.task,
        "robot": args.robot,
        "layout": args.layout,
        "style": args.style,
        "seed": args.seed,
        "mjcf": os.path.basename(xml_path),
        "language": env.get_ep_meta().get("lang", ""),
        "n_qpos": int(model.nq),
        "joints": joints,
        "free_bodies": free_bodies,
        "objects": objects,
        "eef": {
            "site": eef_site,
            "pos": [float(v) for v in data.site_xpos[model.site_name2id(eef_site)]],
        } if eef_site in [model.site_id2name(i) for i in range(model.nsite)] else None,
        "note": ("Object poses come from the placement sampler at reset, not from the XML. "
                 "Apply `joints` by name after building in Genesis to reproduce this instance."),
    }
    json_path = os.path.join(args.out_dir, stem + ".json")
    with open(json_path, "w") as f:
        json.dump(state, f, indent=1)

    n_abs, n_mesh = xml.count('file="/'), xml.count("<mesh ")
    print(f"task: {args.task}  layout={args.layout} style={args.style} seed={args.seed}")
    print(f"lang: {state['language']!r}")
    print(f"mjcf: {xml_path} ({len(xml)} bytes, {n_mesh} meshes, {n_abs} absolute file refs)")
    print(f"nq={model.nq}, {len(joints)} joints, {len(free_bodies)} free bodies: {free_bodies}")
    for key, o in objects.items():
        print(f"  object {key:16s} body={o['root_body']:22s} pos={np.round(o['pos'], 3)}")
    print(f"state: {json_path}")


if __name__ == "__main__":
    main()
