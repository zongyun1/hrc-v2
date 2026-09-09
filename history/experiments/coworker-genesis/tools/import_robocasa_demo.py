"""Turn one recorded RoboCasa demonstration into a scene + reference trajectory.

Each episode in the LeRobot release ships an ``extras/`` folder that is far more useful
than the parquet: ``model.xml.gz`` is the exact MuJoCo model that episode ran against,
``states.npz`` is the full recorded state trajectory, and ``ep_meta.json`` names the
layout, style and instruction. Together they pin down an episode completely, without
re-deriving anything through robosuite.

Two things have to be fixed on the way in:

* **Asset paths point at the machine that recorded the demo** (``/root/robocasa/...``,
  ``/opt/conda/envs/robocasa/...``). They are re-rooted onto this checkout by splitting on
  the ``robocasa``/``robosuite`` asset markers, so the rewrite survives any origin layout.
* **``states`` is robosuite's flattened ``MjSimState``** — ``[time] + qpos + qvel`` — which
  has to be split using the compiled model's own ``nq``/``nv`` rather than a guess.

The end-effector reference is taken by running MuJoCo's own forward kinematics over the
recorded ``qpos``, which yields the grip site's **world** pose directly. That is
deliberately not the parquet's ``observation.state``: those EEF fields are expressed
relative to the mobile base, and a task-space reference that has to be un-rotated by a
moving base before it means anything is a reference waiting to be misused.

Usage (Genesis venv, which has mujoco):
  cd genesis && uv run python ../tools/import_robocasa_demo.py --episode 0
"""

import argparse
import gzip
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "hpmm"))
from hpmm_sim.assets.convert import sanitize_materials_text  # noqa: E402

ASSET_MARKERS = ("/robocasa/models/assets/", "/robosuite/models/assets/")
GRIP_SITE = "gripper0_right_grip_site"
JOINT_NQ = {0: 7, 1: 4, 2: 1, 3: 1}  # free, ball, slide, hinge


def reroot_assets(xml: str, repo_root: str) -> tuple[str, int]:
    """Point every asset reference at this checkout instead of the recording machine."""
    local = {
        "/robocasa/models/assets/": os.path.join(repo_root, "third_party/robocasa/robocasa/models/assets/"),
        "/robosuite/models/assets/": os.path.join(repo_root, "third_party/robosuite/robosuite/models/assets/"),
    }
    n = 0

    def sub(match):
        nonlocal n
        path = match.group(1)
        for marker, root in local.items():
            if marker in path:
                n += 1
                return 'file="' + root + path.split(marker, 1)[1] + '"'
        return match.group(0)

    return re.sub(r'file="([^"]+)"', sub, xml), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="../third_party/robocasa/datasets/v1.0/pretrain/atomic/"
                                         "PickPlaceCounterToCabinet/20250819/lerobot")
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--task", default="PickPlaceCounterToCabinet")
    ap.add_argument("--out_dir", default="../exports")
    args = ap.parse_args()

    import mujoco

    repo_root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    extras = os.path.join(args.dataset, "extras", f"episode_{args.episode:06d}")
    meta = json.load(open(os.path.join(extras, "ep_meta.json")))
    states = np.load(os.path.join(extras, "states.npz"))["states"]

    xml, n_rerooted = reroot_assets(gzip.open(os.path.join(extras, "model.xml.gz"), "rt").read(), repo_root)
    xml, n_materials = sanitize_materials_text(xml)
    stem = f"demo_{args.task}_ep{args.episode:03d}"
    os.makedirs(args.out_dir, exist_ok=True)
    xml_path = os.path.join(args.out_dir, stem + ".xml")
    with open(xml_path, "w") as f:
        f.write(xml)

    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    expected = 1 + model.nq + model.nv
    if states.shape[1] != expected:
        raise RuntimeError(
            f"state width {states.shape[1]} != 1 + nq({model.nq}) + nv({model.nv}) = {expected}; "
            "the flattened MjSimState layout is not what this assumes"
        )
    time = states[:, 0]
    qpos = states[:, 1:1 + model.nq]
    n_frames = len(states)

    # End-effector reference: MuJoCo's own FK on the recorded qpos, in world coordinates.
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, GRIP_SITE)
    if site < 0:
        raise RuntimeError(f"grip site {GRIP_SITE!r} not in this model")
    # The grip site is not a link, so record where it sits inside its parent body. The
    # Genesis side then drives that link and passes the offset as IK's `local_point`,
    # rather than needing MuJoCo to resolve the site itself.
    eef_body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.site_bodyid[site])
    eef_local_pos = [float(v) for v in model.site_pos[site]]
    eef_local_quat = [float(v) for v in model.site_quat[site]]

    eef_pos = np.empty((n_frames, 3))
    eef_mat = np.empty((n_frames, 9))
    for i in range(n_frames):
        data.qpos[:] = qpos[i]
        mujoco.mj_forward(model, data)
        eef_pos[i] = data.site_xpos[site]
        eef_mat[i] = data.site_xmat[site]

    # Gripper opening, as the binary-ish signal a controller actually needs.
    grip_dofs = [model.jnt_qposadr[j] for j in range(model.njnt)
                 if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or "").startswith("gripper0")]
    gripper = qpos[:, grip_dofs] if grip_dofs else np.zeros((n_frames, 0))

    # Initial configuration by joint name, in the same shape shim/sampler.TaskInstance reads.
    joints, free_bodies = {}, []
    for j in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        adr = model.jnt_qposadr[j]
        joints[name] = [float(v) for v in qpos[0, adr:adr + JOINT_NQ[int(model.jnt_type[j])]]]
        if int(model.jnt_type[j]) == 0:
            free_bodies.append(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.jnt_bodyid[j]))

    data.qpos[:] = qpos[0]
    mujoco.mj_forward(model, data)
    objects = {}
    for name in free_bodies:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        objects[name.replace("_main", "")] = {
            "root_body": name,
            "pos": [float(v) for v in data.xpos[bid]],
            "quat": [float(v) for v in data.xquat[bid]],
        }

    state_path = os.path.join(args.out_dir, stem + ".json")
    with open(state_path, "w") as f:
        json.dump({
            "task": args.task,
            "robot": "PandaOmron",
            "layout": meta["layout_id"],
            "style": meta["style_id"],
            "seed": args.episode,
            "mjcf": os.path.basename(xml_path),
            "language": meta["lang"],
            "joints": joints,
            "free_bodies": free_bodies,
            "objects": objects,
            "source": "robocasa_demo",
            "init_robot_base_pos": meta.get("init_robot_base_pos"),
            "init_robot_base_ori": meta.get("init_robot_base_ori"),
            "eef": {"site": GRIP_SITE, "body": eef_body,
                    "local_pos": eef_local_pos, "local_quat": eef_local_quat},
            # RoboCasa's own camera mounts. Worth carrying: they are attached to the robot
            # and therefore always framed on the workspace, whereas a camera positioned by
            # offsetting from the subject lands outside these sealed, ceiling-less rooms
            # and renders the outside of a wall.
            "cam_configs": meta.get("cam_configs"),
        }, f, indent=1)

    traj_path = os.path.join(args.out_dir, stem + "_traj.npz")
    np.savez_compressed(traj_path, time=time, qpos=qpos, eef_pos=eef_pos, eef_mat=eef_mat,
                        gripper=gripper, fps=1.0 / np.median(np.diff(time)))

    travel = np.linalg.norm(np.diff(eef_pos, axis=0), axis=1).sum()
    print(f"episode {args.episode}: layout={meta['layout_id']} style={meta['style_id']}")
    print(f"language: {meta['lang']!r}")
    print(f"model: {xml_path}  ({n_rerooted} asset paths re-rooted, nq={model.nq}, nv={model.nv})")
    if n_materials:
        print(f"materials clamped to legal ranges: {n_materials}")
    print(f"frames: {n_frames} over {time[-1] - time[0]:.2f} s "
          f"({1.0 / np.median(np.diff(time)):.1f} Hz)")
    print(f"objects: {list(objects)}")
    print(f"eef site {GRIP_SITE} on body {eef_body}, local pos {np.round(eef_local_pos, 4)} "
          f"quat {np.round(eef_local_quat, 4)}")
    print(f"EEF path: {travel:.3f} m travelled, "
          f"x[{eef_pos[:,0].min():.2f},{eef_pos[:,0].max():.2f}] "
          f"y[{eef_pos[:,1].min():.2f},{eef_pos[:,1].max():.2f}] "
          f"z[{eef_pos[:,2].min():.2f},{eef_pos[:,2].max():.2f}]")
    if gripper.size:
        print(f"gripper qpos range: [{gripper.min():.4f}, {gripper.max():.4f}]")
    print(f"state: {state_path}\ntraj:  {traj_path}")


if __name__ == "__main__":
    main()
