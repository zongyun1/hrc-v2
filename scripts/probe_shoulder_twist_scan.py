"""Measure the upper-arm (Arm bone) TWIST range used by authored clips.

Twist definition (swing-twist): for each frame, build the zero-twist frame
R0 = swing(rest_upper_dir -> cur_upper_dir) @ R_arm_rest, then decompose
R_rel = R_arm_cur @ R0^T about the current upper-arm axis.  The clips'
percentile range of this angle is the "natural shoulder roll envelope";
outside it the shoulder looks wrong.

Appends per-side {R_arm_rest_world, rest_upper_dir_world, twist_lo/hi_deg
(2.5/97.5 pct)} to envs/avatar/motions/palm_ref_pose.json.

Run (compute node, MAWM_latest_genesis python, repo root):
    python scripts/probe_shoulder_twist_scan.py
"""
import json
import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation as Rot

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

OUT = os.path.join(ROOT, "envs", "avatar", "motions", "palm_ref_pose.json")

M_W_I = np.array([[0.0, -1.0, 0.0],
                  [1.0, 0.0, 0.0],
                  [0.0, 0.0, 1.0]], dtype=np.float64)


def block_world(global_t, idx):
    block = np.asarray(global_t[idx][:3, :3], dtype=np.float64)
    norms = np.linalg.norm(block, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return M_W_I @ (block / norms).T


def swing(u, v):
    u = np.asarray(u, dtype=np.float64); u = u / (np.linalg.norm(u) + 1e-12)
    v = np.asarray(v, dtype=np.float64); v = v / (np.linalg.norm(v) + 1e-12)
    d = float(np.clip(np.dot(u, v), -1.0, 1.0))
    if d > 1.0 - 1e-9:
        return np.eye(3)
    axis = np.cross(u, v)
    n = np.linalg.norm(axis)
    if n < 1e-9:
        axis = np.cross(u, [1.0, 0.0, 0.0])
        n = np.linalg.norm(axis) + 1e-12
    return Rot.from_rotvec(axis / n * np.arccos(d)).as_matrix()


def twist_about(R_rel, axis):
    q = Rot.from_matrix(R_rel).as_quat()  # xyzw
    t = 2.0 * np.arctan2(float(np.dot(q[:3], axis)), float(q[3]))
    return float((t + np.pi) % (2.0 * np.pi) - np.pi)


def main():
    import yaml
    from envs.tasks import TASK_MAP
    from envs.avatar.utils import (
        Mixamo_data_to_controller_pose,
        Mixamo_node_processing,
    )

    with open("config/default.yml") as f:
        config = yaml.safe_load(f)
    config["use_viewer"] = False
    task = TASK_MAP["blocks_ranking_size_assist"](config)
    task.reset(0)
    av = task.avatar
    robot = av.robot
    vgeom = robot.skin.links[0]._vgeoms[0]

    def arm_state(bone_side):
        """(R_arm_world, upper_dir_world) for the current applied pose."""
        global_t = np.asarray(robot.skin.global_transforms, dtype=np.float64)
        i_arm = int(robot.skin.node_findup[f"{bone_side}Arm"])
        i_fore = int(robot.skin.node_findup[f"{bone_side}ForeArm"])
        R_arm = block_world(global_t, i_arm)
        up = M_W_I @ (global_t[i_fore][3, :3] - global_t[i_arm][3, :3])
        n = np.linalg.norm(up)
        return R_arm, (up / n if n > 1e-9 else None)

    def apply_frame(md, i, mat, mat_inv):
        pose = Mixamo_data_to_controller_pose(
            md["trans"][i], md["rot"][i], md["joint"][i]
        )
        robot.pose = pose
        robot.node_trans = Mixamo_node_processing(vgeom, pose, mat, mat_inv)
        robot.global_mat = mat
        robot.global_mat_inv = mat_inv
        robot.update()

    # Rest = idle frame 0.
    idle = av.motion_data["idle"]
    idle_mat = idle["mat"][0]
    idle_mat_inv = np.array([np.linalg.inv(m) for m in idle_mat])
    apply_frame(idle, 0, idle_mat, idle_mat_inv)
    rest = {}
    for side, bone_side in (("left", "Left"), ("right", "Right")):
        R_rest, dir_rest = arm_state(bone_side)
        rest[side] = (R_rest, dir_rest)

    twists = {"left": [], "right": []}
    for clip_name, md in sorted(av.motion_data.items()):
        try:
            n = int(md["trans"].shape[0])
            mat = md["mat"][0]
            mat_inv = np.array([np.linalg.inv(m) for m in mat])
        except Exception:
            continue
        for i in range(0, n, 2):
            try:
                apply_frame(md, i, mat, mat_inv)
            except Exception:
                break
            for side, bone_side in (("left", "Left"), ("right", "Right")):
                R_cur, dir_cur = arm_state(bone_side)
                R_rest, dir_rest = rest[side]
                if dir_cur is None or dir_rest is None:
                    continue
                R0 = swing(dir_rest, dir_cur) @ R_rest
                t = twist_about(R_cur @ R0.T, dir_cur)
                twists[side].append(np.degrees(t))

    with open(OUT) as f:
        report = json.load(f)
    for side in ("left", "right"):
        arr = np.asarray(twists[side], dtype=np.float64)
        lo, hi = np.percentile(arr, [2.5, 97.5])
        print(f"{side}: n={arr.size} mean={arr.mean():.1f} "
              f"p2.5={lo:.1f} p50={np.percentile(arr, 50):.1f} "
              f"p97.5={hi:.1f} min={arr.min():.1f} max={arr.max():.1f}",
              flush=True)
        R_rest, dir_rest = rest[side]
        report[side]["R_arm_rest_world"] = np.round(R_rest, 6).tolist()
        report[side]["rest_upper_dir_world"] = np.round(dir_rest, 6).tolist()
        report[side]["shoulder_twist_lo_deg"] = round(float(lo), 1)
        report[side]["shoulder_twist_hi_deg"] = round(float(hi), 1)
    with open(OUT, "w") as f:
        json.dump(report, f, indent=1)
    print(f"saved {OUT}", flush=True)


if __name__ == "__main__":
    main()
