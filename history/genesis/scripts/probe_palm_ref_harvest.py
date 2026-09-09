"""Harvest the chosen palm-down reference frames with full context
(hand world orientation + forearm direction) into palm_ref_pose.json.

The forearm direction lets the runtime preserve the AUTHORED WRIST-LOCAL
rotation: it aligns the whole reference (forearm+hand) onto the current
IK forearm before extracting the hand orientation, so the wrist bend
stays exactly as animated regardless of how steep the IK reach is.

Run (compute node, MAWM_latest_genesis python, repo root):
    python scripts/probe_palm_ref_harvest.py
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

# side -> (clip, frame): winners of scripts/probe_palm_ref_scan.py.
CHOSEN = {
    "right": ("Inspect", 22),
    "left": ("take_from_human_intention", 202),
}
OUT = os.path.join(ROOT, "envs", "avatar", "motions", "palm_ref_pose.json")


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

    report = {}
    for side, (clip, frame) in CHOSEN.items():
        hand = 0 if side == "left" else 1
        bone_side = "Left" if hand == 0 else "Right"
        md = av.motion_data[clip]
        clip_mat = md["mat"][0]
        clip_mat_inv = np.array([np.linalg.inv(m) for m in clip_mat])
        pose = Mixamo_data_to_controller_pose(
            md["trans"][frame], md["rot"][frame], md["joint"][frame]
        )
        node = Mixamo_node_processing(vgeom, pose, clip_mat, clip_mat_inv)
        robot.pose = pose
        robot.node_trans = node
        robot.global_mat = clip_mat
        robot.global_mat_inv = clip_mat_inv
        robot.update()

        wrist_w, R_hand = robot._get_hand_frame(hand)
        wrist_w = np.asarray(wrist_w, dtype=np.float64).ravel()[:3]
        elbow_w = np.asarray(
            robot.skin.get_global_translation(f"{bone_side}ForeArm")[0],
            dtype=np.float64,
        ).ravel()[:3]
        shoulder_w = np.asarray(
            robot.skin.get_global_translation(f"{bone_side}Arm")[0],
            dtype=np.float64,
        ).ravel()[:3]
        fore = wrist_w - elbow_w
        fore = fore / (np.linalg.norm(fore) + 1e-12)
        upper = elbow_w - shoulder_w
        upper = upper / (np.linalg.norm(upper) + 1e-12)
        az = float(np.arctan2(fore[1], fore[0]))
        e3 = R_hand[:, 2]
        down = float(e3[2]) if hand == 0 else float(-e3[2])

        # Full-arm bone rotation frames in WORLD (column convention).
        # Rows of skin.global_transforms store the column matrix transposed
        # in the internal (axis-swapped) frame: R_world = M_W_I @ block.T.
        M_W_I = np.array([[0.0, -1.0, 0.0],
                          [1.0, 0.0, 0.0],
                          [0.0, 0.0, 1.0]], dtype=np.float64)
        global_t = np.asarray(robot.skin.global_transforms, dtype=np.float64)
        bones = {}
        for key, bone in (("arm", "Arm"), ("forearm", "ForeArm"),
                          ("forearm1", "ForeArm1"), ("forearm2", "ForeArm2"),
                          ("hand", "Hand")):
            idx = int(robot.skin.node_findup[f"{bone_side}{bone}"])
            block = global_t[idx][:3, :3]
            norms = np.linalg.norm(block, axis=1, keepdims=True)
            norms[norms < 1e-12] = 1.0
            bones[key] = np.round(M_W_I @ (block / norms).T, 6).tolist()

        report[side] = {
            "R_hand_world": np.round(R_hand, 6).tolist(),
            "forearm_dir_world": np.round(fore, 6).tolist(),
            "dir_upper_world": np.round(upper, 6).tolist(),
            "R_bones_world": bones,
            "reach_az": round(az, 4),
            "source_clip": clip,
            "source_frame": int(frame),
            "down": round(down, 4),
        }
        print(f"{side}: {clip}[{frame}] down={down:.4f} "
              f"fore={np.round(fore, 3).tolist()} "
              f"upper={np.round(upper, 3).tolist()} az={az:.3f}", flush=True)

    with open(OUT, "w") as f:
        json.dump(report, f, indent=1)
    print(f"saved {OUT}", flush=True)


if __name__ == "__main__":
    main()
