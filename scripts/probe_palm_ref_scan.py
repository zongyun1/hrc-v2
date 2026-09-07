"""Scan all loaded avatar motion clips for naturally palm-down hand postures.

For every clip frame, applies the pose to the avatar skin, measures each
hand's world frame (e1=thumb axis, e2=finger axis, e3=palm normal), and
scores "palm-down-ness" (right hand: -e3_z, left hand: +e3_z — the e1xe2
normal mirrors between sides).  Reports the top frames per hand and dumps
the reference orientation of the best candidates to JSON:

    { "right": {"R_hand_world": 3x3, "reach_az": float,
                "clip": name, "frame": i, "down": d, ...}, "left": {...} }

reach_az = horizontal azimuth of the elbow->wrist forearm direction; used
at runtime to yaw-align the reference onto the IK arm.

Run (compute node, MAWM_latest_genesis python, repo root):
    python scripts/probe_palm_ref_scan.py --out data/motion/palm_ref_scan
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--task", default="blocks_ranking_size_assist")
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--top", type=int, default=25)
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    import yaml
    from envs.tasks import TASK_MAP
    from envs.avatar.utils import (
        Mixamo_data_to_controller_pose,
        Mixamo_node_processing,
    )

    with open("config/default.yml") as f:
        config = yaml.safe_load(f)
    config["use_viewer"] = False

    task = TASK_MAP[args.task](config)
    task.reset(0)
    av = task.avatar
    robot = av.robot
    vgeom = robot.skin.links[0]._vgeoms[0]

    saved_pose = robot.pose.copy()
    saved_node = robot.node_trans.copy()

    def world_pos(name):
        return np.asarray(
            robot.skin.get_global_translation(name)[0], dtype=np.float64
        ).ravel()[:3]

    saved_mat = robot.global_mat
    saved_mat_inv = robot.global_mat_inv

    results = {0: [], 1: []}
    for clip_name, md in sorted(av.motion_data.items()):
        try:
            n = int(md["trans"].shape[0])
            # Each clip carries its own bind matrices (see
            # ReplayMotionModule.__init__) — using another clip's mat
            # distorts the pose.
            clip_mat = md["mat"][0]
            clip_mat_inv = np.array([np.linalg.inv(m) for m in clip_mat])
        except Exception:
            continue
        for i in range(0, n, max(1, args.stride)):
            try:
                pose = Mixamo_data_to_controller_pose(
                    md["trans"][i], md["rot"][i], md["joint"][i]
                )
                node = Mixamo_node_processing(
                    vgeom, pose, clip_mat, clip_mat_inv
                )
                robot.pose = pose
                robot.node_trans = node
                robot.global_mat = clip_mat
                robot.global_mat_inv = clip_mat_inv
                robot.update()
            except Exception:
                break
            for hand in (0, 1):
                side = "Left" if hand == 0 else "Right"
                try:
                    hand_pos, R_hand = robot._get_hand_frame(hand)
                except Exception:
                    continue
                e3 = R_hand[:, 2]
                down = float(e3[2]) if hand == 0 else float(-e3[2])
                if down < 0.85:
                    continue
                elbow = world_pos(f"{side}ForeArm")
                shoulder = world_pos(f"{side}Arm")
                wrist = np.asarray(hand_pos, dtype=np.float64)
                forearm = wrist - elbow
                az = float(np.arctan2(forearm[1], forearm[0]))
                mid = world_pos(f"{side}HandMiddle1")
                finger = mid - wrist
                # wrist bend: angle between forearm and finger directions —
                # natural relaxed reach is ~5-40 deg.
                cosb = float(
                    np.dot(forearm, finger)
                    / (np.linalg.norm(forearm) * np.linalg.norm(finger) + 1e-9)
                )
                bend = float(np.degrees(np.arccos(np.clip(cosb, -1, 1))))
                # prefer reach-like frames: hand below shoulder
                below = float(shoulder[2] - wrist[2])
                results[hand].append({
                    "clip": clip_name, "frame": int(i), "down": round(down, 4),
                    "bend_deg": round(bend, 1), "below_shoulder": round(below, 3),
                    "reach_az": round(az, 4),
                    "R_hand_world": np.round(R_hand, 6).tolist(),
                })

    robot.pose = saved_pose
    robot.node_trans = saved_node
    robot.global_mat = saved_mat
    robot.global_mat_inv = saved_mat_inv
    robot.update()

    report = {}
    for hand in (0, 1):
        side = "left" if hand == 0 else "right"
        # Rank: palm-downness first, then natural (small-ish) wrist bend,
        # then reach-like height.
        cands = [
            r for r in results[hand]
            if r["below_shoulder"] > 0.05 and 3.0 <= r["bend_deg"] <= 55.0
        ]
        cands.sort(key=lambda r: (-r["down"], r["bend_deg"]))
        print(f"=== {side} hand: {len(results[hand])} palm-down frames, "
              f"{len(cands)} reach-like ===", flush=True)
        for r in cands[: args.top]:
            print(f"  {r['clip']}[{r['frame']}] down={r['down']} "
                  f"bend={r['bend_deg']} below={r['below_shoulder']} "
                  f"az={r['reach_az']}", flush=True)
        if cands:
            report[side] = cands[0]
            report[f"{side}_alts"] = cands[1: args.top]

    with open(os.path.join(args.out, "palm_ref_scan.json"), "w") as f:
        json.dump(report, f, indent=1)
    print(f"saved {os.path.join(args.out, 'palm_ref_scan.json')}", flush=True)


if __name__ == "__main__":
    main()
