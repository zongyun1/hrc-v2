"""Smoke test: bring up the pour_water scene with each of the new robots.

Builds the scene + loads actors + loads the robot for each of
{arx_x5, ur5_wsg}, confirms the robot's home pose snaps cleanly, and prints
the resulting EE pose. Skips the avatar drink animation and the grasp/pour
phases — those need per-robot grasp YAMLs that don't exist yet.
"""

import os
import sys
import time
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from envs.tasks import TASK_MAP


def smoke(robot_type: str):
    print(f"\n=== pour_water smoke: robot_type={robot_type} ===")
    config = {
        "robot_type": robot_type,
        "show_viewer": False,
        "save_video": False,
        # Suppress avatar to keep this fast — pour_water can run without the
        # drink phase as long as we don't call play_once().
        "avatar": {
            "motion_data": "avatars/motions/motion.pkl",
            "skin_options": {
                "glb_path": "avatars/models/custom_Adrian_Keller.glb",
                "euler": [-90, 0, 90],
                "pos": [0.0, 0.0, -0.959008030],
            },
            "frame_ratio": 1.0,
        },
    }
    TaskClass = TASK_MAP["pour_water"]
    t0 = time.time()
    task = TaskClass(config)
    task.reset(seed=0)
    elapsed = time.time() - t0
    arm = task.robot.get_arm("right")
    ee = arm.get_ee_pose()
    print(f"  reset OK in {elapsed:.1f}s")
    print(f"  arm.n_arm={arm.n_arm}")
    print(f"  arm.tcp_offset.p={arm.tcp_offset.p.tolist()}")
    print(f"  ee_pose=[{', '.join(f'{x:+.3f}' for x in ee)}]")
    if arm.planner is not None:
        print(f"  planner={type(arm.planner).__name__}")
    print(f"  {robot_type}: SCENE/ROBOT OK")


def main():
    targets = sys.argv[1:] or ["arx_x5", "ur5_wsg"]
    failed = []
    for r in targets:
        try:
            smoke(r)
        except Exception as e:
            print(f"  {r}: FAILED — {type(e).__name__}: {e}")
            traceback.print_exc()
            failed.append(r)
    print(f"\nDone. Failed: {failed if failed else 'none'}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
