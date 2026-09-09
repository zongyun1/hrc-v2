"""Probe: avatar pick_and_place palm-down + easing (before/after review videos).

Runs scripts/collect.py on an assist task with the avatar PickPlaceMotion in
either legacy mode (--mode before: palm_down/ease off) or the new default
(--mode after), and logs the palm normal at the attach/detach keyframes.

Palm-down success metric: right hand e3_z ~= -1, left hand e3_z ~= +1
(the e1 x e2 hand-frame normal mirrors between sides).

Usage (compute node, MAWM_latest_genesis python, repo root):
    python scripts/probe_avatar_palm_down.py --mode after \
        --task blocks_ranking_size_assist --save-dir data/motion/pp_after
"""
import argparse
import os
import runpy
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["before", "after"], required=True)
    parser.add_argument("--task", default="blocks_ranking_size_assist")
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--video-stride", type=int, default=None)
    args = parser.parse_args()

    import envs.avatar.motions.pick_place_motion as ppm

    orig_start = ppm.PickPlaceMotion.start

    def probed_start(self, *a, **kw):
        if args.mode == "before":
            kw["palm_down"] = False
            kw["ease_in_out"] = False
        result = orig_start(self, *a, **kw)
        try:
            if self.frames:
                saved = self.robot.node_trans
                n = len(self.frames)
                marks = [("attach", self.attach_frame),
                         ("detach", self.detach_frame),
                         ("mid_transport",
                          (self.attach_frame + self.detach_frame) // 2)]
                for label, fi in marks:
                    fi = int(np.clip(int(fi), 0, n - 1))
                    self.robot.node_trans = self.frames[fi]
                    self.robot.update()
                    _, R_hand = self.robot._get_hand_frame(self._hand_id)
                    e3 = R_hand[:, 2]
                    print(
                        f"[palm_probe] mode={args.mode} hand={self._hand_id} "
                        f"{label} frame={fi}/{n} e3={np.round(e3, 3).tolist()}",
                        flush=True,
                    )

                # Arm-torso clearance: min horizontal distance from the
                # shoulder->elbow and elbow->wrist segments to the body's
                # vertical axis, over all frames.  Torso capsule radius is
                # ~0.13m, so values well below that mean penetration.
                side = "Left" if int(self._hand_id) == 0 else "Right"
                body_xy = np.asarray(
                    self.robot.get_global_pose()[:2], dtype=np.float64
                )
                min_clear, min_fi = 1e9, -1
                for fi in range(0, n, 4):
                    self.robot.node_trans = self.frames[fi]
                    self.robot.update()
                    pts = {}
                    for key, bone in (("s", "Arm"), ("e", "ForeArm"),
                                      ("w", "Hand")):
                        pts[key] = np.asarray(
                            self.robot.skin.get_global_translation(
                                f"{side}{bone}")[0],
                            dtype=np.float64,
                        ).ravel()[:3]
                    for a_pt, b_pt in ((pts["s"], pts["e"]),
                                       (pts["e"], pts["w"])):
                        for t in np.linspace(0.0, 1.0, 6):
                            p = a_pt * (1 - t) + b_pt * t
                            if 0.65 <= p[2] <= 1.5:  # torso band
                                d = float(np.linalg.norm(p[:2] - body_xy))
                                if d < min_clear:
                                    min_clear, min_fi = d, fi
                # Axis-distance calibration: the idle stance itself puts the
                # elbow ~0.10-0.12m from the body axis, so only clearly
                # smaller values indicate the arm inside the torso volume.
                label = ("PENETRATION" if min_clear < 0.05
                         else "close" if min_clear < 0.10 else "ok")
                print(
                    f"[palm_probe] mode={args.mode} hand={self._hand_id} "
                    f"min_arm_torso_clearance={min_clear:.3f}m "
                    f"at frame {min_fi}/{n} ({label})",
                    flush=True,
                )
                self.robot.node_trans = saved
                self.robot.update()
        except Exception as exc:  # metric only — never break the run
            print(f"[palm_probe] metric failed: {exc!r}", flush=True)
        return result

    ppm.PickPlaceMotion.start = probed_start

    argv = [
        "collect.py",
        "--task", args.task,
        "--episodes", str(args.episodes),
        "--save-dir", args.save_dir,
        "--start-seed", str(args.start_seed),
    ]
    if args.video_stride is not None:
        argv += ["--video-stride", str(args.video_stride)]
    sys.argv = argv
    runpy.run_path(os.path.join(ROOT, "scripts", "collect.py"),
                   run_name="__main__")


if __name__ == "__main__":
    main()
