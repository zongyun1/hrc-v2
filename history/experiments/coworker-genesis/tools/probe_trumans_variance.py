"""How often does a TRUMANS sample come out usable?

The sampler is unseeded and stochastic, and repeated draws on the identical kitchen
trajectory have visibly different quality: some walk, some end up crouched and floating
half a metre off the floor. Phase 1 needs a rejection filter, so first measure what it
would be rejecting.

Grounding metric per draw: the height of the lowest body vertex in each frame. A clean
walk keeps that within a few cm of the floor for every frame; a failed draw floats.

Usage (TRUMANS venv, cwd = trumans_utils):
  ../trumans-venv/bin/python ../../tools/probe_trumans_variance.py --draws 4
"""

import argparse
import os
import sys

import numpy as np
import torch
import smplx

sys.path.insert(0, os.getcwd())
import sample_hsi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=4)
    ap.add_argument("--tx", type=float, default=-0.2)
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    args = ap.parse_args()

    captured = {}
    original = sample_hsi.joints_to_smpl

    def wrapper(*a, **k):
        r = original(*a, **k)
        captured.update(zip(("pose", "transl", "lh", "rh", "verts"), r))
        return r

    sample_hsi.joints_to_smpl = wrapper

    waypoints = [{"x": args.tx, "y": 0.0, "z": float(z)}
                 for z in np.linspace(-2.3, 2.3, 120)]
    seeds = args.seeds if args.seeds else list(range(args.draws))

    print(f"\n{'seed':>5} {'frames':>7} {'sole_med':>9} {'sole_p95':>9} {'float%':>7} "
          f"{'sink%':>6} {'height':>7} {'path_m':>7}")
    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        sample_hsi.sample_wrapper(waypoints, {})
        pose = np.asarray(captured["pose"], np.float64)
        transl = np.asarray(captured["transl"], np.float64)
        lh, rh = (np.asarray(captured[k], np.float64) for k in ("lh", "rh"))
        n = pose.shape[0]

        model = smplx.create("./smpl_models", model_type="smplx", gender="male", ext="npz",
                             num_betas=10, use_pca=False, batch_size=n).cuda().eval()
        t = lambda a: torch.as_tensor(a, dtype=torch.float32, device="cuda")
        with torch.no_grad():
            out = model(transl=t(transl), global_orient=t(pose[:, :3]), body_pose=t(pose[:, 3:]),
                        left_hand_pose=t(lh), right_hand_pose=t(rh), return_verts=True)
        v = out.vertices.cpu().numpy()  # TRUMANS frame, Y-up

        sole = v[..., 1].min(axis=1)                      # lowest vertex per frame
        height = (v[..., 1].max(axis=1) - sole)           # standing height per frame
        path = np.linalg.norm(np.diff(transl[:, [0, 2]], axis=0), axis=1).sum()
        print(f"{seed:>5} {n:>7} {np.median(sole):>9.3f} {np.percentile(sole, 95):>9.3f} "
              f"{100 * (sole > 0.05).mean():>7.1f} {100 * (sole < -0.05).mean():>6.1f} "
              f"{np.median(height):>7.3f} {path:>7.2f}", flush=True)


if __name__ == "__main__":
    main()
