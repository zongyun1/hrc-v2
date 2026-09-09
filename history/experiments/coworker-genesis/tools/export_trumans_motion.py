"""TRUMANS -> the frozen MotionSequence schema, in RoboCasa/Genesis world coordinates.

``sample_hsi.sample_wrapper`` throws away the SMPL-X parameters and returns only mesh
vertices, which cannot drive an articulated skeleton. Rather than fork their sampler, we
intercept ``joints_to_smpl`` (which already computes ``pose``/``transl`` internally) and
keep what it hands back. Note their returned vertices lag the returned parameters by one
Adam step, so vertices here are always recomputed from the parameters.

Two corrections are applied on the way out, both measured rather than assumed:

**Frame change.** TRUMANS works in a Y-up grid centred on the scene; the schema mandates
the scene's own Z-up world frame. Inverting the mapping that
:mod:`tools.voxelize_kitchen_for_trumans` baked into the occupancy grid gives the rigid
transform ``p_RC = M @ p_T + c``, and for SMPL-X that means

    R_root_RC = M @ R_root_T
    transl_RC = M @ (transl_T + J0_rest) + c - J0_rest

with the ``J0_rest`` terms because SMPL-X applies the root rotation *about the rest
pelvis*, not about the origin. Body-joint rotations are parent-relative and pass through
untouched. The result is checked against the rigid map before anything is written.

**Grounding.** TRUMANS puts the soles a systematic few centimetres below its own floor
plane (the occupancy convention fills the y=0 voxel layer). The sequence is shifted
vertically so the median sole height sits on z=0, which grounds the walk on the RoboCasa
floor without touching the motion itself.

The sampler is unseeded and stochastic, and a sizeable minority of draws are degenerate —
the body crouches and floats half a metre up (see :mod:`tools.probe_trumans_variance`). So
draws are gated on grounding quality and retried, which is the first, offline instance of
the plan's rejection-sampling filter.

Usage (TRUMANS venv, cwd must be trumans_utils so ./Checkpoints etc. resolve):
  cd third_party/trumans_utils && ../trumans-venv/bin/python ../../tools/export_trumans_motion.py \
      --out ../../exports/motion_kitchen_walk.npz
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import yaml
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "hpmm"))
sys.path.insert(0, os.getcwd())  # TRUMANS modules are found relative to its own tree
from hpmm_motion.io.schema import MotionSequence  # noqa: E402

import sample_hsi  # noqa: E402
import smplx  # noqa: E402

# T -> RC, the inverse of the voxelizer's cyclic remap (Tx=RCy-ycen, Ty=RCz, Tz=RCx-xcen).
M_T_TO_RC = np.array([[0.0, 0.0, 1.0],
                      [1.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0]])

FLOAT_TOL = 0.05      # m above the floor before a frame counts as floating
MAX_FLOAT_FRAC = 0.05  # reject a draw if more than this fraction of frames float
MIN_HEIGHT = 1.65      # m, median standing height; a crouched degenerate draw falls below


def capture_smplx_params():
    """Wrap ``sample_hsi.joints_to_smpl`` so its SMPL-X output survives the call."""
    captured = {}
    original = sample_hsi.joints_to_smpl

    def wrapper(*args, **kwargs):
        result = original(*args, **kwargs)
        captured.update(zip(("pose", "transl", "left_hand", "right_hand", "vertices"), result))
        return result

    sample_hsi.joints_to_smpl = wrapper
    return captured


def smplx_forward(model_path, gender, pose, transl, left_hand, right_hand, device):
    """SMPL-X joints and vertices for a whole sequence."""
    n = pose.shape[0]
    model = smplx.create(model_path, model_type="smplx", gender=gender, ext="npz",
                         num_betas=10, use_pca=False, batch_size=n).to(device).eval()
    t = lambda a: torch.as_tensor(a, dtype=torch.float32, device=device)
    with torch.no_grad():
        out = model(transl=t(transl), global_orient=t(pose[:, :3]), body_pose=t(pose[:, 3:]),
                    left_hand_pose=t(left_hand), right_hand_pose=t(right_hand), return_verts=True)
    return out.joints.cpu().numpy().astype(np.float64), out.vertices.cpu().numpy().astype(np.float64)


def to_world(pose, transl, rest_pelvis, offset):
    """Map SMPL-X params from the TRUMANS grid frame into the scene world frame."""
    root_t = Rotation.from_rotvec(pose[:, :3]).as_matrix()
    global_orient = Rotation.from_matrix(M_T_TO_RC @ root_t).as_rotvec()
    transl_rc = (transl + rest_pelvis) @ M_T_TO_RC.T + offset - rest_pelvis
    return global_orient, transl_rc


def grounding_quality(verts_world):
    """Sole height per frame, plus the two statistics the draw is gated on."""
    sole = verts_world[..., 2].min(axis=1)
    height = verts_world[..., 2].max(axis=1) - sole
    ground_offset = float(np.median(sole))
    grounded = sole - ground_offset
    return {
        "ground_offset": ground_offset,
        "float_frac": float((grounded > FLOAT_TOL).mean()),
        "height_med": float(np.median(height)),
        "sole_p95": float(np.percentile(grounded, 95)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../../exports/motion_kitchen_walk.npz")
    ap.add_argument("--transform", default="../../exports/kitchen_transform.json")
    ap.add_argument("--scene_name", default="kitchen")
    ap.add_argument("--gender", default="male")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_draws", type=int, default=6, help="rejection-sampling budget")
    ap.add_argument("--no_ground", action="store_true", help="skip the vertical correction")
    # Default path: down the open floor in front of the counter (Phase 0.4-B's route).
    ap.add_argument("--tx", type=float, default=-0.2, help="lateral offset in TRUMANS coords")
    ap.add_argument("--z0", type=float, default=-2.3)
    ap.add_argument("--z1", type=float, default=2.3)
    ap.add_argument("--n_waypoints", type=int, default=120)
    args = ap.parse_args()

    with open(args.transform) as f:
        kitchen_tf = json.load(f)
    offset = np.array([kitchen_tf["xcen"], kitchen_tf["ycen"], 0.0])

    with open("config/config_sample_synhsi.yaml") as f:
        cfg = yaml.safe_load(f)
    cfg["scene_name"] = args.scene_name
    with open("config/config_sample_synhsi.yaml", "w") as f:
        yaml.safe_dump(cfg, f)

    waypoints = [{"x": args.tx, "y": 0.0, "z": float(z)}
                 for z in np.linspace(args.z0, args.z1, args.n_waypoints)]
    captured = capture_smplx_params()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    accepted = None
    rejected = []
    for draw in range(args.max_draws):
        seed = args.seed + draw
        torch.manual_seed(seed)
        np.random.seed(seed)
        sample_hsi.sample_wrapper(waypoints, {})
        if not captured:
            raise RuntimeError("joints_to_smpl was never called; TRUMANS internals changed")

        pose = np.asarray(captured["pose"], dtype=np.float64)          # (T, 66)
        transl = np.asarray(captured["transl"], dtype=np.float64)      # (T, 3)
        left_hand = np.asarray(captured["left_hand"], dtype=np.float64)
        right_hand = np.asarray(captured["right_hand"], dtype=np.float64)

        joints_t, _ = smplx_forward("./smpl_models", args.gender, pose, transl,
                                    left_hand, right_hand, device)
        rest_pelvis = joints_t[0, 0] - transl[0]  # J0_rest at betas=0

        global_orient, transl_rc = to_world(pose, transl, rest_pelvis, offset)
        pose_rc = pose.copy()
        pose_rc[:, :3] = global_orient
        joints_rc, verts_rc = smplx_forward("./smpl_models", args.gender, pose_rc, transl_rc,
                                            left_hand, right_hand, device)

        # The frame change must be exactly the rigid map it claims to be.
        residual = np.abs(joints_rc - (joints_t @ M_T_TO_RC.T + offset)).max()
        if residual > 1e-3:
            raise RuntimeError(f"world transform is not rigid-consistent: {residual:.6f} m")

        q = grounding_quality(verts_rc)
        ok = q["float_frac"] <= MAX_FLOAT_FRAC and q["height_med"] >= MIN_HEIGHT
        print(f"[draw {draw} seed {seed}] frames={pose.shape[0]} "
              f"float={100*q['float_frac']:.1f}% height={q['height_med']:.3f} "
              f"ground_offset={q['ground_offset']:+.3f} residual={residual*1e3:.4f}mm "
              f"-> {'ACCEPT' if ok else 'reject'}", flush=True)
        if ok:
            accepted = dict(seed=seed, pose=pose_rc, transl=transl_rc, left_hand=left_hand,
                            right_hand=right_hand, joints=joints_rc, verts=verts_rc, quality=q)
            break
        rejected.append({"seed": seed, **q})

    if accepted is None:
        raise RuntimeError(f"no draw passed the grounding gate in {args.max_draws} tries: {rejected}")

    pose_rc = accepted["pose"]
    transl_rc = accepted["transl"]
    joints_rc = accepted["joints"]
    verts_rc = accepted["verts"]
    q = accepted["quality"]
    n_frames = pose_rc.shape[0]

    shift = 0.0 if args.no_ground else q["ground_offset"]
    transl_rc = transl_rc - np.array([0.0, 0.0, shift])
    joints_rc = joints_rc - np.array([0.0, 0.0, shift])
    verts_rc = verts_rc - np.array([0.0, 0.0, shift])
    print(f"grounding shift applied: {-shift:+.3f} m")
    print(f"world bounds  x[{verts_rc[...,0].min():.2f},{verts_rc[...,0].max():.2f}] "
          f"y[{verts_rc[...,1].min():.2f},{verts_rc[...,1].max():.2f}] "
          f"z[{verts_rc[...,2].min():.3f},{verts_rc[...,2].max():.2f}]")

    seq = MotionSequence(
        transl=transl_rc[None].astype(np.float32),
        global_orient=pose_rc[:, :3][None].astype(np.float32),
        body_pose=pose_rc[:, 3:].reshape(n_frames, 21, 3)[None].astype(np.float32),
        betas=np.zeros((1, 10), np.float32),
        fps=args.fps,
        gender=args.gender,
        left_hand_pose=accepted["left_hand"].reshape(n_frames, 15, 3)[None].astype(np.float32),
        right_hand_pose=accepted["right_hand"].reshape(n_frames, 15, 3)[None].astype(np.float32),
        joints=joints_rc[None].astype(np.float32),
        meta={
            "backend": "trumans",
            "scene_name": args.scene_name,
            "frame": "robocasa_world_zup",
            "kitchen_transform": kitchen_tf,
            "seed": accepted["seed"],
            "rejected_draws": rejected,
            "quality": q,
            "grounding_shift": -shift,
            "waypoints_trumans_frame": [[w["x"], w["y"], w["z"]] for w in waypoints],
            "note": ("fps not reported by TRUMANS; 30 assumed (interp_s=3 over step=3). "
                     "Vertices recomputed from parameters, not taken from TRUMANS output."),
        },
    )
    out = os.path.abspath(args.out)
    seq.save(out)
    print(f"saved {out}: {seq}")


if __name__ == "__main__":
    main()
