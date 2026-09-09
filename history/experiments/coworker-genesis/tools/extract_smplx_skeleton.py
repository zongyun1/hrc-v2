"""Distil the SMPL-X body model into a committable skeleton template (JSON).

``hpmm_sim`` builds the Genesis-side rigid human from this template, so the Genesis venv
never has to link against ``smplx`` or ship the (licence-encumbered, 100 MB) model file.
Only rest-pose bone geometry is extracted, no mesh data.

Limb thickness is *derived* rather than hand-tuned: each vertex is assigned to the joint
carrying its largest LBS weight, and a bone's capsule radius is the 75th percentile of
the perpendicular distance of its own vertices from the bone axis. That reproduces the
real taper (forearm thinner than thigh) without a table of magic numbers.

Usage (TRUMANS venv):
  cd third_party/trumans_utils && ../trumans-venv/bin/python ../../tools/extract_smplx_skeleton.py
"""

import argparse
import json
import os
import sys

import numpy as np
import smplx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "hpmm"))
from hpmm_motion.io.schema import BODY_JOINT_NAMES, BODY_JOINT_PARENTS  # noqa: E402

N_JOINTS = len(BODY_JOINT_NAMES)  # 22 body joints; fingers/face are not rigid links here
RADIUS_PCTL = 75.0  # per-bone flesh radius percentile
MIN_RADIUS = 0.025  # m, keeps degenerate bones (e.g. collars) from collapsing


def bone_radius(pts, a, b):
    """Percentile perpendicular distance from the segment a->b, of the flesh beside it.

    Points whose projection falls outside the segment are dropped rather than clamped to
    an endpoint: the hand sits past the wrist and the deltoid behind the shoulder, and
    folding them onto the endpoint would inflate the limb to the width of the blob.
    """
    axis = b - a
    length = np.linalg.norm(axis)
    if length < 1e-6 or len(pts) == 0:
        return MIN_RADIUS
    axis = axis / length
    rel = pts - a
    t = rel @ axis
    inside = (t >= 0.0) & (t <= length)
    if inside.sum() < 8:
        return MIN_RADIUS
    perp = np.linalg.norm(rel[inside] - t[inside, None] * axis, axis=1)
    return float(max(np.percentile(perp, RADIUS_PCTL), MIN_RADIUS))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="./smpl_models")
    ap.add_argument("--gender", default="male")
    ap.add_argument("--out", default="../../hpmm/assets/smplx_skeleton_male.json")
    ap.add_argument("--skin_out", default="../../hpmm/assets/smplx_skin_male.npz")
    ap.add_argument("--skin_k", type=int, default=4, help="LBS weights kept per vertex")
    args = ap.parse_args()

    model = smplx.create(args.model_path, model_type="smplx", gender=args.gender,
                         ext="npz", num_betas=10, use_pca=False, batch_size=1)
    out = model(return_verts=True)  # all-zero params == rest pose, betas = 0
    joints = out.joints[0, :N_JOINTS].detach().cpu().numpy().astype(np.float64)
    verts = out.vertices[0].detach().cpu().numpy().astype(np.float64)
    parents = model.parents[:N_JOINTS].cpu().numpy()
    weights = model.lbs_weights.detach().cpu().numpy()  # (V, 55)

    assert list(parents) == list(BODY_JOINT_PARENTS), \
        f"schema parents drifted from the model: {list(parents)}"

    # Assign each vertex to its dominant joint, folding finger/face joints back onto the
    # nearest body ancestor so hands and head keep their bulk.
    full_parents = model.parents.cpu().numpy()
    fold = np.arange(len(full_parents))
    for j in range(len(full_parents)):
        k = j
        while k >= N_JOINTS:
            k = full_parents[k]
        fold[j] = k
    owner = fold[weights.argmax(axis=1)]

    children = {j: [c for c in range(N_JOINTS) if parents[c] == j] for j in range(N_JOINTS)}
    bones = []
    for j in range(N_JOINTS):
        own = verts[owner == j]
        if children[j]:
            # One capsule per child: the flesh between this joint and each of its children.
            for c in children[j]:
                pts = verts[(owner == j) | (owner == c)]
                bones.append({
                    "body": BODY_JOINT_NAMES[j],
                    "name": f"{BODY_JOINT_NAMES[j]}__{BODY_JOINT_NAMES[c]}",
                    "from": (joints[j] - joints[j]).tolist(),          # body-local: origin
                    "to": (joints[c] - joints[j]).tolist(),            # body-local: child
                    "radius": bone_radius(pts, joints[j], joints[c]),
                })
        else:
            # Leaf (head, feet, wrists): extend a capsule from the joint through the
            # centroid of its own flesh, so the head points up and the feet point forward.
            if len(own) < 8:
                continue
            direction = own.mean(axis=0) - joints[j]
            direction /= max(np.linalg.norm(direction), 1e-9)
            tip = joints[j] + direction * ((own - joints[j]) @ direction).max()
            bones.append({
                "body": BODY_JOINT_NAMES[j],
                "name": f"{BODY_JOINT_NAMES[j]}__tip",
                "from": (joints[j] - joints[j]).tolist(),
                "to": (tip - joints[j]).tolist(),
                "radius": bone_radius(own, joints[j], tip),
            })

    template = {
        "model": "smplx",
        "gender": args.gender,
        "n_joints": N_JOINTS,
        "joint_names": list(BODY_JOINT_NAMES),
        "parents": [int(p) for p in parents],
        # Parent-relative rest offsets: the only translation-invariant form, and exactly
        # what an MJCF body's `pos` attribute wants.
        "rest_offsets": [
            (joints[j] - (joints[parents[j]] if parents[j] >= 0 else 0.0)).tolist()
            for j in range(N_JOINTS)
        ],
        "rest_joints": joints.tolist(),
        "bones": bones,
        "height": float(verts[:, 1].max() - verts[:, 1].min()),
        "note": "betas=0 rest pose; SMPL-X model frame is Y-up, converted downstream.",
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(template, f, indent=1)

    # Skinning asset: enough to deform the real SMPL-X surface from the rigid skeleton's
    # link poses, so the Genesis side can show a body rather than a pile of capsules
    # without linking against `smplx` or streaming a vertex trajectory. Finger and face
    # joints are folded onto their nearest body ancestor (`fold`), matching the 22 links
    # the skeleton actually has; only the k largest weights per vertex are kept, which is
    # what LBS implementations use in practice and keeps the file small.
    folded = np.zeros((weights.shape[0], N_JOINTS), dtype=np.float64)
    np.add.at(folded, (np.arange(weights.shape[0])[:, None], fold[None, :].repeat(weights.shape[0], 0)),
              weights)
    order = np.argsort(-folded, axis=1)[:, :args.skin_k]
    vals = np.take_along_axis(folded, order, axis=1)
    vals /= np.maximum(vals.sum(axis=1, keepdims=True), 1e-9)
    np.savez_compressed(
        os.path.abspath(args.skin_out),
        rest_verts=verts.astype(np.float32),
        faces=model.faces.astype(np.int32),
        rest_joints=joints.astype(np.float32),
        weight_idx=order.astype(np.int16),
        weight_val=vals.astype(np.float32),
        joint_names=np.array(list(BODY_JOINT_NAMES)),
    )
    print(f"skin: {len(verts)} verts, {len(model.faces)} faces, top-{args.skin_k} weights "
          f"-> {os.path.abspath(args.skin_out)}")
    print(f"{len(bones)} bones, {N_JOINTS} joints, rest height {template['height']:.3f} m")
    print(f"radius range {min(b['radius'] for b in bones):.3f}..{max(b['radius'] for b in bones):.3f} m")
    print("saved", os.path.abspath(args.out))


if __name__ == "__main__":
    main()
