"""Probe: neutral-mixin prop orientations under latest Genesis (z-up GLB load).

Spawns every prop the neutral avatar mixin uses, once with the CURRENT spawn
quat from envs/task_bases/neutral_avatar_table_work.py and once with the
PROPOSED fix (drop the legacy x90 y-up compensation). Renders at-spawn and
after settling, plus prints world AABB extents per object.

Run (compute node, CPU backend):
    python scripts/probe_neutral_props.py --out data/neut/probe
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import transforms3d as t3d

from envs.genesis_compat import get_genesis, genesis_backend_from_env
from envs.utils import ASSETS_PATH, Pose, load_mesh, load_object, to_numpy

X90 = np.array([0.7071067811865476, 0.7071067811865475, 0.0, 0.0])
IDENT = np.array([1.0, 0.0, 0.0, 0.0])


def q_from_R(R):
    return t3d.quaternions.mat2quat(np.asarray(R, dtype=np.float64)).astype(float)


def book_R_old(facing=0.0):
    return (
        t3d.euler.euler2mat(0.0, 0.0, facing + np.pi / 2.0, "sxyz")
        @ t3d.euler.euler2mat(np.pi / 2.0, 0.0, 0.0, "sxyz")
    )


def pen_R_old(tilt_deg=18.0):
    tilt_axis = np.array([0.0, 1.0, 0.0])
    return (
        t3d.axangles.axangle2mat(tilt_axis, np.deg2rad(tilt_deg))
        @ t3d.euler.euler2mat(np.pi / 2.0, 0.0, 0.0, "sxyz")
    )


RX_NEG90 = t3d.euler.euler2mat(-np.pi / 2.0, 0.0, 0.0, "sxyz")

# (label, kind, asset, quat_current, quat_proposed, scale_or_model, z)
SPECS = [
    ("sponge", "mesh", "cc0_sponge_3/visual/base0.glb", X90, IDENT, 1.0, 0.06),
    ("plate", "mesh", "003_plate/visual/base0.glb", X90, IDENT, 0.025, 0.06),
    ("pen", "mesh", "058_markpen/visual/base0.glb",
     q_from_R(pen_R_old()), q_from_R(pen_R_old() @ RX_NEG90), 0.1, 0.12),
    ("book", "mesh", "092_notebook/visual/base0.glb",
     q_from_R(book_R_old()), q_from_R(book_R_old() @ RX_NEG90), 0.18, 0.06),
    ("mouse", "obj", ("047_mouse", 0), X90, IDENT, None, 0.06),
    ("bottle", "obj", ("001_bottle", 13), X90, IDENT, None, 0.10),
    ("can", "obj", ("071_can", 0), X90, IDENT, None, 0.06),
    ("apple", "obj", ("035_apple", 0), X90, IDENT, None, 0.06),
    ("phone", "obj", ("077_phone", 0), X90, IDENT, None, 0.06),
]


def spawn(scene, spec, pos, quat):
    label, kind, asset, _, _, scale, z = spec
    pose = Pose([pos[0], pos[1], z], np.asarray(quat, dtype=float))
    if kind == "mesh":
        return load_mesh(
            scene, ASSETS_PATH / "objects" / asset, pose,
            scale=scale, is_static=False, convex=True, collision=True,
            friction=3.0, density=250.0,
        )
    name, mid = asset
    return load_object(
        scene, pose, name, model_id=mid, convex=True,
        is_static=False, friction=3.0, density=250.0,
    ).entity


def aabb(ent):
    vs = []
    for link in getattr(ent, "links", []) or []:
        for geom in getattr(link, "_geoms", None) or getattr(link, "geoms", []) or []:
            try:
                v = to_numpy(geom.get_verts()).reshape(-1, 3)
            except Exception:
                continue
            if v.size:
                vs.append(v)
    if not vs:
        return None
    V = np.concatenate(vs, axis=0)
    return V.min(axis=0), V.max(axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/neut/probe")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    gs = get_genesis()
    gs.init(backend=genesis_backend_from_env(gs), logging_level="warning")
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01),
        show_viewer=False,
    )
    scene.add_entity(gs.morphs.Plane())

    ents = {}
    for i, spec in enumerate(SPECS):
        label = spec[0]
        x = 0.55 * i - 2.2
        # row y=0: current quat; row y=0.6: proposed quat
        ents[label + "_cur"] = spawn(scene, spec, (x, 0.0), spec[3])
        ents[label + "_fix"] = spawn(scene, spec, (x, 0.6), spec[4])

    cam_top = scene.add_camera(
        res=(1600, 900), pos=(0.0, 0.3, 3.2), lookat=(0.0, 0.3, 0.0),
        up=(0.0, 1.0, 0.0), fov=50, GUI=False,
    )
    cam_side = scene.add_camera(
        res=(1600, 900), pos=(0.0, -2.6, 0.9), lookat=(0.0, 0.3, 0.05),
        fov=42, GUI=False,
    )
    scene.build()

    import imageio.v2 as imageio

    def snap(tag):
        for cam, view in ((cam_top, "top"), (cam_side, "side")):
            rgb = cam.render(rgb=True)[0]
            imageio.imwrite(out / f"{tag}_{view}.png", np.asarray(rgb).astype(np.uint8))

    snap("spawn")
    for _ in range(150):
        scene.step()
    snap("settled")

    print("\n=== world AABB extents (settled) ===")
    for key, ent in ents.items():
        bb = aabb(ent)
        if bb is None:
            print(f"{key:16s} no verts")
            continue
        lo, hi = bb
        ext = hi - lo
        print(f"{key:16s} ext=({ext[0]:.3f},{ext[1]:.3f},{ext[2]:.3f}) "
              f"zmin={lo[2]:.3f} quat={np.round(to_numpy(ent.get_quat()).ravel()[:4],3).tolist()}")
    print("probe done ->", out)


if __name__ == "__main__":
    main()
