"""Render one frame per candidate camera placement, to check a view before committing.

A full episode render costs minutes; a camera that turns out to be facing the outside of a
wall costs that twice. This builds the scene once and dumps a still from each placement.

Usage (Genesis venv):
  cd genesis && uv run python ../tools/probe_camera.py --state ../exports/demo_....json
  cd genesis && uv run python ../tools/probe_camera.py --mjcf ../exports/robocasa_layout1_style1.xml
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "hpmm"))

import genesis as gs  # noqa: E402

from hpmm_sim.assets import load as assets  # noqa: E402
from hpmm_sim.shim.sampler import TaskInstance  # noqa: E402
from hpmm_sim.vis.camera import look_into_room, mount_camera, room_bounds  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default=None, help="task/demo state JSON")
    ap.add_argument("--mjcf", default=None, help="bare MJCF, if there is no state file")
    ap.add_argument("--lookat", type=float, nargs=3, default=None)
    ap.add_argument("--out_dir", default="../exports/camera_probe")
    args = ap.parse_args()

    instance = TaskInstance.load(args.state) if args.state else None
    mjcf = args.mjcf or str(instance.mjcf_path)

    gs.init(backend=gs.gpu)
    scene = gs.Scene(show_viewer=False, sim_options=gs.options.SimOptions(dt=0.01))
    entity = scene.add_entity(assets.mjcf(mjcf))
    cam = scene.add_camera(res=(960, 540), fov=50, GUI=False)
    scene.build()
    if instance is not None:
        instance.apply(entity, strict=False)
    scene.step()

    os.makedirs(args.out_dir, exist_ok=True)
    lo, hi = room_bounds(entity)
    print(f"room bounds lo={np.round(lo, 2)} hi={np.round(hi, 2)}")

    if args.lookat is not None:
        target = np.array(args.lookat)
    elif instance is not None and instance.objects:
        target = np.asarray(next(iter(instance.objects.values()))["pos"])
    else:
        target = 0.5 * (lo + hi)
    print(f"lookat {np.round(target, 2)}")

    shots = {}
    pos = look_into_room(cam, entity, target)
    shots["room"] = pos
    cam.render()
    _save(cam, os.path.join(args.out_dir, "room.png"))

    # RoboCasa's own mounts, if the scene came with them.
    cfgs = (json.loads(open(args.state).read()).get("cam_configs") or {}) if args.state else {}
    for name, cfg in cfgs.items():
        link_name = cfg.get("parent_body")
        try:
            link = entity.get_link(link_name)
        except Exception as exc:  # a mount whose body Genesis renamed or dropped
            print(f"  skip {name}: {exc}")
            continue
        shots[name] = mount_camera(cam, link, cfg)
        _save(cam, os.path.join(args.out_dir, f"{name}.png"))

    for name, pos in shots.items():
        print(f"  {name:28s} camera at {np.round(pos, 2)}")
    print(f"wrote {len(shots)} stills to {os.path.abspath(args.out_dir)}")


def _save(cam, path):
    import imageio.v3 as iio

    rgb = cam.render()[0]
    iio.imwrite(path, np.asarray(rgb))


if __name__ == "__main__":
    main()
