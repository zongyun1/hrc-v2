"""Probe: NYX URDF sub-scene rendering of the neutral chair/laptop props.

Renders the ensure_nyx_urdf-baked chair + laptop (fixed joints pre-merged,
per-link visuals merged to one OBJ, scale baked) next to the production
cabinet reference, with BOTH a NYX sensor and a rasterizer camera at the same
viewpoint. The rasterizer image is the simulation ground truth; differences
are NYX rendering artifacts.

Run on a GPU node (MAWM_nyx_clean, GENESIS_BACKEND=gpu).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from envs.genesis_compat import ensure_inertial_urdf, ensure_nyx_urdf, get_genesis

CHAIR = "assets/objects/sapien-chair-179/mobility.urdf"
LAPTOP = "assets/objects/015_laptop/9960/mobility.urdf"
CABINET = "assets/objects/036_cabinet/46653/mobility.urdf"


def main():
    gs = get_genesis()
    gs.init(backend=gs.gpu, logging_level="warning")
    from gs_nyx_plugin.nyx_camera_options import NyxCameraOptions
    from gs_nyx_plugin import nyx_camera_sensor  # noqa: F401

    chair_nyx = str(ensure_nyx_urdf(
        ensure_inertial_urdf(Path(CHAIR).resolve()), 0.585))
    laptop_nyx = str(ensure_nyx_urdf(
        ensure_inertial_urdf(Path(LAPTOP).resolve()), 0.28))
    cabinet = str(ensure_inertial_urdf(Path(CABINET).resolve()))

    scene = gs.Scene(show_viewer=False)
    scene.add_entity(gs.morphs.Plane())
    specs = [
        (chair_nyx, 1.0, (-1.2, 0.0, 0.239)),
        (laptop_nyx, 1.0, (0.0, 0.0, 0.084)),
        (cabinet, 0.20, (1.2, 0.0, 0.28)),
    ]
    for path, scale, pos in specs:
        scene.add_entity(gs.morphs.URDF(
            file=path, pos=pos, scale=float(scale), fixed=True,
            merge_fixed_links=False,
        ))

    cam_kw = dict(res=(1600, 700), pos=(0.0, -3.2, 1.5), lookat=(0.0, 0.0, 0.3))
    nyx_cam = scene.add_sensor(NyxCameraOptions(
        spp=4, denoise=True, open_window=False, **cam_kw))
    ras_cam = scene.add_camera(fov=40, GUI=False, **cam_kw)
    scene.build()
    for _ in range(5):
        scene.step()

    import imageio.v2 as imageio
    data = nyx_cam.read()
    rgb = data.rgb.cpu().numpy() if hasattr(data.rgb, "cpu") else np.asarray(data.rgb)
    imageio.imwrite("data/neut/probe/nyx_urdf_probe_nyx.png", rgb.astype(np.uint8))
    ras = ras_cam.render(rgb=True)[0]
    imageio.imwrite("data/neut/probe/nyx_urdf_probe_raster.png",
                    np.asarray(ras).astype(np.uint8))
    print("saved data/neut/probe/nyx_urdf_probe_{nyx,raster}.png")


if __name__ == "__main__":
    main()
