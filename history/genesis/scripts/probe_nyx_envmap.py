#!/usr/bin/env python3
"""Probe: render one NYX frame with an optional EXR environment map.

NYX allows only one scene build per process, so this renders exactly one
configuration per invocation; loop configurations in the shell.

Usage:
    python scripts/probe_nyx_envmap.py OUT.png [ENV_EXR [ROTATION_DEG [MULTIPLIER]]]
"""

from __future__ import annotations

import sys

import imageio.v2 as imageio
import genesis as gs


def main() -> None:
    out_png = sys.argv[1]
    env_exr = sys.argv[2] if len(sys.argv) > 2 else None
    rot_deg = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
    mult = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0

    gs.init(backend=gs.gpu, logging_level="warning")

    from gs_nyx_plugin.nyx_camera_options import NyxCameraOptions
    from gs_nyx_plugin import nyx_camera_sensor as _s  # noqa: F401 - registers sensor type
    import gs_nyx.nyx_py_sdk as nps

    env_maps = ()
    if env_exr:
        em = nps.EnvironmentMapAsset()
        em.texture = env_exr
        em.layout = nps.EEnvMapLayout.LongLat
        em.rotation = rot_deg * 3.14159265 / 180.0
        em.multiplier = mult
        env_maps = (em,)
        print(f"env_map: {env_exr} rot={rot_deg}deg mult={mult}")
    else:
        print("env_map: none (baseline)")

    scene = gs.Scene(show_viewer=False)
    scene.add_entity(gs.morphs.Plane())
    scene.add_entity(gs.morphs.Box(size=(0.2, 0.2, 0.2), pos=(0.0, 0.0, 0.1)))
    camera = scene.add_sensor(
        NyxCameraOptions(
            res=(640, 480),
            pos=(1.2, -1.2, 0.7),
            lookat=(0.0, 0.0, 0.15),
            spp=4,
            denoise=True,
            open_window=False,
            env_maps=env_maps,
        )
    )

    scene.build()
    scene.step()
    data = camera.read()
    rgb = data.rgb
    if hasattr(rgb, "cpu"):
        rgb = rgb.cpu().numpy()
    rgb = rgb.squeeze()
    print(f"rgb shape={rgb.shape} mean={float(rgb.mean()):.2f}")
    imageio.imwrite(out_png, rgb[..., :3])
    print(f"saved {out_png}")


if __name__ == "__main__":
    main()
