#!/usr/bin/env python3
"""Time NYX scene build + steady-state render, with/without an EXR env map.

One configuration per process (NYX single-build limit). Prints build time and
per-frame render stats.

Usage: time_nyx_envmap.py <label> <spp> <denoise:0|1> [ENV_EXR]
"""

from __future__ import annotations

import sys
import time

import genesis as gs


def main() -> None:
    label = sys.argv[1]
    spp = int(sys.argv[2])
    denoise = bool(int(sys.argv[3]))
    env_exr = sys.argv[4] if len(sys.argv) > 4 else None

    gs.init(backend=gs.gpu, logging_level="warning")

    from gs_nyx_plugin.nyx_camera_options import NyxCameraOptions
    from gs_nyx_plugin import nyx_camera_sensor as _s  # noqa: F401
    import gs_nyx.nyx_py_sdk as nps

    env_maps = ()
    if env_exr:
        em = nps.EnvironmentMapAsset()
        em.texture = env_exr
        em.layout = nps.EEnvMapLayout.LongLat
        env_maps = (em,)

    scene = gs.Scene(show_viewer=False)
    scene.add_entity(gs.morphs.Plane(visualization=not env_exr))
    scene.add_entity(gs.morphs.Box(size=(0.2, 0.2, 0.2), pos=(0.0, 0.0, 0.1)))
    # 5 sensors like the real task camera set (head + 2 wrist + recording + side)
    cams = []
    for i in range(5):
        cams.append(scene.add_sensor(NyxCameraOptions(
            res=(640, 480),
            pos=(1.2 - 0.1 * i, -1.2, 0.7),
            lookat=(0.0, 0.0, 0.15),
            spp=spp,
            denoise=denoise,
            open_window=False,
            env_maps=env_maps if i == 0 else (),
        )))

    t0 = time.perf_counter()
    scene.build()
    t_build = time.perf_counter() - t0

    # Warmup (first renders compile/caches)
    for _ in range(5):
        scene.step()
        for c in cams:
            c.read()

    n = 60
    t0 = time.perf_counter()
    for _ in range(n):
        scene.step()
        for c in cams:
            c.read()
    dt = time.perf_counter() - t0

    per_frame_ms = dt / n * 1000.0
    print(f"RESULT {label}: build={t_build:.1f}s  "
          f"step+5cam_render={per_frame_ms:.1f}ms/frame  ({n} frames)")


if __name__ == "__main__":
    main()
