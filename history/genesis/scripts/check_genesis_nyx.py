#!/usr/bin/env python3
"""Smoke-check the latest Genesis + Nyx environment."""

from __future__ import annotations

import torch

import genesis as gs
import gs_nyx
import gs_nyx_plugin


def main() -> None:
    print(f"genesis={gs.__version__}")
    print(f"gs_nyx={getattr(gs_nyx, '__version__', 'unknown')}")
    print(f"gs_nyx_plugin={getattr(gs_nyx_plugin, '__version__', 'unknown')}")
    print(f"torch={torch.__version__} cuda={torch.cuda.is_available()}")

    # Genesis initializes core dtypes used by the plugin sensor implementation.
    gs.init(backend=gs.gpu, logging_level="warning")

    from gs_nyx_plugin.nyx_camera_options import NyxCameraOptions
    from gs_nyx_plugin import nyx_camera_sensor as _nyx_camera_sensor  # noqa: F401 - registers sensor type

    scene = gs.Scene(show_viewer=False)
    scene.add_entity(gs.morphs.Plane())
    scene.add_entity(gs.morphs.Box(size=(0.2, 0.2, 0.2), pos=(0.0, 0.0, 0.1)))
    camera = scene.add_sensor(
        NyxCameraOptions(
            res=(64, 64),
            pos=(1.0, -1.0, 0.8),
            lookat=(0.0, 0.0, 0.1),
            spp=1,
            denoise=False,
            open_window=False,
        )
    )

    scene.build()
    scene.step()
    data = camera.read()
    mean_rgb = float(data.rgb.float().mean().item())
    print(f"nyx_rgb shape={tuple(data.rgb.shape)} dtype={data.rgb.dtype} mean={mean_rgb:.3f}")


if __name__ == "__main__":
    main()
