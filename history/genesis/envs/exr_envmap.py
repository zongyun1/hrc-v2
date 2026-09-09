"""Resolution-independent EXR panorama offsets for NYX environment maps."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np


def normalize_env_offset(value) -> tuple[float, float]:
    """Normalize an offset to ``(horizontal, vertical)`` panorama degrees.

    A mapping may use explicit ``horizontal_deg`` / ``vertical_deg`` keys; a
    two-item sequence is the compact configuration form. Positive values move
    the photographed panorama content image-right and image-up, respectively.
    """
    if value is None:
        return 0.0, 0.0
    if isinstance(value, dict):
        return (
            float(value.get("horizontal_deg", value.get("x", 0.0)) or 0.0),
            float(value.get("vertical_deg", value.get("y", 0.0)) or 0.0),
        )
    values = np.asarray(value, dtype=np.float64).reshape(-1)
    if values.size != 2:
        raise ValueError(
            "nyx env_offset must contain [horizontal_deg, vertical_deg], "
            f"got {value!r}"
        )
    return float(values[0]), float(values[1])


def offset_env_texture(texture_path: str, offset) -> str:
    """Return a cached EXR whose panorama is shifted by ``offset`` degrees.

    NYX environment maps are infinite-distance lighting and have no meaningful
    XYZ translation. This equirectangular image offset provides the useful
    visual operation: move photographed furniture/horizons away from simulated
    task geometry without modifying the source asset.
    """
    horizontal_deg, vertical_deg = normalize_env_offset(offset)
    if abs(horizontal_deg) < 1e-9 and abs(vertical_deg) < 1e-9:
        return texture_path

    root_path = Path(__file__).resolve().parent.parent
    source = Path(texture_path).resolve()
    stat = source.stat()
    cache_key = hashlib.sha256(
        (
            f"{source}|{stat.st_size}|{stat.st_mtime_ns}|"
            f"{horizontal_deg:.9f}|{vertical_deg:.9f}|opencv-half-v1"
        ).encode("utf-8")
    ).hexdigest()[:16]
    cache_dir = root_path / "__nyx_cache__" / "env_offsets"
    cache_dir.mkdir(parents=True, exist_ok=True)
    output = cache_dir / f"{source.stem}__offset_{cache_key}.exr"
    if output.is_file():
        return str(output)

    # OpenCV keeps the pixels as floating-point HDR data. Its EXR codec is
    # runtime-gated, so set the opt-in before importing cv2. The legacy OpenEXR
    # channel writer is intentionally avoided: the version in MAWM_nyx_clean
    # segfaults while writing shifted buffers.
    os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
    import cv2

    pixels = cv2.imread(str(source), cv2.IMREAD_UNCHANGED)
    if pixels is None:
        raise RuntimeError(f"Failed to read EXR environment map: {source}")
    height, width = pixels.shape[:2]
    x_pixels = int(round(horizontal_deg / 360.0 * width))
    y_pixels = -int(round(vertical_deg / 180.0 * height))
    shifted = np.roll(pixels, shift=(y_pixels, x_pixels), axis=(0, 1))

    temporary = output.with_name(f".{output.stem}.{os.getpid()}.tmp.exr")
    try:
        write_options = []
        if hasattr(cv2, "IMWRITE_EXR_TYPE") and hasattr(cv2, "IMWRITE_EXR_TYPE_HALF"):
            write_options.extend([cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_HALF])
        if hasattr(cv2, "IMWRITE_EXR_COMPRESSION") and hasattr(
            cv2, "IMWRITE_EXR_COMPRESSION_PIZ"
        ):
            write_options.extend(
                [cv2.IMWRITE_EXR_COMPRESSION, cv2.IMWRITE_EXR_COMPRESSION_PIZ]
            )
        if not cv2.imwrite(str(temporary), shifted, write_options):
            raise RuntimeError(f"Failed to write offset EXR cache: {temporary}")
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    print(
        f"[nyx] env_offset=[{horizontal_deg:g}, {vertical_deg:g}] deg "
        f"cache={output.name}"
    )
    return str(output)
