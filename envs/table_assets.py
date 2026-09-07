"""Static table asset variants for debug-only table randomization."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import genesis as gs

from .utils import ASSETS_PATH, Pose, create_primitive, load_mesh


TABLE_VARIANT_DEBUG_ENV = "DEBUG_TABLE_RANDOMIZATION"
TABLE_VARIANT_FORCE_ENV = "TABLE_VARIANT_NAME"

DEFAULT_TABLE_EXTENTS = np.array([1.2, 0.7, 0.765], dtype=np.float64)
SAPIEN_TABLE_VARIANTS = (
    "sapien-table-20279",
    "sapien-table-20411",
    "sapien-table-21467",
    "sapien-table-22301",
    "sapien-table-22339",
    "sapien-table-24644",
    "sapien-table-24931",
    "sapien-table-25913",
    "sapien-table-26503",
    "sapien-table-34610",
)
DEFAULT_EXCLUDED_TABLE_VARIANTS = (
    "sapien-table-20411",
    "sapien-table-22339",
    "sapien-table-24644",
    "sapien-table-25913",
)


@dataclass(frozen=True)
class TableVariant:
    name: str
    mesh_path: Path


def table_randomization_enabled(config: dict | None) -> bool:
    """Return True only when both task config and debug flag opt in."""
    config = config or {}
    wants_table = bool(config.get("randomize_table", False))
    debug_on = (
        bool(config.get("debug_table_randomization", False))
        or os.environ.get(TABLE_VARIANT_DEBUG_ENV, "").strip() == "1"
    )
    return wants_table and debug_on


def available_table_variants(names: Iterable[str] = SAPIEN_TABLE_VARIANTS) -> list[TableVariant]:
    variants: list[TableVariant] = []
    for name in names:
        mesh = ASSETS_PATH / "objects" / name / "visual" / "base0.glb"
        if mesh.exists():
            variants.append(TableVariant(name=name, mesh_path=mesh))
    return variants


def _table_variant_pool(config: dict) -> list[str]:
    names = [str(name) for name in config.get("table_variant_pool", SAPIEN_TABLE_VARIANTS)]
    excluded = set(DEFAULT_EXCLUDED_TABLE_VARIANTS)
    excluded.update(str(name) for name in config.get("exclude_table_variants", ()))
    excluded.update(str(name) for name in config.get("table_variant_exclude", ()))
    return [name for name in names if name not in excluded]


def choose_table_variant(config: dict | None) -> TableVariant:
    """Choose a table variant by forced name or deterministic seeded RNG."""
    config = config or {}
    forced = config.get("table_variant_name") or os.environ.get(TABLE_VARIANT_FORCE_ENV, "").strip()
    if forced:
        variants = available_table_variants([forced])
        if not variants:
            raise ValueError(f"Unknown table variant {forced!r}")
        return variants[0]

    pool = _table_variant_pool(config)
    variants = available_table_variants(pool)
    if not variants:
        raise FileNotFoundError(
            "No SAPIEN table variants found after applying table variant exclusions"
        )
    return random.choice(variants)


def _stable_table_surface(config: dict | None):
    config = config or {}
    mode = str(config.get("table_variant_surface", "asset")).strip().lower()
    if mode not in {"stable", "matte", "override"}:
        return None
    color = tuple(float(v) for v in config.get("table_variant_color", (0.74, 0.68, 0.58)))
    roughness = float(config.get("table_variant_roughness", 0.9))
    return gs.surfaces.Default(
        color=color,
        metallic=0.0,
        roughness=roughness,
        smooth=True,
    )


def _nyx_single_mesh(mesh_path: Path) -> Path:
    """Return one geometry-only OBJ for reliable NYX table rendering.

    Processed table GLBs can contain 10–100 geometry/material groups. A few
    otherwise-valid assets exhaust gs-nyx's fixed spill pool while exporting
    those groups. Flattening the already-normalized scene to one mesh keeps
    every triangle and authored node transform while avoiding that renderer
    limit. The cached OBJ is generated atomically and is safe across array jobs.
    """
    dst = mesh_path.with_name(f"{mesh_path.stem}_nyx_single.obj")
    if dst.exists() and dst.stat().st_size > 0:
        return dst

    import trimesh

    scene = trimesh.load(str(mesh_path), force="scene", process=False)
    merged = scene.dump(concatenate=True)
    geometry = trimesh.Trimesh(
        vertices=np.asarray(merged.vertices),
        faces=np.asarray(merged.faces),
        process=False,
    )
    tmp = dst.with_name(f"{dst.stem}.{os.getpid()}.tmp.obj")
    geometry.export(str(tmp))
    os.replace(tmp, dst)
    return dst


def add_table_variant(
    task,
    table_height: float,
    half_size: tuple[float, float] = (0.6, 0.35),
    center_xy: tuple[float, float] | None = None,
):
    """Add one static SAPIEN table mesh, matching the requested table size.

    The processed assets are already baked to BaseTask's default
    ``1.2 x 0.7 x 0.765 m`` table.  This helper applies a final scale only
    when a task uses a non-default footprint or height.
    """
    variant = choose_table_variant(task.config)
    table_thickness = float(getattr(task, "TABLE_THICKNESS", 0.05))
    table_top_z = float(table_height) + table_thickness / 2.0
    target_extents = np.array(
        [2.0 * float(half_size[0]), 2.0 * float(half_size[1]), table_top_z],
        dtype=np.float64,
    )
    scale = tuple((target_extents / DEFAULT_TABLE_EXTENTS).tolist())
    if center_xy is None:
        dx, dy = np.asarray(getattr(task, "table_offset", (0.0, 0.0)), dtype=float)
    else:
        dx, dy = np.asarray(center_xy, dtype=float)
    is_nyx = str(task.config.get("renderer", "")).lower() == "nyx"
    surface = _stable_table_surface(task.config)
    visual_path = _nyx_single_mesh(variant.mesh_path) if is_nyx else variant.mesh_path
    if is_nyx and surface is None:
        # Flattening intentionally strips the many source materials. Give all
        # variants the same stable finish so appearance does not depend on a
        # renderer fallback material.
        surface = gs.surfaces.Default(
            color=tuple(float(v) for v in task.config.get(
                "table_variant_color", (0.58, 0.40, 0.24),
            )),
            metallic=0.0,
            roughness=float(task.config.get("table_variant_roughness", 0.9)),
            smooth=True,
        )
    visual_entity = load_mesh(
        task.scene,
        visual_path,
        Pose(p=[dx, dy, 0.0]),
        scale=scale,
        is_static=True,
        convex=False,
        collision=False,
        visual=True,
        group_by_material=False if is_nyx else True,
        surface=surface,
    )

    table_collision = create_primitive(
        task.scene,
        "box",
        Pose(p=[dx, dy, float(table_height)]),
        size={"half_size": (float(half_size[0]), float(half_size[1]), table_thickness / 2.0)},
        is_static=True,
        collision=True,
        visual=False,
    )
    table_surface_entity = None
    if surface is not None:
        overlay_half_thickness = float(task.config.get("table_variant_overlay_half_thickness", 0.003))
        overlay_z = table_top_z + overlay_half_thickness + 0.001
        table_surface_entity = create_primitive(
            task.scene,
            "box",
            Pose(p=[dx, dy, overlay_z]),
            size={
                "half_size": (
                    float(half_size[0]),
                    float(half_size[1]),
                    overlay_half_thickness,
                )
            },
            is_static=True,
            collision=False,
            visual=True,
            surface=surface,
        )
    leg_height = max(float(table_height) - table_thickness / 2.0, 0.0)
    if leg_height > 0:
        inset_x = min(0.05, float(half_size[0]) * 0.25)
        inset_y = min(0.05, float(half_size[1]) * 0.25)
        leg_x = max(float(half_size[0]) - inset_x, 0.0)
        leg_y = max(float(half_size[1]) - inset_y, 0.0)
        for x_sign in (-1.0, 1.0):
            for y_sign in (-1.0, 1.0):
                create_primitive(
                    task.scene,
                    "cylinder",
                    Pose(p=[dx + x_sign * leg_x, dy + y_sign * leg_y, leg_height / 2.0]),
                    size={"radius": 0.025, "half_length": leg_height / 2.0},
                    is_static=True,
                    collision=True,
                    visual=False,
                )

    task.table = table_collision
    task.table_variant_entity = visual_entity
    task.table_variant_surface_entity = table_surface_entity
    task.table_variant_name = variant.name
    task.TABLE_TOP_Z = table_top_z
    task._table_top_z_fixed = True
    surface_name = "nyx-stable" if is_nyx else ("stable" if surface is not None else "asset")
    print(
        f"[table] debug variant={variant.name} surface={surface_name} "
        f"scale={tuple(round(x, 4) for x in scale)}"
    )
    return visual_entity
