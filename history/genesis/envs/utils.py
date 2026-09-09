"""Core utilities: Pose, Actor wrappers, and object loaders."""

import os
import json
import numpy as np
import transforms3d as t3d
import genesis as gs
from pathlib import Path
from typing import Literal, Tuple

from .genesis_compat import (
    default_file_meshes_are_zup as _compat_default_file_meshes_are_zup,
    mesh_frame_kwargs,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_PATH = Path(__file__).resolve().parent.parent
ASSETS_PATH = ROOT_PATH / "assets"

# Table-top height in meters. Re-exported here so tasks can pull
# everything from envs.utils. The canonical definition lives in
# envs.base_task; if you change one, change both. Added 2026-05-02
# to unblock dump_bin_bigbin (and any future task) that imports
# TABLE_HEIGHT from utils — concurrent agents found this was missing.
TABLE_HEIGHT = 0.74

DEFAULT_CONFIG_PATH = ROOT_PATH / "config" / "default.yml"


def load_config(config_path=None) -> dict:
    """Load config/default.yml as the base, overlaid with ``config_path``.

    Entry points used to start from an empty dict and only read a file when
    --config was passed, so every default in config/default.yml (renderer: nyx,
    the nyx spp/denoise/env_texture block, camera resolutions, ...) was silently
    ignored unless the caller happened to pass that exact file. Loading the
    defaults here makes `collect`, `collect_vla` and `eval` agree on one source
    of truth; an explicit --config still wins key-by-key (top-level merge, with
    a one-level-deep merge for dict values like `nyx:` so a partial override
    doesn't drop the rest of the block).
    """
    import yaml

    config: dict = {}
    if DEFAULT_CONFIG_PATH.exists():
        with open(DEFAULT_CONFIG_PATH) as f:
            config = yaml.safe_load(f) or {}

    if not config_path:
        return config

    with open(config_path) as f:
        override = yaml.safe_load(f) or {}

    for key, value in override.items():
        if isinstance(value, dict) and isinstance(config.get(key), dict):
            merged = dict(config[key])
            merged.update(value)
            config[key] = merged
        else:
            config[key] = value
    return config


# ---------------------------------------------------------------------------
# Pose
# ---------------------------------------------------------------------------

class Pose:
    """Position (3,) + quaternion (w, x, y, z)."""

    def __init__(self, p=None, q=None):
        self.p = np.asarray(p if p is not None else [0, 0, 0], dtype=np.float64)
        self.q = np.asarray(q if q is not None else [1, 0, 0, 0], dtype=np.float64)
        self.q /= np.linalg.norm(self.q)

    @classmethod
    def from_pose7(cls, pose7):
        a = np.asarray(pose7)
        return cls(a[:3], a[3:7])

    @classmethod
    def from_matrix(cls, T):
        T = np.asarray(T)
        return cls(T[:3, 3], t3d.quaternions.mat2quat(T[:3, :3]))

    def to_pose7(self):
        return np.concatenate([self.p, self.q])

    def to_matrix(self):
        return t3d.affines.compose(self.p, t3d.quaternions.quat2mat(self.q), np.ones(3))

    def __mul__(self, other):
        return Pose.from_matrix(self.to_matrix() @ other.to_matrix())

    def inv(self):
        return Pose.from_matrix(np.linalg.inv(self.to_matrix()))

    def copy(self):
        return Pose(self.p.copy(), self.q.copy())

    def __repr__(self):
        return f"Pose(p={self.p.tolist()}, q={self.q.tolist()})"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def to_numpy(arr):
    """Convert torch.Tensor or similar to numpy array."""
    if hasattr(arr, "cpu"):
        arr = arr.cpu().numpy()
    return np.asarray(arr)


def normalize_pos_quat(pos, quat):
    """Flatten and normalize position/quaternion from Genesis API output.

    Genesis returns quaternions in [w, x, y, z] format (same as our Pose convention).
    """
    pos = to_numpy(pos).ravel()[:3]
    quat = to_numpy(quat).ravel()[:4]
    return pos, quat


def safe_mat2quat(rot3x3):
    """Rotation matrix to quaternion with SVD orthogonalization fallback."""
    rot = np.asarray(rot3x3, dtype=np.float64)
    if not np.all(np.isfinite(rot)) or np.allclose(rot, 0):
        return np.array([1.0, 0.0, 0.0, 0.0])
    try:
        U, _, Vt = np.linalg.svd(rot)
        rot_ortho = U @ Vt
        if np.linalg.det(rot_ortho) < 0:
            U[:, -1] *= -1
            rot_ortho = U @ Vt
        return t3d.quaternions.mat2quat(rot_ortho)
    except Exception:
        return np.array([1.0, 0.0, 0.0, 0.0])


# ---------------------------------------------------------------------------
# Actor wrappers
# ---------------------------------------------------------------------------

class Actor:
    """Wrapper around a Genesis rigid entity with metadata (contact points, etc.)."""

    POINTS = {
        "contact": "contact_points_pose",
        "target": "target_pose",
        "functional": "functional_matrix",
        "orientation": "orientation_point",
    }

    def __init__(self, entity, config: dict = None, name: str = None):
        self.entity = entity
        self.config = config or {}
        self.name = name

    def get_pose(self) -> Pose:
        pos, quat = normalize_pos_quat(self.entity.get_pos(), self.entity.get_quat())
        return Pose(pos, quat)

    def get_pos_quat(self) -> Tuple[np.ndarray, np.ndarray]:
        return normalize_pos_quat(self.entity.get_pos(), self.entity.get_quat())

    def _get_pose_matrix(self):
        return self.get_pose().to_matrix()

    def _process_local_matrix(self, local_matrix):
        local_matrix = np.array(local_matrix)
        if local_matrix.ndim == 1 and len(local_matrix) == 16:
            local_matrix = local_matrix.reshape(4, 4)
        scale = self.config.get("scale", [1, 1, 1])
        local_matrix[:3, 3] *= np.array(scale)
        return local_matrix

    def _to_ret(self, world_matrix, ret):
        if ret == "matrix":
            return world_matrix
        elif ret == "list":
            return world_matrix[:3, 3].tolist() + safe_mat2quat(world_matrix[:3, :3]).tolist()
        else:
            return Pose(world_matrix[:3, 3], safe_mat2quat(world_matrix[:3, :3]))

    def get_point(self, type: str, idx: int, ret: str = "list"):
        key = self.POINTS[type]
        local = self._process_local_matrix(self.config[key][idx])
        world = self._get_pose_matrix() @ local
        return self._to_ret(world, ret)

    def get_contact_point(self, idx: int, ret: str = "list"):
        return self.get_point("contact", idx, ret)

    def get_functional_point(self, idx: int, ret: str = "list"):
        return self.get_point("functional", idx, ret)

    def get_target_point(self, idx: int, ret: str = "list"):
        return self.get_point("target", idx, ret)


class ArticulationActor(Actor):
    """Wrapper around an articulated Genesis entity (URDF-loaded)."""

    POINTS = {
        "contact": "contact_points",
        "target": "target_points",
        "functional": "functional_points",
        "orientation": "orientation_point",
    }

    def __init__(self, entity, config: dict = None, name: str = None):
        super().__init__(entity, config, name)
        self.link_dict = {link.name: link for link in entity.links}

    def _get_link_pose_matrix(self, link):
        pos, quat = normalize_pos_quat(link.get_pos(), link.get_quat())
        return Pose(pos, quat).to_matrix()

    def get_point(self, type: str, idx: int, ret: str = "list"):
        key = self.POINTS[type]
        entry = self.config[key][idx]
        local = np.array(entry["matrix"])
        local[:3, 3] *= self.config.get("scale", 1.0)
        link = self.link_dict[entry["base"]]
        world = self._get_link_pose_matrix(link) @ local
        return self._to_ret(world, ret)

    def set_qpos(self, qpos):
        self.entity.set_qpos(qpos)

    def get_qpos(self):
        return to_numpy(self.entity.get_qpos())

    def get_qlimits(self):
        lower, upper = self.entity.get_dofs_limit()
        return np.stack([lower, upper], axis=-1)


# ---------------------------------------------------------------------------
# Object loaders
# ---------------------------------------------------------------------------

def _resolve_model_dir(name: str) -> Path:
    """Resolve model directory under assets/objects/."""
    p = ASSETS_PATH / "objects" / name
    if p.exists():
        return p
    raise FileNotFoundError(f"Object '{name}' not found at {p}")


def _scale_to_tuple3(scale) -> Tuple[float, float, float]:
    a = np.asarray(scale, dtype=np.float64).ravel()
    if a.size == 0:
        return (1.0, 1.0, 1.0)
    if a.size == 1:
        return (float(a[0]),) * 3
    return (float(a[0]), float(a[1]), float(a[2]) if a.size > 2 else 1.0)


def default_file_meshes_are_zup(path: str | Path) -> bool:
    """Return the Genesis up-axis flag for benchmark mesh files."""

    return _compat_default_file_meshes_are_zup(path)


def _find_mesh_file(directory: Path, model_id: int = 0) -> Path | None:
    """Find GLB or OBJ file in directory."""
    for ext in (".glb", ".obj"):
        f = directory / f"base{model_id}{ext}"
        if f.exists():
            return f
        f = directory / f"base{ext}"
        if f.exists():
            return f
    for ext in (".glb", ".obj"):
        files = sorted(directory.glob(f"*{ext}"))
        if files:
            return files[0]
    return None


def load_mesh(
    scene: gs.Scene,
    path: str | Path,
    pose: Pose,
    scale: float | tuple = 1.0,
    is_static: bool = False,
    convex: bool = False,
    collision: bool = True,
    visual: bool = True,
    friction: float | None = None,
    density: float | None = None,
    decompose_object_error_threshold: float | None = None,
    surface=None,
    group_by_material: bool = True,
    align: bool | None = False,
    file_meshes_are_zup: bool | None = None,
) -> "gs.Entity":
    """Load a GLB/OBJ mesh as a rigid body.

    decompose_object_error_threshold: pass a small value (e.g. 0.0) to
    force coacd convex decomposition.  Default (None) uses Genesis's
    default (0.15 = 15%): meshes within 15% volume error of their convex
    hull are kept as a single hull.  For meshes with concave features
    that must be preserved (e.g. mug handle ring), use 0.0.

    align: Genesis 1.x can align rigid Mesh assets to a principal-inertia
    frame. Keep it disabled by default so authored object origins and grasp
    annotations stay in the benchmark frame. Pass None to use Genesis's
    default.

    file_meshes_are_zup: latest Genesis exposes this flag for mesh frame
    handling. It defaults to True for every mesh type so Genesis preserves
    benchmark-authored frames; task poses supply any required upright
    rotation. Pass False only for an asset that explicitly needs Genesis's
    Y-up-to-Z-up conversion.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Mesh file not found: {path}")
    if file_meshes_are_zup is None:
        file_meshes_are_zup = default_file_meshes_are_zup(path)
    mat_kwargs = {}
    if friction is not None:
        mat_kwargs["friction"] = friction
    if density is not None:
        mat_kwargs["rho"] = density
    morph_kwargs = {
        "file": str(path.absolute()),
        "scale": _scale_to_tuple3(scale),
        "pos": tuple(pose.p),
        "quat": tuple(pose.q),
        "convexify": bool(convex),
        "fixed": bool(is_static),
        "collision": bool(collision),
        "visualization": bool(visual),
        "group_by_material": bool(group_by_material),
    }
    if decompose_object_error_threshold is not None:
        morph_kwargs["decompose_object_error_threshold"] = float(
            decompose_object_error_threshold
        )
    morph_kwargs.update(
        mesh_frame_kwargs(path, align=align, file_meshes_are_zup=file_meshes_are_zup)
    )
    entity = scene.add_entity(
        gs.morphs.Mesh(**morph_kwargs),
        material=gs.materials.Rigid(**mat_kwargs),
        surface=surface,
    )
    return entity


def load_urdf(
    scene: gs.Scene,
    path: str | Path,
    pose: Pose,
    scale: float = 1.0,
    fix_root: bool = True,
    decompose_object_error_threshold: float | None = None,
    merge_fixed_links: bool | None = None,
) -> "gs.Entity":
    """Load a URDF as an articulated entity.

    decompose_object_error_threshold: pass a small value (e.g. 0.01) to force
    CoACD convex decomposition of non-convex links — e.g. hollow boxes where
    the single-hull collision would fill the interior.  Default (None) keeps
    Genesis's default (0.15), which uses a single convex hull for most meshes.

    merge_fixed_links: None keeps Genesis's default (True). Pass False when
    the renderer is NYX — its URDF sub-scene export needs the file's link
    tree to map 1:1 onto the simulated links.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"URDF not found: {path}")
    urdf_kwargs = dict(
        file=str(path.absolute()),
        pos=tuple(pose.p),
        quat=tuple(pose.q),
        scale=float(scale),
        fixed=fix_root,
    )
    if merge_fixed_links is not None:
        urdf_kwargs["merge_fixed_links"] = bool(merge_fixed_links)
    if decompose_object_error_threshold is not None:
        urdf_kwargs["convexify"] = True
        urdf_kwargs["decompose_object_error_threshold"] = float(
            decompose_object_error_threshold
        )
    entity = scene.add_entity(gs.morphs.URDF(**urdf_kwargs))
    return entity


def create_primitive(
    scene: gs.Scene,
    shape: Literal["box", "sphere", "cylinder"],
    pose: Pose,
    size: dict = None,
    color: tuple = (1, 1, 1),
    is_static: bool = False,
    collision: bool = True,
    visual: bool = True,
    surface=None,
) -> "gs.Entity":
    """Create a primitive shape (box, sphere, cylinder).

    size keys by shape:
        box:      {"half_size": (x, y, z)}
        sphere:   {"radius": float}
        cylinder: {"radius": float, "half_length": float}
    """
    size = size or {}
    surface = surface or gs.surfaces.Default(color=color)
    pos = tuple(float(x) for x in pose.p)
    quat = tuple(float(x) for x in pose.q)

    if shape == "box":
        half = size.get("half_size", (0.1, 0.1, 0.1))
        morph = gs.morphs.Box(
            size=tuple(2 * x for x in half),
            pos=pos, quat=quat,
            visualization=visual, collision=collision, fixed=is_static,
        )
    elif shape == "sphere":
        morph = gs.morphs.Sphere(
            radius=size.get("radius", 0.05),
            pos=pos, quat=quat,
            visualization=visual, collision=collision, fixed=is_static,
        )
    elif shape == "cylinder":
        morph = gs.morphs.Cylinder(
            radius=size.get("radius", 0.05),
            height=2 * size.get("half_length", 0.1),
            pos=pos, quat=quat,
            visualization=visual, collision=collision, fixed=is_static,
        )
    else:
        raise ValueError(f"Unknown shape: {shape}")

    return scene.add_entity(morph, surface=surface)


def load_object(
    scene: gs.Scene,
    pose: Pose,
    name: str,
    model_id: int = 0,
    convex: bool = False,
    is_static: bool = False,
    friction: float | None = None,
    density: float | None = None,
    decompose_object_error_threshold: float | None = None,
) -> Actor | ArticulationActor:
    """Unified object loader: auto-detects GLB mesh vs URDF articulation.

    decompose_object_error_threshold: forwarded to load_urdf for articulated
    objects where single-hull collision is inadequate (e.g. hollow boxes).
    """
    modeldir = _resolve_model_dir(name)

    # Check for URDF first (articulated objects)
    urdf = modeldir / "mobility.urdf"
    if urdf.exists() and urdf.stat().st_size > 0:
        json_path = modeldir / f"model_data{model_id}.json"
        config = {}
        urdf_scale = 1.0
        if json_path.exists():
            with open(json_path) as f:
                config = json.load(f)
            raw_scale = config.get("scale", 1.0)
            if isinstance(raw_scale, (list, tuple)):
                urdf_scale = float(raw_scale[0])
            else:
                urdf_scale = float(raw_scale)
        entity = load_urdf(
            scene, urdf, pose, scale=urdf_scale, fix_root=is_static,
            decompose_object_error_threshold=decompose_object_error_threshold,
        )
        return ArticulationActor(entity, config, name)

    # Otherwise look for mesh files
    for subdir_name in ("visual", "collision", "."):
        subdir = modeldir if subdir_name == "." else modeldir / subdir_name
        if not subdir.exists():
            continue
        mesh_file = _find_mesh_file(subdir, model_id)
        if mesh_file:
            json_path = modeldir / f"model_data{model_id}.json"
            config = {}
            scale = (1.0, 1.0, 1.0)
            if json_path.exists():
                with open(json_path) as f:
                    config = json.load(f)
                    scale = _scale_to_tuple3(config.get("scale", [1, 1, 1]))
            entity = load_mesh(scene, mesh_file, pose, scale=scale,
                               is_static=is_static, convex=convex,
                               friction=friction, density=density)
            return Actor(entity, config, name)

    raise FileNotFoundError(f"No mesh or URDF found in {modeldir}")


# ---------------------------------------------------------------------------
# Decorative object placement
# ---------------------------------------------------------------------------

# Small/medium table-appropriate objects (exclude large furniture, racks, etc.)
_TABLE_OBJECTS = [
    "001_bottle", "003_plate", "005_french-fries", "006_hamburg",
    "017_calculator", "021_cup", "023_tissue-box",
    "025_chips-tub", "029_olive-oil", "031_jam-jar",
    "035_apple", "038_milk-box", "043_book",
    "046_alarm-clock", "047_mouse", "048_stapler",
    "057_toycar", "059_pencup",
    "065_soy-sauce", "066_vinegar", "068_boxdrink",
    "071_can", "073_rubikscube", "079_remotecontrol",
    "081_playingcards", "086_woodenblock",
    "092_notebook", "095_glue", "100_seal",
    "101_milk-tea", "105_sauce-can", "107_soap", "108_block",
    "112_tea-box", "113_coffee-box", "115_perfume",
    "117_whiteboard-eraser", "118_tooth-paste",
]


def table_object_pool() -> list[str]:
    """Return the default universal tabletop clutter asset ids."""
    return list(_TABLE_OBJECTS)


def place_decorative_objects(
    scene: gs.Scene,
    region: tuple,
    height: float,
    exclude_regions: list = None,
    num_objects: int = 5,
    target_size: float = 0.10,
    seed: int = None,
    object_pool: list = None,
    collision: bool = True,
    is_static: bool = False,
    friction: float | None = None,
    density: float | None = None,
    min_spacing: float = 0.01,
    max_attempts_per_object: int = 40,
) -> list:
    """Place random physical tabletop objects on a surface.

    Args:
        scene: Genesis scene to add objects to.
        region: (xmin, xmax, ymin, ymax) placement bounds.
        height: Surface z height (table top).
        exclude_regions: Zones to avoid. Supported forms:
            - (cx, cy, radius)
            - (xmin, xmax, ymin, ymax)
            - ("circle", cx, cy, radius)
            - ("rect", xmin, xmax, ymin, ymax)
            - {"type": "circle", "center": (cx, cy), "radius": r}
            - {"type": "rect", "bounds": (xmin, xmax, ymin, ymax)}
        num_objects: Number of objects to place.
        target_size: Desired max dimension in meters (used to auto-scale).
        seed: Random seed for reproducibility.
        object_pool: List of object names to sample from (default: _TABLE_OBJECTS).
        collision: Whether objects participate in collision.
        is_static: Whether objects are fixed. Default false so they simulate.
        friction: Optional rigid-body friction override.
        density: Optional rigid-body density override.
        min_spacing: Extra XY spacing between random objects.
        max_attempts_per_object: Rejection-sampling attempts per object.

    Returns:
        List of placed entities.
    """
    rng = np.random.RandomState(seed)
    exclude_regions = exclude_regions or []
    pool = list(object_pool or _TABLE_OBJECTS)
    rng.shuffle(pool)

    xmin, xmax, ymin, ymax = [float(v) for v in region]
    upright_q = np.array([0.70710678, 0.70710678, 0.0, 0.0], dtype=np.float64)

    placed = []  # (cx, cy, radius) of placed objects
    entities = []

    def _blocked_by_exclusion(x: float, y: float, radius: float, exclusion) -> bool:
        if isinstance(exclusion, dict):
            typ = str(exclusion.get("type", "circle")).lower()
            if typ == "rect":
                bounds = exclusion.get("bounds", exclusion.get("region"))
                if bounds is None:
                    return False
                ex0, ex1, ey0, ey1 = [float(v) for v in bounds]
                return ex0 - radius <= x <= ex1 + radius and ey0 - radius <= y <= ey1 + radius
            center = exclusion.get("center", exclusion.get("xy"))
            if center is None:
                return False
            cx, cy = [float(v) for v in center[:2]]
            r = float(exclusion.get("radius", 0.0))
            return np.hypot(x - cx, y - cy) < r + radius

        values = list(exclusion)
        if not values:
            return False
        if isinstance(values[0], str):
            typ = values[0].lower()
            nums = [float(v) for v in values[1:]]
            if typ == "rect" and len(nums) >= 4:
                ex0, ex1, ey0, ey1 = nums[:4]
                return ex0 - radius <= x <= ex1 + radius and ey0 - radius <= y <= ey1 + radius
            if typ == "circle" and len(nums) >= 3:
                cx, cy, r = nums[:3]
                return np.hypot(x - cx, y - cy) < r + radius
            return False

        nums = [float(v) for v in values]
        if len(nums) >= 4:
            ex0, ex1, ey0, ey1 = nums[:4]
            return ex0 - radius <= x <= ex1 + radius and ey0 - radius <= y <= ey1 + radius
        if len(nums) >= 3:
            cx, cy, r = nums[:3]
            return np.hypot(x - cx, y - cy) < r + radius
        return False

    for obj_name in pool:
        if len(entities) >= num_objects:
            break

        obj_dir = ASSETS_PATH / "objects" / obj_name
        if not obj_dir.exists():
            continue

        # Load mesh extent info
        md_path = obj_dir / "model_data0.json"
        if not md_path.exists():
            continue
        with open(md_path) as f:
            md = json.load(f)
        extents = md.get("extents", [1, 1, 1])
        max_ext = max(extents)
        if max_ext <= 0:
            continue

        # Compute scale to reach target_size
        scale = target_size / max_ext
        obj_radius = max(extents[0], extents[2]) * scale / 2  # XZ footprint
        if xmin + obj_radius >= xmax - obj_radius or ymin + obj_radius >= ymax - obj_radius:
            continue

        # Find visual mesh
        visual_dir = obj_dir / "visual"
        if not visual_dir.exists():
            continue
        mesh_file = _find_mesh_file(visual_dir, model_id=0)
        if mesh_file is None:
            continue

        # Try random placements
        for _ in range(int(max_attempts_per_object)):
            x = rng.uniform(xmin + obj_radius, xmax - obj_radius)
            y = rng.uniform(ymin + obj_radius, ymax - obj_radius)

            # Check exclusion zones
            blocked = False
            for exclusion in exclude_regions:
                if _blocked_by_exclusion(x, y, obj_radius, exclusion):
                    blocked = True
                    break
            if blocked:
                continue

            # Check against already-placed objects
            for px, py, pr in placed:
                if np.hypot(x - px, y - py) < pr + obj_radius + float(min_spacing):
                    blocked = True
                    break
            if blocked:
                continue

            # Place it
            obj_height = extents[1] * scale / 2  # half-height offset
            pose = Pose([x, y, height + obj_height], upright_q)
            entity = load_mesh(
                scene, mesh_file, pose,
                scale=(scale, scale, scale),
                convex=True,
                collision=bool(collision),
                is_static=bool(is_static),
                friction=friction,
                density=density,
            )
            placed.append((x, y, obj_radius))
            entities.append(entity)
            break

    return entities
