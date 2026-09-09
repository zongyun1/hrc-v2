"""Compatibility helpers for Genesis version differences.

The benchmark still supports the local ``vico`` Genesis fork, but migration
work targets upstream ``genesis-world>=1``. Keep small API/version shims here
instead of scattering version checks through task and robot code.
"""

from __future__ import annotations

import importlib.metadata as md
import inspect
import os
from pathlib import Path
from functools import lru_cache

import numpy as np


def configure_genesis_runtime() -> None:
    """Set safe process defaults before ``import genesis``.

    Genesis reads some rendering/cache settings during import. Call this from
    package initializers and scripts before importing Genesis directly.
    Existing user-provided environment values always win.
    """

    tmp = Path(os.environ.get("TMPDIR", "/tmp"))
    os.environ.setdefault("NUMBA_CACHE_DIR", str(tmp / "genesis_hr_bench_numba_cache"))
    os.environ.setdefault("MPLCONFIGDIR", str(tmp / "genesis_hr_bench_mpl_cache"))
    os.environ.setdefault("XDG_CACHE_HOME", str(tmp / "genesis_hr_bench_xdg_cache"))
    os.environ.setdefault("MESA_GL_VERSION_OVERRIDE", "3.3")

    if "PYOPENGL_PLATFORM" not in os.environ:
        explicit_platform = os.environ.get("GENESIS_OPENGL_PLATFORM", "").strip()
        if explicit_platform:
            os.environ["PYOPENGL_PLATFORM"] = explicit_platform
        elif os.environ.get("GENESIS_FORCE_EGL", "").strip() == "1":
            os.environ["PYOPENGL_PLATFORM"] = "egl"
    if os.environ.get("GENESIS_SOFTWARE_RENDER", "").strip() == "1":
        os.environ.setdefault("PYGLET_HEADLESS", "1")


configure_genesis_runtime()

_GS = None


def get_genesis():
    """Import Genesis after runtime defaults are configured."""
    global _GS
    if _GS is None:
        configure_genesis_runtime()
        import genesis as gs  # noqa: PLC0415

        _GS = gs
    return _GS


def has_avatar_material() -> bool:
    gs = get_genesis()
    return hasattr(gs.materials, "Avatar")


def avatar_or_kinematic_material():
    gs = get_genesis()
    return gs.materials.Avatar() if has_avatar_material() else gs.materials.Kinematic()


def default_file_meshes_are_zup(path: str | Path) -> bool:
    """Keep every benchmark mesh in its authored frame at load time.

    Task poses already encode any asset-specific upright rotation.  Passing
    ``False`` for GLB/GLTF makes Genesis apply another Y-up-to-Z-up rotation,
    which double-rotates assets such as the kitchen props onto their sides.
    """

    return True


def supported_kwargs(callable_obj, kwargs: dict) -> dict:
    """Filter keyword arguments to those accepted by a Genesis API object."""

    fields = getattr(callable_obj, "model_fields", None)
    if fields:
        return {k: v for k, v in kwargs.items() if k in fields}
    try:
        params = inspect.signature(callable_obj).parameters
    except (TypeError, ValueError):
        return dict(kwargs)
    return {k: v for k, v in kwargs.items() if k in params}


def mesh_frame_kwargs(
    mesh_path: str | Path,
    *,
    align: bool | None = False,
    file_meshes_are_zup: bool | None = None,
) -> dict:
    """Genesis Mesh kwargs that preserve benchmark-authored asset frames."""

    gs = get_genesis()
    if file_meshes_are_zup is None:
        file_meshes_are_zup = default_file_meshes_are_zup(mesh_path)
    kwargs = {}
    if align is not None:
        kwargs["align"] = bool(align)
    if file_meshes_are_zup is not None:
        kwargs["file_meshes_are_zup"] = bool(file_meshes_are_zup)
    return supported_kwargs(
        gs.morphs.Mesh,
        kwargs,
    )


def to_numpy(value):
    if hasattr(value, "cpu"):
        value = value.cpu().numpy()
    return np.asarray(value)


def genesis_backend_from_env(gs=None):
    """Return the Genesis backend object selected by ``GENESIS_BACKEND``."""

    if gs is None:
        gs = get_genesis()
    backend_name = os.environ.get("GENESIS_BACKEND", "cpu").strip().lower()
    return gs.gpu if backend_name == "gpu" else gs.cpu


def set_dofs_kp_kv_compat(entity, kp=None, kv=None, dofs_idx_local=None) -> bool:
    """Best-effort PD gain setter across Genesis actuator variants."""

    ok = True
    if kp is not None:
        try:
            if dofs_idx_local is None:
                entity.set_dofs_kp(kp)
            else:
                try:
                    entity.set_dofs_kp(kp, dofs_idx_local=dofs_idx_local)
                except TypeError:
                    entity.set_dofs_kp(kp, dofs_idx_local)
        except Exception:
            ok = False
    if kv is not None:
        try:
            if dofs_idx_local is None:
                entity.set_dofs_kv(kv)
            else:
                try:
                    entity.set_dofs_kv(kv, dofs_idx_local=dofs_idx_local)
                except TypeError:
                    entity.set_dofs_kv(kv, dofs_idx_local)
        except Exception:
            ok = False
    return ok


def update_dofs_kp_kv_compat(entity, dof_indices, *, kp_value=None, kv_value=None) -> bool:
    """Patch selected DOF gains when Genesis exposes PD-reducible actuators."""

    try:
        kp = to_numpy(entity.get_dofs_kp()).copy()
        kv = to_numpy(entity.get_dofs_kv()).copy()
    except Exception:
        return False
    for idx in dof_indices:
        if kp_value is not None:
            kp[int(idx)] = float(kp_value)
        if kv_value is not None:
            kv[int(idx)] = float(kv_value)
    return set_dofs_kp_kv_compat(entity, kp=kp, kv=kv)


def set_dofs_force_range_compat(entity, lower, upper, dofs_idx_local=None) -> bool:
    """Set force ranges across old and latest Genesis signatures."""

    try:
        if dofs_idx_local is not None:
            entity.set_dofs_force_range(lower, upper, dofs_idx_local=dofs_idx_local)
            return True
    except TypeError:
        pass
    except Exception:
        return False
    try:
        entity.set_dofs_force_range(lower, upper)
        return True
    except Exception:
        return False


def get_dofs_force_range_pair_compat(entity):
    """Return ``(lower, upper)`` arrays for either Genesis force-range shape."""

    force_range = entity.get_dofs_force_range()
    if isinstance(force_range, (tuple, list)) and len(force_range) == 2:
        return to_numpy(force_range[0]).copy(), to_numpy(force_range[1]).copy()
    arr = to_numpy(force_range).copy()
    if arr.ndim == 2 and arr.shape[0] == 2:
        return arr[0].copy(), arr[1].copy()
    if arr.ndim == 2 and arr.shape[1] == 2:
        return arr[:, 0].copy(), arr[:, 1].copy()
    raise ValueError(f"unsupported DOF force-range shape {arr.shape}")


def update_dofs_force_range_compat(
    entity,
    dof_indices,
    *,
    lower_value=None,
    upper_value=None,
) -> bool:
    """Patch selected DOF force ranges when Genesis exposes them."""

    try:
        lower, upper = get_dofs_force_range_pair_compat(entity)
    except Exception:
        return False
    for idx in dof_indices:
        idx = int(idx)
        if lower_value is not None:
            lower[idx] = float(lower_value)
        if upper_value is not None:
            upper[idx] = float(upper_value)
    return set_dofs_force_range_compat(entity, lower, upper)


def hold_dof_position_compat(
    entity,
    dof_index: int,
    target: float,
    *,
    kp: float = 2000.0,
    kv: float = 200.0,
    force_limit: float = 500.0,
) -> bool:
    """Configure and hold one articulation DOF at a physical PD target.

    This is intended for non-robot articulated fixtures such as drawers,
    doors, and lids after another actor has opened them.  It does not mutate
    qpos: the current full articulation state is copied into a position-control
    target and only ``dof_index`` is replaced.  The command therefore resists
    later contact without teleporting the fixture or disturbing sibling DOFs.
    """
    idx = int(dof_index)
    update_dofs_kp_kv_compat(
        entity, [idx], kp_value=float(kp), kv_value=float(kv),
    )
    update_dofs_force_range_compat(
        entity,
        [idx],
        lower_value=-abs(float(force_limit)),
        upper_value=+abs(float(force_limit)),
    )
    try:
        target_qpos = to_numpy(entity.get_qpos()).copy().reshape(-1)
        target_qpos[idx] = float(target)
        entity.control_dofs_position(target_qpos)
        return True
    except Exception:
        return False


def control_dof_force_compat(entity, dof_index: int, force: float) -> bool:
    """Apply motor force to one articulation DOF across Genesis signatures."""
    idx = int(dof_index)
    value = np.asarray([float(force)], dtype=float)
    indices = np.asarray([idx], dtype=np.int32)
    try:
        entity.control_dofs_force(value, dofs_idx_local=indices)
        return True
    except TypeError:
        try:
            entity.control_dofs_force(value, indices)
            return True
        except Exception:
            pass
    except Exception:
        pass
    try:
        full = np.zeros_like(to_numpy(entity.get_qpos()), dtype=float).reshape(-1)
        full[idx] = float(force)
        entity.control_dofs_force(full)
        return True
    except Exception:
        return False


def control_dof_force_toward_position_compat(
    entity,
    dof_index: int,
    target: float,
    *,
    kp: float = 20.0,
    kv: float = 4.0,
    force_limit: float = 4.0,
) -> tuple[bool, float, float, float]:
    """Take one force-control step toward a DOF position target.

    Unlike position control, this does not make the target move on its own:
    callers must invoke it while the physical interaction is active and step
    the simulator.  The returned values are ``(ok, qpos, qvel, force)``.
    """
    idx = int(dof_index)
    try:
        qpos = float(to_numpy(entity.get_qpos()).reshape(-1)[idx])
        qvel = float(to_numpy(entity.get_dofs_velocity()).reshape(-1)[idx])
    except Exception:
        return False, float("nan"), float("nan"), 0.0
    force = float(np.clip(
        float(kp) * (float(target) - qpos) - float(kv) * qvel,
        -abs(float(force_limit)),
        +abs(float(force_limit)),
    ))
    update_dofs_force_range_compat(
        entity,
        [idx],
        lower_value=-abs(float(force_limit)),
        upper_value=+abs(float(force_limit)),
    )
    ok = control_dof_force_compat(entity, idx, force)
    return ok, qpos, qvel, force


def get_dof_idx(joint):
    """Return an entity-local DOF index across Genesis versions."""
    idx = getattr(joint, "dof_idx_local", None)
    if idx is None and hasattr(joint, "get_dof_indices_local"):
        indices = joint.get_dof_indices_local()
        idx = indices[0] if indices else None
    if idx is None:
        idx = getattr(joint, "dof_start", None)
    return int(idx) if idx is not None else None


@lru_cache(maxsize=1)
def genesis_world_version() -> tuple[int, ...]:
    """Return installed genesis-world version as a numeric tuple."""
    try:
        raw = md.version("genesis-world")
    except Exception:
        return ()
    parts = []
    for token in raw.replace("-", ".").split("."):
        if not token.isdigit():
            break
        parts.append(int(token))
    return tuple(parts)


def is_latest_genesis() -> bool:
    """True for upstream Genesis 1.x+ runtimes."""
    version = genesis_world_version()
    return bool(version and version >= (1, 0, 0))


def ensure_inertial_urdf(
    urdf_path,
    *,
    default_mass: float = 1.0,
    default_inertia: float = 0.02,
    link_overrides: dict | None = None,
):
    """Return a URDF path whose links all carry explicit ``<inertial>`` tags.

    Many SAPIEN ``mobility.urdf`` assets ship with NO inertial blocks. Genesis
    1.0.0 then auto-computes inertia from the (often non-watertight) collision
    meshes, which can yield a degenerate/singular mass matrix and crash on the
    first sim step with ``Invalid constraint forces causing 'nan'``. Old
    Genesis tolerated this; 1.0.0 does not.

    On latest Genesis, if ``urdf_path`` has any inertial-less non-empty link we
    write a sibling ``<stem>_inertial.urdf`` with well-conditioned diagonal
    inertials injected and return it; otherwise (or on old Genesis) we return
    ``urdf_path`` unchanged. Generation is idempotent — an up-to-date sibling
    is reused.

    ``link_overrides`` maps link name -> ``{"mass", "ixx", "iyy", "izz"}`` for
    per-link tuning (e.g. a heavier cabinet body); links not listed get the
    ``default_*`` values.
    """
    import xml.etree.ElementTree as ET
    from pathlib import Path

    src = Path(urdf_path)
    if not is_latest_genesis() or not src.exists():
        return src

    link_overrides = link_overrides or {}
    tree = ET.parse(src)
    root = tree.getroot()

    def needs_inertial(link) -> bool:
        # Skip empty placeholder links (e.g. a bare fixed root) — no geometry,
        # nothing to give inertia.
        has_geom = link.find("collision") is not None or link.find("visual") is not None
        return has_geom and link.find("inertial") is None

    targets = [lk for lk in root.findall("link") if needs_inertial(lk)]
    if not targets:
        return src

    dst = src.with_name(f"{src.stem}_inertial{src.suffix}")
    # Idempotent + race-safe: if a previous run already produced a non-empty,
    # well-formed sibling, reuse it. Multiple task processes load the same
    # cabinet concurrently; without this they'd all rewrite ``dst`` and a
    # reader could catch it mid-write (empty) -> ``ParseError: no element
    # found``. We additionally write atomically (temp + os.replace) below.
    if dst.exists() and dst.stat().st_size > 0:
        try:
            ET.parse(dst)
            return dst
        except ET.ParseError:
            pass  # truncated/corrupt — regenerate atomically below

    for link in targets:
        ov = link_overrides.get(link.get("name"), {})
        mass = float(ov.get("mass", default_mass))
        ixx = float(ov.get("ixx", default_inertia))
        iyy = float(ov.get("iyy", default_inertia))
        izz = float(ov.get("izz", default_inertia))
        el = ET.Element("inertial")
        ET.SubElement(el, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
        ET.SubElement(el, "mass", {"value": repr(mass)})
        ET.SubElement(el, "inertia", {
            "ixx": repr(ixx), "ixy": "0", "ixz": "0",
            "iyy": repr(iyy), "iyz": "0", "izz": repr(izz),
        })
        link.insert(0, el)

    # Atomic write: a unique temp file in the same dir, then os.replace().
    # os.replace is atomic on POSIX, so concurrent readers always see either
    # the old complete file or the new complete file — never a partial write.
    tmp = dst.with_name(f"{dst.stem}.{os.getpid()}.tmp")
    tree.write(tmp, encoding="utf-8", xml_declaration=True)
    os.replace(tmp, dst)
    return dst


def ensure_nyx_urdf(
    urdf_path,
    scale: float = 1.0,
    *,
    default_mass: float = 1.0,
    default_inertia: float = 0.02,
):
    """Return a URDF rewritten so the NYX sub-scene renderer shows it whole.

    gs-nyx-plugin 0.1.4 re-parses the URDF file at render time and silently
    DROPS links attached by ``fixed`` joints (only the root and movable-joint
    children render). SAPIEN chairs/laptops hang their whole body under a
    fixed root joint (with an rpy), so they render as floating wheels / a
    detached lid. Cabinets get away with it because their root has only
    movable children. The re-parse also ignores Genesis's runtime ``scale``.

    This helper writes a sibling ``<stem>_nyx_v2_s<key>.urdf`` that:
      1. merges every fixed-joint child into its parent (visual/collision
         origins composed through the joint transform; downstream joints
         re-parented), mirroring Genesis's merge_fixed_links=True at the
         file level so file links == simulated links,
      2. bakes ``scale`` into all lengths (load with scale=1.0),
      3. injects explicit <inertial> tags (same rationale as
         ensure_inertial_urdf).
    Load the result with ``merge_fixed_links=False`` (nothing left to merge,
    and NYX rejects the True flag). Idempotent + atomic.
    """
    import xml.etree.ElementTree as ET
    from pathlib import Path

    import transforms3d as t3d

    src = Path(urdf_path)
    if not src.exists():
        return src
    scale = float(scale)

    key = repr(scale).replace(".", "p").replace("-", "m")
    # v2 stores one textured GLB visual per link. The previous geometry-only
    # OBJ cache fixed missing parts but discarded the source materials.
    dst = src.with_name(f"{src.stem}_nyx_v2_s{key}{src.suffix}")
    if dst.exists() and dst.stat().st_size > 0:
        try:
            cached_root = ET.parse(dst).getroot()
            mesh_files = [
                dst.parent / mesh.get("filename")
                for mesh in cached_root.iter("mesh")
                if mesh.get("filename")
            ]
            if all(path.exists() and path.stat().st_size > 0 for path in mesh_files):
                return dst
        except ET.ParseError:
            pass  # truncated — regenerate atomically below

    tree = ET.parse(src)
    root = tree.getroot()

    def origin_to_T(el):
        T = np.eye(4)
        if el is not None:
            xyz = [float(v) for v in (el.get("xyz") or "0 0 0").split()]
            rpy = [float(v) for v in (el.get("rpy") or "0 0 0").split()]
            T[:3, :3] = t3d.euler.euler2mat(*rpy, "sxyz")
            T[:3, 3] = xyz
        return T

    def T_to_attrs(T):
        rpy = t3d.euler.mat2euler(T[:3, :3], "sxyz")
        return {
            "xyz": " ".join(repr(float(v)) for v in T[:3, 3]),
            "rpy": " ".join(repr(float(v)) for v in rpy),
        }

    def set_element_origin(el, T):
        origin = el.find("origin")
        if origin is None:
            origin = ET.SubElement(el, "origin")
        attrs = T_to_attrs(T)
        origin.set("xyz", attrs["xyz"])
        origin.set("rpy", attrs["rpy"])

    # ---- 1. merge fixed-joint children into their parents ----
    while True:
        fixed = next((j for j in root.findall("joint") if j.get("type") == "fixed"), None)
        if fixed is None:
            break
        parent_name = fixed.find("parent").get("link")
        child_name = fixed.find("child").get("link")
        T_pc = origin_to_T(fixed.find("origin"))
        links = {l.get("name"): l for l in root.findall("link")}
        parent_link = links[parent_name]
        child_link = links[child_name]
        for tag in ("visual", "collision"):
            for el in list(child_link.findall(tag)):
                T_el = origin_to_T(el.find("origin"))
                set_element_origin(el, T_pc @ T_el)
                child_link.remove(el)
                parent_link.append(el)
        for j in root.findall("joint"):
            p = j.find("parent")
            if p.get("link") == child_name:
                p.set("link", parent_name)
                T_j = origin_to_T(j.find("origin"))
                set_element_origin(j, T_pc @ T_j)
        root.remove(fixed)
        root.remove(child_link)

    # ---- 1b. merge each link's visuals into a single mesh ----
    # Empirically the NYX sub-scene shows only single-visual links reliably
    # for these SAPIEN assets (chair wheels/backrest and laptop lid render,
    # the multi-visual bodies vanish). Combine per-link visual geometry into
    # one textured GLB so every link has exactly one URDF <visual> while the
    # original per-part materials and texture images remain embedded.
    for link in root.findall("link"):
        visuals = link.findall("visual")
        if len(visuals) <= 1:
            continue
        import trimesh

        parts = trimesh.Scene()
        for part_index, v in enumerate(visuals):
            geom = v.find("geometry")
            mesh_el = geom.find("mesh") if geom is not None else None
            if mesh_el is None:
                parts = None
                break
            mesh_path = src.parent / mesh_el.get("filename")
            m = trimesh.load(str(mesh_path), force="mesh", process=False)
            mscale = mesh_el.get("scale")
            if mscale:
                m.apply_scale([float(x) for x in mscale.split()])
            T = origin_to_T(v.find("origin"))
            m.apply_transform(T)
            part_name = f"part_{part_index}"
            parts.add_geometry(m, geom_name=part_name, node_name=part_name)
        if parts is None or not parts.geometry:
            continue  # non-mesh visual present — leave the link alone
        out_name = f"_nyx_textured_{link.get('name')}.glb"
        out_path = src.parent / out_name
        if not (out_path.exists() and out_path.stat().st_size > 0):
            tmp_mesh = out_path.with_name(f"{out_path.stem}.{os.getpid()}.tmp.glb")
            tmp_mesh.write_bytes(trimesh.exchange.gltf.export_glb(parts))
            os.replace(tmp_mesh, out_path)
        for v in visuals:
            link.remove(v)
        el = ET.SubElement(link, "visual")
        ET.SubElement(el, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
        geom = ET.SubElement(el, "geometry")
        ET.SubElement(geom, "mesh", {"filename": out_name})

    # ---- 2. bake scale into all lengths ----
    if abs(scale - 1.0) > 1e-9:
        for origin in root.iter("origin"):
            raw = origin.get("xyz")
            if raw is not None:
                origin.set("xyz", " ".join(repr(float(v) * scale) for v in raw.split()))
        for mesh in root.iter("mesh"):
            raw = mesh.get("scale", "1 1 1")
            mesh.set("scale", " ".join(repr(float(v) * scale) for v in raw.split()))
        for box in root.iter("box"):
            raw = box.get("size")
            if raw is not None:
                box.set("size", " ".join(repr(float(v) * scale) for v in raw.split()))
        for cyl in root.iter("cylinder"):
            for attr in ("radius", "length"):
                if cyl.get(attr) is not None:
                    cyl.set(attr, repr(float(cyl.get(attr)) * scale))
        for sph in root.iter("sphere"):
            if sph.get("radius") is not None:
                sph.set("radius", repr(float(sph.get("radius")) * scale))
        for joint in root.iter("joint"):
            if joint.get("type") == "prismatic":
                limit = joint.find("limit")
                if limit is not None:
                    for attr in ("lower", "upper"):
                        if limit.get(attr) is not None:
                            limit.set(attr, repr(float(limit.get(attr)) * scale))

    # ---- 3. inject inertials (see ensure_inertial_urdf) ----
    for link in root.findall("link"):
        has_geom = link.find("collision") is not None or link.find("visual") is not None
        if not has_geom or link.find("inertial") is not None:
            continue
        el = ET.Element("inertial")
        ET.SubElement(el, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
        ET.SubElement(el, "mass", {"value": repr(float(default_mass))})
        ET.SubElement(el, "inertia", {
            "ixx": repr(float(default_inertia)), "ixy": "0", "ixz": "0",
            "iyy": repr(float(default_inertia)), "iyz": "0",
            "izz": repr(float(default_inertia)),
        })
        link.insert(0, el)

    tmp = dst.with_name(f"{dst.stem}.{os.getpid()}.tmp")
    tree.write(tmp, encoding="utf-8", xml_declaration=True)
    os.replace(tmp, dst)
    return dst


def prepare_articulated_urdf_for_renderer(
    urdf_path,
    *,
    renderer: str,
    scale: float = 1.0,
    default_mass: float = 1.0,
    default_inertia: float = 0.02,
    link_overrides: dict | None = None,
):
    """Return a complete, stable URDF load recipe for an articulated asset.

    Physics renderers keep the normal runtime scale and Genesis fixed-link
    merging. NYX instead receives a file with fixed links pre-merged, all
    per-link visuals consolidated, and scale baked into the URDF. This avoids
    the recurring failure mode where only one panel, drawer, or appliance door
    appears even though the articulation exists in physics.

    Returns ``(path, load_scale, merge_fixed_links)``. The final value is
    ``None`` when Genesis's default should be used.
    """
    path = ensure_inertial_urdf(
        urdf_path,
        default_mass=default_mass,
        default_inertia=default_inertia,
        link_overrides=link_overrides,
    )
    load_scale = float(scale)
    merge_fixed_links = None
    if str(renderer).strip().lower() == "nyx":
        path = ensure_nyx_urdf(
            path,
            load_scale,
            default_mass=default_mass,
            default_inertia=default_inertia,
        )
        load_scale = 1.0
        merge_fixed_links = False
    return Path(path), load_scale, merge_fixed_links
