"""Runtime NYX compatibility for latest-Genesis kinematic custom vertices.

Latest Genesis represents the benchmark avatar as a visualization-only
``KinematicEntity`` whose skinned vertices are driven through
``KinematicEntity.set_vverts``.  gs-nyx-plugin 0.1.x only streams dynamic
vertices for FEM/PBD/MPM/SPH entities, so these kinematic custom-vvert meshes
otherwise get exported as static file meshes.
"""

from __future__ import annotations

import numpy as np
import torch

import genesis as gs


_PATCHED = False


def _is_custom_kinematic_entity(entity) -> bool:
    kinematic_cls = getattr(gs.engine.entities, "KinematicEntity", ())
    rigid_cls = getattr(gs.engine.entities, "RigidEntity", ())
    return (
        isinstance(entity, kinematic_cls)
        and not isinstance(entity, rigid_cls)
        and bool(getattr(entity, "_nyx_dynamic_custom_vverts", False))
        and bool(getattr(getattr(entity, "morph", None), "enable_custom_vverts", False))
    )


def _custom_kinematic_entities(renderer) -> list:
    return [
        entity
        for entity in getattr(renderer._scene, "entities", [])
        if _is_custom_kinematic_entity(entity)
    ]


def _entity_vfaces(entity) -> np.ndarray:
    faces = getattr(entity, "_nyx_custom_vfaces", None)
    if faces is not None:
        return np.asarray(faces, dtype=np.int32).reshape(-1, 3)

    parts = []
    offset = 0
    for vgeom in entity.vgeoms:
        vf = np.asarray(vgeom.init_vfaces, dtype=np.int32).reshape(-1, 3)
        parts.append(vf + offset)
        offset += int(vgeom.n_vverts)
    return np.concatenate(parts, axis=0) if parts else np.zeros((0, 3), dtype=np.int32)


def _entity_vuvs(entity) -> np.ndarray:
    uvs = getattr(entity, "_nyx_custom_vuvs", None)
    if uvs is None:
        return np.zeros((int(entity.n_vverts), 2), dtype=np.float32)
    uvs = np.asarray(uvs, dtype=np.float32).reshape(-1, 2)
    if uvs.shape[0] != int(entity.n_vverts):
        return np.zeros((int(entity.n_vverts), 2), dtype=np.float32)
    return uvs


def _base_deform_counts(renderer):
    nr = renderer.__class__
    return (
        nr._nyx_orig_num_deform_verts.fget(renderer),
        nr._nyx_orig_num_deform_tris.fget(renderer),
        nr._nyx_orig_num_deform_objs.fget(renderer),
    )


def apply_nyx_kinematic_patch() -> None:
    """Patch the installed NYX plugin once per process."""

    global _PATCHED
    if _PATCHED:
        return

    from gs_nyx import nyx_py_sdk as nps
    import gs_nyx_plugin.nyx_scene_utils as nsu
    import gs_nyx_plugin.nyx_scene_exporter as nse
    import gs_nyx_plugin.nyx_renderer as nr

    orig_utils_dynamic = nsu.is_dynamic_mesh_entity
    orig_export_dynamic = nse.is_dynamic_mesh_entity
    orig_utils_geom_level = nsu.should_export_at_geom_level
    orig_export_geom_level = nse.should_export_at_geom_level
    orig_export_rigid = nse.is_rigid_entity

    def is_dynamic_mesh_entity(entity):
        return orig_utils_dynamic(entity) or _is_custom_kinematic_entity(entity)

    def export_is_dynamic_mesh_entity(entity):
        return orig_export_dynamic(entity) or _is_custom_kinematic_entity(entity)

    def should_export_at_geom_level(entity):
        if _is_custom_kinematic_entity(entity):
            return False
        return orig_utils_geom_level(entity)

    def export_should_export_at_geom_level(entity):
        if _is_custom_kinematic_entity(entity):
            return False
        return orig_export_geom_level(entity)

    def export_is_rigid_entity(entity):
        return orig_export_rigid(entity) or _is_custom_kinematic_entity(entity)

    nsu.is_dynamic_mesh_entity = is_dynamic_mesh_entity
    nse.is_dynamic_mesh_entity = export_is_dynamic_mesh_entity
    nsu.should_export_at_geom_level = should_export_at_geom_level
    nse.should_export_at_geom_level = export_should_export_at_geom_level
    nse.is_rigid_entity = export_is_rigid_entity

    orig_build_entity_instance = nse.NyxSceneExporter._build_entity_instance

    def build_entity_instance(self, entity, instance_idx):
        if not _is_custom_kinematic_entity(entity):
            return orig_build_entity_instance(self, entity, instance_idx)

        instance = self._scene_asset.get_instance(instance_idx)
        instance.type = nps.EInstanceType.DynamicMesh
        instance.dynamicMesh_numVertices = int(entity.n_vverts)
        instance.dynamicMesh_numTriangles = int(_entity_vfaces(entity).shape[0])
        instance.position = nps.float3(0.0, 0.0, 0.0)
        instance.rotation = nps.quaternion(0.0, 0.0, 0.0, 1.0)
        instance.scale = nps.float3_z_up_to_y_up_a(nps.float3(1.0, 1.0, 1.0))

        surface = getattr(entity, "_nyx_custom_surface", None) or entity.surface
        instance.matOverride = self._build_full_material(surface, entity)
        instance.smooth = False
        instance.enabled = True
        instance.uuid = nps.generate_uuid()
        self._entity_uuid_pairs.append((entity, instance.uuid))
        self._scene_asset.set_instance(instance_idx, instance)
        return instance_idx + 1

    nse.NyxSceneExporter._build_entity_instance = build_entity_instance

    cls = nr.NyxPyRenderer
    if not hasattr(cls, "_nyx_orig_num_deform_verts"):
        cls._nyx_orig_num_deform_verts = cls._num_deform_verts
        cls._nyx_orig_num_deform_tris = cls._num_deform_tris
        cls._nyx_orig_num_deform_objs = cls._num_deform_objs

    def num_custom_kinematic_verts(self) -> int:
        return sum(int(entity.n_vverts) for entity in _custom_kinematic_entities(self))

    def num_custom_kinematic_tris(self) -> int:
        return sum(int(_entity_vfaces(entity).shape[0]) for entity in _custom_kinematic_entities(self))

    def num_custom_kinematic_entities(self) -> int:
        return len(_custom_kinematic_entities(self))

    def num_deform_verts(self) -> int:
        return cls._nyx_orig_num_deform_verts.fget(self) + num_custom_kinematic_verts(self)

    def num_deform_tris(self) -> int:
        return cls._nyx_orig_num_deform_tris.fget(self) + num_custom_kinematic_tris(self)

    def num_deform_objs(self) -> int:
        return cls._nyx_orig_num_deform_objs.fget(self) + num_custom_kinematic_entities(self)

    cls._num_custom_kinematic_verts = property(num_custom_kinematic_verts)
    cls._num_custom_kinematic_tris = property(num_custom_kinematic_tris)
    cls._num_custom_kinematic_entities = property(num_custom_kinematic_entities)
    cls._num_deform_verts = property(num_deform_verts)
    cls._num_deform_tris = property(num_deform_tris)
    cls._num_deform_objs = property(num_deform_objs)

    orig_allocate_cuda_buffers = cls._allocate_cuda_buffers

    def allocate_cuda_buffers(self):
        orig_allocate_cuda_buffers(self)
        base_verts, base_tris, _ = _base_deform_counts(self)
        self._custom_kinematic_vert_start = base_verts
        self._custom_kinematic_tri_start = base_tris
        n_verts = self._num_custom_kinematic_verts
        if n_verts > 0:
            self._deformable_custom_kinematic_verts_view = self._deformable_verts_cuda[
                base_verts : base_verts + n_verts
            ]

    cls._allocate_cuda_buffers = allocate_cuda_buffers

    orig_build_reference_table = cls._build_reference_table

    def build_reference_table(self, update_desc, pair_list):
        orig_build_reference_table(self, update_desc, pair_list)
        _, _, base_objs = _base_deform_counts(self)
        ref_idx = update_desc.numRigidGeom + base_objs
        for entity in _custom_kinematic_entities(self):
            current_ref = update_desc.get_reference(ref_idx)
            current_ref.targetUUID = nr.entity_to_uuid(pair_list, entity)
            current_ref.nodeName = ""
            update_desc.set_reference(ref_idx, current_ref)
            ref_idx += 1

    cls._build_reference_table = build_reference_table

    orig_upload_static_deformable_data = cls._upload_static_deformable_data

    def upload_static_deformable_data(self):
        orig_upload_static_deformable_data(self)
        vert = int(getattr(self, "_custom_kinematic_vert_start", 0))
        tri = int(getattr(self, "_custom_kinematic_tri_start", 0))
        for entity in _custom_kinematic_entities(self):
            n_v = int(entity.n_vverts)
            faces_np = _entity_vfaces(entity)
            n_t = int(faces_np.shape[0])
            if n_v == 0 or n_t == 0:
                continue
            uvs = torch.as_tensor(_entity_vuvs(entity), dtype=torch.float32, device="cuda")
            faces = torch.as_tensor(faces_np, dtype=torch.int32, device="cuda").reshape(-1)
            self._deformable_uvs_cuda[vert : vert + n_v].copy_(uvs)
            self._deformable_indices_cuda[tri * 3 : (tri + n_t) * 3].copy_(faces + vert)
            vert += n_v
            tri += n_t

    cls._upload_static_deformable_data = upload_static_deformable_data

    orig_update_geometry_tensors = cls._update_geometry_tensors

    def update_geometry_tensors(self, env_index):
        orig_update_geometry_tensors(self, env_index)
        entities = _custom_kinematic_entities(self)
        if not entities:
            return
        parts = []
        for entity in entities:
            verts = entity.get_vverts(envs_idx=env_index)
            if hasattr(verts, "detach"):
                verts = verts.detach()
            if getattr(verts, "ndim", 0) == 3:
                verts = verts[0]
            parts.append(verts.to(torch.float32))
        if parts:
            self._deformable_custom_kinematic_verts_view.copy_(torch.cat(parts, dim=0))

    cls._update_geometry_tensors = update_geometry_tensors

    _PATCHED = True
