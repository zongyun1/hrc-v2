"""Compatibility skin wrapper for Genesis builds without AvatarEntity.

Latest upstream Genesis removed the public Avatar material/entity path used by
this benchmark.  KinematicEntity now exposes custom visual vertices through
``set_vverts``; this wrapper restores the small AvatarEntity surface that the
motion code needs and drives those kinematic vertices directly.
"""

from __future__ import annotations

import copy
import hashlib
import os
import tempfile
from pathlib import Path

import numpy as np
import torch
import trimesh
from PIL import Image

import genesis as gs
from genesis.utils import geom as gu
from genesis.utils import mesh as mu

try:
    from genesis.utils import gltf as gltf_utils
except Exception:  # pragma: no cover - import failure is reported at use site.
    gltf_utils = None

from scipy.spatial.transform import Rotation


_MAPPING = [
    0, 1, 2, 3, 4, 7, 10, 11, 12, 13, 16, 17, 18, 19, 20, 21, 22, 23,
    24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
    42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
    58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72,
]


def _transform_quat_by_quat(u, v):
    return np.array([
        u[3] * v[0] + u[0] * v[3] + u[1] * v[2] - u[2] * v[1],
        u[3] * v[1] - u[0] * v[2] + u[1] * v[3] + u[2] * v[0],
        u[3] * v[2] + u[0] * v[1] - u[1] * v[0] + u[2] * v[3],
        u[3] * v[3] - u[0] * v[0] - u[1] * v[1] - u[2] * v[2],
    ])


def _parse_node_matrix(node):
    if node.matrix is not None:
        return np.array(node.matrix, dtype=float).reshape((4, 4))

    matrix = np.identity(4, dtype=float)
    scale_matrix = np.identity(4, dtype=float)
    rotation_matrix = np.identity(4, dtype=float)
    translation_matrix = np.identity(4, dtype=float)

    if node.translation is not None:
        translation_matrix[3, :3] = np.array(node.translation, dtype=float)
    if node.rotation is not None:
        rotation = np.array(node.rotation, dtype=float)  # xyzw
        rotation = [rotation[3], rotation[0], rotation[1], rotation[2]]
        rotation_matrix[:3, :3] = trimesh.transformations.quaternion_matrix(rotation)[:3, :3].T
    if node.scale is not None:
        scale_matrix = np.diag(np.append(np.array(node.scale, dtype=float), 1))

    return scale_matrix @ rotation_matrix @ translation_matrix


def _forward_kinematics(nodes, node_index, root_matrix=None, root_id=None, set_parent=False):
    if root_matrix is None:
        root_matrix = np.identity(4, dtype=float)
    node = nodes[node_index]
    if root_id is None:
        matrix = _parse_node_matrix(node) @ root_matrix
    else:
        matrix = root_matrix

    if node_index == root_id:
        root_id = None

    out = [[node_index, matrix]]
    for child in node.children:
        if set_parent:
            nodes[child].parent = node_index
        out.extend(_forward_kinematics(nodes, child, matrix, root_id, set_parent=set_parent))
    return out


def _parse_avatar_skin_infos(path, group_by_material, scale):
    if gltf_utils is None:
        raise RuntimeError("Genesis GLTF utilities are unavailable; cannot load kinematic avatar skin.")

    glb = gltf_utils.pygltflib.GLTF2().load(path)
    if glb is None:
        raise RuntimeError(f"Failed to load avatar GLB: {path}")
    glb.convert_images(gltf_utils.pygltflib.ImageFormat.DATAURI)
    glb.path = path

    glb_scene = glb.scene or 0
    scene = glb.scenes[glb_scene]
    mesh_list = []
    for node_index in scene.nodes:
        mesh_list.extend(gltf_utils.parse_glb_tree(glb, node_index))
    if not glb.skins:
        raise RuntimeError(f"Avatar GLB has no skin: {path}")

    materials = {}
    for material_idx, material in enumerate(glb.materials or []):
        color_image = None
        color_factor = np.ones(4, dtype=np.float32)
        pbr = material.pbrMetallicRoughness
        if pbr is not None:
            if pbr.baseColorFactor is not None:
                color_factor = np.asarray(pbr.baseColorFactor, dtype=np.float32)
            if pbr.baseColorTexture is not None:
                texture = glb.textures[pbr.baseColorTexture.index]
                color_image = gltf_utils.get_glb_image(glb, texture.source, "RGBA")
        materials[material_idx] = {
            "name": material.name,
            "color_factor": color_factor,
            "color_image": color_image,
        }

    skin = glb.skins[0]
    skin_root = scene.nodes[-1]
    global_transforms = np.asarray(
        [it[1] for it in sorted(_forward_kinematics(glb.nodes, skin_root), key=lambda x: x[0])]
    )
    global_joint_mat = global_transforms[skin.joints]
    inv_bind_mat = gltf_utils.get_glb_data_from_accessor(glb, skin.inverseBindMatrices).reshape(-1, 4, 4).astype(float)

    groups = {}
    for mesh_i, (mesh_index, mesh_transform) in enumerate(mesh_list):
        mesh = glb.meshes[mesh_index]
        for prim_i, primitive in enumerate(mesh.primitives):
            group_idx = primitive.material if group_by_material else mesh_i

            if "KHR_draco_mesh_compression" in primitive.extensions:
                khr_index = primitive.extensions["KHR_draco_mesh_compression"]["bufferView"]
                view = glb.bufferViews[khr_index]
                data = gltf_utils.get_glb_bufferview_data(glb, view)
                decoded = gltf_utils.DracoPy.decode(data[view.byteOffset : view.byteOffset + view.byteLength])
                points = decoded.points.astype(float)
                triangles = decoded.faces.astype(np.int32)
                normals = decoded.normals if decoded.normals is not None and len(decoded.normals) > 0 else None
                uvs = np.zeros((len(points), 2), dtype=float)
                joints = gltf_utils.get_glb_data_from_accessor(glb, primitive.attributes.JOINTS_0).reshape(-1, 4).astype(int)
                weights = gltf_utils.get_glb_data_from_accessor(glb, primitive.attributes.WEIGHTS_0).reshape(-1, 4).astype(float)
            else:
                points = gltf_utils.get_glb_data_from_accessor(glb, primitive.attributes.POSITION).astype(float)
                if primitive.indices is None:
                    triangles = np.arange(len(points), dtype=np.int32).reshape(-1, 3)
                else:
                    triangles = gltf_utils.get_glb_data_from_accessor(glb, primitive.indices).astype(np.int32).reshape(-1, 3)
                if primitive.attributes.NORMAL:
                    normals = gltf_utils.get_glb_data_from_accessor(glb, primitive.attributes.NORMAL).astype(float)
                else:
                    normals = None
                uv_accessor = getattr(primitive.attributes, "TEXCOORD_0", None)
                if uv_accessor is not None:
                    uvs = gltf_utils.get_glb_data_from_accessor(glb, uv_accessor).astype(float)
                else:
                    uvs = np.zeros((len(points), 2), dtype=float)
                joints = gltf_utils.get_glb_data_from_accessor(glb, primitive.attributes.JOINTS_0).reshape(-1, 4).astype(int)
                weights = gltf_utils.get_glb_data_from_accessor(glb, primitive.attributes.WEIGHTS_0).reshape(-1, 4).astype(float)

            skin_mat = inv_bind_mat @ global_joint_mat
            skin_mat = (weights * skin_mat[joints].transpose(2, 3, 0, 1)).transpose(2, 3, 0, 1).sum(axis=1)
            points_homo = np.append(points, np.ones((points.shape[0], 1)), axis=-1)[:, None]
            points = (points_homo @ skin_mat)[:, 0, :3]
            if normals is None:
                normals = trimesh.Trimesh(points, triangles, process=False).vertex_normals
            normals_homo = np.append(normals, np.ones((normals.shape[0], 1)), axis=-1)[:, None]
            normals = (normals_homo @ skin_mat)[:, 0, :3]

            mesh_transform = mesh_transform.astype(float)
            denom = np.abs(mesh_transform).max(axis=-1)
            denom[denom == 0.0] = 1.0
            mesh_transform = mesh_transform / denom[:, None]
            points, normals = mu.apply_transform(mesh_transform, points, normals)
            points = points * scale

            group = groups.setdefault(
                group_idx,
                {
                    "points": [],
                    "triangles": [],
                    "n_points": 0,
                    "init_matrix": mesh_transform,
                    "vert_joints": [],
                    "vert_weights": [],
                    "vuvs": [],
                    "face_materials": [],
                    "vertex_materials": [],
                },
            )
            triangles = triangles + group["n_points"]
            group["points"].append(points)
            group["triangles"].append(triangles)
            group["n_points"] += len(points)
            group["vert_joints"].append(joints)
            group["vert_weights"].append(weights)
            group["vuvs"].append(uvs)
            material_idx = -1 if primitive.material is None else int(primitive.material)
            group["face_materials"].append(
                np.full((triangles.shape[0],), material_idx, dtype=np.int32)
            )
            group["vertex_materials"].append(
                np.full((points.shape[0],), material_idx, dtype=np.int32)
            )

    infos = []
    for group_idx in groups:
        group = groups[group_idx]
        infos.append(
            {
                "vverts": np.concatenate(group["points"]),
                "vfaces": np.concatenate(group["triangles"]),
                "init_matrix": group["init_matrix"],
                "vert_joints": np.concatenate(group["vert_joints"]),
                "vert_invbind": inv_bind_mat,
                "vert_weights": np.concatenate(group["vert_weights"]),
                "vuvs": np.concatenate(group["vuvs"]),
                "face_materials": np.concatenate(group["face_materials"]),
                "vertex_materials": np.concatenate(group["vertex_materials"]),
                "materials": materials,
                "skin_joints": np.asarray(skin.joints),
                "nodes": [skin_root, glb.nodes],
            }
        )
    return infos


def _atlas_image_for_material(material):
    image = material.get("color_image", None)
    factor = np.asarray(material.get("color_factor", [1.0, 1.0, 1.0, 1.0]), dtype=np.float32)
    if image is None:
        rgba = np.clip(factor, 0.0, 1.0)
        if rgba.shape[0] < 4:
            rgba = np.append(rgba[:3], 1.0)
        return np.full((16, 16, 4), np.rint(rgba[:4] * 255.0), dtype=np.uint8)

    rgba = np.asarray(image, dtype=np.float32)
    if rgba.ndim != 3:
        return np.full((16, 16, 4), 255, dtype=np.uint8)
    if rgba.shape[2] == 3:
        alpha = np.full((*rgba.shape[:2], 1), 255.0, dtype=np.float32)
        rgba = np.concatenate([rgba, alpha], axis=2)
    if factor.shape[0] < 4:
        factor = np.append(factor[:3], 1.0)
    rgba[..., :4] *= factor[:4]
    return np.clip(rgba, 0.0, 255.0).astype(np.uint8)


def _write_nyx_texture_atlas(infos, glb_path):
    materials = infos[0].get("materials", {}) if infos else {}
    material_ids = sorted(materials)
    if not material_ids:
        return None

    images = {idx: _atlas_image_for_material(materials[idx]) for idx in material_ids}
    slot_w = max(int(img.shape[1]) for img in images.values())
    slot_h = max(int(img.shape[0]) for img in images.values())
    slot_w = max(slot_w, 16)
    slot_h = max(slot_h, 16)

    atlas = Image.new("RGBA", (slot_w * len(material_ids), slot_h), (255, 255, 255, 255))
    slots = {}
    resample = getattr(getattr(Image, "Resampling", Image), "LANCZOS", Image.BILINEAR)
    for slot, material_idx in enumerate(material_ids):
        img = Image.fromarray(images[material_idx], mode="RGBA")
        if img.size != (slot_w, slot_h):
            img = img.resize((slot_w, slot_h), resample=resample)
        atlas.paste(img, (slot * slot_w, 0))
        slots[material_idx] = slot

    for info in infos:
        uvs = np.asarray(info["vuvs"], dtype=np.float32).copy()
        vertex_materials = np.asarray(info.get("vertex_materials", []), dtype=np.int32)
        if vertex_materials.shape[0] != uvs.shape[0]:
            vertex_materials = np.full((uvs.shape[0],), material_ids[0], dtype=np.int32)
        uvs = uvs - np.floor(uvs)
        for material_idx, slot in slots.items():
            mask = vertex_materials == material_idx
            if np.any(mask):
                uvs[mask, 0] = (slot + uvs[mask, 0]) / len(material_ids)
        info["nyx_vuvs"] = uvs

    texture_dir = Path(
        os.environ.get(
            "NYX_AVATAR_TEXTURE_DIR",
            Path.cwd() / "outputs" / "avatar_latest_genesis_nyx" / "textures",
        )
    )
    texture_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha1(str(Path(glb_path).resolve()).encode("utf-8")).hexdigest()[:10]
    texture_path = texture_dir / f"{Path(glb_path).stem}_{digest}_atlas.png"

    # Atomic write: parallel jobs (e.g. a 30-wide SLURM array) share this cache
    # path. Saving in place lets a concurrent reader open a half-written PNG
    # ("cannot identify image file" / "broken PNG" / "image file is truncated"),
    # which kills AvatarController init. Write to a private temp file in the same
    # directory, then os.replace() — an atomic rename on POSIX — so readers only
    # ever see a complete file.
    if not texture_path.exists():
        fd, tmp_name = tempfile.mkstemp(
            dir=str(texture_dir), prefix=f".{texture_path.stem}.", suffix=".tmp.png"
        )
        os.close(fd)
        try:
            atlas.save(tmp_name)
            os.replace(tmp_name, texture_path)
        finally:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)
    return texture_path


class KinematicAvatarSkin:
    """Old AvatarEntity API backed by a latest-Genesis KinematicEntity."""

    def __init__(self, entity, glb_path: str, group_by_material: bool = False, scale=1.0):
        self._entity = entity
        self._morph = entity.morph
        self._infos = _parse_avatar_skin_infos(glb_path, group_by_material, scale)
        self._nyx_texture_atlas = _write_nyx_texture_atlas(self._infos, glb_path)

        first = self._infos[0]
        self._init_nodes = copy.deepcopy(first["nodes"])
        self._nodes = copy.deepcopy(self._init_nodes)
        self.node_findup = {n.name.split(":")[-1]: i for i, n in enumerate(self._nodes[1])}
        self._skin_joints = np.asarray(first["skin_joints"], dtype=np.int32)

        self._vert_joints_cuda = []
        self._vert_weights_cuda = []
        self._points_homo_cuda = []
        for info in self._infos:
            init_matrix = np.asarray(info["init_matrix"], dtype=gs.np_float)
            init_matrix_inv = np.asarray(np.matrix(init_matrix).I)
            points_homo = (np.append(info["vverts"], np.ones((info["vverts"].shape[0], 1)), axis=-1) @ init_matrix_inv)[:, None]
            self._vert_joints_cuda.append(torch.from_numpy(np.asarray(info["vert_joints"], dtype=gs.np_int)).to(torch.int32).to(gs.device))
            self._vert_weights_cuda.append(torch.from_numpy(np.asarray(info["vert_weights"], dtype=gs.np_float)).to(torch.float64).to(gs.device))
            self._points_homo_cuda.append(torch.from_numpy(points_homo).to(torch.float64).to(gs.device))

        self._vert_joints_cuda = torch.cat(self._vert_joints_cuda, dim=0)
        self._vert_weights_cuda = torch.cat(self._vert_weights_cuda, dim=0)
        self._points_homo_cuda = torch.cat(self._points_homo_cuda, dim=0)
        self._vert_invbind_cuda = torch.from_numpy(np.asarray(first["vert_invbind"], dtype=gs.np_float)).to(torch.float64).to(gs.device)
        self._init_matrix_cuda = torch.from_numpy(np.asarray(first["init_matrix"], dtype=gs.np_float)).to(torch.float64).to(gs.device)

        self.transforms = None
        self.global_transforms = None
        self._latest_vverts = None
        self._latest_render_vverts = None
        self._render_rot = gu.euler_to_R(
            np.asarray(getattr(self._morph, "euler", (0, 0, 0)), dtype=float)
        )
        self._entity._nyx_dynamic_custom_vverts = True
        self._entity._nyx_custom_vuvs = np.concatenate(
            [np.asarray(info.get("nyx_vuvs", info["vuvs"]), dtype=np.float32) for info in self._infos],
            axis=0,
        )
        face_parts = []
        offset = 0
        for info in self._infos:
            faces = np.asarray(info["vfaces"], dtype=np.int32)
            face_parts.append(faces + offset)
            offset += int(info["vverts"].shape[0])
        self._entity._nyx_custom_vfaces = np.concatenate(face_parts, axis=0)
        if self._nyx_texture_atlas is not None:
            self._entity._nyx_custom_surface = gs.surfaces.Default(
                diffuse_texture=gs.textures.ImageTexture(image_path=str(self._nyx_texture_atlas)),
                roughness=0.75,
                double_sided=True,
            )

        # Motion code historically reaches through skin.links[0]._vgeoms[0]
        # to obtain skeleton metadata.  Attach that metadata to latest Genesis'
        # visual geom object so existing call sites keep working.
        try:
            vgeom = self.links[0]._vgeoms[0]
            vgeom._skin_joints = self._skin_joints
            vgeom._nodes = self._nodes
            vgeom._init_nodes = self._init_nodes
        except Exception:
            pass

    def __getattr__(self, name):
        return getattr(self._entity, name)

    @property
    def links(self):
        return self._entity.links

    def calculate_joint_T(self, base_translation, global_mat, global_mat_inv, skin_base_rot, node_trans):
        skin_root_id = self._skin_joints[0]
        root = self._nodes[1][skin_root_id]
        init_root = self._init_nodes[1][skin_root_id]
        if len(self._skin_joints) > 65:
            root.translation = init_root.translation + base_translation
        else:
            root.translation = init_root.translation + base_translation * 100

        rotation = np.matmul(np.matmul(global_mat_inv[0], gu.quat_to_R(skin_base_rot)), global_mat[0])
        rotation = gu.R_to_quat(rotation)[[1, 2, 3, 0]]
        root.rotation = _transform_quat_by_quat(
            init_root.rotation if init_root.rotation is not None else [0, 0, 0, 1],
            rotation,
        )
        return node_trans @ _parse_node_matrix(root) @ _parse_node_matrix(self._nodes[1][self._nodes[0]])

    def calculate_real_pos(self, rotation, base_translation, root_id=None):
        root = self._skin_joints[0]
        init_root = self._init_nodes[1][root]
        if len(self._skin_joints) > 65:
            self._nodes[1][root].translation = init_root.translation + base_translation
            for i, r in enumerate(rotation[:65]):
                joint_id = self._skin_joints[_MAPPING[i]]
                init = self._init_nodes[1][joint_id]
                self._nodes[1][joint_id].rotation = _transform_quat_by_quat(
                    init.rotation if init.rotation is not None else [0, 0, 0, 1], r
                )
        else:
            self._nodes[1][root].translation = init_root.translation + base_translation * 100
            for i, joint_id in enumerate(self._skin_joints):
                init = self._init_nodes[1][joint_id]
                self._nodes[1][joint_id].rotation = _transform_quat_by_quat(
                    init.rotation if init.rotation is not None else [0, 0, 0, 1], rotation[i]
                )

        mats = _forward_kinematics(self._nodes[1], self._nodes[0], root_id=root_id, set_parent=True)
        self.transforms = np.asarray([it[1] for it in sorted(mats, key=lambda x: x[0])])
        return self.transforms

    def update_mesh(self, base_translation, global_mat, global_mat_inv, skin_base_rot, node_trans):
        self.global_transforms = self.calculate_joint_T(
            base_translation, global_mat, global_mat_inv, skin_base_rot, node_trans
        )
        skin_mat = self._vert_invbind_cuda @ torch.from_numpy(
            self.global_transforms[self._skin_joints]
        ).to(torch.float64).to(gs.device)
        skin_mat = torch.einsum("ij,ijmn->imn", self._vert_weights_cuda, skin_mat[self._vert_joints_cuda])
        result = (self._points_homo_cuda @ skin_mat @ self._init_matrix_cuda)[:, 0, :3].cpu().numpy()
        self._latest_vverts = result.astype(np.float32)
        # Keep _latest_vverts in the old software-skin frame for external
        # renderers, but Genesis custom vverts need the mesh morph transform
        # baked in because KinematicEntity.set_vverts bypasses it.
        self._latest_render_vverts = (
            self._latest_vverts @ self._render_rot.T
            + np.asarray(getattr(self._morph, "pos", (0, 0, 0)), dtype=np.float32)
        ).astype(np.float32)

        if self._entity.is_built:
            expected = int(self._entity.n_vverts)
            if self._latest_render_vverts.shape[0] != expected:
                raise RuntimeError(
                    f"Avatar skin vertex count mismatch: computed {self._latest_render_vverts.shape[0]}, "
                    f"Genesis entity expects {expected}."
                )
            self._entity.set_vverts(self._latest_render_vverts)

    def fabrik(self, node_index, root_node_index, target_pos, max_iterations=10, tolerance=1e-3):
        chain = []
        current_index = node_index
        nodes = self._nodes[1]
        while current_index != root_node_index:
            chain.append(current_index)
            current_index = nodes[current_index].parent
        chain.append(root_node_index)
        chain.reverse()

        positions = [self.transforms[idx][-1, :3].copy() for idx in chain]
        bone_lengths = [np.linalg.norm(positions[i + 1] - positions[i]) for i in range(len(chain) - 1)]
        root_pos = positions[0].copy()
        for _ in range(max_iterations):
            positions[-1] = target_pos.copy()
            for i in range(len(positions) - 2, -1, -1):
                direction = (positions[i] - positions[i + 1]) / np.linalg.norm(positions[i] - positions[i + 1])
                positions[i] = positions[i + 1] + bone_lengths[i] * direction
            positions[0] = root_pos
            for i in range(1, len(positions)):
                direction = (positions[i] - positions[i - 1]) / np.linalg.norm(positions[i] - positions[i - 1])
                positions[i] = positions[i - 1] + bone_lengths[i - 1] * direction
            if np.linalg.norm(positions[-1] - target_pos) < tolerance:
                break

        def get_rotation_between_vectors(v1, v2):
            v1 = v1 / np.linalg.norm(v1)
            v2 = v2 / np.linalg.norm(v2)
            cos_theta = np.dot(v1, v2)
            angle = np.arccos(cos_theta)
            axis = np.cross(v1, v2)
            axis = axis / np.linalg.norm(axis)
            return Rotation.from_rotvec(angle * axis).as_quat()

        matrix = self.transforms[nodes[chain[0]].parent]
        for i in range(1, len(chain)):
            pos = np.linalg.inv(matrix.T) @ np.append(positions[i], 1.0)
            vec = pos[:3] - nodes[chain[i - 1]].translation
            nodes[chain[i - 1]].rotation = get_rotation_between_vectors(nodes[chain[i]].translation, vec)
            matrix = _parse_node_matrix(nodes[chain[i - 1]]) @ matrix

        mats = _forward_kinematics(self._nodes[1], self._nodes[0], root_id=self._skin_joints[0], set_parent=True)
        return np.asarray([it[1] for it in sorted(mats, key=lambda x: x[0])])

    def ik_solve(self, hand_id, root_id, target_pos, max_iterations=10, tolerance=1e-3):
        hand_idx = self.node_findup[hand_id]
        root_idx = self.node_findup[root_id]
        target_pos[2] -= self._morph.pos[2]
        target_pos[0] *= -1
        target_pos[:2] = target_pos[:2][::-1]
        return self.fabrik(hand_idx, root_idx, target_pos, max_iterations, tolerance)

    def get_node_translation(self, name):
        return self.global_transforms[self.node_findup[name]]

    def get_global_translation(self, name):
        matrix4x4_t = self.get_node_translation(name).copy()
        pos = matrix4x4_t[-1, :3]
        pos[:2] = pos[:2][::-1]
        pos[0] *= -1
        pos[2] += self._morph.pos[2]
        rot = matrix4x4_t[:3, :3]
        rot = (rot / np.linalg.norm(rot, axis=1, keepdims=True)).T
        return pos, rot
