"""A real SMPL-X surface stretched over the rigid capsule skeleton.

The capsules are what the physics engine sees, and they read as a pile of spheres on
video. This module puts the actual SMPL-X body mesh on top of them, for looking at only.

The mesh is deformed by **linear blend skinning driven from the skeleton's own link
poses**, which Genesis already reports. That avoids the two obvious alternatives and their
costs: linking `smplx` (and its 100 MB licence-encumbered model) into the Genesis venv, or
storing a vertex trajectory (10475 x 3 x T floats — tens of megabytes per episode).

For a vertex ``i`` skinned to joints ``j`` with weights ``w``:

    v_world = sum_j w_ij * [ R_j (v_rest - rest_joint_j) + p_j ]

which is exactly SMPL's LBS, using the fact that in the rest pose every link frame is a
pure translation to its joint. The one thing left out is SMPL-X's pose-corrective
blendshapes, so elbows and knees crease slightly less than in the real model — invisible at
video scale, and worth the independence from the model file.
"""

from __future__ import annotations

import pathlib

import numpy as np

DEFAULT_ASSET = pathlib.Path(__file__).resolve().parents[2] / "assets" / "smplx_skin_male.npz"


def quats_to_R(quats: np.ndarray) -> np.ndarray:
    """``(N, 4)`` w-x-y-z quaternions -> ``(N, 3, 3)`` rotation matrices."""
    q = np.asarray(quats, dtype=np.float64)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], axis=-1),
        np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], axis=-1),
        np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], axis=-1),
    ], axis=-2)


class HumanSkin:
    """The SMPL-X surface plus the weights that bind it to the 22-link skeleton."""

    def __init__(self, rest_verts, faces, rest_joints, weight_idx, weight_val):
        self.rest_verts = np.asarray(rest_verts, dtype=np.float64)
        self.faces = np.asarray(faces, dtype=np.int32)
        self.rest_joints = np.asarray(rest_joints, dtype=np.float64)
        self.weight_idx = np.asarray(weight_idx, dtype=np.int64)
        self.weight_val = np.asarray(weight_val, dtype=np.float64)
        # Precompute each vertex's offset from every joint it is bound to; this is the
        # only per-vertex work that does not change frame to frame.
        self._offsets = self.rest_verts[:, None, :] - self.rest_joints[self.weight_idx]

    @classmethod
    def from_asset(cls, path=DEFAULT_ASSET) -> "HumanSkin":
        path = pathlib.Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"{path} is missing. Body-model assets are derived from SMPL-X and are not "
                "committed (see README, 'Body-model assets'); regenerate with\n"
                "  cd third_party/trumans_utils && ../trumans-venv/bin/python "
                "../../tools/extract_smplx_skeleton.py"
            )
        with np.load(path) as data:
            return cls(data["rest_verts"], data["faces"], data["rest_joints"],
                       data["weight_idx"], data["weight_val"])

    @property
    def n_verts(self) -> int:
        return len(self.rest_verts)

    def __repr__(self) -> str:
        return f"HumanSkin({self.n_verts} verts, {len(self.faces)} faces)"

    def write_obj(self, path) -> pathlib.Path:
        """Write the rest mesh, for Genesis to load as the visual entity.

        Only the topology and vertex *count* matter downstream — every frame overwrites the
        positions — but the rest pose is written faithfully so the file is inspectable.
        """
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            f.write("# SMPL-X body surface, rest pose (model frame)\n")
            np.savetxt(f, self.rest_verts, fmt="v %.6f %.6f %.6f")
            np.savetxt(f, self.faces + 1, fmt="f %d %d %d")
        return path

    def world_vertices(self, link_pos: np.ndarray, link_quat: np.ndarray) -> np.ndarray:
        """Skin the mesh from link world poses, ordered by SMPL-X joint index.

        Parameters
        ----------
        link_pos : ``(22, 3)``
        link_quat : ``(22, 4)`` w-x-y-z

        Returns
        -------
        ``(V, 3)`` world-space vertices.
        """
        R = quats_to_R(link_quat)                       # (22, 3, 3)
        R_v = R[self.weight_idx]                        # (V, k, 3, 3)
        p_v = np.asarray(link_pos, dtype=np.float64)[self.weight_idx]   # (V, k, 3)
        posed = np.einsum("vkij,vkj->vki", R_v, self._offsets) + p_v
        return np.einsum("vk,vki->vi", self.weight_val, posed)


__all__ = ["HumanSkin", "quats_to_R", "DEFAULT_ASSET"]
