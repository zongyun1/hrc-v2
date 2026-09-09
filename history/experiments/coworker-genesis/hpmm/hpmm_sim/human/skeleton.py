"""SMPL-X -> rigid articulated skeleton (MJCF) for Genesis.

The human is a *kinematically driven* rigid body in v0 of the plan: it collides with the
robot and the kitchen, but is never simulated dynamically — every frame its configuration
is overwritten from a :class:`~hpmm_motion.io.schema.MotionSequence`. This module builds
the body; :mod:`hpmm_sim.human.driver` does the driving.

Why an articulated MJCF rather than N free-floating capsules: MuJoCo ball joints map
one-to-one onto SMPL-X's per-joint rotations, and Genesis parses them
(``mjJNT_BALL -> JOINT_TYPE.SPHERICAL``). So a pose transfers as *quaternions straight
into* ``set_qpos`` — no axis-angle-to-Euler decomposition, no gimbal singularities, no
risk of bones drifting apart under the solver. The chain also guarantees the limbs stay
attached no matter what the solver does between overrides.

Layout of the generated model, mirroring SMPL-X's own kinematic tree exactly:

* root body ``pelvis`` carries a ``freejoint``  -> 7 qpos (xyz + wxyz quat)
* each of the other 21 body joints carries a ``ball`` joint -> 4 qpos each
* total 91 qpos / 69 dof, ordered by the depth-first body order of the emitted XML

Bone geometry comes from :mod:`tools.extract_smplx_skeleton`'s JSON template, so nothing
here needs the SMPL-X model file or the ``smplx`` package. Coordinates stay in the
**SMPL-X model frame** (Y-up); the conversion to the scene's Z-up world is carried
entirely by ``global_orient``/``transl``, which the motion backend has already expressed
in world terms per the schema contract.

This module is numpy-only and does not import Genesis, so the MJCF can be generated and
unit-tested in any venv.
"""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass

import numpy as np

DEFAULT_TEMPLATE = pathlib.Path(__file__).resolve().parents[2] / "assets" / "smplx_skeleton_male.json"

#: Target total body mass (kg) for a betas=0 adult male. Density is solved for rather
#: than assumed: the capsules deliberately overlap (three of them share the pelvis), so
#: summing their volumes at flesh density over-counts by roughly 5x. The human is driven
#: kinematically and its own inertia never moves it, but contact force is a Phase 3
#: safety signal, so the body should at least weigh what a body weighs.
TARGET_MASS = 70.0

#: Collision bitmasks. Human geoms see the world but not each other: self-pairs give
#: ``contype & conaffinity == 0`` both ways, while the world's default (1, 1) still
#: matches. Set directly rather than via ``<contact><exclude>`` so Genesis passes the
#: masks through untouched instead of re-solving them.
HUMAN_CONTYPE, HUMAN_CONAFFINITY = 1, 2


def axis_angle_to_quat(aa: np.ndarray) -> np.ndarray:
    """Axis-angle ``(..., 3)`` -> unit quaternion ``(..., 4)`` in w-x-y-z order."""
    aa = np.asarray(aa, dtype=np.float64)
    angle = np.linalg.norm(aa, axis=-1, keepdims=True)
    # sin(a/2)/a, expanded near zero so a rotation of exactly 0 stays finite.
    small = angle < 1e-8
    half = 0.5 * angle
    scale = np.where(small, 0.5 - angle**2 / 48.0, np.sin(half) / np.where(small, 1.0, angle))
    return np.concatenate([np.cos(half), aa * scale], axis=-1)


@dataclass
class HumanSkeleton:
    """The rigid-body stand-in for one SMPL-X subject."""

    joint_names: list[str]
    parents: list[int]
    rest_offsets: np.ndarray   # (J, 3) parent-relative, model frame
    rest_joints: np.ndarray    # (J, 3) model frame
    bones: list[dict]
    gender: str
    prefix: str = "human_"

    @classmethod
    def from_template(cls, path=DEFAULT_TEMPLATE, prefix: str = "human_") -> "HumanSkeleton":
        path = pathlib.Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"{path} is missing. Body-model assets are derived from SMPL-X and are not "
                "committed (see README, 'Body-model assets'); regenerate with\n"
                "  cd third_party/trumans_utils && ../trumans-venv/bin/python "
                "../../tools/extract_smplx_skeleton.py"
            )
        template = json.loads(path.read_text())
        return cls(
            joint_names=list(template["joint_names"]),
            parents=list(template["parents"]),
            rest_offsets=np.asarray(template["rest_offsets"], dtype=np.float64),
            rest_joints=np.asarray(template["rest_joints"], dtype=np.float64),
            bones=list(template["bones"]),
            gender=template["gender"],
            prefix=prefix,
        )

    # -- structure -------------------------------------------------------------

    @property
    def n_joints(self) -> int:
        return len(self.joint_names)

    @property
    def n_qs(self) -> int:
        """7 for the free root + 4 per ball joint."""
        return 7 + 4 * (self.n_joints - 1)

    @property
    def n_dofs(self) -> int:
        return 6 + 3 * (self.n_joints - 1)

    @property
    def rest_pelvis(self) -> np.ndarray:
        """Rest position of joint 0 in the model frame.

        SMPL-X leaves the pelvis at its rest location when posing (the root rotation is
        about that point) and only then adds ``transl``, so the world pelvis is
        ``transl + rest_pelvis``. The free joint needs that sum, not ``transl`` alone.
        """
        return self.rest_joints[0]

    def body_name(self, j: int) -> str:
        return f"{self.prefix}{self.joint_names[j]}"

    def dfs_order(self) -> list[int]:
        """Joint indices in the depth-first order the XML nests them.

        MuJoCo numbers joints by body order, so this is also the order their qpos blocks
        appear in. Kept explicit rather than assumed — :mod:`driver` re-checks it by name.
        """
        children: dict[int, list[int]] = {j: [] for j in range(self.n_joints)}
        for j, p in enumerate(self.parents):
            if p >= 0:
                children[p].append(j)
        order: list[int] = []
        stack = [0]
        while stack:
            j = stack.pop()
            order.append(j)
            stack.extend(reversed(children[j]))
        return order

    # -- pose -> qpos ----------------------------------------------------------

    def qpos(self, transl: np.ndarray, global_orient: np.ndarray, body_pose: np.ndarray) -> np.ndarray:
        """Build Genesis ``qpos`` from SMPL-X parameters.

        Parameters
        ----------
        transl : ``(..., 3)``
        global_orient : ``(..., 3)`` axis-angle
        body_pose : ``(..., 21, 3)`` axis-angle, SMPL-X joint order

        Returns
        -------
        ``(..., n_qs)`` float32, laid out as ``[root xyz, root quat, ball quats...]`` in
        :meth:`dfs_order`.
        """
        transl = np.asarray(transl, dtype=np.float64)
        body_pose = np.asarray(body_pose, dtype=np.float64)
        batch = transl.shape[:-1]
        if body_pose.shape[-2:] != (self.n_joints - 1, 3):
            raise ValueError(f"body_pose must be (..., {self.n_joints - 1}, 3), got {body_pose.shape}")

        root_quat = axis_angle_to_quat(global_orient)
        joint_quats = axis_angle_to_quat(body_pose)  # (..., 21, 4)

        parts = [transl + self.rest_pelvis, root_quat]
        for j in self.dfs_order()[1:]:
            parts.append(joint_quats[..., j - 1, :])  # body_pose is indexed from joint 1
        return np.concatenate([p.reshape(*batch, -1) for p in parts], axis=-1).astype(np.float32)

    # -- MJCF emission ---------------------------------------------------------

    def capsule_volume(self) -> float:
        """Summed volume of the capsules, overlaps counted once per capsule."""
        total = 0.0
        for bone in self.bones:
            r = float(bone["radius"])
            h = float(np.linalg.norm(np.asarray(bone["to"]) - np.asarray(bone["from"])))
            total += np.pi * r * r * h + 4.0 / 3.0 * np.pi * r**3
        return total

    def density(self, target_mass: float = TARGET_MASS) -> float:
        return target_mass / self.capsule_volume()

    def to_mjcf(self, name: str = "smplx_human", rgba=(0.85, 0.65, 0.55, 1.0),
                target_mass: float = TARGET_MASS) -> str:
        children: dict[int, list[int]] = {j: [] for j in range(self.n_joints)}
        for j, p in enumerate(self.parents):
            if p >= 0:
                children[p].append(j)
        by_body: dict[str, list[dict]] = {}
        for bone in self.bones:
            by_body.setdefault(bone["body"], []).append(bone)

        def emit(j: int, depth: int) -> list[str]:
            pad = "  " * depth
            offset = self.rest_offsets[j] if self.parents[j] >= 0 else np.zeros(3)
            lines = [f'{pad}<body name="{self.body_name(j)}" pos="{_vec(offset)}">']
            if self.parents[j] < 0:
                lines.append(f'{pad}  <freejoint name="{self.prefix}root"/>')
            else:
                lines.append(f'{pad}  <joint name="{self.body_name(j)}" type="ball"/>')
            for bone in by_body.get(self.joint_names[j], []):
                lines.append(
                    f'{pad}  <geom name="{self.prefix}{bone["name"]}" type="capsule"'
                    f' fromto="{_vec(bone["from"])} {_vec(bone["to"])}"'
                    f' size="{bone["radius"]:.4f}"/>'
                )
            for c in children[j]:
                lines += emit(c, depth + 1)
            lines.append(f"{pad}</body>")
            return lines

        body_xml = "\n".join(emit(0, 3))
        return f"""<mujoco model="{name}">
  <compiler angle="radian" autolimits="true"/>
  <default>
    <geom density="{self.density(target_mass):.4f}" rgba="{_vec(rgba)}"
          contype="{HUMAN_CONTYPE}" conaffinity="{HUMAN_CONAFFINITY}"
          friction="0.8 0.005 0.0001"/>
  </default>
  <worldbody>
{body_xml}
  </worldbody>
</mujoco>
"""

    def write_mjcf(self, path, **kwargs) -> pathlib.Path:
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_mjcf(**kwargs))
        return path


def _vec(v) -> str:
    return " ".join(f"{float(x):.6g}" for x in np.asarray(v).ravel())


__all__ = ["HumanSkeleton", "axis_angle_to_quat", "DEFAULT_TEMPLATE", "TARGET_MASS"]
