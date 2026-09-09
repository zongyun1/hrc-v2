"""★ FROZEN CONTRACT — the SMPL-X motion sequence format.

This module is interface ② of the project plan: the only channel between the human
motion layer (``hpmm_motion``, simulator-agnostic) and the simulation layer
(``hpmm_sim``, Genesis). Both legs of Phase 1 develop against this file and nothing
else, so changes here are breaking changes: bump ``SCHEMA_VERSION`` and say why.

Conventions (fixed, do not renegotiate per-backend):

* **Batch-first.** Every per-sequence array is ``[B, T, ...]`` where ``B`` is the
  number of parallel environments. A single sequence is ``B == 1``, never a bare
  ``[T, ...]``. Invariant #1 of the plan; retrofitting a batch axis later is costly.
* **World frame is the scene's.** Z-up, metres, RoboCasa/Genesis world coordinates.
  Backends that generate in another frame (TRUMANS is Y-up and scene-centred) do the
  remap *inside the backend*; nothing downstream of this file ever sees generator
  coordinates.
* **Rotations are axis-angle**, SMPL-X ordering, radians. ``global_orient`` rotates the
  body into the world frame; ``transl`` is the pelvis position after that rotation, in
  the same sense SMPL-X's own ``transl`` argument uses.
* **numpy only.** No torch, no genesis. This module must import in the TRUMANS venv,
  the Genesis venv and a bare interpreter alike.

Storage is a single ``.npz`` (``save``/``load``), with ``meta`` carried as an embedded
JSON string so the file stays self-describing without a sidecar.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np

SCHEMA_VERSION = "1.0"

#: SMPL-X body joints driven by ``body_pose`` (``pelvis`` is driven by ``global_orient``).
N_BODY_JOINTS = 21
#: Finger joints per hand in the full (non-PCA) SMPL-X hand pose.
N_HAND_JOINTS = 15
#: Shape coefficients. SMPL-X supports more; the pipeline is fixed at 10.
N_BETAS = 10

GENDERS = ("male", "female", "neutral")

#: Names of the 22 body joints, in SMPL-X order. Index 0 is the root; indices 1..21
#: line up with the second axis of ``body_pose``.
BODY_JOINT_NAMES = (
    "pelvis",
    "left_hip", "right_hip", "spine1",
    "left_knee", "right_knee", "spine2",
    "left_ankle", "right_ankle", "spine3",
    "left_foot", "right_foot", "neck",
    "left_collar", "right_collar", "head",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
)

#: Kinematic-tree parent of each entry of :data:`BODY_JOINT_NAMES` (-1 for the root).
BODY_JOINT_PARENTS = (
    -1,
    0, 0, 0,
    1, 2, 3,
    4, 5, 6,
    7, 8, 9,
    9, 9, 12,
    13, 14,
    16, 17,
    18, 19,
)


class SchemaError(ValueError):
    """Raised when a payload does not conform to this contract."""


@dataclass
class MotionSequence:
    """A batch of SMPL-X motion sequences in scene-world coordinates.

    Attributes
    ----------
    transl, global_orient : ``(B, T, 3)``
        Pelvis translation (m) and root orientation (axis-angle, rad).
    body_pose : ``(B, T, 21, 3)``
        Body joint rotations, axis-angle, parent-relative, SMPL-X order.
    betas : ``(B, 10)``
        Shape coefficients, constant over time.
    fps : float
        Sampling rate of the time axis.
    gender : str
        SMPL-X model gender the parameters were fitted with; see :data:`GENDERS`.
    left_hand_pose, right_hand_pose : ``(B, T, 15, 3)`` or None
        Full (non-PCA) finger rotations. ``None`` means "hands unspecified" — consumers
        fall back to the model's relaxed pose rather than assuming zeros.
    valid : ``(B, T)`` bool or None
        Per-frame mask, for batches whose sequences differ in length. ``None`` means all
        frames of all sequences are valid.
    joints : ``(B, T, J, 3)`` or None
        **Derived, optional cache** of world-frame joint positions, so consumers that
        only need a skeleton (the Genesis driver, clearance metrics) need not link
        against ``smplx``. Never treat it as authoritative: it is regenerable from the
        parameters above and may be absent.
    meta : dict
        Free-form provenance: backend, scene id, layout, seed, waypoints, action labels.
        Never load-bearing for geometry.
    """

    transl: np.ndarray
    global_orient: np.ndarray
    body_pose: np.ndarray
    betas: np.ndarray
    fps: float
    gender: str = "neutral"
    left_hand_pose: np.ndarray | None = None
    right_hand_pose: np.ndarray | None = None
    valid: np.ndarray | None = None
    joints: np.ndarray | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    # -- shape helpers ---------------------------------------------------------

    @property
    def n_envs(self) -> int:
        return int(self.transl.shape[0])

    @property
    def n_frames(self) -> int:
        return int(self.transl.shape[1])

    @property
    def duration(self) -> float:
        """Sequence length in seconds."""
        return self.n_frames / self.fps

    def __len__(self) -> int:
        return self.n_envs

    def __repr__(self) -> str:
        hands = "hands" if self.left_hand_pose is not None else "no-hands"
        joints = f"joints[{self.joints.shape[2]}]" if self.joints is not None else "no-joints"
        return (
            f"MotionSequence(B={self.n_envs}, T={self.n_frames}, {self.fps:g}fps, "
            f"{self.duration:.1f}s, {self.gender}, {hands}, {joints})"
        )

    def env(self, i: int) -> "MotionSequence":
        """Slice out one environment, keeping the batch axis (``B == 1``)."""
        sl = slice(i, i + 1)
        return MotionSequence(
            transl=self.transl[sl],
            global_orient=self.global_orient[sl],
            body_pose=self.body_pose[sl],
            betas=self.betas[sl],
            fps=self.fps,
            gender=self.gender,
            left_hand_pose=None if self.left_hand_pose is None else self.left_hand_pose[sl],
            right_hand_pose=None if self.right_hand_pose is None else self.right_hand_pose[sl],
            valid=None if self.valid is None else self.valid[sl],
            joints=None if self.joints is None else self.joints[sl],
            meta=dict(self.meta),
        )

    # -- validation ------------------------------------------------------------

    def validate(self) -> "MotionSequence":
        """Check every invariant of the contract. Returns self so it can be chained."""
        B, T = self.n_envs, self.n_frames
        _check(self.transl, "transl", (B, T, 3))
        _check(self.global_orient, "global_orient", (B, T, 3))
        _check(self.body_pose, "body_pose", (B, T, N_BODY_JOINTS, 3))
        _check(self.betas, "betas", (B, N_BETAS))
        if self.left_hand_pose is not None:
            _check(self.left_hand_pose, "left_hand_pose", (B, T, N_HAND_JOINTS, 3))
        if self.right_hand_pose is not None:
            _check(self.right_hand_pose, "right_hand_pose", (B, T, N_HAND_JOINTS, 3))
        if self.joints is not None:
            _check(self.joints, "joints", (B, T, None, 3))
        if self.valid is not None:
            if self.valid.shape != (B, T):
                raise SchemaError(f"valid: expected shape {(B, T)}, got {self.valid.shape}")
            if self.valid.dtype != np.bool_:
                raise SchemaError(f"valid: expected dtype bool, got {self.valid.dtype}")

        if not (np.isfinite(self.fps) and self.fps > 0):
            raise SchemaError(f"fps must be finite and positive, got {self.fps!r}")
        if self.gender not in GENDERS:
            raise SchemaError(f"gender must be one of {GENDERS}, got {self.gender!r}")
        if not isinstance(self.meta, dict):
            raise SchemaError(f"meta must be a dict, got {type(self.meta).__name__}")
        return self

    # -- storage ---------------------------------------------------------------

    def save(self, path) -> None:
        """Write to a ``.npz``. Validates first — malformed data never reaches disk."""
        self.validate()
        arrays = {
            "transl": self.transl,
            "global_orient": self.global_orient,
            "body_pose": self.body_pose,
            "betas": self.betas,
        }
        for name in ("left_hand_pose", "right_hand_pose", "valid", "joints"):
            value = getattr(self, name)
            if value is not None:
                arrays[name] = value
        arrays["__header__"] = np.array(
            json.dumps(
                {
                    "schema_version": SCHEMA_VERSION,
                    "fps": float(self.fps),
                    "gender": self.gender,
                    "meta": self.meta,
                }
            )
        )
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path) -> "MotionSequence":
        """Read a ``.npz`` written by :meth:`save`, validating on the way in."""
        with np.load(path, allow_pickle=False) as data:
            header = json.loads(str(data["__header__"]))
            version = header.get("schema_version")
            if version != SCHEMA_VERSION:
                raise SchemaError(
                    f"{path}: schema version {version!r} != this build's {SCHEMA_VERSION!r}"
                )
            kwargs = {k: data[k] for k in data.files if not k.startswith("__")}
        return cls(
            fps=float(header["fps"]),
            gender=header.get("gender", "neutral"),
            meta=header.get("meta", {}),
            **kwargs,
        ).validate()


def _check(arr: Any, name: str, shape: tuple[int | None, ...]) -> None:
    """Validate one float array: type, shape (``None`` = any), dtype, finiteness."""
    if not isinstance(arr, np.ndarray):
        raise SchemaError(f"{name}: expected np.ndarray, got {type(arr).__name__}")
    if len(arr.shape) != len(shape) or any(
        want is not None and got != want for got, want in zip(arr.shape, shape)
    ):
        pretty = tuple("*" if s is None else s for s in shape)
        raise SchemaError(f"{name}: expected shape {pretty}, got {arr.shape}")
    if arr.dtype != np.float32:
        raise SchemaError(f"{name}: expected dtype float32, got {arr.dtype}")
    if not np.isfinite(arr).all():
        raise SchemaError(f"{name}: contains non-finite values")


def empty(n_envs: int, n_frames: int, fps: float = 30.0, **kwargs) -> MotionSequence:
    """A rest-pose sequence of the given size — a scaffold for tests and backends."""
    return MotionSequence(
        transl=np.zeros((n_envs, n_frames, 3), np.float32),
        global_orient=np.zeros((n_envs, n_frames, 3), np.float32),
        body_pose=np.zeros((n_envs, n_frames, N_BODY_JOINTS, 3), np.float32),
        betas=np.zeros((n_envs, N_BETAS), np.float32),
        fps=fps,
        **kwargs,
    )


__all__ = [
    "SCHEMA_VERSION", "MotionSequence", "SchemaError", "empty",
    "N_BODY_JOINTS", "N_HAND_JOINTS", "N_BETAS", "GENDERS",
    "BODY_JOINT_NAMES", "BODY_JOINT_PARENTS",
]
