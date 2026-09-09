"""Simulator-independent playback of HARPv2 schema v1 and its body template.

Internal quaternions are WXYZ. Isaac Lab 3 conversion happens in the adapter.
No Genesis runtime, SMPL-X package, or model weights are imported here.
"""
from pathlib import Path
import hashlib
import numpy as np
from .vendor.schema import MotionSequence, BODY_JOINT_NAMES, BODY_JOINT_PARENTS
from .vendor.skeleton import HumanSkeleton, axis_angle_to_quat
from .vendor.skin import HumanSkin, quats_to_R


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def multiply(a, b):
    aw, av, bw, bv = a[..., :1], a[..., 1:], b[..., :1], b[..., 1:]
    return np.concatenate((aw * bw - np.sum(av * bv, axis=-1, keepdims=True),
                           aw * bv + bw * av + np.cross(av, bv)), axis=-1)


def slerp(a, b, fraction):
    dot = np.sum(a * b, axis=-1, keepdims=True)
    b = np.where(dot < 0, -b, b)
    dot = np.clip(np.abs(dot), 0, 1)
    angle = np.arccos(dot)
    sine = np.sin(angle)
    safe = np.where(sine < 1e-8, 1, sine)
    out = np.where(dot > .9995, a + fraction * (b - a),
                   np.sin((1 - fraction) * angle) / safe * a +
                   np.sin(fraction * angle) / safe * b)
    return out / np.linalg.norm(out, axis=-1, keepdims=True)


class MotionPlayer:
    def __init__(self, sequence, skeleton, skin=None, *, yaw=0., translation=(0., 0., 0.)):
        self.sequence, self.skeleton, self.skin = sequence.validate(), skeleton, skin
        if sequence.n_envs < 1 or sequence.n_frames < 2:
            raise ValueError('Motion requires at least one environment and two frames')
        if skeleton.joint_names != list(BODY_JOINT_NAMES) or skeleton.parents != list(BODY_JOINT_PARENTS):
            raise ValueError('Skeleton does not match schema v1 joint names/parents')
        if sequence.gender != skeleton.gender:
            raise ValueError('Motion gender and body template disagree')
        if not np.allclose(sequence.betas, 0):
            raise ValueError('The supplied template is betas=0; shaped motion needs its own validated template')
        if sequence.valid is not None and not sequence.valid.all():
            raise ValueError('Padded/invalid frames must be trimmed before playback')
        for array in (skeleton.rest_joints, skeleton.rest_offsets):
            if array.shape != (22, 3) or not np.isfinite(array).all():
                raise ValueError('Invalid skeleton rest geometry')
        expected = skeleton.rest_joints[1:] - skeleton.rest_joints[np.array(skeleton.parents[1:])]
        if not np.allclose(expected, skeleton.rest_offsets[1:], atol=1e-6):
            raise ValueError('Skeleton rest offsets disagree with rest joints')
        if skin is not None:
            if not np.allclose(skin.rest_joints, skeleton.rest_joints, atol=1e-6):
                raise ValueError('Skin and skeleton must come from the same body model')
            if not np.isfinite(skin.rest_verts).all() or not np.isfinite(skin.weight_val).all():
                raise ValueError('Non-finite skin data')
            if (skin.weight_idx < 0).any() or (skin.weight_idx >= 22).any() or (skin.weight_val < 0).any():
                raise ValueError('Invalid skin weights')
            if not np.allclose(skin.weight_val.sum(-1), 1, atol=1e-4):
                raise ValueError('Skin weights must sum to one')
        self.yaw = axis_angle_to_quat(np.array([0., 0., yaw]))
        self.rotation = quats_to_R(self.yaw[None])[0]
        self.translation = np.asarray(translation, dtype=float)
        if self.translation.shape != (3,) or not np.isfinite(self.translation).all():
            raise ValueError('Invalid scene translation')
        self.local_quats = axis_angle_to_quat(np.concatenate(
            (sequence.global_orient[:, :, None, :], sequence.body_pose), axis=2))
        self.body_indices = []
        for bone in skeleton.bones:
            self.body_indices.append(skeleton.joint_names.index(bone['body']))
            if not np.isfinite([*bone['from'], *bone['to'], bone['radius']]).all() or bone['radius'] <= 0:
                raise ValueError('Invalid capsule geometry')
        self.starts = np.array([b['from'] for b in skeleton.bones])
        self.ends = np.array([b['to'] for b in skeleton.bones])
        self.radii = np.array([b['radius'] for b in skeleton.bones])

    @classmethod
    def load(cls, motion, skeleton, skin=None, **kwargs):
        return cls(MotionSequence.load(motion), HumanSkeleton.from_template(skeleton),
                   HumanSkin.from_asset(skin) if skin else None, **kwargs)

    @property
    def end_time(self):
        return (self.sequence.n_frames - 1) / self.sequence.fps

    def pose(self, time):
        if not np.isfinite(time):
            raise ValueError('Time must be finite')
        frame = np.clip(time * self.sequence.fps, 0, self.sequence.n_frames - 1)
        lo, hi = int(np.floor(frame)), min(int(np.floor(frame)) + 1, self.sequence.n_frames - 1)
        u = frame - lo
        local = slerp(self.local_quats[:, lo], self.local_quats[:, hi], u)
        pos = np.zeros((self.sequence.n_envs, 22, 3))
        quat = np.zeros((self.sequence.n_envs, 22, 4))
        # SMPL-X rotates around its rest pelvis; do not rotate that offset twice.
        pos[:, 0] = (1-u)*self.sequence.transl[:, lo] + u*self.sequence.transl[:, hi] + self.skeleton.rest_pelvis
        quat[:, 0] = local[:, 0]
        for j in range(1, 22):
            parent = self.skeleton.parents[j]
            rotation = quats_to_R(quat[:, parent])
            pos[:, j] = pos[:, parent] + np.einsum('bij,j->bi', rotation, self.skeleton.rest_offsets[j])
            quat[:, j] = multiply(quat[:, parent], local[:, j])
        pos = pos @ self.rotation.T + self.translation
        quat = multiply(np.broadcast_to(self.yaw, quat.shape), quat)
        return pos, quat

    def capsules(self, time):
        pos, quat = self.pose(time)
        rotation = quats_to_R(quat[:, self.body_indices].reshape(-1, 4)).reshape(
            self.sequence.n_envs, -1, 3, 3)
        root = pos[:, self.body_indices]
        return (root + np.einsum('bnij,nj->bni', rotation, self.starts),
                root + np.einsum('bnij,nj->bni', rotation, self.ends), self.radii)

    def vertices(self, time):
        if self.skin is None:
            raise ValueError('No skin asset loaded')
        pos, quat = self.pose(time)
        return np.stack([self.skin.world_vertices(p, q) for p, q in zip(pos, quat)])


def point_clearance(point, starts, ends, radii):
    """Signed distance to capsule surfaces, not joint-center distance."""
    axes = ends - starts
    fraction = np.clip(np.sum((point-starts)*axes, axis=-1) /
                       np.maximum(np.sum(axes*axes, axis=-1), 1e-12), 0, 1)
    return np.linalg.norm(point-starts-fraction[..., None]*axes, axis=-1)-radii


class YieldGate:
    """Hold a robot target while the human occupies its goal's protected region."""
    def __init__(self, stop=.45, resume=.55, clear_time=.3):
        if not 0 < stop < resume or clear_time <= 0:
            raise ValueError('Require 0 < stop < resume and positive clear_time')
        self.stop, self.resume, self.clear_time = stop, resume, clear_time
        self.reset()

    def reset(self):
        self.paused = False
        self.clear_elapsed = 0.
        self.entries = self.resumes = 0

    def step(self, clearance, dt):
        if not np.isfinite(clearance) or not np.isfinite(dt) or dt <= 0:
            raise ValueError('Invalid gate measurement')
        if clearance < self.stop:
            if not self.paused:
                self.entries += 1
            self.paused, self.clear_elapsed = True, 0.
        elif self.paused:
            self.clear_elapsed = self.clear_elapsed + dt if clearance > self.resume else 0.
            if self.clear_elapsed + 1e-10 >= self.clear_time:
                self.paused = False
                self.resumes += 1
        return self.paused
