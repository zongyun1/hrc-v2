"""Reproduce a robosuite placement-sampler outcome inside Genesis.

RoboCasa decides where objects go at ``env.reset()``, by sampling, and writes the result
into MuJoCo's ``qpos``. The exported MJCF carries only each body's authored default pose,
so loading the XML alone gives a *different* episode from the one robosuite produced. To
pin an episode down, the sampled configuration has to be replayed onto the built scene.

This is the plan's ``shim/sampler.py``: not a reimplementation of RoboCasa's sampler, but a
redirection of its *output* onto Genesis entities.

Everything is keyed by **joint name**, never by index. Genesis assigns its own dof
ordering when it parses the MJCF, and matching positionally would appear to work while
quietly scrambling the scene.
"""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass

import numpy as np


@dataclass
class TaskInstance:
    """One exported RoboCasa episode: the scene file plus its post-reset configuration."""

    task: str
    robot: str
    layout: int
    style: int
    seed: int
    mjcf: str
    language: str
    joints: dict[str, list[float]]
    free_bodies: list[str]
    objects: dict[str, dict]
    #: RoboCasa's own camera mounts, when the episode came with them. Optional because a
    #: scene exported by our own task exporter does not carry them.
    cam_configs: dict[str, dict] | None = None

    @classmethod
    def load(cls, path) -> "TaskInstance":
        path = pathlib.Path(path)
        raw = json.loads(path.read_text())
        known = {f for f in cls.__dataclass_fields__}
        instance = cls(**{k: v for k, v in raw.items() if k in known})
        instance._dir = path.parent
        return instance

    @property
    def mjcf_path(self) -> pathlib.Path:
        """The scene file, resolved next to the state JSON."""
        return getattr(self, "_dir", pathlib.Path(".")) / self.mjcf

    def __repr__(self) -> str:
        return (f"TaskInstance({self.task}, L{self.layout}S{self.style}, seed={self.seed}, "
                f"{len(self.objects)} objects, {len(self.joints)} joints)")

    # -- application -----------------------------------------------------------

    def apply(self, entity, strict: bool = True) -> dict[str, int]:
        """Write the recorded configuration onto a built Genesis entity.

        Returns a report of how many joints matched, were missing from the entity, or were
        skipped because their width disagrees — a silent partial application would leave
        the scene subtly different from the episode it claims to reproduce.
        """
        by_name = {joint.name: joint for joint in entity.joints if joint.n_qs > 0}
        qs_idx, values = [], []
        report = {"matched": 0, "missing": 0, "width_mismatch": 0}
        missing, mismatched = [], []
        for name, value in self.joints.items():
            joint = by_name.get(name)
            if joint is None:
                report["missing"] += 1
                missing.append(name)
                continue
            if joint.n_qs != len(value):
                report["width_mismatch"] += 1
                mismatched.append(f"{name}: entity {joint.n_qs} vs recorded {len(value)}")
                continue
            qs_idx.extend(joint.qs_idx_local)
            values.extend(value)
            report["matched"] += 1

        if strict and (missing or mismatched):
            raise RuntimeError(
                f"cannot reproduce {self.task} faithfully — "
                f"{len(missing)} joints absent from the Genesis entity ({missing[:5]}), "
                f"{len(mismatched)} width mismatches ({mismatched[:3]})"
            )
        qpos = np.asarray(values, dtype=np.float32)
        idx = np.asarray(qs_idx, dtype=np.int32)
        if entity._solver.scene.n_envs > 0:
            qpos = np.tile(qpos, (entity._solver.scene.n_envs, 1))
        entity.set_qpos(qpos, qs_idx_local=idx, zero_velocity=True)
        return report

    def object_pose(self, key: str) -> tuple[np.ndarray, np.ndarray]:
        """Recorded world ``(pos, quat)`` of one object, for checking a reproduction."""
        obj = self.objects[key]
        return np.asarray(obj["pos"]), np.asarray(obj["quat"])


__all__ = ["TaskInstance"]
