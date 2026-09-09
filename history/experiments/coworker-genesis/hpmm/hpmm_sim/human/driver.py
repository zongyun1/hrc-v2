"""Play a :class:`MotionSequence` on the rigid human skeleton inside Genesis.

The human is kinematic, not dynamic: every frame its full configuration is written with
``set_qpos(..., zero_velocity=True)``. Phase 0.6 established that this is what makes the
human *block* the robot without being pushed by it — mass alone does not, because gravity
is mass-independent and a sustained contact force will still carry a free body away. The
override is therefore load-bearing, not an optimisation, and must run every step.

The qpos layout is derived from :class:`~hpmm_sim.human.skeleton.HumanSkeleton`, but never
trusted: :meth:`HumanDriver.bind` re-derives it from the built entity's own joints by name
and raises if the two disagree, so a change in how Genesis orders joints surfaces as an
error rather than as a silently scrambled pose.
"""

from __future__ import annotations

import pathlib

import numpy as np

from hpmm_sim.assets import load as assets
from hpmm_sim.human.skeleton import HumanSkeleton
from hpmm_sim.human.skin import HumanSkin


class HumanDriver:
    """Owns one human entity in a Genesis scene and drives it from a motion sequence.

    Parameters
    ----------
    scene : genesis.Scene
        Must not be built yet — the entity has to be added first.
    skeleton : HumanSkeleton, optional
        Defaults to the packaged male template.
    mjcf_path : str, optional
        Where to write the generated MJCF. Genesis loads entities from files, and keeping
        the artefact on disk makes the generated body inspectable.
    skin : bool | str | HumanSkin, optional
        Add the SMPL-X body surface as a visual-only entity, deformed each frame from the
        skeleton's link poses. The capsules then stop being drawn — they remain the
        collision geometry, which is the thing that was ever physical about them.
        ``True`` (the default) loads the packaged asset; pass a path to use a different
        body model's, or ``False`` to render the capsules instead. A video of a person
        should show a person, hence the default.

    Notes
    -----
    Both the skeleton and the skin come from files distilled out of a body model, which
    are not committed. To drive a different model — another gender, another SMPL variant —
    regenerate them with ``tools/extract_smplx_skeleton.py`` and pass the results here::

        HumanDriver(scene,
                    skeleton=HumanSkeleton.from_template("hpmm/assets/smplx_skeleton_female.json"),
                    skin="hpmm/assets/smplx_skin_female.npz")

    The skeleton and the skin must come from the *same* model: the skin's weights index the
    skeleton's joints, and :meth:`bind` checks the vertex count but cannot notice a
    mismatched body shape.
    """

    def __init__(self, scene, skeleton: HumanSkeleton | None = None,
                 mjcf_path: str = "exports/human_smplx.xml",
                 skin: "bool | str | HumanSkin" = True, **morph_kwargs):
        self.skeleton = skeleton or HumanSkeleton.from_template()
        self.mjcf_path = str(self.skeleton.write_mjcf(mjcf_path))
        if isinstance(skin, HumanSkin):
            self.skin = skin
        elif skin is True:
            self.skin = HumanSkin.from_asset()
        elif skin:
            self.skin = HumanSkin.from_asset(skin)
        else:
            self.skin = None
        # Capsules are already convex, so decomposition has nothing to do; skipping it
        # keeps the human out of the multi-minute coacd path the kitchen meshes take.
        self.scene = scene
        # With a skin on, the capsules are collision-only. `visualization=False` is the
        # honest way to hide them: they keep doing their job, they just stop being drawn.
        if self.skin is not None:
            morph_kwargs.setdefault("visualization", False)
        self.entity = scene.add_entity(
            assets.mjcf(self.mjcf_path, convexify=False, **morph_kwargs)
        )
        self.skin_entity = None
        if self.skin is not None:
            import genesis as gs

            obj_path = self.skin.write_obj(str(pathlib.Path(self.mjcf_path).with_name("human_skin.obj")))
            self.skin_entity = scene.add_entity(
                gs.morphs.Mesh(
                    file=str(obj_path), fixed=True, collision=False, visualization=True,
                    # Decimation would change the vertex count that `set_vverts` writes to,
                    # and convexification is meaningless for a surface we drive directly.
                    decimate=False, convexify=False, enable_custom_vverts=True,
                ),
                surface=gs.surfaces.Default(color=(0.82, 0.66, 0.56)),
            )
        self._qs_idx: np.ndarray | None = None
        self._n_envs = 1
        # The schema is batch-first everywhere, but a Genesis scene built with n_envs=0
        # has no batch axis at all. Squeeze it away at this boundary and nowhere else.
        self._batched = False

    # -- setup -----------------------------------------------------------------

    def bind(self) -> "HumanDriver":
        """Resolve the qpos layout against the built entity. Call after ``scene.build()``."""
        expected = [self.skeleton.prefix + "root"] + [
            self.skeleton.body_name(j) for j in self.skeleton.dfs_order()[1:]
        ]
        actual = [j.name for j in self.entity.joints if j.n_qs > 0]
        if actual != expected:
            raise RuntimeError(
                "human joint order from Genesis does not match the skeleton's DFS order.\n"
                f"  expected: {expected}\n  actual:   {actual}"
            )
        if self.entity.n_qs != self.skeleton.n_qs:
            raise RuntimeError(
                f"human n_qs mismatch: entity {self.entity.n_qs} vs skeleton {self.skeleton.n_qs}"
            )
        # Local q indices, concatenated in the order `skeleton.qpos` emits them.
        self._qs_idx = np.concatenate([
            j.qs_idx_local for j in self.entity.joints if j.n_qs > 0
        ]).astype(np.int32)
        self._batched = self.scene.n_envs > 0
        self._n_envs = max(1, int(self.scene.n_envs))

        if self.skin_entity is not None:
            if self.skin_entity.n_vverts != self.skin.n_verts:
                raise RuntimeError(
                    f"skin mesh arrived with {self.skin_entity.n_vverts} vertices but the "
                    f"asset has {self.skin.n_verts}; something decimated it, and the "
                    "per-frame vertex write would be meaningless"
                )
            # Genesis reports links in the MJCF's depth-first order; the skin weights are
            # indexed by SMPL-X joint. Resolve the permutation once, by name.
            joint_of = {self.skeleton.body_name(j): j for j in range(self.skeleton.n_joints)}
            link_to_joint = np.array([joint_of[link.name] for link in self.entity.links])
            self._link_of_joint = np.argsort(link_to_joint)
        return self

    # -- driving ---------------------------------------------------------------

    def qpos_for(self, seq, frame: int) -> np.ndarray:
        """``qpos`` for one frame of a motion sequence, shaped ``(B, n_qs)``."""
        f = min(frame, seq.n_frames - 1)
        return self.skeleton.qpos(
            seq.transl[:, f], seq.global_orient[:, f], seq.body_pose[:, f]
        )

    @staticmethod
    def frame_at(seq, t: float) -> int:
        """Motion frame index at simulated time ``t`` (seconds), clamped to the sequence.

        The motion has its own sampling rate and the solver has its own ``dt``; they are
        not the same number and must not be assumed to be. Indexing by frame *per sim
        step* silently plays the human at ``fps * dt`` times real speed — at the usual
        30 fps motion and dt=0.01 that is 3.3x too fast, and every contact force and
        approach-speed metric downstream of it is then wrong.
        """
        return int(min(max(t, 0.0) * seq.fps, seq.n_frames - 1))

    def set_time(self, seq, t: float) -> int:
        """Drive the human to simulated time ``t``. Returns the frame it landed on."""
        frame = self.frame_at(seq, t)
        self.set_frame(seq, frame)
        return frame

    def set_frame(self, seq, frame: int) -> None:
        """Snap the human to one frame. Zeroes velocity so contacts cannot accumulate."""
        if self._qs_idx is None:
            raise RuntimeError("call bind() after scene.build() before driving the human")
        qpos = self.qpos_for(seq, frame)
        if not self._batched:
            qpos = qpos[0]
        elif qpos.shape[0] == 1 and self._n_envs > 1:
            qpos = np.repeat(qpos, self._n_envs, axis=0)
        self.entity.set_qpos(qpos, qs_idx_local=self._qs_idx, zero_velocity=True)
        self.update_skin()

    def update_skin(self) -> None:
        """Re-deform the visual surface onto the skeleton's current pose.

        Called by :meth:`set_frame`; `set_qpos` has already run forward kinematics, so the
        link poses read here are the ones the capsules are actually in.
        """
        if self.skin_entity is None:
            return
        pos = np.asarray(self.entity.get_links_pos().cpu()).reshape(-1, 3)[self._link_of_joint]
        quat = np.asarray(self.entity.get_links_quat().cpu()).reshape(-1, 4)[self._link_of_joint]
        self.skin_entity.set_vverts(self.skin.world_vertices(pos, quat).astype(np.float32))

    # -- introspection ---------------------------------------------------------

    def joint_positions(self) -> np.ndarray:
        """World positions of the human's link origins, ``(B, n_links, 3)``.

        These are the SMPL-X body joints (each link's origin *is* its joint), so this is
        directly comparable to the ``joints`` cache in the motion sequence — which is how
        :mod:`tools.verify_human_bridge` checks that the pose transferred intact.
        """
        return np.asarray(self.entity.get_links_pos().cpu()).reshape(self._n_envs, -1, 3)

    @property
    def link_names(self) -> list[str]:
        return [link.name for link in self.entity.links]

    def total_mass(self) -> float:
        masses = np.asarray(self.entity.get_links_mass().cpu())
        return float(masses.reshape(self._n_envs, -1)[0].sum())


__all__ = ["HumanDriver"]
