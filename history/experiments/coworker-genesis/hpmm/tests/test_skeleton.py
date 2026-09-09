"""Unit tests for the SMPL-X -> MJCF rigid skeleton. No Genesis, no GPU, no model file."""

from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np
import pytest

from hpmm_motion.io.schema import BODY_JOINT_NAMES, BODY_JOINT_PARENTS, N_BODY_JOINTS
from hpmm_sim.human.skeleton import DEFAULT_TEMPLATE, HumanSkeleton, TARGET_MASS, axis_angle_to_quat
from hpmm_sim.human.skin import DEFAULT_ASSET

#: The body-model assets are derived from SMPL-X and deliberately not committed (see
#: README). Everything that needs them is skipped rather than failed on a fresh clone;
#: the pure-maths tests below still run.
needs_body_model = pytest.mark.skipif(
    not DEFAULT_TEMPLATE.exists() or not DEFAULT_ASSET.exists(),
    reason="body-model assets absent; run tools/extract_smplx_skeleton.py",
)


@pytest.fixture(scope="module")
def skeleton():
    if not DEFAULT_TEMPLATE.exists():
        pytest.skip("body-model assets absent; run tools/extract_smplx_skeleton.py")
    return HumanSkeleton.from_template()


def test_template_matches_the_frozen_schema(skeleton):
    assert skeleton.joint_names == list(BODY_JOINT_NAMES)
    assert skeleton.parents == list(BODY_JOINT_PARENTS)
    assert skeleton.n_joints == N_BODY_JOINTS + 1


def test_axis_angle_to_quat_is_a_unit_quaternion():
    rng = np.random.default_rng(0)
    aa = rng.normal(size=(64, 3)) * 2.0
    q = axis_angle_to_quat(aa)
    assert q.shape == (64, 4)
    assert np.allclose(np.linalg.norm(q, axis=-1), 1.0)


def test_axis_angle_to_quat_handles_zero_rotation():
    """The small-angle branch must not divide by the angle it is guarding against."""
    q = axis_angle_to_quat(np.zeros((3, 3)))
    assert np.allclose(q, [1.0, 0.0, 0.0, 0.0])
    tiny = axis_angle_to_quat(np.full((1, 3), 1e-12))
    assert np.isfinite(tiny).all() and np.isclose(np.linalg.norm(tiny), 1.0)


def test_axis_angle_to_quat_known_value():
    """90 degrees about +z."""
    q = axis_angle_to_quat(np.array([0.0, 0.0, np.pi / 2]))
    assert np.allclose(q, [np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)])


def test_qpos_layout(skeleton):
    assert skeleton.n_qs == 7 + 4 * N_BODY_JOINTS
    assert skeleton.n_dofs == 6 + 3 * N_BODY_JOINTS
    q = skeleton.qpos(np.zeros((5, 3)), np.zeros((5, 3)), np.zeros((5, N_BODY_JOINTS, 3)))
    assert q.shape == (5, skeleton.n_qs) and q.dtype == np.float32
    # Rest pose: root at transl + rest pelvis, every rotation identity.
    assert np.allclose(q[:, :3], skeleton.rest_pelvis)
    quats = q[:, 3:].reshape(5, -1, 4)
    assert np.allclose(quats[..., 0], 1.0) and np.allclose(quats[..., 1:], 0.0)


def test_qpos_places_the_root_at_transl_plus_rest_pelvis(skeleton):
    transl = np.array([[1.0, -2.0, 0.5]])
    q = skeleton.qpos(transl, np.zeros((1, 3)), np.zeros((1, N_BODY_JOINTS, 3)))
    assert np.allclose(q[0, :3], transl[0] + skeleton.rest_pelvis)


def test_qpos_rejects_wrong_pose_shape(skeleton):
    with pytest.raises(ValueError, match="body_pose"):
        skeleton.qpos(np.zeros((1, 3)), np.zeros((1, 3)), np.zeros((1, 5, 3)))


def test_dfs_order_is_a_valid_traversal(skeleton):
    order = skeleton.dfs_order()
    assert sorted(order) == list(range(skeleton.n_joints))
    seen = set()
    for j in order:
        parent = skeleton.parents[j]
        assert parent < 0 or parent in seen, f"{skeleton.joint_names[j]} precedes its parent"
        seen.add(j)


def test_mjcf_structure(skeleton):
    root = ET.fromstring(skeleton.to_mjcf())
    bodies = list(root.iter("body"))
    assert len(bodies) == skeleton.n_joints
    assert len(list(root.iter("freejoint"))) == 1
    balls = [j for j in root.iter("joint") if j.get("type") == "ball"]
    assert len(balls) == N_BODY_JOINTS
    geoms = list(root.iter("geom"))
    capsules = [g for g in geoms if g.get("type") == "capsule"]
    assert len(capsules) == len(skeleton.bones)
    # Every bone must have a positive radius and a non-degenerate axis.
    for g in capsules:
        assert float(g.get("size")) > 0
        p = np.array([float(v) for v in g.get("fromto").split()])
        assert np.linalg.norm(p[3:] - p[:3]) > 1e-3


def test_mjcf_body_offsets_reproduce_the_rest_skeleton(skeleton):
    """Nesting the parent-relative offsets must give back the SMPL-X rest skeleton.

    The root body sits at the XML origin rather than at ``rest_joints[0]``: its world
    placement comes from the free joint, whose position ``qpos`` sets to
    ``transl + rest_pelvis``. So the accumulated offsets reproduce the rest joints
    *relative to the pelvis*, and adding the free joint's value back recovers SMPL-X's
    own world joints — which is exactly what ``qpos`` does.
    """
    root = ET.fromstring(skeleton.to_mjcf())
    pos_by_name, parent_of = {}, {}

    def walk(elem, parent_name=None):
        for body in elem.findall("body"):
            name = body.get("name")
            pos_by_name[name] = np.array([float(v) for v in body.get("pos").split()])
            parent_of[name] = parent_name
            walk(body, name)

    walk(root.find("worldbody"))
    world = {}
    for j in skeleton.dfs_order():
        name = skeleton.body_name(j)
        parent = parent_of[name]
        world[name] = pos_by_name[name] + (world[parent] if parent else 0.0)
        expected = skeleton.rest_joints[j] - skeleton.rest_pelvis
        assert np.allclose(world[name], expected, atol=1e-6), name
    assert np.allclose(world[skeleton.body_name(0)], 0.0)


def test_density_hits_the_target_mass(skeleton):
    """Overlapping capsules over-count volume, so density is solved, not assumed."""
    volume = skeleton.capsule_volume()
    assert np.isclose(skeleton.density() * volume, TARGET_MASS)
    assert skeleton.density() < 985.0, "solved density should be below flesh density"


def test_motion_plays_at_its_own_rate_not_the_solver_rate():
    """A motion frame is chosen by elapsed time, never by step count.

    Indexing the sequence once per solver step plays the human at ``fps * dt`` times real
    speed. At the usual 30 fps and dt=0.01 that is 3.3x too fast, which would corrupt every
    contact force and approach-speed metric while still looking plausible on video.
    """
    from hpmm_motion.io.schema import empty
    from hpmm_sim.human.driver import HumanDriver

    seq = empty(1, 381, fps=30.0)
    assert HumanDriver.frame_at(seq, 0.0) == 0
    assert HumanDriver.frame_at(seq, 1.0) == 30
    assert HumanDriver.frame_at(seq, 0.01) == 0        # a 100 Hz step is a third of a frame
    assert HumanDriver.frame_at(seq, seq.duration) == seq.n_frames - 1
    assert HumanDriver.frame_at(seq, 1e6) == seq.n_frames - 1, "must clamp, not wrap"
    assert HumanDriver.frame_at(seq, -5.0) == 0
    # Stepping at 100 Hz through the whole sequence must visit every frame exactly once
    # in order, and end on the last one.
    visited = [HumanDriver.frame_at(seq, i * 0.01) for i in range(int(seq.duration / 0.01))]
    assert visited == sorted(visited)
    assert set(visited) == set(range(seq.n_frames))


def test_self_collision_is_filtered_but_world_collision_is_not(skeleton):
    """Human geoms must not collide with each other, yet must still see the scene."""
    from hpmm_sim.human.skeleton import HUMAN_CONAFFINITY, HUMAN_CONTYPE

    assert not (HUMAN_CONTYPE & HUMAN_CONAFFINITY), "human geoms would self-collide"
    world_contype = world_conaffinity = 1  # MuJoCo/RoboCasa default
    assert (HUMAN_CONTYPE & world_conaffinity) or (world_contype & HUMAN_CONAFFINITY)


@needs_body_model
def test_skin_reproduces_the_rest_mesh_from_rest_link_poses(skeleton):
    """LBS from the skeleton's own rest pose must return the SMPL-X surface unchanged.

    This is the check that the binding convention is right: in the rest pose every link
    frame is a pure translation to its joint, so ``sum_j w_ij (R_j (v - J_j) + p_j)``
    collapses back to ``v``. If the offsets were taken against the wrong frame the error
    would be metres, not microns.
    """
    import numpy as np

    from hpmm_sim.human.skin import HumanSkin

    skin = HumanSkin.from_asset()
    pos = np.asarray(skeleton.rest_joints)
    quat = np.tile([1.0, 0.0, 0.0, 0.0], (skeleton.n_joints, 1))
    assert np.abs(skin.world_vertices(pos, quat) - skin.rest_verts).max() < 1e-6


@needs_body_model
def test_skin_weights_are_a_partition_of_unity():
    import numpy as np

    from hpmm_sim.human.skin import HumanSkin

    skin = HumanSkin.from_asset()
    assert np.allclose(skin.weight_val.sum(axis=1), 1.0, atol=1e-6)
    assert skin.weight_idx.min() >= 0 and skin.weight_idx.max() < len(skin.rest_joints)
    assert skin.faces.max() < skin.n_verts
