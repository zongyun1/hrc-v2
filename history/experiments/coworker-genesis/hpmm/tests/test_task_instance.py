"""Unit tests for the RoboCasa task-instance shim. No Genesis needed."""

from __future__ import annotations

import json

import numpy as np
import pytest

from hpmm_sim.shim.sampler import TaskInstance

RAW = {
    "task": "PickPlaceCounterToCabinet",
    "robot": "PandaOmron",
    "layout": 1,
    "style": 1,
    "seed": 0,
    "mjcf": "scene.xml",
    "language": "Pick the sugar cube from the counter and place it in the cabinet.",
    "n_qpos": 80,                       # extra keys in the file must not break loading
    "joints": {
        "obj_joint0": [0.421, -0.323, 0.934, 1.0, 0.0, 0.0, -0.031],
        "cab_1_main_group_hingedoor_joint": [0.0],
    },
    "free_bodies": ["obj_main"],
    "objects": {"obj": {"root_body": "obj_main",
                        "pos": [0.421, -0.323, 0.934],
                        "quat": [1.0, 0.0, 0.0, -0.031]}},
    "note": "ignored",
}


@pytest.fixture
def state_file(tmp_path):
    path = tmp_path / "task.json"
    path.write_text(json.dumps(RAW))
    return path


def test_load_ignores_unknown_keys(state_file):
    instance = TaskInstance.load(state_file)
    assert instance.task == "PickPlaceCounterToCabinet"
    assert instance.language.startswith("Pick the sugar cube")
    assert instance.free_bodies == ["obj_main"]


def test_mjcf_path_resolves_next_to_the_state_file(state_file):
    """The pair travels together; a bare filename must not resolve against the cwd."""
    instance = TaskInstance.load(state_file)
    assert instance.mjcf_path == state_file.parent / "scene.xml"


def test_object_pose_round_trip(state_file):
    pos, quat = TaskInstance.load(state_file).object_pose("obj")
    assert np.allclose(pos, RAW["objects"]["obj"]["pos"])
    assert np.allclose(quat, RAW["objects"]["obj"]["quat"])


def test_free_joint_width_is_seven(state_file):
    """A free joint is xyz + wxyz in both MuJoCo and Genesis; anything else is a bug."""
    instance = TaskInstance.load(state_file)
    for body in instance.free_bodies:
        matching = [v for k, v in instance.joints.items() if k.startswith(body.split("_main")[0])]
        assert matching and all(len(v) == 7 for v in matching)


def test_apply_refuses_to_half_reproduce_an_episode(state_file):
    """Strict mode must fail loudly rather than place some objects and not others."""
    class FakeJoint:
        def __init__(self, name, n_qs, start):
            self.name, self.n_qs = name, n_qs
            self.qs_idx_local = list(range(start, start + n_qs))

    class FakeEntity:
        joints = [FakeJoint("obj_joint0", 7, 0)]   # the door joint is absent

    with pytest.raises(RuntimeError, match="absent from the Genesis entity"):
        TaskInstance.load(state_file).apply(FakeEntity(), strict=True)


def test_mjcf_loader_pins_align_false():
    """Genesis's ``align`` default (None) re-frames free bodies onto their COM.

    That silently changes what a free joint's qpos position means, displacing every
    single-body object by its own ``body_ipos`` — 100 mm on a deliberately offset test
    model, 0.01-12.45 mm on a real RoboCasa episode. Small enough to pass for noise,
    large enough to aim an IK grasp at the wrong place.
    """
    from hpmm_sim.assets.load import MJCF_DEFAULTS

    assert MJCF_DEFAULTS["align"] is False


def test_sanitize_materials_clamps_only_out_of_range_values():
    """RoboCasa ships `shininess="-1"`, which makes Genesis compute a complex roughness.

    Genesis converts glossiness to roughness as ``(2 / (glossiness + 2)) ** 0.25`` with
    ``glossiness = shininess * 128``. A negative shininess is then a fourth root of a
    negative number — a complex number in Python — and the scene fails to build with a
    texture-validation message that names no material at all.
    """
    import xml.etree.ElementTree as ET

    from hpmm_sim.assets.convert import sanitize_materials

    root = ET.fromstring(
        '<mujoco><asset>'
        '<material name="stool" shininess="-1" specular="3.0" rgba="1 1 1 1"/>'
        '<material name="fine" shininess="0.4" specular="0.5"/>'
        '</asset></mujoco>'
    )
    fixed = sanitize_materials(root)
    assert fixed == {"shininess": 1, "specular": 1}
    stool, fine = list(root.iter("material"))
    assert float(stool.get("shininess")) == 0.0
    assert float(stool.get("specular")) == 1.0
    # An in-range material must pass through untouched, attribute text included.
    assert fine.get("shininess") == "0.4" and fine.get("specular") == "0.5"


def test_glossiness_conversion_is_what_actually_breaks():
    """Pin the arithmetic, so the reason for clamping survives a future refactor."""
    shininess = -1.0
    glossiness = shininess * 128.0
    roughness = (2 / (glossiness + 2)) ** 0.25
    assert isinstance(roughness, complex)
    assert abs(roughness.real - 0.25098) < 1e-4 and abs(roughness.imag - 0.25098) < 1e-4
