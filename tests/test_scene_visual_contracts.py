"""Regression tests for task-table geometry and NYX articulated visuals."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from envs.base_task import BaseTask
from envs.camera import Camera
from envs.genesis_compat import ensure_nyx_urdf
from envs.utils import load_config


def test_rectangular_table_uses_one_geometry_contract():
    task = BaseTask.__new__(BaseTask)
    task.scene = object()
    task.table_offset = np.array([0.0, 0.0])
    task._try_create_table_variant = lambda *args, **kwargs: False

    created = []

    def fake_create_primitive(scene, kind, pose, *, size, **kwargs):
        entity = object()
        created.append((kind, np.asarray(pose.p), size, kwargs, entity))
        return entity

    with patch("envs.base_task.create_primitive", fake_create_primitive):
        table = task._create_rectangular_table(
            table_height=0.74,
            half_size=(0.53, 0.685),
            center_xy=(0.0, 0.185),
        )

    assert table is created[0][4]
    assert created[0][0] == "box"
    assert created[0][2]["half_size"] == (0.53, 0.685, 0.025)
    assert np.allclose(task._table_center_xy, [0.0, 0.185])
    assert np.allclose(task._table_half_size_xy, [0.53, 0.685])
    assert np.isclose(task.TABLE_TOP_Z, 0.765)
    assert len(created) == 5  # one top and four legs


def test_rectangular_table_passes_exact_dimensions_to_variant():
    task = BaseTask.__new__(BaseTask)
    task.table_offset = np.array([0.0, 0.0])
    task.table = object()
    captured = {}

    def fake_variant(height, *, half_size, center_xy):
        captured.update(height=height, half_size=half_size, center_xy=center_xy)
        return True

    task._try_create_table_variant = fake_variant
    result = task._create_rectangular_table(
        table_height=0.74,
        half_size=(0.60, 0.575),
        center_xy=(0.0, -0.225),
    )

    assert result is task.table
    assert captured == {
        "height": 0.74,
        "half_size": (0.60, 0.575),
        "center_xy": (0.0, -0.225),
    }


def test_demo_video_uses_tuned_left_right_pair_but_vla_does_not():
    task = BaseTask.__new__(BaseTask)
    task.recording_camera_pos = [9.0, 8.0, 7.0]
    task.recording_camera_lookat = [0.0, 0.0, 0.0]
    task.side_camera_pos = [6.0, 5.0, 4.0]
    task.side_camera_lookat = [1.0, 1.0, 1.0]
    task.config = load_config()

    left, right = task._resolve_video_camera_specs()
    assert task._demo_video_camera_pair_enabled()
    assert np.allclose(left["position"], [-1.2727922061, -1.2727922061, 1.1])
    assert np.allclose(right["position"], [1.2727922061, -1.2727922061, 1.1])
    assert left["fov"] == right["fov"] == 40.0

    task.config["record_stride"] = 50
    assert not task._demo_video_camera_pair_enabled()
    recording, side = task._resolve_video_camera_specs()
    assert recording["position"] == [9.0, 8.0, 7.0]
    assert side["position"] == [6.0, 5.0, 4.0]

    task.config.pop("record_stride")
    task.config["vla_recording"] = {"enabled": True}
    assert not task._demo_video_camera_pair_enabled()
    recording, side = task._resolve_video_camera_specs()
    assert recording["position"] == [9.0, 8.0, 7.0]
    assert side["position"] == [6.0, 5.0, 4.0]


def test_camera_manager_keeps_demo_pair_pose_and_fov():
    class FakeCamera:
        pass

    class FakeScene:
        def __init__(self):
            self.specs = []

        def add_camera(self, **kwargs):
            self.specs.append(kwargs)
            return FakeCamera()

    scene = FakeScene()
    manager = Camera(
        scene,
        static_cameras=[],
        recording_pos=[-1.2727922061, -1.2727922061, 1.1],
        recording_lookat=[0.0, 0.0, 1.1],
        side_pos=[1.2727922061, -1.2727922061, 1.1],
        side_lookat=[0.0, 0.0, 1.1],
        recording_fov=40.0,
        side_fov=40.0,
    )

    assert set(manager.get_camera_names()) == {
        "left_wrist", "right_wrist", "recording", "side"
    }
    recording = manager._cameras["recording"]
    side = manager._cameras["side"]
    assert recording[3] == side[3] == 40.0
    assert np.allclose(recording[4], [-1.2727922061, -1.2727922061, 1.1])
    assert np.allclose(side[4], [1.2727922061, -1.2727922061, 1.1])

def test_nyx_urdf_premerge_keeps_all_visual_geometry():
    with TemporaryDirectory() as tmp:
        _check_nyx_urdf_premerge(Path(tmp))


def _check_nyx_urdf_premerge(tmp_path: Path):
    triangle = "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n"
    for name in ("root_a.obj", "body_a.obj", "body_b.obj", "door.obj"):
        (tmp_path / name).write_text(triangle)

    urdf = tmp_path / "fixture.urdf"
    urdf.write_text(
        """<robot name="fixture">
  <link name="root"><visual><geometry><mesh filename="root_a.obj"/></geometry></visual></link>
  <link name="body">
    <visual><origin xyz="1 0 0"/><geometry><mesh filename="body_a.obj"/></geometry></visual>
    <visual><origin xyz="0 1 0"/><geometry><mesh filename="body_b.obj"/></geometry></visual>
  </link>
  <link name="door"><visual><geometry><mesh filename="door.obj"/></geometry></visual></link>
  <joint name="root_body" type="fixed"><parent link="root"/><child link="body"/></joint>
  <joint name="hinge" type="revolute">
    <parent link="body"/><child link="door"/><axis xyz="0 0 1"/>
    <limit lower="0" upper="1" effort="1" velocity="1"/>
  </joint>
</robot>"""
    )

    rewritten = ensure_nyx_urdf(urdf, scale=0.2)
    root = ET.parse(rewritten).getroot()

    assert all(joint.get("type") != "fixed" for joint in root.findall("joint"))
    assert len(root.findall("link")) == 2
    assert all(len(link.findall("visual")) == 1 for link in root.findall("link"))
    visual_meshes = root.findall("./link/visual/geometry/mesh")
    assert visual_meshes
    # Multi-visual links are consolidated to GLB; a link that already had one
    # source visual can keep that original textured mesh.
    assert any(mesh.get("filename").endswith(".glb") for mesh in visual_meshes)
    for mesh in root.iter("mesh"):
        assert (rewritten.parent / mesh.get("filename")).is_file()
        assert mesh.get("scale") == "0.2 0.2 0.2"


if __name__ == "__main__":
    test_rectangular_table_uses_one_geometry_contract()
    test_rectangular_table_passes_exact_dimensions_to_variant()
    test_demo_video_uses_tuned_left_right_pair_but_vla_does_not()
    test_camera_manager_keeps_demo_pair_pose_and_fov()
    test_nyx_urdf_premerge_keeps_all_visual_geometry()
    print("scene visual contract tests passed")
