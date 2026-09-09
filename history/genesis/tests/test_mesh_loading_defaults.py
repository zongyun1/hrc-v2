from __future__ import annotations

from pathlib import Path

from envs import genesis_compat as compat


class _FakeMesh:
    def __init__(
        self,
        *,
        file,
        scale,
        pos,
        quat,
        convexify,
        fixed,
        collision,
        visualization,
        align=None,
        file_meshes_are_zup=None,
    ):
        kwargs = {
            "file": file,
            "scale": scale,
            "pos": pos,
            "quat": quat,
            "convexify": convexify,
            "fixed": fixed,
            "collision": collision,
            "visualization": visualization,
            "align": align,
            "file_meshes_are_zup": file_meshes_are_zup,
        }
        self.kwargs = kwargs


class _FakeGenesis:
    class morphs:
        Mesh = _FakeMesh


def test_mesh_kwargs_keep_glb_z_up_by_default(monkeypatch, tmp_path: Path) -> None:
    mesh_path = tmp_path / "cup.glb"
    monkeypatch.setattr(compat, "get_genesis", lambda: _FakeGenesis)

    kwargs = compat.mesh_frame_kwargs(mesh_path, align=False)

    assert kwargs["align"] is False
    assert kwargs["file_meshes_are_zup"] is True


def test_mesh_kwargs_keep_obj_z_up_by_default(monkeypatch, tmp_path: Path) -> None:
    mesh_path = tmp_path / "table.obj"
    monkeypatch.setattr(compat, "get_genesis", lambda: _FakeGenesis)

    kwargs = compat.mesh_frame_kwargs(mesh_path, align=False)

    assert kwargs["file_meshes_are_zup"] is True


def test_mesh_kwargs_allow_explicit_y_up_override(monkeypatch, tmp_path: Path) -> None:
    mesh_path = tmp_path / "nonstandard.glb"
    monkeypatch.setattr(compat, "get_genesis", lambda: _FakeGenesis)

    kwargs = compat.mesh_frame_kwargs(
        mesh_path,
        align=False,
        file_meshes_are_zup=False,
    )

    assert kwargs["file_meshes_are_zup"] is False
