from __future__ import annotations

import numpy as np
import transforms3d as t3d

from envs import base_task as _base_task
from envs.grasp import GraspPose
from envs.task_bases.neutral_avatar_table_work import NeutralAvatarTableWorkMixin
from envs.task_bases.place_bread_in_basket import PlaceBreadInBasket
from envs.task_bases.place_burger_fries import PlaceBurgerFries, _FRIES_SPEC
from envs.task_bases.place_food_in_skillet import _Q_UPRIGHT as SKILLET_UPRIGHT
from envs.task_bases.stack_bowls_three import StackBowlsThree
from envs.tasks.categorize_cooperative import CategorizeCooperative
from envs.tasks.stack_bowls_three_interrupt import StackBowlsThreeInterrupt
from envs.tasks.pour_water import (
    _Q_MUG_HANDLE_TOWARD_ROBOT,
    _Q_UPRIGHT as POUR_WATER_UPRIGHT,
)


def _assert_local_y_maps_to_world_z(quat) -> None:
    rotation = t3d.quaternions.quat2mat(np.asarray(quat, dtype=np.float64))
    actual = rotation @ np.array([0.0, 1.0, 0.0], dtype=np.float64)
    np.testing.assert_allclose(actual, [0.0, 0.0, 1.0], atol=1e-6)


def test_y_up_task_assets_use_authored_frame_upright_rotation() -> None:
    quats = (
        PlaceBreadInBasket.SPAWN_QUAT,
        PlaceBurgerFries._Q_TRAY,
        _FRIES_SPEC.spawn_quat,
        SKILLET_UPRIGHT,
        StackBowlsThree.BOWL_UPRIGHT_QUAT,
        CategorizeCooperative.SPAWN_UPRIGHT_QUAT,
        POUR_WATER_UPRIGHT,
        _Q_MUG_HANDLE_TOWARD_ROBOT,
    )
    for quat in quats:
        _assert_local_y_maps_to_world_z(quat)


def test_neutral_props_use_authored_frame_upright_rotation() -> None:
    task = NeutralAvatarTableWorkMixin.__new__(NeutralAvatarTableWorkMixin)
    task.config = {}
    task.avatar_init_rot = np.eye(3, dtype=np.float64)

    for name in ("bottle", "sponge", "apple", "plate", "object_1"):
        _assert_local_y_maps_to_world_z(task._neutral_avatar_prop_quat(name))


def test_neutral_mouse_keeps_upright_rotation_when_yawed() -> None:
    task = NeutralAvatarTableWorkMixin.__new__(NeutralAvatarTableWorkMixin)
    task.config = {}
    task.avatar_init_rot = np.eye(3, dtype=np.float64)
    task._neutral_avatar_job = None

    _assert_local_y_maps_to_world_z(task._neutral_typing_mouse_quat(np.pi / 3.0))


def test_interrupt_bowl_grasp_lift_uses_authored_y_up(monkeypatch) -> None:
    task = StackBowlsThreeInterrupt.__new__(StackBowlsThreeInterrupt)
    task.GRASP_Z_LIFT = 0.005
    task._bowl_scale = 0.05
    grasps = [
        GraspPose(
            "runtime",
            [1.0, 2.0, 3.0],
            [1.0, 0.0, 0.0, 0.0],
            scale_frame="runtime",
        ),
        GraspPose(
            "mesh_unit",
            [1.0, 2.0, 3.0],
            [1.0, 0.0, 0.0, 0.0],
            scale_frame="mesh_unit",
        ),
    ]
    monkeypatch.setattr(_base_task, "load_grasp_poses", lambda *args, **kwargs: grasps)

    with task._with_grasp_z_lift():
        lifted = _base_task.load_grasp_poses()

    np.testing.assert_allclose(lifted[0].pose.p, [1.0, 2.005, 3.0])
    np.testing.assert_allclose(lifted[1].pose.p, [1.0, 2.1, 3.0])
    np.testing.assert_allclose(grasps[0].pose.p, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(grasps[1].pose.p, [1.0, 2.0, 3.0])
