import numpy as np
import transforms3d as t3d

from envs.base_task import BaseTask
from envs.genesis_compat import (
    control_dof_force_toward_position_compat,
    hold_dof_position_compat,
)
from envs.manipulation import PickSpec, TopDownPickPlaceMixin
from envs.object_catalog import CATALOG
from envs.tasks.put_object_cabinet_interrupt import PutObjectCabinetInterrupt


def test_episode_rng_does_not_advance_task_rng():
    task = BaseTask.__new__(BaseTask)
    task._episode_seed = 17
    task._episode_rngs = {}

    np.random.seed(17)
    expected = np.random.uniform(size=4)
    np.random.seed(17)
    task.episode_rng("nyx_env_texture").choice(["a", "b", "c"])
    actual = np.random.uniform(size=4)

    np.testing.assert_allclose(actual, expected)


def test_episode_rng_streams_are_repeatable_and_independent():
    def sample(stream):
        task = BaseTask.__new__(BaseTask)
        task._episode_seed = 23
        task._episode_rngs = {}
        return task.episode_rng(stream).integers(0, 2**31, size=8)

    np.testing.assert_array_equal(sample("avatar_skin"), sample("avatar_skin"))
    assert not np.array_equal(sample("avatar_skin"), sample("nyx_env_texture"))


def test_support_release_height_accounts_for_object_and_grasp_offset():
    release_z = TopDownPickPlaceMixin.support_release_tcp_z(
        support_z=0.99,
        object_half_height=0.05,
        held_center_offset_z=-0.02,
        clearance=0.01,
    )
    assert release_z == 1.07


def test_live_aabb_center_targets_physical_bounds():
    class _Entity:
        def get_AABB(self):
            return np.array([[-0.2, -0.4, 0.7], [0.0, -0.2, 0.9]])

    np.testing.assert_allclose(
        TopDownPickPlaceMixin.live_entity_aabb_center(_Entity()),
        [-0.1, -0.3, 0.8],
    )


def test_topdown_catalog_options_prefer_controller_specific_hints():
    opts = TopDownPickPlaceMixin.topdown_pick_options({
        "close_value": 0.3,
        "grasp_z_offset": 0.02,
        "topdown_close_value": 0.55,
        "topdown_grasp_z_offset": 0.0,
        "topdown_approach_xy_offset": (0.0, 0.0),
        "topdown_preopen_steps": 80,
    })
    assert opts == {
        "close_value": 0.55,
        "grasp_z_offset": 0.0,
        "approach_xy_offset": (0.0, 0.0),
        "preopen_steps": 80,
    }

    jam_opts = TopDownPickPlaceMixin.topdown_pick_options(
        CATALOG["031_jam-jar"].as_spec(),
    )
    assert jam_opts["approach_xy_offset"] == (0.0, 0.0)
    assert jam_opts["grasp_z_offset"] == 0.0
    assert jam_opts["close_value"] == 0.55


def test_object_aware_drawer_target_uses_payload_and_joint_limit():
    class _LimitedFixture:
        def get_dofs_limit(self):
            return np.array([0.0]), np.array([0.132])

    target = TopDownPickPlaceMixin.object_aware_prismatic_open_target(
        _LimitedFixture(), 0, payload_extent=0.097,
    )
    assert np.isclose(target, 0.127)
    capped = TopDownPickPlaceMixin.object_aware_prismatic_open_target(
        _LimitedFixture(), 0, payload_extent=0.200,
    )
    assert np.isclose(capped, 0.128)


def test_contained_center_keeps_full_object_behind_edge():
    center = TopDownPickPlaceMixin.contained_center_from_edge(
        edge=-0.44,
        object_extent=0.070,
        inward_sign=-1.0,
        clearance=0.010,
    )
    assert np.isclose(center, -0.485)
    assert center + 0.035 <= (-0.44 - 0.010) + 1e-12

    held_center = TopDownPickPlaceMixin.contained_center_from_edge(
        edge=-0.44,
        object_extent=0.070,
        inward_sign=-1.0,
        clearance=0.0,
        held_inward_offset=0.005,
    )
    assert np.isclose(held_center, -0.470)

    deeper_center = TopDownPickPlaceMixin.contained_center_from_edge(
        edge=-0.44,
        object_extent=0.070,
        inward_sign=-1.0,
        clearance=0.050,
        held_inward_offset=0.005,
    )
    assert np.isclose(deeper_center, -0.520)


def test_relative_parallel_jaw_yaw_tracks_object_with_half_turn_symmetry():
    reference = t3d.euler.euler2quat(0.0, 0.0, 0.0)
    current = t3d.euler.euler2quat(0.0, 0.0, np.deg2rad(170.0))
    yaw = TopDownPickPlaceMixin.relative_parallel_jaw_yaw(
        current, reference,
    )
    assert np.isclose(np.rad2deg(yaw), -10.0)


def test_pick_spec_grasp_offset_gate_is_opt_in():
    spec = PickSpec(get_center=lambda: np.zeros(3), radius=0.03)
    assert spec.max_grasp_offset is None
    assert not hasattr(spec, "max_lift_attempts")


def test_cabinet_standard_inspect_motions_attach_target():
    standard_entries = PutObjectCabinetInterrupt.INSPECT_POOL[:3]
    assert [entry["name"] for entry in standard_entries] == [
        "Inspect2", "Inspect1", "Inspect4",
    ]
    assert all(entry["attach_object"] for entry in standard_entries)
    assert not any(
        entry["attach_object"]
        for entry in PutObjectCabinetInterrupt.INSPECT_POOL[3:]
    )


class _FixtureEntity:
    def __init__(self):
        self.qpos = np.array([0.0, 0.12, 0.0])
        self.kp = np.ones(3)
        self.kv = np.ones(3)
        self.force_range = np.column_stack((-np.ones(3), np.ones(3)))
        self.target = None
        self.velocity = np.array([0.0, 0.01, 0.0])
        self.force = None

    def get_qpos(self):
        return self.qpos.copy()

    def get_dofs_velocity(self):
        return self.velocity.copy()

    def get_dofs_kp(self):
        return self.kp.copy()

    def get_dofs_kv(self):
        return self.kv.copy()

    def set_dofs_kp(self, value):
        self.kp = np.asarray(value).copy()

    def set_dofs_kv(self, value):
        self.kv = np.asarray(value).copy()

    def get_dofs_force_range(self):
        return self.force_range.copy()

    def set_dofs_force_range(self, lower, upper):
        self.force_range = np.column_stack((lower, upper))

    def control_dofs_position(self, value):
        self.target = np.asarray(value).copy()

    def control_dofs_force(self, value, dofs_idx_local=None):
        self.force = (np.asarray(value).copy(), np.asarray(dofs_idx_local).copy())


def test_hold_dof_position_preserves_sibling_targets():
    entity = _FixtureEntity()
    assert hold_dof_position_compat(entity, 1, 0.05)
    np.testing.assert_allclose(entity.target, [0.0, 0.05, 0.0])
    assert entity.kp[1] == 2000.0
    assert entity.kv[1] == 200.0
    np.testing.assert_allclose(entity.force_range[1], [-500.0, 500.0])


def test_force_controller_moves_toward_target_without_setting_qpos():
    entity = _FixtureEntity()
    before = entity.qpos.copy()
    ok, qpos, qvel, force = control_dof_force_toward_position_compat(
        entity, 1, 0.20, kp=20.0, kv=4.0, force_limit=4.0,
    )
    assert ok
    assert np.isclose(qpos, 0.12)
    assert np.isclose(qvel, 0.01)
    assert force > 0.0
    np.testing.assert_allclose(entity.qpos, before)
    np.testing.assert_array_equal(entity.force[1], [1])


if __name__ == "__main__":
    test_episode_rng_does_not_advance_task_rng()
    test_episode_rng_streams_are_repeatable_and_independent()
    test_support_release_height_accounts_for_object_and_grasp_offset()
    test_topdown_catalog_options_prefer_controller_specific_hints()
    test_object_aware_drawer_target_uses_payload_and_joint_limit()
    test_contained_center_keeps_full_object_behind_edge()
    test_relative_parallel_jaw_yaw_tracks_object_with_half_turn_symmetry()
    test_pick_spec_grasp_offset_gate_is_opt_in()
    test_cabinet_standard_inspect_motions_attach_target()
    test_hold_dof_position_preserves_sibling_targets()
    test_force_controller_moves_toward_target_without_setting_qpos()
    print("test_generalized_placement: 11 passed")
