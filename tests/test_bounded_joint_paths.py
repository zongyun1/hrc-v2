import numpy as np

from envs.planning.mplib_planner import MplibPlanner


def _planner(n_arm=7):
    planner = MplibPlanner.__new__(MplibPlanner)
    planner.n_arm = n_arm
    return planner


def test_debug_flag_is_opt_in(monkeypatch):
    monkeypatch.delenv("DEBUG_BOUNDED_JOINT_PATHS", raising=False)
    assert not MplibPlanner._bounded_paths_debug_enabled()

    monkeypatch.setenv("DEBUG_BOUNDED_JOINT_PATHS", "1")
    assert MplibPlanner._bounded_paths_debug_enabled()


def test_metrics_detect_gradual_full_range_wrist_windup():
    q0 = np.zeros(7)
    path = np.zeros((101, 7))
    # Every step is innocuous (0.056 rad), but the complete motion is not.
    path[:, 6] = np.linspace(0.0, 5.6, len(path))

    metrics = MplibPlanner._joint_path_metrics(q0, path, 7)
    assert metrics["max_step"] < 0.1
    assert np.isclose(metrics["travel"][6], 5.6)
    assert np.isclose(metrics["excursion"][6], 5.6)

    assert not _planner()._debug_path_quality_ok(
        q0, path, motion_kind="screw", log=False
    )


def test_quality_gate_accepts_local_multi_joint_motion():
    q0 = np.array([0.1, -0.3, 0.2, -1.5, 0.0, 1.2, 0.4])
    goal = q0 + np.array([0.2, -0.8, 0.1, -0.1, 0.55, 0.25, 0.35])
    alpha = np.linspace(0.0, 1.0, 80)[:, None]
    path = q0[None, :] * (1.0 - alpha) + goal[None, :] * alpha

    assert _planner()._debug_path_quality_ok(
        q0, path, motion_kind="screw", log=False
    )


def test_transit_gate_is_wider_than_screw_gate():
    q0 = np.zeros(7)
    path = np.zeros((60, 7))
    path[:, 2] = np.linspace(0.0, 2.2, len(path))

    planner = _planner()
    assert not planner._debug_path_quality_ok(
        q0, path, motion_kind="screw", log=False
    )
    assert planner._debug_path_quality_ok(
        q0, path, motion_kind="transit", log=False
    )
