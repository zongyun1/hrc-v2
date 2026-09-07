import numpy as np
import transforms3d as t3d

from envs.grasp import load_grasp_poses
from envs.tasks.categorize_cooperative import CategorizeCooperative


def test_milk_box_uses_four_side_grasps_only():
    names = CategorizeCooperative.EXCLUSIVE_GRASP_NAMES["038_milk-box"]
    assert len(names) == 4
    assert "038_milk-box" in CategorizeCooperative.YAML_ONLY_GRASP_LABELS

    by_name = {
        grasp.name: grasp
        for grasp in load_grasp_poses("038_milk-box", model_id=0, robot_type="franka")
    }
    for name in names:
        approach = t3d.quaternions.quat2mat(by_name[name].pose.q)[:, 2]
        # Authored +Y is world-up after the categorize spawn transform.
        assert abs(float(approach[1])) < 0.20


def test_rubiks_cube_uses_authored_top_down_face_grasps_only():
    names = CategorizeCooperative.EXCLUSIVE_GRASP_NAMES["073_rubikscube"]
    assert names == (
        "manual_topdown_close_x",
        "manual_topdown_close_x_flipfinger",
        "manual_topdown_close_y",
        "manual_topdown_close_y_flipfinger",
    )
    assert "073_rubikscube" in CategorizeCooperative.YAML_ONLY_GRASP_LABELS

    by_name = {
        grasp.name: grasp
        for grasp in load_grasp_poses("073_rubikscube", model_id=1, robot_type="franka")
    }
    assert set(names) <= set(by_name)


def test_apple_is_not_silently_skipped():
    assert "035_apple" not in CategorizeCooperative.RADIUS_SKIP_LABELS
    assert np.isclose(CategorizeCooperative.RADIUS_CLOSE_OVERRIDE["035_apple"], 0.58)
    assert CategorizeCooperative.APPLE_YAML_MIN_X == 0.0
    assert CategorizeCooperative.APPLE_CENTER_MIN_X == 0.20
    assert CategorizeCooperative.APPLE_CENTER_GRASP_NAMES[0] == "manual_center_p0_yaw0"
    assert CategorizeCooperative.APPLE_UNDERHAND_GRASP_NAMES[0] == "manual_underhand_p0_yaw0"


def test_stapler_bans_long_axis_yaw_zero():
    assert np.isclose(
        CategorizeCooperative.RADIUS_GRASP_YAW["048_stapler"],
        np.pi / 2.0,
    )
