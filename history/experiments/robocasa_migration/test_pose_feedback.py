import unittest
import numpy as np
from pose_feedback import held_object_target, rotate, blend_quat
from task_semantics import upright_tilt_degrees


class PoseFeedbackTests(unittest.TestCase):
    def test_retarged_gripper_preserves_held_transform(self):
        half = np.sqrt(.5)
        cup_q = np.array([half, 0., 0., half])
        cup_pos = np.array([.1, .2, .3])
        eef_pos = np.array([.1, .3, .3])
        target = np.array([1., 2., 3.])
        pos, quat = held_object_target(eef_pos, cup_q, cup_pos, cup_q, target, [0, 0, 0, 1])
        np.testing.assert_allclose(quat, [0, 0, 0, 1], atol=1e-12)
        np.testing.assert_allclose(pos+rotate(quat, [0, 0, .1]), target, atol=1e-12)

    def test_quaternion_sign_does_not_change_interpolation(self):
        np.testing.assert_allclose(blend_quat([0, 0, 0, 1], [0, 0, 0, -1], .5), [0, 0, 0, 1])

    def test_upright_distinguishes_yaw_sideways_and_inverted(self):
        half = np.sqrt(.5)
        self.assertAlmostEqual(upright_tilt_degrees([0, 0, half, half]), 0)
        self.assertAlmostEqual(upright_tilt_degrees([half, 0, 0, half]), 90)
        self.assertAlmostEqual(upright_tilt_degrees([1, 0, 0, 0]), 180)
        self.assertGreater(upright_tilt_degrees([0, 0, 0, 0]), 10)


if __name__ == '__main__':
    unittest.main()
