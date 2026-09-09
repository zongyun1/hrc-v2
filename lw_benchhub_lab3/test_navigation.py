import unittest
import numpy as np
from isaac_human.motion import MotionPlayer, YieldGate
from lw_benchhub_lab3.navigation import yaw_xyzw, wrap, clearance, base_command, aisle_scenario


class NavigationTests(unittest.TestCase):
    def test_lab3_heading_and_wrap(self):
        for yaw in (0., .8, -2.7, np.pi):
            q = [0, 0, np.sin(yaw/2), np.cos(yaw/2)]
            self.assertAlmostEqual(wrap(yaw_xyzw(q)-yaw), 0.)

    def test_command_rotates_into_anchor_frame_and_limits_speed(self):
        cmd = base_command([0, 0], 0., [10, 0], 0., np.pi/2, .02)
        self.assertAlmostEqual(cmd['mobilebase_forward'], 0.)
        self.assertAlmostEqual(cmd['mobilebase_side'], -.44)
        self.assertTrue(all(v == 0 for v in base_command([1, 2], .7, [1, 2], .7, 0., .02).values()))

    def test_lookahead_stops_before_crossing_and_resumes_after(self):
        class Human:
            def capsules(self, t):
                return (np.array([[[1.2-t, 0., 0.]]]), np.array([[[1.2-t, 0., 1.7]]]), np.array([.1]))
        human = Human()
        gate = YieldGate(.3, .45, .4)
        self.assertGreater(clearance(human, 0., [0, 0], horizon=0.), .3)
        self.assertTrue(gate.step(clearance(human, 0., [0, 0]), .02))
        for _ in range(20):
            paused = gate.step(clearance(human, 3., [0, 0]), .02)
        self.assertFalse(paused)
        self.assertEqual((gate.entries, gate.resumes), (1, 1))

    def test_real_clip_transform_preserves_duration_and_displacement(self):
        from pathlib import Path
        assets = Path(__file__).resolve().parents[1]/'outputs/isaac_human/trumans_744000'
        if not assets.exists():
            self.skipTest('Real TRUMANS export unavailable')
        p = MotionPlayer.load(assets/'motion_kitchen_walk.npz', assets/'smplx_skeleton_male.json')
        waypoints, yaw, shift = aisle_scenario(p, [1.05, -2.975, 0.], [3.2, -.85, 0.])
        q = MotionPlayer(p.sequence, p.skeleton, yaw=yaw, translation=shift)
        midpoint = q.pose(0.)[0][0, 0]
        np.testing.assert_allclose(midpoint[:2], [1.15, -1.45], atol=1e-6)
        self.assertEqual(p.end_time, q.end_time)
        a = p.pose(p.end_time)[0]-p.pose(0.)[0]
        b = q.pose(q.end_time)[0]-q.pose(0.)[0]
        np.testing.assert_allclose(np.linalg.norm(a, axis=-1), np.linalg.norm(b, axis=-1))
        # The reviewed rear aisle lies between the island, rear fixtures, and side walls.
        for t in np.linspace(0., q.end_time, q.sequence.n_frames):
            a, b, radius = q.capsules(t)
            lower = (np.minimum(a, b)[0]-radius[:, None]).min(axis=0)
            upper = (np.maximum(a, b)[0]+radius[:, None]).max(axis=0)
            self.assertGreater(lower[0], .70)
            self.assertLess(upper[0], 5.10)
            self.assertGreater(lower[1], -2.40)
            self.assertLess(upper[1], -.86)
        with self.assertRaises(ValueError):
            aisle_scenario(p, [0., 0., 0.], [3.2, -.85, 0.])


if __name__ == '__main__':
    unittest.main()
