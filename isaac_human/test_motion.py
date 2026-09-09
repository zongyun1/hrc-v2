import ast
import tempfile
import unittest
from pathlib import Path
import numpy as np
from .motion import MotionPlayer, YieldGate, point_clearance, slerp
from .adapter import capsule_quaternion, capsule_pose
from .diagnostic_fixture import write_fixture
from .vendor.schema import MotionSequence
from .vendor.skin import HumanSkin, quats_to_R


class MotionTests(unittest.TestCase):
    def test_capsule_pose_runtime_conventions(self):
        for end in ([0., 0., 2.], [0., 0., -2.], [1., 2., 3.]):
            xyzw = capsule_pose([0., 0., 0.], end)
            wxyz = capsule_pose([0., 0., 0.], end, 'wxyz')
            np.testing.assert_allclose(xyzw[:3], np.asarray(end) / 2)
            np.testing.assert_allclose(wxyz[:3], xyzw[:3])
            np.testing.assert_allclose(wxyz[3:][[1, 2, 3, 0]], xyzw[3:])
            rotation = quats_to_R(wxyz[3:][None])[0]
            np.testing.assert_allclose(rotation @ np.array([0., 0., 1.]),
                                       np.asarray(end) / np.linalg.norm(end), atol=1e-7)
        with self.assertRaises(ValueError):
            capsule_pose([0, 0, 0], [0, 0, 1], 'invalid')

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.motion, self.skeleton = write_fixture(self.temp.name)
        self.player = MotionPlayer.load(self.motion, self.skeleton)

    def test_exact_schema_roundtrip_and_time_interpolation(self):
        seq = MotionSequence.load(self.motion)
        self.assertEqual(seq.n_frames, 241)
        before = self.player.pose(0.)[0]
        after = self.player.pose(1/30)[0]
        midpoint = self.player.pose(1/60)[0]
        np.testing.assert_allclose(midpoint, (before+after)/2, atol=1e-7)
        np.testing.assert_allclose(self.player.pose(-1)[0], before)
        np.testing.assert_allclose(self.player.pose(100)[0], self.player.pose(8)[0])

    def test_fk_preserves_bone_lengths_and_smpl_pelvis_offset(self):
        pos, quat = self.player.pose(2.)
        np.testing.assert_allclose(pos[0, 0], [0, -.9, 1], atol=1e-6)
        for j in range(1, 22):
            p = self.player.skeleton.parents[j]
            self.assertAlmostEqual(np.linalg.norm(pos[0,j]-pos[0,p]),
                                   np.linalg.norm(self.player.skeleton.rest_offsets[j]))
        np.testing.assert_allclose(np.linalg.norm(quat, axis=-1), 1, atol=1e-6)

    def test_batch_pose_and_skin_remain_distinct(self):
        seq = self.player.sequence
        for field in ('transl', 'global_orient', 'body_pose', 'betas'):
            setattr(seq, field, np.repeat(getattr(seq, field), 2, axis=0))
        seq.transl[1, :, 0] += 3
        joints = self.player.skeleton.rest_joints
        skin = HumanSkin(joints, [[0,1,2]], joints, np.arange(22)[:, None], np.ones((22,1)))
        player = MotionPlayer(seq, self.player.skeleton, skin)
        vertices = player.vertices(2.)
        np.testing.assert_allclose(vertices[1]-vertices[0], np.tile([3,0,0], (22,1)), atol=1e-6)
        np.testing.assert_allclose(vertices, player.pose(2.)[0], atol=1e-6)

    def test_world_transform_applies_once_to_capsules(self):
        moved = MotionPlayer(self.player.sequence, self.player.skeleton, yaw=np.pi/2, translation=(2,3,0))
        a, b, r = self.player.capsules(1.)
        aa, bb, _ = moved.capsules(1.)
        for source, actual in ((a, aa), (b, bb)):
            expected = np.stack([-source[...,1], source[...,0], source[...,2]], axis=-1)+[2,3,0]
            np.testing.assert_allclose(expected, actual, atol=1e-6)

    def test_invalid_assets_are_rejected(self):
        self.player.sequence.betas[0, 0] = 1
        with self.assertRaises(ValueError):
            MotionPlayer(self.player.sequence, self.player.skeleton)
        self.player.sequence.betas[:] = 0
        self.player.sequence.valid = np.ones((1,241), dtype=bool)
        self.player.sequence.valid[0, 10] = False
        with self.assertRaises(ValueError):
            MotionPlayer(self.player.sequence, self.player.skeleton)

    def test_preflight_checks_source_joint_cache(self):
        from argparse import Namespace
        from .run_demo import prepare
        args = Namespace(motion=str(self.motion), skeleton=str(self.skeleton), skin=None,
                         capsules=True, allow_diagnostic=True, crossing_x=1.05)
        seq = self.player.sequence
        seq.joints = np.stack([self.player.pose(i/seq.fps)[0]
                               for i in range(seq.n_frames)], axis=1).astype(np.float32)
        seq.save(self.motion)
        _, report = prepare(args)
        self.assertLess(report['source_joint_cache_max_error_m'], 1e-6)
        seq.joints[..., 0] += .1
        seq.save(self.motion)
        with self.assertRaisesRegex(ValueError, 'FK differs'):
            prepare(args)

    def test_slerp_handles_quaternion_sign(self):
        q = np.array([[1., 0, 0, 0]])
        np.testing.assert_allclose(slerp(q, -q, .5), q)

    def test_capsule_orientation_handles_down_and_zero(self):
        for axis in ([0,0,-1], [1,2,3], [0,0,0]):
            xyzw = capsule_quaternion(np.array(axis, dtype=float))
            matrix = quats_to_R(xyzw[[3,0,1,2]][None])[0]
            if np.linalg.norm(axis):
                np.testing.assert_allclose(matrix @ [0,0,1], np.array(axis)/np.linalg.norm(axis), atol=1e-6)

    def test_capsule_surface_distance_not_joint_distance(self):
        distance = point_clearance(np.array([.05,0,.5]), np.array([[0,0,0]]), np.array([[0,0,1]]), np.array([.1]))
        self.assertAlmostEqual(float(distance[0]), -.05)

    def test_yield_hysteresis_wait_and_reset(self):
        gate = YieldGate()
        self.assertTrue(gate.step(.2, .1))
        self.assertTrue(gate.step(.6, .1))
        self.assertTrue(gate.step(.5, .1))
        self.assertTrue(gate.step(.6, .1))
        self.assertTrue(gate.step(.6, .1))
        self.assertFalse(gate.step(.6, .1))
        self.assertEqual((gate.entries, gate.resumes), (1,1))
        gate.reset()
        self.assertEqual((gate.entries, gate.resumes), (0,0))
        with self.assertRaises(ValueError):
            gate.step(float('nan'), .1)

    def test_no_genesis_imports_in_adapted_package(self):
        for path in Path(__file__).parent.rglob('*.py'):
            for node in ast.walk(ast.parse(path.read_text())):
                names = ([a.name for a in node.names] if isinstance(node, ast.Import)
                         else [node.module or ''] if isinstance(node, ast.ImportFrom) else [])
                self.assertFalse(any(n.split('.')[0] == 'genesis' for n in names), str(path))


if __name__ == '__main__':
    unittest.main()
