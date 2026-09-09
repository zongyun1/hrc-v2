import tempfile
import unittest
from pathlib import Path

from lw_benchhub_lab3.migrate import port_array_access, port_literal_rotations, replace


class MigrationTests(unittest.TestCase):
    def test_array_access_is_idempotent_and_preserves_explicit_backends(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'sample.py'
            path.write_text('label = "中文"; q = robot.data.joint_pos[:, 0]\n'
                            'r = robot.data.root_quat_w.torch\n'
                            'w = robot.data.root_pos_w.warp\n'
                            'names = robot.data.joint_names\n')
            port_array_access(path)
            migrated = path.read_text()
            self.assertIn('robot.data.joint_pos.torch[:, 0]', migrated)
            self.assertIn('robot.data.root_pos_w.warp', migrated)
            self.assertIn('robot.data.joint_names', migrated)
            port_array_access(path)
            self.assertEqual(path.read_text(), migrated)

    def test_only_authored_config_quaternions_are_reordered(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'sample.py'
            path.write_text('a = Cfg(rot=(1., 0., 0., 0.))\n'
                            'b = Cfg(rot=(0.5, -0.5, 0.5, -0.5))\n'
                            'p = Pose(rotation_wxyz=(1., 0., 0., 0.))\n'
                            'v = Cfg(rot=runtime_quaternion)\n')
            port_literal_rotations(path)
            migrated = path.read_text()
            self.assertIn('rot=(0.0, 0.0, 0.0, 1.0)', migrated)
            self.assertIn('rot=(-0.5, 0.5, -0.5, 0.5)', migrated)
            self.assertIn('rotation_wxyz=(1., 0., 0., 0.)', migrated)
            port_literal_rotations(path)
            self.assertEqual(path.read_text(), migrated)

    def test_insertion_does_not_repeat(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'sample.py').write_text('def f():\n    pass\n')
            for _ in range(2):
                replace(root, 'sample.py', 'def f():', '# comment\ndef f():')
            self.assertEqual((root / 'sample.py').read_text().count('# comment'), 1)

    def test_unwrapping_still_applies_when_new_is_inside_old(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / 'sample.py'
            path.write_text('a = convert(quat())\nb = quat()\n')
            for _ in range(2):
                replace(root, 'sample.py', 'convert(quat())', 'quat()')
            self.assertEqual(path.read_text(), 'a = quat()\nb = quat()\n')


if __name__ == '__main__':
    unittest.main()
