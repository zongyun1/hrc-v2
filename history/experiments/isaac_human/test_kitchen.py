"""Preflight contract tests; USD loading/physics are validated on the GPU."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import numpy as np
from .diagnostic_fixture import write_fixture
from .run_kitchen_cosim import prepare


class KitchenPreflightTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        motion, skeleton = write_fixture(root/'inputs')
        source = root/'source'/'NavigateKitchen'
        source.mkdir(parents=True)
        for name in ('manifest.json','scene.xml','mobile_dynamics.json'):
            (source/name).write_text('{}')
        target = root/'usd'/'NavigateKitchen'
        target.mkdir(parents=True)
        self.usd = target/'scene.usda'
        self.usd.write_text('#usda 1.0')
        (target/'conversion.json').write_text(json.dumps({'usd_path':str(self.usd)}))
        self.args = SimpleNamespace(motion=motion, skeleton=skeleton, skin=None,
            human_yaw=0., human_translation=[0.,0.,0.], allow_diagnostic=True,
            capsules=True, migration_root=root)

    def test_preserves_source_world_frame_and_records_explicit_transform(self):
        player, _, _, provenance = prepare(self.args)
        np.testing.assert_allclose(player.pose(0)[0][0,0], [0,-1.8,1], atol=1e-6)
        self.assertTrue(provenance['synthetic_diagnostic'])
        self.args.human_translation = [2.5,-2.1,0.]
        moved, _, _, provenance = prepare(self.args)
        np.testing.assert_allclose(moved.pose(0)[0]-player.pose(0)[0],
                                  np.broadcast_to([2.5,-2.1,0.],(1,22,3)), atol=1e-6)
        self.assertEqual(provenance['human_translation_m'],self.args.human_translation)
        self.assertEqual(len(provenance['files'][str(self.usd)]),64)

    def test_missing_scene_is_not_silently_replaced(self):
        self.usd.unlink()
        with self.assertRaises(FileNotFoundError):
            prepare(self.args)

    def test_diagnostic_and_capsule_modes_require_explicit_selection(self):
        self.args.allow_diagnostic = False
        with self.assertRaisesRegex(ValueError,'allow_diagnostic'):
            prepare(self.args)
        self.args.allow_diagnostic = True
        self.args.capsules = False
        with self.assertRaisesRegex(ValueError,'skin'):
            prepare(self.args)

    def test_legacy_mode_cannot_silently_replace_smpl_motion(self):
        self.args.legacy_avatar_glb = self.usd.parent/'avatar.glb'
        self.args.legacy_avatar_glb.write_bytes(b'preflight path-only fixture')
        self.args.walk_duration = 10.
        with self.assertRaisesRegex(ValueError,'separate modes'):
            prepare(self.args)
        self.args.motion = self.args.skeleton = None
        self.args.capsules = False
        player, _, _, provenance = prepare(self.args)
        self.assertEqual(player.end_time,10.)
        self.assertIn('not TRUMANS',provenance['source_meta']['motion'])
        self.args.walk_duration = float('nan')
        with self.assertRaisesRegex(ValueError,'finite'):
            prepare(self.args)


if __name__ == '__main__':
    unittest.main()
