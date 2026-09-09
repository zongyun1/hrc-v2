import json
from pathlib import Path
import tempfile
import unittest
from dataclasses import replace
from unittest.mock import patch
from types import SimpleNamespace

from hrc_bench.config import BenchmarkConfig, HumanMotionConfig
from hrc_bench.human import TrumanHumanMotion

ROOT = Path(__file__).resolve().parents[1]


class HumanInterfaceTests(unittest.TestCase):
    def test_disabled_needs_no_assets(self):
        human = TrumanHumanMotion(HumanMotionConfig(enabled=False, motion='/missing'))
        human.advance(.5)
        self.assertEqual(human.observe()['elapsed_s'], .5)
        self.assertIsNone(human.observe()['peak_robot_contact_N'])
        human.reset()
        self.assertEqual(human.elapsed_s, 0.)
        self.assertEqual(human.last_episode['elapsed_s'], .5)

    def test_config_paths_and_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'config.json'
            path.write_text(json.dumps({'human': {'enabled': True, 'motion': 'm.npz', 'skeleton': 's.json'}}))
            cfg = BenchmarkConfig.load(path)
            self.assertEqual(cfg.human.motion, str((Path(directory)/'m.npz').resolve()))
        for params in ({'num_envs': 2}, {'backend': 'genesis'}, {'episode_length_s': float('nan')}):
            with self.assertRaises(ValueError):
                BenchmarkConfig(**params)
        with self.assertRaises(ValueError):
            HumanMotionConfig(enabled=True)
        with self.assertRaises(ValueError):
            HumanMotionConfig(start_delay_s=-1.)

    def test_reject_non_trumans(self):
        fake = SimpleNamespace(sequence=SimpleNamespace(meta={'backend': 'synthetic'}, n_envs=1))
        with patch('isaac_human.motion.MotionPlayer.load', return_value=fake):
            with self.assertRaisesRegex(ValueError, 'real TRUMANS'):
                TrumanHumanMotion(HumanMotionConfig(enabled=True, motion='x', skeleton='y'))

    def test_real_motion_clock_reset_and_end_hold(self):
        config = BenchmarkConfig.load(ROOT/'configs/lightwheel.json').human
        if not Path(config.motion).is_file():
            self.skipTest('Real TRUMANS assets are external to Git')
        human = TrumanHumanMotion(replace(config, start_delay_s=.5))
        first = human.observe()['joints_w']
        human.advance(.25)
        self.assertEqual(human.observe()['joints_w'], first)
        human.advance(.75)
        self.assertEqual(human.source_time_s, .5)
        self.assertNotEqual(human.observe()['joints_w'], first)
        human.reset()
        self.assertEqual(human.observe()['joints_w'], first)
        self.assertEqual(human.elapsed_s, 0.)
        human.advance(human.player.end_time+.5)
        self.assertTrue(human.observe()['finished'])
        last = human.observe()['joints_w']
        human.advance(2.)
        self.assertEqual(human.observe()['joints_w'], last)
        for dt in (0., -1., float('nan')):
            with self.assertRaises(ValueError):
                human.advance(dt)

    def test_substep_contact_is_flushed_once_and_reset_clears_state(self):
        human = TrumanHumanMotion(HumanMotionConfig())
        class Avatar:
            sensors = []
            max_tracking_error = .2
            updates = 0
            def set_time(self, t, render):
                self.time = t
            def update(self, dt):
                self.updates += 1
                return .4
        human.avatar = Avatar()
        human.reset()
        human.before_physics(.01)
        human.before_physics(.01)
        human.after_physics(.01)
        human.after_physics(.01)
        self.assertEqual(human.avatar.updates, 2)
        self.assertEqual(human.peak_contact_N, .4)
        human.reset()
        self.assertEqual(human.peak_contact_N, 0.)
        self.assertEqual(human.last_episode['peak_robot_contact_N'], .4)
        self.assertEqual(human.avatar.max_tracking_error, 0.)


if __name__ == '__main__':
    unittest.main()
