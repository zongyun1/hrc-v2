"""Boundary cases that catch inverted hinges and premature placement success."""
import unittest
import tempfile
from pathlib import Path
from task_semantics import all_doors_open, counter_to_sink_success, navigation_success, serve_tea_success, evaluate
from geometry_rules import source_geom_rules


class PredicateTests(unittest.TestCase):
    def test_serve_tea_requires_both_contacts(self):
        args = ([0., 0., .1], [0., 0., 0.], [1., 0., .1], .1)
        self.assertTrue(serve_tea_success(*args, True, True))
        self.assertFalse(serve_tea_success(*args, False, True))
        self.assertFalse(serve_tea_success(*args, True, False))

    def test_serve_tea_strict_distance_boundaries(self):
        self.assertTrue(serve_tea_success([.069, 0, 0], [0, 0, 0], [1, 0, 0], .1, True, True))
        self.assertFalse(serve_tea_success([.7*.1, 0, 0], [0, 0, 0], [1, 0, 0], .1, True, True))
        self.assertFalse(serve_tea_success([0, 0, 0], [0, 0, 0], [.25, 0, 0], .1, True, True))
        self.assertFalse(serve_tea_success([float('nan'), 0, 0], [0, 0, 0], [1, 0, 0], .1, True, True))

    def test_serve_tea_missing_contact_data_is_an_error(self):
        manifest = {"success_spec": {"kind": "serve_tea"}, "objects": {
            "teacup": {"body": "cup"}, "saucer": {"body": "plate", "horizontal_radius": .1}}}
        with self.assertRaises(ValueError):
            evaluate(manifest, {"cup": [0, 0, 0], "plate": [0, 0, 0]}, {}, [1, 0, 0])
        self.assertTrue(evaluate(manifest, {"cup": [0, 0, 0], "plate": [0, 0, 0]}, {}, [1, 0, 0],
                                 {"teacup_saucer": True, "saucer_table": True}))

    def test_navigation_needs_distance_and_heading(self):
        import math
        self.assertTrue(navigation_success([.2, 0, 0], 2*math.pi, [0, 0, 0], 0))
        self.assertFalse(navigation_success([.201, 0, 0], 0, [0, 0, 0], 0))
        self.assertFalse(navigation_success([0, 0, 0], .21, [0, 0, 0], 0))
        self.assertFalse(navigation_success([float('nan'), 0, 0], 0, [0, 0, 0], 0))
    def test_semantic_regions_are_neither_colliders_nor_visible(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "scene.xml"
            p.write_text('<mujoco><worldbody><geom name="region" group="1" contype="0" conaffinity="0" rgba="0 1 0 0"/>'
                         '<geom name="visual" group="1" contype="0" conaffinity="0"/>'
                         '<geom name="collision"/></worldbody></mujoco>')
            r = source_geom_rules(p)
            self.assertEqual(r["region"], {"collision": False, "visible": False})
            self.assertEqual(r["visual"], {"collision": False, "visible": True})
            self.assertEqual(r["collision"], {"collision": True, "visible": False})
    def test_negative_hinge_range_and_both_doors(self):
        limits = {"left": [-2., 0.], "right": [0., 2.]}
        self.assertTrue(all_doors_open({"left": -1.9, "right": 1.9}, limits))
        self.assertFalse(all_doors_open({"left": -1.9, "right": 0.1}, limits))
        self.assertFalse(all_doors_open({}, {}))

    def test_inside_requires_gripper_released_and_far(self):
        # A rotated region: local x points world +y, local y points world -x.
        region = {"basin": [[2, 3, 1], [2, 4, 1], [1, 3, 1], [2, 3, 2]]}
        self.assertTrue(counter_to_sink_success([1.5, 3.5, 1.5], [1.5, 3.5, 2.5], region))
        self.assertFalse(counter_to_sink_success([1.5, 3.5, 1.5], [1.5, 3.5, 1.6], region))
        self.assertFalse(counter_to_sink_success([2.5, 3.5, 1.5], [1.5, 3.5, 2.5], region))


if __name__ == "__main__":
    unittest.main()
