"""Focused regression tests for VLA avatar-collision success policy."""

import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.task_bases.place_bread_in_basket import PlaceBreadInBasket
from envs.tasks.place_bread_in_basket_assist import PlaceBreadInBasketAssist
from scripts.vla_data.collect_vla import _success_from_evaluation


def _bread_assist_stub(*, require_no_collision: bool):
    task = object.__new__(PlaceBreadInBasketAssist)
    task.config = {
        "success_require_no_avatar_collision": require_no_collision,
    }
    task.avatar_collision_checker = object()
    task.avatar_collided = True
    return task


class VlaCollisionSuccessPolicyTest(unittest.TestCase):
    def test_bread_assist_known_collision_does_not_override_geometric_success(self):
        task = _bread_assist_stub(require_no_collision=False)
        with patch.object(PlaceBreadInBasket, "check_success", return_value=True):
            self.assertTrue(task.check_success())

    def test_bread_assist_can_explicitly_require_collision_free_success(self):
        task = _bread_assist_stub(require_no_collision=True)
        with patch.object(PlaceBreadInBasket, "check_success", return_value=True):
            self.assertFalse(task.check_success())

    def test_collector_keeps_task_success_when_collision_is_telemetry(self):
        evaluation = {
            "success": True,
            "avatar_collision": {"enabled": True, "any_collision": True},
        }

        task = SimpleNamespace(VLA_ALLOW_AVATAR_COLLISION=True)
        self.assertTrue(_success_from_evaluation(task, evaluation))

    def test_collector_still_rejects_collision_without_task_exemption(self):
        evaluation = {
            "success": True,
            "avatar_collision": {"enabled": True, "any_collision": True},
        }

        self.assertFalse(_success_from_evaluation(SimpleNamespace(), evaluation))


if __name__ == "__main__":
    unittest.main()
