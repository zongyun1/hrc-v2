from __future__ import annotations

import json
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.task_bases import neutral_avatar_table_work as neutral_module
from envs.task_bases.neutral_avatar_table_work import NeutralAvatarTableWorkMixin


def _task(config: dict | None = None) -> NeutralAvatarTableWorkMixin:
    task = NeutralAvatarTableWorkMixin.__new__(NeutralAvatarTableWorkMixin)
    task.config = dict(config or {})
    task._neutral_avatar_reset_seed = 7
    task._neutral_avatar_job = None
    task._target_entry = None
    return task


def _clips() -> dict:
    with open(neutral_module._NEUTRAL_MOTION_META, encoding="utf-8") as stream:
        return json.load(stream)["clips"]


def test_take_apple_motion_identifies_apple_prop() -> None:
    task = _task()

    identities = task._neutral_clip_object_ids(_clips()["take_objects_from_bowl"])

    assert "035_apple" in identities
    assert "apple" in identities


def test_put_objects_override_identifies_configured_apple_prop() -> None:
    task = _task({"neutral_put_objects_object": "035_apple"})

    identities = task._neutral_clip_object_ids(_clips()["put_objects_in_bowl"])

    assert "035_apple" in identities
    assert "apple" in identities


def test_sampler_excludes_take_apple_motion_for_apple_target() -> None:
    task = _task({
        "neutral_avatar_motion_choices": [
            "take_objects_from_bowl",
            "use_mouse",
        ],
    })

    task._resolve_neutral_avatar_job(excluded_target_ids={"035_apple", "apple"})

    assert task._neutral_avatar_job["base_motion"] == "use_mouse"


def test_forced_duplicate_motion_is_rejected_clearly() -> None:
    task = _task({"neutral_avatar_motion": "take_objects_from_bowl"})
    task._target_entry = SimpleNamespace(key="035_apple", object_id="035_apple")
    task._resolve_neutral_avatar_job()

    try:
        task._ensure_neutral_avatar_job_is_target_distinct()
    except ValueError as exc:
        assert "duplicates robot target" in str(exc)
    else:
        raise AssertionError("forced target-duplicate neutral motion was accepted")


def test_multi_object_targets_detect_default_can_motion_conflict() -> None:
    task = _task()
    task._episode_dump_specs = [
        {"object_id": "038_milk-box"},
        {"object_id": "071_can"},
    ]
    task._resolve_neutral_avatar_job = lambda **_kwargs: None
    clip = _clips()["put_objects_in_bowl"]
    task._neutral_avatar_job = {
        "motion": "put_objects_in_bowl",
        "base_motion": "put_objects_in_bowl",
        "clip": clip,
    }

    targets = task._neutral_robot_target_object_ids()
    conflicts = targets & task._neutral_clip_object_ids(clip)

    assert "071_can" in conflicts
    assert "can" in conflicts


def test_oil_bottle_target_does_not_conflict_with_water_bottle_prop() -> None:
    task = _task()
    task._target_entry = SimpleNamespace(
        key="029_olive-oil",
        object_id="029_olive-oil",
    )
    clip = _clips()["open_bottle"]

    conflicts = (
        task._neutral_robot_target_object_ids()
        & task._neutral_clip_object_ids(clip)
    )

    assert not conflicts


if __name__ == "__main__":
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
