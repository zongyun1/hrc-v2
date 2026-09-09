import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.base_task import BaseTask
from envs.object_catalog import CATEGORIES, resolve_object_set
from envs.task_bases.dump_bin import DumpBin
from envs.tasks.deliver_to_human_easy import DeliverToHumanEasy
from envs.tasks.factory_inspect_pack import FactoryInspectPack
from envs.tasks.factory_kit_packing import FactoryKitPacking
from envs.tasks.take_from_human_easy import TakeFromHumanEasy


class _PoolTask(BaseTask):
    INSTRUCTION = "move the {object}"
    OBJECT_SET = "tabletop_pick_pool"


def _bare_ids(tokens):
    return {token.split("@", 1)[0] for token in tokens}


def test_shared_pool_membership():
    assert CATEGORIES["human_transfer_pool"] == [
        "035_apple@1",
        "073_rubikscube@1",
    ]
    assert _bare_ids(CATEGORIES["factory_product_pool"]) == {
        "113_coffee-box",
        "038_milk-box",
        "112_tea-box",
        "023_tissue-box",
        "073_rubikscube",
        "086_woodenblock",
    }
    assert len(CATEGORIES["tabletop_pick_pool"]) == len(
        set(CATEGORIES["tabletop_pick_pool"])
    )


def test_multi_object_resolver_is_seeded_and_distinct():
    np.random.seed(7)
    first = _PoolTask({}).resolve_target_objects(4)
    np.random.seed(7)
    second = _PoolTask({}).resolve_target_objects(4)
    first_ids = [(entry.key, entry.model_id) for entry in first]
    second_ids = [(entry.key, entry.model_id) for entry in second]
    assert first_ids == second_ids
    assert len(first_ids) == len(set(first_ids))


def test_forced_multi_object_override_repeats_for_debug_sweeps():
    entries = _PoolTask({"object_name": "035_apple"}).resolve_target_objects(3)
    assert [(entry.key, entry.model_id) for entry in entries] == [
        ("035_apple", 1),
        ("035_apple", 1),
        ("035_apple", 1),
    ]


def test_tasks_point_at_the_shared_pools():
    assert DeliverToHumanEasy.OBJECT_SET == "human_transfer_pool"
    assert TakeFromHumanEasy.OBJECT_SET == "human_transfer_pool"
    assert FactoryInspectPack.OBJECT_SET == "factory_product_pool"
    assert FactoryKitPacking.OBJECT_SET == "factory_product_pool"


def test_factory_task_local_tuning_covers_the_shared_pool():
    expected = _bare_ids(resolve_object_set("factory_product_pool"))
    inspect_ids = {item[0] for item in FactoryInspectPack.ITEM_POOL}
    kit_ids = {item[0] for item in FactoryKitPacking.OBJECT_POOL}
    assert inspect_ids == expected
    assert kit_ids == expected


def test_dump_objects_randomize_by_default():
    assert DumpBin({})._dump_random_objects is True


def test_dump_uses_task_local_round_and_carton_grips():
    apple = DumpBin({"object_name": "035_apple"})._sample_dump_object_spec(0)
    jam = DumpBin({"object_name": "031_jam-jar"})._sample_dump_object_spec(0)
    milk = DumpBin({"object_name": "038_milk-box"})._sample_dump_object_spec(0)
    assert "close_value" not in apple
    assert jam["close_value"] == 0.55
    assert milk["close_value"] == 0.55
    assert DumpBin.MILK_SIDE_GRASP_NAME == "grasp_065"
    assert DumpBin.MILK_SIDE_HOLD_STEPS == 100


if __name__ == "__main__":
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
