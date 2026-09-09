"""Task registry for the benchmark tasks described in ``README.md``."""

from .blocks_ranking_rgb_assist import BlocksRankingRGBAssist
from .blocks_ranking_rgb_interrupt import BlocksRankingRGBInterrupt
from .blocks_ranking_rgb_neutral import BlocksRankingRGBNeutral
from .blocks_ranking_size_assist import BlocksRankingSizeAssist
from .blocks_ranking_size_interrupt import BlocksRankingSizeInterrupt
from .blocks_ranking_size_neutral import BlocksRankingSizeNeutral
from .categorize_cooperative import CategorizeCooperative
from .categorize_interrupt import CategorizeInterrupt
from .categorize_neutral import CategorizeNeutral
from .deliver_to_human_easy import DeliverToHumanEasy
from .dump_bin_assist import DumpBinAssist
from .dump_bin_interrupt import DumpBinInterrupt
from .dump_bin_neutral import DumpBinNeutral
from .dump_bin_xarm_calibrated import (
    DumpBinXArmCalibrated,
    DumpBinXArmCalibratedAssist,
    DumpBinXArmCalibratedInterrupt,
)
from .factory_inspect_pack import FactoryInspectPack
from .factory_kit_packing import FactoryKitPacking
from .factory_line_feeding import FactoryLineFeeding
from .oil_bottle_recovery import OilBottleRecovery
from .open_microwave import OpenMicrowave
from .place_bread_in_basket_assist import PlaceBreadInBasketAssist
from .place_bread_in_basket_interrupt import PlaceBreadInBasketInterrupt
from .place_bread_in_basket_neutral import PlaceBreadInBasketNeutral
from .place_burger_fries_assist import PlaceBurgerFriesAssist
from .place_burger_fries_interrupt import PlaceBurgerFriesInterrupt
from .place_burger_fries_neutral import PlaceBurgerFriesNeutral
from .place_dual_shoes_assist import PlaceDualShoesAssist
from .place_dual_shoes_interrupt import PlaceDualShoesInterrupt
from .place_dual_shoes_neutral import PlaceDualShoesNeutral
from .place_food_in_skillet_assist import PlaceFoodInSkilletAssist
from .place_food_in_skillet_interrupt import PlaceFoodInSkilletInterrupt
from .place_food_in_skillet_neutral import PlaceFoodInSkilletNeutral
from .pour_water import PourWater
from .put_object_cabinet_assist import PutObjectCabinetAssist
from .put_object_cabinet_interrupt import PutObjectCabinetInterrupt
from .put_object_cabinet_neutral import PutObjectCabinetNeutral
from .stack_bowls_three_assist import StackBowlsThreeAssist
from .stack_bowls_three_interrupt import StackBowlsThreeInterrupt
from .stack_bowls_three_neutral import StackBowlsThreeNeutral
from .stamp_documents import StampDocuments
from .take_from_human_easy import TakeFromHumanEasy
from .take_from_human_safety import TakeFromHumanSafety
from ..task_bases.blocks_ranking_rgb import BlocksRankingRGB
from ..task_bases.blocks_ranking_size import BlocksRankingSize
from ..task_bases.dump_bin import DumpBin
from ..task_bases.place_bread_in_basket import PlaceBreadInBasket
from ..task_bases.place_burger_fries import PlaceBurgerFries
from ..task_bases.place_dual_shoes import PlaceDualShoes
from ..task_bases.place_food_in_skillet import PlaceFoodInSkillet
from ..task_bases.put_object_cabinet import PutObjectCabinet
from ..task_bases.stack_bowls_three import StackBowlsThree


TASK_MAP = {
    # Intention / singleton tasks.
    "take_from_human_easy": TakeFromHumanEasy,
    "deliver_to_human_easy": DeliverToHumanEasy,
    "take_from_human_safety": TakeFromHumanSafety,
    "stamp_documents": StampDocuments,
    "pour_water": PourWater,
    "oil_bottle_recovery": OilBottleRecovery,
    "open_microwave": OpenMicrowave,

    # Categorization family.  `categorize_interrupt` is kept below as a
    # compatibility alias because it shares the cooperative instruction string
    # in the LeRobot task table.
    "categorize_cooperative": CategorizeCooperative,
    "categorize_neutral": CategorizeNeutral,

    # Base/no-human tasks present as distinct LeRobot language instructions.
    "put_object_cabinet": PutObjectCabinet,
    "stack_bowls_three": StackBowlsThree,
    "place_bread_in_basket": PlaceBreadInBasket,
    "dump_bin": DumpBin,
    "dump_bin_xarm_calibrated": DumpBinXArmCalibrated,
    "place_burger_fries": PlaceBurgerFries,
    "place_dual_shoes": PlaceDualShoes,
    "place_food_in_skillet": PlaceFoodInSkillet,
    "blocks_ranking_rgb": BlocksRankingRGB,
    "blocks_ranking_size": BlocksRankingSize,

    # Assist / interrupt / neutral task families.
    "put_object_cabinet_assist": PutObjectCabinetAssist,
    "put_object_cabinet_interrupt": PutObjectCabinetInterrupt,
    "put_object_cabinet_neutral": PutObjectCabinetNeutral,
    "stack_bowls_three_assist": StackBowlsThreeAssist,
    "stack_bowls_three_interrupt": StackBowlsThreeInterrupt,
    "stack_bowls_three_neutral": StackBowlsThreeNeutral,
    "place_bread_in_basket_assist": PlaceBreadInBasketAssist,
    "place_bread_in_basket_interrupt": PlaceBreadInBasketInterrupt,
    "place_bread_in_basket_neutral": PlaceBreadInBasketNeutral,
    "dump_bin_assist": DumpBinAssist,
    "dump_bin_interrupt": DumpBinInterrupt,
    "dump_bin_xarm_calibrated_assist": DumpBinXArmCalibratedAssist,
    "dump_bin_xarm_calibrated_interrupt": DumpBinXArmCalibratedInterrupt,
    "dump_bin_neutral": DumpBinNeutral,
    "place_burger_fries_assist": PlaceBurgerFriesAssist,
    "place_burger_fries_interrupt": PlaceBurgerFriesInterrupt,
    "place_burger_fries_neutral": PlaceBurgerFriesNeutral,
    "place_dual_shoes_assist": PlaceDualShoesAssist,
    "place_dual_shoes_interrupt": PlaceDualShoesInterrupt,
    "place_dual_shoes_neutral": PlaceDualShoesNeutral,
    "place_food_in_skillet_assist": PlaceFoodInSkilletAssist,
    "place_food_in_skillet_interrupt": PlaceFoodInSkilletInterrupt,
    "place_food_in_skillet_neutral": PlaceFoodInSkilletNeutral,
    "blocks_ranking_rgb_assist": BlocksRankingRGBAssist,
    "blocks_ranking_rgb_interrupt": BlocksRankingRGBInterrupt,
    "blocks_ranking_rgb_neutral": BlocksRankingRGBNeutral,
    "blocks_ranking_size_assist": BlocksRankingSizeAssist,
    "blocks_ranking_size_interrupt": BlocksRankingSizeInterrupt,
    "blocks_ranking_size_neutral": BlocksRankingSizeNeutral,
    # Factory work-cell HRC family (docs/factory_tasks.md).
    "factory_inspect_pack": FactoryInspectPack,
    "factory_kit_packing": FactoryKitPacking,
    "factory_line_feeding": FactoryLineFeeding,
}

TASK_ALIASES = {
    "categorize_interrupt": CategorizeInterrupt,
}


NO_HUMAN_CANONICAL_TASK_MAP = {
    "categorize": CategorizeCooperative,
    "put_object_cabinet": PutObjectCabinet,
    "stack_bowls_three": StackBowlsThree,
    "place_bread_in_basket": PlaceBreadInBasket,
    "dump_bin": DumpBin,
    "place_burger_fries": PlaceBurgerFries,
    "place_dual_shoes": PlaceDualShoes,
    "place_food_in_skillet": PlaceFoodInSkillet,
    "blocks_ranking_rgb": BlocksRankingRGB,
    "blocks_ranking_size": BlocksRankingSize,
}


NO_HUMAN_VARIANT_ALIASES = {
    "categorize_cooperative": "categorize",
    "categorize_interrupt": "categorize",
    "categorize_neutral": "categorize",
    "put_object_cabinet_assist": "put_object_cabinet",
    "put_object_cabinet_interrupt": "put_object_cabinet",
    "put_object_cabinet_neutral": "put_object_cabinet",
    "stack_bowls_three_assist": "stack_bowls_three",
    "stack_bowls_three_interrupt": "stack_bowls_three",
    "stack_bowls_three_neutral": "stack_bowls_three",
    "place_bread_in_basket_assist": "place_bread_in_basket",
    "place_bread_in_basket_interrupt": "place_bread_in_basket",
    "place_bread_in_basket_neutral": "place_bread_in_basket",
    "dump_bin_assist": "dump_bin",
    "dump_bin_interrupt": "dump_bin",
    "dump_bin_neutral": "dump_bin",
    "place_burger_fries_assist": "place_burger_fries",
    "place_burger_fries_interrupt": "place_burger_fries",
    "place_burger_fries_neutral": "place_burger_fries",
    "place_dual_shoes_assist": "place_dual_shoes",
    "place_dual_shoes_interrupt": "place_dual_shoes",
    "place_dual_shoes_neutral": "place_dual_shoes",
    "place_food_in_skillet_assist": "place_food_in_skillet",
    "place_food_in_skillet_interrupt": "place_food_in_skillet",
    "place_food_in_skillet_neutral": "place_food_in_skillet",
    "blocks_ranking_rgb_assist": "blocks_ranking_rgb",
    "blocks_ranking_rgb_interrupt": "blocks_ranking_rgb",
    "blocks_ranking_rgb_neutral": "blocks_ranking_rgb",
    "blocks_ranking_size_assist": "blocks_ranking_size",
    "blocks_ranking_size_interrupt": "blocks_ranking_size",
    "blocks_ranking_size_neutral": "blocks_ranking_size",
}


NO_HUMAN_TASK_MAP = {
    **NO_HUMAN_CANONICAL_TASK_MAP,
    **{
        name: NO_HUMAN_CANONICAL_TASK_MAP[canonical]
        for name, canonical in NO_HUMAN_VARIANT_ALIASES.items()
    },
}


def resolve_task_class(task_name: str, no_human: bool = False):
    """Return (resolved_task_name, TaskClass), applying no-human collapse."""
    if no_human:
        canonical = NO_HUMAN_VARIANT_ALIASES.get(task_name, task_name)
        if canonical not in NO_HUMAN_CANONICAL_TASK_MAP:
            supported = sorted(NO_HUMAN_CANONICAL_TASK_MAP)
            raise ValueError(
                f"Task {task_name!r} has no no-human variant. "
                f"No-human is only defined for assist/interrupt/neutral "
                f"families: {supported}"
            )
        return canonical, NO_HUMAN_CANONICAL_TASK_MAP[canonical]

    if task_name in TASK_MAP:
        return task_name, TASK_MAP[task_name]
    if task_name in TASK_ALIASES:
        return task_name, TASK_ALIASES[task_name]
    available = sorted([*TASK_MAP.keys(), *TASK_ALIASES.keys()])
    raise ValueError(f"Unknown task: {task_name}. Available: {available}")
