"""Reusable scene builders (kitchen, desk, etc.) composed into tasks."""

from .factory import (
    FactorySceneMixin,
    FactoryRack,
    FactoryFloorTape,
    FactoryPallet,
    build_factory_rack,
    build_factory_floor_tape,
    build_factory_pallet,
)
from .kitchen import (
    KitchenSceneMixin,
    KitchenItem,
    KitchenShelfItem,
    KitchenBackdrop,
    DEFAULT_KITCHEN_LAYOUT,
    DEFAULT_CABINET_ITEMS,
    build_kitchen_table,
    build_kitchen_backdrop,
    build_kitchen_cooktop,
    COOKTOP_XY,
    COOKTOP_HEIGHT_M,
)

__all__ = [
    "FactorySceneMixin",
    "FactoryRack",
    "FactoryFloorTape",
    "FactoryPallet",
    "build_factory_rack",
    "build_factory_floor_tape",
    "build_factory_pallet",
    "KitchenSceneMixin",
    "KitchenItem",
    "KitchenShelfItem",
    "KitchenBackdrop",
    "DEFAULT_KITCHEN_LAYOUT",
    "DEFAULT_CABINET_ITEMS",
    "build_kitchen_table",
    "build_kitchen_backdrop",
    "build_kitchen_cooktop",
    "COOKTOP_XY",
    "COOKTOP_HEIGHT_M",
]
