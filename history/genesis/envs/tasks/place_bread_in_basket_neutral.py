"""Neutral-avatar variant for ``place_bread_in_basket``."""

from ..utils import to_numpy
from ..task_bases.neutral_avatar_table_work import (
    NeutralAvatarTableWorkMixin,
    NeutralTaskSpec,
)
from ..task_bases.place_bread_in_basket import PlaceBreadInBasket


class PlaceBreadInBasketNeutral(NeutralAvatarTableWorkMixin, PlaceBreadInBasket):
    """Robot places bread in a basket while the human performs unrelated table work."""

    NEUTRAL_TASK_SPEC = NeutralTaskSpec(
        # Parent objects live in x=[-0.20,0.20], y=[-0.36,0.04].
        work_xy=(0.42, 0.12),
        mouse_work_xy=(0.42, 0.16),
        table_xy_bounds=(-0.60, 0.60, -0.90, 0.20),
        max_table_edge_gap=0.67,
    )

    INSTRUCTION = (
        "place both pieces of bread in the basket while the human works nearby "
        "on the same table"
    )

    def _neutral_avatar_forbidden_regions(self):
        regions = [
            (center, radius, label)
            for center, radius, label in super()._neutral_avatar_forbidden_regions()
            if str(label) not in {"basket"} and not str(label).startswith("breads:")
        ]
        for i, bread in enumerate(getattr(self, "breads", []) or []):
            ent = getattr(bread, "entity", bread)
            regions.append((to_numpy(ent.get_pos()).ravel()[:2], 0.055, f"breads:{i}"))
        basket = getattr(self, "basket", None)
        if basket is not None:
            ent = getattr(basket, "entity", basket)
            regions.append((to_numpy(ent.get_pos()).ravel()[:2], 0.105, "basket"))
        return regions
