"""Neutral-avatar variant for ``place_food_in_skillet``."""

from ..task_bases.neutral_avatar_table_work import (
    NeutralAvatarTableWorkMixin,
    NeutralTaskSpec,
)
from ..task_bases.place_food_in_skillet import PlaceFoodInSkillet


class PlaceFoodInSkilletNeutral(NeutralAvatarTableWorkMixin, PlaceFoodInSkillet):
    """Robot places food in the skillet while the human performs unrelated table work."""

    def __init__(self, config: dict = None):
        cfg = dict(config or {})
        # Neutral data collection is a single-food task, but the food identity
        # is sampled per seeded episode from PlaceFoodInSkillet.RANDOM_FOOD_POOL.
        # Callers can still pin it with foods: [name] or disable sampling with
        # random_object: false for deterministic diagnostics.
        cfg.setdefault("random_object", True)
        super().__init__(cfg)

    NEUTRAL_TASK_SPEC = NeutralTaskSpec(
        # Kitchen table is wide; keep neutral work away from the cooktop
        # centered at (-0.12,0.08), the skillet start, and food starts.
        work_xy=(0.58, 0.26),
        mouse_work_xy=(0.58, 0.32),
        table_xy_bounds=(-0.80, 0.80, -0.45, 0.55),
        max_table_edge_gap=0.67,
    )

    INSTRUCTION = (
        "place the food in the skillet while the human works nearby on the "
        "same table"
    )
