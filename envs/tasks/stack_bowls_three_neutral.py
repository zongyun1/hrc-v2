"""Neutral-avatar variant for ``stack_bowls_three``."""

from ..utils import to_numpy
from ..task_bases.neutral_avatar_table_work import (
    NeutralAvatarTableWorkMixin,
    NeutralTaskSpec,
)
from ..task_bases.stack_bowls_three import StackBowlsThree


class StackBowlsThreeNeutral(NeutralAvatarTableWorkMixin, StackBowlsThree):
    """Robot stacks bowls while the human performs unrelated table work."""

    NEUTRAL_TASK_SPEC = NeutralTaskSpec(
        # Bowls sample in x=[-0.18,0.22], y=[-0.18,0.18]; use far right/back.
        work_xy=(0.42, 0.26),
        mouse_work_xy=(0.42, 0.30),
        table_xy_bounds=(-0.60, 0.60, -0.90, 0.20),
        max_table_edge_gap=0.67,
    )

    INSTRUCTION = (
        "stack the bowls while the human works nearby on the same table"
    )

    def _neutral_avatar_forbidden_regions(self):
        regions = [
            (center, radius, label)
            for center, radius, label in super()._neutral_avatar_forbidden_regions()
            if not str(label).startswith("bowls:")
        ]
        for i, bowl in enumerate(getattr(self, "bowls", []) or []):
            ent = getattr(bowl, "entity", bowl)
            regions.append((to_numpy(ent.get_pos()).ravel()[:2], 0.10, f"bowls:{i}"))
        return regions
