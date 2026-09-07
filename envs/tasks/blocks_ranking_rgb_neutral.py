"""Neutral-avatar variant for ``blocks_ranking_rgb``."""

import numpy as np

from ..task_bases.blocks_ranking_rgb import BlocksRankingRGB
from ..task_bases.neutral_avatar_table_work import (
    NeutralAvatarTableWorkMixin,
    NeutralTaskSpec,
)


class BlocksRankingRGBNeutral(NeutralAvatarTableWorkMixin, BlocksRankingRGB):
    """Robot orders RGB blocks while the human performs unrelated table work."""

    NEUTRAL_TASK_SPEC = NeutralTaskSpec(
        work_xy=(0.34, 0.10),
        mouse_work_xy=(0.34, 0.02),
        max_table_edge_gap=0.67,
    )

    INSTRUCTION = (
        "place the red, green, and blue blocks in left-to-right order while "
        "the human works nearby on the same table"
    )

    def _neutral_avatar_body_pose_is_valid(
        self, job: dict, xy: np.ndarray, body_pos: np.ndarray
    ) -> bool:
        if job.get("kind") in ("mouse", "sponge"):
            return True
        return super()._neutral_avatar_body_pose_is_valid(job, xy, body_pos)
