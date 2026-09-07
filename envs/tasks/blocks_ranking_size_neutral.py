"""Neutral-avatar variant for ``blocks_ranking_size``."""

import numpy as np

from ..task_bases.blocks_ranking_size import BlocksRankingSize
from ..task_bases.neutral_avatar_table_work import (
    NeutralAvatarTableWorkMixin,
    NeutralTaskSpec,
)


class BlocksRankingSizeNeutral(NeutralAvatarTableWorkMixin, BlocksRankingSize):
    """Robot orders size blocks while the human performs unrelated table work."""

    recording_camera_pos = [-0.85, 0.65, 1.55]
    recording_camera_lookat = [0.0, -0.22, 0.82]

    NEUTRAL_TASK_SPEC = NeutralTaskSpec(
        work_xy=(0.34, 0.10),
        mouse_work_xy=(0.34, 0.15),
        max_table_edge_gap=0.67,
    )

    INSTRUCTION = (
        "place the blocks from largest to smallest while the human works "
        "nearby on the same table"
    )

    def _neutral_avatar_body_pose_is_valid(
        self, job: dict, xy: np.ndarray, body_pos: np.ndarray
    ) -> bool:
        if job.get("kind") in ("mouse", "sponge"):
            return True
        return super()._neutral_avatar_body_pose_is_valid(job, xy, body_pos)
