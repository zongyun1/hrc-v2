"""Neutral-avatar variant for ``put_object_cabinet``."""

from ..task_bases.neutral_avatar_table_work import (
    NeutralAvatarTableWorkMixin,
    NeutralTaskSpec,
)
from ..task_bases.put_object_cabinet import PutObjectCabinet


class PutObjectCabinetNeutral(NeutralAvatarTableWorkMixin, PutObjectCabinet):
    """Robot puts an object in the cabinet while the human performs unrelated table work."""

    NEUTRAL_TASK_SPEC = NeutralTaskSpec(
        # Cabinet uses the full front/back lane: object starts near y=-0.2,
        # the drawer opens at y=-0.55, and the Franka hand parks above that
        # corridor. Match PutObjectCabinet._create_table exactly: center
        # (0,-0.35), half_size=(0.50,0.45), so tabletop y ends at +0.10.
        work_xy=(0.42, 0.12),
        mouse_work_xy=(0.42, 0.16),
        table_xy_bounds=(-0.50, 0.50, -0.80, 0.10),
        work_candidates=(
            (0.42, 0.12),
            (0.42, 0.20),
            (0.34, 0.16),
            (0.50, 0.12),
        ),
    )

    INSTRUCTION = (
        "put the table object in the cabinet drawer while the human works "
        "nearby on the same table"
    )
