"""Neutral-avatar variant for ``place_dual_shoes``."""

from ..utils import to_numpy
from ..task_bases.neutral_avatar_table_work import (
    NeutralAvatarTableWorkMixin,
    NeutralTaskSpec,
)
from ..task_bases.place_dual_shoes import PlaceDualShoes


class PlaceDualShoesNeutral(NeutralAvatarTableWorkMixin, PlaceDualShoes):
    """Robot places both shoes while the human performs unrelated table work."""

    NEUTRAL_TASK_SPEC = NeutralTaskSpec(
        # Parent shoe/box region is x=[-0.22,0.22], y=[-0.38,0.06].
        work_xy=(0.42, 0.12),
        mouse_work_xy=(0.42, 0.16),
        table_xy_bounds=(-0.60, 0.60, -0.90, 0.20),
        max_table_edge_gap=0.67,
    )

    INSTRUCTION = (
        "place both shoes in the shoebox while the human works nearby on the "
        "same table"
    )

    def _neutral_avatar_forbidden_regions(self):
        regions = [
            (center, radius, label)
            for center, radius, label in super()._neutral_avatar_forbidden_regions()
            if str(label) != "shoebox" and not str(label).startswith("shoes:")
        ]
        # NOTE: these radii CANNOT be safely enlarged to remove the neutral
        # prop-vs-box overlap the user reported.  The neutral "can layout"
        # sampler for some seeds (e.g. seed 2) needs a spot right next to the
        # shoebox; widening the box keepout starves it and raises
        # "no neutral can layout candidate" at reset — which FAILS the whole
        # episode (strictly worse than a cosmetic overlap).  0.13 is already at
        # the feasibility edge (seed 2 flips pass/fail with FP/node noise), so
        # we keep the proven-safe original radii.  A real fix belongs in the
        # shared NeutralAvatarTableWorkMixin can-layout (more candidate
        # regions / graceful fallback), not this single-task keepout.
        for i, shoe in enumerate(getattr(self, "shoes", []) or []):
            ent = getattr(shoe, "entity", shoe)
            regions.append((to_numpy(ent.get_pos()).ravel()[:2], 0.060, f"shoes:{i}"))
        shoebox = getattr(self, "shoebox", None)
        if shoebox is not None:
            ent = getattr(shoebox, "entity", shoebox)
            regions.append((to_numpy(ent.get_pos()).ravel()[:2], 0.110, "shoebox"))
        return regions
