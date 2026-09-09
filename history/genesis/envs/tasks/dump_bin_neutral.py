"""Neutral-avatar variant for ``dump_bin``."""

from ..task_bases.dump_bin import DumpBin
from ..task_bases.neutral_avatar_table_work import (
    NeutralAvatarTableWorkMixin,
    NeutralTaskSpec,
)


class DumpBinNeutral(NeutralAvatarTableWorkMixin, DumpBin):
    """Robot dumps objects while the human performs unrelated table work."""

    NEUTRAL_TASK_SPEC = NeutralTaskSpec(
        # DumpBin uses a wider table footprint than the default tabletop.
        work_xy=(-0.36, 0.02),
        mouse_work_xy=(-0.36, 0.06),
        table_xy_bounds=(-0.56, 0.70, -0.82, 0.22),
        blocked_avatar_sides=("right",),
        work_candidates=(
            (-0.36, 0.02),
            (-0.42, 0.02),
            (-0.36, 0.08),
            (-0.42, 0.08),
            (-0.28, 0.06),
            (0.14, 0.11),
            (0.10, 0.10),
            (0.18, 0.10),
        ),
        mouse_work_candidates=(
            (-0.50, 0.00),
            (-0.50, 0.06),
            (-0.48, 0.02),
            (-0.48, 0.08),
        ),
    )

    INSTRUCTION = (
        "put the table objects into the bin while the human works nearby on "
        "the same table"
    )

    def play_once(self) -> bool:
        arm_tag = "right"
        self._boost_finger_pd(arm_tag)

        with self.suppress_recording():  # pre-task settle, excluded from recordings
            for _ in range(80):
                self.step_sim()

        self._start_neutral_avatar_table_work(add_start_delay=True)

        ok = True
        for i, cube in enumerate(self.objects):
            self._neutral_debug(
                f"[dump_bin_neutral] === picking object {i + 1}/{self.num_objects} ==="
            )
            try:
                cube_ok = self._pick_and_drop_cube(cube, arm_tag)
            except Exception as e:
                self._neutral_debug(
                    f"[dump_bin_neutral] object {i + 1} raised {type(e).__name__}: {e}"
                )
                cube_ok = False
            if not cube_ok:
                ok = False
                self._neutral_debug(
                    f"[dump_bin_neutral] object {i + 1} failed; continuing"
                )
                self.open_gripper(arm_tag)
                for _ in range(40):
                    self.step_sim()

        return ok

    def check_success(self) -> bool:
        if not DumpBin.check_success(self):
            return False
        if self.avatar_collision_checker is not None and self.avatar_collided:
            summary = self.avatar_collision_summary()
            self._neutral_debug(
                "[dump_bin_neutral] avatar collision detected: "
                f"first_frame={summary.get('first_collision_frame')} "
                f"deepest={summary.get('deepest_depth_m')}"
            )
            return False
        return True
