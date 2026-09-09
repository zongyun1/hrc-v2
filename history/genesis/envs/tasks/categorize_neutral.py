"""Neutral-avatar variant scaffold for ``categorize``."""

import numpy as np

from ..base_task import BaseTask
from .categorize_cooperative import PickSpec, PlaceSpec
from .categorize_interrupt import CategorizeInterrupt
from ..task_bases.neutral_avatar_table_work import (
    NeutralAvatarTableWorkMixin,
    NeutralTaskSpec,
)


class CategorizeNeutral(NeutralAvatarTableWorkMixin, CategorizeInterrupt):
    """Robot categorizes objects while the human performs unrelated table work."""

    NEUTRAL_TASK_SPEC = NeutralTaskSpec(
        # Categorize uses a larger table; keep neutral work on the right/back
        # edge, away from the basket/object columns.
        work_xy=(0.50, 0.24),
        mouse_work_xy=(0.50, 0.30),
        sponge_work_region=(0.46, 0.66, 0.22, 0.34),
        table_xy_bounds=(-0.80, 0.80, -0.55, 0.25),
        work_candidates=(
            (0.50, 0.24),
            (0.58, 0.24),
            (0.50, 0.32),
            (0.58, 0.32),
            (-0.50, 0.24),
            (-0.58, 0.24),
        ),
        mouse_work_candidates=(
            (0.50, 0.30),
            (0.58, 0.30),
            (0.50, 0.34),
            (0.58, 0.34),
            (-0.50, 0.30),
        ),
    )

    INSTRUCTION = (
        "sort the objects into matching baskets while the human works nearby "
        "on the same table"
    )

    def play_once(self) -> bool:
        arm_tag = "right"
        self._boost_arm_pd()
        self._start_neutral_avatar_table_work(add_start_delay=True)
        self._open_basket_lids()
        self._move_to_safe(arm_tag)

        ok_any = False
        order = list(range(len(self._sort_targets)))
        order.sort(reverse=True)
        for idx in order:
            actor, obj_name, model_id, basket_idx = self._sort_targets[idx]
            self._neutral_debug(
                f"[categorize_neutral] robot sorting {idx}: {obj_name} -> basket {basket_idx}"
            )
            pick = PickSpec(
                get_center=lambda a=actor, n=obj_name, m=model_id: self._get_object_world_center(a, n, m),
                radius=self._get_object_cross_section_radius(obj_name, model_id),
                label=obj_name,
                close_target_m=self._get_grasp_close_target(obj_name, model_id),
                actor=actor,
            )
            basket_pos = np.array(self.basket_poses[basket_idx].p, dtype=float)
            drop_from_z = basket_pos[2] + (0.10 if obj_name == "100_seal" else 0.18)
            place = PlaceSpec(
                pos=basket_pos,
                label=f"basket {basket_idx}",
                basket_idx=basket_idx,
                drop_from_z=float(drop_from_z),
            )
            try:
                ok = self.pick_and_place(pick, place, arm_tag)
            except Exception as e:
                self._neutral_debug(
                    f"[categorize_neutral] {obj_name} raised {type(e).__name__}: {e}"
                )
                ok = False
            self._move_to_safe(arm_tag)
            ok_any = bool(ok) or ok_any

        return ok_any

    def _robot_basket_metrics(self) -> dict:
        center_threshold = 0.10
        footprint_half = self._BASKET_INNER_HALF_XY + self._BASKET_FOOTPRINT_MARGIN
        per_object = []
        all_correct = True
        for actor, obj_name, model_id, basket_idx in self._sort_targets:
            obj_center = self._get_object_world_center(actor, obj_name, model_id)
            basket_pos = np.array(self.basket_poses[basket_idx].p, dtype=float)
            dist_xy = float(np.linalg.norm(obj_center[:2] - basket_pos[:2]))
            lo_xy, hi_xy = self._actor_footprint_aabb(actor, obj_name, model_id)
            rel_lo = lo_xy - basket_pos[:2]
            rel_hi = hi_xy - basket_pos[:2]
            center_ok = bool(dist_xy <= center_threshold)
            footprint_ok = bool(
                np.all(rel_lo >= -footprint_half)
                and np.all(rel_hi <= footprint_half)
            )
            ok = bool(center_ok and footprint_ok)
            all_correct = all_correct and ok
            per_object.append({
                "side": "robot",
                "object": obj_name,
                "model_id": int(model_id),
                "basket_idx": int(basket_idx),
                "object_pos": obj_center.tolist(),
                "basket_pos": basket_pos.tolist(),
                "dist_xy": dist_xy,
                "rel_aabb_lo_xy": rel_lo.tolist(),
                "rel_aabb_hi_xy": rel_hi.tolist(),
                "center_success": center_ok,
                "footprint_success": footprint_ok,
                "success": ok,
            })
        return {
            "basket_center_threshold": center_threshold,
            "basket_footprint_half_xy": float(footprint_half),
            "basket_all_correct": bool(all_correct),
            "basket_objects": per_object,
        }

    def check_success(self) -> bool:
        if not self.plan_success:
            return False
        metrics = self._robot_basket_metrics()
        if self.avatar_collision_checker is not None and self.avatar_collided:
            summary = self.avatar_collision_summary()
            self._neutral_debug(
                "[categorize_neutral] avatar collision detected: "
                f"first_frame={summary.get('first_collision_frame')} "
                f"deepest={summary.get('deepest_depth_m')}"
            )
            return False
        return bool(metrics["basket_all_correct"])

    def evaluate(self) -> dict:
        metrics = BaseTask.evaluate(self)
        metrics.update(self._robot_basket_metrics())
        metrics["neutral_robot_only_grading"] = True
        return metrics
