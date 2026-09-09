"""Minimal AvatarController: Tier 1 (play_animation, reset, step) + walk + turn."""
import os
import pickle as pkl
import numpy as np
import genesis as gs
from genesis.utils.misc import get_assets_dir, get_cvx_cache_dir

from .utils import Mixamo_data_to_controller_pose, Mixamo_node_processing
from .robot import AvatarRobot
from .motions.walk_motion import WalkMotion
from .motions.turn_motion import TurnMotion
from .motions.play_animation_motion import PlayAnimationMotion
from .motions.pick_place_motion import PickPlaceMotion
from .motions.styled_pick_place_motion import StyledPickPlaceMotion
from .motions.drawer_pull_motion import DrawerPullMotion
from .motions.transition_motion import TransitionMotion

# Default path for task-level generated motions, resolved relative to
# ASSETS_PATH by _resolve_path below.
DEFAULT_GENERATED_MOTIONS_PATH = "avatars/motions/generated_motions.pkl"


def _resolve_path(path: str, assets_dir: str) -> str:
    """Resolve path: if absolute, use as-is; else join with assets_dir."""
    if os.path.isabs(path):
        return path
    return os.path.join(assets_dir, path)


def _make_hold_reverse_motion(motion_data: dict, hold_at_frame: int,
                              hold_duration: int, reverse_to_frame: int = 0,
                              hold_end: int = 30) -> dict:
    """Return a forward-to-frame, hold, reverse-to-start motion clip."""
    n_frames = int(motion_data["trans"].shape[0])
    hold_at = int(np.clip(hold_at_frame, 0, n_frames - 1))
    reverse_to = int(np.clip(reverse_to_frame, 0, hold_at))
    hold_duration = max(0, int(hold_duration))
    hold_end = max(0, int(hold_end))
    indices = list(range(0, hold_at + 1))
    indices += [hold_at] * hold_duration
    indices += list(range(hold_at - 1, reverse_to - 1, -1))
    indices += [reverse_to] * hold_end
    idx = np.asarray(indices, dtype=np.int64)
    return {k: v[idx].copy() for k, v in motion_data.items()}


def _install_builtin_generated_motions(motion_data: dict) -> None:
    """Add lightweight derived motions used by benchmark tasks."""
    specs = [
        ("Inspect1", "Inspect1_touch_hold_reverse", 34, 60, 0, 30),
        ("Inspect", "Inspect_touch_hold_reverse", 34, 60, 0, 30),
        ("Inspect2", "Inspect2_touch_hold_reverse", 31, 60, 0, 30),
    ]
    for src, dst, hold_at, hold_duration, reverse_to, hold_end in specs:
        if dst in motion_data or src not in motion_data:
            continue
        motion_data[dst] = _make_hold_reverse_motion(
            motion_data[src],
            hold_at_frame=hold_at,
            hold_duration=hold_duration,
            reverse_to_frame=reverse_to,
            hold_end=hold_end,
        )


class AvatarController:
    """Minimal avatar controller for genesis. Tier 1 + walk + turn."""

    full_motion_data = None
    full_motion_data_path = None

    def __init__(
        self,
        scene,
        motion_data_path: str,
        skin_options=None,
        frame_ratio=1.0,
        name=None,
        assets_dir=None,
        generated_motion_path: str = None,
    ):
        self.scene = scene
        debug_avatar = bool(getattr(scene, "_debug_avatar_init", False))
        if debug_avatar:
            print("[avatar_init] begin", flush=True)

        _assets = assets_dir if assets_dir else get_assets_dir()
        motion_data_path = _resolve_path(motion_data_path, _assets)
        if skin_options:
            skin_options = dict(skin_options)
            skin_options["glb_path"] = _resolve_path(skin_options["glb_path"], _assets)

        if debug_avatar:
            print("[avatar_init] AvatarRobot begin", flush=True)
        self.robot = AvatarRobot(scene, skin_options, name)
        if debug_avatar:
            print("[avatar_init] AvatarRobot done", flush=True)
        self.box = self.robot.box
        if AvatarController.full_motion_data_path is None:
            AvatarController.full_motion_data_path = f"{motion_data_path}.full"

        if debug_avatar:
            print(f"[avatar_init] load motion {motion_data_path}", flush=True)
        with open(motion_data_path, "rb") as f:
            motion_data = pkl.load(f)
        if debug_avatar:
            print("[avatar_init] load motion done", flush=True)

        # Load generated motions (text-to-motion) from dedicated path and merge
        gen_path = generated_motion_path if generated_motion_path is not None else DEFAULT_GENERATED_MOTIONS_PATH
        if gen_path and gen_path != "":
            gen_full = gen_path if os.path.isabs(gen_path) else _resolve_path(gen_path, _assets)
            if os.path.isfile(gen_full):
                if debug_avatar:
                    print(f"[avatar_init] load generated {gen_full}", flush=True)
                with open(gen_full, "rb") as f:
                    generated = pkl.load(f)
                if debug_avatar:
                    print("[avatar_init] load generated done", flush=True)
                for k, v in generated.items():
                    if k not in motion_data:
                        motion_data[k] = v
                gs.logger.info(f"AvatarController: loaded {len(generated)} generated motions from {gen_full}")
            elif generated_motion_path is not None:
                gs.logger.warning(f"Generated motions file not found: {gen_full}")

        _install_builtin_generated_motions(motion_data)

        self.motion_data = motion_data
        self.frame_ratio = frame_ratio

        if debug_avatar:
            print("[avatar_init] process idle begin", flush=True)
        vgeom = self.robot.skin.links[0]._vgeoms[0]
        self.robot.stop_pose = Mixamo_data_to_controller_pose(
            motion_data["idle"]["trans"][0],
            motion_data["idle"]["rot"][0],
            motion_data["idle"]["joint"][0],
        )
        self.robot.stop_mat = motion_data["idle"]["mat"][0]
        self.robot.stop_mat_inv = np.array([np.linalg.inv(m) for m in self.robot.stop_mat])
        self.robot.stop_node = Mixamo_node_processing(
            vgeom, self.robot.stop_pose, self.robot.stop_mat, self.robot.stop_mat_inv
        )
        if debug_avatar:
            print("[avatar_init] process idle done", flush=True)

        if debug_avatar:
            print("[avatar_init] process stand begin", flush=True)
        self.robot.sit_pose = Mixamo_data_to_controller_pose(
            motion_data["stand"]["trans"][0],
            motion_data["stand"]["rot"][0],
            motion_data["stand"]["joint"][0],
        )
        self.robot.sit_mat = motion_data["stand"]["mat"][0]
        self.robot.sit_mat_inv = np.array([np.linalg.inv(m) for m in self.robot.sit_mat])
        self.robot.sit_node = Mixamo_node_processing(
            vgeom, self.robot.sit_pose, self.robot.sit_mat, self.robot.sit_mat_inv
        )
        if debug_avatar:
            print("[avatar_init] process stand done", flush=True)

        self.motion_modules = {}
        cache_name = (name or "default") + str(frame_ratio)

        if not bool(getattr(scene, "_skip_avatar_walk_modules", False)):
            if debug_avatar:
                print("[avatar_init] walk module begin", flush=True)
            # Walk
            if "walk" in motion_data:
                md = motion_data["walk"].copy()
                for k, v in md.items():
                    v_len = max(2, int(v.shape[0] * frame_ratio))
                    md[k] = v[np.round(np.linspace(0, v.shape[0] - 1, v_len)).astype(int)]
                self.motion_modules["walk"] = WalkMotion(
                    motion_name="walk", motion_data=md, robot=self.robot, name=cache_name
                )
            else:
                gs.logger.warning("motion_data has no 'walk' key")
            if debug_avatar:
                print("[avatar_init] walk module done", flush=True)

            if debug_avatar:
                print("[avatar_init] turn module begin", flush=True)
            # Turn (no motion_data)
            self.motion_modules["turn"] = TurnMotion(motion_name="turn", robot=self.robot)
            if debug_avatar:
                print("[avatar_init] turn module done", flush=True)
        elif debug_avatar:
            print("[avatar_init] skipped walk/turn modules", flush=True)

        self._keep_motion_name = None
        self._pending_idle_return_frames = None
        if debug_avatar:
            print("[avatar_init] done", flush=True)

    def reset(
        self,
        global_trans: np.ndarray = np.zeros(3, dtype=np.float64),
        global_rot: np.ndarray = np.eye(3, dtype=np.float64),
        update_mesh: bool = True,
    ):
        if not isinstance(global_trans, np.ndarray) or not isinstance(global_rot, np.ndarray):
            raise TypeError("Avatar global trans and global rot should be np.ndarray")
        self.robot.reset(global_trans, global_rot, update_mesh=update_mesh)
        self._keep_motion_name = None
        self._pending_idle_return_frames = None

    def get_global_pose(self):
        return self.robot.get_global_pose()

    def get_global_xy(self):
        return self.robot.get_global_xy()

    def get_global_height(self):
        return self.robot.get_global_height()

    def get_hand_pos(self, hand_id=1):
        """Return world position of the specified hand (0=left, 1=right)."""
        hand_pos, _ = self.robot._get_hand_ref(hand_id)
        return hand_pos

    def step(self, skip_avatar_animation=False):
        from .utils import AvatarState

        if self.robot.action_state == AvatarState.NO_ACTION:
            # After animation ends: still refresh skin + attached entities every sim step (hammer, etc.).
            if (
                any(x is not None for x in self.robot.attached_object)
                or self.robot.two_hand_attached_object is not None
            ):
                self.robot.update()
            return
        if self.robot.action_state in self.motion_modules:
            self.motion_modules[self.robot.action_state].step(skip_avatar_animation)
        self.robot.update()

        # If a motion just finished this step and the caller queued an auto
        # return-to-idle (via `play_animation(..., return_to_idle_after=N)`),
        # fire the transition now — so the avatar eases back to idle on its
        # own schedule instead of freezing on the last frame until the task
        # code polls spare().
        if (self.robot.action_state == AvatarState.NO_ACTION
                and self._pending_idle_return_frames is not None):
            frames = self._pending_idle_return_frames
            self._pending_idle_return_frames = None
            self.play_return_to_idle(frames=frames)

    def walk(self, distance, speed=1.0):
        from .utils import AvatarState

        if self.robot.base_state == AvatarState.SLEEPING:
            self.robot.base_state = AvatarState.STANDING
        if "walk" in self.motion_modules:
            self.motion_modules["walk"].start(distance, speed)

    def turn(self, angle, turn_frame_limit=15, turn_sec_limit=1500):
        from .utils import AvatarState

        if self.robot.base_state == AvatarState.SLEEPING:
            self.robot.base_state = AvatarState.STANDING
        self.motion_modules["turn"].start(angle, turn_frame_limit, turn_sec_limit)

    def attach_object_to_hand(self, entity, hand_id=1):
        """Attach an object entity to avatar hand immediately."""
        self.robot.attach_object_to_hand(hand_id, entity)
        self.robot.update()

    def _natural_pick_place_pose(
        self,
        pick_pos,
        place_pos,
        hand_id=None,
        body_margin=0.32,
        body_x_bounds=None,
        body_y_bounds=None,
        yaw_limit_deg=30.0,
    ):
        """Choose a standing pose that makes FABRIK pick/place look natural.

        This is an opt-in preconditioner for table-side reaching.  It does
        not change the arm IK itself; it moves/yaws the avatar body so the
        target midpoint is in front of the torso, then chooses the same-side
        hand when the caller did not force one.
        """
        pick = np.asarray(pick_pos, dtype=np.float64).ravel()[:3]
        place = np.asarray(place_pos, dtype=np.float64).ravel()[:3]
        target_xy = 0.5 * (pick[:2] + place[:2])

        base_pos = np.asarray(self.robot.global_trans, dtype=np.float64).copy()
        base_rot = np.asarray(self.robot.global_rot, dtype=np.float64).copy()
        forward_xy = base_rot[:, 0][:2]
        forward_xy = forward_xy / (np.linalg.norm(forward_xy) + 1e-9)

        def _pose_for_shift(s):
            """Body pos/rot for a lateral stance shift s (metres along the
            body's lateral axis), midpoint-backed-off as before."""
            lateral0 = base_rot[:, 1][:2]
            lateral0 = lateral0 / (np.linalg.norm(lateral0) + 1e-9)
            body_xy = target_xy - forward_xy * float(body_margin) + lateral0 * s
            if body_x_bounds is not None:
                body_xy[0] = float(
                    np.clip(body_xy[0], body_x_bounds[0], body_x_bounds[1])
                )
            if body_y_bounds is not None:
                body_xy[1] = float(
                    np.clip(body_xy[1], body_y_bounds[0], body_y_bounds[1])
                )
            body_pos = base_pos.copy()
            body_pos[:2] = body_xy
            desired = target_xy - body_pos[:2]
            desired_norm = np.linalg.norm(desired)
            if desired_norm > 1e-9:
                desired = desired / desired_norm
                cross = forward_xy[0] * desired[1] - forward_xy[1] * desired[0]
                dot = float(np.clip(np.dot(forward_xy, desired), -1.0, 1.0))
                yd = float(np.rad2deg(np.arctan2(cross, dot)))
                yd = float(np.clip(yd, -float(yaw_limit_deg), float(yaw_limit_deg)))
            else:
                yd = 0.0
            yaw = np.deg2rad(yd)
            cz, sz = np.cos(yaw), np.sin(yaw)
            rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]],
                          dtype=np.float64)
            return body_pos, rz @ base_rot, yd

        def _legacy():
            """Old behavior: stand at the midpoint, hand from pick side."""
            body_pos, body_rot, yd = _pose_for_shift(0.0)
            h = hand_id
            if h is None:
                lateral = body_rot[:, 1][:2]
                rel = pick[:2] - body_pos[:2]
                # hand_id 0=left, 1=right; standard table-facing pose maps
                # x>=body_x to left.
                h = 0 if float(np.dot(rel, lateral)) >= 0.0 else 1
            return body_pos, body_rot, int(h), yd

        # Shoulder offsets in the body frame (measured live so different
        # avatar rigs keep working; fall back to rig-typical constants).
        shoulder_local = {}
        try:
            for h, bone in ((0, "LeftArm"), (1, "RightArm")):
                sw = np.asarray(
                    self.robot.skin.get_global_translation(bone)[0],
                    dtype=np.float64,
                ).ravel()[:3]
                shoulder_local[h] = base_rot.T @ (sw - base_pos)
        except Exception:
            shoulder_local = {
                0: np.array([0.0, 0.17, 1.21]),
                1: np.array([0.0, -0.17, 1.21]),
            }

        # Joint search over (hand, lateral stance shift): keep BOTH endpoints
        # on the chosen hand's side of the body midline.  The old
        # stand-at-the-midpoint rule put pick and place on opposite sides by
        # construction, guaranteeing a cross-body carry (forearm through the
        # torso) whenever the span was wide.
        REACH_SOFT = 0.60   # penalise beyond this (docs: palm reach ~0.65)
        REACH_HARD = 0.68   # infeasible beyond this
        CROSS_MARGIN = 0.05  # want endpoints >=5cm onto the hand's side
        hands = (0, 1) if hand_id is None else (int(hand_id),)
        best = None
        for h in hands:
            side_sign = 1.0 if h == 0 else -1.0  # lateral axis points left
            for s in np.arange(-0.35, 0.351, 0.05):
                body_pos, body_rot, yd = _pose_for_shift(float(s))
                lateral = body_rot[:, 1][:2]
                lateral = lateral / (np.linalg.norm(lateral) + 1e-9)
                shoulder = body_pos + body_rot @ shoulder_local[h]
                cross_pen = 0.0
                reach_pen = 0.0
                feasible = True
                for p in (pick, place):
                    lat = side_sign * float(
                        np.dot(p[:2] - body_pos[:2], lateral)
                    )
                    cross_pen += max(0.0, CROSS_MARGIN - lat)
                    d = float(np.linalg.norm(p - shoulder))
                    if d > REACH_HARD:
                        feasible = False
                        break
                    reach_pen += max(0.0, d - REACH_SOFT)
                if not feasible:
                    continue
                score = 4.0 * cross_pen + 2.0 * reach_pen + 0.6 * abs(float(s))
                if best is None or score < best[0]:
                    best = (score, h, float(s), body_pos, body_rot, yd,
                            cross_pen)
        if best is None:
            # No reachable candidate (very tight bounds / wide span): keep
            # the proven legacy stance rather than inventing a worse one.
            print("[avatar] natural pose search infeasible — legacy stance",
                  flush=True)
            return _legacy()

        _, h, s, body_pos, body_rot, yaw_deg, cross_pen = best
        if cross_pen > 1e-6:
            print(
                f"[avatar] natural pose: residual cross-body "
                f"{cross_pen:.3f}m (hand={h}, shift={s:+.2f})",
                flush=True,
            )
        return body_pos, body_rot, int(h), yaw_deg

    def pick_and_place(self, pick_pos, place_pos, attach_obj=None, hand_id=1,
                       approach_frames=80, transport_frames=120,
                       retract_frames=100, approach_arc=0.15,
                       transport_arc=0.20, retract_arc=0.15, refine_iters=2,
                       natural=False, body_margin=0.32,
                       body_x_bounds=None, body_y_bounds=None,
                       yaw_limit_deg=30.0, settle_steps=0,
                       palm_down=True, palm_down_mode="reference",
                       palm_down_weight=1.0, palm_down_ramp_frac=0.5,
                       ease_in_out=True, attach_palm_offset=0.05):
        """Reach the palm to pick_pos, attach attach_obj, carry to place_pos.

        Generic task pick/place uses the legacy pure-FABRIK PickPlaceMotion
        path.  The Inspect2-styled variant remains available through the
        explicit ``styled_pick_and_place`` method, but it is not the default
        because several tabletop layouts need the stable arm-only behavior.

        palm_down (default on) post-rotates the hand so the palm faces the
        table during the reach/carry instead of fabrik's natural palm-up;
        ease_in_out (default on) smooths the per-phase pacing.
        """
        from .utils import AvatarState
        if self.robot.base_state == AvatarState.SLEEPING:
            self.robot.base_state = AvatarState.STANDING

        if natural:
            body_pos, body_rot, hand_id, yaw_deg = self._natural_pick_place_pose(
                pick_pos=pick_pos,
                place_pos=place_pos,
                hand_id=hand_id,
                body_margin=body_margin,
                body_x_bounds=body_x_bounds,
                body_y_bounds=body_y_bounds,
                yaw_limit_deg=yaw_limit_deg,
            )
            self.reset(body_pos, body_rot)
            for _ in range(max(0, int(settle_steps))):
                self.step()
            print(
                f"[avatar] natural legacy pick_place body={body_pos.round(3).tolist()} "
                f"yaw={yaw_deg:+.1f} hand={hand_id}",
                flush=True,
            )

        name = "pick_and_place"
        self.motion_modules[name] = PickPlaceMotion(name, self.robot)
        self.motion_modules[name].start(
            pick_pos=pick_pos, place_pos=place_pos,
            attach_obj=attach_obj, hand_id=hand_id,
            approach_frames=approach_frames,
            transport_frames=transport_frames,
            retract_frames=retract_frames,
            approach_arc=approach_arc,
            transport_arc=transport_arc,
            retract_arc=retract_arc,
            refine_iters=refine_iters,
            palm_down=palm_down,
            palm_down_mode=palm_down_mode,
            palm_down_weight=palm_down_weight,
            palm_down_ramp_frac=palm_down_ramp_frac,
            ease_in_out=ease_in_out,
            attach_palm_offset=attach_palm_offset,
        )
        return self.motion_modules[name]

    def styled_pick_and_place(self, pick_pos, place_pos, attach_obj=None, hand_id=1,
                              style_motion_name="Inspect2", attach_frame=None,
                              detach_frame=None, max_correction=0.22,
                              keyframe_extra_correction=0.30,
                              transport_arc=0.12, refine_iters=2,
                              style_residual_weight=0.65,
                              palm_down=False, palm_down_weight=1.0,
                              palm_down_mode="forearm_roll",
                              palm_down_contact_only=False,
                              palm_down_contact_window=18,
                              lock_root_motion=False,
                              lock_lower_body=False,
                              natural=False, body_margin=0.32,
                              body_x_bounds=None, body_y_bounds=None,
                              yaw_limit_deg=30.0, settle_steps=0):
        """Pick/place using an authored clip as style and IK as correction.

        Experimental: this does not replace ``pick_and_place``.  It is for
        visual review of tabletop avatar actions before tasks opt in.
        """
        from .utils import AvatarState
        if self.robot.base_state == AvatarState.SLEEPING:
            self.robot.base_state = AvatarState.STANDING
        if natural:
            body_pos, body_rot, hand_id, yaw_deg = self._natural_pick_place_pose(
                pick_pos=pick_pos,
                place_pos=place_pos,
                hand_id=hand_id,
                body_margin=body_margin,
                body_x_bounds=body_x_bounds,
                body_y_bounds=body_y_bounds,
                yaw_limit_deg=yaw_limit_deg,
            )
            self.reset(body_pos, body_rot)
            for _ in range(max(0, int(settle_steps))):
                self.step()
            print(
                f"[avatar] natural styled_pick_place body={body_pos.round(3).tolist()} "
                f"yaw={yaw_deg:+.1f} hand={hand_id}",
                flush=True,
            )

        if style_motion_name not in self.motion_data:
            if AvatarController.full_motion_data is None:
                AvatarController.full_motion_data = pkl.load(
                    open(AvatarController.full_motion_data_path, "rb")
                )
            if style_motion_name in AvatarController.full_motion_data:
                md = AvatarController.full_motion_data[style_motion_name].copy()
            else:
                print(f"Motion {style_motion_name} not found!")
                return None
        else:
            md = self.motion_data[style_motion_name].copy()

        for k, v in md.items():
            v_len = max(1, int(v.shape[0] * self.frame_ratio))
            md[k] = v[np.round(np.linspace(0, v.shape[0] - 1, v_len)).astype(int)]

        name = "styled_pick_and_place"
        self.motion_modules[name] = StyledPickPlaceMotion(name, md, self.robot)
        self.motion_modules[name].start(
            pick_pos=pick_pos,
            place_pos=place_pos,
            attach_obj=attach_obj,
            hand_id=hand_id,
            attach_frame=attach_frame,
            detach_frame=detach_frame,
            max_correction=max_correction,
            keyframe_extra_correction=keyframe_extra_correction,
            transport_arc=transport_arc,
            refine_iters=refine_iters,
            style_residual_weight=style_residual_weight,
            palm_down=palm_down,
            palm_down_weight=palm_down_weight,
            palm_down_mode=palm_down_mode,
            palm_down_contact_only=palm_down_contact_only,
            palm_down_contact_window=palm_down_contact_window,
            lock_root_motion=lock_root_motion,
            lock_lower_body=lock_lower_body,
        )
        return self.motion_modules[name]

    def pull_drawer(self, handle_start, handle_end, hand_id=0,
                    approach_frames=30, pull_frames=50, retract_frames=30,
                    retract_offset=(0.12, 0.02, 0.0), refine_iters=1):
        """Drive one palm along a drawer-handle pull path.

        This is a lightweight drawer-specific companion to pick_and_place:
        it solves IK at key poses and blends the skin transforms during
        replay, which is much cheaper for small handle motions.
        """
        from .utils import AvatarState
        if self.robot.base_state == AvatarState.SLEEPING:
            self.robot.base_state = AvatarState.STANDING
        name = "drawer_pull"
        self.motion_modules[name] = DrawerPullMotion(name, self.robot)
        self.motion_modules[name].start(
            handle_start=handle_start,
            handle_end=handle_end,
            hand_id=hand_id,
            approach_frames=approach_frames,
            pull_frames=pull_frames,
            retract_frames=retract_frames,
            retract_offset=retract_offset,
            refine_iters=refine_iters,
        )
        return self.motion_modules[name]

    def play_animation(self, name, attach_obj=None, hand_id=1, attach_frames_early=0,
                       attach_frame=None, detach_frame=None, return_to_idle_after=None,
                       attach_mode="hand", two_hand_offset=None, two_hand_snap=False):
        """Play animation by name. If attach_obj is set, attach it to hand at the chosen frame.

        attach_frame: absolute frame index (after frame_ratio resampling) for attachment.
        detach_frame: absolute frame index for detachment.
        return_to_idle_after: if set, automatically start a smooth transition back
            to the idle pose over this many frames the moment the animation ends
            (avoids freezing on the last frame while other code keeps stepping sim).
        """
        from .utils import AvatarState

        if self.robot.base_state == AvatarState.SLEEPING:
            self.robot.base_state = AvatarState.STANDING
        if name not in self.motion_data:
            if AvatarController.full_motion_data is None:
                AvatarController.full_motion_data = pkl.load(
                    open(AvatarController.full_motion_data_path, "rb")
                )
            if name in AvatarController.full_motion_data:
                md = AvatarController.full_motion_data[name].copy()
            else:
                print(f"Motion {name} not found!")
                return
        else:
            md = self.motion_data[name].copy()
        for k, v in md.items():
            v_len = max(1, int(v.shape[0] * self.frame_ratio))
            md[k] = v[np.round(np.linspace(0, v.shape[0] - 1, v_len)).astype(int)]
        self.motion_modules[name] = PlayAnimationMotion(
            motion_name=name, motion_data=md, robot=self.robot
        )
        self.motion_modules[name].start(
            attach_obj=attach_obj, hand_id=hand_id,
            attach_frames_early=attach_frames_early,
            attach_frame=attach_frame, detach_frame=detach_frame,
            attach_mode=attach_mode, two_hand_offset=two_hand_offset,
            two_hand_snap=two_hand_snap,
        )
        self._pending_idle_return_frames = (
            int(return_to_idle_after) if return_to_idle_after else None
        )

    def play_bowl_motion(self, attach_obj=None, attach_frame=None, detach_frame=None,
                         two_hand_offset=(0.0, 0.0, -0.015),
                         return_to_idle_after=None, snap_to_hands=True):
        """Play the clipped Bowl motion with a two-hand object attachment."""
        if attach_frame is None:
            attach_frame = int(round(45 * self.frame_ratio))
        if detach_frame is None:
            detach_frame = int(round(168 * self.frame_ratio))
        return self.play_animation(
            "Bowl_clip",
            attach_obj=attach_obj,
            attach_frame=attach_frame,
            detach_frame=detach_frame,
            return_to_idle_after=return_to_idle_after,
            attach_mode="two_hand",
            two_hand_offset=two_hand_offset,
            two_hand_snap=snap_to_hands,
        )

    def play_animation_hold_reverse(self, name, hold_at_frame, hold_duration,
                                       reverse_to_frame=1, hold_end=30, **kwargs):
        """Play animation to hold_at_frame, hold, reverse to reverse_to_frame, then hold at end.

        Args:
            hold_at_frame: frame index in the motion data to freeze at
            hold_duration: number of frames to hold
            reverse_to_frame: frame index to reverse back to
            hold_end: frames to hold at reverse_to_frame at the end
        """
        from .utils import AvatarState

        if self.robot.base_state == AvatarState.SLEEPING:
            self.robot.base_state = AvatarState.STANDING
        if name not in self.motion_data:
            if AvatarController.full_motion_data is None:
                AvatarController.full_motion_data = pkl.load(
                    open(AvatarController.full_motion_data_path, "rb")
                )
            if name in AvatarController.full_motion_data:
                md = AvatarController.full_motion_data[name].copy()
            else:
                print(f"Motion {name} not found!")
                return
        else:
            md = self.motion_data[name].copy()
        # Resample
        for k, v in md.items():
            v_len = max(1, int(v.shape[0] * self.frame_ratio))
            md[k] = v[np.round(np.linspace(0, v.shape[0] - 1, v_len)).astype(int)]

        # Build frame index sequence: forward → hold → reverse → hold_end
        indices = list(range(0, hold_at_frame + 1))
        indices += [hold_at_frame] * hold_duration
        indices += list(range(hold_at_frame, reverse_to_frame - 1, -1))
        indices += [reverse_to_frame] * hold_end

        # Rearrange all motion arrays
        for k, v in md.items():
            md[k] = v[np.array(indices)]

        mod_name = name + "_hold_rev"
        self.motion_modules[mod_name] = PlayAnimationMotion(
            motion_name=mod_name, motion_data=md, robot=self.robot
        )
        self.motion_modules[mod_name].start(**kwargs)

    def play_animation_reverse(self, name):
        """Play a previously loaded animation in reverse (last frame → first frame)."""
        from .utils import AvatarState

        if self.robot.base_state == AvatarState.SLEEPING:
            self.robot.base_state = AvatarState.STANDING
        if name not in self.motion_data:
            if AvatarController.full_motion_data is None:
                AvatarController.full_motion_data = pkl.load(
                    open(AvatarController.full_motion_data_path, "rb")
                )
            if name in AvatarController.full_motion_data:
                md = AvatarController.full_motion_data[name].copy()
            else:
                print(f"Motion {name} not found!")
                return
        else:
            md = self.motion_data[name].copy()
        # Reverse all frame arrays
        for k, v in md.items():
            v_len = max(1, int(v.shape[0] * self.frame_ratio))
            md[k] = v[np.round(np.linspace(0, v.shape[0] - 1, v_len)).astype(int)]
            md[k] = md[k][::-1].copy()
        rev_name = name + "_reverse"
        self.motion_modules[rev_name] = PlayAnimationMotion(
            motion_name=rev_name, motion_data=md, robot=self.robot
        )
        self.motion_modules[rev_name].start()

    def play_transition(self, target_pose, target_node, target_mat, target_mat_inv,
                        frames=30):
        """Smoothly blend from the avatar's current pose to a target pose.

        Used after a motion ends to return to idle (`play_return_to_idle`), or
        before `play_animation` to ease into a motion's first frame.
        """
        from .utils import AvatarState
        if self.robot.base_state == AvatarState.SLEEPING:
            self.robot.base_state = AvatarState.STANDING
        name = "transition"
        self.motion_modules[name] = TransitionMotion(
            robot=self.robot,
            target_pose=target_pose,
            target_node=target_node,
            target_mat=target_mat,
            target_mat_inv=target_mat_inv,
            frames=frames,
        )
        self.motion_modules[name].start()

    def play_return_to_idle(self, frames=30):
        """Smoothly return to the idle (stop) pose from wherever the avatar is."""
        self.play_transition(
            target_pose=self.robot.stop_pose,
            target_node=self.robot.stop_node,
            target_mat=self.robot.stop_mat,
            target_mat_inv=self.robot.stop_mat_inv,
            frames=frames,
        )

    def play_ease_into_animation(self, name, frames=30):
        """Smoothly blend from the current pose to the first frame of motion
        `name` (frame 0 of its resampled data); leaves the avatar at rest on
        that frame.  Caller then invokes `play_animation(name)` to play on.
        """
        if name not in self.motion_data:
            if AvatarController.full_motion_data is None:
                AvatarController.full_motion_data = pkl.load(
                    open(AvatarController.full_motion_data_path, "rb")
                )
            md_src = AvatarController.full_motion_data.get(name)
            if md_src is None:
                gs.logger.warning(f"play_ease_into_animation: motion {name} not found")
                return
        else:
            md_src = self.motion_data[name]
        trans0 = md_src["trans"][0]
        rot0 = md_src["rot"][0]
        joint0 = md_src["joint"][0]
        mat0 = md_src["mat"][0]
        mat0_inv = np.array([np.linalg.inv(m) for m in mat0])
        target_pose = Mixamo_data_to_controller_pose(trans0, rot0, joint0)
        vgeom = self.robot.skin.links[0]._vgeoms[0]
        target_node = Mixamo_node_processing(vgeom, target_pose, mat0, mat0_inv)
        self.play_transition(target_pose, target_node, mat0, mat0_inv, frames=frames)

    def spare(self):
        from .utils import AvatarState

        return self.robot.action_state == AvatarState.NO_ACTION

    def action_status(self):
        return self.robot.action_status
