"""Evaluate a VLA model on Genesis HR Bench tasks.

Usage:
    python scripts/eval_vla.py --task pour_water --model openvla_oft \
        --server-url http://127.0.0.1:8769 --action-type ee --episodes 10
"""

import argparse
import os
import sys
import yaml
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.eval import eval_policy
from scripts.vla_client import VLA_REGISTRY, get_task_instruction
from envs.tasks import TASK_MAP, resolve_task_class
from envs.scene_init import load_scene_init_file


VLA_DT_001_TASKS = {
    "categorize_cooperative",
    "take_from_human_easy",
    "take_from_human_safety",
}


VLA_COLLECTION_TASK_DEFAULTS = {
    # Matched to local VLA meta.json config_snapshot values where available.
    "pour_water": {"randomize_avatar": False, "randomize_table": False, "random_object": False},
    "take_from_human_easy": {"random_object": False},
    "deliver_to_human_easy": {"random_object": True},
    "categorize_cooperative": {"random_object": False},
    "put_object_cabinet_assist": {"random_object": False},
}


def _default_vla_action_substeps(task_name: str, target_hz: float = 20.0) -> int:
    sim_dt = 0.001 if task_name in VLA_DT_001_TASKS else 0.002
    return max(1, int(round(1.0 / (sim_dt * target_hz))))


def main():
    parser = argparse.ArgumentParser(description="Evaluate a VLA policy")
    parser.add_argument("--task", required=True, help="Task name from TASK_MAP")
    parser.add_argument("--model", required=True,
                        help="VLA model name — a key in scripts.vla_client.VLA_REGISTRY "
                             "(openvla_oft, pi0, pi05, pi0_fast, rdt, act, "
                             "vqbet, lerobot_diffusion, smolvla, ...)")
    parser.add_argument("--checkpoint", default=None,
                        help="Checkpoint path — unused for remote policies "
                             "(the server holds the checkpoint); required for "
                             "in-process policies.")
    parser.add_argument("--server-url", default=None,
                        help="For remote VLAs: HTTP URL of the model server, "
                             "e.g. http://127.0.0.1:8767")
    parser.add_argument("--action-type", default="qpos", choices=["qpos", "qpos_abs", "ee"])
    parser.add_argument("--dual-arm", action="store_true")
    parser.add_argument("--instruction", default=None, help="Override task instruction")
    parser.add_argument("--config", default=None, help="Base task config YAML")
    parser.add_argument("--vla-config", default=None, help="VLA config YAML override")
    parser.add_argument("--unnorm-key", default=None,
                        help="Action/proprio normalization stats key for remote VLA servers.")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--scene-init-file", default=None,
                        help="YAML/JSON/JSONL deterministic scene-init file. "
                             "When set, eval episodes are selected by index "
                             "instead of start-seed-only initialization.")
    parser.add_argument("--scene-init-start-index", type=int, default=0)
    parser.add_argument("--output-json", default=None, help="Save results to JSON")
    parser.add_argument("--video-dir", default=None,
                        help="If set, write one MP4 per episode to this directory")
    parser.add_argument("--video-limit", type=int, default=None,
                        help="When --video-dir is set, save videos only for the first N episodes.")
    parser.add_argument("--video-stride", type=int, default=None,
                        help="Capture one video frame every N sim steps when "
                             "--video-dir is set. Defaults to 10 for video "
                             "evals to avoid recording every physics substep.")
    parser.add_argument("--max-videos-per-dir", type=int, default=None,
                        help="When --video-dir is set, render at most this many "
                             "videos in the output directory; remaining episodes "
                             "still run without video capture.")
    parser.add_argument("--trace-actions-dir", default=None,
                        help="If set, write per-step action/qpos/object traces there.")
    parser.add_argument("--interpolate-qpos-actions", action="store_true",
                        help="Linearly interpolate qpos/qpos_abs arm and "
                             "gripper targets across each action_substeps window")
    parser.add_argument("--qpos-gripper-max-delta", type=float, default=None,
                        help="Clamp qpos/qpos_abs gripper command change per "
                             "policy action, e.g. 0.08 closes over many steps.")
    parser.add_argument("--qpos-gripper-close-hold-steps", type=int, default=None,
                        help="Extra sim steps to hold after each closing qpos "
                             "gripper command.")
    parser.add_argument("--qpos-gripper-immediate", action="store_true",
                        help="With --interpolate-qpos-actions, apply gripper "
                             "targets immediately while still interpolating arm qpos.")
    parser.add_argument("--qpos-gripper-pause-arm-on-close", action="store_true",
                        help="With qpos/qpos_abs actions, keep the arm fixed "
                             "during action substeps whose gripper target is closing.")
    parser.add_argument("--policy-gripper-kp", type=float, default=None,
                        help="Override eval-mode policy gripper finger kp.")
    parser.add_argument("--policy-gripper-kv", type=float, default=None,
                        help="Override eval-mode policy gripper finger kv.")
    parser.add_argument("--disable-eval-avatar-pre-policy", action="store_true",
                        help="For tasks that support it, do not run avatar "
                             "pre-policy warmup before the robot policy starts.")
    parser.add_argument("--disable-stop-on-success", action="store_true",
                        help="Do not terminate rollout early when check_success() "
                             "becomes true.")
    parser.add_argument("--success-hold-steps", type=int, default=None,
                        help="Require success for N consecutive policy steps before "
                             "early termination. Default: 1.")
    parser.add_argument("--act-gripper-closed-cmd", type=float, default=None,
                        help="Override ACT qpos/qpos_abs snapped closed command.")
    parser.add_argument("--track-avatar-collision", action="store_true")
    parser.add_argument("--no-human", action="store_true",
                        help="Collapse an assist/interrupt/neutral family task to "
                             "its canonical robot-only no-human task. Intent tasks "
                             "do not support this flag.")
    parser.add_argument("--static-avatar", action="store_true",
                        help="Evaluate base/no-human task behavior while rendering "
                             "a still avatar in the scene.")
    parser.add_argument("--static-avatar-pos", type=float, nargs=3,
                        default=[0.0, 0.55, -0.18],
                        metavar=("X", "Y", "Z"),
                        help="Still avatar body position for --static-avatar.")
    parser.add_argument("--hard", action="store_true",
                        help="Use the legacy multi-object task variant where supported.")
    parser.add_argument("--table-random-objects",
                        "--task-irrelevant-objects",
                        "--with-irrelevant-objects",
                        dest="table_random_objects",
                        action="store_true",
                        help="Enable random background tabletop objects. Default: disabled.")
    parser.add_argument("--no-table-random-objects",
                        "--no-task-irrelevant-objects",
                        dest="no_table_random_objects",
                        action="store_true",
                        help="Disable random background tabletop objects, overriding config.")
    parser.add_argument("--avatar-safety-margin", type=float, default=None)
    parser.add_argument("--avatar-retreat-enabled", action="store_true")
    parser.add_argument("--avatar-retreat-margin", type=float, default=None)
    parser.add_argument("--pace-interrupt", action="store_true",
                        help="For supported interrupt tasks, trigger the "
                             "avatar interruption from action-completion "
                             "progress instead of a random policy step.")
    parser.add_argument("--pace-trigger-progress", type=float, default=None,
                        help="PACE progress threshold in [0,1]. Default is "
                             "task-specific.")
    parser.add_argument("--pace-trigger-dist-xy", type=float, default=None,
                        help="PACE fallback EE-to-object xy distance trigger "
                             "in meters. Default is task-specific.")
    parser.add_argument("--pace-min-policy-steps", type=int, default=None,
                        help="Minimum policy steps before a PACE interrupt "
                             "can fire. Default is task-specific.")
    parser.add_argument("--pace-no-human-ablation", action="store_true",
                        help="Keep the interrupt task class but disable the "
                             "avatar, so PACE only performs robot "
                             "retreat-return on the base scene.")
    parser.add_argument("--pace-visual-model", default=None,
                        help="Checkpoint for learned visual PACE progress detector.")
    parser.add_argument("--pace-visual-generic-only", action="store_true",
                        help="Use the visual detector only for eval-level "
                             "generic retreat/return, without switching the "
                             "task's own interrupt trigger mode to visual_pace.")
    parser.add_argument("--pace-visual-trigger-progress", type=float, default=None)
    parser.add_argument("--pace-visual-present-threshold", type=float, default=None)
    parser.add_argument("--pace-visual-active-threshold", type=float, default=None)
    parser.add_argument("--pace-visual-motion-threshold", type=float, default=None)
    parser.add_argument("--pace-visual-motion-steps-to-full", type=float, default=None)
    parser.add_argument("--pace-visual-min-updates", type=int, default=None)
    parser.add_argument("--pace-visual-human-start-step", type=int, default=None,
                        help="For supported interrupt eval tasks, start the "
                             "avatar event independently; the learned visual "
                             "detector decides when the robot yields.")
    parser.add_argument("--pace-visual-update-stride", type=int, default=None,
                        help="Run the visual detector every N policy steps.")
    parser.add_argument("--manicast-model", default=None,
                        help="STS-GCN avatar-motion forecaster checkpoint; "
                             "enables the ManiCast-ACT anticipatory yield "
                             "trigger (observation-only).")
    parser.add_argument("--manicast-pose-model", default=None,
                        help="AvatarPoseNet checkpoint (RGB -> avatar joints) "
                             "required by --manicast-model.")
    parser.add_argument("--manicast-trigger-dist", type=float, default=None,
                        help="Yield when the forecast wrist comes within this "
                             "distance (m) of the robot TCP.")
    parser.add_argument("--manicast-clear-dist", type=float, default=None)
    parser.add_argument("--manicast-clear-consecutive", type=int, default=None)
    parser.add_argument("--manicast-presence-threshold", type=float, default=None)
    parser.add_argument("--manicast-min-updates", type=int, default=None)
    parser.add_argument("--manicast-horizon-steps", type=int, default=None)
    parser.add_argument("--pace-generic-wait-mode",
                        choices=["avatar_done", "detector"], default=None,
                        help="End-of-wait source for the generic yield: task "
                             "avatar state (default) or detector-based "
                             "observation-only clearance.")
    parser.add_argument("--pace-generic-retreat-return", action="store_true",
                        help="Use an eval-level visual-PACE yield wrapper: "
                             "retreat, return to the saved qpos, reset policy.")
    parser.add_argument("--pace-generic-hold-position", action="store_true",
                        help="Use an eval-level visual-PACE yield wrapper "
                             "that holds the current qpos until the avatar "
                             "motion/event is done, then resumes policy.")
    parser.add_argument("--pace-generic-retreat-steps", type=int, default=None)
    parser.add_argument("--pace-generic-return-steps", type=int, default=None)
    parser.add_argument("--pace-generic-return-max-steps", type=int, default=None)
    parser.add_argument("--pace-generic-return-qpos-tol", type=float, default=None)
    parser.add_argument("--pace-generic-return-correction-gain", type=float, default=None)
    parser.add_argument("--pace-generic-return-correction-clip", type=float, default=None)
    parser.add_argument("--pace-generic-no-wait-avatar-done", action="store_true",
                        help="Do not hold the generic yield until the task "
                             "avatar event/motion is done.")
    parser.add_argument("--pace-generic-wait-max-steps", type=int, default=None)
    parser.add_argument("--pace-generic-retreat-blend", type=float, default=None)
    parser.add_argument("--pace-generic-return-blend", type=float, default=None)
    parser.add_argument("--pace-generic-no-policy-reset", action="store_true",
                        help="After generic retreat/return, do not reset the "
                             "remote policy action chunk cache.")
    parser.add_argument("--pace-generic-resume-snapshot-obs", action="store_true",
                        help="After generic retreat/return, feed the saved "
                             "pre-interrupt observation to the next policy call.")
    parser.add_argument("--pace-generic-restore-scene-state", action="store_true",
                        help="After generic retreat/return, restore task "
                             "object poses captured at the trigger step.")
    parser.add_argument("--pace-generic-no-trigger-action-capture", action="store_true",
                        help="Do not call the policy at the trigger step for "
                             "generic yield; keeps ACT chunk time frozen while "
                             "the robot is held.")
    parser.add_argument("--hri-initiative-mode",
                        choices=["none", "oracle", "reactive", "proactive"],
                        default=None,
                        help="Baraglia-style initiative trigger mode. The "
                             "trigger starts the existing generic yield "
                             "controller; use with --pace-generic-retreat-return "
                             "or --pace-generic-hold-position.")
    parser.add_argument("--hri-initiative-step", type=int, default=None,
                        help="Fixed proactive trigger step. Also used as "
                             "fallback for oracle when a task exposes no "
                             "event step.")
    parser.add_argument("--hri-initiative-min-steps", type=int, default=None,
                        help="Minimum policy steps before any initiative "
                             "trigger can fire.")
    parser.add_argument("--hri-initiative-oracle-offset-steps", type=int, default=None,
                        help="Offset applied to task event step in oracle "
                             "mode. Negative values trigger before the event.")
    parser.add_argument("--hri-initiative-motion-threshold", type=float, default=None,
                        help="Reactive visual motion threshold.")
    parser.add_argument("--hri-initiative-progress-threshold", type=float, default=None,
                        help="Reactive visual progress threshold.")
    parser.add_argument("--hri-initiative-proximity-threshold", type=float, default=None,
                        help="Reactive safe-distance threshold in meters.")
    parser.add_argument("--hri-phase-controller", default=None,
                        help="Pickle checkpoint for the learned PbD HRI phase "
                             "controller. Its yield prediction starts the "
                             "generic retreat/wait/return wrapper.")
    parser.add_argument("--hri-phase-min-confidence", type=float, default=None,
                        help="Minimum phase prediction confidence before "
                             "triggering generic yield.")
    parser.add_argument("--hri-phase-min-step", type=int, default=None,
                        help="Minimum policy step before the learned phase "
                             "controller can trigger.")
    parser.add_argument("--hri-phase-consecutive", type=int, default=None,
                        help="Require N consecutive trigger-label predictions.")
    parser.add_argument("--hri-phase-trigger-labels", default=None,
                        help="Comma-separated labels that trigger yield. "
                             "Default: yield.")
    parser.add_argument("--pace-disable-task-robot-yield", action="store_true",
                        help="For supported interrupt tasks, keep the task "
                             "avatar event but disable the task's own robot "
                             "retreat/wait/return controller.")
    parser.add_argument("--pace-no-pan-ready", action="store_true",
                        help="For supported skillet tasks, allow PACE to "
                             "trigger before the pan is near the cooktop.")
    parser.add_argument("--pace-pan-ready-xy", type=float, default=None,
                        help="For supported skillet tasks, xy tolerance in "
                             "meters for considering the pan ready.")
    parser.add_argument("--static-pan", action="store_true",
                        help="For supported skillet tasks, start the pan on "
                             "the cooktop instead of forcing a dynamic pan.")
    parser.add_argument("--no-vla-train-defaults", action="store_true",
                        help="Do not apply VLA collection defaults such as "
                             "20 Hz action_substeps, randomized layout, and "
                             "VLA top-down head camera.")
    args = parser.parse_args()

    config = {}
    default_config = os.path.join(os.path.dirname(__file__), "..", "config", "default.yml")
    if os.path.exists(default_config):
        with open(default_config) as f:
            config = yaml.safe_load(f) or {}
    if args.config:
        with open(args.config) as f:
            config.update(yaml.safe_load(f) or {})

    vla_config = {}
    vla_default = os.path.join(os.path.dirname(__file__), "..", "config", "vla.yml")
    if os.path.exists(vla_default):
        with open(vla_default) as f:
            vla_config = (yaml.safe_load(f) or {}).get("vla", {})
    if args.vla_config:
        with open(args.vla_config) as f:
            vla_config.update((yaml.safe_load(f) or {}).get("vla", {}))

    vla_config["model_name"] = args.model
    vla_config["checkpoint_path"] = args.checkpoint
    vla_config["action_type"] = args.action_type
    vla_config["dual_arm"] = args.dual_arm
    if args.server_url:
        vla_config["server_url"] = args.server_url
    if args.unnorm_key is not None:
        vla_config["unnorm_key"] = args.unnorm_key
    if args.act_gripper_closed_cmd is not None:
        vla_config["act_gripper_closed_cmd"] = float(args.act_gripper_closed_cmd)

    # VLA baselines always plan with Genesis damped-least-squares IK: it stays
    # near current qpos on every single-step EE move, which keeps the
    # benchmark's PD loop from amplifying branch jumps (cuRobo/mplib pick
    # globally-optimal IK solutions that can be far from the current pose,
    # producing runaway under high-frequency servoing). BaseTask only applies
    # this override when `planner_override` is set, so `scripts/collect.py`
    # (which never sets it) is unaffected.
    config.setdefault("planner_override", "genesis_ik")
    config.setdefault("table_random_objects", False)
    config.setdefault("task_irrelevant_objects", False)
    if args.table_random_objects:
        config["table_random_objects"] = True
        config["task_irrelevant_objects"] = True
    if args.no_table_random_objects:
        config["table_random_objects"] = False
        config["task_irrelevant_objects"] = False

    if not args.no_vla_train_defaults:
        # Match the VLA training-data collection surface. The raw VLA dataset
        # records at 20 Hz, but some tasks use dt=0.001 while BaseTask defaults
        # to dt=0.002; action_substeps must therefore be task-dependent.
        config.setdefault("action_substeps", _default_vla_action_substeps(args.task))
        # Apply task-specific collection metadata before generic defaults.
        # Explicit config files still win, while known task metadata (for
        # example pour_water's fixed avatar/table layout) can override broad
        # VLA defaults.
        task_defaults = VLA_COLLECTION_TASK_DEFAULTS.get(args.task, {})
        for key, value in task_defaults.items():
            config.setdefault(key, value)
        # Collection used the VLA top-down head camera via
        # vla_recording.opposite_head_camera. During eval we need the same
        # camera without enabling a recorder.
        config.setdefault("vla_head_camera", True)
        config.setdefault("randomize_avatar", True)
        config.setdefault("randomize_table", True)
        config.setdefault("track_avatar_collision", True)

    if args.static_avatar:
        config["static_avatar"] = True
        config["no_human"] = True
        config["randomize_avatar"] = False
        config["randomize_table"] = False
        config["random_object"] = False
        config["track_avatar_collision"] = False
        config["use_avatar_collider"] = False
        config["static_avatar_as_planner_obstacle"] = False
        config["static_avatar_pos"] = list(args.static_avatar_pos)
        config["eval_mode"] = False

    if args.no_human:
        config["no_human"] = True
        config["randomize_avatar"] = False
        config["track_avatar_collision"] = False
        config["avatar_retreat_enabled"] = False
        config["eval_mode"] = False

    # Turn on the avatar's eval-mode interrupt state machine. Interrupt
    # tasks (dump_bin_interrupt, place_*_interrupt, …) override `take_action`
    # to drive the avatar via `_eval_step_avatar` only when this flag is on
    # (see envs/avatar/eval_mode_mixin.py). Without it the avatar stands
    # idle during policy eval — i.e. the "interrupt" never happens, which
    # makes the policy's task look easier than the one it was trained on.
    # For non-interrupt tasks (pour_water, …) the EvalModeMixin's
    # `_eval_at_step` is a no-op, so this is safe everywhere.
    if not args.no_human and not args.static_avatar:
        config.setdefault("eval_mode", True)

    resolved_task_name, TaskClass = resolve_task_class(
        args.task,
        no_human=bool(config.get("no_human", False) or config.get("static_avatar", False)),
    )
    instruction = (
        args.instruction
        or (TaskClass.default_instruction() if TaskClass is not None
            and hasattr(TaskClass, "default_instruction") else None)
        or get_task_instruction(resolved_task_name)
    )
    # Only force the task's instruction on an explicit override; otherwise let
    # the task template its per-episode object name into self.instruction, which
    # the eval loop feeds to the policy each reset via set_instruction.
    if args.instruction:
        config["instruction"] = args.instruction
    if (
        not args.no_human
        and (args.track_avatar_collision
             or args.avatar_safety_margin is not None
             or args.avatar_retreat_enabled)
    ):
        config["track_avatar_collision"] = True
    if args.hard:
        config["hard"] = True
        config["difficulty"] = "hard"
    if args.avatar_safety_margin is not None:
        config["avatar_safety_margin"] = float(args.avatar_safety_margin)
    if args.avatar_retreat_enabled:
        config["avatar_retreat_enabled"] = True
    if args.avatar_retreat_margin is not None:
        config["avatar_retreat_margin"] = float(args.avatar_retreat_margin)
    if args.pace_interrupt:
        config["eval_interrupt_trigger_mode"] = "pace"
    if args.pace_no_human_ablation:
        config["pace_no_human_ablation"] = True
        config["no_avatar"] = True
        config["randomize_avatar"] = False
        config["track_avatar_collision"] = False
        config["avatar_retreat_enabled"] = False
        config["eval_mode"] = True
    if args.pace_visual_model:
        if not args.pace_visual_generic_only:
            config["eval_interrupt_trigger_mode"] = "visual_pace"
        config["pace_visual_model"] = args.pace_visual_model
    if args.pace_visual_trigger_progress is not None:
        config["pace_visual_trigger_progress"] = float(args.pace_visual_trigger_progress)
    if args.pace_visual_present_threshold is not None:
        config["pace_visual_present_threshold"] = float(args.pace_visual_present_threshold)
    if args.pace_visual_active_threshold is not None:
        config["pace_visual_active_threshold"] = float(args.pace_visual_active_threshold)
    if args.pace_visual_motion_threshold is not None:
        config["pace_visual_motion_threshold"] = float(args.pace_visual_motion_threshold)
    if args.pace_visual_motion_steps_to_full is not None:
        config["pace_visual_motion_steps_to_full"] = float(args.pace_visual_motion_steps_to_full)
    if args.pace_visual_min_updates is not None:
        config["pace_visual_min_updates"] = max(0, int(args.pace_visual_min_updates))
    if args.pace_visual_human_start_step is not None:
        config["pace_visual_human_start_step"] = max(0, int(args.pace_visual_human_start_step))
    if args.pace_visual_update_stride is not None:
        config["pace_visual_update_stride"] = max(1, int(args.pace_visual_update_stride))
    if args.manicast_model:
        if not args.manicast_pose_model:
            raise SystemExit("--manicast-model requires --manicast-pose-model")
        config["manicast_model"] = args.manicast_model
        config["manicast_pose_model"] = args.manicast_pose_model
    if args.manicast_trigger_dist is not None:
        config["manicast_trigger_dist"] = float(args.manicast_trigger_dist)
    if args.manicast_clear_dist is not None:
        config["manicast_clear_dist"] = float(args.manicast_clear_dist)
    if args.manicast_clear_consecutive is not None:
        config["manicast_clear_consecutive"] = max(1, int(args.manicast_clear_consecutive))
    if args.manicast_presence_threshold is not None:
        config["manicast_presence_threshold"] = float(args.manicast_presence_threshold)
    if args.manicast_min_updates is not None:
        config["manicast_min_updates"] = max(0, int(args.manicast_min_updates))
    if args.manicast_horizon_steps is not None:
        config["manicast_horizon_steps"] = max(1, int(args.manicast_horizon_steps))
    if args.pace_generic_wait_mode is not None:
        config["pace_generic_wait_mode"] = args.pace_generic_wait_mode
    if args.pace_generic_retreat_return:
        config["pace_generic_retreat_return"] = True
    if args.pace_generic_hold_position:
        config["pace_generic_hold_position"] = True
    if args.pace_generic_retreat_steps is not None:
        config["pace_generic_retreat_steps"] = max(0, int(args.pace_generic_retreat_steps))
    if args.pace_generic_return_steps is not None:
        config["pace_generic_return_steps"] = max(0, int(args.pace_generic_return_steps))
    if args.pace_generic_return_max_steps is not None:
        config["pace_generic_return_max_steps"] = max(1, int(args.pace_generic_return_max_steps))
    if args.pace_generic_return_qpos_tol is not None:
        config["pace_generic_return_qpos_tol"] = float(args.pace_generic_return_qpos_tol)
    if args.pace_generic_return_correction_gain is not None:
        config["pace_generic_return_correction_gain"] = float(args.pace_generic_return_correction_gain)
    if args.pace_generic_return_correction_clip is not None:
        config["pace_generic_return_correction_clip"] = float(args.pace_generic_return_correction_clip)
    if args.pace_generic_no_wait_avatar_done:
        config["pace_generic_wait_avatar_done"] = False
    if args.pace_generic_wait_max_steps is not None:
        config["pace_generic_wait_max_steps"] = max(1, int(args.pace_generic_wait_max_steps))
    if args.pace_generic_retreat_blend is not None:
        config["pace_generic_retreat_blend"] = float(args.pace_generic_retreat_blend)
    if args.pace_generic_return_blend is not None:
        config["pace_generic_return_blend"] = float(args.pace_generic_return_blend)
    if args.pace_generic_no_policy_reset:
        config["pace_generic_reset_policy_after_return"] = False
    if args.pace_generic_resume_snapshot_obs:
        config["pace_generic_resume_snapshot_obs"] = True
    if args.pace_generic_restore_scene_state:
        config["pace_generic_restore_scene_state"] = True
    if args.pace_generic_no_trigger_action_capture:
        config["pace_generic_capture_trigger_action"] = False
    if args.hri_initiative_mode is not None:
        config["hri_initiative_mode"] = args.hri_initiative_mode
    if args.hri_initiative_step is not None:
        config["hri_initiative_step"] = max(0, int(args.hri_initiative_step))
    if args.hri_initiative_min_steps is not None:
        config["hri_initiative_min_steps"] = max(0, int(args.hri_initiative_min_steps))
    if args.hri_initiative_oracle_offset_steps is not None:
        config["hri_initiative_oracle_offset_steps"] = int(args.hri_initiative_oracle_offset_steps)
    if args.hri_initiative_motion_threshold is not None:
        config["hri_initiative_motion_threshold"] = float(args.hri_initiative_motion_threshold)
    if args.hri_initiative_progress_threshold is not None:
        config["hri_initiative_progress_threshold"] = float(args.hri_initiative_progress_threshold)
    if args.hri_initiative_proximity_threshold is not None:
        config["hri_initiative_proximity_threshold"] = float(args.hri_initiative_proximity_threshold)
    if args.hri_phase_controller:
        config["hri_phase_controller"] = args.hri_phase_controller
    if args.hri_phase_min_confidence is not None:
        config["hri_phase_min_confidence"] = float(args.hri_phase_min_confidence)
    if args.hri_phase_min_step is not None:
        config["hri_phase_min_step"] = max(0, int(args.hri_phase_min_step))
    if args.hri_phase_consecutive is not None:
        config["hri_phase_consecutive"] = max(1, int(args.hri_phase_consecutive))
    if args.hri_phase_trigger_labels is not None:
        config["hri_phase_trigger_labels"] = args.hri_phase_trigger_labels
    if args.pace_disable_task_robot_yield:
        config["eval_robot_yield_enabled"] = False
    if args.pace_trigger_progress is not None:
        config["pace_trigger_progress"] = float(args.pace_trigger_progress)
    if args.pace_trigger_dist_xy is not None:
        config["pace_trigger_dist_xy"] = float(args.pace_trigger_dist_xy)
    if args.pace_min_policy_steps is not None:
        config["pace_min_policy_steps"] = max(0, int(args.pace_min_policy_steps))
    if args.pace_no_pan_ready:
        config["pace_require_pan_ready"] = False
    if args.pace_pan_ready_xy is not None:
        config["pace_pan_ready_xy"] = float(args.pace_pan_ready_xy)
    if args.static_pan:
        config["force_dynamic_pan"] = False
    if args.video_stride is not None:
        config["video_stride"] = max(1, int(args.video_stride))
    elif args.video_dir:
        config.setdefault("video_stride", 10)
    if args.max_videos_per_dir is not None:
        config["max_videos_per_dir"] = max(0, int(args.max_videos_per_dir))
    if args.interpolate_qpos_actions:
        config["interpolate_qpos_actions"] = True
    if args.qpos_gripper_max_delta is not None:
        config["qpos_gripper_max_delta_per_action"] = float(args.qpos_gripper_max_delta)
    if args.qpos_gripper_close_hold_steps is not None:
        config["qpos_gripper_close_hold_steps"] = max(0, int(args.qpos_gripper_close_hold_steps))
    if args.qpos_gripper_immediate:
        config["qpos_gripper_immediate"] = True
    if args.qpos_gripper_pause_arm_on_close:
        config["qpos_gripper_pause_arm_on_close"] = True
    if args.policy_gripper_kp is not None:
        config["policy_gripper_kp"] = float(args.policy_gripper_kp)
    if args.policy_gripper_kv is not None:
        config["policy_gripper_kv"] = float(args.policy_gripper_kv)
    if args.disable_eval_avatar_pre_policy:
        config["eval_avatar_pre_policy"] = False
    if args.disable_stop_on_success:
        config["eval_stop_on_success"] = False
    if args.success_hold_steps is not None:
        config["eval_success_hold_steps"] = max(1, int(args.success_hold_steps))
    if args.trace_actions_dir:
        config["trace_actions_dir"] = args.trace_actions_dir

    if args.model not in VLA_REGISTRY:
        raise ValueError(f"Unknown model: {args.model}. Available: {list(VLA_REGISTRY.keys())}")

    PolicyClass = VLA_REGISTRY[args.model]
    policy = PolicyClass(
        checkpoint_path=args.checkpoint,
        task_instruction=instruction,
        config=vla_config,
        action_type=args.action_type,
        dual_arm=args.dual_arm,
    )
    policy.load_model()

    scene_inits = load_scene_init_file(args.scene_init_file) if args.scene_init_file else None
    eval_results = eval_policy(
        resolved_task_name, config, policy,
        args.episodes, args.max_steps, args.start_seed,
        action_type=args.action_type,
        video_dir=args.video_dir,
        video_limit=args.video_limit,
        return_details=True,
        scene_inits=scene_inits,
        scene_init_start_index=args.scene_init_start_index,
    )
    success_rate = float(eval_results["success_rate"])

    if args.output_json:
        results = {
            "task": resolved_task_name,
            "model": args.model,
            "checkpoint": args.checkpoint,
            "action_type": args.action_type,
            "dual_arm": args.dual_arm,
            "episodes": args.episodes,
            "success_rate": success_rate,
            "successes": eval_results.get("successes"),
            "episode_results": eval_results.get("episode_results", []),
        }
        if args.scene_init_file:
            results["scene_init_file"] = args.scene_init_file
            results["scene_init_start_index"] = args.scene_init_start_index
        output_parent = os.path.dirname(args.output_json)
        if output_parent:
            os.makedirs(output_parent, exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output_json}")


if __name__ == "__main__":
    main()
