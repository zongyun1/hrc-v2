#!/usr/bin/env python3
"""Download an ACT checkpoint from Hugging Face and submit eval shards.

Example:
  python scripts/eval_act_hf.py --task pour_water --submit-jobs 5

By default this downloads the latest available step for the task from
Miiche/act-genesis-hr-bench, then submits 5 Slurm jobs. Each job runs 2 eval
processes x 5 episodes, so 5 jobs cover seeds 0-49.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


REPO_ROOT = Path(__file__).resolve().parents[1]
HF_REPO_ID = "Miiche/act-genesis-hr-bench"
HF_LOCAL_ROOT = REPO_ROOT / "runs" / "act" / "hf_models" / "Miiche_act-genesis-hr-bench"
EVAL_ROOT = REPO_ROOT / "runs" / "act" / "eval_hf"
EVAL_SCRIPT = "scripts/launch/eval_act_seed_ranges_multi_fixed_out.sbatch"

DEFAULT_EVAL_STEPS = 1200
LOCAL_ACT_UNSUPPORTED_CONFIG_KEYS = {
    # Newer LeRobot checkpoint metadata not accepted by the local ACTConfig.
    "use_peft",
    "push_to_hub",
    "repo_id",
    "private",
    "tags",
    "license",
    "pretrained_path",
}
TASK_EVAL_STEPS = {
    "dump_bin": 744,
    "place_bread_in_basket": 478,
    "place_burger_fries": 692,
    "place_dual_shoes": 674,
    "place_food_in_skillet": 708,
    "put_object_cabinet": 1530,
    "stack_bowls_three": 420,
    "take_from_human_easy": 390,
    "deliver_to_human_easy": 544,
    "take_from_human_safety": 800,
    "stamp_documents": 1846,
    "pour_water": 406,
    "oil_bottle_recovery": 1078,
    "open_microwave": 842,
    "put_object_cabinet_assist": 824,
}


def run(cmd: list[str], *, dry_run: bool = False) -> subprocess.CompletedProcess | None:
    print("+ " + " ".join(cmd), flush=True)
    if dry_run:
        return None
    return subprocess.run(cmd, cwd=REPO_ROOT, text=True, capture_output=True, check=True)


def repo_files(repo_id: str) -> list[str]:
    return HfApi().list_repo_files(repo_id=repo_id, repo_type="model")


def available_checkpoints(files: list[str]) -> dict[str, list[int]]:
    out: dict[str, set[int]] = {}
    pattern = re.compile(r"^([^/]+)/step_(\d+)/model\.safetensors$")
    for path in files:
        match = pattern.match(path)
        if not match:
            continue
        task, step = match.group(1), int(match.group(2))
        out.setdefault(task, set()).add(step)
    return {task: sorted(steps) for task, steps in sorted(out.items())}


def latest_step(files: list[str], task: str, requested_step: int | None) -> int:
    checkpoints = available_checkpoints(files)
    if task not in checkpoints:
        choices = ", ".join(checkpoints)
        raise SystemExit(f"task {task!r} not found in {HF_REPO_ID}; available: {choices}")
    if requested_step is not None:
        if requested_step not in checkpoints[task]:
            raise SystemExit(f"{task} has steps {checkpoints[task]}, not {requested_step}")
        return requested_step
    return checkpoints[task][-1]


def download_checkpoint(repo_id: str, task: str, step: int, local_root: Path) -> Path:
    checkpoint_dir = local_root / task / f"step_{step}"
    required = [
        "config.json",
        "model.safetensors",
        "train_config.json",
        "policy_preprocessor.json",
        "policy_preprocessor_step_3_normalizer_processor.safetensors",
        "policy_postprocessor.json",
        "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
    ]
    if all((checkpoint_dir / name).exists() for name in required):
        print(f"[eval_act_hf] reusing {checkpoint_dir}")
        sanitize_act_config(checkpoint_dir)
        return checkpoint_dir

    local_root.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        allow_patterns=[f"{task}/step_{step}/*"],
        local_dir=local_root,
        local_dir_use_symlinks=False,
    )
    missing = [name for name in required if not (checkpoint_dir / name).exists()]
    if missing:
        raise SystemExit(f"download incomplete for {checkpoint_dir}; missing {missing}")
    sanitize_act_config(checkpoint_dir)
    return checkpoint_dir


def sanitize_act_config(checkpoint_dir: Path) -> None:
    config_path = checkpoint_dir / "config.json"
    data = json.loads(config_path.read_text())
    removed = sorted(key for key in LOCAL_ACT_UNSUPPORTED_CONFIG_KEYS if key in data)
    if not removed:
        inject_old_lerobot_stats(checkpoint_dir)
        disable_new_lerobot_processors(checkpoint_dir)
        return
    for key in removed:
        data.pop(key, None)
    backup = checkpoint_dir / "config.hf_original.json"
    if not backup.exists():
        backup.write_text(config_path.read_text())
    config_path.write_text(json.dumps(data, indent=2) + "\n")
    print(f"[eval_act_hf] sanitized config.json; removed {removed}")
    inject_old_lerobot_stats(checkpoint_dir)
    disable_new_lerobot_processors(checkpoint_dir)


def _processor_state_path(checkpoint_dir: Path, name: str) -> Path:
    active = checkpoint_dir / name
    if active.exists():
        return active
    disabled = checkpoint_dir / f"{name}.disabled_for_local_eval"
    if disabled.exists():
        return disabled
    raise FileNotFoundError(active)


def inject_old_lerobot_stats(checkpoint_dir: Path) -> None:
    """Move new LeRobot processor stats into the older policy state layout."""
    from safetensors.torch import load_file, save_file

    model_path = checkpoint_dir / "model.safetensors"
    model = load_file(model_path)
    if "normalize_inputs.buffer_observation_state.mean" in model:
        return

    pre = load_file(_processor_state_path(
        checkpoint_dir,
        "policy_preprocessor_step_3_normalizer_processor.safetensors",
    ))
    post = load_file(_processor_state_path(
        checkpoint_dir,
        "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
    ))
    additions = {
        "normalize_inputs.buffer_observation_image.mean": pre["observation.image.mean"],
        "normalize_inputs.buffer_observation_image.std": pre["observation.image.std"],
        "normalize_inputs.buffer_observation_state.mean": pre["observation.state.mean"],
        "normalize_inputs.buffer_observation_state.std": pre["observation.state.std"],
        "normalize_inputs.buffer_observation_wrist_image.mean": pre["observation.wrist_image.mean"],
        "normalize_inputs.buffer_observation_wrist_image.std": pre["observation.wrist_image.std"],
        "normalize_targets.buffer_action.mean": pre["action.mean"],
        "normalize_targets.buffer_action.std": pre["action.std"],
        "unnormalize_outputs.buffer_action.mean": post["action.mean"],
        "unnormalize_outputs.buffer_action.std": post["action.std"],
    }
    backup = checkpoint_dir / "model.hf_original.safetensors"
    if not backup.exists():
        backup.write_bytes(model_path.read_bytes())
    save_file({**model, **additions}, model_path)
    print("[eval_act_hf] injected processor normalization stats into model.safetensors")


def disable_new_lerobot_processors(checkpoint_dir: Path) -> None:
    """Hide newer LeRobot processor files from the older local ACT server.

    The local ACT server can load the policy/config/model directly, matching
    the repo's older ACT checkpoints. If policy_preprocessor.json is present,
    it tries to import lerobot.policies.factory, which is only available in
    newer LeRobot installs and fails in this cluster env.
    """
    for name in (
        "policy_preprocessor.json",
        "policy_preprocessor_step_3_normalizer_processor.safetensors",
        "policy_postprocessor.json",
        "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
    ):
        path = checkpoint_dir / name
        if not path.exists():
            continue
        disabled = checkpoint_dir / f"{name}.disabled_for_local_eval"
        if not disabled.exists():
            path.rename(disabled)
        else:
            path.unlink()
        print(f"[eval_act_hf] disabled newer LeRobot processor file: {name}")


def eval_args_for(task: str) -> list[str]:
    eval_config = (
        "runs/act/eval/eval_v1_100_act_open_microwave_config.yml"
        if task == "open_microwave"
        else "runs/act/eval/eval_v1_100_act_config.yml"
    )
    if task == "place_bread_in_basket":
        return [
            "--config", eval_config,
            "--vla-config", "runs/act/eval/act_place_bread_gripper_vla_config.yml",
            "--no-vla-train-defaults",
            "--max-videos-per-dir", "3",
            "--act-gripper-closed-cmd", "0.82",
            "--interpolate-qpos-actions",
            "--qpos-gripper-max-delta", "0.08",
            "--qpos-gripper-close-hold-steps", "25",
            "--success-hold-steps", "1",
        ]
    if task == "deliver_to_human_easy":
        return [
            "--config", "runs/act/eval/eval_v1_100_act_train_defaults_config.yml",
            "--max-videos-per-dir", "3",
            "--act-gripper-closed-cmd", "0.0",
            "--interpolate-qpos-actions",
            "--qpos-gripper-max-delta", "0.08",
            "--qpos-gripper-close-hold-steps", "25",
            "--success-hold-steps", "1",
        ]
    return [
        "--config", eval_config,
        "--no-vla-train-defaults",
        "--max-videos-per-dir", "3",
        "--act-gripper-closed-cmd", "0.0",
        "--interpolate-qpos-actions",
        "--qpos-gripper-max-delta", "0.08",
        "--qpos-gripper-close-hold-steps", "25",
        "--success-hold-steps", "1",
    ]


def submit_eval_jobs(args: argparse.Namespace, checkpoint: Path, step: int) -> list[str]:
    eval_steps = int(args.max_steps or TASK_EVAL_STEPS.get(args.task, DEFAULT_EVAL_STEPS))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) if args.output_dir else (
        EVAL_ROOT / f"{args.task}_hf_step{step}_s{eval_steps}_{stamp}"
    )
    jobs = []
    shard_size = args.episodes_per_proc * args.num_proc
    for index in range(args.submit_jobs):
        start_seed = args.start_seed + index * shard_size
        port = args.port_base + index * args.num_proc
        cmd = [
            "sbatch",
            "--parsable",
            "-p", args.partition,
            "--constraint", args.constraint,
            EVAL_SCRIPT,
            args.task,
            str(checkpoint),
            str(port),
            str(eval_steps),
            str(args.episodes_per_proc),
            str(start_seed),
            str(out_dir),
            str(args.num_proc),
            *eval_args_for(args.task),
        ]
        proc = run(cmd, dry_run=args.dry_run)
        job_id = "DRYRUN" if proc is None else proc.stdout.strip()
        jobs.append(f"{job_id}:{args.task}:{start_seed}-{start_seed + shard_size - 1}")
    print(f"[eval_act_hf] checkpoint={checkpoint}")
    print(f"[eval_act_hf] output={out_dir}")
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default=HF_REPO_ID)
    parser.add_argument("--task", help="Task name in the HF repo.")
    parser.add_argument("--step", type=int, default=None, help="Checkpoint step. Default: latest.")
    parser.add_argument("--local-root", type=Path, default=HF_LOCAL_ROOT)
    parser.add_argument("--list", action="store_true", help="List available HF tasks and steps.")
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--submit-jobs", type=int, default=5)
    parser.add_argument("--partition", default="gpu-preempt")
    parser.add_argument("--constraint", default="1080_ti|2080_ti|titan_x|rtx_8000|a4000|l4")
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--episodes-per-proc", type=int, default=5)
    parser.add_argument("--num-proc", type=int, default=2)
    parser.add_argument("--port-base", type=int, default=9300)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    files = repo_files(args.repo_id)
    checkpoints = available_checkpoints(files)
    if args.list:
        for task, steps in checkpoints.items():
            print(f"{task}: {','.join(str(step) for step in steps)}")
        return
    if not args.task:
        raise SystemExit("--task is required unless --list is set")
    step = latest_step(files, args.task, args.step)
    checkpoint = download_checkpoint(args.repo_id, args.task, step, args.local_root)
    if args.download_only:
        print(checkpoint)
        return
    jobs = submit_eval_jobs(args, checkpoint, step)
    print("[eval_act_hf] submitted " + " ".join(jobs))


if __name__ == "__main__":
    main()
