# ACT Train/Eval Howto

This repo's ACT baseline is trained through LeRobot on the qpos-target VLA data.
Current standard setting:

- raw data root: `vla_data_qpos_target_20260524`
- control mode: `qpos_target_abs`
- input: state dim 8 plus head camera and wrist camera
- output: action dim 8, absolute qpos target
- frequency: 20 Hz via task-dependent `action_substeps`
- chunk size: `100`
- training steps: `25000`
- checkpoint save frequency: `5000`

## Train One Task

Use the qpos-target launch script:

```bash
sbatch -p gpu scripts/launch/train_act_qpos_target_20260524_one_task.sbatch <task>
```

Useful overrides:

```bash
ACT_TARGET_EPISODES=300 \
ACT_STEPS=25000 \
ACT_CHUNK_SIZE=100 \
ACT_RUN_ID=full300_c100_s25000_novae \
sbatch -p gpu scripts/launch/train_act_qpos_target_20260524_one_task.sbatch <task>
```

For a filtered/raw-data variant:

```bash
ACT_RAW_DATA_ROOT=/abs/path/to/filtered_raw_root \
ACT_TARGET_EPISODES=<N> \
ACT_RUN_ID=<run_tag> \
sbatch -p gpu scripts/launch/train_act_qpos_target_20260524_one_task.sbatch <task>
```

The script builds:

1. symlinked raw subset under `runs/act/qpos_target_20260524/raw/`
2. RLDS/OpenVLA dataset under `runs/act/qpos_target_20260524/data/<task>/<run_suffix>/openvla`
3. LeRobot dataset under `runs/act/qpos_target_20260524/data/<task>/<run_suffix>/lerobot`
4. ACT checkpoint under `runs/act/qpos_target_20260524/finetune/`

## Latest Checkpoint Pattern

Standard 300-episode ACT checkpoints:

```text
runs/act/qpos_target_20260524/finetune/act_<TASK>_20260524_qpos_target_abs_300ep_full300_c100_s25000_novae_v1/checkpoints/025000/pretrained_model
```

The special no-reclose categorize checkpoint:

```text
runs/act/qpos_target_20260524/finetune/act_categorize_cooperative_20260524_qpos_target_abs_296ep_full296_noreclose_c100_s25000_novae_v1/checkpoints/025000/pretrained_model
```

Resume/latest symlink, when present:

```text
runs/act/qpos_target_20260524/finetune/act_<TASK>_20260524_<RUN_SUFFIX>_v1/checkpoints/last/pretrained_model
```

## Eval One Task

Direct eval command:

```bash
scripts/run_vla_eval.sh \
  --model act \
  --checkpoint runs/act/qpos_target_20260524/finetune/act_<TASK>_20260524_qpos_target_abs_300ep_full300_c100_s25000_novae_v1/checkpoints/025000/pretrained_model \
  --task <task> \
  --episodes 10 \
  --max-steps <steps> \
  --video-dir runs/act/eval/<task>_manual_eval \
  -- \
  --config runs/act/eval/eval_v1_100_act_config.yml \
  --no-vla-train-defaults \
  --max-videos-per-dir 3 \
  --act-gripper-closed-cmd 0.0 \
  --interpolate-qpos-actions \
  --qpos-gripper-max-delta 0.08 \
  --qpos-gripper-close-hold-steps 25 \
  --success-hold-steps 1
```

For `open_microwave`, use its config:

```bash
--config runs/act/eval/eval_v1_100_act_open_microwave_config.yml
```

For `deliver_to_human_easy`, use train-default eval config:

```bash
--config runs/act/eval/eval_v1_100_act_train_defaults_config.yml
```

## Slurm Eval Shards

For full eval, use the fixed-output shard launcher:

```bash
sbatch -p gpu-preempt scripts/launch/eval_act_seed_ranges_multi_fixed_out.sbatch \
  <task> \
  <checkpoint> \
  9000 \
  <max_steps> \
  5 \
  <start_seed> \
  runs/act/eval/<task>_eval \
  2 \
  --config runs/act/eval/eval_v1_100_act_config.yml \
  --no-vla-train-defaults \
  --max-videos-per-dir 3 \
  --act-gripper-closed-cmd 0.0 \
  --interpolate-qpos-actions \
  --qpos-gripper-max-delta 0.08 \
  --qpos-gripper-close-hold-steps 25 \
  --success-hold-steps 1
```

This runs two processes in one Slurm job. Each process evaluates 5 seeds, so one
job covers 10 seeds.

## Automated Eval Maintainer

The eval queue maintainer is:

```bash
python scripts/maintain_act_eval_queue.py
```

It reads ACT checkpoints from:

```text
runs/act/qpos_target_20260524/finetune
```

It writes/updates:

```text
runs/act/eval/success_rate_latest.md
table.txt
```

Currently skipped in the maintainer:

- `take_from_human_easy`

## Current Latest Model Locations

All latest standard ACT models follow:

```text
runs/act/qpos_target_20260524/finetune/act_<TASK>_20260524_qpos_target_abs_300ep_full300_c100_s25000_novae_v1/checkpoints/025000/pretrained_model
```

Known completed tasks include the intent tasks, most base tasks, and the
assist/interrupt/neutral tasks under `vla_data_qpos_target_20260524/task_pool.txt`.
At the time this note was written, base `blocks_ranking_rgb` and
`blocks_ranking_size` did not have standard base-task ACT checkpoints, while
their assist/interrupt/neutral variants did.
