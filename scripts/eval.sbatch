#!/bin/bash
# Generic SLURM wrapper for VLA / imitation-policy evaluation through Genesis.
# Spawns the matching baseline server (per scripts/run_vla_eval.sh's dispatch
# table) and runs scripts/eval_vla.py from the MAWM env against it.
#
# Usage:
#   sbatch scripts/eval.sbatch <model> [<checkpoint>] [extra run_vla_eval.sh args...]
#
# Models (matches `run_vla_eval.sh --model ...`):
#   openvla_oft pi0 pi05 pi0_fast rdt
#   diffusion_policy dp3 act vqbet lerobot_diffusion smolvla
#
# Examples:
#   # default-checkpoint baselines (openvla_oft, rdt) — omit the checkpoint
#   sbatch scripts/eval.sbatch rdt --task pour_water --episodes 3 --max-steps 400
#
#   # baselines whose checkpoint is per-task / per-run (pi0/pi05/pi0_fast/DP/
#   # DP3/ACT/VQ-BeT/LeRobot-Diffusion/SmolVLA) — pass the checkpoint:
#   sbatch scripts/eval.sbatch act runs/act/finetune/full_57611830/checkpoints/last/pretrained_model \
#       --task pour_water
#   sbatch scripts/eval.sbatch diffusion_policy \
#       runs/diffusion_policy/finetune/pour_water/full_57612000/checkpoints/epoch=0200-val_loss=0.0045.ckpt \
#       --task pour_water --episodes 3
#
#   # forward eval_vla.py args after `--` (e.g. --output-json, --vla-config):
#   sbatch scripts/eval.sbatch act <ckpt> --task pour_water -- --output-json runs/act/eval/job_$.json
#
# Output:
#   runs/<model>/eval/job_<jobid>/  ← videos + optional results.json
#   runs/<model>/eval/server_logs/<model>_server.<jobid>.log  ← server log (persisted)

#SBATCH --job-name=eval_vla
#SBATCH --partition=gpu-preempt
#SBATCH --constraint="a100|h100|l40s|a40"
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3:00:00
#SBATCH --output=runs/_eval_slurm/slurm_%j.out
#SBATCH --error=runs/_eval_slurm/slurm_%j.err

set -eo pipefail

MODEL="${1:?usage: sbatch $0 <model> [<checkpoint>] [extra run_vla_eval.sh args...]}"
shift

CKPT_ARG=()
# Heuristic: if the next arg exists and is not a flag, treat it as the checkpoint.
if [[ $# -gt 0 && "$1" != --* && "$1" != -* ]]; then
  CKPT_ARG=(--checkpoint "$1")
  shift
fi

mkdir -p "runs/_eval_slurm"
cd /scratch4/workspace/qinhongzhou_umass_edu-simple/yz/code/genesis-hr-bench

echo "=== node: $(hostname)  job: $SLURM_JOB_ID ==="
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv | head -5
echo "model:    $MODEL"
echo "ckpt:     ${CKPT_ARG[*]:-(default)}"
echo "extra:    $*"
echo

# Per-baseline output layout: runs/<model>/eval/job_<jobid>_<model>/
OUT="runs/${MODEL}/eval/job_${SLURM_JOB_ID}"
mkdir -p "$OUT"

# Persistent server log under runs/<model>/eval/server_logs/ — survives the
# compute-node /tmp wipe, so silent server hangs are diagnosable post-hoc.
export SERVER_LOG_DIR="runs/${MODEL}/eval/server_logs"

exec scripts/run_vla_eval.sh \
    --model "$MODEL" "${CKPT_ARG[@]}" \
    --video-dir "$OUT" \
    "$@"
