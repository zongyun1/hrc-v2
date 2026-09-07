#!/bin/bash
# Generic SLURM wrapper for unified VLA finetuning.
#
# Submitted by `scripts/finetune.sh --slurm`, which builds the `sbatch`
# command line (partition / constraint / account / nodes / gres / mem / time).
# This file therefore carries NO static `#SBATCH` cluster names — it is
# portable to any SLURM cluster via scripts/env.local.sh.
#
# Not meant to be sbatch'd by hand. Use:
#   scripts/finetune.sh --model <m> --slurm --nodes N --gpus-per-node M -- <args>
#
# Script args (supplied by finetune.sh):
#   --model M  --nodes N  --gpus-per-node G  -- <finetune passthrough...>

#SBATCH --job-name=finetune_vla
#SBATCH --output=runs/_finetune_slurm/slurm_%j.out
#SBATCH --error=runs/_finetune_slurm/slurm_%j.err

set -euo pipefail

MODEL=""
NNODES=1
GPUS_PER_NODE=1
PASS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)         MODEL="$2"; shift 2 ;;
    --nodes)         NNODES="$2"; shift 2 ;;
    --gpus-per-node) GPUS_PER_NODE="$2"; shift 2 ;;
    --)              shift; PASS+=("$@"); break ;;
    *) echo "finetune.sbatch: unexpected arg '$1'" >&2; exit 2 ;;
  esac
done
[[ -n "$MODEL" ]] || { echo "finetune.sbatch: --model required" >&2; exit 2; }

# Repo root via env.sh — no hardcoded paths. SLURM may copy this script to a
# spool dir, so resolve from SLURM_SUBMIT_DIR (where `sbatch` was invoked).
# shellcheck source=env.sh
source "${SLURM_SUBMIT_DIR:-$PWD}/scripts/env.sh"
cd "$REPO_ROOT"

# Distributed contract for scripts/_dist.sh. First node = rendezvous master.
export NNODES GPUS_PER_NODE
export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)"
export MASTER_PORT="$(( 20000 + (SLURM_JOB_ID % 20000) ))"

echo "=============================================================="
echo "=== finetune.sbatch  job=$SLURM_JOB_ID  model=$MODEL"
echo "=== nodes=$NNODES  gpus/node=$GPUS_PER_NODE  total=$((NNODES * GPUS_PER_NODE))"
echo "=== nodelist=$SLURM_JOB_NODELIST"
echo "=== rendezvous=$MASTER_ADDR:$MASTER_PORT"
echo "=============================================================="
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv | head -5
echo

if [[ "$NNODES" -gt 1 ]]; then
  # One task per node; each runs finetune.sh -> the per-baseline driver ->
  # torchrun/accelerate, which fan out GPUS_PER_NODE procs on that node.
  # srun sets SLURM_NODEID per task, which _dist.sh reads as the node rank.
  exec srun --nodes="$NNODES" --ntasks="$NNODES" --ntasks-per-node=1 \
       scripts/finetune.sh --model "$MODEL" "${PASS[@]}"
else
  exec scripts/finetune.sh --model "$MODEL" "${PASS[@]}"
fi
