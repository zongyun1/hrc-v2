#!/bin/bash
# scripts/_dist.sh — shared multi-node / multi-GPU launch helper.
#
# Sourced (never executed) by the per-baseline VLA finetune drivers:
#   finetune_openvla_oft.sh   family: torchrun
#   finetune_lerobot_v2.sh    family: accelerate   (smolvla / pi0 / pi05 / pi0_fast)
#   finetune_rdt.sh           family: accelerate   (+ DeepSpeed ZeRO-2)
#
# Contract — two knobs, read from the environment:
#   NNODES         number of nodes         (default 1)
#   GPUS_PER_NODE  GPUs / procs per node    (default 1)
#
# Rendezvous — set by scripts/finetune.sbatch; defaults suit a local run:
#   MASTER_ADDR                default 127.0.0.1
#   MASTER_PORT                default 29500
#   SLURM_NODEID / NODE_RANK   this node's 0-based rank (default 0)
#
# Usage from a driver:
#   source "$(dirname "${BASH_SOURCE[0]}")/_dist.sh"
#   dist_setup torchrun   "$TORCHRUN_BIN"
#   # or: dist_setup accelerate "$ACCELERATE_BIN"
#   exec "${DIST_LAUNCHER[@]}" <target> <target-args...>
#
# After dist_setup the following globals are populated:
#   DIST_LAUNCHER     bash array — the launch-command prefix
#   DIST_NPROC_TOTAL  NNODES * GPUS_PER_NODE
#   DIST_SUMMARY      one-line human description (for echo / logs)
#
# accelerate callers that need extra `accelerate launch` flags (e.g.
# --mixed_precision=bf16) append them to DIST_LAUNCHER before the target:
#   DIST_LAUNCHER+=(--mixed_precision=bf16)

_dist_die() { echo "[_dist] $*" >&2; exit 2; }

_dist_is_posint() { [[ "$1" =~ ^[1-9][0-9]*$ ]]; }

dist_setup() {
  local family="${1:?dist_setup: family (torchrun|accelerate) required}"
  local launcher="${2:?dist_setup: launcher binary path required}"

  local nnodes="${NNODES:-1}"
  local gpn="${GPUS_PER_NODE:-1}"
  _dist_is_posint "$nnodes" || _dist_die "NNODES must be a positive integer, got '$nnodes'"
  _dist_is_posint "$gpn"    || _dist_die "GPUS_PER_NODE must be a positive integer, got '$gpn'"

  local node_rank="${SLURM_NODEID:-${NODE_RANK:-0}}"
  local master_addr="${MASTER_ADDR:-127.0.0.1}"
  local master_port="${MASTER_PORT:-29500}"
  DIST_NPROC_TOTAL=$(( nnodes * gpn ))

  [[ -x "$launcher" ]] || _dist_die "launcher not executable: $launcher"

  case "$family" in
    torchrun)
      if [[ "$nnodes" -eq 1 ]]; then
        # --standalone picks its own free rendezvous port on this node.
        DIST_LAUNCHER=("$launcher" --standalone --nnodes=1 --nproc-per-node="$gpn")
      else
        DIST_LAUNCHER=("$launcher"
          --nnodes="$nnodes" --nproc-per-node="$gpn"
          --node-rank="$node_rank"
          --rdzv-backend=c10d
          --rdzv-id="${SLURM_JOB_ID:-genesis-hr-bench}"
          --rdzv-endpoint="${master_addr}:${master_port}")
      fi
      ;;
    accelerate)
      if [[ "$nnodes" -eq 1 && "$gpn" -eq 1 ]]; then
        DIST_LAUNCHER=("$launcher" launch --num_processes=1)
      elif [[ "$nnodes" -eq 1 ]]; then
        DIST_LAUNCHER=("$launcher" launch --multi_gpu --num_processes="$gpn")
      else
        DIST_LAUNCHER=("$launcher" launch --multi_gpu
          --num_machines="$nnodes"
          --num_processes="$DIST_NPROC_TOTAL"
          --machine_rank="$node_rank"
          --main_process_ip="$master_addr"
          --main_process_port="$master_port")
      fi
      ;;
    *)
      _dist_die "unknown family '$family' (expected torchrun|accelerate)"
      ;;
  esac

  DIST_SUMMARY="nodes=$nnodes gpus/node=$gpn total_procs=$DIST_NPROC_TOTAL node_rank=$node_rank"
  if [[ "$nnodes" -gt 1 ]]; then
    DIST_SUMMARY+=" rdzv=${master_addr}:${master_port}"
  fi
  # Explicit success — a trailing `[[ ]] &&` would otherwise leak exit 1 into
  # a `set -e` caller when the condition is false.
  return 0
}
