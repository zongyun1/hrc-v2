#!/bin/bash
# Unified fine-tuning launcher for genesis-hr-bench baselines.
#
# Two modes:
#
#   Local (default) — dispatch to the per-baseline driver in-process:
#     scripts/finetune.sh --model <name> [--sanity|--full] [--task TASK] ...
#     Multi-GPU on the current box:  --gpus-per-node N
#
#   SLURM (--slurm) — build an `sbatch` command and submit the generic
#     scripts/finetune.sbatch, with multi-node / multi-GPU as variables:
#     scripts/finetune.sh --model <name> --slurm \
#         --nodes N --gpus-per-node M [--time T --mem M --cpus C \
#         --partition P --qos Q --constraint C --account A --exclude NODELIST] \
#         [--sanity|--full] [...] [-- <framework passthrough>]
#
# --model and the --slurm/--nodes/--gpus-per-node/--time/--mem/--cpus/
# --partition/--qos/--constraint/--account flags are consumed here. Everything
# else (including a `--` and whatever follows) is forwarded to the per-baseline
# driver verbatim.
#
# Trainable VLA models (multi-node capable, --slurm supported):
#   openvla_oft, pi0, pi05, pi0_fast, smolvla, rdt
# Other trainable baselines (local dispatch only; use their own *.sbatch for
# SLURM): diffusion_policy, dp3, act, vqbet, lerobot_diffusion
#
# Examples:
#   scripts/finetune.sh --model smolvla --sanity
#   scripts/finetune.sh --model pi0 --gpus-per-node 2 --full
#   scripts/finetune.sh --model openvla_oft --slurm --nodes 2 --gpus-per-node 4 --full
#   scripts/finetune.sh --model dp3 --task pour_water --full -- training.num_epochs=2000

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL=""
USE_SLURM=0
NODES_FLAG=""
GPN_FLAG=""
TIME_FLAG=""
MEM_FLAG=""
CPUS_FLAG=""
PARTITION_FLAG=""
QOS_FLAG=""
CONSTRAINT_FLAG=""
ACCOUNT_FLAG=""
EXCLUDE_FLAG=""
DEPENDENCY_FLAG=""
REST=()

usage() { sed -n '2,33p' "${BASH_SOURCE[0]}"; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)         MODEL="$2"; shift 2 ;;
    --slurm)         USE_SLURM=1; shift ;;
    --nodes)         NODES_FLAG="$2"; shift 2 ;;
    --gpus-per-node) GPN_FLAG="$2"; shift 2 ;;
    --time)          TIME_FLAG="$2"; shift 2 ;;
    --mem)           MEM_FLAG="$2"; shift 2 ;;
    --cpus)          CPUS_FLAG="$2"; shift 2 ;;
    --partition)     PARTITION_FLAG="$2"; shift 2 ;;
    --qos)           QOS_FLAG="$2"; shift 2 ;;
    --constraint)    CONSTRAINT_FLAG="$2"; shift 2 ;;
    --account)       ACCOUNT_FLAG="$2"; shift 2 ;;
    --exclude)       EXCLUDE_FLAG="$2"; shift 2 ;;
    --dependency)    DEPENDENCY_FLAG="$2"; shift 2 ;;
    -h|--help)       usage; exit 0 ;;
    --)              shift; REST+=(-- "$@"); break ;;
    *)               REST+=("$1"); shift ;;
  esac
done

if [[ -z "$MODEL" ]]; then
  echo "--model required. trainable: openvla_oft|pi0|pi05|pi0_fast|smolvla|rdt|diffusion_policy|dp3|act|vqbet|lerobot_diffusion" >&2
  exit 2
fi

# --------------------------------------------------------------------------
# SLURM submission mode — build the sbatch command and submit finetune.sbatch.
# --------------------------------------------------------------------------
if [[ "$USE_SLURM" -eq 1 ]]; then
  # shellcheck source=env.sh
  source "$HERE/env.sh"

  case "$MODEL" in
    openvla_oft|pi0|pi05|pi0_fast|smolvla|rdt) ;;
    diffusion_policy|dp3|act|vqbet|lerobot_diffusion)
      echo "--slurm covers the VLA models only. For '$MODEL' submit scripts/finetune_${MODEL}.sbatch directly." >&2
      exit 2 ;;
    *) echo "unknown --model: $MODEL" >&2; exit 2 ;;
  esac

  NODES="${NODES_FLAG:-1}"
  GPN="${GPN_FLAG:-1}"
  # Per-model minimum RAM (floor). Whole-node 8-GPU runs blow past these via
  # the per-GPU scaling below; the floor only matters at GPN=1 on tiny clusters.
  case "$MODEL" in
    rdt)     MEM_MIN_GB=96 ;;
    smolvla) MEM_MIN_GB=64 ;;
    *)       MEM_MIN_GB=80 ;;
  esac
  # Per-GPU scaling so a whole-node 8-GPU submit auto-asks for ~whole-node
  # CPUs + RAM (otherwise --cpus-per-task=8 starves data-loaders and the
  # default per-CPU memory slice OOM-kills the trainer). See scripts/env.sh
  # for the FT_*_PER_GPU knobs and env.local.sh for cluster overrides.
  DEFAULT_CPUS=$(( GPN * FT_CPUS_PER_GPU ))
  DEFAULT_MEM_GB=$(( GPN * FT_MEM_GB_PER_GPU ))
  (( DEFAULT_MEM_GB < MEM_MIN_GB )) && DEFAULT_MEM_GB=$MEM_MIN_GB
  TIME="${TIME_FLAG:-12:00:00}"
  MEM="${MEM_FLAG:-${DEFAULT_MEM_GB}G}"
  CPUS="${CPUS_FLAG:-$DEFAULT_CPUS}"
  PARTITION="${PARTITION_FLAG:-$FT_PARTITION}"
  QOS="${QOS_FLAG:-$FT_QOS}"
  CONSTRAINT="${CONSTRAINT_FLAG:-$FT_CONSTRAINT}"
  ACCOUNT="${ACCOUNT_FLAG:-$FT_ACCOUNT}"

  OUT_DIR="$REPO_ROOT/runs/${MODEL}/finetune"
  mkdir -p "$OUT_DIR"

  SBATCH_ARGS=(
    --job-name="ft_${MODEL}"
    --nodes="$NODES"
    --ntasks-per-node=1
    --gres="${FT_GRES}:${GPN}"
    --cpus-per-task="$CPUS"
    --mem="$MEM"
    --time="$TIME"
    --output="$OUT_DIR/slurm_%j.out"
    --error="$OUT_DIR/slurm_%j.err"
  )
  [[ -n "$PARTITION"     ]] && SBATCH_ARGS+=(--partition="$PARTITION")
  [[ -n "$QOS"           ]] && SBATCH_ARGS+=(--qos="$QOS")
  [[ -n "$CONSTRAINT"    ]] && SBATCH_ARGS+=(--constraint="$CONSTRAINT")
  [[ -n "$ACCOUNT"       ]] && SBATCH_ARGS+=(--account="$ACCOUNT")
  [[ -n "$EXCLUDE_FLAG"     ]] && SBATCH_ARGS+=(--exclude="$EXCLUDE_FLAG")
  [[ -n "$DEPENDENCY_FLAG"  ]] && SBATCH_ARGS+=(--dependency="$DEPENDENCY_FLAG")

  echo "==============================================================="
  echo "[finetune] submitting SLURM job for model=$MODEL"
  echo "[finetune]   nodes=$NODES  gpus/node=$GPN  (total GPUs=$((NODES * GPN)))"
  echo "[finetune]   partition=${PARTITION:-(none)}  qos=${QOS:-(none)}  constraint=${CONSTRAINT:-(none)}  account=${ACCOUNT:-(none)}"
  echo "[finetune]   gres=${FT_GRES}:${GPN}  cpus=$CPUS  mem=$MEM  time=$TIME"
  echo "[finetune]   passthrough: ${REST[*]:-(none)}"
  echo "==============================================================="
  exec sbatch "${SBATCH_ARGS[@]}" \
       "$HERE/finetune.sbatch" \
       --model "$MODEL" --nodes "$NODES" --gpus-per-node "$GPN" \
       -- "${REST[@]}"
fi

# --------------------------------------------------------------------------
# Local dispatch — run the per-baseline driver in-process.
# NNODES / GPUS_PER_NODE: a --nodes/--gpus-per-node flag wins; otherwise keep
# whatever finetune.sbatch already exported; otherwise default to 1.
# --------------------------------------------------------------------------
export NNODES="${NODES_FLAG:-${NNODES:-1}}"
export GPUS_PER_NODE="${GPN_FLAG:-${GPUS_PER_NODE:-1}}"

case "$MODEL" in
  openvla_oft)       exec "$HERE/finetune_openvla_oft.sh"                        "${REST[@]}" ;;
  pi0|pi05|pi0_fast) exec "$HERE/finetune_lerobot_v2.sh" --policy-type "$MODEL"  "${REST[@]}" ;;
  smolvla)           exec "$HERE/finetune_lerobot_v2.sh" --policy-type smolvla   "${REST[@]}" ;;
  rdt)               exec "$HERE/finetune_rdt.sh"                                "${REST[@]}" ;;
  diffusion_policy)  exec "$HERE/finetune_diffusion_policy.sh"                   "${REST[@]}" ;;
  dp3)               exec "$HERE/finetune_dp3.sh"                                "${REST[@]}" ;;
  act)               exec "$HERE/finetune_act.sh"                                "${REST[@]}" ;;
  vqbet)             exec "$HERE/finetune_vqbet.sh"                               "${REST[@]}" ;;
  lerobot_diffusion) exec "$HERE/finetune_lerobot_diffusion.sh"                  "${REST[@]}" ;;
  *)
    echo "unknown --model: $MODEL" >&2
    echo "expected: openvla_oft|pi0|pi05|pi0_fast|smolvla|rdt|diffusion_policy|dp3|act|vqbet|lerobot_diffusion" >&2
    exit 2 ;;
esac
