#!/bin/bash
# Fine-tune RDT-1B on genesis-hr-bench (joint-space, Franka single-arm).
#
# RDT trains with HuggingFace accelerate + DeepSpeed ZeRO-2, so multi-node /
# multi-GPU rides the standard NNODES x GPUS_PER_NODE contract (scripts/_dist.sh).
#
# Usage:
#   scripts/finetune_rdt.sh --sanity
#   scripts/finetune_rdt.sh --full --exp-name run1
#   GPUS_PER_NODE=4 scripts/finetune_rdt.sh --full
#   NNODES=2 GPUS_PER_NODE=8 scripts/finetune_rdt.sh --full
#   scripts/finetune_rdt.sh --full -- --learning_rate 5e-5
#
# Prereqs:
#   * RDT venv (yz/env/rdt) with deepspeed + accelerate + torch.
#   * genesis-hr-bench RDT HDF5 dataset, built once with
#       scripts/convert_genesis_hr_bench_to_rdt_hdf5.py --out-dir runs/rdt/data/genesis_hr_bench
#   * T5-XXL + SigLIP encoders download from HF on first launch (~20 GB).
#
# Env:
#   NNODES, GPUS_PER_NODE   distributed shape (default 1x1; see scripts/_dist.sh)
#   RDT_HDF5_DIR            dataset root (default runs/rdt/data/genesis_hr_bench)
#   RDT_PY                  rdt venv python (resolved by scripts/env.sh)
#   WANDB_ENTITY/PROJECT    override the W&B target (full mode)
#
# Output: runs/rdt/finetune/<exp-name>/

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "$HERE/env.sh"
# shellcheck source=_dist.sh
source "$HERE/_dist.sh"

ENV_PY="$RDT_PY"
RDT_DIR="$REPO_ROOT/baseline/rdt"

MODE=""
EXP_NAME=""
EXTRA=()
usage() { sed -n '2,26p' "${BASH_SOURCE[0]}"; }
while [[ $# -gt 0 ]]; do
  case "$1" in
    --sanity)   MODE="sanity"; shift ;;
    --full)     MODE="full"; shift ;;
    --exp-name) EXP_NAME="$2"; shift 2 ;;
    --)         shift; EXTRA+=("$@"); break ;;
    -h|--help)  usage; exit 0 ;;
    *) echo "unknown arg: $1 (use -- to forward to main.py)" >&2; exit 2 ;;
  esac
done
case "$MODE" in
  sanity|full) ;;
  *) echo "required: --sanity or --full" >&2; exit 2 ;;
esac

[[ -x "$ENV_PY" ]]        || { echo "rdt venv python not found: $ENV_PY" >&2; exit 1; }
[[ -f "$RDT_DIR/main.py" ]] || { echo "RDT repo not found at $RDT_DIR" >&2; exit 1; }

if [[ -z "$EXP_NAME" ]]; then
  EXP_NAME="${MODE}_${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}"
fi

RDT_HDF5_DIR="${RDT_HDF5_DIR:-$REPO_ROOT/runs/rdt/data/genesis_hr_bench}"
export RDT_HDF5_DIR
if [[ -z "$(find -L "$RDT_HDF5_DIR" -name '*.hdf5' -print -quit 2>/dev/null)" ]]; then
  echo "[finetune_rdt] no *.hdf5 under RDT_HDF5_DIR=$RDT_HDF5_DIR" >&2
  echo "[finetune_rdt] build it: scripts/convert_genesis_hr_bench_to_rdt_hdf5.py --out-dir $RDT_HDF5_DIR" >&2
  exit 1
fi

OUTPUT_DIR="$REPO_ROOT/runs/rdt/finetune/$EXP_NAME"
mkdir -p "$OUTPUT_DIR"

if [[ "$MODE" == "sanity" ]]; then
  MAX_STEPS=200; BATCH=2; CKPT_PERIOD=200; SAMPLE_PERIOD=100000
  export WANDB_MODE=disabled
else
  MAX_STEPS=200000; BATCH=16; CKPT_PERIOD=1000; SAMPLE_PERIOD=500
  export WANDB_PROJECT="${WANDB_PROJECT:-rdt_genesis_hr_bench}"
  export WANDB_ENTITY="${WANDB_ENTITY:-multi-agent-world-model}"
fi

# Preempt-resume: RDT writes checkpoint-<step>/ under OUTPUT_DIR. If any exist,
# resume from the highest-numbered one.
RESUME_FLAG=()
LATEST_CKPT="$(ls -d "$OUTPUT_DIR"/checkpoint-* 2>/dev/null | sed 's/.*checkpoint-//' | sort -n | tail -1 || true)"
if [[ -n "$LATEST_CKPT" ]]; then
  RESUME_FLAG=(--resume_from_checkpoint="checkpoint-$LATEST_CKPT")
  echo "[finetune_rdt] resuming from checkpoint-$LATEST_CKPT"
fi

# DeepSpeed probes for a CUDA toolkit (CUDA_HOME) at `import deepspeed` time
# and aborts with MissingCUDAException if absent. Honor a pre-set CUDA_HOME;
# otherwise load the cluster's CUDA module ($CUDA_MODULE, from env.sh).
if [[ -z "${CUDA_HOME:-}" ]]; then
  if ! command -v module >/dev/null 2>&1; then
    for _i in /etc/profile.d/lmod.sh /etc/profile.d/modules.sh /usr/share/lmod/lmod/init/bash; do
      [[ -r "$_i" ]] && source "$_i" && break
    done
  fi
  command -v module >/dev/null 2>&1 && module load "${CUDA_MODULE:-cuda/12.1}" 2>/dev/null || true
fi
if [[ -z "${CUDA_HOME:-}" ]]; then
  echo "[finetune_rdt] CUDA_HOME unset and 'module load ${CUDA_MODULE:-cuda/12.1}' did not set it." >&2
  echo "[finetune_rdt] DeepSpeed needs a CUDA toolkit — set CUDA_HOME or CUDA_MODULE in scripts/env.local.sh." >&2
  exit 1
fi
export CUDA_HOME
echo "[finetune_rdt] CUDA_HOME=$CUDA_HOME"

# Apply baseline/rdt_overrides/ on top of the submodule. The rdt submodule
# tracks upstream thu-ml/RoboticsDiffusionTransformer (bimanual ALOHA template),
# which we don't control — so the genesis-hr-bench schema lives in the parent
# repo and gets layered in at run time. Idempotent: cp -f overwrites every run.
# See baseline/rdt_overrides/README.md.
OVERRIDES_DIR="$REPO_ROOT/baseline/rdt_overrides"
if [[ -d "$OVERRIDES_DIR" ]]; then
  echo "[finetune_rdt] applying $OVERRIDES_DIR -> $RDT_DIR"
  ( cd "$OVERRIDES_DIR" && find . -type f ! -name README.md -print0 ) | \
    while IFS= read -r -d '' rel; do
      src="$OVERRIDES_DIR/$rel"
      dst="$RDT_DIR/$rel"
      mkdir -p "$(dirname "$dst")"
      cp -f "$src" "$dst"
    done
fi

# main.py + configs/ resolve relative to the RDT repo root.
cd "$RDT_DIR"

# 1) Dataset statistics — idempotent (--skip_exist no-ops if already computed).
echo "[finetune_rdt] computing dataset stats (configs/dataset_stat.json) ..."
"$ENV_PY" -m data.compute_dataset_stat_hdf5 --skip_exist

# 2) Distributed launch prefix (accelerate; DeepSpeed ZeRO-2 via --deepspeed).
ACCELERATE_BIN="${ENV_PY%/python}/accelerate"
dist_setup accelerate "$ACCELERATE_BIN"

echo "==============================================================="
echo "[finetune_rdt] mode      = $MODE  (steps=$MAX_STEPS, batch=$BATCH/gpu)"
echo "[finetune_rdt] exp-name  = $EXP_NAME"
echo "[finetune_rdt] dataset   = $RDT_HDF5_DIR"
echo "[finetune_rdt] output    = $OUTPUT_DIR"
echo "[finetune_rdt] dist      = $DIST_SUMMARY"
echo "==============================================================="

exec "${DIST_LAUNCHER[@]}" \
    main.py \
    --deepspeed="./configs/zero2.json" \
    --pretrained_model_name_or_path="robotics-diffusion-transformer/rdt-1b" \
    --pretrained_text_encoder_name_or_path="google/t5-v1_1-xxl" \
    --pretrained_vision_encoder_name_or_path="google/siglip-so400m-patch14-384" \
    --output_dir="$OUTPUT_DIR" \
    --train_batch_size="$BATCH" \
    --sample_batch_size="$BATCH" \
    --max_train_steps="$MAX_STEPS" \
    --checkpointing_period="$CKPT_PERIOD" \
    --sample_period="$SAMPLE_PERIOD" \
    --checkpoints_total_limit=10 \
    --lr_scheduler="constant" \
    --learning_rate=1e-4 \
    --mixed_precision="bf16" \
    --dataloader_num_workers=8 \
    --image_aug \
    --dataset_type="finetune" \
    --state_noise_snr=40 \
    --load_from_hdf5 \
    --report_to=wandb \
    "${RESUME_FLAG[@]}" \
    "${EXTRA[@]}"
