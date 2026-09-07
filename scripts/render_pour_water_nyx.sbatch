#!/bin/bash
#SBATCH --job-name=nyx_pour
#SBATCH --partition=gpu-preempt
#SBATCH --constraint="a100|h100|l40s|a40|a16"
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=00:15:00
#SBATCH --output=job_logs/nyx_pour_%j.log
#SBATCH --error=job_logs/nyx_pour_%j.log

set -eo pipefail

cd /scratch4/workspace/qinhongzhou_umass_edu-simple/yz/code/genesis-hr-bench
mkdir -p job_logs outputs/nyx_pour_water_avatar

export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export GENESIS_BACKEND=gpu
export GENESIS_SOFTWARE_RENDER=0
export NUMBA_CACHE_DIR="${TMPDIR:-/tmp}/genesis_hr_bench_numba_cache"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/genesis_hr_bench_mpl_cache"
export XDG_CACHE_HOME="${TMPDIR:-/tmp}/genesis_hr_bench_xdg_cache"

PY=/scratch4/workspace/qinhongzhou_umass_edu-simple/yz/env/MAWM_nyx_clean/bin/python3

echo "=== nyx pour_water job ${SLURM_JOB_ID:-local} on $(hostname) ==="
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv | head -5
"$PY" -c "import torch, importlib.metadata as md; print('genesis-world', md.version('genesis-world')); print('gs-nyx', md.version('gs-nyx')); print('gs-nyx-plugin', md.version('gs-nyx-plugin')); print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"

srun "$PY" scripts/collect.py \
  --task pour_water \
  --config config/latest_genesis/pour_water_nyx.yml \
  --renderer nyx \
  --episodes 1 \
  --start-seed 0 \
  --save-dir outputs/nyx_pour_water_avatar
