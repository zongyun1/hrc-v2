#!/usr/bin/env bash
#SBATCH --job-name=robocasa-usdfix
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=/scratch/jiabenchen_umass/yz/job_logs/robocasa_usdfix_%j.log
set -euo pipefail
cd /scratch/jiabenchen_umass/yz/hrc-v2
export UV_CACHE_DIR=/scratch/jiabenchen_umass/yz/.cache/robocasa-uv
export PYTHONUNBUFFERED=1
if [ ! -x .venv-usd-robocasa/bin/python ]; then
    uv venv --python 3.11 .venv-usd-robocasa
    uv pip install --python .venv-usd-robocasa/bin/python usd-core
fi
.venv-usd-robocasa/bin/python robocasa_migration/repair_existing.py
