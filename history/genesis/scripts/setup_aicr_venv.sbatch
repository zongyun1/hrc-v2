#!/usr/bin/env bash
#SBATCH --job-name=setup_genesis_env
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=job_logs/setup_venv_%j.log
#SBATCH --error=job_logs/setup_venv_%j.log
set -euo pipefail

cd /scratch/jiabenchen_umass/yz/code
mkdir -p job_logs
PY=/home/jiabenchen_umass/.local/bin/python3.10
if [[ ! -x "$PY" ]]; then PY=python3.10; fi
if [[ ! -x /scratch/jiabenchen_umass/yz/.venv/bin/python ]]; then
  "$PY" -m venv /scratch/jiabenchen_umass/yz/.venv
fi
if ! /scratch/jiabenchen_umass/yz/.venv/bin/python -m pip --version >/dev/null 2>&1; then
  curl -fsSL https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip-aicr.py
  /scratch/jiabenchen_umass/yz/.venv/bin/python /tmp/get-pip-aicr.py
fi
/scratch/jiabenchen_umass/yz/.venv/bin/python -m pip install -U pip wheel
/scratch/jiabenchen_umass/yz/.venv/bin/python -m pip install -r requirements-genesis-latest.txt
echo "VENV_SETUP_DONE $(date)"
