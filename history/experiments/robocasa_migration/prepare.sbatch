#!/usr/bin/env bash
#SBATCH --job-name=robocasa-prepare
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/jiabenchen_umass/yz/job_logs/robocasa_prepare_%j.log
set -euo pipefail
PROJECT=/scratch/jiabenchen_umass/yz/hrc-v2
cd "$PROJECT"
export UV_CACHE_DIR=/scratch/jiabenchen_umass/yz/.cache/robocasa-uv
export MUJOCO_GL=disable
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
python3 robocasa_migration/fetch_sources.py
if [ ! -x .venv-robocasa/bin/python ]; then uv venv --python 3.11 .venv-robocasa; fi
if ! .venv-robocasa/bin/python -c 'import importlib.metadata as m; assert m.version("robocasa") == "1.0.1"; assert m.version("robosuite") == "1.5.2"; assert m.version("mujoco") == "3.3.1"' 2>/dev/null; then
  uv pip install --python .venv-robocasa/bin/python -e external/robosuite \
    numpy==2.2.5 mujoco==3.3.1 numba==0.61.2 scipy==1.15.3 \
    pygame pyyaml imageio imageio-ffmpeg h5py lxml gymnasium hidapi
  uv pip install --python .venv-robocasa/bin/python --no-deps -e external/robocasa
fi
.venv-robocasa/bin/python robocasa_migration/download_assets.py
.venv-robocasa/bin/python robocasa_migration/export_tasks.py
