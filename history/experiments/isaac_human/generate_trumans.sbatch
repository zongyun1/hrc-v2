#!/usr/bin/env bash
#SBATCH --job-name=trumans-kitchen
#SBATCH --partition=rtx-devel,rtx-batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:25:00
#SBATCH --output=/scratch/jiabenchen_umass/yz/job_logs/trumans_%j.log
set -euo pipefail
PROJECT=/scratch/jiabenchen_umass/yz/hrc-v2
RUNTIME=/scratch/jiabenchen_umass/moment-contact-memory/external/isaaclab300b2p1_ngc_container_v1/isaac-lab_3.0.0-beta2-post1_amd64.sif
cd "$PROJECT"
export PYTHONUNBUFFERED=1
OUT="$PROJECT/outputs/isaac_human/trumans_${SLURM_JOB_ID}"
mkdir -p "$OUT"
.venv-robocasa/bin/python isaac_human/trumans_occupancy.py \
  --source outputs/robocasa_migration/source/NavigateKitchen \
  --output external/trumans_utils/Data_blocks_motion_all/Scene/kitchen.npy \
  --transform "$OUT/kitchen_transform.json"
export APPTAINERENV_PYTHONPATH="$PROJECT/external/trumans-runtime/deps"
apptainer exec --nv --bind /scratch:/scratch "$RUNTIME" /isaac-sim/python.sh -m pip install \
  --target "$PROJECT/external/trumans-runtime/deps" --no-deps \
  hydra-core==1.3.2 omegaconf==2.3.0 antlr4-python3-runtime==4.9.3 \
  einops==0.7.0 smplx==0.1.28 vit-pytorch==1.4.4 dill==0.3.8 av==18.1.0
cd "$PROJECT/external/trumans_utils"
apptainer exec --nv --bind /scratch:/scratch "$RUNTIME" /isaac-sim/python.sh \
  "$PROJECT/coworker-genesis/tools/extract_smplx_skeleton.py" \
  --out "$OUT/smplx_skeleton_male.json" --skin_out "$OUT/smplx_skin_male.npz"
apptainer exec --nv --bind /scratch:/scratch "$RUNTIME" /isaac-sim/python.sh \
  "$PROJECT/coworker-genesis/tools/export_trumans_motion.py" \
  --transform "$OUT/kitchen_transform.json" --scene_name kitchen \
  --tx -.6 --z0 1.35 --z1 -1.85 --seed 0 --max_draws 6 \
  --out "$OUT/motion_kitchen_walk.npz"
test -s "$OUT/motion_kitchen_walk.npz"
